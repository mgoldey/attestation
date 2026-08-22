"""Synthetic reader reactions, so the ranker has negatives to learn from.

The classifier needs both classes. `classifier_probs` returns None on a
single-class history, so a persona with only positive feedback ranks by
embedding similarity alone forever -- which is the state every real account in
this database is in. Nothing marks items *not* useful, because saying "no" to a
recommendation is work nobody does.

**Why not reuse `bootstrap_persona`.** It labels by a linear threshold on the
same embedding the classifier trains on, so a classifier fit on X to predict a
threshold of X recovers it perfectly. Its own docstring says so, and
`evaluate_user` excludes `source='bootstrap'` for exactly that reason. Labels
generated here must be independent of the embedding or they rebuild that
tautology in a new costume.

So a chat model reads the title and abstract and reacts *as the persona would*,
in prose, then commits to a verdict. That judgement runs on text, through a
different model, with no access to the vector -- which is what makes the
resulting rows worth training on.

These are still simulated readers, not real ones. Rows are written with
`source='simulated'` so they can be told apart from `ui` and `agent` clicks
forever, and so a future evaluation can exclude them the way `evaluate_user`
excludes bootstrap rows.
"""

import logging
import sqlite3
from collections import Counter

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

SOURCE = "simulated"


class Reaction(BaseModel):
    """A reader's response to one item.

    `reasoning` comes before `verdict` on purpose: a small model asked for a
    bare label picks one and rationalises afterwards, while one asked to react
    first commits to something it then has to be consistent with. It is also
    the part a human can audit -- a verdict with no reasoning is unreviewable.
    """

    reasoning: str = Field(min_length=1, max_length=400)
    verdict: bool
    confidence: int = Field(ge=1, le=5)


def _prompt(persona: str, interests: str, title: str, summary: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                f"You are {persona}, a researcher whose interests are: {interests}.\n"
                "React to one item from your feed as yourself. First say in one"
                " or two sentences what you actually think of it -- be specific"
                " about why it does or does not bear on your work. Then commit:"
                " verdict true if you would read it, false if you would not.\n"
                "confidence is 1-5 for how SURE you are of that verdict, not"
                " how much you liked the item. A paper far outside your field"
                " is a confident no: confidence 5, verdict false. Use a low"
                " confidence only when you genuinely cannot tell.\n"
                "Say no when it deserves a no. A researcher who finds"
                " everything useful has no taste, and a feed tuned on nothing"
                " but approval learns nothing."
            ),
        },
        {
            "role": "user",
            "content": f"Title: {title}\nAbstract: {summary[:800]}\n\nYour reaction?",
        },
    ]


def react_to_item(chat_fn, persona: str, interests: str, title: str, summary: str) -> Reaction:
    """One simulated reaction. Raises on an unusable response.

    No retry and no fallback verdict: an unparseable reply must not become a
    silent `useful=False`, which would poison training data with the model's
    formatting problems rather than its judgement.
    """
    out = chat_fn(_prompt(persona, interests, title, summary), Reaction.model_json_schema())
    return Reaction.model_validate(out)


def simulate_feedback(
    conn: sqlite3.Connection,
    chat_fn,
    user_name: str,
    items: list[sqlite3.Row],
    *,
    min_confidence: int = 3,
) -> dict:
    """Record a simulated reaction per item. Returns what was written and why.

    Unsure verdicts are dropped rather than recorded, since a coin-flip
    recorded as a label is noise the classifier then has to fit.

    The field is `confidence`, not `strength`, because the first version asked
    for "how strongly you feel" and the model read that as enthusiasm: a
    correct, well-argued rejection of a sourdough recipe came back at strength
    1 and was filtered out as indifference. Every negative was discarded and
    the classifier still could not train -- the exact failure this module
    exists to fix, reintroduced by the word used to ask.
    """
    from attestation.rank import get_user, record_click

    user = get_user(conn, user_name)
    if user is None:
        raise ValueError(f"unknown user: {user_name!r}")

    written = {"useful": 0, "not_useful": 0, "skipped_unsure": 0, "failed": 0}
    reactions = []
    for item in items:
        try:
            reaction = react_to_item(
                chat_fn, user_name, user["interests"], item["title"], item["summary"] or ""
            )
        except Exception:
            # One unusable reply costs one item, not the run: the aggregate is
            # the point, and `failed` reports how many were lost so a bad run
            # is visible rather than quietly smaller.
            log.warning("simulated reaction failed for item %s", item["id"], exc_info=True)
            written["failed"] += 1
            continue

        if reaction.confidence < min_confidence:
            written["skipped_unsure"] += 1
            continue

        record_click(conn, user["id"], item["id"], reaction.verdict, source=SOURCE)
        written["useful" if reaction.verdict else "not_useful"] += 1
        reactions.append(
            {
                "item_id": item["id"],
                "title": item["title"],
                "verdict": reaction.verdict,
                "confidence": reaction.confidence,
                "reasoning": reaction.reasoning,
            }
        )

    conn.commit()
    return {"counts": written, "reactions": reactions}


def classifier_would_train(conn: sqlite3.Connection, user_id: int) -> bool:
    """Whether this user's history has both classes, so the classifier fires.

    The one question simulation exists to turn from no into yes.
    """
    rows = conn.execute(
        "SELECT DISTINCT useful FROM clicks WHERE user_id = ?", (user_id,)
    ).fetchall()
    return len({r["useful"] for r in rows}) > 1


def source_skew_caveat(conn: sqlite3.Connection, user_id: int) -> str | None:
    """Warn when the two classes are separable by feed rather than by topic.

    Sampling round-robin across feeds is what finally produced negatives, but
    it has a cost: on matt's history it left 42 of 45 positives in arXiv cs.LG
    while negatives spread over nine other sources. A classifier fit on that
    can score a perfect AUC by learning "cs.LG means useful" -- which the
    embedding encodes trivially -- and learn nothing about the reader.

    An AUC of 1.0 usually means the task was easy, not that the model is good.
    Saying so with the result is cheaper than someone trusting it.
    """
    rows = conn.execute(
        "SELECT c.useful, i.feed_id FROM clicks c JOIN items i ON i.id = c.item_id"
        " WHERE c.user_id = ? AND c.source != 'bootstrap'",
        (user_id,),
    ).fetchall()
    positive = [r["feed_id"] for r in rows if r["useful"]]
    if len(positive) < 5:
        return None
    top = Counter(positive).most_common(1)[0][1]
    share = top / len(positive)
    if share < 0.8:
        return None
    return (
        f"{share:.0%} of this persona's positive feedback comes from one feed,"
        " so a classifier can separate the classes by source rather than by"
        " topic -- rate some items from that feed as not-useful before trusting"
        " an evaluation score"
    )

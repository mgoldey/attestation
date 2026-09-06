"""Personalized science feed tools: the `feed.*` namespace.

The reader-facing half of the server: ranked feed items, the feedback that
trains the ranker, the personas that own that training, feed subscriptions, and
the topic-grouped digest that composes ranking with the reading graph.

Two properties this namespace maintains deliberately. Ranking honesty: every
tool that returns an order also returns `ranking_quality`, because a feed built
from an untrained ranker looks exactly like one built from a good one. And no
LLM runs inside `digest` -- it returns structure, never prose, because the
caller is a model that can write the prose itself.
"""

from typing import Annotated, Literal

from pydantic import Field

from attestation.explain import explain as explain_item_fn
from attestation.llm import default_chat_fn
from attestation.mcp._shared import (
    MAX_LIST_LIMIT,
    ItemId,
    Limit,
    SinceDays,
    clamp_limit,
    get_embedder,
    ranked_items,
    ranking_quality,
    validate_window,
)
from attestation.mcp._tool import ToolError, tool, unknown_user_message
from attestation.mcp.personas import (
    _create_persona,
    _delete_persona,
    _propose_interests,
    _reset_feedback,
    _update_persona,
)
from attestation.rank import (
    RELEVANCE_ANCHOR,  # noqa: F401 -- re-exported: the measured policy is read from here
    RELEVANCE_FLOOR,  # noqa: F401
    apply_relevance_floor,
    record_click,
    vector_search,
)

# CLAUDE.md states the invariant: "an empty interests string embeds to
# nothing". Autocreate seeds a new reader from the corpus's top tags precisely
# to avoid it, and the explicit-create path had no guard -- `name=""` was
# refused while `interests=""` created a persona whose ranking has nothing to
# start from.
Interests = Annotated[
    str, Field(min_length=1, description="What this reader wants to follow, in their own words.")
]


def register(mcp) -> None:
    """Attach every feed.* tool to the server."""

    @mcp.tool(name="feed.list")
    # Four, matching search. Five REAL items (title p95 127, four tags each)
    # plus a full ranking caveat reached 2443 against the 2000 budget; the
    # fixture had been using 67-char titles, so nothing saw it. Clipping the
    # title to 120 recovers most of it and one fewer item recovers the rest --
    # and a result the caller cannot render is worth less than the result that
    # was dropped.
    def list_feed(user: str, limit: Limit = DEFAULT_LIST_LIMIT, since_days: SinceDays = 14) -> dict:
        """List this user's currently ranked, unread feed items (best first).

        Returns each item's `item_id`, title, url, source feed name, topic `tags`
        and `content_type` (paper/survey/announcement/release/blog/other; both
        empty until the tagging pass has processed it). No HTML and no article
        text -- call `feed.read` for one item's abstract. `limit` is capped at 16.

        **Show the reader the `url`, as a linked title.** A watched session
        rendered title/source/tags/`item_id` for five papers and omitted every
        link, and the reader replied "you didn't give links" -- they had five
        titles they could not open. `item_id` is an argument for your NEXT tool
        call (`feed.read`, `feed.rate`); it is not an address and means nothing
        to a human. If you print only one identifier per item, print the url.

        An unrecognised `user` is CREATED, not refused: a new reader is seeded
        from the corpus's most common topics, and the response says so. Do not
        use this tool to test whether a persona exists -- `feed.persona_status`
        (with no `user`) lists them without writing.

        `since_days` bounds how far back the feed reaches -- defaults to 14, so an
        empty result may mean "nothing published in the window" rather than
        "nothing relevant"; pass a larger value or `None` (unbounded) to tell them
        apart. Use `feed.search` instead for the whole archive including already-
        rated items.

        **Read `ranking_quality` before trusting the order.** With a single-class
        click history the click classifier never fires, and depending on click
        count the order may fall back partly or fully to embedding similarity --
        see the `caveat` field for which terms are actually contributing.
        """
        return _list_feed(user, limit, since_days)

    @mcp.tool(name="feed.rate")
    def record_feedback(user: str, item_id: ItemId, useful: bool) -> dict:
        """Record whether a feed item was useful for this user, retraining their ranking.

        This writes (or overwrites) a single click record for (user, item_id) --
        calling it again for the same item just replaces the previous verdict, so it
        is safe to call repeatedly. The next `feed.list` call for this user will
        reflect the updated ranking once enough mixed feedback has accumulated.
        """
        return _record_feedback(user, item_id, useful)

    @mcp.tool(name="feed.read")
    def read_item(user: str, item_id: ItemId) -> dict:
        """Read ONE item in full -- title, source, and its abstract.

        Use this when asked to summarise, explain or judge a specific paper.
        The list tools deliberately omit the abstract so that ten items stay
        renderable; this is where the text lives.

        Returns one item, not a list. Pass the `item_id` from a previous
        feed.list, feed.search or feed.digest result.
        """
        return _read_item(user, item_id)

    @mcp.tool(name="feed.explain")
    def explain_item(user: str, item_id: ItemId) -> dict:
        """Explain in one sentence why a specific feed item was ranked for this user.

        SLOW: this calls a local LLM (via the configured OpenAI-compatible backend)
        on first request for a given (user, item_id) pair and can take several
        seconds to ~1-2 minutes depending on the chat model and hardware; results
        are cached afterward and return instantly. Only call this for items the
        user is asking about specifically, not for every item in a list.
        """
        return _explain_item(user, item_id)

    @mcp.tool(name="feed.persona_create")
    def create_persona(name: str, interests: Interests) -> dict:
        """Create a reader persona from a name and an interests description.

        Ranking starts from the interests text and personalizes from the first
        feed.rate call. Use feed.persona_suggest_interests first if you want suggestions
        drawn from what is actually in the feed.
        """
        return _create_persona(name, interests)

    @mcp.tool(name="feed.persona_update")
    def update_persona(name: str, interests: Interests) -> dict:
        """Replace a persona's interests text; re-steers ranking immediately."""
        return _update_persona(name, interests)

    @mcp.tool(name="feed.persona_suggest_interests")
    def propose_interests(limit: Limit = 12) -> dict:
        """List the most common tags in the feed, to help write an interests string."""
        return _propose_interests(limit)

    @mcp.tool(name="feed.persona_status")
    def profile_status(user: str | None = None) -> dict:
        """Who the reader personas are, or how well-trained one of them is.

        Omit `user` to list every persona's name and interests -- use this to
        discover which `user` values are valid for `feed.list`, feed.rate, and
        feed.explain before calling them. Pass `user` for one persona's detail:
        click count, how much of the ranking is driven by behaviour versus the
        written interests, and the tags this reader liked and disliked most."""
        return _profile_status(user)

    @mcp.tool(name="feed.search")
    def search_feed(
        user: str,
        query: str,
        tag: str | None = None,
        content_type: Annotated[
            Literal["paper", "survey", "announcement", "release", "blog", "other"] | None,
            Field(description="Filter to one content type."),
        ] = None,
        # Four, not five. A search row carries `match` and `relevance` that a
        # list row does not, so five of them plus a full ranking caveat came to
        # 2041 against the 2000 budget -- and a payload an agent cannot hold is
        # one the tool should not send. Trimming the row was tried first
        # (null content_type and false already_rated are now omitted); this is
        # the remainder, and one fewer result is a smaller loss than a result
        # the caller cannot read.
        limit: Annotated[int, Field(ge=1, le=MAX_SEARCH_LIMIT)] = DEFAULT_LIST_LIMIT,
    ) -> dict:
        """Search the reader's own corpus of PAPERS AND ARTICLES by keyword.

        USE THIS for "find me papers/research/articles on X". The corpus is
        built from subscribed feeds -- on this installation, mostly arXiv
        (cs.LG and neighbours), plus Nature, Scientific Reports, Hugging Face
        and HN. Call `feed.sources` to see the actual list.

        This is a local semantic index, not a live query to arxiv.org, but for
        "what recent papers are there on X" it is usually the right answer and
        it needs no network. An agent that read the old one-line description
        ("Search items by keyword") could not tell what was IN the corpus, and
        so answered a request for arXiv papers on KV-cache optimization with
        "I do not have a tool that can search academic repositories" -- while
        this tool held 3,106 arXiv papers and returned four directly on-topic
        ones for that exact query.

        Unlike `feed.list` this searches the whole archive and includes items already
        rated, flagging each with already_rated.

        **Each row carries a `url` -- show it, as a linked title.** `item_id` is
        an argument for your next tool call, not an address; a reader given only
        titles and ids cannot open any of the results.

        **Read `ranking_quality` before trusting the order.** It reports whether the
        click classifier is actually active -- with a single-class click history it
        never fires, and depending on click count the order may fall back partly or
        fully to embedding similarity; see the `caveat` field for which terms are
        actually contributing.
        """
        return _search_feed(user, query, tag, content_type, limit)

    @mcp.tool(name="feed.persona_delete")
    def delete_persona(name: str, confirm: bool = False) -> dict:
        """Delete a persona AND all its feedback. Requires confirm=true. Irreversible."""
        return _delete_persona(name, confirm)

    @mcp.tool(name="feed.persona_reset")
    def reset_feedback(name: str, confirm: bool = False) -> dict:
        """Erase a persona's clicks but keep the persona. Requires confirm=true."""
        return _reset_feedback(name, confirm)

    @mcp.tool(name="feed.harvest_engagement")
    def harvest_engagement(user: str) -> dict:
        """Turn this user's past "why is this here?" questions into weak positive feedback.

        Cheap and idempotent -- no LLM call, no network. Reads what is already
        recorded and records nothing twice.

        Explicit feedback is a channel almost nobody uses, so a ranker waiting
        to be told what is useful waits forever. Asking why an item was ranked
        is engagement: weaker than a stated opinion, but real, and already
        logged. Each such item that was never rated becomes one weak positive
        under source='implicit', so a later evaluation can weight or exclude
        them.

        An item the user already rated is left alone -- an explicit "not
        useful" is never overwritten by the reader's own curiosity.

        Only positives are inferred. No behaviour reliably means "not useful",
        and inferring rejection from silence would poison the class the ranker
        is starving for. Use feed.rate for negatives.
        """
        return _harvest_engagement(user)

    @mcp.tool(name="feed.simulate_ratings")
    def simulate_feedback(user: str, limit: Limit = 10, confirm: bool = False) -> dict:
        """Generate simulated reader reactions for a persona, to train its ranking.

        WRITES CLICK ROWS -- needs confirm=true. SLOW: one local LLM call per
        item, several seconds each.

        The ranker's click classifier needs both useful AND not-useful feedback
        to fire at all; a persona that has only ever marked things useful ranks
        by embedding similarity alone forever. Real feedback is overwhelmingly
        positive because saying no to a recommendation is work nobody does, so
        this asks a local model to read each item and react AS the persona --
        in prose, then a verdict -- and records what it decides.

        Rows are written with source='simulated' and stay distinguishable from
        real `ui`/`agent` clicks permanently. These are a training aid, not
        evidence about the persona: each reaction carries the reasoning behind
        its verdict so a human can audit what the ranker was taught.

        Returns per-item verdicts with reasoning, plus counts. Items the model
        is unsure about are skipped rather than recorded as a coin flip.
        """
        return _simulate_feedback(user, limit, confirm)

    @mcp.tool(name="feed.digest")
    def digest(
        user: str,
        days: Annotated[int, Field(ge=0, le=3650)] = 7,
        per_topic: Annotated[int, Field(ge=1, le=20)] = 3,
        # The digest CONSIDERS more items than a list returns -- it groups
        # them, and MAX_DIGEST_ITEMS bounds what it renders. So it does not
        # share Limit, which is sized for how many rows fit in one response.
        limit: Annotated[int, Field(ge=1, le=60)] = DEFAULT_DIGEST_LIMIT,
    ) -> dict:
        """This user's unread feed, ranked and grouped by topic — the weekly review.

        Composes the ranked feed with the reading graph's topic clusters: each item
        joins the cluster its tags overlap most, so the result reads as "here is
        what is new, by subject" rather than a flat list. Items whose tags match no
        cluster are returned in `unclustered` rather than dropped — that bucket's
        size is a real signal, since most tags are used once and never form
        concepts.

        `days` bounds how far back the ranked feed reaches (echoed as
        `window_days`); `per_topic` caps how many items each topic shows while
        `n_total` reports how many it actually had, so truncation is visible.

        Returns structure, never prose: no LLM runs inside this tool. Per-item
        `explanation` is surfaced only when `feed.explain` already cached one.

        **Read `ranking_quality` before trusting the order.** It reports whether the
        click classifier is actually active -- with a single-class click history it
        never fires, and depending on click count the order may fall back partly or
        fully to embedding similarity; see the `caveat` field for which terms are
        actually contributing.
        """
        return _digest(user, days, per_topic, limit)


# How many tags to return per item. Four was the natural output of the tagging
# pass and roughly doubled a row's length; the first few carry the topic and
# the rest are refinements a reader does not need in a list. Mirrors
# rank.MAX_TAGS_SHOWN, which RankedItem.to_row uses for the same cap on a
# ranked item's wire row -- this one caps the digest topic listing below,
# a different call site over the same tag list.
MAX_TAGS_SHOWN = 3


# A title's share of a row. Real titles reach 223 chars (p95 127) against a
# fixture that used 67, so five real rows plus a caveat reached 2443 against a
# 2000 budget while every guard stayed green. Bounding the field means no
# fixture can be wrong about it again -- the lesson from the provenance caveat,
# applied to the other axis.
# The default number of items feed.list and feed.search return. Named so the
# response budget can be derived from it rather than asserted beside it.
DEFAULT_LIST_LIMIT = 4

# A search row carries already_rated, match and relevance that a list row does
# not -- ~60 chars more each -- so it cannot return as many. Measured on the
# live database: at the shared limit of 16 it emitted 7476 against a 7000
# ceiling while the fixture-driven census passed, because fixture rows are
# cheaper than real ones. This is the shared cap minus what the extra fields
# cost.
MAX_SEARCH_LIMIT = 13
# The digest groups by topic and is the biggest response the feed produces --
# 4296 chars as emitted against list's 1835. Named so its budget derives from
# it rather than being asserted beside it.
DEFAULT_DIGEST_LIMIT = 30

# Items the digest may RENDER, across topics and unclustered together. The
# digest emitted 6905 chars against feed.list's 1835 and had no bound at all.
#
# Bounding topics alone was measured and rejected: materials-scientist shows
# only 3 topics yet cost 5031, because 5 unclustered items sit outside the
# topic list -- so a topic cap left the one persona with a messy profile over
# budget, which is the persona most likely to be real.
#
# Topics are already sorted largest-first, so this drops the singleton tail
# first -- the 440-char groups that exist to say "one paper on biology" -- and
# n_total keeps every omission reportable.
MAX_DIGEST_ITEMS = 12

MAX_TITLE_CHARS = 90


def _clip_title(title: str | None) -> str:
    """A title trimmed to fit, with the cut made visible.

    Silent truncation would let an agent quote half a title as though it were
    the whole one; `feed.read` returns the full record for anything that needs
    it.
    """
    text = " ".join((title or "").split())
    return text if len(text) <= MAX_TITLE_CHARS else text[:MAX_TITLE_CHARS].rstrip() + "…"


# How much abstract one item carries. The list tools cap at rank.SUMMARY_CHARS
# (used inside RankedItem.to_row) because ten of them must fit together; a
# single item can afford more, and an abstract cut mid-sentence gets quoted
# as though it were the finding.
FULL_SUMMARY_CHARS = 2000


@tool(empty={"item": None}, needs_user=True, autocreate_user=True, label="read_item")
def _read_item(conn, user_row, item_id: ItemId) -> dict:
    """One item with its text, for an agent asked to summarise or judge it.

    Trimming `summary` from the list shape is what stopped a ten-item payload
    from truncating, and it left an agent with a title and a url and nothing
    to read. Watched live: asked to summarise a paper, the model correctly
    answered that it had no tool for the job and told the reader to open the
    link. The answer is not to put abstracts back in the list; it is one tool
    that returns one item.
    """
    row = conn.execute(
        "SELECT i.id, i.title, i.url, i.summary, i.published, f.title AS source"
        "  FROM items i LEFT JOIN feeds f ON f.id = i.feed_id"
        " WHERE i.id = ?",
        (item_id,),
    ).fetchone()
    if row is None:
        raise ToolError(f"unknown item_id: {item_id}")

    text = (row["summary"] or "").strip()
    truncated = len(text) > FULL_SUMMARY_CHARS
    if truncated:
        text = text[:FULL_SUMMARY_CHARS].rstrip() + "\u2026"

    tags = [
        r["tag"]
        for r in conn.execute(
            "SELECT tag FROM item_tags WHERE item_id = ? ORDER BY tag", (item_id,)
        )
    ]
    rated = conn.execute(
        "SELECT useful FROM clicks WHERE user_id = ? AND item_id = ?",
        (user_row["id"], item_id),
    ).fetchone()
    # Reading a paper in full is the strongest thing a reader does short of
    # rating it, and it required no gesture. INSERT OR IGNORE: re-reading is
    # not a second signal, and the primary key makes that idempotent.
    conn.execute(
        "INSERT OR IGNORE INTO engagement(user_id, item_id, kind) VALUES (?, ?, 'read')",
        (user_row["id"], item_id),
    )
    conn.commit()
    return {
        # A clipped echo, not a second copy. `message` is the envelope's
        # one-line status and the caller already has the full title in
        # item.title; repeating it verbatim cost ~220 chars on the corpus's
        # longest titles, on the one response deliberately allowed to be large.
        "message": _clip_title(row["title"]),
        "item": {
            "item_id": row["id"],
            # The FULL title here -- unclipped, unlike a list row. That
            # asymmetry is what makes clipping the list acceptable.
            "title": row["title"],
            "url": row["url"],
            "source": row["source"],
            "published": row["published"],
            "summary": text,
            "truncated": truncated,
            "tags": tags[:MAX_TAGS_SHOWN],
            "n_tags": len(tags),
            # So the agent knows a verdict already exists rather than asking
            # the reader to repeat one.
            "already_rated": rated is not None,
            "rated_useful": bool(rated["useful"]) if rated else None,
        },
    }


@tool(
    empty={"items": [], "ranking_quality": {}},
    needs_user=True,
    autocreate_user=True,
    label="list_feed",
)
def _list_feed(conn, user_row, limit: int = 4, since_days: SinceDays = 14) -> dict:
    """`since_days` defaults to rank_items' own 14-day window so list_feed's
    behavior is unchanged; digest passes its `days` through here.

    The default is 5, not 10. Ten items is a web page's worth; in a chat the
    payload ran past 2,900 characters and gemma4:e2b could not reproduce it --
    it truncated, apologised, re-rendered, truncated again. Five is ~380
    tokens, which a small model can quote and reason over, and a reader asking
    "what should I read" wants a handful rather than a page. Callers who want
    more can say so.
    """
    limit = clamp_limit(limit)
    since_days = validate_window(since_days)
    items = ranked_items(conn, user_row, limit + 1, since_days)
    more = len(items) > limit
    items = items[:limit]
    return {
        "message": (
            f"{len(items)} item(s), best first"
            + (f"; more available -- raise limit (max {MAX_LIST_LIMIT})" if more else "")
        ),
        "items": [it.to_row() for it in items],
        "ranking_quality": ranking_quality(conn, user_row["id"]),
    }


@tool(needs_user=True, label="record_feedback")
def _record_feedback(conn, user_row, item_id: ItemId, useful: bool) -> dict:
    item = conn.execute("SELECT 1 FROM items WHERE id = ?", (item_id,)).fetchone()
    if item is None:
        raise ToolError(f"unknown item_id: {item_id}")
    record_click(conn, user_row["id"], item_id, useful, source="agent")
    return {
        "message": f"recorded useful={useful} for item {item_id} (user {user_row['name']}); "
        "ranking will reflect this on the next list_feed call",
    }


@tool(empty={"explanation": None}, needs_user=True, label="explain_item")
def _explain_item(conn, user_row, item_id: ItemId) -> dict:
    item = conn.execute("SELECT 1 FROM items WHERE id = ?", (item_id,)).fetchone()
    if item is None:
        raise ToolError(f"unknown item_id: {item_id}")
    # Asking WHY an item ranked is engagement whether or not the model answers,
    # and it is the signal implicit.harvest was built on. Recorded before the
    # call, so a model that is down costs the explanation and not the evidence
    # that this reader cared about this item.
    conn.execute(
        "INSERT OR IGNORE INTO engagement(user_id, item_id, kind) VALUES (?, ?, 'explain')",
        (user_row["id"], item_id),
    )
    conn.commit()
    result = explain_item_fn(conn, user_row["id"], item_id, chat_fn=default_chat_fn)
    if result.reason == "unknown_user":
        # needs_user already resolves user_row above, so this path is not
        # reachable through the tool call itself -- but explain() names the
        # cause precisely now, and collapsing it back into the generic
        # "model unreachable" wording would misdirect a caller that reaches
        # this some other way (e.g. the persona was deleted between the user
        # lookup above and this call).
        raise ToolError(f"unknown user_id: {user_row['id']}")
    if result.text is None:
        # The other two reasons keep today's single wording -- a caller
        # cannot act differently on "the model is down" vs "it answered with
        # nothing", and splitting them was not asked for; only unknown_user
        # got a distinct message above.
        raise ToolError(
            "could not generate an explanation -- the local model is"
            " unreachable or returned nothing. Check `attest install --check`;"
            " the ranking itself needs no model and feed.list still works."
        )
    return {
        "explanation": result.text,
        # Says what this call just did. implicit.py is honest in its docstring
        # that "curiosity is not approval, and a reader may well have asked
        # precisely because the item looked wrong" -- and the surface that
        # records it said nothing, so a reader interrogating a SUSPICIOUS
        # recommendation was upvoting it silently. One line, on the one tool
        # whose side effect is invisible.
        "message": "recorded as engagement; feed.rate says whether it was actually useful",
    }


# Two genuinely different shapes, not one shape with optional fields: the
# no-user branch is the retired feed.personas tool's old {message, users}
# payload verbatim,
# and forcing the seven per-user training keys onto it (their old fixed
# `empty=`) meant `persona_status(user=None)` returned ten keys instead of
# three, four of them meaningless zeros/Nones for a call that never ran a
# GROUP BY. A callable `empty` keeps each branch's envelope honest.
def _profile_status_empty(kwargs: dict) -> dict:
    """The declared shape for whichever `_profile_status` branch this call
    will take, so a failure -- including "unknown user" -- carries exactly
    the keys that branch's success would have."""
    if kwargs.get("user") is None:
        return {"users": []}
    return {
        "user": None,
        "interests": None,
        "clicks": 0,
        "clicks_by_source": {},
        "blend_weight": 0.0,
        "top_liked": [],
        "top_disliked": [],
    }


@tool(empty=_profile_status_empty, label="profile_status")
def _profile_status(conn, user: str | None = None) -> dict:
    """Every persona (no `user`), or one persona's training detail.

    The two questions this answered as separate tools -- "which personas
    exist" and "how well-trained is this one" -- share one row of the `users`
    table, so they are one tool distinguished by arity: omitting `user` is the
    fast, no-training-computation discovery path (the retired feed.personas
    tool's payload, unchanged), and passing it runs the `GROUP BY` and
    `top_and_bottom_keys` a discovery call does not need. `needs_user` is not
    used here because that resolves-or-refuses before the body runs, and the
    no-user branch must stay reachable rather than always requiring one.
    """
    if user is None:
        rows = conn.execute("SELECT name, interests FROM users ORDER BY name").fetchall()
        return {
            "message": f"{len(rows)} user(s)",
            "users": [{"name": r["name"], "interests": r["interests"]} for r in rows],
        }

    from attestation.features import top_and_bottom_keys
    from attestation.rank import blend_weight, get_user

    user_row = get_user(conn, user)
    if user_row is None:
        raise ToolError(unknown_user_message(conn, user))

    n_clicks = conn.execute(
        "SELECT COUNT(*) c FROM clicks WHERE user_id = ?", (user_row["id"],)
    ).fetchone()["c"]
    top_liked, top_disliked = top_and_bottom_keys(conn, user_row["id"])
    by_source = {
        r["source"]: r["n"]
        for r in conn.execute(
            "SELECT source, COUNT(*) AS n FROM clicks WHERE user_id = ? GROUP BY source",
            (user_row["id"],),
        )
    }
    return {
        "user": user_row["name"],
        "interests": user_row["interests"],
        "clicks": n_clicks,
        "clicks_by_source": by_source,
        "blend_weight": round(blend_weight(n_clicks), 3),
        "top_liked": top_liked,
        "top_disliked": top_disliked,
        "message": (
            f"{n_clicks} click(s); ranking is {round(blend_weight(n_clicks) * 100)}% "
            "driven by observed behavior and the rest by the interests text"
        ),
    }


# How much a query's semantic match counts against the reader's profile.
# Search is a directed question, so relevance to the query dominates -- but a
# profile still breaks ties, which is why a persona's search differs from a
# stranger's. Ranking purely by profile was the old bug: the query barely
# participated and results were "highest-ranked items that contain the string".
QUERY_WEIGHT = 0.75

# The relevance floor and the sqlite-vec query moved to rank.py on 2026-09-05
# so the library's search (library.py, a domain module that may not import
# from mcp/) shares one policy with the feed's. The private names are kept as
# aliases: tests and this module's own _semantic_hits call them by these.
_vector_search = vector_search
_apply_relevance_floor = apply_relevance_floor


def _semantic_hits(conn, embedder, query: str, k: int) -> dict[int, float]:
    """item_id -> similarity, via the sqlite-vec index.

    Indexed with DOC_PROMPT and searched with QUERY_PROMPT: embed.py's prompts
    are asymmetric because the model was trained that way, and mixing them
    measurably degrades retrieval. `embed_query` existed for exactly this and
    had no caller until now.

    Vectors are L2-normalised by truncate_normalize, so sqlite-vec's L2
    distance d relates to cosine similarity as cos = 1 - d^2/2.
    """
    return _apply_relevance_floor(_vector_search(conn, embedder, query, k))


def _resolved_tag(conn, tag: str | None) -> str | None:
    """A caller's tag turned into one the corpus actually uses, or a refusal.

    The tag vocabulary is database content, so it cannot be a schema Literal --
    it needs a runtime check with a message, which is what the kg namespace
    already does. This had neither: `tag="large language model"` and
    `tag="LARGE-LANGUAGE-MODELS"` both returned `ok:true, "0 item(s) filtered"`
    against 1,072 matching items, and "0 items" reads as a fact about the
    archive rather than a spelling problem.

    canonical() is identity for anything with spaces or capitals; resolve_query
    handles exactly those, and is what every kg lookup uses. The membership
    check itself is kg.resolve_or_raise, shared with knowledge._path's
    resolve loop -- but over every STORED tag, not the graph's concepts:
    build_graph drops tags used fewer than MIN_TAG_USES times, so filtering
    on graph membership refused rare-but-real tags. A search for a concept
    mentioned once is a legitimate search, and this is a filter, not a graph
    query.
    """
    if not tag:
        return tag
    from attestation.kg import canonical, resolve_or_raise

    stored = {canonical(r["tag"]) for r in conn.execute("SELECT DISTINCT tag FROM item_tags")}
    try:
        return resolve_or_raise(tag, stored, kind="tag")
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


def _passes_filters(item, needle: str, similarity: dict, tag, content_type) -> bool:
    """Whether an item survives the query and the explicit filters.

    A query keeps an item if the semantic index reached it OR the words appear
    literally -- the two find different things, which is why both run.

    Tags compare CANONICAL to CANONICAL. kg_concepts hands an agent the graph's
    aliased names, and the documented workflow is to feed one straight back in
    here; comparing it against the raw stored spellings made that workflow
    under-return badly. On the live corpus `hugging-face` reached 23 of the 380
    items carrying that concept, because the other 357 are stored as
    `huggingface`, and `large-language-models` reached 226 of 1,072.
    """
    from attestation.kg import canonical

    if needle and item.item_id not in similarity and not _literal_match(item, needle):
        return False
    if tag and canonical(tag) not in {canonical(t) for t in (item.tags or [])}:
        return False
    return not (content_type and item.content_type != content_type)


def _literal_match(item, needle: str) -> bool:
    return needle in (item.title or "").lower() or needle in (item.summary or "").lower()


def _score_matches(kept: list, ranked: list, needle: str, similarity: dict) -> list:
    """Blend query relevance with the reader's profile, best first.

    The profile is a tie-breaker, not the ranking: search is a directed
    question. Ordering by profile and filtering afterwards was the old bug --
    results were the highest-ranked items that happened to contain the string,
    so the query barely participated.
    """
    # profile_similarity, not a rank. The rank version was 1 - position/n over
    # whatever set was ranked, so bounding search's candidates renumbered every
    # item -- archive position 5227 became position 6, and a tie-breaker that
    # scored 0.003 scored 0.999 instead: 7 of 8 live queries changed results.
    # The cosine is absolute, which is what makes bounding the set safe.
    scored = []
    for item in kept:
        literal = bool(needle) and _literal_match(item, needle)
        sim = similarity.get(item.item_id)
        relevance = sim if sim is not None else 0.0
        if literal:
            # A boost, not a floor. Flooring every literal hit at one value
            # made 711 items matching "llm" tie at the same score, so profile
            # rank silently decided the order and the query stopped
            # discriminating. A title match outweighs a body match, since a
            # body can mention a term in passing.
            relevance += 0.35 if needle in (item.title or "").lower() else 0.15
        profile_score = max(0.0, item.profile_similarity)
        combined = QUERY_WEIGHT * min(relevance, 1.0) + (1.0 - QUERY_WEIGHT) * profile_score
        how = "both" if literal and sim is not None else ("literal" if literal else "semantic")
        scored.append((combined, item, how))
    scored.sort(key=lambda t: (-t[0], t[1].item_id))
    return scored


@tool(
    empty={"items": [], "ranking_quality": {}},
    needs_user=True,
    autocreate_user=True,
    label="search_feed",
)
def _search_feed(
    conn,
    user_row,
    query: str,
    tag: str | None = None,
    content_type: str | None = None,
    limit: int = 4,
) -> dict:
    from attestation.rank import SEARCH_CANDIDATES, literal_candidates, rank_items

    limit = clamp_limit(limit)
    tag = _resolved_tag(conn, tag)
    clicked = {
        r["item_id"]
        for r in conn.execute("SELECT item_id FROM clicks WHERE user_id = ?", (user_row["id"],))
    }

    # An empty query is a filter, not a search: keep profile order and let the
    # tag/content_type filters do the work.
    similarity: dict[int, float] = {}
    only_ids: list[int] | None = None
    if query.strip():
        # sqlite-vec answers the query FIRST and only its hits get ranked; this
        # used to run the other way round, ranking the archive and filtering
        # afterwards. See rank.SEARCH_CANDIDATES for the measurement. Over-fetch
        # so tag/content_type post-filtering still has candidates, and add the
        # literal hits, which need not be semantic ones.
        similarity = _semantic_hits(
            conn, get_embedder(), query, k=max(limit * 10, SEARCH_CANDIDATES)
        )
        only_ids = sorted(set(similarity) | literal_candidates(conn, query, SEARCH_CANDIDATES))

    ranked = rank_items(
        conn,
        get_embedder(),
        user_row["id"],
        since_days=None,
        exclude_clicked=False,
        only_ids=only_ids,
    )

    needle = query.lower().strip()
    kept = [item for item in ranked if _passes_filters(item, needle, similarity, tag, content_type)]
    scored = _score_matches(kept, ranked, needle, similarity)
    matches = [
        {
            **item.to_row(),
            # Always present, even when false. Making it conditional saved ~24
            # chars a row and broke the contract: a caller reading
            # item["already_rated"] got a KeyError instead of False, which is a
            # worse failure than the bytes were worth. The savings came from
            # dropping null content_type and one default result instead.
            "already_rated": item.item_id in clicked,
            # How this item was found, so a caller can tell relevance from
            # noise when a result looks surprising. Two decimals: the third
            # never changed a decision and cost a character per row.
            "match": how if needle else "filter",
            "relevance": round(score, 2),
        }
        for score, item, how in scored[:limit]
    ]
    mode = "filtered" if not needle else "searched"
    return {
        "message": f"{len(matches)} item(s) {mode}, best first",
        "items": matches,
        "ranking_quality": ranking_quality(conn, user_row["id"]),
    }


def _digest(
    user: str, days: int = 7, per_topic: int = 3, limit: int = DEFAULT_DIGEST_LIMIT
) -> dict:
    """`window_days` echoes the window that was ASKED for, on failure as well as
    success -- a caller that got nothing back still needs to know which window
    came up empty. `empty=` on `_digest_body` is a plain dict, so it cannot
    carry this per-call value; a callable `empty` (see `_tool.tool`'s
    docstring, added for `feed.persona_status`) is not used here on purpose --
    this value depends on the ARGUMENT alone, not on which of two differently-
    shaped branches ran, so patching it in after the envelope is simpler than
    threading it through a lambda. `_digest_body`'s own `empty=` still needs
    `window_days: 0` as a placeholder so the key exists to overwrite.
    """
    out = _digest_body(user, days, per_topic, limit)
    out["window_days"] = out.get("window_days") or days
    return out


def _best_cluster(tags: set, members: list) -> tuple[str | None, int]:
    """The cluster sharing the most tags with an item.

    Ties break on label, lexically smallest first, so repeated calls agree --
    the digest is read by a model and an order that shuffles between identical
    calls reads as new information.
    """
    best_label, best_n = None, 0
    for label, concepts in members:
        overlap = len(tags & concepts)
        if overlap > best_n or (overlap == best_n and overlap and label < (best_label or "~")):
            best_label, best_n = label, overlap
    return best_label, best_n


def _cluster(items: list, members: list, cached: dict) -> tuple[dict[str, list], list]:
    """Group items by their strongest tag overlap, attaching cached explanations.

    Items matching no cluster are returned separately rather than dropped: a
    digest that silently omits them would misreport how much was read.
    """
    grouped: dict[str, list] = {}
    unclustered: list = []
    for item in items:
        label, overlap = _best_cluster(set(item.get("tags") or []), members)
        enriched = dict(item)
        if item["item_id"] in cached:
            enriched["explanation"] = cached[item["item_id"]]
        if overlap and label is not None:
            grouped.setdefault(label, []).append(enriched)
        else:
            unclustered.append(enriched)
    return grouped, unclustered


def _allocate_digest_budget(
    grouped: dict[str, list], unclustered: list, per_topic: int, budget: int
) -> dict:
    """Spend `budget` items across `grouped` topics (largest first, capped at
    `per_topic` each) then on leftover `unclustered` items, and report exactly
    what was shipped. Pure so the undercount this function's own history
    recorded -- every live persona's digest said "16 item(s)" while shipping
    6 to 11 -- has a DB-free regression test: `shipped` must equal what is
    actually in the returned topics plus `shown_unclustered`, not a count
    that forgets truncation inside a shown topic.
    """
    topics = []
    shipped_in_topics = 0
    for label, group in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if budget <= 0:
            break
        shown = group[: max(1, min(int(per_topic), budget))]
        budget -= len(shown)
        shipped_in_topics += len(shown)
        topics.append({"label": label, "n_total": len(group), "items": shown})

    # Unclustered draws from the SAME budget, after topics: they are the
    # leftovers, and a digest that spent its whole allowance on them would bury
    # the grouping that is the point of the tool.
    shown_unclustered = unclustered[: max(0, budget)]
    # Count ITEMS not shown, from all three causes. This counted omitted groups
    # and leftover unclustered items but never items cut inside a SHOWN topic
    # by per_topic -- so every live persona read "16 item(s)" while shipping
    # 6 to 11. The existing guard asserts n_total per topic and never the
    # message, which is how it passed.
    shipped = shipped_in_topics + len(shown_unclustered)
    return {
        "topics": topics,
        "shown_unclustered": shown_unclustered,
        "shipped": shipped,
    }


@tool(
    empty={"topics": [], "unclustered": [], "ranking_quality": {}, "window_days": 0},
    needs_user=True,
    autocreate_user=True,
    label="digest",
)
def _digest_body(conn, user_row, days: int = 7, per_topic: int = 3, limit: int = 30) -> dict:
    from attestation import kg

    # digest alone used to answer an unknown user with `unknown user 'x'`,
    # where the other nine name the valid personas. That was an inconsistency,
    # not a choice: the caller is a model, and the list is the thing it can act
    # on. It now goes through needs_user like the rest.
    row = user_row
    # the caller's connection is reused deliberately: digest must not open a
    # second one to rank the same feed
    items_ranked = ranked_items(conn, row, min(limit, MAX_LIST_LIMIT), days)
    items = [it.to_row() for it in items_ranked]
    if not items:
        raise ToolError("no unread items to digest")

    communities = kg.communities(conn, min_size=3)
    members = [(c["label"], set(c["members"])) for c in communities]
    cached = {
        r["item_id"]: r["text"]
        for r in conn.execute(
            "SELECT item_id, text FROM explanations WHERE user_id = ?", (row["id"],)
        )
    }

    grouped, unclustered = _cluster(items, members, cached)

    out = _allocate_digest_budget(grouped, unclustered, per_topic, MAX_DIGEST_ITEMS)
    dropped = len(items) - out["shipped"]
    note = f"; showing {out['shipped']} -- {dropped} not shown" if dropped else ""
    return {
        "message": f"{len(items)} item(s) in {len(grouped)} topic(s){note}",
        "topics": out["topics"],
        "unclustered": out["shown_unclustered"],
        "ranking_quality": ranking_quality(conn, row["id"]),
        "window_days": days,
    }


@tool(
    empty={"counts": {}, "reactions": []},
    needs_user=True,
    label="simulate_feedback",
)
def _simulate_feedback(conn, user_row, limit: int = 10, confirm: bool = False) -> dict:
    from attestation.llm import default_chat_fn
    from attestation.simulate import simulate_feedback as run_simulation
    from attestation.simulate import source_skew_caveat

    if not confirm:
        raise ToolError(
            "refusing to simulate without confirm=true. This writes click rows"
            f" for {user_row['name']!r} (marked source='simulated', so they stay"
            " distinguishable from real feedback) and calls a local LLM once per"
            " item."
        )
    limit = clamp_limit(limit)
    # Sample round-robin across FEEDS, not down the ranking.
    #
    # Ranked candidates are the wrong pool twice over. They are ordered by
    # relevance to this very persona, so the top of the list asks the model to
    # reject what the ranker just selected for relevance; and the archive is
    # lopsided -- 1,744 of any 2,000 candidates here are arXiv cs.LG, whose
    # subject matter IS matt's stated interest, so walking further down samples
    # more of the same. Measured: 1 negative in 12, and that one's own
    # reasoning called the paper "highly relevant".
    #
    # Feed is the axis that actually varies. Taking the newest few from each in
    # turn puts chemistry, neuroscience and general news in front of a reader
    # of ML papers, and the same model rejected 7 of 8 such items at full
    # confidence. A simulated reader can only reject what it is shown.
    rows = conn.execute(
        "SELECT id, title, summary FROM ("
        "  SELECT i.id, i.title, i.summary,"
        "         ROW_NUMBER() OVER (PARTITION BY i.feed_id ORDER BY i.published DESC) rn"
        "  FROM items i"
        "  WHERE i.summary IS NOT NULL AND i.summary != ''"
        "    AND i.id NOT IN (SELECT item_id FROM clicks WHERE user_id = ?)"
        ") WHERE rn <= ? ORDER BY rn, id LIMIT ?",
        (user_row["id"], max(limit // 3, 2), limit),
    ).fetchall()
    if not rows:
        raise ToolError("no unrated items to react to -- run an ingest first")
    out = run_simulation(conn, default_chat_fn, user_row["name"], rows)
    counts = out["counts"]
    caveat = source_skew_caveat(conn, user_row["id"])
    return {
        "message": (
            f"{counts['useful']} useful, {counts['not_useful']} not-useful"
            f" ({counts['skipped_unsure']} unsure, {counts['failed']} failed)"
            + (f". {caveat}" if caveat else "")
        ),
        "counts": counts,
        "reactions": out["reactions"],
        "caveat": caveat,
    }


@tool(
    empty={"candidates": 0, "recorded": 0, "skipped_already_rated": 0},
    needs_user=True,
    label="harvest_engagement",
)
def _harvest_engagement(conn, user_row) -> dict:
    from attestation.implicit import harvest

    out = harvest(conn, user_row["name"])
    return {
        "message": (
            f"recorded {out['recorded']} weak positive(s) from engagement"
            f" ({out['skipped_already_rated']} already rated)"
        ),
        **out,
    }

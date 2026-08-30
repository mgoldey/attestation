"""LangGraph explain agent: click history -> profile -> one-sentence 'why ranked here'.

Hermes is the orchestrator; the chat model is a swappable OpenAI-compatible backend
(see src/attestation/llm.py).

Reliability contract: lazy, cached, degrades to None. Ranking never waits on this.
"""

import logging
import sqlite3
from typing import Literal, NamedTuple

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class ExplainResult(NamedTuple):
    """`explain()`'s answer: the text, or which of three unrelated causes
    produced none.

    A bare `None` return collapsed "this user does not exist" (the caller's
    argument is wrong), "the model is unreachable" (retry later), and "the
    model answered but produced nothing usable" (also worth a retry, but not
    the same failure) into one value -- so `_explain_item` could only ever
    raise one generic `ToolError` for all three, and its own docstring had to
    spell out the ambiguity a return type should have carried. `reason="ok"`
    is set only when `text` is not `None`, so a caller can also just check
    truthiness of `.text` the way the old return worked.
    """

    text: str | None
    reason: Literal["ok", "unknown_user", "model_unreachable", "no_answer"]


class Explanation(BaseModel):
    """The chat model's structured reply: one short sentence, nothing else."""

    text: str = Field(min_length=1, max_length=300)


class ExplainState(BaseModel):
    """LangGraph state threaded profile -> explanation across the two nodes."""

    user_id: int
    item_id: int
    profile: str = ""
    explanation: str | None = None
    # Set by generate_explanation to "model_unreachable" or "no_answer" when
    # `explanation` stays None, so explain() can build the right ExplainResult
    # without re-deriving the distinction from a bare None.
    explanation_reason: str = "no_answer"


def explanation_messages(profile: str, title: str, summary: str) -> list[dict]:
    """The ONE renderer of the explanation prompt.

    `generate_explanation` and `evals/explanation_eval.py` both call this, so
    a score is always a score of the prompt `feed.explain` actually sends.
    """
    return [
        {
            "role": "system",
            "content": (
                # Measured against gemma4:e2b over four items, one of them
                # deliberately irrelevant. The previous wording ("You
                # explain feed rankings. One sentence, second person,
                # grounded ONLY in the reader profile given. No hedging.")
                # produced 26-word sentences at 2.0s that opened "You will
                # find that..." and manufactured a connection for a paper
                # about termite feed additives. This runs at 1.5s and 9
                # words, and the refusal clause is what makes it honest --
                # without it the model still claimed the termite paper
                # shared "scientific evaluation methodology".
                "Name the one topic this item shares with the reader's"
                " interests. Under 15 words. Address them as 'you'."
                " No preamble."
                " If it shares nothing, say 'Outside your stated"
                " interests.' and stop."
            ),
        },
        {
            "role": "user",
            "content": (f"Reader's interests: {profile}\nItem: {title}\n{summary[:400]}"),
        },
    ]


def profile_synthesis_messages(titles: list[str]) -> list[dict]:
    """The ONE renderer of the profile-synthesis fallback prompt.

    `synthesize_profile` calls this when a persona has clicks but no stored
    interests text -- see its docstring for why that stays the fallback
    rather than the default. Giving it a renderer is what makes it scoreable
    the way the other three prompts already are (no corpus or optimizer is
    added by this change; the renderer only makes one possible).
    """
    return [
        {"role": "system", "content": "Summarize this reader in one sentence."},
        {"role": "user", "content": "Recently useful titles:\n- " + "\n- ".join(titles)},
    ]


def _build_graph(conn: sqlite3.Connection, chat_fn):
    def synthesize_profile(state: ExplainState) -> dict:
        """The `profile` node: prefer stored interests text, synthesize only
        when a persona has clicks but no interests -- see the module-level
        comment below for why synthesis is the fallback, not the default."""
        titles = [
            r["title"]
            for r in conn.execute(
                "SELECT i.title FROM clicks c JOIN items i ON i.id = c.item_id"
                " WHERE c.user_id = ? AND c.useful = 1"
                " ORDER BY c.clicked_at DESC, c.id DESC LIMIT 20",
                (state.user_id,),
            )
        ]
        # A deleted persona (or a stale session holding its id) leaves no row.
        # Fail with a named error rather than subscripting None -- explain()
        # catches it and degrades to no explanation.
        row = conn.execute("SELECT interests FROM users WHERE id = ?", (state.user_id,)).fetchone()
        if row is None:
            raise ValueError(f"unknown user_id: {state.user_id}")
        interests = row["interests"]
        if not titles:
            return {"profile": interests}
        if interests and interests.strip():
            # The stored interests text beats a synthesized summary, measured
            # against gemma4:e2b: synthesis returns meta-description ("This
            # list of recently useful titles covers a diverse range of...")
            # and the explanations built on it were vaguer ("Large Language
            # Models and agent systems" against "LLM instruction following
            # and reasoning"). It also claimed a paper on termite feed
            # additives matched "advanced topics like AI and machine
            # learning", where the interests string correctly refused.
            #
            # Worse answers for an extra 2.1 seconds and a second chance to
            # fail. Synthesis stays as the fallback for a persona that has
            # clicks but no interests text.
            return {"profile": interests}
        try:
            out = chat_fn(profile_synthesis_messages(titles), Explanation.model_json_schema())
            return {"profile": Explanation.model_validate(out).text}
        except Exception:
            log.debug("explain attempt failed", exc_info=True)
            log.warning("profile synthesis failed; using interests text")
            return {"profile": interests}

    def generate_explanation(state: ExplainState) -> dict:
        """The `explain` node: one retry per spec, `None` rather than a
        traceback if both attempts fail to parse -- ranking never waits on
        this, so a bad reply must degrade, not raise.

        Also records the LAST attempt's failure kind as `explanation_reason`,
        so `explain()` can tell "the model was unreachable" (an
        `OSError`/`ConnectionError` -- retry later) from "the model answered
        but the reply did not parse" (a validation error -- also worth a
        retry, but a different failure) without re-deriving the distinction
        from a bare `None`.
        """
        item = conn.execute(
            "SELECT title, summary FROM items WHERE id = ?", (state.item_id,)
        ).fetchone()
        messages = explanation_messages(state.profile, item["title"], item["summary"])
        reason = "no_answer"
        for _ in range(2):  # one retry per spec
            try:
                out = chat_fn(messages, Explanation.model_json_schema())
                return {
                    "explanation": Explanation.model_validate(out).text,
                    "explanation_reason": "ok",
                }
            except (OSError, ConnectionError):
                # The chat backend itself is unreachable -- a network/socket
                # failure, not a reply the model returned. Retrying a dead
                # connection twice is still the existing contract; only the
                # reported reason changes.
                log.debug("explain attempt failed", exc_info=True)
                reason = "model_unreachable"
                continue
            except Exception:
                log.debug("explain attempt failed", exc_info=True)
                reason = "no_answer"
                continue
        return {"explanation": None, "explanation_reason": reason}

    graph = StateGraph(ExplainState)
    graph.add_node("profile", synthesize_profile)
    graph.add_node("explain", generate_explanation)
    graph.set_entry_point("profile")
    graph.add_edge("profile", "explain")
    graph.add_edge("explain", END)
    return graph.compile()


def explain(conn, user_id: int, item_id: int, chat_fn) -> ExplainResult:
    """Why this item was ranked here for this reader, cached after the first
    successful call.

    Never raises -- per this module's reliability contract, ranking never
    waits on an explanation, so this must never raise into that path. But it
    no longer collapses every failure into the same `None`: `reason` names
    which of three unrelated causes produced no text -- an unknown user_id
    (the caller's argument is wrong), the chat backend being unreachable
    (retry later), or the model answering with nothing usable (also worth a
    retry, but a different failure) -- so a caller does not have to
    re-derive the distinction `_explain_item` used to reconstruct by hand.
    """
    cached = conn.execute(
        "SELECT text FROM explanations WHERE user_id = ? AND item_id = ?",
        (user_id, item_id),
    ).fetchone()
    if cached:
        return ExplainResult(text=cached["text"], reason="ok")
    # Cheap guard before the graph: a deleted persona (or a stale browser
    # session still holding its id) would otherwise cost an LLM call and a
    # logged traceback per request.
    if conn.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone() is None:
        log.warning("explain called for unknown user_id=%s", user_id)
        return ExplainResult(text=None, reason="unknown_user")
    try:
        result = _build_graph(conn, chat_fn).invoke(ExplainState(user_id=user_id, item_id=item_id))
    except Exception:
        log.exception("explain graph failed")
        return ExplainResult(text=None, reason="model_unreachable")
    text = result.get("explanation")
    if text:
        conn.execute(
            "INSERT OR IGNORE INTO explanations(user_id, item_id, text) VALUES (?, ?, ?)",
            (user_id, item_id, text),
        )
        conn.commit()
        return ExplainResult(text=text, reason="ok")
    reason = result.get("explanation_reason", "no_answer")
    return ExplainResult(text=None, reason=reason)

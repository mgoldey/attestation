"""LangGraph explain agent: click history -> profile -> one-sentence 'why ranked here'.

Hermes is the orchestrator; the chat model is a swappable OpenAI-compatible backend
(see src/attestation/llm.py).

Reliability contract: lazy, cached, degrades to None. Ranking never waits on this.
"""

import logging
import sqlite3

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from attestation.llm import default_chat_fn

log = logging.getLogger(__name__)


class Explanation(BaseModel):
    text: str = Field(min_length=1, max_length=300)


class ExplainState(BaseModel):
    user_id: int
    item_id: int
    profile: str = ""
    explanation: str | None = None


def _build_graph(conn: sqlite3.Connection, chat_fn):
    def synthesize_profile(state: ExplainState) -> dict:
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
        try:
            out = chat_fn(
                [
                    {"role": "system", "content": "Summarize this reader in one sentence."},
                    {
                        "role": "user",
                        "content": "Recently useful titles:\n- " + "\n- ".join(titles),
                    },
                ],
                Explanation.model_json_schema(),
            )
            return {"profile": Explanation.model_validate(out).text}
        except Exception:
            log.debug("explain attempt failed", exc_info=True)
            log.warning("profile synthesis failed; using interests text")
            return {"profile": interests}

    def generate_explanation(state: ExplainState) -> dict:
        item = conn.execute(
            "SELECT title, summary FROM items WHERE id = ?", (state.item_id,)
        ).fetchone()
        messages = [
            {
                "role": "system",
                "content": (
                    "You explain feed rankings. One sentence, second person,"
                    " grounded ONLY in the reader profile given. No hedging."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Reader profile: {state.profile}\n"
                    f"Item: {item['title']}\n{item['summary'][:500]}\n"
                    "Why is this ranked here for this reader?"
                ),
            },
        ]
        for _ in range(2):  # one retry per spec
            try:
                out = chat_fn(messages, Explanation.model_json_schema())
                return {"explanation": Explanation.model_validate(out).text}
            except Exception:
                log.debug("explain attempt failed", exc_info=True)
                continue
        return {"explanation": None}

    graph = StateGraph(ExplainState)
    graph.add_node("profile", synthesize_profile)
    graph.add_node("explain", generate_explanation)
    graph.set_entry_point("profile")
    graph.add_edge("profile", "explain")
    graph.add_edge("explain", END)
    return graph.compile()


def explain(conn, user_id: int, item_id: int, chat_fn=default_chat_fn) -> str | None:
    cached = conn.execute(
        "SELECT text FROM explanations WHERE user_id = ? AND item_id = ?",
        (user_id, item_id),
    ).fetchone()
    if cached:
        return cached["text"]
    # Cheap guard before the graph: a deleted persona (or a stale browser
    # session still holding its id) would otherwise cost an LLM call and a
    # logged traceback per request.
    if conn.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone() is None:
        log.warning("explain called for unknown user_id=%s", user_id)
        return None
    try:
        result = _build_graph(conn, chat_fn).invoke(ExplainState(user_id=user_id, item_id=item_id))
    except Exception:
        log.exception("explain graph failed")
        return None
    text = result.get("explanation")
    if text:
        conn.execute(
            "INSERT OR IGNORE INTO explanations(user_id, item_id, text) VALUES (?, ?, ?)",
            (user_id, item_id, text),
        )
        conn.commit()
    return text

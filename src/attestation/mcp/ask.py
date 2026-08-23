"""The four `.ask` tools: they call a router, then the tool it names.

The routing rules themselves live in `routing.py` -- kept separate because
they are pure functions over a string, testable with no model and no
database, and they are the regression guard for the measured 13/15. This
module is the part that touches the rest of the system.
"""

import os

from pydantic import BaseModel, Field

from attestation.mcp.routing import (
    Decision,
    route_feed,
    route_kg,
    route_runs,
    route_sym,
)

# --- the tools ------------------------------------------------------------


class Ref(BaseModel):
    """An item, reduced to what a follow-up call needs.

    Id and url only. Deliberately too little to tempt a model into rewriting
    the list -- rewriting a ten-item payload is what looped in Telegram.
    """

    item_id: int
    url: str | None = None


class Answer(BaseModel):
    """The shape every `ask` tool returns, declared so a UI can render it.

    `answer` is written here and relayed VERBATIM. `caveat` carries
    ranking_quality's honesty note and runs.compare's caveats unabridged,
    because an answer that drops them reads as confident. `options` is
    populated only when the router declined and needs the caller to choose.
    """

    ok: bool
    answer: str = Field(description="One or two lines. Render verbatim; do not reformat.")
    refs: list[Ref] = Field(default_factory=list, description="Ids for follow-up calls.")
    caveat: str | None = Field(default=None, description="Reasons not to trust the answer.")
    options: list[str] = Field(default_factory=list, description="Tools to choose between.")
    tool_used: str | None = Field(default=None, description="Which tool answered.")


def _declined(decision: Decision) -> dict:
    return {
        "ok": False,
        "answer": decision.question,
        "refs": [],
        "caveat": None,
        "options": list(decision.options),
        "tool_used": None,
    }


def _feed_ask(user: str, question: str) -> dict:
    from attestation.mcp import feed as feed_mod

    decision = route_feed(question)
    if decision.tool is None:
        return _declined(decision)

    if decision.tool == "feed.search":
        out = feed_mod._search_feed(user, decision.kwargs.get("query", question))
    elif decision.tool == "feed.digest":
        out = feed_mod._digest(user)
    elif decision.tool in {"feed.sources", "feed.source_suggest"}:
        from attestation.mcp import subscriptions as subs

        out = (
            subs._suggest_feeds(user)
            if decision.tool == "feed.source_suggest"
            else subs._list_feeds()
        )
    elif decision.tool == "feed.persona_status":
        out = feed_mod._profile_status(user)
    elif decision.tool == "feed.read":
        return {
            "ok": False,
            "answer": "Which item? Pass the item_id and I will read it in full.",
            "refs": [],
            "caveat": None,
            "options": ["feed.read"],
            "tool_used": None,
        }
    elif decision.tool == "feed.source_preview":
        return {
            "ok": False,
            "answer": "Which feed? Give me its URL and I will show recent entries.",
            "refs": [],
            "caveat": None,
            "options": ["feed.source_preview", "feed.source_suggest"],
            "tool_used": None,
        }
    elif decision.tool in {"feed.rate", "feed.explain", "feed.source_add", "feed.source_remove"}:
        # These need an item or a url the question does not carry. Naming the
        # tool is the answer: the caller supplies the argument it already has.
        return {
            "ok": False,
            "answer": f"Tell me which item, then I will call {decision.tool}.",
            "refs": [],
            "caveat": None,
            "options": [decision.tool],
            "tool_used": None,
        }
    else:
        out = feed_mod._list_feed(user)
    return _compose(out, decision.tool)


def _compose(out: dict, tool: str) -> dict:
    """Turn a tool's payload into an Answer, keeping every caveat."""
    items = out.get("items") or []
    refs = [{"item_id": i["item_id"], "url": i.get("url")} for i in items if "item_id" in i]
    quality = out.get("ranking_quality") or {}
    caveats = [c for c in (quality.get("caveat"), out.get("caveat")) if c]
    caveats += list(out.get("caveats") or [])

    answer = _summarise(out)
    return {
        "ok": bool(out.get("ok")),
        "answer": answer.strip(),
        "refs": refs,
        "caveat": " ".join(caveats) or None,
        "options": [],
        "tool_used": tool,
    }


# Where a tool puts its results. Different names for the same idea, and a
# caller should not have to know which -- nor should _compose branch over all
# of them inline.
# The payload keys a summary may draw its answer from. `path` and `neighbors`
# were missing, so kg.path and kg.neighbors -- two of kg.ask's five routes --
# named the right tool and threw its result away: "2 hop(s)" and "7
# neighbour(s)", with the concepts sitting unused in the payload. That is the
# failure _summarise's own docstring describes.
#
# A route whose result key is absent here degrades to a bare count, which is
# silent, so test_ask_routing asserts the ANSWER names concepts rather than
# counting them.
_RESULT_KEYS = (
    "items",
    "nodes",
    "concepts",
    "neighbors",
    "path",
    "arms",
    "communities",
    "feeds",
    "runs",
    "suggestions",
)


def _summarise(out: dict) -> str:
    """One line naming the actual results, not just counting them.

    "top 10 by degree" tells a reader nothing; the ten concepts do.
    """
    named: list = next((out[k] for k in _RESULT_KEYS if out.get(k)), [])
    labels = [x for x in (_label(n) for n in named[:5]) if x]
    headline = (out.get("message") or "").split(";")[0].strip()
    if not labels:
        return out.get("message") or ""
    return f"{headline}: {'; '.join(labels)}" if headline else "; ".join(labels)


# One label's share of an answer. Five labels ride in every router reply, and
# `_summarise` capped their COUNT but not their length -- the longest title in
# the live database is 223 chars, so five ordinary arXiv titles made a 1115-char
# answer, and a synthetic 360-char title made 1901. Same shape as the
# provenance caveat that breached the payload budget three rounds running: a
# string on a hot path whose length is a property of the data, not the code.
MAX_LABEL_CHARS = 90


def _label(x) -> str:
    """One readable line for whatever a tool returned, truncated to fit.

    Items, graph nodes, concepts, comparison arms and feeds all have different
    key names for the same idea, and a caller should not have to know which.

    Truncation is visible (an ellipsis) rather than silent: a caller quoting a
    half-title should be able to see that it was cut, and `refs` carries the
    ids for anything that needs the full record.
    """
    if isinstance(x, str):
        return _clip(x)
    if not isinstance(x, dict):
        return _clip(str(x))
    title = x.get("title") or x.get("name") or x.get("label") or x.get("tag")
    source = x.get("source") or x.get("project")
    if title and source:
        return f"{_clip(str(title))} ({source})"
    return _clip(str(title or ""))


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_LABEL_CHARS else text[:MAX_LABEL_CHARS].rstrip() + "…"


def _which_family(listed: dict) -> str:
    """The disambiguation prompt, saying how many families it did not name.

    Every other truncation in this repo reports itself -- runs.list's own
    message is "showing 8 of 40 families -- pass project= to narrow them". A
    silent one is worse here than there: a caller whose family is the ninth
    reads a complete-looking list, does not find their sweep on it, and
    concludes it was never scanned. `n_families` was already in hand.
    """
    shown = list(listed.get("families") or [])
    names = ", ".join(f["family"] for f in shown)
    total = listed.get("n_families") or len(shown)
    answer = f"Which family? Comparable families include: {names}"
    if total > len(shown):
        answer += (
            f" -- {len(shown)} of {total}; name yours directly,"
            " or call runs.list with project= to narrow them"
        )
    return answer


def _runs_ask(question: str, family: str | None = None, path: str | None = None) -> dict:
    from attestation.mcp import provenance as prov

    decision = route_runs(question)
    if decision.tool is None:
        return _declined(decision)
    if decision.tool == "runs.compare":
        if not family:
            return {
                "ok": False,
                "answer": _which_family(prov._list(limit=5)),
                "refs": [],
                "caveat": None,
                "options": ["runs.compare"],
                "tool_used": None,
            }
        out = prov._compare(family)
    elif decision.tool == "runs.claims_check":
        out = prov._check(path)
    elif decision.tool == "runs.claims_coverage":
        out = prov._coverage(path)
    elif decision.tool == "runs.scan":
        out = prov._scan(confirm=True)
    else:
        out = prov._list(limit=10)
    return _compose(out, decision.tool)


def _kg_ask(question: str, source: str | None = None, target: str | None = None) -> dict:
    from attestation.mcp import knowledge as kg_mod

    decision = route_kg(question)
    if decision.tool is None:
        return _declined(decision)
    if decision.tool == "kg.path":
        if not (source and target):
            return {
                "ok": False,
                "answer": "Which two concepts? Pass source and target.",
                "refs": [],
                "caveat": None,
                "options": ["kg.path", "kg.concepts"],
                "tool_used": None,
            }
        out = kg_mod._path(source, target)
    elif decision.tool == "kg.central":
        out = kg_mod._central()
    elif decision.tool == "kg.communities":
        out = kg_mod._communities()
    elif decision.tool == "kg.neighbors":
        if not source:
            return {
                "ok": False,
                "answer": "Which concept?",
                "refs": [],
                "caveat": None,
                "options": ["kg.neighbors", "kg.concepts"],
                "tool_used": None,
            }
        out = kg_mod._neighbors(source)
    else:
        out = kg_mod._concepts()
    return _compose(out, decision.tool)


def _sym_ask(expr: str, question: str = "simplify") -> dict:
    from attestation.mcp import symbolic as sym_mod

    decision = route_sym(question, expr)
    if decision.tool is None:
        return _declined(decision)
    if decision.tool == "sym.verify":
        return {
            "ok": False,
            "answer": "Give me both sides to compare, as lhs and rhs.",
            "refs": [],
            "caveat": None,
            "options": ["sym.verify"],
            "tool_used": None,
        }
    dispatch = {
        "sym.simplify": sym_mod._sym_simplify,
        "sym.solve": sym_mod._sym_solve,
        "sym.differentiate": sym_mod._sym_differentiate,
        "sym.integrate": sym_mod._sym_integrate,
        "sym.derivation": sym_mod._sym_derivation,
        "sym.evaluate": sym_mod._sym_evaluate,
    }
    # Default rather than .get(): route_sym only returns names in this map or
    # None, both handled above, so a miss would be a routing bug -- and
    # simplify is the honest thing to do with an expression regardless.
    out = dispatch.get(decision.tool, sym_mod._sym_simplify)(expr)
    return {
        "ok": bool(out.get("ok")),
        "answer": str(out.get("result") or out.get("message") or ""),
        "refs": [],
        "caveat": None,
        "options": [],
        "tool_used": decision.tool,
    }


# Re-exported: the disclosure tools are registered from ask.register, and
# test_architecture requires every tool's implementation to be importable from
# the module that registers it.
from attestation.mcp.disclosure import _tools_listing  # noqa: E402,F401


def register(mcp) -> None:
    """Attach the four `ask` routers and their disclosure tools."""

    # Registered ONLY when this agent's specifics are hidden. With the full
    # surface served they are all visible, and four identical tools claiming
    # otherwise is a falsehood in the commonest configuration.
    from attestation.mcp.disclosure import register_disclosure

    register_disclosure(
        mcp,
        os.environ.get("ATTEST_TOOLS", "").strip(),
        os.environ.get("ATTEST_EXPAND", "").strip() not in ("", "0", "false"),
    )

    @mcp.tool(name="feed.ask")
    def feed_ask(user: str, question: str) -> Answer:
        """Ask anything about this reader's PAPERS AND ARTICLES, in their words.

        Start here, including for "find me papers on X" and "what's new in Y".
        The reader's corpus is built from subscribed feeds -- on this
        installation mostly arXiv, plus Nature, Scientific Reports, Hugging
        Face and HN -- so questions about recent research are answerable here
        without any network call.

        Routes to the right feed tool by rule -- no extra model call, no
        guessing -- and returns a line to relay VERBATIM plus the ids needed
        to act. Reads `caveat` before trusting the order.

        If the question is ambiguous it asks back and names the alternatives
        in `options` rather than picking a default.
        """
        return Answer(**_feed_ask(user, question))

    @mcp.tool(name="runs.ask")
    def runs_ask(question: str, family: str | None = None, path: str | None = None) -> Answer:
        """Ask about recorded experiment runs, or numbers written in a draft.

        Start here. `family` names a set of arms to compare; `path` names a
        Markdown file to check. Caveats from a comparison pass through
        unabridged -- report them.
        """
        return Answer(**_runs_ask(question, family, path))

    @mcp.tool(name="kg.ask")
    def kg_ask(question: str, source: str | None = None, target: str | None = None) -> Answer:
        """Ask about the reading knowledge graph, in plain words.

        Start here. `source` and `target` name concepts when the question is
        about how two topics connect.
        """
        return Answer(**_kg_ask(question, source, target))

    @mcp.tool(name="sym.ask")
    def sym_ask(expr: str, question: str = "simplify") -> Answer:
        """Do something with a mathematical expression.

        `expr` is the expression; `question` says what to do with it --
        simplify, solve, differentiate, integrate, evaluate, or show steps.
        """
        return Answer(**_sym_ask(expr, question))

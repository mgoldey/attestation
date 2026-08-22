"""Intent routers: one question in, one tool call out, by rule.

A single agent choosing among 37 tools picked correctly 8 times in 15,
measured on gemma4:e2b over realistic turns, three runs, no variance. Routing
to four intent tools scored 13/15 at the same latency. An LLM swarm --
supervisor picks a namespace, subagent picks the tool -- scored 7.3/15 at
twice the latency, because a second model call is a second chance to be wrong
and a namespace miss cannot be recovered.

So the routing here is rules. That is what holds latency flat, what makes the
15 cases testable without a model or a database, and what makes a wrong route
a bug someone can fix rather than a sampling accident.

**No catch-all.** An early version had a `doctor` destination for "diagnose
the system" and it became a magnet: three of four remaining misses went to it.
A question the rules do not claim confidently returns a question and the
alternatives, never a default.
"""

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class Decision:
    """Where a question routes, or why it does not.

    `tool` is None exactly when the router declines. `question` and `options`
    are then populated so the caller can ask rather than guess.
    """

    tool: str | None
    kwargs: dict = field(default_factory=dict)
    question: str = ""
    options: tuple[str, ...] = ()


def _has(text: str, *phrases: str) -> bool:
    return any(p in text for p in phrases)


def _strip_topic(text: str) -> str:
    """The subject of a search, with the asking removed.

    Rule-based on purpose: "anything new on retrieval augmented generation?"
    should search for the topic, not for the whole sentence.
    """
    cleaned = re.sub(
        r"^\s*(is there |are there |do you have |show me |find me |"
        r"anything |any )?(new |recent |good )?(on|about|for|regarding)?\s*",
        "",
        text.strip().rstrip("?").lower(),
    )
    return cleaned.strip() or text.strip().rstrip("?")


def route_feed(question: str) -> Decision:
    """Route a question about the reader's feed or persona."""
    q = question.lower().strip()
    if not q:
        return Decision(
            None,
            question="What would you like to know about your feed?",
            options=("feed.list", "feed.search", "feed.digest"),
        )

    # Order matters: the more specific intents are tested first, because
    # "why did that rank" also contains words that look like a search.
    if _has(q, "why did", "why is", "why was", "explain", "how come"):
        return Decision("feed.explain", {})
    if _has(
        q,
        "not relevant",
        "isn't relevant",
        "not interested",
        "already read",
        "not useful",
        "useful",
        "good find",
        "not my area",
        "wrong subfield",
    ):
        return Decision("feed.rate", {})
    if _has(
        q,
        "been reading",
        "reading about lately",
        "digest",
        "by topic",
        "grouped",
        "themes",
        "what have i read",
    ):
        return Decision("feed.digest", {})
    # Mutations before the listing they mention: "add arxiv cs.CL to my
    # feeds" contains "my feeds" but is not a request to see them.
    if _has(q, "add ", "subscribe"):
        return Decision("feed.source_add", {})
    if _has(
        q,
        "my feeds",
        "subscriptions",
        "subscribed",
        "which feeds",
        "list feeds",
        "show me my feeds",
    ):
        return Decision("feed.sources", {})
    if _has(q, "unsubscribe", "remove feed", "drop feed"):
        return Decision("feed.source_remove", {})
    if _has(q, "persona", "profile", "how well trained", "how trained", "my interests"):
        return Decision("feed.persona_status", {})
    if _has(
        q,
        "should i read",
        "what's new",
        "whats new",
        "read today",
        "recommend",
        "my feed",
        "anything for me",
    ):
        return Decision("feed.list", {})
    # A search needs a subject AND a word that means searching. " about " is
    # too weak: "tell me about machine learning" could mean the feed, the
    # graph or the archive, and guessing between them is the behaviour the
    # catch-all finding warns against.
    if _has(
        q,
        "anything new",
        "anything on",
        "papers on",
        "papers about",
        "articles on",
        "work on",
        "find",
        "search",
        "look for",
    ):
        topic = _strip_topic(question)
        if topic and len(topic.split()) >= 2:
            return Decision("feed.search", {"query": topic})
    return Decision(
        None,
        question="Did you mean your current feed, or a search of the whole archive?",
        options=("feed.list", "feed.search", "feed.digest"),
    )


def route_runs(question: str) -> Decision:
    """Route a question about recorded runs or claims in a draft."""
    q = question.lower().strip()
    if not q:
        return Decision(
            None,
            question="What would you like to know about your runs?",
            options=("runs.list", "runs.compare", "runs.claims_check"),
        )
    if _has(
        q,
        "forget to cite",
        "forgot to cite",
        "not cite",
        "uncovered",
        "no claim",
        "not cited",
        "coverage",
    ):
        return Decision("runs.claims_coverage", {})
    if _has(
        q,
        "draft",
        "claim",
        "manuscript",
        "paper right",
        "numbers right",
        "check the numbers",
        "verify",
    ):
        return Decision("runs.claims_check", {})
    if _has(q, "won", "winner", "best arm", "compare", "which arm", "ablation", "sweep"):
        return Decision("runs.compare", {})
    if _has(q, "detail", "config", "one run", "show run"):
        return Decision("runs.detail", {})
    if _has(q, "recorded", "what runs", "list runs", "my runs", "families"):
        return Decision("runs.list", {})
    if _has(q, "scan", "re-read", "reread", "pick up new"):
        return Decision("runs.scan", {})
    return Decision(
        None,
        question="Did you mean comparing arms, listing runs, or checking a draft?",
        options=("runs.compare", "runs.list", "runs.claims_check"),
    )


def route_kg(question: str) -> Decision:
    """Route a question about the reading knowledge graph."""
    q = question.lower().strip()
    if not q:
        return Decision(
            None,
            question="What would you like to know about your reading graph?",
            options=("kg.concepts", "kg.path", "kg.central"),
        )
    if _has(q, "connect", "relate", "between", "path", "link"):
        return Decision("kg.path", {})
    if _has(
        q, "most", "central", "important", "biggest", "read about most", "dominant", "top topic"
    ):
        return Decision("kg.central", {})
    if _has(q, "cluster", "group", "theme", "communit"):
        return Decision("kg.communities", {})
    if _has(q, "next to", "adjacent", "neighbour", "neighbor", "related to", "near "):
        return Decision("kg.neighbors", {})
    if _has(q, "concept", "topic", "vocabulary", "what exists", "what do i", "tags"):
        return Decision("kg.concepts", {})
    return Decision(
        None,
        question="Did you mean listing concepts, connecting two, or finding the central ones?",
        options=("kg.concepts", "kg.path", "kg.central"),
    )


def route_sym(question: str, expr: str = "") -> Decision:
    """Route a symbolic-mathematics request.

    `expr` is separate from the question because an expression is data, not
    phrasing: extracting `x**2 - 4` from prose is exactly the guessing this
    module exists to avoid.
    """
    q = question.lower().strip()
    subject = expr or _extract_expr(question)
    if _has(q, "equal", "same as", "identity", "verify", "prove"):
        return Decision("sym.verify", {"expr": subject})
    if _has(q, "solve", "root", "zero of"):
        return Decision("sym.solve", {"expr": subject})
    if _has(q, "differentiate", "derivative", "d/dx"):
        return Decision("sym.differentiate", {"expr": subject})
    if _has(q, "integrate", "integral", "antiderivative"):
        return Decision("sym.integrate", {"expr": subject})
    if _has(q, "step", "derivation", "show the work", "how do you get"):
        return Decision("sym.derivation", {"expr": subject})
    if _has(q, "evaluate", "compute", "value of", "how much"):
        return Decision("sym.evaluate", {"expr": subject})
    if _has(q, "simplify", "reduce", "canonical"):
        return Decision("sym.simplify", {"expr": subject})
    return Decision(
        None,
        question=(
            "What should I do with it -- simplify, solve, differentiate, integrate, or verify?"
        ),
        options=("sym.simplify", "sym.solve", "sym.differentiate", "sym.verify"),
    )


def _extract_expr(text: str) -> str:
    """The mathematical part of a sentence, if it looks like one.

    Deliberately conservative: a wrong expression is worse than none, since
    the caller can pass `expr` explicitly.
    """
    match = re.search(r"[A-Za-z0-9_.\s()*/+^-]*[*^/][A-Za-z0-9_.\s()*/+^-]*", text)
    candidate = (match.group(0) if match else "").strip()
    return candidate if any(c.isalnum() for c in candidate) else ""


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
    elif decision.tool == "feed.sources":
        out = feed_mod._list_feeds()
    elif decision.tool == "feed.persona_status":
        out = feed_mod._profile_status(user)
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
_RESULT_KEYS = ("items", "nodes", "concepts", "arms", "communities", "feeds", "runs")


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


def _label(x) -> str:
    """One readable line for whatever a tool returned.

    Items, graph nodes, concepts, comparison arms and feeds all have different
    key names for the same idea, and a caller should not have to know which.
    """
    if isinstance(x, str):
        return x
    if not isinstance(x, dict):
        return str(x)
    title = x.get("title") or x.get("name") or x.get("label") or x.get("tag")
    source = x.get("source") or x.get("project")
    if title and source:
        return f"{title} ({source})"
    return str(title or "")


def _runs_ask(question: str, family: str | None = None, path: str | None = None) -> dict:
    from attestation.mcp import provenance as prov

    decision = route_runs(question)
    if decision.tool is None:
        return _declined(decision)
    if decision.tool == "runs.compare":
        if not family:
            listed = prov._list(limit=5)
            names = ", ".join(f["family"] for f in (listed.get("families") or [])[:8])
            return {
                "ok": False,
                "answer": f"Which family? Comparable families include: {names}",
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


def _tools_listing(surface: str) -> dict:
    """What the specific tools in this agent are, and how to reach them."""
    return {
        "ok": True,
        "answer": (
            f"The {surface} agent's specific tools are hidden by default,"
            " because a visible tool gets called: measured on gemma4:e2b, a"
            " model picked the ask router 1 time in 26 when the specifics were"
            f" listed alongside it, and 26 in 26 when they were not. Set"
            f" ATTEST_EXPAND=1 on the server to see them, or keep using"
            f" {surface}.ask, which routes by rule and asks back when a"
            " question is ambiguous."
        ),
        "refs": [],
        "caveat": None,
        "options": [f"{surface}.ask"],
        "tool_used": f"{surface}.tools",
    }


def register(mcp) -> None:
    """Attach the four `ask` routers and their disclosure tools."""

    for _surface in ("feed", "runs", "kg", "sym"):

        def _make(surface=_surface):
            @mcp.tool(name=f"{surface}.tools")
            def _list_tools() -> Answer:
                """Explain how to reach this agent's specific tools.

                The specific tools are hidden by default so the router is
                chosen instead of guessed at. This says how to reveal them.
                """
                return Answer(**_tools_listing(surface))

            return _list_tools

        _make()

    @mcp.tool(name="feed.ask")
    def feed_ask(user: str, question: str) -> Answer:
        """Ask anything about this reader's feed, in their own words.

        Start here. Routes to the right feed tool by rule -- no extra model
        call, no guessing -- and returns a line to relay VERBATIM plus the ids
        needed to act. Reads `caveat` before trusting the order.

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

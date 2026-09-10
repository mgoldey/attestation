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


# The feed rule table, in order. First match wins, and the ORDER is the
# design: "add arxiv cs.CL to my feeds" contains "my feeds" but is not a
# request to see them, and "what feeds should I subscribe to" contains
# "subscribe" but is a question rather than an instruction.
#
# Data rather than a chain of ifs because the chain reached 17 branches and
# the ordering -- the part that actually matters -- was invisible in it.
_FEED_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Content before ranking: "what is that paper about" asks what it SAYS,
    # "why is it here" asks why it RANKED.
    (
        "feed.read",
        (
            "summarize",
            "summarise",
            "abstract",
            "what is that paper about",
            "what is it about",
            "what does it say",
            "read me",
            "tell me about item",
        ),
    ),
    ("feed.explain", ("why did", "why is", "why was", "explain", "how come")),
    (
        "feed.rate",
        (
            "not relevant",
            "isn't relevant",
            "not interested",
            "already read",
            "not useful",
            "useful",
            "good find",
            "not my area",
            "wrong subfield",
        ),
    ),
    (
        "feed.digest",
        (
            "been reading",
            "reading about lately",
            "digest",
            "by topic",
            "grouped",
            "themes",
            "what have i read",
        ),
    ),
    (
        "feed.source_preview",
        ("preview", "what's in that feed", "whats in that feed", "what it publishes"),
    ),
    ("feed.source_add", ("add ", "subscribe", "follow ")),
    (
        "feed.sources",
        (
            "my feeds",
            "subscriptions",
            "subscribed",
            "which feeds",
            "list feeds",
            "show me my feeds",
        ),
    ),
    ("feed.source_remove", ("unsubscribe", "remove feed", "drop feed")),
    (
        "feed.persona_status",
        ("persona", "profile", "how well trained", "how trained", "my interests"),
    ),
    (
        "feed.list",
        (
            "should i read",
            "what's new",
            "whats new",
            "read today",
            "recommend",
            "my feed",
            "anything for me",
            # Naming the feed by rank or by day. "my feed" alone missed
            # "my ranked feed" (the adjective split the phrase) and nothing
            # matched "today's feed" -- measured over Discord 2026-09-04,
            # where the ambiguity reply made the model claim it had no tool.
            "ranked feed",
            "daily feed",
            "today's feed",
            "feed today",
            "for me today",
            "top papers",
        ),
    ),
)

# Advice beats action, but only when no URL is present: a reader who names a
# feed is not asking for advice about it.
_SUGGEST_PHRASES = (
    "should i subscribe",
    "should i follow",
    "suggest",
    "recommend feed",
    "expand my source",
    "more coverage",
    "missing any source",
    "what feeds",
)

# A search needs a subject AND a word that means searching. " about " is too
# weak -- "tell me about machine learning" could mean the feed, the graph or
# the archive, and guessing is what the catch-all finding warns against.
_SEARCH_PHRASES = (
    "anything new",
    "anything on",
    "papers on",
    "papers about",
    "articles on",
    "work on",
    "find",
    "search",
    "look for",
)


def route_feed(question: str) -> Decision:
    """Route a question about the reader's feed or persona."""
    q = question.lower().strip()
    if not q:
        return Decision(
            None,
            question="What would you like to know about your feed?",
            options=("feed.list", "feed.search", "feed.digest"),
        )

    if "http" not in q and _has(q, *_SUGGEST_PHRASES):
        return Decision("feed.source_suggest", {})

    for tool_name, phrases in _FEED_RULES:
        if _has(q, *phrases):
            return Decision(tool_name, {})

    if _has(q, *_SEARCH_PHRASES):
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
    # Checked first: "save these metrics for my sweep" contains "sweep" (the
    # compare rule's phrase) and "what runs do I have recorded?" contains
    # "record" as a substring of "recorded" (the list rule's phrase) -- both
    # false-positive against a later rule if this one is not checked first.
    # "record " (trailing space) rather than bare "record" for the same
    # reason: it must not also match "recorded".
    if _has(
        q,
        "record ",
        "write the results",
        "leave files for the ledger",
        "save these metrics",
    ):
        return Decision("runs.record", {})
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

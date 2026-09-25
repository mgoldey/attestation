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


# The feed rule tables, in order. First match wins, and the ORDER is the
# design: "add arxiv cs.CL to my feeds" contains "my feeds" but is not a
# request to see them, and "what feeds should I subscribe to" contains
# "subscribe" but is a question rather than an instruction.
#
# Data rather than a chain of ifs because the chain reached 17 branches and
# the ordering -- the part that actually matters -- was invisible in it.
#
# Split into content and source tables with the research/track hook checked
# BETWEEN them in `route_feed` -- see `routing_research`'s module docstring
# for why that ordering is load-bearing.
_CONTENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
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
)
_SOURCE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # "subscribe " with its space: bare "subscribe" also matched "subscribed",
    # so "what feeds am I subscribed to?" was routed to ADD a feed.
    ("feed.source_add", ("add ", "subscribe ", "subscribe me", "follow ")),
    (
        "feed.sources",
        (
            "my feeds",
            "subscriptions",
            "subscribed",
            "which feeds",
            "list feeds",
            "show me my feeds",
            # Freshness: every row carries last_fetched, and a reader asking
            # "when was your recent scrape?" (real session, 2026-09-04) was
            # told it could not be known.
            "last fetched",
            "last updated",
            "recent scrape",
            "last scrape",
            "last ingest",
            "last refresh",
            "how fresh",
            "how current",
        ),
    ),
    ("feed.source_remove", ("unsubscribe", "remove feed", "drop feed")),
    (
        "feed.persona_status",
        (
            "persona",
            "profile",
            "how well trained",
            "how trained",
            "my interests",
            # "why aren't you learning from what I read?" (real session). Not
            # the bare word "learning" -- "papers on machine learning" would
            # land here instead of in search.
            "are you learning",
            "aren't you learning",
            "you learning from",
            "learn from what",
            "learn from me",
        ),
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

# "what feeds am I subscribed to?" contains "what feeds", so the advice rule
# above claimed it and answered with feeds the reader does NOT have -- the
# opposite of the question. A reader asking about their own subscriptions says
# so; advice wins only when nothing marks the feeds as already theirs.
# Narrow on purpose: bare "am i " also caught "am I missing any sources?", which
# IS a request for advice.
_OWNED_SOURCE_MARKERS = ("subscribed", "feeds am i", "feeds do i", "do i follow", "feeds i have")
_OWNED_FEED_QUESTIONS = (
    "subscribed to",
    "am i subscribed",
    "feeds am i",
    "feeds do i",
    "do i follow",
    "i'm subscribed",
    "im subscribed",
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

# Phrases that introduce a SUBJECT, so the text after them is the query.
# MEASURED 2026-09-25 against real Hermes transcripts: "latest in memory
# systems for LLMs", "What is known about orthogonalized mixtures of agents?"
# and "What articles support these claims about mixture of experts?" all fell
# through to the feed-or-archive clarifier, and on the feed surface the agent
# cannot call either option -- so it told the reader nothing was found, while
# 49 items were tagged memory and 66 mixture-of-experts. Each of these names
# its subject explicitly, which is what separates them from the ambiguous
# "tell me about X" the catch-all finding warns against.
_SUBJECT_PHRASES = (
    "articles about",
    "articles supporting",
    "papers supporting",
    "latest in ",
    "latest on ",
    "latest work on ",
    "known about ",
    "what do we know about ",
    "evidence for ",
    "evidence on ",
    "evidence about ",
    "literature on ",
    "literature about ",
    "claims about ",
)


def _subject_after(question: str) -> str | None:
    """The subject following a `_SUBJECT_PHRASES` introducer, or None.

    Returns the text after the introducer, with a trailing '?' and a leading
    "these claims about" removed. None when no introducer is present or the
    subject is shorter than two words -- one word is too thin to search on
    without guessing.
    """
    lowered = question.lower()
    for phrase in _SUBJECT_PHRASES:
        at = lowered.find(phrase)
        if at < 0:
            continue
        subject = question[at + len(phrase) :].strip().rstrip("?.! ").strip()
        subject = re.sub(r"^(these |those |the )?claims? (about|on) ", "", subject, flags=re.I)
        if len(subject.split()) >= 2:
            return subject
    return None


def _match_rules(q: str, rules: tuple[tuple[str, tuple[str, ...]], ...]) -> Decision | None:
    """The first rule table entry whose phrases appear in `q`, or None."""
    for tool_name, phrases in rules:
        if _has(q, *phrases):
            return Decision(tool_name, {})
    return None


def _before_source_rules(q: str) -> Decision | None:
    """Two readings the source-rule table would otherwise get wrong.

    "the top of my feeds" means ranked items, not the subscription list the
    "my feeds" source rule returns; and "which feeds do I follow?" is a
    question about what the reader already has, which the add rule's
    "follow " would read as an instruction (both from real sessions).
    """
    if _has(q, "top of my feed", "top of the feed", "top items", "top of my list"):
        return Decision("feed.list", {})
    if _has(q, *_OWNED_FEED_QUESTIONS):
        return Decision("feed.sources", {})
    return None


def _search_decision(q: str, question: str) -> Decision | None:
    """feed.search when the question names a subject; None otherwise.

    An explicit introducer ("known about", "latest in") wins over the looser
    search phrases, because its subject is exactly the text that follows it.
    """
    if (subject := _subject_after(question)) is not None:
        return Decision("feed.search", {"query": subject})
    if _has(q, *_SEARCH_PHRASES):
        topic = _strip_topic(question)
        if topic and len(topic.split()) >= 2:
            return Decision("feed.search", {"query": topic})
    return None


def route_feed(question: str) -> Decision:
    """Route a question about the reader's feed or persona.

    Content and feedback rules run first, the going-and-looking router
    (`routing_research.route_research`) runs between the two feed-rule
    tables, and source rules run last -- see `routing_research`'s docstring
    for why that order is load-bearing rather than arbitrary.
    """
    from attestation.mcp.routing_research import route_research

    q = question.lower().strip()
    if not q:
        return Decision(
            None,
            question="What would you like to know about your feed?",
            options=("feed.list", "feed.search", "feed.digest"),
        )

    if "http" not in q and _has(q, *_SUGGEST_PHRASES) and not _has(q, *_OWNED_SOURCE_MARKERS):
        return Decision("feed.source_suggest", {})

    if (content := _match_rules(q, _CONTENT_RULES)) is not None:
        return content

    if (research := route_research(q, question)) is not None:
        return research

    if (early := _before_source_rules(q)) is not None:
        return early

    if (source := _match_rules(q, _SOURCE_RULES)) is not None:
        return source

    if (search := _search_decision(q, question)) is not None:
        return search

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
        "not backed",
        "unbacked",
        "backed by a claim",
        "which numbers",
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
    # "sweep" as a WORD: as a substring it matched run names -- "show me the
    # details of kdsweep_t4" was sent to compare, not detail (measured
    # 2026-09-25, the same collision class as subscribe/subscribed).
    if _has(q, "won", "winner", "best arm", "compare", "which arm", "ablation") or re.search(
        r"\bsweep", q
    ):
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
    if _has(q, "cluster", "group", "theme", "communit", "areas", "fields i", "subfields"):
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

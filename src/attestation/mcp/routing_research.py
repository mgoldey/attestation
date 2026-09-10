"""Going-and-looking routing: `feed.research` and the `research:` track route.

Split out of `routing.py` (2026-09-10) because it is one cohesive new concern
-- phrase tables, a noise regex and the query-extraction logic for a single
router function -- and left inline it pushed `routing.py` over its size cap
for a reason unrelated to the rest of that module's four routers.

**Ordering, which is load-bearing, not incidental.** `route_feed` calls
`route_research` AFTER its content/feedback rules (`feed.read`, `feed.explain`,
`feed.rate`, `feed.digest`, `feed.source_preview`) and BEFORE its source rules
(`feed.source_add`, `feed.sources`, `feed.source_remove`, `feed.persona_status`,
`feed.list`). "summarize the journal article" and "why is this ranked so high
in the journal" both contain the loose " journal" research phrase, but they
ask what an item SAYS or why it RANKED -- a content question about something
already in hand -- so content must win. "track/follow X", on the other hand,
must beat `feed.source_add`'s bare "follow " rule, so the hook runs before the
source table rather than after it. Get this backwards and a routine "why is
this ranked" turn goes to the network instead of `feed.explain`.

**"papers on X" stays local, deliberately.** The feed surface's own goal text
promises "find me recent papers on X" is answered without a network call, and
a 9,000-item local archive is the right first answer to that phrasing. This
router claims a question only when it names a PLACE to look (arXiv, PubMed,
CrossRef, a journal, "the literature", "published") or says "research"/"look
up" outright -- never on "papers on X" alone, which stays on `feed.search`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from urllib.parse import quote_plus

if TYPE_CHECKING:
    from attestation.mcp.routing import Decision

# These name a PLACE to look or say so outright; "papers on X" alone stays
# local, because the feed surface promises that question is answered without
# a network call and the archive is the right first answer.
_RESEARCH_SOURCES = (
    ("pubmed", "pubmed"),
    ("arxiv", "arxiv"),
    ("crossref", "crossref"),
)
_RESEARCH_PHRASES = (
    "on arxiv",
    "on pubmed",
    "from arxiv",
    "from pubmed",
    "search arxiv",
    "search pubmed",
    "crossref",
    "in the literature",
    "published on",
    "published about",
    "published in",
    "been published",
    "research ",
    "look up",
    " journal",
    "outside my feed",
    "beyond my feed",
)
_TRACK_PHRASES = ("track ", "follow ", "watch ", "monitor ", "keep an eye on ", "add a topic")

# A track phrase claims a question only at its START (a leading pleasantry
# allowed): "track X" is an instruction, but "what should I watch out for"
# and "follow up on that paper" merely contain the word mid-sentence, and
# routing those to feed.source_add registered feeds with idiom fragments as
# their query (q=up+on+that+paper, q=my+feed). Anchoring at the start catches
# the instruction shape without a stoplist doing all the work alone.
_TRACK_LEAD = re.compile(r"^\s*(please\s+|can you\s+|could you\s+)*", re.IGNORECASE)

# Even an instruction-shaped, start-anchored track phrase can extract a
# useless topic: "monitor my feed" and "watch this space" both start with a
# track phrase, but the "topic" left over after stripping it is an idiom, not
# a subject. Refused when the extracted topic IS one of these or starts with
# one followed by a space -- a prefix check, not a substring one, so a real
# topic that happens to contain "it" (e.g. "circuit design") is not caught.
_TRACK_STOPLIST = (
    "up",
    "up on",
    "out",
    "out for",
    "my feed",
    "this space",
    "that paper",
    "it",
    "this",
    "that",
)

_RESEARCH_NOISE = re.compile(
    r"\b(on|from|in|via|search|the)\s+(arxiv|pubmed|crossref|the literature)\b"
    r"|\b(recent|new)\s+papers?\b|\bpapers?\s+(on|about|in)\b|\bfor me\b|\bwhat has been\b"
    r"|\bpublished\s+(on|about|in)\b|\blook up\b|\bresearch\b"
    r"|^(search|track|follow|watch|monitor)\b",
    re.IGNORECASE,
)

# A venue named mid-sentence ("papers in Nature Methods on cryo-EM") rather
# than trailing the question. Matched on the ORIGINAL casing before the rest
# of the pipeline lowercases everything, since a capitalised run of words is
# the only signal that it is a venue name and not part of the topic.
_MID_VENUE = re.compile(
    r"\b(in|from)\s+(the\s+)?journal\b" r"|\b(in|from)\s+[A-Z][\w-]*(\s+[A-Z][\w-]*)*(?=\s+on\b)"
)
_TRAIL_VENUE = re.compile(r"\b(in|on)\s+[A-Z][\w-]*(\s+[A-Z][\w-]*)*\s*$")


def _clean(text: str, *, strip_mid_venue: bool) -> str:
    """One pass of noise-stripping, with the mid-venue strip optional."""
    from attestation.mcp.routing import _strip_topic

    q = text
    if strip_mid_venue:
        q = _MID_VENUE.sub(" ", q)
    q = _TRAIL_VENUE.sub(" ", q)
    q = _RESEARCH_NOISE.sub(" ", q)
    return " ".join(_strip_topic(q).split())


def _research_query(question: str) -> str:
    """The topic with the going-and-looking words and any venue removed.

    Stripping a mid-sentence venue name ("papers in Nature Methods on
    cryo-EM") can leave too little behind when the venue was most of the
    sentence, so that pass is tried first and only kept when at least two
    words remain -- otherwise the venue words stay in the topic rather than
    starving it down to one word that then fails the caller's own length
    gate.
    """
    stripped = question.strip().rstrip("?")
    cleaned = _clean(stripped, strip_mid_venue=True)
    if len(cleaned.split()) >= 2:
        return cleaned
    return _clean(stripped, strip_mid_venue=False)


def _is_track_instruction(q: str) -> bool:
    """A track phrase claims the question only when it opens it."""
    stripped = _TRACK_LEAD.sub("", q)
    return stripped.startswith(_TRACK_PHRASES)


def _refused_track_topic(topic: str) -> bool:
    """True when `topic` is idiom debris rather than a subject to track."""
    if not topic:
        return True
    return any(topic == stop or topic.startswith(stop + " ") for stop in _TRACK_STOPLIST)


def _route_track(q: str, question: str) -> Decision | None:
    """`feed.source_add` with a `research:` url for a track/follow instruction.

    Returns None both when the question is not a track instruction at all
    and when it is one whose topic is refused (idiom debris) -- either way
    the caller falls through to its own next check.
    """
    from attestation.mcp.routing import _SUGGEST_PHRASES, Decision, _has

    if not _is_track_instruction(q) or _has(q, *_SUGGEST_PHRASES):
        return None
    topic = _research_query(question)
    if _refused_track_topic(topic) or len(topic.split()) < 2:
        return None
    url = f"research:arxiv,pubmed?q={quote_plus(topic)}"
    return Decision("feed.source_add", {"url": url})


def route_research(q: str, question: str) -> Decision | None:
    """`feed.research` for a named source/venue; `feed.source_add` for tracking.

    `q` is the already-lowercased, stripped question `route_feed` computed;
    `question` is the original casing, needed for `_MID_VENUE` and for the
    URL-encoded query. Returns None to decline -- the caller falls through to
    its own rule tables -- rather than guessing.
    """
    from attestation.mcp.routing import Decision, _has

    if "http" in q:
        return None
    if (track := _route_track(q, question)) is not None:
        return track
    if _has(q, *_RESEARCH_PHRASES):
        topic = _research_query(question)
        if len(topic.split()) >= 2:
            sources = ",".join(s for word, s in _RESEARCH_SOURCES if word in q) or "arxiv,pubmed"
            return Decision("feed.research", {"query": topic, "sources": sources})
    return None

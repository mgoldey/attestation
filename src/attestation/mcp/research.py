"""Ad hoc paper research: the `feed.research` tool.

Searches arXiv, PubMed and CrossRef by query and stores the hits in the
reference library -- NOT the feed. The feed is what arrives on a schedule
(a standing topic is a `research:` feed, registered with feed.source_add);
the library is what the reader keeps, and an ad hoc question is the second
kind. Lives in the feed namespace because the chat surfaces run with
ATTEST_TOOLS=feed and must be able to reach it.

The clients are built ONCE, when `register` runs, so ATTEST_RESEARCH_WEB is
read at construction like every other network flag here.
"""

from datetime import UTC, date, datetime, timedelta
from typing import Annotated

from pydantic import Field

from attestation.mcp._shared import Limit, clamp_limit
from attestation.mcp._tool import ToolError, tool

# 8, not feed.list's 16: a research row carries authors and a venue that a
# feed row does not. Measured in test_research_tools: 8 worst-case rows plus
# the envelope stay under the 7000-char ceiling a 2B model can render.
MAX_RESEARCH_LIMIT = 8
DEFAULT_RESEARCH_LIMIT = 5
CLIENTS: dict | None = None


def _clients() -> dict:
    """The module's clients; built lazily for callers that skip `register` (tests, ask)."""
    global CLIENTS
    if CLIENTS is None:
        from attestation import research

        CLIENTS = research.clients_from_env()
    return CLIENTS


def paper_row(paper, stored: bool) -> dict:
    """One hit's wire row: the feed's clipping rules, 3 authors with the true count.

    Delegates to `Paper.to_row` -- the projection lives on the domain type,
    same as `SearchHit.to_row` and `RankedItem.to_row`, so the mcp layer never
    reaches past `attestation.research`'s public surface for it.
    """
    return paper.to_row(stored=stored)


def _store_papers(conn, papers) -> int:
    """Upsert every paper into the library; return how many were new-or-changed."""
    from attestation import library, research

    today = datetime.now(UTC).date().isoformat()
    stored = 0
    for rec in research.as_records(papers, today):
        _rid, how = library.upsert(conn, rec)
        stored += how != "unchanged"
    conn.commit()
    return stored


def _research_message(fetched, topic, journal: str | None, store: bool, stored: int) -> str:
    """The human-readable summary: offline gets its own sentence, never 'no results'."""
    if fetched.offline:
        return (
            "research clients are off (ATTEST_RESEARCH_WEB=0 when the server started):"
            " nothing was searched. This is not 'no results'."
        )
    return (
        f"{len(fetched.papers)} paper(s) from {', '.join(topic.clients)}"
        + (f" in {journal}" if journal else "")
        + (f"; {stored} new in the library" if store else "; preview only, nothing stored")
        + (f"; {len(fetched.errors)} client(s) failed" if fetched.errors else "")
    )


@tool(
    empty={"papers": [], "n_found": 0, "stored": 0, "offline": False, "errors": []},
    label="research",
)
def _research(
    conn,
    query: str,
    sources: str = "arxiv,pubmed",
    journal: str | None = None,
    since_days: int | None = 365,
    limit: int = DEFAULT_RESEARCH_LIMIT,
    store: bool = True,
) -> dict:
    from attestation import research

    limit = min(clamp_limit(limit), MAX_RESEARCH_LIMIT)
    names = tuple(s.strip() for s in sources.split(",") if s.strip())
    try:
        topic = research.parse_topic(research.topic_url(names, query, journal))
    except research.TopicError as exc:
        raise ToolError(str(exc)) from exc
    since = date.today() - timedelta(days=since_days) if since_days else None
    fetched = research.fetch_topic(topic, _clients(), since=since, limit=limit)
    stored = _store_papers(conn, fetched.papers) if store and fetched.papers else 0
    return {
        "message": _research_message(fetched, topic, journal, store, stored),
        "papers": [paper_row(p, store) for p in fetched.papers[:limit]],
        "n_found": len(fetched.papers),
        "stored": stored,
        "offline": fetched.offline,
        "errors": fetched.errors,
    }


def register(mcp) -> None:
    """Attach feed.research; build the clients now (flag read at construction)."""
    from attestation import research

    global CLIENTS
    CLIENTS = research.clients_from_env()

    @mcp.tool(name="feed.research")
    def research_papers(
        query: Annotated[str, Field(min_length=2, description="what the papers are about")],
        sources: Annotated[
            str, Field(description="comma-separated subset of arxiv, pubmed, crossref")
        ] = "arxiv,pubmed",
        journal: Annotated[
            str | None, Field(description="journal name; pubmed/crossref only", max_length=120)
        ] = None,
        since_days: Annotated[int | None, Field(ge=1, le=3650)] = 365,
        limit: Limit = DEFAULT_RESEARCH_LIMIT,
        store: bool = True,
    ) -> dict:
        """Search arXiv, PubMed and/or CrossRef for papers on a topic, and keep them.

        Goes to the network (unlike feed.search, which searches what already
        arrived). Hits are stored in the reference library -- cite.lookup and
        cite.search find them afterwards by DOI, arXiv id or topic -- and are
        NOT added to the feed; to follow a topic on the hourly schedule, call
        feed.source_add with `research:arxiv,pubmed?q=<topic>`. `store=false`
        previews. `limit` is capped at 8. **Show the reader each `url`.**
        `offline: true` means the clients are disabled, not that nothing matched.
        """
        return _research(query, sources, journal, since_days, limit, store)

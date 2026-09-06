"""Bibliographic tools: the `cite.*` namespace.

Reads the reference library first -- the deduplicated store `cite.sync` fills
from Zotero, `.bib` files and the feed -- and falls back to the disk readers
when the store has nothing. Only when `ATTEST_CITATION_WEB` / `ATTEST_CITATION_SCHOLAR`
were set at server start do arXiv, CrossRef and Semantic Scholar enrich it.
`cite.sources` reports which of those are configured and which can reach the
network, because the answer to "did this leave my machine" has to be askable
from the same surface that would have done the leaving.
"""

import json
from typing import Annotated

from pydantic import Field

from attestation.mcp._shared import Limit, clamp_limit
from attestation.mcp._tool import ToolError, tool


def _resolver():
    from attestation import citations

    return citations.Resolver.from_env()


def _embedder():
    """The embedder; library.search degrades to fielded when it cannot be reached."""
    from attestation.embed import Embedder

    return Embedder()


@tool(empty={"reference": None, "sources": [], "conflicts": {}}, label="cite_lookup")
def _lookup(conn, key: str) -> dict:
    from attestation import library

    row = library.lookup_row(conn, key)
    if row is not None:
        ref = library.to_reference(conn, row)
        sources = [
            dict(r)
            for r in conn.execute(
                "SELECT source, source_key, fetched_at, raw FROM reference_sources"
                " WHERE reference_id = ? ORDER BY source",
                (row["id"],),
            )
        ]
        conflicts = {s["source"]: json.loads(s.pop("raw")).get("conflicts", {}) for s in sources}
        return {
            "reference": ref.to_row(),
            "sources": sources,
            "conflicts": {k: v for k, v in conflicts.items() if v},
        }
    resolver = _resolver()
    found = resolver.lookup(key)
    if found is None:
        configured = ", ".join(s["name"] for s in resolver.sources()) or "none"
        stored = library.status(conn)["references"]
        raise ToolError(
            f"no source has {key!r} (disk readers: {configured};"
            f" library store: {stored} references)"
        )
    return {"reference": found.to_row(), "sources": [], "conflicts": {}}


@tool(
    empty={"references": [], "n_matches": 0, "semantic": False, "caveat": None},
    label="cite_search",
)
def _search(
    conn,
    query: str,
    limit: int = 5,
    author: str | None = None,
    year: int | None = None,
    tag: str | None = None,
) -> dict:
    from attestation import library

    limit = clamp_limit(limit)
    if library.status(conn)["references"] == 0:
        matches = _resolver().search(query)
        return {
            "references": [r.to_row() for r in matches[:limit]],
            "n_matches": len(matches),
            "semantic": False,
            "caveat": (
                "library store is empty: substring search over the disk readers"
                " -- run cite.sync (or `attest library sync`) to fill it"
            ),
        }
    res = library.search(
        conn, query, embedder=_embedder(), author=author, year=year, tag=tag, limit=limit
    )
    return {
        "references": [h.to_row() for h in res.hits],
        "n_matches": res.n_matches,
        "semantic": res.semantic,
        "caveat": res.caveat,
    }


@tool(empty={"uncited": [], "n_claims": 0, "checked": []}, label="cite_check")
def _check(conn, path: str) -> dict:
    from pathlib import Path

    from attestation import citations, claims

    root = Path(path)
    if not root.exists():
        raise ToolError(f"no such path: {path}")

    found, problems = claims.find_claims(root)
    # The store answers first: a key synced from Zotero resolves here even when
    # no .bib sits in the working directory.
    resolver = citations.Resolver.from_env(store=lambda: conn)
    uncited = claims.check_citations(found, resolver)
    # The message states this tool's SCOPE, not just its result. It returned
    # ok=true with an empty message on a document holding a contradicted claim,
    # and gemma4:e2b reported that verbatim as "OK: true (meaning all claims
    # were supported by runs)" -- 3 times out of 3, across three phrasings. A
    # false clean bill of health on a document whose numbers are wrong is the
    # worst answer this repo can give.
    #
    # The docstring already says "pair with runs.claims_check". A model reads
    # that when CHOOSING and then reasons from the payload, so the payload has
    # to carry it too.
    verdict = (
        f"{len(uncited)} claim(s) cite a key no source can resolve"
        if uncited
        else "every cited key resolves"
    )
    return {
        "message": (
            f"{len(found)} claim(s) scanned for CITATION KEYS only -- {verdict}."
            " This says nothing about whether the numbers are right:"
            " call runs.claims_check for that."
        ),
        "n_claims": len(found),
        "malformed": problems[:5],
        "uncited": [
            {"key": v.claim.cite, "where": f"{v.claim.path}:{v.claim.line}", "why": v.message}
            for v in uncited
        ],
        # States the same pairing runs.claims_check's "checked" states: this
        # tool only ever checked citation keys, never the numbers.
        "checked": ["citation"],
    }


@tool(empty={"sources": [], "store": {}}, label="cite_sources")
def _sources(conn) -> dict:
    from attestation import citations, library

    sources = _resolver().sources()
    network = any(s["network"] for s in sources) or citations.s2_enabled()
    if citations.s2_enabled():
        sources = [*sources, {"name": "s2", "network": True}]
    return {"sources": sources, "store": library.status(conn), "offline": not network}


@tool(
    empty={"sources": {}, "embedded": 0, "unembedded": 0, "conflicts": 0},
    label="cite_sync",
)
def _sync(conn, sources: list[str] | None = None, limit: int | None = None) -> dict:
    from attestation import library, library_readers

    readers = library_readers.readers_from_env(conn, sources=sources)
    report = library.sync(conn, readers, embedder=_embedder(), limit=limit)
    out = report.to_dict()
    lines = []
    for name, b in out["sources"].items():
        line = f"{name}: +{b['added']} added, {b['merged']} merged, {b['unchanged']} unchanged"
        if b["enriched"]:
            line += f", {b['enriched']} enriched"
        if b["failed"]:
            line += f", {b['failed']} failed"
        lines.append(line)
    out["message"] = "; ".join(lines) or "no sources configured"
    return out


def register(mcp) -> None:
    """Attach every cite.* tool to the server."""

    @mcp.tool(name="cite.lookup")
    def cite_lookup(
        key: Annotated[str, Field(description="citation key, DOI, arXiv id, or library identity")],
    ) -> dict:
        """One bibliographic record, with every source that contributed to it.

        `conflicts` lists any field two sources disagreed on, per source.
        Looks in the reference library first, then a local Zotero library and
        any .bib files. Reaches CrossRef only when the operator enabled the
        network reader; the returned `source` says which one answered.
        """
        return _lookup(key)

    @mcp.tool(name="cite.search")
    def cite_search(
        query: Annotated[str, Field(description="what the paper is about, a title, or an author")],
        limit: Limit = 5,
        author: Annotated[
            str | None, Field(description="surname filter", min_length=1, max_length=80)
        ] = None,
        year: Annotated[
            int | None, Field(description="exact year filter", ge=1800, le=2100)
        ] = None,
        tag: Annotated[
            str | None, Field(description="a reference tag, as tagged", min_length=1, max_length=32)
        ] = None,
    ) -> dict:
        """Find references in the library by what they are about, a title, or an author.

        Semantic when the library is embedded, substring otherwise -- read
        `semantic` and `caveat` in the reply before describing the result.
        Never queries the network, even when the web readers are enabled: a
        free-text search that fanned out to CrossRef would break the offline
        guarantee in the one configuration where it is possible.
        """
        return _search(query, limit, author, year, tag)

    @mcp.tool(name="cite.check")
    def cite_check(
        path: Annotated[str, Field(description="a Markdown file or a directory of them")],
    ) -> dict:
        """Claims whose citation key no configured source can resolve.

        A lint: it reports that a key is unknown here, never that the cited
        work fails to support the claim. Pair with `runs.claims_check`, which
        verifies the numbers against recorded runs.
        """
        return _check(path)

    @mcp.tool(name="cite.sources")
    def cite_sources() -> dict:
        """Does anything here reach the network? Answers "is this all local".

        Reports every configured bibliographic source and whether each reads
        from disk or online, plus what the library store holds. `offline: true`
        means nothing can leave this machine. The feed, ranking, graph, ledger
        and symbolic tools are always local; the only possible network readers
        are CrossRef and arXiv (ATTEST_CITATION_WEB) and Semantic Scholar
        (ATTEST_CITATION_SCHOLAR) for citations, and this says whether they are on.

        Both gemma4:e2b and hermes3:8b skipped this tool when asked "does
        anything I do here send data over the internet" -- one declined, and
        one asserted from the tool NAMES that everything was local, which is
        the confident wrong answer this tool exists to prevent.
        """
        return _sources()

    @mcp.tool(name="cite.sync")
    def cite_sync(
        sources: Annotated[
            list[str] | None,
            Field(
                description="subset of bibtex, zotero, feed, arxiv, crossref, s2; default all",
                max_length=6,
            ),
        ] = None,
        limit: Annotated[
            int | None, Field(description="max rows per enricher / embed pass", ge=1, le=10000)
        ] = None,
    ) -> dict:
        """Read every configured library into the one deduplicated store.

        BibTeX files, Zotero and the feed's own items with a DOI or arXiv id
        become rows; the same paper from three sources is one row with three
        source entries. arXiv/CrossRef (ATTEST_CITATION_WEB) and Semantic
        Scholar reference lists (ATTEST_CITATION_SCHOLAR) only run if the operator
        set the flag before the server started -- this tool cannot arm them.
        Idempotent: re-running with unchanged sources changes nothing.
        """
        return _sync(sources, limit)

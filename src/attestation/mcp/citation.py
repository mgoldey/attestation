"""Bibliographic lookup tools: the `cite.*` namespace.

Reads references from disk -- a Zotero library and any `.bib` files -- and, only
when `ATTEST_CITATION_WEB` is set, from CrossRef. `cite.sources` reports which
of those are configured and which can reach the network, because the answer to
"did this leave my machine" has to be askable from the same surface that would
have done the leaving.

`needs_db=False` throughout: these read files, not the feed database.
"""

from typing import Annotated

from pydantic import Field

from attestation.mcp._shared import Limit
from attestation.mcp._tool import ToolError, tool


def _resolver():
    from attestation import citations

    return citations.Resolver.from_env()


def _as_dict(ref) -> dict:
    return {
        "key": ref.key,
        "title": ref.title,
        "authors": ref.authors[:6],
        "n_authors": len(ref.authors),
        "year": ref.year,
        "doi": ref.doi,
        "url": ref.url,
        # The provenance pair, on every record. This is what makes the offline
        # guarantee's exception inspectable rather than merely documented.
        "source": ref.source,
        "fetched_at": ref.fetched_at,
    }


@tool(empty={"reference": None}, needs_db=False, label="cite_lookup")
def _lookup(key: str) -> dict:
    resolver = _resolver()
    found = resolver.lookup(key)
    if found is None:
        configured = ", ".join(s["name"] for s in resolver.sources()) or "none"
        raise ToolError(f"no source has {key!r} (configured: {configured})")
    return {"reference": _as_dict(found)}


@tool(empty={"references": [], "n_matches": 0}, needs_db=False, label="cite_search")
def _search(query: str, limit: int = 5) -> dict:
    matches = _resolver().search(query)
    return {
        "references": [_as_dict(r) for r in matches[:limit]],
        "n_matches": len(matches),
    }


@tool(empty={"uncited": [], "n_claims": 0}, needs_db=False, label="cite_check")
def _check(path: str) -> dict:
    from pathlib import Path

    from attestation import claims

    root = Path(path)
    if not root.exists():
        raise ToolError(f"no such path: {path}")

    found, problems = claims.find_claims(root)
    uncited = claims.check_citations(found, _resolver())
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
    }


@tool(empty={"sources": []}, needs_db=False, label="cite_sources")
def _sources() -> dict:
    sources = _resolver().sources()
    return {
        "sources": sources,
        "offline": not any(s["network"] for s in sources),
    }


def register(mcp) -> None:
    """Attach every cite.* tool to the server."""

    @mcp.tool(name="cite.lookup")
    def cite_lookup(
        key: Annotated[str, Field(description="citation key, DOI, or arXiv id")],
    ) -> dict:
        """One bibliographic record, and which source it came from.

        Looks in a local Zotero library and any .bib files. Reaches CrossRef
        only when the operator enabled the network reader; the returned
        `source` says which one answered.
        """
        return _lookup(key)

    @mcp.tool(name="cite.search")
    def cite_search(
        query: Annotated[str, Field(description="words from a title or an author's name")],
        limit: Limit = 5,
    ) -> dict:
        """Find references by title or author, from local sources only.

        Never queries the network, even when the web reader is enabled: a
        free-text search that fanned out to CrossRef would break the offline
        guarantee in the one configuration where it is possible.
        """
        return _search(query, limit)

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
        from disk or online. `offline: true` means nothing can leave this
        machine. The feed, ranking, graph, ledger and symbolic tools are always
        local; the only possible network reader is CrossRef for citations, and
        this says whether it is enabled.

        Both gemma4:e2b and hermes3:8b skipped this tool when asked "does
        anything I do here send data over the internet" -- one declined, and
        one asserted from the tool NAMES that everything was local, which is
        the confident wrong answer this tool exists to prevent.
        """
        return _sources()

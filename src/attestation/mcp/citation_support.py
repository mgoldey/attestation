"""`cite.support`: does the cited paper say what the citing sentence says?

Its own module rather than a ninth tool in `citation.py`: that module is at its
size cap with lookup/search/sync wiring, and this tool shares none of it -- the
work lives in `attestation.support`, and this file is the envelope only.
Served on `provenance` beside `cite.check` (see `AGENT_SURFACES`) and on
`knowledge` with the rest of `cite.*`.
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from attestation.mcp._tool import ToolError, tool

# Citations JUDGED per call: each is one model call, ~5-10 s on gemma4:e2b,
# so 20 keeps a whole-draft call within a client's patience.
MAX_SUPPORT_CHECKS = 20
# Result rows RETURNED, worst first. A row is bounded by SupportResult.to_row's
# field caps (~1.3 KB at their maxima); 4 of them stay under the 7000-char
# response ceiling with the envelope, measured in tests/test_support.py.
MAX_SUPPORT_ROWS = 4


def _found(path: str | None, sentence: str | None, cite: str | None) -> list:
    from attestation import support, support_text

    if sentence:
        if not cite:
            raise ToolError("`sentence` needs `cite`: the key, DOI or arXiv id it cites")
        return [support_text.CitingSentence("<sentence>", 1, sentence, (cite,))]
    if not path:
        raise ToolError("pass `path` (a Markdown or .tex draft) or `sentence` with `cite`")
    if not Path(path).expanduser().exists():
        raise ToolError(f"no such path: {path}")
    return support.citing_sentences(Path(path))


@tool(
    empty={"results": [], "counts": {}, "n_citations": 0, "n_checked": 0, "checked": []},
    label="cite_support",
)
def _support(
    conn,
    path: str | None = None,
    sentence: str | None = None,
    cite: str | None = None,
    judge: str = "auto",
) -> dict:
    from attestation import citations, support

    found = _found(path, sentence, cite)
    try:
        judge_ = support.Judge(judge)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    checker = support.Checker(
        conn=conn, resolver=citations.Resolver.from_env(store=lambda: conn), judge=judge_
    )
    out = support.run(checker, found, limit=MAX_SUPPORT_CHECKS)
    results = support.by_severity(out["results"])
    summary = ", ".join(f"{n} {k}" for k, n in sorted(out["counts"].items()))
    return {
        "message": (
            f"{len(results)} citation(s) checked against the cited paper's own text:"
            f" {summary or 'no citing sentences found'}. `text_level` says what was read"
            " (full, abstract or title): not_found on an abstract may still be in the"
            " body. Every `quote` is verbatim from the source."
        ),
        "results": [r.to_row() for r in results[:MAX_SUPPORT_ROWS]],
        "counts": out["counts"],
        "n_citations": out["n_citations"],
        "n_checked": len(results),
        "judges": out["judges"],
        "notes": [n[:160] for n in out["notes"]],
        "checked": ["support"],
    }


def register(mcp) -> None:
    """Attach cite.support to the server."""

    @mcp.tool(name="cite.support")
    def cite_support(
        path: Annotated[
            str | None, Field(description="a Markdown or .tex draft, or a directory of them")
        ] = None,
        sentence: Annotated[
            str | None, Field(description="one sentence to check instead of a draft")
        ] = None,
        cite: Annotated[
            str | None, Field(description="the key, DOI or arXiv id `sentence` cites")
        ] = None,
        judge: Annotated[
            Literal["auto", "llm", "none"],
            Field(description="auto: local model, else deterministic; none: no model"),
        ] = "auto",
    ) -> dict:
        """Does the cited paper actually say what the citing sentence says?

        For each sentence citing a key ([@key], claim cite=, \\cite{key}),
        reads the paper's best stored text and returns a verdict --
        supported, partially_supported, contradicted, not_found,
        source_unavailable -- with a verbatim quote, `text_level`, and a
        reason. Catches a sentence that keeps the paper's number but changes
        what it measures. Worst first; at most 20 citations judged.
        """
        return _support(path, sentence, cite, judge)

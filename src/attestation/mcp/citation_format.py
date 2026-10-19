"""`cite.format`: formatted citations and BibTeX for library references.

Its own module rather than another tool in mcp/citation.py, which is at its
size cap: the tool is a wrapper, and resolving keys, rendering (citeproc-py
over the bundled CSL styles) and writing BibTeX all live in
`citation_manager.format_keys`, `cite_format` and `library.bibtex`.

Served wherever `cite.*` is: the default surface and `knowledge`.
"""

from typing import Annotated, Literal

from pydantic import Field

from attestation.mcp._tool import ToolError, tool

# At most this many keys per call. Measured worst case: one reference's
# formatted text (capped below) plus its BibTeX is ~1,100 characters, so five
# stay well under the 7,000-character ceiling a small model can render.
MAX_KEYS = 5
MAX_TEXT_CHARS = 600
MAX_BIBTEX_CHARS = 1500

Style = Literal["apa", "ieee", "chicago-author-date", "nature", "vancouver"]


def _clip(text: str, cap: int) -> str:
    return text if len(text) <= cap else text[: cap - 1] + "…"


@tool(empty={"style": None, "styles": [], "references": [], "missing": []}, label="cite_format")
def _format(conn, keys: list[str], style: str | None = None) -> dict:
    from attestation import citation_manager

    try:
        out = citation_manager.format_keys(conn, keys[:MAX_KEYS], style)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    if not out["references"]:
        raise ToolError(
            f"no library reference matches {', '.join(repr(k) for k in out['missing'])}:"
            " pass a key, DOI or arXiv id cite.search or cite.lookup returned"
        )
    for ref in out["references"]:
        ref["text"] = _clip(ref["text"], MAX_TEXT_CHARS)
        # A cut BibTeX entry is not BibTeX: an oversized one is withheld whole.
        if len(ref["bibtex"]) > MAX_BIBTEX_CHARS:
            ref["bibtex"] = None
    n, missing = len(out["references"]), out["missing"]
    out["message"] = f"{n} formatted in {out['style']}" + (
        f"; not in the library: {', '.join(missing)}" if missing else ""
    )
    return out


def register(mcp) -> None:
    """Attach cite.format to the server."""

    @mcp.tool(name="cite.format")
    def cite_format(
        keys: Annotated[
            list[str],
            Field(
                description="citation keys, DOIs or arXiv ids from cite.search or cite.lookup",
                min_length=1,
                max_length=MAX_KEYS,
            ),
        ],
        style: Annotated[
            Style | None,
            Field(description="citation style; default: the operator's (APA unless set)"),
        ] = None,
    ) -> dict:
        """Formatted citations, ready to paste, plus each reference's BibTeX.

        Use when the reader asks for a reference list, "cite this in APA",
        "give me the BibTeX", or a citation in IEEE, Chicago, Nature or
        Vancouver style. Each entry has `text` (the formatted citation),
        `cite_key` (for `\\cite{}`) and `bibtex`. Rendered locally from the
        library with the bundled CSL styles; never queries the network.
        `missing` names any key the library does not hold.
        """
        return _format(keys, style)

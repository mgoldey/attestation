# examples/citations/generate.py
"""Writes references.bib with real BibTeX software, not by hand.

Four well-known works, entered as bibtexparser v2 `Entry`/`Field` records and
serialised with `bibtexparser.write_string()`. The committed file's
formatting -- tab indentation, trailing comma-free field list, brace
quoting -- is the library's writer, the same shape JabRef and Zotero's Better
BibTeX export produce, not something typed to look like it.

bibtexparser 1.x and 2.x have unrelated APIs (1.x is dict-based via
`bibtexparser.bparser`; 2.x is the `Library`/`Entry`/`Field`/`write_string`
API used here). `--with bibtexparser` alone resolves to the latest *stable*
release, which is still 1.x as of this writing and has no `Library` to
import -- `build_library()` below raises `ImportError` under it rather than
writing something that looks right. The v2 API needs the beta pinned
explicitly:

    uv run --with "bibtexparser>=2.0.0b9" --no-project python generate.py

That rewrites `references.bib` in place. Regenerating is deliberate: entries
are the point of this fixture, and a diff after editing them should be
reviewed like any other fixture change.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent

# Real papers, cited accurately. Keys match the `cite=` fields in DRAFT.md:
# three of these resolve there, and a fourth key in DRAFT.md (doe2099imaginary)
# names no entry here on purpose.
ENTRIES = [
    {
        "type": "misc",
        "key": "vaswani2017attention",
        "fields": {
            "title": "Attention Is All You Need",
            "author": (
                "Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, "
                "Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, Lukasz and "
                "Polosukhin, Illia"
            ),
            "year": "2017",
            "doi": "10.48550/arXiv.1706.03762",
            "url": "https://arxiv.org/abs/1706.03762",
        },
    },
    {
        "type": "article",
        "key": "hohenberg1964inhomogeneous",
        "fields": {
            "title": "Inhomogeneous Electron Gas",
            "author": "Hohenberg, Pierre and Kohn, Walter",
            "year": "1964",
            "journal": "Physical Review",
            "volume": "136",
            "pages": "B864--B871",
            "doi": "10.1103/PhysRev.136.B864",
        },
    },
    {
        "type": "misc",
        "key": "kingma2015adam",
        "fields": {
            "title": "Adam: A Method for Stochastic Optimization",
            "author": "Kingma, Diederik P. and Ba, Jimmy",
            "year": "2015",
            "doi": "10.48550/arXiv.1412.6980",
            "url": "https://arxiv.org/abs/1412.6980",
        },
    },
    {
        "type": "misc",
        "key": "hinton2015distilling",
        "fields": {
            "title": "Distilling the Knowledge in a Neural Network",
            "author": "Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff",
            "year": "2015",
            "doi": "10.48550/arXiv.1503.02531",
            "url": "https://arxiv.org/abs/1503.02531",
        },
    },
]


def build_library():
    from bibtexparser import Library
    from bibtexparser.model import Entry, Field

    library = Library()
    for spec in ENTRIES:
        fields = [Field(name, value) for name, value in spec["fields"].items()]
        library.add(Entry(spec["type"], spec["key"], fields))
    return library


def main() -> None:
    import bibtexparser

    library = build_library()
    text = bibtexparser.write_string(library)
    out = HERE / "references.bib"
    out.write_text(text)
    version = bibtexparser.__version__
    print(f"wrote {out.name}: {len(library.entries)} entries (bibtexparser {version})")
    for entry in library.entries:
        print(f"  {entry.key}")


if __name__ == "__main__":
    main()

"""library.py: identity, merge, upsert, sync, search."""

from pathlib import Path

import pytest

from attestation import library

FIX = Path(__file__).parent / "fixtures" / "library"


@pytest.mark.parametrize(
    ("doi", "arxiv", "title", "year", "want"),
    [
        ("10.1038/S41467-026-74391-4", None, "x", 2026, "doi:10.1038/s41467-026-74391-4"),
        ("https://doi.org/10.1000/ABC", None, "x", 2020, "doi:10.1000/abc"),
        ("doi:10.1000/abc", None, "x", 2020, "doi:10.1000/abc"),
        # DOI beats arXiv: a published preprint keeps its arXiv id but gains a DOI.
        ("10.1000/abc", "2106.02347v3", "x", 2021, "doi:10.1000/abc"),
        (None, "2106.02347v3", "x", 2021, "arxiv:2106.02347"),
        (None, "arXiv:2106.02347", "x", 2021, "arxiv:2106.02347"),
        (None, "cond-mat/0301234", "x", 2003, "arxiv:cond-mat/0301234"),
        (
            None,
            None,
            "SchNet: A continuous-filter CNN",
            2017,
            "title:schnet a continuous filter cnn:2017",
        ),
        (None, None, "  Équivariant  Force-Fields! ", None, "title:equivariant force fields:-"),
    ],
)
def test_identity_prefers_doi_then_arxiv_then_title(doi, arxiv, title, year, want):
    assert library.identity(doi, arxiv, title, year) == want


def test_identity_needs_something():
    with pytest.raises(ValueError):
        library.identity(None, None, "", None)


def test_merge_fills_empty_keeps_first_and_records_conflicts():
    existing = {"title": "SchNet", "abstract": None, "year": 2017, "authors": ["Schütt, K."]}
    incoming = {
        "title": "SchNet: a CNN",
        "abstract": "We present...",
        "year": 2018,
        "authors": ["Schütt, K.", "Kindermans, P."],
    }
    merged, conflicts = library.merge(existing, incoming)
    assert merged["abstract"] == "We present..."  # filled
    assert merged["title"] == "SchNet"  # kept
    assert merged["year"] == 2017  # kept
    assert conflicts["title"] == {"kept": "SchNet", "offered": "SchNet: a CNN"}
    assert conflicts["year"] == {"kept": 2017, "offered": 2018}
    # A longer author list EXTENDS rather than conflicts (a .bib truncated with "and others").
    assert merged["authors"] == ["Schütt, K.", "Kindermans, P."]
    assert "authors" not in conflicts


def test_merge_author_disagreement_is_a_conflict():
    merged, conflicts = library.merge({"authors": ["A, B"]}, {"authors": ["C, D"]})
    assert merged["authors"] == ["A, B"]
    assert conflicts["authors"] == {"kept": ["A, B"], "offered": ["C, D"]}

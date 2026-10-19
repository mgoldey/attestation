"""LaTeX parity for the claim checker, coverage, and the citation lint.

Most tests run against `tests/fixtures/latex/`: a three-level paper
(`main.tex` -> `sections/*.tex` -> `tables/results.tex`), its `refs.bib`, a
second `other.bib` it never loads, and a ledger workspace of three runs. The
verdicts and counts are pinned, the way `examples/citations/` pins its own.
"""

import shutil
from pathlib import Path

import pytest

from attestation import claims, latex, ledger, mcp_server
from attestation.cli import main
from attestation.db import get_db

FIXTURE = Path(__file__).parent / "fixtures" / "latex"


@pytest.fixture
def paper(tmp_path, monkeypatch):
    """The fixture paper copied into tmp (stale-ness reads mtimes), its
    ledger scanned, and every configured bibliography source pointed away."""
    shutil.copytree(FIXTURE, tmp_path / "latex")
    empty = tmp_path / "no-bibs"
    empty.mkdir()
    monkeypatch.setenv("ATTEST_BIB_PATHS", str(empty))
    monkeypatch.setenv("ATTEST_ZOTERO_PATH", str(tmp_path / "no-zotero.sqlite"))
    monkeypatch.delenv("ATTEST_BIB_OUT", raising=False)
    monkeypatch.delenv("ATTEST_CITATION_WEB", raising=False)
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, tmp_path / "latex" / "workspace")
    yield conn, tmp_path / "latex" / "paper"
    conn.close()


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# --------------------------------------------------------------------------- comments


@pytest.mark.parametrize(
    ("line", "start"),
    [
        ("plain text", -1),
        ("text % comment", 5),
        (r"12.2\% of runs", -1),  # an escaped percent is a percent sign
        (r"12.2\% of runs % but this is a comment", 15),
        (r"end of row \\% comment after a line break", 13),
        ("% whole line", 0),
    ],
)
def test_comment_start_tells_a_percent_sign_from_a_comment(line, start):
    assert latex.comment_start(line) == start


def test_claim_annotations_read_trailing_and_whole_line_comments():
    text = (
        "& 0.0710 \\\\ % claim: p/r metric=wer value=0.0710\n"
        "%% claim: p/s metric=wer value=0.5\n"
        "5\\% claim: is not a comment\n"
        "% a note that mentions no claim\n"
    )
    assert list(latex.claim_annotations(text)) == [
        (1, "p/r metric=wer value=0.0710"),
        (2, "p/s metric=wer value=0.5"),
    ]


# --------------------------------------------------------------------------- the tree


def test_tree_follows_input_and_include_in_document_order(paper):
    _, root = paper
    files = latex.tree(root / "main.tex")
    rel = [str(p.relative_to(root.resolve())) for p in files]
    # sections/unfinished is \input only inside a comment, and is never read.
    assert rel == ["main.tex", "sections/intro.tex", "sections/results.tex", "tables/results.tex"]


def test_tree_survives_a_cycle_and_a_missing_file(tmp_path):
    write(tmp_path / "a.tex", "\\input{b}\n\\input{nowhere}\n")
    write(tmp_path / "b.tex", "\\input{a.tex}\n")
    names = [p.name for p in latex.tree(tmp_path / "a.tex")]
    assert names == ["a.tex", "b.tex"]


# --------------------------------------------------------------------------- claims


def test_the_five_verdicts_on_a_multi_file_paper(paper):
    conn, root = paper
    out = claims.check(conn, root / "main.tex")

    got = {(Path(v.claim.path).name, v.claim.line): v.verdict for v in out["verdicts"]}
    assert got == {
        ("main.tex", 10): "supported",  # the \newcommand that defines \bestwer
        ("results.tex", 4): "stale",  # sections/results.tex; as_of=2020-01-01
        ("results.tex", 5): "ambiguous",  # kdsweep_* matches t2 and t4
        ("results.tex", 8): "contradicted",  # tables/results.tex: 0.0710 vs 0.0701
        ("results.tex", 10): "unsupported",  # no kdsweep_t8 run
    }
    assert out["malformed"] == []
    # Every finding names the file it came from, not the root.
    paths = {Path(v.claim.path).relative_to(root.resolve()).as_posix() for v in out["verdicts"]}
    assert paths == {"main.tex", "sections/results.tex", "tables/results.tex"}


def test_a_malformed_latex_claim_is_reported_not_skipped(tmp_path):
    doc = write(tmp_path / "p.tex", "x % claim: p/r value=0.5\n")
    found, problems = claims.parse_file(doc)
    assert found == []
    assert problems == [f"{doc}:1: missing metric"]


def test_a_folder_scan_reads_markdown_and_latex(tmp_path):
    write(tmp_path / "notes.md", "<!-- claim: p/a metric=wer value=0.1 -->\n")
    write(tmp_path / "paper" / "main.tex", "0.2 % claim: p/b metric=wer value=0.2\n")
    found, _ = claims.find_claims(tmp_path)
    assert sorted(c.run for c in found) == ["a", "b"]


def test_cli_claims_on_a_tex_exits_nonzero_on_the_contradiction(paper, capsys):
    _, root = paper
    rc = main(["claims", str(root / "main.tex")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "contradicted" in out
    assert "5 claim(s)" in out


def test_mcp_claims_check_accepts_a_tex_path(paper):
    _, root = paper
    out = mcp_server._claims_check_impl(str(root / "main.tex"))
    assert out["ok"] is True
    assert out["counts"] == {
        "supported": 1,
        "stale": 1,
        "ambiguous": 1,
        "contradicted": 1,
        "unsupported": 1,
    }
    # cite=hinton2015distilling on the \bestwer claim resolves through the
    # refs.bib beside the paper, so there is no `uncited` verdict.
    assert out["checked"] == ["numeric", "citation"]


# --------------------------------------------------------------------------- coverage


def test_coverage_of_the_paper_is_pinned(paper):
    _, root = paper
    cov = claims.coverage(root / "main.tex")

    assert cov["files"] == 4
    assert cov["numbers"] == 11
    uncovered = {(Path(u["file"]).name, u["line"], u["value"]) for u in cov["uncovered"]}
    assert uncovered == {
        ("intro.tex", 5, 1234.5),  # 1{,}234.5 -- the LaTeX thousands separator
        ("intro.tex", 9, 3.5),  # 3.5\% -- the trailing comment's 9.99 is not prose
        ("results.tex", 6, 12.2),  # \SI{12.2}{\percent}
        ("results.tex", 8, 0.023),  # table cells
        ("results.tex", 9, 0.021),  # $0.021$: a number alone in math is a result
        ("results.tex", 10, 0.022),
    }


def test_a_claim_in_main_covers_a_number_in_an_input_file(paper):
    """One paper is one document: \\num{0.0642} in sections/results.tex is
    covered by the claim on main.tex's \\newcommand."""
    _, root = paper
    cov = claims.coverage(root / "main.tex")
    assert 0.0642 not in {u["value"] for u in cov["uncovered"]}


@pytest.mark.parametrize(
    "line",
    [
        "$\\alpha = 0.5$ is a parameter",
        "\\begin{equation} x = 0.5 y \\end{equation}",
        "\\[ y = 0.25 x \\]",
        "\\includegraphics[width=0.48\\textwidth, angle=1.5]{fig.pdf}",
        "\\vspace{0.5cm} and \\\\[2.5pt]",
        "see Table~\\ref{tab:1.2} and \\label{eq:0.5}",
        "\\citep[p.~3.4]{a}",
        "\\usepackage[margin=1.5in]{geometry}",
        "% 0.5 inside a comment",
        "\\renewcommand{\\arraystretch}{1.2}",
    ],
)
def test_coverage_ignores_structural_latex_numbers(tmp_path, line):
    doc = write(tmp_path / "p.tex", line + "\n")
    assert claims.coverage(doc)["numbers"] == 0


@pytest.mark.parametrize(
    ("line", "value"),
    [
        ("WER fell to 0.053.", 0.053),
        ("\\num{0.0642}", 0.0642),
        ("\\num[round-precision=2]{1.2e-3}", 1.2e-3),
        ("\\SI{12.2}{\\percent}", 12.2),
        ("\\qty{3.25}{\\second}", 3.25),
        ("$0.91$", 0.91),
        ("$12.2\\%$", 12.2),
        ("\\(0.4\\)", 0.4),
        ("cost 1{,}234.5 hours", 1234.5),
        ("& \\textbf{0.0701} \\\\", 0.0701),
    ],
)
def test_coverage_reads_latex_number_forms(tmp_path, line, value):
    doc = write(tmp_path / "p.tex", line + "\n")
    cov = claims.coverage(doc)
    assert [u["value"] for u in cov["uncovered"]] == [pytest.approx(value)]


# --------------------------------------------------------------------------- citations


def test_cited_keys_reads_every_cite_form():
    text = (
        "\\cite{a} \\citep[see][p.~3]{b, c} \\citet*{d} \\citealp{e}\n"
        "\\parencite{f} \\textcite[ch.~2]{g} \\autocite{h} \\footcite{i}\n"
        "\\cites{j}[p.~1]{k} \\nocite{*}\n"
        "\\setcitestyle{numbers} \\citestyle{plain}\n"
        "% \\cite{commented}\n"
        "5\\% of \\cite{l}\n"
    )
    keys = [k for k, _ in latex.cited_keys(text)]
    assert keys == list("abcdefghijkl")
    assert dict(latex.cited_keys(text))["l"] == 6


def test_cite_check_on_the_paper_names_the_unloaded_file(paper):
    _, root = paper
    out = latex.check_citations(root / "main.tex")

    assert out["citation_keys_used"] == 7
    assert out["bib_files"] == ["refs.bib"]
    missing = {m["key"]: m for m in out["missing_keys"]}
    assert set(missing) == {"lecun1998gradient", "missing2020ghost"}
    assert missing["lecun1998gradient"]["defined_in_unloaded"] == "other.bib"
    assert missing["lecun1998gradient"]["where"] == "sections/intro.tex:11"
    assert "defined_in_unloaded" not in missing["missing2020ghost"]
    assert out["unresolved_bib_resources"] == []


def test_without_a_named_bibliography_any_bib_beside_the_paper_counts(tmp_path):
    write(tmp_path / "a.bib", "@misc{one,\n  title = {One}\n}\n")
    write(tmp_path / "b.bib", "@misc{two,\n  title = {Two}\n}\n")
    doc = write(tmp_path / "p.tex", "\\documentclass{article}\n\\cite{one,two,three}\n")
    out = latex.check_citations(doc, extra_bibs=[])
    assert [m["key"] for m in out["missing_keys"]] == ["three"]
    assert out["bib_files"] == ["a.bib", "b.bib"]


def test_a_named_bib_that_does_not_exist_is_reported(tmp_path):
    doc = write(tmp_path / "p.tex", "\\cite{x}\n\\addbibresource{refs.bib}\n")
    out = latex.check_citations(doc, extra_bibs=[])
    assert out["unresolved_bib_resources"] == ["refs.bib"]
    assert [m["key"] for m in out["missing_keys"]] == ["x"]


def test_lookup_resolves_only_when_the_paper_names_no_bib(tmp_path):
    """A key in Zotero or the library store would build as [?] once the paper
    names its .bib files -- so it resolves there only without them, and is
    otherwise reported as defined somewhere the paper does not load."""

    class Ref:
        source = "zotero"

    def lookup(key):
        return Ref() if key == "inzotero" else None

    loose = write(tmp_path / "loose" / "p.tex", "\\cite{inzotero}\n")
    assert latex.check_citations(loose, extra_bibs=[], lookup=lookup)["missing_keys"] == []

    write(tmp_path / "named" / "refs.bib", "@misc{other,\n  title = {O}\n}\n")
    named = write(tmp_path / "named" / "p.tex", "\\cite{inzotero}\n\\bibliography{refs}\n")
    (m,) = latex.check_citations(named, extra_bibs=[], lookup=lookup)["missing_keys"]
    assert m["defined_in_unloaded"] == "zotero"


def test_the_owner_bib_counts_when_loaded_or_when_nothing_is_named(tmp_path, monkeypatch):
    owner = tmp_path / "bibliography" / "attestation"
    write(owner / "Matthew-Goldey.bib", "@article{saved2024,\n  title = {Saved}\n}\n")
    monkeypatch.setenv("ATTEST_BIB_OUT", str(owner))
    monkeypatch.setenv("ATTEST_BIB_PATHS", str(tmp_path / "none"))

    loose = write(tmp_path / "a" / "p.tex", "\\cite{saved2024}\n")
    assert latex.check_citations(loose)["missing_keys"] == []

    named = write(tmp_path / "b" / "p.tex", "\\cite{saved2024}\n\\bibliography{Matthew-Goldey}\n")
    assert latex.check_citations(named)["missing_keys"] == []

    write(tmp_path / "c" / "refs.bib", "@misc{x,\n  title = {X}\n}\n")
    other = write(tmp_path / "c" / "p.tex", "\\cite{saved2024}\n\\bibliography{refs}\n")
    (m,) = latex.check_citations(other)["missing_keys"]
    assert m["defined_in_unloaded"] == "Matthew-Goldey.bib"


def test_a_markdown_target_has_no_latex_citation_report(tmp_path):
    write(tmp_path / "doc.md", "\\cite{not-latex}\n")
    out = latex.check_citations(tmp_path)
    assert out["papers"] == [] and out["missing_keys"] == []


def test_a_folder_checks_each_documentclass_file_as_a_paper(paper):
    _, root = paper
    out = latex.check_citations(root)
    # main.tex only: the sections are not papers of their own.
    assert [Path(p).name for p in out["papers"]] == ["main.tex"]


def test_mcp_cite_check_accepts_a_tex_path(paper):
    _, root = paper
    from attestation.mcp.citation import _check

    out = _check(path=str(root / "main.tex"))
    assert out["ok"] is True
    assert {m["key"] for m in out["missing_keys"]} == {"lecun1998gradient", "missing2020ghost"}
    assert out["uncited"] == []  # cite=hinton2015distilling resolves via refs.bib
    assert out["citation_keys_used"] == 7
    assert "CITATION KEYS only" in out["message"]
    assert "runs.claims_check" in out["message"]
    assert out["checked"] == ["citation"]


def test_cli_citations_exits_nonzero_on_a_missing_key(paper, capsys):
    _, root = paper
    rc = main(["claims", "--citations", str(root / "main.tex")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "lecun1998gradient" in out and "other.bib" in out
    assert "missing2020ghost" in out


def test_the_annotate_skills_latex_examples_parse(tmp_path):
    """attestation-annotate teaches the `% claim:` form by example; an example
    that does not parse would teach an annotation the checker skips."""
    from importlib.resources import files

    text = files("attestation").joinpath("skills/attestation-annotate/SKILL.md").read_text()
    lines = [ln for ln in text.splitlines() if "% claim:" in ln and "<project>" not in ln]
    assert len(lines) == 2, lines
    doc = write(tmp_path / "example.tex", "\n".join(lines) + "\n")
    found, problems = claims.parse_file(doc)
    assert problems == []
    assert [(c.run, c.value) for c in found] == [("kdsweep_t4", 0.053)] * 2

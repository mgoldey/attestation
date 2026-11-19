"""Figure provenance: every included figure is linked and fresh, or reported.

Design: docs/superpowers/specs/2026-11-19-figure-provenance-design.md. Fixtures
build real files in tmp_path and control mtimes with os.utime, because the
whole check is "which file is newer".
"""

import hashlib
import os
import socket

import pytest

from attestation import figures
from attestation.figures import FigureVerdict as V

OLD, MID, NEW = 1_700_000_000, 1_700_100_000, 1_700_200_000


def touch(path, text="x", when=MID):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, (when, when))
    return path


def verdicts(report):
    """{figure as written: verdict}."""
    return {f.path: f.verdict for f in report["figures"]}


def _markdown_draft(tmp_path):
    """One of each outcome: fresh, stale (mtime), stale (sha256), orphan,
    missing source, and a malformed annotation (whose figure is then orphan)."""
    touch(tmp_path / "results" / "a.json", when=OLD)
    touch(tmp_path / "results" / "new.json", when=NEW)
    for name in "abcdef":
        touch(tmp_path / "figs" / f"{name}.png", text=f"bytes-{name}", when=MID)
    wrong = "0" * 64
    right = hashlib.sha256(b"bytes-a").hexdigest()
    draft = tmp_path / "draft.md"
    draft.write_text(
        "![a](figs/a.png)\n"
        f"<!-- figure: figs/a.png from=results/a.json sha256={right} -->\n"
        "![b](figs/b.png)\n"
        "<!-- figure: figs/b.png from=results/new.json -->\n"
        "![c](figs/c.png)\n"
        f"<!-- figure: figs/c.png from=results/a.json sha256={wrong} -->\n"
        "![d](figs/d.png)\n"
        "![e](figs/e.png)\n"
        "<!-- figure: figs/e.png from=results/nope.json -->\n"
        "![f](figs/f.png)\n"
        "<!-- figure: figs/f.png frm=results/a.json -->\n"
    )
    return draft


def test_markdown_every_outcome_has_exact_counts(tmp_path):
    report = figures.check(_markdown_draft(tmp_path))

    assert verdicts(report) == {
        "figs/a.png": V.FRESH,
        "figs/b.png": V.STALE,  # a source is newer than the figure
        "figs/c.png": V.STALE,  # sha256 no longer matches
        "figs/d.png": V.ORPHAN,
        "figs/e.png": V.MISSING,  # a from= file is absent
        "figs/f.png": V.ORPHAN,  # its annotation is malformed, so it never attached
    }
    assert dict(report["counts"]) == {"fresh": 1, "stale": 2, "orphan": 2, "missing": 1}
    assert len(report["malformed"]) == 1
    assert "frm=" in report["malformed"][0]
    messages = {f.path: f.message for f in report["figures"]}
    assert "new.json changed after the figure" in messages["figs/b.png"]
    assert "sha256" in messages["figs/c.png"]
    assert "results/nope.json" in messages["figs/e.png"]


def test_latex_every_outcome_has_exact_counts(tmp_path):
    touch(tmp_path / "results" / "a.json", when=OLD)
    touch(tmp_path / "results" / "new.json", when=NEW)
    touch(tmp_path / "figs" / "a.pdf", text="bytes-a")
    touch(tmp_path / "figs" / "b.png", text="bytes-b")
    touch(tmp_path / "figs" / "c.png", text="bytes-c")
    touch(tmp_path / "figs" / "d.png", text="bytes-d")
    wrong = "f" * 64
    paper = tmp_path / "main.tex"
    paper.write_text(
        "\\documentclass{article}\n\\begin{document}\n"
        "\\includegraphics[width=\\linewidth]{figs/a}\n"  # extensionless -> a.pdf
        "% figure: figs/a.pdf from=results/a.json\n"
        "\\includegraphics{figs/b.png}\n"
        "% figure: figs/b.png from=results/new.json\n"
        "\\includegraphics{figs/c.png}\n"
        f"% figure: figs/c.png sha256={wrong}\n"
        "\\includegraphics{figs/d.png}\n"
        "\\includegraphics{figs/gone}\n"
        "% figure: figs/gone.png from=results/a.json\n"
        "% figure: figs/d.png\n"  # links nothing: malformed
        "\\end{document}\n"
    )

    report = figures.check(paper)

    assert verdicts(report) == {
        "figs/a": V.FRESH,
        "figs/b.png": V.STALE,
        "figs/c.png": V.STALE,
        "figs/d.png": V.ORPHAN,
        "figs/gone": V.MISSING,
    }
    assert len(report["malformed"]) == 1
    assert "links nothing" in report["malformed"][0]


def test_the_stale_case_is_the_mtime_comparison(tmp_path):
    """Mutation target: drop the mtime comparison and this fails."""
    touch(tmp_path / "src.json", when=NEW)
    touch(tmp_path / "f.png", when=MID)
    doc = touch(tmp_path / "d.md", "![f](f.png)\n<!-- figure: f.png from=src.json -->\n")
    assert verdicts(figures.check(doc)) == {"f.png": V.STALE}
    os.utime(tmp_path / "src.json", (OLD, OLD))
    assert verdicts(figures.check(doc)) == {"f.png": V.FRESH}
    # equal mtimes are fresh: "newer" is strict
    os.utime(tmp_path / "src.json", (MID, MID))
    assert verdicts(figures.check(doc)) == {"f.png": V.FRESH}


def test_a_changed_sha256_is_stale_even_when_mtimes_agree(tmp_path):
    """Mutation target: drop the sha256 comparison and this fails."""
    touch(tmp_path / "src.json", when=OLD)
    fig = touch(tmp_path / "f.png", "v1")
    pin = hashlib.sha256(b"v1").hexdigest().upper()  # case-insensitive
    doc = touch(tmp_path / "d.md", f"![f](f.png)\n<!-- figure: f.png sha256={pin} -->\n")
    assert verdicts(figures.check(doc)) == {"f.png": V.FRESH}
    touch(fig, "v2")
    assert verdicts(figures.check(doc)) == {"f.png": V.STALE}


def test_missing_wins_over_stale_and_a_missing_figure_over_orphan(tmp_path):
    touch(tmp_path / "src.json", when=NEW)
    touch(tmp_path / "f.png", when=OLD)
    doc = touch(
        tmp_path / "d.md",
        "![f](f.png)\n<!-- figure: f.png from=src.json,gone.json -->\n![g](g.png)\n",
    )
    assert verdicts(figures.check(doc)) == {"f.png": V.MISSING, "g.png": V.MISSING}


# ----------------------------------------------------------------- what is not a figure


def test_commented_and_coded_figures_are_not_found(tmp_path):
    """A comment renders as nothing and asserts nothing (the coverage rule).
    Mutation target: drop the masking and this fails."""
    touch(tmp_path / "x.png")
    md = touch(
        tmp_path / "d.md",
        "<!-- ![hidden](x.png) -->\n"
        '<!--\n<img src="x.png">\n-->\n'
        "`![inline](x.png)`\n"
        "```\n![fenced](x.png)\n<!-- figure: x.png from=y.json -->\n```\n",
    )
    tex = touch(tmp_path / "p.tex", "% \\includegraphics{x.png}\ntext 50\\%\n")
    assert figures.check(md)["total"] == 0
    assert figures.check(tex)["total"] == 0


def test_an_annotation_inside_a_code_fence_is_not_an_annotation(tmp_path):
    touch(tmp_path / "f.png")
    doc = touch(
        tmp_path / "d.md",
        "![f](f.png)\n```\n<!-- figure: f.png from=nope.json -->\n```\n",
    )
    assert verdicts(figures.check(doc)) == {"f.png": V.ORPHAN}


def test_html_img_and_angle_bracket_markdown_paths_are_found(tmp_path):
    touch(tmp_path / "a b.png")
    touch(tmp_path / "c.png")
    doc = touch(tmp_path / "d.md", "![x](<a b.png> \"t\")\n<img alt='k'\n  src='c.png'>\n")
    assert set(verdicts(figures.check(doc))) == {"a b.png", "c.png"}


def test_input_tree_figures_are_found_once_even_on_a_cycle(tmp_path):
    """Mutation target: walk includes without latex.tree's guard and this loops
    or double counts."""
    touch(tmp_path / "f.png")
    (tmp_path / "main.tex").write_text(
        "\\documentclass{article}\n\\input{a}\n\\includegraphics{f.png}\n"
    )
    (tmp_path / "a.tex").write_text("\\input{main}\n\\includegraphics{f.png}\n")
    report = figures.check(tmp_path / "main.tex")
    assert report["total"] == 1
    # also by folder: the paper is one unit, a.tex is not judged alone
    assert figures.check(tmp_path)["total"] == 1


def test_an_input_files_figure_is_found_through_the_root(tmp_path):
    """The figure sits only in the \\input-ed file: dropping the tree walk loses it."""
    touch(tmp_path / "figs" / "f.png")
    (tmp_path / "main.tex").write_text("\\documentclass{article}\n\\input{chapters/one}\n")
    (tmp_path / "chapters").mkdir()
    (tmp_path / "chapters" / "one.tex").write_text("\\includegraphics{figs/f.png}\n")
    assert verdicts(figures.check(tmp_path / "main.tex")) == {"figs/f.png": V.ORPHAN}
    # by folder, the chapter resolves from the paper's folder, not its own
    assert verdicts(figures.check(tmp_path)) == {"figs/f.png": V.ORPHAN}


def test_graphicspath_is_searched(tmp_path):
    touch(tmp_path / "img" / "f.png")
    paper = touch(tmp_path / "p.tex", "\\graphicspath{{img/}}\n\\includegraphics{f}\n")
    assert verdicts(figures.check(paper)) == {"f": V.ORPHAN}


def test_remote_and_macro_paths_are_skipped_and_never_fetched(tmp_path, monkeypatch):
    def no_socket(*a, **k):
        raise AssertionError("figures must never open a socket")

    monkeypatch.setattr(socket, "socket", no_socket)
    monkeypatch.setattr(socket, "create_connection", no_socket)
    md = touch(
        tmp_path / "d.md",
        '![r](https://example.com/a.png)\n<img src="//cdn.example.com/b.png">\n'
        "![d](data:image/png;base64,AAAA)\n",
    )
    tex = touch(
        tmp_path / "p.tex",
        "\\includegraphics{http://example.com/c.png}\n\\includegraphics{\\figdir/x.png}\n",
    )
    md_report, tex_report = figures.check(md), figures.check(tex)
    assert (md_report["total"], md_report["skipped"]) == (0, 3)
    assert (tex_report["total"], tex_report["skipped"]) == (0, 2)


def test_annotation_problems_are_reported_not_skipped(tmp_path):
    touch(tmp_path / "f.png")
    doc = touch(
        tmp_path / "d.md",
        "![f](f.png)\n"
        "<!-- figure: from=a.json -->\n"
        "<!-- figure: f.png sha256=abc -->\n"
        "<!-- figure: f.png from=a.json b.json -->\n"
        "<!-- figure: f.png from=a.json --><!-- figure: f.png from=b.json -->\n",
    )
    report = figures.check(doc)
    text = "\n".join(report["malformed"])
    assert len(report["malformed"]) == 4
    assert "expected a figure path first" in text
    assert "64 hex" in text
    assert "unexpected 'b.json'" in text
    assert "duplicate annotation" in text


def test_same_file_included_in_two_documents_is_one_finding(tmp_path):
    touch(tmp_path / "f.png")
    touch(tmp_path / "s.json", when=OLD)
    touch(tmp_path / "a.md", "![f](f.png)\n")
    touch(tmp_path / "b.md", "![f](f.png)\n<!-- figure: f.png from=s.json -->\n")
    report = figures.check(tmp_path)
    assert [f.verdict for f in report["figures"]] == [V.FRESH]


def test_hidden_folders_below_the_root_are_not_scanned(tmp_path):
    touch(tmp_path / ".venv" / "x.md", "![f](f.png)\n")
    touch(tmp_path / "ok.md", "no figures\n")
    assert figures.check(tmp_path)["total"] == 0


# ----------------------------------------------------------------- gate and block


def test_failing_counts_stale_missing_and_malformed_but_not_orphans(tmp_path):
    touch(tmp_path / "f.png")
    orphan = touch(tmp_path / "o.md", "![f](f.png)\n")
    assert figures.failing(figures.check(orphan)) == 0
    assert figures.failing(figures.check(_markdown_draft(tmp_path / "x"))) == 2 + 1 + 1


def test_message_suffix_names_only_what_fails(tmp_path):
    clean = figures.check(touch(tmp_path / "o.md", "no figures\n"))
    assert figures.message_suffix(clean) == ""
    assert figures.message_suffix(figures.check(_markdown_draft(tmp_path / "x"))) == (
        "; figures: 2 stale, 1 missing"
    )


def test_block_is_bounded_and_keeps_complete_counts(tmp_path):
    touch(tmp_path / "s.json", when=NEW)
    lines = []
    for i in range(50):
        touch(tmp_path / "figs" / f"f{i}.png", when=OLD)
        lines.append(f"![f](figs/f{i}.png)\n<!-- figure: figs/f{i}.png from=s.json -->\n")
    doc = touch(tmp_path / "d.md", "".join(lines))
    out = figures.block(figures.check(doc))
    assert out["total"] == 50 and out["counts"] == {"stale": 50}
    assert len(out["findings"]) == figures.MAX_FIGURE_FINDINGS
    assert out["truncated"] == 50 - figures.MAX_FIGURE_FINDINGS
    assert out["note"] == figures.NOTE


def test_empty_block_has_the_same_keys_as_a_populated_one(tmp_path):
    doc = touch(tmp_path / "d.md", "no figures\n")
    assert figures.block(figures.check(doc)) == figures.empty_block()
    populated = figures.block(figures.check(_markdown_draft(tmp_path / "x")))
    assert set(figures.empty_block()) == set(populated)


# ----------------------------------------------------------------- CLI


def run_cli(*argv, capsys):
    from attestation import cli

    code = cli.main(list(argv))
    return code, capsys.readouterr().out


def test_cli_exit_codes_and_output(tmp_path, capsys):
    bad = _markdown_draft(tmp_path / "bad")
    code, out = run_cli("claims", "--figures", str(bad), capsys=capsys)
    assert code == 1
    assert "6 figure(s): 1 fresh, 1 missing, 2 orphan, 2 stale" in out
    assert "fresh means no declared source is newer" in out
    assert "malformed" in out

    touch(tmp_path / "ok" / "f.png")
    ok = touch(tmp_path / "ok" / "d.md", "![f](f.png)\n")
    code, out = run_cli("claims", "--figures", str(ok), capsys=capsys)
    assert code == 0, "orphans are a lint and must not fail the gate"
    assert "orphan" in out

    none = touch(tmp_path / "none.md", "text\n")
    code, out = run_cli("claims", "--figures", str(none), capsys=capsys)
    assert code == 0 and "no figures found" in out


def test_cli_verdict_filter_applies_to_figures(tmp_path, capsys):
    draft = _markdown_draft(tmp_path)
    _, out = run_cli("claims", "--figures", "--verdict", "missing", str(draft), capsys=capsys)
    listed = [
        line
        for line in out.splitlines()
        if line.startswith("  ") and ".png" in line and "<!--" not in line
    ]
    assert [line.split()[0] for line in listed] == ["missing"]
    assert "e.png" in listed[0]


# ----------------------------------------------------------------- MCP


@pytest.fixture
def mcp_env(tmp_path, monkeypatch):
    from attestation.db import get_db

    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    get_db(tmp_path / "t.db").close()
    monkeypatch.delenv("ATTEST_CITATION_WEB", raising=False)
    return tmp_path


def test_claims_check_carries_the_figures_block(mcp_env):
    from attestation.mcp.claims_tools import _check

    draft = _markdown_draft(mcp_env / "w")
    out = _check(path=str(draft))
    assert out["ok"] is True
    assert out["figures"]["total"] == 6
    assert out["figures"]["counts"] == {"fresh": 1, "stale": 2, "orphan": 2, "missing": 1}
    assert out["message"].endswith("; figures: 2 stale, 1 missing")
    worst = out["figures"]["findings"][0]
    assert worst["verdict"] == "missing" and worst["figure"] == "figs/e.png"
    assert worst["file"] == "draft.md"


def test_a_draft_with_no_figures_keeps_its_envelope(mcp_env):
    """Same keys as before plus an EMPTY `figures`; the message is untouched."""
    from attestation.mcp.claims_tools import _check

    doc = mcp_env / "d.md"
    doc.write_text("Plain.\n<!-- claim: proj/missing metric=wer value=0.1 -->\n")
    out = _check(path=str(doc))
    assert set(out) == {"ok", "message", "claims", "counts", "malformed", "checked", "figures"}
    assert out["figures"] == figures.empty_block()
    assert out["message"] == "1 claim(s): 1 unsupported"
    assert out["counts"] == {"unsupported": 1}


def test_the_failure_envelope_has_the_same_figures_key(mcp_env):
    from attestation.mcp.claims_tools import _check

    out = _check(path=str(mcp_env / "nope.md"))
    assert out["ok"] is False and out["figures"] == figures.empty_block()


def test_the_workspace_example_still_reports_its_pinned_line(tmp_path, monkeypatch):
    """examples/workspace has no figures: README and test_golden_paths pin its
    summary, and a figure-less draft must not change it."""
    import shutil
    from pathlib import Path

    from attestation.mcp import provenance
    from attestation.mcp.claims_tools import _check

    ws = tmp_path / "ws"
    shutil.copytree(Path(__file__).parent.parent / "examples" / "workspace", ws)
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    monkeypatch.setenv("RESEARCH_ROOT", str(ws))
    provenance._scan(confirm=True)
    out = _check()
    assert out["message"] == "7 claim(s): 1 contradicted, 5 supported, 1 unsupported"
    assert out["figures"] == figures.empty_block()


def test_a_stale_sighting_is_not_hidden_by_a_fresh_one_elsewhere(tmp_path):
    """Two documents annotate one figure; the one whose pin no longer matches
    must win, or `attest claims --figures` exits 0 on a stale figure."""
    touch(tmp_path / "f.png", "v1")
    good = hashlib.sha256(b"v1").hexdigest()
    touch(tmp_path / "a.md", f"![f](f.png)\n<!-- figure: f.png sha256={good} -->\n")
    touch(tmp_path / "b.md", f"![f](f.png)\n<!-- figure: f.png sha256={'0' * 64} -->\n")
    report = figures.check(tmp_path)
    assert [f.verdict for f in report["figures"]] == [V.STALE]
    assert figures.failing(report) == 1


def test_an_unreadable_figure_does_not_cost_the_claim_verdicts(mcp_env, monkeypatch):
    from attestation.mcp.claims_tools import _check

    draft = _markdown_draft(mcp_env / "w")

    def unreadable(_root):
        raise PermissionError("figs/a.png")

    monkeypatch.setattr(figures, "check", unreadable)
    out = _check(path=str(draft))
    assert out["ok"] is True
    assert set(out["figures"]) == set(figures.empty_block())
    assert "figures not checked" in out["figures"]["note"]
    assert out["message"].endswith("; figures not checked")

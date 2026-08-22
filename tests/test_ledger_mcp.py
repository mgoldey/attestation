"""MCP wiring for the run ledger."""

import json

import pytest

from attestation import mcp_server
from attestation.db import get_db


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    get_db(tmp_path / "t.db").close()
    root = tmp_path / "ws"
    write(root / "proj" / "configs" / "sweep_a.yaml", "# hypothesis: a\nmodel:\n  x: 1\n")
    write(root / "proj" / "results" / "eval_step_100.json", json.dumps({"wer": 0.4}))
    write(root / "proj" / "results" / "eval_step_200.json", json.dumps({"wer": 0.2}))
    write(root / "proj" / "results" / "energy_adz.json", json.dumps({"case": {"rhf": -152.0}}))
    write(root / "proj" / "results" / "energy_atz.json", json.dumps({"case": {"rhf": -152.1}}))
    monkeypatch.setenv("RESEARCH_ROOT", str(root))
    return root


def test_scan_without_confirm_mutates_nothing(workspace):
    out = mcp_server._runs_scan_impl()

    assert out["ok"] is False
    assert "confirm=true" in out["message"]
    assert out["scanned"] == {}
    assert mcp_server._runs_list_impl()["runs"] == []


def test_scan_reads_the_workspace_from_the_environment(workspace):
    out = mcp_server._runs_scan_impl(confirm=True)

    assert out["ok"] is True
    assert out["scanned"] == {"proj": 5}


def test_scan_tells_the_caller_why_a_project_was_empty(tmp_path, monkeypatch):
    """The caller is a model. A bare "0 run(s)" leaves it nothing to relay or
    act on, so the reason each project came back empty has to cross the MCP
    boundary too -- not just print in the CLI."""
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    get_db(tmp_path / "t.db").close()
    root = tmp_path / "ws"
    # an ordinary layout the conventions do not recognise: results sit in a
    # per-arm directory rather than results/
    write(root / "asr" / "baseline" / "results.json", json.dumps({"wer": 0.05}))
    monkeypatch.setenv("RESEARCH_ROOT", str(root))

    out = mcp_server._runs_scan_impl(confirm=True)

    assert out["scanned"] == {}
    assert "asr" in out["diagnostics"]
    assert "results/" in out["diagnostics"]["asr"]


def test_scan_without_a_configured_root_says_so(monkeypatch, tmp_path):
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    monkeypatch.delenv("RESEARCH_ROOT", raising=False)

    out = mcp_server._runs_scan_impl(confirm=True)

    assert out["ok"] is False
    assert "RESEARCH_ROOT" in out["message"]


def test_list_before_any_scan_directs_the_caller(monkeypatch, tmp_path):
    monkeypatch.setenv("RSS_DB", str(tmp_path / "empty.db"))
    get_db(tmp_path / "empty.db").close()

    out = mcp_server._runs_list_impl()

    assert out["ok"] is False
    assert "runs.scan" in out["message"]
    assert out["runs"] == [] and out["families"] == []


def test_compare_ranks_and_names_a_winner(workspace):
    mcp_server._runs_scan_impl(confirm=True)

    out = mcp_server._runs_compare_impl("eval", metric="wer")

    assert out["ok"] is True
    assert out["winner"] == "eval_step_200"  # 0.2 beats 0.4, lower is better


def test_compare_surfaces_an_undeclared_direction_not_internal_error(workspace):
    """A caller-fixable problem must say what to fix. Flattening it to
    "internal error" would hide the one thing that resolves it."""
    mcp_server._runs_scan_impl(confirm=True)

    out = mcp_server._runs_compare_impl("energy", metric="rhf")

    assert out["ok"] is False
    assert "unknown direction" in out["message"]
    assert out["arms"] == []


def test_detail_returns_the_config_header(workspace):
    mcp_server._runs_scan_impl(confirm=True)

    out = mcp_server._runs_detail_impl("proj", "sweep_a")

    assert out["ok"] is True
    assert "hypothesis: a" in out["run"]["notes"]


def test_detail_for_a_missing_run_preserves_success_keys(workspace):
    mcp_server._runs_scan_impl(confirm=True)

    out = mcp_server._runs_detail_impl("proj", "nope")

    assert out["ok"] is False
    assert out["run"] is None


def test_all_four_tools_are_served():
    names = {t.name for t in mcp_server.mcp._tool_manager.list_tools()}

    assert {"runs.scan", "runs.list", "runs.compare", "runs.detail"} <= names


def test_claims_check_reports_each_verdict(workspace, tmp_path):
    mcp_server._runs_scan_impl(confirm=True)
    doc = tmp_path / "R.md"
    doc.write_text(
        "<!-- claim: proj/eval_step_100 metric=wer value=0.4 -->\n"
        "<!-- claim: proj/eval_step_100 metric=wer value=0.9 -->\n"
        "<!-- claim: proj/missing metric=wer value=0.1 -->\n"
    )

    out = mcp_server._claims_check_impl(str(doc))

    assert out["ok"] is True
    assert out["counts"] == {"supported": 1, "contradicted": 1, "unsupported": 1}


def test_claims_check_filters_by_verdict(workspace, tmp_path):
    """The question a writer actually asks: what in my docs is unsupported?"""
    mcp_server._runs_scan_impl(confirm=True)
    doc = tmp_path / "R.md"
    doc.write_text(
        "<!-- claim: proj/eval_step_100 metric=wer value=0.4 -->\n"
        "<!-- claim: proj/missing metric=wer value=0.1 -->\n"
    )

    out = mcp_server._claims_check_impl(str(doc), verdict="unsupported")

    assert [c["verdict"] for c in out["claims"]] == ["unsupported"]
    assert out["counts"]["supported"] == 1, "counts still describe every claim"


def test_claims_check_without_a_path_or_root_says_so(monkeypatch, tmp_path):
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    get_db(tmp_path / "t.db").close()
    monkeypatch.delenv("RESEARCH_ROOT", raising=False)

    out = mcp_server._claims_check_impl()

    assert out["ok"] is False
    assert "RESEARCH_ROOT" in out["message"]


def test_runs_list_does_not_bury_the_runs_under_a_family_dump(tmp_path, monkeypatch):
    """`families` exists so a caller can find a name for runs.compare.

    Fifty of 403 is neither a complete list nor a useful sample -- it was 73%
    of the response (3,030 of 4,151 chars) and an agent could not act on any
    of it. A UI agent asking "which arm won?" got a wall of family names ahead
    of the runs it asked for.

    Show a few, say how many there are, and tell the caller how to narrow.
    """
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    conn = get_db(tmp_path / "t.db")
    for i in range(60):
        conn.execute(
            "INSERT INTO runs(project, name, family, status, source_path)"
            " VALUES (?, ?, ?, 'recorded', ?)",
            (f"proj{i % 3}", f"fam{i}_arm", f"fam{i}", f"/tmp/r{i}.json"),
        )
    conn.commit()
    conn.close()

    out = mcp_server._runs_list_impl(limit=5)

    assert out["ok"] is True
    assert len(out["runs"]) == 5
    assert len(out["families"]) <= 12, (
        f"{len(out['families'])} families returned; the caller cannot use a dump"
    )
    assert out["n_families"] == 60, "the true count must survive truncation"
    assert "60" in out["message"], "the caller must be told how many exist"
    assert "project=" in out["message"], "and how to narrow them"

    payload = len(json.dumps(out))
    assert payload < 2500, f"runs.list is {payload} chars; families should not dominate"

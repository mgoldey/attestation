"""MCP wiring for the run ledger."""

import json
from pathlib import Path

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


# --- runs.record ------------------------------------------------------------


@pytest.fixture
def record_workspace(tmp_path, monkeypatch):
    """An empty workspace with one project dir, RESEARCH_ROOT-configured --
    distinct from `workspace` above, which pre-populates results/configs.
    `runs.record` is the thing writing those files in these tests, so
    starting empty is the point."""
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    get_db(tmp_path / "t.db").close()
    root = tmp_path / "ws"
    (root / "proj").mkdir(parents=True)
    monkeypatch.setenv("RESEARCH_ROOT", str(root))
    return root


def _two_arms():
    return [
        {"name": "baseline", "metrics": {"wer": 0.12}},
        {"name": "biglm", "metrics": {"wer": 0.08}},
    ]


def test_record_preview_writes_nothing(record_workspace):
    out = mcp_server._runs_record_impl("asr", _two_arms(), project="proj", corpus="librispeech")

    assert out["ok"] is True
    assert out["written"] == []
    assert out["compare"] is None
    assert set(out["manifest"]) == {
        "results/asr_baseline.json",
        "results/asr_biglm.json",
        "configs/asr_baseline.yaml",
        "configs/asr_biglm.yaml",
        "corpora.toml",
    }
    # Nothing touched the workspace: the only file that exists is the project
    # directory itself, still empty.
    assert list((record_workspace / "proj").rglob("*")) == []
    # Census entry (test_response_size.py's "Bounded in tests/test_ledger_
    # mcp.py, not here"): a two-arm manifest is small, but it is the one
    # branch that returns file CONTENTS rather than paths, so it is the
    # shape most likely to grow past the ceiling.
    assert len(json.dumps(out, indent=2)) < 7000


def test_record_confirm_writes_and_compares(record_workspace):
    out = mcp_server._runs_record_impl("asr", _two_arms(), project="proj", confirm=True)

    assert out["ok"] is True
    assert out["manifest"] == {}
    written = {str(Path(p).relative_to(record_workspace / "proj")) for p in out["written"]}
    assert written == {
        "results/asr_baseline.json",
        "results/asr_biglm.json",
        "configs/asr_baseline.yaml",
        "configs/asr_biglm.yaml",
    }
    for path in out["written"]:
        assert Path(path).exists()

    compare = out["compare"]
    assert compare is not None
    assert compare["winner"] == "asr_biglm"  # 0.08 beats 0.12, wer is lower_is_better
    # Census entry: the confirm branch returns `compare`'s own payload, whose
    # size is already bounded by runs.compare's own MAX_ARMS_SHOWN.
    assert len(json.dumps(out, indent=2)) < 7000


def test_record_collision_refuses_before_any_write(record_workspace):
    """One arm's result file already exists -- the whole call must refuse
    before writing anything, including the OTHER arm's file."""
    existing = record_workspace / "proj" / "results" / "asr_baseline.json"
    existing.parent.mkdir(parents=True)
    existing.write_text('{"wer": 0.5}')

    out = mcp_server._runs_record_impl("asr", _two_arms(), project="proj", confirm=True)

    assert out["ok"] is False
    assert "asr_baseline" in out["message"]
    assert out["written"] == []
    # The other arm's file must NOT have been created.
    assert not (record_workspace / "proj" / "results" / "asr_biglm.json").exists()
    assert not (record_workspace / "proj" / "configs" / "asr_biglm.yaml").exists()


def test_record_undeclared_metric_refuses_with_the_compare_sentence(record_workspace):
    from attestation import ledger

    arms = [
        {"name": "run1", "metrics": {"novelty_rate": 0.31}},
        {"name": "run2", "metrics": {"novelty_rate": 0.44}},
    ]

    out = mcp_server._runs_record_impl("lora", arms, project="proj")

    assert out["ok"] is False
    assert out["message"] == ledger.unknown_direction_message("novelty_rate")
    assert out["written"] == []
    assert out["manifest"] == {}


def test_record_undeclared_metric_refuses_even_with_confirm(record_workspace):
    """Confirm does not bypass the direction refusal -- it is not `force`."""
    arms = [{"name": "run1", "metrics": {"novelty_rate": 0.31}}]

    out = mcp_server._runs_record_impl("lora", arms, project="proj", confirm=True)

    assert out["ok"] is False
    assert "novelty_rate" in out["message"]
    assert out["written"] == []


def test_record_has_no_force_argument():
    """Per spec: an agent overwriting a result file is the failure the
    ledger exists to catch; `force` stays CLI-only."""
    tool = next(t for t in mcp_server.mcp._tool_manager.list_tools() if t.name == "runs.record")
    assert "force" not in (tool.parameters.get("properties") or {})


def test_record_confirm_then_detail_shows_config_as_provenance_not_metrics(
    record_workspace,
):
    """End-to-end: recording a run and then reading it back with runs.detail
    shows `scanned_at`, and the declared `--config` pair never leaks into the
    metrics list -- `record.plan` writes it to a SEPARATE provenance-only
    `configs/*.yaml` (never a metric value, per that module's own docstring),
    so `runs.detail`'s metrics stay exactly the recorded metrics."""
    out = mcp_server._runs_record_impl(
        "asr",
        _two_arms(),
        project="proj",
        config={"lr": "0.001"},
        confirm=True,
    )
    assert out["ok"] is True
    config_path = record_workspace / "proj" / "configs" / "asr_baseline.yaml"
    assert config_path.exists()
    assert "lr: 0.001" in config_path.read_text()

    detail = mcp_server._runs_detail_impl("proj", "asr_baseline")

    assert detail["ok"] is True
    run = detail["run"]
    assert run["scanned_at"]
    metric_names = {m["metric"] for m in run["metrics"]}
    assert metric_names == {"wer"}
    assert "lr" not in metric_names


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

    # indent=2, which is what FastMCP emits. Measured compact this read 2500
    # while the model received 1.238x that; round 9 measured this tool at 5926
    # chars emitted against the live database.
    payload = len(json.dumps(out, indent=2))
    assert payload < 2500, f"runs.list is {payload} chars; families should not dominate"


def test_claims_check_surfaces_an_unresolvable_citation(workspace, tmp_path, monkeypatch):
    """The tool whose name says it checks claims must check the whole claim.

    `check_citations` existed but was reachable only from `cite.check`, so a
    claim resting on a key no source has read as plain `supported` here -- the
    number agreed, and nothing said the citation did not.
    """
    monkeypatch.chdir(tmp_path)  # no .bib on the path: nothing resolves
    mcp_server._runs_scan_impl(confirm=True)
    doc = tmp_path / "R.md"
    doc.write_text(
        "<!-- claim: proj/eval_step_100 metric=wer value=0.4 cite=ghost2099nothing -->\n"
    )

    out = mcp_server._claims_check_impl(str(doc))

    verdicts = {c["verdict"] for c in out["claims"]}
    assert "supported" in verdicts, "the number still agrees"
    assert "uncited" in verdicts, "the unresolvable key was not surfaced"
    uncited = next(c for c in out["claims"] if c["verdict"] == "uncited")
    assert uncited["cite"] == "ghost2099nothing"
    assert out["counts"]["uncited"] == 1


def test_claims_check_leaves_uncited_claims_alone_when_they_cite_nothing(
    workspace, tmp_path, monkeypatch
):
    """Every claim written so far has no `cite=`. Linting them all would be a
    document-wide false alarm, so the lint must skip them entirely."""
    monkeypatch.chdir(tmp_path)
    mcp_server._runs_scan_impl(confirm=True)
    doc = tmp_path / "R.md"
    doc.write_text("<!-- claim: proj/eval_step_100 metric=wer value=0.4 -->\n")

    out = mcp_server._claims_check_impl(str(doc))

    assert out["counts"] == {"supported": 1}


def test_claims_check_says_nothing_uncited_when_the_key_resolves(workspace, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "refs.bib").write_text(
        "@article{real2020thing,\n  title = {A Real Thing},\n  year = {2020},\n}\n"
    )
    mcp_server._runs_scan_impl(confirm=True)
    doc = tmp_path / "R.md"
    doc.write_text("<!-- claim: proj/eval_step_100 metric=wer value=0.4 cite=real2020thing -->\n")

    out = mcp_server._claims_check_impl(str(doc))

    assert out["counts"] == {"supported": 1}


def test_claims_check_response_states_what_it_checked(workspace, tmp_path, monkeypatch):
    """`checked` states the numeric/citation pairing as data, not only in the
    prose a small model was measured to miss (3/3 on gemma4:e2b, per
    cite.check's own comment)."""
    monkeypatch.chdir(tmp_path)  # a resolvable .bib on the path
    (tmp_path / "refs.bib").write_text(
        "@article{real2020thing,\n  title = {A Real Thing},\n  year = {2020},\n}\n"
    )
    mcp_server._runs_scan_impl(confirm=True)
    doc = tmp_path / "R.md"
    doc.write_text("<!-- claim: proj/eval_step_100 metric=wer value=0.4 cite=real2020thing -->\n")

    out = mcp_server._claims_check_impl(str(doc))

    assert out["checked"] == ["numeric", "citation"]


def test_claims_check_checked_omits_citation_when_the_resolver_cannot_build(
    workspace, tmp_path, monkeypatch
):
    """An unbuildable resolver means the citation lint did not run, so
    `checked` must not claim it did -- the same false-clean-bill-of-health
    failure `cite.check`'s own comment warns against."""
    from attestation import citations

    def _boom(*args, **kwargs):
        raise RuntimeError("no citation backend available")

    monkeypatch.setattr(citations.Resolver, "from_env", _boom)
    mcp_server._runs_scan_impl(confirm=True)
    doc = tmp_path / "R.md"
    doc.write_text("<!-- claim: proj/eval_step_100 metric=wer value=0.4 -->\n")

    out = mcp_server._claims_check_impl(str(doc))

    assert out["checked"] == ["numeric"]


def test_runs_list_does_not_paste_absolute_paths_for_every_run(tmp_path, monkeypatch):
    """`source_path` is the most expensive field in a runs.list row and the
    least useful there.

    Measured against the live ledger: a DEFAULT runs.list emits 5926 chars,
    of which 4655 is the runs array -- 20 rows each carrying a full absolute
    path like /home/matt/qc/ablation/results/layer_importance.json. The tool's
    own message already says "showing 20 of 858"; a caller narrowing that list
    needs project, name and family, and `runs.detail` is the tool that returns
    the path.

    The fixture-based size assertion above passed the whole time, because a
    tmp_path workspace has short paths. Same shape as every other budget this
    review found: cheap fixture, expensive production.
    """
    import json

    from attestation.mcp import provenance as pv

    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    conn = get_db(tmp_path / "t.db")
    for i in range(20):
        conn.execute(
            "INSERT INTO runs(project, name, family, status, source_path)"
            " VALUES (?, ?, ?, 'recorded', ?)",
            (
                "ablation",
                f"sweep_arm_{i}",
                "sweep",
                # A realistic absolute path, like the live ledger's.
                f"/home/matt/qc/ablation/results/layer_importance_variant_{i}.json",
            ),
        )
    conn.commit()
    conn.close()

    out = pv._list()
    assert out["runs"], "fixture produced no runs"
    for run in out["runs"]:
        assert "source_path" not in run, "a list row carried a full absolute path"
    # And the detail tool still gives it to anyone who asks.
    first = out["runs"][0]
    detail = pv._detail(first["project"], first["name"])
    assert detail["run"]["source_path"], "runs.detail must still return the path"
    assert len(json.dumps(out, indent=2)) < 2500


def test_runs_detail_caps_its_metric_rows(tmp_path, monkeypatch):
    """The largest response on the whole surface, filed under "returns a status".

    The census listed runs.detail as a mutator or single-value tool. Its own
    docstring says "One run in full: config shape, EVERY metric" -- and the
    metrics array was uncapped. Measured across all 858 live runs: median 1070,
    max 60680, and 72 runs over the 7000-char ceiling. The worst is 49945 chars
    of metrics in one response whose message reads "429 metric row(s)".

    That is 8.7x what a small model can hold, and it is the exact failure the
    response-size module docstring describes. The census gates on NAME, so a
    tool already on a list can grow without limit -- this is that blind spot
    already realised.
    """
    import json

    from attestation.mcp import provenance as pv

    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    conn = get_db(tmp_path / "t.db")
    conn.execute(
        "INSERT INTO runs(project, name, family, status, source_path)"
        " VALUES ('p', 'r', 'f', 'recorded', '/tmp/r.json')"
    )
    for i in range(429):
        conn.execute(
            "INSERT INTO run_metrics(run_id, metric, value, step, split)"
            " VALUES (1, ?, ?, ?, 'test')",
            (f"conformer_energy_variant_{i}", float(i), i),
        )
    conn.commit()
    conn.close()

    out = pv._detail("p", "r")
    assert len(out["run"]["metrics"]) <= pv.MAX_METRIC_ROWS, (
        f"{len(out['run']['metrics'])} metric rows returned uncapped"
    )
    assert out["run"]["n_metrics"] == 429, "the true count must survive truncation"
    assert "429" in out["message"], "the caller must be told what was not shown"
    assert len(json.dumps(out, indent=2)) < 7000


def test_runs_list_reports_what_it_did_not_show(tmp_path, monkeypatch):
    """Halving the default limit was justified by a message that did not exist.

    `DEFAULT_RUNS_LIMIT` went 20 -> 10 on the stated grounds that "the message
    already reports what was not shown". It did not: `project=ablation` has 222
    runs, the tool returns 10, and the message reads "10 run(s)". The families
    truncation in the SAME response is reported exactly, and feed.list says
    "more available -- raise limit (max 50)", so this broke the codebase's own
    convention while halving the visible answer.
    """
    from attestation.mcp import provenance as pv

    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    conn = get_db(tmp_path / "t.db")
    for i in range(40):
        conn.execute(
            "INSERT INTO runs(project, name, family, status, source_path)"
            " VALUES ('p', ?, 'f', 'recorded', '/tmp/x.json')",
            (f"run{i}",),
        )
    conn.commit()
    conn.close()

    out = pv._list()
    assert len(out["runs"]) == pv.DEFAULT_RUNS_LIMIT
    assert "40" in out["message"], f"40 runs exist, message says {out['message']!r}"
    assert "limit" in out["message"], "the caller must be told how to see more"


def test_detail_truncation_keeps_distinct_metrics_not_the_first_alphabetically(
    tmp_path, monkeypatch
):
    """Capping ROWS destroyed the thing runs.detail exists to show.

    ledger.detail orders by (metric, step), so taking the first 40 rows of a
    429-row run returned `b_g` forty times -- one metric name at forty steps --
    while 32 other distinct metrics vanished. The live worst case is exactly
    that: 33 distinct names averaging 13 steps each.

    A caller reading ONE run wants its shape: which quantities were measured.
    The step series belongs to runs.compare. So the cap keeps the last value of
    each distinct metric, and says how many steps it collapsed.
    """
    from attestation.mcp import provenance as pv

    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    conn = get_db(tmp_path / "t.db")
    conn.execute(
        "INSERT INTO runs(project, name, family, status, source_path)"
        " VALUES ('p', 'r', 'f', 'recorded', '/tmp/r.json')"
    )
    for name in ("alpha", "beta", "gamma", "delta", "epsilon"):
        for step in range(30):
            conn.execute(
                "INSERT INTO run_metrics(run_id, metric, value, step, split)"
                " VALUES (1, ?, ?, ?, 'test')",
                (name, float(step), step),
            )
    conn.commit()
    conn.close()

    out = pv._detail("p", "r")
    kept = {m["metric"] for m in out["run"]["metrics"]}
    assert kept == {"alpha", "beta", "gamma", "delta", "epsilon"}, (
        f"truncation dropped whole metrics: kept {sorted(kept)}"
    )
    assert out["run"]["n_metrics"] == 150


def test_runs_list_samples_across_projects_not_alphabetically(tmp_path, monkeypatch):
    """A default listing showed one project and hid seventeen.

    ORDER BY project, family, name plus LIMIT 10 means the first project
    alphabetically fills the whole answer. Measured on the live ledger: 18
    projects, 858 runs, and a default runs.list returned 10 runs all from
    `ablation`. The message said "10 run(s) of 858" -- true, and it did not say
    they were all one project, so a researcher asking what runs they have sees
    one seventeenth of the answer with no sign of the rest.

    `families` in the same response already samples across projects. This makes
    the runs do the same, which is what makes the count honest.
    """
    from attestation.mcp import provenance as pv

    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    conn = get_db(tmp_path / "t.db")
    for project in ("aardvark", "beta", "gamma", "delta", "epsilon"):
        for i in range(30):
            conn.execute(
                "INSERT INTO runs(project, name, family, status, source_path)"
                " VALUES (?, ?, 'f', 'recorded', '/tmp/x.json')",
                (project, f"{project}_run{i}"),
            )
    conn.commit()
    conn.close()

    out = pv._list()
    projects = {r["project"] for r in out["runs"]}
    assert len(projects) >= 3, f"a default listing of 5 projects showed only {sorted(projects)}"


def test_runs_list_fits_at_the_limit_it_advertises(tmp_path, monkeypatch):
    """The escape moved from "a tool nobody measured" to "an argument nobody
    passed".

    Every size guard drives runs.list at its DEFAULT. Its schema allows
    limit=50 and its own message tells the caller to use it -- "raise limit
    (max 50)" -- and at 50 it emitted 9965 chars against a 7000 ceiling.
    Round 11's honesty fix is what made the breach reachable: the message now
    advertises the limit that blows it.
    """
    import json

    from attestation.mcp import provenance as pv
    from attestation.mcp._shared import MAX_LIST_LIMIT

    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    conn = get_db(tmp_path / "t.db")
    for project in ("alpha", "beta", "gamma"):
        for i in range(40):
            conn.execute(
                "INSERT INTO runs(project, name, family, status, source_path)"
                " VALUES (?, ?, ?, 'recorded', '/tmp/x.json')",
                (project, f"{project}_experiment_variant_{i}", f"{project}_family_{i % 7}"),
            )
    conn.commit()
    conn.close()

    out = pv._list(limit=MAX_LIST_LIMIT)
    size = len(json.dumps(out, indent=2))
    assert size <= 7000, f"runs.list at its advertised max emits {size} chars"


def test_runs_compare_fits_at_its_widest_family(tmp_path, monkeypatch):
    """A composition tool is exempt from the conversational budget, not from
    what a caller can hold.

    runs.compare had no cap at all and reached 13624 chars on a 48-arm family
    -- larger than the pre-fix runs.detail. Being declared a composition tool
    had become permanent permission to grow, which is round 11's finding in a
    tool that was left out of the driven census.
    """
    import json

    from attestation.mcp import provenance as pv

    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    conn = get_db(tmp_path / "t.db")
    for i in range(80):
        conn.execute(
            "INSERT INTO runs(project, name, family, status, source_path)"
            " VALUES ('p', ?, 'sweep', 'recorded', ?)",
            (f"sweep_arm_variant_{i}", f"/home/matt/qc/p/results/arm_{i}.json"),
        )
        conn.execute(
            "INSERT INTO run_metrics(run_id, metric, value, step, split)"
            " VALUES ((SELECT id FROM runs WHERE name = ?), 'wer', ?, 1, 'test')",
            (f"sweep_arm_variant_{i}", 0.5 - i * 0.001),
        )
    conn.commit()
    conn.close()

    out = pv._compare("sweep")
    size = len(json.dumps(out, indent=2))
    assert size <= 7000, f"runs.compare on an 80-arm family emits {size} chars"
    assert out["n_arms"] >= 80, "the true arm count must survive truncation"


def test_a_refusal_never_discards_the_message_the_domain_layer_built(tmp_path, monkeypatch):
    """The structural defect behind three separate findings.

    `ledger.scan` sets `message` when the root does not exist; `_list` returns
    "no runs recorded -- call runs.scan(confirm=true) first". Both are correct
    where they are written, and both were thrown away by a wrapper that
    rebuilt its own: `runs.scan` reported ok=true / "0 run(s) across 0
    project(s)" for a typo'd path, and `runs.ask` answered "Which family?
    Comparable families include: " -- a sentence stopping mid-list.

    The @tool decorator fixed the envelope's SHAPE; nothing made a body relay
    the content it was handed.
    """
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    from attestation.mcp.ask import _runs_ask
    from attestation.mcp.provenance import _scan

    missing = _scan(root=str(tmp_path / "definitely-absent"), confirm=True)
    assert missing["ok"] is False, "a nonexistent root reported success"
    assert "no such directory" in missing["message"], missing["message"]

    answer = _runs_ask("which arm won my sweep")["answer"]
    assert answer.strip(), "the disambiguation prompt is empty"
    assert not answer.rstrip().endswith(":"), (
        f"the answer stops mid-list, asking the caller to choose from nothing: {answer!r}"
    )
    assert "runs.scan" in answer, f"an empty ledger must name the call that fills it: {answer!r}"


def test_a_filter_that_matches_nothing_is_not_an_empty_ledger(tmp_path, monkeypatch):
    """`runs.list(project="nosuch")` said "no runs recorded -- call
    runs.scan(confirm=true) first" against a ledger holding nine runs.
    Following that advice re-scans a database that is already correct."""
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    import pathlib

    from attestation.mcp.provenance import _list, _scan

    root = pathlib.Path(__file__).resolve().parents[1] / "examples" / "workspace"
    _scan(root=str(root), confirm=True)

    miss = _list(project="nosuch")
    assert miss["ok"] is False
    assert "no runs recorded" not in miss["message"], (
        f"a filter miss is reported as an empty ledger: {miss['message']!r}"
    )
    assert "though the ledger holds" in miss["message"], miss["message"]


def test_a_family_in_another_project_says_which(tmp_path, monkeypatch):
    """It denied the family and listed it as available in the same sentence,
    never mentioning the `project` argument that actually excluded it."""
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    import pathlib

    from attestation.mcp.provenance import _compare, _scan

    root = pathlib.Path(__file__).resolve().parents[1] / "examples" / "workspace"
    _scan(root=str(root), confirm=True)

    out = _compare(family="kdsweep", project="retrieval-ablation")
    assert out["ok"] is False
    assert "exists, but not in project" in out["message"], out["message"]
    assert "speech-distill" in out["message"], (
        f"the refusal does not say where the family actually is: {out['message']!r}"
    )

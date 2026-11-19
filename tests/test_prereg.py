"""Hypothesis preregistration: signing, tamper detection, the verdict rule, the surfaces.

Most tests here change something after registration -- a character of the
index, the canonical file, the record, an entry removed -- and assert it
reads as `altered`. The verdict tests are table-driven over literal dicts in
the shape `ledger.compare()` returns, with and without builder A's replicate
keys, because `evaluate` must work either way.
"""

import json
import os
import socket
from datetime import UTC, datetime, timedelta

import pytest

from attestation import ledger, prereg
from attestation.db import get_db
from attestation.prereg import Hypothesis, VerdictKind
from attestation.run_record import keys, verify

# --- fixtures ----------------------------------------------------------------


@pytest.fixture
def signer(tmp_path, monkeypatch):
    """A key under tmp_path, named by ATTEST_RECORD_KEY; HOME holds none."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home" / ".hermes"))
    path = tmp_path / "keys" / "prereg.pem"
    keys.LocalKeySource(path).signer()
    monkeypatch.setenv(keys.ENV_KEY, str(path))
    return keys.resolve_key_source().signer()


@pytest.fixture
def root(tmp_path):
    out = tmp_path / "ws"
    out.mkdir()
    return out


def hyp(**kw) -> Hypothesis:
    base = {"family": "kdsweep", "metric": "wer", "expect": "kdsweep_t4 < kdsweep_base", "by": 0.01}
    return Hypothesis(**(base | kw))


def arm(name, value, **extra) -> dict:
    return {"name": name, "value": value, "source_path": None, **extra}


def result(*arms, metric="wer") -> dict:
    return {"family": "kdsweep", "metric": metric, "arms": list(arms)}


# --- the hypothesis itself -----------------------------------------------------


def test_the_id_is_the_sha256_of_the_canonical_content():
    import hashlib

    h = hyp()
    assert h.id == hashlib.sha256(h.canonical()).hexdigest()
    assert hyp(expect="kdsweep_t4<kdsweep_base").id == h.id  # spelling, not meaning
    assert hyp(by=0.02).id != h.id
    assert hyp(note="why").id != h.id


@pytest.mark.parametrize(
    "bad",
    [
        {"expect": "kdsweep_t4"},
        {"expect": "a < < b"},
        {"expect": "a <= b"},
        {"expect": "a < a"},
        {"by": -0.1},
        {"by": float("nan")},
        {"by": True},
        {"note": "line\nbreak"},
        {"family": ""},
        {"supersedes": "abc"},
    ],
)
def test_a_malformed_hypothesis_is_refused(bad):
    with pytest.raises(ValueError):
        hyp(**bad)


def test_the_index_block_round_trips_through_toml(tmp_path):
    h = hyp(note='said "quietly" -- naïve ✓', allow_single_run=True, by=1e-05)
    path = tmp_path / "hypotheses.toml"
    path.write_text(prereg.index_block(h, "2026-11-19T00:00:00Z"))
    [entry] = prereg.parse_file(path)
    assert entry.problem is None
    assert entry.hypothesis == h
    assert entry.id == h.id


# --- registering and reading back ------------------------------------------------


def test_a_registration_verifies_as_a_run_record(root, signer):
    done = prereg.register(hyp(), root, signer)

    report = verify.verify_file(done.record, root=root)
    assert report.ok, report.lines()
    statuses = {i.what: i.status for i in report}
    assert statuses[f"subject prereg/{done.id[:16]}.hypothesis.json"] == verify.OK
    [reg] = prereg.audit(root)
    assert reg.flags == []
    assert reg.hypothesis == hyp()
    assert reg.keyid == signer.keyid
    assert reg.registered_at == done.registered_at


def test_registering_the_same_content_twice_is_refused(root, signer):
    prereg.register(hyp(), root, signer)
    with pytest.raises(FileExistsError, match="already registered"):
        prereg.register(hyp(), root, signer)


def test_one_changed_character_in_the_index_reads_as_altered(root, signer):
    prereg.register(hyp(by=0.01), root, signer)
    index = root / "hypotheses.toml"
    index.write_text(index.read_text().replace("by = 0.01", "by = 0.02"))

    [reg] = prereg.audit(root)
    assert prereg.ALTERED in reg.flags
    assert any("no longer hashes to its id" in r for r in reg.reasons)
    # the verdict is computed from the SIGNED text, never from the edited copy
    assert reg.hypothesis.by == 0.01


def test_editing_the_id_too_still_reads_as_altered(root, signer):
    """Re-hashing the edited text and writing the new id is the "silent re-hash"
    the index cannot be allowed to get away with: the record for the new id
    does not exist, and the old record is still there with no entry."""
    done = prereg.register(hyp(by=0.01), root, signer)
    forged = hyp(by=0.02)
    index = root / "hypotheses.toml"
    index.write_text(
        index.read_text().replace("by = 0.01", "by = 0.02").replace(done.id, forged.id)
    )

    regs = {r.id: r for r in prereg.audit(root)}
    assert prereg.ALTERED in regs[forged.id].flags
    assert any("no signed record" in r for r in regs[forged.id].reasons)
    assert prereg.ALTERED in regs[done.id].flags
    assert any("no entry" in r for r in regs[done.id].reasons)


def test_a_changed_registration_time_reads_as_altered(root, signer):
    done = prereg.register(hyp(), root, signer, registered_at="2026-11-19T12:00:00Z")
    index = root / "hypotheses.toml"
    index.write_text(index.read_text().replace(done.registered_at, "2026-01-01T00:00:00Z"))

    [reg] = prereg.audit(root)
    assert prereg.ALTERED in reg.flags
    assert reg.registered_at == "2026-11-19T12:00:00Z"  # the signed time wins


def test_an_edited_canonical_file_reads_as_altered(root, signer):
    done = prereg.register(hyp(), root, signer)
    done.canonical.write_bytes(done.canonical.read_bytes().replace(b"0.01", b"0.02"))

    [reg] = prereg.audit(root)
    assert prereg.ALTERED in reg.flags
    assert any("sha256 differs" in r for r in reg.reasons)


def test_an_edited_record_reads_as_altered(root, signer):
    done = prereg.register(hyp(), root, signer)
    envelope = json.loads(done.record.read_text())
    sig = envelope["signatures"][0]["sig"]
    envelope["signatures"][0]["sig"] = ("A" if sig[0] != "A" else "B") + sig[1:]
    done.record.write_text(json.dumps(envelope))

    [reg] = prereg.audit(root)
    assert prereg.ALTERED in reg.flags
    assert any("signature" in r for r in reg.reasons)


def test_a_removed_index_entry_is_still_reported(root, signer):
    prereg.register(hyp(), root, signer)
    (root / "hypotheses.toml").write_text("")

    [reg] = prereg.audit(root)
    assert prereg.ALTERED in reg.flags
    assert reg.hypothesis == hyp()
    assert any("no entry in hypotheses.toml" in r for r in reg.reasons)


def test_a_missing_record_reads_as_altered(root, signer):
    done = prereg.register(hyp(), root, signer)
    done.record.unlink()

    [reg] = prereg.audit(root)
    assert prereg.ALTERED in reg.flags


def test_a_superseded_registration_stays_and_is_flagged(root, signer):
    first = prereg.register(hyp(by=0.01), root, signer)
    entries = prereg.parse_file(root / "hypotheses.toml")
    full = prereg.resolve_supersedes(first.id[:8], entries)
    prereg.register(hyp(by=0.005, supersedes=full), root, signer)

    regs = {r.id: r for r in prereg.audit(root)}
    assert len(regs) == 2
    assert prereg.SUPERSEDED in regs[first.id].flags
    with pytest.raises(ValueError, match="no registered hypothesis"):
        prereg.resolve_supersedes("deadbeefdeadbeef", entries)


def test_a_key_problem_writes_nothing(root, tmp_path, monkeypatch):
    monkeypatch.setenv(keys.ENV_KEY, str(tmp_path / "missing.pem"))
    with pytest.raises(keys.RecordKeyError):
        prereg.register(hyp(), root, keys.resolve_key_source().signer())
    assert not (root / "hypotheses.toml").exists()
    assert not (root / "prereg").exists()


# --- the verdict rule ----------------------------------------------------------------

CI = {"n_runs": 3}

VERDICT_CASES = [
    # (id, hypothesis kwargs, arms, expected verdict)
    (
        "intervals separated by the margin",
        {},
        [
            arm("kdsweep_t4", 0.05, ci_low=0.048, ci_high=0.052, **CI),
            arm("kdsweep_base", 0.07, ci_low=0.068, ci_high=0.072, **CI),
        ],
        VerdictKind.CONFIRMED,
    ),
    (
        "intervals exclude the margin",
        {},
        [
            arm("kdsweep_t4", 0.07, ci_low=0.069, ci_high=0.071, **CI),
            arm("kdsweep_base", 0.072, ci_low=0.071, ci_high=0.073, **CI),
        ],
        VerdictKind.REFUTED,
    ),
    (
        "intervals overlap the margin",
        {},
        [
            arm("kdsweep_t4", 0.06, ci_low=0.055, ci_high=0.065, **CI),
            arm("kdsweep_base", 0.07, ci_low=0.065, ci_high=0.075, **CI),
        ],
        VerdictKind.INCONCLUSIVE,
    ),
    (
        "single runs without allow_single_run",
        {},
        [arm("kdsweep_t4", 0.05), arm("kdsweep_base", 0.09)],
        VerdictKind.INCONCLUSIVE,
    ),
    (
        "single runs refuting without allow_single_run",
        {},
        [arm("kdsweep_t4", 0.09), arm("kdsweep_base", 0.05)],
        VerdictKind.INCONCLUSIVE,
    ),
    (
        "single runs with allow_single_run, margin met",
        {"allow_single_run": True},
        [arm("kdsweep_t4", 0.06), arm("kdsweep_base", 0.07)],
        VerdictKind.CONFIRMED,
    ),
    (
        "single runs with allow_single_run, margin missed",
        {"allow_single_run": True},
        [arm("kdsweep_t4", 0.065), arm("kdsweep_base", 0.07)],
        VerdictKind.REFUTED,
    ),
    (
        "one arm replicated, one not",
        {},
        [arm("kdsweep_t4", 0.05, ci_low=0.048, ci_high=0.052, **CI), arm("kdsweep_base", 0.09)],
        VerdictKind.INCONCLUSIVE,
    ),
    (
        "two replicates: mean, no interval (builder A's n_runs < 3)",
        {"allow_single_run": True},
        [arm("kdsweep_t4", 0.05, n_runs=2, runs=["a", "b"]), arm("kdsweep_base", 0.07)],
        VerdictKind.CONFIRMED,
    ),
    (
        # A's real shape for n_runs < 3: the interval keys are PRESENT but None
        "two replicates with ci keys None, default",
        {},
        [
            arm("kdsweep_t4", 0.05, n_runs=2, mean=0.05, ci_low=None, ci_high=None, runs=["a"]),
            arm("kdsweep_base", 0.09, n_runs=2, mean=0.09, ci_low=None, ci_high=None, runs=["b"]),
        ],
        VerdictKind.INCONCLUSIVE,
    ),
    (
        "two replicates with ci keys None, allow_single_run",
        {"allow_single_run": True},
        [
            arm("kdsweep_t4", 0.05, n_runs=2, mean=0.05, ci_low=None, ci_high=None, runs=["a"]),
            arm("kdsweep_base", 0.09, n_runs=2, mean=0.09, ci_low=None, ci_high=None, runs=["b"]),
        ],
        VerdictKind.CONFIRMED,
    ),
    (
        "greater-than",
        {"expect": "kdsweep_base > kdsweep_t4", "allow_single_run": True},
        [arm("kdsweep_t4", 0.06), arm("kdsweep_base", 0.07)],
        VerdictKind.CONFIRMED,
    ),
    (
        "a missing arm is untested, not refuted",
        {"allow_single_run": True},
        [arm("kdsweep_t4", 0.06)],
        VerdictKind.UNTESTED,
    ),
    (
        "an arm without a value is untested",
        {"allow_single_run": True},
        [arm("kdsweep_t4", 0.06), arm("kdsweep_base", None)],
        VerdictKind.UNTESTED,
    ),
    (
        "a refuted pair settles a chain with an untested pair",
        {"expect": "kdsweep_t4 < kdsweep_base < kdsweep_gone", "allow_single_run": True},
        [arm("kdsweep_t4", 0.09), arm("kdsweep_base", 0.05)],
        VerdictKind.REFUTED,
    ),
    (
        "a chain confirmed pair by pair",
        {"expect": "kdsweep_t4 < kdsweep_t2 < kdsweep_base", "allow_single_run": True},
        [arm("kdsweep_t4", 0.05), arm("kdsweep_t2", 0.06), arm("kdsweep_base", 0.07)],
        VerdictKind.CONFIRMED,
    ),
]


@pytest.mark.parametrize(
    "kwargs,arms,expected", [c[1:] for c in VERDICT_CASES], ids=[c[0] for c in VERDICT_CASES]
)
def test_evaluate(kwargs, arms, expected):
    assert prereg.evaluate(hyp(**kwargs), result(*arms)).verdict == expected


def test_the_margin_forgives_float_noise():
    """0.07 - 0.06 is 0.010000000000000009 one way and 0.0099999... another."""
    h = hyp(by=0.01, allow_single_run=True)
    out = prereg.evaluate(h, result(arm("kdsweep_t4", 0.05), arm("kdsweep_base", 0.06)))
    assert out.verdict == VerdictKind.CONFIRMED


def test_a_seed_named_instead_of_its_arm_is_not_evidence():
    h = hyp(expect="kdsweep_t4_seed1 < kdsweep_base", allow_single_run=True)
    arms = [
        arm("kdsweep_t4", 0.05, ci_low=0.04, ci_high=0.06, n_runs=3, runs=["kdsweep_t4_seed1"]),
        arm("kdsweep_base", 0.09),
    ]
    out = prereg.evaluate(h, result(*arms))
    assert out.verdict == VerdictKind.INCONCLUSIVE
    assert "replicate" in out.reasons[0]


def test_a_comparison_on_another_metric_is_untested():
    h = hyp(allow_single_run=True)
    out = prereg.evaluate(
        h, result(arm("kdsweep_t4", 0.5), arm("kdsweep_base", 0.9), metric="accuracy")
    )
    assert out.verdict == VerdictKind.UNTESTED
    assert "accuracy" in out.reasons[0]


def test_post_hoc_is_flagged_beside_a_computed_verdict():
    h = hyp(allow_single_run=True)
    registered = "2026-11-19T12:00:00Z"
    when = datetime(2026, 11, 19, 12, tzinfo=UTC)
    arms = result(arm("kdsweep_t4", 0.05), arm("kdsweep_base", 0.07))

    late = prereg.evaluate(
        h,
        arms,
        {"kdsweep_base": when - timedelta(days=1), "kdsweep_t4": when + timedelta(days=1)},
        registered_at=registered,
    )
    assert late.post_hoc is True
    assert late.verdict == VerdictKind.CONFIRMED  # reported, not suppressed
    assert set(late.predates) == {"kdsweep_base"}

    early = prereg.evaluate(
        h, arms, {"kdsweep_base": when + timedelta(seconds=1)}, registered_at=registered
    )
    assert early.post_hoc is False


def test_evaluate_is_pure():
    """No file, clock, socket or database: the same inputs, the same answer."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(prereg.evaluate))
    called = {
        n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and hasattr(n.func, "id")
    }
    assert not called & {"open", "evidence_times", "audit", "now"}


# --- refusals ------------------------------------------------------------------------


def test_an_undeclared_metric_is_refused_with_the_ledgers_sentence():
    with pytest.raises(ValueError) as exc:
        prereg.refuse_undeclared("n_widgets")
    assert str(exc.value) == ledger.unknown_direction_message("n_widgets")


def test_a_declared_stem_is_accepted():
    prereg.refuse_undeclared("best_val_loss")  # stem `loss`, as compare matches it
    prereg.refuse_undeclared("wer")


# --- the CLI ---------------------------------------------------------------------------

LIMIT_PHRASES = (
    "proved: the key named in each record signed exactly this hypothesis text",
    "evidence only: registered_at is this machine's clock",
    "post_hoc unset proves nothing",
    "cannot check: that the prediction was written before anyone saw the data",
    "who holds the key",
)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def ledger_ws(tmp_path, monkeypatch, signer):
    """A scanned workspace with a three-arm kdsweep family and RESEARCH_ROOT set."""
    ws = tmp_path / "research"
    for name, wer in (("kdsweep_base", 0.08), ("kdsweep_t2", 0.07), ("kdsweep_t4", 0.06)):
        _write(ws / "proj" / "results" / f"{name}.json", json.dumps({"wer": wer}))
    monkeypatch.setenv("RESEARCH_ROOT", str(ws))
    monkeypatch.setenv("ATTEST_DB", str(tmp_path / "t.db"))
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, ws)
    conn.close()
    return ws


def _date_artifacts(ws, days: int):
    """Move every artifact's mtime `days` from now: positive keeps post_hoc
    unset for a registration made now, negative sets it."""
    when = (datetime.now(UTC) + timedelta(days=days)).timestamp()
    for path in (ws / "proj" / "results").iterdir():
        os.utime(path, (when, when))


def _future_artifacts(ws):
    _date_artifacts(ws, 1)


def _cli(capsys, *argv) -> tuple[int, str]:
    from attestation.cli import main

    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def test_cli_add_check_list_round_trip_and_state_the_limits(ledger_ws, capsys):
    _future_artifacts(ledger_ws)
    code, out = _cli(
        capsys,
        *("prereg", "add", "kdsweep", "--metric", "wer"),
        *("--expect", "kdsweep_t4 < kdsweep_base", "--by", "0.01", "--allow-single-run"),
    )
    assert code == 0, out
    assert "registered " in out
    for phrase in LIMIT_PHRASES:
        assert phrase in out

    code, out = _cli(capsys, "prereg", "check")
    assert code == 0, out
    assert "CONFIRMED" in out
    assert "post_hoc" not in out.split("What a preregistration shows")[0]
    for phrase in LIMIT_PHRASES:
        assert phrase in out

    code, out = _cli(capsys, "prereg", "list", "kdsweep")
    assert code == 0
    assert "kdsweep wer: kdsweep_t4 < kdsweep_base by 0.01" in out
    for phrase in LIMIT_PHRASES:
        assert phrase in out


def test_cli_check_json_carries_the_limits(ledger_ws, capsys):
    _cli(
        capsys,
        "prereg",
        "add",
        "kdsweep",
        "--metric",
        "wer",
        "--expect",
        "kdsweep_t4 < kdsweep_base",
        "--by",
        "0",
    )
    code, out = _cli(capsys, "prereg", "check", "--json")
    payload = json.loads(out)
    joined = " ".join(payload["limits"])
    for phrase in LIMIT_PHRASES:
        assert phrase in joined
    assert payload["hypotheses"][0]["verdict"] == "inconclusive"


def test_cli_check_exits_1_on_refuted(ledger_ws, capsys):
    _future_artifacts(ledger_ws)
    argv = ("prereg", "add", "kdsweep", "--metric", "wer", "--by", "0", "--allow-single-run")
    _cli(capsys, *argv, "--expect", "kdsweep_base < kdsweep_t4")
    code, out = _cli(capsys, "prereg", "check")
    assert code == 1
    assert "REFUTED" in out


def test_cli_check_exits_1_on_altered_but_not_on_post_hoc_alone(ledger_ws, capsys):
    argv = ("prereg", "add", "kdsweep", "--metric", "wer", "--by", "0.01", "--allow-single-run")
    _date_artifacts(ledger_ws, -1)
    _cli(capsys, *argv, "--expect", "kdsweep_t4 < kdsweep_base")
    code, out = _cli(capsys, "prereg", "check")  # artifacts predate: post_hoc
    assert code == 0, out
    assert "[post_hoc]" in out
    assert "CONFIRMED" in out  # the verdict is still computed

    index = ledger_ws / "hypotheses.toml"
    index.write_text(index.read_text().replace("by = 0.01", "by = 0.001"))
    code, out = _cli(capsys, "prereg", "check")
    assert code == 1
    assert "altered" in out


def test_cli_add_refuses_an_undeclared_metric_and_writes_nothing(ledger_ws, capsys):
    code, out = _cli(
        capsys,
        "prereg",
        "add",
        "kdsweep",
        "--metric",
        "n_widgets",
        "--expect",
        "a < b",
        "--by",
        "1",
    )
    assert code == 1
    assert ledger.unknown_direction_message("n_widgets") in out
    assert not (ledger_ws / "hypotheses.toml").exists()


def test_cli_check_for_a_family_that_is_not_in_the_ledger_is_untested(ledger_ws, capsys):
    argv = ("prereg", "add", "nosuch", "--metric", "wer", "--by", "0", "--expect", "a < b")
    assert _cli(capsys, *argv)[0] == 0
    code, out = _cli(capsys, "prereg", "check")
    assert code == 0
    assert "UNTESTED" in out


def test_nothing_reaches_the_network_and_no_key_lands_in_home(
    ledger_ws, capsys, monkeypatch, tmp_path
):
    def no_socket(*a, **k):
        raise AssertionError("prereg opened a socket")

    monkeypatch.setattr(socket, "socket", no_socket)
    argv = (
        "prereg",
        "add",
        "kdsweep",
        "--metric",
        "wer",
        "--by",
        "0",
        "--expect",
        "kdsweep_t4 < kdsweep_base",
    )
    assert _cli(capsys, *argv)[0] == 0
    assert _cli(capsys, "prereg", "check")[0] == 0
    assert not (tmp_path / "home" / ".hermes" / "attestation" / "keys").exists()


# --- the runs.compare block ----------------------------------------------------------


def test_compare_block_states_the_limits_and_judges_on_all_arms(ledger_ws, signer):
    from attestation import mcp_server

    prereg.register(
        hyp(expect="kdsweep_t4 < kdsweep_base", allow_single_run=True), ledger_ws, signer
    )
    out = mcp_server._runs_compare_impl("kdsweep", metric="wer")

    assert out["ok"] is True
    assert out["n_hypotheses"] == 1
    [h] = out["hypotheses"]
    assert h["verdict"] == "confirmed"
    assert h["expect"] == "kdsweep wer: kdsweep_t4 < kdsweep_base by 0.01"
    joined = " ".join(out["prereg_limits"])
    for phrase in LIMIT_PHRASES:
        assert phrase in joined


def test_compare_block_does_not_judge_another_metric(ledger_ws, signer):
    prereg.register(hyp(metric="loss"), ledger_ws, signer)
    block = prereg.attach(result(arm("kdsweep_t4", 0.05), arm("kdsweep_base", 0.07)), ledger_ws)
    [h] = block["hypotheses"]
    assert h["verdict"] == "untested"
    assert "registered on loss" in h["reasons"][0]


def test_compare_block_is_capped_and_small(root, signer):
    for i in range(MAXH := prereg.MAX_HYPOTHESES_SHOWN + 3):
        prereg.register(hyp(by=0.001 * (i + 1), note="n" * 60), root, signer)
    block = prereg.attach(result(arm("kdsweep_t4", 0.05), arm("kdsweep_base", 0.07)), root)
    assert block["n_hypotheses"] == MAXH
    assert len(block["hypotheses"]) == prereg.MAX_HYPOTHESES_SHOWN
    assert len(json.dumps(block)) < 2500


def test_compare_block_is_absent_without_registrations(root):
    assert prereg.attach(result(arm("kdsweep_t4", 0.05)), root) == {}
    assert prereg.attach(result(arm("kdsweep_t4", 0.05)), None) == {}

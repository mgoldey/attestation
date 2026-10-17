"""Run records: signing, verifying, tamper detection, canonical form.

A record's whole value is that changing it after the fact is detectable, so
most of these tests change something -- a verdict, an input file, the key --
and assert the verifier FAILS and says what.
"""

import ast
import base64
import json
import stat
from pathlib import Path

import pytest

from attestation.run_record import build, canonical, keys, verify


@pytest.fixture
def key_home(tmp_path, monkeypatch):
    """Keys under tmp_path, never the real ~/.hermes."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv(keys.ENV_KEY, raising=False)
    return tmp_path / "hermes"


@pytest.fixture
def signer(key_home):
    return keys.resolve_key_source().signer()


@pytest.fixture
def run(tmp_path):
    """A small run: a draft, a results file, a bib, an output brief."""
    root = tmp_path / "work"
    root.mkdir()
    (root / "draft.md").write_text("WER is 5.3%.\n")
    (root / "results.json").write_text('{"wer": 0.053}\n')
    (root / "refs.bib").write_text("@article{smith2020, title={T}}\n")
    (root / "brief.md").write_text("# Brief\n")
    rec = build.RunRecorder("research-brief", root=root)
    rec.add_input("draft.md", "draft")
    rec.add_input("results.json", "results")
    rec.add_input("refs.bib", "bib")
    rec.add_output("brief.md")
    rec.add_model("gemma4:e2b", "chat", "http://localhost:11434/v1")
    rec.add_query("openalex", "interatomic potentials", 12)
    rec.add_query("pubmed", "interatomic potentials", 0)
    rec.add_paper("doi:10.1/abc", "full", "A paper")
    rec.add_paper("arxiv:2401.00001", "abstract")
    rec.add_check(
        build.CheckResult(
            "claims",
            [
                build.Finding("draft.md:1", "supported", "", {"value": 0.053}),
                build.Finding("draft.md:9", "contradicted", "run says 0.061"),
            ],
            target="draft.md",
        )
    )
    return root, rec


def _statement(envelope):
    return json.loads(base64.b64decode(envelope["payload"]))


def _reseal_payload(envelope, statement):
    """Swap in an edited statement WITHOUT re-signing: what a tamperer can do."""
    return envelope | {"payload": base64.b64encode(canonical.canonicalize(statement)).decode()}


def _status(report, what):
    return next(i.status for i in report if i.what == what)


# --- round trip -----------------------------------------------------------


def test_round_trip_verifies_every_item(run, signer, tmp_path):
    root, rec = run
    out = build.write_record(rec.sign(signer), tmp_path / "rec.json")

    report = verify.verify_file(out, root=root)

    assert report.ok, report.lines()
    statuses = {i.what: i.status for i in report}
    for name in ("envelope", "signature", "canonical form", "statement", "check 'claims'"):
        assert statuses[name] == verify.OK, (name, statuses)
    for f in ("draft.md", "results.json", "refs.bib"):
        assert statuses[f"input {f}"] == verify.OK
    assert statuses["subject brief.md"] == verify.OK
    # No key pinned: identity is honestly unchecked, not "ok".
    assert statuses["signer identity"] == verify.UNCHECKED


def test_record_holds_what_the_run_did(run, signer):
    root, rec = run
    pred = _statement(rec.sign(signer))["predicate"]

    assert pred["queries"] == [
        {"source": "openalex", "query": "interatomic potentials", "hits": 12},
        {"source": "pubmed", "query": "interatomic potentials", "hits": 0},
    ]
    assert [p["depth"] for p in pred["papers"]] == ["full", "abstract"]
    assert {i["annotations"]["role"] for i in pred["inputs"]} == {"draft", "results", "bib"}
    assert pred["checks"][0]["summary"] == {"contradicted": 1, "supported": 1}
    assert pred["tool"]["name"] == "research-brief"
    assert pred["tool"]["attestation_version"]
    assert pred["signer"]["keyid"] == signer.keyid
    assert pred["run"]["id"] and pred["run"]["started_at"].endswith("Z")
    # Names are relative to the run's root, so the record is portable.
    assert all("/" not in i["name"] for i in pred["inputs"])


def test_verify_without_the_files_reports_them_unchecked_not_ok(run, signer, tmp_path):
    _, rec = run
    out = build.write_record(rec.sign(signer), tmp_path / "rec.json")
    empty = tmp_path / "elsewhere"
    empty.mkdir()

    report = verify.verify_file(out, root=empty)

    assert report.ok  # nothing failed...
    assert _status(report, "input draft.md") == verify.UNCHECKED  # ...but nothing passed either


def test_pinned_key_passes_for_its_own_records(run, signer, tmp_path):
    root, rec = run
    out = build.write_record(rec.sign(signer), tmp_path / "rec.json")
    pub = tmp_path / "pub.pem"
    pub.write_text(signer.public_pem())

    report = verify.verify_file(out, root=root, key=pub)

    assert report.ok
    assert _status(report, "signer identity") == verify.OK


# --- tampering ------------------------------------------------------------


def test_edited_verdict_fails_the_signature(run, signer):
    root, rec = run
    envelope = rec.sign(signer)
    st = _statement(envelope)
    st["predicate"]["checks"][0]["results"][1]["verdict"] = "supported"
    st["predicate"]["checks"][0]["summary"] = {"supported": 2}

    report = verify.verify(_reseal_payload(envelope, st), root=root)

    assert not report.ok
    assert _status(report, "signature") == verify.FAILED


def test_summary_that_disagrees_with_its_results_fails(run, signer):
    """Even a correctly signed record (a buggy or hostile producer) must not
    show a summary its own results contradict."""
    root, rec = run
    rec.checks[0]["summary"] = {"supported": 2}

    report = verify.verify(rec.sign(signer), root=root)

    assert _status(report, "signature") == verify.OK
    assert _status(report, "check 'claims'") == verify.FAILED
    assert not report.ok


def test_edited_input_file_fails(run, signer, tmp_path):
    root, rec = run
    out = build.write_record(rec.sign(signer), tmp_path / "rec.json")
    (root / "results.json").write_text('{"wer": 0.049}\n')

    report = verify.verify_file(out, root=root)

    assert not report.ok
    assert _status(report, "input results.json") == verify.FAILED
    assert _status(report, "signature") == verify.OK


def test_record_signed_by_another_key_but_naming_ours_fails(run, signer, tmp_path, monkeypatch):
    """Swap the embedded identity for another key's: the signature no longer
    verifies under the key the record names."""
    root, rec = run
    envelope = rec.sign(signer)
    other = keys.LocalKeySource(tmp_path / "other.pem").signer()
    st = _statement(envelope)
    st["predicate"]["signer"] = other.identity()

    report = verify.verify(_reseal_payload(envelope, st), root=root)

    assert _status(report, "signature") == verify.FAILED
    assert not report.ok


def test_wrong_pinned_key_fails(run, signer, tmp_path):
    root, rec = run
    out = build.write_record(rec.sign(signer), tmp_path / "rec.json")
    other = keys.LocalKeySource(tmp_path / "other.pem").signer()
    pub = tmp_path / "other-pub.pem"
    pub.write_text(other.public_pem())

    report = verify.verify_file(out, root=root, key=pub)

    assert _status(report, "signature") == verify.OK  # it IS validly signed...
    assert _status(report, "signer identity") == verify.FAILED  # ...by someone else
    assert not report.ok


def test_signature_replayed_under_another_payload_type_fails(run, signer):
    """DSSE signs the payload TYPE too, so the signature cannot be lifted
    onto the same bytes presented as something else."""
    root, rec = run
    envelope = rec.sign(signer) | {"payloadType": "application/json"}

    report = verify.verify(envelope, root=root)

    assert not report.ok
    assert _status(report, "envelope") == verify.FAILED


def test_garbage_is_reported_not_raised():
    for junk in ({}, [], {"payloadType": verify.PAYLOAD_TYPE, "payload": "%%%"}):
        report = verify.verify(junk)
        assert not report.ok


# --- canonicalization -----------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # RFC 8785 Appendix B / ES6 Number.prototype.toString
        (0.0, "0"),
        (-0.0, "0"),
        (1.0, "1"),
        (1e-7, "1e-7"),
        (0.000001, "0.000001"),
        (1e21, "1e+21"),
        (1e20, "100000000000000000000"),
        (1e16, "10000000000000000"),
        (5e-324, "5e-324"),
        (1.7976931348623157e308, "1.7976931348623157e+308"),
        (-1.5e-10, "-1.5e-10"),
        (333333333.3333333, "333333333.3333333"),
        (295147905179352830000.0, "295147905179352830000"),
        (0.053, "0.053"),
    ],
)
def test_jcs_numbers_follow_ecmascript(value, expected):
    assert canonical.canonicalize(value).decode() == expected


def test_jcs_sorts_keys_by_utf16_code_unit():
    """RFC 8785 section 3.2.3's own example: code-point order would put the
    emoji (U+1F600) after U+FB33; UTF-16 order puts it before."""
    obj = {
        "€": "Euro Sign",
        "\r": "Carriage Return",
        "דּ": "Hebrew Letter Dalet With Dagesh",
        "1": "One",
        "\U0001f600": "Emoji: Grinning Face",
        "\u0080": "Control",
        "ö": "Latin Small Letter O With Diaeresis",
    }
    keys_in_order = [k for k in json.loads(canonical.canonicalize(obj))]
    assert keys_in_order == ["\r", "1", "\u0080", "ö", "€", "\U0001f600", "דּ"]


def test_jcs_is_independent_of_input_key_order_and_whitespace():
    a = {"b": [1, 2.5, {"y": None, "x": True}], "a": "é\n"}
    b = json.loads(json.dumps({"a": "é\n", "b": [1, 2.5, {"x": True, "y": None}]}, indent=4))
    assert canonical.canonicalize(a) == canonical.canonicalize(b)
    assert canonical.canonicalize(a) == b'{"a":"\xc3\xa9\\n","b":[1,2.5,{"x":true,"y":null}]}'


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 2**53, {1: "x"}, object()])
def test_jcs_refuses_what_it_cannot_carry_exactly(bad):
    with pytest.raises((TypeError, ValueError)):
        canonical.canonicalize(bad)


def test_same_run_signs_to_the_same_payload(run, signer):
    """Stable bytes: signing twice yields an identical payload (ed25519 is
    deterministic, so the signature is identical too)."""
    _, rec = run
    rec.finished_at = "2026-10-17T00:00:00Z"
    first, second = rec.sign(signer), rec.sign(signer)
    assert first == second


# --- keys -----------------------------------------------------------------


def test_default_key_is_generated_once_with_private_permissions(key_home):
    first = keys.resolve_key_source().signer()
    second = keys.resolve_key_source().signer()
    path = keys.default_key_path()

    assert first.keyid == second.keyid
    assert path.is_relative_to(key_home)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_world_readable_key_is_refused(key_home):
    keys.resolve_key_source().signer()
    keys.default_key_path().chmod(0o644)
    with pytest.raises(keys.RecordKeyError, match="chmod 600"):
        keys.resolve_key_source().signer()


def test_env_key_is_used_and_never_generated(key_home, tmp_path, monkeypatch):
    missing = tmp_path / "typo.pem"
    monkeypatch.setenv(keys.ENV_KEY, str(missing))
    with pytest.raises(keys.RecordKeyError, match="does not exist"):
        keys.resolve_key_source().signer()
    assert not missing.exists()

    made = keys.LocalKeySource(tmp_path / "researcher.pem").signer()
    monkeypatch.setenv(keys.ENV_KEY, str(tmp_path / "researcher.pem"))
    assert keys.resolve_key_source().signer().keyid == made.keyid


# --- spec / generic API ---------------------------------------------------


def test_from_spec_builds_the_same_shape(tmp_path, signer):
    (tmp_path / "brief.md").write_text("# B\n")
    (tmp_path / "sources.md").write_text("- a paper\n")
    spec = {
        "tool": "research-brief",
        "queries": [{"source": "openalex", "query": "q", "hits": 3}],
        "papers": [{"id": "doi:10.1/x", "depth": "title"}],
        "inputs": [{"path": "sources.md", "role": "sources"}],
        "outputs": [{"path": "brief.md"}],
        "checks": [{"check": "check_brief", "results": [{"subject": "row 1", "verdict": "cited"}]}],
    }
    report = verify.verify(build.from_spec(spec, root=tmp_path).sign(signer), root=tmp_path)
    assert report.ok, report.lines()
    assert report.statement["subject"][0]["name"] == "brief.md"
    assert report.statement["predicate"]["checks"][0]["summary"] == {"cited": 1}


@pytest.mark.parametrize(
    "spec",
    [
        {"tool": "t", "paper": []},  # misspelt field must not vanish silently
        {"tool": "t", "papers": [{"id": "x", "depth": "skimmed"}]},
        {"tool": "t", "queries": [{"source": "s", "query": "q", "hits": -1}]},
        {"tool": "t", "inputs": [{"path": "nope.md"}]},
        {"tool": "t", "inputs": [{"role": "draft"}]},
    ],
)
def test_from_spec_refuses_malformed_specs(spec, tmp_path):
    with pytest.raises(ValueError):
        build.from_spec(spec, root=tmp_path)


def test_a_record_must_name_a_file(signer, tmp_path):
    with pytest.raises(ValueError, match="at least one file"):
        build.RunRecorder("t", root=tmp_path).sign(signer)


def test_write_record_refuses_to_overwrite(run, signer, tmp_path):
    _, rec = run
    out = build.write_record(rec.sign(signer), tmp_path / "rec.json")
    with pytest.raises(FileExistsError):
        build.write_record(rec.sign(signer), out)
    build.write_record(rec.sign(signer), out, force=True)


# --- the verifier stays standalone ----------------------------------------


def test_verifier_imports_nothing_heavy():
    """verify.py and canonical.py must run from a bare install: stdlib,
    cryptography, and each other -- no sqlite3, no attestation.db, no model."""
    pkg = Path(verify.__file__).parent
    for name in ("verify.py", "canonical.py"):
        tree = ast.parse((pkg / name).read_text())
        imported = {
            (n.module or "") if isinstance(n, ast.ImportFrom) else a.name
            for n in ast.walk(tree)
            if isinstance(n, ast.Import | ast.ImportFrom)
            for a in n.names
        }
        attest = {m for m in imported if m.startswith("attestation")}
        assert attest <= {"attestation.run_record.canonical"}, (name, attest)
        assert "sqlite3" not in imported


def test_python_m_verify_runs(run, signer, tmp_path, capsys):
    root, rec = run
    out = build.write_record(rec.sign(signer), tmp_path / "rec.json")
    assert verify.main([str(out), "--root", str(root), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    (root / "draft.md").write_text("changed\n")
    assert verify.main([str(out), "--root", str(root)]) == 1


# --- hooks: attest claims, cite.check, MCP --------------------------------


def _draft(tmp_path):
    work = tmp_path / "paper"
    work.mkdir()
    (work / "paper.md").write_text(
        "WER 5.3%.\n<!-- claim: ablation/stack_4 metric=wer value=0.053 cite=nobody2020 -->\n"
    )
    return work


def test_attest_claims_record(tmp_path, monkeypatch, key_home, capsys):
    from attestation.cli import main

    monkeypatch.setenv("ATTEST_DB", str(tmp_path / "t.db"))
    monkeypatch.chdir(tmp_path)
    work = _draft(tmp_path)
    out = tmp_path / "claims-record.json"

    main(["claims", str(work), "--record", str(out)])

    report = verify.verify_file(out, root=work)
    assert report.ok, report.lines()
    checks = {c["check"]: c for c in report.statement["predicate"]["checks"]}
    assert checks["claims"]["summary"] == {"unsupported": 1}
    assert checks["cite.check"]["summary"] == {"uncited": 1}
    assert report.statement["subject"][0]["name"] == "paper.md"
    assert report.statement["predicate"]["tool"]["name"] == "attest claims"

    assert main(["record", "verify", str(out), "--root", str(work)]) == 0
    assert main(["claims", str(work), "--record", str(out)]) == 1  # never overwrites


def test_attest_record_create_and_key(tmp_path, monkeypatch, key_home, capsys):
    from attestation.cli import main

    (tmp_path / "brief.md").write_text("# B\n")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"tool": "research-brief", "outputs": [{"path": "brief.md"}]}))
    out = tmp_path / "rec.json"

    assert main(["record", "create", "--spec", str(spec), "--out", str(out)]) == 1  # cwd root
    assert (
        main(["record", "create", "--spec", str(spec), "--out", str(out), "--root", str(tmp_path)])
        == 0
    )
    pub = tmp_path / "pub.pem"
    assert main(["record", "key", "--export", str(pub)]) == 0
    assert main(["record", "verify", str(out), "--root", str(tmp_path), "--key", str(pub)]) == 0


def test_cite_check_writes_a_record(tmp_path, monkeypatch, key_home):
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    monkeypatch.chdir(tmp_path)
    from attestation.mcp.citation import _check

    work = _draft(tmp_path)
    out = tmp_path / "cite-record.json"

    result = _check(path=str(work / "paper.md"), record=str(out))

    assert result["ok"] is True, result
    assert result["record"]["path"] == str(out)
    report = verify.verify_file(out, root=work)
    assert report.ok
    assert report.statement["predicate"]["checks"][0]["check"] == "cite.check"
    assert "record" not in _check(path=str(work / "paper.md"))


def test_mcp_attest_and_verify(tmp_path, key_home):
    from attestation.mcp.run_record import RecordSpec, _attest, _verify

    (tmp_path / "brief.md").write_text("# B\n")
    spec = RecordSpec(
        tool="research-brief",
        papers=[{"id": "doi:10.1/x", "depth": "abstract"}],
        outputs=[{"path": "brief.md"}],
    )
    out = tmp_path / "rec.json"

    made = _attest(spec, str(out), root=str(tmp_path))
    assert made["ok"] is True, made

    checked = _verify(str(out), root=str(tmp_path), keyid=made["record"]["keyid"])
    assert checked["verified"] is True, checked
    assert checked["cannot_check"]

    (tmp_path / "brief.md").write_text("# edited\n")
    tampered = _verify(str(out), root=str(tmp_path))
    assert tampered["verified"] is False
    assert "subject brief.md" in tampered["message"]

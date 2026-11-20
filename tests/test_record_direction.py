"""A core record's own metric direction ranks a family no declaration names.

See docs/superpowers/specs/2026-11-20-record-declared-direction-design.md:
the declaration (built-in table + metric_direction.toml) wins and a record
that disagrees with it is a caveat; absent a declaration, every run reporting
the metric must declare the same direction; anything less refuses, naming the
runs. Never from the name.
"""

import json
import sqlite3
from pathlib import Path

import attestation_schema as schema
import pytest
from test_response_size import HARD_RESPONSE_CEILING

from attestation import cli, ledger, prereg
from attestation.db import get_db

METRIC = "paired_ddE.mean"  # in no built-in table: only a record can direct it


def _native(ws: Path, run_id: str, value: float, direction, *, metric=METRIC, family="fam"):
    """A native core record in its own run directory (what smeltery writes)."""
    record = {
        "schema": schema.SCHEMA,
        "run_id": run_id,
        "family": family,
        "arm": run_id,
        "status": "complete",
        "started_at": "2026-11-20T10:00:00.000000Z",
        "finished_at": None,
        "producer": {"name": "test", "version": None},
        "code": {
            "git_sha": None,
            "dirty": None,
            "repo": None,
            "version": None,
            "provenance": "unknown",
            "evidence": None,
        },
        "config": {"arm": run_id},
        "data": [],
        "metrics": [
            {
                "name": metric,
                "value": value,
                "direction": direction,
                "step": None,
                "split": "test",
                "unit": None,
                "labels": {},
            }
        ],
        "history": None,
        "seeds": None,
        "dependencies": [],
        "ext": {},
    }
    schema.write(schema.from_dict(record), ws / "proj" / "runs" / run_id)


def _scan(tmp_path, runs: dict[str, tuple[float, str | None]], **kw) -> sqlite3.Connection:
    ws = tmp_path / "ws"
    for run_id, (value, direction) in runs.items():
        _native(ws, run_id, value, direction, **kw)
    conn = get_db(tmp_path / "t.db")
    ledger.scan(conn, ws)
    return conn


def _declare(tmp_path, monkeypatch, direction: str) -> None:
    path = tmp_path / "metric_direction.toml"
    path.write_text(f'[metric_direction]\n"{METRIC}" = "{direction}"\n')
    monkeypatch.setenv(ledger.METRIC_DIRECTION_PATH_ENV, str(path))


def test_records_declaring_min_rank_without_any_declaration(tmp_path):
    conn = _scan(tmp_path, {"a": (0.5, "min"), "b": (0.2, "min"), "c": (0.9, "min")})

    out = ledger.compare(conn, "fam", metric=METRIC)

    assert out["direction"] == "lower_is_better"
    assert out["direction_source"] == "records"
    assert [a["name"] for a in out["arms"]] == ["b", "a", "c"]
    assert out["winner"] == "b"
    assert not any("disagree" in c for c in out["caveats"])


def test_records_declaring_max_rank_the_other_way(tmp_path):
    conn = _scan(tmp_path, {"a": (0.5, "max"), "b": (0.2, "max")})
    out = ledger.compare(conn, "fam", metric=METRIC)
    assert out["direction"] == "higher_is_better"
    assert out["winner"] == "a"


def test_discovery_picks_a_metric_only_records_direct(tmp_path):
    """compare with no metric named: a smeltery-style family used to refuse
    with "no metric with a known direction"."""
    conn = _scan(tmp_path, {"a": (0.5, "min"), "b": (0.2, "min")})
    out = ledger.compare(conn, "fam")
    assert out["metric"] == METRIC
    assert out["direction_source"] == "records"


def test_the_stored_direction_is_the_records_word(tmp_path):
    conn = _scan(tmp_path, {"a": (0.5, "min")})
    rows = conn.execute("SELECT metric, direction FROM run_metrics").fetchall()
    assert [tuple(r) for r in rows] == [(METRIC, "min")]


def test_an_agreeing_declaration_is_silent_and_keeps_the_old_keys(tmp_path, monkeypatch):
    _declare(tmp_path, monkeypatch, "lower_is_better")
    conn = _scan(tmp_path, {"a": (0.5, "min"), "b": (0.2, "min")})

    out = ledger.compare(conn, "fam", metric=METRIC)

    assert "direction_source" not in out
    assert out["winner"] == "b"
    assert not any("disagree" in c for c in out["caveats"])


def test_a_declaration_overrides_disagreeing_records_with_a_caveat(tmp_path, monkeypatch):
    _declare(tmp_path, monkeypatch, "higher_is_better")
    conn = _scan(tmp_path, {"a": (0.5, "min"), "b": (0.2, "min"), "c": (0.1, None)})

    out = ledger.compare(conn, "fam", metric=METRIC)

    assert out["direction"] == "higher_is_better"
    assert "direction_source" not in out
    assert out["winner"] == "a"
    # first: it bears on the ranking itself; the silent run is no dissent
    assert out["caveats"][0].startswith(f"{METRIC} is declared higher_is_better")
    assert out["caveats"][0].endswith(
        "but its records disagree: a, b declare min -- ranked by the declaration"
    )


def test_the_built_in_table_is_a_declaration_too(tmp_path):
    """`wer` is lower_is_better by definition; a record saying max is the one
    likelier wrong, so the table ranks and the record is a caveat."""
    conn = _scan(tmp_path, {"a": (0.5, "max"), "b": (0.2, None)}, metric="wer")
    out = ledger.compare(conn, "fam", metric="wer")
    assert out["direction"] == "lower_is_better"
    assert out["winner"] == "b"
    assert "a declares max" in out["caveats"][0]


def test_contradictory_records_refuse_naming_the_runs(tmp_path):
    conn = _scan(tmp_path, {"a": (0.5, "min"), "b": (0.2, "max"), "c": (0.3, "min")})
    with pytest.raises(ValueError) as exc:
        ledger.compare(conn, "fam", metric=METRIC)
    message = str(exc.value)
    assert "refusing to rank: its records disagree (b declares max; a, c declare min)" in message
    assert "[metric_direction]" in message


def test_partly_declaring_records_refuse_naming_the_silent_runs(tmp_path):
    """Silence is not consent: the core schema's null means "did not say"."""
    conn = _scan(tmp_path, {"a": (0.5, "min"), "b": (0.2, None)})
    with pytest.raises(ValueError, match=r"do not all declare one \(a declares min; b declares"):
        ledger.compare(conn, "fam", metric=METRIC)


def test_contradictory_records_are_not_a_discovery_candidate(tmp_path):
    conn = _scan(tmp_path, {"a": (0.5, "min"), "b": (0.2, "max")})
    with pytest.raises(ValueError, match="no metric with a known direction"):
        ledger.compare(conn, "fam")


def test_records_with_no_direction_refuse_with_the_old_sentence(tmp_path):
    """Legacy and silent native records: unchanged, byte for byte."""
    conn = _scan(tmp_path, {"a": (0.5, None), "b": (0.2, None)})
    with pytest.raises(ValueError) as exc:
        ledger.compare(conn, "fam", metric=METRIC)
    assert str(exc.value) == ledger.unknown_direction_message(METRIC)


def test_names_in_a_note_are_capped():
    runs = {f"r{i:02d}": "max" for i in range(8)}
    _, _, note = ledger.resolve_direction("wer", ledger.METRIC_DIRECTION, runs)
    assert "r00, r01, r02, r03, r04 (+3 more) declare max" in note


def test_resolve_never_reads_the_name():
    """A metric called `loss_something` with no declaration and no record
    stays undirected -- the stem rule is the table's, not an inference."""
    assert ledger.resolve_direction("score", {}, {"a": None}) == (None, None, None)
    assert ledger.resolve_direction("score", {}, {}) == (None, None, None)


def test_legacy_family_output_is_unchanged():
    """The example workspace's legacy readers emit direction null: kdsweep
    keeps its keys and its pinned order."""
    root = Path(__file__).resolve().parent.parent / "examples" / "workspace"
    conn = get_db(":memory:")
    ledger.scan(conn, root)
    out = ledger.compare(conn, "kdsweep", metric="wer")
    assert "direction_source" not in out
    assert (
        conn.execute("SELECT COUNT(*) FROM run_metrics WHERE direction IS NOT NULL").fetchone()[0]
        == 0
    )


def test_the_cli_header_names_the_records_only_when_they_decided(tmp_path, capsys, monkeypatch):
    conn = _scan(tmp_path, {"a": (0.5, "min"), "b": (0.2, "min")})
    cli._print_compare(ledger.compare(conn, "fam", metric=METRIC))
    assert f"fam — ranked by {METRIC} (lower_is_better, declared by its records)" in (
        capsys.readouterr().out
    )

    _declare(tmp_path, monkeypatch, "lower_is_better")
    cli._print_compare(ledger.compare(conn, "fam", metric=METRIC))
    assert f"fam — ranked by {METRIC} (lower_is_better)\n" in capsys.readouterr().out


def test_migration_017_adds_the_column_to_a_pre_017_table_and_is_idempotent():
    from attestation import db

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE run_metrics(run_id INTEGER NOT NULL, metric TEXT NOT NULL,"
        " value REAL NOT NULL, step INTEGER, split TEXT)"
    )
    conn.execute("INSERT INTO run_metrics VALUES (1, 'wer', 0.2, NULL, 'test')")
    db._migration_017_add_run_metrics_direction(conn)
    db._migration_017_add_run_metrics_direction(conn)  # a no-op on the current shape
    rows = conn.execute("SELECT metric, direction FROM run_metrics").fetchall()
    assert [tuple(r) for r in rows] == [("wer", None)]  # scanned before 017 = NULL
    assert db.SCHEMA_VERSION == 17


def test_the_mcp_payload_carries_the_source_cheaply(tmp_path, monkeypatch):
    """runs.compare is a composition tool (size follows the family), so what
    is pinned is that the direction source and the disagreement caveat cost
    little, and that a realistic family stays under the hard ceiling."""
    from attestation.mcp import provenance as pv

    long = "x" * 40
    runs = {f"{long}{i:02d}": (0.1 * i, "min") for i in range(8)}
    _scan(tmp_path, runs)
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))

    by_records = pv._compare(family="fam", metric=METRIC)
    _declare(tmp_path, monkeypatch, "lower_is_better")
    agreeing = pv._compare(family="fam", metric=METRIC)
    _declare(tmp_path, monkeypatch, "higher_is_better")
    disagreeing = pv._compare(family="fam", metric=METRIC)

    size = {
        k: len(json.dumps(v, indent=2))
        for k, v in {"records": by_records, "agree": agreeing, "disagree": disagreeing}.items()
    }
    assert by_records["direction_source"] == "records" and "direction_source" not in agreeing
    assert size["records"] - size["agree"] < 60, size
    # names capped at five, whatever the family size
    assert "(+3 more) declare min" in disagreeing["caveats"][0]
    # five 42-char names plus the TOML path the caveat names (a pytest tmp
    # path here; measured 455)
    assert size["disagree"] - size["agree"] < 600, size
    assert max(size.values()) <= HARD_RESPONSE_CEILING, size


# --- preregistration (spec Decision 4) -------------------------------------


def test_prereg_add_still_requires_a_declaration(tmp_path):
    """A registration precedes its runs: there is no record to read yet, and
    reading the ledger would accept only post-hoc registrations."""
    _scan(tmp_path, {"a": (0.5, "min"), "b": (0.2, "min")})
    with pytest.raises(ValueError) as exc:
        prereg.refuse_undeclared(METRIC)
    assert str(exc.value) == ledger.unknown_direction_message(METRIC)


def test_prereg_evaluation_ranks_through_record_directions(tmp_path):
    """`prereg check` reads ledger.compare, which now ranks by records."""
    conn = _scan(tmp_path, {"a": (0.5, "min"), "b": (0.2, "min")})
    hypothesis = prereg.Hypothesis(
        family="fam", metric=METRIC, expect="b < a", by=0.1, allow_single_run=True
    )
    verdict = prereg.evaluate(hypothesis, ledger.compare(conn, "fam", metric=METRIC))
    assert verdict.verdict == prereg.VerdictKind.CONFIRMED, verdict

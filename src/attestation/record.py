"""`attest runs record`'s domain: derive the ledger's on-disk shape, don't
transcribe it by hand.

Measured 2026-09-01 (see `docs/superpowers/specs/2026-09-01-attest-record-
design.md`): the `attestation-record` skill's five-step manual procedure --
one JSON per arm, a config with the exact same stem, a `[metric_direction]`
entry for any metric the ledger does not know, a `corpora.toml` when
detection cannot see the corpus, then scan and compare -- is followed at
>=0.91 on the file-shape steps by a small local model, and at 0/15 on the
declaration step. A model that must *remember* to declare a direction will
sometimes forget; a command that *refuses* to write until it has one cannot.

Two pure functions and one I/O function, deliberately split so the manifest
`--dry-run` prints is the exact same dict the eval scorer builds from and
`write()` materialises: `plan()` returns `{relpath: content}` for the results,
configs, and (declared) corpus files an invocation would leave, over plain
data; `undeclared()` names which metrics still need a `--direction`, so the
CLI can refuse before calling `plan()` at all; `write()` is the only function
here that touches a filesystem, and only ever creates NEW files unless told
to overwrite.

Neither `sqlite3` nor `attestation.llm` is imported here (see
`tests/test_architecture.py`'s `test_domain_reaches_models_only_through_ports`
and this module's own `test_record.py::test_record_module_avoids_sqlite3_
and_llm`): recording a run needs no database read and no model call, and
`--scan` -- which does need both -- is `cli.py`'s job, not this module's.
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

# The same grammar `claims.py::_FIELD_RE` uses for a claim's `key=value`
# fields (`re.compile(r"(\w+)=([^\s]+)")`): a metric name is `\w+`. Matching
# it here, rather than importing claims.py's private `_FIELD_RE`, keeps this
# module's only cross-domain dependency at zero -- the shapes are identical
# by construction, and `tests/test_record.py` pins that they stay identical.
METRIC_NAME_RE = re.compile(r"\w+")

# Recognised by `ledger_adapters/generic.py`'s `discover()` -- the first
# entries in `RESULT_DIRS`/`CONFIG_DIRS` -- so a manifest this module plans
# is guaranteed to be read back the way it was written.
RESULTS_DIR = "results"
CONFIGS_DIR = "configs"
CORPORA_FILENAME = "corpora.toml"


def validate_metric_name(name: str) -> None:
    """Raise `ValueError` if `name` is not a valid metric name.

    Same grammar the claim parser accepts (`\\w+`) so a typo here cannot
    write a metric `runs.compare`/`attest claims` would then treat as
    unparseable or, worse, silently split into two fields.
    """
    if not METRIC_NAME_RE.fullmatch(name):
        raise ValueError(f"invalid metric name {name!r} -- must match {METRIC_NAME_RE.pattern!r}")


def parse_metric_value(raw: str) -> float:
    """`raw` (a `METRIC=VALUE` argument's value half) as a float, or raise.

    Values are numbers, never strings: `runs.compare` ranks on `float`, so a
    non-numeric value recorded now would only refuse later, at comparison
    time, with a less specific error than refusing it here at the source.
    """
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"metric value {raw!r} is not a number") from exc


def undeclared(arms: dict[str, dict[str, float]], known_directions: dict[str, str]) -> list[str]:
    """Metric names across `arms` that `known_directions` has no entry for.

    `arms` is `{arm_name: {metric: value}}`. `known_directions` is normally
    `ledger.metric_directions()` merged with any `--direction` the caller is
    about to declare -- passing it in rather than reading the ladder here
    keeps this function pure and testable with literal dicts. Sorted and
    de-duplicated so the refusal message names each metric once.
    """
    names: set[str] = set()
    for metrics in arms.values():
        names.update(metrics)
    return sorted(name for name in names if name not in known_directions)


def _config_yaml(
    *, family: str, arm: str, corpus: str | None, recorded_at: str, config: dict[str, str] | None
) -> str:
    """Flat, unquoted `key: value` lines -- the shape `ledger_adapters.
    generic._yaml_scalars` reads, and the only shape this codebase writes
    YAML in (see that function's own docstring: no YAML parser is a declared
    dependency, so nothing here writes anything a hand-rolled reader can't
    read back). Provenance only: never a metric value, so `discover()` never
    mistakes this file for a second run of the same arm.
    """
    lines = [f"family: {family}", f"arm: {arm}"]
    if corpus:
        lines.append(f"corpus: {corpus}")
    lines.append(f"recorded_at: {recorded_at}")
    for key, value in (config or {}).items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n"


def plan(
    family: str,
    arms: dict[str, dict[str, float]],
    *,
    corpus: str | None = None,
    directions: dict[str, str] | None = None,
    config: dict[str, str] | None = None,
    recorded_at: str | None = None,
    known_directions: dict[str, str] | None = None,
) -> dict[str, str]:
    """The manifest `{relpath: content}` one `attest runs record` invocation
    would leave on disk -- pure over plain data, called by both `--dry-run`
    (prints it) and `write()` (materialises it), so the two can never
    disagree about what "the files this command writes" means.

    `arms` is `{arm_name: {metric: value}}`, already parsed to float and
    already validated (`plan` does not re-validate; `undeclared`/
    `_validate_metric`/`parse_metric_value` are the callers' job, so a
    refusal happens before any manifest is even built). `directions` is the
    `{metric: "lower_is_better"|"higher_is_better"}` pairs this invocation is
    declaring via `--direction`; `known_directions` is what the ledger
    already knows (built-in plus its TOML file) -- passed in only so `plan`
    can decide whether a `metric_direction.toml` entry is redundant, never to
    re-run the undeclared check itself (the caller already refused if it
    would fail).

    `recorded_at` defaults to now (UTC, ISO-8601) if not given; a caller
    (the eval, a test) that needs a byte-exact manifest passes it explicitly.
    """
    recorded_at = recorded_at or datetime.now(UTC).isoformat()
    directions = directions or {}
    known_directions = known_directions or {}

    files: dict[str, str] = {}
    for arm, metrics in arms.items():
        stem = f"{family}_{arm}"
        files[f"{RESULTS_DIR}/{stem}.json"] = json.dumps(metrics, indent=2) + "\n"
        files[f"{CONFIGS_DIR}/{stem}.yaml"] = _config_yaml(
            family=family, arm=arm, corpus=corpus, recorded_at=recorded_at, config=config
        )

    if corpus:
        files[CORPORA_FILENAME] = (
            f'[corpus.{corpus}]\nsource = "{corpus}"\n\n[assign.family]\n{family} = "{corpus}"\n'
        )

    new_directions = {
        metric: direction
        for metric, direction in directions.items()
        if metric not in known_directions
    }
    if new_directions:
        lines = ["[metric_direction]"]
        for metric, direction in new_directions.items():
            lines.append(f'{metric} = "{direction}"')
        files["metric_direction.toml"] = "\n".join(lines) + "\n"

    return files


def write(root: Path, manifest: dict[str, str], *, force: bool = False) -> list[Path]:
    """Materialise `manifest` (a `plan()` result) under `root`. The only I/O
    in this module.

    New files only: every target is checked for existence BEFORE any file is
    written, and the whole call refuses (raising `FileExistsError`, listing
    every existing target) rather than writing some of the manifest and
    refusing partway through -- a caller left with half a manifest on disk
    cannot tell "this run wasn't recorded" from "this run was recorded
    wrong". `force=True` skips the check and overwrites.

    A TOML target (`corpora.toml`, `metric_direction.toml`) is the one
    exception `merge_toml_table` exists for: those are meant to accumulate
    across calls, so the CLI merges into them itself rather than routing
    them through this refuse-or-overwrite path -- `write()` only ever sees
    the per-arm results/configs files that must each be new.
    """
    targets = {root / relpath: content for relpath, content in manifest.items()}
    if not force:
        existing = sorted(str(p) for p in targets if p.exists())
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing file(s) (pass --force to overwrite): "
                + ", ".join(existing)
            )
    written = []
    for path, content in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        written.append(path)
    return written


def _table_at(doc: dict, table: str) -> dict:
    """`doc[table]`, walking a dotted `table` path (`"assign.family"`)
    through nested dicts. Missing at any segment reads as an empty table --
    a file that has never declared `[table]` at all is not a conflict, it
    is simply nothing to check entries against."""
    current = doc
    for segment in table.split("."):
        current = current.get(segment, {}) if isinstance(current, dict) else {}
    return current


def _table_conflicts(current: dict, entries: dict[str, str]) -> dict[str, tuple[str, str]]:
    """`{key: (old, new)}` for every entry in `entries` that disagrees with
    an already-present value in `current` -- an identical value is not a
    conflict, only a differing one is."""
    return {
        key: (current[key], value)
        for key, value in entries.items()
        if key in current and current[key] != value
    }


def _replace_existing_keys(text: str, replacing: dict[str, str]) -> str:
    """`text` with each of `replacing`'s keys' existing `key = ...` line
    swapped for its new value, in place -- never appended a second time.
    `tomllib`'s own parser treats a key repeated under one table as
    malformed TOML ("Cannot overwrite a value"), so a naive append-only
    insertion for a key that's already there would corrupt the very file
    `--force` is meant to fix; this is only reachable with `force=True`,
    since `merge_toml_table`'s conflict check already refused any differing
    value without it."""
    out = text
    for key, value in replacing.items():
        pattern = re.compile(rf"(?m)^{re.escape(key)}\s*=\s*.*$")
        out = pattern.sub(f'{key} = "{value}"', out, count=1)
    return out


def _append_new_keys(text: str, table: str, appending: dict[str, str]) -> str:
    """`text` with `appending`'s keys added under `[table]` -- inserted
    right after an existing header if `[table]` is already present,
    otherwise the header and its entries are appended at the end."""
    if not appending:
        return text
    lines = "\n".join(f'{k} = "{v}"' for k, v in appending.items())
    header = f"[{table}]"
    if header in text:
        # Insert right after the table header, before its first existing
        # entry (or before the next table/EOF if the table is now empty --
        # unreachable in this module's own callers, but correct regardless).
        idx = text.index(header) + len(header)
        return text[:idx] + "\n" + lines + text[idx:]
    out = text
    if out and not out.endswith("\n\n"):
        out += "\n" if out.endswith("\n") else "\n\n"
    return out + header + "\n" + lines + "\n"


def _ensure_trailing_newline(text: str) -> str:
    """`text` with exactly one trailing newline, unless it is empty (an
    absent file stays `""`, not `"\\n"`)."""
    if not text or text.endswith("\n"):
        return text
    return text + "\n"


def _refuse_conflicts(table: str, current: dict, entries: dict[str, str]) -> None:
    """Raise `ValueError` naming every entry in `entries` that disagrees
    with an already-present value under `[table]` -- `merge_toml_table`'s
    `force=False` path, split out so that function's own body stays a
    single straight line: compute the table, maybe refuse, compute what's
    new, write it."""
    conflicts = _table_conflicts(current, entries)
    if conflicts:
        detail = ", ".join(f"{k}: {old!r} -> {new!r}" for k, (old, new) in conflicts.items())
        raise ValueError(
            f"refusing to overwrite existing [{table}] entr{'y' if len(conflicts) == 1 else 'ies'}"
            f" without --force: {detail}"
        )


def merge_toml_table(
    existing_text: str, table: str, entries: dict[str, str], *, force: bool
) -> str:
    """`existing_text` (a TOML file's current content, `""` if absent) with
    `entries` merged into `[table]`, keeping every foreign entry (a
    different table, or a different key within `[table]`) untouched.
    `table` may be dotted (`"assign.family"`, `"corpus.wikitext2"`) to reach
    a nested table -- both `corpora.toml` tables this command writes
    (`[corpus.<name>]`, `[assign.family]`) need that; `[metric_direction]`
    does not.

    Refuses (`ValueError`) when an entry already present under `[table]`
    disagrees with the new value and `force` is false -- a declaration is a
    promise about meaning, not a cache entry, so a silent overwrite could
    flip a metric's direction or a corpus's source out from under whoever
    declared it first. An identical existing value is not a conflict: two
    `--direction wer=lower_is_better` calls for the same family are not a
    disagreement, just redundant.

    Deliberately not a general TOML writer: this codebase has no TOML-write
    dependency (only stdlib `tomllib`, read-only -- see `_config_shape`'s and
    `_yaml_scalars`'s docstrings on why an undeclared parser dependency is
    avoided here), and both files this writes (`corpora.toml`,
    `metric_direction.toml`) hold only flat `key = "value"` tables, however
    deeply nested the table itself is. Round-trips through `tomllib.loads` to
    detect the conflict, then re-emits textually rather than re-serialising
    the whole document, so an unrelated table's comments and formatting
    survive untouched.
    """
    import tomllib

    doc = tomllib.loads(existing_text) if existing_text.strip() else {}
    current = _table_at(doc, table)

    if not force:
        _refuse_conflicts(table, current, entries)

    to_add = {k: v for k, v in entries.items() if current.get(k) != v}
    if not to_add:
        return _ensure_trailing_newline(existing_text)

    out = _ensure_trailing_newline(existing_text)
    out = _replace_existing_keys(out, {k: v for k, v in to_add.items() if k in current})
    return _append_new_keys(out, table, {k: v for k, v in to_add.items() if k not in current})

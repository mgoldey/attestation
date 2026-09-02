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


# `family` and an arm's `name` both become PATH SEGMENTS (`plan()` builds
# `results/<family>_<arm>.json`) -- unlike a metric name, which only ever
# becomes a JSON key. `\w+` alone would still allow a lone `.` or `..`
# component if dots were permitted at all; this grammar additionally
# forbids `/` (no traversal via a literal separator) and a leading `.`
# (blocks `.`/`..` outright, and a dotfile stem), while still allowing the
# `.`/`-` a real family or arm name commonly carries (`gpt-4.1`, `v1.2`).
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def validate_name(name: str, *, label: str) -> None:
    """Raise `ValueError` if `name` is unsafe as a path segment.

    `family` and an arm's `name` are both interpolated directly into a
    relpath by `plan()` (`results/<family>_<arm>.json`) and then joined
    under `root` by `write()` -- so `family="../../victim/asr"` or an arm
    named `"../../victim/pwned"` walks `write()`'s `root / relpath` clean
    out of `root`. Checked here rather than only in `write()` because a
    caller (an agent, in the MCP tool's case) should be refused before
    `plan()` even builds a manifest naming a path outside the workspace --
    the refusal message should point at the bad NAME, not a confusing
    "already exists" one root-escape could produce by colliding with an
    unrelated file. `write()` still asserts containment as a second line of
    defence (see its own docstring) in case a caller reaches it some other
    way.
    """
    if not NAME_RE.fullmatch(name):
        raise ValueError(f"invalid {label} {name!r} -- must match {NAME_RE.pattern!r}")


VALID_DIRECTIONS = ("lower_is_better", "higher_is_better")


def validate_direction(metric: str, direction: str) -> None:
    """Raise `ValueError` if `direction` is not one of the two the ledger
    ranks by.

    `ledger._compare`'s `rank_key` treats anything other than the literal
    string `"lower_is_better"` as higher-is-better -- so an unvalidated
    typo (`"lower_is_bettr"`, or a wholly made-up value) does not refuse,
    it silently ranks backwards, which is exactly the failure the
    `unknown_direction_message` refusal exists to prevent. The CLI already
    validates this (`cli._parse_record_args`); this is the same check
    available to `mcp.provenance`, which does not otherwise share that
    parsing path.
    """
    if direction not in VALID_DIRECTIONS:
        raise ValueError(
            f"invalid direction {direction!r} for metric {metric!r} -- must be one of"
            f" {VALID_DIRECTIONS}"
        )


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

    Second line of defence against a `relpath` that escapes `root`: even
    though `validate_name` is meant to catch a `..`/`/`-carrying `family`
    or arm name before `plan()` ever builds a manifest, `write()` asserts
    every resolved target is still inside `root` (`Path.relative_to`) --
    belt and braces, since a manifest could in principle reach here some
    other way, and root-escape is a write-outside-the-workspace bug, not
    an ordinary refusal to paper over.
    """
    root = Path(root).resolve()
    targets = {root / relpath: content for relpath, content in manifest.items()}
    escaped = sorted(str(p) for p in targets if not p.resolve().is_relative_to(root))
    if escaped:
        raise ValueError(f"refusing to write outside root {root}: {', '.join(escaped)}")
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


def _ensure_trailing_newline(text: str) -> str:
    """`text` with exactly one trailing newline, unless it is empty (an
    absent file stays `""`, not `"\\n"`)."""
    if not text or text.endswith("\n"):
        return text
    return text + "\n"


def _leading_comment_block(text: str) -> str:
    """Consecutive `#`-prefixed (or blank) lines at the very TOP of `text`,
    before the first table header -- the one piece of hand-written text
    this module's dict-level merge cannot recover from `tomllib.loads`
    (comments are not part of the parsed structure at all). Preserved
    verbatim at the head of a re-emitted file; anything else -- an inline
    comment beside a `key = value` line, or one directly above a `[table]`
    header partway through the file -- is a hand-editing convenience this
    module does not promise to keep, the same limit a "no TOML-write
    dependency" implementation already had for OTHER hand formatting
    (blank-line spacing between tables, alignment)."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            lines.append(line)
            continue
        break
    return ("\n".join(lines) + "\n") if lines else ""


def _set_at(doc: dict, table: str, entries: dict[str, str]) -> None:
    """`doc[table] = {**doc.get(table, {}), **entries}`, walking a dotted
    `table` path (`"assign.family"`) and creating intermediate dicts as
    needed -- the merge half of `_table_at`'s read."""
    current = doc
    segments = table.split(".")
    for segment in segments[:-1]:
        current = current.setdefault(segment, {})
    current.setdefault(segments[-1], {}).update(entries)


def _emit_table(path: tuple[str, ...], values: dict[str, str]) -> str:
    """One `[dotted.path]` header plus its flat `key = "value"` lines --
    the only table shape `toml_tables()` ever hands `merge_toml_table` (a
    dotted path to a table whose own values are all strings), so this is
    the whole emitter this codebase's two TOML files need."""
    header = f"[{'.'.join(path)}]"
    lines = "\n".join(f'{k} = "{v}"' for k, v in values.items())
    return f"{header}\n{lines}\n"


def _leaf_tables(doc: dict, prefix: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], dict]]:
    """Every leaf table in `doc` (a dict whose own values are all strings,
    the shape `toml_tables()`'s walk produces) as `(dotted_path, values)`,
    in dict insertion order -- which is FIRST-SEEN order for a table
    already in `existing_text` (Python dicts preserve insertion order, and
    `tomllib.loads` inserts in the order tables appear in the source text),
    so an unrelated table's position in the re-emitted file matches where
    it already was. A table `_set_at` just created for a brand-new `table`
    argument is inserted (dict `setdefault`) after every table that was
    already present, so it is appended at the end -- matching the old
    text-appending behaviour for a table not yet in the file.
    """
    out: list[tuple[tuple[str, ...], dict]] = []
    for key, value in doc.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            if value and all(isinstance(v, str) for v in value.values()):
                out.append((path, value))
            else:
                out.extend(_leaf_tables(value, path))
    return out


def _emit_toml(doc: dict, header_comment: str) -> str:
    """`doc` (as merged by `_set_at`) re-emitted deterministically: the
    preserved leading comment block, then every leaf table in first-seen
    order. Replaces the old text-substitution path entirely -- merging
    happens on the PARSED structure, so a foreign table's entry sharing a
    key name with the target table is never touched, only the table
    actually named by `_set_at`'s dotted path is."""
    return header_comment + "\n".join(
        _emit_table(path, values) for path, values in _leaf_tables(doc)
    )


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
    deeply nested the table itself is. Merges on the PARSED document (not
    text) and re-emits deterministically -- `_emit_toml` -- rather than a
    text-level substitution: a text-level replace matched the first
    `key = ...` line ANYWHERE in the file, so a foreign table declaring the
    same key name earlier than the target table got silently rewritten
    while the intended entry stayed stale (found in review, reproduced
    through the real CLI). Every leaf table keeps its first-seen position
    (`_leaf_tables`); a leading `#` comment block is preserved verbatim
    (`_leading_comment_block`) since it is the one thing `tomllib.loads`
    cannot hand back -- an inline comment elsewhere in the file is not.
    """
    import tomllib

    doc = tomllib.loads(existing_text) if existing_text.strip() else {}
    current = _table_at(doc, table)

    if not force:
        _refuse_conflicts(table, current, entries)

    to_add = {k: v for k, v in entries.items() if current.get(k) != v}
    if not to_add:
        return _ensure_trailing_newline(existing_text)

    _set_at(doc, table, to_add)
    return _emit_toml(doc, _leading_comment_block(existing_text))


def toml_tables(fresh_content: str) -> list[tuple[str, dict[str, str]]]:
    """`plan()`'s fresh-file TOML content (already-valid TOML, one or more
    `[table]\\nkey = "value"` blocks) parsed back into `(table, entries)`
    pairs `merge_toml_table` can fold into whatever already exists on disk.

    `plan()` builds these strings directly rather than through
    `merge_toml_table` (there is nothing to merge into yet -- the manifest
    is what a fresh write would contain), so this is the one place that
    content is read back as data instead of re-derived, keeping `plan()`
    itself free of any notion of "what's already on disk". Public (not
    `_toml_tables`) because both `cli.py`'s `runs record` and `mcp/
    provenance.py`'s `runs.record` need it, and the mcp/ layer may not
    import a private name from a non-mcp module (see `tests/
    test_architecture.py::test_no_mcp_module_imports_a_private_domain_name`).
    """
    import tomllib

    doc = tomllib.loads(fresh_content)
    tables: list[tuple[str, dict[str, str]]] = []

    def _walk(prefix: str, node: dict) -> None:
        if all(isinstance(v, str) for v in node.values()) and node:
            tables.append((prefix, dict(node)))
            return
        for key, value in node.items():
            if isinstance(value, dict):
                _walk(f"{prefix}.{key}" if prefix else key, value)

    for top_key, top_value in doc.items():
        if isinstance(top_value, dict):
            _walk(top_key, top_value)
    return tables

"""The core run record, `attestation.run/1` (v1-DRAFT, not frozen).

`dumps`/`write` are the only sanctioned way to produce a record: they validate
with the same rules as reading, then serialize as RFC 8785 JCS, so the same
record always gives the same bytes. `loads`/`read` parse strictly, dispatch on
the record's `schema`, and raise a `RecordError` subclass naming the offending
path on anything non-conforming -- nothing is defaulted, coerced or dropped.

Every key is required; nullable keys must be spelled `null`. Unknown keys are
refused everywhere except inside `config` and `ext.<namespace>`, which are
carried verbatim. Curves never live here: `history` is a reference to the
sidecar `attestation/history.jsonl` (format and reader: `history.py`), bound by
sha256, and reading a record never opens it.
Spec: docs/superpowers/specs/2026-11-19-core-run-record-design.md.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from attestation_schema.canonical import MAX_SAFE_INTEGER, canonicalize

KIND = "attestation.run"
VERSION = 1
SCHEMA = f"{KIND}/{VERSION}"
# "draft" until the freeze recorded in the spec; then "frozen". A draft
# record may stop reading if the draft changes.
SCHEMA_STATUS = "draft"

STATUSES = ("complete", "failed", "cancelled")
DIRECTIONS = ("min", "max")
PROVENANCES = ("verified", "inferred", "unknown")

_SHA = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_SHA256 = re.compile(r"[0-9a-f]{64}")
# One instant, one spelling, and text order = time order: UTC `Z` and always
# exactly six fraction digits (`:37.500000Z`, `:37.000000Z`). A variable-width
# fraction sorted `:37.5Z` before `:37Z` (spec, Producer findings 2 and 4).
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z")
# `dependencies[].source`: where a dependency was installed from, in a closed,
# prefixed vocabulary so a reader can filter on it (spec, Producer findings 2).
SOURCE_KINDS = ("index", "git+", "file", "conda", "system")
_SOURCE = re.compile(r"index(?::.+)?|git\+.+|file|conda(?::.+)?|system", re.DOTALL)
# Non-finite metric values have no JSON number; these strings spell them.
NON_FINITE = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}
_NAMESPACE = re.compile(r"[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)*")
_SCHEMA_ID = re.compile(r"([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*)/([1-9][0-9]*)")
_RESERVED_NAMESPACE = "attestation"
# The history sidecar, relative to the run directory. The record names it (so
# a reader needs no layout knowledge) but may name nothing else: a reference
# that could point anywhere could point out of the run directory.
HISTORY_PATH = "attestation/history.jsonl"


class RecordError(ValueError):
    """A record that does not conform. `path` locates it (`metrics[2].direction`)."""

    def __init__(self, path: str, message: str):
        self.path = path
        super().__init__(f"{path}: {message}" if path else message)


class MissingFieldError(RecordError):
    """A required key is absent (a nullable key must still be present as null)."""


class UnknownFieldError(RecordError):
    """A key the schema does not define, outside `config` and `ext`."""


class InvalidFieldError(RecordError):
    """A key is present with a value the schema does not allow."""


class UnsupportedSchemaError(RecordError):
    """`schema` names a kind or version this reader does not know."""


class HistoryError(RecordError):
    """The history sidecar does not match the record that references it.
    `path` locates it (`attestation/history.jsonl:17` for a line)."""


class HistoryMissingError(HistoryError):
    """The record references a sidecar that is not there."""


class HistoryDigestError(HistoryError):
    """The sidecar's bytes are not the ones the record bound by sha256."""


class HistoryFormatError(HistoryError):
    """A sidecar line that does not conform (malformed, truncated, out of order,
    or counts that disagree with the reference)."""


@dataclass(frozen=True)
class Producer:
    """What wrote the record (not the code that ran -- that is `CodeIdentity`)."""

    name: str
    version: str | None


@dataclass(frozen=True)
class CodeIdentity:
    """The experiment code, and how sure that identity is -- the same ladder as
    a `Dependency`. `dirty=None` is unknown, never clean; `version` is the
    code's own version (a non-editable install may have nothing else)."""

    git_sha: str | None
    dirty: bool | None
    repo: str | None
    version: str | None
    provenance: str
    evidence: str | None


@dataclass(frozen=True)
class DataDigest:
    """Data the run read or produced. `sha256=None`: named, but its bytes
    were not digested (the producer knows the run read it, not what it held)."""

    name: str
    sha256: str | None
    role: str


@dataclass(frozen=True)
class Metric:
    """A final value. `direction=None` means the producer did not say.

    `value` may be NaN or +/-infinity (a diverged loss is a result, not a
    gap); on the wire those are the strings in `NON_FINITE`."""

    name: str
    value: int | float
    direction: str | None
    step: int | None
    split: str | None
    unit: str | None
    labels: dict[str, str]


@dataclass(frozen=True)
class HistoryRef:
    """Where the run's curves are and which bytes they are: the sidecar at
    `path` (always `HISTORY_PATH`), its sha256, and how many series and points
    it holds, so a reader can size it without opening it. The record holds
    this, never the points (`history.write_history` / `HistoryWriter` make it,
    `history.read_history` follows it)."""

    path: str
    sha256: str
    series: int
    points: int


@dataclass(frozen=True)
class Dependency:
    """Identity of something the code ran on, and how sure that identity is."""

    name: str
    version: str | None
    build_sha: str | None
    dirty: bool | None
    provenance: str
    evidence: str | None
    source: str | None


@dataclass(frozen=True)
class RunRecord:
    """One experiment run. Field meanings: see the spec's table."""

    run_id: str
    family: str | None
    arm: str | None
    status: str
    started_at: str | None
    finished_at: str | None
    producer: Producer
    code: CodeIdentity
    config: dict[str, Any]
    data: list[DataDigest]
    metrics: list[Metric]
    history: HistoryRef | None
    seeds: dict[str, int | list[int]] | None
    dependencies: list[Dependency] | None
    ext: dict[str, dict[str, Any]]
    schema: str = SCHEMA

    def to_dict(self) -> dict[str, Any]:
        """The record as plain JSON values, in their wire spelling (a NaN metric
        value is `"NaN"`), so `from_dict(r.to_dict()) == r`. Unvalidated;
        `dumps` and `from_dict` validate."""
        obj = asdict(self)
        _spell_values(obj)
        return obj


# ---- primitive checks -------------------------------------------------------


def _at(path: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    return f"{path}.{key}" if path else key


def _tname(value: object) -> str:
    return "null" if value is None else type(value).__name__


def _object(value: Any, path: str, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InvalidFieldError(path, f"expected an object, got {_tname(value)}")
    for key in keys:
        if key not in value:
            raise MissingFieldError(_at(path, key), "required field is missing")
    extra = sorted(str(k) for k in value if k not in keys)
    if extra:
        raise UnknownFieldError(_at(path, str(extra[0])), f"unknown field (allowed: {list(keys)})")
    return value


def _opt_string(value: Any, path: str, *, pattern: re.Pattern[str] | None = None) -> str | None:
    return None if value is None else _string(value, path, pattern=pattern, nullable=True)


def _string(
    value: Any, path: str, *, pattern: re.Pattern[str] | None = None, nullable: bool = False
) -> str:
    if not isinstance(value, str) or not value:
        want = "a non-empty string" + (" or null" if nullable else "")
        raise InvalidFieldError(path, f"expected {want}, got {_tname(value)}")
    _encodable(value, path)
    if pattern is not None and not pattern.fullmatch(value):
        raise InvalidFieldError(path, f"{value!r} does not match {pattern.pattern}")
    return value


def _choice(value: Any, path: str, choices: tuple[str, ...], *, nullable: bool = False) -> Any:
    if value is None and nullable:
        return None
    if value not in choices or not isinstance(value, str):
        raise InvalidFieldError(path, f"expected one of {list(choices)}, got {value!r}")
    return value


def _bool(value: Any, path: str) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise InvalidFieldError(path, f"expected true, false or null, got {_tname(value)}")
    return value


def _integer(value: Any, path: str, *, nullable: bool = False, minimum: int | None = None) -> Any:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidFieldError(path, f"expected an integer, got {_tname(value)}")
    if abs(value) > MAX_SAFE_INTEGER:
        raise InvalidFieldError(path, f"integer {value} is outside +/-2^53")
    if minimum is not None and value < minimum:
        raise InvalidFieldError(path, f"expected an integer >= {minimum}, got {value}")
    return value


def _metric_value(value: Any, path: str) -> int | float:
    """A metric value: a finite JSON number, or a `NON_FINITE` spelling."""
    if isinstance(value, str):
        if value not in NON_FINITE:
            raise InvalidFieldError(
                path, f"expected a number or one of {list(NON_FINITE)}, got {value!r}"
            )
        return NON_FINITE[value]
    return _number(value, path)


def _number(value: Any, path: str) -> int | float:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidFieldError(path, f"expected a finite number, got {value!r}")
        return value
    return _integer(value, path)


def _encodable(value: str, path: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InvalidFieldError(path, "string is not valid Unicode (lone surrogate)") from exc


def _json_value(value: Any, path: str) -> None:
    """Refuse anything canonicalization could not write, or would write lossily."""
    if value is None or isinstance(value, bool | str):
        if isinstance(value, str):
            _encodable(value, path)
    elif isinstance(value, int | float):
        _number(value, path)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _json_value(item, _at(path, i))
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise InvalidFieldError(path, f"object key {key!r} is not a string")
            _encodable(key, path)
            _json_value(item, _at(path, key))
    else:
        raise InvalidFieldError(path, f"{_tname(value)} is not a JSON value")


def _timestamp(value: Any, path: str) -> str | None:
    text = _opt_string(value, path, pattern=_TIMESTAMP)
    if text is not None:
        try:
            datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise InvalidFieldError(path, f"{text!r} is not a real instant") from exc
    return text


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise InvalidFieldError(path, f"expected an array, got {_tname(value)}")
    return value


def _unique(names: list[Any], path: str, what: str) -> None:
    seen: set[Any] = set()
    for i, name in enumerate(names):
        if name in seen:
            raise InvalidFieldError(_at(path, i), f"duplicate {what} {name!r}")
        seen.add(name)


# ---- v1 sections ------------------------------------------------------------

_TOP_V1 = (
    "schema",
    "run_id",
    "family",
    "arm",
    "status",
    "started_at",
    "finished_at",
    "producer",
    "code",
    "config",
    "data",
    "metrics",
    "history",
    "seeds",
    "dependencies",
    "ext",
)


def _producer(value: Any, path: str) -> Producer:
    d = _object(value, path, ("name", "version"))
    return Producer(
        name=_string(d["name"], _at(path, "name")),
        version=_opt_string(d["version"], _at(path, "version")),
    )


def _no_credentials(location: str, path: str) -> None:
    parts = urlsplit(location)
    web = parts.scheme.rsplit("+", 1)[-1] in ("http", "https")  # git+https too
    if parts.password is not None or (web and parts.username):
        raise InvalidFieldError(path, "URL carries credentials; strip them")


def _repo(value: Any, path: str) -> str | None:
    """A nullable repo location with no `user:password@` in it."""
    repo = _opt_string(value, path)
    if repo is not None:
        _no_credentials(repo, path)
    return repo


def _source(value: Any, path: str) -> str | None:
    """`index[:<name or url>]`, `git+<url>`, `file`, `conda[:<channel>]` or
    `system`; any URL in it free of credentials."""
    source = _opt_string(value, path)
    if source is None:
        return None
    if not _SOURCE.fullmatch(source):
        raise InvalidFieldError(
            path,
            f"{source!r} is not index[:<name>], git+<url>, file, conda[:<channel>] or system",
        )
    _no_credentials(
        source.partition(":")[2] if source[:6] in ("index:", "conda:") else source, path
    )
    return source


def _verified_needs(identity: str | None, provenance: str, path: str, what: str) -> None:
    if provenance == "verified" and identity is None:
        raise InvalidFieldError(_at(path, "provenance"), f"'verified' needs {what} to verify")


def _unknown_claims_no_sha(sha: str | None, provenance: str, path: str, field: str) -> None:
    """A commit is an identity claim; `unknown` makes none. A sha held but not
    vouched for is `inferred`; several (a run mixing builds or commits) is
    `unknown` with `evidence` naming each (spec, Producer findings 2)."""
    if provenance == "unknown" and sha is not None:
        raise InvalidFieldError(
            _at(path, field), f"'unknown' claims no {field}; a sha you hold is 'inferred'"
        )


def _code(value: Any, path: str) -> CodeIdentity:
    keys = ("git_sha", "dirty", "repo", "version", "provenance", "evidence")
    d = _object(value, path, keys)
    code = CodeIdentity(
        git_sha=_opt_string(d["git_sha"], _at(path, "git_sha"), pattern=_SHA),
        dirty=_bool(d["dirty"], _at(path, "dirty")),
        repo=_repo(d["repo"], _at(path, "repo")),
        version=_opt_string(d["version"], _at(path, "version")),
        provenance=_choice(d["provenance"], _at(path, "provenance"), PROVENANCES),
        evidence=_opt_string(d["evidence"], _at(path, "evidence")),
    )
    _verified_needs(code.git_sha or code.version, code.provenance, path, "a git_sha or a version")
    _unknown_claims_no_sha(code.git_sha, code.provenance, path, "git_sha")
    return code


def _data(value: Any, path: str) -> list[DataDigest]:
    out = []
    for i, item in enumerate(_list(value, path)):
        p = _at(path, i)
        d = _object(item, p, ("name", "sha256", "role"))
        out.append(
            DataDigest(
                name=_string(d["name"], _at(p, "name")),
                sha256=_opt_string(d["sha256"], _at(p, "sha256"), pattern=_SHA256),
                role=_string(d["role"], _at(p, "role")),
            )
        )
    _unique([d.name for d in out], path, "data name")
    return out


def _labels(value: Any, path: str) -> dict[str, str]:
    """`{key: value}`, both non-empty strings: what a value is ABOUT (an
    analogue, a model), so a reader groups without parsing metric names."""
    if not isinstance(value, dict):
        raise InvalidFieldError(path, f"expected an object, got {_tname(value)}")
    for key, label in value.items():
        _string(key, path)
        _string(label, _at(path, key))
    return value


def _label_key(labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(labels.items()))


def _metric(value: Any, path: str) -> Metric:
    keys = ("name", "value", "direction", "step", "split", "unit", "labels")
    d = _object(value, path, keys)
    return Metric(
        name=_string(d["name"], _at(path, "name")),
        value=_metric_value(d["value"], _at(path, "value")),
        direction=_choice(d["direction"], _at(path, "direction"), DIRECTIONS, nullable=True),
        step=_integer(d["step"], _at(path, "step"), nullable=True, minimum=0),
        split=_opt_string(d["split"], _at(path, "split")),
        unit=_opt_string(d["unit"], _at(path, "unit")),
        labels=_labels(d["labels"], _at(path, "labels")),
    )


def _one_per_name(out: list[Metric], path: str, field: str) -> None:
    seen: dict[str, Any] = {}
    for i, m in enumerate(out):
        mine = getattr(m, field)
        if seen.setdefault(m.name, mine) != mine:
            raise InvalidFieldError(
                _at(_at(path, i), field),
                f"metric {m.name!r} is {seen[m.name]!r} elsewhere; one {field} per name",
            )


def _metrics(value: Any, path: str) -> list[Metric]:
    out = [_metric(item, _at(path, i)) for i, item in enumerate(_list(value, path))]
    _unique(
        [(m.name, m.step, m.split, _label_key(m.labels)) for m in out],
        path,
        "metric (name, step, split, labels)",
    )
    _one_per_name(out, path, "direction")
    _one_per_name(out, path, "unit")
    return out


def _history_ref(value: Any, path: str) -> HistoryRef | None:
    """`null` (no curve was recorded) or the sidecar reference. The points
    themselves are never in the record (spec, Producer findings 3)."""
    if value is None:
        return None
    d = _object(value, path, ("path", "sha256", "series", "points"))
    if d["path"] != HISTORY_PATH:
        raise InvalidFieldError(_at(path, "path"), f"expected {HISTORY_PATH!r}, got {d['path']!r}")
    ref = HistoryRef(
        path=HISTORY_PATH,
        sha256=_string(d["sha256"], _at(path, "sha256"), pattern=_SHA256),
        series=_integer(d["series"], _at(path, "series"), minimum=1),
        points=_integer(d["points"], _at(path, "points"), minimum=1),
    )
    if ref.points < ref.series:
        raise InvalidFieldError(_at(path, "points"), "every series has at least one point")
    return ref


def _seeds(value: Any, path: str) -> dict[str, int | list[int]] | None:
    """`{name: int | [int, ...]}`: a list is one stage run once per seed
    (docking's `seeds: [7]`), in the producer's order; never empty."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise InvalidFieldError(path, f"expected an object or null, got {_tname(value)}")
    for name, seed in value.items():
        _string(name, _at(path, str(name)))
        if isinstance(seed, list):
            if not seed:
                raise InvalidFieldError(_at(path, name), "a seed list has at least one seed")
            for i, one in enumerate(seed):
                _integer(one, _at(_at(path, name), i))
        else:
            _integer(seed, _at(path, name))
    return value


def _dependency(value: Any, path: str) -> Dependency:
    keys = ("name", "version", "build_sha", "dirty", "provenance", "evidence", "source")
    d = _object(value, path, keys)
    dep = Dependency(
        name=_string(d["name"], _at(path, "name")),
        version=_opt_string(d["version"], _at(path, "version")),
        build_sha=_opt_string(d["build_sha"], _at(path, "build_sha"), pattern=_SHA),
        dirty=_bool(d["dirty"], _at(path, "dirty")),
        provenance=_choice(d["provenance"], _at(path, "provenance"), PROVENANCES),
        evidence=_opt_string(d["evidence"], _at(path, "evidence")),
        source=_source(d["source"], _at(path, "source")),
    )
    _verified_needs(dep.version or dep.build_sha, dep.provenance, path, "a version or a build_sha")
    _unknown_claims_no_sha(dep.build_sha, dep.provenance, path, "build_sha")
    return dep


def _dependencies(value: Any, path: str) -> list[Dependency] | None:
    """`null` = the producer could not observe the environment; `[]` = it
    did, and the code ran on nothing worth naming (as `seeds` null vs {})."""
    if value is None:
        return None
    out = [_dependency(item, _at(path, i)) for i, item in enumerate(_list(value, path))]
    _unique([d.name for d in out], path, "dependency name")
    return out


def _ext(value: Any, path: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise InvalidFieldError(path, f"expected an object, got {_tname(value)}")
    for namespace, body in value.items():
        p = _at(path, str(namespace))
        if not isinstance(namespace, str) or not _NAMESPACE.fullmatch(namespace):
            raise InvalidFieldError(
                p, f"namespace {namespace!r} does not match {_NAMESPACE.pattern}"
            )
        if namespace.split(".")[0] == _RESERVED_NAMESPACE:
            raise InvalidFieldError(p, "namespace 'attestation' is reserved for the schema")
        if not isinstance(body, dict):
            raise InvalidFieldError(p, f"an extension must be an object, got {_tname(body)}")
        _json_value(body, p)
    return value


def _config(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InvalidFieldError(path, f"expected an object, got {_tname(value)}")
    _json_value(value, path)
    return value


def _read_v1(obj: dict[str, Any]) -> RunRecord:
    d = _object(obj, "", _TOP_V1)
    record = RunRecord(
        schema=SCHEMA,
        run_id=_string(d["run_id"], "run_id"),
        family=_opt_string(d["family"], "family"),
        arm=_opt_string(d["arm"], "arm"),
        status=_choice(d["status"], "status", STATUSES),
        started_at=_timestamp(d["started_at"], "started_at"),
        finished_at=_timestamp(d["finished_at"], "finished_at"),
        producer=_producer(d["producer"], "producer"),
        code=_code(d["code"], "code"),
        config=_config(d["config"], "config"),
        data=_data(d["data"], "data"),
        metrics=_metrics(d["metrics"], "metrics"),
        history=_history_ref(d["history"], "history"),
        seeds=_seeds(d["seeds"], "seeds"),
        dependencies=_dependencies(d["dependencies"], "dependencies"),
        ext=_ext(d["ext"], "ext"),
    )
    if record.started_at and record.finished_at:
        start = datetime.fromisoformat(record.started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(record.finished_at.replace("Z", "+00:00"))
        if end < start:
            raise InvalidFieldError("finished_at", "is earlier than started_at")
    return record


# One entry per version this reader understands. A future version adds its
# reader here, and older readers gain an upgrade step to the newest model.
READERS: dict[str, Callable[[dict[str, Any]], RunRecord]] = {SCHEMA: _read_v1}


# ---- public API -------------------------------------------------------------


def from_dict(obj: Any) -> RunRecord:
    """Validate parsed JSON and return the record, dispatching on `schema`."""
    if not isinstance(obj, dict):
        raise InvalidFieldError("", f"a record is a JSON object, got {_tname(obj)}")
    if "schema" not in obj:
        raise MissingFieldError("schema", "required field is missing")
    schema = obj["schema"]
    match = _SCHEMA_ID.fullmatch(schema) if isinstance(schema, str) else None
    if match is None:
        raise InvalidFieldError("schema", f"expected '<kind>/<version>', got {schema!r}")
    if match.group(1) != KIND:
        raise UnsupportedSchemaError("schema", f"{schema!r} is not a run record ({KIND}/N)")
    reader = READERS.get(schema)
    if reader is None:
        known = sorted(READERS)
        raise UnsupportedSchemaError(
            "schema", f"{schema!r} is not a version this reader knows {known}; upgrade"
        )
    return reader(obj)


def _duplicate_free(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise InvalidFieldError("", f"duplicate object key {key!r}")
        out[key] = value
    return out


def _no_constant(name: str) -> Any:
    raise InvalidFieldError("", f"{name} is not a JSON number")


def _strict_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise InvalidFieldError("", f"{text} overflows a double")
    return value


def _strict_int(text: str) -> int:
    value = int(text)
    if abs(value) > MAX_SAFE_INTEGER:
        raise InvalidFieldError("", f"integer {text} is outside +/-2^53")
    return value


def loads(data: bytes | str) -> RunRecord:
    """Parse and validate a record. Raises `RecordError` on anything non-conforming."""
    if isinstance(data, bytes):
        try:
            data = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidFieldError("", f"not UTF-8: {exc}") from exc
    try:
        obj = json.loads(
            data,
            object_pairs_hook=_duplicate_free,
            parse_constant=_no_constant,
            parse_float=_strict_float,
            parse_int=_strict_int,
        )
    except json.JSONDecodeError as exc:
        raise InvalidFieldError("", f"not JSON: {exc}") from exc
    return from_dict(obj)


def dumps(record: RunRecord) -> bytes:
    """Validate `record` and serialize it as RFC 8785 canonical JSON (no newline)."""
    if not isinstance(record, RunRecord):
        raise TypeError(f"dumps() takes a RunRecord, got {type(record).__name__}")
    obj = record.to_dict()
    from_dict(obj)  # the reader's rules: a record that would not read cannot be written
    return canonicalize(obj)


def _spell_values(obj: dict[str, Any]) -> None:
    """Rewrite non-finite metric values as their wire strings, wherever the
    shape is the expected one (anything else the reader refuses)."""
    for holder in obj["metrics"] if isinstance(obj["metrics"], list) else []:
        if isinstance(holder, dict) and "value" in holder:
            holder["value"] = spell_value(holder["value"])


def spell_value(value: Any) -> Any:
    """A metric or point value in its wire spelling: a non-finite float as its
    `NON_FINITE` string, anything else unchanged (the reader then judges it).
    For a producer building record dicts by hand; `to_dict`, `dumps` and the
    history writer already apply it. `from_dict` takes only the wire form, so
    that a dict it accepts is always JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def format_timestamp(when: datetime) -> str:
    """The one spelling of an instant the record accepts: UTC, `Z`, and always
    six fraction digits (`2026-11-19T10:00:37.500000Z`), so timestamps sort
    as text in time order.

    A naive datetime is refused rather than assumed UTC.
    """
    if when.tzinfo is None:
        raise ValueError("format_timestamp needs an aware datetime; a naive one has no instant")
    utc = when.astimezone(UTC)
    return f"{utc.year:04d}-{utc:%m-%dT%H:%M:%S}.{utc.microsecond:06d}Z"


def record_digest(record: RunRecord) -> str:
    """The record's identity: hex sha256 of its canonical bytes (= of its file)."""
    return hashlib.sha256(dumps(record)).hexdigest()


def record_path(run_dir: str | os.PathLike[str]) -> Path:
    """Where a run directory's core record lives."""
    return Path(run_dir) / "attestation" / "run.json"


def write(record: RunRecord, run_dir: str | os.PathLike[str], *, overwrite: bool = False) -> Path:
    """Write the record to `<run_dir>/attestation/run.json`, atomically.

    Refuses an existing record unless `overwrite` (FileExistsError). The bytes
    are `dumps(record)`; a reader racing the write sees no file or the whole one.
    A record whose `history` names a sidecar is written only if that sidecar is
    already there with those bytes (`HistoryMissingError`, `HistoryDigestError`):
    the history is finalized before the record, never after.
    """
    payload = dumps(record)
    if record.history is not None:
        check_history_digest(run_dir, record.history)
    target = record_path(run_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        if overwrite:
            os.replace(tmp, target)
        else:
            os.link(tmp, target)  # atomic no-clobber: fails if a record exists
    finally:
        tmp.unlink(missing_ok=True)
    return target


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_history_digest(run_dir: str | os.PathLike[str], ref: HistoryRef) -> Path:
    """The sidecar `ref` names exists under `run_dir` with `ref.sha256`; its path."""
    path = Path(run_dir) / ref.path
    if not path.is_file():
        raise HistoryMissingError(ref.path, "the record references a history file that is missing")
    actual = _sha256_file(path)
    if actual != ref.sha256:
        raise HistoryDigestError(
            ref.path, f"sha256 is {actual}, the record binds {ref.sha256}: the file changed"
        )
    return path


def read(path: str | os.PathLike[str]) -> RunRecord:
    """Read a record file, or a run directory's `attestation/run.json`.

    Never opens the history sidecar: `history.read_history` does that, on request.
    """
    p = Path(path)
    if p.is_dir():
        p = record_path(p)
    return loads(p.read_bytes())

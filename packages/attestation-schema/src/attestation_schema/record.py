"""The core run record, `attestation.run/1` (v1-DRAFT, not frozen).

`dumps`/`write` are the only sanctioned way to produce a record: they validate
with the same rules as reading, then serialize as RFC 8785 JCS, so the same
record always gives the same bytes. `loads`/`read` parse strictly, dispatch on
the record's `schema`, and raise a `RecordError` subclass naming the offending
path on anything non-conforming -- nothing is defaulted, coerced or dropped.

Every key is required; nullable keys must be spelled `null`. Unknown keys are
refused everywhere except inside `config` and `ext.<namespace>`, which are
carried verbatim. Spec: docs/superpowers/specs/2026-11-19-core-run-record-design.md.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
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
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z")
_NAMESPACE = re.compile(r"[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)*")
_SCHEMA_ID = re.compile(r"([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*)/([1-9][0-9]*)")
_RESERVED_NAMESPACE = "attestation"


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


@dataclass(frozen=True)
class Producer:
    """What wrote the record (not the code that ran -- that is `CodeIdentity`)."""

    name: str
    version: str | None


@dataclass(frozen=True)
class CodeIdentity:
    """The experiment code. `dirty=None` is unknown, never clean."""

    git_sha: str | None
    dirty: bool | None
    repo: str | None


@dataclass(frozen=True)
class DataDigest:
    """A file the run read or produced, by sha256."""

    name: str
    sha256: str
    role: str


@dataclass(frozen=True)
class Metric:
    """A final value. `direction=None` means the producer did not say."""

    name: str
    value: int | float
    direction: str | None
    step: int | None
    split: str | None


@dataclass(frozen=True)
class Dependency:
    """Identity of something the code ran on, and how sure that identity is."""

    name: str
    version: str | None
    build_sha: str | None
    dirty: bool | None
    provenance: str
    evidence: str | None


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
    seeds: dict[str, int] | None
    dependencies: list[Dependency]
    ext: dict[str, dict[str, Any]]
    schema: str = SCHEMA

    def to_dict(self) -> dict[str, Any]:
        """The record as plain JSON values (unvalidated; `dumps` validates)."""
        return asdict(self)


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


def _repo(value: Any, path: str) -> str | None:
    repo = _opt_string(value, path)
    if repo is not None:
        parts = urlsplit(repo)
        if parts.password is not None or (parts.scheme in ("http", "https") and parts.username):
            raise InvalidFieldError(path, "repository URL carries credentials; strip them")
    return repo


def _code(value: Any, path: str) -> CodeIdentity:
    d = _object(value, path, ("git_sha", "dirty", "repo"))
    return CodeIdentity(
        git_sha=_opt_string(d["git_sha"], _at(path, "git_sha"), pattern=_SHA),
        dirty=_bool(d["dirty"], _at(path, "dirty")),
        repo=_repo(d["repo"], _at(path, "repo")),
    )


def _data(value: Any, path: str) -> list[DataDigest]:
    out = []
    for i, item in enumerate(_list(value, path)):
        p = _at(path, i)
        d = _object(item, p, ("name", "sha256", "role"))
        out.append(
            DataDigest(
                name=_string(d["name"], _at(p, "name")),
                sha256=_string(d["sha256"], _at(p, "sha256"), pattern=_SHA256),
                role=_string(d["role"], _at(p, "role")),
            )
        )
    _unique([d.name for d in out], path, "data name")
    return out


def _metric(value: Any, path: str) -> Metric:
    d = _object(value, path, ("name", "value", "direction", "step", "split"))
    return Metric(
        name=_string(d["name"], _at(path, "name")),
        value=_number(d["value"], _at(path, "value")),
        direction=_choice(d["direction"], _at(path, "direction"), DIRECTIONS, nullable=True),
        step=_integer(d["step"], _at(path, "step"), nullable=True, minimum=0),
        split=_opt_string(d["split"], _at(path, "split")),
    )


def _metrics(value: Any, path: str) -> list[Metric]:
    out = [_metric(item, _at(path, i)) for i, item in enumerate(_list(value, path))]
    _unique([(m.name, m.step, m.split) for m in out], path, "metric (name, step, split)")
    direction: dict[str, str | None] = {}
    for i, m in enumerate(out):
        if direction.setdefault(m.name, m.direction) != m.direction:
            raise InvalidFieldError(
                _at(_at(path, i), "direction"),
                f"metric {m.name!r} is {direction[m.name]!r} elsewhere; one direction per name",
            )
    return out


def _seeds(value: Any, path: str) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise InvalidFieldError(path, f"expected an object or null, got {_tname(value)}")
    for name, seed in value.items():
        _string(name, _at(path, str(name)))
        _integer(seed, _at(path, name))
    return value


def _dependency(value: Any, path: str) -> Dependency:
    keys = ("name", "version", "build_sha", "dirty", "provenance", "evidence")
    d = _object(value, path, keys)
    dep = Dependency(
        name=_string(d["name"], _at(path, "name")),
        version=_opt_string(d["version"], _at(path, "version")),
        build_sha=_opt_string(d["build_sha"], _at(path, "build_sha"), pattern=_SHA),
        dirty=_bool(d["dirty"], _at(path, "dirty")),
        provenance=_choice(d["provenance"], _at(path, "provenance"), PROVENANCES),
        evidence=_opt_string(d["evidence"], _at(path, "evidence")),
    )
    if dep.provenance == "verified" and dep.version is None and dep.build_sha is None:
        raise InvalidFieldError(
            _at(path, "provenance"), "'verified' needs a version or a build_sha to verify"
        )
    return dep


def _dependencies(value: Any, path: str) -> list[Dependency]:
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
    """
    payload = dumps(record)
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


def read(path: str | os.PathLike[str]) -> RunRecord:
    """Read a record file, or a run directory's `attestation/run.json`."""
    p = Path(path)
    if p.is_dir():
        p = record_path(p)
    return loads(p.read_bytes())

"""The dataset record, `attestation.dataset/1` (DRAFT, not frozen).

A dataset directory holds the dataset's files and, under `attestation/`, the
frozen `attestation.digests/1` manifest of them and this record:

    <dataset_dir>/<files>                    tables, split artifacts, anything
    <dataset_dir>/attestation/digests.json   sha256 of every file outside attestation/
    <dataset_dir>/attestation/dataset.json   the record; written last

**Content digest** = sha256 of the bytes of `attestation/digests.json`: the
RFC 8785 form of `{"schema": "attestation.digests/1", "files": {path:
sha256}}`. It is a digest of real bytes (`sha256sum` checks it), so a run
names the dataset in its `data[]` with exactly run/1's meaning of `sha256`,
and the reader recomputes it from `files[]` alone (`content_digest`).

Conventions are run/1's: every key required, null spelled out, unknown keys
refused outside `ext`, RFC 8785 bytes, fixed-width timestamps, null = could
not say vs [] = said and none. Unlike run/1, a URL with credentials is refused
in ANY string of the record.

Draft (`DATASET_SCHEMA_STATUS`): the identifier stays `attestation.dataset/1`
until two producers have written it, and a draft change may stop older draft
records from reading. Then it freezes under run/1's compatibility policy.
Spec: docs/superpowers/specs/2026-11-19-dataset-records-design.md.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from attestation_schema import layout
from attestation_schema.canonical import _utf16_key, canonicalize
from attestation_schema.record import (
    _SCHEMA_ID,
    _SHA256,
    CodeIdentity,
    DataDigest,
    InvalidFieldError,
    MissingFieldError,
    Producer,
    RecordError,
    UnsupportedSchemaError,
    _at,
    _code,
    _ext,
    _integer,
    _list,
    _no_credentials,
    _object,
    _opt_string,
    _producer,
    _string,
    _timestamp,
    _tname,
    _unique,
    parse_strict,
)

DATASET_KIND = "attestation.dataset"
DATASET_VERSION = 1
DATASET_SCHEMA = f"{DATASET_KIND}/{DATASET_VERSION}"
# "draft" until two producers have written dataset records the reader accepts
# and one is consumed by a run the lineage resolver links end to end (spec,
# Decision 5). Then "frozen", with the schema file and a reference record's
# bytes pinned in tests/test_frozen.py.
DATASET_SCHEMA_STATUS = "draft"

DATASET_JSON = layout.DATASET_JSON

# RFC 6838 shape, lowercase, no parameters: `application/vnd.apache.parquet`.
_MEDIA_TYPE = re.compile(r"[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*")
_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>]+")
# SPDX license expression tokens (SPDX 2.3, Annex D): parentheses, operators,
# and ids (`MIT`, `GPL-2.0+`, `LicenseRef-x`, `DocumentRef-d:LicenseRef-x`).
_SPDX_TOKEN = re.compile(r"\s*(\(|\)|[A-Za-z0-9.+:-]+)")
_SPDX_ID = re.compile(
    r"(?:DocumentRef-[A-Za-z0-9.-]+:)?LicenseRef-[A-Za-z0-9.-]+|[A-Za-z0-9][A-Za-z0-9.-]*\+?"
)
_SPDX_OPERATORS = ("AND", "OR", "WITH")
_SPDX_REFUSED = ("NOASSERTION", "NONE")

# A fixed table, never the host's `mimetypes` (whose answers differ by
# machine, which would make a record's bytes host-dependent).
MEDIA_TYPES = {
    ".parquet": "application/vnd.apache.parquet",
    ".arrow": "application/vnd.apache.arrow.file",
    ".feather": "application/vnd.apache.arrow.file",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".json": "application/json",
    ".jsonl": "application/jsonl",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".sdf": "chemical/x-mdl-sdfile",
    ".npy": "application/x-npy",
}


class DatasetManifestError(RecordError):
    """`attestation/digests.json` is missing, or is not the bytes the record's
    `content_sha256` names."""


@dataclass(frozen=True)
class DatasetFile:
    """One file of the dataset: its path under the dataset directory, the
    sha256 and size of its bytes, and its media type (None = not stated)."""

    path: str
    sha256: str
    bytes: int
    media_type: str | None


@dataclass(frozen=True)
class Column:
    """One column of a tabular dataset. `type` is the producer's spelling
    (`float64`); `unit` UCUM suggested. Either None = not stated."""

    name: str
    type: str | None
    unit: str | None


@dataclass(frozen=True)
class Split:
    """A split artifact inside the dataset: `path` is one of the record's
    files and `sha256` is that file's (the reader makes them agree)."""

    name: str
    path: str
    sha256: str
    kind: str
    seed: int | list[int] | None


@dataclass(frozen=True)
class SourceRun:
    """A run whose outputs the dataset was made from. `record_sha256` is the
    sha256 of that run's `attestation/run.json` bytes (its `record_digest`);
    None = named, not digested."""

    run_id: str
    family: str | None
    record_sha256: str | None


@dataclass(frozen=True)
class DatasetRecord:
    """One dataset. Field meanings: the spec's table (Decision 4)."""

    dataset_id: str
    name: str | None
    version: str | None
    created_at: str | None
    producer: Producer
    code: CodeIdentity
    content_sha256: str
    files: list[DatasetFile]
    rows: int | None
    columns: list[Column] | None
    splits: list[Split]
    source_runs: list[SourceRun] | None
    license: str | None
    ext: dict[str, dict[str, Any]]
    schema: str = DATASET_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        """The record as plain JSON values. Unvalidated; `dumps_dataset` and
        `dataset_from_dict` validate."""
        return asdict(self)


# ---- the content digest ---------------------------------------------------------


def manifest_of(files: Iterable[DatasetFile] | Mapping[str, str]) -> dict[str, Any]:
    """The `attestation.digests/1` object for these files (path -> sha256)."""
    pairs = files.items() if isinstance(files, Mapping) else ((f.path, f.sha256) for f in files)
    return {"schema": layout.DIGESTS_SCHEMA, "files": dict(pairs)}


def content_digest(files: Iterable[DatasetFile] | Mapping[str, str]) -> str:
    """sha256 of the canonical manifest of `files`: the bytes
    `attestation/digests.json` holds, hence the dataset's content digest."""
    return hashlib.sha256(canonicalize(manifest_of(files))).hexdigest()


# ---- v1 sections ------------------------------------------------------------------

_TOP_V1 = (
    "schema",
    "dataset_id",
    "name",
    "version",
    "created_at",
    "producer",
    "code",
    "content_sha256",
    "files",
    "rows",
    "columns",
    "splits",
    "source_runs",
    "license",
    "ext",
)


def _path(value: Any, path: str) -> str:
    """A relative POSIX path under the dataset directory, outside attestation/."""
    text = _string(value, path)
    parts = text.split("/")
    if (
        text.startswith("/")
        or "\x00" in text
        or any(p in ("", ".", "..") for p in parts)
        or parts[0] == layout.ATTESTATION_DIR
    ):
        raise InvalidFieldError(
            path,
            f"{text!r} is not a relative POSIX path (no empty, '.' or '..' segment) outside"
            f" {layout.ATTESTATION_DIR}/",
        )
    return text


def _files(value: Any, path: str) -> list[DatasetFile]:
    items = _list(value, path)
    if not items:
        raise InvalidFieldError(path, "a dataset has at least one file")
    out = []
    for i, item in enumerate(items):
        p = _at(path, i)
        d = _object(item, p, ("path", "sha256", "bytes", "media_type"))
        out.append(
            DatasetFile(
                path=_path(d["path"], _at(p, "path")),
                sha256=_string(d["sha256"], _at(p, "sha256"), pattern=_SHA256),
                bytes=_integer(d["bytes"], _at(p, "bytes"), minimum=0),
                media_type=_opt_string(d["media_type"], _at(p, "media_type"), pattern=_MEDIA_TYPE),
            )
        )
    for i in range(1, len(out)):
        if _utf16_key(out[i - 1].path) >= _utf16_key(out[i].path):
            raise InvalidFieldError(
                _at(_at(path, i), "path"),
                f"{out[i].path!r} is out of order or repeated: files are sorted by path"
                " (UTF-16 code units, the JCS key order), each once",
            )
    return out


def _columns(value: Any, path: str) -> list[Column] | None:
    if value is None:
        return None
    out = []
    for i, item in enumerate(_list(value, path)):
        p = _at(path, i)
        d = _object(item, p, ("name", "type", "unit"))
        out.append(
            Column(
                name=_string(d["name"], _at(p, "name")),
                type=_opt_string(d["type"], _at(p, "type")),
                unit=_opt_string(d["unit"], _at(p, "unit")),
            )
        )
    _unique([c.name for c in out], path, "column name")
    return out


def _seed(value: Any, path: str) -> int | list[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            raise InvalidFieldError(path, "a seed list has at least one seed")
        return [_integer(one, _at(path, i)) for i, one in enumerate(value)]
    return _integer(value, path)


def _splits(value: Any, path: str, files: list[DatasetFile]) -> list[Split]:
    by_path = {f.path: f.sha256 for f in files}
    out = []
    for i, item in enumerate(_list(value, path)):
        p = _at(path, i)
        d = _object(item, p, ("name", "path", "sha256", "kind", "seed"))
        split = Split(
            name=_string(d["name"], _at(p, "name")),
            path=_path(d["path"], _at(p, "path")),
            sha256=_string(d["sha256"], _at(p, "sha256"), pattern=_SHA256),
            kind=_string(d["kind"], _at(p, "kind")),
            seed=_seed(d["seed"], _at(p, "seed")),
        )
        if split.path not in by_path:
            raise InvalidFieldError(_at(p, "path"), f"{split.path!r} is not one of the files")
        if by_path[split.path] != split.sha256:
            raise InvalidFieldError(
                _at(p, "sha256"), f"files[] gives {split.path!r} sha256 {by_path[split.path]}"
            )
        out.append(split)
    _unique([s.name for s in out], path, "split name")
    return out


def _source_runs(value: Any, path: str) -> list[SourceRun] | None:
    """`null` = the producer could not say; `[]` = it can, and there are none."""
    if value is None:
        return None
    out = []
    for i, item in enumerate(_list(value, path)):
        p = _at(path, i)
        d = _object(item, p, ("run_id", "family", "record_sha256"))
        out.append(
            SourceRun(
                run_id=_string(d["run_id"], _at(p, "run_id")),
                family=_opt_string(d["family"], _at(p, "family")),
                record_sha256=_opt_string(
                    d["record_sha256"], _at(p, "record_sha256"), pattern=_SHA256
                ),
            )
        )
    _unique([(r.run_id, r.family, r.record_sha256) for r in out], path, "source run")
    digests = [r.record_sha256 for r in out]
    _unique([d if d is not None else ("null", i) for i, d in enumerate(digests)], path, "digest")
    return out


def check_spdx(expression: str) -> None:
    """Raise ValueError unless `expression` is SPDX license-expression syntax.

    Syntax only (ids are not checked against the SPDX list, which a stdlib
    package cannot ship current). `NOASSERTION`/`NONE` are refused: they are
    SPDX document values, and `null` already means "not stated"."""
    tokens: list[str] = []
    pos = 0
    while pos < len(expression):
        if expression[pos:].strip() == "":
            break
        m = _SPDX_TOKEN.match(expression, pos)
        if m is None:
            raise ValueError(f"unexpected {expression[pos:]!r}")
        tokens.append(m.group(1))
        pos = m.end()
    for tok in tokens:
        if tok in _SPDX_REFUSED:
            raise ValueError(f"{tok} is an SPDX document value, not a license; use null")
    rest = _spdx_or(tokens, 0)
    if rest != len(tokens):
        raise ValueError(f"unexpected {tokens[rest]!r}" if rest < len(tokens) else "incomplete")


def _spdx_or(tokens: list[str], i: int) -> int:
    i = _spdx_and(tokens, i)
    while i < len(tokens) and tokens[i] == "OR":
        i = _spdx_and(tokens, i + 1)
    return i


def _spdx_and(tokens: list[str], i: int) -> int:
    i = _spdx_simple(tokens, i)
    while i < len(tokens) and tokens[i] == "AND":
        i = _spdx_simple(tokens, i + 1)
    return i


def _spdx_simple(tokens: list[str], i: int) -> int:
    if i >= len(tokens):
        raise ValueError("incomplete: a license id or '(' is missing")
    if tokens[i] == "(":
        i = _spdx_or(tokens, i + 1)
        if i >= len(tokens) or tokens[i] != ")":
            raise ValueError("unbalanced parenthesis")
        return i + 1
    if tokens[i] in _SPDX_OPERATORS or not _SPDX_ID.fullmatch(tokens[i]):
        raise ValueError(f"{tokens[i]!r} is not a license id")
    i += 1
    if i < len(tokens) and tokens[i] == "WITH":
        if i + 1 >= len(tokens) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", tokens[i + 1]):
            raise ValueError("WITH needs an exception id")
        i += 2
    return i


def _license(value: Any, path: str) -> str | None:
    text = _opt_string(value, path)
    if text is not None:
        if text != text.strip():
            raise InvalidFieldError(path, f"{text!r} has surrounding whitespace")
        try:
            check_spdx(text)
        except ValueError as exc:
            raise InvalidFieldError(path, f"{text!r} is not an SPDX license expression: {exc}")
    return text


def _strings(value: Any, path: str) -> Iterable[tuple[str, str]]:
    """(path, text) for every string in a parsed JSON value, keys included."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _strings(item, _at(path, i))
    elif isinstance(value, dict):
        for key, item in value.items():
            yield _at(path, str(key)), str(key)
            yield from _strings(item, _at(path, str(key)))


def _no_credentials_anywhere(obj: dict[str, Any]) -> None:
    """A URL with credentials anywhere in the record is refused (run/1 checks
    two fields only; a new kind starts where v1 should have)."""
    for path, text in _strings(obj, ""):
        for match in _URL.finditer(text):
            try:
                _no_credentials(match.group(0), path)
            except ValueError as exc:
                if isinstance(exc, RecordError):
                    raise
                continue  # not parseable as a URL (urlsplit): not a credential either


def _read_v1(obj: dict[str, Any]) -> DatasetRecord:
    d = _object(obj, "", _TOP_V1)
    files = _files(d["files"], "files")
    record = DatasetRecord(
        schema=DATASET_SCHEMA,
        dataset_id=_string(d["dataset_id"], "dataset_id"),
        name=_opt_string(d["name"], "name"),
        version=_opt_string(d["version"], "version"),
        created_at=_timestamp(d["created_at"], "created_at"),
        producer=_producer(d["producer"], "producer"),
        code=_code(d["code"], "code"),
        content_sha256=_string(d["content_sha256"], "content_sha256", pattern=_SHA256),
        files=files,
        rows=_integer(d["rows"], "rows", nullable=True, minimum=0),
        columns=_columns(d["columns"], "columns"),
        splits=_splits(d["splits"], "splits", files),
        source_runs=_source_runs(d["source_runs"], "source_runs"),
        license=_license(d["license"], "license"),
        ext=_ext(d["ext"], "ext"),
    )
    expected = content_digest(files)
    if record.content_sha256 != expected:
        raise InvalidFieldError(
            "content_sha256",
            f"is {record.content_sha256}, but the manifest of files[] hashes to {expected}",
        )
    _no_credentials_anywhere(obj)
    return record


# One entry per version this reader understands (run/1's READERS pattern).
DATASET_READERS: dict[str, Callable[[dict[str, Any]], DatasetRecord]] = {DATASET_SCHEMA: _read_v1}


# ---- public API -------------------------------------------------------------------


def dataset_from_dict(obj: Any) -> DatasetRecord:
    """Validate parsed JSON and return the dataset record, dispatching on `schema`."""
    if not isinstance(obj, dict):
        raise InvalidFieldError("", f"a dataset record is a JSON object, got {_tname(obj)}")
    if "schema" not in obj:
        raise MissingFieldError("schema", "required field is missing")
    schema = obj["schema"]
    match = _SCHEMA_ID.fullmatch(schema) if isinstance(schema, str) else None
    if match is None:
        raise InvalidFieldError("schema", f"expected '<kind>/<version>', got {schema!r}")
    if match.group(1) != DATASET_KIND:
        raise UnsupportedSchemaError(
            "schema", f"{schema!r} is not a dataset record ({DATASET_KIND}/N)"
        )
    reader = DATASET_READERS.get(schema)
    if reader is None:
        known = sorted(DATASET_READERS)
        raise UnsupportedSchemaError(
            "schema", f"{schema!r} is not a version this reader knows {known}; upgrade"
        )
    return reader(obj)


def loads_dataset(data: bytes | str) -> DatasetRecord:
    """Parse strictly and validate a dataset record."""
    return dataset_from_dict(parse_strict(data))


def dumps_dataset(record: DatasetRecord) -> bytes:
    """Validate `record` and serialize it as RFC 8785 canonical JSON (no newline)."""
    if not isinstance(record, DatasetRecord):
        raise TypeError(f"dumps_dataset() takes a DatasetRecord, got {type(record).__name__}")
    obj = record.to_dict()
    dataset_from_dict(obj)
    return canonicalize(obj)


def dataset_digest(record: DatasetRecord) -> str:
    """The record's own identity: sha256 of its canonical bytes (= of dataset.json).
    Not the content digest: two records describing the same bytes differ here."""
    return hashlib.sha256(dumps_dataset(record)).hexdigest()


def dataset_path(dataset_dir: str | os.PathLike[str]) -> Path:
    """Where a dataset directory's record lives."""
    return Path(dataset_dir) / DATASET_JSON


def read_dataset(path: str | os.PathLike[str]) -> DatasetRecord:
    """Read a `dataset.json`, or a dataset directory's record."""
    p = Path(path)
    if p.is_dir():
        p = dataset_path(p)
    return loads_dataset(p.read_bytes())


def suggest_media_type(path: str) -> str | None:
    """A media type for `path` by extension, from the fixed `MEDIA_TYPES`
    table; None when the extension is not in it."""
    return MEDIA_TYPES.get(PurePosixPath(path).suffix.lower())


def dataset_files(
    dataset_dir: str | os.PathLike[str],
    media_types: Mapping[str, str | None] | None = None,
) -> list[DatasetFile]:
    """`files[]` for the directory as it is now: every file outside
    attestation/, sorted, with its sha256 and size. A path's media type is
    `media_types[path]` when given (None included), else `suggest_media_type`.
    Symlinks are refused (`layout.LayoutError`), as for a run's manifest."""
    root = Path(dataset_dir)
    digests = layout.build_manifest(root)["files"]
    given = media_types or {}
    return [
        DatasetFile(
            path=p,
            sha256=digests[p],
            bytes=(root / p).stat().st_size,
            media_type=given[p] if p in given else suggest_media_type(p),
        )
        for p in sorted(digests, key=_utf16_key)
    ]


def write_dataset_manifest(dataset_dir: str | os.PathLike[str]) -> str:
    """Write `attestation/digests.json` over the directory's files; return the
    content digest. Refused once `dataset.json` exists (the record binds it)."""
    root = Path(dataset_dir)
    if (root / DATASET_JSON).exists():
        raise layout.LayoutError(
            f"{root / DATASET_JSON} exists; the manifest is written before the record"
        )
    payload = canonicalize(layout.build_manifest(root))
    target = root / layout.DIGESTS_JSON
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, target)
    return hashlib.sha256(payload).hexdigest()


def write_dataset(
    record: DatasetRecord, dataset_dir: str | os.PathLike[str], *, overwrite: bool = False
) -> Path:
    """Write the record to `<dataset_dir>/attestation/dataset.json`, atomically.

    Refused (`DatasetManifestError`) unless `attestation/digests.json` is on
    disk with the bytes `content_sha256` names: files -> manifest -> record.
    Refuses an existing record unless `overwrite` (FileExistsError); a dataset
    directory is immutable, so a new version belongs in a new directory."""
    payload = dumps_dataset(record)
    root = Path(dataset_dir)
    manifest = root / layout.DIGESTS_JSON
    if not manifest.is_file():
        raise DatasetManifestError(
            layout.DIGESTS_JSON, "missing: write_dataset_manifest() before the record"
        )
    actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if actual != record.content_sha256:
        raise DatasetManifestError(
            layout.DIGESTS_JSON,
            f"sha256 is {actual}, the record's content_sha256 is {record.content_sha256}",
        )
    if (root / layout.RUN_JSON).exists() or (root / layout.STATE_JSON).exists():
        raise layout.LayoutError(f"{root} is a run directory; a dataset needs its own")
    target = root / DATASET_JSON
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


def dataset_lineage_entries(
    record: DatasetRecord, *, role: str = "train", split: str | None = None
) -> list[DataDigest]:
    """The `data[]` entries a consuming run lists: the dataset by its content
    digest (named by `dataset_id`, the name the lineage resolver diagnoses a
    version mismatch by), and, with `split`, that split's file by its digest
    (role `split`)."""
    out = [DataDigest(name=record.dataset_id, sha256=record.content_sha256, role=role)]
    if split is not None:
        found = [s for s in record.splits if s.name == split]
        if not found:
            raise KeyError(f"no split {split!r} in {record.dataset_id}")
        out.append(
            DataDigest(
                name=f"{record.dataset_id}/{found[0].path}", sha256=found[0].sha256, role="split"
            )
        )
    return out

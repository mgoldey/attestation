"""A W&B run's `.wandb` transaction log, decoded with the standard library.

An offline W&B run records its project, config, every `wandb.log` row, its
summary, its git commit and its exit code ONLY in this binary log (measured
2026-11-20, wandb 0.17.6: `files/` holds a metadata file and a summary of
`{"_wandb": {"runtime": 0}}`). The format is wandb's
`sdk/internal/datastore.py`: a LevelDB log -- a 7-byte file header, 32 KiB
blocks, each record `crc32 (LE) | length (LE u16) | type | data`, records
FULL or split FIRST/MIDDLE.../LAST -- whose payloads are protobuf `Record`
messages (`wandb_internal.proto`). This module reads both halves without
wandb or protobuf: the checksum is `zlib.crc32` seeded with the type byte's
CRC, as wandb computes it, and only varint and length-delimited fields are
decoded, by number (`FIELDS`). Every other field is skipped by wire type,
as protobuf itself skips an unknown field.

The numbers are bound by wandb's own compatibility promise -- `wandb sync`
uploads offline directories written by older clients -- and are pinned
against wandb's descriptors by a test that runs when wandb is importable.

Spec: docs/superpowers/specs/2026-11-20-wandb-hydra-adapters-design.md.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FILE_HEADER = struct.Struct("<4sHB")
IDENT, MAGIC, VERSION = b":W&B", 0xBEE1, 0
BLOCK = 32768
RECORD_HEADER = struct.Struct("<IHB")
FULL, FIRST, MIDDLE, LAST = 1, 2, 3, 4
_CRC_SEED = {kind: zlib.crc32(bytes([kind])) & 0xFFFFFFFF for kind in (FULL, FIRST, MIDDLE, LAST)}

# The protobuf field numbers read, per message (wandb_internal.proto).
FIELDS = {
    "Record": {"history": 2, "summary": 3, "config": 5, "run": 17, "exit": 18},
    "RunRecord": {
        "run_id": 1,
        "project": 3,
        "config": 4,
        "summary": 5,
        "display_name": 8,
        "start_time": 17,
        "git": 21,
    },
    "HistoryRecord": {"item": 1, "step": 2},
    "HistoryStep": {"num": 1},
    "Item": {"key": 1, "nested_key": 2, "value_json": 16},
    "UpdateRecord": {"update": 1, "remove": 2},
    "RunExitRecord": {"exit_code": 1},
    "GitRepoRecord": {"remote_url": 1, "commit": 2},
    "Timestamp": {"seconds": 1, "nanos": 2},
}
_R, _RUN, _HIST, _ITEM, _UPD = (
    FIELDS["Record"],
    FIELDS["RunRecord"],
    FIELDS["HistoryRecord"],
    FIELDS["Item"],
    FIELDS["UpdateRecord"],
)


class LogError(ValueError):
    """The log is damaged (bad header, checksum, framing or payload): the run
    is refused, never half-read. A log that merely ENDS early is not an
    error -- see `WandbLog.truncated`."""


@dataclass
class HistoryRow:
    """One `wandb.log` call: its items by key path, and the record's own step
    (`None` when unset; W&B normally carries the step as the `_step` item)."""

    items: dict[tuple[str, ...], Any]
    step: int | None = None


@dataclass
class WandbLog:
    """What a run's log says, in the order it said it. Config and summary are
    the state after every update and remove; `exit_code` is None when the log
    has no exit record (in flight, or the process died before finishing)."""

    run_id: str | None = None
    project: str | None = None
    display_name: str | None = None
    started: datetime | None = None
    git_commit: str | None = None
    git_remote: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    rows: list[HistoryRow] = field(default_factory=list)
    exit_code: int | None = None
    truncated: bool = False


# ---- LevelDB framing -----------------------------------------------------------


def _check_file_header(data: bytes) -> None:
    if len(data) < FILE_HEADER.size:
        raise LogError("not a W&B log: shorter than its file header")
    if FILE_HEADER.unpack_from(data) != (IDENT, MAGIC, VERSION):
        raise LogError("not a W&B log: bad file header (ident, magic or version)")


def _record_at(data: bytes, pos: int) -> tuple[int, bytes] | None:
    """`(type, data)` of the record at `pos`, or None if the file ends inside
    it. Raises on a checksum or type that is wrong."""
    if len(data) - pos < RECORD_HEADER.size:
        return None
    crc, length, kind = RECORD_HEADER.unpack_from(data, pos)
    start = pos + RECORD_HEADER.size
    chunk = data[start : start + length]
    if len(chunk) < length:
        return None
    if kind not in _CRC_SEED:
        raise LogError(f"record at byte {pos}: unknown record type {kind}")
    if zlib.crc32(chunk, _CRC_SEED[kind]) & 0xFFFFFFFF != crc:
        raise LogError(f"record at byte {pos}: checksum mismatch")
    return kind, chunk


def _assemble(kind: int, chunk: bytes, pending: bytes | None, pos: int):
    """(completed payload or None, still-pending bytes) for one fragment."""
    if kind in (FULL, FIRST) and pending is not None:
        raise LogError(f"record at byte {pos}: a new record before the last one ended")
    if kind in (MIDDLE, LAST) and pending is None:
        raise LogError(f"record at byte {pos}: a continuation with nothing to continue")
    if kind == FULL:
        return chunk, None
    if kind == FIRST:
        return None, chunk
    assert pending is not None
    return (pending + chunk, None) if kind == LAST else (None, pending + chunk)


def payloads(data: bytes) -> tuple[list[bytes], bool]:
    """Every complete payload in the log, and whether it ends mid-record."""
    _check_file_header(data)
    out: list[bytes] = []
    pending: bytes | None = None
    pos = FILE_HEADER.size
    while pos < len(data):
        space = BLOCK - pos % BLOCK
        if space < RECORD_HEADER.size:
            if data[pos : pos + space].strip(b"\0"):
                raise LogError(f"byte {pos}: block padding is not zero")
            pos += space
            continue
        record = _record_at(data, pos)
        if record is None:
            return out, True
        kind, chunk = record
        done, pending = _assemble(kind, chunk, pending, pos)
        if done is not None:
            out.append(done)
        pos += RECORD_HEADER.size + len(chunk)
    return out, pending is not None


# ---- protobuf wire format --------------------------------------------------------


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while pos < len(buf) and shift < 70:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
    raise LogError("protobuf payload: truncated or overlong varint")


_FIXED = {1: 8, 5: 4}


def fields(buf: bytes) -> dict[int, list[Any]]:
    """Every field of one message by number: ints for varints, bytes for
    length-delimited and fixed-width fields. Repeated fields keep order."""
    out: dict[int, list[Any]] = {}
    pos = 0
    while pos < len(buf):
        key, pos = _varint(buf, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 2 or wire in _FIXED:
            if wire == 2:
                size, pos = _varint(buf, pos)
            else:
                size = _FIXED[wire]
            value, pos = buf[pos : pos + size], pos + size
            if pos > len(buf):
                raise LogError("protobuf payload: a field runs past the message")
        else:
            raise LogError(f"protobuf payload: unsupported wire type {wire}")
        out.setdefault(number, []).append(value)
    return out


def _text(msg: dict[int, list[Any]], number: int) -> str | None:
    """The last value of a string field (protobuf: the last one wins)."""
    raw = msg.get(number)
    if not raw:
        return None
    try:
        return raw[-1].decode("utf-8") or None
    except (AttributeError, UnicodeDecodeError):
        raise LogError(f"protobuf payload: field {number} is not UTF-8 text") from None


def _int(msg: dict[int, list[Any]], number: int) -> int | None:
    raw = msg.get(number)
    if not raw or not isinstance(raw[-1], int):
        return None
    value = raw[-1]
    return value - (1 << 64) if value >= 1 << 63 else value  # int32/int64: two's complement


def _sub(msg: dict[int, list[Any]], number: int) -> list[dict[int, list[Any]]]:
    return [fields(b) for b in msg.get(number, []) if isinstance(b, bytes)]


# ---- the records read ------------------------------------------------------------


def _item(msg: dict[int, list[Any]]) -> tuple[tuple[str, ...], Any]:
    """A config/summary/history item: its key path and its decoded value.
    `nested_key`, when set, IS the path; else the single `key`."""
    nested = [b.decode("utf-8", "replace") for b in msg.get(_ITEM["nested_key"], [])]
    path = tuple(nested) if nested else (_text(msg, _ITEM["key"]) or "",)
    text = _text(msg, _ITEM["value_json"])
    try:
        value = json.loads(text) if text is not None else None
    except json.JSONDecodeError:
        raise LogError(f"item {'.'.join(path)}: value_json is not JSON") from None
    return path, value


def set_path(target: dict, path: tuple[str, ...], value: Any) -> None:
    """Set `value` at a key path in nested dicts, creating them as needed
    (how W&B nests a `nested_key` item)."""
    for key in path[:-1]:
        nxt = target.get(key)
        if not isinstance(nxt, dict):
            nxt = target[key] = {}
        target = nxt
    target[path[-1]] = value


def _remove(target: dict, path: tuple[str, ...]) -> None:
    node: Any = target
    for key in path[:-1]:
        node = node.get(key)
        if not isinstance(node, dict):
            return
    node.pop(path[-1], None)


def _apply(target: dict, msg: dict[int, list[Any]]) -> None:
    """A ConfigRecord or SummaryRecord: its updates, then its removes."""
    for item in _sub(msg, _UPD["update"]):
        set_path(target, *_item(item))
    for item in _sub(msg, _UPD["remove"]):
        _remove(target, _item(item)[0])


def _timestamp(msg: dict[int, list[Any]]) -> datetime | None:
    ts = FIELDS["Timestamp"]
    seconds = _int(msg, ts["seconds"])
    if seconds is None:
        return None
    nanos = _int(msg, ts["nanos"]) or 0
    return datetime.fromtimestamp(seconds, UTC).replace(microsecond=nanos // 1000)


def _read_git(log: WandbLog, git: dict[int, list[Any]]) -> None:
    names = FIELDS["GitRepoRecord"]
    log.git_commit = _text(git, names["commit"]) or log.git_commit
    log.git_remote = _text(git, names["remote_url"]) or log.git_remote


def _read_run(log: WandbLog, run: dict[int, list[Any]]) -> None:
    """The run record: identity, start, git, and the config/summary it
    carries (a resumed run's starting state). A later run record overrides
    only the fields it sets."""
    log.run_id = _text(run, _RUN["run_id"]) or log.run_id
    log.project = _text(run, _RUN["project"]) or log.project
    log.display_name = _text(run, _RUN["display_name"]) or log.display_name
    for start in _sub(run, _RUN["start_time"]):
        log.started = _timestamp(start) or log.started
    for git in _sub(run, _RUN["git"]):
        _read_git(log, git)
    for config in _sub(run, _RUN["config"]):
        _apply(log.config, config)
    for summary in _sub(run, _RUN["summary"]):
        _apply(log.summary, summary)


def _read_history(log: WandbLog, msg: dict[int, list[Any]]) -> None:
    items = dict(_item(i) for i in _sub(msg, _HIST["item"]))
    step = None
    for s in _sub(msg, _HIST["step"]):
        step = _int(s, FIELDS["HistoryStep"]["num"])
    log.rows.append(HistoryRow(items, step))


def _read_record(log: WandbLog, record: dict[int, list[Any]]) -> None:
    for run in _sub(record, _R["run"]):
        _read_run(log, run)
    for config in _sub(record, _R["config"]):
        _apply(log.config, config)
    for history in _sub(record, _R["history"]):
        _read_history(log, history)
    for summary in _sub(record, _R["summary"]):
        _apply(log.summary, summary)
    for exit_record in _sub(record, _R["exit"]):
        # proto3 omits a zero: an exit record with no exit_code field is 0.
        log.exit_code = _int(exit_record, FIELDS["RunExitRecord"]["exit_code"]) or 0


def read_log_bytes(data: bytes) -> WandbLog:
    """Decode one log's bytes. Raises `LogError` for a damaged log."""
    found, truncated = payloads(data)
    log = WandbLog(truncated=truncated)
    for payload in found:
        _read_record(log, fields(payload))
    return log


def read_log(path: Path) -> WandbLog:
    """Decode one `.wandb` log file. Raises `LogError` for a damaged one and
    `OSError` for one that cannot be read."""
    return read_log_bytes(Path(path).read_bytes())


__all__ = [
    "FIELDS",
    "HistoryRow",
    "LogError",
    "WandbLog",
    "fields",
    "payloads",
    "read_log",
    "read_log_bytes",
    "set_path",
]

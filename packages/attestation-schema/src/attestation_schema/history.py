"""The history sidecar: a run's logged curves, outside the core record.

    <run_dir>/attestation/history.jsonl

`run.json` never embeds a curve point. Its `history` key is `null` (no curve
was recorded) or a `HistoryRef` -- `{path, sha256, series, points}` -- that
binds this file by digest. Reading a record never opens this file;
`read_history` does, on request, and verifies the digest first.

The format (`attestation.history/1`) is JSON Lines, one object per line, each
line RFC 8785 canonical JSON followed by `\\n` (the last line too: a file
without a final newline was cut short):

    {"schema":"attestation.history/1"}                          line 1, always
    {"labels":{},"name":"loss","series":0,"split":"train"}      declares series 0
    {"at":"2026-11-19T10:00:01.500000Z","series":0,"step":0,"value":0.9}   a point
    ...

A series is declared once, immediately before its first point, with ids
0, 1, 2... in order of declaration, and is identified by `(name, split,
labels)`; every point names its series by id. Points of different series may
interleave -- a training loop logs `train/loss` and `eval/loss` alternately,
and requiring each series to be contiguous would make a long-running producer
buffer every curve until the end. One point per line, rather than a chunk of
points per line, because a chunk size is a free parameter: the same points
chunked twice would be two different files with two different digests. The
bytes are a pure function of the sequence of `HistoryWriter` calls.

`value` is spelled as in `metrics`: a finite JSON number, or `"NaN"`,
`"Infinity"`, `"-Infinity"`. Spec:
docs/superpowers/specs/2026-11-19-core-run-record-design.md.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from attestation_schema.canonical import canonicalize
from attestation_schema.files import _commit, _temporary
from attestation_schema.layout import RUN_JSON, LayoutError
from attestation_schema.record import (
    HISTORY_PATH,
    HistoryError,
    HistoryFormatError,
    HistoryRef,
    InvalidFieldError,
    RecordError,
    RunRecord,
    _at,
    _duplicate_free,
    _integer,
    _label_key,
    _labels,
    _metric_value,
    _no_constant,
    _object,
    _opt_string,
    _strict_float,
    _strict_int,
    _string,
    _timestamp,
    check_history_digest,
    read,
    spell_value,
)

HISTORY_SCHEMA = "attestation.history/1"
_HEADER = {"schema": HISTORY_SCHEMA}
_SERIES_KEYS = ("series", "name", "split", "labels")
_POINT_KEYS = ("series", "step", "value", "at")


@dataclass(frozen=True)
class Point:
    """One logged value of a series, in the order the producer logged it."""

    step: int
    value: int | float
    at: str | None


@dataclass(frozen=True)
class Series:
    """A metric's logged history (a curve). Steps may repeat and need not
    increase: it is a log, not a map. Final values live in `metrics`."""

    name: str
    split: str | None
    labels: dict[str, str]
    points: list[Point]


def _line(obj: dict[str, Any]) -> bytes:
    return canonicalize(obj) + b"\n"


class HistoryWriter:
    """Stream a run's curves into `attestation/history.jsonl`, then `close()`
    to get the `HistoryRef` the record carries.

        with HistoryWriter(run_dir) as hist:
            loss = hist.series("loss", split="train")
            for step in ...:
                hist.point(loss, step, value, at=format_timestamp(now))
        record = RunRecord(..., history=hist.ref, ...)

    Lines go to a temporary file in `attestation/` as they are logged (call
    `flush()` to push them to the OS); `close()` fsyncs it and moves it into
    place, so `history.jsonl` exists only once finalized, and a producer that
    dies mid-run leaves no sidecar a reader could mistake for a whole one.
    Write order: history (any time during the run) -> `digests.json` ->
    `run.json`; the writer refuses (`LayoutError`, as `write_manifest` does) to
    start or finish once `run.json` exists.

    Every value is validated as the reader will validate it, so a history the
    reader would refuse cannot be written. A series declared but never given
    a point writes nothing; a history with no points closes to `None`, which is
    what the record says when no curve was recorded.
    """

    def __init__(self, run_dir: str | os.PathLike[str], *, overwrite: bool = False):
        self.run_dir = Path(run_dir)
        self.target = self.run_dir / HISTORY_PATH
        self._refuse_after_record()
        if self.target.exists() and not overwrite:
            raise FileExistsError(f"{self.target} exists; pass overwrite=True to replace it")
        self._overwrite = overwrite
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = _temporary(self.target)
        self._fh: IO[bytes] | None = open(self._tmp, "xb")
        self._hash = hashlib.sha256()
        self._declared: list[tuple[str, str | None, dict[str, str]]] = []
        self._keys: set[tuple[Any, ...]] = set()
        self._ids: dict[int, int] = {}  # handle -> id in the file (first point order)
        self._points = 0
        self.ref: HistoryRef | None = None
        self._emit(_HEADER)

    def _refuse_after_record(self) -> None:
        record = self.run_dir / RUN_JSON
        if record.exists():
            raise LayoutError(f"{record} exists; the history is finalized before the record")

    def _emit(self, obj: dict[str, Any]) -> None:
        if self._fh is None:
            raise ValueError("this HistoryWriter is closed")
        line = _line(obj)
        self._fh.write(line)
        self._hash.update(line)

    def series(
        self, name: str, split: str | None = None, labels: dict[str, str] | None = None
    ) -> int:
        """Declare a series; returns the handle `point()` takes. One series
        per `(name, split, labels)`."""
        labels = dict(_labels({} if labels is None else labels, "labels"))
        _string(name, "name")
        _opt_string(split, "split")
        key = (name, split, _label_key(labels))
        if key in self._keys:
            raise _invalid("series", f"duplicate series {key!r}")
        self._keys.add(key)
        self._declared.append((name, split, labels))
        return len(self._declared) - 1

    def point(self, series: int, step: int, value: int | float, at: str | None = None) -> None:
        """Append one logged value to `series` (a handle from `series()`)."""
        if isinstance(series, bool) or not isinstance(series, int):
            raise _invalid("series", f"expected a handle from series(), got {series!r}")
        if not 0 <= series < len(self._declared):
            raise _invalid("series", f"no series was declared with handle {series}")
        _integer(step, "step", minimum=0)
        wire = spell_value(value)
        _metric_value(wire, "value")
        _timestamp(at, "at")
        if series not in self._ids:
            self._ids[series] = len(self._ids)
            name, split, labels = self._declared[series]
            sid = self._ids[series]
            self._emit({"series": sid, "name": name, "split": split, "labels": labels})
        self._emit({"series": self._ids[series], "step": step, "value": wire, "at": at})
        self._points += 1

    def flush(self) -> None:
        """Push buffered lines to the OS (the file is still the temporary one)."""
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> HistoryRef | None:
        """Finalize: the sidecar's `HistoryRef`, or `None` if nothing was logged."""
        if self._fh is None:
            return self.ref
        self._refuse_after_record()
        fh, self._fh = self._fh, None
        try:
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fh.close()
        try:
            if not self._points:
                return None
            # the shared commit step (`files`): replace, or an atomic no-clobber link
            _commit(self._tmp, self.target, exclusive=not self._overwrite)
            self.ref = HistoryRef(
                path=HISTORY_PATH,
                sha256=self._hash.hexdigest(),
                series=len(self._ids),
                points=self._points,
            )
            return self.ref
        finally:
            self._tmp.unlink(missing_ok=True)

    def abort(self) -> None:
        """Discard everything logged; no sidecar is left behind."""
        if self._fh is not None:
            fh, self._fh = self._fh, None
            fh.close()
        self._tmp.unlink(missing_ok=True)

    def __enter__(self) -> HistoryWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def _invalid(path: str, message: str) -> RecordError:
    return InvalidFieldError(path, message)


def write_history(
    run_dir: str | os.PathLike[str], series: Iterable[Series], *, overwrite: bool = False
) -> HistoryRef | None:
    """Write whole curves at once (each series in order, its points in order);
    return the `HistoryRef` for the record, or `None` if there are none."""
    with HistoryWriter(run_dir, overwrite=overwrite) as writer:
        for i, s in enumerate(series):
            if not s.points:
                raise _invalid(f"series[{i}].points", "a series has at least one point")
            handle = writer.series(s.name, s.split, s.labels)
            for p in s.points:
                writer.point(handle, p.step, p.value, p.at)
    return writer.ref


def read_history(
    run_dir: str | os.PathLike[str], ref: HistoryRef | RunRecord | None = None
) -> list[Series] | None:
    """The run's curves, or `None` when its record says no history was recorded.

    `ref` is the record (or its `history`); omitted, the run directory's
    `run.json` is read for it. The sidecar's sha256 is checked against the
    reference before a line is parsed. Raises `HistoryMissingError`,
    `HistoryDigestError`, or `HistoryFormatError` (a malformed or truncated
    line, or counts that disagree with the reference) -- all `HistoryError`,
    so all `RecordError`.
    """
    if ref is None:
        ref = read(run_dir)
    if isinstance(ref, RunRecord):
        ref = ref.history
    if ref is None:
        return None
    path = check_history_digest(run_dir, ref)
    with path.open("rb") as fh:
        series = _parse(fh)
    total = sum(len(s.points) for s in series)
    if (len(series), total) != (ref.series, ref.points):
        raise HistoryFormatError(
            HISTORY_PATH,
            f"holds {len(series)} series / {total} points; "
            f"the record says {ref.series} / {ref.points}",
        )
    return series


def _loads_line(raw: bytes, where: str) -> Any:
    if not raw.endswith(b"\n"):
        raise HistoryFormatError(where, "no final newline: the file was cut short")
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicate_free,
            parse_constant=_no_constant,
            parse_float=_strict_float,
            parse_int=_strict_int,
        )
    except UnicodeDecodeError as exc:
        raise HistoryFormatError(where, f"not UTF-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise HistoryFormatError(where, f"not JSON: {exc}") from exc


def _parse(lines: Iterable[bytes]) -> list[Series]:
    declared: list[Series] = []
    keys: set[tuple[Any, ...]] = set()
    lineno = 0
    for lineno, raw in enumerate(lines, start=1):
        where = f"{HISTORY_PATH}:{lineno}"
        try:
            obj = _loads_line(raw, where)
            if lineno == 1:
                if obj != _HEADER:
                    raise HistoryFormatError(where, f"first line must be {_HEADER}, got {obj!r}")
                continue
            if not isinstance(obj, dict):
                raise HistoryFormatError(where, f"expected an object, got {type(obj).__name__}")
            if "name" in obj:
                _series_line(obj, declared, keys)
            else:
                _point_line(obj, declared)
        except HistoryError:
            raise
        except RecordError as exc:
            raise HistoryFormatError(_at(where, exc.path) if exc.path else where, str(exc)) from exc
    if lineno == 0:
        raise HistoryFormatError(HISTORY_PATH, "empty file: no header line")
    if declared and not declared[-1].points:
        raise HistoryFormatError(HISTORY_PATH, f"series {len(declared) - 1} has no point")
    return declared


def _series_line(obj: dict[str, Any], declared: list[Series], keys: set[Any]) -> None:
    d = _object(obj, "", _SERIES_KEYS)
    sid = _integer(d["series"], "series", minimum=0)
    if sid != len(declared):
        raise _invalid("series", f"series ids are declared in order; expected {len(declared)}")
    if declared and not declared[-1].points:
        raise _invalid("series", f"series {sid - 1} was declared and given no point")
    s = Series(
        name=_string(d["name"], "name"),
        split=_opt_string(d["split"], "split"),
        labels=_labels(d["labels"], "labels"),
        points=[],
    )
    key = (s.name, s.split, _label_key(s.labels))
    if key in keys:
        raise _invalid("name", f"duplicate series {key!r}")
    keys.add(key)
    declared.append(s)


def _point_line(obj: dict[str, Any], declared: list[Series]) -> None:
    d = _object(obj, "", _POINT_KEYS)
    sid = _integer(d["series"], "series", minimum=0)
    if sid >= len(declared):
        raise _invalid("series", f"point for series {sid}, which is not declared yet")
    declared[sid].points.append(
        Point(
            step=_integer(d["step"], "step", minimum=0),
            value=_metric_value(d["value"], "value"),
            at=_timestamp(d["at"], "at"),
        )
    )

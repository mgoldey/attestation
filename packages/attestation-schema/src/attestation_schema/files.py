"""The package's one durable-write path, and its one file digest.

Every file this package writes -- `run.json`, `state.json`,
`history.jsonl`, both `digests.json` writers and `dataset.json` -- goes
through `_atomic_write` (or, for the streamed history sidecar, `_commit`):
the bytes go to a uniquely named temporary file beside the target, are
fsynced, and are moved into place with `os.replace` (overwrite) or
`os.link` (exclusive: an atomic no-clobber that fails if the target
exists); then the directory is fsynced so the rename itself survives a
crash. A reader racing a writer sees no file or the whole one.

The fsync matters for ORDER, not only for the one file: the layout's write
order is digests.json -> run.json, and a crash that kept a durable run.json
but lost a buffered digests.json would leave a record whose manifest is
missing. Before 0.2.0 the two manifest writers did not fsync while run.json
did (task-44 audit, items 3 and 6). Spec: docs/superpowers/specs/
2026-11-19-core-run-record-design.md, "Shared helpers (0.2.0)".
"""

from __future__ import annotations

import hashlib
import os
import secrets
from pathlib import Path

_CHUNK = 1 << 20


def sha256_file(path: str | os.PathLike[str]) -> str:
    """A file's hex sha256, read in 1 MiB chunks (never the whole file at once)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _temporary(target: Path) -> Path:
    """A fresh hidden name beside `target`: pid AND random, so two writers in
    one process (threads) never share a temporary file."""
    return target.with_name(f".{target.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")


def _fsync_dir(directory: Path) -> None:
    """Make a rename in `directory` durable. Best effort: a platform that
    cannot open a directory (Windows) has nothing to fsync."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _commit(tmp: Path, target: Path, *, exclusive: bool, fsync: bool = True) -> None:
    """Move a finished temporary file into place. `exclusive` refuses an
    existing target (FileExistsError) atomically; otherwise it is replaced.
    The caller unlinks `tmp` afterwards (after a link it is a second name)."""
    if exclusive:
        os.link(tmp, target)
    else:
        os.replace(tmp, target)
    if fsync:
        _fsync_dir(target.parent)


def _atomic_write(target: Path, data: bytes, *, exclusive: bool, fsync: bool = True) -> Path:
    """Write `data` to `target` atomically (see the module docstring); `target`."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temporary(target)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            if fsync:
                fh.flush()
                os.fsync(fh.fileno())
        _commit(tmp, target, exclusive=exclusive, fsync=fsync)
    finally:
        tmp.unlink(missing_ok=True)
    return target

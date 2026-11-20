"""Model registries read from disk: where a model reference's files are.

A registered model version is a NAME (`iris-clf` version 1, a Hub repo at a
commit) that some registry maps to a directory of files. This module answers
"which directory?" from the registry's own files, never from its server:

- `models:/<name>/<version>` and `models:/<name>@<alias>` -- MLflow's own
  URI syntax for a registered model version, resolved in an MLflow file
  store (`<root>/mlruns/models/<name>/version-<n>/meta.yaml`, the store the
  `mlflow` ledger adapter reads), whose `storage_location` names the files
  and whose `run_id` is MLflow's word for the run that made them;
- `hf:<repo>[@<revision>]` -- a Hugging Face Hub repo at a revision (a
  branch or tag under `refs/`, or a commit), resolved in a LOCAL hub cache
  (`models--<org>--<name>/snapshots/<commit>/`, the layout huggingface_hub
  downloads into). A cached LFS file is a link to `blobs/<sha256>`, so its
  bytes are checked against the name the cache gave them.

Nothing here opens a socket (the offline guarantee): a version that is not
on disk is "not found here", never fetched. The registry's run id is a NAME
and only diagnoses; the link to a run is made by digest in
`attestation.model_trace`. This module is the only place outside the
adapters' readers that knows these layouts, so it lives beside them
(tests/test_tracker_names.py).
Spec: docs/superpowers/specs/2026-11-20-model-registry-linkage-design.md.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlsplit

import attestation_schema as schema

from attestation.ledger_adapters.mlflow_store import STORE_DIR, _yaml

MLFLOW_SCHEME = "models:/"
HF_SCHEME = "hf:"
SCHEMES = (MLFLOW_SCHEME, HF_SCHEME)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{7,40}")
_NAME = re.compile(r"[^\s/@]+")
_REPO = re.compile(r"[^\s/@]+(?:/[^\s/@]+)?")


class RefError(ValueError):
    """A reference in a registry's scheme that does not parse."""


@dataclass(frozen=True)
class Located:
    """One registered version found on disk.

    `registry` names the layout (`mlflow` or `hf-cache`), `name`/`version`
    are its coordinates (an MLflow version number, a Hub commit), `location`
    is the directory holding the version's files, `source_run_id` is the run
    the registry says made it (None when it says nothing), `notes` say how
    the reference was resolved and `problems` what does not hold (a cached
    file whose bytes disagree with its blob's sha256)."""

    registry: str
    name: str
    version: str | None
    location: Path
    store: Path
    source_run_id: str | None = None
    notes: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()
    files: tuple[tuple[str, Path], ...] = field(default=(), repr=False)

    @property
    def label(self) -> str:
        """`mlflow iris-clf version 1`, `hf-cache org/name @ <commit>`."""
        if self.registry == "mlflow":
            return f"mlflow {self.name} version {self.version}"
        return f"{self.registry} {self.name} @ {self.version}"


def is_registry_ref(ref: str) -> bool:
    """Is `ref` in a registry scheme (rather than a directory path)?"""
    return ref.startswith(SCHEMES)


def _children(root: Path) -> list[Path]:
    try:
        return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    except OSError:
        return []


def _walk_files(folder: Path) -> tuple[tuple[str, Path], ...]:
    """Every file under `folder` (links followed), hidden names skipped:
    huggingface_hub's own `.cache/` beside a local download is not the model."""
    out = []
    for dirpath, dirnames, filenames in os.walk(folder, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if not name.startswith("."):
                path = Path(dirpath) / name
                out.append((path.relative_to(folder).as_posix(), path))
    return tuple(out)


# ---- MLflow's model registry ----------------------------------------------------


def parse_mlflow(ref: str) -> tuple[str, str | None, str | None]:
    """`models:/<name>/<version>` -> (name, version, None); `models:/<name>@<alias>`
    -> (name, None, alias). A logged model (`models:/m-<id>`) is not a
    registered version and is refused with that said."""
    body = ref[len(MLFLOW_SCHEME) :]
    if "@" in body:
        name, _, alias = body.partition("@")
        if _NAME.fullmatch(name) and _NAME.fullmatch(alias):
            return name, None, alias
    else:
        name, _, version = body.partition("/")
        if _NAME.fullmatch(name) and version.isdigit():
            return name, version, None
        if _NAME.fullmatch(body) and body.startswith("m-"):
            raise RefError(f"{ref}: a logged model id, not a registered version (name/version)")
    raise RefError(f"{ref}: expected models:/<name>/<version> or models:/<name>@<alias>")


def mlflow_stores(roots: list[Path]) -> list[Path]:
    """Every `mlruns/` the ledger's MLflow adapter would read under `roots`:
    `<root>/mlruns`, `<root>/<project>/mlruns`, or a root that is one."""
    out: list[Path] = []
    for root in roots:
        for candidate in (root, root / STORE_DIR, *(c / STORE_DIR for c in _children(root))):
            if candidate.name == STORE_DIR and candidate.is_dir() and candidate not in out:
                out.append(candidate)
    return out


def _in_store(uri: str, store: Path) -> Path | None:
    """The directory a `storage_location` names, read in THIS store when its
    tail is a path inside it (a moved or scrubbed store names another
    machine's prefix), else where it says if that is a local directory."""
    parts = urlsplit(uri)
    if parts.scheme not in ("", "file"):
        return None
    path = Path(unquote(parts.netloc + parts.path))
    pieces = path.parts
    for i, piece in enumerate(pieces):
        if piece in (".", "..") or Path(piece).is_absolute():
            continue
        if (store / piece).is_dir():
            candidate = store.joinpath(*pieces[i:])
            if candidate.is_dir():
                return candidate
    return path if path.is_absolute() and path.is_dir() else None


def _registered_dir(store: Path, name: str) -> Path | None:
    direct = store / "models" / name
    if (direct / "meta.yaml").is_file():
        return direct
    for d in _children(store / "models"):
        if str((_yaml(d / "meta.yaml") or {}).get("name")) == name:
            return d
    return None


def _alias(model_dir: Path, alias: str) -> str | None:
    """The version a registered model's alias points at, or None."""
    aliases = (_yaml(model_dir / "meta.yaml") or {}).get("aliases") or {}
    return str(aliases[alias]) if alias in aliases else None


def _locate_mlflow(ref: str, store: Path) -> tuple[Located | None, str]:
    name, version, alias = parse_mlflow(ref)
    model_dir = _registered_dir(store, name)
    if model_dir is None:
        return None, f"{store}: no registered model {name!r}"
    notes: list[str] = []
    if alias is not None:
        version = _alias(model_dir, alias)
        if version is None:
            return None, f"{store}: {name!r} has no alias {alias!r}"
        notes.append(f"alias {alias} -> version {version}")
    meta = _yaml(model_dir / f"version-{version}" / "meta.yaml")
    if meta is None:
        return None, f"{store}: {name!r} has no version {version}"
    status = meta.get("status")
    if status not in (None, "READY"):
        notes.append(f"version status {status}")
    uri = str(meta.get("storage_location") or meta.get("source") or "")
    location = _in_store(uri, store)
    if location is None:
        return (
            None,
            f"{store}: {name!r} version {version} is stored at {uri!r}, not a directory here",
        )
    run_id = meta.get("run_id")
    return (
        Located(
            registry="mlflow",
            name=name,
            version=str(version),
            location=location,
            store=store,
            source_run_id=str(run_id) if run_id not in (None, "") else None,
            notes=tuple(notes),
            files=_walk_files(location),
        ),
        "",
    )


# ---- a local Hugging Face hub cache ---------------------------------------------


def parse_hf(ref: str) -> tuple[str, str]:
    """`hf:<org>/<name>@<revision>` -> (repo, revision); no `@` means `main`."""
    body = ref[len(HF_SCHEME) :]
    repo, at, revision = body.partition("@")
    if not _REPO.fullmatch(repo) or (at and not _NAME.fullmatch(revision)):
        raise RefError(f"{ref}: expected hf:<org>/<name> or hf:<org>/<name>@<revision>")
    return repo, revision or "main"


def hf_caches(roots: list[Path], repo: str) -> list[Path]:
    """`models--<org>--<name>` directories for `repo` within two levels of a
    root (a root may be the hub cache itself, or hold `hub/`)."""
    folder = "models--" + repo.replace("/", "--")
    out: list[Path] = []
    for root in roots:
        for candidate in (root / folder, *(c / folder for c in _children(root))):
            for c in (candidate, candidate.parent / "hub" / folder):
                if (c / "snapshots").is_dir() and c not in out:
                    out.append(c)
    return out


def _blob_problems(files: tuple[tuple[str, Path], ...]) -> tuple[str, ...]:
    """A cached LFS file links to `blobs/<sha256 of its bytes>`: say so when
    the bytes disagree (a git blob's name is a sha1, not checked)."""
    out = []
    for rel, path in files:
        target = Path(os.readlink(path)).name if path.is_symlink() else ""
        if _SHA256.fullmatch(target) and schema.sha256_file(path) != target:
            out.append(f"{rel}: its bytes are not blob {target[:12]}... they link to")
    return tuple(out)


def _locate_hf(ref: str, cache: Path) -> tuple[Located | None, str]:
    repo, revision = parse_hf(ref)
    notes: list[str] = []
    commit = None
    ref_file = cache / "refs" / revision
    if ref_file.is_file():
        commit = ref_file.read_text().strip()
        notes.append(f"refs/{revision} -> {commit}")
    elif _COMMIT.fullmatch(revision):
        hits = [p.name for p in _children(cache / "snapshots") if p.name.startswith(revision)]
        if len(hits) > 1:
            return None, f"{cache}: revision {revision} matches {len(hits)} snapshots"
        commit = hits[0] if hits else None
    snapshot = cache / "snapshots" / commit if commit else None
    if snapshot is None or not snapshot.is_dir():
        return None, f"{cache}: no snapshot of {repo} at {revision}"
    files = _walk_files(snapshot)
    return (
        Located(
            registry="hf-cache",
            name=repo,
            version=commit,
            location=snapshot,
            store=cache,
            notes=tuple(notes),
            problems=_blob_problems(files),
            files=files,
        ),
        "",
    )


def locate(ref: str, roots: list[Path]) -> tuple[list[Located], list[str]]:
    """Every store under `roots` holding the version `ref` names, and why each
    store that does not hold it does not (the diagnosis of "not found").
    Raises RefError for a reference in a registry scheme that does not parse."""
    if ref.startswith(MLFLOW_SCHEME):
        parse_mlflow(ref)
        stores, find = mlflow_stores(roots), _locate_mlflow
        nowhere = "no MLflow store (mlruns/) under the search paths"
    elif ref.startswith(HF_SCHEME):
        repo, _ = parse_hf(ref)
        stores, find = hf_caches(roots, repo), _locate_hf
        nowhere = f"no local hub cache of {repo} (models--...) under the search paths"
    else:
        raise RefError(f"{ref}: not a registry reference")
    found: list[Located] = []
    why: list[str] = []
    for store in stores:
        hit, reason = find(ref, store)
        if hit is not None:
            found.append(hit)
        else:
            why.append(reason)
    return found, why or ([] if found else [nowhere])

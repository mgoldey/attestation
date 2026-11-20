"""One rule for "which build of this package ran": `package_identity`.

Before 0.2.0 there were two divergent copies of this rule -- smeltery's
`record.package_identity` (mapped to the core record by `core_record`) and this
package's private `integrations._core.dependency_identity` -- so the same
ferric install got a different `provenance` and `source` depending on which
producer wrote the record (task-44 audit, item 1). This module is the one
rule; both producers call it. Every branch below is a decision recorded in
docs/superpowers/specs/2026-11-19-dependency-identity-design.md, which says
why each divergence went the way it did.

The ladder, first match wins:

1. a verified build stamp: the module exposes `__build__ == {"git_sha": <40
   or 64 lowercase hex>, "dirty": <bool>}` (ferric's shape; any package may
   stamp itself) -> `verified`, `build_sha` and `dirty` from the stamp;
2. a VCS install whose PEP 610 `vcs_info` names a commit sha -> `verified`,
   `build_sha` = that commit, `dirty` = False (the installer built a fresh
   checkout of it);
3. an index install (no `direct_url.json`, `INSTALLER` pip or uv) whose
   imported module's `__version__` agrees with the installed metadata ->
   `verified` (an index release is immutable, and the agreement shows this
   process loaded that release);
4. anything else that names a version or a commit -> `inferred`;
5. nothing -> `unknown`.

`source` is read independently of the ladder, from `direct_url.json` and
`INSTALLER`, in the core record's closed vocabulary, credential-free.

Standard library only. `dependency_identity` is pure (every observation is an
argument); `package_identity` observes this process, and `git_state` runs git.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess  # nosec B404 -- git, with a fixed argv, no shell
import sys
from collections.abc import Mapping
from dataclasses import replace
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple
from urllib.parse import unquote, urlsplit, urlunsplit

from attestation_schema.record import _SHA, Dependency, _carries_credentials

BUILD_STAMP = "__build__"
# PEP 610 `vcs_info.vcs` values the core `source` vocabulary spells `<vcs>+<url>`.
VCS_KINDS = ("git", "hg", "svn", "bzr")
# `INSTALLER` values (PEP 376/627) that install from a package index when no
# `direct_url.json` says otherwise.
INDEX_INSTALLERS = frozenset({"pip", "uv"})
# A conda-family environment: the core `source` `conda` (the channel is not
# recorded). micromamba and mamba write their own names.
CONDA_INSTALLERS = frozenset({"conda", "mamba", "micromamba"})
# Distribution package managers that write an `INSTALLER` (Fedora's RPM
# macros write `rpm`, Debian's tooling `debian`): the core `source` `system`.
# Any OTHER installer is not evidence of where the bytes came from: source null.
SYSTEM_INSTALLERS = frozenset({"rpm", "debian"})
GIT_TIMEOUT_S = 10.0


# ---- URLs ----------------------------------------------------------------------


def strip_credentials(url: str) -> str:
    """`url` with the credentials the core record's reader refuses removed.

    Exactly the reader's rule (`record._carries_credentials`): the whole
    `user[:password]@` goes when it holds a password (any scheme:
    `ssh://u:p@`) or when the scheme is web (`https://token@`,
    `git+https://token@`); `ssh://git@host` and scp-style `git@host:path`
    carry no secret and are returned unchanged. A `dependencies[].source`
    value is accepted too: an `index:`, `conda:` or `url:` prefix is kept and
    the location after it stripped, and a `url:` source also loses its query
    string (a signed URL's credential lives there) while its fragment -- a
    `#sha256=` pin -- is kept. Whatever this returns, the reader accepts as a
    `repo` or (for a source of valid shape) a `source`.
    """
    for prefix in ("index:", "conda:", "url:"):
        if url.startswith(prefix):
            location = _strip_userinfo(url[len(prefix) :])
            if prefix == "url:" and _query(location):
                location = urlunsplit(urlsplit(location)._replace(query=""))
            return prefix + location
    return _strip_userinfo(url)


def _query(url: str) -> str:
    try:
        return urlsplit(url).query
    except ValueError:
        return ""


def _strip_userinfo(url: str) -> str:
    try:
        refused = _carries_credentials(url)
    except ValueError:  # urlsplit: not parseable as a URL (the reader refuses it anyway)
        return url
    if not refused:
        return url
    parts = urlsplit(url)
    return urlunsplit(parts._replace(netloc=parts.netloc.rpartition("@")[2]))


# ---- git -----------------------------------------------------------------------


class GitState(NamedTuple):
    """A checkout as git reports it now. `sha` None: not a checkout (or no
    git); `dirty` None: status could not be read; `remote` is origin's URL
    with credentials stripped, None when there is none."""

    sha: str | None
    dirty: bool | None
    remote: str | None


def _git(directory: Path, *args: str, timeout: float) -> str | None:
    try:
        done = subprocess.run(  # nosec B603 B607 -- fixed argv, no shell
            ["git", "-C", str(directory), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def git_state(directory: str | os.PathLike[str], *, timeout: float = GIT_TIMEOUT_S) -> GitState:
    """HEAD, whether tracked files differ from it (untracked files are not the
    code), and origin's credential-free URL, for the checkout at `directory`.
    Never raises: a missing git, a non-checkout or a timeout reads as None."""
    where = Path(directory)
    sha = _git(where, "rev-parse", "HEAD", timeout=timeout)
    if sha is None or not _SHA.fullmatch(sha):
        return GitState(None, None, None)
    status = _git(where, "status", "--porcelain", "--untracked-files=no", timeout=timeout)
    remote = _git(where, "config", "--get", "remote.origin.url", timeout=timeout)
    return GitState(
        sha=sha,
        dirty=None if status is None else bool(status),
        remote=strip_credentials(remote) if remote else None,
    )


# ---- the rule (pure) -------------------------------------------------------------


def build_stamp(module: object) -> tuple[str, bool] | None:
    """`(git_sha, dirty)` from `module.__build__` when it has the stamp shape
    `{"git_sha": <40/64 lowercase hex>, "dirty": <bool>}` (extra keys
    allowed); None otherwise -- absent, `None` (ferric's "no git metadata at
    build"), or malformed. A stamp is a claim the build made about itself, so
    a malformed one counts as none rather than being repaired."""
    stamp = getattr(module, BUILD_STAMP, None)
    if not isinstance(stamp, Mapping):
        return None
    sha, dirty = stamp.get("git_sha"), stamp.get("dirty")
    if isinstance(sha, str) and _SHA.fullmatch(sha) and isinstance(dirty, bool):
        return sha, dirty
    return None


def _same_release(module_version: str | None, dist_version: str | None) -> bool:
    """The imported module and the installed metadata name one release. A
    PEP 440 local label only the module reports agrees: PyPI's torch wheel
    says `2.15.0` in its metadata and `2.15.0+cu132` in `__version__`
    (measured, torch 2.15.0); the label is build detail the module adds."""
    if module_version is None or dist_version is None:
        return False
    if module_version == dist_version:
        return True
    public, _, local = module_version.partition("+")
    return bool(local) and "+" not in dist_version and public == dist_version


def _url_source(url: str) -> str | None:
    """`url:<url>` for a direct-URL archive (`file` for a local one); None for
    something that is not a `scheme://` URL (the vocabulary has no spelling)."""
    if url.startswith("file:"):
        return "file"
    source = strip_credentials(f"url:{url}")
    return source if re.fullmatch(r"url:[a-z][a-z0-9.-]*://.+", source, re.DOTALL) else None


def _install(
    direct_url: Mapping[str, Any] | None, installer: str | None, git: GitState | None
) -> tuple[str | None, bool, str | None, bool | None, list[str]]:
    """(source, may_verify, build_sha, dirty, evidence) for how it was installed.
    `may_verify`: whether a version agreement may make this `verified`."""
    url = direct_url.get("url") if direct_url else None
    url = url if isinstance(url, str) and url else None
    vcs = direct_url.get("vcs_info") if direct_url else None
    if isinstance(vcs, Mapping):
        kind, commit = vcs.get("vcs"), vcs.get("commit_id")
        if kind not in VCS_KINDS or url is None:
            return None, False, None, None, [f"direct_url.json vcs_info names vcs {kind!r}"]
        location = url if url.startswith(f"{kind}+") else f"{kind}+{url}"
        source = strip_credentials(location)
        if isinstance(commit, str) and _SHA.fullmatch(commit):
            return source, True, commit, False, [f"direct_url.json vcs_info ({kind}) commit_id"]
        return source, False, None, None, [f"direct_url.json vcs_info ({kind}) names no commit"]
    if direct_url and "archive_info" in direct_url:
        info = direct_url.get("archive_info")
        info = info if isinstance(info, Mapping) else {}
        hashes = info.get("hashes")
        digest = hashes.get("sha256") if isinstance(hashes, Mapping) else None
        digest = digest or info.get("hash")
        pinned = f" ({digest})" if digest else " (no archive hash recorded)"
        return _url_source(url or ""), False, None, None, ["direct_url.json archive_info" + pinned]
    if direct_url and "dir_info" in direct_url:
        info = direct_url.get("dir_info")
        editable = isinstance(info, Mapping) and bool(info.get("editable"))
        evidence = ["direct_url.json dir_info" + (" (editable)" if editable else "")]
        if git is not None and git.sha is not None:
            evidence.append("build_sha = the directory's git HEAD now, not at install")
            return "file", False, git.sha, git.dirty, evidence
        return "file", False, None, None, [*evidence, "the directory is not a git checkout"]
    if direct_url:
        return None, False, None, None, ["direct_url.json names no install kind"]
    tool = installer.strip().lower() if isinstance(installer, str) else ""
    if not tool:
        return None, False, None, None, ["no direct_url.json; no INSTALLER file"]
    evidence = [f"no direct_url.json; INSTALLER={tool}"]
    if tool in INDEX_INSTALLERS:
        return "index", True, None, None, evidence
    if tool in CONDA_INSTALLERS:
        return "conda", False, None, None, evidence
    if tool in SYSTEM_INSTALLERS:
        return "system", False, None, None, evidence
    return None, False, None, None, evidence


def dependency_identity(
    name: str,
    *,
    module_version: str | None,
    dist_version: str | None,
    direct_url: Mapping[str, Any] | None,
    installer: str | None,
    stamp: tuple[str, bool] | None = None,
    git: GitState | None = None,
    stamp_from: str = "the module",
) -> Dependency:
    """The `Dependency` for what was observed (pure): the imported module's
    `__version__`, the installed distribution's metadata version, its PEP 610
    `direct_url.json` (parsed), its `INSTALLER` text, the module's build stamp
    (`build_stamp`), and, for a directory install, that directory's
    `git_state`. `package_identity` gathers these for an installed package."""
    version = module_version or dist_version
    source, may_verify, build_sha, dirty, evidence = _install(direct_url, installer, git)
    if dist_version is None:
        evidence.append("no installed distribution metadata")
    agree = _same_release(module_version, dist_version)
    if module_version and dist_version and not agree:
        evidence.append(f"__version__ {module_version} != metadata {dist_version}")
    elif agree and module_version != dist_version:
        evidence.append(f"__version__ {module_version} is metadata {dist_version} + local label")
    elif agree:
        evidence.append(f"__version__ == metadata {dist_version}")
    elif dist_version and module_version is None:
        evidence.append("no module __version__ to check the metadata against")
    if stamp is not None:
        if build_sha is not None and build_sha != stamp[0]:
            evidence.append(f"stamp overrides {build_sha}")
        build_sha, dirty = stamp
        evidence.insert(0, f"{stamp_from}.{BUILD_STAMP} (stamped at build time)")
        provenance = "verified"
    elif build_sha is not None and may_verify:
        provenance = "verified"  # a VCS install's commit
    elif version is None and build_sha is None:
        provenance = "unknown"
    elif may_verify and agree:
        provenance = "verified"  # an index release the loaded module agrees with
    else:
        provenance = "inferred"
    return Dependency(
        name=name,
        version=version,
        build_sha=build_sha,
        dirty=dirty,
        provenance=provenance,
        evidence="; ".join(evidence),
        source=source,
    )


# ---- the rule, observed ----------------------------------------------------------


def _resolve_module(module: str | ModuleType | None) -> tuple[ModuleType | None, str | None]:
    """(module, why it is missing). A name already imported is not re-imported."""
    if module is None or isinstance(module, ModuleType):
        return module, None
    loaded = sys.modules.get(module)
    if loaded is not None:
        return loaded, None
    try:
        return importlib.import_module(module), None
    except ImportError as exc:
        return None, f"module {module} not importable ({exc.__class__.__name__})"


def _directory(url: str | None) -> Path | None:
    if not url or not url.startswith("file:"):
        return None
    return Path(unquote(urlsplit(url).path))


def package_identity(
    dist_name: str,
    module: str | ModuleType | None = None,
    *,
    name: str | None = None,
    probe_git: bool = True,
) -> Dependency:
    """Which build of the installed distribution `dist_name` this process runs,
    as a core-record `Dependency` named `name` (default `dist_name`).

    `module` is the import that distribution provides (a module object, or a
    name: taken from `sys.modules`, else IMPORTED -- pass a module object to
    avoid that). It supplies the build stamp and the `__version__` an index
    install is verified against; without it only the metadata is read, which
    can at best be `inferred`. `probe_git=False` skips the git probe of a
    directory install. Never raises for a missing distribution or module:
    what could not be observed is null, and `evidence` says why.
    """
    mod, missing = _resolve_module(module)
    raw_version = getattr(mod, "__version__", None) if mod is not None else None
    module_version = raw_version if isinstance(raw_version, str) and raw_version else None
    stamp = build_stamp(mod) if mod is not None else None
    try:
        dist = metadata.distribution(dist_name)
    except metadata.PackageNotFoundError:
        dist = None
    dist_version = dist.version if dist is not None else None
    direct_url: Mapping[str, Any] | None = None
    installer: str | None = None
    if dist is not None:
        raw = dist.read_text("direct_url.json")
        try:
            parsed = json.loads(raw) if raw else None
        except ValueError:
            parsed = None
        direct_url = parsed if isinstance(parsed, Mapping) else None
        installer = dist.read_text("INSTALLER")
    git = None
    if probe_git and direct_url is not None and "dir_info" in direct_url:
        where = _directory(direct_url.get("url"))
        git = git_state(where) if where is not None else None
    dep = dependency_identity(
        name or dist_name,
        module_version=module_version,
        dist_version=dist_version,
        direct_url=direct_url,
        installer=installer,
        stamp=stamp,
        git=git,
        stamp_from=mod.__name__ if mod is not None else "the module",
    )
    if missing is None and mod is None:
        missing = "no module given: metadata only"
    if missing:
        evidence = f"{dep.evidence}; {missing}" if dep.evidence else missing
        dep = replace(dep, evidence=evidence)
    return dep

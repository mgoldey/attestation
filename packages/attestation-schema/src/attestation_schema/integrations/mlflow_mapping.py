"""MLflow -> core record: the one mapping, and the provider's tags.

Standard library only: the exporter (`examples/mlflow-records`, through
MlflowClient), the ledger's file-store adapter (`attestation`, no mlflow
installed) and the run-context provider (`integrations/mlflow.py`) all call
this module, so no mlflow import may appear here.

Three parts:

- the pure mapping both readers share: status, git tag names, MLflow's
  timestamps, its latest-point key, the (metric, dataset) grouping;
- `observe_tags()`: what the run-context provider records at run start --
  the code's git state and the identity of every already-imported key
  package, through `identity.package_identity` (the one rule). Never raises;
- `code_from_tags` / `dependencies_from_tags`: those tags read back into a
  record, validated by the record reader's own field rules.

Spec: docs/superpowers/specs/2026-11-20-mlflow-plugin-design.md.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from importlib import metadata
from typing import Any, NamedTuple

from attestation_schema.canonical import canonicalize
from attestation_schema.identity import git_state, package_identity, strip_credentials
from attestation_schema.record import (
    CodeIdentity,
    Dependency,
    RecordError,
    _code,
    _dependencies,
    format_timestamp,
)

# ---- the shared mapping ---------------------------------------------------------

STATUS = {"FINISHED": "complete", "FAILED": "failed", "KILLED": "cancelled"}
GIT_COMMIT = "mlflow.source.git.commit"
GIT_REPO = "mlflow.source.git.repoURL"
RUN_NAME_TAG = "mlflow.runName"
CONTEXT_TAG = "mlflow.data.context"


def timestamp_ms(ms: int | None) -> str | None:
    """MLflow's epoch milliseconds as the record's one spelling of that instant."""
    if ms is None:
        return None
    whole, frac = divmod(int(ms), 1000)
    return format_timestamp(datetime.fromtimestamp(whole, UTC).replace(microsecond=frac * 1000))


def latest(points: Sequence[Any]) -> Any:
    """MLflow's own 'latest' point, by MLflow's own key: greatest step, then
    latest timestamp, then largest value (`FileStore._get_metric_from_file`)."""
    return max(points, key=lambda p: (p.step, p.timestamp, p.value))


def group_points(
    points: Iterable[Any], dataset: Callable[[Any], str | None]
) -> dict[tuple[str, str | None], list[Any]]:
    """Points per (metric, linked dataset) in logging order: sorted by
    timestamp, stably, so points logged in one millisecond keep the order
    the store returned them. `dataset` reads a point's linked dataset name
    (MlflowClient's `dataset_name`, the file store's fifth field)."""
    series: dict[tuple[str, str | None], list[Any]] = {}
    for p in sorted(points, key=lambda p: p.timestamp):
        series.setdefault((p.key, dataset(p) or None), []).append(p)
    return series


def series_order(keys: Iterable[tuple[str, str | None]]) -> list[tuple[str, str | None]]:
    """(metric, split) keys in the record's order: by name, then split."""
    return sorted(keys, key=lambda k: (k[0], k[1] or ""))


def _mlflow_code(tags: Mapping[str, str]) -> CodeIdentity:
    """MLflow's git tags: HEAD of the source's checkout at run start --
    inferred, since MLflow never records uncommitted changes."""
    sha = tags.get(GIT_COMMIT) or None
    repo = tags.get(GIT_REPO) or None
    return CodeIdentity(
        git_sha=sha,
        dirty=None,
        repo=strip_credentials(repo) if repo else None,
        version=None,
        provenance="inferred" if sha else "unknown",
        evidence=f"MLflow tag {GIT_COMMIT}: HEAD of the source's checkout at run start; "
        "uncommitted changes are not recorded"
        if sha
        else None,
    )


# ---- the provider's tags ----------------------------------------------------------

PREFIX = "attestation."
FORMAT_TAG = "attestation.format"
FORMAT = "attestation.mlflow-tags/1"
PRODUCER_TAG = "attestation.producer"
CODE_TAG = "attestation.code"
DEPENDENCIES_TAG = "attestation.dependencies"
ERROR_TAG = "attestation.error"
PRODUCER_NAME = "attestation-schema mlflow-run-context"
# `ATTESTATION_MLFLOW_TAGS=0` (or false/no/off) turns the provider off.
OPT_OUT_ENV = "ATTESTATION_MLFLOW_TAGS"
_OFF = frozenset({"0", "false", "no", "off"})
# `dist[:module],...`: packages to name beyond KEY_PACKAGES.
PACKAGES_ENV = "ATTESTATION_MLFLOW_PACKAGES"
# MLflow 3 caps a tag value at 8000 characters (`MAX_TAG_VAL_LENGTH`); older
# tracking servers at 5000. Every value is kept under the smaller.
MAX_TAG_VALUE = 5000
# A run start must not hang on a slow filesystem: identity's default is 10 s.
GIT_TIMEOUT_S = 2.0
# (record name, module, distributions to try): the packages an ML run's
# results depend on. Named only when the run's process already imported them.
KEY_PACKAGES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("mlflow", "mlflow", ("mlflow", "mlflow-skinny")),
    ("numpy", "numpy", ("numpy",)),
    ("pandas", "pandas", ("pandas",)),
    ("scikit-learn", "sklearn", ("scikit-learn",)),
    ("scipy", "scipy", ("scipy",)),
    ("torch", "torch", ("torch",)),
    ("tensorflow", "tensorflow", ("tensorflow", "tensorflow-cpu")),
    ("keras", "keras", ("keras",)),
    ("jax", "jax", ("jax",)),
    ("transformers", "transformers", ("transformers",)),
    ("lightning", "lightning", ("lightning",)),
    ("xgboost", "xgboost", ("xgboost",)),
    ("lightgbm", "lightgbm", ("lightgbm",)),
    ("catboost", "catboost", ("catboost",)),
)
TAG_EVIDENCE = (
    "observed at MLflow run start by attestation-schema's run-context provider"
    f" (tag {DEPENDENCIES_TAG})"
)


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    """False only when `ATTESTATION_MLFLOW_TAGS` says off. Reads one
    variable, nothing else: MLflow asks this for every run."""
    env = os.environ if environ is None else environ
    return env.get(OPT_OUT_ENV, "").strip().lower() not in _OFF


def _json(value: Any) -> str:
    return canonicalize(value).decode("utf-8")


def _script_directory() -> str:
    """The running script's directory -- the checkout MLflow's own git
    provider reads (`sys.argv[0]`) -- or the working directory."""
    script = sys.argv[0] if sys.argv and sys.argv[0] else ""
    if script and os.path.isfile(script):
        return os.path.dirname(os.path.abspath(script))
    return os.getcwd()


def observe_code(directory: str | None = None) -> CodeIdentity:
    """The script's checkout now (at run start): HEAD, dirty, origin."""
    git = git_state(directory or _script_directory(), timeout=GIT_TIMEOUT_S)
    if git.sha is None:
        return CodeIdentity(
            git_sha=None,
            dirty=None,
            repo=None,
            version=None,
            provenance="unknown",
            evidence="no git checkout found at the script at MLflow run start"
            " (or git is not installed)",
        )
    return CodeIdentity(
        git_sha=git.sha,
        dirty=git.dirty,
        repo=git.remote,
        version=None,
        provenance="inferred",
        evidence="git HEAD of the script's checkout at MLflow run start;"
        " dirty = tracked files differ from HEAD (git status)",
    )


def _installed(dists: Sequence[str]) -> str | None:
    for dist in dists:
        try:
            metadata.distribution(dist)
        except metadata.PackageNotFoundError:
            continue
        return dist
    return None


def declared_packages(environ: Mapping[str, str] | None = None) -> list[tuple[str, str]]:
    """`(distribution, module)` pairs from `ATTESTATION_MLFLOW_PACKAGES`
    (`dist[:module],...`; the module defaults to the name with - as _)."""
    env = os.environ if environ is None else environ
    out = []
    for item in env.get(PACKAGES_ENV, "").split(","):
        dist, _, module = item.strip().partition(":")
        if dist.strip():
            out.append((dist.strip(), module.strip() or dist.strip().replace("-", "_")))
    return out


def observe_dependencies(environ: Mapping[str, str] | None = None) -> list[Dependency]:
    """Python, then every key package this process has already imported,
    then the declared ones. The module object is passed, never a name, so
    nothing is imported here; a declared package not imported is identified
    from its metadata alone."""
    out = [
        Dependency(
            name="python",
            version=platform.python_version(),
            build_sha=None,
            dirty=None,
            provenance="verified",
            evidence="sys.version_info",
            source=None,
        )
    ]
    names = {"python"}
    for name, module, dists in KEY_PACKAGES:
        mod = sys.modules.get(module)
        dist = _installed(dists) if mod is not None else None
        if mod is not None and dist is not None:
            out.append(package_identity(dist, mod, name=name))
            names.add(name)
    for dist, module in declared_packages(environ):
        if dist not in names:
            out.append(package_identity(dist, sys.modules.get(module), name=dist))
            names.add(dist)
    return out


def _fit(deps: list[Dependency]) -> tuple[str, int]:
    """The dependencies tag value under `MAX_TAG_VALUE`, and how many fit
    (python first, so it always does)."""
    n = len(deps)
    while n > 1:
        value = _json([asdict(d) for d in deps[:n]])
        if len(value) <= MAX_TAG_VALUE:
            return value, n
        n -= 1
    return _json([asdict(d) for d in deps[:1]]), 1


def _error_line(what: str, exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    return f"{what}: {type(exc).__name__}: {text[0] if text else ''}"[:500]


def observe_tags(
    environ: Mapping[str, str] | None = None,
    *,
    code: Callable[[], CodeIdentity] | None = None,
    dependencies: Callable[[Mapping[str, str] | None], list[Dependency]] | None = None,
) -> dict[str, str]:
    """The provider's tags for the run starting now. Never raises: a probe
    that fails leaves its tag out and says why in `attestation.error`,
    and the other tags are still returned. `code`/`dependencies` default to
    `observe_code`/`observe_dependencies`, looked up at call time."""
    from attestation_schema import __version__

    code = code or observe_code
    dependencies = dependencies or observe_dependencies

    tags = {
        FORMAT_TAG: FORMAT,
        PRODUCER_TAG: _json({"name": PRODUCER_NAME, "version": __version__}),
    }
    errors: list[str] = []
    try:
        tags[CODE_TAG] = _json(asdict(code()))
    except Exception as exc:  # noqa: BLE001 -- never fail the user's run; reported on it
        errors.append(_error_line("code not observed", exc))
    try:
        deps = dependencies(environ)
        value, kept = _fit(deps)
        tags[DEPENDENCIES_TAG] = value
        if kept < len(deps):
            errors.append(
                f"{kept} of {len(deps)} dependencies fit MLflow's {MAX_TAG_VALUE}-character"
                f" tag limit; dropped: {', '.join(d.name for d in deps[kept:])}"
            )
    except Exception as exc:  # noqa: BLE001 -- never fail the user's run; reported on it
        errors.append(_error_line("dependencies not observed", exc))
    if errors:
        tags[ERROR_TAG] = "; ".join(errors)[:MAX_TAG_VALUE]
    return tags


# ---- the tags, read back ------------------------------------------------------------


class TagReading(NamedTuple):
    """What a run's tags say about its identity, and what could not be read."""

    code: CodeIdentity
    dependencies: list[Dependency] | None
    problems: list[str]


def _parsed(tags: Mapping[str, str], key: str, problems: list[str]) -> Any:
    raw = tags.get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        problems.append(f"tag {key} is not JSON; ignored")
        return None


def _format_ok(tags: Mapping[str, str], problems: list[str]) -> bool:
    if not any(k in tags for k in (CODE_TAG, DEPENDENCIES_TAG)):
        return False
    fmt = tags.get(FORMAT_TAG)
    if fmt != FORMAT:
        problems.append(f"tag {FORMAT_TAG} is {fmt!r}, not {FORMAT!r}; attestation tags ignored")
        return False
    return True


def code_from_tags(tags: Mapping[str, str], problems: list[str] | None = None) -> CodeIdentity:
    """The run's `code`, from the provider's tag and MLflow's git tags.

    Both name one sha: the provider's (it saw `dirty`). They disagree:
    `unknown`, both named (the record's rule for mixed commits). Only one:
    that one. Neither: `unknown`."""
    problems = [] if problems is None else problems
    theirs = _mlflow_code(tags)
    ours = None
    if _format_ok(tags, problems):
        raw = _parsed(tags, CODE_TAG, problems)
        if raw is not None:
            try:
                ours = _code(raw, CODE_TAG)
            except RecordError as exc:
                problems.append(f"tag {CODE_TAG} is not a core code object ({exc}); ignored")
    if ours is None or ours.git_sha is None:
        if theirs.git_sha is None and ours is not None:
            return replace(ours, repo=ours.repo or theirs.repo)
        return theirs
    if theirs.git_sha is not None and theirs.git_sha != ours.git_sha:
        return CodeIdentity(
            git_sha=None,
            dirty=None,
            repo=ours.repo or theirs.repo,
            version=None,
            provenance="unknown",
            evidence=f"two commits: {CODE_TAG} says {ours.git_sha},"
            f" {GIT_COMMIT} says {theirs.git_sha}",
        )
    evidence = ours.evidence or ""
    if theirs.git_sha is not None:
        evidence = f"{evidence}; {GIT_COMMIT} agrees" if evidence else f"{GIT_COMMIT} agrees"
    return replace(ours, repo=ours.repo or theirs.repo, evidence=evidence or None)


def dependencies_from_tags(
    tags: Mapping[str, str], problems: list[str] | None = None
) -> list[Dependency] | None:
    """The provider's dependency identities, provenance as the identity rule
    gave it at run start (a transcription neither raises nor lowers it), each
    `evidence` naming the tag. None (not observed) when there is no tag or it
    does not parse -- never `[]`."""
    problems = [] if problems is None else problems
    if not _format_ok(tags, problems):
        return None
    raw = _parsed(tags, DEPENDENCIES_TAG, problems)
    if raw is None:
        return None
    try:
        deps = _dependencies(raw, DEPENDENCIES_TAG)
    except RecordError as exc:
        problems.append(f"tag {DEPENDENCIES_TAG} is not a core dependency list ({exc}); ignored")
        return None
    if deps is None:
        return None
    return [
        replace(d, evidence=f"{d.evidence}; {TAG_EVIDENCE}" if d.evidence else TAG_EVIDENCE)
        for d in deps
    ]


def read_tags(tags: Mapping[str, str]) -> TagReading:
    """`code_from_tags` and `dependencies_from_tags` together, with every
    problem either found (each once), plus the provider's own
    `attestation.error`."""
    problems: list[str] = []
    code = code_from_tags(tags, problems)
    found: list[str] = []
    deps = dependencies_from_tags(tags, found)
    problems += [p for p in found if p not in problems]
    if tags.get(ERROR_TAG):
        problems.append(f"the run-context provider reported: {tags[ERROR_TAG]}")
    return TagReading(code, deps, problems)


__all__ = [
    "CODE_TAG",
    "CONTEXT_TAG",
    "DEPENDENCIES_TAG",
    "ERROR_TAG",
    "FORMAT",
    "FORMAT_TAG",
    "GIT_COMMIT",
    "GIT_REPO",
    "KEY_PACKAGES",
    "OPT_OUT_ENV",
    "PACKAGES_ENV",
    "PRODUCER_TAG",
    "RUN_NAME_TAG",
    "STATUS",
    "TagReading",
    "code_from_tags",
    "dependencies_from_tags",
    "enabled",
    "group_points",
    "latest",
    "observe_code",
    "observe_dependencies",
    "observe_tags",
    "read_tags",
    "series_order",
    "timestamp_ms",
]

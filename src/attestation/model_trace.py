"""Trace a model version to the run that trained it, and on to its training data.

The chain, each arrow by DIGEST:

    model version ─► its files (sha256) ─► the run whose core record lists
    those files as `output` data ─► that run's data[] ─► dataset records ─►
    their source runs ...

The first arrow is the registry's (`ledger_adapters.registries`, which
knows each registry's layout) or simply a directory of
saved model files. The second is new here: a run produced a version when its
record lists the version's files, by sha256, as outputs -- the files a
training callback records as checkpoints, or the logged-model files the
tracker adapters read from a run's declared outputs. Everything upstream of
that run is `attestation_schema.resolve_lineage`, unchanged.

Names only diagnose, as in lineage: the run id a registry names turns "no run
produced these files" into "the registry says run X, whose record lists none
of them", and a registry naming a run the digests do not agree with is
broken, never resolved by name. Reads files only, never a registry server,
never the ledger database. Pure over its inputs once the index is built.
Spec: docs/superpowers/specs/2026-11-20-model-registry-linkage-design.md.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import attestation_schema as schema
from attestation_schema import layout
from attestation_schema.lineage import BROKEN, UNVERIFIED, Lineage, LineageIssue

from attestation.ledger_adapters import import_project, registries

# kind -> severity, beside lineage.ISSUE_KINDS (whose issues a trace carries too).
MODEL_ISSUE_KINDS = {
    "unresolved-model": BROKEN,
    "empty-model": BROKEN,
    "model-file-mismatch": BROKEN,
    "no-producing-run": BROKEN,
    "registry-run-mismatch": BROKEN,
    # Files of the version no run output are bytes nothing traces: changed
    # weights, or a file added afterwards. Broken, not "mostly verified": a
    # shared config.json matching is no evidence about the weights beside it.
    "partial-output": BROKEN,
    "ambiguous-model": UNVERIFIED,
    "ambiguous-producer": UNVERIFIED,
    "untraced-training-data": UNVERIFIED,
}
MAX_NAMED = 3


@dataclass(frozen=True)
class ModelFile:
    """One file of a model version: path inside the version, sha256, size."""

    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class ModelVersion:
    """A model version found on disk. `registry` names the layout it was
    read from (`directory` for a plain one); `name`/`version` are its
    coordinates (None for a directory); `label` says them in one phrase."""

    ref: str
    registry: str
    name: str | None
    version: str | None
    location: str
    files: tuple[ModelFile, ...]
    source_run_id: str | None = None
    notes: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()
    label: str = "directory"


@dataclass(frozen=True)
class TracedRun:
    """A run record and where it was found (the ledger's `source_path`)."""

    record: schema.RunRecord = field(repr=False)
    source_path: str
    digest: str

    @property
    def label(self) -> str:
        """`family/run_id`, as lineage labels a run."""
        r = self.record
        return f"{r.family}/{r.run_id}" if r.family else r.run_id


@dataclass
class Index:
    """Every run and dataset record under the search paths."""

    runs: list[TracedRun] = field(default_factory=list)
    datasets: list[schema.DatasetRecord] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Producer:
    """A run whose outputs include files of the version: which, and the
    lineage upstream of it."""

    run: TracedRun
    version: ModelVersion
    matched: tuple[str, ...]
    lineage: Lineage = field(repr=False)

    @property
    def complete(self) -> bool:
        """It output every file of the version."""
        return len(self.matched) == len(self.version.files)


@dataclass(frozen=True)
class ModelTrace:
    """The chain from a model reference down to its training data."""

    ref: str
    versions: tuple[ModelVersion, ...]
    producers: tuple[Producer, ...]
    readers: tuple[TracedRun, ...]
    issues: tuple[LineageIssue, ...]

    @property
    def ok(self) -> bool:
        """No link is broken (unverified ones may remain)."""
        return not any(i.severity == BROKEN for i in self.issues)

    @property
    def verified(self) -> bool:
        """Every link holds by digest, model to its training data's source runs."""
        return not self.issues

    @property
    def verdict(self) -> str:
        """`verified`, `unverified links` or `BROKEN`."""
        return "BROKEN" if not self.ok else "unverified links" if self.issues else "verified"

    def connected(self, source_path: str) -> bool:
        """Did the run found at `source_path` produce or read this version?"""
        want = _norm(source_path)
        runs = [p.run for p in self.producers] + list(self.readers)
        return any(_norm(r.source_path) == want for r in runs)

    def chain(self) -> str:
        """One line: ref -> producing run(s) -> datasets reached [verdict]."""
        parts = [self.ref]
        if self.producers:
            parts.append(
                ", ".join(
                    f"run {p.run.label} ({len(p.matched)}/{len(p.version.files)} files)"
                    for p in self.producers
                )
            )
            data = []
            for p in self.producers:
                data += [n.label for n in p.lineage.nodes if n.kind == "dataset"]
            parts.append("dataset " + ", ".join(dict.fromkeys(data)) if data else "no dataset")
        return " -> ".join(parts) + f" [{self.verdict}]"

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON: versions with their files, producers with their lineage, issues."""
        return {
            "ref": self.ref,
            "ok": self.ok,
            "verified": self.verified,
            "versions": [asdict(v) for v in self.versions],
            "producers": [
                {
                    "run": p.run.label,
                    "source_path": p.run.source_path,
                    "record_sha256": p.run.digest,
                    "version": p.version.label,
                    "matched": list(p.matched),
                    "complete": p.complete,
                    "lineage": p.lineage.to_dict(),
                }
                for p in self.producers
            ],
            "readers": [{"run": r.label, "source_path": r.source_path} for r in self.readers],
            "issues": [asdict(i) for i in self.issues],
        }

    def format(self) -> str:
        """The version, its producing runs with their lineage trees, the issues."""
        lines = [f"model {self.ref}"]
        for v in self.versions:
            lines += _version_lines(v)
        for p in self.producers:
            lines.append(
                f"<- produced by run {p.run.label} (run:{p.run.digest[:12]})"
                f"  [output: {len(p.matched)}/{len(p.version.files)} files]"
            )
            for line in p.lineage.format().splitlines():
                if not line or line.startswith("lineage: "):
                    break
                lines.append("   " + line)
        for r in self.readers:
            lines.append(f"-> read by run {r.label} (run:{r.digest[:12]})")
        if self.issues:
            lines.append("")
            for i in self.issues:
                lines.append(f"{i.severity.upper()} [{i.kind}] {i.node} at {i.at}: {i.message}")
        lines.append(f"model trace: {self.verdict}")
        return "\n".join(lines)


def _version_lines(v: ModelVersion) -> list[str]:
    """A version's head lines: what and where, how it was resolved, whose run."""
    lines = [f"   {v.label} ({len(v.files)} file(s) at {v.location})"]
    lines += [f"   {note}" for note in v.notes]
    if v.source_run_id:
        lines.append(f"   the registry names run {v.source_run_id}")
    return lines


def _norm(path: str) -> str:
    p = Path(path)
    if p.name == "run.json" and p.parent.name == "attestation":
        p = p.parent.parent
    try:
        return str(p.resolve())
    except OSError:
        return str(p)


def _named(items: list[str]) -> str:
    shown = ", ".join(items[:MAX_NAMED])
    return shown + (f", ... ({len(items)} in all)" if len(items) > MAX_NAMED else "")


# ---- finding records -------------------------------------------------------------


def _projects(root: Path) -> list[Path]:
    """`root` and each project directory in it -- the ledger's view of a
    workspace (`ledger.scan` imports every subdirectory as a project)."""
    try:
        kids = [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    except OSError:
        return []
    return [root, *kids]


def build_index(roots: list[Path]) -> Index:
    """Every run (through the ledger's own adapters, so a tracker's store is
    read as the ledger reads it) and every dataset record under `roots`.
    A record that does not read is listed in `skipped`, never guessed."""
    out = Index()
    seen: set[tuple[str, str]] = set()
    for root in roots:
        for project in _projects(root):
            result = import_project(project)
            out.skipped += [r for r in result.refused if r not in out.skipped]
            for item in result.imported:
                digest = schema.record_digest(item.record)
                key = (_norm(item.source_path), digest)
                if key not in seen:
                    seen.add(key)
                    out.runs.append(TracedRun(item.record, item.source_path, digest))
        _add_datasets(out, root)
    return out


def _add_datasets(out: Index, root: Path) -> None:
    """Every dataset record under `root` not already in `out`, by record digest."""
    for d in layout.find_datasets(root):
        try:
            record = schema.read_dataset(d)
        except (schema.RecordError, OSError) as exc:
            out.skipped.append(f"{d}: {exc}")
            continue
        if all(schema.dataset_digest(record) != schema.dataset_digest(x) for x in out.datasets):
            out.datasets.append(record)


# ---- resolving the reference -----------------------------------------------------


def _files(entries: tuple[tuple[str, Path], ...]) -> tuple[ModelFile, ...]:
    return tuple(ModelFile(rel, schema.sha256_file(p), p.stat().st_size) for rel, p in entries)


def resolve(
    ref: str, roots: list[Path], base: Path | None = None
) -> tuple[list[ModelVersion], str]:
    """The versions `ref` names, found on disk, and (when none) why not.
    A registry reference is resolved by `ledger_adapters.registries`; anything
    else is a directory (or one file) of model files, relative to `base`.
    Raises `registries.RefError` for a registry reference that does not parse."""
    if registries.is_registry_ref(ref):
        found, why = registries.locate(ref, roots)
        versions = [
            ModelVersion(
                ref=ref,
                registry=f.registry,
                name=f.name,
                version=f.version,
                location=str(f.location),
                files=_files(f.files),
                source_run_id=f.source_run_id,
                notes=f.notes,
                problems=f.problems,
                label=f.label,
            )
            for f in found
        ]
        return versions, "; ".join(why)
    path = Path(ref).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    if path.is_file():
        entries: tuple[tuple[str, Path], ...] = ((path.name, path),)
    elif path.is_dir():
        entries = registries._walk_files(path)
    else:
        return [], f"{ref}: neither a registry reference (models:/..., hf:...) nor a path here"
    return [ModelVersion(ref, "directory", None, None, str(path), _files(entries))], ""


# ---- tracing ------------------------------------------------------------------------


def _digests(record: schema.RunRecord) -> tuple[set[str], set[str]]:
    """(sha256s the run output, sha256s it read) from its `data[]`."""
    outputs: set[str] = set()
    inputs: set[str] = set()
    for d in record.data:
        if d.sha256:
            (outputs if d.role == "output" else inputs).add(d.sha256)
    return outputs, inputs


class _Trace:
    def __init__(self, ref: str, index: Index):
        self.ref = ref
        self.node = f"model:{ref}"
        self.index = index
        self.issues: list[LineageIssue] = []
        self.producers: list[Producer] = []
        self.readers: list[TracedRun] = []

    def issue(self, kind: str, at: str, message: str, node: str | None = None) -> None:
        """Record one issue of `kind` (its severity from MODEL_ISSUE_KINDS), once."""
        severity = MODEL_ISSUE_KINDS[kind]
        item = LineageIssue(severity, kind, node or self.node, at, message)
        if item not in self.issues:
            self.issues.append(item)

    def version(self, v: ModelVersion) -> None:
        """Find the runs that output `v`'s files, their lineage, and every issue."""
        if not v.files:
            self.issue("empty-model", v.label, f"{v.location} holds no files")
            return
        for problem in v.problems:
            self.issue("model-file-mismatch", v.label, problem)
        candidates = self._candidates(v)
        if not candidates:
            self._no_producer(v)
            return
        chosen = self._choose(v, candidates)
        for run, matched in chosen:
            self._produced(v, run, matched)
        self._registry_agrees(v, [r for r, _ in chosen])

    def _candidates(self, v: ModelVersion) -> list[tuple[TracedRun, tuple[str, ...]]]:
        """(run, version files it output) for every run that output any; a run
        that only READ some (an evaluation) is kept in `readers`."""
        by_sha: dict[str, list[str]] = {}
        for f in v.files:
            by_sha.setdefault(f.sha256, []).append(f.path)
        out = []
        for run in self.index.runs:
            outputs, inputs = _digests(run.record)
            matched = tuple(sorted(p for sha in outputs & by_sha.keys() for p in by_sha[sha]))
            if matched:
                out.append((run, matched))
            elif inputs & by_sha.keys() and run not in self.readers:
                self.readers.append(run)
        return out

    def _choose(self, v: ModelVersion, candidates: list) -> list[tuple[TracedRun, tuple[str, ...]]]:
        """The runs covering the most files; partial or several is an issue."""
        best = max(len(m) for _, m in candidates)
        chosen = [(r, m) for r, m in candidates if len(m) == best]
        if best < len(v.files):
            missing = sorted({f.path for f in v.files} - set(chosen[0][1]))
            self.issue(
                "partial-output",
                v.label,
                f"the closest run(s) output {best} of {len(v.files)} file(s); not output: "
                + _named(missing),
            )
        if len(chosen) > 1:
            names = _named([r.label for r, _ in chosen])
            self.issue("ambiguous-producer", v.label, f"{len(chosen)} runs output these: {names}")
        return chosen

    def _produced(self, v: ModelVersion, run: TracedRun, matched: tuple[str, ...]) -> None:
        """One producer, the lineage upstream of it, and whether it reaches data."""
        lineage = schema.resolve_lineage(
            run.record, self.index.datasets, [r.record for r in self.index.runs]
        )
        self.producers.append(Producer(run, v, matched, lineage))
        self.issues += [i for i in lineage.issues if i not in self.issues]
        if any(n.kind == "dataset" for n in lineage.nodes):
            return
        inputs = [d.name for d in run.record.data if d.role not in ("output", "manifest")]
        why = (
            f"its inputs ({_named(inputs)}) match no dataset record by digest"
            if inputs
            else "its record lists no input data"
        )
        self.issue("untraced-training-data", "data", why, node=f"run:{run.digest}")

    def _registry_agrees(self, v: ModelVersion, runs: list[TracedRun]) -> None:
        if v.source_run_id and all(r.record.run_id != v.source_run_id for r in runs):
            self.issue(
                "registry-run-mismatch",
                v.label,
                f"the registry names run {v.source_run_id}, but the run(s) that output"
                f" these files are {_named([r.label for r in runs])}",
            )

    def _no_producer(self, v: ModelVersion) -> None:
        named = [
            r for r in self.index.runs if v.source_run_id and r.record.run_id == v.source_run_id
        ]
        if named:
            why = (
                f"the registry names run {v.source_run_id}; its record"
                f" ({named[0].label}) lists none of these files as output"
            )
        elif v.source_run_id:
            why = (
                f"the registry names run {v.source_run_id}; no record of it under the search paths"
            )
        else:
            why = "no run record under the search paths lists any of these files as output"
        self.issue("no-producing-run", v.label, why)


def trace(ref: str, index: Index, roots: list[Path], base: Path | None = None) -> ModelTrace:
    """The chain from `ref` down, resolved against `index` (`build_index(roots)`).
    Never raises for a broken link: the issues say what does not hold.
    Raises `registries.RefError` for a registry reference that does not parse."""
    versions, why = resolve(ref, roots, base)
    t = _Trace(ref, index)
    if not versions:
        t.issue("unresolved-model", "ref", why or "not found under the search paths")
    if len(versions) > 1:
        t.issue(
            "ambiguous-model",
            "ref",
            f"{len(versions)} stores hold it: {_named([v.location for v in versions])}; all kept",
        )
    for v in versions:
        t.version(v)
    return ModelTrace(ref, tuple(versions), tuple(t.producers), tuple(t.readers), tuple(t.issues))


class Tracer:
    """Lazy and cached, for the claim checker: the index is built on the first
    claim that names a model, so a draft with no `model=` reads nothing."""

    def __init__(self, roots: list[Path]):
        self.roots = [Path(r) for r in dict.fromkeys(os.fspath(r) for r in roots)]
        self._index: Index | None = None
        self._cache: dict[tuple[str, str], ModelTrace | str] = {}

    def __call__(self, ref: str, base: Path | None = None) -> ModelTrace | str:
        """The trace, or the complaint about a reference that does not parse."""
        key = (ref, str(base))
        if key not in self._cache:
            if self._index is None:
                self._index = build_index(self.roots)
            try:
                self._cache[key] = trace(ref, self._index, self.roots, base)
            except registries.RefError as exc:
                self._cache[key] = str(exc)
        return self._cache[key]


def tracer_for(target: Path) -> Tracer:
    """The claim checker's tracer for a draft at `target`: search the draft's
    own directory and the configured workspace (RESEARCH_ROOT), where the
    ledger found the runs the claims cite."""
    from attestation import ledger

    target = Path(target)
    roots = [target if target.is_dir() else target.parent]
    workspace = ledger.workspace_root()
    if workspace is not None:
        roots.append(Path(workspace))
    return Tracer(roots)

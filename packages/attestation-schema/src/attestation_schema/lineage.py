"""Resolving lineage: run -> datasets -> source runs -> their datasets ..., by digest.

`resolve_lineage(run, datasets, runs)` is pure: it takes records, reads no
file, and returns the graph upstream of `run` with every issue. A link is
only ever established by a digest -- a run's `data[].sha256` equal to a
dataset's content digest, one of its split files or one of its files, and a
dataset's `source_runs[].record_sha256` equal to a run's `record_digest`.
Names (`dataset_id`, `(family, run_id)`) only diagnose: they turn "this
matched nothing" into "you trained on another version" or "that run was
rewritten", and link a source run whose digest the producer did not record,
flagged `unverified`. Several matches are all kept, never one picked.

Standard library only. Spec:
docs/superpowers/specs/2026-11-19-dataset-records-design.md (Decision 6).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from attestation_schema.dataset import DatasetRecord, dataset_digest
from attestation_schema.record import Dependency, RunRecord, record_digest

BROKEN = "broken"
UNVERIFIED = "unverified"

# kind -> severity. The spec's table (Decision 6) is this dict.
ISSUE_KINDS = {
    "missing-source-run": BROKEN,
    "source-run-digest-mismatch": BROKEN,
    "source-run-identity-mismatch": BROKEN,
    "dataset-digest-mismatch": BROKEN,
    "undigested-source-run": UNVERIFIED,
    "undigested-data": UNVERIFIED,
    "unrecorded-source-runs": UNVERIFIED,
    "ambiguous": UNVERIFIED,
}


@dataclass(frozen=True)
class LineageNode:
    """A run (`run:<record digest>`) or a dataset record (`dataset:<its digest>`)."""

    key: str
    kind: str
    digest: str
    label: str
    record: RunRecord | DatasetRecord = field(repr=False, compare=False)


@dataclass(frozen=True)
class LineageEdge:
    """Data flowed from `source` to `target` (both node keys). `via` says how
    the link was established (`content`, `split`, `file`, `source_run`); `at`
    is the entry stating it, in the record of the downstream node."""

    source: str
    target: str
    via: str
    at: str
    detail: str


@dataclass(frozen=True)
class LineageIssue:
    """A link that does not hold (`broken`) or holds by name only / cannot be
    checked (`unverified`). `node` is the record carrying the entry `at`."""

    severity: str
    kind: str
    node: str
    at: str
    message: str


@dataclass(frozen=True)
class Lineage:
    """The graph upstream of `root`, every node reached, and every issue."""

    root: str
    nodes: tuple[LineageNode, ...]
    edges: tuple[LineageEdge, ...]
    issues: tuple[LineageIssue, ...]

    @property
    def ok(self) -> bool:
        """No link is broken (unverified links may remain)."""
        return not any(i.severity == BROKEN for i in self.issues)

    @property
    def verified(self) -> bool:
        """Every link holds by digest: no issue at all. The end-to-end claim."""
        return not self.issues

    def node(self, key: str) -> LineageNode:
        """The node with this key."""
        return next(n for n in self.nodes if n.key == key)

    def upstream(self, key: str) -> list[LineageEdge]:
        """The edges into `key`, in the order its record states them."""
        return [e for e in self.edges if e.target == key]

    def dependencies(self, name: str) -> list[tuple[str, Dependency]]:
        """(run node key, dependency) for every run reached that lists `name`
        -- how a build identity (ferric's `build_sha`) appears in the lineage."""
        out = []
        for n in self.nodes:
            if isinstance(n.record, RunRecord):
                out += [(n.key, d) for d in n.record.dependencies or [] if d.name == name]
        return out

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON: nodes without their records, edges, issues, ok, verified."""
        nodes = []
        for n in self.nodes:
            item: dict[str, Any] = {"key": n.key, "kind": n.kind, "digest": n.digest}
            item["label"] = n.label
            if isinstance(n.record, DatasetRecord):
                item["content_sha256"] = n.record.content_sha256
            nodes.append(item)
        return {
            "root": self.root,
            "ok": self.ok,
            "verified": self.verified,
            "nodes": nodes,
            "edges": [asdict(e) for e in self.edges],
            "issues": [asdict(i) for i in self.issues],
        }

    def format(self) -> str:
        """The graph as an indented tree upstream from the root, then the issues."""
        lines: list[str] = []
        self._tree(self.root, "", None, set(), lines)
        if self.issues:
            lines.append("")
            for i in self.issues:
                lines.append(f"{i.severity.upper()} [{i.kind}] {i.node} at {i.at}: {i.message}")
        verdict = "BROKEN" if not self.ok else "unverified links" if self.issues else "verified"
        lines.append(f"lineage: {verdict}")
        return "\n".join(lines)

    def _tree(
        self, key: str, indent: str, edge: LineageEdge | None, shown: set[str], lines: list[str]
    ) -> None:
        """One node, then (the first time it is shown) its builds and upstream:
        a node reached twice -- a dataset named by content and by split, a
        diamond, a cycle -- is expanded once."""
        n = self.node(key)
        how = f"  [{edge.via} via {edge.at}: {edge.detail}]" if edge else ""
        head = f"{indent}{'<- ' if edge else ''}{n.kind} {n.label} ({n.kind}:{n.digest[:12]}){how}"
        if key in shown:
            lines.append(head + "  (shown above)")
            return
        shown.add(key)
        lines.append(head)
        if isinstance(n.record, RunRecord):
            for d in n.record.dependencies or []:
                if d.build_sha is not None:
                    v = f" {d.version}" if d.version else ""
                    built = f"{d.name}{v} @ {d.build_sha} ({d.provenance})"
                    lines.append(f"{indent}   built on {built}")
        for e in self.upstream(key):
            self._tree(e.source, indent + "   ", e, shown, lines)


def _run_label(r: RunRecord) -> str:
    return f"{r.family}/{r.run_id}" if r.family else r.run_id


def _dataset_label(d: DatasetRecord) -> str:
    return f"{d.dataset_id}@{d.version}" if d.version else d.dataset_id


class _Resolver:
    def __init__(self, datasets: Iterable[DatasetRecord], runs: Iterable[RunRecord]):
        self.nodes: dict[str, LineageNode] = {}
        self.edges: list[LineageEdge] = []
        self.issues: list[LineageIssue] = []
        self.by_content: dict[str, list[str]] = {}
        self.by_split: dict[str, list[tuple[str, str]]] = {}
        self.by_file: dict[str, list[tuple[str, str]]] = {}
        self.versions: dict[str, list[str]] = {}  # dataset_id -> content digests
        self.runs_by_name: dict[tuple[str | None, str], list[str]] = {}
        for d in datasets:
            key = self._add_dataset(d)
            self.by_content.setdefault(d.content_sha256, []).append(key)
            for s in d.splits:
                self.by_split.setdefault(s.sha256, []).append((key, f"split {s.name} ({s.path})"))
            for f in d.files:
                self.by_file.setdefault(f.sha256, []).append((key, f"file {f.path}"))
            self.versions.setdefault(d.dataset_id, []).append(d.content_sha256)
        for r in runs:
            self.add_run(r)

    def _add_dataset(self, d: DatasetRecord) -> str:
        digest = dataset_digest(d)
        key = f"dataset:{digest}"
        if key not in self.nodes:
            self.nodes[key] = LineageNode(key, "dataset", digest, _dataset_label(d), d)
        return key

    def add_run(self, r: RunRecord) -> str:
        digest = record_digest(r)
        key = f"run:{digest}"
        if key not in self.nodes:
            self.nodes[key] = LineageNode(key, "run", digest, _run_label(r), r)
            self.runs_by_name.setdefault((r.family, r.run_id), []).append(key)
        return key

    def issue(self, kind: str, node: str, at: str, message: str) -> None:
        self.issues.append(LineageIssue(ISSUE_KINDS[kind], kind, node, at, message))

    def _known_id(self, name: str) -> str | None:
        for dataset_id in self.versions:
            if name == dataset_id or name.startswith(dataset_id + "/"):
                return dataset_id
        return None

    def _match_data(self, sha: str) -> tuple[str, list[tuple[str, str]]]:
        if sha in self.by_content:
            return "content", [(k, "the whole dataset") for k in self.by_content[sha]]
        if sha in self.by_split:
            return "split", self.by_split[sha]
        return "file", self.by_file.get(sha, [])

    def run_upstream(self, key: str, r: RunRecord) -> list[str]:
        found: list[str] = []
        for i, entry in enumerate(r.data):
            at = f"data[{i}]"
            known = self._known_id(entry.name)
            if entry.sha256 is None:
                if known is not None:
                    self.issue(
                        "undigested-data",
                        key,
                        at,
                        f"{entry.name!r} names dataset {known!r} without a sha256: not linked",
                    )
                continue
            via, matches = self._match_data(entry.sha256)
            if not matches:
                if known is not None:
                    self.issue(
                        "dataset-digest-mismatch",
                        key,
                        at,
                        f"{entry.name!r} has sha256 {entry.sha256}, which is no given version of"
                        f" {known!r} (content digests: {', '.join(self.versions[known])})",
                    )
                continue
            if len({m for m, _ in matches}) > 1:
                self.issue(
                    "ambiguous",
                    key,
                    at,
                    f"sha256 {entry.sha256} matches {len(matches)} dataset records; all are kept",
                )
            for dkey, what in matches:
                self.edges.append(LineageEdge(dkey, key, via, at, f"role {entry.role}, {what}"))
                found.append(dkey)
        return found

    def dataset_upstream(self, key: str, d: DatasetRecord) -> list[str]:
        if d.source_runs is None:
            self.issue("unrecorded-source-runs", key, "source_runs", "the producer did not say")
            return []
        found: list[str] = []
        for i, sr in enumerate(d.source_runs):
            at = f"source_runs[{i}]"
            by_name = self.runs_by_name.get((sr.family, sr.run_id), [])
            who = f"{sr.family}/{sr.run_id}" if sr.family else sr.run_id
            rkey = f"run:{sr.record_sha256}" if sr.record_sha256 else None
            if rkey is not None and rkey in self.nodes:
                rec = self.nodes[rkey].record
                assert isinstance(rec, RunRecord)
                if (rec.run_id, rec.family) != (sr.run_id, sr.family):
                    self.issue(
                        "source-run-identity-mismatch",
                        key,
                        at,
                        f"names {who}, but the run with that digest is {_run_label(rec)}",
                    )
                self.edges.append(LineageEdge(rkey, key, "source_run", at, who))
                found.append(rkey)
            elif rkey is not None:
                kind = "source-run-digest-mismatch" if by_name else "missing-source-run"
                also = f"; {len(by_name)} run(s) named {who} have other digests" if by_name else ""
                self.issue(kind, key, at, f"no run record has digest {sr.record_sha256}{also}")
            elif not by_name:
                self.issue("missing-source-run", key, at, f"no run record named {who}")
            else:
                self.issue(
                    "undigested-source-run", key, at, f"{who} linked by name only (no digest)"
                )
                if len(by_name) > 1:
                    self.issue(
                        "ambiguous", key, at, f"{len(by_name)} runs are named {who}; all kept"
                    )
                for rk in by_name:
                    self.edges.append(LineageEdge(rk, key, "source_run", at, f"{who} by name"))
                found += by_name
        return found


def resolve_lineage(
    run: RunRecord, datasets: Iterable[DatasetRecord], runs: Iterable[RunRecord] = ()
) -> Lineage:
    """The lineage upstream of `run`, resolved by digest against `datasets`
    and `runs` (which may include `run`). Pure; never raises for a broken
    link -- the result's `issues` say what does not hold."""
    res = _Resolver(datasets, runs)
    root = res.add_run(run)
    order: list[str] = []
    queue = [root]
    while queue:
        key = queue.pop(0)
        if key in order:
            continue
        order.append(key)
        rec = res.nodes[key].record
        if isinstance(rec, RunRecord):
            queue += res.run_upstream(key, rec)
        else:
            queue += res.dataset_upstream(key, rec)
    seen_edges: list[LineageEdge] = []
    for e in res.edges:
        if e not in seen_edges:
            seen_edges.append(e)
    return Lineage(
        root=root,
        nodes=tuple(res.nodes[k] for k in order),
        edges=tuple(seen_edges),
        issues=tuple(res.issues),
    )

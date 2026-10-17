"""Build and sign a run record.

`RunRecorder` accumulates one run -- inputs, outputs, queries, papers read,
models called, check results -- and `sign()` turns it into a DSSE envelope
over a JCS-canonical in-toto Statement. `from_spec()` builds the same thing
from plain JSON, which is how a tool outside this package (an agent script,
AgentMarkit's brief checker) records a run without importing anything but
this.

Files are named relative to the run's `root`, so a reviewer who reproduces
the tree anywhere can recompute every digest. Nothing here opens the
database or calls a model.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from attestation.run_record.canonical import canonicalize
from attestation.run_record.verify import (
    PAYLOAD_TYPE,
    PREDICATE_TYPE,
    STATEMENT_TYPE,
    pae,
    sha256_file,
)

DEPTHS = ("full", "abstract", "title")
SPEC_KEYS = frozenset(
    {"tool", "root", "started_at", "finished_at", "models", "queries", "papers"}
    | {"inputs", "outputs", "subjects", "checks"}
)


def now() -> str:
    """UTC now, to the second, as ISO 8601 with a Z."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Finding:
    """One result inside a check: what was judged, the verdict, and why."""

    subject: str
    verdict: str
    message: str = ""
    data: dict | None = None

    def to_json(self) -> dict:
        """This finding as a record entry; empty message/data are omitted."""
        out: dict = {"subject": self.subject, "verdict": str(self.verdict)}
        if self.message:
            out["message"] = self.message
        if self.data:
            out["data"] = self.data
        return out


@dataclass
class CheckResult:
    """The generic check entry. `summary` is always derived from `results`,
    never supplied, so the two cannot disagree in a record this built."""

    check: str
    results: list[Finding] = field(default_factory=list)
    target: str | None = None

    def to_json(self) -> dict:
        """This check as a record entry, its summary tallied from its results."""
        results = [r.to_json() for r in self.results]
        out: dict = {
            "check": self.check,
            "summary": dict(sorted(Counter(r["verdict"] for r in results).items())),
            "results": results,
        }
        if self.target:
            out["target"] = self.target
        return out


def _git_state() -> tuple[str | None, bool | None]:
    """HEAD and dirtiness of the checkout Attestation runs from, if it runs from
    one. Only `<checkout>/src/attestation/` counts: an installed wheel inside
    some other repository's `.venv` must not report THAT repository's sha."""
    here = Path(__file__).resolve()
    checkout = here.parents[3]
    if here.parents[2].name != "src" or not (checkout / ".git").exists():
        return None, None
    try:
        head = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return head or None, bool(status.strip())


def tool_identity(name: str) -> dict:
    """The `predicate.tool` block: name, Attestation version, checkout sha/dirtiness."""
    from attestation import __version__

    sha, dirty = _git_state()
    return {"name": name, "attestation_version": __version__, "git_sha": sha, "git_dirty": dirty}


class RunRecorder:
    """One run, accumulated as it happens, signed at the end."""

    def __init__(self, tool: str, *, root: Path | None = None, started_at: str | None = None):
        if not tool or not isinstance(tool, str):
            raise ValueError("a run record needs the tool's name")
        self.tool = tool
        self.root = Path(root if root is not None else Path.cwd()).expanduser().resolve()
        self.run_id = str(uuid.uuid4())
        self.started_at = started_at or now()
        self.finished_at: str | None = None
        self.models: list[dict] = []
        self.queries: list[dict] = []
        self.papers: list[dict] = []
        self.inputs: list[dict] = []
        self.outputs: list[dict] = []
        self.subjects: list[dict] = []
        self.checks: list[dict] = []

    def _descriptor(self, path, role: str | None = None) -> dict:
        p = Path(path).expanduser()
        p = (self.root / p) if not p.is_absolute() else p
        if not p.is_file():
            raise ValueError(f"no such file to record: {p}")
        out: dict = {
            "name": Path(os.path.relpath(p.resolve(), self.root)).as_posix(),
            "digest": {"sha256": sha256_file(p)},
        }
        annotations = {"bytes": p.stat().st_size} | ({"role": role} if role else {})
        out["annotations"] = annotations
        return out

    def add_input(self, path, role: str = "other") -> None:
        """Hash an input file under `role` (draft, results, bib, sources, other); once per name."""
        d = self._descriptor(path, role)
        if all(x["name"] != d["name"] for x in self.inputs):
            self.inputs.append(d)

    def add_output(self, path) -> None:
        """Hash a file the run produced."""
        self.outputs.append(self._descriptor(path))

    def add_subject(self, path) -> None:
        """Name a file the record is about when the run produced none."""
        self.subjects.append(self._descriptor(path))

    def add_query(self, source: str, query: str, hits: int) -> None:
        """One search against one source and how many hits it returned."""
        if not isinstance(hits, int) or isinstance(hits, bool) or hits < 0:
            raise ValueError(f"hits must be a count, got {hits!r}")
        self.queries.append({"source": str(source), "query": str(query), "hits": hits})

    def add_paper(self, paper_id: str, depth: str, title: str | None = None) -> None:
        """A paper the run read, at `depth` full, abstract or title."""
        if depth not in DEPTHS:
            raise ValueError(f"depth must be one of {', '.join(DEPTHS)}, got {depth!r}")
        paper = {"id": str(paper_id), "depth": depth} | ({"title": title} if title else {})
        self.papers.append(paper)

    def add_model(self, name: str, role: str = "chat", endpoint: str | None = None) -> None:
        """A model the run called."""
        model = {"name": str(name), "role": str(role)}
        self.models.append(model | ({"endpoint": endpoint} if endpoint else {}))

    def add_check(self, result: CheckResult) -> None:
        """A checker's results, in the generic check shape."""
        self.checks.append(result.to_json())

    def _subject(self) -> list[dict]:
        """in-toto requires a subject: what this record is ABOUT. Outputs if
        the run made any, else explicit subjects, else the drafts it checked,
        else every input."""
        drafts = [d for d in self.inputs if d["annotations"].get("role") == "draft"]
        chosen = self.outputs or self.subjects or drafts or self.inputs
        if not chosen:
            raise ValueError("a run record must name at least one file (input or output)")
        return [{"name": d["name"], "digest": d["digest"]} for d in chosen]

    def statement(self, signer) -> dict:
        """The in-toto Statement this run signs to, naming `signer` inside it."""
        return {
            "_type": STATEMENT_TYPE,
            "subject": self._subject(),
            "predicateType": PREDICATE_TYPE,
            "predicate": {
                "run": {
                    "id": self.run_id,
                    "started_at": self.started_at,
                    "finished_at": self.finished_at or now(),
                },
                "tool": tool_identity(self.tool),
                "models": self.models,
                "queries": self.queries,
                "papers": self.papers,
                "inputs": self.inputs,
                "outputs": self.outputs,
                "checks": self.checks,
                "signer": signer.identity(),
            },
        }

    def sign(self, signer) -> dict:
        """The DSSE envelope: a JCS payload and an ed25519 signature over its PAE."""
        try:
            payload = canonicalize(self.statement(signer))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"record holds a value JSON cannot carry exactly: {exc}") from exc
        return {
            "payloadType": PAYLOAD_TYPE,
            "payload": base64.b64encode(payload).decode("ascii"),
            "signatures": [
                {
                    "keyid": signer.keyid,
                    "sig": base64.b64encode(signer.sign(pae(PAYLOAD_TYPE, payload))).decode(),
                }
            ],
        }


def write_record(envelope: dict, out: Path, *, force: bool = False) -> Path:
    """Write the envelope. New files only unless `force`, like `runs.record`."""
    out = Path(out).expanduser()
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists; pass force to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
    return out


def _findings(results) -> list[Finding]:
    return [
        Finding(r["subject"], r["verdict"], r.get("message", ""), r.get("data")) for r in results
    ]


_SPEC_ADDERS = (
    ("models", lambda rec, m: rec.add_model(m["name"], m.get("role", "chat"), m.get("endpoint"))),
    ("queries", lambda rec, q: rec.add_query(q["source"], q["query"], q["hits"])),
    ("papers", lambda rec, p: rec.add_paper(p["id"], p["depth"], p.get("title"))),
    ("inputs", lambda rec, i: rec.add_input(i["path"], i.get("role", "other"))),
    ("outputs", lambda rec, o: rec.add_output(o["path"])),
    ("subjects", lambda rec, s: rec.add_subject(s["path"])),
    (
        "checks",
        lambda rec, c: rec.add_check(
            CheckResult(c["check"], _findings(c.get("results", [])), c.get("target"))
        ),
    ),
)


def from_spec(spec: dict, *, root: Path | str | None = None) -> RunRecorder:
    """A recorder from plain JSON -- the generic entry point for other tools.

    `inputs`/`outputs`/`subjects` give `path` (relative to `root`) and, for
    inputs, `role`; `checks` give `check`, optional `target`, and `results`
    of `{subject, verdict, message?, data?}`. Unknown top-level keys are
    refused: a misspelt `papers` would otherwise vanish from the record."""
    unknown = set(spec) - SPEC_KEYS
    if unknown:
        raise ValueError(f"unknown spec field(s): {', '.join(sorted(unknown))}")
    base = root if root is not None else spec.get("root")
    rec = RunRecorder(
        spec.get("tool", ""),
        root=Path(base) if base is not None else None,
        started_at=spec.get("started_at"),
    )
    rec.finished_at = spec.get("finished_at")
    try:
        for key, add in _SPEC_ADDERS:
            for entry in spec.get(key, []):
                add(rec, entry)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed spec: missing or wrong field {exc}") from exc
    return rec

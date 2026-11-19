"""Conformance: every rule a run directory must keep, checked at once.

The reader (`read`) refuses a record that breaks the record format, and stops
at the first problem. A run directory has more rules than that -- the record's
bytes must be the canonical ones, the manifest must match the files, the
history sidecar its digest, `state.json` its record -- and nothing else
checks them. `check_run_dir` runs every rule and returns every violation, each
naming its rule, the file it is in and the JSON path inside it.

Standard library only. The pytest plugin over this module is
`attestation_schema.pytest_plugin` (the `[conformance]` extra); a producer
that does not use pytest calls `check_run_dir` / `assert_conformant` directly.

Checks only; never writes, never changes a format. Spec:
docs/superpowers/specs/2026-11-19-conformance-suite-design.md.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from attestation_schema import layout
from attestation_schema.canonical import canonicalize
from attestation_schema.history import read_history
from attestation_schema.jobs import TERMINAL, JobSpec
from attestation_schema.record import (
    _NAMESPACE,
    _RESERVED_NAMESPACE,
    HistoryError,
    RecordError,
    RunRecord,
    _source,
    dumps,
    from_dict,
    parse_strict,
)
from attestation_schema.state import StateFile, dumps_state, loads_state, spec_conflicts

RULES = (
    "read",
    "canonical",
    "manifest",
    "history",
    "state",
    "terminal",
    "spec",
    "source",
    "credentials",
    "ext-namespace",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Violation:
    """One broken rule: which, in which file, where in it, and why."""

    rule: str
    file: str
    json_path: str
    message: str

    def __str__(self) -> str:
        return f"{self.file} at {self.json_path or '$'}: [{self.rule}] {self.message}"


@dataclass(frozen=True)
class Report:
    """Every violation found in one run directory (empty = conformant)."""

    run_dir: Path
    violations: tuple[Violation, ...]
    record: RunRecord | None = None
    state: StateFile | None = None

    @property
    def ok(self) -> bool:
        """True when no rule is broken."""
        return not self.violations

    @property
    def rules(self) -> list[str]:
        """The broken rules, in `RULES` order, each once."""
        broken = {v.rule for v in self.violations}
        return [r for r in RULES if r in broken]

    def format(self) -> str:
        """A multi-line message naming the run directory and each violation."""
        if self.ok:
            return f"{self.run_dir}: conforms"
        lines = [f"{self.run_dir}: {len(self.violations)} conformance violation(s)"]
        lines += [f"  {v}" for v in self.violations]
        return "\n".join(lines)


class ConformanceError(AssertionError):
    """A run directory broke at least one rule; `report` holds every violation."""

    def __init__(self, report: Report):
        self.report = report
        super().__init__(report.format())


# ---- JSON paths ---------------------------------------------------------------

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")


def _key(path: str, key: object) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    k = str(key)
    return f"{path}.{k}" if _IDENT.fullmatch(k) else f"{path}[{json.dumps(k)}]"


def _jsonpath(reader_path: str) -> str:
    """The reader's `metrics[2].direction` as `$.metrics[2].direction`."""
    if not reader_path:
        return "$"
    return "$" + ("" if reader_path.startswith("[") else ".") + reader_path


# ---- credentials ----------------------------------------------------------------

_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>]+")
_QUERY_SECRETS = frozenset(
    {
        "token",
        "access_token",
        "api_key",
        "apikey",
        "key",
        "sig",
        "signature",
        "password",
        "secret",
        "client_secret",
        "x-amz-signature",
        "x-amz-credential",
        "x-amz-security-token",
        "x-goog-signature",
        "x-goog-credential",
    }
)
_TOKEN_SHAPES = (
    ("an AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    (
        "a GitHub token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,})"),
    ),
    ("a GitLab token", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}")),
    ("a Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("a Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("an API secret key (sk-...)", re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}")),
    ("a private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)
_SECRET_KEYS = (
    "password",
    "passwd",
    "secret",
    "client_secret",
    "secret_key",
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "refresh_token",
    "bearer_token",
    "session_token",
    "private_key",
)
_REDACTED = re.compile(
    r"\*+|(?i:redacted)|<[^<>]*>|\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Z_][A-Z0-9_]*"
)


def _secret_key(key: str) -> bool:
    k = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    return any(k == s or k.endswith("_" + s) for s in _SECRET_KEYS)


def credential_in(text: str) -> str | None:
    """Why `text` carries a credential, or None. Heuristic, and deliberately
    so: a false positive costs a rename; a leaked key in a record whose
    digest is its identity cannot be edited out."""
    for match in _URL.finditer(text):
        try:
            parts = urlsplit(match.group(0))
            password, username = parts.password, parts.username
        except ValueError:
            continue
        web = parts.scheme.lower().rsplit("+", 1)[-1] in ("http", "https")
        if password is not None or (web and username):
            return "a URL with credentials (user:password@); strip them"
        names = {k.lower() for k, _ in parse_qsl(parts.query, keep_blank_values=True)}
        hit = sorted(names & _QUERY_SECRETS)
        if hit:
            return f"a URL whose query carries a credential ({hit[0]}=); strip the query"
    for what, pattern in _TOKEN_SHAPES:
        if pattern.search(text):
            return f"what looks like {what}; remove it"
    return None


def _walk(value: Any, path: str) -> Iterator[tuple[str, str | None, Any]]:
    """(json path, key it sits under, value) for every node."""
    if isinstance(value, dict):
        for k, v in value.items():
            p = _key(path, k)
            yield p, str(k), v
            yield from _walk_children(v, p)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            p = _key(path, i)
            yield p, None, v
            yield from _walk_children(v, p)


def _walk_children(value: Any, path: str) -> Iterator[tuple[str, str | None, Any]]:
    if isinstance(value, dict | list):
        yield from _walk(value, path)


def credential_findings(obj: Any, root: str = "$") -> list[tuple[str, str]]:
    """`(json path, reason)` for every credential in a parsed JSON value."""
    out: list[tuple[str, str]] = []
    for path, key, value in _walk(obj, root):
        if key is not None and credential_in(key):
            out.append((path, f"its key is {credential_in(key)}"))
        if not isinstance(value, str):
            continue
        reason = credential_in(value)
        if reason is None and key is not None and _secret_key(key) and value.strip():
            if not _REDACTED.fullmatch(value.strip()):
                reason = f"a value under the secret-named key {key!r}; redact it"
        if reason:
            out.append((path, reason))
    return out


# ---- the checker ---------------------------------------------------------------


class _Collector:
    def __init__(self) -> None:
        self.items: list[Violation] = []
        self._seen: set[tuple[str, str, str]] = set()

    def add(self, rule: str, file: Path | str, json_path: str, message: str) -> None:
        key = (rule, str(file), json_path)
        if key in self._seen:
            return
        self._seen.add(key)
        self.items.append(Violation(rule, str(file), json_path, message))


def _reader_rule(err: RecordError) -> str:
    """The rule a reader refusal belongs to (most are `read`)."""
    msg = str(err)
    if "credential" in msg:
        return "credentials"
    if re.fullmatch(r"dependencies\[\d+\]\.source", err.path):
        return "source"
    if err.path.startswith("ext.") and (
        "namespace" in msg or "an extension must be an object" in msg
    ):
        return "ext-namespace"
    return "read"


def _raw_record_checks(obj: Any, file: Path, out: _Collector) -> None:
    """The rules checked on the parsed JSON, independently of the reader, so
    each bad source, namespace and credential is reported -- not just the
    first thing the reader trips on."""
    if not isinstance(obj, dict):
        return
    deps = obj.get("dependencies")
    for i, dep in enumerate(deps if isinstance(deps, list) else []):
        if isinstance(dep, dict) and "source" in dep:
            where = f"dependencies[{i}].source"
            try:
                _source(dep["source"], where)
            except RecordError as exc:
                rule = "credentials" if "credential" in str(exc) else "source"
                out.add(rule, file, _jsonpath(where), str(exc).split(": ", 1)[-1])
    ext = obj.get("ext")
    for ns, body in ext.items() if isinstance(ext, dict) else []:
        where = _key("$.ext", ns)
        if not _NAMESPACE.fullmatch(ns):
            out.add("ext-namespace", file, where, f"namespace {ns!r} is not {_NAMESPACE.pattern}")
        elif ns.split(".")[0] == _RESERVED_NAMESPACE:
            out.add("ext-namespace", file, where, "namespace 'attestation' is reserved")
        elif not isinstance(body, dict):
            out.add("ext-namespace", file, where, "an extension must be an object")
    for path, reason in credential_findings(obj):
        out.add("credentials", file, path, reason)


def _first_difference(a: bytes, b: bytes) -> int:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return min(len(a), len(b))


def _check_record(rd: Path, out: _Collector) -> RunRecord | None:
    file = rd / layout.RUN_JSON
    raw = file.read_bytes()
    try:
        obj = parse_strict(raw)
    except RecordError as exc:
        out.add("read", file, "$", str(exc))
        return None
    _raw_record_checks(obj, file, out)
    try:
        record = from_dict(obj)
    except RecordError as exc:
        out.add(_reader_rule(exc), file, _jsonpath(exc.path), str(exc).split(": ", 1)[-1])
        return None
    canonical = dumps(record)
    if canonical != raw:
        at = _first_difference(canonical, raw)
        out.add(
            "canonical",
            file,
            "$",
            f"the bytes are not the canonical form (first difference at byte {at}): its sha256"
            " is not its record_digest; write it with attestation_schema.write",
        )
    return record


def _check_history(rd: Path, record: RunRecord, out: _Collector) -> None:
    file = rd / layout.HISTORY_JSONL
    if record.history is None:
        if file.exists():
            out.add(
                "history",
                file,
                "$",
                "a history sidecar exists but the record's history is null: nothing binds it",
            )
        return
    try:
        read_history(rd, record)
    except HistoryError as exc:
        line = exc.path.partition(":")[2]
        where = f"line {line}" if line else "$"
        if "." in line:
            n, _, field = line.partition(".")
            where = f"line {n}: $.{field}"
        out.add("history", file, where, str(exc).split(": ", 1)[-1])
        return
    for i, raw in enumerate(file.read_bytes().splitlines(), start=1):
        for path, reason in credential_findings(json.loads(raw), f"line {i}: $"):
            out.add("credentials", file, path, reason)


def _manifest_shape(obj: Any, file: Path, out: _Collector) -> dict[str, str] | None:
    if not isinstance(obj, dict) or set(obj) != {"schema", "files"}:
        keys = sorted(obj) if isinstance(obj, dict) else type(obj).__name__
        out.add("manifest", file, "$", f"expected exactly {{schema, files}}, got {keys}")
        return None
    if obj["schema"] != layout.DIGESTS_SCHEMA:
        out.add("manifest", file, "$.schema", f"expected {layout.DIGESTS_SCHEMA!r}")
        return None
    files = obj["files"]
    if not isinstance(files, dict):
        out.add("manifest", file, "$.files", "expected an object {path: sha256}")
        return None
    good = True
    for name, digest in files.items():
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            out.add("manifest", file, _key("$.files", name), "expected 64 lowercase hex")
            good = False
    return files if good else None


def _compare_outputs(rd: Path, listed: dict[str, str], file: Path, out: _Collector) -> None:
    try:
        actual = layout.build_manifest(rd)["files"]
    except layout.LayoutError as exc:
        out.add("manifest", file, "$.files", f"cannot rebuild the manifest: {exc}")
        return
    for name in sorted(set(listed) | set(actual)):
        where = _key("$.files", name)
        if name not in actual:
            out.add("manifest", file, where, "listed, but the file is missing")
        elif name not in listed:
            out.add("manifest", file, where, "the file exists but the manifest does not list it")
        elif listed[name] != actual[name]:
            out.add(
                "manifest",
                file,
                where,
                f"sha256 is {actual[name]}, the manifest says {listed[name]}",
            )


def _check_manifest(
    rd: Path, record: RunRecord | None, verify_outputs: bool, out: _Collector
) -> None:
    file = rd / layout.DIGESTS_JSON
    run_json = rd / layout.RUN_JSON
    bound = [d for d in (record.data if record else []) if d.name == layout.DIGESTS_JSON]
    if not file.is_file():
        if bound:
            out.add(
                "manifest",
                run_json,
                f"$.data[{record.data.index(bound[0])}]" if record else "$.data",
                f"the record binds {layout.DIGESTS_JSON} but it is missing",
            )
        return
    raw = file.read_bytes()
    try:
        obj = parse_strict(raw)
    except RecordError as exc:
        out.add("manifest", file, "$", str(exc))
        return
    for path, reason in credential_findings(obj):
        out.add("credentials", file, path, reason)
    listed = _manifest_shape(obj, file, out)
    if canonicalize(obj) != raw:
        out.add(
            "manifest", file, "$", "the bytes are not the canonical form (RFC 8785, no newline)"
        )
    if record is not None:
        digest = hashlib.sha256(raw).hexdigest()
        if not bound:
            out.add(
                "manifest",
                run_json,
                "$.data",
                f"{layout.DIGESTS_JSON} exists but the record does not list it with role"
                " 'manifest': nothing binds it",
            )
        else:
            i = record.data.index(bound[0])
            if bound[0].role != layout.MANIFEST_ROLE:
                out.add("manifest", run_json, f"$.data[{i}].role", "expected 'manifest'")
            if bound[0].sha256 != digest:
                out.add(
                    "manifest",
                    run_json,
                    f"$.data[{i}].sha256",
                    f"the record binds {bound[0].sha256}, {layout.DIGESTS_JSON} is {digest}",
                )
    if listed is not None and verify_outputs and record is not None:
        _compare_outputs(rd, listed, file, out)


def _check_state(rd: Path, record: RunRecord | None, out: _Collector) -> StateFile | None:
    file = rd / layout.STATE_JSON
    if not file.is_file():
        return None
    raw = file.read_bytes()
    try:
        st = loads_state(raw)
    except RecordError as exc:
        out.add("state", file, _jsonpath(exc.path), str(exc).split(": ", 1)[-1])
        return None
    for path, reason in credential_findings(st.to_dict()):
        out.add("credentials", file, path, reason)
    if dumps_state(st) != raw:
        out.add("state", file, "$", "the bytes are not the canonical form (RFC 8785, no newline)")
    if record is not None and st.run_id != record.run_id:
        out.add("state", file, "$.run_id", f"is {st.run_id!r}, the record's is {record.run_id!r}")
    if st.state in TERMINAL and not (rd / layout.RUN_JSON).is_file():
        out.add(
            "terminal",
            file,
            "$.state",
            f"the job is {st.state.value} but {layout.RUN_JSON} is missing: every terminal job"
            " has a record",
        )
    if record is not None:
        if record.status is None:
            out.add(
                "terminal",
                rd / layout.RUN_JSON,
                "$.status",
                "a job's record says how it ended; null is for importers that could not observe",
            )
        elif st.state in TERMINAL and record.status != st.state.value:
            out.add(
                "terminal",
                file,
                "$.state",
                f"the job is {st.state.value}, its record says {record.status}",
            )
    return st


def check_run_dir(
    run_dir: str | os.PathLike[str],
    *,
    spec: JobSpec | None = None,
    verify_outputs: bool = True,
) -> Report:
    """Every rule in `RULES` against one run directory; never raises for a
    non-conforming directory (that is what the report is for).

    `spec`: the `JobSpec` the run was submitted with, if known; checks the
    record against it (`spec` rule). `verify_outputs`: re-hash every file the
    manifest lists (the cost scales with the run's outputs)."""
    rd = Path(run_dir)
    out = _Collector()
    has_record = (rd / layout.RUN_JSON).is_file()
    if not layout.is_run_dir(rd):
        out.add(
            "read",
            rd / layout.RUN_JSON,
            "$",
            f"not a run directory: neither {layout.RUN_JSON} nor {layout.STATE_JSON} exists",
        )
        return Report(rd, tuple(out.items))
    record = _check_record(rd, out) if has_record else None
    if record is not None:
        _check_history(rd, record, out)
        if spec is not None:
            for conflict in spec_conflicts(spec, record):
                field = conflict.split(" ", 1)[0]
                out.add("spec", rd / layout.RUN_JSON, f"$.{field}", conflict)
    _check_manifest(rd, record, verify_outputs and has_record, out)
    st = _check_state(rd, record, out)
    return Report(rd, tuple(out.items), record, st)


def assert_conformant(
    run_dir: str | os.PathLike[str],
    *,
    spec: JobSpec | None = None,
    verify_outputs: bool = True,
) -> Report:
    """`check_run_dir`, raising `ConformanceError` (an AssertionError) listing
    every violation; returns the report when there are none."""
    report = check_run_dir(run_dir, spec=spec, verify_outputs=verify_outputs)
    if not report.ok:
        raise ConformanceError(report)
    return report


def _run_dirs_for(match: Path) -> list[Path]:
    if match.is_file() and match.name in ("run.json", "state.json"):
        match = match.parent.parent
    if layout.is_run_dir(match):
        return [match]
    return layout.find_runs(match) if match.is_dir() else []


def find_targets(
    patterns: Iterable[str], base: str | os.PathLike[str] = "."
) -> tuple[list[Path], list[str]]:
    """`(run directories, patterns that matched none)`. A pattern is a run
    directory, a directory holding runs, a `run.json`/`state.json` path, or a
    glob (`**` allowed), relative to `base`. A pattern matching nothing is
    returned, never dropped: a typo must not turn a suite into a no-op."""
    found: list[Path] = []
    empty: list[str] = []
    for pattern in patterns:
        full = Path(base) / os.path.expanduser(pattern)
        matches = [Path(m) for m in sorted(glob.glob(str(full), recursive=True))]
        dirs = [d for m in matches for d in _run_dirs_for(m)]
        if not dirs:
            empty.append(pattern)
        found += [d for d in dirs if d not in found]
    return found, empty

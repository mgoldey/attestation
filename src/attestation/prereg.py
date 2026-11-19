"""Hypothesis preregistration: a signed prediction, evaluated later against the ledger.

A hypothesis names a family, a metric, a predicted ordering of arms by the
metric's raw values (`a < b`), and a minimum margin. `register` writes it as
three files under the workspace -- the `hypotheses.toml` index, the
hypothesis's canonical JSON (whose sha256 is its id) and a signed run record
whose subject is that file -- and `audit` reads them back, reporting any text
that no longer matches what was signed. `evaluate` is pure: it takes the dict
`ledger.compare()` returns and decides confirmed / refuted / inconclusive /
untested by the rule in the design spec.

What a registration proves, and what it cannot, is `LIMITS`, printed by every
surface that reports one. See
`docs/superpowers/specs/2026-11-19-preregistration-design.md`.

No database, no model, no network: the ledger is the caller's to read.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from attestation.run_record.canonical import canonicalize

SCHEMA = "attestation/prereg/v1"
INDEX_NAME = "hypotheses.toml"
RECORD_DIR = "prereg"
TOOL_NAME = "attest prereg add"
CHECK_NAME = "prereg"
ID_PREFIX = 16  # hex digits of the id used in file names and listings
MIN_ID_PREFIX = 8  # shortest prefix `--supersedes` accepts

# Hypotheses one runs.compare reply may carry. Each summary is ~200 chars; five
# plus LIMITS stays well inside the 7000-char ceiling beside 20 arms.
MAX_HYPOTHESES_SHOWN = 5

# The trust model, in the words every surface prints. Tests pin each one in the
# CLI and the MCP payload: a signature over a "confirmed" reads as "predicted
# in advance" to anyone not told otherwise.
LIMITS = (
    "proved: the key named in each record signed exactly this hypothesis text, and the"
    " text has not changed since (any edit reads as altered)",
    "evidence only: registered_at is this machine's clock, compared with artifact"
    " modification times on this disk; whoever controls the machine controls both, so"
    " post_hoc set suggests a late registration and post_hoc unset proves nothing",
    "cannot check: that the prediction was written before anyone saw the data, that no"
    " registration was deleted along with its record, or who holds the key unless you"
    " pin it (attest record verify --keyid)",
)

_FIELDS = ("family", "metric", "expect", "by", "allow_single_run", "note", "supersedes")
_INDEX_KEYS = frozenset({"id", "registered_at", *_FIELDS})
_OPS = ("<", ">")
_ARM_RE = re.compile(r"[^\s<>]+")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class VerdictKind(StrEnum):
    """What the evidence says about a hypothesis."""

    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"
    UNTESTED = "untested"


POST_HOC, ALTERED, SUPERSEDED = "post_hoc", "altered", "superseded"


def parse_expect(text: str) -> tuple[list[str], list[str]]:
    """`"a < b > c"` -> (["a", "b", "c"], ["<", ">"]). Raises ValueError."""
    tokens = re.findall(r"<|>|[^\s<>]+", text or "")
    arms, ops = tokens[0::2], tokens[1::2]
    if (
        len(arms) < 2
        or len(arms) != len(ops) + 1
        or any(op not in _OPS for op in ops)
        or any(not _ARM_RE.fullmatch(a) for a in arms)
    ):
        raise ValueError(f"--expect takes arms joined by < or >, e.g. 'a < b'; got {text!r}")
    if len(set(arms)) != len(arms):
        raise ValueError(f"--expect names an arm twice: {text!r}")
    return arms, ops


def normalize_expect(text: str) -> str:
    """The canonical spelling of an expectation: single spaces around each operator."""
    arms, ops = parse_expect(text)
    out = arms[0]
    for op, arm in zip(ops, arms[1:], strict=True):
        out += f" {op} {arm}"
    return out


def _clean_text(name: str, value) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {type(value).__name__}")
    if _CONTROL_RE.search(value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _margin(value) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"by must be a number, got {value!r}")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"by is a minimum margin and must be finite and >= 0, got {value}")
    return float(value)


@dataclass(frozen=True)
class Hypothesis:
    """A prediction about one family's arms on one metric. Its id is the
    sha256 of `content()` in canonical JSON, so equal content is one id."""

    family: str
    metric: str
    expect: str
    by: float
    allow_single_run: bool = False
    note: str = ""
    supersedes: str | None = None

    def __post_init__(self):
        for name in ("family", "metric", "note"):
            _clean_text(name, getattr(self, name))
        if not self.family.strip() or not self.metric.strip():
            raise ValueError("a hypothesis needs a family and a metric")
        object.__setattr__(self, "expect", normalize_expect(_clean_text("expect", self.expect)))
        object.__setattr__(self, "by", _margin(self.by))
        if not isinstance(self.allow_single_run, bool):
            raise ValueError("allow_single_run must be true or false")
        if self.supersedes is not None and not re.fullmatch(r"[0-9a-f]{64}", self.supersedes):
            raise ValueError(f"supersedes must be a full hypothesis id, got {self.supersedes!r}")

    @property
    def arms(self) -> list[str]:
        """The arm names the expectation predicts an ordering of."""
        return parse_expect(self.expect)[0]

    def content(self) -> dict:
        """Everything that is hashed and signed -- the hypothesis, nothing else."""
        return {"schema": SCHEMA, **{name: getattr(self, name) for name in _FIELDS}}

    def canonical(self) -> bytes:
        """`content()` as RFC 8785 JCS bytes."""
        return canonicalize(self.content())

    @property
    def id(self) -> str:
        """The hex sha256 of the canonical content."""
        return hashlib.sha256(self.canonical()).hexdigest()

    def describe(self) -> str:
        """One line: `family metric: a < b by 0.01`."""
        return f"{self.family} {self.metric}: {self.expect} by {self.by:g}"

    @classmethod
    def from_content(cls, data: dict) -> Hypothesis:
        """The inverse of `content()`; raises ValueError on anything else."""
        if not isinstance(data, dict) or data.get("schema", SCHEMA) != SCHEMA:
            raise ValueError("not a prereg v1 hypothesis")
        unknown = set(data) - {"schema", *_FIELDS}
        if unknown:
            raise ValueError(f"unknown hypothesis field(s): {', '.join(sorted(unknown))}")
        try:
            return cls(**{k: v for k, v in data.items() if k != "schema"})
        except TypeError as exc:
            raise ValueError(f"malformed hypothesis: {exc}") from exc


# --- the verdict -----------------------------------------------------------


@dataclass
class HypothesisVerdict:
    """What `evaluate` concluded, with the per-pair reasoning and flags."""

    verdict: VerdictKind
    pairs: list[dict] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    post_hoc: bool = False
    predates: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict:
        """JSON-ready form."""
        return {
            "verdict": str(self.verdict),
            "pairs": self.pairs,
            "reasons": self.reasons,
            "post_hoc": self.post_hoc,
            "predates": self.predates,
        }


def _at_least(gap: float, by: float) -> bool:
    """`gap >= by`, forgiving float noise: 0.06 - 0.05 is 0.00999..."""
    return gap > by or math.isclose(gap, by, rel_tol=1e-9, abs_tol=1e-12)


def _interval(arm: dict) -> tuple[float, float] | None:
    lo, hi = arm.get("ci_low"), arm.get("ci_high")
    if lo is None or hi is None:
        return None
    return float(lo), float(hi)


def _arm_text(arm: dict) -> str:
    text = f"{arm['name']} {float(arm['value']):g}"
    span = _interval(arm)
    if span is not None:
        text += f" [{span[0]:g}, {span[1]:g}] over {arm.get('n_runs', '?')} runs"
    return text


def _judge_pair(lo: dict, hi: dict, by: float, allow_single_run: bool) -> tuple[VerdictKind, str]:
    """`lo` is predicted lower than `hi` by at least `by`. Intervals when both
    arms carry one, point values only when the hypothesis allowed them."""
    lo_span, hi_span = _interval(lo), _interval(hi)
    if lo_span is not None and hi_span is not None:
        if _at_least(hi_span[0] - lo_span[1], by):
            return VerdictKind.CONFIRMED, "intervals separated by at least the margin"
        if not _at_least(hi_span[1] - lo_span[0], by):
            return VerdictKind.REFUTED, "intervals exclude a gap as large as the margin"
        return VerdictKind.INCONCLUSIVE, "intervals neither separate by the margin nor exclude it"
    gap = float(hi["value"]) - float(lo["value"])
    if not allow_single_run:
        bare = lo["name"] if lo_span is None else hi["name"]
        return (
            VerdictKind.INCONCLUSIVE,
            f"{bare} has no interval (single run or too few replicates), and point values"
            " decide only with allow_single_run",
        )
    if _at_least(gap, by):
        return VerdictKind.CONFIRMED, f"point values differ by {gap:g} (allow_single_run)"
    return VerdictKind.REFUTED, f"point values differ by {gap:g} (allow_single_run)"


def _find_arm(name: str, arms: list[dict]) -> tuple[dict | None, str | None]:
    """The arm named exactly `name`, or why there is none. A name that is only
    one replicate of another arm is refused: that would let a hypothesis pick
    the seed that agrees with it."""
    for arm in arms:
        if arm.get("name") == name:
            if arm.get("value") is None:
                return None, f"{name} has no value for this metric"
            return arm, None
    # By the seed rule itself, not membership in `runs`: that list is capped at
    # replicates.MAX_RUN_NAMES, so a sixth seed used to read "no arm named".
    from attestation import replicates

    parsed = replicates.seed_of(name)
    for arm in arms:
        if name in (arm.get("runs") or []) or (
            arm.get("n_runs") and parsed is not None and parsed[0] == arm.get("name")
        ):
            return None, f"{name} is one replicate of arm {arm['name']}; name the arm, not a seed"
    return None, f"no arm named {name} in this comparison"


def _pair_verdict(left: str, op: str, right: str, arms: list[dict], hyp: Hypothesis) -> dict:
    lo_name, hi_name = (left, right) if op == "<" else (right, left)
    lo, lo_why = _find_arm(lo_name, arms)
    hi, hi_why = _find_arm(hi_name, arms)
    out = {"pair": f"{left} {op} {right}"}
    if lo is None or hi is None:
        why = lo_why or hi_why or "arm not found"
        kind = VerdictKind.INCONCLUSIVE if "replicate" in why else VerdictKind.UNTESTED
        return out | {"verdict": str(kind), "why": why}
    kind, why = _judge_pair(lo, hi, hyp.by, hyp.allow_single_run)
    return out | {"verdict": str(kind), "why": why, "arms": [_arm_text(lo), _arm_text(hi)]}


def _overall(kinds: list[str]) -> VerdictKind:
    for kind in (VerdictKind.REFUTED, VerdictKind.UNTESTED, VerdictKind.INCONCLUSIVE):
        if kind in kinds:
            return kind
    return VerdictKind.CONFIRMED


def parse_time(text: str | None) -> datetime | None:
    """An ISO 8601 UTC time (`...Z`) as an aware datetime, or None."""
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def iso(moment: datetime) -> str:
    """A datetime as ISO 8601 UTC to the second, with a Z."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _predates(hyp: Hypothesis, evidence_times: dict, registered_at: str | None) -> dict:
    when = parse_time(registered_at)
    if when is None:
        return {}
    # registered_at is stored to the second, mtimes are finer: compare both at
    # whole seconds, and an artifact from the registration's own second counts
    # as predating it -- post_hoc errs toward flagging, never toward clearing.
    return {
        arm: iso(evidence_times[arm])
        for arm in hyp.arms
        if arm in evidence_times and evidence_times[arm].replace(microsecond=0) <= when
    }


def evaluate(
    hypothesis: Hypothesis,
    compare_result: dict,
    evidence_times: dict[str, datetime] | None = None,
    *,
    registered_at: str | None = None,
) -> HypothesisVerdict:
    """Judge `hypothesis` against `ledger.compare()`'s result for its family
    and metric. Pure: no file, clock or database is read here.

    `evidence_times` maps an arm name to when its evidence existed (the
    caller's choice of clock -- `evidence_times()` uses artifact mtimes);
    any named arm earlier than `registered_at` sets `post_hoc`, and the
    verdict is still computed and reported beside it."""
    predates = _predates(hypothesis, evidence_times or {}, registered_at)
    if compare_result.get("metric") != hypothesis.metric:
        why = (
            f"compared on {compare_result.get('metric')!r}, but the hypothesis is about"
            f" {hypothesis.metric!r}; compare on that metric"
        )
        return HypothesisVerdict(
            VerdictKind.UNTESTED, reasons=[why], post_hoc=bool(predates), predates=predates
        )
    arms = compare_result.get("arms") or []
    names, ops = parse_expect(hypothesis.expect)
    pairs = [
        _pair_verdict(left, op, right, arms, hypothesis)
        for left, op, right in zip(names[:-1], ops, names[1:], strict=True)
    ]
    kind = _overall([p["verdict"] for p in pairs])
    reasons = [p["why"] for p in pairs]
    return HypothesisVerdict(kind, pairs, reasons, post_hoc=bool(predates), predates=predates)


def evidence_times(compare_result: dict, names: list[str]) -> dict[str, datetime]:
    """When each named arm's earliest artifact was last written, by this disk's
    clock. A pooled arm is judged by EVERY replicate's file (`source_paths`),
    not only the first: re-running one seed must not hide that the others
    predate the registration. Evidence, not proof: an mtime is the LAST write,
    so a rewritten artifact hides that it existed earlier."""
    out: dict[str, datetime] = {}
    for arm in compare_result.get("arms") or []:
        if arm.get("name") not in names:
            continue
        times = [_mtime(p) for p in arm.get("source_paths") or [arm.get("source_path")] if p]
        known = [t for t in times if t is not None]
        if known:
            out[arm["name"]] = min(known)
    return out


def _mtime(path: str) -> datetime | None:
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime, UTC)
    except OSError:
        return None


# --- the index file ----------------------------------------------------------


@dataclass
class IndexEntry:
    """One `[[hypothesis]]` table: the id and time it claims, and its text."""

    id: str
    registered_at: str | None
    hypothesis: Hypothesis | None
    problem: str | None = None


def _entry(table: dict, position: int) -> IndexEntry:
    raw_id, raw_stamp = table.get("id"), table.get("registered_at")
    entry_id = raw_id if isinstance(raw_id, str) else f"#{position}"
    stamp = raw_stamp if isinstance(raw_stamp, str) else None
    unknown = set(table) - _INDEX_KEYS
    if unknown:
        problem = f"unknown key(s) {', '.join(sorted(unknown))}"
        return IndexEntry(entry_id, stamp, None, problem)
    try:
        hyp = Hypothesis.from_content({k: v for k, v in table.items() if k in _FIELDS})
    except ValueError as exc:
        return IndexEntry(entry_id, stamp, None, str(exc))
    return IndexEntry(entry_id, stamp, hyp)


def parse_file(path: Path) -> list[IndexEntry]:
    """Every `[[hypothesis]]` in `hypotheses.toml`; [] when there is no file.
    Raises ValueError when the file is not TOML at all."""
    if not path.is_file():
        return []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path} is not valid TOML: {exc}") from exc
    tables = data.get("hypothesis", [])
    if not isinstance(tables, list):
        raise ValueError(f"{path}: `hypothesis` must be an array of tables ([[hypothesis]])")
    return [_entry(t if isinstance(t, dict) else {}, i) for i, t in enumerate(tables, 1)]


def _toml_value(value) -> str:
    # json.dumps of a string with no control characters (Hypothesis refuses
    # them) is a valid TOML basic string; repr of a finite float is a TOML float.
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return json.dumps(value, ensure_ascii=False)


def index_block(hyp: Hypothesis, registered_at: str) -> str:
    """The `[[hypothesis]]` table `register` appends for one registration."""
    lines = ["[[hypothesis]]", f'id = "{hyp.id}"', f'registered_at = "{registered_at}"']
    for name, value in hyp.content().items():
        if name != "schema" and value is not None and value != "":
            lines.append(f"{name} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


# --- registering -------------------------------------------------------------


def paths_for(root: Path, hyp_id: str) -> tuple[Path, Path]:
    """(canonical hypothesis file, signed record) for an id under `root`."""
    base = Path(root) / RECORD_DIR / hyp_id[:ID_PREFIX]
    return base.with_name(base.name + ".hypothesis.json"), base.with_name(base.name + ".json")


def _workspace_git(root: Path) -> dict | None:
    """HEAD and dirtiness of the workspace, if it is a git checkout. Evidence
    of what the repository held, not of when."""
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return {"sha": head, "dirty": bool(status.strip())} if head else None


def resolve_supersedes(prefix: str | None, entries: list[IndexEntry]) -> str | None:
    """The full id an `--supersedes` prefix names; raises ValueError unless exactly one."""
    if prefix is None:
        return None
    prefix = prefix.strip().lower()
    if len(prefix) < MIN_ID_PREFIX:
        raise ValueError(f"--supersedes needs at least {MIN_ID_PREFIX} hex digits of an id")
    found = sorted({e.id for e in entries if e.id.startswith(prefix)})
    if len(found) != 1:
        what = "no registered hypothesis" if not found else f"{len(found)} hypotheses"
        raise ValueError(f"--supersedes {prefix}: {what} in {INDEX_NAME} match")
    return found[0]


@dataclass
class Registered:
    """What `register` wrote."""

    id: str
    registered_at: str
    keyid: str
    record: Path
    canonical: Path
    index: Path


def register(hyp: Hypothesis, root: Path, signer, *, registered_at: str | None = None):
    """Write the canonical file, the signed record and the index entry.

    New files only: the same content registered twice is refused, since its
    id -- and so its files -- would be the same. The signer is passed in
    already resolved so a key problem refuses before anything is written."""
    from attestation.run_record.build import now

    root = Path(root).expanduser().resolve()
    index = root / INDEX_NAME
    canonical_path, record_path = paths_for(root, hyp.id)
    taken = {e.id for e in parse_file(index)}
    if hyp.id in taken or canonical_path.exists() or record_path.exists():
        raise FileExistsError(f"already registered as {hyp.id[:ID_PREFIX]}")
    stamp = registered_at or now()
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    with canonical_path.open("xb") as fh:
        fh.write(hyp.canonical())
    try:
        _sign_and_write(hyp, root, signer, stamp, canonical_path, record_path)
    except BaseException:
        # A half-written registration (canonical file, no record) would refuse
        # every retry as "already registered" while list/check show nothing.
        record_path.unlink(missing_ok=True)
        canonical_path.unlink(missing_ok=True)
        raise
    prefix = "\n" if index.is_file() and index.read_text(encoding="utf-8").strip() else ""
    with index.open("a", encoding="utf-8") as fh:
        fh.write(prefix + index_block(hyp, stamp))
    return Registered(hyp.id, stamp, signer.keyid, record_path, canonical_path, index)


def _sign_and_write(hyp, root: Path, signer, stamp: str, canonical_path: Path, record_path: Path):
    """The signed run record whose subject is the canonical file."""
    from attestation.run_record.build import CheckResult, Finding, RunRecorder, write_record

    rec = RunRecorder(TOOL_NAME, root=root, started_at=stamp)
    rec.finished_at = stamp
    rec.add_subject(canonical_path)
    data = {
        "hypothesis": hyp.content(),
        "registered_at": stamp,
        "workspace_git": _workspace_git(root),
    }
    rec.add_check(CheckResult(CHECK_NAME, [Finding(hyp.id, "registered", hyp.describe(), data)]))
    write_record(rec.sign(signer), record_path)


# --- reading registrations back ---------------------------------------------


@dataclass
class Registration:
    """One registration as found on disk, with what does not match.

    `hypothesis` is the SIGNED text when the record verifies -- an edited
    copy in the index cannot change the verdict reported beside it -- and the
    index's text otherwise (flagged altered)."""

    id: str
    hypothesis: Hypothesis | None
    registered_at: str | None = None
    keyid: str | None = None
    workspace_git: dict | None = None
    flags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    superseded_by: list[str] = field(default_factory=list)

    @property
    def short(self) -> str:
        """The id's first ID_PREFIX hex digits."""
        return self.id[:ID_PREFIX]

    def flag(self, name: str, reason: str) -> None:
        """Add `name` (once) with a reason a reader can act on."""
        if name not in self.flags:
            self.flags.append(name)
        self.reasons.append(reason)


@dataclass
class _Signed:
    ok: bool
    hypothesis: Hypothesis | None = None
    registered_at: str | None = None
    keyid: str | None = None
    workspace_git: dict | None = None
    problems: list[str] = field(default_factory=list)


def _signed_finding(statement: dict) -> dict:
    for check in (statement.get("predicate") or {}).get("checks") or []:
        if check.get("check") == CHECK_NAME and check.get("results"):
            return check["results"][0]
    return {}


def _read_record(path: Path, root: Path, expected_id: str | None) -> _Signed:
    """Verify one record and pull the signed hypothesis out of it."""
    from attestation.run_record import verify

    try:
        report = verify.verify_file(path, root=root)
    except (OSError, ValueError) as exc:
        return _Signed(False, problems=[f"record {path.name} unreadable: {exc}"])
    problems = [f"record {i.what}: {i.detail}" for i in report if i.status == verify.FAILED]
    return _signed_from(report.statement or {}, problems, expected_id)


def _signed_from(statement: dict, problems: list[str], expected_id: str | None) -> _Signed:
    """The hypothesis a verified Statement carries, and whether it is the one
    filed under `expected_id`."""
    finding = _signed_finding(statement)
    data = finding.get("data") or {}
    try:
        hyp = Hypothesis.from_content(data.get("hypothesis") or {})
    except ValueError as exc:
        return _Signed(False, problems=[*problems, f"record holds no hypothesis: {exc}"])
    if finding.get("subject") != hyp.id or (expected_id and hyp.id != expected_id):
        problems.append("the signed hypothesis does not hash to the id it is filed under")
    signer = (statement.get("predicate") or {}).get("signer") or {}
    return _Signed(
        ok=not problems,
        hypothesis=hyp,
        registered_at=data.get("registered_at"),
        keyid=signer.get("keyid"),
        workspace_git=data.get("workspace_git"),
        problems=problems,
    )


def _record_paths(root: Path) -> list[Path]:
    folder = Path(root) / RECORD_DIR
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*.json") if not p.name.endswith(".hypothesis.json"))


def _from_index(entry: IndexEntry, root: Path) -> Registration:
    reg = Registration(entry.id, entry.hypothesis, entry.registered_at)
    if entry.problem or entry.hypothesis is None:
        reg.flag(ALTERED, f"{INDEX_NAME} entry {entry.id} is malformed: {entry.problem}")
        return reg
    if entry.hypothesis.id != entry.id:
        reg.flag(ALTERED, f"the text in {INDEX_NAME} no longer hashes to its id (edited)")
    record_path = paths_for(root, entry.id)[1]
    if not record_path.is_file():
        reg.flag(ALTERED, f"no signed record at {RECORD_DIR}/{record_path.name}")
        return reg
    _merge_signed(reg, _read_record(record_path, root, entry.id))
    if reg.registered_at != entry.registered_at:
        reg.flag(ALTERED, f"registered_at in {INDEX_NAME} differs from the signed time")
    return reg


def _merge_signed(reg: Registration, signed: _Signed) -> None:
    for problem in signed.problems:
        reg.flag(ALTERED, problem)
    if signed.hypothesis is not None and signed.ok:
        reg.hypothesis = signed.hypothesis
    reg.registered_at = signed.registered_at
    reg.keyid = signed.keyid
    reg.workspace_git = signed.workspace_git


def _orphan(path: Path, root: Path) -> Registration:
    signed = _read_record(path, root, None)
    if signed.hypothesis is None:
        reg = Registration(path.stem, None)
        reg.flag(ALTERED, f"{RECORD_DIR}/{path.name}: " + "; ".join(signed.problems))
        return reg
    reg = Registration(signed.hypothesis.id, None)
    _merge_signed(reg, signed)
    reg.hypothesis = signed.hypothesis
    reg.flag(ALTERED, f"a signed registration has no entry in {INDEX_NAME} (removed?)")
    return reg


def audit(root: Path) -> list[Registration]:
    """Every registration under `root`: the index's entries, plus any signed
    record the index no longer lists. Raises ValueError on an unreadable index."""
    root = Path(root).expanduser().resolve()
    entries = parse_file(root / INDEX_NAME)
    regs = [_from_index(e, root) for e in entries]
    seen = {e.id[:ID_PREFIX] for e in entries}
    for path in _record_paths(root):
        if path.stem not in seen:
            regs.append(_orphan(path, root))
    _mark_superseded(regs)
    return regs


def _mark_superseded(regs: list[Registration]) -> None:
    by_id = {r.id: r for r in regs}
    for reg in regs:
        target = reg.hypothesis.supersedes if reg.hypothesis else None
        if target in by_id:
            by_id[target].superseded_by.append(reg.id)
            if SUPERSEDED not in by_id[target].flags:
                by_id[target].flags.append(SUPERSEDED)


# --- one hypothesis, judged --------------------------------------------------


def judge(reg: Registration, compare_result: dict | None, why_not: str | None = None) -> dict:
    """A registration plus its verdict as one JSON-ready dict. `compare_result`
    None means the comparison could not run (`why_not` says why): untested."""
    out: dict = {
        "id": reg.short,
        "expect": reg.hypothesis.describe() if reg.hypothesis else None,
        "registered_at": reg.registered_at,
        "keyid": reg.keyid,
    }
    flags = list(reg.flags)
    reasons = list(reg.reasons)
    if reg.hypothesis is None or compare_result is None:
        verdict = HypothesisVerdict(VerdictKind.UNTESTED, reasons=[why_not or "nothing to judge"])
    else:
        times = evidence_times(compare_result, reg.hypothesis.arms)
        verdict = evaluate(reg.hypothesis, compare_result, times, registered_at=reg.registered_at)
        if verdict.post_hoc:
            flags.insert(0, POST_HOC)
            earliest = ", ".join(f"{a} {t}" for a, t in sorted(verdict.predates.items()))
            reasons.append(f"registered after the evidence for: {earliest}")
    return out | {
        "verdict": str(verdict.verdict),
        "flags": flags,
        "reasons": [*verdict.reasons, *reasons],
        "pairs": verdict.pairs,
    }


def fails(judged: dict) -> bool:
    """True for what `attest prereg check` exits 1 on: refuted, or altered."""
    return judged["verdict"] == VerdictKind.REFUTED or ALTERED in judged["flags"]


def attach(compare_result: dict, root: Path | None) -> dict:
    """The `runs.compare` block for this family's registrations under `root`:
    {} when there are none, so a family without any keeps today's keys.

    Registrations on another metric are listed `untested` with the metric to
    compare on rather than re-read here: this judges the evidence in hand."""
    family = compare_result.get("family")
    if root is None or not family:
        return {}
    try:
        regs = [r for r in audit(Path(root)) if r.hypothesis and r.hypothesis.family == family]
    except (OSError, ValueError) as exc:
        return {"hypotheses": [], "n_hypotheses": 0, "prereg_error": str(exc)}
    if not regs:
        return {}
    judged = [_brief(_judge_in_hand(reg, compare_result)) for reg in regs]
    return {
        "hypotheses": judged[:MAX_HYPOTHESES_SHOWN],
        "n_hypotheses": len(judged),
        "prereg_limits": list(LIMITS),
    }


def _judge_in_hand(reg: Registration, compare_result: dict) -> dict:
    """Judge on `compare_result` only when it is on the registration's metric."""
    metric = reg.hypothesis.metric if reg.hypothesis else None
    if metric is not None and metric != compare_result.get("metric"):
        return judge(reg, None, f"registered on {metric}; compare with that metric")
    return judge(reg, compare_result)


def _brief(judged: dict) -> dict:
    """A verdict trimmed for a tool reply: no per-pair arm text, two reasons."""
    keep = ("id", "expect", "verdict", "flags", "registered_at")
    return {k: judged[k] for k in keep} | {"reasons": judged["reasons"][:2]}


def workspace(explicit: str | None) -> Path:
    """Where registrations live: `explicit`, else RESEARCH_ROOT, else cwd."""
    from attestation import ledger

    return ledger.workspace_root(explicit) or Path(os.getcwd())


def refuse_undeclared(metric: str) -> None:
    """Raise ValueError -- with the sentence runs.compare uses -- unless
    `metric` has a declared direction (stems matched as compare matches them)."""
    from attestation import ledger

    # ledger's own stem matcher, not a copy: a hypothesis must be refused
    # exactly when compare would refuse to rank its metric.
    if ledger._metric_direction(metric, ledger.metric_directions()) is None:
        raise ValueError(ledger.unknown_direction_message(metric))

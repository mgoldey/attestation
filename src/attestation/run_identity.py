"""What a core record leaves unverified about the code and environment that ran.

Pure and schema-driven: it reads only `attestation.run/1`'s own `code` and
`dependencies` fields, so it says the same thing about every producer that
writes a core record and names none of them. The ledger stores the findings
beside a run's caveats (`runs.caveats_json`, key `provenance`), and surfaces
them as CAVEATS -- in `runs.detail`, `runs.compare` and the claim checker --
never as a verdict: a number from a dirty checkout is still the number the run
recorded; what is not established is which code produced it.

Spec: docs/superpowers/specs/2026-11-20-smeltery-adapter-design.md.
"""

from __future__ import annotations

from collections import defaultdict

# The provenance levels below `verified`, in the order findings are listed.
_UNVERIFIED = ("inferred", "unknown")
# How many run names a grouped caveat lists before `and N more`.
MAX_NAMED_RUNS = 5


def _code_finding(code) -> str | None:
    """`code inferred`, `code unknown, dirty`, ... or None for verified and
    not known dirty. `dirty: null` is "not observed", not a finding: the
    provenance level already says how much is known."""
    words = []
    if code.provenance in _UNVERIFIED:
        words.append(code.provenance)
    if code.dirty is True:
        words.append("dirty")
    return f"code {', '.join(words)}" if words else None


def _dependency_findings(dependencies) -> list[str]:
    """Findings for the environment: none recorded, or the entries whose
    identity is not verified or whose build was dirty, names sorted."""
    if dependencies is None:
        return ["dependencies not recorded"]
    by_level: dict[str, list[str]] = defaultdict(list)
    for dep in dependencies:
        if dep.provenance in _UNVERIFIED:
            by_level[dep.provenance].append(dep.name)
        if dep.dirty is True:
            by_level["dirty"].append(dep.name)
    return [
        f"dependencies {level}: {', '.join(sorted(by_level[level]))}"
        for level in (*_UNVERIFIED, "dirty")
        if by_level[level]
    ]


def provenance_findings(record) -> list[str]:
    """Short phrases naming what `record` does not establish about the code
    and environment that produced it; `[]` when every identity it carries is
    verified and none is known dirty."""
    code = _code_finding(record.code)
    return ([code] if code else []) + _dependency_findings(record.dependencies)


def provenance_caveats(runs: list[tuple[str, list[str]]]) -> list[str]:
    """One caveat per distinct set of findings, naming the runs that share it.

    `runs` is `(name, findings)` in the order a caller shows them; runs with
    no findings are skipped. Grouped so a pooled arm of many seed replicates
    with one shared gap contributes one line, not one per replicate."""
    groups: dict[tuple[str, ...], list[str]] = {}
    for name, findings in runs:
        if findings:
            groups.setdefault(tuple(findings), []).append(name)
    out = []
    for findings, names in groups.items():
        shown = ", ".join(names[:MAX_NAMED_RUNS])
        if len(names) > MAX_NAMED_RUNS:
            shown += f" and {len(names) - MAX_NAMED_RUNS} more"
        out.append(
            f"provenance not verified for {shown}: {'; '.join(findings)} -- the values"
            " stand, but which code and environment produced them is not established"
            " (runs.detail lists each run's)"
        )
    return out

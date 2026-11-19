"""No code outside the adapters package names a specific tracker (roadmap 11).

The ledger and the claim checker read only core run records; everything that
knows a tracker's layout lives in `src/attestation/ledger_adapters/`. A
tracker name anywhere else in shipped code is the leak this test exists to
catch -- `ledger.ADAPTER_CAVEATS` keyed by tracker, a substring match on one
tracker's note, a comment listing trackers in `db.py` were all such leaks.
Spec: docs/superpowers/specs/2026-11-19-ledger-reads-core-records-design.md.

The detector is a function over text, and `test_detector_can_fail` proves it
fires (a guard that cannot fail protects nothing --
docs/measurement-lessons.md).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNED = (ROOT / "src" / "attestation", ROOT / "packages" / "attestation-schema" / "src")
ADAPTERS_PKG = ROOT / "src" / "attestation" / "ledger_adapters"

TRACKERS = (
    "wandb",
    "w&b",
    "mlflow",
    "mlruns",
    "sacred",
    "dvc",
    "hydra",
    "multirun",
    "tensorboard",
    "tensorflow",
)
# `_` counts as a boundary (so `sacred_runs` and `_wandb_runs` are caught) and
# letters/digits do not (so `hydrate` and `dvcs` are not).
_PATTERN = re.compile(
    r"(?<![a-z0-9])(" + "|".join(re.escape(t) for t in TRACKERS) + r")(?![a-z0-9])",
    re.IGNORECASE,
)

# Documentation only. The bundled skills teach an agent which layouts the
# ledger reads; they are prose shipped with the package, not code.
ALLOWLIST = ("src/attestation/skills/*/SKILL.md",)

_TEXT_SUFFIXES = {".py", ".toml", ".json", ".md", ".txt", ".html", ".j2", ".yaml", ".yml", ".sh"}


def tracker_mentions(text: str) -> list[tuple[int, str]]:
    """(line, tracker) for every tracker name in `text`."""
    return [
        (n, m.group(1).lower())
        for n, line in enumerate(text.splitlines(), 1)
        for m in _PATTERN.finditer(line)
    ]


def _files() -> list[Path]:
    out = []
    for base in SCANNED:
        for path in sorted(base.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix not in _TEXT_SUFFIXES or ADAPTERS_PKG in path.parents:
                continue
            rel = path.relative_to(ROOT)
            if any(rel.match(pattern) for pattern in ALLOWLIST):
                continue
            out.append(path)
    return out


def test_scan_is_not_vacuous():
    files = {p.relative_to(ROOT).as_posix() for p in _files()}
    assert {"src/attestation/ledger.py", "src/attestation/db.py", "src/attestation/claims.py"} <= (
        files
    )
    assert "packages/attestation-schema/src/attestation_schema/record.py" in files
    assert not any(f.startswith("src/attestation/ledger_adapters/") for f in files)


def test_no_code_outside_the_adapters_names_a_tracker():
    offenders = [
        f"{p.relative_to(ROOT)}:{line} names {name}"
        for p in _files()
        for line, name in tracker_mentions(p.read_text(errors="replace"))
    ]
    assert not offenders, (
        "tracker knowledge belongs in src/attestation/ledger_adapters/ and travels "
        "to the ledger as data in the core record:\n  " + "\n  ".join(offenders)
    )


def test_the_adapters_package_does_name_them():
    """The other half of the boundary: the knowledge moved, it did not vanish."""
    names = {
        name for path in ADAPTERS_PKG.glob("*.py") for _, name in tracker_mentions(path.read_text())
    }
    assert {"wandb", "mlflow", "sacred", "dvc", "hydra"} <= names


def test_detector_can_fail():
    hit = tracker_mentions
    assert hit("import mlflow\n") == [(1, "mlflow")]
    assert hit("x = 1\nreads wandb-summary.json\n") == [(2, "wandb")]
    assert hit("def _sacred_runs(root):") == [(1, "sacred")]
    assert hit('"read by the DVC adapter"') == [(1, "dvc")]
    assert hit("a W&B run") == [(1, "w&b")]
    assert hit("root / 'mlruns'") == [(1, "mlruns")]
    assert hit("Hydra --multirun sweep") == [(1, "hydra"), (1, "multirun")]
    # Look-alikes are not trackers.
    assert hit("hydrate the cache; dvcs; mlflowy; sacredness") == []


def test_allowlist_covers_only_documentation():
    for pattern in ALLOWLIST:
        assert pattern.endswith(".md"), pattern
    allowed = [
        p
        for base in SCANNED
        for p in base.rglob("*")
        if p.is_file() and any(p.relative_to(ROOT).match(a) for a in ALLOWLIST)
    ]
    assert allowed, "an allowlist entry that matches nothing is stale"

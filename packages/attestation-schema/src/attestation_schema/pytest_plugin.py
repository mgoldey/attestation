"""The conformance suite as a pytest plugin, for a producer's own CI.

Opt-in, never auto-loaded (no `pytest11` entry point), so a test suite that
merely has the package installed is unchanged::

    pip install "attestation-schema[conformance]"
    pytest -p attestation_schema.pytest_plugin --attestation-runs 'runs/*'

or in the producer's pytest configuration::

    [tool.pytest.ini_options]
    addopts = "-p attestation_schema.pytest_plugin"
    attestation_runs = ["runs"]

Each run directory becomes one test item (marker `attestation_conformance`);
a pattern that matches no run directory is a failing item, never zero items.
The `attestation_conformance` fixture checks a run directory a test made
itself, optionally against the `JobSpec` it was submitted with.

The only module in the package that imports pytest (the `[conformance]`
extra). Spec: docs/superpowers/specs/2026-11-19-conformance-suite-design.md.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from attestation_schema.conformance import Report, check_run_dir, find_targets
from attestation_schema.jobs import JobSpec

MARKER = "attestation_conformance"
COLLECTOR = "attestation-conformance"


def pytest_addoption(parser: pytest.Parser) -> None:
    """`--attestation-runs` (repeatable) and the ini key `attestation_runs`."""
    group = parser.getgroup("attestation", "attestation run-record conformance")
    group.addoption(
        "--attestation-runs",
        action="append",
        default=[],
        metavar="PATTERN",
        help="a run directory, a directory of runs, or a glob; one conformance test per run"
        " (repeatable; replaces the attestation_runs ini list)",
    )
    parser.addini(
        "attestation_runs",
        "run directories / globs checked for conformance (relative to the ini file), used"
        " when pytest is run without explicit paths",
        type="linelist",
        default=[],
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the marker the generated items carry."""
    config.addinivalue_line(
        "markers", f"{MARKER}: an attestation run-record conformance check (generated)"
    )


def _explicit_paths(config: pytest.Config) -> bool:
    source = getattr(config, "args_source", None)
    return source is not None and source == pytest.Config.ArgsSource.ARGS


def _patterns(config: pytest.Config) -> tuple[list[str], Path]:
    cli = config.getoption("attestation_runs") or []
    if cli:
        return list(cli), Path(config.invocation_params.dir)
    if _explicit_paths(config):
        return [], Path(".")
    base = Path(config.inipath).parent if config.inipath else Path(config.rootpath)
    return list(config.getini("attestation_runs")), base


class ConformanceFailure(Exception):
    """A run directory broke a rule; carries the full report."""

    def __init__(self, text: str):
        super().__init__(text)
        self.text = text


class RunDirItem(pytest.Item):
    """One run directory, checked against every rule."""

    def __init__(self, *, run_dir: Path | None, pattern: str | None = None, **kw: Any):
        super().__init__(**kw)
        self.run_dir = run_dir
        self.pattern = pattern
        self.add_marker(MARKER)

    def runtest(self) -> None:
        """Fail with every violation, or with the pattern that matched nothing."""
        if self.run_dir is None:
            raise ConformanceFailure(
                f"no run directory matched {self.pattern!r}: a pattern that matches nothing"
                " checks nothing (fix the path, or drop it)"
            )
        report = check_run_dir(self.run_dir)
        if not report.ok:
            raise ConformanceFailure(report.format())

    def repr_failure(self, excinfo: pytest.ExceptionInfo[BaseException], style=None) -> str:  # type: ignore[override]
        """The report itself, no traceback: the paths say where to look."""
        if isinstance(excinfo.value, ConformanceFailure):
            return excinfo.value.text
        return str(super().repr_failure(excinfo))

    def reportinfo(self) -> tuple[Path | str, int | None, str]:
        """Where the item points: the run directory (or the empty pattern)."""
        where = self.run_dir if self.run_dir is not None else f"<{self.pattern}>"
        return where, None, f"attestation conformance: {self.name}"


class RunsCollector(pytest.Collector):
    """The synthetic parent of every generated item."""

    def __init__(self, *, targets: list[Path], empty: list[str], **kw: Any):
        super().__init__(**kw)
        self.targets = targets
        self.empty = empty

    def collect(self) -> list[pytest.Item]:
        """One item per run directory, one failing item per empty pattern."""
        root = Path(self.config.rootpath)
        items: list[pytest.Item] = []
        for d in self.targets:
            try:
                name = d.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                name = d.resolve().as_posix()
            items.append(RunDirItem.from_parent(self, name=name, run_dir=d))
        for pattern in self.empty:
            items.append(
                RunDirItem.from_parent(self, name=f"<{pattern}>", run_dir=None, pattern=pattern)
            )
        return items


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Append the generated items before `-k`/`-m` deselect anything."""
    patterns, base = _patterns(config)
    if not patterns:
        return
    targets, empty = find_targets(patterns, base)
    # nodeid given explicitly: the session's is "", which would prefix "::".
    collector = RunsCollector.from_parent(
        session, name=COLLECTOR, nodeid=COLLECTOR, targets=targets, empty=empty
    )
    items.extend(collector.collect())


class Checker:
    """What the `attestation_conformance` fixture returns."""

    def __call__(
        self,
        run_dir: str | os.PathLike[str],
        spec: JobSpec | None = None,
        *,
        verify_outputs: bool = True,
    ) -> Report:
        """Fail the test with every violation; return the report when there are none."""
        report = check_run_dir(run_dir, spec=spec, verify_outputs=verify_outputs)
        if not report.ok:
            pytest.fail(report.format(), pytrace=False)
        return report

    def report(
        self,
        run_dir: str | os.PathLike[str],
        spec: JobSpec | None = None,
        *,
        verify_outputs: bool = True,
    ) -> Report:
        """The report, without failing (for asserting on a broken directory)."""
        return check_run_dir(run_dir, spec=spec, verify_outputs=verify_outputs)


@pytest.fixture
def attestation_conformance() -> Checker:
    """Check a run directory the test produced: `attestation_conformance(run_dir, spec)`."""
    return Checker()

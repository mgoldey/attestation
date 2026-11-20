"""Who may import whom across the three repos (roadmap 43).

attestation is the observer: it owns the run-record schema and knows no
chemistry, so nothing under src/attestation or the schema package imports
smeltery or ferric. The schema package is what producers install to WRITE a
record, so it imports nothing from attestation and nothing outside the
standard library. Spec: docs/superpowers/specs/2026-11-19-core-run-record-design.md.

Each detector is a function over source text, so the self-tests below prove
it fires on a violation (the pattern from smeltery's test_migration_rules.py;
a guard that cannot fail protects nothing -- docs/measurement-lessons.md, 4).
"""

from __future__ import annotations

import ast
import pathlib
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
ATTESTATION_SRC = ROOT / "src" / "attestation"
SCHEMA_PKG = ROOT / "packages" / "attestation-schema"
SCHEMA_SRC = SCHEMA_PKG / "src" / "attestation_schema"

APPLICATION_REPOS = ("smeltery", "ferric")
_DYNAMIC_IMPORTERS = {"import_module", "__import__"}


def _modules(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def imported_modules(src: str) -> list[tuple[int, str]]:
    """Every absolute import in `src`, anywhere (function bodies included):
    `import x`, `from x import y`, and `importlib.import_module("x")` /
    `__import__("x")` with a literal name -- smeltery reaches ferric through
    import_module, so a static-only scan would miss the real idiom."""
    out: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            out += [(node.lineno, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node.lineno, node.module))
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            arg = node.args[0] if node.args else None
            if (
                name in _DYNAMIC_IMPORTERS
                and isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
            ):
                out.append((node.lineno, arg.value))
    return out


def forbidden_imports(src: str, roots: tuple[str, ...]) -> list[tuple[int, str]]:
    """Imports in `src` whose top-level package is one of `roots`."""
    return [(ln, m) for ln, m in imported_modules(src) if m.split(".")[0] in roots]


def non_stdlib_imports(src: str, allowed: tuple[str, ...]) -> list[tuple[int, str]]:
    """Imports in `src` that are neither the standard library nor `allowed`."""
    ok = set(sys.stdlib_module_names) | set(allowed)
    return [(ln, m) for ln, m in imported_modules(src) if m.split(".")[0] not in ok]


def _offenders(files: list[pathlib.Path], check) -> list[str]:
    return [
        f"{f.relative_to(ROOT)}:{ln} imports {m}" for f in files for ln, m in check(f.read_text())
    ]


def test_scans_are_not_vacuous():
    assert len(_modules(ATTESTATION_SRC)) >= 50
    assert {p.name for p in _modules(SCHEMA_SRC)} >= {"record.py", "layout.py", "jobs.py"}


def test_attestation_imports_neither_smeltery_nor_ferric():
    offenders = _offenders(
        _modules(ATTESTATION_SRC) + _modules(SCHEMA_SRC),
        lambda s: forbidden_imports(s, APPLICATION_REPOS),
    )
    assert not offenders, (
        "attestation is the observer and knows no chemistry; it reads what "
        "smeltery/ferric WRITE, never imports them:\n  " + "\n  ".join(offenders)
    )


def test_schema_package_imports_nothing_from_attestation():
    offenders = _offenders(_modules(SCHEMA_SRC), lambda s: forbidden_imports(s, ("attestation",)))
    assert not offenders, (
        "attestation_schema is what producers install; importing attestation would "
        "drag the observer into every producer:\n  " + "\n  ".join(offenders)
    )


# The ONE module allowed past the standard library: the conformance suite's
# pytest plugin, behind the `[conformance]` extra and loaded only with `-p`
# (docs/superpowers/specs/2026-11-19-conformance-suite-design.md). Everything
# else -- the checker it wraps included -- stays stdlib, so a producer that
# writes records installs nothing.
PYTEST_PLUGIN = SCHEMA_SRC / "pytest_plugin.py"
# The training-loop callbacks, each behind its own extra and imported only by
# whoever uses it (docs/superpowers/specs/2026-11-19-training-callbacks-design.md):
# each may import its framework and nothing else outside the stdlib. Their
# shared core (`integrations/_core.py`) stays stdlib. The MLflow run-context
# provider follows the same rule: `integrations/mlflow.py` is the ONE module
# importing mlflow, and the mapping it and the readers share
# (`integrations/mlflow_mapping.py`) is stdlib, so attestation imports it with
# no mlflow (docs/superpowers/specs/2026-11-20-mlflow-plugin-design.md).
FRAMEWORK_MODULES = {
    SCHEMA_SRC / "integrations" / "huggingface.py": ("transformers",),
    SCHEMA_SRC / "integrations" / "lightning.py": ("lightning",),
    SCHEMA_SRC / "integrations" / "mlflow.py": ("mlflow",),
}
EXTRAS = {
    "conformance": ["pytest>=7.2"],
    "hf": ["transformers>=4.41", "torch>=2.1", "accelerate>=0.26"],
    "lightning": ["lightning>=2.1"],
    "mlflow": ["mlflow-skinny>=2.20"],
}
ENTRY_POINTS = {
    "mlflow.run_context_provider": {
        "attestation": "attestation_schema.integrations.mlflow:AttestationRunContext"
    }
}


def test_schema_package_is_standard_library_only():
    optional = {PYTEST_PLUGIN, *FRAMEWORK_MODULES}
    stdlib_only = [p for p in _modules(SCHEMA_SRC) if p not in optional]
    assert SCHEMA_SRC / "conformance.py" in stdlib_only
    assert SCHEMA_SRC / "integrations" / "_core.py" in stdlib_only
    assert SCHEMA_SRC / "integrations" / "mlflow_mapping.py" in stdlib_only
    offenders = _offenders(stdlib_only, lambda s: non_stdlib_imports(s, ("attestation_schema",)))
    assert not offenders, "attestation_schema is stdlib-only:\n  " + "\n  ".join(offenders)
    plugin = _offenders(
        [PYTEST_PLUGIN], lambda s: non_stdlib_imports(s, ("attestation_schema", "pytest"))
    )
    assert not plugin, "the plugin may import pytest and nothing else:\n  " + "\n  ".join(plugin)
    for module, framework in FRAMEWORK_MODULES.items():
        assert module.is_file(), module
        extra = _offenders(
            [module], lambda s, f=framework: non_stdlib_imports(s, ("attestation_schema", *f))
        )
        assert not extra, f"{module.name} may import {framework} only:\n  " + "\n  ".join(extra)
    project = tomllib.loads((SCHEMA_PKG / "pyproject.toml").read_text())["project"]
    assert project["dependencies"] == [], "attestation-schema declares no dependencies"
    assert project.get("optional-dependencies") == EXTRAS, (
        "the extras are the conformance plugin's pytest and the integrations' frameworks"
    )
    assert project.get("entry-points") == ENTRY_POINTS, (
        "the only entry point is MLflow's run-context provider"
    )


def test_only_the_provider_module_imports_mlflow():
    """attestation reads MLflow stores as files and never imports mlflow
    (it is not a runtime dependency); in the schema package only the
    provider does."""
    importers = [
        f"{p.relative_to(ROOT)}:{ln}"
        for p in _modules(ATTESTATION_SRC) + _modules(SCHEMA_SRC)
        for ln, _ in forbidden_imports(p.read_text(), ("mlflow",))
    ]
    provider = SCHEMA_SRC / "integrations" / "mlflow.py"
    assert importers and all(i.startswith(str(provider.relative_to(ROOT))) for i in importers), (
        importers
    )


def test_boundary_detectors_can_fail():
    hit = forbidden_imports
    assert hit("import ferric\n", APPLICATION_REPOS) == [(1, "ferric")]
    assert hit("from smeltery.record import RunRecord\n", APPLICATION_REPOS) == [
        (1, "smeltery.record")
    ]
    assert hit("import os, smeltery.funnel as f\n", APPLICATION_REPOS) == [(1, "smeltery.funnel")]
    assert hit("def f():\n    import ferric\n", APPLICATION_REPOS) == [(2, "ferric")]
    assert hit('importlib.import_module("ferric")\n', APPLICATION_REPOS) == [(1, "ferric")]
    assert hit('import_module("smeltery.x")\n', APPLICATION_REPOS) == [(1, "smeltery.x")]
    assert hit('__import__("ferric")\n', APPLICATION_REPOS) == [(1, "ferric")]
    # Not imports, or not these packages: prose, look-alike names, relative imports.
    assert hit('"""ferric.__build__ is read by smeltery"""\n# import ferric\n', ("ferric",)) == []
    assert hit("import ferricyanide\nfrom smelteryx import y\n", APPLICATION_REPOS) == []
    assert hit("from . import record\nfrom .ferric import x\n", APPLICATION_REPOS) == []
    # The schema package's own name is not attestation's.
    assert hit("from attestation_schema import x\n", ("attestation",)) == []
    assert hit("from attestation.ledger import x\n", ("attestation",)) == [
        (1, "attestation.ledger")
    ]
    assert hit("import attestation\n", ("attestation",)) == [(1, "attestation")]


def test_stdlib_detector_can_fail():
    assert non_stdlib_imports("import json, hashlib\nfrom pathlib import Path\n", ()) == []
    assert non_stdlib_imports("import jsonschema\n", ()) == [(1, "jsonschema")]
    assert non_stdlib_imports("from pydantic import BaseModel\n", ()) == [(1, "pydantic")]
    assert non_stdlib_imports("from attestation_schema.record import X\n", ()) != []
    assert (
        non_stdlib_imports("from attestation_schema.record import X\n", ("attestation_schema",))
        == []
    )

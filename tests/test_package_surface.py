"""The package's own surface: version, docstring, `__all__`, and py.typed.

`docs/superpowers/specs/2026-08-29-package-docs-design.md`'s "The package
surface" is what this pins: `pyproject.toml` stays the one source for the
version, nothing is re-exported (the modules ARE the API, per
`2026-08-21-onion-refactor-design.md`), and a downstream type checker finds
`py.typed` in the wheel.
"""

from importlib.metadata import version
from pathlib import Path

import attestation


def test_version_matches_pyproject():
    """`__version__` is read from package metadata, not hand-duplicated --
    so `pyproject.toml`'s version is the only place that number is typed."""
    assert attestation.__version__ == version("attestation")


def test_all_lists_exactly_the_spec_modules_and_each_imports():
    """`__all__` names the modules a user is meant to import, per the spec's
    "The package surface" list -- and every one must actually import, since
    a stale name here would be silent until someone tried it."""
    expected = [
        "ledger",
        "claims",
        "citations",
        "rank",
        "ingest",
        "features",
        "simulate",
        "explain",
        "symbolic",
        "kg",
        "emit",
        "install",
        "llm",
        "embed",
        "db",
        "ports",
    ]
    assert attestation.__all__ == expected
    for name in attestation.__all__:
        __import__(f"attestation.{name}")


def test_package_docstring_is_non_empty():
    """The one-paragraph statement of what the package is, not silence."""
    assert attestation.__doc__ and attestation.__doc__.strip()


def test_nothing_is_reexported_from_init():
    """`__init__.py` states the surface; it does not import from it. Every
    name in `__all__` must be resolvable as a submodule import, not an
    attribute this module pulled in itself -- so `attestation.ledger` works
    only once someone does `import attestation.ledger` or
    `from attestation import ledger`, exactly as the spec decided."""
    src = Path(attestation.__file__).read_text()
    for name in attestation.__all__:
        assert f"from attestation.{name} import" not in src
        assert f"from .{name} import" not in src


def test_py_typed_ships_next_to_the_package():
    """The marker a downstream type checker looks for -- see the wheel-smoke
    CI step that asserts the same thing against the built, installed wheel."""
    assert Path(attestation.__file__).with_name("py.typed").exists()

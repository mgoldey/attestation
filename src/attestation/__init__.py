"""Auditable research provenance: experiment runs, verifiable claims, a
reading graph, and symbolic derivations -- fully local.

Nothing is re-exported here: the modules listed in `__all__` are the public
API, imported directly (`from attestation import ledger`, `import
attestation.claims`), as decided in
`docs/superpowers/specs/2026-08-21-onion-refactor-design.md`. This module
states what those are and nothing more.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("attestation")
except PackageNotFoundError:
    # Not installed (e.g. src/ on sys.path with no editable install) --
    # `pyproject.toml` stays the one source, but a bare checkout must not
    # raise just because the package metadata was never generated.
    __version__ = "0+unknown"

__all__ = [
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

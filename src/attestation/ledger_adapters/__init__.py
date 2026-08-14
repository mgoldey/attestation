"""Adapters that turn a project's artifacts into runs.

`generic` is the default and handles any project that follows the conventions
ML/science repos already share (`results/`, `logs/`, `configs/`, ... holding
JSON/JSONL/YAML/TOML). No project is named here and none needs to be: `scan`
walks a workspace root, treats every subdirectory as a project, and applies the
generic adapter.

A named adapter is an escape hatch for a layout the conventions genuinely
cannot express, registered in `NAMED`. Prefer teaching `generic` a new
convention over adding one -- a convention helps every project, a named adapter
helps exactly one.
"""

from attestation.ledger_adapters import generic

# Optional per-project overrides. Empty by design: everything shipped works by
# convention. Populate only when a real layout defeats the generic reader.
NAMED: dict = {}


def adapter_for(project: str):
    return NAMED.get(project, generic)

"""What attestation is, emitted from attestation rather than copied about.

A packager integrating this tool needs a specific, small set of facts: which
console scripts exist, which MCP server to declare, which skills get installed,
which environment variables move things, where the database resolves, and what
still works when there is no model. agentmarkit needed exactly those and wrote
them by hand into its own repo -- a runtime map in prose plus a preset -- where
they promptly went stale: the pin sat 81 commits behind, and `HERMES_HOME` was
absent because it postdated the transcription.

Everything here is DERIVED. The version comes from installed metadata, the
console scripts from the entry points, the skills from the shipped directory,
the surfaces from `AGENT_SURFACES`. Nothing is a literal that a human has to
remember to update, which is the entire point: `emit.py`'s docstring already
says it -- "two copies of one fact with no check between them is the whole
problem".

**No credentials, by construction.** agentmarkit's preset schema bans `url`,
`token`, `api_key` and `secret` from an MCP declaration, on the rule that a
package declares that a server should EXIST while only the instance holds its
address and its credential. That rule is right independently of who is reading,
so the manifest states that `attest-mcp` exists and how to verify it, never
where it lives or what key it needs.
"""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path

SCHEMA = "attestation/manifest/1"
DISTRIBUTION = "attestation"

# Capabilities that need an OpenAI-compatible endpoint, and those that do not.
# The split is a real property of the code -- the ledger, claim checking and
# SymPy never call a model -- and it is what lets a starter be sold honestly to
# someone who has not connected one yet. Prose in a consumer's doc is where
# this lived before.
STILL_WORKS_WITHOUT_MODEL = (
    "run ledger discovery and comparison",
    "deterministic claim checking",
    "citation key linting",
    "symbolic algebra and calculus",
)
NEEDS_MODEL = (
    "ingest embeddings",
    "tagging",
    "ranking and recommendations",
    "explanations",
    "semantic search",
)

# Environment variables a packager actually has to set or know about. Kept to
# the ones that change WHERE things go or whether the network is touched --
# `.env.sample` remains the full list, and a test keeps this a subset of it.
ENV = {
    "HERMES_HOME": {
        "default": "~/.hermes",
        "purpose": "agent home: skills, caches, refresh script, ledger config fallback",
    },
    "ATTEST_DB": {
        "default": "(see db_path_precedence)",
        "purpose": "the SQLite database path",
        "legacy_alias": "RSS_DB",
    },
    "ATTEST_TOOLS": {
        "default": "(unset: all tools)",
        "purpose": "restrict the MCP server to one agent surface",
    },
    "LLM_BASE_URL": {
        "default": "http://localhost:11434/v1",
        "purpose": "any OpenAI-compatible endpoint; not necessarily Ollama",
    },
    "ATTEST_RESEARCH_WEB": {
        "default": "1",
        "purpose": "arXiv/PubMed/CrossRef search clients; 0 disables every network reader",
    },
}

# Read in the order `db.resolve_db_path` tries them. Stated rather than
# computed because it is an ORDER, not a value -- and pinned by a test that
# drives the real function rather than reading this tuple.
DB_PATH_PRECEDENCE = (
    "explicit --db argument",
    "ATTEST_DB, or RSS_DB (its pre-rename name)",
    "<HERMES_HOME>/skills/science-recommendations/data/hermes.db, when it exists",
    "./hermes.db",
)


def _version() -> str:
    """The installed version, falling back to pyproject in a source checkout."""
    try:
        return metadata.version(DISTRIBUTION)
    except metadata.PackageNotFoundError:  # pragma: no cover - checkout-only path
        import tomllib

        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        return tomllib.loads(pyproject.read_text())["project"]["version"]


def _requires_python() -> str:
    try:
        value = metadata.metadata(DISTRIBUTION)["Requires-Python"]
        if value:
            return value
    except metadata.PackageNotFoundError:  # pragma: no cover - checkout-only path
        pass
    import tomllib

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    return tomllib.loads(pyproject.read_text())["project"]["requires-python"]


def _console_scripts() -> dict[str, str]:
    """Entry points as packaged: name -> "module:function"."""
    try:
        points = metadata.distribution(DISTRIBUTION).entry_points
        scripts = {e.name: e.value for e in points if e.group == "console_scripts"}
        if scripts:
            return scripts
    except metadata.PackageNotFoundError:  # pragma: no cover - checkout-only path
        pass
    import tomllib

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    return dict(tomllib.loads(pyproject.read_text())["project"]["scripts"])


def _skills() -> list[str]:
    """Every bundled skill directory, from the package as installed."""
    root = Path(__file__).resolve().parent / "skills"
    if not root.is_dir():  # pragma: no cover - defensive
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "SKILL.md").is_file())


def _surfaces() -> dict[str, dict]:
    from attestation.mcp import AGENT_SURFACES

    return {
        name: {"env": f"ATTEST_TOOLS={name}", "summary": surface.summary}
        for name, surface in AGENT_SURFACES.items()
    }


def build() -> dict:
    """The manifest as a plain dict, ready for `json.dumps`."""
    return {
        "schema": SCHEMA,
        "name": DISTRIBUTION,
        "version": _version(),
        "requires_python": _requires_python(),
        "console_scripts": _console_scripts(),
        "install": {
            "from_pypi": "uvx attestation install",
            "health_check": "attest install --check",
        },
        "mcp_servers": [
            {
                "name": "attest-mcp",
                "transport": "stdio",
                "command": "attest-mcp",
                "verify": "attest install --check",
            }
        ],
        "surfaces": _surfaces(),
        "skills": _skills(),
        "env": ENV,
        "db_path_precedence": list(DB_PATH_PRECEDENCE),
        "degrades_without_model": {
            "still_works": list(STILL_WORKS_WITHOUT_MODEL),
            "unavailable": list(NEEDS_MODEL),
        },
    }


def render() -> str:
    """The manifest as indented JSON with a trailing newline."""
    return json.dumps(build(), indent=2, sort_keys=False) + "\n"

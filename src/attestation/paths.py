"""Where the agent's home directory is, and what lives under it.

Eight modules each wrote `Path.home() / ".hermes"` for themselves. That is
fine on a laptop with one agent and fatal anywhere else: agentmarkit
provisions attestation onto an isolated Agent37 VM and recorded exactly this
as the blocker that stopped it -- `attest install` would write into the
operator's real Hermes home, because there was no single thing to point
somewhere else.

`HERMES_HOME` is that thing. It is read at CALL time, never captured into a
module-level constant, because a constant is what made this unfixable from
outside: a provisioning script that exports the variable before running
`attest install` must be obeyed, and `db.SKILL_DATA_DB` (evaluated at import)
could only ever be monkeypatched by a test that already had the process.

The DATABASE is deliberately not re-rooted here. `db.resolve_db_path` has its
own documented precedence -- explicit `--db`, then `ATTEST_DB`/`RSS_DB`, then
the legacy skill-data path, then `./hermes.db` -- which callers and
agentmarkit's own runtime map depend on. `HERMES_HOME` moves where the legacy
skill-data path LOOKS; it does not outrank the two overrides above it.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_HOME_DIRNAME = ".hermes"


def hermes_home() -> Path:
    """The agent home: `HERMES_HOME` if set to something, else `~/.hermes`.

    `~` is expanded, because a preset writes paths like `~/.agent37-pilots/...`
    and an unexpanded tilde silently makes a directory literally named `~` in
    whatever the cwd happens to be. A blank value (`HERMES_HOME=` in a .env)
    means unset, not "the current directory".
    """
    raw = (os.environ.get("HERMES_HOME") or "").strip()
    if not raw:
        return Path.home() / DEFAULT_HOME_DIRNAME
    return Path(raw).expanduser()


def citation_cache() -> Path:
    """Where `citations.WebReader` and the library enrichers cache responses."""
    return hermes_home() / "citation-cache"


def skills_dir() -> Path:
    """`~/.hermes/skills` -- the tree `attest install` syncs the bundled skills into."""
    return hermes_home() / "skills"


def scripts_dir() -> Path:
    """`~/.hermes/scripts` -- where the scheduled-refresh script is written."""
    return hermes_home() / "scripts"

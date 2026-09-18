"""`HERMES_HOME` is the one knob that moves every agent-home path.

agentmarkit provisions attestation onto an isolated Agent37 VM and recorded
this as the blocker that stopped it: "attestation hardcodes
Path.home()/'.hermes' ... and ignores HERMES_HOME, so `attest install` would
write to the real Hermes home. Isolation unsolved." Eight sites resolved that
directory independently, so there was no single thing to point elsewhere.

The DB is deliberately NOT part of this: `resolve_db_path` has its own
documented precedence (explicit --db, then ATTEST_DB/RSS_DB, then the legacy
skill-data path, then ./hermes.db) that callers already rely on, and
agentmarkit's runtime map documents it. `HERMES_HOME` moves where the legacy
skill-data path LOOKS, without displacing either override above it.
"""

import importlib
from pathlib import Path

from attestation import paths


def test_hermes_home_defaults_to_the_real_agent_home(monkeypatch):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    assert paths.hermes_home() == Path.home() / ".hermes"


def test_hermes_home_follows_the_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))
    assert paths.hermes_home() == tmp_path / "isolated"


def test_hermes_home_is_read_at_call_time_not_import_time(tmp_path, monkeypatch):
    """A module-level constant is what made this unfixable from outside.

    `db.SKILL_DATA_DB` was evaluated at import, so tests had to monkeypatch the
    attribute and a provisioning script could not move it at all. Reading the
    environment per call is the property that makes one env var enough.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "first"))
    assert paths.hermes_home() == tmp_path / "first"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "second"))
    assert paths.hermes_home() == tmp_path / "second"


def test_expanduser_and_relative_paths_resolve(tmp_path, monkeypatch):
    """A preset writes `~/.agent37-pilots/...`; an unexpanded ~ makes a
    literal directory named '~' in the cwd, which is silent and wrong."""
    monkeypatch.setenv("HERMES_HOME", "~/some-agent-home")
    assert paths.hermes_home() == Path.home() / "some-agent-home"


def test_an_empty_value_is_not_a_home(monkeypatch):
    """HERMES_HOME= in a .env is 'unset', not 'the current directory'."""
    monkeypatch.setenv("HERMES_HOME", "   ")
    assert paths.hermes_home() == Path.home() / ".hermes"


def test_every_agent_home_path_moves_with_it(tmp_path, monkeypatch):
    """The point of the resolver: one env var relocates all of them.

    Each of these was its own `Path.home() / ".hermes"` literal. If a new one
    is added without going through `hermes_home()`, this test keeps passing --
    so `test_no_module_hardcodes_the_agent_home` below is the real guard.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "vm"))
    home = tmp_path / "vm"

    from attestation import citations, db, install, ledger, library_readers

    importlib.reload(library_readers)

    assert paths.citation_cache() == home / "citation-cache"
    assert library_readers.DEFAULT_CACHE == home / "citation-cache"
    assert citations.WebReader().cache_dir == home / "citation-cache"
    # `_config_ladder` falls back to the agent home when neither its env var
    # nor a workspace file supplies one.
    monkeypatch.delenv("ATTEST_NO_SUCH_VAR", raising=False)
    assert ledger._config_ladder("ATTEST_NO_SUCH_VAR", "x.toml") == home / "x.toml"
    assert install._refresh_script_path() == home / "scripts" / install.REFRESH_SCRIPT_NAME
    # `db` is asserted separately: conftest's hermetic fixture patches
    # SKILL_DATA_DB, and that patch deliberately OUTRANKS HERMES_HOME.
    assert db is not None


def test_no_module_hardcodes_the_agent_home():
    """The guard that survives a new call site.

    Eight literals drifted apart precisely because nothing stopped a ninth.
    `paths.py` is the only file allowed to name the directory.
    """
    src = Path(__file__).resolve().parents[1] / "src" / "attestation"
    offenders = []
    for py in src.rglob("*.py"):
        if py.name == "paths.py":
            continue
        text = py.read_text()
        for n, line in enumerate(text.splitlines(), 1):
            if '".hermes"' in line and "Path.home()" in line:
                offenders.append(f"{py.relative_to(src)}:{n}")
    assert not offenders, "resolve through paths.hermes_home() instead: " + ", ".join(offenders)


def test_skill_data_db_follows_hermes_home_when_nothing_patched(tmp_path, monkeypatch):
    """The provisioning case: export HERMES_HOME, get an isolated skill-data path.

    conftest's hermetic fixture patches `SKILL_DATA_DB` for the whole suite, so
    this restores the real default first and then moves HERMES_HOME under it.
    """
    from attestation import db

    monkeypatch.setattr(
        db,
        "SKILL_DATA_DB",
        Path.home()
        / paths.DEFAULT_HOME_DIRNAME
        / "skills"
        / "science-recommendations"
        / "data"
        / "hermes.db",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "vm"))

    expected = tmp_path / "vm" / "skills" / "science-recommendations" / "data" / "hermes.db"
    assert db.skill_data_db() == expected


def test_a_patched_skill_data_db_still_outranks_hermes_home(tmp_path, monkeypatch):
    """`live-db-in-tests` is why: the hermetic fixture points SKILL_DATA_DB at a
    path that does not exist so the suite can never open the real database. If
    HERMES_HOME overrode that, exporting it in a shell would silently re-arm the
    bug the fixture exists to prevent.
    """
    from attestation import db

    sentinel = tmp_path / "patched-absent.db"
    monkeypatch.setattr(db, "SKILL_DATA_DB", sentinel)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "vm"))

    assert db.skill_data_db() == sentinel


def test_attest_db_still_beats_hermes_home(tmp_path, monkeypatch):
    """ATTEST_DB is the more specific override and keeps its place in
    `resolve_db_path`'s documented precedence; agentmarkit's runtime map
    documents that ladder and its preset depends on it."""
    from attestation import db

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "vm"))
    monkeypatch.setenv("ATTEST_DB", str(tmp_path / "explicit.db"))

    assert db.resolve_db_path(None) == tmp_path / "explicit.db"

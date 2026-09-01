"""step_skill_copy after the split: five bundled skills, profile skill trees,
and the legacy monolith.

test_install.py covers the step's original contract (creates files, second
run is OK, never touches a planted data/, check mode, package-relative
source). This file covers what changed on 2026-08-30: one skill became five,
`~/.hermes/profiles/*/skills/` is synced alongside `~/.hermes/skills/`, a
profile's own disable marker is respected, and the superseded
`research-provenance` skill is disabled rather than left in the index beside
the skills that replace it. Spec: docs/bundled-skills-research.md.
"""

from pathlib import Path

import attestation.install as install


def _fresh_home(monkeypatch, tmp_path: Path) -> Path:
    fake_home = tmp_path / "split-home"
    fake_home.mkdir()
    monkeypatch.setattr(install.Path, "home", lambda: fake_home)
    return fake_home


def _profile_skills(fake_home: Path, profile: str = "research") -> Path:
    """A Hermes profile with its own skills tree, the way `hermes profile
    create` lays one out -- a near-mirror of ~/.hermes/skills that nothing
    refreshed until step_skill_copy learned about it."""
    skills = fake_home / ".hermes" / "profiles" / profile / "skills"
    skills.mkdir(parents=True)
    return skills


def test_skill_copy_installs_every_bundled_skill(monkeypatch, tmp_path):
    """One skill became five; the installer syncs all of them, not the first."""
    fake_home = _fresh_home(monkeypatch, tmp_path)

    result = install.step_skill_copy(check=False)

    assert result.status == "FIXED"
    dest = fake_home / ".hermes" / "skills"
    assert set(install.SKILL_NAMES) == {
        "attestation-setup",
        "attestation-feed",
        "attestation-provenance",
        "attestation-knowledge",
        "attestation-symbolic",
    }
    for name in install.SKILL_NAMES:
        assert (dest / name / "SKILL.md").is_file(), name
    assert (dest / "attestation-setup" / "scripts" / "setup.sh").is_file()


def test_skill_copy_syncs_a_profile_skill_tree(monkeypatch, tmp_path):
    """The `research` profile ran a 24.9 KB skill from before `cite.*` existed
    because only ~/.hermes/skills was ever synced."""
    fake_home = _fresh_home(monkeypatch, tmp_path)
    profile_skills = _profile_skills(fake_home)

    install.step_skill_copy(check=False)

    for name in install.SKILL_NAMES:
        assert (profile_skills / name / "SKILL.md").is_file(), name
    main = fake_home / ".hermes" / "skills" / "attestation-feed" / "SKILL.md"
    assert (profile_skills / "attestation-feed" / "SKILL.md").read_bytes() == main.read_bytes()


def test_skill_copy_ignores_a_profile_without_a_skills_dir(monkeypatch, tmp_path):
    """A profile that never had a skills tree is not given one: Hermes reads
    skills from the profile only when the directory exists, and creating it
    would change which tree that profile loads."""
    fake_home = _fresh_home(monkeypatch, tmp_path)
    (fake_home / ".hermes" / "profiles" / "bare").mkdir(parents=True)

    install.step_skill_copy(check=False)

    assert not (fake_home / ".hermes" / "profiles" / "bare" / "skills").exists()


def test_skill_copy_respects_a_profile_disable_marker(monkeypatch, tmp_path):
    """A skill disabled by renaming its SKILL.md (the convention already on
    this machine: `SKILL.md.disabled-collides-with-attestation`) stays
    disabled -- re-creating SKILL.md beside the marker would silently
    re-enable it."""
    fake_home = _fresh_home(monkeypatch, tmp_path)
    profile_skills = _profile_skills(fake_home)
    disabled = profile_skills / "attestation-feed"
    disabled.mkdir()
    (disabled / "SKILL.md.disabled-by-me").write_text("old\n")

    result = install.step_skill_copy(check=False)

    assert not (disabled / "SKILL.md").exists()
    assert (disabled / "SKILL.md.disabled-by-me").read_text() == "old\n"
    # the other four still land in the profile
    assert (profile_skills / "attestation-setup" / "SKILL.md").is_file()
    assert result.status == "FIXED"
    # and a second run is quiet: the marker is not "missing" forever
    assert install.step_skill_copy(check=True).status == "OK"


def test_skill_copy_check_mode_reports_a_stale_profile_copy(monkeypatch, tmp_path):
    fake_home = _fresh_home(monkeypatch, tmp_path)
    profile_skills = _profile_skills(fake_home)
    install.step_skill_copy(check=False)
    (profile_skills / "attestation-feed" / "SKILL.md").write_text("stale\n")

    result = install.step_skill_copy(check=True)

    assert result.status == "BROKEN"
    assert "1 file(s) stale or missing" in result.detail


def test_skill_copy_supersedes_the_legacy_monolith(monkeypatch, tmp_path):
    """The old `research-provenance` skill covered all four surfaces in one
    39 KB body with a description that collides with every one of the five
    that replace it. Left in place it stays in the index beside them. It is
    disabled the way this machine disables skills -- SKILL.md renamed, nothing
    deleted -- and its directory (which may hold a planted data/) is otherwise
    untouched."""
    fake_home = _fresh_home(monkeypatch, tmp_path)
    legacy = fake_home / ".hermes" / "skills" / "research-provenance"
    (legacy / "scripts").mkdir(parents=True)
    (legacy / "SKILL.md").write_text("# the monolith\n")
    (legacy / "scripts" / "setup.sh").write_text("echo old\n")
    (legacy / "data").mkdir()
    (legacy / "data" / "hermes.db").write_bytes(b"\x00db")

    result = install.step_skill_copy(check=False)

    assert result.status == "FIXED"
    assert not (legacy / "SKILL.md").exists()
    assert (legacy / "SKILL.md.superseded-by-attestation-split").read_text() == "# the monolith\n"
    assert (legacy / "scripts" / "setup.sh").read_text() == "echo old\n"
    assert (legacy / "data" / "hermes.db").read_bytes() == b"\x00db"
    assert install.step_skill_copy(check=True).status == "OK"


def test_skill_copy_check_mode_reports_the_legacy_monolith(monkeypatch, tmp_path):
    fake_home = _fresh_home(monkeypatch, tmp_path)
    install.step_skill_copy(check=False)
    legacy = fake_home / ".hermes" / "profiles" / "research" / "skills" / "research-provenance"
    legacy.mkdir(parents=True)
    (legacy / "SKILL.md").write_text("# the monolith\n")

    result = install.step_skill_copy(check=True)

    assert result.status == "BROKEN"
    assert "research-provenance" in result.detail

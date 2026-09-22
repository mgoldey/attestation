"""The manifest is DERIVED, never typed.

agentmarkit hand-transcribed ~40 facts about attestation into
`lab/docs/01-attestation-runtime-map.md` and `lab/presets/research-attestation/
preset.yaml`: entry points, the seven skill names, `resolve_db_path`'s ladder,
env var names including the legacy `RSS_DB`, and which capabilities survive
without a model. Hand-copied facts rot -- their pin sat 81 commits stale and
`HERMES_HOME` was missing entirely because it postdated the transcription.

Every test here asserts a manifest field against the thing it describes, so a
new skill, a new surface or a version bump cannot leave the manifest behind.
`test_manifest_declares_no_credentials` is the one that is not about drift: it
pins agentmarkit's own rule, which attestation should honour regardless of who
is reading -- "a package declares that a server should exist; only the
instance holds its address and its credential".
"""

import json
import tomllib
from pathlib import Path

from attestation import manifest

ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def test_version_and_python_come_from_pyproject():
    m = manifest.build()
    pj = _pyproject()
    assert m["version"] == pj["version"]
    assert m["requires_python"] == pj["requires-python"]


def test_console_scripts_match_the_packaging_exactly():
    """The runtime map names `attest`, `attest-mcp` and (since 0.2.0) the
    `attestation` alias that makes `uvx attestation install` work. A manifest
    that lagged here would send a provisioner at an entry point that is not
    installed."""
    assert manifest.build()["console_scripts"] == _pyproject()["scripts"]


def test_skills_are_the_directories_actually_shipped():
    """Seven today. The count is not written down anywhere in the manifest --
    the list is the directory listing, so an eighth skill needs no edit."""
    shipped = sorted(p.name for p in (ROOT / "src/attestation/skills").iterdir() if p.is_dir())
    assert manifest.build()["skills"] == shipped
    assert shipped, "no skills found; the glob is wrong, not the package"


def test_skills_in_the_manifest_are_in_the_wheel():
    """`export_paths` in their preset lists every skill by path. The wheel is
    what a `uvx attestation install` actually has to copy from, so a skill in
    the manifest that is not packaged is a promise the install cannot keep."""
    from importlib.util import find_spec

    pkg = Path(find_spec("attestation").origin).parent
    for name in manifest.build()["skills"]:
        assert (pkg / "skills" / name / "SKILL.md").is_file(), name


def test_surfaces_come_from_agent_surfaces():
    from attestation.mcp import AGENT_SURFACES

    surfaces = manifest.build()["surfaces"]
    assert set(surfaces) == set(AGENT_SURFACES)
    for name, surface in AGENT_SURFACES.items():
        assert surfaces[name]["summary"] == surface.summary
        assert surfaces[name]["env"] == f"ATTEST_TOOLS={name}"


def test_db_precedence_matches_resolve_db_path(tmp_path, monkeypatch):
    """Not a transcription of the docstring -- the ladder is exercised.

    Their runtime map states this precedence in prose and their preset depends
    on step 4 resolving to the checkout. If the code's order ever changed, the
    manifest would keep asserting the old one; these calls would not.
    """
    from attestation import db

    steps = manifest.build()["db_path_precedence"]
    assert len(steps) == 4

    monkeypatch.setattr(db, "SKILL_DATA_DB", tmp_path / "absent.db")
    monkeypatch.delenv("ATTEST_DB", raising=False)
    monkeypatch.delenv("RSS_DB", raising=False)

    assert db.resolve_db_path("/explicit/x.db") == Path("/explicit/x.db")
    monkeypatch.setenv("RSS_DB", str(tmp_path / "legacy.db"))
    assert db.resolve_db_path(None) == tmp_path / "legacy.db"
    monkeypatch.setenv("ATTEST_DB", str(tmp_path / "current.db"))
    assert db.resolve_db_path(None) == tmp_path / "current.db"
    monkeypatch.delenv("ATTEST_DB")
    monkeypatch.delenv("RSS_DB")
    assert db.resolve_db_path(None) == Path("hermes.db")


def test_declared_env_vars_are_read_by_the_code():
    """`.env.sample` is already pinned to what the code reads
    (test_llm.py::test_env_sample_documents_exactly_the_vars_the_code_reads).
    This keeps the manifest inside that same set rather than inventing a knob.
    """
    sample = (ROOT / ".env.sample").read_text()
    for name in manifest.build()["env"]:
        assert name in sample, f"{name} is in the manifest but not .env.sample"


def test_manifest_declares_no_credentials():
    """agentmarkit's preset schema bans url/token/api_key/secret from an MCP
    entry: a package says a server should exist, the instance holds its
    address and key. Attestation honours that at the source."""
    banned = ("url", "token", "api_key", "secret", "password")
    blob = json.dumps(manifest.build()).lower()
    for server in manifest.build()["mcp_servers"]:
        assert not set(server) & set(banned), server
    for needle in ("api_key", "password", "authorization"):
        assert needle not in blob, needle


def test_every_mcp_server_carries_a_verification():
    """Their schema: 'declare the check that proves the server is really
    there'. A declaration nobody can verify is the thing being avoided."""
    for server in manifest.build()["mcp_servers"]:
        assert server["name"]
        assert server["verify"], server["name"]


def test_the_degradation_split_names_real_capabilities():
    """Their `degrades_without` is copied from prose in the runtime map. Both
    halves must be non-empty: 'everything breaks' and 'nothing breaks' are the
    two claims that make the field useless."""
    d = manifest.build()["degrades_without_model"]
    assert d["still_works"] and d["unavailable"]
    assert not set(d["still_works"]) & set(d["unavailable"])


def test_manifest_is_json_serialisable_and_schema_versioned():
    m = manifest.build()
    assert m["schema"] == "attestation/manifest/1"
    assert json.loads(json.dumps(m)) == m


def test_embedding_block_states_the_constraint_a_packager_must_satisfy():
    """agentmarkit asked its buyer for "embedding dimensions" as one of six
    required_user_inputs. That question is malformed, and the manifest is
    where the real constraint belongs.

    EMBED_DIMS is not the model's width -- it is a Matryoshka SLICE.
    `embed.truncate_normalize` takes any vector of at least that many dims,
    slices to it and renormalises; a narrower one RAISES rather than
    zero-padding, which would fabricate signal. So the constraint is
    one-directional (>= stored_dims works, wider is fine, narrower is not),
    and `attest install --check` already probes it live -- step_hosted_models
    makes a real embedding request and measures the reply.

    Stating it here turns "what are your embedding dimensions?", which a buyer
    cannot answer and can answer WRONG permanently (the vec0 table width is
    fixed at first ingest), into a check the installer runs.
    """
    from attestation.db import embed_dims

    block = manifest.build()["embedding"]

    assert block["stored_dims"] == embed_dims()
    # The floor IS the stored width: that is what truncate_normalize requires.
    assert block["min_model_dims"] == embed_dims()
    assert block["probe"], "a constraint with no way to check it is trivia"
    assert "fresh" in block["fixed_at"].lower(), block["fixed_at"]


def test_the_embedding_floor_matches_what_truncate_normalize_enforces():
    """Not a transcription: the manifest's floor is exercised against the
    function that enforces it, so the two cannot drift.
    """
    import numpy as np
    import pytest

    from attestation.embed import truncate_normalize

    floor = manifest.build()["embedding"]["min_model_dims"]

    wide = truncate_normalize(np.ones(floor + 128, dtype=np.float32))
    assert len(wide) == floor, "a wider model must be sliced to the stored width"

    exact = truncate_normalize(np.ones(floor, dtype=np.float32))
    assert len(exact) == floor

    with pytest.raises(ValueError):
        truncate_normalize(np.ones(floor - 1, dtype=np.float32))

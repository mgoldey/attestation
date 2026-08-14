import hashlib

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _hermetic_env(tmp_path, monkeypatch):
    """Tests never see the shell's project env vars or the real checkout .env.

    delenv strips inherited values; repointing _REPO_ROOT keeps load_env()
    (invoked by every main()-calling test) from importing the checkout's .env
    into os.environ mid-suite.

    The tmp_path gets a pyproject.toml marker so it looks like a real
    checkout to install._checkout_root(); tests that need the packaged-install
    behavior repoint _REPO_ROOT at a markerless directory themselves.
    """
    import attestation.llm

    for var in (*attestation.llm.ENV_VARS, "EMBED_DIMS", "RSS_DB"):
        monkeypatch.delenv(var, raising=False)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "attestation"\n')
    monkeypatch.setattr(attestation.llm, "_REPO_ROOT", tmp_path)


class FakeEmbedder:
    """Deterministic stand-in for Embedder: text -> stable unit vector."""

    dims = 256

    def _vec(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(256).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_document(self, title: str, text: str) -> np.ndarray:
        return self._vec(f"doc:{title}:{text}")

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(f"query:{text}")


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()

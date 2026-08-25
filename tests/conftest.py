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

    The two per-user TOML ladders are pointed at paths inside tmp_path that do
    not exist. Both fall back to ~/.hermes/, so a contributor who followed the
    advice in ledger.py's own "no known direction" error -- declare it under
    [metric_direction] -- then watched an unrelated test fail:
    test_examples.py asserts ndcg_at_10 has NO declared direction, which was
    true only of a machine where nobody had declared one. CI passes because
    its runners have no ~/.hermes. Repointing rather than deleting the env var
    is deliberate: an unset variable falls back to $HOME, which is the bug.
    """
    import attestation.corpus
    import attestation.ledger
    import attestation.llm

    for var in (*attestation.llm.ENV_VARS, "EMBED_DIMS", "RSS_DB"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(
        attestation.ledger.METRIC_DIRECTION_PATH_ENV, str(tmp_path / "absent-metric_direction.toml")
    )
    monkeypatch.setenv(attestation.corpus.CORPUS_FILE_ENV, str(tmp_path / "absent-corpus.toml"))
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


def seeded_db(path):
    """get_db plus the three demo personas.

    get_db creates an EMPTY database (see its docstring); the tests that
    exercise ranking, digests and the web UI need personas to rank for, and
    most were written when get_db seeded them itself. INSERT OR IGNORE, so
    reopening through this helper is as safe as reopening through get_db.
    """
    from attestation.db import get_db, seed_demo_users

    conn = get_db(path)
    seed_demo_users(conn)
    return conn

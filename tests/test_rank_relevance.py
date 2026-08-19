"""Golden-set relevance test: does ranking actually recover relevant items?

Every other rank test runs against `FakeEmbedder` (tests/conftest.py), which
hashes text with SHA-256 into a Gaussian vector -- semantic distance carries
zero information there by design (see conftest.py docstring). That makes the
rest of the suite a plumbing check: it can catch a crash or a wiring error,
but it cannot tell a correct recommender from an inverted one, because
"inverted" and "correct" look identical when similarity is noise.

This file is the one place semantic similarity is real. tests/fixtures/
golden_vectors.npz holds 30 real `embeddinggemma` embeddings (title + text,
via attestation.embed.Embedder, truncated/normalized to 256 dims exactly like
production) for a hand-built corpus: 10 quantum-computing items and 20
distractors (neuroscience, climate science, molecular biology). A
quantum-computing persona's interests embedding is included as
`query_vector`. Regenerate via the script noted at the bottom of this file if
the fixture ever needs to change -- do not hand-edit the .npz.

The test replaces only the embedder (a `GoldenEmbedder` that looks up
precomputed vectors by title/text instead of calling Ollama) -- `rank_items`,
`_candidate_items`, `_profile_vector`, and the DB schema are all exercised
for real. No network access at test time.
"""

from pathlib import Path

import numpy as np
import pytest

from attestation.db import get_db
from attestation.rank import rank_items

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "golden_vectors.npz"


class GoldenEmbedder:
    """Serves precomputed real embeddings from the golden fixture.

    Not a FakeEmbedder: vectors come from an actual embedding model run
    offline (see regen script below), so cosine similarity between them
    reflects real semantic relatedness. `embed_document` is keyed by title
    (unique in the fixture); `embed_query` returns the fixture's single
    precomputed persona query vector for the expected persona text and raises
    for anything else, so a typo in the test can't silently fall through to
    an unrelated vector.
    """

    dims = 256

    def __init__(self, titles, vectors, query_text, query_vector):
        self._by_title = dict(zip(titles.tolist(), vectors, strict=True))
        self._query_text = str(query_text)
        self._query_vector = query_vector

    def embed_document(self, title: str, text: str) -> np.ndarray:
        return self._by_title[title]

    def embed_query(self, text: str) -> np.ndarray:
        if text != self._query_text:
            raise KeyError(f"GoldenEmbedder has no query vector for {text!r}")
        return self._query_vector


@pytest.fixture
def golden():
    data = np.load(FIXTURE_PATH, allow_pickle=True)
    return {
        "titles": data["titles"],
        "vectors": data["vectors"],
        "query_text": str(data["query_text"]),
        "query_vector": data["query_vector"],
        "relevant_mask": data["relevant_mask"],
    }


def _seed_corpus(conn, embedder, titles, vectors):
    ids = []
    for i, title in enumerate(titles):
        cur = conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, published, content_hash)"
            " VALUES (NULL, ?, 'http://x', ?, datetime('now'), ?)",
            (title, f"summary of {title}", f"hash-{title}"),
        )
        conn.execute(
            "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)",
            (cur.lastrowid, vectors[i].tobytes()),
        )
        ids.append(cur.lastrowid)
    conn.commit()
    return ids


def _ndcg_at_k(relevance_in_rank_order: list[int], k: int = 10) -> float:
    """Standard NDCG@k for binary relevance (no discount subtlety needed)."""
    rel = relevance_in_rank_order[:k]
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rel))
    ideal = sorted(relevance_in_rank_order, reverse=True)[:k]
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def test_ranking_beats_random_and_recency_baselines(tmp_path, golden):
    """rank_items on the golden fixture must clearly outperform both a fixed-seed
    random shuffle and a recency-only ordering, measured by NDCG@10.

    This is the test designed to catch the verified mutation: inverting
    `profile_rank = ranks(-(X @ profile_vec))` to rank by anti-similarity
    (the maximally-wrong recommender) collapses NDCG@10 for this fixture to
    ~0 (all 10 relevant items are the LEAST similar under the true profile
    direction, since the fixture's real embedder made them the MOST similar
    -- inverting the sign inverts the entire order). The margin below is
    generous relative to run-to-run noise (there is none: everything here is
    static data and a fixed-seed baseline) but tight enough that the
    anti-similarity mutation fails it.
    """
    conn = get_db(tmp_path / "golden.db")
    embedder = GoldenEmbedder(
        golden["titles"], golden["vectors"], golden["query_text"], golden["query_vector"]
    )
    titles = golden["titles"].tolist()
    mask = golden["relevant_mask"]

    # Insert in a fixed-seed shuffled order rather than the fixture's authored
    # order (which happens to list all 10 relevant items first). Otherwise
    # "recency" (insertion/id order) would trivially reproduce perfect
    # relevance ordering regardless of any ranking logic, making it a
    # worthless baseline.
    perm = np.random.default_rng(1).permutation(len(titles))
    shuffled_titles = [titles[i] for i in perm]
    shuffled_vectors = golden["vectors"][perm]
    shuffled_mask = mask[perm]

    ids = _seed_corpus(conn, embedder, shuffled_titles, shuffled_vectors)
    relevant_ids = {ids[i] for i, r in enumerate(shuffled_mask) if r}

    cur = conn.execute(
        "INSERT INTO users(name, interests) VALUES (?, ?)",
        ("quantum-persona", golden["query_text"]),
    )
    conn.commit()
    user_id = cur.lastrowid

    # since_days=None: items above were inserted with published=now via
    # datetime('now'), but rank_items' default window is 14 days anyway --
    # None just makes the fixture's semantics explicit and independent of
    # rank_items' default should that default ever change.
    result = rank_items(conn, embedder, user_id, since_days=None, exclude_clicked=False)
    assert len(result) == 30

    ranked_relevance = [1 if r.item_id in relevant_ids else 0 for r in result]
    model_ndcg = _ndcg_at_k(ranked_relevance, k=10)

    # Random baseline: fixed-seed shuffle of the same 30 relevance labels.
    rng = np.random.default_rng(0)
    random_relevance = list(golden["relevant_mask"])
    rng.shuffle(random_relevance)
    random_ndcg = _ndcg_at_k(random_relevance, k=10)

    # Recency baseline: all items share the same published timestamp here, so
    # "recency order" is insertion order (id order) -- a defensible stand-in
    # for "no personalization signal at all, just show what's newest."
    recency_relevance = [1 if i in relevant_ids else 0 for i in ids]
    recency_ndcg = _ndcg_at_k(recency_relevance, k=10)

    assert model_ndcg > random_ndcg + 0.3, (
        f"model NDCG@10={model_ndcg:.3f} did not clear random "
        f"baseline={random_ndcg:.3f} by the required margin"
    )
    assert model_ndcg > recency_ndcg + 0.3, (
        f"model NDCG@10={model_ndcg:.3f} did not clear recency "
        f"baseline={recency_ndcg:.3f} by the required margin"
    )
    # The fixture was built so profile similarity perfectly separates the two
    # classes (see gen script) -- pin the achieved value too, not just the
    # margin, so a silent quality regression that still clears the loose
    # margin above still shows up as a diff here.
    assert model_ndcg == pytest.approx(1.0)


# Fixture regeneration (not run by pytest): requires a local Ollama serving
# `embeddinggemma` at LLM_BASE_URL. See the script this docstring describes;
# it lived at repo-root scratch during authoring and is reproduced here for
# anyone who needs to regenerate the fixture:
#
#   .venv/bin/python -c "
#   from attestation.embed import Embedder
#   import numpy as np
#   items = [...]  # (title, text) pairs, see this file's ITEMS-shaped list
#   embedder = Embedder()
#   vecs = np.stack([embedder.embed_document(t, x) for t, x in items])
#   query_vector = embedder.embed_query('<persona interests text>')
#   np.savez_compressed('tests/fixtures/golden_vectors.npz',
#       titles=np.array([t for t, _ in items]), vectors=vecs.astype('float32'),
#       query_text=np.array('<persona interests text>'),
#       query_vector=query_vector.astype('float32'), relevant_mask=np.array([...]))
#   "

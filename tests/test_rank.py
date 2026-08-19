import hashlib

import numpy as np
import pytest

from attestation.db import get_db
from attestation.rank import (
    RankedItem,
    avg_ranks,
    blend_weight,
    bootstrap_persona,
    classifier_probs,
    evaluate_user,
    get_user,
    rank_items,
    ranks,
    record_click,
)


def add_item(conn, embedder, title, days_ago=0):
    cur = conn.execute(
        "INSERT INTO items(feed_id, title, url, summary, published, content_hash)"
        " VALUES (NULL, ?, 'http://x', ?, datetime('now', ?), ?)",
        (title, f"summary of {title}", f"-{days_ago} days", f"hash-{title}"),
    )
    vec = embedder.embed_document(title, f"summary of {title}")
    conn.execute(
        "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)",
        (cur.lastrowid, vec.tobytes()),
    )
    return cur.lastrowid


def seed_corpus(conn, embedder, n=20):
    return [add_item(conn, embedder, f"item {i}") for i in range(n)]


def get_user_id(conn, name: str) -> int:
    """get_user() is typed Row | None; tests always seed the user, so assert and
    narrow here once instead of repeating the None-check at every call site."""
    user = get_user(conn, name)
    assert user is not None
    return user["id"]


def test_blend_weight_ramp():
    assert blend_weight(0) == 0.0
    assert np.isclose(blend_weight(5), 0.5)
    assert blend_weight(20) > 0.75


def test_ranks_lower_is_better():
    r = ranks(np.array([0.1, 0.9, 0.5]))
    assert list(r) == [2, 0, 1]


def test_cold_start_no_clicks_uses_profile(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    seed_corpus(conn, fake_embedder)
    user_id = get_user_id(conn, "matt")
    result = rank_items(conn, fake_embedder, user_id)
    assert len(result) == 20
    assert isinstance(result[0], RankedItem)
    assert result[0].score <= result[-1].score  # best (lowest) first


def test_classifier_guard_single_class(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    ids = seed_corpus(conn, fake_embedder)
    user_id = get_user_id(conn, "matt")
    for i in ids[:4]:  # four clicks, ALL positive -> one class
        conn.execute("INSERT INTO clicks(user_id, item_id, useful) VALUES (?, ?, 1)", (user_id, i))
    X = np.stack([fake_embedder.embed_document(f"item {i}", "") for i in range(3)])
    assert classifier_probs(conn, user_id, X) is None
    # rank_items must not crash on single-class history
    assert rank_items(conn, fake_embedder, user_id)


def test_clicked_items_excluded(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    ids = seed_corpus(conn, fake_embedder)
    user_id = get_user_id(conn, "matt")
    conn.execute("INSERT INTO clicks(user_id, item_id, useful) VALUES (?, ?, 1)", (user_id, ids[0]))
    result = rank_items(conn, fake_embedder, user_id)
    assert ids[0] not in [r.item_id for r in result]


def test_recency_window(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    add_item(conn, fake_embedder, "fresh", days_ago=1)
    add_item(conn, fake_embedder, "stale", days_ago=40)
    result = rank_items(conn, fake_embedder, get_user_id(conn, "matt"))
    assert [r.title for r in result] == ["fresh"]


def test_clicks_shift_ranking(tmp_path, fake_embedder):
    """After mixed clicks, classifier blends in and changes the order."""
    conn = get_db(tmp_path / "t.db")
    ids = seed_corpus(conn, fake_embedder, n=30)
    user_id = get_user_id(conn, "matt")
    before = [r.item_id for r in rank_items(conn, fake_embedder, user_id)]
    for i in ids[:5]:
        conn.execute("INSERT INTO clicks(user_id, item_id, useful) VALUES (?, ?, 1)", (user_id, i))
    for i in ids[5:10]:
        conn.execute("INSERT INTO clicks(user_id, item_id, useful) VALUES (?, ?, 0)", (user_id, i))
    after = [r.item_id for r in rank_items(conn, fake_embedder, user_id)]
    remaining = [i for i in before if i not in ids[:10]]
    assert after != remaining  # order changed among unclicked items


def test_persona_ordering_differs(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    seed_corpus(conn, fake_embedder, n=15)
    chem = rank_items(conn, fake_embedder, get_user_id(conn, "bench-chemist"))
    ml = rank_items(conn, fake_embedder, get_user_id(conn, "ml-engineer"))
    assert [r.item_id for r in chem] != [r.item_id for r in ml]


def test_bootstrap_persona_writes_clicks(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    seed_corpus(conn, fake_embedder, n=40)
    n = bootstrap_persona(conn, fake_embedder, "bench-chemist", k=30)
    assert n == 30
    rows = conn.execute(
        "SELECT useful, COUNT(*) c FROM clicks GROUP BY useful ORDER BY useful"
    ).fetchall()
    assert [r["c"] for r in rows] == [15, 15]


def test_evaluate_user_insufficient_data(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    seed_corpus(conn, fake_embedder, n=5)
    assert evaluate_user(conn, get_user_id(conn, "matt")) is None


def test_evaluate_user_excludes_bootstrap_leakage(tmp_path, fake_embedder):
    """bootstrap_persona labels are a deterministic linear function of the same
    embedding the classifier trains on (argsort(X @ profile_vec)), so scoring
    over them is a tautological AUC of 1.0, not a measurement. evaluate_user
    must exclude source='bootstrap' clicks rather than report that leaked 1.0."""
    conn = get_db(tmp_path / "t.db")
    seed_corpus(conn, fake_embedder, n=60)
    user_id = get_user_id(conn, "bench-chemist")
    bootstrap_persona(conn, fake_embedder, "bench-chemist", k=30)
    # Only bootstrap clicks exist -- after filtering there is nothing left to
    # evaluate on, so this must be an honest None, never the leaked 1.0.
    assert evaluate_user(conn, user_id) is None


def test_evaluate_user_mixed_real_clicks_returns_float(tmp_path, fake_embedder):
    """A persona with genuinely mixed non-bootstrap clicks (interleaved labels,
    not a label-sorted run) yields a real AUC instead of tripping the
    single-class-tail guard that leave-last-N-out was vulnerable to."""
    conn = get_db(tmp_path / "t.db")
    item_ids = seed_corpus(conn, fake_embedder, n=40)
    user_id = get_user_id(conn, "matt")
    # Alternate useful/not-useful so the tail is never single-class regardless
    # of insertion order -- this is the "naturally label-sorted rating
    # session" failure mode the leave-last-N-out split fell into.
    for i, item_id in enumerate(item_ids):
        record_click(conn, user_id, item_id, useful=bool(i % 2), source="ui")
    result = evaluate_user(conn, user_id)
    assert result is not None
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


class SpyEmbedder:
    """Wraps fake_embedder, counting embed_query calls."""

    def __init__(self, inner):
        self.inner = inner
        self.dims = inner.dims
        self.embed_query_calls = 0

    def embed_document(self, title, text):
        return self.inner.embed_document(title, text)

    def embed_query(self, text):
        self.embed_query_calls += 1
        return self.inner.embed_query(text)


def test_profile_vector_cached_across_calls(tmp_path, fake_embedder):
    """Second rank_items call for the same user + unchanged interests must not
    re-embed the profile text."""
    conn = get_db(tmp_path / "t.db")
    seed_corpus(conn, fake_embedder)
    user_id = get_user_id(conn, "matt")
    spy = SpyEmbedder(fake_embedder)

    rank_items(conn, spy, user_id)
    assert spy.embed_query_calls == 1

    rank_items(conn, spy, user_id)
    assert spy.embed_query_calls == 1  # still 1: served from cache


def test_profile_vector_survives_embedder_failure_after_warm_call(tmp_path, fake_embedder):
    """Once the profile vector is cached, an embedder outage must not break ranking."""
    conn = get_db(tmp_path / "t.db")
    seed_corpus(conn, fake_embedder)
    user_id = get_user_id(conn, "matt")

    # warm the cache
    warm = rank_items(conn, fake_embedder, user_id)
    assert warm

    class DyingEmbedder:
        dims = fake_embedder.dims

        def embed_document(self, title, text):
            return fake_embedder.embed_document(title, text)

        def embed_query(self, text):
            raise ConnectionError("ollama down")

    result = rank_items(conn, DyingEmbedder(), user_id)
    assert result  # served from cache despite embedder outage


def _tag_item(conn, item_id, content_type, tags):
    conn.execute(
        "INSERT OR REPLACE INTO item_features(item_id, content_type, model) VALUES (?, ?, 'm')",
        (item_id, content_type),
    )
    for t in tags:
        conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, ?)", (item_id, t))


def test_downvoted_tag_sinks_similar_item_even_single_class(tmp_path, fake_embedder):
    """Covers two spec behaviors: downvoted-tag demotion, and only-downvotes users
    (single-class history disables the classifier but NOT the pref term)."""
    conn = get_db(tmp_path / "t.db")
    ids = seed_corpus(conn, fake_embedder, n=40)
    user_id = get_user_id(conn, "matt")
    for i in ids[:3]:  # three items share a tag+type; the third is the survivor
        _tag_item(conn, i, "announcement", ["junk"])
    downvoted = ids[:2] + ids[3:21]  # 20 downvotes, ALL useful=0 -> single class
    for i in downvoted:
        conn.execute("INSERT INTO clicks(user_id, item_id, useful) VALUES (?, ?, 0)", (user_id, i))
    # classifier is off (single class) -- any influence below is the pref term
    assert classifier_probs(conn, user_id, np.zeros((1, 256), dtype=np.float32)) is None
    result = rank_items(conn, fake_embedder, user_id)
    assert len(result) == 20  # 40 seeded - 20 clicked
    pos = {r.item_id: i for i, r in enumerate(result)}
    # Deterministic bound, independent of tie-breaking and profile order:
    # w = blend_weight(20) = 0.8. With avg_ranks, the 19 neutral items all share
    # mean pref rank 9 ((0+...+18)/19), while the survivor uniquely holds pref
    # rank 19. Every neutral item's final <= 0.8*9 + 0.2*19 = 11.0, while the
    # survivor's final >= 0.8*19 + 0.2*0 = 15.2 for ANY profile order. So the
    # survivor is strictly last.
    assert pos[ids[2]] == 19


def test_clicks_without_feature_data_leave_profile_order_intact(tmp_path, fake_embedder):
    """Pref term must not inject tie-break noise when no feature key has click data."""
    conn = get_db(tmp_path / "t.db")
    ids = seed_corpus(conn, fake_embedder)
    user_id = get_user_id(conn, "matt")
    conn.execute("INSERT INTO clicks(user_id, item_id, useful) VALUES (?, ?, 0)", (user_id, ids[0]))
    with_click = [r.item_id for r in rank_items(conn, fake_embedder, user_id)]
    conn.execute("DELETE FROM clicks WHERE user_id = ?", (user_id,))
    baseline = [r.item_id for r in rank_items(conn, fake_embedder, user_id) if r.item_id != ids[0]]
    assert with_click == baseline  # identical order: pref term contributed nothing


def test_no_clicks_ranking_unchanged_by_tags(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    ids = seed_corpus(conn, fake_embedder)
    user_id = get_user_id(conn, "matt")
    baseline = [r.item_id for r in rank_items(conn, fake_embedder, user_id)]
    _tag_item(conn, ids[0], "paper", ["dft"])
    assert [r.item_id for r in rank_items(conn, fake_embedder, user_id)] == baseline


def test_ranked_items_carry_tags_and_content_type(tmp_path, fake_embedder):
    conn = get_db(tmp_path / "t.db")
    ids = seed_corpus(conn, fake_embedder)
    user_id = get_user_id(conn, "matt")
    _tag_item(conn, ids[0], "paper", ["dft", "catalysis"])
    result = rank_items(conn, fake_embedder, user_id)
    by_id = {r.item_id: r for r in result}
    assert by_id[ids[0]].content_type == "paper"
    assert by_id[ids[0]].tags == ["catalysis", "dft"]  # alphabetical
    assert by_id[ids[1]].content_type is None
    assert by_id[ids[1]].tags == []


def test_avg_ranks_ties_share_mean_rank():
    r = avg_ranks(np.array([0.5, 0.9, 0.5, 0.5]))
    assert r[1] == 0.0
    assert list(r[[0, 2, 3]]) == [2.0, 2.0, 2.0]  # tied block spans ranks 1..3, mean 2


def test_partial_tie_neutral_items_keep_profile_relative_order(tmp_path, fake_embedder):
    """With avg_ranks, items sharing the neutral pref score keep their profile-order
    relative to each other even while a downvoted-tag item carries real pref signal."""
    conn = get_db(tmp_path / "t.db")
    ids = seed_corpus(conn, fake_embedder)
    user_id = get_user_id(conn, "matt")
    _tag_item(conn, ids[0], "announcement", ["junk"])
    conn.execute("INSERT INTO clicks(user_id, item_id, useful) VALUES (?, ?, 0)", (user_id, ids[0]))
    with_click = [r.item_id for r in rank_items(conn, fake_embedder, user_id)]
    conn.execute("DELETE FROM clicks WHERE user_id = ?", (user_id,))
    baseline = [r.item_id for r in rank_items(conn, fake_embedder, user_id) if r.item_id != ids[0]]
    # every candidate is neutral (untagged, no feed) -> all share one avg pref rank ->
    # relative order must exactly match the profile-only baseline
    assert with_click == baseline


def test_record_click_writes_source_and_rejects_invalid(tmp_path):
    from attestation.db import get_db
    from attestation.rank import record_click

    conn = get_db(tmp_path / "t.db")
    uid = get_user_id(conn, "matt")
    conn.execute(
        "INSERT INTO items(id, feed_id, title, url, summary, content_hash)"
        " VALUES (1, NULL, 't', 'u', 's', 'h')"
    )
    conn.commit()

    record_click(conn, uid, 1, True, source="agent")
    row = conn.execute("SELECT useful, source FROM clicks WHERE item_id = 1").fetchone()
    assert row["source"] == "agent"
    assert row["useful"] == 1

    # re-recording the same (user, item) replaces rather than duplicating
    record_click(conn, uid, 1, False, source="ui")
    rows = conn.execute("SELECT useful, source FROM clicks WHERE item_id = 1").fetchall()
    assert len(rows) == 1
    assert rows[0]["source"] == "ui"
    assert rows[0]["useful"] == 0

    with pytest.raises(ValueError, match="invalid click source"):
        record_click(conn, uid, 1, True, source="telepathy")
    conn.close()


def test_create_user_returns_id_and_rejects_duplicates(tmp_path):
    from attestation.db import get_db
    from attestation.rank import create_user, get_user

    conn = get_db(tmp_path / "t.db")

    uid = create_user(conn, "newbie", "protein folding, cryo-EM")

    newbie = get_user(conn, "newbie")
    assert newbie is not None
    assert newbie["id"] == uid
    assert newbie["interests"] == "protein folding, cryo-EM"
    with pytest.raises(ValueError, match="already exists"):
        create_user(conn, "newbie", "something else")
    conn.close()


def test_candidate_items_can_include_clicked_and_drop_window(tmp_path, fake_embedder):
    """search_feed needs both: default behavior must be unchanged."""
    from attestation.db import get_db
    from attestation.rank import _candidate_items, record_click

    conn = get_db(tmp_path / "t.db")
    user_id = get_user_id(conn, "matt")
    # one recent item, one far outside the default 14-day window
    for i, published in ((1, "datetime('now')"), (2, "datetime('now', '-400 days')")):
        conn.execute(
            f"INSERT INTO items(id, feed_id, title, url, summary, published, content_hash)"
            f" VALUES ({i}, NULL, 't{i}', 'u', 's', {published}, 'h{i}')"
        )
        vec = fake_embedder.embed_document(f"t{i}", "s")
        conn.execute("INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)", (i, vec.tobytes()))
    conn.commit()
    record_click(conn, user_id, 1, True)

    default_ids = {r["id"] for r in _candidate_items(conn, user_id, 14)}
    assert default_ids == set(), "clicked item and out-of-window item both excluded by default"

    search_ids = {r["id"] for r in _candidate_items(conn, user_id, None, exclude_clicked=False)}
    assert search_ids == {1, 2}
    conn.close()


def test_record_click_defaults_to_ui(tmp_path):
    from attestation.db import get_db
    from attestation.rank import record_click

    conn = get_db(tmp_path / "t.db")
    uid = get_user_id(conn, "matt")
    conn.execute(
        "INSERT INTO items(id, feed_id, title, url, summary, content_hash)"
        " VALUES (1, NULL, 't', 'u', 's', 'h')"
    )
    conn.commit()

    record_click(conn, uid, 1, True)

    assert conn.execute("SELECT source FROM clicks").fetchone()["source"] == "ui"
    conn.close()


class _SmallDimsFakeEmbedder:
    """Like conftest's FakeEmbedder but at a small, configurable dimensionality
    -- keeps a >32766-item ranking test fast without needing EMBED_DIMS=256."""

    def __init__(self, dims=16):
        self.dims = dims

    def _vec(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dims).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_document(self, title, text):
        return self._vec(f"doc:{title}:{text}")

    def embed_query(self, text):
        return self._vec(f"query:{text}")


def test_rank_items_beyond_sqlite_variable_limit(tmp_path, monkeypatch):
    """search_feed calls rank_items(since_days=None, exclude_clicked=False), which
    by design ranks every item in the archive. Past SQLite's default
    SQLITE_LIMIT_VARIABLE_NUMBER (32766), a naive `IN (?,?,...)` built with one
    placeholder per candidate raises OperationalError. A small EMBED_DIMS keeps
    this fast: the vector math is the same regardless of dimensionality."""
    monkeypatch.setenv("EMBED_DIMS", "16")
    embedder = _SmallDimsFakeEmbedder(dims=16)
    conn = get_db(tmp_path / "t.db")
    n = 32766 + 500
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        vec = rng.standard_normal(16).astype(np.float32)
        vec /= np.linalg.norm(vec)
        rows.append((i + 1, f"item {i}", "http://x", f"summary {i}", f"hash-{i}", vec.tobytes()))
    conn.executemany(
        "INSERT INTO items(id, feed_id, title, url, summary, content_hash) "
        "VALUES (?, NULL, ?, ?, ?, ?)",
        [(r[0], r[1], r[2], r[3], r[4]) for r in rows],
    )
    conn.executemany(
        "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)",
        [(r[0], r[5]) for r in rows],
    )
    conn.commit()

    result = rank_items(
        conn, embedder, get_user_id(conn, "matt"), since_days=None, exclude_clicked=False
    )

    assert len(result) == n
    conn.close()

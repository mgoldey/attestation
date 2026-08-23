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


def test_updating_interests_evicts_the_cached_profile_vector(tmp_path, fake_embedder, monkeypatch):
    """A changed persona must not keep ranking against its old interests.

    The hash check normally saves this: changed text misses the cache and gets
    recomputed. But the embedder-down fallback returns the cached vector
    WITHOUT comparing hashes -- deliberately, since a stale vector beats a dead
    feed. Those two behaviours combine badly: update the interests, lose the
    embedder, and the fallback serves a vector computed from text the user
    already replaced, with no signal that it happened.

    Eviction on update closes it. The user then gets the honest error (cold
    cache, embedder unavailable) instead of silently-wrong ranking.
    """
    from attestation import rank
    from attestation.mcp import feed as feed_mod

    db = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(db))  # the tool resolves its own connection
    conn = get_db(db)
    conn.execute("INSERT INTO users(name, interests) VALUES ('ana', 'quantum chemistry')")
    conn.commit()
    user_id = conn.execute("SELECT id FROM users WHERE name='ana'").fetchone()["id"]

    rank._PROFILE_VEC_CACHE.clear()
    warm = rank._profile_vector(conn, fake_embedder, user_id, "quantum chemistry")
    key = (rank._db_identity(conn), user_id)
    assert key in rank._PROFILE_VEC_CACHE

    conn.close()
    feed_mod._update_persona("ana", "medieval poetry")

    conn = get_db(db)
    assert (rank._db_identity(conn), user_id) not in rank._PROFILE_VEC_CACHE, (
        "update_persona must evict; otherwise the embedder-down fallback serves "
        "a vector computed from the interests text the user just replaced"
    )

    class DeadEmbedder:
        dims = 256

        def embed_query(self, text):
            raise RuntimeError("ollama is down")

        def embed_document(self, title, text):
            raise RuntimeError("ollama is down")

    with pytest.raises(RuntimeError, match="no cached profile vector"):
        rank._profile_vector(conn, DeadEmbedder(), user_id, "medieval poetry")

    assert warm is not None
    conn.close()


def test_one_click_does_not_reorder_the_whole_feed(tmp_path, fake_embedder):
    """A single click must not move items by hundreds of positions.

    Measured on the live corpus before this guard: one positive click on
    materials-scientist's own top item left 1 of the top 10 in place and pushed
    the rest from #4 to #187 and #9 to #196. A new reader's feed got worse the
    moment they engaged with it, which is the opposite of the intended effect.

    Two causes, both in the preference term. `_score` is Laplace-smoothed as
    (u+1)/(u+n+2), so a positives-only history can never score any key BELOW
    0.5 -- the term stops measuring preference and starts measuring "how many
    of this item's keys has the reader touched at all". And it fired at
    n_clicks > 0, so one observation moved every candidate.
    """
    conn = get_db(tmp_path / "t.db")
    conn.execute("INSERT INTO users(name, interests) VALUES ('ana', 'protein folding')")
    for i in range(1, 61):
        cur = conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash)"
            " VALUES (NULL, ?, 'u', ?, ?)",
            (f"Item {i}", f"Summary {i}", f"h{i}"),
        )
        vec = fake_embedder.embed_document(f"Item {i}", f"Summary {i}")
        conn.execute(
            "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)",
            (cur.lastrowid, vec.tobytes()),
        )
        conn.execute(
            "INSERT INTO item_tags(item_id, tag) VALUES (?, ?)", (cur.lastrowid, f"t{i % 7}")
        )
    conn.commit()
    uid = get_user(conn, "ana")["id"]

    before = [it.item_id for it in rank_items(conn, fake_embedder, uid, since_days=None)]
    record_click(conn, uid, before[0], True)
    conn.commit()
    after = [it.item_id for it in rank_items(conn, fake_embedder, uid, since_days=None)]

    kept = len(set(before[:10]) & set(after[:10]))
    assert kept >= 8, (
        f"one click changed {10 - kept} of the top 10; a single observation "
        "must not reorder the feed"
    )
    conn.close()


def test_the_preference_term_waits_for_both_classes(tmp_path, fake_embedder):
    """It cannot separate anything from a single-class history.

    With only positives every key scores at or above neutral, so the ordering
    it produces reflects coverage rather than taste. The classifier already
    refuses to fire in that situation (classifier_probs returns None); the
    preference term now applies the same rule instead of confidently ranking
    on nothing.
    """
    from attestation.rank import _preference_ready

    conn = get_db(tmp_path / "t.db")
    conn.execute("INSERT INTO users(name, interests) VALUES ('ana', 'x')")
    for i in range(1, 31):
        conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash)"
            " VALUES (NULL, ?, 'u', 's', ?)",
            (f"t{i}", f"h{i}"),
        )
    conn.commit()
    uid = get_user(conn, "ana")["id"]

    assert not _preference_ready(conn, uid), "no clicks at all"
    for i in range(1, 13):
        record_click(conn, uid, i, True)
    conn.commit()
    assert not _preference_ready(conn, uid), "12 upvotes cannot rank anything below neutral"

    record_click(conn, uid, 13, False)
    conn.commit()
    assert _preference_ready(conn, uid), "one downvote gives the term something to separate"

    # A purely-downvoting reader keeps its signal: not-useful takes a key to
    # 0.333 and below, which genuinely distinguishes it from untouched at 0.5.
    conn.execute("DELETE FROM clicks WHERE user_id = ?", (uid,))
    record_click(conn, uid, 1, False)
    conn.commit()
    assert _preference_ready(conn, uid), "a single downvote is real signal"
    conn.close()


# ---------------------------------------------------------------------------
# Provenance of the training signal
# ---------------------------------------------------------------------------


def _mixed_history(tmp_path, embedder, sources):
    """A user whose clicks come from the given provenance sources, alternating
    useful/not-useful so the classifier's single-class guard never fires."""
    from attestation.db import get_db
    from attestation.rank import create_user

    conn = get_db(tmp_path / "t.db")
    user_id = create_user(conn, "mixed", "machine learning")
    for i, source in enumerate(sources):
        item_id = add_item(conn, embedder, f"item {i}")
        conn.execute(
            "INSERT INTO clicks(user_id, item_id, useful, source) VALUES (?,?,?,?)",
            (user_id, item_id, i % 2, source),
        )
    conn.commit()
    return conn, user_id


def test_ranking_quality_says_how_much_of_its_training_is_real(tmp_path, fake_embedder):
    """A click count mixing real judgement with synthetic labels overstates the
    evidence, and `clicks: 70` reads as seventy human decisions.

    Measured on the live database when this was written: matt had 70 clicks and
    `classifier_active: true`, of which 8 were `ui` and 3 `agent` -- eleven real
    ones. The other 59 were `implicit` (harvested explanation requests) and
    `simulated` (a chat model reacting as the persona). Two other personas were
    58% `bootstrap`, which `evaluate_user` excludes as tautological because the
    label is a linear threshold on the same embedding the classifier trains on.

    This function exists because "a digest built from an untrained ranker looks
    exactly like one built from a good one". A digest built from a
    SYNTHETICALLY-trained ranker looked exactly like both.
    """
    from attestation.rank import ranking_quality

    conn, user_id = _mixed_history(
        tmp_path,
        fake_embedder,
        ["ui", "ui", "agent", "simulated", "simulated", "implicit", "bootstrap", "bootstrap"],
    )

    quality = ranking_quality(conn, user_id)

    assert quality["clicks"] == 8
    assert quality["real_clicks"] == 3, "ui + agent are the only human judgements"
    assert quality["synthetic_clicks"] == 5
    # The per-source breakdown is carried by the caveat prose, not by a
    # `by_source` key: this dict ships inside every feed envelope's 2000-char
    # budget, and an envelope key duplicating the sentence beside it is the
    # cheapest thing to cut. See the size test below.
    assert "2 bootstrap" in quality["caveat"], quality["caveat"]


def test_a_mostly_synthetic_history_is_caveated_even_when_the_classifier_fires(
    tmp_path, fake_embedder
):
    """classifier_active=true with a synthetic majority is the case that
    produced no caveat at all: both classes are present, so the guard passes,
    and nothing said the classes came from a model rather than a reader."""
    from attestation.rank import ranking_quality

    conn, user_id = _mixed_history(tmp_path, fake_embedder, ["ui"] + ["simulated"] * 9)

    quality = ranking_quality(conn, user_id)

    assert quality["classifier_active"] is True
    assert "caveat" in quality, "a 9/10-synthetic history reported no caveat"
    assert "simulated" in quality["caveat"]


def test_bootstrap_labels_are_named_as_tautological(tmp_path, fake_embedder):
    """bootstrap is not merely synthetic, it is circular: the label is a linear
    threshold on the same vector the classifier then trains on. evaluate_user
    excludes it for that reason; a reader of the live ranking deserves the same
    warning, since nothing excludes it there."""
    from attestation.rank import ranking_quality

    conn, user_id = _mixed_history(tmp_path, fake_embedder, ["bootstrap"] * 6)

    caveat = ranking_quality(conn, user_id).get("caveat", "")
    assert "bootstrap" in caveat.lower(), caveat


def test_an_all_human_history_gets_no_provenance_caveat(tmp_path, fake_embedder):
    """The caveat must stay a real signal rather than decoration on every
    response."""
    from attestation.rank import ranking_quality

    conn, user_id = _mixed_history(tmp_path, fake_embedder, ["ui"] * 25)

    quality = ranking_quality(conn, user_id)
    assert quality["real_clicks"] == 25
    assert quality["synthetic_clicks"] == 0
    assert "caveat" not in quality, quality.get("caveat")


def test_ranking_quality_stays_small_enough_to_ship_in_every_envelope(tmp_path, fake_embedder):
    """This dict rides along on feed.list, feed.search and feed.digest, and
    those responses have a measured 2000-char budget (test_response_size.py):
    a 2B model that cannot hold the payload loops truncate-apologise-redump.

    The provenance keys were added to stop `clicks: 70` overstating the
    evidence, and adding them pushed feed.search to 2050 chars. The fix is not
    a bigger budget. Two things are dropped instead:

    - `by_source` never ships. Every count it holds is already spelled out in
      the caveat string, which a reader has to read anyway -- an envelope key
      duplicating prose is the cheapest thing to cut.
    - `real_clicks`/`synthetic_clicks` are omitted when there are no clicks at
      all, where "0 of 0 are real" is noise rather than an honesty note.

    What must NOT be dropped is the split on a history that has clicks: that
    is the whole point of the change.
    """
    import json

    from attestation.rank import create_user, ranking_quality

    conn, user_id = _mixed_history(tmp_path, fake_embedder, ["ui", "simulated"])
    quality = ranking_quality(conn, user_id)
    assert "by_source" not in quality, "the caveat already names every source count"
    assert quality["real_clicks"] == 1, "the split survives on a history with clicks"
    assert quality["synthetic_clicks"] == 1

    conn2 = get_db(tmp_path / "empty.db")
    empty_user = create_user(conn2, "nobody", "machine learning")
    empty = ranking_quality(conn2, empty_user)
    assert empty["clicks"] == 0
    assert "real_clicks" not in empty, "0 of 0 real clicks is noise, not honesty"
    assert "synthetic_clicks" not in empty
    assert len(json.dumps(empty)) <= 260, (
        f"{len(json.dumps(empty))} chars of quality metadata on a fresh user; "
        "this dict ships inside a 2000-char response budget"
    )


def test_the_provenance_caveat_is_bounded_regardless_of_source_mix(tmp_path, fake_embedder):
    """The caveat must not grow with the number of click sources.

    Three rounds running, a payload guard missed the worst case because its
    fixture was hand-built while the worst case is a property of the live data
    distribution. The root cause is here rather than in any fixture: the
    caveat joined EVERY source present, so its length was unbounded in
    `len(CLICK_SOURCES)` -- 116 chars at the two sources the guard used, 144 at
    the five a real persona can reach via bootstrap-persona. It rides on every
    ranked response, so the budget was 28 chars closer to breaching than any
    test showed.

    Bounding it here means no future fixture can be wrong about it.
    """
    from attestation.rank import CLICK_SOURCES, _provenance_caveat

    every_source = dict.fromkeys(CLICK_SOURCES, 999999)
    worst = _provenance_caveat(sum(every_source.values()), 0, every_source)

    assert len(worst) <= 160, f"{len(worst)} chars with every source present: {worst}"
    # Still says the two things that matter: how much is real, and that
    # bootstrap is circular.
    assert "0/" in worst
    assert "bootstrap" in worst

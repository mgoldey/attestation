"""Per-user ranking: profile cosine + click-trained classifier + feature-preference
term, blended by rank."""

import hashlib
import itertools
import logging
import sqlite3

import numpy as np
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from attestation.features import pref_scores_for_items

# SQLite's SQLITE_LIMIT_VARIABLE_NUMBER default: the max bind parameters in one
# statement. rank_items and search_feed can pass every item in the archive as
# `ids`, so IN (...) queries built from `ids` must chunk below this limit
# rather than build one query with len(ids) placeholders.
_SQL_VAR_CHUNK = 900

log = logging.getLogger(__name__)

# Module-level in-memory cache: (db file path, user_id) -> (interests_hash, profile_vector).
# Ranking never blocks on the embedder: a warm entry is served even if the
# embedder starts raising (e.g. Ollama down), and is only recomputed when the
# user's interests text actually changes. Keyed on the database's file path
# (not just user_id) so distinct databases -- e.g. separate test databases
# that reuse the same small integer user ids -- never collide with each
# other's cached vectors. The db path, not id(conn), is used because
# sqlite3.Connection object ids get reused once a prior connection is
# garbage-collected.
_PROFILE_VEC_CACHE: dict[tuple[str, int], tuple[str, np.ndarray]] = {}


def _db_identity(conn: sqlite3.Connection) -> str:
    return conn.execute("PRAGMA database_list").fetchone()["file"]


def forget_profile_vector(conn: sqlite3.Connection, user_id: int) -> None:
    """Drop this user's cached profile vector. Call after changing or deleting
    the persona.

    Necessary because of how the two cache paths interact. The hash check in
    _profile_vector normally makes eviction unnecessary -- changed interests
    text hashes differently and misses. But the embedder-down fallback
    deliberately returns a cached vector WITHOUT comparing hashes, since a
    stale vector beats a dead feed. Together: change the interests, lose the
    embedder, and the fallback serves a vector computed from text the user
    already replaced, silently.

    Evicting here means that case raises the honest cold-cache error instead.

    Exposed as a function so callers do not reconstruct the key by hand --
    mcp_server used to do exactly that, which is why update_persona was missed.
    """
    _PROFILE_VEC_CACHE.pop((_db_identity(conn), user_id), None)


class RankedItem(BaseModel):
    item_id: int
    title: str
    url: str | None
    source: str | None
    score: float  # blended rank; lower = better
    explanation: str | None = None
    tags: list[str] = []
    content_type: str | None = None
    summary: str | None = None


def get_user(conn: sqlite3.Connection, name: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM users WHERE name = ?", (name,)).fetchone()


def create_user(conn, name: str, interests: str) -> int:
    """Insert a persona. Ranking starts from the interests embedding alone."""
    if get_user(conn, name) is not None:
        raise ValueError(f"user already exists: {name!r}")
    cur = conn.execute("INSERT INTO users(name, interests) VALUES (?, ?)", (name, interests))
    conn.commit()
    return cur.lastrowid


CLICK_SOURCES = ("ui", "agent", "bootstrap")


def record_click(conn, user_id: int, item_id: int, useful: bool, source: str = "ui") -> None:
    """The single click write path. `source` records provenance (see CLICK_SOURCES).

    SQLite cannot express a CHECK constraint added via ALTER TABLE, so the enum
    is enforced here rather than in the schema.
    """
    if source not in CLICK_SOURCES:
        raise ValueError(f"invalid click source: {source!r} (expected one of {CLICK_SOURCES})")
    conn.execute(
        "INSERT OR REPLACE INTO clicks(user_id, item_id, useful, source) VALUES (?, ?, ?, ?)",
        (user_id, item_id, int(useful), source),
    )
    conn.commit()


def blend_weight(n_clicks: int) -> float:
    return n_clicks / (n_clicks + 5)


def ranks(scores: np.ndarray) -> np.ndarray:
    """Rank 0 = highest score."""
    order = np.argsort(-scores)
    out = np.empty(len(scores), dtype=np.int64)
    out[order] = np.arange(len(scores))
    return out


def avg_ranks(scores: np.ndarray) -> np.ndarray:
    """Rank 0 = highest score; tied scores share their mean rank (no tie-break noise)."""
    order = np.argsort(-scores)
    out = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        out[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return out


def _click_training_data(conn, user_id: int):
    rows = conn.execute(
        "SELECT c.useful, v.embedding FROM clicks c"
        " JOIN item_vectors v ON v.rowid = c.item_id WHERE c.user_id = ?",
        (user_id,),
    ).fetchall()
    if not rows:
        return None, None
    y = np.array([r["useful"] for r in rows])
    X = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    return X, y


def classifier_probs(conn, user_id: int, X: np.ndarray) -> np.ndarray | None:
    X_train, y = _click_training_data(conn, user_id)
    if y is None or len(set(y.tolist())) < 2:
        return None  # single-class guard: never let sklearn see one class
    clf = LogisticRegression(class_weight="balanced", C=0.1, max_iter=1000)
    clf.fit(X_train, y)
    return clf.predict_proba(X)[:, 1]


def _candidate_items(conn, user_id: int, since_days: int | None, *, exclude_clicked: bool = True):
    """Rankable items. Defaults reproduce feed behavior: recent and unclicked.

    search_feed passes since_days=None / exclude_clicked=False, since finding
    an older or already-rated item is a legitimate search result.
    """
    sql = (
        "SELECT i.id, i.title, i.url, i.summary, f.title AS source, v.embedding"
        " FROM items i JOIN item_vectors v ON v.rowid = i.id"
        " LEFT JOIN feeds f ON f.id = i.feed_id"
        " WHERE 1=1"
    )
    params: list = []
    if since_days is not None:
        sql += " AND i.published >= datetime('now', ?)"
        params.append(f"-{since_days} days")
    if exclude_clicked:
        sql += " AND i.id NOT IN (SELECT item_id FROM clicks WHERE user_id = ?)"
        params.append(user_id)
    return conn.execute(sql, params).fetchall()


def _profile_vector(conn, embedder, user_id: int, interests_text: str) -> np.ndarray:
    """Cached profile embedding: recompute only when interests text changes.

    On embedder failure, fall back to a cached vector for this user if one
    exists (even if stale, i.e. computed for older interests text). Raises
    only when there is no cached vector at all -- the truly-cold case.
    """
    cache_key = (_db_identity(conn), user_id)
    text_hash = hashlib.sha256(interests_text.encode()).hexdigest()
    cached = _PROFILE_VEC_CACHE.get(cache_key)
    if cached is not None and cached[0] == text_hash:
        return cached[1]

    try:
        vec = embedder.embed_query(interests_text)
    except Exception:  # noqa: BLE001 - any embedder failure must fall back to cache,
        # not propagate: ranking is on the request path and a broken LLM backend
        # must degrade rather than take the feed down.
        if cached is not None:
            log.warning(
                "embed_query failed for user_id=%s; serving stale cached profile vector",
                user_id,
            )
            return cached[1]
        raise RuntimeError(
            f"no cached profile vector for user_id={user_id} and embedder is unavailable"
        ) from None

    _PROFILE_VEC_CACHE[cache_key] = (text_hash, vec)
    return vec


def rank_items(
    conn, embedder, user_id: int, since_days: int | None = 14, *, exclude_clicked: bool = True
) -> list[RankedItem]:
    rows = _candidate_items(conn, user_id, since_days, exclude_clicked=exclude_clicked)
    if not rows:
        return []
    X = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    profile_vec = _profile_vector(conn, embedder, user_id, user["interests"] or user["name"])
    profile_rank = ranks(X @ profile_vec)

    probs = classifier_probs(conn, user_id, X)
    n_clicks = conn.execute(
        "SELECT COUNT(*) c FROM clicks WHERE user_id = ?", (user_id,)
    ).fetchone()["c"]

    click_ranks = []
    if probs is not None:
        click_ranks.append(ranks(probs))
    if n_clicks > 0:
        pref = pref_scores_for_items(conn, user_id, [r["id"] for r in rows])
        # Tie-averaged ranks: tied (mostly neutral-0.5) items share one rank value, so the
        # pref term adds no tie-break noise and an all-neutral array is a blend no-op.
        click_ranks.append(avg_ranks(pref))

    if not click_ranks:
        final = profile_rank.astype(np.float64)
    else:
        w = blend_weight(n_clicks)
        final = w * np.mean(click_ranks, axis=0) + (1 - w) * profile_rank

    ids = [r["id"] for r in rows]
    ctype: dict[int, str] = {}
    tags_by: dict[int, list[str]] = {}
    for chunk in itertools.batched(ids, _SQL_VAR_CHUNK):
        qmarks = ",".join("?" * len(chunk))
        for r in conn.execute(
            f"SELECT item_id, content_type FROM item_features WHERE item_id IN ({qmarks})",
            chunk,
        ):
            ctype[r["item_id"]] = r["content_type"]
        for r in conn.execute(
            f"SELECT item_id, tag FROM item_tags WHERE item_id IN ({qmarks}) ORDER BY tag",
            chunk,
        ):
            tags_by.setdefault(r["item_id"], []).append(r["tag"])

    order = np.argsort(final)
    return [
        RankedItem(
            item_id=rows[i]["id"],
            title=rows[i]["title"],
            url=rows[i]["url"],
            source=rows[i]["source"],
            score=float(final[i]),
            tags=tags_by.get(rows[i]["id"], []),
            content_type=ctype.get(rows[i]["id"]),
            summary=rows[i]["summary"],
        )
        for i in order
    ]


def bootstrap_persona(conn, embedder, user_name: str, k: int = 30) -> int:
    """Pseudo-clicks for a synthetic persona: top-k/2 by profile similarity -> useful,
    bottom-k/2 -> not useful. Optional demo garnish; persona switch works without it.

    DEMO FIXTURE ONLY -- NOT GROUND TRUTH. The useful/not-useful label is a
    deterministic linear threshold on the exact same embedding X that
    classifier_probs() trains on (`argsort(X @ embed_query(interests))`,
    top half vs. bottom half). A linear classifier fit on X to predict a
    linear threshold of X recovers it essentially perfectly, so any AUC
    computed over these rows is a tautology, not a measurement of ranking
    quality. evaluate_user() excludes source='bootstrap' clicks for exactly
    this reason -- never remove that filter to "get more eval data".
    """
    user = get_user(conn, user_name)
    if user is None:
        raise ValueError(f"unknown user: {user_name!r}")
    rows = conn.execute(
        "SELECT i.id, v.embedding FROM items i JOIN item_vectors v ON v.rowid = i.id"
    ).fetchall()
    if len(rows) < k:
        k = len(rows) - (len(rows) % 2)
    if k == 0:
        return 0
    X = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    sims = X @ embedder.embed_query(user["interests"])
    order = np.argsort(-sims)
    half = k // 2
    chosen = [(rows[i]["id"], 1) for i in order[:half]]
    chosen += [(rows[i]["id"], 0) for i in order[-half:]]
    for item_id, useful in chosen:
        conn.execute(
            "INSERT OR IGNORE INTO clicks(user_id, item_id, useful, source)"
            " VALUES (?, ?, ?, 'bootstrap')",
            (user["id"], item_id, useful),
        )
    conn.commit()
    return len(chosen)


def evaluate_user(conn, user_id: int, n_holdout: int = 5) -> float | None:
    """Stratified-holdout AUC over real (non-bootstrap) clicks only.

    Honest noise at small n -- never present as evidence.

    source='bootstrap' clicks are excluded: their useful/not-useful label is a
    deterministic linear threshold on the same embedding the classifier
    trains on (see bootstrap_persona docstring), so any AUC computed over
    them is a tautology rather than a measurement.

    The holdout fold is drawn with a fixed-seed stratified split rather than
    "last n_holdout rows in clicked_at order": clicked_at defaults to
    second-resolution datetime('now'), so rows inserted in one batch (as
    bootstrap_persona and any naturally label-sorted rating session do) tie
    and fall back to insertion order, making a trailing slice single-class by
    construction -- always returning None instead of an honest score.
    """
    rows = conn.execute(
        "SELECT c.useful, v.embedding FROM clicks c"
        " JOIN item_vectors v ON v.rowid = c.item_id"
        " WHERE c.user_id = ? AND c.source != 'bootstrap'"
        " ORDER BY c.clicked_at, c.id",
        (user_id,),
    ).fetchall()
    if len(rows) < n_holdout + 5:
        return None
    y = np.array([r["useful"] for r in rows])
    X = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    if len(set(y.tolist())) < 2:
        return None  # single-class overall: no split can produce a mixed test fold

    n_splits = max(2, len(rows) // n_holdout)
    n_splits = min(n_splits, np.bincount(y).min())  # each class needs >= n_splits members
    if n_splits < 2:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        y_train, y_test = y[train_idx], y[test_idx]
        if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
            continue
        clf = LogisticRegression(class_weight="balanced", C=0.1, max_iter=1000)
        clf.fit(X[train_idx], y_train)
        aucs.append(roc_auc_score(y_test, clf.predict_proba(X[test_idx])[:, 1]))
    if not aucs:
        return None
    return float(np.mean(aucs))

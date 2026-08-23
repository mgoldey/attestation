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


# Where a click came from, kept distinguishable forever because provenance
# decides what a row may be used for. `bootstrap` labels are a linear threshold
# on the embedding the classifier trains on, so evaluate_user excludes them --
# see bootstrap_persona. `simulated` rows are a chat model reacting to text as
# a persona would: independent of the embedding, which is what makes them
# trainable, but still not a person's judgement. `implicit` is inferred from
# engagement (asking why an item was ranked) rather than stated -- weak
# positive evidence only; nothing here ever infers a negative from silence.
CLICK_SOURCES = ("ui", "agent", "bootstrap", "simulated", "implicit")

# Which sources are a person actually deciding something. `ui` is a button
# press; `agent` is a verdict extracted from what the reader said. Everything
# else is inferred or generated, and a click count that pools them overstates
# how much the ranker was told -- see ranking_quality.
HUMAN_CLICK_SOURCES = frozenset({"ui", "agent"})


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


# How much feedback the preference term needs before it may rank anything.
#
# It fired at n_clicks > 0, and one click was enough to move items 190
# positions on the live corpus: a single positive on materials-scientist's own
# top item left 1 of the top 10 in place and pushed the rest to #187 and #196.
#
# The cause is an asymmetry in `_score`'s Laplace smoothing, (u+1)/(u+n+2).
# Downvotes work: one not-useful takes a key to 0.333, twenty take it to 0.045,
# so disliked keys separate cleanly from untouched ones at 0.5. Upvotes cannot:
# a key the reader has only ever liked lands at 0.667 or 0.955, and an
# untouched key sits at 0.500, so nothing is ever ranked BELOW neutral. The
# term stops measuring preference and starts measuring how many of an item's
# keys the reader has touched at all -- coverage, which tracks feed size and so
# ranks by whatever they subscribe to most.
#
# Hence the rule below is one-directional. A purely-downvoting reader has real
# signal and keeps it; a purely-upvoting one does not, and waits.
MIN_PREF_UPVOTE_CLICKS = 10


def _preference_ready(conn: sqlite3.Connection, user_id: int) -> bool:
    """Whether the feature-preference term has anything to say.

    Any not-useful click is enough, because a downvote genuinely separates a
    key from the untouched baseline. Upvotes alone need volume AND a downvote
    to compare against, since on their own they cannot rank anything below
    neutral.
    """
    row = conn.execute(
        "SELECT COUNT(*) n, SUM(useful) pos FROM clicks WHERE user_id = ?", (user_id,)
    ).fetchone()
    total, positive = row["n"], row["pos"] or 0
    negative = total - positive
    if negative:
        return True
    return total >= MIN_PREF_UPVOTE_CLICKS and positive < total


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
    if _preference_ready(conn, user_id):
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


def ranking_quality(conn, user_id: int) -> dict:
    """How much to trust the ordering, stated up front.

    A digest built from an untrained ranker looks exactly like one built from a
    good one. rank.classifier_probs returns None when a user's clicks are all
    one class (rank.py's single-class guard), so the click-CLASSIFIER term
    never fires -- but rank_items blends in a second, independent term
    (avg_ranks over pref_scores_for_items) whenever n_clicks > 0, regardless of
    the guard. So a single-class history with at least one click is NOT pure
    embedding similarity: the feature-preference term still contributes, only
    the classifier is silent. Naming which terms are actually contributing
    matters more than a blanket "profile-embedding only" claim, which is wrong
    in exactly the case this caveat exists to describe.
    """
    rows = conn.execute(
        "SELECT useful, source, COUNT(*) n FROM clicks WHERE user_id = ? GROUP BY useful, source",
        (user_id,),
    ).fetchall()
    counts: dict[int, int] = {}
    by_source: dict[str, int] = {}
    for row in rows:
        counts[int(row["useful"])] = counts.get(int(row["useful"]), 0) + row["n"]
        by_source[row["source"]] = by_source.get(row["source"], 0) + row["n"]

    total = sum(counts.values())
    active = len(counts) > 1
    real = sum(n for src, n in by_source.items() if src in HUMAN_CLICK_SOURCES)
    out = {
        "clicks": total,
        "useful": counts.get(1, 0),
        "not_useful": counts.get(0, 0),
        "classifier_active": active,
    }
    # A bare count reads as that many human decisions. Measured on the live
    # database: one persona showed `clicks: 70, classifier_active: true` while
    # only 11 were a person deciding anything -- the rest harvested or
    # generated. Two others were 58% bootstrap.
    #
    # This dict rides on every feed envelope, inside a measured 2000-char
    # budget (test_response_size.py), so it pays for its own weight twice over:
    # the per-source breakdown is NOT a key here, because _provenance_caveat
    # already spells out every count in the prose a reader has to read anyway.
    # On a user with no clicks the split says nothing either, so it is omitted
    # rather than shipped as "0 of 0".
    if total:
        out["real_clicks"] = real
        out["synthetic_clicks"] = total - real
    if not active:
        if total > 0:
            out["caveat"] = (
                f"ranking is running WITHOUT its click classifier: {total} click(s), "
                f"all {'useful' if counts.get(1) else 'not-useful'}. Order blends "
                "profile-embedding similarity with a feature-preference term learned "
                "from those clicks -- the classifier term is silent (needs both "
                "useful and not-useful clicks to fire), but the preference term is "
                "still contributing. Mark some items the other way to train the "
                "classifier too."
            )
        else:
            out["caveat"] = (
                "ranking is running WITHOUT its click classifier or any "
                "feature-preference signal: 0 clicks recorded. Order is "
                "profile-embedding similarity only."
            )
    elif total < 20:
        out["caveat"] = f"only {total} clicks: the classifier is active but weakly trained"

    # Provenance caveat, appended to whatever the training-strength caveat said.
    # It is a separate concern: a history can be large, both-class and
    # well-trained, and still be almost entirely machine-generated.
    provenance = _provenance_caveat(total, real, by_source)
    if provenance:
        out["caveat"] = f"{out['caveat']} {provenance}" if out.get("caveat") else provenance
    return out


def _provenance_caveat(total: int, real: int, by_source: dict[str, int]) -> str:
    """What to say when the training signal is mostly not a person.

    `bootstrap` is called out separately because it is not merely synthetic, it
    is circular: `bootstrap_persona` labels by a linear threshold on the very
    embedding `classifier_probs` then trains on. `evaluate_user` excludes it for
    that reason and its docstring says never to remove that filter to get more
    eval data -- but nothing excludes it from the LIVE ranking, so a persona
    seeded for a demo keeps being ranked by a classifier fit on its own input.
    """
    if not total or real == total:
        return ""

    parts = ", ".join(f"{n} {src}" for src, n in sorted(by_source.items(), key=lambda kv: -kv[1]))
    caveat = (
        f"only {real} of {total} clicks are a person deciding something"
        f" (by source: {parts}); the rest are inferred or generated,"
        " so the ranker knows less than the count suggests."
    )
    if by_source.get("bootstrap"):
        caveat += (
            f" {by_source['bootstrap']} are bootstrap labels, which are a threshold on the"
            " same embedding the classifier trains on -- circular, and excluded from"
            " `attest eval` for that reason."
        )
    return caveat


STARTER_INTERESTS = "general science and technology research"


def autocreate_user(conn: sqlite3.Connection, name: str):
    """Create a reader on first sight, seeded from what the corpus covers.

    Refusing an unknown name and listing the valid ones taught agents to call
    persona_create with whatever string they had: the live database grew a
    duplicate persona with zero clicks that way, days after that reader had
    been merged away. The refusal did not prevent the duplicate, it caused it.
    """
    from attestation.features import tag_vocabulary

    try:
        topics = tag_vocabulary(conn, limit=6)
    except sqlite3.Error:
        # A database without the tag tables yet is not a reason to refuse a
        # reader; the starter string is a placeholder either way. Narrow to
        # sqlite errors on purpose -- anything else here is a real bug and
        # should reach the decorator's own handler.
        topics = []
    interests = ", ".join(topics) if topics else STARTER_INTERESTS
    create_user(conn, name, interests)
    conn.commit()
    return get_user(conn, name), interests

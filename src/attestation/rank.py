"""Per-user ranking: profile cosine + click-trained classifier + feature-preference
term, blended by rank."""

import hashlib
import itertools
import logging
import sqlite3
from collections.abc import Sequence

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
    # Cosine of the item against the reader's profile vector: ABSOLUTE, unlike
    # `score`, which is a rank within whatever candidate set was ranked. Search
    # needs an absolute profile term because it ranks a few hundred candidates
    # rather than the archive, and a rank-percentile silently renumbers when
    # the set narrows -- an item at archive position 5227 becomes position 6.
    profile_similarity: float = 0.0
    explanation: str | None = None
    tags: list[str] = []
    content_type: str | None = None
    summary: str | None = None


def get_user(conn: sqlite3.Connection, name: str) -> sqlite3.Row | None:
    """A persona by name, matched without regard to case.

    Case-sensitive lookup turned one shift key into a second account:
    `feed.list(user="Matt")` missed `matt`, autocreate made a fresh persona,
    and 70 clicks were discarded while the reader was greeted as new. That is
    the duplicate-persona failure CLAUDE.md records -- autocreate removed the
    refusal that caused it and left the key's case scope untouched.

    `create_user` checks through here, so creation inherits the same folding
    and a second spelling is refused rather than shadowing the first. The
    stored spelling is whatever was passed: preserved on write, folded on read.
    """
    return conn.execute("SELECT * FROM users WHERE name = ? COLLATE NOCASE", (name,)).fetchone()


def create_user(conn, name: str, interests: str) -> int:
    """Insert a persona. Ranking starts from the interests embedding alone.

    The INSERT is authoritative, not the preceding read. Check-then-insert has
    no transaction around the pair, so concurrent first sight of one reader had
    15 of 16 callers raise -- and that escapes /list as a 500 on the FIRST page
    load for a new reader, which is what autocreate exists to serve. The
    database was never wrong (UNIQUE held, one row); only the losers were told
    something false.

    Losing the race is not an error: the persona the caller asked for exists.
    A caller that genuinely needs "did I create this" can compare the returned
    id, and `feed.persona_create` still refuses a name that already existed
    when it looked -- that refusal is a UX decision, made above this line.
    """
    # An empty or whitespace name is not a persona. personas.py's own comment
    # says "a duplicate or empty name is the caller's to fix", and only the
    # duplicate half was enforced -- by the UNIQUE constraint, not by code. A
    # persona named '   ' persists and then appears in every "Valid users:"
    # list, corrupting the one message the rest of this surface relies on to
    # be actionable.
    if not name.strip():
        raise ValueError("persona name must not be empty")
    existing = get_user(conn, name)
    if existing is not None:
        raise ValueError(f"user already exists: {name!r}")
    try:
        cur = conn.execute("INSERT INTO users(name, interests) VALUES (?, ?)", (name, interests))
        conn.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        # Someone inserted it between the read and the write. The row exists,
        # which is all the caller wanted.
        conn.rollback()
        raced = get_user(conn, name)
        if raced is None:
            raise
        return raced["id"]


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

# Below this many clicks in the SMALLER class, a both-class history is active
# in name only -- `classifier_active` tests `len(counts) > 1`, which reads as
# "trained" and means "both labels appear at least once". Five is where a
# StratifiedKFold can still put a minority member in more than one fold, which
# is the same threshold evaluate_user's n_splits guard already enforces.
MIN_MINORITY_CLICKS = 5


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

    A MIRROR GUARD WAS TRIED AND REVERTED. A review found that with 2 positives
    against 20 negatives no key clears neutral, so the term scores a perfect
    in-sample AUC while ranking nothing -- the ordering it contributes there is
    by how UNFAMILIAR an item is. That diagnosis is correct. But requiring a
    minimum positive count made the outcome that matters slightly WORSE:
    top-20 relevance against stated interests fell from 64/100 to 62/100
    across the five live personas, because unfamiliarity is a weak signal and
    not a wrong one when nothing better is available. Do not re-add the guard
    without a measurement that improves, not merely a story that convinces.
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


# How many items search ranks. The old path ranked the whole archive and
# filtered afterwards, so this number did not exist and the work grew without
# bound. 300 keeps every measured query's result set identical to that path's
# while bounding the cost: on the live corpus the deepest hit any of the eight
# probe queries actually returned sat at candidate 41.
SEARCH_CANDIDATES = 300


def literal_candidates(conn, query: str, limit: int) -> set[int]:
    """Item ids whose title or summary contains the query as a substring.

    Search blends a literal match with a semantic one, and a literal hit is not
    always a semantic hit -- "CRISPR" in a title the embedding places elsewhere
    is still the item the reader asked for. The old path got these free by
    ranking every row; restricting to sqlite-vec's top-k would have silently
    dropped them, which is a behaviour change rather than an optimisation.
    """
    needle = query.lower().strip()
    if not needle:
        return set()
    rows = conn.execute(
        "SELECT id FROM items"
        " WHERE lower(title) LIKE ? OR lower(summary) LIKE ?"
        " ORDER BY published DESC LIMIT ?",
        (f"%{needle}%", f"%{needle}%", limit),
    )
    return {r["id"] for r in rows}


def _candidate_items(
    conn,
    user_id: int,
    since_days: int | None,
    *,
    exclude_clicked: bool = True,
    only_ids: Sequence[int] | None = None,
):
    """Rankable items. Defaults reproduce feed behavior: recent and unclicked.

    search_feed passes since_days=None / exclude_clicked=False, since finding
    an older or already-rated item is a legitimate search result.

    `only_ids` restricts to a caller-supplied set. That combination -- no
    window, nothing excluded -- means no WHERE clause at all, so search read
    every row of items JOIN item_vectors, stacked it into one array, and built
    a RankedItem for each, in order to return four. Measured by growing a copy
    of the live database with its own rows: 5243 items 0.25s/161MB, 60k
    1.94s/421MB, 150k 4.93s/849MB. sqlite-vec had already found the answer in
    87ms; the archive scan was the caller ranking everything and filtering
    afterwards.
    """
    sql = (
        "SELECT i.id, i.title, i.url, i.summary, f.title AS source, v.embedding"
        " FROM items i JOIN item_vectors v ON v.rowid = i.id"
        " LEFT JOIN feeds f ON f.id = i.feed_id"
        " WHERE 1=1"
    )
    params: list = []
    if only_ids is not None:
        ids = list(only_ids)
        if not ids:
            return []
        sql += f" AND i.id IN ({','.join('?' * len(ids))})"
        params.extend(ids)
    if since_days is not None:
        # replace(published,'T',' '), not a bare comparison. `published` is
        # stored ISO-8601 with a T separator -- all 5222 items in the live
        # corpus -- while datetime() renders a space, and 'T' (0x54) sorts
        # after ' ' (0x20). So "2026-08-19T00:00:00" compares as NEWER than
        # "2026-08-19 10:29:46" despite being ten hours older, which is
        # exactly how arXiv stamps its items: midnight on the day.
        #
        # Measured on the live database before this: a 12-day window returned
        # 3120 items where 2494 qualify, 626 of them silently too old.
        sql += " AND replace(i.published, 'T', ' ') >= datetime('now', ?)"
        params.append(f"-{since_days} days")
    if exclude_clicked:
        sql += " AND i.id NOT IN (SELECT item_id FROM clicks WHERE user_id = ?)"
        params.append(user_id)
    return conn.execute(sql, params).fetchall()


class EmbedderUnavailable(RuntimeError):
    """The embedder is down and there is no cached vector to fall back on.

    Its own type so the web server can tell this expected, cold-start
    condition apart from a bug: the profile-vector cache is in-process
    memory, so a freshly started `attest serve` is cold for every reader and
    this is the FIRST thing a new user hits when Ollama is not running yet.
    """


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
        raise EmbedderUnavailable(
            f"no cached profile vector for user_id={user_id} and embedder is unavailable"
        ) from None

    _PROFILE_VEC_CACHE[cache_key] = (text_hash, vec)
    return vec


def rank_items(
    conn,
    embedder,
    user_id: int,
    since_days: int | None = 14,
    *,
    exclude_clicked: bool = True,
    only_ids: Sequence[int] | None = None,
) -> list[RankedItem]:
    rows = _candidate_items(
        conn, user_id, since_days, exclude_clicked=exclude_clicked, only_ids=only_ids
    )
    if not rows:
        return []
    X = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    profile_vec = _profile_vector(conn, embedder, user_id, user["interests"] or user["name"])
    profile_sims = X @ profile_vec
    profile_rank = ranks(profile_sims)

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
            profile_similarity=float(profile_sims[i]),
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


def _provenance_auc(rows, X, skf) -> float | None:
    """How well the same model predicts WHERE a label came from, not what it says.

    bootstrap is excluded from evaluation as tautological, and nothing checked
    whether the remaining sources are any better. Measured on the live
    database: of matt's 70 non-bootstrap clicks, all 35 implicit and all 8 ui
    are positive while 21 of 24 simulated are negative -- so the label is very
    nearly a restatement of the source. Fitting this same classifier to predict
    provenance scores 1.000 against 0.942 for usefulness, and the two label
    vectors agree on 94% of rows.

    The mechanism is item SELECTION, not the labels themselves. simulate.py is
    careful that a chat model reads text rather than the vector -- but harvested
    positives are things the ranker surfaced (median archive rank 88 of 5273)
    and simulated negatives are sampled round-robin across feeds to find things
    to reject (median 3426). Two populations drawn from different regions of
    the archive separate perfectly whatever the labels say.

    Returns None when there is only one source, which is the case where the
    question does not arise.
    """
    # HARVESTED from this reader (ui, agent, implicit) vs GENERATED for them.
    # Not HUMAN_CLICK_SOURCES, which is {ui, agent}: implicit rows come from
    # the reader's own actions and are selected the same way -- by what the
    # ranker put in front of them -- so they sit on the harvested side of the
    # split this is testing for.
    harvested = HUMAN_CLICK_SOURCES | {"implicit"}
    provenance = np.array([1 if r["source"] in harvested else 0 for r in rows])
    if len(set(provenance.tolist())) < 2:
        return None
    scores = []
    for train_idx, test_idx in skf.split(X, provenance):
        if len(set(provenance[train_idx].tolist())) < 2:
            continue
        if len(set(provenance[test_idx].tolist())) < 2:
            continue
        clf = LogisticRegression(class_weight="balanced", C=0.1, max_iter=1000)
        clf.fit(X[train_idx], provenance[train_idx])
        scores.append(roc_auc_score(provenance[test_idx], clf.predict_proba(X[test_idx])[:, 1]))
    return float(np.mean(scores)) if scores else None


def evaluate_user(conn, user_id: int, n_holdout: int = 5) -> dict | None:
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
        "SELECT c.useful, c.source, v.embedding FROM clicks c"
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
    # A dict, not a bare float. The number covers the CLICK CLASSIFIER only --
    # this fits LogisticRegression on click embeddings and never touches the
    # profile-similarity term or the feature-preference term, which together
    # are most of what rank_items orders by. Measured: replacing a persona's
    # interests with unrelated text changed its top five to 1-of-5 overlap and
    # left this AUC bit-identical, so a reader taking it for "the ranking is
    # good" is taking it for something it cannot see.
    return {
        "auc": float(np.mean(aucs)),
        "n_clicks": len(rows),
        "n_splits": len(aucs),
        "provenance_auc": _provenance_auc(rows, X, skf),
        "measures": (
            "the click classifier alone, on stored embeddings."
            " NOT the profile-similarity or feature-preference terms, which"
            " also order the feed -- changing a persona's interests does not"
            " move this number"
        ),
    }


def _blend_disclosure(total: int, real: int) -> dict:
    """How much of the ORDERING the click terms won, and on what.

    The prose caveat says the training is synthetic; it did not say synthetic
    labels decide the order almost outright. Measured: four of five personas
    have zero human clicks and still hand 91% of their ordering to click terms
    -- one of them on 30 bootstrap rows, whose labels are a threshold on the
    very embedding being ranked.

    Reported rather than changed. The weight is what it is, and the fix for an
    undisclosed quantity is to disclose it.
    """
    out = {"blend_weight": round(blend_weight(total), 2)}
    if real != total:
        out["blend_weight_if_human_only"] = round(blend_weight(real), 2)
    return out


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
        # How much of the ORDERING the click terms won, and how much they would
        # have won on human clicks alone. The prose caveat says the training is
        # synthetic; it did not say synthetic labels decide the order almost
        # outright. Measured: four of five personas have zero human clicks and
        # still hand 91% of their ordering to click terms -- one of them on 30
        # bootstrap rows, whose labels are a threshold on the very embedding
        # being ranked. Reported rather than changed: the weight is what it is,
        # and a reader can now see it.
        out.update(_blend_disclosure(total, real))
    if not active:
        if total > 0:
            # Terse: this rides on every ranked response. It was 363 chars of
            # explanation, which together with the provenance line pushed
            # feed.list to 2098 against a 2000 budget. State the fact and the
            # fix; the mechanism is in this function's docstring.
            out["caveat"] = (
                f"classifier OFF: all {total} clicks are"
                f" {'useful' if counts.get(1) else 'not-useful'}, so only the"
                " profile embedding and a preference term rank this."
                " Mark some items the other way to train it."
            )
        else:
            out["caveat"] = (
                "classifier OFF: 0 clicks recorded, so this is profile-embedding similarity only."
            )
    elif total < 20:
        out["caveat"] = f"only {total} clicks: the classifier is active but weakly trained"
    elif min(counts.values()) < MIN_MINORITY_CLICKS:
        # Both classes present is not the same as trained on both. A history of
        # 199 useful and 1 not-useful satisfied every branch above -- active,
        # over 20 clicks, entirely human -- and shipped with NO caveat, while
        # `attest eval` on the same data printed "insufficient click data for a
        # meaningful holdout" and exited 1. Two surfaces flatly contradicting
        # each other. Measured: that classifier's probabilities over 1000 items
        # span 0.31-0.80 with std 0.057, which is a near-constant.
        #
        # This is on the path engagement harvesting creates: reads become
        # implicit positives, so a reader who reads a lot and rejects once
        # lands here.
        fewer = "not-useful" if counts.get(0, 0) < counts.get(1, 0) else "useful"
        out["caveat"] = (
            f"only {min(counts.values())} {fewer} click(s) of {total}: the"
            " classifier is active but has almost nothing to contrast against."
        )

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

    # Terse on purpose. This string rides on EVERY ranked response, and the
    # first version cost 463 chars in its bootstrap branch -- which pushed
    # feed.list to 2316 chars against a 2000 budget for a bootstrap-heavy
    # persona, reintroducing the payload failure this project already fixed
    # once. The facts are what a caller needs; the essay explaining why
    # bootstrap labels are circular belongs in the docstring, not in the
    # response, because the response repeats it forever.
    # Only the two largest sources are named. Joining every source made the
    # string grow with len(CLICK_SOURCES) -- 116 chars at two sources, 144 at
    # five, 168 at the theoretical worst -- on a string that rides on every
    # ranked response. Three separate payload-budget failures traced back to a
    # fixture that happened to use fewer sources than a real persona.
    #
    # The proportion is what a caller acts on; the full breakdown is available
    # from the clicks table, and bootstrap is called out below regardless of
    # rank because it is circular rather than merely synthetic.
    ranked = sorted(by_source.items(), key=lambda kv: (-kv[1], kv[0]))
    parts = "/".join(f"{n} {src}" for src, n in ranked[:2])
    if len(ranked) > 2:
        parts += f"/+{len(ranked) - 2} more"

    # Zero human clicks is a different claim from "mostly synthetic", and the
    # difference is measurable: deleting the simulated clicks improved or
    # matched top-10 relevance for all five live personas and hurt none --
    # structural-biologist went 2/5 to 5/5 in the top five. So a wholly
    # synthetic history is not a weakly-trained ranker, it is an untrained one
    # wearing `classifier_active: true`, and a reader deciding whether to trust
    # the order needs that conclusion rather than the arithmetic behind it.
    if real == 0:
        caveat = (
            f"NOT judged by a person: all {total} clicks are generated ({parts});"
            " untrained in practice"
        )
    else:
        caveat = f"only {real}/{total} clicks are human ({parts})"
    # Skipped when the message already says the order was not judged by a
    # person: "untrained in practice" subsumes "these particular labels are
    # circular", and both together exceed the 160-char bound.
    if by_source.get("bootstrap") and real:
        # Named, not explained: bootstrap is circular rather than merely
        # synthetic, and a caller that wants the reason can read the docstring.
        caveat += "; bootstrap labels are circular (excluded from `attest eval`)"
    return caveat + "."


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
    try:
        create_user(conn, name, interests)
    except ValueError:
        # Lost the race for a name nobody had a moment ago. create_user's own
        # IntegrityError handler covers only the window between ITS pre-check
        # and ITS insert -- microseconds. The wider and far likelier window is
        # two autocreates both passing that pre-check, one inserting, and the
        # other's pre-check then finding the row: that path raises before any
        # INSERT is attempted, so the handler below it never runs.
        #
        # Measured: idle, 0/30 trials raised; with the box loaded to nproc,
        # 7 of 8 runs raised. The test written for this in round 17 therefore
        # passed here and failed on a fresh clone, which is how it survived
        # 21 review rounds.
        #
        # Refusing here would be wrong on its own terms: autocreate exists to
        # make the reader exist, and it does exist. `feed.persona_create` keeps
        # its refusal -- that is a UX decision made above this line.
        existing = get_user(conn, name)
        if existing is None:
            raise
        return existing, existing["interests"]
    conn.commit()
    return get_user(conn, name), interests

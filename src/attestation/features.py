"""Feature extraction: LLM tagging pass + per-key click-preference scores.

Reliability contract mirrors explain.py: tagging is lazy relative to ingest,
validated, and skips (never blocks) on failure. Nothing in this module runs
on the rank path except pure-SQL/numpy preference scoring.
"""

import itertools
import logging
import sqlite3
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)

CONTENT_TYPES = ("paper", "survey", "announcement", "release", "blog", "other")
TAG_PATTERN = r"^[a-z0-9][a-z0-9-]{0,31}$"

# SQLite's SQLITE_LIMIT_VARIABLE_NUMBER default: the max bind parameters in one
# statement. item_ids here can be every item in the archive (via rank_items'
# unconditional call chain when exclude_clicked=False), so IN (...) queries
# must chunk below this limit rather than build one query per item.
_SQL_VAR_CHUNK = 900

# Tags that describe an item's provenance or its post type rather than its
# subject. Both axes are already recorded structurally -- the publication in
# items.feed_id, the post type in item_features.content_type -- so as tags they
# only add edges between items that share a source, not a topic. On the live
# graph these pulled `nature`, `science-feed` and `retraction` into the
# `biology` cluster, making a feed look like a research area. The prompt asks
# the model to avoid them; this enforces it, because a prompt is advisory.
NON_TOPIC_TAGS = frozenset(
    {
        "announcement",
        "arxiv",
        "blog",
        "conference",
        "journal",
        "nature",
        "news",
        "newsletter",
        "preprint",
        "press-release",
        "release",
        "report",
        "retraction",
        "science",
        "science-feed",
        "update",
    }
)


class ItemTags(BaseModel):
    content_type: Literal["paper", "survey", "announcement", "release", "blog", "other"]
    tags: list[str] = Field(min_length=1, max_length=4)

    @field_validator("tags")
    @classmethod
    def _tags_shape(cls, v: list[str]) -> list[str]:
        """Normalize what is recoverable, drop what is not, keep the item.

        TAG_PATTERN is lowercase-only, so every capitalised acronym the
        literature actually uses -- rRNA, DNA, CRISPR, RNA-seq, GPT-4 -- failed
        it. Rejecting the whole item for one such tag meant the item never
        tagged at all and was retried on every cron run forever, discarding the
        two or three perfectly good tags alongside it. One Nature paper sat
        untagged through a full re-tag for exactly this reason: the model
        returned ['biology', 'rRNA', 'exosome', 'nucleolar'] on three
        consecutive attempts and lost all four each time.

        Case and surrounding whitespace carry no meaning here (the graph
        lowercases everything anyway), so folding them is lossless. A tag that
        is still malformed after folding -- truncated, punctuated, too long --
        is genuinely unusable and is dropped on its own. The item survives if
        at least one tag does; only an item with nothing usable fails.
        """
        import re

        well_formed: list[str] = []
        for t in v:
            candidate = t.strip().lower().replace(" ", "-")
            if re.match(TAG_PATTERN, candidate) and candidate not in well_formed:
                well_formed.append(candidate)
        if not well_formed:
            raise ValueError(f"no usable tags in {v!r}")

        # Filtering to empty is NOT a failure: a release note really can be
        # about nothing but its own release. Failing here would leave the item
        # untagged and retried on every cron run forever -- the same trap the
        # all-or-nothing validator fell into. Keep the item with no topic tags;
        # content_type still records what it is, and a tagless item simply
        # contributes nothing to the graph, which is the correct outcome.
        return [t for t in well_formed if t not in NON_TOPIC_TAGS]


def tag_vocabulary(conn: sqlite3.Connection, limit: int = 150) -> list[str]:
    """Most-used tags, canonicalized, to steer the model toward existing vocabulary.

    Counts are summed over `kg.canonical` before ranking, which fixes a
    feedback loop: the raw table ranks each spelling separately, so the
    vocabulary shown to the model listed `machine-learning` (872) AND
    `machinelearning` (463), and `llm` (642) beside `language-models` (212),
    spending three of forty slots re-teaching the model the very variants the
    graph then merges away. Worse, the spellings it taught were the ones
    `canonical()` rewrites, so the model was being steered toward deprecated
    forms. Merging first frees those slots and lets real concepts
    (hugging-face, natural-language-processing, continual-learning) into the
    list instead.

    `limit` is 150 rather than 40 because 40 covered only 59% of tag
    assignments on the live corpus against 67% at 150 -- a model shown 40 tags
    for a 5000-item archive meets an unfamiliar subject constantly and mints a
    new tag when it does. 150 canonical tags is ~1.4KB of prompt, which is
    affordable next to the item summary it accompanies.

    Excludes NON_TOPIC_TAGS: the prompt asks the model not to emit them, so
    suggesting them here would work against it.
    """
    from attestation.kg import canonical

    totals: dict[str, int] = {}
    for row in conn.execute("SELECT tag, COUNT(*) n FROM item_tags GROUP BY tag"):
        name = canonical(row["tag"])
        if name in NON_TOPIC_TAGS:
            continue
        totals[name] = totals.get(name, 0) + row["n"]
    ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    return [tag for tag, _ in ranked[: max(0, int(limit))]]


def _tag_prompt(item: sqlite3.Row, vocab: list[str]) -> list[dict]:
    vocab_line = ", ".join(vocab) if vocab else "(none yet)"
    return [
        {
            "role": "system",
            "content": (
                "You label science-feed items. Reply with JSON: content_type"
                " (one of paper, survey, announcement, release, blog, other)"
                " and tags: 1-4 short lowercase-hyphenated topic tags."
                " Strongly prefer tags from the existing vocabulary; invent a"
                " new tag only if nothing in it fits."
                " Tag the SUBJECT MATTER only. Never tag where the item came"
                " from (nature, arxiv, science-feed) or what kind of post it"
                " is (release, announcement, newsletter) -- the publication is"
                " already recorded, and the kind of post is content_type."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Existing vocabulary: {vocab_line}\n\n"
                f"Title: {item['title']}\nSummary: {item['summary'][:1000]}"
            ),
        },
    ]


def tag_one_item(conn, item: sqlite3.Row, chat_fn, vocab: list[str], model_name: str) -> bool:
    """One LLM call (plus one retry) -> item_features + item_tags rows.
    False = skipped."""
    parsed = None
    for _ in range(2):  # one retry, per spec
        try:
            out = chat_fn(_tag_prompt(item, vocab), ItemTags.model_json_schema())
            parsed = ItemTags.model_validate(out)
            break
        except Exception:
            log.debug("tagging attempt failed for item %s", item["id"], exc_info=True)
    if parsed is None:
        return False
    conn.execute(
        "INSERT OR REPLACE INTO item_features(item_id, content_type, model) VALUES (?, ?, ?)",
        (item["id"], parsed.content_type, model_name),
    )
    conn.execute("DELETE FROM item_tags WHERE item_id = ?", (item["id"],))
    conn.executemany(
        "INSERT INTO item_tags(item_id, tag) VALUES (?, ?)",
        [(item["id"], t) for t in dict.fromkeys(parsed.tags)],
    )
    conn.commit()
    return True


def run_tagging(conn, chat_fn=None, limit: int | None = None) -> dict:
    """Tag every untagged item, newest first. Returns {"tagged": n, "failed": n}.

    Failed items are skipped (stay untagged) and retried on the next run.
    """
    if chat_fn is None:
        from attestation.llm import default_chat_fn

        chat_fn = default_chat_fn
    from tqdm import tqdm  # lazy: keeps the rank path free of tagging-only imports

    from attestation.llm import chat_model

    sql = (
        "SELECT i.id, i.title, i.summary FROM items i"
        " LEFT JOIN item_features f ON f.item_id = i.id"
        " WHERE f.item_id IS NULL ORDER BY i.published DESC"
    )
    params: tuple = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    rows = conn.execute(sql, params).fetchall()

    # Resolve the model ONCE and report it. Reading it per item let a single
    # run straddle two models if the environment changed underneath, and
    # returning it makes the tagging model visible to callers -- item_features
    # records a model per row but nothing ever surfaced it, so a run against
    # the wrong model (e.g. a standalone script that never called load_env and
    # silently got DEFAULT_CHAT_MODEL) looked identical to a correct one.
    model_name = chat_model()
    stats = {"tagged": 0, "failed": 0, "model": model_name}
    # disable=None -> bar only on a TTY; cron/pytest logs stay clean
    with tqdm(rows, desc="tagging", unit="item", disable=None) as bar:
        for item in bar:
            vocab = tag_vocabulary(conn)  # re-read so tags minted this run join the vocabulary
            if tag_one_item(conn, item, chat_fn, vocab, model_name):
                stats["tagged"] += 1
            else:
                stats["failed"] += 1
            bar.set_postfix(tagged=stats["tagged"], failed=stats["failed"], refresh=False)
    return stats


def _key_stats(conn, user_id: int) -> dict[str, list[int]]:
    """Per-feature-key [useful_count, not_useful_count] over this user's clicks."""
    rows = conn.execute(
        "SELECT 'tag:' || t.tag AS key, c.useful FROM clicks c"
        " JOIN item_tags t ON t.item_id = c.item_id WHERE c.user_id = ?"
        " UNION ALL"
        " SELECT 'type:' || f.content_type AS key, c.useful FROM clicks c"
        " JOIN item_features f ON f.item_id = c.item_id WHERE c.user_id = ?"
        " UNION ALL"
        " SELECT 'source:' || i.feed_id AS key, c.useful FROM clicks c"
        " JOIN items i ON i.id = c.item_id WHERE c.user_id = ? AND i.feed_id IS NOT NULL",
        (user_id, user_id, user_id),
    ).fetchall()
    stats: dict[str, list[int]] = {}
    for r in rows:
        u_n = stats.setdefault(r["key"], [0, 0])
        u_n[0 if r["useful"] else 1] += 1
    return stats


def _score(stats: dict[str, list[int]], key: str) -> float:
    u, n = stats.get(key, (0, 0))
    return (u + 1) / (u + n + 2)  # Laplace-smoothed; 0.5 = neutral


def _item_keys(conn, item_ids: list[int]) -> dict[int, list[str]]:
    keys: dict[int, list[str]] = {i: [] for i in item_ids}
    # Chunk below SQLite's SQLITE_LIMIT_VARIABLE_NUMBER (32766): item_ids can be
    # every item in the archive (via pref_scores_for_items <- rank_items).
    for chunk in itertools.batched(item_ids, _SQL_VAR_CHUNK):
        qmarks = ",".join("?" * len(chunk))
        for r in conn.execute(
            f"SELECT item_id, 'tag:' || tag AS key FROM item_tags WHERE item_id IN ({qmarks})",
            chunk,
        ):
            keys[r["item_id"]].append(r["key"])
        for r in conn.execute(
            f"SELECT item_id, 'type:' || content_type AS key FROM item_features"
            f" WHERE item_id IN ({qmarks})",
            chunk,
        ):
            keys[r["item_id"]].append(r["key"])
        for r in conn.execute(
            f"SELECT id, 'source:' || feed_id AS key FROM items"
            f" WHERE id IN ({qmarks}) AND feed_id IS NOT NULL",
            chunk,
        ):
            keys[r["id"]].append(r["key"])
    return keys


def pref_scores_for_items(conn, user_id: int, item_ids: list[int]) -> np.ndarray:
    """Mean per-key preference score per item, aligned with item_ids. 0.5 = neutral."""
    out = np.full(len(item_ids), 0.5, dtype=np.float64)
    if not item_ids:
        return out
    stats = _key_stats(conn, user_id)
    keys = _item_keys(conn, item_ids)
    for idx, iid in enumerate(item_ids):
        ks = keys[iid]
        if ks:
            out[idx] = float(np.mean([_score(stats, k) for k in ks]))
    return out

"""Tool responses must fit in a small model's working memory.

The failure that prompted this was watched in a Telegram chat: gemma4:e2b
called feed.list, could not reproduce the ten-item payload in prose, and fell
into a loop -- apologising, re-rendering as a bullet list, truncating,
apologising again, dumping raw JSON, truncating again. The model never
recovered and the reader saw half an item.

The fix belongs in the tool, not in the prompt. A response an agent cannot
hold is a response the tool should not have sent.

Two rules follow, and they are different:

- Say less per item. `score` was a blended rank carrying seventeen digits of
  precision that mean nothing across calls -- the same item scores 11.19 in a
  14-day window and 14.30 unbounded -- so it cost tokens and invited
  misreading. Tags beyond the first few add length without adding meaning.
- Say how much was left out. Truncation the caller cannot see is how an agent
  confidently reports a partial answer as a complete one.
"""

import json

import pytest

from attestation.db import get_db
from attestation.mcp import _shared
from attestation.mcp import feed as feed_mod

# The invariant is PER ITEM, not per response: a caller who explicitly asks
# for fifty items has asked for a big answer and should get one. What must
# stay small is the cost of each row, and the DEFAULT response an agent gets
# when it does not think about limits at all.
MAX_ITEM_CHARS = 300
MAX_DEFAULT_RESPONSE_CHARS = 2000


@pytest.fixture
def stocked(tmp_path, monkeypatch, fake_embedder):
    db = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(db))
    conn = get_db(db)
    conn.execute("INSERT INTO users(name, interests) VALUES ('ana', 'machine learning')")
    conn.execute("INSERT INTO feeds(url, title) VALUES ('http://x', 'arXiv cs.LG')")
    for i in range(1, 41):
        cur = conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash) VALUES (1, ?, ?, ?, ?)",
            (
                f"A Fairly Long Paper Title About Topic Number {i} And Its Consequences",
                f"https://arxiv.org/abs/2508.{i:05d}",
                "An abstract " * 40,
                f"h{i}",
            ),
        )
        for t in ("machine-learning", "evaluation-metrics", "agentic-workflows", "reasoning"):
            conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, ?)", (cur.lastrowid, t))
        conn.execute(
            "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)",
            (cur.lastrowid, fake_embedder.embed_document("t", "s").tobytes()),
        )
    conn.commit()
    conn.close()
    monkeypatch.setattr(_shared, "_embedder", fake_embedder)
    monkeypatch.setattr(_shared, "get_embedder", lambda: fake_embedder)
    return db


def test_the_default_feed_fits_in_a_small_context(stocked):
    """No limit argument at all -- what an agent sends when it is not
    thinking about payload size, which is most of the time."""
    out = feed_mod._list_feed("ana")
    size = len(json.dumps(out))
    assert size <= MAX_DEFAULT_RESPONSE_CHARS, (
        f"the default feed.list is {size} chars; a 2B model loses the thread "
        "rendering payloads this size"
    )


def test_each_item_stays_cheap(stocked):
    """The per-row cost, which is what makes a large explicit limit tolerable."""
    for item in feed_mod._list_feed("ana", limit=10)["items"]:
        size = len(json.dumps(item))
        assert size <= MAX_ITEM_CHARS, f"{size} chars for one item: {item}"


def test_search_is_bounded_too(stocked):
    out = feed_mod._search_feed("ana", "topic")
    assert len(json.dumps(out)) <= MAX_DEFAULT_RESPONSE_CHARS


def test_a_truncated_list_says_more_is_available(stocked):
    """Silent truncation is how an agent reports five of forty as if it were
    everything."""
    out = feed_mod._list_feed("ana", limit=3)
    assert "more available" in out["message"], out["message"]


def test_the_uninterpretable_score_is_gone(stocked):
    """`score` was a rank within a candidate set: the same item scored 11.19
    with a 14-day window and 14.30 unbounded. Nothing a caller can act on."""
    item = feed_mod._list_feed("ana", limit=3)["items"][0]
    assert "score" not in item


def test_every_item_still_carries_what_a_reader_needs(stocked):
    """Shrinking must not remove the point. A reader needs to know what it is,
    where it came from, and how to open it."""
    item = feed_mod._list_feed("ana", limit=3)["items"][0]
    assert item["title"]
    assert item["url"], "without a URL the reader cannot open it"
    assert item["source"], "without a source the reader cannot judge it"
    assert item["item_id"], "without an id the agent cannot rate it"


def test_tags_are_capped_and_the_cap_is_visible(stocked):
    """Four tags per item is length without meaning at ten items. Dropping
    them silently would hide that the item has more."""
    item = feed_mod._list_feed("ana", limit=3)["items"][0]
    assert len(item["tags"]) <= 3
    assert item.get("n_tags") == 4, "the true tag count must survive truncation"


def test_a_truncated_summary_says_so(stocked):
    """An agent quoting a silently-cut abstract misquotes the paper."""
    out = feed_mod._search_feed("ana", "topic", limit=2)
    for item in out["items"]:
        if "summary" in item and item["summary"]:
            assert item["summary"].endswith("…") or len(item["summary"]) < 300

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


def test_the_default_feed_fits_with_the_worst_case_caveat(stocked):
    """The budget must hold for a REAL persona, not just a pristine one.

    The fixture above has zero clicks, so `ranking_quality` emits its cheapest
    caveat (229 chars) and the guard reported ~126 chars of margin. The
    expensive branch -- a single-class history whose clicks are mostly
    bootstrap -- is 463 chars, a 234-char delta the test never exercised.
    Measured on the live database after that caveat shipped: bench-chemist
    2038 chars and ml-engineer 2083, both over a 2000 budget, with three more
    personas inside 130 chars of it.

    This is the exact failure the module docstring describes, reintroduced by
    a fix. The guard must exercise the most expensive ranking_quality a real
    user can produce.
    """
    conn = get_db(stocked)
    user_id = conn.execute("SELECT id FROM users WHERE name = 'ana'").fetchone()[0]
    # Bootstrap-heavy and single-class: the longest caveat the code can emit.
    for item_id in range(1, 31):
        conn.execute(
            "INSERT INTO clicks(user_id, item_id, useful, source) VALUES (?,?,1,?)",
            (user_id, item_id, "bootstrap" if item_id <= 20 else "simulated"),
        )
    conn.commit()
    conn.close()

    out = feed_mod._list_feed("ana")
    size = len(json.dumps(out))
    assert out["ranking_quality"].get("caveat"), "this fixture must produce a caveat"
    assert size <= MAX_DEFAULT_RESPONSE_CHARS, (
        f"the default feed.list is {size} chars for a bootstrap-heavy persona; "
        "the zero-click fixture hid this"
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


# --- reading one item -----------------------------------------------------


def test_an_item_can_be_read_in_full(stocked):
    """The gap that made an agent tell a reader to go open the link.

    Watched live: asked to summarise a paper, the model answered "I do not
    have a direct tool available to read and summarize the full abstract...
    I recommend visiting the link directly". It was right -- the compact item
    shape carries a title and a url and nothing to summarise, because trimming
    `summary` is what stopped ten-item payloads from truncating.

    The fix is not to put abstracts back in the list. It is one tool that
    returns ONE item in full, so the list stays cheap and reading stays
    possible.
    """
    from attestation.mcp import feed as f

    listed = f._list_feed("ana", limit=1)["items"][0]
    out = f._read_item("ana", listed["item_id"])

    assert out["ok"] is True
    assert out["item"]["summary"], "the abstract is in the database and must come back"
    assert out["item"]["title"] == listed["title"]
    assert out["item"]["url"] == listed["url"]


def test_reading_an_unknown_item_says_so(stocked):
    from attestation.mcp import feed as f

    out = f._read_item("ana", 999999)
    assert out["ok"] is False
    assert "999999" in out["message"]


def test_a_long_abstract_is_truncated_visibly(stocked):
    """One full item must still fit a small model. Silent truncation would
    have it quote half a sentence as though it were the whole finding."""
    from attestation.mcp import feed as f

    out = f._read_item("ana", 1)
    summary = out["item"]["summary"]
    assert len(summary) <= 2100
    if out["item"].get("truncated"):
        assert summary.endswith("…"), "truncation must be visible in the text"


def test_reading_stays_cheap_enough_to_render(stocked):
    """A single item, not a payload. The list is 5 items at ~1.5KB; one item
    read in full should not dwarf it."""
    import json

    from attestation.mcp import feed as f

    assert len(json.dumps(f._read_item("ana", 1))) <= 2600

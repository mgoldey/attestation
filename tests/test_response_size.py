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
# Both from measurement, not preference. 300 was a round number chosen against
# a fixture whose rows were 270 chars; real list rows reach 305 and real SEARCH
# rows 366, because a search row carries already_rated/match/relevance that a
# list row does not. 28 of 40 live rows violated the single 300 limit while
# every test passed -- a guard set below what production produces is a guard
# that lies. Two constants, because the two shapes are genuinely different.
MAX_ITEM_CHARS = 320
MAX_SEARCH_ITEM_CHARS = 380
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
                # 223 chars: the longest title in the live database, measured.
                # The fixture used 67 -- half the real p95 of 127 -- so every
                # budget here was checked against items cheaper than
                # production, and 5 of 15 real rows exceed MAX_ITEM_CHARS while
                # the guard stayed green. Same lesson as the caveat: the
                # fixture must be the worst realistic input, and the field it
                # measures must be bounded so no fixture can be wrong again.
                (
                    f"Retraction Note {i}: Fabrication of New Composite Materials for "
                    "Photocatalytic Degradation of Organic Pollutants Under Visible "
                    "Light Irradiation With Enhanced Stoichiometric Control"
                ),
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


def test_each_search_item_stays_cheap(stocked):
    """Search rows carry three fields a list row does not, so they get their
    own budget rather than being measured against the list one."""
    for item in feed_mod._search_feed("ana", "topic", limit=10)["items"]:
        size = len(json.dumps(item))
        assert size <= MAX_SEARCH_ITEM_CHARS, f"{size} chars for one search item: {item}"


def test_search_is_bounded_too(stocked):
    out = feed_mod._search_feed("ana", "topic")
    assert len(json.dumps(out)) <= MAX_DEFAULT_RESPONSE_CHARS


def test_search_is_bounded_with_the_worst_case_caveat(stocked):
    """Same trap as the feed guard: `stocked` has no clicks, so this measured
    ranking_quality's cheapest branch. Every budget test needs the most
    expensive realistic input, not a convenient one."""
    conn = get_db(stocked)
    user_id = conn.execute("SELECT id FROM users WHERE name = 'ana'").fetchone()[0]
    for item_id in range(1, 31):
        conn.execute(
            "INSERT INTO clicks(user_id, item_id, useful, source) VALUES (?,?,1,?)",
            (user_id, item_id, "bootstrap" if item_id <= 20 else "simulated"),
        )
    conn.commit()
    conn.close()

    out = feed_mod._search_feed("ana", "topic")
    assert out["ranking_quality"].get("caveat"), "this fixture must produce a caveat"
    size = len(json.dumps(out))
    assert size <= MAX_DEFAULT_RESPONSE_CHARS, f"search is {size} chars with a full caveat"


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


def test_no_field_in_a_row_is_unbounded(stocked):
    """Per-field bounds, so no fixture can be wrong about the row again.

    MAX_ITEM_CHARS was checked only against the fixture, and real rows exceeded
    it: 28 of 40 across the live personas, up to 305, driven by long tag names
    rather than titles. Rather than tune the fixture a fifth time, assert the
    property -- every variable-length field a row carries has a stated cap --
    so the row's worst case is a fact about the code.
    """
    from attestation.mcp.feed import MAX_TAGS_SHOWN, MAX_TITLE_CHARS, _clip_title

    # A title is clipped whatever the input.
    assert len(_clip_title("x" * 5000)) <= MAX_TITLE_CHARS + 1  # +1 for the ellipsis
    assert len(_clip_title(None)) == 0

    # Tags are capped in count. Their individual length is bounded by the
    # tagging vocabulary, not by this module, so the cap on COUNT is what this
    # row controls -- and n_tags reports what was dropped.
    out = feed_mod._list_feed("ana")
    for item in out["items"]:
        assert len(item["tags"]) <= MAX_TAGS_SHOWN
        assert len(item["title"]) <= MAX_TITLE_CHARS + 1


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
    assert out["item"]["url"] == listed["url"]
    # read returns the FULL title where the list clips it. That asymmetry is
    # the point of this tool: the list stays cheap, and reading stays
    # possible. The listed form must still be a recognisable prefix, or a
    # caller could not tell it is the same item.
    assert len(out["item"]["title"]) >= len(listed["title"])
    assert out["item"]["title"].startswith(listed["title"].rstrip("…")[:60])


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


def test_router_answers_are_bounded_by_title_length(stocked):
    """The `.ask` answers had no size guard, and they are the primary entry point.

    `_summarise` caps the number of results at five but not their length, and
    `_label` returns a title verbatim. The longest title in the live database
    is 223 chars, so five of them is ~1115 for the answer string alone --
    twice the largest answer observed there, and reached by ordinary arXiv
    titles rather than anything pathological.

    This is the same shape as the provenance caveat that breached the payload
    budget three rounds running: a string on a hot path whose length is a
    property of the data rather than of the code.
    """
    conn = get_db(stocked)
    conn.execute(
        "UPDATE items SET title = ?",
        ("A Remarkably Long Paper Title " * 12,),  # 360 chars, > the live max
    )
    conn.commit()
    conn.close()

    from attestation.mcp.ask import _feed_ask

    out = _feed_ask("ana", "what should I read today")
    assert len(out["answer"]) <= 600, f"answer is {len(out['answer'])} chars"
    assert len(json.dumps(out)) <= MAX_DEFAULT_RESPONSE_CHARS

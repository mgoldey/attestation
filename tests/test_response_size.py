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


def emitted(payload) -> int:
    """Characters the MODEL receives, not the characters we happened to measure.

    Every budget in this file measured `json.dumps(out)` -- compact, no spaces.
    FastMCP serialises tool results with `indent=2`, a measured 1.238x
    inflation on a real response: 1562 compact became 1934 emitted. So the
    guards were green while 27 of 30 default calls against the live database
    exceeded 2000 as actually sent, which is precisely the payload the module
    docstring says gemma4:e2b could not render.

    A budget that measures something other than what the caller receives is
    not a budget.
    """
    return len(json.dumps(payload, indent=2))


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
# DERIVED from the field caps, not measured from a sample. Both were sized
# from observed rows twice -- 300 from a fixture, then 320/380 from live data --
# and both times the caps permitted more than the budget allowed, because a
# sample is not a bound. Importing the caps makes the two impossible to
# disagree: tightening a budget now requires tightening a cap.
def _row_budget(extra: int = 0) -> int:
    from attestation.mcp.feed import (
        MAX_SOURCE_CHARS,
        MAX_TAG_CHARS,
        MAX_TAGS_SHOWN,
        MAX_TITLE_CHARS,
        MAX_URL_CHARS,
    )

    fields = MAX_TITLE_CHARS + MAX_URL_CHARS + MAX_SOURCE_CHARS + MAX_TAGS_SHOWN * MAX_TAG_CHARS
    # 200 covers the JSON scaffolding AS EMITTED: keys, quotes, commas, plus
    # indent=2's newlines and leading spaces, which cost ~50 more per row than
    # the compact form these budgets were originally derived against.
    return fields + 200 + extra


MAX_ITEM_CHARS = _row_budget()
# A search row adds already_rated, match and relevance.
MAX_SEARCH_ITEM_CHARS = _row_budget(60)


# DERIVED, like the row budgets: the default item count times what a row may
# cost, plus the envelope. A fixed 2000 disagreed with the caps -- 4 rows at
# the permitted 496 is already 1984, leaving 16 chars for ok/message/
# ranking_quality, which alone measures 245-393. That contradiction was
# invisible because the guard measured COMPACT json while FastMCP emits
# indent=2 (1.238x), so all five live personas exceeded 2000 as actually sent.
def _response_budget() -> int:
    from attestation.mcp.feed import DEFAULT_LIST_LIMIT

    envelope = 500  # ok, message, ranking_quality with its longest caveat
    return DEFAULT_LIST_LIMIT * MAX_SEARCH_ITEM_CHARS + envelope


MAX_DEFAULT_RESPONSE_CHARS = _response_budget()


# feed.read carries one full abstract, so it is deliberately the largest
# single-item response. Derived from the abstract cap plus the row, rather than
# the round 2600 that was chosen when nobody had measured an emitted payload.
def _read_budget() -> int:
    from attestation.mcp.feed import FULL_SUMMARY_CHARS

    return FULL_SUMMARY_CHARS + _row_budget() + 400


MAX_READ_RESPONSE_CHARS = _read_budget()


def _digest_budget() -> int:
    from attestation.mcp.feed import MAX_DIGEST_ITEMS

    return MAX_DIGEST_ITEMS * MAX_ITEM_CHARS + 900  # + topic labels and envelope


MAX_DIGEST_RESPONSE_CHARS = _digest_budget()


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
    size = emitted(out)
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
    size = emitted(out)
    assert out["ranking_quality"].get("caveat"), "this fixture must produce a caveat"
    assert size <= MAX_DEFAULT_RESPONSE_CHARS, (
        f"the default feed.list is {size} chars for a bootstrap-heavy persona; "
        "the zero-click fixture hid this"
    )


def test_each_item_stays_cheap(stocked):
    """The per-row cost, which is what makes a large explicit limit tolerable."""
    for item in feed_mod._list_feed("ana", limit=10)["items"]:
        size = emitted(item)
        assert size <= MAX_ITEM_CHARS, f"{size} chars for one item: {item}"


def test_each_search_item_stays_cheap(stocked):
    """Search rows carry three fields a list row does not, so they get their
    own budget rather than being measured against the list one."""
    for item in feed_mod._search_feed("ana", "topic", limit=10)["items"]:
        size = emitted(item)
        assert size <= MAX_SEARCH_ITEM_CHARS, f"{size} chars for one search item: {item}"


def test_search_is_bounded_too(stocked):
    out = feed_mod._search_feed("ana", "topic")
    assert emitted(out) <= MAX_DEFAULT_RESPONSE_CHARS


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
    size = emitted(out)
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

    from attestation.mcp import feed as f

    # emitted(), not compact -- this assertion sat in the same file that added
    # emitted() and kept the old units, and 9 of the 40 longest real items
    # breach 2600 as actually sent while it stays green.
    assert emitted(f._read_item("ana", 1)) <= MAX_READ_RESPONSE_CHARS


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

    from attestation.mcp.ask import MAX_LABEL_CHARS, _feed_ask, _label

    out = _feed_ask("ana", "what should I read today")
    assert len(out["answer"]) <= 600, f"answer is {len(out['answer'])} chars"
    assert emitted(out) <= MAX_DEFAULT_RESPONSE_CHARS

    # _label directly, not only through the router: _item_row now clips the
    # title first, so a long title never reaches _label via the feed path and
    # removing _label's own clip left this test green. A router also labels
    # graph nodes, run arms and feed names, which no other clip touches.
    assert len(_label({"title": "T" * 900})) <= MAX_LABEL_CHARS + 1
    assert len(_label({"name": "N" * 900, "project": "p"})) <= MAX_LABEL_CHARS + 20
    assert len(_label("S" * 900)) <= MAX_LABEL_CHARS + 1


def test_the_item_budget_is_at_least_what_the_field_caps_permit(stocked):
    """The budget must be derivable from the caps, not measured from a sample.

    Round 6 bounded the title and asserted "every variable-length field has a
    stated cap" -- but only title and tags were checked. `url` and `source`
    were uncapped, so the caps permitted 439 chars of field text against a
    320-char budget, and 350 of 5222 real search rows breached it. Five of five
    ordinary queries against the live database produced an over-budget row.

    Both numbers had been set from OBSERVED rows, which is the same mistake as
    sizing them from a fixture: a sample is not a bound. Asserting the budget
    against the caps makes the two impossible to disagree.
    """
    from attestation.mcp.feed import (
        MAX_SOURCE_CHARS,
        MAX_TAG_CHARS,
        MAX_TAGS_SHOWN,
        MAX_TITLE_CHARS,
        MAX_URL_CHARS,
    )

    # Field text the caps allow, plus JSON keys/quotes/commas for the fixed
    # scaffolding (item_id, n_tags, content_type and the punctuation).
    scaffolding = 150
    worst_list = (
        MAX_TITLE_CHARS + MAX_URL_CHARS + MAX_SOURCE_CHARS + MAX_TAGS_SHOWN * MAX_TAG_CHARS
    ) + scaffolding
    assert worst_list <= MAX_ITEM_CHARS, (
        f"the field caps permit a {worst_list}-char row against a {MAX_ITEM_CHARS} budget"
    )
    # A search row adds already_rated, match and relevance.
    assert worst_list + 60 <= MAX_SEARCH_ITEM_CHARS, (
        f"the caps permit {worst_list + 60} against {MAX_SEARCH_ITEM_CHARS}"
    )

    # And the caps must be ENFORCED, not merely declared. Comparing constants
    # to each other is self-consistent and toothless: with `url` left uncapped
    # in _item_row the arithmetic above still passed, because both sides came
    # from the same numbers. Drive a row whose every field exceeds its cap.
    conn = get_db(stocked)
    conn.execute(
        "UPDATE items SET title = ?, url = ?",
        ("T" * 500, "https://example.com/" + "p" * 500),
    )
    conn.execute("UPDATE feeds SET title = ?", ("S" * 500,))
    conn.execute("UPDATE item_tags SET tag = tag || ?", ("g" * 200,))
    conn.commit()
    conn.close()

    for item in feed_mod._list_feed("ana", limit=5)["items"]:
        assert len(item["url"]) <= MAX_URL_CHARS + 1, item["url"]
        assert len(item["source"]) <= MAX_SOURCE_CHARS + 1, item["source"]
        assert len(item["title"]) <= MAX_TITLE_CHARS + 1
        for tag in item["tags"]:
            assert len(tag) <= MAX_TAG_CHARS + 1, tag
        assert emitted(item) <= MAX_ITEM_CHARS


def test_emitted_size_is_what_is_measured(stocked):
    """The guard must measure what FastMCP sends, not what we serialise here.

    Every budget in this file measured compact `json.dumps`. FastMCP emits
    `indent=2` -- a measured 1.238x on a real response (1562 -> 1934) -- so all
    five live personas exceeded 2000 on feed.search as actually sent while
    every test here was green.

    Anchored in the difference itself, so reverting `emitted()` to compact
    fails: a fixture cheap enough to fit either way cannot catch that.
    """
    out = feed_mod._list_feed("ana")
    assert emitted(out) > len(json.dumps(out)), (
        "emitted() is not measuring the indented form FastMCP actually sends"
    )
    assert emitted(out) / len(json.dumps(out)) > 1.1


def test_the_digest_has_a_stated_budget_too(stocked):
    """feed.digest had no size guard at all, and it is the largest response.

    Measured on the live database as FastMCP emits it: feed.list 1835,
    feed.search 2083, feed.digest 4296. The digest groups by topic and defaults
    to limit=30 against list's 4, so it legitimately costs more -- but "more"
    was unbounded and untested, which is how the one tool a reader is most
    likely to call each morning became the one nothing measured.

    Its budget is derived from its own defaults, the same way the list budget
    is, so a default change cannot silently invalidate it.
    """
    from attestation.mcp.feed import DEFAULT_DIGEST_LIMIT

    out = feed_mod._digest("ana")
    size = emitted(out)
    ceiling = DEFAULT_DIGEST_LIMIT * MAX_ITEM_CHARS + 900  # + topic labels, envelope
    assert size <= ceiling, f"digest is {size} chars, ceiling {ceiling}"


def test_read_does_not_pay_for_the_title_twice(stocked):
    """`message` repeated the full title that `item.title` already carries.

    On the live corpus's longest titles that is ~220 wasted chars on the one
    response deliberately allowed to be large, and it is the field a caller
    never reads for content -- `message` is the envelope's one-line status,
    not a second copy of the payload.
    """
    conn = get_db(stocked)
    conn.execute("UPDATE items SET title = ?", ("T" * 400,))
    conn.commit()
    conn.close()

    out = feed_mod._read_item("ana", 1)
    assert out["item"]["title"].startswith("T"), "the full title must still be in item.title"
    assert len(out["message"]) < 200, f"message is {len(out['message'])} chars of duplicate title"
    assert emitted(out) <= MAX_READ_RESPONSE_CHARS


def test_the_digest_is_bounded_by_total_items_not_by_topic_count(stocked):
    """A cap on topics alone does not bound the digest.

    Measured on the live database: ml-engineer and structural-biologist each
    emit ~6900 from 6 topics, but materials-scientist emits 5031 from only 3 --
    because 5 `unclustered` items sit OUTSIDE the topic list and no cap covered
    them. Capping topics would have left the one persona with a messy profile
    over budget, which is the persona most likely to be real.

    So the bound is on total items across topics AND unclustered, which is what
    the reader actually has to render.
    """
    conn = get_db(stocked)
    user_id = conn.execute("SELECT id FROM users WHERE name = 'ana'").fetchone()[0]
    # Spread items across many distinct tags so clustering produces both
    # several topics and an unclustered remainder.
    for i in range(1, 41):
        conn.execute("DELETE FROM item_tags WHERE item_id = ?", (i,))
        conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, ?)", (i, f"topic-{i % 9}"))
    conn.commit()
    conn.close()
    assert user_id

    out = feed_mod._digest("ana")
    shown = sum(len(t.get("items", [])) for t in out.get("topics", [])) + len(
        out.get("unclustered", [])
    )
    assert shown <= feed_mod.MAX_DIGEST_ITEMS, f"digest rendered {shown} items"
    assert emitted(out) <= MAX_DIGEST_RESPONSE_CHARS, f"digest is {emitted(out)} chars"


# Tools whose size is a deliberate design choice, with the reason. Everything
# else must fit the conversational budget -- a tool absent from both this list
# and the assertions above is unmeasured, which is how feed.digest reached
# 6905 chars without anyone noticing.
COMPOSITION_TOOLS = {
    # CLAUDE.md: "No LLM in composition tools: digest/runs_compare return
    # structure, never prose -- the caller is a model." These are read by a
    # model that will select from them, not rendered verbatim to a person.
    "runs.compare": "one row per arm; size is a function of the family, not a choice",
    "runs.list": "one row per run; the caller narrows with project=",
    "runs.claims_check": "one verdict per claim in the document",
    "runs.claims_coverage": "one row per uncovered number",
    "kg.communities": "one cluster per topic in the graph",
    "kg.neighbors": "one row per adjacent concept",
    "kg.concepts": "the graph vocabulary, which is the answer",
    "feed.sources": "one row per subscribed feed",
    "feed.digest": "grouped, and separately bounded at MAX_DIGEST_ITEMS",
    "feed.personas": "one row per reader; small and bounded by persona count",
    "feed.source_suggest": "a scored shortlist the caller picks from",
}


def test_every_tool_is_either_budgeted_or_declared_a_composition_tool():
    """No tool may be simply unmeasured.

    Round 9 measured all 50 as emitted and found 44 with no stated budget --
    including the six largest, up to 13624 chars. feed.digest reached 6905
    that way: coverage had followed what was easy to test rather than what was
    called. This does not impose a number on a composition tool; it requires a
    DECISION to have been made and written down.
    """
    import asyncio

    from mcp.server.fastmcp import FastMCP

    from attestation.mcp import register_all

    server = FastMCP("budget-census")
    register_all(server)
    names = {t.name for t in asyncio.run(server.list_tools())}

    # Tools with an explicit size assertion in this module.
    budgeted = {"feed.list", "feed.search", "feed.read", "feed.digest"}
    # Routers bound their own answers via _summarise's label cap.
    routers = {n for n in names if n.endswith(".ask") or n.endswith(".tools")}
    # Mutators and single-value tools return a status, not a payload.
    trivial = {
        n
        for n in names
        if n.startswith("sym.")
        or n.startswith("cite.")
        or n.startswith("feed.persona")
        or n.startswith("feed.source_")
        or n in {"feed.rate", "feed.explain", "feed.harvest_engagement", "feed.simulate_ratings"}
        or n in {"runs.scan", "runs.detail", "kg.path", "kg.central"}
    }

    unaccounted = names - budgeted - routers - trivial - set(COMPOSITION_TOOLS)
    assert not unaccounted, (
        "these tools have no budget and are not declared composition tools: "
        + ", ".join(sorted(unaccounted))
    )

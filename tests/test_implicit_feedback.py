"""Engagement that already happens, counted as the signal it is.

Explicit feedback is a channel almost nobody uses: this database holds 68 UI
clicks and 2 agent clicks against 5,167 items, ever. A ranker waiting to be
told what is useful waits forever, and the classifier stays off.

Meanwhile 99 explanation requests were logged, 91 of them for items that were
never rated. Asking "why is this here?" is engagement -- weaker than a stated
opinion, but not nothing, and it is already being collected.

So it is recorded as a weak positive under its own source. Weak matters: a
curious reader is not an approving one, and `evaluate_user` must be able to
exclude these the way it excludes bootstrap rows.
"""

import pytest

from attestation import implicit
from attestation.db import get_db
from attestation.rank import CLICK_SOURCES, get_user, record_click


@pytest.fixture
def seeded(tmp_path):
    conn = get_db(tmp_path / "t.db")
    conn.execute("INSERT INTO users(name, interests) VALUES ('ana', 'protein folding')")
    for i in range(5):
        conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash)"
            " VALUES (NULL, ?, 'http://x', ?, ?)",
            (f"Item {i}", f"Summary {i}", f"h{i}"),
        )
    conn.commit()
    return conn


def _explain(conn, uid, item_id, text="because"):
    conn.execute(
        "INSERT OR IGNORE INTO explanations(user_id, item_id, text) VALUES (?, ?, ?)",
        (uid, item_id, text),
    )
    conn.commit()


def test_asking_why_becomes_a_weak_positive(seeded):
    uid = get_user(seeded, "ana")["id"]
    _explain(seeded, uid, 1)
    _explain(seeded, uid, 2)

    out = implicit.harvest(seeded, "ana")

    assert out["recorded"] == 2
    rows = seeded.execute("SELECT item_id, useful, source FROM clicks ORDER BY item_id").fetchall()
    assert [r["item_id"] for r in rows] == [1, 2]
    assert all(r["useful"] == 1 for r in rows)
    assert all(r["source"] == "implicit" for r in rows)


def test_implicit_is_its_own_source_not_ui(seeded):
    """A curious reader is not an approving one. If these were written as `ui`
    they would be indistinguishable from a person pressing the button, and no
    later evaluation could separate them."""
    assert "implicit" in CLICK_SOURCES
    uid = get_user(seeded, "ana")["id"]
    _explain(seeded, uid, 1)
    implicit.harvest(seeded, "ana")
    assert seeded.execute("SELECT source FROM clicks").fetchone()["source"] == "implicit"


def test_a_stated_opinion_is_never_overwritten(seeded):
    """The whole risk of implicit signal. Someone who explicitly said 'not
    useful' and then asked why must not have that flipped to useful."""
    uid = get_user(seeded, "ana")["id"]
    record_click(seeded, uid, 1, False, source="ui")
    _explain(seeded, uid, 1)

    out = implicit.harvest(seeded, "ana")

    assert out["skipped_already_rated"] == 1
    assert out["recorded"] == 0
    row = seeded.execute("SELECT useful, source FROM clicks WHERE item_id = 1").fetchone()
    assert row["useful"] == 0, "an explicit rejection was overwritten by curiosity"
    assert row["source"] == "ui"


def test_harvesting_twice_records_nothing_new(seeded):
    uid = get_user(seeded, "ana")["id"]
    _explain(seeded, uid, 1)
    assert implicit.harvest(seeded, "ana")["recorded"] == 1
    assert implicit.harvest(seeded, "ana")["recorded"] == 0
    assert seeded.execute("SELECT COUNT(*) n FROM clicks").fetchone()["n"] == 1


def test_nothing_to_harvest_is_reported_not_an_error(seeded):
    out = implicit.harvest(seeded, "ana")
    assert out["recorded"] == 0
    assert out["candidates"] == 0


def test_unknown_user_is_refused(seeded):
    with pytest.raises(ValueError, match="unknown user"):
        implicit.harvest(seeded, "nobody")


def test_reset_does_not_let_harvest_resurrect_a_rejection_as_a_positive(tmp_path, monkeypatch):
    """`persona_reset` cleared clicks and left explanations, so the next
    harvest turned a stated "not useful" into "useful".

    implicit.py's docstring states the invariant: "an item already carrying a
    click is excluded here rather than being overwritten later, so a stated
    'not useful' is never flipped to useful by the reader's own curiosity."
    The LEFT JOIN enforces that only while the click exists. Reset removed the
    guard's evidence and not its subject.

    Direction matters: CLAUDE.md records that negatives are the class the
    ranker is starving for, and only positives are ever inferred. Converting
    the scarce class into the abundant one is the worst available error.
    """
    from attestation import implicit
    from attestation.db import get_db
    from attestation.mcp.feed import _reset_feedback
    from attestation.rank import create_user

    # RSS_DB, or _reset_feedback opens a DIFFERENT database via
    # resolve_db_path and the test passes while proving nothing.
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    conn = get_db(tmp_path / "t.db")
    user_id = create_user(conn, "ana", "machine learning")
    conn.execute(
        "INSERT INTO items(feed_id, title, url, summary, content_hash)"
        " VALUES (NULL, 't', 'u', 's', 'h')"
    )
    conn.execute(
        "INSERT INTO clicks(user_id, item_id, useful, source) VALUES (?, 1, 0, 'agent')",
        (user_id,),
    )
    conn.execute(
        "INSERT INTO explanations(user_id, item_id, text) VALUES (?, 1, 'why')", (user_id,)
    )
    conn.commit()
    conn.close()

    _reset_feedback("ana", confirm=True)

    conn = get_db(tmp_path / "t.db")
    implicit.harvest(conn, "ana")
    rows = list(conn.execute("SELECT useful, source FROM clicks WHERE user_id = ?", (user_id,)))
    assert not any(r["useful"] for r in rows), (
        f"reset then harvest recreated the cleared rating as a positive: {[dict(r) for r in rows]}"
    )


def test_harvest_holds_the_write_lock_briefly(tmp_path):
    """A long harvest refused a reader's rating.

    `record_click` commits on every call and harvest calls it per candidate, so
    the run is N commits, not one. Measured: 800 rows took 6.1s -- past
    busy_timeout=5000, which makes a concurrent write GUARANTEED to be refused
    rather than merely unlucky, and the candidate set has no upper bound.

    The user-visible failure is a reader's explicit "not useful" coming back as
    "the database is busy" while a background harvest runs -- rejecting the
    scarce class to record the abundant one.
    """
    import time

    from attestation.db import get_db
    from attestation.rank import create_user

    conn = get_db(tmp_path / "t.db")
    user_id = create_user(conn, "bulk", "machine learning")
    for i in range(1, 601):
        conn.execute(
            "INSERT INTO items(id, feed_id, title, url, summary, content_hash)"
            " VALUES (?, NULL, 't', 'u', 's', ?)",
            (i, f"h{i}"),
        )
        conn.execute(
            "INSERT INTO explanations(user_id, item_id, text) VALUES (?, ?, 'w')", (user_id, i)
        )
    conn.commit()

    from attestation import implicit

    started = time.perf_counter()
    out = implicit.harvest(conn, "bulk")
    elapsed = time.perf_counter() - started

    assert out["recorded"] == 600
    assert elapsed < 2.0, f"harvest of 600 rows took {elapsed:.1f}s; busy_timeout is 5s"


def test_reading_an_item_becomes_harvestable_signal(tmp_path, monkeypatch):
    """Signal arrived only when the reader pressed a button.

    Measured on the live database: 11 human clicks across 19 days, on 3 days,
    all from deliberately sitting down to rate -- while the corpus held 5,265
    items. SKILL.md already predicts this ("users never press buttons"), and
    implicit.py harvested one action only: asking why an item ranked.

    Reading a paper in full is the strongest thing a reader does short of
    rating it, and it required no gesture. It is recorded in `engagement` and
    harvested as a weak positive tagged `implicit`, so provenance still decides
    what it may be used for and `evaluate_user` still excludes it.
    """
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    from attestation.db import get_db
    from attestation.implicit import candidates, harvest
    from attestation.mcp.feed import _read_item

    conn = get_db(tmp_path / "t.db")
    conn.execute("INSERT INTO users(name, interests) VALUES ('reader', 'science')")
    conn.execute("INSERT INTO feeds(title, url) VALUES ('f', 'http://f')")
    for i in (1, 2):
        conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash)"
            " VALUES (1, ?, 'http://x', 'body', ?)",
            (f"paper {i}", f"h{i}"),
        )
    conn.commit()
    user_id = conn.execute("SELECT id FROM users WHERE name='reader'").fetchone()["id"]

    assert candidates(conn, user_id) == [], "nothing engaged with yet"

    _read_item("reader", item_id=1)

    fresh = get_db(tmp_path / "t.db")
    assert candidates(fresh, user_id) == [1], "a read left no harvestable trace"
    out = harvest(fresh, "reader")
    assert out["recorded"] == 1, out
    row = fresh.execute(
        "SELECT useful, source FROM clicks WHERE user_id = ? AND item_id = 1", (user_id,)
    ).fetchone()
    assert row["useful"] == 1 and row["source"] == "implicit", dict(row)


def test_reading_an_item_never_overturns_a_stated_verdict(tmp_path, monkeypatch):
    """Only POSITIVES are inferred, and an explicit judgement outranks a read.

    A reader who marked something "not useful" and later opened it to check
    must not have that rejection flipped by their own curiosity -- the class
    the ranker is starving for is the one this would poison.
    """
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    from attestation.db import get_db
    from attestation.implicit import harvest
    from attestation.mcp.feed import _read_item
    from attestation.rank import record_click

    conn = get_db(tmp_path / "t.db")
    conn.execute("INSERT INTO users(name, interests) VALUES ('reader', 'science')")
    conn.execute("INSERT INTO feeds(title, url) VALUES ('f', 'http://f')")
    conn.execute(
        "INSERT INTO items(feed_id, title, url, summary, content_hash)"
        " VALUES (1, 'rejected paper', 'http://x', 'body', 'h1')"
    )
    conn.commit()
    user_id = conn.execute("SELECT id FROM users WHERE name='reader'").fetchone()["id"]
    record_click(conn, user_id, 1, useful=False, source="ui")

    _read_item("reader", item_id=1)
    fresh = get_db(tmp_path / "t.db")
    harvest(fresh, "reader")

    row = fresh.execute(
        "SELECT useful, source FROM clicks WHERE user_id = ? AND item_id = 1", (user_id,)
    ).fetchone()
    assert row["useful"] == 0, "a read overwrote a stated rejection"
    assert row["source"] == "ui", f"provenance was downgraded to {row['source']!r}"


def test_asking_why_records_engagement_even_when_the_model_is_down(tmp_path, monkeypatch):
    """Two defects in one call, found because `engagement` was still empty two
    rounds after it shipped.

    `feed.explain` returned `ok: true, explanation: null, message: ""` when the
    local model was unreachable -- collapsing "no explanation could be
    generated", "the model is down" and "the item has no text" into one
    indistinguishable success. A caller reading `explanation: null` has no
    reason to retry and tells the reader there is no explanation.

    And a failed explain wrote no `explanations` row, so it also lost the
    ENGAGEMENT: asking why an item ranked is evidence this reader cared about
    it, whether or not the model answers. That is the signal implicit.harvest
    was built on, and a cold model silently discarded it.
    """
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    from attestation.db import get_db
    from attestation.implicit import candidates
    from attestation.mcp.feed import _explain_item

    conn = get_db(tmp_path / "t.db")
    conn.execute("INSERT INTO users(name, interests) VALUES ('reader', 'science')")
    conn.execute("INSERT INTO feeds(title, url) VALUES ('f', 'http://f')")
    conn.execute(
        "INSERT INTO items(feed_id, title, url, summary, content_hash)"
        " VALUES (1, 'a paper', 'http://x', 'body', 'h1')"
    )
    conn.commit()
    user_id = conn.execute("SELECT id FROM users WHERE name='reader'").fetchone()["id"]

    # A model that cannot answer. Stubs the NEW contract -- explain() never
    # returns a bare None (see explain.ExplainResult); the old stub
    # (lambda *a, **k: None) simulated a contract this module no longer has
    # and made _explain_item crash on `.reason` instead of exercising the
    # "model unreachable" refusal this test is meant to cover.
    from attestation.explain import ExplainResult

    monkeypatch.setattr(
        "attestation.mcp.feed.explain_item_fn",
        lambda *a, **k: ExplainResult(text=None, reason="model_unreachable"),
    )
    out = _explain_item("reader", item_id=1)
    assert out["ok"] is False, "a failed explanation was reported as success"
    assert "unreachable" in out["message"], out["message"]

    fresh = get_db(tmp_path / "t.db")
    assert candidates(fresh, user_id) == [1], (
        "a failed explain discarded the engagement as well as the explanation"
    )

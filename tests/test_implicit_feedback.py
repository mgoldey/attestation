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

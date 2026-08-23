"""An unknown persona is created, not refused.

Three behaviours, all one idea: the agent should never make a reader do
bookkeeping.

**Auto-create.** Refusing an unknown name and listing the valid ones taught
agents to call persona_create with whatever string they had. The live database
grew a `Matthew Goldey` with zero clicks that way, days after that persona had
been merged into `matt` -- the refusal did not prevent the duplicate, it
caused it.

**Never ask for a name.** A name is bookkeeping the reader did not ask to do.
Whatever the caller passed IS the name.

**Ask for interests instead.** That is the one thing only the reader knows,
and it is what the ranking is actually built from -- the interests string is
the profile embedding.
"""

import pytest

from attestation.db import get_db
from attestation.mcp import feed as feed_mod
from attestation.rank import get_user


@pytest.fixture
def seeded(tmp_path, monkeypatch, fake_embedder):
    from attestation.mcp import _shared

    db = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(db))
    conn = get_db(db)
    conn.execute("DELETE FROM users")
    conn.execute("INSERT INTO users(id, name, interests) VALUES (1, 'ana', 'protein folding')")
    for i in range(1, 9):
        cur = conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash)"
            " VALUES (NULL, ?, 'u', ?, ?)",
            (f"Paper {i}", f"About topic {i}", f"h{i}"),
        )
        conn.execute(
            "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)",
            (cur.lastrowid, fake_embedder.embed_document("t", "s").tobytes()),
        )
        conn.execute(
            "INSERT INTO item_tags(item_id, tag) VALUES (?, 'machine-learning')",
            (cur.lastrowid,),
        )
    conn.commit()
    conn.close()
    monkeypatch.setattr(_shared, "_embedder", fake_embedder)
    monkeypatch.setattr(_shared, "get_embedder", lambda: fake_embedder)
    return db


def test_an_unknown_persona_is_created_and_served(seeded):
    """No refusal, no name list, no round trip."""
    out = feed_mod._list_feed("newcomer", limit=3)

    assert out["ok"] is True, out["message"]
    assert out["items"], "a new reader must still get a feed"
    conn = get_db(seeded)
    assert get_user(conn, "newcomer") is not None, "the persona was not created"
    conn.close()


def test_a_created_persona_is_announced_not_silent(seeded):
    """Creating a reader profile is a real side effect. Doing it silently
    means a typo becomes a permanent persona nobody knows about."""
    out = feed_mod._list_feed("typoo", limit=2)
    assert "typoo" in out["message"]
    assert "created" in out["message"].lower()


def test_a_new_persona_is_asked_for_interests_not_a_name(seeded):
    """The name is whatever was passed. Interests are the thing only the
    reader knows -- and the interests string IS the profile embedding."""
    out = feed_mod._list_feed("newcomer", limit=2)
    message = out["message"].lower()
    # Asks for the subject matter, in the reader's terms. "what you actually
    # read about" beats "supply an interests string" for a person, so the
    # assertion is on intent rather than on the word "interests".
    assert any(p in message for p in ("monitor", "topics", "read about", "interest")), out[
        "message"
    ]
    for bookkeeping in ("what name", "which persona", "choose a profile", "pick a name"):
        assert bookkeeping not in message, f"asked the reader to name things: {out['message']}"


def test_an_existing_persona_is_untouched(seeded):
    """Auto-create must not overwrite a reader who already exists."""
    conn = get_db(seeded)
    before = get_user(conn, "ana")["interests"]
    conn.close()

    feed_mod._list_feed("ana", limit=2)

    conn = get_db(seeded)
    assert get_user(conn, "ana")["interests"] == before
    assert conn.execute("SELECT COUNT(*) n FROM users").fetchone()["n"] == 1
    conn.close()


def test_creating_twice_does_not_duplicate(seeded):
    feed_mod._list_feed("newcomer", limit=1)
    feed_mod._list_feed("newcomer", limit=1)
    conn = get_db(seeded)
    n = conn.execute("SELECT COUNT(*) n FROM users WHERE name = 'newcomer'").fetchone()["n"]
    conn.close()
    assert n == 1


def test_a_new_persona_gets_a_usable_starting_profile(seeded):
    """An empty interests string ranks on nothing. A new reader should get a
    feed that reflects the corpus until they say otherwise."""
    feed_mod._list_feed("newcomer", limit=2)
    conn = get_db(seeded)
    interests = get_user(conn, "newcomer")["interests"]
    conn.close()
    assert interests and interests.strip(), "a new persona must not start with empty interests"


def test_destructive_tools_still_refuse_an_unknown_persona(seeded):
    """Auto-create is for reading. Deleting or resetting a persona that does
    not exist is a mistake, and creating one to delete it is absurd."""
    out = feed_mod._delete_persona("ghost", confirm=True)
    assert out["ok"] is False
    assert "unknown user" in out["message"]

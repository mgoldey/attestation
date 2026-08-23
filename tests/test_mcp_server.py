import pytest

from attestation import mcp_server
from attestation.db import get_db
from attestation.mcp import _shared


@pytest.fixture(autouse=True)
def _patch_env_db(tmp_path, monkeypatch):
    """Every test gets its own DB via RSS_DB, honoring resolve_db_path()."""
    db_path = tmp_path / "t.db"
    monkeypatch.setenv("RSS_DB", str(db_path))
    return db_path


@pytest.fixture
def seeded_conn(_patch_env_db, fake_embedder, monkeypatch):
    """Seed the env-pointed DB with items + vectors, patch the module-level embedder."""
    conn = get_db(_patch_env_db)
    for i in range(15):
        cur = conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash)"
            " VALUES (NULL, ?, 'http://x', ?, ?)",
            (f"item {i}", f"summary {i}", f"hash-{i}"),
        )
        vec = fake_embedder.embed_document(f"item {i}", f"summary {i}")
        conn.execute(
            "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)",
            (cur.lastrowid, vec.tobytes()),
        )
    conn.commit()
    conn.close()
    monkeypatch.setattr(_shared, "_embedder", fake_embedder)
    monkeypatch.setattr(_shared, "get_embedder", lambda: fake_embedder)
    return _patch_env_db


class TestListFeed:
    def test_returns_items_with_required_keys(self, seeded_conn):
        result = mcp_server._list_feed_impl("matt", limit=5)
        assert "items" in result
        assert len(result["items"]) == 5
        for item in result["items"]:
            assert {
                "item_id",
                "title",
                "url",
                "source",
                "tags",
                "content_type",
            } <= set(item.keys()), item

    def test_respects_limit(self, seeded_conn):
        result = mcp_server._list_feed_impl("matt", limit=3)
        assert len(result["items"]) == 3

    def test_limit_capped_at_50(self, seeded_conn):
        result = mcp_server._list_feed_impl("matt", limit=9999)
        # only 15 items seeded, so this also proves the cap didn't error / blow up
        assert len(result["items"]) <= 50
        assert len(result["items"]) == 15

    def test_unknown_user_gives_clear_error_not_exception(seeded_conn):
        """An unknown reader is now CREATED, not refused.

        Refusing and listing the valid names taught agents to call
        persona_create with whatever string they had -- the live database
        grew a duplicate persona that way, days after that reader had been
        merged away. The refusal caused the duplicate it was meant to
        prevent, so read-side tools create on first sight and say so.
        """
        out = mcp_server._list_feed_impl("nobody-yet", limit=2)

        assert out["ok"] is True, out["message"]
        assert "created" in out["message"].lower()
        assert "nobody-yet" in out["message"]
        # and it asks the one question only the reader can answer
        assert "monitor" in out["message"].lower() or "topics" in out["message"].lower()


def test_record_feedback_records_agent_source(seeded_conn):
    from attestation.db import get_db, resolve_db_path

    out = mcp_server._record_feedback_impl("matt", 1, True)
    assert out["ok"] is True

    conn = get_db(resolve_db_path(None))
    row = conn.execute("SELECT source FROM clicks WHERE item_id = 1").fetchone()
    assert row["source"] == "agent"
    conn.close()


def test_remove_feed_without_confirm_mutates_nothing(seeded_conn):
    from attestation.db import get_db, resolve_db_path

    conn = get_db(resolve_db_path(None))
    conn.execute("INSERT INTO feeds(id, url, title) VALUES (77, 'http://x/rss', 'X')")
    conn.commit()
    conn.close()

    out = mcp_server._remove_feed_impl(77, confirm=False)

    assert out["ok"] is False
    assert "confirm" in out["message"]
    conn = get_db(resolve_db_path(None))
    assert conn.execute("SELECT COUNT(*) n FROM feeds WHERE id = 77").fetchone()["n"] == 1
    conn.close()


def test_create_and_update_persona(seeded_conn):
    created = mcp_server._create_persona_impl("chemist", "catalysis, spectroscopy")
    assert created["ok"] is True

    dup = mcp_server._create_persona_impl("chemist", "anything")
    assert dup["ok"] is False

    updated = mcp_server._update_persona_impl("chemist", "electrochemistry")
    assert updated["ok"] is True

    status = mcp_server._profile_status_impl("chemist")
    assert status["interests"] == "electrochemistry"
    assert status["clicks"] == 0
    assert status["blend_weight"] == 0.0


def test_destructive_tools_refuse_without_confirm(seeded_conn):
    from attestation.db import get_db, resolve_db_path

    mcp_server._create_persona_impl("victim", "x")
    mcp_server._record_feedback_impl("victim", 1, True)

    assert mcp_server._delete_persona_impl("victim", confirm=False)["ok"] is False
    assert mcp_server._reset_feedback_impl("victim", confirm=False)["ok"] is False

    conn = get_db(resolve_db_path(None))
    assert conn.execute("SELECT COUNT(*) n FROM users WHERE name = 'victim'").fetchone()["n"] == 1
    assert conn.execute("SELECT COUNT(*) n FROM clicks").fetchone()["n"] == 1
    conn.close()

    assert mcp_server._reset_feedback_impl("victim", confirm=True)["ok"] is True
    conn = get_db(resolve_db_path(None))
    assert conn.execute("SELECT COUNT(*) n FROM clicks").fetchone()["n"] == 0
    # persona itself survives a reset
    assert conn.execute("SELECT COUNT(*) n FROM users WHERE name = 'victim'").fetchone()["n"] == 1
    conn.close()


def test_search_feed_finds_already_rated_items(seeded_conn):
    mcp_server._create_persona_impl("searcher", "items")
    mcp_server._record_feedback_impl("searcher", 1, True)

    # An explicit limit: search defaults to 5 now, and the rated item is not
    # necessarily in the top few. The property under test is that search
    # REACHES already-rated items at all -- list_feed excludes them -- not
    # where they land.
    out = mcp_server._search_feed_impl("searcher", "item", limit=20)

    assert out["ok"] is True
    assert out["items"], "search must reach items list_feed would exclude"
    assert any(i["already_rated"] for i in out["items"])


def test_search_feed_matches_summary_only(seeded_conn, fake_embedder):
    """The needle appears only in the summary, never in any seeded title --
    proves search_feed matches against summary text, not title alone."""
    conn = get_db(seeded_conn)
    cur = conn.execute(
        "INSERT INTO items(feed_id, title, url, summary, content_hash)"
        " VALUES (NULL, 'unrelated headline', 'http://x', ?, 'hash-summary-only')",
        ("a survey of quixotic transformers in the wild",),
    )
    vec = fake_embedder.embed_document("unrelated headline", "a survey of quixotic transformers")
    conn.execute(
        "INSERT INTO item_vectors(rowid, embedding) VALUES (?, ?)", (cur.lastrowid, vec.tobytes())
    )
    conn.commit()
    conn.close()

    mcp_server._create_persona_impl("summary-searcher", "transformers")

    out = mcp_server._search_feed_impl("summary-searcher", "quixotic")

    assert out["ok"] is True
    assert any(i["item_id"] == cur.lastrowid for i in out["items"]), (
        "search_feed must match against summary text, not title alone"
    )


def test_search_feed_nonpositive_limit_is_clamped(seeded_conn):
    mcp_server._create_persona_impl("searcher2", "items")

    zero = mcp_server._search_feed_impl("searcher2", "item", limit=0)
    negative = mcp_server._search_feed_impl("searcher2", "item", limit=-5)

    assert len(zero["items"]) <= 1
    assert len(negative["items"]) <= 1


def test_delete_persona_does_not_leak_explanations_to_a_recreated_id(seeded_conn):
    """users.id is a reused rowid: deleting alice and recreating a persona at
    the same id must not hand the new persona alice's cached explanation."""
    from attestation.db import get_db, resolve_db_path

    mcp_server._create_persona_impl("alice", "quantum chemistry")
    conn = get_db(resolve_db_path(None))
    alice_id = conn.execute("SELECT id FROM users WHERE name = 'alice'").fetchone()["id"]
    conn.execute(
        "INSERT INTO explanations(user_id, item_id, text) VALUES (?, 1, 'alice-only text')",
        (alice_id,),
    )
    conn.commit()
    conn.close()

    out = mcp_server._delete_persona_impl("alice", confirm=True)
    assert out["ok"] is True

    conn = get_db(resolve_db_path(None))
    remaining = conn.execute(
        "SELECT COUNT(*) n FROM explanations WHERE user_id = ?", (alice_id,)
    ).fetchone()["n"]
    assert remaining == 0
    conn.close()

    # a new persona landing on the same reused id must see no explanations
    mcp_server._create_persona_impl("bob", "unrelated interests")
    conn = get_db(resolve_db_path(None))
    bob_id = conn.execute("SELECT id FROM users WHERE name = 'bob'").fetchone()["id"]
    assert bob_id == alice_id, "test assumes SQLite reused the freed rowid"
    bob_explanations = conn.execute(
        "SELECT COUNT(*) n FROM explanations WHERE user_id = ?", (bob_id,)
    ).fetchone()["n"]
    conn.close()
    assert bob_explanations == 0


def test_delete_highest_rowid_item_leaves_no_stale_item_vectors_row(seeded_conn):
    """items.id is a reused rowid: deleting the highest-id item must also drop
    its item_vectors row so a future item can't inherit a stale vector."""
    from attestation.db import get_db, resolve_db_path

    conn = get_db(resolve_db_path(None))
    highest_id = conn.execute("SELECT MAX(id) m FROM items").fetchone()["m"]

    conn.execute("DELETE FROM items WHERE id = ?", (highest_id,))
    conn.commit()

    remaining = conn.execute(
        "SELECT COUNT(*) n FROM item_vectors WHERE rowid = ?", (highest_id,)
    ).fetchone()["n"]
    conn.close()
    assert remaining == 0


class TestNonsenseArguments:
    """Arguments no sane caller sends, which an agent sends anyway.

    Found by driving the MCP server with adversarial inputs. Both cases here
    returned ok=true with a wrong answer, which is worse than an error: an
    agent reports "nothing to read" to a researcher when the truth is "you
    asked for items published in the future".
    """

    def test_a_negative_window_is_refused_not_silently_empty(self, seeded_conn):
        out = mcp_server._list_feed_impl("matt", since_days=-30)
        assert out["ok"] is False
        assert "since_days" in out["message"]
        assert out["items"] == []

    def test_a_zero_limit_is_refused_rather_than_answered_with_one(self, seeded_conn):
        """Clamping 0 up to 1 answers a question nobody asked."""
        out = mcp_server._list_feed_impl("matt", limit=0)
        assert out["ok"] is False
        assert "limit" in out["message"]

    def test_a_negative_limit_is_refused(self, seeded_conn):
        out = mcp_server._list_feed_impl("matt", limit=-5)
        assert out["ok"] is False
        assert "limit" in out["message"]

    def test_an_oversized_limit_is_capped_not_refused(self, seeded_conn):
        """Asking for more than the cap is a reasonable request with a
        reasonable answer -- unlike asking for zero or a negative window."""
        out = mcp_server._list_feed_impl("matt", limit=100000)
        assert out["ok"] is True
        assert len(out["items"]) <= 50

    def test_search_rejects_the_same_nonsense(self, seeded_conn):
        assert mcp_server._search_feed_impl("matt", "x", limit=0)["ok"] is False


def test_propose_interests_folds_tags_the_way_autocreate_does(tmp_path, monkeypatch):
    """The two tools that answer "what should a new persona read about"
    disagreed, and the one an agent is told to call was the wrong one.

    `_propose_interests` grouped raw item_tags; `features.tag_vocabulary`
    canonicalises first, and `autocreate_user` uses that one. On the live
    corpus the raw version spent slots on `llm` and `machinelearning` --
    fragments `kg.canonical` folds into `large-language-models` and
    `machine-learning` -- and pushed `natural-language-processing` out
    entirely.

    This string IS the profile embedding, and features.tag_vocabulary's
    docstring documents fixing exactly this feedback loop.
    """
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    conn = get_db(tmp_path / "t.db")
    for i, tag in enumerate(
        ["llm"] * 6 + ["language-models"] * 5 + ["machinelearning"] * 4 + ["evaluation"] * 3,
        start=1,
    ):
        conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash)"
            " VALUES (NULL, ?, 'u', 's', ?)",
            (f"i{i}", f"h{i}"),
        )
        conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, ?)", (i, tag))
    conn.commit()
    conn.close()

    from attestation.mcp.feed import _propose_interests

    tags = _propose_interests()["prevalent_tags"]
    assert "llm" not in tags, f"an unfolded fragment reached the profile text: {tags}"
    assert "machinelearning" not in tags, f"a deprecated spelling was proposed: {tags}"

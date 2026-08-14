from attestation.db import get_db
from attestation.explain import explain


def setup_db(tmp_path):
    conn = get_db(tmp_path / "t.db")
    conn.execute(
        "INSERT INTO items(feed_id, title, summary, content_hash)"
        " VALUES (NULL, 'Attention Is Enough', 'a paper', 'h1')"
    )
    conn.execute("INSERT INTO clicks(user_id, item_id, useful) VALUES (1, 1, 1)")
    return conn


def good_chat(messages, schema):
    return {"text": "Because you clicked similar ranking papers."}


def test_explain_unknown_user_is_handled_quietly(tmp_path, caplog):
    """A deleted persona (or a stale session holding its id) must degrade quietly.

    Previously synthesize_profile subscripted a None row from
    `SELECT interests FROM users WHERE id = ?`, so a TypeError tore through the
    LangGraph run. explain() caught it, so the return value was already None --
    but every request logged a full traceback at ERROR. The guard now short-
    circuits before the graph, so this is a single WARNING and no exception.
    """
    import logging

    conn = setup_db(tmp_path)

    with caplog.at_level(logging.WARNING, logger="attestation.explain"):
        # user 99 does not exist; get_db seeds ids 1-3 only
        assert explain(conn, user_id=99, item_id=1, chat_fn=good_chat) is None

    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], (
        "unknown user must not produce an ERROR/traceback"
    )
    assert any("unknown user_id" in r.message for r in caplog.records)


def test_explain_returns_and_caches(tmp_path):
    conn = setup_db(tmp_path)
    calls = []

    def counting_chat(messages, schema):
        calls.append(1)
        return good_chat(messages, schema)

    text = explain(conn, user_id=1, item_id=1, chat_fn=counting_chat)
    assert text == "Because you clicked similar ranking papers."
    n_first = len(calls)
    # second call: served from cache, no new LLM calls
    assert explain(conn, user_id=1, item_id=1, chat_fn=counting_chat) == text
    assert len(calls) == n_first
    row = conn.execute("SELECT text FROM explanations WHERE user_id=1 AND item_id=1").fetchone()
    assert row["text"] == text


def test_explain_retries_once_then_none(tmp_path):
    conn = setup_db(tmp_path)
    attempts = []

    def bad_chat(messages, schema):
        attempts.append(1)
        return {"wrong_key": 42}  # fails pydantic validation every time

    assert explain(conn, user_id=1, item_id=1, chat_fn=bad_chat) is None
    # profile node (1 try, falls back to interests) + explain node (2 tries)
    assert len(attempts) == 3
    assert conn.execute("SELECT COUNT(*) c FROM explanations").fetchone()["c"] == 0


def test_explain_never_raises_on_chat_exception(tmp_path):
    conn = setup_db(tmp_path)

    def dead_chat(messages, schema):
        raise ConnectionError("ollama down")

    assert explain(conn, user_id=1, item_id=1, chat_fn=dead_chat) is None

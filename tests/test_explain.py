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
    # Explain node only, twice. The profile node no longer calls the model
    # when the persona has interests text -- synthesis produced vaguer
    # profiles than the stored string and cost 2.1s, so it is now a fallback.
    assert len(attempts) == 2
    assert conn.execute("SELECT COUNT(*) c FROM explanations").fetchone()["c"] == 0


def test_explain_never_raises_on_chat_exception(tmp_path):
    conn = setup_db(tmp_path)

    def dead_chat(messages, schema):
        raise ConnectionError("ollama down")

    assert explain(conn, user_id=1, item_id=1, chat_fn=dead_chat) is None


def test_a_persona_with_interests_skips_profile_synthesis(tmp_path):
    """The stored interests text beats a synthesized summary and costs nothing.

    Explaining used to make TWO model calls: synthesize a profile from recent
    liked titles, then explain against it. Measured against gemma4:e2b, the
    synthesis produced meta-description -- "This list of recently useful
    titles covers a diverse range of..." -- and the explanations built on it
    were vaguer than those built on the interests string directly ("Large
    Language Models and agent systems" vs "LLM instruction following and
    reasoning"). It also claimed a paper about termite feed additives matched
    "advanced topics like AI and machine learning", where the interests string
    correctly returned "Outside your stated interests."

    Worse answers for an extra 2.1 seconds. Synthesis is now the fallback for
    a persona with no interests text, not the default path.
    """
    conn = get_db(tmp_path / "t.db")
    conn.execute("DELETE FROM users")  # get_db seeds demo personas at id 1..3
    conn.execute(
        "INSERT INTO users(id, name, interests) VALUES (1, 'ana', 'protein folding, cryo-EM')"
    )
    conn.execute(
        "INSERT INTO items(id, feed_id, title, url, summary, content_hash)"
        " VALUES (1, NULL, 'A paper', 'u', 'about folding', 'h')"
    )
    conn.execute("INSERT INTO clicks(user_id, item_id, useful) VALUES (1, 1, 1)")
    conn.commit()

    calls = []

    def chat_fn(messages, schema):
        calls.append(messages[0]["content"])
        return {"text": "you follow protein folding closely."}

    explain(conn, 1, 1, chat_fn=chat_fn)

    assert len(calls) == 1, f"expected one model call, got {len(calls)}: {calls}"
    assert "Summarize this reader" not in " ".join(calls)
    conn.close()


def test_a_persona_with_no_interests_still_gets_a_synthesized_profile(tmp_path):
    """The fallback has to keep working -- an agent-created persona may have
    an empty interests string and only clicks to go on."""
    conn = get_db(tmp_path / "t.db")
    conn.execute("DELETE FROM users")  # get_db seeds demo personas at id 1..3
    conn.execute("INSERT INTO users(id, name, interests) VALUES (1, 'ana', '')")
    conn.execute(
        "INSERT INTO items(id, feed_id, title, url, summary, content_hash)"
        " VALUES (1, NULL, 'Cryo-EM structure', 'u', 'a structure', 'h')"
    )
    conn.execute("INSERT INTO clicks(user_id, item_id, useful) VALUES (1, 1, 1)")
    conn.commit()

    calls = []

    def chat_fn(messages, schema):
        calls.append(messages[0]["content"])
        return {"text": "structural biology."}

    explain(conn, 1, 1, chat_fn=chat_fn)

    assert len(calls) == 2, "synthesis must still run when there is no interests text"
    assert any("Summarize this reader" in c for c in calls)
    conn.close()

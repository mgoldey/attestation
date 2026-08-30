import ast
import pathlib

from conftest import seeded_db

from attestation.db import get_db
from attestation.explain import explain


def setup_db(tmp_path):
    conn = seeded_db(tmp_path / "t.db")  # personas 1-3 exist; clicks reference user 1
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
        # user 99 does not exist; seeded_db plants ids 1-3 only
        res = explain(conn, user_id=99, item_id=1, chat_fn=good_chat)
        assert res.text is None and res.reason == "unknown_user"

    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], (
        "unknown user must not produce an ERROR/traceback"
    )
    assert any("unknown user_id" in r.message for r in caplog.records)


def test_explain_distinguishes_unknown_user_from_model_down(tmp_path):
    """`ExplainResult.reason` tells the caller which of three unrelated causes
    produced no text -- an unknown user_id is the caller's argument being
    wrong, not the model being unreachable, and the two need different
    responses (`_explain_item` raises a different `ToolError` for each)."""
    conn = seeded_db(tmp_path / "t.db")
    ok_chat = lambda messages, schema: {"text": "You follow this topic."}  # noqa: E731
    res = explain(conn, user_id=999_999, item_id=1, chat_fn=ok_chat)
    assert res.text is None and res.reason == "unknown_user"


def test_explain_returns_and_caches(tmp_path):
    conn = setup_db(tmp_path)
    calls = []

    def counting_chat(messages, schema):
        calls.append(1)
        return good_chat(messages, schema)

    res = explain(conn, user_id=1, item_id=1, chat_fn=counting_chat)
    text = res.text
    assert text == "Because you clicked similar ranking papers."
    assert res.reason == "ok"
    n_first = len(calls)
    # second call: served from cache, no new LLM calls
    assert explain(conn, user_id=1, item_id=1, chat_fn=counting_chat).text == text
    assert len(calls) == n_first
    row = conn.execute("SELECT text FROM explanations WHERE user_id=1 AND item_id=1").fetchone()
    assert row["text"] == text


def test_explain_retries_once_then_none(tmp_path):
    conn = setup_db(tmp_path)
    attempts = []

    def bad_chat(messages, schema):
        attempts.append(1)
        return {"wrong_key": 42}  # fails pydantic validation every time

    res = explain(conn, user_id=1, item_id=1, chat_fn=bad_chat)
    assert res.text is None and res.reason == "no_answer"
    # Explain node only, twice. The profile node no longer calls the model
    # when the persona has interests text -- synthesis produced vaguer
    # profiles than the stored string and cost 2.1s, so it is now a fallback.
    assert len(attempts) == 2
    assert conn.execute("SELECT COUNT(*) c FROM explanations").fetchone()["c"] == 0


def test_explain_never_raises_on_chat_exception(tmp_path):
    """A genuinely dead backend raises `httpx.ConnectError`/`ConnectTimeout`
    -- what `llm.ChatClient` actually raises when Ollama is not running, and
    the same exception `ports.backend_unreachable` is built to recognise (see
    `test_run_tagging_stops_at_an_unreachable_backend_and_says_so` in
    `test_features.py` and `test_a_dead_embedder_is_named_once_not_blamed_on_
    every_feed` in `test_ingest.py` for the same construction). A bare
    `ConnectionError` -- Python's own stdlib exception, not httpx's -- does
    NOT reach `generate_explanation`'s classifier the same way and would pass
    against a fixture that does not match production.
    """
    import httpx

    conn = setup_db(tmp_path)

    def dead_chat(messages, schema):
        raise httpx.ConnectError("connection refused")

    res = explain(conn, user_id=1, item_id=1, chat_fn=dead_chat)
    assert res.text is None and res.reason == "model_unreachable"


def test_explain_reports_no_answer_for_a_non_transport_exception(tmp_path):
    """The sibling case: an exception that is NOT a backend-unreachable
    transport failure (a plain bug in the chat function, say) must still
    report `no_answer`, not be swept into `model_unreachable` by a classifier
    that matches too broadly."""
    conn = setup_db(tmp_path)

    def broken_chat(messages, schema):
        raise ValueError("the chat client blew up on a malformed prompt")

    res = explain(conn, user_id=1, item_id=1, chat_fn=broken_chat)
    assert res.text is None and res.reason == "no_answer"


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
    # get_db creates an EMPTY database; this test makes its own persona
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
    # get_db creates an EMPTY database; this test makes its own persona
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


def test_changing_interests_drops_the_cached_explanations(tmp_path, monkeypatch):
    """An explanation cached under one persona outlived the persona.

    The cache key is (user_id, item_id) -- the interests text is not in it. Both
    sites that change interests call `forget_profile_vector` for the vector
    cache and leave this one alone, so `update_persona` promises "ranking
    re-embeds on next use", the ranking does, and the explanations then
    contradict it indefinitely.

    Structurally the same omission `forget_profile_vector`'s own docstring
    describes ("which is why update_persona was missed"), recurring for the
    second cache. delete_persona and reset_feedback both clear this table with
    a comment saying why; the update path was the gap.
    """
    from attestation.db import get_db
    from attestation.mcp.feed import _update_persona
    from attestation.rank import create_user

    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    conn = get_db(tmp_path / "t.db")
    user_id = create_user(conn, "ana", "protein structure, cryo-EM")
    conn.execute(
        "INSERT INTO items(feed_id, title, url, summary, content_hash)"
        " VALUES (NULL, 't', 'u', 's', 'h')"
    )
    conn.execute(
        "INSERT INTO explanations(user_id, item_id, text) VALUES (?, 1, 'about protein folding')",
        (user_id,),
    )
    conn.commit()
    conn.close()

    _update_persona("ana", "medieval poetry and 14th century manuscripts")

    conn = get_db(tmp_path / "t.db")
    left = conn.execute(
        "SELECT COUNT(*) FROM explanations WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    assert left == 0, f"{left} explanation(s) survived a change of interests"


def test_merging_personas_drops_the_loser_and_keeper_explanations(tmp_path, monkeypatch):
    """The merge path got the fix and not the test.

    `_update_persona`'s explanation delete is covered; `personas.merge`'s is
    not -- deleting that line leaves the whole suite green. The leak is real:
    after merging bob into alice, alice keeps "matches your interest in cryo-EM
    protein structure" under her new merged interests.

    Same reasoning as the sibling: the cache key is (user_id, item_id) with no
    interests in it, so an explanation outlives the profile that produced it.
    """
    from attestation.db import get_db
    from attestation.personas import merge
    from attestation.rank import create_user

    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    conn = get_db(tmp_path / "t.db")
    keeper = create_user(conn, "alice", "cryo-EM protein structure")
    loser = create_user(conn, "bob", "medieval poetry")
    conn.execute(
        "INSERT INTO items(feed_id, title, url, summary, content_hash)"
        " VALUES (NULL, 't', 'u', 's', 'h')"
    )
    for user_id in (keeper, loser):
        conn.execute(
            "INSERT INTO explanations(user_id, item_id, text)"
            " VALUES (?, 1, 'matches your interest in cryo-EM protein structure')",
            (user_id,),
        )
    conn.commit()

    merge(conn, into="alice", drop=["bob"])

    left = conn.execute("SELECT COUNT(*) FROM explanations").fetchone()[0]
    assert left == 0, f"{left} explanation(s) survived a merge that changed the interests"


def test_synthesize_profile_renders_through_profile_synthesis_messages():
    from attestation.explain import profile_synthesis_messages

    msgs = profile_synthesis_messages(["A paper", "B paper"])
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "Summarize this reader in one sentence."
    assert msgs[1]["content"] == "Recently useful titles:\n- A paper\n- B paper"


def test_explain_module_has_exactly_two_system_prompts_both_in_renderers():
    """The one-renderer rule, machine-checked: every `{"role": "system"}`
    literal in explain.py must live inside a function named `*_messages`, so
    a third inline prompt cannot creep back in beside the two renderers."""
    path = pathlib.Path(__file__).resolve().parent.parent / "src" / "attestation" / "explain.py"
    tree = ast.parse(path.read_text())

    def _in_messages_fn(node, ancestors):
        return any(
            isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef)) and a.name.endswith("_messages")
            for a in ancestors
        )

    found = []
    stack = [(tree, [])]
    while stack:
        node, ancestors = stack.pop()
        if (
            isinstance(node, ast.Dict)
            and any(
                isinstance(k, ast.Constant) and k.value == "role"
                for k in node.keys
                if k is not None
            )
            and any(
                isinstance(v, ast.Constant) and v.value == "system"
                for k, v in zip(node.keys, node.values)
                if isinstance(k, ast.Constant) and k.value == "role"
            )
        ):
            found.append(_in_messages_fn(node, ancestors))
        for child in ast.iter_child_nodes(node):
            stack.append((child, [*ancestors, node]))

    assert len(found) == 2, f"expected exactly two system-role prompts, found {len(found)}"
    assert all(found), "a system-role prompt literal lives outside a *_messages renderer"

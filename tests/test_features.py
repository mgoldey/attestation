import pytest
from pydantic import ValidationError

from attestation.db import get_db
from attestation.features import (
    ItemTags,
    pref_scores_for_items,
    run_tagging,
    tag_one_item,
    tag_vocabulary,
)


def add_item(conn, title, summary="s", days_ago=0):
    cur = conn.execute(
        "INSERT INTO items(feed_id, title, url, summary, published, content_hash)"
        " VALUES (NULL, ?, 'http://x', ?, datetime('now', ?), ?)",
        (title, summary, f"-{days_ago} days", f"hash-{title}"),
    )
    return cur.lastrowid


def get_item_row(conn, item_id):
    return conn.execute("SELECT id, title, summary FROM items WHERE id = ?", (item_id,)).fetchone()


def good_chat_fn(messages, schema):
    return {"content_type": "paper", "tags": ["quantum-chemistry", "dft"]}


def test_itemtags_validates_good_output():
    parsed = ItemTags.model_validate({"content_type": "paper", "tags": ["a-tag", "b2"]})
    assert parsed.content_type == "paper"
    assert parsed.tags == ["a-tag", "b2"]


@pytest.mark.parametrize(
    "bad",
    [
        {"content_type": "poem", "tags": ["ok"]},  # bad enum
        {"content_type": "paper", "tags": []},  # too few tags
        {"content_type": "paper", "tags": ["a", "b", "c", "d", "e"]},  # too many
        {"content_type": "paper", "tags": ["Bad Tag!"]},  # bad charset
        {"content_type": "paper", "tags": ["x" * 40]},  # too long
    ],
)
def test_itemtags_rejects_bad_output(bad):
    with pytest.raises(ValidationError):
        ItemTags.model_validate(bad)


def test_tag_one_item_writes_features_and_tags(tmp_path):
    conn = get_db(tmp_path / "t.db")
    item_id = add_item(conn, "DFT paper")
    ok = tag_one_item(conn, get_item_row(conn, item_id), good_chat_fn, [], "testmodel")
    assert ok is True
    feat = conn.execute(
        "SELECT content_type, model FROM item_features WHERE item_id = ?", (item_id,)
    ).fetchone()
    assert feat["content_type"] == "paper"
    assert feat["model"] == "testmodel"
    tags = {
        r["tag"] for r in conn.execute("SELECT tag FROM item_tags WHERE item_id = ?", (item_id,))
    }
    assert tags == {"quantum-chemistry", "dft"}


def test_tag_one_item_retries_once_then_succeeds(tmp_path):
    conn = get_db(tmp_path / "t.db")
    item_id = add_item(conn, "flaky")
    calls = {"n": 0}

    def flaky_chat_fn(messages, schema):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("garbled JSON")
        return {"content_type": "blog", "tags": ["mlops"]}

    assert tag_one_item(conn, get_item_row(conn, item_id), flaky_chat_fn, [], "m") is True
    assert calls["n"] == 2


def test_tag_one_item_gives_up_after_retry_writes_nothing(tmp_path):
    conn = get_db(tmp_path / "t.db")
    item_id = add_item(conn, "hopeless")

    def bad_chat_fn(messages, schema):
        # Unusable even after normalization: leading punctuation cannot be
        # folded away, unlike case or spaces. ("INVALID TAG" would now
        # normalize to "invalid-tag" and succeed, which is the point of
        # test_tags_are_normalized_rather_than_rejected below.)
        return {"content_type": "paper", "tags": ["!!!", "@@@"]}

    assert tag_one_item(conn, get_item_row(conn, item_id), bad_chat_fn, [], "m") is False
    feat_row = conn.execute("SELECT 1 FROM item_features WHERE item_id = ?", (item_id,)).fetchone()
    assert feat_row is None
    tag_row = conn.execute("SELECT 1 FROM item_tags WHERE item_id = ?", (item_id,)).fetchone()
    assert tag_row is None


def test_vocab_appears_in_prompt(tmp_path):
    conn = get_db(tmp_path / "t.db")
    item_id = add_item(conn, "vocab check")
    seen = {}

    def spy_chat_fn(messages, schema):
        seen["prompt"] = "\n".join(m["content"] for m in messages)
        return {"content_type": "paper", "tags": ["dft"]}

    tag_one_item(conn, get_item_row(conn, item_id), spy_chat_fn, ["dft", "llm-eval"], "m")
    assert "dft" in seen["prompt"] and "llm-eval" in seen["prompt"]


def test_tag_vocabulary_orders_by_use(tmp_path):
    conn = get_db(tmp_path / "t.db")
    ids = [add_item(conn, f"i{n}") for n in range(3)]
    for i in ids:
        conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, 'common')", (i,))
    conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, 'rare')", (ids[0],))
    assert tag_vocabulary(conn) == ["common", "rare"]
    assert tag_vocabulary(conn, limit=1) == ["common"]


def test_run_tagging_tags_all_untagged_then_is_idempotent(tmp_path):
    conn = get_db(tmp_path / "t.db")
    for n in range(3):
        add_item(conn, f"item {n}")
    stats = run_tagging(conn, chat_fn=good_chat_fn)
    assert (stats["tagged"], stats["failed"]) == (3, 0)
    again = run_tagging(conn, chat_fn=good_chat_fn)
    assert (again["tagged"], again["failed"]) == (0, 0)


def test_tags_are_normalized_rather_than_rejected():
    """Regression: TAG_PATTERN is lowercase-only, so every capitalised acronym
    the literature uses failed it -- and one bad tag rejected the WHOLE item,
    which then retried on every cron run forever. A real Nature paper sat
    untagged through a full re-tag because the model kept returning 'rRNA'
    alongside three good tags.
    """
    parsed = ItemTags.model_validate(
        {"content_type": "paper", "tags": ["biology", "rRNA", " Exosome ", "RNA-seq"]}
    )

    assert parsed.tags == ["biology", "rrna", "exosome", "rna-seq"]


def test_unusable_tags_are_dropped_but_the_item_survives():
    """A tag that is still malformed after folding is genuinely unusable and is
    dropped alone -- the item keeps whatever else was valid."""
    parsed = ItemTags.model_validate(
        {"content_type": "paper", "tags": ["biology", "!!!", "a" * 40, "genomics"]}
    )

    assert parsed.tags == ["biology", "genomics"]


def test_normalization_deduplicates():
    """Folding case can collide two tags into one; the pair key on item_tags
    would reject the duplicate insert."""
    parsed = ItemTags.model_validate(
        {"content_type": "paper", "tags": ["Biology", "biology", "BIOLOGY"]}
    )

    assert parsed.tags == ["biology"]


def test_provenance_tags_are_dropped():
    """`nature`, `science-feed` and `retraction` describe where an item came
    from, not what it is about. As tags they linked items that share a source
    rather than a topic -- on the live graph they pulled a journal feed into
    the `biology` cluster, making a publication look like a research area.
    The publication is already in items.feed_id.
    """
    parsed = ItemTags.model_validate(
        {"content_type": "paper", "tags": ["biology", "nature", "genomics", "retraction"]}
    )

    assert parsed.tags == ["biology", "genomics"]


def test_post_type_tags_are_dropped_since_content_type_records_them():
    """`release`/`announcement` duplicate content_type, which already holds 209
    and 241 items respectively."""
    parsed = ItemTags.model_validate(
        {"content_type": "release", "tags": ["pytorch", "release", "announcement"]}
    )

    assert parsed.tags == ["pytorch"]


def test_an_item_that_is_only_provenance_survives_with_no_tags():
    """Filtering to empty is NOT a failure. A release note really can be about
    nothing but its own release, and failing it would leave the item untagged
    and retried on every cron run forever -- the trap the all-or-nothing
    validator fell into. content_type still records what the item is.
    """
    parsed = ItemTags.model_validate(
        {"content_type": "release", "tags": ["release", "announcement", "update"]}
    )

    assert parsed.tags == []


def test_item_with_no_usable_tags_still_fails():
    """Normalization must not turn a genuinely empty result into a success."""
    with pytest.raises(ValidationError):
        ItemTags.model_validate({"content_type": "paper", "tags": ["!!!", "###"]})


def test_run_tagging_reports_the_model_it_used(tmp_path, monkeypatch):
    """item_features records a model per row, but nothing surfaced it, so a run
    against the wrong model looked exactly like a correct one. run_tagging now
    returns the model it resolved, and every row it writes must match it --
    resolved once up front rather than per item, so a mid-run environment
    change cannot split one run across two models.
    """
    monkeypatch.setenv("CHAT_MODEL", "test-model:1b")
    conn = get_db(tmp_path / "t.db")
    for n in range(3):
        add_item(conn, f"item {n}")

    stats = run_tagging(conn, chat_fn=good_chat_fn)

    assert stats["model"] == "test-model:1b"
    written = {r["model"] for r in conn.execute("SELECT DISTINCT model FROM item_features")}
    assert written == {"test-model:1b"}


def test_run_tagging_newest_first_and_limit(tmp_path):
    conn = get_db(tmp_path / "t.db")
    old = add_item(conn, "old", days_ago=5)
    new = add_item(conn, "new", days_ago=0)
    stats = run_tagging(conn, chat_fn=good_chat_fn, limit=1)
    assert stats["tagged"] == 1
    assert conn.execute("SELECT 1 FROM item_features WHERE item_id = ?", (new,)).fetchone()
    assert conn.execute("SELECT 1 FROM item_features WHERE item_id = ?", (old,)).fetchone() is None


def test_run_tagging_counts_failures_and_continues(tmp_path):
    conn = get_db(tmp_path / "t.db")
    add_item(conn, "will-fail")
    add_item(conn, "will-succeed")

    def chat_fn(messages, schema):
        if "will-fail" in messages[1]["content"]:
            raise ValueError("ollama down for this one")
        return {"content_type": "paper", "tags": ["dft"]}

    stats = run_tagging(conn, chat_fn=chat_fn)
    assert (stats["tagged"], stats["failed"]) == (1, 1)


def test_new_tags_enter_vocabulary_within_a_run(tmp_path):
    conn = get_db(tmp_path / "t.db")
    add_item(conn, "first", days_ago=0)
    add_item(conn, "second", days_ago=1)
    prompts = []

    def chat_fn(messages, schema):
        prompts.append(messages[1]["content"])
        return {"content_type": "paper", "tags": ["fresh-tag"]}

    run_tagging(conn, chat_fn=chat_fn)
    assert "fresh-tag" not in prompts[0]  # vocab empty on first item
    assert "fresh-tag" in prompts[1]  # second item sees the new tag


def _tag(conn, item_id, content_type, tags):
    conn.execute(
        "INSERT OR REPLACE INTO item_features(item_id, content_type, model) VALUES (?, ?, 'm')",
        (item_id, content_type),
    )
    for t in tags:
        conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, ?)", (item_id, t))


def _click(conn, user_id, item_id, useful):
    conn.execute(
        "INSERT INTO clicks(user_id, item_id, useful) VALUES (?, ?, ?)", (user_id, item_id, useful)
    )


def _matt(conn):
    return conn.execute("SELECT id FROM users WHERE name = 'matt'").fetchone()["id"]


def test_pref_neutral_with_no_data(tmp_path):
    conn = get_db(tmp_path / "t.db")
    a = add_item(conn, "a")  # untagged, no feed -> no keys at all
    b = add_item(conn, "b")
    _tag(conn, b, "paper", ["dft"])  # tagged but user has no clicks
    scores = pref_scores_for_items(conn, _matt(conn), [a, b])
    assert scores[0] == 0.5
    assert scores[1] == 0.5  # keys exist but all have (0,0) stats -> 0.5


def test_downvoted_tag_scores_below_neutral(tmp_path):
    conn = get_db(tmp_path / "t.db")
    uid = _matt(conn)
    bad1, bad2, candidate, control = (add_item(conn, t) for t in ("bad1", "bad2", "cand", "ctrl"))
    for i in (bad1, bad2, candidate):
        _tag(conn, i, "announcement", ["llm-benchmarks"])
    _tag(conn, control, "paper", ["dft"])
    _click(conn, uid, bad1, useful=0)
    _click(conn, uid, bad2, useful=0)
    scores = pref_scores_for_items(conn, uid, [candidate, control])
    # candidate shares tag AND content type with two downvotes: well below neutral
    assert scores[0] < 0.4
    assert scores[1] == 0.5  # control untouched by those clicks


def test_upvotes_score_above_neutral_and_mix_averages(tmp_path):
    conn = get_db(tmp_path / "t.db")
    uid = _matt(conn)
    liked, candidate = add_item(conn, "liked"), add_item(conn, "cand")
    _tag(conn, liked, "paper", ["dft"])
    _tag(conn, candidate, "paper", ["dft"])
    _click(conn, uid, liked, useful=1)
    assert pref_scores_for_items(conn, uid, [candidate])[0] > 0.5


def test_source_key_used_when_feed_present(tmp_path):
    conn = get_db(tmp_path / "t.db")
    uid = _matt(conn)
    conn.execute("INSERT INTO feeds(id, url, title) VALUES (7, 'http://f', 'Feed7')")

    def add_fed(title):
        return conn.execute(
            "INSERT INTO items(feed_id, title, url, summary, content_hash)"
            " VALUES (7, ?, 'http://x', 's', ?)",
            (title, f"h-{title}"),
        ).lastrowid

    clicked, candidate = add_fed("clicked"), add_fed("cand")
    _click(conn, uid, clicked, useful=0)  # downvote something from feed 7
    # candidate is UNTAGGED but shares the source -> sinks below neutral anyway
    assert pref_scores_for_items(conn, uid, [candidate])[0] < 0.5


def test_scores_are_per_user(tmp_path):
    conn = get_db(tmp_path / "t.db")
    uid = _matt(conn)
    other = conn.execute("SELECT id FROM users WHERE name = 'ml-engineer'").fetchone()["id"]
    a, b = add_item(conn, "a"), add_item(conn, "b")
    _tag(conn, a, "paper", ["dft"])
    _tag(conn, b, "paper", ["dft"])
    _click(conn, uid, a, useful=0)
    assert pref_scores_for_items(conn, other, [b])[0] == 0.5


def test_tag_vocabulary_excludes_non_topic_tags(tmp_path):
    """The prompt tells the model not to emit provenance tags, so suggesting
    them as vocabulary would work against it. They are low-use today, but a
    corpus with more release notes would push them into the top slots."""
    conn = get_db(tmp_path / "v.db")
    for i, tags in enumerate(
        [["nature", "biology"]] * 5 + [["release", "pytorch"]] * 4 + [["genomics"]], start=1
    ):
        conn.execute(
            "INSERT INTO items(id, feed_id, title, url, summary, content_hash)"
            " VALUES (?, NULL, ?, 'u', 's', ?)",
            (i, f"t{i}", f"h{i}"),
        )
        for t in tags:
            conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, ?)", (i, t))
    conn.commit()

    vocab = tag_vocabulary(conn)

    assert "nature" not in vocab and "release" not in vocab
    assert {"biology", "pytorch", "genomics"} <= set(vocab)
    conn.close()


def test_the_vocabulary_is_not_re_read_for_every_item(tmp_path):
    """tag_vocabulary is a full GROUP BY over item_tags plus a canonical()
    pass, so its cost tracks the whole archive rather than the run: measured on
    the live corpus at 12.3ms with 20k tag rows and 67.9ms with 163k. Calling
    it per item spent 68 seconds of database time in a 1000-item run, on top of
    inference.

    It must still be re-read when a genuinely new tag appears -- see
    test_new_tags_enter_vocabulary_within_a_run, which is the guarantee this
    optimisation must not break.
    """
    from attestation import features

    conn = get_db(tmp_path / "t.db")
    for i in range(12):
        add_item(conn, f"item-{i}", days_ago=i)
    # An established tag: already used by an item that is not in this run.
    conn.execute(
        "INSERT INTO items(feed_id, title, url, content_hash)"
        " VALUES (NULL, 'seed', 'http://seed', 'hseed')"
    )
    seed_id = conn.execute("SELECT id FROM items WHERE content_hash='hseed'").fetchone()["id"]
    conn.execute(
        "INSERT INTO item_features(item_id, content_type, model) VALUES (?, 'paper', 'm')",
        (seed_id,),
    )
    conn.execute("INSERT INTO item_tags(item_id, tag) VALUES (?, 'settled-tag')", (seed_id,))
    conn.commit()

    calls = {"n": 0}
    real = features.tag_vocabulary

    def counting(c, limit=150):
        calls["n"] += 1
        return real(c, limit)

    features.tag_vocabulary = counting
    try:
        run_tagging(conn, chat_fn=lambda m, s: {"content_type": "paper", "tags": ["settled-tag"]})
    finally:
        features.tag_vocabulary = real

    assert calls["n"] <= 2, (
        f"tag_vocabulary ran {calls['n']} times for 12 items that minted no new"
        " tag -- the per-item re-read is back"
    )

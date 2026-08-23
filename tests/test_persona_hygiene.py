"""Personas and the provenance of their feedback.

Two problems this covers. Seed data written before the `clicks.source` column
existed is labelled `ui`, so bootstrap rows -- whose labels are a linear
threshold on the very embedding the classifier consumes -- look like a person
pressing a button. `evaluate_user` excludes `bootstrap` precisely to avoid
scoring against them, and that exclusion silently does nothing for mislabelled
rows.

And a database accumulates personas: near-duplicate interests, demo seeds, and
accounts that never got a single click. Every one of them is a choice the agent
has to make when a user says "my feed" without naming a persona.
"""

import pytest

from attestation import personas
from attestation.db import get_db
from attestation.rank import get_user, record_click


@pytest.fixture
def seeded(tmp_path):
    conn = get_db(tmp_path / "t.db")
    conn.execute("DELETE FROM users")
    conn.execute("INSERT INTO users(id, name, interests) VALUES (1, 'real', 'protein folding')")
    conn.execute("INSERT INTO users(id, name, interests) VALUES (2, 'seeded', 'catalysis')")
    conn.execute("INSERT INTO users(id, name, interests) VALUES (3, 'empty', 'polymers')")
    for i in range(1, 41):
        conn.execute(
            "INSERT INTO items(id, feed_id, title, url, summary, content_hash)"
            " VALUES (?, NULL, ?, 'u', 's', ?)",
            (i, f"t{i}", f"h{i}"),
        )
    conn.commit()
    return conn


def test_bootstrap_rows_mislabelled_as_ui_are_detected(seeded):
    """The signature: k clicks at one instant, split exactly half and half.

    A person cannot press 30 buttons in the same second, and real feedback is
    not perfectly balanced. bootstrap_persona writes exactly k//2 useful and
    k//2 not-useful in one transaction.
    """
    uid = get_user(seeded, "seeded")["id"]
    for i in range(1, 31):
        seeded.execute(
            "INSERT INTO clicks(user_id, item_id, useful, source, clicked_at)"
            " VALUES (?, ?, ?, 'ui', '2026-08-04 22:25:12')",
            (uid, i, 1 if i <= 15 else 0),
        )
    # A genuine click history: spread over time, unbalanced.
    real = get_user(seeded, "real")["id"]
    for n, i in enumerate(range(31, 39)):
        seeded.execute(
            "INSERT INTO clicks(user_id, item_id, useful, source, clicked_at)"
            " VALUES (?, ?, 1, 'ui', ?)",
            (real, i, f"2026-08-0{n + 1} 10:00:00"),
        )
    seeded.commit()

    suspect = personas.mislabelled_bootstrap(seeded)
    assert "seeded" in suspect, "a 30-click same-instant 15/15 block was not flagged"
    assert "real" not in suspect, "genuine spread-out clicks were flagged as synthetic"


def test_relabelling_moves_them_out_of_the_trusted_sources(seeded):
    uid = get_user(seeded, "seeded")["id"]
    for i in range(1, 31):
        seeded.execute(
            "INSERT INTO clicks(user_id, item_id, useful, source, clicked_at)"
            " VALUES (?, ?, ?, 'ui', '2026-08-04 22:25:12')",
            (uid, i, 1 if i <= 15 else 0),
        )
    seeded.commit()

    moved = personas.relabel_bootstrap(seeded)
    assert moved == 30
    sources = {
        r["source"]
        for r in seeded.execute("SELECT DISTINCT source FROM clicks WHERE user_id = ?", (uid,))
    }
    assert sources == {"bootstrap"}


def test_relabelling_is_idempotent(seeded):
    uid = get_user(seeded, "seeded")["id"]
    for i in range(1, 31):
        seeded.execute(
            "INSERT INTO clicks(user_id, item_id, useful, source, clicked_at)"
            " VALUES (?, ?, ?, 'ui', '2026-08-04 22:25:12')",
            (uid, i, 1 if i <= 15 else 0),
        )
    seeded.commit()
    assert personas.relabel_bootstrap(seeded) == 30
    assert personas.relabel_bootstrap(seeded) == 0


def test_an_unused_persona_is_reported_with_its_nearest_neighbour(seeded):
    """Consolidation needs to say WHAT to merge into, not just what is unused."""
    record_click(seeded, get_user(seeded, "real")["id"], 1, True)
    seeded.commit()

    report = personas.survey(seeded)
    by_name = {p["name"]: p for p in report}
    assert by_name["empty"]["clicks"] == 0
    assert by_name["real"]["clicks"] == 1
    assert "nearest" in by_name["empty"], "an unused persona needs a merge candidate"


def test_survey_separates_trusted_from_synthetic_counts(seeded):
    """A persona with 30 bootstrap clicks and 0 real ones is untrained, and a
    bare count says the opposite."""
    uid = get_user(seeded, "seeded")["id"]
    for i in range(1, 31):
        record_click(seeded, uid, i, i <= 15, source="bootstrap")
    seeded.commit()

    row = next(p for p in personas.survey(seeded) if p["name"] == "seeded")
    assert row["clicks"] == 30
    assert row["trainable"] == 0, "bootstrap rows must not count as training signal"


def test_merging_moves_feedback_and_unions_interests(seeded):
    """A merge must not silently drop the feedback it was done to consolidate."""
    keep = get_user(seeded, "real")["id"]
    drop = get_user(seeded, "seeded")["id"]
    record_click(seeded, keep, 1, True)
    record_click(seeded, drop, 2, True)
    record_click(seeded, drop, 3, False)
    seeded.commit()

    out = personas.merge(seeded, into="real", drop=["seeded"])

    assert out["moved"] == 2
    assert get_user(seeded, "seeded") is None, "the merged-away persona still exists"
    kept = get_user(seeded, "real")
    assert "catalysis" in kept["interests"], "the dropped persona's interests were lost"
    assert "protein folding" in kept["interests"]
    assert (
        seeded.execute("SELECT COUNT(*) n FROM clicks WHERE user_id = ?", (kept["id"],)).fetchone()[
            "n"
        ]
        == 3
    )


def test_a_conflicting_verdict_keeps_the_survivors(seeded):
    """Both personas rated the same item differently. clicks has a UNIQUE
    (user_id, item_id), so one verdict has to win -- it must be the surviving
    persona's, and the loss must be reported rather than swallowed."""
    keep = get_user(seeded, "real")["id"]
    drop = get_user(seeded, "seeded")["id"]
    record_click(seeded, keep, 5, True)
    record_click(seeded, drop, 5, False)
    seeded.commit()

    out = personas.merge(seeded, into="real", drop=["seeded"])

    assert out["conflicts"] == 1
    row = seeded.execute(
        "SELECT useful FROM clicks WHERE user_id = ? AND item_id = 5", (keep,)
    ).fetchone()
    assert row["useful"] == 1, "the surviving persona's own verdict was overwritten"


def test_merging_into_an_unknown_persona_is_refused(seeded):
    with pytest.raises(ValueError, match="unknown"):
        personas.merge(seeded, into="nobody", drop=["seeded"])


def test_a_persona_cannot_be_merged_into_itself(seeded):
    with pytest.raises(ValueError, match="itself"):
        personas.merge(seeded, into="real", drop=["real"])


def test_merging_drops_exact_duplicate_interests(seeded):
    """A merge concatenates interests, and two personas describing one person
    repeat themselves. `matt` came out of its merge carrying `quantum
    chemistry` twice and both `LLM systems` and `LLM`.

    Only EXACT repeats are dropped. Near-duplicates stay: measured on the live
    profile, the redundant string scored matt's real positives at 0.588 mean
    similarity against 0.568 for a hand-tightened rewrite and 0.582 for an
    aggressive dedup. Repetition weights the embedding toward what the reader
    actually clicks, so tidying it makes the ranking worse.
    """
    seeded.execute("UPDATE users SET interests = 'catalysis, spectroscopy' WHERE name = 'seeded'")
    seeded.execute("UPDATE users SET interests = 'protein folding, catalysis' WHERE name = 'real'")
    seeded.commit()

    out = personas.merge(seeded, into="real", drop=["seeded"])

    parts = [p.strip() for p in out["interests"].split(",")]
    assert parts.count("catalysis") == 1, f"exact duplicate survived: {out['interests']}"
    assert "protein folding" in parts
    assert "spectroscopy" in parts


def test_merging_does_not_mangle_words(seeded):
    """A first attempt stripped a leading "and " with str.lstrip, which strips
    a CHARACTER SET -- `attention` became `ttention` and `diarization` became
    `iarization`. A profile is embedded text; corrupting it corrupts ranking.
    """
    seeded.execute("UPDATE users SET interests = 'attention, diarization' WHERE name = 'seeded'")
    seeded.execute("UPDATE users SET interests = 'agents, and photovoltaics' WHERE name = 'real'")
    seeded.commit()

    out = personas.merge(seeded, into="real", drop=["seeded"])

    # Compare the PARTS, not substrings: "ttention" is inside "attention",
    # so a naive `not in` on the whole string always passes and proves nothing.
    parts = {p.strip() for p in out["interests"].split(",")}
    assert "attention" in parts
    assert "diarization" in parts
    assert "ttention" not in parts
    assert "iarization" not in parts
    # A dangling conjunction from a merged tail is noise in an embedded string.
    assert "and photovoltaics" not in out["interests"]
    assert "photovoltaics" in out["interests"]


def test_a_persona_name_that_is_only_whitespace_is_refused(tmp_path, monkeypatch):
    """personas.py's comment says "a duplicate or empty name is the caller's to
    fix", and only the duplicate half was enforced -- by the UNIQUE constraint,
    not by code. A persona named '   ' persisted and then appeared in every
    "unknown user ... Valid users:" list, corrupting the one message the rest
    of this surface relies on to be actionable.
    """
    monkeypatch.setenv("RSS_DB", str(tmp_path / "t.db"))
    from attestation.mcp.personas import _create_persona

    for bad in ("", "   ", "\t\n"):
        out = _create_persona(name=bad, interests="science")
        assert out["ok"] is False, f"{bad!r} was accepted as a persona name"
        assert "must not be empty" in out["message"], out["message"]

    assert _create_persona(name="real-reader", interests="science")["ok"] is True

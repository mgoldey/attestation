"""Feedback inferred from what a reader already did, not from what they said.

Explicit feedback is a channel almost nobody uses. This project's own database
holds 68 UI clicks and 2 agent clicks against 5,167 items, ever -- roughly one
opinion per seventy-five items, and all of them positive, which is why the
click classifier has never fired for a real account.

Meanwhile 99 explanation requests sit in `explanations`, and when this module
was written 91 of them were for items that were never rated. Running `harvest`
converted 34, so the live figure moves as the tool is used -- these counts say
why the module exists, not what the database holds now.

Asking "why is this here?" is engagement. It is weaker
evidence than a stated opinion -- curiosity is not approval, and a reader may
well have asked precisely because the item looked wrong -- but it is real, it
is already collected, and counting it costs the reader nothing.

**The asymmetry is deliberate.** Only positives are inferred. There is no
behaviour in this system that reliably means "not useful": not opening an item
could mean bored, busy, or already-knew-it. Inferring rejection from silence
would fill the negative class with noise, and the negative class is the one
the classifier is starving for -- so it must come from a real judgement
(`ui`, `agent`) or an argued one (`simulated`), never from an absence.

Rows are written with `source='implicit'` so a later evaluation can weight or
exclude them, the way `evaluate_user` excludes `bootstrap`.
"""

import sqlite3

SOURCE = "implicit"


def candidates(conn: sqlite3.Connection, user_id: int) -> list[int]:
    """Items this user asked about but never rated.

    The LEFT JOIN is the point: an item already carrying a click is excluded
    here rather than being overwritten later, so a stated "not useful" is never
    flipped to useful by the reader's own curiosity.
    """
    return [
        r["item_id"]
        for r in conn.execute(
            "SELECT e.item_id FROM explanations e"
            " LEFT JOIN clicks c ON c.user_id = e.user_id AND c.item_id = e.item_id"
            " WHERE e.user_id = ? AND c.id IS NULL"
            " ORDER BY e.item_id",
            (user_id,),
        )
    ]


def harvest(conn: sqlite3.Connection, user_name: str) -> dict:
    """Record a weak positive for every unrated item this user asked about.

    Idempotent: an item is a candidate only while it has no click, so a second
    run over the same history records nothing.
    """
    from attestation.rank import get_user

    user = get_user(conn, user_name)
    if user is None:
        raise ValueError(f"unknown user: {user_name!r}")

    asked = conn.execute(
        "SELECT COUNT(*) n FROM explanations WHERE user_id = ?", (user["id"],)
    ).fetchone()["n"]
    fresh = candidates(conn, user["id"])

    # One transaction, not one per row. record_click commits on every call, so
    # this was N commits: 600 rows took 4.8s against a busy_timeout of 5s,
    # which makes a concurrent write GUARANTEED to be refused rather than
    # merely unlucky -- and the user-visible failure is a reader's explicit
    # "not useful" coming back as "the database is busy" while a background
    # harvest runs. The candidate set has no upper bound.
    #
    # The source enum is still enforced: SOURCE is a module constant checked
    # against CLICK_SOURCES at import, not caller input.
    from attestation.rank import CLICK_SOURCES

    if SOURCE not in CLICK_SOURCES:  # pragma: no cover - guards a typo here
        raise ValueError(f"invalid click source: {SOURCE!r}")
    conn.executemany(
        "INSERT OR REPLACE INTO clicks(user_id, item_id, useful, source) VALUES (?, ?, 1, ?)",
        [(user["id"], item_id, SOURCE) for item_id in fresh],
    )
    conn.commit()

    return {
        "candidates": len(fresh),
        "recorded": len(fresh),
        # Reported rather than silently dropped: a reader who rated everything
        # they asked about has no implicit signal left to give, and that is a
        # different situation from having asked about nothing.
        "skipped_already_rated": asked - len(fresh),
    }

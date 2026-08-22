"""Persona hygiene: what a database of readers accumulates, and what to do about it.

Two problems, both discovered on the live database rather than imagined.

**Feedback whose provenance is a lie.** `clicks.source` was added by migration
001; rows written before it defaulted to `'ui'`. Two personas here carry 30
clicks each, all at a single timestamp, split exactly 15 useful and 15 not --
`bootstrap_persona`'s signature at its default `k=30`, since it takes the top
k/2 by profile similarity as useful and the bottom k/2 as not. A person cannot
press thirty buttons in one second, and real feedback is never perfectly
balanced.

That matters because `evaluate_user` excludes `source='bootstrap'` on the
grounds that those labels are a linear threshold on the very embedding the
classifier consumes, making any score over them a tautology. The exclusion
silently does nothing for rows mislabelled as `ui`.

**Personas nobody uses.** Eight exist here; four have never been clicked, and
two pairs overlap heavily in interests. Every one is a choice an agent must
make when someone says "my feed" without naming one, and an unused persona
ranks by its interests string alone forever.

Nothing here deletes anything. `survey` reports, `relabel_bootstrap` corrects
provenance in place, and merging is left to a caller who can ask.
"""

import sqlite3

# bootstrap_persona's default k. A block of exactly this many clicks, all at
# one timestamp and evenly split, is its fingerprint.
_BOOTSTRAP_K = 30

# Sources whose labels are worth training on. `bootstrap` is excluded for the
# reason evaluate_user gives; the rest are either a person's judgement or an
# argued one.
TRAINABLE_SOURCES = ("ui", "agent", "simulated", "implicit")


def mislabelled_bootstrap(conn: sqlite3.Connection) -> dict[str, int]:
    """Persona name -> count of clicks that look seeded but claim to be real.

    Three signals together, because any one alone has honest explanations: a
    single timestamp (a person clicks over time), an even split (real feedback
    is lopsided -- every genuine account in this database is all-positive), and
    a block size matching the seeder's default.
    """
    out: dict[str, int] = {}
    rows = conn.execute(
        "SELECT u.name, k.clicked_at, COUNT(*) n,"
        "       SUM(k.useful) pos, SUM(1 - k.useful) neg"
        "  FROM clicks k JOIN users u ON u.id = k.user_id"
        " WHERE k.source = 'ui'"
        " GROUP BY k.user_id, k.clicked_at"
    ).fetchall()
    for r in rows:
        even_split = r["pos"] == r["neg"]
        if r["n"] >= _BOOTSTRAP_K and even_split:
            out[r["name"]] = out.get(r["name"], 0) + r["n"]
    return out


def relabel_bootstrap(conn: sqlite3.Connection) -> int:
    """Move mislabelled seed rows to `source='bootstrap'`. Returns rows changed.

    Idempotent: a second run finds nothing, because the rows no longer claim
    to be `ui`.
    """
    suspects = mislabelled_bootstrap(conn)
    if not suspects:
        return 0
    changed = 0
    for name in suspects:
        cur = conn.execute(
            "UPDATE clicks SET source = 'bootstrap'"
            " WHERE source = 'ui'"
            "   AND user_id = (SELECT id FROM users WHERE name = ?)"
            "   AND clicked_at IN ("
            "     SELECT clicked_at FROM clicks WHERE user_id ="
            "       (SELECT id FROM users WHERE name = ?)"
            "     GROUP BY clicked_at"
            "     HAVING COUNT(*) >= ? AND SUM(useful) = SUM(1 - useful))",
            (name, name, _BOOTSTRAP_K),
        )
        changed += cur.rowcount
    conn.commit()
    return changed


def _overlap(a: str, b: str) -> float:
    """Jaccard over interest words. Crude on purpose -- it ranks merge
    candidates for a human to confirm, it does not decide anything."""
    stop = {"and", "of", "in", "for", "the", "with", "to", "a"}
    wa = {w.strip(",").lower() for w in (a or "").split()} - stop
    wb = {w.strip(",").lower() for w in (b or "").split()} - stop
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def survey(conn: sqlite3.Connection) -> list[dict]:
    """Every persona with what it actually has to rank on.

    `clicks` is the raw count and `trainable` excludes bootstrap rows, because
    a persona with 30 seeded clicks and no real ones is untrained and a bare
    count says the opposite. Personas with no trainable feedback carry their
    nearest neighbour by interest overlap, so a caller proposing a merge has a
    target rather than just a complaint.
    """
    users = conn.execute("SELECT id, name, interests FROM users ORDER BY id").fetchall()
    placeholders = ",".join("?" * len(TRAINABLE_SOURCES))
    out = []
    for u in users:
        total = conn.execute(
            "SELECT COUNT(*) n FROM clicks WHERE user_id = ?", (u["id"],)
        ).fetchone()["n"]
        trainable = conn.execute(
            f"SELECT COUNT(*) n, SUM(useful) pos FROM clicks"
            f" WHERE user_id = ? AND source IN ({placeholders})",
            (u["id"], *TRAINABLE_SOURCES),
        ).fetchone()
        row = {
            "name": u["name"],
            "interests": u["interests"],
            "clicks": total,
            "trainable": trainable["n"],
            "trainable_positive": trainable["pos"] or 0,
            # Both classes are what the click classifier needs to fire at all.
            "classifier_ready": bool(
                trainable["n"] and 0 < (trainable["pos"] or 0) < trainable["n"]
            ),
        }
        if not trainable["n"]:
            others = [o for o in users if o["id"] != u["id"]]
            if others:
                best = max(others, key=lambda o: _overlap(u["interests"], o["interests"]))
                row["nearest"] = best["name"]
                row["nearest_overlap"] = round(_overlap(u["interests"], best["interests"]), 3)
        out.append(row)
    return out


def merge(conn: sqlite3.Connection, *, into: str, drop: list[str]) -> dict:
    """Fold personas into one, keeping their feedback and their interests.

    Destructive and deliberately narrow: it moves clicks and explanations,
    unions the interests text, deletes the emptied personas, and reports what
    it had to resolve. It does not decide WHICH personas to merge -- interest
    overlap is a weak signal (`Matthew Goldey` and `matt` share 0.083 by
    Jaccard while plainly being one person), so that judgement stays with a
    caller who can ask.

    `clicks` has a UNIQUE(user_id, item_id), so when both personas rated the
    same item one verdict must lose. The surviving persona's own verdict wins
    -- it is the account that will keep being used -- and the count of
    discarded verdicts is returned rather than swallowed, because a merge that
    quietly changes a rating is worse than one that refuses.
    """
    from attestation.rank import forget_profile_vector, get_user

    keeper = get_user(conn, into)
    if keeper is None:
        raise ValueError(f"unknown persona: {into!r}")
    if into in drop:
        raise ValueError(f"cannot merge {into!r} into itself")

    moved = conflicts = 0
    interests = [keeper["interests"] or ""]

    for name in drop:
        loser = get_user(conn, name)
        if loser is None:
            raise ValueError(f"unknown persona: {name!r}")

        conflicts += conn.execute(
            "SELECT COUNT(*) n FROM clicks a JOIN clicks b"
            "  ON a.item_id = b.item_id AND a.user_id = ? AND b.user_id = ?",
            (keeper["id"], loser["id"]),
        ).fetchone()["n"]

        # INSERT OR IGNORE, not REPLACE: the keeper's verdict stands.
        cur = conn.execute(
            "INSERT OR IGNORE INTO clicks(user_id, item_id, useful, clicked_at, source)"
            " SELECT ?, item_id, useful, clicked_at, source FROM clicks WHERE user_id = ?",
            (keeper["id"], loser["id"]),
        )
        moved += cur.rowcount
        conn.execute(
            "INSERT OR IGNORE INTO explanations(user_id, item_id, text)"
            " SELECT ?, item_id, text FROM explanations WHERE user_id = ?",
            (keeper["id"], loser["id"]),
        )
        conn.execute("DELETE FROM clicks WHERE user_id = ?", (loser["id"],))
        conn.execute("DELETE FROM explanations WHERE user_id = ?", (loser["id"],))
        conn.execute("DELETE FROM users WHERE id = ?", (loser["id"],))
        forget_profile_vector(conn, loser["id"])
        if loser["interests"]:
            interests.append(loser["interests"])

    merged_interests = ", ".join(t.strip().rstrip(",") for t in interests if t.strip())
    conn.execute("UPDATE users SET interests = ? WHERE id = ?", (merged_interests, keeper["id"]))
    # The keeper's cached profile vector was computed from the old text.
    forget_profile_vector(conn, keeper["id"])
    conn.commit()
    return {
        "into": into,
        "dropped": drop,
        "moved": moved,
        "conflicts": conflicts,
        "interests": merged_interests,
    }

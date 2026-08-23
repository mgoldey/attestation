"""Persona lifecycle behind the `feed.persona_*` tools.

Split out of feed.py, which held six concerns and hit its size cap three times
in one day. The cap's own comment named this seam and said the next tool to
land here should come with a split rather than another raised number.

The tools stay in the `feed.*` namespace: personas exist to own feed feedback,
and a reader who wanted `persona.*` would be looking for a second surface that
does not exist. This is a source split, not a namespace change.
"""

from attestation.mcp._tool import ToolError, tool
from attestation.rank import forget_profile_vector


@tool(empty={"user_id": None}, label="create_persona")
def _create_persona(conn, name: str, interests: str) -> dict:
    from attestation.rank import create_user

    try:
        uid = create_user(conn, name, interests)
    except ValueError as exc:
        # a duplicate or empty name is the caller's to fix, so the reason is
        # surfaced rather than flattened to "internal error"
        raise ToolError(str(exc)) from exc
    return {
        "user_id": uid,
        "message": (
            f"created persona {name!r}. Ranking starts from its interests text; "
            "feed.rate calls will personalize it from the first click."
        ),
    }


@tool(needs_user=True, label="update_persona")
def _update_persona(conn, user_row, interests: str) -> dict:
    name = user_row["name"]
    conn.execute("UPDATE users SET interests = ? WHERE id = ?", (interests, user_row["id"]))
    conn.commit()
    # Without this, the embedder-down fallback in rank._profile_vector would
    # serve a vector computed from the interests text just replaced -- it
    # returns a cached entry without comparing hashes, by design.
    forget_profile_vector(conn, user_row["id"])
    # Explanations too. Their cache key is (user_id, item_id) with no interests
    # in it, so they outlive the persona that produced them: this message
    # promises the ranking re-embeds, it does, and the cached explanations then
    # contradict it indefinitely. delete_persona and reset_feedback both clear
    # this table; the update path was the gap.
    dropped = conn.execute("DELETE FROM explanations WHERE user_id = ?", (user_row["id"],)).rowcount
    conn.commit()
    note = f"; dropped {dropped} stale explanation(s)" if dropped else ""
    return {"message": f"updated interests for {name!r}; ranking re-embeds on next use{note}"}


@tool(empty={"prevalent_tags": []}, label="propose_interests")
def _propose_interests(conn, limit: int = 12) -> dict:
    from attestation.features import tag_vocabulary

    # tag_vocabulary, not a raw GROUP BY. This grouped item_tags directly, so
    # on the live corpus it proposed `llm` and `machinelearning` -- fragments
    # kg.canonical folds into large-language-models and machine-learning --
    # and pushed natural-language-processing out of the list entirely.
    #
    # This string becomes a persona's interests text, which IS the profile
    # embedding, and autocreate_user already uses the canonicalising version.
    # Two tools answering "what should a new reader follow" must not disagree,
    # least of all with the one an agent is told to call being the wrong one.
    tags = tag_vocabulary(conn, limit=limit)
    return {
        "prevalent_tags": tags,
        "message": (
            "most common tags in the current feed; combine the relevant ones into "
            "an interests string and pass it to create_persona"
        ),
    }


@tool(needs_user=True, label="delete_persona")
def _delete_persona(conn, user_row, confirm: bool = False) -> dict:
    name = user_row["name"]
    n = conn.execute(
        "SELECT COUNT(*) c FROM clicks WHERE user_id = ?", (user_row["id"],)
    ).fetchone()["c"]
    if not confirm:
        raise ToolError(
            f"refusing to delete {name!r} without confirm=true. This would "
            f"permanently remove the persona and its {n} click(s) of training data."
        )
    conn.execute("DELETE FROM clicks WHERE user_id = ?", (user_row["id"],))
    # users.id is a rowid alias SQLite reuses after the highest-id row is
    # deleted -- without this, a future persona created at the same id
    # would inherit this persona's cached explanations verbatim.
    conn.execute("DELETE FROM explanations WHERE user_id = ?", (user_row["id"],))
    conn.execute("DELETE FROM users WHERE id = ?", (user_row["id"],))
    conn.commit()
    forget_profile_vector(conn, user_row["id"])
    return {"message": f"deleted persona {name!r} and its {n} click(s)"}


@tool(needs_user=True, label="reset_feedback")
def _reset_feedback(conn, user_row, confirm: bool = False) -> dict:
    name = user_row["name"]
    n = conn.execute(
        "SELECT COUNT(*) c FROM clicks WHERE user_id = ?", (user_row["id"],)
    ).fetchone()["c"]
    if not confirm:
        raise ToolError(
            f"refusing to reset {name!r} without confirm=true. This would erase "
            f"{n} click(s); the persona and its interests text would be kept."
        )
    explained = conn.execute(
        "SELECT COUNT(*) c FROM explanations WHERE user_id = ?", (user_row["id"],)
    ).fetchone()["c"]
    conn.execute("DELETE FROM clicks WHERE user_id = ?", (user_row["id"],))
    # Explanations go too. implicit.harvest reads an explanation with no click
    # as a weak positive, and its docstring promises "a stated 'not useful' is
    # never flipped to useful by the reader's own curiosity" -- a promise the
    # LEFT JOIN keeps only while the click exists. Clearing clicks alone
    # removed the guard's evidence and not its subject, so the next harvest
    # resurrected every cleared rating as useful=1. Negatives are the class
    # this ranker is starving for, so that is the worst available direction.
    conn.execute("DELETE FROM explanations WHERE user_id = ?", (user_row["id"],))
    conn.commit()
    forget_profile_vector(conn, user_row["id"])
    return {"message": f"cleared {n} click(s) and {explained} explanation(s) for {name!r}"}

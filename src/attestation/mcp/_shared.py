"""Helpers used by more than one domain module.

Kept deliberately small. Anything only one domain needs lives in that domain's
module -- a shared-helpers file is where god objects reconstitute themselves
after a split.
"""

import logging

from attestation.ports import EmbedderPort
from attestation.rank import rank_items

log = logging.getLogger(__name__)

MAX_LIST_LIMIT = 50

_embedder = None


def get_embedder() -> EmbedderPort:
    """Lazily built and shared across calls -- it is just an httpx client.

    Constructed on first use rather than at import so that starting the server
    (or importing it in a test) never reaches for a model.
    """
    global _embedder
    if _embedder is None:
        from attestation.embed import Embedder

        _embedder = Embedder()
    return _embedder


def ranked_items(conn, user_row, limit: int, since_days: int | None) -> list:
    """Rank items for an already-resolved user row against a connection the
    caller owns. Shared by list_feed and digest so digest does not open a
    second connection to rank the same feed."""
    return rank_items(conn, get_embedder(), user_row["id"], since_days)[:limit]


def ranking_quality(conn, user_id: int) -> dict:
    """How much to trust the ordering, stated up front.

    A digest built from an untrained ranker looks exactly like one built from a
    good one. rank.classifier_probs returns None when a user's clicks are all
    one class (rank.py's single-class guard), so the click-CLASSIFIER term
    never fires -- but rank_items blends in a second, independent term
    (avg_ranks over pref_scores_for_items) whenever n_clicks > 0, regardless of
    the guard. So a single-class history with at least one click is NOT pure
    embedding similarity: the feature-preference term still contributes, only
    the classifier is silent. Naming which terms are actually contributing
    matters more than a blanket "profile-embedding only" claim, which is wrong
    in exactly the case this caveat exists to describe.
    """
    rows = conn.execute(
        "SELECT useful, COUNT(*) n FROM clicks WHERE user_id = ? GROUP BY useful",
        (user_id,),
    ).fetchall()
    counts = {int(r["useful"]): r["n"] for r in rows}
    total = sum(counts.values())
    active = len(counts) > 1
    out = {
        "clicks": total,
        "useful": counts.get(1, 0),
        "not_useful": counts.get(0, 0),
        "classifier_active": active,
    }
    if not active:
        if total > 0:
            out["caveat"] = (
                f"ranking is running WITHOUT its click classifier: {total} click(s), "
                f"all {'useful' if counts.get(1) else 'not-useful'}. Order blends "
                "profile-embedding similarity with a feature-preference term learned "
                "from those clicks -- the classifier term is silent (needs both "
                "useful and not-useful clicks to fire), but the preference term is "
                "still contributing. Mark some items the other way to train the "
                "classifier too."
            )
        else:
            out["caveat"] = (
                "ranking is running WITHOUT its click classifier or any "
                "feature-preference signal: 0 clicks recorded. Order is "
                "profile-embedding similarity only."
            )
    elif total < 20:
        out["caveat"] = f"only {total} clicks: the classifier is active but weakly trained"
    return out

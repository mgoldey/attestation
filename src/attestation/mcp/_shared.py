"""Helpers used by more than one domain module.

Kept deliberately small. Anything only one domain needs lives in that domain's
module -- a shared-helpers file is where god objects reconstitute themselves
after a split.
"""

import logging
from typing import Annotated

from pydantic import Field

from attestation.mcp._tool import ToolError
from attestation.ports import EmbedderPort
from attestation.rank import rank_items, ranking_quality

log = logging.getLogger(__name__)

# 18, from measurement. A feed row costs ~370 chars emitted, so 50 -- the old
# value -- produced 18457 for feed.list and 22530 for feed.search, against a
# 7000-char ceiling. Every size guard drove the DEFAULT, so a limit the schema
# advertised and no test exercised blew it silently.
#
# The number is what fits, not a preference: 16 rows plus envelope lands just
# under (measured: 18 gave 7130, 130 over). A caller who wants 50 items is asking for a payload that
# does not survive the trip, and paging is the honest answer.
MAX_LIST_LIMIT = 16

# Argument constraints live in the SCHEMA, not just in runtime checks, so an
# MCP client rejects a bad call before it is made. Shared here because three
# domain modules declare bounded arguments.
Limit = Annotated[int, Field(ge=1, le=MAX_LIST_LIMIT)]


def clamp_limit(limit: int) -> int:
    """A caller's `limit`, capped at MAX_LIST_LIMIT. Refuses nonsense.

    Asking for more than the cap is a reasonable request with a reasonable
    answer, so it is capped silently. Asking for zero or fewer is not a
    request at all, and the previous behaviour -- clamping up to 1 -- answered
    a question nobody asked. An agent that sends limit=0 has a bug, and being
    told is more useful than being handed one item.
    """
    limit = int(limit)
    if limit < 1:
        raise ToolError(f"limit must be at least 1, got {limit}")
    return min(limit, MAX_LIST_LIMIT)


def validate_window(since_days: int | None) -> int | None:
    """`since_days` bounds how far BACK a feed reaches, so a negative value
    asks for items published in the future.

    That returned an empty list with ok=true, and an agent reading it told the
    reader there was nothing to read -- a wrong answer dressed as a real one.
    Zero is allowed and means today.
    """
    if since_days is not None and int(since_days) < 0:
        raise ToolError(
            f"since_days must be zero or positive, got {since_days}"
            " -- it is a window into the past; pass None for no limit"
        )
    return since_days


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


# ranking_quality now lives in rank.py -- the web UI needs it too, and
# importing an mcp/ module from server.py would invert the layering.
__all__ = ["ranking_quality"]


# Bounded numeric arguments, declared once so a new tool inherits the bound.
# A schema constraint is a client-side reject: the caller is told the argument
# is wrong instead of the tool failing on it. `limit=0` and `since_days=-30`
# both reached a tool body before these existed.
ItemId = Annotated[int, Field(ge=1, description="row id from a previous result")]
SinceDays = Annotated[int | None, Field(ge=1, le=3650, description="lookback window in days")]

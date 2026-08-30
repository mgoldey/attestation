"""MCP stdio server: exposes attestation as native tool-calling for MCP clients.

The tools themselves live in `attestation.mcp`, one module per domain. This
file is the entry point and nothing else.

Each tool opens its own short-lived DB connection via resolve_db_path(None), so
this process stays stateless between calls and honours ATTEST_DB like the CLI and
web server do. The embedder is constructed lazily and shared across calls --
it is just an httpx client.
"""

import logging

from mcp.server.fastmcp import FastMCP

from attestation.mcp import register_all

log = logging.getLogger(__name__)

mcp = FastMCP("attestation")
register_all(mcp)


def main() -> None:
    """The `attest-mcp` console script: load `.env`, then serve over stdio.

    Runs until the client disconnects or the process is signalled -- see
    `attest reload`, the CLI command that SIGTERMs every live one of these
    so a session picks up code edits instead of holding a stale process.
    """
    from attestation.llm import load_env

    load_env()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Backward-compatible `_<tool>_impl` names.
#
# 83 test call sites reach these directly, a seam that existed because the old
# wrappers were FastMCP-decorated and so awkward to call. The domain modules no
# longer have that problem -- `register(mcp)` decorates thin wrappers and the
# implementations below them are plain functions -- but the names are kept for
# one release so the split is not entangled with rewriting every test.
#
# These are aliases, not copies: each points at the live implementation, so a
# test exercising `_list_feed_impl` exercises exactly what the tool serves.
# ---------------------------------------------------------------------------

from attestation.mcp import feed as _feed  # noqa: E402
from attestation.mcp import knowledge as _kg  # noqa: E402
from attestation.mcp import personas as _personas  # noqa: E402
from attestation.mcp import provenance as _prov  # noqa: E402
from attestation.mcp import subscriptions as _subs  # noqa: E402
from attestation.mcp import symbolic as _sym  # noqa: E402

_list_feed_impl = _feed._list_feed
_record_feedback_impl = _feed._record_feedback
_explain_item_impl = _feed._explain_item
_add_feed_impl = _subs._add_feed
_list_feeds_impl = _subs._list_feeds
_remove_feed_impl = _subs._remove_feed
_preview_feed_impl = _subs._preview_feed
_suggest_feeds_impl = _subs._suggest_feeds
_create_persona_impl = _personas._create_persona
_update_persona_impl = _personas._update_persona
_propose_interests_impl = _personas._propose_interests
_profile_status_impl = _feed._profile_status
_search_feed_impl = _feed._search_feed
_delete_persona_impl = _personas._delete_persona
_reset_feedback_impl = _personas._reset_feedback
_digest_impl = _feed._digest

_kg_neighbors_impl = _kg._neighbors
_kg_path_impl = _kg._path
_kg_central_impl = _kg._central
_kg_communities_impl = _kg._communities
_kg_concepts_impl = _kg._concepts

_runs_scan_impl = _prov._scan
_runs_list_impl = _prov._list
_runs_compare_impl = _prov._compare
_runs_detail_impl = _prov._detail
_claims_check_impl = _prov._check
_claims_coverage_impl = _prov._coverage

_sym_simplify_impl = _sym._sym_simplify
_sym_solve_impl = _sym._sym_solve
_sym_differentiate_impl = _sym._sym_differentiate
_sym_integrate_impl = _sym._sym_integrate
_sym_derivation_impl = _sym._sym_derivation
_sym_verify_impl = _sym._sym_verify
_sym_evaluate_impl = _sym._sym_evaluate
_simulate_feedback_impl = _feed._simulate_feedback
_harvest_engagement_impl = _feed._harvest_engagement

_read_item_impl = _feed._read_item

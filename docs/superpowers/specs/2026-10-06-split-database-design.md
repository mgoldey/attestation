# Split database: pin ATTEST_DB to the database that holds the data

**Date:** 2026-10-06
**Status:** approved by request ("fix the split database issue in attestation").
**Depends on:** the install `env_file` step (`_db_needs_pinning`, added 2026-09-28
for the Agent37 image) and `resolve_db_path`.

## What happened

Before v0.2.2, nothing pinned `ATTEST_DB`. `resolve_db_path` fell back to
`./hermes.db`, and Hermes starts `attest-mcp` from its own working directory,
so on an AgentMarkit research machine the agent's tools wrote
`~/hermes.db` while ingest, run from the checkout, wrote
`<checkout>/hermes.db`. On 2026-10-06 one such machine (attestation v0.2.0)
had 130 references, a persona and its vectors in `~/hermes.db`, and an
empty database in the checkout.

v0.2.2's fix pins `ATTEST_DB` to **the checkout's** database whenever the
path is unpinned. On that machine an upgrade would have switched the agent
onto the empty database and left everything it had saved behind, silently,
with every install check green.

## Decisions

1. **Pin to the database that has data.** When `ATTEST_DB` needs pinning,
   the candidates are `<checkout>/hermes.db`, `~/hermes.db` and
   `$HERMES_HOME/hermes.db` (the directories Hermes has been seen to start
   the MCP server from), deduplicated, existing files only. "Has data" is a
   read-only count of rows in `items`, `"references"` and `clicks` (a table
   that is missing counts as zero; an unreadable file counts as zero). One
   candidate with data: pin to it. None: pin to the checkout's, as before.
2. **More than one with data: refuse, in both check and fix modes.** Merging
   two databases is not something an installer should guess at. The step is
   BROKEN, names each path with its counts, and says to set `ATTEST_DB` in
   the checkout's `.env` to the one to keep (`attest backup` the other).
3. **An existing pin to an empty database while another candidate has data is
   reported, never overridden.** That is the machine already upgraded past
   v0.2.2 with its data orphaned. An explicit `ATTEST_DB` is the operator's
   choice, so the step is BROKEN with the paths and counts instead of
   rewriting it.
4. Reads are `mode=ro` URIs: `--check` must write nothing, and `get_db` would
   create tables.

# Agent-config emitters

**Date:** 2026-08-22
**Status:** implemented 2026-08-23 in `d29c604`. One addition the spec did
not anticipate: the doctor found a state the installer could not reach --
it reported four missing surfaces that nothing created -- so
`step_mcp_wiring` now registers them rather than only reporting them. A
check for an unreachable state tells every user their install is broken.
**Roadmap:** rescopes spec 5 of `2026-08-21-architecture-roadmap.md`
**Depends on:** nothing. See "What changed since the roadmap".

## What changed since the roadmap

The roadmap made spec 5 depend on spec 4 and generate from the `swarm.toml`
that spec 4 would have produced. Spec 4 is closed
(`2026-08-22-swarm-refutation.md`): the swarm was measured at 7.3/15 against
8/15 for doing nothing, and there will be no `swarm.toml`.

The need it named did not go away — it moved. Agent definitions still exist,
they are just produced by `AGENT_SURFACES` instead of a swarm config, and they
are currently written **by hand**. So the emitter generates from
`AGENT_SURFACES`, which exists and is already the single source of truth for
what a surface contains.

## Problem, demonstrated live

`~/.hermes/config.yaml` on this machine contains five attestation entries: the
full `attestation` server, plus `attestation-feed`, `attestation-provenance`,
`attestation-knowledge`, `attestation-symbolic`, each setting `ATTEST_TOOLS`
and each `enabled: false`.

`install.py`'s `step_mcp_wiring` writes exactly one of those — the full
`attestation` entry — and returns `ok` the moment it sees the string
`attestation` in `hermes mcp list`. **The four surface entries were typed by
hand during this session and nothing in the repo knows they exist.**

The consequences are already latent:

- **Nothing detects drift.** Add a fifth surface to `AGENT_SURFACES` and the
  config keeps offering four. Rename one and the config points at a name that
  now raises — `register_all` raises on an unknown `ATTEST_TOOLS` rather than
  silently serving everything, so the failure is loud, but it happens at the
  user's next tool call rather than at install time.
- **The doctor cannot see it.** `mcp_wiring` matches on a substring. A config
  with the full server and zero surfaces reports `ok`, which is the same
  false-clean the scheduler check had before it learned to look for a second
  entry.
- **A fresh install gets none of them.** Anyone who runs `attest install`
  today gets the flat surface, which measured 8/15 — not the four
  surfaces, which are the reason the surfaces exist.

This is a source-of-truth problem with two copies and no check between them.

## Scope

One generator, two consumers, one check.

### The generator

`AGENT_SURFACES` gains, alongside each surface's tool prefixes, the prose a
config needs: a one-line description and the model hint. Today that prose
exists only in a comment above the table and in the hand-typed YAML.

```python
@dataclass(frozen=True)
class Surface:
    prefixes: frozenset[str]
    summary: str          # one line, shown in the agent picker
    rationale: str        # why this is its own session — for the emitted comment
```

`AGENT_SURFACES` becomes `dict[str, Surface]`. `_allowed` reads
`.prefixes`; the change is mechanical and the existing architecture tests that
enforce namespacing keep passing unchanged.

The rationale strings already exist as the comment block above the table and as
the "why it is its own agent" column in `2026-08-22-agent-surfaces-design.md`.
Moving them into the data is the point: prose that explains a config should
live where the config is generated from, or it rots separately from it.

### Consumer 1 — Hermes MCP entries

`emit_hermes_config() -> dict` returns the `mcp_servers` fragment: one entry
per surface, `command: uv`, args `run --project <root> attest-mcp`, env
`ATTEST_TOOLS: <name>`, `enabled: false`.

**`enabled: false` is the correct default and stays.** A surface is chosen at
launch by a person, per the refutation's finding; five servers all enabled
would put every tool back in one session and undo the split.

### Consumer 2 — Claude Code subagents

`emit_claude_agents() -> dict[str, str]` returns `.claude/agents/<name>.md` per
surface, with frontmatter carrying `description` (the summary) and the tool
allow-list. Same schema, different runtime — which is the roadmap's actual
point: *"One schema, many consumers, so the definitions cannot drift."*

### The check

`attest install --check` gains surface verification, and this is the part that
earns the spec.

Following the scheduler fix from the same review: **the doctor must compare
against the generator, not against a substring.** For each surface in
`AGENT_SURFACES`, the config must have an entry whose `ATTEST_TOOLS` matches
and whose args point at this checkout. Report:

- **missing** — surface exists, config does not have it
- **stale** — config entry points at a different checkout path
- **orphaned** — config has an `attestation-*` entry naming a surface that no
  longer exists

Orphaned is the one that matters most and the one a substring check can never
find. It is the same shape as the duplicate crontab entry: a hand-added thing
that used to be right, that the tool that owns the domain cannot see.

## Hand-edited emitted files

The roadmap asked how to detect one. The answer is to not need to.

**Do not checksum, and do not overwrite.** A checksum turns a user's edit into
a warning they cannot act on, and silently overwriting is worse. Instead:

- Emitted files carry a header naming their generator and the command to
  regenerate.
- `--check` reports a difference between generated and on-disk as `BROKEN` with
  a diff, and never rewrites.
- Writing happens only on explicit `attest emit --write`.

A user who deliberately edits an emitted file gets a persistent, accurate
report that it differs, and the choice of what to do about it. That is the same
contract `_write_refresh_script` already uses ("missing or stale"), so it is
this repo's existing convention rather than a new one.

## Where emission lives

**A CLI subcommand, `attest emit`, and a call from `install.py`.** Not a build
step: there is no build, and adding one to keep two YAML fragments in sync is
disproportionate. Not install-only: a user who adds a surface should not have
to re-run an installer to get its config.

`step_mcp_wiring` calls the same generator so install and emit cannot diverge —
which is the whole failure being fixed, and it would be an embarrassing one to
reintroduce in the fix.

## Success criteria

- `attest emit --check` reports BROKEN on this machine's current config **only
  after** a surface is added or renamed, and `ok` for the four entries as they
  stand — verified by adding a fifth surface, observing BROKEN, and removing it.
- Adding a surface to `AGENT_SURFACES` and running `attest emit --write`
  produces a working Hermes entry with no hand editing.
- Renaming a surface produces `orphaned` for the old entry, not silence.
- A hand-edited emitted file is reported, never overwritten.
- `install.py` and `attest emit` produce byte-identical output for the same
  surfaces — asserted by a test, because two callers of one generator is
  exactly where this class of bug returns.
- Full pre-commit gate green.

## Out of scope

Emitting for runtimes nobody here uses. Two consumers is enough to prove the
schema is not shaped around one of them; a third is speculative until someone
wants it.

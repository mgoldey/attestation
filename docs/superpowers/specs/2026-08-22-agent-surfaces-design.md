# Agent surfaces — design

**Date:** 2026-08-22
**Status:** proposed

## Problem

One agent chooses among 37 tools and gets it wrong more than half the time.

Measured on `gemma4:e2b-it-q4_K_M` over 15 realistic user turns, repeated three
times with no variance: **8/15 correct**. The misses are not exotic. "what
should I read today?" picked `feed.digest`; "which arm of my sweep won?" picked
`kg.central`; "are the numbers in my draft right?" picked `sym.verify`.

Three further failures were reported from live use: the agent cannot chain
calls that have an order (`kg.concepts` before `kg.path`), it mangles rendering
(a watched Telegram session looped truncate → apologise → re-render → dump
JSON), and it loses the thread across turns.

## What was measured, and what it rejected

Three architectures, same 15 turns, same model, three runs each.

| architecture | correct | latency |
|---|---|---|
| **Routed** — 4 intent tools, deterministic dispatch | **13/15** | 1.3s |
| Flat — 37 tools (today) | 8/15 | 1.3s |
| Swarm — supervisor LLM + namespace subagent LLM | **7.3/15** | **2.8s** |

**The swarm was rejected by measurement, not by preference.** It performed
worse than doing nothing at twice the latency, and the mechanism is plain: a
second model call is a second chance to be wrong, and a namespace miss is
unrecoverable, so errors compound rather than cancel. This is the determinism
argument from `2026-08-21-architecture-roadmap.md` spec 4, now with numbers
instead of an opinion. That spec's swarm should be considered refuted for
routing; it may still have a case for orchestration, which is a different job.

Two details moved routing from 9.7 to 13, and both are worth keeping:

- **No catch-all.** An early routed version had a `doctor` tool for "diagnose
  the system". It became a magnet: three of four remaining misses went to it.
  An ambiguous question must produce a question back, never a default.
- **Descriptions that name the user's words.** "which topics are most central
  or most read about" catches a turn that "what is central" does not.

## Design

### Four agents, one namespace each

Separate agents help when a PERSON chooses which to talk to. They hurt when a
MODEL chooses at runtime — that is what the swarm arm measured. So the split is
by session, selected at launch, and enforced at tool registration.

| agent | tools | why it is its own agent |
|---|---|---|
| `attestation-feed` | `feed.*` (19) | Conversational; a wrong guess costs a retry |
| `attestation-provenance` | `runs.*` (6) | Verification: a wrong answer reaches a manuscript, and the caveats are the product |
| `attestation-knowledge` | `kg.*` (5) + `feed.search` | Exploratory, read-only |
| `attestation-symbolic` | `sym.*` (7) | Sandboxed subprocess, touches no database |

Claims live with runs rather than with the knowledge graph. `runs.claims_check`
verifies numbers in Markdown **against recorded runs** — it shares a database
and a failure mode with `runs.compare`, and separating them would put a claim
checker in a session that cannot see what it checks against.

`feed.search` is duplicated into the knowledge agent because "how does X
connect to Y, and what did I read about it" is one question, and a knowledge
session with no way to reach the items is a dead end.

**Mechanism:** `ATTEST_TOOLS=feed` on the server process registers only that
namespace. Four entries in `~/.hermes/config.yaml` pointing at the same
command with different env. No new protocol and no A2A: restriction happens at
registration, so a tool outside the agent's remit is *absent* rather than
discouraged.

Unset `ATTEST_TOOLS` registers everything, so the single-agent setup keeps
working and nothing existing breaks.

### Progressive disclosure inside each agent

Each agent opens with **one** tool: its `ask` router. A 1-way choice cannot be
mis-picked, and the router's own dispatch is deterministic.

```
feed.ask(user, question)   runs.ask(question)
kg.ask(question)           sym.ask(expr, question)
```

A second tool, `<namespace>.tools()`, lists the specific tools and enables them
for the rest of that conversation. This is the escape hatch: an agent that
knows it wants `runs.compare(family="kdsweep", metric="wer")` should not have
to phrase it as a question, and a question the router mis-routes must have a
way out.

Enabling is per-session and additive — tools are never removed mid-conversation,
because a tool disappearing between turns is indistinguishable from a bug.

### Routing is deterministic

`question -> (tool, kwargs)` as a pure function: keyword and shape rules, no
model call. That is what holds latency at 1.3s and what makes the 15 cases
unit-testable without a database or a model.

When rules do not select confidently, the router **asks**:

```
{"ok": false, "answer": "Did you mean the current feed, or the whole archive?",
 "options": ["feed.list", "feed.search"]}
```

Never a default. The `doctor` result is the evidence.

### Responses are schema-defined, not prose

Every `ask` tool declares a Pydantic return type, which FastMCP emits as MCP
`outputSchema` — verified working. Today no tool declares one, so every
response is an untyped dict the model must interpret and reformat, which is
where the rendering loop lives.

```python
class Answer(BaseModel):
    ok: bool
    answer: str        # one line, rendered VERBATIM by the UI
    items: list[Ref]   # id + url only: enough to act, not enough to reformat
    caveat: str | None # ranking_quality / runs.compare caveats, unabridged
    options: list[str] # populated only when the router needs a disambiguation
```

`answer` is written by the router and relayed, not re-rendered. `items` carries
ids and urls only — deliberately too little to tempt a model into rewriting the
list, which is what truncated in Telegram.

**`caveat` is load-bearing.** `ranking_quality`'s honesty note and
`runs.compare`'s caveats pass through verbatim. A composed answer that drops
them is worse than the payload it replaced, because it reads as confident.

## Testing

- **The 15 routing cases as unit tests.** Routers are pure, so these run with
  no model and no database, and they are the regression guard for 13/15.
- **A `live_model` test** re-running the three-way comparison, so the claim
  "routing beats flat beats swarm" is checked rather than remembered.
- **Schema conformance:** every `ask` tool declares an `outputSchema`, and a
  failure response validates against the same model as a success — the
  property `_tool.py` already enforces for the flat surface.
- **Restriction:** with `ATTEST_TOOLS=feed`, `list_tools` returns only feed
  tools. A test asserts a provenance tool is genuinely absent, not merely
  undocumented.
- **Caveat preservation:** a comparison with caveats must surface them in
  `caveat`; a test drives `runs.ask` at a family with a known caveat and
  asserts the text survives.

## What this does not do

No LLM subagents. No A2A. No inter-agent messaging — the four agents never talk
to each other, because nothing in the measurement suggested they should, and a
message bus between them would reintroduce exactly the compounding-error
mechanism the swarm arm demonstrated.

The 33 existing tools keep working unchanged for anything that calls them
directly, including the CLI and the web UI.

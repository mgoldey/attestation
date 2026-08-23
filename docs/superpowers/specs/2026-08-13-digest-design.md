# Digest — design

**Date:** 2026-08-13
**Status:** implemented as the `feed.digest` tool; moved into `mcp/feed.py` by the domain split in `0db570c`.

## Problem

The question a scientist actually asks is *"what's worth reading this week, and
why?"* Answering it today takes three tool calls and manual assembly:
`list_feed` for the ranking, `kg_communities` for the topic structure, and
`explain_item` per item for the rationale. An agent on a weekly schedule has to
orchestrate all three and invent the grouping itself.

Reading is where science users spend their time, and it is the part of this
toolkit that improves with use — every `record_feedback` call trains the
ranker. A digest is the natural unit of that loop.

## Decisions

### Compose, don't compute

Everything needed exists: `rank_items` orders the feed, `kg.communities`
already partitions concepts into 13 coherent topics, and items carry the tags
that link them. Measured on the live database, **12 of 12 ranked items assign
to a community** by tag overlap, so the join is not hypothetical.

The digest adds no new scoring, no new storage, and no new model call.

### No LLM inside the tool

The caller *is* a model. Putting a local LLM inside the digest to write prose
would add a 5–90s serial path, GPU contention, and nondeterminism, to produce
something the calling agent does better with structured input in hand. The tool
returns structure; the agent writes the summary.

This is the same reasoning that keeps `runs_compare` from narrating its own
results, and it is why `explanation` is included per item as *cached text* —
`explain_item` already stores explanations, so surfacing them costs nothing,
but the digest never generates one.

### Assignment: strongest tag overlap, deterministic

An item joins the community sharing the most of its tags. Ties break on the
community label, so repeated calls agree — the same determinism guarantee the
rest of the graph layer makes.

An item whose tags match no community lands in `unclustered` and is reported
there rather than hidden. 86% of the tag vocabulary is singletons that never
become graph nodes, so this bucket is expected to be non-empty and its size is
a real signal about the week's reading.

### Honest about a cold profile

`profile_status` exists to report how well-trained a persona is, and nothing
surfaces it proactively. A digest built from an untrained ranker looks
identical to one built from a good one.

The digest therefore carries `ranking_quality`: click count, whether the
classifier is actually active, and a plain-language caveat when it is not. On
the author's own profile — 8 clicks, all positive — the single-class guard
means **the classifier has never fired**, so the ranking is pure embedding
similarity. A reader deserves to know that before trusting the order.

## Tool

| Tool | Behavior |
|---|---|
| `digest(user, days=7, per_topic=3, limit=30)` | Ranked unread items grouped by topic, with cached explanations and a ranking-quality caveat. |

Returns:

```
{"ok": true, "message": "...",
 "topics": [{"label": "machine-learning", "items": [...], "n_total": 9}],
 "unclustered": [...],
 "ranking_quality": {"clicks": 8, "classifier_active": false, "caveat": "..."},
 "window_days": 7}
```

`per_topic` caps items shown per group while `n_total` reports how many that
group actually had — truncation must be visible, not silent.

## Testing

- Items group under the community their tags overlap most.
- An item matching no community lands in `unclustered`, not dropped.
- Assignment is deterministic across repeated calls.
- `per_topic` truncates but `n_total` still reports the true count.
- `ranking_quality.classifier_active` is false for a single-class click history
  and true once both classes exist — the case that silently degrades ranking.
- An empty feed returns `ok: false` with the success-path keys present.
- No LLM call is made (a chat_fn stub must go untouched).

## Out of scope

Generating prose summaries; scheduling; email/notification delivery; a new
ranking signal; cross-user digests.

---
name: attestation-feed
description: "Rank, read and rate the reader's personal science feed -- today's recommended papers from their own subscribed sources, a search of everything already ingested, a weekly digest, and the feedback that trains the ranking. One tool, feed.ask, answers all of it. Local only; nothing is fetched from the web unless the reader asks you to go and look."
version: 3.0.0
author: attestation project
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [feed, recommendations, ranking, papers, local-api]
    related_skills: [attestation-setup, attestation-knowledge]
---

# attestation: the feed

Use this when the reader asks about their feed, wants papers to read, wants to
find something already ingested, reacts to an item ("not my area", "good
find"), or asks what you can do.

**You have one tool for all of it: `feed.ask`.** (`feed.tools` explains why the
others are hidden.) It routes the reader's words by rule -- no model call, so a
wrong route is a bug someone can fix -- and measured on gemma4:e2b the router
was chosen 26 times in 26 when it stood alone, against 1 in 26 when specific
tools were listed beside it. Listing, searching, the digest and rating are all
`feed.ask` -- there are no separate tools for them in your list. **Never call a
tool that is not in your tool list**, whatever an answer names: a 2026-09-25
session called listing and digest tools it had read about but could not see,
the client rewrote each to `feed.ask` with the wrong arguments, every call
failed, and the reader was told the feed was "unreachable".

## When NOT to use this

- The open web or news outside the subscribed feeds.
- "How do these two topics connect?", "what have I been reading about?" -- the
  knowledge agent (`attestation-knowledge`). Say so if your session lacks it.
- Experiment arms, sweeps, the numbers in a draft -- the provenance agent
  (`attestation-provenance`). Do not search the feed for a sweep name.

## How to call it

```
feed.ask(user="<name>", question="<the reader's words>")
feed.ask(user="<name>", question="useful", item_id=<id>)      # act on one item
feed.ask(user="<name>", question="subscribe", url="<feed url>")  # act on one feed
```

These dotted names are for you to read, not a literal call string. Invoke your
real tool-calling mechanism; never print the call as your reply. A 2026-09-03 session reasoned to
the right call and then printed it as its answer, which the reader saw as
"nothing happened". Some clients rewrite the dotted name to something like
`mcp__attestation_feed__feed_ask`: call the name your own tool list shows,
never a plausible variant of it.

- **`user`** is the name you already have for this reader. Never ask for a
  persona name, never invent one, never pass a placeholder like "user" -- an
  unknown name is created on sight, and a model that passed `user="user"` 9
  times in 9 made an empty persona that ranks badly forever.
- **`question`** is the reader's own words. Pass them as they said them; do not
  shorten them. A 2026-09-25 session cut "research sparse mixture of experts on
  arxiv" to "sparse mixture of experts" and lost the part that said *where to
  look*.
- **`item_id`** comes from `refs` in an earlier answer. **`url`** is a feed
  address the reader gave you. Neither is ever required to ask a question.

`user` and `question` are always enough to start. Do not invent a precondition
first ("I need their subscriptions") -- a real session did exactly that and
answered nothing.

## What comes back

`answer` -- relay it verbatim. `refs` -- the ids and urls of what it names.
`caveat` -- read it and say what it says. `tool_used` -- what answered.
`options` -- set only when the router declined (below).

**When `ok` is false and `options` is set, the router asked you a question.**
The options are routes, NOT tools you can call. Ask the reader the question in
`answer`, or -- when one plainly fits -- call `feed.ask` again with the question
reworded to say it ("what should I read?" for the list, "find papers on X" for a
search, "digest of the week" for the digest). Never tell the reader you have no
tool for their feed: a 2026-09-04 session did, with `feed.ask` in its own tool
list.

## "What should I read?" / "find me papers on X"

**Present each item as one line: a linked title, then source and topics.**

```
1. [LogicIF: Towards Complex Logic Instruction Following](https://arxiv.org/abs/2508.09125)
   arXiv cs.LG · language-models, reasoning
```

Markdown links
are correct on every surface this agent ships to (measured on Telegram, five of
five); do not hand-write another surface's syntax. **List every item it
returned** -- the ORDER answers "what first?"; a 2B model told only how to
present rendered one item of five, and five of five once told to list them all.
Never restate ids or reproduce JSON: a reader once got `ID: 2385` instead of a
link and replied "you didn't give links".

**Say what `caveat` says.** It is the difference between "what the system
learned about you" and similarity alone -- if the ranking is not yet trained on
this reader's judgements, the caveat says so, and so should you.

A short search result is usually the relevance floor doing its job, not a
failure. If a response is too long to render, say so in one sentence and ask
for fewer -- do not re-render the same payload in another format.

## Acting on one item -- the part that gets skipped

Pass `item_id` from `refs` with the reader's verdict:

```
feed.ask(user="<name>", question="not useful", item_id=15032)
feed.ask(user="<name>", question="useful", item_id=15032)
```

**Record the opinions the reader never labels as feedback.** The ranker needs
BOTH kinds to learn at all, and on this installation nothing was recorded for a
month because nothing asked. Treat these as `"not useful"`: "not my area",
"already read that", "old news", "too applied", "too theoretical", asking for
something *instead of* what you showed. Treat these as `"useful"`: "that looks
interesting", "send me that one", "good find", a follow-up about the item's
*content*. Find the id by title or position ("the second one"); when a remark
is ambiguous, ask once.

With an `item_id`, "why is this here?" explains its rank (a local model call,
cached afterwards -- seconds are normal), and anything else opens it and
returns its text.

## Going and looking, and following a topic

When the reader names a place -- arXiv, PubMed, CrossRef, a journal, "the
literature", "what's been published" -- they want you to search the network:
pass their words as they said them ("search arxiv for equivariant force
fields"). Hits go to the reference library, not the feed; show every url. **If
`caveat` says a source failed, say which one** -- zero results from a source
that refused the request is not "nothing has been published".

"Track / follow / watch X" registers a standing topic that every hourly ingest
searches. Say "papers appear after the next ingest", which is what it answers.

"Subscribe me to <feed>" with its `url` adds an RSS/Atom feed; items appear
after the next ingest, not immediately.

## Things this session cannot do -- say so, don't improvise

Changing the reader's interests text, deleting or resetting a persona, and
removing a feed are not reachable through `feed.ask`. Tell the reader plainly
and point to `attestation-setup`; do not look for a tool to do it. "How well is
this trained?", "what are my interests?" and "when did the feed last update?"
ARE answerable -- just ask.

## Mistakes that look reasonable

| Instead of | Do |
|---|---|
| Calling a tool the page or an answer names | Call only `feed.ask`; reword the question |
| Shortening the reader's question | Pass their words as they said them |
| Asking the reader for a persona name | Use the name you have |
| Treating `options` as tools to call | Ask the reader, or reword and call `feed.ask` |
| Only recording what the reader liked | Record the negatives; the ranker needs both |
| Reporting zero research results as "nothing published" | Read `caveat`; a source may have failed |
| Retrying a failed call with the same arguments | Read the message -- it names the fix |

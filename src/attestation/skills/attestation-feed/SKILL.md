---
name: attestation-feed
description: "Rank, read and rate the reader's personal science feed -- today's recommended papers from their own subscribed sources, a search of everything already ingested, a weekly digest, and the feedback that trains the ranking. Local only; nothing is fetched from the web."
version: 2.0.0
author: attestation project
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [feed, recommendations, ranking, papers, local-api]
    related_skills: [attestation-setup, attestation-knowledge]
---

# attestation: the feed

Use this when the reader asks about their feed, wants today's recommended
papers, wants to find something already ingested, or gives an opinion on an
item ("mark that useful", "not my area", "why did this rank first?").

**Also use it when they ask what you can do.** Answer by doing: call
`feed.ask(user, question="what should I read?")`, name two papers, and offer
the obvious next step. A list of capabilities is a menu, and nobody asked for
a menu. If no persona exists yet, one is created on that first call, so there
is nothing to set up and nothing to ask.

The ranking is deterministic -- a profile embedding plus a classifier
trained on the reader's own clicks -- with an explanation layer that fills in
lazily and never blocks the feed. Corpus: the reader's subscribed feeds,
mostly arXiv, plus whatever they have added. Setup, reload and the HTTP
fallback are in `attestation-setup`.

## When NOT to use this

- General web search or news outside the subscribed feeds: this skill knows
  only what is already in the database, and adds nothing from the web.
- "How do these two topics connect" or "what have I been reading about" --
  that is the knowledge agent (`attestation-knowledge`), which sees the
  concept graph. If your session also has it, use it; if not, say so.

## Ask the router first

`feed.ask(user, question)` takes the question in the reader's own words and
routes it by rule -- a table of phrases, no model call, so a wrong route is a
bug someone can fix. Measured on gemma4:e2b over 15 turns, three runs:
routing 13/15 against 8/15 for picking from the flat surface.

```
feed.ask(user="<name>", question="what should I read today?")        # -> the ranked list
feed.ask(user="<name>", question="anything new on protein folding?")  # -> a search
feed.ask(user="<name>", question="what's worth reading this week?")   # -> the digest
```

These dotted names are for you to read, not a literal call string: some MCP
clients rewrite `feed.ask` to something like `mcp__attestation__feed_ask`
before it ever reaches you. Call the tool by the exact name your own tool
list shows for the same arguments, and never retry a plausible-looking
variant of the dotted name.

It returns `answer` (relay it VERBATIM), `refs` (item ids for your next
call), `caveat` (ranking honesty, unabridged), `options` and `tool_used`.
**A router never guesses.** When it does not claim a question confidently it
returns `ok=false` with a question in `answer` and the alternatives in
`options` -- ask the reader that question; do not pick from `options`
yourself. Fall through to a specific tool only when the router declined and
the reader has now said which, or when the call needs an argument the
question does not carry (an `item_id` to rate, a URL to add) -- the router
says so and names the tool. Specific tools may be hidden from your session;
`feed.tools` explains why and how to reveal them.

## The persona -- never make the reader do bookkeeping

`feed.list`, `feed.search`, `feed.digest` and `feed.read` create a persona on
first sight and answer in the same call. So:

- **Never ask "which persona?" or "what should I call your profile?"** The
  name is whatever you already have -- the chat handle, the username. A
  reader asked to invent a profile name has been handed admin work.
- **Do not call `feed.persona_create` to get started.** It builds a *second*
  reader (a colleague's profile, a demo persona). Reaching for it on an
  unknown name is what put a duplicate persona in this database days after
  that reader had been merged away.
- **Never pass a placeholder.** With no name in the prompt, a model passed
  `user="user"` on 9 of 9 calls and autocreate turned each into an empty
  persona that ranks badly forever.
- **Ask what they read about, once, after answering.** A new persona starts
  from the corpus's own common topics and says so. That is the moment to ask
  the only question worth asking, because the interests text IS the profile
  embedding: "what do you actually work on?" Then
  `feed.persona_update(name, interests)`. Ranking re-steers immediately.

`feed.persona_status(user)` answers "how well is this trained?" with click
counts and how much of the order is behaviour-driven versus text-driven --
an answer rather than a claim. Omit `user` and it lists every persona.

## "What should I read?"

**Present each item as one line: a linked title, then source and topic.**

```
1. [LogicIF: Towards Complex Logic Instruction Following](https://arxiv.org/abs/2508.09125)
   arXiv cs.LG · language-models, reasoning
```

**Markdown link syntax is correct on every surface this agent ships to.**
Telegram is the measured one: the gateway converts Markdown to MarkdownV2
and `[title](url)` arrives as a real link, five of five. Do NOT hand-write
another surface's syntax to "help": Slack's `<url|title>` contains angle
brackets, which trips the same sender's HTML auto-detect, and the whole
message goes out as HTML where `<url|title>` is not a tag.

**List every item the tool returned.** "What should I read first?" is
answered by the ORDER, not by truncating to one. Measured: with the
presentation rule alone a 2B model rendered one item of five; told to list
them all, five of five.

**Nothing else.** Do not restate `item_id`, `content_type` or `n_tags` in
prose -- they are for your next call. Do not reproduce the JSON. A reader
once got `ID: 2385` instead of a link and replied "you didn't give links";
the urls were in the payload the whole time.

**If a response is too long to render, say so in one sentence and show
fewer.** Do not apologise, re-render the same payload in another format, or
dump raw JSON: a watched failure ran exactly that loop and the reader saw
half of one item. `feed.list` defaults to a few items for this reason; ask
for more only when the reader does.

**Read `ranking_quality` before you present the order.** It is on every
ranked response and it is the difference between "what the system learned
about you" and "cosine similarity in a trenchcoat". `classifier_active:
false` means the click classifier has never fired -- feedback of one kind
only -- and `caveat` says which terms are actually contributing, and whether
the clicks were a person's at all. Say so rather than implying the ranking
is personalised. A caveat is absent only when it has been earned.

`feed.explain(user, item_id)` answers "why is this here?" in one sentence.
It is a local model call, cached afterwards; seconds are normal. Never let
it block the list.

## "Find me papers about X"

```
feed.ask(user="<name>", question="anything on protein folding?")
```

The router strips the asking from the question and searches the topic. The
search is semantic: "LLM" finds papers titled "Large Language Models" with
the acronym nowhere in the text, so do not fall back to keyword guessing
when a query returns little. Every result carries `match` (`semantic`,
`literal`, `both`) and a `relevance` score; `both` is the strongest signal.
**A short result list is usually the relevance floor doing its job, not a
failure.**

`feed.search(user, query, tag, content_type, limit)` searches the *whole*
archive, rated items included, optionally filtered by tag or content type.
An empty `query` with a `tag` is a filter, which is right when the reader
named a topic rather than described one. Do not guess a tag: an unknown one
is refused with the nearest real names, so pass what the reader said and
read the refusal.

## The digest

`feed.digest(user, days, per_topic, limit)` is the weekly review: the ranked
unread feed already grouped by topic. Each item joins the cluster its tags
overlap most; items matching no cluster come back in `unclustered` rather
than being dropped, and a large `unclustered` is a signal, not a bug.
`per_topic` caps what is shown per group while `n_total` reports the true
size, so truncation is visible. It returns structure and never prose -- no
model runs inside it, and a per-item `explanation` appears only when
`feed.explain` already cached one. `days` bounds the window (default 7,
echoed as `window_days`); widen it when a quiet week returns little.

## Feedback: the part that gets skipped

`feed.rate(user, item_id, useful)` after the reader expresses an opinion --
**including the opinions they never label as feedback**. The classifier
needs BOTH classes to fire at all: a reader whose history is all positive is
ranked by embedding similarity alone, forever, no matter how many items they
approve. In this project's own database, real readers had recorded 70 clicks
across 5,167 items and every one was positive.

Ordinary conversation is full of verdicts. Treat these as `useful=false`:

- "not really what I'm after" / "that's not my area"
- "I've already read that one" / "old news"
- "too applied" / "too theoretical" / "wrong subfield"
- asking for something *instead of* an item -- "anything on X rather than
  these?" rejects what was shown
- skipping past items to ask about one further down

And these as `useful=true`:

- "that looks interesting", "send me that one", "good find"
- a follow-up question about the item's *content* (not about why it ranked)

Find the `item_id` from the list you just presented -- by title, or by
position ("the second one"). When unsure whether a remark is a verdict, ask;
one short question is cheaper than a wrong label. But do not wait to be
told: a reader who never presses a button still has opinions, and an
unrecorded opinion trains nothing.

Two tools exist because real feedback is scarce:

- `feed.harvest_engagement(user)` -- free, instant, no model. Turns past
  "why is this here?" questions and reads into weak positives. Idempotent.
- `feed.simulate_ratings(user, confirm=true)` -- slow, one model call per
  item, generates BOTH classes so the classifier can fire. Read the
  `caveat` it returns: if most of a persona's positives come from one feed,
  the classifier separates by publication rather than topic and any
  evaluation score is meaningless. Its rows are marked `simulated` forever
  and `ranking_quality` says when the order was never judged by a person.

## Growing the feed is your job, not theirs

A reader who says "I want more on X" is asking you to widen their sources,
not to hand them a URL. The path is `feed.source_suggest(user)` ->
`feed.source_preview(url)` -> confirm -> `feed.source_add(url, title)`.
Suggestions come from a curated list scored against tags this reader liked
-- never web-searched, never invented. Preview before subscribing: items are
permanent once ingested. `feed.source_add` is **register-only**: new items
appear after the next ingest (hourly cron, or `attest ingest`), so do not
expect the list to show them moments later. `feed.sources()` shows what is
subscribed, with item counts and last-fetch times.

**Destructive tools require `confirm=true`**: `feed.persona_delete`,
`feed.persona_reset` (clears clicks, keeps the persona) and
`feed.source_remove` (keeps existing items and feedback). Without it they
return a refusal instead of mutating -- use that as a dry run.

## Mistakes that look reasonable

| Instead of | Do |
|---|---|
| Picking a specific tool from the whole surface | Ask the router: 13/15 against 8/15 |
| Picking one of the `options` a router returned | Ask the reader which; the router declined on purpose |
| Asking the reader for a persona name | Use the name you have; ask what they work on instead |
| Guessing a tag name | Pass what the reader said; the refusal lists the real ones |
| Presenting a ranking alone | Read `ranking_quality.caveat` and say what it says |
| Only recording what the reader liked | Record the negatives; the classifier needs both |
| Treating an empty search as broken | The relevance floor cuts weak matches on purpose |
| Retrying a failed call with different arguments | Read the message -- it names the fix |
| Concluding a missing tool is broken | It is hidden (`feed.tools`) or the server is stale (`attestation-setup`) |

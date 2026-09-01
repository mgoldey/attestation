---
name: attestation-knowledge
description: "Connect the concepts in the reader's own reading: how two topics link, which concepts are central or bridging, what the main clusters are, which items sit behind a concept, and which references those items cite. A concept graph derived from the reader's items, plus a local citation resolver -- exploratory and read-only."
version: 2.0.0
author: attestation project
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [knowledge-graph, concepts, citations, reading, local-api]
    related_skills: [attestation-setup, attestation-feed, attestation-provenance]
---

# attestation: the reading graph

Use this when the reader asks "what have I been reading about", "how do
these two topics connect", "what are my main research areas", or wants a
reference looked up by key, DOI or arXiv id. The graph is built from the
reader's own items, not from the web, and everything here is read-only.

## When NOT to use this

- Today's recommendations, rating an item, or widening the feed: the feed
  agent (`attestation-feed`). This surface has `feed.search` for finding
  the items behind a concept, and nothing else from the feed.
- Checking a draft's numbers or citation keys against runs: the
  provenance agent (`attestation-provenance`).
- Anything needing the web: the graph and the citation readers are local.

## Ask the router first

```
kg.ask(question="how do protein folding and diffusion models connect?",
       source="protein-folding", target="diffusion-models")
kg.ask(question="what are my main areas?")
```

Returns `answer` (relay VERBATIM), `refs`, `caveat`, `options` and
`tool_used`; `ok=false` with `options` means ask the reader, never pick for
them. Specific tools may be hidden from your session; `kg.tools` explains
why and how to reveal them.

## Concept names are the vocabulary, not your phrasing

Concepts are tags used at least twice, lowercase and hyphenated, linked when
they co-occur on at least two items. Spelling variants are merged, so
`machine-learning` and `machinelearning` are one hub. **Do not guess a
name.** `kg.concepts(prefix, limit)` lists the valid ones (`prefix` is a
case-insensitive substring; `n_concepts` reports how many matched, so a
capped list is never mistaken for the whole vocabulary). Call it whenever
you are not certain a name exists, and before a tag filter:

```
kg.concepts(prefix="protein")   -> ["protein", "protein-engineering", "protein-folding", ...]
feed.search(user="<name>", query="", tag="protein-folding")
```

A name that is not a concept is refused separately and says so, so a typo
can never come back as "these topics never co-occur".

## The tools

- `kg.neighbors(node, limit)` -- concepts directly adjacent to one,
  strongest co-occurrence first: "what else should I read about this".
  Direct neighbours only.
- `kg.path(source, target)` -- the shortest chain of concepts linking two
  topics. `ok=false` with `path=null` means they never co-occur, which is a
  real answer, not an error.
- `kg.central(metric, limit)` -- the most-connected (`degree`) or
  most-bridging (`betweenness`) concepts.
- `kg.communities(min_size)` -- topic clusters by modularity, each labelled
  by its hub. A dense hub cannot swallow the graph: concepts join a group
  only when their links there beat chance, and each belongs to exactly one.
- `feed.search(user, query, tag, content_type, limit)` -- the items behind
  a concept: an empty `query` with a `tag` is a filter. Results carry
  `match` and `relevance`; a short list is the relevance floor working.

The graph is derived fresh from the tags on every call; there is nothing to
rebuild and no staleness to report. On an untagged database every tool
returns an empty graph -- a setup gap (`attest tag`, see
`attestation-setup`), not a finding about the reading.

## References

Read from disk: a Zotero library at `~/Zotero/zotero.sqlite` and any `.bib`
files in the working directory. `cite.lookup(key)` returns one record by
citation key, DOI or arXiv id; every record carries `source` (which reader
answered) and `fetched_at` (`null` means from disk). `cite.search(query,
limit)` finds references by title words or an author's name, and **never
touches the network**, even when a web reader is configured.

`cite.check(path)` lints a Markdown draft's `cite=<key>` annotations for
keys no configured source resolves. It is a lint -- the key is unknown
here -- never "the cited work does not support this".

**Everything above is local.** The one exception is `ATTEST_CITATION_WEB`,
off by default, which adds a CrossRef reader when set; it is read when the
resolver is *built*, so a disabled reader cannot be coaxed into a request.
`cite.sources()` answers "did anything leave my machine" from the surface
that would have done the leaving: each configured reader with a `network`
flag, and `offline: true` means every one is on disk. Call it before telling
a reader their bibliography stayed local, rather than asserting it.

## Hand-offs

The graph is a good source for a reading wiki or vault: `kg.communities`
gives the top-level structure, `kg.neighbors` the cross-references, and
`feed.search` the items to summarise. Note-taking itself is another skill's
job (an Obsidian or wiki skill, if installed); this one supplies the map.

## Mistakes that look reasonable

| Instead of | Do |
|---|---|
| Guessing a concept name | `kg.concepts(prefix=...)` first |
| Reading `path=null` as an error | It means the topics never co-occur |
| Reading `cite.check` as "unsupported claim" | It says the key does not resolve, nothing more |
| Asserting the bibliography stayed local | `cite.sources()` and read `offline` |
| Picking one of a router's `options` | Ask the reader which |
| Concluding a missing tool is broken | It is hidden (`kg.tools`) or the server is stale (`attestation-setup`) |

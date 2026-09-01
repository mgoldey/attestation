# Recommendation approaches in the literature, reviewed against attestation

**Date:** 2026-08-30
**Status:** review. Supersedes the regime description in
`recommendation-refinements.md` (2026-08-21), which was written against
hermes3:8b, ~1,500 items and a leave-last-5-out eval, none of which is true
now. That survey's five "adopt now" items are re-verdicted in §15.
**Method:** local only. No web lookup; every citation is from memory and
given as author-year so it can be checked before it is quoted anywhere. Where
this file says "measured", the number comes from a docstring or doc in this
repo, and the source is named.

## The regime this is reviewed against

Everything below is judged against what `rank.py` and its neighbours do
today, not against recommender practice in general:

- **Users:** one real reader plus a handful of demo personas. There is no
  user-user signal and never will be.
- **Feedback:** 11 human clicks (8 `ui` + 3 `agent`) on 3 days across 19,
  against ~5,265 items (`implicit.py`, `CLAUDE.md`); all positive. The
  click classifier has never fired for a real account. `implicit.py`
  harvests explanation requests and reads as weak positives; `simulate.py`
  has a chat model react as the persona to supply negatives. Provenance is
  recorded on every row and `evaluate_user` excludes `bootstrap`.
- **Items:** ~5,200-5,500 in the archive, a 14-day candidate window,
  `embeddinggemma` 256-dim unit vectors in sqlite-vec, an LLM tag per item
  (`item_tags`), and `content_type` in `item_features`.
- **Ranker:** `rank_rows` — rank of profile cosine, rank of a per-user
  `LogisticRegression(C=0.1, balanced)` over click embeddings (silent on a
  single-class history), and tie-averaged rank of a per-tag Laplace
  preference term (`features.pref_scores_for_items`, gated by
  `_preference_ready`), mixed by `blend_weight(n) = n/(n+5)`.
- **Graph:** `kg.py` is a tag co-occurrence graph — concepts are tags with
  ≥2 uses, edges are co-occurrence ≥2, with alias folding, BFS paths,
  degree/betweenness centrality and Louvain-phase-1 communities. It is
  derived from `item_tags`, never stored.
- **Surfaces:** `feed.list` (top-4, cap 13, ~7,000-char ceiling),
  `feed.search` (sqlite-vec top-300 + literal substring, blended
  `QUERY_WEIGHT=0.75` with absolute profile cosine, relative
  `RELEVANCE_FLOOR`), `feed.digest` (ranked items clustered by KG community,
  `per_topic` cap, budgeted), `feed.explain` (one LLM call, post hoc),
  `kg.*`, `cite.*`, `runs.*`.
- **Compute and honesty constraints:** CPU numpy/sklearn, a 2B chat model
  that cannot render more than ~13 rows, an offline guarantee, and a house
  rule (`measurement-lessons.md`) that a number is about the artifact it
  was taken from. Two measured failures shape everything here: an AUC of
  0.964 that turned out to classify *provenance* (harvested positives vs
  generated negatives are drawn from different regions of the archive), and
  synthetic labels deciding 91% of the ordering for four of five personas.

The two facts that matter most: **signal is scarce and biased by exposure**,
and **the only per-item signals independent of the ranking embedding are the
tags and the graph built from them.** Most of the verdicts below follow from
those two.

## Summary table

| # | Approach | Feed reranking | Knowledge graph | Search / digest / other | Verdict |
|---|---|---|---|---|---|
| 1 | Content-based profile (Rocchio) | is the baseline | tags are the content features | search profile term | keep; add click drift |
| 2 | Collaborative filtering / MF | no users to collaborate | co-tag edges ARE item-item CF | — | not applicable as user CF |
| 3 | Implicit feedback, PU learning | positives-only is a PU problem | — | impression logging | log exposure; sample negatives from shown |
| 4 | Rank fusion (Borda, RRF) | blend is Borda; RRF weights the top | fuse centrality metrics | search literal+semantic fusion | adopt RRF, measure |
| 5 | Learning to rank | pairwise from clicks at n=11 is noise | — | search weights are hand LTR | skip until n≫100 |
| 6 | Exploration / bandits | 256-dim LinTS never sees reward | **communities are ~20 arms** | digest topic allocation | community-level bandit, not contextual |
| 7 | Diversity: MMR, calibration, DPP | MMR on top-13 | communities are the calibration partition | digest already covers topics | MMR + calibration check |
| 8 | Near-duplicate detection | same story on two feeds | merges nodes | search result dedup | cosine threshold, cheap |
| 9 | Temporal: decay, bursts | decay clicks; burst boost | weekly co-occurrence = trending concept | digest "new this week" | burst detection on tags |
| 10 | KG-based recommendation | personalized PageRank as a 3rd, non-embedding term | the point of the graph | cold start, query expansion | **adopt** — breaks the tautology |
| 11 | Explanations | path/tag explanations are faithful | paths are explanations | explain.py grounding | ground the LLM on a path |
| 12 | Cold start / elicitation | rate-6-items beats describe-interests | one item per community | onboarding | adopt in `persona_suggest_interests` |
| 13 | LLM-based reranking / labelling | listwise on 2B fails; pairwise on top-5 | LLM tags already feed the graph | simulated labels | pairwise only, measured |
| 14 | Evaluation: interleaving, IPS | **interleaving lets one reader compare two rankers** | — | — | **adopt** — the eval this regime lacks |
| 15 | Popularity / exposure bias | pref term measured coverage, not preference | hub concepts dominate | — | already partially handled |

## 1. Content-based filtering and profile vectors

**Literature.** Rocchio (1971) relevance feedback; Pazzani & Billsus (2007)
and Lops, de Gemmis & Semeraro (2011) on content-based recommenders; and
more recently natural-language user profiles — Balog, Radlinski & Arakelyan
(2019) and Radlinski et al. (2022) on *scrutable* set-based or sentence
profiles the user can read and edit.

**Where attestation stands.** `users.interests` is exactly a scrutable
natural-language profile, embedded with `QUERY_PROMPT` and compared by cosine
to items embedded with `DOC_PROMPT` (asymmetric on purpose, `embed.py`). That
is the strongest term in the blend for every real reader, because the click
terms have nothing to train on. The literature's finding that editable text
profiles are competitive with latent ones *at small n* and vastly more
transparent is the case for keeping this as the anchor rather than replacing
it with a learned vector.

**Feed reranking.** The gap is that the profile is deaf to clicks until the
classifier has both classes. The classical fix is Rocchio drift:
`v ← normalize((1-α)·v_text + α·mean(useful) − β·mean(not_useful))`. The
earlier survey proposed the positive half (EMA drift, α=0.15). Two cautions
from this repo's own record: `_profile_vector` caches by a hash of the
interests text and serves stale on embedder failure, so a drifted vector has
to be keyed on (text hash, click count) or the cache silently serves the
undrifted one; and with 11 all-positive human clicks, the drift is toward the
mean of items the ranker itself surfaced — a mild version of the bootstrap
tautology. Weight it by provenance (`ui`/`agent` full, `implicit` half,
`simulated` less) rather than pooling.

**Knowledge graph.** Tags are the content features the graph is built from.
A profile can be expressed as a *set of concept nodes* (`kg.resolve_query`
already maps a phrase to a node), which is the set-based scrutable profile
of Balog et al. and what §10 needs.

**Search.** `_score_matches` already uses the absolute profile cosine as a
tie-breaker at weight 0.25. Nothing to change.

## 2. Collaborative filtering and matrix factorisation

**Literature.** Koren, Bell & Volinsky (2009) on matrix factorisation;
Sarwar et al. (2001) item-item CF; Hu, Koren & Volinsky (2008) for implicit
data. The modern caveat: Dacrema, Cremonesi & Jannach (2019) and Rendle et
al. (2020) showed well-tuned neighbourhood and MF baselines match or beat
most published neural recommenders — the field's *strong* baselines are
simple ones.

**Where attestation stands.** User-user CF needs users; there is one. This is
not a gap to close.

**Knowledge graph.** Item-item CF's core object — "items that co-occur in
the same baskets" — is structurally what `kg.build_graph` computes over tags:
two concepts are linked when the same items carry both. So the KG *is* an
item-side collaborative signal, with tags standing in for users. That is
worth saying plainly because it means the graph can do the job CF would do
here: "readers who liked X liked Y" becomes "concepts that appear with X
appear with Y", which is §10.

**Verdict.** No user CF. Treat co-tag edges as the item-item signal.

## 3. Implicit feedback and positive-unlabeled learning

**Literature.** Hu, Koren & Volinsky (2008): treat every unobserved
(user, item) as a weak negative with confidence `c = 1 + α·r`, never as
missing. Pan et al. (2008) one-class CF with weighted or sampled negatives.
Rendle et al. (2009) BPR: learn from (observed ≻ unobserved) pairs. Elkan &
Noto (2008): positive-unlabeled learning — with positives and an unlabeled
pool, a classifier trained on P vs U recovers P vs N up to a constant, *if
positives are labelled at random*. Marlin & Zemel (2009) and Schnabel et al.
(2016): they are not — feedback is missing-not-at-random, and the fix is to
model exposure (propensity) and reweight.

**Where attestation stands.** `implicit.py` is a deliberate PU design that
infers positives only, on the argument that no behaviour reliably means "not
useful" and a noisy negative class poisons the one the ranker is starving
for. That argument is sound *for labelling*. But the measured provenance
problem (`_provenance_auc`: harvested positives sit at median archive rank
88, simulated negatives at 3,426, so the classifier learns "was this shown")
is precisely the missing-not-at-random failure the literature names, and the
literature's answer is not a better label — it is to **record exposure**.

**Feed reranking.** The `engagement` table records reads and explanation
requests. It does not record *impressions* — which items `feed.list` and
`feed.digest` returned to whom. Logging impressions is the single cheapest
change in this review and it unlocks three things:

1. Negatives sampled from *shown-and-not-engaged* items come from the same
   exposure population as the positives, so a classifier can no longer
   separate them by "was this surfaced". That is the direct fix for
   `provenance_auc ≈ 1.0`, and it is exactly Hu et al.'s weak negative with
   a confidence below a stated "not useful" — Elkan & Noto's assumption
   (positives labelled at random *within the exposed set*) is far closer to
   true than within the archive.
2. Dwell-style signals become possible: an item shown three times and never
   read is a stronger weak negative than one shown once.
3. Interleaving evaluation (§14) needs impressions to attribute a click to a
   ranker.

The house rule still holds — nothing here writes a `useful=0` row from
silence. Shown-not-engaged rows would carry their own source (`impression`)
and a confidence weight, used by training and never shown as a judgement.

**Digest.** The digest is the surface most read and least clicked; its
impressions are where the reader's real exposure lives.

**Verdict.** Log impressions now. Sample training negatives from impressions
with low weight, keep provenance. BPR itself remains unnecessary — the
system has explicit labels where it has anything — but its *negative
sampling from exposure* is the part worth taking.

## 4. Rank fusion: Borda, CombSUM, reciprocal rank fusion

**Literature.** Fox & Shaw (1994) CombSUM/CombMNZ; Borda-count rank
averaging; Cormack, Clarke & Büttcher (2009) reciprocal rank fusion,
`score(i) = Σ_r 1/(k + rank_r(i))`, k≈60, which beat CombMNZ and learned
fusion on TREC runs with no training.

**Where attestation stands.** `rank_rows` averages ranks — Borda. That was
the right first choice for the reason the earlier survey gave (cosine and
`predict_proba` are on different scales, and ranks sidestep calibration).
But Borda is linear in rank: it treats the gap between positions 1 and 2
the same as between 4,000 and 4,001. On a feed where only 4-13 of ~1,500
windowed candidates are ever shown, the ordering that matters is at the very
top, and a term that puts an item at position 3 should count for more than
one that puts it at position 800, not the same amount of "rank". RRF's
`1/(k+r)` gives the top of each list most of the weight and lets one term's
confident top pick survive another term's indifference.

**Feed reranking.** Replace `np.mean(click_ranks)` and the profile rank in
the blend with RRF contributions, keeping `blend_weight` as the per-term
weight. This is a ~10-line change in a pure function (`rank_rows`) with a
DB-free harness already in `examples/ranking/`. It is also the first thing
interleaving (§14) should be pointed at, because it is a genuine A-vs-B
question with no obvious prior.

**Search.** `_score_matches` fuses literal and semantic evidence by a
hand-weighted linear combination with a literal *boost* (0.35 title / 0.15
body) — a small learned-to-rank without the learning. RRF over (sqlite-vec
rank, literal-match rank ordered by recency) is the standard lexical-plus-
dense fusion and has no weight to tune. The measured pitfall in the code —
flooring literal hits made 711 "llm" matches tie — is one RRF avoids
naturally because rank position breaks ties.

**Knowledge graph.** `kg.central` offers degree *or* betweenness. RRF over
both (and over weighted degree) is the standard way to produce one "key
concepts" list without arguing about which metric is right.

**Verdict.** Adopt RRF in `rank_rows`; measure by interleaving; consider it
for search fusion second.

## 5. Learning to rank

**Literature.** Joachims (2002) pairwise from clicks (RankSVM); Burges et
al. (2005) RankNet; Burges (2010) LambdaMART; listwise losses (Cao et al.
2007 ListNet). Unbiased LTR: Joachims, Swaminathan & Schnabel (2017), Wang
et al. (2018) on position bias.

**Where attestation stands.** Every LTR method learns from preference pairs
or lists, and needs hundreds to thousands of them. With 11 human clicks, the
earlier survey's SKIP on BPR/hinge-rank stands, and for the same reason
applies to the whole family. Position bias correction (IPS) needs a
propensity model; with a 4-row feed the propensity is close to uniform over
what is shown and zero elsewhere, which is the impressions problem of §3,
not a weighting problem.

**Verdict.** Skip. Revisit when there are on the order of 500 labelled
pairs, which at the measured click rate is not this year unless impressions
(§3) supply them.

## 6. Exploration and contextual bandits

**Literature.** Li, Chu, Langford & Schapire (2010) LinUCB on Yahoo! news;
Chapelle & Li (2011) Thompson sampling; Agrawal & Goyal (2013) linear
Thompson sampling; ε-greedy and its offline-evaluable cousins (Li et al.
2011 replay evaluation). Radlinski, Kleinberg & Joachims (2008) ranked
bandits for diverse rankings.

**Where attestation stands.** The 2026-08-21 survey's top pick was a LinTS
head over 256-dim item embeddings. This review downgrades it, for a reason
the literature makes explicit: a bandit learns from *observed rewards*, and
a contextual bandit in d dimensions needs on the order of d observations
before its posterior is narrower than its prior. The measured reward rate is
11 clicks in 19 days. A 256-dim LinTS would sample essentially from its
prior for months, which is random exploration wearing a Bayesian costume —
and every exploratory slot it spends is a slot in a 4-row feed.

**Knowledge graph — where the bandit actually fits.** `kg.communities`
partitions the concept graph into on the order of tens of topic clusters.
That is a *K-armed* bandit with K≈20, not a 256-dimensional one: Beta-
Bernoulli Thompson sampling over "which community should the exploratory
slot draw from" converges on tens of observations, not hundreds. The reward
is any engagement (read, explain, rate) on the drawn item. This is the form
of exploration the regime can afford: one of the four `feed.list` rows, or
one topic in the digest, drawn by Thompson sampling over communities the
reader has not clicked in, with the rest exploited as now. It also answers
the calibration question in §7 with data rather than a prior.

**Digest.** `_allocate_digest_budget` spends its budget largest-cluster-
first, which is pure exploitation of corpus volume. A community-arm bandit
would allocate one slot to a topic the reader has never engaged with, with
the allocation recorded so the outcome can be scored.

**Verdict.** Not a contextual bandit. A K-armed Thompson sampler over KG
communities, feeding one exploratory slot, with impressions (§3) recording
the draw. Measure the arm's engagement rate against the exploited slots.

## 7. Diversity, calibration and coverage

**Literature.** Carbonell & Goldstein (1998) MMR; Ziegler et al. (2005)
topic diversification (users preferred slightly less accurate but more
diverse lists); Kulesza & Taskar (2012) determinantal point processes;
Steck (2018) calibrated recommendations — the *proportions* of topics in a
list should match the proportions in the user's history, measured by KL
divergence, fixed by greedy re-ranking; Radlinski et al. (2008) ranked
bandits for coverage.

**Where attestation stands.** The feed has no diversity step; five arXiv
listings of the same story on the same day can fill the four rows. The
digest already has a coverage mechanism: cluster by KG community, cap
`per_topic`, budget across clusters — that is a coverage-based (submodular-
style) diversification in all but name, and it is the right shape.

**Feed reranking.** MMR on the top ~50 by blended score, selecting the 4-13
shown, with unit vectors already in memory, is the earlier survey's cheapest
item and still is. `λ=0.7`; the measurement is a before/after count of
pairwise cosines > 0.9 among shown rows, which the ranking harness can
compute offline.

**Knowledge graph.** Communities are the natural partition for Steck's
calibration: compute the reader's click distribution over communities and
the shown list's distribution, and report the KL divergence in
`ranking_quality`. That turns a vague "is the feed too narrow" into one
number, and the digest's `per_topic` cap into a tunable with a target.

**Search.** Not wanted — a directed query should return near-duplicates if
they match; the reader asked.

**Verdict.** MMR at the feed's shown-row boundary; calibration divergence
reported over communities; DPPs are overkill at k≤13.

## 8. Near-duplicate detection

**Literature.** Broder (1997) shingling and MinHash; Charikar (2002)
SimHash; embedding-threshold dedup in news aggregation.

**Where attestation stands.** Multiple feeds cover the same paper (an arXiv
listing plus a blog post plus a cross-list). Nothing merges them. MMR (§7)
suppresses the second copy in the top rows but does not tell the reader
"this is the same thing from two sources".

**Feed / search / digest.** A cosine threshold on the stored unit vectors
(≥0.95, to be measured on the live archive) at ingest or at ranking time,
collapsing duplicates into one row with `n_sources`. Cheap, and it feeds
the KG: tags from both copies merge into one node's neighbourhood instead
of double-counting an edge.

**Verdict.** Adopt at ranking time first (no schema change), then at ingest
if the threshold holds up.

## 9. Temporal dynamics: decay, sessions, bursts

**Literature.** Koren (2009) timeSVD++ (interest drift); Hidasi et al.
(2016) GRU4Rec and Kang & McAuley (2018) SASRec for session/sequence
models — with Jannach & Ludewig (2017) showing session-kNN matches them;
Kleinberg (2002) burst detection in streams.

**Where attestation stands.** Recency is a hard 14-day window on
candidates, and clicks are unweighted by age. There are no sessions to
model — one reader, a few interactions a week — so sequential recommenders
are inapplicable regardless of their merits.

**Feed reranking.** Recency-weighting clicks (half-life ≈ the 14-day
window) is still right and still ~10 lines; it matters more once
impressions (§3) add volume.

**Knowledge graph — the interesting one.** Co-occurrence edges are computed
over the whole archive. Computed per week, the *change* in a concept's
degree or a pair's weight is a burst signal: "sparse attention went from 3
to 19 co-occurrences this week" is news the embedding cannot express, and a
"trending" boost is a ranker term independent of the profile vector. It is
also a digest section ("new this week in your graph") that costs one extra
group-by on `item_tags` joined to `items.published`.

**Verdict.** Recency-weight clicks; add weekly burst detection over tags as
a KG feature and a digest section.

## 10. Knowledge-graph-based recommendation

**Literature.** Haveliwala (2002) topic-sensitive / personalized PageRank;
spreading activation over content graphs (Crestani 1997); Wang et al.
(2018) RippleNet and Wang et al. (2019) KGAT propagating user preference
over entity graphs; Ai et al. (2018) and Xian et al. (2019) using KG paths
as explanations; Grover & Leskovec (2016) node2vec for graph embeddings.

**Where attestation stands.** The graph exists, is measured to be usable
(`kg.health`), and is used for digest clustering and the `kg.*` tools. It
does not participate in ranking at all. That is the largest untapped asset
in the system, for a specific reason: **every ranking term today is a
function of the same 256-dim embedding**, which is why bootstrap labels
were a tautology, why the pref term's "coverage" failure looked like
signal, and why `evaluate_user` cannot see most of what orders the feed.
The tags come from a different model reading the text (the tagging prompt,
`features.tag_messages`), and the graph's structure comes from co-occurrence
across items. A ranker term derived from the graph is the first term that
is *not* a linear function of the vector being ranked.

**Feed reranking — personalized PageRank.** Seed a random walk from the
reader's clicked concepts (and from the nodes `kg.resolve_query` maps
their interests text to), restart probability ≈0.15, over the weighted
co-occurrence graph. The stationary distribution scores every concept by
proximity to what the reader cares about; an item's score is the sum (or
max) over its tags. At ~700 nodes this is a few power iterations on a
sparse matrix, well under a millisecond, and it degrades gracefully: a
reader with no clicks seeds from interests text alone. Blend it as a third
rank term. Because it is independent of the embedding, `evaluate_user`
gains something to measure, and the bootstrap tautology cannot recur
through it.

**Knowledge graph — the graph gains a purpose.** Today the graph answers
questions (`kg.neighbors`, `kg.path`). As a ranker term it also becomes
*accountable*: if PPR-ranked items are not engaged with, that is evidence
about the tagging quality, which the tagging eval can then target.

**Cold start.** A new reader's interests string resolves to concept nodes;
their neighbourhood is a starting profile richer than the string alone and
does not require an embedder call. `autocreate_user` seeds from the top-6
tags of the corpus — the graph's central nodes (`kg.central`) or one node
per community are better seeds, since the top-6 by frequency are the hubs
every reader shares.

**Search — query expansion.** A query resolved to a node can be expanded by
its strongest neighbours before the literal match, giving "sparse attention"
the items tagged "long-context" too. Classic and cheap; the literal-boost
mechanism already exists to receive it.

**Explanations.** §11.

**Verdict.** Adopt: PPR from clicked concepts as a third, non-embedding
rank term, blended by RRF (§4), with a DB-free test on a synthetic graph
whose right answer is known, and a mutation that swaps the walk for a
degree ranking to confirm the test bites.

## 11. Explanations

**Literature.** Tintarev & Masthoff (2007) on the aims of explanations
(transparency, scrutability, trust, effectiveness); Zhang & Chen (2020)
survey; Herlocker et al. (2000) found feature-based ("because it is about
X") explanations most persuasive among many; KG-path explanations (Ai et
al. 2018; Xian et al. 2019 policy-guided path reasoning). The consistent
caveat: post-hoc explanations generated separately from the ranker are
rationalisations, and can be confidently wrong.

**Where attestation stands.** `explain.py` makes one LLM call given the
interests text and the item, with a refusal clause. It has no access to
*why the ranker* placed the item — and the measured failure (a termite-feed
paper explained as sharing "advanced topics like AI"; refusal recall 0.4 on
the 40-case corpus) is the literature's post-hoc rationalisation failure
exactly.

**Knowledge graph.** `kg.shortest_path(clicked_concept, item_tag)` *is* an
explanation, and a faithful one: it names the concepts that link what the
reader engaged with to what was ranked. It costs no model call. Handing
that path to the LLM as the thing to phrase ("this is about X, which
connects to Y you read about via Z") grounds the explanation in the
ranker's evidence and gives the refusal clause a concrete condition: no
path within 2 hops → refuse. That is a measurable change on the existing
explanation corpus.

**Feed.** The pref term's per-tag scores are also a feature-based
explanation ("you have marked 4 items tagged X useful") that the literature
rates highly and that the system already computes.

**Verdict.** Ground `explain.py` on a KG path or a pref-term tag when one
exists; refuse when neither does; score on `evals/explanation_cases.json`.

## 12. Cold start and preference elicitation

**Literature.** Schein et al. (2002) cold-start metrics; Rashid et al.
(2002) "Getting to know you" — asking new users to rate a few well-chosen
items (popular *and* contrastive, high entropy) outperforms asking them to
describe themselves; active learning for recommenders (Rubens et al. 2015
survey); Golbandi, Koren & Lempel (2011) decision-tree elicitation.

**Where attestation stands.** Autocreate seeds a reader from the corpus's
top tags and the SKILL.md tells the agent to ask what they read about. That
is "describe yourself", which the literature finds weaker than "react to
these". `feed.persona_suggest_interests` proposes tags.

**Feed.** An elicitation of 6 items, one per KG community, chosen as each
community's most central item, asked as "useful or not?" through the agent.
Six answers give the classifier both classes on day one — which no real
account has ever had — and the negatives are *human*, which is the class
`simulate.py` exists to fake.

**Knowledge graph.** Communities are the stratification; centrality picks
the representative. Both exist.

**Verdict.** Adopt as an onboarding path in the agent skill, backed by a
`feed.persona_suggest_items` style tool or a parameter on the existing one.

## 13. LLM-based recommendation

**Literature.** Sun et al. (2023) RankGPT listwise reranking; Hou et al.
(2024) LLMs as zero-shot rankers, with strong position bias and sensitivity
to candidate order; Geng et al. (2022) P5; LLM-as-labeler for training data,
with the known risk that model-generated labels encode the labeler's biases
rather than the user's.

**Where attestation stands.** The LLM already does three recommendation
jobs: tagging (the KG's source), simulated reactions (training negatives),
and explanation. The tagging pipeline is the most mature — corpus,
optimizer, transfer gate. The simulated-label path is where the literature's
caution landed hardest: the labels were independent of the embedding as
designed, but the *item selection* was not, and the provenance AUC found it.

**Feed reranking.** Listwise reranking is out on a 2B model that cannot
render 10 rows; pairwise comparison of the top 2-5 ("which of these two is
more useful to someone who reads about X?") is the form small models handle,
and it is offline-evaluable against the 100-case reaction corpus. Its
measured value is unknown; the earlier survey's "adopt later" stands, now
with a corpus to measure it on.

**Knowledge graph.** The LLM's tags are the graph. Improving tag quality
(the existing eval loop) improves every §10 term for free.

**Verdict.** No new LLM in the ranking path. Fix simulated-label selection
by sampling candidates from impressions (§3) rather than round-robin across
feeds, which removes the exposure confound at its source.

## 14. Evaluation: interleaving and counterfactual estimation

**Literature.** Radlinski, Kurup & Joachims (2008) team-draft interleaving;
Chapelle et al. (2012) large-scale validation showing interleaving detects
ranker differences with 10-100× fewer clicks than A/B; Li et al. (2011)
replay evaluation for bandits; Krichene & Rendle (2020) on why sampled
offline metrics mislead; Schnabel et al. (2016) IPS-corrected offline
metrics.

**Where attestation stands.** The repo's eval story is honest and
therefore thin: `evaluate_user` measures the classifier alone, "top-20
relevance against stated interests" is a hand-scored number (64/100 across
five personas, `_preference_ready` docstring), and `measurement-lessons.md`
records how often offline numbers were about the wrong artifact. There is
no way for the one real reader to compare two rankers.

**Feed reranking — the method this regime lacks.** Team-draft interleaving
takes rankers A and B, builds one list by alternately drafting each one's
next unpicked item, records which ranker each shown row came from
(impressions, §3), and credits a click to its ranker. With one reader and a
4-row feed it yields a paired comparison on every engagement, and the
literature's sensitivity gain is largest exactly where clicks are rare.
Every proposal in this file — RRF vs Borda, PPR term on vs off, MMR on vs
off, community-bandit slot vs none — becomes an interleaving experiment
rather than an argument.

**Digest.** Interleave at the topic level: alternate which ranker fills
each topic's slots.

**Knowledge graph.** No direct role, except that the PPR term (§10) is the
first candidate to test.

**Verdict.** Adopt: impressions + team-draft interleaving, surfaced through
`ranking_quality` as "ranker A vs B: n engagements, credit a/b". This is
the one change that makes the others decidable.

## 15. Popularity and exposure bias

**Literature.** Abdollahpouri, Burke & Mobasher (2017) on popularity bias
in LTR; the long-tail literature (Anderson 2006; Yin et al. 2012); exposure
bias as a special case of §3's missing-not-at-random.

**Where attestation stands.** Two instances were already found and handled
by measurement: the pref term ranked by *coverage* (how many of an item's
tags the reader had touched, which tracks feed volume) rather than
preference, and `MIN_PREF_UPVOTE_CLICKS` gates it; and hub tags dominate
the corpus's top-6, which is what autocreate seeds from. `kg.py`'s alias
folding and `MIN_TAG_USES` are a popularity floor with a stated reason.

**Feed / KG.** The remaining exposure: high-volume feeds contribute more
candidates to every window, so a rank-based blend surfaces them more. A
per-source cap (the earlier survey's "adopt later") or per-community
calibration (§7) are the two remedies; calibration is preferable because it
is measured against the reader rather than against the feed list.

**Verdict.** Handled where it was measured; calibration (§7) covers the rest.

## Re-verdicting the 2026-08-21 survey

| 2026-08-21 verdict | Now | Why |
|---|---|---|
| LinTS contextual bandit — adopt now | **downgrade** to a K-armed Thompson sampler over KG communities (§6) | 256-dim posterior never narrows on 11 clicks / 19 days; communities give ~20 arms that can learn |
| MMR — adopt now | **keep** (§7), scoped to the shown rows and measured by pairwise-cosine count | unchanged; digest already has the coverage half |
| EMA profile drift — adopt now | **keep with two conditions** (§1): provenance-weighted, and the cache key includes click state | drift toward ranker-surfaced positives is a mild tautology; `_profile_vector`'s stale-serve path would hide the drift |
| Recency-weighted training — adopt now | **keep** (§9) | still ~10 lines; matters more once impressions add rows |
| Persist LLM profile summary into ranking — adopt now | **drop** | measured since (`explain.py`, `CLAUDE.md`): synthesis returned meta-description and produced *vaguer* explanations than the interests string, +2.1s; the explain graph no longer has that node |
| BPR — skip | **skip the loss, take the negative sampling** (§3) | the exposure-based negative sampling is what fixes provenance AUC |
| IPW for position bias — skip | **skip** | still right; impressions make the propensity nearly uniform anyway |
| LLM listwise rerank — adopt later | **pairwise only, measured** (§13) | a corpus now exists to measure it on |

## Ranked changes, with how each is measured

Ordered by how much each makes the *next* decision measurable, then by
cost. Every one is local, CPU-only, and needs no new dependency.

1. **Impressions + team-draft interleaving** (§3, §14). Schema: an
   `impressions` table (user, item, surface, ranker, shown_at). Feed and
   digest write it; `ranking_quality` reports per-ranker credit. Measured
   by: the table filling, and a first A/B of Borda vs RRF.
2. **RRF in `rank_rows`** (§4). Pure-function change; the mutant is
   "revert to mean rank" and the DB-free test is a case where one term's
   top pick and another's indifference must beat two lukewarm terms.
   Measured by interleaving.
3. **Personalized PageRank from clicked concepts as a third rank term**
   (§10). Pure over `build_graph`'s output; synthetic-graph test; mutant =
   degree ranking. Measured by interleaving and by `evaluate_user` finally
   having a non-embedding term to score.
4. **Exposure-sampled negatives** (§3). Training rows from impressions
   with `source='impression'`, weight < a stated judgement; simulated-label
   candidates drawn from impressions too. Measured by `provenance_auc`
   falling from ~1.0 toward 0.5.
5. **MMR on shown rows + community calibration in `ranking_quality`**
   (§7). Measured by the near-duplicate count and the KL number.
6. **Community-arm Thompson slot** (§6). One row in four; measured by the
   slot's engagement rate against exploited rows, from impressions.
7. **Rate-six-items onboarding** (§12) and **KG-path-grounded explanations**
   (§11). Measured on the explanation corpus and by whether a real account
   reaches a two-class history in its first session.
8. **Weekly burst detection over tags** (§9) as a digest section and a
   rank feature. Measured by interleaving once 1-3 are in.
9. **Near-duplicate collapse** (§8) and **recency-weighted clicks** (§9).
   Cheap; measured by counts and by interleaving respectively.

## What this review does not claim

- No citation here has been checked against the literature during this
  review; it was written offline. Verify author, year and venue before
  quoting any of them in a paper or a spec.
- No number here is a prediction of improvement. Every proposal names its
  measurement because this repo's record is that the prior is wrong about
  half the time (`measurement-lessons.md`).
- Nothing here changes the offline guarantee, the response budget, or the
  provenance rules. A proposal that would is out of scope by construction.

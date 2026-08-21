# Recommendation-system refinement survey — hermes-rss

Scope: assess fit for the actual regime, not general recsys practice. Regime: single
machine, 1-3 users, tens-to-hundreds of total clicks, ~1,500 live items, ~900
new/day, CPU-only numpy/sklearn, no training infra, local 8B Ollama chat model
(~5.7s/call warm) used today only for post-hoc explanation.

Current system (`src/hermes/rank.py`): items and a profile-text query both
embedded with `embeddinggemma` (256-dim, unit-norm) via sqlite-vec; candidate
score = `w * rank(classifier_probs) + (1-w) * rank(profile_cosine)`,
`w = n_clicks/(n_clicks+5)`; per-user `LogisticRegression(class_weight="balanced",
C=0.1)` trained on raw embeddings of clicked items; single-class guard
short-circuits to profile-only; 14-day recency window; clicked items excluded
from candidates; eval is leave-last-5-out AUC, `None` under 10 mixed clicks.

---

## 1. Low-data personalization: alternatives to plain LogisticRegression

Regularized logistic over embeddings (current) is already close to the right
complexity class for tens-to-hundreds of points in 256 dims — `C=0.1` is doing
real work as an implicit prior. The candidates worth naming:

- **Bayesian logistic / Laplace approximation**: same decision boundary as MAP
  logistic regression, but keeps a posterior covariance, which is exactly what's
  missing for principled exploration (see §3). Cheap: Newton-Raphson to find the
  mode, Hessian at the mode gives the Gaussian approximation. This is the one
  worth adopting, but bundled with the bandit head in §3 rather than as a
  separate change — a Laplace-approximated logistic head and a LinTS ridge head
  cover overlapping ground, and standing up both would be redundant for this
  regime.
- **Item-item kNN over clicked items**: mean cosine to the k nearest *clicked-useful*
  items. Interpretable, zero training, degrades gracefully at n=1. Useful as a
  cheap explanation signal or a fallback score, not as the primary ranker —
  it doesn't discount clicked-not-useful items the way logistic regression's
  weights naturally do.
- **Rocchio/centroid updates**: `profile_vec += α·mean(useful) - β·mean(not_useful)`.
  This is strictly weaker than what's already running (logistic regression
  finds a max-margin-ish direction; Rocchio just averages) and buys nothing new.
- **EMA profile drift**: exponential decay on the *profile* vector itself so
  recent clicks pull the cosine-similarity anchor without waiting for the
  classifier to accumulate two-class data. This is genuinely different from
  what exists — the profile vector today is static (`user.interests` text,
  re-embedded fresh each rank call) and never learns from clicks directly; only
  the classifier does.

**Verdict: ADOPT NOW (EMA profile drift only)** — cheap, fixes a real gap (profile
is deaf to clicks pre-classifier), composes with everything else.
**ADOPT LATER**: Bayesian/Laplace logistic head — fold into the bandit work in §3.
**SKIP**: kNN as primary signal (redundant with classifier); Rocchio (dominated).

Implementation sketch — EMA profile drift (~15 lines):
```python
# rank.py
def drifted_profile_vec(conn, embedder, user, alpha: float = 0.15) -> np.ndarray:
    """Static interests-text embedding, pulled toward recent useful clicks."""
    base = embedder.embed_query(user["interests"] or user["name"])
    rows = conn.execute(
        "SELECT v.embedding FROM clicks c JOIN item_vectors v ON v.rowid = c.item_id"
        " WHERE c.user_id = ? AND c.useful = 1 ORDER BY c.clicked_at DESC LIMIT 10",
        (user["id"],),
    ).fetchall()
    if not rows:
        return base
    recent = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]).mean(axis=0)
    v = (1 - alpha) * base + alpha * recent
    return v / np.linalg.norm(v)
```
Swap the `embedder.embed_query(...)` call in `rank_items` for this. `alpha`
tunable; start at 0.15 so it nudges rather than overrides the stated interests.

---

## 2. Pairwise/ranking losses (BPR, hinge-rank) at small n

Pairwise losses are the standard recsys-literature answer for implicit
feedback and do have better generalization scaling on very sparse signals in
principle. But the field's motivating case is *unlabeled* implicit feedback
(clicks-only, no explicit negatives, thousands of items per user) — this
system already has explicit y/n labels, which is a strictly richer signal than
what BPR is designed to extract from. Implementing BPR here means hand-rolling
SGD over `(useful, not_useful)` pairs with no sklearn primitive to lean on,
against a training set of tens-to-hundreds of points where pair-sampling noise
would dominate any ranking-loss benefit. The thing BPR buys — not needing
explicit negatives — is not the bottleneck this system has.

**Verdict: SKIP.** Solves a data-sparsity problem this system doesn't have
(explicit labels already exist); would add real implementation complexity
(custom SGD loop, pair sampling, convergence tuning) for a benefit that's
speculative at this n. Revisit only if the click model shifts to implicit-only
signals (e.g., dwell time, opens without explicit rating).

---

## 3. Exploration: LinTS/LinUCB contextual bandit

This is the strongest candidate in the survey. The system's actual weak point
today isn't the model class — it's that ranking is **pure exploitation**:
every rank call is `argmax` over the blended score, so items far from the
current profile/classifier direction never surface long enough to get
labeled, and the classifier can only refine a direction it's already been
shown. A contextual bandit (item embedding = context, click = reward) with
principled uncertainty is exactly the right fix, and it's cheap:
Sherman-Morrison keeps the per-round update at O(d²) with d=256, well within
numpy budget at 1,500 items/rank-call.

- **LinUCB** (deterministic upper-confidence-bound) vs **LinTS** (Thompson
  sampling: draw θ ~ N(θ̂, A⁻¹), score by dot product): LinTS is simpler to
  implement (no confidence-radius tuning parameter `α` to hand-pick) and
  empirically matches or beats LinUCB on regret in most reported comparisons,
  at the cost of one Cholesky/eigendecomp per round for the sample — trivial
  at d=256. LinTS is the better fit here: fewer knobs, naturally graceful with
  small n (wide posterior → naturally exploratory), and it drops in as a
  *replacement for the LogisticRegression classifier head*, not an addition —
  same input (clicked-item embeddings + labels), same output (a score per
  candidate), but ridge-regression math instead of iterative logistic fit, and
  the posterior gives you calibrated exploration for free.
- This also symbiotically closes the gap identified in §6 (position bias):
  bandit exploration is the standard mechanism for *preventing* position bias
  from calcifying, not just correcting it after the fact.

**Verdict: ADOPT NOW.**

Implementation sketch — LinTS head (~40-50 lines, numpy only):
```python
# rank.py
class LinTS:
    """Thompson-sampled ridge regression bandit head. Replaces/augments
    the LogisticRegression classifier score."""

    def __init__(self, d: int, lam: float = 1.0, sigma: float = 1.0):
        self.A = lam * np.eye(d, dtype=np.float64)  # precision matrix
        self.b = np.zeros(d, dtype=np.float64)  # A @ theta_hat
        self.sigma = sigma

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinTS":
        # y in {0,1} treated as reward; closed-form ridge update, no iteration
        self.A += X.T @ X
        self.b += X.T @ (2 * y - 1)  # map to {-1,+1} reward
        return self

    def sample_scores(self, X_candidates: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        A_inv = np.linalg.inv(self.A)
        theta_hat = A_inv @ self.b
        theta_sample = rng.multivariate_normal(theta_hat, self.sigma**2 * A_inv)
        return X_candidates @ theta_sample


def bandit_scores(conn, user_id: int, X: np.ndarray, d: int = 256) -> np.ndarray | None:
    X_train, y = _click_training_data(conn, user_id)
    if y is None or len(set(y.tolist())) < 2:
        return None
    head = LinTS(d).fit(X_train.astype(np.float64), y.astype(np.float64))
    rng = np.random.default_rng()  # fresh draw per rank call = exploration
    return head.sample_scores(X.astype(np.float64), rng)
```
Wire into `rank_items` as a third ranked term alongside profile and (initially,
in parallel for comparison) the existing classifier — blend weight can reuse
`blend_weight(n_clicks)`. Note: `A` is rebuilt from scratch each call here
(cheap at this n — hundreds of rows, O(n·d²) per rank request is
sub-millisecond); no need for incremental Sherman-Morrison updates until click
volume is orders of magnitude higher.

Single-class guard carries over unchanged (`len(set(y)) < 2 → None`).

---

## 4. Diversity/serendipity: MMR, topic caps, submodular selection

Cheapest, most demo-visible win in the whole survey. Right now nothing stops
the top-N from being five near-duplicate items from the same feed on the same
story — a classic embedding-similarity failure mode, and a *daily digest* is
exactly the setting MMR was designed for (Carbonell & Goldstein 1998,
document/summary selection). Submodular selection is the principled
generalization of MMR but adds complexity (coverage functions, lazy greedy)
that isn't justified until the candidate pool or the diversity requirements
get more structured than "don't show duplicates." Per-source topic caps are
even simpler than MMR but blunter — they cap by feed metadata, not semantic
similarity, so two different feeds covering the same story still both get
through.

**Verdict: ADOPT NOW (MMR).** Runs entirely on embeddings already in memory,
greedy, no new dependencies, O(N·k) at N≈1500 candidates and k≈20-30 displayed
— trivial cost. **SKIP submodular** as overkill for this regime. **ADOPT LATER
(maybe never)**: per-source cap as a supplementary hard constraint only if MMR
alone doesn't fix source-clustering in practice.

Implementation sketch (~20 lines):
```python
# rank.py
def mmr_rerank(X: np.ndarray, relevance: np.ndarray, k: int, lam: float = 0.7) -> list[int]:
    """Greedy MMR over already-scored candidates. relevance: higher=better,
    same length/order as X rows. Returns indices into X, len <= k."""
    remaining = list(range(len(X)))
    selected: list[int] = []
    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            sel_vecs = X[selected]  # rows are unit-norm -> dot = cosine

            def mmr_score(i):
                sim_to_selected = (X[i] @ sel_vecs.T).max()
                return lam * relevance[i] - (1 - lam) * sim_to_selected

            best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)
    return selected
```
Apply only to the top ~50 candidates by blended score (not all 1,500) before
truncating to the displayed page — keeps it O(50·k) not O(1500·k), and avoids
diversity-selecting items that were never going to be relevant. `lam=0.7`
biases toward relevance, tune by eyeballing the feed.

---

## 5. Calibration + blending: rank-blend vs score calibration vs BMA

Rank-blending (current) sidesteps calibration entirely by construction —
ranks are comparable across models even when raw scores aren't (cosine
similarity and predict_proba live on totally different scales, which is
exactly why rank-blend was the right initial choice). Platt scaling needs a
held-out calibration set to fit the sigmoid, which at n<100 either eats into
already-scarce training data or overfits the calibration curve itself;
isotonic regression is even more data-hungry (nonparametric, needs monotonic
structure to hold over enough bins) and is a bad fit below a few hundred
points per class. Bayesian model averaging of profile+classifier requires a
principled likelihood for each model's output, which cosine similarity
doesn't naturally have (it's not a probability) — you'd end up building an ad
hoc calibration step to get there anyway, which is the same problem BMA is
supposed to avoid.

**Verdict: SKIP all three for now — current rank-blend is the correct choice
for this regime**, not a stopgap. Revisit isotonic/Platt only if a downstream
consumer needs actual probability estimates (e.g., a numeric "match %" shown
to the user) rather than a ranking — that's a product decision, not a
modeling gap.

---

## 6. Position bias in implicit feedback

Real mechanism, marginal impact at this scale. Position bias matters most
when (a) the ranker is confident and consistent call-to-call, so the same
items keep landing at rank 1-3 and hoovering up clicks regardless of true
relevance, and (b) the candidate pool is large enough that good items
plausibly get buried past where users scroll. At 1-3 users clicking tens of
items against a ranker that's already noisy (LogisticRegression refit from
scratch each call on a moving click set, `C=0.1` heavy regularization,
different random top-N each day as new items arrive) there's naturally more
score churn than a stable production ranker would have — some organic
position-shuffling already happens as a side effect of the system being
small and dynamic, not because it was designed for debiasing.

The cheapest real correction is **result randomization**, not inverse-propensity
weighting: IPW requires a propensity model (probability of examining rank r)
that itself needs calibration data this system doesn't have and isn't worth
building for tens of clicks. Randomization needs nothing but a coin flip.

**Verdict: ADOPT LATER**, and only in the specific form of *light position
jitter in the top-K* (e.g., small Gumbel noise added to blended scores before
final sort, or restricting strict-relevance-order to top 3 and shuffling
ranks 4-15) — cheap, no propensity model needed, directly increases the
diversity of what gets a chance to be clicked. Not urgent: the bandit
exploration in §3, once shipped, subsumes most of the benefit here (score
uncertainty already produces rank churn), so implement this only if clicks
visibly concentrate on top-3 after §3 ships and position bias still looks
like the explanation.
**SKIP** inverse-propensity weighting — the propensity model it requires costs
more to build and validate than the correction is worth at this click volume.

---

## 7. LLM-in-the-loop beyond explanation

At ~5.7s/call on the fast model (`hermes3:8b`), the LLM budget is "acceptable
for one lazy, cached, non-blocking call per item" (per README's own
reliability contract) and nothing more. Evaluate each proposed use against
that budget:

- **LLM-as-labeler for cold-start pseudo-clicks**: `bootstrap_persona()`
  already solves cold start via profile-similarity pseudo-clicks with zero LLM
  calls and zero latency. Routing this through the LLM (e.g., "would this
  persona click this?") would trade a free, instant heuristic for a slow,
  non-deterministic one, for a demo-only feature. No benefit identified.
- **LLM profile summarization into interests text**: this already exists —
  `explain.py`'s `synthesize_profile` node does exactly this (summarizes
  recent useful titles into one sentence) but currently only as scratch state
  *inside* the explain graph, thrown away after generating one explanation.
  Persisting that summary back into `user.interests` (or into the new EMA
  profile from §1) turns a wasted computation into a second, semantically
  richer profile-drift signal — same LLM call already being paid for on the
  explain path, just also written to the DB. This is nearly free because the
  cost is already sunk.
- **LLM reranking of the top-20**: listwise LLM reranking research this year
  shows real quality gains but at real latency cost, and it's aimed at
  correcting *retrieval* mistakes (bringing in items an embedding model missed
  entirely) — this system's ranking is already list-complete (all 1,500
  candidates scored), so LLM reranking would only be reordering a set that's
  already reasonably ordered. At 5.7s/call, reranking 20 items either means 20
  sequential calls (110+ seconds, dead on arrival for an interactive feed) or
  one big listwise call stuffing 20 items into context (single ~6-10s call,
  more plausible) — but with no evidence yet that the embedding-based rank
  order has systematic top-20 errors worth paying for.

**Verdict:**
**ADOPT NOW**: persist the explain-graph's profile summary as a second
profile-drift input (reuses existing sunk LLM cost, ~10 lines).
**SKIP**: LLM-as-labeler (dominated by existing free heuristic).
**ADOPT LATER**: LLM listwise reranking of top-20 — worth trying once there's
click evidence that embedding rank-order is systematically wrong in ways MMR/
bandit exploration don't already fix; budget one blocking ~8s call, cached
like explanations, never on the hot path.

Implementation sketch — persist profile summary (~10 lines, in `explain.py`):
```python
# explain.py, inside synthesize_profile, after computing the summary:
conn.execute(
    "UPDATE users SET llm_profile_summary = ? WHERE id = ?",
    (Explanation.model_validate(out).text, state.user_id),
)
conn.commit()
```
Requires one migration: `ALTER TABLE users ADD COLUMN llm_profile_summary TEXT`.
Then in `rank.py`'s profile embedding step, blend `llm_profile_summary` (if
present) into the text passed to `embedder.embed_query`, alongside
`interests` — e.g. concatenate both strings before embedding. Degrades to
current behavior when the column is null (never yet explained).

---

## 8. Session/freshness: time-decayed clicks, novelty boosts

Two sub-questions. Time-decayed clicks (down-weighting old clicks in
classifier/bandit training so the model tracks drifting interest) — real
value, cheap, and cleanly folds into the LinTS update in §3 (weight each
training row's contribution to `A`/`b` by a recency multiplier instead of
treating all history as equally fresh). Novelty boosts (explicitly promoting
items dissimilar from anything the user has seen/clicked before, independent
of relevance) — this is a *distinct* goal from diversity-within-a-single-feed
(§4's MMR); it's closer to exploration (§3) than to diversity, and the bandit
head already provides exploration via posterior uncertainty. Adding a
separate novelty boost on top would be solving the same problem twice with
two different mechanisms.

**Verdict: ADOPT NOW (recency-weighted training)**, bundled as a parameter on
the §3 LinTS `fit()`, not a separate system. **SKIP standalone novelty boost**
— redundant with bandit exploration; revisit only if post-launch behavior
shows the bandit converging too fast on a narrow interest cluster despite
exploration (unlikely at this data volume, where posterior stays wide for a
long time anyway).

Implementation sketch — recency weighting (~10 lines, modifies the LinTS.fit
call site):
```python
# rank.py, in _click_training_data or a new variant:
def _click_training_data_weighted(conn, user_id: int, half_life_days: float = 21.0):
    rows = conn.execute(
        "SELECT c.useful, v.embedding, c.clicked_at FROM clicks c"
        " JOIN item_vectors v ON v.rowid = c.item_id WHERE c.user_id = ?",
        (user_id,),
    ).fetchall()
    if not rows:
        return None, None, None
    y = np.array([r["useful"] for r in rows], dtype=np.float64)
    X = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    age_days = np.array([_age_in_days(r["clicked_at"]) for r in rows])
    weights = 0.5 ** (age_days / half_life_days)
    return X, y, weights
```
In `LinTS.fit`, scale each row's contribution: `self.A += (X * weights[:, None]).T @ X`
and `self.b += (X * weights[:, None]).T @ (2*y - 1)`. `half_life_days=21`
means a 3-week-old click carries half the weight of a fresh one — tune against
the 14-day recency window already used for candidate selection so the two
timescales are roughly consistent.

---

## Ranked next five changes

Optimized for demo-visible learning speed and daily-use quality, in build order:

1. **LinTS contextual bandit head** (§3) — replaces the LogisticRegression
   classifier with a Thompson-sampled ridge head; turns pure exploitation into
   principled explore/exploit, the single biggest lever on click-efficiency.
   ~50 lines.
2. **MMR re-ranking of top candidates** (§4) — most demo-visible change for
   least effort; stops near-duplicate items from dominating the digest.
   ~20 lines.
3. **EMA profile drift from recent useful clicks** (§1) — makes the
   profile-cosine term responsive to clicks immediately, not just after the
   classifier has two-class data. ~15 lines.
4. **Recency-weighted bandit training** (§8) — half-life decay on click
   contributions to `A`/`b`; keeps the model tracking drifting interest
   instead of averaging over all history equally. ~10 lines, folds into #1.
5. **Persist LLM profile summary from the explain graph into ranking** (§7) —
   reuses an LLM call already being paid for on the explain path; adds a
   second, semantically richer profile signal at near-zero marginal cost.
   ~10 lines + 1 migration.

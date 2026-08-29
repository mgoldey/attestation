## What you get

Twenty hand-built rows ranked with `attestation.rank.rank_rows` -- the same
pure blend `rank_items` calls once it has real rows out of SQLite, exercised
here on literal vectors instead: no database, no embedder, no model server.
Two clusters of unit vectors stand in for "items about protein folding" and
"items about graph learning"; a profile vector sits on the protein axis, and
six labelled clicks (four `useful` near protein, two `not useful` near
graph) stand in for a click history. The script prints the classifier-only
AUC on those six clicks beside the rank-order AUC of the full blended list --
the two numbers `evaluate_user`'s own "measures" string says are not the
same question.

## Prerequisites

`none — pure local computation`

numpy and scikit-learn are already dependencies (`uv run python -c "import
sklearn"` to check); nothing here talks to a network or a model.

## Run it

```bash
./run.sh
```

or, from the repo root:

```bash
uv run python examples/ranking/rank_rows.py
```

## What it prints

```
classifier-only AUC on the six clicks: 1.000 -- measures the click classifier alone, NOT the profile or preference terms
```

Full output: the profile-only top 5 (ranked by `profile_similarity` alone,
`click_rows=None`); the blended top 5 once the six clicks are folded in
(`n_clicks=6`, printed `blend_weight`); the classifier-only AUC line pinned
above; and a final line, `blended-order AUC over all twenty rows (hit =
protein cluster): 0.970`.

## What it demonstrates

`rank_rows` is a pure function of rows, a profile vector, and click rows --
no connection, no embedder, so it can be exercised on literal vectors in a
few milliseconds. Folding in the six clicks visibly changes the ranking:
`protein-10`, at profile_similarity 0.574 and well outside the profile-only
top 5, enters the blended top 5 because the click classifier favours it,
displacing `protein-9` (profile_similarity 0.826), which the classifier does
not.

The two AUCs printed at the end measure different things, and the gap
between them is the point. The classifier-only AUC (1.000) is
`classifier_probs` scored on the same six clicks it was trained on -- it
says the classifier separates its own training rows perfectly, and nothing
about the other fourteen rows or the profile term. The blended-order AUC
(0.970) is the rank-order of the full twenty-row list against cluster
membership -- what a reader actually sees, after profile similarity and the
classifier are both folded in. `evaluate_user`'s own "measures" string in
`src/attestation/rank.py` makes the same point about the live ranking: a
perfect classifier score is not a claim about the list a reader is handed.

## When it goes wrong

- `ModuleNotFoundError: No module named 'sklearn'` -- run `uv sync`; numpy
  and scikit-learn are both declared dependencies, not extras.
- A different AUC or a different top 5 than the ones pinned above --
  `rank_rows.py` seeds `numpy.random.default_rng(0)`, so the rows, the
  clicks and every printed number are exactly reproducible; a difference
  means the script or the seed changed, not that this run drew new rows.

## Next

`examples/flows/` runs the live end-to-end AUC over forty labelled items and
a real persona, instead of twenty hand-built rows.

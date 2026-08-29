"""Twenty hand-built rows, ranked with `rank.rank_rows` -- no database, no
model server. `attestation.rank.rank_rows` is the pure function `rank_items`
calls once it has real rows out of SQLite; this script hands it literal
vectors instead, the way `rank.py`'s own docstring says the blend can be
exercised.

Two clusters of unit vectors stand in for "items about protein folding" and
"items about graph learning": ten rows noised around a `protein_axis`, ten
around a `graph_axis`. A profile vector sits on the protein axis (a persona
whose stated interest is protein folding), and six labelled clicks (four
`useful` near protein, two `not useful` near graph) stand in for a click
history.

Prints, in order: the profile-only order (no clicks); the blended order
(clicks folded in via `blend_weight`); the classifier-only AUC on the six
clicks (`classifier_probs` fit and scored against its own training rows, via
`sklearn.metrics.roc_auc_score`); and the rank-order AUC of the full blended
list against cluster membership. The two AUCs are not the same question --
`evaluate_user`'s own "measures" string says why -- and this script prints
both side by side so the difference is visible in one run rather than
asserted in prose.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from attestation.rank import blend_weight, classifier_probs, rank_rows

N_DIM = 8
N_PER_CLUSTER = 10
# Spread within a cluster. Small values (e.g. 0.25) separate the two
# clusters so cleanly that both AUCs below print 1.000 -- correct, but it
# hides the point of printing two numbers instead of one. 0.7 is the
# smallest scale (checked in steps against this same seed) at which the
# blended list pulls in a classifier-favoured row ahead of a purely
# profile-similar one, so the blended-order AUC is honestly imperfect while
# the classifier-only AUC, scored on the six rows it was trained on, stays
# at its ceiling.
CLUSTER_NOISE_SCALE = 0.7


def _unit(vec: np.ndarray) -> np.ndarray:
    """Normalise to a float32 unit vector -- the shape `rank_rows` expects."""
    return (vec / np.linalg.norm(vec)).astype(np.float32)


def _build_rows(rng: np.random.Generator) -> tuple[list[dict], np.ndarray, dict[int, str]]:
    """Twenty rows in two clusters, plus the profile vector and each row's
    cluster label (for scoring only -- `rank_rows` never sees the label)."""
    protein_axis = _unit(rng.normal(size=N_DIM))
    graph_axis = _unit(rng.normal(size=N_DIM))

    rows: list[dict] = []
    cluster_of: dict[int, str] = {}
    for cluster, axis in (("protein", protein_axis), ("graph", graph_axis)):
        for i in range(N_PER_CLUSTER):
            item_id = len(rows) + 1
            noise = rng.normal(scale=CLUSTER_NOISE_SCALE, size=N_DIM)
            rows.append(
                {
                    "id": item_id,
                    "title": f"{cluster}-{i + 1}",
                    "url": None,
                    "source": "demo",
                    "summary": f"a {cluster} item",
                    "embedding": _unit(axis + noise),
                }
            )
            cluster_of[item_id] = cluster

    profile_vec = protein_axis.astype(np.float32)
    return rows, profile_vec, cluster_of


def _print_top5(label: str, ranked) -> None:
    print(f"\n{label}")
    for item in ranked[:5]:
        sim = item.profile_similarity
        print(f"  {item.item_id:>2}  {item.title:<12} profile_similarity={sim:.3f}")


def main() -> None:
    """Build the rows, rank them twice (profile-only, then blended with
    clicks), and print both AUCs the blend produces."""
    rng = np.random.default_rng(0)
    rows, profile_vec, cluster_of = _build_rows(rng)
    by_id = {r["id"]: r for r in rows}

    # Four "useful" clicks near the protein axis, two "not useful" near
    # graph -- a small but two-class click history, the minimum
    # `classifier_probs` needs to fit at all.
    useful_ids = [1, 2, 3, 4]
    not_useful_ids = [11, 12]
    click_rows = [{"useful": True, "embedding": by_id[i]["embedding"]} for i in useful_ids] + [
        {"useful": False, "embedding": by_id[i]["embedding"]} for i in not_useful_ids
    ]
    n_clicks = len(click_rows)

    profile_only = rank_rows(rows, profile_vec, None, None, 0)
    _print_top5("profile-only order (click_rows=None), top 5:", profile_only)

    w = blend_weight(n_clicks)
    blended = rank_rows(rows, profile_vec, click_rows, None, n_clicks)
    _print_top5(f"blended order (n_clicks={n_clicks}, blend_weight={w:.3f}), top 5:", blended)

    # (1) Classifier-only AUC: fit on the six clicks, scored back on those
    # same six rows -- exactly what evaluate_user's holdout AUC generalises
    # with a StratifiedKFold; this script skips the split because six rows
    # is too few to hold any out and still train.
    X_clicks = np.stack([np.asarray(r["embedding"], dtype=np.float32) for r in click_rows])
    y_clicks = np.array([int(r["useful"]) for r in click_rows])
    click_probs = classifier_probs(click_rows, X_clicks)
    classifier_auc = roc_auc_score(y_clicks, click_probs)
    print(
        f"\nclassifier-only AUC on the six clicks: {classifier_auc:.3f}"
        " -- measures the click classifier alone, NOT the profile or preference terms"
    )

    # (2) Rank-order AUC of the full blended list against cluster
    # membership (a "hit" is a protein-cluster row) -- this is the number a
    # reader actually feels: how well the list they were handed separates
    # the two topics, after profile similarity AND the classifier are both
    # folded in.
    blended_ids = [item.item_id for item in blended]
    labels = np.array([1 if cluster_of[i] == "protein" else 0 for i in blended_ids])
    position_scores = np.array([-pos for pos in range(len(blended_ids))])
    blended_auc = roc_auc_score(labels, position_scores)
    print(f"blended-order AUC over all twenty rows (hit = protein cluster): {blended_auc:.3f}")


if __name__ == "__main__":
    main()

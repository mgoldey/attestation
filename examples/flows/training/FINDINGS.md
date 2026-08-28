# c_sweep findings

Four arms of `LogisticRegression` on scikit-learn's breast-cancer set, C in [0.01, 0.1, 1.0, 10.0].
The best held-out AUC was C=10.0 at 0.9960.

- C=0.01: AUC 0.9864, precision 0.9467, recall 0.9861 <!-- claim: training/c_sweep/879a73ad metric=auc value=0.9864 tol=0.00005 -->
- C=0.1: AUC 0.9940, precision 0.9474, recall 1.0000 <!-- claim: training/c_sweep/b677f691 metric=auc value=0.9940 tol=0.00005 -->
- C=1.0: AUC 0.9957, precision 0.9730, recall 1.0000 <!-- claim: training/c_sweep/0e564d9a metric=auc value=0.9957 tol=0.00005 -->
- C=10.0: AUC 0.9960, precision 0.9726, recall 0.9861 <!-- claim: training/c_sweep/ddc43d66 metric=auc value=0.9960 tol=0.00005 -->

### Deliberately wrong claim, for the demo

- The best arm reached AUC 0.9460 <!-- claim: training/c_sweep/ddc43d66 metric=auc value=0.9460 -->

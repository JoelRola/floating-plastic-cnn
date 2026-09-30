# Results

## A. Historical dissertation results

These values are historical, report-derived results and are not outputs of the refactored code:

| Metric | Historical reported value |
|---|---:|
| Baseline validation accuracy | 74.3% |
| Baseline specificity | 0% |
| 1:1 balancing specificity | 44% |
| Final reported specificity | 100% |
| Final reported sensitivity | 100% |
| Severity MAE | 5.3 percentage points |
| Mean classification confidence | 93% |

Specificity and sensitivity were not both derived solely from the positive-only UGV set. The reported 0% specificity shows how overall accuracy can hide a majority-class failure mode.

## B. Reproduced portfolio results

### FloPWD-only control

The full unbalanced FloPWD baseline is complete and is the control for future comparisons. Its run artifacts remain local under ignored `runs/flopwd_original_seed42` and were not altered in Phase 5.

| Test metric | Portfolio control |
|---|---:|
| Accuracy | 96.0% |
| Balanced accuracy | 96.05% |
| Sensitivity | 95.95% |
| Specificity | 96.15% |
| Severity MAE | 2.17 percentage points |

This is not an exact historical replication: it uses a deterministic stratified 70/15/15 split, official ResNet50 preprocessing, bounded severity output, and a separate held-out test protocol.

### Combined FloPWD + UGV

Combined training and evaluation are pending. The grouped UGV manifest and masked multi-task/domain-sampling framework are prepared, but no combined metrics are available. Do not interpret the profile definitions as experimental results.

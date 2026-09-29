# Results

## A. Historical dissertation results

These values are reported in the dissertation materials and have not yet been reproduced by this repository:

| Metric | Historical reported value |
|---|---:|
| Baseline validation accuracy | 74.3% |
| Baseline specificity | 0% |
| 1:1 balancing specificity | 44% |
| Final reported specificity | 100% |
| Final reported sensitivity | 100% |
| Severity MAE | 5.3 percentage points |
| Mean classification confidence | 93% |

Specificity and sensitivity were not both derived solely from the positive-only UGV set. The original evaluation construction and other caveats are described in [reproducibility](reproducibility.md).

The reported 0% baseline specificity indicates failure to correctly classify negative examples. This failure mode is obscured when only overall accuracy is considered.

## B. Reproduced portfolio results

Reproduction is pending. The refactored pipeline is implemented, but no training or test evaluation has been run, so no portfolio metric is reported here. The protocol uses a separate deterministic, stratified FloPWD 70/15/15 filename manifest for train, validation, and untouched test partitions. Severity MAE will be computed in percentage points with sample counts, for all test samples and separately for plastic-positive samples. UGV remains a future OOD source and will not provide specificity in the absence of clean-water negatives. Results will be added only after an actual run and its split, label semantics, preprocessing, and outputs have been checked and recorded.

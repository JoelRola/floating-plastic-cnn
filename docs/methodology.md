# Methodology

## Historical dissertation methodology

The dissertation frames two image-level tasks: binary plastic-presence classification and regression of plastic foreground coverage as a percentage of image area. The stated datasets are FloPWD and UGV-NBWASTE. Historical metrics and limitations are recorded separately in [results](results.md) and [reproducibility](reproducibility.md). This target is image-area coverage, not a physical estimate of water-surface area.

## Model concept

The documented architecture is a shared ImageNet-pretrained ResNet50 feature extractor with a sigmoid binary-classification head and a linear coverage-regression head. Inputs are 224 × 224 RGB images. The cleaned model factory applies the ResNet50 application preprocessing in the model graph. The appendix instead scales pixels to [0, 1]; this implementation difference requires controlled validation before comparison. FloPWD coverage labels and the reusable severity MAE use percentage points on a 0–100 scale. The local label audit also found eight positive binary labels paired with zero severity; this discrepancy is retained and disclosed.

## Class balance and evaluation

## Portfolio reproduction protocol

FloPWD has no local official train/validation/test split. The proposed portfolio split is deterministic 70/15/15, stratified by binary image label, with a configurable seed and a filename-only JSON manifest in `experiments/splits/`. The initial inspected-data manifest uses seed 42. Its filenames and seed are not claimed to match the dissertation. Near-duplicate imagery cannot be detected from tabular metadata alone and remains a limitation; inspect filename/source grouping before training.

Balancing is a training-set operation and must follow splitting. Compare balanced and unbalanced runs on the same training records, validation protocol, epoch budget, and untouched test manifest. Use validation only for model selection and threshold selection; reserve test data until final evaluation.

The classification question is in-distribution FloPWD binary classification. Report accuracy, sensitivity, specificity, and balanced accuracy with sample counts. Severity regression uses the same held-out FloPWD filenames and reports MAE in percentage points (0–100) and sample count. Preserve the eight positive/zero-severity cases and report their presence.

Treat UGV as a separate cross-domain dataset. It has oriented object annotations, while the current project head is image-level classification/regression. Restore the exact local class mapping and define a defensible image-level target before evaluating this model on UGV. If the target is defined so selected UGV examples are all positive, positive-class recall can be reported; specificity cannot be estimated from UGV alone. Do not combine FloPWD negatives with UGV positives into one benchmark.

Cross-domain results are affected by geography, camera perspective, background, image scale, acquisition method, and the UGV split/export overlap. Any such evaluation is descriptive OOD analysis, not a clean causal estimate of dataset-origin generalization.

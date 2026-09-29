# Methodology

## Historical dissertation methodology

The dissertation frames two image-level tasks: binary plastic-presence classification and regression of plastic foreground coverage as a percentage of image area. The stated datasets are FloPWD and UGV-NBWASTE. Historical metrics and limitations are recorded separately in [results](results.md) and [reproducibility](reproducibility.md). This target is image-area coverage, not a physical estimate of water-surface area.

## Portfolio model and preprocessing

The cleaned architecture uses a shared ResNet50 feature extractor (`include_top=False`), GlobalAveragePooling2D, shared dense/dropout layers, a sigmoid binary-classification head, and a bounded coverage-regression head. Inputs are 224 x 224 RGB images. The cleaned model embeds official `keras.applications.resnet.preprocess_input` in the model graph. Historical appendix preprocessing scales pixels by `/255`; this is a methodological difference requiring separate validation. Severity uses sigmoid scaled by 100, so output and targets are percentage points in [0, 100]. The local label audit found eight positive binary labels paired with zero severity; the portfolio pipeline preserves these source labels.

The initial configuration freezes the ImageNet-pretrained ResNet50 backbone, uses binary cross-entropy for classification, MAE for severity, and loss weights 1.0 and 0.5. These settings are a starting point, not a claim of methodological identity with the dissertation. Backbone fine-tuning and losses are configurable. The TensorFlow input pipeline streams images on demand, resizes them, shuffles only training records using the configured seed, and batches/prefetches. The same model-embedded preprocessing is used during training, evaluation, and prediction.

## Class balance and evaluation

Balancing is a training-set operation and must follow splitting. Compare balanced and unbalanced runs on the same training records, validation protocol, epoch budget, and untouched test manifest. Use validation only for model selection and threshold selection; reserve test data until final evaluation. The current training CLI provides the unbalanced baseline; balancing comparisons remain a subsequent extension.

## Portfolio reproduction protocol

FloPWD has no local official train/validation/test split. The portfolio uses a deterministic 70/15/15 split, stratified by binary image label, with a configurable seed and filename-only JSON manifest in `experiments/splits/`. The seed-42 manifest is a portfolio split, not the dissertation split. Near-duplicate imagery cannot be detected from tabular metadata alone and remains a limitation; inspect filename/source grouping before training.

The classification question is in-distribution FloPWD binary classification. Report accuracy, sensitivity, specificity, balanced accuracy, precision, F1, confusion counts, and sample count. Severity regression uses held-out FloPWD filenames and reports MAE and RMSE in percentage points with sample counts, for all test images and separately for plastic-positive images. The positive-only MAE is conditional on the binary target and should not replace the all-image regression result. Preserve and disclose the eight positive/zero-severity cases.

Treat UGV as a separate future cross-domain dataset. It has oriented object annotations, while this model has image-level classification/regression heads. Restore the exact local class mapping and define a defensible image-level target before evaluating this model on UGV. If selected UGV examples are all positive, positive-class recall can be reported; specificity cannot be estimated from UGV alone. Do not combine FloPWD negatives with UGV positives into one benchmark.

Cross-domain results are affected by geography, camera perspective, background, image scale, acquisition method, and UGV export/split overlap. Any valid UGV evaluation is descriptive OOD analysis, not a clean causal estimate of dataset-origin generalization.

# Model card

## Intended use

Research and educational exploration of floating-plastic image classification and image-area coverage estimation. This project is not validated for operational environmental monitoring or policy decisions.

## Model

The implemented architecture uses ImageNet-pretrained ResNet50 transfer learning (`include_top=False`), global average pooling, shared dense/dropout layers, and binary classification and bounded coverage regression heads. The default freezes the backbone. The classification head is sigmoid; the regression head is sigmoid scaled to 0-100 percentage points. No validated checkpoint is included.

The cleaned model applies official `keras.applications.resnet.preprocess_input` to resized RGB pixels. The historical appendix instead divided pixels by 255. This difference is explicit; historical metrics must not be attributed to this implementation.

## Data

The dissertation describes FloPWD and UGV-NBWASTE. The local FloPWD labels include eight positive binary labels with zero coverage. The local UGV copy has no `data.yaml`; class IDs 0–6 are visible, but the semantic mapping is unresolved. Dataset licences remain with their original sources.

## Evaluation and limitations

Historical dissertation figures are not reproduced results. Portfolio evaluation uses a deterministic stratified FloPWD train/validation/test filename manifest and an untouched test split. UGV is not scored in this phase; future OOD analysis requires its target mapping to be verified. UGV alone cannot estimate specificity for plastic-versus-clean-water classification. Local UGV variants share apparent source IDs across splits. Protocol concerns are documented in [`reproducibility.md`](reproducibility.md). Performance across locations, conditions, and devices is not established. The target runtime is Python 3.10 with TensorFlow 2.15.x; the lightweight test environment does not contain TensorFlow.

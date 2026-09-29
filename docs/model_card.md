# Model card

## Intended use

Research and educational exploration of floating-plastic image classification and image-area coverage estimation. This project is not validated for operational environmental monitoring or policy decisions.

## Model

The intended architecture uses ImageNet-pretrained ResNet50 transfer learning with binary classification and coverage regression heads. The cleaned implementation is a model factory; no newly validated checkpoint is included.

## Data

The dissertation describes FloPWD and UGV-NBWASTE. The local FloPWD labels include eight positive binary labels with zero coverage. The local UGV copy has no `data.yaml`; class IDs 0–6 are visible, but the semantic mapping is unresolved. Dataset licences remain with their original sources.

## Evaluation and limitations

Historical dissertation figures are not reproduced results. Portfolio evaluation is designed around a deterministic stratified FloPWD train/validation/test split, with UGV treated as a separate OOD dataset only after its target mapping is verified. UGV alone cannot estimate specificity for plastic-versus-clean-water classification. Local UGV variants share apparent source IDs across splits. Evaluation protocol concerns and preprocessing uncertainty are documented in [`reproducibility.md`](reproducibility.md). Performance across locations, conditions, and devices is not established.

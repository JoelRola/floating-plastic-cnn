# Model card

## Intended use

Research and educational exploration of floating-plastic image classification and image-area coverage estimation. This project is not validated for operational environmental monitoring or policy decisions.

## Model

The intended architecture uses ImageNet-pretrained ResNet50 transfer learning with binary classification and coverage regression heads. The cleaned implementation is a model factory; no newly validated checkpoint is included.

## Data

The dissertation describes FloPWD and UGV-NBWASTE. Dataset licences remain with their original sources. The UGV class-ID mapping used for binary plastic labels has not yet been verified in this repository.

## Evaluation and limitations

Historical dissertation figures are not reproduced results. Evaluation protocol concerns and preprocessing uncertainty are documented in [`reproducibility.md`](reproducibility.md). Performance across locations, conditions, and devices is not established.

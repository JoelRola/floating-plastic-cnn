# Methodology

## Tasks

The dissertation frames two image-level tasks: binary plastic-presence classification and regression of plastic foreground coverage as a percentage of image area. This is image-area coverage, not a physical estimate of water-surface area.

## Model concept

The documented architecture is a shared ImageNet-pretrained ResNet50 feature extractor with a sigmoid binary-classification head and a linear coverage-regression head. Inputs are 224 × 224 RGB images. The cleaned model factory applies the ResNet50 application preprocessing in the model graph. The appendix instead scales pixels to [0, 1]; this implementation difference requires controlled validation before comparison. FloPWD coverage labels and the reusable severity MAE use percentage points on a 0–100 scale.

## Class balance and evaluation

Balancing is a training-set operation and must follow dataset splitting. Evaluation should report classification metrics and coverage MAE on explicitly described held-out data, with source/dataset composition stated. UGV class IDs require a verified mapping before use. See [reproducibility](reproducibility.md) and [results](results.md).

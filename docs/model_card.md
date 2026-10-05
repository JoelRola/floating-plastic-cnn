# Model card

## Intended use

Research and educational exploration of floating-plastic image classification, cross-domain annotated-waste positive detection, and FloPWD image-area coverage estimation. These models are not validated for operational monitoring, automated enforcement, or policy decisions.

## Model families and capabilities

All models use ImageNet ResNet50 features with official Keras preprocessing and a frozen backbone. A/B/C/E use the shared trainable tower; D/F use independent task-specific towers after the frozen shared representation.

- **Model A:** validated functional FloPWD severity head. This is the only model in the completed matrix whose coverage estimate should be presented as a usable portfolio prediction.
- **Models B/C/D/E/F:** classification models with severity heads that collapsed near zero or near a constant. Their severity outputs are marked `collapsed_not_for_use` in the result summaries and are withheld by the prediction CLI.

Model C is the multi-domain high-specificity operating point, not a universal winner. Model F is a diagnostic result, not a production severity model.

## Data and target semantics

FloPWD contains aerial images with binary plastic-presence labels and mask-derived coverage percentages. UGV-NBWASTE contains ground-level images with oriented waste annotations. The UGV image-level target used here is annotated waste present; it is broader than FloPWD plastic presence and has no clean negative test examples. UGV provides no comparable severity target.

## Evaluation

FloPWD metrics use a frozen 300-image test set and threshold 0.5. UGV metrics use 536 grouped positive test records and report annotated-waste positive recall and score distribution only. No UGV specificity, balanced accuracy, or binary accuracy is claimed. See the complete [results](results.md) and [machine-readable summaries](../experiments/results/).

## Limitations and risks

- A single seed and one fixed split limit estimates of run-to-run and population variation.
- UGV positives have broader semantics and no clean negatives.
- Filename-derived groups reduce a known leakage risk but cannot prove original-image identity.
- The benchmarks ran on CPU and do not establish deployment performance.
- Multi-domain severity outputs are collapsed and must not be used as coverage estimates.
- Geographic, seasonal, sensor, and acquisition shifts remain untested.
- Historical dissertation results are not reproduced results; see the historical/rebuilt distinction in [results](results.md).

## Data and licensing

Dataset files and model checkpoints are not distributed. Dataset use remains subject to source licences and terms documented in [`data/README.md`](../data/README.md). Source code is MIT licensed.

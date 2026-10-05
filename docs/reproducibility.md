# Reproducibility

## Rebuilt portfolio protocol

The main controlled comparison used Python 3.10.11, TensorFlow 2.15.1, Keras 2.15.0, NumPy 1.26.4, seed 42, and official `keras.applications.resnet.preprocess_input`. The ResNet50 ImageNet backbone was frozen. The main multi-domain runs used Adam (0.001), batch size 32, 15 epochs, 44 steps per epoch, 660 optimizer updates, classification/severity weights 1.0/0.5, and classification threshold 0.5.

The committed FloPWD manifest records a deterministic stratified 70/15/15 split. The UGV manifest groups canonical filename-derived source proxies before a 70/15/15 split. SHA-256 values for both manifests and per-run provenance are recorded in [`experiments/results/`](../experiments/results/). Full checkpoints and large prediction tables are excluded from this repository.

All six experiments are completed and the tracked matrix preserves the frozen settings with their final statuses. **Experimental phase closed after Model F.** No future benchmark is planned. The summaries retain the training/evaluation commit SHAs and matrix hash used at the time; the current status-only matrix revision is not claimed as the training-time specification.

## Reproduce the software environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Place the unmodified datasets locally and set the paths in commands as `<PATH_TO_FLOPWD>` and `<PATH_TO_UGV>`. Dataset inspection, split creation, A training, B/C training, evaluation, prediction, and tests are shown in the top-level [README](../README.md). The manifests used for the portfolio are already committed; regenerate them only when deliberately creating a new study.

## Evaluation semantics

FloPWD binary plastic-presence metrics are calculated from the frozen test partition at threshold 0.5. Severity is reported in percentage points, separately for all test images and positive images. UGV is positive-only under the annotated-waste-present target; supported reporting is sample count, UGV annotated-waste positive recall, and probability distribution. It does not support specificity or balanced accuracy.

The cross-domain recall gap compares distinct positive-target definitions and is diagnostic, not matched-label accuracy. No validation/test metrics were used to alter a frozen model, threshold, or sampling policy.

## Historical appendix

`legacy/` retains the dissertation implementation for provenance. It differs in preprocessing, split handling, label assumptions, and metric semantics. Historical 100% claims remain report-derived and are not presented as reproduced portfolio results. The rebuild uses new deterministic splits, official ResNet preprocessing, grouped UGV partitions, explicit partial supervision, and matched optimizer-step budgets.

## Repeatability limits

Seeds and deterministic TensorFlow operations were requested, but bit-for-bit repeatability may depend on TensorFlow build, hardware, and kernels. CPU-only results and a single seed do not estimate performance variance across seeds or deployment populations. Filename grouping is a proxy and cannot detect every duplicate image. Dataset exports can change; the documented UGV v13 mapping and hashes apply only to the inspected local export, which is not included.

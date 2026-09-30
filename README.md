# Floating Plastic Detection & Severity Estimation

A deep-learning computer vision project for detecting floating plastic in water and estimating image-area plastic coverage, developed as my Computer Science with AI dissertation.

This portfolio repository separates the supplied dissertation implementation and historical findings from a cleaned implementation and a new heterogeneous multi-domain research design. The work investigates transfer learning, partial supervision, class imbalance, specificity collapse, out-of-domain evaluation, and severity regression.

Author: **Joel Rola**

## Project

The completed portfolio control is a FloPWD-only ResNet50 experiment. Its metrics and provenance are recorded in [results](docs/results.md); generated run artifacts remain local under ignored `runs/`. The next planned comparison adds grouped UGV training data, but combined training has not been run.

The historical appendix scales RGB pixels by `/255`. The portfolio model instead embeds the official `keras.applications.resnet.preprocess_input` transform in the saved model graph. Severity is constrained to 0-100 percentage points using a sigmoid output scaled by 100. These choices differ from the historical implementation; the cleaned pipeline is not claimed to reproduce dissertation methodology or metrics.

## Dissertation results

The following are historical results reported in the dissertation materials. They have **not yet been reproduced by the refactored repository**:

| Historical metric | Reported value |
|---|---:|
| Baseline validation accuracy | 74.3% |
| Baseline specificity | 0% |
| 1:1 balancing specificity | 44% |
| Final reported specificity | 100% |
| Final reported sensitivity | 100% |
| Severity MAE | 5.3 percentage points |
| Mean classification confidence | 93% |

The 0% baseline specificity is notable because overall accuracy alone hid a majority-class failure mode: the baseline did not correctly identify negative examples. Specificity and sensitivity were not both derived solely from the positive-only UGV set. The completed FloPWD portfolio control uses a separate protocol; it is not an exact historical replication. See [results](docs/results.md) and [reproducibility notes](docs/reproducibility.md).

## Multi-domain design

The scientific question is whether visually diverse ground-level UGV imagery can improve cross-domain annotated-waste detection while preserving aerial FloPWD classification and coverage estimation. Rather than forcing heterogeneous datasets into a single label format, the reproduction uses partial supervision: FloPWD provides classification and coverage regression labels, while UGV contributes cross-domain classification evidence where supported.

The complete local UGV v13 export contains 3,600 images, with all eight class IDs mapped by its local `data.yaml`; its original partitions contain 25 filename-derived source groups crossing splits. A deterministic grouped manifest is provided for future experiments. UGV positives mean **annotated waste present**, not clean-water absence and not necessarily plastic for every category. UGV severity is unavailable. See [`data/README.md`](data/README.md) and [`docs/methodology.md`](docs/methodology.md).

## Data and setup

Datasets are not included. See [`data/README.md`](data/README.md) for source citations, licence information, expected layout, and schema limitations.

Inspect local metadata without training:

```bash
python scripts/inspect_data.py --dataset flopwd --path data/FloPWD
python scripts/inspect_data.py --dataset ugv --path data/UGV_NBWASTE
```

For FloPWD, `--write-split-manifest` saves a deterministic stratified 70/15/15 filename manifest. The seed-42 manifest is a portfolio split, not the dissertation split. To inspect UGV and create the source-grouped manifest and original-split leakage report, use `--write-grouped-split --write-leakage-report` with the complete local export.

The runtime targets Python 3.10 with TensorFlow 2.15.x and NumPy below 2.0. Install `requirements.txt` for training/inference. Lightweight tests use `requirements-dev.txt`; standard CI does not install TensorFlow or download ImageNet weights. The local TensorFlow runtime was validated for the FloPWD pipeline. Combined multi-domain training is not yet run.

Train with local FloPWD data and the committed seed-42 manifest:

```powershell
python scripts/train.py --data-dir "PATH_TO_FLOPWD" --config configs/default.yaml --split-manifest experiments/splits/flopwd_seed42.json --output-dir runs/flopwd_baseline
```

Evaluate the untouched test split and predict one image:

```powershell
python scripts/evaluate.py --data-dir "PATH_TO_FLOPWD" --model runs/flopwd_baseline/model.keras --split-manifest experiments/splits/flopwd_seed42.json --output-dir runs/flopwd_baseline/evaluation
python scripts/predict.py --model runs/flopwd_baseline/model.keras --image path/to/image.jpg
```

`--max-train-samples` is debug-only. A limited run is marked as a subset, not a full benchmark. Historical values above are not outputs of this code.

## Repository map

- `src/floating_plastic/` - model, dataset contracts, grouped splits, masked losses, sampling, pipeline, and metrics.
- `scripts/` - training, evaluation, prediction, and data-inspection commands.
- `configs/default.yaml`, `configs/experiments.yaml` - runtime defaults and unrun experiment profiles.
- `legacy/` - appendix implementation retained for provenance.
- `docs/` - methodology, results, model card, and reproducibility notes.
- `tests/` - lightweight dataset-independent unit tests.

## License

Original source code in this repository is licensed under MIT. Dataset assets remain subject to their original licences and are not redistributed here. See [LICENSE](LICENSE).

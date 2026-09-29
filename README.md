# Floating Plastic Detection & Severity Estimation

A deep-learning computer vision project for detecting floating plastic in water and estimating image-area plastic coverage, developed as my Computer Science with AI dissertation.

This portfolio repository separates the supplied dissertation implementation and its historical findings from a cleaned implementation whose results remain to be validated. The work investigates transfer learning, multi-task learning, class imbalance, specificity collapse, out-of-distribution evaluation, and severity regression.

Author: **Joel Rola**

## Project

The documented model concept uses an ImageNet-pretrained ResNet50 backbone with a binary plastic-presence head and an image-area coverage regression head. Inputs are resized to 224 × 224. The refactored package provides reusable model, data, balancing, and metric interfaces; training and prediction workflows are being made reproducible before results are reported.

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

The 0% baseline specificity is notable because the overall accuracy alone hid a majority-class failure mode: the baseline did not correctly identify negative examples. Specificity and sensitivity were not both derived solely from the positive-only UGV set. The portfolio reproduction work is intended to make the evaluation protocol clearer. See [results](docs/results.md) and [reproducibility notes](docs/reproducibility.md) for context.

## Data and setup

Datasets are not included. See [`data/README.md`](data/README.md) for the expected layout and current schema limitations. Download links will be added from the original public sources after their provenance and licence details are verified.

For model work, create a Python environment and install the packages in `requirements.txt`. TensorFlow installation depends on the target operating system and accelerator; choose a compatible TensorFlow build for your environment. For lightweight local tests, install `requirements-dev.txt` instead. The tests do not instantiate ResNet50 or download pretrained weights.

The clean training/evaluation/prediction command-line workflows are not yet ready to claim a reproduced experiment. Do not interpret historical values above as outputs of this code.

## Repository map

- `src/floating_plastic/` — small reusable implementation modules.
- `scripts/` — command-line entry points.
- `configs/default.yaml` — documented experiment defaults.
- `legacy/` — appendix implementation retained for provenance.
- `docs/` — methodology, results, model card, and reproducibility notes.
- `tests/` — lightweight unit tests.

## License

Original source code in this repository is licensed under MIT. Dataset assets remain subject to their original licences and are not redistributed here. See [LICENSE](LICENSE).

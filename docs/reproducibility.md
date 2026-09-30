# Reproducibility

## Historical appendix differences and limitations

The appendix source remains available at `legacy/appendix_main.py`; it has not been rewritten as part of this phase. Its known issues include:

- **Dataset paths:** path spellings differ between code and README (`FlOPWD`/`FloPWD`) and use a relative local layout.
- **Partial image limits:** hidden/default sample caps select subsets; directory traversal order was not reliably deterministic.
- **Evaluation split:** constructed evaluation mixes FloPWD negative examples and UGV positive examples, confounding source and class. Some validation examples are reused, and the UGV loader does not consistently use its designated held-out split.
- **Randomness:** not every NumPy/TensorFlow source was seeded.
- **UGV target assumptions:** images were treated as positive without validating annotation contents or class IDs; empty annotations were not distinguished from labeled negatives.
- **Regression evaluation:** the appendix trains a severity output but does not establish its reported severity MAE through a complete held-out evaluation.
- **Preprocessing:** appendix RGB inputs are scaled by `/255`; the portfolio model embeds official `keras.applications.resnet.preprocess_input`. This is a methodological difference.
- **Metric naming:** a comparison plot labels accuracy as balanced accuracy; these metrics are not generally interchangeable.

Historical figures remain report-derived and are distinct from portfolio runs.

## FloPWD portfolio control

The local runtime targets Python 3.10, TensorFlow 2.15.x, and NumPy below 2.0. The ResNet50 preprocessing layer is embedded in the saved model, consumes resized RGB values in [0,255], and applies official Keras ResNet preprocessing. Severity is sigmoid-scaled to 0–100 percentage points. The seed-42 FloPWD split is a deterministic stratified 70/15/15 filename manifest; it is not the dissertation split. The completed baseline/control artifacts are in ignored `runs/flopwd_original_seed42` and summarized in [`results.md`](results.md).

The FloPWD loader checks CSV schema, duplicate/missing labels, file correspondence, masks, and severity range. Training shuffles only training examples. Validation and test remain deterministic and test evaluation is performed separately. The eight positive-label/zero-severity rows are retained exactly as supplied.

Exact bit-for-bit repeatability can depend on TensorFlow version, hardware, and backend kernels even when seeds and deterministic operations are requested. Runtime versions, split and model hashes, and experiment configuration should be retained with every run.

## UGV local export and grouped split

The complete inspected local folder is a Roboflow UGV-NBWASTE v13 export. Its README identifies the [v13 Roboflow dataset export](https://universe.roboflow.com/ugv-nbwaste/ugv-nbwaste/dataset/13). It contains 2,160/720/720 images in train/valid/test, 3,600 total, each with a non-empty OBB text file. The count and nominal 60/20/20 ratio match the paper's report, but filename-derived groups cross partitions in this export. The YAML has IDs 0–7 and all eight are observed. There are 4,095 annotation rows; 536 rows have at least one finite coordinate outside [0,1], retained by the parser without clipping. The paper describes 3,600 original images and eight waste categories ([paper](https://doi.org/10.1016/j.dib.2025.111559); [dataset DOI](https://doi.org/10.17632/fv28xxn4f3.3)).

A separate local directory contains model/metadata files but no image splits; its seven-ID mapping is not applied to the v13 export. Earlier project notes described a 6,030-image copy, but that image export is not present among the currently inspected directories, and its relationship to the complete v13 export cannot be established. For the v13 YAML, class IDs map locally to bottle, cocksheet, hardplastic, mask, medicine, packet, polythene, and sandal. This mapping is based on the actual local export metadata rather than guessed from category ordering in prose.

The current export's filenames yield 3,549 canonical source-ID groups. Removing only a terminal case-insensitive `.rf.<hex hash>` suffix identifies 25 groups present in multiple original partitions: 15 train+validation, 5 train+test, 4 validation+test, and 1 across all three. The machine-readable evidence is `experiments/splits/ugv_leakage_report.json`. This is a filename-based proxy; it neither proves that matching IDs are pixel-identical nor detects duplicate imagery with different stems.

The grouped seed-42 manifest at `experiments/splits/ugv_grouped_seed42.json` assigns each group to one 70/15/15 partition: train 2,484 groups/2,522 images; validation 533/542; test 532/536. It is not the paper's split. Source images are not renamed or modified.

## Multi-domain protocol (planned; no combined training yet)

FloPWD supplies classification and mask-derived coverage regression. UGV records supply classification evidence only when a non-empty annotation verifies an annotated waste object. With an optional included-class filter, only images with at least one included class are eligible. Empty/nonmatching UGV annotations are unavailable labels, not clean negatives. The UGV target is “annotated waste present”; category names do not justify relabeling every example as plastic. UGV severity remains unavailable because OBB geometry is not the same target as image-area plastic coverage.

The training representation stores per-task values and availability flags. Masked binary cross-entropy and masked severity MAE contribute only for records with the relevant target. Domain sampling balances source domains independently of class balancing; class balancing, if enabled, is restricted to FloPWD training samples. Validation and test sets remain separate by domain. Never report a combined-domain metric without accompanying per-domain metrics.

UGV's eligible test set is positive-only under the current broad annotated-waste target, so report positive recall and score distribution only. Specificity, balanced accuracy, and binary accuracy require negative ground truth and must not be calculated from UGV alone. The difference between FloPWD positive recall and UGV positive recall is a descriptive domain gap with distinct target semantics, not necessarily an accuracy degradation.

## Dataset-free validation

Unit tests use synthetic CSV/YAML/annotation fixtures and do not access source dataset images. CI does not install TensorFlow, download weights, or require a GPU. Full multi-domain training remains pending; configurations are experiment definitions and not performance claims.

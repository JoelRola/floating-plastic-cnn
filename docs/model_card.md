# Model card

## Intended use

Research and educational exploration of plastic/waste image classification and FloPWD image-area coverage estimation. The work is not validated for operational environmental monitoring or policy decisions.

## Model and tasks

The model uses ImageNet-pretrained ResNet50 (`include_top=False`), global average pooling, shared dense/dropout layers, a sigmoid classification head, and a severity head bounded to 0–100 percentage points. Input is resized RGB with official Keras ResNet preprocessing embedded in the graph. The historical appendix instead divides RGB values by 255.

Portfolio design uses partial supervision rather than forcing heterogeneous labels into one format: FloPWD supplies binary plastic-presence and coverage-regression targets; UGV can supply a positive annotated-waste-presence target where OBB labels support it. UGV does not supply coverage labels. A masked task loss excludes unavailable targets. This combined design has not yet been trained or evaluated.

## Data and label semantics

FloPWD contains aerial scenes and image-level plastic labels plus image-area coverage percentages. Its eight positive-label/zero-coverage observations are retained as provided.

The inspected complete UGV v13 export has 3,600 ground-level images and a local `data.yaml` mapping for IDs 0–7. Every image has a non-empty annotation in this copy. The image-level UGV target is “annotated waste present”; it is not a clean-water-negative label and is not generalized to plastic presence for every semantic category. Empty and nonmatching annotations are unavailable, not negative. A filename-derived group split prevents detected same-stem variants from spanning partitions.

## Evaluation and limitations

The FloPWD-only portfolio control has been completed and is recorded in [`results.md`](results.md). The combined model remains unverified. Report FloPWD metrics separately from UGV positive recall/probability distribution. UGV positive-only data cannot support specificity, balanced accuracy, or binary accuracy. A cross-domain positive-recall difference is descriptive and reflects different domains and positive-target semantics.

Domain shift includes aerial versus ground viewpoint, geography, camera, background, image scale, and acquisition method. Filename grouping cannot identify differently named duplicates. The current OBB coordinates also include values outside [0,1]; no auxiliary box task is trained in this phase. No checkpoint is distributed.

Dataset licences remain with their sources. Runtime and protocol details are in [`reproducibility.md`](reproducibility.md).

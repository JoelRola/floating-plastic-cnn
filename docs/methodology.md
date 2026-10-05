# Methodology

## Research question

Can ground-level waste imagery improve cross-domain positive detection while preserving aerial FloPWD plastic classification and mask-derived coverage estimation? The experiment matrix tests data-domain composition, classification class prior, task-head sharing, and severity-example sampling as distinct factors.

## Data and supervision

**FloPWD** provides aerial imagery with image-level plastic-present/absent labels and continuous mask-derived coverage in 0–100 percentage points. The eight positive-label/zero-coverage observations were retained. The deterministic seed-42 manifest partitions 2,002 filename records into 1,402 train, 300 validation, and 300 test examples.

**UGV-NBWASTE v13** provides ground-level imagery and oriented non-biodegradable-waste annotations. An eligible non-empty annotation supports the target **annotated waste present**. The grouped seed-42 test set contains 536 positives. UGV contributes no comparable coverage label; no severity values are fabricated from boxes. Its positive-only labels do not support specificity or balanced accuracy.

The domains differ in view, acquisition, and target semantics. UGV annotated-waste positive recall is reported separately and is not treated as matched-label FloPWD accuracy.

## Model and training protocol

The feature extractor is ImageNet-initialized ResNet50 (`include_top=False`) followed by global average pooling. It is frozen. Classification predicts a sigmoid probability; severity is parameterized as sigmoid × 100 percentage points. Inputs use official Keras ResNet preprocessing embedded in the model.

Models A/B/C/E use the shared trainable multi-task tower. D/F share only the frozen feature vector before separate trainable classification and severity towers. All full controlled runs use seed 42, Adam at 0.001, batch size 32, 15 epochs, 44 steps per epoch, and 660 optimizer updates. Classification and severity loss weights are 1.0 and 0.5. The classification threshold is fixed at 0.5.

| Model | Training design | Changed factor |
|---|---|---|
| A | FloPWD only; original FloPWD prior | Control |
| B | FloPWD + UGV; 0.5/0.5 domains; original FloPWD prior | Add domain |
| C | Same as B; 1:1 FloPWD negative:positive classification samples | Classification prior |
| D | Same sampling as C; task-decoupled trainable towers | Architecture |
| E | Same shared-tower protocol as C; 2:1 FloPWD negative:positive samples | Classification prior |
| F | Same architecture/classification stream as D; separate original-prior FloPWD severity stream | Severity sampling distribution |

F takes K independent original-prior FloPWD severity records per update, where K is the number of FloPWD records in the mixed classification batch. This matches D's severity-example count and optimizer updates but adds a separate severity-only forward pass, so total image-forward/FLOP cost is higher.

## Experimental findings

The class-prior ablation moves the fixed-threshold operating point: B has high sensitivity and lower specificity; C recovers specificity and balanced accuracy; E reduces sensitivity and does not exceed C's specificity. UGV annotated-waste positive recall remains high across multi-domain models.

Severity is functional in Control A. B/C/E show near-zero or near-constant collapse. D isolates the trainable task towers but remains collapsed, indicating shared trainable head layers alone were not sufficient to explain the failure. Classification balancing changes the severity stream's exact-zero fraction from 26.32% in original FloPWD training records to about 50.24% in D's balanced sampling cycle. F restores the original severity sampling distribution but also remains collapsed. The sigmoid×100 output and optimization dynamics are plausible unresolved mechanisms; they are not proven causes.

The design rationale and controlled conditions are described in the tracked [experiment matrix](../experiments/specs/multidomain_matrix_seed42.yaml). **Experimental phase closed after Model F.** All six benchmarks are complete.

## Evaluation and leakage controls

FloPWD reports accuracy, balanced accuracy, sensitivity, specificity, precision, F1, confusion counts, and severity metrics. UGV reports only UGV annotated-waste positive recall and probability summaries. The difference between those recall values is a domain diagnostic, not a matched-domain accuracy metric.

UGV grouping removes a terminal Roboflow-style `.rf.<hex>` suffix from filename stems and assigns each canonical group to one split. It detected 25 cross-partition groups in the original export. Grouping is a leakage-control proxy, not proof of original-image identity or a detector for differently named duplicates.

Validation metrics were descriptive and did not change thresholds, sampling, or frozen configurations. Each final benchmark used the final-epoch checkpoint and the frozen 0.5 threshold. See [results](results.md), [reproducibility](reproducibility.md), and the compact [result package](../experiments/results/).

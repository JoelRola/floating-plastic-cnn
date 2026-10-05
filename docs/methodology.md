# Methodology

## Historical dissertation implementation

The original appendix implementation is retained under `legacy/`. It combines transfer learning with image-level binary plastic classification and image-area coverage regression. The historical preprocessing, split construction, class balancing, and evaluation limitations are described separately in [`reproducibility.md`](reproducibility.md); those historical results are not attributed to the current portfolio code.

## Portfolio research question

Does adding visually diverse ground-level UGV imagery improve cross-domain plastic/waste detection while preserving aerial FloPWD classification and severity-estimation performance? The completed FloPWD-only run is the control. The frozen A/B/C/E multi-domain runs vary source composition and FloPWD class prior; D isolates trainable task representations; F isolates severity sampling.

Rather than forcing heterogeneous datasets into a single label format, the reproduction uses partial supervision: FloPWD provides classification and coverage regression labels, while UGV contributes cross-domain classification evidence where supported.

## Domains and supervision

FloPWD supplies aerial image samples with an image-level plastic-presence target and a continuous image-area coverage target in 0–100 percentage points. Both targets are available for the verified FloPWD record; the eight positive-label/zero-coverage cases remain unchanged.

The complete local UGV Roboflow v13 export supplies ground-level images and oriented waste-object annotations. The local `data.yaml` mapping is the parser source of truth for that export: ID 0 `bottle`, 1 `cocksheet`, 2 `hardplastic`, 3 `mask`, 4 `medicine`, 5 `packet`, 6 `polythene`, 7 `sandal`. The paper describes the same broad eight waste categories, though its prose/order and naming are not a substitute for the export's ID mapping. A separate local metadata-only folder exposes IDs 0–6 with a different mapping and no image splits; it is not used.

For the complete export, a non-empty annotation supports an image-level **annotated waste present** positive. This target is not called plastic presence for every class. Empty or class-filtered nonmatching records are label-unavailable, never clean-water negatives. The local v13 export has 3,600/3,600 non-empty annotations, so all its images are eligible positive examples under the broad waste-presence target. UGV severity is unavailable: OBB occupancy is not interchangeable with mask-derived image-area plastic coverage.

## Leakage-aware UGV partitions

Roboflow-style `.rf.<hex hash>` suffixes are removed from filename stems to construct a deterministic, case-insensitive canonical source-ID proxy. This naming rule found 3,549 groups among 3,600 exported images; 25 groups crossed the export's existing train/valid/test boundaries. This is evidence of split overlap under the filename-derived proxy, not proof that all distinct IDs are independent source images.

`experiments/splits/ugv_grouped_seed42.json` provides a deterministic 70/15/15 split by group. It assigns all variants from each canonical group to one partition. Counts are 2,484/533/532 groups and 2,522/542/536 images for train/validation/test. This portfolio split does not reproduce the publication's split.

## Heterogeneous model training design

Models B/C/E use the existing ResNet50 shared tower and retain binary classification and 0–100 severity outputs. Model D shares only the frozen ResNet50/GAP representation; separate trainable classification and severity towers follow it. Dataset records carry separate target values, availability flags, and source-domain IDs. Keras loss inputs encode `[target, available]`; masked classification loss ignores samples without a supported image-level target, and masked regression loss ignores samples without genuine severity labels. UGV examples therefore contribute classification supervision only, without fabricated severity targets.

Domain sampling and class balancing are separate controls. `domain_sampling.strategy: balanced` gives each active source equal example-sampling probability regardless of export size. Optional binary class balancing applies only to labeled FloPWD training records, because positive-only UGV provides no clean-water negatives. Validation and test remain separated by source domain and are reported independently.

The frozen seed-42 experiment matrix in `experiments/specs/multidomain_matrix_seed42.yaml` defines:

- **A:** FloPWD-only control with the original FloPWD class distribution.
- **B:** FloPWD plus grouped UGV, balanced domain sampling, original FloPWD class distribution.
- **C:** same multi-domain protocol with 1:1 negative:positive sampling within FloPWD training records.
- **E:** same shared-tower protocol as C with 2:1 negative:positive sampling within FloPWD training records.
- **D:** same data protocol as C, but independent trainable classification and severity towers after the frozen shared feature extractor.
- **F:** same task-decoupled architecture and balanced multi-domain classification stream as D, but draws severity examples independently from the original, unbalanced FloPWD training distribution. Each update draws exactly as many severity examples as FloPWD examples in its classification batch.

B/C/E classification results show that changing the effective class prior changes the fixed-threshold sensitivity/specificity tradeoff. Severity remained near-zero collapsed for B, C, and E, with all-image MAE about 5.47–5.48 percentage points. Full Model D also remained near-zero collapsed after task-head decoupling, so shared trainable task layers alone were not sufficient to explain the failure. The audit found that 1:1 classification balancing shifted the severity draw's exact-zero fraction from 26.32% to about 50.24%, where zero is the MAE-optimal constant; sigmoid×100 outputs also rapidly saturated toward zero. Model F therefore separates classification and severity sampling while holding architecture, loss, optimizer, and optimizer-update budget fixed. F's engineering smoke is not benchmark evidence, and no F benchmark claim is made here.

Model F adds a severity-only forward pass over an independent original-prior FloPWD stream for every mixed classification batch. This matches D's optimizer updates, classification sampling policy, and severity-example count when its deterministic classification stream is reproduced. Its total image-forward/FLOP cost is higher than D and is not described as compute-matched.

Comparisons retain the same split manifests, seed, preprocessing, losses and weights, optimizer-step budget, and evaluation implementation unless the matrix explicitly names the architecture or FloPWD class-ratio ablation.

## Evaluation design

FloPWD validation/test metrics include accuracy, balanced accuracy, sensitivity, specificity, precision, F1, confusion counts, and image-area severity MAE/RMSE in percentage points. Severity is also summarized for positive images separately.

UGV evaluation uses only supported metrics: annotated-waste positive recall at a predeclared threshold, sample count, and predicted-probability distribution. Without labeled clean-water negatives, UGV specificity, balanced accuracy, and binary accuracy are undefined and omitted. A cross-domain positive-recall difference may be reported as a **domain gap** between different domains and positive-target semantics; it is not called accuracy degradation.

Do not combine FloPWD negatives and UGV positives into one benchmark. Geography, viewpoint, camera, backgrounds, image scale, acquisition, and export history differ, so a mixed source/class split would confound class and domain.

## Potential later UGV auxiliary task

The annotations could support object detection or object-count prediction after validating boxes and defining a separate task. Detection would add stronger computer-vision evidence than an occupancy proxy while preserving the distinction from FloPWD mask-derived image coverage. No auxiliary target is implemented in this phase.

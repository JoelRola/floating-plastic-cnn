# Reproducibility notes

The appendix implementation is preserved in `legacy/appendix_main.py`. Its results are historical claims; the cleaned package does not claim to reproduce them.

Known issues and uncertainties in the appendix version:

- **Dataset paths:** the script expects `./data/FlOPWD`, while its README says `./data/FloPWD`; the supplied FloPWD archive is nested under a longer directory name. Case differences also affect case-sensitive systems.
- **Partial image limits:** FloPWD is capped at 500 rows and UGV at 100 directory entries by defaults in the script. Directory order is not sorted, so selected samples may vary.
- **Evaluation/test split:** validation negatives are reused in the constructed test set. The test positives come from UGV while negatives come from FloPWD, confounding source and class. The UGV loader also takes training images rather than its named test split.
- **Random seeds:** the split receives a fixed seed, but NumPy oversampling and TensorFlow initialization are not seeded comprehensively.
- **UGV labels:** every loaded UGV image is assigned a positive label without reading its annotations or verifying class IDs.
- **Regression evaluation:** the script trains a coverage output but reports no regression metric. The poster's severity MAE therefore cannot be verified from the supplied script alone.
- **Preprocessing:** appendix images are RGB and scaled to [0, 1]. Keras ResNet50 expects its application-specific preprocessing; the exact historical training environment and preprocessing are not established here.
- **Balancing comparison:** the appendix trains baseline and balanced variants for different epoch counts (10 and 15), so the comparison changes more than class ratio.
- **Metrics:** the comparison plot labels accuracy as balanced accuracy. They happen to agree on an exactly balanced evaluation set, but are different metrics in general.

## Portfolio data audit

The locally available FloPWD copy contains 2,002 image files, 2,002 binary CSV rows, 2,002 severity CSV rows, and 2,002 masks. Filenames match across all four sources and are unique. Binary counts are 1,482 yes and 520 no. The severity column is in percentage points from 0.00 to 81.91 and the source describes it as mask-derived. Eight yes-labelled images have a zero severity value; neither target is silently changed. The deterministic loader validates exact required CSV columns, filenames, duplicate rows, image/mask correspondence, and severity bounds.

The local UGV export contains train/validation/test image counts 4,824/603/603 (6,030 total), the same count of annotation files, 7,563 OBB rows, and 13/2/3 empty annotations. OBB text uses a class ID followed by eight corner coordinates. IDs 0–6 are observed. Among the OBB rows, 125 have a coordinate outside [0, 1]; all values are finite and are retained without clipping. The local folder has no `data.yaml` or dataset README; names are therefore unmapped. The loader fails clearly when class metadata is required. The paper names eight classes; the public Roboflow project currently lists nine, further reason not to infer this export's mapping.

The official paper describes 3,600 images split 60/20/20. The local 6,030-image 80/10/10 export has Roboflow-style `.rf.<hash>` filenames. Removing this suffix yields 980 apparent source IDs that occur across multiple splits. The public Roboflow project history lists an earlier 6,030-image version as well as later 3,600-image versions. This makes an earlier Roboflow export a plausible explanation for the local counts, but the missing local YAML/README prevents confirmation of the exact version, preprocessing, or augmentation history. Cross-split variants create potential leakage; do not treat the local UGV split as an independent benchmark.

## Portfolio split and evaluation protocol

Create a deterministic 70/15/15 FloPWD train/validation/test split stratified by the binary label and controlled by a recorded seed. Save a filename-only manifest under `experiments/splits/`; no images or targets are written to it. The committed/local seed-42 manifest is a portfolio split, not the dissertation split. Because scene/group identifiers are unavailable, this random split may still place related neighboring images across partitions; assess grouping/near-duplicates before interpreting test metrics.

Fit and balance using training records only. Use validation for model selection and thresholds, then evaluate once on the untouched FloPWD test manifest. Report classification metrics with sample counts. Evaluate severity MAE in percentage points on the corresponding held-out records and state the sample count. Report the eight target inconsistencies. Evaluate UGV separately and only after recovering its local class-name metadata and defining which categories satisfy the image-level target. UGV-only data cannot yield plastic-versus-clean-water specificity, and empty annotation files are not presumed to be clean-water examples. Geography, perspective, background, and acquisition differ between FloPWD and UGV, so any valid UGV result is an OOD analysis with those domain differences reported.

Reproduced model results remain pending. The inspection CLI reads metadata and annotation text only; it does not decode or modify image data.

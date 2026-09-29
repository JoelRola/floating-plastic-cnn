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

The clean package makes UGV schema uncertainty explicit and avoids constructing a mixed-source test set. A reproducible protocol and results remain pending dataset/schema verification and controlled execution.

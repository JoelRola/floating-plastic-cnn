# Dataset sources and local layout

Datasets are not redistributed by this repository. Download the data from the original source and review its licence and version terms before use. The expected local layout is:

```text
data/
├── FloPWD/
│   ├── Raw_Images/
│   ├── Segmentation_Masks/
│   ├── Image_labels_Binary Classification Task.csv
│   └── Mask_foreground_percentages_Regression Task.csv
└── UGV_NWBWASTE/
    ├── data.yaml
    ├── train/{images,labels}/
    ├── valid/{images,labels}/
    └── test/{images,labels}/
```

## FloPWD

**Official title:** Dal Lake Floating Plastic Waste Detection Dataset (FloPWD 2025), Mendeley Data version 2, DOI [10.17632/znxjncgjkc.2](https://doi.org/10.17632/znxjncgjkc.2). The source describes approximately 2,000 1280 × 720 aerial images, binary presence labels, segmentation masks, and a per-image foreground-coverage CSV. The source lists a **CC BY 4.0** licence.

The local copy inspected for this project has the exact directory and filenames shown above. Each of 2,002 images has one row in both CSVs and one correspondingly named mask. The image-area severity percentages range from 0 to 81.91. Eight records are labelled plastic-present while their coverage CSV value is zero; inspection reports this discrepancy without changing either label. Severity is described by the dataset source as derived from the corresponding segmentation masks.

## UGV-NBWASTE

**Official title:** UGV-NBWASTE: An Oriented Non-Biodegradable Waste Dataset in Bangladesh, Mendeley Data version 3, DOI [10.17632/fv28xxn4f3.3](https://doi.org/10.17632/fv28xxn4f3.3). The dataset page lists **CC BY 4.0**. The associated Data in Brief paper is [10.1016/j.dib.2025.111559](https://doi.org/10.1016/j.dib.2025.111559): Md Riadul Islam et al., "UGV-NBWASTE: An oriented dataset for non-biodegradable waste in Bangladesh," *Data in Brief* 60 (2025), 111559.

The [UGV-NBWASTE Roboflow Universe project](https://universe.roboflow.com/ugv-nbwaste/ugv-nbwaste) currently lists nine class labels, while the paper describes eight. Its [version history](https://universe.roboflow.com/ugv-nbwaste/ugv-nbwaste/dataset/13) includes a 6,030-image earlier version and later 3,600-image versions. This is consistent with a changing export/schema history, but does not prove the provenance of the local 6,030-image copy.

The paper describes eight categories: Plastic Bottle, Hard Plastic, Mask, Medicine Packet, Packet, Polythene, Cocksheet (Styrofoam), and Plastic Sandal. A YOLO-OBB export needs its own `data.yaml` to map numeric IDs to names. The local copy inspected here has no `data.yaml` or dataset README, and only raw class IDs 0-6 appear in annotation text. The semantic mapping and the status of the unobserved eighth category cannot be recovered safely from annotation IDs alone. The loader therefore requires local class-name metadata and does not infer binary plastic labels from IDs. The local OBB files have class ID plus eight corner coordinates; 125 non-empty box rows have coordinates outside the normalized 0-1 range, with observed extrema -0.1436 and 1.1398. The parser retains these finite source values without clipping.

The local directory contains `train`, `valid`, and `test` splits with `images/` and `labels/` subdirectories. The export contains 4,824 / 603 / 603 images respectively (6,030 total), 7,563 annotated OBB rows, and 18 empty annotation files. Filenames use Roboflow-style `.rf.<hash>` suffixes. Removing that suffix reveals 980 source-like filename IDs represented in more than one split. These findings indicate repeated/derived exports and split overlap by apparent source ID; they do not establish the exact augmentation or export history. The original dataset paper describes 3,600 images, 4,095 annotations, and a 60/20/20 split, while the local copy is 80/10/10. The local split is therefore not assumed to be the paper's published split.

UGV-NBWASTE contains waste-object annotations, not a curated clean-water negative class. The current multi-head model produces image-level classification and regression outputs, not object detections. Its image-level outputs could be assessed out of domain only after a documented mapping from the local object categories to the task's definition of “plastic present.” UGV alone cannot estimate specificity for plastic-versus-clean-water classification. Empty annotation files are not automatically treated as clean-water negatives.

The local metadata inspection used filenames, CSV/YAML metadata, and annotation text only; image pixels were not scanned. Links identify the authoritative dataset and paper records. Dataset downloads will remain external to Git.

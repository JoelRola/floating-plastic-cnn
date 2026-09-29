# Dataset layout

Do not commit datasets, derived image exports, or model weights to this repository. Dataset download links will be added from the original public sources after source provenance and licensing are verified.

Expected local layout:

```text
data/
├── FloPWD/
│   ├── Raw_Images/
│   ├── Segmentation_Masks/
│   ├── Image_labels_Binary Classification Task.csv
│   └── Mask_foreground_percentages_Regression Task.csv
└── UGV_NWBWASTE/
    ├── train/{images,labels}/
    ├── valid/{images,labels}/
    └── test/{images,labels}/
```

FloPWD CSV labels are parsed by the package when both files contain matching image names. UGV annotations use numeric class IDs; the mapping from those IDs to plastic presence must be confirmed from authoritative dataset documentation before loading or evaluating UGV labels. The current adapter fails explicitly until that mapping is supplied and implemented.

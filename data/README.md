# Dataset sources and local layout

Datasets are not redistributed here. FloPWD and UGV source data, local exports, and annotations remain subject to their upstream licences and terms; this repository does not grant additional redistribution rights. Download links point to the original public sources. Keep downloaded data outside Git and do not commit source images, masks, or labels.

```text
data/
├── FloPWD/
│   ├── Raw_Images/
│   ├── Segmentation_Masks/
│   ├── Image_labels_Binary Classification Task.csv
│   └── Mask_foreground_percentages_Regression Task.csv
└── UGV_NBWASTE/
    ├── data.yaml
    ├── train/{images,labels}/
    ├── valid/{images,labels}/
    └── test/{images,labels}/
```

## FloPWD

**Official title:** Dal Lake Floating Plastic Waste Detection Dataset (FloPWD 2025), Mendeley Data version 2, DOI [10.17632/znxjncgjkc.2](https://doi.org/10.17632/znxjncgjkc.2). The dataset describes aerial imagery, binary plastic-presence labels, segmentation masks, and per-image foreground coverage. The listed licence is CC BY 4.0.

The inspected local version has 2,002 images, corresponding rows in both CSV files and 2,002 masks. Coverage is in percentage points from 0 to 81.91 and is described as mask-derived. Eight positive binary labels have zero coverage; the loader and training pipeline preserve these source values.

## UGV-NBWASTE

**Official title:** UGV-NBWASTE: An Oriented Non-Biodegradable Waste Dataset in Bangladesh, Mendeley Data version 3, DOI [10.17632/fv28xxn4f3.3](https://doi.org/10.17632/fv28xxn4f3.3), listed as CC BY 4.0. Associated paper: Md Riadul Islam et al., “UGV-NBWASTE: An oriented dataset for non-biodegradable waste in Bangladesh,” *Data in Brief* 60 (2025), 111559, [DOI 10.1016/j.dib.2025.111559](https://doi.org/10.1016/j.dib.2025.111559). The [Roboflow v13 export](https://universe.roboflow.com/ugv-nbwaste/ugv-nbwaste/dataset/13) identifies the version inspected here.

The complete local v13 export has 2,160/720/720 images across train/valid/test, 3,600 total, with one non-empty annotation file per image. The size and nominal 60/20/20 split match the publication's reported dataset totals, although this export's filename-derived groups still cross partitions. It has 4,095 OBB rows. Its local `data.yaml` is authoritative for parsing this export and maps IDs as follows: 0 bottle, 1 cocksheet, 2 hardplastic, 3 mask, 4 medicine, 5 packet, 6 polythene, 7 sandal. This order and exact spelling are local export metadata; do not transfer it to other UGV versions. A second local folder has a seven-ID YAML and model weights but no image split directories, so it is not used by this pipeline. Earlier project notes described a 6,030-image copy; that image export is not present in the currently inspected directories, so its relationship to the complete v13 folder is unresolved.

The original export partitions have 25 filename-derived source groups crossing splits. See [`../experiments/splits/ugv_leakage_report.json`](../experiments/splits/ugv_leakage_report.json). The portfolio grouped manifest `../experiments/splits/ugv_grouped_seed42.json` assigns all variants of a canonical filename-derived group to one partition. Canonicalization removes only a terminal `.rf.<hex hash>` suffix. It is a reproducible grouping proxy, not proof that different IDs correspond to different source photos. The resulting grouped split has train 2,484 groups/2,522 images, validation 533/542, and test 532/536; it does not claim to reproduce the publication split.

UGV contains waste-object annotations, not curated clean-water negatives. A non-empty label file supports an image-level **annotated waste present** target; it is not automatically a “plastic present” label for every class. Empty or nonmatching records are label-unavailable, never clean-water negatives. UGV does not provide a comparable mask-derived coverage target; no severity is inferred from OBBs.

Dataset inspection reads paths, CSV/YAML metadata, and annotation text; it does not modify source files or scan image pixels. Downloaded datasets and model weights remain outside Git.

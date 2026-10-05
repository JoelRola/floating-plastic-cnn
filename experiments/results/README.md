# Completed result package

This directory is the compact tracked snapshot of the completed A–F seed-42 experiment matrix. Values are transcribed from the persisted full-run metadata and final test-evaluation JSONs. Debug/smoke results are excluded.

Each `model_*.json` contains the experiment identity and design, final FloPWD classification/severity results, only supported UGV metrics, capability status, known training/evaluation Git SHAs, and source-artifact hashes. A null provenance field means that the source run did not record that value. A's UGV score comes from the persisted Control-A comparison inside the Model B evaluation workflow: `runs/multidomain_seed42/evaluation/metrics.json` and `ugv_control_a_predictions.csv`. The Model A summary records those paths and their hashes; it does not imply a separate standalone A UGV run.

`comparison.csv` is the concise A–F portfolio table. `provenance.json` collects training/evaluation SHAs, training-time matrix hashes, split-manifest hashes, and hashes for source artifacts. The matrix SHA at training is preserved even though the tracked specification now records final completion statuses.

The package intentionally excludes `.keras` files, large prediction CSVs, raw run directories, caches, datasets, and debug artifacts. The full reproducibility record remains in local ignored run directories and is not required to interpret the tracked metrics.

## Capability status

- Model A: `severity_status: functional`.
- Models B, C, D, E, and F: `severity_status: collapsed_not_for_use`.

Only Model A coverage estimates are exposed by the prediction CLI. Unknown models are treated as unverified and their severity values are withheld.

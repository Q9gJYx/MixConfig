# Data Artifacts

This directory is reserved for **pre-extracted configurations and energy statistics** that let MixConfig run end-to-end without the upstream configuration extractor. See [docs/configurations.md](../docs/configurations.md) for the file-format contract and [docs/dependencies.md](../docs/dependencies.md) for why the extractor code is not in this repository.

## Contents

| File | Dataset | Shape (`configs` / `energy_stats`) | Source | Status |
|---|---|---|---|---|
| `mnist_configs.npz` | MNIST-70k with HOG features | `[70000, 9]` int64 / `[9, 4]` float32 | Authors' Python port of Parallel-DT / BlueRed (`Q9gJYx/bluered`) + `scripts/extract_mnist_artifacts.py` | shipped with v1.0.0 |

The MNIST file additionally carries `labels` (MNIST class targets) and the BlueRed-specific metadata `bluered_lambda` / `bluered_mu` / `bluered_gamma_rng` for inspection; the MixConfig selector consumes only `configs` and `energy_stats`. Additional benchmark artifacts (OpenML-CC18, BBBP, SST-2) are tracked in [ROADMAP.md](../ROADMAP.md).

## How the artifact was produced

`scripts/extract_mnist_artifacts.py` loads the BlueRed configurations from the upstream `bluered` cache, computes HOG features (9 orientations, 7×7 pixels per cell, 2×2 cells per block; matches the upstream demo notebook), and runs the same energy-statistics formulation as `src/mixconfig/energy.py::precompute_energy_statistics`. The H/h_a/h_r/delta_gamma values are therefore in the same feature space as the upstream clustering, not a feature-space proxy.

For configurations with more than 1 500 clusters (e.g., the trivial singleton-endpoint of the BlueRed front), inter-centroid statistics are estimated by a uniform-random Monte Carlo subsample to keep memory bounded; the relative bias on the mean is well below 1%.

## Substituting your own extractor

Drop in `configs.npy` (or a bundled `.npz` with keys `configs` and `energy_stats`) following the shapes documented in `docs/configurations.md`. A validator is exposed as `from src.mixconfig import validate_configurations`.

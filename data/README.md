# Data Artifacts

This directory is reserved for **pre-extracted configurations and energy statistics** that let MixConfig run end-to-end without the upstream configuration extractor. See [docs/configurations.md](../docs/configurations.md) for the file-format contract and [docs/dependencies.md](../docs/dependencies.md) for why the extractor code is not in this repository.

## Contents

| File | Dataset | Shape (`configs` / `energy_stats`) | Source | Status |
|---|---|---|---|---|
| `mnist5k_configs.npz` | MNIST-5000 subset with HOG features | `[5000, m]` int64 / `[m, 4]` float32 | Authors' Python port of Parallel-DT / BlueRed (`Q9gJYx/bluered`, sklearn-style demo) | shipped with v1.0.0 |
| `mnist_70k_configs.npz` | MNIST (70 000 samples) | `[70000, m]` int64 / `[m, 4]` float32 | Same | planned for v1.1 |

Additional benchmark artifacts (OpenML-CC18, BBBP, SST-2) are tracked in [ROADMAP.md](../ROADMAP.md).

## Substituting your own extractor

Drop in `configs.npy` (or a bundled `.npz` with keys `configs` and `energy_stats`) following the shapes documented in `docs/configurations.md`. A validator is exposed as `from src.mixconfig import validate_configurations`.

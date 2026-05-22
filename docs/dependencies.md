# Dependencies and Attribution

This document records the status of code MixConfig depends on but does not author, with full provenance.

## Parallel-DT and BlueRed configuration extractor

The configuration extractor used by MixConfig is the **Parallel-DT** algorithm with the **BlueRed front** convex-hull characterization. The algorithms were developed at Duke University by Tiancheng Liu, Dimitris Floros, Nikos Pitsianis, and Xiaobai Sun, and published in the IEEE High Performance Extreme Computing Conference:

- Liu, T., Floros, D., Pitsianis, N., and Sun, X. **"Digraph Clustering by the BlueRed Method."** *2021 IEEE High Performance Extreme Computing Conference (HPEC)*, 1 to 7. DOI: [10.1109/HPEC49654.2021.9622834](https://doi.org/10.1109/HPEC49654.2021.9622834).
- Pitsianis, N., Floros, D., Liu, T., and Sun, X. **"Parallel Clustering with Resolution Variation."** *2023 IEEE High Performance Extreme Computing Conference (HPEC)*, 1 to 8. DOI: [10.1109/HPEC58863.2023.10363552](https://doi.org/10.1109/HPEC58863.2023.10363552).

An updated reference implementation by the same group is currently under journal review and is not yet publicly available.

## MixConfig authors' Python port

The MixConfig authors separately maintain a Python port of Parallel-DT and BlueRed (internal codename: `bluered`, currently version 0.1.0). It is MATLAB-parity verified against the published reference, with some advanced branches (BR-III, the asa / adjust_brf chain, and several rarer transform families) stubbed.

The port is held in a private development repository pending:

1. Feature parity with the upstream journal version, so that we do not release an imperfect re-implementation that could pre-empt or confuse the upstream authors' submission.
2. Coordination with Pitsianis et al. on public release timing, in keeping with academic conventions for code that closely mirrors another group's unpublished work.

Researchers interested in early access for academic collaboration are welcome to contact the corresponding author at `shixin.xu@dukekunshan.edu.cn`.

## What we release to support reproducibility

Even without the extractor code, we want readers and reviewers to be able to exercise the MixConfig pipeline end-to-end. To that end:

- **Pre-extracted configurations and energy statistics** for the MNIST-70k reference demo are released at `data/mnist_configs.npz` (~408 KB). Configurations come from the BlueRed front; energy statistics (H, h_a, h_r, delta_gamma) are computed on HOG features by `scripts/extract_mnist_artifacts.py`, matching the schema in `src/mixconfig/energy.py`. These are *outputs* of running our Python port on public data, with no IP overlap with the upstream algorithm code.
- **An interface contract** (`docs/configurations.md`) documents the exact `.npy` / `.npz` array shapes and semantics, so that any user with their own multi-resolution clustering routine can substitute it for our extractor.
- **Pre-extracted artifacts for additional benchmarks** (OpenML-CC18, BBBP, SST-2) are tracked in [ROADMAP.md](../ROADMAP.md) and will land in subsequent releases.

## Summary table

| Component | Authored by | Status | License |
|---|---|---|---|
| Parallel-DT algorithm description | Liu / Floros / Pitsianis / Sun (HPEC 2021, 2023) | Published in HPEC proceedings | IEEE proceedings |
| Up-to-date reference implementation | Pitsianis et al. | Under journal review, not public | TBD by upstream |
| Python port (`bluered`) | MixConfig authors | Private development repo, v0.1.0 | TBD |
| Pre-extracted configurations + energy statistics (MNIST-70k) | MixConfig authors | Released here, see `data/mnist_configs.npz` (v1.0.0) | MIT (this repo) |
| MixConfig selector + predictors | MixConfig authors | Released here, see `src/` | MIT (this repo) |

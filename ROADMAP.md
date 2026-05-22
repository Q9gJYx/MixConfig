# Roadmap

This file tracks planned releases beyond the ICML 2026 camera-ready snapshot (`v1.0.0-icml2026`).

## v1.1 (target: post-conference, July 2026)

- **Pre-extracted MNIST artifacts.** Ship `data/mnist_70k_configs.npz` containing configurations and energy statistics from our Python port of Parallel-DT / BlueRed, enabling end-to-end MixConfig reproduction without the extractor.
- **MNIST demo notebook.** `notebooks/demo_mnist.ipynb` that loads the .npz artifacts and runs the Energy-Aware Selector end-to-end.
- **Pre-extracted artifacts for additional benchmarks.** OpenML-CC18, BBBP, SST-2.
- **Reproducibility hardening.** Exact-version `environment.lock.yml` exported from the validated tag environment; per-script seed and runtime annotations.

## v1.2 (target: late 2026)

- **QM9 molecular pipeline.** Released alongside the configuration extractor (see below).
- **Pretrained selector checkpoints** for the headline benchmarks.

## Pending external

- **Public release of the Python port of Parallel-DT / BlueRed.** Held in a private development repository pending (a) feature parity with the upstream journal version (currently under review by Pitsianis et al.), and (b) coordination with the original authors on release timing. Researchers may contact the corresponding author (`shixin.xu@dukekunshan.edu.cn`) for academic-collaboration access.

## Out of scope

- Porting the configuration extractor to other graph backends (e.g., DGL, NetworkX). The interface contract in `docs/configurations.md` is intended to let downstream users plug in alternative extractors directly.

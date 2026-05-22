# Configuration File Format

MixConfig accepts pre-extracted configurations and energy statistics as plain NumPy arrays. This document fixes the contract, so users who substitute their own multi-resolution clustering routine can plug it in directly.

## Files

For a dataset of `n` samples processed at `m` resolutions, two arrays are expected:

| File | Shape | Dtype | Semantics |
|---|---|---|---|
| `configs.npy` | `[n, m]` | `int64` | `configs[i, j]` is the cluster id assigned to sample `i` at the j-th configuration. Cluster ids within a configuration must be contiguous integers starting from 0. |
| `energy_stats.npy` | `[m, 4]` | `float32` | Row `j` is the energy statistics vector `[H_j, h_a^(j), h_r^(j), delta_gamma_j]` for the j-th configuration. |

For convenience, the two arrays can also be bundled in a single `.npz` archive with keys `configs` and `energy_stats`.

## Energy statistics

The four scalars per configuration are produced by the Parallel-DT algorithm and describe the resolution-band structure that makes the configuration *stable*:

| Symbol | Meaning |
|---|---|
| `H` | Entropy of the partition |
| `h_a` | Attractive (within-cluster) graph energy |
| `h_r` | Repulsive (between-cluster) graph energy |
| `delta_gamma` | Width of the resolution band over which this partition is structurally stable |

If you supply your own configurations from a different multi-resolution clustering routine, you must provide all four scalars. Reasonable substitutes:

- `H`: Shannon entropy of the cluster-assignment distribution.
- `h_a` and `h_r`: Sum of edge weights inside and across clusters, respectively, on the same k-NN graph used to generate the assignment.
- `delta_gamma`: Width of any resolution-like band the routine produces; if absent, supply 1.0 uniformly.

The selector treats these as features only and will degrade gracefully if any column is constant; it will not error.

## Loading

```python
import numpy as np
import torch

# Two-file form
configs      = np.load("configs.npy")                          # [n, m] int64
energy_stats = np.load("energy_stats.npy")                     # [m, 4] float32
energy_stats_t = torch.tensor(energy_stats, dtype=torch.float32)

# Bundled form (MNIST-70k reference demo shipped with v1.0.0)
bundle = np.load("data/mnist_configs.npz")
configs, energy_stats = bundle["configs"], bundle["energy_stats"]
energy_stats_t = torch.tensor(energy_stats, dtype=torch.float32)
# Additional keys in the shipped bundle: labels, bluered_lambda, bluered_mu, bluered_gamma_rng.
```

## Sanity checks

A minimal validator is provided in `src/mixconfig/config_extractor.py::validate_configurations`; call it once at load time to catch shape and dtype mismatches before training.

## `max_clusters` and the singleton endpoint

`EnergyAwareSelector` allocates one `nn.Embedding(max_clusters, embed_dim)` table per configuration (see `src/mixconfig/embedder.py`). Cluster ids in any column of `configs` must therefore be strictly less than `max_clusters`. Two practical consequences:

- For the shipped MNIST-70k artifact, the BlueRed front includes a singleton-endpoint configuration (k = 70000, every sample is its own cluster). Either drop it (`configs[:, :-1]`) before passing to the selector, or set `max_clusters=70000` (wasteful — one 70 000-row embedding table per configuration).
- For your own configurations, size `max_clusters` to `int(configs.max()) + 1` and pass `n_configs = configs.shape[1]`. The reference demo at `notebooks/demo_mnist.ipynb` shows the pattern.

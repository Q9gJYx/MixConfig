# MixConfig: Mixing Configurations for Downstream Prediction

[![ICML 2026](https://img.shields.io/badge/ICML-2026-1d4ed8.svg)](https://openreview.net/forum?id=MomisBwZhT)
[![arXiv](https://img.shields.io/badge/arXiv-2510.19248-b31b1b.svg)](https://arxiv.org/abs/2510.19248)
[![OpenReview](https://img.shields.io/badge/OpenReview-MomisBwZhT-8c1b13.svg)](https://openreview.net/forum?id=MomisBwZhT)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)

Official implementation of the ICML 2026 paper **"Mixing Configurations for Downstream Prediction"** by Juntang Wang, Hao Wu, Yihan Wang, Dongmian Zou, and Shixin Xu (Zu Chongzhi Center, Duke Kunshan University). Equal contribution: Wang, Wu, Wang. Corresponding author: Shixin Xu (`shixin.xu@dukekunshan.edu.cn`).

> Clustering-based features are widely used in machine learning, but most methods must choose a resolution: a choice that is global, fixed, and ad hoc. Recent work shows that varying the resolution parameter produces only a finite set of structurally stable partitions, known as configurations. Based on this, we introduce Configuration-Mixed Prediction (CMP), a setting where models learn to adaptively weight these configurations per sample for downstream prediction. We propose **MixConfig**, a plug-and-play feature augmentation module that extracts configurations from any frozen embedding and learns energy-aware mixing weights via a novel selector that jointly reasons about sample context, cluster assignments, and stability statistics. Experiments across tabular, molecular, vision, and text domains demonstrate consistent improvements over single-resolution and static baselines across diverse predictor architectures, with gains particularly pronounced in low-data regimes.

## 📰 News

- **2026-05** Accepted to ICML 2026 (Seoul, July 6 to 11). 🎉
- **2025-10** Preprint released on arXiv ([2510.19248](https://arxiv.org/abs/2510.19248)).

## 📌 Method Overview

MixConfig consists of two components:

1. **Configuration Extraction.** Given input data X, we construct a k-NN graph and apply the Parallel-DT algorithm to obtain multiple configurations (cluster assignments) at the structurally stable resolutions. See `docs/dependencies.md` for the status of this code.
2. **Energy-Aware Selector.** For each sample x, the selector computes:
   - Sample context: `h = MLP_enc(x)`
   - Cluster embeddings: `c_i = Embed_i(omega_i(x))`
   - Energy statistics: `e_i = [H_i, h_a^(i), h_r^(i), delta_gamma_i]`
   - Compatibility scores: `s_i = MLP_score([h; c_i; e_i])`
   - Configuration weights: `w_i(x) = softmax(s_i)`
   - Mixed representation: `z(x) = sum_i w_i(x) * c_i`

## 📦 Installation

```bash
git clone https://github.com/Q9gJYx/MixConfig.git
cd MixConfig

# Conda environment
conda env create -f environment.yml
conda activate mixconfig
```

## 🚀 Quick Start

```python
from src.mixconfig import EnergyAwareSelector
import numpy as np
import torch

selector = EnergyAwareSelector(
    input_dim=X_train.shape[1],
    n_configs=8,
    context_dim=64,
    cluster_embed_dim=32,
)

# Load externally extracted configurations + energy statistics.
# See docs/configurations.md for the file format contract.
configs = np.load("configs.npy")                          # [n_samples, n_configs] int
energy_stats = np.load("energy_stats.npy")                # [n_configs, 4] float
energy_stats_t = torch.tensor(energy_stats, dtype=torch.float32)

z = selector.get_mixed_representation(X_train, configs, energy_stats_t)
```

## 📊 Results

MixConfig improves downstream prediction across four modalities relative to single-resolution and static-mixing baselines. Headline benchmarks:

| Domain | Datasets | Predictors evaluated |
|---|---|---|
| Tabular | OpenML-CC18 (18 tasks) | MLP, TabPFN, FT-Transformer, XGBoost, RF |
| Vision | CIFAR-100, ImageNet-1K | Linear probe on frozen CLIP embeddings |
| Molecular | MolHIV, BBBP, BACE, QM9 | GNN, MLP on Morgan + GIN features |
| Text | SST-2, AG News | Linear probe on BERT, RoBERTa |

Gains are most pronounced in low-data regimes; see `experiments/run_lowdata.py` for the BBBP 5-fold CV protocol. Full numerical results and ablations are in the [camera-ready paper](https://openreview.net/forum?id=MomisBwZhT) (Tables 1 to 4).

## 🔁 Reproducing Paper Results

The MixConfig selector and predictors are fully released. Configuration extraction is described in `docs/dependencies.md`; for end-to-end reproduction, see "Pre-extracted artifacts" below.

### Tabular (OpenML-CC18)

```bash
python experiments/run_tabular.py --dataset openml-cc18 --config experiments/configs/datasets/tabular.yaml --mode base
```

### Vision (CIFAR-100, ImageNet-1K)

```bash
python experiments/run_vision.py --dataset cifar100 --config experiments/configs/datasets/vision.yaml --mode base
python experiments/run_vision.py --dataset imagenet1k --config experiments/configs/datasets/vision.yaml --mode base
```

### Molecular (MolHIV, BBBP, BACE)

```bash
python experiments/run_molecular.py --dataset molhiv --config experiments/configs/datasets/molecular.yaml --mode base
python experiments/run_molecular.py --dataset bbbp   --config experiments/configs/datasets/molecular.yaml --mode base
python experiments/run_molecular.py --dataset bace   --config experiments/configs/datasets/molecular.yaml --mode base
```

### Text (SST-2, AG News)

```bash
python experiments/run_text.py --dataset sst2    --config experiments/configs/datasets/text.yaml --mode base
python experiments/run_text.py --dataset ag_news --config experiments/configs/datasets/text.yaml --mode base
```

### Low-data and ablation

```bash
python experiments/run_lowdata.py  --dataset bbbp --mode base
python experiments/run_ablation.py --dataset bbbp --ablation full
```

### Pre-extracted artifacts

For end-to-end runs without the external configuration extractor, we ship pre-extracted configurations and energy statistics under `data/` (see roadmap below for full benchmark coverage). MNIST is included as the reference demo. See `docs/configurations.md` for the file format and substitution interface.

## 🧩 Repository Structure

```
MixConfig/
├── src/
│   ├── mixconfig/            # Core MixConfig implementation
│   │   ├── selector.py       # Energy-Aware Selector
│   │   ├── encoder.py        # Sample context encoder
│   │   ├── embedder.py       # Cluster assignment embedder
│   │   ├── energy.py         # Energy statistics computation
│   │   └── config_extractor.py  # Interface stub (see docs/dependencies.md)
│   ├── datasets/             # Tabular / vision / molecular / text loaders
│   ├── predictors/           # Neural and classical downstream heads
│   ├── models.py             # Model definitions
│   ├── utils.py              # Utilities
│   └── visuals.py            # Visualization
├── experiments/              # Reproducibility scripts and configs
├── data/                     # Pre-extracted configurations + energy stats
├── docs/                     # Dependency status, interface contracts
├── CITATION.cff              # Machine-readable citation metadata
├── ROADMAP.md                # Planned releases
├── environment.yml           # Conda environment
├── LICENSE                   # MIT
└── README.md
```

## 🔗 Dependencies and Attribution

Configuration extraction uses the **Parallel-DT algorithm** with the **BlueRed front** of [Liu, Floros, Pitsianis, and Sun (HPEC 2021)](https://doi.org/10.1109/HPEC49654.2021.9622834) and [Pitsianis, Floros, Liu, and Sun (HPEC 2023)](https://doi.org/10.1109/HPEC58863.2023.10363552). An updated reference implementation is under journal review by the upstream group. We ship pre-extracted configurations and energy statistics in `data/` so the MixConfig pipeline can be reproduced end-to-end; an interface contract for substituting your own multi-resolution clustering routine is documented in `docs/configurations.md`. See `docs/dependencies.md` for full status of the extractor code.

## 🗺️ Roadmap

- Public release of our Python port of Parallel-DT / BlueRed, pending parity with the upstream journal version and coordination with the original authors.
- QM9 molecular pipeline and pre-extracted configurations for the remaining benchmarks.
- Minimal MNIST demo notebook that loads `data/mnist_70k_configs.npz` and runs the selector end-to-end.

See [ROADMAP.md](ROADMAP.md) for the full plan and tracking issues.

## 🙏 Acknowledgments

We thank Tiancheng Liu, Dimitris Floros, Nikos Pitsianis, and Xiaobai Sun (Duke University) for developing the Parallel-DT and BlueRed algorithms that underpin our configuration extraction. All experiments use publicly available datasets under their respective licenses: OpenML-CC18, CIFAR-100, ImageNet-1K, OGBG-MolHIV, MoleculeNet (BBBP, BACE, QM9), GLUE (SST-2), and AG News.

## 📚 Citation

If you use MixConfig in your research, please cite:

```bibtex
@inproceedings{wang2026mixing,
  title     = {Mixing Configurations for Downstream Prediction},
  author    = {Wang, Juntang and Wu, Hao and Wang, Yihan and Zou, Dongmian and Xu, Shixin},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning ({ICML})},
  year      = {2026},
  note      = {arXiv:2510.19248}
}
```

A machine-readable [CITATION.cff](CITATION.cff) is also provided, which GitHub renders into a "Cite this repository" button on the project page.

## 📝 License

Released under the MIT License. See [LICENSE](LICENSE) for details.

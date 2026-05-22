# [ICML 2026] MixConfig: Mixing Configurations for Downstream Prediction

[![ICML 2026](https://img.shields.io/badge/ICML-2026-1d4ed8.svg)](https://icml.cc/virtual/2026/poster/63010)
[![arXiv](https://img.shields.io/badge/arXiv-2510.19248-b31b1b.svg)](https://arxiv.org/abs/2510.19248)
[![OpenReview](https://img.shields.io/badge/OpenReview-aw6alulxr8-8c1b13.svg)](https://openreview.net/forum?id=aw6alulxr8)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)

Official implementation of the ICML 2026 paper **"Mixing Configurations for Downstream Prediction"** by Juntang Wang, Hao Wu, Yihan Wang, Dongmian Zou, and Shixin Xu (Zu Chongzhi Center, Duke Kunshan University). Equal contribution: Wang, Wu, Wang. Corresponding author: Shixin Xu (`shixin.xu@dukekunshan.edu.cn`).

> Clustering-based features are widely used in machine learning, but most methods must choose a resolution: a choice that is global, fixed, and ad hoc. Recent work shows that varying the resolution parameter produces only a finite set of structurally stable partitions, known as configurations. Based on this, we introduce Configuration-Mixed Prediction (CMP), a setting where models learn to adaptively weight these configurations per sample for downstream prediction. We propose **MixConfig**, a plug-and-play feature augmentation module that extracts configurations from any frozen embedding and learns energy-aware mixing weights via a novel selector that jointly reasons about sample context, cluster assignments, and stability statistics. Experiments across tabular, molecular, vision, and text domains demonstrate consistent improvements over single-resolution and static baselines across diverse predictor architectures, with gains particularly pronounced in low-data regimes.

## 📰 News

- **2026-05** Accepted to ICML 2026 (Seoul, July 6 to 11). 🎉
- **2025-10** Preprint released on arXiv ([2510.19248](https://arxiv.org/abs/2510.19248)).

## 📌 Method

Configurations are the finite, structurally stable partitions discovered by Parallel-DT on a k-NN graph; the Energy-Aware Selector learns per-sample mixing weights over them from sample context, cluster-assignment embeddings, and four stability statistics `[H, h_a, h_r, Δγ]`. Full definitions are in the [paper](https://icml.cc/virtual/2026/poster/63010); the public interface lives in [`src/mixconfig/`](src/mixconfig/).

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
import numpy as np, torch
from src.mixconfig import EnergyAwareSelector, validate_configurations

bundle = np.load("data/mnist_configs.npz")
configs, energy_stats = bundle["configs"][:, :-1], bundle["energy_stats"][:-1]  # drop singleton endpoint
max_id = validate_configurations(configs, energy_stats)

selector = EnergyAwareSelector(
    input_dim=X.shape[1], n_configs=configs.shape[1], max_clusters=max_id + 1,
)
z = selector.get_mixed_representation(
    torch.tensor(X), torch.tensor(configs, dtype=torch.long),
    torch.tensor(energy_stats, dtype=torch.float32),
)
```

The complete end-to-end demo (load, train, compare to single-config and uniform-mix baselines) lives in [`notebooks/demo_mnist.ipynb`](notebooks/demo_mnist.ipynb) — click to run on Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Q9gJYx/MixConfig/blob/main/notebooks/demo_mnist.ipynb)

See [`docs/configurations.md`](docs/configurations.md) for the file-format contract if you want to plug in your own multi-resolution clustering routine.

## 📊 Results

MixConfig improves downstream prediction across four modalities relative to single-resolution and static-mixing baselines. Headline benchmarks:

| Domain | Datasets | Predictors evaluated |
|---|---|---|
| Tabular | OpenML-CC18 (18 tasks) | MLP, TabPFN, FT-Transformer, XGBoost, RF |
| Vision | CIFAR-100, ImageNet-1K | Linear probe on frozen CLIP embeddings |
| Molecular | MolHIV, BBBP, BACE, QM9 | GNN, MLP on Morgan + GIN features |
| Text | SST-2, AG News | Linear probe on BERT, RoBERTa |

Gains are most pronounced in low-data regimes; see `experiments/run_lowdata.py` for the BBBP 5-fold CV protocol. Full numerical results and ablations are in the camera-ready paper ([ICML poster page](https://icml.cc/virtual/2026/poster/63010), [OpenReview](https://openreview.net/forum?id=aw6alulxr8), Tables 1 to 4).

## 🔁 Reproducing Paper Results

Every benchmark uses the same pattern (`--mode base` swaps in the single-resolution baseline; `--mode +config` adds MixConfig):

```bash
python experiments/run_<modality>.py --dataset <name> --config experiments/configs/datasets/<modality>.yaml --mode +config
```

| Modality | Script | Datasets |
|---|---|---|
| Tabular | `run_tabular.py` | `openml-cc18` |
| Vision | `run_vision.py` | `cifar100`, `imagenet1k` |
| Molecular | `run_molecular.py` | `molhiv`, `bbbp`, `bace` |
| Text | `run_text.py` | `sst2`, `ag_news` |
| Low-data + ablation | `run_lowdata.py`, `run_ablation.py` | `bbbp` |

## 🧩 Repository Structure

```
MixConfig/
├── src/
│   ├── mixconfig/            # Core MixConfig implementation
│   │   ├── selector.py       # Energy-Aware Selector
│   │   ├── encoder.py        # Sample context encoder
│   │   ├── embedder.py       # Cluster assignment embedder
│   │   ├── energy.py         # Energy statistics computation
│   │   └── config_extractor.py  # Interface + validator (extractor backend: docs/dependencies.md)
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

Configuration extraction uses the **Parallel-DT algorithm** with the **BlueRed front** of [Liu, Floros, Pitsianis, and Sun (HPEC 2021)](https://doi.org/10.1109/HPEC49654.2021.9622834) and [Pitsianis, Floros, Liu, and Sun (HPEC 2023)](https://doi.org/10.1109/HPEC58863.2023.10363552). An updated reference implementation is under journal review by the upstream group. To support end-to-end reproduction without the extractor, pre-extracted configurations and energy statistics for MNIST-70k (HOG feature space) are released at `data/mnist_configs.npz`; an interface contract for substituting your own multi-resolution clustering routine is documented in `docs/configurations.md`. See `docs/dependencies.md` for full status of the extractor code.

## 🗺️ Roadmap

Headlines: public release of our Python port of Parallel-DT / BlueRed (pending the upstream journal version), QM9 pipeline, and larger benchmark coverage. Full plan in [ROADMAP.md](ROADMAP.md).

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

"""
Smoke test mirroring notebooks/demo_mnist.ipynb without Jupyter.

Run inside the bluered uv env (it has numpy, sklearn, scipy, joblib, skimage)
plus torch added on the fly via uv's --with:

    cd bluered && uv run --extra demo --with torch python ../MixConfig/scripts/smoke_test.py

Asserts hard invariants:
  - Shipped .npz validates via src.mixconfig.validate_configurations
  - BlueRed-front structure matches what data/README.md advertises
  - EnergyAwareSelector forward pass runs and returns the expected shape
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from sklearn.datasets import fetch_openml

from src.mixconfig import EnergyAwareSelector, validate_configurations


def main() -> None:
    bundle_path = REPO / "data" / "mnist_configs.npz"
    print(f"[1/5] Loading {bundle_path}")
    bundle = np.load(bundle_path)
    expected = {
        "configs", "energy_stats", "labels",
        "bluered_lambda", "bluered_mu", "bluered_gamma_rng",
    }
    keys = set(bundle.files)
    assert expected <= keys, f"missing keys: {expected - keys}"
    print(f"      keys: {sorted(keys)}")

    configs = bundle["configs"]
    energy_stats = bundle["energy_stats"]
    print(f"      configs={configs.shape}{configs.dtype}  energy_stats={energy_stats.shape}{energy_stats.dtype}")

    print("[2/5] validate_configurations(...)")
    validate_configurations(configs, energy_stats)
    print("      OK")

    print("[3/5] Verifying BlueRed-front structure")
    n_clusters = [int(len(np.unique(configs[:, j]))) for j in range(configs.shape[1])]
    expected_cluster_counts = [1, 2, 3, 4, 9, 12, 47, 191, 70000]
    assert n_clusters == expected_cluster_counts, (
        f"BlueRed front cluster counts changed: got {n_clusters}, "
        f"expected {expected_cluster_counts}"
    )
    print(f"      cluster counts: {n_clusters} (matches data/README.md)")

    print("[4/5] Sanity-checking energy statistics")
    H = energy_stats[:, 0]
    h_a = energy_stats[:, 1]
    h_r = energy_stats[:, 2]
    delta_gamma = energy_stats[:, 3]
    # Entropy must be non-decreasing along the BlueRed front (more clusters -> at least as much entropy).
    assert np.all(np.diff(H) >= -1e-3), f"entropy not monotonic: {H}"
    # h_a (mean intra-cluster distance) starts non-zero at k=1 and ends at 0 for the singleton config.
    assert h_a[0] > 0 and h_a[-1] == 0, f"h_a endpoints wrong: {h_a}"
    # h_r (inter-centroid distance) is zero at k=1 (only one centroid) and positive elsewhere.
    assert h_r[0] == 0 and (h_r[1:] > 0).all(), f"h_r endpoints wrong: {h_r}"
    print(f"      H range:        [{H.min():.3f}, {H.max():.3f}]")
    print(f"      h_a range:      [{h_a.min():.3f}, {h_a.max():.3f}]")
    print(f"      h_r range:      [{h_r.min():.3f}, {h_r.max():.3f}]")
    print(f"      delta_gamma:    [{delta_gamma.min():.3f}, {delta_gamma.max():.3f}]")

    print("[5/5] EnergyAwareSelector forward pass on 1000 samples")
    # Drop the singleton-endpoint config (k = n samples): each sample its own cluster,
    # too large for the bounded ClusterAssignmentEmbedder and uninformative for the
    # selector. Keep the 8 lower-resolution configs (k up to 191).
    keep = configs.shape[1] - 1
    configs_kept = configs[:, :keep]
    energy_kept = energy_stats[:keep]
    max_k = int(configs_kept.max()) + 1
    print(f"      keeping {keep}/{configs.shape[1]} configs; max cluster id = {max_k - 1}")

    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X = mnist.data[:1000].astype(np.float32) / 255.0
    configs_sub = configs_kept[:1000].astype(np.int64)

    selector = EnergyAwareSelector(
        input_dim=X.shape[1],
        n_configs=keep,
        max_clusters=max_k,
        context_dim=64,
        cluster_embed_dim=32,
    )
    selector.eval()

    with torch.no_grad():
        z = selector.get_mixed_representation(
            torch.tensor(X),
            torch.tensor(configs_sub, dtype=torch.long),
            torch.tensor(energy_kept, dtype=torch.float32),
        )
    assert z.shape == (1000, 32), f"expected (1000, 32), got {tuple(z.shape)}"
    assert torch.isfinite(z).all(), "non-finite values in mixed representation"
    print(f"      z={tuple(z.shape)} {z.dtype}  finite=True  mean={z.mean().item():.4f}  std={z.std().item():.4f}")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()

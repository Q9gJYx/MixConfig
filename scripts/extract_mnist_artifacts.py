"""
Extract MixConfig pre-extraction artifacts for the MNIST-70k reference demo.

Inputs (read-only):
    bluered/notebooks/cache/mnist_70k.npz   -- produced by bluered/scripts/precompute_demo_cache.py
    MNIST-784 from OpenML                   -- cached by sklearn after first fetch

Output:
    MixConfig/data/mnist_configs.npz with keys:
      - configs            [n, m] int64    cluster assignments from BlueRed front
      - energy_stats       [m, 4] float32  [H, h_a, h_r, delta_gamma]
      - labels             [n]    int64    MNIST class labels (for downstream evaluation)
      - bluered_lambda     [m]    float64  BlueRed persistence (informational)
      - bluered_mu         [m]    float64  BlueRed steadiness (informational)
      - bluered_gamma_rng  [m, 2] float64  BlueRed resolution intervals (informational)

H, h_a, h_r follow the schema in src/mixconfig/energy.py (Shannon entropy of
the partition; mean intra-cluster distance and mean inter-centroid distance in
HOG feature space). delta_gamma follows the paper definition: the width of the
resolution band on the gamma axis, taken directly from bluered_gamma_rng. The
singleton-endpoint configuration has an open-ended resolution band, capped here
at 2x the largest finite band so the model input stays finite.

Run:
    cd bluered && uv run --extra demo python ../MixConfig/scripts/extract_mnist_artifacts.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("KMP_WARNINGS", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist
from skimage.feature import hog
from sklearn.datasets import fetch_openml


HERE = Path(__file__).resolve().parent
MIXCONFIG_ROOT = HERE.parent
BLUERED_CACHE = MIXCONFIG_ROOT.parent / "bluered" / "notebooks" / "cache" / "mnist_70k.npz"
OUTPUT_PATH = MIXCONFIG_ROOT / "data" / "mnist_configs.npz"


def _timed(label, fn, *args, **kwargs):
    t = time.perf_counter()
    out = fn(*args, **kwargs)
    print(f"  [{label}] {time.perf_counter() - t:.1f}s")
    return out


def _hog_one(img):
    return hog(img, orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2))


def compute_hog_features(X: np.ndarray) -> np.ndarray:
    return np.stack(Parallel(n_jobs=-1)(delayed(_hog_one)(img) for img in X)).astype(np.float32)


def _inter_stats(centroids: np.ndarray, max_full: int = 1500, rng_seed: int = 0):
    """Mean and max of pairwise centroid distances.

    Full pdist is O(n^2) memory. For n > max_full, fall back to a Monte Carlo
    estimate on a uniform random subsample of centroids. The relative bias of
    the resulting mean estimate is well under 1% for n_sample = 1500.
    """
    n = len(centroids)
    if n < 2:
        return 0.0, 0.0
    if n <= max_full:
        d = pdist(centroids)
        return float(d.mean()), float(d.max())
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(n, size=max_full, replace=False)
    d = pdist(centroids[idx])
    return float(d.mean()), float(d.max())


def _gamma_band_widths(gamma_rng: np.ndarray) -> np.ndarray:
    """Resolution-band widths from BlueRed gamma intervals (paper-side delta_gamma).

    The singleton-endpoint configuration has gamma_rng[m-1, 1] = inf. Replace
    that infinity with 2x the largest finite width so the resulting energy_stats
    column stays bounded for the model.
    """
    widths = gamma_rng[:, 1] - gamma_rng[:, 0]
    finite_mask = np.isfinite(widths)
    if finite_mask.all():
        return widths
    cap = 2.0 * widths[finite_mask].max() if finite_mask.any() else 1.0
    return np.where(finite_mask, widths, cap)


def compute_energy_stats(
    features: np.ndarray,
    configs: np.ndarray,
    gamma_rng: np.ndarray,
) -> np.ndarray:
    """[H, h_a, h_r, delta_gamma] per configuration.

    H, h_a, h_r are computed in feature space following src/mixconfig/energy.py.
    delta_gamma is the gamma-axis band width from `gamma_rng` (paper definition).
    """
    n_configs = configs.shape[1]
    if gamma_rng.shape != (n_configs, 2):
        raise ValueError(
            f"gamma_rng shape {gamma_rng.shape} does not match configs columns {n_configs}"
        )
    delta_gamma_all = _gamma_band_widths(gamma_rng).astype(np.float32)
    out = np.zeros((n_configs, 4), dtype=np.float32)

    for j in range(n_configs):
        cfg = configs[:, j]
        counts = np.bincount(cfg.astype(np.int64))
        counts = counts[counts > 0].astype(np.float64)
        probs = counts / counts.sum()
        H = float(-np.sum(probs * np.log(probs + 1e-12)))

        centroids, intra = [], []
        for c in np.unique(cfg):
            mask = cfg == c
            feats = features[mask]
            centroid = feats.mean(axis=0)
            centroids.append(centroid)
            if mask.sum() > 1:
                intra.append(float(np.linalg.norm(feats - centroid, axis=1).mean()))
        centroids = np.stack(centroids)

        h_a = float(np.mean(intra)) if intra else 0.0
        h_r, _max_inter = _inter_stats(centroids)
        out[j] = [H, h_a, h_r, float(delta_gamma_all[j])]
        n_clusters = len(np.unique(cfg))
        sampled = " (sampled 1500/{n})".format(n=n_clusters) if n_clusters > 1500 else ""
        print(
            f"  config[{j}] k={n_clusters} "
            f"H={H:.3f} h_a={h_a:.3f} h_r={h_r:.3f} dg={delta_gamma_all[j]:.4g}{sampled}"
        )
    return out


def main() -> None:
    if not BLUERED_CACHE.exists():
        raise FileNotFoundError(
            f"bluered cache not found at {BLUERED_CACHE}. "
            "Run bluered/scripts/precompute_demo_cache.py first."
        )

    print(f"Loading bluered cache: {BLUERED_CACHE}")
    cache = np.load(BLUERED_CACHE)
    cid = cache["cid"].astype(np.int64)
    y = cache["y"].astype(np.int64)
    print(f"  cid={cid.shape}  labels={y.shape}  n_configs={cid.shape[1]}")

    print("Fetching MNIST-784 (cached on second run)...")
    mnist = _timed("fetch_openml", fetch_openml, "mnist_784", version=1, as_frame=False, parser="liac-arff")
    X = mnist.data.astype(np.float32).reshape(-1, 28, 28) / 255.0
    assert X.shape[0] == cid.shape[0], f"size mismatch: MNIST {X.shape[0]} vs cid {cid.shape[0]}"

    print("Computing HOG features (9 orient, 7x7/cell, 2x2/block)...")
    hog_features = _timed("hog", compute_hog_features, X)
    print(f"  hog={hog_features.shape}")

    print("Computing energy statistics (H/h_a/h_r from HOG features; delta_gamma from bluered_gamma_rng):")
    energy_stats = compute_energy_stats(hog_features, cid, cache["gamma_rng"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        configs=cid,
        energy_stats=energy_stats,
        labels=y,
        bluered_lambda=cache["lambda_"],
        bluered_mu=cache["mu"],
        bluered_gamma_rng=cache["gamma_rng"],
    )
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"\nWrote {OUTPUT_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()

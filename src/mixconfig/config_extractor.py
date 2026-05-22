"""
Configuration Extractor for MixConfig.

This module defines the interface for configuration extraction. The Parallel-DT
algorithm with the BlueRed front (Liu et al., HPEC 2021; Pitsianis et al., HPEC
2023) is the upstream backend; see docs/dependencies.md for full attribution
and release status. Pre-extracted configurations and energy statistics are
provided under data/ (see data/README.md for current contents and roadmap);
the file-format contract is documented in docs/configurations.md.
"""

from typing import Tuple, Optional, Dict
import numpy as np


def validate_configurations(
    configs: np.ndarray,
    energy_stats: np.ndarray,
) -> None:
    """
    Sanity-check pre-extracted configurations and energy statistics.

    Raises ValueError on shape or dtype mismatches, with a message that
    points the caller to docs/configurations.md.
    """
    if configs.ndim != 2:
        raise ValueError(
            f"configs must be 2-D [n_samples, n_configs], got shape {configs.shape}. "
            "See docs/configurations.md."
        )
    if not np.issubdtype(configs.dtype, np.integer):
        raise ValueError(
            f"configs must be integer dtype, got {configs.dtype}. "
            "See docs/configurations.md."
        )
    if energy_stats.ndim != 2 or energy_stats.shape[1] != 4:
        raise ValueError(
            f"energy_stats must be 2-D [n_configs, 4], got shape {energy_stats.shape}. "
            "See docs/configurations.md."
        )
    if not np.issubdtype(energy_stats.dtype, np.floating):
        raise ValueError(
            f"energy_stats must be a floating-point dtype, got {energy_stats.dtype}. "
            "See docs/configurations.md."
        )
    if energy_stats.shape[0] != configs.shape[1]:
        raise ValueError(
            f"energy_stats rows ({energy_stats.shape[0]}) must equal "
            f"configs columns ({configs.shape[1]}). See docs/configurations.md."
        )


class ConfigExtractor:
    """
    Configuration extraction interface; the upstream backend is described in
    docs/dependencies.md. For end-to-end runs, load pre-extracted artifacts from
    data/ instead of instantiating this class.

    Expected API:
        extractor = ConfigExtractor(n_neighbors=15, n_configs=8)
        configs, energy_stats = extractor.extract(X)

    Args:
        n_neighbors: Number of nearest neighbors for k-NN graph.
        n_configs: Number of configurations to extract.
        metric: Distance metric for k-NN. Default: 'euclidean'.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        n_configs: int = 8,
        metric: str = "euclidean",
        random_state: Optional[int] = 42,
    ):
        self.n_neighbors = n_neighbors
        self.n_configs = n_configs
        self.metric = metric
        self.random_state = random_state

        self._is_fitted = False
        self._knn_graph = None
        self._configs = None
        self._energy_stats = None

    def _build_knn_graph(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build k-NN graph from input data.

        Args:
            X: Input features of shape [n_samples, n_features].

        Returns:
            Tuple of (distances, indices) arrays.
        """
        raise NotImplementedError(
            "k-NN graph construction is not exposed in this release. "
            "See docs/dependencies.md for the status of the upstream extractor."
        )

    def extract(
        self,
        X: np.ndarray,
        return_graph: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Extract configurations and energy statistics from input data.

        Args:
            X: Input features of shape [n_samples, n_features].
            return_graph: Whether to also return the k-NN graph.

        Returns:
            Tuple of:
            - configs: Configuration assignments of shape [n_samples, n_configs]
            - energy_stats: Dictionary containing energy statistics
            - (optional) knn_graph: Tuple of (distances, indices) if return_graph=True
        """
        raise NotImplementedError(
            "ConfigExtractor.extract() is not available in this release. "
            "See docs/dependencies.md for the status of the upstream extractor "
            "and docs/configurations.md for how to plug in your own."
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply fitted configuration extraction to new data.

        Args:
            X: New input features of shape [n_samples, n_features].

        Returns:
            Configuration assignments of shape [n_samples, n_configs].
        """
        if not self._is_fitted:
            raise RuntimeError("ConfigExtractor must be fitted before transform.")

        raise NotImplementedError(
            "ConfigExtractor.transform() is not available in this release. "
            "See docs/dependencies.md for the status of the upstream extractor."
        )

    @property
    def configs(self) -> Optional[np.ndarray]:
        """Return extracted configurations."""
        return self._configs

    @property
    def energy_stats(self) -> Optional[Dict[str, np.ndarray]]:
        """Return computed energy statistics."""
        return self._energy_stats

    @property
    def n_clusters_per_config(self) -> Optional[np.ndarray]:
        """Return number of clusters in each configuration."""
        if self._configs is None:
            return None
        return np.array([len(np.unique(self._configs[:, i])) for i in range(self.n_configs)])

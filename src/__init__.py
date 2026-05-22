"""MixConfig: Mixing Configurations for Downstream Prediction (ICML 2026).

This top-level package exposes the experiment-runner helpers under src.utils and
the core MixConfig modules under src.mixconfig. Importing this package does not
load PyTorch Lightning or plotting dependencies; submodules pull them in only
when actually used.
"""

__all__ = ["mixconfig", "datasets", "predictors", "models", "utils", "visuals"]

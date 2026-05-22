"""MixConfig: Mixing Configurations for Downstream Prediction (ICML 2026).

This top-level package re-exports the core MixConfig modules so
``from src import *`` works without users having to know the submodule layout.
Importing this package does not load PyTorch Lightning or plotting
dependencies; the submodules pull them in only when actually used.
"""

from . import mixconfig

__all__ = ["mixconfig"]

"""BAPMOS — Bandit-Adaptive Prompting for Multi-Organ Segmentation."""

__version__ = "0.1.0"

# Deprecated ``bapmos.optimization`` / ``singleorgan`` / ``ablations`` /
# ``pfus1_advanced`` resolve to ``bapmos.legacy.*`` (same module objects).
from bapmos._legacy_aliases import install as _install_legacy_aliases

_install_legacy_aliases()

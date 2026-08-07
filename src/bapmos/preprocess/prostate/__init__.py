"""Prostate preprocessing: RTSTRUCT→masks, stratified splits, pooled corpus.

Package entry (``python -m bapmos.preprocess.prostate``) builds ``data/prostate/pooled/``.
Other tools are submodule CLIs, e.g.::

    python -m bapmos.preprocess.prostate.run_rtstruct_masks --help
    python -m bapmos.preprocess.prostate.create_stratified_splits --help
"""

from bapmos.preprocess.prostate.build_pooled import build_pooled, main

__all__ = ["build_pooled", "main"]

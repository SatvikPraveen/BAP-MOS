"""Mask-overlap / delineation QA for prostate site corpora.

Commands: ``overlays``, ``summarize``, ``organ-presence``
(``python -m bapmos.preprocess.delineation --help``).
"""

from __future__ import annotations

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    from bapmos.preprocess.delineation._cli import main as _main

    return _main(argv)

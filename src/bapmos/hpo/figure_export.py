"""Save HPO / outer-loop matplotlib figures as PNG and PDF (separate savefig calls)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SavedFigure:
    png: Path
    pdf: Path


def save_matplotlib_figure(
    fig: Any,
    output_path: Path,
    *,
    dpi: int = 160,
    bbox_inches: str = "tight",
    **kwargs: Any,
) -> SavedFigure:
    """
    Write the same figure to ``.png`` and ``.pdf`` via matplotlib ``savefig``.

    *output_path* may end in ``.png`` or ``.pdf``; the sibling extension is derived
    automatically.
    """
    path = Path(output_path)
    if path.suffix.lower() == ".pdf":
        pdf_path = path
        png_path = path.with_suffix(".png")
    else:
        png_path = path.with_suffix(".png")
        pdf_path = path.with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_kw = {"dpi": dpi, "bbox_inches": bbox_inches, **kwargs}
    fig.savefig(png_path, **save_kw)
    fig.savefig(pdf_path, format="pdf", **save_kw)
    return SavedFigure(png=png_path, pdf=pdf_path)

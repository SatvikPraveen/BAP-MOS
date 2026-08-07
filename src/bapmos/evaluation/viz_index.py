"""Write visualization_index.csv next to qualitative figure exports."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

# ``panel_*``: multiclass test panels (baseline viz).
# ``viz_*`` / ``diff_*``: overlay + difference_v1 pairs from ``export_stratified_test_bundle``.
VISUALIZATION_INDEX_COLUMNS = [
    "sample_id",
    "split",
    "mean_dice",
    "mean_msd",
    "mean_hd95",
    "distance_unit",
    "panel_png_relative",
    "panel_pdf_relative",
    "viz_png_relative",
    "viz_pdf_relative",
    "diff_png_relative",
    "diff_pdf_relative",
    "visualization_selection_mode",
]


def write_visualization_index_csv(output_dir: Path, rows: List[Dict[str, Any]]) -> Path:
    """
    Write ``visualization_index.csv`` under ``output_dir``.

    Paths in rows should be relative to ``output_dir``. Unknown keys are ignored;
    missing columns are written empty.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "visualization_index.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=VISUALIZATION_INDEX_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in VISUALIZATION_INDEX_COLUMNS}
            # If a caller only filled viz_* (export_bundle), mirror into panel_* so
            # older index consumers that read panel_png_relative still work.
            if not out.get("panel_png_relative") and out.get("viz_png_relative"):
                out["panel_png_relative"] = out["viz_png_relative"]
            if not out.get("panel_pdf_relative") and out.get("viz_pdf_relative"):
                out["panel_pdf_relative"] = out["viz_pdf_relative"]
            w.writerow(out)
    return path

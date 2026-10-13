"""Shared defaults for stratified test exports under ``inference_output/``.

Canonical viz protocol: see ``bapmos.inference_output.protocol``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from bapmos.inference_output.protocol import (
    DEFAULT_VIZ_PDF_MAX,
    export_viz_protocol_kwargs,
)

# Back-compat name used by older CLIs / docs.
DEFAULT_PFUS1_VIZ_PER_PATIENT = DEFAULT_VIZ_PDF_MAX


def export_viz_kwargs_for_data_root(
    data_root: Union[str, Path],
    *,
    per_patient_max: Optional[int] = None,
    viz_pdf_max: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Return kwargs for ``export_stratified_test_bundle(...)``.

    Protocol (all corpora):
      - prediction PNGs + matplotlib overlay/difference PNG+PDF for the first
        ``viz_pdf_max`` evaluated slices only (default 10; same stems across methods)
      - metrics still computed on the full test split
      - panel DPI = 350 (``PDF_EXPORT_DPI``)

    ``per_patient_max`` is accepted for CLI back-compat but ignored; the public
    protocol is global first-N image/PDF export, not per-patient subsampling.
    """
    del data_root, per_patient_max  # corpus-agnostic protocol
    return export_viz_protocol_kwargs(viz_pdf_max=viz_pdf_max)

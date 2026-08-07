"""
Resolve folders that contain MR + RTSTRUCT DICOM for each training corpus.

Tries canonical ``data/.../Dicom`` paths first, then common alternates (e.g.
``dcm_files``). Used by ``bapmos.preprocess.prostate.run_rtstruct_masks`` and
``bapmos.preprocess.prostate.rtstruct_export_slice_masks``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from bapmos.paths import project_root, research_tree_root


def _has_dicom_files(p: Path) -> bool:
    return p.is_dir() and any(p.glob("*.dcm"))


def candidate_dicom_dirs_for_case(case: str) -> List[Path]:
    """Ordered search list (may include non-existent paths)."""
    roots = [project_root()]
    research = research_tree_root()
    if research is not None and research not in roots:
        roots.append(research)

    k = case.strip().lower().replace("-", "_")
    rels: List[Path]
    if k in ("case1", "case_1"):
        rels = [
            Path("data") / "real_data" / "Case 1" / "Dicom",
            Path("data") / "real_data" / "Case1" / "Dicom",
            Path("data") / "real_data" / "Case 1" / "dcm_files",
            Path("data") / "real_data" / "Case1" / "dcm_files",
        ]
    elif k in ("case2", "case_2"):
        rels = [
            Path("data") / "real_data" / "Case 2" / "Dicom",
            Path("data") / "real_data" / "Case2" / "Dicom",
            Path("data") / "real_data" / "Case 2" / "dcm_files",
            Path("data") / "real_data" / "Case2" / "dcm_files",
        ]
    elif k in ("simulation", "sim", "simulation_data"):
        rels = [Path("data") / "simulation_data" / "dicom"]
    else:
        raise ValueError(f"Unknown case {case!r}")

    out: List[Path] = []
    for root in roots:
        for rel in rels:
            out.append(root / rel)
    return out


def resolve_dicom_dir_for_case(case: str) -> Optional[Path]:
    """First candidate under ``candidate_dicom_dirs_for_case`` that contains ``*.dcm``."""
    for c in candidate_dicom_dirs_for_case(case):
        if _has_dicom_files(c):
            return c.resolve()
    return None

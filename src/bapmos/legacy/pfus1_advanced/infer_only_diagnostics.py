"""Infer-only PFUS1-advanced diagnostics (canonical checkpoint, isolated outputs)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from bapmos.paths import project_root

_PFUS1_ADVANCED_CONFIGS_NOTE = (
    "Historical PFUS1-advanced YAML under configs/pfus1_advanced/ is not included "
    "in this repository. Provide those configs locally to run infer-only diagnostics."
)


@dataclass(frozen=True)
class InferOnlyDiagnostic:
    diag_id: str
    description: str
    data_root: str
    prompt_config: Optional[Path]
    checkpoint: Path
    output_dir: Path
    splits_subdir: str


def _load_doc() -> Dict[str, Any]:
    path = project_root() / "configs/pfus1_advanced/infer_only_diagnostics.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: {_PFUS1_ADVANCED_CONFIGS_NOTE}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def discover_infer_only_diagnostics() -> List[InferOnlyDiagnostic]:
    root = project_root()
    doc = _load_doc()
    ckpt = root / doc["checkpoint"]
    out_root = root / doc.get("output_root", "output/pfus1_advanced/diagnostics_infer_only")
    splits = doc.get("splits_subdir", "splits_patient_70_15_15_seed42")
    cases: List[InferOnlyDiagnostic] = []
    for diag_id, spec in (doc.get("diagnostics") or {}).items():
        prompt = spec.get("prompt_config")
        cases.append(
            InferOnlyDiagnostic(
                diag_id=diag_id,
                description=spec.get("description", ""),
                data_root=spec["data_root"],
                prompt_config=(root / prompt) if prompt else None,
                checkpoint=ckpt,
                output_dir=out_root / diag_id / "test",
                splits_subdir=splits,
            )
        )
    return cases


def resolve_infer_only_manifest(diag: InferOnlyDiagnostic) -> Dict[str, Any]:
    root = project_root()
    test_out = diag.output_dir
    summary = test_out / "summary_metrics.csv"
    pred_ids = test_out / "predictions/multiclass"
    has_preds = pred_ids.is_dir() and any(pred_ids.glob("*_pred_ids.png"))
    return {
        "diag_id": diag.diag_id,
        "kind": "infer_only",
        "description": diag.description,
        "checkpoint": str(diag.checkpoint.relative_to(root)),
        "data_root": diag.data_root,
        "prompt_config": (
            str(diag.prompt_config.relative_to(root)) if diag.prompt_config else None
        ),
        "output_dir": str(test_out.relative_to(root)),
        "summary_metrics_csv": (
            str(summary.relative_to(root)) if summary.is_file() else None
        ),
        "pred_ids_dir": (
            str(pred_ids.relative_to(root)) if has_preds else None
        ),
        "infer_status": "complete" if summary.is_file() else "pending",
    }


def _link_outputs(subdir: Path, test_output: Optional[Path]) -> None:
    link = subdir / "outputs"
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir() and not link.is_symlink():
        return
    if test_output is None or not test_output.is_dir():
        return
    rel = os.path.relpath(test_output.resolve(), subdir.resolve())
    link.symlink_to(rel)


def write_infer_only_tree(
    out_root: Optional[Path] = None,
    *,
    diag_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    root = project_root()
    out_root = (out_root or root / "diagnostics/pfus1_advanced/infer_only").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"infer_only": {}}
    for diag in discover_infer_only_diagnostics():
        if diag_ids and diag.diag_id not in diag_ids:
            continue
        subdir = out_root / diag.diag_id
        subdir.mkdir(parents=True, exist_ok=True)
        manifest = resolve_infer_only_manifest(diag)
        (subdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        test_p = root / manifest["output_dir"] if manifest.get("output_dir") else None
        _link_outputs(subdir, test_p)
        report["infer_only"][diag.diag_id] = manifest
        print(f"[infer_only] {diag.diag_id} infer={manifest['infer_status']}")
    return report


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Infer-only diagnostic manifests + symlinks")
    p.add_argument("--list", action="store_true")
    p.add_argument("--diag", action="append", dest="diags")
    args = p.parse_args()

    if args.list:
        for d in discover_infer_only_diagnostics():
            print(f"  {d.diag_id}: {d.description}")
        return 0

    write_infer_only_tree(diag_ids=args.diags)
    print("Wrote diagnostics/pfus1_advanced/infer_only/<diag_id>/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

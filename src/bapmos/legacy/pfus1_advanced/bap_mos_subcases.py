"""
BAP-MOS-Tuned subcases: one folder per experiment, symlink to existing outputs.

- ``reference/canonical_pfus1`` — links to completed ``output/pfus1/...`` (no rerun).
- ``subcases/*`` — pfus1_advanced prompt-geometry variants; ``outputs`` symlink when infer done.

Does not include U-Net / nnU-Net / MedSAM.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from bapmos.paths import project_root


_PFUS1_ADVANCED_CONFIGS_NOTE = (
    "Historical PFUS1-advanced YAML under configs/pfus1_advanced/ is not shipped "
    "in this repository. Provide those configs locally to "
    "run infer-only / subcase diagnostics."
)


@dataclass(frozen=True)
class BapMosSubcase:
    subcase_id: str
    experiment: str
    config_path: Path
    description: str
    data_root: str
    prompt_geometry_profile: str
    bundle: str
    is_reference: bool = False


def _require_config(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{path}: {_PFUS1_ADVANCED_CONFIGS_NOTE}")
    return path


def _load_yaml() -> Dict[str, Any]:
    path = _require_config(project_root() / "configs/pfus1_advanced/experiment_subcases.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def discover_train_subcases() -> List[BapMosSubcase]:
    root = project_root()
    cases: List[BapMosSubcase] = []
    doc = _load_yaml()
    for sid, spec in doc.get("subcases", {}).items():
        cfg_path = root / spec["config"]
        common: Dict[str, Any] = {}
        if cfg_path.is_file():
            with open(cfg_path) as f:
                common = (yaml.safe_load(f) or {}).get("common") or {}
        cases.append(
            BapMosSubcase(
                subcase_id=sid,
                experiment=spec["experiment"],
                config_path=cfg_path,
                description=spec.get("description", ""),
                data_root=common.get("data_root", "data/bladder/pfus1_advanced"),
                prompt_geometry_profile=common.get("prompt_geometry_profile", sid),
                bundle="pfus1_advanced",
                is_reference=False,
            )
        )
    return cases


def discover_reference_subcases() -> List[BapMosSubcase]:
    root = project_root()
    ref_path = root / "configs/pfus1_advanced/reference_canonical.yaml"
    if not ref_path.is_file():
        return []
    with open(ref_path) as f:
        ref = yaml.safe_load(f) or {}
    return [
        BapMosSubcase(
            subcase_id=ref.get("reference_id", "canonical_pfus1"),
            experiment=ref["experiment"],
            config_path=root / ref["config"],
            description="Existing canonical PFUS1 BAP-MOS-Tuned (reference — no rerun)",
            data_root=ref.get("data_root", "data/bladder/pfus1"),
            prompt_geometry_profile=ref.get("prompt_geometry_profile", "fixed_ring"),
            bundle=ref.get("bundle", "pfus1"),
            is_reference=True,
        )
    ]


def _find_run_dir(bundle: str, experiment: str) -> Optional[Path]:
    opt = project_root() / "runs" / bundle / "Optimization"
    if not opt.is_dir():
        return None
    matches: List[Path] = []
    for method_dir in opt.iterdir():
        if not method_dir.is_dir():
            continue
        matches.extend(sorted(method_dir.glob(f"{experiment}*")))
    for m in reversed(matches):
        if (m / "best_checkpoint.pth").is_file() or (m / "last_checkpoint.pth").is_file():
            return m
    return matches[-1] if matches else None


def _find_test_output(bundle: str, experiment: str) -> Optional[Path]:
    base = project_root() / "output" / bundle / "Optimization"
    if not base.is_dir():
        return None
    candidates: List[Path] = []
    for method_dir in base.iterdir():
        if not method_dir.is_dir():
            continue
        for run_dir in method_dir.glob(f"{experiment}*"):
            test_dir = run_dir / "test"
            if test_dir.is_dir():
                candidates.append(test_dir)
        direct = method_dir / "test"
        if method_dir.name.startswith(experiment) and direct.is_dir():
            candidates.append(direct)
    return sorted(candidates)[-1] if candidates else None


def resolve_subcase_outputs(sc: BapMosSubcase) -> Dict[str, Any]:
    root = project_root()
    run_dir = _find_run_dir(sc.bundle, sc.experiment)
    test_out = _find_test_output(sc.bundle, sc.experiment)

    ckpt_best = run_dir / "best_checkpoint.pth" if run_dir else None
    pred_ids = None
    if test_out and (test_out / "predictions/multiclass").is_dir():
        ids = test_out / "predictions/multiclass"
        if any(ids.glob("*_pred_ids.png")):
            pred_ids = ids

    summary = test_out / "summary_metrics.csv" if test_out else None

    return {
        "subcase_id": sc.subcase_id,
        "is_reference": sc.is_reference,
        "experiment": sc.experiment,
        "bundle": sc.bundle,
        "data_root": sc.data_root,
        "prompt_geometry_profile": sc.prompt_geometry_profile,
        "config_path": str(sc.config_path.relative_to(root)),
        "description": sc.description,
        "run_dir": str(run_dir.relative_to(root)) if run_dir else None,
        "checkpoint_best": str(ckpt_best.relative_to(root)) if ckpt_best and ckpt_best.is_file() else None,
        "test_output_dir": str(test_out.relative_to(root)) if test_out else None,
        "summary_metrics_csv": str(summary.relative_to(root)) if summary and summary.is_file() else None,
        "pred_ids_dir": str(pred_ids.relative_to(root)) if pred_ids else None,
        "train_status": "complete" if ckpt_best and ckpt_best.is_file() else "not_started",
        "infer_status": "complete" if summary and summary.is_file() else "pending",
    }


def _link_outputs(subdir: Path, test_output: Optional[Path]) -> None:
    """Symlink subdir/outputs -> test_output (relative). No file copies."""
    link = subdir / "outputs"
    if link.is_symlink() or link.exists():
        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.is_dir() and not link.is_symlink():
            return  # do not delete real dirs
    if test_output is None or not test_output.is_dir():
        return
    rel = os.path.relpath(test_output.resolve(), subdir.resolve())
    link.symlink_to(rel)


def write_subcase_tree(
    out_root: Optional[Path] = None,
    *,
    subcase_ids: Optional[List[str]] = None,
    include_reference: bool = True,
) -> Dict[str, Any]:
    root = project_root()
    out_root = (out_root or root / "diagnostics/pfus1_advanced").resolve()

    report: Dict[str, Any] = {
        "reference": {},
        "subcases": {},
        "note": "Manifests + outputs symlinks only. No prediction copies. No external baselines.",
    }

    if include_reference:
        ref_root = out_root / "reference"
        ref_root.mkdir(parents=True, exist_ok=True)
        for sc in discover_reference_subcases():
            subdir = ref_root / sc.subcase_id
            subdir.mkdir(parents=True, exist_ok=True)
            manifest = resolve_subcase_outputs(sc)
            (subdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
            test_p = root / manifest["test_output_dir"] if manifest.get("test_output_dir") else None
            _link_outputs(subdir, test_p)
            report["reference"][sc.subcase_id] = manifest
            print(f"[reference] {sc.subcase_id} infer={manifest['infer_status']}")

    sub_root = out_root / "subcases"
    sub_root.mkdir(parents=True, exist_ok=True)
    for sc in discover_train_subcases():
        if subcase_ids and sc.subcase_id not in subcase_ids:
            continue
        subdir = sub_root / sc.subcase_id
        subdir.mkdir(parents=True, exist_ok=True)
        manifest = resolve_subcase_outputs(sc)
        (subdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        test_p = root / manifest["test_output_dir"] if manifest.get("test_output_dir") else None
        _link_outputs(subdir, test_p)
        report["subcases"][sc.subcase_id] = manifest
        print(f"[subcase] {sc.subcase_id} train={manifest['train_status']} infer={manifest['infer_status']}")

    (out_root / "index.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="BAP-MOS subcase folders → existing outputs")
    p.add_argument("--list", action="store_true")
    p.add_argument("--subcase", action="append", dest="subcases")
    p.add_argument("--no-reference", action="store_true")
    args = p.parse_args()

    if args.list:
        print("Reference (no rerun):")
        for sc in discover_reference_subcases():
            print(f"  {sc.subcase_id}")
        print("Train subcases (pfus1_advanced):")
        for sc in discover_train_subcases():
            print(f"  {sc.subcase_id}: {sc.description}")
        return 0

    write_subcase_tree(subcase_ids=args.subcases, include_reference=not args.no_reference)
    print(f"\nWrote diagnostics/pfus1_advanced/{{reference,subcases}}/<id>/manifest.json + outputs symlink")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

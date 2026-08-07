"""Job discovery helpers under ``inference_output/`` (canonical layout via ``paths``).

Prefer :func:`bapmos.paths.inference_output_layout_root` for layout roots.
This module keeps checkpoint→job discovery for legacy ``runs/<bundle>/`` trees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from bapmos.paths import (
    inference_output_layout_root,
    method_slug_from_checkpoint,
    pfus1_advanced_bundle_dir,
    pfus1_bundle_dir,
    project_root,
    real_case_dataset_dir,
    resolve_under_project,
    simulation_dataset_dir,
)

_TS_SUFFIX = re.compile(r"_\d{8}_\d{6}$")

# Bundle ids match ``runs/<bundle>/`` (underscore form: case_1, case_2).
_RUNS_BUNDLES = frozenset({"simulation", "case_1", "case_2", "pfus1", "pfus1_advanced"})

_NNUNET_PTV_BUNDLES = frozenset({"simulation", "case_1", "case_2", "pfus1_advanced"})


def strip_run_timestamp(name: str) -> str:
    """Remove trailing ``_YYYYMMDD_HHMMSS`` from run folder names."""
    return _TS_SUFFIX.sub("", name)


def inference_output_root(repo_root: Optional[Path] = None) -> Path:
    root = resolve_under_project(repo_root) if repo_root else project_root()
    return (root / "inference_output").resolve()


def inference_output_dir(
    data_root: str | Path,
    *parts: str,
    repo_root: Optional[Path] = None,
) -> Path:
    """``<layout_root>/<parts...>/`` using :func:`inference_output_layout_root`.

    Layout roots (canonical)::

        data/prostate/pooled → inference_output/prostate/pooled/
        data/bladder/pfus1   → inference_output/bladder/pfus1/
    """
    base = inference_output_layout_root(data_root, repo_root=repo_root)
    if not parts:
        return base.resolve()
    return (base / Path(*parts)).resolve()


def data_root_for_runs_bundle(bundle: str) -> str:
    """Resolve training ``data_root`` for a ``runs/<bundle>/`` id via path helpers.

    Prostate site corpora use ``simulation_dataset_dir`` / ``real_case_dataset_dir``
    (prefer ``data/prostate/...``, optional parent-layout ``preprocessing/...`` fallback).
    Classic PFUS1 uses ``pfus1_bundle_dir()`` (``data/bladder/pfus1``).
    PFUS1-advanced uses ``pfus1_advanced_bundle_dir()`` (with optional parent-layout fallback).
    """
    if bundle == "simulation":
        return str(simulation_dataset_dir())
    if bundle == "case_1":
        return str(real_case_dataset_dir("case1"))
    if bundle == "case_2":
        return str(real_case_dataset_dir("case2"))
    if bundle == "pfus1":
        return str(pfus1_bundle_dir())
    if bundle == "pfus1_advanced":
        return str(pfus1_advanced_bundle_dir())
    raise ValueError(f"Unknown runs bundle {bundle!r}")


@dataclass(frozen=True)
class InferenceJob:
    """One stratified test export job."""

    family: str  # core | unet | nnunet | bapmos
    bundle: str  # simulation | case_1 | case_2 | pfus1 | pfus1_advanced
    data_root: str
    output_dir: Path
    label: str
    checkpoint: Optional[Path] = None


_MULTI_EXPERIMENT_STRATEGIES = {
    "box_point": "box_point",
    "three_way": "three_way",
    "boxpoint_box_point": "three_way",
}


def _classify_core_checkpoint(rel: Path, ckpt: Path, bundle: str, data_root: str) -> Optional[InferenceJob]:
    if rel.parts[1] != "Optimization":
        return None
    strategy = rel.parts[2]
    run_folder = rel.parts[3] if len(rel.parts) > 3 else ""

    if strategy.startswith("bapmos_"):
        ablation_id = strategy
        experiment = run_folder
        out = inference_output_dir(data_root, "bapmos", ablation_id, experiment)
        return InferenceJob(
            family="bapmos",
            bundle=bundle,
            data_root=data_root,
            output_dir=out,
            label=f"bapmos/{ablation_id}/{experiment}",
            checkpoint=ckpt,
        )

    if strategy in _MULTI_EXPERIMENT_STRATEGIES:
        folder = _MULTI_EXPERIMENT_STRATEGIES[strategy]
        exp = strip_run_timestamp(run_folder)
        out = inference_output_dir(data_root, "core", folder, exp)
        return InferenceJob(
            family="core",
            bundle=bundle,
            data_root=data_root,
            output_dir=out,
            label=f"core/{folder}/{exp}",
            checkpoint=ckpt,
        )

    out = inference_output_dir(data_root, "core", strategy)
    return InferenceJob(
        family="core",
        bundle=bundle,
        data_root=data_root,
        output_dir=out,
        label=f"core/{strategy}",
        checkpoint=ckpt,
    )


def _classify_baseline_checkpoint(rel: Path, ckpt: Path, bundle: str, data_root: str) -> Optional[InferenceJob]:
    if rel.parts[1] != "Baseline":
        return None
    slug = method_slug_from_checkpoint(ckpt)
    out = inference_output_dir(data_root, "core", slug)
    return InferenceJob(
        family="core",
        bundle=bundle,
        data_root=data_root,
        output_dir=out,
        label=f"core/{slug}",
        checkpoint=ckpt,
    )


def _classify_unet_checkpoint(rel: Path, ckpt: Path, bundle: str, data_root: str) -> Optional[InferenceJob]:
    if rel.parts[1] != "ExternalBaselines" or len(rel.parts) < 3 or rel.parts[2] != "unet":
        return None
    out = inference_output_dir(data_root, "unet")
    return InferenceJob(
        family="unet",
        bundle=bundle,
        data_root=data_root,
        output_dir=out,
        label="unet",
        checkpoint=ckpt,
    )


def _classify_medsam_checkpoint(rel: Path, ckpt: Path, bundle: str, data_root: str) -> Optional[InferenceJob]:
    if rel.parts[1] != "ExternalBaselines" or len(rel.parts) < 3 or rel.parts[2] != "medsam_init":
        return None
    out = inference_output_dir(data_root, "medsam")
    return InferenceJob(
        family="medsam",
        bundle=bundle,
        data_root=data_root,
        output_dir=out,
        label="medsam",
        checkpoint=ckpt,
    )


def classify_checkpoint(ckpt: Path, *, repo_root: Optional[Path] = None) -> Optional[InferenceJob]:
    """Map ``runs/<bundle>/.../best_checkpoint.pth`` to an inference_output job."""
    ckpt = resolve_under_project(ckpt)
    root = resolve_under_project(repo_root) if repo_root else project_root()
    runs_root = root / "runs"
    try:
        rel = ckpt.relative_to(runs_root)
    except ValueError:
        return None
    if ckpt.name != "best_checkpoint.pth":
        return None
    if len(rel.parts) < 2:
        return None
    if rel.parts[0] not in _RUNS_BUNDLES:
        return None
    if "SingleOrgan" in rel.parts:
        return None

    bundle = rel.parts[0]
    data_root = data_root_for_runs_bundle(bundle)

    if rel.parts[1] == "Optimization":
        if len(rel.parts) < 3:
            return None
        return _classify_core_checkpoint(rel, ckpt, bundle, data_root)
    if rel.parts[1] == "Baseline":
        return _classify_baseline_checkpoint(rel, ckpt, bundle, data_root)
    if rel.parts[1] == "ExternalBaselines":
        if job := _classify_unet_checkpoint(rel, ckpt, bundle, data_root):
            return job
        return _classify_medsam_checkpoint(rel, ckpt, bundle, data_root)
    return None


def discover_checkpoint_jobs(*, repo_root: Optional[Path] = None) -> List[InferenceJob]:
    """All checkpoint-backed jobs under ``runs/{simulation,case_1,case_2,...}/``."""
    root = resolve_under_project(repo_root) if repo_root else project_root()
    jobs: List[InferenceJob] = []
    for bundle in sorted(_RUNS_BUNDLES):
        bundle_root = root / "runs" / bundle
        if not bundle_root.is_dir():
            continue
        for ckpt in sorted(bundle_root.rglob("best_checkpoint.pth")):
            job = classify_checkpoint(ckpt, repo_root=root)
            if job is not None:
                jobs.append(job)
    return jobs


def nnunet_jobs(*, repo_root: Optional[Path] = None) -> List[InferenceJob]:
    """One nnU-Net re-predict + eval job per PTV-style bundle (not classic PFUS1).

    ``checkpoint`` is left ``None``: prostate/advanced nnU-Net export is driven by
    an external ``nnUNet_predict`` (or package wrapper) that locates fold checkpoints
    under ``runs/<bundle>/nnUNet_results*`` itself; this job only reserves the
    ``inference_output/.../nnunet`` destination.
    """
    root = resolve_under_project(repo_root) if repo_root else project_root()
    jobs: List[InferenceJob] = []
    for bundle in sorted(_NNUNET_PTV_BUNDLES):
        data_root = data_root_for_runs_bundle(bundle)
        out = inference_output_dir(data_root, "nnunet", repo_root=root)
        jobs.append(
            InferenceJob(
                family="nnunet",
                bundle=bundle,
                data_root=data_root,
                output_dir=out,
                label="nnunet",
                checkpoint=None,
            )
        )
    return jobs


def pfus1_nnunet_job(*, repo_root: Optional[Path] = None) -> Optional[InferenceJob]:
    """Classic PFUS1 nnU-Net predict + ``inference_output/bladder/pfus1/nnunet`` export.

    Unlike :func:`nnunet_jobs`, this attaches the known fold ``checkpoint_best.pth``
    when present so callers can pass it to the PFUS1 nnU-Net wrapper without a
    second discovery pass.
    """
    root = resolve_under_project(repo_root) if repo_root else project_root()
    data_root = data_root_for_runs_bundle("pfus1")
    ckpt = (
        root
        / "runs/pfus1/nnUNet_results/Dataset503_BapMosPfus1TrainVal"
        / "nnUNetTrainerBapMosProtocol__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
    )
    if not ckpt.is_file():
        alt = (
            root
            / "runs/pfus1/nnUNet_results_bapmos_protocol/Dataset503_BapMosPfus1TrainVal"
            / "nnUNetTrainerBapMosProtocol__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
        )
        if alt.is_file():
            ckpt = alt
        else:
            return None
    out = inference_output_dir(data_root, "nnunet", repo_root=root)
    return InferenceJob(
        family="nnunet_pfus1",
        bundle="pfus1",
        data_root=data_root,
        output_dir=out,
        label="nnunet",
        checkpoint=ckpt,
    )


def discover_pfus1_jobs(*, repo_root: Optional[Path] = None) -> List[InferenceJob]:
    """Checkpoint-backed jobs under ``runs/pfus1/`` plus PFUS1 nnU-Net when trained."""
    root = resolve_under_project(repo_root) if repo_root else project_root()
    jobs = [job for job in discover_checkpoint_jobs(repo_root=root) if job.bundle == "pfus1"]
    nn_job = pfus1_nnunet_job(repo_root=root)
    if nn_job is not None:
        jobs.append(nn_job)
    return _dedupe_jobs_by_output(jobs)


def discover_all_jobs(*, repo_root: Optional[Path] = None) -> List[InferenceJob]:
    """Checkpoint jobs + PTV nnU-Net placeholders + classic PFUS1 nnU-Net when trained."""
    root = resolve_under_project(repo_root) if repo_root else project_root()
    jobs = discover_checkpoint_jobs(repo_root=root) + nnunet_jobs(repo_root=root)
    pfus1_nn = pfus1_nnunet_job(repo_root=root)
    if pfus1_nn is not None:
        jobs.append(pfus1_nn)
    return _dedupe_jobs_by_output(jobs)


def _dedupe_jobs_by_output(jobs: List[InferenceJob]) -> List[InferenceJob]:
    """When multiple checkpoints map to the same output_dir, keep the newest checkpoint."""
    best: dict[Path, InferenceJob] = {}
    for job in jobs:
        prev = best.get(job.output_dir)
        if prev is None:
            best[job.output_dir] = job
            continue
        if job.checkpoint is None:
            continue
        if prev.checkpoint is None:
            best[job.output_dir] = job
            continue
        if job.checkpoint.stat().st_mtime >= prev.checkpoint.stat().st_mtime:
            best[job.output_dir] = job
    return sorted(best.values(), key=lambda j: (j.bundle, j.label))


def iter_job_lines(jobs: Iterable[InferenceJob]) -> Iterator[str]:
    for job in jobs:
        ckpt = job.checkpoint.as_posix() if job.checkpoint else ""
        yield "|".join(
            [
                job.family,
                job.bundle,
                job.data_root,
                str(job.output_dir),
                job.label,
                ckpt,
            ]
        )

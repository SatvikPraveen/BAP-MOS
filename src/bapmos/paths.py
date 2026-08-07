"""Portable path helpers for the BAPMOS checkout.

``project_root()`` is the BAPMOS package root (contains ``src/``, ``data/``, ``configs/``).
Optional parent-layout fallbacks are available via ``BAP_MOS_RESEARCH_ROOT`` for
legacy nested checkouts; a standalone tree only needs paths under this package.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def project_root() -> Path:
    """BAPMOS checkout root (contains ``src/``, ``data/``, ``configs/``)."""
    return Path(__file__).resolve().parents[2]


def research_tree_root() -> Optional[Path]:
    """
    Optional parent directory for legacy nested layouts.

    Override with ``BAP_MOS_RESEARCH_ROOT``. Returns ``None`` for a standalone checkout.
    """
    import os

    env = os.environ.get("BAP_MOS_RESEARCH_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    parent = project_root().parent
    if (parent / "core").is_dir() and (parent / "preprocessing").is_dir():
        return parent.resolve()
    return None


def _first_existing_dir(*candidates: Path) -> Path:
    for c in candidates:
        if c is not None and c.is_dir():
            return c
    return candidates[0]


def exports_root() -> Path:
    """
    nnU-Net export tree (default: ``<project>/exports``).

    Override with ``BAP_MOS_EXPORTS_ROOT`` or ``BAP_MOS_CLUSTER_EXPORTS`` when exports
    live outside the checkout.
    """
    import os

    env = os.environ.get("BAP_MOS_EXPORTS_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    repo_exports = project_root() / "exports"
    if repo_exports.is_symlink() or repo_exports.is_dir():
        return repo_exports.resolve()
    # Optional cluster override (never auto-pick a machine-specific absolute path).
    cluster = os.environ.get("BAP_MOS_CLUSTER_EXPORTS", "").strip()
    if cluster:
        cluster_path = Path(cluster).expanduser()
        if cluster_path.is_dir():
            return cluster_path.resolve()
    return repo_exports.resolve()


def nnunet_raw_root(bundle: str) -> Path:
    """``exports/nnUNet_raw/<bundle>/`` (e.g. bundle ``case_1``, ``simulation``)."""
    return exports_root() / "nnUNet_raw" / bundle


def nnunet_preprocessed_root(bundle: str) -> Path:
    """``exports/nnUNet_preprocessed/<bundle>/``."""
    return exports_root() / "nnUNet_preprocessed" / bundle


def preprocessing_root() -> Path:
    """Derived datasets / QA under an optional parent layout; else ``data/``."""
    research = research_tree_root()
    if research is not None:
        return research / "preprocessing"
    return project_root() / "data"


def simulation_dataset_dir() -> Path:
    """Simulation training bundle: images, ``masks/``, ``splits_stratified/``."""
    research = research_tree_root()
    return _first_existing_dir(
        project_root() / "data" / "prostate" / "simulation",
        *( [research / "preprocessing" / "simulation_data"] if research else [] ),
        project_root() / "data" / "prostate" / "simulation",
    )


def real_case_dataset_dir(case: str) -> Path:
    """Clinical case training bundle: images, ``masks/``, ``splits_stratified/``."""
    key = case.lower().replace("_", "")
    if key not in ("case1", "case2"):
        raise ValueError(f"Unknown case {case!r}; expected 'case1' or 'case2'")
    research = research_tree_root()
    return _first_existing_dir(
        project_root() / "data" / "prostate" / key,
        *( [research / "preprocessing" / "real_data" / key] if research else [] ),
        project_root() / "data" / "prostate" / key,
    )


def pooled_prostate_dataset_dir() -> Path:
    """Canonical pooled corpus: ``BAPMOS/data/prostate/pooled`` (create or symlink as needed)."""
    return project_root() / "data" / "prostate" / "pooled"


def is_pooled_prostate_training_root(data_root) -> bool:
    """True when ``data_root`` resolves to the canonical pooled prostate bundle."""
    p = Path(data_root)
    if not p.is_absolute():
        p = project_root() / p
    return p.resolve() == pooled_prostate_dataset_dir().resolve()


def global_split_test_txt(
    data_root: str | Path,
    splits_subdir: str = "splits_stratified",
) -> Path:
    """``<data_root>/<splits_subdir>/test.txt`` (may not exist for pooled prostate)."""
    p = Path(data_root)
    if not p.is_absolute():
        p = project_root() / p
    return p.resolve() / splits_subdir / "test.txt"


def should_skip_in_train_global_test(
    data_root: str | Path,
    splits_subdir: str = "splits_stratified",
    *,
    force_skip: bool = False,
) -> bool:
    """Whether trainers should omit a global ``test.txt`` DataLoader.

    Pooled prostate uses ``site_tests/<site>/test.txt`` only — stratified export is a
    separate post-train step. Also skip when the global list file is missing.
    """
    if force_skip:
        return True
    if is_pooled_prostate_training_root(data_root):
        return True
    return not global_split_test_txt(data_root, splits_subdir).is_file()


def pooled_site_tests_root(data_root: str | Path | None = None) -> Path:
    """``data/prostate/pooled/site_tests/`` (per-cohort ``test.txt`` lists)."""
    root = pooled_prostate_dataset_dir() if data_root is None else Path(data_root)
    if not root.is_absolute():
        root = project_root() / root
    return root.resolve() / "site_tests"


def list_pooled_site_test_names(data_root: str | Path | None = None) -> list[str]:
    """Sorted site names under ``site_tests/`` that contain ``test.txt``."""
    site_root = pooled_site_tests_root(data_root)
    if not site_root.is_dir():
        return []
    names: list[str] = []
    for child in sorted(site_root.iterdir()):
        if child.is_dir() and (child / "test.txt").is_file():
            names.append(child.name)
    return names


def simulation_cleaned_dataset_dir() -> Path:
    """Canonical simulation ``data_root`` (same as ``simulation_dataset_dir``)."""
    return simulation_dataset_dir()


def real_data_processing_root() -> Path:
    """Legacy Processing tree under preprocessing (optional runs/outputs)."""
    return preprocessing_root() / "real_data" / "Processing"


def case_real_data_dir(case: str) -> Path:
    """case: 'case1' or 'case2'. Prefer ``data/prostate/caseN``; else legacy nested bundles."""
    key = case.lower().replace("_", "")
    flat = real_case_dataset_dir(key)
    if flat.is_dir():
        return flat
    if key == "case1":
        return real_data_processing_root() / "case1_real_data"
    if key == "case2":
        return real_data_processing_root() / "case2_real_data"
    raise ValueError(f"Unknown case {case!r}; expected 'case1' or 'case2'")


def find_training_images_dir(dataset_root: Path) -> Path:
    """
    Directory of slice PNGs for training / stratification.

    Tries ``images/`` first, then common DICOM-export folder names.
    """
    candidates = [
        dataset_root / "images",
        dataset_root / "case1_dicom_png",
        dataset_root / "case2_dicom_png",
        dataset_root / "simulation_dicom_png",
    ]
    for c in candidates:
        if c.is_dir() and any(c.glob("*.png")):
            return c
    raise FileNotFoundError(
        f"No image directory with PNGs under {dataset_root}. Tried: {[str(x) for x in candidates]}"
    )


def find_combined_masks_dir(dataset_root: Path) -> Path:
    """
    Directory containing training multiclass files ``*_combined_mask.png``.

    Only that exact suffix is matched; optional ``combined_masks/combined_mask_preview/``
    (per-slice QA PNG + PDF, rasterized at ``bapmos.pdf_export.PDF_EXPORT_DPI``) is ignored by
    training and by this discovery helper.

    Matches the clinical bundle layout under legacy
    ``.../case1_real_data/data/masks/``:

    - ``masks/combined_masks/*_combined_mask.png`` — multiclass training labels (used first)
    - ``masks/Bladder``, ``PTV``, ``Rectum``, ``Urethra`` — per-organ exports
      (ignored by loaders; copy the whole ``masks/`` tree when migrating)

    If ``combined_masks`` is missing, falls back to ``masks/`` when combined
    PNGs live there directly.
    """
    candidates = [
        dataset_root / "masks" / "combined_masks",
        dataset_root / "masks",
    ]
    for c in candidates:
        if c.is_dir() and any(c.glob("*_combined_mask.png")):
            return c
    raise FileNotFoundError(
        f"No combined-mask directory found under {dataset_root}. "
        f"Tried: {[str(x) for x in candidates]}"
    )


def dataset_bundle_tag(dataset_root: str | Path) -> str:
    """
    Stable slug matching Slurm/log layout under ``logs/<tag>/``.

    Used for defaults like ``runs/<tag>/Baseline`` so checkpoints stay separate per
    simulation / clinical case / PFUS1 bundles.
    """
    p = Path(dataset_root)
    if not p.is_absolute():
        p = project_root() / p
    try:
        if p.resolve() == pooled_prostate_dataset_dir().resolve():
            return "prostate_pooled"
    except OSError:
        pass
    parts_lower = tuple(part.lower() for part in p.parts)
    if "pooled" in parts_lower and (
        "prostate" in parts_lower or "prostate_investigation" in parts_lower
    ):
        return "prostate_pooled"
    if "pfus1_advanced" in parts_lower:
        return "pfus1_advanced"
    if "pfus1" in parts_lower:
        return "pfus1"
    if "simulation_data" in parts_lower or (
        "prostate" in parts_lower and "simulation" in parts_lower
    ):
        return "simulation"
    for part in parts_lower:
        if part == "case2" or part.startswith("case2"):
            return "case_2"
        if part == "case1" or part.startswith("case1"):
            return "case_1"
    return "other"


def _case_key_for_test_output_combined_masks(dataset_root: Path) -> Optional[str]:
    """Map preprocessing dataset root to ``test_output/<bundle>/<key>/`` case folder name."""
    tag = dataset_bundle_tag(dataset_root)
    if tag == "simulation":
        return "simulation"
    if tag == "case_1":
        return "case1"
    if tag == "case_2":
        return "case2"
    return None


def find_combined_masks_dir_with_repo_fallback(
    dataset_root: Path,
    repo_root: Optional[Path] = None,
    *,
    allow_test_output_fallback: bool = False,
) -> Path:
    """
    Resolve ``*_combined_mask.png`` directory for ``dataset_root``.

    When :func:`find_combined_masks_dir` finds masks under ``dataset_root`` (canonical
    ``masks/combined_masks`` or ``masks/``), that directory is returned — it is never
    overridden by a larger pool under ``test_output/`` (which would mis-associate stems).

    If that lookup fails, searches any ``masks/**`` tree under ``dataset_root``, then
    optionally ``<repo_root>/test_output/<bundle>/<case>/masks/combined_masks`` and the
    same pattern under ``<repo_root>/preprocessing/<bundle>/…`` (largest pool wins).

    **Training loaders must use** :func:`find_combined_masks_dir` only. Set
    ``allow_test_output_fallback=True`` only for QA utilities (e.g. stratified-split
  generation when canonical masks are missing).

    Raises ``FileNotFoundError`` if no multiclass masks exist anywhere.
    """
    if repo_root is None:
        repo_root = project_root()
    try:
        return find_combined_masks_dir(dataset_root).resolve()
    except FileNotFoundError:
        pass

    candidates: List[Path] = []

    masks = dataset_root / "masks"
    if masks.is_dir():
        by_parent: Dict[Path, int] = defaultdict(int)
        for p in masks.rglob("*_combined_mask.png"):
            parts_lower = {x.lower() for x in p.parts}
            if "combined_mask_preview" in parts_lower:
                continue
            by_parent[p.parent] += 1
        candidates.extend(by_parent.keys())

    key = _case_key_for_test_output_combined_masks(dataset_root)
    test_out = repo_root / "test_output"
    if allow_test_output_fallback and key and test_out.is_dir():
        try:
            for bundle in sorted(test_out.iterdir()):
                if not bundle.is_dir():
                    continue
                cand = bundle / key / "masks" / "combined_masks"
                if cand.is_dir():
                    candidates.append(cand)
        except OSError:
            pass

    prep = preprocessing_root()
    if key and prep.is_dir():
        try:
            for bundle in sorted(prep.iterdir()):
                if not bundle.is_dir():
                    continue
                cand = bundle / key / "masks" / "combined_masks"
                if cand.is_dir():
                    candidates.append(cand)
        except OSError:
            pass

    seen: set[str] = set()
    uniq: List[Path] = []
    for p in candidates:
        try:
            r = p.resolve()
        except OSError:
            r = p
        k = str(r)
        if k not in seen:
            seen.add(k)
            uniq.append(r)

    best: Optional[Path] = None
    best_n = 0
    for c in uniq:
        if not c.is_dir():
            continue
        n = sum(1 for _ in c.glob("*_combined_mask.png"))
        if n > best_n:
            best_n = n
            best = c
    if best is None or best_n == 0:
        raise FileNotFoundError(
            f"No *_combined_mask.png found under {dataset_root.resolve()} or under "
            f"{repo_root.resolve()}/test_output/*/…/combined_masks or preprocessing/*/…/combined_masks. "
            "Generate masks (e.g. preprocessing_run_rtstruct_masks or run_rtstruct_masks_test_output) first."
        )
    return best


def resolve_training_data_root(case: str) -> Path:
    """``data_root`` for MultiOrganDataset: flat case dir, or legacy ``.../data`` if present."""
    base = case_real_data_dir(case)
    legacy = base / "data"
    if legacy.is_dir():
        return legacy
    return base


def _resolve_under_project_root(p: Path) -> Path:
    if not p.is_absolute():
        p = project_root() / p
    return p.resolve()


def resolve_under_project(path_like) -> Path:
    """Resolve ``path_like`` relative to the project root unless already absolute."""
    return _resolve_under_project_root(Path(path_like))


def resolve_model_checkpoint(path_like) -> Path:
    """
    Resolve a foundation model weight under ``BAPMOS/models/``.

    Canonical layout (standalone checkout)::

        BAPMOS/models/sam_base/sam_vit_b_01ec64.pth
        BAPMOS/models/medsam/medsam_vit_b.pth

    If the local file is missing, optionally falls back to
    ``<BAP_MOS_RESEARCH_ROOT>/models/...`` for legacy nested layouts.

    Absolute paths from another machine that point into a ``.../BAPMOS/models/...``
    suffix are remapped onto this checkout's ``models/`` tree when the foreign
    file is absent.
    """
    root = project_root().resolve()

    def _research_alt(rel_posix: str) -> Optional[Path]:
        research = research_tree_root()
        if research is None:
            return None
        alt = (research / rel_posix).resolve()
        return alt if alt.is_file() else None

    def _remap_foreign_absolute(abs_path: Path) -> Optional[Path]:
        s = abs_path.as_posix()
        # Prefer stripping through the BAPMOS package root marker.
        for marker in ("/BAP-MOS_Research_Project/BAPMOS/", "/BAPMOS/"):
            if marker in s:
                rel = s.split(marker, 1)[1]
                local = (root / rel).resolve()
                if local.is_file():
                    return local
        if "/models/" in s:
            rel = "models/" + s.split("/models/", 1)[1]
            local = (root / rel).resolve()
            if local.is_file():
                return local
        return None

    p = Path(path_like)
    if p.is_absolute():
        # Do not Path.resolve() missing foreign paths (can raise on dangling links).
        if p.is_file():
            return p.resolve()
        remapped = _remap_foreign_absolute(p)
        if remapped is not None:
            return remapped
        try:
            rel = p.resolve().relative_to(root).as_posix()
        except (ValueError, OSError, RuntimeError):
            return p
        alt = _research_alt(rel)
        return alt if alt is not None else p

    rel = p.as_posix().lstrip("./")
    local = (root / rel).resolve()
    if local.is_file():
        return local
    alt = _research_alt(rel)
    return alt if alt is not None else local


def pfus1_bundle_dir() -> Path:
    """Canonical PFUS1 training bundle: ``BAPMOS/data/bladder/pfus1`` (create or symlink)."""
    return project_root() / "data" / "bladder" / "pfus1"


def pfus1_advanced_bundle_dir() -> Path:
    """
    PFUS1 advanced bundle: cone-cropped, letterboxed images + warped masks.

    Prefer ``BAPMOS/data/bladder/pfus1_advanced``; optional parent-layout
    fallback to ``preprocessing/pfus1_advanced``.
    """
    research = research_tree_root()
    return _first_existing_dir(
        project_root() / "data" / "bladder" / "pfus1_advanced",
        *( [research / "preprocessing" / "pfus1_advanced"] if research else [] ),
        project_root() / "data" / "bladder" / "pfus1_advanced",
    )


def pfus1_advanced_image_root() -> Path:
    """Standardized PFUS1 frames: ``…/pfus1_advanced/images/Pxxx/``."""
    return pfus1_advanced_bundle_dir() / "images"


def pfus1_image_root() -> Path:
    """
    Raw PFUS1 acquisitions: ``Pxxx/frame_*.png`` (+ JSON).

    Prefer ``data/bladder/pfus1_raw``; optional parent-layout fallback to ``data/pfus1``.
    """
    research = research_tree_root()
    return _first_existing_dir(
        project_root() / "data" / "bladder" / "pfus1_raw",
        *( [research / "data" / "pfus1"] if research else [] ),
        project_root() / "data" / "bladder" / "pfus1_raw",
    )


def is_pfus1_training_root(data_root) -> bool:
    """True when ``data_root`` resolves to the canonical PFUS1 bundle."""
    return _resolve_under_project_root(Path(data_root)) == pfus1_bundle_dir().resolve()


def is_pfus1_advanced_training_root(data_root) -> bool:
    """True when ``data_root`` resolves to the PFUS1-advanced bundle."""
    return _resolve_under_project_root(Path(data_root)) == pfus1_advanced_bundle_dir().resolve()


def is_pfus1_family_training_root(data_root) -> bool:
    """True for canonical PFUS1 or PFUS1-advanced preprocessing bundles."""
    return is_pfus1_training_root(data_root) or is_pfus1_advanced_training_root(data_root)


_RUNS_BUNDLES = (
    "prostate",
    "bladder",
    "simulation",
    "case_1",
    "case_2",
    "pfus1",
    "pfus1_advanced",
)
# Back-compat alias used in older call sites
_INTERNAL_RUNS_BUNDLES = _RUNS_BUNDLES


def inference_output_layout_root(
    data_root: str | Path,
    repo_root: Optional[Path] = None,
) -> Path:
    """
    Top-level inference output directory for a dataset bundle.

    BAP-MOS layout::

        inference_output/prostate/pooled/
        inference_output/bladder/pfus1/

    Legacy clinical / simulation layout (unchanged)::

        output/simulation/
        output/real_data/case1/
        output/real_data/case2/
        output/pfus1/            # legacy alias; prefer inference_output/bladder/pfus1
    """
    root = resolve_under_project(repo_root) if repo_root else project_root()
    tag = dataset_bundle_tag(str(data_root))
    if tag == "prostate_pooled":
        return root / "inference_output" / "prostate" / "pooled"
    if tag == "pfus1":
        return root / "inference_output" / "bladder" / "pfus1"
    if tag == "pfus1_advanced":
        return root / "inference_output" / "bladder" / "pfus1_advanced"
    if tag == "simulation":
        return root / "output" / "simulation"
    if tag == "case_1":
        return root / "output" / "real_data" / "case1"
    if tag == "case_2":
        return root / "output" / "real_data" / "case2"
    raise ValueError(
        f"No inference output layout for data_root={data_root!r} (tag={tag!r}). "
        "Expected prostate pooled, pfus1, simulation, clinical case1/case2, or pfus1_advanced."
    )


def model_rel_path_from_checkpoint(
    checkpoint: str | Path,
    repo_root: Optional[Path] = None,
) -> Path:
    """
    Relative path under ``runs/<bundle>/`` for a trained run directory.

    Example::

        runs/case_1/Optimization/box_point/box90_point10_20260516_205046/best_checkpoint.pth
        -> Optimization/box_point/box90_point10_20260516_205046
    """
    ckpt = resolve_under_project(checkpoint)
    root = resolve_under_project(repo_root) if repo_root else project_root()
    runs_root = root / "runs"
    try:
        rel = ckpt.parent.relative_to(runs_root)
    except ValueError as exc:
        raise ValueError(
            f"Checkpoint must live under {runs_root}/<bundle>/.../best_checkpoint.pth; got {ckpt}"
        ) from exc
    bundle = rel.parts[0]
    if bundle not in _RUNS_BUNDLES:
        raise ValueError(
            f"Unknown runs bundle {bundle!r} in {ckpt}; expected one of {_RUNS_BUNDLES}"
        )
    return Path(*rel.parts[1:])


# Flat method folder names under ``output/<layout>/`` (PTV / simulation bundles).
_EXTERNAL_BASELINE_METHOD_SLUGS: Dict[str, str] = {
    "unet": "u_net",
    "nnunet2d": "nn_unet",
    "medsam_init": "medsam_init",
}

_OPTIMIZATION_STRATEGY_METHOD_SLUGS: Dict[str, str] = {
    "ucb1_global": "ucb1_global",
    "ucb1_per_organ": "ucb1_per_organ",
    "bap_mos_tuned": "bap_mos_tuned",
    "ucb_tuned_per_organ_majority": "ucb_tuned_per_organ_majority",
    "box_point": "box_point",
    "boxpoint_box_point": "boxpoint_box_point",
    "epsilon_greedy_per_organ": "epsilon_greedy_per_organ",
    "epsilon_decay_per_organ": "epsilon_decay_per_organ",
}


def method_slug_from_checkpoint_rel(rel: Path) -> str:
    """
    Map ``runs/<bundle>/...`` relative path (without bundle) to a flat output method folder.

    Examples::

        Optimization/ucb1_global/run_name -> ucb1_global
        ExternalBaselines/unet/run_name -> u_net
        Baseline/run_name -> multiorgan_points | multiorgan_box (from checkpoint config when loaded)
    """
    parts = rel.parts
    if not parts:
        raise ValueError(f"Empty checkpoint relative path: {rel}")
    if parts[0] == "Optimization" and len(parts) >= 2:
        strategy = parts[1]
        return _OPTIMIZATION_STRATEGY_METHOD_SLUGS.get(strategy, strategy)
    if parts[0] == "ExternalBaselines" and len(parts) >= 2:
        sub = parts[1]
        return _EXTERNAL_BASELINE_METHOD_SLUGS.get(sub, sub.replace("-", "_"))
    if parts[0] == "Baseline":
        return "multiorgan_baseline"
    return parts[0].replace("-", "_")


def method_slug_from_checkpoint(
    checkpoint: str | Path,
    *,
    repo_root: Optional[Path] = None,
) -> str:
    """Infer method folder slug; refines ``Baseline`` runs using checkpoint ``config``."""
    rel = model_rel_path_from_checkpoint(checkpoint, repo_root=repo_root)
    slug = method_slug_from_checkpoint_rel(rel)
    if slug != "multiorgan_baseline":
        return slug

    ckpt = resolve_under_project(checkpoint)
    try:
        import torch

        try:
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(ckpt, map_location="cpu")
        cfg = payload.get("config") or {}
        pt = str(cfg.get("prompt_type", "")).lower()
        if "box" in pt and "point" not in pt:
            return "multiorgan_box"
        if "point" in pt:
            return "multiorgan_points"
    except Exception:
        pass
    name = ckpt.parent.name.lower()
    if "box" in name:
        return "multiorgan_box"
    if "point" in name:
        return "multiorgan_points"
    return "multiorgan_baseline"


def method_test_output_dir(
    data_root: str | Path,
    method_slug: str,
    *,
    repo_root: Optional[Path] = None,
) -> Path:
    """
    Flat test output root for a dataset + method::

        output/real_data/case1/ucb1_global/
        output/simulation/u_net/
    """
    layout = inference_output_layout_root(data_root, repo_root=repo_root)
    slug = method_slug.strip().replace("-", "_")
    if not slug:
        raise ValueError("method_slug must be non-empty")
    return (layout / slug).resolve()


def method_test_output_dir_from_checkpoint(
    checkpoint: str | Path,
    data_root: str | Path,
    *,
    method_slug: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> Path:
    """``method_test_output_dir`` using slug inferred from checkpoint (or override)."""
    slug = method_slug or method_slug_from_checkpoint(checkpoint, repo_root=repo_root)
    return method_test_output_dir(data_root, slug, repo_root=repo_root)


def write_method_evaluation_meta(
    output_dir: Path,
    *,
    checkpoint: str | Path,
    data_root: str | Path,
    method_slug: str,
    split: str = "test",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``evaluation_meta.json`` at the method output root for traceability."""
    import json
    from datetime import datetime, timezone

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = resolve_under_project(checkpoint)
    meta: Dict[str, Any] = {
        "method_slug": method_slug,
        "split": split,
        "checkpoint": str(ckpt.resolve()),
        "data_root": str(resolve_under_project(data_root).resolve()),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        meta.update(extra)
    path = output_dir / "evaluation_meta.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path


def inference_output_dir_for_checkpoint(
    checkpoint: str | Path,
    data_root: str | Path,
    split: str = "test",
    repo_root: Optional[Path] = None,
    *,
    layout_style: str = "method",
    method_slug: Optional[str] = None,
) -> Path:
    """
    Canonical test output directory for an internal-bundle checkpoint.

    **method** (default for PTV workflows): ``output/<layout>/<method_slug>/``

    **legacy**: ``output/<layout>/<runs-relative-path>/test/`` (mirrors old runs tree).
    """
    if layout_style == "legacy":
        layout = inference_output_layout_root(data_root, repo_root=repo_root)
        rel = model_rel_path_from_checkpoint(checkpoint, repo_root=repo_root)
        return (layout / rel / split).resolve()
    if layout_style != "method":
        raise ValueError(f"Unknown layout_style: {layout_style!r}")
    del split  # method layout uses flat folder; split is recorded in evaluation_meta.json
    return method_test_output_dir_from_checkpoint(
        checkpoint, data_root, method_slug=method_slug, repo_root=repo_root
    )

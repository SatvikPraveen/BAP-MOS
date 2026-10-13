"""Shared stratified site-loop export for method / baseline ``run_test_inference`` CLIs.

Pooled prostate: one folder per ``site_tests/<site>/`` under ``output_dir`` plus
``sites.json``. Non-pooled: single bundle under ``output_dir``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from torch.utils.data import DataLoader

from bapmos.inference_output.export_bundle import export_stratified_test_bundle
from bapmos.method.data_adapter import iter_test_split_datasets
from bapmos.multiorgan.dataset_multi_organ import multi_organ_collate_fn
from bapmos.paths import is_pooled_prostate_training_root

PredictFn = Callable[..., Any]
MetaExtraFn = Callable[[str], Mapping[str, Any]]


def resolve_splits_subdir(
    cfg_splits: Any,
    cli_splits: Optional[str],
    data_root: str | Path,
) -> str:
    """Prefer config, then CLI, then taxonomy default. Treat JSON ``null`` as missing."""
    from bapmos.train.training_taxonomy import default_splits_subdir

    for candidate in (cfg_splits, cli_splits):
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text and text.lower() != "none":
            return text
    return default_splits_subdir(data_root)


def resolve_torch_device(requested: str = "cuda"):
    """Map ``cuda`` / ``cuda:N`` to CPU when CUDA is unavailable."""
    import torch

    req = (requested or "cuda").strip()
    if req.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(req)


def load_run_config(run_dir: Path, checkpoint: Path) -> dict:
    """Load ``config.json`` beside the checkpoint, else the checkpoint ``config`` blob."""
    import torch

    cfg_path = run_dir / "config.json"
    if cfg_path.is_file():
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt.get("config")
    if not isinstance(cfg, dict):
        raise ValueError(f"No config in {checkpoint}")
    return cfg


def write_sites_json(
    output_dir: Path,
    site_labels: list[str],
    *,
    run_name: Optional[str] = None,
) -> None:
    payload: dict[str, Any] = {"sites": list(site_labels)}
    if run_name is not None:
        payload["run_name"] = run_name
    (Path(output_dir) / "sites.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def export_stratified_sites(
    *,
    data_root: str | Path,
    output_dir: Path,
    splits_subdir: str,
    predict_fn: PredictFn,
    taxonomy: Any,
    distance_unit: str,
    checkpoint: str | Path,
    method_slug: str,
    viz_kwargs: Mapping[str, Any],
    evaluation_meta_extra: Optional[MetaExtraFn] = None,
    run_name: Optional[str] = None,
) -> list[str]:
    """Run ``export_stratified_test_bundle`` once per test site; return site labels.

    For pooled prostate, ``splits_subdir`` of the form ``site_tests/<site>`` exports
    only that site (needed after a per-site ``nnUNetv2_predict``).
    """
    from bapmos.multiorgan.dataset_multi_organ import MultiOrganDataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pooled = is_pooled_prostate_training_root(data_root)
    site_labels: list[str] = []

    splits_text = str(splits_subdir or "")
    if pooled and splits_text.startswith("site_tests/"):
        site = splits_text.split("/", 1)[1].strip()
        if not site:
            raise ValueError(f"Invalid site_tests splits_subdir: {splits_subdir!r}")
        site_iter = [
            (
                site,
                MultiOrganDataset(
                    str(data_root),
                    split="test",
                    splits_subdir=f"site_tests/{site}",
                ),
            )
        ]
    else:
        site_iter = list(iter_test_split_datasets(data_root, splits_subdir=splits_subdir))

    for label, test_ds in site_iter:
        site_labels.append(label)
        site_out = output_dir / label if pooled else output_dir
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=multi_organ_collate_fn,
        )
        extra = dict(evaluation_meta_extra(label)) if evaluation_meta_extra else {}
        extra.setdefault("test_site", label)
        if run_name is not None:
            extra.setdefault("run_name", run_name)
        export_stratified_test_bundle(
            test_loader=test_loader,
            predict_fn=predict_fn,
            output_dir=site_out,
            taxonomy=taxonomy,
            distance_unit=distance_unit,
            checkpoint=checkpoint,
            data_root=data_root,
            method_slug=method_slug,
            evaluation_meta_extra=extra,
            **dict(viz_kwargs),
        )

    if pooled:
        write_sites_json(output_dir, site_labels, run_name=run_name)
    return site_labels

"""
Thin dataset adapter: optional runtime import of legacy MultiOrganDataset.

Does not modify ``legacy/``; relative paths resolve under the BAPMOS checkout root
(``bapmos.paths.project_root()``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from bapmos.paths import project_root


def repo_root() -> Path:
    """BAPMOS checkout root (contains ``src/``, ``data/``, ``configs/``, ``experiments/``).

    Alias of :func:`bapmos.paths.project_root` for method/HPO call sites.
    """
    return project_root()


def resolve_path(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (repo_root() / p).resolve()


def load_yaml_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = repo_root() / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def simulation_organ_to_class() -> Dict[str, int]:
    return {"rectum": 1, "bladder": 2, "ptv1": 3}


def clinical_organ_to_class() -> Dict[str, int]:
    return {"bladder": 1, "ptv": 2, "rectum": 3, "urethra": 4}


def infer_organ_to_class(data_root: Path, organ_keys: List[str]) -> Dict[str, int]:
    """Pick taxonomy from data_root path and organ_keys."""
    from bapmos.train.training_taxonomy import get_baseline_taxonomy_profile

    try:
        tax = get_baseline_taxonomy_profile(str(data_root))
        m = dict(tax.organ_to_class)
        return {k: m[k] for k in organ_keys if k in m}
    except ValueError:
        pass

    parts = {p.lower() for p in data_root.parts}
    if "simulation" in parts or "ptv1" in organ_keys:
        m = simulation_organ_to_class()
    else:
        m = clinical_organ_to_class()
    return {k: m[k] for k in organ_keys if k in m}


def build_dataloaders(
    data_root: str,
    splits_subdir: str = "splits_stratified",
    batch_size: int = 1,
    num_workers: int = 0,
    organ_keys: Optional[List[str]] = None,
    *,
    include_test: bool = True,
    seed: Optional[int] = None,
    image_embedding_dir: Optional[str] = None,
):
    """
    Build train/val/(test) DataLoaders via MultiOrganDataset (runtime import only).

    ``organ_keys`` is used only to resolve ``organ_to_class`` / ``num_classes`` for the
    trainer; the dataset itself loads all classes present in the mask files.
    When omitted or empty, keys come from :func:`get_baseline_taxonomy_profile`.

    When ``seed`` is set, the training loader uses a seeded ``torch.Generator`` for
    shuffle order. Exact multi-worker reproducibility may still need a
    ``worker_init_fn`` if ``__getitem__`` draws additional randomness.

    When ``image_embedding_dir`` is set (PFUS1 family), each sample includes
    ``embedding_path`` / ``sam_cached_pack`` for decoder-only training.

    For pooled prostate, the test loader concatenates ``site_tests/<site>/test.txt``
    (there is no global ``splits_stratified/test.txt``).

    Returns ``val_dataset`` for bandit probe indexing.
    """
    import torch
    from torch.utils.data import DataLoader

    from bapmos.multiorgan.dataset_multi_organ import MultiOrganDataset, multi_organ_collate_fn
    from bapmos.train.training_taxonomy import get_baseline_taxonomy_profile

    root = resolve_path(data_root)
    if not organ_keys:
        try:
            organ_keys = list(get_baseline_taxonomy_profile(str(root)).organ_keys)
        except ValueError:
            organ_keys = list(
                infer_organ_to_class(root, ["rectum", "bladder", "ptv1"]).keys()
            ) or ["rectum", "bladder", "ptv1"]

    ds_kw: Dict[str, Any] = {"splits_subdir": splits_subdir}
    if image_embedding_dir:
        emb = resolve_path(image_embedding_dir)
        ds_kw["image_embedding_dir"] = str(emb)

    train_ds = MultiOrganDataset(
        str(root),
        split="train",
        **ds_kw,
    )
    val_ds = MultiOrganDataset(
        str(root),
        split="val",
        **ds_kw,
    )

    train_kw: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "collate_fn": multi_organ_collate_fn,
        "pin_memory": True,
    }
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(int(seed))
        train_kw["generator"] = g

    train_loader = DataLoader(train_ds, **train_kw)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=multi_organ_collate_fn,
        pin_memory=True,
    )
    test_loader = None
    if include_test:
        test_loader = build_test_dataloader(
            str(root),
            splits_subdir=splits_subdir,
            batch_size=batch_size,
            num_workers=num_workers,
            image_embedding_dir=image_embedding_dir,
        )

    organ_to_class = infer_organ_to_class(root, organ_keys)
    try:
        from bapmos.train.training_taxonomy import get_baseline_taxonomy_profile

        num_classes = int(get_baseline_taxonomy_profile(str(root)).num_classes)
    except ValueError:
        num_classes = max(organ_to_class.values()) + 1
    return train_loader, val_loader, test_loader, val_ds, organ_to_class, num_classes


def iter_test_split_datasets(
    data_root: str,
    *,
    splits_subdir: str = "splits_stratified",
):
    """
    Yield ``(label, MultiOrganDataset)`` for held-out test export.

    - Pooled prostate: one dataset per ``site_tests/<site>/`` (label = site name).
    - Otherwise: a single dataset from ``<splits_subdir>/test.txt`` (label = ``"test"``).
    """
    from bapmos.multiorgan.dataset_multi_organ import MultiOrganDataset
    from bapmos.paths import is_pooled_prostate_training_root, list_pooled_site_test_names

    root = resolve_path(data_root)
    if is_pooled_prostate_training_root(root):
        sites = list_pooled_site_test_names(root)
        if not sites:
            raise FileNotFoundError(
                f"Pooled prostate has no site_tests/*/test.txt under {root}. "
                "Rebuild with bapmos.preprocess.prostate.build_pooled."
            )
        for site in sites:
            yield site, MultiOrganDataset(
                str(root),
                split="test",
                splits_subdir=f"site_tests/{site}",
            )
        return

    yield "test", MultiOrganDataset(
        str(root),
        split="test",
        splits_subdir=splits_subdir,
    )


def build_test_dataloader(
    data_root: str,
    *,
    splits_subdir: str = "splits_stratified",
    batch_size: int = 1,
    num_workers: int = 0,
    image_embedding_dir: Optional[str] = None,
):
    """Test DataLoader; concatenates pooled ``site_tests`` when applicable."""
    from torch.utils.data import ConcatDataset, DataLoader

    from bapmos.multiorgan.dataset_multi_organ import MultiOrganDataset, multi_organ_collate_fn

    if image_embedding_dir:
        emb = str(resolve_path(image_embedding_dir))
        root = resolve_path(data_root)
        datasets = []
        for label, _base_ds in iter_test_split_datasets(data_root, splits_subdir=splits_subdir):
            # Rebuild with embedding dir (iter yields datasets without cache paths).
            if label == "test":
                datasets.append(
                    MultiOrganDataset(
                        str(root),
                        split="test",
                        splits_subdir=splits_subdir,
                        image_embedding_dir=emb,
                    )
                )
            else:
                datasets.append(
                    MultiOrganDataset(
                        str(root),
                        split="test",
                        splits_subdir=f"site_tests/{label}",
                        image_embedding_dir=emb,
                    )
                )
    else:
        datasets = [ds for _label, ds in iter_test_split_datasets(data_root, splits_subdir=splits_subdir)]
    if not datasets:
        raise FileNotFoundError(f"No test split datasets under {data_root!r}")
    test_ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=multi_organ_collate_fn,
        pin_memory=True,
    )

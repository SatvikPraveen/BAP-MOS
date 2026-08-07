"""Filesystem layout for collated test metrics (prostate / bladder)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Optional

from bapmos.paths import project_root

Corpus = Literal["prostate", "bladder"]

# Stable method ids for prostate ladder + baselines (folder names under by_seed/).
PROSTATE_METHOD_IDS: tuple[str, ...] = (
    "box",
    "point",
    "box_point",
    "boxpoint_box_point",
    "ucb1_global",
    "ucb1_per_organ",
    "epsilon_greedy_per_organ",
    "epsilon_decay_per_organ",
    "bapmos",
    "bapmos_medsam",
    "bapmos_random",
    "bapmos_heuristic",
    "bapmos_greedy",
    "unet",
    "nnunet",
    "medsam",
)

BLADDER_METHOD_IDS: tuple[str, ...] = (
    "box",
    "point",
    "bapmos_sam",
    "bapmos_medsam",
    "unet",
    "nnunet",
    "medsam",
)

# Default dataset slug under each corpus.
DEFAULT_DATASET: dict[str, str] = {
    "prostate": "pooled",
    "bladder": "pfus1",
}

# Point estimates taken from each seed's summary_metrics.csv (slice-level mean).
# Paper tables use Dice / MSD / HD95 only (IoU is computed at export but not collated).
SEED_METRIC_COLUMNS: tuple[str, ...] = (
    "dice",
    "msd_mm",
    "hd95_mm",
)

# Prostate pooled corpus: per-site inference for clarity, then slice-weighted pool.
PROSTATE_TEST_SITES: tuple[str, ...] = ("simulation", "case1", "case2")
POOLED_SITE_LABEL = "pooled"

# Map summary CSV columns → normalized seed-row fields.
SUMMARY_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "dice": ("dice", "dice_mean"),
    "msd_mm": ("msd_mm", "msd_mm_mean"),
    "hd95_mm": ("hd95_mm", "hd95_mm_mean"),
}


def results_root(repo: Optional[Path] = None) -> Path:
    return (repo or project_root()) / "results"


def corpus_results_root(
    corpus: Corpus,
    dataset: Optional[str] = None,
    *,
    repo: Optional[Path] = None,
) -> Path:
    ds = dataset or DEFAULT_DATASET[corpus]
    return results_root(repo) / corpus / ds


def method_by_seed_dir(
    corpus: Corpus,
    method: str,
    *,
    dataset: Optional[str] = None,
    repo: Optional[Path] = None,
) -> Path:
    return corpus_results_root(corpus, dataset, repo=repo) / "by_seed" / _slug(method)


def method_combined_dir(
    corpus: Corpus,
    *,
    dataset: Optional[str] = None,
    repo: Optional[Path] = None,
) -> Path:
    return corpus_results_root(corpus, dataset, repo=repo) / "combined"


def by_seed_csv_path(
    corpus: Corpus,
    method: str,
    run_name: str,
    *,
    dataset: Optional[str] = None,
    repo: Optional[Path] = None,
) -> Path:
    return method_by_seed_dir(corpus, method, dataset=dataset, repo=repo) / f"{_slug(run_name)}.csv"


def known_methods(corpus: Corpus) -> tuple[str, ...]:
    return PROSTATE_METHOD_IDS if corpus == "prostate" else BLADDER_METHOD_IDS


def list_method_dirs(corpus: Corpus, *, dataset: Optional[str] = None, repo: Optional[Path] = None) -> list[Path]:
    root = corpus_results_root(corpus, dataset, repo=repo) / "by_seed"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def _slug(name: str) -> str:
    s = str(name).strip().replace("-", "_").replace(" ", "_")
    if not s:
        raise ValueError("empty method/run name")
    return s


def infer_seed_from_run_name(run_name: str) -> Optional[int]:
    """``pooled_seed43_rep2`` → 43; ``pfus1_seed42`` → 42."""
    import re

    m = re.search(r"seed(\d+)", str(run_name))
    return int(m.group(1)) if m else None


def experiment_key_from_run_name(run_name: str) -> str:
    """Strip replicate suffix so seeds group to one experiment (``box50_point50``)."""
    import re

    key = re.sub(r"_seed\d+(?:_rep\d+)?$", "", str(run_name).strip())
    return key or str(run_name).strip()

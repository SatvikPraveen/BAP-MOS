"""Apply independent training seeds from Slurm replicate env.

Environment prefixes:

- ``PI_*`` — prostate (pooled) jobs
- ``BPI_*`` — bladder (PFUS1) jobs

Whichever prefix is set wins (first non-empty among the pair for each seed role).
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and Torch RNGs (non-deterministic cudnn by default)."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def training_seed_from_env(*, default: int = 42) -> int:
    for key in ("PI_TRAIN_SEED", "BPI_TRAIN_SEED"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return int(raw)
    return int(default)


def probe_seed_from_env(*, default: int) -> int:
    for key in ("PI_PROBE_SEED", "BPI_PROBE_SEED"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return int(raw)
    return int(default)


def kfold_seed_from_env(*, default: int) -> int:
    for key in ("PI_CHECKPOINT_KFOLD_SEED", "BPI_CHECKPOINT_KFOLD_SEED"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return int(raw)
    return int(default)


def apply_training_seed_env(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Apply training-seed env overrides without moving fixed protocol seeds.

    - ``experiment_seed`` / training seed may change across replicates (42/43/44).
    - ``bandit.probe_seed`` and ``evaluation.kfold_seed`` stay at the YAML value
      (protocol default **42**) unless ``PI_PROBE_SEED`` / ``BPI_PROBE_SEED`` or
      ``PI_CHECKPOINT_KFOLD_SEED`` / ``BPI_CHECKPOINT_KFOLD_SEED`` is set.

    Returns a shallow copy of ``cfg`` with fresh ``bandit`` / ``evaluation`` dicts
    so nested seed fields do not mutate the caller's originals.
    """
    out = dict(cfg)
    base_default = int(out.get("experiment_seed", 42))
    seed = training_seed_from_env(default=base_default)
    out["experiment_seed"] = seed

    out.setdefault("bandit", {})
    out["bandit"] = dict(out["bandit"])
    probe_default = int(out["bandit"].get("probe_seed", 42))
    out["bandit"]["probe_seed"] = probe_seed_from_env(default=probe_default)

    out.setdefault("evaluation", {})
    out["evaluation"] = dict(out["evaluation"])
    kfold_default = int(out["evaluation"].get("kfold_seed", 42))
    out["evaluation"]["kfold_seed"] = kfold_seed_from_env(default=kfold_default)
    return out

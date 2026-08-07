"""Naming helpers for training replicates (isolated run dirs + distinct training seeds).

Convention (seed-true run names)::

    primary  seed 42 → ``pooled_seed42``
    rep2     seed 43 → ``pooled_seed43_rep2``
    rep3     seed 44 → ``pooled_seed44_rep3``

Same pattern for bladder: ``pfus1_seed42`` / ``pfus1_seed43_rep2`` / ``pfus1_seed44_rep3``.

See ``docs/SEEDS.md``.
"""

from __future__ import annotations

import re

REPLICATE_INDICES: tuple[int, ...] = (2, 3)

_SEED_SUFFIX = re.compile(r"^(?P<prefix>.*seed)(?P<seed>\d+)$")


def replicate_training_seed(primary_seed: int, replicate: int) -> int:
    """Independent seed for replicate N: primary+1 (rep2), primary+2 (rep3), etc."""
    if replicate not in REPLICATE_INDICES:
        raise ValueError(
            f"replicate must be one of {REPLICATE_INDICES}, got {replicate!r}"
        )
    return int(primary_seed) + int(replicate) - 1


def _parse_primary_seed(primary_run_name: str, primary_seed: int | None) -> int:
    if primary_seed is not None:
        return int(primary_seed)
    m = _SEED_SUFFIX.match(str(primary_run_name).strip())
    if m:
        return int(m.group("seed"))
    raise ValueError(
        f"Cannot infer primary seed from run name {primary_run_name!r}; "
        "pass primary_seed= explicitly (expected names like 'pooled_seed42')."
    )


def replicate_index(run_name: str, primary_run_name: str, *, primary_seed: int | None = None) -> int | None:
    for idx in REPLICATE_INDICES:
        if run_name == replicate_run_name(
            primary_run_name, idx, primary_seed=primary_seed
        ):
            return idx
    return None


def replicate_run_name(
    primary_run_name: str,
    replicate: int,
    *,
    primary_seed: int | None = None,
) -> str:
    """
    Return seed-true replicate run name.

    Example: ``pooled_seed42`` + rep 2 → ``pooled_seed43_rep2``
    (training seed 43, replicate index 2).
    """
    if replicate not in REPLICATE_INDICES:
        raise ValueError(
            f"replicate must be one of {REPLICATE_INDICES}, got {replicate!r}"
        )
    base = str(primary_run_name).strip()
    if not base:
        raise ValueError("primary_run_name must be non-empty")
    for idx in REPLICATE_INDICES:
        suffix = f"_rep{idx}"
        if base.endswith(suffix):
            raise ValueError(
                f"primary_run_name {primary_run_name!r} already looks like a replicate"
            )

    seed0 = _parse_primary_seed(base, primary_seed)
    train_seed = replicate_training_seed(seed0, replicate)
    m = _SEED_SUFFIX.match(base)
    if not m:
        raise ValueError(
            f"primary_run_name {primary_run_name!r} must end with 'seed<digits>' "
            f"(e.g. 'pooled_seed42')"
        )
    # Ensure the embedded seed matches primary_seed when both are known
    embedded = int(m.group("seed"))
    if primary_seed is not None and embedded != int(primary_seed):
        raise ValueError(
            f"primary_run_name {primary_run_name!r} embeds seed {embedded}, "
            f"but primary_seed={primary_seed} was passed"
        )
    new_base = f"{m.group('prefix')}{train_seed}"
    return f"{new_base}_rep{replicate}"


def is_replicate_run_name(
    run_name: str,
    primary_run_name: str,
    *,
    primary_seed: int | None = None,
) -> bool:
    return run_name in {
        replicate_run_name(primary_run_name, idx, primary_seed=primary_seed)
        for idx in REPLICATE_INDICES
    }


def all_study_run_names(
    primary_run_name: str,
    *,
    primary_seed: int | None = None,
) -> tuple[str, str, str]:
    """Primary + rep2 + rep3 run names (e.g. seed42, seed43_rep2, seed44_rep3)."""
    return (
        primary_run_name,
        replicate_run_name(primary_run_name, 2, primary_seed=primary_seed),
        replicate_run_name(primary_run_name, 3, primary_seed=primary_seed),
    )

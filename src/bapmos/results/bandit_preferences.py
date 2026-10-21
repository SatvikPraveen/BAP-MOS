"""Per-organ learned prompt preferences from BAP-MOS checkpoints.

Reads ``bandit_state`` from each run's ``best_checkpoint.pth`` (the state used for
test-time prompting) and writes one row per (run, organ, arm) with the pull count N_a,
the windowed mean reward Q_a, the selection share and whether the arm is the test-time
choice (highest Q_a). A second table gives mean ± std over runs per (organ, arm).

    python -m bapmos.results.bandit_preferences \\
      --runs runs/prostate/bapmos/inner_loop/tpe/pooled_seed42 ... \\
      --out results/camera_ready/bandit_preferences

Organs whose arms all have N_a = 0 never received a probe reward; they are flagged
``never_rewarded`` because their test-time arm falls back to the first arm (box).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch


def preference_rows(run_dir: Path) -> List[Dict[str, Any]]:
    """One row per (organ, arm) from ``run_dir/best_checkpoint.pth``."""
    ckpt_path = Path(run_dir) / "best_checkpoint.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("bandit_state") or {}
    scope = state.get("scope", "per_organ")
    if scope == "fixed":
        raise ValueError(f"{ckpt_path}: fixed schedules have no learned preferences")
    rows: List[Dict[str, Any]] = []
    for organ, blob in (state.get("organs") or {}).items():
        n_a = {a: int(n) for a, n in blob["N_a"].items()}
        q_a = {a: float(q) for a, q in blob["Q_a"].items()}
        total = sum(n_a.values())
        best = max(q_a, key=lambda a: q_a[a])  # same rule as get_best_arms_per_organ
        for arm in q_a:
            rows.append(
                {
                    "run": Path(run_dir).name,
                    "scope": scope,
                    "epoch_index": int(ckpt.get("epoch_index", -1)),
                    "organ": organ,
                    "arm": arm,
                    "N_a": n_a[arm],
                    "Q_a": q_a[arm],
                    "share": n_a[arm] / total if total else 0.0,
                    "test_arm": arm == best,
                    "never_rewarded": total == 0,
                }
            )
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std over runs of N_a, Q_a and share, plus how many runs test with the arm."""
    g = df.groupby(["organ", "arm"], sort=False)
    out = g.agg(
        runs=("run", "nunique"),
        N_a_mean=("N_a", "mean"),
        N_a_std=("N_a", "std"),
        Q_a_mean=("Q_a", "mean"),
        Q_a_std=("Q_a", "std"),
        share_mean=("share", "mean"),
        share_std=("share", "std"),
        test_arm_runs=("test_arm", "sum"),
    )
    return out.reset_index()


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", nargs="+", required=True, help="Run dirs with best_checkpoint.pth")
    p.add_argument("--out", required=True, help="Output prefix (writes _per_run.csv, _summary.csv)")
    args = p.parse_args(argv)
    rows = [r for run in args.runs for r in preference_rows(Path(run))]
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{out}_per_run.csv", index=False)
    summary = summarize(df)
    summary.to_csv(f"{out}_summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

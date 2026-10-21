"""Paired per-slice comparison of two methods on the same test slices.

Each method is given as one or more export dirs (one per seed). Every export holds
``metrics/per_slice_metrics.csv`` directly or under ``<site>/metrics/`` (pooled
prostate). Per (image_id, organ) the metric is averaged over the method's seeds, the two
methods are paired on the slices both scored, and a two-sided Wilcoxon signed-rank test
is run per organ and over all organs, with Holm correction across the organ tests.

    python -m bapmos.results.paired_stats \\
      --a inference_output/camera_ready/reference_seed42 ... --a-name BAP-MOS \\
      --b inference_output/camera_ready/box_only_seed42 ... --b-name Box-only \\
      --out results/camera_ready/paired_bapmos_vs_box_only.csv

Slices from one cohort come from one volume and are correlated, so p-values are
optimistic; report them with that caveat. On PFUS1, ``--unit patient`` first averages each
patient's frames (image ids ``P<id>_frame_<n>``), so the test pairs independent patients. Slice-organ pairs where either method has no
valid boundary (e.g. an empty prediction) are not paired; their counts are reported as
``a_invalid`` / ``b_invalid`` so missed organs stay visible.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = ("dice", "hd95_mm", "msd_mm")
_PATIENT_RE = re.compile(r"^(P\d+)_frame_")


def patient_of(image_id: str) -> str:
    """PFUS1 patient id from ``P<id>_frame_<n>`` image ids (raises for other ids)."""
    m = _PATIENT_RE.match(Path(str(image_id)).stem)
    if not m:
        raise ValueError(f"No patient id in image id {image_id!r}; use --unit slice")
    return m.group(1)


def to_patient_level(df: "pd.DataFrame") -> "pd.DataFrame":
    """Average seed-mean slice metrics per (site, patient, organ); invalid counts are summed."""
    out = df.copy()
    out["image_id"] = out["image_id"].map(patient_of)
    agg = {m: "mean" for m in METRICS}
    agg["invalid"] = "sum"
    return out.groupby(["site", "image_id", "organ"], as_index=False).agg(agg)


def load_per_slice(export_dir: Path) -> pd.DataFrame:
    """Concatenate every ``per_slice_metrics.csv`` under one export dir (all sites)."""
    files = sorted(Path(export_dir).glob("**/metrics/per_slice_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No metrics/per_slice_metrics.csv under {export_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["site"] = f.parent.parent.name if f.parent.parent != Path(export_dir) else "test"
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df[~df["gt_empty"].astype(bool)]


def seed_mean(export_dirs: Sequence[Path]) -> pd.DataFrame:
    """Per (site, image_id, organ): mean over seeds of valid metrics, plus invalid counts."""
    frames = []
    for i, d in enumerate(export_dirs):
        df = load_per_slice(Path(d))
        df["seed_idx"] = i
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["valid"] = df["valid_boundary"].astype(bool)
    keys = ["site", "image_id", "organ"]
    valid = df[df["valid"]].groupby(keys)[list(METRICS)].mean()
    invalid = df.groupby(keys)["valid"].apply(lambda v: int((~v).sum())).rename("invalid")
    return valid.join(invalid, how="outer").reset_index()


def holm(pvals: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values (NaNs are left as NaN)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    idx = [i for i in np.argsort(p) if np.isfinite(p[i])]
    m = len(idx)
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()


def paired_table(a: pd.DataFrame, b: pd.DataFrame, a_name: str, b_name: str) -> pd.DataFrame:
    keys = ["site", "image_id", "organ"]
    m = a.merge(b, on=keys, suffixes=("_a", "_b"), how="outer")
    rows: List[Dict[str, object]] = []
    groups = [(organ, g) for organ, g in m.groupby("organ")] + [("ALL", m)]
    for organ, g in groups:
        for metric in METRICS:
            pair = g[[f"{metric}_a", f"{metric}_b"]].dropna()
            diff = pair[f"{metric}_a"] - pair[f"{metric}_b"]
            p = float("nan")
            if len(diff) >= 6 and np.any(diff != 0):
                p = float(wilcoxon(pair[f"{metric}_a"], pair[f"{metric}_b"]).pvalue)
            rows.append(
                {
                    "organ": organ,
                    "metric": metric,
                    "n_pairs": int(len(pair)),
                    f"{a_name}_mean": float(pair[f"{metric}_a"].mean()) if len(pair) else np.nan,
                    f"{b_name}_mean": float(pair[f"{metric}_b"].mean()) if len(pair) else np.nan,
                    "median_diff_a_minus_b": float(diff.median()) if len(diff) else np.nan,
                    "a_better_frac": float(
                        ((diff > 0) if metric == "dice" else (diff < 0)).mean()
                    )
                    if len(diff)
                    else np.nan,
                    "p_wilcoxon": p,
                    "a_invalid": int(g["invalid_a"].fillna(0).sum()),
                    "b_invalid": int(g["invalid_b"].fillna(0).sum()),
                }
            )
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for metric in METRICS:
        sel = (out["metric"] == metric) & (out["organ"] != "ALL")
        out.loc[sel, "p_holm"] = holm(out.loc[sel, "p_wilcoxon"].tolist())
    return out


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--a", nargs="+", required=True, help="Export dirs of method A (one per seed)")
    p.add_argument("--b", nargs="+", required=True, help="Export dirs of method B (one per seed)")
    p.add_argument("--a-name", default="A")
    p.add_argument("--b-name", default="B")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--unit",
        choices=("slice", "patient"),
        default="slice",
        help="Pairing unit: slices (TRUS) or patients (PFUS1, P<id>_frame_<n> ids)",
    )
    args = p.parse_args(argv)
    a = seed_mean([Path(d) for d in args.a])
    b = seed_mean([Path(d) for d in args.b])
    if args.unit == "patient":
        a, b = to_patient_level(a), to_patient_level(b)
    table = paired_table(a, b, args.a_name, args.b_name)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

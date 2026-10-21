"""Seed aggregation shared by the workbook builder and its checks."""
import json
import re
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = ("seed42", "seed43_rep2", "seed44_rep3")
SEED_NO = {"seed42": 42, "seed43_rep2": 43, "seed44_rep3": 44}
METRICS = ("dice", "hd95_mm", "msd_mm")


def load():
    return json.load(open(HERE / "collected.json"))


def agg(entry, organ, metric):
    """(mean, sd, [per-seed values], n_seeds) with sample SD (ddof=1); None if any seed lacks it."""
    vals = []
    for s in SEEDS:
        if s not in entry:
            continue
        m = entry[s]["metrics"].get(organ)
        if m is None:
            return None
        vals.append(m[metric])
    if not vals:
        return None
    mean = st.fmean(vals)
    sd = st.stdev(vals) if len(vals) >= 2 else None
    return mean, sd, vals, len(vals)


def old_method_id(path):
    """Method id used by bapmos.results.camera_ready_tables for an export path."""
    group, name = path.split("camera_ready/")[1].split("/")
    stem = re.sub(r"_(seed42|seed43_rep2|seed44_rep3)$", "", name)
    return stem if group in ("unet", "medsam", "box", "nnunet") else f"{group}_{stem}"

"""Collect every camera-ready result as aggregate numbers (run on GAIVI from the repo root).

Per export (one seed): per-organ and all-organ means of Dice / HD95 / MSD over the
slice-organ rows with a valid boundary, pooled across sites (the repository's
collation rule). Two independent checks per export: the site summary files, and the
seed-row CSVs of the repository's earlier collation (compared locally). Paired tests use
bapmos.results.paired_stats unchanged. Output: JSON on stdout, no image ids.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bapmos.results.paired_stats import paired_table, seed_mean, to_patient_level

IO = Path("inference_output/camera_ready")
SEEDS = ("seed42", "seed43_rep2", "seed44_rep3")
METRICS = ("dice", "hd95_mm", "msd_mm")
COND = [f"{m}{n}" for m in ("pshift", "ploose") for n in (5, 10, 20)]

# (dataset, block, condition, model id) -> export dir template ({s} = seed suffix)
CAT = {}


def add(ds, block, cond, model, tmpl):
    CAT[(ds, block, cond, model)] = tmpl


# TRUS, exact prompts
for model, tmpl in {
    "nnunet": "nnunet/nnunet_pooled_{s}", "unet": "unet/unet_pooled_{s}",
    "sam_box": "table1/multi_box_decoder_{s}", "medsam_box": "medsam/medsam_pooled_{s}",
    "box_only": "suite/box_only_{s}", "reference": "suite/reference_{s}",
    "medsam_reference": "suite/medsam_reference_{s}",
    "reference_evaltrain": "suite/reference_evaltrain_{s}",
    "global": "suite/global_{s}", "dice_reward": "suite/dice_reward_{s}",
    "fixed_ring": "suite/fixed_ring_{s}", "no_boundary_loss": "suite/no_boundary_loss_{s}",
    "probe_disjoint": "suite/probe_disjoint_{s}", "fixed_235": "suite/fixed_235_{s}",
    "box_only_j10": "suite/box_only_j10_{s}", "box_only_j20": "suite/box_only_j20_{s}",
    "fixed_111": "suite/fixed_111_{s}", "fixed_511": "suite/fixed_511_{s}", "fixed_151": "suite/fixed_151_{s}",
}.items():
    add("TRUS", "exact", "exact", model, tmpl)
for c in COND:
    for model in ("reference", "box_only", "sam_box", "medsam_box", "medsam_reference",
                  "fixed_235", "global", "box_only_j10", "box_only_j20",
                  "fixed_111", "fixed_511", "fixed_151"):
        add("TRUS", "robust", c, model, f"{c}/{model}_{{s}}")
for model in ("box_only", "reference", "reference_evaltrain", "sam_box", "medsam_box", "medsam_reference"):
    add("TRUS", "auto_unet", "unet_boxes", model, f"auto_unet/{model}_{{s}}")
for k in (1, 2):
    add("TRUS", "loco", f"case{k}", "reference", f"suite/loco_case{k}_reference_{{s}}")
    add("TRUS", "loco", f"case{k}", "box_only", f"suite/loco_case{k}_box_only_{{s}}")
    add("TRUS", "loco", f"case{k}", "unet", f"unet/unet_loco_case{k}_{{s}}")
    add("TRUS", "loco", f"case{k}", "sam_box", f"table1/multi_box_decoder_loco_case{k}_{{s}}")
    add("TRUS", "loco", f"case{k}", "medsam_box", f"medsam/medsam_loco_case{k}_{{s}}")
    add("TRUS", "loco", f"case{k}", "medsam_reference", f"suite/loco_case{k}_medsam_reference_{{s}}")
for cell in ("single_points_decoder", "single_points_full", "single_box_decoder", "single_box_full",
             "multi_points_decoder", "multi_points_full", "multi_box_decoder", "multi_box_full"):
    add("TRUS", "table1_rerun", "exact", cell, f"table1/{cell}_{{s}}")
for c in ("noise5", "noise10", "noise20", "loose5", "loose10", "loose20"):
    for model in ("reference", "box_only", "reference_evaltrain"):
        add("TRUS", "superseded", c, model, f"{c}/{model}_{{s}}")
# PFUS1
for model, tmpl in {
    "unet": "unet/unet_pfus1_{s}", "nnunet": "nnunet/nnunet_pfus1_{s}", "sam_box": "box/box_pfus1_{s}",
    "medsam_box": "medsam/medsam_pfus1_{s}", "bapmos_sam": "bapmos_pfus1_sam/pfus1_{s}",
    "bapmos_medsam": "bapmos_pfus1_medsam/pfus1_{s}",
}.items():
    add("PFUS1", "exact", "exact", model, tmpl)
for c in COND:
    for model in ("sam_box", "medsam_box", "bapmos_sam", "bapmos_medsam"):
        add("PFUS1", "robust", c, model, f"pf_{c}/{model}_{{s}}")


def finished(d: Path) -> bool:
    return (d / "sites.json").is_file() or (d / "metrics" / "summary_metrics.csv").is_file()


def per_slice(d: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(d.glob("**/metrics/per_slice_metrics.csv")):
        df = pd.read_csv(f)
        df["site"] = f.parent.parent.name if f.parent.parent != d else "single"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def seed_metrics(d: Path):
    df = per_slice(d)
    v = df[df["valid_boundary"].astype(str).str.lower() == "true"]
    out = {}
    for organ, g in list(v.groupby("organ")) + [("ALL", v)]:
        out[organ] = {m: float(g[m].mean()) for m in METRICS} | {"n": int(len(g))}
    # Independent check: site summaries, weighted by each site's valid-boundary count.
    chk = {}
    for f in sorted(d.glob("**/metrics/summary_metrics.csv")):
        s = pd.read_csv(f)
        for _, r in s.iterrows():
            o = r["organ"]
            if o not in out or o == "ALL":
                continue
            w = float(r["n_valid_boundary"])
            acc = chk.setdefault(o, {"w": 0.0, "hd95_mm": 0.0, "msd_mm": 0.0})
            if w > 0:
                acc["w"] += w
                acc["hd95_mm"] += w * float(r["hd95_mm_mean"])
                acc["msd_mm"] += w * float(r["msd_mm_mean"])
    diffs = []
    for o, acc in chk.items():
        if acc["w"] > 0:
            for m in ("hd95_mm", "msd_mm"):
                diffs.append(abs(acc[m] / acc["w"] - out[o][m]))
            if int(acc["w"]) != out[o]["n"]:
                diffs.append(float("inf"))
    sites = sorted(df["site"].unique().tolist())
    return out, (max(diffs) if diffs else None), sites


result = {"exports": {}, "missing": [], "paired": {}}
dirs = {}
for key, tmpl in CAT.items():
    got = {}
    for s in SEEDS:
        d = IO / tmpl.format(s=s)
        if finished(d):
            m, dmax, sites = seed_metrics(d)
            got[s] = {"metrics": m, "summary_check_maxdiff": dmax, "sites": sites, "path": str(d)}
        else:
            result["missing"].append(str(d))
    result["exports"]["|".join(key)] = got
    dirs[key] = [IO / tmpl.format(s=s) for s in SEEDS if s in got]


def paired(name, ka, kb, unit="slice"):
    da, db = dirs.get(ka, []), dirs.get(kb, [])
    if len(da) < 3 or len(db) < 3:
        result["paired"][name] = {"status": "pending", "a": "|".join(ka), "b": "|".join(kb)}
        return
    a, b = seed_mean(da), seed_mean(db)
    if unit == "patient":
        a, b = to_patient_level(a), to_patient_level(b)
    t = paired_table(a, b, "A", "B")
    t = t.rename(columns={"A_mean": "A_mean", "B_mean": "B_mean"})
    result["paired"][name] = {"status": "ok", "a": "|".join(ka), "b": "|".join(kb), "unit": unit,
                              "rows": json.loads(t.to_json(orient="records"))}


T = lambda c, m, blk="exact": ("TRUS", blk, c, m)
# TRUS exact
for a, b in [("box_only", "reference"), ("reference", "sam_box"), ("medsam_reference", "medsam_box"),
             ("medsam_box", "sam_box"), ("medsam_reference", "reference"),
             ("reference_evaltrain", "reference"), ("box_only", "reference_evaltrain"),
             ("nnunet", "unet"), ("box_only", "nnunet"), ("reference", "nnunet"),
             ("global", "reference"), ("dice_reward", "reference"), ("fixed_ring", "reference"),
             ("no_boundary_loss", "reference"), ("probe_disjoint", "reference"),
             ("fixed_235", "reference"), ("fixed_235", "box_only"),
             ("box_only_j10", "box_only"), ("box_only_j20", "box_only"),
             ("reference", "box_only_j10"), ("reference", "box_only_j20"),
             ("fixed_111", "reference"), ("fixed_511", "reference"), ("fixed_151", "reference"),
             ("fixed_111", "box_only"), ("fixed_511", "box_only"), ("fixed_151", "box_only")]:
    paired(f"TRUS|exact|{a}_vs_{b}", T("exact", a), T("exact", b))
# TRUS robustness
for c in COND:
    for a, b in [("reference", "box_only"), ("reference", "sam_box"), ("medsam_reference", "medsam_box"),
                 ("fixed_235", "reference"), ("global", "reference"), ("fixed_235", "box_only"),
                 ("global", "box_only"), ("reference", "box_only_j10"), ("reference", "box_only_j20"),
                 ("fixed_235", "box_only_j10"), ("fixed_235", "box_only_j20"),
                 ("box_only_j10", "box_only"), ("box_only_j20", "box_only"),
                 ("fixed_111", "reference"), ("fixed_511", "reference"), ("fixed_151", "reference"),
                 ("fixed_111", "box_only"), ("fixed_511", "box_only"), ("fixed_151", "box_only")]:
        paired(f"TRUS|{c}|{a}_vs_{b}", T(c, a, "robust"), T(c, b, "robust"))
# TRUS automatic prompts and LOCO
for a, b in [("reference", "box_only"), ("reference", "sam_box"), ("medsam_reference", "medsam_box")]:
    paired(f"TRUS|unet_boxes|{a}_vs_{b}", T("unet_boxes", a, "auto_unet"),
           T("unet_boxes", b, "auto_unet"))
for m in ("reference", "box_only"):
    paired(f"TRUS|unet_boxes|{m}_vs_unet", T("unet_boxes", m, "auto_unet"), T("exact", "unet"))
    paired(f"TRUS|unet_boxes|{m}_vs_nnunet", T("unet_boxes", m, "auto_unet"), T("exact", "nnunet"))
for k in (1, 2):
    paired(f"TRUS|loco_case{k}|reference_vs_box_only", T(f"case{k}", "reference", "loco"),
           T(f"case{k}", "box_only", "loco"))
    paired(f"TRUS|loco_case{k}|reference_vs_unet", T(f"case{k}", "reference", "loco"),
           T(f"case{k}", "unet", "loco"))
    paired(f"TRUS|loco_case{k}|box_only_vs_unet", T(f"case{k}", "box_only", "loco"),
           T(f"case{k}", "unet", "loco"))
    for a, b in [("reference", "sam_box"), ("medsam_reference", "medsam_box")]:
        paired(f"TRUS|loco_case{k}|{a}_vs_{b}", T(f"case{k}", a, "loco"), T(f"case{k}", b, "loco"))
# PFUS1, patient level
P = lambda c, m, blk="exact": ("PFUS1", blk, c, m)
for a, b in [("bapmos_sam", "nnunet"), ("bapmos_medsam", "nnunet"), ("nnunet", "unet"),
             ("bapmos_sam", "sam_box"), ("bapmos_medsam", "medsam_box"), ("bapmos_sam", "unet"),
             ("bapmos_medsam", "unet"), ("sam_box", "unet"), ("medsam_box", "unet"),
             ("medsam_box", "sam_box")]:
    paired(f"PFUS1|exact|{a}_vs_{b}", P("exact", a), P("exact", b), unit="patient")
for c in COND:
    for a, b in [("bapmos_sam", "sam_box"), ("bapmos_medsam", "medsam_box")]:
        paired(f"PFUS1|{c}|{a}_vs_{b}", P(c, a, "robust"), P(c, b, "robust"), unit="patient")

# Paired boxes must be identical across every export of a condition.
from bapmos.inference_output.prompt_log import compare_prompt_logs
pc = {}
for ds, root in (("TRUS", ""), ("PFUS1", "pf_")):
    for c in COND:
        logs = sorted((IO / f"{root}{c}").glob("*/prompts_used.json"))
        if len(logs) < 2:
            continue
        res = [compare_prompt_logs(logs[0], f) for f in logs[1:]]
        pc[f"{ds}|{c}"] = {"exports": len(logs), "boxes_per_export": int(res[0][0] + res[0][1]),
                           "mismatches": int(sum(r[1] for r in res))}
result["prompt_check"] = pc
json.dump(result, sys.stdout, default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else o)

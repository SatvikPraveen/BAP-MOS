"""Labels shared by both workbook builders (extracted from build.py lines 20-74)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import agg  # noqa: E402
d = agg.load()
NAMES = {
    "nnunet": "nnU-Net", "unet": "U-Net", "sam_box": "SAM (box)", "medsam_box": "MedSAM (box)",
    "box_only": "Box only (BAP-MOS loss)", "reference": "BAP-MOS (SAM)",
    "medsam_reference": "BAP-MOS (MedSAM)",
    "reference_evaltrain": "BAP-MOS (SAM), train-metrics pass on",
    "global": "Shared global selector", "dice_reward": "Dice-only reward",
    "fixed_ring": "Fixed 10 px ring (r_min)", "no_boundary_loss": "No boundary loss (alpha=1)",
    "probe_disjoint": "Disjoint probe / selection", "fixed_235": "Fixed 2:3:5 (same loss)",
    "fixed_111": "Fixed 1:1:1 (same loss; box at test)", "fixed_511": "Fixed 5:1:1 (same loss; box at test)",
    "fixed_151": "Fixed 1:5:1 (same loss; point at test)",
    "box_only_j10": "Box only, 0-10 px train jitter", "box_only_j20": "Box only, 0-20 px train jitter",
    "bapmos_sam": "BAP-MOS (SAM)", "bapmos_medsam": "BAP-MOS (MedSAM)",
    "single_points_decoder": "Single-organ, points, decoder-only",
    "single_points_full": "Single-organ, points, full fine-tuning",
    "single_box_decoder": "Single-organ, box, decoder-only",
    "single_box_full": "Single-organ, box, full fine-tuning",
    "multi_points_decoder": "Multi-organ, points, decoder-only",
    "multi_points_full": "Multi-organ, points, full fine-tuning",
    "multi_box_decoder": "Multi-organ, box, decoder-only",
    "multi_box_full": "Multi-organ, box, full fine-tuning",
}
COND = {
    "exact": "Exact boxes", "unet_boxes": "Boxes from U-Net masks",
    "case1": "LOCO: clinical case 1 held out", "case2": "LOCO: clinical case 2 held out",
    **{f"pshift{n}": f"Shifted ±{n} px" for n in (5, 10, 20)},
    **{f"ploose{n}": f"Enlarged +{n} px" for n in (5, 10, 20)},
    **{f"noise{n}": f"Shifted ±{n} px, unpaired (superseded)" for n in (5, 10, 20)},
    **{f"loose{n}": f"Enlarged +{n} px, unpaired (superseded)" for n in (5, 10, 20)},
}
ROBUST = ["exact"] + [f"pshift{n}" for n in (5, 10, 20)] + [f"ploose{n}" for n in (5, 10, 20)]
ORGANS = {"TRUS": ["PTV", "Bladder", "Rectum", "Urethra", "ALL"],
          "PFUS1": ["Bladder", "Anus", "Levator ani muscle", "Pubis", "Rectum", "Urethra", "Uterus",
                    "Vagina", "ALL"]}
UNIT = {"TRUS": "mm", "PFUS1": "px"}
MLABEL = {"dice": "Dice", "hd95_mm": "HD95", "msd_mm": "MSD"}


def decimals(ds, metric):
    return 3 if (metric == "dice" or ds == "TRUS") else 2


def fmt(a, ds, metric):
    if a is None or a[3] < 3:
        return None
    k = decimals(ds, metric)
    return f"{a[0]:.{k}f} ± {a[1]:.{k}f}"


def entry(ds, block, cond, model):
    return d["exports"].get(f"{ds}|{block}|{cond}|{model}", {})


def cell(ds, block, cond, model, organ, metric):
    e = entry(ds, block, cond, model)
    a = agg.agg(e, organ, metric) if e else None
    return fmt(a, ds, metric) or ("pending" if not e or len(e) < 3 else "n/a")


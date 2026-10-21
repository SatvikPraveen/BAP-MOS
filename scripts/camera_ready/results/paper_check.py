"""Compare every number the draft reports for finished results with collected.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agg  # noqa: E402

d = agg.load()
res = []


def mean_sd(key, organ, metric, exp_mean, exp_sd=None, dec=3, label=""):
    a = agg.agg(d["exports"][key], organ, metric)
    got_m = round(a[0], dec)
    ok = abs(got_m - exp_mean) < 0.5 * 10 ** -dec + 1e-12
    got = f"{a[0]:.{dec}f}"
    if exp_sd is not None:
        ok &= abs(round(a[1], dec) - exp_sd) < 0.5 * 10 ** -dec + 1e-12
        got += f" ± {a[1]:.{dec}f}"
    exp = f"{exp_mean:.{dec}f}" + (f" ± {exp_sd:.{dec}f}" if exp_sd is not None else "")
    res.append((label or f"{key} {organ} {metric}", exp, got, ok))


def prow(name, organ, metric):
    for r in d["paired"][name]["rows"]:
        if r["organ"] == organ and r["metric"] == metric:
            return r
    raise KeyError((name, organ, metric))


def pval(name, organ, metric, cmp, label):
    """cmp: function of (p_raw, p_holm) -> bool, describing the claim in the text."""
    r = prow(name, organ, metric)
    ok = cmp(r["p_wilcoxon"], r["p_holm"])
    res.append((label, "claim in text", f"p={r['p_wilcoxon']:.3g}, Holm={r['p_holm'] if r['p_holm'] is None else format(r['p_holm'], '.3g')}", ok))


def frac(name, organ, metric, exp, label):
    r = prow(name, organ, metric)
    res.append((label, f"{exp}%", f"{100 * r['a_better_frac']:.1f}%", round(100 * r["a_better_frac"]) == exp))


E = lambda m: f"TRUS|exact|exact|{m}"
# Table III (PTV)
for m, v in {"nnunet": (0.976, .002, .646, .041, .264, .014), "unet": (.966, .004, .923, .100, .358, .037),
             "sam_box": (.980, .002, .536, .041, .210, .015), "medsam_box": (.981, .001, .526, .027, .208, .017),
             "box_only": (.982, .000, .506, .013, .200, .002), "reference": (.980, .001, .556, .044, .214, .011),
             "medsam_reference": (.979, .001, .590, .032, .229, .011)}.items():
    for i, me in enumerate(agg.METRICS):
        mean_sd(E(m), "PTV", me, v[2 * i], v[2 * i + 1], label=f"Table III {m} PTV {me}")
# Table I (PTV; decoder-only vs full fine-tuning, camera-ready rerun)
for m, v in {"single_box_decoder": (.977, .001, .609, .022, .237, .009), "single_points_decoder": (.962, .004, .980, .093, .399, .040),
             "single_box_full": (.976, .003, .679, .118, .241, .022), "single_points_full": (.969, .000, .923, .053, .336, .010),
             "multi_box_decoder": (.980, .002, .536, .041, .210, .015), "multi_points_decoder": (.956, .013, 1.065, .177, .445, .077),
             "multi_box_full": (.980, .001, .550, .027, .223, .010), "multi_points_full": (.962, .019, .932, .254, .355, .092)}.items():
    for i, me in enumerate(agg.METRICS):
        mean_sd(f"TRUS|table1_rerun|exact|{m}", "PTV", me, v[2 * i], v[2 * i + 1], label=f"Table I {m} PTV {me}")
# Table IV (PTV)
for m, v in {"global": (.980, .000, .550, .026, .218, .005), "dice_reward": (.982, .002, .498, .044, .199, .023),
             "fixed_ring": (.979, .001, .584, .027, .229, .010), "no_boundary_loss": (.981, .001, .534, .024, .209, .010),
             "probe_disjoint": (.979, .001, .582, .023, .225, .005), "fixed_235": (.980, .001, .554, .033, .218, .016)}.items():
    for i, me in enumerate(agg.METRICS):
        mean_sd(E(m), "PTV", me, v[2 * i], v[2 * i + 1], label=f"Table IV {m} PTV {me}")
# Table V (HD95 means, PTV / all organs, 2 dp)
cols = ["box_only", "sam_box", "reference", "medsam_box", "medsam_reference"]
TV = {"exact": "0.51/0.52 0.54/0.53 0.56/0.58 0.53/0.53 0.59/0.58",
      "pshift5": "0.79/0.70 0.83/0.71 0.83/0.75 0.79/0.76 0.76/0.69",
      "pshift10": "1.42/1.17 1.47/1.23 1.40/1.08 1.42/1.47 1.26/1.03",
      "pshift20": "2.67/2.14 2.74/2.40 2.59/1.91 2.67/3.15 2.23/1.89",
      "ploose5": "0.72/0.64 0.75/0.63 0.75/0.68 0.79/0.66 0.80/0.67",
      "ploose10": "1.36/1.05 1.38/1.03 1.35/1.03 1.39/1.19 1.25/0.98",
      "ploose20": "2.35/2.13 2.36/2.09 2.27/1.97 2.32/3.15 2.08/2.17"}
for c, s in TV.items():
    for m, pair in zip(cols, s.split()):
        p, a = map(float, pair.split("/"))
        key = E(m) if c == "exact" else f"TRUS|robust|{c}|{m}"
        mean_sd(key, "PTV", "hd95_mm", p, dec=2, label=f"Table V {c} {m} PTV HD95")
        mean_sd(key, "ALL", "hd95_mm", a, dec=2, label=f"Table V {c} {m} all-organ HD95")
for m, (p, a) in {"box_only": (1.22, 1.28), "reference": (1.13, 1.19), "sam_box": (1.25, 1.23),
                  "medsam_box": (1.18, 1.32), "medsam_reference": (1.18, 1.23)}.items():
    mean_sd(f"TRUS|auto_unet|unet_boxes|{m}", "PTV", "hd95_mm", p, dec=2, label=f"Table V U-Net box {m} PTV")
    mean_sd(f"TRUS|auto_unet|unet_boxes|{m}", "ALL", "hd95_mm", a, dec=2, label=f"Table V U-Net box {m} all")
for k, v in {"case1": {"box_only": (1.88, 9.46), "reference": (3.01, 11.42), "unet": (10.39, 12.30),
                       "sam_box": (4.63, 13.27), "medsam_box": (2.63, 7.89), "medsam_reference": (2.75, 9.20)},
             "case2": {"box_only": (3.33, 3.04), "reference": (2.45, 2.95), "unet": (9.49, 13.35),
                       "sam_box": (3.91, 7.36), "medsam_box": (2.55, 2.45), "medsam_reference": (3.42, 3.49)}}.items():
    for m, (p, a) in v.items():
        mean_sd(f"TRUS|loco|{k}|{m}", "PTV", "hd95_mm", p, dec=2, label=f"Table V LOCO {k} {m} PTV")
        mean_sd(f"TRUS|loco|{k}|{m}", "ALL", "hd95_mm", a, dec=2, label=f"Table V LOCO {k} {m} all")
for m, (p, a) in {"unet": (0.92, 1.11), "nnunet": (0.65, 0.59)}.items():
    mean_sd(E(m), "PTV", "hd95_mm", p, dec=2, label=f"Table V note {m} PTV")
    mean_sd(E(m), "ALL", "hd95_mm", a, dec=2, label=f"Table V note {m} all")
# Text numbers
R = lambda c, m: f"TRUS|robust|{c}|{m}"
mean_sd(R("pshift20", "reference"), "Urethra", "hd95_mm", 1.83, dec=2, label="Text ±20 BAP-MOS urethra")
mean_sd(R("pshift20", "box_only"), "Urethra", "hd95_mm", 2.40, dec=2, label="Text ±20 box-only urethra")
mean_sd(R("pshift20", "reference"), "Rectum", "hd95_mm", 1.07, dec=2, label="Text ±20 BAP-MOS rectum")
mean_sd(R("pshift20", "box_only"), "Rectum", "hd95_mm", 1.31, dec=2, label="Text ±20 box-only rectum")
mean_sd(R("pshift20", "medsam_box"), "Rectum", "hd95_mm", 4.05, dec=2, label="Text ±20 MedSAM box rectum")
mean_sd(R("ploose20", "medsam_box"), "Rectum", "hd95_mm", 4.65, dec=2, label="Text +20 MedSAM box rectum")
mean_sd(E("nnunet"), "Rectum", "hd95_mm", 0.28, dec=2, label="Text nnU-Net rectum")
mean_sd(E("nnunet"), "Urethra", "hd95_mm", 0.30, dec=2, label="Text nnU-Net urethra")
mean_sd(E("box_only"), "Rectum", "hd95_mm", 0.45, dec=2, label="Text box-only rectum")
mean_sd(E("box_only"), "Urethra", "hd95_mm", 0.43, dec=2, label="Text box-only urethra")
mean_sd(E("reference_evaltrain"), "PTV", "hd95_mm", 0.551, label="Text eval-train PTV HD95")
mean_sd("PFUS1|exact|exact|unet", "Bladder", "dice", 0.792, 0.011, label="PFUS1 table U-Net Dice")
mean_sd("PFUS1|exact|exact|unet", "Bladder", "hd95_mm", 39.06, 3.36, dec=2, label="PFUS1 table U-Net HD95")
mean_sd("PFUS1|exact|exact|unet", "Bladder", "msd_mm", 14.94, 0.95, dec=2, label="PFUS1 table U-Net MSD")
frac("TRUS|pshift20|reference_vs_box_only", "ALL", "hd95_mm", 82, "Text ±20: BAP-MOS better on 82% of pairs")
frac("TRUS|pshift20|medsam_reference_vs_medsam_box", "PTV", "hd95_mm", 97, "Text ±20 MedSAM: 97% of PTV slices")
frac("TRUS|exact|box_only_vs_reference", "PTV", "hd95_mm", 83, "Text exact: box-only better on 83% of PTV")
# p-values quoted in the text
lt = lambda t: (lambda p, h: p < t)
H = lambda t: (lambda p, h: h is not None and h < t)
approx = lambda v, use="p": (lambda p, h: abs((p if use == "p" else h) - v) < 0.5 * 10 ** -(len(str(v).split(".")[1])))
pval("TRUS|exact|box_only_vs_reference", "PTV", "hd95_mm", lambda p, h: round(h, 4) == 3e-4, "Exact: box-only vs BAP-MOS PTV Holm p=3e-4")
pval("TRUS|exact|reference_vs_sam_box", "PTV", "hd95_mm", approx(0.42), "Exact: BAP-MOS vs SAM (box) PTV p=0.42")
pval("TRUS|exact|medsam_reference_vs_medsam_box", "PTV", "hd95_mm", approx(0.06, "h"), "Exact: MedSAM pair PTV Holm p=0.06")
pval("TRUS|exact|medsam_reference_vs_medsam_box", "ALL", "hd95_mm", lt(1e-3), "Exact: MedSAM pair all organs p<1e-3")
pval("TRUS|exact|reference_evaltrain_vs_reference", "PTV", "hd95_mm", approx(0.35), "Exact: eval-train vs BAP-MOS p=0.35")
pval("TRUS|exact|box_only_vs_reference_evaltrain", "PTV", "hd95_mm", lambda p, h: p < 1e-3 or (h is not None and h < 1e-3), "Exact: eval-train behind box-only p<1e-3")
pval("TRUS|exact|nnunet_vs_unet", "PTV", "hd95_mm", lambda p, h: p < 1e-3 or h < 1e-3, "Exact: nnU-Net vs U-Net PTV p<1e-3")
pval("TRUS|exact|box_only_vs_nnunet", "PTV", "hd95_mm", lambda p, h: round(p, 3) == 0.006 or round(h, 3) == 0.006, "Exact: box-only vs nnU-Net PTV p=0.006")
pval("TRUS|exact|box_only_vs_nnunet", "Rectum", "hd95_mm", H(0.0205), "Exact: nnU-Net rectum Holm p<=0.02")
pval("TRUS|exact|box_only_vs_nnunet", "Urethra", "hd95_mm", H(0.0205), "Exact: nnU-Net urethra Holm p<=0.02")
pval("TRUS|exact|box_only_vs_nnunet", "ALL", "hd95_mm", approx(0.81), "Exact: box-only vs nnU-Net all p=0.81")
pval("TRUS|exact|reference_vs_nnunet", "ALL", "hd95_mm", approx(0.18), "Exact: BAP-MOS vs nnU-Net all p=0.18")
pval("TRUS|exact|global_vs_reference", "PTV", "hd95_mm", approx(0.91), "Ablation: global PTV p=0.91")
pval("TRUS|exact|global_vs_reference", "ALL", "hd95_mm", approx(0.49), "Ablation: global all p=0.49")
pval("TRUS|exact|fixed_235_vs_reference", "PTV", "hd95_mm", approx(0.57, "h"), "Ablation: fixed 2:3:5 PTV Holm p=0.57")
pval("TRUS|exact|dice_reward_vs_reference", "PTV", "hd95_mm", H(1e-3), "Ablation: Dice-only Holm p<1e-3")
pval("TRUS|exact|no_boundary_loss_vs_reference", "PTV", "hd95_mm", approx(0.22, "h"), "Ablation: no boundary loss PTV Holm p=0.22")
pval("TRUS|exact|no_boundary_loss_vs_reference", "ALL", "hd95_mm", approx(0.006), "Ablation: no boundary loss all p=0.006")
pval("TRUS|exact|fixed_ring_vs_reference", "PTV", "hd95_mm", lambda p, h: approx(0.035)(p, h) or approx(0.035, "h")(p, h), "Ablation: fixed ring PTV p=0.035")
pval("TRUS|exact|fixed_ring_vs_reference", "Rectum", "hd95_mm", lambda p, h: p < 1e-3 or h < 1e-3, "Ablation: fixed ring rectum p<1e-3")
pval("TRUS|exact|probe_disjoint_vs_reference", "ALL", "hd95_mm", approx(0.75), "Ablation: disjoint probe all-organ p=0.75")
pval("TRUS|exact|probe_disjoint_vs_reference", "Urethra", "hd95_mm", H(1e-3), "Ablation: disjoint probe urethra Holm p<1e-3")
pval("TRUS|exact|probe_disjoint_vs_reference", "Rectum", "hd95_mm", approx(0.03, "h"), "Ablation: disjoint probe rectum Holm p=0.03")
pval("TRUS|pshift5|reference_vs_box_only", "ALL", "hd95_mm", approx(0.004), "±5: BAP-MOS vs box-only all p=0.004")
pval("TRUS|pshift5|reference_vs_sam_box", "ALL", "hd95_mm", approx(0.003), "±5: BAP-MOS vs SAM (box) all p=0.003")
pval("TRUS|ploose5|reference_vs_box_only", "ALL", "hd95_mm", lt(1e-3), "+5: BAP-MOS worse all p<1e-3 (box-only)")
pval("TRUS|ploose5|reference_vs_sam_box", "ALL", "hd95_mm", lt(1e-3), "+5: BAP-MOS worse all p<1e-3 (SAM box)")
pval("TRUS|pshift10|reference_vs_box_only", "ALL", "hd95_mm", lt(1e-3), "±10: vs box-only all p<1e-3")
pval("TRUS|pshift10|reference_vs_sam_box", "ALL", "hd95_mm", lt(1e-3), "±10: vs SAM (box) all p<1e-3")
pval("TRUS|pshift20|reference_vs_box_only", "PTV", "hd95_mm", approx(0.02, "h"), "±20: PTV vs box-only Holm p=0.02")
pval("TRUS|pshift20|reference_vs_sam_box", "PTV", "hd95_mm", lambda p, h: approx(0.006)(p, h) or approx(0.006, "h")(p, h), "±20: PTV vs SAM (box) p=0.006")
pval("TRUS|ploose10|reference_vs_box_only", "ALL", "hd95_mm", approx(0.045), "+10: vs box-only all p=0.045")
pval("TRUS|ploose20|reference_vs_box_only", "PTV", "hd95_mm", lambda p, h: approx(0.008)(p, h) or approx(0.008, "h")(p, h), "+20: PTV vs box-only p=0.008")
pval("TRUS|ploose20|reference_vs_sam_box", "PTV", "hd95_mm", lambda p, h: approx(0.008)(p, h) or approx(0.008, "h")(p, h), "+20: PTV vs SAM (box) p=0.008")
pval("TRUS|ploose20|reference_vs_box_only", "ALL", "hd95_mm", lt(1e-5), "+20: all vs box-only p<1e-5")
pval("TRUS|ploose20|reference_vs_sam_box", "ALL", "hd95_mm", lt(1e-5), "+20: all vs SAM (box) p<1e-5")
pval("TRUS|pshift20|medsam_reference_vs_medsam_box", "PTV", "hd95_mm", H(1e-5), "±20 MedSAM: PTV Holm p<1e-5")
pval("TRUS|unet_boxes|reference_vs_box_only", "PTV", "hd95_mm", lambda p, h: round(h, 2) == 1.0, "U-Net boxes: Holm p=1.0")
pval("TRUS|unet_boxes|reference_vs_unet", "PTV", "hd95_mm", lambda p, h: approx(0.009)(p, h) or approx(0.009, "h")(p, h), "U-Net boxes: BAP-MOS vs U-Net p=0.009")
# Marks in the U-Net-box / LOCO rows: BAP-MOS better (+) / worse (-) than every box decoder, all pairs.
def mark(names, sign, label):
    rows = [prow(n, "ALL", "hd95_mm") for n in names]
    ok = all(r["p_wilcoxon"] < 0.05 and (r["median_diff_a_minus_b"] < 0) == (sign == "+") for r in rows)
    res.append((label, sign, ", ".join(f"p={r['p_wilcoxon']:.2g} d={r['median_diff_a_minus_b']:+.2f}" for r in rows), ok))
def nomark(names, label):
    rows = [prow(n, "ALL", "hd95_mm") for n in names]
    sig = [(r["p_wilcoxon"] < 0.05, r["median_diff_a_minus_b"] < 0) for r in rows]
    ok = not (all(s and b for s, b in sig) or all(s and not b for s, b in sig))
    res.append((label, "no mark", ", ".join(f"p={r['p_wilcoxon']:.2g}" for r in rows), ok))
mark(["TRUS|unet_boxes|medsam_reference_vs_medsam_box"], "+", "Table V U-Net box BAP-MOS (MedSAM) +")
pval("TRUS|unet_boxes|medsam_reference_vs_medsam_box", "ALL", "hd95_mm", lambda p, h: round(p, 3) == 0.008, "U-Net boxes: BAP-MOS (MedSAM) vs MedSAM (box) p=0.008")
for k in (1, 2):
    mark([f"TRUS|loco_case{k}|medsam_reference_vs_medsam_box"], "-", f"Table V LOCO {k} BAP-MOS (MedSAM) -")
    nomark([f"TRUS|loco_case{k}|reference_vs_box_only", f"TRUS|loco_case{k}|reference_vs_sam_box"], f"Table V LOCO {k} BAP-MOS (SAM) no mark")
nomark(["TRUS|unet_boxes|reference_vs_box_only", "TRUS|unet_boxes|reference_vs_sam_box"], "Table V U-Net box BAP-MOS (SAM) no mark")
pval("TRUS|unet_boxes|reference_vs_nnunet", "PTV", "hd95_mm", lambda p, h: p < 1e-5 or h < 1e-5, "U-Net boxes: vs nnU-Net p<1e-5")


# Table 'Where the robustness comes from' (tab:mixture), parsed from the draft, 2 decimals, plus its text.
import re as _re
DRAFT = Path(sys.argv[1]) if len(sys.argv) > 1 else None
MIXROW = {"Box only, 2\\,px jitter": "box_only", "Box only, 10\\,px jitter": "box_only_j10",
          "Box only, 20\\,px jitter": "box_only_j20", "Fixed 5:1:1": "fixed_511", "BAP-MOS": "reference",
          "Shared selector": "global", "Fixed 1:1:1": "fixed_111", "Fixed 2:3:5": "fixed_235", "Fixed 1:5:1": "fixed_151"}
if DRAFT:
    tex = DRAFT.read_text()
    blk = tex[tex.index("\\label{tab:mixture}"):]
    blk = blk[:blk.index("\\bottomrule")]
    for lab, mod in MIXROW.items():
        line = [l for l in blk.splitlines() if l.startswith(lab + " &")]
        assert len(line) == 1, lab
        cells = [c.strip() for c in line[0].rstrip("\\ ").split("&")][3:]
        for c, cond in zip(cells, ("exact", "pshift20", "ploose20")):
            ptv, al = _re.match(r"([\d.]+)/([\d.]+)", c).groups()
            key = E(mod) if cond == "exact" else f"TRUS|robust|{cond}|{mod}"
            mean_sd(key, "PTV", "hd95_mm", float(ptv), dec=2, label=f"Mixture table {lab} {cond} PTV")
            mean_sd(key, "ALL", "hd95_mm", float(al), dec=2, label=f"Mixture table {lab} {cond} all")
for m in ("fixed_511", "global", "fixed_235", "fixed_111"):
    pval(f"TRUS|exact|{m}_vs_reference", "ALL", "hd95_mm", lambda p, h: p >= 0.295, f"Mixture text: exact {m} vs BAP-MOS all p>=0.30")
frac("TRUS|pshift20|fixed_111_vs_box_only", "PTV", "hd95_mm", 93, "Mixture text: 1:1:1 vs box-only ±20 PTV 93%")
pval("TRUS|pshift20|fixed_111_vs_box_only", "PTV", "hd95_mm", H(1e-6), "Mixture text: 1:1:1 vs box-only ±20 PTV Holm p<1e-6")
pval("TRUS|pshift20|fixed_511_vs_box_only", "PTV", "hd95_mm", approx(0.68, "h"), "Mixture text: 5:1:1 vs box-only ±20 PTV p=0.68")
for c in ("pshift5", "pshift10", "pshift20", "ploose5", "ploose10", "ploose20"):
    pval(f"TRUS|{c}|fixed_235_vs_reference", "ALL", "hd95_mm", lambda p, h: p <= 0.0075, f"Mixture text: 2:3:5 better than BAP-MOS all organs {c} p<=0.007")
    r = prow(f"TRUS|{c}|fixed_235_vs_reference", "ALL", "hd95_mm")
    res.append((f"Mixture text: 2:3:5 direction {c}", "2:3:5 lower", f"{r['A_mean']:.3f} vs {r['B_mean']:.3f}", r["A_mean"] < r["B_mean"]))
    pval(f"TRUS|{c}|global_vs_reference", "PTV", "hd95_mm", lambda p, h: h >= 0.05, f"Mixture text: shared selector = per-organ on PTV {c}")
frac("TRUS|pshift20|fixed_235_vs_reference", "PTV", "hd95_mm", 100, "Mixture text: 2:3:5 better on every PTV slice ±20")
pval("TRUS|pshift20|fixed_235_vs_reference", "PTV", "hd95_mm", H(1e-7), "Mixture text: 2:3:5 vs BAP-MOS ±20 PTV Holm p<1e-7")
mean_sd(E("box_only_j20"), "PTV", "hd95_mm", 1.41, 0.41, dec=2, label="Mixture text: 20 px jitter exact PTV 1.41 ± 0.41")

# PFUS1 table (tab:pfus1) and text
PF = lambda m: f"PFUS1|exact|exact|{m}"
for m, v in {"sam_box": (0.839, 0.009, 28.99, 2.31, 11.65, 0.55), "bapmos_sam": (0.857, 0.001, 25.59, 0.54, 10.29, 0.26),
             "medsam_box": (0.825, 0.005, 32.04, 0.41, 12.57, 0.13), "bapmos_medsam": (0.840, 0.004, 29.19, 1.81, 11.74, 0.46)}.items():
    mean_sd(PF(m), "Bladder", "dice", v[0], v[1], label=f"PFUS1 table {m} Dice")
    mean_sd(PF(m), "Bladder", "hd95_mm", v[2], v[3], dec=2, label=f"PFUS1 table {m} HD95")
    mean_sd(PF(m), "Bladder", "msd_mm", v[4], v[5], dec=2, label=f"PFUS1 table {m} MSD")
for m, v in {"sam_box": 25.50, "bapmos_sam": 23.56, "medsam_box": 26.83, "bapmos_medsam": 24.99}.items():
    mean_sd(PF(m), "ALL", "hd95_mm", v, dec=2, label=f"PFUS1 text all structures {m} HD95")
for m, v in {"sam_box": 32.39, "bapmos_sam": 29.13, "medsam_box": 35.76, "bapmos_medsam": 32.43}.items():
    mean_sd(f"PFUS1|robust|pshift20|{m}", "Bladder", "hd95_mm", v, dec=2, label=f"PFUS1 text ±20 {m} bladder HD95")
pval("PFUS1|exact|bapmos_sam_vs_sam_box", "Bladder", "dice", approx(0.04, "h"), "PFUS1: SAM bladder Dice Holm 0.04")
pval("PFUS1|exact|bapmos_sam_vs_sam_box", "Bladder", "hd95_mm", approx(0.27, "h"), "PFUS1: SAM bladder HD95 Holm 0.27")
pval("PFUS1|exact|bapmos_sam_vs_sam_box", "ALL", "hd95_mm", lambda p, h: p < 1e-5, "PFUS1: SAM all structures p<1e-5")
pval("PFUS1|exact|bapmos_medsam_vs_medsam_box", "Bladder", "hd95_mm", approx(0.46, "h"), "PFUS1: MedSAM bladder HD95 Holm 0.46")
pval("PFUS1|exact|bapmos_medsam_vs_medsam_box", "ALL", "hd95_mm", approx(0.14), "PFUS1: MedSAM all structures p=0.14")
pval("PFUS1|pshift20|bapmos_sam_vs_sam_box", "Bladder", "hd95_mm", approx(0.02, "h"), "PFUS1: ±20 SAM Holm 0.02")
pval("PFUS1|pshift20|bapmos_medsam_vs_medsam_box", "Bladder", "hd95_mm", approx(0.046, "h"), "PFUS1: ±20 MedSAM Holm 0.046")
for c in ("pshift5", "pshift10", "pshift20", "ploose5", "ploose10", "ploose20"):
    for v in ("sam", "medsam"):
        r = prow(f"PFUS1|{c}|bapmos_{v}_vs_{v}_box", "Bladder", "hd95_mm")
        res.append((f"PFUS1 text: BAP-MOS ({v}) lower bladder HD95 {c}", "lower", f"{r['A_mean']:.2f} vs {r['B_mean']:.2f}", r["A_mean"] < r["B_mean"]))
        if c != "pshift20":
            pval(f"PFUS1|{c}|bapmos_{v}_vs_{v}_box", "Bladder", "hd95_mm", lambda p, h: h >= 0.05, f"PFUS1 text: only ±20 significant ({v}, {c})")
for v in ("sam", "medsam"):
    pval(f"PFUS1|exact|bapmos_{v}_vs_unet", "Bladder", "hd95_mm", lambda p, h: h <= 0.0075, f"PFUS1: BAP-MOS ({v}) vs U-Net bladder Holm<=0.007")

# PFUS1 text: degradation from exact to ±20 px shifts / +20 px enlarged (bladder), and the relative rise on both datasets
_h = lambda k, o: agg.agg(d["exports"][k], o, "hd95_mm")[0]
for m, v in {"bapmos_sam": 3.5, "sam_box": 3.4, "bapmos_medsam": 3.2, "medsam_box": 3.7}.items():
    x = _h(f"PFUS1|robust|pshift20|{m}", "Bladder") - _h(PF(m), "Bladder")
    res.append((f"PFUS1 text: ±20 bladder rise {m}", f"{v}", f"{x:.2f}", round(x, 1) == v))
for m, v in {"bapmos_sam": 3.5, "sam_box": 2.2}.items():
    x = _h(f"PFUS1|robust|ploose20|{m}", "Bladder") - _h(PF(m), "Bladder")
    res.append((f"PFUS1 text: +20 bladder rise {m}", f"{v}", f"{x:.2f}", round(x, 1) == v))
_r = [_h(f"PFUS1|robust|pshift20|{m}", o) / _h(PF(m), o) for m in ("sam_box", "bapmos_sam", "medsam_box", "bapmos_medsam") for o in ("Bladder", "ALL")]
res.append(("PFUS1 text: ±20 raises HD95 by 11-18%", "11-18%", f"{100*(min(_r)-1):.1f}-{100*(max(_r)-1):.1f}%", round(100*(min(_r)-1)) >= 11 and round(100*(max(_r)-1)) <= 18))
_t = [_h(f"TRUS|robust|pshift20|{m}", o) / _h(E(m), o) for m in ("sam_box", "box_only", "reference") for o in ("PTV", "ALL")]
res.append(("PFUS1 text: ±20 three- to fivefold on TRUS", "3-5x", f"{min(_t):.2f}-{max(_t):.2f}x", 3 <= min(_t) and max(_t) < 5.5))

json.dump(res, open(Path(__file__).resolve().parent / "paper_check.json", "w"))
bad = [r for r in res if not r[3]]
print(f"{len(res)} draft numbers checked, {len(bad)} disagree")
for r in bad:
    print("  DISAGREES:", r)

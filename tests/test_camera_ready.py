"""Tests for the camera-ready (reviewer-response) controls and ablations (no GPU / no data)."""

from __future__ import annotations

import numpy as np
import pytest


ORGANS = ["rectum", "bladder", "ptv1"]


def test_global_policy_one_arm_per_block_and_mean_reward():
    from bapmos.method.bandit_policy import BlockScheduler, GlobalBanditPolicy

    policy = GlobalBanditPolicy(ORGANS, window_size=10, warmup_blocks_per_arm=1, seed=0)
    sched = BlockScheduler(policy, block_size_batches=1)
    arms = sched.begin_block(ORGANS)
    assert len(set(arms.values())) == 1
    arm = arms["rectum"]
    sched.end_block({"rectum": 0.2, "bladder": 0.4, "ptv1": 0.9})
    # One bandit step whose reward is the organ mean.
    assert policy.shared.t == 1
    assert policy.shared.N_a[arm] == 1
    assert policy.shared.Q_a[arm] == pytest.approx(0.5)
    best = policy.get_best_arms_per_organ()
    assert set(best.values()) == {arm}


def test_global_policy_state_round_trip():
    from bapmos.method.bandit_policy import BlockScheduler, GlobalBanditPolicy

    policy = GlobalBanditPolicy(ORGANS, warmup_blocks_per_arm=1, seed=3)
    sched = BlockScheduler(policy, block_size_batches=1)
    for r in (0.1, 0.5, 0.7, 0.3):
        sched.begin_block(ORGANS)
        sched.end_block({o: r for o in ORGANS})
    state = policy.export_state()
    clone = GlobalBanditPolicy(ORGANS, warmup_blocks_per_arm=1, seed=3)
    clone.load_state(state)
    assert clone.shared.Q_a == policy.shared.Q_a
    assert clone.shared.N_a == policy.shared.N_a
    assert clone.get_best_arms_per_organ() == policy.get_best_arms_per_organ()


def test_per_organ_scheduler_unchanged():
    from bapmos.method.bandit_policy import BlockScheduler, PerOrganBanditPolicy

    policy = PerOrganBanditPolicy(ORGANS, warmup_blocks_per_arm=1, seed=0)
    sched = BlockScheduler(policy, block_size_batches=1)
    sched.begin_block(ORGANS)
    sched.end_block({"rectum": 0.3, "bladder": 0.6})
    assert policy.bandits["rectum"].t == 1
    assert policy.bandits["bladder"].t == 1
    assert policy.bandits["ptv1"].t == 0  # no reward → no commit


def test_fixed_schedule_frequencies_and_test_arm():
    from bapmos.method.bandit_policy import BlockScheduler, FixedSchedulePolicy

    policy = FixedSchedulePolicy(ORGANS, {"box": 2, "point": 3, "both": 5}, seed=0)
    sched = BlockScheduler(policy, block_size_batches=1)
    counts = {"box": 0, "point": 0, "both": 0}
    for _ in range(3000):
        for arm in sched.begin_block(ORGANS).values():
            counts[arm] += 1
        sched.end_block({o: 1.0 for o in ORGANS})
    total = sum(counts.values())
    assert counts["box"] / total == pytest.approx(0.2, abs=0.02)
    assert counts["point"] / total == pytest.approx(0.3, abs=0.02)
    assert counts["both"] / total == pytest.approx(0.5, abs=0.02)
    assert set(policy.get_best_arms_per_organ().values()) == {"both"}
    restored = FixedSchedulePolicy(ORGANS, {"box": 2, "point": 3, "both": 5}, seed=0)
    restored.load_state(policy.export_state())
    assert restored.select_arm("rectum") == policy.select_arm("rectum")


def test_build_prompt_policy_rejects_bad_scope():
    from bapmos.method.bandit_policy import build_prompt_policy

    with pytest.raises(ValueError):
        build_prompt_policy("organwise", ORGANS)
    with pytest.raises(ValueError):
        build_prompt_policy("fixed", ORGANS)
    with pytest.raises(ValueError):
        build_prompt_policy("fixed", ORGANS, fixed_weights={"box": 1, "ring": 1})


def test_dice_only_reward():
    from bapmos.method.reward import dice_only_rewards

    out = dice_only_rewards(
        {
            "rectum": {"dice": 0.8, "msd_mm": 50.0, "hd95_mm": 90.0},
            "bladder": {"dice": float("nan"), "msd_mm": 0.1, "hd95_mm": 0.1},
        }
    )
    assert out == {"rectum": pytest.approx(0.8)}


def _square_prompt(mode="both"):
    from bapmos.method.prompt_geometry import build_organ_prompt

    mask = np.zeros((64, 80), dtype=np.uint8)
    mask[20:40, 30:50] = 1
    return build_organ_prompt(
        "bladder",
        mask,
        mode,
        image_hw=mask.shape,
        rng=np.random.default_rng(0),
        train=False,
        ring_mode="fixed",
        ring_width_fixed=4,
    )


def test_prompt_perturbation_bounded_and_deterministic():
    from bapmos.method.prompt_geometry import perturb_slice_prompts

    prompt = _square_prompt()
    clean = {"bladder": prompt}
    a = perturb_slice_prompts(clean, 5, (64, 80), "slice_7", 0)["bladder"]
    b = perturb_slice_prompts(clean, 5, (64, 80), "slice_7", 0)["bladder"]
    assert np.array_equal(a.box_xyxy, b.box_xyxy)
    assert np.array_equal(a.points_xy, b.points_xy)
    assert np.abs(a.box_xyxy - prompt.box_xyxy).max() <= 5
    assert np.abs(a.points_xy - prompt.points_xy).max() <= 5
    assert np.array_equal(a.labels, prompt.labels)
    # The original prompt is not modified in place.
    assert np.array_equal(_square_prompt().box_xyxy, prompt.box_xyxy)
    # Zero noise is the identity.
    same = perturb_slice_prompts(clean, 0, (64, 80), "slice_7", 0)["bladder"]
    assert same is prompt


def test_prompt_perturbation_keeps_box_valid_at_image_edge():
    from bapmos.method.prompt_geometry import perturb_organ_prompt
    from bapmos.method.types import OrganPrompts

    prompt = OrganPrompts("rectum", "box", "box", np.array([0, 0, 2, 2]), None, None)
    for seed in range(50):
        out = perturb_organ_prompt(prompt, 20, (32, 32), np.random.default_rng(seed))
        x0, y0, x1, y1 = out.box_xyxy.tolist()
        assert 0 <= x0 < x1 <= 31 and 0 <= y0 < y1 <= 31


def test_uniform_prompt_perturbation_unchanged_by_loose_mode():
    """Golden offsets recorded before the loose mode existed: uniform exports stay identical."""
    from bapmos.method.prompt_geometry import perturb_slice_prompts

    golden = {
        ("slice_7", 5): ([33, 23, 53, 43], [[34, 34], [33, 29], [50, 15], [33, 33]]),
        ("slice_7", 20): ([21, 7, 65, 41], [[22, 32], [22, 16], [29, 27], [45, 34]]),
        ("case2_0041", 10): ([30, 20, 55, 38], [[29, 35], [27, 33], [44, 14], [27, 34]]),
    }
    prompt = _square_prompt()
    for (sid, n), (box, pts) in golden.items():
        for kwargs in ({}, {"mode": "uniform"}):
            out = perturb_slice_prompts({"bladder": prompt}, n, (64, 80), sid, 0, **kwargs)
            assert out["bladder"].box_xyxy.tolist() == box
            assert out["bladder"].points_xy.astype(int).tolist() == pts


def test_loose_prompt_perturbation_only_enlarges_the_box():
    from bapmos.method.prompt_geometry import perturb_slice_prompts

    prompt = _square_prompt()
    x0, y0, x1, y1 = prompt.box_xyxy.tolist()
    seen = set()
    for i in range(30):
        sid = f"slice_{i}"
        a = perturb_slice_prompts({"bladder": prompt}, 10, (64, 80), sid, 0, mode="loose")["bladder"]
        b = perturb_slice_prompts({"bladder": prompt}, 10, (64, 80), sid, 0, mode="loose")["bladder"]
        assert np.array_equal(a.box_xyxy, b.box_xyxy)
        lx0, ly0, lx1, ly1 = a.box_xyxy.tolist()
        assert max(0, x0 - 10) <= lx0 <= x0 and max(0, y0 - 10) <= ly0 <= y0
        assert x1 <= lx1 <= min(79, x1 + 10) and y1 <= ly1 <= min(63, y1 + 10)
        assert np.array_equal(a.points_xy, prompt.points_xy)
        assert np.array_equal(a.labels, prompt.labels)
        seen.add(tuple(a.box_xyxy.tolist()))
    assert len(seen) > 1
    assert perturb_slice_prompts({"bladder": prompt}, 0, (64, 80), "s", 0, mode="loose")["bladder"] is prompt


def test_unknown_prompt_noise_mode_is_rejected():
    import pytest

    from bapmos.method.prompt_geometry import perturb_slice_prompts

    with pytest.raises(ValueError):
        perturb_slice_prompts({"bladder": _square_prompt()}, 5, (64, 80), "s", 0, mode="shifted")


def test_paired_box_offsets_do_not_depend_on_prompt_type_or_other_organs():
    from bapmos.method.prompt_geometry import perturb_box_paired, perturb_slice_prompts_paired
    from bapmos.method.types import OrganPrompts

    box = np.array([20, 18, 40, 44])
    pts = np.array([[30, 30], [10, 10]])
    lab = np.array([1, 0])
    o2c = {"bladder": 1, "ptv": 2}
    for mode in ("shift_paired", "loose_paired"):
        as_box = {"bladder": OrganPrompts("bladder", "box", "box", box, None, None),
                  "ptv": OrganPrompts("ptv", "box", "box", box, None, None)}
        as_both = {"bladder": OrganPrompts("bladder", "both", "both", box, pts, lab),
                   "ptv": OrganPrompts("ptv", "both", "both", box, pts, lab)}
        alone = {"ptv": OrganPrompts("ptv", "box", "box", box, None, None)}
        a = perturb_slice_prompts_paired(as_box, 10, mode, (64, 80), "s7", o2c, 0)
        b = perturb_slice_prompts_paired(as_both, 10, mode, (64, 80), "s7", o2c, 0)
        c = perturb_slice_prompts_paired(alone, 10, mode, (64, 80), "s7", o2c, 0)
        assert np.array_equal(a["ptv"].box_xyxy, b["ptv"].box_xyxy)
        assert np.array_equal(a["ptv"].box_xyxy, c["ptv"].box_xyxy)
        assert np.array_equal(a["bladder"].box_xyxy, b["bladder"].box_xyxy)
        # The baseline path calls perturb_box_paired directly with the class id.
        direct = perturb_box_paired(box, 10, mode, (64, 80), "s7", 2, 0)
        assert np.array_equal(direct, a["ptv"].box_xyxy)
        if mode == "loose_paired":
            assert np.array_equal(b["ptv"].points_xy, pts)
            x0, y0, x1, y1 = direct.tolist()
            assert 10 <= x0 <= 20 and 8 <= y0 <= 18 and 40 <= x1 <= 50 and 44 <= y1 <= 54
        else:
            assert np.abs(direct - box).max() <= 10
            assert np.abs(b["ptv"].points_xy - pts).max() <= 10
    # Different organs and conditions draw different offsets.
    assert not np.array_equal(
        perturb_box_paired(box, 20, "shift_paired", (64, 80), "s7", 1, 0),
        perturb_box_paired(box, 20, "shift_paired", (64, 80), "s7", 2, 0),
    )


def test_compare_prompt_logs(tmp_path):
    from bapmos.inference_output.prompt_log import compare_prompt_logs, write_prompt_log

    a = write_prompt_log({"s1": {"1": [1, 2, 3, 4]}, "s2": {"2": [0, 0, 5, 5]}}, tmp_path / "a", {})
    b = write_prompt_log({"s1": {"1": [1, 2, 3, 4]}, "s2": {"2": [0, 0, 5, 6]}}, tmp_path / "b", {})
    shared, mismatched, examples = compare_prompt_logs(a, b)
    assert (shared, mismatched) == (1, 1) and "s2" in examples[0]


def test_box_trainer_refuses_mismatched_cached_embedding():
    from types import SimpleNamespace

    import pytest

    pytest.importorskip("segment_anything")
    from bapmos.multiorgan.train_sam_multiorgan_decoder_box import SAMMultiOrganTrainer

    fake = SimpleNamespace(cfg={"sam_checkpoint": "models/sam_base/sam_vit_b_01ec64.pth"})
    img = np.zeros((40, 50, 3), dtype=np.uint8)
    ok = {"checkpoint": "sam_vit_b_01ec64.pth", "original_size": (40, 50)}
    SAMMultiOrganTrainer._check_cached_pack(fake, ok, img)
    with pytest.raises(ValueError):
        SAMMultiOrganTrainer._check_cached_pack(fake, {**ok, "checkpoint": "medsam_vit_b.pth"}, img)
    with pytest.raises(ValueError):
        SAMMultiOrganTrainer._check_cached_pack(fake, {**ok, "original_size": (41, 50)}, img)


def test_parallel_metric_scoring_matches_serial(monkeypatch):
    import math

    from bapmos.legacy.optimization.metrics.evaluator import MetricsEvaluator

    rng = np.random.default_rng(0)
    mapping = {1: "A", 2: "B", 3: "C"}
    slices = []
    for i in range(70):  # crosses the 64-slice chunk boundary
        gt = np.zeros((48, 40), np.uint8); pr = np.zeros((48, 40), np.uint8)
        for c in (1, 2, 3):
            if rng.random() < 0.8:
                y, x = rng.integers(2, 30, 2); gt[y:y + 10, x:x + 8] = c
            if rng.random() < 0.8:
                y, x = rng.integers(2, 30, 2); pr[y:y + 9, x:x + 9] = c
        slices.append((pr, gt, f"s{i}"))

    def run():
        ev = MetricsEvaluator(pixel_spacing=(0.5, 0.5), organs=list(mapping.values()))
        for k, (pr, gt, sid) in enumerate(slices):
            ev.evaluate_multiclass_slice(pr, gt, slice_idx=k, image_id=sid, class_mapping=mapping)
            if k == 30:
                ev.evaluate_slice(pr == 1, gt == 1, "A", k, sid + "_direct")  # mixed with queued slices
        return ev.per_slice_metrics

    monkeypatch.setenv("BAPMOS_METRIC_WORKERS", "0")
    serial = run()
    monkeypatch.setenv("BAPMOS_METRIC_WORKERS", "2")
    parallel = run()
    assert len(serial) == len(parallel) == 70 * 3 + 1
    for a, b in zip(serial, parallel):
        assert a.keys() == b.keys()
        for k in a:
            x, y = a[k], b[k]
            if isinstance(x, float) and math.isnan(x):
                assert isinstance(y, float) and math.isnan(y)
            else:
                assert x == y, (k, x, y)


def test_loco_lists_hold_out_whole_cohort():
    from bapmos.preprocess.prostate.create_loco_splits import loco_lists

    cohorts = {
        "simulation": {"train": ["s1", "s2"], "val": ["s3"], "test": ["s4"]},
        "case1": {"train": ["a1", "a2"], "val": ["a3"], "test": ["a4"]},
        "case2": {"train": ["b1"], "val": ["b2"], "test": ["b3"]},
    }
    out = loco_lists(cohorts, "case2")
    assert sorted(out["test"]) == ["b1", "b2", "b3"]
    assert sorted(out["train"]) == ["a1", "a2", "a4", "s1", "s2", "s4"]
    assert sorted(out["val"]) == ["a3", "s3"]
    with pytest.raises(ValueError):
        loco_lists(cohorts, "case3")


def test_camera_ready_suite_uses_paper_tpe_configuration():
    from bapmos.method.config_utils import compose_config, load_composed_config

    base = compose_config("bapmos_camera_ready", "pooled")
    prod = compose_config("bapmos", "pooled")
    # Paper TPE trial 61: (tau, alpha, r_min, W, K) = (26, 0.108, 10, 50, 70).
    assert base["meta"]["selection"]["trial_number"] == 61
    assert base["reward"]["clip_max_mm"] == 26.0
    assert base["prompt_geometry"]["alpha"] == pytest.approx(0.10884662160748368)
    assert base["prompt_geometry"]["r_min"] == 10
    assert base["bandit"]["window_size"] == 50
    assert base["bandit"]["block_size_batches"] == 70
    assert base["run_root"] == "runs/prostate/camera_ready"
    # Everything but the five searched parameters matches the production inner loop.
    for block, keys in (
        ("reward", {"clip_max_mm"}),
        ("prompt_geometry", {"alpha", "r_min"}),
        ("bandit", {"window_size", "block_size_batches"}),
    ):
        strip = lambda d: {k: v for k, v in d.items() if k not in keys}
        assert strip(base[block]) == strip(prod[block]), block
    assert base["training"] == prod["training"]
    cache_keys = {"use_cached_image_embeddings", "image_embedding_dir"}
    assert {k: v for k, v in base["common"].items() if k not in cache_keys} == prod["common"]
    assert base["common"]["use_cached_image_embeddings"] is True
    assert base["evaluation"]["eval_train_metrics"] is False

    names = [e["name"] for e in base["experiments"]]
    assert len(names) == len(set(names))
    for name in names:
        assert name.endswith(("_seed42", "_seed43_rep2", "_seed44_rep3"))

    ref = load_composed_config(
        version="bapmos_camera_ready", dataset="pooled", experiment_name="reference_seed42"
    )
    for block in ("reward", "prompt_geometry", "bandit", "training", "common"):
        assert ref[block] == base[block], block

    glob = load_composed_config(
        version="bapmos_camera_ready", dataset="pooled", experiment_name="global_seed43_rep2"
    )
    assert glob["bandit"]["scope"] == "global"
    assert glob["experiment_seed"] == 43
    assert {k: v for k, v in glob["bandit"].items() if k != "scope"} == base["bandit"]

    ring = load_composed_config(
        version="bapmos_camera_ready", dataset="pooled", experiment_name="fixed_ring_seed42"
    )
    assert ring["prompt_geometry"]["ring_mode"] == "fixed"
    assert ring["prompt_geometry"]["ring_width"] == base["prompt_geometry"]["r_min"]

    nbl = load_composed_config(
        version="bapmos_camera_ready", dataset="pooled", experiment_name="no_boundary_loss_seed42"
    )
    assert nbl["training"]["kervadec"]["alpha_start"] == 1.0
    assert nbl["training"]["kervadec"]["alpha_end"] == 1.0

    med = load_composed_config(
        version="bapmos_camera_ready", dataset="pooled", experiment_name="medsam_reference_seed44_rep3"
    )
    assert med["common"]["sam_init"] == "medsam"
    assert med["common"]["image_embedding_dir"].endswith("medsam_vit_b_embeddings")
    assert med["prompt_geometry"] == base["prompt_geometry"] and med["experiment_seed"] == 44

    loco = load_composed_config(
        version="bapmos_camera_ready", dataset="pooled", experiment_name="loco_case2_box_only_seed44_rep3"
    )
    assert loco["common"]["splits_subdir"] == "splits_loco/heldout_case2"
    assert loco["bandit"]["arms"] == ["box"]

    # BAP-MOS (MedSAM) held out per case: MedSAM init and cache, LOCO split, paper TPE vector.
    for case in ("case1", "case2"):
        mloco = load_composed_config(
            version="bapmos_camera_ready", dataset="pooled",
            experiment_name=f"loco_{case}_medsam_reference_seed43_rep2",
        )
        assert mloco["common"]["splits_subdir"] == f"splits_loco/heldout_{case}"
        assert mloco["common"]["sam_init"] == "medsam"
        assert mloco["common"]["image_embedding_dir"] == med["common"]["image_embedding_dir"]
        assert mloco["prompt_geometry"] == base["prompt_geometry"]
        assert mloco["bandit"] == base["bandit"] and mloco["experiment_seed"] == 43


def test_exclusive_val_loader_drops_probe_slices():
    from torch.utils.data import DataLoader

    from bapmos.method.bap_mos_trainer import exclusive_val_loader

    data = list(range(10))
    loader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=lambda b: b)
    out = exclusive_val_loader(loader, np.array([1, 4, 7]))
    seen = [batch[0] for batch in out]
    assert seen == [0, 2, 3, 5, 6, 8, 9]
    with pytest.raises(ValueError):
        exclusive_val_loader(loader, np.arange(10))


def _write_per_slice(path, rows):
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _slice_row(image_id, organ, hd95, valid=True):
    return {
        "image_id": image_id,
        "organ": organ,
        "hd95_mm": hd95 if valid else None,
        "msd_mm": hd95 / 3 if valid else None,
        "dice": 0.9 if valid else 0.0,
        "gt_empty": False,
        "valid_boundary": valid,
    }


def test_paired_stats_pairs_slices_and_counts_invalid(tmp_path):
    from bapmos.results.paired_stats import paired_table, seed_mean

    ids = [f"s{i}.png" for i in range(8)]
    for site in ("case1", "case2"):
        _write_per_slice(
            tmp_path / "a" / site / "metrics" / "per_slice_metrics.csv",
            [_slice_row(f"{site}_{i}", "PTV", 1.0 + 0.1 * k) for k, i in enumerate(ids)],
        )
        rows_b = [_slice_row(f"{site}_{i}", "PTV", 2.0 + 0.1 * k) for k, i in enumerate(ids)]
        rows_b[0] = _slice_row(f"{site}_{ids[0]}", "PTV", 0.0, valid=False)
        _write_per_slice(tmp_path / "b" / site / "metrics" / "per_slice_metrics.csv", rows_b)
    table = paired_table(seed_mean([tmp_path / "a"]), seed_mean([tmp_path / "b"]), "A", "B")
    row = table[(table.organ == "PTV") & (table.metric == "hd95_mm")].iloc[0]
    assert row.n_pairs == 14  # 16 slices minus the 2 invalid in B
    assert row.b_invalid == 2 and row.a_invalid == 0
    assert row.median_diff_a_minus_b == pytest.approx(-1.0)
    assert row.a_better_frac == 1.0
    assert row.p_wilcoxon < 0.01


def test_holm_adjustment():
    from bapmos.results.paired_stats import holm

    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    out = holm([float("nan"), 0.02])
    assert np.isnan(out[0]) and out[1] == pytest.approx(0.02)


def test_bandit_preferences_from_checkpoint(tmp_path):
    import torch

    from bapmos.method.bandit_policy import BlockScheduler, PerOrganBanditPolicy
    from bapmos.results.bandit_preferences import preference_rows, summarize

    policy = PerOrganBanditPolicy(["rectum", "ptv1"], warmup_blocks_per_arm=1, seed=0)
    sched = BlockScheduler(policy, block_size_batches=1)
    for _ in range(6):
        arms = sched.begin_block(["rectum"])
        sched.end_block({"rectum": 0.9 if arms["rectum"] == "point" else 0.1})
    run = tmp_path / "run_seed42"
    run.mkdir()
    torch.save({"bandit_state": policy.export_state(), "epoch_index": 3}, run / "best_checkpoint.pth")
    import pandas as pd

    df = pd.DataFrame(preference_rows(run))
    rectum = df[df.organ == "rectum"].set_index("arm")
    assert rectum.loc["point", "test_arm"]
    assert rectum["N_a"].sum() == 6
    assert df[df.organ == "ptv1"]["never_rewarded"].all()
    summary = summarize(df)
    assert set(summary.columns) >= {"organ", "arm", "Q_a_mean", "test_arm_runs"}


def test_load_prompt_mask_finds_site_file_and_refuses_missing(tmp_path):
    import cv2

    from bapmos.method.run_test_inference import load_prompt_mask

    site = tmp_path / "case1" / "predictions" / "multiclass"
    site.mkdir(parents=True)
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:5, 3:7] = 2
    cv2.imwrite(str(site / "MR.1.2_pred_ids.png"), mask)
    assert np.array_equal(load_prompt_mask(tmp_path, "MR.1.2.png", (10, 12)), mask)
    assert np.array_equal(load_prompt_mask(site, "MR.1.2.png", (10, 12)), mask)
    with pytest.raises(FileNotFoundError):
        load_prompt_mask(tmp_path, "MR.9.9.png", (10, 12))
    with pytest.raises(ValueError):
        load_prompt_mask(tmp_path, "MR.1.2.png", (11, 12))


def test_finetune_scope_grads_and_checkpoint_parts():
    import torch

    from bapmos.multiorgan.finetune_scope import (
        apply_finetune_scope,
        encoder_grad,
        load_sam_model_state,
        sam_model_state,
    )

    class TinySam(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = torch.nn.Linear(2, 2)
            self.prompt_encoder = torch.nn.Linear(2, 2)
            self.mask_decoder = torch.nn.Linear(2, 2)

    model = TinySam()
    apply_finetune_scope(model, "decoder")
    assert not any(p.requires_grad for p in model.image_encoder.parameters())
    assert all(p.requires_grad for p in model.mask_decoder.parameters())
    assert set(sam_model_state(model, "decoder")) == {"mask_decoder"}
    with encoder_grad("decoder", True):
        assert not torch.is_grad_enabled()

    apply_finetune_scope(model, "full")
    assert all(p.requires_grad for p in model.parameters())
    with encoder_grad("full", True):
        assert torch.is_grad_enabled()
    with encoder_grad("full", False):
        assert not torch.is_grad_enabled()
    state = sam_model_state(model, "full")
    assert set(state) == {"image_encoder", "prompt_encoder", "mask_decoder"}

    other = TinySam()
    load_sam_model_state(other, state)
    assert torch.equal(other.image_encoder.weight, model.image_encoder.weight)
    with pytest.raises(ValueError):
        apply_finetune_scope(model, "encoder")
    with pytest.raises(KeyError):
        load_sam_model_state(other, {"image_encoder": state["image_encoder"]})


@pytest.mark.parametrize(
    "backbone,trial,clip,r_min,window,block",
    [("sam", 18, 20.0, 6, 50, 50), ("medsam", 35, 25.0, 5, 85, 25)],
)
def test_camera_ready_pfus1_suites_use_paper_era_search(backbone, trial, clip, r_min, window, block):
    from bapmos.method.config_utils import compose_config, load_composed_config

    cfg = compose_config(f"bapmos_camera_ready_pfus1_{backbone}", "pfus1")
    prod = compose_config(f"bapmos_{backbone}", "pfus1")
    assert cfg["meta"]["selection"]["trial_number"] == trial
    assert cfg["reward"]["clip_max_mm"] == clip
    assert cfg["prompt_geometry"]["r_min"] == r_min
    assert cfg["bandit"]["window_size"] == window and cfg["bandit"]["block_size_batches"] == block
    # Every box-prompted camera-ready method uses a 2 px train-time box jitter.
    assert cfg["prompt_geometry"]["box_jitter_px"] == 2
    assert cfg["run_root"] == f"runs/bladder/camera_ready/{backbone}"
    assert cfg["common"] == prod["common"]
    assert cfg["training"] == prod["training"]
    run = load_composed_config(
        version=f"bapmos_camera_ready_pfus1_{backbone}", dataset="pfus1", experiment_name="pfus1_seed43_rep2"
    )
    assert run["experiment_seed"] == 43


def test_write_splits_final_skips_identical_and_replaces_changed(tmp_path):
    import json

    from bapmos.external_baselines.nnunet2d.patient_splits import write_splits_final

    splits = [{"train": ["a", "b"], "val": ["c"]}]
    out = write_splits_final(tmp_path, splits)
    mtime = out.stat().st_mtime_ns
    assert write_splits_final(tmp_path, splits) == out
    assert out.stat().st_mtime_ns == mtime  # identical content: not rewritten
    write_splits_final(tmp_path, [{"train": ["a"], "val": ["b", "c"]}])
    assert json.loads(out.read_text())[0]["val"] == ["b", "c"]
    assert not list(tmp_path.glob("*.tmp"))


def test_paired_stats_patient_level_averages_frames():
    import pandas as pd

    from bapmos.results.paired_stats import patient_of, to_patient_level

    assert patient_of("P012_frame_0034") == "P012"
    with pytest.raises(ValueError):
        patient_of("MR.1.2.png")
    df = pd.DataFrame(
        {
            "site": ["test"] * 4,
            "image_id": ["P001_frame_1", "P001_frame_2", "P002_frame_1", "P002_frame_2"],
            "organ": ["Bladder"] * 4,
            "dice": [0.8, 0.6, 0.9, 0.7],
            "hd95_mm": [2.0, 4.0, 1.0, 3.0],
            "msd_mm": [1.0, 1.0, 1.0, 1.0],
            "invalid": [0, 1, 0, 0],
        }
    )
    out = to_patient_level(df).set_index("image_id")
    assert out.loc["P001", "dice"] == pytest.approx(0.7)
    assert out.loc["P002", "hd95_mm"] == pytest.approx(2.0)
    assert out.loc["P001", "invalid"] == 1


def _loop_surface_distances(pred, gt, spacing):
    # The pre-vectorisation per-point loop, kept verbatim as the reference.
    from bapmos.legacy.optimization.metrics.boundary_metrics import get_boundary_points

    pb, gb = get_boundary_points(pred), get_boundary_points(gt)
    a = []
    for p in pb:
        a.append(np.min(np.sqrt(np.sum((gb * np.array(spacing) - p * np.array(spacing)) ** 2, axis=1))))
    b = []
    for g in gb:
        b.append(np.min(np.sqrt(np.sum((pb * np.array(spacing) - g * np.array(spacing)) ** 2, axis=1))))
    d = np.array(a + b)
    return float(np.percentile(d, 95)), float(np.mean(d))


@pytest.mark.parametrize("spacing", [(1.0, 1.0), (0.154, 0.154), (0.2, 0.31), (1, 1)])
def test_vectorised_surface_distances_bit_identical(spacing):
    import cv2

    from bapmos.legacy.optimization.metrics import boundary_metrics as bm

    rng = np.random.default_rng(0)
    for _ in range(12):
        gt = np.zeros((180, 220), np.uint8)
        cv2.ellipse(gt, (int(rng.integers(60, 160)), int(rng.integers(50, 130))),
                    (int(rng.integers(10, 50)), int(rng.integers(10, 40))), float(rng.uniform(0, 180)), 0, 360, 1, -1)
        pred = (rng.random(gt.shape) < rng.uniform(0.002, 0.3)).astype(np.uint8) | np.roll(gt, int(rng.integers(-9, 9)), 1)
        hd, msd = _loop_surface_distances(pred, gt, spacing)
        assert bm.hausdorff_distance_95(pred, gt, spacing) == hd
        assert bm.mean_surface_distance(pred, gt, spacing) == msd
        m = bm.compute_all_metrics(pred, gt, spacing)
        assert m["hd95_mm"] == hd and m["msd_mm"] == msd == m["boundary_dist_mm"]


def test_baseline_exports_accept_external_prompt_masks():
    """SAM (box) and MedSAM (box) exports can take boxes from a first-stage model's masks."""
    import inspect

    from bapmos.external_baselines.medsam_init.run_test_inference import run_medsam_test_export
    from bapmos.multiorgan.run_test_inference import run_multiorgan_sam_test_export

    for fn in (run_multiorgan_sam_test_export, run_medsam_test_export):
        assert inspect.signature(fn).parameters["prompt_mask_dir"].default is None


def test_boundary_distances_equal_brute_force():
    """k-d tree nearest-boundary distances equal the all-pairs minimum exactly."""
    from bapmos.legacy.optimization.metrics import boundary_metrics as bm

    def brute(src, dst, spacing):
        s = np.array(spacing)
        diff = (dst * s)[None, :, :] - (src * s)[:, None, :]
        return np.min(np.sqrt(np.sum(diff ** 2, axis=2)), axis=1)

    rng = np.random.default_rng(0)
    for trial in range(40):
        h, w = rng.integers(20, 90, size=2)
        pred = (rng.random((h, w)) < rng.uniform(0.05, 0.6)).astype(np.uint8)
        gt = np.zeros((h, w), np.uint8)
        y, x = rng.integers(2, h - 2), rng.integers(2, w - 2)
        gt[max(0, y - 15):y + 15, max(0, x - 20):x + 20] = 1
        spacing = (0.159072, 0.159072) if trial % 2 else (1.0, 1.0)
        pb, gb = bm.get_boundary_points(pred), bm.get_boundary_points(gt)
        if len(pb) == 0 or len(gb) == 0:
            continue
        assert np.array_equal(bm._directed_min_distances(pb, gb, spacing), brute(pb, gb, spacing))
        assert np.array_equal(bm._directed_min_distances(gb, pb, spacing), brute(gb, pb, spacing))
        both = np.concatenate([brute(pb, gb, spacing), brute(gb, pb, spacing)])
        m = bm.compute_all_metrics(pred, gt, spacing)
        assert m["hd95_mm"] == float(np.percentile(both, 95))
        assert m["msd_mm"] == float(np.mean(both)) == m["boundary_dist_mm"]

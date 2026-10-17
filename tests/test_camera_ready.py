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
    assert base["common"] == prod["common"]

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

    loco = load_composed_config(
        version="bapmos_camera_ready", dataset="pooled", experiment_name="loco_case2_box_only_seed44_rep3"
    )
    assert loco["common"]["splits_subdir"] == "splits_loco/heldout_case2"
    assert loco["bandit"]["arms"] == ["box"]


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

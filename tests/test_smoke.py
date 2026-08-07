"""Smoke tests for the organized BAPMOS package (no GPU / no data required)."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _allow_placeholder_selected_for_compose(monkeypatch):
    """Production selected/*.yaml ship as generation:0 placeholders until export."""
    monkeypatch.setenv("BAPMOS_ALLOW_PLACEHOLDER_SELECTED", "1")


def test_import_package():
    import bapmos
    assert hasattr(bapmos, "__version__")


def test_kervadec_style_signed_distance():
    from bapmos.losses.kervadec_style import class_distance_map

    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    dist = class_distance_map(mask)
    assert dist[16, 16] < 0
    assert dist[0, 0] > 0


def test_checkpoint_objective_default():
    from bapmos.checkpoint_selection import parse_checkpoint_objective_config

    cfg = parse_checkpoint_objective_config(
        {"checkpoint_objective": {"metric": "ptv_kfold_msd", "kfold_n": 5}}
    )
    assert cfg.metric == "ptv_kfold_msd"
    assert cfg.kfold_n == 5


def test_boundary_loss_finite():
    from bapmos.losses.kervadec_style import boundary_loss

    logits = torch.randn(1, 4, 16, 16)
    gt = torch.zeros(1, 1, 16, 16, dtype=torch.long)
    gt[0, 0, 4:12, 4:12] = 1
    loss = boundary_loss(logits, gt, num_classes=4)
    assert torch.isfinite(loss)


def test_preprocess_clis_dry_run():
    from bapmos.preprocess.prostate import main as prostate_main
    from bapmos.preprocess.bladder import main as bladder_main
    from bapmos.preprocess.delineation import main as delineation_main

    assert prostate_main(["--dry-run"]) == 0
    assert bladder_main(["--dry-run"]) == 0
    assert delineation_main(["help"]) == 0


def test_inference_output_difference_v1_helper():
    import numpy as np

    from bapmos.evaluation.difference_v1 import build_difference_v1_rgb
    from bapmos.inference_output import export_stratified_test_bundle

    assert callable(export_stratified_test_bundle)
    gt = np.zeros((8, 8), dtype=np.uint8)
    gt[2:6, 2:6] = 1
    pred = gt.copy()
    pred[2, 2] = 0
    pred[1, 1] = 1
    rgb = build_difference_v1_rgb(gt, pred, foreground_class_ids=[1])
    assert rgb.shape == (8, 8, 3)


def test_outer_loop_tpe_budget_constants():
    from bapmos.hpo.catalog_search import (
        INNER_LOOP_TPE_N_STARTUP_TRIALS,
        OUTER_LOOP_TPE_N_STARTUP_TRIALS,
        OUTER_LOOP_TPE_N_TRIALS,
    )

    assert OUTER_LOOP_TPE_N_STARTUP_TRIALS == 20
    assert OUTER_LOOP_TPE_N_TRIALS == 100
    # Deprecated alias kept for older scripts.
    assert INNER_LOOP_TPE_N_STARTUP_TRIALS == OUTER_LOOP_TPE_N_STARTUP_TRIALS


def test_trainer_requires_protocol_fields():
    import pytest

    from bapmos.method.bap_mos_trainer import BAPMOSTrainer

    with pytest.raises(ValueError, match="data_root"):
        BAPMOSTrainer({"common": {}, "evaluation": {}})
    from pathlib import Path

    import yaml

    from bapmos.metrics.checkpoint_selection import parse_checkpoint_objective_config

    proto = Path(__file__).resolve().parents[1] / "configs" / "common" / "protocol.yaml"
    raw = yaml.safe_load(proto.read_text())
    cfg = parse_checkpoint_objective_config(raw)
    assert cfg.metric == "ptv_kfold_msd"


def test_evaluation_block_checkpoint_metric():
    """BAP-MOS version.yaml style: evaluation.objective_metric / kfold_*."""
    from bapmos.metrics.checkpoint_selection import parse_checkpoint_objective_config

    cfg = parse_checkpoint_objective_config(
        {
            "evaluation": {
                "objective_metric": "ptv_kfold_msd",
                "objective_organ": "Bladder",
                "kfold_n": 5,
                "kfold_seed": 42,
            }
        }
    )
    assert cfg.metric == "ptv_kfold_msd"
    assert cfg.objective_organ == "Bladder"
    assert cfg.kfold_n == 5
    assert cfg.kfold_seed == 42


def test_legacy_package_importable():
    import bapmos.legacy
    import bapmos.legacy.optimization  # noqa: F401
    assert bapmos.legacy.__doc__ or True


def test_root_shims_alias_canonical_modules():
    """Deprecated root paths must resolve to the same objects as canonical modules."""
    from bapmos.checkpoint_selection import parse_checkpoint_objective_config as shim_parse
    from bapmos.metrics.checkpoint_selection import parse_checkpoint_objective_config as can_parse
    from bapmos.organ_registry import PFUS1_ORGAN_TO_CLASS as shim_organs
    from bapmos.data.organ_registry import PFUS1_ORGAN_TO_CLASS as can_organs
    from bapmos.training_taxonomy import get_baseline_taxonomy_profile as shim_tax
    from bapmos.train.training_taxonomy import get_baseline_taxonomy_profile as can_tax
    from bapmos.baseline_validation_metrics import organ_balanced_validation_summary as shim_sum
    from bapmos.metrics.baseline_validation_metrics import (
        organ_balanced_validation_summary as can_sum,
    )
    from bapmos.legacy.optimization.metrics import MetricsEvaluator as legacy_me
    from bapmos.optimization.metrics import MetricsEvaluator as root_me
    from bapmos.pdf_export import INFERENCE_PANEL_PDF_DPI, PDF_EXPORT_DPI

    assert shim_parse is can_parse
    assert shim_organs is can_organs
    assert shim_tax is can_tax
    assert shim_sum is can_sum
    assert root_me is legacy_me
    assert PDF_EXPORT_DPI == 350
    assert INFERENCE_PANEL_PDF_DPI == 350
    assert INFERENCE_PANEL_PDF_DPI == PDF_EXPORT_DPI


def test_inference_output_protocol_seed42_and_viz():
    from bapmos.inference_output.protocol import (
        DEFAULT_VIZ_PDF_MAX,
        PANEL_EXPORT_DPI,
        export_viz_protocol_kwargs,
        is_primary_inference_seed_run,
        require_primary_inference_seed_run,
    )
    import pytest

    assert PANEL_EXPORT_DPI == 350
    assert DEFAULT_VIZ_PDF_MAX == 10
    assert is_primary_inference_seed_run(run_name="pooled_seed42")
    assert is_primary_inference_seed_run(experiment_seed=42)
    assert is_primary_inference_seed_run(run_name="pooled_seed42", experiment_seed=42)
    assert not is_primary_inference_seed_run(run_name="pooled_seed43_rep2")
    assert not is_primary_inference_seed_run(experiment_seed=43)
    # Name/seed disagreement must not export as primary.
    assert not is_primary_inference_seed_run(run_name="pooled_seed42", experiment_seed=43)
    assert not is_primary_inference_seed_run(run_name="pooled_seed43", experiment_seed=42)
    with pytest.raises(ValueError, match="primary"):
        require_primary_inference_seed_run(run_name="pfus1_seed44_rep3")
    require_primary_inference_seed_run(run_name="pfus1_seed42")
    kw = export_viz_protocol_kwargs()
    assert kw["viz_selection"] == "all"
    assert kw["viz_pdf_max"] == 10
    assert kw["viz_pdf_dpi"] == 350


def test_compose_backbone_suites_keep_sam_init_distinct():
    """Bladder/prostate MedSAM outer suites must not collapse onto Meta SAM YAML."""
    import os

    from bapmos.hpo.paths import HpoPaths
    from bapmos.method.config_utils import compose_config

    os.environ.pop("BAPMOS_SELECTED_SUBDIR", None)

    sam = compose_config("bapmos_sam", "pfus1")
    med = compose_config("bapmos_medsam", "pfus1")
    assert sam["ablation"]["version"] == "bapmos_sam"
    assert med["ablation"]["version"] == "bapmos_medsam"
    assert med["common"].get("sam_init") == "medsam"
    assert "medsam" in str(med["common"]["sam_checkpoint"])
    assert sam["common"]["patience"] == 20 and sam["common"]["max_epochs"] == 100

    pooled = compose_config("bapmos", "pooled")
    assert pooled["meta"].get("selection", {}).get("search_method") == "tpe"
    assert pooled["common"]["patience"] == 40
    assert pooled["common"]["max_epochs"] == 300

    med_pooled = compose_config("bapmos_medsam_pooled", "pooled")
    assert med_pooled["common"].get("sam_init") == "medsam"

    sam_sel = HpoPaths.for_suite("bapmos_bo_sam").selected_params_path("pfus1")
    med_sel = HpoPaths.for_suite("bapmos_bo_medsam").selected_params_path("pfus1")
    assert sam_sel != med_sel
    assert sam_sel.is_file() and med_sel.is_file()


def test_outer_loop_search_budgets_and_loss():
    """Outer search budgets + kervadec loss must stay aligned with protocol tables."""
    import pytest

    from bapmos.method.config_utils import compose_config

    prostate = compose_config("bapmos_outer_loop", "pooled")
    assert prostate["common"]["max_epochs"] == 100
    assert prostate["common"]["patience"] == 15
    assert prostate["training"]["loss_mode"] == "kervadec"
    assert prostate["run_root"].endswith("outer_loop/tpe")
    assert prostate["evaluation"]["cleanup_checkpoints_after_train"] is True

    for suite, method in (
        ("bapmos_outer_loop_random", "random"),
        ("bapmos_outer_loop_greedy", "greedy"),
        ("bapmos_outer_loop_heuristic", "heuristic"),
    ):
        cfg = compose_config(suite, "pooled")
        assert cfg["common"]["max_epochs"] == 100
        assert cfg["common"]["patience"] == 15
        assert cfg["training"]["loss_mode"] == "kervadec"
        assert cfg["run_root"].endswith(f"outer_loop/{method}")
        assert cfg["evaluation"]["cleanup_checkpoints_after_train"] is True

    # Removed pre-rename / silent aliases must fail loudly.
    for removed in (
        "bapmos_inner_loop",
        "bapmos_sam_inner_loop",
        "bapmos_medsam_inner_loop",
        "bapmos_kervadec",
    ):
        ds = "pfus1" if "sam" in removed else "pooled"
        with pytest.raises(ValueError, match="Removed"):
            compose_config(removed, ds)

    for suite in ("bapmos_sam_outer_loop", "bapmos_medsam_outer_loop"):
        cfg = compose_config(suite, "pfus1")
        assert cfg["common"]["max_epochs"] == 15
        assert cfg["common"]["patience"] == 4
        assert cfg["training"]["loss_mode"] == "kervadec"
        assert "outer_loop" in cfg["run_root"]
        assert cfg["evaluation"]["cleanup_checkpoints_after_train"] is True
        # Outer-loop cheap run: bandit/early-stop structure matches inner.
        assert float(cfg["common"].get("val_msd_min_delta", 0.0)) == 0.0
        assert cfg["bandit"]["warmup_blocks_per_arm"] == 5
        assert cfg["bandit"]["min_pulls_per_arm"] == 3

    # Inner loop keeps production checkpoints.
    for suite, ds in (
        ("bapmos", "pooled"),
        ("bapmos_medsam_pooled", "pooled"),
        ("bapmos_sam", "pfus1"),
        ("bapmos_medsam", "pfus1"),
    ):
        inner = compose_config(suite, ds)
        assert inner["evaluation"].get("cleanup_checkpoints_after_train") is False
        assert inner["evaluation"].get("run_test_after_train") is False
        assert "inner_loop" in inner["run_root"]


def test_training_seed_preserves_fixed_kfold_and_probe():
    """Inner-loop seeds 43/44 must not drag kfold_seed / probe_seed off 42."""
    from bapmos.method.bapmos_seeds import apply_training_seed_env

    cfg = {
        "experiment_seed": 43,
        "bandit": {"probe_seed": 42},
        "evaluation": {"kfold_seed": 42},
    }
    out = apply_training_seed_env(cfg)
    assert out["experiment_seed"] == 43
    assert out["bandit"]["probe_seed"] == 42
    assert out["evaluation"]["kfold_seed"] == 42
    # Original dict must not be mutated.
    assert cfg["bandit"]["probe_seed"] == 42


def test_outer_loop_deletes_trial_checkpoints(tmp_path):
    """Scored outer-loop trials must drop .pth; keep-metrics left intact."""
    import optuna

    from bapmos.hpo.objective import (
        build_trial_config,
        cleanup_bo_trial_artifacts,
        delete_trial_checkpoints,
    )

    run_dir = tmp_path / "trial_0"
    run_dir.mkdir()
    (run_dir / "best_checkpoint.pth").write_bytes(b"x")
    (run_dir / "last_checkpoint.pth").write_bytes(b"y")
    (run_dir / "metrics.csv").write_text("epoch,val_msd_mm\n1,1.0\n", encoding="utf-8")
    assert delete_trial_checkpoints(run_dir) == 2
    assert not (run_dir / "best_checkpoint.pth").exists()
    assert (run_dir / "metrics.csv").is_file()

    (run_dir / "best_checkpoint.pth").write_bytes(b"z")
    cleanup_bo_trial_artifacts(run_dir)
    assert not (run_dir / "best_checkpoint.pth").exists()
    assert (run_dir / "metrics.csv").is_file()

    study = optuna.create_study(direction="minimize")
    cfg = build_trial_config("pooled", study.ask(), hpo_suite="bapmos_bo")
    assert cfg["evaluation"]["cleanup_checkpoints_after_train"] is True


def test_hpo_suite_wiring_and_study_names():
    """HPO suite map, selected paths, and study_name placeholders must stay locked."""
    from pathlib import Path

    import yaml

    from bapmos.hpo.paths import HPO_SUITES, HpoPaths
    from bapmos.method.config_utils import EXTERNAL_SUITE_DIRS, SUITE_BY_VERSION

    assert not (Path("experiments/prostate/bapmos/outer_loop/version.yaml")).exists()

    for suite, spec in HPO_SUITES.items():
        hpo_ver = spec["hpo_version"]
        assert hpo_ver in SUITE_BY_VERSION
        assert hpo_ver in EXTERNAL_SUITE_DIRS
        assert (EXTERNAL_SUITE_DIRS[hpo_ver] / "version.yaml").is_file()

        hp = HpoPaths.for_suite(suite)
        ds = "pfus1" if suite in ("bapmos_bo_sam", "bapmos_bo_medsam") else "pooled"
        sel = hp.selected_params_path(ds)
        assert sel.is_file(), sel
        meta = yaml.safe_load(sel.read_text()).get("selection_meta") or {}
        assert meta.get("study_name") == hp.study_name(ds)
        assert meta.get("hpo_suite") == suite


def test_hpo_trial_wandb_not_tagged_production():
    """Outer-loop Optuna trials must not inherit the production W&B tag."""
    import optuna
    import pytest

    from bapmos.hpo.objective import build_trial_config
    from bapmos.hpo.paths import HpoPaths, _suite_for_production
    from bapmos.method.config_utils import compose_config

    study = optuna.create_study(direction="minimize")
    cfg = build_trial_config("pooled", study.ask(), hpo_suite="bapmos_bo")
    tags = cfg["wandb"]["tags"]
    assert "production" not in tags
    assert "hpo" in tags and "outer_loop" in tags
    assert cfg["wandb"]["group"] == "bapmos_bo_pooled"

    # Outer-loop trainer CLI path (resolve_experiment) must also avoid production.
    from bapmos.method.config_utils import resolve_experiment

    outer_cli = resolve_experiment(compose_config("bapmos_outer_loop", "pooled"), "bo_trial_seed42")
    assert "production" not in outer_cli["wandb"]["tags"]
    assert "hpo" in outer_cli["wandb"]["tags"]

    inner_cli = resolve_experiment(compose_config("bapmos", "pooled"), "pooled_seed42")
    assert "production" in inner_cli["wandb"]["tags"]
    assert "hpo" not in inner_cli["wandb"]["tags"]

    cfg_sam = build_trial_config("pfus1", study.ask(), hpo_suite="bapmos_bo_sam")
    cfg_med = build_trial_config("pfus1", study.ask(), hpo_suite="bapmos_bo_medsam")
    assert cfg_sam["wandb"]["group"] != cfg_med["wandb"]["group"]
    assert "production" not in cfg_sam["wandb"]["tags"]

    # Prostate MedSAM outer is independent of Meta SAM TPE.
    med_outer = compose_config("bapmos_medsam_pooled_outer_loop", "pooled")
    sam_outer = compose_config("bapmos_outer_loop", "pooled")
    assert med_outer["common"]["max_epochs"] == 100
    assert med_outer["common"]["patience"] == 15
    assert med_outer["common"].get("sam_init") == "medsam"
    assert "medsam" in str(med_outer["common"]["sam_checkpoint"])
    assert med_outer["run_root"] != sam_outer["run_root"]
    assert _suite_for_production("bapmos_medsam_pooled") == "bapmos_bo_medsam_pooled"
    hp = HpoPaths.for_suite("bapmos_bo_medsam_pooled")
    assert hp.selected_params_path("pooled").is_file()
    assert hp.trial_run_root() == med_outer["run_root"]

    with pytest.raises(ValueError, match="No HPO suite maps"):
        _suite_for_production("not_a_real_production_suite")



def test_config_yaml_self_sufficient_for_trainer():
    """``--config`` loads as-is: version.yaml must carry data_root + pixel_spacing."""
    from bapmos.method.data_adapter import load_yaml_config, repo_root

    paths = [
        "experiments/prostate/bapmos/outer_loop/tpe/version.yaml",
        "experiments/prostate/bapmos/outer_loop/random/version.yaml",
        "experiments/prostate/bapmos/outer_loop/greedy/version.yaml",
        "experiments/prostate/bapmos/outer_loop/heuristic/version.yaml",
        "experiments/prostate/bapmos/outer_loop/medsam/version.yaml",
        "experiments/prostate/bapmos/inner_loop/version.yaml",
        "experiments/prostate/bapmos/inner_loop/medsam/version.yaml",
        "experiments/bladder/bapmos/outer_loop/sam/version.yaml",
        "experiments/bladder/bapmos/outer_loop/medsam/version.yaml",
        "experiments/bladder/bapmos/inner_loop/sam/version.yaml",
        "experiments/bladder/bapmos/inner_loop/medsam/version.yaml",
    ]
    root = repo_root()
    for rel in paths:
        cfg = load_yaml_config(str(root / rel))
        common = cfg.get("common", {})
        assert common.get("data_root"), rel
        assert common.get("pixel_spacing"), rel
        assert common.get("sam_checkpoint"), rel


def test_replicate_run_names_seed_true():
    from bapmos.replicate_runs import (
        all_study_run_names,
        is_replicate_run_name,
        replicate_run_name,
        replicate_training_seed,
    )

    primary = "pooled_seed42"
    assert replicate_training_seed(42, 2) == 43
    assert replicate_training_seed(42, 3) == 44
    assert replicate_run_name(primary, 2) == "pooled_seed43_rep2"
    assert replicate_run_name(primary, 3) == "pooled_seed44_rep3"
    assert all_study_run_names(primary) == (
        "pooled_seed42",
        "pooled_seed43_rep2",
        "pooled_seed44_rep3",
    )
    assert is_replicate_run_name("pooled_seed43_rep2", primary)
    assert not is_replicate_run_name("pooled_seed42", primary)
    assert replicate_run_name("pfus1_seed42", 2) == "pfus1_seed43_rep2"
    assert replicate_run_name("pfus1_seed42", 3) == "pfus1_seed44_rep3"


def test_project_root_is_bapmos_checkout():
    from bapmos.paths import (
        dataset_bundle_tag,
        pfus1_bundle_dir,
        pooled_prostate_dataset_dir,
        project_root,
    )

    root = project_root()
    assert (root / "src" / "bapmos").is_dir()
    assert (root / "configs" / "common" / "protocol.yaml").is_file()
    assert root.name == "BAPMOS"
    assert pooled_prostate_dataset_dir() == root / "data" / "prostate" / "pooled"
    assert pfus1_bundle_dir() == root / "data" / "bladder" / "pfus1"
    assert dataset_bundle_tag("data/prostate/pooled") == "prostate_pooled"
    assert dataset_bundle_tag("data/bladder/pfus1") == "pfus1"


def test_pfus1_shim_reexports_preprocess_bladder():
    """``bapmos.pfus1.*`` must re-export the same objects as ``preprocess.bladder`` (no drift)."""
    import bapmos.pfus1 as shim_pkg
    import bapmos.preprocess.bladder as bladder_pkg
    from bapmos.pfus1 import (
        PFUS1_ALL_LABELS,
        PFUS1Dataset,
        LabelMode,
        parse_sample_line,
        remap_mask_to_subset,
    )
    from bapmos.pfus1 import create_pfus1_splits, dataset_pfus1, verify_label_registry
    from bapmos.preprocess.bladder import (
        PFUS1_ALL_LABELS as labels,
        PFUS1Dataset as Dataset,
        LabelMode as Mode,
        parse_sample_line as parse,
        remap_mask_to_subset as remap,
    )
    from bapmos.preprocess.bladder import create_splits, dataset, verify_label_registry as verify

    assert PFUS1_ALL_LABELS is labels
    assert PFUS1Dataset is Dataset
    assert LabelMode is Mode
    assert parse_sample_line is parse
    assert remap_mask_to_subset is remap
    assert create_pfus1_splits.main is create_splits.main
    assert dataset_pfus1.parse_sample_line is dataset.parse_sample_line
    assert verify_label_registry.main is verify.main
    for name in shim_pkg.__all__:
        assert getattr(shim_pkg, name) is getattr(bladder_pkg, name)


def test_pooled_site_tests_and_runs_layout_helpers():
    """Pooled site_tests API + inference_output/prostate|bladder layout."""
    from bapmos.paths import (
        inference_output_layout_root,
        list_pooled_site_test_names,
        project_root,
    )

    assert str(inference_output_layout_root("data/prostate/pooled")).endswith(
        "inference_output/prostate/pooled"
    )
    assert str(inference_output_layout_root("data/bladder/pfus1")).endswith(
        "inference_output/bladder/pfus1"
    )
    pooled = project_root() / "data" / "prostate" / "pooled"
    # Empty when corpus not built; non-empty when site_tests present.
    assert isinstance(list_pooled_site_test_names(pooled), list)


def test_inner_loop_experiment_names_are_seed_true_run_names():
    from bapmos.method.config_utils import compose_config, list_experiments

    prostate = compose_config("bapmos", "pooled")
    names = [e["name"] for e in list_experiments(prostate)]
    assert names == ["pooled_seed42", "pooled_seed43_rep2", "pooled_seed44_rep3"]

    bladder = compose_config("bapmos_sam", "pfus1")
    names_b = [e["name"] for e in list_experiments(bladder)]
    assert names_b == ["pfus1_seed42", "pfus1_seed43_rep2", "pfus1_seed44_rep3"]


def test_results_collate_seeds_mean_std(tmp_path):
    """Ingest three seed summaries → paper combined CSVs only."""
    from bapmos.results.collate_seeds import build_corpus, ingest_summary
    from bapmos.results.layout import by_seed_csv_path, corpus_results_root

    repo = tmp_path
    # Minimal summary_metrics-like CSVs
    for seed, run, dice, msd in (
        (42, "pooled_seed42", 0.90, 1.0),
        (43, "pooled_seed43_rep2", 0.92, 1.2),
        (44, "pooled_seed44_rep3", 0.94, 0.8),
    ):
        summary = tmp_path / f"sum_{seed}.csv"
        summary.write_text(
            "organ,dice_mean,msd_mm_mean,hd95_mm_mean,iou_mean,n_slices\n"
            f"Bladder,{dice},{msd},2.0,0.8,10\n"
            f"all,{dice},{msd},2.0,0.8,10\n",
            encoding="utf-8",
        )
        ingest_summary(
            summary,
            corpus="prostate",
            method="box",
            run_name=run,
            seed=seed,
            repo=repo,
        )

    paths = build_corpus("prostate", method="box", repo=repo)
    names = {p.name for p in paths}
    assert names == {"prostate_pooled_mean_std.csv", "prostate_pooled_ptv_mean_std.csv"}
    combined = corpus_results_root("prostate", repo=repo) / "combined"
    assert sorted(p.name for p in combined.glob("*.csv")) == sorted(names)
    mean_csv = combined / "prostate_pooled_mean_std.csv"
    text = mean_csv.read_text(encoding="utf-8")
    assert "dice_mean_pm_std" in text
    assert "msd_mm_mean_pm_std" in text
    assert "hd95_mm_mean_pm_std" in text
    assert "iou" not in text
    assert "Bladder" in text
    assert by_seed_csv_path("prostate", "box", "pooled_seed42", repo=repo).is_file()


def test_collate_slice_weighted_pool_matches_investigation(tmp_path):
    """sum(site_mean × n) / N from per-slice valid_boundary rows across sites."""
    from bapmos.results.collate_seeds import (
        ingest_inference_run,
        pool_organs_from_per_slice,
    )
    from bapmos.results.layout import POOLED_SITE_LABEL, by_seed_csv_path

    run_dir = tmp_path / "box_pooled_seed42"
    # Unequal site sizes (like investigation 9+9+12): means must be slice-weighted.
    site_msds = {
        "simulation": [1.0, 3.0],  # mean 2, n=2
        "case1": [2.0],  # mean 2, n=1
        "case2": [4.0, 4.0, 4.0],  # mean 4, n=3 → pooled = 3.0
    }
    for site, msds in site_msds.items():
        metrics = run_dir / site / "metrics"
        metrics.mkdir(parents=True)
        lines = ["image_id,organ,msd_mm,hd95_mm,dice,valid_boundary"]
        for i, m in enumerate(msds):
            lines.append(f"{site}_{i},PTV,{m},{m},0.9,True")
        # Invalid-boundary row must be excluded from the pool.
        lines.append(f"{site}_bad,PTV,99.0,99.0,0.1,False")
        (metrics / "per_slice_metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (metrics / "summary_metrics.csv").write_text(
            "organ,n_slices,dice_mean,msd_mm_mean,hd95_mm_mean\n"
            f"PTV,{len(msds)+1},0.5,9.0,9.0\n",
            encoding="utf-8",
        )

    site_dirs = {s: run_dir / s / "metrics" for s in site_msds}
    pooled = pool_organs_from_per_slice(
        site_dirs, method="box", run_name="pooled_seed42", seed=42
    )
    ptv = next(r for r in pooled if r["organ"] == "PTV")
    assert ptv["site"] == POOLED_SITE_LABEL
    assert ptv["n_slices"] == 6.0
    assert abs(float(ptv["msd_mm"]) - 3.0) < 1e-9
    assert abs(float(ptv["dice"]) - 0.9) < 1e-9

    dest = ingest_inference_run(
        run_dir,
        corpus="prostate",
        method="box",
        run_name="pooled_seed42",
        seed=42,
        repo=tmp_path,
    )
    assert dest == by_seed_csv_path("prostate", "box", "pooled_seed42", repo=tmp_path)
    text = dest.read_text(encoding="utf-8")
    assert ",pooled,PTV," in text
    assert ",simulation,PTV," in text
    assert ",case1,PTV," in text
    assert ",case2,PTV," in text


def test_collate_mean_std_splits_ratio_experiments(tmp_path):
    """Multi-ratio methods get separate combined rows per experiment key."""
    from bapmos.results.collate_seeds import build_corpus, ingest_summary
    from bapmos.results.layout import POOLED_SITE_LABEL, corpus_results_root

    repo = tmp_path
    for run_name, dice in (
        ("box10_point90_seed42", 0.91),
        ("box10_point90_seed43_rep2", 0.92),
        ("box50_point50_seed42", 0.95),
        ("box50_point50_seed43_rep2", 0.96),
    ):
        summary = tmp_path / f"{run_name}.csv"
        summary.write_text(
            "organ,dice_mean,msd_mm_mean,hd95_mm_mean,n_slices\n"
            f"PTV,{dice},1.0,2.0,10\n",
            encoding="utf-8",
        )
        ingest_summary(
            summary,
            corpus="prostate",
            method="box_point",
            run_name=run_name,
            site=POOLED_SITE_LABEL,
            repo=repo,
        )

    build_corpus("prostate", method="box_point", repo=repo)
    text = (corpus_results_root("prostate", repo=repo) / "combined" / "prostate_pooled_ptv_mean_std.csv").read_text(
        encoding="utf-8"
    )
    assert "box10_point90" in text
    assert "box50_point50" in text
    assert text.count("box_point,box10_point90") == 1
    assert text.count("box_point,box50_point50") == 1


def test_collate_seeds_bladder_same_metrics(tmp_path):
    """Bladder uses the same Dice/MSD/HD95 collation + dedicated corpus CSV."""
    from bapmos.results.collate_seeds import build_corpus, ingest_summary
    from bapmos.results.layout import by_seed_csv_path, corpus_results_root

    repo = tmp_path
    for seed, run_name in (
        (42, "pfus1_seed42"),
        (43, "pfus1_seed43_rep2"),
        (44, "pfus1_seed44_rep3"),
    ):
        summary = tmp_path / f"summary_bladder_{seed}.csv"
        summary.write_text(
            "organ,n_slices,dice_mean,iou_mean,msd_mm_mean,hd95_mm_mean\n"
            f"Bladder,10,{0.80 + seed * 0.001:.4f},0.7,1.2,3.0\n",
            encoding="utf-8",
        )
        ingest_summary(
            corpus="bladder",
            method="bapmos_sam",
            run_name=run_name,
            summary_csv=summary,
            seed=seed,
            repo=repo,
        )

    paths = build_corpus("bladder", method="bapmos_sam", repo=repo)
    assert any(p.name == "bladder_pfus1_mean_std.csv" for p in paths)
    assert {p.name for p in paths} == {"bladder_pfus1_mean_std.csv"}
    mean_csv = corpus_results_root("bladder", repo=repo) / "combined" / "bladder_pfus1_mean_std.csv"
    text = mean_csv.read_text(encoding="utf-8")
    assert "dice_mean_pm_std" in text
    assert "msd_mm_mean_pm_std" in text
    assert "hd95_mm_mean_pm_std" in text
    assert "iou" not in text
    assert by_seed_csv_path("bladder", "bapmos_sam", "pfus1_seed42", repo=repo).is_file()


def test_compose_merges_protocol_floor():
    from bapmos.method.config_utils import compose_config

    cfg = compose_config("bapmos_outer_loop", "pooled")
    assert cfg["meta"].get("protocol_merged") is True
    assert cfg["reward"]["mode"] == "composite_fixed_clip"
    assert cfg["evaluation"]["objective_metric"] == "ptv_kfold_msd"
    assert cfg["evaluation"]["kfold_seed"] == 42
    assert cfg["training"]["loss_mode"] == "kervadec"


def test_selected_subdir_no_silent_fallback(monkeypatch):
    """Typoed BAPMOS_SELECTED_SUBDIR must not load selected/ silently."""
    from bapmos.method.config_utils import compose_config

    monkeypatch.setenv("BAPMOS_SELECTED_SUBDIR", "selected/does_not_exist")
    with pytest.raises(FileNotFoundError, match="bayesian_selected|No silent fallback"):
        compose_config("bapmos", "pooled")


def test_selected_subdir_bare_name_normalizes(monkeypatch):
    from bapmos.method.config_utils import (
        compose_config,
        normalize_selected_subdir,
        selected_hyperparameters_path,
    )

    assert normalize_selected_subdir("tpe") == "selected/tpe"
    assert normalize_selected_subdir("selected/random") == "selected/random"
    monkeypatch.setenv("BAPMOS_SELECTED_SUBDIR", "random")
    path = selected_hyperparameters_path("bapmos", "pooled")
    assert path.as_posix().endswith("selected/random/pooled.yaml")
    cfg = compose_config("bapmos", "pooled")
    assert cfg["run_root"] == "runs/prostate/bapmos/inner_loop/random"
    assert cfg["meta"]["inner_search_method"] == "random"


def test_placeholder_selected_refused_without_allow(monkeypatch):
    from bapmos.method.config_utils import compose_config

    monkeypatch.delenv("BAPMOS_ALLOW_PLACEHOLDER_SELECTED", raising=False)
    monkeypatch.setenv("BAPMOS_ALLOW_PLACEHOLDER_SELECTED", "0")
    with pytest.raises(RuntimeError, match="placeholder selected|generation"):
        compose_config("bapmos", "pooled")


def test_prostate_sam_inner_run_roots_isolated_by_search_method(monkeypatch):
    from bapmos.method.config_utils import compose_config

    monkeypatch.delenv("BAPMOS_SELECTED_SUBDIR", raising=False)
    tpe = compose_config("bapmos", "pooled")
    assert tpe["run_root"] == "runs/prostate/bapmos/inner_loop/tpe"

    monkeypatch.setenv("BAPMOS_SELECTED_SUBDIR", "selected/greedy")
    greedy = compose_config("bapmos", "pooled")
    assert greedy["run_root"] == "runs/prostate/bapmos/inner_loop/greedy"
    assert tpe["run_root"] != greedy["run_root"]


def test_enrich_wandb_outer_vs_inner_tags():
    from bapmos.method.config_utils import compose_config, enrich_wandb_run_metadata, resolve_experiment

    outer = enrich_wandb_run_metadata(
        resolve_experiment(compose_config("bapmos_outer_loop", "pooled"))
    )
    assert "hpo" in outer["wandb"]["tags"]
    assert "production" not in outer["wandb"]["tags"]

    inner = enrich_wandb_run_metadata(
        resolve_experiment(compose_config("bapmos", "pooled"), "pooled_seed42")
    )
    assert "production" in inner["wandb"]["tags"]
    assert "hpo" not in inner["wandb"]["tags"]


def test_seed_everything_sets_torch_numpy():
    import numpy as np
    import torch

    from bapmos.method.bapmos_seeds import seed_everything

    seed_everything(123)
    a = torch.rand(2)
    b = np.random.rand(2)
    seed_everything(123)
    assert torch.allclose(a, torch.rand(2))
    assert np.allclose(b, np.random.rand(2))


def test_external_baselines_reject_kervadec_loss():
    """UNet / MedSAM+box / nnU-Net path must not train with Kervadec."""
    import pytest
    import torch

    from bapmos.external_baselines.baseline_training_protocol import (
        EXTERNAL_BASELINE_LOSS_MODE,
        force_external_baseline_ce_dice_loss,
    )
    from bapmos.external_baselines.common import multiclass_ce_dice_loss
    from bapmos.losses.loss_policy import (
        BAPMOS_METHOD_LOSS_MODE,
        force_bapmos_method_kervadec_loss,
    )

    cfg = {"training": {"loss_mode": "kervadec", "kervadec": {"alpha_start": 1.0}}}
    force_external_baseline_ce_dice_loss(cfg)
    assert cfg["training"]["loss_mode"] == EXTERNAL_BASELINE_LOSS_MODE
    assert "kervadec" not in cfg["training"]

    method_cfg = {"training": {"loss_mode": "ce_dice"}}
    force_bapmos_method_kervadec_loss(method_cfg)
    assert method_cfg["training"]["loss_mode"] == BAPMOS_METHOD_LOSS_MODE
    assert "kervadec" in method_cfg["training"]

    logits = torch.zeros(1, 3, 4, 4)
    gt = torch.zeros(1, 4, 4, dtype=torch.long)
    loss = multiclass_ce_dice_loss(logits, gt, 3)
    assert torch.isfinite(loss)
    with pytest.raises(ValueError, match="CE\\+Dice|Kervadec"):
        multiclass_ce_dice_loss(logits, gt, 3, loss_mode="kervadec")


def test_legacy_fixed_ratio_path_forces_ce_dice():
    """Fixed-ratio / legacy optimization trainer must not use Kervadec."""
    from bapmos.losses.loss_policy import NON_METHOD_LOSS_MODE, force_non_method_ce_dice_loss

    cfg = {"training": {"loss_mode": "kervadec", "kervadec": {"alpha_start": 1.0}}}
    force_non_method_ce_dice_loss(cfg)
    assert cfg["training"]["loss_mode"] == NON_METHOD_LOSS_MODE
    assert "kervadec" not in cfg["training"]


def test_medsam_weight_loader_parsing_and_config_detection():
    import torch

    from bapmos.external_baselines.medsam_init.weight_loader import (
        MedsamLoadReport,
        _pick_state_dict,
        is_medsam_init_config,
    )

    flat = {"a": torch.zeros(1), "b": torch.ones(2)}
    assert _pick_state_dict(flat) is flat
    wrapped = {"state_dict": flat}
    assert _pick_state_dict(wrapped) is flat

    assert is_medsam_init_config({"baseline": "medsam_init_box_decoder_finetune"})
    assert is_medsam_init_config({"medsam_checkpoint": "models/medsam/medsam_vit_b.pth"})
    assert not is_medsam_init_config({"baseline": "unet_multiclass"})

    report = MedsamLoadReport(
        checkpoint_path="/tmp/medsam_vit_b.pth",
        missing_keys=("mask_decoder.foo",),
        unexpected_keys=(),
        model_keys=100,
        checkpoint_keys=99,
        matched_keys=99,
    )
    assert report.missing_n == 1
    assert report.overlap_ratio == 0.99
    d = report.to_dict()
    assert d["medsam_overlap_ratio"] == 0.99
    report.assert_acceptable(max_missing_keys=5, min_overlap_ratio=0.95)
    try:
        report.assert_acceptable(max_missing_keys=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for high missing count")


def test_resolve_model_checkpoint_prefers_bapmos_models():
    from bapmos.paths import project_root, resolve_model_checkpoint

    sam = resolve_model_checkpoint("models/sam_base/sam_vit_b_01ec64.pth")
    medsam = resolve_model_checkpoint("models/medsam/medsam_vit_b.pth")
    root = str(project_root().resolve() / "models")
    assert sam.is_file(), f"missing local SAM weights: {sam}"
    assert medsam.is_file(), f"missing local MedSAM weights: {medsam}"
    assert str(sam.resolve()).startswith(root)
    assert str(medsam.resolve()).startswith(root)


def test_medsam_vit_b_load_overlap_when_weights_present():
    """Integration: canonical MedSAM checkpoint should load with high overlap."""
    import pytest

    pytest.importorskip("segment_anything")
    from segment_anything import sam_model_registry

    from bapmos.external_baselines.medsam_init.weight_loader import load_medsam_weights_into_sam_vit_b
    from bapmos.paths import project_root, resolve_model_checkpoint

    medsam = resolve_model_checkpoint("models/medsam/medsam_vit_b.pth")
    sam = resolve_model_checkpoint("models/sam_base/sam_vit_b_01ec64.pth")
    assert medsam.is_file() and sam.is_file()

    model = sam_model_registry["vit_b"](checkpoint=str(sam))
    report = load_medsam_weights_into_sam_vit_b(model, str(medsam))
    assert report.missing_n <= 20, f"unexpected missing_keys={report.missing_n}"
    assert report.unexpected_n <= 20, f"unexpected unexpected_keys={report.unexpected_n}"
    assert report.overlap_ratio >= 0.95
    assert str(report.checkpoint_path).startswith(str(project_root().resolve() / "models"))


def test_should_skip_in_train_global_test_pooled():
    """Pooled prostate has no global test.txt — trainers must skip in-train test."""
    from bapmos.paths import should_skip_in_train_global_test

    assert should_skip_in_train_global_test("data/prostate/pooled", "splits_stratified") is True
    assert should_skip_in_train_global_test("data/prostate/pooled", force_skip=True) is True
    # Missing corpus / missing file → skip (safe default)
    assert should_skip_in_train_global_test("/tmp/bapmos_no_such_root", "splits_stratified") is True


def test_inference_output_paths_match_canonical_layout():
    """Stale discovery helper must agree with paths.inference_output_layout_root."""
    from bapmos.inference_output_paths import inference_output_dir
    from bapmos.paths import inference_output_layout_root, project_root

    root = project_root()
    pooled = inference_output_layout_root("data/prostate/pooled", repo_root=root)
    pfus1 = inference_output_layout_root("data/bladder/pfus1", repo_root=root)
    assert pooled == root / "inference_output" / "prostate" / "pooled"
    assert pfus1 == root / "inference_output" / "bladder" / "pfus1"
    assert inference_output_dir("data/prostate/pooled", "unet", repo_root=root) == pooled / "unet"
    assert inference_output_dir("data/bladder/pfus1", "nnunet", repo_root=root) == pfus1 / "nnunet"


def test_resolve_splits_subdir_null_and_taxonomy_defaults():
    from bapmos.inference_output.site_export import resolve_splits_subdir

    assert resolve_splits_subdir(None, None, "data/bladder/pfus1") == (
        "splits_patient_70_15_15_seed42"
    )
    assert resolve_splits_subdir(None, None, "data/prostate/pooled") == "splits_stratified"
    # JSON null / empty must not win over taxonomy default.
    assert resolve_splits_subdir(None, None, "data/bladder/pfus1") == (
        resolve_splits_subdir(None, "splits_patient_70_15_15_seed42", "data/bladder/pfus1")
    )
    assert resolve_splits_subdir(None, "custom_splits", "data/bladder/pfus1") == "custom_splits"
    assert resolve_splits_subdir("from_cfg", "from_cli", "data/bladder/pfus1") == "from_cfg"


def test_selected_study_db_paths_are_repo_relative():
    """Committed selected HP YAMLs must not embed absolute cluster home paths."""
    from pathlib import Path

    import yaml

    from bapmos.paths import project_root

    selected = list((project_root() / "experiments").rglob("selected/**/*.yaml"))
    assert selected, "expected selected/*.yaml under experiments/"
    for path in selected:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        meta = doc.get("selection_meta") or {}
        study_db = str(meta.get("study_db") or "")
        assert study_db, f"missing study_db in {path}"
        assert not study_db.startswith("/"), f"absolute study_db in {path}: {study_db}"
        assert "/home/" not in study_db and "/general/" not in study_db
        assert study_db.startswith("optuna_studies/")


def test_exports_root_ignores_hardcoded_cluster_default(monkeypatch, tmp_path):
    """Without BAP_MOS_EXPORTS_ROOT, prefer checkout exports/ (portable GitHub tree)."""
    import bapmos.paths as paths_mod

    monkeypatch.delenv("BAP_MOS_EXPORTS_ROOT", raising=False)
    monkeypatch.delenv("BAP_MOS_CLUSTER_EXPORTS", raising=False)
    monkeypatch.setattr(paths_mod, "project_root", lambda: tmp_path)
    out = paths_mod.exports_root()
    assert out == (tmp_path / "exports").resolve()


def test_medsam_test_export_does_not_import_legacy():
    """Main-path MedSAM inference must not depend on legacy.optimization."""
    import ast
    from pathlib import Path

    from bapmos.paths import project_root

    src = (
        project_root()
        / "src"
        / "bapmos"
        / "external_baselines"
        / "medsam_init"
        / "run_test_inference.py"
    )
    tree = ast.parse(src.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
    assert not any(m.startswith("bapmos.legacy") for m in imports)

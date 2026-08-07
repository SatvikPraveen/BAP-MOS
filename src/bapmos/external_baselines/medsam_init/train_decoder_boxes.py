"""
MedSAM-**initialized** SAM ViT-B decoder fine-tuning with box prompts.

This is **not** frozen MedSAM fixed-box inference: it builds SAM ViT-B, loads
MedSAM-compatible weights, then trains the mask decoder only (same protocol as
``bapmos.multiorgan.train_sam_multiorgan_decoder_box``). Paper label:
**MedSAM-init + box decoder fine-tuning**.

Requires Meta ViT-B architecture file for ``sam_model_registry`` init, then loads
MedSAM (or other compatible) weights on top.

Example:

    python -m bapmos.external_baselines.medsam_init.train_decoder_boxes \\
        --data_root data/bladder/pfus1 \\
        --sam_checkpoint models/sam_base/sam_vit_b_01ec64.pth \\
        --medsam_checkpoint models/medsam/medsam_vit_b.pth

Checkpoints default: ``runs/ExternalBaselines/medsam_init/<run_name>/``; optional
``--run_root runs/pfus1/ExternalBaselines`` for cohort-specific layouts.

Resume with ``--resume auto`` and a fixed ``--run_name``.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import wandb
from torch.utils.data import DataLoader

from bapmos.external_baselines.common import external_run_dir, seed_everything
from bapmos.external_baselines.baseline_training_protocol import (
    CLINICAL_BATCH_SIZE,
    CLINICAL_MAX_EPOCHS,
    CLINICAL_PATIENCE,
    PFUS1_BATCH_SIZE,
    PFUS1_MAX_EPOCHS,
    PFUS1_PATIENCE,
    force_external_baseline_ce_dice_loss,
)
from bapmos.checkpoint_selection import add_checkpoint_objective_cli, apply_checkpoint_objective_cli
from bapmos.evaluation.baseline_epoch_monitoring import init_external_baseline_wandb
from bapmos.external_baselines.medsam_init.weight_loader import apply_medsam_encoder_init
from bapmos.multiorgan.dataset_multi_organ import MultiOrganDataset, multi_organ_collate_fn
from bapmos.multiorgan.train_sam_multiorgan_decoder_box import (
    SAMMultiOrganTrainer,
    seed_worker as mo_seed_worker,
)
from bapmos.paths import (
    dataset_bundle_tag,
    inference_output_dir_for_checkpoint,
    project_root,
    resolve_model_checkpoint,
    resolve_training_data_root,
    resolve_under_project,
)
from bapmos.training_taxonomy import default_splits_subdir, get_baseline_taxonomy_profile


def _apply_training_cli_overrides(
    config: dict,
    *,
    max_epochs: Optional[int],
    patience: Optional[int],
) -> None:
    if max_epochs is not None:
        config["max_epochs"] = int(max_epochs)
    if patience is not None:
        config["patience"] = int(patience)


def main() -> None:
    root = project_root()
    p = argparse.ArgumentParser(description="MedSAM-init SAM multi-organ box baseline")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument(
        "--sam_checkpoint",
        type=str,
        default=None,
        help="SAM ViT-B checkpoint used to build the architecture (default: models/sam_base/...).",
    )
    p.add_argument(
        "--medsam_checkpoint",
        type=str,
        required=True,
        help="MedSAM (or compatible SAM ViT-B) state dict to load with strict=False.",
    )
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument(
        "--run_root",
        type=str,
        default=None,
        help=(
            "Parent directory for ExternalBaselines runs. Default runs/ExternalBaselines; "
            "use runs/pfus1/ExternalBaselines for PFUS1 parity with internal baselines."
        ),
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume from last_checkpoint.pth: pass a path, or 'auto' to use "
            "<run_root>/medsam_init/<run_name>/last_checkpoint.pth (requires --run_name)."
        ),
    )
    p.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Default: 100 (PFUS1) or 300 (prostate). When resuming, overrides checkpoint if set.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Default: 128 (PFUS1) or 1 (prostate).",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Default: 20 (PFUS1) or 40 (prostate). When resuming, overrides checkpoint if set.",
    )
    p.add_argument("--box_noise_pixels", type=int, default=10)
    p.add_argument("--wandb_project", type=str, default="bap-mos-medsam-init-box")
    p.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (default: WANDB_ENTITY env if set).",
    )
    p.add_argument(
        "--splits_subdir",
        type=str,
        default=None,
        help="Split folder under data_root (default: taxonomy-specific).",
    )
    p.add_argument(
        "--save-test-visualizations",
        action="store_true",
        help="After testing, save 4-panel figures under run_dir/test_results/visualizations.",
    )
    p.add_argument(
        "--test-viz-selection",
        choices=["all", "random", "worst_msd", "best_msd", "per_patient_even"],
        default="all",
    )
    p.add_argument("--test-viz-max", type=int, default=None)
    p.add_argument("--test-viz-seed", type=int, default=42)
    p.add_argument(
        "--test-only",
        action="store_true",
        help="Load best checkpoint and export test metrics only (no training).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Export test CSVs here (default: output/<bundle>/.../test from checkpoint path).",
    )
    p.add_argument(
        "--skip-test-after-train",
        action="store_true",
        help="Do not run test split export after training (e.g. pooled data with per-site tests only).",
    )
    add_checkpoint_objective_cli(p)
    args = p.parse_args()

    resume_arg = (args.resume or "").strip()
    ckpt_obj = None

    if resume_arg:
        if resume_arg.lower() == "auto":
            if not args.run_name:
                raise ValueError("--resume auto requires --run_name matching the interrupted job.")
            rr = (args.run_root or "runs/ExternalBaselines").strip()
            base = Path(rr) if Path(rr).is_absolute() else (root / rr)
            ckpt_fp = base.resolve() / "medsam_init" / args.run_name / "last_checkpoint.pth"
        else:
            ckpt_fp = resolve_under_project(resume_arg)
        if not ckpt_fp.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_fp}")
        try:
            ckpt_obj = torch.load(ckpt_fp, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt_obj = torch.load(ckpt_fp, map_location="cpu")
        config = copy.deepcopy(ckpt_obj["config"])
        _apply_training_cli_overrides(
            config,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        run_dir = Path(config["run_dir"])
        if not run_dir.is_absolute():
            run_dir = (root / run_dir).resolve()
        else:
            run_dir = run_dir.resolve()
        config["run_dir"] = str(run_dir)
        run_name = config["run_name"]
        data_root = config["data_root"]
        sam_ckpt = config["sam_checkpoint"]
        medsam_path = Path(config["medsam_checkpoint"])
        print(f"[resume] Continuing run {run_name!r} | run_dir={run_dir}")
    else:
        data_root = args.data_root or str(resolve_training_data_root("case1"))
        sam_ckpt = args.sam_checkpoint or "models/sam_base/sam_vit_b_01ec64.pth"
        sam_resolved = resolve_model_checkpoint(sam_ckpt)
        if not sam_resolved.is_file():
            raise FileNotFoundError(f"SAM checkpoint not found: {sam_ckpt!r}")
        sam_ckpt = str(sam_resolved)
        medsam_path = resolve_model_checkpoint(args.medsam_checkpoint)
        if not medsam_path.is_file():
            raise FileNotFoundError(f"MedSAM checkpoint not found: {args.medsam_checkpoint!r}")

        seed_everything(args.seed, deterministic=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tax = get_baseline_taxonomy_profile(data_root)
        splits_subdir = args.splits_subdir or default_splits_subdir(data_root)
        is_pfus1 = "pfus1" in tax.taxonomy_name
        if args.max_epochs is not None:
            max_epochs = args.max_epochs
        else:
            max_epochs = PFUS1_MAX_EPOCHS if is_pfus1 else CLINICAL_MAX_EPOCHS
        if args.patience is not None:
            patience = args.patience
        else:
            patience = PFUS1_PATIENCE if is_pfus1 else CLINICAL_PATIENCE
        if args.batch_size is not None:
            batch_size = args.batch_size
        else:
            batch_size = PFUS1_BATCH_SIZE if is_pfus1 else CLINICAL_BATCH_SIZE
        run_name = args.run_name or f"medsam_init_box_{tax.taxonomy_name}_{ts}"
        run_dir = external_run_dir("medsam_init", run_name, external_baselines_parent=args.run_root)
        run_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "baseline": "medsam_init_box_decoder_finetune",
            "baseline_note": "MedSAM weights init SAM ViT-B; decoder-only box fine-tuning (not frozen MedSAM inference).",
            "paper_label": "MedSAM-init + box decoder fine-tuning",
            "external_baselines_run_root": args.run_root,
            "data_root": str(resolve_under_project(data_root)),
            "sam_checkpoint": sam_ckpt,
            "medsam_checkpoint": str(medsam_path),
            "run_dir": str(run_dir),
            "run_name": run_name,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "lr": args.lr,
            "flat_lr": True,
            "compute_train_boundary_metrics": True,
            "seed": args.seed,
            "splits_subdir": splits_subdir,
            "distance_unit": "px" if is_pfus1 else "mm",
            "box_noise_pixels": args.box_noise_pixels,
            "patience": patience,
            "prompt_type": "box",
            "num_classes": tax.num_classes,
            "taxonomy": tax.taxonomy_name,
            "test_visualizations": {
                "enabled": args.save_test_visualizations,
                "selection": args.test_viz_selection,
                "max": args.test_viz_max,
                "seed": args.test_viz_seed,
            },
        }
        apply_checkpoint_objective_cli(config, args)

    if ckpt_obj is not None:
        seed_everything(config["seed"], deterministic=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    sam_resolved = resolve_model_checkpoint(config["sam_checkpoint"])
    if not sam_resolved.is_file():
        raise FileNotFoundError(f"SAM checkpoint not found: {config['sam_checkpoint']!r}")
    config["sam_checkpoint"] = str(sam_resolved)
    medsam_resolved = resolve_model_checkpoint(str(medsam_path))
    if not medsam_resolved.is_file():
        raise FileNotFoundError(f"MedSAM checkpoint not found: {medsam_path!r}")
    config["medsam_checkpoint"] = str(medsam_resolved)

    # External MedSAM+box baseline: regional CE+Dice only (never Kervadec).
    force_external_baseline_ce_dice_loss(config)

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    init_external_baseline_wandb(
        project=args.wandb_project,
        run_name=run_name,
        config=config,
        run_dir=run_dir,
        resumed=ckpt_obj is not None,
        wandb_entity=args.wandb_entity,
        group=f"medsam_init_box_{dataset_bundle_tag(data_root).replace('-', '_')}",
        tags=[
            f"dataset={dataset_bundle_tag(data_root)}",
            "baseline=medsam_init",
            "prompt=box",
        ],
    )

    trainer = SAMMultiOrganTrainer(config)
    overlap_report = apply_medsam_encoder_init(trainer.model, config)
    overlap_fields = overlap_report.to_dict()
    config.update(overlap_fields)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    if ckpt_obj is None:
        wandb.config.update(overlap_fields, allow_val_change=True)
    else:
        print(
            f"[resume] Re-applied MedSAM encoder init "
            f"(overlap={overlap_report.overlap_ratio:.1%}, "
            f"missing={overlap_report.missing_n}, unexpected={overlap_report.unexpected_n})"
        )

    g = torch.Generator()
    g.manual_seed(config["seed"])
    trainer._train_generator_ref = g
    if ckpt_obj is not None:
        trainer.resume_from(ckpt_obj, g)

    train_loader = DataLoader(
        MultiOrganDataset(data_root, split="train", splits_subdir=config["splits_subdir"]),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=mo_seed_worker,
        generator=g,
        collate_fn=multi_organ_collate_fn,
    )
    val_loader = DataLoader(
        MultiOrganDataset(data_root, split="val", splits_subdir=config["splits_subdir"]),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=multi_organ_collate_fn,
    )
    skip_test_after_train = bool(args.skip_test_after_train)
    from bapmos.paths import should_skip_in_train_global_test

    auto_skip = should_skip_in_train_global_test(
        data_root,
        config["splits_subdir"],
        force_skip=skip_test_after_train,
    )
    test_loader = None
    if args.test_only or not auto_skip:
        if auto_skip and args.test_only:
            from bapmos.method.data_adapter import build_test_dataloader

            test_loader = build_test_dataloader(
                data_root,
                splits_subdir=config["splits_subdir"],
                batch_size=config["batch_size"],
                num_workers=0,
            )
        elif not auto_skip:
            test_loader = DataLoader(
                MultiOrganDataset(data_root, split="test", splits_subdir=config["splits_subdir"]),
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
                collate_fn=multi_organ_collate_fn,
            )
    elif not skip_test_after_train and auto_skip:
        print(
            f"[info] Skipping in-train test eval "
            f"(no global {config['splits_subdir']}/test.txt; "
            "pooled uses site_tests/; run stratified inference_output separately)."
        )
        skip_test_after_train = True


    if args.test_only:
        if test_loader is None:
            raise ValueError("--test-only requires a test split (omit --skip-test-after-train).")
        if not resume_arg:
            raise ValueError("--test-only requires --resume auto or a checkpoint .pth path")
        out_dir = None
        if args.output_dir:
            out_dir = Path(args.output_dir).expanduser()
            if not out_dir.is_absolute():
                out_dir = (root / out_dir).resolve()
        else:
            ckpt_for_out = (
                (base / "medsam_init" / args.run_name / "best_checkpoint.pth")
                if resume_arg.lower() == "auto"
                else ckpt_fp
            )
            out_dir = inference_output_dir_for_checkpoint(
                ckpt_for_out, config["data_root"], split="test"
            )
        result = trainer.export_test_split_metrics(test_loader, output_dir=out_dir)
        print(f"[TEST] {result}")
        if out_dir:
            print(f"Metrics exported to: {out_dir}")
        wandb.finish()
        return

    trainer.train_loop(train_loader, val_loader)
    if not skip_test_after_train and test_loader is not None:
        trainer.evaluate_test_once(test_loader)
    wandb.finish()
    print(f"Done. Checkpoints: {run_dir}")


if __name__ == "__main__":
    main()

"""Protocol-constrained nnU-Net trainer (not untouched stock nnU-Net).

Aligned with the external U-Net / BAP-MOS Slurm protocol (optimizer schedule, batch,
patience, patient splits, checkpoint objective). Loss remains nnU-Net's stock Dice+CE
(``DC_and_CE_loss`` family) — **not** Kervadec (reserved for the main BAP-MOS method).

Defaults (override via env before ``nnUNetv2_train``):

- ``NNUNET_BATCH_SIZE`` — default **128** (BAP-MOS PFUS1 parity). Lower to 4–8 if OOM on full-res 2d patches.
- ``NNUNET_LR`` — default **1e-4** (flat; no poly decay)
- ``NNUNET_MAX_EPOCHS`` (100)
- ``NNUNET_PATIENCE`` (20) — early stop when validation checkpoint metric does not improve
- ``NNUNET_CHECKPOINT_OBJECTIVE`` — default ``ptv_kfold_msd`` (BAP-MOS / U-Net parity).
  For PFUS1 set ``NNUNET_CHECKPOINT_OBJECTIVE_ORGAN=Bladder``. Alias ``ema_fg_dice`` maps to
  organ-balanced MSD without requiring a named organ.

Each epoch iterates ``ceil(n_train/batch)`` train steps and ``ceil(n_val/batch)`` val steps (full cohort).
Training uses the **patient-level** train/val split from ``data/`` (via ``splits_final.json``),
not nnU-Net's default 5-fold CV. Set ``NNUNET_PATIENT_SPLIT=0`` to restore stock nnU-Net splitting.
Boundary MSD/HD95 logged to W&B every epoch via ``epoch_boundary_wandb`` (BAP-MOS metric keys).
Default: val cohort only (``NNUNET_BOUNDARY_VAL_ONLY=1``); set ``=0`` for train+val.
Predictions for boundary scoring go to a temp dir (not under the run folder) and are
deleted after metrics unless ``NNUNET_KEEP_VALIDATION_DIR=1``.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from bapmos.checkpoint_selection import (
    CheckpointObjectiveConfig,
    checkpoint_scores_from_evaluator,
    is_better_checkpoint,
)
from bapmos.external_baselines.baseline_training_protocol import (
    PFUS1_BATCH_SIZE,
    PFUS1_LR,
    PFUS1_MAX_EPOCHS,
    PFUS1_PATIENCE,
    make_constant_lr_scheduler,
    setup_wandb_metric_axes,
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _checkpoint_objective_from_env() -> CheckpointObjectiveConfig:
    metric = os.environ.get("NNUNET_CHECKPOINT_OBJECTIVE", "ptv_kfold_msd").strip()
    if metric == "ema_fg_dice":
        return CheckpointObjectiveConfig(metric="organ_balanced_msd")
    organ = os.environ.get("NNUNET_CHECKPOINT_OBJECTIVE_ORGAN", "").strip() or None
    return CheckpointObjectiveConfig(
        metric=metric,
        kfold_n=_env_int("NNUNET_CHECKPOINT_KFOLD_N", 5),
        kfold_seed=_env_int("NNUNET_CHECKPOINT_KFOLD_SEED", 42),
        objective_organ=organ,
    )


def _configure_torch_multiprocessing() -> None:
    """Use file_system tensor sharing so DDP + DA workers do not exhaust /dev/shm."""
    if os.environ.get("NNUNET_MP_FILE_SYSTEM", "1").strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return
    try:
        import torch.multiprocessing as mp

        mp.set_sharing_strategy("file_system")
    except Exception:
        pass


class nnUNetTrainerBapMosProtocol(nnUNetTrainer):
    """Full-cohort epochs, batch 128, flat lr 1e-4, patience 20; BAP-MOS external baseline."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        _configure_torch_multiprocessing()
        super().__init__(plans, configuration, fold, dataset_json, device)
        batch_size = int(self.configuration_manager.configuration["batch_size"])
        raw_bs = os.environ.get("NNUNET_BATCH_SIZE")
        if raw_bs is not None and raw_bs != "":
            batch_size = int(raw_bs)
        else:
            batch_size = PFUS1_BATCH_SIZE
        self.configuration_manager.configuration["batch_size"] = batch_size

        self.initial_lr = _env_float("NNUNET_LR", PFUS1_LR)
        self.num_epochs = _env_int("NNUNET_MAX_EPOCHS", PFUS1_MAX_EPOCHS)
        self.patience = _env_int("NNUNET_PATIENCE", PFUS1_PATIENCE)
        self._epochs_without_improvement = 0
        self._checkpoint_objective = _checkpoint_objective_from_env()
        self._bapmos_best_ptv_msd: Optional[float] = None
        self._bapmos_val_evaluator = None
        self._bapmos_warned_missing_kfold = False
        ckpt_desc = (
            f"PTV k-fold MSD (K={self._checkpoint_objective.kfold_n}, "
            f"seed={self._checkpoint_objective.kfold_seed})"
            if self._ptv_checkpoint_mode()
            else "EMA foreground Dice"
        )
        self.print_to_log_file(
            f"BapMos protocol: batch_size={batch_size} lr={self.initial_lr} "
            f"max_epochs={self.num_epochs} patience={self.patience} "
            f"(full cohort/epoch, flat lr, early stop on {ckpt_desc})",
            also_print_to_console=True,
        )

    def _ptv_checkpoint_mode(self) -> bool:
        return self._checkpoint_objective.metric == "ptv_kfold_msd"

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        lr_scheduler = make_constant_lr_scheduler(optimizer)
        return optimizer, lr_scheduler

    def _patient_split_enabled(self) -> bool:
        return os.environ.get("NNUNET_PATIENT_SPLIT", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )

    def _install_patient_splits_final(self) -> None:
        from bapmos.external_baselines.nnunet2d.epoch_boundary_wandb import (
            _resolve_case_mapping,
        )
        from bapmos.external_baselines.nnunet2d.patient_splits import (
            ensure_patient_splits_final,
        )

        dataset_id = _env_int("DATASET_ID", 503)
        dataset_name = os.environ.get("DATASET_NAME", "BapMosPfus1TrainVal")
        ds_tag = getattr(self.plans_manager, "dataset_name", "")
        if ds_tag.startswith("Dataset") and "_" in ds_tag:
            try:
                dataset_id = int(ds_tag.replace("Dataset", "").split("_")[0])
                dataset_name = ds_tag.split("_", 1)[1]
            except (ValueError, IndexError):
                pass

        case_mapping = _resolve_case_mapping(dataset_id, dataset_name)
        if case_mapping is None:
            raise FileNotFoundError(
                "Patient-level nnU-Net split requires case_mapping.json under nnUNet_raw "
                "(or BAPMOS_NNUNET_CASE_MAPPING)."
            )

        if int(self.fold) != 0:
            self.print_to_log_file(
                f"WARNING: NNUNET patient split is defined for fold 0; training fold={self.fold}.",
                also_print_to_console=True,
            )

        out = ensure_patient_splits_final(
            Path(self.preprocessed_dataset_folder_base),
            case_mapping,
            fold=int(self.fold),
        )
        with open(case_mapping, encoding="utf-8") as f:
            import json

            doc = json.load(f)
        n_tr = len(doc.get("train") or [])
        n_val = len(doc.get("val") or [])
        self.print_to_log_file(
            f"Patient-level split installed: {out} "
            f"(train={n_tr} val={n_val} cases; matches BAP-MOS/U-Net splits)",
            also_print_to_console=True,
        )

    def on_train_start(self) -> None:
        if self._patient_split_enabled():
            self._install_patient_splits_final()
        super().on_train_start()
        self._apply_full_cohort_iterations()
        # Rank-0 epoch-end boundary eval can run 30–90+ min on PFUS1 val; extend
        # NCCL timeout on every rank so the post-boundary barrier does not die at 10m.
        try:
            from bapmos.external_baselines.nnunet2d.epoch_boundary_wandb import (
                extend_ddp_timeout_for_boundary_eval,
            )

            extend_ddp_timeout_for_boundary_eval()
            self.print_to_log_file(
                "Extended DDP/NCCL timeout for boundary eval "
                f"(NNUNET_DDP_TIMEOUT_SEC={os.environ.get('NNUNET_DDP_TIMEOUT_SEC', '7200')})",
                also_print_to_console=True,
            )
        except Exception as exc:
            self.print_to_log_file(
                f"WARNING: could not extend DDP timeout: {exc}",
                also_print_to_console=True,
            )
        if os.environ.get("nnUNet_wandb_enabled", "").strip() in ("1", "true", "True"):
            try:
                setup_wandb_metric_axes()
            except Exception:
                pass

    def _apply_full_cohort_iterations(self) -> None:
        tr_keys, val_keys = self.do_split()
        batch_size = max(1, int(self.configuration_manager.configuration["batch_size"]))
        self.num_iterations_per_epoch = max(1, math.ceil(len(tr_keys) / batch_size))
        self.num_val_iterations_per_epoch = max(1, math.ceil(len(val_keys) / batch_size))
        self.print_to_log_file(
            f"Full-cohort epoch: train_iters={self.num_iterations_per_epoch} "
            f"({len(tr_keys)} cases) val_iters={self.num_val_iterations_per_epoch} "
            f"({len(val_keys)} cases) batch_size={batch_size}",
            also_print_to_console=True,
        )

    def _evaluator_organ_labels(self) -> List[str]:
        from bapmos.external_baselines.nnunet2d.bapmos_env import resolve_evaluator_organ_labels

        return resolve_evaluator_organ_labels()

    def _ptv_checkpoint_score(self) -> Optional[float]:
        evaluator = getattr(self, "_bapmos_val_evaluator", None)
        if evaluator is None or not evaluator.per_slice_metrics:
            return None
        organs = self._evaluator_organ_labels()
        if not organs:
            return None
        scores = checkpoint_scores_from_evaluator(
            evaluator, self._checkpoint_objective, organs
        )
        return scores.primary_msd

    def _restore_best_ptv_msd_if_needed(self) -> None:
        """After resume, keep the historical best so we do not clobber checkpoint_best."""
        if self._bapmos_best_ptv_msd is not None:
            return
        raw = os.environ.get("NNUNET_RESUME_BEST_PTV_MSD", "").strip()
        if raw:
            self._bapmos_best_ptv_msd = float(raw)
            self.print_to_log_file(
                f"Restored best PTV k-fold MSD={self._bapmos_best_ptv_msd} "
                "from NNUNET_RESUME_BEST_PTV_MSD",
                also_print_to_console=True,
            )
            return
        import re

        best: Optional[float] = None
        for log in sorted(Path(self.output_folder).glob("training_log*.txt")):
            try:
                text = log.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for match in re.finditer(
                r"New best PTV k-fold MSD:\s*([0-9.]+)", text
            ):
                best = float(match.group(1))
        if best is not None:
            self._bapmos_best_ptv_msd = best
            self.print_to_log_file(
                f"Restored best PTV k-fold MSD={best} from training log "
                f"(patience counter reset for resume)",
                also_print_to_console=True,
            )

    def save_checkpoint(self, filename: str) -> None:
        super().save_checkpoint(filename)
        if self.local_rank != 0 or self.disable_checkpointing:
            return
        try:
            checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
            checkpoint["_bapmos_best_ptv_msd"] = self._bapmos_best_ptv_msd
            checkpoint["_epochs_without_improvement"] = (
                self._epochs_without_improvement
            )
            torch.save(checkpoint, filename)
        except Exception as exc:
            self.print_to_log_file(
                f"WARNING: could not append BAP-MOS fields to {filename}: {exc}",
                also_print_to_console=True,
            )

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        # Parent nnUNetTrainer.load_checkpoint only binds local `checkpoint` when
        # given a filesystem path. Always call super() with a path.
        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(
                filename_or_checkpoint, map_location=self.device, weights_only=False
            )
            super().load_checkpoint(filename_or_checkpoint)
        else:
            checkpoint = filename_or_checkpoint
            fd, tmp_name = tempfile.mkstemp(suffix=".pth")
            os.close(fd)
            tmp_path = Path(tmp_name)
            try:
                torch.save(checkpoint, tmp_path)
                super().load_checkpoint(str(tmp_path))
            finally:
                tmp_path.unlink(missing_ok=True)

        stored = checkpoint.get("_bapmos_best_ptv_msd")
        if stored is not None:
            self._bapmos_best_ptv_msd = float(stored)
        # Fresh patience window on resume (do not restore exhausted counter).
        self._epochs_without_improvement = 0
        self._restore_best_ptv_msd_if_needed()

    def _ddp_broadcast_stop(self, local_stop: bool) -> bool:
        """All ranks must agree to stop — only rank 0 has PTV k-fold scores.

        Without this, rank 0 early-stops while other ranks keep training, then DDP
        hangs until NCCL timeout and Slurm records FAILED after a successful run.
        """
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return local_stop
        flag = torch.zeros(1, device=self.device, dtype=torch.int32)
        if local_stop:
            flag.fill_(1)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        return bool(int(flag.item()))

    def run_training(self) -> None:
        self.on_train_start()
        self._restore_best_ptv_msd_if_needed()
        best_score: Optional[float] = self._bapmos_best_ptv_msd
        if best_score is not None:
            self.print_to_log_file(
                f"Early-stop / checkpoint reference PTV k-fold MSD={best_score}",
                also_print_to_console=True,
            )

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for _batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for _batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

            candidate: Optional[float] = None
            local_stop = False
            if self._ptv_checkpoint_mode():
                candidate = self._ptv_checkpoint_score()
                improved = candidate is not None and (
                    best_score is None
                    or is_better_checkpoint(
                        candidate,
                        best_score,
                        min_delta=self._checkpoint_objective.min_delta,
                    )
                )
            else:
                improved = self._best_ema is not None and (
                    best_score is None or self._best_ema > best_score
                )
                if improved:
                    best_score = self._best_ema

            if self._ptv_checkpoint_mode() and improved:
                best_score = candidate

            if improved:
                self._epochs_without_improvement = 0
            elif self._ptv_checkpoint_mode() and candidate is None:
                if not self._bapmos_warned_missing_kfold:
                    self.print_to_log_file(
                        "PTV k-fold checkpoint unavailable: set BAPMOS_NNUNET_DATA_ROOT "
                        "(or BAPMOS_DATA_ROOT / PTV_DATA_ROOT) and ensure val boundary "
                        "metrics run each epoch. Early-stop patience is frozen until "
                        "a score is computed.",
                        also_print_to_console=True,
                    )
                    self._bapmos_warned_missing_kfold = True
            else:
                self._epochs_without_improvement += 1
                if self._epochs_without_improvement >= self.patience:
                    metric = (
                        "PTV k-fold MSD"
                        if self._ptv_checkpoint_mode()
                        else "EMA foreground Dice"
                    )
                    self.print_to_log_file(
                        f"Early stopping: no {metric} improvement for "
                        f"{self.patience} epochs",
                        also_print_to_console=True,
                    )
                    local_stop = True

            if self._ddp_broadcast_stop(local_stop):
                break

        self.on_train_end()

    def on_epoch_end(self) -> None:
        from bapmos.external_baselines.nnunet2d.epoch_boundary_wandb import (
            maybe_log_boundary_metrics,
        )

        dataset_id = _env_int("DATASET_ID", 503)
        dataset_name = os.environ.get("DATASET_NAME", "BapMosPfus1TrainVal")
        ds_tag = getattr(self.plans_manager, "dataset_name", "")
        if ds_tag.startswith("Dataset") and "_" in ds_tag:
            try:
                dataset_id = int(ds_tag.replace("Dataset", "").split("_")[0])
                dataset_name = ds_tag.split("_", 1)[1]
            except (ValueError, IndexError):
                pass

        maybe_log_boundary_metrics(
            self,
            epoch=self.current_epoch,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
        )

        if self._ptv_checkpoint_mode():
            self._on_epoch_end_ptv_checkpoint()
            return

        super().on_epoch_end()

    def _on_epoch_end_ptv_checkpoint(self) -> None:
        """Epoch end with PTV k-fold MSD best checkpoint (skip EMA Dice selection)."""
        self.logger.log("epoch_end_timestamps", time.time(), self.current_epoch)

        self.print_to_log_file(
            "train_loss",
            np.round(self.logger.get_value("train_losses", step=-1), decimals=4),
        )
        self.print_to_log_file(
            "val_loss",
            np.round(self.logger.get_value("val_losses", step=-1), decimals=4),
        )
        self.print_to_log_file(
            "Pseudo dice",
            [
                np.round(i, decimals=4)
                for i in self.logger.get_value("dice_per_class_or_region", step=-1)
            ],
        )
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.get_value('epoch_end_timestamps', step=-1) - self.logger.get_value('epoch_start_timestamps', step=-1), decimals=2)} s"
        )

        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (
            self.num_epochs - 1
        ):
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))

        ptv_msd = self._ptv_checkpoint_score()
        if ptv_msd is None and not self._evaluator_organ_labels():
            if not self._bapmos_warned_missing_kfold:
                self.print_to_log_file(
                    "Skipping PTV k-fold checkpoint save: no evaluator organs "
                    "(set BAPMOS_NNUNET_DATA_ROOT or PTV_DATA_ROOT).",
                    also_print_to_console=True,
                )
        elif is_better_checkpoint(
            ptv_msd,
            self._bapmos_best_ptv_msd if self._bapmos_best_ptv_msd is not None else float("inf"),
            min_delta=self._checkpoint_objective.min_delta,
        ):
            self._bapmos_best_ptv_msd = float(ptv_msd)  # type: ignore[arg-type]
            self.print_to_log_file(
                f"Yayy! New best PTV k-fold MSD: {np.round(self._bapmos_best_ptv_msd, decimals=4)}"
            )
            self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def perform_inference_on_split_keys(self, keys: List[str], folder_name: str) -> Path:
        """Sliding-window inference; PNG label maps go to a temp dir (not the run folder).

        Writing under ``output_folder/boundary_*`` filled home quota on PFUS1 (~12k
        train+val PNGs/epoch). Callers must delete the returned directory when done
        (see ``epoch_boundary_wandb`` / ``NNUNET_KEEP_VALIDATION_DIR``).
        """
        from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
        from nnunetv2.inference.export_prediction import export_prediction_from_logits
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian

        if getattr(self, "local_rank", 0) != 0:
            from bapmos.external_baselines.nnunet2d.bapmos_env import resolve_boundary_tmpdir

            return Path(
                tempfile.mkdtemp(
                    prefix=f"nnunet_{folder_name}_rank_",
                    dir=resolve_boundary_tmpdir(),
                )
            )

        self.set_deep_supervision_enabled(False)
        self.network.eval()

        # Prefer a large scratch dir (NNUNET_BOUNDARY_TMPDIR / exports tmp); node /tmp is often small.
        from bapmos.external_baselines.nnunet2d.bapmos_env import resolve_boundary_tmpdir

        output_folder = tempfile.mkdtemp(
            prefix=f"nnunet_{folder_name}_",
            dir=resolve_boundary_tmpdir(),
        )
        maybe_mkdir_p(output_folder)

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.manual_initialization(
            self.network,
            self.plans_manager,
            self.configuration_manager,
            None,
            self.dataset_json,
            self.__class__.__name__,
            self.inference_allowed_mirroring_axes,
        )

        dataset = self.dataset_class(
            self.preprocessed_dataset_folder,
            keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
        )

        for k in dataset.identifiers:
            data, _, seg_prev, properties = dataset.load_case(k)
            data = data[:]
            if self.is_cascaded:
                from nnunetv2.utilities.label_handling.label_handling import (
                    convert_labelmap_to_one_hot,
                )

                seg_prev = seg_prev[:]
                data = np.vstack(
                    (
                        data,
                        convert_labelmap_to_one_hot(
                            seg_prev,
                            self.label_manager.foreground_labels,
                            output_dtype=data.dtype,
                        ),
                    )
                )
            data_t = torch.from_numpy(data)
            output_filename_truncated = join(output_folder, k)
            prediction = predictor.predict_sliding_window_return_logits(data_t).cpu()
            export_prediction_from_logits(
                prediction,
                properties,
                self.configuration_manager,
                self.plans_manager,
                self.dataset_json,
                output_filename_truncated,
                False,
            )

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()
        return Path(output_folder)

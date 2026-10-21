"""Resuming the BAP-MOS nnU-Net trainer from checkpoint_latest reproduces an uninterrupted run.

Drives the real run_training / epoch-end checkpoint selection / save_checkpoint /
load_checkpoint with nnU-Net's network, data and logging steps stubbed and a scripted
validation score per epoch. Needs nnunetv2 (the nnU-Net environment); skipped elsewhere.
"""
from __future__ import annotations

from pathlib import Path

import pytest

nnunetv2 = pytest.importorskip("nnunetv2")
torch = pytest.importorskip("torch")

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer  # noqa: E402

from bapmos.external_baselines.nnunet2d import nnunet_trainer_bap_mos as mod  # noqa: E402

# Improves for 8 epochs, stalls 15, improves once, then stalls until patience (20) runs out.
SCORES = [10.0 - i for i in range(8)] + [3.5] * 15 + [2.5] + [2.9] * 40


class _Interrupt(Exception):
    pass


class _Logger:
    def log(self, *a, **k):
        pass

    def get_value(self, key, step=-1):
        return [0.0] if key == "dice_per_class_or_region" else 0.0

    def plot_progress_png(self, *a, **k):
        pass


class _Stub(mod.nnUNetTrainerBapMosProtocol):
    """The trainer with nnU-Net's heavy steps replaced; checkpointing and selection are real."""

    def __init__(self, out: Path, stop_after: int | None = None):  # noqa: D107 (no super())
        self.output_folder = str(out)
        self.num_epochs, self.patience, self.save_every = 100, 20, 1
        self.current_epoch, self.local_rank, self.disable_checkpointing = 0, 0, False
        self.num_iterations_per_epoch = self.num_val_iterations_per_epoch = 0
        self.device = torch.device("cpu")
        self._epochs_without_improvement = 0
        self._checkpoint_objective = mod._checkpoint_objective_from_env()
        self._bapmos_best_ptv_msd = None
        self._bapmos_val_evaluator = None
        self._bapmos_warned_missing_kfold = False
        self.logger = _Logger()
        self.stop_after = stop_after
        self.best_saved_at: list[int] = []

    # nnU-Net steps with no bearing on checkpoint selection
    def on_train_start(self):
        pass

    def on_epoch_start(self):
        if self.stop_after is not None and self.current_epoch == self.stop_after + 1:
            raise _Interrupt  # the job is killed after epoch `stop_after` was saved

    def on_train_epoch_start(self):
        pass

    def on_train_epoch_end(self, outputs):
        pass

    def on_validation_epoch_start(self):
        pass

    def on_validation_epoch_end(self, outputs):
        pass

    def on_train_end(self):
        pass

    def print_to_log_file(self, *a, **k):
        pass

    def on_epoch_end(self):
        # The epoch's validation score is stored before the epoch counter advances, as the
        # real boundary evaluator does; then the real PTV path runs (boundary logging stubbed).
        self._score = SCORES[self.current_epoch]
        self._on_epoch_end_ptv_checkpoint()

    def _ptv_checkpoint_mode(self):
        return True

    def _ptv_checkpoint_score(self):
        return self._score

    def save_checkpoint(self, filename):
        if filename.endswith("checkpoint_best.pth"):
            self.best_saved_at.append(self.current_epoch)
        super().save_checkpoint(filename)


@pytest.fixture
def fake_base_io(monkeypatch):
    """nnU-Net's own save/load: store and restore current_epoch + 1, as nnU-Net v2 does."""
    def save(self, filename):
        torch.save({"current_epoch": self.current_epoch + 1}, filename)

    def load(self, filename):
        self.current_epoch = torch.load(filename, weights_only=False)["current_epoch"]

    monkeypatch.setattr(nnUNetTrainer, "save_checkpoint", save)
    monkeypatch.setattr(nnUNetTrainer, "load_checkpoint", load)


def _run(out: Path, stop_after=None, resume=False):
    out.mkdir(exist_ok=True)
    t = _Stub(out, stop_after)
    if resume:
        t.load_checkpoint(str(out / "checkpoint_latest.pth"))
    try:
        t.run_training()
    except _Interrupt:
        pass
    return t


@pytest.mark.parametrize("stop_after", [0, 7, 8, 15, 23, 24, 35])
def test_resume_matches_uninterrupted_run(tmp_path, fake_base_io, stop_after):
    ref = _run(tmp_path / "ref")
    first = _run(tmp_path / "int", stop_after=stop_after)
    resumed = _run(tmp_path / "int", resume=True)
    assert ref.current_epoch == 24 + 20  # stops after 20 epochs without beating epoch 23
    assert resumed.current_epoch == ref.current_epoch
    assert resumed._bapmos_best_ptv_msd == ref._bapmos_best_ptv_msd == 2.5
    assert first.best_saved_at + resumed.best_saved_at == ref.best_saved_at


# The objective organ is never predicted after epoch 2: those epochs have no k-fold score.
SCORES_MISSING = [10.0, 9.0, 8.0] + [None] * 40


@pytest.mark.parametrize("stop_after", [None, 5, 21])
def test_epochs_without_a_score_count_toward_patience(tmp_path, fake_base_io, monkeypatch, stop_after):
    monkeypatch.setitem(globals(), "SCORES", SCORES_MISSING)
    t = _run(tmp_path, stop_after=stop_after)
    if stop_after is not None:
        t = _run(tmp_path, resume=True)
    assert t.current_epoch == 3 + 20  # epochs 3..22 do not improve on epoch 2
    assert t._bapmos_best_ptv_msd == 8.0


@pytest.mark.parametrize("name", ["sgd", "adam"])
def test_optimizer_choice(name):
    t = object.__new__(mod.nnUNetTrainerBapMosProtocol)
    t.network = torch.nn.Linear(4, 2)
    t.initial_lr, t.weight_decay, t.optimizer_name = 1e-4, 3e-5, name
    opt, sched = t.configure_optimizers()
    group = opt.param_groups[0]
    if name == "adam":
        assert type(opt) is torch.optim.Adam and group["weight_decay"] == 0
    else:
        assert type(opt) is torch.optim.SGD and group["nesterov"] and group["momentum"] == 0.99
    for _ in range(5):
        opt.step()
        sched.step()
    assert group["lr"] == 1e-4  # flat schedule

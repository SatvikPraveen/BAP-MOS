"""Resuming a trainer reproduces the uninterrupted run (checkpoint selection and early stopping).

Drives the real epoch loops of ``BAPMOSTrainer.train``, ``SAMMultiOrganTrainer.train_loop`` and
``UNetMultiOrganTrainer.train_loop``
(selection, save_checkpoint, resume, reconcile) with the training and validation passes stubbed:
a scripted validation objective per epoch and a random step on a tiny decoder, so the kept
checkpoints are distinguishable and the RNG restore is exercised. Each run is stopped at every
epoch and at every point inside the epoch, resumed with a fresh trainer, and must end with the
same files as the uninterrupted run.

Before the fix, last_checkpoint.pth was written before the epoch's best value and patience counter
were updated: a resume after an improving epoch forgot the improvement (a later, worse epoch could
replace best_checkpoint.pth), and a resume after any other epoch trained one extra epoch.
"""
from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segment_anything")

# Improves, stalls (fewer than PATIENCE epochs), improves once more, stalls until patience runs out.
SCORES = [0.50, 0.45, 0.40] + [0.42] * 4 + [0.38] + [0.41] * 30
PATIENCE = 5
BEST, STOP = 8, 8 + PATIENCE  # 1-based epochs
POINTS = ["after_validation", "after_commit", "before_rows", "after_rows"]


class _Interrupt(Exception):
    pass


# -- BAP-MOS trainer ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_sam_cls():
    """FrozenSAMMultiOrgan with random ViT-B weights (no checkpoint download)."""
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide

    from bapmos.method.sam_model import FrozenSAMMultiOrgan

    torch.manual_seed(0)
    sam = sam_model_registry["vit_b"](checkpoint=None)

    class _TinySAM(FrozenSAMMultiOrgan):
        def __init__(self, checkpoint, model_type="vit_b", *args, **kwargs):
            torch.nn.Module.__init__(self)
            import copy

            self.sam = copy.deepcopy(sam)
            self.img_size = self.sam.image_encoder.img_size
            self.transform = ResizeLongestSide(self.img_size)
            for p in self.sam.parameters():
                p.requires_grad = False

    return _TinySAM


def _bapmos(tmp_path, monkeypatch, tiny_sam_cls, kill=None):
    import bapmos.method.bap_mos_trainer as mod

    monkeypatch.setattr(mod, "FrozenSAMMultiOrgan", tiny_sam_cls)
    ckpt = tmp_path / "fake.pth"
    ckpt.write_bytes(b"x")
    cfg = {
        "common": {
            "data_root": "data/none",
            "sam_checkpoint": str(ckpt),
            "pixel_spacing": [1.0, 1.0],
            "max_epochs": 40,
            "patience": PATIENCE,
            "use_cached_image_embeddings": True,
            "image_embedding_dir": str(tmp_path),
        },
        "bandit": {"organs": ["bladder", "rectum"], "arms": ["box", "point"]},
        "evaluation": {"objective_metric": "organ_balanced_msd"},
        "wandb": {"enabled": False},
    }
    tr = mod.BAPMOSTrainer(cfg)
    tr.set_data_taxonomy({"bladder": 1, "rectum": 2}, 3)
    # Every fresh run starts from the same RNG state; a resume restores it from the checkpoint.
    torch.manual_seed(0)
    np.random.seed(0)
    tr.model.mask_decoder = torch.nn.Linear(4, 4)
    tr.optimizer = torch.optim.SGD(tr.model.mask_decoder.parameters(), lr=0.1)

    def train_epoch(train_loader, val_loader=None):
        with torch.no_grad():
            for p in tr.model.mask_decoder.parameters():
                p.add_(0.01 * torch.randn_like(p) + float(np.random.rand()))
        return 0.1, mod.empty_prompt_counts()

    def split_pass(trainer, loader, split, *a, **k):
        return 0.2, 0.5, {"val_msd": None}, SimpleNamespace(per_slice_metrics=[]), mod.empty_prompt_counts()

    def objective(*a, **k):
        if kill == (tr.epoch, "after_validation"):
            raise _Interrupt
        return SCORES[tr.epoch]

    tr.train_epoch = train_epoch
    monkeypatch.setattr(mod, "run_split_metrics_pass", split_pass)
    monkeypatch.setattr(mod, "validation_objective_msd", objective)
    monkeypatch.setattr(mod, "summarize_evaluator_metrics", lambda ev: {})
    monkeypatch.setattr(mod, "per_organ_wandb_dict", lambda *a, **k: {})
    monkeypatch.setattr(mod, "build_bandit_epoch_wandb_dict", lambda *a, **k: {})
    monkeypatch.setattr(mod, "build_epoch_wandb_log", lambda **k: {})
    if kill is not None:
        _arm_kill(tr, mod, monkeypatch, *kill, epoch_attr="epoch")
    return tr


def _arm_kill(tr, mod, monkeypatch, epoch, point, *, epoch_attr, log_attr=None):
    at = lambda: getattr(tr, epoch_attr) == epoch
    if point == "after_commit":  # last_checkpoint.pth written, nothing derived from it yet
        real = tr.save_checkpoint

        def save(path, *a, **k):
            real(path, *a, **k)
            if at() and path.name == "last_checkpoint.pth":
                raise _Interrupt
        tr.save_checkpoint = save
    elif point == "before_rows":  # best_checkpoint.pth written, metrics row not
        real_last = mod._last_csv_epoch
        armed = [True]  # module-level patch: fire once, not again in the resumed run

        def last(path):
            if armed[0] and at() and path.name == "metrics.csv":
                armed[0] = False
                raise _Interrupt
            return real_last(path)
        monkeypatch.setattr(mod, "_last_csv_epoch", last)
    elif point == "after_rows":  # every file written; before the early-stopping test
        armed = [True]

        def log(*a, **k):
            if armed[0] and at():
                armed[0] = False
                raise _Interrupt
        if log_attr is None:
            tr._wandb_log_epoch = log
        else:
            monkeypatch.setattr(mod, log_attr, log)


def _run_bapmos(run_dir, tr, resume=False):
    run_dir.mkdir(parents=True, exist_ok=True)
    start = 0
    if resume:
        start = tr.apply_checkpoint(torch.load(run_dir / "last_checkpoint.pth", weights_only=False))
    try:
        tr.train([], [], run_dir, skip_test=True, start_epoch=start)
    except _Interrupt:
        return False
    return True


def _state(run_dir, weights_key):
    best = torch.load(run_dir / "best_checkpoint.pth", weights_only=False)
    def flat(w):  # state dicts, possibly nested (the box trainer stores {part: state_dict})
        return [flat(v) for v in w.values()] if isinstance(w, dict) else w.tolist()

    return {
        "best_epoch_index": best["epoch_index"],
        "best_weights": flat(best[weights_key]),
        "metrics": (run_dir / "metrics.csv").read_text(),
    }


@pytest.fixture(scope="module")
def bapmos_reference(tmp_path_factory, tiny_sam_cls):
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("ref")
    assert _run_bapmos(root / "run", _bapmos(root, mp, tiny_sam_cls))
    state = _state(root / "run", "mask_decoder")
    mp.undo()
    return state


def test_bapmos_reference_selects_and_stops_as_specified(bapmos_reference):
    epochs = [int(r["epoch"]) for r in csv.DictReader(bapmos_reference["metrics"].splitlines())]
    assert epochs == list(range(1, STOP + 1)) and bapmos_reference["best_epoch_index"] == BEST - 1


@pytest.mark.parametrize("point", POINTS)
@pytest.mark.parametrize("epoch", range(STOP))
def test_bapmos_resume_equals_uninterrupted(tmp_path, monkeypatch, tiny_sam_cls, bapmos_reference, epoch, point):
    if point == "after_validation" and epoch == 0:
        pytest.skip("stopped before the first commit: no checkpoint to resume from")
    run = tmp_path / "run"
    assert not _run_bapmos(run, _bapmos(tmp_path, monkeypatch, tiny_sam_cls, kill=(epoch, point)))
    assert _run_bapmos(run, _bapmos(tmp_path, monkeypatch, tiny_sam_cls), resume=True)
    assert _state(run, "mask_decoder") == bapmos_reference


def _legacy(path, epoch, best_key, counter_key):
    """Rewrite a committed last_checkpoint.pth of 0-based ``epoch`` as the old trainers wrote it:
    best value and patience counter as before the epoch, and no committed-state keys."""
    best, counter = float("inf"), 0
    for s in SCORES[:epoch]:
        best, counter = (s, 0) if s < best else (best, counter + 1)
    ckpt = torch.load(path, weights_only=False)
    ckpt[best_key], ckpt[counter_key] = best, counter
    for key in ("best_epoch_index", "pending_row"):
        ckpt.pop(key)
    torch.save(ckpt, path)


@pytest.mark.parametrize("epoch", range(1, STOP))
def test_bapmos_legacy_checkpoint_is_reconciled(tmp_path, monkeypatch, tiny_sam_cls, bapmos_reference, epoch):
    run = tmp_path / "run"
    assert not _run_bapmos(run, _bapmos(tmp_path, monkeypatch, tiny_sam_cls, kill=(epoch, "after_rows")))
    _legacy(run / "last_checkpoint.pth", epoch, "best_val_msd_mm", "epochs_without_improvement")
    assert _run_bapmos(run, _bapmos(tmp_path, monkeypatch, tiny_sam_cls), resume=True)
    assert _state(run, "mask_decoder") == bapmos_reference


def test_bapmos_stopped_run_does_not_train_after_resume(tmp_path, monkeypatch, tiny_sam_cls, bapmos_reference):
    run = tmp_path / "run"
    assert not _run_bapmos(run, _bapmos(tmp_path, monkeypatch, tiny_sam_cls, kill=(STOP - 1, "after_rows")))
    resumed = _bapmos(tmp_path, monkeypatch, tiny_sam_cls)
    resumed.train_epoch = lambda *a: pytest.fail("trained after the stop")
    assert _run_bapmos(run, resumed, resume=True)
    assert _state(run, "mask_decoder") == bapmos_reference


# -- box-decoder trainer (SAM (box), MedSAM (box)) ---------------------------------------


class _TinySamModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = torch.nn.Linear(2, 2)
        self.image_encoder.img_size = 64
        self.prompt_encoder = torch.nn.Linear(2, 2)
        self.mask_decoder = torch.nn.Linear(4, 4)


def _box(tmp_path, monkeypatch, kill=None, run_dir=None):
    import bapmos.multiorgan.train_sam_multiorgan_decoder_box as mod

    torch.manual_seed(0)
    np.random.seed(0)
    monkeypatch.setitem(mod.sam_model_registry, "vit_b", lambda checkpoint=None: _TinySamModel())
    monkeypatch.setattr(mod, "get_baseline_taxonomy_profile",
                        lambda root: SimpleNamespace(num_classes=3, evaluator_organ_labels=["Bladder", "Rectum"]))
    monkeypatch.setattr(mod, "log_baseline_taxonomy_startup", lambda *a, **k: None)
    cfg = {
        "sam_checkpoint": None, "data_root": "data/none", "lr": 1e-3, "max_epochs": 40,
        "patience": PATIENCE, "run_dir": str(run_dir or tmp_path / "run"),
        "compute_train_boundary_metrics": False, "flat_lr": False,
    }
    tr = mod.SAMMultiOrganTrainer(cfg)

    def run_epoch(loader, train=True, return_evaluator=False):
        if train:
            with torch.no_grad():
                for p in tr.model.mask_decoder.parameters():
                    p.add_(0.01 * torch.randn_like(p) + float(np.random.rand()))
            tr.optimizer.step()
            return 0.1, 0.5, None
        return 0.2, 0.5, None, object()

    def scores(*a, **k):
        if kill == (tr.current_epoch, "after_validation"):
            raise _Interrupt
        return SimpleNamespace(primary_msd=SCORES[tr.current_epoch], organ_balanced_hd95=None,
                               organ_balanced_dice=None, ptv_hd95="", ptv_dice="")

    tr.run_epoch = run_epoch
    monkeypatch.setattr(mod, "validation_checkpoint_scores", scores)
    monkeypatch.setattr(mod, "summarize_evaluator_metrics", lambda ev: {})
    monkeypatch.setattr(mod, "per_organ_wandb_dict", lambda *a, **k: {})
    monkeypatch.setattr(mod, "build_epoch_wandb_log", lambda **k: {})
    monkeypatch.setattr(mod, "log_epoch_wandb", lambda d: None)
    if kill is not None:
        _arm_kill(tr, mod, monkeypatch, *kill, epoch_attr="current_epoch", log_attr="log_epoch_wandb")
    return tr


def _run_box(tr, resume=False):
    from pathlib import Path

    run_dir = Path(tr.cfg["run_dir"])
    if resume:
        tr.resume_from(torch.load(run_dir / "last_checkpoint.pth", weights_only=False))
    try:
        tr.train_loop([], [])
    except _Interrupt:
        return False
    return True


def _box_state(run_dir):
    state = _state(run_dir, "model_state")
    last = torch.load(run_dir / "last_checkpoint.pth", weights_only=False)
    state["scheduler"] = last["scheduler_state"]["last_epoch"]
    return state


@pytest.fixture(scope="module")
def box_reference(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("boxref")
    assert _run_box(_box(root, mp))
    state = _box_state(root / "run")
    mp.undo()
    return state


def test_box_reference_selects_and_stops_as_specified(box_reference):
    epochs = [int(r["epoch"]) for r in csv.DictReader(box_reference["metrics"].splitlines())]
    # The stopping epoch has its row (it used to be dropped).
    assert epochs == list(range(1, STOP + 1)) and box_reference["best_epoch_index"] == BEST - 1


@pytest.mark.parametrize("point", POINTS)
@pytest.mark.parametrize("epoch", range(STOP))
def test_box_resume_equals_uninterrupted(tmp_path, monkeypatch, box_reference, epoch, point):
    if point == "after_validation" and epoch == 0:
        pytest.skip("stopped before the first commit: no checkpoint to resume from")
    assert not _run_box(_box(tmp_path, monkeypatch, kill=(epoch, point)))
    assert _run_box(_box(tmp_path, monkeypatch), resume=True)
    assert _box_state(tmp_path / "run") == box_reference


@pytest.mark.parametrize("epoch", range(1, STOP))
def test_box_legacy_checkpoint_is_reconciled(tmp_path, monkeypatch, box_reference, epoch):
    assert not _run_box(_box(tmp_path, monkeypatch, kill=(epoch, "after_rows")))
    _legacy(tmp_path / "run" / "last_checkpoint.pth", epoch, "best_val_msd_tracked", "epochs_without_improvement")
    assert _run_box(_box(tmp_path, monkeypatch), resume=True)
    assert _box_state(tmp_path / "run") == box_reference


# -- U-Net external baseline --------------------------------------------------------------


def _unet(tmp_path, monkeypatch, kill=None):
    import bapmos.external_baselines.unet.train_multiclass as mod

    torch.manual_seed(0)
    np.random.seed(0)
    monkeypatch.setattr(mod.smp, "Unet", lambda **k: torch.nn.Linear(4, 4))  # no ImageNet download
    monkeypatch.setattr(mod, "get_baseline_taxonomy_profile",
                        lambda root: SimpleNamespace(num_classes=3, evaluator_organ_labels=["Bladder", "Rectum"]))
    monkeypatch.setattr(mod, "log_baseline_taxonomy_startup", lambda *a, **k: None)
    cfg = {
        "data_root": "data/none", "num_classes": 3, "lr": 1e-3, "max_epochs": 40, "patience": PATIENCE,
        "run_dir": str(tmp_path / "run"), "compute_train_boundary_metrics": False, "flat_lr": False,
    }
    tr = mod.UNetMultiOrganTrainer(cfg)

    def run_epoch(loader, train):
        if train:
            # Seeded by epoch: this trainer does not checkpoint the RNG (resumed data order is not
            # bit-identical), so the stub's step must not depend on it; selection is under test.
            g = torch.Generator().manual_seed(1000 + tr.current_epoch)
            with torch.no_grad():
                for p in tr.model.parameters():
                    p.add_(0.01 * torch.randn(p.shape, generator=g) + float(torch.rand(1, generator=g)))
            tr.optimizer.step()
            return 0.1, 0.5, None, None
        return 0.2, 0.5, {}, object()

    def scores(*a, **k):
        if kill == (tr.current_epoch, "after_validation"):
            raise _Interrupt
        return SimpleNamespace(primary_msd=SCORES[tr.current_epoch], organ_balanced_hd95=None,
                               organ_balanced_dice=None, ptv_hd95="", ptv_dice="")

    tr.run_epoch = run_epoch
    monkeypatch.setattr(mod, "validation_checkpoint_scores", scores)
    monkeypatch.setattr(mod, "per_organ_wandb_dict", lambda *a, **k: {})
    monkeypatch.setattr(mod, "build_epoch_wandb_log", lambda **k: {})
    monkeypatch.setattr(mod, "log_epoch_wandb", lambda d: None)
    if kill is not None:
        _arm_kill(tr, mod, monkeypatch, *kill, epoch_attr="current_epoch", log_attr="log_epoch_wandb")
    return tr


def _run_unet(tr, resume=False):
    from pathlib import Path

    run_dir = Path(tr.cfg["run_dir"])
    start = 0
    if resume:
        start = tr.apply_checkpoint(torch.load(run_dir / "last_checkpoint.pth", weights_only=False))
    try:
        tr.train_loop([], [], start_epoch=start)
    except _Interrupt:
        return False
    return True


def _unet_state(run_dir):
    best = torch.load(run_dir / "best_checkpoint.pth", weights_only=False)
    last = torch.load(run_dir / "last_checkpoint.pth", weights_only=False)
    return {"best_epoch_index": best["epoch"], "best_weights": [w.tolist() for w in best["model_state"].values()],
            "metrics": (run_dir / "metrics.csv").read_text(), "scheduler": last["scheduler_state"]["last_epoch"]}


@pytest.fixture(scope="module")
def unet_reference(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("unetref")
    assert _run_unet(_unet(root, mp))
    state = _unet_state(root / "run")
    mp.undo()
    return state


def test_unet_reference_selects_and_stops_as_specified(unet_reference):
    epochs = [int(r["epoch"]) for r in csv.DictReader(unet_reference["metrics"].splitlines())]
    # The stopping epoch has its row (it used to be dropped).
    assert epochs == list(range(1, STOP + 1)) and unet_reference["best_epoch_index"] == BEST - 1


@pytest.mark.parametrize("point", POINTS)
@pytest.mark.parametrize("epoch", range(STOP))
def test_unet_resume_equals_uninterrupted(tmp_path, monkeypatch, unet_reference, epoch, point):
    if point == "after_validation" and epoch == 0:
        pytest.skip("stopped before the first commit: no checkpoint to resume from")
    assert not _run_unet(_unet(tmp_path, monkeypatch, kill=(epoch, point)))
    assert _run_unet(_unet(tmp_path, monkeypatch), resume=True)
    assert _unet_state(tmp_path / "run") == unet_reference


@pytest.mark.parametrize("epoch", range(1, STOP))
def test_unet_legacy_checkpoint_is_reconciled(tmp_path, monkeypatch, unet_reference, epoch):
    assert not _run_unet(_unet(tmp_path, monkeypatch, kill=(epoch, "after_rows")))
    _legacy(tmp_path / "run" / "last_checkpoint.pth", epoch, "best_val_msd", "epochs_without_improvement")
    assert _run_unet(_unet(tmp_path, monkeypatch), resume=True)
    assert _unet_state(tmp_path / "run") == unet_reference

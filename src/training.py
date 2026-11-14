"""Shared training machinery: splitting, the optimisation loop, metrics, calibration.

The datasets of this project are small, so the loop is built to fail safe:
validation data never shares a recording passage with training data, the best
weights (by validation loss) are restored at the end, training stops once it no
longer improves, and classifier probabilities are calibrated so that the live
threshold means what it says.
"""
from __future__ import annotations

import copy
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class TrainingError(RuntimeError):
    """Raised when training cannot proceed or diverges."""


@dataclass
class TrainingHistory:
    """Per-epoch losses and the summary of a training run."""

    losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        def clean(values: Sequence[float]) -> List[Optional[float]]:
            return [None if (v is None or not math.isfinite(v)) else round(float(v), 6) for v in values]

        return {
            "losses": clean(self.losses),
            "val_losses": clean(self.val_losses),
            "best_epoch": self.best_epoch,
            "stopped_early": self.stopped_early,
            "metrics": {
                key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                for key, value in self.metrics.items()
            },
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Pick the requested device, falling back to CPU when CUDA is unavailable."""

    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA was requested but is not available; using the CPU instead.")
        return torch.device("cpu")
    return device


def grouped_split(
    groups: Sequence[str],
    labels: Optional[Sequence[int]],
    validation_split: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Split sample indices into train/validation without splitting a group.

    Windows cut from the same passage overlap; keeping each group on one side
    stops the validation score from being inflated by near-duplicates.  With
    ``labels`` the split is stratified so both sides see both classes whenever
    the data allows it.
    """

    count = len(groups)
    indices = list(range(count))
    if validation_split <= 0 or count < 4:
        return indices, []

    rng = random.Random(seed)
    by_group: Dict[str, List[int]] = {}
    for index, group in enumerate(groups):
        by_group.setdefault(str(group), []).append(index)

    if labels is None:
        strata: Dict[int, List[str]] = {0: list(by_group)}
    else:
        strata = {}
        for group, members in by_group.items():
            majority = int(round(sum(int(labels[i]) for i in members) / len(members)))
            strata.setdefault(majority, []).append(group)

    validation_groups: List[str] = []
    for _, names in sorted(strata.items()):
        names = sorted(names)
        rng.shuffle(names)
        if len(names) < 2:
            continue
        take = int(round(len(names) * validation_split))
        take = min(max(take, 1), len(names) - 1)
        validation_groups.extend(names[:take])

    chosen = set(validation_groups)
    validation = [i for i in indices if str(groups[i]) in chosen]
    training = [i for i in indices if str(groups[i]) not in chosen]
    if not validation or not training:
        return indices, []
    return training, validation


def augment_eeg(eeg: Tensor, *, noise: float = 0.1, offset: float = 0.15) -> Tensor:
    """Light augmentation of normalised band matrices.

    Adds sample noise and shifts each band by a small constant, which mimics
    the baseline being slightly off — the main source of mismatch between a
    recorded session and a live one.
    """

    if noise > 0:
        eeg = eeg + noise * torch.randn_like(eeg)
    if offset > 0:
        eeg = eeg + offset * torch.randn(eeg.size(0), eeg.size(1), 1, device=eeg.device)
    return eeg


def make_loader(
    tensors: Sequence[Tensor],
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=max(1, int(batch_size)),
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


StepFunction = Callable[[Sequence[Tensor], bool], Tuple[Tensor, Tensor]]


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    step: StepFunction,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    patience: int = 10,
    min_epochs: int = 5,
    grad_clip: float = 1.0,
    label: str = "model",
) -> Tuple[TrainingHistory, Tuple[Tensor, Tensor]]:
    """Optimise ``model`` and restore its best weights.

    ``step(batch, training)`` returns ``(loss, outputs)`` for a batch whose
    tensors are already on ``device``.  Returns the history and the
    ``(outputs, targets)`` of the evaluation split under the restored weights
    (the validation split, or the training split when there is none).
    """

    if epochs < 1:
        raise TrainingError("epochs must be at least 1")
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=learning_rate * 0.1
    )
    history = TrainingHistory()
    best_loss = math.inf
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    report_every = max(1, epochs // 10)

    def run_epoch(loader: DataLoader, training: bool) -> float:
        model.train(training)
        total, count = 0.0, 0
        for batch in loader:
            batch = [tensor.to(device) for tensor in batch]
            if training:
                optimiser.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                loss, _ = step(batch, training)
            if not torch.isfinite(loss):
                raise TrainingError(
                    f"Training of the {label} diverged (loss is not finite). "
                    "Check the input data for corrupt recordings or lower the learning rate."
                )
            if training:
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimiser.step()
            size = batch[0].size(0)
            total += float(loss.item()) * size
            count += size
        return total / max(count, 1)

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(train_loader, True)
        scheduler.step()
        history.losses.append(train_loss)

        if val_loader is None:
            history.val_losses.append(float("nan"))
            monitored = train_loss
        else:
            monitored = run_epoch(val_loader, False)
            history.val_losses.append(monitored)

        if val_loader is not None:
            if monitored < best_loss - 1e-5:
                best_loss = monitored
                best_state = copy.deepcopy(model.state_dict())
                history.best_epoch = epoch
                stale = 0
            else:
                stale += 1
        else:
            history.best_epoch = epoch

        if epoch == 1 or epoch % report_every == 0 or epoch == epochs:
            if val_loader is None:
                logger.info("  %s epoch %d/%d  loss %.4f", label, epoch, epochs, train_loss)
            else:
                logger.info(
                    "  %s epoch %d/%d  loss %.4f  validation %.4f",
                    label,
                    epoch,
                    epochs,
                    train_loss,
                    monitored,
                )

        if val_loader is not None and epoch >= min_epochs and stale >= patience:
            history.stopped_early = True
            logger.info(
                "  %s stopped at epoch %d (no improvement for %d epochs; best was epoch %d)",
                label,
                epoch,
                patience,
                history.best_epoch,
            )
            break

    if val_loader is not None:
        model.load_state_dict(best_state)

    evaluation_loader = val_loader if val_loader is not None else train_loader
    model.eval()
    outputs: List[Tensor] = []
    targets: List[Tensor] = []
    with torch.no_grad():
        for batch in evaluation_loader:
            batch = [tensor.to(device) for tensor in batch]
            _, output = step(batch, False)
            outputs.append(output.detach().cpu())
            targets.append(batch[-1].detach().cpu())
    return history, (torch.cat(outputs), torch.cat(targets))


# ---------------------------------------------------------------------------
# Classifier evaluation and calibration
# ---------------------------------------------------------------------------


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve (probability a positive outranks a negative)."""

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[labels].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def classification_metrics(logits: Tensor, labels: Tensor, temperature: float = 1.0) -> Dict[str, float]:
    scores = torch.sigmoid(logits.flatten() / temperature).numpy()
    truth = labels.flatten().numpy() > 0.5
    predicted = scores >= 0.5
    positives = truth.sum()
    negatives = (~truth).sum()
    sensitivity = float((predicted & truth).sum() / positives) if positives else float("nan")
    specificity = float((~predicted & ~truth).sum() / negatives) if negatives else float("nan")
    if positives and negatives:
        balanced = (sensitivity + specificity) / 2.0
    else:
        balanced = float("nan")
    return {
        "accuracy": float((predicted == truth).mean()) if len(truth) else float("nan"),
        "balanced_accuracy": balanced,
        "auc": roc_auc(scores, truth),
        "positives": int(positives),
        "negatives": int(negatives),
    }


def fit_temperature(logits: Tensor, labels: Tensor) -> float:
    """Temperature that makes the predicted probabilities match observed frequencies.

    Small networks trained on little data are over-confident; dividing the
    logits by a temperature above one tempers them.  The temperature is never
    allowed below one: a small validation set that happens to be classified
    perfectly must not make the model *more* certain.  Needs both classes and
    a handful of samples, otherwise the logits are left as they are.
    """

    logits = logits.flatten().double()
    labels = labels.flatten().double()
    positives = int((labels > 0.5).sum())
    negatives = int((labels <= 0.5).sum())
    if positives < 3 or negatives < 3:
        return 1.0
    best_temperature, best_loss = 1.0, math.inf
    for temperature in np.exp(np.linspace(0.0, math.log(12.0), 80)):
        loss = nn.functional.binary_cross_entropy_with_logits(logits / temperature, labels).item()
        if loss < best_loss:
            best_loss, best_temperature = loss, float(temperature)
    return best_temperature


def positive_class_weight(labels: Tensor) -> Tensor:
    """Weight that balances the two classes in the loss."""

    positives = float((labels > 0.5).sum())
    negatives = float((labels <= 0.5).sum())
    if positives == 0 or negatives == 0:
        return torch.ones(())
    return torch.tensor(min(max(negatives / positives, 0.1), 10.0))


__all__ = [
    "TrainingError",
    "TrainingHistory",
    "augment_eeg",
    "classification_metrics",
    "fit",
    "fit_temperature",
    "grouped_split",
    "make_loader",
    "positive_class_weight",
    "resolve_device",
    "roc_auc",
    "seed_everything",
]

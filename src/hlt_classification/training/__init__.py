"""Deterministic training and checkpoint-selection engines."""

from .checkpoints import (
    CHECKPOINT_SELECTOR,
    SelectionRecord,
    TRAINING_CHECKPOINT_CONTRACT,
    load_checkpoint,
)
from .engine import (
    ReplicaCacheSet,
    TrainingConfig,
    epoch_batch_plan,
    learning_rate_for_update,
    train_fixed_budget,
)

__all__ = [
    "CHECKPOINT_SELECTOR",
    "ReplicaCacheSet",
    "SelectionRecord",
    "TRAINING_CHECKPOINT_CONTRACT",
    "TrainingConfig",
    "epoch_batch_plan",
    "learning_rate_for_update",
    "load_checkpoint",
    "train_fixed_budget",
]

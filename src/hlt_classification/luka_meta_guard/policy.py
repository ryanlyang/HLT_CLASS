"""Deterministic, replayable guard decisions without access to validation."""
import math
from .contracts import INITIAL_LR, TOLERANCE, PATIENCE, MAX_ROLLBACKS


class Guard:
    def __init__(self, initial_auc):
        self._check(initial_auc)
        self.anchor_auc = initial_auc
        self.anchor_block = 0
        self.bad_blocks = self.rollbacks = 0

    @staticmethod
    def _check(value):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Nonfinite/invalid TRAIN guard AUC')

    @property
    def lr(self):
        return INITIAL_LR * .5 ** self.rollbacks

    def observe(self, block, auc):
        self._check(auc)
        previous = self.anchor_block
        action = 'keep'
        if auc > self.anchor_auc + TOLERANCE:
            self.anchor_auc, self.anchor_block, self.bad_blocks = auc, block, 0
            action = 'accept'
        elif auc < self.anchor_auc - TOLERANCE:
            self.bad_blocks += 1
            if self.bad_blocks >= PATIENCE:
                if self.rollbacks < MAX_ROLLBACKS:
                    self.rollbacks += 1
                    self.bad_blocks = 0
                    action = 'rollback'
                else:
                    action = 'budget_exhausted_continue'
        else:
            self.bad_blocks = 0
        return dict(action=action, restore_block=previous if action == 'rollback' else None,
            anchor_block=self.anchor_block, anchor_auc=self.anchor_auc, bad_blocks=self.bad_blocks,
            rollbacks=self.rollbacks, next_lr=self.lr)

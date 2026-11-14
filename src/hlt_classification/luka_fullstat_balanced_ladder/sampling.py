"""Equal class/bin probability with reproducible replacement draws."""
import numpy as np
from hlt_classification.operations.fullstat_pt_metadata import array_sha256
from .contracts import EDGES, SAMPLER_SEED, DRAWS, BATCH


class BalancedSampler:
    def __init__(self, labels, pt, *, role='train', expected_counts=None, seed=SAMPLER_SEED):
        labels, pt = np.asarray(labels), np.asarray(pt)
        if (role != 'train' or labels.ndim != 1 or pt.shape != labels.shape
                or labels.dtype.kind not in 'iu' or not len(labels)
                or np.any((labels < 0) | (labels >= 11)) or not np.isfinite(pt).all()
                or np.any(pt <= 0) or type(seed) is not int or seed < 0):
            raise ValueError('Expected finite labeled TRAIN pT population')
        self.original_rows = len(labels)
        self.indices = np.flatnonzero((pt >= EDGES[0]) & (pt < EDGES[-1]))
        self.labels = labels[self.indices].astype(np.int64)
        self.pt = pt[self.indices].astype(np.float64)
        bins = np.searchsorted(EDGES, self.pt, side='right')-1
        self.codes = self.labels*18+bins
        self.counts = np.bincount(self.codes, minlength=198).reshape(11,18)
        if np.any(self.counts == 0):
            raise ValueError('Empty class/pT cell; do not silently redistribute probability')
        if expected_counts is not None and not np.array_equal(self.counts, expected_counts):
            raise ValueError('Eligible class/pT counts differ from frozen audit')
        self.rows = np.argsort(self.codes, kind='stable')
        self.offsets = np.r_[0, np.cumsum(self.counts.ravel())]
        self.seed = seed
        for value in (self.indices,self.labels,self.pt,self.codes,self.counts,self.rows,self.offsets):
            value.flags.writeable = False

    def draw(self, epoch, *, draws=DRAWS):
        if type(epoch) is not int or epoch < 1 or type(draws) is not int or draws < 1:
            raise ValueError('Positive integer epoch/draw budget required')
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed,epoch])))
        # One uniform cell is exactly uniform class then uniform conditional bin.
        cells = rng.integers(0, 198, size=draws, dtype=np.int64)
        positions = rng.integers(0, self.counts.ravel()[cells], size=draws, dtype=np.int64)
        order = self.rows[self.offsets[cells]+positions]
        order.flags.writeable = False
        return order

    def evidence(self, order):
        order = np.asarray(order)
        if (order.ndim != 1 or order.dtype.kind not in 'iu' or not len(order)
                or np.any((order < 0) | (order >= len(self.indices)))):
            raise ValueError('Invalid eligible draw indices')
        unique = int(np.unique(order).size)
        return dict(draws=len(order), unique_jets=unique, repeated_draws=len(order)-unique,
                    class_bin_counts=np.bincount(self.codes[order],minlength=198).reshape(11,18).tolist(),
                    order_sha256=array_sha256('eligible_draw_indices',order.astype(np.int64)))

    @staticmethod
    def batches(order, *, start_update=0, batch_size=BATCH):
        if type(start_update) is not int or start_update < 0 or batch_size < 1:
            raise ValueError('Invalid batch cursor')
        for start in range(start_update*batch_size,len(order),batch_size):
            yield start, order[start:start+batch_size]

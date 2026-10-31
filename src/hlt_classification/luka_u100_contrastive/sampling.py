"""Deterministic TRAIN-only candidate selection, bounded in memory and work."""
import hashlib
import json
from copy import copy
from collections import OrderedDict
import numpy as np
import torch
from .contracts import NEGATIVES, POOL, sampler_recipe


def fit_bins(pt, count):
    pt, count = np.asarray(pt), np.asarray(count)
    if (pt.shape != count.shape or pt.ndim != 1 or not np.isfinite(pt).all()
            or not np.isfinite(count).all() or np.any(pt < 0) or np.any(count < 0)):
        raise ValueError('Invalid TRAIN HLT summaries')
    values = (np.log(np.maximum(pt, 1e-6)), count)
    edges = [np.unique(np.quantile(v, np.linspace(0, 1, n+1), method='linear'))
             for v, n in zip(values, (20, 10))]
    bins = np.stack([np.searchsorted(e[1:-1], v, side='right') for e, v in zip(edges, values)], axis=1)
    return bins, edges


class Sampler:
    def __init__(self, keys, identities, labels, *, role='train', root_seed=2026103102,
                 replicate=0, pt=None, count=None, event_groups=None, alias_groups=None, device='cpu'):
        n = len(labels)
        if (role != 'train' or keys.shape != (n, 128) or keys.dtype != np.float32
                or identities.shape != (n, 32) or identities.dtype != np.uint8
                or labels.shape != (n,) or labels.dtype != np.int64
                or not np.array_equal(np.unique(labels), np.arange(11))
                or len(np.unique(identities, axis=0)) != n or not np.isfinite(keys).all()):
            raise ValueError('Sampler requires a unique authenticated eleven-class TRAIN bank')
        norms = np.linalg.norm(keys, axis=1)
        if not np.all((norms < 1e-8) | np.isclose(norms, 1., atol=2e-6)):
            raise ValueError('OFFLINE keys must be fixed unit vectors (or finite zero vectors)')
        self.identities, self.labels = identities, labels
        self.keys = torch.from_numpy(keys).to(device)
        self.root_seed, self.replicate = root_seed, replicate
        self.rng_version = sampler_recipe()['contract']
        self.stream = 'training'
        self.classes = [np.flatnonzero(labels == c) for c in range(11)]
        canonical = np.array([bytes(row).hex() for row in identities])
        self.tie_rank = np.argsort(np.argsort(canonical, kind='stable'), kind='stable')
        self.groups = []
        for grouping in (event_groups, alias_groups):
            if grouping is not None:
                grouping = np.asarray(grouping)
                if grouping.shape != (n,) or grouping.dtype.kind not in 'iuUS':
                    raise ValueError('Malformed authenticated exclusion groups')
                index = {}
                for row, group in enumerate(grouping.tolist()):
                    index.setdefault(group, []).append(row)
                self.groups.append((grouping, index))
        self.bins, self.edges = (None, None) if pt is None else fit_bins(pt, count)
        self._regions = OrderedDict()
        self._region_bytes = 0
        # A sufficient, declared bound covers the largest 4,096/10-class proposal
        # quota, same-class draws, and all broad quotas for every anchor. It is
        # checked on the entire population, not only a miniature's anchor subset.
        counts = np.bincount(labels, minlength=11)
        worst = np.ones(11, dtype=np.int64)
        if self.groups:
            for i in range(n):
                excluded = list(self.excluded(i))
                worst = np.maximum(worst, np.bincount(labels[excluded], minlength=11))
        if np.any(counts-worst < 410) or n-int(worst.sum())-512 < POOL:
            raise ValueError('Insufficient eligible per-class capacity for all registered samplers (410 required)')
        self.capacity = dict(rows=n, class_counts=counts.tolist(), maximum_excluded_by_class=worst.tolist(),
            minimum_required_per_class=410, event_groups_available=event_groups is not None,
            alias_groups_available=alias_groups is not None, zero_keys=int(np.sum(norms < 1e-8)))

    def excluded(self, anchor):
        result = {int(anchor)}
        for grouping, index in self.groups:
            result.update(index[grouping[anchor].item()])
        return result

    def rng(self, epoch, anchor, component):
        if type(epoch) is not int or epoch < 1:
            raise ValueError('Epoch is one-based')
        data = [self.rng_version, self.root_seed, self.replicate,
                epoch, bytes(self.identities[anchor]).hex(), self.stream, component]
        seed = int.from_bytes(hashlib.sha256(json.dumps(data, separators=(',', ':')).encode()).digest(), 'little')
        return np.random.default_rng(seed)

    def diagnostic_sampler(self):
        result = copy(self)
        result.stream = 'diagnostic'
        result._regions = OrderedDict()
        result._region_bytes = 0
        return result

    @staticmethod
    def draw(pool, size, blocked, rng):
        # Uniform without replacement. Overdraw at most the number excluded;
        # filtering a random permutation prefix retains the uniform law and
        # avoids scanning the full TRAIN bank once per anchor.
        if len(pool) < size:
            raise ValueError('Candidate quota unavailable')
        # All primitive pools are sorted row indices. Ignore exclusions outside
        # this pool so, for example, the S stream is identical across mix/hard:
        # their differently sized H components contain only other classes.
        if blocked:
            values = np.asarray(sorted(blocked), dtype=np.int64)
            positions = np.searchsorted(pool, values)
            valid = positions < len(pool)
            values, positions = values[valid], positions[valid]
            blocked = set(map(int, values[pool[positions] == values]))
        if len(blocked) > len(pool)//4:
            pool = pool[~np.isin(pool, list(blocked))]
            if len(pool) < size:
                raise ValueError('Eligible candidate quota unavailable')
            return pool[rng.choice(len(pool), size, replace=False)]
        proposal = pool[rng.choice(len(pool), min(len(pool), size+len(blocked)), replace=False)]
        result = np.array([j for j in proposal if int(j) not in blocked], dtype=np.int64)[:size]
        if len(result) != size:
            raise ValueError('Eligible candidate quota unavailable')
        return result

    def balanced(self, size, blocked, rng, classes=None, pools=None):
        pools = self.classes if pools is None else pools
        classes = list(range(11)) if classes is None else list(classes)
        if not classes:
            raise ValueError('Empty eligible region')
        quota = {c: size//len(classes) for c in classes}
        for c in rng.permutation(classes)[:size % len(classes)]:
            quota[int(c)] += 1
        return np.concatenate([self.draw(pools[c], quota[c], blocked, rng) for c in classes])

    def region(self, anchor, radius):
        key = (*self.bins[anchor], radius)
        if key not in self._regions:
            rows = np.flatnonzero(np.max(np.abs(self.bins-self.bins[anchor]), axis=1) <= radius)
            self._regions[key] = [rows[self.labels[rows] == c] for c in range(11)]
            self._region_bytes += sum(p.nbytes for p in self._regions[key])
            # Cache bin regions, never a jet-by-jet matrix. Keep common regions
            # across batches without retaining an unbounded full-bank index.
            while self._region_bytes > 128*2**20 and len(self._regions) > 1:
                _, removed = self._regions.popitem(last=False)
                self._region_bytes -= sum(p.nbytes for p in removed)
        self._regions.move_to_end(key)
        return self._regions[key]

    def kinematic(self, anchor, epoch, blocked):
        if self.bins is None:
            raise ValueError('kinmix requires TRAIN HLT pT/count bins')
        maximum = int(np.max(np.ptp(self.bins, axis=0)))
        for radius in range(maximum+1):
            pools = self.region(anchor, radius)
            eligible = [len(p)-sum(1 for j in blocked if self.labels[j] == c
                                   and np.max(np.abs(self.bins[j]-self.bins[anchor])) <= radius)
                        for c, p in enumerate(pools)]
            if sum(eligible) < POOL:
                continue
            classes = [c for c in range(11) if eligible[c] > 0]
            try:
                result = self.balanced(512, blocked, self.rng(epoch, anchor, 'K'), classes, pools)
            except ValueError:
                continue
            return result, dict(radius=radius, full_region=sum(map(len, pools)) == len(self.labels))
        raise ValueError('K region cannot provide registered quotas; no fallback sampling')

    def sample(self, anchors, epoch, strategy):
        if strategy not in sampler_recipe()['strategy_counts']:
            raise ValueError('Unregistered strategy')
        anchors = np.asarray(anchors, dtype=np.int64)
        if anchors.ndim != 1 or np.any(anchors < 0) or np.any(anchors >= len(self.labels)):
            raise ValueError('Invalid anchor index')
        results, evidence = [], []
        # Limit mining temporary to 8 * 4096 * 128 FP32 values (16 MiB).
        for offset in range(0, len(anchors), 8):
            batch = anchors[offset:offset+8]
            pools, rankings = [], []
            if strategy in ('mix', 'hard', 'kinmix'):
                for i in batch:
                    pools.append(self.balanced(POOL, self.excluded(i), self.rng(epoch, i, 'H_pool'),
                                              [c for c in range(11) if c != self.labels[i]]))
                pool_ix = torch.as_tensor(np.stack(pools), device=self.keys.device)
                with torch.no_grad(), torch.autocast(self.keys.device.type, enabled=False):
                    similarity = (self.keys[pool_ix] * self.keys[torch.as_tensor(batch, device=self.keys.device), None]).sum(-1)
                for pool, values in zip(pools, similarity.cpu().numpy()):
                    rankings.append(np.lexsort((self.tie_rank[pool], -values)))
            for b, i in enumerate(batch):
                blocked = self.excluded(i); parts = {}; info = dict(anchor=int(i), components={})
                if rankings:
                    positions = self.rng(epoch, i, 'H_select').choice(np.arange(256, 1024),
                                                                   512 if strategy == 'hard' else 256, replace=False)
                    parts['H'] = pools[b][rankings[b][positions]]
                    info['hard_ranks'] = (positions+1).tolist()
                    blocked.update(map(int, parts['H']))
                    parts['S'] = self.draw(self.classes[self.labels[i]], 256, blocked, self.rng(epoch, i, 'S'))
                    blocked.update(map(int, parts['S']))
                if strategy == 'kinmix':
                    parts['K'], info['kinematic'] = self.kinematic(i, epoch, blocked)
                elif strategy == 'cross':
                    parts['X'] = self.balanced(1024, blocked, self.rng(epoch, i, 'X'),
                                               [c for c in range(11) if c != self.labels[i]])
                else:
                    size = {'random': 1024, 'mix': 512, 'hard': 256}[strategy]
                    parts['G'] = self.balanced(size, blocked, self.rng(epoch, i, 'G'))
                negative = np.concatenate(list(parts.values()))
                if (len(negative) != NEGATIVES or len(np.unique(negative)) != NEGATIVES
                        or self.excluded(i).intersection(map(int, negative))):
                    raise ValueError('Candidate identity exclusion/uniqueness failed')
                for component, ix in parts.items():
                    info['components'][component] = np.bincount(self.labels[ix], minlength=11).tolist()
                info['duplicates'] = 0
                info['excluded_count'] = len(self.excluded(i))
                info['same_class_fraction'] = float(np.mean(self.labels[negative] == self.labels[i]))
                if strategy == 'kinmix':
                    info['kinematic']['mean_bin_distance'] = float(np.max(
                        np.abs(self.bins[parts['K']]-self.bins[i]), axis=1).mean())
                results.append(np.concatenate(([i], negative))); evidence.append(info)
        return np.stack(results), evidence

    def candidate_keys(self, indices):
        return self.keys[torch.as_tensor(indices, dtype=torch.long, device=self.keys.device)].detach()

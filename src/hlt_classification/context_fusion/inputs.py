"""Identity-joined paired RAM views, sharing the unchanged single-view kernel.

The transport axis labels two views, not particle ancestry. The model unpacks
them before either encoder. No concatenated-particle model is introduced.
"""
import numpy as np
import torch

from hlt_classification.cms_proxy_ladder.cache import prepare_cache, preparation_bound
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import DzfixFusionParticleTransformer
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer


class PairedCache:
    def __init__(self, primary, context):
        if (primary.role != context.role or primary.role not in {'train', 'validation'}
                or primary.foundation_sha256 != context.foundation_sha256
                or not np.array_equal(primary.identities, context.identities)
                or not np.array_equal(primary.labels, context.labels)):
            raise ValueError('Paired cache identity/label/role join differs')
        self.primary, self.context = primary, context
        self.role, self.coordinate_name = primary.role, primary.coordinate_name
        self.foundation_sha256 = primary.foundation_sha256
        self.labels, self.identities = primary.labels, primary.identities
        self.nbytes = primary.nbytes + (0 if primary is context else context.nbytes)

    def __len__(self):
        return len(self.primary)

    def batch(self, indices):
        left, right = self.primary.batch(indices), self.context.batch(indices)
        result = {k: left[k] for k in ('labels', 'identities')}
        size = max(left['mask'].shape[-1], right['mask'].shape[-1])
        for key in ('features', 'vectors', 'mask'):
            values = [np.pad(r[key], ((0, 0), (0, 0), (0, size-r[key].shape[-1]))) for r in (left, right)]
            result[key] = np.stack(values, axis=1)
        return result


def unpack(features, vectors, mask):
    if (features.ndim != 4 or features.shape[1:3] != (2, 17)
            or vectors.shape != (features.shape[0], 2, 4, features.shape[-1])
            or mask.shape != (features.shape[0], 2, 1, features.shape[-1])):
        raise ValueError('Expected two independent 17-channel views')
    result = []
    for i in range(2):
        # Restore each native cache batch width; do not alter pair-BN populations.
        width = max(16, int(mask[:, i].sum(-1).max().item()))
        result.append(tuple(t[:, i, :, :width].contiguous() for t in (features, vectors, mask)))
    return result


class FusionModel(DzfixFusionParticleTransformer):
    def forward(self, features, vectors, mask):
        primary, context = unpack(features, vectors, mask)
        return self.forward_fused(*primary, *context, alpha=1.).logits


def new_model(node):
    torch.manual_seed(node['initialization_seed'])
    return (DelphesParticleTransformer() if node['context_coordinate'] is None else
            FusionModel(context_initialization_seed=node['context_initialization_seed']))


def budgets(parent, *, workers, memory_mb):
    # Four resident role/views plus bounded preparation temporaries. Reserve
    # 35% for model, pinned pair saves, optimizer, worker overhead and batches.
    foundation = parent['foundation']
    bounds = {r: preparation_bound(foundation, r, workers) for r in ('train', 'validation')}
    if 2*sum(bounds.values()) > .65*memory_mb*1024**2:
        raise MemoryError('Paired cache envelope exceeds registered RAM')
    return bounds


def cache(parent, node, role, *, workers, memory_mb):
    if role not in {'train', 'validation'}:
        raise PermissionError('No final-test cache capability')
    bound = budgets(parent, workers=workers, memory_mb=memory_mb)[role]
    def single(coordinate):
        return prepare_cache(parent['foundation'], foundation_root=parent['foundation_root'],
            role=role, coordinate=coordinate, workers=workers, max_ram_bytes=bound)
    primary = single(node['coordinate'])
    other = node['context_coordinate']
    if other is None:
        return primary
    context = primary if other == node['coordinate'] else single(other)
    return PairedCache(primary, context)

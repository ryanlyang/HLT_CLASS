"""Versioned synchronous restoration; old fusion factories remain unchanged."""
from copy import deepcopy
from threading import Lock

import torch

from hlt_classification.context_fusion.inputs import FusionModel
from hlt_classification.jetclass2_delphes import dzfix_fusion_model as old
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer

PAIR_STORAGE = dict(deepcopy(old.PAIR_OFFLOAD_POLICY),
    version='pair_saved_tensors_cpu_sync_v3', h2d_non_blocking=False)
_PARITY_LOCK = Lock()


def unpack(packed):
    device, values, shape, strides = packed
    if shape is None:
        return values
    return values.to(device, non_blocking=False).as_strided(shape, strides)


class SynchronousFusionModel(FusionModel):
    """Same parameter/state layout, pair population and physical storage span."""
    def _pair_embedding(self, mod, vectors, mask):
        # Call the native pair operation directly, never nest old async hooks.
        native = old.DelphesAdjacentFusionParticleTransformer._pair_embedding
        if not (self.pair_offload and self.training and torch.is_grad_enabled()
                and vectors.device.type == 'cuda'):
            return native(self, mod, vectors, mask)
        name = next(name for name, candidate in (
            ('context', self.context_mod), ('primary', self.primary_mod),
            ('cross', self.cross_pair_mod)) if candidate is mod)
        stats = self._pair_offload_stats[name]
        stats['calls'] += 1

        def pack(tensor):
            value = old._pack_pair_tensor(tensor)
            if tensor.device.type == 'cuda':
                stats['saved_cuda_tensors'] += 1
                stats['saved_cuda_bytes'] += tensor.numel() * tensor.element_size()
            return value

        def restore(packed):
            value = unpack(packed)
            if value.device.type == 'cuda':
                stats['restored_cuda_tensors'] += 1
            return value

        with torch.autograd.graph.saved_tensors_hooks(pack, restore):
            return native(self, mod, vectors, mask)


def new_model(node):
    torch.manual_seed(node['initialization_seed'])
    return (DelphesParticleTransformer() if node['context_coordinate'] is None else
            SynchronousFusionModel(context_initialization_seed=node['context_initialization_seed']))


def native_offload_parity(primary, context, *, device, bf16=False):
    """Reuse strict three-update checks with the actual production subclass.

    The reference has offload disabled. Only the parity helper's constructor is
    replaced, under a lock and restored on error. Production never patches globals.
    """
    with _PARITY_LOCK:
        original = old.DzfixFusionParticleTransformer
        try:
            old.DzfixFusionParticleTransformer = SynchronousFusionModel
            report = old.native_offload_parity(primary, context, device=device, bf16=bf16)
        finally:
            old.DzfixFusionParticleTransformer = original
    return dict(report, saved_tensor_storage=deepcopy(PAIR_STORAGE))

"""Recompute TP encoders without repeating persistent normalization updates."""
from contextlib import nullcontext
from functools import partial
from types import MethodType
import torch
from torch.utils.checkpoint import checkpoint


def block_forward(block, *args, **kwargs):
    # Call the installed class implementation; no copied attention mathematics.
    original = partial(type(block).forward, block)
    if block._bridge_checkpoint_enabled and block.training and torch.is_grad_enabled():
        return checkpoint(original, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)
    return original(*args, **kwargs)


def install_blocks(native):
    """Nested checkpoints bound the single-encoder backward peak as well."""
    for block in (*native.blocks, *native.cls_blocks):
        # Unlike the whole encoder, these blocks must be stateless: LayerNorm,
        # attention, MLP and dropout only. New mutable-buffer factories fail closed.
        if any(True for _ in block.buffers()) or 'forward' in block.__dict__:
            raise ValueError('Unexpected mutable or overridden native TP block')
        block._bridge_checkpoint_enabled = True
        block.forward = MethodType(block_forward, block)


class ReplayBuffers:
    """Reusable context: autograd.grad(retain_graph=True) may replay repeatedly.

    Training BatchNorm/SyncBatchNorm uses current batch statistics. Its running
    statistics/counters are outputs, not inputs to that normalization. Replay
    writes disposable copies; the original buffers (and their version counters)
    stay untouched. Replacing references avoids invalidating saved tensors.
    """
    def __init__(self, module):
        self.module = module
        self.saved = None

    def __enter__(self):
        if self.saved is not None or not self.module.training:
            raise RuntimeError('Invalid TP checkpoint replay context')
        self.saved = []
        try:
            for module in self.module.modules():
                for name, value in module._buffers.items():
                    if value is not None:
                        self.saved.append((module, name, value))
                        module._buffers[name] = value.detach().clone()
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        for module, name, value in self.saved:
            module._buffers[name] = value
        self.saved = None


def encode(owner, function, features, vectors, mask):
    """Same full batch/native forward; no changed weights or inference path."""
    if not owner.training or not torch.is_grad_enabled():
        return function(features, vectors, mask)
    if owner.mod.trimmer.enabled:
        raise ValueError('TP checkpointing requires the registered trim=False encoder')
    return checkpoint(function, features, vectors, mask, use_reentrant=False,
                      preserve_rng_state=True,
                      context_fn=lambda: (nullcontext(), ReplayBuffers(owner.mod)))

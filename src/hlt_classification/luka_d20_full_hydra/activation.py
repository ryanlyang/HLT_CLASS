"""Recompute native FP32 activations while preserving RNG and BN updates.

Adapted from luka_embedding_bridge_hydra/activation.py at d534fcfe;
the exact donor digest is recorded in adapted_donors.json.
"""
from contextlib import nullcontext
from functools import partial
from types import MethodType
import torch
from torch.utils.checkpoint import checkpoint
from . import contracts as c


def policy():
    return c.artifact('ACTIVATION_EXECUTION_POLICY',
        method='native_encoder_and_stateless_blocks_checkpoint',
        use_reentrant=False,preserve_rng_state=True,
        recompute_buffers='disposable_clones_restore_original_references',
        scope='all_arms_training_grad_enabled',global_batch=c.BATCH,precision='fp32',
        independent_reference='checkpointing_disabled',
        inference='native_unchanged',state_dict='native_keys_unchanged',
        accumulation=False,truncation=False)


def block_forward(block,*args,**kwargs):
    original=partial(type(block).forward,block)
    if block._d20_checkpoint_enabled and block.training and torch.is_grad_enabled():
        return checkpoint(original,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
    return original(*args,**kwargs)


def install_blocks(native):
    for block in (*native.blocks,*native.cls_blocks):
        if any(True for _ in block.buffers()) or 'forward' in block.__dict__:
            raise ValueError('Unexpected mutable or overridden native block')
        block._d20_checkpoint_enabled=True
        block.forward=MethodType(block_forward,block)


class ReplayBuffers:
    """Replay with disposable BN buffers; original values/versions stay intact."""
    def __init__(self,module):
        self.module=module;self.saved=None

    def __enter__(self):
        if self.saved is not None or not self.module.training:
            raise RuntimeError('Invalid native checkpoint replay context')
        self.saved=[]
        try:
            for module in self.module.modules():
                for name,value in module._buffers.items():
                    if value is not None:
                        self.saved.append((module,name,value))
                        module._buffers[name]=value.detach().clone()
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def __exit__(self,*exc):
        for module,name,value in self.saved:module._buffers[name]=value
        self.saved=None


def encode(owner,function,features,vectors,mask):
    if not owner.training or not torch.is_grad_enabled():return function(features,vectors,mask)
    if owner.mod.trimmer.enabled:raise ValueError('Activation replay requires trim=False')
    return checkpoint(function,features,vectors,mask,use_reentrant=False,preserve_rng_state=True,
        context_fn=lambda:(nullcontext(),ReplayBuffers(owner.mod)))

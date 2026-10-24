"""Bounded disposable safety checks inside export, never scientific initialization."""
import gc
import numpy as np
import torch
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import _forward
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_fullsim.preflight_v2 import memory_snapshot, check_headroom
from .contracts import artifact, validate, WEIGHTS, recipe
from .model import new_model
from .training import batch_loss, finite_gradients, optimizer


def counts(cache):
    return np.concatenate([np.diff(block.offsets) for block in cache.blocks])


def run(spec, tr, val, bank, device):
    raw = tr.batch(np.arange(min(4, len(tr)), dtype=np.int64))
    node = spec['nodes']['k1']
    torch.manual_seed(node['initialization_seed']); native = DelphesParticleTransformer().to(device).eval()
    model = new_model(node).to(device).eval(); model.capture = True
    with parity_backend(device), torch.no_grad():
        for bf16 in (False, True):
            expected = _forward(native, raw, device, bf16=bf16)
            actual = _forward(model, raw, device, bf16=bf16)
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    del native, model, expected, actual
    tr_lengths, val_lengths = counts(tr), counts(val)
    ix = np.full(128, int(np.argmax(tr_lengths)), dtype=np.int64)
    raw = tr.batch(ix)
    validation = val.batch(np.full(128, int(np.argmax(val_lengths)), dtype=np.int64))
    for arm in WEIGHTS:
        student = new_model(spec['nodes'][arm]).to(device).train(); student.capture = True
        opt = optimizer(student, recipe(arm)['base'])
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            loss, ce, kd, embed = batch_loss(student, raw, bank, ix, arm, device)
            head = tuple(student.mod.fc.parameters())
            expected = torch.autograd.grad(.5*ce, head, retain_graph=True)
            loss.backward(); finite_gradients(student)
            for parameter, reference in zip(head, expected):
                torch.testing.assert_close(parameter.grad, reference, rtol=0, atol=0)
            if arm == 'k0':
                if any(p.grad is not None for p in student.logit_heads.parameters()):
                    raise ValueError('Zero-control logit head receives a gradient')
            elif any(p.grad is None for p in student.logit_heads.parameters()):
                raise ValueError('Auxiliary logit head disconnected')
            if any(p.grad is None for p in student.predictors.parameters()):
                raise ValueError('Embedding head disconnected')
            opt.step()
            student.predictions = student.auxiliary_logits = None
            del loss, ce, kd, embed, head, expected, parameter, reference
        student.eval(); student.capture = False; student.predictions = student.auxiliary_logits = None
        with torch.inference_mode():
            if not torch.isfinite(_forward(student, validation, device, bf16=str(device).startswith('cuda'))).all():
                raise ValueError('Maximum-validation native output nonfinite')
        del student, opt
        gc.collect()
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
    memory = memory_snapshot(torch)
    if str(device).startswith('cuda'):
        check_headroom(memory, memory_mb=spec['memory_mb'], future_cache_bytes=0,
                       gpu_bytes=torch.cuda.get_device_properties(torch.device(device)).total_memory)
    return artifact('STARTUP', parents=dict(spec=spec['content_hash'], bundle=bank.manifest['content_hash']),
        passed=True, native_fp32_bf16_parity=True, main_head_gradient_CE_only=True,
        zero_control_disconnected=True, models_discarded=True, steps_per_arm=3, batch_size=128,
        arms=list(WEIGHTS), maximum_train=int(tr_lengths.max()), maximum_validation=int(val_lengths.max()), memory=memory)


def check(value, spec, bundle_hash):
    validate(value, 'STARTUP', dict(spec=spec['content_hash'], bundle=bundle_hash))
    if (any(value[k] is not True for k in ('passed', 'native_fp32_bf16_parity', 'main_head_gradient_CE_only',
                                         'zero_control_disconnected', 'models_discarded'))
            or value['steps_per_arm'] != 3 or value['batch_size'] != 128 or value['arms'] != list(WEIGHTS)
            or any(type(value[k]) is not int or not 1 <= value[k] <= 512 for k in ('maximum_train', 'maximum_validation'))):
        raise ValueError('Export startup acceptance differs')

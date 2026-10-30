"""Reuse the historical single predictor; add a frozen differentiable readout."""
from pathlib import Path
import math
import numpy as np
import torch
from hlt_classification.luka_embedding_aux.model import new_model, head_input
from hlt_classification.luka_u100_multiview.model import FrozenReadout
from hlt_classification.luka_fullstat_u100_embedding.training import objective as original_objective
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref


def make_readout(bank, directory, refs, device):
    if bank.manifest['checkpoint'] != refs['checkpoint'] or bank.manifest['report'] != refs['report']:
        raise ValueError('Bank and classifier teacher differ')
    validate_file_ref(refs['report'])
    state = torch.load(validate_file_ref(refs['checkpoint']), map_location='cpu', weights_only=True)
    path = validate_file_ref(bank.manifest['payload'], root=Path(directory))
    # Targets have already been authenticated by Bank; avoid a second full target allocation.
    with np.load(path, allow_pickle=False) as arrays:
        mean, scale = arrays['mean'], arrays['scale']
    return FrozenReadout(mean, scale, state['mod.fc.0.weight'], state['mod.fc.0.bias']).to(device)


def objective(logits, labels, prediction, target, readout, *, pass_coordinate):
    if not math.isfinite(pass_coordinate) or not 0 <= pass_coordinate <= 20:
        raise ValueError('Readout ramp coordinate outside 20 registered passes')
    base, ce, aux = original_objective(logits, labels, prediction, target, coefficient=5.)
    with torch.autocast(prediction.device.type, enabled=False):
        readout_logits = readout(prediction.float())
        readout_ce = torch.nn.functional.cross_entropy(readout_logits, labels.long())
    weight = min(pass_coordinate/3., 1.)
    # A zero-coordinate parity probe has exactly the historical loss graph.
    total = base if weight == 0 else base + weight*readout_ce
    if not torch.isfinite(readout_ce) or not torch.isfinite(total):
        raise ValueError('Nonfinite readout supervision')
    return total, ce, aux, readout_ce, weight


def batch_loss(student, raw, bank, ix, readout, device, position):
    from hlt_classification.jetclass2_delphes.runner import _forward
    logits = _forward(student, raw, device, bf16=str(device).startswith('cuda'))
    labels = torch.from_numpy(raw['labels']).to(device)
    targets = torch.from_numpy(bank.targets[ix]).to(device)
    return objective(logits, labels, student.prediction, targets, readout, pass_coordinate=position)

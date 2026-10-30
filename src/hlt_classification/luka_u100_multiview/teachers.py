"""New oracle factories; both expose the exact final classifier input."""
import torch
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.salience_learned_model import DelphesAdjacentFusionParticleTransformer


def new_teacher(name, node):
    torch.manual_seed(node['initialization_seed'])
    if name in ('U100', 'CONCAT'):
        return DelphesParticleTransformer()
    if name == 'FUSION':
        # The historical factory forks CPU initialization. Also preserve live CUDA
        # streams when its context seed calls torch.manual_seed internally.
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        result = DelphesAdjacentFusionParticleTransformer(
            context_initialization_seed=node['initialization_seed']+200000)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        else:
            torch.cuda.manual_seed_all(node['initialization_seed'])
        return result
    raise ValueError('Unknown teacher')


def classifier(model, name):
    fc = model.primary_mod.fc if name == 'FUSION' else model.mod.fc
    if (not isinstance(fc, torch.nn.Sequential) or len(fc) != 1
            or not isinstance(fc[0], torch.nn.Linear)
            or (fc[0].in_features, fc[0].out_features) != (128, 11)):
        raise ValueError('Teacher is not the registered 128-to-11 classifier')
    return fc


def forward(model, raw, device, *, bf16, name=None, capture=False):
    tensors = [torch.from_numpy(raw[k]).to(device) for k in ('features', 'vectors', 'mask')]
    if name == 'FUSION':
        tensors += [torch.from_numpy(raw['context_'+k]).to(device) for k in ('features', 'vectors', 'mask')]
    tapped = []
    handle = classifier(model, name).register_forward_pre_hook(lambda module, args: tapped.append(args[0])) if capture else None
    try:
        with torch.autocast(device_type=torch.device(device).type, enabled=bf16, dtype=torch.bfloat16):
            logits = model(*tensors)
    finally:
        if handle is not None:
            handle.remove()
    if capture:
        if len(tapped) != 1 or tapped[0].shape != (len(logits), 128):
            raise ValueError('Wrong teacher embedding tap')
        return logits, tapped[0]
    return logits

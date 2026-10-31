"""Full-support fusion with the existing stride-preserving synchronous offload.

Donors at 2b1f0370: luka_fullsim/alljets_fusion_sync.py and
jetclass2_delphes/{salience_learned_model,dzfix_fusion_model}.py.
Only the six-tensor calling interface differs from the packed-view donor.
"""
import torch
from hlt_classification.luka_fullsim import alljets_fusion_sync as storage
from hlt_classification.jetclass2_delphes import dzfix_fusion_model as compact
from hlt_classification.jetclass2_delphes.salience_learned_model import DelphesAdjacentFusionParticleTransformer


class FusionTeacher(storage.SynchronousFusionModel):
    forward = DelphesAdjacentFusionParticleTransformer.forward


def new_teacher(node):
    torch.manual_seed(node['initialization_seed'])
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    result = FusionTeacher(context_initialization_seed=node['initialization_seed']+200000)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    else:
        torch.cuda.manual_seed_all(node['initialization_seed'])
    return result


def parity(primary, context, device):
    # Native mask equivalence and storage equivalence are checked separately.
    # The storage helper checks logits, feature/parameter gradients, BN buffers,
    # AdamW states and updated weights through three nonzero-residual updates.
    mask = compact.native_mask_parity(primary, device=device)
    reports = [storage.native_offload_parity(primary, context, device=device, bf16=False)]
    if torch.device(device).type == 'cuda':
        reports.append(storage.native_offload_parity(primary, context, device=device, bf16=True))
    return dict(native_mask_parity=mask, storage=reports)


def check_parity(value):
    if not isinstance(value, dict) or value.get('native_mask_parity') is not True:
        raise ValueError('Missing native fusion mask parity')
    rows = value['storage']
    if not rows or rows[0]['device_type'] not in ('cpu', 'cuda'):
        raise ValueError('Missing fusion storage parity')
    device = rows[0]['device_type']
    if [r['precision'] for r in rows] != (['fp32', 'bf16'] if device == 'cuda' else ['fp32']):
        raise ValueError('Incomplete fusion precision coverage')
    for row in rows:
        if (row['passed'] is not True or row['steps'] != 3 or row['device_type'] != device
                or row['checks'] != compact.PARITY_CHECKS or row['saved_tensor_storage'] != storage.PAIR_STORAGE
                or row['parity_backend'] != compact.PARITY_BACKEND
                or row['tolerance'] != compact.PARITY_TOLERANCES[row['precision']]):
            raise ValueError('Fusion storage parity differs')
        if device == 'cuda': compact.validate_offload_stats(row['offload_stats'], calls=3)

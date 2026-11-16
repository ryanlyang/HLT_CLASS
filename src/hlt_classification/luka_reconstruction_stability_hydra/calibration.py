"""Frozen TRAIN-only temperature scale and deterministic numerical transport checks."""
from pathlib import Path
import math
import re
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.luka_next_supervision_hydra.model import Student
from hlt_classification.luka_next_supervision_hydra import reconstruction_model
from .contracts import artifact,validate,SOFT_PROTOCOL,scientific_cfg
from .identity import tensor_hash
from .readouts import fixed_indices
from .optimization import isolated_rng
from . import transport


def numerical_check():
    """Check masses, dual bound, envelope finite differences and a hard optimum."""
    from scipy.optimize import linear_sum_assignment
    matrix=torch.tensor([[.1,1.,.4],[1.,.1,.3],[.6,.5,.1]],dtype=torch.float64)
    masses=torch.ones(3,dtype=torch.float64);tau=.08
    plan,info=transport.solve(matrix,masses,tau);errors=[]
    for i,j in ((0,0),(0,1),(2,2)):
        perturb=torch.zeros_like(matrix);perturb[i,j]=1e-5
        fd=(transport.solve(matrix+perturb,masses,tau)[1]['primal']-transport.solve(matrix-perturb,masses,tau)[1]['primal'])/2e-5
        errors.append(abs(float(plan[i,j])-fd))
    row,col=linear_sum_assignment(matrix.numpy());cold,_=transport.solve(matrix,masses,.001)
    hard_error=abs(float((cold*matrix).sum())-float(matrix[row,col].sum()))
    null,null_info=transport.solve(torch.tensor([[.1,1.],[1.,.1],[.8,.2]],dtype=torch.float64),torch.tensor([1.,2.]),.1)
    result=dict(finite_difference_max=max(errors),hard_limit_abs=hard_error,full_marginal=info['marginal_residual'],
        full_gap=info['normalized_gap'],null_marginal=null_info['marginal_residual'],null_gap=null_info['normalized_gap'])
    if max(errors)>1e-4 or hard_error>1e-5:raise ValueError('Transport numerical qualification failed')
    return result


def prepare(spec,train,targets,ctx):
    if train.role!='train' or len(train)!=spec['counts']['train']:raise PermissionError('Temperature calibration requires full TRAIN')
    if 'recovery_parent' in spec:
        from .recovery import calibration
        value=calibration(spec)
        if [bytes(train.identities[i]).hex() for i in value['indices']]!=value['identities']:
            raise ValueError('Recovered calibration TRAIN identities differ')
        ctx.main(lambda:write_immutable_json(Path(spec['root'])/'setup_pascal/soft_calibration.json',value))
        return value
    ix=fixed_indices(train,SOFT_PROTOCOL['calibration_rows'],'stability-soft-calibration-v1')
    if len(ix)!=512:raise ValueError('Need 512 TRAIN calibration identities')
    def run():
        from .kernel import native,clear
        seed=SOFT_PROTOCOL['initialization_seed']
        with isolated_rng(ctx.device,seed=seed):
            model=Student('R24');model.cfg=scientific_cfg('SOFT_s7')
            # Technical seed index derives the decoder's existing isolated seed.
            # Both seeds are frozen here and the complete initialization is hashed.
            model.cfg['seed_index']=seed
            reconstruction_model.attach(model,spec['maximum']);model=model.to(ctx.device).eval();model.capture=True
            initialization=tensor_hash(model.state_dict());scales=[]
            with torch.no_grad():
                for start in range(0,len(ix),8):
                    part=ix[start:start+8];native(model,train.batch(part),ctx.device)
                    scores=model.aux(model.jet,model.particle_tokens,model.particle_mask)
                    values=targets.batch(part,1,model.cfg)[0]
                    matrices,_,_=transport.costs(scores,values)
                    for matrix in matrices:
                        if matrix is not None:scales.append(float((matrix-matrix.mean(1,keepdim=True)).abs().mean()))
                    clear(model)
            del model
        if not scales or not np.isfinite(scales).all():raise ValueError('No finite nonempty temperature calibration')
        result=artifact('SOFT_CALIBRATION',parents=dict(spec=spec['content_hash']),protocol=SOFT_PROTOCOL,
            role='train',domain='stability-soft-calibration-v1',indices=ix.tolist(),identities=[bytes(train.identities[i]).hex() for i in ix],
            initialization=initialization,native_seed=seed,decoder_seed=2026110601+1000003*seed,
            cost_scales=scales,scale=max(SOFT_PROTOCOL['scale_floor'],float(np.median(scales))),
            numerical_checks=numerical_check())
        write_immutable_json(Path(spec['root'])/'setup_pascal/soft_calibration.json',result);return result
    value=ctx.main(run);return value


def load(spec):
    if 'recovery_parent' in spec:
        from .recovery import calibration
        value=calibration(spec)
        if load_json(Path(spec['root'])/'setup_pascal/soft_calibration.json')!=value:
            raise ValueError('Recovered calibration differs')
        return value
    value=load_json(Path(spec['root'])/'setup_pascal/soft_calibration.json');validate(value,'SOFT_CALIBRATION',dict(spec=spec['content_hash']))
    scales=value['cost_scales'];seed=SOFT_PROTOCOL['initialization_seed']
    if (value['protocol']!=SOFT_PROTOCOL or value['role']!='train' or value['domain']!='stability-soft-calibration-v1'
        or len(value['indices'])!=512 or len(set(value['indices']))!=512
        or any(type(i)!=int or not 0<=i<spec['counts']['train'] for i in value['indices'])
        or len(value['identities'])!=512 or len(set(value['identities']))!=512
        or any(not re.fullmatch('[0-9a-f]{64}',v) for v in value['identities'])
        or not re.fullmatch('[0-9a-f]{64}',value['initialization'])
        or not scales or len(scales)>512 or not np.isfinite(scales).all() or min(scales)<0
        or value['scale']!=max(SOFT_PROTOCOL['scale_floor'],float(np.median(scales)))
        or value['native_seed']!=seed or value['decoder_seed']!=2026110601+1000003*seed):raise ValueError('Temperature calibration differs')
    checks=value['numerical_checks']
    if set(checks)!={'finite_difference_max','hard_limit_abs','full_marginal','full_gap','null_marginal','null_gap'} or any(
        not math.isfinite(v) or v<0 or v>(1e-5 if n=='hard_limit_abs' else 1e-4) for n,v in checks.items()):raise ValueError('Transport qualification differs')
    return value

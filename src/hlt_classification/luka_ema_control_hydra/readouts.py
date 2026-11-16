"""Native-only EMA parameters and TRAIN-only BatchNorm calibration on copies."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from .contracts import artifact,validate,READOUTS,TRAIN_ROWS
from .optimization import isolated_rng


def check_parent_membership(spec,value):
    from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
    validate_file_ref(spec['reused_membership'])
    old=load_json(spec['reused_membership']['path'])
    if any(value[k]!=old[k] for k in ('rows','domain','role','indices','identities','global_batch','dropout')):
        raise ValueError('Parent TRAIN BatchNorm membership differs')


def fixed_indices(train,count,domain):
    if train.role!='train':raise PermissionError('Calibration membership is TRAIN-only')
    ids=[bytes(v) for v in train.identities]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate calibration identities')
    return np.asarray(sorted(range(len(ids)),key=lambda i:(hashlib.sha256(domain.encode()+b'\0'+ids[i]).digest(),ids[i]))[:count],np.int64)


def membership(spec,train,*,technical=False):
    ix=fixed_indices(train,min(8192,len(train)),'stability-bn-calibration-v1')
    value=artifact('BN_MEMBERSHIP',parents=dict(spec=spec['content_hash']),rows=len(ix),
        domain='stability-bn-calibration-v1',role='train',indices=ix.tolist(),
        identities=[bytes(train.identities[i]).hex() for i in ix],global_batch=128,dropout=False)
    if not technical and (len(ix)!=8192 or len(train)!=spec['counts']['train']):raise ValueError('BN calibration population differs')
    return value


def save_membership(spec,train,ctx):
    value=membership(spec,train)
    check_parent_membership(spec,value)
    ctx.main(lambda:write_immutable_json(Path(spec['root'])/'setup_a6000/bn_membership.json',value))
    return value


def load_membership(spec,train,*,technical=False):
    value=membership(spec,train,technical=technical)
    if not technical:
        stored=load_json(Path(spec['root'])/'setup_a6000/bn_membership.json')
        if stored!=value:raise ValueError('Frozen BN membership differs')
    return value


class Averages:
    def __init__(self):self.values={}

    @torch.no_grad()
    def update(self,model,position,examples):
        if position<3.:return
        parameters={n:p.detach() for n,p in model.named_parameters() if n.startswith('mod.') and p.is_floating_point()}
        if not parameters:raise ValueError('No native parameters to average')
        for name,half_life in (('EMA05',.5),('EMA10',1.)):
            if name not in self.values:self.values[name]={n:v.clone() for n,v in parameters.items()};continue
            decay=2.**(-examples/(half_life*TRAIN_ROWS))
            for n,p in parameters.items():self.values[name][n].lerp_(p,1.-decay)

    def state(self,model,name):
        state=model.deployment_state()
        if name.startswith('EMA'):
            if name not in self.values:return None
            for n,p in self.values[name].items():state[n]=p.detach().cpu().clone()
        return state


def native_copy(state,ctx,factory):
    with isolated_rng(ctx.device,seed=2026111502):model=factory()
    if ctx.world>1:model=nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model=model.to(ctx.device);model.load_state_dict(state,strict=True);return model.eval()


@torch.no_grad()
def recalibrate(model,train,manifest,ctx,forward):
    if train.role!='train' or manifest['role']!='train':raise PermissionError('BN calibration requires TRAIN')
    indices=np.asarray(manifest['indices'],np.int64)
    if [bytes(train.identities[i]).hex() for i in indices]!=manifest['identities']:raise ValueError('BN calibration identities changed')
    model.eval();normalizers=[]
    for module in model.modules():
        if isinstance(module,nn.modules.batchnorm._BatchNorm):
            normalizers.append((module,module.momentum));module.reset_running_stats();module.momentum=None;module.train()
        elif isinstance(module,nn.Dropout):module.eval()
    try:
        with isolated_rng(ctx.device,seed=2026111503):
            for start in range(0,len(indices),128):
                ix=indices[start:start+128];local=np.array_split(np.arange(len(ix)),ctx.world)[ctx.rank]
                if not len(local):raise ValueError('Calibration needs nonempty local shard')
                raw=train.batch(ix);forward(model,{n:v[local] for n,v in raw.items()},ctx.device)
    finally:
        for module,momentum in normalizers:module.momentum=momentum;module.eval()
        model.eval()


def evaluate_all(model,averages,train,val,manifest,ctx,evaluate,forward,factory):
    """Return metrics, aligned predictions and native states; never mutate live buffers."""
    result={};buffers={n:v.clone() for n,v in model.named_buffers()}
    rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state(ctx.device).clone() if ctx.device.type=='cuda' else None
    with isolated_rng(ctx.device,seed=2026111504):
        metrics,probabilities=evaluate(model,val,ctx)
        result['RAW']=(metrics,probabilities,model.deployment_state())
        for name in READOUTS[1:]:
            state=averages.state(model,name)
            if state is None:continue
            copied=native_copy(state,ctx,factory)
            recalibrate(copied,train,manifest,ctx,forward)
            metrics,probabilities=evaluate(copied,val,ctx)
            state={n:v.detach().cpu().clone() for n,v in copied.state_dict().items()}
            result[name]=(metrics,probabilities,state);del copied
    if any(not torch.equal(v,buffers[n]) for n,v in model.named_buffers()):raise ValueError('Readout changed live buffers')
    if not torch.equal(rng,torch.get_rng_state()) or (cuda_rng is not None and not torch.equal(cuda_rng,torch.cuda.get_rng_state(ctx.device))):
        raise ValueError('Readout changed training RNG')
    return result

"""RAW primary; copy-only EMA05 retained exclusively for outgoing teachers."""
import torch
from hlt_classification.luka_reconstruction_stability_hydra.readouts import native_copy,recalibrate
from hlt_classification.luka_reconstruction_stability_hydra.optimization import isolated_rng
from .sampling import BalancedSampler
from . import contracts as c


class Averages:
    def __init__(self,task):self.values={};self.enabled='EMA05' in c.readouts(task)

    @torch.no_grad()
    def update(self,net,position,examples):
        if not self.enabled or position<3:return
        parameters={k:p.detach() for k,p in net.named_parameters() if k.startswith('mod.') and p.is_floating_point()}
        if not parameters:raise ValueError('Missing native EMA parameters')
        if not self.values:self.values['EMA05']={k:p.clone() for k,p in parameters.items()};return
        decay=2**(-examples/(.5*c.COUNTS['train']))
        for k,p in parameters.items():self.values['EMA05'][k].lerp_(p,1-decay)

    def restore(self,values,device):
        if set(values)-({'EMA05'} if self.enabled else set()):raise ValueError('Unexpected teacher EMA state')
        self.values={n:{k:v.to(device) for k,v in row.items()} for n,row in values.items()}


def membership(train,pt,population_hash):
    sampler=BalancedSampler(train.labels,pt,seed=2026112427)
    indices=sampler.draw(1,draws=c.BN_ROWS)
    return c.artifact('BN_MEMBERSHIP',parents=dict(population=population_hash),role='train',
        indices=indices.tolist(),identities=[bytes(train.identities[i]).hex() for i in indices],
        global_batch=128,dropout=False,sampling='fixed_balanced_TRAIN_draws_with_replacement')


def evaluate_all(net,averages,train,val,manifest,ctx,evaluate,forward,factory,*,task):
    buffers={k:v.clone() for k,v in net.named_buffers()};result={}
    with isolated_rng(ctx.device,seed=2026112404):
        metrics,p=evaluate(net,val,ctx);result['RAW']=(metrics,p,net.deployment_state())
        if 'EMA05' in c.readouts(task) and averages.values:
            state=net.deployment_state();state.update({k:v.detach().cpu().clone() for k,v in averages.values['EMA05'].items()})
            copied=native_copy(state,ctx,factory);recalibrate(copied,train,manifest,ctx,forward)
            metrics,p=evaluate(copied,val,ctx)
            result['EMA05']=(metrics,p,{k:v.detach().cpu().clone() for k,v in copied.state_dict().items()})
            del copied
    if any(not torch.equal(v,buffers[k]) for k,v in net.named_buffers()):raise ValueError('Readout changed live buffers')
    return result

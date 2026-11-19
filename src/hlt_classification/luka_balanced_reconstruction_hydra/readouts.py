"""Draw-clock native EMA and balanced TRAIN-only BN recalibration on copies."""
import numpy as np
import torch
from hlt_classification.luka_reconstruction_stability_hydra.readouts import evaluate_all
from hlt_classification.luka_sixclass_balanced_oscar.sampling import BalancedSampler
from . import contracts as c


class Averages:
    def __init__(self):self.values={}

    @torch.no_grad()
    def update(self,net,position,examples):
        if position<3:return
        ps={k:p.detach() for k,p in net.named_parameters() if k.startswith('mod.') and p.is_floating_point()}
        if not ps:raise ValueError('Missing native EMA parameters')
        for name,half in [('EMA05',.5),('EMA10',1.)]:
            if name not in self.values:self.values[name]={k:p.clone() for k,p in ps.items()};continue
            decay=2**(-examples/(half*c.COUNTS['train']))
            for k,p in ps.items():self.values[name][k].lerp_(p,1-decay)

    def state(self,net,name):
        state=net.deployment_state()
        if name.startswith('EMA'):
            if name not in self.values:return None
            state.update({k:p.detach().cpu().clone() for k,p in self.values[name].items()})
        return state

    def restore(self,state,device):
        if set(state)-{'EMA05','EMA10'}:raise ValueError('Unexpected EMA state')
        self.values={n:{k:v.to(device) for k,v in values.items()} for n,values in state.items()}


def membership(train,pt,population_hash):
    if train.role!='train':raise PermissionError('TRAIN-only BN calibration')
    sampler=BalancedSampler(train.labels,pt,seed=2026111927)
    ix=sampler.draw(1,draws=c.BN_ROWS)
    return c.artifact('BN_MEMBERSHIP',parents=dict(population=population_hash),role='train',
        sampling='fixed_balanced_TRAIN_draws_with_replacement',indices=ix.tolist(),
        identities=[bytes(train.identities[i]).hex() for i in ix],global_batch=128,dropout=False)

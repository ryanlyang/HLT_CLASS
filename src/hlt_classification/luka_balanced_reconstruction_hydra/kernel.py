"""Four-rank FP32, global128, exact chunked decoder VJPs and CE-only controls."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from hlt_classification.luka_r24_curriculum_hydra.kernel import configure, reduce_gradients, disable_dropout
from hlt_classification.luka_reconstruction_stability_hydra.optimization import isolated_rng
from hlt_classification.luka_next_supervision_hydra.reconstruction_training import decode_backward
from hlt_classification.luka_fullstat_readout.training import optimizer
from hlt_classification.luka_fullstat.recipe import learning_rate
from . import contracts as c, model


def build(task,maximum,ctx,state=None,*,parity=False):
    net=model.fresh(task,maximum)
    if state is not None:net.load_state_dict({k:state[k] for k in net.state_dict()},strict=True)
    if parity:disable_dropout(net)
    if ctx.world>1:net=nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net=net.to(ctx.device);opt=optimizer(net,c.recipe()['optimizer'])
    torch.manual_seed(c.seeds(task)['dropout']+ctx.rank)
    return net,opt


def native(net,raw,device):
    if set(raw)-{'features','vectors','mask','identities','labels'}:raise PermissionError('Undeclared native input')
    with torch.autocast(torch.device(device).type,enabled=False):
        logits=net(**{k:torch.as_tensor(raw[k],device=device) for k in ('features','vectors','mask')})
    if logits.shape!=(len(raw['labels']),6) or not torch.isfinite(logits).all():raise ValueError('Invalid native outputs')
    return logits.float()


def update(net,opt,raw,embedding,probabilities,targets,task,ctx,position):
    x=c.cell(task);n=len(raw['labels'])
    if not 0<n<=c.BATCH or ctx.world>n:raise ValueError('Invalid global batch/shards')
    if not x['embedding_weight'] and embedding is not None:raise PermissionError('Embedding target forbidden for this arm')
    if not x['main_kd_weight'] and probabilities is not None:raise PermissionError('KD target forbidden for this arm')
    if not x['reconstruction'] and targets is not None:raise PermissionError('Reconstruction target forbidden for this arm')
    if bool(x['embedding_weight'])!=(embedding is not None) or bool(x['main_kd_weight'])!=(probabilities is not None):
        raise ValueError('Missing registered teacher targets')
    ix=np.array_split(np.arange(n),ctx.world)[ctx.rank];fraction=len(ix)/n
    local={k:v[ix] for k,v in raw.items()};net.train();net.capture=bool(x['embedding_weight']);opt.zero_grad(set_to_none=True)
    for group in opt.param_groups:group['lr']=learning_rate(position)
    logits=native(net,local,ctx.device);ce=F.cross_entropy(logits,torch.as_tensor(local['labels'],device=ctx.device).long())
    emb=ce.new_zeros(());kd=ce.new_zeros(());base=x['ce_weight']*ce
    if embedding is not None:
        target=torch.as_tensor(embedding[ix],device=ctx.device).detach().float()
        if target.shape!=net.prediction.shape or not torch.isfinite(target).all():raise ValueError('Invalid embedding targets')
        emb=F.smooth_l1_loss(net.prediction,target);base=base+5*emb
    if probabilities is not None:
        q=torch.as_tensor(probabilities[ix],device=ctx.device).detach().float()
        if q.shape!=logits.shape or not torch.isfinite(q).all() or (q<0).any() or not torch.allclose(q.sum(1),q.new_ones(len(q)),atol=2e-6,rtol=0):
            raise ValueError('Invalid already-softened T2 probabilities')
        kd=4*F.kl_div(F.log_softmax(logits/2,dim=-1),q,reduction='batchmean');base=base+.75*kd
    ew,dw=c.weights(task,position);aux=0.;gh=gt=None;parts={}
    if x['reconstruction']:
        if targets is None or len(targets)!=n:raise ValueError('Missing full TRAIN sets')
        if dw>0:
            with isolated_rng(ctx.device,seed=c.seeds(task)['decoder_dropout']):
                aux,parts,gh,gt,_=decode_backward(net.aux,net.jet,net.particle_tokens,net.particle_mask,
                    [targets[i] for i in ix],dw*fraction,chunk=8,consistency=False)
            gh=gh*(ew/dw);gt=gt*(ew/dw)
        else:
            # The donor's zero-weight backward creates zero decoder gradients:
            # AdamW still advances its moments/step and applies weight decay.
            # Preserve that bookkeeping while omitting the expensive zero VJP.
            for parameter in net.aux.parameters():parameter.grad=torch.zeros_like(parameter)
    outputs=[base*fraction];grads=[torch.ones_like(base)]
    if gh is not None:outputs.extend([net.jet,net.particle_tokens]);grads.extend([gh,gt])
    torch.autograd.backward(outputs,grads);reduce_gradients(net,ctx.world)
    if any(not torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None):raise ValueError('Nonfinite gradients')
    if not x['embedding_weight'] and any(p.grad is not None for p in net.predictor.parameters()):
        raise ValueError('CE/KD-only predictor received gradients')
    opt.step()
    values=torch.tensor([float(ce.detach()),float(kd.detach()),float(emb.detach()),aux,
        float(base.detach())+ew*aux,ew*aux,dw*aux],device=ctx.device,dtype=torch.float64)*fraction
    if ctx.world>1:torch.distributed.all_reduce(values)
    net.clear()
    return dict(zip(('ce','main_kd','embedding','reconstruction','total','encoder_reconstruction','decoder_reconstruction'),values.cpu().tolist()))

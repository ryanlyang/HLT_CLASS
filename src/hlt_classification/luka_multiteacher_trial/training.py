"""Matched fixed30 training plus optional frozen-reference relation prediction."""
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from hlt_classification.jetclass2_delphes.runner import _forward, predict, selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from .contracts import artifact, validate, recipe, TEACHERS


class AnchorSampler:
    def __init__(self, labels):
        if labels.dtype != np.int64 or labels.ndim != 1 or set(np.unique(labels)) != set(range(11)):
            raise ValueError('Reference sampler requires canonical TRAIN classes')
        self.classes = [np.flatnonzero(labels == c) for c in range(11)]
        if min(map(len, self.classes)) < 187:
            raise ValueError('Need187 distinct TRAIN rows per class; no fallback')

    def sample(self, epoch, batch, global_step):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([314159, epoch, batch])))
        extra = {(2*global_step)%11, (2*global_step+1)%11}
        return np.concatenate([rng.choice(ix,186+int(c in extra),replace=False) for c,ix in enumerate(self.classes)])


def reference_kl(prediction, target, anchors, valid):
    if (prediction.ndim != 2 or prediction.shape[1] != 128 or target.shape != prediction.shape
            or anchors.ndim != 2 or anchors.shape[1] != 128 or valid.dtype != torch.bool
            or valid.shape != (len(prediction),len(anchors)) or not valid.any(1).all()
            or any(not torch.isfinite(x).all() for x in (prediction,target,anchors))):
        raise ValueError('Invalid frozen-reference relation inputs')
    with torch.autocast(prediction.device.type, enabled=False):
        with torch.no_grad():
            a=F.normalize(anchors.detach().float(),dim=-1,eps=1e-8)
            t=F.normalize(target.detach().float(),dim=-1,eps=1e-8)
            lt=F.log_softmax((t@a.T/.2).masked_fill(~valid,-torch.inf),dim=-1)
            pt=lt.exp()
            safe_t=torch.where(valid,lt,torch.zeros_like(lt))
        s=F.normalize(prediction.float(),dim=-1,eps=1e-8)
        ls=F.log_softmax((s@a.T/.2).masked_fill(~valid,-torch.inf),dim=-1)
        safe_s=torch.where(valid,ls,torch.zeros_like(ls))
        return (pt*(safe_t-safe_s)).sum(-1).mean()


def objective(logits, labels, predictions, targets, arm, *, beta=0., anchors=None, valid=None):
    r=recipe(arm)
    if (set(predictions)!=set(TEACHERS) or set(targets)!=set(TEACHERS)
            or logits.shape!=(len(labels),11) or not torch.isfinite(logits).all()
            or not math.isfinite(beta) or not 0<=beta<=1
            or (not r['relation']['enabled'] and (beta != 0 or anchors is not None or valid is not None))):
        raise ValueError('Wrong registered objective inputs')
    ce=F.cross_entropy(logits.float(),labels.long()); aux={}
    for slot, teacher in r['targets'].items():
        p,t=predictions[slot],targets[teacher]
        if p.shape!=(len(labels),128) or t.shape!=p.shape or not torch.isfinite(p).all() or not torch.isfinite(t).all():
            raise ValueError('Invalid prediction/target')
        aux[slot]=F.smooth_l1_loss(p.float(),t.detach().float(),beta=1.,reduction='mean')
    rel=ce.new_zeros(())
    if beta:
        if anchors is None or set(anchors)!=set(TEACHERS): raise ValueError('Missing relation anchors')
        rel=sum(reference_kl(predictions[t],targets[t],anchors[t],valid) for t in TEACHERS)/4
    return ce+sum(r['weights'][t]*aux[t] for t in TEACHERS)+beta*rel,ce,aux,rel


def train(model, train_cache, val, *, bank, node, device):
    arm=node['branch']; r=recipe(arm); config=r['base']
    if (not len(train_cache) or not len(val) or train_cache.role!='train' or val.role!='validation'
            or train_cache.coordinate_name!='D000' or val.coordinate_name!='D000'
            or train_cache.foundation_sha256!=val.foundation_sha256
            or not np.array_equal(bank.identities,train_cache.identities) or not np.array_equal(bank.labels,train_cache.labels)
            or node['coordinate']!='D000' or node['teacher'] is not None or node['context_coordinate'] is not None
            or not node['deployable'] or node['auxiliary_weight']!=r['total_weight']):
        raise ValueError('HLT training population/recipe differs')
    sampler=AnchorSampler(bank.labels) if r['relation']['enabled'] else None
    model.to(device); excluded=model.no_weight_decay()
    groups=[dict(params=[p for n,p in model.named_parameters() if p.requires_grad and (n in excluded)==ex],
                 weight_decay=0. if ex else config['weight_decay']) for ex in (False,True)]
    opt=torch.optim.AdamW(groups,lr=config['peak_lr'],betas=tuple(config['betas']),eps=config['eps'])
    steps=math.ceil(len(train_cache)/128); update=0; best=None; selected=None; state=None; history=[]
    for epoch in range(1,31):
        model.train(); model.capture=True
        order=np.random.default_rng(np.random.SeedSequence([node['sampler_seed'],epoch])).permutation(len(train_cache))
        ce_sum=rel_sum=weighted_rel_sum=0.; sums={t:0. for t in TEACHERS}; started=time.monotonic()
        for batch,start in enumerate(range(0,len(order),128)):
            ix=order[start:start+128]; update+=1; lr=learning_rate(update/steps)
            for group in opt.param_groups: group['lr']=lr
            raw=train_cache.batch(ix); labels=torch.from_numpy(raw['labels']).to(device)
            targets={t:torch.from_numpy(bank.targets[t][ix]).to(device) for t in TEACHERS}
            anchors=valid=None; beta=0.
            if sampler is not None:
                rows=sampler.sample(epoch,batch,update-1)
                # Exact row masking is equivalent to identity masking after the authenticated unique join.
                valid=torch.from_numpy(ix[:,None]!=rows[None,:]).to(device)
                anchors={t:torch.from_numpy(bank.targets[t][rows]).to(device) for t in TEACHERS}
                beta=min(1.,update/(2*steps))
            opt.zero_grad(set_to_none=True)
            logits=_forward(model,raw,device,bf16=str(device).startswith('cuda'))
            loss,ce,aux,rel=objective(logits,labels,model.predictions,targets,arm,beta=beta,anchors=anchors,valid=valid)
            if not torch.isfinite(loss): raise ValueError('Nonfinite training objective')
            loss.backward()
            finite=[torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
            if not finite or not torch.stack(finite).all(): raise ValueError('Nonfinite gradient')
            opt.step()
            ce_sum+=float(ce.detach())*len(ix); rel_sum+=float(rel.detach())*len(ix)
            weighted_rel_sum+=beta*float(rel.detach())*len(ix)
            for t in TEACHERS: sums[t]+=float(aux[t].detach())*len(ix)
            model.predictions=None
            if (batch+1)%128==0: print(f'MULTI5 {arm} epoch={epoch}/30 batch={batch+1}/{steps}',flush=True)
        seconds=time.monotonic()-started; started=time.monotonic()
        model.capture=False; model.predictions=None
        probabilities=predict(model,val,device=device,batch_size=128)
        metrics=evaluate_probabilities(val.labels,probabilities)
        row=dict(pass_number=epoch,update=update,learning_rate=lr,validation=metrics,
            ce_loss=ce_sum/len(train_cache),auxiliary_losses={t:sums[t]/len(train_cache) for t in TEACHERS},
            relation_loss=rel_sum/len(train_cache),weighted_relation_loss=weighted_rel_sum/len(train_cache),
            relation_beta=beta,train_seconds=seconds,validation_seconds=time.monotonic()-started)
        row['weighted_auxiliary_loss']=sum(r['weights'][t]*row['auxiliary_losses'][t] for t in TEACHERS)
        row['total_loss']=row['ce_loss']+row['weighted_auxiliary_loss']+row['weighted_relation_loss']
        history.append(row); key=selection_key(metrics,update)
        if best is None or key>best: best,selected,state=key,row,model.native_state()
        print(f'MULTI5 {arm} epoch={epoch}/30 auc={metrics["macro_ovr_auc"]:.8f} '
              f'best_auc={selected["validation"]["macro_ovr_auc"]:.8f} best_epoch={selected["pass_number"]} '
              f'CE={row["ce_loss"]:.6f} aux={row["auxiliary_losses"]} relation={row["relation_loss"]:.6f} '
              f'beta={beta:.4f} train_s={seconds:.2f}',flush=True)
    full=model.state_dict(); full.update(state); model.load_state_dict(full,strict=True)
    model.eval(); model.capture=False; model.predictions=None
    report=artifact('TRAINING_REPORT',parents=dict(recipe=r['content_hash'],prepared=train_cache.foundation_sha256,
        bank=bank.manifest['content_hash']),node=node,passes=30,scientific_fit=True,
        selected_pass=selected['pass_number'],validation=selected['validation'],validation_history=history,
        selected_weights_restored=True,initialization='from_scratch_matched_M0HLT',selected_state='native_backbone_only')
    return report,state


def validate_report(report, spec, arm, bank_hash):
    r=recipe(arm); validate(report,'TRAINING_REPORT',dict(recipe=r['content_hash'],
        prepared=spec['anchor']['evidence']['prepared'],bank=bank_hash))
    steps=math.ceil(spec['counts']['train']/128); history=report['validation_history']
    if (report['node']!=spec['nodes'][arm] or report['passes']!=30 or len(history)!=30
            or not report['scientific_fit'] or not report['selected_weights_restored']
            or report['initialization']!='from_scratch_matched_M0HLT' or report['selected_state']!='native_backbone_only'):
        raise ValueError('Report registration differs')
    for e,row in enumerate(history,1):
        if (row['pass_number']!=e or row['update']!=e*steps or row['learning_rate']!=learning_rate(e)
                or row['validation']['rows']!=spec['counts']['validation'] or set(row['auxiliary_losses'])!=set(TEACHERS)
                or row['relation_beta']!=(min(1.,e/2) if r['relation']['enabled'] else 0.)
                or row['weighted_auxiliary_loss']!=sum(r['weights'][t]*row['auxiliary_losses'][t] for t in TEACHERS)
                or row['total_loss']!=row['ce_loss']+row['weighted_auxiliary_loss']+row['weighted_relation_loss']
                or (not r['relation']['enabled'] and (row['relation_loss']!=0 or row['weighted_relation_loss']!=0))):
            raise ValueError('Epoch schedule/loss/coverage differs')
        numbers=[row[k] for k in ('ce_loss','relation_loss','weighted_relation_loss','train_seconds','validation_seconds')]
        numbers+=list(row['auxiliary_losses'].values())
        if not np.isfinite(numbers).all() or min(numbers)<-1e-6: raise ValueError('Nonfinite/invalid epoch loss')
    selected=max(history,key=lambda r:selection_key(r['validation'],r['update']))
    if report['selected_pass']!=selected['pass_number'] or report['validation']!=selected['validation']:
        raise ValueError('Wrong AUC-selected checkpoint')

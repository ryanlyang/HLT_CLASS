"""Fixed30 training and explicit deterministic per-component audit records."""
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from hlt_classification.jetclass2_delphes.runner import _forward,predict,selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.luka_auxiliary_grid.training import optimizer,finite_gradients
from .contracts import artifact,validate,recipe,teacher_recipe,TEACHERS,TEACHER_TASKS
from .model import new_model,neighborhood_loss
from hlt_classification.luka_supervision_depth.training import objective as baseline_objective


def registered(task):
    return teacher_recipe(task) if task in TEACHER_TASKS else recipe(task)


def clear(model):
    for attr in ('predictions','auxiliary_logits','token_predictions','projection','neighborhood_predictions','neighborhood_errors'):
        if hasattr(model,attr): setattr(model,attr,None)


def objective(model,logits,labels,bank,ix,raw,device):
    loss,terms,weights=baseline_objective(model,logits,labels,bank,ix,raw,device)
    r=model.registered
    if r['neighborhood_weight']:
        values=bank.neighborhood.batch(ix,raw['mask'].shape[-1],r['neighborhood'])
        total,families,channels=neighborhood_loss(model.neighborhood_predictions,torch.from_numpy(values).to(device),
                                                torch.from_numpy(raw['mask']).to(device),details=True)
        model.neighborhood_errors=channels.detach()
        # Keep equal family weighting; sorted JSON arithmetic is deterministic.
        for name,value in zip(('neighborhood_count','neighborhood_pt'),families):
            terms[name]=value; weights[name]=r['neighborhood_weight']*.5
        loss=loss+r['neighborhood_weight']*total
    if not torch.isfinite(loss): raise ValueError('Nonfinite neighborhood objective')
    return loss,terms,weights


def batch_loss(model,cache,bank,ix,device):
    raw=cache.batch(ix)
    logits=_forward(model,raw,device,bf16=str(device).startswith('cuda'))
    return objective(model,logits,torch.from_numpy(raw['labels']).to(device),bank,ix,raw,device)


def train(tr,val,*,node,bank,device,comparison=None):
    task=node['arm']; r=registered(task); teacher=task in TEACHER_TASKS
    coordinate='OFFLINE' if teacher else 'D000'
    if (tr.role!='train' or val.role!='validation' or tr.coordinate_name!=coordinate or val.coordinate_name!=coordinate
            or tr.foundation_sha256!=val.foundation_sha256 or not len(tr) or not len(val)):
        raise ValueError('Training view/role mismatch')
    if not teacher and (not np.array_equal(tr.identities,bank.identities) or not np.array_equal(tr.labels,bank.labels)):
        raise ValueError('Target TRAIN join differs')
    model=new_model(node).to(device); opt=optimizer(model,r['base']); steps=math.ceil(len(tr)/128)
    update=0; history=[]; best=None; selected=None; state=None
    for epoch in range(1,31):
        model.train(); model.capture=True; totals={}; diagnostic=np.zeros(36,np.float64); started=time.monotonic()
        order=np.random.default_rng(np.random.SeedSequence([node['sampler_seed'],epoch])).permutation(len(tr))
        for start in range(0,len(order),128):
            ix=order[start:start+128]; update+=1; lr=learning_rate(update/steps)
            for group in opt.param_groups: group['lr']=lr
            opt.zero_grad(set_to_none=True)
            loss,terms,weights=batch_loss(model,tr,bank,ix,device)
            loss.backward(); finite_gradients(model); opt.step()
            for k,v in terms.items(): totals[k]=totals.get(k,0.)+float(v.detach())*len(ix)
            if getattr(model,'neighborhood_errors',None) is not None:
                diagnostic+=model.neighborhood_errors.cpu().numpy().astype(np.float64)*len(ix)
            clear(model); del loss,terms
            if (start//128+1)%128==0:
                print(f'NEIGHBOR progress arm={task} epoch={epoch} batch={start//128+1}/{steps} '
                      f'epoch_percent={100*min(start+128,len(order))/len(order):.1f}',flush=True)
        seconds=time.monotonic()-started; model.capture=False; clear(model); started=time.monotonic()
        metrics=evaluate_probabilities(val.labels,predict(model,val,device=device,batch_size=128))
        raw={k:totals[k]/len(tr) for k in sorted(totals)}
        weighted={k:weights[k]*raw[k] for k in raw}
        row=dict(pass_number=epoch,update=update,learning_rate=lr,validation=metrics,losses=raw,weights=weights,
            weighted_losses=weighted,total_loss=sum(weighted[k] for k in sorted(weighted)),train_seconds=seconds,
            validation_seconds=time.monotonic()-started,neighborhood_channel_errors=(diagnostic/len(tr)).tolist())
        suffix=''
        if comparison is not None:
            auc=metrics['macro_ovr_auc']; row['delta_repeat']=auc-comparison['baseline']
            gap=comparison['offline']-comparison['hlt']
            row['recovery_percent']=100*(auc-comparison['hlt'])/gap if gap>0 else None
            recovered='n/a' if row['recovery_percent'] is None else f'{row["recovery_percent"]:.2f}%'
            suffix=f' delta_repeat={row["delta_repeat"]:+.8f} recovery={recovered}'
        history.append(row); key=selection_key(metrics,update)
        if best is None or key>best: best,selected,state=key,row,model.native_state()
        print(f'NEIGHBOR arm={task} epoch={epoch}/30 auc={metrics["macro_ovr_auc"]:.8f} '
              f'best_auc={selected["validation"]["macro_ovr_auc"]:.8f} best_epoch={selected["pass_number"]} '
              f'loss={row["total_loss"]:.6f} train_s={seconds:.2f}'+suffix,flush=True)
    full=model.state_dict(); full.update(state); model.load_state_dict(full,strict=True)
    model.eval(); model.capture=False; clear(model)
    report=artifact('TRAINING_REPORT',parents=dict(recipe=r['content_hash'],prepared=tr.foundation_sha256,bank=bank.manifest['content_hash']),
        node=node,scientific_fit=True,technical=False,passes=30,validation_history=history,
        selected_pass=selected['pass_number'],validation=selected['validation'],selected_state='native_backbone_only',selected_weights_restored=True,
        comparison=comparison)
    return model,report,state


def expected_weights(task):
    r=registered(task); result={'ce':r['ce_coefficient']}
    if task in TEACHER_TASKS:
        if r['supcon_coefficient']: result['supcon']=r['supcon_coefficient']
    else:
        result.update({'kd_'+t:.25 for t in TEACHERS})
        result.update({'jet_'+t:w for t,w in r['embedding_weights'].items()})
        result.update({name:r['neighborhood_weight']*.5 for name in ('neighborhood_count','neighborhood_pt')})
    return result


def validate_report(report,spec,task,bank_hash):
    validate(report,'TRAINING_REPORT',dict(recipe=registered(task)['content_hash'],prepared=spec['anchor']['evidence']['prepared'],bank=bank_hash))
    history=report['validation_history']; weights=expected_weights(task); steps=math.ceil(spec['counts']['train']/128)
    comparison=report['comparison']
    if (not isinstance(comparison,dict) or set(comparison)!={'baseline','hlt','offline'}
            or not all(np.isfinite(v) and 0<=v<=1 for v in comparison.values())):
        raise ValueError('Invalid fixed reference anchors')
    if (report['node']!=spec['nodes'][task] or report['passes']!=30 or len(history)!=30 or report['scientific_fit'] is not True
            or report['technical'] is not False or report['selected_state']!='native_backbone_only' or report['selected_weights_restored'] is not True):
        raise ValueError('Scientific fit registration differs')
    for epoch,row in enumerate(history,1):
        if (row['pass_number']!=epoch or row['update']!=epoch*steps or row['learning_rate']!=learning_rate(epoch)
                or row['weights']!=weights or set(row['losses'])!=set(weights) or set(row['weighted_losses'])!=set(weights)
                or row['validation']['rows']!=spec['counts']['validation']): raise ValueError('Epoch registration differs')
        auc=row['validation']['macro_ovr_auc']
        gap=comparison['offline']-comparison['hlt']
        recovered=100*(auc-comparison['hlt'])/gap if gap>0 else None
        if (row['delta_repeat']!=auc-comparison['baseline']
                or row['recovery_percent']!=recovered):
            raise ValueError('Epoch comparison differs')
        values=[*row['losses'].values(),*row['weighted_losses'].values(),row['total_loss'],row['train_seconds'],row['validation_seconds']]
        if not np.isfinite(values).all() or min(values)<-1e-6: raise ValueError('Invalid epoch loss/timing')
        diagnostic=np.asarray(row['neighborhood_channel_errors'])
        if (diagnostic.shape!=(36,) or not np.isfinite(diagnostic).all() or np.any(diagnostic<0)
                or not np.allclose([diagnostic[:18].mean(),diagnostic[18:].mean()],
                    [row['losses']['neighborhood_count'],row['losses']['neighborhood_pt']],rtol=1e-5,atol=1e-7)):
            raise ValueError('Per-category/scale diagnostics differ')
        if (any(row['weighted_losses'][k]!=weights[k]*row['losses'][k] for k in weights)
                or row['total_loss']!=sum(row['weighted_losses'][k] for k in sorted(weights))): raise ValueError('Loss arithmetic differs')
    selected=max(history,key=lambda x:selection_key(x['validation'],x['update']))
    if report['selected_pass']!=selected['pass_number'] or report['validation']!=selected['validation']:
        raise ValueError('Selection differs')

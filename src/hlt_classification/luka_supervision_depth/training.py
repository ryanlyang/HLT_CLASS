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
from .model import new_model,token_loss,supcon


def registered(task):
    return teacher_recipe(task) if task in TEACHER_TASKS else recipe(task)


def clear(model):
    for attr in ('predictions','auxiliary_logits','token_predictions','projection'):
        if hasattr(model,attr): setattr(model,attr,None)


def objective(model,logits,labels,bank,ix,raw,device):
    r=model.registered; ce=F.cross_entropy(logits.float(),labels.long())
    terms={'ce':ce}; weights={'ce':r['ce_coefficient']}
    if 'supcon_coefficient' in r:
        if r['supcon_coefficient']:
            terms['supcon']=supcon(model.projection,labels,r['temperature']); weights['supcon']=r['supcon_coefficient']
    else:
        for t in TEACHERS:
            q=torch.from_numpy(bank.probabilities[t][ix]).to(device).detach().float()
            p=model.auxiliary_logits[t].float()
            if q.shape!=logits.shape or (q<0).any() or not torch.allclose(q.sum(-1),torch.ones_like(q[:,0]),atol=1e-6,rtol=1e-5):
                raise ValueError('Invalid KD targets')
            terms['kd_'+t]=4*F.kl_div((p/2).log_softmax(-1),q,reduction='batchmean'); weights['kd_'+t]=.25
        for head,target in r['target_names'].items():
            z=torch.from_numpy(bank.targets[target][ix]).to(device).detach().float()
            pred=model.predictions[head].float()
            if z.shape!=pred.shape or z.shape!=(len(labels),128): raise ValueError('Invalid jet targets')
            terms['jet_'+head]=F.smooth_l1_loss(pred,z,beta=1.,reduction='mean'); weights['jet_'+head]=r['embedding_weights'][head]
        if r['depths']:
            values=bank.tokens.batch(ix,raw['mask'].shape[-1])
            targets={d:torch.from_numpy(v).to(device) for d,v in values.items()}
            _,losses=token_loss(model.token_predictions,targets,torch.from_numpy(raw['mask']).to(device),r['matching'])
            for d in r['depths']:
                terms['token_'+str(d)]=losses[d]; weights['token_'+str(d)]=r['token_weight']/len(r['depths'])
    if any(not torch.isfinite(v).all() for v in (logits,*terms.values())): raise ValueError('Nonfinite required loss')
    # Same summation grouping as old baseline, so parity is not changed by rounding.
    loss=weights['ce']*ce
    if 'supcon_coefficient' in r:
        if 'supcon' in terms: loss=loss+weights['supcon']*terms['supcon']
    else:
        loss=loss+sum(weights['jet_'+t]*terms['jet_'+t] for t in r['embedding_weights'])
        loss=loss+sum(weights['kd_'+t]*terms['kd_'+t] for t in TEACHERS)
        if r['depths']: loss=loss+sum(weights['token_'+str(d)]*terms['token_'+str(d)] for d in r['depths'])
    return loss,terms,weights


def batch_loss(model,cache,bank,ix,device):
    raw=cache.batch(ix)
    logits=_forward(model,raw,device,bf16=str(device).startswith('cuda'))
    return objective(model,logits,torch.from_numpy(raw['labels']).to(device),bank,ix,raw,device)


def train(tr,val,*,node,bank,device):
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
        model.train(); model.capture=True; totals={}; started=time.monotonic()
        order=np.random.default_rng(np.random.SeedSequence([node['sampler_seed'],epoch])).permutation(len(tr))
        for start in range(0,len(order),128):
            ix=order[start:start+128]; update+=1; lr=learning_rate(update/steps)
            for group in opt.param_groups: group['lr']=lr
            opt.zero_grad(set_to_none=True)
            loss,terms,weights=batch_loss(model,tr,bank,ix,device)
            loss.backward(); finite_gradients(model); opt.step()
            for k,v in terms.items(): totals[k]=totals.get(k,0.)+float(v.detach())*len(ix)
            clear(model); del loss,terms
            if (start//128+1)%128==0: print(f'DEPTH progress arm={task} epoch={epoch} batch={start//128+1}/{steps}',flush=True)
        seconds=time.monotonic()-started; model.capture=False; clear(model); started=time.monotonic()
        metrics=evaluate_probabilities(val.labels,predict(model,val,device=device,batch_size=128))
        raw={k:totals[k]/len(tr) for k in sorted(totals)}
        weighted={k:weights[k]*raw[k] for k in raw}
        row=dict(pass_number=epoch,update=update,learning_rate=lr,validation=metrics,losses=raw,weights=weights,
            weighted_losses=weighted,total_loss=sum(weighted[k] for k in sorted(weighted)),train_seconds=seconds,
            validation_seconds=time.monotonic()-started)
        history.append(row); key=selection_key(metrics,update)
        if best is None or key>best: best,selected,state=key,row,model.native_state()
        print(f'DEPTH arm={task} epoch={epoch}/30 auc={metrics["macro_ovr_auc"]:.8f} '
              f'best_auc={selected["validation"]["macro_ovr_auc"]:.8f} best_epoch={selected["pass_number"]} '
              f'loss={row["total_loss"]:.6f} train_s={seconds:.2f}',flush=True)
    full=model.state_dict(); full.update(state); model.load_state_dict(full,strict=True)
    model.eval(); model.capture=False; clear(model)
    report=artifact('TRAINING_REPORT',parents=dict(recipe=r['content_hash'],prepared=tr.foundation_sha256,bank=bank.manifest['content_hash']),
        node=node,scientific_fit=True,technical=False,passes=30,validation_history=history,
        selected_pass=selected['pass_number'],validation=selected['validation'],selected_state='native_backbone_only',selected_weights_restored=True)
    return model,report,state


def expected_weights(task):
    r=registered(task); result={'ce':r['ce_coefficient']}
    if task in TEACHER_TASKS:
        if r['supcon_coefficient']: result['supcon']=r['supcon_coefficient']
    else:
        result.update({'kd_'+t:.25 for t in TEACHERS})
        result.update({'jet_'+t:w for t,w in r['embedding_weights'].items()})
        result.update({'token_'+str(d):r['token_weight']/len(r['depths']) for d in r['depths']})
    return result


def validate_report(report,spec,task,bank_hash):
    validate(report,'TRAINING_REPORT',dict(recipe=registered(task)['content_hash'],prepared=spec['anchor']['evidence']['prepared'],bank=bank_hash))
    history=report['validation_history']; weights=expected_weights(task); steps=math.ceil(spec['counts']['train']/128)
    if (report['node']!=spec['nodes'][task] or report['passes']!=30 or len(history)!=30 or report['scientific_fit'] is not True
            or report['technical'] is not False or report['selected_state']!='native_backbone_only' or report['selected_weights_restored'] is not True):
        raise ValueError('Scientific fit registration differs')
    for epoch,row in enumerate(history,1):
        if (row['pass_number']!=epoch or row['update']!=epoch*steps or row['learning_rate']!=learning_rate(epoch)
                or row['weights']!=weights or set(row['losses'])!=set(weights) or set(row['weighted_losses'])!=set(weights)
                or row['validation']['rows']!=spec['counts']['validation']): raise ValueError('Epoch registration differs')
        values=[*row['losses'].values(),*row['weighted_losses'].values(),row['total_loss'],row['train_seconds'],row['validation_seconds']]
        if not np.isfinite(values).all() or min(values)<-1e-6: raise ValueError('Invalid epoch loss/timing')
        if (any(row['weighted_losses'][k]!=weights[k]*row['losses'][k] for k in weights)
                or row['total_loss']!=sum(row['weighted_losses'][k] for k in sorted(weights))): raise ValueError('Loss arithmetic differs')
    selected=max(history,key=lambda x:selection_key(x['validation'],x['update']))
    if report['selected_pass']!=selected['pass_number'] or report['validation']!=selected['validation']:
        raise ValueError('Selection differs')

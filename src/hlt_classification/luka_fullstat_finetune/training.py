"""Frozen-native head calibration, then ten full-TRAIN low-LR epochs."""
from copy import deepcopy
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from hlt_classification.jetclass2_delphes.runner import predict, selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_embedding_aux.model import head_input
from hlt_classification.luka_fullstat_fine_tail import training as old, banks
from hlt_classification.luka_fullstat_fine_tail.model import new_model
from .contracts import artifact, validate, recipe, learning_rate, EPOCHS, CALIBRATION_EPOCHS


def cpu_state(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_state(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_state(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_state(v) for v in value)
    return value


def restore(node,native,device):
    model=new_model(node)
    expected=model.native_state()
    if native.keys()!=expected.keys(): raise ValueError('Expected native HLT checkpoint only')
    state=model.state_dict(); state.update(native); model.load_state_dict(state,strict=True)
    return model.to(device)


def calibration_loss(model,h,bank,ix,device):
    r=recipe()['objective']; kd={}; embed={}
    for t in r['teacher_order']:
        p=model.logit_heads[t](h); z=model.predictors[t](h)
        q=torch.from_numpy(bank.probabilities[t][ix]).to(device)
        target=torch.from_numpy(bank.targets[t][ix]).to(device)
        kd[t]=4*F.kl_div((p.float()/2).log_softmax(-1),q.detach(),reduction='batchmean')
        embed[t]=F.smooth_l1_loss(z.float(),target.detach(),beta=1.,reduction='mean')
    loss=sum(r['logit_weights'][t]*kd[t]+r['weights'][t]*embed[t] for t in r['teacher_order'])
    if not torch.isfinite(loss): raise ValueError('Nonfinite head calibration')
    return loss


def calibrate(model,tr,bank,node,device):
    banks.check_join(bank,tr,'DE000')
    before=model.native_state(); model.eval(); model.capture=False
    for n,p in model.named_parameters(): p.requires_grad_(n.startswith(('predictors.','logit_heads.')))
    # Only process-local HLT features: no persistent cache or offline input.
    features=np.empty((len(tr),128),np.float32); started=time.monotonic()
    with torch.no_grad():
        for start in range(0,len(tr),128):
            raw=tr.batch(np.arange(start,min(start+128,len(tr))))
            tensors={k:torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')}
            with torch.autocast(torch.device(device).type,dtype=torch.bfloat16,
                                enabled=str(device).startswith('cuda')):
                _,h=head_input(model,**tensors)
            if not torch.isfinite(h).all(): raise ValueError('Nonfinite frozen HLT features')
            features[start:start+len(h)]=h.float().cpu().numpy()
            if start//128%1024==0:
                print(f'FINETUNE frozen_features rows={start+len(h)}/{len(tr)}',flush=True)
    extraction_seconds=time.monotonic()-started
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=0.)
    history=[]
    for epoch in range(1,CALIBRATION_EPOCHS+1):
        started=time.monotonic(); total=0.
        order=np.random.default_rng(np.random.SeedSequence([node['sampler_seed'],73011,epoch])).permutation(len(tr))
        for start in range(0,len(tr),128):
            ix=order[start:start+128]; opt.zero_grad(set_to_none=True)
            loss=calibration_loss(model,torch.from_numpy(features[ix]).to(device),bank,ix,device)
            loss.backward(); old.finite_gradients(model); opt.step(); total+=float(loss.detach())*len(ix)
        history.append(dict(epoch=epoch,loss=total/len(tr),seconds=time.monotonic()-started))
        print(f'FINETUNE head_calibration={epoch}/{CALIBRATION_EPOCHS} loss={total/len(tr):.8f}',flush=True)
    if any(not torch.equal(v,model.native_state()[k]) for k,v in before.items()):
        raise ValueError('Calibration changed native weights or buffers')
    for p in model.parameters(): p.requires_grad_(True)
    model.predictions=model.auxiliary_logits=None
    return cpu_state(model.state_dict()),dict(history=history,rows=len(tr),native_unchanged=True,
        native_feature_bytes=features.nbytes,extraction_seconds=extraction_seconds,validation_used=False)


def validate_calibration(value,rows):
    if (value.get('rows')!=rows or value.get('native_unchanged') is not True
            or value.get('validation_used') is not False or value.get('native_feature_bytes')!=rows*128*4
            or [r['epoch'] for r in value['history']]!=list(range(1,CALIBRATION_EPOCHS+1))
            or not math.isfinite(value['extraction_seconds']) or value['extraction_seconds']<0
            or any(not math.isfinite(r[k]) or r[k]<0 for r in value['history'] for k in ('loss','seconds'))):
        raise ValueError('Frozen-native calibration evidence differs')


def full_state(model,opt,epoch):
    return cpu_state(dict(student=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,
        absolute_epoch=20+epoch,rng_cpu=torch.get_rng_state(),
        rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))


def train(tr,val,bank,node,initial,baseline,parents,device,*,save_epoch=None):
    banks.check_join(bank,tr,'DE000')
    if (val.role!='validation' or val.coordinate_name!='D000' or not len(val)
            or tr.foundation_sha256!=val.foundation_sha256 or baseline['rows']!=len(val)
            or node['node_id']!='DE000' or not node['deployable']):
        raise PermissionError('Same native HLT TRAIN/validation populations required')
    model=new_model(node).to(device); model.load_state_dict(initial,strict=True)
    torch.manual_seed(node['initialization_seed'])
    opt=old.optimizer(model,recipe()['objective']['base'])
    for group in opt.param_groups: group['lr']=learning_rate(0)
    selected=dict(pass_number=0,absolute_epoch=20,update=0,validation=deepcopy(baseline))
    selected_state=full_state(model,opt,0); history=[]; steps=math.ceil(len(tr)/128)
    r=recipe()['objective']; order_teachers=r['teacher_order']
    if save_epoch: save_epoch(0,selected,selected_state)
    for epoch in range(1,EPOCHS+1):
        model.train(); model.capture=True; started=time.monotonic()
        order=np.random.default_rng(np.random.SeedSequence([node['sampler_seed'],20+epoch])).permutation(len(tr))
        ce_sum=0.; kd_sum={t:0. for t in order_teachers}; embed_sum=dict(kd_sum)
        for start in range(0,len(tr),128):
            index=start//128+1; ix=order[start:start+128]; lr=learning_rate(epoch-1+index/steps)
            for group in opt.param_groups: group['lr']=lr
            opt.zero_grad(set_to_none=True)
            loss,ce,kd,embed=old.batch_loss(model,tr.batch(ix),bank,ix,'DE000',device)
            loss.backward(); old.finite_gradients(model); opt.step()
            ce_sum+=float(ce.detach())*len(ix)
            for t in order_teachers:
                kd_sum[t]+=float(kd[t].detach())*len(ix); embed_sum[t]+=float(embed[t].detach())*len(ix)
            model.predictions=model.auxiliary_logits=None
            del loss,ce,kd,embed
            if index%128==0 or index==steps:
                print(f'FINETUNE_PROGRESS epoch={epoch}/{EPOCHS} absolute_epoch={20+epoch} '
                      f'batch={index}/{steps} elapsed_seconds={time.monotonic()-started:.2f}',flush=True)
        seconds=time.monotonic()-started; started=time.monotonic(); model.capture=False
        metrics=evaluate_probabilities(val.labels,predict(model,val,device=device,batch_size=128))
        row=dict(pass_number=epoch,absolute_epoch=20+epoch,update=epoch*steps,learning_rate=lr,
            validation=metrics,ce_loss=ce_sum/len(tr),logit_losses={t:kd_sum[t]/len(tr) for t in order_teachers},
            embedding_losses={t:embed_sum[t]/len(tr) for t in order_teachers},
            train_seconds=seconds,validation_seconds=time.monotonic()-started)
        row['total_loss']=.5*row['ce_loss']+sum(r['logit_weights'][t]*row['logit_losses'][t]
            +r['weights'][t]*row['embedding_losses'][t] for t in order_teachers)
        history.append(row); state=full_state(model,opt,epoch)
        if save_epoch: save_epoch(epoch,row,state)
        if selection_key(metrics,row['update'])>selection_key(selected['validation'],selected['update']):
            selected=deepcopy(row); selected_state=state
        print(f'FINETUNE epoch={epoch}/{EPOCHS} auc={metrics["macro_ovr_auc"]:.8f} '
              f'best_auc={selected["validation"]["macro_ovr_auc"]:.8f} best_added_epoch={selected["pass_number"]} '
              f'lr={lr:.8g} train_seconds={seconds:.2f}',flush=True)
    model.load_state_dict(selected_state['student'],strict=True); model.eval(); model.capture=False
    best_new=max(history,key=lambda x:selection_key(x['validation'],x['update']))
    report=artifact('TRAINING_REPORT',parents=dict(parents,recipe=recipe()['content_hash']),node=node,
        passes=EPOCHS,scientific_fit=True,technical=False,validation_history=history,baseline=baseline,
        selected_pass=selected['pass_number'],selected_absolute_epoch=selected['absolute_epoch'],
        validation=selected['validation'],best_new_pass=best_new['pass_number'],best_new_validation=best_new['validation'],
        baseline_retained=selected['pass_number']==0,optimizer_reset=True,exact_resume=False,
        all_native_trainable=True,selected_weights_restored=True)
    return report,model.native_state(),selected_state


def validate_report(report,spec,parents):
    validate(report,'TRAINING_REPORT',dict(parents,recipe=recipe()['content_hash']))
    history=report['validation_history']; steps=math.ceil(spec['counts']['train']/128)
    r=recipe()['objective']; ts=r['teacher_order']
    if (report['node']!=spec['node'] or report['passes']!=EPOCHS
            or [v['pass_number'] for v in history]!=list(range(1,EPOCHS+1))
            or report['baseline']!=spec['reuse']['parent_validation']
            or any(report[k] is not True for k in ('scientific_fit','optimizer_reset','all_native_trainable','selected_weights_restored'))
            or report['technical'] is not False or report['exact_resume'] is not False):
        raise ValueError('Continuation registration differs')
    for row in history:
        if (row['absolute_epoch']!=20+row['pass_number'] or row['update']!=row['pass_number']*steps
                or row['learning_rate']!=learning_rate(row['pass_number'])
                or row['validation']['rows']!=spec['counts']['validation']
                or set(row['logit_losses'])!=set(ts) or set(row['embedding_losses'])!=set(ts)
                or row['total_loss']!=.5*row['ce_loss']+sum(r['logit_weights'][t]*row['logit_losses'][t]
                    +r['weights'][t]*row['embedding_losses'][t] for t in ts)
                or any(not math.isfinite(v) or v < -1e-6 for v in
                    [row[k] for k in ('ce_loss','total_loss','train_seconds','validation_seconds')]
                    +list(row['logit_losses'].values())+list(row['embedding_losses'].values()))):
            raise ValueError('Epoch supervision, schedule or population differs')
    zero=dict(pass_number=0,absolute_epoch=20,update=0,validation=report['baseline'])
    best=max([zero]+history,key=lambda x:selection_key(x['validation'],x['update']))
    new=max(history,key=lambda x:selection_key(x['validation'],x['update']))
    if (report['selected_pass']!=best['pass_number'] or report['selected_absolute_epoch']!=best['absolute_epoch']
            or report['validation']!=best['validation'] or report['baseline_retained'] is not (best['pass_number']==0)
            or report['best_new_pass']!=new['pass_number'] or report['best_new_validation']!=new['validation']):
        raise ValueError('Baseline-inclusive validation selection differs')

"""FullStat20 fits with source-bound targets and durable full-state epochs."""
import math
import time
import numpy as np
import torch
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_fullstat.recipe import learning_rate
from hlt_classification.luka_fullstat_readout.training import cpu_state, optimizer, finite_gradients
from hlt_classification.luka_u100_multiview.training import predict, gradient_diagnostic
from . import model
from .contracts import artifact, validate, recipe, TEACHERS


def train(tr,val,*,banks,node,task,parents,device,save_epoch):
    name=model.teacher_name(task)
    expected=name or 'D000'
    if (tr.role!='train' or val.role!='validation' or tr.coordinate_name!=expected or val.coordinate_name!=expected
            or tr.foundation_sha256!=val.foundation_sha256 or len(tr)==0 or len(val)==0
            or set(banks)!=(set() if name else set(TEACHERS))):
        raise PermissionError('Wrong scientific roles/views/targets')
    for bank in banks.values():
        if not np.array_equal(bank.identities,tr.identities) or not np.array_equal(bank.labels,tr.labels):
            raise ValueError('Supervision row join differs')
    net=model.new_model(node,task).to(device)
    opt=optimizer(net,recipe()['base']);steps=math.ceil(len(tr)/128)
    history=[]; best_key=None; full_best=None; selected=None;update=0
    for epoch in range(1,21):
        net.train()
        if not name: net.capture=True
        order=np.random.default_rng(np.random.SeedSequence([node['sampler_seed'],epoch])).permutation(len(tr))
        begun=time.monotonic(); sums={}; diagnostic=None
        for start in range(0,len(order),128):
            ids=order[start:start+128];update+=1;lr=learning_rate(update/steps)
            for group in opt.param_groups: group['lr']=lr
            opt.zero_grad(set_to_none=True)
            loss,parts=model.loss_for(net,task,tr.batch(ids),ids,banks,device)
            if not name and start==0 and epoch in (1,3,10,20): diagnostic=gradient_diagnostic(parts,net)
            loss.backward();finite_gradients(net);opt.step()
            for key,value in parts.items(): sums[key]=sums.get(key,0.)+float(value.detach())*len(ids)
            if not name: net.predictions={}
            done=start//128+1
            if done%128==0:
                elapsed=time.monotonic()-begun
                print(f'FULLSTAT MULTIVIEW progress task={task} epoch={epoch}/20 batch={done}/{steps} '
                      f'elapsed_seconds={elapsed:.1f} epoch_remaining_seconds={elapsed*(steps-done)/done:.1f}',flush=True)
        train_seconds=time.monotonic()-begun;begun=time.monotonic()
        probabilities=predict(net,val,device,name=name)
        metrics=evaluate_probabilities(val.labels,probabilities)
        losses={k:v/len(tr) for k,v in sums.items()}
        losses['total']=losses['ce']+sum((10./3)*losses[t] for t in TEACHERS if t in losses)
        row=dict(pass_number=epoch,update=update,learning_rate=lr,validation=metrics,losses=losses,
            train_seconds=train_seconds,validation_seconds=time.monotonic()-begun,gradient_diagnostic=diagnostic)
        full=dict(student=cpu_state(net.state_dict()),optimizer=cpu_state(opt.state_dict()),epoch=epoch,update=update,
            cpu_rng=torch.get_rng_state().clone(),cuda_rng=[v.cpu() for v in torch.cuda.get_rng_state_all()] if torch.device(device).type=='cuda' else [])
        save_epoch(epoch,row,full);history.append(row)
        key=selection_key(metrics,update)
        if best_key is None or key>best_key: best_key,selected,full_best=key,row,full
        print(f'FULLSTAT MULTIVIEW task={task} epoch={epoch}/20 auc={metrics["macro_ovr_auc"]:.8f} '
              f'best_auc={selected["validation"]["macro_ovr_auc"]:.8f} best_epoch={selected["pass_number"]} '
              f'train_seconds={train_seconds:.2f}',flush=True)
    net.load_state_dict(full_best['student'],strict=True);net.eval()
    if not name: net.capture=False;net.predictions={}
    report=artifact('TRAINING_REPORT',parents=dict(parents,recipe=recipe()['content_hash']),task_id=task,node=node,
        passes=20,scientific_fit=True,technical=False,selected_pass=selected['pass_number'],
        validation=selected['validation'],validation_history=history,selected_weights_restored=True,
        deployed_graph='privileged_'+name if name else 'native_HLT',
        initialization='fresh_original_OFFLINE_seeds' if name else 'fresh_original_U100_w5_seeds')
    return net,report,full_best


def validate_report(report,spec,task,parents):
    validate(report,'TRAINING_REPORT',dict(parents,recipe=recipe()['content_hash']))
    name=model.teacher_name(task); history=report['validation_history']
    if (report['task_id']!=task or report['node']!=spec['nodes'][task] or report['passes']!=20
            or report['scientific_fit'] is not True or report['technical'] is not False
            or report['selected_weights_restored'] is not True
            or report['deployed_graph']!=('privileged_'+name if name else 'native_HLT')
            or report['initialization']!=('fresh_original_OFFLINE_seeds' if name else 'fresh_original_U100_w5_seeds')
            or [r['pass_number'] for r in history]!=list(range(1,21))): raise ValueError('Fit identity differs')
    for r in history:
        losses=r['losses']; expected={'ce','total'} if name else {'ce','total',*TEACHERS}
        if (set(losses)!=expected or not np.isfinite(list(losses.values())).all() or min(losses.values())<0
                or r['update']!=r['pass_number']*math.ceil(spec['counts']['train']/128)
                or r['learning_rate']!=learning_rate(r['pass_number'])
                or r['validation']['rows']!=spec['counts']['validation']
                or losses['total']!=losses['ce']+sum((10./3)*losses[t] for t in TEACHERS if t in losses)
                or any(not math.isfinite(r[k]) or r[k]<0 for k in ('train_seconds','validation_seconds'))):
            raise ValueError('Schedule/loss/population differs')
        d=r['gradient_diagnostic']
        if not name and r['pass_number'] in (1,3,10,20):
            if (not d or set(d['norms'])!={'ce',*TEACHERS} or set(d['cosine_with_ce'])!={'ce',*TEACHERS}
                    or any(not math.isfinite(x) or x<0 for x in d['norms'].values())
                    or any(x is not None and not math.isfinite(x) for x in d['cosine_with_ce'].values())):
                raise ValueError('Gradient evidence differs')
        elif d is not None: raise ValueError('Unexpected gradient diagnostic')
    best=max(history,key=lambda r:selection_key(r['validation'],r['update']))
    if report['selected_pass']!=best['pass_number'] or report['validation']!=best['validation']:
        raise ValueError('Selected checkpoint differs')

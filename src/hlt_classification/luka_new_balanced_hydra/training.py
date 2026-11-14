"""Fixed draw-budget training with complete epoch-boundary continuation."""
from contextlib import nullcontext
import math
import time
import numpy as np
import torch
from hlt_classification.luka_fullstat.recipe import learning_rate
from hlt_classification.luka_fullstat_readout.training import optimizer, finite_gradients, cpu_state
from hlt_classification.luka_u100_multiview.training import predict, gradient_diagnostic
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.runner import selection_key
from . import contracts as c
from hlt_classification.luka_fullstat_balanced_ladder import model, contrastive


def join(tr,val,banks,sampler,stage):
    if (tr.role!='train' or val.role!='validation' or tr.coordinate_name!=c.COORDINATES[stage]
            or val.coordinate_name!=c.COORDINATES[stage] or tr.foundation_sha256!=val.foundation_sha256
            or not np.array_equal(tr.labels,sampler.labels) or len(tr)!=len(sampler.indices)
            or set(banks)!=set(c.TEACHERS[stage])):
        raise ValueError('Training population/coordinate/supervision differs')
    for bank in banks.values():
        if not np.array_equal(bank.identities,tr.identities) or not np.array_equal(bank.labels,tr.labels):
            raise ValueError('Ordered teacher/student join differs')


def train(tr,val,*,banks,sampler,stage,parents,device,save_epoch,matching=None,resume=None):
    join(tr,val,banks,sampler,stage)
    if (matching is None)!=(stage!='DE000'):raise ValueError('Matching stage differs')
    if matching is not None and (not np.array_equal(matching.identities,tr.identities)
                                or not np.array_equal(matching.labels,tr.labels)):
        raise ValueError('Matching bank join differs')
    net=model.new_model(stage).to(device);opt=optimizer(net,c.recipe())
    steps=math.ceil(c.DRAWS/c.BATCH);history=[];selected=None;best=None;update=0
    if resume is not None:
        state=resume['state'];history=list(resume['history']);best=resume['best']
        if not history or state['epoch']!=len(history) or state['update']!=len(history)*steps:
            raise ValueError('Resume epoch cursor differs')
        for epoch,row in enumerate(history,1):check_row(row,epoch,stage,len(val),sampler)
        selected=max(history,key=lambda r:selection_key(r['validation'],r['update']))
        if best['epoch']!=selected['pass_number']:raise ValueError('Resume best-state cursor differs')
        if state['draw_order_sha256']!=history[-1]['sampling']['order_sha256']:
            raise ValueError('Resume draw sequence differs')
        net.load_state_dict(state['student'],strict=True);opt.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng'])
        if torch.device(device).type=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
        elif state['cuda_rng']:raise ValueError('CUDA scientific continuation cannot move to CPU')
        update=state['update']
    for epoch in range(len(history)+1,c.EPOCHS+1):
        net.train();order=sampler.draw(epoch,draws=c.DRAWS);exposure=sampler.evidence(order)
        begun=time.monotonic();sums=None;names=None;diagnostic=None
        iterator=(contrastive.Lookahead(matching,order,epoch,batch_size=c.BATCH) if matching is not None else
                  nullcontext(((ix,None) for _,ix in sampler.batches(order,batch_size=c.BATCH))))
        with iterator as batches:
            for done,(ix,candidates) in enumerate(batches,1):
                update+=1;lr=learning_rate(update/steps)
                for group in opt.param_groups:group['lr']=lr
                opt.zero_grad(set_to_none=True)
                loss,parts=model.objective(net,tr.batch(ix),ix,banks,update=update,steps=steps,device=device,
                                           sampler=matching,candidates=candidates)
                if done==1 and epoch in (1,3,10,20):
                    diagnostic=gradient_diagnostic({k:v for k,v in parts.items() if k not in ('total','weighted_matching')},net)
                loss.backward();finite_gradients(net);opt.step()
                values=torch.stack([v.detach() for v in parts.values()]).cpu().numpy().astype(np.float64)
                if sums is None:sums=np.zeros_like(values);names=list(parts)
                sums+=values*len(ix);net.clear()
                if done%128==0:
                    seconds=time.monotonic()-begun
                    print(f'BALANCED progress stage={stage} epoch={epoch}/{c.EPOCHS} batch={done}/{steps} '
                          f'elapsed_seconds={seconds:.1f} epoch_remaining_seconds={seconds*(steps-done)/done:.1f}',flush=True)
        seconds=time.monotonic()-begun;begun=time.monotonic()
        metrics=evaluate_probabilities(val.labels,predict(net,val,device))
        row=dict(pass_number=epoch,update=update,learning_rate=lr,validation=metrics,
            losses=dict(zip(names,(sums/c.DRAWS).tolist())),sampling=exposure,
            matching_coefficient=c.coefficient(stage,update,steps),train_seconds=seconds,
            validation_seconds=time.monotonic()-begun,gradient_diagnostic=diagnostic)
        state=dict(student=cpu_state(net.state_dict()),optimizer=cpu_state(opt.state_dict()),epoch=epoch,update=update,
            draw_order_sha256=exposure['order_sha256'],cpu_rng=torch.get_rng_state().clone(),
            cuda_rng=[x.cpu() for x in torch.cuda.get_rng_state_all()] if torch.device(device).type=='cuda' else [])
        save_epoch(epoch,row,state);history.append(row)
        if selected is None or selection_key(metrics,update)>selection_key(selected['validation'],selected['update']):
            selected=row;best=state
        print(f'BALANCED epoch={epoch}/{c.EPOCHS} stage={stage} auc={metrics["macro_ovr_auc"]:.8f} '
              f'best_auc={selected["validation"]["macro_ovr_auc"]:.8f} best_epoch={selected["pass_number"]}',flush=True)
    net.load_state_dict(best['student'],strict=True);net.eval();net.capture=False;net.clear()
    report=c.artifact('TRAINING_REPORT',parents=dict(parents,recipe=c.recipe()['content_hash']),
        stage=stage,node=c.node(stage),passes=c.EPOCHS,selected_pass=selected['pass_number'],
        validation=selected['validation'],validation_history=history,selected_weights_restored=True,
        scientific_fit=True,deployed_graph='native_HLT' if stage in ('M0HLT','DE000') else 'privileged_native')
    return net,report,best


def check_row(row,epoch,stage,validation_rows,sampler=None):
    steps=math.ceil(c.DRAWS/c.BATCH);losses=row['losses'];names=c.TEACHERS[stage]
    expected={'ce','total',*('embedding_'+t for t in names),*('kd_'+t for t in names)}
    if stage=='DE000':expected|={'matching','weighted_matching'}
    total=losses['ce']+sum(10/len(names)*losses['embedding_'+t]+1/len(names)*losses['kd_'+t] for t in names)
    total+=losses.get('weighted_matching',0.)
    if (row['pass_number']!=epoch or row['update']!=epoch*steps or row['learning_rate']!=learning_rate(epoch)
            or row['validation']['rows']!=validation_rows or set(losses)!=expected
            or not np.isfinite(list(losses.values())).all() or min(losses.values()) < -1e-6
            or not math.isclose(losses['total'],total,rel_tol=2e-6,abs_tol=1e-7)
            or row['matching_coefficient']!=c.coefficient(stage,epoch*steps,steps)
            or any(not math.isfinite(row[k]) or row[k]<0 for k in ('train_seconds','validation_seconds'))):
        raise ValueError('Epoch science/schedule differs')
    e=row['sampling'];counts=np.asarray(e['class_bin_counts'])
    if (e['draws']!=c.DRAWS or counts.shape!=(11,18) or counts.dtype.kind not in 'iu'
            or np.any(counts<0) or counts.sum()!=c.DRAWS or not 1<=e['unique_jets']<=c.DRAWS
            or e['repeated_draws']!=c.DRAWS-e['unique_jets']):
        raise ValueError('Draw exposure receipt differs')
    if sampler is not None and e!=sampler.evidence(sampler.draw(epoch,draws=c.DRAWS)):
        raise ValueError('Draw sequence/exposure replay differs')


def validate_report(report,stage,parents,validation_rows,sampler=None):
    c.validate(report,'TRAINING_REPORT',dict(parents,recipe=c.recipe()['content_hash']))
    h=report['validation_history']
    if (report['stage']!=stage or report['node']!=c.node(stage) or report['passes']!=c.EPOCHS or len(h)!=c.EPOCHS
            or report['scientific_fit'] is not True or report['selected_weights_restored'] is not True
            or report['deployed_graph']!=('native_HLT' if stage in ('M0HLT','DE000') else 'privileged_native')):
        raise ValueError('Completed training report differs')
    for epoch,row in enumerate(h,1):check_row(row,epoch,stage,validation_rows,sampler)
    best=max(h,key=lambda r:selection_key(r['validation'],r['update']))
    if report['selected_pass']!=best['pass_number'] or report['validation']!=best['validation']:
        raise ValueError('Selected checkpoint differs')

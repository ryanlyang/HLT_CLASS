"""FullStat20 optimization with deterministic epoch-boundary continuation."""
import math
import time
import numpy as np
import torch
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_fullstat.recipe import learning_rate
from hlt_classification.luka_fullstat_readout.training import optimizer, finite_gradients, cpu_state
from hlt_classification.luka_u100_multiview.training import predict, gradient_diagnostic
from . import model
from .sampling import Lookahead
from .contracts import artifact, validate, recipe, coefficient, TASK, TEACHERS


def train(tr,val,*,banks,sampler,node,parents,device,save_epoch,resume=None):
    if (tr.role != 'train' or val.role != 'validation' or tr.coordinate_name != 'D000'
            or val.coordinate_name != 'D000' or tr.foundation_sha256 != val.foundation_sha256
            or not len(tr) or not len(val) or set(banks) != set(TEACHERS)
            or not np.array_equal(sampler.identities,tr.identities)
            or not np.array_equal(sampler.labels,tr.labels)):
        raise PermissionError('Wrong HLT roles, identities or targets')
    for bank in banks.values():
        if not np.array_equal(bank.identities,tr.identities) or not np.array_equal(bank.labels,tr.labels):
            raise ValueError('Embedding row join differs')
    net = model.new_model(node).to(device); opt = optimizer(net,recipe()['multi3']['base'])
    steps = math.ceil(len(tr)/128); history=[]; update=0; selected=None; best=None
    if resume is not None:
        state, history, best = resume['state'],list(resume['history']),resume['best']
        if state['epoch'] != len(history) or state['update'] != len(history)*steps:
            raise ValueError('Resume cursor differs')
        net.load_state_dict(state['student'],strict=True);opt.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng'])
        if torch.device(device).type == 'cuda': torch.cuda.set_rng_state_all(state['cuda_rng'])
        elif state['cuda_rng']: raise ValueError('Cannot move a CUDA fit to CPU')
        update=state['update']; selected=max(history,key=lambda r:selection_key(r['validation'],r['update']))
        if best['epoch'] != selected['pass_number']: raise ValueError('Resume selected state differs')
    for epoch in range(len(history)+1,21):
        net.train();net.capture=True
        order = np.random.default_rng(np.random.SeedSequence([node['sampler_seed'],epoch])).permutation(len(tr))
        begun=time.monotonic();sums=None;diagnostic=None
        with Lookahead(sampler,order,epoch) as batches:
            for done,(ids,candidates) in enumerate(batches,1):
                update+=1;lr=learning_rate(update/steps)
                for group in opt.param_groups: group['lr']=lr
                opt.zero_grad(set_to_none=True)
                loss,parts=model.objective(net,tr.batch(ids),ids,banks,sampler,candidates,
                    coefficient(update,steps),device)
                if done==1 and epoch in (1,3,10,20):
                    diagnostic=gradient_diagnostic({k:v for k,v in parts.items() if k not in ('total','weighted_matching')},net)
                loss.backward();finite_gradients(net);opt.step()
                # One ordered scalar transfer, instead of one device sync per loss.
                values=torch.stack([v.detach() for v in parts.values()]).cpu().numpy().astype(np.float64)
                if sums is None: sums=np.zeros_like(values);names=list(parts)
                sums+=values*len(ids);net.clear()
                if done%128==0:
                    elapsed=time.monotonic()-begun
                    print(f'MULTI3 MIX progress epoch={epoch}/20 batch={done}/{steps} '
                          f'elapsed_seconds={elapsed:.1f} epoch_remaining_seconds={elapsed*(steps-done)/done:.1f}',flush=True)
        train_seconds=time.monotonic()-begun;begun=time.monotonic()
        metrics=evaluate_probabilities(val.labels,predict(net,val,device))
        losses=dict(zip(names,(sums/len(tr)).tolist()))
        row=dict(pass_number=epoch,update=update,learning_rate=lr,validation=metrics,losses=losses,
            matching_coefficient=coefficient(update,steps),train_seconds=train_seconds,
            validation_seconds=time.monotonic()-begun,gradient_diagnostic=diagnostic)
        full=dict(student=cpu_state(net.state_dict()),optimizer=cpu_state(opt.state_dict()),epoch=epoch,update=update,
            cpu_rng=torch.get_rng_state().clone(),cuda_rng=[x.cpu() for x in torch.cuda.get_rng_state_all()]
                if torch.device(device).type=='cuda' else [])
        save_epoch(epoch,row,full);history.append(row)
        if selected is None or selection_key(metrics,update)>selection_key(selected['validation'],selected['update']):
            selected=row;best=full
        print(f'MULTI3 MIX epoch={epoch}/20 auc={metrics["macro_ovr_auc"]:.8f} '
              f'best_auc={selected["validation"]["macro_ovr_auc"]:.8f} best_epoch={selected["pass_number"]} '
              f'train_seconds={train_seconds:.2f}',flush=True)
    net.load_state_dict(best['student'],strict=True);net.eval();net.capture=False;net.clear()
    report=artifact('TRAINING_REPORT',parents=dict(parents,recipe=recipe()['content_hash']),task_id=TASK,node=node,
        passes=20,scientific_fit=True,selected_pass=selected['pass_number'],validation=selected['validation'],
        validation_history=history,selected_weights_restored=True,deployed_graph='native_HLT')
    return net,report,best


def check_row(row, epoch, counts):
    steps=math.ceil(counts['train']/128);losses=row['losses']
    if (row['pass_number']!=epoch or row['update']!=epoch*steps or row['learning_rate']!=learning_rate(epoch)
            or row['validation']['rows']!=counts['validation']
            or row['matching_coefficient']!=coefficient(epoch*steps,steps)
            or set(losses)!={'ce',*TEACHERS,'matching','weighted_matching','total'}
            or not np.isfinite(list(losses.values())).all() or min(losses.values())<0
            or not math.isclose(losses['total'],losses['ce']+sum(10/3*losses[t] for t in TEACHERS)
                                +losses['weighted_matching'],rel_tol=2e-6,abs_tol=1e-7)
            or any(not math.isfinite(row[k]) or row[k]<0 for k in ('train_seconds','validation_seconds'))):
        raise ValueError('Epoch science/schedule differs')
    d=row['gradient_diagnostic']
    if epoch in (1,3,10,20):
        expected={'ce',*TEACHERS,'matching'}
        if (not d or set(d['norms'])!=expected or set(d['cosine_with_ce'])!=expected
                or any(not math.isfinite(v) or v<0 for v in d['norms'].values())
                or any(v is not None and not math.isfinite(v) for v in d['cosine_with_ce'].values())):
            raise ValueError('Gradient diagnostics differ')
    elif d is not None: raise ValueError('Unexpected diagnostic')


def validate_report(report,spec,parents):
    validate(report,'TRAINING_REPORT',dict(parents,recipe=recipe()['content_hash']))
    h=report['validation_history']
    if (report['task_id']!=TASK or report['node']!=spec['nodes'][TASK] or report['passes']!=20
            or report['scientific_fit'] is not True or report['selected_weights_restored'] is not True
            or report['deployed_graph']!='native_HLT' or len(h)!=20): raise ValueError('Wrong completed fit')
    for epoch,row in enumerate(h,1): check_row(row,epoch,spec['counts'])
    best=max(h,key=lambda r:selection_key(r['validation'],r['update']))
    if report['selected_pass']!=best['pass_number'] or report['validation']!=best['validation']:
        raise ValueError('Selected checkpoint differs')

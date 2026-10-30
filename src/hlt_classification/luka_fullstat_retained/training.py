"""Matched FullStat20 optimization with durable full-graph checkpoints."""
import math
import time
import numpy as np
import torch
from hlt_classification.jetclass2_delphes.runner import predict, selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_fullstat.recipe import learning_rate
from hlt_classification.luka_fullstat_readout.training import cpu_state, optimizer, finite_gradients
from hlt_classification.luka_fullstat_u100_embedding import banks
from hlt_classification.luka_u100_multiview.training import gradient_diagnostic
from . import model
from .contracts import recipe, artifact, validate, check_node


def train(tr, val, *, bank, node, task, parents, device, save_epoch=None):
    check_node(node,task)
    if (len(tr)==0 or len(val)==0 or tr.role!='train' or val.role!='validation'
            or tr.coordinate_name!='D000' or val.coordinate_name!='D000'
            or tr.foundation_sha256!=val.foundation_sha256):
        raise PermissionError('Only authenticated native HLT TRAIN/validation')
    if task=='train_retained': banks.check_join(bank,tr)
    elif bank is not None: raise PermissionError('CE control cannot read teacher targets')
    student=model.new_model(node,task).to(device)
    registered=recipe(); opt=optimizer(student,registered['base']); steps=math.ceil(len(tr)/128)
    update=0; history=[]; best_key=None; best_full=None; selected=None
    frozen=cpu_state(student.mod.fc.state_dict())
    for epoch in range(1,21):
        student.train(); student.capture=True
        order=np.random.default_rng(np.random.SeedSequence([node['sampler_seed'],epoch])).permutation(len(tr))
        begun=time.monotonic(); ce_sum=aux_sum=0.; diagnostic=None
        for start in range(0,len(order),128):
            ids=order[start:start+128]; update+=1; lr=learning_rate(update/steps)
            for group in opt.param_groups: group['lr']=lr
            opt.zero_grad(set_to_none=True)
            loss,parts=model.batch_loss(student,tr.batch(ids),bank,ids,task,device)
            if start==0 and epoch in (1,3,10,20): diagnostic=gradient_diagnostic(parts,student)
            loss.backward(); finite_gradients(student); opt.step()
            ce_sum+=float(parts['ce'].detach())*len(ids)
            if bank is not None: aux_sum+=float(parts['U100'].detach())*len(ids)
            student.predictions={}
            done=start//128+1
            if done%128==0:
                elapsed=time.monotonic()-begun
                print(f'FULLSTAT RETAINED progress task={task} epoch={epoch}/20 batch={done}/{steps} '
                      f'elapsed_seconds={elapsed:.1f} epoch_remaining_seconds={elapsed*(steps-done)/done:.1f}',flush=True)
        train_seconds=time.monotonic()-begun; student.capture=False; begun=time.monotonic()
        probabilities=predict(student,val,device=device,batch_size=128)
        metrics=evaluate_probabilities(val.labels,probabilities); student.predictions={}
        row=dict(pass_number=epoch,update=update,learning_rate=lr,validation=metrics,
            ce_loss=ce_sum/len(tr),auxiliary_loss=aux_sum/len(tr),train_seconds=train_seconds,
            validation_seconds=time.monotonic()-begun,gradient_diagnostic=diagnostic)
        row['weighted_auxiliary_loss']=node['auxiliary_weight']*row['auxiliary_loss']
        row['total_loss']=row['ce_loss']+row['weighted_auxiliary_loss']
        if any(not torch.equal(v.cpu(),frozen[k]) for k,v in student.mod.fc.state_dict().items()):
            raise ValueError('Unused native classifier mutated')
        history.append(row)
        full=dict(student=cpu_state(student.state_dict()),optimizer=cpu_state(opt.state_dict()),
            epoch=epoch,update=update,cpu_rng=torch.get_rng_state().clone(),
            cuda_rng=[v.cpu() for v in torch.cuda.get_rng_state_all()] if torch.device(device).type=='cuda' else [])
        if save_epoch is not None: save_epoch(epoch,row,full)
        key=selection_key(metrics,update)
        if best_key is None or key>best_key: best_key,selected,best_full=key,row,full
        print(f'FULLSTAT RETAINED task={task} epoch={epoch}/20 auc={metrics["macro_ovr_auc"]:.8f} '
              f'best_auc={selected["validation"]["macro_ovr_auc"]:.8f} best_epoch={selected["pass_number"]} '
              f'CE={row["ce_loss"]:.5f} embedding={row["auxiliary_loss"]:.5f} '
              f'weight={node["auxiliary_weight"]:g} train_s={train_seconds:.2f}',flush=True)
    student.load_state_dict(best_full['student'],strict=True)
    student.capture=False; student.eval()
    report=artifact('TRAINING_REPORT',parents=dict(parents,recipe=registered['content_hash']),
        node=node,task_id=task,scientific_fit=True,technical=False,passes=20,
        selected_pass=selected['pass_number'],validation=selected['validation'],validation_history=history,
        selected_weights_restored=True,initialization='from_scratch_matched_M0HLT',
        selected_state='encoder_predictor_combined_classifier',deployed_graph='retained_predictor',
        teacher_targets_used=bank is not None)
    # Full selected state is the deployed state; nothing is stripped.
    return report,best_full['student'],best_full


def validate_report(report,spec,task,parents):
    validate(report,'TRAINING_REPORT',dict(parents,recipe=recipe()['content_hash']))
    node=spec['nodes'][task]; check_node(node,task); history=report['validation_history']
    if (report['node']!=node or report['task_id']!=task or report['passes']!=20
            or report['scientific_fit'] is not True or report['technical'] is not False
            or report['selected_weights_restored'] is not True
            or report['selected_state']!='encoder_predictor_combined_classifier'
            or report['deployed_graph']!='retained_predictor'
            or report['initialization']!='from_scratch_matched_M0HLT'
            or report['teacher_targets_used']!=(task=='train_retained')
            or [r['pass_number'] for r in history]!=list(range(1,21))):
        raise ValueError('Retained fit identity/status differs')
    for r in history:
        values=[r[k] for k in ('ce_loss','auxiliary_loss','weighted_auxiliary_loss','total_loss','train_seconds','validation_seconds')]
        if (not np.isfinite(values).all() or min(values)<0
                or r['update']!=r['pass_number']*math.ceil(spec['counts']['train']/128)
                or r['learning_rate']!=learning_rate(r['pass_number'])
                or r['validation']['rows']!=spec['counts']['validation']
                or r['weighted_auxiliary_loss']!=node['auxiliary_weight']*r['auxiliary_loss']
                or r['total_loss']!=r['ce_loss']+r['weighted_auxiliary_loss']
                or (task=='train_control' and r['auxiliary_loss']!=0)):
            raise ValueError('Retained schedule/loss/population differs')
        d=r['gradient_diagnostic']; names={'ce','U100'} if task=='train_retained' else {'ce'}
        if r['pass_number'] in (1,3,10,20):
            if (not d or set(d['norms'])!=names or set(d['cosine_with_ce'])!=names
                    or any(not math.isfinite(v) or v<0 for v in d['norms'].values())
                    or any(v is not None and not math.isfinite(v) for v in d['cosine_with_ce'].values())):
                raise ValueError('Gradient diagnostic differs')
        elif d is not None: raise ValueError('Unexpected gradient diagnostic')
    best=max(history,key=lambda r:selection_key(r['validation'],r['update']))
    if report['selected_pass']!=best['pass_number'] or report['validation']!=best['validation']:
        raise ValueError('Combined-classifier checkpoint selection differs')

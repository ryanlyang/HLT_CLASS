"""Fixed30 matched training; CE/KD changes independently from total embedding weight5."""
import math
import time
import numpy as np
import torch
from hlt_classification.jetclass2_delphes.runner import _forward, predict, selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.luka_fullstat_hybrid_embedding.training import objective as donor_objective
from .contracts import artifact, validate, recipe, coefficients, teachers, STAGES
from .model import new_model


def objective(logits, labels, predictions, targets, probabilities, *, arm, stage, epoch):
    _, ce, kd, aux = donor_objective(logits,labels,predictions,targets,stage=stage,probabilities=probabilities)
    cw, kw = coefficients(arm,epoch)
    return cw*ce + kw*kd + sum(5./len(aux)*aux[t] for t in teachers(stage)), ce, kd, aux


def train(train_cache, validation, *, bank, node, device):
    arm,stage = node['arm'],node['node_id']; registered = recipe(arm,stage)
    if (train_cache.role != 'train' or validation.role != 'validation'
            or train_cache.coordinate_name != STAGES[stage] or validation.coordinate_name != STAGES[stage]
            or train_cache.foundation_sha256 != validation.foundation_sha256
            or not np.array_equal(bank.identities,train_cache.identities)
            or not np.array_equal(bank.labels,train_cache.labels)
            or bank.manifest['arm'] != arm or bank.manifest['stage'] != stage
            or node['teacher'] != registered['kd_teacher'] or node['context_coordinate'] is not None
            or not len(train_cache) or not len(validation)):
        raise ValueError('Rung/branch/population/target mismatch')
    student = new_model(node).to(device); config = registered['base']; order_teachers = teachers(stage)
    excluded = student.no_weight_decay()
    groups = [dict(params=[p for n,p in student.named_parameters() if p.requires_grad and (n in excluded)==exclude],
                   weight_decay=0. if exclude else config['weight_decay']) for exclude in (False,True)]
    optimizer = torch.optim.AdamW(groups,lr=config['peak_lr'],betas=tuple(config['betas']),eps=config['eps'])
    steps = math.ceil(len(train_cache)/128); update=0; history=[]; best_key=None; selected=None; state=None
    for epoch in range(1,31):
        student.train(); student.capture=True
        order=np.random.default_rng(np.random.SeedSequence([node['sampler_seed'],epoch])).permutation(len(train_cache))
        start_time=time.monotonic(); ce_sum=kd_sum=0.; aux_sum={t:0. for t in order_teachers}
        for start in range(0,len(order),128):
            ix=order[start:start+128]; update+=1; lr=learning_rate(update/steps)
            for group in optimizer.param_groups: group['lr']=lr
            raw=train_cache.batch(ix); optimizer.zero_grad(set_to_none=True)
            logits=_forward(student,raw,device,bf16=str(device).startswith('cuda'))
            targets={t:torch.from_numpy(bank.targets[t][ix]).to(device) for t in order_teachers}
            loss,ce,kd,aux=objective(logits,torch.from_numpy(raw['labels']).to(device),student.predictions,
                targets,torch.from_numpy(bank.probabilities[ix]).to(device),arm=arm,stage=stage,epoch=epoch)
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward()
            finite=[torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None]
            if not finite or not torch.stack(finite).all(): raise ValueError('Nonfinite gradients')
            optimizer.step(); ce_sum+=float(ce.detach())*len(ix); kd_sum+=float(kd.detach())*len(ix)
            for t in order_teachers: aux_sum[t]+=float(aux[t].detach())*len(ix)
            student.predictions=None
            if (start//128+1)%128==0:
                print(f'TRIAL_HYBRID progress arm={arm} stage={stage} epoch={epoch}/30 batch={start//128+1}/{steps}',flush=True)
        train_seconds=time.monotonic()-start_time
        student.capture=False; student.predictions=None; start_time=time.monotonic()
        metrics=evaluate_probabilities(validation.labels,predict(student,validation,device=device,batch_size=128))
        cw,kw=coefficients(arm,epoch)
        row=dict(pass_number=epoch,update=update,learning_rate=lr,validation=metrics,ce_coefficient=cw,kd_coefficient=kw,
            ce_loss=ce_sum/len(train_cache),kd_loss=kd_sum/len(train_cache),
            auxiliary_losses={t:aux_sum[t]/len(train_cache) for t in order_teachers},
            train_seconds=train_seconds,validation_seconds=time.monotonic()-start_time)
        row['weighted_auxiliary_losses']={t:5./len(order_teachers)*row['auxiliary_losses'][t] for t in order_teachers}
        row['weighted_auxiliary_loss']=sum(row['weighted_auxiliary_losses'].values())
        row['weighted_ce_loss']=cw*row['ce_loss']; row['weighted_kd_loss']=kw*row['kd_loss']
        row['total_loss']=row['weighted_ce_loss']+row['weighted_kd_loss']+row['weighted_auxiliary_loss']
        history.append(row); key=selection_key(metrics,update)
        if best_key is None or key>best_key: best_key,selected,state=key,row,student.native_state()
        print(f'TRIAL_HYBRID arm={arm} stage={stage} epoch={epoch}/30 auc={metrics["macro_ovr_auc"]:.8f} '
              f'best_auc={selected["validation"]["macro_ovr_auc"]:.8f} best_epoch={selected["pass_number"]} '
              f'CE_weight={cw:.6f} KD_weight={kw:.6f} CE={row["ce_loss"]:.6f} KD_T2={row["kd_loss"]:.6f} '
              f'weighted_aux={row["weighted_auxiliary_loss"]:.6f} train_s={train_seconds:.2f}',flush=True)
    full=student.state_dict(); full.update(state); student.load_state_dict(full,strict=True)
    student.eval(); student.capture=False; student.predictions=None
    report=artifact('TRAINING_REPORT',parents=dict(recipe=registered['content_hash'],
        prepared=train_cache.foundation_sha256,bank=bank.manifest['content_hash']),node=node,
        scientific_fit=True,technical=False,passes=30,validation_history=history,
        selected_pass=selected['pass_number'],validation=selected['validation'],selected_weights_restored=True,
        selected_state='native_backbone_only',initialization='from_scratch_coordinate_paired')
    return student,report,state


def validate_report(report,spec,arm,stage,bank_hash):
    r=recipe(arm,stage); order=teachers(stage); history=report['validation_history']; steps=math.ceil(spec['counts']['train']/128)
    validate(report,'TRAINING_REPORT',dict(recipe=r['content_hash'],prepared=spec['anchor']['evidence']['prepared'],bank=bank_hash))
    if (report['node']!=spec['nodes'][arm][stage] or report['scientific_fit'] is not True or report['technical'] is not False
            or report['passes']!=30 or len(history)!=30 or [x['pass_number'] for x in history]!=list(range(1,31))
            or report['selected_state']!='native_backbone_only' or report['selected_weights_restored'] is not True
            or report['initialization']!='from_scratch_coordinate_paired'):
        raise ValueError('Training semantics differ')
    for x in history:
        cw,kw=coefficients(arm,x['pass_number']); aux=x['auxiliary_losses']; weighted=x['weighted_auxiliary_losses']
        nums=[x[k] for k in ('ce_loss','kd_loss','total_loss','weighted_ce_loss','weighted_kd_loss','weighted_auxiliary_loss','train_seconds','validation_seconds')]
        if (set(aux)!=set(order) or set(weighted)!=set(order)
                or not np.isfinite(nums+list(aux.values())+list(weighted.values())).all()
                or min(nums+list(aux.values())+list(weighted.values())) < -1e-6
                or x['update']!=x['pass_number']*steps or x['learning_rate']!=learning_rate(x['pass_number'])
                or x['validation']['rows']!=spec['counts']['validation']
                or (x['ce_coefficient'],x['kd_coefficient'])!=(cw,kw)
                or x['weighted_ce_loss']!=cw*x['ce_loss'] or x['weighted_kd_loss']!=kw*x['kd_loss']
                or any(weighted[t]!=5./len(order)*aux[t] for t in order)
                or x['weighted_auxiliary_loss']!=sum(weighted[t] for t in order)
                or x['total_loss']!=x['weighted_ce_loss']+x['weighted_kd_loss']+x['weighted_auxiliary_loss']):
            raise ValueError('Epoch loss/schedule/population evidence differs')
    best=max(history,key=lambda x:selection_key(x['validation'],x['update']))
    if report['selected_pass']!=best['pass_number'] or report['validation']!=best['validation']:
        raise ValueError('Selection differs from full-validation AUC rule')

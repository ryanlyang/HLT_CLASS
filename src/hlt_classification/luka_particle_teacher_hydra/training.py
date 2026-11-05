"""Production and miniature kernels, HLT-only selection and executable exports."""
from pathlib import Path
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import (load_json,load_npz_arrays,write_immutable_json,
    atomic_publish_bytes,deterministic_npz_bytes)
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from hlt_classification.luka_cross_message_hydra.training import optimizer, predict, forward as old_forward
from .contracts import ARMS, artifact, validate, recipe, seed, coefficient
from .model import new_model, deployment, particle_loss

LOSS_KEYS=('ce_hlt','embedding_loss','particle_loss','weighted_particle','total_loss')+tuple(
    f'd{d}_{k}' for d in (6,8) for k in ('full','centered','loss','prediction_variance','target_variance'))


def step(model,opt,raw,device,*,setting,position,embedding,targets=None):
    if set(raw)-{'features','vectors','mask','identities','labels'}:
        raise PermissionError('Unexpected student input fields')
    if (targets is None)!= (not setting['depths']): raise ValueError('Wrong particle supervision role')
    for group in opt.param_groups: group['lr']=learning_rate(position)
    model.train(); model.capture=True; opt.zero_grad(set_to_none=True)
    logits=old_forward(model,raw,device)
    mask=torch.from_numpy(raw['mask'][:,0]).to(device).bool()
    labels=torch.from_numpy(raw['labels']).to(device).long()
    ce=F.cross_entropy(logits.float(),labels)
    embedding=torch.as_tensor(embedding,device=device).detach().float()
    if embedding.shape!=model.prediction.shape or not torch.isfinite(embedding).all(): raise ValueError('Invalid embedding target')
    emb=F.smooth_l1_loss(model.prediction,embedding)
    corr=ce.new_zeros(()); details={}
    if targets is not None:
        if set(targets)!=set(setting['depths']): raise ValueError('Wrong particle target depths')
        tensors={d:torch.as_tensor(v,device=device).detach().float() for d,v in targets.items()}
        corr,details=particle_loss(model.token_predictions,tensors,mask,centered=setting['centered'])
    weighted=coefficient(setting,position)*corr
    total=ce+5*emb+weighted
    if not torch.isfinite(total): raise ValueError('Nonfinite objective')
    total.backward()
    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise ValueError('Nonfinite gradient')
    opt.step()
    result=dict(ce_hlt=ce,embedding_loss=emb,particle_loss=corr,weighted_particle=weighted,total_loss=total)
    for d in (6,8):
        for key in ('full','centered','loss','prediction_variance','target_variance'):
            result[f'd{d}_{key}']=details.get(str(d),{}).get(key,ce.new_zeros(()))
    return np.array([float(result[k].detach()) for k in LOSS_KEYS],np.float64)


def fit_parents(spec,task,bank_hash):
    from .inputs import registered_metadata
    parents=dict(spec=spec['content_hash'],recipe=recipe()['content_hash'],
        prepared=spec['anchor']['evidence']['prepared'],embedding=bank_hash)
    if ARMS[task]['depths']:
        parents.update(encoder=spec['encoder_hash'],teacher=spec['teacher_proof']['report_hash'],
            targets=registered_metadata(spec)['content_hash'])
    return parents


def fit(folder,spec,task,train,val,bank,*,device='cuda',targets=None,technical=False):
    setting=ARMS[task]
    if (train.role!='train' or val.role!='validation' or train.coordinate_name!='D000'
            or val.coordinate_name!='D000' or train.foundation_sha256!=val.foundation_sha256
            or train.foundation_sha256!=spec['anchor']['evidence']['prepared']):
        raise PermissionError('Expected authenticated HLT TRAIN and validation')
    counts=dict(train=len(train),validation=len(val))
    if not technical and counts!=spec['counts']: raise ValueError('Fit population differs')
    if any(not np.array_equal(getattr(bank,k),getattr(train,k)) for k in ('identities','labels')):
        raise ValueError('Embedding bank alignment differs')
    if bank.manifest['content_hash']!=spec['embedding_hash']: raise ValueError('Embedding parent differs')
    if (targets is None)!=(not setting['depths']): raise ValueError('Wrong target data role')
    if targets is not None:
        from .inputs import registered_metadata
        if (targets.role!='train' or any(not np.array_equal(getattr(targets,k),getattr(train,k)) for k in ('identities','labels'))
                or targets.metadata!=registered_metadata(spec)):
            raise ValueError('TRAIN particle target alignment or metadata differs')
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=False)
    model=new_model(spec['imported'],task).to(device); opt=optimizer(model)
    epochs=1 if technical else 30; steps=math.ceil(len(train)/128)
    history=[]; best=None; state=None; selected=None; update=0
    for epoch in range(1,epochs+1):
        # Miniature rows start with the largest128 multiplicities. Preserve that
        # full worst-shape batch; science retains the registered shuffled order.
        order=(np.arange(len(train)) if technical else np.random.default_rng(np.random.SeedSequence(
            [seed(spec['imported'],setting,'sampler_seed'),epoch])).permutation(len(train)))
        sums=np.zeros(len(LOSS_KEYS)); times=[]; sizes=[]
        for start in range(0,len(order),128):
            ix=order[start:start+128]; update+=1
            position=recipe()['technical_position'] if technical else update/steps
            if str(device).startswith('cuda'): torch.cuda.synchronize()
            began=time.perf_counter()
            raw=train.batch(ix)
            particle_targets=None if targets is None else targets.batch(ix,raw['mask'][:,0],setting['depths'])
            sums+=len(ix)*step(model,opt,raw,device,setting=setting,position=position,
                embedding=bank.targets[ix],targets=particle_targets)
            if str(device).startswith('cuda'): torch.cuda.synchronize()
            times.append(time.perf_counter()-began); sizes.append(len(ix))
            if start//128%100==0 or start+128>=len(order):
                print(f'PARTICLE_TEACHER {task} epoch={epoch}/{epochs} batch={start//128+1}/{steps}',flush=True)
        began=time.perf_counter(); probabilities=predict(model,val,device)
        metrics=evaluate_probabilities(val.labels,probabilities)
        row=dict(pass_number=epoch,update=update,validation=metrics,train_seconds=sum(times),
                 validation_seconds=time.perf_counter()-began,**dict(zip(LOSS_KEYS,(sums/len(train)).tolist())))
        if technical: row.update(batch_seconds=times,batch_sizes=sizes,schedule_position=recipe()['technical_position'])
        history.append(row); key=selection_key(metrics,update)
        if best is None or key>best: best=key; selected=epoch; state=model.deployment_state()
        write_immutable_json(folder/f'epoch_{epoch:02d}.json',artifact('EPOCH',parents=dict(spec=spec['content_hash']),task=task,**row))
        print(f'PARTICLE_TEACHER {task} epoch={epoch}/{epochs} auc={metrics["macro_ovr_auc"]:.8f} best_epoch={selected} CE={row["ce_hlt"]:.6f} embed={row["embedding_loss"]:.6f} particle={row["particle_loss"]:.6f}',flush=True)
    del model,opt
    save_state(folder/'selected.pt',state)
    restored=deployment(torch.load(folder/'selected.pt',map_location='cpu',weights_only=True),device)
    actual=predict(restored,val,device); metrics=evaluate_probabilities(val.labels,actual)
    if metrics!=history[selected-1]['validation']: raise ValueError('Selected HLT deployment replay differs')
    report=artifact('TRAINING_REPORT',parents=fit_parents(spec,task,bank.manifest['content_hash']),task=task,
        settings=setting,technical=technical,counts=counts,epochs=epochs,selected_pass=selected,
        validation=metrics,validation_history=history,deployable_hlt_only=True,
        initialization_seed=seed(spec['imported'],setting,'initialization_seed'),sampler_seed=seed(spec['imported'],setting,'sampler_seed'))
    atomic_publish_bytes(folder/'validation_predictions.npz',deterministic_npz_bytes(dict(
        identities=val.identities,labels=val.labels,probabilities=actual)))
    write_immutable_json(folder/'training_report.json',report)
    receipt=artifact('FIT',parents=dict(spec=spec['content_hash'],report=report['content_hash']),task=task,
        reload_verified=True,outputs=[file_ref(folder/name,root=folder) for name in ('selected.pt','training_report.json','validation_predictions.npz')])
    write_immutable_json(folder/'receipt.json',receipt)
    validate_fit(folder,spec,task,technical=technical)
    return report


def validate_fit(folder,spec,task,*,technical=False):
    folder=Path(folder); report=load_json(folder/'training_report.json'); receipt=load_json(folder/'receipt.json')
    validate(report,'TRAINING_REPORT',fit_parents(spec,task,spec['embedding_hash']))
    validate(receipt,'FIT',dict(spec=spec['content_hash'],report=report['content_hash']))
    epochs=1 if technical else 30; setting=ARMS[task]
    if (report['task']!=task or receipt['task']!=task or report['settings']!=setting or report['technical'] is not technical
            or report['epochs']!=epochs or not report['deployable_hlt_only'] or not receipt['reload_verified']
            or (not technical and report['counts']!=spec['counts'])
            or report['initialization_seed']!=seed(spec['imported'],setting,'initialization_seed')
            or report['sampler_seed']!=seed(spec['imported'],setting,'sampler_seed')
            or [r['path'] for r in receipt['outputs']]!=['selected.pt','training_report.json','validation_predictions.npz']):
        raise ValueError('Fit registration differs')
    for ref in receipt['outputs']: validate_file_ref(ref,root=folder)
    history=report['validation_history']; steps=math.ceil(report['counts']['train']/128)
    if [r['pass_number'] for r in history]!=list(range(1,epochs+1)): raise ValueError('Incomplete fit history')
    for row in history:
        values=[row[k] for k in (*LOSS_KEYS,'train_seconds','validation_seconds')]
        if (not np.isfinite(values).all() or min(values)<0 or row['update']!=row['pass_number']*steps
                or row['validation']['rows']!=report['counts']['validation']
                or not math.isclose(row['total_loss'],row['ce_hlt']+5*row['embedding_loss']+row['weighted_particle'],rel_tol=1e-6,abs_tol=1e-6)):
            raise ValueError('Invalid finite loss/schedule evidence')
        if not setting['depths'] and any(row[k]!=0 for k in LOSS_KEYS if k not in ('ce_hlt','embedding_loss','total_loss')):
            raise ValueError('Control acquired privileged supervision')
        if technical and row.get('schedule_position')!=recipe()['technical_position']: raise ValueError('Inactive miniature supervision')
        epoch=load_json(folder/f'epoch_{row["pass_number"]:02d}.json')
        if epoch!=artifact('EPOCH',parents=dict(spec=spec['content_hash']),task=task,**row): raise ValueError('Epoch/report mismatch')
    best=max(history,key=lambda r:selection_key(r['validation'],r['update']))
    if best['pass_number']!=report['selected_pass'] or best['validation']!=report['validation']: raise ValueError('Wrong HLT selection')
    arrays=load_npz_arrays(folder/'validation_predictions.npz'); validate_predictions(arrays,report['counts']['validation'])
    if evaluate_probabilities(arrays['labels'],arrays['probabilities'])!=report['validation']: raise ValueError('Saved metrics mismatch')
    return report,arrays

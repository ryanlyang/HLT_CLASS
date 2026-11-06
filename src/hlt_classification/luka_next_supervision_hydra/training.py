"""Shared native CE+U100-w5 kernel and independent compatibility objectives."""
from pathlib import Path
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.data.cache_contracts import (load_json,load_npz_arrays,write_immutable_json,
    atomic_publish_bytes,deterministic_npz_bytes)
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from hlt_classification.luka_cross_message_hydra.training import optimizer,predict,forward
from .contracts import artifact,validate,setting,recipe,seed,coefficient
from .model import new_model,deployment,auxiliary_loss
from .sampling import tensors

LOSS_KEYS=('ce_hlt','embedding_loss','auxiliary_loss','weighted_auxiliary','total_loss')
PHASE_KEYS=('native_forward','candidate_sampling','auxiliary_forward','diagnostics','backward_optimizer','input_and_dispatch')


def step(model,opt,raw,embedding,device,*,candidates,indices,epoch,position,probe=True,measure=False):
    phases={}; began=time.perf_counter()
    def timing(name):
        nonlocal began
        if not measure: return
        if str(device).startswith('cuda'): torch.cuda.synchronize()
        now=time.perf_counter(); phases[name]=now-began; began=now
    if set(raw)-{'features','vectors','mask','identities','labels'}: raise PermissionError('Unexpected student input')
    cfg=model.cfg
    if (candidates is None)!=(cfg['family']=='C'): raise ValueError('Wrong target capability')
    model.train(); model.capture=True; opt.zero_grad(set_to_none=True)
    for group in opt.param_groups: group['lr']=learning_rate(position)
    logits=forward(model,raw,device).float(); labels=torch.as_tensor(raw['labels'],device=device).long()
    target=torch.as_tensor(embedding,device=device).float().detach()
    if target.shape!=model.prediction.shape or not torch.isfinite(target).all(): raise ValueError('Invalid U100 embedding target')
    ce=F.cross_entropy(logits,labels); emb=F.smooth_l1_loss(model.prediction,target)
    timing('native_forward')
    aux=ce.new_zeros(()); evidence=[]; diagnostics={}
    if candidates is not None:
        def mine(j,features):
            # LayerNorm/MLP only: no dropout, BatchNorm, native forward or RNG.
            single=features.ndim==2; features=features[None] if single else features
            query=model.jet[j:j+1] if np.isscalar(j) else model.jet[torch.as_tensor(j,device=device)]
            scores=[]
            with torch.no_grad(),torch.autocast(torch.device(device).type,enabled=False):
                for start in range(0,features.shape[1],512):
                    b={'features':torch.as_tensor(features[:,start:start+512],device=device)}
                    scores.append(model.auxiliary(b,query=query.detach()).float().cpu().numpy())
            values=np.concatenate(scores,axis=1)
            return values[0] if single else values
        batch,evidence=candidates.batch(indices,epoch,cfg,mine=mine if cfg['law']=='Hsemi' else None)
        timing('candidate_sampling')
        batch=tensors(batch,device)
        with torch.autocast(torch.device(device).type,enabled=False):
            scores=model.auxiliary(batch); aux=auxiliary_loss(scores,batch,cfg['family'])
            timing('auxiliary_forward')
            if probe and cfg['family'] in ('Q','V'):
                gradient=torch.autograd.grad(aux,model.jet,retain_graph=True)[0]
                diagnostics['auxiliary_jet_gradient_norm']=float(gradient.detach().norm())
            with torch.no_grad():
                if probe:
                    if cfg['family'] in ('Q','V'):
                        from .structured_model import diagnostics as structured_diagnostics
                        diagnostics.update(structured_diagnostics(scores,batch,cfg['family']))
                    elif cfg['family']=='J':
                        diagnostics['retrieval_top1']=float((scores.argmax(-1)==batch['positive']).float().mean())
                    else:
                        y=batch['labels']; diagnostics.update(positive_score=float(scores[y].sigmoid().mean()),
                            negative_score=float(scores[~y].sigmoid().mean()),positive_fraction=float(y.float().mean()))
                    # The published probe is the first TRAIN batch of each
                    # epoch. Repeating it on all other batches was discarded
                    # work; these deterministic heads consume no native RNG.
                    permutation=torch.arange(len(labels),device=device)
                    for c in torch.unique(labels):
                        members=torch.where(labels==c)[0]
                        if len(members)>1: permutation[members]=members.roll(1)
                    diagnostics['within_class_swapped_fraction']=float((permutation!=torch.arange(len(labels),device=device)).float().mean())
                    for name,q in (('zero_query',torch.zeros_like(model.jet)),('within_class_shuffle',model.jet[permutation])):
                        diagnostics[name+'_loss']=float(auxiliary_loss(model.auxiliary(batch,query=q.detach()),batch,cfg['family']))
        timing('diagnostics')
    weighted=coefficient(cfg,position)*aux; total=ce+5*emb+weighted
    if not torch.isfinite(total): raise ValueError('Nonfinite objective')
    total.backward()
    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None): raise ValueError('Nonfinite gradient')
    opt.step()
    losses=np.array([float(x.detach()) for x in (ce,emb,aux,weighted,total)])
    timing('backward_optimizer')
    details=dict(sampling=evidence,probes=diagnostics)
    if measure: details['phase_seconds']={k:phases.get(k,0.) for k in PHASE_KEYS}
    return losses,details


def parents(spec):
    return dict(spec=spec['content_hash'],recipe=spec.get('recipe',recipe())['content_hash'],embedding=spec['embedding_hash'],
        prepared=spec['anchor']['evidence']['prepared'],raw=spec['raw_hash'],keys=spec['key_hash'])


def fit(folder,spec,task,train,val,bank,*,candidates=None,device='cuda',technical=False):
    cfg=setting(task)
    if (train.role!='train' or val.role!='validation' or train.coordinate_name!='D000' or val.coordinate_name!='D000'
            or train.foundation_sha256!=val.foundation_sha256 or train.foundation_sha256!=parents(spec)['prepared']):
        raise PermissionError('Expected authenticated HLT views')
    if any(not np.array_equal(getattr(bank,k),getattr(train,k)) for k in ('identities','labels')) or bank.manifest['content_hash']!=spec['embedding_hash']:
        raise ValueError('U100 target join/parent differs')
    counts=dict(train=len(train),validation=len(val))
    if not technical and counts!=spec['counts']: raise ValueError('Science population differs')
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=False)
    model=new_model(spec['imported'],task).to(device); opt=optimizer(model)
    parameter_counts=dict(trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        deployed=sum(p.numel() for p in model.mod.parameters()),auxiliary=0 if model.aux is None else sum(p.numel() for p in model.aux.parameters()))
    steps=math.ceil(len(train)/128); epochs=1 if technical else 30; history=[]; best=None; state=None; update=0
    for epoch in range(1,epochs+1):
        order=np.arange(len(train)) if technical else np.random.default_rng(np.random.SeedSequence(
            [seed(spec['imported'],cfg,'sampler_seed'),epoch])).permutation(len(train))
        sums=np.zeros(len(LOSS_KEYS)); times=[]; sizes=[]; evidence=[]; phases=[]
        for start in range(0,len(order),128):
            ix=order[start:start+128]; update+=1
            if str(device).startswith('cuda'): torch.cuda.synchronize()
            began=time.perf_counter()
            losses,details=step(model,opt,train.batch(ix),bank.targets[ix],device,candidates=candidates,indices=ix,
                epoch=epoch,position=7. if technical else update/steps,probe=start==0,measure=technical)
            if str(device).startswith('cuda'): torch.cuda.synchronize()
            times.append(time.perf_counter()-began); sizes.append(len(ix)); sums+=len(ix)*losses
            if technical:
                phase=details.pop('phase_seconds')
                phase['input_and_dispatch']=max(0.,times[-1]-sum(phase.values()))
                phases.append(phase)
            if start==0: evidence=details
            if start//128%100==0 or start+128>=len(order):
                print(f'NEXT_SUPERVISION {task} epoch={epoch}/{epochs} batch={start//128+1}/{steps}',flush=True)
        began=time.perf_counter(); probabilities=predict(model,val,device); metrics=evaluate_probabilities(val.labels,probabilities)
        row=dict(pass_number=epoch,update=update,validation=metrics,train_seconds=sum(times),validation_seconds=time.perf_counter()-began,
            **dict(zip(LOSS_KEYS,(sums/len(train)).tolist())))
        if technical: row.update(batch_seconds=times,batch_sizes=sizes,batch_phase_seconds=phases,schedule_position=7.)
        history.append(row); key=selection_key(metrics,update)
        if best is None or key>best: best=key; selected=epoch; state=model.deployment_state()
        write_immutable_json(folder/f'epoch_{epoch:02d}.json',artifact('EPOCH',parents=dict(spec=spec['content_hash']),task=task,**row))
        write_immutable_json(folder/f'probe_{epoch:02d}.json',artifact('TRAIN_PROBE',parents=dict(spec=spec['content_hash']),task=task,
            epoch=epoch,scope='first_shuffled_TRAIN_batch_not_independent_evaluation',**evidence))
        print(f'NEXT_SUPERVISION {task} epoch={epoch}/{epochs} auc={metrics["macro_ovr_auc"]:.8f} best_epoch={selected} aux={row["auxiliary_loss"]:.6f}',flush=True)
    del model,opt
    save_state(folder/'selected.pt',state); restored=deployment(torch.load(folder/'selected.pt',map_location='cpu',weights_only=True),device)
    actual=predict(restored,val,device); metrics=evaluate_probabilities(val.labels,actual)
    if metrics!=history[selected-1]['validation']: raise ValueError('Native deployment replay differs')
    atomic_publish_bytes(folder/'validation_predictions.npz',deterministic_npz_bytes(dict(identities=val.identities,labels=val.labels,probabilities=actual)))
    report=artifact('TRAINING_REPORT',parents=parents(spec),task=task,settings=cfg,technical=technical,counts=counts,
        epochs=epochs,selected_pass=selected,validation=metrics,validation_history=history,deployable_hlt_only=True,
        parameter_counts=parameter_counts,
        initialization_seed=seed(spec['imported'],cfg,'initialization_seed'),sampler_seed=seed(spec['imported'],cfg,'sampler_seed'))
    write_immutable_json(folder/'training_report.json',report)
    write_immutable_json(folder/'receipt.json',artifact('FIT',parents=dict(spec=spec['content_hash'],report=report['content_hash']),
        task=task,reload_verified=True,outputs=[file_ref(folder/n,root=folder) for n in ('selected.pt','training_report.json','validation_predictions.npz')]))
    validate_fit(folder,spec,task,technical=technical)
    return report


def validate_fit(folder,spec,task,*,technical=False):
    folder=Path(folder); report=load_json(folder/'training_report.json'); receipt=load_json(folder/'receipt.json')
    validate(report,'TRAINING_REPORT',parents(spec)); validate(receipt,'FIT',dict(spec=spec['content_hash'],report=report['content_hash']))
    cfg=setting(task); epochs=1 if technical else 30
    if (report['task']!=task or receipt['task']!=task or report['settings']!=cfg or report['technical'] is not technical
            or report['epochs']!=epochs or not report['deployable_hlt_only'] or not receipt['reload_verified']
            or (not technical and report['counts']!=spec['counts'])
            or report['initialization_seed']!=seed(spec['imported'],cfg,'initialization_seed')
            or report['sampler_seed']!=seed(spec['imported'],cfg,'sampler_seed')
            or [r['path'] for r in receipt['outputs']]!=['selected.pt','training_report.json','validation_predictions.npz']):
        raise ValueError('Fit registration differs')
    for ref in receipt['outputs']: validate_file_ref(ref,root=folder)
    history=report['validation_history']; steps=math.ceil(report['counts']['train']/128)
    if [r['pass_number'] for r in history]!=list(range(1,epochs+1)): raise ValueError('Incomplete epoch inventory')
    for r in history:
        if (not np.isfinite([r[k] for k in LOSS_KEYS]).all() or min(r[k] for k in LOSS_KEYS)<0
                or r['update']!=r['pass_number']*steps or r['validation']['rows']!=report['counts']['validation']
                or not math.isclose(r['total_loss'],r['ce_hlt']+5*r['embedding_loss']+r['weighted_auxiliary'],rel_tol=1e-6,abs_tol=1e-6)
                or (cfg['family']=='C' and (r['auxiliary_loss']!=0 or r['weighted_auxiliary']!=0))): raise ValueError('Invalid objective/schedule')
        if load_json(folder/f'epoch_{r["pass_number"]:02d}.json')!=artifact('EPOCH',parents=dict(spec=spec['content_hash']),task=task,**r):
            raise ValueError('Epoch/report mismatch')
        probe=load_json(folder/f'probe_{r["pass_number"]:02d}.json'); validate(probe,'TRAIN_PROBE',dict(spec=spec['content_hash']))
        if probe['task']!=task or probe['epoch']!=r['pass_number']: raise ValueError('Probe ownership differs')
    best=max(history,key=lambda r:selection_key(r['validation'],r['update']))
    if best['pass_number']!=report['selected_pass'] or best['validation']!=report['validation']: raise ValueError('Wrong checkpoint selection')
    a=load_npz_arrays(folder/'validation_predictions.npz'); validate_predictions(a,report['counts']['validation'])
    if evaluate_probabilities(a['labels'],a['probabilities'])!=report['validation']: raise ValueError('Saved metric mismatch')
    return report,a

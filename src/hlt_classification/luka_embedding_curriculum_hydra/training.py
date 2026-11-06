"""Production two-stage training, complete transition snapshots, native deployment."""
from pathlib import Path
from copy import deepcopy
import io
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from hlt_classification.data.cache_contracts import (load_json,load_npz_arrays,write_immutable_json,
    atomic_publish_bytes,deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_embedding_aux.model import AuxiliaryParT
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from hlt_classification.luka_u100_contrastive.training import optimizer
from hlt_classification.luka_cross_message_hydra.training import forward as native_forward, predict
from .contracts import artifact,validate,setting,weights,learning_rate,recipe

LOSS_KEYS=('ce','embedding','weighted_ce','weighted_embedding','total')


def new_model(spec,device):
    torch.manual_seed(spec['imported']['node']['initialization_seed'])
    return AuxiliaryParT().to(device)


def forward(model,raw,device):
    if set(raw)!={'features','vectors','mask','identities','labels'}: raise PermissionError('Unexpected HLT input fields')
    return native_forward(model,raw,device)


def join(cache,bank,role):
    if (cache.role!=role or cache.coordinate_name!='D000' or bank.role!=role
            or any(not np.array_equal(getattr(cache,k),getattr(bank,k)) for k in ('identities','labels'))
            or bank.targets.shape!=(len(cache),128) or bank.targets.dtype!=np.float32
            or not np.isfinite(bank.targets).all()): raise PermissionError('Wrong target role, population or identity join')


def step(model,opt,raw,target,task,position,device):
    ce_weight,emb_weight=weights(task,position)
    for group in opt.param_groups: group['lr']=learning_rate(task,position)
    model.train(); model.capture=True; opt.zero_grad(set_to_none=True)
    logits=forward(model,raw,device)
    target=torch.as_tensor(target,device=device).detach().float()
    if target.shape!=model.prediction.shape or not torch.isfinite(target).all(): raise ValueError('Invalid TRAIN embedding target')
    ce=F.cross_entropy(logits.float(),torch.from_numpy(raw['labels']).to(device).long())
    emb=F.smooth_l1_loss(model.prediction.float(),target,beta=1.,reduction='mean')
    # Exclude zero-weight graphs: zero tensors would still create AdamW decay.
    total=ce*ce_weight if emb_weight==0 else emb*emb_weight if ce_weight==0 else ce*ce_weight+emb*emb_weight
    if not torch.isfinite(total) or not torch.isfinite(ce) or not torch.isfinite(emb): raise ValueError('Nonfinite objective')
    total.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not torch.stack([torch.isfinite(g).all() for g in grads]).all(): raise ValueError('Missing/nonfinite gradients')
    opt.step(); model.prediction=None
    return np.array([float(x.detach()) for x in (ce,emb,ce*ce_weight,emb*emb_weight,total)],np.float64)


def evaluate(model,val,bank,device):
    join(val,bank,'validation'); model.eval(); model.capture=True
    probabilities=np.empty((len(val),11),np.float32); total=0.
    with torch.inference_mode():
        for start in range(0,len(val),128):
            ix=np.arange(start,min(start+128,len(val))); logits=forward(model,val.batch(ix),device)
            loss=F.smooth_l1_loss(model.prediction.float(),torch.as_tensor(bank.targets[ix],device=device),reduction='mean')
            p=logits.float().softmax(-1)
            if not torch.isfinite(p).all() or not torch.isfinite(loss): raise ValueError('Nonfinite validation')
            probabilities[ix]=p.cpu().numpy(); total+=float(loss)*len(ix)
    model.prediction=None; model.capture=False
    return evaluate_probabilities(val.labels,probabilities),total/len(val)


def cpu_copy(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_copy(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_copy(v) for v in value)
    return deepcopy(value)


def snapshot(model,opt,epoch,embedding_loss,device):
    return dict(model=cpu_copy(model.state_dict()),optimizer=cpu_copy(opt.state_dict()),
        cpu_rng=torch.get_rng_state().clone(),cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else [],
        epoch=epoch,embedding_loss=embedding_loss)


def transition(model,opt,state,task,device):
    c=setting(task); model.load_state_dict(state['model'],strict=True)
    if c['reset_optimizer']: opt=optimizer(model)
    else: opt.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['cpu_rng'])
    if str(device).startswith('cuda'): torch.cuda.set_rng_state_all(state['cuda_rng'])
    model.prediction=None
    return opt


def schedule(task,technical):
    c=setting(task)
    # A real pre-transition, post-transition and late fade update with the
    # production kernels. Scientific fits always traverse all thirty epochs.
    return [max(c['switch'],1),max(c['switch'],1)+1,30] if technical else list(range(1,31))


def fit_parents(spec,val_bank_hash):
    return dict(spec=spec['content_hash'],recipe=recipe()['content_hash'],embedding=spec['embedding_hash'],
        validation_bank=val_bank_hash,prepared=spec['anchor']['evidence']['prepared'])


def fit(folder,spec,task,train,val,bank,vbank,*,device='cuda',technical=False):
    join(train,bank,'train'); join(val,vbank,'validation')
    if (train.foundation_sha256!=val.foundation_sha256 or train.foundation_sha256!=spec['anchor']['evidence']['prepared']
            or bank.manifest['content_hash']!=spec['embedding_hash']): raise ValueError('Preparation/target parents differ')
    counts=dict(train=len(train),validation=len(val))
    if not technical and counts!=spec['counts']: raise ValueError('Population differs')
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=False)
    model=new_model(spec,device); opt=optimizer(model); cfg=setting(task)
    history=[]; pre=None; transitioned=False; selected=None; best=None; state=None; transition_ref=None
    steps=math.ceil(len(train)/128); positions=schedule(task,technical); spent=0
    for number,epoch in enumerate(positions,1):
        if cfg['switch'] and epoch>cfg['switch'] and not transitioned:
            if pre is None: raise ValueError('Missing selected pretraining snapshot')
            buffer=io.BytesIO(); torch.save(pre,buffer); atomic_publish_bytes(folder/'transition.pt',buffer.getvalue())
            transition_ref=file_ref(folder/'transition.pt',root=folder)
            restored=torch.load(folder/'transition.pt',map_location='cpu',weights_only=True)
            opt=transition(model,opt,restored,task,device); transitioned=True
            write_immutable_json(folder/'transition.json',artifact('TRANSITION',parents=fit_parents(spec,vbank.manifest['content_hash']),
                task=task,selected_epoch=pre['epoch'],embedding_validation_loss=pre['embedding_loss'],spent_pretraining_epochs=cfg['switch'],
                reset_optimizer=cfg['reset_optimizer'],restart_lr=cfg['restart_lr'],snapshot=transition_ref,clock_rewound=False))
        order=(np.arange(len(train)) if technical else np.random.default_rng(np.random.SeedSequence(
            [spec['imported']['node']['sampler_seed'],epoch])).permutation(len(train)))
        sums=np.zeros(len(LOSS_KEYS)); times=[]; sizes=[]
        for batch,start in enumerate(range(0,len(order),128),1):
            ix=order[start:start+128]; position=epoch-1+batch/steps
            if str(device).startswith('cuda'): torch.cuda.synchronize()
            began=time.perf_counter(); sums+=len(ix)*step(model,opt,train.batch(ix),bank.targets[ix],task,position,device)
            if str(device).startswith('cuda'): torch.cuda.synchronize()
            times.append(time.perf_counter()-began); sizes.append(len(ix)); spent+=1
            if batch%100==1 or batch==steps:
                print(f'CURRICULUM {task} epoch={epoch}/30 batch={batch}/{steps}',flush=True)
        began=time.perf_counter(); metrics,vloss=evaluate(model,val,vbank,device); elapsed=time.perf_counter()-began
        phase='pretrain' if cfg['pretrain'] and epoch<=cfg['pretrain'] else 'classification'
        row=dict(pass_number=epoch,completed_epochs=number,update=spent,phase=phase,validation=metrics,
            embedding_validation_loss=vloss,train_seconds=sum(times),validation_seconds=elapsed,
            learning_rate=learning_rate(task,epoch),ce_weight_end=weights(task,epoch)[0],embedding_weight_end=weights(task,epoch)[1],
            **dict(zip(LOSS_KEYS,(sums/len(train)).tolist())))
        if technical: row.update(batch_seconds=times,batch_sizes=sizes)
        history.append(row)
        if cfg['switch'] and epoch<=cfg['switch']:
            if pre is None or cfg['selection']=='last' or vloss<pre['embedding_loss']:
                pre=snapshot(model,opt,epoch,vloss,device)
        if phase=='classification':
            key=selection_key(metrics,spent)
            if best is None or key>best: best=key; selected=epoch; state=model.native_state()
        write_immutable_json(folder/f'epoch_{epoch:02d}.json',artifact('EPOCH',parents=dict(spec=spec['content_hash']),task=task,**row))
        print(f'CURRICULUM {task} epoch={epoch}/30 phase={phase} auc={metrics["macro_ovr_auc"]:.8f} '
            f'val_embed={vloss:.7f} best_classification_epoch={selected}',flush=True)
    del model,opt,pre
    if state is None: raise ValueError('No classification checkpoint')
    save_state(folder/'selected.pt',state)
    deployed=DelphesParticleTransformer().to(device)
    deployed.load_state_dict(torch.load(folder/'selected.pt',map_location='cpu',weights_only=True),strict=True)
    actual=predict(deployed,val,device); metrics=evaluate_probabilities(val.labels,actual)
    selected_row=next(r for r in history if r['pass_number']==selected)
    if metrics!=selected_row['validation']: raise ValueError('Native deployment replay differs')
    report=artifact('TRAINING_REPORT',parents=fit_parents(spec,vbank.manifest['content_hash']),task=task,settings=cfg,
        technical=technical,counts=counts,epochs=len(history),spent_epoch_budget=30,selected_pass=selected,
        validation=metrics,validation_history=history,transition=transition_ref,deployable_hlt_only=True,
        initialization_seed=spec['imported']['node']['initialization_seed'],sampler_seed=spec['imported']['node']['sampler_seed'])
    atomic_publish_bytes(folder/'validation_predictions.npz',deterministic_npz_bytes(dict(
        identities=val.identities,labels=val.labels,probabilities=actual)))
    write_immutable_json(folder/'training_report.json',report)
    names=['selected.pt','training_report.json','validation_predictions.npz']
    if transition_ref: names+=['transition.pt','transition.json']
    receipt=artifact('FIT',parents=dict(spec=spec['content_hash'],report=report['content_hash']),task=task,
        reload_verified=True,outputs=[file_ref(folder/n,root=folder) for n in names])
    write_immutable_json(folder/'receipt.json',receipt)
    validate_fit(folder,spec,task,technical=technical,val_bank_hash=vbank.manifest['content_hash'])
    return report


def validate_fit(folder,spec,task,*,technical=False,val_bank_hash=None):
    folder=Path(folder); r=load_json(folder/'training_report.json'); receipt=load_json(folder/'receipt.json')
    if val_bank_hash is None:
        m=load_json(Path(spec['root'])/'setup/validation_embeddings.json'); val_bank_hash=m['content_hash']
    validate(r,'TRAINING_REPORT',fit_parents(spec,val_bank_hash)); validate(receipt,'FIT',dict(spec=spec['content_hash'],report=r['content_hash']))
    cfg=setting(task); history=r['validation_history']; expected=schedule(task,technical); steps=math.ceil(r['counts']['train']/128)
    names=['selected.pt','training_report.json','validation_predictions.npz']+(['transition.pt','transition.json'] if cfg['switch'] else [])
    if (r['task']!=task or receipt['task']!=task or r['settings']!=cfg or r['technical'] is not technical
            or r['epochs']!=len(expected) or r['spent_epoch_budget']!=30 or not r['deployable_hlt_only'] or not receipt['reload_verified']
            or r['initialization_seed']!=spec['imported']['node']['initialization_seed']
            or r['sampler_seed']!=spec['imported']['node']['sampler_seed']
            or (not technical and r['counts']!=spec['counts']) or [x['path'] for x in receipt['outputs']]!=names
            or [x['pass_number'] for x in history]!=expected): raise ValueError('Fit registry/history differs')
    for ref in receipt['outputs']: validate_file_ref(ref,root=folder)
    for i,row in enumerate(history,1):
        e=row['pass_number']; vals=[row[k] for k in (*LOSS_KEYS,'embedding_validation_loss','train_seconds','validation_seconds')]
        if (not np.isfinite(vals).all() or min(vals)<0 or row['update']!=i*steps or row['completed_epochs']!=i
                or row['phase']!=('pretrain' if cfg['pretrain'] and e<=cfg['pretrain'] else 'classification')
                or row['learning_rate']!=learning_rate(task,e) or (row['ce_weight_end'],row['embedding_weight_end'])!=weights(task,e)
                or not math.isclose(row['total'],row['weighted_ce']+row['weighted_embedding'],rel_tol=1e-6,abs_tol=1e-6)
                or load_json(folder/f'epoch_{e:02d}.json')!=artifact('EPOCH',parents=dict(spec=spec['content_hash']),task=task,**row)):
            raise ValueError('Epoch/loss/clock evidence differs')
    eligible=[x for x in history if x['phase']=='classification']; best=max(eligible,key=lambda x:selection_key(x['validation'],x['update']))
    if best['pass_number']!=r['selected_pass'] or best['validation']!=r['validation']: raise ValueError('Wrong native checkpoint selection')
    if cfg['switch']:
        pre=[x for x in history if x['pass_number']<=cfg['switch']]
        chosen=pre[-1] if cfg['selection']=='last' else min(pre,key=lambda x:(x['embedding_validation_loss'],x['pass_number']))
        expected_transition=artifact('TRANSITION',parents=fit_parents(spec,val_bank_hash),task=task,selected_epoch=chosen['pass_number'],
            embedding_validation_loss=chosen['embedding_validation_loss'],spent_pretraining_epochs=cfg['switch'],
            reset_optimizer=cfg['reset_optimizer'],restart_lr=cfg['restart_lr'],snapshot=file_ref(folder/'transition.pt',root=folder),clock_rewound=False)
        if load_json(folder/'transition.json')!=expected_transition or r['transition']!=expected_transition['snapshot']:
            raise ValueError('Transition checkpoint/selection differs')
    elif r['transition'] is not None: raise ValueError('Control acquired a transition')
    a=load_npz_arrays(folder/'validation_predictions.npz'); validate_predictions(a,r['counts']['validation'])
    if evaluate_probabilities(a['labels'],a['probabilities'])!=r['validation']: raise ValueError('Saved metrics differ')
    return r,a

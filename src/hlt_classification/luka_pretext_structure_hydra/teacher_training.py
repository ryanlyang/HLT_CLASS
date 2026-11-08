"""Thirty-epoch pretext selection by fixed validation loss, never class AUC."""
from pathlib import Path
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.luka_u100_contrastive.training import optimizer
from .contracts import artifact, validate, TEACHERS
from .teacher_model import Teacher, objective
from .target_banks import primary
from . import corruption as co


def prepare_batch(physical,structure,task,indices,epoch,draw):
    views=[]; clean=[]; main=[]; signature=[]; applicable=[]
    for i in indices:
        original=physical.jet(i); visible,a,removed=co.corrupt(original,physical.identities[i],task,epoch,draw)
        views.append(visible); clean.append(original)
        main.append(primary(task,a,visible,removed,structure,i))
        signature.append({n:structure.target(i,n) for n in ('C2','C3','B','R')})
        applicable.append(len(a)>=2)
    return co.batch(views),co.batch(clean),main,signature,applicable


def step(model,opt,physical,structure,indices,position,epoch,device,*,draw=0,update=True,diagnose=False,phases=None):
    def stamp():
        if phases is not None and torch.device(device).type=='cuda': torch.cuda.synchronize(device)
        return time.perf_counter()
    began=stamp()
    if update and (physical.role!='train' or structure.role!='train'):
        raise PermissionError('Teacher updates require TRAIN')
    model.train(update)
    if update:
        for group in opt.param_groups: group['lr']=learning_rate(position)
        opt.zero_grad(set_to_none=True)
    view,clean,main,signature,applicable=prepare_batch(physical,structure,model.task,indices,epoch,draw)
    ready=stamp()
    with torch.set_grad_enabled(update):
        h=model.embedding(view,device); hc=model.embedding(clean,device)
        encoded=stamp()
        loss,terms=objective(model,h,hc,main,signature,applicable)
        decoded=stamp(); model.gradient_diagnostic=None
        if update and diagnose:
            params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('mod.')]
            norms={}
            for name,value in terms.items():
                gs=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
                squares=[g.detach().float().square().sum() for g in gs if g is not None]
                norms[name]=float(torch.stack(squares).sum().sqrt()) if squares else 0.
            model.gradient_diagnostic=dict(weighted_encoder_pooling_norms=norms,quality_gate=False)
        diagnosed=stamp()
        if update:
            loss.backward()
            grads=[torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
            if not grads or not torch.stack(grads).all(): raise ValueError('Nonfinite teacher gradients')
            if any(p.grad is not None for p in model.mod.fc.parameters()): raise ValueError('Class head received pretext gradients')
            opt.step()
    done=stamp()
    if phases is not None:
        phases.update(input_corruption=ready-began,native_forward=encoded-ready,
            auxiliary_forward_assignment=decoded-encoded,diagnostics=diagnosed-decoded,backward_optimizer=done-diagnosed)
    return {k:float(v.detach()) for k,v in terms.items()}


def evaluate(model,physical,structure,device):
    if physical.role!='validation': raise PermissionError('Pretext selection requires validation capability')
    sums=dict(primary=0.,clean_signature=0.)
    for start in range(0,len(physical),128):
        ix=np.arange(start,min(start+128,len(physical)))
        for draw in range(2):
            terms=step(model,None,physical,structure,ix,0.,0,device,draw=draw,update=False)
            for k,v in terms.items(): sums[k]+=v*len(ix)/2
    return {k:v/len(physical) for k,v in sums.items()}


def parents(spec,task,train,val,ts,vs,statistics):
    return dict(spec=spec['content_hash'],recipe=spec['recipe']['content_hash'],train=train.manifest['content_hash'],
        validation=val.manifest['content_hash'],train_structure=ts.manifest['content_hash'],
        validation_structure=vs.manifest['content_hash'],statistics=statistics['content_hash'])


def new_model(spec,task,maximum,statistics,signature):
    torch.manual_seed(spec['nodes'][task]['initialization_seed'])
    return Teacher(task,maximum,statistics['teachers'][task],signature)


def fit(folder,spec,task,train,val,train_structure,val_structure,statistics,maximum,*,device,technical=False):
    if (task not in TEACHERS or train.role!='train' or val.role!='validation' or train.source!='OFFLINE'
            or val.source!='OFFLINE' or train_structure.role!='train' or val_structure.role!='validation'):
        raise PermissionError('Teacher input roles differ')
    validate(statistics,'STATISTICS')
    if statistics['role']!='train': raise PermissionError('Teacher statistics require TRAIN')
    if not technical and dict(train=len(train),validation=len(val))!=spec['counts']: raise ValueError('Population differs')
    if set(map(bytes,train.identities))&set(map(bytes,val.identities)): raise ValueError('TRAIN/validation overlap')
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=False)
    p=parents(spec,task,train,val,train_structure,val_structure,statistics)
    model=new_model(spec,task,maximum,statistics,train_structure.normalizers).to(device); opt=optimizer(model)
    epochs=2 if technical else 30; steps=math.ceil(len(train)/128); history=[]; best=None; state=None
    for epoch in range(1,epochs+1):
        order=np.random.default_rng(np.random.SeedSequence([spec['nodes'][task]['sampler_seed'],epoch])).permutation(len(train))
        sums=dict(primary=0.,clean_signature=0.); started=time.perf_counter(); diagnostic=None
        for start in range(0,len(order),128):
            ix=order[start:start+128]; position=epoch-1+(start//128+1)/steps
            values=step(model,opt,train,train_structure,ix,position,epoch,device,
                diagnose=start==0 and epoch in (1,3,10,20,30))
            if model.gradient_diagnostic is not None: diagnostic=model.gradient_diagnostic
            for k,v in values.items(): sums[k]+=v*len(ix)
            if start//128%100==0: print(f'PRETEXT {task} epoch={epoch}/{epochs} batch={start//128+1}/{steps}',flush=True)
        seconds=time.perf_counter()-started; began=time.perf_counter()
        values=evaluate(model,val,val_structure,device); score=sum(values.values())
        row=artifact('TEACHER_EPOCH',parents=p,task=task,epoch=epoch,technical=technical,
            update=epoch*steps,learning_rate=learning_rate(epoch),train_losses={k:v/len(train) for k,v in sums.items()},
            validation_losses=values,selection_loss=score,train_seconds=seconds,validation_seconds=time.perf_counter()-began,
            gradient_diagnostic=diagnostic)
        history.append(row); write_immutable_json(folder/f'epoch-{epoch:02}.json',row)
        if best is None or score<best['selection_loss']:
            best=row; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f'PRETEXT {task} epoch={epoch}/{epochs} loss={score:.6f} selected={best["epoch"]}',flush=True)
    model.load_state_dict(state,strict=True); model.eval()
    save_state(folder/'selected_full.pt',state); save_state(folder/'selected.pt',model.native_state())
    replay=evaluate(model,val,val_structure,device)
    if not np.allclose(list(replay.values()),list(best['validation_losses'].values()),rtol=1e-7,atol=1e-8):
        raise ValueError('Selected pretext checkpoint replay differs')
    # Label-blind fixed identity order. Descriptive decoder dependence only.
    probe_indices=np.array(sorted(range(len(train)),key=lambda i:bytes(train.identities[i]))[:128])
    view,clean,main,signature,applicable=prepare_batch(train,train_structure,task,probe_indices,0,0)
    with torch.inference_mode():
        h=model.embedding(view,device); hc=model.embedding(clean,device)
        intact,_=objective(model,h,hc,main,signature,applicable)
        shuffled,_=objective(model,h.roll(1,0),hc.roll(1,0),main,signature,applicable)
    probe=dict(rows=len(probe_indices),intact=float(intact),shuffled_pooled_embedding=float(shuffled),quality_gate=False)
    report=artifact('TEACHER_REPORT',parents=p,task=task,technical=technical,epochs=epochs,history=history,
        counts=dict(train=len(train),validation=len(val)),maximum=maximum,selected_epoch=best['epoch'],
        selection_loss=best['selection_loss'],selection='minimum_validation_pretext_earliest_tie',
        class_labels_used=False,selected_reload=True,decoder_probe=probe)
    write_immutable_json(folder/'training_report.json',report)
    write_immutable_json(folder/'receipt.json',artifact('TEACHER_FIT',parents=dict(spec=spec['content_hash'],report=report['content_hash']),
        task=task,outputs=[file_ref(folder/name,root=folder) for name in ('selected.pt','selected_full.pt','training_report.json')]))
    validate_fit(folder,spec,task,technical=technical)
    return model,report


def validate_fit(folder,spec,task,*,technical=False):
    folder=Path(folder); r=load_json(folder/'training_report.json'); receipt=load_json(folder/'receipt.json')
    validate(r,'TEACHER_REPORT'); validate(receipt,'TEACHER_FIT',dict(spec=spec['content_hash'],report=r['content_hash']))
    if (task not in TEACHERS or r['task']!=task or receipt['task']!=task or r['technical']!=technical
            or r['epochs']!=(2 if technical else 30) or len(r['history'])!=r['epochs']
            or r['class_labels_used'] is not False or r['selected_reload'] is not True
            or r['parents'].get('spec')!=spec['content_hash'] or r['parents'].get('recipe')!=spec['recipe']['content_hash']
            or [x['path'] for x in receipt['outputs']]!=['selected.pt','selected_full.pt','training_report.json']):
        raise ValueError('Teacher completion meaning differs')
    for ref in receipt['outputs']: validate_file_ref(ref,root=folder)
    for epoch,row in enumerate(r['history'],1):
        validate(row,'TEACHER_EPOCH',r['parents'])
        if (row!=load_json(folder/f'epoch-{epoch:02}.json') or row['task']!=task or row['epoch']!=epoch
                or row['technical']!=technical or row['update']!=epoch*math.ceil(r['counts']['train']/128)
                or row['learning_rate']!=learning_rate(epoch) or row['selection_loss']!=sum(row['validation_losses'].values())
                or not np.isfinite(list(row['train_losses'].values())+list(row['validation_losses'].values())).all()):
            raise ValueError('Teacher epoch evidence differs')
    best=min(r['history'],key=lambda row:(row['selection_loss'],row['epoch']))
    if r['selected_epoch']!=best['epoch'] or r['selection_loss']!=best['selection_loss']:
        raise ValueError('Teacher task-only selection differs')
    return r

"""Matched full30 training, second-order microbatches and holdout-blind fold fits."""
from pathlib import Path
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json,load_npz_arrays,write_immutable_json,atomic_publish_bytes,deterministic_npz_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import _forward,predict,selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.luka_u100_contrastive.training import optimizer,recovery_percent
from hlt_classification.luka_u100_contrastive.resources import Subset
from hlt_classification.luka_teacher_sweep_hydra.training import gradients
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from .contracts import TEACHERS,FOLDS,STUDENTS,FITS,artifact,validate,term_names,recipe
from .model import new_model,objective
from . import explanations,folds


def clear(model):
    model.h=model.predictions=None


def step(model,task,opt,raw,banks,indices,position,device,*,context=None,diagnose=False):
    context=context or {}; model.train(); model.capture=True
    for group in opt.param_groups: group['lr']=learning_rate(position)
    opt.zero_grad(set_to_none=True)
    logits=_forward(model,raw,device,bf16=torch.device(device).type=='cuda')
    targets={k:torch.as_tensor(b.targets[indices],device=device) for k,b in banks.items()}
    difficulty=torch.as_tensor(context['weights'][indices],device=device) if task=='TF' else None
    loss,terms=objective(model,task,logits,torch.as_tensor(raw['labels'],device=device),targets,difficulty)
    diag=gradients(terms,model) if diagnose else None
    loss.backward(); values={k:float(v.detach()) for k,v in terms.items()}; clear(model)
    del loss,terms,logits,targets
    if task=='TE':
        reference=context['reference']; total=0.; coverage=0.; energy=0.; micro=recipe()['explanation']['microbatch']
        for start in range(0,len(indices),micro):
            stop=min(start+micro,len(indices)); part={k:torch.as_tensor(raw[k][start:stop],device=device) for k in ('features','vectors','mask','labels')}
            penalty,details=explanations.penalty(model,reference,part)
            fraction=(stop-start)/len(indices); (penalty*fraction).backward()
            total+=float(penalty.detach())*fraction; coverage+=details['mask_fraction']*fraction; energy+=details['reference_energy']*fraction
            del penalty
        values['explanation']=total
        if diagnose:
            # Keep the original diagnostic's final eight jets independent of
            # explanation microbatch size. It does not affect the objective.
            count=recipe()['explanation']['diagnostic_rows']
            part={k:torch.as_tensor(raw[k][-count:],device=device) for k in ('features','vectors','mask','labels')}
            diag.update(explanation=dict(mask_fraction=coverage,reference_energy=energy),
                        finite_response=explanations.finite_response(model,reference,part))
    checks=[torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    if not checks or not torch.stack(checks).all(): raise ValueError('Missing/nonfinite required gradients')
    opt.step(); clear(model)
    return values,diag


def parents(spec,task,banks,context):
    return dict(spec=spec['content_hash'],recipe=spec['recipe']['content_hash'],
        **{k:b.manifest['content_hash'] for k,b in banks.items()},**context.get('parents',{}))


def check_inputs(spec,task,train,val,banks,context,technical):
    teacher=task in (*TEACHERS,*FOLDS); coordinate='OFFLINE' if teacher else 'D000'
    if (task not in FITS or train.role!='train' or train.coordinate_name!=coordinate
        or train.foundation_sha256!=spec['anchor']['evidence']['prepared']
        or set(banks)!=set(STUDENTS.get(task,()))): raise PermissionError('Input/target roles differ')
    if len(np.unique(train.identities,axis=0))!=len(train): raise ValueError('Duplicated TRAIN identities')
    if task in FOLDS:
        if val is not None: raise PermissionError('No holdout or development validation access during OOF fit')
        if not technical:
            _,a=folds.load(spec); keep=a['fold']!=int(task[1])
            if any(not np.array_equal(getattr(train,k),a[k][keep]) for k in ('identities','labels')):
                raise PermissionError('OOF training includes wrong fold/member/order')
    else:
        if (val is None or val.role!='validation' or val.coordinate_name!=coordinate
            or val.foundation_sha256!=train.foundation_sha256 or len(np.unique(val.identities,axis=0))!=len(val)
            or set(map(bytes,train.identities)) & set(map(bytes,val.identities))): raise PermissionError('Validation boundary differs')
        if not technical and dict(train=len(train),validation=len(val))!=spec['counts']: raise ValueError('Scientific population differs')
    for b in banks.values():
        if b.manifest['role']!='train' or b.targets.shape!=(len(train),128) or any(not np.array_equal(getattr(b,k),getattr(train,k)) for k in ('identities','labels')):
            raise ValueError('Ordered TRAIN target join differs')
    if task=='TE' and 'reference' not in context: raise ValueError('Missing frozen T0')
    if task=='TF' and (context.get('weights',np.empty(0)).shape!=(len(train),) or not np.isfinite(context['weights']).all() or np.any(context['weights']<=0)):
        raise ValueError('Missing/nonfinite/zero difficulty weights')


def fit(folder,spec,task,train,val,banks,*,device,technical=False,context=None):
    context=context or {}; check_inputs(spec,task,train,val,banks,context,technical)
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=False); lineage=parents(spec,task,banks,context)
    model=new_model(spec,task).to(device); opt=optimizer(model); steps=math.ceil(len(train)/128); epochs=2 if technical else 30
    history=[]; best=None; selected=None; best_state=None
    for epoch in range(1,epochs+1):
        order=np.random.default_rng(np.random.SeedSequence([spec['nodes'][task]['sampler_seed'],epoch])).permutation(len(train))
        sums={}; diagnostic=None; started=time.perf_counter()
        for start in range(0,len(order),128):
            ix=order[start:start+128]; update=(epoch-1)*steps+start//128+1
            values,diag=step(model,task,opt,train.batch(ix),banks,ix,update/steps,device,context=context,
                diagnose=start==0 and epoch in (1,3,10,20,30))
            if diag is not None: diagnostic=diag
            for k,v in values.items(): sums[k]=sums.get(k,0.)+v*len(ix)
            if start//128%100==0 or start+128>=len(order):
                print(f'COMPLEMENTARY {task} epoch={epoch}/{epochs} batch={start//128+1}/{steps}',flush=True)
        seconds=time.perf_counter()-started; model.capture=False; clear(model); started=time.perf_counter()
        metrics=None if task in FOLDS else evaluate_probabilities(val.labels,predict(model,val,device=device,batch_size=128))
        row=artifact('EPOCH',parents=lineage,task=task,epoch=epoch,update=update,learning_rate=learning_rate(epoch),
            validation=metrics,weighted_losses={k:v/len(train) for k,v in sums.items()},gradient_diagnostic=diagnostic,
            train_seconds=seconds,validation_seconds=time.perf_counter()-started,technical=technical)
        history.append(row); key=(epoch,) if task in FOLDS else selection_key(metrics,update)
        if best is None or key>best:
            best,selected=key,row; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        write_immutable_json(folder/f'epoch-{epoch:02}.json',row)
        score='TRAIN only; holdout unseen' if metrics is None else f'AUC={metrics["macro_ovr_auc"]:.8f} recovery={recovery_percent(metrics["macro_ovr_auc"],spec["recovery_anchors"]):.1f}%'
        print(f'COMPLEMENTARY {task} epoch={epoch}/{epochs} {score} best_epoch={selected["epoch"]} weighted_losses={row["weighted_losses"]}',flush=True)
    model.load_state_dict(best_state,strict=True); model.eval(); model.capture=False; clear(model)
    save_state(folder/'selected_full.pt',best_state); save_state(folder/'selected.pt',model.native_state())
    with torch.random.fork_rng(devices=[]): native=DelphesParticleTransformer().to(device).eval()
    native.load_state_dict(torch.load(folder/'selected.pt',map_location='cpu',weights_only=True),strict=True)
    replay=Subset(train,np.arange(min(128,len(train)))) if task in FOLDS else val
    actual=predict(native,replay,device=device,batch_size=128); expected=predict(model,replay,device=device,batch_size=128)
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
    if task not in FOLDS and evaluate_probabilities(val.labels,actual)!=selected['validation']: raise ValueError('Selected native metric replay differs')
    name='training_replay.npz' if task in FOLDS else 'validation_predictions.npz'
    atomic_publish_bytes(folder/name,deterministic_npz_bytes(dict(identities=replay.identities,labels=replay.labels,probabilities=actual)))
    report=artifact('TRAINING_REPORT',parents=lineage,task=task,technical=technical,node=spec['nodes'][task],
        counts=dict(train=len(train),validation=0 if val is None else len(val)),epochs=epochs,selected_epoch=selected['epoch'],
        validation=selected['validation'],history=history,native_reload=True,oracle=task in TEACHERS,
        selection='last_epoch_no_holdout_access' if task in FOLDS else 'full_validation',
        trained_identity_hash=folds.identity_hash(train.identities))
    write_immutable_json(folder/'training_report.json',report)
    write_immutable_json(folder/'receipt.json',artifact('FIT',parents=dict(spec=spec['content_hash'],report=report['content_hash']),task=task,
        outputs=[file_ref(folder/n,root=folder) for n in ('selected.pt','selected_full.pt',name,'training_report.json')]))
    validate_fit(folder,spec,task,technical=technical)
    return native,report


def validate_fit(folder,spec,task,*,technical=False):
    folder=Path(folder); report=load_json(folder/'training_report.json'); receipt=load_json(folder/'receipt.json')
    validate(report,'TRAINING_REPORT'); validate(receipt,'FIT',dict(spec=spec['content_hash'],report=report['content_hash']))
    name='training_replay.npz' if task in FOLDS else 'validation_predictions.npz'
    if receipt['task']!=task or [r['path'] for r in receipt['outputs']]!=['selected.pt','selected_full.pt',name,'training_report.json']:
        raise ValueError('Fit inventory differs')
    for ref in receipt['outputs']: validate_file_ref(ref,root=folder)
    epochs=2 if technical else 30
    expected_parents={'spec','recipe',*STUDENTS.get(task,())}
    if task=='TE': expected_parents.add('reference')
    if task=='TF': expected_parents.add('difficulty')
    if task in FOLDS: expected_parents.add('folds')
    if (report['task']!=task or report['technical']!=technical or report['epochs']!=epochs or len(report['history'])!=epochs
        or report['node']!=spec['nodes'][task] or report['native_reload'] is not True or report['oracle']!=(task in TEACHERS)
        or report['parents'].get('spec')!=spec['content_hash'] or report['parents'].get('recipe')!=spec['recipe']['content_hash']
        or set(report['parents'])!=expected_parents): raise ValueError('Training report meaning differs')
    expected_counts=spec['counts']
    if task in FOLDS:
        if report['counts']['validation']!=0 or report['selection']!='last_epoch_no_holdout_access': raise ValueError('Fold consulted validation')
        if not technical:
            fm,a=folds.load(spec); ix=a['fold']!=int(task[1]); expected_counts=dict(train=int(ix.sum()),validation=0)
            if report['trained_identity_hash']!=folds.identity_hash(a['identities'][ix]) or report['parents']['folds']!=fm['content_hash']:
                raise ValueError('OOF training lineage differs')
    elif report['selection']!='full_validation': raise ValueError('Selection rule differs')
    if not technical and report['counts']!=expected_counts: raise ValueError('Scientific fit counts differ')
    for e,row in enumerate(report['history'],1):
        validate(row,'EPOCH',report['parents'])
        if (row!=load_json(folder/f'epoch-{e:02}.json') or row['task']!=task or row['epoch']!=e or row['technical']!=technical
            or row['update']!=e*math.ceil(report['counts']['train']/128) or row['learning_rate']!=learning_rate(e)
            or set(row['weighted_losses'])!=term_names(task) or not np.isfinite(list(row['weighted_losses'].values())).all()
            or (task in FOLDS and row['validation'] is not None)
            or (task not in FOLDS and row['validation']['rows']!=report['counts']['validation'])): raise ValueError('Epoch evidence differs')
    selected=report['history'][-1] if task in FOLDS else max(report['history'],key=lambda r:selection_key(r['validation'],r['update']))
    if report['selected_epoch']!=selected['epoch'] or report['validation']!=selected['validation']: raise ValueError('Selected epoch differs')
    arrays=load_npz_arrays(folder/name); validate_predictions(arrays,min(128,report['counts']['train']) if task in FOLDS else report['counts']['validation'])
    if task not in FOLDS and evaluate_probabilities(arrays['labels'],arrays['probabilities'])!=report['validation']: raise ValueError('Prediction metric differs')
    return report,arrays

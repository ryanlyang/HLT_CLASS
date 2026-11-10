"""Real production miniatures and warmed A6000 admission; never a quality gate."""
from pathlib import Path
from types import SimpleNamespace
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json,load_npz_arrays,write_immutable_json,atomic_publish_bytes,deterministic_npz_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer,installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.runner import _forward,predict
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_fullsim.preflight_v2 import memory_snapshot
from hlt_classification.luka_u100_multiview.views import lengths
from hlt_classification.luka_u100_contrastive.resources import Subset
from hlt_classification.luka_u100_contrastive.training import optimizer
from hlt_classification.luka_teacher_sweep_hydra.resources import free,mini_indices,subset_banks
from hlt_classification.luka_embedding_aux import targets as target_impl
from .contracts import TEACHERS,FOLDS,STUDENTS,FITS,artifact,validate,recipe
from .model import new_model
from . import inputs,training,folds

PROFILE_TASKS=FITS


def subset_context(context,ix):
    result=dict(context)
    if 'weights' in result: result['weights']=result['weights'][ix]
    return result


def miniature(folder,spec,task,train,val,banks,context,device):
    ix=mini_indices(train); vi=mini_indices(val) if val is not None else None
    native,report=training.fit(folder,spec,task,Subset(train,ix),None if val is None else Subset(val,vi),
        subset_banks(banks,ix),device=device,technical=True,context=subset_context(context,ix))
    if task in TEACHERS:
        target_impl.export(Path(folder)/'bank',native,Subset(train,ix),teacher_ref=file_ref(Path(folder)/'selected.pt'),device=device)
        target_impl.Bank(Path(folder)/'bank',Subset(train,ix),teacher_ref=file_ref(Path(folder)/'selected.pt'))
    del native
    return report


def replay(spec,val,device):
    b=inputs.Bundle(spec['bundle_root']); saved=load_npz_arrays(b.path('reference/matched/predictions.npz'))
    if any(not np.array_equal(saved[k],getattr(val,k)) for k in ('identities','labels')): raise ValueError('Native replay population differs')
    with torch.random.fork_rng(devices=[]): native=DelphesParticleTransformer().to(device).eval()
    cp=file_ref(b.path('reference/matched/checkpoint.pt')); native.load_state_dict(torch.load(cp['path'],map_location='cpu',weights_only=True),strict=True)
    actual=predict(native,val,device=device,batch_size=128); folder=Path(spec['root'])/'setup'
    atomic_publish_bytes(folder/'replay_predictions.npz',deterministic_npz_bytes(dict(identities=val.identities,labels=val.labels,probabilities=actual)))
    old=evaluate_probabilities(val.labels,saved['probabilities'])['macro_ovr_auc']; new=evaluate_probabilities(val.labels,actual)['macro_ovr_auc']
    value=artifact('REPLAY',parents=dict(spec=spec['content_hash']),checkpoint=cp,payload=file_ref(folder/'replay_predictions.npz',root=folder),
        historical_auc=old,actual_auc=new,difference=abs(new-old),tolerance=5e-5)
    write_immutable_json(folder/'replay.json',value)
    if abs(new-old)>5e-5: raise ValueError('Historical BF16 numerical replay differs')
    del native;free()
    return value


def check_replay(spec):
    folder=Path(spec['root'])/'setup'; m=load_json(folder/'replay.json');validate(m,'REPLAY',dict(spec=spec['content_hash']))
    b=inputs.Bundle(spec['bundle_root']); saved=load_npz_arrays(b.path('reference/matched/predictions.npz'))
    if m['checkpoint']!=file_ref(b.path('reference/matched/checkpoint.pt')) or m['payload']['path']!='replay_predictions.npz': raise ValueError('Replay lineage differs')
    validate_file_ref(m['payload'],root=folder); actual=load_npz_arrays(folder/'replay_predictions.npz')
    if any(not np.array_equal(saved[k],actual[k]) for k in ('identities','labels')): raise ValueError('Replay join differs')
    old=evaluate_probabilities(saved['labels'],saved['probabilities'])['macro_ovr_auc']; new=evaluate_probabilities(actual['labels'],actual['probabilities'])['macro_ovr_auc']
    if m['historical_auc']!=old or m['actual_auc']!=new or m['difference']!=abs(new-old) or m['tolerance']!=5e-5 or abs(new-old)>5e-5:
        raise ValueError('Replay values differ')
    return m


def technical_context(spec,task,train,device):
    if task=='TE':
        cp=file_ref(Path(spec['root'])/'setup/mini_T0/selected.pt')
        with torch.random.fork_rng(devices=[]): ref=DelphesParticleTransformer().to(device).eval()
        ref.load_state_dict(torch.load(cp['path'],map_location='cpu',weights_only=True),strict=True)
        for p in ref.parameters(): p.requires_grad_(False)
        return dict(reference=ref,parents=dict(reference=cp['sha256']))
    if task=='TF':
        value=artifact('TECHNICAL_DIFFICULTY',role='train',science_reusable=False)
        return dict(weights=np.ones(len(train),np.float32),parents=dict(difficulty=value['content_hash']))
    if task in FOLDS: return dict(parents=dict(folds=folds.load(spec)[0]['content_hash']))
    return {}


def oof_miniature(spec,train,device):
    """Production fits with explicitly technical, disjoint TRAIN folds and no selection data."""
    ix=mini_indices(train); small=Subset(train,ix); assignments=folds.assignment(small.identities,small.labels)
    pred=np.empty((len(small),11),np.float32); refs=[]
    for task in FOLDS:
        hold=np.flatnonzero(assignments==int(task[1])); keep=np.flatnonzero(assignments!=int(task[1]))
        folder=Path(spec['root'])/'setup'/('oof_'+task)
        ref=artifact('TECHNICAL_FOLDS',role='train',science_reusable=False,identities=folds.identity_hash(small.identities))
        native,report=training.fit(folder,spec,task,Subset(small,keep),None,{},device=device,technical=True,context=dict(parents=dict(folds=ref['content_hash'])))
        if set(map(bytes,small.identities[keep])) & set(map(bytes,small.identities[hold])): raise ValueError('Technical fold overlap')
        pred[hold]=predict(native,Subset(small,hold),device=device,batch_size=128)
        if report['selected_epoch']!=2 or report['validation'] is not None: raise ValueError('Fold selection consulted heldout')
        refs.append(file_ref(folder/'receipt.json')); del native;free()
    weights=folds.difficulty(small.labels,pred)
    value=artifact('OOF_MINIATURE',parents=dict(spec=spec['content_hash']),role='technical_train_only',
        fits=refs,rows=len(small),finite_positive=bool(np.isfinite(weights).all() and (weights>0).all()),
        class_means=[float(weights[small.labels==k].mean()) for k in range(11)],holdout_blind_selection=True)
    write_immutable_json(Path(spec['root'])/'setup/oof_miniature.json',value)
    return value


def measure(spec,device='cuda'):
    if torch.device(device).type!='cuda': raise PermissionError('Real GPU qualification required')
    if installed_environment()!=spec['reference']['environment']: raise ValueError('Environment differs')
    root=Path(spec['root'])/'setup'; profiles={}; status='failed';error=None
    torch.cuda.reset_peak_memory_stats()
    try:
        for teacher,tasks in ((True,('T0',*FOLDS,'TE','TF')),(False,tuple(STUDENTS))):
            started=time.perf_counter(); full,val=inputs.views(spec,teacher)
            expected=load_npz_arrays(inputs.Bundle(spec['bundle_root']).path('targets/u100/targets.npz'))
            if any(not np.array_equal(expected[k],getattr(full,k)) for k in ('identities','labels')): raise ValueError('Original TRAIN population differs')
            del expected
            expected=load_npz_arrays(inputs.Bundle(spec['bundle_root']).path('reference/matched/predictions.npz'))
            if any(not np.array_equal(expected[k],getattr(val,k)) for k in ('identities','labels')): raise ValueError('Original validation population differs')
            del expected
            if teacher:
                folds.build(spec,full);oof_miniature(spec,full,device)
            else: replay(spec,val,device)
            load_seconds=time.perf_counter()-started
            for task in tasks:
                print('COMPLEMENTARY setup profiling '+task,flush=True)
                train,hold=folds.split(spec,full,task) if task in FOLDS else (full,None)
                selection=None if task in FOLDS else val
                context=technical_context(spec,task,train,device)
                banks={t:SimpleNamespace(targets=np.zeros((len(train),128),np.float32),identities=train.identities,labels=train.labels,
                    manifest=artifact('TECHNICAL_BANK',role='train',teacher=t,science_reusable=False)) for t in STUDENTS.get(task,())}
                model=new_model(spec,task).to(device).eval();model.capture=False
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(spec['nodes'][task]['initialization_seed']);native=DelphesParticleTransformer().to(device).eval()
                raw=train.batch(np.arange(min(128,len(train))))
                with torch.no_grad(): torch.testing.assert_close(_forward(model,raw,device,bf16=False),_forward(native,raw,device,bf16=False),rtol=0,atol=0)
                del model,native;free()
                miniature(root/('mini_'+task),spec,task,train,selection,banks,context,device)
                model=new_model(spec,task).to(device);opt=optimizer(model)
                order=np.random.default_rng(20261109).permutation(len(train))
                def timed(ix,diagnose=False):
                    torch.cuda.synchronize();start=time.perf_counter()
                    training.step(model,task,opt,train.batch(ix),banks,ix,3.,device,context=context,diagnose=diagnose)
                    torch.cuda.synchronize();return time.perf_counter()-start
                all_lengths=lengths(full); ll=all_lengths[train.indices] if task in FOLDS else all_lengths
                timed(np.argsort(ll,kind='stable')[-min(128,len(train)):]);timed(order[:11]);timed(order[:128]);timed(order[128:256] if len(train)>128 else order[:128])
                # Always use full representative batches, even on small synthetic validation fixtures.
                samples=[timed(np.resize(order,14*128)[b*128:(b+1)*128]) for b in range(2,14)]
                probe=timed(order[:128],True);model.eval();model.capture=False;training.clear(model)
                start=time.perf_counter()
                if selection is not None: predict(model,selection,device=device,batch_size=128)
                torch.cuda.synchronize();validation_seconds=time.perf_counter()-start if selection is not None else 0.
                start=time.perf_counter()
                with torch.no_grad():
                    for _ in range(5): _forward(model,raw,device,bf16=False)
                torch.cuda.synchronize();per_export=(time.perf_counter()-start)/5
                export_rows=len(full) if task in TEACHERS else len(hold) if task in FOLDS else 0
                profiles[task]=dict(initial_native_fp32_parity=True,miniature_complete=True,deployment_reload=True,
                    largest_support_backward=True,partial_batch_backward=True,ordered_reference_joins=True,
                    train_rows=len(train),validation_rows=0 if selection is None else len(selection),export_rows=export_rows,
                    step_samples=samples,step_seconds=float(np.mean(samples)),validation_seconds=validation_seconds,
                    probe_seconds=probe,load_seconds=load_seconds,export_seconds=per_export*math.ceil(export_rows/128)*1.5,
                    second_order_both_inputs=task=='TE',targets_technical=task in STUDENTS or task=='TF',
                    explanation_microbatch=recipe()['explanation']['microbatch'] if task=='TE' else None)
                print(f'COMPLEMENTARY setup {task}: step={np.mean(samples):.3f}s validation={validation_seconds:.1f}s',flush=True)
                del model,opt,banks,context,train,hold;free()
            del full,val;free()
        status='complete'
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}';raise
    finally:
        diagnostic=artifact('RESOURCES',parents=dict(spec=spec['content_hash']),status=status,error=error,profiles=profiles,
            environment=installed_environment(),gpu=gpu_identity(),memory=memory_snapshot(torch))
        write_immutable_json(root/'resource_diagnostic.json',diagnostic)
    value=accept(spec,diagnostic);write_immutable_json(root/'acceptance.json',value);return value


def accept(spec,d):
    validate(d,'RESOURCES',dict(spec=spec['content_hash']))
    if d['status']!='complete' or d['error'] is not None or set(d['profiles'])!=set(PROFILE_TASKS): raise ValueError('Incomplete setup')
    if d['environment']!=spec['reference']['environment'] or d['gpu']['name']!='NVIDIA RTX A6000' or d['gpu']['compute_capability']!=[8,6]: raise ValueError('Wrong GPU/environment')
    if d['memory']['peak_rss_bytes']>=.85*spec['memory_mb']*1024**2 or d['memory']['gpu_peak_bytes']>=.88*d['gpu']['total_memory_bytes']:
        raise MemoryError('Memory envelope exceeded; preserve diagnostics')
    fm,_=folds.load(spec);minutes={}
    for t,p in d['profiles'].items():
        rows=spec['counts']['train']-(fm['counts'][int(t[1])] if t in FOLDS else 0)
        vr=0 if t in FOLDS else spec['counts']['validation']
        er=spec['counts']['train'] if t in TEACHERS else fm['counts'][int(t[1])] if t in FOLDS else 0
        if (not all(p[k] is True for k in ('initial_native_fp32_parity','miniature_complete','deployment_reload','largest_support_backward','partial_batch_backward','ordered_reference_joins'))
            or p['train_rows']!=rows or p['validation_rows']!=vr or p['export_rows']!=er or len(p['step_samples'])!=12
            or not np.isfinite(p['step_samples']).all() or min(p['step_samples'])<=0 or p['step_seconds']!=float(np.mean(p['step_samples']))
            or p['second_order_both_inputs']!=(t=='TE') or p['targets_technical']!=(t in STUDENTS or t=='TF')
            or p.get('explanation_microbatch')!=(recipe()['explanation']['microbatch'] if t=='TE' else None)): raise ValueError('Profile coverage differs')
        times=[p[k] for k in ('validation_seconds','probe_seconds','load_seconds','export_seconds')]
        if not np.isfinite(times).all() or min(times)<0: raise ValueError('Invalid resource timing')
        seconds=30*(math.ceil(rows/128)*p['step_seconds']+p['validation_seconds'])+5*p['probe_seconds']+p['load_seconds']+p['export_seconds']
        minutes[t]=max(60,math.ceil((2*seconds+900)/60))+15
        if minutes[t]>spec['fit_max_minutes']: raise RuntimeError(f'{t} exceeds fit ceiling: {minutes[t]} minutes; preserve diagnostics')
    return artifact('ACCEPTANCE',parents=dict(spec=spec['content_hash'],diagnostic=d['content_hash']),minutes=minutes,
        gpu=d['gpu'],environment=d['environment'],technical_weights_discarded=True)


def accepted(spec):
    folder=Path(spec['root'])/'setup'; value=load_json(folder/'acceptance.json')
    if value!=accept(spec,load_json(folder/'resource_diagnostic.json')): raise ValueError('Acceptance differs from measurements')
    check_replay(spec)
    m=load_json(folder/'oof_miniature.json');validate(m,'OOF_MINIATURE',dict(spec=spec['content_hash']))
    if m['role']!='technical_train_only' or not m['finite_positive'] or not m['holdout_blind_selection'] or len(m['fits'])!=3 or not np.allclose(m['class_means'],1.,atol=1e-6,rtol=0):
        raise ValueError('OOF miniature differs')
    for t,r in zip(FOLDS,m['fits']):
        if r!=file_ref(folder/('oof_'+t)/'receipt.json'): raise ValueError('OOF miniature ownership differs')
        training.validate_fit(folder/('oof_'+t),spec,t,technical=True)
    return value

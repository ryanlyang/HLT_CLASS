"""Measured shared setup, independent continuations and authenticated reporting."""
from pathlib import Path
from copy import deepcopy
import gc
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json,load_npz_arrays,write_immutable_json,
    atomic_publish_bytes,deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer,installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities,recovery
from hlt_classification.jetclass2_delphes.acceptance import installed_parity
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog,memory_snapshot
from hlt_classification.luka_fullstat_u100_embedding.worker import data
from hlt_classification.luka_fullstat_attention.worker import metadata as cache_metadata
from .execution import execution
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from hlt_classification.luka_fullstat_u100_embedding import banks
from hlt_classification.luka_fullstat_u100_embedding.contracts import validate as validate_bank
from hlt_classification.luka_embedding_aux.preflight import parity
from .contracts import artifact,validate,recipe,TASKS,ARMS
from . import campaign,source,submission,training,model as models
from hlt_classification.luka_meta_embedding import meta as meta_engine

REFS=('M0HLT','OFFLINE','U100_EMBED_W5')
FIT_FILES=('selected.pt','training_state.pt','completed_training.json','training_report.json','validation_predictions.npz')
SETUP_FILES=['initial/student.pt','initial/classifier.pt','initial/statistics.npz',
    'setup/initializer.json','setup/measurement.json','setup/checks.json']+[
    'references/'+n+'.npz' for n in REFS]+['setup/probe_'+a+'/'+n for a in ARMS for n in FIT_FILES]


def output_names(task):
    return SETUP_FILES if task=='setup' else [task+'/fit/'+n for n in FIT_FILES]


def fit_parents(spec):
    root=Path(spec['root']); init=load_json(root/'setup/initializer.json')
    manifest=load_json(validate_file_ref(spec['bank_manifest']))
    return dict(spec=spec['content_hash'],initializer=init['content_hash'],bank=manifest['content_hash'],
                prepared=spec['anchor']['evidence']['prepared'])


def snapshot_references(spec,val,metadata):
    root=Path(spec['root'])/'references'; root.mkdir(exist_ok=False)
    refs={n:spec['anchor']['evidence']['models'][n]['predictions'] for n in ('M0HLT','OFFLINE')}
    refs['U100_EMBED_W5']=spec['reuse']['reference']['predictions']
    for name,ref in refs.items():
        validate_file_ref(ref); arrays=load_npz_arrays(ref['path']); validate_predictions(arrays,len(val))
        if not np.array_equal(arrays['identities'],val.identities) or not np.array_equal(arrays['labels'],val.labels):
            raise ValueError('Reference validation populations differ')
        atomic_publish_bytes(root/(name+'.npz'),deterministic_npz_bytes(arrays)); validate_file_ref(ref)


def load_inputs(spec,metadata=None):
    metadata=cache_metadata(spec) if metadata is None else metadata
    train,val=(data(spec,role,'D000',authenticated=metadata) for role in ('train','validation'))
    directory=Path(validate_file_ref(spec['bank_manifest'])).parent
    bank=banks.Bank(directory,train,refs=spec['teacher_refs'])
    if bank.manifest['content_hash']!=spec['reuse']['bank_hash']:
        raise ValueError('Original U100 bank changed')
    return train,val,bank


def read_shared(spec):
    root=Path(spec['root']); record=load_json(root/'setup/initializer.json')
    validate(record,'INITIALIZER',dict(spec=spec['content_hash'],bank=load_json(validate_file_ref(spec['bank_manifest']))['content_hash'],
        initial_checkpoint=spec['initial_checkpoint']['sha256']))
    if [r['path'] for r in record['outputs']]!=['initial/student.pt','initial/classifier.pt','initial/statistics.npz']:
        raise ValueError('Shared initialization inventory differs')
    for ref in record['outputs']: validate_file_ref(ref,root=root)
    if record['calibration']['native_unchanged'] is not True or len(record['calibration']['history'])!=3:
        raise ValueError('Shared calibration incomplete')
    state=torch.load(root/'initial/student.pt',map_location='cpu',weights_only=True)
    classifier=torch.nn.Linear(128,11)
    classifier.load_state_dict(torch.load(root/'initial/classifier.pt',map_location='cpu',weights_only=True),strict=True)
    for p in classifier.parameters(): p.requires_grad_(False)
    arrays=load_npz_arrays(root/'initial/statistics.npz')
    if (set(arrays)!={'mean','scale'} or any(v.dtype!=np.float64 or v.shape!=(128,)
            or not np.isfinite(v).all() for v in arrays.values()) or np.any(arrays['scale']<1e-3)):
        raise ValueError('Original U100 normalization is invalid')
    return state,classifier,torch.from_numpy(arrays['mean']).float(),torch.from_numpy(arrays['scale']).float()


def fit(folder,spec,train,val,bank,arm,device,*,technical=False):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=False)
    initial,classifier,mean,scale=read_shared(spec); model=models.new_model(spec['node'])
    parents=fit_parents(spec)
    report,native_state,full=training.train(model,train,val,bank,spec['node'],arm,device,
        initial=initial,classifier=classifier,mean=mean,scale=scale,parents=parents,technical=technical)
    # Preserve the completed expensive fit before publication/readback checks.
    save_state(folder/'selected.pt',native_state); save_state(folder/'training_state.pt',full)
    write_immutable_json(folder/'completed_training.json',report)
    training.validate_report(report,spec,parents,arm,technical=technical)
    init=load_json(Path(spec['root'])/'setup/initializer.json')
    if report['split_sha256']!=init['calibration']['split_sha256']: raise ValueError('TRAIN split changed')
    restored=torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)
    native=DelphesParticleTransformer().to(device).eval(); native.load_state_dict(restored,strict=True)
    expected=predict(model,val,device=device,batch_size=128); actual=predict(native,val,device=device,batch_size=128)
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
    if evaluate_probabilities(val.labels,actual)!=report['selected']['validation']:
        raise ValueError('Selected native checkpoint probability readback differs')
    restored_full=torch.load(folder/'training_state.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(restored_full['student'],strict=True)
    for k,v in native_state.items():
        if not torch.equal(v,model.native_state()[k]): raise ValueError('Full/native selected state mismatch')
    arrays=dict(identities=val.identities,labels=val.labels,probabilities=actual)
    atomic_publish_bytes(folder/'validation_predictions.npz',deterministic_npz_bytes(arrays))
    record=artifact('FIT',parents=dict(report=report['content_hash']),report=report,
        checkpoint=file_ref(folder/'selected.pt'),training_state=file_ref(folder/'training_state.pt'),
        native_probability_readback=True)
    write_immutable_json(folder/'training_report.json',record)
    return record


def validate_fit(folder,spec,arm,*,technical=False):
    folder=Path(folder); record=load_json(folder/'training_report.json'); report=record['report']
    validate(record,'FIT',dict(report=report['content_hash']))
    training.validate_report(report,spec,fit_parents(spec),arm,technical=technical)
    if load_json(folder/'completed_training.json')!=report:
        raise ValueError('Completed training snapshot differs')
    if (record['checkpoint']!=file_ref(folder/'selected.pt') or record['training_state']!=file_ref(folder/'training_state.pt')
            or record['native_probability_readback'] is not True
            or report['split_sha256']!=load_json(Path(spec['root'])/'setup/initializer.json')['calibration']['split_sha256']):
        raise ValueError('Selected checkpoint/initializer binding differs')
    arrays=load_npz_arrays(folder/'validation_predictions.npz'); validate_predictions(arrays,spec['counts']['validation'])
    if evaluate_probabilities(arrays['labels'],arrays['probabilities'])!=report['selected']['validation']:
        raise ValueError('Saved selected metrics differ')
    return record,arrays


def stress(train,val,bank,node,initial,classifier,mean,scale,device):
    model=models.new_model(node); model.load_state_dict(initial,strict=True); model.to(device)
    optimizer=training.optimizer_for(model); adapter=models.Adapter().to(device)
    rows=[]
    for cache in (train,val):
        lengths=np.concatenate([np.diff(b.offsets) for b in cache.blocks]); witness=int(np.argmax(lengths))
        ix=np.repeat(witness,128); raw=cache.batch(ix); model.train(cache.role=='train')
        with torch.set_grad_enabled(cache.role=='train'):
            logits,pred=models.forward(model,raw,device)
            if cache.role=='train':
                optimizer.zero_grad(set_to_none=True)
                loss,_,_=models.objective(logits,torch.from_numpy(raw['labels']).to(device),pred,
                                          torch.from_numpy(bank.targets[ix]).to(device))
                loss.backward(); meta_engine.finite_gradients(model.parameters()); optimizer.step()
            elif not torch.isfinite(logits).all(): raise ValueError('Nonfinite maximum support')
        rows.append(dict(role=cache.role,maximum=int(lengths.max()),batch=128,backward=cache.role=='train'))
    lengths=np.concatenate([np.diff(b.offsets) for b in train.blocks]); witness=int(np.argmax(lengths))
    support=[training.batch(train,bank,np.repeat(witness,16),device,target=True) for _ in range(5)]
    query=training.batch(train,bank,np.repeat(witness,64),device,target=False)
    state=deepcopy(model.state_dict()); modes=[mod.training for mod in model.modules()]
    model.zero_grad(set_to_none=True)
    loss=meta_engine.virtual_query(model,optimizer,adapter,support,query,[3e-5]*5,seed=819)
    grads=torch.autograd.grad(loss,tuple(adapter.parameters()))
    if (any(not torch.isfinite(g).all() for g in grads) or not any(torch.count_nonzero(g) for g in grads)
            or any(not torch.equal(v,model.state_dict()[k]) for k,v in state.items())
            or modes!=[mod.training for mod in model.modules()] or any(p.grad is not None for p in model.parameters())):
        raise ValueError('Five-step hypergradient or virtual-state isolation failed')
    witness_row=dict(steps=5,support_batch=16,query_batch=64,maximum=int(lengths.max()),
                     full_encoder=True,finite_nonzero_hypergradient=True,real_state_unchanged=True)
    del loss,grads,support,query
    torch.cuda.synchronize() if str(device).startswith('cuda') else None
    return rows,witness_row


def admission(spec,measured):
    if measured['environment']!=spec['anchor']['evidence']['environment']:
        raise ValueError('Original installed environment differs')
    if (any(type(measured[k]) is not int or measured[k]<=0 for k in ('peak_rss_bytes','gpu_peak_bytes'))
            or measured['peak_rss_bytes']>=.85*spec['memory_mb']*1024**2
            or measured['gpu_peak_bytes']>=.9*measured['gpu']['total_memory_bytes']):
        raise MemoryError('Measured resource envelope exceeded')
    times=measured['timings']
    expected={'authentication','load_HLT','fixed_train','fixed_validation','meta_train','meta_validation','readback'}
    if (set(times)!=expected or any(not math.isfinite(v) or v<0 for v in times.values())
            or any(times[a+'_'+phase]<=0 for a in ARMS for phase in ('train','validation'))):
        raise ValueError('Missing whole-population resource timings')
    result={}
    for arm in ARMS:
        result[arm]=max(30,math.ceil(1.75*(times['authentication']+times['load_HLT']+times['readback']+
            20*(times[arm+'_train']+times[arm+'_validation'])+300)/60))
        if result[arm]>spec['minutes'][arm]:
            raise ValueError('Measured continuation exceeds registered walltime: '+arm)
    return result


def setup(spec,folder,log,metadata,authentication,device):
    root=Path(spec['root']); teacher_ref=spec['teacher_refs']['checkpoint']
    manifest=load_json(validate_file_ref(spec['bank_manifest']))
    validate_bank(manifest,'BANK',banks.parents(spec['anchor']['evidence']['prepared'],spec['teacher_refs']))
    if manifest['content_hash']!=spec['reuse']['bank_hash']:
        raise ValueError('Must reuse the original U100-w5 target bank')
    with log.phase('reuse_exact_U100_TRAIN_bank_and_frozen_native_classifier'):
        validate_file_ref(teacher_ref)
        teacher=DelphesParticleTransformer().to(device).eval().requires_grad_(False)
        teacher.load_state_dict(torch.load(teacher_ref['path'],map_location='cpu',weights_only=True),strict=True)
        if len(teacher.mod.fc)!=1 or not isinstance(teacher.mod.fc[0],torch.nn.Linear):
            raise ValueError('Expected original Linear128-to11 native teacher classifier')
        payload=validate_file_ref(manifest['payload'],root=Path(spec['bank_manifest']['path']).parent)
        with np.load(payload,allow_pickle=False) as archive:
            statistics={k:archive[k] for k in ('mean','scale')}
        if any(v.dtype!=np.float64 or v.shape!=(128,) or not np.isfinite(v).all() for v in statistics.values()):
            raise ValueError('Original target normalization differs')
        (root/'initial').mkdir(exist_ok=False)
        save_state(root/'initial/classifier.pt',training.cpu_state(teacher.mod.fc[0].state_dict()))
        atomic_publish_bytes(root/'initial/statistics.npz',deterministic_npz_bytes(statistics))
        del teacher; gc.collect()
        if str(device).startswith('cuda'): torch.cuda.empty_cache()
    with log.phase('load_HLT_and_authenticate_initializer') as loaded:
        train,val,bank=load_inputs(spec,metadata); snapshot_references(spec,val,metadata)
        model=models.new_model(spec['node']); ref=spec['initial_checkpoint']; validate_file_ref(ref)
        models.initialize(model,torch.load(ref['path'],map_location='cpu',weights_only=True)); model.to(device).eval()
        saved=load_npz_arrays(root/'references/U100_EMBED_W5.npz')
        actual=predict(model,val,device=device,batch_size=128)
        np.testing.assert_allclose(actual,saved['probabilities'],rtol=1e-5,atol=1e-6)
        error=float(np.max(np.abs(actual-saved['probabilities'])))
    with log.phase('shared_predictor_calibration_native_frozen'):
        state,calibration=training.calibrate(model,train,bank,spec['node'],device)
        save_state(root/'initial/student.pt',state); validate_file_ref(ref)
        write_immutable_json(folder/'initializer.json',artifact('INITIALIZER',
            parents=dict(spec=spec['content_hash'],bank=manifest['content_hash'],initial_checkpoint=ref['sha256']),
            calibration=calibration,readback_rows=len(val),native_readback_max_error=error,
            outputs=[file_ref(root/n,root=root) for n in ('initial/student.pt','initial/classifier.pt','initial/statistics.npz')]))
        del model; gc.collect(); torch.cuda.empty_cache()
    with log.phase('installed_Weaver_native_and_five_step_maximum_support_checks'):
        with parity_backend(device): native=installed_parity(train,device=device)
        parities=[parity(train.batch(np.arange(4)),spec['node'],device,bf16=b) for b in (False,True)]
        initial,classifier,mean,scale=read_shared(spec)
        stresses,virtual=stress(train,val,bank,spec['node'],initial,classifier,mean,scale,device)
        maxima={role:metadata[0]['summaries'][role]['D000']['maximum'] for role in ('train','validation')}
        if any(r['maximum']!=maxima[r['role']] for r in stresses) or virtual['maximum']!=maxima['train']:
            raise ValueError('Maximum support differs from authenticated metadata')
        write_immutable_json(folder/'checks.json',artifact('CHECKS',parents=dict(spec=spec['content_hash']),native=native,parities=parities,
                                                        stress=stresses,virtual=virtual,maxima=maxima))
        gc.collect(); torch.cuda.empty_cache()
    with log.phase('whole_population_disposable_continuation_probes') as probe:
        probes={a:fit(folder/('probe_'+a),spec,train,val,bank,a,device,technical=True) for a in ARMS}
    times=dict(authentication=authentication,load_HLT=loaded['seconds'])
    for arm,r in probes.items():
        row=r['report']['history'][0]
        times[arm+'_train'],times[arm+'_validation']=row['train_seconds'],row['validation_seconds']
    if not probes['meta']['report']['history'][0]['meta_updates']:
        raise ValueError('Resource probe must execute real five-step adapter updates')
    times['readback']=max(.001,probe['seconds']-sum(v for k,v in times.items() if k.endswith(('_train','_validation'))))
    memory=memory_snapshot(torch)
    measured=dict(environment=installed_environment(),gpu=gpu_identity(),timings=times,
                  peak_rss_bytes=memory['peak_rss_bytes'],gpu_peak_bytes=memory['gpu_peak_bytes'])
    write_immutable_json(folder/'measurement.json',artifact('MEASUREMENT',parents=dict(spec=spec['content_hash']),
        **measured,projected_minutes=admission(spec,measured)))


def validate_task(spec,task,seen=None):
    if task not in TASKS: raise ValueError('Unknown task')
    seen={} if seen is None else seen
    if task in seen: return seen[task]
    root=Path(spec['root']); row=load_json(root/task/'receipt.json')
    deps={name:validate_task(spec,name,seen)['content_hash'] for name in TASKS[task]}
    validate(row,'TASK',dict(spec=spec['content_hash'],**deps))
    if row['task']!=task or [r['path'] for r in row['outputs']]!=output_names(task):
        raise ValueError('Task output inventory differs')
    for ref in row['outputs']: validate_file_ref(ref,root=root)
    if row['job_id']!=submission.accepted_jobs(spec,partial=task=='setup').get(task):
        raise ValueError('Task receipt differs from the exact accepted job')
    if task=='setup':
        manifest=load_json(validate_file_ref(spec['bank_manifest'])); validate_bank(manifest,'BANK')
        if manifest['content_hash']!=spec['reuse']['bank_hash']: raise ValueError('Target bank changed')
        measured=load_json(root/'setup/measurement.json'); validate(measured,'MEASUREMENT',dict(spec=spec['content_hash']))
        if measured['projected_minutes']!=admission(spec,measured): raise ValueError('Resource admission differs')
        checks=load_json(root/'setup/checks.json'); validate(checks,'CHECKS',dict(spec=spec['content_hash']))
        if (checks['maxima']!={r:spec['reuse']['summaries'][r]['D000']['maximum'] for r in ('train','validation')}
                or checks['native']['passed'] is not True or checks['virtual']!=dict(steps=5,support_batch=16,query_batch=64,
                maximum=checks['maxima']['train'],full_encoder=True,finite_nonzero_hypergradient=True,real_state_unchanged=True)
                or [p['bf16'] for p in checks['parities']]!=[False,True]
                or any(p[k] is not True for p in checks['parities'] for k in
                       ('native_forward','zero_control_update','auxiliary_backbone_gradient','auxiliary_head_gradient'))
                or checks['stress']!=[dict(role=r,maximum=checks['maxima'][r],batch=128,backward=r=='train') for r in ('train','validation')]):
            raise ValueError('Incomplete native/meta acceptance evidence')
        init=load_json(root/'setup/initializer.json')
        validate(init,'INITIALIZER',dict(spec=spec['content_hash'],bank=manifest['content_hash'],initial_checkpoint=spec['initial_checkpoint']['sha256']))
        if (init['readback_rows']!=spec['counts']['validation'] or not 0<=init['native_readback_max_error']<=1.1e-5
                or init['calibration']['native_unchanged'] is not True
                or len(init['calibration']['history'])!=3
                or init['calibration']['query_rows']!=spec['counts']['train']//20
                or init['calibration']['inner_rows']!=spec['counts']['train']-spec['counts']['train']//20):
            raise ValueError('Initializer evidence differs')
        for a in ARMS: validate_fit(root/'setup'/('probe_'+a),spec,a,technical=True)
    else: validate_fit(root/task/'fit',spec,task)
    seen[task]=row
    return row


def run(spec,task):
    if task not in TASKS: raise ValueError('Unknown task')
    begun=time.monotonic(); campaign.check(spec); authentication=time.monotonic()-begun
    jobs=submission.accepted_jobs(spec,partial=task=='setup'); deps={}; measured=None
    for dep in TASKS[task]:
        receipt=validate_task(spec,dep)
        if receipt['job_id']!=jobs[dep]: raise ValueError('Dependency job differs from journal')
        submission.successful(spec,dep,jobs[dep]); deps[dep]=receipt['content_hash']
    if task!='setup': measured=load_json(Path(spec['root'])/'setup/measurement.json')
    job=execution(spec,spec['reuse'] if measured is None else measured)
    if job!=jobs.get(task): raise ValueError('Worker job differs from exact accepted ID')
    if installed_environment()!=spec['anchor']['evidence']['environment']: raise ValueError('Original environment changed')
    campaign.storage(spec); root=Path(spec['root']); folder=root/task; folder.mkdir(exist_ok=False)
    log=PhaseLog(folder,lambda:memory_snapshot(torch)); torch.cuda.reset_peak_memory_stats()
    with log.phase('authenticate_prepared_metadata'): metadata=cache_metadata(spec)
    if task=='setup': setup(spec,folder,log,metadata,authentication,'cuda')
    else:
        with log.phase('load_authenticated_HLT_targets_and_shared_initializer'): train,val,bank=load_inputs(spec,metadata)
        with log.phase('matched_full20_continuation'): fit(folder/'fit',spec,train,val,bank,task,'cuda')
    campaign.storage(spec)
    if source.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']: raise ValueError('Source changed during execution')
    receipt=artifact('TASK',parents=dict(spec=spec['content_hash'],**deps),task=task,job_id=job,
                     outputs=[file_ref(root/n,root=root) for n in output_names(task)])
    write_immutable_json(folder/'receipt.json',receipt); validate_task(spec,task)
    return receipt


def results(spec):
    campaign.validate_spec(spec); root=Path(spec['root'])
    if not (root/'setup/receipt.json').is_file(): return dict(rows=[],pending=list(ARMS))
    validate_task(spec,'setup')
    arrays={n:load_npz_arrays(root/'references'/(n+'.npz')) for n in REFS}; pending=[]
    epochs={}
    for n in REFS:
        refs=spec['reuse']['reference'] if n=='U100_EMBED_W5' else spec['anchor']['evidence']['models'][n]
        epochs[n]=load_json(validate_file_ref(refs['report']))['selected_pass']
    for arm in ARMS:
        if not (root/arm/'receipt.json').is_file(): pending.append(arm); continue
        validate_task(spec,arm); fit_,arrays[ARMS[arm]]=validate_fit(root/arm/'fit',spec,arm)
        epochs[ARMS[arm]]=fit_['report']['selected_pass']
    ref=arrays['M0HLT']; rows=[]
    for name,a in arrays.items():
        validate_predictions(a,spec['counts']['validation'])
        if not np.array_equal(a['identities'],ref['identities']) or not np.array_equal(a['labels'],ref['labels']):
            raise ValueError('Comparison population differs')
        rows.append(dict(model=name,selected_epoch=epochs[name],input='ORACLE' if name=='OFFLINE' else 'single HLT',
            validation=evaluate_probabilities(a['labels'],a['probabilities'])))
    control=next((r for r in rows if r['model']==ARMS['fixed']),None)
    for row in rows:
        row['validation_recovery']=recovery(row['validation'],rows[0]['validation'],rows[1]['validation'])
        row['validation_delta_fixed']=None if control is None else row['validation']['macro_ovr_auc']-control['validation']['macro_ovr_auc']
    return artifact('RESULTS',parents=dict(spec=spec['content_hash']),rows=rows,pending=pending,
        meta_query_is_independent_holdout=False,primary_comparison='meta_minus_matched_fixed',
        historical_reference_epochs_are_not_continuation_epochs=True)

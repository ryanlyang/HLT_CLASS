"""Authenticated borrowed targets, inline acceptance, and guarded native fits."""
from pathlib import Path
import gc
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays, write_immutable_json,
    atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer, installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from hlt_classification.jetclass2_delphes.acceptance import installed_parity
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_fullsim import regular_cache
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from hlt_classification.luka_fullsim.science_campaign import preflight_accounting
from hlt_classification.luka_attention.worker import execution
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from hlt_classification.luka_embedding_aux.preflight import parity
from hlt_classification.luka_meta_embedding import worker as old_worker, model as models, meta
from hlt_classification.luka_meta_embedding.training import cpu_state, optimizer_for, batch
from .contracts import artifact, validate, TASKS, ARMS, INITIAL_LR
from . import campaign, source, submission, training

FIT_FILES = ('selected.pt','training_state.pt','training_report.json','validation_predictions.npz')
REFS = ('M0HLT','OFFLINE','OFFLINE_EMBED_W5')


def fit_parents(spec):
    return dict(spec=spec['content_hash'],initializer=spec['parent']['files']['state']['sha256'],
        bank=spec['parent']['bank_hash'],prepared=spec['anchor']['evidence']['prepared'])


def load_inputs(spec, metadata=None):
    parent = load_json(spec['parent']['spec']['path'])
    return old_worker.load_inputs(parent, metadata)


def read_shared(spec):
    files=spec['parent']['files']
    for name in ('state','native','classifier','bank'): validate_file_ref(files[name])
    full=torch.load(files['state']['path'],map_location='cpu',weights_only=True)
    mode=spec['initializer_source']
    if mode=='completed_fixed':
        initial=full['student']
        if full['epoch']!=spec['parent']['selected']['pass_number']:
            raise ValueError('Parent full state is not its selected FIXED epoch')
    elif mode=='shared_w5':
        initial=full  # Original setup persists a full state_dict, not a fit checkpoint.
    else:
        raise ValueError('Unknown initializer source; no fallback')
    student=models.new_model(spec['node']); student.load_state_dict(initial,strict=True)
    native=torch.load(files['native']['path'],map_location='cpu',weights_only=True)
    native_initial=student.native_state()
    if (native.keys()!=native_initial.keys()
            or any(not torch.equal(v,native_initial[k]) for k,v in native.items())
            or any(not torch.isfinite(v).all() for v in initial.values())):
        raise ValueError('Parent selected full/native state mismatch or nonfinite value')
    classifier=torch.nn.Linear(128,11)
    classifier.load_state_dict(torch.load(files['classifier']['path'],map_location='cpu',weights_only=True),strict=True)
    for p in classifier.parameters(): p.requires_grad_(False)
    arrays=load_npz_arrays(files['bank']['path'])
    return initial,classifier,torch.from_numpy(arrays['mean']).float(),torch.from_numpy(arrays['scale']).float()


def fit(folder, spec, train, val, bank, arm, device, *, technical=False):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=False)
    (folder/'snapshots').mkdir(); (folder/'history').mkdir()
    initial,classifier,mean,scale=read_shared(spec); model=models.new_model(spec['node'])
    def publish(kind, block, value):
        if kind=='selected':
            path=folder/'snapshots'/f'selected-{block:03d}.pt'; save_state(path,value)
        elif kind=='history':
            path=folder/'history'/f'block-{block:03d}.json'; write_immutable_json(path,value)
        else: raise ValueError('Unknown publication type')
        return file_ref(path,root=folder)
    report,native,full=training.train(model,train,val,bank,spec['node'],arm,device,
        initial=initial,classifier=classifier,mean=mean,scale=scale,parents=fit_parents(spec),
        publish=publish,technical=technical,initializer_source=spec['initializer_source'])
    training.validate_report(report,spec,fit_parents(spec),arm,technical=technical)
    save_state(folder/'selected.pt',native); save_state(folder/'training_state.pt',full)
    restored=DelphesParticleTransformer().to(device).eval()
    restored.load_state_dict(torch.load(folder/'selected.pt',map_location='cpu',weights_only=True),strict=True)
    expected=predict(model,val,device=device,batch_size=128)
    actual=predict(restored,val,device=device,batch_size=128)
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
    if evaluate_probabilities(val.labels,actual)!=report['selected']['validation']:
        raise ValueError('Native selected readback differs')
    saved=torch.load(folder/'training_state.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(saved['student'],strict=True)
    restored_native=model.native_state()
    if any(not torch.equal(v,restored_native[k]) for k,v in native.items()):
        raise ValueError('Selected full/native states differ')
    atomic_publish_bytes(folder/'validation_predictions.npz',deterministic_npz_bytes(
        dict(identities=val.identities,labels=val.labels,probabilities=actual)))
    record=artifact('FIT',parents=dict(report=report['content_hash']),report=report,
        native=file_ref(folder/'selected.pt',root=folder),state=file_ref(folder/'training_state.pt',root=folder),
        native_probability_readback=True)
    write_immutable_json(folder/'training_report.json',record)
    return record


def validate_fit(folder, spec, arm, *, technical=False):
    folder=Path(folder); record=load_json(folder/'training_report.json'); report=record['report']
    validate(record,'FIT',dict(report=report['content_hash']))
    training.validate_report(report,spec,fit_parents(spec),arm,technical=technical)
    if (record['native']!=file_ref(folder/'selected.pt',root=folder)
            or record['state']!=file_ref(folder/'training_state.pt',root=folder)
            or record['native_probability_readback'] is not True):
        raise ValueError('Native/full state inventory differs')
    for ref in report['snapshots']: validate_file_ref(ref,root=folder)
    for row in report['history']:
        saved=load_json(folder/'history'/f'block-{row["block"]:03d}.json')
        validate(saved,'BLOCK',fit_parents(spec))
        if saved['row']!=row or saved['arm']!=arm or saved['technical'] is not technical:
            raise ValueError('Durable block history differs')
    arrays=load_npz_arrays(folder/'validation_predictions.npz'); validate_predictions(arrays,spec['counts']['validation'])
    if evaluate_probabilities(arrays['labels'],arrays['probabilities'])!=report['selected']['validation']:
        raise ValueError('Saved selected metrics differ')
    return record,arrays


def fit_names(folder):
    report=load_json(Path(folder)/'training_report.json')['report']
    return list(FIT_FILES)+[r['path'] for r in report['snapshots']]+[
        f'history/block-{r["block"]:03d}.json' for r in report['history']]


def output_names(spec, task):
    root=Path(spec['root'])
    if task!='setup': return [task+'/fit/'+n for n in fit_names(root/task/'fit')]
    return ['setup/checks.json','setup/measurement.json']+[
        'setup/probe_'+a+'/'+n for a in ARMS for n in fit_names(root/'setup'/('probe_'+a))]


def admission(spec, measured):
    if measured['environment']!=spec['anchor']['evidence']['environment']:
        raise ValueError('Installed environment differs from original')
    if (any(type(measured[k]) is not int or measured[k]<=0 for k in ('peak_rss_bytes','gpu_peak_bytes'))
            or measured['peak_rss_bytes']>=.85*spec['memory_mb']*1024**2
            or measured['gpu_peak_bytes']>=.9*measured['gpu']['total_memory_bytes']):
        raise MemoryError('Measured resource envelope exceeded')
    result={}
    for arm in ARMS:
        t=measured['timings'][arm]
        required={'authentication','load','readback','initialization','headstart','later_phase','block'}
        if set(t)!=required or any(not math.isfinite(v) or v<0 for v in t.values()) or t['block']<=0:
            raise ValueError('Missing whole-block timing evidence')
        # Count every later phase at the measured expensive block-2 rate, including
        # frozen phases; allow a second guard readout worth of rollback/copy margin.
        seconds=t['authentication']+t['load']+t['readback']+t['initialization']+t['headstart']+19*t['later_phase']+20*t['block']+300
        result[arm]=max(30,math.ceil(1.75*seconds/60))
        if result[arm]>spec['minutes'][arm]: raise ValueError('Measured guarded fit exceeds debug walltime: '+arm)
    return result


def setup(spec, folder, log, metadata, authentication, device):
    with log.phase('load_borrowed_TRAIN_bank_and_'+spec['initializer_source']) as loaded:
        train,val,bank=load_inputs(spec,metadata); initial,classifier,mean,scale=read_shared(spec)
        model=models.new_model(spec['node']).to(device); model.load_state_dict(initial,strict=True)
        actual=predict(model,val,device=device,batch_size=128)
        saved=load_npz_arrays(spec['parent']['files']['predictions']['path'])
        if not np.array_equal(saved['identities'],val.identities) or not np.array_equal(saved['labels'],val.labels):
            raise ValueError('Original initializer validation identities differ')
        np.testing.assert_allclose(actual,saved['probabilities'],rtol=1e-5,atol=1e-6)
        error=float(np.max(np.abs(actual-saved['probabilities'])))
        if bank.manifest['content_hash']!=spec['parent']['bank_hash']: raise ValueError('Borrowed bank differs')
        parts=training.split(train.identities); split_hash=training.split_hash(train.identities,parts)
        del model; gc.collect(); torch.cuda.empty_cache()
    with log.phase('installed_native_and_maximum_support_acceptance'):
        with parity_backend(device): native=installed_parity(train,device=device)
        parities=[parity(train.batch(np.arange(4)),spec['node'],device,bf16=b) for b in (False,True)]
        stresses,virtual=old_worker.stress(train,val,bank,spec['node'],initial,classifier,mean,scale,device)
        # Additional real recipe LR witness; legacy maximum probe above is stricter
        # in step size, but cannot by itself establish a gradient at 5e-6.
        model=models.new_model(spec['node']).to(device); model.load_state_dict(initial,strict=True)
        optimizer=optimizer_for(model); adapter=models.Adapter().to(device)
        lengths=np.concatenate([np.diff(b.offsets) for b in train.blocks]); witness=int(np.argmax(lengths))
        supports=[batch(train,bank,np.repeat(witness,16),device,target=True) for _ in range(5)]
        q=batch(train,bank,np.repeat(witness,64),device,target=False)
        before=cpu_state(model.state_dict()); model.zero_grad(set_to_none=True)
        loss=meta.virtual_query(model,optimizer,adapter,supports,q,[INITIAL_LR]*5,seed=915)
        grads=torch.autograd.grad(loss,tuple(adapter.parameters()))
        gradient_ok=all(torch.isfinite(g).all() for g in grads) and any(torch.count_nonzero(g) for g in grads)
        if not gradient_ok or any(not torch.equal(v,model.state_dict()[k].cpu()) for k,v in before.items()) or optimizer.state:
            raise ValueError('Recipe-LR hypergradient or virtual isolation failed')
        maxima={r:metadata[1]['summaries'][r]['D000']['maximum'] for r in ('train','validation')}
        if any(r['maximum']!=maxima[r['role']] for r in stresses) or virtual['maximum']!=maxima['train']:
            raise ValueError('Maximum support differs from metadata')
        checks=artifact('CHECKS',native=native,parities=parities,stress=stresses,virtual=virtual,maxima=maxima,
            recipe_lr_hypergradient=True,split_sha256=split_hash,readback_rows=len(val),readback_max_error=error,
            initializer=spec['parent']['files']['state'],bank=spec['parent']['files']['bank'])
        write_immutable_json(folder/'checks.json',checks)
        del model,optimizer,adapter,supports,q,loss,grads; gc.collect(); torch.cuda.empty_cache()
    probes={}; timings={}
    for arm in ARMS:
        with log.phase('two_complete_disposable_blocks_'+arm) as timing:
            probes[arm]=fit(folder/('probe_'+arm),spec,train,val,bank,arm,device,technical=True)
        report=probes[arm]['report']; rows=report['history']
        consumed=report['initialization_seconds']+sum(sum(r[k] for k in
            ('adapter_seconds','train_seconds','guard_seconds','validation_seconds')) for r in rows)
        timings[arm]=dict(authentication=authentication,load=loaded['seconds'],
            readback=max(.001,timing['seconds']-consumed),initialization=report['initialization_seconds'],
            headstart=rows[0]['adapter_seconds'],later_phase=rows[1]['adapter_seconds'],
            block=max(r['train_seconds']+2*r['guard_seconds']+r['validation_seconds'] for r in rows))
        gc.collect(); torch.cuda.empty_cache()
    memory=memory_snapshot(torch)
    measured=dict(environment=installed_environment(),gpu=gpu_identity(),timings=timings,
        peak_rss_bytes=memory['peak_rss_bytes'],gpu_peak_bytes=memory['gpu_peak_bytes'])
    write_immutable_json(folder/'measurement.json',artifact('MEASUREMENT',**measured,projected_minutes=admission(spec,measured)))


def validate_task(spec, task, seen=None):
    if task not in TASKS: raise ValueError('Unknown task')
    seen={} if seen is None else seen
    if task in seen: return seen[task]
    root=Path(spec['root']); receipt=load_json(root/task/'receipt.json')
    deps={d:validate_task(spec,d,seen)['content_hash'] for d in TASKS[task]}
    validate(receipt,'TASK',dict(spec=spec['content_hash'],**deps))
    if receipt['task']!=task or [r['path'] for r in receipt['outputs']]!=output_names(spec,task):
        raise ValueError('Exact output inventory differs')
    for ref in receipt['outputs']: validate_file_ref(ref,root=root)
    if task=='setup':
        checks=load_json(root/'setup/checks.json'); validate(checks,'CHECKS')
        if (checks['native']['passed'] is not True or checks['recipe_lr_hypergradient'] is not True
                or checks['virtual']!=dict(steps=5,support_batch=16,query_batch=64,maximum=checks['maxima']['train'],
                    full_encoder=True,finite_nonzero_hypergradient=True,real_state_unchanged=True)
                or checks['stress']!=[dict(role=r,maximum=checks['maxima'][r],batch=128,backward=r=='train') for r in ('train','validation')]
                or [p['bf16'] for p in checks['parities']]!=[False,True]
                or any(p[k] is not True for p in checks['parities'] for k in
                    ('native_forward','zero_control_update','auxiliary_backbone_gradient','auxiliary_head_gradient'))
                or checks['readback_rows']!=spec['counts']['validation'] or not 0<=checks['readback_max_error']<=1.1e-5
                or checks['initializer']!=spec['parent']['files']['state'] or checks['bank']!=spec['parent']['files']['bank']):
            raise ValueError('Incomplete shared acceptance evidence')
        measured=load_json(root/'setup/measurement.json'); validate(measured,'MEASUREMENT')
        if measured['projected_minutes']!=admission(spec,measured): raise ValueError('Resource projection differs')
        for arm in ARMS:
            record,_=validate_fit(root/'setup'/('probe_'+arm),spec,arm,technical=True)
            if record['report']['split_sha256']!=checks['split_sha256']: raise ValueError('Probe TRAIN split differs')
    else:
        record,_=validate_fit(root/task/'fit',spec,task)
        if record['report']['split_sha256']!=load_json(root/'setup/checks.json')['split_sha256']:
            raise ValueError('Science TRAIN split differs')
    seen[task]=receipt
    return receipt


def run(spec, task):
    if task not in TASKS: raise ValueError('Unknown task')
    started=time.monotonic(); campaign.check(spec); authentication=time.monotonic()-started
    jobs=submission.accepted_jobs(spec,partial=task=='setup'); deps={}; measured=None
    for dep in TASKS[task]:
        row=validate_task(spec,dep)
        if row['job_id']!=jobs[dep]: raise ValueError('Dependency ID differs from journal')
        preflight_accounting(jobs[dep]); deps[dep]=row['content_hash']
    if task!='setup': measured=load_json(Path(spec['root'])/'setup/measurement.json')
    job=execution(spec,measured)
    if job!=jobs.get(task): raise ValueError('Current job ID differs from exact journal')
    if installed_environment()!=spec['anchor']['evidence']['environment']: raise ValueError('Original environment changed')
    campaign.storage(spec); root=Path(spec['root']); folder=root/task; folder.mkdir(exist_ok=False)
    log=PhaseLog(folder,lambda:memory_snapshot(torch)); torch.cuda.reset_peak_memory_stats()
    with log.phase('authenticate_prepared_metadata'): metadata=regular_cache.metadata(spec['anchor']['evidence']['prepared_root'])
    if task=='setup': setup(spec,folder,log,metadata,authentication,'cuda')
    else:
        with log.phase('load_borrowed_targets'): train,val,bank=load_inputs(spec,metadata)
        with log.phase('twenty_guarded_half_pass_blocks'): fit(folder/'fit',spec,train,val,bank,task,'cuda')
    campaign.storage(spec)
    if source.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']: raise ValueError('Source changed during execution')
    source.validate_parent(spec['parent'])
    receipt=artifact('TASK',parents=dict(spec=spec['content_hash'],**deps),task=task,job_id=job,
        outputs=[file_ref(root/n,root=root) for n in output_names(spec,task)])
    write_immutable_json(folder/'receipt.json',receipt); validate_task(spec,task)
    return receipt


def results(spec):
    campaign.validate_spec(spec); root=Path(spec['root']); files=spec['parent']['files']
    arrays={n:load_npz_arrays(files[n]['path']) for n in REFS}
    label='PARENT_FIXED' if spec['initializer_source']=='completed_fixed' else 'PARENT_SHARED_W5'
    arrays[label]=load_npz_arrays(files['predictions']['path'])
    pending=[]
    for arm in ARMS:
        if not (root/arm/'receipt.json').is_file(): pending.append(arm); continue
        validate_task(spec,arm); _,arrays[ARMS[arm]]=validate_fit(root/arm/'fit',spec,arm)
    ref=arrays['M0HLT']; ix=load_npz_arrays(files['original50k']['path'])['indices']
    if (ix.dtype!=np.int64 or ix.shape!=(spec['original50k_rows'],) or len(np.unique(ix))!=len(ix)
            or np.any(ix<0) or np.any(ix>=len(ref['labels']))): raise ValueError('Report-only subset differs')
    rows=[]
    for name,a in arrays.items():
        validate_predictions(a,spec['counts']['validation'])
        if not np.array_equal(a['identities'],ref['identities']) or not np.array_equal(a['labels'],ref['labels']):
            raise ValueError('Comparison populations differ')
        rows.append(dict(model=name,input='ORACLE' if name=='OFFLINE' else 'single HLT',
            validation=evaluate_probabilities(a['labels'],a['probabilities']),
            original50k=evaluate_probabilities(a['labels'][ix],a['probabilities'][ix])))
    for row in rows:
        for role in ('validation','original50k'):
            row[role+'_recovery']=recovery(row[role],rows[0][role],rows[1][role])
    return dict(rows=rows,pending=pending,independent_test_available=False)

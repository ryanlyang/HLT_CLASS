"""Exact-job workers and read-only completed-results reporting."""
from pathlib import Path
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json,load_npz_arrays,write_immutable_json,
    atomic_publish_bytes,deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer,installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_fullsim.preflight_v2 import require_backend
from hlt_classification.luka_u100_pair_hydra import portability
from . import execution as site
from .contracts import TASKS,TEACHERS,ARMS,settings,artifact,validate,teacher_task,architecture_control
from .model import new_model
from . import campaign,inputs,resources,training,submission


def setup(spec):
    root=Path(spec['root'])/'setup'; root.mkdir(exist_ok=False); started=time.perf_counter()
    b=inputs.Bundle(spec['bundle_root']); b.verify_all(); inputs.delta(spec).verify_all(); inputs.encoder_delta(spec).verify_all()
    train=inputs.load_view(spec,'train','D000'); val=inputs.load_view(spec,'validation','D000')
    oracle_train=inputs.oracle_cache(spec,train); oracle_val=inputs.oracle_cache(spec,val)
    bank=inputs.embedding_bank(spec,train)
    from hlt_classification.luka_u100_contrastive.training import peaks
    extraction_memory=peaks('cuda')
    if (extraction_memory['gpu_peak_bytes']>gpu_identity()['total_memory_bytes']*.85
            or (extraction_memory.get('peak_rss_bytes') or 0)>spec['memory_mb']*1024**2*.85):
        write_immutable_json(root/'feature_resource_failure.json',artifact('RESOURCE_FAILURE',memory=extraction_memory))
        raise MemoryError('Frozen-feature preparation exceeds reviewed resources')
    write_immutable_json(root/'portability.json',site.setup_record(spec,val))
    actual=torch.from_numpy(oracle_val.logits).softmax(-1).numpy()
    ref=load_npz_arrays(inputs.encoder_delta(spec).path('validation_predictions.npz'))
    if any(not np.array_equal(ref[k],getattr(val,k)) for k in ('identities','labels')): raise ValueError('U100 replay population differs')
    metrics=evaluate_probabilities(val.labels,actual); original=evaluate_probabilities(ref['labels'],ref['probabilities'])
    diff=abs(metrics['macro_ovr_auc']-original['macro_ovr_auc'])
    record=artifact('ENCODER_REPLAY',parents=dict(spec=spec['content_hash'],encoder=spec['encoder_hash']),validation=metrics,
        historical=original,auc_abs=diff,auc_tolerance=5e-5,checkpoint=file_ref(inputs.encoder_delta(spec).path('U100.pt')),
        train_cache_bytes=oracle_train.nbytes,validation_cache_bytes=oracle_val.nbytes,
        train_build_seconds=oracle_train.build_seconds,validation_build_seconds=oracle_val.build_seconds,
        extraction_memory=extraction_memory,
        probability_max_abs=float(np.abs(actual-ref['probabilities']).max()),probability_difference_report_only=True)
    write_immutable_json(root/'encoder_replay.json',record)
    atomic_publish_bytes(root/'encoder_predictions.npz',deterministic_npz_bytes(dict(identities=val.identities,labels=val.labels,probabilities=actual)))
    if diff>5e-5: raise ValueError('U100 numerical replay failed; inspect saved evidence')
    resources.measure(spec,train,val,oracle_train,oracle_val,bank,time.perf_counter()-started)


def receipt(spec,task):
    root=Path(spec['root'])/task; value=load_json(root/'receipt.json')
    validate(value,'TASK',dict(spec=spec['content_hash']))
    if (value['task']!=task or value['job_id']!=submission.accepted_jobs(spec,'setup' if task=='setup' else 'science').get(task)
            or len({r['path'] for r in value['outputs']})!=len(value['outputs'])):
        raise ValueError('Task exact-job/output ownership differs')
    for ref in value['outputs']: validate_file_ref(ref,root=root)
    return value


def completed_setup(spec):
    record=receipt(spec,'setup'); root=Path(spec['root'])/'setup'
    required={'portability.json','encoder_replay.json','encoder_predictions.npz','acceptance.json'} | {
        f'mini_{task}/receipt.json' for task in TASKS}
    if not required <= {r['path'] for r in record['outputs']}: raise ValueError('Shared setup incomplete')
    site.validate_setup(spec,load_json(root/'portability.json'))
    replay=load_json(root/'encoder_replay.json'); validate(replay,'ENCODER_REPLAY',dict(spec=spec['content_hash'],encoder=spec['encoder_hash']))
    if replay['auc_abs']>5e-5 or replay['auc_tolerance']!=5e-5: raise ValueError('Unqualified U100 replay')
    return resources.accepted(spec)


def load_teacher(spec,width,device='cuda'):
    task='teacher_m'+str(width); receipt(spec,task)
    folder=Path(spec['root'])/task/'fit'; report,_=training.validate_fit(folder,spec,task)
    model=training.deployment(settings(task),torch.load(folder/'selected.pt',map_location='cpu',weights_only=True),device)
    model.requires_grad_(False)
    scale=torch.tensor(report['statistics']['scale'],device=device)
    return model,scale,report['content_hash']


def run(spec,task):
    if task not in ('setup','report',*TASKS): raise ValueError('Unregistered task')
    campaign.check(spec); require_backend(); job=site.allocation(spec)
    stage='setup' if task=='setup' else 'science'
    for _ in range(12):
        if submission.accepted_jobs(spec,stage).get(task)==job: break
        time.sleep(5)
    else: raise PermissionError('Worker is not the accepted exact job')
    claim=Path(spec['root'])/'claims'/task; claim.mkdir(parents=True,exist_ok=False)
    write_immutable_json(claim/'intent.json',artifact('RUN_CLAIM',parents=dict(spec=spec['content_hash']),task=task,job_id=job))
    if task=='setup': setup(spec)
    else:
        acceptance=completed_setup(spec)
        if acceptance['environment']!=installed_environment() or acceptance['gpu']!=gpu_identity():
            raise PermissionError('Measured environment/GPU differs')
        folder=Path(spec['root'])/task; folder.mkdir(exist_ok=False)
        if task=='report':
            value=results(spec); write_immutable_json(folder/'results.json',value); print_results(value)
        else:
            setting=settings(task); isteacher=task in TEACHERS
            tr=inputs.load_view(spec,'train','D000'); va=inputs.load_view(spec,'validation','D000')
            ot=None; teacher=None; scale=None; parent=None
            bank=inputs.embedding_bank(spec,tr) if setting['embedding'] else None
            if isteacher or setting['mode'] in ('aux','both'):
                ot=inputs.oracle_cache(spec,tr)
            if isteacher:
                tr=ot; va=inputs.oracle_cache(spec,va); ot=None
            elif setting['mode'] in ('aux','both'):
                teacher,scale,parent=load_teacher(spec,setting['width'])
            model=new_model(spec['imported'],setting,None)
            training.fit(folder/'fit',spec,task,model,tr,va,device='cuda',bank=bank,teacher=teacher,
                oracle_train=ot,scale=scale,teacher_parent=parent)
    campaign.check(spec); root=Path(spec['root'])/task
    value=artifact('TASK',parents=dict(spec=spec['content_hash']),task=task,job_id=job,
        outputs=[file_ref(p,root=root) for p in sorted(root.rglob('*')) if p.is_file()])
    write_immutable_json(root/'receipt.json',value)
    return value


def results(spec):
    campaign.check(spec,check_storage=False); b=inputs.Bundle(spec['bundle_root'])
    arrays={name:load_npz_arrays(b.path(f'reference/{name}/predictions.npz')) for name in ('M0HLT','OFFLINE')}
    arrays['U100_W5_SPORC']=load_npz_arrays(b.path('reference/matched/predictions.npz'))
    if spec.get('gpu_profile')=='a6000':
        completed_setup(spec)
    arrays['U100_W5_HYDRA_REPLAY']=load_npz_arrays(site.control_predictions(spec))
    pending=[]; selected={}; parents={}; oracle_controls=set()
    for task in TASKS:
        folder=Path(spec['root'])/task
        if not (folder/'receipt.json').exists(): pending.append(task); continue
        receipt(spec,task); report,a=training.validate_fit(folder/'fit',spec,task)
        if report['parents']['prepared']!=spec['anchor']['evidence']['prepared']: raise ValueError('Result input lineage differs')
        if task in ARMS:
            cfg=ARMS[task]
            if cfg['mode']!='ce':
                parent_task=teacher_task(task); receipt(spec,parent_task)
                parent_report,_=training.validate_fit(Path(spec['root'])/parent_task/'fit',spec,parent_task)
                if report['parents'].get('teacher')!=parent_report['content_hash']: raise ValueError('Result teacher parent differs')
            if cfg['embedding'] and report['parents'].get('embedding')!=b.json('targets/u100/manifest.json')['content_hash']:
                raise ValueError('Result embedding parent differs')
        arrays[task]=a; selected[task]=report['selected_pass']; parents[task]=report['content_hash']
        if task in TEACHERS:
            name=task+'_message_off'
            arrays[name]=load_npz_arrays(folder/'fit/message_disabled_predictions.npz')
            selected[name]=report['selected_pass']; oracle_controls.add(name)
    subset=load_npz_arrays(b.path('reference/historical/subset.npz'))['indices']
    if (subset.dtype!=np.int64 or subset.shape!=(spec['original50k_rows'],) or len(np.unique(subset))!=len(subset)
            or np.any(subset<0) or np.any(subset>=spec['counts']['validation'])): raise ValueError('Invalid report subset')
    metrics={}
    from hlt_classification.luka_embedding_weight.worker import validate_predictions
    for name,a in arrays.items():
        validate_predictions(a,spec['counts']['validation'])
        if any(not np.array_equal(a[k],arrays['M0HLT'][k]) for k in ('identities','labels')): raise ValueError('Result rows differ')
        metrics[name]={role:evaluate_probabilities(a['labels'][ix],a['probabilities'][ix]) for role,ix in
            (('validation',slice(None)),('original50k',subset))}
    rows=[]
    for name,values in metrics.items():
        row=dict(model=name,selected_epoch=selected.get(name),privileged_oracle=name in TEACHERS or name in oracle_controls,
            diagnostic=name.endswith('_message_off'),
            site='Hydra' if name in TASKS or name in oracle_controls else 'SPORC weights / Hydra replay' if 'REPLAY' in name else 'SPORC',**values)
        for role,v in values.items():
            lo=metrics['M0HLT'][role]['macro_ovr_auc']; hi=metrics['OFFLINE'][role]['macro_ovr_auc']
            row[role+'_recovery']=100*(v['macro_ovr_auc']-lo)/(hi-lo) if hi!=lo else None
            row[role+'_delta_u100']=v['macro_ovr_auc']-metrics['U100_W5_HYDRA_REPLAY'][role]['macro_ovr_auc']
        control_name=architecture_control(name) if name in ARMS else None
        row['architecture_control']=control_name
        for role,v in values.items():
            row[role+'_delta_architecture_control']=(v['macro_ovr_auc']-metrics[control_name][role]['macro_ovr_auc']
                if control_name in metrics else None)
        rows.append(row)
    return artifact('RESULTS',parents=dict(spec=spec['content_hash'],**parents),rows=rows,pending=pending,
        recovery_anchors='historical_SPORC_M0HLT_OFFLINE',independent_test=False,
        reference_retrained=False,gpu_profile=spec.get('gpu_profile','l40'))


def print_results(value):
    print('Execution GPU profile: '+value.get('gpu_profile','l40')+'; U100 replay uses that GPU.')
    print('Cross-attention U100 message teachers; twelve fresh CE+U100-w5 students. Exploratory validation; matched seed pairs.')
    print('Teachers are privileged oracles. Historical SPORC recovery anchors; original50k overlaps validation.')
    print('teacher_*_message_off: same selected weights, message disabled; distribution-shift diagnostic, no reselection.')
    for role in ('validation','original50k'):
        print('\n'+role); print(f'{"model":26} {"ep":>3} {"AUC":>12} {"recovery%":>10} {"delta U100 replay":>18} {"delta arch ctrl":>16}')
        for row in value['rows']:
            r=row[role+'_recovery']; recovery='n/a' if r is None else f'{r:.1f}'
            diff=row[role+'_delta_architecture_control']; arch='--' if diff is None else f'{diff:+.8f}'
            print(f'{row["model"]:26} {str(row["selected_epoch"] or "-"):>3} {row[role]["macro_ovr_auc"]:12.8f} {recovery:>10} {row[role+"_delta_u100"]:+18.8f} {arch:>16}')
    print('Pending/incomplete: '+(', '.join(value['pending']) or 'none'))

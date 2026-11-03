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
from hlt_classification.luka_u100_pair_hydra import site,portability
from .contracts import TASKS,TEACHERS,ARMS,settings,artifact,validate
from .model import new_model
from . import campaign,inputs,resources,training,submission


def checkpoint(spec):
    return torch.load(inputs.delta(spec).path('M0HLT.pt'),map_location='cpu',weights_only=True)


def setup(spec):
    root=Path(spec['root'])/'setup'; root.mkdir(exist_ok=False); started=time.perf_counter()
    b=inputs.Bundle(spec['bundle_root']); b.verify_all(); inputs.delta(spec).verify_all()
    train=inputs.load_view(spec,'train','D000'); val=inputs.load_view(spec,'validation','D000')
    u=inputs.load_view(spec,'train','U100'); vu=inputs.oracle_validation(spec)
    oracle_train=inputs.OracleCache(train,u); oracle_val=inputs.OracleCache(val,vu)
    bank=inputs.embedding_bank(spec,train); state=checkpoint(spec)
    write_immutable_json(root/'portability.json',portability.setup_record(spec,val))
    native=DelphesParticleTransformer().cuda().eval(); native.load_state_dict(state,strict=True)
    actual=training.predict(native,val,'cuda'); ref=load_npz_arrays(b.path('reference/M0HLT/predictions.npz'))
    if any(not np.array_equal(ref[k],getattr(val,k)) for k in ('identities','labels')): raise ValueError('M0HLT replay population differs')
    metrics=evaluate_probabilities(val.labels,actual); original=evaluate_probabilities(ref['labels'],ref['probabilities'])
    diff=abs(metrics['macro_ovr_auc']-original['macro_ovr_auc'])
    record=artifact('M0_REPLAY',parents=dict(spec=spec['content_hash']),validation=metrics,
        historical=original,auc_abs=diff,auc_tolerance=5e-5,checkpoint=file_ref(inputs.delta(spec).path('M0HLT.pt')),
        probability_max_abs=float(np.abs(actual-ref['probabilities']).max()),probability_difference_report_only=True)
    write_immutable_json(root/'m0_replay.json',record)
    atomic_publish_bytes(root/'m0_predictions.npz',deterministic_npz_bytes(dict(identities=val.identities,labels=val.labels,probabilities=actual)))
    if diff>5e-5: raise ValueError('M0HLT numerical replay failed; inspect saved evidence')
    del native
    resources.measure(spec,train,val,oracle_train,oracle_val,bank,state,time.perf_counter()-started)


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
    required={'portability.json','m0_replay.json','m0_predictions.npz','acceptance.json'} | {
        f'mini_{task}/receipt.json' for task in TASKS}
    if not required <= {r['path'] for r in record['outputs']}: raise ValueError('Shared setup incomplete')
    portability.validate_setup(spec,load_json(root/'portability.json'))
    replay=load_json(root/'m0_replay.json'); validate(replay,'M0_REPLAY',dict(spec=spec['content_hash']))
    if replay['auc_abs']>5e-5 or replay['auc_tolerance']!=5e-5: raise ValueError('Unqualified M0HLT replay')
    return resources.accepted(spec)


def load_teacher(spec,site_name,device='cuda'):
    task='teacher_'+site_name; receipt(spec,task)
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
                ot=inputs.OracleCache(tr,inputs.load_view(spec,'train','U100'))
            if isteacher:
                tr=ot; va=inputs.OracleCache(va,inputs.oracle_validation(spec)); ot=None
            elif setting['mode'] in ('aux','both'):
                teacher,scale,parent=load_teacher(spec,setting['site'])
            model=new_model(spec['imported'],setting,checkpoint(spec) if isteacher else None)
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
    arrays['U100_W5_HYDRA_REPLAY']=load_npz_arrays(spec['portability_evidence']['refs']['predictions']['path'])
    pending=[]; selected={}; parents={}
    for task in TASKS:
        folder=Path(spec['root'])/task
        if not (folder/'receipt.json').exists(): pending.append(task); continue
        receipt(spec,task); report,a=training.validate_fit(folder/'fit',spec,task)
        if report['parents']['prepared']!=spec['anchor']['evidence']['prepared']: raise ValueError('Result input lineage differs')
        if task in ARMS:
            cfg=ARMS[task]
            if cfg['mode']!='ce':
                parent_task='teacher_'+cfg['site']; receipt(spec,parent_task)
                parent_report,_=training.validate_fit(Path(spec['root'])/parent_task/'fit',spec,parent_task)
                if report['parents'].get('teacher')!=parent_report['content_hash']: raise ValueError('Result teacher parent differs')
            if cfg['embedding'] and report['parents'].get('embedding')!=b.json('targets/u100/manifest.json')['content_hash']:
                raise ValueError('Result embedding parent differs')
        arrays[task]=a; selected[task]=report['selected_pass']; parents[task]=report['content_hash']
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
        row=dict(model=name,selected_epoch=selected.get(name),privileged_oracle=name in TEACHERS,
            site='Hydra' if name in TASKS else 'SPORC weights / Hydra replay' if 'REPLAY' in name else 'SPORC',**values)
        for role,v in values.items():
            lo=metrics['M0HLT'][role]['macro_ovr_auc']; hi=metrics['OFFLINE'][role]['macro_ovr_auc']
            row[role+'_recovery']=100*(v['macro_ovr_auc']-lo)/(hi-lo) if hi!=lo else None
            row[role+'_delta_u100']=v['macro_ovr_auc']-metrics['U100_W5_HYDRA_REPLAY'][role]['macro_ovr_auc']
        rows.append(row)
    return artifact('RESULTS',parents=dict(spec=spec['content_hash'],**parents),rows=rows,pending=pending,
        recovery_anchors='historical_SPORC_M0HLT_OFFLINE',independent_test=False)


def print_results(value):
    print('Frozen M0HLT privileged adapters; twelve fresh HLT students. Single-seed exploratory validation.')
    print('Teachers are privileged oracles. Historical SPORC recovery anchors; original50k overlaps validation.')
    for role in ('validation','original50k'):
        print('\n'+role); print(f'{"model":26} {"ep":>3} {"AUC":>12} {"recovery%":>10} {"delta U100 replay":>18}')
        for row in value['rows']:
            r=row[role+'_recovery']; recovery='n/a' if r is None else f'{r:.1f}'
            print(f'{row["model"]:26} {str(row["selected_epoch"] or "-"):>3} {row[role]["macro_ovr_auc"]:12.8f} {recovery:>10} {row[role+"_delta_u100"]:+18.8f}')
    print('Pending/incomplete: '+(', '.join(value['pending']) or 'none'))

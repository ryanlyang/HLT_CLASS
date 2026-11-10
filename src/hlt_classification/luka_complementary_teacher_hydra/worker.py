"""Exact-job workers; automatic release follows successful technical qualification."""
from pathlib import Path
import time
import torch
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.luka_fullsim.preflight_v2 import require_backend,memory_snapshot
from hlt_classification.luka_teacher_sweep_hydra import site
from hlt_classification.luka_embedding_aux import targets as bank_impl
from .contracts import TEACHERS,FOLDS,STUDENTS,FITS,artifact,validate
from . import campaign,submission,inputs,training,resources,folds


def receipt(spec,task):
    folder=Path(spec['root'])/task; m=load_json(folder/'receipt.json');validate(m,'TASK',dict(spec=spec['content_hash']))
    stage='bootstrap' if task in ('setup','release') else 'science'
    if (m['task']!=task or m['job_id']!=submission.accepted_jobs(spec,stage).get(task)
        or m['environment']!=spec['reference']['environment'] or len({r['path'] for r in m['outputs']})!=len(m['outputs'])):
        raise ValueError('Task/job/environment ownership differs')
    for r in m['outputs']: validate_file_ref(r,root=folder)
    names={r['path'] for r in m['outputs']}
    if task=='setup':
        required={'acceptance.json','resource_diagnostic.json','replay.json','replay_predictions.npz','folds.json','folds.npz','oof_miniature.json'}
        required|={f'mini_{t}/receipt.json' for t in FITS}
    elif task=='report': required={'results.json'}
    else:
        required={'fit/selected.pt','fit/selected_full.pt','fit/training_report.json','fit/receipt.json',
            'fit/training_replay.npz' if task in FOLDS else 'fit/validation_predictions.npz'}
        if task in TEACHERS: required|={'bank/manifest.json','bank/targets.npz'}
        if task in FOLDS: required|={'oof.json','oof.npz'}
        if task in ('TE','TF',*STUDENTS): required|={'startup/receipt.json','startup/training_report.json','startup_memory.json'}
        if task=='TF': required.add('difficulty.json')
    if not required<=names: raise ValueError('Incomplete task inventory')
    return m


def completed_setup(spec):
    receipt(spec,'setup');value=resources.accepted(spec)
    for t in FITS: training.validate_fit(Path(spec['root'])/'setup'/('mini_'+t),spec,t,technical=True)
    return value


def validate_teacher(spec,task):
    if task not in TEACHERS: raise ValueError('Unknown teacher')
    receipt(spec,task);folder=Path(spec['root'])/task; report,a=training.validate_fit(folder/'fit',spec,task)
    m=load_json(folder/'bank/manifest.json');ref=file_ref(folder/'fit/selected.pt')
    bank_impl.validate(m,'BANK',dict(recipe=bank_impl.recipe()['content_hash'],prepared=spec['anchor']['evidence']['prepared'],teacher=ref['sha256']))
    if (m['teacher']!=ref or m['role']!='train' or m['coordinate']!='OFFLINE' or m['width']!=128
        or m['rows']!=spec['counts']['train'] or m['payload']['path']!='targets.npz'): raise ValueError('Selected teacher bank differs')
    validate_file_ref(m['payload'],root=folder/'bank')
    if report['parents']!=fit_parents(spec,task): raise ValueError('Teacher fit parent lineage differs')
    return report,a


def fit_parents(spec,task):
    result=dict(spec=spec['content_hash'],recipe=spec['recipe']['content_hash'])
    if task=='TE':
        validate_teacher(spec,'T0');result['reference']=file_ref(Path(spec['root'])/'T0/fit/selected.pt')['sha256']
    if task=='TF': result['difficulty']=folds.assemble(spec)[1]['content_hash']
    if task in FOLDS: result['folds']=folds.load(spec)[0]['content_hash']
    for teacher in STUDENTS.get(task,()):
        validate_teacher(spec,teacher);result[teacher]=load_json(Path(spec['root'])/teacher/'bank/manifest.json')['content_hash']
    return result


def run(spec,task):
    if task not in ('setup','release','report',*FITS): raise ValueError('Unregistered task')
    campaign.check(spec)
    if task in ('release','report'): job=site.cpu_allocation(spec)
    else: require_backend();job=site.allocation(spec)
    stage='bootstrap' if task in ('setup','release') else 'science'
    for _ in range(12):
        if submission.accepted_jobs(spec,stage).get(task)==job: break
        time.sleep(5)
    else: raise PermissionError('Worker is not the exact accepted job')
    if installed_environment()!=spec['reference']['environment']: raise ValueError('Installed environment changed')
    if task=='release': return submission.finish(spec,job)
    claim=Path(spec['root'])/'claims'/task;claim.mkdir(parents=True,exist_ok=False)
    write_immutable_json(claim/'intent.json',artifact('RUN_CLAIM',parents=dict(spec=spec['content_hash']),task=task,job_id=job))
    folder=Path(spec['root'])/task;folder.mkdir(exist_ok=False)
    if task=='setup':
        inputs.Bundle(spec['bundle_root']).verify_all();inputs.delta(spec).verify_all();resources.measure(spec)
    else:
        acceptance=completed_setup(spec)
        if task!='report' and acceptance['gpu']!=gpu_identity(): raise PermissionError('GPU differs from measured allocation')
        if task=='report':
            from .reporting import results
            write_immutable_json(folder/'results.json',results(spec))
        else:
            full,val=inputs.fit_views(spec,task)
            train=folds.split(spec,full,task)[0] if task in FOLDS else full
            if task in FOLDS: val=None
            banks=inputs.banks(spec,task,train);context=inputs.context(spec,task,train,'cuda')
            if task=='TF': write_immutable_json(folder/'difficulty.json',context['manifest'])
            if task in ('TE','TF',*STUDENTS):
                torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
                resources.miniature(folder/'startup',spec,task,train,val,banks,context,'cuda');memory=memory_snapshot(torch)
                write_immutable_json(folder/'startup_memory.json',artifact('STARTUP',parents=training.parents(spec,task,banks,context),
                    memory=memory,seconds=time.perf_counter()-started,actual_completed_parent=True,technical_weights_discarded=True))
                if memory['peak_rss_bytes']>=.85*spec['memory_mb']*1024**2 or memory['gpu_peak_bytes']>=.88*acceptance['gpu']['total_memory_bytes']:
                    raise MemoryError('Actual-parent startup exceeds memory envelope')
                resources.free()
            native,report=training.fit(folder/'fit',spec,task,train,val,banks,device='cuda',context=context)
            if report['parents']!=fit_parents(spec,task): raise ValueError('Scientific fit parents differ')
            if task in TEACHERS:
                with torch.autocast('cuda',enabled=False): bank_impl.export(folder/'bank',native,train,teacher_ref=file_ref(folder/'fit/selected.pt'),device='cuda')
                bank_impl.Bank(folder/'bank',train,teacher_ref=file_ref(folder/'fit/selected.pt'))
            elif task in FOLDS: folds.export(spec,task,native,full,'cuda')
    campaign.check(spec)
    value=artifact('TASK',parents=dict(spec=spec['content_hash']),task=task,job_id=job,environment=installed_environment(),
        gpu=None if task=='report' else gpu_identity(),outputs=[file_ref(p,root=folder) for p in sorted(folder.rglob('*')) if p.is_file()])
    write_immutable_json(folder/'receipt.json',value)
    return value

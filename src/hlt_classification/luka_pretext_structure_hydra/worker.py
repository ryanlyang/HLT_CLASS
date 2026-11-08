"""Exact accepted-job dispatch and authenticated completion receipts."""
from pathlib import Path
import time
import torch
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.luka_fullsim.preflight_v2 import require_backend, memory_snapshot
from hlt_classification.luka_selective_teacher_hydra import site
from .contracts import FITS, TEACHERS, STUDENTS, artifact, validate
from . import campaign, submission, inputs, resources, teacher_training, student_training, embedding_banks


def receipt(spec,task):
    folder=Path(spec['root'])/task; value=load_json(folder/'receipt.json')
    validate(value,'TASK',dict(spec=spec['content_hash']))
    stage='bootstrap' if task=='setup' else 'science'
    if (value['task']!=task or value['job_id']!=submission.accepted_jobs(spec,stage).get(task)
            or value['environment']!=spec['reference']['environment']
            or len({r['path'] for r in value['outputs']})!=len(value['outputs'])):
        raise ValueError('Exact task ownership differs')
    for ref in value['outputs']: validate_file_ref(ref,root=folder)
    names={r['path'] for r in value['outputs']}
    if task=='setup':
        required={'acceptance.json','resource_diagnostic.json','preparation.json','statistics.json','replay.json','replay_predictions.npz','output_inventory.json'}
        required|={'technical_'+t+'/fit/receipt.json' for t in TEACHERS}|{'technical_'+t+'/bank/manifest.json' for t in TEACHERS}
        required|={'mini_'+t+'/receipt.json' for t in STUDENTS}
    elif task in TEACHERS:
        required={'fit/receipt.json','fit/selected.pt','fit/selected_full.pt','fit/training_report.json','bank/manifest.json','bank/targets.npy'}
    elif task in STUDENTS:
        required={'fit/receipt.json','fit/selected.pt','fit/selected_full.pt','fit/training_report.json',
                  'fit/validation_predictions.npz','fit/deployment.json','startup/receipt.json','startup_memory.json'}
    elif task=='report': required={'results.json'}
    else: raise ValueError('Unknown receipt task')
    if not required<=names or any(n.endswith('.partial') for n in names): raise ValueError('Incomplete task output inventory')
    return value


def completed_setup(spec):
    receipt(spec,'setup'); resources.check_replay(spec); inputs.preparation(spec)
    for task in TEACHERS:
        teacher_training.validate_fit(Path(spec['root'])/'setup'/('technical_'+task)/'fit',spec,task,technical=True)
    for task in STUDENTS:
        student_training.validate_fit(Path(spec['root'])/'setup'/('mini_'+task),spec,task,technical=True)
    return resources.accepted(spec)


def validate_teacher(spec,task):
    receipt(spec,task); root=Path(spec['root'])/task
    report=teacher_training.validate_fit(root/'fit',spec,task)
    prep,_=inputs.preparation(spec)
    if (report['parents']!=inputs.teacher_parents(spec) or report['counts']!=spec['counts']
            or report['maximum']!=prep['maximum']):
        raise ValueError('Teacher fit prepared population lineage differs')
    m=load_json(root/'bank/manifest.json'); validate(m,'EMBEDDING_BANK',dict(spec=spec['content_hash'],
        physical=prep['physicals']['train_OFFLINE'],teacher=file_ref(root/'fit/selected.pt')['sha256']))
    if (m['teacher']!=file_ref(root/'fit/selected.pt') or m['technical'] is not False or m['task']!=task
            or m['parents']['spec']!=spec['content_hash'] or m['role']!='train'
            or m['coordinate']!='PRETEXT_'+task or m['rows']!=spec['counts']['train'] or m['native_reload'] is not True):
        raise ValueError('Scientific teacher bank ownership differs')
    for ref in m['files']: validate_file_ref(ref,root=root/'bank')
    return report


def run(spec,task):
    if task not in ('setup','release','report',*FITS): raise ValueError('Unregistered task')
    campaign.check(spec); cpu=task in ('release','report')
    if cpu: job=site.cpu_allocation(spec)
    else: require_backend(); job=site.allocation(spec)
    stage='bootstrap' if task in ('setup','release') else 'science'
    for _ in range(12):
        if submission.accepted_jobs(spec,stage).get(task)==job: break
        time.sleep(5)
    else: raise PermissionError('Not the exact accepted job')
    if installed_environment()!=spec['reference']['environment']: raise ValueError('Installed environment changed')
    if task=='release': return submission.finish(spec,job)
    claim=Path(spec['root'])/'claims'/task; claim.mkdir(parents=True,exist_ok=False)
    write_immutable_json(claim/'intent.json',artifact('RUN_CLAIM',parents=dict(spec=spec['content_hash']),task=task,job_id=job))
    folder=Path(spec['root'])/task; folder.mkdir(exist_ok=False)
    if task=='setup':
        inputs.Bundle(spec['bundle_root']).verify_all(); resources.measure(spec)
    elif task=='report':
        completed_setup(spec)
        from .reporting import results
        write_immutable_json(folder/'results.json',results(spec))
    else:
        acceptance=completed_setup(spec)
        if acceptance['gpu']!=gpu_identity(): raise PermissionError('GPU differs from measured setup')
        train,val=inputs.views(spec)
        if task in TEACHERS:
            pt,pv=inputs.physical(spec,train,'OFFLINE'),inputs.physical(spec,val,'OFFLINE')
            st,sv=inputs.structure(spec,train,'OFFLINE'),inputs.structure(spec,val,'OFFLINE')
            prep,stats=inputs.preparation(spec)
            model,report=teacher_training.fit(folder/'fit',spec,task,pt,pv,st,sv,stats,prep['maximum'],device='cuda')
            embedding_banks.export(folder/'bank',model,pt,spec,task,file_ref(folder/'fit/selected.pt'),device='cuda')
        else:
            banks=inputs.banks(spec,task,train); torch.cuda.reset_peak_memory_stats()
            resources.miniature(folder/'startup',spec,task,train,val,banks,'cuda')
            memory=memory_snapshot(torch)
            write_immutable_json(folder/'startup_memory.json',artifact('STARTUP',parents=inputs.fit_parents(spec,task),
                memory=memory,real_targets=True,technical_weights_discarded=True))
            if memory['peak_rss_bytes']>=.85*spec['memory_mb']*2**20 or memory['gpu_peak_bytes']>=.88*acceptance['gpu']['total_memory_bytes']:
                raise MemoryError('Actual selected-target startup exceeds memory envelope')
            resources.free(); _,report=student_training.fit(folder/'fit',spec,task,train,val,banks,device='cuda')
            if report['parents']!=inputs.fit_parents(spec,task): raise ValueError('Scientific target lineage differs')
    campaign.check(spec)
    value=artifact('TASK',parents=dict(spec=spec['content_hash']),task=task,job_id=job,
        environment=installed_environment(),gpu=None if cpu else gpu_identity(),
        outputs=[file_ref(p,root=folder) for p in sorted(folder.rglob('*')) if p.is_file()])
    write_immutable_json(folder/'receipt.json',value)
    return value

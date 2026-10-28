"""Production preparation/fit/report with native-only readback and exact lineage."""
from pathlib import Path
import os
import platform
import re
import subprocess
import sys
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json,load_npz_arrays,write_immutable_json,atomic_publish_bytes,deterministic_npz_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities,recovery
from hlt_classification.luka_attention.worker import execution as gpu_execution
from hlt_classification.luka_embedding_weight.worker import storage,validate_predictions
from hlt_classification.luka_trial_hybrid.worker import native,release
from hlt_classification.luka_trial_hybrid import banks as compact
from hlt_classification.luka_fullsim import regular_cache
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog,memory_snapshot
from hlt_classification.luka_supervision_depth import data
from .contracts import artifact,validate,ARMS,TEACHERS,REFERENCES,recipe,FAMILIES
from . import campaign,source,submission,banks,training,startup

FIT_FILES=('selected.pt','training_report.json','validation_predictions.npz','receipt.json')

def execution(spec, task):
    job = os.environ.get('SLURM_JOB_ID', ''); site = spec['execution_site']
    if not re.fullmatch('[1-9][0-9]*', job):
        raise PermissionError('A real Slurm allocation is required')
    details = subprocess.run(['scontrol', 'show', 'job', '-o', job], check=True, capture_output=True, text=True).stdout
    fields = dict(x.split('=', 1) for x in details.split() if '=' in x)
    if (fields.get('JobName') != 'lukanb_'+task or fields.get('Comment') != 'lukanb:'+spec['content_hash']+':'+task
            or fields.get('WorkDir') != spec['project_dir'] or fields.get('TimeLimit') != _walltime(spec['task_minutes'][task])):
        raise PermissionError('Exact task identity differs')
    if task not in ('report',):
        return gpu_execution(spec, spec['reuse']['acceptance']['preflight'])
    prefix = site['conda_base']+'/envs/'+site['conda_env']
    if (fields.get('Account') != site['account'] or fields.get('Partition') != site['partition'] or fields.get('QOS') != site['qos']
            or fields.get('NumNodes') != '1' or fields.get('NumTasks') != '1' or fields.get('NumCPUs') != str(spec['report_cpus'])
            or os.environ.get('SLURM_CLUSTER_NAME') != site['cluster'] or platform.machine() != site['architecture']
            or os.environ.get('SLURM_CPUS_PER_TASK') != str(spec['report_cpus'])
            or os.environ.get('SLURM_MEM_PER_NODE') != str(spec['report_memory_mb'])
            or sys.prefix != prefix or os.environ.get('CONDA_PREFIX') != prefix or os.environ.get('PYTHONNOUSERSITE') != '1'):
        raise PermissionError('Report allocation/environment differs')
    return job

def snapshot_references(spec,val):
    folder=Path(spec['root'])/'targets/references'; folder.mkdir(exist_ok=False)
    for name,ref in spec['imported']['references'].items():
        validate_file_ref(ref); a=load_npz_arrays(ref['path']); validate_predictions(a,len(val))
        if not np.array_equal(a['identities'],val.identities) or not np.array_equal(a['labels'],val.labels): raise ValueError('Reference population differs')
        raw=Path(ref['path']).read_bytes(); validate_file_ref(ref); atomic_publish_bytes(folder/(name+'.npz'),raw)
    ref=spec['imported']['subset']; validate_file_ref(ref)
    raw=Path(ref['path']).read_bytes(); validate_file_ref(ref); atomic_publish_bytes(folder/'original50k.npz',raw)
    ref=spec['imported']['baseline_report']; validate_file_ref(ref)
    raw=Path(ref['path']).read_bytes(); validate_file_ref(ref); atomic_publish_bytes(folder.parent/'baseline_report.json',raw)


def comparison_anchors(spec):
    folder=Path(spec['root'])/'targets/references'; result={}
    for key,name in (('hlt','M0HLT'),('offline','OFFLINE'),('baseline','BASELINE_REPEAT')):
        a=load_npz_arrays(folder/(name+'.npz'))
        result[key]=evaluate_probabilities(a['labels'],a['probabilities'])['macro_ovr_auc']
    return result


def fit(spec,task,tr,val,bank,device):
    folder=Path(spec['root'])/task/'fit'; folder.mkdir(exist_ok=False)
    model,report,state=training.train(tr,val,node=spec['nodes'][task],bank=bank,device=device,comparison=comparison_anchors(spec))
    training.validate_report(report,spec,task,bank.manifest['content_hash'])
    expected=predict(model,val,device=device,batch_size=128); save_state(folder/'selected.pt',state)
    saved=torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)
    if saved.keys()!=state.keys() or any(not torch.equal(saved[k],state[k]) for k in state): raise ValueError('Native state roundtrip differs')
    del model; release(); restored=native(file_ref(folder/'selected.pt'),device)
    actual=predict(restored,val,device=device,batch_size=128)
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
    if evaluate_probabilities(val.labels,actual)!=report['validation']: raise ValueError('Native metrics readback differs')
    atomic_publish_bytes(folder/'validation_predictions.npz',deterministic_npz_bytes(dict(identities=val.identities,labels=val.labels,probabilities=actual)))
    write_immutable_json(folder/'training_report.json',report)
    rec=artifact('FIT',parents=dict(spec=spec['content_hash'],bundle=bank.manifest['content_hash']),task=task,
        checkpoint_probability_roundtrip=True,outputs=[file_ref(folder/f,root=folder) for f in FIT_FILES[:-1]])
    write_immutable_json(folder/'receipt.json',rec)
    del restored; release()


def validate_fit(spec,task):
    folder=Path(spec['root'])/task; bundle=load_json(folder/'bundle.json'); validate(bundle,'BUNDLE')
    validate_bundle(spec,task,bundle)
    row=load_json(folder/'fit/receipt.json'); validate(row,'FIT',dict(spec=spec['content_hash'],bundle=bundle['content_hash']))
    if row['task']!=task or row['checkpoint_probability_roundtrip'] is not True or [r['path'] for r in row['outputs']]!=list(FIT_FILES[:-1]):
        raise ValueError('Fit publication differs')
    for ref in row['outputs']: validate_file_ref(ref,root=folder/'fit')
    report=load_json(folder/'fit/training_report.json'); training.validate_report(report,spec,task,bundle['content_hash'])
    if report['comparison']!=comparison_anchors(spec): raise ValueError('Fixed comparison references differ')
    arrays=load_npz_arrays(folder/'fit/validation_predictions.npz'); validate_predictions(arrays,spec['counts']['validation'])
    if evaluate_probabilities(arrays['labels'],arrays['probabilities'])!=report['validation']: raise ValueError('Saved metrics differ')
    startup.check(load_json(folder/'startup.json'),spec,task,bundle['content_hash'])
    return report,arrays


def validate_bundle(spec,task,value):
    root=Path(spec['root']); m=load_json(root/'targets/neighborhoods/manifest.json'); records={}
    for t in TEACHERS:
        ref=spec['imported']['banks'][t]; validate_file_ref(ref)
        records[t]=compact.manifest(Path(ref['path']).parent,spec['reuse']['teachers'][t],spec['anchor']['evidence']['prepared'],spec['counts']['train'],payload=False)
    expected=artifact('BUNDLE',parents=dict(spec=spec['content_hash'],recipe=recipe(task)['content_hash'],
        neighborhoods=m['content_hash'],**{t:r['content_hash'] for t,r in records.items()}),arm=task,banks=records,neighborhoods=m)
    if value!=expected: raise ValueError('Neighborhood/teacher bundle differs')


def output_names(task):
    if task=='targets':
        return (['targets/references/'+n+'.npz' for n in (*REFERENCES,'original50k')]
            + ['targets/baseline_report.json','targets/bundle.json','targets/startup.json','targets/costs.json']
            + ['targets/neighborhoods/'+n for n in ('manifest.json','index.npz',*(f+'.npy' for f in FAMILIES))])
    if task=='report': return ['report/results.json','report/costs.json']
    if task not in ARMS: raise ValueError('Unknown task')
    return [task+'/fit/'+f for f in FIT_FILES]+[task+'/'+f for f in ('bundle.json','startup.json','costs.json')]


def validate_task(spec,task,seen=None,retired=None):
    seen={} if seen is None else seen
    if task in seen: return seen[task]
    root=Path(spec['root']); row=load_json(root/task/'receipt.json')
    deps=['targets'] if task in ARMS else []
    parents=dict(spec=spec['content_hash'],**{d:validate_task(spec,d,seen,retired)['content_hash'] for d in deps})
    validate(row,'TASK',parents)
    if (row['task']!=task or row['job_id']!=submission.accepted_jobs(spec).get(task)
            or [r['path'] for r in row['outputs']]!=output_names(task)): raise ValueError('Task/journal/inventory differs')
    removed={} if retired is None else {r['path']:r for r in retired['payloads']}
    for ref in row['outputs']:
        if ref['path'] in removed:
            if ref!=removed[ref['path']] or (root/ref['path']).exists() or (root/ref['path']).is_symlink(): raise ValueError('Retirement mismatch')
        else: validate_file_ref(ref,root=root)
    costs=load_json(root/task/'costs.json'); validate(costs,'COSTS',dict(spec=spec['content_hash']))
    if (costs['job_id']!=row['job_id'] or costs['task']!=task or costs['environment']!=spec['anchor']['evidence']['environment']
            or costs['separate_preflight_performed'] is not False): raise ValueError('Runtime evidence differs')
    if task in ARMS: validate_fit(spec,task)
    elif task=='targets':
        m=load_json(root/'targets/neighborhoods/manifest.json'); index=load_npz_arrays(root/'targets/neighborhoods/index.npz')
        # Retained index can validate metadata after explicitly authorized retirement.
        from types import SimpleNamespace
        class Retained:
            role='train'; coordinate_name='D000'; foundation_sha256=spec['anchor']['evidence']['prepared']
            identities=index['identities']; labels=index['labels']; blocks=[SimpleNamespace(offsets=index['offsets'])]
            def __len__(self): return spec['counts']['train']
        banks.validate_index(m,index,Retained(),spec['source_snapshot']['content_hash'])
        for ref in m['outputs']:
            path='targets/neighborhoods/'+ref['path']
            task_ref=next(r for r in row['outputs'] if r['path']==path)
            if any(task_ref[k]!=ref[k] for k in ('sha256','bytes')): raise ValueError('Manifest payload binding differs')
        bundle=load_json(root/'targets/bundle.json'); validate_bundle(spec,'smooth_w3',bundle)
        proof=load_json(root/'targets/startup.json'); startup.check(proof,spec,'smooth_w3',bundle['content_hash'])
        if proof['maximum_train']!=int(np.diff(index['offsets']).max()): raise ValueError('Startup TRAIN support differs')
    seen[task]=row
    return row


def results(spec):
    from .retention import retired_evidence
    campaign.validate_spec(spec,imports=False); root=Path(spec['root']); seen={}; retired=retired_evidence(spec)
    if not (root/'targets/receipt.json').is_file(): raise ValueError('Shared preparation has no accepted receipt')
    row=validate_task(spec,'targets',seen,retired); submission.successful(spec,'targets',row['job_id'])
    refs=root/'targets/references'; arrays={n:load_npz_arrays(refs/(n+'.npz')) for n in REFERENCES}
    anchor=arrays['M0HLT']; ix=load_npz_arrays(refs/'original50k.npz')['indices']
    if (ix.dtype!=np.int64 or ix.shape!=(spec['original50k_rows'],) or len(np.unique(ix))!=len(ix)
            or np.any(ix<0) or np.any(ix>=len(anchor['labels']))): raise ValueError('Invalid overlapping subset')
    picks={}; pending=[]
    for task in ARMS:
        if not (root/task/'receipt.json').is_file(): pending.append(task); continue
        row=validate_task(spec,task,seen,retired); submission.successful(spec,task,row['job_id'])
        report,arrays[task]=validate_fit(spec,task); picks[task]=report['selected_pass']
    rows=[]
    for name,a in arrays.items():
        validate_predictions(a,spec['counts']['validation'])
        if not np.array_equal(a['identities'],anchor['identities']) or not np.array_equal(a['labels'],anchor['labels']): raise ValueError('Comparison population differs')
        rows.append(dict(model=name,selected_epoch=picks.get(name),input='ORACLE' if name=='OFFLINE' else 'single HLT',
            validation=evaluate_probabilities(a['labels'],a['probabilities']),original50k=evaluate_probabilities(a['labels'][ix],a['probabilities'][ix])))
    by={r['model']:r for r in rows}
    for row in rows:
        for role in ('validation','original50k'):
            row[role+'_recovery']=recovery(row[role],by['M0HLT'][role],by['OFFLINE'][role])
            row[role+'_delta_baseline']=row[role]['macro_ovr_auc']-by['BASELINE_REPEAT'][role]['macro_ovr_auc']
    return artifact('RESULTS',parents=dict(spec=spec['content_hash'],**{t:r['content_hash'] for t,r in seen.items()}),
        rows=rows,pending=pending,student_seeds=1,independent_test=False,original50k_selection=False)


def run(spec,task,*,device='cuda'):
    started=time.perf_counter()
    if task not in {r['task_id'] for r in spec['tasks']}: raise ValueError('Unknown task')
    campaign.check(spec); job=execution(spec,task)
    if submission.accepted_jobs(spec).get(task)!=job: raise ValueError('Allocation is not accepted')
    from .retention import retired_evidence
    if retired_evidence(spec) is not None: raise ValueError('Campaign retired; no new consumers')
    parents=dict(spec=spec['content_hash'])
    if task in ARMS:
        rec=validate_task(spec,'targets'); submission.successful(spec,'targets',rec['job_id']); parents['targets']=rec['content_hash']
    if installed_environment()!=spec['anchor']['evidence']['environment']: raise ValueError('Environment differs')
    used=storage(spec); root=Path(spec['root']); folder=root/task; folder.mkdir(exist_ok=False)
    if task=='report':
        value=results(spec); write_immutable_json(folder/'results.json',value); print_results(value)
    else:
        if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats()
        log=PhaseLog(folder,lambda:memory_snapshot(torch)); meta=regular_cache.metadata(spec['anchor']['evidence']['prepared_root'])
        with log.phase('authenticated_HLT_population'):
            tr,val=(data.load(spec,role,'D000',meta) for role in ('train','validation'))
        if task=='targets':
            with log.phase('TRAIN_neighborhoods_and_reference_snapshots'):
                banks.build(folder/'neighborhoods',tr,banks.physical_rows(spec,tr,meta),
                    source_hash=spec['source_snapshot']['content_hash'],available_bytes=spec['storage_budget_bytes']-used)
                snapshot_references(spec,val)
        neighborhood=banks.NeighborhoodBank(root/'targets/neighborhoods',tr,spec['source_snapshot']['content_hash'])
        try:
            arm='smooth_w3' if task=='targets' else task
            with log.phase('disposable_native_max_support_check'):
                bank=banks.Bundle(spec,tr,arm,neighborhood)
                write_immutable_json(folder/'bundle.json',bank.manifest)
                proof=startup.run(spec,tr,val,bank,spec['nodes'][arm],device,
                                  elapsed_before_seconds=time.perf_counter()-started)
                write_immutable_json(folder/'startup.json',proof); release()
            if task!='targets':
                with log.phase('fresh_fixed30_fit'): fit(spec,task,tr,val,bank,device)
            del tr,val,bank; release()
        finally: neighborhood.close()
    costs=artifact('COSTS',parents=dict(spec=spec['content_hash']),task=task,job_id=job,environment=installed_environment(),
        memory=memory_snapshot(torch),separate_preflight_performed=False)
    write_immutable_json(folder/'costs.json',costs); storage(spec)
    if source.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']: raise ValueError('Source changed during worker')
    row=artifact('TASK',parents=parents,task=task,job_id=job,outputs=[file_ref(root/n,root=root) for n in output_names(task)])
    write_immutable_json(folder/'receipt.json',row); validate_task(spec,task)
    return row


def print_results(value):
    print('Fixed native HLT CE0.5/K1/E40 plus TRAIN-only offline-neighborhood prediction. No independent test.')
    print('Original50k overlaps validation and is report-only. Auxiliary heads removed for inference.')
    for role in ('validation','original50k'):
        print('\n'+role); print(f'{"model":24} {"pick":>5} {"AUC":>11} {"recovery%":>10} {"delta repeat":>13}')
        for row in value['rows']:
            rec=row[role+'_recovery']['macro_ovr_auc']; rec='n/a' if rec is None else f'{rec:.1f}'
            print(f'{row["model"]:24} {str(row["selected_epoch"] or "-"):>5} {row[role]["macro_ovr_auc"]:11.8f} {rec:>10} {row[role+"_delta_baseline"]:+13.8f}')
    print('Pending/incomplete: '+(', '.join(value['pending']) or 'none'))

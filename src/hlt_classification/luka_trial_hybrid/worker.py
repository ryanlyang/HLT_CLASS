"""Production export/fits/report; native-only inference and bounded owned retirement."""
from pathlib import Path
import gc
import os
import platform
import re
import subprocess
import sys
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json,load_npz_arrays,write_immutable_json,
    atomic_publish_bytes,deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer,installed_environment
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities,recovery
from hlt_classification.luka_attention.worker import execution as gpu_execution
from hlt_classification.luka_embedding_prediction.data import load_views
from hlt_classification.luka_embedding_aux.targets import offline_train
from hlt_classification.luka_embedding_weight.worker import storage,validate_predictions
from hlt_classification.luka_joint_prediction.worker import snapshot_references
from hlt_classification.luka_fullsim import regular_cache
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog,memory_snapshot
from .contracts import artifact,validate,ARMS,STAGES,task_id,split_task,export_kinds
from . import campaign,source,submission,banks,training

REFERENCES=('M0HLT','OFFLINE','COARSE_U100','U100_EMBED_W5','OFFLINE_EMBED_W5','TOKEN_ALIGNED','TOKEN_SET')
FIT_FILES=('selected.pt','training_report.json','validation_predictions.npz','receipt.json')


def data(spec,role,coordinate,meta):
    if coordinate!='OFFLINE':
        return load_views(spec,role,[coordinate],authenticated=meta)[coordinate]
    if role!='train': raise PermissionError('OFFLINE only supplies TRAIN targets')
    root=Path(spec['anchor']['evidence']['prepared_root'])
    present=[(root/r['path']).exists() or (root/r['path']).is_symlink() for r in meta[0]['blocks'][role][coordinate]]
    if all(present):
        return regular_cache.load(root,role=role,coordinate=coordinate,max_ram_bytes=int(.25*spec['memory_mb']*1024**2),authenticated=meta)
    if any(present): raise ValueError('Partial OFFLINE cache: preserve and investigate')
    return offline_train(spec,authenticated=meta)


def execution(spec,task):
    job=os.environ.get('SLURM_JOB_ID',''); site=spec['execution_site']
    if not re.fullmatch('[1-9][0-9]*',job): raise PermissionError('A real allocation is required')
    details=subprocess.run(['scontrol','show','job','-o',job],check=True,capture_output=True,text=True).stdout
    f=dict(x.split('=',1) for x in details.split() if '=' in x)
    if (f.get('JobName')!='lukahy_'+task or f.get('Comment')!='lukahy:'+spec['content_hash']+':'+task
            or f.get('WorkDir')!=spec['project_dir']): raise PermissionError('Exact task identity differs')
    if task!='report': return gpu_execution(spec,spec['acceptance']['preflight'])
    prefix=site['conda_base']+'/envs/'+site['conda_env']
    if (f.get('Account')!=site['account'] or f.get('Partition')!=site['partition'] or f.get('QOS')!=site['qos']
            or f.get('NumNodes')!='1' or f.get('NumTasks')!='1' or f.get('NumCPUs')!=str(spec['report_cpus'])
            or os.environ.get('SLURM_CLUSTER_NAME')!=site['cluster']
            or os.environ.get('SLURM_CPUS_PER_TASK')!=str(spec['report_cpus'])
            or os.environ.get('SLURM_MEM_PER_NODE')!=str(spec['report_memory_mb'])
            or platform.machine()!=site['architecture'] or sys.prefix!=prefix
            or os.environ.get('CONDA_PREFIX')!=prefix or os.environ.get('PYTHONNOUSERSITE')!='1'):
        raise PermissionError('Report allocation/environment differs')
    return job


def native(ref,device):
    validate_file_ref(ref); model=DelphesParticleTransformer().to(device).eval()
    model.load_state_dict(torch.load(ref['path'],map_location='cpu',weights_only=True),strict=True)
    validate_file_ref(ref); return model


def release():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def output_names(task):
    if task=='report': raise ValueError('Report uses completion/retirement records')
    if task=='targets':
        return (['targets/references/'+n+'.npz' for n in (*REFERENCES,'original50k')]
            +['targets/banks/'+t+'/'+f for t in ('OFFLINE','M0HLT') for f in ('manifest.json','targets.npz')]+['targets/costs.json'])
    _,stage=split_task(task)
    return ([task+'/fit/'+f for f in FIT_FILES]+[task+'/bundle.json',task+'/costs.json']
        +([task+'/banks/'+stage+'/'+f for f in ('manifest.json','targets.npz')] if export_kinds(stage) else []))


def target_payloads():
    return [p for r in campaign.tasks() if r['task_id']!='report' for p in output_names(r['task_id'])
            if '/banks/' in p and p.endswith('/targets.npz')]


def fit(folder,spec,arm,stage,tr,val,bank,device):
    folder.mkdir(exist_ok=False)
    student,report,state=training.train(tr,val,bank=bank,node=spec['nodes'][arm][stage],device=device)
    training.validate_report(report,spec,arm,stage,bank.manifest['content_hash'])
    expected=predict(student,val,device=device,batch_size=128)
    save_state(folder/'selected.pt',state)
    saved=torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)
    if saved.keys()!=state.keys() or any(not torch.equal(saved[k],state[k]) for k in state): raise ValueError('State roundtrip differs')
    del student; release()
    selected=native(file_ref(folder/'selected.pt'),device)
    actual=predict(selected,val,device=device,batch_size=128)
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
    if evaluate_probabilities(val.labels,actual)!=report['validation']: raise ValueError('Native selected metrics differ')
    atomic_publish_bytes(folder/'validation_predictions.npz',deterministic_npz_bytes(dict(
        identities=val.identities,labels=val.labels,probabilities=actual)))
    write_immutable_json(folder/'training_report.json',report)
    receipt=artifact('FIT',parents=dict(spec=spec['content_hash'],bundle=bank.manifest['content_hash']),arm=arm,stage=stage,
        checkpoint_probability_roundtrip=True,outputs=[file_ref(folder/f,root=folder) for f in FIT_FILES[:-1]])
    write_immutable_json(folder/'receipt.json',receipt)
    return selected


def validate_fit(spec,arm,stage):
    folder=Path(spec['root'])/task_id(arm,stage); bundle=load_json(folder/'bundle.json')
    rec=load_json(folder/'fit/receipt.json'); validate(rec,'FIT',dict(spec=spec['content_hash'],bundle=bundle['content_hash']))
    if (rec['arm']!=arm or rec['stage']!=stage or rec['checkpoint_probability_roundtrip'] is not True
            or [r['path'] for r in rec['outputs']]!=list(FIT_FILES[:-1])): raise ValueError('Fit output inventory differs')
    for ref in rec['outputs']: validate_file_ref(ref,root=folder/'fit')
    report=load_json(folder/'fit/training_report.json'); training.validate_report(report,spec,arm,stage,bundle['content_hash'])
    a=load_npz_arrays(folder/'fit/validation_predictions.npz'); validate_predictions(a,spec['counts']['validation'])
    if evaluate_probabilities(a['labels'],a['probabilities'])!=report['validation']: raise ValueError('Saved prediction metrics differ')
    return report,a


def validate_task(spec,task,seen=None,*,retired=None):
    seen={} if seen is None else seen
    if task in seen: return seen[task]
    root=Path(spec['root']); row=load_json(root/task/'receipt.json')
    deps=next(r['dependencies'] for r in spec['tasks'] if r['task_id']==task)
    parents=dict(spec=spec['content_hash'],**{d:validate_task(spec,d,seen,retired=retired)['content_hash'] for d in deps})
    validate(row,'TASK',parents)
    if (row['task']!=task or not re.fullmatch('[1-9][0-9]*',row['job_id'])
            or row['job_id']!=submission.accepted_jobs(spec).get(task)
            or [r['path'] for r in row['outputs']]!=output_names(task)):
        raise ValueError('Task publication/journal differs')
    removed={} if retired is None else {r['path']:r for r in retired['payloads']}
    for ref in row['outputs']:
        if ref['path'] in removed:
            if ref!=removed[ref['path']] or (root/ref['path']).exists() or (root/ref['path']).is_symlink():
                raise ValueError('Retired target descriptor/absence differs')
        else: validate_file_ref(ref,root=root)
    costs=load_json(root/task/'costs.json'); validate(costs,'COSTS',dict(spec=spec['content_hash']))
    if (costs['task']!=task or costs['job_id']!=row['job_id'] or costs['new_preflight_performed'] is not False
            or costs['environment']!=spec['anchor']['evidence']['environment']): raise ValueError('Execution evidence differs')
    if task=='targets':
        for t in ('OFFLINE','M0HLT'):
            banks.manifest(banks.directory(spec,ARMS[0],t),banks.teacher_record(spec,ARMS[0],t,payloads=retired is None),
                spec['anchor']['evidence']['prepared'],spec['counts']['train'],payload=retired is None)
    else:
        arm,stage=split_task(task)
        banks.validate_bundle(load_json(root/task/'bundle.json'),spec,arm,stage,payloads=retired is None)
        validate_fit(spec,arm,stage)
        if export_kinds(stage): banks.manifest(banks.directory(spec,arm,stage),banks.teacher_record(spec,arm,stage),
            spec['anchor']['evidence']['prepared'],spec['counts']['train'],payload=retired is None)
    seen[task]=row; return row


def retired_evidence(spec):
    root=Path(spec['root'])
    if not (root/'report/retirement.json').exists(): return None
    if not (root/'report/complete.json').is_file(): raise ValueError('Partial target retirement; preserve evidence, explicit recovery required')
    r=load_json(root/'report/retirement.json'); validate(r,'RETIREMENT',dict(spec=spec['content_hash']))
    done=load_json(root/'report/complete.json')
    validate(done,'COMPLETE',dict(spec=spec['content_hash'],retirement=r['content_hash']))
    if ([v['path'] for v in r['payloads']]!=target_payloads() or done['job_id']!=submission.accepted_jobs(spec).get('report')
            or r['results']!=done['results'] or done['retirement']!=file_ref(root/'report/retirement.json',root=root)
            or done['tasks']!={t['task_id']:load_json(root/t['task_id']/'receipt.json')['content_hash']
                               for t in spec['tasks'] if t['task_id']!='report'}):
        raise ValueError('Retirement coverage/completion differs')
    validate_file_ref(done['results'],root=root)
    return r


def results(spec):
    campaign.validate_spec(spec,imports=False); root=Path(spec['root']); retired=retired_evidence(spec); seen={}
    validate_task(spec,'targets',seen,retired=retired)
    refs=root/'targets/references'
    arrays={n:load_npz_arrays(refs/(n+'.npz')) for n in REFERENCES}
    anchor=arrays['M0HLT']; ix=load_npz_arrays(refs/'original50k.npz')['indices']
    if (ix.dtype!=np.int64 or ix.shape!=(spec['original50k_rows'],) or len(np.unique(ix))!=len(ix)
            or np.any(ix<0) or np.any(ix>=len(anchor['labels']))): raise ValueError('Report subset differs')
    pending=[]; picks={}
    for arm in ARMS:
        for stage in STAGES:
            task=task_id(arm,stage)
            if not (root/task/'receipt.json').is_file(): pending.append(task); continue
            validate_task(spec,task,seen,retired=retired); report,arrays[task]=validate_fit(spec,arm,stage)
            picks[task]=report['selected_pass']
    rows=[]
    for name,a in arrays.items():
        validate_predictions(a,spec['counts']['validation'])
        if not np.array_equal(a['identities'],anchor['identities']) or not np.array_equal(a['labels'],anchor['labels']):
            raise ValueError('Comparison population differs')
        rows.append(dict(model=name,selected_epoch=picks.get(name),input='ORACLE' if name in ('OFFLINE','COARSE_U100') or
            (name in picks and not name.endswith('_DE000')) else 'single HLT',
            validation=evaluate_probabilities(a['labels'],a['probabilities']),
            original50k=evaluate_probabilities(a['labels'][ix],a['probabilities'][ix])))
    for row in rows:
        for role in ('validation','original50k'):
            row[role+'_recovery']=recovery(row[role],rows[0][role],rows[1][role])
    return artifact('RESULTS',parents=dict(spec=spec['content_hash'],**{t:r['content_hash'] for t,r in seen.items()}),
        counts=spec['counts'],rows=rows,pending=pending,single_training_seed=True,original50k_selection=False)


def retire(spec,result,job):
    if result['pending']: raise ValueError('Cannot retire targets before all twenty fits finish')
    root=Path(spec['root']).absolute()
    if root!=root.resolve(strict=True): raise PermissionError('Retirement root is not canonical')
    folder=root/'report'
    write_immutable_json(folder/'results.json',result)
    refs=[]; targets=[]
    # Check the original registered paths before file_ref can resolve an alias.
    for name in target_payloads():
        path=root/name
        if path.is_symlink() or path!=path.resolve(strict=True) or not path.resolve().is_relative_to(root):
            raise PermissionError('Retirement target escapes owned canonical root')
        ref=file_ref(path,root=root)
        if ref['path']!=name: raise PermissionError('Retirement target alias differs')
        refs.append(ref); targets.append(path)
    intent=artifact('RETIREMENT',parents=dict(spec=spec['content_hash']),payloads=refs,
        results=file_ref(folder/'results.json',root=root),reason='all_consumers_complete_saved_results_native_models_retained')
    # Resolve and authenticate every exact leaf before the first destructive action.
    for path,ref in zip(targets,refs):
        if path.is_symlink() or path.absolute()!=path.resolve(strict=True) or not path.resolve().is_relative_to(root):
            raise PermissionError('Retirement target escapes owned canonical root')
        validate_file_ref(ref,root=root)
    write_immutable_json(folder/'retirement.json',intent)
    for path,ref in zip(targets,refs):
        validate_file_ref(ref,root=root); path.unlink()
    done=artifact('COMPLETE',parents=dict(spec=spec['content_hash'],retirement=intent['content_hash']),job_id=job,
        results=intent['results'],retirement=file_ref(folder/'retirement.json',root=root),
        tasks={r['task_id']:load_json(root/r['task_id']/'receipt.json')['content_hash'] for r in spec['tasks'] if r['task_id']!='report'})
    write_immutable_json(folder/'complete.json',done)
    print(f'Retired {len(refs)} owned TRAIN payloads ({sum(r["bytes"] for r in refs)/2**30:.3f} GiB); no backup. '
          'Native models, moments, manifests, predictions and reports retained; targets can be re-exported.',flush=True)
    return done


def run(spec,task,*,device='cuda'):
    if task not in {r['task_id'] for r in spec['tasks']}: raise ValueError('Unknown task')
    campaign.check(spec); job=execution(spec,task); jobs=submission.accepted_jobs(spec)
    if jobs.get(task)!=job: raise ValueError('Allocation is not exact accepted task')
    deps=next(r['dependencies'] for r in spec['tasks'] if r['task_id']==task); seen={}
    parents=dict(spec=spec['content_hash'])
    for dep in deps:
        rec=validate_task(spec,dep,seen); submission.successful(spec,dep,rec['job_id']); parents[dep]=rec['content_hash']
    if installed_environment()!=spec['anchor']['evidence']['environment']: raise ValueError('Installed environment differs')
    storage(spec); root=Path(spec['root']); folder=root/task; folder.mkdir(exist_ok=False)
    if task=='report':
        value=results(spec)
        if source.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']:
            raise ValueError('Source changed before target retirement')
        done=retire(spec,value,job)
        if results(spec)!=value: raise ValueError('Post-retirement result validation differs')
        print_results(value); return done
    if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    log=PhaseLog(folder,lambda:memory_snapshot(torch)); meta=regular_cache.metadata(spec['anchor']['evidence']['prepared_root'])
    if task=='targets':
        for name in ('OFFLINE','M0HLT'):
            with log.phase('export_frozen_'+name+'_TRAIN_targets'):
                record=banks.teacher_record(spec,ARMS[0],name); model=native(record['files']['checkpoint'],device)
                tr=data(spec,'train',record['coordinate'],meta)
                banks.export(banks.directory(spec,ARMS[0],name),model,tr,record,device=device)
                del tr,model; release()
        with log.phase('snapshot_historical_validation_comparisons'):
            val=data(spec,'validation','D000',meta)
            snapshot_references(folder/'references',dict(spec,teacher=spec['acceptance']['teacher'],
                reference=spec['acceptance']['reference']),val,meta)
            del val; release()
    else:
        arm,stage=split_task(task)
        with log.phase('load_rung_and_exact_branch_targets'):
            tr,val=(data(spec,r,STAGES[stage],meta) for r in ('train','validation'))
            bank=banks.Bundle(spec,arm,stage,tr); write_immutable_json(folder/'bundle.json',bank.manifest)
        with log.phase('fresh_fixed30_science'):
            model=fit(folder/'fit',spec,arm,stage,tr,val,bank,device)
        if export_kinds(stage):
            with log.phase('export_selected_native_TRAIN_targets'):
                record=banks.teacher_record(spec,arm,stage)
                banks.export(banks.directory(spec,arm,stage),model,tr,record,device=device)
        del model,tr,val,bank; release()
    costs=artifact('COSTS',parents=dict(spec=spec['content_hash']),task=task,job_id=job,
        environment=installed_environment(),memory=memory_snapshot(torch),new_preflight_performed=False)
    write_immutable_json(folder/'costs.json',costs); storage(spec)
    if source.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']: raise ValueError('Source changed during science')
    row=artifact('TASK',parents=parents,task=task,job_id=job,outputs=[file_ref(root/p,root=root) for p in output_names(task)])
    write_immutable_json(folder/'receipt.json',row); validate_task(spec,task)
    return row


def print_results(value):
    print('Four independent hybrid ladders; CE/KD ratios, total embedding weight5. NO independent test.')
    print('Original50k report-only; only DE000 is HLT-deployable. Same full-validation-selected checkpoint throughout.')
    for role in ('validation','original50k'):
        print('\n'+role)
        print(f'{"model":24} {"pick":>5} {"accuracy":>10} {"AUC":>10} {"AUC rec%":>10} input')
        for row in value['rows']:
            m=row[role]; rec=row[role+'_recovery']['macro_ovr_auc']; text='n/a' if rec is None else f'{rec:.1f}'
            print(f'{row["model"]:24} {str(row["selected_epoch"] or "-"):>5} {m["accuracy"]:10.6f} {m["macro_ovr_auc"]:10.6f} {text:>10} {row["input"]}')
            print('  QCD rejection @50%: '+', '.join(n+'='+('censored' if v['rejection_at_50pct'] is None else f'{v["rejection_at_50pct"]:.1f}') for n,v in m['per_class'].items()))
    print('Pending: '+(', '.join(value['pending']) or 'none'))

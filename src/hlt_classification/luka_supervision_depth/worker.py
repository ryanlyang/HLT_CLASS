"""Production workers with retained native results and bounded owned scratch."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import os
import platform
import re
import shutil
import subprocess
import sys
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json,load_npz_arrays,write_immutable_json,atomic_publish_bytes,deterministic_npz_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities,recovery
from hlt_classification.luka_attention.worker import execution as gpu_execution
from hlt_classification.luka_attention.banks import counts
from hlt_classification.luka_embedding_weight.worker import storage,validate_predictions
from hlt_classification.luka_trial_hybrid.worker import native,release
from hlt_classification.luka_joint_prediction.worker import scratch_directory
from hlt_classification.luka_embedding_aux import targets as embedding
from hlt_classification.luka_fullsim import regular_cache
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog,memory_snapshot
from .contracts import artifact,validate,ARMS,TEACHERS,TEACHER_TASKS,REFERENCES,recipe
from . import campaign,source,submission,banks,training,startup,data

FIT_FILES=('selected.pt','training_report.json','validation_predictions.npz','receipt.json')

def execution(spec, task):
    job = os.environ.get('SLURM_JOB_ID', ''); site = spec['execution_site']
    if not re.fullmatch('[1-9][0-9]*', job):
        raise PermissionError('A real Slurm allocation is required')
    details = subprocess.run(['scontrol', 'show', 'job', '-o', job], check=True, capture_output=True, text=True).stdout
    fields = dict(x.split('=', 1) for x in details.split() if '=' in x)
    if (fields.get('JobName') != 'lukasd_'+task or fields.get('Comment') != 'lukasd:'+spec['content_hash']+':'+task
            or fields.get('WorkDir') != spec['project_dir']):
        raise PermissionError('Exact task identity differs')
    if task not in ('report','cleanup'):
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

def output_names(task):
    if task=='targets':
        return (['targets/references/'+n+'.npz' for n in (*REFERENCES,'original50k')]
            + ['targets/banks/'+t+'/'+f for t in TEACHERS for f in ('manifest.json','targets.npz')]
            + ['targets/baseline_report.json','targets/shared.json','targets/startup.json','targets/token_shape_startup.json','targets/costs.json'])
    if task=='report': return ['report/results.json','report/costs.json']
    names=[task+'/fit/'+f for f in FIT_FILES]+[task+'/'+f for f in ('bundle.json','startup.json','costs.json')]
    if task in TEACHER_TASKS: names += [task+'/bank/'+f for f in ('manifest.json','targets.npz','normalization.npz')]
    elif task in ARMS and ARMS[task]['depths']: names += [task+'/'+f for f in ('token_manifest.json','token_index.npz','scratch.json')]
    elif task not in ARMS: raise ValueError('Unknown task')
    return names


def target_payloads():
    return ['targets/banks/'+t+'/targets.npz' for t in TEACHERS]+[t+'/bank/targets.npz' for t in TEACHER_TASKS]


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


def fit(spec,task,tr,val,bank,device):
    folder=Path(spec['root'])/task/'fit'; folder.mkdir(exist_ok=False)
    model,report,state=training.train(tr,val,node=spec['nodes'][task],bank=bank,device=device)
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
    arrays=load_npz_arrays(folder/'fit/validation_predictions.npz'); validate_predictions(arrays,spec['counts']['validation'])
    if evaluate_probabilities(arrays['labels'],arrays['probabilities'])!=report['validation']: raise ValueError('Saved metrics differ')
    startup.check(load_json(folder/'startup.json'),spec,task,bundle['content_hash'])
    return report,arrays


def validate_bundle(spec,task,value):
    if task in TEACHER_TASKS:
        expected=artifact('BUNDLE',parents=dict(spec=spec['content_hash'],recipe=training.registered(task)['content_hash']),arm=task,records={})
    else:
        records=banks.grid.manifests(spec,payloads=False); r=recipe(task)
        for t in sorted(set(r['target_names'].values())-set(TEACHERS)):
            folder=Path(spec['root'])/t
            m=load_json(folder/'bank/manifest.json'); ref=file_ref(folder/'fit/selected.pt')
            embedding.validate(m,'BANK',dict(recipe=embedding.recipe()['content_hash'],prepared=spec['anchor']['evidence']['prepared'],teacher=ref['sha256']))
            if (m['teacher']!=ref or m['payload']['path']!='targets.npz' or m['role']!='train'
                    or m['coordinate']!='OFFLINE' or m['rows']!=spec['counts']['train'] or m['width']!=128):
                raise ValueError('New teacher target semantics differ')
            records[t]=m
        if r['depths']: records['tokens']=load_json(Path(spec['root'])/task/'token_manifest.json')
        expected=artifact('BUNDLE',parents=dict(spec=spec['content_hash'],recipe=r['content_hash'],**{t:m['content_hash'] for t,m in records.items()}),arm=task,records=records)
    if value!=expected: raise ValueError('Exact recipe/teacher target bundle differs')


def retired_evidence(spec):
    folder=Path(spec['root'])/'cleanup'
    if not (folder/'intent.json').exists(): return None
    if not (folder/'complete.json').is_file(): raise ValueError('Partial cleanup: preserve evidence, explicit recovery required')
    intent=load_json(folder/'intent.json'); done=load_json(folder/'complete.json')
    validate(intent,'RETIREMENT',dict(spec=spec['content_hash']))
    validate(done,'COMPLETE',dict(spec=spec['content_hash'],retirement=intent['content_hash']))
    if (done['job_id']!=submission.accepted_jobs(spec).get('cleanup') or done['results']!=intent['results']
            or [r['path'] for r in intent['payloads']]!=target_payloads()): raise ValueError('Retirement identity differs')
    validate_file_ref(done['results'],root=Path(spec['root']))
    return intent


def validate_task(spec,task,seen=None,retired=None):
    seen={} if seen is None else seen
    if task in seen: return seen[task]
    root=Path(spec['root']); row=load_json(root/task/'receipt.json')
    deps=next(t['dependencies'] for t in spec['tasks'] if t['task_id']==task)
    parents=dict(spec=spec['content_hash'],**{d:validate_task(spec,d,seen,retired)['content_hash'] for d in deps})
    validate(row,'TASK',parents)
    if (row['task']!=task or row['job_id']!=submission.accepted_jobs(spec).get(task)
            or [r['path'] for r in row['outputs']]!=output_names(task)): raise ValueError('Task/journal/output inventory differs')
    removed={} if retired is None else {r['path']:r for r in retired['payloads']}
    for ref in row['outputs']:
        if ref['path'] in removed:
            if ref!=removed[ref['path']] or (root/ref['path']).exists() or (root/ref['path']).is_symlink(): raise ValueError('Retired payload differs')
        else: validate_file_ref(ref,root=root)
    costs=load_json(root/task/'costs.json'); validate(costs,'COSTS',dict(spec=spec['content_hash']))
    if costs['task']!=task or costs['job_id']!=row['job_id'] or costs['environment']!=spec['anchor']['evidence']['environment'] or costs['separate_preflight_performed'] is not False:
        raise ValueError('Runtime evidence differs')
    if task in ARMS or task in TEACHER_TASKS:
        validate_fit(spec,task)
        if task in ARMS and ARMS[task]['depths']: retained_tokens(spec,task)
        if task in TEACHER_TASKS:
            a=load_npz_arrays(root/task/'bank/normalization.npz')
            if (set(a)!={'identities','labels','mean','scale'} or a['identities'].shape!=(spec['counts']['train'],32)
                    or a['identities'].dtype!=np.uint8 or len(np.unique(a['identities'],axis=0))!=spec['counts']['train']
                    or a['labels'].shape!=(spec['counts']['train'],) or a['labels'].dtype!=np.int64
                    or np.any(a['labels']<0) or np.any(a['labels']>=11)
                    or any(a[k].shape!=(128,) or a[k].dtype!=np.float64 or not np.isfinite(a[k]).all() for k in ('mean','scale'))
                    or np.any(a['scale']<1e-3)): raise ValueError('Retained teacher moments invalid')
    elif task=='targets':
        shared=load_json(root/'targets/shared.json'); validate(shared,'SHARED',dict(spec=spec['content_hash']))
        expected=banks.grid.manifests(spec,payloads=retired is None)
        if shared['banks']!=expected: raise ValueError('Shared bank descriptors differ')
        for filename,name in (('startup.json','baseline'),('token_shape_startup.json','set248_p10')):
            value=load_json(root/'targets'/filename)
            startup.check(value,spec,name,value['parents']['bundle'])
    seen[task]=row; return row


def retained_tokens(spec,task):
    folder=Path(spec['root'])/task; m=load_json(folder/'token_manifest.json'); teacher=banks.record(spec,'UE100'); depths=ARMS[task]['depths']
    validate(m,'TOKEN_BANK',dict(teacher=teacher['content_hash'],prepared=spec['anchor']['evidence']['prepared']))
    if (m['teacher']!=teacher or m['depths']!=depths or m['rows']!=spec['counts']['train'] or m['role']!='train'
            or [r['path'] for r in m['outputs']]!=['index.npz',*(f'tokens{d}.npy' for d in depths)]): raise ValueError('Retained token binding differs')
    ref=file_ref(folder/'token_index.npz')
    if any(ref[k]!=m['outputs'][0][k] for k in ('bytes','sha256')): raise ValueError('Retained index differs')
    a=load_npz_arrays(folder/'token_index.npz')
    class RetainedRows:
        role='train'
        coordinate_name='D000'
        foundation_sha256=spec['anchor']['evidence']['prepared']
        identities=a['identities']
        labels=a['labels']
        blocks=[SimpleNamespace(offsets=a['offsets'])]
        def __len__(self): return spec['counts']['train']
    banks.validate_token_index(m,a,RetainedRows(),teacher,depths)
    proof=load_json(folder/'scratch.json'); validate(proof,'SCRATCH',dict(spec=spec['content_hash'],bank=m['content_hash']))
    if proof['task']!=task or proof['retired'] is not True or proof['persistent_payloads'] is not False: raise ValueError('Scratch ownership differs')
    bundle=load_json(folder/'bundle.json')
    if bundle['records']['tokens']!=m: raise ValueError('Fit consumed different tokens')
    return m


def student_run(spec,task,tr,val,meta,device):
    folder=Path(spec['root'])/task; r=recipe(task)
    def execute(bank):
        write_immutable_json(folder/'bundle.json',bank.manifest)
        check=startup.run(spec,tr,val,bank,spec['nodes'][task],device)
        write_immutable_json(folder/'startup.json',check); release()
        fit(spec,task,tr,val,bank,device)
    if not r['depths']:
        execute(banks.Bundle(spec,tr,task)); return
    local=scratch_directory(spec)
    required=int(counts(tr).sum())*128*4*len(r['depths'])+512*2**20
    if shutil.disk_usage(local).free<required: raise OSError(f'Need {required/2**30:.2f}GiB node-local disk; no home fallback')
    token_bank=None
    with TemporaryDirectory(prefix='lukasd-'+os.environ.get('SLURM_JOB_ID','local')+'-',dir=local) as temporary:
        try:
            rich=data.load(spec,'train','U100',meta); teacher=banks.record(spec,'UE100'); frozen=native(teacher['files']['checkpoint'],device)
            m=banks.export_tokens(Path(temporary)/'tokens',frozen,rich,tr,teacher,r['depths'],device)
            write_immutable_json(folder/'token_manifest.json',m)
            atomic_publish_bytes(folder/'token_index.npz',(Path(temporary)/'tokens/index.npz').read_bytes())
            del frozen,rich; release()
            token_bank=banks.TokenBank(Path(temporary)/'tokens',tr,teacher,r['depths'])
            execute(banks.Bundle(spec,tr,task,token_bank))
        finally:
            if token_bank is not None: token_bank.close()
    write_immutable_json(folder/'scratch.json',artifact('SCRATCH',parents=dict(spec=spec['content_hash'],bank=m['content_hash']),
        task=task,retired=True,persistent_payloads=False))
    print('Removed this job-owned token scratch bank. Selected model, results, index and moments retained.',flush=True)


def results(spec):
    campaign.validate_spec(spec,imports=False); root=Path(spec['root']); seen={}; retired=retired_evidence(spec)
    receipt=validate_task(spec,'targets',seen,retired)
    submission.successful(spec,'targets',receipt['job_id']); refs=root/'targets/references'
    arrays={n:load_npz_arrays(refs/(n+'.npz')) for n in REFERENCES}; anchor=arrays['M0HLT']
    ix=load_npz_arrays(refs/'original50k.npz')['indices']
    if (ix.dtype!=np.int64 or ix.shape!=(spec['original50k_rows'],) or len(np.unique(ix))!=len(ix)
            or np.any(ix<0) or np.any(ix>=len(anchor['labels']))): raise ValueError('Invalid report-only subset')
    picks={}; pending=[]; token_refs={}
    for task in (*TEACHER_TASKS,*ARMS):
        if not (root/task/'receipt.json').is_file(): pending.append(task); continue
        receipt=validate_task(spec,task,seen,retired); submission.successful(spec,task,receipt['job_id'])
        report,arrays[task]=validate_fit(spec,task); picks[task]=report['selected_pass']
        if task in ARMS and ARMS[task]['depths']:
            m=retained_tokens(spec,task); a=load_npz_arrays(root/task/'token_index.npz')
            for d,ref in zip(m['depths'],m['outputs'][1:]):
                evidence=(ref,a[f'mean{d}'].tolist(),a[f'scale{d}'].tolist())
                if d in token_refs and token_refs[d]!=evidence: raise ValueError('Cross-fit token targets differ at depth '+str(d))
                token_refs[d]=evidence
    rows=[]
    for name,a in arrays.items():
        validate_predictions(a,spec['counts']['validation'])
        if not np.array_equal(a['identities'],anchor['identities']) or not np.array_equal(a['labels'],anchor['labels']): raise ValueError('Comparison population differs')
        rows.append(dict(model=name,selected_epoch=picks.get(name),input='ORACLE' if name=='OFFLINE' or name in TEACHER_TASKS else 'single HLT',
            validation=evaluate_probabilities(a['labels'],a['probabilities']),original50k=evaluate_probabilities(a['labels'][ix],a['probabilities'][ix])))
    by={r['model']:r for r in rows}
    for row in rows:
        for role in ('validation','original50k'):
            row[role+'_recovery']=recovery(row[role],by['M0HLT'][role],by['OFFLINE'][role])
            row[role+'_delta_baseline']=row[role]['macro_ovr_auc']-by['K1_E40_PLAIN'][role]['macro_ovr_auc']
    return artifact('RESULTS',parents=dict(spec=spec['content_hash'],**{t:r['content_hash'] for t,r in seen.items()}),rows=rows,pending=pending,
        original50k_selection=False,student_seeds=2,independent_test=False)


def retire(spec,job):
    root=Path(spec['root']).absolute(); folder=root/'cleanup'; refs=[]; paths=[]
    if root!=root.resolve(strict=True): raise PermissionError('Noncanonical owned root')
    result=results(spec)
    if result['pending'] or load_json(root/'report/results.json')!=result: raise ValueError('All results must validate before cleanup')
    for name in target_payloads():
        path=root/name
        if path.is_symlink() or path!=path.resolve(strict=True) or not path.is_relative_to(root) or path.stat().st_nlink!=1:
            raise PermissionError('Not an exclusively owned canonical payload')
        ref=file_ref(path,root=root); validate_file_ref(ref,root=root); refs.append(ref); paths.append(path)
    intent=artifact('RETIREMENT',parents=dict(spec=spec['content_hash']),payloads=refs,results=file_ref(root/'report/results.json',root=root))
    write_immutable_json(folder/'intent.json',intent)
    for path,ref in zip(paths,refs):
        if path.is_symlink() or path!=path.resolve(strict=True) or path.stat().st_nlink!=1: raise PermissionError('Payload ownership changed')
        validate_file_ref(ref,root=root); path.unlink()
    done=artifact('COMPLETE',parents=dict(spec=spec['content_hash'],retirement=intent['content_hash']),job_id=job,results=intent['results'])
    write_immutable_json(folder/'complete.json',done)
    if results(spec)!=result: raise ValueError('Post-retirement results differ')
    print('Retired six new owned compact TRAIN payloads; no backup. All selected models and results retained.',flush=True)
    return done


def run(spec,task,*,device='cuda'):
    if task not in {r['task_id'] for r in spec['tasks']}: raise ValueError('Unknown task')
    campaign.check(spec); job=execution(spec,task)
    if submission.accepted_jobs(spec).get(task)!=job: raise ValueError('Allocation not accepted')
    parents=dict(spec=spec['content_hash']); seen={}
    for dep in next(t['dependencies'] for t in spec['tasks'] if t['task_id']==task):
        rec=validate_task(spec,dep,seen); submission.successful(spec,dep,rec['job_id']); parents[dep]=rec['content_hash']
    if installed_environment()!=spec['anchor']['evidence']['environment']: raise ValueError('Environment differs')
    storage(spec); root=Path(spec['root']); folder=root/task; folder.mkdir(exist_ok=False)
    if task=='cleanup': return retire(spec,job)
    if task=='report':
        value=results(spec)
        if value['pending']: raise ValueError('Incomplete campaign')
        write_immutable_json(folder/'results.json',value); print_results(value)
    else:
        if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats()
        log=PhaseLog(folder,lambda:memory_snapshot(torch)); meta=regular_cache.metadata(spec['anchor']['evidence']['prepared_root'])
        if task=='targets':
            for t in TEACHERS:
                with log.phase('export_compact_'+t):
                    teacher=banks.record(spec,t); frozen=native(teacher['files']['checkpoint'],device)
                    rich=data.load(spec,'train',teacher['coordinate'],meta)
                    banks.export_shared(spec,t,frozen,rich,device); del rich,frozen; release()
            with log.phase('references_and_disposable_safety'):
                tr,val=(data.load(spec,role,'D000',meta) for role in ('train','validation'))
                snapshot_references(spec,val); bank=banks.Bundle(spec,tr,'baseline')
                shared=artifact('SHARED',parents=dict(spec=spec['content_hash']),banks=banks.grid.manifests(spec))
                write_immutable_json(folder/'shared.json',shared)
                proof=startup.run(spec,tr,val,bank,spec['nodes']['baseline'],device); write_immutable_json(folder/'startup.json',proof)
                # Shape/resource-only token probe; no target statistics or science derived from it.
                class ShapeTokens:
                    def batch(self,ix,width): return {d:np.zeros((len(ix),width,128),np.float32) for d in (2,4,8)}
                bank.tokens=ShapeTokens()
                bank.manifest=artifact('SHAPE_PROBE',parents=dict(spec=spec['content_hash'],shared=shared['content_hash']),synthetic_targets=True)
                proof=startup.run(spec,tr,val,bank,spec['nodes']['set248_p10'],device)
                write_immutable_json(folder/'token_shape_startup.json',proof)
                del tr,val,bank; release()
        elif task in TEACHER_TASKS:
            with log.phase('OFFLINE_teacher_fixed30_and_compact_export'):
                tr,val=(data.load(spec,role,'OFFLINE',meta) for role in ('train','validation'))
                bank=SimpleNamespace(manifest=artifact('BUNDLE',parents=dict(spec=spec['content_hash'],recipe=training.registered(task)['content_hash']),arm=task,records={}))
                write_immutable_json(folder/'bundle.json',bank.manifest)
                proof=startup.run(spec,tr,val,bank,spec['nodes'][task],device); write_immutable_json(folder/'startup.json',proof); release()
                fit(spec,task,tr,val,bank,device)
                ref=file_ref(folder/'fit/selected.pt'); frozen=native(ref,device)
                embedding.export(folder/'bank',frozen,tr,teacher_ref=ref,device=device)
                a=load_npz_arrays(folder/'bank/targets.npz')
                atomic_publish_bytes(folder/'bank/normalization.npz',deterministic_npz_bytes({k:a[k] for k in ('identities','labels','mean','scale')}))
                del tr,val,frozen; release()
        else:
            with log.phase('HLT_student_fixed30_owned_scratch'):
                tr,val=(data.load(spec,role,'D000',meta) for role in ('train','validation'))
                student_run(spec,task,tr,val,meta,device); del tr,val; release()
    costs=artifact('COSTS',parents=dict(spec=spec['content_hash']),task=task,job_id=job,environment=installed_environment(),
        memory=memory_snapshot(torch),separate_preflight_performed=False)
    write_immutable_json(folder/'costs.json',costs); storage(spec)
    if source.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']: raise ValueError('Source changed during worker')
    row=artifact('TASK',parents=parents,task=task,job_id=job,outputs=[file_ref(root/n,root=root) for n in output_names(task)])
    write_immutable_json(folder/'receipt.json',row); validate_task(spec,task)
    return row


def print_results(value):
    print('Fixed K1/E40/CE0.5 plus tokens/depth/diversity. Fresh HLT students; two OFFLINE oracle teachers.')
    print('Exploratory validation; NO independent test. Original50k report-only; all auxiliary heads removed at inference.')
    for role in ('validation','original50k'):
        print('\n'+role); print(f'{"model":25} {"pick":>5} {"AUC":>11} {"recovery%":>10} {"delta K1/E40":>14}')
        for row in value['rows']:
            rec=row[role+'_recovery']['macro_ovr_auc']; rec='n/a' if rec is None else f'{rec:.1f}'
            print(f'{row["model"]:25} {str(row["selected_epoch"] or "-"):>5} {row[role]["macro_ovr_auc"]:11.8f} {rec:>10} {row[role+"_delta_baseline"]:+14.8f}')
    print('Pending: '+(', '.join(value['pending']) or 'none'))

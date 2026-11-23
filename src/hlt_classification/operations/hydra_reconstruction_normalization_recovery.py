"""Versioned execution adapter for the immutable dc5920f0 reconstruction study.

Only new, campaign-bound attempts use this adapter. Original source/specs and
completed artifacts remain untouched. Replacement fit parents record this repair.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import types

BASE=Path('/cs/home/rlyang/hlt_u100_pairs')
PARENT=BASE/'runs/balanced_reco_600k_dc5920f0_r1'
COMMIT='dc5920f0ed95beeb288816276beb36c485332561'
QUAL_JOB='219148'
QUAL_NODE='gpu1905'
NORMAL_HASH='c06fa2b1d349e0a8615d0ef14132975bd6f1b543a46d3df3ebe2c455da9ffc59'
AUTH='AUTHORIZE_BALANCED_RECO_NORMALIZATION_RECOVERY_V1'
PREFIX='LUKA_BALANCED_RECO_NORMALIZATION_RECOVERY'


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact(kind,**fields):
    obj=dict(contract=PREFIX+'_'+kind+'/v1',schema_version=1,final_test_accessed=False,
        final_test_sealed=True,evaluation_scope='development_validation',**fields)
    return dict(obj,content_hash=hashlib.sha256(canonical(obj)).hexdigest())


def read(path):
    obj=json.loads(Path(path).read_bytes());body={k:v for k,v in obj.items() if k!='content_hash'}
    require(obj.get('content_hash')==hashlib.sha256(canonical(body)).hexdigest(),'Metadata checksum differs: '+str(path))
    return obj


def ref(path):
    path=Path(path).resolve(strict=True)
    return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)


def referenced(row):
    path=Path(row['path']);require(path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],
        'Referenced file changed: '+str(path));return path


def source(project):
    project=Path(project).resolve(strict=True)
    require(Path(__file__).resolve()==project/'src/hlt_classification/operations/hydra_reconstruction_normalization_recovery.py',
        'Repair adapter imported outside its pinned source')
    def git(*args):return subprocess.check_output(['git','-C',str(project),*args],timeout=60)
    require(not git('status','--porcelain','--untracked-files=all').strip(),'Recovery source must be clean')
    commit=git('rev-parse','HEAD').decode().strip()
    subprocess.run(['git','-C',str(project),'merge-base','--is-ancestor',commit,'origin/main'],check=True,timeout=30)
    paths=git('ls-files','-z').decode().split('\0');digest=hashlib.sha256()
    for name in filter(None,paths):
        digest.update(name.encode()+b'\0'+(project/name).read_bytes()+b'\0')
    return dict(commit=commit,tracked_files_sha256=digest.hexdigest())


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def original():
    old=read(PARENT/'campaign_spec.json')
    require(old['root']==str(PARENT) and old['commit']==COMMIT,'Unexpected original execution')
    sys.path.insert(0,str(Path(old['project'])/'src'))
    from hlt_classification.luka_balanced_reconstruction_hydra import (
        campaign,contracts,resources,artifacts,inputs,worker,training,launch,kernel,diagnostics)
    require(Path(campaign.__file__).resolve()==Path(old['project'])/'src/hlt_classification/luka_balanced_reconstruction_hydra/campaign.py',
        'Original package was already imported from another source')
    diagnostics.enable_stall_traces();kernel.configure(1,12);campaign.check(old)
    q=resources.qualification(old)
    require(q['target_normalization_sha256']==NORMAL_HASH,'Unexpected qualification normalization')
    task=worker.task_receipt(old,'qualify')
    require(task['job']==QUAL_JOB,'Original qualification job changed')
    state=states([QUAL_JOB])[QUAL_JOB]
    require(state==('COMPLETED','0:0'),'Original qualification did not complete successfully')
    node=subprocess.check_output(['sacct','-X','-n','-P','-j',QUAL_JOB,'-o','NodeList'],text=True,timeout=30).strip().strip('|')
    require(node==QUAL_NODE,'Original qualification node differs')
    return types.SimpleNamespace(spec=old,q=q,c=contracts,campaign=campaign,resources=resources,
        artifacts=artifacts,inputs=inputs,worker=worker,training=training,launch=launch)


def states(jobs):
    jobs=list(jobs)
    if not jobs:return {}
    text=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(jobs),
        '-o','JobIDRaw,State,ExitCode'],text=True,timeout=30)
    rows={}
    for line in text.splitlines():
        fields=line.strip().split('|')
        if len(fields)>=3 and fields[0] in jobs:
            require(fields[0] not in rows,'Ambiguous accounting row');rows[fields[0]]=(fields[1].split()[0],fields[2])
    require(set(rows)==set(jobs),'Missing scheduler accounting; wait before recovery')
    return rows


def pending_fields(old,row):
    text=subprocess.check_output(['scontrol','show','job','-o',row['job']],text=True,timeout=30)
    fields=dict(x.split('=',1) for x in text.split() if '=' in x)
    require(fields.get('JobId')==row['job'] and fields.get('JobName')=='lukabrec_'+row['task']
        and fields.get('Comment')==old['content_hash']+':'+row['task']
        and fields.get('WorkDir')==old['project'],'Scheduler job is not bound to this campaign')
    return fields


def module_paths(project):
    folder=Path(project)/'src/hlt_classification/luka_balanced_reconstruction_hydra'
    return dict(normalization=ref(folder/'normalization.py'),inputs=ref(folder/'inputs.py'))


def create(project,commit,candidate=None):
    project=Path(project).resolve(strict=True);src=source(project)
    require(src['commit']==commit and re.fullmatch('[0-9a-f]{40}',commit),'Exact pushed repair commit required')
    require(project.is_relative_to(BASE),'Use a Hydra source worktree')
    old=original();root=BASE/'runs'/('balanced_reco_normalization_'+commit[:8]+'_r1')
    require(not root.exists() and not root.is_symlink(),'Recovery already exists; inspect saved records')
    modules=module_paths(project);jobs=old.campaign.accepted_jobs(old.spec)
    for row in jobs:
        require(not row.get('execution_recovery'),'An earlier normalization repair already owns this campaign')
    candidate_ref=None
    if candidate is not None:
        norms=load_file('hlt_classification.luka_balanced_reconstruction_hydra.normalization',referenced(modules['normalization']))
        norms.validate(read(candidate),old.spec,expected_hash=NORMAL_HASH);candidate_ref=ref(candidate)
    root.mkdir(parents=True)
    spec=artifact('SPEC',root=str(root),project=str(project),source=src,parent_spec=ref(PARENT/'campaign_spec.json'),
        qualification=ref(PARENT/'qualification.json'),qualification_job=QUAL_JOB,normalization_sha256=NORMAL_HASH,
        qualification_node=QUAL_NODE,modules=modules,candidate=candidate_ref,
        original_jobs={row['job']:row['task'] for row in jobs},policy=dict(
            pending_reconstruction='cancel_exact_bound_pending_only',running_jobs='preserve',completed_fits='preserve',
            replacement_fits='terminal_reconstruction_without_scientific_epochs_only',
            scheduler_edges='teacher_only',normalization='recreate_exact_qualified_hash_then_reuse',
            new_fit_parents='execution_recovery_and_comparison_hash',old_source_and_spec='unchanged'))
    publish(root/'repair_spec.json',spec);return spec


def publish(path,obj):
    from hlt_classification.data.cache_contracts import write_immutable_json
    write_immutable_json(Path(path),obj)


def check(spec):
    require(spec['contract']==PREFIX+'_SPEC/v1' and spec['final_test_accessed'] is False,'Unexpected recovery scope')
    require(spec['normalization_sha256']==NORMAL_HASH and spec['qualification_node']==QUAL_NODE
        and spec['qualification_job']==QUAL_JOB,'Recovery qualification binding differs')
    require(read(Path(spec['root'])/'repair_spec.json')==spec,'Recovery spec changed')
    require(source(spec['project'])==spec['source'],'Repair source changed')
    referenced(spec['parent_spec']);referenced(spec['qualification'])
    for row in spec['modules'].values():referenced(row)
    return original()


def wrapper(spec,phase,task=None):
    args=[str(Path(spec['project'])/'sbatch/run_luka_reconstruction_normalization_recovery.sh'),
        spec['project'],str(Path(spec['root'])/'repair_spec.json'),phase]
    return args+([task] if task is not None else [])


def bootstrap_commands(spec):
    old=read(referenced(spec['parent_spec']));root=Path(spec['root']);rows=[]
    for phase in ('normalize','release'):
        gpu=phase=='normalize' and not spec.get('candidate');args=['sbatch','--parsable','--partition='+('gpus' if gpu else 'compute'),
            '--account=default','--qos=normal','--nodes=1','--ntasks=1',
            '--cpus-per-task='+('4' if gpu else '1'),'--mem='+('64000M' if gpu else '4000M'),
            '--time='+('240' if gpu else '60'),'--job-name=lukabnorm_'+phase,
            '--output='+str(root/(phase+'-%j.out')),'--chdir='+old['project'],'--export=NONE',
            '--comment='+spec['content_hash']+':'+phase,'--kill-on-invalid-dep=yes']
        if gpu:args+=['--gres=gpu:geforce_gtx_1080_ti:1','--nodelist='+spec['qualification_node']]
        if phase=='release':args+=['--dependency=afterok:{normalize}']
        rows.append(dict(task=phase,command=args+wrapper(spec,phase)))
    return rows


def scheduler_env():
    return {k:v for k,v in os.environ.items() if not k.startswith(('SBATCH_','SLURM_'))}


def batch_script_index(command):
    require(command and command[0]=='sbatch','Expected a direct sbatch command')
    cut=next((i for i in range(1,len(command)) if not command[i].startswith('-')),None)
    require(cut is not None,'Missing batch script')
    script=Path(command[cut])
    require(script.is_absolute() and script.is_file()
        and script.name=='run_luka_reconstruction_normalization_recovery.sh',
        'Pass the absolute recovery batch script directly to sbatch, without a bash prefix')
    with script.open('rb') as stream:header=stream.readline()
    require(header==b'#!/usr/bin/env bash\n','Batch script requires a Bash shebang and Unix line endings')
    return cut


def dry_review(rows):
    for row in rows:batch_script_index(row['command'])
    config=subprocess.check_output(['scontrol','show','config'],text=True,timeout=30)
    require(re.search(r'(?m)^\s*ClusterName\s*=\s*hydra\s*$',config),'Hydra required')
    for row in rows:
        args=[x for x in row['command'][1:] if not x.startswith('--dependency=')]
        subprocess.run(['sbatch','--test-only',*args],check=True,timeout=30,env=scheduler_env())


def submit_rows(spec,rows,folder,*,old=None):
    for row in rows:batch_script_index(row['command'])
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);jobs={}
    plan=artifact('PLAN',parents=dict(repair=spec['content_hash']),commands=rows)
    publish(folder/'plan.json',plan)
    for row in rows:
        args=list(row['command'])
        for task,job in jobs.items():args=[x.replace('{'+task+'}',job) for x in args]
        require(not any('{' in x or '}' in x for x in args),'Unresolved recovery dependency')
        fields=dict(parents=dict(spec=old.spec['content_hash']) if old else dict(repair=spec['content_hash']),
            task=row['task'],plan=plan['content_hash'],command=args,execution_recovery=spec['content_hash'])
        make=old.c.artifact if old else artifact
        intent=make('SUBMISSION_INTENT',**fields)
        publish(folder/(row['task']+'.intent.json'),intent)
        result=subprocess.run(args,capture_output=True,text=True,timeout=30,env=scheduler_env())
        publish(folder/(row['task']+'.response.json'),artifact('SUBMISSION_RESPONSE',
            parents=dict(intent=intent['content_hash']),task=row['task'],returncode=result.returncode,
            stdout=result.stdout,stderr=result.stderr))
        job=result.stdout.strip().split(';')[0]
        require(result.returncode==0 and re.fullmatch('[1-9][0-9]*',job),
            'Uncertain submission; inspect intent before retry: '+result.stdout+result.stderr)
        publish(folder/(row['task']+'.submitted.json'),make('SUBMITTED',job=job,**fields))
        jobs[row['task']]=job;print('SUBMITTED '+row['task']+': '+job,flush=True)
    return jobs


def cancel_pending(spec,old):
    rows=old.campaign.accepted_jobs(old.spec);known=spec['original_jobs']
    require({r['job']:r['task'] for r in rows}==known,'Submission ownership changed before repair')
    status=states(known)
    for row in rows:
        if row['task']!='report' and not (row['task'] in old.c.FITS and old.c.cell(row['task'])['reconstruction']):continue
        if status[row['job']][0]!='PENDING':continue
        fields=pending_fields(old.spec,row)
        if fields['JobState']!='PENDING':continue
        intent=artifact('CANCEL_PENDING',parents=dict(repair=spec['content_hash']),job=row['job'],task=row['task'])
        publish(Path(spec['root'])/'scheduler'/(row['job']+'.intent.json'),intent)
        subprocess.run(['scancel','--state=PENDING',row['job']],check=True,timeout=30)
        publish(Path(spec['root'])/'scheduler'/(row['job']+'.done.json'),artifact('CANCEL_REQUESTED',
            parents=dict(intent=intent['content_hash']),job=row['job'],state_filter='PENDING'))


def queue(spec,*,execute=False,authorization=None):
    old=check(spec);rows=bootstrap_commands(spec)
    preview=replacements(spec,old,preview=True)
    print(json.dumps(dict(bootstrap=rows,replacement_preview=preview,policy=spec['policy']),indent=2),flush=True)
    dry_review([*rows,*preview])
    if not execute:return rows
    require(authorization==AUTH,'Exact normalization recovery authorization required')
    claim=Path(spec['root'])/'submission'
    require(not claim.exists(),'Already launched or uncertain; inspect saved records')
    with old.campaign.submission_lock(old.spec['root']):
        # Claim first; never repeat cancellations/submissions after uncertainty.
        claim.mkdir();publish(claim/'claim.json',artifact('CLAIM',parents=dict(repair=spec['content_hash'])))
        publish(claim/'review.json',artifact('DRY_REVIEW',parents=dict(repair=spec['content_hash']),
            commands=[*rows,*preview]))
        cancel_pending(spec,old)
        return submit_rows(spec,rows,claim/'bootstrap')


def bound(spec,phase):
    job=os.environ.get('SLURM_JOB_ID','');require(re.fullmatch('[1-9][0-9]*',job),'Slurm job required')
    receipt=Path(spec['root'])/'submission/bootstrap'/(phase+'.submitted.json')
    for _ in range(12):
        if receipt.exists():break
        time.sleep(1)
    row=read(receipt);intent=read(receipt.with_name(phase+'.intent.json'))
    plan=read(receipt.parent/'plan.json')
    require(row['job']==job and row['task']==phase and row['parents']==dict(repair=spec['content_hash'])
        and row['execution_recovery']==spec['content_hash'] and row['plan']==plan['content_hash']
        and all(row[k]==intent[k] for k in ('parents','task','plan','command','execution_recovery')),
        'Recovery job binding differs')
    fields=dict(x.split('=',1) for x in subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=30).split() if '=' in x)
    gpu=phase=='normalize' and not spec.get('candidate')
    require(fields.get('JobState')=='RUNNING' and fields.get('JobName')=='lukabnorm_'+phase
        and fields.get('Comment')==spec['content_hash']+':'+phase and fields.get('Account')=='default'
        and fields.get('QOS')=='normal' and fields.get('Partition')==('gpus' if gpu else 'compute')
        and fields.get('WorkDir')==read(referenced(spec['parent_spec']))['project']
        and fields.get('NumNodes')=='1' and fields.get('NumTasks')=='1'
        and fields.get('CPUs/Task')==('4' if gpu else '1')
        and fields.get('TimeLimit')==('04:00:00' if gpu else '01:00:00'),'Recovery allocation differs')
    from decimal import Decimal
    tres=dict(x.split('=',1) for x in fields.get('AllocTRES','').split(',') if '=' in x)
    mem=re.fullmatch(r'([0-9.]+)([KMGT]?)',tres.get('mem',''))
    require(mem is not None and Decimal(mem[1])*{'':1,'M':1,'G':1024,'T':1024**2,'K':Decimal(1)/1024}[mem[2]]
        ==(64000 if gpu else 4000),'Recovery memory differs')
    if gpu:
        require(fields.get('NodeList')==spec['qualification_node'],'Use the original qualification node')
        require(tres.get('gres/gpu')=='1','Normalization replay owns one GPU slot; it performs CPU work only')
    return job


def frozen(spec,old):
    row=read(Path(spec['root'])/'normalization/complete.json')
    accepted=read(Path(spec['root'])/'submission/bootstrap/normalize.submitted.json')
    require(row['parents']==dict(repair=spec['content_hash'],qualification=old.q['content_hash']),
        'Normalization replay parents differ')
    require(row['job']==accepted['job'] and accepted['execution_recovery']==spec['content_hash'],
        'Normalization completion job differs')
    record=read(referenced(row['record']));norms=load_file(
        'hlt_classification.luka_balanced_reconstruction_hydra.normalization',referenced(spec['modules']['normalization']))
    return norms.validate(record,old.spec,expected_hash=spec['normalization_sha256'])


def normalize(spec):
    old=check(spec);job=bound(spec,'normalize');root=Path(spec['root']);node=None;owned=False
    try:
        if spec['candidate'] is not None:record=read(referenced(spec['candidate']))
        else:
            from hlt_classification.luka_balanced_reconstruction_hydra import population
            node=Path('/dev/shm')/('rlyang_lukabnorm_'+job)
            require(not node.exists() and not node.is_symlink(),'Normalization scratch already exists')
            node.mkdir(mode=0o700);owned=True
            loaded=population.load(old.spec['dataset'],Path(old.spec['root'])/'population')
            record=old.inputs.prepare_node(old.spec,node/'D000','D000',targets=True,loaded=loaded)['normalization']
        publish(root/'normalization/replayed.json',record)
        require(record['content_hash']==spec['normalization_sha256'],
            'Original qualified hash was not recovered. Full replayed statistics saved; no fits released.')
        norms=load_file('hlt_classification.luka_balanced_reconstruction_hydra.normalization',referenced(spec['modules']['normalization']))
        norms.validate(record,old.spec,expected_hash=NORMAL_HASH)
        publish(root/'normalization/frozen.json',record)
        publish(root/'normalization/complete.json',artifact('NORMALIZATION_READY',
            parents=dict(repair=spec['content_hash'],qualification=old.q['content_hash']),
            job=job,record=ref(root/'normalization/frozen.json'),node=os.uname().nodename if hasattr(os,'uname') else 'local',
            method='qualified_hash_candidate' if spec['candidate'] else 'original_source_and_qualification_node_replay'))
    finally:
        if owned:
            require(node==Path('/dev/shm')/('rlyang_lukabnorm_'+job) and node.resolve()==node
                and not node.is_symlink() and node.stat().st_uid==os.getuid(),'Unexpected scratch cleanup target')
            shutil.rmtree(node)


def replacements(spec,old,*,preview=False):
    records=old.campaign.accepted_jobs(old.spec)
    for row in records:
        require(row['job'] in spec['original_jobs'] or row.get('execution_recovery')==spec['content_hash'],
            'Unrecognized new submission owns this campaign; inspect before continuing')
    latest={r['task']:r for r in records}
    status=states(r['job'] for r in latest.values());tasks=[];active={}
    terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
    for task in old.c.FITS:
        row=latest[task];state=status[row['job']]
        if (old.artifacts.fit_root(old.spec,task)/'result.json').exists():
            r=old.artifacts.completed(old.spec,task)
            require(states([r['receipt']['job_id']])[r['receipt']['job_id']]==('COMPLETED','0:0'),'Published result exit differs')
            continue
        if state[0] not in terminal and not (preview and state[0]=='PENDING' and old.c.cell(task)['reconstruction']):
            active[task]=row['job'];continue
        require(old.c.cell(task)['reconstruction'],'A baseline/teacher failed; do not silently replace it')
        require(not list((old.artifacts.fit_root(old.spec,task)/'epochs').glob('*.json')),
            'Partial scientific history exists; preserve it for explicit authenticated resume')
        tasks.append(task)
    for task in tasks:
        teacher=old.c.cell(task)['teacher']
        require(teacher in active or (old.artifacts.fit_root(old.spec,teacher)/'result.json').exists(),
            'Required teacher is neither active nor completed')
    rows=old.campaign.commands(old.spec,[*tasks,'report'],existing=active)
    for row in rows:
        args=row['command'];cut=len(args)-4
        # Keep only true teacher edges; drop the old incidental serial chain.
        args=[x for x in args[:cut] if not x.startswith('--dependency=')]
        if row['task']=='report':
            deps=list(dict.fromkeys([*active.values(),*['{'+t+'}' for t in tasks]]))
            if deps:args+=['--dependency=afterany:'+':'.join(deps)]
        else:
            teacher=old.c.cell(row['task'])['teacher']
            if teacher in active:args+=['--dependency=afterok:'+active[teacher]]
        row['command']=args+wrapper(spec,'run',row['task'])
    return rows


def release(spec):
    old=check(spec);bound(spec,'release');frozen(spec,old)
    done=read(Path(spec['root'])/'normalization/complete.json')
    require(states([done['job']])[done['job']]==('COMPLETED','0:0'),'Normalization job did not complete')
    with old.campaign.submission_lock(old.spec['root']):
        require(not (Path(spec['root'])/'released.json').exists(),'Repair already released')
        rows=replacements(spec,old);dry_review(rows)
        attempts=sorted((PARENT/'submission').glob('attempt_*'))
        folder=PARENT/'submission'/f'attempt_{len(attempts)+1:03d}'
        # A new original-format attempt records the actual adapter command and repair hash.
        jobs=submit_rows(spec,rows,folder,old=old)
        publish(Path(spec['root'])/'released.json',artifact('RELEASED',parents=dict(repair=spec['content_hash']),
            jobs=jobs,attempt=str(folder)))
    return jobs


def install(spec,old):
    """Scoped runtime adaptation; source files and original scientific spec stay frozen."""
    record=frozen(spec,old)
    name='hlt_classification.luka_balanced_reconstruction_hydra.normalization'
    norms=load_file(name,referenced(spec['modules']['normalization']))
    package=sys.modules['hlt_classification.luka_balanced_reconstruction_hydra'];package.normalization=norms
    new=load_file('hlt_classification.luka_balanced_reconstruction_hydra._repaired_inputs',referenced(spec['modules']['inputs']))
    # Spawned CPU children must resolve real frozen module names. Only parent-
    # side moment selection/publication changes; decoding stays in old source.
    new.decoded=old.inputs.decoded
    prepare=new.prepare_node
    def inputs(*args,**kwargs):
        if kwargs.get('targets'):
            task=os.environ['LUKA_NORMALIZATION_TASK'];job=os.environ['SLURM_JOB_ID']
            kwargs.update(frozen_normalization=record,normalization_evidence=Path(spec['root'])/'comparisons'/f'{task}-{job}.json')
        return prepare(*args,**kwargs)
    old.inputs.prepare_node=inputs
    original_command=old.launch.command
    def command(*args,**kwargs):
        argv=original_command(*args,**kwargs);i=argv.index('--')+1
        argv[i]=str(Path(spec['project'])/'scripts/recover_luka_reconstruction_normalization.py')
        return argv+['--repair-spec',str(Path(spec['root'])/'repair_spec.json')]
    old.launch.command=command
    parents=old.training.fit_parents
    def fit_parents(*args,**kwargs):
        result=parents(*args,**kwargs);task=args[1];job=os.environ['SLURM_JOB_ID']
        evidence=Path(spec['root'])/'comparisons'/f'{task}-{job}.json'
        result.update(execution_recovery=spec['content_hash'],normalization_comparison=read(evidence)['content_hash'])
        return result
    old.training.fit_parents=fit_parents
    return old


def run(spec,task,*,node=None,startup_only=False):
    old=check(spec)
    require(task in old.c.FITS or task=='report','Unregistered replacement task')
    job=os.environ.get('SLURM_JOB_ID','')
    rows=[r for r in old.campaign.accepted_jobs(old.spec) if r['job']==job and r['task']==task]
    require(len(rows)==1 and rows[0].get('execution_recovery')==spec['content_hash'],
        'Job was not submitted through this repair')
    if task!='report':require(old.c.cell(task)['reconstruction'],'Only reconstruction fits may use this adapter')
    os.environ['LUKA_NORMALIZATION_TASK']=task
    old=install(spec,old)
    return old.worker.run(old.spec,task) if node is None else old.worker.rank(old.spec,task,node,startup_only=startup_only)

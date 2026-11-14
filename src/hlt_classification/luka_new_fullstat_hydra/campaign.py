"""Pinned Hydra registration and two-job submission with explicit journals."""
from pathlib import Path
import math
import os
import re
import subprocess
from hlt_classification.data.cache_contracts import load_json,sha256_file,write_immutable_json
from hlt_classification.provenance import capture_source_snapshot
from . import contracts as c,population,prepare,lineage

AUTH='AUTHORIZE_NEW_FULLSTAT_HYDRA_OFFLINE_V1'


def source(project,commit):
    project=Path(project).resolve(strict=True)
    if Path(__file__).resolve()!=project/'src/hlt_classification/luka_new_fullstat_hydra/campaign.py':raise ValueError('Import outside pinned source')
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Full commit required')
    s=capture_source_snapshot(project,require_clean=True)
    if s['git_commit']!=commit:raise ValueError('Source commit differs')
    subprocess.run(['git','-C',str(project),'merge-base','--is-ancestor',commit,'origin/main'],check=True,capture_output=True)
    tracked=set(subprocess.run(['git','-C',str(project),'ls-files'],check=True,capture_output=True,text=True).stdout.splitlines())
    required={p.relative_to(project).as_posix() for p in Path(__file__).parent.glob('*.py')}
    required|={'scripts/luka_new_fullstat_hydra.py','scripts/transfer_luka_new_fullstat.py',
        'sbatch/run_luka_new_fullstat_hydra.sh','tests/test_luka_new_fullstat_hydra.py',
        'docs/plans/LUKA_NEW_FULLSTAT_HYDRA_OFFLINE_PLAN.md','docs/contracts/LUKA_NEW_FULLSTAT_HYDRA.md'}
    if not required<=tracked:raise ValueError('Commit all new workflow files first')
    return s


def runtime():
    from hlt_classification.jetclass2_delphes.model import installed_environment
    return installed_environment()


def create(dataset,root,cache,project,commit,environment,*,gpu='l40s',available_gib,minutes=2880):
    dataset,root,cache,project,environment=[Path(p).resolve() for p in (dataset,root,cache,project,environment)]
    for i,a in enumerate((dataset,root,cache,project)):
        for b in (dataset,root,cache,project)[i+1:]:
            if a.is_relative_to(b) or b.is_relative_to(a):raise ValueError('Dataset/cache/source/results must be separate')
    if root.exists() or cache.exists():raise FileExistsError('Use new results/cache paths; do not overwrite')
    if gpu not in c.GPUS or type(minutes)!=int or not 60<=minutes<=2880:raise ValueError('Unregistered GPU/walltime')
    import sys
    if Path(sys.prefix).resolve()!=environment:raise ValueError('Run registration in the intended HLT environment')
    src=source(project,commit);f,_=population.load(dataset/'foundation')
    producer=lineage.verify(dataset,project,f['content_hash'])
    complete=load_json(dataset/'transfer_complete.json')
    manifest=load_json(dataset/'transfer_manifest.json')
    from hlt_classification.data.cache_contracts import canonical_sha256
    if (complete.get('manifest_sha256')!=canonical_sha256(manifest)
            or manifest.get('foundation_sha256')!=f['content_hash'] or complete.get('all_files_verified') is not True):
        raise ValueError('Transfer completion/foundation binding differs')
    need=int(prepare.resident_bytes(f)*1.15)+8*2**30
    if type(available_gib)!=int or available_gib*2**30<need:raise ValueError(f'Need {math.ceil(need/2**30)} GiB available for cache/results')
    memory_mb=max(32000,math.ceil((prepare.resident_bytes(f)/.65+8*2**30)/1024**2/8000)*8000)
    spec=c.artifact('SPEC',root=str(root),dataset=str(dataset),cache=str(cache),project=str(project),
        commit=commit,source=src,environment_dir=str(environment),environment=runtime(),gpu=gpu,
        foundation_sha256=f['content_hash'],foundation_producer=producer,counts=c.COUNTS,recipe=c.recipe(),node=c.node(),
        cpus=6,prepare_cpus=4,memory_mb=memory_mb,prepare_memory_mb=16000,minutes=minutes,
        prepare_minutes=1440,available_gib=available_gib,cache_resident_bytes=prepare.resident_bytes(f),
        transfer_manifest_sha256=canonical_sha256(manifest))
    root.mkdir(parents=True);write_immutable_json(root/'campaign_spec.json',spec)
    return spec


def check(spec,*,live=True):
    c.validate(spec,'SPEC')
    if spec['counts']!=c.COUNTS or spec['recipe']!=c.recipe() or spec['node']!=c.node() or spec['gpu'] not in c.GPUS:
        raise ValueError('Scientific registration differs')
    if live and source(spec['project'],spec['commit'])!=spec['source']:raise ValueError('Source drift')


def commands(spec,*,resume=False):
    root=Path(spec['root']);script=str(Path(spec['project'])/'sbatch/run_luka_new_fullstat_hydra.sh')
    result=[]
    for task in (('train',) if resume else ('prepare','train')):
        cpu=task=='prepare'
        args=['sbatch','--parsable','--partition='+('compute' if cpu else 'gpus'),'--account=default','--qos=normal',
              '--nodes=1','--ntasks=1',f'--cpus-per-task={spec["prepare_cpus"] if cpu else spec["cpus"]}',
              f'--mem={spec["prepare_memory_mb"] if cpu else spec["memory_mb"]}M',
              f'--time={spec["prepare_minutes"] if cpu else spec["minutes"]}',
              f'--job-name=lukanew_{task}_OFFLINE',f'--output={root}/{task}-%j.out',
              f'--chdir={spec["project"]}','--export=NONE','--comment='+spec['content_hash']]
        if not cpu:args+=['--gres=gpu:'+c.GPUS[spec['gpu']][0]+':1']
        if task=='train' and not resume:args+=['--dependency=afterok:{prepare}']
        args += [script,spec['project'],str(root/'campaign_spec.json'),task]
        result.append(dict(task=task,command=args))
    return result


def clean_env():
    return {k:v for k,v in os.environ.items() if not k.startswith(('SBATCH_','SLURM_'))}


def site(plan):
    output=subprocess.run(['scontrol','show','config'],env=clean_env(),check=True,capture_output=True,text=True).stdout
    if not re.search(r'(?m)^\s*ClusterName\s*=\s*hydra\s*$',output):raise PermissionError('Brown CS Hydra required')
    for row in plan:
        args=[x for x in row['command'][:-4] if not x.startswith('--dependency=')]
        args += ['--test-only','--wrap=true']
        result=subprocess.run(args,env=clean_env(),check=False,capture_output=True,text=True)
        print((result.stdout+result.stderr).strip(),flush=True)
        if result.returncode:raise PermissionError('Hydra rejected the reviewed resource request; see scheduler message above')


def queue(spec,*,execute=False,authorization=None,resume=False):
    check(spec);root=Path(spec['root']);plan=c.artifact('PLAN',spec=spec['content_hash'],commands=commands(spec,resume=resume),resume=resume)
    print(__import__('json').dumps(plan,indent=2),flush=True)
    site(plan['commands'])
    if not execute:return plan
    if authorization!=AUTH:raise PermissionError('Exact baseline authorization required')
    attempts=sorted((root/'submission').glob('attempt_*')) if (root/'submission').exists() else []
    if attempts and not resume:raise FileExistsError('Already attempted; inspect journals')
    if resume:
        if not attempts or (root/'result.json').exists():raise ValueError('No failed/incomplete baseline to resume')
        # Refuse any ambiguous prior command or live predecessor, including prepare.
        for attempt in attempts:
            for intent in attempt.glob('*.intent.json'):
                receipt=intent.with_name(intent.name.replace('.intent.json','.submitted.json'))
                if not receipt.exists():raise ValueError('Ambiguous submission; reconcile exact scheduler ID first')
                r=load_json(receipt);c.validate(r,'SUBMITTED')
                proc=subprocess.run(['sacct','-X','-n','-P','-j',r['job'],'-o','JobIDRaw,State,ExitCode'],check=True,capture_output=True,text=True)
                matches=[x.split('|') for x in proc.stdout.splitlines() if x.split('|')[0]==r['job']]
                if len(matches)!=1 or matches[0][1].split()[0] not in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}:
                    raise ValueError('Prior job not confirmed terminal')
        if not (Path(spec['cache'])/'cache.json').exists():raise ValueError('Preparation incomplete; do not queue train-only resume')
    attempt=root/'submission'/f'attempt_{len(attempts)+1:03d}';attempt.mkdir(parents=True,exist_ok=False)
    write_immutable_json(attempt/'plan.json',plan);jobs={}
    for row in plan['commands']:
        task=row['task'];args=[x.replace('{prepare}',jobs.get('prepare','UNBOUND')) for x in row['command']]
        write_immutable_json(attempt/(task+'.intent.json'),c.artifact('SUBMISSION_INTENT',spec=spec['content_hash'],plan=plan['content_hash'],command=args))
        proc=subprocess.run(args,env=clean_env(),check=False,capture_output=True,text=True)
        if proc.returncode:
            print((proc.stdout+proc.stderr).strip(),flush=True)
            raise RuntimeError('sbatch failed after intent publication; inspect journals and scheduler before retrying')
        job=proc.stdout.strip().split(';')[0]
        if not re.fullmatch('[1-9][0-9]*',job):raise RuntimeError('Ambiguous sbatch response; inspect scheduler, do not rerun')
        jobs[task]=job
        write_immutable_json(attempt/(task+'.submitted.json'),c.artifact('SUBMITTED',spec=spec['content_hash'],task=task,job=job,command=args))
        print(f'SUBMITTED {task}: {job}',flush=True)
    return jobs


def job_binding(spec,task,job):
    import time
    for _ in range(12):
        refs=list((Path(spec['root'])/'submission').glob(f'attempt_*/{task}.submitted.json'))
        for path in refs:
            r=load_json(path);c.validate(r,'SUBMITTED')
            if r['job']==job and r['task']==task and r['spec']==spec['content_hash']:return r
        time.sleep(1)
    raise ValueError('No exact submitted job binding')

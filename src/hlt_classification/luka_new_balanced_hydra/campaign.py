"""Pinned six-fit Hydra DAG, quota reservations, reviewed journals and resumes."""
from pathlib import Path
import math
import os
import re
import shutil
import subprocess
import sys
import time
from hlt_classification.data.cache_contracts import canonical_sha256, load_json, write_immutable_json
from hlt_classification.provenance import capture_source_snapshot
from hlt_classification.luka_new_fullstat_hydra import lineage
from hlt_classification.luka_new_fullstat_hydra.campaign import runtime, clean_env
from . import contracts as c, population


def source(project,commit):
    project=Path(project).resolve(strict=True)
    if Path(__file__).resolve()!=project/'src/hlt_classification/luka_new_balanced_hydra/campaign.py':
        raise ValueError('Import outside pinned source')
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Full pushed commit required')
    snapshot=capture_source_snapshot(project,require_clean=True)
    if snapshot['git_commit']!=commit:raise ValueError('Source commit differs')
    subprocess.run(['git','-C',str(project),'merge-base','--is-ancestor',commit,'origin/main'],check=True,capture_output=True)
    tracked=set(subprocess.run(['git','-C',str(project),'ls-files'],check=True,capture_output=True,text=True).stdout.splitlines())
    required={p.relative_to(project).as_posix() for p in Path(__file__).parent.glob('*.py')}
    required|={'scripts/luka_new_balanced_hydra.py','scripts/launch_luka_new_balanced_hydra.sh',
        'sbatch/run_luka_new_balanced_hydra.sh','tests/test_luka_new_balanced_hydra.py',
        'docs/plans/LUKA_NEW_BALANCED_HYDRA_PLAN.md','docs/contracts/LUKA_NEW_BALANCED_HYDRA.md'}
    if not required<=tracked:raise ValueError('Commit all campaign implementation files first')
    return snapshot


def tree_bytes(path):
    path=Path(path)
    if not path.exists():return 0
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def reserve(dataset,baseline_spec):
    """Reserve the existing baseline's bounded remaining cache/results footprint."""
    from hlt_classification.luka_new_fullstat_hydra import contracts as baseline
    path=Path(baseline_spec).resolve(strict=True);s=load_json(path);baseline.validate(s,'SPEC')
    if Path(s['dataset']).resolve()!=Path(dataset).resolve() or s['foundation_sha256']!=c.PARENT_HASH:
        raise ValueError('Natural baseline reservation must refer to this dataset')
    used=tree_bytes(s['root'])+tree_bytes(s['cache'])
    return dict(spec_path=str(path),spec_hash=s['content_hash'],
                total_budget_bytes=c.BASELINE_RESERVE_GIB*2**30,observed_bytes=used,
                remaining_bytes=max(0,c.BASELINE_RESERVE_GIB*2**30-used))


def committed_bytes(root):
    """Quota credit only for committed outputs, never abandoned attempt files.

    This is storage accounting, not artifact reuse. Full checksum and lineage
    authentication still precede every scientific continuation or teacher read.
    """
    root=Path(root).resolve();paths=set()
    def add(ref,relative=None):
        path=(Path(relative)/ref['path'] if relative else Path(ref['path'])).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size!=ref['bytes']:
            raise ValueError('Committed quota payload path/size differs')
        paths.add(path);return path
    for marker in root.glob('train_*/epochs/*.json'):
        r=load_json(marker);c.validate(r,'EPOCH_COMMIT');add(r['state'])
    for marker in root.glob('train_*/result.json'):
        r=load_json(marker);c.validate(r,'RESULT')
        for ref in r['files'].values():add(ref)
        if 'bank' in r['files']:
            path=add(r['files']['bank']);bank=load_json(path);c.validate(bank,'BANK')
            for ref in bank['payloads'].values():add(ref,path.parent)
    return sum(p.stat().st_size for p in paths)


def storage(dataset,root,baseline_spec,available_gib,*,resume=False):
    if type(available_gib)!=int or available_gib<0:raise ValueError('Declare current free quota in whole GiB')
    r=reserve(dataset,baseline_spec)
    own=committed_bytes(root) if resume else 0
    needed=max(0,c.RESULT_GIB*2**30-own)+r['remaining_bytes']
    anchor=Path(root)
    while not anchor.exists():anchor=anchor.parent
    free=shutil.disk_usage(anchor).free
    if min(free,available_gib*2**30)<needed:
        raise OSError(f'Need {math.ceil(needed/2**30)} GiB currently free, including the natural baseline reserve; do not delete its cache')
    return dict(new_results_budget_bytes=c.RESULT_GIB*2**30,natural_baseline=r,
                declared_available_gib=available_gib,filesystem_free_bytes=free,required_free_bytes=needed,
                committed_output_bytes_credited=own)


def create(dataset,root,project,commit,environment,baseline_spec,*,gpu='l40s',available_gib,minutes=2880):
    dataset,root,project,environment=[Path(p).resolve() for p in (dataset,root,project,environment)]
    for i,a in enumerate((dataset,root,project)):
        for b in (dataset,root,project)[i+1:]:
            if a.is_relative_to(b) or b.is_relative_to(a):raise ValueError('Dataset/source/results must be separate')
    if root.exists():raise FileExistsError('Use a fresh campaign root')
    if gpu not in c.GPUS or type(minutes)!=int or not 60<=minutes<=2880:raise ValueError('Unsupported GPU/walltime')
    if Path(sys.prefix).resolve()!=environment:raise ValueError('Registration must use the intended environment')
    src=source(project,commit);env=runtime()
    # Force actual imports; a partially purged installation can still have metadata.
    from weaver.nn.model.ParticleTransformer import ParticleTransformer
    import torch.nn.functional, scipy, uproot, threadpoolctl
    producer=lineage.verify(dataset,project,c.PARENT_HASH)
    complete=load_json(dataset/'transfer_complete.json');manifest=load_json(dataset/'transfer_manifest.json')
    if (complete.get('manifest_sha256')!=canonical_sha256(manifest) or manifest.get('foundation_sha256')!=c.PARENT_HASH
            or complete.get('all_files_verified') is not True):raise ValueError('Dataset transfer is not authenticated')
    quota=storage(dataset,root,baseline_spec,available_gib)
    root.mkdir(parents=True,exist_ok=False)
    pop=population.build(dataset,root/'population')
    spec=c.artifact('SPEC',root=str(root),dataset=str(dataset),project=str(project),commit=commit,source=src,
        environment_dir=str(environment),environment=env,gpu=gpu,foundation_producer=producer,
        population=pop,population_root=str(root/'population'),recipe=c.recipe(),nodes={s:c.node(s) for s in c.STAGES},
        counts=dict(c.COUNTS),cpus=6,memory_mb=192000,minutes={task:minutes for task in c.tasks()},
        preparation_workers=4,candidate_workers=3,storage=quota,tasks=c.tasks(),
        transfer_manifest_sha256=canonical_sha256(manifest))
    write_immutable_json(root/'campaign_spec.json',spec)
    return spec


def check(spec,*,live=True):
    c.validate(spec,'SPEC');c.validate(spec['population'],'POPULATION',dict(foundation=c.PARENT_HASH))
    if (spec['counts']!=c.COUNTS or spec['recipe']!=c.recipe() or spec['nodes']!={s:c.node(s) for s in c.STAGES}
            or spec['tasks']!=c.tasks() or spec['gpu'] not in c.GPUS or spec['cpus']!=6 or spec['memory_mb']!=192000
            or spec['preparation_workers']!=4 or spec['candidate_workers']!=3
            or set(spec['minutes'])!=set(c.tasks()) or any(type(m)!=int or not 60<=m<=2880 for m in spec['minutes'].values())
            or spec['population']['policy']!=c.population_policy()
            or spec['population']['counts']!=dict(c.COUNTS,unused=689)
            or spec['population']['outside_counts']!=c.OUTSIDE):
        raise ValueError('Scientific/resource registration differs')
    if Path(spec['population_root'])!=Path(spec['root'])/'population':raise ValueError('Population root differs')
    if live and source(spec['project'],spec['commit'])!=spec['source']:raise ValueError('Source drift')


def accepted_jobs(spec,*,require_unambiguous=True):
    result=[]
    for attempt in sorted((Path(spec['root'])/'submission').glob('attempt_*')):
        for intent in sorted(attempt.glob('*.intent.json')):
            saved=load_json(intent);c.validate(saved,'SUBMISSION_INTENT',dict(spec=spec['content_hash']))
            receipt=intent.with_name(intent.name.replace('.intent.json','.submitted.json'))
            if not receipt.exists():
                if require_unambiguous:raise ValueError('Ambiguous sbatch attempt; reconcile its exact scheduler ID before retry')
                continue
            r=load_json(receipt);c.validate(r,'SUBMITTED',dict(spec=spec['content_hash']))
            if (r['task']!=saved['task'] or r['command']!=saved['command'] or r['plan']!=saved['plan']
                    or r['task'] not in c.tasks() or not re.fullmatch('[1-9][0-9]*',r['job'])):
                raise ValueError('Submitted job intent differs')
            result.append(r)
    if len({r['job'] for r in result})!=len(result):raise ValueError('Duplicate scheduler job')
    return result


def terminal(job):
    out=subprocess.run(['sacct','-X','-n','-P','-j',job,'-o','JobIDRaw,State,ExitCode'],check=True,capture_output=True,text=True).stdout
    rows=[x.split('|') for x in out.splitlines() if x.split('|')[0]==job]
    if len(rows)!=1 or rows[0][1].split()[0] not in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'}:
        raise ValueError(f'Prior job {job} is not confirmed terminal; pending descendants must be handled explicitly')
    return rows[0]


def remaining(spec,resume):
    from . import artifacts
    jobs=accepted_jobs(spec)
    if jobs and not resume:raise FileExistsError('Already submitted; use explicit resume only after terminal accounting')
    complete={}
    if resume:
        if not jobs:raise ValueError('No submission to resume')
        states={r['job']:terminal(r['job']) for r in jobs}
        for stage in c.STAGES:
            if (Path(spec['root'])/('train_'+stage)/'result.json').exists():
                result=artifacts.completed(spec,stage)
                row=states[result['receipt']['job_id']]
                if row[1].split()[0]!='COMPLETED' or row[2]!='0:0':raise ValueError('Published stage did not exit successfully')
                complete[stage]=result['receipt']['content_hash']
    tasks=[t for t in c.tasks() if t.removeprefix('train_') not in complete]
    if not tasks:raise ValueError('All six stages are already published')
    return tasks,complete


def commands(spec,tasks):
    out=[];root=Path(spec['root']);script=str(Path(spec['project'])/'sbatch/run_luka_new_balanced_hydra.sh')
    for task in tasks:
        args=['sbatch','--parsable','--partition=gpus','--account=default','--qos=normal',
            '--nodes=1','--ntasks=1',f'--cpus-per-task={spec["cpus"]}',f'--mem={spec["memory_mb"]}M',
            f'--gres=gpu:{c.GPUS[spec["gpu"]][0]}:1',f'--time={spec["minutes"][task]}',
            '--job-name=lukanbal_'+task,f'--output={root}/{task}-%j.out',f'--chdir={spec["project"]}',
            '--export=NONE','--kill-on-invalid-dep=yes','--comment='+spec['content_hash']+':'+task]
        dependencies=[x for x in c.tasks()[task] if x in tasks]
        if dependencies:args+=['--dependency=afterok:'+':'.join('{'+x+'}' for x in dependencies)]
        args+=[script,spec['project'],str(root/'campaign_spec.json'),task]
        out.append(dict(task=task,dependencies=dependencies,command=args))
    return out


def site(rows):
    config=subprocess.run(['scontrol','show','config'],env=clean_env(),check=True,capture_output=True,text=True).stdout
    if not re.search(r'(?m)^\s*ClusterName\s*=\s*hydra\s*$',config):raise PermissionError('Brown CS Hydra required')
    for row in rows:
        args=[x for x in row['command'][:-4] if not x.startswith('--dependency=')]+['--test-only','--wrap=true']
        proc=subprocess.run(args,env=clean_env(),check=False,capture_output=True,text=True)
        print((proc.stdout+proc.stderr).strip(),flush=True)
        if proc.returncode:raise RuntimeError('Scheduler dry review rejected resources; no jobs submitted')


def queue(spec,*,execute=False,authorization=None,resume=False,available_gib=None):
    check(spec);root=Path(spec['root']);tasks,complete=remaining(spec,resume)
    budget=spec['storage']['declared_available_gib'] if available_gib is None else available_gib
    quota=storage(spec['dataset'],root,spec['storage']['natural_baseline']['spec_path'],budget,resume=True)
    plan=c.artifact('PLAN',parents=dict(spec=spec['content_hash']),commands=commands(spec,tasks),
                    completed=complete,resume=resume,storage=quota,scientific_fits=len(tasks),preflight_jobs=0)
    print(__import__('json').dumps(plan,indent=2),flush=True);site(plan['commands'])
    if not execute:return plan
    if authorization!=c.AUTH:raise PermissionError('Exact six-fit authorization required')
    attempts=sorted((root/'submission').glob('attempt_*'))
    attempt=root/'submission'/f'attempt_{len(attempts)+1:03d}';attempt.mkdir(parents=True,exist_ok=False)
    write_immutable_json(attempt/'plan.json',plan);jobs={}
    for row in plan['commands']:
        args=list(row['command'])
        for task,job in jobs.items():args=[x.replace('{'+task+'}',job) for x in args]
        if any('{' in x or '}' in x for x in args):raise ValueError('Unresolved job dependency')
        task=row['task']
        fields=dict(parents=dict(spec=spec['content_hash']),task=task,plan=plan['content_hash'],command=args)
        write_immutable_json(attempt/(task+'.intent.json'),c.artifact('SUBMISSION_INTENT',**fields))
        proc=subprocess.run(args,env=clean_env(),check=False,capture_output=True,text=True)
        job=proc.stdout.strip().split(';')[0]
        if proc.returncode or not re.fullmatch('[1-9][0-9]*',job):
            print(proc.stdout+proc.stderr,flush=True)
            raise RuntimeError('Uncertain submission; preserve journal and reconcile scheduler before retry')
        jobs[task]=job
        write_immutable_json(attempt/(task+'.submitted.json'),c.artifact('SUBMITTED',job=job,**fields))
        print(f'SUBMITTED {task}: {job}',flush=True)
    return jobs


def binding(spec,task,job):
    for _ in range(12):
        matches=[r for r in accepted_jobs(spec,require_unambiguous=False) if r['job']==job and r['task']==task]
        if len(matches)==1:return matches[0]
        time.sleep(1)
    raise ValueError('No exact worker submission binding')

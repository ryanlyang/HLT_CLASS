"""Pinned two-fit Hydra DAG, quota reservations, reviewed journals and resumes."""
from pathlib import Path
from contextlib import contextmanager, nullcontext
import math
import os
import re
import shutil
import subprocess
import sys
import time
from hlt_classification.data.cache_contracts import canonical_sha256, load_json, write_immutable_json, sha256_file
from hlt_classification.provenance import capture_source_snapshot
from hlt_classification.luka_new_fullstat_hydra import lineage
from hlt_classification.luka_new_fullstat_hydra.campaign import runtime, clean_env
from . import contracts as c, bundle


def locations(dataset,root,project,environment):
    dataset,root,project,environment=[Path(p).resolve() for p in (dataset,root,project,environment)]
    home=Path(c.BASE)
    if dataset!=Path(c.DATASET) or not root.is_relative_to(home/'runs') or not project.is_relative_to(home):
        raise ValueError('Use registered Hydra data/source/results')
    if not environment.is_relative_to(home) or not (environment/'bin/python').is_file():
        raise ValueError('Persistent Hydra runtime required')
    for a in (dataset,root,project):
        if environment.is_relative_to(a) or a.is_relative_to(environment):
            raise ValueError('Runtime must be separate from data/source/results')


def backend():
    import torch
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python')
    return dict(cublas=':4096:8',cudnn_v8_disabled=os.environ.get('TORCH_CUDNN_V8_API_DISABLED','unset'),
                cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32)


def donor_sources(project):
    """Pin reused population, model, sampler and target semantics to donor bytes."""
    manifest=load_json(Path(__file__).with_name('donors.json'))
    for name,digest in manifest['files'].items():
        if sha256_file(Path(project)/name)!=digest:
            raise ValueError('Frozen donor source differs: '+name)
    return manifest


def source(project,commit):
    project=Path(project).resolve(strict=True)
    if Path(__file__).resolve()!=project/'src/hlt_classification/luka_sixclass_tail_hydra/campaign.py':
        raise ValueError('Import outside pinned source')
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Full pushed commit required')
    snapshot=capture_source_snapshot(project,require_clean=True)
    if snapshot['git_commit']!=commit:raise ValueError('Source commit differs')
    subprocess.run(['git','-C',str(project),'merge-base','--is-ancestor',commit,'origin/main'],check=True,capture_output=True)
    tracked=set(subprocess.run(['git','-C',str(project),'ls-files'],check=True,capture_output=True,text=True).stdout.splitlines())
    donors=donor_sources(project)
    required={p.relative_to(project).as_posix() for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.json')}
    required|=set(donors['files'])
    required|={'scripts/luka_sixclass_tail_hydra.py','scripts/launch_luka_sixclass_tail_hydra.sh',
        'sbatch/run_luka_sixclass_tail_hydra.sh','tests/test_luka_sixclass_tail_hydra.py',
        'docs/plans/LUKA_SIXCLASS_TAIL_HYDRA_PLAN.md','docs/contracts/LUKA_SIXCLASS_TAIL_HYDRA.md',
        'docs/LUKA_SIXCLASS_TAIL_HYDRA_RUNBOOK.md','docs/LUKA_SIXCLASS_TAIL_HYDRA_VALIDATION.md',
        'scripts/queue_luka_sixclass_tail_hydra.sh','sbatch/luka_sixclass_tail_hydra_common.sh',
        'scripts/transfer_luka_sixclass_tail.py','scripts/export_luka_sixclass_tail.sh'}
    if not required<=tracked:raise ValueError('Commit all campaign implementation files first')
    return snapshot


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
            path=add(r['files']['bank']);bank=load_json(path);c.science.validate(bank,'BANK')
            for ref in bank['payloads'].values():add(ref,path.parent)
    return sum(p.stat().st_size for p in paths)


def storage(root,available_gib,*,resume=False):
    if type(available_gib)!=int or available_gib<0:raise ValueError('Declare current free quota in whole GiB')
    own=committed_bytes(root) if resume else 0
    needed=max(0,c.RESULT_GIB*2**30-own)
    anchor=Path(root)
    while not anchor.exists():anchor=anchor.parent
    free=shutil.disk_usage(anchor).free
    if min(free,available_gib*2**30)<needed:
        raise OSError(f'Need {math.ceil(needed/2**30)} GiB currently free for this campaign')
    return dict(new_results_budget_bytes=c.RESULT_GIB*2**30,
                declared_available_gib=available_gib,filesystem_free_bytes=free,required_free_bytes=needed,
                committed_output_bytes_credited=own)


def create(dataset,root,project,commit,environment,*,import_root,bundle_hash,available_gib):
    dataset,root,project,environment,import_root=[Path(p).resolve() for p in (dataset,root,project,environment,import_root)]
    locations(dataset,root,project,environment)
    paths=(dataset,root,project,environment,import_root)
    for i,a in enumerate(paths):
        for b in paths[i+1:]:
            if a.is_relative_to(b) or b.is_relative_to(a):raise ValueError('Execution/data/import paths overlap')
    if root.exists():raise FileExistsError('Use a fresh campaign root')
    if not import_root.is_relative_to(Path(c.BASE)/'imports'):raise ValueError('Use Hydra imports directory')
    if Path(sys.prefix).resolve()!=environment:raise ValueError('Use the intended Hydra runtime')
    src=source(project,commit);env=runtime()
    from weaver.nn.model.ParticleTransformer import ParticleTransformer
    import torch.nn.functional, scipy, uproot, threadpoolctl
    imported=bundle.Import(import_root,bundle_hash,verify_payloads=True)
    if imported.manifest['donor_sources']!=donor_sources(project):raise ValueError('Export/import donor semantics differ')
    # CUDA/Python builds may differ; Weaver model source and torch release may not.
    original_env=imported.spec['environment']
    if (env['versions']['torch'].split('+')[0]!=original_env['versions']['torch'].split('+')[0]
            or env['versions']['weaver-core']!=original_env['versions']['weaver-core']
            or env['weaver_python_sources']!=original_env['weaver_python_sources']):
        raise ValueError('Incompatible Hydra torch/Weaver; do not silently upgrade runtime')
    producer=lineage.verify(dataset,project,c.PARENT_HASH)
    manifest=load_json(dataset/'transfer_manifest.json');complete=load_json(dataset/'transfer_complete.json')
    if (canonical_sha256(manifest)!=c.TRANSFER_HASH or complete.get('manifest_sha256')!=c.TRANSFER_HASH
            or complete.get('all_files_verified') is not True or manifest.get('foundation_sha256')!=c.PARENT_HASH):
        raise ValueError('Raw dataset transfer differs')
    loaded=imported.authenticate_population(dataset);pop=loaded[0]
    # Authenticate bank row joins before any submission, without loading particles.
    class Rows:
        role='train';identities=loaded[4]['train_identities'];labels=loaded[4]['train_labels']
        def __len__(self):return len(self.labels)
    for stage in c.IMPORTED:
        print('Checking imported TRAIN bank:',stage,flush=True)
        imported.bank(stage,Rows())
    quota=storage(root,available_gib)
    spec=c.artifact('SPEC',parents=dict(oscar_spec=imported.spec['content_hash'],bundle=bundle_hash),
        root=str(root),dataset=str(dataset),project=str(project),commit=commit,source=src,
        environment_dir=str(environment),environment=env,gpu='l40',foundation_producer=producer,
        population=pop,population_root=str(import_root/'population'),recipe=c.recipe(),
        nodes={s:c.node(s) for s in c.STAGES},counts=dict(c.COUNTS),
        import_root=str(import_root),bundle_sha256=bundle_hash,imported_teachers=imported.manifest['teachers'],
        cpus=6,memory_mb=192000,minutes={task:2880 for task in c.tasks()},
        preparation_workers=4,candidate_workers=3,storage=quota,tasks=c.tasks(),
        transfer_manifest_sha256=c.TRANSFER_HASH,execution_site=c.SITE,backend=backend())
    root.mkdir(parents=True,exist_ok=False)
    write_immutable_json(root/'campaign_spec.json',spec)
    return spec


def check(spec,*,live=True):
    c.validate(spec,'SPEC')
    c.science.validate(spec['population'],'POPULATION',dict(foundation=c.PARENT_HASH,role_population=c.ROLE_POPULATION_HASH))
    if (spec['counts']!=c.COUNTS or spec['recipe']!=c.recipe() or spec['nodes']!={s:c.node(s) for s in c.STAGES}
            or spec['tasks']!=c.tasks() or spec['gpu']!='l40' or spec['cpus']!=6 or spec['memory_mb']!=192000
            or spec['preparation_workers']!=4 or spec['candidate_workers']!=3
            or spec['minutes']!={task:2880 for task in c.tasks()}
            or spec['population']['content_hash']!=c.POPULATION_HASH
            or spec['execution_site']!=c.SITE or spec['transfer_manifest_sha256']!=c.TRANSFER_HASH
            or spec['storage']['new_results_budget_bytes']!=c.RESULT_GIB*2**30):
        raise ValueError('Scientific/resource registration differs')
    imported=bundle.Import(spec['import_root'],spec['bundle_sha256'])
    if (spec['parents']!=dict(oscar_spec=imported.spec['content_hash'],bundle=spec['bundle_sha256'])
            or spec['population']!=imported.spec['population']
            or spec['imported_teachers']!=imported.manifest['teachers']
            or Path(spec['population_root'])!=Path(spec['import_root'])/'population'):
        raise ValueError('Original teacher/population binding differs')
    if live:
        locations(spec['dataset'],spec['root'],spec['project'],spec['environment_dir'])
        if source(spec['project'],spec['commit'])!=spec['source']:raise ValueError('Source drift')
        if imported.manifest['donor_sources']!=donor_sources(spec['project']):raise ValueError('Donor drift')
        if backend()!=spec['backend']:raise ValueError('Backend drift')
        if Path(sys.prefix).resolve()!=Path(spec['environment_dir']).resolve() or runtime()!=spec['environment']:
            raise ValueError('Registered runtime differs')


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
    if not tasks:raise ValueError('Both tail stages are already published')
    return tasks,complete


def commands(spec,tasks):
    out=[];root=Path(spec['root']);script=str(Path(spec['project'])/'sbatch/run_luka_sixclass_tail_hydra.sh')
    for task in tasks:
        args=['sbatch','--parsable','--partition=gpus','--account=default','--qos=normal',
            '--nodes=1','--ntasks=1',f'--cpus-per-task={spec["cpus"]}',f'--mem={spec["memory_mb"]}M',
            f'--gres=gpu:{c.GPUS[spec["gpu"]][0]}:1',f'--time={spec["minutes"][task]}',
            '--job-name=luka6tail_'+task,f'--output={root}/{task}-%j.out',f'--chdir={spec["project"]}',
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


@contextmanager
def submission_lock(root):
    """Serialize writers; the OS releases the lock if a launcher dies."""
    path=Path(root)/'submission'/'writer.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as handle:
        if os.name=='nt':
            import msvcrt
            if path.stat().st_size==0:handle.write(b'0');handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            if os.name=='nt':
                handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def queue(spec,*,execute=False,authorization=None,resume=False,available_gib=None):
    if execute and authorization!=c.AUTH:raise PermissionError('Exact two-fit authorization required')
    check(spec)
    with submission_lock(spec['root']) if execute else nullcontext():
        return _queue(spec,execute=execute,authorization=authorization,resume=resume,available_gib=available_gib)


def _queue(spec,*,execute=False,authorization=None,resume=False,available_gib=None):
    root=Path(spec['root']);tasks,complete=remaining(spec,resume)
    budget=spec['storage']['declared_available_gib'] if available_gib is None else available_gib
    quota=storage(root,budget,resume=True)
    plan=c.artifact('PLAN',parents=dict(spec=spec['content_hash']),commands=commands(spec,tasks),
                    completed=complete,resume=resume,storage=quota,scientific_fits=len(tasks),preflight_jobs=0)
    print(__import__('json').dumps(plan,indent=2),flush=True);site(plan['commands'])
    if not execute:return plan
    if authorization!=c.AUTH:raise PermissionError('Exact two-fit authorization required')
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

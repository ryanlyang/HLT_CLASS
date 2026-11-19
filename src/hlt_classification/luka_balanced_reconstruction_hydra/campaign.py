"""Pinned source, shared-reservation registration and durable two-phase submission."""
from pathlib import Path
import os
import re
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json,write_immutable_json,canonical_sha256,sha256_file
from hlt_classification.provenance import capture_source_snapshot
from hlt_classification.luka_new_fullstat_hydra.campaign import runtime,clean_env
from hlt_classification.luka_new_fullstat_hydra import lineage
from hlt_classification.luka_sixclass_tail_hydra.campaign import submission_lock,terminal
from . import contracts as c,population,storage,artifacts


def source(project,commit):
    project=Path(project).resolve(strict=True)
    if Path(__file__).resolve()!=project/'src/hlt_classification/luka_balanced_reconstruction_hydra/campaign.py':raise ValueError('Import outside pinned source')
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Exact pushed commit required')
    snapshot=capture_source_snapshot(project,require_clean=True)
    if snapshot['git_commit']!=commit:raise ValueError('Source commit differs')
    subprocess.run(['git','-C',str(project),'merge-base','--is-ancestor',commit,'origin/main'],check=True,capture_output=True)
    pins=load_json(Path(__file__).with_name('donors.json'))
    for name,digest in pins['files'].items():
        if sha256_file(project/name)!=digest:raise ValueError('Frozen scientific donor differs: '+name)
    required={p.relative_to(project).as_posix() for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.json')}
    required|=set(pins['files'])
    required|={'scripts/luka_balanced_reconstruction_hydra.py','scripts/launch_luka_balanced_reconstruction_hydra.sh',
        'scripts/queue_luka_balanced_reconstruction_hydra.sh','sbatch/run_luka_balanced_reconstruction_hydra.sh',
        'sbatch/luka_balanced_reconstruction_hydra_common.sh','tests/test_luka_balanced_reconstruction_hydra.py',
        'docs/plans/LUKA_BALANCED_RECONSTRUCTION_HYDRA_PLAN.md','docs/contracts/LUKA_BALANCED_RECONSTRUCTION_HYDRA.md',
        'docs/LUKA_BALANCED_RECONSTRUCTION_HYDRA_RUNBOOK.md','docs/LUKA_BALANCED_RECONSTRUCTION_HYDRA_VALIDATION.md'}
    tracked=set(subprocess.check_output(['git','-C',str(project),'ls-files'],text=True).splitlines())
    if not required<=tracked:raise ValueError('Commit all campaign files before deployment')
    return snapshot


def backend():
    import torch
    return dict(cublas=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,benchmark=torch.backends.cudnn.benchmark,
        deterministic=torch.backends.cudnn.deterministic)


def locations(dataset,root,project,environment):
    base=Path(c.BASE)
    if (dataset!=Path(c.DATASET) or not root.is_relative_to(base/'runs') or not project.is_relative_to(base)
            or not environment.is_relative_to(base) or not (environment/'bin/python').is_file()):
        raise ValueError('Use registered Hydra data/source/runtime/results')
    paths=(dataset,root,project,environment)
    for i,a in enumerate(paths):
        if any(p.is_symlink() for p in (a,*a.parents)):raise PermissionError('Execution path symlink')
        for b in paths[i+1:]:
            if a.is_relative_to(b) or b.is_relative_to(a):raise ValueError('Execution roots overlap')


def create(dataset,root,project,commit,environment,available_gib):
    from . import kernel
    dataset,root,project,environment=[Path(x).absolute() for x in (dataset,root,project,environment)]
    locations(dataset,root,project,environment)
    if root.exists():raise FileExistsError('Fresh study root required')
    if Path(sys.prefix).resolve()!=environment.resolve():raise ValueError('Use intended Hydra environment')
    kernel.configure(1,12)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('Set CUBLAS_WORKSPACE_CONFIG before Python')
    src=source(project,commit);env=runtime()
    from weaver.nn.model.ParticleTransformer import ParticleTransformer
    import torch.nn.functional,threadpoolctl,scipy,uproot
    if env['versions']['torch'].split('+')[0]!='2.5.1' or env['versions']['weaver-core']!='0.5.3':raise ValueError('Use qualified torch2.5.1/Weaver0.5.3')
    producer=lineage.verify(dataset,project,c.parent.PARENT_HASH)
    manifest=load_json(dataset/'transfer_manifest.json');complete=load_json(dataset/'transfer_complete.json')
    if (canonical_sha256(manifest)!=c.parent.TRANSFER_HASH or complete.get('manifest_sha256')!=c.parent.TRANSFER_HASH
            or complete.get('all_files_verified') is not True):raise ValueError('New dataset transfer differs')
    from hlt_classification.luka_sixclass_tail_hydra.contracts import validate as validate_tail
    other=load_json(Path(c.OTHER_ROOT)/'campaign_spec.json');validate_tail(other,'SPEC')
    if other['root']!=c.OTHER_ROOT:raise ValueError('Protected campaign root differs')
    preliminary=dict(root=str(root),protected_campaign=dict(root=c.OTHER_ROOT,spec_sha256=other['content_hash']))
    quota=storage.check(preliminary,declared_gib=available_gib)
    root.mkdir(parents=True,exist_ok=False)
    p=population.build(dataset,root/'population')
    maximum=max(p['maximum_support']['train'].values());inventory=storage.estimate(maximum)
    spec=c.artifact('SPEC',root=str(root),dataset=str(dataset),project=str(project),commit=commit,source=src,
        environment_dir=str(environment),environment=env,backend=backend(),population=p,maximum=maximum,
        recipe=c.recipe(),tasks=c.tasks(),resources=c.RESOURCES,execution_site=c.SITE,
        foundation_producer=producer,transfer_manifest_sha256=c.parent.TRANSFER_HASH,
        protected_campaign=preliminary['protected_campaign'],storage=quota,storage_policy=storage.policy(),output_inventory=inventory,
        output_allowance_bytes=c.RESULT_BYTES,reserve_bytes=c.RESERVE_BYTES)
    write_immutable_json(root/'campaign_spec.json',spec);return spec


def check(spec,*,live=True):
    c.validate(spec,'SPEC');c.validate(spec['population'],'POPULATION',dict(parent=c.parent.POPULATION_HASH))
    if spec.get('storage_policy')!=storage.policy():
        raise ValueError('Execution storage policy differs; register a fresh study with this source')
    if (spec['recipe']!=c.recipe() or spec['tasks']!=c.tasks() or spec['resources']!=c.RESOURCES
            or spec['execution_site']!=c.SITE or spec['population']['counts']!=c.COUNTS
            or spec['population']['content_hash']!=c.POPULATION_HASH
            or spec['maximum']!=max(spec['population']['maximum_support']['train'].values())
            or spec['output_allowance_bytes']!=c.RESULT_BYTES or spec['reserve_bytes']!=c.RESERVE_BYTES
            or spec['transfer_manifest_sha256']!=c.parent.TRANSFER_HASH or spec['protected_campaign']['root']!=c.OTHER_ROOT):
        raise ValueError('Scientific/resource policy differs')
    if live:
        locations(*(Path(spec[k]) for k in ('dataset','root','project','environment_dir')))
        if source(spec['project'],spec['commit'])!=spec['source']:raise ValueError('Source drift')
        if runtime()!=spec['environment'] or Path(sys.prefix).resolve()!=Path(spec['environment_dir']).resolve():raise ValueError('Environment drift')
        if backend()!=spec['backend']:raise ValueError('FP32/backend policy drift')


def accepted_jobs(spec,*,unambiguous=True):
    result=[]
    for path in sorted((Path(spec['root'])/'submission').glob('attempt_*/*.intent.json')):
        intent=load_json(path);c.validate(intent,'SUBMISSION_INTENT',dict(spec=spec['content_hash']))
        receipt=path.with_name(path.name.replace('.intent.json','.submitted.json'))
        if not receipt.exists():
            if unambiguous:raise ValueError('Ambiguous submission; reconcile exact scheduler ID before retry')
            continue
        r=load_json(receipt);c.validate(r,'SUBMITTED',dict(spec=spec['content_hash']))
        if (any(r[k]!=intent[k] for k in ('task','plan','command')) or r['task'] not in c.tasks()
                or not re.fullmatch('[1-9][0-9]*',r['job'])):raise ValueError('Submission binding differs')
        result.append(r)
    if len({r['job'] for r in result})!=len(result):raise ValueError('Duplicate scheduler IDs')
    return result


def commands(spec,tasks,existing=None):
    existing={} if existing is None else existing;rows=[]
    for task in tasks:
        gpu=task=='qualify' or task in c.FITS
        args=['sbatch','--parsable','--partition='+('gpus' if gpu else 'compute'),'--account=default','--qos=normal',
            '--nodes=1','--ntasks=1','--cpus-per-task='+str(12 if gpu else 4 if task=='prepare' else 1),
            '--mem='+str(64000 if gpu or task=='prepare' else 4000)+'M',
            '--time='+str(360 if task=='qualify' else 2880 if gpu else 240 if task=='prepare' else 60),
            '--job-name=lukabrec_'+task,'--output='+str(Path(spec['root'])/(task+'-%j.out')),
            '--chdir='+spec['project'],'--export=NONE','--kill-on-invalid-dep=yes',
            '--comment='+spec['content_hash']+':'+task]
        if gpu:args+=['--gres=gpu:geforce_gtx_1080_ti:4']
        deps=[]
        for teacher in c.tasks()[task]:
            kind='afterany' if task=='report' else 'afterok'
            if teacher in tasks:deps.append((kind,'{'+teacher+'}'))
            elif teacher in existing:deps.append((kind,existing[teacher]))
        if task in c.FITS:
            before=[t for t in tasks[:tasks.index(task)] if t in c.FITS]
            if before and before[-1] not in c.tasks()[task]:deps.append(('afterany','{'+before[-1]+'}'))
        if deps:args+=['--dependency='+','.join(kind+':'+job for kind,job in deps)]
        args += [str(Path(spec['project'])/'sbatch/run_luka_balanced_reconstruction_hydra.sh'),spec['project'],
                 str(Path(spec['root'])/'campaign_spec.json'),task]
        rows.append(dict(task=task,command=args))
    return rows


def site(rows):
    config=subprocess.run(['scontrol','show','config'],env=clean_env(),check=True,capture_output=True,text=True).stdout
    if not re.search(r'(?m)^\s*ClusterName\s*=\s*hydra\s*$',config):raise PermissionError('Brown CS Hydra required')
    for row in rows:
        args=[a for a in row['command'][:-4] if not a.startswith('--dependency=')]+['--test-only','--wrap=true']
        p=subprocess.run(args,env=clean_env(),capture_output=True,text=True)
        print((p.stdout+p.stderr).strip(),flush=True)
        if p.returncode:raise RuntimeError('Scheduler dry review failed; no live submission')


def submit(spec,tasks,*,execute,authorization=None,existing=None,resume=False):
    if execute and authorization!=c.AUTH:raise PermissionError('Exact eighteen-fit authorization required')
    check(spec)
    from contextlib import nullcontext
    with submission_lock(spec['root']) if execute else nullcontext():
        prior=accepted_jobs(spec)
        already={r['task'] for r in prior}
        if any(t in already for t in tasks) and not resume:raise FileExistsError('Task already submitted; use explicit terminal resume')
        if resume:
            for row in prior:terminal(row['job'])
        rows=commands(spec,tasks,existing)
        plan=c.artifact('PLAN',parents=dict(spec=spec['content_hash']),commands=rows,
            scientific_fits=len([t for t in tasks if t in c.FITS]),qualification='measured_four_GPU_before_science_release')
        print(__import__('json').dumps(plan,indent=2),flush=True);site(rows)
        if not execute:return plan
        attempts=sorted((Path(spec['root'])/'submission').glob('attempt_*'))
        folder=Path(spec['root'])/'submission'/f'attempt_{len(attempts)+1:03d}';folder.mkdir(exist_ok=False)
        write_immutable_json(folder/'plan.json',plan);jobs={}
        for row in rows:
            args=list(row['command'])
            for task,job in jobs.items():args=[a.replace('{'+task+'}',job) for a in args]
            if any('{' in a or '}' in a for a in args):raise ValueError('Unresolved dependency')
            fields=dict(parents=dict(spec=spec['content_hash']),task=row['task'],plan=plan['content_hash'],command=args)
            write_immutable_json(folder/(row['task']+'.intent.json'),c.artifact('SUBMISSION_INTENT',**fields))
            p=subprocess.run(args,env=clean_env(),capture_output=True,text=True);job=p.stdout.strip().split(';')[0]
            if p.returncode or not re.fullmatch('[1-9][0-9]*',job):raise RuntimeError('Uncertain sbatch result; preserve and reconcile journal: '+p.stdout+p.stderr)
            write_immutable_json(folder/(row['task']+'.submitted.json'),c.artifact('SUBMITTED',job=job,**fields))
            jobs[row['task']]=job;print('SUBMITTED',row['task']+':',job,flush=True)
        return jobs


def bootstrap(spec,*,execute=False,authorization=None):
    # Review the full authorized envelope before the bootstrap can release it.
    check(spec)
    if accepted_jobs(spec):raise FileExistsError('Already submitted; inspect records before explicit recovery')
    site(commands(spec,list(c.tasks())))
    return submit(spec,['prepare','qualify','release'],execute=execute,authorization=authorization)


def release(spec):
    from .resources import qualification
    q=qualification(spec)
    jobs=accepted_jobs(spec)
    qualify=[r for r in jobs if r['task']=='qualify']
    if not qualify or terminal(qualify[-1]['job'])[1:]!=['COMPLETED','0:0']:raise ValueError('Qualification did not exit successfully')
    return submit(spec,[*c.FITS,'report'],execute=True,authorization=c.AUTH)


def resume(spec,*,authorization,available_gib):
    if authorization!=c.AUTH:raise PermissionError('Exact continuation authorization required')
    check(spec);storage.check(spec,declared_gib=available_gib)
    prior=accepted_jobs(spec)
    if not prior:raise ValueError('No prior submission to resume')
    states={r['job']:terminal(r['job']) for r in prior}
    from . import worker
    if (Path(spec['root'])/'tasks/qualify.json').exists():
        qr=worker.task_receipt(spec,'qualify')
        if states[qr['job']][1:]!=['COMPLETED','0:0']:raise ValueError('Qualification task exit differs')
        from .resources import qualification
        qualification(spec)
        complete={}
        for task in c.FITS:
            if (artifacts.fit_root(spec,task)/'result.json').exists():
                r=artifacts.completed(spec,task)
                if states[r['receipt']['job_id']][1:]!=['COMPLETED','0:0']:raise ValueError('Published fit scheduler exit differs')
                complete[task]=r['receipt']['job_id']
        tasks=[t for t in c.FITS if t not in complete]
        if not tasks and (Path(spec['root'])/'results.json').exists():raise ValueError('All eighteen fits published; read saved results')
        return submit(spec,[*tasks,'report'],execute=True,authorization=authorization,existing=complete,resume=True)
    tasks=['qualify','release'] if (Path(spec['root'])/'tasks/prepare.json').exists() else ['prepare','qualify','release']
    return submit(spec,tasks,execute=True,authorization=authorization,resume=True)

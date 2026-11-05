"""Durable Slurm release after setup; twelve independent exact-ID fits."""
from pathlib import Path
import re
import subprocess
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag,load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger
from hlt_classification.luka_fullsim.regular_submission import clean_environment
from .contracts import TASKS,AUTHORIZATION,artifact
from . import campaign


def command(spec,task,minutes,*,cpu=False,dependencies=(),after='afterok'):
    project=spec['project_dir']; root=Path(spec['root']); site=spec['cpu_site'] if cpu else spec['execution_site']
    cmd=['sbatch','--parsable','--no-requeue','--kill-on-invalid-dep=yes','--nodes=1','--ntasks=1','--export=NONE',
        '--partition='+site['partition'],'--account='+site['account'],'--qos='+site['qos'],
        '--cpus-per-task='+str(site['cpus'] if cpu else spec['cpus']),
        '--mem='+str(site['memory_mb'] if cpu else spec['memory_mb'])+'M','--time='+_walltime(minutes),
        '--job-name=lukapt_'+task,'--comment=lukapt:'+spec['content_hash']+':'+task,
        '--chdir='+project,'--output='+str(root/(task+'-%j.out'))]
    if not cpu: cmd.append('--gres='+site['gres'])
    if dependencies: cmd.append('--dependency='+after+':'+':'.join('${JOB_'+t+'}' for t in dependencies))
    cmd += [project+'/sbatch/run_luka_particle_teacher_hydra.sh',project,str(root/'campaign_spec.json'),task]
    return dict(task_id=task,dependencies=list(dependencies),command=cmd)


def plan(spec,stage,acceptance=None):
    parents=dict(spec=spec['content_hash'])
    if stage=='bootstrap':
        rows=[command(spec,'setup',spec['setup_minutes']),command(spec,'release',spec['release_minutes'],cpu=True,dependencies=['setup'])]
    elif stage=='science':
        if acceptance is None: raise ValueError('Measured acceptance required')
        parents['acceptance']=acceptance['content_hash']
        rows=[command(spec,t,acceptance['minutes'][t]) for t in TASKS]
        rows.append(command(spec,'report',spec['report_minutes'],cpu=True,dependencies=TASKS,after='afterany'))
    else: raise ValueError('Unknown submission stage')
    return artifact('PLAN',parents=parents,stage=stage,commands=rows,authorization=AUTHORIZATION,
        science_tasks=list(TASKS),science_max_minutes=spec['fit_max_minutes'],registration=campaign.registration())


def scheduler_review(value):
    env=clean_environment()
    config=subprocess.run(['scontrol','show','config'],check=True,capture_output=True,text=True,env=env).stdout
    if not re.search(r'(?m)^\s*ClusterName\s*=\s*hydra\s*$',config): raise PermissionError('Submit on Brown CS Hydra')
    shapes=set()
    for row in value['commands']:
        cmd=row['command']; shape=tuple(x for x in cmd if x.startswith(('--partition=','--mem=','--time=','--cpus-per-task=','--gres=')))
        if shape in shapes: continue
        shapes.add(shape)
        args=[x for x in cmd[:-4] if not x.startswith('--dependency=')]
        args.insert(1,'--test-only'); args.append('--wrap=true')
        result=subprocess.run(args,env=env,capture_output=True,text=True)
        print((result.stdout+result.stderr).strip(),flush=True)
        if result.returncode: raise RuntimeError('Hydra rejected resource shape during dry review')


def accepted_jobs(spec,stage):
    folder=Path(spec['root'])/('submission_'+stage); saved=folder/'command_plan.json'
    if not saved.exists(): return {}
    acceptance=load_json(Path(spec['root'])/'setup/acceptance.json') if stage=='science' else None
    value=plan(spec,stage,acceptance)
    if load_json(saved)!=value: raise ValueError('Saved exact command plan differs')
    events,jobs=load_exact_dag_journal(folder/'submission_ledger_journal',identity=value['content_hash'],plan=value)
    ledger=folder/'submission_ledger.json'
    if ledger.exists() and load_json(ledger)!=assemble_submission_ledger(events,campaign_spec_sha256=value['content_hash']):
        raise ValueError('Submission ledger/journal mismatch')
    return jobs


def submit_stage(spec,value,*,execute):
    folder=Path(spec['root'])/('submission_'+value['stage']); folder.mkdir(exist_ok=True)
    saved=folder/'command_plan.json'; dry=folder/'dry_run_submission_ledger.json'; live=folder/'submission_ledger.json'
    if execute:
        if not saved.exists() or load_json(saved)!=value: raise PermissionError('Exact dry review required')
        if live.exists():
            jobs=accepted_jobs(spec,value['stage'])
            if set(jobs)!={r['task_id'] for r in value['commands']}: raise ValueError('Incomplete submission ledger')
            return load_json(live)
        if (folder/'live_claim').exists() or (folder/'submission_ledger_journal').exists():
            raise RuntimeError('Submission active or ambiguous/partial; preserve and inspect journals')
    write_immutable_json(saved,value)
    if not execute: scheduler_review(value)
    if execute:
        (folder/'live_claim').mkdir(exist_ok=False)
        write_immutable_json(folder/'live_claim/intent.json',artifact('SUBMIT_CLAIM',parents=dict(plan=value['content_hash']),authorization=AUTHORIZATION))
    return submit_exact_dag(identity=value['content_hash'],plan=value,output=live if execute else dry,
        canonical_dry_run=dry,execute=execute,environment=clean_environment())


def launch(spec,*,execute=False,plan_hash=None,authorization=None):
    campaign.check(spec); value=plan(spec,'bootstrap')
    if execute and (plan_hash!=value['content_hash'] or authorization!=AUTHORIZATION):
        raise PermissionError('Authorize the reviewed setup/release and twelve-fit envelope')
    result=submit_stage(spec,value,execute=execute)
    if execute: print('Setup and durable Slurm release queued. Safe to log off; release submits all twelve fits after measured acceptance.',flush=True)
    return result


def finish(spec):
    from .worker import completed_setup
    campaign.check(spec)
    bootstrap=plan(spec,'bootstrap'); folder=Path(spec['root'])/'submission_bootstrap'
    claim=load_json(folder/'live_claim/intent.json')
    if claim!=artifact('SUBMIT_CLAIM',parents=dict(plan=bootstrap['content_hash']),authorization=AUTHORIZATION):
        raise PermissionError('Original campaign authorization missing')
    if 'setup' not in accepted_jobs(spec,'bootstrap'): raise PermissionError('No accepted setup job')
    acceptance=completed_setup(spec); value=plan(spec,'science',acceptance)
    # A complete stage is idempotent. An existing partial live claim is never
    # resumed automatically: an sbatch response may have been lost.
    live=Path(spec['root'])/'submission_science/submission_ledger.json'
    if live.exists(): return submit_stage(spec,value,execute=True)
    submit_stage(spec,value,execute=False)
    return submit_stage(spec,value,execute=True)

"""Exact six-stage DAG with append-only submission journals."""
from pathlib import Path
import getpass
import subprocess
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag,load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger
from hlt_classification.luka_fullstat.submission import check_site,environment
from . import contracts as c,campaign


def render(spec):
    site=spec['execution_site'];root=Path(spec['root']);project=spec['project_dir'];commands=[]
    for stage in c.STAGES:
        task='train_'+stage
        if task not in spec['tasks']:continue
        deps=spec['tasks'][task]
        args=['sbatch','--parsable','--no-requeue','--kill-on-invalid-dep=yes','--nodes=1','--ntasks=1','--export=NONE',
            '--partition='+site['partition'],'--account='+site['account'],'--qos='+site['qos'],
            '--cpus-per-task='+str(spec['cpus']),'--mem='+str(spec['memory_mb'])+'M',
            '--time='+_walltime(spec['minutes'][task]),'--gres='+site['gres'],
            '--job-name=lukafsbal_'+task,'--comment=lukafsbal:'+spec['content_hash']+':'+task,
            '--chdir='+project,'--output='+str(root/(task+'-%j.out'))]
        if deps:args+=['--dependency=afterok:'+':'.join('${JOB_'+d+'}' for d in deps)]
        args += [project+'/sbatch/run_luka_fullstat_balanced_ladder.sh',project,str(root/'campaign_spec.json'),task]
        commands.append(dict(task_id=task,dependencies=deps,command=args))
    return c.artifact('PLAN',parents=dict(spec=spec['content_hash']),commands=commands,jobs=len(commands),
        scientific_fits=len(commands)-sum(v.get('completed_epochs')==20 for v in spec['continuations'].values()),
        preflight_jobs=0,setup_jobs=0,mode='recovery' if spec['recovery'] else 'science',authorization=c.AUTHORIZATION)


def accepted_jobs(spec,*,partial=False):
    folder=Path(spec['root'])/'submission'
    events,jobs=load_exact_dag_journal(folder/'submission_ledger_journal',identity=spec['content_hash'],plan=render(spec))
    if not set(jobs)<=set(spec['tasks']) or (not partial and set(jobs)!=set(spec['tasks'])):
        raise ValueError('Exact job bindings incomplete/different')
    saved=folder/'submission_ledger.json'
    if saved.exists():
        if load_json(saved)!=assemble_submission_ledger(events=events,campaign_spec_sha256=spec['content_hash']):
            raise ValueError('Submission ledger differs from journal')
    elif not partial:raise ValueError('Final submission ledger absent')
    return jobs


def submit(spec,*,execute=False,plan_hash=None,authorization=None):
    campaign.check(spec);plan=render(spec);folder=Path(spec['root'])/'submission';folder.mkdir(exist_ok=True)
    saved=folder/'command_plan.json';dry=folder/'dry_run_submission_ledger.json'
    if execute:
        if plan_hash!=plan['content_hash'] or authorization!=c.AUTHORIZATION:
            raise PermissionError('Exact reviewed plan hash and authorization required')
        if not saved.is_file() or load_json(saved)!=plan or not dry.is_file():raise ValueError('Canonical dry review required')
    write_immutable_json(saved,plan);check_site(plan)
    if execute:
        if spec['recovery']:
            from .recovery import claim
            claim(spec,plan)
        (folder/'live_claim').mkdir(exist_ok=False)
        write_immutable_json(folder/'live_claim/intent.json',c.artifact('CLAIM',parents=dict(plan=plan['content_hash'])))
    result=submit_exact_dag(identity=spec['content_hash'],plan=plan,
        output=folder/'submission_ledger.json' if execute else dry,canonical_dry_run=dry,
        execute=execute,environment=environment())
    return result if execute else plan


def accounting(spec,task,*,success=True):
    job=accepted_jobs(spec,partial=True)[task]
    out=subprocess.run(['sacct','-S','2026-11-01','-j',job,'-X','-n','-P',
        '-o','JobIDRaw,JobName%80,User,Account,State%40,ExitCode'],check=True,capture_output=True,text=True).stdout
    rows=[line.strip().split('|')[:6] for line in out.splitlines() if line.strip()]
    if len(rows)!=1 or rows[0][:4]!=[job,'lukafsbal_'+task,getpass.getuser(),spec['execution_site']['account']]:
        raise ValueError('Scheduler ownership/identity differs')
    state=rows[0][4].split(' by ')[0]
    if success:
        if state!='COMPLETED' or rows[0][5]!='0:0':raise ValueError('Dependency not completed successfully')
    elif state not in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'):
        raise ValueError('Recovery requires all old jobs terminal; exact pending jobs must be cancelled first')
    return rows[0]

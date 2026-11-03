"""Exactly one export followed by one student, never a new teacher fit."""
from pathlib import Path
import getpass
import subprocess
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag, load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger, build_submission_ledger
from hlt_classification.luka_fullstat.submission import check_site, environment
from .contracts import artifact, TASKS, AUTHORIZATION
from . import campaign


def render(spec):
    commands = []; site = spec['execution_site']; root = Path(spec['root']); project = spec['project_dir']
    for task,deps in TASKS.items():
        args = ['sbatch','--parsable','--no-requeue','--nodes=1','--ntasks=1','--export=NONE',
            '--partition='+site['partition'],'--account='+site['account'],'--qos='+site['qos'],
            '--cpus-per-task='+str(spec['cpus']),'--mem='+str(spec['memory_mb'])+'M',
            '--time='+_walltime(spec['minutes'][task]),'--gres='+site['gres'],
            '--job-name=lukafsmx_'+task,'--comment=lukafsmx:'+spec['content_hash']+':'+task,
            '--chdir='+project,'--output='+str(root/(task+'-%j.out'))]
        if deps: args += ['--dependency=afterok:'+':'.join('${JOB_'+d+'}' for d in deps)]
        args += [project+'/sbatch/run_luka_fullstat_multiview_export.sh',project,str(root/'campaign_spec.json'),task]
        commands.append(dict(task_id=task,dependencies=deps,command=args))
    return artifact('PLAN',parents=dict(spec=spec['content_hash']),commands=commands,mode='export_recovery',
        jobs=2,scientific_fits=1,export_jobs=1,preflight_jobs=0,authorization=AUTHORIZATION)


def accepted_jobs(spec, *, partial=False):
    folder = Path(spec['root'])/'submission'
    events,jobs = load_exact_dag_journal(folder/'submission_ledger_journal',identity=spec['content_hash'],plan=render(spec))
    if not jobs or not set(jobs) <= set(TASKS) or (not partial and set(jobs) != set(TASKS)):
        raise ValueError('Incomplete exact submission journal')
    saved = folder/'submission_ledger.json'
    if saved.exists():
        if load_json(saved) != assemble_submission_ledger(events=events,campaign_spec_sha256=spec['content_hash']):
            raise ValueError('Submission ledger differs')
    elif not partial: raise ValueError('Missing completed submission ledger')
    return jobs


def submit(spec, *, execute=False, plan_hash=None, authorization=None):
    campaign.check(spec)
    if any((Path(spec['root'])/task).exists() for task in TASKS):
        raise FileExistsError('Task outputs already exist; no implicit retry')
    plan = render(spec)
    if execute and (plan_hash != plan['content_hash'] or authorization != AUTHORIZATION):
        raise PermissionError('Exact reviewed plan/authorization required')
    folder = Path(spec['root'])/'submission'; folder.mkdir(exist_ok=True)
    saved = folder/'command_plan.json'; dry = folder/'dry_run_submission_ledger.json'
    if execute and (not saved.is_file() or load_json(saved) != plan):
        raise ValueError('Saved unchanged dry review required')
    write_immutable_json(saved,plan); check_site(plan)
    if execute:
        expected = build_submission_ledger(campaign_spec_sha256=spec['content_hash'],
            jobs={r['task_id']:'1' for r in plan['commands']},
            commands={r['task_id']:r['command'] for r in plan['commands']},dry_run=True)
        if not dry.is_file() or load_json(dry) != expected: raise ValueError('Canonical dry ledger missing')
        campaign.claim(spec,plan)
        (folder/'live_claim').mkdir(exist_ok=False)
        write_immutable_json(folder/'live_claim/intent.json',artifact('CLAIM',parents=dict(plan=plan['content_hash'])))
    result = submit_exact_dag(identity=spec['content_hash'],plan=plan,
        output=folder/'submission_ledger.json' if execute else dry,canonical_dry_run=dry,
        execute=execute,environment=environment())
    return result if execute else plan


def successful(spec, task, job):
    if accepted_jobs(spec).get(task) != job: raise ValueError('Wrong dependency job')
    out = subprocess.run(['sacct','-S','2026-11-03','-j',job,'-X','-n','-P',
        '-o','JobIDRaw,JobName%80,User,Account,State,ExitCode'],check=True,capture_output=True,text=True).stdout
    rows = [r.strip().split('|')[:6] for r in out.splitlines() if r.strip()]
    if rows != [[job,'lukafsmx_'+task,getpass.getuser(),spec['execution_site']['account'],'COMPLETED','0:0']]:
        raise ValueError('Dependency must be registered COMPLETED/0:0')

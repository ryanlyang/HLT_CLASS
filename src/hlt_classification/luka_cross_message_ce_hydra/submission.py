"""Two independent jobs; no setup dispatcher or parent scheduler mutations."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag, load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger
from hlt_classification.luka_fullsim.regular_submission import clean_environment
from hlt_classification.luka_cross_message_hydra.execution import check_site
from .contracts import TASKS, AUTHORIZATION, artifact
from . import campaign


def plan(spec):
    root = Path(spec['root']); project = spec['project_dir']; site = spec['execution_site']; rows = []
    for task in TASKS:
        cmd = ['sbatch','--parsable','--no-requeue','--nodes=1','--ntasks=1','--export=NONE',
            '--partition='+site['partition'],'--account='+site['account'],'--qos='+site['qos'],
            '--cpus-per-task='+str(spec['cpus']),'--mem='+str(spec['memory_mb'])+'M',
            '--gres='+site['gres'],'--time='+_walltime(spec['minutes'][task]),
            '--job-name=lukac0_'+task,'--comment=lukac0:'+spec['content_hash']+':'+task,
            '--chdir='+project,'--output='+str(root/(task+'-%j.out')),
            project+'/sbatch/run_luka_cross_message_ce_hydra.sh',project,str(root/'campaign_spec.json'),task]
        rows.append(dict(task_id=task, dependencies=[], command=cmd))
    return artifact('PLAN',parents=dict(spec=spec['content_hash']),commands=rows,jobs=2,authorization=AUTHORIZATION)


def accepted_jobs(spec):
    folder = Path(spec['root'])/'submission'; value = plan(spec)
    if not (folder/'command_plan.json').exists(): return {}
    if load_json(folder/'command_plan.json') != value: raise ValueError('Saved CE-only plan differs')
    events, jobs = load_exact_dag_journal(folder/'submission_ledger_journal',identity=value['content_hash'],plan=value)
    ledger = folder/'submission_ledger.json'
    if ledger.exists() and load_json(ledger) != assemble_submission_ledger(events,campaign_spec_sha256=value['content_hash']):
        raise ValueError('CE-only ledger/journal mismatch')
    return jobs


def submit(spec, *, execute=False, plan_hash=None, authorization=None):
    campaign.check(spec); value = plan(spec); folder = Path(spec['root'])/'submission'
    folder.mkdir(exist_ok=True); saved = folder/'command_plan.json'
    dry = folder/'dry_run_submission_ledger.json'; live = folder/'submission_ledger.json'
    if execute:
        if (plan_hash != value['content_hash'] or authorization != AUTHORIZATION
                or not saved.exists() or load_json(saved) != value):
            raise PermissionError('Exact dry plan and CE-only authorization required')
        if live.exists():
            if set(accepted_jobs(spec)) != set(TASKS): raise ValueError('Incomplete CE-only ledger')
            return load_json(live)
        if (folder/'live_claim').exists() or (folder/'submission_ledger_journal').exists():
            raise RuntimeError('Submission active or partial; preserve and inspect journals')
    write_immutable_json(saved,value); check_site(value)
    if execute:
        (folder/'live_claim').mkdir(exist_ok=False)
        write_immutable_json(folder/'live_claim/intent.json',artifact('SUBMIT_CLAIM',parents=dict(plan=value['content_hash'])))
    result = submit_exact_dag(identity=value['content_hash'],plan=value,output=live if execute else dry,
        canonical_dry_run=dry,execute=execute,environment=clean_environment())
    if execute: print('Both CE-only controls submitted independently. Safe to log off.',flush=True)
    return result

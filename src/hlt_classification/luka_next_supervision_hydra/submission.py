"""Two explicit reviews: setup first, then all independent interleaved Group A fits."""
from pathlib import Path
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag,load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger
from hlt_classification.luka_fullsim.regular_submission import clean_environment
from hlt_classification.luka_cross_message_hydra.execution import check_site
from hlt_classification.luka_u100_pair_hydra.submission import accounting
from .contracts import AUTHORIZATION,artifact
from . import campaign


def command(spec,task,minutes):
    site=spec['execution_site']; project=spec['project_dir']; root=Path(spec['root'])
    cmd=['sbatch','--parsable','--no-requeue','--nodes=1','--ntasks=1','--export=NONE',
        '--partition='+site['partition'],'--account='+site['account'],'--qos='+site['qos'],'--gres='+site['gres'],
        '--cpus-per-task='+str(spec['cpus']),'--mem='+str(spec['memory_mb'])+'M','--time='+_walltime(minutes),
        '--job-name=lukanx_'+task,'--comment=lukanx:'+spec['content_hash']+':'+task,
        '--chdir='+project,'--output='+str(root/(task+'-%j.out')),
        project+'/sbatch/run_luka_next_supervision_hydra.sh',project,str(root/'campaign_spec.json'),task]
    return dict(task_id=task,dependencies=[],command=cmd)


def plan(spec,stage,acceptance=None):
    parents=dict(spec=spec['content_hash'])
    if stage=='setup': commands=[command(spec,'setup',spec['setup_minutes'])]
    elif stage=='science':
        if acceptance is None: raise ValueError('Completed measured setup required')
        parents['acceptance']=acceptance['content_hash']
        commands=[command(spec,t,acceptance['minutes'][t]) for t in spec['remaining']]
    else: raise ValueError('Only setup or Group A science can be released')
    return artifact('PLAN',parents=parents,stage=stage,scope='group_A_only',commands=commands,
        authorization=AUTHORIZATION,science_dependencies='none',reused=list(spec['reused']),deferred_groups=['B','C'])


def accepted_jobs(spec,stage):
    folder=Path(spec['root'])/('submission_'+stage); saved=folder/'command_plan.json'
    if not saved.exists(): return {}
    acceptance=load_json(Path(spec['root'])/'setup/acceptance.json') if stage=='science' else None
    value=plan(spec,stage,acceptance)
    if load_json(saved)!=value: raise ValueError('Saved exact command plan differs')
    events,jobs=load_exact_dag_journal(folder/'submission_ledger_journal',identity=value['content_hash'],plan=value)
    ledger=folder/'submission_ledger.json'
    if ledger.exists() and load_json(ledger)!=assemble_submission_ledger(events,campaign_spec_sha256=value['content_hash']):
        raise ValueError('Ledger/journal mismatch')
    return jobs


def submit(spec,stage,*,execute=False,plan_hash=None,authorization=None):
    campaign.check(spec)
    if stage=='science':
        from .worker import completed_setup
        acceptance=completed_setup(spec)
        setup_job=accepted_jobs(spec,'setup').get('setup')
        if setup_job is None or accounting(setup_job)!=['COMPLETED','0:0']:
            raise PermissionError('Exact setup must be COMPLETED/0:0 before independent science release')
    else: acceptance=None
    value=plan(spec,stage,acceptance); folder=Path(spec['root'])/('submission_'+stage); folder.mkdir(exist_ok=True)
    saved=folder/'command_plan.json'; dry=folder/'dry_run_submission_ledger.json'; live=folder/'submission_ledger.json'
    if execute:
        if authorization!=AUTHORIZATION or plan_hash!=value['content_hash'] or not saved.exists() or load_json(saved)!=value:
            raise PermissionError('Authorize this exact dry-reviewed stage and hash')
        if live.exists():
            if set(accepted_jobs(spec,stage))!={r['task_id'] for r in value['commands']}: raise ValueError('Incomplete ledger')
            return load_json(live)
        if (folder/'live_claim').exists() or (folder/'submission_ledger_journal').exists():
            raise RuntimeError('Active or ambiguous partial submission; inspect exact-ID journals before recovery')
    write_immutable_json(saved,value)
    if not execute: check_site(value)
    else:
        (folder/'live_claim').mkdir(exist_ok=False)
        write_immutable_json(folder/'live_claim/intent.json',artifact('SUBMIT_CLAIM',parents=dict(plan=value['content_hash']),authorization=AUTHORIZATION))
    return submit_exact_dag(identity=value['content_hash'],plan=value,output=live if execute else dry,
        canonical_dry_run=dry,execute=execute,environment=clean_environment())

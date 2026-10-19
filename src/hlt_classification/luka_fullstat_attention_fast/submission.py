"""One performance gate, then attention-only fit and historical comparison."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag,load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger,build_submission_ledger
from hlt_classification.luka_fullstat.submission import check_site,environment
from . import campaign
from .contracts import artifact,AUTHORIZATION


def is_gate(s):
    return s.get('contract') == 'LUKA_FULLSTAT_FAST_ATTENTION_GATE/v1'


def is_resume(s):
    return s.get('contract') == 'LUKA_FULLSTAT_FAST_ATTENTION_RESUME/v1'


def render(s, g=None):
    g = s if is_gate(s) else g or load_json(s['gate_path'])
    rows = [dict(task_id='preflight',dependencies=[])] if is_gate(s) else s['tasks']
    old = g['reuse']['gate']; site=old['execution_site']; commands=[]
    for row in rows:
        name=row['task_id']; gpu=name!='aggregate'
        minutes=g['policy']['gate_minutes'] if is_gate(s) else s['train_minutes'] if gpu else 60
        if not 1 <= minutes <= 2880: raise ValueError('Walltime outside accepted Oscar envelope')
        args=['sbatch','--parsable','--no-requeue','--nodes=1','--ntasks=1','--export=NONE',
              '--partition='+(site['partition'] if gpu else 'batch'),'--account='+site['account'],
              '--cpus-per-task='+str(g['policy']['cpus'] if gpu else 1),
              '--mem='+str(g['policy']['memory_mb'] if gpu else 16384)+'M','--time='+_walltime(minutes),
              '--job-name=lukafsaf_'+name,'--comment=lukafsaf:'+s['content_hash']+':'+name,
              '--chdir='+s['project_dir'],'--output='+str(Path(s['root'])/(name+'-%j.out'))]
        if gpu: args += ['--gres='+site['gres'],'--qos='+site['qos']]
        if row['dependencies']: args += ['--dependency=afterok:'+':'.join('${JOB_'+d+'}' for d in row['dependencies'])]
        args += [s['project_dir']+'/sbatch/run_luka_fullstat_attention_fast.sh',s['project_dir'],
                 str(Path(s['root'])/('gate_spec.json' if is_gate(s) else 'resume_spec.json' if is_resume(s) else 'campaign_spec.json')),name]
        commands.append(dict(task_id=name,dependencies=row['dependencies'],command=args))
    return artifact('PLAN',parents=dict(spec=s['content_hash']),commands=commands,jobs=len(commands),
                    mode='gate' if is_gate(s) else 'science',authorization=AUTHORIZATION)


def registered_jobs(s):
    root=Path(s['root'])/'submission'; plan=render(s)
    events,jobs=load_exact_dag_journal(root/'submission_ledger_journal',identity=s['content_hash'],plan=plan)
    if (set(jobs)!={r['task_id'] for r in plan['commands']} or load_json(root/'submission_ledger.json') !=
            assemble_submission_ledger(campaign_spec_sha256=s['content_hash'],events=events)):
        raise ValueError('Missing/corrupt exact execution journal')
    return jobs


def accepted_jobs(g):
    return registered_jobs(g)


def submit(s, *, execute=False, plan_hash=None, authorization=None):
    if is_gate(s): campaign.check_gate(s)
    elif is_resume(s): campaign.check_resume(s,require_stopped=True)
    else: campaign.check_spec(s)
    plan=render(s)
    if execute and (plan_hash!=plan['content_hash'] or authorization!=AUTHORIZATION):
        raise PermissionError('Exact dry-reviewed hash and authorization required')
    root=Path(s['root'])/'submission'; root.mkdir(exist_ok=True)
    saved,dry,ledger=root/'command_plan.json',root/'dry_run_submission_ledger.json',root/'submission_ledger.json'
    if execute and (not saved.is_file() or load_json(saved)!=plan): raise ValueError('Saved dry review required')
    write_immutable_json(saved,plan); check_site(plan)
    if execute:
        expected=build_submission_ledger(campaign_spec_sha256=s['content_hash'],jobs={r['task_id']:'1' for r in plan['commands']},
                                        commands={r['task_id']:r['command'] for r in plan['commands']},dry_run=True)
        if not dry.is_file() or load_json(dry)!=expected: raise ValueError('Canonical dry ledger required')
        if not ledger.is_file():
            (root/'live_claim').mkdir(exist_ok=False)
            write_immutable_json(root/'live_claim/intent.json',artifact('CLAIM',parents=dict(spec=s['content_hash'],plan=plan['content_hash'])))
    result=submit_exact_dag(identity=s['content_hash'],plan=plan,output=ledger if execute else dry,
                           canonical_dry_run=dry,execute=execute,environment=environment())
    return result if execute else plan

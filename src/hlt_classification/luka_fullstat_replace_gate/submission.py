"""Explicit reviewed five-job DAG; no automatic retry or unrelated mutation."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag, load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger, build_submission_ledger
from hlt_classification.luka_fullstat.submission import check_site, environment as clean_environment
from . import campaign
from .contracts import artifact


def is_gate(spec):
    return spec.get('contract') == 'LUKA_FULLSTAT_REPLACE_GATE_GATE/v1'


def render(spec, gate=None):
    gate = spec if is_gate(spec) else gate or load_json(spec['gate_path'])
    rows = [dict(task_id='preflight', kind='preflight', dependencies=[])] if is_gate(spec) else gate['tasks']
    commands = []
    site, root = gate['execution_site'], Path(spec['root'])
    for row in rows:
        name, kind = row['task_id'], row['kind']
        gpu = kind != 'aggregate'
        minutes = gate['gate_minutes'] if is_gate(spec) else spec['minutes'][name]
        if type(minutes) is not int or not 1 <= minutes <= 2880:
            raise ValueError('Oscar 48-hour ceiling exceeded')
        args = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1', '--export=NONE',
            '--partition='+(site['partition'] if gpu else gate['cpu_partition']), '--account='+site['account'],
            '--cpus-per-task='+str(gate['cpus'] if gpu else 1),
            '--mem='+str(gate['memory_mb'] if gpu else 16384)+'M', '--time='+_walltime(minutes),
            '--job-name=lukafsg_'+name, '--comment=lukafsg:'+spec['content_hash']+':'+name,
            '--chdir='+spec['project_dir'], '--output='+str(root/(name+'-%j.out'))]
        if gpu:
            args += ['--gres='+site['gres'], '--qos='+site['qos']]
        if row['dependencies']:
            args.append('--dependency=afterok:'+':'.join('${JOB_'+d+'}' for d in row['dependencies']))
        args += [spec['project_dir']+'/sbatch/run_luka_fullstat_replace_gate.sh', spec['project_dir'],
                 str(root/('gate_spec.json' if is_gate(spec) else 'campaign_spec.json')), name]
        commands.append(dict(task_id=name, dependencies=row['dependencies'], command=args))
    return artifact('PLAN', parents=dict(spec=spec['content_hash']), commands=commands, jobs=len(commands),
                    mode='gate' if is_gate(spec) else 'science', authorization=campaign.AUTHORIZATION)


def accepted_jobs(gate):
    root = Path(gate['root'])/'submission'
    events, jobs = load_exact_dag_journal(root/'submission_ledger_journal',
                                        identity=gate['content_hash'], plan=render(gate))
    if (jobs.keys() != {'preflight'} or load_json(root/'submission_ledger.json') !=
            assemble_submission_ledger(campaign_spec_sha256=gate['content_hash'], events=events)):
        raise ValueError('Missing/corrupt gate journal')
    return jobs


def submit(spec, *, execute=False, plan_hash=None, authorization=None):
    campaign.check_gate(spec) if is_gate(spec) else campaign.check_spec(spec)
    plan = render(spec)
    if execute and (plan_hash != plan['content_hash'] or authorization != campaign.AUTHORIZATION):
        raise PermissionError('Exact reviewed plan and authorization required')
    root = Path(spec['root'])/'submission'
    root.mkdir(exist_ok=True)
    saved, dry, ledger = root/'command_plan.json', root/'dry_run_submission_ledger.json', root/'submission_ledger.json'
    if execute and (not saved.is_file() or load_json(saved) != plan):
        raise ValueError('Unchanged saved dry review required')
    write_immutable_json(saved, plan)
    check_site(plan)
    if execute:
        expected = build_submission_ledger(campaign_spec_sha256=spec['content_hash'],
            jobs={r['task_id']: '1' for r in plan['commands']},
            commands={r['task_id']: r['command'] for r in plan['commands']}, dry_run=True)
        if not dry.is_file() or load_json(dry) != expected:
            raise ValueError('Canonical dry ledger required')
        if not ledger.is_file():
            (root/'live_claim').mkdir(exist_ok=False)
            write_immutable_json(root/'live_claim/intent.json', artifact('SUBMISSION_CLAIM',
                parents=dict(spec=spec['content_hash'], plan=plan['content_hash'])))
    result = submit_exact_dag(identity=spec['content_hash'], plan=plan, output=ledger if execute else dry,
        canonical_dry_run=dry, execute=execute, environment=clean_environment())
    return result if execute else plan

"""Exact reviewed commands with durable single-submit journals."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag, load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger, build_submission_ledger
from hlt_classification.luka_fullsim.regular_submission import check_site, clean_environment
from . import campaign
from .contracts import artifact, AUTHORIZATION


def is_gate(spec):
    return spec.get('contract') == 'LUKA_EMBEDDING_LADDER_GATE/v1'


def render(spec):
    gate = spec if is_gate(spec) else load_json(spec['gate_path'])
    site, root = gate['execution_site'], Path(spec['root'])
    name = 'preflight' if is_gate(spec) else 'ladder'
    job_name = 'lukael_preflight' if is_gate(spec) else 'lukael_'+spec['arm'].lower()+'_ladder'
    minutes = gate['gate_minutes'] if is_gate(spec) else spec['train_minutes']
    if type(minutes) is not int or not 1 <= minutes <= 1440:
        raise ValueError('Registered walltime envelope exceeded')
    command = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1', '--export=NONE',
        '--partition='+site['partition'], '--account='+site['account'], '--qos='+site['qos'],
        '--cpus-per-task='+str(gate['cpus']), '--mem='+str(gate['memory_mb'])+'M',
        '--time='+_walltime(minutes), '--gres='+site['gres'], '--job-name='+job_name,
        '--comment=lukael:'+spec['content_hash']+':'+name, '--chdir='+spec['project_dir'],
        '--output='+str(root/(name+'-%j.out')),
        spec['project_dir']+'/sbatch/run_luka_embedding_ladder.sh', spec['project_dir'],
        str(root/('gate_spec.json' if is_gate(spec) else 'campaign_spec.json')), name]
    return artifact('PLAN', parents=dict(spec=spec['content_hash']), jobs=1,
        mode='gate' if is_gate(spec) else 'science', authorization=AUTHORIZATION,
        commands=[dict(task_id=name, dependencies=[], command=command)])


def accepted_jobs(gate):
    root = Path(gate['root'])/'submission'
    events, jobs = load_exact_dag_journal(root/'submission_ledger_journal', identity=gate['content_hash'], plan=render(gate))
    if (jobs.keys() != {'preflight'} or load_json(root/'submission_ledger.json') !=
            assemble_submission_ledger(campaign_spec_sha256=gate['content_hash'], events=events)):
        raise ValueError('Missing/corrupt exact gate submission journal')
    return jobs


def submit(spec, *, execute=False, plan_hash=None, authorization=None):
    campaign.check_gate(spec) if is_gate(spec) else campaign.check_spec(spec)
    plan = render(spec)
    if execute and (plan_hash != plan['content_hash'] or authorization != AUTHORIZATION):
        raise PermissionError('Exact reviewed plan hash and explicit authorization required')
    root = Path(spec['root'])/'submission'; root.mkdir(exist_ok=True)
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
            raise ValueError('Canonical dry-run ledger required')
        if not ledger.is_file():
            (root/'live_claim').mkdir(exist_ok=False)
            write_immutable_json(root/'live_claim/intent.json', artifact('SUBMISSION_CLAIM',
                parents=dict(spec=spec['content_hash'], plan=plan['content_hash'])))
    result = submit_exact_dag(identity=spec['content_hash'], plan=plan, output=ledger if execute else dry,
        canonical_dry_run=dry, execute=execute, environment=clean_environment())
    return result if execute else plan

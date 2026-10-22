"""Submit setup and four sequential native embedding students once, using exact afterok dependencies."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag, load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger, build_submission_ledger
from hlt_classification.luka_fullsim.regular_submission import check_site, clean_environment
from .contracts import artifact, TASKS, AUTHORIZATION
from . import campaign


def render(spec):
    root, site, project = Path(spec['root']), spec['execution_site'], spec['project_dir']
    commands = []
    for task, dependencies in TASKS.items():
        command = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1', '--export=NONE',
            '--partition='+site['partition'], '--account='+site['account'], '--qos='+site['qos'],
            '--cpus-per-task='+str(spec['cpus']), '--mem='+str(spec['memory_mb'])+'M',
            '--time='+_walltime(spec['minutes'][task]), '--gres='+site['gres'],
            '--job-name=lukaepl_'+task, '--comment=lukaepl:'+spec['content_hash']+':'+task,
            '--chdir='+project, '--output='+str(root/(task+'-%j.out'))]
        if dependencies:
            command += ['--kill-on-invalid-dep=yes', '--dependency=afterok:'+':'.join('${JOB_'+d+'}' for d in dependencies)]
        command += [project+'/sbatch/run_luka_embedding_prediction.sh', project, str(root/'campaign_spec.json'), task]
        commands.append(dict(task_id=task, dependencies=dependencies, command=command))
    return artifact('PLAN', parents=dict(spec=spec['content_hash']), jobs=5,
                    authorization=AUTHORIZATION, commands=commands)


def accepted_jobs(spec, *, partial=False):
    root = Path(spec['root'])/'submission'
    events, jobs = load_exact_dag_journal(root/'submission_ledger_journal', identity=spec['content_hash'], plan=render(spec))
    ledger = root/'submission_ledger.json'
    if not jobs or not set(jobs) <= TASKS.keys() or (not partial and set(jobs) != TASKS.keys()):
        raise ValueError('Exact submission journal incomplete')
    if ledger.is_file():
        if load_json(ledger) != assemble_submission_ledger(campaign_spec_sha256=spec['content_hash'], events=events):
            raise ValueError('Final submission ledger differs')
    elif not partial:
        raise ValueError('Missing final submission ledger')
    return jobs


def submit(spec, *, execute=False, plan_hash=None, authorization=None):
    campaign.check(spec)
    plan = render(spec)
    if execute and (plan_hash != plan['content_hash'] or authorization != AUTHORIZATION):
        raise PermissionError('Review and authorize this exact five-job plan')
    root = Path(spec['root'])/'submission'; root.mkdir(exist_ok=True)
    saved, dry, ledger = root/'command_plan.json', root/'dry_run_submission_ledger.json', root/'submission_ledger.json'
    if execute and (not saved.is_file() or load_json(saved) != plan):
        raise ValueError('Saved unchanged dry plan required')
    write_immutable_json(saved, plan); check_site(plan)
    if execute:
        expected = build_submission_ledger(campaign_spec_sha256=spec['content_hash'],
            jobs={r['task_id']: '1' for r in plan['commands']},
            commands={r['task_id']: r['command'] for r in plan['commands']}, dry_run=True)
        if not dry.is_file() or load_json(dry) != expected:
            raise ValueError('Canonical dry-run ledger required')
        if not ledger.is_file():
            (root/'live_claim').mkdir(exist_ok=False)
            write_immutable_json(root/'live_claim/intent.json', artifact('CLAIM', parents=dict(plan=plan['content_hash'])))
    result = submit_exact_dag(identity=spec['content_hash'], plan=plan, output=ledger if execute else dry,
        canonical_dry_run=dry, execute=execute, environment=clean_environment())
    return result if execute else plan

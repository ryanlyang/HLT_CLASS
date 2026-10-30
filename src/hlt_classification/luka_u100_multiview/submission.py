"""Reviewed exact-ID debug DAG with true producer dependencies only."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag, load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger
from hlt_classification.luka_fullsim.regular_submission import check_site, clean_environment
from .contracts import artifact, validate, TASKS, AUTHORIZATION
from . import campaign


def render(spec, stage):
    if stage not in ('accept', 'science'):
        raise ValueError('Unknown submission stage')
    from .acceptance import accepted
    evidence = accepted(spec) if stage == 'science' else None
    tasks = {'accept': []} if stage == 'accept' else {k: [d for d in v if d != 'accept']
                                                   for k, v in TASKS.items() if k != 'accept'}
    commands = []; site = spec['execution_site']; root = Path(spec['root']); project = spec['project_dir']
    for task, deps in tasks.items():
        minutes = spec['acceptance_minutes'] if stage == 'accept' else evidence['minutes'][task]
        command = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1', '--export=NONE',
            '--partition='+site['partition'], '--account='+site['account'], '--qos='+site['qos'],
            '--cpus-per-task='+str(spec['cpus']), '--mem='+str(spec['memory_mb'])+'M',
            '--gres='+site['gres'], '--time='+_walltime(minutes), '--job-name=lukauv_'+task,
            '--comment=lukauv:'+spec['content_hash']+':'+task, '--chdir='+project,
            '--output='+str(root/(task+'-%j.out'))]
        if deps:
            condition = 'afterany' if task == 'report' else 'afterok'
            command += ['--dependency='+condition+':'+':'.join('${JOB_'+d+'}' for d in deps)]
            if task != 'report':
                command += ['--kill-on-invalid-dep=yes']
        command += [project+'/sbatch/run_luka_u100_multiview.sh', project, str(root/'campaign_spec.json'), task]
        commands.append(dict(task_id=task, dependencies=deps, command=command))
    return artifact('PLAN', parents=dict(spec=spec['content_hash'], **({'acceptance': evidence['content_hash']} if evidence else {})),
                    stage=stage, commands=commands, jobs=len(commands), authorization=AUTHORIZATION)


def accepted_jobs(spec, stage):
    root = Path(spec['root'])/('submission_'+stage)
    plan = render(spec, stage)
    events, jobs = load_exact_dag_journal(root/'submission_ledger_journal', identity=plan['content_hash'], plan=plan)
    ledger = root/'submission_ledger.json'
    if ledger.exists() and load_json(ledger) != assemble_submission_ledger(campaign_spec_sha256=plan['content_hash'], events=events):
        raise ValueError('Exact ledger differs from journal')
    return jobs


def submit(spec, stage, *, execute=False, plan_hash=None, authorization=None):
    campaign.check(spec); plan = render(spec, stage)
    if execute and (plan_hash != plan['content_hash'] or authorization != AUTHORIZATION):
        raise PermissionError('Exact reviewed plan hash and authorization required')
    folder = Path(spec['root'])/('submission_'+stage); folder.mkdir(exist_ok=True)
    saved, dry, ledger = (folder/n for n in ('command_plan.json', 'dry_run_submission_ledger.json', 'submission_ledger.json'))
    if execute and (not saved.is_file() or load_json(saved) != plan):
        raise ValueError('Saved unchanged dry review required')
    write_immutable_json(saved, plan); check_site(plan)
    if execute and not ledger.exists():
        # No blind retry after a crash between sbatch and journal publication.
        (folder/'live_claim').mkdir(exist_ok=False)
        write_immutable_json(folder/'live_claim/intent.json', artifact('CLAIM', parents=dict(plan=plan['content_hash'])))
    result = submit_exact_dag(identity=plan['content_hash'], plan=plan, output=ledger if execute else dry,
        canonical_dry_run=dry, execute=execute, environment=clean_environment())
    return result if execute else plan

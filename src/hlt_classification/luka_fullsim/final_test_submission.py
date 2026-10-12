"""Two-job exact dry/live final-test submission; no training jobs or retries."""
from pathlib import Path

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger
from . import final_test_campaign as c
from .science_submission import check_site, clean_environment


def plan(path):
    execution = c.authenticate(path)[0]
    root, project, r = Path(execution['root']), execution['project_dir'], execution['resources']
    commands = []
    for task in ('evaluate', 'report'):
        gpu = task == 'evaluate'
        deps = [] if gpu else ['evaluate']
        args = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1', '--export=NONE',
            '--partition='+r['site']['partition'], '--account='+r['site']['account'], '--qos='+r['site']['qos'],
            '--cpus-per-task='+str(r['cpus'] if gpu else 1),
            '--mem='+str(r['memory_mb'] if gpu else r['report_memory_mb'])+'M',
            '--time='+_walltime(r['gpu_minutes'] if gpu else r['report_minutes']),
            '--job-name=luka_test_'+task, '--comment=luka_test:'+execution['content_hash']+':'+task,
            '--chdir='+project, '--output='+str(root/(task+'-%j.out'))]
        if gpu:
            args.append('--gres='+r['site']['gres'])
        else:
            args.append('--dependency=afterok:${JOB_evaluate}')
        args += [project+'/sbatch/run_luka_fullsim_final_test.sh', project, str(root/'execution_lock.json'), task]
        commands.append(dict(task_id=task, dependencies=deps, command=args))
    return c.artifact('PLAN', parents=dict(execution=execution['content_hash'],
        finalists=execution['parents']['finalists']), commands=commands, jobs=2,
        authorization=c.AUTHORIZATION, final_test_accessed=False)


def submit(path, *, execute=False, plan_hash=None, authorization=None):
    value = plan(path)
    execution = load_json(path)
    if execute and (plan_hash != value['content_hash'] or authorization != c.AUTHORIZATION):
        raise PermissionError('Exact final-test plan hash and explicit authorization required')
    root = Path(execution['root'])/'submission'
    root.mkdir(exist_ok=True)
    saved, dry, ledger = root/'command_plan.json', root/'dry_run_submission_ledger.json', root/'submission_ledger.json'
    if execute and (not saved.is_file() or load_json(saved) != value):
        raise ValueError('Review and save the complete dry plan before execution')
    write_immutable_json(saved, value)
    if ledger.is_file() and execute:
        return submit_exact_dag(identity=execution['content_hash'], plan=value, output=ledger,
            canonical_dry_run=dry, execute=True, environment=clean_environment())
    if ledger.is_file():
        return value  # Dry inspection of an already submitted plan never submits.
    if Path(execution['claim_dir']).exists():
        raise PermissionError('Test already claimed; preserve existing evidence, no new submission')
    check_site(value)
    if not execute:
        submit_exact_dag(identity=execution['content_hash'], plan=value, output=dry, canonical_dry_run=dry, execute=False)
        return value
    expected = build_submission_ledger(campaign_spec_sha256=execution['content_hash'],
        jobs={r['task_id']: '1' for r in value['commands']},
        commands={r['task_id']: r['command'] for r in value['commands']}, dry_run=True)
    if load_json(dry) != expected:
        raise ValueError('Exact final-test dry ledger differs')
    claim = root/'live_claim'
    try:
        claim.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise PermissionError('Submission active/interrupted; preserve journals, no duplicate retry') from exc
    write_immutable_json(claim/'intent.json', c.artifact('SUBMISSION_CLAIM',
        parents=dict(execution=execution['content_hash'], plan=value['content_hash']), final_test_accessed=False))
    return submit_exact_dag(identity=execution['content_hash'], plan=value, output=ledger,
        canonical_dry_run=dry, execute=True, environment=clean_environment())

"""Seven measured replacement jobs and a report; never resubmit completed fits."""
from pathlib import Path
import os
import time
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag, load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger
from hlt_classification.luka_fullsim.regular_submission import clean_environment
from hlt_classification.luka_particle_teacher_hydra.submission import scheduler_review
from . import core


def plan(spec):
    rows = []; root = Path(spec['root']); project = spec['project_dir']
    for task in (*core.RESTART, 'report_pascal'):
        cpu = task == 'report_pascal'
        minutes = 60 if cpu else spec['minutes'][task]
        if type(minutes) is not int or not 0 < minutes <= 2880: raise ValueError('Invalid measured walltime')
        args = ['sbatch', '--parsable', '--no-requeue', '--kill-on-invalid-dep=yes', '--nodes=1', '--ntasks=1', '--export=NONE',
                '--partition=' + ('compute' if cpu else 'gpus'), '--account=default', '--qos=normal',
                '--cpus-per-task=' + ('1' if cpu else '12'), '--mem=' + ('4000M' if cpu else '64000M'),
                '--time=' + _walltime(minutes), '--job-name=lukarsf_' + task,
                '--comment=lukarsf:' + spec['content_hash'] + ':' + task, '--chdir=' + project,
                '--output=' + str(root / (task + '-%j.out'))]
        dependencies = list(core.RESTART) if cpu else []
        if cpu: args.append('--dependency=afterany:' + ':'.join('${JOB_' + t + '}' for t in dependencies))
        else: args.append('--gres=gpu:geforce_gtx_1080_ti:4')
        args += [project + '/sbatch/run_luka_stability_storage_recovery.sh', project, str(root / 'recovery_spec.json'), task]
        rows.append(dict(task_id=task, dependencies=dependencies, command=args))
    return core.artifact('PLAN', parents=dict(spec=spec['content_hash']), commands=rows, authorization=core.AUTHORIZATION)


def accepted_jobs(spec):
    value = plan(spec); folder = Path(spec['root']) / 'submission'
    if not (folder / 'command_plan.json').exists(): return {}
    if load_json(folder / 'command_plan.json') != value: raise ValueError('Submission plan differs')
    events, jobs = load_exact_dag_journal(folder / 'submission_ledger_journal', identity=value['content_hash'], plan=value)
    if (folder / 'submission_ledger.json').exists():
        if load_json(folder / 'submission_ledger.json') != assemble_submission_ledger(events, campaign_spec_sha256=value['content_hash']):
            raise ValueError('Submission journal and ledger differ')
    return jobs


def require_job(spec, task):
    if task not in (*core.RESTART, 'report_pascal'): raise ValueError('Unregistered recovery task')
    job = os.environ.get('SLURM_JOB_ID', '')
    for _ in range(12):
        if job.isdigit() and accepted_jobs(spec).get(task) == job: return job
        time.sleep(5)
    raise PermissionError('Not the exact accepted recovery job')


def launch(spec, *, execute=False, plan_hash=None, authorization=None):
    p, _, runtime = core.check(spec); value = plan(spec)
    if execute and (plan_hash != value['content_hash'] or authorization != core.AUTHORIZATION):
        raise PermissionError('Authorize the exact seven replacements and report')
    folder = Path(spec['root']) / 'submission'; folder.mkdir(exist_ok=True)
    saved = folder / 'command_plan.json'; dry = folder / 'dry_run_submission_ledger.json'; live = folder / 'submission_ledger.json'
    if live.exists():
        if set(accepted_jobs(spec)) != set(core.RESTART) | {'report_pascal'}: raise ValueError('Incomplete live ledger')
        return load_json(live)
    if (folder / 'live_claim.json').exists() or (folder / 'submission_ledger_journal').exists():
        raise RuntimeError('Ambiguous partial submission; inspect journals, do not retry blindly')
    if core.evidence(p, runtime) != spec['failures']: raise ValueError('Old failure evidence changed')
    if runtime('worker').completed_setup(p, 'pascal')['content_hash'] != spec['parents']['acceptance']:
        raise ValueError('Original production qualification changed')
    if execute and (not saved.exists() or load_json(saved) != value or not dry.exists()):
        raise PermissionError('Full scheduler dry review required first')
    write_immutable_json(saved, value)
    if not execute: scheduler_review(value)
    else:
        # One common claim across new recovery roots; never create two replacements
        # for an old failed ID merely by choosing another output directory.
        lock = Path(p['root']).parent / ('stability_storage_' + p['content_hash'] + '.json')
        intent = core.artifact('SUBMIT_CLAIM', parents=dict(spec=spec['content_hash'], plan=value['content_hash']), root=spec['root'])
        if lock.exists() and load_json(lock) != intent: raise RuntimeError('Another storage recovery already owns these failed jobs')
        write_immutable_json(lock, intent); write_immutable_json(folder / 'live_claim.json', intent)
    return submit_exact_dag(identity=value['content_hash'], plan=value, output=live if execute else dry,
                            canonical_dry_run=dry, execute=execute, environment=clean_environment())

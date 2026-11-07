"""Exact setup then measured two-fit release, with only the real teacher dependency."""
from pathlib import Path
import time
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag, load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger
from hlt_classification.luka_fullsim.regular_submission import check_site, clean_environment
from hlt_classification.luka_u100_pair_correction.submission import accounting, within_envelope
from .contracts import TEACHERS, STUDENTS, FITS, AUTHORIZATION, artifact
from . import campaign, resources


def render(spec, stage, *, acceptance=None):
    if stage not in ('setup', 'science', 'envelope'):
        raise ValueError('Unknown stage')
    tasks = {'setup': []} if stage == 'setup' else {
        **{t: [] for t in TEACHERS}, **{s: [t] for s, t in STUDENTS.items()}, 'report': list(FITS)}
    if stage == 'science':
        acceptance = acceptance or resources.accepted(spec)
    commands = []; root = Path(spec['root']); project = spec['project_dir']; site = spec['execution_site']
    for task, deps in tasks.items():
        minutes = spec['setup_minutes'] if task == 'setup' else spec['report_minutes'] if task == 'report' else (
            spec['fit_max_minutes'] if stage == 'envelope' else acceptance['minutes'][task])
        command = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1', '--export=NONE',
            '--partition='+site['partition'], '--account='+site['account'], '--qos='+site['qos'],
            '--cpus-per-task='+str(spec['cpus']), '--mem='+str(spec['memory_mb'])+'M', '--gres='+site['gres'],
            '--time='+_walltime(minutes), '--job-name=lukasc5_'+task,
            '--comment=lukasc5:'+spec['content_hash']+':'+task, '--chdir='+project,
            '--output='+str(root/(task+'-%j.out'))]
        if deps:
            mode = 'afterany' if task == 'report' else 'afterok'
            command += ['--dependency='+mode+':'+':'.join('${JOB_'+d+'}' for d in deps)]
            if task in STUDENTS:
                command += ['--kill-on-invalid-dep=yes']
        command += [project+'/sbatch/run_luka_supcon5.sh', project, str(root/'campaign_spec.json'), task]
        commands.append(dict(task_id=task, dependencies=deps, command=command))
    return artifact('PLAN', parents=dict(spec=spec['content_hash'], **(
        {'acceptance': acceptance['content_hash']} if stage == 'science' else {})),
        stage=stage, commands=commands, jobs=len(commands), authorization=AUTHORIZATION)


def submit(spec, plan, *, execute):
    folder = Path(spec['root'])/('submission_'+plan['stage']); folder.mkdir(exist_ok=True)
    saved, dry, live = (folder/n for n in ('command_plan.json', 'dry_run_submission_ledger.json', 'submission_ledger.json'))
    if execute and (not saved.exists() or load_json(saved) != plan):
        raise ValueError('Saved exact dry review required')
    write_immutable_json(saved, plan)
    if execute and not live.exists():
        (folder/'live_claim').mkdir(exist_ok=False)
        write_immutable_json(folder/'live_claim/intent.json', artifact('CLAIM', parents=dict(plan=plan['content_hash'])))
    return submit_exact_dag(identity=plan['content_hash'], plan=plan, output=live if execute else dry,
        canonical_dry_run=dry, execute=execute, environment=clean_environment())


def accepted_jobs(spec, stage):
    folder = Path(spec['root'])/('submission_'+stage); path = folder/'command_plan.json'
    if not path.exists():
        return {}
    plan = load_json(path)
    if plan != render(spec, stage):
        raise ValueError('Accepted plan differs')
    events, jobs = load_exact_dag_journal(folder/'submission_ledger_journal', identity=plan['content_hash'], plan=plan)
    if (folder/'submission_ledger.json').exists() and load_json(folder/'submission_ledger.json') != assemble_submission_ledger(
            campaign_spec_sha256=plan['content_hash'], events=events):
        raise ValueError('Ledger/journal mismatch')
    return jobs


def review(spec):
    campaign.check(spec)
    folder = Path(spec['root'])/'submission'; folder.mkdir(exist_ok=True)
    setup, envelope = render(spec, 'setup'), render(spec, 'envelope')
    check_site(setup); check_site(envelope)
    value = artifact('REVIEW', parents=dict(spec=spec['content_hash']), setup=setup, envelope=envelope,
        automatic_science=True, scientific_fits=2, support_jobs=2, authorization=AUTHORIZATION,
        changes_allowed_after_measurement='walltime_decrease_only')
    write_immutable_json(folder/'review.json', value)
    submit(spec, setup, execute=False); submit(spec, envelope, execute=False)
    return value


def dispatch(spec, *, plan_hash, authorization, poll_seconds=30):
    campaign.check(spec)
    value = load_json(Path(spec['root'])/'submission/review.json')
    if authorization != AUTHORIZATION or plan_hash != value['content_hash'] or review(spec) != value:
        raise PermissionError('Exact reviewed campaign authorization required')
    claim = Path(spec['root'])/'submission/dispatcher_claim'; claim.mkdir(exist_ok=False)
    write_immutable_json(claim/'intent.json', artifact('DISPATCH', parents=dict(review=plan_hash)))
    submit(spec, value['setup'], execute=True)
    job = accepted_jobs(spec, 'setup')['setup']
    print(f'Setup {job} submitted. Detached dispatcher will release one teacher, one student and report.', flush=True)
    while True:
        record = accounting(job)
        if record == ['COMPLETED', '0:0']:
            break
        if record and record[0].split()[0] not in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED'):
            raise RuntimeError(f'Setup {job} ended {record}; science not released')
        print(f'Waiting for setup {job}: {record}', flush=True); time.sleep(poll_seconds)
    from .worker import completed_setup
    completed_setup(spec); campaign.check(spec)
    plan = render(spec, 'science')
    if not within_envelope(value['envelope'], plan):
        raise PermissionError('Measured plan exceeds authorization')
    check_site(plan); submit(spec, plan, execute=False)
    return submit(spec, plan, execute=True)

"""One setup, then automatic measured submission of twenty fits in two lanes."""
from pathlib import Path
import re
import subprocess
import time
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag, load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger
from hlt_classification.luka_fullsim.regular_submission import check_site, clean_environment
from .contracts import ARMS, AUTHORIZATION, artifact
from . import campaign, resources


def render(spec, stage, *, acceptance=None):
    if stage not in ('setup', 'science', 'envelope'):
        raise ValueError('Unregistered submission stage')
    arms = list(ARMS)
    tasks = {'setup': []} if stage == 'setup' else {
        a: ([] if i < 2 else [arms[i-2]]) for i, a in enumerate(arms)}
    if stage != 'setup':
        tasks['report'] = arms
    if stage == 'science':
        acceptance = acceptance or resources.accepted(spec)
    commands = []; site = spec['execution_site']; root = Path(spec['root']); project = spec['project_dir']
    for task, deps in tasks.items():
        minutes = (spec['setup_minutes'] if task == 'setup' else spec['report_minutes'] if task == 'report'
                   else spec['fit_max_minutes'] if stage == 'envelope' else acceptance['minutes'][task])
        args = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1', '--export=NONE',
            '--partition='+site['partition'], '--account='+site['account'], '--qos='+site['qos'],
            '--cpus-per-task='+str(spec['cpus']), '--mem='+str(spec['memory_mb'])+'M',
            '--gres='+site['gres'], '--time='+_walltime(minutes), '--job-name=lukaoc_'+task,
            '--comment=lukaoc:'+spec['content_hash']+':'+task, '--chdir='+project,
            '--output='+str(root/(task+'-%j.out'))]
        if deps:
            # Ordering only, never a scientific teacher/data dependency. Failed
            # student fits do not prevent independent later students from running.
            args += ['--dependency=afterany:'+':'.join('${JOB_'+d+'}' for d in deps)]
        args += [project+'/sbatch/run_luka_u100_contrastive.sh', project, str(root/'campaign_spec.json'), task]
        commands.append(dict(task_id=task, dependencies=deps, command=args))
    return artifact('PLAN', parents=dict(spec=spec['content_hash'], **(
        {'acceptance': acceptance['content_hash']} if stage == 'science' else {})),
        stage=stage, commands=commands, jobs=len(commands), authorization=AUTHORIZATION)


def review(spec):
    campaign.check(spec)
    folder = Path(spec['root'])/'submission'; folder.mkdir(exist_ok=True)
    setup, envelope = render(spec, 'setup'), render(spec, 'envelope')
    check_site(setup); check_site(envelope)
    value = artifact('REVIEW', parents=dict(spec=spec['content_hash']), setup=setup, envelope=envelope,
        automatic_science=True, scientific_fits=20, support_jobs=2,
        changes_allowed_after_measurement='walltime_may_decrease_within_envelope_only',
        maximum_concurrency=2, authorization=AUTHORIZATION)
    write_immutable_json(folder/'review.json', value)
    _submit(spec, setup, execute=False)
    # Full scientific command graph is saved/reviewed even before measurements.
    write_immutable_json(folder/'science_envelope.json', envelope)
    return value


def within_envelope(envelope, measured):
    if len(envelope['commands']) != len(measured['commands']):
        return False
    for a, b in zip(envelope['commands'], measured['commands']):
        if a['task_id'] != b['task_id'] or a['dependencies'] != b['dependencies']:
            return False
        if [v for v in a['command'] if not v.startswith('--time=')] != [v for v in b['command'] if not v.startswith('--time=')]:
            return False
        def seconds(command):
            values = [v.split('=', 1)[1] for v in command if v.startswith('--time=')]
            if len(values) != 1:
                raise ValueError('Exactly one walltime required')
            value = values[0]; days = 0
            if '-' in value:
                days, value = value.split('-', 1); days = int(days)
            h, m, s = map(int, value.split(':'))
            return days*86400+h*3600+m*60+s
        if not 0 < seconds(b['command']) <= seconds(a['command']):
            return False
    return True


def _submit(spec, plan, *, execute):
    folder = Path(spec['root'])/('submission_'+plan['stage']); folder.mkdir(exist_ok=True)
    saved, dry, live = (folder/n for n in ('command_plan.json', 'dry_run_submission_ledger.json', 'submission_ledger.json'))
    if execute and (not saved.exists() or load_json(saved) != plan):
        raise ValueError('Exact saved dry review required')
    write_immutable_json(saved, plan)
    if execute and not live.exists():
        (folder/'live_claim').mkdir(exist_ok=False)
        write_immutable_json(folder/'live_claim/intent.json', artifact('SUBMIT_CLAIM', parents=dict(plan=plan['content_hash'])))
    return submit_exact_dag(identity=plan['content_hash'], plan=plan, output=live if execute else dry,
        canonical_dry_run=dry, execute=execute, environment=clean_environment())


def accepted_jobs(spec, stage):
    folder = Path(spec['root'])/('submission_'+stage)
    path = folder/'command_plan.json'
    if not path.exists():
        return {}
    plan = load_json(path)
    if plan != render(spec, stage):
        raise ValueError('Saved command plan drift')
    events, jobs = load_exact_dag_journal(folder/'submission_ledger_journal', identity=plan['content_hash'], plan=plan)
    if (folder/'submission_ledger.json').exists() and load_json(folder/'submission_ledger.json') != assemble_submission_ledger(
            campaign_spec_sha256=plan['content_hash'], events=events):
        raise ValueError('Ledger/journal mismatch')
    return jobs


def accounting(job):
    if not re.fullmatch('[1-9][0-9]*', job):
        raise ValueError('Not an exact Slurm job ID')
    result = subprocess.run(['sacct', '-X', '-n', '-P', '-j', job, '-o', 'JobIDRaw,State,ExitCode'],
                            check=True, capture_output=True, text=True, env=clean_environment())
    rows = [r.split('|') for r in result.stdout.splitlines() if r.split('|')[0] == job]
    if len(rows) > 1:
        raise ValueError('Ambiguous exact accounting record')
    return rows[0][1:3] if rows else None


def dispatch(spec, *, plan_hash, authorization, poll_seconds=30):
    """Run under nohup on the submit host; no GPU used while waiting for setup."""
    campaign.check(spec)
    path = Path(spec['root'])/'submission/review.json'; value = load_json(path)
    if value != review(spec) or plan_hash != value['content_hash'] or authorization != AUTHORIZATION:
        raise PermissionError('Full reviewed automatic campaign authorization required')
    claim = Path(spec['root'])/'submission/dispatcher_claim'; claim.mkdir(exist_ok=False)
    write_immutable_json(claim/'intent.json', artifact('DISPATCH_CLAIM', parents=dict(review=value['content_hash'])))
    _submit(spec, value['setup'], execute=True)
    job = accepted_jobs(spec, 'setup')['setup']
    print(f'Setup submitted as {job}. Automatic measured science submission is active; safe to log off.', flush=True)
    while True:
        record = accounting(job)
        if record and record[0] == 'COMPLETED' and record[1] == '0:0':
            break
        if record and record[0].split()[0] not in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED'):
            raise RuntimeError(f'Shared setup {job} ended {record}; no scientific jobs submitted. Preserve diagnostics.')
        print(f'Waiting for exact setup {job}: {record or "accounting pending"}', flush=True)
        time.sleep(poll_seconds)
    from .worker import completed_setup
    completed_setup(spec)
    campaign.check(spec)
    plan = render(spec, 'science')
    if not within_envelope(value['envelope'], plan):
        raise ValueError('Measured plan exceeds reviewed scientific envelope')
    check_site(plan)
    _submit(spec, plan, execute=False)
    print('Measured complete scientific dry plan:', plan, flush=True)
    result = _submit(spec, plan, execute=True)
    print('All twenty students and the report are queued. No submit-host process is needed now.', flush=True)
    return result

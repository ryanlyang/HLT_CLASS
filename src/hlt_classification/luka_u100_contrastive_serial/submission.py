"""Pending-only exact-ID transition, followed by one automatic afterany chain."""
from pathlib import Path
import getpass
import re
import subprocess
import time
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag, load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger
from hlt_classification.luka_fullsim.regular_submission import check_site, clean_environment
from .contracts import ARMS, ORDER, AUTHORIZATION, artifact, validate
from . import campaign, source

TERMINAL = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def terminal(record):
    return record is not None and record['state'] in TERMINAL


def accounting(jobs):
    if not jobs or any(not re.fullmatch('[1-9][0-9]*', str(j)) for j in jobs.values()):
        raise ValueError('Exact nonempty campaign job IDs required')
    if len(set(jobs.values())) != len(jobs):
        raise ValueError('Duplicate campaign job ID')
    by_id = {str(j): task for task, j in jobs.items()}
    text = subprocess.run(['sacct', '-X', '-n', '-P', '-S', '2026-10-01', '-j', ','.join(by_id),
        '-o', 'JobIDRaw,JobName%80,User%80,State%40,ExitCode'], check=True,
        capture_output=True, text=True, env=clean_environment()).stdout
    result = {}
    for line in text.splitlines():
        fields = line.strip().split('|')
        if not fields or fields[0] not in by_id:
            continue
        j, name, user, state, exitcode = fields[:5]
        if j in result or name != 'lukaoc_'+by_id[j] or user != getpass.getuser():
            raise ValueError('Ambiguous or differently owned original job')
        result[j] = dict(state=state.split()[0], exit_code=exitcode, name=name, user=user)
    return result


def validate_completed(proof, states):
    if set(proof['jobs']) != {*ARMS, 'report'} or set(proof['setup_jobs']) != {'setup'}:
        raise ValueError('Incomplete original accepted job graph')
    for task, row in {'setup': {'job_id': proof['setup_jobs']['setup']}, **proof['completed']}.items():
        j = str(row['job_id']); state = states.get(j)
        expected = proof['setup_jobs'].get(task, proof['jobs'].get(task))
        if str(expected) != j or state is None or (state['state'], state['exit_code']) != ('COMPLETED', '0:0'):
            raise ValueError('Reusable task lacks exact COMPLETED/0:0 accounting: '+task)
    for task, j in proof['jobs'].items():
        if task != 'report' and states.get(str(j), {}).get('state') == 'COMPLETED' and task not in proof['completed']:
            raise ValueError('Completed job lacks authenticated receipt: '+task)


def render(root, project, site, cpus, memory_mb, remaining, minutes, identity):
    commands = []
    for i, task in enumerate([*remaining, 'report']):
        dependencies = ([remaining[-1]] if remaining else []) if task == 'report' else ([] if i == 0 else [remaining[i-1]])
        args = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1', '--export=NONE',
            '--partition='+site['partition'], '--account='+site['account'], '--qos='+site['qos'],
            '--cpus-per-task='+str(cpus), '--mem='+str(memory_mb)+'M', '--gres='+site['gres'],
            '--time='+_walltime(120 if task == 'report' else minutes[task]),
            '--job-name=lukaos_'+task, '--comment=lukaos:'+identity+':'+task,
            '--chdir='+project, '--output='+str(Path(root)/(task+'-%j.out'))]
        if dependencies:
            args += ['--dependency=afterany:'+':'.join('${JOB_'+d+'}' for d in dependencies)]
        args += [project+'/sbatch/run_luka_u100_contrastive_serial.sh', project,
                 str(Path(root)/'campaign_spec.json'), task]
        commands.append(dict(task_id=task, dependencies=dependencies, command=args))
    return artifact('PLAN', parents=dict(execution=identity), commands=commands, jobs=len(commands),
                    maximum_concurrency=1, authorization=AUTHORIZATION)


def plan(spec):
    return render(spec['root'], spec['project_dir'], spec['execution_site'], spec['cpus'], spec['memory_mb'],
                  spec['remaining'], spec['parent']['acceptance']['minutes'], spec['content_hash'])


def review(request):
    old = campaign.check_request(request)
    remaining = [a for a in ORDER if a not in request['parent']['completed']]
    envelope = render(request['root'], request['project_dir'], old['execution_site'], old['cpus'], old['memory_mb'],
                      remaining, request['parent']['acceptance']['minutes'], request['content_hash'])
    check_site(envelope)
    value = artifact('REVIEW', parents=dict(request=request['content_hash']), envelope=envelope,
        cancel_pending_exact_ids=list(request['parent']['jobs'].values()),
        preserve_running=True, reuse_completed=True, restart_partial_in_fresh_root=True,
        allowed_final_changes='remove_newly_completed_rows_relink_single_lane_bind_final_spec',
        authorization=AUTHORIZATION)
    write_immutable_json(Path(request['root'])/'review.json', value)
    return value


def submit(spec, *, execute):
    p = plan(spec); folder = Path(spec['root'])/'submission'; folder.mkdir(exist_ok=True)
    saved, dry, live = (folder/n for n in ('command_plan.json', 'dry_run_submission_ledger.json', 'submission_ledger.json'))
    if execute and (not saved.exists() or load_json(saved) != p):
        raise ValueError('Exact final dry plan required')
    write_immutable_json(saved, p)
    if execute and not live.exists():
        (folder/'live_claim').mkdir(exist_ok=False)
        write_immutable_json(folder/'live_claim/intent.json', artifact('SUBMIT_CLAIM', parents=dict(plan=p['content_hash'])))
    return submit_exact_dag(identity=p['content_hash'], plan=p, output=live if execute else dry,
        canonical_dry_run=dry, execute=execute, environment=clean_environment())


def accepted_jobs(spec):
    folder = Path(spec['root'])/'submission'; p = load_json(folder/'command_plan.json')
    if p != plan(spec):
        raise ValueError('Saved command plan drift')
    events, jobs = load_exact_dag_journal(folder/'submission_ledger_journal', identity=p['content_hash'], plan=p)
    if (folder/'submission_ledger.json').exists() and load_json(folder/'submission_ledger.json') != assemble_submission_ledger(
            campaign_spec_sha256=p['content_hash'], events=events):
        raise ValueError('Ledger/journal mismatch')
    return jobs


def cancellation_command(jobs):
    ids = [str(j) for j in jobs.values()]
    if not ids or len(set(ids)) != len(ids) or any(not re.fullmatch('[1-9][0-9]*', j) for j in ids):
        raise ValueError('Only exact nonempty campaign IDs may be cancelled')
    # Filtering is done atomically by Slurm: a pending->running race cannot kill a running fit.
    return ['scancel', '--quiet', '--me', '--state=PENDING', *ids]


def dispatch(request, *, plan_hash, authorization):
    if authorization != AUTHORIZATION:
        raise PermissionError('Explicit reviewed transition authorization required')
    value = review(request)
    if value['content_hash'] != plan_hash:
        raise ValueError('Reviewed transition hash differs')
    root = Path(request['root']); claim = root/'transition_claim'; claim.mkdir(exist_ok=False)
    proof = request['parent']; jobs = proof['jobs'] | proof['setup_jobs']
    states = accounting(jobs); validate_completed(proof, states)
    command = cancellation_command(proof['jobs'])
    write_immutable_json(claim/'intent.json', artifact('TRANSITION_CLAIM', parents=dict(review=plan_hash),
        command=command, states=states, never_cancel_running=True))
    subprocess.run(command, check=True, env=clean_environment())
    print('Pending old campaign jobs cancelled by exact ID. Running fits are preserved; waiting for their publication.', flush=True)
    while True:
        states = accounting(jobs)
        if all(terminal(states.get(str(j))) for j in jobs.values()):
            break
        live = {task: states.get(str(j)) for task, j in jobs.items() if not terminal(states.get(str(j)))}
        print('Waiting for original allocations/accounting:', live, flush=True)
        time.sleep(30)
    print('Authenticating completed original fits and targets; no export or preflight.', flush=True)
    final_proof = source.inspect_parent(proof['spec']['path'])
    spec = campaign.finalize(request, final_proof, states)
    p = plan(spec); check_site(p)
    # Final rows are exactly the reviewed order minus additional verified completions.
    reviewed = [x['task_id'] for x in value['envelope']['commands'] if x['task_id'] != 'report']
    if spec['remaining'] != [a for a in reviewed if a not in final_proof['completed']]:
        raise ValueError('Final jobs exceed reviewed transition')
    submit(spec, execute=False)
    print('Final single-lane scientific dry plan:', p, flush=True)
    result = submit(spec, execute=True)
    print('All remaining fits and the report are queued in one debug lane. Safe to log off.', flush=True)
    return result

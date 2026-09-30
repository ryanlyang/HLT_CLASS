"""One reviewed, disconnect-safe handoff from missing pilot to bounded bulk."""
from contextlib import contextmanager
from pathlib import Path
import time

from . import campaign as c, recovery as r, submission as sub, output
from .contracts import artifact, validate, checked, ref, write, safe, load_json


@contextmanager
def controller_lock(root):
    import fcntl  # Tigris; kernel releases the lock even after a disconnect/crash.
    path = safe(root, 'locks/recovery_controller.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PermissionError('Another recovery controller is active; do not launch twice') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def ledger(study, attempt, commands):
    path = safe(study['root'], f'attempts/{attempt["name"]}/submission_ledger.json')
    if not path.exists():
        return None
    value = load_json(path)
    validate(value, 'LEDGER', parents={'attempt': attempt['content_hash']}, test=False)
    if (value['dry_run'] is not False or value['plan'] != commands['content_hash']
            or set(value['jobs']) != {row['task'] for row in commands['commands']}
            or any(sub.submitted_identity(study, attempt, task) != job for task, job in value['jobs'].items())):
        raise ValueError('Recovery ledger/journal differs')
    return value


def submit_once(study, attempt):
    path = safe(study['root'], f'attempts/{attempt["name"]}/attempt_spec.json')
    _, saved, commands = sub.get_plan(path)
    if saved != attempt:
        raise ValueError('Attempt bytes changed')
    existing = ledger(study, attempt, commands)
    if existing is not None:
        print(f'CMS2JC2-RECOVERY following_existing_jobs={existing["jobs"]}', flush=True)
        return existing
    # sub.submit refuses any partial submission journal; no blind sbatch retry.
    print(f'CMS2JC2-RECOVERY submitting={attempt["name"]} plan={commands["content_hash"]}', flush=True)
    return sub.submit(path, execute=True, plan_hash=commands['content_hash'], phrase=commands['authorization_phrase'])


def wait_pilot(study, attempt, submitted, *, sleep=time.sleep, monotonic=time.monotonic):
    job = submitted['jobs']['generate']
    expected = {f'{job}_{i}' for i in range(len(attempt['shards']))}
    deadline = monotonic()+r.POLICY['max_wait_days']*86400
    previous = None
    while monotonic() < deadline:
        rows = sub.accounting([job])
        if any(row['name'] != 'c2jp_generate' for row in rows):
            raise PermissionError('Unexpected pilot job identity')
        bad = [row for row in rows if row['state'] in sub.TERMINAL
               and (row['state'] != 'COMPLETED' or row['exit_code'] != '0:0')]
        if bad:
            raise RuntimeError(f'Recovery pilot failed; no bulk submitted: {bad}')
        state = [(row['job'], row['state']) for row in rows]
        if state != previous:
            print(f'CMS2JC2-RECOVERY pilot_status={state}', flush=True)
            previous = state
        done = {row['job'] for row in rows if row['state'] == 'COMPLETED' and row['exit_code'] == '0:0'}
        if expected <= done:
            sub.require_terminal(study)
            receipts = output.completed(study)
            if not set(attempt['shards']) <= set(receipts):
                raise ValueError('Completed scheduler job lacks authenticated pilot receipt')
            return
        print('CMS2JC2-RECOVERY waiting_for_missing_pilot; no duplicate jobs', flush=True)
        sleep(r.POLICY['poll_seconds'])
    raise TimeoutError('Recovery wait exceeded 14 days; jobs preserved, no bulk submitted')


def continuation(study, plan):
    path = safe(study['root'], 'recovery/continuation.json')
    if not path.exists():
        return None
    value = load_json(path)
    validate(value, 'RECOVERY_CONTINUATION', parents={'recovery_plan': plan['content_hash']}, test=False)
    original, attempt, commands = sub.get_plan(checked(value['attempt']))
    if (original != study or attempt['kind'] != 'bulk'
            or attempt.get('execution_repair') != plan['execution_repair']
            or attempt['shards'] != [s['shard_id'] for s in study['shards'] if not s['pilot']]
            or commands != load_json(checked(value['plan']))):
        raise ValueError('Continuation execution differs')
    c.require_admission(study, attempt)
    c.test_lock(study)
    return attempt


def run(study_path, *, plan_hash, authorize_test=False):
    if not authorize_test:
        raise PermissionError('Explicit sealed-test materialization permission required')
    study = load_json(study_path)
    with controller_lock(study['root']):
        repair, initial, plan = r.read_plan(study)
        if plan['content_hash'] != plan_hash:
            raise PermissionError('Exact reviewed recovery plan hash required')
        r.validate_execution(study, plan['execution_repair'])
        # A durable continuation permits following already-submitted bulk, not
        # repeating the initial pilot or declaring a running attempt terminal.
        bulk = continuation(study, plan)
        if bulk is None:
            allowed = {checked(row['attempt']).resolve() for row in repair['original_attempts']}
            allowed.add(checked(plan['initial_attempt']).resolve())
            present = {p.resolve() for p in safe(study['root'], 'attempts').glob('*/attempt_spec.json')}
            if present != allowed:
                raise PermissionError('Unjournaled/foreign attempt exists; inspect before continuation')
            submitted = submit_once(study, initial)
            wait_pilot(study, initial, submitted)
            r.validate_execution(study, plan['execution_repair'])
            bulk = c.advance(study, authorize_test=True, execution_repair=plan['execution_repair'])
            directory = safe(study['root'], f'attempts/{bulk["name"]}')
            value = artifact('RECOVERY_CONTINUATION', parents={'recovery_plan': plan['content_hash']},
                attempt=ref(directory/'attempt_spec.json'), plan=ref(directory/'command_plan.json'))
            write(safe(study['root'], 'recovery/continuation.json'), value)
            continuation(study, plan)  # authenticate measured resources before submitting
        result = submit_once(study, bulk)
        handoff = artifact('RECOVERY_HANDOFF', parents={'recovery_plan': plan['content_hash']},
            ledger=ref(safe(study['root'], f'attempts/{bulk["name"]}/submission_ledger.json')),
            jobs=result['jobs'], dataset_complete=False, no_more_operator_handoff_required=True)
        write(safe(study['root'], 'recovery/handoff.json'), handoff)
        print(f'CMS2JC2-RECOVERY bulk_and_finalizer_queued={result["jobs"]}; '
              'complete dataset requires dataset_manifest.json', flush=True)
        return handoff

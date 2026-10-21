"""Scheduling-only handover; scientific code is loaded from its original worktree.

Never edit the measured source or gate. The old controller must be stopped on
its own host, and no science directory may exist. A new reviewed authorization
retires only that process and reuses the original successful preflight.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import socket
import subprocess
import sys
from types import SimpleNamespace

DONOR = 'fd687181a60c1557f98d7498f577e26d6dbb35a7'
ORDER = ('w50', 'w500', 'w1000', 'w0', 'w100', 'w10', 'w2')
AUTHORIZATION = 'AUTHORIZE_FULLSTAT_EMBEDDING_ORDER_V2'
ATTEMPT = 'ordered_followup_v2'
FAILED_DONOR = 'b356267787a1ce543db6d327781008315608b12c'
PIDFD_ERROR = 'Linux Python with pidfd required; no unsafe PID fallback'
FILES = ('scripts/order_luka_fullstat_embedding_sweep.py',
         'scripts/queue_luka_fullstat_embedding_order.sh',
         'src/hlt_classification/operations/fullstat_embedding_order.py')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def artifact(kind, *, version=2, **fields):
    value = dict(contract=f'LUKA_FULLSTAT_EMBEDDING_ORDER_{kind}/v{version}',
                 schema_version=version, final_test_accessed=False, **fields)
    return dict(value, content_hash=digest(value))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def git(project, *args):
    return subprocess.run(['git', '-C', str(project), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def source_identity(project, commit):
    project = Path(project).resolve()
    if not re.fullmatch('[0-9a-f]{40}', commit) or git(project, 'rev-parse', 'HEAD') != commit:
        raise PermissionError('Ordering tool requires the exact pushed commit')
    if git(project, 'status', '--porcelain', '--untracked-files=all'):
        raise PermissionError('Ordering tool worktree must be clean')
    git(project, 'merge-base', '--is-ancestor', commit, 'origin/main')
    if Path(__file__).resolve() != project / FILES[2]:
        raise PermissionError('Ordering module is not executing from the pinned worktree')
    hashes = {}
    for name in FILES:
        git(project, 'ls-files', '--error-unmatch', name)
        hashes[name] = hashlib.sha256((project/name).read_bytes()).hexdigest()
    return dict(commit=commit, project=str(project), files=hashes)


def bootstrap(gate_path):
    """Only the unchanged scientific donor supplies validation and workers."""
    gate = read(gate_path)
    original = Path(gate['project_dir']).resolve()
    if (gate['source_commit'] != DONOR or git(original, 'rev-parse', 'HEAD') != DONOR
            or git(original, 'status', '--porcelain', '--untracked-files=all')):
        raise PermissionError('Original measured source is not the clean fd687181 worktree')
    if 'hlt_classification' in sys.modules:
        raise PermissionError('Start a fresh Python process; do not mix runtime worktrees')
    sys.path.insert(0, str(original/'src'))
    prefix = 'hlt_classification.luka_fullstat_embedding_sweep.'
    modules = {name: importlib.import_module(prefix+name)
               for name in ('campaign', 'followup', 'submission', 'contracts')}
    if Path(modules['campaign'].__file__).resolve() != original/'src/hlt_classification/luka_fullstat_embedding_sweep/campaign.py':
        raise PermissionError('Wrong scientific module imported')
    modules['exact'] = importlib.import_module('hlt_classification.scouting.hcwdl_exact_dag_submission')
    modules['cache'] = importlib.import_module('hlt_classification.data.cache_contracts')
    modules['ledger'] = importlib.import_module('hlt_classification.scouting.hcwdl_recovery')
    return gate, SimpleNamespace(**modules)


def controller_identity(pid, host, gate, *, proc=Path('/proc')):
    if socket.gethostname().split('.')[0] != host or type(pid) is not int or pid <= 1:
        raise PermissionError('Inspect the exact controller on its original login host')
    directory = proc/str(pid)
    stat = (directory/'stat').read_text().rsplit(')', 1)[1].split()
    uid = directory.stat().st_uid
    argv = [v.decode() for v in (directory/'cmdline').read_bytes().split(b'\0') if v]
    cli = gate['project_dir']+'/scripts/luka_fullstat_embedding_sweep.py'
    expected = [cli, 'followup', '--spec', str(Path(gate['root'])/'gate_spec.json'),
                '--execute', '--plan-hash']
    if (uid != os.getuid() or stat[0] != 'T' or len(argv) != 13
            or argv[1:4] != ['-u', '-B', '-s'] or argv[4:10] != expected
            or argv[11:] != ['--authorization', 'AUTHORIZE_FULLSTAT_EMBEDDING_SWEEP_FOLLOWUP_V1']):
        raise PermissionError('Expected this user\'s STOPPED original followup process; no signal sent')
    # argv: python -u -B -s CLI followup --spec PATH --execute --plan-hash HASH --authorization AUTH
    return dict(pid=pid, host=host, uid=uid, start_ticks=stat[19], argv=argv)


def old_binding(gate, backend):
    b = backend
    b.campaign.check_gate(gate)
    b.followup.no_science(gate)
    root = Path(gate['root'])
    review = b.followup.plan(gate)
    if read(root/'followup/plan.json') != review:
        raise ValueError('Original authorized followup plan differs')
    intent = b.contracts.artifact('FOLLOWUP_AUTHORIZATION', parents=dict(plan=review['content_hash']),
                                source_commit=gate['source_commit'], gate=gate['content_hash'])
    if read(root/'followup/authorization.json') != intent:
        raise ValueError('Original followup authorization differs')
    jobs = b.submission.accepted_jobs(gate)
    ledger = read(root/'submission/submission_ledger.json')
    armed = b.contracts.artifact('FOLLOWUP_ARMED', parents=dict(
        authorization=intent['content_hash'], gate_ledger=ledger['content_hash']),
        preflight_job=jobs['preflight'])
    if read(root/'followup/armed.json') != armed:
        raise ValueError('Original armed controller/gate journal differs')
    return review, armed


def ordered_plan(original):
    """Add scheduling edges only; preserve every worker/resource/scientific arg."""
    rows = {r['task_id']: r for r in original['commands']}
    if (len(original['commands']) != 8 or len(rows) != 8
            or set(rows) != {*ORDER, 'aggregate'} or original['mode'] != 'science'):
        raise ValueError('Expected exactly the seven existing fits and aggregate')
    result = []
    for index, name in enumerate(ORDER):
        row = deepcopy(rows[name])
        if row['dependencies'] or any(a.startswith('--dependency=') for a in row['command']):
            raise ValueError('Original scientific arms must be independent')
        if index:
            previous = ORDER[index-1]
            row['dependencies'] = [previous]
            row['command'].insert(-4, '--dependency=afterany:${JOB_'+previous+'}')
        result.append(row)
    aggregate = deepcopy(rows['aggregate'])
    if set(aggregate['dependencies']) != set(ORDER):
        raise ValueError('Aggregate must still require every fit')
    expected = '--dependency=afterok:'+':'.join('${JOB_'+d+'}' for d in aggregate['dependencies'])
    if [a for a in aggregate['command'] if a.startswith('--dependency=')] != [expected]:
        raise ValueError('Aggregate successful-fit dependency changed')
    result.append(aggregate)
    return artifact('PLAN', parents=dict(original_plan=original['content_hash']),
                    commands=result, jobs=8, mode='science', order=list(ORDER),
                    scientific_dependencies_unchanged=True, scheduling_dependency='afterany')


def prior_failure(gate, process, original, armed):
    """Only the known pre-signal v1 portability failure permits a fresh v2 claim."""
    directory = Path(gate['root'])/'ordered_followup_v1'
    if not directory.exists() and not directory.is_symlink():
        return None
    allowed = {'plan.json', 'authorization.json', 'failure.json', 'live_claim'}
    if (directory.is_symlink() or not directory.is_dir()
            or {p.name for p in directory.iterdir()} != allowed
            or any(p.is_symlink() for p in directory.iterdir())
            or not (directory/'live_claim').is_dir()
            or any((directory/'live_claim').iterdir())):
        raise PermissionError('Prior attempt is not the isolated pre-signal pidfd failure')
    plan = read(directory/'plan.json')
    if (plan.get('content_hash') != digest({k:v for k,v in plan.items() if k != 'content_hash'})
            or plan.get('contract') != 'LUKA_FULLSTAT_EMBEDDING_ORDER_AUTHORIZATION_PLAN/v1'
            or plan.get('schema_version') != 1 or plan.get('final_test_accessed') is not False
            or plan.get('tool_source', {}).get('commit') != FAILED_DONOR
            or plan.get('parents') != dict(gate=gate['content_hash'],
                original_followup=original['content_hash'], original_armed=armed['content_hash'])
            or plan.get('controller') != process or plan.get('preflight_job') != armed['preflight_job']
            or plan.get('order') != list(ORDER)):
        raise PermissionError('Prior failed authorization/process binding differs')
    expected_intent = artifact('AUTHORIZATION', version=1,
        parents=dict(plan=plan['content_hash']), controller=process)
    expected_failure = artifact('FAILURE', version=1, parents=dict(plan=plan['content_hash']),
        error_type='RuntimeError', error=PIDFD_ERROR, automatic_retry=False)
    if (read(directory/'authorization.json') != expected_intent
            or read(directory/'failure.json') != expected_failure):
        raise PermissionError('Only the exact pre-signal missing-pidfd failure is recoverable')
    return dict(directory=str(directory), plan=plan['content_hash'],
                failure=expected_failure['content_hash'], authorization=expected_intent['content_hash'])


def pidfd_api():
    """Keep kernel PID-reuse protection even if Python was built without wrappers."""
    if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
        return SimpleNamespace(name='python_pidfd', open=os.pidfd_open, send=signal.pidfd_send_signal)
    # Oscar uses the native 64-bit x86 ABI. Never guess syscall numbers for
    # other ABIs. Linux v5.14 arch/x86/entry/syscalls/syscall_64.tbl: 434/424.
    import ctypes
    if (sys.platform != 'linux' or platform.machine() != 'x86_64'
            or ctypes.sizeof(ctypes.c_void_p) != 8 or ctypes.sizeof(ctypes.c_long) != 8):
        raise RuntimeError('No native pidfd wrappers; syscall adapter requires Linux x86_64 LP64')
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long

    def call(number, *args):
        result = syscall(ctypes.c_long(number), *args)
        if result == -1:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        return int(result)

    def open_pid(pid):
        return call(434, ctypes.c_int(pid), ctypes.c_uint(0))

    def send(fd, sig):
        call(424, ctypes.c_int(fd), ctypes.c_int(sig), ctypes.c_void_p(None), ctypes.c_uint(0))

    return SimpleNamespace(name='linux_x86_64_syscall_pidfd', open=open_pid, send=send)


def probe_pidfd():
    """Read-only capability probe during dry review; signal 0 delivers no signal."""
    import select
    if not hasattr(select, 'poll'):
        raise RuntimeError('Linux poll support required')
    api = pidfd_api()
    fd = api.open(os.getpid())
    try:
        api.send(fd, 0)
    finally:
        os.close(fd)
    return api.name


def review(gate, backend, tool_source, pid, host):
    capability = probe_pidfd()  # fail before any permanent claim or retirement
    original, armed = old_binding(gate, backend)
    process = controller_identity(pid, host, gate)
    if process['argv'][10] != original['content_hash']:
        raise PermissionError('Stopped controller has a different authorization hash')
    previous = prior_failure(gate, process, original, armed)
    ceiling = ordered_plan(original['science_ceiling_plan'])
    return artifact('AUTHORIZATION_PLAN', parents=dict(gate=gate['content_hash'],
        original_followup=original['content_hash'], original_armed=armed['content_hash']),
        tool_source=tool_source, controller=process, preflight_job=armed['preflight_job'],
        pidfd_backend=capability, prior_failed_attempt=previous,
        science_ceiling_plan=ceiling, order=list(ORDER), jobs=8,
        max_train_minutes=2880, authorization=AUTHORIZATION,
        action='retire_stopped_submit_controller_then_wait_for_original_gate',
        cancel_preflight=False, automatic_retry=False)


def retire_controller(expected, gate):
    """pidfd prevents PID reuse; never signal Slurm or the shell parent."""
    api = pidfd_api()
    fd = api.open(expected['pid'])
    try:
        if controller_identity(expected['pid'], expected['host'], gate) != expected:
            raise PermissionError('Controller identity changed; no signal sent')
        api.send(fd, signal.SIGTERM)
        try:
            api.send(fd, signal.SIGCONT)
        except ProcessLookupError:
            pass  # TERM already exited it; the pidfd is still the same process.
        # A stopped process first needs CONT to deliver its pending TERM.
        import select
        poll = select.poll()
        poll.register(fd, select.POLLIN)
        if not poll.poll(10000):
            raise RuntimeError('Controller has not exited; science remains blocked')
    finally:
        os.close(fd)


def run(gate, backend, tool_source, *, pid, host, execute=False, plan_hash=None,
        authorization=None, recheck_source=None):
    b = backend
    root = Path(gate['root'])
    directory = root/ATTEMPT
    reviewed = review(gate, b, tool_source, pid, host)
    directory.mkdir(exist_ok=True)
    if not execute:
        b.submission.check_site(reviewed['science_ceiling_plan'])
        b.cache.write_immutable_json(directory/'plan.json', reviewed)
        return reviewed
    if (plan_hash != reviewed['content_hash'] or authorization != AUTHORIZATION
            or read(directory/'plan.json') != reviewed):
        raise PermissionError('Exact saved order-plan hash and explicit authorization required')
    (directory/'live_claim').mkdir(exist_ok=False)
    b.cache.write_immutable_json(directory/'authorization.json', artifact('AUTHORIZATION',
        parents=dict(plan=reviewed['content_hash']), controller=reviewed['controller']))
    try:
        b.followup.no_science(gate)
        retire_controller(reviewed['controller'], gate)
        b.cache.write_immutable_json(directory/'retired.json', artifact('RETIRED',
            parents=dict(plan=reviewed['content_hash']), controller=reviewed['controller'],
            preflight_cancelled=False))
        b.followup.no_science(gate)
        print('ARMED ORDERED HANDOVER: preflight '+reviewed['preflight_job']+
              '; order 50 -> 500 -> 1000 -> control -> 100 -> 10 -> 2. '
              'Old controller retired; no preflight resubmitted.', flush=True)
        b.followup.wait_gate(gate, reviewed['preflight_job'])
        if recheck_source is not None and recheck_source() != tool_source:
            raise PermissionError('Ordering tool source changed while waiting')
        original, armed = old_binding(gate, b)
        if (original['content_hash'] != reviewed['parents']['original_followup']
                or armed['content_hash'] != reviewed['parents']['original_armed']):
            raise ValueError('Original binding changed while waiting')
        # Original science() fully authenticates technical outputs/environment,
        # resources, journal, source and successful exact-job accounting.
        spec = b.campaign.science(gate)
        # Block any later attempt to use the old independent submission path.
        old_submission = Path(spec['root'])/'submission'
        old_submission.mkdir(exist_ok=False)
        (old_submission/'live_claim').mkdir(exist_ok=False)
        b.cache.write_immutable_json(old_submission/'live_claim/ordering_override.json',
            artifact('ORIGINAL_SUBMIT_DISABLED', parents=dict(plan=reviewed['content_hash']),
                     replacement_journal=str(directory/'submission_ledger_journal')))
        original_exact = b.followup.review_science(gate, spec, original)
        if b.submission.submit(spec) != original_exact:  # original DRY review, never live
            raise ValueError('Original measured dry plan differs')
        exact = ordered_plan(original_exact)
        if ordered_plan(original['science_ceiling_plan']) != reviewed['science_ceiling_plan']:
            raise ValueError('Authorized order/ceiling changed')
        b.submission.check_site(exact)
        b.cache.write_immutable_json(directory/'command_plan.json', exact)
        dry = directory/'dry_run_submission_ledger.json'
        identity = exact['content_hash']
        b.exact.submit_exact_dag(identity=identity, plan=exact, output=dry,
                                canonical_dry_run=dry, execute=False)
        b.cache.write_immutable_json(directory/'measured_review.json', artifact('MEASURED_REVIEW',
            parents=dict(authorization=reviewed['content_hash'], spec=spec['content_hash'],
                         plan=exact['content_hash']), original_plan=original_exact['content_hash']))
        ledger = b.exact.submit_exact_dag(identity=identity, plan=exact,
            output=directory/'submission_ledger.json', canonical_dry_run=dry,
            execute=True, environment=b.submission.clean_environment())
        events, jobs = b.exact.load_exact_dag_journal(directory/'submission_ledger_journal',
                                                    identity=identity, plan=exact)
        if (set(jobs) != {*ORDER, 'aggregate'} or len(set(jobs.values())) != 8
                or ledger != b.ledger.assemble_submission_ledger(events, campaign_spec_sha256=identity)):
            raise ValueError('Ordered submission journal/ledger incomplete')
        result = artifact('COMPLETE', parents=dict(plan=identity, authorization=reviewed['content_hash']),
                          jobs=jobs, order=list(ORDER), training_complete=False)
        b.cache.write_immutable_json(directory/'complete.json', result)
        print('ALL 8 JOBS QUEUED IN REQUESTED ORDER: '+json.dumps(jobs), flush=True)
        return result
    except BaseException as exc:
        b.cache.write_immutable_json(directory/'failure.json', artifact('FAILURE',
            parents=dict(plan=reviewed['content_hash']), error_type=type(exc).__name__, error=str(exc),
            automatic_retry=False))
        print('STOPPED: inspect '+ATTEMPT+'/failure.json and journals. '
              'Do not rerun or resume the old controller.', flush=True)
        raise


def main(project):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--gate', required=True)
    parser.add_argument('--controller-pid', required=True, type=int)
    parser.add_argument('--controller-host', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--plan-hash')
    parser.add_argument('--authorization')
    a = parser.parse_args()
    source = source_identity(project, a.commit)
    gate, backend = bootstrap(a.gate)
    if Path(a.gate).resolve() != (Path(gate['root'])/'gate_spec.json').resolve():
        raise ValueError('Gate path does not match registered root')
    value = run(gate, backend, source, pid=a.controller_pid, host=a.controller_host,
                execute=a.execute, plan_hash=a.plan_hash, authorization=a.authorization,
                recheck_source=lambda: source_identity(project, a.commit))
    print(json.dumps(value, indent=2, sort_keys=True))
    if not a.execute:
        print('ORDER_PLAN_HASH='+value['content_hash'])
        print('Dry review only; no process signaled and no jobs submitted.')

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
import re
import signal
import socket
import subprocess
import sys
from types import SimpleNamespace

DONOR = 'fd687181a60c1557f98d7498f577e26d6dbb35a7'
ORDER = ('w50', 'w500', 'w1000', 'w0', 'w100', 'w10', 'w2')
AUTHORIZATION = 'AUTHORIZE_FULLSTAT_EMBEDDING_ORDER_V1'
FILES = ('scripts/order_luka_fullstat_embedding_sweep.py',
         'scripts/queue_luka_fullstat_embedding_order.sh',
         'src/hlt_classification/operations/fullstat_embedding_order.py')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def artifact(kind, **fields):
    value = dict(contract='LUKA_FULLSTAT_EMBEDDING_ORDER_'+kind+'/v1',
                 schema_version=1, final_test_accessed=False, **fields)
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


def review(gate, backend, tool_source, pid, host):
    original, armed = old_binding(gate, backend)
    process = controller_identity(pid, host, gate)
    if process['argv'][10] != original['content_hash']:
        raise PermissionError('Stopped controller has a different authorization hash')
    ceiling = ordered_plan(original['science_ceiling_plan'])
    return artifact('AUTHORIZATION_PLAN', parents=dict(gate=gate['content_hash'],
        original_followup=original['content_hash'], original_armed=armed['content_hash']),
        tool_source=tool_source, controller=process, preflight_job=armed['preflight_job'],
        science_ceiling_plan=ceiling, order=list(ORDER), jobs=8,
        max_train_minutes=2880, authorization=AUTHORIZATION,
        action='retire_stopped_submit_controller_then_wait_for_original_gate',
        cancel_preflight=False, automatic_retry=False)


def retire_controller(expected, gate):
    """pidfd prevents PID reuse; never signal Slurm or the shell parent."""
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('Linux Python with pidfd required; no unsafe PID fallback')
    fd = os.pidfd_open(expected['pid'])
    try:
        if controller_identity(expected['pid'], expected['host'], gate) != expected:
            raise PermissionError('Controller identity changed; no signal sent')
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        try:
            signal.pidfd_send_signal(fd, signal.SIGCONT)
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
    directory = root/'ordered_followup_v1'
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
        print('STOPPED: inspect ordered_followup_v1/failure.json and journals. '
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

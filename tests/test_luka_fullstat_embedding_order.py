"""Scheduling-only overlay tests; existing scientific/kernel suites stay unchanged."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_luka_fullstat_embedding_sweep_followup import controller
from hlt_classification.operations import fullstat_embedding_order as o
from hlt_classification.luka_fullstat_embedding_sweep import campaign, contracts, followup, submission
from hlt_classification.scouting import hcwdl_exact_dag_submission as exact, hcwdl_recovery as ledger
from hlt_classification.data import cache_contracts as cache


@pytest.fixture
def handover(controller, monkeypatch):
    c = controller
    plan = submission.submit(c.gate)
    submission.submit(c.gate, execute=True, plan_hash=plan['content_hash'], authorization=contracts.AUTHORIZATION)
    old = followup.plan(c.gate)
    intent = contracts.artifact('FOLLOWUP_AUTHORIZATION', parents=dict(plan=old['content_hash']),
                               source_commit=c.gate['source_commit'], gate=c.gate['content_hash'])
    gate_ledger = cache.load_json(c.root/'submission/submission_ledger.json')
    armed = contracts.artifact('FOLLOWUP_ARMED', parents=dict(authorization=intent['content_hash'],
        gate_ledger=gate_ledger['content_hash']), preflight_job='12345')
    for name, value in [('plan', old), ('authorization', intent), ('armed', armed)]:
        cache.write_immutable_json(c.root/f'followup/{name}.json', value)
    argv = ['python', '-u', '-B', '-s', c.gate['project_dir']+'/scripts/luka_fullstat_embedding_sweep.py',
        'followup', '--spec', str(c.root/'gate_spec.json'), '--execute', '--plan-hash',
        old['content_hash'], '--authorization', followup.AUTHORIZATION]
    process = dict(pid=300411, host='login010', uid=123, start_ticks='123456', argv=argv)
    monkeypatch.setattr(o, 'controller_identity', lambda *args, **kwargs: deepcopy(process))
    retired = []
    monkeypatch.setattr(o, 'retire_controller', lambda p, g: retired.append(p))
    backend = SimpleNamespace(campaign=campaign, contracts=contracts, followup=followup,
        submission=submission, exact=exact, ledger=ledger, cache=cache)
    return SimpleNamespace(c=c, backend=backend, source=dict(commit='a'*40, files={}),
                           process=process, retired=retired)


def run(h, **kwargs):
    return o.run(h.c.gate, h.backend, h.source, pid=300411, host='login010', **kwargs)


def execute(h):
    reviewed = run(h)
    return run(h, execute=True, plan_hash=reviewed['content_hash'], authorization=o.AUTHORIZATION)


def test_dry_review_leaves_gate_and_controller_alone(handover):
    h = handover
    reviewed = run(h)
    assert not h.retired and len(h.c.calls) == 1
    assert not (h.c.root/'science').exists()
    assert reviewed['jobs'] == 8 and reviewed['preflight_job'] == '12345'
    assert reviewed['order'] == ['w50', 'w500', 'w1000', 'w0', 'w100', 'w10', 'w2']
    assert reviewed['cancel_preflight'] is False
    assert cache.load_json(h.c.root/'ordered_followup_v1/plan.json') == reviewed


def test_exact_success_handover_uses_existing_gate_and_preserves_original_plan(handover):
    h = handover
    before = (h.c.root/'followup/plan.json').read_bytes()
    result = execute(h)
    assert len(h.retired) == 1 and h.c.sleeps == [60, 60]
    assert len(h.c.calls) == 9  # one pre-existing gate, eight new jobs
    assert (h.c.root/'followup/plan.json').read_bytes() == before
    assert result['training_complete'] is False
    commands = h.c.calls[1:]
    assert [cmd[-1] for cmd in commands] == [*o.ORDER, 'aggregate']
    assert not any(arg.startswith('--dependency=') for arg in commands[0])
    for i, cmd in enumerate(commands[1:7], 1):
        assert '--dependency=afterany:'+result['jobs'][o.ORDER[i-1]] in cmd
    deps = next(v for v in commands[-1] if v.startswith('--dependency='))
    assert deps == '--dependency=afterok:'+':'.join(result['jobs'][a] for a in contracts.WEIGHTS)
    spec = cache.load_json(h.c.root/'science/campaign_spec.json')
    assert spec['source_commit'] == h.c.gate['source_commit']
    original = submission.render(spec, h.c.gate)
    for row in original['commands']:
        actual = next(cmd for cmd in commands if cmd[-1] == row['task_id'])
        if row['task_id'] != 'aggregate':
            actual = [v for v in actual if not v.startswith('--dependency=')]
        expected = row['command']
        for arm, job in result['jobs'].items():
            expected = [v.replace('${JOB_'+arm+'}', job) for v in expected]
        assert actual == expected
    assert not (h.c.root/'science/submission/submission_ledger.json').exists()
    assert (h.c.root/'science/submission/live_claim/ordering_override.json').is_file()
    assert (h.c.root/'ordered_followup_v1/complete.json').is_file()
    with pytest.raises(FileExistsError): execute(h)
    assert len(h.c.calls) == 9


@pytest.mark.parametrize('state', ['FAILED', 'CANCELLED by 123', 'OUT_OF_MEMORY', 'TIMEOUT', 'NODE_FAIL'])
def test_gate_failure_never_queues_science(handover, state):
    h = handover
    h.c.states.clear(); h.c.states.append(state)
    with pytest.raises(RuntimeError, match='no science'): execute(h)
    assert h.retired and len(h.c.calls) == 1 and not (h.c.root/'science').exists()
    assert (h.c.root/'ordered_followup_v1/failure.json').is_file()


@pytest.mark.parametrize('key', ['bad_receipt', 'missing_receipt'])
def test_completed_gate_still_needs_authentic_technical_evidence(handover, key):
    h = handover
    h.c.control[key] = True
    with pytest.raises((ValueError, FileNotFoundError)): execute(h)
    assert len(h.c.calls) == 1 and not (h.c.root/'science').exists()


@pytest.mark.parametrize('name', ['plan', 'authorization', 'armed'])
def test_original_binding_tamper_rejected_before_retirement(handover, name):
    h = handover
    path = h.c.root/f'followup/{name}.json'
    path.write_text('{}')
    with pytest.raises(ValueError): run(h)
    assert not h.retired and len(h.c.calls) == 1


def test_wrong_authorization_or_plan_cannot_retire(handover):
    h = handover
    reviewed = run(h)
    for hash_, auth in [('b'*64, o.AUTHORIZATION), (reviewed['content_hash'], followup.AUTHORIZATION)]:
        with pytest.raises(PermissionError):
            run(h, execute=True, plan_hash=hash_, authorization=auth)
    assert not h.retired and not (h.c.root/'ordered_followup_v1/live_claim').exists()


def test_process_reused_or_resumed_between_review_and_execute_blocks(handover):
    h = handover
    reviewed = run(h)
    h.process['start_ticks'] = '999'
    with pytest.raises(PermissionError):
        run(h, execute=True, plan_hash=reviewed['content_hash'], authorization=o.AUTHORIZATION)
    assert not h.retired


def test_existing_science_blocks_even_dry_handover(handover):
    h = handover
    (h.c.root/'science').mkdir()
    with pytest.raises(FileExistsError): run(h)
    assert not h.retired


def test_ambiguous_partial_submission_never_retried(handover):
    h = handover
    h.c.control['fail_submit'] = 4
    with pytest.raises(RuntimeError, match='Ambiguous'): execute(h)
    saved = h.c.root/'ordered_followup_v1/submission_ledger_journal'
    assert len(list(saved.glob('*.json'))) == 2
    with pytest.raises(FileExistsError): execute(h)
    assert len(h.c.calls) == 4


def test_failed_retirement_cannot_wait_or_submit(handover, monkeypatch):
    h = handover
    def fail(*args): raise PermissionError('PID changed')
    monkeypatch.setattr(o, 'retire_controller', fail)
    with pytest.raises(PermissionError): execute(h)
    assert not h.c.sleeps and len(h.c.calls) == 1


def test_order_source_drift_during_wait_blocks_science(handover):
    h = handover
    reviewed = run(h)
    with pytest.raises(PermissionError, match='source changed'):
        run(h, execute=True, plan_hash=reviewed['content_hash'], authorization=o.AUTHORIZATION,
            recheck_source=lambda: {})
    assert not (h.c.root/'science').exists() and len(h.c.calls) == 1


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'deps', 'aggregate'])
def test_order_overlay_rejects_other_graphs(handover, mutation):
    h = handover
    plan = followup.plan(h.c.gate)['science_ceiling_plan']
    if mutation == 'missing': plan['commands'].pop(0)
    if mutation == 'duplicate': plan['commands'].append(plan['commands'][0])
    if mutation == 'deps': plan['commands'][0]['dependencies'] = ['w2']
    if mutation == 'aggregate': plan['commands'][-1]['command'].insert(-4, '--dependency=afterany:1')
    with pytest.raises(ValueError): o.ordered_plan(plan)


def test_proc_identity_reads_exact_stopped_command(tmp_path, monkeypatch):
    gate = dict(root='/campaign', project_dir='/source')
    directory = tmp_path/'300411'; directory.mkdir()
    argv = ['python', '-u', '-B', '-s', '/source/scripts/luka_fullstat_embedding_sweep.py',
        'followup', '--spec', str(Path('/campaign')/'gate_spec.json'), '--execute', '--plan-hash', 'a'*64,
        '--authorization', followup.AUTHORIZATION]
    (directory/'cmdline').write_bytes(b'\0'.join(v.encode() for v in argv)+b'\0')
    # state=field3, starttime=field22; process comm may include spaces and parens.
    (directory/'stat').write_text('300411 (python (worker)) T '+' '.join(['0']*18+['123456']))
    monkeypatch.setattr(o.socket, 'gethostname', lambda: 'login010.oscar')
    monkeypatch.setattr(o.os, 'getuid', lambda: directory.stat().st_uid, raising=False)
    value = o.controller_identity(300411, 'login010', gate, proc=tmp_path)
    assert value['start_ticks'] == '123456' and value['argv'] == argv
    with pytest.raises(PermissionError): o.controller_identity(300411, 'login009', gate, proc=tmp_path)
    (directory/'stat').write_text((directory/'stat').read_text().replace(' T ', ' S '))
    with pytest.raises(PermissionError): o.controller_identity(300411, 'login010', gate, proc=tmp_path)


def test_standalone_cli_does_not_import_new_scientific_source():
    text = Path('scripts/order_luka_fullstat_embedding_sweep.py').read_text()
    assert 'spec_from_file_location' in text and 'sys.path.insert' not in text
    helper = Path('scripts/queue_luka_fullstat_embedding_order.sh').read_text()
    assert 'export PROJECT_DIR="${ORIGINAL}"' in helper
    assert 'source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"' in helper


def test_pidfd_retirement_signals_only_verified_process(monkeypatch):
    import sys
    calls = []
    process = dict(pid=300411, host='login010')
    monkeypatch.setattr(o.os, 'pidfd_open', lambda pid: calls.append(('open', pid)) or 77, raising=False)
    monkeypatch.setattr(o.os, 'close', lambda fd: calls.append(('close', fd)))
    monkeypatch.setattr(o.signal, 'SIGCONT', 18, raising=False)
    monkeypatch.setattr(o.signal, 'pidfd_send_signal', lambda *a: calls.append(('signal', *a)), raising=False)
    monkeypatch.setattr(o, 'controller_identity', lambda *a: process)
    fake_poll = SimpleNamespace(register=lambda *a: None, poll=lambda ms: [(77, 1)])
    monkeypatch.setitem(sys.modules, 'select', SimpleNamespace(poll=lambda: fake_poll, POLLIN=1))
    o.retire_controller(process, {})
    assert calls == [('open', 300411), ('signal', 77, o.signal.SIGTERM),
                     ('signal', 77, 18), ('close', 77)]
    calls.clear()
    monkeypatch.setattr(o, 'controller_identity', lambda *a: dict(process, start_ticks='different'))
    with pytest.raises(PermissionError): o.retire_controller(process, {})
    assert calls == [('open', 300411), ('close', 77)]


def test_site_rejection_blocks_review_without_signals(handover, monkeypatch):
    h = handover
    def fail(plan): raise PermissionError('Oscar resource shape rejected')
    monkeypatch.setattr(submission, 'check_site', fail)
    with pytest.raises(PermissionError): run(h)
    assert not h.retired and len(h.c.calls) == 1

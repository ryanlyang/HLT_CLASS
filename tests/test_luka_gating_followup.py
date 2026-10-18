"""Scheduler-free gate continuation tests, including real exact-DAG journals."""
from collections import deque
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_luka_gating import gate_fixture
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.luka_gating import campaign, contracts, followup as f, preflight, submission as s


@pytest.fixture
def controller(tmp_path, monkeypatch):
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    gate = gate_fixture(tmp_path)
    write_immutable_json(tmp_path/'gate_spec.json', gate)
    monkeypatch.setattr(campaign, 'check_gate', lambda gate: None)
    monkeypatch.setattr(s, 'check_site', lambda plan: None)
    monkeypatch.setattr(f.getpass, 'getuser', lambda: 'owner')
    calls, sleeps, authenticated = [], [], []
    states = deque(['PENDING', 'RUNNING', 'COMPLETED'])
    control = dict(completed=False, bad_receipt=False, fail_submit=None, missing_receipt=False)

    def scheduler(args, **kwargs):
        if args[0] == 'sacct':
            assert args[args.index('-j')+1] == '12345'
            assert kwargs['timeout'] == 60
            state = states.popleft()
            if state == 'COMPLETED':
                control['completed'] = True
                if not control['missing_receipt']:
                    write_immutable_json(tmp_path/'preflight/receipt.json', contracts.artifact('PREFLIGHT',
                        job_id='12345', train_minutes=96, export_minutes=15, technical_checks_passed=True))
            return SimpleNamespace(stdout=f'12345|lukag_preflight|owner|reu-aisocial|{state}|0:0\n')
        assert args[0] == 'sbatch' and '--no-requeue' in args
        calls.append(args)
        if len(calls) > 1:
            assert control['completed'] and authenticated
            assert not any('${JOB_' in arg for arg in args)
        if len(calls) == control['fail_submit']:
            raise RuntimeError('Ambiguous sbatch response')
        return SimpleNamespace(stdout=str(12344+len(calls))+'\n')

    def receipt(gate, report):
        assert control['completed']
        if control['bad_receipt']:
            raise ValueError('Technical receipt/source/resource admission invalid')
        authenticated.append(report['content_hash'])

    def accounted(job):
        assert job == '12345' and control['completed']

    monkeypatch.setattr(exact.subprocess, 'run', scheduler)
    monkeypatch.setattr(preflight, 'validate_report', receipt)
    monkeypatch.setattr(campaign, 'preflight_accounting', accounted)
    wait = f.wait_gate
    monkeypatch.setattr(f, 'wait_gate', lambda gate, job: wait(gate, job, sleep=sleeps.append))
    return SimpleNamespace(root=tmp_path, gate=gate, calls=calls, sleeps=sleeps,
                           states=states, control=control, authenticated=authenticated)


def execute(c, review=None):
    review = f.run(c.gate) if review is None else review
    return f.run(c.gate, execute=True, plan_hash=review['content_hash'], authorization=f.AUTHORIZATION)


def test_dry_scope_and_explicit_authorization(controller):
    c = controller
    review = f.run(c.gate)
    assert not c.calls and not (c.root/'science').exists()
    assert review['fits'] == 13 and review['science_jobs'] == 23 and review['max_total_jobs'] == 24
    assert review['gate_plan']['commands'][0]['command'].count('--time=08:00:00') == 1
    assert review['max_train_minutes'] == review['max_export_minutes'] == 1440
    for row in review['science_ceiling_plan']['commands']:
        assert '--partition=debug' in row['command']
        assert ('--time=1-00:00:00' in row['command']) if row['task_id'] != 'aggregate' else ('--time=01:00:00' in row['command'])
    for hash_, auth in [('bad', f.AUTHORIZATION), (review['content_hash'], campaign.AUTHORIZATION)]:
        with pytest.raises(PermissionError):
            f.run(c.gate, execute=True, plan_hash=hash_, authorization=auth)
    assert not c.calls and not (c.root/'followup/live_claim').exists()


def test_success_waits_authenticates_and_queues_exact_dag_once(controller, capsys):
    c = controller
    result = execute(c)
    assert c.sleeps == [60, 60] and len(c.calls) == 24
    assert result['training_complete'] is False and len(result['jobs']) == 23
    assert load_json(c.root/'followup/armed.json')['preflight_job'] == '12345'
    assert load_json(c.root/'followup/complete.json') == result
    spec = load_json(c.root/'science/campaign_spec.json')
    assert spec['train_minutes'] == 96 and spec['export_minutes'] == 15
    assert '--time=01:36:00' in next(cmd for cmd in c.calls if cmd[-1] == 'train_WARM_D066')
    assert 'ARMED: preflight 12345' in capsys.readouterr().out
    with pytest.raises(FileExistsError): execute(c)
    assert len(c.calls) == 24


def test_existing_exact_preflight_is_bound_not_duplicated(controller):
    c = controller
    review = s.submit(c.gate)
    s.submit(c.gate, execute=True, plan_hash=review['content_hash'], authorization=campaign.AUTHORIZATION)
    assert len(c.calls) == 1
    execute(c)
    assert len(c.calls) == 24


@pytest.mark.parametrize('state', ['FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'CANCELLED by 1234', 'NODE_FAIL'])
def test_terminal_gate_never_creates_science(controller, state):
    c = controller
    c.states.clear(); c.states.append(state)
    with pytest.raises(RuntimeError, match='no science submission'): execute(c)
    assert len(c.calls) == 1 and not (c.root/'science').exists()
    assert (c.root/'followup/failure.json').is_file()


@pytest.mark.parametrize('bad,missing,error', [(True, False, ValueError), (False, True, FileNotFoundError)])
def test_success_exit_without_authenticated_receipt_cannot_queue(controller, bad, missing, error):
    c = controller
    c.control.update(bad_receipt=bad, missing_receipt=missing)
    with pytest.raises(error): execute(c)
    assert len(c.calls) == 1 and not (c.root/'science').exists()


def test_ambiguous_gate_submission_never_retried(controller):
    c = controller
    c.control['fail_submit'] = 1
    review = f.run(c.gate)
    with pytest.raises(RuntimeError, match='Ambiguous'): execute(c, review)
    with pytest.raises(PermissionError, match='ambiguous'): execute(c, review)
    assert len(c.calls) == 1 and not (c.root/'science').exists()


def test_partial_science_submission_preserves_journal_and_blocks_retry(controller):
    c = controller
    c.control['fail_submit'] = 5
    review = f.run(c.gate)
    with pytest.raises(RuntimeError, match='Ambiguous'): execute(c, review)
    assert len(list((c.root/'science/submission/submission_ledger_journal').glob('*.json'))) == 3
    assert not (c.root/'followup/complete.json').exists()
    with pytest.raises(FileExistsError): execute(c, review)
    assert len(c.calls) == 5


def test_duplicate_armed_controller_fails_closed(controller, monkeypatch):
    c = controller
    review = f.run(c.gate)
    original = f.wait_gate

    def duplicate(gate, job):
        with pytest.raises(FileExistsError): execute(c, review)
        return original(gate, job)

    monkeypatch.setattr(f, 'wait_gate', duplicate)
    execute(c, review)
    assert len(c.calls) == 24


def test_manual_science_directory_is_not_taken_over(controller):
    c = controller
    review = f.run(c.gate)
    (c.root/'science').mkdir()
    with pytest.raises(FileExistsError, match='Science already exists'): execute(c, review)
    assert not c.calls


def test_measurement_plan_ceiling_and_changed_scope(controller, monkeypatch):
    c = controller
    review = f.run(c.gate)
    spec = dict(f.ceiling_spec(c.gate), content_hash='a'*64, train_minutes=90, export_minutes=15)
    monkeypatch.setattr(campaign, 'check_spec', lambda spec: (c.gate, {}))
    assert f.review_science(c.gate, spec, review)['jobs'] == 23
    for value in (1441, 0, True, float('nan')):
        with pytest.raises(ValueError): f.review_science(c.gate, dict(spec, train_minutes=value), review)
    for key, value in [('root', '/wrong/root'), ('project_dir', '/wrong/project')]:
        with pytest.raises(ValueError): f.review_science(c.gate, dict(spec, **{key:value}), review)
    changed = deepcopy(review)
    changed['science_ceiling_plan']['commands'].pop()
    with pytest.raises(ValueError): f.review_science(c.gate, spec, changed)


@pytest.mark.parametrize('raw,error', [
    ('12345|lukag_preflight|owner|reu-aisocial|COMPLETED|1:0', RuntimeError),
    ('12345|lukag_preflight|other|reu-aisocial|COMPLETED|0:0', PermissionError),
    ('12346|lukag_preflight|owner|reu-aisocial|COMPLETED|0:0', PermissionError),
    ('12345|different_job|owner|reu-aisocial|COMPLETED|0:0', PermissionError),
    ('12345|lukag_preflight|owner|wrong_account|COMPLETED|0:0', PermissionError),
    ('garbled', ValueError),
    ('12345|lukag_preflight|owner|reu-aisocial|UNKNOWN|0:0', RuntimeError),
])
def test_accounting_fails_closed(raw, error):
    with pytest.raises(error): f.accounting(raw, job='12345', user='owner', account='reu-aisocial')
    assert f.accounting('', job='12345', user='owner', account='reu-aisocial') == 'ACCOUNTING_PENDING'


def test_wait_bound_is_not_success(controller):
    c = controller
    # Access original wait function because the fixture supplies a no-sleep wrapper.
    import inspect
    wait = inspect.getclosurevars(f.wait_gate).nonlocals['wait']
    ticks = iter([0, f.WAIT_SECONDS+1])
    with pytest.raises(TimeoutError): wait(c.gate, '12345', monotonic=lambda:next(ticks))
    assert not c.calls

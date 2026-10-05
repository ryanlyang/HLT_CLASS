"""CPU restoration regression; the supplied native v1 acceptance remains authoritative."""
from copy import deepcopy
from pathlib import Path

import pytest

from hlt_classification.data.cache_contracts import load_json, sha256_file, write_immutable_json
from hlt_classification.k2_segmented import campaign, memory_migration as migration, restoration, runtime, training
from hlt_classification.scouting.hcwdl_recovery import build_submission_event, assemble_submission_ledger
from test_k2_segmented_memory import prepared, rehash


@pytest.fixture
def attempt(prepared):
    p = prepared
    p.unused = migration.create(donor_spec=p.root/'campaign_spec.json', campaign_root=p.new_root,
        project_dir=p.new_project, source_commit=restoration.ABANDONED_COMMIT)
    events, jobs = [], {}
    for index, row in enumerate(campaign.plan(p.unused, 'gate')['commands']):
        job = str(5000 + index)
        events.append(build_submission_event(campaign_spec_sha256=p.unused['content_hash'],
            task_id=row['task_id'], job_id=job, command=campaign._resolved(row, jobs), sequence=index))
        jobs[row['task_id']] = job
        p.states[job] = ('PENDING', '0:0')
    ledger = assemble_submission_ledger(events, campaign_spec_sha256=p.unused['content_hash'])
    write_immutable_json(p.new_root/'submissions_gate/submission_ledger.json', ledger)
    for job in p.unused['old_jobs'].values():
        p.states[job] = ('CANCELLED', '0:0')
    p.restore_root = p.root.parent/'restored'
    p.restore_project = p.root.parent/'restored_code'
    target = p.restore_project/'src/hlt_classification/k2_segmented/training.py'
    target.parent.mkdir(parents=True)
    target.write_bytes(Path(training.__file__).read_bytes())
    return p


def create(p):
    return restoration.create(abandoned_spec=p.new_root/'campaign_spec.json', campaign_root=p.restore_root,
        project_dir=p.restore_project, source_commit='f'*40)


def tree(root):
    return {str(p): sha256_file(p) for p in root.rglob('*') if p.is_file()}


def test_restore_has_eleven_original_resource_jobs_and_no_new_gate(attempt):
    p = attempt
    before = (tree(p.root), tree(p.new_root))
    spec = create(p)
    assert spec['contract'] == 'K2_SEGMENTED_CAMPAIGN_SPEC/v3'
    assert spec['resources'] == p.donor['resources'] == campaign.RESOURCES
    assert spec['training'] == p.donor['training']
    assert spec['resume_import']['checkpoint']['pass_number'] == 4
    for stage in ('science', 'full'):
        rows = campaign.plan(spec, stage)['commands']
        assert len(rows) == 11 and rows[0]['task_id'] == migration.FIRST_TASK
        assert not rows[0]['dependencies']
        for row in rows:
            assert '--partition=debug' in row['command']
            assert row['task_id'] not in ('preflight', 'after_gate', *migration.PREFIX[1:])
            if '--gres=gpu:a100:1' in row['command']:
                assert '--mem=320000M' in row['command']
            if row['task_id'].startswith('train_'):
                assert '--time=1380' in row['command']
        for previous, row in zip(rows, rows[1:]):
            assert row['dependencies'] == [previous['task_id']]
    assert campaign.gate(spec) == campaign.gate(p.donor)
    assert not (p.restore_root/'preflight').exists()
    assert not (p.restore_root/'tasks/preflight.json').exists()
    assert before == (tree(p.root), tree(p.new_root))
    with pytest.raises(ValueError, match='not another gate'):
        campaign.submit(spec, stage='gate')


def test_actual_resumed_state_equals_original_without_gpu_preflight(attempt, monkeypatch):
    p = attempt
    monkeypatch.setattr(runtime, 'preflight', lambda *a: pytest.fail('No fresh preflight'))
    spec = create(p)
    before = tree(p.root)
    monkeypatch.setattr(runtime, 'can_start', lambda *a, **k: True)
    result = runtime.run(spec, migration.FIRST_TASK, device='cpu')
    assert result['result']['fit_complete'] and result['result']['passes'] == 75
    assert tree(p.root) == before
    runtime.run(p.donor, migration.FIRST_TASK, device='cpu')
    binding = spec['resume_import']['checkpoint']['binding']
    left, _ = training.load_checkpoint(p.root/'checkpoints'/migration.NODE, binding)
    right, _ = training.load_checkpoint(p.restore_root/'checkpoints'/migration.NODE, binding)
    for key in left.keys() - {'history', 'runtime_seconds'}:
        runtime.assert_state_equal(left[key], right[key])
    runtime.assert_state_equal(runtime.scientific_history(left['history']), runtime.scientific_history(right['history']))


def test_all_downstream_tasks_use_restored_parents(attempt, monkeypatch):
    p = attempt
    spec = create(p)
    monkeypatch.setattr(runtime, 'can_start', lambda *a, **k: True)
    for row in campaign.graph(spec)[:-2]:
        receipt = runtime.run(spec, row['task_id'], device='cpu')
        assert runtime.completed(spec, row['task_id']) == receipt
    for node in campaign.NODES[1:]:
        report = load_json(p.restore_root/'training'/node/'training_report.json')
        assert report['teacher_lineage']['task_sha256'] == runtime.completed(
            spec, 'reduce_' + report['node']['teacher_distribution'])['content_hash']
    deployment = load_json(p.restore_root/'training/HLT_X1_COMPRESSED/deployment.json')
    assert deployment['native_hlt_copies'] == 1 and not deployment['offline_inputs']


def test_retirement_only_cancels_two_unused_gates_then_live_science(attempt, monkeypatch):
    p = attempt
    spec = create(p)
    calls = []
    def cancel(cmd, **kw):
        calls.append(cmd)
        for job in cmd[2:]:
            p.states[job] = ('CANCELLED', '0:0')
    monkeypatch.setattr(campaign.subprocess, 'run', cancel)
    kwargs = dict(stage='science', execute=True, authorization=campaign.AUTHORIZE, debug_policy_confirmed=True)
    with pytest.raises(PermissionError, match='retired'):
        campaign.submit(spec, **kwargs)
    campaign.retire(spec, execute=True, authorization=campaign.RETIRE)
    assert len(calls) == 1 and calls[0][:2] == ['scancel', '--state=PENDING']
    assert set(calls[0][2:]) == {'5000', '5001'}
    monkeypatch.setattr(campaign, '_guarded_exact_submission', lambda s, plan, root: plan)
    plan = campaign.submit(spec, **kwargs)
    assert len(plan['commands']) == 11
    assert all(p.states[j] == ('COMPLETED', '0:0') for j in spec['resume_import']['completed_jobs'].values())


@pytest.mark.parametrize('state', ['RUNNING', 'COMPLETED', 'FAILED', 'UNKNOWN'])
def test_started_or_unknown_new_gate_blocks_restoration(attempt, state):
    attempt.states['5000'] = (state, '0:0')
    with pytest.raises(PermissionError, match='never running'):
        create(attempt)
    assert not attempt.restore_root.exists()


@pytest.mark.parametrize('marker', ['claims/preflight.claim', 'execution/preflight.json',
    'tasks/preflight.json', 'submissions_science/submission_intents/0000.json',
    'submissions_science/submission_ledger_journal/0000.json', 'submissions_science/submission_in_progress.claim'])
def test_work_or_ambiguous_submission_cannot_be_discarded(attempt, marker):
    path = attempt.new_root/marker
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{}')
    with pytest.raises(PermissionError, match='started'):
        create(attempt)


def test_original_reactivated_or_advanced_blocks_restoration(attempt):
    p = attempt
    p.states[p.unused['old_jobs'][migration.FIRST_TASK]] = ('PENDING', '0:0')
    with pytest.raises(PermissionError, match='remain cancelled'):
        create(p)


def test_rehashed_resource_and_acceptance_tampering_rejected(attempt):
    spec = create(attempt)
    changed = deepcopy(spec)
    changed['resources']['segment']['memory_mb'] = 131072
    with pytest.raises(ValueError):
        campaign.validate_spec(rehash(changed))
    path = attempt.restore_root/'reused_native_gate.json'
    evidence = load_json(path)
    evidence['donor_acceptance_sha256'] = '0'*64
    path.write_text(__import__('json').dumps(rehash(evidence)))
    with pytest.raises(ValueError, match='provenance'):
        campaign.gate(spec)


def test_no_native_gate_or_after_gate_execution_is_allowed(attempt, monkeypatch):
    spec = create(attempt)
    monkeypatch.setattr(runtime, 'authenticate_job', lambda *a: pytest.fail('Must reject before job execution'))
    for task in ('preflight', 'after_gate'):
        with pytest.raises(ValueError, match='no preflight'):
            runtime.run(spec, task, device='cpu')


def test_cancellation_race_blocks_science_without_killing_running_work(attempt, monkeypatch):
    p = attempt
    spec = create(p)
    def raced(cmd, **kw):
        assert cmd[:2] == ['scancel', '--state=PENDING']
        assert set(cmd[2:]) == {'5000', '5001'}
        p.states['5000'] = ('RUNNING', '0:0')
        p.states['5001'] = ('CANCELLED', '0:0')
    monkeypatch.setattr(campaign.subprocess, 'run', raced)
    monkeypatch.setattr(campaign.time, 'sleep', lambda *a: None)
    with pytest.raises(PermissionError, match='retired'):
        campaign.retire(spec, execute=True, authorization=campaign.RETIRE)
    assert p.states['5000'][0] == 'RUNNING'
    assert not (p.restore_root/'submissions_science/submission_ledger.json').exists()


def test_checkpoint_copy_conflict_rejected_before_live_submission(attempt):
    p = attempt
    spec = create(p)
    for job in spec['old_jobs'].values():
        p.states[job] = ('CANCELLED', '0:0')
    target = p.restore_root/'checkpoints'/migration.NODE/'epoch_001.pt'
    target.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='copy changed'):
        campaign.submit(spec, stage='science', execute=True, authorization=campaign.AUTHORIZE,
                        debug_policy_confirmed=True)
    assert not (p.restore_root/'submissions_science/submission_in_progress.claim').exists()


def test_prepare_is_idempotent_and_does_not_rerun_native_gate(attempt, monkeypatch):
    p = attempt
    spec = create(p)
    monkeypatch.setattr(runtime, 'preflight', lambda *a: pytest.fail('No native rerun'))
    before = tree(p.restore_root)
    restoration.prepare(spec)
    assert tree(p.restore_root) == before

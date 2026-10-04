"""Resume equivalence and fail-closed scheduling without any remote mutation."""
from copy import deepcopy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from hlt_classification.k2_segmented import training as engine, campaign, runtime
from hlt_classification.jetclass2_delphes import concat_k2_campaign as k2
from hlt_classification.jetclass2_delphes import salience_learned_training as donor
from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json
from test_jetclass2_concat_k2_batch128 import TrackedCache


class DropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(17)
        self.dropout = nn.Dropout(.3)
        self.linear = nn.Linear(17, 11)

    def forward(self, features, vectors, mask):
        return self.linear(self.dropout(self.bn(features).mean(-1)))


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def setup_fit(kd=True):
    torch.manual_seed(17)
    np.random.seed(18)
    random.seed(19)
    train, val = TrackedCache('train'), TrackedCache('validation')
    node = k2.nodes()[4 if kd else 3]
    q = np.eye(11, dtype=np.float32)[train.labels] if kd else None
    return train, val, node, q


@pytest.mark.parametrize('kd', [False, True])
def test_exact_disk_resume_with_dropout_adamw_bn_and_partial_batch(tmp_path, kd):
    train, val, node, q = setup_fit(kd)
    model = DropoutModel()
    initial_weights, initial_rng = engine.cpu_tree(model.state_dict()), engine.rng_state()
    kwargs = dict(node=node, device='cpu', teacher=q, binding={'campaign': 'c'*64}, maximum=3, acceptance=True)
    uninterrupted = engine.train_segment(model, train, val, **kwargs)
    model = DropoutModel()
    model.load_state_dict(initial_weights)
    engine.restore_rng(initial_rng)
    first = engine.train_segment(model, train, val, **kwargs, on_epoch=lambda _: True)
    receipt = engine.write_checkpoint(tmp_path, first)
    resumed_state, read_receipt = engine.load_checkpoint(tmp_path, first['binding'])
    assert receipt == read_receipt
    del model, first
    random.random(); np.random.rand(19); torch.rand(30)
    model = DropoutModel()
    resumed = engine.train_segment(model, train, val, **kwargs, resume=resumed_state)
    for key in uninterrupted.keys() - {'history', 'runtime_seconds'}:
        runtime.assert_state_equal(uninterrupted[key], resumed[key])
    runtime.assert_state_equal(runtime.scientific_history(uninterrupted['history']), runtime.scientific_history(resumed['history']))
    assert resumed['update'] == 9  # 128 + 128 + 44, over three epochs
    assert resumed['selected_pass'] in (1, 2, 3)


@pytest.mark.parametrize('kd', [False, True])
def test_new_kernel_is_the_original_recipe(tmp_path, monkeypatch, kd):
    train, val, node, q = setup_fit(kd)
    def new_model(_):
        torch.manual_seed(23)
        return DropoutModel()
    monkeypatch.setattr(runtime.legacy, 'new_model', new_model)
    size = runtime.resume_parity(train, val, node, q, tmp_path, 'cpu')
    assert size > 0


def test_schedule_and_patience_continue_across_epoch60(tmp_path, monkeypatch):
    train, val, node, q = setup_fit()
    # Fixed poor-but-finite science metrics must still train until epoch75.
    metric = dict(macro_ovr_auc=.5, cross_entropy=1., macro_mean_log_qcd_rejection_at_50pct_signal=None)
    monkeypatch.setattr(donor, 'evaluate_probabilities', lambda *a: deepcopy(metric))
    kwargs = dict(node=node, device='cpu', teacher=q, binding={'campaign': 'd'*64})
    torch.manual_seed(30)
    initial = DropoutModel()
    weights, rng = engine.cpu_tree(initial.state_dict()), engine.rng_state()
    uninterrupted = engine.train_segment(initial, train, val, **kwargs)
    model = DropoutModel(); model.load_state_dict(weights); engine.restore_rng(rng)
    partial = engine.train_segment(model, train, val, **kwargs, on_epoch=lambda s: s['pass_number'] == 61)
    assert partial['pass_number'] == 61 and not partial['complete']
    continued = engine.train_segment(DropoutModel(), train, val, **kwargs, resume=partial)
    assert uninterrupted['pass_number'] == continued['pass_number'] == 75
    assert uninterrupted['selected_pass'] == continued['selected_pass'] == 1
    runtime.assert_state_equal(uninterrupted['optimizer'], continued['optimizer'])
    runtime.assert_state_equal(uninterrupted['model'], continued['model'])
    runtime.assert_state_equal(runtime.scientific_history(uninterrupted['history']), runtime.scientific_history(continued['history']))
    assert continued['history'][60]['learning_rate'] == donor.learning_rate(61.)


@pytest.mark.parametrize('change', ['teacher', 'node', 'identities', 'maximum'])
def test_cross_identity_resume_rejected(change):
    train, val, node, q = setup_fit()
    kwargs = dict(node=node, device='cpu', teacher=q, binding={'teacher': 't'*64}, maximum=3, acceptance=True)
    state = engine.train_segment(DropoutModel(), train, val, **kwargs, on_epoch=lambda _: True)
    if change == 'teacher': kwargs['binding']['teacher'] = 'q'*64
    if change == 'node': kwargs['node'] = {**node, 'node_id': 'wrong'}
    if change == 'identities': train.identities[0, 0] ^= 1
    if change == 'maximum': kwargs['maximum'] = 2
    with pytest.raises(ValueError, match='Resume population'):
        engine.train_segment(DropoutModel(), train, val, **kwargs, resume=state)


def test_payload_corruption_and_orphan_rejected(tmp_path):
    train, val, node, q = setup_fit()
    state = engine.train_segment(DropoutModel(), train, val, node=node, device='cpu', teacher=q,
        binding={}, maximum=3, acceptance=True, on_epoch=lambda _: True)
    engine.write_checkpoint(tmp_path, state)
    payload = tmp_path / 'epoch_001.pt'
    payload.write_bytes(payload.read_bytes() + b'corrupt')
    with pytest.raises(ValueError, match='bytes differ'):
        engine.load_checkpoint(tmp_path, state['binding'])
    orphan = tmp_path / 'orphan'; orphan.mkdir()
    (orphan / 'epoch_001.pt').write_bytes(b'partial')
    with pytest.raises(ValueError, match='Uncommitted'):
        engine.load_checkpoint(orphan, state['binding'])


def test_partial_is_not_a_teacher_and_deadline_reserves_whole_epoch():
    with pytest.raises(PermissionError, match='intermediate'):
        engine.kernel_report({'complete': False})
    assert runtime.can_start({'history': []}, 8000, now=0)
    assert not runtime.can_start({'history': []}, 7000, now=0)
    assert not runtime.can_start({'history': [dict(train_seconds=1000, validation_seconds=100)]}, 3400, now=0)


def small_spec(tmp_path):
    return campaign.artifact('CAMPAIGN_SPEC', campaign_root=str(tmp_path/'new'), project_dir=str(tmp_path/'code'),
        source_commit='a'*40, tasks=campaign.graph(), resources=campaign.RESOURCES,
        old_jobs={name: str(100+i) for i, name in enumerate(campaign.REMAINING)},
        allowed_partitions=['debug', 'tier3'])


def test_graph_limits_dependencies_and_only_remaining_fits(tmp_path):
    spec = small_spec(tmp_path)
    rows = campaign.plan(spec, 'science')['commands']
    assert len(rows) == 13
    assert len(campaign.plan(spec, 'gate')['commands']) == 2
    assert len(campaign.plan(spec, 'full')['commands']) == 15
    assert sum(r['task_id'].startswith('train_') for r in rows) == 9
    assert not any('assign' in r['task_id'] or 'D050_part' in r['task_id'] for r in rows)
    for row in rows:
        cmd = row['command']
        assert '--partition=debug' in cmd
        assert '--no-requeue' in cmd
        assert not any('afterany' in arg for arg in cmd)
        if row['task_id'].startswith('train_'):
            assert '--time=1380' in cmd
        if row['task_id'].startswith('reduce_'):
            assert row['dependencies'][0].endswith('part3')
            assert '--time=360' in cmd
    assert rows[0]['dependencies'] == []  # Staged science starts only after authenticated gate.


@pytest.mark.parametrize('state', ['RUNNING', 'COMPLETED', 'FAILED', 'UNKNOWN'])
def test_retirement_never_cancels_nonpending_jobs(tmp_path, monkeypatch, state):
    spec = small_spec(tmp_path)
    monkeypatch.setattr(campaign, 'validate_spec', lambda *a: None)
    monkeypatch.setattr(campaign, 'accounting', lambda ids: {i: (state, '0:0') for i in ids})
    monkeypatch.setattr(campaign.subprocess, 'run', lambda *a, **k: pytest.fail('No cancellation allowed'))
    with pytest.raises(PermissionError):
        campaign.retire(spec, execute=True, authorization=campaign.RETIRE)


def test_retirement_is_dry_default_and_state_filtered(tmp_path, monkeypatch):
    spec = small_spec(tmp_path)
    monkeypatch.setattr(campaign, 'validate_spec', lambda *a: None)
    states = {i: ('PENDING', '0:0') for i in spec['old_jobs'].values()}
    monkeypatch.setattr(campaign, 'accounting', lambda ids: states)
    calls = []
    def cancel(cmd, **kw):
        calls.append(cmd)
        for job in spec['old_jobs'].values(): states[job] = ('CANCELLED', '0:0')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(campaign.subprocess, 'run', cancel)
    campaign.retire(spec)
    assert not calls
    campaign.retire(spec, execute=True, authorization=campaign.RETIRE)
    assert calls == [['scancel', '--state=PENDING', *spec['old_jobs'].values()]]


def test_dry_plan_no_scheduler_calls_and_live_requires_gate_policy_retirement(tmp_path, monkeypatch):
    spec = small_spec(tmp_path)
    monkeypatch.setattr(campaign, 'validate_spec', lambda *a: None)
    monkeypatch.setattr(campaign.subprocess, 'run', lambda *a, **k: pytest.fail('No Slurm call'))
    campaign.submit(spec, stage='full')
    with pytest.raises(PermissionError):
        campaign.submit(spec, stage='full', execute=True, authorization=campaign.AUTHORIZE)
    with pytest.raises(PermissionError, match='Operator must confirm'):
        campaign.submit(spec, stage='gate', execute=True, authorization=campaign.AUTHORIZE)
    monkeypatch.setattr(campaign, 'require_retired', lambda s: None)
    with pytest.raises(PermissionError, match='Fresh native resume'):
        campaign.submit(spec, stage='science', execute=True, authorization=campaign.AUTHORIZE, debug_policy_confirmed=True)


def test_original_k2_recipe_unchanged():
    spec = k2.registration()
    assert spec['rolling_resume'] is False
    assert spec['resources']['train']['minutes'] == 5760
    assert spec['training']['batch_size'] == 128
    assert spec['training']['maximum_passes'] == 100


def test_resume_rejects_execution_precision_drift():
    train, val, node, q = setup_fit()
    kwargs = dict(node=node, device='cpu', teacher=q, binding={}, maximum=3, acceptance=True)
    state = engine.train_segment(DropoutModel(), train, val, **kwargs, on_epoch=lambda _: True)
    state['backend']['cudnn_benchmark'] = not state['backend']['cudnn_benchmark']
    with pytest.raises(ValueError, match='precision'):
        engine.train_segment(DropoutModel(), train, val, **kwargs, resume=state)


def test_auto_submission_requires_durable_operator_authorization(tmp_path):
    spec = small_spec(tmp_path)
    with pytest.raises(PermissionError, match='operator authorization'):
        campaign.require_authorization(spec)
    path = tmp_path / 'new/authorization.json'
    write_immutable_json(path, campaign.authorization_record(spec))
    campaign.require_authorization(spec)
    changed = {**spec, 'content_hash': 'b'*64}
    with pytest.raises(PermissionError):
        campaign.require_authorization(changed)


@pytest.mark.parametrize('partition', ['debug', 'tier3'])
def test_worker_accepts_only_partition_moves_with_exact_resources(tmp_path, monkeypatch, partition):
    spec = small_spec(tmp_path)
    name = 'train_CONCAT_K2_D025_part1'
    monkeypatch.setattr(campaign, '_journal', lambda *a, **k: ([], {name: '123'}))
    monkeypatch.setattr(campaign, 'allocation', lambda *a: ('123', 4, 320000))
    monkeypatch.setattr(campaign.sys, 'prefix', '/home/ryreu/miniconda3/envs/atlas_kd_sporc')
    for key, value in dict(SLURM_JOB_ID='123', SLURM_CLUSTER_NAME='sporc',
            SLURM_JOB_PARTITION=partition, SLURM_MEM_PER_NODE='320000', PYTHONNOUSERSITE='1').items():
        monkeypatch.setenv(key, value)
    fields = dict(JobId='123', Partition=partition, Account='reu-aisocial', QOS='qos_tier3',
        NumCPUs='4', NumNodes='1', NumTasks='1', TimeLimit='23:00:00', RunTime='00:01:00', NodeList='skl-test')
    monkeypatch.setattr(campaign.subprocess, 'run', lambda *a, **k:
        SimpleNamespace(stdout=' '.join(f'{k}={v}' for k, v in fields.items())))
    assert campaign.authenticate_job(spec, name) > 0
    assert load_json(tmp_path/'new/execution'/f'{name}.json')['actual_partition'] == partition
    fields['TimeLimit'] = '24:00:00'
    with pytest.raises(PermissionError, match='allocation'):
        campaign.authenticate_job(spec, name)
    fields['TimeLimit'], fields['Partition'] = '23:00:00', 'tigris'
    with pytest.raises(PermissionError, match='allocation'):
        campaign.authenticate_job(spec, name)


def test_checkpoint_chain_rejects_reordered_parent_and_missing_epoch(tmp_path):
    train, val, node, q = setup_fit()
    previous = None
    def save(state):
        nonlocal previous
        previous = engine.write_checkpoint(tmp_path, state, previous=previous)['content_hash']
    state = engine.train_segment(DropoutModel(), train, val, node=node, device='cpu', teacher=q,
        binding={}, maximum=3, acceptance=True, on_epoch=save)
    restored, _ = engine.load_checkpoint(tmp_path, state['binding'])
    assert restored['complete']
    path = tmp_path/'epoch_002.json'
    receipt = load_json(path)
    receipt.pop('content_hash')
    receipt['previous_sha256'] = 'f'*64
    import json
    path.write_text(json.dumps(with_content_hash(receipt)))
    with pytest.raises(ValueError, match='chain'):
        engine.load_checkpoint(tmp_path, state['binding'])


def test_full_three_fit_handoff_uses_final_banks_not_partial_weights(tmp_path, monkeypatch):
    """Exercise real reports, state IO, task parents and T2 banks, tiny CPU data."""
    from pathlib import Path
    from hlt_classification.data.cache_contracts import sha256_file
    train, val, _, _ = setup_fit()
    root, old = tmp_path/'new', tmp_path/'old'
    old.mkdir()
    write_immutable_json(old/'validation_partition.json', {'content_hash': 'v'*64})
    source = dict(campaign_root=str(old), content_hash='o'*64, source_commit=campaign.DONOR_COMMIT,
        nodes=k2.nodes(), training=k2.registration()['training'], foundation=dict(content_hash='f'*64, inputs={}))
    spec = dict(small_spec(tmp_path), original_commit=campaign.DONOR_COMMIT,
        training=source['training'], source_spec=dict(content_hash=source['content_hash']),
        maximum_checkpoint_bytes=32*1024**3, minimum_free_bytes=0, reserve_seconds=1800)
    monkeypatch.setattr(runtime, 'validate_spec', lambda s: source)
    accepted = dict(gpu={}, environment={}, checkpoint_bytes_upper_bound=128000)
    monkeypatch.setattr(runtime, 'gate', lambda s: accepted)
    monkeypatch.setattr(runtime, 'gpu_identity', lambda: {})
    monkeypatch.setattr(runtime, 'installed_environment', lambda: {})
    def auth(s, name):
        write_immutable_json(root/'execution'/f'{name}.json', campaign.artifact('EXECUTION', task_id=name))
        return 1e20
    monkeypatch.setattr(runtime, 'authenticate_job', auth)
    def caches(s, node, train_only=False):
        return dict(train=train, checkpoint=val, report=val)
    monkeypatch.setattr(runtime.legacy, 'caches', caches)
    def model(n):
        torch.manual_seed(n['initialization_seed'])
        return DropoutModel()
    monkeypatch.setattr(runtime.legacy, 'new_model', model)
    old_q = np.eye(11, dtype=np.float32)[train.labels]
    monkeypatch.setattr(runtime.legacy, 'teacher', lambda s, n, ids: (old_q, {'bank_manifest_sha256': 'a'*64}))
    metric = dict(macro_ovr_auc=.5, cross_entropy=1., macro_mean_log_qcd_rejection_at_50pct_signal=None)
    monkeypatch.setattr(donor, 'evaluate_probabilities', lambda *a: deepcopy(metric))
    acceptance = root/'preflight/acceptance.json'
    write_immutable_json(acceptance, {'passed': True})
    write_immutable_json(root/'tasks/preflight.json', campaign.artifact('TASK_REPORT',
        campaign_sha256=spec['content_hash'], task_id='preflight', parents={}, result={},
        executor_source=spec['source_commit'], original_scientific_source=campaign.DONOR_COMMIT,
        outputs=[dict(path='preflight/acceptance.json', sha256=sha256_file(acceptance))], final_test_accessed=False))
    for row in campaign.graph()[1:-2]:
        if row['kind'] == 'segment':
            limit = 25 * row['segment']
            monkeypatch.setattr(runtime, 'can_start', lambda state, *a, **k: state['pass_number'] < limit)
        receipt = runtime.run(spec, row['task_id'], device='cpu')
        assert runtime.completed(spec, row['task_id']) == receipt
        if row['kind'] == 'segment':
            assert receipt['result']['passes'] == row['segment'] * 25
            assert receipt['result']['fit_complete'] is (row['segment'] == 3)
            if row['segment'] < 3:
                assert not (root/'training'/row['node_id']/'training_report.json').exists()
            else:
                report = load_json(root/receipt['result']['training_report'])
                assert report['kernel_report']['passes'] == 75
                assert report['kernel_report']['scientific_fit']
                assert report['kernel_report']['selected_pass'] == 1
        else:
            assert (root/receipt['result']['bank']/'manifest.json').is_file()
    for name in campaign.NODES:
        assert len(list((root/'checkpoints'/name).glob('epoch_*.json'))) == 75
    assert list(old.iterdir()) == [old/'validation_partition.json']  # Source never written.


def test_spare_segment_does_not_load_data_or_train(tmp_path, monkeypatch):
    spec = small_spec(tmp_path)
    source = dict(nodes=k2.nodes())
    result = dict(fit_complete=True, spare_segment=False, passes=75)
    monkeypatch.setattr(runtime, 'completed', lambda *a: dict(result=result,
        outputs=[dict(path='training/selected.pt'), dict(path='execution/previous.json')]))
    monkeypatch.setattr(runtime.legacy, 'caches', lambda *a, **k: pytest.fail('No spare cache build'))
    row = next(r for r in campaign.graph() if r['task_id'] == 'train_CONCAT_K2_D025_part2')
    got, paths = runtime.fit_segment(spec, source, row, 0, 'cpu')
    assert got['spare_segment'] and got['passes'] == 75
    assert len(paths) == 1


def child_resume(checkpoint_directory, output_path):
    """Used by a fresh interpreter, not by a forked in-memory model."""
    torch.set_num_threads(1)
    train, val, node, q = setup_fit()
    binding = runtime.full_binding({'child_test': True}, node, train, val, maximum=3, acceptance=True)
    state, _ = engine.load_checkpoint(checkpoint_directory, binding)
    random.seed(99); np.random.seed(101); torch.manual_seed(103)
    result = engine.train_segment(DropoutModel(), train, val, node=node, device='cpu',
        teacher=q, binding={'child_test': True}, maximum=3, acceptance=True, resume=state)
    torch.save(result, output_path)


def test_resume_in_fresh_python_process(tmp_path):
    from pathlib import Path
    import subprocess
    import sys
    import os
    train, val, node, q = setup_fit()
    model = DropoutModel()
    initial = engine.cpu_tree(model.state_dict()), engine.rng_state()
    kwargs = dict(node=node, device='cpu', teacher=q, binding={'child_test': True}, maximum=3, acceptance=True)
    full = engine.train_segment(model, train, val, **kwargs)
    model = DropoutModel(); model.load_state_dict(initial[0]); engine.restore_rng(initial[1])
    part = engine.train_segment(model, train, val, **kwargs, on_epoch=lambda _: True)
    engine.write_checkpoint(tmp_path/'checkpoints', part)
    output = tmp_path/'resumed.pt'
    code = (
        'import sys; sys.path.insert(0, sys.argv[1]); import hlt_classification; '
        'hlt_classification.__path__.append(sys.argv[2]); sys.path.insert(0, sys.argv[3]); '
        'from test_k2_segmented import child_resume; child_resume(sys.argv[4], sys.argv[5])'
    )
    result = subprocess.run([sys.executable, '-s', '-c', code,
        str(Path(donor.__file__).resolve().parents[2]), str(Path(engine.__file__).resolve().parents[1]),
        str(Path(__file__).resolve().parent), str(tmp_path/'checkpoints'), str(output)],
        check=True, text=True, capture_output=True, timeout=90,
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    resumed = torch.load(output, map_location='cpu', weights_only=True)
    assert result.returncode == 0
    for key in full.keys() - {'history', 'runtime_seconds'}:
        runtime.assert_state_equal(full[key], resumed[key])
    runtime.assert_state_equal(runtime.scientific_history(full['history']), runtime.scientific_history(resumed['history']))

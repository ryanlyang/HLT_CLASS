"""Synthetic CPU evidence only; never substitutes for the real 128-GiB gate."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hlt_classification.data.cache_contracts import load_json, sha256_file, with_content_hash, write_immutable_json
from hlt_classification.k2_segmented import campaign, memory_migration as migration, runtime, training
from hlt_classification.scouting.hcwdl_recovery import build_submission_event, assemble_submission_ledger
from test_k2_segmented import DropoutModel, setup_fit
from hlt_classification.jetclass2_delphes import concat_k2_campaign as k2
from hlt_classification.jetclass2_delphes import salience_learned_training as donor_kernel


def rehash(value):
    return with_content_hash({k: v for k, v in value.items() if k != 'content_hash'})


def publish_ledger(spec, stage):
    rows, events, jobs = campaign.plan(spec, stage)['commands'], [], {}
    for index, row in enumerate(rows):
        job = str((1000 if stage == 'gate' else 2000) + index)
        command = campaign._resolved(row, jobs)
        event = build_submission_event(campaign_spec_sha256=spec['content_hash'], task_id=row['task_id'],
            job_id=job, command=command, sequence=index)
        events.append(event)
        jobs[row['task_id']] = job
    value = assemble_submission_ledger(events, campaign_spec_sha256=spec['content_hash'])
    write_immutable_json(Path(spec['campaign_root']) / f'submissions_{stage}/submission_ledger.json', value)
    return value


def preflight_receipt(spec, **extra):
    root = Path(spec['campaign_root'])
    evidence = campaign.artifact('ACCEPTANCE', campaign_sha256=spec['content_hash'], passed=True,
        uninterrupted_resumed_parity=True, legacy_kernel_parity=True, final_test_accessed=False,
        gpu={}, environment={}, checkpoint_bytes_upper_bound=128000,
        peak_cpu_bytes=extra.pop('peak_cpu_bytes', 35*1024**3), **extra)
    path = root / 'preflight/acceptance.json'
    write_immutable_json(path, evidence)
    receipt = campaign.artifact('TASK_REPORT', campaign_sha256=spec['content_hash'], task_id='preflight',
        parents={}, result=dict(acceptance='preflight/acceptance.json'), executor_source=spec['source_commit'],
        original_scientific_source=campaign.DONOR_COMMIT,
        outputs=[dict(path='preflight/acceptance.json', sha256=sha256_file(path))], final_test_accessed=False)
    write_immutable_json(root / 'tasks/preflight.json', receipt)


@pytest.fixture
def prepared(tmp_path, monkeypatch, request):
    torch.set_num_threads(1)
    train, val, _, _ = setup_fit()
    old_root, project, new_project = tmp_path/'donor', tmp_path/'old_code', tmp_path/'new_code'
    relative = 'src/hlt_classification/k2_segmented/training.py'
    for p in (project, new_project):
        (p/relative).parent.mkdir(parents=True)
        (p/relative).write_bytes(Path(training.__file__).read_bytes())
    source = with_content_hash(dict(campaign_root=str(tmp_path/'original'), project_dir=str(tmp_path/'scientific_code'),
        data_root=str(tmp_path/'readonly_data'), nodes=k2.nodes(), source_commit=campaign.DONOR_COMMIT,
        training=k2.registration()['training'], role_counts=dict(train=len(train), validation=len(val)),
        foundation=dict(content_hash='f'*64, inputs={})))
    source_path = Path(source['campaign_root'])/'campaign_spec.json'
    write_immutable_json(source_path, source)
    write_immutable_json(source_path.parent/'validation_partition.json', dict(content_hash='v'*64))
    old_jobs = {name: str(100+i) for i, name in enumerate(campaign.REMAINING)}
    donor = campaign.artifact('CAMPAIGN_SPEC', campaign_root=str(old_root), project_dir=str(project),
        source_commit=migration.EXECUTOR_DONOR, source_spec=migration.descriptor(source_path),
        original_commit=campaign.DONOR_COMMIT, original_project_dir=source['project_dir'],
        source_ledger_sha256='l'*64, imported_receipts={}, old_jobs=old_jobs,
        tasks=campaign.graph(), resources=campaign.RESOURCES, training=source['training'], role_counts=source['role_counts'],
        segments_per_fit=3, segment_minutes=1380, reserve_seconds=1800,
        maximum_checkpoint_bytes=32*1024**3, minimum_free_bytes=4*1024**3,
        initial_partition='debug', allowed_partitions=['debug', 'tier3'], final_test_accessed=False,
        original_artifacts_read_only=True)
    write_immutable_json(old_root/'campaign_spec.json', donor)
    # Only external immutable original campaign/source authentication is doubled.
    # Real v1/v2 validators, ledgers, task receipts, state IO and trainer run below.
    monkeypatch.setattr(campaign, '_source', lambda *a: None)
    monkeypatch.setattr(campaign, 'validate_campaign', lambda *a: None)
    monkeypatch.setattr(campaign, 'source_ledger', lambda *a: dict(content_hash='l'*64, jobs=old_jobs))
    monkeypatch.setattr(campaign, 'verify_prefix', lambda *a: {})
    preflight_receipt(donor)
    gates, science = publish_ledger(donor, 'gate'), publish_ledger(donor, 'science')
    states = {job: ('PENDING', '0:0') for job in science['jobs'].values()}
    states[gates['jobs']['preflight']] = ('COMPLETED', '0:0')
    for name in migration.PREFIX[1:]:
        states[science['jobs'][name]] = ('COMPLETED', '0:0')
    monkeypatch.setattr(campaign, 'accounting', lambda ids: {i: states[i] for i in ids})
    monkeypatch.setattr(runtime, 'gpu_identity', lambda: {})
    monkeypatch.setattr(runtime, 'installed_environment', lambda: {})
    def auth(s, name):
        write_immutable_json(Path(s['campaign_root'])/'execution'/f'{name}.json',
            campaign.artifact('EXECUTION', task_id=name))
        return 1e20
    monkeypatch.setattr(runtime, 'authenticate_job', auth)
    monkeypatch.setattr(runtime.legacy, 'caches', lambda *a, **k: dict(train=train, checkpoint=val, report=val))
    def model(node):
        torch.manual_seed(node['initialization_seed'])
        return DropoutModel()
    monkeypatch.setattr(runtime.legacy, 'new_model', model)
    teacher = dict(content_hash='t'*64, result=dict(bank_manifest_sha256='b'*64))
    monkeypatch.setattr(runtime.legacy, 'completed', lambda *a: teacher)
    lineage = dict(task_sha256='t'*64, teacher_node='CONCAT_K2_D050', bank_manifest_sha256='b'*64)
    q = np.eye(11, dtype=np.float32)[train.labels]
    monkeypatch.setattr(runtime.legacy, 'teacher', lambda *a: (q, lineage))
    metric = dict(macro_ovr_auc=.5, cross_entropy=1., macro_mean_log_qcd_rejection_at_50pct_signal=None)
    monkeypatch.setattr(donor_kernel, 'evaluate_probabilities', lambda *a: deepcopy(metric))
    for part, limit in ((1, 2), (2, getattr(request, 'param', 4))):
        monkeypatch.setattr(runtime, 'can_start', lambda state, *a, _limit=limit, **k: state['pass_number'] < _limit)
        runtime.run(donor, f'train_CONCAT_K2_D025_part{part}', device='cpu')
    return SimpleNamespace(donor=donor, source=source, root=old_root, new_root=tmp_path/'new',
        new_project=new_project, states=states, train=train, val=val)


def create(p):
    return migration.create(donor_spec=p.root/'campaign_spec.json', campaign_root=p.new_root,
                            project_dir=p.new_project, source_commit='e'*40)


def test_registration_and_dry_plans_keep_old_jobs_science_and_resources(prepared):
    p = prepared
    before = {str(f): sha256_file(f) for f in p.root.rglob('*') if f.is_file()}
    spec = create(p)
    assert spec['contract'] == 'K2_SEGMENTED_CAMPAIGN_SPEC/v2'
    assert spec['resume_import']['checkpoint']['pass_number'] == 4
    assert len(spec['old_jobs']) == 11
    assert not any(name in spec['old_jobs'] for name in migration.PREFIX)
    assert campaign.validate_spec(spec) == p.source
    assert spec['training'] == p.donor['training']
    assert spec['resources']['segment']['memory_mb'] == 131072
    assert p.donor['resources']['segment']['memory_mb'] == 320000
    assert len(campaign.plan(spec, 'full')['commands']) == 13
    rows = campaign.plan(spec, 'science')['commands']
    assert len(rows) == 11 and rows[0]['task_id'] == migration.FIRST_TASK
    for row in rows:
        assert '--partition=debug' in row['command']
        if '--gres=gpu:a100:1' in row['command']:
            assert '--mem=131072M' in row['command']
        if row['task_id'].startswith('train_'):
            assert '--time=1380' in row['command']
    assert {str(f): sha256_file(f) for f in p.root.rglob('*') if f.is_file()} == before


def test_byte_exact_migration_and_real_resume_to_same_final_state(prepared, monkeypatch):
    p = prepared
    spec = create(p)
    before = {str(f): sha256_file(f) for f in p.root.rglob('*') if f.is_file()}
    migration.materialize(spec)
    migration.verify_copy(spec)
    for row in spec['resume_import']['checkpoint']['files']:
        assert sha256_file(p.new_root/row['path']) == sha256_file(p.root/row['path'])
        assert (p.new_root/row['path']).stat().st_ino != (p.root/row['path']).stat().st_ino
    preflight_receipt(spec, memory_request_mb=131072, resume_import_sha256=spec['resume_import']['content_hash'])
    monkeypatch.setattr(runtime, 'can_start', lambda *a, **k: True)
    migrated = runtime.run(spec, migration.FIRST_TASK, device='cpu')
    assert migrated['result']['fit_complete']
    assert {str(f): sha256_file(f) for f in p.root.rglob('*') if f.is_file()} == before
    original = runtime.run(p.donor, migration.FIRST_TASK, device='cpu')
    binding = spec['resume_import']['checkpoint']['binding']
    left, _ = training.load_checkpoint(p.root/'checkpoints'/migration.NODE, binding)
    right, _ = training.load_checkpoint(p.new_root/'checkpoints'/migration.NODE, binding)
    assert left['pass_number'] == right['pass_number'] == 75
    for key in left.keys() - {'history', 'runtime_seconds'}:
        runtime.assert_state_equal(left[key], right[key])
    runtime.assert_state_equal(runtime.scientific_history(left['history']), runtime.scientific_history(right['history']))
    assert original['result']['passes'] == migrated['result']['passes']


def test_migrated_remainder_publishes_banks_and_single_view_deployment(prepared, monkeypatch):
    p = prepared
    spec = create(p)
    migration.materialize(spec)
    preflight_receipt(spec, memory_request_mb=131072, resume_import_sha256=spec['resume_import']['content_hash'])
    monkeypatch.setattr(runtime, 'can_start', lambda *a, **k: True)
    before = {str(f): sha256_file(f) for f in p.root.rglob('*') if f.is_file()}
    for row in campaign.graph(spec)[1:-2]:
        receipt = runtime.run(spec, row['task_id'], device='cpu')
        assert runtime.completed(spec, row['task_id']) == receipt
        if row['kind'] == 'segment':
            assert receipt['result']['fit_complete'] and receipt['result']['passes'] == 75
        else:
            assert (p.new_root/receipt['result']['bank']/'manifest.json').is_file()
    for node in campaign.NODES:
        report = load_json(p.new_root/'training'/node/'training_report.json')
        assert report['campaign_sha256'] == spec['content_hash']
        assert report['kernel_report']['scientific_fit'] and not report['final_test_accessed']
        if node != migration.NODE:
            assert report['teacher_lineage']['task_sha256'] == runtime.completed(
                spec, 'reduce_' + report['node']['teacher_distribution'])['content_hash']
    deployment = load_json(p.new_root/'training/HLT_X1_COMPRESSED/deployment.json')
    assert deployment['native_hlt_copies'] == 1 and not deployment['offline_inputs']
    assert {str(f): sha256_file(f) for f in p.root.rglob('*') if f.is_file()} == before


@pytest.mark.parametrize('prepared', [75], indirect=True)
def test_already_complete_part2_is_imported_without_training_again(prepared, monkeypatch):
    p = prepared
    spec = create(p)
    assert spec['resume_import']['checkpoint']['fit_complete']
    migration.materialize(spec)
    preflight_receipt(spec, memory_request_mb=131072, resume_import_sha256=spec['resume_import']['content_hash'])
    monkeypatch.setattr(runtime.legacy, 'caches', lambda *a, **k: pytest.fail('Spare must not rebuild science caches'))
    receipt = runtime.run(spec, migration.FIRST_TASK, device='cpu')
    assert receipt['result']['spare_segment'] and receipt['result']['passes'] == 75
    report_path = receipt['result']['training_report']
    assert sha256_file(p.new_root/report_path) == sha256_file(p.root/report_path)
    assert runtime.completed(spec, migration.FIRST_TASK) == receipt


@pytest.mark.parametrize('kind', ['RUNNING', 'COMPLETED', 'FAILED', 'UNKNOWN'])
def test_no_cutover_if_remainder_not_pending(prepared, kind):
    p = prepared
    science = migration.ledger(p.donor, 'science')
    p.states[science['jobs'][migration.FIRST_TASK]] = (kind, '0:0')
    with pytest.raises(PermissionError, match='pending or cancelled'):
        create(p)
    assert not p.new_root.exists()


@pytest.mark.parametrize('kind', ['claims', 'execution', 'tasks'])
def test_donor_advancement_rejected_even_before_accounting_changes(prepared, kind):
    p = prepared
    suffix = '.claim' if kind == 'claims' else '.json'
    path = p.root/kind/(migration.FIRST_TASK + suffix)
    path.parent.mkdir(exist_ok=True)
    path.write_text('{}')
    with pytest.raises(PermissionError, match='started/advanced'):
        create(p)


@pytest.mark.parametrize('kind', ['past_payload', 'orphan', 'receipt', 'training_source'])
def test_migration_rejects_corruption_or_unreviewed_kernel(prepared, kind):
    p = prepared
    directory = p.root/'checkpoints'/migration.NODE
    if kind == 'past_payload':
        (directory/'epoch_001.pt').write_bytes(b'corrupt')
    elif kind == 'orphan':
        (directory/'epoch_005.pt').write_bytes(b'incomplete')
    elif kind == 'receipt':
        (p.root/'tasks'/f'{migration.PREFIX[-1]}.json').write_text('{}')
    else:
        (p.new_project/'src/hlt_classification/k2_segmented/training.py').write_text('changed')
    with pytest.raises((ValueError, KeyError)):
        create(p)


def test_partial_copy_idempotent_and_conflict_never_overwritten(prepared):
    p = prepared
    spec = create(p)
    migration.materialize(spec)
    migration.materialize(spec)
    target = p.new_root/'checkpoints'/migration.NODE/'epoch_001.pt'
    target.write_bytes(b'conflict')
    with pytest.raises(ValueError, match='conflict'):
        migration.materialize(spec)
    assert target.read_bytes() == b'conflict'


def test_wrong_population_or_binding_is_rejected_after_import(prepared):
    p = prepared
    spec = create(p)
    migration.materialize(spec)
    binding = deepcopy(spec['resume_import']['checkpoint']['binding'])
    binding['identities']['train'] = '9'*64
    with pytest.raises(ValueError, match='identity'):
        training.load_checkpoint(p.new_root/'checkpoints'/migration.NODE, binding)


@pytest.mark.parametrize('partition', ['debug', 'tier3'])
def test_128g_worker_exact_request_and_partition_portability(prepared, monkeypatch, partition):
    p = prepared
    spec = create(p)
    name = migration.FIRST_TASK
    monkeypatch.setattr(campaign, '_journal', lambda *a, **k: ([], {name: '123'}))
    monkeypatch.setattr(campaign, 'allocation', lambda *a: ('123', 4, 131072))
    monkeypatch.setattr(campaign.sys, 'prefix', '/home/ryreu/miniconda3/envs/atlas_kd_sporc')
    for key, value in dict(SLURM_JOB_ID='123', SLURM_CLUSTER_NAME='sporc', SLURM_JOB_PARTITION=partition,
                          SLURM_MEM_PER_NODE='131072', PYTHONNOUSERSITE='1').items():
        monkeypatch.setenv(key, value)
    fields = dict(JobId='123', Partition=partition, Account='reu-aisocial', QOS='qos_tier3', NumCPUs='4',
                  NumNodes='1', NumTasks='1', TimeLimit='23:00:00', RunTime='00:01:00', NodeList='synthetic')
    monkeypatch.setattr(campaign.subprocess, 'run', lambda *a, **k:
                        SimpleNamespace(stdout=' '.join(f'{k}={v}' for k, v in fields.items())))
    assert campaign.authenticate_job(spec, name) > 0
    monkeypatch.setenv('SLURM_MEM_PER_NODE', '65536')
    with pytest.raises(PermissionError, match='allocation'):
        campaign.authenticate_job(spec, name)


def test_gate_requires_new_memory_bound_and_resume_identity(prepared):
    p = prepared
    spec = create(p)
    preflight_receipt(spec)  # A generic v1 acceptance cannot release migrated science.
    with pytest.raises(ValueError, match='128-GiB'):
        campaign.gate(spec)


@pytest.mark.parametrize('wrong', [dict(memory_request_mb=320000), dict(resume_import_sha256='9'*64),
                                  dict(peak_cpu_bytes=105*1024**3), dict(peak_cpu_bytes=0)])
def test_migration_gate_rejects_stale_or_insufficient_resource_evidence(prepared, wrong):
    spec = create(prepared)
    evidence = dict(memory_request_mb=131072, resume_import_sha256=spec['resume_import']['content_hash'])
    evidence.update(wrong)
    preflight_receipt(spec, **evidence)
    with pytest.raises(ValueError, match='128-GiB'):
        campaign.gate(spec)


def test_fresh_gate_and_confirmed_retirement_are_both_mandatory(prepared):
    spec = create(prepared)
    kwargs = dict(stage='science', execute=True, authorization=campaign.AUTHORIZE, debug_policy_confirmed=True)
    with pytest.raises(PermissionError, match='retired'):
        campaign.submit(spec, **kwargs)
    for job in spec['old_jobs'].values():
        prepared.states[job] = ('CANCELLED', '0:0')
    with pytest.raises(PermissionError, match='Fresh native resume gate'):
        campaign.submit(spec, **kwargs)
    with pytest.raises(PermissionError, match='never live full-DAG'):
        campaign.submit(spec, **dict(kwargs, stage='full'))


def test_retirement_race_never_cancels_a_running_job(prepared, monkeypatch):
    spec = create(prepared)
    job = spec['old_jobs'][migration.FIRST_TASK]
    def raced_cancel(cmd, **kw):
        assert cmd[:2] == ['scancel', '--state=PENDING']
        for other in cmd[2:]:
            prepared.states[other] = ('CANCELLED', '0:0')
        prepared.states[job] = ('RUNNING', '0:0')
    monkeypatch.setattr(campaign.subprocess, 'run', raced_cancel)
    monkeypatch.setattr(campaign.time, 'sleep', lambda *a: None)
    with pytest.raises(PermissionError, match='retired'):
        campaign.retire(spec, execute=True, authorization=campaign.RETIRE)
    assert prepared.states[job][0] == 'RUNNING'
    assert not (prepared.new_root/'retirement.json').exists()


def test_retirement_uses_only_eleven_unfinished_ids(prepared, monkeypatch):
    spec = create(prepared)
    calls = []
    def cancel(cmd, **kw):
        calls.append(cmd)
        for job in cmd[2:]:
            prepared.states[job] = ('CANCELLED', '0:0')
    monkeypatch.setattr(campaign.subprocess, 'run', cancel)
    campaign.retire(spec)
    assert not calls
    campaign.retire(spec, execute=True, authorization=campaign.RETIRE)
    assert calls == [['scancel', '--state=PENDING', *spec['old_jobs'].values()]]
    assert all(prepared.states[j] == ('COMPLETED', '0:0')
               for j in spec['resume_import']['completed_jobs'].values())

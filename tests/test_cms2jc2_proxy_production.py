"""Production boundary, deterministic physical bank and scheduler safety tests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import uproot

from hlt_classification.cms2jc2_production import (
    contracts as k, population as p, campaign as c, storage as s, output as o,
    worker as w, submission as sub,
)
from hlt_classification.cms2jc2_response import generation_benchmark_engine as e
from hlt_classification.cms2jc2_response.contracts import artifact as old_artifact, with_content_hash
from hlt_classification.cms2jc2_response.readers import Pair
from hlt_classification.cms2jc2_response.assumptions import provisional_compatibility
from hlt_classification.jetclass2_delphes.split_registry import pack_entries, select_profile
from test_cms2jc2_generation_benchmark import bundle, sample, native_threads
from test_cms2jc2_bounded import LocalMeasurement
from test_jetclass2_delphes_split_registry import registered
from test_jetclass2_delphes import snapshot


@pytest.fixture
def population(monkeypatch):
    monkeypatch.setattr(p, 'COUNTS', dict(train=160, validation=16, final_test=32))
    monkeypatch.setattr(p, 'PILOT_JETS', 64)
    monkeypatch.setattr(p, 'SHARD_JETS', 16)
    files = [dict(path=f'source{i}/data.root', source=f'source{i}', sha256=f'{i+1:064x}',
                  tree_key='tree;1', entries=128) for i in range(6)]
    inv = old_artifact('FIXTURE', files=files)
    members = {}
    groups = []
    for role, indexes, count in (('train', (0, 1), 80), ('validation', (2, 3), 24), ('final_test', (4, 5), 16)):
        members[role] = dict(files=[dict(path=files[i]['path'], entry_mask=pack_entries(range(count), 128)) for i in indexes])
        groups.extend(dict(path=files[i]['path'], role=role) for i in indexes)
    profile = old_artifact('FIXTURE', profile='TRAIN_1M', registry_sha256='f'*64,
                          memberships=members, groups=groups)
    donor = deepcopy(profile)
    donor['profile'] = 'TRAIN_500K'
    for row in donor['memberships']['train']['files']:
        row['entry_mask'] = pack_entries(range(40), 128)
    donor = with_content_hash(donor)
    monkeypatch.setattr(p, 'validate_inventory', lambda inv: inv['content_hash'])
    monkeypatch.setattr(p, 'validate_split_profile', lambda *a: None)
    return inv, profile, donor, p.build(inv, profile, donor)


@pytest.fixture
def study(tmp_path, monkeypatch, population, bundle):
    inv, profile, donor, population = population
    root = tmp_path/'production'; root.mkdir()
    monkeypatch.setattr(s, 'check_space', lambda *a, **k: None)
    source = k.artifact('SOURCE', commit='a'*40, files={}, executable=True)
    environment = old_artifact('NUMERICAL_ENVIRONMENT', fixture=True)
    values = dict(bundle=bundle, inventory=inv, profile=profile, donor_profile=donor,
        tigris_gate=dict(serial=dict(output_bytes=1024)))
    imports = {}
    for name, value in values.items():
        relative = f'provenance/{name}.json'
        ref = k.write(root/relative, value)
        imports[name] = dict(relative=relative, sha256=ref['sha256'], bytes=ref['bytes'])
    value = k.artifact('STUDY', parents={'source': source['content_hash'],
        'population': population['content_hash'], 'confirmation': 'b'*64},
        root=str(root), project_dir=str(tmp_path/'project'),
        source=source, numerical_environment=environment, population=population,
        shards=p.shards(population), counts=p.COUNTS, candidate='JOINT', replica=0,
        generation_workers=36, max_concurrent=16, physics_production_qualified=False,
        dataset_kind='controlled_CMS_calibrated_proxy', imports=imports,
        data_root=str(tmp_path/'raw'), review=provisional_compatibility(inventory_hash='a'*64),
        confirmation_status='inconclusive', reviewed_confirmation_hash='b'*64,
        storage=dict(persistent_attested=True, budget_bytes=50*k.GIB, available_quota_bytes=60*k.GIB))
    k.write(root/'study_spec.json', value)
    monkeypatch.setattr(c, 'source_snapshot', lambda *a: source)
    monkeypatch.setattr(w, 'numerical_environment', lambda: environment)
    receipt = k.artifact('PREFLIGHT', parents={'study': value['content_hash']}, imports=imports,
        source=source['content_hash'], environment=environment['content_hash'], passed=True)
    k.write(root/'preflight.json', receipt)
    return value


def lock_test(study):
    lock = k.artifact('TEST_BUILD_LOCK', parents={'study': study['content_hash'],
        'population': study['population']['content_hash'], 'source': study['source']['content_hash'],
        'bundle': c.imported(study, 'bundle')['content_hash']}, materialize_only=True, evaluate=False)
    k.write(Path(study['root'])/'test_build_lock.json', lock)
    return lock


def pairs_for(study, shard):
    row, entries = p.entries_for(study['population'], shard)
    particle = sample(1)[0].offline
    return [Pair(i, row['sha256'], particle, None)
            for i in p.ids(study['population']['parents']['inventory'], row, entries)]


def fixture_generate(monkeypatch, study, shard, bundle, *, real_pool=False, attempt_name='pilot_000'):
    original_parallel = e.parallel
    def parallel(stream, bundle, workers, chunk, trace):
        assert workers == 36 and chunk == 32 and trace is False
        return original_parallel(stream, bundle, 4 if real_pool else 1, chunk, trace)
    monkeypatch.setattr(e, 'parallel', parallel)
    def iterate(*args, **kwargs):
        if shard['role'] == 'final_test':
            assert kwargs['test_lock'] == c.test_lock(study)
        yield from pairs_for(study, shard)
    monkeypatch.setattr(p, 'iterate', iterate)
    reservation = s.reserve(study, attempt_name, shard['shard_id'], 5*2**20)
    row = w.generate(study, {'name': attempt_name}, shard, reservation, bundle=bundle,
                     measurement_factory=LocalMeasurement)
    receipt = k.artifact('SHARD', parents={'study': study['content_hash'],
        'population': study['population']['content_hash']}, test=shard['role'] == 'final_test',
        **row, source=study['source']['content_hash'], environment=study['numerical_environment']['content_hash'],
        authentication_seconds=.1, allocation=dict(job_id='100'))
    k.write(Path(study['root'])/'shards'/f'{shard["shard_id"]}.json', receipt)
    return receipt


def test_new_contract_truthful_test_materialization():
    a = k.artifact('SHARD', test=True)
    k.validate(a, 'SHARD', test=True)
    assert a['final_test_accessed'] and a['final_test_materialized'] and not a['final_test_evaluated']
    with pytest.raises(ValueError): k.validate(a, 'SHARD', test=False)
    with pytest.raises(ValueError): k.artifact('SHARD', final_test_accessed=False)


def test_population_exact_disjoint_nested_and_deterministic(population):
    inv, profile, donor, value = population
    assert value == p.build(inv, profile, donor)
    shards = p.shards(value)
    covered = set()
    for shard in shards:
        row, entries = p.entries_for(value, shard)
        keys = set(p.ids(inv['content_hash'], row, entries))
        assert not keys & covered
        covered |= keys
    assert len(covered) == sum(p.COUNTS.values())
    assert [s['pilot'] for s in shards[:2]] == [True, True]
    assert all(s['role'] == 'train' for s in shards[:2])
    assert all(s['jets'] <= 16 for s in shards[2:])
    modified = deepcopy(profile); modified['groups'][0]['role'] = 'final_test'
    with pytest.raises(ValueError): p.build(inv, modified, donor)
    modified = deepcopy(profile); modified['registry_sha256'] = 'a'*64
    with pytest.raises(ValueError): p.build(inv, modified, donor)
    modified = deepcopy(profile); modified['memberships']['train']['files'][0]['entry_mask'] = pack_entries(range(10, 90), 128)
    with pytest.raises(ValueError, match='nested'): p.build(inv, modified, donor)


def test_real_profile_and_offline_root_reading(registered, monkeypatch):
    root, inv, _, registry = registered
    profile = select_profile(registry, inv, 'TRAIN_44')
    donor = select_profile(registry, inv, 'TRAIN_22')
    monkeypatch.setattr(p, 'PROFILE', 'TRAIN_44')
    monkeypatch.setattr(p, 'COUNTS', dict(train=44, validation=5, final_test=11))
    value = p.build(inv, profile, donor)
    arrays = uproot.behaviors.TBranch.HasBranches.arrays
    seen = []
    def offline_only(self, expressions=None, *args, **kwargs):
        assert expressions == list(p.BRANCHES)
        assert not any('hlt' in e or 'label' in e for e in expressions)
        seen.append(expressions)
        return arrays(self, expressions, *args, **kwargs)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, 'arrays', offline_only)
    row = next(r for r in value['files'] if r['role'] == 'validation')
    entries = p.unpack_entries(row['entry_mask'], row['raw_entries'])
    shard = dict(role='validation', path=row['path'], start=0, stop=len(entries), jets=len(entries),
                 ordered_identities=k.digest_ids(p.ids(inv['content_hash'], row, entries)))
    rows = list(p.iterate(root, value, shard, provisional_compatibility(inventory_hash='a'*64)))
    assert seen and len(rows) == len(entries) and all(r.hlt is None for r in rows)
    shard['role'] = 'final_test'
    with pytest.raises(PermissionError, match='lock'):
        list(p.iterate(root, value, shard, {}))


def test_real_process_kernel_pilot_roundtrip(study, bundle, monkeypatch):
    row = fixture_generate(monkeypatch, study, study['shards'][0], bundle, real_pool=True)
    assert row['serial_process_exact'] is True and row['jets'] == 64
    o.verify_shard(study, row)
    assert set(o.completed(study)) == {row['shard_id']}
    values = o.arrays(Path(study['root'])/row['blocks'][0]['relative'])
    assert set(values) == {*e.FIELDS, 'offsets', 'jet_identity'}
    assert values['p4'].dtype == np.dtype('<f8')


def test_block_corruption_and_reordering_rejected(study, bundle, monkeypatch):
    row = fixture_generate(monkeypatch, study, study['shards'][0], bundle)
    changed = with_content_hash(dict(row, ordered_identities='f'*64))
    with pytest.raises(ValueError): o.verify_shard(study, changed)
    path = Path(study['root'])/row['blocks'][0]['relative']
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'): o.verify_shard(study, row)


def test_full_synthetic_bank_and_sealed_reader(study, bundle, monkeypatch):
    lock_test(study)
    original = e.parallel
    for shard in study['shards']:
        monkeypatch.setattr(e, 'parallel', original)
        fixture_generate(monkeypatch, study, shard, bundle,
                         attempt_name='pilot_000' if shard['pilot'] else 'bulk_001')
    result = o.finalize(study)
    assert result['counts'] == p.COUNTS and result['final_test_accessed']
    assert not result['physics_production_qualified']
    assert len(list(o.read_role(study['root'], 'train'))) == p.COUNTS['train']
    with pytest.raises(PermissionError): list(o.read_role(study['root'], 'final_test'))
    assert o.finalize(study) == result  # immutable idempotent commit marker


def test_incomplete_dataset_not_published(study):
    lock_test(study)
    with pytest.raises(ValueError, match='Incomplete'): o.finalize(study)
    assert not (Path(study['root'])/'dataset_manifest.json').exists()


def test_reservations_preserve_failed_attempts_and_bound_writes(study):
    a = s.reserve(study, 'pilot_000', 's00000', 1024)
    assert s.reserve(study, 'pilot_000', 's00000', 1024) == a
    s.publish_block(study, a, 'tiny.bin', b'one')
    with pytest.raises(FileExistsError): s.publish_block(study, a, 'tiny.bin', b'two')
    with pytest.raises(OSError): s.publish_block(study, a, 'large.bin', bytes(2048))
    b = s.reserve(study, 'pilot_001', 's00000', 2048)
    assert a['relative'] != b['relative']
    with pytest.raises(OSError): s.reserve(study, 'bulk_002', 's00001', 50*k.GIB)


@pytest.mark.parametrize('relative', ['../escape', '/absolute', 'x/../../escape', 'x\\y', 'x//y', 'C:/x'])
def test_path_escape_rejected(tmp_path, relative):
    with pytest.raises(ValueError): k.safe(tmp_path, relative)


def test_measured_admission_and_no_unmeasured_bulk():
    row = dict(jets=10000, processing_seconds=1000, authentication_seconds=60,
        role='train', serial_process_exact=True, output_bytes=30_000_000,
        allocation=dict(partition='tigris', cpus=36, gpus=0, job_id='123'),
        measurement=dict(sampled_peak_tree_rss_bytes=5*k.GIB, samples=100))
    assert c.resources([row, row]) == dict(memory_gib=32, hours=3, bytes_per_jet=12000.)
    with pytest.raises(PermissionError): c.resources([row])
    with pytest.raises(PermissionError): c.resources([dict(row, processing_seconds=50000), row])
    with pytest.raises(PermissionError): c.resources([dict(row, measurement=dict(sampled_peak_tree_rss_bytes=100*k.GIB)), row])


def test_pilot_and_bulk_exact_array_plans(study):
    a = c.make_attempt(study, 'pilot', study['shards'][:2], memory_gib=64, hours=4, bytes_per_jet=10000)
    plan = sub.plan(study, a)
    assert [r['task'] for r in plan['commands']] == ['preflight', 'generate']
    assert '--array=0-1%2' in plan['commands'][1]['argv']
    assert plan['commands'][1]['dependency'] == 'afterok:preflight'
    lock_test(study)
    a = c.make_attempt(study, 'bulk', study['shards'][2:], memory_gib=32, hours=2, bytes_per_jet=10000)
    plan = sub.plan(study, a)
    assert plan['commands'][-1]['dependency'] == 'afterany:generate'
    assert plan['gpus'] == 0 and plan['generation_cpu_cap'] == 576
    for r in plan['commands']:
        assert '--partition=tigris' in r['argv'] and '--account=reu-aisocial' in r['argv']
        assert not any('gpu' in arg or arg.startswith('--qos') for arg in r['argv'])
    corrupt = with_content_hash(dict(a, resources=dict(a['resources'], concurrent=17)))
    with pytest.raises(ValueError): c.validate_attempt(corrupt, study)


def test_dry_submit_never_calls_scheduler(study, monkeypatch):
    a = c.make_attempt(study, 'pilot', study['shards'][:2], memory_gib=64, hours=4, bytes_per_jet=10000)
    path = Path(study['root'])/'attempts'/a['name']/'attempt_spec.json'
    def forbidden(*a, **k): raise AssertionError('No scheduler calls allowed')
    monkeypatch.setattr(sub.subprocess, 'run', forbidden)
    assert sub.submit(path)['contract'] == 'CMS2JC2_PROXY_PLAN/v1'
    with pytest.raises(PermissionError): sub.submit(path, execute=True, plan_hash='bad', phrase='bad')


def test_submission_journal_dependency_and_double_submit(study, monkeypatch):
    a = c.make_attempt(study, 'pilot', study['shards'][:2], memory_gib=64, hours=4, bytes_per_jet=10000)
    path = Path(study['root'])/'attempts'/a['name']/'attempt_spec.json'
    plan = sub.plan(study, a)
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        assert not any(k.startswith(('SBATCH_', 'SLURM_')) for k in kwargs['env'])
        return SimpleNamespace(returncode=0, stdout='123\n' if len(calls) == 3 else '124\n', stderr='')
    monkeypatch.setattr(sub.subprocess, 'run', run)
    result = sub.submit(path, execute=True, plan_hash=plan['content_hash'], phrase=plan['authorization_phrase'])
    assert result['jobs'] == dict(preflight='123', generate='124')
    assert '--dependency=afterok:123' in calls[-1]
    with pytest.raises(PermissionError, match='already attempted'):
        sub.submit(path, execute=True, plan_hash=plan['content_hash'], phrase=plan['authorization_phrase'])


def test_ambiguous_submission_is_not_retried(study, monkeypatch):
    a = c.make_attempt(study, 'pilot', study['shards'][:2], memory_gib=64, hours=4, bytes_per_jet=10000)
    path = Path(study['root'])/'attempts'/a['name']/'attempt_spec.json'
    plan = sub.plan(study, a)
    monkeypatch.setattr(sub.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='unexpected', stderr=''))
    with pytest.raises(RuntimeError, match='Ambiguous'):
        sub.submit(path, execute=True, plan_hash=plan['content_hash'], phrase=plan['authorization_phrase'])
    assert list((path.parent/'submission').glob('*_intent.json'))
    with pytest.raises(PermissionError, match='already attempted'):
        sub.submit(path, execute=True, plan_hash=plan['content_hash'], phrase=plan['authorization_phrase'])


def test_slurm_field_parser_keeps_cpus_per_task(monkeypatch):
    monkeypatch.setattr(sub.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='JobId=123 CPUs/Task=36 NumCPUs=36'))
    assert sub.fields('123')['CPUs/Task'] == '36'


def test_array_accounting_preserves_array_identity(monkeypatch):
    def run(argv, **kwargs):
        assert 'JobID%80,State,ExitCode,User,Account,JobName%80' in argv
        return SimpleNamespace(stdout=f'123_0|COMPLETED|0:0|{sub.getpass.getuser()}|reu-aisocial|c2jp_generate|\n')
    monkeypatch.setattr(sub.subprocess, 'run', run)
    assert sub.accounting(['123'])[0]['job'] == '123_0'


def terminal_fixture(study):
    a = c.make_attempt(study, 'pilot', study['shards'][:2], memory_gib=64, hours=4, bytes_per_jet=10000)
    directory = Path(study['root'])/'attempts'/a['name']
    k.write(directory/'submission_ledger.json', k.artifact('LEDGER', parents={'attempt': a['content_hash']},
        plan=sub.plan(study, a)['content_hash'], jobs=dict(preflight='120', generate='123'), dry_run=False))
    plan = sub.plan(study, a)
    parents = {'attempt': a['content_hash'], 'plan': plan['content_hash']}
    for i, (row, job) in enumerate(zip(plan['commands'], ('120', '123'))):
        argv = list(row['argv'])
        if i: argv.insert(1, '--dependency=afterok:120')
        prefix = directory/'submission'/f'{i:02d}_{row["task"]}'
        k.write(str(prefix)+'_intent.json', k.artifact('SUBMIT_INTENT', parents=parents, argv=argv))
        k.write(str(prefix)+'_response.json', k.artifact('SUBMIT_RESPONSE', parents=parents,
            returncode=0, stdout=job+'\n', stderr=''))
    return a


@pytest.mark.parametrize('state', ['RUNNING', 'PENDING', 'UNKNOWN'])
def test_recovery_rejects_any_nonterminal_array_element(study, monkeypatch, state):
    terminal_fixture(study)
    rows = [dict(job='120', state='COMPLETED', name='c2jp_preflight'),
        dict(job='123_0', state='COMPLETED', name='c2jp_generate'),
        dict(job='123_1', state=state, name='c2jp_generate')]
    monkeypatch.setattr(sub, 'accounting', lambda ids: rows)
    with pytest.raises(PermissionError, match='active/unknown'): sub.require_terminal(study)


def test_recovery_requires_every_array_index_and_queue_absence(study, monkeypatch):
    terminal_fixture(study)
    rows = [dict(job='120', state='COMPLETED', name='c2jp_preflight'),
        dict(job='123_0', state='COMPLETED', name='c2jp_generate')]
    monkeypatch.setattr(sub, 'accounting', lambda ids: rows)
    with pytest.raises(PermissionError, match='Incomplete'): sub.require_terminal(study)
    rows.append(dict(job='123_1', state='FAILED', name='c2jp_generate'))
    monkeypatch.setattr(sub.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='123_1|RUNNING', stderr=''))
    with pytest.raises(PermissionError, match='live queue'): sub.require_terminal(study)
    monkeypatch.setattr(sub.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='', stderr=''))
    sub.require_terminal(study)


def test_retry_only_unfinished_pilot_not_committed_shard(study, bundle, monkeypatch):
    c.make_attempt(study, 'pilot', study['shards'][:2], memory_gib=64, hours=4, bytes_per_jet=10000)
    receipt = fixture_generate(monkeypatch, study, study['shards'][0], bundle)
    monkeypatch.setattr(sub, 'require_terminal', lambda *a, **k: None)
    attempt = c.advance(study, recovery=True)
    assert attempt['kind'] == 'pilot' and attempt['shards'] == [study['shards'][1]['shard_id']]
    assert load_json_receipt(study, receipt['shard_id']) == receipt


def load_json_receipt(study, shard_id):
    return k.load_json(Path(study['root'])/'shards'/f'{shard_id}.json')


def test_bulk_requires_explicit_test_permission_and_genuine_measurement(study, bundle, monkeypatch):
    original = e.parallel
    for shard in study['shards'][:2]:
        monkeypatch.setattr(e, 'parallel', original)
        fixture_generate(monkeypatch, study, shard, bundle)
    monkeypatch.setattr(sub, 'require_terminal', lambda *a, **k: None)
    # Local synthetic timing must never stand in for remote evidence.
    with pytest.raises(PermissionError, match='Genuine'):
        c.advance(study, authorize_test=True)
    monkeypatch.setattr(c, 'resources', lambda *a: dict(memory_gib=32, hours=1, bytes_per_jet=1000))
    with pytest.raises(PermissionError, match='Explicit'):
        c.advance(study)
    attempt = c.advance(study, authorize_test=True)
    assert attempt['shards'] == [s['shard_id'] for s in study['shards'][2:]]
    assert attempt['kind'] == 'bulk'
    assert c.test_lock(study)['evaluate'] is False
    c.require_admission(study, attempt)
    changed = with_content_hash(dict(attempt, resources=dict(attempt['resources'], hours=2)))
    with pytest.raises(ValueError, match='resources'): c.require_admission(study, changed)


def test_preflight_failure_recovery_uses_fresh_pilot(study, monkeypatch):
    (Path(study['root'])/'preflight.json').unlink()  # only disposable fixture
    monkeypatch.setattr(sub, 'require_terminal', lambda *a, **k: None)
    with pytest.raises(PermissionError, match='Preflight'): c.advance(study)
    attempt = c.advance(study, recovery=True)
    assert attempt['kind'] == 'pilot' and len(attempt['shards']) == 2


def test_storage_attestation_and_overlap_checks(tmp_path):
    parent = tmp_path/'persistent'; parent.mkdir()
    raw = parent/'raw'; raw.mkdir()
    assert c.location(parent/'new', parent, [raw])[0] == (parent/'new').resolve()
    with pytest.raises(PermissionError): c.location(raw/'nested', raw, [raw])
    with pytest.raises(ValueError): c.location(raw, parent, [])
    with pytest.raises(PermissionError): c.create(acknowledge_proxy=False, persistent_attested=False,
        gate_spec='', confirmation_spec='', confirmation_hash='', profile='', project_dir='', source_commit='',
        root='', persistent_parent='', budget_gib=50, available_quota_gib=60)


def test_worker_scheduler_rejects_identity_or_resource_mutation(study, monkeypatch):
    a = c.make_attempt(study, 'pilot', study['shards'][:2], memory_gib=64, hours=4, bytes_per_jet=10000)
    environment = dict(CONDA_PREFIX=c.ENVIRONMENT, SLURM_JOB_PARTITION='tigris',
        SLURM_JOB_ACCOUNT='reu-aisocial', SLURM_CPUS_PER_TASK='36', SLURM_JOB_NUM_NODES='1',
        PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', LD_LIBRARY_PATH=c.ENVIRONMENT+'/lib',
        SLURM_JOB_ID='123', SLURM_ARRAY_JOB_ID='120', SLURM_ARRAY_TASK_ID='0',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    for name, value in environment.items(): monkeypatch.setenv(name, value)
    monkeypatch.setattr(sub.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(sub, 'submitted_identity', lambda *a, **k: '120')
    f = dict(Comment=f'c2jp:{a["content_hash"]}:generate', JobName='c2jp_generate',
        Partition='tigris', Account='reu-aisocial', NumCPUs='36', NumNodes='1', NumTasks='1',
        TimeLimit='04:00:00', MinMemoryNode='64G', JobState='RUNNING', WorkDir=study['project_dir'],
        UserId=sub.getpass.getuser()+'(123)', ReqTRES='cpu=36,mem=64G,node=1', ArrayTaskId='0', ArrayJobId='120')
    f['CPUs/Task'] = '36'
    monkeypatch.setattr(sub, 'fields', lambda job: f)
    assert sub.worker_identity(study, a, 'generate', 0)['cpus'] == 36
    for key, value in [('Comment', 'other'), ('NumCPUs', '72'), ('MinMemoryNode', '32G'),
                       ('ArrayTaskId', '1'), ('ReqTRES', 'cpu=36,gres/gpu=1')]:
        old = f[key]; f[key] = value
        with pytest.raises(PermissionError): sub.worker_identity(study, a, 'generate', 0)
        f[key] = old


def test_invalid_bank_shapes_and_dtypes_fail(tmp_path):
    particle = sample(1)[0].offline
    values = e.pack([(sample(1)[0].identity, particle, 'a'*64)])
    values['p4'] = values['p4'].astype('float32')
    path = tmp_path/'bad.npz'; path.write_bytes(e.encode(values, 'stored'))
    with pytest.raises(ValueError, match='dtype'): o.arrays(path)


def test_create_and_full_preflight_keep_frozen_lineage(population, bundle, tmp_path, monkeypatch):
    inv, prof, donor, population = population
    parent = tmp_path/'persistent'; parent.mkdir()
    project = tmp_path/'project'; project.mkdir()
    old = tmp_path/'old'; old.mkdir()
    raw = tmp_path/'jetclass2_10M_20260918_dzfix_partial_v1'/'jetclass2'; raw.mkdir(parents=True)
    source = k.artifact('SOURCE', files={}, executable=True, commit='a'*40)
    env = old_artifact('NUMERICAL_ENVIRONMENT', fixture=True)
    model = with_content_hash(dict(bundle, parents={'comparison': 'c'*64}))
    def put(name, value): return k.write(old/(name+'.json'), value)
    original = dict(root=str(old/'gen'), data_root=str(raw), review={'frozen': True},
        inventory=put('inventory', inv), profile=put('donor_profile', donor), bundle=put('bundle', model))
    tg = dict(root=str(old/'tigris'), source=source, numerical_environment=env,
              imported=put('import', dict(original_study=put('original', original))))
    gs = dict(stage='tigris_direct_gate', study=put('tg_study', tg))
    cs = dict(stage='frozen_confirm', root=str(old/'confirmation'))
    gate_path = old/'gs.json'; k.write(gate_path, gs)
    confirm_path = old/'cs.json'; k.write(confirm_path, cs)
    profile_path = old/'profile.json'; k.write(profile_path, prof)
    gr = old_artifact('TG_GATE', compatible=True, within_site_exact=True, serial=dict(output_bytes=10000))
    cr = old_artifact('FROZEN_REPORT', selected='JOINT', decision={'status': 'rejected'})
    monkeypatch.setattr(c.dev, 'product', lambda spec, task, *a: gr if task == 'jt_gate' else cr)
    monkeypatch.setattr(c.confirmation, 'donor', lambda spec: {'content_hash': 'c'*64})
    monkeypatch.setattr(c, 'source_snapshot', lambda *a: source)
    monkeypatch.setattr(c, 'numerical_environment', lambda: env)
    monkeypatch.setattr(s, 'check_space', lambda *a: None)
    attempt = c.create(gate_spec=gate_path, confirmation_spec=confirm_path,
        confirmation_hash=cr['content_hash'], profile=profile_path, project_dir=project,
        source_commit='a'*40, root=parent/'production', persistent_parent=parent,
        budget_gib=50, available_quota_gib=60, acknowledge_proxy=True, persistent_attested=True)
    study = k.load_json(k.checked(attempt['study']))
    assert study['confirmation_status'] == 'rejected'  # poor physics is not an execution error
    assert c.imported(study, 'bundle') == model
    assert study['population'] == population and study['counts'] == p.COUNTS
    assert not (parent/'production'/'test_build_lock.json').exists()
    monkeypatch.setattr(c.direct, 'validate_stage', lambda *a, **k: tg)
    monkeypatch.setattr(c.direct, 'accepted', lambda *a: gr)
    monkeypatch.setattr(c.direct, 'inputs', lambda *a: (original, {}, model))
    monkeypatch.setattr(c.confirmation, 'read', lambda *a: cr)
    receipt = c.preflight(study)
    assert receipt['passed'] and receipt['confirmation_status'] == 'rejected'
    monkeypatch.setattr(c.confirmation, 'donor', lambda *a: {'content_hash': 'd'*64})
    with pytest.raises(ValueError, match='authentication'): c.preflight(study)


def test_worker_binds_exact_durable_submission(study, monkeypatch):
    a = c.make_attempt(study, 'pilot', study['shards'][:2], memory_gib=64, hours=4, bytes_per_jet=10000)
    directory = Path(study['root'])/'attempts'/a['name']
    plan = sub.plan(study, a)
    counter = []
    def run(argv, **kwargs):
        counter.append(argv)
        return SimpleNamespace(returncode=0, stdout='120\n' if len(counter) == 3 else '123\n', stderr='')
    monkeypatch.setattr(sub.subprocess, 'run', run)
    sub.submit(directory/'attempt_spec.json', execute=True,
               plan_hash=plan['content_hash'], phrase=plan['authorization_phrase'])
    assert sub.submitted_identity(study, a, 'preflight') == '120'
    assert sub.submitted_identity(study, a, 'generate') == '123'
    path = directory/'submission'/'01_generate_intent.json'
    value = k.load_json(path)
    value['argv'][1] = '--dependency=afterok:999'
    path.write_text(json.dumps(with_content_hash(value)))
    with pytest.raises(PermissionError, match='exact plan'): sub.submitted_identity(study, a, 'generate')

"""Frozen-production tests: synthetic ROOT, no real jobs/test dataset access."""
import copy
from pathlib import Path
import subprocess

import numpy as np
import pytest

from test_literature_proxy_v3_campaign import parent, parent_v2
from hlt_classification.literature_proxy_v3 import campaign as v3, worker as w3
from hlt_classification.literature_proxy_production import campaign as c, codec, engine, output, population as pop, storage, submission as s, worker
from hlt_classification.literature_proxy_production.contracts import artifact, load_json, write, with_content_hash, GIB, safe
from hlt_classification.literature_proxy_v3.contracts import read_reference


@pytest.fixture(scope='module')
def final_pilot(parent_v2, tmp_path_factory):
    temp = tmp_path_factory.mktemp('production-parent')
    with pytest.MonkeyPatch.context() as m:
        m.setattr(v3, 'PROJECT', temp/'v3-code')
        m.setattr(v3, 'source', lambda *a, **kw: dict(commit='c'*40, file_sha256={}))
        for name in engine.THREADS:
            m.setenv(name, '1')
        m.setenv('SLURM_CPUS_PER_TASK', '2')
        spec = v3.create(project=v3.PROJECT, commit='c'*40, parent_spec=parent_v2, root=temp/'v3')
        w3.run(spec, workers=2)
    return Path(spec['root'])/'study_spec.json'


@pytest.fixture
def study(final_pilot, tmp_path, monkeypatch):
    # Only tests can downsize: the public CLI has no population/count override.
    counts = dict(train=44, validation=11, final_test=11)
    monkeypatch.setattr(pop.registered, 'COUNTS', counts)
    monkeypatch.setattr(pop.registered, 'PROFILE', 'TRAIN_44')
    monkeypatch.setattr(pop, 'COUNTS', counts)
    monkeypatch.setattr(c, 'PROJECT', tmp_path/'production-code')
    monkeypatch.setattr(c, 'source', lambda *a, **kw: dict(commit='d'*40, file_sha256={}))
    for name in engine.THREADS:
        monkeypatch.setenv(name, '1')
    spec = load_json(final_pilot)
    original = read_reference(read_reference(spec['parent_spec'])['parent_spec'])
    result = c.create(project=c.PROJECT, commit='d'*40, parent_spec=final_pilot,
        profile=original['profile']['path'], root=tmp_path/'dataset', persistent_parent=tmp_path,
        available_quota_gib=8, budget_gib=4, persistent_attested=True, acknowledge_synthetic=True)
    return result


def initial(study):
    return Path(study['root'])/'attempts/initial/attempt_spec.json'


def admit(study, attempt):
    # Unit-test-only gate evidence. Remote worker is separately scheduler-bound.
    result = artifact('PREFLIGHT', parents=dict(study=study['content_hash'], attempt=attempt['content_hash']),
                      exact_replay=True, resource_envelope_ok=True)
    write(Path(study['root'])/f'attempts/{attempt["name"]}/preflight.json', result)


def serial_bulk(monkeypatch):
    old = engine.process
    monkeypatch.setattr(engine, 'process', lambda stream, rates, workers, publish, **kw:
                        old(stream, rates, 1, publish, **kw))


def test_full_chain_raw_saved_v3_process_writer_parity(study):
    spec = read_reference(c.bundle(study)['pilot'])
    attempt = load_json(initial(study))
    result = worker.replay(study, attempt, spec, c.bundle(study)['calibration'], workers=2)
    assert result['exact_replay'] and result['jets'] == 12 and result['bytes'] > 0
    assert c.bundle(study)['calibration'] == spec['calibration']
    assert c.bundle(study)['recipe'] == v3.recipe()
    assert c.validate_study(study) == study


def test_identical_existing_population_no_root_read_for_selection(study, monkeypatch):
    import uproot
    monkeypatch.setattr(uproot, 'open', lambda *a, **k: pytest.fail('Metadata build opened ROOT'))
    original = pop.build(read_reference(study['inventory']), load_json(study['profile']['path']),
                         read_reference(study['donor_profile']))
    assert original == study['population']
    assert study['shards'] == pop.shards(original)
    assert [s['role'] for s in study['shards']] == sorted(
        [s['role'] for s in study['shards']], key=list(pop.COUNTS).index)
    assert {r: sum(s['jets'] for s in study['shards'] if s['role'] == r) for r in pop.COUNTS} == pop.COUNTS


def test_raw_reader_only_offline_padded_keys_and_pairing(study, monkeypatch):
    import uproot
    old = uproot.behaviors.TBranch.HasBranches.arrays
    requests = []
    def tracked(self, expressions, **kw):
        requests.append(expressions)
        assert expressions == list(pop.BRANCHES)
        assert kw['entry_stop']-kw['entry_start'] <= 512
        return old(self, expressions, **kw)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, 'arrays', tracked)
    shard = study['shards'][0]
    rows = list(pop.iterate(study['data_root'], study['population'], shard))
    assert requests and len(rows) == shard['jets']
    record, entries = pop.entries_for(study['population'], shard)
    assert [i for i, _ in rows] == list(pop.ids(study['population']['parents']['inventory'], record, entries))
    for _, p in rows:
        assert p.keys == tuple(f'part:{i:08d}' for i in range(len(p)))


def test_sealed_test_before_open_and_explicit_materialization(study, monkeypatch):
    shard = next(s for s in study['shards'] if s['role'] == 'final_test')
    real = pop.authenticated_open
    monkeypatch.setattr(pop, 'authenticated_open', lambda *a: pytest.fail('Unauthorized test open'))
    with pytest.raises(PermissionError, match='lock'):
        list(pop.iterate(study['data_root'], study['population'], shard))
    wrong = with_content_hash(dict(c.test_lock(study), evaluate=True))
    with pytest.raises(PermissionError):
        list(pop.iterate(study['data_root'], study['population'], shard, test_lock=wrong))
    with pytest.raises(PermissionError):
        list(output.read_role(study['root'], 'final_test'))
    monkeypatch.setattr(pop, 'authenticated_open', real)
    assert len(list(pop.iterate(study['data_root'], study['population'], shard,
                                test_lock=c.test_lock(study)))) == shard['jets']


def test_end_to_end_release_train_before_test_and_complete(study, monkeypatch):
    attempt = load_json(initial(study))
    admit(study, attempt)
    serial_bulk(monkeypatch)
    rates = c.bundle(study)['calibration']
    with pytest.raises(ValueError, match='Incomplete'):
        output.manifest(study)
    for shard in study['shards']:
        if shard['role'] == 'train':
            result = engine.generate_shard(study, attempt, shard, rates)
            assert result['final_test_accessed'] is False
    release = output.manifest(study, 'train')
    assert release['counts'] == {'train': 44} and not release['final_test_accessed']
    rows = list(output.read_role(study['root'], 'train'))
    assert len(rows) == 44 and len({i for i, _ in rows}) == 44
    assert all(set(v) == set(codec.FIELDS) for _, v in rows)
    assert output.status(study)['committed_jets'] == dict(train=44, validation=0, final_test=0)
    for shard in study['shards']:
        if shard['role'] != 'train':
            result = engine.generate_shard(study, attempt, shard, rates)
            assert result['final_test_materialized'] == (shard['role'] == 'final_test')
            assert not result['final_test_evaluated']
    result = output.manifest(study)
    assert result['counts'] == pop.COUNTS and result['final_test_materialized']
    assert not result['final_test_evaluated'] and not result['physics_production_qualified']
    output.manifest(study, 'validation')
    assert len(list(output.read_role(study['root'], 'validation'))) == 11
    # Corrupt test bank: train reader neither inspects nor depends on it.
    receipt = load_json(Path(study['root'])/f'shards/{study["shards"][-1]["shard_id"]}.json')
    safe(study['root'], receipt['blocks'][0]['relative']).write_bytes(b'broken')
    assert len(list(output.read_role(study['root'], 'train'))) == 44
    with pytest.raises(ValueError, match='checksum'):
        output.manifest(study)


def test_incomplete_no_manifest_and_no_overwrite(study, monkeypatch):
    attempt = load_json(initial(study))
    rates = c.bundle(study)['calibration']
    with pytest.raises(FileNotFoundError):
        engine.generate_shard(study, attempt, study['shards'][0], rates)
    admit(study, attempt)
    serial_bulk(monkeypatch)
    receipt = engine.generate_shard(study, attempt, study['shards'][0], rates)
    assert output.verify_shard(study, receipt) == receipt
    with pytest.raises(PermissionError, match='immutable'):
        engine.generate_shard(study, attempt, study['shards'][0], rates)
    with pytest.raises(ValueError, match='Incomplete'):
        output.manifest(study)
    assert not (Path(study['root'])/'dataset_manifest.json').exists()


def test_shard_chunk_worker_invariance_and_no_extra_bank_fields(study, tmp_path):
    shard = study['shards'][0]
    raw = list(pop.iterate(study['data_root'], study['population'], shard))
    rates = c.bundle(study)['calibration']
    serial = [r for rows, _ in engine.parallel(raw, rates, 1, chunk=1) for r in rows]
    proc = [r for rows, _ in engine.parallel(raw, rates, 2, chunk=2) for r in rows]
    assert [codec.physical_digest(i, p) for i, p, _ in serial] == [codec.physical_digest(i, p) for i, p, _ in proc]
    assert all(r[2] is None for r in serial)
    packed = codec.pack(serial)
    assert set(packed) == {*codec.FIELDS, 'offsets', 'jet_identity'}
    path = tmp_path/'bank.npz'
    path.write_bytes(codec.encode(packed, 'stored'))
    assert codec.readback(path, packed) == [codec.physical_digest(i, p) for i, p, _ in serial]
    with np.load(path, allow_pickle=False) as data:
        assert data['tracking'].dtype == np.dtype('<f8')


def test_single_dag_exact_dry_plan_and_no_duplicate_submission(study, monkeypatch):
    calls, number = [], 100
    def fake(argv, **kwargs):
        nonlocal number
        calls.append(argv)
        assert not any(k.startswith(('SBATCH_', 'SLURM_')) for k in kwargs['env'])
        if '--test-only' in argv:
            return subprocess.CompletedProcess(argv, 0, 'shape accepted', '')
        number += 1
        return subprocess.CompletedProcess(argv, 0, f'{number}\n', '')
    monkeypatch.setattr(s.subprocess, 'run', fake)
    plan = s.submit(initial(study))
    assert not calls and plan['gpus'] == 0 and plan['generation_cpu_cap'] == 576
    tasks = plan['commands']
    assert [t['task'] for t in tasks] == ['preflight', 'generate', 'finalize']
    assert tasks[1]['dependency'] == 'afterok:preflight' and tasks[1]['array'].endswith('%16')
    assert tasks[2]['dependency'] == 'afterany:generate'
    with pytest.raises(PermissionError):
        s.submit(initial(study), execute=True, plan_hash='wrong', phrase=plan['authorization_phrase'])
    ledger = s.submit(initial(study), execute=True, plan_hash=plan['content_hash'], phrase=plan['authorization_phrase'])
    assert ledger['jobs'] == dict(preflight='101', generate='102', finalize='103')
    assert len(calls) == 6 and '--dependency=afterok:101' in calls[4] and '--dependency=afterany:102' in calls[5]
    attempt = load_json(initial(study))
    assert s.submitted_identity(study, attempt, 'generate') == '102'
    with pytest.raises(PermissionError, match='already'):
        s.submit(initial(study), execute=True, plan_hash=plan['content_hash'], phrase=plan['authorization_phrase'])
    assert len(calls) == 6


def test_ambiguous_submission_preserved_no_retry(study, monkeypatch):
    monkeypatch.setattr(s.subprocess, 'run', lambda argv, **kw: subprocess.CompletedProcess(argv, 0, 'unclear', ''))
    p = s.submit(initial(study))
    with pytest.raises(RuntimeError, match='Ambiguous'):
        s.submit(initial(study), execute=True, plan_hash=p['content_hash'], phrase=p['authorization_phrase'])
    with pytest.raises(PermissionError, match='already'):
        s.submit(initial(study), execute=True, plan_hash=p['content_hash'], phrase=p['authorization_phrase'])
    assert (initial(study).parent/'submission/00_preflight_intent.json').exists()


def test_failed_storage_reservations_still_count(study):
    storage.reserve(study, 'failed', 's00000', GIB)
    storage.reserve(study, 'next', 's00001', GIB)
    with pytest.raises(OSError, match='budget exhausted'):
        storage.reserve(study, 'third', 's00002', 1)
    assert len(list((Path(study['root'])/'reservations').glob('*.json'))) == 2
    reservation = load_json(Path(study['root'])/'reservations/failed_s00000.json')
    with pytest.raises(ValueError, match='Unsafe'):
        storage.publish_block(study, reservation, '../escape', b'no')


def test_recovery_requires_terminal_and_only_missing(study, monkeypatch):
    with pytest.raises(PermissionError, match='unsubmitted'):
        c.create_attempt(study, 'retry', recovery=True)
    attempt = load_json(initial(study))
    admit(study, attempt)
    serial_bulk(monkeypatch)
    first = study['shards'][0]
    engine.generate_shard(study, attempt, first, c.bundle(study)['calibration'])
    monkeypatch.setattr(s, 'require_terminal', lambda *a, **k: None)
    path = c.create_attempt(study, 'retry', recovery=True)
    retry = load_json(path)
    assert first['shard_id'] not in retry['shards'] and set(retry['retained_shards']) == {first['shard_id']}
    c.validate_attempt(retry, study)
    with pytest.raises(ValueError, match='all and only'):
        c.validate_attempt(with_content_hash(dict(retry, shards=retry['shards'][1:])), study)


@pytest.mark.parametrize('text', [
    'JobId=123 ArrayJobId=100 ArrayTaskId=0\nJobId=124 ArrayJobId=100 ArrayTaskId=1',
    'JobId=123 ArrayJobId=100 ArrayTaskId=1',
    'JobId=123 JobId=124 ArrayJobId=100 ArrayTaskId=0',
])
def test_scheduler_exact_array_parser_rejects_siblings(monkeypatch, text):
    monkeypatch.setattr(s.subprocess, 'run', lambda argv, **kw: subprocess.CompletedProcess(argv, 0, text, ''))
    with pytest.raises(PermissionError):
        s.fields('100_0')


def test_scheduler_placeholder_element_uses_parent_index(monkeypatch):
    seen = []
    def fake(argv, **kw):
        seen.append(argv[-1])
        return subprocess.CompletedProcess(argv, 0, 'JobId=100 ArrayJobId=100 ArrayTaskId=0', '')
    monkeypatch.setattr(s.subprocess, 'run', fake)
    assert s.fields('100_0')['JobId'] == '100' and seen == ['100_0']


def test_environment_and_changed_frozen_rates_fail(study):
    value = c.bundle(study)
    worker.require_environment(value)
    bad = copy.deepcopy(value)
    bad['environment']['packages']['numpy'] = 'other'
    with pytest.raises(PermissionError, match='NumPy'):
        worker.require_environment(bad)
    bad_study = with_content_hash(dict(study, resources=dict(c.RESOURCES, cpus=72)))
    with pytest.raises(ValueError, match='scope'):
        c.validate_study(bad_study)
    bad_attempt = with_content_hash(dict(load_json(initial(study)), shards=[]))
    with pytest.raises(ValueError, match='all and only'):
        c.validate_attempt(bad_attempt, study)


def test_pushed_source_and_absolute_environment_scripts():
    root = Path(__file__).resolve().parents[1]
    shell = (root/'sbatch/run_jetclass2_literature_proxy_dataset.sh').read_text()
    assert '"${PROJECT_DIR}/scripts/jetclass2_literature_proxy_dataset.py"' in shell
    assert 'PYTHONNOUSERSITE=1' in shell and '${CONDA_PREFIX}/lib' in shell
    helper = (root/'scripts/queue_jetclass2_literature_proxy_dataset.sh').read_text()
    assert 'AUTHORIZE JC2 LITERATURE DATASET INITIAL EXACT PLAN' in helper
    assert '--plan-hash "$5"' in helper and 'LIT_PERSISTENT_STORAGE' in helper
    assert c.RESOURCES == dict(cpus=36, memory_gib=64, hours=2, concurrent=16,
                               partition='tigris', account='reu-aisocial', gpus=0)


def test_exact_worker_allocation_and_placeholder_identity(study, monkeypatch):
    attempt = load_json(initial(study))
    for name in list(s.os.environ):
        if name.startswith('SLURM_'):
            monkeypatch.delenv(name)
    env = dict(CONDA_PREFIX=c.ENVIRONMENT, SLURM_JOB_PARTITION='tigris',
        SLURM_JOB_ACCOUNT='reu-aisocial', SLURM_CPUS_PER_TASK='36', SLURM_JOB_NUM_NODES='1',
        PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', LD_LIBRARY_PATH=c.ENVIRONMENT+'/lib',
        SLURM_JOB_ID='123', SLURM_ARRAY_JOB_ID='123', SLURM_ARRAY_TASK_ID='0',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(s.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(s, 'submitted_identity', lambda *a, **k: '123')
    fields = dict(JobId='123', Comment=f'jc2litp:{attempt["content_hash"]}:generate',
        JobName='jc2litp_generate', Partition='tigris', Account='reu-aisocial', NumCPUs='36', NumNodes='1',
        NumTasks='1', TimeLimit='02:00:00', MinMemoryNode='64G', JobState='RUNNING', WorkDir=study['project_dir'],
        UserId=s.getpass.getuser()+'(123)', ReqTRES='cpu=36,mem=64G,node=1', ArrayTaskId='0', ArrayJobId='123')
    fields['CPUs/Task'] = '36'
    def fake(argv, **kw):
        assert argv == ['scontrol', 'show', 'job', '-o', '123_0']
        return subprocess.CompletedProcess(argv, 0, ' '.join(f'{k}={v}' for k, v in fields.items()), '')
    monkeypatch.setattr(s.subprocess, 'run', fake)
    assert s.worker_identity(study, attempt, 'generate', 0)['job_id'] == '123'
    for key, changed in [('JobId', '444'), ('Comment', 'unrelated'), ('NumCPUs', '72'),
                         ('MinMemoryNode', '32G'), ('ArrayTaskId', '1'), ('ReqTRES', 'cpu=36,gres/gpu=1')]:
        original = fields[key]
        fields[key] = changed
        with pytest.raises(PermissionError):
            s.worker_identity(study, attempt, 'generate', 0)
        fields[key] = original
    for bad in (None, True, -1, len(attempt['shards'])):
        with pytest.raises(PermissionError):
            s.worker_identity(study, attempt, 'generate', bad)


def test_terminal_requires_every_element_and_no_live_queue(study, monkeypatch):
    attempt = load_json(initial(study))
    plan = s.plan(study, attempt)
    jobs = dict(preflight='120', generate='123', finalize='124')
    write(initial(study).parent/'submission_ledger.json', artifact('LEDGER',
        parents={'attempt': attempt['content_hash']}, plan=plan['content_hash'], jobs=jobs, dry_run=False))
    monkeypatch.setattr(s, 'submitted_identity', lambda a, b, task: jobs[task])
    rows = [dict(job='120', state='COMPLETED', name='jc2litp_preflight'),
            dict(job='124', state='FAILED', name='jc2litp_finalize')]
    rows.extend(dict(job=f'123_{i}', state='COMPLETED', name='jc2litp_generate')
                for i in range(len(attempt['shards'])))
    monkeypatch.setattr(s, 'accounting', lambda ids: rows)
    rows[-1]['state'] = 'RUNNING'
    with pytest.raises(PermissionError, match='active/unknown'):
        s.require_terminal(study)
    rows[-1]['state'] = 'FAILED'
    last = rows.pop()
    with pytest.raises(PermissionError, match='Incomplete'):
        s.require_terminal(study)
    rows.append(last)
    monkeypatch.setattr(s.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, '123_0|RUNNING', ''))
    with pytest.raises(PermissionError, match='live queue'):
        s.require_terminal(study)
    monkeypatch.setattr(s.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, '', ''))
    s.require_terminal(study)


def test_actual_preflight_plumbing_and_environment_gate(study, monkeypatch):
    class LocalMeasurement:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def report(self):
            return dict(sampled_peak_tree_rss_bytes=GIB, samples=1, wall_seconds=1,
                        fixture_only_not_remote_evidence=True)
    monkeypatch.setattr(worker, 'Measurement', LocalMeasurement)
    old = worker.replay
    monkeypatch.setattr(worker, 'replay', lambda a, b, c, d, workers: old(a, b, c, d, workers=2))
    attempt = load_json(initial(study))
    result = worker.preflight(study, attempt)
    assert result['resource_envelope_ok'] and result['exact_replay']
    assert result['measurement']['fixture_only_not_remote_evidence']
    assert not result['final_test_accessed']
    assert c.require_preflight(study, attempt) == result


def test_no_silent_storage_or_population_resize(final_pilot, tmp_path):
    args = dict(project=tmp_path, commit='d'*40, parent_spec=final_pilot, profile='unused',
                root=tmp_path/'output', persistent_parent=tmp_path, available_quota_gib=21)
    with pytest.raises(PermissionError):
        c.create(**args)
    with pytest.raises(ValueError, match='quota'):
        c.create(**args, persistent_attested=True, acknowledge_synthetic=True)


def test_source_pins_new_and_reused_implementations(tmp_path, monkeypatch):
    monkeypatch.setattr(v3, 'source', lambda *a, **kw: dict(commit='d'*40, file_sha256={'v3': 'frozen'}))
    requested = []
    monkeypatch.setattr(v3.old.old, 'git', lambda project, *args: requested.extend(args))
    names = [*c.EXTRA, 'src/hlt_classification/literature_proxy_production/codec.py',
             'src/hlt_classification/cms2jc2_production/population.py',
             'src/hlt_classification/cms2jc2_response/readers.py']
    for name in names:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    result = c.source(tmp_path, 'd'*40)
    assert result['file_sha256']['v3'] == 'frozen'
    assert all(n in result['file_sha256'] and n in requested for n in names)


def test_list_input_is_consumed_once():
    assert list(engine.chunks([1, 2, 3], 2)) == [[1, 2], [3]]


def test_frozen_scientific_code_cannot_change_behind_recipe():
    name = 'src/hlt_classification/literature_proxy_v3/kernel.py'
    old = dict(file_sha256={name: 'frozen'})
    new = dict(file_sha256={name: 'frozen', 'src/hlt_classification/literature_proxy_production/engine.py': 'new'})
    c.require_frozen_code(new, old)
    for altered in (dict(file_sha256={name: 'changed'}), dict(file_sha256={})):
        with pytest.raises(ValueError, match='implementation changed'):
            c.require_frozen_code(altered, old)

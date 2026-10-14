"""New regular graph and durable inputs; CPU fixtures are not Slurm evidence."""
from copy import deepcopy
from pathlib import Path
import subprocess

import numpy as np
import pytest

from test_luka_fullsim_stage2 import dataset, prepared, bind, config
from hlt_classification.data.cache_contracts import (
    load_json, write_immutable_json, with_content_hash, deterministic_npz_bytes,
)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullsim import regular_campaign as c, regular_cache as cache
from hlt_classification.luka_fullsim import regular_submission as sub, regular_worker as w
from hlt_classification.luka_fullsim import regular_preflight as pre
from hlt_classification.luka_fullsim.stage2_cache import prepare_cache


def test_regular_graph_semantics():
    ns, ts = c.nodes(), c.tasks()
    assert len(ns) == 9 and len(ts) == 16
    assert sum(t['kind'] == 'reduce' for t in ts) == 5
    assert all(n['context_coordinate'] is None for n in ns)
    assert [n['teacher'] for n in ns[4:]] == ['U000', 'COARSE_U050', 'COARSE_U100', 'COARSE_D066', 'COARSE_D033']
    assert all(n['initialization_seed'] == ns[0]['initialization_seed'] and n['sampler_seed'] == ns[0]['sampler_seed']
               for n in ns if n['coordinate'] == 'D000')
    assert not any('FUSION' in n['node_id'] or 'DENSE' in n['node_id'] for n in ns)
    from hlt_classification.jetclass2_delphes.runner import recipe
    assert c.registration()['training_recipe'] == recipe()
    seen = set()
    for task in ts:
        assert set(task['dependencies']) <= seen
        seen.add(task['task_id'])
    assert c.registration()['counts'] == dict(train=100000, validation=50000)


@pytest.fixture(scope='module')
def durable(dataset, prepared, tmp_path_factory):
    container, inventory, splits = dataset
    # Inspect fixture shape without requiring production clean pushed checkout.
    root = tmp_path_factory.mktemp('regular_serial')
    result = cache.build_rows(container, inventory, splits, config(), root, workers=1)
    return root, result


def test_parallel_bytes_and_legacy_fingerprints(dataset, durable, tmp_path):
    container, inventory, splits = dataset
    root, serial = durable
    parallel = cache.build_rows(container, inventory, splits, config(), tmp_path, workers=2)
    for key in ('assignments', 'blocks', 'summaries', 'issues'):
        assert serial[key] == parallel[key]
    for role in ('train', 'validation'):
        for co in cache.s.COORDINATES:
            for ref in serial['blocks'][role][co]:
                assert (root/ref['path']).read_bytes() == (tmp_path/ref['path']).read_bytes()
    reference = cache.s.prepare_rows(container, inventory, splits, config(), tmp_path/'legacy')
    assert reference['summaries'] == serial['summaries']
    assert reference['assignments'] == serial['assignments']
    from hlt_classification.luka_fullsim.particle_audit import scan
    audit = scan(container, inventory, splits)
    assert audit['issues'] == serial['issues']
    for role in ('train', 'validation'):
        for key, row in audit['statistics'][role].items():
            actual = serial['statistics'][role][key]
            for field in ('finite', 'zero', 'negative', 'nonfinite', 'abs_bin_counts'):
                assert actual[field] == row[field]
            for field in ('mean', 'sd'):
                if row[field] is not None:
                    assert actual[field] == pytest.approx(row[field], abs=1e-12)


def cached_fixture(prepared, durable, monkeypatch):
    original, p = bind(prepared, monkeypatch)
    root, products = durable
    _, foundation, inventory, splits = cache.s.load_prepared(original)
    assert p['summaries'] == products['summaries']
    manifest = c.make_artifact('CACHE', parents=dict(prepared=p['content_hash'], gate='b'*64),
                               blocks=products['blocks'], serial_reference_passed=True, serial_reference_rows=2)
    write_immutable_json(root/'cache_manifest.json', manifest)
    monkeypatch.setattr(cache.s, 'load_prepared', lambda _: (p, foundation, inventory, splits))
    return root, p, inventory, splits


def test_disk_load_exact_masks_rows_all_coords_no_root(prepared, durable, monkeypatch):
    root, p, _, _ = cached_fixture(prepared, durable, monkeypatch)
    original_root, _ = bind(prepared, monkeypatch)
    expected = {}
    for role in ('train', 'validation'):
        for co in cache.s.COORDINATES:
            expected[role, co] = prepare_cache(original_root, role=role, coordinate=co, max_ram_bytes=2**30)
    monkeypatch.setattr(cache.uproot, 'open', lambda *a, **k: pytest.fail('GPU cache must not open ROOT'))
    for (role, co), reference in expected.items():
        actual = cache.load(root, role=role, coordinate=co, max_ram_bytes=2**30)
        for key, value in reference.batch(np.arange(len(reference))).items():
            np.testing.assert_array_equal(value, actual.batch(np.arange(len(actual)))[key])
    for role in ('final_test', 'unused', 'test'):
        with pytest.raises(PermissionError):
            cache.load('missing', role=role, coordinate='D000', max_ram_bytes=2**30)
    with pytest.raises(MemoryError):
        cache.load(root, role='train', coordinate='D000', max_ram_bytes=1)


def test_corrupt_blocks_and_identity_shape_checks(dataset, durable, tmp_path):
    _, inventory, splits = dataset
    root, products = durable
    ref = products['blocks']['train']['D000'][0]
    member = splits['memberships']['train'][0]
    data = (root/ref['path']).read_bytes()
    target = tmp_path/'block.npz'
    target.write_bytes(data+b'corrupt')
    with pytest.raises(ValueError, match='bytes differ'):
        cache.read_block(tmp_path, {**ref, 'path':'block.npz'}, member, inventory)
    original = cache.load_npz_arrays(root/ref['path'])
    for key, value in [('labels', original['labels'][::-1].copy()+1),
                        ('identities', original['identities'][::-1].copy()),
                        ('offsets', np.zeros_like(original['offsets'])),
                        ('features', original['features'].astype(np.float64))]:
        target.write_bytes(deterministic_npz_bytes({**original, key:value}))
        revised = dict(file_ref(target, root=tmp_path), file_index=ref['file_index'], rows=ref['rows'])
        with pytest.raises(ValueError, match='Durable input'):
            cache.read_block(tmp_path, revised, member, inventory)


def test_changed_fingerprint_file_order_and_lineage(prepared, durable, monkeypatch):
    root, p, inventory, splits = cached_fixture(prepared, durable, monkeypatch)
    authenticated = cache.metadata(root)
    bad = deepcopy(authenticated)
    bad[1]['summaries']['train']['D000']['input_sha256'] = '0'*64
    with pytest.raises(ValueError, match='fingerprint'):
        cache.load(root, role='train', coordinate='D000', max_ram_bytes=2**30, authenticated=bad)


def fixture_spec(tmp_path, p=None):
    root = tmp_path/'science'
    root.mkdir()
    return c.make_artifact('SPEC', root=str(root), project_dir=str(tmp_path/'project'), source_commit='a'*40,
        prepared_sha256='f'*64 if p is None else p['content_hash'], preflight_sha256='d'*64,
        admission=dict(prepared_root='fixture', preflight=dict(path='fixture')), train_minutes=200, reduce_minutes=20,
        **{**c.registration(), 'counts':c.COUNTS if p is None else p['counts']})


@pytest.mark.parametrize('gate', [False, True])
def test_dry_live_exact_dag_and_no_duplicates(tmp_path, monkeypatch, gate):
    spec = (c.make_artifact('GATE', root=str(tmp_path), project_dir=str(tmp_path/'project'), **c.gate_policy())
            if gate else fixture_spec(tmp_path))
    monkeypatch.setattr(c, 'validate_gate', lambda _: None)
    monkeypatch.setattr(c, 'validate_spec', lambda _: None)
    monkeypatch.setattr(sub, 'check_site', lambda _: None)
    calls = []
    def scheduler(args, **kwargs):
        assert not any('${JOB_' in x for x in args)
        assert not any(k.startswith(('SBATCH_', 'SLURM_')) for k in kwargs['env'])
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, str(900+len(calls))+';sporc\n', '')
    monkeypatch.setattr(sub.subprocess, 'run', scheduler)
    value = sub.submit(spec)
    assert len(value['commands']) == (3 if gate else 16) and not calls
    for row in value['commands']:
        assert '--partition=debug' in row['command'] and '--export=NONE' in row['command']
    with pytest.raises(PermissionError):
        sub.submit(spec, execute=True, plan_hash='0'*64, authorization=c.AUTHORIZATION)
    ledger = sub.submit(spec, execute=True, plan_hash=value['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == value['jobs']
    assert sub.submit(spec, execute=True, plan_hash=value['content_hash'], authorization=c.AUTHORIZATION) == ledger
    assert len(calls) == value['jobs']


def test_ambiguous_submission_never_retried(tmp_path, monkeypatch):
    spec = fixture_spec(tmp_path)
    monkeypatch.setattr(c, 'validate_spec', lambda _: None)
    monkeypatch.setattr(sub, 'check_site', lambda _: None)
    plan = sub.submit(spec)
    calls = []
    def scheduler(args, **kwargs):
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError('reply lost')
        return subprocess.CompletedProcess(args, 0, '901\n', '')
    monkeypatch.setattr(sub.subprocess, 'run', scheduler)
    with pytest.raises(RuntimeError):
        sub.submit(spec, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    with pytest.raises(PermissionError, match='interrupted'):
        sub.submit(spec, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 2


def test_complete_regular_cpu_training_graph(prepared, durable, tmp_path, monkeypatch):
    import torch
    from test_luka_fullsim_science import TinyModel
    from hlt_classification.jetclass2_delphes import runner
    root, p, _, _ = cached_fixture(prepared, durable, monkeypatch)
    spec = fixture_spec(tmp_path, p)
    spec['admission']['prepared_root'] = str(root)
    spec = with_content_hash(spec)
    monkeypatch.setattr(c, 'validate_spec', lambda _: (p, {}))
    monkeypatch.setattr(w, 'execution', lambda *a: 'fixture-not-slurm')
    monkeypatch.setattr(w, 'memory_snapshot', lambda _: dict(rss_bytes=1, peak_rss_bytes=1, gpu_peak_bytes=0))
    monkeypatch.setattr(w, 'new_model', lambda n: (torch.manual_seed(n['initialization_seed']), TinyModel())[1])
    monkeypatch.setattr(w, 'train_kernel', lambda model, train, val, **kwargs:
                        runner.train_kernel(model, train, val, **{**kwargs, 'device':'cpu'}))
    monkeypatch.setattr(w, 'predict', lambda model, train, **kwargs:
                        runner.predict(model, train, **{**kwargs, 'device':'cpu'}))
    monkeypatch.setattr(TinyModel, 'to', lambda self, *a, **k: self)
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for task in c.tasks():
            result = w.run(spec, task['task_id'])
            assert result['final_test_accessed'] is False
            assert w.run(spec, task['task_id']) == result
        assert len(w.result_rows(spec)) == 9
        assert w.completed(spec, 'complete')['result']['fits'] == 9
        for node in c.nodes():
            row = load_json(Path(spec['root'])/('train_'+node['node_id'])/'training_report.json')
            assert 60 <= row['passes'] <= 100
        with pytest.raises(ValueError):
            w.teacher(spec, 'U000', np.zeros((1, 32), np.uint8))
    finally:
        torch.set_num_threads(before)


def test_source_worker_and_guardrails():
    root = Path(__file__).resolve().parents[1]
    worker = (root/'sbatch/run_luka_fullsim_regular.sh').read_text()
    assert worker.index('CUBLAS_WORKSPACE_CONFIG=:4096:8') < worker.index('exec python')
    for text in ('${PROJECT_DIR}/sbatch/common.sh', 'PYTHONNOUSERSITE=1', '${CONDA_PREFIX}/lib', 'atlas_kd_sporc'):
        assert text in worker
    with pytest.raises(ValueError):
        c.check_population(dict(content_hash='a'*64), dict(content_hash='b'*64))
    with pytest.raises(PermissionError):
        c.validate(with_content_hash(dict(c.make_artifact('CACHE'), final_test_accessed=True)), 'CACHE')
    assert len(pre.probe_nodes()) == 7 and pre.probe_nodes()[0]['node_id'] == 'U000_probe'


def evidence_fixture():
    from hlt_classification.jetclass2_delphes.contracts import artifact
    from hlt_classification.jetclass2_delphes.model import model_contract
    gate = dict(content_hash='a'*64, execution_site=c.registration()['execution_site'], source_snapshot=dict(content_hash='b'*64))
    p = dict(content_hash='c'*64, source_snapshot=gate['source_snapshot'],
        summaries={role:{co:dict(maximum=n) for co in cache.s.COORDINATES} for role, n in [('train',5),('validation',9)]})
    manifest = dict(content_hash='d'*64)
    measurements = []
    for node in pre.probe_nodes():
        training = artifact('KERNEL_TRAINING_REPORT', node=node, foundation_sha256=p['content_hash'],
            recipe_sha256=c.recipe()['content_hash'], scientific_fit=False, acceptance_only=True,
            selected_weights_restored=True, final_test_accessed=False, passes=1, runtime_seconds=10., validation=dict(rows=50000))
        measurements.append(dict(node=node, training=training, counts=c.COUNTS,
            cache_seconds=1., inference_seconds=2., stress_batch=256, stress_steps=3,
            validation_stress_batch=256, validation_stress_steps=3, validation_stress_inference_only=True))
    train, reduce = pre.projections(measurements)
    r = c.make_artifact('PREFLIGHT', version=2, parents=dict(gate='a'*64, prepared='c'*64, cache='d'*64),
        counts=c.COUNTS, cpus=6, memory_mb=90000, site=gate['execution_site'], recipe=c.recipe(),
        source_snapshot=gate['source_snapshot'], scientific_fit=False, scientific_metrics_for_selection=False,
        checkpoint_probability_roundtrip=True, technical_checks_passed=True, resource_envelope_ok=True, fits_debug_24h=True,
        job_id='123', parities=dict(installed=artifact('WEAVER_PARITY', model=model_contract(), device='cuda', passed=True, forward_and_feature_and_parameter_gradients=True)),
        peak_rss_bytes=2**30, gpu_peak_bytes=2**30, gpu=dict(total_memory_bytes=40*2**30, name='A100'),
        measurements=measurements, projected_train_minutes=train, projected_reduce_minutes=reduce,
        preparation_source_snapshot=p['source_snapshot'], stress_policy=pre.stress_policy(),
        stress_maxima={role:{co:n for co in cache.s.COORDINATES} for role,n in [('train',5),('validation',9)]},
        outputs=[dict(path=name, bytes=1, sha256='f'*64) for name in ['runtime.json', *(n['node_id']+'.pt' for n in pre.probe_nodes()),
                                                                   *(n['node_id']+'_bank/manifest.json' for n in pre.probe_nodes())]])
    return r, gate, p, manifest


def test_preflight_rejects_small_or_fused_or_unmeasured_evidence():
    r, gate, p, manifest = evidence_fixture()
    assert pre.validate_evidence(r, gate, p, manifest) == r
    for key, value in [('counts', dict(train=100, validation=50)), ('memory_mb', 256000),
                       ('scientific_fit', True), ('resource_envelope_ok', False), ('final_test_accessed', True),
                       ('gpu_peak_bytes', 39*2**30), ('peak_rss_bytes', 100*2**30),
                       ('projected_train_minutes', 1), ('outputs', []), ('measurements', [])]:
        bad = with_content_hash(dict(r, **{key:value}))
        with pytest.raises((ValueError, PermissionError)):
            pre.validate_evidence(bad, gate, p, manifest)
    bad = deepcopy(r)
    bad['measurements'][0]['counts'] = dict(train=64, validation=50)
    with pytest.raises(ValueError, match='coverage'):
        pre.validate_evidence(with_content_hash(bad), gate, p, manifest)
    bad = deepcopy(r)
    bad['measurements'][0]['node']['context_coordinate'] = 'D000'
    with pytest.raises(ValueError):
        pre.validate_evidence(with_content_hash(bad), gate, p, manifest)


@pytest.mark.parametrize('family', ['regular', 'alljets'])
def test_gate_ledger_and_science_creation(tmp_path, monkeypatch, family):
    from hlt_classification.luka_fullsim import regular_campaign as c
    if family == 'alljets':
        from hlt_classification.luka_fullsim import alljets_campaign as c
    gate = c.make_artifact('GATE', root=str(tmp_path), project_dir=str(tmp_path/'project'),
        source_commit='a'*40, source_snapshot=dict(content_hash='b'*64), **c.gate_policy())
    monkeypatch.setattr(c, 'validate_gate', lambda _: None)
    monkeypatch.setattr(sub, 'check_site', lambda _: None)
    calls = []
    def scheduler(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, str(900+len(calls)), '')
    monkeypatch.setattr(sub.subprocess, 'run', scheduler)
    plan = sub.submit(gate)
    ledger = sub.submit(gate, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    assert c.gate_ledger(gate) == ledger
    r = dict(job_id='903', content_hash='d'*64, projected_train_minutes=100, projected_reduce_minutes=20)
    (tmp_path/'preflight').mkdir()
    write_immutable_json(tmp_path/'preflight/preflight.json', r)
    monkeypatch.setattr(c, 'evidence', lambda _: (dict(content_hash='e'*64), dict(content_hash='f'*64), r))
    accounting = []
    def account(job):
        accounting.append(job)
        return dict(job_id=job, state='COMPLETED', exit_code='0:0')
    monkeypatch.setattr(c, 'preflight_accounting', account)
    spec = c.create_science(gate)
    assert set(accounting) == {'901','902','903'}
    assert spec['jobs'] == 16 and spec['counts'] == c.COUNTS
    write_immutable_json(tmp_path/'gate_spec.json', gate)
    assert c.validate_spec(spec)[1] == r
    event = tmp_path/'submission/submission_ledger_journal/0001_prepare.json'
    event.write_bytes(event.read_bytes()+b'corrupt')
    with pytest.raises(ValueError):
        c.gate_ledger(gate)


def test_selected_only_parallel_reader_firewall(dataset, tmp_path, monkeypatch):
    container, inventory, splits = dataset
    members = splits['memberships']['train'][:2]
    allowed = {inventory['files'][m['file_index']]['path']:set(sum(m['entries_by_class'], [])) for m in members}
    import uproot
    original = uproot.behaviors.TBranch.HasBranches.arrays
    calls = []
    def guarded(tree, expressions, *args, **kwargs):
        name = Path(tree.file.file_path).relative_to(container/'raw').as_posix()
        assert name in allowed and set(range(kwargs['entry_start'], kwargs['entry_stop'])) <= allowed[name]
        calls.append(name)
        return original(tree, expressions, *args, **kwargs)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, 'arrays', guarded)
    cache._init(container, inventory, splits, config(), tmp_path)
    for m in members:
        cache._work(('train', m['file_index']))
    assert calls
    other_file = next(i for i in range(len(inventory['files'])) if i not in {m['file_index'] for m in splits['memberships']['train']})
    with pytest.raises(PermissionError):
        cache._work(('final_test', other_file))
    with pytest.raises(PermissionError):
        cache._work(('train', other_file))


def test_cpu_prepare_completion_and_consumed_test_flag(prepared, tmp_path, monkeypatch):
    _, p, foundation, inventory, splits = prepared
    gate = c.make_artifact('GATE', root=str(tmp_path), counts=p['counts'],
        source_snapshot=p['source_snapshot'], input_container=foundation['input_container'])
    monkeypatch.setattr(c, 'validate_gate', lambda _: None)
    monkeypatch.setattr(c, 'check_population', lambda *a: None)
    monkeypatch.setattr(c, 'load_foundation', lambda _: (foundation, inventory, splits))
    monkeypatch.setattr(cache.s, 'load_foundation', lambda _: (foundation, inventory, splits))
    original = cache.build_rows
    monkeypatch.setattr(cache, 'build_rows', lambda *a, **kw: original(*a, **{**kw, 'workers':1}))
    manifest = cache.prepare(gate)
    assert manifest['serial_reference_passed'] is True and manifest['historical_test_evaluated'] is True
    assert manifest['final_test_accessed'] is False
    loaded, new_p, _, _ = cache.metadata(tmp_path/'prepared')
    assert loaded == manifest
    assert new_p['summaries'] == p['summaries']
    assert new_p['conventions'] == c.conventions(new_p['audit']['content_hash'])
    with pytest.raises(FileExistsError):
        cache.prepare(gate)


@pytest.mark.parametrize('fail_stress', [False, True, 'validation'])
@pytest.mark.parametrize('profile', ['ordinary', 'recovery', 'alljets'])
def test_full_preflight_orchestration_roundtrips(prepared, durable, tmp_path, monkeypatch, fail_stress, profile):
    """Real cache and bank math; explicitly mock Slurm/GPU model execution only."""
    import torch
    from test_luka_fullsim_science import TinyModel
    from hlt_classification.context_fusion import inputs, worker
    from hlt_classification.jetclass2_delphes import execution, model, acceptance, runner
    from hlt_classification.cms2jc2_response import measurement
    from hlt_classification.jetclass2_delphes.contracts import artifact
    root, p, inventory, splits = cached_fixture(prepared, durable, monkeypatch)
    authenticated = cache.metadata(root)
    manifest = deepcopy(authenticated[0])
    p = deepcopy(p)
    p['foundation_root'] = str(tmp_path/'foundation')
    gate = dict(root=str(tmp_path), content_hash='b'*64, source_snapshot=p['source_snapshot'], execution_site=c.registration()['execution_site'])
    recovered = profile == 'recovery'
    if recovered:
        original = dict(gate)
        write_immutable_json(tmp_path/'original_gate.json', original)
        gate = dict(gate, contract='LUKA_FULLSIM_REGULAR_RECOVERY_GATE/v1', content_hash='c'*64,
            source_snapshot=dict(content_hash='e'*64),
            reuse=dict(prepared_root=str(root), original_gate=dict(path=str(tmp_path/'original_gate.json'))))
    if profile == 'alljets':
        gate.update(contract='LUKA_FULLSIM_ALLJETS_GATE/v1',
                    preflight_cache_policy='one_coordinate_train_validation_at_a_time')
    policy = c.for_spec(gate)
    monkeypatch.setattr(policy, 'COUNTS', p['counts'])
    monkeypatch.setattr(policy, 'validate_gate', lambda _: None)
    monkeypatch.setattr(policy, 'check_population', lambda *a: None)
    monkeypatch.setattr(policy, 'conventions', lambda _: p['conventions'])
    monkeypatch.setattr(cache, 'metadata', lambda _: (manifest, p, inventory, splits))
    actual_load = cache.load
    loaded = []
    def load_once(location, **kw):
        loaded.append((kw['role'], kw['coordinate']))
        assert Path(location) == (root if recovered else tmp_path/'prepared')
        return actual_load(root, **kw)
    monkeypatch.setattr(cache, 'load', load_once)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    monkeypatch.setattr(execution, 'allocation', lambda _: ('123', 6, 90000))
    monkeypatch.setattr(execution, 'gpu_identity', lambda: dict(name='A100', total_memory_bytes=40*2**30))
    monkeypatch.setattr(model, 'installed_environment', lambda: dict(mocked_local=True))
    monkeypatch.setattr(acceptance, 'installed_parity', lambda *a, **k: artifact('WEAVER_PARITY', model=model.model_contract(),
        passed=True, device='cuda', forward_and_feature_and_parameter_gradients=True))
    monkeypatch.setattr(pre, 'memory_snapshot', lambda _: dict(rss_bytes=1000, peak_rss_bytes=2000, gpu_peak_bytes=3000))
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(worker, 'clear', lambda: None)
    monkeypatch.setattr(inputs, 'new_model', lambda n: (torch.manual_seed(n['initialization_seed']), TinyModel())[1])
    events = []
    def stress(model, train):
        events.append('stress')
        if fail_stress is True:
            raise MemoryError('injected stress failure')
    monkeypatch.setattr(worker, 'stress', stress)
    actual_validation_stress = pre.validation_stress
    def validation_stress(model, val):
        events.append('validation_inference')
        if fail_stress == 'validation':
            raise MemoryError('injected validation stress failure')
        actual_validation_stress(model, val, device='cpu')
    monkeypatch.setattr(pre, 'validation_stress', validation_stress)
    actual_train, actual_predict = runner.train_kernel, runner.predict
    def timed_train(model, train, val, **kw):
        report, state = actual_train(model, train, val, **{**kw, 'device':'cpu'})
        # Tiny CPU fixtures can complete below Windows' timer resolution.
        return with_content_hash(dict(report, runtime_seconds=max(.001, report['runtime_seconds']))), state
    monkeypatch.setattr(runner, 'train_kernel', timed_train)
    monkeypatch.setattr(runner, 'predict', lambda model, train, **kw:
        actual_predict(model, train, **{**kw, 'device':'cpu'}))
    class Measured:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def report(self):
            return dict(sampled_peak_tree_rss_bytes=1000)
    monkeypatch.setattr(measurement, 'Measurement', Measured)
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        if fail_stress:
            with pytest.raises(MemoryError, match='injected'):
                pre.run(gate)
            assert not (tmp_path/'preflight/preflight.json').exists()
            assert events == (['stress', 'validation_inference'] if fail_stress == 'validation' else ['stress'])
        else:
            result = pre.run(gate)
            assert result['checkpoint_probability_roundtrip'] is True
            assert result['scientific_fit'] is False and len(result['measurements']) == 7
            assert events == ['stress','validation_inference']*7
            assert result['schema_version'] == 2 and set(result['stress_maxima']) == {'train','validation'}
            if profile == 'alljets':
                assert loaded == [(role, n['coordinate']) for n in pre.probe_nodes() for role in ('train', 'validation')]
                assert result['historical_test_repurposed_for_training'] is True
    finally:
        torch.set_num_threads(before)

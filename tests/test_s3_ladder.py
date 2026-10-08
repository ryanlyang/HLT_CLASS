"""Synthetic real readers/processes/CE/KD; no genuine GPU acceptance claimed."""
import copy
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import pytest
import torch

from test_literature_proxy import particles
from test_correlated_topology import parent, final_pilot, study, dataset, released, topology_request, gated
from test_gap_sweep import _tiny_spawn_initializer
from test_jetclass2_delphes_training import TinyModel
from hlt_classification.s3_ladder import views as v, cache, campaign as c, worker as w
from hlt_classification.s3_ladder.contracts import artifact, validate, write_json, load_json, file_ref
from hlt_classification.gap_sweep import campaign as screen, worker as sw, cache as scache
from hlt_classification.gap_sweep.contracts import artifact as screen_artifact
from hlt_classification.data.cache_contracts import with_content_hash, sha256_file


def test_frozen_endpoint_and_interpolation_do_not_restore_lost_information():
    from hlt_classification.correlated_topology.kernel import generate as base_generate
    from test_correlated_topology import frozen
    from hlt_classification.gap_sweep.kernel import generate
    offline = particles((0, 0, 1, 2, 3, 4)*10, [1., 3., 5., 7., 11., 15.]*10)
    base = base_generate(offline, 'frozen', frozen())
    endpoint = generate(base.particles, 'frozen', 'S3')[0]
    assert len(endpoint) < len(offline)
    assert endpoint.valid.sum() < offline.valid.sum()
    end = v.build_view(proxy=base.particles, identity='frozen', coordinate='D000')
    for name in ('p4', 'charge', 'category', 'tracking', 'valid'):
        np.testing.assert_array_equal(getattr(end, name), getattr(endpoint, name))
    indices = {key: i for i, key in enumerate(base.particles.keys)}
    original = offline.tracking[base.mapping[[indices[key] for key in end.keys]]]
    for coord, fraction in (('D066', 1/3), ('D033', 2/3)):
        out = v.build_view(proxy=base.particles, offline=offline, identity='frozen', coordinate=coord, mapping=base.mapping)
        assert out.keys == end.keys
        for name in ('p4', 'charge', 'category', 'valid'):
            np.testing.assert_array_equal(getattr(out, name), getattr(end, name))
        eligible = end.valid[:, :2] & end.valid[:, 2:]
        expected = original[:, :2]+fraction*(end.tracking[:, :2]-original[:, :2])
        np.testing.assert_array_equal(out.tracking[:, :2][eligible], expected[eligible])
        np.testing.assert_array_equal(out.tracking[~out.valid], np.zeros((~out.valid).sum()))
    with pytest.raises(ValueError):
        v.build_view(proxy=base.particles, identity='frozen', coordinate='D066')
    with pytest.raises(ValueError):
        v.build_view(proxy=base.particles, offline=offline, identity='frozen', coordinate='D033', mapping=base.mapping.astype(float))
    with pytest.raises(ValueError):
        v.build_view(proxy=base.particles, identity='frozen', coordinate='D050')


def test_endpoint_does_not_touch_offline_empty_and_invalid_tracking():
    class Forbidden:
        def __getattribute__(self, name):
            raise AssertionError('D000 must not touch offline')
    p = particles((1, 2), [3., 4.])
    for proxy in (p, p.take([])):
        out = v.build_view(proxy=proxy, offline=Forbidden(), mapping=Forbidden(), identity='x', coordinate='D000')
        assert not out.valid.any()
        for coord in ('D066', 'D033'):
            mid = v.build_view(proxy=proxy, offline=proxy, identity='x', coordinate=coord,
                               mapping=np.arange(len(proxy), dtype=np.int32))
            assert mid.keys == out.keys and not mid.valid.any()


@pytest.mark.parametrize('role', ['final_test', 'test', 'validation_extra'])
def test_forbidden_roles_fail_before_reading(role):
    with pytest.raises(PermissionError):
        cache.prepare({}, 'D000', role, workers=1, input_identity='a'*64, max_ram_bytes=1)
    with pytest.raises(PermissionError):
        cache.identities({}, role)


def test_source_drift_rejected_without_touching_historical_sources(tmp_path, monkeypatch):
    path = tmp_path/'old.py'; path.write_text('old')
    sweep = dict(source=dict(files={'old.py': sha256_file(path)}, content_hash='a'*64))
    path.write_text('drift')
    monkeypatch.setattr(c, '_source', lambda *a: None)
    monkeypatch.setattr(c.subprocess, 'run', lambda *a, **kw: pytest.fail('Fail before Git'))
    with pytest.raises(ValueError, match='Inherited scientific source'):
        c.source_lock(sweep, tmp_path, 'f'*40)


class CpuModel(TinyModel):
    def to(self, *args, **kwargs):
        return super().to('cpu')


def hardware_substitutes(monkeypatch, parent_campaign):
    from hlt_classification.jetclass2_delphes.runner import train_kernel, predict
    from hlt_classification.jetclass2_delphes.contracts import artifact as training_artifact
    # Windows Python 3.10 monotonic may quantize a tiny CPU pass to zero.
    # High-resolution test clock only; production acceptance remains strict.
    monkeypatch.setattr(time, 'monotonic', time.perf_counter)
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 1024)
    for module in (w, sw):
        monkeypatch.setattr(module, '_execution_gate', lambda *a: None)
        monkeypatch.setattr(module, 'DelphesParticleTransformer', CpuModel)
        monkeypatch.setattr(module, 'installed_environment', lambda: parent_campaign['runtime_profile']['installed_environment'])
        monkeypatch.setattr(module, 'gpu_identity', lambda: parent_campaign['runtime_profile']['gpu'])
        monkeypatch.setattr(module, 'installed_parity', lambda *a, **kw: training_artifact('WEAVER_PARITY',
            model=parent_campaign['model'], device='cuda', passed=True, forward_and_feature_and_parameter_gradients=True,
            final_test_accessed=False))
        monkeypatch.setattr(module, 'train_kernel', lambda *a, **kw: train_kernel(*a, **dict(kw, device='cpu')))
    monkeypatch.setattr(w, 'predict', lambda *a, **kw: predict(*a, **dict(kw, device='cpu')))


@pytest.fixture
def completed_screen(gated, tmp_path, monkeypatch):
    from hlt_classification.cms_proxy_ladder import correlated_topology as adapter, production as p
    torch.set_num_threads(1)
    parent_campaign = adapter.create_campaign(gate_root=gated['gate_root'], campaign_root=tmp_path/'original')
    monkeypatch.setattr(p, '_execution_gate', lambda *a: None)
    monkeypatch.setattr(p, 'DelphesParticleTransformer', TinyModel)
    original = p.prepare_cache
    monkeypatch.setattr(p, 'prepare_cache', lambda *a, **kw: original(*a, **dict(kw, workers=1)))
    for task in ('train_OFFLINE', 'train_M0HLT', 'reduce_OFFLINE'):
        p.run_task(parent_campaign, task, attempt='test', device='cpu')
    hardware_substitutes(monkeypatch, parent_campaign)
    monkeypatch.setattr(screen, 'BASE_COMMIT', 'e'*40)
    monkeypatch.setattr(screen, 'COUNTS', parent_campaign['foundation']['role_counts'])
    monkeypatch.setattr(screen, 'source_lock', lambda par, project, commit: screen_artifact('SOURCE',
        commit=commit, files={}, unchanged_parent_source=par['source']['content_hash']))
    s = screen.create(parent_spec=tmp_path/'original/campaign_spec.json', project_dir=gated['project_dir'],
        source_commit=c.SCREEN_COMMIT, root=tmp_path/'screen')
    monkeypatch.setattr(scache, '_limit_worker_threads', _tiny_spawn_initializer)
    original_cache = scache.prepare
    monkeypatch.setattr(scache, 'prepare', lambda *a, **kw: original_cache(*a, **dict(kw, workers=1)))
    sw.run(s, 'preflight')
    for name in screen.CANDIDATES:
        sw.run(s, 'fit_'+name)
    # The tiny classifier is not calibrated to select S3. Simulate only the
    # already-tested screening decision here, not any reader/training/source check.
    actual_choose = screen.choose
    def chosen(offline, metrics):
        return dict(actual_choose(offline, metrics), selected='S3', status='development_selection_only')
    monkeypatch.setattr(screen, 'choose', chosen)
    sw.run(s, 'summary')
    monkeypatch.setattr(c, 'source_lock', lambda sweep, project, commit: artifact('SOURCE',
        commit=commit, files={}, unchanged_screen_source=sweep['source']['content_hash']))
    spec = c.create(sweep_spec=tmp_path/'screen/study_spec.json', project_dir=gated['project_dir'],
        source_commit='f'*40, root=tmp_path/'s3')
    return spec, parent_campaign, s


def test_full_saved_workflow_and_corruption(completed_screen, monkeypatch):
    spec, parent, sweep = completed_screen
    assert c.validate_spec(spec) == parent
    assert c.screen_inputs(spec['sweep_spec']['path'])[0] == sweep
    for field, value in (('views', {}), ('counts', {}), ('workers', 4), ('frozen', {}), ('input_identity', 'a'*64)):
        with pytest.raises(ValueError):
            c.validate_spec(with_content_hash(dict(spec, **{field: value})))
    with pytest.raises(FileNotFoundError):
        c.plan(spec, 'science')
    c.submit(spec, mode='gate')
    monkeypatch.setattr(cache, '_limit_worker_threads', _tiny_spawn_initializer)
    one, _ = cache.prepare(parent, 'D066', 'train', workers=1, input_identity=spec['input_identity'], max_ram_bytes=2**30)
    two, _ = cache.prepare(parent, 'D066', 'train', workers=2, input_identity=spec['input_identity'], max_ram_bytes=2**30)
    for key in ('features', 'vectors', 'mask', 'labels', 'identities'):
        np.testing.assert_array_equal(one.batch(np.arange(len(one)))[key], two.batch(np.arange(len(two)))[key])
    with pytest.raises(MemoryError):
        cache.prepare(parent, 'D066', 'train', workers=2, input_identity=spec['input_identity'], max_ram_bytes=1)
    real_prepare = cache.prepare
    monkeypatch.setattr(cache, 'prepare', lambda *a, **kw: real_prepare(*a, **dict(kw, workers=1)))
    acceptance = w.run(spec, 'preflight')
    assert acceptance == w.run(spec, 'preflight')
    assert acceptance['endpoint_sha256'] == spec['frozen']['endpoint_sha256']
    assert acceptance['acceptance']['acceptance_only']
    for field, value in (('endpoint_sha256', {}), ('counts', {}), ('train_minutes', 1441),
                          ('gpu_peak_bytes', 10**15), ('cache_bytes', 10**15), ('installed_environment', {})):
        with pytest.raises(ValueError):
            c.validate_preflight(with_content_hash(dict(acceptance, **{field: value})), spec, parent)
    plan = c.submit(spec, mode='science')
    assert len(plan['commands']) == 7
    assert not (Path(spec['root'])/'submission_science/submission_ledger.json').exists()
    with pytest.raises(PermissionError):
        c.submit(spec, mode='science', execute=True, authorization='wrong', plan_hash=plan['content_hash'])
    with pytest.raises(ValueError, match='All four'):
        c.summarize(spec, parent)
    for task in c.tasks(spec):
        result = w.run(spec, task['task_id'])
        assert result == w.run(spec, task['task_id'])
        if task['kind'] == 'fit':
            _, training = c.completed_fit(spec, parent, task['node_id'])
            assert training['scientific_fit'] and training['passes'] >= 60
    rows = c.result_rows(spec, parent, complete=True)
    assert len(rows) == 6 and rows[1]['node_id'] == 'S3'
    assert rows[1]['recovery']['accuracy'] in (0., None)
    assert all(r['state'] == 'COMPLETE' for r in rows)
    assert not (Path(spec['root'])/'fit_OFFLINE').exists()
    assert not (Path(spec['root'])/'reduce_OFFLINE').exists()
    reducer = Path(spec['root'])/('reduce_'+spec['graph']['reducers'][0])
    assert not (reducer/'validation').exists()
    (reducer/'train/000000000.npz').write_bytes(b'broken')
    with pytest.raises(ValueError):
        c.completed_reducer(spec, parent, spec['graph']['reducers'][0])
    with pytest.raises(ValueError):
        w.run(spec, 'summary')


def graph_fixture():
    from hlt_classification.cms_proxy_ladder.campaign import coordinate, paired_seed
    nodes = [dict(node_id='OFFLINE', branch='CONTROL', coordinate='OFFLINE', teacher=None)]
    for branch, coords in (('DIRECT', ['D000']), ('COARSE', ['D066', 'D033', 'D000'])):
        teacher, previous = 'OFFLINE', 'OFFLINE'
        for coord in coords:
            name = f'CORRHT_{branch}_{coord}_from_{previous}'
            u, f = coordinate(coord)
            nodes.append(dict(node_id=name, branch=branch, coordinate=coord, teacher=teacher,
                u=[u.numerator, u.denominator], f=[f.numerator, f.denominator],
                initialization_seed=paired_seed(coord, 'initialization'), sampler_seed=paired_seed(coord, 'sampler'),
                deployable=coord == 'D000'))
            teacher, previous = name, coord
    return c.graph(dict(scientific_plan=dict(nodes=nodes)))


def test_graph_and_real_journal_dependency_resolution(tmp_path, monkeypatch):
    from hlt_classification.jetclass2_delphes.execution import execution_site
    from hlt_classification.scouting import hcwdl_exact_dag_submission as dag
    spec = dict(root=str(tmp_path), project_dir='/exact/project', content_hash='a'*64,
        graph=graph_fixture(), execution_site=execution_site('sporc_a100_debug'), cpus=6, memory_mb=90000)
    monkeypatch.setattr(c, 'validate_spec', lambda *a: {})
    monkeypatch.setattr(c, 'preflight', lambda *a: dict(content_hash='b'*64, train_minutes=180, reduce_minutes=30))
    gate = c.plan(spec, 'gate')
    assert '--time=02:00:00' in gate['commands'][0]['command']
    assert '--gres=gpu:a100:1' in gate['commands'][0]['command']
    plan = c.submit(spec, mode='science')
    rows = plan['commands']
    assert len(rows) == 7 and rows[0]['dependencies'] == rows[1]['dependencies'] == []
    assert '--gres=gpu:a100:1' not in rows[-1]['command']
    assert len(rows[-1]['dependencies']) == 4
    direct, coarse_final = spec['graph']['nodes'][0], spec['graph']['nodes'][-1]
    assert direct['initialization_seed'] == coarse_final['initialization_seed']
    assert direct['sampler_seed'] == coarse_final['sampler_seed']
    assert spec['graph']['recipe']['kd_weight'] == .75
    assert not any('OFFLINE' == n['node_id'] for n in spec['graph']['nodes'])
    calls = []
    monkeypatch.setattr(c, 'check_submission_site', lambda *a: None)
    def submit(argv, **kw):
        assert not any(k.startswith(('SBATCH_', 'SLURM_')) for k in kw['env'])
        assert all('${JOB_' not in arg for arg in argv)
        calls.append(argv)
        return SimpleNamespace(stdout=str(900+len(calls))+'\n')
    monkeypatch.setattr(dag.subprocess, 'run', submit)
    monkeypatch.setenv('SBATCH_PARTITION', 'wrong')
    for options in (dict(plan_hash='c'*64, authorization=c.AUTHORIZATION),
                    dict(plan_hash=plan['content_hash'], authorization='wrong')):
        with pytest.raises(PermissionError):
            c.submit(spec, mode='science', execute=True, **options)
    ledger = c.submit(spec, mode='science', execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 7 and ledger['jobs']['summary'] == '907'
    assert '--dependency=afterok:901:902:904:906' in calls[-1]
    assert c.submit(spec, mode='science', execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION) == ledger
    assert len(calls) == 7


def test_ambiguous_submission_and_worker_claims_are_not_retried(tmp_path, monkeypatch):
    spec = dict(root=str(tmp_path), content_hash='a'*64, graph=graph_fixture())
    plan = artifact('PLAN', commands=[dict(task_id='preflight', dependencies=[], command=['sbatch', '--wrap=true'])])
    monkeypatch.setattr(c, 'plan', lambda *a: plan)
    monkeypatch.setattr(c, 'check_submission_site', lambda *a: None)
    c.submit(spec, mode='gate')
    (tmp_path/'submission_gate/live_submission_claim.json').write_text('{}')
    with pytest.raises(PermissionError, match='interrupted'):
        c.submit(spec, mode='gate', execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    monkeypatch.setattr(c, 'validate_spec', lambda *a: {})
    (tmp_path/'preflight.claim').write_text('{}')
    monkeypatch.setattr(w, 'measure', lambda *a: pytest.fail('Do not rerun'))
    with pytest.raises(FileExistsError):
        w.run(spec, 'preflight')
    with pytest.raises(ValueError):
        w.run(spec, 'fit_OTHER')


def test_non_s3_selection_is_not_reinterpreted(tmp_path, monkeypatch):
    sweep = dict(root=str(tmp_path), source_commit=c.SCREEN_COMMIT)
    write_json(tmp_path/'study_spec.json', sweep)
    write_json(tmp_path/'summary.json', dict(selected=None))
    monkeypatch.setattr(screen, 'validate_spec', lambda *a: {})
    monkeypatch.setattr(screen, 'summarize', lambda *a: dict(selected=None))
    with pytest.raises(ValueError, match='must select S3'):
        c.screen_inputs(tmp_path/'study_spec.json')


def test_protected_roots_and_final_test_metadata_are_rejected(tmp_path, monkeypatch):
    protected = tmp_path/'historical'
    protected.mkdir()
    monkeypatch.setattr(screen, 'protected_roots', lambda *a: [protected])
    sweep = dict(root=str(tmp_path/'screen'), project_dir=str(tmp_path/'old-source'))
    for target in (protected, protected/'child', tmp_path, Path(sweep['root'])/'bad'):
        with pytest.raises(PermissionError):
            c._protect(target.resolve(), {}, sweep, tmp_path/'new-source')
    c._protect((tmp_path/'new-output').resolve(), {}, sweep, tmp_path/'new-source')
    with pytest.raises(PermissionError):
        validate(with_content_hash(dict(artifact('SPEC'), final_test_accessed=True)), 'SPEC')


def test_d000_digest_mismatch_blocks_fits(monkeypatch):
    parent = dict(foundation={})
    spec = dict(memory_mb=90000, workers=6, input_identity='a'*64,
                frozen=dict(endpoint_sha256=dict(train='b'*64, validation='b'*64)))
    monkeypatch.setattr(cache, 'cache_budgets', lambda *a: dict(train=1, validation=1))
    monkeypatch.setattr(cache, 'prepare', lambda *a, **kw: (None, dict(endpoint_sha256='c'*64)))
    with pytest.raises(ValueError, match='does not match completed baseline'):
        w._caches(spec, parent, 'D000')

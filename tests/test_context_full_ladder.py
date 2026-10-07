"""Synthetic fixtures only: full CONTEXT_V1 population, never RC submissions."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from test_literature_context_consumer import parent, parent_v2, final_pilot, study, relocated, consumer
import test_context_ladder as small_tests
from hlt_classification.cms_proxy_ladder import context as small, context_full as full
from hlt_classification.cms_proxy_ladder import data, cache, cache_full, release, gate, submission, production
from hlt_classification.cms_proxy_ladder.campaign import task_graph
from hlt_classification.cms_proxy_ladder.context_inputs import build_inputs
from hlt_classification.data.cache_contracts import load_json, with_content_hash
from hlt_classification.jetclass2_delphes.campaign import recipe


def initialize_synthetic_worker(counts, digest):
    # Spawned Windows workers do not inherit pytest's fixture monkeypatches.
    # Only this test initializer changes the synthetic population/source pin.
    full.COUNTS = {role: counts[role] for role in ('train', 'validation')}
    full.MANIFEST_SHA256 = digest
    consumer.population.COUNTS = counts
    consumer.population.registered.COUNTS = counts
    consumer.population.registered.PROFILE = 'TRAIN_44'
    cache._limit_worker_threads()


def test_separate_full_policy_preserves_small_defaults_and_science():
    assert small.COUNTS == dict(train=100000, validation=50000)
    assert full.COUNTS == dict(train=1000000, validation=250000)
    assert full.DOMAIN != small.DOMAIN
    f = dict(schema_version=6, release=dict(schema_version=6), role_counts=full.COUNTS, content_hash='a'*64)
    plan = full.scientific_plan(f)
    old = small.scientific_plan(dict(f, schema_version=3, release=dict(schema_version=3), role_counts=small.COUNTS))
    assert plan['schema_version'] == 8 and plan['recipe'] == recipe() == old['recipe']
    assert len(task_graph(plan)) == 16 and len(plan['nodes']) == 9
    assert len(plan['probability_publications']) == 5 and list(plan['branches']) == ['DIRECT', 'COARSE']
    for a, b in zip(plan['nodes'], old['nodes'], strict=True):
        for field in ('coordinate', 'initialization_seed', 'sampler_seed', 'deployable', 'u', 'f'):
            assert a[field] == b[field]
    assert plan['imported_models'] == [] and plan['final_test_evaluation'] is False
    with pytest.raises(ValueError, match='1M/250k'):
        full.scientific_plan(dict(f, role_counts=small.COUNTS))
    assert full.gate_tasks()[-1] == dict(task_id='preflight', kind='gpu', dependencies=['build_foundation'],
                                      cpus=6, memory_mb=180000, minutes=720)


def test_ordered_pool_bounds_outstanding_work():
    class Pool:
        active = peak = 0
        def submit(self, function, arg):
            self.active += 1
            self.peak = max(self.peak, self.active)
            owner = self
            class Future:
                def result(self):
                    owner.active -= 1
                    return function(arg)
            return Future()
    pool = Pool()
    assert list(cache_full.ordered_results(pool, lambda x: x*x, range(15), 3)) == [x*x for x in range(15)]
    assert pool.active == 0 and pool.peak == 3


def test_full_sized_conservative_budget_and_sealed_roles(monkeypatch):
    counts = dict(train=1000000, validation=250000)
    arrays = dict(role=np.r_[np.zeros(counts['train'], np.uint8), np.ones(counts['validation'], np.uint8)],
                  source_file=np.arange(1250000, dtype=np.int32)//10000)
    monkeypatch.setattr(release, 'load_bank', lambda *a, **k: arrays)
    f = dict(schema_version=6, role_counts=counts, inputs=dict(capacity=512),
             release=dict(source_files=[{}]*125), release_root='unused',
             assignment_slots=50000000, cache_preparation='source_file_bounded/v1')
    budgets = cache.cache_budgets(f, 180000, 6)
    assert sum(budgets.values()) == 180000*1024**2*3//4
    for role in counts:
        assert counts[role]*512*84 < cache.preparation_bound(f, role, 6) <= budgets[role]
    with pytest.raises(MemoryError):
        cache.cache_budgets(f, 32000, 6)
    with pytest.raises(PermissionError):
        cache.preparation_bound(f, 'final_test', 6)
    with pytest.raises(ValueError, match='subsample'):
        cache.preparation_bound(f, 'train', 6, population_selection={})


@pytest.fixture
def ready(relocated, tmp_path, monkeypatch):
    root, digest, original = relocated
    monkeypatch.setattr(full, 'COUNTS', {r: original['counts'][r] for r in ('train', 'validation')})
    monkeypatch.setattr(full, 'MANIFEST_SHA256', digest)
    monkeypatch.setattr(small_tests, 'l', full)
    monkeypatch.setattr(gate, 'source_lock', small_tests.fake_source)
    project = tmp_path/'project'
    project.mkdir()
    spec = full.create_gate(study_root=root/consumer.PROXY_DIRECTORY,
        offline_root=root/consumer.SNAPSHOT_DIRECTORY/'jetclass2', provenance_root=root/'oscar_provenance_v1',
        gate_root=tmp_path/'gate', project_dir=project, source_commit='e'*40)
    with pytest.raises(FileNotFoundError):
        full.create_campaign(gate_root=spec['gate_root'], campaign_root=tmp_path/'premature')
    # Deny sealed block access even though fixture generated dummy test blocks.
    forbidden = set()
    for shard in original['shards']:
        if shard['role'] == 'final_test':
            record = load_json(root/consumer.PROXY_DIRECTORY/'shards'/f"{shard['shard_id']}.json")
            forbidden.update((root/consumer.PROXY_DIRECTORY/b['relative']).resolve() for b in record['blocks'])
    native_open = Path.open
    def guarded(path, *a, **k):
        assert path.resolve() not in forbidden, 'Final-test particle block opened'
        return native_open(path, *a, **k)
    monkeypatch.setattr(Path, 'open', guarded)
    gate.run_gate_task(spec, 'authenticate_release')
    build = gate.build_foundation
    monkeypatch.setattr(gate, 'build_foundation', lambda *a, **k: build(*a, **dict(k, workers=1)))
    gate.run_gate_task(spec, 'build_foundation')
    froot = Path(spec['gate_root'])/'foundation'
    f = load_json(froot/'foundation.json')
    profile = small_tests.profile_for(spec, f)
    profile = with_content_hash(dict(profile, contract='JETCLASS2_CMS_PROXY_LADDER_RUNTIME_PROFILE/v11',
        schema_version=11, memory_mb=180000, cache_budgets=cache.cache_budgets(f, 180000, 6)))
    return spec, f, froot, profile


def test_full_release_foundation_cache_parity_and_rejections(ready, tmp_path, monkeypatch):
    spec, f, froot, profile = ready
    assert f['schema_version'] == 6 and f['release']['counts'] == full.COUNTS
    assert f['release']['total_rows'] == sum(full.COUNTS.values())
    assert f['views']['schema_version'] == 3 and f['inputs']['schema_version'] == 2
    assert f['cache_preparation'] == 'source_file_bounded/v1'
    assert data.validate_foundation(f, root=froot)
    with pytest.raises(ValueError):
        small.validate_request(spec['request'])
    monkeypatch.setattr(data, 'from_jc2', lambda *a, **k: pytest.fail('No CMS unit conversion'))
    for role in full.COUNTS:
        rows = list(data.iter_paired(f['release'], release_root=Path(f['release_root']), role=role))
        for coordinate in ('D000', 'OFFLINE', 'U000', 'U050', 'U100', 'D066', 'D033'):
            c = cache.prepare_cache(f, foundation_root=froot, role=role, coordinate=coordinate,
                                    workers=1, max_ram_bytes=profile['cache_budgets'][role])
            assert len(c) == full.COUNTS[role]
            assert [bytes(i).hex() for i in c.identities] == [r.identity for r in rows]
            if coordinate in ('D000', 'OFFLINE'):
                expected = np.concatenate([build_inputs(r.proxy if coordinate == 'D000' else r.offline).features for r in rows])
                np.testing.assert_array_equal(np.concatenate([b.features for b in c.blocks]), expected)
    # Real spawned cache workers, preserving identity and exact bytes.
    def pool(**kwargs):
        return ProcessPoolExecutor(**dict(kwargs, initializer=initialize_synthetic_worker,
                                          initargs=(dict(full.COUNTS, final_test=11), full.MANIFEST_SHA256)))
    monkeypatch.setattr(cache, 'ProcessPoolExecutor', pool)
    parallel = cache.prepare_cache(f, foundation_root=froot, role='train', coordinate='D000',
                                   workers=2, max_ram_bytes=profile['cache_budgets']['train'])
    serial = cache.prepare_cache(f, foundation_root=froot, role='train', coordinate='D000',
                                 workers=1, max_ram_bytes=profile['cache_budgets']['train'])
    np.testing.assert_array_equal(parallel.identities, serial.identities)
    for a, b in zip(parallel.blocks, serial.blocks, strict=True):
        np.testing.assert_array_equal(a.features, b.features)
        np.testing.assert_array_equal(a.vectors, b.vectors)
    monkeypatch.setattr(data, 'ProcessPoolExecutor', pool)
    parallel_foundation = data.build_foundation(f['release'], release_root=Path(f['release_root']),
        output_root=tmp_path/'parallel_foundation', workers=2)
    assert parallel_foundation['assignments']['sha256'] == f['assignments']['sha256']
    assert parallel_foundation['identity_sha256'] == f['identity_sha256']
    with pytest.raises(ValueError, match='slot'):
        data.validate_foundation(with_content_hash(dict(f, assignment_slots=f['assignment_slots']+1)), root=froot)
    with pytest.raises(PermissionError):
        list(data.iter_paired(f['release'], release_root=Path(f['release_root']), role='final_test'))
    with pytest.raises(ValueError):
        full.validate_request(with_content_hash(dict(spec['request'], counts=dict(train=1, validation=1))))


def test_full_gate_preflight_dispatch_and_exact_science_dag(ready, tmp_path, monkeypatch):
    spec, f, froot, profile = ready
    assert gate.validate_profile(profile, foundation=f, spec=spec)
    # Exercise the actual preflight body with mocked hardware/kernel, not a
    # production GPU claim. It must run full CE + KD + both endpoint caches.
    import itertools
    from hlt_classification.jetclass2_delphes import acceptance
    calls = []
    class FakeCache:
        nbytes = 1024
        def __init__(self, role):
            self.role = role
            self.identities = np.arange(full.COUNTS[role])
        def __len__(self):
            return full.COUNTS[self.role]
    def prepare(foundation, **kw):
        calls.append((kw['role'], kw['coordinate']))
        assert kw['population_selection'] is None and kw['workers'] == 6
        return FakeCache(kw['role'])
    def train(model, train, val, **kw):
        assert len(train) == full.COUNTS['train'] and len(val) == full.COUNTS['validation']
        assert kw['acceptance_passes'] == 1
        if kw['node']['teacher']:
            assert kw['teacher_probabilities'].shape == (len(train), 11)
            return profile['acceptance_kd_training_report'], {}
        return profile['acceptance_training_report'], {}
    monkeypatch.setattr(gate, 'allocation', lambda _: ('123', 6, 180000))
    monkeypatch.setattr(gate, 'installed_environment', lambda: profile['installed_environment'])
    monkeypatch.setattr(gate, 'gpu_identity', lambda: profile['gpu'])
    monkeypatch.setattr(gate, 'prepare_cache', prepare)
    monkeypatch.setattr(acceptance, 'installed_parity', lambda *a, **k: profile['installed_weaver_parity'])
    monkeypatch.setattr(gate, 'DelphesParticleTransformer', object)
    monkeypatch.setattr(gate, 'train_kernel', train)
    monkeypatch.setattr(gate, 'predict', lambda model, c, **k: np.zeros((len(c), 11), np.float32))
    monkeypatch.setattr(gate.torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(gate.torch.cuda, 'max_memory_allocated', lambda: 2**30)
    monkeypatch.setattr(gate.torch.cuda, 'empty_cache', lambda: None)
    ticks = itertools.count(100.)
    monkeypatch.setattr(gate.time, 'monotonic', lambda: next(ticks))
    gate.run_gate_task(spec, 'preflight')
    assert calls == [('train', 'U000'), ('validation', 'U000'), ('train', 'D050'), ('validation', 'D050')]
    campaign = full.create_campaign(gate_root=spec['gate_root'], campaign_root=tmp_path/'science')
    assert production.validate_campaign(campaign, check_source=True)
    assert campaign['schema_version'] == 8 and campaign['foundation']['role_counts'] == full.COUNTS
    for subject, mode, count in ((spec, 'gate', 3), (campaign, 'science', 16)):
        plan = (submission.gate_plan if mode == 'gate' else submission.science_plan)(subject)
        assert len(plan['commands']) == count
        for row in plan['commands']:
            argv = row['command']
            assert '--partition=gpu' in argv and '--account=default' in argv
            if row['task_id'] == 'preflight' or row['task_id'].startswith(('train_', 'reduce_')):
                assert '--gres=gpu:l40s:1' in argv and '--mem=180000M' in argv
            assert not any('DENSE' in a for a in argv)
        submission.submit(subject=subject, mode=mode, execute=False, authorization_phrase=None)
        with pytest.raises(PermissionError):
            submission.submit(subject=subject, mode=mode, execute=True, authorization_phrase='wrong')
    for key, value in (('memory_mb', 90000), ('schema_version', 8), ('measured_role_counts', small.COUNTS)):
        with pytest.raises(ValueError):
            gate.validate_profile(with_content_hash(dict(profile, **{key: value})), foundation=f, spec=spec)
    assert not (tmp_path/'science/submission_ledger.json').exists()
    assert (tmp_path/'science/dry_run_submission_ledger.json').exists()
    # Live dispatch exercised with a stub scheduler, no real submission.
    admitted = []
    monkeypatch.setattr(small, 'check_submission_site', lambda plan: admitted.append(plan['mode']))
    monkeypatch.setattr(small, 'submit_claimed', lambda subject, plan, root: {'stub': True})
    assert submission.submit(subject=campaign, mode='science', execute=True,
                             authorization_phrase=full.SCIENCE_AUTHORIZATION) == {'stub': True}
    assert admitted == ['science']


def test_full_wrapper_is_explicit_and_does_not_cancel():
    text = (Path(__file__).resolve().parents[1]/'scripts/queue_jetclass2_context_full_ladder.sh').read_text()
    assert 'JC2_SITE=oscar_l40s' in text and 'CONTEXT 1M OSCAR' in text
    assert '--plan-hash "$6"' in text and 'Dry review only; no jobs submitted' in text
    assert '"${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"' in text
    assert 'scancel' not in text

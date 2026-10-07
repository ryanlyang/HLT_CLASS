"""Synthetic files and mocked Slurm only; real process replay has its own test."""
import copy
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_context_ladder import (parent, parent_v2, final_pilot, study, relocated, dataset,
                                gated, profile_for, MEASURE_PREFLIGHT)
from hlt_classification.context_v2 import dataset as d, kernel as k, workflow as w
from hlt_classification.context_v2.contracts import artifact, write, load_json, GIB
from hlt_classification.cms_proxy_ladder import context as old, context_v2 as v2, gate, data, cache, release, submission
from hlt_classification.cms_proxy_ladder.contracts import artifact as shared
from hlt_classification.data.cache_contracts import with_content_hash
from hlt_classification.literature_context import transform as v1
from hlt_classification.literature_context_production import output


@pytest.fixture
def spec(gated, tmp_path, monkeypatch):
    source = d.source(dict(source_request=gated['request']))
    monkeypatch.setattr(d, 'COUNTS', source.study['counts'])
    monkeypatch.setattr(w, 'COUNTS', old.COUNTS)
    monkeypatch.setattr(w, 'source_lock', lambda project, commit: artifact('SOURCE', commit=commit, files={'synthetic': 'f'*64}))
    # Old synthetic donor has no source-file inventory; production cannot bypass it.
    monkeypatch.setattr(d, 'frozen_inverse', lambda parent: None)
    monkeypatch.setattr(d.shutil, 'disk_usage', lambda path: SimpleNamespace(free=1000*GIB))
    return d.create(original_gate=Path(gated['gate_root'])/'gate_spec.json', root=tmp_path/'workflow',
        dataset_root=tmp_path/'v2', project_dir=gated['project_dir'], commit='f'*40,
        available_quota_gib=40, persistent=True)


@pytest.fixture
def generated(spec, monkeypatch):
    monkeypatch.setattr(d, 'ProcessPoolExecutor', lambda max_workers, mp_context: ThreadPoolExecutor(max_workers=max_workers))
    for i in range(len(spec['shards'])):
        d.generate(spec, i, workers=2)
    d.finalize(spec)
    return spec


def test_recipe_small_change_and_reversible_kernel(spec):
    assert v1.AMPLITUDE == .65 and k.recipe()['shear_amplitude'] == 1.
    identity, original = next(d.source(spec).iter_proxy('train'))
    _, result, stats = k.reencode((identity, original))
    base = v1.transform(original, inverse=True)
    assert stats['inverse_max_scaled_error'] <= 1e-10
    for field in ('p4', 'charge', 'category', 'valid'):
        assert getattr(result, field).tobytes() == getattr(original, field).tobytes()
    np.testing.assert_allclose(k.transform(result, inverse=True).tracking, base.tracking, atol=1e-10)
    assert np.any(result.tracking != original.tracking)
    perm = np.arange(len(result))[::-1]
    from hlt_classification.cms2jc2_response.bridge import Particles
    shuffled = Particles(*(getattr(base, name)[perm] for name in ('p4', 'charge', 'category', 'tracking', 'valid')),
                         tuple(base.keys[i] for i in perm))
    np.testing.assert_allclose(k.transform(shuffled).tracking, result.tracking[perm], atol=1e-12)


def test_real_spawn_byte_replay(spec, tmp_path):
    parent = d.source(spec)
    row = parent._receipt(parent._shards('train', None)[0])
    block = row['blocks'][0]
    args = (str(parent.proxy_root/block['relative']), block, block['relative'])
    serial = d._block_task(args)
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        parallel = pool.submit(d._block_task, args).result()
    assert serial[1] == parallel[1]
    assert serial[2]['tracking'].tobytes() == parallel[2]['tracking'].tobytes()


def test_generation_full_counts_idempotence_corruption_and_sealing(generated):
    spec = generated
    consumer = d.Dataset(spec['dataset_root'])
    assert consumer.manifest['counts'] == spec['counts']
    assert consumer.manifest['final_test_materialized'] and not consumer.manifest['final_test_evaluated']
    receipt = d.generate(spec, 0)
    assert receipt['jets'] == spec['shards'][0]['jets']
    assert receipt['exact_process_replay'] and receipt['physical_schema_verified']
    assert len(list(consumer.iter_proxy('train'))) == spec['counts']['train']
    with pytest.raises(PermissionError):
        list(consumer.iter_proxy('final_test'))
    test = next(s for s in spec['shards'] if s['role'] == 'final_test')
    path = Path(spec['dataset_root'])/'shards'/f"{test['shard_id']}.json"
    path.write_bytes(b'opaque test corruption')
    assert len(list(d.Dataset(spec['dataset_root']).iter_proxy('train'))) == spec['counts']['train']
    with pytest.raises(ValueError):
        d.finalize(spec)
    block = Path(spec['dataset_root'])/receipt['blocks'][0]['relative']
    block.write_bytes(b'bad block')
    with pytest.raises(ValueError):
        d.generate(spec, 0)


def test_release_exact_original_index_and_context_caches(generated, tmp_path):
    request = v2.release_request(generated['dataset_root'])
    root = tmp_path/'newrelease'
    r = release.build_release(request, output_root=root)
    original = load_json(generated['original_release']['path'])
    old_root = Path(generated['original_release']['path']).parent
    assert (root/r['bank']['path']).read_bytes() == (old_root/original['bank']['path']).read_bytes()
    assert r['identity_sha256'] == generated['selection_identities']
    assert r['schema_version'] == 5
    foundation = data.build_foundation(r, release_root=root, output_root=tmp_path/'foundation', capacity=512, workers=1)
    plan = v2.scientific_plan(foundation, foundation_root=tmp_path/'foundation')
    assert plan['fresh_fit_count'] == 9 and plan['probability_publication_count'] == 5
    assert list(plan['branches']) == ['DIRECT', 'COARSE']
    assert plan['schema_version'] == 7
    from hlt_classification.jetclass2_delphes.campaign import recipe as training_recipe
    assert plan['recipe'] == training_recipe()
    d000_caches = {}
    for role in w.COUNTS:
        rows = list(data.iter_paired(r, release_root=root, role=role))
        for coordinate in ('D000', 'OFFLINE', 'U050', 'U100', 'D066', 'D033'):
            c = cache.prepare_cache(foundation, foundation_root=tmp_path/'foundation', role=role,
                                    coordinate=coordinate, workers=1, max_ram_bytes=GIB)
            assert [bytes(i).hex() for i in c.identities] == [row.identity for row in rows]
            if coordinate == 'D000':
                d000_caches[role] = c
            if coordinate in ('D000', 'OFFLINE'):
                expected = np.concatenate([v1.build_inputs(row.proxy if coordinate == 'D000' else row.offline).features for row in rows])
                np.testing.assert_array_equal(np.concatenate([b.features for b in c.blocks]), expected)
    with pytest.raises(PermissionError):
        list(data.iter_paired(r, release_root=root, role='final_test'))
    bad = with_content_hash(dict(r, identity_sha256=dict(train='0'*64, validation='0'*64)))
    with pytest.raises(ValueError):
        release.validate_release(bad, root=root)
    # Real CPU CE/KD optimizer and validation through the unchanged kernel.
    # This tiny test double is NOT installed-Weaver architecture/parity evidence.
    import torch
    from hlt_classification.jetclass2_delphes.runner import train_kernel
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(17, 11)
        def forward(self, features, vectors, mask):
            return self.head((features*mask).sum(-1)/mask.sum(-1).clamp_min(1))
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for name in ('M0HLT', 'CTXV2_DIRECT_D000_from_U000'):
            node = next(n for n in plan['nodes'] if n['node_id'] == name)
            kd = {} if name == 'M0HLT' else dict(
                teacher_probabilities=np.full((w.COUNTS['train'], 11), 1/11, dtype=np.float32),
                teacher_identities=d000_caches['train'].identities)
            report, weights = train_kernel(Tiny(), d000_caches['train'],
                d000_caches['validation'], node=node, device='cpu', acceptance_passes=1, **kd)
            assert report['acceptance_only'] and not report['scientific_fit'] and weights
            assert np.isfinite(report['validation']['accuracy'])
    finally:
        torch.set_num_threads(threads)


def test_exact_workflow_dry_authority_and_failure_claim(spec, monkeypatch):
    monkeypatch.setattr(w, 'site_checks', lambda plan: None)
    value = w.submit(spec)
    assert len(value['commands']) == 5
    assert [r['dependencies'] for r in value['commands']] == [[], ['generate'], ['finalize'], ['prepare'], ['preflight']]
    assert not any('pilot' in r['task_id'] for r in value['commands'])
    assert value['followup_policy']['science_jobs'] == 16
    for row in value['commands']:
        assert '--export=NONE' in row['command']
        assert any('--dependency=afterok:' in a for a in row['command']) == bool(row['dependencies'])
        assert any('--gres=' in a for a in row['command']) == (row['task_id'] == 'preflight')
    with pytest.raises(ValueError, match='reviewed'):
        w.submit(spec, execute=True, reviewed_hash='0'*64, phrase=w.AUTHORIZATION)
    root = Path(spec['root'])
    assert not (root/'authorization.json').exists()
    def fail(*args, **kwargs):
        raise RuntimeError('simulated interrupted sbatch')
    monkeypatch.setattr(w.subprocess, 'run', fail)
    with pytest.raises(RuntimeError):
        w.submit(spec, execute=True, reviewed_hash=value['content_hash'], phrase=w.AUTHORIZATION)
    assert (root/'live_submission_claim.json').is_file()
    with pytest.raises(PermissionError, match='interrupted'):
        w.submit(spec, execute=True, reviewed_hash=value['content_hash'], phrase=w.AUTHORIZATION)


def test_dataset_rejects_recipe_scope_population_and_storage_drift(spec):
    for key, value in [('recipe', dict(k.recipe(), shear_amplitude=2.)), ('materialize_final_test', False),
                       ('automatic_science', False), ('selection_identities', {'train': '0'*64}),
                       ('allowances', {}), ('dataset_root', spec['source_request']['study_root'])]:
        with pytest.raises((ValueError, KeyError)):
            d.validate_spec(with_content_hash(dict(spec, **{key: value})))


def test_full_automatic_followup_with_mock_runtime_and_scheduler(generated, monkeypatch):
    spec = generated
    real = gate.build_foundation
    monkeypatch.setattr(gate, 'build_foundation', lambda *args, **kwargs: real(*args, **dict(kwargs, workers=1)))
    def measured(g, f, **kwargs):
        # Existing unchanged measurement schema, with the explicit V2 plan/versions.
        with monkeypatch.context() as m:
            m.setattr(old, 'scientific_plan', v2.scientific_plan)
            p = profile_for(g, f)
        return with_content_hash(dict(p, contract='JETCLASS2_CMS_PROXY_LADDER_RUNTIME_PROFILE/v10', schema_version=10))
    monkeypatch.setattr(gate, '_measure_preflight', measured)
    monkeypatch.setattr(w, '_bound_job', lambda spec, name: -1)
    w.run(spec, 'prepare')
    w.run(spec, 'preflight')
    monkeypatch.setattr(old, 'check_submission_site', lambda plan: None)
    accepted = []
    def sbatch(args, **kwargs):
        assert args[0] == 'sbatch'
        accepted.append(args)
        return SimpleNamespace(stdout=str(8000+len(accepted)), returncode=0)
    monkeypatch.setattr(w.subprocess, 'run', sbatch)
    result = w.run(spec, 'launch_science')
    assert len(accepted) == len(result['jobs']) == 16
    assert not any('DENSE' in a for row in accepted for a in row)
    assert len([row for row in accepted if any('--gres=' in a for a in row)]) == 14
    campaign = load_json(Path(spec['root'])/'science/campaign_spec.json')
    v2.validate_campaign(campaign)
    assert campaign['schema_version'] == 7 and not campaign['final_test_accessed']
    # Never silently retry a launch with an already-created campaign.
    with pytest.raises(FileExistsError):
        w.run(spec, 'launch_science')


def test_bound_array_job_and_authority_fail_closed(tmp_path, monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as journal
    spec = dict(content_hash='a'*64, root=str(tmp_path), project_dir='/example/pinned', shards=[{}, {}])
    value = w.plan(spec)
    write(tmp_path/'command_plan.json', value)
    write(tmp_path/'authorization.json', w.authorization(spec, value))
    prefix = '/oscar/scratch/rlyang/hlt_classification/environments/envs/atlas_kd_oscar'
    for key, val in dict(SLURM_ARRAY_JOB_ID='123', SLURM_ARRAY_TASK_ID='1',
        SLURM_JOB_ID='456', SLURM_CPUS_PER_TASK='6', SLURM_MEM_PER_NODE='32000', SLURM_CLUSTER_NAME='slurmctld',
        PYTHONNOUSERSITE='1', JC2_SITE='oscar_l40s', CONDA_PREFIX=prefix, LD_LIBRARY_PATH=prefix+'/lib', USER='example').items():
        monkeypatch.setenv(key, val)
    monkeypatch.setattr(w.sys, 'prefix', prefix)
    monkeypatch.setattr(journal, 'load_exact_dag_journal', lambda *a, **k: ([], {'generate': '123'}))
    fields = dict(JobId='456', ArrayJobId='123', ArrayTaskId='1', JobName='jc2ctx2_generate',
        Comment=f"ctx2:{spec['content_hash']}:generate", Account='default', Partition='batch',
        UserId='example(123)', WorkDir='/example/pinned', JobState='RUNNING', NumNodes='1', NumTasks='1',
        ReqTRES='cpu=6,mem=32000M', AllocTRES='cpu=6,mem=32000M', **{'CPUs/Task': '6'})
    monkeypatch.setattr(w.subprocess, 'run', lambda *a, **k: SimpleNamespace(
        stdout=' '.join(f'{k}={v}' for k, v in fields.items()), returncode=0))
    assert w._bound_job(spec, 'generate') == 1
    for key, bad in [('ArrayTaskId', '0'), ('Account', 'other'), ('Partition', 'gpu'),
                     ('JobState', 'PENDING'), ('WorkDir', '/elsewhere'), ('ReqTRES', 'cpu=6,gres/gpu=1')]:
        original = fields[key]
        fields[key] = bad
        with pytest.raises(ValueError):
            w._bound_job(spec, 'generate')
        fields[key] = original
    monkeypatch.setenv('SLURM_ARRAY_TASK_ID', '2')
    with pytest.raises(ValueError, match='index'):
        w._bound_job(spec, 'generate')


def test_empty_partial_tails_and_no_ancestry_inputs():
    from test_literature_proxy import particles
    from hlt_classification.cms2jc2_response.bridge import Particles
    for p in (particles(()), particles((1, 2)), particles((4,))):
        _, result, evidence = k.reencode(('0'*64, v1.transform(p)))
        assert evidence['inverse_max_scaled_error'] < 1e-10
        np.testing.assert_allclose(k.transform(result, inverse=True).tracking, p.tracking, atol=1e-10)
    p = particles((4,)*32)
    rng = np.random.default_rng(61)
    tr = p.tracking.copy()
    tr[:, :2] = rng.normal(size=(32, 2))*np.geomspace(1e-6, 1e5, 32)[:, None]
    tr[:, 2:] = np.exp(rng.uniform(-10, 3, (32, 2)))
    valid = p.valid.copy()
    tr[0, 2], valid[0, 2] = 0., False
    p = Particles(p.p4, p.charge, p.category, tr, valid, p.keys)
    _, result, evidence = k.reencode(('0'*64, v1.transform(p)))
    np.testing.assert_array_equal(result.tracking[0], p.tracking[0])
    assert np.isfinite(v1.build_inputs(result).features).all()
    assert evidence['frontend_max_scaled_error'] < 1e-4
    other = Particles(p.p4, p.charge, p.category, tr, valid, tuple('new'+str(i) for i in range(len(p))))
    np.testing.assert_array_equal(k.transform(p).tracking, k.transform(other).tracking)
    with pytest.raises(ValueError):
        k.transform(p, inverse=1)


def test_historical_inverse_code_hash_is_mandatory():
    from hlt_classification.context_v2.contracts import sha256_file
    path = 'src/hlt_classification/literature_context/transform.py'
    fake = SimpleNamespace(bundle={'source': {'file_sha256': {path: sha256_file(Path(v1.__file__))}}})
    d.frozen_inverse(fake)
    fake.bundle['source']['file_sha256'][path] = '0'*64
    with pytest.raises(ValueError, match='inverse implementation'):
        d.frozen_inverse(fake)

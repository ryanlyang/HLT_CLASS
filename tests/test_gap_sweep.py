"""Bounded screen semantics; real synthetic readers/processes/CPU CE, never real submissions."""
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from test_literature_proxy import particles
from test_correlated_topology import parent, final_pilot, study, dataset, released, topology_request, gated
from hlt_classification.gap_sweep import kernel as k, campaign as c, cache, worker as w
from hlt_classification.gap_sweep.contracts import artifact, validate, write_json, load_json, file_ref
from hlt_classification.data.cache_contracts import with_content_hash


def _tiny_spawn_initializer():
    """Mirror fixture-only constants in spawned readers, never relax production code."""
    from hlt_classification.cms_proxy_ladder.cache import _limit_worker_threads
    from hlt_classification.cms_proxy_ladder import correlated
    from hlt_classification.correlated_topology import dataset as topology
    from hlt_classification.correlated_tracking_production import population, evidence
    from test_correlated_topology import rates
    _limit_worker_threads()
    correlated.COUNTS = topology.COUNTS = dict(train=22, validation=11)
    population.COUNTS = population.registered.COUNTS = dict(train=44, validation=11, final_test=11)
    population.registered.PROFILE = 'TRAIN_44'
    evidence.APPROVED_PILOT_COMMIT = 'c'*40
    topology.noise_calibration = lambda *a, **kw: rates()


def test_frozen_formula_context_and_shared_draws(monkeypatch):
    p = particles((0, 1, 2, 3, 4), [2., 20., 20., 20., 20.])
    calls = []
    def draw(identity, domain, n):
        calls.append((identity, domain, n))
        return np.ones(n) if domain == 'drop' else np.zeros(n)
    monkeypatch.setattr(k, '_draw', draw)
    out, counts = k.generate(p, 'one', 'S1')
    np.testing.assert_array_equal(out.p4, p.p4)
    assert out.category[0] == 1 and out.charge[0] == 0
    assert not out.valid.any() and not out.tracking.any()
    assert counts['charged_to_neutral'] == 1
    assert counts['input_particles'] == counts['output_particles']
    assert [d for _, d, _ in calls] == ['drop', 'neutral', 'tracking']
    assert k.recipe()['candidates'] == dict(S1=.5, S2=1., S3=1.5)
    with pytest.raises(ValueError):
        k.generate(p, 'one', 'S4')


def test_probability_equations_use_original_context(monkeypatch):
    p = particles((0, 0), [2., 2.])
    q = .5*(np.maximum(0., 1.-np.hypot(p.eta[0]-p.eta[1], p.phi[0]-p.phi[1])/.08)
             +np.tanh(np.log1p(np.abs(p.tracking[:, :2])/np.array([.2, .5])).sum(1)/4.))
    threshold = .10*(.5+.5*q)*(1-p.pt/10)
    monkeypatch.setattr(k, '_draw', lambda identity, domain, n: threshold if domain == 'drop' else np.ones(n))
    assert len(k.generate(p, 'edge', 'S2')[0]) == 2  # strict probability comparison
    assert len(k.generate(p, 'edge', 'S3')[0]) == 0


def test_nested_loss_permutation_and_empty_inputs():
    p = particles((0, 1, 2, 3, 4)*12, [1.]*60)
    original = {key: i for i, key in enumerate(p.keys)}
    totals = {n: [] for n in k.CANDIDATES}
    for identity in ('a', 'b', 'c', 'd'):
        previous_keys, previous_valid, previous_neutral = set(p.keys), set(p.keys), set()
        for name in k.CANDIDATES:
            out, count = k.generate(p, identity, name)
            perm, _ = k.generate(p.take(list(reversed(range(len(p))))), identity, name)
            assert out.keys == perm.keys
            for field in ('p4', 'category', 'charge', 'tracking', 'valid'):
                np.testing.assert_array_equal(getattr(out, field), getattr(perm, field))
            assert set(out.keys) <= previous_keys
            valid = {key for key, v in zip(out.keys, out.valid) if v.any()}
            assert valid <= previous_valid
            neutral = {key for key, cat in zip(out.keys, out.category) if cat == 1 and p.category[original[key]] == 0}
            assert previous_neutral & set(out.keys) <= neutral
            previous_keys, previous_valid, previous_neutral = set(out.keys), valid, neutral
            np.testing.assert_array_equal(out.p4, p.p4[[original[key] for key in out.keys]])
            assert count['input_particles']-count['dropped_particles'] == count['output_particles']
            assert np.all(out.tracking[~out.valid] == 0)
            totals[name].append(count['output_particles'])
    assert sum(totals['S3']) < sum(totals['S1'])
    empty, counts = k.generate(p.take([]), 'empty', 'S3')
    assert len(empty) == 0 and counts['empty_output'] == 1


@pytest.mark.parametrize('values,expected', [
    ([.86, .855, .85], 'S2'),  # 1pp exactly is excluded
    ([.855, .85, .8], 'S1'),
    ([.86, .85, .855], 'S2'),  # no assumed monotonic accuracy
    ([.8, .7, .6], None), ([.88, .875, .87], None)])
def test_mildest_in_band_no_unbounded_retuning(values, expected):
    chosen = c.choose(.87, {n: dict(accuracy=v) for n, v in zip(k.CANDIDATES, values)})
    assert chosen['selected'] == expected
    assert chosen['status'] == ('development_selection_only' if expected else 'no_candidate_in_band')
    with pytest.raises(ValueError):
        c.choose(.87, dict(S1=dict(accuracy=.85)))
    with pytest.raises(ValueError):
        c.choose(.87, {n: dict(accuracy=float('nan')) for n in k.CANDIDATES})


def test_test_seal_and_unknown_inputs_fail_before_read():
    with pytest.raises(PermissionError):
        cache.prepare({}, 'S1', 'final_test', workers=1, input_identity='a'*64, max_ram_bytes=1)
    bad = with_content_hash(dict(artifact('SPEC'), final_test_accessed=True))
    with pytest.raises(PermissionError):
        validate(bad, 'SPEC')


def test_full_synthetic_workflow_cache_parity_dag_and_failure_guards(gated, tmp_path, monkeypatch):
    import torch
    from test_jetclass2_delphes_training import TinyModel
    from hlt_classification.cms_proxy_ladder import correlated_topology as adapter, production as p
    from hlt_classification.jetclass2_delphes.runner import train_kernel
    from hlt_classification.jetclass2_delphes.contracts import artifact as training_artifact
    torch.set_num_threads(1)
    parent_campaign = adapter.create_campaign(gate_root=gated['gate_root'], campaign_root=tmp_path/'parent-science')
    parent_file = tmp_path/'parent-science/campaign_spec.json'
    monkeypatch.setattr(p, '_execution_gate', lambda *a: None)
    monkeypatch.setattr(p, 'DelphesParticleTransformer', TinyModel)
    original_cache = p.prepare_cache
    monkeypatch.setattr(p, 'prepare_cache', lambda *a, **kw: original_cache(*a, **dict(kw, workers=1)))
    for n in ('OFFLINE', 'M0HLT'):
        p.run_task(parent_campaign, 'train_'+n, attempt='test', device='cpu')
    monkeypatch.setattr(c, 'BASE_COMMIT', 'e'*40)
    monkeypatch.setattr(c, 'COUNTS', parent_campaign['foundation']['role_counts'])
    monkeypatch.setattr(c, 'source_lock', lambda par, project, commit: artifact('SOURCE', commit=commit,
        files={}, unchanged_parent_source=par['source']['content_hash']))
    spec = c.create(parent_spec=parent_file, project_dir=gated['project_dir'], source_commit='f'*40, root=tmp_path/'screen')
    assert c.validate_spec(spec) == parent_campaign
    # Only completed controls are inspected, never unavailable KD results.
    assert set(spec['controls']) == {'OFFLINE', 'M0HLT'}
    with pytest.raises(FileExistsError):
        c.create(parent_spec=parent_file, project_dir=gated['project_dir'], source_commit='f'*40, root=tmp_path/'screen')
    for field, value in (('workers', 5), ('counts', dict(train=1, validation=1)), ('selection', {})):
        with pytest.raises(ValueError):
            c.validate_spec(with_content_hash(dict(spec, **{field: value})))
    with pytest.raises(FileNotFoundError):
        c.plan(spec, 'science')
    gate_plan = c.submit(spec, mode='gate')
    assert len(gate_plan['commands']) == 1
    with pytest.raises(PermissionError):
        c.submit(spec, mode='gate', execute=True, plan_hash=gate_plan['content_hash'], authorization='wrong')
    assert not (tmp_path/'screen/submission_gate/submission_ledger.json').exists()
    identity = c.candidate_identity(spec, parent_campaign, 'S1')
    monkeypatch.setattr(cache, '_limit_worker_threads', _tiny_spawn_initializer)
    train1, diag1 = cache.prepare(parent_campaign, 'S1', 'train', workers=1, input_identity=identity, max_ram_bytes=2**30)
    train2, diag2 = cache.prepare(parent_campaign, 'S1', 'train', workers=2, input_identity=identity, max_ram_bytes=2**30)
    for field in ('features', 'vectors', 'mask', 'identities', 'labels'):
        np.testing.assert_array_equal(train1.batch(np.arange(len(train1)))[field], train2.batch(np.arange(len(train2)))[field])
    assert diag1['mechanisms'] == diag2['mechanisms'] and diag1['statistics'] == diag2['statistics']
    with pytest.raises(MemoryError):
        cache.prepare(parent_campaign, 'S1', 'train', workers=2, input_identity=identity, max_ram_bytes=1)
    replay = cache.replay(parent_campaign, workers=2)
    assert len(replay['hashes']) == 3*min(32, len(train1))
    # Exercise actual coordinator and complete CE kernel; only hardware/model are substituted.
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setattr(w, '_execution_gate', lambda *a: None)
    monkeypatch.setattr(w, 'DelphesParticleTransformer', TinyModel)
    monkeypatch.setattr(w, 'installed_environment', lambda: parent_campaign['runtime_profile']['installed_environment'])
    monkeypatch.setattr(w, 'gpu_identity', lambda: parent_campaign['runtime_profile']['gpu'])
    monkeypatch.setattr(w, 'installed_parity', lambda *a, **kw: training_artifact('WEAVER_PARITY',
        model=parent_campaign['model'], device='cuda', passed=True, forward_and_feature_and_parameter_gradients=True,
        final_test_accessed=False))
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 1024)
    monkeypatch.setattr(w, 'train_kernel', lambda *a, **kw: train_kernel(*a, **dict(kw, device='cpu')))
    # Avoid repeating process startup six times; real two-worker cache/replay tested above.
    old_prepare = cache.prepare
    monkeypatch.setattr(cache, 'prepare', lambda *a, **kw: old_prepare(*a, **dict(kw, workers=1)))
    measured = w.run(spec, 'preflight')
    assert w.run(spec, 'preflight') == measured
    assert measured['acceptance']['acceptance_only']
    for field, value in (('train_minutes', 1441), ('counts', {}), ('exact_process_replay', False),
                         ('cache_bytes', 10**15), ('gpu_peak_bytes', 10**15), ('installed_environment', {})):
        with pytest.raises(ValueError):
            c.validate_preflight(with_content_hash(dict(measured, **{field: value})), spec, parent_campaign)
    scientific = c.submit(spec, mode='science')
    assert len(scientific['commands']) == 4
    for row in scientific['commands'][:3]:
        assert row['dependencies'] == [] and '--gres=gpu:a100:1' in row['command']
        assert '--partition=debug' in row['command'] and '--export=NONE' in row['command']
    assert scientific['commands'][-1]['dependencies'] == ['fit_S1', 'fit_S2', 'fit_S3']
    assert not any(a.startswith('--gres') for a in scientific['commands'][-1]['command'])
    for name in k.CANDIDATES:
        w.run(spec, 'fit_'+name)
        receipt, report = c.completed_fit(spec, parent_campaign, name)
        assert report['scientific_fit'] and not report['acceptance_only'] and report['passes'] >= 60
        assert w.run(spec, 'fit_'+name) == receipt
        assert report['node']['teacher'] is None
    summary = w.run(spec, 'summary')
    assert w.run(spec, 'summary') == summary
    assert set(summary['metrics']) == set(k.CANDIDATES) and not summary['final_test_accessed']
    assert summary['selection_policy']['kd_results_used'] is False
    # Corruption fails closed, not silently interpreted as a pending fit.
    target = tmp_path/'screen/fit_S2/selected.pt'
    target.write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        c.completed_fit(spec, parent_campaign, 'S2')
    with pytest.raises(ValueError):
        w.run(spec, 'summary')


def test_source_transfer_refuses_changed_science_before_git(tmp_path, monkeypatch):
    from hlt_classification.data.cache_contracts import sha256_file
    original = tmp_path/'science.py'; original.write_text('old')
    parent = dict(source=dict(files={'science.py': sha256_file(original)}, content_hash='a'*64))
    original.write_text('changed')
    monkeypatch.setattr(c, '_source', lambda *a: None)
    monkeypatch.setattr(c.subprocess, 'run', lambda *a, **kw: pytest.fail('Reject drift before Git lookup'))
    with pytest.raises(ValueError, match='scientific source changed'):
        c.source_lock(parent, tmp_path, 'f'*40)


def test_claim_refuses_partial_rerun(tmp_path, monkeypatch):
    spec = dict(root=str(tmp_path), content_hash='a'*64)
    monkeypatch.setattr(c, 'validate_spec', lambda s: {})
    (tmp_path/'fit_S1.claim').write_text('a'*64)
    monkeypatch.setattr(w, 'fit', lambda *a: pytest.fail('No repeated fit'))
    with pytest.raises(FileExistsError):
        w.run(spec, 'fit_S1')
    with pytest.raises(ValueError):
        w.run(spec, 'fit_OTHER')


def test_real_journal_exact_authority_dependencies_and_idempotence(tmp_path, monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as dag
    spec = dict(root=str(tmp_path), content_hash='a'*64)
    tasks = [('fit_'+n, []) for n in k.CANDIDATES]+[('summary', ['fit_'+n for n in k.CANDIDATES])]
    commands = [dict(task_id=n, dependencies=deps, command=['sbatch', '--parsable', '--export=NONE',
        *( ['--dependency=afterok:'+':'.join('${JOB_'+d+'}' for d in deps)] if deps else []), '--wrap=true'])
        for n, deps in tasks]
    plan = artifact('PLAN', parents=dict(spec=spec['content_hash']), mode='science', commands=commands)
    monkeypatch.setattr(c, 'plan', lambda *a: plan)
    monkeypatch.setattr(c, 'check_submission_site', lambda p: None)
    calls = []
    def submit(argv, **kw):
        assert not any(n.startswith(('SLURM_', 'SBATCH_')) for n in kw['env'])
        assert all('${JOB_' not in a for a in argv)
        calls.append(argv)
        return SimpleNamespace(stdout=str(900+len(calls))+'\n')
    monkeypatch.setenv('SBATCH_PARTITION', 'wrong')
    monkeypatch.setattr(dag.subprocess, 'run', submit)
    c.submit(spec, mode='science')
    for options in (dict(plan_hash='b'*64, authorization=c.AUTHORIZATION),
                    dict(plan_hash=plan['content_hash'], authorization='wrong')):
        with pytest.raises(PermissionError):
            c.submit(spec, mode='science', execute=True, **options)
    assert not calls
    live = c.submit(spec, mode='science', execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 4 and live['jobs']['summary'] == '904'
    assert '--dependency=afterok:901:902:903' in calls[-1]
    assert c.submit(spec, mode='science', execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION) == live
    assert len(calls) == 4


def test_unresolved_submission_claim_never_requeues(tmp_path, monkeypatch):
    spec = dict(root=str(tmp_path), content_hash='a'*64)
    plan = artifact('PLAN', commands=[dict(task_id='preflight', dependencies=[], command=['sbatch', '--wrap=true'])])
    monkeypatch.setattr(c, 'plan', lambda *a: plan)
    monkeypatch.setattr(c, 'check_submission_site', lambda p: None)
    c.submit(spec, mode='gate')
    (tmp_path/'submission_gate/live_submission_claim.json').write_text('{}')
    with pytest.raises(PermissionError, match='interrupted'):
        c.submit(spec, mode='gate', execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)


def test_short_gate_request_preserves_full_science_recipe(tmp_path, monkeypatch):
    from hlt_classification.jetclass2_delphes.execution import execution_site
    monkeypatch.setattr(c, 'validate_spec', lambda spec: None)
    spec = dict(root=str(tmp_path), project_dir='/exact/project', content_hash='a'*64,
                cpus=6, memory_mb=90000, execution_site=execution_site('sporc_a100_debug'))
    row, = c.plan(spec, 'gate')['commands']
    assert '--time=02:00:00' in row['command'] and '--export=NONE' in row['command']
    assert row['dependencies'] == [] and '--gres=gpu:a100:1' in row['command']
    assert c.training_recipe()['minimum_passes'] == 60
    assert c.training_recipe()['maximum_passes'] == 100

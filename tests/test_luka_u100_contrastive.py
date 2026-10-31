"""Scientific sampler, gradient, native deployment and exact-DAG regression tests."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import math
import subprocess
import numpy as np
import pytest
import torch
from test_luka_attention import cache, parent_nodes
from test_luka_embedding_aux import tiny
from test_luka_u100_embedding import bank_fixture
from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_u100_embedding import contracts as uc, training as original
from hlt_classification.luka_embedding_aux.model import new_model as old_model
from hlt_classification.jetclass2_delphes.runner import _forward, predict
from hlt_classification.luka_u100_contrastive import (
    contracts as c, model, losses, sampling, targets, training, resources, campaign, submission, source)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def imported():
    return dict(node=uc.node(dict(nodes=parent_nodes())))


def spec(root, rows=22):
    return dict(campaign.registration(), root=str(root), project_dir='/project', content_hash='e'*64,
                imported=imported(), counts=dict(train=rows, validation=22))


def sampler_fixture(device='cpu', **kwargs):
    tr = cache(coordinate='D000', rows=6006)
    keys, _, _ = targets.normalized_keys(np.random.default_rng(91).normal(size=(len(tr), 128)).astype(np.float32))
    return sampling.Sampler(keys, tr.identities, tr.labels, device=device,
        pt=np.arange(len(tr), dtype=float)+1, count=np.arange(len(tr))%80+1, **kwargs)


def test_exact_plan_registration():
    assert len(c.ARMS) == 20 and [v['rank'] for v in c.ARMS.values()] == list(range(1, 21))
    assert sum(x['replicate'] == 1 for x in c.ARMS.values()) == 2
    assert [k for k, v in c.ARMS.items() if v['coefficient'] == 0] == ['u100_w5_s0', 'u100_w5_s1']
    assert all(sum(x.values()) == 1024 for x in c.sampler_recipe()['strategy_counts'].values())
    text = Path('docs/plans/LUKA_U100_OFFLINE_CONTRASTIVE_PLAN.md').read_text(encoding='utf-8')
    assert all(f'`{name}`' in text for name in c.ARMS)
    r = c.recipe(); assert (r['ce'], r['embedding'], r['epochs'], r['batch_size']) == (1., 5., 30, 128)
    assert r['u100'] == uc.recipe()
    assert c.ramp(.3, 1, 2338) == pytest.approx(.3/(3*2338), rel=1e-15)
    assert c.ramp(.3, 3*2338, 2338) == c.ramp(.3, 30*2338, 2338) == .3
    s0, s1 = c.seeds(imported(), 'u100_w5_s0'), c.seeds(imported(), 'u100_w5_s1')
    assert all(s1[k] == s0[k]+1 for k in s0)


@pytest.mark.parametrize('strategy', ['random', 'mix', 'cross', 'hard', 'kinmix'])
def test_full_size_sampler_invariants(strategy):
    sampler = sampler_fixture()
    ix, proof = sampler.sample([0, 17, 333], 2, strategy)
    assert ix.shape == (3, 1025) and np.array_equal(ix[:, 0], [0, 17, 333])
    for row, evidence in zip(ix, proof):
        assert len(np.unique(row)) == 1025
        assert {k: sum(v) for k, v in evidence['components'].items()} == c.sampler_recipe()['strategy_counts'][strategy]
        if 'hard_ranks' in evidence:
            assert min(evidence['hard_ranks']) >= 257 and max(evidence['hard_ranks']) <= 1024
            h = sum(evidence['components']['H'])
            assert np.all(sampler.labels[row[1:1+h]] != sampler.labels[row[0]])
            assert evidence['components']['S'][sampler.labels[row[0]]] == 256
            proposal = sampler.balanced(4096, sampler.excluded(row[0]), sampler.rng(2, row[0], 'H_pool'),
                                       [c for c in range(11) if c != sampler.labels[row[0]]])
            sims = (sampler.keys[proposal]*sampler.keys[row[0]]).sum(-1).numpy()
            ranked = proposal[np.lexsort((sampler.tie_rank[proposal], -sims))]
            np.testing.assert_array_equal(row[1:1+h], ranked[np.array(evidence['hard_ranks'])-1])
        if strategy == 'cross':
            assert np.all(sampler.labels[row[1:]] != sampler.labels[row[0]])
        for name in ('G', 'X', 'K'):
            if name in evidence['components']:
                counts = evidence['components'][name]
                if name in ('X', 'K'):
                    counts = [n for n in counts if n]
                assert max(counts)-min(counts) <= 1
    reverse, _ = sampler.sample([333, 0], 2, strategy)
    np.testing.assert_array_equal(reverse, ix[[2, 0]])
    next_epoch, _ = sampler.sample([0], 3, strategy)
    assert not np.array_equal(next_epoch[0], ix[0])


def test_exclusions_capacity_and_kinematic_only_hlt():
    groups = np.arange(6006); groups[[11, 22]] = 0
    aliases = np.arange(6006); aliases[[33, 44]] = 0
    sampler = sampler_fixture(event_groups=groups, alias_groups=aliases)
    for strategy in c.sampler_recipe()['strategy_counts']:
        a, _ = sampler.sample([0], 1, strategy)
        assert not {11, 22, 33, 44}.intersection(a[0])
    with pytest.raises(ValueError):
        sampler_fixture(role='validation')
    with pytest.raises(ValueError, match='capacity'):
        sampler_fixture(event_groups=np.zeros(6006, dtype=np.int64))
    bins, edges = sampling.fit_bins(np.ones(6006), np.ones(6006))
    assert np.all(bins == 0) and all(len(e) == 1 for e in edges)
    train = cache(coordinate='D000'); pt, count = targets.summaries(train)
    raw = train.batch(np.arange(len(train)))
    expected = raw['vectors'][:, :2].astype(np.float64).sum(-1)
    np.testing.assert_allclose(pt, np.hypot(expected[:, 0], expected[:, 1]))
    np.testing.assert_array_equal(count, raw['mask'][:, 0].sum(-1))
    with pytest.raises(PermissionError):
        targets.summaries(cache('validation', 'D000'))
    mix, _ = sampler.sample([0], 1, 'mix')
    hard, _ = sampler.sample([0], 1, 'hard')
    kin, _ = sampler.sample([0], 1, 'kinmix')
    np.testing.assert_array_equal(mix[0, 257:513], hard[0, 513:769])
    np.testing.assert_array_equal(mix[0, 257:513], kin[0, 257:513])


def test_analytical_loss_gradient_permutation_detachment():
    q = torch.zeros(3, 128); q[:, 0] = 1.; q.requires_grad_()
    keys = torch.zeros(3, 1025, 128); keys[:, 0, 0] = 1.; keys[:, 1:, 1] = 1.; keys.requires_grad_()
    loss = losses.match(q, keys)
    assert float(loss) == pytest.approx(math.log(1+1024*math.exp(-10)), abs=5e-6)
    grad = torch.autograd.grad(loss, q, retain_graph=True)[0]
    assert torch.all(grad[:, 1] > 0)
    permutation = torch.randperm(1025); pos = int(torch.nonzero(permutation == 0)[0])
    changed = losses.match(q, keys[:, permutation], torch.full((3,), pos, dtype=torch.long))
    torch.testing.assert_close(changed, loss, rtol=0, atol=5e-6)
    torch.testing.assert_close(torch.autograd.grad(changed, q)[0], grad)
    loss.backward(); assert keys.grad is None
    assert losses.match(q[:1], keys[:1]) == loss
    with pytest.raises(ValueError):
        losses.match(q, keys[:, :-1])
    with pytest.raises(ValueError):
        losses.match(q*float('nan'), keys)


def test_native_and_u100_initialization_rng_and_gradients(monkeypatch):
    tiny(monkeypatch)
    for rep in (0, 1):
        baseline = model.new_model(imported(), f'u100_w5_s{rep}')
        state = torch.get_rng_state().clone()
        matching = model.new_model(imported(), f'mix_l0p3_s{rep}')
        assert torch.equal(state, torch.get_rng_state())
        for name, p in baseline.state_dict().items():
            assert torch.equal(p, matching.state_dict()[name])
    matching.capture = True; raw = cache(coordinate='D000').batch(np.arange(11))
    logits = _forward(matching, raw, 'cpu', bf16=False)
    target = torch.randn(11, 128, requires_grad=True)
    keys = torch.nn.functional.normalize(torch.randn(11, 1025, 128), dim=-1).requires_grad_()
    total, ce, e, m = losses.objective(logits, torch.arange(11), matching.prediction, target, matching.query, keys, .3)
    torch.testing.assert_close(total, ce+5*e+.3*m)
    total.backward()
    assert target.grad is None and keys.grad is None
    for prefix in ('predictor.', 'matching.', 'mod.'):
        assert any(p.grad is not None and torch.count_nonzero(p.grad) for n, p in matching.named_parameters() if n.startswith(prefix))
    assert not any(n.startswith(('predictor.', 'matching.')) for n in matching.native_state())
    matching.eval(); matching.capture = False
    first = _forward(matching, raw, 'cpu', bf16=False)
    with torch.no_grad():
        for head in (matching.predictor, matching.matching):
            for p in head.parameters(): p.fill_(float('nan'))
    assert torch.equal(first, _forward(matching, raw, 'cpu', bf16=False))


def test_original_zero_kernel_parity_and_no_sampler(tmp_path, monkeypatch):
    tiny(monkeypatch); tr, bank, _ = bank_fixture(tmp_path); val = cache('validation', 'D000')
    history, selected, state = training.train(model.new_model(imported(), 'u100_w5_s0'), tr, val, bank, None,
        imported=imported(), arm='u100_w5_s0', device='cpu', technical=True)
    old, expected = original.train(old_model(imported()['node']), tr, val, bank=bank,
                                  node=imported()['node'], device='cpu', technical=True)
    assert history[0]['validation'] == old['validation'] and selected == old['selected_pass']
    assert all(torch.equal(state[k], expected[k]) for k in state)
    with pytest.raises(ValueError, match='OFFLINE'):
        training.train(model.new_model(imported(), 'u100_w5_s0'), tr, val, bank, object(),
                       imported=imported(), arm='u100_w5_s0', device='cpu', technical=True)


def test_key_bank_mean_floor_role_order_and_corruption(tmp_path):
    tr = cache(coordinate='D000'); raw = np.random.default_rng(7).normal(size=(22, 128)).astype(np.float32)
    raw[:, 0] = 0
    teacher = tmp_path/'teacher.pt'; teacher.write_bytes(b'authenticated-fixture-bytes')
    kwargs = dict(spec_hash='a'*64, imported_hash='b'*64)
    targets.publish(tmp_path/'keys', raw, tr, file_ref(teacher), **kwargs)
    bank = targets.KeyBank(tmp_path/'keys', tr, file_ref(teacher), **kwargs)
    np.testing.assert_allclose(bank.mean, raw.mean(0, dtype=np.float64), atol=0)
    assert bank.scale[0] == .001 and np.all(bank.keys[:, 0] == 0)
    expected = (raw.astype(np.float64)-bank.mean)/bank.scale
    expected /= np.maximum(np.linalg.norm(expected, axis=1, keepdims=True), 1e-12)
    np.testing.assert_allclose(bank.keys, expected, atol=3e-8)
    bad = deepcopy(tr); bad.identities = bad.identities[::-1]
    with pytest.raises(ValueError): targets.KeyBank(tmp_path/'keys', bad, file_ref(teacher), **kwargs)
    bad = deepcopy(tr); bad.role = 'validation'
    with pytest.raises(ValueError): targets.KeyBank(tmp_path/'keys', bad, file_ref(teacher), **kwargs)
    (tmp_path/'keys/keys.npz').write_bytes(b'broken')
    with pytest.raises(ValueError): targets.KeyBank(tmp_path/'keys', tr, file_ref(teacher), **kwargs)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_weaver_production_fit_full_candidates_native_reload(tmp_path, monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    tiny(monkeypatch); tr, bank, _ = bank_fixture(tmp_path); val = cache('validation', 'D000')
    full = sampler_fixture(device)
    sampler = resources.SubsetSampler(full, np.arange(len(tr)))
    s = spec(tmp_path); arm = 'mix_l0p3_s0'
    student = model.new_model(imported(), arm).to(device)
    before = torch.get_rng_state().clone(); buffers = {k: v.clone() for k, v in student.named_buffers()}
    probe = training.diagnostic(student, tr, bank, sampler, arm, 1, device)
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(v, buffers[k]) for k, v in student.named_buffers())
    assert all(p.grad is None for p in student.parameters())
    assert len(probe['weighted_encoder_gradient_norms']) == 3
    report = training.fit(tmp_path/'fit', s, arm, student, tr, val, bank, sampler, device=device,
                          technical=True, key_hash='f'*64)
    saved, arrays = training.validate_fit(tmp_path/'fit', s, arm, technical=True)
    assert saved == report and arrays['probabilities'].shape == (22, 11)
    from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
    native = DelphesParticleTransformer().to(device)
    native.load_state_dict(torch.load(tmp_path/'fit/selected.pt', weights_only=True, map_location=device), strict=True)
    np.testing.assert_allclose(predict(native, val, device=device, batch_size=128), arrays['probabilities'], atol=1e-6)
    assert all('matching' not in k and 'predictor' not in k for k in native.state_dict())
    bad = with_content_hash(dict(saved, selected_pass=2))
    with pytest.raises(ValueError): training.validate_report(bad, s, arm, technical=True)


def test_debug_two_lane_graph_and_resource_envelope(tmp_path):
    s = spec(tmp_path); envelope = submission.render(s, 'envelope')
    tasks = envelope['commands']; assert len(tasks) == 21
    assert [r['task_id'] for r in tasks[:-1]] == list(c.ARMS)
    assert tasks[0]['dependencies'] == tasks[1]['dependencies'] == []
    for i, row in enumerate(tasks[2:-1], 2):
        assert row['dependencies'] == [tasks[i-2]['task_id']]
        assert any(a.startswith('--dependency=afterany:') for a in row['command'])
        assert not any('afterok:' in a or 'kill-on-invalid' in a for a in row['command'])
    assert len(tasks[-1]['dependencies']) == 20
    assert all('--partition=debug' in row['command'] for row in tasks)
    acceptance = c.artifact('ACCEPTANCE', minutes={a: 180 for a in c.ARMS})
    measured = submission.render(s, 'science', acceptance=acceptance)
    assert submission.within_envelope(envelope, measured)
    measured['commands'][0]['command'].append('--gres=gpu:a100:2')
    assert not submission.within_envelope(envelope, measured)
    assert source.compatible_changes(['A\tsrc/hlt_classification/luka_u100_contrastive/losses.py'])
    assert not source.compatible_changes(['M\tsrc/hlt_classification/luka_u100_embedding/training.py'])


def test_representative_resource_estimate_not_worstcase_and_fail_closed(tmp_path):
    s = spec(tmp_path, 299234)
    profiles = {family: dict(miniature_complete=True, full_candidate_bank_rows=299234,
        representative_step_seconds=.1, worst_support_stress_seconds=100., probe_seconds=.2)
        for family in ('none', 'mix', 'random', 'cross', 'hard', 'kinmix')}
    d = c.artifact('RESOURCE_DIAGNOSTIC', parents=dict(spec=s['content_hash']), status='complete', profiles=profiles,
        memory=dict(peak_rss_bytes=10**9, gpu_peak_bytes=10**9), gpu=dict(total_memory_bytes=40*2**30),
        environment={}, validation_seconds=10., load_seconds=120., baseline_native_tensor_parity=True)
    a = resources.accept_diagnostic(d, s)
    assert all(180 < n < 240 for n in a['minutes'].values())
    d['profiles']['mix']['representative_step_seconds'] = 100.
    with pytest.raises(ValueError, match='walltime'):
        resources.accept_diagnostic(with_content_hash(d), s)


def test_exact_journal_duplicate_and_partial_submission_protection(tmp_path, monkeypatch):
    s = spec(tmp_path); plan = submission.render(s, 'setup')
    submission._submit(s, plan, execute=False)
    calls = []
    def run(command, **kwargs):
        calls.append(command); return SimpleNamespace(stdout='12345\n')
    monkeypatch.setattr(subprocess, 'run', run)
    submission._submit(s, plan, execute=True)
    assert submission.accepted_jobs(s, 'setup') == {'setup': '12345'}
    submission._submit(s, plan, execute=True); assert len(calls) == 1
    other = spec(tmp_path/'partial'); Path(other['root']).mkdir()
    plan = submission.render(other, 'setup'); submission._submit(other, plan, execute=False)
    def failed(command, **kwargs): raise RuntimeError('lost scheduler response')
    monkeypatch.setattr(subprocess, 'run', failed)
    with pytest.raises(RuntimeError): submission._submit(other, plan, execute=True)
    with pytest.raises(FileExistsError): submission._submit(other, plan, execute=True)


def test_script_environment_and_queue_automation():
    worker = Path('sbatch/run_luka_u100_contrastive.sh').read_text()
    queue = Path('scripts/queue_luka_u100_contrastive.sh').read_text()
    for text in (worker, queue):
        assert 'PYTHONNOUSERSITE=1' in text and 'PYTHONDONTWRITEBYTECODE=1' in text
        assert '${CONDA_PREFIX}/lib' in text and 'atlas_kd_sporc' in text
    assert '${PROJECT_DIR}/scripts/luka_u100_contrastive.py' in worker
    assert 'nohup python' in queue and ' dispatch ' in queue
    assert 'merge-base --is-ancestor' in queue


def test_full30_poor_finite_results_complete_and_probe_stream_independent(tmp_path, monkeypatch):
    tiny(monkeypatch); tr, bank, _ = bank_fixture(tmp_path); val = cache('validation', 'D000')
    full = sampler_fixture(); sampler = resources.SubsetSampler(full, np.arange(len(tr)))
    train_candidates, _ = full.sample([0], 1, 'mix')
    diag_candidates, _ = full.diagnostic_sampler().sample([0], 1, 'mix')
    assert not np.array_equal(train_candidates, diag_candidates)
    np.testing.assert_array_equal(full.sample([0], 1, 'mix')[0], train_candidates)
    evaluate = training.evaluate_probabilities
    def poor(labels, probabilities):
        result = evaluate(labels, probabilities); result['macro_ovr_auc'] = 0.
        return result
    monkeypatch.setattr(training, 'evaluate_probabilities', poor)
    s = spec(tmp_path); arm = 'mix_l0p3_s0'
    report = training.fit(tmp_path/'fit', s, arm, model.new_model(imported(), arm), tr, val,
                          bank, sampler, device='cpu', key_hash='f'*64)
    training.validate_fit(tmp_path/'fit', s, arm)
    assert len(report['validation_history']) == report['epochs'] == 30
    assert all(r['validation']['macro_ovr_auc'] == 0 for r in report['validation_history'])
    assert len(list((tmp_path/'fit/epochs').glob('*-probe.json'))) == 30


def test_diagnostic_published_before_admission_failure(tmp_path, monkeypatch):
    tiny(monkeypatch); tr, bank, _ = bank_fixture(tmp_path); val = cache('validation', 'D000')
    s = spec(tmp_path); (tmp_path/'setup').mkdir()
    def fail(*args, **kwargs): raise MemoryError('injected before fit admission')
    monkeypatch.setattr(resources, 'make_sampler', fail)
    with pytest.raises(MemoryError):
        resources.measure(s, tr, val, bank, object(), load_seconds=1., device='cpu')
    saved = load_json(tmp_path/'setup/resource_diagnostic.json')
    assert saved['status'] == 'failed' and 'injected' in saved['error']
    assert not (tmp_path/'setup/acceptance.json').exists()


def test_automatic_dispatch_success_and_setup_failure(tmp_path, monkeypatch):
    from hlt_classification.luka_u100_contrastive import worker
    monkeypatch.setattr(campaign, 'check', lambda s, **kwargs: s)
    monkeypatch.setattr(submission, 'check_site', lambda plan: None)
    monkeypatch.setattr(worker, 'completed_setup', lambda s: {})
    monkeypatch.setattr(resources, 'accepted', lambda s: c.artifact('ACCEPTANCE', minutes={a: 180 for a in c.ARMS}))
    calls = []
    def run(command, **kwargs):
        assert command[0] == 'sbatch'; calls.append(command)
        return SimpleNamespace(stdout=str(30000+len(calls)))
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(submission, 'accounting', lambda job: ['COMPLETED', '0:0'])
    s = spec(tmp_path); review = submission.review(s)
    submission.dispatch(s, plan_hash=review['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 22
    assert [next(v for v in command if v.startswith('--job-name=')) for command in calls][:3] == [
        '--job-name=lukaoc_setup', '--job-name=lukaoc_mix_l0p3_s0', '--job-name=lukaoc_u100_w5_s0']
    assert len(submission.accepted_jobs(s, 'science')) == 21
    with pytest.raises(FileExistsError):
        submission.dispatch(s, plan_hash=review['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 22
    failed_root = tmp_path/'failed'; failed_root.mkdir()
    failed = spec(failed_root); review = submission.review(failed)
    monkeypatch.setattr(submission, 'accounting', lambda job: ['FAILED', '1:0'])
    with pytest.raises(RuntimeError, match='no scientific'):
        submission.dispatch(failed, plan_hash=review['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 23 and not (failed_root/'submission_science').exists()


def test_saved_extended_metrics_and_censoring():
    from hlt_classification.luka_u100_contrastive.worker import extended_metrics
    labels = np.tile(np.arange(11), 3)
    p = np.eye(11, dtype=np.float32)[labels]*.9+.1/11
    result = extended_metrics(labels, p)
    assert result['macro_ovr_auc'] == 1. and len(result['per_class_auc']) == 11
    assert len(result['rejection_at_30pct']) == 10
    assert all(v['rejection'] is None and v['zero_fpr_censored'] for v in result['rejection_at_30pct'].values())

from copy import deepcopy
import math
import numpy as np
import pytest
import torch
from hlt_classification.luka_u100_pair_correction import contracts as c, geometry as g, model as m, losses, targets
from hlt_classification.luka_embedding_aux.model import new_model as original_model
from hlt_classification.luka_u100_embedding.contracts import node
from test_luka_attention import cache, parent_nodes
from test_luka_fullsim_stage2 import dataset, prepared
from test_luka_fullsim_alljets import all_prepared as donor_all_prepared


@pytest.fixture(scope='module')
def all_prepared(dataset, prepared, tmp_path_factory):
    # This tiny synthetic ROOT fixture writes only MiB. The donor's production
    # five-GiB storage reserve is not what this byte-replay test measures.
    from types import SimpleNamespace
    from hlt_classification.luka_fullsim import regular_cache
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(regular_cache.shutil, 'disk_usage', lambda _: SimpleNamespace(free=100*2**30))
        return donor_all_prepared.__wrapped__(dataset, prepared, tmp_path_factory)


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def imported():
    return dict(node=node(dict(nodes=parent_nodes())))


def stats():
    return dict(particle_scale=[.2, .1, .1, .1], pair_scale=[.4]*4, pair_mean=[0.]*4, pair_std=[2.]*4)


def compact(monkeypatch):
    from hlt_classification.jetclass2_delphes import model
    config = model.model_config()
    config.update(embed_dims=[128], pair_embed_dims=[8], num_layers=8, num_heads=8, num_cls_layers=2)
    config['block_params'] = dict(ffn_ratio=1, dropout=0, attn_dropout=0, activation_dropout=0)
    config['cls_block_params'] = dict(ffn_ratio=1, dropout=0, attn_dropout=0, activation_dropout=0)
    monkeypatch.setattr(model, 'model_config', lambda: deepcopy(config))


def inputs(rows=4, device='cpu'):
    raw = cache(coordinate='D000', rows=rows).batch(np.arange(rows))
    return {k: torch.from_numpy(raw[k]).to(device) for k in ('features', 'vectors', 'mask')}


def test_native_pair_values_and_collinear_backward():
    import weaver.nn.model.ParticleTransformer as w
    f = getattr(w, 'pairwise_lv_fts_pp', getattr(w, 'pairwise_lv_fts', None))
    raw = inputs()['vectors'].transpose(1, 2)
    p = raw[raw[..., 3] > 0].double().requires_grad_()
    q = p.roll(1, 0)
    torch.testing.assert_close(g.pair_features(p, q), f(p.T[None], q.T[None]).squeeze(0).T, atol=1e-12, rtol=1e-12)
    col = g.pair_features(p, p)
    torch.testing.assert_close(col, f(p.T[None], p.T[None]).squeeze(0).T, atol=1e-12, rtol=1e-12)
    col.sum().backward(); assert torch.isfinite(p.grad).all()
    assert g.wrap(torch.tensor(2*math.pi-.1)) == pytest.approx(-.1, abs=1e-6)


def test_physical_decode_and_roundoff():
    p = torch.tensor([[3., 4., 1., math.sqrt(26.)]], dtype=torch.float64)
    raw = torch.randn(1, 4, dtype=torch.float64, requires_grad=True)
    value = g.decode(p, raw, torch.tensor(stats()['particle_scale']), torch.ones(1))
    assert g.physical_audit(value) == 0
    g.coordinates(value).sum().backward(); assert torch.isfinite(raw.grad).all()
    identity = g.decode(p, torch.zeros_like(raw), torch.ones(4), torch.ones(1))
    torch.testing.assert_close(p, identity, atol=1e-7, rtol=1e-7)
    bad = p.clone(); bad[:, 3] = 1
    with pytest.raises(ValueError, match='Nonphysical'): g.physical_audit(bad)


@pytest.mark.parametrize('arm', c.ARMS)
def test_native_initialization_forward_rng_and_retained_reload(monkeypatch, arm):
    compact(monkeypatch)
    original = original_model(imported()['node']).eval(); rng = torch.get_rng_state()
    corrected = m.new_model(imported(), arm, stats()).eval()
    assert torch.equal(rng, torch.get_rng_state())
    for k, v in original.state_dict().items():
        assert torch.equal(v, corrected.state_dict()[k]), k
    x = inputs(); corrected.capture = True
    with torch.no_grad():
        expected = original(**x); actual = corrected(**x)
    assert torch.equal(expected, actual)
    assert corrected._hits == 6
    # Nonzero retained correction survives auxiliary removal and reload.
    with torch.no_grad():
        corrected.bias_map.weight.normal_(0, .1)
        head = corrected.particle_head or corrected.pair_head
        head.net[-1].weight.normal_(0, .03)
    corrected.capture = False
    expected = corrected(**x).detach()
    state = corrected.deployment_state()
    assert not any(k.startswith('predictor.') for k in state)
    other = m.new_model(imported(), arm, stats()).load_deployment(state)
    torch.testing.assert_close(other(**x), expected, rtol=0, atol=0)
    with pytest.raises(ValueError): other.load_deployment({k: v for k, v in state.items() if k != 'bias_map.weight'})


@pytest.mark.parametrize('arm', c.ARMS)
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_losses_gradient_routing_padding_and_no_targets_at_inference(monkeypatch, arm, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    compact(monkeypatch); net = m.new_model(imported(), arm, stats()).to(device).eval(); net.capture = True
    x = inputs(device=device); net(**x)
    target = (x['vectors'].transpose(1, 2)*1.2).requires_grad_()
    matched = x['mask'][:, 0].clone(); matched[0] = False
    extra, lp, lr = losses.correction_losses(net, target, matched)
    extra.backward(retain_graph=True)
    assert target.grad is None
    if c.ARMS[arm]['particle']:
        assert net.particle_head.net[-1].weight.grad.abs().sum() > 0
    if c.ARMS[arm]['direct']:
        assert net.pair_head.net[-1].weight.grad.abs().sum() > 0
    if arm == 'p4_u100': assert lr == 0
    if arm == 'dual_u100':
        gp = torch.autograd.grad(lr, net.particle_head.net[-1].weight, allow_unused=True, retain_graph=True)[0]
        assert gp is None or torch.count_nonzero(gp) == 0
    if arm == 'p4_pair_u100':
        gp = torch.autograd.grad(lr, net.particle_head.net[-1].weight, retain_graph=True)[0]
        assert gp.abs().sum() > 0
    # Once final layers and W move, classification supervises both branches and encoder.
    with torch.no_grad():
        net.bias_map.weight.normal_(0, .1)
        for head in (net.particle_head, net.pair_head):
            if head is not None: head.net[-1].weight.normal_(0, .05)
    net.zero_grad(set_to_none=True); logits = net(**x)
    torch.nn.functional.cross_entropy(logits, torch.arange(4, device=device)).backward()
    for head in (net.particle_head, net.pair_head):
        if head is not None: assert head.net[-1].weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.mod.blocks[0].parameters())
    net.capture = False
    assert torch.isfinite(net(**x)).all() and net.aux is None
    assert net._inputs is net._bias is None


def test_unique_pair_symmetry_zero_eligible_and_jet_weighting():
    mask = torch.tensor([[True, True, False], [True, False, False]])
    b, i, j = g.pair_indices(mask)
    assert (b.tolist(), i.tolist(), j.tolist()) == ([0], [0], [1])
    value = torch.tensor([2., 4., 10.], requires_grad=True)
    assert losses.jet_mean(value, torch.tensor([0, 0, 1]), 3).item() == pytest.approx(13/3)
    # Empty rows contribute zero, not a smaller denominator.
    assert losses.jet_mean(value[:0], torch.empty(0, dtype=torch.long), 3).item() == 0
    head = m.PairHead(); a, b = torch.randn(7, 32), torch.randn(7, 32); r = torch.randn(7, 4)
    with torch.no_grad(): head.net[-1].weight.normal_()
    assert torch.equal(head.pair(a, b, r), head.pair(b, a, r))


def test_chunk_gradient_parity_and_permutation(monkeypatch):
    compact(monkeypatch); a = m.new_model(imported(), 'dual_u100', stats()).eval(); a.capture = True
    with torch.no_grad():
        a.bias_map.weight.normal_(0, .2); a.pair_head.net[-1].weight.normal_(0, .05)
    b = m.new_model(imported(), 'dual_u100', stats()).eval(); b.load_state_dict(a.state_dict()); b.capture = True
    a.chunk_size = 2; b.chunk_size = 100000
    x = inputs(); ya = a(**x); yb = b(**x)
    torch.testing.assert_close(ya, yb, atol=1e-6, rtol=1e-5)
    ya.square().sum().backward(); yb.square().sum().backward()
    for (ka, pa), (kb, pb) in zip(a.named_parameters(), b.named_parameters()):
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad, pb.grad, atol=2e-5, rtol=3e-4)
    perm = torch.randperm(x['mask'].shape[-1]); xp = {k: v[..., perm] for k, v in x.items()}
    with torch.no_grad(): torch.testing.assert_close(a(**xp), ya, atol=1e-5, rtol=1e-5)


def test_pack_rejects_bad_mapping_and_statistics_train_only():
    tr = cache(coordinate='D000'); u = cache(coordinate='U100')
    block = tr.blocks[0]
    assigned = dict(offsets=block.offsets, mapping=np.concatenate([np.arange(h-l) for l,h in zip(block.offsets[:-1],block.offsets[1:])]),
                    identities=tr.identities, labels=tr.labels)
    u.blocks[0].vectors *= 1.1
    a = targets.pack(tr, u, [assigned]); bank = targets.ParticleBank(a, None)
    s = targets.statistics(tr, bank, 'cpu')
    assert s['role'] == 'train' and s['matched_particles'] == len(block.vectors)
    assert np.isfinite(s['pair_scale']).all()
    bad = deepcopy(assigned); bad['identities'][0, 3] = 8
    with pytest.raises(ValueError): targets.pack(tr, u, [bad])
    u.role = 'validation'
    with pytest.raises(ValueError): targets.pack(tr, u, [assigned])
    tr.role = 'validation'
    with pytest.raises(PermissionError): targets.statistics(tr, bank, 'cpu')


def fixture(root, rows=22):
    from types import SimpleNamespace
    from hlt_classification.luka_u100_pair_correction import campaign
    from hlt_classification.luka_u100_embedding import contracts as uc
    tr, val = cache(coordinate='D000', rows=rows), cache(role='validation', coordinate='D000', rows=rows)
    u = cache(coordinate='U100', rows=rows); u.blocks[0].vectors *= 1.1
    b = tr.blocks[0]
    assigned = dict(offsets=b.offsets, mapping=np.concatenate([np.arange(h-l) for l,h in zip(b.offsets[:-1],b.offsets[1:])]),
                    identities=tr.identities, labels=tr.labels)
    particles = targets.ParticleBank(targets.pack(tr, u, [assigned]), c.artifact('PARTICLE_BANK', statistics=stats()))
    bank = SimpleNamespace(targets=np.random.default_rng(4).normal(size=(rows, 128)).astype(np.float32),
        manifest=uc.artifact('BANK', role='train', coordinate='U100'), identities=tr.identities, labels=tr.labels)
    spec = dict(campaign.registration(), root=str(root), project_dir=str(root/'source'), content_hash='f'*64,
                imported=imported(), recovery_anchors=dict(M0HLT=.95, OFFLINE=.98))
    spec['counts'] = dict(train=rows, validation=rows)
    return spec, tr, val, bank, particles


@pytest.mark.parametrize('arm', c.ARMS)
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_production_fit_selection_reload_and_probe_isolation(tmp_path, monkeypatch, arm, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    from hlt_classification.luka_u100_pair_correction import training
    compact(monkeypatch); spec, tr, val, bank, particles = fixture(tmp_path)
    model = m.new_model(imported(), arm, stats()).to(device)
    report = training.fit(tmp_path/'fit', spec, arm, model, tr, val, bank, particles, device=device, technical=True)
    checked, arrays = training.validate_fit(tmp_path/'fit', spec, arm, technical=True)
    assert report == checked and report['epochs'] == 1 and len(arrays['labels']) == 22
    state = {k: v.clone() for k,v in model.state_dict().items()}
    before = torch.get_rng_state(); cuda_before = torch.cuda.get_rng_state() if device == 'cuda' else None
    model.train(); model.capture = False; model.alpha = .17
    training.diagnostic(model, tr, bank, particles, rho=.5, device=device)
    assert model.training and not model.capture and model.alpha == .17
    assert torch.equal(before, torch.get_rng_state())
    if cuda_before is not None: assert torch.equal(cuda_before, torch.cuda.get_rng_state())
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in state.items())
    # File hashes, not existence, establish reusable selected results.
    with (tmp_path/'fit/selected.pt').open('ab') as f: f.write(b'corruption')
    with pytest.raises(ValueError): training.validate_fit(tmp_path/'fit', spec, arm, technical=True)


def test_full30_and_finite_poor_metrics_never_gate(tmp_path, monkeypatch):
    from hlt_classification.luka_u100_pair_correction import training
    compact(monkeypatch); s, tr, val, bank, particles = fixture(tmp_path)
    # Deliberately uniform finite predictions: no quality threshold or early stop.
    monkeypatch.setattr(training, 'predict', lambda model, cache, **kw: np.full((len(cache), 11), 1/11, np.float32))
    model = m.new_model(imported(), 'pair_u100', stats())
    report = training.fit(tmp_path/'fit', s, 'pair_u100', model, tr, val, bank, particles, device='cpu')
    assert len(report['validation_history']) == 30 and report['selected_pass'] == 1
    assert report['validation']['macro_ovr_auc'] == .5
    training.validate_fit(tmp_path/'fit', s, 'pair_u100')


def test_zero_eligible_and_target_detachment(monkeypatch):
    compact(monkeypatch); model = m.new_model(imported(), 'p4_pair_u100', stats()).eval(); model.capture = True
    x = inputs(); model(**x)
    target = x['vectors'].transpose(1,2).clone().requires_grad_()
    loss, _, _ = losses.correction_losses(model, target, torch.zeros_like(x['mask'][:,0]))
    assert loss.item() == 0
    loss.backward(); assert target.grad is None
    # Diagonal and padding never get a correction, even when heads are nonzero.
    with torch.no_grad(): model.bias_map.weight.normal_(); model.particle_head.net[-1].weight.normal_(0,.01)
    model(**x)
    a = model.aux
    assert (a['i'] < a['j']).all()
    assert (a['mask'][a['batch'], a['i']] & a['mask'][a['batch'], a['j']]).all()
    assert a['bias'].abs().max() <= math.log(2)


def test_four_fit_debug_graph_no_control_teacher_or_target_reruns(tmp_path):
    from hlt_classification.luka_u100_pair_correction import submission, campaign
    s, *_ = fixture(tmp_path)
    setup, env = submission.render(s, 'setup'), submission.render(s, 'envelope')
    assert setup['jobs'] == 1 and env['jobs'] == 5 and campaign.registration()['reference_fits'] == 0
    rows = env['commands']
    assert [r['task_id'] for r in rows] == ['p4_pair_u100','pair_u100','dual_u100','p4_u100','report']
    assert [r['dependencies'] for r in rows[:4]] == [[],[],['p4_pair_u100'],['pair_u100']]
    for r in [*setup['commands'], *rows]:
        assert '--partition=debug' in r['command'] and '--gres=gpu:a100:1' in r['command']
        assert not any('afterok:' in arg for arg in r['command'])
    acceptance = c.artifact('ACCEPTANCE', minutes={a: 120 for a in c.ARMS})
    measured = submission.render(s, 'science', acceptance=acceptance)
    assert submission.within_envelope(env, measured)
    bad = deepcopy(measured); bad['commands'][0]['command'].append('--mem=500000M')
    assert not submission.within_envelope(env, bad)


def test_real_campaign_storage_guard_checks_fixed_allowance_and_headroom(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hlt_classification.luka_u100_pair_correction import campaign
    s, *_ = fixture(tmp_path)
    monkeypatch.setattr(campaign.shutil, 'disk_usage', lambda _: SimpleNamespace(free=8*2**30))
    with pytest.raises(OSError, match='headroom'): campaign.storage(s)
    monkeypatch.setattr(campaign.shutil, 'disk_usage', lambda _: SimpleNamespace(free=10*2**30))
    assert campaign.storage(s) == 0


def test_exact_dispatch_success_and_no_science_after_setup_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hlt_classification.luka_u100_pair_correction import submission, campaign, resources, worker
    from hlt_classification.scouting import hcwdl_exact_dag_submission as dag
    calls = []
    def run(command, **kwargs):
        calls.append(command); return SimpleNamespace(returncode=0, stdout=str(1000+len(calls))+'\n', stderr='')
    monkeypatch.setattr(dag.subprocess, 'run', run)
    monkeypatch.setattr(campaign, 'check', lambda *a, **kw: None)
    monkeypatch.setattr(submission, 'check_site', lambda *a: None)
    monkeypatch.setattr(resources, 'accepted', lambda s: c.artifact('ACCEPTANCE', minutes={a:120 for a in c.ARMS}))
    monkeypatch.setattr(worker, 'completed_setup', lambda s: None)
    monkeypatch.setattr(submission, 'accounting', lambda j: ['COMPLETED', '0:0'])
    s, *_ = fixture(tmp_path)
    review = submission.review(s)
    submission.dispatch(s, plan_hash=review['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 6 and len(submission.accepted_jobs(s, 'science')) == 5
    with pytest.raises(FileExistsError): submission.dispatch(s, plan_hash=review['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 6
    root = tmp_path/'failed'; root.mkdir(); failed, *_ = fixture(root)
    review = submission.review(failed)
    monkeypatch.setattr(submission, 'accounting', lambda j: ['FAILED','1:0'])
    with pytest.raises(RuntimeError, match='no scientific'):
        submission.dispatch(failed, plan_hash=review['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 7 and not (root/'submission_science').exists()


def test_resource_admission_uses_all_four_and_no_auc_threshold(tmp_path):
    from hlt_classification.luka_u100_pair_correction import resources
    from hlt_classification.data.cache_contracts import with_content_hash
    s, *_ = fixture(tmp_path)
    p = dict(miniature_complete=True, initial_native_parity=True, deployment_reload=True, train_bank_rows=22,
             validation_rows=22, largest_support_backward=True, largest_matched_pairs_backward=True,
             step_samples_seconds=[1.]*12, representative_step_seconds=1., validation_seconds=2., probe_seconds=1.)
    d = c.artifact('RESOURCE_DIAGNOSTIC', parents=dict(spec=s['content_hash']), status='complete', error=None,
        profiles={a:deepcopy(p) for a in c.ARMS}, native_pair_parity=True, device_type='cuda',
        memory=dict(peak_rss_bytes=1000, gpu_peak_bytes=1000), gpu=dict(total_memory_bytes=100000), environment={}, load_seconds=1.)
    assert set(resources.accept_diagnostic(d,s)['minutes']) == set(c.ARMS)
    bad = deepcopy(d); bad['profiles']['pair_u100']['largest_support_backward'] = False
    with pytest.raises(ValueError): resources.accept_diagnostic(with_content_hash(bad), s)
    bad = deepcopy(d); bad['memory']['gpu_peak_bytes'] = 99999
    with pytest.raises(MemoryError): resources.accept_diagnostic(with_content_hash(bad), s)


def test_source_scope_preserves_old_campaigns():
    from hlt_classification.luka_u100_pair_correction import source
    assert source.compatible_changes(['A\tsrc/hlt_classification/luka_u100_pair_correction/model.py'])
    assert not source.compatible_changes(['M\tsrc/hlt_classification/luka_u100_embedding/training.py'])
    assert '21993056' in source.REPLAY and "task='u100_w5_s0'" in source.REPLAY
    assert 'training.validate_fit' in source.REPLAY and 'worker.completed_setup' in source.REPLAY
    assert 'training.fit(' not in source.REPLAY


def test_pair_mlp_checkpoint_matches_dense_double_backward():
    from torch.utils.checkpoint import checkpoint
    a = m.PairHead().double(); b = deepcopy(a)
    with torch.no_grad():
        a.net[-1].weight.normal_(); b.load_state_dict(a.state_dict())
    x = torch.randn(13, 32, dtype=torch.float64, requires_grad=True)
    y = torch.randn(13, 32, dtype=torch.float64, requires_grad=True)
    r = torch.randn(13, 4, dtype=torch.float64)
    xa = x.detach().clone().requires_grad_(); ya = y.detach().clone().requires_grad_()
    dense = a.pair(x, y, r)
    chunks = torch.cat([checkpoint(b.pair, xa[k:k+3], ya[k:k+3], r[k:k+3], use_reentrant=False, preserve_rng_state=False)
                        for k in range(0, len(x), 3)])
    torch.testing.assert_close(dense, chunks, atol=1e-12, rtol=1e-12)
    dense.square().sum().backward(); chunks.square().sum().backward()
    torch.testing.assert_close(x.grad, xa.grad, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(y.grad, ya.grad, atol=1e-11, rtol=1e-11)
    for pa, pb in zip(a.net.parameters(), b.net.parameters()):
        torch.testing.assert_close(pa.grad, pb.grad, atol=1e-11, rtol=1e-11)


def test_actual_particle_export_reload_and_payload_corruption(tmp_path, monkeypatch):
    from hlt_classification.data.cache_contracts import atomic_publish_bytes, deterministic_npz_bytes
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    s, tr, val, bank, particles = fixture(tmp_path)
    prepared = tmp_path/'prepared'; prepared.mkdir(); (tmp_path/'setup').mkdir()
    b = tr.blocks[0]
    assignment = dict(offsets=b.offsets, mapping=np.concatenate([np.arange(h-l) for l,h in zip(b.offsets[:-1],b.offsets[1:])]).astype(np.int32),
                      identities=tr.identities, labels=tr.labels)
    atomic_publish_bytes(prepared/'mapping.npz', deterministic_npz_bytes(assignment))
    ref = dict(file_ref(prepared/'mapping.npz', root=prepared), file_index=0, rows=len(tr))
    meta = (None, dict(assignments=dict(train=[ref])), None, None)
    monkeypatch.setattr(targets, 'metadata', lambda _: meta)
    u = cache(coordinate='U100'); u.blocks[0].vectors *= 1.1
    calls = []
    def load(spec, role, coordinate, meta):
        calls.append((role, coordinate)); return u
    monkeypatch.setattr(targets, 'load_view', load)
    s['anchor'] = dict(evidence=dict(prepared_root=str(prepared)))
    actual = targets.export(s, tr, device='cpu')
    restored = targets.particle_bank(s, tr)
    np.testing.assert_array_equal(actual.vectors, restored.vectors)
    assert calls == [('train','U100')] and actual.manifest['pair_payload_stored'] is False
    with pytest.raises(PermissionError): targets.particle_bank(s, val)
    with (tmp_path/'setup/particles/targets.npz').open('ab') as f: f.write(b'bad')
    with pytest.raises(ValueError): targets.particle_bank(s, tr)


def test_failed_resource_check_publishes_diagnostic(tmp_path, monkeypatch):
    from hlt_classification.luka_u100_pair_correction import resources
    from hlt_classification.data.cache_contracts import load_json
    s, tr, val, bank, particles = fixture(tmp_path); (tmp_path/'setup').mkdir()
    monkeypatch.setattr(resources, 'installed_environment', lambda: {})
    s['reference'] = dict(environment={})
    def fail(*a, **kw): raise MemoryError('injected stress failure')
    monkeypatch.setattr(resources, 'native_geometry_parity', fail)
    with pytest.raises(MemoryError): resources.measure(s, tr, val, bank, particles, load_seconds=1., device='cpu')
    d = load_json(tmp_path/'setup/resource_diagnostic.json')
    assert d['status'] == 'failed' and 'injected stress' in d['error']
    assert not (tmp_path/'setup/acceptance.json').exists()


def test_partial_submission_preserves_claim_and_cannot_duplicate(tmp_path, monkeypatch):
    from hlt_classification.luka_u100_pair_correction import submission
    from hlt_classification.scouting import hcwdl_exact_dag_submission as dag
    s, *_ = fixture(tmp_path)
    plan = submission.render(s, 'science', acceptance=c.artifact('ACCEPTANCE', minutes={a:120 for a in c.ARMS}))
    submission._submit(s, plan, execute=False)
    calls = []
    def lost(command, **kwargs):
        calls.append(command); raise RuntimeError('scheduler response lost')
    monkeypatch.setattr(dag.subprocess, 'run', lost)
    with pytest.raises(RuntimeError, match='lost'): submission._submit(s, plan, execute=True)
    with pytest.raises(FileExistsError): submission._submit(s, plan, execute=True)
    assert len(calls) == 1 and (tmp_path/'submission_science/live_claim/intent.json').is_file()


def test_shell_environment_and_reuse_only_queue():
    from pathlib import Path
    queue = Path('scripts/queue_luka_u100_pair_correction.sh').read_text()
    worker = Path('sbatch/run_luka_u100_pair_correction.sh').read_text()
    for text in (queue, worker):
        assert 'PYTHONNOUSERSITE=1' in text and 'PYTHONDONTWRITEBYTECODE=1' in text
        assert '${CONDA_PREFIX}/lib' in text and 'atlas_kd_sporc' in text
    assert '21993056' in queue and 'nohup python' in queue
    assert '${PROJECT_DIR}/scripts/luka_u100_pair_correction.py' in worker


def test_production_dimensions_initial_parity_without_compact_override():
    native = original_model(imported()['node']).eval()
    net = m.new_model(imported(), 'dual_u100', stats()).eval()
    x = inputs()
    with torch.no_grad(): assert torch.equal(native(**x), net(**x))
    assert len(net.mod.blocks) == 8 and len(net.mod.cls_blocks) == 2


def test_geometry_double_finite_differences():
    p = torch.tensor([[2.,1.,.3,3.]], dtype=torch.float64)
    q = torch.tensor([[.5,2.,-.7,3.]], dtype=torch.float64)
    raw = torch.tensor([[.1,-.2,.3,.1]], dtype=torch.float64, requires_grad=True)
    scale = torch.tensor([.2,.1,.1,.2], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda r: g.pair_features(g.decode(p,r,scale,torch.ones(1)),q), (raw,), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize('coordinate,role', [('U100','train'),('D000','train'),('D000','validation')])
def test_missing_input_replay_exact_bytes_no_writes(all_prepared, tmp_path, coordinate, role):
    from pathlib import Path
    import shutil
    from hlt_classification.data.cache_contracts import sha256_file
    gate, manifest, prep, inventory, splits = all_prepared
    root = tmp_path/'prepared'; shutil.copytree(Path(gate['root'])/'prepared', root)
    s = dict(memory_mb=120000, counts=prep['counts'], anchor=dict(evidence=dict(prepared_root=str(root),
        raw_container=gate['input_container'], cache=manifest['content_hash'], prepared=prep['content_hash'])))
    meta = manifest, prep, inventory, splits
    expected = targets.load_view(s, role, coordinate, meta)
    ref = manifest['blocks'][role][coordinate][0]; (root/ref['path']).unlink()
    before = {str(p.relative_to(root)):sha256_file(p) for p in root.rglob('*') if p.is_file()}
    actual = targets.load_view(s, role, coordinate, meta)
    ix = np.arange(len(actual))[::-1]
    for k, v in expected.batch(ix).items(): np.testing.assert_array_equal(actual.batch(ix)[k], v)
    assert before == {str(p.relative_to(root)):sha256_file(p) for p in root.rglob('*') if p.is_file()}
    # A present corrupt cache is never silently reconstructed.
    (root/ref['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='bytes differ'): targets.load_view(s, role, coordinate, meta)
    with pytest.raises(PermissionError): targets.load_view(s, 'validation', 'U100', meta)


def test_reference_replay_program_binds_exact_zero_control_and_bank(tmp_path, monkeypatch, capsys):
    """Execute the original-source replay program with real saved control artifacts."""
    import sys
    from pathlib import Path
    from hlt_classification.data.cache_contracts import write_immutable_json
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    from hlt_classification.luka_u100_pair_correction import source
    from hlt_classification.luka_u100_contrastive import campaign as oc, worker as ow, training as ot, resources as ore, contracts as cc
    from hlt_classification.luka_u100_contrastive.model import new_model as cm
    from test_luka_u100_embedding import bank_fixture
    from test_luka_embedding_aux import tiny
    tiny(monkeypatch)
    tr, bank, teacher = bank_fixture(tmp_path)
    s = dict(oc.registration(), root=str(tmp_path), content_hash='f'*64, imported=imported(),
             anchor=dict(evidence=dict(prepared=tr.foundation_sha256)))
    s['counts'] = dict(train=len(tr), validation=22)
    write_immutable_json(tmp_path/'campaign_spec.json', s)
    ot.fit(tmp_path/'u100_w5_s0/fit', s, 'u100_w5_s0', cm(imported(),'u100_w5_s0'), tr,
           cache(role='validation',coordinate='D000'), bank, None, device='cpu')
    (tmp_path/'setup').mkdir()
    write_immutable_json(tmp_path/'setup/bundle.json', cc.artifact('BUNDLE', parents=dict(spec=s['content_hash']),
        u100_manifest=file_ref(tmp_path/'targets/manifest.json')))
    for path in (tmp_path/'setup/receipt.json', tmp_path/'u100_w5_s0/receipt.json'):
        write_immutable_json(path, dict(job_id='21993056'))
    write_immutable_json(tmp_path/'setup/acceptance.json', dict(environment={}))
    monkeypatch.setattr(oc, 'check', lambda *a,**kw: None)
    monkeypatch.setattr(ow, 'completed_setup', lambda *a,**kw: None)
    monkeypatch.setattr(ow, '_receipt', lambda *a,**kw: dict(job_id='21993056'))
    monkeypatch.setattr(ore, 'accepted', lambda _: dict(environment={}))
    monkeypatch.setattr(sys, 'argv', ['replay', str(tmp_path/'campaign_spec.json')])
    exec(source.REPLAY, {})
    assert 'PAIR_CONTROL=' in capsys.readouterr().out
    monkeypatch.setattr(ow, '_receipt', lambda *a,**kw: dict(job_id='12345'))
    with pytest.raises(ValueError, match='21993056'): exec(source.REPLAY,{})
    monkeypatch.setattr(ow, '_receipt', lambda *a,**kw: dict(job_id='21993056'))
    with (tmp_path/'targets/targets.npz').open('ab') as f: f.write(b'bad')
    with pytest.raises(ValueError, match='bytes differ'): exec(source.REPLAY,{})

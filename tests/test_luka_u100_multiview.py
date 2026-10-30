"""Scientific boundaries, native parity, artifact failure and queue isolation."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from hlt_classification.luka_u100_multiview import contracts as c, views, teachers, targets, training, submission
from hlt_classification.luka_u100_multiview.model import Student, FrozenReadout, objective, new_student
from hlt_classification.luka_embedding_aux.model import new_model as old_model
from hlt_classification.luka_u100_embedding.training import objective as old_objective
from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.cms2jc2_response.bridge import Particles
from test_luka_u100_pretrain import cache
from test_luka_embedding_aux import tiny


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def node(): return dict(initialization_seed=13, sampler_seed=29, coordinate='D000', teacher=None)


def particles(n, shift=0):
    pt = np.arange(1, n+1, dtype=float) + shift
    p4 = np.column_stack((pt, pt*.2, pt*.1, pt*1.2))
    return Particles(p4, np.zeros(n), np.ones(n), np.zeros((n, 4)), np.zeros((n, 4), bool), tuple(map(str, range(n))))


def test_inventory_and_rational_coefficients():
    assert len(c.ARMS) == 8
    assert sum(k.startswith('teacher_') for k in c.TASKS) == 2
    for arm in c.ARMS:
        assert sum(c.weights(arm).values()) == (0 if arm == 'hlt_capacity_control' else 5)
    assert c.weights('u100_concat_fusion')['U100'] == Fraction(5, 3)
    assert c.recipe()['base']['maximum_passes'] == 30
    for arm in c.ARMS:
        assert not set(c.TASKS[arm]) & set(c.ARMS)
    imported = dict(node=dict(node(), auxiliary_weight=5), oracle_node=dict(node(), coordinate='OFFLINE'))
    assert c.node(imported, 'hlt_capacity_control')['total_embedding_weight'] == 0
    assert 'auxiliary_weight' not in c.node(imported, 'hlt_capacity_control')
    assert c.node(imported, 'teacher_CONCAT')['coordinate'] == 'CONCAT'
    assert c.node(imported, 'teacher_FUSION')['context_coordinate'] == 'OFFLINE'
    for arm in c.ARMS:
        assert c.node(imported, arm)['node_id'] == arm
        assert c.node(imported, arm)['initialization_seed'] == node()['initialization_seed']


def test_union_untagged_full_support_order_and_duplicates():
    h, o = particles(300), particles(301)
    a, b = views.union(h, o), views.union(o.take(np.arange(301)[::-1]), h)
    assert len(a) == 601
    np.testing.assert_array_equal(a.p4, b.p4)
    x, y = (views.build_inputs(v, capacity=len(v)) for v in (a, b))
    assert x.features.shape == (601, 17)
    np.testing.assert_array_equal(x.features, y.features)
    assert len(views.union(h, h)) == 600
    # Common union axis changes relative features compared with source-local preprocessing.
    other = particles(2, 1000)
    assert not np.array_equal(views.build_inputs(views.union(h, other), capacity=302).features[-300:, 2],
                              views.build_inputs(h, capacity=300).features[:, 2])


def test_new_heads_preserve_native_and_u100_rng(monkeypatch):
    tiny(monkeypatch)
    old = old_model(node())
    for arm in c.ARMS:
        model = new_student(arm, node())
        for k, v in old.state_dict().items():
            if k.startswith('mod.'):
                assert torch.equal(model.state_dict()[k], v)
        if 'U100' in model.predictors:
            for a, b in zip(old.predictor.parameters(), model.predictors['U100'].parameters()):
                assert torch.equal(a, b)
    torch.manual_seed(13); old_model(node()); expected = torch.rand(3)
    new_student('u100_concat_fusion', node()); assert torch.equal(expected, torch.rand(3))


def test_single_target_kernel_and_retained_initialization(monkeypatch):
    tiny(monkeypatch)
    raw = cache(coordinate='D000').batch(np.arange(3))
    old = old_model(node()).eval(); old.capture = True
    model = new_student('u100_w5', node()).eval()
    old_logits = teachers.forward(old, raw, 'cpu', bf16=False)
    logits = teachers.forward(model, raw, 'cpu', bf16=False)
    assert torch.equal(logits, old_logits)
    target = torch.randn(3, 128); labels = torch.tensor(raw['labels'])
    a = old_objective(logits, labels, old.prediction, target, coefficient=5)[0]
    b = objective('u100_w5', logits, labels, model.predictions, {'U100': target}, pass_coordinate=1)[0]
    assert torch.equal(a, b)
    for arm in ('u100_use_prediction', 'hlt_capacity_control'):
        retained = new_student(arm, node()).eval()
        actual = teachers.forward(retained, raw, 'cpu', bf16=False)
        torch.testing.assert_close(actual, logits, atol=1e-6, rtol=1e-6)
        assert retained.combined.weight[:, 128:].count_nonzero() == 0
        native = retained.deployed()
        torch.testing.assert_close(teachers.forward(native, raw, 'cpu', bf16=False), actual)


@pytest.mark.parametrize('mode', ['ce', 'kd'])
def test_frozen_readout_gradient_direction_and_temperature(mode):
    torch.manual_seed(7)
    pred = torch.randn(4, 128, requires_grad=True)
    readout = FrozenReadout(np.ones(128), np.full(128, 2.), torch.randn(11, 128), torch.randn(11))
    state = deepcopy(readout.state_dict())
    labels = torch.arange(4); logits = torch.randn(4, 11, requires_grad=True); target = torch.randn(4, 128)
    teacher = torch.randn(4, 11, requires_grad=True)
    loss, parts = objective('u100_readout_'+mode, logits, labels, {'U100': pred}, {'U100': target},
        pass_coordinate=1.5, readout=readout, teacher_logits=teacher)
    r = readout(pred)
    if mode == 'kd':
        q = teacher.detach().softmax(1)  # deliberately wrong T is not the expected value
        expected = 4*torch.nn.functional.kl_div((r/2).log_softmax(1), (teacher.detach()/2).softmax(1), reduction='batchmean')
        torch.testing.assert_close(parts['readout'], expected)
    else:
        torch.testing.assert_close(parts['readout'], torch.nn.functional.cross_entropy(r, labels))
    torch.testing.assert_close(parts['weighted_readout'], parts['readout']/2)
    loss.backward()
    assert pred.grad.norm() > 0 and logits.grad.norm() > 0 and teacher.grad is None
    assert not list(readout.parameters())
    assert all(torch.equal(v, state[k]) for k, v in readout.state_dict().items())


def make_bank(tmp_path, monkeypatch):
    tiny(monkeypatch)
    model = teachers.new_teacher('U100', node())
    save_state(tmp_path/'teacher.pt', model.state_dict()); ref = file_ref(tmp_path/'teacher.pt')
    targets.export(tmp_path/'bank', model, cache(coordinate='U100'), 'U100', ref, device='cpu', source_hash='f'*64)
    return targets.Bank(tmp_path/'bank', cache(coordinate='D000'), 'U100', ref, source_hash='f'*64), ref


def test_bank_authentication_and_train_only(tmp_path, monkeypatch):
    bank, ref = make_bank(tmp_path, monkeypatch)
    assert bank.arrays['targets'].dtype == np.float32
    assert bank.arrays['mean'].dtype == np.float64
    with pytest.raises(PermissionError):
        targets.Bank(tmp_path/'bank', cache('validation', coordinate='D000'), 'U100', ref, source_hash='f'*64)
    with pytest.raises(ValueError):
        targets.Bank(tmp_path/'bank', cache(coordinate='D000'), 'CONCAT', ref, source_hash='f'*64)
    wrong = cache(coordinate='D000'); wrong.identities = wrong.identities[::-1].copy()
    with pytest.raises(ValueError): targets.Bank(tmp_path/'bank', wrong, 'U100', ref, source_hash='f'*64)
    bad = deepcopy(bank.arrays); bad['mean'] += 3
    with pytest.raises(AssertionError): targets.roundtrip(bad)
    (tmp_path/'bank/targets.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError): targets.Bank(tmp_path/'bank', cache(coordinate='D000'), 'U100', ref, source_hash='f'*64)


@pytest.mark.parametrize('arm', list(c.ARMS))
def test_complete_miniature_and_deployed_readback(tmp_path, monkeypatch, arm):
    bank, _ = make_bank(tmp_path, monkeypatch)
    banks = {n: bank for n in c.ARMS[arm]['targets']}
    spec = dict(content_hash='a'*64, counts=dict(train=22, validation=22))
    train, val = cache(coordinate='D000'), cache('validation', coordinate='D000')
    model = new_student(arm, node())
    training.fit(tmp_path/'fit', spec, arm, model, train, val, node(), banks, device='cpu', technical=True)
    report, a = training.validate_fit(tmp_path/'fit', spec, arm, technical=True)
    assert report['selected_epoch'] in (1, 2) and a['probabilities'].shape == (22, 11)
    state = torch.load(tmp_path/'fit/selected.pt', weights_only=True)
    assert any(k.startswith('predictors.') for k in state) == c.ARMS[arm]['retained']


def test_poor_finite_fit_runs_all30(tmp_path, monkeypatch):
    bank, _ = make_bank(tmp_path, monkeypatch)
    spec = dict(content_hash='a'*64, counts=dict(train=22, validation=22))
    training.fit(tmp_path/'fit', spec, 'u100_w5', new_student('u100_w5', node()),
        cache(coordinate='D000'), cache('validation', coordinate='D000'), node(), {'U100': bank}, device='cpu')
    report, _ = training.validate_fit(tmp_path/'fit', spec, 'u100_w5')
    assert len(report['history']) == 30


def test_debug_dag_isolation_and_afterany_report(monkeypatch):
    from hlt_classification.luka_u100_multiview import acceptance
    measured = dict(content_hash='b'*64, minutes={n: 180 for n in c.TASKS if n != 'accept'})
    monkeypatch.setattr(acceptance, 'accepted', lambda _: measured)
    spec = dict(content_hash='a'*64, root='/campaign', project_dir='/project',
        execution_site=execution_site('sporc_a100_debug'), cpus=6, memory_mb=120000, acceptance_minutes=720)
    plan = submission.render(spec, 'science'); rows = {r['task_id']: r for r in plan['commands']}
    assert len(rows) == 12
    assert rows['u100_w5']['dependencies'] == ['export_U100']
    assert rows['hlt_capacity_control']['dependencies'] == []
    assert set(rows['u100_concat_fusion']['dependencies']) == {'export_U100', 'teacher_CONCAT', 'teacher_FUSION'}
    assert any(x.startswith('--dependency=afterany:') for x in rows['report']['command'])
    assert all('--partition=debug' in r['command'] for r in rows.values())
    assert submission.render(spec, 'accept')['jobs'] == 1


def test_source_and_shell_boundaries():
    root = Path(__file__).resolve().parents[1]
    worker = (root/'sbatch/run_luka_u100_multiview.sh').read_text()
    assert '${PROJECT_DIR}/scripts/luka_u100_multiview.py' in worker
    assert 'PYTHONNOUSERSITE=1' in worker and '${CONDA_PREFIX}/lib' in worker
    files = list((root/'src/hlt_classification/luka_u100_multiview').glob('*.py'))
    assert all('import Fresh_check' not in p.read_text() for p in files)


def test_native_union_permutation_and_fusion_context_gradients():
    # Authoritative installed factory at the full registered dimensions/depths.
    from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
    torch.manual_seed(13); native = DelphesParticleTransformer().eval()
    torch.manual_seed(13); fusion = teachers.new_teacher('FUSION', node()).eval()
    raw = cache(coordinate='D000').batch(np.arange(2))
    original = teachers.forward(native, raw, 'cpu', bf16=False)
    reordered = dict(raw)
    for k in ('features', 'vectors', 'mask'): reordered[k] = raw[k][..., ::-1].copy()
    torch.testing.assert_close(teachers.forward(native, reordered, 'cpu', bf16=False), original, rtol=2e-5, atol=2e-6)
    paired = dict(raw, **{'context_'+k: raw[k].copy() for k in ('features', 'vectors', 'mask')})
    zero = teachers.forward(fusion, paired, 'cpu', bf16=False, name='FUSION')
    torch.testing.assert_close(zero, original, rtol=2e-5, atol=2e-6)
    torch.nn.functional.cross_entropy(zero, torch.tensor(raw['labels'])).backward()
    assert all(layer.residual_projection.weight.grad.norm() > 0 for layer in fusion.injections)
    with torch.no_grad():
        for layer in fusion.injections: layer.residual_projection.weight.copy_(torch.eye(128)*.1)
    fusion.zero_grad(set_to_none=True)
    actual = teachers.forward(fusion, paired, 'cpu', bf16=False, name='FUSION')
    actual.square().mean().backward()
    assert fusion.context_mod.embed.input_bn.weight.grad.norm() > 0
    changed = deepcopy(paired); changed['context_features'] += .5
    assert not torch.allclose(teachers.forward(fusion, changed, 'cpu', bf16=False, name='FUSION'), actual)
    # Changing padded context fields must not change valid inference.
    padded = deepcopy(paired)
    mask = np.broadcast_to(~padded['context_mask'], padded['context_features'].shape)
    padded['context_features'][mask] = 123
    torch.testing.assert_close(teachers.forward(fusion, padded, 'cpu', bf16=False, name='FUSION'), actual, rtol=2e-5, atol=2e-6)


def test_gradient_diagnostic_preserves_rng_state_and_update(monkeypatch):
    tiny(monkeypatch)
    a, b = (new_student('u100_w5', node()) for _ in range(2))
    raw = cache(coordinate='D000').batch(np.arange(4)); target = torch.zeros(4, 128)
    states = []
    for model, diagnostic in ((a, False), (b, True)):
        torch.manual_seed(77)
        opt = training.optimizer(model); logits = teachers.forward(model, raw, 'cpu', bf16=False)
        loss, parts = objective('u100_w5', logits, torch.tensor(raw['labels']), model.predictions,
                                {'U100': target}, pass_coordinate=.5)
        before = torch.random.get_rng_state().clone()
        if diagnostic: training.gradient_diagnostic(parts, model)
        assert torch.equal(before, torch.random.get_rng_state())
        loss.backward(); opt.step()
        states.append(deepcopy(model.state_dict()))
    assert all(torch.equal(v, states[1][k]) for k, v in states[0].items())


def test_target_nonfinite_and_control_rejects_teacher():
    logits = torch.zeros(2, 11); labels = torch.tensor([0, 1]); pred = torch.zeros(2, 128)
    with pytest.raises(ValueError):
        objective('hlt_capacity_control', logits, labels, {'U100': pred}, {'U100': pred}, pass_coordinate=1)
    with pytest.raises(ValueError):
        objective('u100_w5', logits, labels, {'U100': pred}, {'U100': pred+float('nan')}, pass_coordinate=1)


def test_full_baseline_training_parity(tmp_path, monkeypatch):
    from hlt_classification.luka_u100_embedding.training import train as original_train
    bank, _ = make_bank(tmp_path, monkeypatch)
    tr, val = cache(coordinate='D000'), cache('validation', coordinate='D000')
    # The old kernel validates its own bank envelope; use the old exporter rather than changing it.
    from hlt_classification.luka_u100_embedding import targets as original_targets, contracts as original_contracts
    teacher = teachers.new_teacher('U100', node()); save_state(tmp_path/'original_teacher.pt', teacher.state_dict())
    record = original_contracts.artifact('TEACHER', coordinate='U100', target_role='train', prepared=tr.foundation_sha256,
                                        files=dict(checkpoint=file_ref(tmp_path/'original_teacher.pt')))
    original_targets.export(tmp_path/'old_bank', teacher, cache(coordinate='U100'), teacher_record=record, device='cpu')
    oldbank = original_targets.Bank(tmp_path/'old_bank', tr, teacher_record=record)
    np.testing.assert_array_equal(bank.arrays['targets'], oldbank.targets)
    oldreport, oldstate = original_train(old_model(node()), tr, val, bank=oldbank,
        node=dict(node(), auxiliary_weight=5), device='cpu')
    spec = dict(content_hash='a'*64, counts=dict(train=22, validation=22))
    training.fit(tmp_path/'new', spec, 'u100_w5', new_student('u100_w5', node()), tr, val, node(), {'U100': bank}, device='cpu')
    report, _ = training.validate_fit(tmp_path/'new', spec, 'u100_w5')
    state = torch.load(tmp_path/'new/selected.pt', weights_only=True)
    assert all(torch.equal(state[k], v) for k, v in oldstate.items())
    assert [r['validation'] for r in report['history']] == [r['validation'] for r in oldreport['validation_history']]


def test_retirement_exact_ownership_and_missing_evidence(tmp_path):
    from hlt_classification.luka_u100_multiview.retention import validate_output, allowed
    spec = dict(root=str(tmp_path), content_hash='a'*64)
    refs = []
    for relative in allowed():
        p = tmp_path/relative; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'payload')
        refs.append(file_ref(p, root=tmp_path))
    missing = tmp_path/allowed()[0]; missing.unlink()
    with pytest.raises(FileNotFoundError): validate_output(spec, refs[0], root=tmp_path)
    record = c.artifact('RETIREMENT', parents=dict(spec=spec['content_hash']), outputs=refs)
    write_immutable_json(tmp_path/'retirement_intent.json', record)
    assert validate_output(spec, refs[0], root=tmp_path) == missing
    with pytest.raises(ValueError):
        validate_output(spec, dict(refs[0], path='teacher_CONCAT/fit/selected.pt'), root=tmp_path)


def test_exact_submission_duplicates_partial_and_source_drift(tmp_path, monkeypatch):
    from hlt_classification.luka_u100_multiview import acceptance, campaign
    from hlt_classification.scouting import hcwdl_exact_dag_submission as exact
    measured = dict(content_hash='b'*64, minutes={n: 180 for n in c.TASKS if n != 'accept'})
    monkeypatch.setattr(acceptance, 'accepted', lambda _: measured)
    monkeypatch.setattr(campaign, 'check', lambda s: s)
    monkeypatch.setattr(submission, 'check_site', lambda _: None)
    spec = dict(content_hash='a'*64, root=str(tmp_path), project_dir='/project',
        execution_site=execution_site('sporc_a100_debug'), cpus=6, memory_mb=120000, acceptance_minutes=720)
    submitted = []
    def fake_run(command, **kwargs):
        submitted.append(command)
        return SimpleNamespace(returncode=0, stdout=str(100+len(submitted))+'\n', stderr='')
    monkeypatch.setattr(exact.subprocess, 'run', fake_run)
    plan = submission.submit(spec, 'science')
    with pytest.raises(PermissionError):
        submission.submit(spec, 'science', execute=True, plan_hash='c'*64, authorization=c.AUTHORIZATION)
    ledger = submission.submit(spec, 'science', execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    assert len(submitted) == 12 and len(ledger['jobs']) == 12
    assert submission.submit(spec, 'science', execute=True, plan_hash=plan['content_hash'],
                             authorization=c.AUTHORIZATION) == ledger
    assert len(submitted) == 12
    def drift(_): raise ValueError('source drift')
    monkeypatch.setattr(campaign, 'check', drift)
    with pytest.raises(ValueError, match='source drift'): submission.submit(spec, 'science')
    monkeypatch.setattr(campaign, 'check', lambda s: s)
    # A permanent live claim blocks a partial re-submit, even without a final ledger.
    second = dict(spec, root=str(tmp_path/'partial'))
    Path(second['root']).mkdir()
    p = submission.submit(second, 'science')
    (Path(second['root'])/'submission_science/live_claim').mkdir()
    with pytest.raises(FileExistsError):
        submission.submit(second, 'science', execute=True, plan_hash=p['content_hash'], authorization=c.AUTHORIZATION)
    assert len(submitted) == 12


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_full_models_readout_and_fusion_minis(tmp_path):
    from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
    spec = dict(content_hash='a'*64, counts=dict(train=22, validation=22))
    tr, va = cache(coordinate='D000'), cache('validation', coordinate='D000')
    teacher = teachers.new_teacher('U100', node()).to('cuda')
    save_state(tmp_path/'teacher.pt', {k: v.cpu() for k, v in teacher.state_dict().items()})
    ref = file_ref(tmp_path/'teacher.pt')
    targets.export(tmp_path/'bank', teacher, cache(coordinate='U100'), 'U100', ref, device='cuda', source_hash='f'*64)
    bank = targets.Bank(tmp_path/'bank', tr, 'U100', ref, source_hash='f'*64)
    with parity_backend('cuda'):
        raw = tr.batch(np.arange(4))
        old = old_model(node()).to('cuda').eval(); old.capture = True
        new = new_student('u100_w5', node()).to('cuda').eval()
        old_out = teachers.forward(old, raw, 'cuda', bf16=False)
        new_out = teachers.forward(new, raw, 'cuda', bf16=False)
        torch.testing.assert_close(new_out, old_out, rtol=0, atol=0)
        old_out.sum().backward(); new_out.sum().backward()
        old_grad = dict(old.named_parameters())
        for k, p in new.named_parameters():
            if k.startswith('mod.') and p.grad is not None:
                torch.testing.assert_close(p.grad, old_grad[k].grad, rtol=0, atol=0)
    training.fit(tmp_path/'readout', spec, 'u100_readout_ce', new_student('u100_readout_ce', node()),
                 tr, va, node(), {'U100': bank}, device='cuda', technical=True)
    training.validate_fit(tmp_path/'readout', spec, 'u100_readout_ce', technical=True)
    training.fit(tmp_path/'fusion', spec, 'teacher_FUSION', teachers.new_teacher('FUSION', node()),
        views.PairedCache(tr, cache(coordinate='OFFLINE')),
        views.PairedCache(va, cache('validation', coordinate='OFFLINE')), node(), {}, device='cuda', technical=True)
    training.validate_fit(tmp_path/'fusion', spec, 'teacher_FUSION', technical=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA production-worker fixture')
def test_synthetic_complete_science_worker_dag(tmp_path, monkeypatch):
    """Only source/allocation/data providers are fixtures; all task code publishes/replays."""
    from hlt_classification.luka_u100_multiview import worker, campaign, acceptance, retention
    from hlt_classification.data.cache_contracts import atomic_publish_bytes, deterministic_npz_bytes
    tiny(monkeypatch)
    # Keep actual fusion code with its full depths; only native toy models shrink.
    site = execution_site('sporc_a100_debug'); active = ['export_U100']
    jobs = {t: str(1000+i) for i, t in enumerate(c.TASKS)}
    spec = dict(content_hash='a'*64, root=str(tmp_path), recipe=c.recipe(),
        counts=dict(train=22, validation=22), original50k_rows=22,
        source_snapshot=dict(content_hash='f'*64), execution_site=site, cpus=6, memory_mb=120000,
        anchor=dict(evidence=dict(prepared='a'*64, models={})),
        imported=dict(node=node(), oracle_node=node(), teacher=dict(node=node(), files={}), files={}))
    # Published real frozen native target bank, reused through export_U100 worker.
    teacher = teachers.new_teacher('U100', node()).to('cuda')
    save_state(tmp_path/'original.pt', teacher.state_dict())
    ref = file_ref(tmp_path/'original.pt'); spec['imported']['teacher']['files']['checkpoint'] = ref
    targets.export(tmp_path/'accept/bank_U100', teacher, cache(coordinate='U100'), 'U100', ref,
                   device='cuda', source_hash='f'*64)
    val = cache('validation', coordinate='D000')
    probabilities = np.full((len(val), 11), 1/11, np.float32)
    probabilities[np.arange(len(val)), val.labels] += .1
    probabilities /= probabilities.sum(1, keepdims=True)
    arrays = dict(identities=val.identities, labels=val.labels, probabilities=probabilities)
    atomic_publish_bytes(tmp_path/'reference.npz', deterministic_npz_bytes(arrays))
    pred_ref = file_ref(tmp_path/'reference.npz')
    spec['anchor']['evidence']['models'] = {n: dict(predictions=pred_ref) for n in ('M0HLT', 'OFFLINE')}
    spec['imported']['files']['predictions'] = pred_ref
    spec['imported']['teacher']['files']['predictions'] = pred_ref
    atomic_publish_bytes(tmp_path/'subset.npz', deterministic_npz_bytes(dict(indices=np.arange(22, dtype=np.int64))))
    spec['imported']['files']['subset'] = file_ref(tmp_path/'subset.npz')
    measured = dict(content_hash='b'*64, environment=dict(fixture=True), gpu=dict(fixture=True))
    monkeypatch.setattr(campaign, 'check', lambda s: s)
    monkeypatch.setattr(worker, 'allocation', lambda _: (jobs[active[0]], 6, 120000))
    monkeypatch.setattr(submission, 'accepted_jobs', lambda *_: jobs)
    monkeypatch.setattr(acceptance, 'accepted', lambda _: measured)
    monkeypatch.setattr(worker, 'installed_environment', lambda: dict(fixture=True))
    monkeypatch.setattr(worker, 'gpu_identity', lambda: dict(fixture=True))
    def data(s, role, coordinate):
        if coordinate == 'FUSION': return views.PairedCache(cache(role, coordinate='D000'), cache(role, coordinate='OFFLINE'))
        return cache(role, coordinate=coordinate)
    monkeypatch.setattr(views, 'load', data)
    for task in c.TASKS:
        if task == 'accept': continue
        active[0] = task
        worker.run(spec, task)
        worker.completed(spec, task)
    result = worker.results(spec)
    assert result['pending'] == []
    assert len(result['rows']) == 14
    with pytest.raises(FileExistsError):
        active[0] = 'u100_w5'; worker.run(spec, 'u100_w5')
    # Capacity-control receipt binds no teacher bank parents.
    control = load_json(tmp_path/'hlt_capacity_control/fit/training_report.json')
    assert set(control['parents']) == {'spec', 'recipe', 'prepared'}


def test_historical_result_archive_allows_retired_bank_but_authenticates_results(tmp_path, monkeypatch, capsys):
    """Original source validates the full fit; only retired TRAIN bytes are irrelevant."""
    import sys
    from test_luka_u100_embedding import bank_fixture, fixture_gate
    from hlt_classification.luka_u100_embedding import direct, worker, contracts as old
    from hlt_classification.luka_u100_multiview import source
    from hlt_classification.data.cache_contracts import atomic_publish_bytes, deterministic_npz_bytes
    tiny(monkeypatch)
    tr, bank, teacher = bank_fixture(tmp_path)
    spec = fixture_gate(tmp_path, teacher=deepcopy(teacher))
    spec['teacher'] = teacher
    manifest = load_json(tmp_path/'targets/manifest.json')
    spec.update(resources=dict(content_hash='e'*64), original50k_rows=22)
    write_immutable_json(tmp_path/'campaign_spec.json', spec)
    monkeypatch.setattr(direct, 'check_spec', lambda s: s)
    val = cache('validation', coordinate='D000')
    fit = worker.fit(tmp_path/'fit', spec, tr, val, bank, device='cpu', technical=False)
    _, student = worker.validate_fit(tmp_path/'fit', spec, manifest['content_hash'], technical=False)
    refs = tmp_path/'references'; refs.mkdir()
    for name in ('M0HLT', 'OFFLINE', 'OFFLINE_EMBED_W5', 'COARSE_U100'):
        atomic_publish_bytes(refs/(name+'.npz'), deterministic_npz_bytes(student))
    atomic_publish_bytes(refs/'original50k.npz', deterministic_npz_bytes(dict(indices=np.arange(22, dtype=np.int64))))
    proof = old.artifact('TEACHER_READBACK', parents=dict(teacher=teacher['content_hash']),
        native_probability_parity=True, normalization_used=False, coordinate='U100', role='validation',
        rows=22, rtol=1e-5, atol=1e-6, maximum_absolute_error=0.)
    write_immutable_json(tmp_path/'teacher_readback.json', proof)
    result = old.artifact('DIRECT_RESULTS', parents=dict(spec=spec['content_hash'], fit=fit['content_hash']),
        counts=spec['counts'], new_preflight_performed=False, selection_population='entire_validation',
        original50k_selection=False, single_training_seed=True,
        rows=worker.comparison(spec, student, folder=refs))
    write_immutable_json(tmp_path/'results.json', result)
    files = ['teacher_readback.json', 'targets/manifest.json', 'targets/targets.npz', 'results.json',
             'fit/receipt.json', 'fit/selected.pt', 'fit/training_report.json', 'fit/validation_predictions.npz']
    files += ['references/'+n+'.npz' for n in ('M0HLT', 'OFFLINE', 'OFFLINE_EMBED_W5', 'COARSE_U100', 'original50k')]
    receipt = old.artifact('DIRECT_SCIENCE', parents=dict(spec=spec['content_hash'], resources='e'*64),
        job_id='123', new_preflight_performed=False, outputs=[file_ref(tmp_path/f, root=tmp_path) for f in files])
    write_immutable_json(tmp_path/'receipt.json', receipt)
    (tmp_path/'targets/targets.npz').unlink()
    monkeypatch.setattr(sys, 'argv', ['replay', str(tmp_path/'campaign_spec.json')])
    exec(compile(source.REPLAY, '<archive-replay>', 'exec'), {})
    assert 'MULTIVIEW_IMPORT=' in capsys.readouterr().out
    (tmp_path/'fit/validation_predictions.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError): exec(compile(source.REPLAY, '<archive-replay>', 'exec'), {})

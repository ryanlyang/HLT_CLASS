"""Scientific parity, selected-target transfer, exact release and failure checks."""
from types import SimpleNamespace
import subprocess
import numpy as np
import pytest
import torch
from test_luka_u100_pretrain import cache
from test_luka_embedding_aux import tiny
from hlt_classification.data.cache_contracts import (load_json, write_immutable_json, with_content_hash,
    atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import _forward
from hlt_classification.luka_embedding_aux.model import AuxiliaryParT
from hlt_classification.luka_embedding_aux import targets as target_impl
from hlt_classification.luka_supervision_depth.model import supcon
from hlt_classification.luka_supcon5 import contracts as c, model, training, campaign, submission, resources, worker, inputs


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def spec(root):
    return dict(campaign.policy(), root=str(root), project_dir='/project', content_hash='c'*64,
        anchor=dict(evidence=dict(prepared='a'*64)), counts=dict(train=22, validation=22),
        nodes={t: dict(initialization_seed=23, sampler_seed=29) for t in c.FITS},
        recovery_anchors=dict(M0HLT=.95, OFFLINE=.98), reference=dict(environment={'fixture': True}))


def banks(train):
    rng = np.random.default_rng(8)
    return {k: SimpleNamespace(targets=rng.normal(size=(len(train), 128)).astype(np.float32),
        identities=train.identities, labels=train.labels, manifest=c.artifact('TEST_BANK', role='train', name=k))
        for k in ('u100', 'offline')}


def test_exact_two_fits_and_dependency_graph(tmp_path):
    s = spec(tmp_path); r = c.recipe()
    assert c.FITS == ('teacher_supcon5', 'student_supcon5')
    assert r['teacher_coefficients'] == {'teacher_supcon5': 5.}
    assert r['student_weights'] == {'U100': 5., 'OFFLINE': 5.}
    assert not r['reference_retrained'] and r['teacher_export'] == 'native_preclassifier_128_not_projection'
    rows = submission.render(s, 'envelope')['commands']
    assert [r['task_id'] for r in rows] == [*c.FITS, 'report']
    assert rows[0]['dependencies'] == [] and rows[1]['dependencies'] == ['teacher_supcon5']
    assert '--dependency=afterok:${JOB_teacher_supcon5}' in rows[1]['command']
    assert '--kill-on-invalid-dep=yes' in rows[1]['command']
    assert '--dependency=afterany:${JOB_teacher_supcon5}:${JOB_student_supcon5}' in rows[2]['command']
    assert all('--partition=debug' in r['command'] and '--gres=gpu:a100:1' in r['command'] for r in rows)
    a = c.artifact('ACCEPTANCE', minutes={t: 90 for t in c.FITS})
    assert submission.within_envelope(submission.render(s, 'envelope'), submission.render(s, 'science', acceptance=a))
    assert len(submission.render(s, 'setup')['commands']) == 1


@pytest.mark.parametrize('task', c.FITS)
def test_isolated_heads_native_initialization_and_no_target_inference(tmp_path, monkeypatch, task):
    tiny(monkeypatch); s = spec(tmp_path); m = model.new_model(s, task).eval()
    rng = torch.random.get_rng_state().clone()
    torch.manual_seed(23); native = DelphesParticleTransformer().eval()
    assert torch.equal(rng, torch.random.get_rng_state())
    for k, v in native.state_dict().items():
        torch.testing.assert_close(m.native_state()[k], v, rtol=0, atol=0)
    raw = cache().batch(np.arange(4))
    with torch.no_grad():
        expected = _forward(native, raw, 'cpu', bf16=False)
        m.capture = True; actual = _forward(m, raw, 'cpu', bf16=False)
        m.capture = False; deployed = _forward(m, raw, 'cpu', bf16=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(deployed, expected, rtol=0, atol=0)
    assert all(k.startswith('mod.') for k in m.native_state())


def test_teacher_exact_loss_and_fiftyfold_old_weight_gradient(tmp_path, monkeypatch):
    tiny(monkeypatch); s = spec(tmp_path); m = model.new_model(s, 'teacher_supcon5'); m.capture = True
    raw = cache(coordinate='OFFLINE').batch(np.arange(22)); y = torch.from_numpy(raw['labels'])
    logits = _forward(m, raw, 'cpu', bf16=False); total, terms = model.objective(m, 'teacher_supcon5', logits, y)
    reference = torch.nn.functional.cross_entropy(logits.float(), y) + 5*supcon(m.projection, y)
    torch.testing.assert_close(total, reference, rtol=0, atol=0)
    p = next(m.mod.embed.parameters())
    g5 = torch.autograd.grad(terms['supcon'], p, retain_graph=True)[0]
    g01 = torch.autograd.grad(.1*supcon(m.projection, y), p, retain_graph=True)[0]
    torch.testing.assert_close(g5, 50*g01, rtol=1e-4, atol=1e-6)
    diag = training.gradients(terms, m)
    assert diag['weighted_encoder_norms']['supcon'] > 0
    projection = torch.randn(4,128, requires_grad=True)
    zero = supcon(projection, torch.arange(4)); zero.backward()
    assert zero.item() == 0 and projection.grad.abs().sum() == 0


def test_student_original_u100_parity_detached_targets_and_gradient_routes(tmp_path, monkeypatch):
    tiny(monkeypatch); m = model.new_model(spec(tmp_path), 'student_supcon5'); m.eval(); m.capture = True
    torch.manual_seed(23); old = AuxiliaryParT().eval(); old.capture = True
    raw = cache().batch(np.arange(4)); labels = torch.from_numpy(raw['labels'])
    a, b = _forward(m, raw, 'cpu', bf16=False), _forward(old, raw, 'cpu', bf16=False)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(m.prediction, old.prediction, rtol=0, atol=0)
    u, o = (torch.randn(4,128, requires_grad=True) for _ in range(2))
    loss, terms = model.objective(m, 'student_supcon5', a, labels, u, o)
    expected = torch.nn.functional.cross_entropy(b, labels) + 5*torch.nn.functional.smooth_l1_loss(old.prediction, u.detach())
    torch.testing.assert_close(terms['ce']+terms['u100'], expected)
    grad = torch.autograd.grad(terms['offline'], list(m.mod.fc.parameters()), allow_unused=True, retain_graph=True)
    assert all(g is None for g in grad)
    loss.backward()
    assert u.grad is None and o.grad is None
    assert m.offline_predictor[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_full_teacher_to_student_train_export_roundtrip(tmp_path, monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    tiny(monkeypatch); s = spec(tmp_path)
    tr, va = cache(coordinate='OFFLINE'), cache('validation', 'OFFLINE')
    native, teacher_report = training.fit(tmp_path/'teacher', s, 'teacher_supcon5', tr, va, {}, device=device)
    assert len(teacher_report['history']) == 30 and teacher_report['oracle']
    ref = file_ref(tmp_path/'teacher/selected.pt')
    target_impl.export(tmp_path/'targets', native, tr, teacher_ref=ref, device=device)
    h, hv = cache(), cache('validation'); bb = banks(h)
    bb['offline'] = target_impl.Bank(tmp_path/'targets', h, teacher_ref=ref)
    _, report = training.fit(tmp_path/'student', s, 'student_supcon5', h, hv, bb, device=device)
    assert len(report['history']) == 30 and not report['oracle'] and report['native_reload']
    training.validate_fit(tmp_path/'student', s, 'student_supcon5')
    with pytest.raises((ValueError, PermissionError)):
        target_impl.Bank(tmp_path/'targets', hv, teacher_ref=ref)
    data = bytearray((tmp_path/'targets/targets.npz').read_bytes()); data[-1] ^= 1
    (tmp_path/'targets/targets.npz').write_bytes(data)
    with pytest.raises(ValueError):
        target_impl.Bank(tmp_path/'targets', h, teacher_ref=ref)


@pytest.mark.parametrize('violation', ['role', 'coordinate', 'identity', 'population'])
def test_forbidden_roles_and_joins(tmp_path, violation):
    s = spec(tmp_path); tr, va = cache(), cache('validation'); bb = banks(tr)
    if violation == 'role': tr.role = 'validation'
    if violation == 'coordinate': tr.coordinate_name = 'OFFLINE'
    if violation == 'identity': bb['offline'].identities = tr.identities[::-1]
    if violation == 'population': s['counts']['train'] = 33
    with pytest.raises((ValueError, PermissionError)):
        training.check_inputs(s, 'student_supcon5', tr, va, bb, False)


def diagnostic(s):
    return c.artifact('RESOURCES', parents=dict(spec=s['content_hash']), status='complete', error=None,
        environment=s['reference']['environment'], gpu=dict(name='NVIDIA A100', total_memory_bytes=40*2**30),
        memory=dict(peak_rss_bytes=4*2**30, gpu_peak_bytes=5*2**30), profiles={t: dict(
            initial_native_fp32_parity=True, miniature_complete=True, deployment_reload=True,
            largest_support_backward=True, partial_batch_backward=True, step_samples=[.1]*12,
            step_seconds=float(np.mean([.1]*12)), train_rows=s['counts']['train'], validation_rows=s['counts']['validation'],
            validation_seconds=1., probe_seconds=.2, load_seconds=2., export_seconds=1.) for t in c.FITS})


@pytest.mark.parametrize('bad', ['runtime', 'memory', 'incomplete', 'environment', 'nan'])
def test_resource_admission_fails_for_execution_not_accuracy(tmp_path, bad):
    s = spec(tmp_path); d = diagnostic(s)
    assert set(resources.accept(s, d)['minutes']) == set(c.FITS)
    if bad == 'runtime': s['fit_max_minutes'] = 1
    if bad == 'memory': d['memory']['gpu_peak_bytes'] = 40*2**30
    if bad == 'incomplete': d['profiles']['teacher_supcon5']['deployment_reload'] = False
    if bad == 'environment': d['environment'] = {}
    if bad == 'nan': d['profiles']['teacher_supcon5']['step_samples'][0] = float('inf')
    if bad != 'nan': d = with_content_hash({k:v for k,v in d.items() if k != 'content_hash'})
    with pytest.raises((ValueError, MemoryError, RuntimeError)):
        resources.accept(s, d)


def test_successful_setup_dispatches_only_two_fits_and_failure_stops(tmp_path, monkeypatch):
    s = spec(tmp_path); (tmp_path/'submission').mkdir()
    accepted = c.artifact('ACCEPTANCE', minutes={t:100 for t in c.FITS})
    monkeypatch.setattr(campaign, 'check', lambda *a, **k:None)
    monkeypatch.setattr(submission, 'check_site', lambda _:None)
    monkeypatch.setattr(resources, 'accepted', lambda _:accepted)
    records = []
    monkeypatch.setattr(submission, 'submit', lambda spec, plan, execute: records.append((plan['stage'], execute)) or plan)
    monkeypatch.setattr(submission, 'accepted_jobs', lambda *a:{'setup':'123'})
    monkeypatch.setattr(submission, 'accounting', lambda _:['COMPLETED','0:0'])
    monkeypatch.setattr(worker, 'completed_setup', lambda _:None)
    value = submission.review(s)
    submission.dispatch(s, plan_hash=value['content_hash'], authorization=c.AUTHORIZATION)
    assert ('science', True) in records
    with pytest.raises(FileExistsError):
        submission.dispatch(s, plan_hash=value['content_hash'], authorization=c.AUTHORIZATION)
    other = tmp_path/'failed'; other.mkdir(); s2 = dict(s, root=str(other)); records.clear()
    value = submission.review(s2)
    monkeypatch.setattr(submission, 'accounting', lambda _:['FAILED','1:0'])
    with pytest.raises(RuntimeError, match='science not released'):
        submission.dispatch(s2, plan_hash=value['content_hash'], authorization=c.AUTHORIZATION)
    assert ('science', True) not in records


def test_dry_plan_required_and_partial_claim_never_retried(tmp_path, monkeypatch):
    s = spec(tmp_path); plan = submission.render(s, 'setup')
    with pytest.raises(ValueError, match='dry review'):
        submission.submit(s, plan, execute=True)
    def boundary(**kw):
        if kw['execute']: raise RuntimeError('ambiguous scheduler acceptance')
        return {}
    monkeypatch.setattr(submission, 'submit_exact_dag', boundary)
    submission.submit(s, plan, execute=False)
    with pytest.raises(RuntimeError): submission.submit(s, plan, execute=True)
    with pytest.raises(FileExistsError): submission.submit(s, plan, execute=True)


def test_real_exact_journals_resolve_dependencies_and_prevent_duplicates(tmp_path, monkeypatch):
    s = spec(tmp_path); a = c.artifact('ACCEPTANCE', minutes={t:90 for t in c.FITS})
    monkeypatch.setattr(resources, 'accepted', lambda _:a)
    plan = submission.render(s, 'science'); calls = []
    submission.submit(s, plan, execute=False)
    def run(command, **kwargs):
        assert command[0] == 'sbatch' and kwargs['env'] is not None
        calls.append(command)
        return SimpleNamespace(stdout=str(400+len(calls))+'\n')
    monkeypatch.setattr(subprocess, 'run', run)
    submission.submit(s, plan, execute=True)
    assert submission.accepted_jobs(s, 'science') == dict(teacher_supcon5='401', student_supcon5='402', report='403')
    assert '--dependency=afterok:401' in calls[1]
    assert '--dependency=afterany:401:402' in calls[2]
    submission.submit(s, plan, execute=True)
    assert len(calls) == 3
    ledger = tmp_path/'submission_science/submission_ledger.json'
    tampered = load_json(ledger); tampered['jobs']['student_supcon5'] = '999'
    ledger.write_text(__import__('json').dumps(tampered))
    with pytest.raises(ValueError): submission.accepted_jobs(s, 'science')


@pytest.mark.parametrize('task', c.FITS)
def test_full_native_physical_batch_partial_update_and_export(tmp_path, task):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    s = spec(tmp_path); tr = cache(coordinate='OFFLINE' if task in c.TEACHERS else 'D000')
    bb = {} if task in c.TEACHERS else banks(tr)
    m = model.new_model(s, task).cuda(); opt = training.optimizer(m)
    assert len(m.mod.blocks) == 8 and len(m.mod.cls_blocks) == 2
    for size in (128, 11):
        ix = np.arange(size)%len(tr)
        values, _ = training.step(m, task, opt, tr.batch(ix), bb, ix, 1., 'cuda')
        assert all(np.isfinite(v) for v in values.values())
    m.eval(); m.capture = False
    native = DelphesParticleTransformer().cuda().eval(); native.load_state_dict(m.native_state(), strict=True)
    with torch.no_grad():
        expected = _forward(m, tr.batch(np.arange(11)), 'cuda', bf16=False)
        actual = _forward(native, tr.batch(np.arange(11)), 'cuda', bf16=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_production_teacher_student_report_receipts_and_epoch_readout(tmp_path, monkeypatch, capsys):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    tiny(monkeypatch); s = spec(tmp_path); h, hv = cache(), cache('validation')
    bb = banks(h); u = tmp_path/'old_u100.json'; write_immutable_json(u, bb['u100'].manifest)
    s['reference']['u100_manifest'] = file_ref(u)
    p = np.full((22,11), .01, np.float32); p[np.arange(22), hv.labels] = .9
    path = tmp_path/'old_predictions.npz'
    atomic_publish_bytes(path, deterministic_npz_bytes(dict(identities=hv.identities, labels=hv.labels, probabilities=p)))
    s['reference']['files'] = dict(predictions=file_ref(path))
    s['anchor']['evidence']['models'] = {k:dict(predictions=file_ref(path)) for k in ('M0HLT','OFFLINE')}
    subset = tmp_path/'subset.npz'
    atomic_publish_bytes(subset, deterministic_npz_bytes(dict(indices=np.arange(22, dtype=np.int64))))
    s['imported'] = dict(files=dict(subset=file_ref(subset))); s['original50k_rows'] = 22
    gpu = dict(name='NVIDIA A100', total_memory_bytes=40*2**30)
    job = ['401']
    monkeypatch.setattr(campaign, 'check', lambda *a, **k:None)
    monkeypatch.setattr(worker, 'require_backend', lambda:None)
    monkeypatch.setattr(worker, 'allocation', lambda _: (job[0],s['cpus'],s['memory_mb']))
    monkeypatch.setattr(worker, 'installed_environment', lambda:s['reference']['environment'])
    monkeypatch.setattr(worker, 'gpu_identity', lambda:gpu)
    monkeypatch.setattr(worker, 'memory_snapshot', lambda _:dict(peak_rss_bytes=2**30, gpu_peak_bytes=2**30))
    monkeypatch.setattr(worker, 'completed_setup', lambda _:None)
    monkeypatch.setattr(resources, 'accepted', lambda _:dict(gpu=gpu))
    monkeypatch.setattr(submission, 'accepted_jobs', lambda *a:dict(teacher_supcon5='401',student_supcon5='402',report='403'))
    monkeypatch.setattr(inputs, 'views', lambda _, teacher: (cache(coordinate='OFFLINE'),cache('validation','OFFLINE')) if teacher else (h,hv))
    monkeypatch.setattr(inputs, 'embedding_bank', lambda *a:bb['u100'])
    worker.run(s, 'teacher_supcon5'); job[0]='402'; worker.run(s, 'student_supcon5')
    worker.epochs(s); job[0]='403'; worker.run(s, 'report')
    result = load_json(tmp_path/'report/results.json')
    assert not result['pending'] and len(result['rows']) == 5
    assert {r['model'] for r in result['rows'] if r['oracle']} == {'OFFLINE','teacher_supcon5'}
    assert 'student_supcon5 epoch=30' in capsys.readouterr().out
    with pytest.raises(FileExistsError): worker.run(s, 'report')
    teacher_receipt = tmp_path/'teacher_supcon5/receipt.json'
    value = load_json(teacher_receipt); value['job_id']='wrong'
    teacher_receipt.write_text(__import__('json').dumps(with_content_hash(value)))
    with pytest.raises(ValueError, match='ownership'): worker.validate_teacher(s, 'teacher_supcon5')


def test_poor_finite_accuracy_completes_all_epochs(tmp_path, monkeypatch):
    tiny(monkeypatch); s = spec(tmp_path); evaluate = training.evaluate_probabilities
    def poor(y,p):
        value = evaluate(y,p); value['macro_ovr_auc'] = 0.; return value
    monkeypatch.setattr(training, 'evaluate_probabilities', poor)
    _, report = training.fit(tmp_path/'poor', s, 'teacher_supcon5', cache(coordinate='OFFLINE'),
        cache('validation','OFFLINE'), {}, device='cpu')
    assert report['epochs'] == 30 and report['validation']['macro_ovr_auc'] == 0.


def test_current_source_rejects_changes_to_donor_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, '_source', lambda *a:dict(content_hash='a'*64))
    monkeypatch.setattr(campaign, '__file__', str(tmp_path/'src/hlt_classification/luka_supcon5/campaign.py'))
    change = ['A\tsrc/hlt_classification/luka_supcon5/model.py']
    def git(args, **kwargs):
        return SimpleNamespace(stdout='\n'.join(campaign.REQUIRED) if 'ls-files' in args else '\n'.join(change))
    monkeypatch.setattr(campaign.subprocess, 'run', git)
    campaign.current(tmp_path, 'a'*40)
    change[0] = 'M\tsrc/hlt_classification/luka_embedding_aux/model.py'
    with pytest.raises(ValueError, match='Unexpected runtime'): campaign.current(tmp_path, 'a'*40)


def test_setup_measurement_pipeline_and_failure_diagnostic(tmp_path, monkeypatch):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    from test_luka_attention import cache as larger_cache
    tiny(monkeypatch); s = spec(tmp_path); s['counts']['train'] = 1793
    (tmp_path/'setup').mkdir()
    def views(_, teacher):
        coordinate = 'OFFLINE' if teacher else 'D000'
        return larger_cache('train',coordinate,1793), cache('validation',coordinate)
    monkeypatch.setattr(inputs, 'views', views)
    monkeypatch.setattr(inputs, 'embedding_bank', lambda _, tr:banks(tr)['u100'])
    monkeypatch.setattr(resources, 'installed_environment', lambda:s['reference']['environment'])
    monkeypatch.setattr(resources, 'gpu_identity', lambda:dict(name='NVIDIA A100',total_memory_bytes=40*2**30))
    monkeypatch.setattr(resources, 'memory_snapshot', lambda _:dict(peak_rss_bytes=2**30,gpu_peak_bytes=2**30))
    a = resources.measure(s)
    assert a == resources.accepted(s) and set(a['minutes']) == set(c.FITS)
    for task in c.FITS:
        training.validate_fit(tmp_path/'setup'/('mini_'+task), s, task, technical=True)
    other = tmp_path/'failed'; (other/'setup').mkdir(parents=True); s2 = dict(s,root=str(other))
    def failure(*a, **k): raise RuntimeError('injected measurement error')
    monkeypatch.setattr(training, 'step', failure)
    with pytest.raises(RuntimeError, match='injected'): resources.measure(s2)
    d = load_json(other/'setup/resource_diagnostic.json')
    assert d['status'] == 'failed' and 'injected' in d['error'] and not (other/'setup/acceptance.json').exists()

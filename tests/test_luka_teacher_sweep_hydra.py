"""Teacher semantics, HLT-only transfer, measured admission and durable exact DAG."""
from pathlib import Path
from types import SimpleNamespace
import copy
import subprocess
import numpy as np
import pytest
import torch
from test_luka_u100_pretrain import cache
from test_luka_embedding_aux import tiny
from test_luka_u100_pair_correction import dataset, prepared, all_prepared
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import _forward
from hlt_classification.luka_embedding_aux import targets as bank_impl
from hlt_classification.luka_supcon5 import model as original_model
from hlt_classification.luka_teacher_sweep_hydra import (
    contracts as c, model, targets, training, resources, campaign, submission, worker, transfer, inputs, reporting)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def spec(root):
    return dict(campaign.registration(), root=str(root), project_dir='/project', content_hash='c'*64,
        anchor=dict(evidence=dict(prepared='a'*64)), counts=dict(train=22, validation=22),
        nodes={t: dict(initialization_seed=23, sampler_seed=29) for t in c.FITS},
        recovery_anchors=dict(M0HLT=.95, OFFLINE=.98), reference=dict(environment={'fixture': True}),
        offline_hash='d'*64, bundle_root='/bundle', offline_root='/offline', environment_dir='/env')


def stats(s):
    (Path(s['root'])/'setup').mkdir(exist_ok=True)
    return targets.build(s, cache(coordinate='OFFLINE'))


def banks(train):
    rng = np.random.default_rng(8)
    return {k: SimpleNamespace(targets=rng.normal(size=(len(train), 128)).astype(np.float32),
        identities=train.identities, labels=train.labels, manifest=c.artifact('TEST_BANK', role='train', name=k))
        for k in ('u100', 'offline')}


def diagnostic(s):
    return c.artifact('RESOURCES', parents=dict(spec=s['content_hash']), status='complete', error=None,
        environment=s['reference']['environment'], gpu=dict(name='NVIDIA RTX A6000', compute_capability=[8, 6], total_memory_bytes=48*2**30),
        memory=dict(peak_rss_bytes=4*2**30, gpu_peak_bytes=5*2**30), profiles={t: dict(
            initial_native_fp32_parity=True, miniature_complete=True, deployment_reload=True,
            largest_support_backward=True, partial_batch_backward=True, train_rows=s['counts']['train'],
            validation_rows=s['counts']['validation'], step_samples=[.3]*12, step_seconds=float(np.mean([.3]*12)),
            validation_seconds=1., probe_seconds=1., load_seconds=1., export_seconds=1.) for t in resources.PROFILE_TASKS})


def reseal(d):
    return with_content_hash({k: v for k, v in d.items() if k != 'content_hash'})


def test_twentyfour_fits_teacher_only_dependencies_and_full_bootstrap_review(tmp_path):
    s = spec(tmp_path); boot = submission.plan(s, 'bootstrap'); env = submission.plan(s, 'envelope')
    assert len(c.TEACHERS) == len(c.STUDENTS) == 12 and len(set(c.NAMES)) == 12
    assert [r['task_id'] for r in env['commands']] == [*c.FITS, 'report']
    assert boot['upper_bound_commands'] == env['commands']
    assert [r['task_id'] for r in boot['commands']] == ['setup', 'release']
    assert '--dependency=afterok:${JOB_setup}' in boot['commands'][1]['command']
    assert not any(x.startswith('--gres') for x in boot['commands'][1]['command'])
    for row in env['commands'][:-1]:
        t = row['task_id']; assert row['dependencies'] == ([c.STUDENTS[t]] if t in c.STUDENTS else [])
        assert '--gres=gpu:nvidia_rtx_a6000:1' in row['command']
    assert env['commands'][-1]['dependencies'] == list(c.FITS)
    assert any(x.startswith('--dependency=afterany:') for x in env['commands'][-1]['command'])
    accepted = resources.accept(s, diagnostic(s)); measured = submission.plan(s, 'science', accepted)
    assert submission.within_envelope(env, measured)
    changed = copy.deepcopy(measured); changed['commands'][0]['command'].append('--gres=gpu:2')
    with pytest.raises(PermissionError): submission.within_envelope(env, changed)
    bad = copy.deepcopy(accepted); bad['minutes']['T00'] = 2881
    with pytest.raises(ValueError): submission.plan(s, 'science', bad)


@pytest.mark.parametrize('task', c.TEACHERS+('S00',))
def test_native_initialization_inference_parity_and_objective_routes(tmp_path, monkeypatch, task):
    tiny(monkeypatch); s = spec(tmp_path); stats(s)
    m = model.new_model(s, task).eval(); rng = torch.random.get_rng_state().clone()
    torch.manual_seed(23); native = DelphesParticleTransformer().eval()
    assert torch.equal(rng, torch.random.get_rng_state())
    for k, v in native.state_dict().items(): torch.testing.assert_close(m.native_state()[k], v, rtol=0, atol=0)
    raw = cache().batch(np.arange(22)); y = torch.from_numpy(raw['labels'])
    m.capture = True; a = _forward(m, raw, 'cpu', bf16=False); b = _forward(native, raw, 'cpu', bf16=False)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    u, o = (torch.randn(22, 128, requires_grad=True) for _ in range(2))
    loss, terms = model.objective(m, task, a, y, u, o, seed=7) if task in c.STUDENTS else model.objective(m, task, a, y, seed=7)
    assert set(terms) == c.term_names(task) and torch.isfinite(loss)
    diag = training.gradients(terms, m)
    assert diag['weighted_encoder_norms']['ce'] > 0
    for key, term in terms.items():
        if key == 'ce': continue
        g = torch.autograd.grad(term, list(m.mod.fc.parameters()), retain_graph=True, allow_unused=True)
        assert all(v is None for v in g), key
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert u.grad is o.grad is None
    m.capture = False
    with torch.no_grad():
        torch.testing.assert_close(_forward(m, raw, 'cpu', bf16=False), _forward(native, raw, 'cpu', bf16=False), rtol=0, atol=0)


def test_supcon5_exact_sporc_formula_and_20_gradient_ratio(tmp_path, monkeypatch):
    tiny(monkeypatch); s = spec(tmp_path)
    m = model.new_model(s, 'T01'); m.capture = True
    torch.manual_seed(23); old = original_model.Teacher(); old.capture = True
    raw = cache().batch(np.arange(22)); y = torch.from_numpy(raw['labels'])
    rng = torch.random.get_rng_state().clone()
    a = _forward(m, raw, 'cpu', bf16=False)
    torch.random.set_rng_state(rng)
    b = _forward(old, raw, 'cpu', bf16=False)
    actual, terms = model.objective(m, 'T01', a, y)
    expected, _ = original_model.objective(old, 'teacher_supcon5', b, y)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _, stronger = model.objective(m, 'T02', a, y)
    torch.testing.assert_close(stronger['supcon'], 4*terms['supcon'], rtol=0, atol=0)


def test_thinning_is_seeded_zeroed_and_preserves_at_least_one():
    raw = {k: torch.from_numpy(v) for k, v in cache().batch(np.arange(4)).items() if k in ('features','vectors','mask')}
    raw['mask'][0, :, 1:] = False
    before = torch.random.get_rng_state().clone()
    a, removed = model.thin(raw, torch.Generator().manual_seed(1), .4, removed_target=True)
    b, _ = model.thin(raw, torch.Generator().manual_seed(1), .4, removed_target=True)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(a['mask'], b['mask']) and a['mask'].sum(-1).min() >= 1
    assert not removed[0].any() and removed[1:].any(1).all()
    assert torch.count_nonzero(a['features']*~a['mask']) == 0
    assert torch.equal(a['mask'][:,0] | removed, raw['mask'][:,0])


def test_set_reconstruction_permutation_and_gradient():
    torch.manual_seed(7); out = torch.randn(2, 5, 22, requires_grad=True)
    counts = torch.randn(2, 6, requires_grad=True); y = torch.randn(2, 7, 21)
    mask = torch.tensor([[1,1,1,0,0,0,0],[0,0,0,0,0,0,0]], dtype=torch.bool)
    a = model.set_loss(out, counts, y, mask)
    order = torch.tensor([3,1,6,2,0,5,4]); b = model.set_loss(out, counts, y[:,order], mask[:,order])
    torch.testing.assert_close(a, b); a.backward()
    assert out.grad.abs().sum() > 0 and counts.grad.abs().sum() > 0
    with pytest.raises(ValueError): model.set_loss(out, counts, y, torch.ones_like(mask))


def test_statistics_train_only_summary_permutation_and_corruption(tmp_path):
    s = spec(tmp_path); value = stats(s); tr = cache(coordinate='OFFLINE')
    raw = tr.batch(np.arange(22)); p = np.arange(raw['features'].shape[-1])[::-1]
    shuffled = {k: v[...,p] if k in ('features','vectors','mask') else v for k,v in raw.items()}
    np.testing.assert_allclose(targets.summaries(raw), targets.summaries(shuffled), rtol=1e-6, atol=1e-6)
    assert value['rows'] == 22 and len(value['moments']['summary']['mean']) == 36
    with pytest.raises(PermissionError): targets.build(s, cache('validation', 'OFFLINE'))
    path = tmp_path/'setup/statistics.json'; bad = copy.deepcopy(value); bad['moments']['particle']['scale'][0] = 0
    path.write_text(__import__('json').dumps(reseal(bad)))
    with pytest.raises(ValueError): targets.load(s)


@pytest.mark.parametrize('task', c.TEACHERS)
def test_real_fit_two_epochs_native_reload_and_train_target_bank(tmp_path, monkeypatch, task):
    tiny(monkeypatch); s = spec(tmp_path); stats(s)
    tr, va = cache(coordinate='OFFLINE'), cache('validation', 'OFFLINE')
    native, r = training.fit(tmp_path/task, s, task, tr, va, {}, device='cpu', technical=True)
    assert r['epochs'] == 2 and r['oracle'] and r['native_reload']
    ref = file_ref(tmp_path/task/'selected.pt')
    bank_impl.export(tmp_path/'bank', native, tr, teacher_ref=ref, device='cpu')
    b = bank_impl.Bank(tmp_path/'bank', cache(), teacher_ref=ref)
    assert b.targets.shape == (22,128) and np.isfinite(b.targets).all()
    training.validate_fit(tmp_path/task, s, task, technical=True)
    with pytest.raises(ValueError): bank_impl.Bank(tmp_path/'bank', cache('validation'), teacher_ref=ref)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_full_thirty_teacher_bank_student_pipeline(tmp_path, monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    tiny(monkeypatch); s = spec(tmp_path); stats(s)
    tr, va = cache(coordinate='OFFLINE'), cache('validation','OFFLINE')
    native, r = training.fit(tmp_path/'teacher', s, 'T01', tr, va, {}, device=device)
    assert len(r['history']) == 30
    ref = file_ref(tmp_path/'teacher/selected.pt'); bank_impl.export(tmp_path/'bank', native, tr, teacher_ref=ref, device=device)
    h, hv = cache(), cache('validation'); bb = banks(h); bb['offline'] = bank_impl.Bank(tmp_path/'bank', h, teacher_ref=ref)
    _, student = training.fit(tmp_path/'student', s, 'S01', h, hv, bb, device=device)
    assert len(student['history']) == 30 and not student['oracle']
    training.validate_fit(tmp_path/'student', s, 'S01')
    bad = bytearray((tmp_path/'bank/targets.npz').read_bytes()); bad[-1] ^= 1; (tmp_path/'bank/targets.npz').write_bytes(bad)
    with pytest.raises(ValueError): bank_impl.Bank(tmp_path/'bank', h, teacher_ref=ref)


@pytest.mark.parametrize('bad', ['memory','time','missing','gpu','nonfinite'])
def test_resource_failures_are_technical_not_quality(tmp_path, bad):
    s = spec(tmp_path); d = diagnostic(s)
    assert resources.accept(s, d)['minutes'].keys() == set(c.FITS)
    if bad == 'memory': d['memory']['peak_rss_bytes'] = 32*2**30
    elif bad == 'time':
        d['profiles']['T08']['step_samples'] = [1e6]*12; d['profiles']['T08']['step_seconds'] = 1e6
    elif bad == 'missing': del d['profiles']['T03']
    elif bad == 'gpu': d['gpu']['name'] = 'NVIDIA L40'
    else: d['profiles']['T00']['step_samples'][0] = -1
    with pytest.raises((ValueError, MemoryError, RuntimeError)): resources.accept(s, reseal(d))


def fake_scheduler(monkeypatch):
    calls = []; reviewed = []
    monkeypatch.setattr(campaign, 'check', lambda *a, **k: None)
    monkeypatch.setattr(submission, 'scheduler_review', lambda p: reviewed.append(p))
    import hlt_classification.scouting.hcwdl_exact_dag_submission as dag
    def run(command, **kwargs):
        calls.append(command); return SimpleNamespace(stdout=str(8000+len(calls))+'\n', returncode=0)
    monkeypatch.setattr(dag.subprocess, 'run', run)
    return calls, reviewed


def test_durable_release_exact_ids_logout_and_idempotent_completed_submission(tmp_path, monkeypatch):
    s = spec(tmp_path); stats(s); a = resources.accept(s, diagnostic(s))
    write_immutable_json(tmp_path/'setup/acceptance.json', a)
    calls, reviewed = fake_scheduler(monkeypatch)
    submission.launch(s); p = submission.plan(s, 'bootstrap')
    with pytest.raises(PermissionError): submission.launch(s, execute=True, plan_hash='wrong', authorization=c.AUTHORIZATION)
    ledger = submission.launch(s, execute=True, plan_hash=p['content_hash'], authorization=c.AUTHORIZATION)
    assert ledger['jobs'] == {'setup':'8001', 'release':'8002'}
    assert '--dependency=afterok:8001' in calls[1]
    monkeypatch.setattr(submission, 'accounting', lambda job: ['COMPLETED','0:0'])
    monkeypatch.setattr(worker, 'completed_setup', lambda spec: a)
    result = submission.finish(s, '8002')
    assert len(result['jobs']) == 25 and len(calls) == 27
    for student, teacher in c.STUDENTS.items():
        cmd = result['commands'][student]
        assert '--dependency=afterok:'+result['jobs'][teacher] in cmd
    assert submission.finish(s, '8002') == result and len(calls) == 27
    assert [x['stage'] for x in reviewed] == ['bootstrap','envelope','science']


def test_ambiguous_submission_does_not_retry(tmp_path, monkeypatch):
    s = spec(tmp_path); calls, _ = fake_scheduler(monkeypatch)
    submission.launch(s); folder = tmp_path/'submission_bootstrap'; (folder/'live_claim').mkdir()
    p = submission.plan(s, 'bootstrap')
    with pytest.raises(RuntimeError, match='ambiguous'):
        submission.launch(s, execute=True, plan_hash=p['content_hash'], authorization=c.AUTHORIZATION)
    assert not calls


def test_failed_setup_does_not_release_science(tmp_path, monkeypatch):
    s = spec(tmp_path); calls, _ = fake_scheduler(monkeypatch)
    submission.launch(s); p = submission.plan(s, 'bootstrap')
    submission.launch(s, execute=True, plan_hash=p['content_hash'], authorization=c.AUTHORIZATION)
    monkeypatch.setattr(submission, 'accounting', lambda job: ['FAILED','1:0'])
    with pytest.raises(RuntimeError, match='setup failed'): submission.finish(s, '8002')
    assert len(calls) == 2 and not (tmp_path/'submission_science').exists()


def test_runtime_has_no_historical_edits_and_scripts_parse():
    root = Path(__file__).resolve().parents[1]
    assert all((root/p).is_file() for p in campaign.RUNTIME)
    for p in campaign.RUNTIME:
        text = (root/p).read_text()
        assert 'Fresh_check' not in text
        if p.endswith('.py'): compile(text, p, 'exec')
    bash = Path('C:/Program Files/Git/bin/bash.exe')
    if bash.exists():
        for p in ('scripts/queue_luka_teacher_sweep_hydra.sh', 'sbatch/run_luka_teacher_sweep_hydra.sh'):
            subprocess.run([str(bash), '-n', p], check=True)


@pytest.mark.parametrize('task', c.TEACHERS+('S00',))
def test_full_native_batch128_partial_update_and_clean_reload_cuda(tmp_path, task):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    s = spec(tmp_path); stats(s); tr = cache(coordinate='OFFLINE' if task in c.TEACHERS else 'D000')
    bb = {} if task in c.TEACHERS else banks(tr)
    m = model.new_model(s, task).cuda(); opt = training.optimizer(m)
    assert len(m.mod.blocks) == 8 and len(m.mod.cls_blocks) == 2
    for size in (128, 11):
        ix = np.arange(size)%len(tr)
        values, _ = training.step(m, task, opt, tr.batch(ix), bb, ix, 1., 'cuda', update_seed=size)
        assert set(values) == c.term_names(task) and np.isfinite(list(values.values())).all()
    m.eval(); m.capture = False; native = DelphesParticleTransformer().cuda().eval()
    native.load_state_dict(m.native_state(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(_forward(native, tr.batch(np.arange(11)), 'cuda', bf16=False),
            _forward(m, tr.batch(np.arange(11)), 'cuda', bf16=False), rtol=0, atol=0)


def test_decoder_full298_slots_with_physical128_batch_cuda():
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(17); decoder = model.SetDecoder(298).cuda()
    h = torch.randn(128, 128, device='cuda', requires_grad=True)
    target = torch.randn(128, 298, 21, device='cuda')
    mask = torch.arange(298, device='cuda')[None] < torch.arange(1,129, device='cuda')[:,None]
    out, counts = decoder(h); loss = model.set_loss(out, counts, target, mask); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(h.grad).all() and h.grad.abs().sum() > 0


def test_native_offline_export_relocation_fingerprint_and_corruption(all_prepared, tmp_path, monkeypatch):
    from hlt_classification.luka_fullsim import regular_cache
    gate, m, p, inv, splits = all_prepared; meta = (m,p,inv,splits)
    anchor = dict(prepared_root=str(Path(gate['root'])/'prepared'), raw_container=gate['input_container'], prepared=p['content_hash'], cache=m['content_hash'])
    proof = dict(counts=p['counts'], prepared=p['content_hash'], cache=m['content_hash'], campaign='b'*64)
    original = c.artifact('TEST_ORIGIN', root=str(tmp_path/'old_run'), anchor=dict(evidence=anchor))
    origin_path = tmp_path/'origin.json'; write_immutable_json(origin_path, original)
    monkeypatch.setattr(transfer.old_transfer, 'authenticate', lambda *a: proof)
    monkeypatch.setattr(transfer, 'COUNTS', p['counts'])
    monkeypatch.setattr(transfer.regular_cache, 'metadata', lambda _: meta)
    monkeypatch.setattr(transfer.shutil, 'disk_usage', lambda _: SimpleNamespace(free=100*2**30))
    out = tmp_path/'delta'; result = transfer.export(origin_path, out, dict(content_hash='e'*64))
    assert result['payload_bytes'] == sum(r['bytes'] for role in p['counts'] for r in m['blocks'][role]['OFFLINE'])
    bundle = SimpleNamespace(manifest=dict(origin=proof), json=lambda _:m)
    monkeypatch.setattr(transfer, 'Bundle', lambda _:bundle)
    monkeypatch.setattr(transfer, 'metadata', lambda _:meta)
    delta = transfer.Delta(out, result['manifest_hash'], '/bundle'); delta.verify_all()
    s = dict(counts=p['counts'], memory_mb=32000, bundle_root='/bundle')
    for role in ('train','validation'):
        moved = delta.load(s,role); original = transfer.offline_load(anchor,role,meta,16*2**30)
        np.testing.assert_array_equal(moved.identities, original.identities)
        for a,b in zip(moved.blocks,original.blocks): np.testing.assert_array_equal(a.features,b.features)
    with pytest.raises(PermissionError): delta.load(s,'test')
    with pytest.raises(PermissionError): delta.path('inputs/validation/U100/0000.npz')
    with pytest.raises(ValueError): transfer.Delta(out,'0'*64,'/bundle')
    with pytest.raises(ValueError): transfer.export(origin_path,out,dict(content_hash='e'*64))
    path = out/next(iter(delta.rows)); data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
    with pytest.raises(ValueError): delta.verify_all()


def test_current_source_rejects_modified_donors(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, '_source', lambda *a:dict(content_hash='a'*64))
    monkeypatch.setattr(campaign, '__file__', str(tmp_path/'src/hlt_classification/luka_teacher_sweep_hydra/campaign.py'))
    change = ['A\tsrc/hlt_classification/luka_teacher_sweep_hydra/model.py']
    def git(args, **kwargs):
        return SimpleNamespace(stdout='\n'.join(campaign.REQUIRED) if 'ls-files' in args else '\n'.join(change))
    monkeypatch.setattr(campaign.subprocess, 'run', git)
    campaign.current(tmp_path, 'a'*40)
    change[0] = 'M\tsrc/hlt_classification/luka_supcon5/model.py'
    with pytest.raises(ValueError, match='donor runtime'): campaign.current(tmp_path, 'a'*40)


def test_poor_finite_teacher_still_completes_full30(tmp_path, monkeypatch):
    tiny(monkeypatch); s = spec(tmp_path); evaluate = training.evaluate_probabilities
    def poor(y,p):
        value = evaluate(y,p); value['macro_ovr_auc']=0.; return value
    monkeypatch.setattr(training,'evaluate_probabilities',poor)
    _, r = training.fit(tmp_path/'poor',s,'T05',cache(coordinate='OFFLINE'),cache('validation','OFFLINE'),{},device='cpu')
    assert r['epochs']==30 and r['validation']['macro_ovr_auc']==0.


def test_setup_real_measurement_pipeline_and_persistent_failure(tmp_path, monkeypatch):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    from test_luka_attention import cache as large
    from hlt_classification.data.cache_contracts import deterministic_npz_bytes, atomic_publish_bytes
    tiny(monkeypatch); s = spec(tmp_path); s['counts']['train']=1793
    (tmp_path/'setup').mkdir()
    h,hv = large('train','D000',1793),cache('validation')
    paths = {}
    for name, data in (('targets/u100/targets.npz',dict(identities=h.identities,labels=h.labels)),
            ('reference/matched/predictions.npz',dict(identities=hv.identities,labels=hv.labels))):
        p=tmp_path/(str(len(paths))+'.npz'); atomic_publish_bytes(p,deterministic_npz_bytes(data)); paths[name]=p
    monkeypatch.setattr(inputs,'Bundle',lambda _:SimpleNamespace(path=lambda name:paths[name]))
    monkeypatch.setattr(inputs,'views',lambda _,teacher:(large('train','OFFLINE' if teacher else 'D000',1793),cache('validation','OFFLINE' if teacher else 'D000')))
    monkeypatch.setattr(inputs,'embedding_bank',lambda _,tr:banks(tr)['u100'])
    monkeypatch.setattr(resources,'installed_environment',lambda:s['reference']['environment'])
    gpu=dict(name='NVIDIA RTX A6000',compute_capability=[8,6],total_memory_bytes=48*2**30)
    monkeypatch.setattr(resources,'gpu_identity',lambda:gpu)
    monkeypatch.setattr(resources,'memory_snapshot',lambda _:dict(peak_rss_bytes=2**30,gpu_peak_bytes=2**30))
    a=resources.measure(s)
    assert a==resources.accepted(s) and set(a['minutes'])==set(c.FITS)
    for t in resources.PROFILE_TASKS:
        training.validate_fit(tmp_path/'setup'/('mini_'+t),s,t,technical=True)
    other=tmp_path/'failed'; (other/'setup').mkdir(parents=True); s2=dict(s,root=str(other))
    def fail(*a,**kw): raise RuntimeError('injected measurement error')
    monkeypatch.setattr(training,'step',fail)
    with pytest.raises(RuntimeError,match='injected'): resources.measure(s2)
    d=load_json(other/'setup/resource_diagnostic.json')
    assert d['status']=='failed' and 'injected' in d['error'] and not (other/'setup/acceptance.json').exists()


def test_production_workers_selected_teacher_real_student_and_cpu_report(tmp_path, monkeypatch, capsys):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    from hlt_classification.data.cache_contracts import deterministic_npz_bytes, atomic_publish_bytes
    tiny(monkeypatch); s=spec(tmp_path); h,hv=cache(),cache('validation'); bb=banks(h)
    s['imported']={}; s['control']={}; s['original50k_rows']=22
    paths={}
    for name, strength in (('M0HLT',0.),('OFFLINE',.9),('matched',.2),('C0',.4)):
        p=np.full((22,11),(1-strength)/11,np.float32); p[np.arange(22),hv.labels]+=strength
        path=tmp_path/(name+'.npz')
        atomic_publish_bytes(path,deterministic_npz_bytes(dict(identities=hv.identities,labels=hv.labels,probabilities=p)))
        paths[f'reference/{name}/predictions.npz']=path
    subset=tmp_path/'subset.npz'; atomic_publish_bytes(subset,deterministic_npz_bytes(dict(indices=np.arange(22,dtype=np.int64))))
    paths['reference/historical/subset.npz']=subset
    bundle=SimpleNamespace(path=lambda name:paths[name],json=lambda name:bb['u100'].manifest)
    monkeypatch.setattr(inputs,'Bundle',lambda _:bundle)
    monkeypatch.setattr(inputs,'embedding_bank',lambda *a:bb['u100'])
    monkeypatch.setattr(inputs,'views',lambda _,teacher:(cache(coordinate='OFFLINE'),cache('validation','OFFLINE')) if teacher else (h,hv))
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    job=['901']; gpu=dict(name='NVIDIA RTX A6000',compute_capability=[8,6],total_memory_bytes=48*2**30)
    monkeypatch.setattr(worker,'require_backend',lambda:None)
    monkeypatch.setattr(worker.site,'allocation',lambda _:job[0])
    monkeypatch.setattr(worker.site,'cpu_allocation',lambda _:job[0])
    monkeypatch.setattr(worker,'gpu_identity',lambda:gpu)
    monkeypatch.setattr(worker,'installed_environment',lambda:s['reference']['environment'])
    monkeypatch.setattr(worker,'memory_snapshot',lambda _:dict(peak_rss_bytes=2**30,gpu_peak_bytes=2**30))
    monkeypatch.setattr(worker,'completed_setup',lambda _:dict(gpu=gpu))
    monkeypatch.setattr(resources,'accepted',lambda _:dict(gpu=gpu))
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:dict(T01='901',S01='902',report='903'))
    monkeypatch.setattr(reporting,'states',lambda _:dict(T01=['COMPLETED','0:0'],S01=['COMPLETED','0:0']))
    monkeypatch.setattr(reporting,'reuse_check',lambda *a,**kw:paths['reference/C0/predictions.npz'])
    worker.run(s,'T01'); job[0]='902'; worker.run(s,'S01')
    reporting.progress(s); job[0]='903'; worker.run(s,'report')
    result=load_json(tmp_path/'report/results.json')
    assert len(result['pending'])==22 and len(result['rows'])==12
    assert {r['model'] for r in result['rows'] if r['oracle']}=={'OFFLINE','T01'}
    assert 'epoch 30' in capsys.readouterr().out
    assert load_json(tmp_path/'report/receipt.json')['gpu'] is None
    assert load_json(tmp_path/'S01/startup_memory.json')['real_selected_teacher_targets']
    with pytest.raises(FileExistsError): worker.run(s,'report')
    receipt=tmp_path/'T01/receipt.json'; bad=load_json(receipt); bad['job_id']='999'; receipt.write_text(__import__('json').dumps(reseal(bad)))
    with pytest.raises(ValueError,match='ownership'): worker.validate_teacher(s,'T01')


@pytest.mark.parametrize('violation',['role','coordinate','identity','population','overlap'])
def test_student_input_and_target_firewall(tmp_path,violation):
    s=spec(tmp_path); tr,va=cache(),cache('validation'); bb=banks(tr)
    if violation=='role': tr.role='validation'
    elif violation=='coordinate': tr.coordinate_name='OFFLINE'
    elif violation=='identity': bb['offline'].identities=tr.identities[::-1]
    elif violation=='population': s['counts']['train']=100
    else: va.identities=tr.identities.copy()
    with pytest.raises((ValueError,PermissionError)): training.check_inputs(s,'S00',tr,va,bb,False)

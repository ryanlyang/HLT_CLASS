"""Width-specific deployment, matched objectives, saved ensembles and exact release."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import subprocess
import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score
from test_luka_u100_pretrain import cache
from hlt_classification.data.cache_contracts import (load_json, write_immutable_json, with_content_hash,
    deterministic_npz_bytes, atomic_publish_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.jetclass2_delphes.runner import _forward
from hlt_classification.luka_capacity_hydra import (contracts as c, model, inputs, training,
    resources, submission, campaign, worker, reporting, complementarity as co, reuse)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def spec(root):
    s = dict(campaign.registration(), root=str(root), project_dir='/project', content_hash='c'*64,
        anchor=dict(evidence=dict(prepared='a'*64)), reference=dict(environment={'fixture': True}),
        bundle_root='/bundle', environment_dir='/env', counts=dict(train=22, validation=22),
        parent_hash='d'*64, recovery_anchors=dict(M0HLT=.95, OFFLINE=.98), original50k_rows=22)
    s['nodes'] = campaign.nodes(dict(imported=dict(node=dict(initialization_seed=23, sampler_seed=29))))
    return s


def banks(train, task):
    rng = np.random.default_rng(8)
    return {k: SimpleNamespace(targets=rng.normal(size=(len(train), 128)).astype(np.float32),
        identities=train.identities, labels=train.labels,
        manifest=c.artifact('TEST_BANK', role='train', coordinate='IT' if k == 'it' else 'U100')) for k in c.weights(task)}


def small(monkeypatch):
    original = c.model_config
    def config(width):
        return dict(original(width), num_layers=1, num_cls_layers=1, pair_embed_dims=[8, 8, 8])
    monkeypatch.setattr(model, 'model_config', config)


def diagnostic(s):
    return c.artifact('RESOURCES', parents=dict(spec=s['content_hash']), status='complete', error=None,
        environment=s['reference']['environment'], gpu=dict(name='NVIDIA RTX A6000', compute_capability=[8, 6], total_memory_bytes=48*2**30),
        profiles={t: dict(initial_factory_fp32_parity=True, miniature_complete=True, deployment_reload=True,
            largest_support_backward=True, partial_batch_backward=True, real_targets=True,
            train_rows=s['counts']['train'], validation_rows=s['counts']['validation'], step_samples=[.3]*12,
            step_seconds=float(np.mean([.3]*12)), validation_seconds=1., probe_seconds=1., load_seconds=1.,
            memory=dict(peak_rss_bytes=4*2**30, gpu_peak_bytes=5*2**30)) for t in c.FITS})


def reseal(d):
    return with_content_hash({k: v for k, v in d.items() if k != 'content_hash'})


def test_registered_matrix_and_bounded_dag(tmp_path):
    s = spec(tmp_path); boot = submission.plan(s, 'bootstrap'); env = submission.plan(s, 'envelope')
    assert len(c.FITS) == 12 and len(set(c.FITS)) == 12
    assert {(c.cell(t)['width'], c.cell(t)['objective'], c.cell(t)['seed']) for t in c.FITS} == {
        (w, o, seed) for w in (128,256) for o in ('CE','U','UI') for seed in (0,1)}
    assert boot['upper_bound_commands'] == env['commands']
    assert [r['task_id'] for r in boot['commands']] == ['analysis','setup','release']
    assert '--dependency=afterok:${JOB_setup}' in boot['commands'][2]['command']
    assert not boot['commands'][0]['dependencies']
    for row in env['commands'][:-1]:
        assert not row['dependencies']
        assert not any(x.startswith('--dependency=') for x in row['command'])
        assert '--gres=gpu:nvidia_rtx_a6000:1' in row['command']
        assert '--ntasks=1' in row['command'] and '--no-requeue' in row['command']
    assert env['commands'][-1]['dependencies'] == list(c.FITS)
    assert any(x.startswith('--dependency=afterany:') for x in env['commands'][-1]['command'])
    a = resources.accept(s, diagnostic(s)); measured = submission.plan(s, 'science', a)
    assert submission.within_envelope(env, measured)
    measured['commands'][0]['command'].append('--gres=gpu:2')
    with pytest.raises(PermissionError): submission.within_envelope(env, measured)
    a['minutes'][c.FITS[0]] = 2881
    with pytest.raises(ValueError): submission.plan(s, 'science', a)
    native, wide = c.model_config(128), c.model_config(256)
    assert native['num_layers'] == wide['num_layers'] == 8
    assert {k for k in native if native[k] != wide[k]} == {'embed_dims'}


@pytest.mark.parametrize('task', c.FITS)
def test_initial_parity_head_rng_gradient_routes_and_inference(tmp_path, monkeypatch, task):
    small(monkeypatch); s = spec(tmp_path)
    m = model.new_model(s, task).eval(); after = torch.random.get_rng_state().clone()
    torch.manual_seed(s['nodes'][task]['initialization_seed']); native = model.Native(c.cell(task)['width']).eval()
    assert torch.equal(after, torch.random.get_rng_state())
    for k, v in native.state_dict().items(): torch.testing.assert_close(m.native_state()[k], v, rtol=0, atol=0)
    raw = cache().batch(np.arange(22)); m.capture = True
    logits = _forward(m, raw, 'cpu', bf16=False)
    torch.testing.assert_close(logits, _forward(native, raw, 'cpu', bf16=False), rtol=0, atol=0)
    tt = {k: torch.randn(22, 128, requires_grad=True) for k in c.weights(task)}
    loss, terms = model.objective(m, task, logits, torch.from_numpy(raw['labels']), **tt)
    assert set(terms) == c.term_names(task) and set(m.heads) == set(tt)
    diag = training.gradients(terms, m)
    assert all(v > 0 for v in diag['weighted_encoder_norms'].values())
    for k in tt:
        gs = torch.autograd.grad(terms[k], list(m.mod.fc.parameters()), retain_graph=True, allow_unused=True)
        assert all(g is None for g in gs)
        torch.testing.assert_close(terms[k], c.weights(task)[k]*torch.nn.functional.smooth_l1_loss(m.predictions[k], tt[k].detach()))
    loss.backward(); assert all(v.grad is None for v in tt.values())
    m.capture = False
    with torch.no_grad():
        torch.testing.assert_close(_forward(m, raw, 'cpu', bf16=False), _forward(native, raw, 'cpu', bf16=False), rtol=0, atol=0)
    with pytest.raises(PermissionError): model.objective(m, task, logits, torch.from_numpy(raw['labels']), forbidden=torch.zeros(22,128))


@pytest.mark.parametrize('task', c.FITS)
def test_full_thirty_epoch_width_specific_fit(tmp_path, monkeypatch, task):
    small(monkeypatch); s = spec(tmp_path); tr, va = cache(), cache('validation')
    native, report = training.fit(tmp_path/task, s, task, tr, va, banks(tr, task), device='cpu')
    assert len(report['history']) == 30 and report['native_reload'] and not report['oracle']
    assert native.width == c.cell(task)['width']
    assert all(k.startswith('mod.') for k in torch.load(tmp_path/task/'selected.pt', weights_only=True))
    training.validate_fit(tmp_path/task, s, task)
    with pytest.raises(RuntimeError):
        model.Native(384-native.width).load_state_dict(native.state_dict(), strict=True)
    assert report['history'][0]['gradient_diagnostic']['weighted_encoder_norms']['ce'] > 0
    p = tmp_path/task/'validation_predictions.npz'; p.write_bytes(p.read_bytes()+b'bad')
    with pytest.raises(ValueError): training.validate_fit(tmp_path/task, s, task)


@pytest.mark.parametrize('task', c.FITS[:6])
def test_full_depth_128_and_partial11_cuda(tmp_path, task):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    s = spec(tmp_path); tr = cache(); bb = banks(tr, task)
    m = model.new_model(s, task).cuda(); opt = training.optimizer(m)
    assert len(m.mod.blocks) == 8 and len(m.mod.cls_blocks) == 2
    for size in (128,11):
        ix = np.arange(size)%len(tr)
        values, _ = training.step(m, task, opt, tr.batch(ix), bb, ix, 1., 'cuda')
        assert set(values) == c.term_names(task) and np.isfinite(list(values.values())).all()
    m.eval(); m.capture = False; native = model.Native(c.cell(task)['width']).cuda().eval()
    native.load_state_dict(m.native_state(), strict=True)
    with torch.no_grad(): torch.testing.assert_close(_forward(m,tr.batch(np.arange(11)),'cuda',bf16=True),
        _forward(native,tr.batch(np.arange(11)),'cuda',bf16=True), rtol=0, atol=0)


@pytest.mark.parametrize('task', ('N_CE_s0', 'W_U_s0', 'W_UI_s0'))
def test_cuda_production_miniature(tmp_path, monkeypatch, task):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    small(monkeypatch); s = spec(tmp_path); tr, va = cache(), cache('validation')
    resources.miniature(tmp_path/task, s, task, tr, va, banks(tr, task), 'cuda')
    training.validate_fit(tmp_path/task, s, task, technical=True)


def test_ce_only_never_opens_target_bank(tmp_path, monkeypatch):
    def forbidden(*a, **k): raise AssertionError('Read forbidden target')
    monkeypatch.setattr(inputs, 'embedding_bank', forbidden); monkeypatch.setattr(inputs, 'Bank', forbidden)
    monkeypatch.setattr(inputs, 'validate_file_ref', forbidden); monkeypatch.setattr(inputs, 'Bundle', forbidden)
    s = spec(tmp_path); assert inputs.banks(s, 'W_CE_s0', cache()) == {}
    assert inputs.fit_parents(s, 'W_CE_s0') == dict(spec=s['content_hash'], recipe=s['recipe']['content_hash'])
    training.check_inputs(s, 'W_CE_s0', cache(), cache('validation'), {}, False)
    with pytest.raises(PermissionError): training.check_inputs(s,'N_UI_s0',cache(),cache('validation'),{},False)
    with pytest.raises(PermissionError): training.check_inputs(s,'N_CE_s0',cache(),cache(),{},False)
    val = cache('validation'); val.identities[:] = cache().identities
    with pytest.raises(ValueError, match='overlapping'): training.check_inputs(s,'N_CE_s0',cache(),val,{},False)


@pytest.mark.parametrize('bad', ('memory','time','missing','gpu','negative','targets','samples'))
def test_resource_failure_does_not_use_quality(tmp_path, bad):
    s = spec(tmp_path); d = diagnostic(s); t = c.FITS[0]
    assert resources.accept(s, d)['minutes'].keys() == set(c.FITS)
    if bad == 'memory': d['profiles'][t]['memory']['peak_rss_bytes'] = 32*2**30
    elif bad == 'time': d['profiles'][t].update(step_samples=[1e6]*12, step_seconds=1e6)
    elif bad == 'missing': del d['profiles'][t]
    elif bad == 'gpu': d['gpu']['name'] = 'NVIDIA L40'
    elif bad == 'negative': d['profiles'][t]['load_seconds'] = -1
    elif bad == 'targets': d['profiles'][t]['real_targets'] = False
    else: d['profiles'][t]['step_samples'] = [.3]
    with pytest.raises((ValueError,MemoryError,RuntimeError)): resources.accept(s, reseal(d))


def prediction_fixture():
    labels = np.tile(np.arange(11), 8).astype(np.int64); n = len(labels)
    ids = np.zeros((n,32), np.uint8); ids[:,0] = np.arange(n)
    def arrays(p): return dict(identities=ids.copy(), labels=labels.copy(), probabilities=np.asarray(p,np.float32))
    base = np.ones((n,11),np.float32)/11
    oracle = .1*base + .9*np.eye(11,dtype=np.float32)[labels]
    members = {}
    for j, name in enumerate(c.MEMBERS):
        p = .01*base + .99*np.eye(11,dtype=np.float32)[labels]
        ix = np.arange(n)%3 == j
        p[ix] = .5*base[ix] + .5*np.eye(11,dtype=np.float32)[(labels[ix]+1)%11]
        members[name] = arrays(p)
    return members, dict(M0HLT=arrays(base), OFFLINE=arrays(oracle)), np.arange(n,dtype=np.int64)


def test_identity_join_rejects_duplicates_missing_and_labels():
    members, refs, _ = prediction_fixture(); a = refs['M0HLT']; b = members['S2IT']
    perm = np.random.default_rng(3).permutation(len(a['labels']))
    actual = co.align(a, {k:v[perm] for k,v in b.items()})
    for k in b: np.testing.assert_array_equal(actual[k], b[k])
    for bad in ('duplicate','label','missing'):
        x = copy.deepcopy(b)
        if bad == 'duplicate': x['identities'][0] = x['identities'][1]
        elif bad == 'label': x['labels'][0] = 1
        else: x = {k:v[:-1] for k,v in x.items()}
        with pytest.raises(ValueError): co.align(a, x)


def test_weighted_rank_auc_ties_matches_sklearn():
    rng = np.random.default_rng(8); y = np.tile(np.arange(11),20)
    p = rng.integers(1,5,(len(y),11)).astype(np.float32); p /= p.sum(1,keepdims=True)
    weights = rng.integers(0,5,len(y)); normalized = p.astype(np.float64); normalized /= normalized.sum(1,keepdims=True)
    expected = roc_auc_score(y, normalized, multi_class='ovr', average='macro', sample_weight=weights)
    assert co.RankedAUC(y,p)(weights) == pytest.approx(expected,abs=1e-14)


def test_fixed_ensembles_paired_uncertainty_and_no_quality_gate():
    members, refs, subset = prediction_fixture()
    a = co.compute(members, refs, subset, replicates=12, seed=9)
    assert a == co.compute(members, refs, subset, replicates=12, seed=9)
    rows = a['populations']['validation']['rows']; triple = rows['S2IT_S4_S6']
    assert all(triple['delta_each_member'][k] > 0 for k in c.MEMBERS)
    assert triple['metrics']['macro_ovr_auc'] == 1.
    assert a['bootstrap']['event_independence'] == 'unverified' and not a['fitted_weights']
    identical = {k: members['S2IT'] for k in c.MEMBERS}
    b = co.compute(identical, refs, subset, replicates=8, seed=9)
    for value in b['populations']['validation']['rows']['S2IT_S4_S6']['paired_intervals_vs_each_member'].values():
        assert value['auc_delta_95pct'] == [0.,0.]
    # Finite poor predictions remain valid scientific results.
    bad = {k: refs['M0HLT'] for k in c.MEMBERS}
    assert co.compute(bad, refs, subset, replicates=4)['populations']['validation']['rows']['S4']['recovery'] == 0
    with pytest.raises(ValueError): co.compute(dict(members, OFFLINE=refs['OFFLINE']), refs, subset)


def fake_scheduler(monkeypatch):
    calls = []; reviews = []
    monkeypatch.setattr(campaign, 'check', lambda *a, **k: None)
    monkeypatch.setattr(submission, 'scheduler_review', lambda p: reviews.append(p))
    import hlt_classification.scouting.hcwdl_exact_dag_submission as dag
    def run(command, **kwargs):
        calls.append(command); return SimpleNamespace(stdout=str(8000+len(calls))+'\n', returncode=0)
    monkeypatch.setattr(dag.subprocess, 'run', run)
    return calls, reviews


def test_durable_release_exact_ids_independent_science_and_no_duplicates(tmp_path, monkeypatch):
    s = spec(tmp_path); (tmp_path/'setup').mkdir(); a = resources.accept(s, diagnostic(s))
    write_immutable_json(tmp_path/'setup/acceptance.json', a)
    calls, reviews = fake_scheduler(monkeypatch); submission.launch(s); p = submission.plan(s, 'bootstrap')
    with pytest.raises(PermissionError): submission.launch(s, execute=True, plan_hash='bad', authorization=c.AUTHORIZATION)
    ledger = submission.launch(s, execute=True, plan_hash=p['content_hash'], authorization=c.AUTHORIZATION)
    assert ledger['jobs'] == dict(analysis='8001', setup='8002', release='8003')
    monkeypatch.setattr(submission, 'accounting', lambda job: ['COMPLETED','0:0'])
    monkeypatch.setattr(worker, 'completed_setup', lambda s: a)
    result = submission.finish(s, '8003')
    assert len(result['jobs']) == 13 and len(calls) == 16
    assert all(not any(x.startswith('--dependency=') for x in result['commands'][t]) for t in c.FITS)
    assert submission.finish(s, '8003') == result and len(calls) == 16
    assert [x['stage'] for x in reviews] == ['bootstrap','envelope','science']


def test_failed_setup_and_partial_submission_fail_closed(tmp_path, monkeypatch):
    s = spec(tmp_path); calls, _ = fake_scheduler(monkeypatch); submission.launch(s)
    p = submission.plan(s, 'bootstrap')
    submission.launch(s, execute=True, plan_hash=p['content_hash'], authorization=c.AUTHORIZATION)
    monkeypatch.setattr(submission, 'accounting', lambda job: ['FAILED','1:0'])
    with pytest.raises(RuntimeError, match='setup failed'): submission.finish(s, '8003')
    assert len(calls) == 3 and not (tmp_path/'submission_science').exists()
    other = tmp_path/'other'; other.mkdir(); s2 = spec(other); submission.launch(s2)
    (other/'submission_bootstrap/live_claim').mkdir()
    with pytest.raises(RuntimeError, match='ambiguous'):
        submission.launch(s2, execute=True, plan_hash=submission.plan(s2,'bootstrap')['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 3


def test_matched_capacity_contrasts_and_pending_controls():
    aucs = dict(N_CE_s0=.90, W_CE_s0=.91, N_U_s0=.92, W_U_s0=.95, N_UI_s0=.94, W_UI_s0=.99)
    result = reporting.contrasts(aucs)
    assert result['per_seed']['0']['capacity_interaction_UI_minus_U'] == pytest.approx(.02)
    assert not result['two_seed_descriptive']
    aucs.update({k.replace('_s0','_s1'):v+.001 for k,v in list(aucs.items())})
    assert reporting.contrasts(aucs)['two_seed_descriptive']['wide_minus_native_UI']['mean'] == pytest.approx(.05)
    del aucs['W_CE_s0']
    assert reporting.contrasts(aucs)['per_seed']['0']['W_UI_minus_CE'] is None


def test_runtime_scripts_and_cli_parse():
    root = Path(__file__).resolve().parents[1]
    for p in campaign.RUNTIME:
        source = (root/p).read_text(); assert 'Fresh_check' not in source
        if p.endswith('.py'): compile(source,p,'exec')
    bash = Path('C:/Program Files/Git/bin/bash.exe')
    if bash.exists():
        for p in ('scripts/queue_luka_capacity_hydra.sh','sbatch/run_luka_capacity_hydra.sh'):
            subprocess.run([str(bash),'-n',p],check=True)


def test_native_dual_predictor_exact_donor_initialization_and_update(tmp_path):
    from hlt_classification.luka_selective_teacher_hydra import model as donor, training as donor_training
    s = spec(tmp_path); dspec = dict(s, nodes=dict(S2IT=s['nodes']['N_UI_s0']))
    old = donor.new_model(dspec, 'S2IT'); fresh = model.new_model(s, 'N_UI_s0')
    mapping = lambda k: k.replace('predictor.', 'heads.u100.').replace('extra.it.', 'heads.it.')
    for k, v in old.state_dict().items(): torch.testing.assert_close(v, fresh.state_dict()[mapping(k)], rtol=0, atol=0)
    tr = cache(); bb = banks(tr, 'N_UI_s0'); ix = np.arange(22); raw = tr.batch(ix)
    torch.manual_seed(33)
    v1, _ = donor_training.step(old, 'S2IT', donor_training.optimizer(old), raw, bb, ix, 1., 'cpu')
    torch.manual_seed(33)
    v2, _ = training.step(fresh, 'N_UI_s0', training.optimizer(fresh), raw, bb, ix, 1., 'cpu')
    assert v1 == v2
    for k, v in old.state_dict().items(): torch.testing.assert_close(v, fresh.state_dict()[mapping(k)], rtol=0, atol=0)


def test_reused_selected_it_bank_and_role_corruption_firewall(tmp_path, monkeypatch):
    from hlt_classification.luka_selective_teacher_hydra import targets
    from hlt_classification.context_fusion.worker import save_state
    small(monkeypatch); tr = cache(); s = spec(tmp_path); teacher = model.Native(128)
    checkpoint = tmp_path/'selected.pt'; save_state(checkpoint, teacher.state_dict()); ref = file_ref(checkpoint)
    target_dir = tmp_path/'bank'
    targets.export(target_dir, teacher, cache(coordinate='IT'), teacher_ref=ref, device='cpu', spec_hash=s['parent_hash'])
    task_receipt = tmp_path/'receipt.json'; write_immutable_json(task_receipt, {'test':'placeholder task receipt; parent authentication tested separately'})
    s['reused'] = dict(it_checkpoint=ref, it_bank=file_ref(target_dir/'manifest.json'),
        it_targets=file_ref(target_dir/'targets.npz'), it_receipt=file_ref(task_receipt))
    monkeypatch.setattr(inputs,'embedding_bank',lambda s,train:banks(train,'N_U_s0')['u100'])
    bb = inputs.banks(s,'W_UI_s0',tr)
    training.check_inputs(s,'W_UI_s0',tr,cache('validation'),bb,False)
    assert bb['it'].manifest['parents']['teacher'] == ref['sha256']
    with pytest.raises(ValueError): inputs.banks(s,'W_UI_s0',cache('validation'))
    altered = cache(); altered.identities[:] = altered.identities[::-1]
    with pytest.raises(ValueError): inputs.banks(s,'W_UI_s0',altered)
    p=target_dir/'targets.npz'; p.write_bytes(p.read_bytes()+b'corrupt')
    with pytest.raises(ValueError): inputs.banks(s,'W_UI_s0',tr)


def saved_predictions(tmp_path, monkeypatch):
    members, anchors, ix = prediction_fixture(); s = spec(tmp_path)
    s['counts']['validation'] = len(ix); s['original50k_rows'] = len(ix)
    s['reused'] = {}
    for name,a in {**members, **anchors, 'matched':members['S2IT']}.items():
        p=tmp_path/(name+'.npz'); atomic_publish_bytes(p,deterministic_npz_bytes(a))
        if name in c.MEMBERS: s['reused'][name+'_predictions']=file_ref(p)
    atomic_publish_bytes(tmp_path/'subset.npz',deterministic_npz_bytes(dict(indices=ix)))
    class Bundle:
        def __init__(self, root): pass
        def path(self, key):
            if key=='reference/historical/subset.npz': return tmp_path/'subset.npz'
            return tmp_path/(key.split('/')[1]+'.npz')
    monkeypatch.setattr(inputs,'Bundle',Bundle)
    return s


def test_analysis_cpu_worker_and_read_only_report_artifact(tmp_path, monkeypatch):
    s=saved_predictions(tmp_path,monkeypatch)
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(worker.site,'cpu_allocation',lambda s:'901')
    monkeypatch.setattr(worker,'installed_environment',lambda:s['reference']['environment'])
    monkeypatch.setattr(submission,'accepted_jobs',lambda s,stage:dict(analysis='901'))
    settings=c.recipe(); settings['bootstrap']['replicates']=8
    monkeypatch.setattr(co,'recipe',lambda:settings); monkeypatch.setattr(reporting,'recipe',lambda:settings)
    worker.run(s,'analysis'); worker.receipt(s,'analysis')
    a=reporting.analysis(s)
    assert a['final_test_accessed'] is False and a['prediction_refs']=={k:s['reused'][k+'_predictions'] for k in c.MEMBERS}
    with pytest.raises(FileExistsError): worker.run(s,'analysis')
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:{})
    r=reporting.results(s)
    assert r['pending']==list(c.FITS) and not r['deployments']
    assert all(x['delta_matched_CE'] is None for x in r['rows'])
    assert not reporting.progress(s)['best']


def test_real_cuda_setup_path_all_profiles_with_simulated_site(tmp_path, monkeypatch):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    from test_luka_attention import cache as raw_cache
    from hlt_classification.context_fusion.worker import save_state
    from hlt_classification.jetclass2_delphes.runner import predict
    small(monkeypatch); monkeypatch.setattr(resources,'model_config',model.model_config)
    s=spec(tmp_path); tr=raw_cache(coordinate='D000',rows=1804); va=cache('validation')
    s['counts']['train']=len(tr)
    monkeypatch.setattr(inputs,'views',lambda *a:(tr,va)); monkeypatch.setattr(inputs,'banks',lambda _,t,train:banks(train,t))
    monkeypatch.setattr(inputs,'fit_parents',lambda _,t:dict(spec=s['content_hash'],recipe=s['recipe']['content_hash'],
        **{k:b.manifest['content_hash'] for k,b in banks(tr,t).items()}))
    fixture=tmp_path/'fixture'; fixture.mkdir(); native=model.Native(128).cuda().eval()
    save_state(fixture/'checkpoint.pt',native.state_dict())
    atomic_publish_bytes(fixture/'predictions.npz',deterministic_npz_bytes(dict(identities=va.identities,
        labels=va.labels,probabilities=predict(native,va,device='cuda',batch_size=128))))
    del native
    class Bundle:
        def __init__(self,*a): pass
        def verify_all(self): pass
        def path(self,key): return fixture/key.split('/')[-1]
    monkeypatch.setattr(inputs,'Bundle',Bundle)
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(worker.site,'allocation',lambda s:'902'); monkeypatch.setattr(worker,'require_backend',lambda:None)
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:dict(setup='902'))
    for module in (worker,resources):
        monkeypatch.setattr(module,'installed_environment',lambda:s['reference']['environment'])
        monkeypatch.setattr(module,'gpu_identity',lambda:diagnostic(s)['gpu'])
    # Windows lacks Linux process peak RSS. This site/accounting envelope is simulated;
    # the GPU model, replay, forward/backward, timings and file validators below are real.
    monkeypatch.setattr(resources,'memory_snapshot',lambda _:dict(peak_rss_bytes=4*2**30,gpu_peak_bytes=torch.cuda.max_memory_allocated()))
    worker.run(s,'setup'); a=worker.completed_setup(s)
    assert set(a['minutes'])==set(c.FITS)
    d=load_json(tmp_path/'setup/resource_diagnostic.json')
    assert all(len(p['step_samples'])==12 for p in d['profiles'].values())
    for t in c.FITS:
        assert (tmp_path/'setup'/('mini_'+t)/'deployment.json').is_file()


def test_original_authentication_uses_own_project_and_exact_sentinel(tmp_path, monkeypatch):
    from hlt_classification.luka_selective_teacher_hydra.contracts import artifact
    old=artifact('SPEC',root='/old',project_dir='/old_code'); path=tmp_path/'old.json';write_immutable_json(path,old)
    calls=[]
    def run(command,**kwargs):
        calls.append((command,kwargs));return SimpleNamespace(returncode=0,stdout='CAPACITY_PARENT='+old['content_hash']+'\n',stderr='')
    monkeypatch.setattr(reuse.subprocess,'run',run)
    assert reuse.original(path)==old
    command,kwargs=calls[0]
    assert kwargs['cwd']=='/old_code' and kwargs['env']['PYTHONPATH']==str(Path('/old_code')/'src')
    assert 'worker.validate_teacher' in command[4] and command[-1]=='/old_code'
    monkeypatch.setattr(reuse.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='bad',stderr=''))
    with pytest.raises(ValueError,match='authentication failed'):reuse.original(path)


def test_source_guard_rejects_changes_to_existing_runtime(monkeypatch):
    root=Path(__file__).resolve().parents[1]; monkeypatch.setattr(campaign,'_source',lambda *a:dict(content_hash='a'*64))
    def git(command,**kwargs):
        if 'ls-files' in command:return SimpleNamespace(stdout='\n'.join(campaign.REQUIRED))
        if 'diff' in command:return SimpleNamespace(stdout='A\tsrc/hlt_classification/luka_capacity_hydra/model.py\n')
        return SimpleNamespace(stdout='')
    monkeypatch.setattr(campaign.subprocess,'run',git)
    assert campaign.current(root,'b'*40)['content_hash']=='a'*64
    def bad(command,**kwargs):
        if 'diff' in command:return SimpleNamespace(stdout='M\tsrc/hlt_classification/luka_selective_teacher_hydra/model.py\n')
        return git(command,**kwargs)
    monkeypatch.setattr(campaign.subprocess,'run',bad)
    with pytest.raises(ValueError):campaign.current(root,'b'*40)


def test_immutable_registration_round_trip_and_changed_target_rejected(tmp_path, monkeypatch):
    import sys
    from hlt_classification.luka_selective_teacher_hydra.contracts import artifact
    parent_dir=tmp_path/'parent'; parent_dir.mkdir(); old_code=tmp_path/'old_code'; old_code.mkdir()
    project=tmp_path/'code'; project.mkdir(); bundle=tmp_path/'bundle'; bundle.mkdir()
    old=artifact('SPEC',root=str(parent_dir),project_dir=str(old_code),environment_dir=str(Path(sys.prefix).resolve()),
        bundle_root=str(bundle), reference=dict(environment={'fixture':True}),
        imported=dict(node=dict(initialization_seed=23,sampler_seed=29)),anchor=dict(evidence=dict(prepared='a'*64)),
        embedding_hash='e'*64,qualification={'test':'parent proof'},recovery_anchors=dict(M0HLT=.95,OFFLINE=.98))
    path=parent_dir/'campaign_spec.json'; write_immutable_json(path,old)
    sentinel=parent_dir/'target';sentinel.write_bytes(b'original immutable payload')
    refs=dict(it_checkpoint=file_ref(sentinel))
    monkeypatch.setattr(campaign,'current',lambda *a:dict(content_hash='f'*64))
    monkeypatch.setattr(reuse,'original',lambda p:load_json(p))
    monkeypatch.setattr(reuse,'references',lambda p:dict(it_checkpoint=file_ref(sentinel)))
    monkeypatch.setattr(campaign.site,'environment',lambda e:None)
    monkeypatch.setattr(campaign,'storage',lambda s:None)
    root=tmp_path/'new'
    s=campaign.create(root,path,project,'b'*40,9)
    campaign.check(s)
    assert s['reused']==refs and len(s['nodes'])==12
    for seed in (0,1):
        assert len({n['initialization_seed'] for t,n in s['nodes'].items() if c.cell(t)['seed']==seed})==1
        assert len({n['sampler_seed'] for t,n in s['nodes'].items() if c.cell(t)['seed']==seed})==1
    with pytest.raises(ValueError,match='fresh separate'):campaign.create(root,path,project,'b'*40,9)
    with pytest.raises(ValueError):campaign.create(tmp_path/'another',path,project,'b'*40,8)
    sentinel.write_bytes(b'substitution')
    with pytest.raises(ValueError,match='Immutable'):campaign.check(s)


def test_completed_results_and_progress_authenticate_selected_population(tmp_path, monkeypatch):
    small(monkeypatch); s=saved_predictions(tmp_path,monkeypatch)
    # Use the actual training-validation population for all saved references.
    tr,va=cache(),cache('validation'); s['counts']['validation']=22; s['original50k_rows']=22
    arrays=dict(identities=va.identities,labels=va.labels,probabilities=np.full((22,11),1/11,np.float32))
    for name in (*c.MEMBERS,'M0HLT','OFFLINE','matched'):
        a=copy.deepcopy(arrays)
        if name=='OFFLINE':a['probabilities']=.1*a['probabilities']+.9*np.eye(11,dtype=np.float32)[va.labels]
        atomic_publish_bytes(tmp_path/(name+'_real.npz'),deterministic_npz_bytes(a))
        if name in c.MEMBERS:s['reused'][name+'_predictions']=file_ref(tmp_path/(name+'_real.npz'))
    atomic_publish_bytes(tmp_path/'real_subset.npz',deterministic_npz_bytes(dict(indices=np.arange(22,dtype=np.int64))))
    class Bundle:
        def __init__(self,*a):pass
        def path(self,k):return tmp_path/('real_subset.npz' if 'subset' in k else k.split('/')[1]+'_real.npz')
    monkeypatch.setattr(inputs,'Bundle',Bundle)
    task='N_CE_s0';folder=tmp_path/task;folder.mkdir()
    _,r=training.fit(folder/'fit',s,task,tr,va,{},device='cpu')
    # Outer allocation receipt is tested independently; this verifies real fit files/reporting.
    write_immutable_json(folder/'receipt.json',{'simulated':'allocation receipt'})
    monkeypatch.setattr(worker,'receipt',lambda *a:None)
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:{})
    result=reporting.results(s); progress=reporting.progress(s)
    assert task not in result['pending'] and progress['best'][task]['epoch']==r['selected_epoch']
    selected=[row for row in result['rows'] if row['model']==task]
    assert len(selected)==2 and all(row['delta_matched_CE']==0 and row['delta_matched_U100'] is None for row in selected)
    c.validate(result['deployments'][task]['profile'],'DEPLOYMENT')
    p=folder/'fit/epoch-01.json';p.write_text('{}')
    with pytest.raises(ValueError):reporting.progress(s)


@pytest.mark.parametrize('allocated', ('1','2','3','gpu'))
def test_cpu_allocation_keeps_observed_hydra_rounding(tmp_path, monkeypatch, allocated):
    s=spec(tmp_path)
    for k in ('SLURM_NTASKS','SLURM_CPUS_PER_TASK'):monkeypatch.setenv(k,'1')
    monkeypatch.setenv('SLURM_JOB_ID','901');monkeypatch.setenv('SLURM_MEM_PER_NODE','4000')
    count='1' if allocated=='gpu' else allocated
    gpu=',gres/gpu=1' if allocated=='gpu' else ''
    fields=('JobId=901 JobState=RUNNING Partition=compute Account=default QOS=normal '
        f'NumNodes=1 NumTasks=1 CPUs/Task=1 NumCPUs={count} '
        f'ReqTRES=cpu=1,mem=4000M,node=1 AllocTRES=cpu={count},mem=4000M,node=1{gpu}')
    def run(command,**kwargs):return SimpleNamespace(stdout='ClusterName = hydra\n' if 'config' in command else fields)
    monkeypatch.setattr(subprocess,'run',run)
    if allocated in ('1','2'):assert worker.site.cpu_allocation(s)=='901'
    else:
        with pytest.raises(PermissionError):worker.site.cpu_allocation(s)


def test_batch_accounting_uses_exact_science_ids(tmp_path,monkeypatch):
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:dict(N_CE_s0='901',N_U_s0='902'))
    calls=[]
    def run(command,**kwargs):
        calls.append(command);return SimpleNamespace(stdout='901|COMPLETED|0:0\n999|RUNNING|0:0\n')
    monkeypatch.setattr(subprocess,'run',run)
    assert reporting.states(spec(tmp_path))==dict(N_CE_s0=['COMPLETED','0:0'],N_U_s0=['ACCOUNTING_PENDING',''])
    assert len(calls)==1 and '901,902' in calls[0]

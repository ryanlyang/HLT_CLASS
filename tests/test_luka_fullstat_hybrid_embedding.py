"""Exact hybrid loss, frozen dual-target ancestry and all five production workers."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import os
import re
import shutil
import subprocess
import sys
import numpy as np
import pytest
import torch
from test_luka_embedding_aux import tiny, threads
from test_luka_fullstat_cumulative_embedding import fixture as prior_fixture
from test_luka_fullstat import fullstat, durable
from hlt_classification.data.cache_contracts import (
    load_json, load_npz_arrays, write_immutable_json, with_content_hash)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullsim.regular_campaign import nodes as original_nodes
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer, distillation_loss
from hlt_classification.jetclass2_delphes.runner import _forward, predict
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_fullstat_multiteacher import model as multi_model
from hlt_classification.luka_fullstat_hybrid_embedding import (
    contracts as c, campaign, source, banks, kd_banks, model, training, worker, submission)


def fixture(path, *, export=True):
    old, caches = prior_fixture(path, export=export)
    fields = {k: v for k, v in old['reuse'].items()
              if k not in {'contract', 'schema_version', 'content_hash', 'parents', *c.FLAGS}}
    fields['references'] = dict(fields['references'], DIRECT_D000=old['references']['M0HLT'],
                                COARSE_D000=old['references']['M0HLT'])
    preflight = load_json(fields['original_preflight']['path'])
    preflight['measurements'] = [dict(coordinate=co) for co in ('OFFLINE', *c.STAGES.values())]
    write_immutable_json(path/'hybrid_preflight.json', preflight)
    fields['original_preflight'] = file_ref(path/'hybrid_preflight.json')
    for role in ('train', 'validation'):
        for co in ('OFFLINE', 'U000'):
            fields['summaries'][role][co] = dict(maximum=4, resident_bytes=10000)
            caches[role][co] = deepcopy(caches[role]['D000'])
            caches[role][co].coordinate_name = co
    reuse = c.artifact('REUSE', **fields)
    root = path/'hybrid'; root.mkdir()
    s = c.artifact('SPEC', parents=dict(source='c'*64, reuse=reuse['content_hash']),
        root=str(root), project_dir='/project', source_commit='b'*40, source_snapshot=dict(content_hash='c'*64),
        reuse=reuse, available_result_bytes=16*2**30, **campaign.policy(reuse))
    s = with_content_hash(dict(s, counts=old['counts']))
    write_immutable_json(root/'campaign_spec.json', s)
    return s, caches


def originals(s, caches, *, device='cpu'):
    worker.export_target(s, 'M0HLT', 'embedding', caches['train']['D000'], device=device)
    worker.export_target(s, 'OFFLINE', 'kd', caches['train']['OFFLINE'], device=device)


def mock_execution(s, caches, monkeypatch):
    monkeypatch.setattr(campaign, 'check', lambda _: None)
    monkeypatch.setattr(worker, 'execution', lambda *a: os.environ['SLURM_JOB_ID'])
    monkeypatch.setattr(worker, 'metadata', lambda _: (dict(content_hash=s['anchor']['evidence']['prepared'],
        summaries=s['reuse']['summaries']), None, None))
    monkeypatch.setattr(worker, 'data', lambda s, r, co, **kw: caches[r][co])
    monkeypatch.setattr(submission, 'accepted_jobs', lambda _: {'train_'+stage: str(101+i) for i, stage in enumerate(c.STAGES)})
    monkeypatch.setattr(submission, 'successful', lambda *a: None)
    monkeypatch.setattr(worker, 'memory_snapshot', lambda _: dict(peak_rss_bytes=2**30, rss_bytes=2**30,
        gpu_peak_bytes=max(1, int(torch.cuda.max_memory_allocated()))))
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def test_exact_graph_independent_teacher_sets_weights_seeds_and_roles(tmp_path):
    s, _ = fixture(tmp_path, export=False)
    expected = {
        'UE000': ('OFFLINE', ['OFFLINE', 'M0HLT']),
        'UE100': ('UE000', ['OFFLINE', 'M0HLT']),
        'DE066': ('UE100', ['OFFLINE', 'UE100', 'M0HLT']),
        'DE033': ('DE066', ['OFFLINE', 'UE100', 'DE066', 'M0HLT']),
        'DE000': ('DE033', ['OFFLINE', 'UE100', 'DE066', 'DE033']),
    }
    assert list(c.STAGES) == list(expected)
    original = {n['coordinate']: n for n in original_nodes()}
    for i, (stage, (kd, emb)) in enumerate(expected.items()):
        r, n = s['recipes'][stage], s['nodes'][stage]
        assert r['kd_teacher'] == n['teacher'] == c.kd_teacher(stage) == kd
        assert r['teacher_order'] == c.teachers(stage) == emb
        assert r['weights'] == {t: 5/len(emb) for t in emb}
        assert (r['ce_coefficient'], r['kd_coefficient'], r['temperature']) == (.25, .75, 2.)
        assert r['epochs'] == 20 and r['batch_size'] == 128
        assert not any(r[k] for k in ('attention', 'relational', 'warm_start', 'teacher_embedding_average'))
        assert n['context_coordinate'] is None and n['deployable'] is (stage == 'DE000')
        for key in ('initialization_seed', 'sampler_seed', 'u', 'f'):
            assert n[key] == original[c.STAGES[stage]][key]
        assert c.tasks()[i]['dependencies'] == ['train_'+t for t in list(c.STAGES)[:i]]
    assert s['train_minutes'] == 1209 and s['task_minutes']['UE000'] == 1269
    assert s['new_scientific_fits'] == 5 and s['preflight_jobs'] == s['setup_jobs'] == 0
    assert s['startup'] == dict(stress_steps=3, technical_epochs=0, benchmark_steps=0, full_validation_pass=False)
    assert s['final_test_sealed'] and not s['final_test_accessed']
    assert 'UE000' not in c.EMBEDDING_EXPORTS and 'M0HLT' not in c.teachers('DE000')
    with pytest.raises(ValueError): c.target_recipe('UE000')
    with pytest.raises(ValueError): c.kd_recipe('M0HLT')


@pytest.mark.parametrize('key,value', [('counts', {}), ('gpu', dict(name='A100')), ('references', {}),
    ('original_train_minutes', True), ('multi_train_minutes', 2750), ('original_train_minutes', 699)])
def test_inapplicable_reuse_or_walltime_refused(tmp_path, key, value):
    s, _ = fixture(tmp_path, export=False)
    with pytest.raises(ValueError): campaign.policy(with_content_hash(dict(s['reuse'], **{key: value})))


def test_resource_evidence_and_export_memory_cannot_be_bypassed(tmp_path):
    s, _ = fixture(tmp_path, export=False)
    bad = deepcopy(s['reuse']); bad['summaries']['train']['OFFLINE']['resident_bytes'] = 200*2**30
    with pytest.raises(MemoryError): campaign.policy(with_content_hash(bad))
    p = Path(s['reuse']['original_preflight']['path']); p.write_bytes(p.read_bytes()+b'damage')
    with pytest.raises(ValueError): campaign.policy(s['reuse'])


@pytest.mark.parametrize('damage', ['negative', 'nonfinite', 'dtype', 'normalization', 'width'])
def test_kd_array_content_fails_closed(damage):
    p = np.full((3, 11), 1/11, np.float32)
    if damage == 'negative': p[0, 0] = -.1
    if damage == 'nonfinite': p[0, 0] = np.nan
    if damage == 'dtype': p = p.astype(np.float64)
    if damage == 'normalization': p *= 2
    if damage == 'width': p = p[:, :-1]
    with pytest.raises(ValueError): kd_banks.validate_probabilities(p, 3)


def test_export_inventory_has_no_unused_targets_or_hidden_tasks():
    assert worker.exports('UE000') == [('M0HLT', 'embedding'), ('OFFLINE', 'kd'), ('UE000', 'kd')]
    assert worker.exports('UE100') == [('UE100', 'embedding'), ('UE100', 'kd')]
    assert worker.exports('DE000') == []
    paths = [p for s in c.STAGES for p in worker.outputs(s)]
    assert len(paths) == len(set(paths))
    assert not any('targets/UE000/targets.npz' in p or 'final_test' in p for p in paths)


def test_new_teacher_reports_cannot_use_old_or_technical_checkpoint(tmp_path, monkeypatch):
    tiny(monkeypatch)
    s, _ = fixture(tmp_path, export=False)
    teacher = DelphesParticleTransformer()
    checkpoint = s['offline_refs']['checkpoint']
    teacher.load_state_dict(torch.load(checkpoint['path'], weights_only=True))
    for name, fields in [('old', dict(contract='old', schema_version=1)),
        ('technical', dict(node=dict(node_id='UE100'), scientific_fit=False, technical=True, selected_state='native_backbone_only')),
        ('wrong_node', dict(node=dict(node_id='COARSE_U100'), scientific_fit=True, technical=False, selected_state='native_backbone_only')),
        ('predicted', dict(node=dict(node_id='UE100'), scientific_fit=True, technical=False, selected_state='prediction_head'))]:
        value = with_content_hash(fields) if name == 'old' else c.artifact('TRAINING_REPORT', **fields)
        path = tmp_path/(name+'.json'); write_immutable_json(path, value)
        with pytest.raises(ValueError): banks.check_selected(teacher, dict(checkpoint=checkpoint, report=file_ref(path)), 'UE100')


@pytest.mark.parametrize('stage', c.STAGES)
def test_loss_exact_original_KD_plus_independent_means_and_detached_targets(stage):
    x = torch.randn(5, 11, requires_grad=True); y = torch.arange(5)
    q = torch.softmax(torch.randn(5, 11)/2, -1).requires_grad_(True)
    ps = {t: torch.randn(5, 128, requires_grad=True) for t in c.teachers(stage)}
    ts = {t: torch.randn(5, 128, requires_grad=True) for t in c.teachers(stage)}
    loss, ce, kd, aux = training.objective(x, y, ps, ts, stage=stage, probabilities=q)
    ref = distillation_loss(x, y, teacher_probabilities=q) + sum(5/len(ts)*aux[t] for t in ts)
    torch.testing.assert_close(loss, ref, rtol=0, atol=0)
    torch.testing.assert_close(kd, torch.nn.functional.kl_div(
        torch.log_softmax(x.float()/2, -1), q.detach(), reduction='batchmean')*4, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(loss, x, retain_graph=True)[0],
                               torch.autograd.grad(ref, x, retain_graph=True)[0], rtol=0, atol=0)
    loss.backward()
    assert q.grad is None
    for t in ts:
        assert ts[t].grad is None
        torch.testing.assert_close(ps[t].grad, (5/len(ts))*(ps[t].detach()-ts[t].detach()).clamp(-1, 1)/ps[t].numel())
    for bad in (None, q*2, torch.full_like(q, float('nan'))):
        with pytest.raises(ValueError): training.objective(x, y, ps, ts, stage=stage, probabilities=bad)
    with pytest.raises(ValueError): training.objective(x, y, ps, {}, stage=stage, probabilities=q)


@pytest.mark.parametrize('bf16', [False, True])
def test_native_head_rng_loss_gradients_and_optimizer_match_accepted_kernels(tmp_path, monkeypatch, bf16):
    tiny(monkeypatch)
    s, cs = fixture(tmp_path, export=False)
    if bf16 and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    device = 'cuda' if bf16 else 'cpu'
    old = multi_model.new_model(s['reuse']['base_node']); rng = torch.get_rng_state().clone()
    new = model.new_model(s['nodes']['DE000'])
    assert torch.equal(rng, torch.get_rng_state())
    mapped = dict(OFFLINE='OFFLINE', UE100='COARSE_U100', DE066='COARSE_D066', DE033='COARSE_D033')
    assert all(torch.equal(v, old.native_state()[k]) for k, v in new.native_state().items())
    for t, old_t in mapped.items():
        assert all(torch.equal(v, old.predictors[old_t].state_dict()[k]) for k, v in new.predictors[t].state_dict().items())
    old.to(device).train(); new.to(device).train(); old.capture = new.capture = True
    raw = cs['train']['D000'].batch(np.arange(4)); y = torch.from_numpy(raw['labels']).to(device)
    targets = {t: torch.randn(4, 128, device=device) for t in mapped}
    q = torch.softmax(torch.randn(4, 11, device=device)/2, -1)
    with parity_backend(device):
        seed = torch.get_rng_state(); gpu_seed = torch.cuda.get_rng_state() if device == 'cuda' else None
        logits = _forward(old, raw, device, bf16=bf16)
        a = distillation_loss(logits, y, teacher_probabilities=q) + sum(
            1.25*torch.nn.functional.smooth_l1_loss(old.predictions[mapped[t]], targets[t]) for t in mapped)
        torch.set_rng_state(seed)
        if gpu_seed is not None: torch.cuda.set_rng_state(gpu_seed)
        nx = _forward(new, raw, device, bf16=bf16)
        b, *_ = training.objective(nx, y, new.predictions, targets, stage='DE000', probabilities=q)
        torch.testing.assert_close(logits, nx, rtol=0, atol=0)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.backward(); b.backward()
        for k, p in new.named_parameters():
            old_k = k
            for t, old_t in mapped.items(): old_k = old_k.replace('predictors.'+t+'.', 'predictors.'+old_t+'.')
            torch.testing.assert_close(p.grad, dict(old.named_parameters())[old_k].grad, rtol=0, atol=0)
        for m in (old, new): torch.optim.AdamW(m.parameters(), lr=1e-3).step()
        assert all(torch.equal(v, old.native_state()[k]) for k, v in new.native_state().items())
    new.eval(); new.capture = False
    for head in new.predictors.values():
        for p in head.parameters(): p.data.fill_(float('nan'))
    with torch.no_grad(): assert torch.isfinite(_forward(new, raw, device, bf16=bf16)).all()
    DelphesParticleTransformer().load_state_dict(new.native_state(), strict=True)


@pytest.mark.parametrize('role,co', [('final_test', 'D000'), ('unused', 'U100'), ('validation', 'OFFLINE'), ('train', 'U050')])
def test_only_registered_roles_and_views(role, co):
    with pytest.raises(PermissionError): worker.data({}, role, co)


def test_real_prepared_cache_all_rungs(durable, monkeypatch):
    output, meta = durable
    s = dict(memory_mb=180000, counts=meta[1]['counts'], anchor=dict(evidence=dict(
        prepared=meta[0]['content_hash'], cache_root=str(output))))
    monkeypatch.setattr(worker, 'metadata', lambda _: meta)
    for role in ('train', 'validation'):
        original = worker.data(s, role, 'D000')
        for co in c.STAGES.values():
            value = worker.data(s, role, co)
            np.testing.assert_array_equal(value.identities, original.identities)
            np.testing.assert_array_equal(value.labels, original.labels)


def test_original_banks_native_targets_temperature_roundtrip_and_joins(tmp_path, monkeypatch):
    tiny(monkeypatch)
    s, cs = fixture(tmp_path)
    source_bank = deepcopy(s['offline_bank'])
    originals(s, cs)
    tr = cs['train']['U000']
    bundle = banks.publish(s, 'UE000', tr)
    assert set(bundle.targets) == {'OFFLINE', 'M0HLT'}
    assert bundle.manifest['kd_teacher'] == 'OFFLINE'
    teacher = DelphesParticleTransformer()
    teacher.load_state_dict(torch.load(s['offline_refs']['checkpoint']['path'], weights_only=True))
    teacher.eval().requires_grad_(False)
    np.testing.assert_array_equal(bundle.probabilities,
        predict(teacher, cs['train']['OFFLINE'], device='cpu', temperature=2., batch_size=128))
    assert not np.array_equal(bundle.probabilities, predict(teacher, cs['train']['OFFLINE'], device='cpu'))
    from hlt_classification.luka_embedding_aux.model import head_input
    from hlt_classification.luka_embedding_aux.targets import standardize
    teacher.load_state_dict(torch.load(s['references']['M0HLT']['checkpoint']['path'], weights_only=True))
    raw = cs['train']['D000'].batch(np.arange(len(tr)))
    with torch.inference_mode():
        _, native = head_input(teacher.eval(), **{k: torch.from_numpy(raw[k]) for k in ('features', 'vectors', 'mask')})
    expected, _, _ = standardize(native.numpy())
    np.testing.assert_allclose(bundle.targets['M0HLT'], expected, atol=1e-5)
    assert file_ref(source_bank['path']) == source_bank
    assert not bundle.probabilities.flags.writeable
    for key in ('identities', 'labels'):
        wrong = deepcopy(tr); setattr(wrong, key, getattr(wrong, key)[::-1])
        with pytest.raises(ValueError): banks.Bundle(s, 'UE000', wrong)
    with pytest.raises(PermissionError): banks.Bundle(s, 'UE000', cs['validation']['U000'])
    with pytest.raises(ValueError, match='checkpoint'):
        kd_banks.export(tmp_path/'bad', DelphesParticleTransformer(), cs['train']['OFFLINE'],
            refs=s['offline_refs'], device='cpu', teacher_id='OFFLINE')
    for forbidden in (cs['validation']['OFFLINE'], cs['train']['U000']):
        with pytest.raises(PermissionError): kd_banks.export(tmp_path/'forbidden', teacher, forbidden,
            refs=s['offline_refs'], device='cpu', teacher_id='OFFLINE')
    manifest = Path(s['root'])/'logit_targets/OFFLINE/manifest.json'; before = manifest.read_bytes()
    for key, value in [('temperature', 1.), ('teacher', 'M0HLT'), ('coordinate', 'U100'),
                       ('role', 'validation'), ('rows', 1), ('final_test_accessed', True)]:
        manifest.write_text(json.dumps(with_content_hash(dict(json.loads(before), **{key: value}))))
        with pytest.raises((ValueError, PermissionError)): banks.Bundle(s, 'UE000', tr)
    manifest.write_bytes(before)
    payload = manifest.parent/'probabilities.npz'; payload.write_bytes(payload.read_bytes()+b'corrupt')
    with pytest.raises(ValueError): banks.Bundle(s, 'UE000', tr)


@pytest.mark.parametrize('measured', [dict(peak_rss_bytes=0, gpu_peak_bytes=1),
    dict(peak_rss_bytes=180000*1024**2, gpu_peak_bytes=1), dict(peak_rss_bytes=1, gpu_peak_bytes=48*2**30)])
def test_memory_failure_no_fallback(tmp_path, measured):
    s, _ = fixture(tmp_path, export=False)
    with pytest.raises(MemoryError): worker.limits(s, measured)


def test_exact_five_job_dry_live_submission_and_claim(tmp_path, monkeypatch):
    s, _ = fixture(tmp_path, export=False)
    monkeypatch.setattr(campaign, 'check', lambda _: None)
    monkeypatch.setattr(submission, 'check_site', lambda _: None)
    plan = submission.submit(s)
    assert plan['jobs'] == 5 and plan['setup_jobs'] == plan['preflight_jobs'] == 0
    assert [r['task_id'] for r in plan['commands']] == ['train_'+stage for stage in c.STAGES]
    for i, r in enumerate(plan['commands']):
        assert '--kill-on-invalid-dep=yes' in r['command'] and '--no-requeue' in r['command']
        assert '--export=NONE' in r['command']
        if i: assert '--dependency=afterok:'+':'.join('${JOB_train_'+t+'}' for t in list(c.STAGES)[:i]) in r['command']
    for auth, h in [(None, plan['content_hash']), (c.AUTHORIZATION, 'bad')]:
        with pytest.raises(PermissionError): submission.submit(s, execute=True, authorization=auth, plan_hash=h)
    calls = []
    def fake(args, **kw):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=str(100+len(calls))+'\n', stderr='')
    monkeypatch.setattr(subprocess, 'run', fake)
    submission.submit(s, execute=True, authorization=c.AUTHORIZATION, plan_hash=plan['content_hash'])
    assert submission.accepted_jobs(s) == {'train_'+stage: str(101+i) for i, stage in enumerate(c.STAGES)}
    assert '--dependency=afterok:101:102:103:104' in calls[4]
    with pytest.raises(FileExistsError): submission.submit(s, execute=True, authorization=c.AUTHORIZATION, plan_hash=plan['content_hash'])
    assert len(calls) == 5


def test_ambiguous_submission_never_retries(tmp_path, monkeypatch):
    s, _ = fixture(tmp_path, export=False)
    monkeypatch.setattr(campaign, 'check', lambda _: None)
    monkeypatch.setattr(submission, 'check_site', lambda _: None)
    plan = submission.submit(s); calls = []
    def ambiguous(args, **kw):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout='ambiguous\n', stderr='')
    monkeypatch.setattr(subprocess, 'run', ambiguous)
    with pytest.raises(Exception): submission.submit(s, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    with pytest.raises(FileExistsError): submission.submit(s, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 1


def test_failed_accounting_and_missing_parent_block_fit(tmp_path, monkeypatch):
    s, _ = fixture(tmp_path, export=False)
    monkeypatch.setattr(submission, 'accepted_jobs', lambda _: dict(train_UE000='101', train_UE100='102'))
    monkeypatch.setattr(submission.getpass, 'getuser', lambda: 'rlyang')
    valid = '101|lukafsh_train_UE000|rlyang|default|COMPLETED|0:0\n'
    for text in ('', valid.replace('COMPLETED', 'FAILED'), valid.replace('0:0', '1:0'), valid.replace('rlyang', 'other')):
        monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=text))
        with pytest.raises(ValueError): submission.successful(s, 'train_UE000', '101')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=valid))
    submission.successful(s, 'train_UE000', '101')
    monkeypatch.setattr(campaign, 'check', lambda _: None)
    monkeypatch.setattr(worker, 'execution', lambda *a: '102')
    monkeypatch.setattr(worker, 'fit', lambda *a, **kw: pytest.fail('No fit without completed predecessor'))
    with pytest.raises(ValueError, match='ancestor incomplete'): worker.run(s, 'train_UE100')
    assert not (Path(s['root'])/'train_UE100_attempt').exists()


def test_actual_full_architecture_cuda_five_worker_hundred_epoch_miniature(tmp_path, monkeypatch):
    if not torch.cuda.is_available(): pytest.skip('No CUDA; not Oscar certification')
    s, cs = fixture(tmp_path)
    mock_execution(s, cs, monkeypatch)
    source_bank = deepcopy(s['offline_bank'])
    for i, stage in enumerate(c.STAGES):
        monkeypatch.setenv('SLURM_JOB_ID', str(101+i))
        receipt = worker.run(s, 'train_'+stage)
        assert receipt['stage'] == stage
        report = load_json(Path(s['root'])/('train_'+stage)/'training_report.json')
        assert report['passes'] == 20 and len(report['validation_history']) == 20
        assert all(set(r['auxiliary_losses']) == set(c.teachers(stage)) for r in report['validation_history'])
        assert all(r['weighted_kd_loss'] == .75*r['kd_loss'] for r in report['validation_history'])
        startup = load_json(Path(s['root'])/('train_'+stage+'_attempt')/'startup.json')
        assert startup['stress_steps'] == 3 and startup['technical_epochs'] == startup['benchmark_steps'] == 0
        state = torch.load(Path(s['root'])/('train_'+stage)/'selected.pt', weights_only=True)
        assert not any(k.startswith('predictors.') for k in state)
        if stage in c.EMBEDDING_EXPORTS:
            a = load_npz_arrays(Path(s['root'])/'targets'/stage/'targets.npz')
            np.testing.assert_allclose(a['targets'].mean(0), 0, atol=1e-6)
        if stage in c.KD_EXPORTS:
            native = DelphesParticleTransformer().cuda(); native.load_state_dict(state)
            a = load_npz_arrays(Path(s['root'])/'logit_targets'/stage/'probabilities.npz')
            np.testing.assert_array_equal(a['probabilities'], predict(native, cs['train'][c.STAGES[stage]],
                device='cuda', temperature=2., batch_size=128))
            del native
        if i: assert receipt['parents'][list(c.STAGES)[i-1]] == previous_receipt['content_hash']
        previous_receipt = receipt
        summary = worker.results(s)
        assert summary['pending'] == list(c.STAGES)[i+1:]
    assert file_ref(source_bank['path']) == source_bank
    assert not (Path(s['root'])/'targets/UE000').exists()
    assert not summary['final_test_accessed'] and not summary['fresh_matched_control']
    assert summary['rows'][-1]['input'] == 'native_HLT'
    assert all(r['input'] == 'richer_input_oracle' for r in summary['rows'][6:10])
    for first, second in [('UE100', 'DE066'), ('DE066', 'DE033')]:
        a = load_npz_arrays(Path(s['root'])/'targets'/first/'targets.npz')
        b = load_npz_arrays(Path(s['root'])/'targets'/second/'targets.npz')
        assert not np.array_equal(a['mean'], b['mean'])
        assert not np.array_equal(a['targets'], b['targets'])
    # A KD-only ancestor still belongs to final transitive provenance.
    task = Path(s['root'])/'train_UE000/task_receipt.json'; before = task.read_bytes()
    task.write_text(json.dumps(with_content_hash(dict(load_json(task), job_id='999'))))
    with pytest.raises(ValueError): worker.results(s)
    task.write_bytes(before)
    target = Path(s['root'])/'logit_targets/UE000/probabilities.npz'
    target.write_bytes(target.read_bytes()+b'damage')
    with pytest.raises(ValueError): worker.completed(s, 'DE000')


def test_poor_science_still_completes_and_initialization_resets(tmp_path, monkeypatch):
    tiny(monkeypatch)
    s, cs = fixture(tmp_path); originals(s, cs)
    tr, val = cs['train']['U000'], cs['validation']['U000']
    bank = banks.publish(s, 'UE000', tr)
    monkeypatch.setattr(training, 'predict', lambda *a, **kw: np.full((len(val), 11), 1/11, np.float32))
    torch.manual_seed(42)
    report, state = training.train(tr, val, bank=bank, node=s['nodes']['UE000'], device='cpu')
    worker.report_valid(report, s, 'UE000', bank.manifest['content_hash'])
    assert report['passes'] == 20 and report['selected_pass'] == 1
    assert all(r['validation']['macro_ovr_auc'] == .5 for r in report['validation_history'])
    torch.manual_seed(999)
    again, other = training.train(tr, val, bank=bank, node=s['nodes']['UE000'], device='cpu')
    assert all(torch.equal(v, other[k]) for k, v in state.items())
    for field in ('weighted_kd_loss', 'weighted_ce_loss', 'total_loss'):
        bad = deepcopy(report); bad['validation_history'][0][field] += .01
        with pytest.raises(ValueError): worker.report_valid(with_content_hash(bad), s, 'UE000', bank.manifest['content_hash'])


def test_ranks_ties_censoring():
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    p = np.zeros((13, 11)); p[:, 0] = 1
    scores = np.array([.85, .65, .2, 1., .9, .8, .7, .6, .5, .4, .3, .2, .1])
    p[:, 0] = 1-scores; p[:, 1] = scores
    assert worker.rejection(labels, p, .3)['X_bb'] == 3
    assert worker.rejection(labels, p, .5)['X_bb'] == 1.5
    p[0, 1] = .8; p[0, 0] = .2
    assert worker.rejection(labels, p, .3)['X_bb'] == 3
    p[:3, 1] = 0; p[:3, 0] = 1
    assert worker.rejection(labels, p, .5)['X_bb'] is None


def test_source_replay_and_changed_donor_runtime_refused(tmp_path, monkeypatch):
    original = tmp_path/'old'; project = tmp_path/'new'; root = tmp_path/'parent'; root.mkdir()
    for directory in (original, project):
        (directory/'src').mkdir(parents=True); (directory/'src/kernel.py').write_text('# accepted\n')
    write_immutable_json(root/'campaign_spec.json', dict(project_dir=str(original), root=str(root),
        source_commit=source.DONOR, counts=c.COUNTS))
    evidence = dict(jobs=dict(train_multi4='123'), execution_site=dict(account='default'))
    def replay(args, **kw):
        if args[0] == 'git': return SimpleNamespace(stdout='src/kernel.py\n')
        if args[0] == 'sacct': return SimpleNamespace(stdout='123|lukafsm_train_multi4|rlyang|default|COMPLETED|0:0\n')
        assert kw['cwd'] == original and kw['env']['PYTHONPATH'] == str(original/'src')
        return SimpleNamespace(returncode=0, stdout='HYBRID_REUSE='+json.dumps(evidence)+'\n', stderr='')
    monkeypatch.setattr(subprocess, 'run', replay); monkeypatch.setattr(source.getpass, 'getuser', lambda: 'rlyang')
    c.validate(source.authenticate(root/'campaign_spec.json', project), 'REUSE')
    compile(source.REPLAY, 'replay', 'exec')
    assert 'original.completed' in source.REPLAY
    (project/'src/kernel.py').write_text('# changed\n')
    with pytest.raises(ValueError, match='Changed donor'): source.authenticate(root/'campaign_spec.json', project)


@pytest.mark.parametrize('output,returncode', [('', 1), ('', 0), ('HYBRID_REUSE={}\nHYBRID_REUSE={}\n', 0)])
def test_missing_failed_or_ambiguous_original_replay_blocks_reuse(tmp_path, monkeypatch, output, returncode):
    original = tmp_path/'old'; project = tmp_path/'new'; root = tmp_path/'parent'; root.mkdir()
    for directory in (original, project):
        (directory/'src').mkdir(parents=True); (directory/'src/kernel.py').write_text('')
    write_immutable_json(root/'campaign_spec.json', dict(project_dir=str(original), root=str(root),
        source_commit=source.DONOR, counts=c.COUNTS))
    monkeypatch.setattr(subprocess, 'run', lambda args, **kw: SimpleNamespace(stdout='src/kernel.py\n')
        if args[0] == 'git' else SimpleNamespace(returncode=returncode, stdout=output, stderr='evidence rejected'))
    with pytest.raises(ValueError): source.authenticate(root/'campaign_spec.json', project)


def test_additive_source_guard_real_donor_compatibility_and_required_files(monkeypatch):
    project = Path(__file__).resolve().parents[1]
    changed = subprocess.run(['git', 'diff', '--name-only', '--diff-filter=MDR', source.DONOR, source.BASELINE,
        '--', 'src', 'scripts', 'sbatch'], cwd=project, check=True, capture_output=True, text=True).stdout.splitlines()
    assert not changed
    monkeypatch.setattr(source, '_source', lambda *a: dict(content_hash='a'*64))
    monkeypatch.setattr(subprocess, 'run', lambda cmd, **kw: SimpleNamespace(stdout='\n'.join(source.REQUIRED)
        if 'ls-files' in cmd else '\n'.join(source.ADDITIONS|source.ENTRYPOINTS) if 'diff' in cmd else ''))
    assert source.current(project, 'b'*40)['content_hash'] == 'a'*64
    for p in source.REQUIRED: assert (project/p).is_file(), p
    monkeypatch.setattr(subprocess, 'run', lambda cmd, **kw: SimpleNamespace(stdout='\n'.join(source.REQUIRED)
        if 'ls-files' in cmd else 'src/old.py' if 'diff' in cmd else ''))
    with pytest.raises(ValueError): source.current(project, 'b'*40)


def test_shell_cli_docs_syntax():
    project = Path(__file__).resolve().parents[1]
    bash = Path('C:/Program Files/Git/bin/bash.exe')
    if not bash.is_file(): bash = shutil.which('bash')
    if not bash: pytest.skip('Bash unavailable')
    for p in ('scripts/queue_luka_fullstat_hybrid_embedding.sh', 'sbatch/run_luka_fullstat_hybrid_embedding.sh'):
        subprocess.run([str(bash), '-n', p], cwd=project, check=True, capture_output=True, timeout=30)
    book = (project/'docs/LUKA_FULLSTAT_HYBRID_EMBEDDING_RUNBOOK.md').read_text()
    for block in re.findall(r'^```bash\n(.*?)^```', book, re.M|re.S):
        subprocess.run([str(bash), '-n'], input=block, check=True, text=True, capture_output=True, timeout=30)
    subprocess.run([sys.executable, '-B', 'scripts/luka_fullstat_hybrid_embedding.py', '--help'],
        cwd=project, check=True, capture_output=True, timeout=60)
    for p in source.REQUIRED:
        text = (project/p).read_text(); assert all(line == line.rstrip() for line in text.splitlines()), p
        if p.endswith('.md'):
            for link in re.findall(r'\]\(([^)#]+)(?:#[^)]*)?\)', text):
                assert (project/p).parent.joinpath(link).is_file(), (p, link)


def test_parallel_trial_additions_do_not_require_changing_old_runtime_guard():
    companion = source.COMPANION_ADDITIONS
    assert source.compatible_changes(source.ADDITIONS | companion, companion)
    assert not source.compatible_changes(companion, set())
    assert not source.compatible_changes({'src/hlt_classification/jetclass2_delphes/model.py'}, companion)
    assert not source.compatible_changes({'src/hlt_classification/unknown/__init__.py'}, companion)
    project = Path(__file__).resolve().parents[1]
    for p in source.ADDITIONS - {'src/hlt_classification/luka_fullstat_hybrid_embedding/source.py'}:
        assert 'luka_particle_prediction' not in (project/p).read_text()

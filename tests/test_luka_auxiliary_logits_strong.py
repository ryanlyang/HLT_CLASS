"""Loss isolation, exact c50 imports, native parity, four-job workflow and retention."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from test_luka_u100_pretrain import cache
from test_luka_embedding_aux import tiny
from test_luka_trial_hybrid import bare_spec as parent_spec, science_fixture as parent_fixture
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays, with_content_hash,
    write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer, installed_environment
from hlt_classification.jetclass2_delphes.runner import _forward, predict
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_trial_hybrid import contracts as old, worker as old_worker, banks as old_banks
from hlt_classification.luka_auxiliary_logits_strong import contracts as c, model, training, banks, campaign, source, submission, worker, startup


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def bare(path):
    p = parent_spec(Path('/parent'))
    return dict(root=str(path), project_dir='/project', source_commit='a'*40, source_snapshot=dict(content_hash='b'*64),
        content_hash='c'*64, anchor=p['anchor'], reuse=dict(acceptance=p['acceptance'], teachers={}),
        available_quota_gib=3, **campaign.policy(p))


def fixture(tmp_path, monkeypatch):
    root = tmp_path/'new'; root.mkdir(); spec = bare(root); env = installed_environment()
    spec['anchor']['evidence']['environment'] = env
    spec['reuse']['acceptance']['preflight']['environment'] = env
    spec['counts'] = dict(train=22, validation=22); spec['original50k_rows'] = 22
    tr = {co: cache('train', co) for co in ('OFFLINE', 'U100', 'D066', 'D033', 'D000')}
    val = cache('validation', 'D000')
    for i, t in enumerate(c.TEACHERS):
        folder = tmp_path/t; folder.mkdir(); torch.manual_seed(200+i)
        n = DelphesParticleTransformer(); save_state(folder/'selected.pt', n.state_dict())
        write_immutable_json(folder/'report.json', dict(fixture=True))
        files = dict(checkpoint=file_ref(folder/'selected.pt'), report=file_ref(folder/'report.json'))
        spec['reuse']['teachers'][t] = old.artifact('TEACHER', parents=dict(spec='f'*64,
            prepared=spec['anchor']['evidence']['prepared'], **{k: r['sha256'] for k, r in files.items()}),
            scope='shared' if t == 'OFFLINE' else 'c50', teacher=t,
            coordinate=dict(OFFLINE='OFFLINE', UE100='U100', DE066='D066', DE033='D033')[t], files=files)
    references = tmp_path/'references'; references.mkdir(); spec['reuse']['references'] = {}
    uniform = np.full((len(val), 11), 1/11, np.float32)
    for name in c.REFERENCES:
        path = references/(name+'.npz')
        atomic_publish_bytes(path, deterministic_npz_bytes(dict(identities=val.identities, labels=val.labels, probabilities=uniform)))
        spec['reuse']['references'][name] = file_ref(path)
    atomic_publish_bytes(references/'subset.npz', deterministic_npz_bytes(dict(indices=np.arange(len(val), dtype=np.int64))))
    spec['reuse']['subset'] = file_ref(references/'subset.npz')
    spec = with_content_hash(spec); write_immutable_json(root/'campaign_spec.json', spec)
    jobs = {t['task_id']: str(101+i) for i, t in enumerate(c.tasks())}
    monkeypatch.setattr(campaign, 'check', lambda _: None)
    monkeypatch.setattr(campaign, 'validate_spec', lambda *a, **kw: None)
    monkeypatch.setattr(source, 'current', lambda *a: spec['source_snapshot'])
    monkeypatch.setattr(worker, 'execution', lambda _, t: jobs[t])
    monkeypatch.setattr(worker, 'storage', lambda _: 0)
    monkeypatch.setattr(worker.regular_cache, 'metadata', lambda _: None)
    monkeypatch.setattr(worker, 'data', lambda _, r, co, meta: tr[co] if r == 'train' else val)
    monkeypatch.setattr(submission, 'accepted_jobs', lambda _: jobs)
    monkeypatch.setattr(submission, 'successful', lambda *a: None)
    monkeypatch.setattr(worker, 'memory_snapshot', lambda _: dict(peak_rss_bytes=2**30, gpu_peak_bytes=2**30))
    monkeypatch.setattr(startup, 'memory_snapshot', worker.memory_snapshot)
    return spec, tr, val


def tensors():
    x = torch.randn(4, 11, requires_grad=True); labels = torch.arange(4)
    p = {t: torch.randn(4, 11, requires_grad=True) for t in c.TEACHERS}
    z = {t: torch.randn(4, 128, requires_grad=True) for t in c.TEACHERS}
    q = {t: torch.softmax(torch.randn(4, 11)/2, -1).requires_grad_(True) for t in c.TEACHERS}
    targets = {t: torch.randn(4, 128, requires_grad=True) for t in c.TEACHERS}
    return x, labels, p, z, q, targets


@pytest.mark.parametrize('arm', c.WEIGHTS)
def test_exact_objective_main_gradient_detached_targets_and_zero_control(arm):
    x, y, p, z, q, targets = tensors()
    loss, ce, kd, embed = training.objective(x, y, p, z, q, targets, arm=arm)
    expected = .5*F.cross_entropy(x, y)+sum(1.25*F.smooth_l1_loss(z[t], targets[t].detach()) for t in c.TEACHERS)
    if c.WEIGHTS[arm]:
        expected += sum(c.WEIGHTS[arm]/4*4*F.kl_div((p[t]/2).log_softmax(-1), q[t].detach(), reduction='batchmean') for t in c.TEACHERS)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    actual = torch.autograd.grad(loss, x, retain_graph=True)[0]
    torch.testing.assert_close(actual, torch.autograd.grad(.5*ce, x, retain_graph=True)[0], rtol=0, atol=0)
    loss.backward()
    assert all(v.grad is None for v in [*q.values(), *targets.values()])
    assert all(v.grad is not None and v.grad.abs().sum() > 0 for v in z.values())
    assert all(v.grad is not None and v.grad.abs().sum() > 0 for v in p.values())


@pytest.mark.parametrize('bad', ['nan', 'negative', 'normalization', 'inventory', 'shape'])
def test_bad_targets_rejected(bad):
    x, y, p, z, q, targets = tensors()
    if bad == 'nan': targets['OFFLINE'] = torch.full((4, 128), float('nan'))
    if bad == 'negative': q['DE033'] = -q['DE033']
    if bad == 'normalization': q['DE066'] = q['DE066']*2
    if bad == 'inventory': del q['UE100']
    if bad == 'shape': z['DE033'] = torch.zeros(4, 127)
    with pytest.raises(ValueError): training.objective(x, y, p, z, q, targets, arm='k5')


def test_graph_weights_seeds_resource_allowance_and_native_nodes():
    spec = bare(Path('/new')); plan = submission.render(spec)
    assert plan['jobs'] == 4 and plan['scientific_fits'] == 2 and plan['preflight_jobs'] == 0
    assert spec['technical_epochs'] == 0 and spec['startup_steps_per_arm'] == 3
    assert len(plan['commands']) == 4
    assert [r['task_id'] for r in plan['commands']] == ['targets', 'k5', 'k10', 'report']
    assert [c.WEIGHTS[a] for a in c.FIT_SUBMISSION_ORDER] == [5., 10.]
    for arm, weight in c.WEIGHTS.items():
        r = c.recipe(arm); node = spec['nodes'][arm]
        assert r['ce_coefficient'] == .5 and r['embedding_total'] == 5 and r['logit_total'] == weight
        assert sum(r['logit_weights'].values()) == weight and sum(r['embedding_weights'].values()) == 5
        assert node['teacher'] is None and node['coordinate'] == 'D000' and node['deployable']
        assert r['embedding_seeds'] == old.recipe('c50', 'DE000')['predictor_seeds']
        assert next(r for r in plan['commands'] if r['task_id'] == arm)['dependencies'] == ['targets']
    assert plan['commands'][-1]['dependencies'] == ['targets', *c.FIT_SUBMISSION_ORDER]
    assert all('--partition=debug' in r['command'] for r in plan['commands'])
    assert all('--kill-on-invalid-dep=yes' in r['command'] for r in plan['commands'])
    assert not any(x.startswith('--gres=') for x in plan['commands'][-1]['command'])
    bad = parent_spec(Path('/old')); bad['task_minutes']['c50_DE000'] = 1441
    with pytest.raises(ValueError, match='24h'): campaign.policy(bad)
    with pytest.raises(ValueError): c.recipe('unregistered')


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('bf16', [False, True])
def test_full_native_architecture_parity_gradient_isolation_and_all_arm_rng(device, bf16):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    spec = bare(Path('/new')); raw = cache().batch(np.arange(4))
    torch.manual_seed(spec['nodes']['k5']['initialization_seed']); native = DelphesParticleTransformer().to(device).eval()
    reference_rng = torch.random.get_rng_state(); previous = None
    for arm in c.WEIGHTS:
        student = model.new_model(spec['nodes'][arm]).to(device).eval(); student.capture = True
        assert torch.equal(torch.random.get_rng_state(), reference_rng)
        assert all(torch.equal(v, native.state_dict()[k].cpu()) for k, v in student.native_state().items())
        state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        if previous is not None: assert all(torch.equal(v, previous[k]) for k, v in state.items())
        previous = state
        with parity_backend(device):
            x = _forward(native, raw, device, bf16=bf16); actual = _forward(student, raw, device, bf16=bf16)
            torch.testing.assert_close(x, actual, rtol=0, atol=0)
            assert all(v.dtype == torch.float32 for v in student.auxiliary_logits.values())
            aux = sum(v.square().mean() for v in [*student.auxiliary_logits.values(), *student.predictions.values()])
            aux.backward()
            assert all(p.grad is None for p in student.mod.fc.parameters())
            assert all(p.grad is not None for p in student.logit_heads.parameters())
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in student.named_parameters() if n.startswith('mod.'))
            student.zero_grad(set_to_none=True)
            actual = _forward(student, raw, device, bf16=bf16); F.cross_entropy(actual.float(), torch.from_numpy(raw['labels']).to(device)).backward()
            assert all(p.grad is None for p in [*student.predictors.parameters(), *student.logit_heads.parameters()])
        assert not any(k.startswith(('predictors.', 'logit_heads.')) for k in student.native_state())


def test_native_targets_normalization_exact_joins_and_no_wrong_branch(tmp_path, monkeypatch):
    tiny(monkeypatch); spec, tr, val = fixture(tmp_path, monkeypatch)
    worker.run(spec, 'targets', device='cpu')
    bundle = banks.Bundle(spec, tr['D000']); banks.validate_bundle(bundle.manifest, spec)
    for teacher in c.TEACHERS:
        record = banks.record(spec, teacher)
        m, a = old_banks.load(banks.directory(spec, teacher), tr['D000'], record)
        np.testing.assert_allclose(a['targets'].mean(0), 0, atol=1e-6)
        assert a['mean'].dtype == a['scale'].dtype == np.float64 and min(a['scale']) >= 1e-3
        n = worker.native(record['files']['checkpoint'], 'cpu')
        np.testing.assert_allclose(a['probabilities'], predict(n, tr[record['coordinate']], device='cpu', temperature=2.), rtol=1e-5, atol=1e-6)
    with pytest.raises(PermissionError): banks.Bundle(spec, val)
    bad = deepcopy(tr['D000']); bad.identities = bad.identities[::-1].copy()
    with pytest.raises(ValueError): banks.Bundle(spec, bad)
    wrong = deepcopy(spec); r = wrong['reuse']['teachers']['DE033']; r['scope'] = 'c25'
    wrong['reuse']['teachers']['DE033'] = with_content_hash(r)
    with pytest.raises(ValueError): banks.record(wrong, 'DE033')


@pytest.mark.parametrize('damage', ['probability', 'target', 'label', 'moment'])
def test_rehashed_bad_banks_fail_semantic_validation(tmp_path, monkeypatch, damage):
    tiny(monkeypatch); spec, tr, val = fixture(tmp_path, monkeypatch)
    teacher = 'OFFLINE'; record = banks.record(spec, teacher)
    banks.export(spec, teacher, worker.native(record['files']['checkpoint'], 'cpu'), tr['OFFLINE'], device='cpu')
    folder = banks.directory(spec, teacher); a = load_npz_arrays(folder/'targets.npz')
    if damage == 'probability': a['probabilities'][0] *= 2
    if damage == 'target': a['targets'][0, 0] = np.nan
    if damage == 'label': a['labels'][0] = (a['labels'][0]+1) % 11
    if damage == 'moment': a['mean'][0] += 1
    (folder/'targets.npz').write_bytes(deterministic_npz_bytes(a))
    m = load_json(folder/'manifest.json'); m['payload'] = file_ref(folder/'targets.npz', root=folder)
    (folder/'manifest.json').write_text(json.dumps(with_content_hash(m)))
    with pytest.raises(ValueError): old_banks.load(folder, tr['D000'], record)


def test_startup_failure_does_not_publish_targets_or_allow_science(tmp_path, monkeypatch):
    tiny(monkeypatch); spec, _, _ = fixture(tmp_path, monkeypatch)
    def fail(*a, **kw): raise ValueError('injected startup failure')
    monkeypatch.setattr(startup, 'run', fail)
    with pytest.raises(ValueError, match='startup'): worker.run(spec, 'targets', device='cpu')
    assert not (Path(spec['root'])/'targets/receipt.json').exists()
    with pytest.raises(FileNotFoundError): worker.run(spec, 'k5', device='cpu')
    assert not (Path(spec['root'])/'k5').exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual installed-Weaver CUDA required')
def test_full_architecture_two_fixed30_workers_native_checkpoints_and_retirement(tmp_path, monkeypatch):
    spec, tr, val = fixture(tmp_path, monkeypatch)
    teacher_refs = [r['files']['checkpoint'] for r in spec['reuse']['teachers'].values()]
    worker.run(spec, 'targets')
    assert worker.results(spec)['pending'] == list(c.WEIGHTS)
    startup.check(load_json(Path(spec['root'])/'targets/startup.json'), spec,
                  load_json(Path(spec['root'])/'targets/bundle.json')['content_hash'])
    for arm in c.WEIGHTS:
        worker.run(spec, arm)
        report, arrays = worker.validate_fit(spec, arm)
        assert report['passes'] == 30 and len(report['validation_history']) == 30
        selected = torch.load(Path(spec['root'])/arm/'fit/selected.pt', weights_only=True)
        assert not any(k.startswith(('predictors.', 'logit_heads.')) for k in selected)
    before = worker.results(spec); assert not before['pending'] and len(before['rows']) == 7
    worker.run(spec, 'report')
    assert before == worker.results(spec)
    assert all(not (Path(spec['root'])/p).exists() for p in worker.target_payloads())
    for ref in teacher_refs: validate_file_ref(ref)
    for role in ('validation', 'original50k'):
        assert all(row[role]['rows'] == 22 for row in before['rows'])
    # Retained results need neither old teacher weights nor retired target arrays.
    monkeypatch.setattr(source, 'validate_reuse', lambda *a: (_ for _ in ()).throw(AssertionError('teacher access')))
    assert before == worker.results(spec)
    path = Path(spec['root'])/'k10/fit/validation_predictions.npz'; path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError): worker.results(spec)


def test_poor_science_finishes_high_weight_heads_update_and_loss_tampering(tmp_path, monkeypatch):
    tiny(monkeypatch); spec, tr, val = fixture(tmp_path, monkeypatch); worker.run(spec, 'targets', device='cpu')
    bank = banks.Bundle(spec, tr['D000']); reference = model.new_model(spec['nodes']['k5'])
    initial = {k: v.clone() for k, v in reference.logit_heads.state_dict().items()}
    monkeypatch.setattr(training, 'predict', lambda *a, **kw: np.full((22, 11), 1/11, np.float32))
    net, report, state = training.train(tr['D000'], val, bank=bank, node=spec['nodes']['k5'], device='cpu')
    training.validate_report(report, spec, 'k5', bank.manifest['content_hash'])
    assert report['passes'] == 30 and report['selected_pass'] == 1
    assert any(not torch.equal(initial[k], v) for k, v in net.logit_heads.state_dict().items())
    bad = deepcopy(report); bad['validation_history'][0]['weighted_logit_losses']['OFFLINE'] = .1
    with pytest.raises(ValueError): training.validate_report(with_content_hash(bad), spec, 'k5', bank.manifest['content_hash'])


def test_four_job_dry_live_exact_dependencies_and_duplicate_refusal(tmp_path, monkeypatch):
    spec = bare(tmp_path); calls = []
    monkeypatch.setattr(campaign, 'check', lambda _: None); monkeypatch.setattr(submission, 'check_site', lambda _: None)
    def sbatch(cmd, **kw):
        calls.append(cmd); return SimpleNamespace(returncode=0, stdout=str(100+len(calls))+'\n', stderr='')
    monkeypatch.setattr(subprocess, 'run', sbatch)
    plan = submission.submit(spec); assert calls == []
    with pytest.raises(PermissionError): submission.submit(spec, execute=True, plan_hash='wrong', authorization=c.AUTHORIZATION)
    submission.submit(spec, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 4 and len(submission.accepted_jobs(spec)) == 4
    assert [next(v for v in cmd if v.startswith('--job-name=')) for cmd in calls] == [
        '--job-name=lukaas_'+t for t in ('targets', 'k5', 'k10', 'report')]
    assert submission.accepted_jobs(spec) == dict(targets='101', k5='102', k10='103', report='104')
    assert all('--dependency=afterok:101' in command for command in calls[1:3])
    assert '--dependency=afterok:101:102:103' in calls[3]
    assert not any('${JOB_' in word for command in calls for word in command)
    with pytest.raises(FileExistsError): submission.submit(spec, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)


def test_ambiguous_submission_no_retry(tmp_path, monkeypatch):
    spec = bare(tmp_path)
    monkeypatch.setattr(campaign, 'check', lambda _: None); monkeypatch.setattr(submission, 'check_site', lambda _: None)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0, stdout='ambiguous', stderr=''))
    plan = submission.submit(spec)
    with pytest.raises(ValueError): submission.submit(spec, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    with pytest.raises(FileExistsError): submission.submit(spec, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)


def test_original_source_replay_accepts_only_completed_c50_and_other_arms_pending(tmp_path, monkeypatch):
    tiny(monkeypatch); parent, _, _ = parent_fixture(tmp_path, monkeypatch)
    old_worker.run(parent, 'targets', device='cpu')
    for stage in old.STAGES: old_worker.run(parent, 'c50_'+stage, device='cpu')
    checked = []
    from hlt_classification.luka_trial_hybrid import submission as old_submission
    monkeypatch.setattr(old_submission, 'successful', lambda _, t, j: checked.append(t))
    monkeypatch.setattr(sys, 'argv', ['replay', str(Path(parent['root'])/'campaign_spec.json')])
    out = io.StringIO()
    with redirect_stdout(out): exec(source.REPLAY, {})
    value = json.loads(out.getvalue().split('AUXLOGIT_STRONG_REUSE=')[1])
    assert set(checked) == {'targets', *('c50_'+s for s in old.STAGES)}
    assert set(value['teachers']) == set(c.TEACHERS)
    assert all(value['teachers'][t]['scope'] == 'c50' for t in c.TEACHERS if t != 'OFFLINE')
    assert not (Path(parent['root'])/'c25_DE000').exists()
    ref = file_ref(Path(parent['root'])/'campaign_spec.json')
    reuse = c.artifact('REUSE', parents=dict(spec=parent['content_hash']), spec=ref, **value)
    monkeypatch.setattr(source, 'parent', lambda _: parent)
    monkeypatch.setattr(source, 'validate_source_snapshot', lambda *a, **kw: None)
    source.validate_reuse(reuse)
    bad = deepcopy(reuse); bad['teachers']['DE033']['scope'] = 'c25'
    bad['teachers']['DE033'] = with_content_hash(bad['teachers']['DE033'])
    with pytest.raises(ValueError): source.validate_reuse(with_content_hash(bad))
    # The same c50 import must survive the parent's normal all-arm retirement.
    for arm in old.ARMS:
        if arm != 'c50':
            for stage in old.STAGES: old_worker.run(parent, arm+'_'+stage, device='cpu')
    old_worker.run(parent, 'report', device='cpu')
    assert all(not (Path(parent['root'])/p).exists() for p in old_worker.target_payloads())
    out = io.StringIO()
    with redirect_stdout(out): exec(source.REPLAY, {})
    assert json.loads(out.getvalue().split('AUXLOGIT_STRONG_REUSE=')[1]) == value
    source.validate_reuse(reuse)


def test_create_exact_policy_and_reporting_without_old_payload_access(tmp_path, monkeypatch):
    project = tmp_path/'project'; project.mkdir()
    parent_root = tmp_path/'parent'; parent_root.mkdir()
    p = parent_spec(parent_root)
    p['project_dir'] = str(tmp_path/'parent_source')
    p['anchor']['evidence']['prepared_root'] = str(tmp_path/'prepared')
    p['anchor']['evidence']['raw_container'] = str(tmp_path/'raw')
    p = with_content_hash(p); path = parent_root/'campaign_spec.json'; write_immutable_json(path, p)
    reused = c.artifact('REUSE', parents=dict(spec=p['content_hash']), spec=file_ref(path), anchor=p['anchor'])
    snapshot = dict(content_hash='b'*64)
    monkeypatch.setattr(source, 'current', lambda *a: snapshot)
    monkeypatch.setattr(source, 'import_parent', lambda *a: (reused, p))
    monkeypatch.setattr(source, 'validate_reuse', lambda _: p)
    monkeypatch.setattr(campaign, 'storage', lambda _: 0)
    for bad_root, quota, error in ((parent_root/'nested', 3, PermissionError), (project, 3, PermissionError),
                                    (tmp_path/'bad_quota', 2, ValueError)):
        with pytest.raises(error): campaign.create(bad_root, path, project, 'a'*40, available_quota_gib=quota)
    spec = campaign.create(tmp_path/'new', path, project, 'a'*40, available_quota_gib=3)
    campaign.check(spec)
    with pytest.raises(FileExistsError): campaign.create(tmp_path/'new', path, project, 'a'*40, available_quota_gib=3)
    monkeypatch.setattr(source, 'validate_reuse', lambda _: (_ for _ in ()).throw(AssertionError('historical access')))
    campaign.validate_spec(spec, imports=False)
    for field, value in (('preflight_jobs', 1), ('cpus', 8), ('available_quota_gib', 2), ('scientific_fits', 5)):
        bad = deepcopy(spec); bad[field] = value; bad = with_content_hash(bad)
        (tmp_path/'new/campaign_spec.json').write_text(json.dumps(bad))
        with pytest.raises(ValueError): campaign.validate_spec(bad, imports=False)


@pytest.mark.parametrize('bad', ['environment', 'source', 'teacher'])
def test_bad_execution_prerequisites_cannot_publish_any_outputs(tmp_path, monkeypatch, bad):
    tiny(monkeypatch); spec, _, _ = fixture(tmp_path, monkeypatch)
    if bad == 'environment':
        monkeypatch.setattr(worker, 'installed_environment', lambda: {})
    else:
        def fail(_): raise ValueError('Invalid '+bad)
        monkeypatch.setattr(campaign, 'check', fail)
    with pytest.raises(ValueError): worker.run(spec, 'targets', device='cpu')
    assert not (Path(spec['root'])/'targets').exists()


def test_retirement_pending_escape_and_partial_journal_fail_closed(tmp_path, monkeypatch):
    root = tmp_path/'new'; root.mkdir(); (root/'report').mkdir(); spec = bare(root)
    result = dict(pending=['k10'])
    with pytest.raises(ValueError): worker.retire(spec, result, '106')
    assert not (root/'report/results.json').exists()
    outside = tmp_path/'protected'; outside.write_bytes(b'keep')
    monkeypatch.setattr(worker, 'target_payloads', lambda: ['../protected'])
    with pytest.raises(PermissionError): worker.retire(spec, dict(pending=[]), '106')
    assert outside.read_bytes() == b'keep'
    assert not (root/'report/retirement.json').exists()
    write_immutable_json(root/'report/retirement.json', dict(partial=True))
    with pytest.raises(ValueError, match='Partial target retirement'): worker.retired_evidence(spec)


def test_source_guard_requires_complete_files_and_no_legacy_runtime_edits(monkeypatch):
    monkeypatch.setattr(source, '_source', lambda *a: dict(content_hash='a'*64))
    monkeypatch.setattr(subprocess, 'run', lambda args, **kw: SimpleNamespace(stdout='\n'.join(source.REQUIRED) if 'ls-files' in args else 'M\tsrc/old.py'))
    with pytest.raises(ValueError, match='runtime'): source.current(Path(__file__).resolve().parents[1], 'b'*40)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=''))
    with pytest.raises(ValueError, match='Commit'): source.current(Path(__file__).resolve().parents[1], 'b'*40)


def test_exact_ancestor_accounting_rejects_failure_or_wrong_job(monkeypatch):
    spec = bare(Path('/new'))
    monkeypatch.setattr(submission, 'accepted_jobs', lambda _: dict(targets='123'))
    monkeypatch.setattr(submission.getpass, 'getuser', lambda: 'ryreu')
    for line in ('', '123|lukaas_targets|ryreu|reu-aisocial|FAILED|1:0', '123|other|ryreu|reu-aisocial|COMPLETED|0:0'):
        monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=line))
        with pytest.raises(ValueError): submission.successful(spec, 'targets', '123')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='123|lukaas_targets|ryreu|reu-aisocial|COMPLETED|0:0'))
    submission.successful(spec, 'targets', '123')


def test_registered_shells_and_docs_complete():
    project = Path(__file__).resolve().parents[1]
    assert all((project/p).is_file() for p in source.REQUIRED)
    shell = (project/'sbatch/run_luka_auxiliary_logits_strong.sh').read_text()
    assert '${PROJECT_DIR}/sbatch/common.sh' in shell and 'PYTHONNOUSERSITE=1' in shell and '${CONDA_PREFIX}/lib' in shell
    helper = (project/'scripts/queue_luka_auxiliary_logits_strong.sh').read_text()
    assert c.AUTHORIZATION in helper and '</dev/tty' in helper and '--plan-hash' in helper
    assert 'scancel' not in helper and 'scontrol update' not in helper


def test_only_two_new_weights_and_no_old_contract_reinterpretation():
    from hlt_classification.luka_auxiliary_logits import contracts as previous
    assert c.WEIGHTS == {'k5': 5., 'k10': 10.}
    assert previous.WEIGHTS == {'k0': 0., 'k025': .25, 'k05': .5, 'k1': 1.}
    for arm in previous.WEIGHTS:
        with pytest.raises(ValueError): c.recipe(arm)
    for arm in c.WEIGHTS:
        with pytest.raises(ValueError): previous.recipe(arm)
        with pytest.raises(ValueError): previous.validate(c.recipe(arm), 'RECIPE')
        with pytest.raises(ValueError): c.validate(previous.recipe('k1'), 'RECIPE')


def test_high_weight_model_rng_and_exact_loss_gradient_scaling_against_original_k1():
    from hlt_classification.luka_auxiliary_logits import model as previous_model, training as previous_training
    spec = bare(Path('/new'))
    original_node = dict(spec['nodes']['k5'], arm='k1')
    original = previous_model.new_model(original_node)
    for arm, weight in c.WEIGHTS.items():
        student = model.new_model(spec['nodes'][arm])
        assert all(torch.equal(v, original.state_dict()[k]) for k, v in student.state_dict().items())
        x, y, p, z, q, targets = tensors()
        old_loss, _, _, _ = previous_training.objective(x, y, p, z, q, targets, arm='k1')
        new_loss, _, _, _ = training.objective(x, y, p, z, q, targets, arm=arm)
        parameters = [x, *p.values(), *z.values()]
        before = torch.autograd.grad(old_loss, parameters, retain_graph=True)
        after = torch.autograd.grad(new_loss, parameters)
        for index, (a, b) in enumerate(zip(after, before)):
            multiplier = weight if 1 <= index <= 4 else 1.
            torch.testing.assert_close(a, multiplier*b, rtol=2e-6, atol=1e-7)

"""FullStat cold replacement science, failure injection, and production-model witnesses."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import numpy as np
import pytest
import torch
from test_luka_attention import cache, tiny_config, parent_nodes, threads
from test_luka_fullstat_embedding import fixture as embedding_fixture
from test_luka_fullstat import fullstat, durable
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_gating import banks as shared
from hlt_classification.luka_gating.model import centered_gate_loss, log_gate
from hlt_classification.luka_gating_cold.training import new_student, initial_digest
from hlt_classification.luka_fullstat_replace_gate import (
    contracts as c, campaign, banks, training, worker, preflight, submission, source, checkpoints)


def fixture(root):
    old, tr, val, _ = embedding_fixture(root, export=False)
    g = c.artifact('GATE', root=str(root), project_dir='/project', source_commit='b'*40,
        source_snapshot=old['source_snapshot'], anchor=old['anchor'], available_result_bytes=24*2**30,
        **{**campaign.registration(dict(nodes=parent_nodes())),
           'counts': dict(train=len(tr), validation=len(val))})
    (root/'gate_spec.json').write_text(json.dumps(g))
    ctx, vctx = cache(coordinate='D033'), cache('validation', coordinate='D033')
    # Richer features differ but canonical identities, labels, support remain equal.
    for value in (ctx, vctx):
        value.blocks[0].features[:, 0] += .7
    return g, tr, val, ctx, vctx


def technical_teacher(root, monkeypatch):
    tiny_config(monkeypatch)
    g, tr, val, ctx, vctx = fixture(root)
    bank = banks.original(g, tr)
    worker.fit(root/'PRIV', g, g['nodes'][0], tr, val, ctx, vctx, bank, device='cpu', technical=True)
    banks.export(root/'targets', g, tr, ctx, root/'PRIV', technical=True, device='cpu')
    return g, tr, val, ctx, vctx, banks.Resident(root/'targets', g, tr, root/'PRIV', technical=True)


def test_policy_and_exact_primary_contrast():
    ns = c.nodes(dict(nodes=parent_nodes()))
    assert [n['node_id'] for n in ns] == list(c.NAMES)
    assert [n['gate_weight'] for n in ns] == [0., 0., .1]
    assert [n['context_coordinate'] for n in ns] == ['D033', None, None]
    assert [n['teacher'] for n in ns] == ['COARSE_D033', 'PRIV_D000', 'PRIV_D000']
    assert all(n['freeze_epochs'] == 0 and n['warm_start'] is None for n in ns)
    assert len({(n['initialization_seed'], n['sampler_seed']) for n in ns}) == 1
    assert c.recipe()['base']['maximum_passes'] == 20
    assert c.recipe()['gate']['blocks'] == [6, 7]
    assert c.recipe()['batch_size'] == 128 and c.recipe()['final_test_sealed']
    assert c.COUNTS == dict(train=3000000, validation=500000, final_test=1500000, unused=105325)
    assert len(c.tasks()) == 5
    a, b = dict(ns[1]), dict(ns[2])
    for k in ('node_id', 'gate_weight'):
        a.pop(k); b.pop(k)
    assert a == b


def test_cold_state_gate_context_and_loss(monkeypatch):
    tiny_config(monkeypatch)
    nodes = c.nodes(dict(nodes=parent_nodes()))
    assert len({initial_digest(new_student(n)) for n in nodes}) == 1
    with pytest.raises(ValueError):
        new_student(dict(nodes[2], warm_start='PRIV_D000'))
    model = new_student(nodes[2])
    raw = cache(coordinate='D000').batch(np.arange(3))
    tensors = {k: torch.from_numpy(raw[k]) for k in ('features', 'vectors', 'mask')}
    with pytest.raises(PermissionError):
        model(**tensors, context_features=tensors['features'], context_mask=tensors['mask'])
    factors = torch.randn(3, raw['mask'].shape[-1], 9, requires_grad=True)
    teacher = torch.randn_like(factors, requires_grad=True)
    mask = tensors['mask'][:, 0]
    a, b = log_gate(factors, mask), log_gate(teacher, mask)
    expected = []
    for i in range(3):
        n = int(mask[i].sum())
        x, y = a[i, :n, :n], b[i, :n, :n]
        expected.append(((x-x.mean(-1, keepdim=True))-(y-y.mean(-1, keepdim=True))).square().mean())
    loss = centered_gate_loss(factors, teacher, mask)
    torch.testing.assert_close(loss, torch.stack(expected).mean())
    loss.backward()
    assert factors.grad is not None and teacher.grad is None


def test_resident_bank_equal_to_export_and_scopes(tmp_path, monkeypatch):
    g, tr, val, ctx, vctx, bank = technical_teacher(tmp_path/'g', monkeypatch)
    assert bank.manifest['technical'] and bank.factors.dtype == np.float32
    assert not bank.factors.flags.writeable and not bank.probabilities.flags.writeable
    assert 'historical_test_evaluated' not in load_json(tmp_path/'g/targets/payload/manifest.json')
    ix = np.array([8, 1, 20, 8])
    actual = bank.factor_batch(ix, 16)
    for j, i in enumerate(ix):
        n = bank.lengths[i]
        np.testing.assert_array_equal(actual[j, :n], bank.factors[bank.offsets[i]:bank.offsets[i+1]])
        assert not actual[j, n:].any()
    with pytest.raises(ValueError, match='technical'):
        banks.Resident(tmp_path/'g/targets', g, tr, tmp_path/'g/PRIV')
    with pytest.raises(PermissionError):
        banks.original(g, val)
    with pytest.raises(PermissionError):
        banks.export(tmp_path/'bad', g, val, vctx, tmp_path/'g/PRIV', technical=True, device='cpu')
    with pytest.raises(ValueError, match='join'):
        training.train(tr, val, None, None, bank=bank, node=g['nodes'][2], device='cpu')
    changed = deepcopy(tr)
    changed.identities = changed.identities[::-1].copy()
    with pytest.raises(ValueError, match='population'):
        banks.Resident(tmp_path/'g/targets', g, changed, tmp_path/'g/PRIV', technical=True)
    path = tmp_path/'g/targets/payload/factors-0000000.npy'
    path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError, match='bytes differ|checksum'):
        banks.Resident(tmp_path/'g/targets', g, tr, tmp_path/'g/PRIV', technical=True)


def test_full20_fits_no_gate_stripping_and_poor_result_continues(tmp_path, monkeypatch):
    tiny_config(monkeypatch)
    g, tr, val, ctx, vctx = fixture(tmp_path/'g')
    original = banks.original(g, tr)
    worker.fit(tmp_path/'g/PRIV', g, g['nodes'][0], tr, val, ctx, vctx, original, device='cpu')
    banks.export(tmp_path/'g/targets', g, tr, ctx, tmp_path/'g/PRIV', technical=False, device='cpu')
    bank = banks.Resident(tmp_path/'g/targets', g, tr, tmp_path/'g/PRIV')
    for node in g['nodes'][1:]:
        store = checkpoints.EpochStore(tmp_path/(node['node_id']+'_epochs'), binding=dict(node=node['node_id']))
        out = tmp_path/node['node_id']
        worker.fit(out, g, node, tr, val, None, None, bank, device='cpu', store=store)
        report = worker.validate_fit(out, g, node, bank.manifest['content_hash'])
        assert report['passes'] == len(report['validation_history']) == 20
        assert report['validation_history'][-1]['learning_rate'] == pytest.approx(1.5e-5)
        assert report['validation_history'][-1]['update'] == 20
        saved = torch.load(out/'selected.pt', weights_only=True)
        assert any(k.startswith('gate.') for k in saved)
        restored = new_student(node)
        restored.load_state_dict(saved, strict=True)
        if node['node_id'] == 'REPLACE_GATE':
            assert any(r['losses']['gate'] > 0 for r in report['validation_history'])
        else:
            assert all(r['losses']['gate'] == 0 for r in report['validation_history'])
        row = checkpoints.check(store.root/'epoch-20.json', dict(node=node['node_id']))
        assert torch.load(row['payload']['path'], weights_only=True)['history'] == report['validation_history']
        for change in (dict(passes=30), dict(selected_state='native_only'), dict(final_test_accessed=True)):
            with pytest.raises((ValueError, PermissionError)):
                worker.report_valid(with_content_hash(dict(report, **change)), g, node, bank.manifest['content_hash'])


@pytest.mark.parametrize('role,coordinate', [('final_test', 'D000'), ('unused', 'D033'),
    ('train', 'OFFLINE'), ('validation', 'U100')])
def test_forbidden_cache_roles(role, coordinate):
    with pytest.raises(PermissionError):
        worker.data({}, role, coordinate)


def test_d033_slot_alignment_fails_closed(monkeypatch):
    tiny_config(monkeypatch)
    tr, ctx = cache(coordinate='D000'), cache(coordinate='D033')
    shared.paired(tr, ctx)
    ctx.labels = ctx.labels[::-1].copy()
    with pytest.raises(ValueError, match='Unaligned'):
        shared.paired(tr, ctx)


def test_exact_submission_and_uncertain_retry(tmp_path, monkeypatch):
    tiny_config(monkeypatch)
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    g, *_ = fixture(tmp_path/'g')
    monkeypatch.setattr(campaign, 'check_gate', lambda _: None)
    monkeypatch.setattr(submission, 'check_site', lambda _: None)
    calls = []
    def scheduler(cmd, **kw):
        assert not any(k.startswith(('SBATCH_', 'SLURM_')) for k in kw['env'])
        calls.append(cmd)
        return SimpleNamespace(stdout='123456;oscar\n')
    monkeypatch.setattr(exact.subprocess, 'run', scheduler)
    plan = submission.submit(g)
    assert not calls and plan['jobs'] == 1
    assert '--time=08:00:00' in plan['commands'][0]['command']
    with pytest.raises(PermissionError):
        submission.submit(g, execute=True, plan_hash='bad', authorization=c.AUTHORIZATION)
    for _ in range(2):
        submission.submit(g, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 1 and submission.accepted_jobs(g) == {'preflight': '123456'}
    spec = c.artifact('SPEC', root=str(tmp_path/'science'), gate_path=str(Path(g['root'])/'gate_spec.json'),
        project_dir='/project', minutes={t['task_id']: 900 for t in c.tasks()})
    science = submission.render(spec)
    assert science['jobs'] == 5
    assert '--dependency=afterok:${JOB_export_PRIV_D000}' in science['commands'][2]['command']
    assert '--dependency=afterok:${JOB_export_PRIV_D000}' in science['commands'][3]['command']
    assert '--partition=batch' in science['commands'][-1]['command']
    # A second fresh gate simulates ambiguous sbatch acceptance.
    g2, *_ = fixture(tmp_path/'g2')
    p2 = submission.submit(g2)
    def uncertain(*a, **kw):
        calls.append(a)
        raise RuntimeError('lost connection')
    monkeypatch.setattr(exact.subprocess, 'run', uncertain)
    with pytest.raises(RuntimeError):
        submission.submit(g2, execute=True, plan_hash=p2['content_hash'], authorization=c.AUTHORIZATION)
    with pytest.raises((FileExistsError, ValueError)):
        submission.submit(g2, execute=True, plan_hash=p2['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 2


def test_failed_gate_never_admits_science(tmp_path, monkeypatch):
    tiny_config(monkeypatch)
    g, *_ = fixture(tmp_path/'g')
    (Path(g['root'])/'preflight').mkdir()
    write_immutable_json(Path(g['root'])/'preflight/receipt.json', dict(job_id='7'))
    monkeypatch.setattr(campaign, 'check_gate', lambda _: None)
    monkeypatch.setattr(preflight, 'validate_report', lambda *a: None)
    monkeypatch.setattr(submission, 'accepted_jobs', lambda _: dict(preflight='7'))
    monkeypatch.setattr(campaign, 'preflight_accounting',
                        lambda _: (_ for _ in ()).throw(ValueError('FAILED accounting')))
    with pytest.raises(ValueError, match='FAILED'):
        campaign.science(g)
    assert not (Path(g['root'])/'science').exists()


def test_science_submission_exact_afterok_jobs(tmp_path, monkeypatch):
    tiny_config(monkeypatch)
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    g, *_ = fixture(tmp_path/'g')
    root = tmp_path/'science'
    root.mkdir()
    spec = c.artifact('SPEC', root=str(root), project_dir='/project', source_commit='b'*40,
        gate_path=str(Path(g['root'])/'gate_spec.json'),
        minutes={t['task_id']: 720 if t['kind'] == 'train' else 60 for t in c.tasks()})
    monkeypatch.setattr(campaign, 'check_spec', lambda _: (g, {}))
    monkeypatch.setattr(submission, 'check_site', lambda _: None)
    calls = []
    def submit(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(stdout=str(1000+len(calls))+'\n')
    monkeypatch.setattr(exact.subprocess, 'run', submit)
    plan = submission.submit(spec)
    assert not calls
    submission.submit(spec, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 5
    assert '--dependency=afterok:1001' in calls[1]
    assert '--dependency=afterok:1002' in calls[2] and '--dependency=afterok:1002' in calls[3]
    assert '--dependency=afterok:1001:1003:1004' in calls[4]
    submission.submit(spec, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 5


def test_resources_not_scores():
    g = dict(max_minutes=2880, memory_mb=180000, available_result_bytes=24*2**30)
    m = dict(fits={n: dict(train_seconds=1200., validation_seconds=60., other_seconds=20.,
                          checkpoint_bytes=20*2**20) for n in c.NAMES},
        authentication_seconds=10., load_seconds=20., export_seconds=100.,
        peak_rss_bytes=20*2**30, gpu_peak_bytes=8*2**30, gpu=dict(total_memory_bytes=48*2**30),
        bank_bytes=4*2**30, preflight_bytes=5*2**30)
    result = preflight.budgets(g, m)
    assert len(result[0]) == 5 and preflight.budgets(g, dict(m, accuracy=0.)) == result
    for change in (dict(peak_rss_bytes=170*2**30), dict(gpu_peak_bytes=47*2**30),
                   dict(export_seconds=float('nan')), dict(bank_bytes=30*2**30)):
        with pytest.raises(ValueError):
            preflight.budgets(g, dict(m, **change))


def test_source_and_shell(monkeypatch):
    project = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(source, '_source', lambda *a: dict(content_hash='a'*64))
    monkeypatch.setattr(source.subprocess, 'run', lambda cmd, **kw: SimpleNamespace(
        stdout='\n'.join(source.REQUIRED) if 'ls-files' in cmd else '\n'.join(source.RUNTIME)))
    assert source.current(project, 'b'*40)['content_hash'] == 'a'*64
    for path in source.REQUIRED:
        assert (project/path).is_file(), path
    monkeypatch.setattr(source.subprocess, 'run', lambda cmd, **kw: SimpleNamespace(
        stdout='\n'.join(source.REQUIRED) if 'ls-files' in cmd else 'src/hlt_classification/luka_gating/model.py'))
    with pytest.raises(ValueError, match='frozen'):
        source.current(project, 'b'*40)


def test_shell_and_document_blocks():
    import re
    project = Path(__file__).resolve().parents[1]
    bash = Path('C:/Program Files/Git/bin/bash.exe')
    if not bash.is_file():
        import shutil
        bash = shutil.which('bash')
    if not bash:
        pytest.skip('Bash unavailable')
    for name in ('scripts/queue_luka_fullstat_replace_gate.sh', 'sbatch/run_luka_fullstat_replace_gate.sh'):
        subprocess.run([str(bash), '-n', name], cwd=project, check=True, capture_output=True)
    text = (project/'docs/LUKA_FULLSTAT_REPLACE_GATE_RUNBOOK.md').read_text()
    for block in re.findall(r'```bash\n(.*?)```', text, re.S):
        subprocess.run([str(bash), '-n'], input=block, text=True, check=True, capture_output=True)
    for path in (project/'docs/LUKA_FULLSTAT_REPLACE_GATE_RUNBOOK.md',
                 project/'docs/plans/LUKA_FULLSTAT_REPLACE_GATE_PLAN.md',
                 project/'docs/contracts/LUKA_FULLSTAT_REPLACE_GATE.md'):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            assert (path.parent/target).is_file(), target
    worker_text = (project/'sbatch/run_luka_fullstat_replace_gate.sh').read_text()
    assert '${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh' in worker_text
    assert 'CUBLAS_WORKSPACE_CONFIG=:4096:8' in worker_text


@pytest.mark.parametrize('key,value', [('technical', False), ('teacher', 'COARSE_D033'),
                                     ('final_test_sealed', False), ('final_test_accessed', True)])
def test_resigned_target_scope_rejected(tmp_path, monkeypatch, key, value):
    g, tr, *_ = technical_teacher(tmp_path/'g', monkeypatch)
    path = tmp_path/'g/targets/manifest.json'
    path.write_text(json.dumps(with_content_hash(dict(load_json(path), **{key:value}))))
    with pytest.raises((ValueError, PermissionError)):
        banks.Resident(path.parent, g, tr, tmp_path/'g/PRIV', technical=True)


def test_fresh_gate_protected_roots_quota_policy_and_source(tmp_path, monkeypatch):
    parent_root = tmp_path/'original/science'
    parent_root.mkdir(parents=True)
    parent = dict(root=str(parent_root), project_dir=str(tmp_path/'old_source'), nodes=parent_nodes())
    path = parent_root/'campaign_spec.json'
    write_immutable_json(path, parent)
    anchor = c.artifact('ANCHOR', spec=file_ref(path), evidence=dict(cache_root=str(tmp_path/'cache'),
        foundation_root=str(tmp_path/'foundation'), raw_container=str(tmp_path/'raw')))
    snap = dict(content_hash='c'*64)
    monkeypatch.setattr(source, 'current', lambda *a: snap)
    monkeypatch.setattr(source, 'authenticate', lambda *a: (anchor, parent))
    project = tmp_path/'new_source'
    for root in (project/'output', parent_root/'output', tmp_path/'cache/new', tmp_path/'raw/new', tmp_path):
        with pytest.raises(PermissionError):
            campaign.create(root, path, project, 'b'*40, quota_gib=24)
    with pytest.raises(ValueError, match='24 GiB'):
        campaign.create(tmp_path/'small', path, project, 'b'*40, quota_gib=1)
    gate = campaign.create(tmp_path/'good', path, project, 'b'*40, quota_gib=24)
    assert campaign.check_gate(gate) == parent
    changed = deepcopy(gate)
    changed['nodes'][2]['gate_weight'] = .2
    with pytest.raises(ValueError, match='registration'):
        campaign.check_gate(with_content_hash(changed))
    monkeypatch.setattr(source, 'current', lambda *a: dict(content_hash='d'*64))
    with pytest.raises(ValueError, match='snapshot'):
        campaign.check_gate(gate)


def test_original_anchor_authentication_not_trial_reinterpretation(monkeypatch):
    original = dict(content_hash='a'*64, spec=dict(path='/original/campaign_spec.json'),
        evidence=dict(prepared='p'*64), logits=dict(path='/original/reduce_COARSE_D033/train/manifest.json'))
    parent = dict(nodes=parent_nodes())
    monkeypatch.setattr(source, 'authenticate_d033', lambda *a: (original, parent))
    anchor, result = source.authenticate('/original/campaign_spec.json', '/project')
    assert result == parent and anchor['original'] == original
    assert anchor['parents'] == {'original': 'a'*64}
    assert anchor['final_test_sealed'] and not anchor['final_test_accessed']
    assert 'historical_test_evaluated' not in anchor


def test_objective_matches_frozen_ce_kd_and_gate(monkeypatch):
    tiny_config(monkeypatch)
    nodes = c.nodes(dict(nodes=parent_nodes()))
    model = new_student(nodes[2])
    raw = cache(coordinate='D000').batch(np.arange(3))
    logits = shared.forward(model, raw, None, 'cpu')
    # Nonzero frozen teacher gate target; gradients must not enter the target.
    factors = np.ones((3, raw['mask'].shape[-1], 9), np.float32)*.2
    q = np.full((3, 11), 1/11, np.float32)
    bank = SimpleNamespace(probabilities=q, factor_batch=lambda i, w: factors)
    loss, main, aux = training.objective(model, logits, raw, bank, np.arange(3), nodes[2], 'cpu')
    f = torch.nn.functional
    expected = .25*f.cross_entropy(logits.float(), torch.from_numpy(raw['labels']))
    expected += .75*4*f.kl_div(f.log_softmax(logits.float()/2, dim=-1), torch.from_numpy(q), reduction='batchmean')
    torch.testing.assert_close(main, expected)
    torch.testing.assert_close(loss, expected+.1*aux)
    control = training.objective(model, logits, raw, bank, np.arange(3), nodes[1], 'cpu')
    torch.testing.assert_close(control[0], main)
    assert control[2].item() == 0.


def test_real_root_aligned_d033_d000(durable):
    output, meta = durable
    g = dict(memory_mb=180000, counts=meta[1]['counts'],
        anchor=dict(evidence=dict(prepared=meta[0]['content_hash'], cache_root=str(output))))
    tr = worker.data(g, 'train', 'D000', authenticated=meta)
    ctx = worker.data(g, 'train', 'D033', authenticated=meta)
    shared.paired(tr, ctx)
    assert len(tr) == len(ctx) == 264
    np.testing.assert_array_equal(tr.identities, ctx.identities)


def test_actual_weaver_cuda_preflight_and_three_full20_workers(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable; real Oscar gate still required')
    # Production architecture; fixture populations and scheduler identity ONLY.
    g, tr, val, ctx, vctx = fixture(tmp_path/'g')
    monkeypatch.setattr(campaign, 'check_gate', lambda _: None)
    monkeypatch.setattr(worker, 'execution', lambda *a: '77')
    monkeypatch.setattr(worker, 'data', lambda g, r, co, **kw:
                        {('train', 'D000'):tr, ('validation', 'D000'):val,
                         ('train', 'D033'):ctx, ('validation', 'D033'):vctx}[r, co])
    meta = (dict(summaries={r: {'D000':dict(maximum=4), 'D033':dict(maximum=4)}
                           for r in ('train', 'validation')}), None, None)
    monkeypatch.setattr(worker, 'metadata', lambda _: meta)
    monkeypatch.setattr(preflight, 'installed_environment', lambda: g['anchor']['evidence']['environment'])
    monkeypatch.setattr(preflight, 'gpu_identity', lambda: dict(name='FIXTURE L40S', total_memory_bytes=48*2**30))
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    monkeypatch.delenv('TORCH_CUDNN_V8_API_DISABLED', raising=False)
    def memory(_):
        return dict(rss_bytes=2**30, peak_rss_bytes=2**30, gpu_peak_bytes=int(torch.cuda.max_memory_allocated()))
    monkeypatch.setattr(preflight, 'memory_snapshot', memory)
    monkeypatch.setattr(worker, 'memory_snapshot', memory)
    report = preflight.run(g)
    assert report['full_population_training_measured'] and report['fresh_initializations_equal']
    monkeypatch.setattr(submission, 'accepted_jobs', lambda _: dict(preflight='77'))
    monkeypatch.setattr(campaign, 'preflight_accounting', lambda _: None)
    spec = campaign.science(g)
    for task in c.tasks():
        result = worker.run(spec, task['task_id'])
        assert result == worker.run(spec, task['task_id'])
    results = worker.results(spec, g)
    assert not results['pending'] and results['contrasts'] is not None
    assert results['final_test_sealed'] and not results['final_test_accessed']
    for key in ('technical_checks_passed', 'full_validation_measured', 'fresh_initializations_equal'):
        with pytest.raises(ValueError):
            preflight.validate_report(g, with_content_hash(dict(report, **{key:False})))
    path = Path(spec['root'])/'train_REPLACE_GATE/selected.pt'
    path.write_bytes(path.read_bytes()+b'bad')
    with pytest.raises(ValueError, match='bytes differ|checksum'):
        worker.results(spec, g)

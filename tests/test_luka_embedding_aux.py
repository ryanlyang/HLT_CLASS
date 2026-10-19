"""Real native-model and production-path fixtures; never pretend to be Slurm evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch

from test_luka_attention import cache, parent_nodes
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.data.cache_contracts import (load_json, with_content_hash, write_immutable_json,
    deterministic_npz_bytes, atomic_publish_bytes, load_npz_arrays)
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import _forward, train_kernel
from hlt_classification.luka_embedding_aux import (
    contracts as c, model as m, targets, training, worker, preflight, campaign, submission, source)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny(monkeypatch):
    from hlt_classification.jetclass2_delphes import model
    config = model.model_config()
    # Keep the actual contracted 128-dimensional tap while reducing block count.
    config.update(embed_dims=[128], pair_embed_dims=[4], num_heads=2, num_layers=1, num_cls_layers=1, fc_params=[])
    monkeypatch.setattr(model, 'model_config', lambda: deepcopy(config))


def node():
    return c.node(dict(nodes=parent_nodes()))


def gate_fixture(root, rows=22, partition='tier3'):
    anchor = c.artifact('ANCHOR', spec=dict(path='/parent/campaign_spec.json'), evidence=dict(
        prepared_root='/prepared', raw_container='/raw', prepared='a'*64, cache='b'*64,
        environment=dict(fixture=True), models={}))
    snapshot = dict(content_hash='c'*64)
    gate = c.artifact('GATE', parents=dict(source=snapshot['content_hash'], anchor=anchor['content_hash']),
        root=str(root), project_dir='/project', source_commit='f'*40, source_snapshot=snapshot,
        anchor=anchor, available_quota_gib=2,
        **{**campaign.registration(dict(nodes=parent_nodes()), partition),
           'counts':dict(train=rows, validation=rows), 'original50k_rows':rows})
    return gate


def bank_fixture(directory, tr=None):
    tr = cache(coordinate='D000') if tr is None else tr
    off = cache(coordinate='OFFLINE', rows=len(tr))
    teacher = DelphesParticleTransformer()
    save_state(directory/'teacher.pt', teacher.state_dict())
    ref = file_ref(directory/'teacher.pt')
    targets.export(directory/'bank', teacher, off, teacher_ref=ref, device='cpu')
    return tr, targets.Bank(directory/'bank', tr, teacher_ref=ref), ref


def test_fixed_science_registration_and_no_hidden_kd():
    r = c.recipe()
    assert r['ce_coefficient'] == 1 and r['auxiliary_coefficient'] == .3
    assert r['epochs'] == r['base']['maximum_passes'] == 30
    assert not r['warm_start'] and not r['logit_kd'] and not r['attention_kd']
    assert node()['coordinate'] == 'D000' and node()['context_coordinate'] is None and node()['teacher'] is None
    assert node()['initialization_seed'] == parent_nodes()[0]['initialization_seed']
    assert campaign.registration(dict(nodes=parent_nodes()), 'tier3')['execution_site']['partition'] == 'tier3'
    assert campaign.registration(dict(nodes=parent_nodes()), 'debug')['execution_site']['partition'] == 'debug'
    with pytest.raises(ValueError):
        campaign.registration(dict(nodes=parent_nodes()), 'tigris')
    bad = with_content_hash(dict(c.artifact('BANK'), final_test_accessed=True))
    with pytest.raises(PermissionError):
        c.validate(bad, 'BANK')


def test_rng_native_forward_head_removal_and_auxiliary_gradient(monkeypatch):
    tiny(monkeypatch)
    torch.manual_seed(node()['initialization_seed'])
    native = DelphesParticleTransformer()
    cpu_after = torch.get_rng_state().clone()
    cuda_after = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    augmented = m.new_model(node())
    assert torch.equal(torch.get_rng_state(), cpu_after)
    for a, b in zip(cuda_after, torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []):
        assert torch.equal(a, b)
    native.eval(); augmented.eval(); augmented.capture = True
    raw = cache(coordinate='D000').batch(np.arange(4))
    expected = _forward(native, raw, 'cpu', bf16=False)
    actual = _forward(augmented, raw, 'cpu', bf16=False)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert len(augmented.mod.fc._forward_pre_hooks) == 0
    labels = torch.from_numpy(raw['labels'])
    target = torch.ones((4, 128), requires_grad=True)
    loss, ce, aux = m.objective(actual, labels, augmented.prediction, target)
    torch.testing.assert_close(loss, ce+.3*aux)
    aux.backward()
    assert target.grad is None
    for prefix in ('mod.', 'predictor.'):
        assert any(p.grad is not None and torch.count_nonzero(p.grad) for n, p in augmented.named_parameters() if n.startswith(prefix))
    deploy = DelphesParticleTransformer()
    deploy.load_state_dict(augmented.native_state(), strict=True)
    assert not any(k.startswith('predictor.') for k in augmented.native_state())
    augmented.capture = False
    monkeypatch.setattr(augmented.predictor, 'forward', lambda *_: pytest.fail('Inference must not execute predictor'))
    _forward(augmented, raw, 'cpu', bf16=False)


def test_lambda_zero_matches_original_native_ce_epoch(tmp_path, monkeypatch):
    tiny(monkeypatch)
    tr, bank, _ = bank_fixture(tmp_path)
    val = cache('validation', coordinate='D000')
    torch.manual_seed(node()['initialization_seed'])
    native = DelphesParticleTransformer()
    expected, state = train_kernel(native, tr, val, node=node(), device='cpu', acceptance_passes=1,
                                   training_recipe=c.recipe()['base'])
    actual, produced = training.train(m.new_model(node()), tr, val, node=node(), bank=bank,
        device='cpu', technical=True, zero_control=True)
    assert actual['validation'] == expected['validation']
    assert state.keys() == produced.keys()
    for key in state:
        torch.testing.assert_close(state[key], produced[key], rtol=0, atol=0)
    with pytest.raises(PermissionError):
        training.train(m.new_model(node()), tr, val, node=node(), bank=bank, device='cpu', zero_control=True)


def test_normalization_train_only_and_bank_integrity(tmp_path, monkeypatch):
    tiny(monkeypatch)
    tr, bank, ref = bank_fixture(tmp_path)
    assert bank.targets.shape == (22, 128) and bank.targets.dtype == np.float32
    np.testing.assert_allclose(bank.targets.mean(0), 0, atol=1e-6)
    assert np.all(bank.targets.std(0) <= 1.00001)
    values = np.ones((10, 128), np.float32)
    normalized, mean, scale = targets.standardize(values)
    assert not normalized.any() and np.all(mean == 1) and np.all(scale == 1e-3)
    for wrong in (cache('validation', coordinate='OFFLINE'), cache(coordinate='D000')):
        with pytest.raises(PermissionError):
            targets.export(tmp_path/'forbidden', DelphesParticleTransformer(), wrong, teacher_ref=ref, device='cpu')
    assert not (tmp_path/'forbidden').exists()
    changed = deepcopy(tr); changed.identities = tr.identities[::-1].copy()
    with pytest.raises(ValueError, match='identity join'):
        targets.Bank(tmp_path/'bank', changed, teacher_ref=ref)
    path = tmp_path/'bank/targets.npz'
    path.write_bytes(path.read_bytes()+b'bad')
    with pytest.raises(ValueError, match='bytes differ'):
        targets.Bank(tmp_path/'bank', tr, teacher_ref=ref)


@pytest.mark.parametrize('key,value', [('role', 'validation'), ('coordinate', 'U100'), ('width', 127)])
def test_resigned_wrong_bank_semantics(tmp_path, monkeypatch, key, value):
    tiny(monkeypatch)
    tr, _, ref = bank_fixture(tmp_path)
    path = tmp_path/'bank/manifest.json'
    manifest = with_content_hash(dict(load_json(path), **{key:value}))
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        targets.Bank(tmp_path/'bank', tr, teacher_ref=ref)


def test_complete_fixed30_scientific_fit_and_roundtrip(tmp_path, monkeypatch):
    tiny(monkeypatch)
    tr, bank, _ = bank_fixture(tmp_path)
    gate = gate_fixture(tmp_path)
    val = cache('validation', coordinate='D000')
    result = worker.fit(tmp_path/'fit', gate, tr, val, bank, device='cpu', technical=False)
    assert not result['technical'] and result['checkpoint_probability_roundtrip']
    report, arrays = worker.validate_fit(tmp_path/'fit', gate, bank.manifest['content_hash'], technical=False)
    assert report['passes'] == len(report['validation_history']) == 30
    assert report['validation']['rows'] == len(val)
    assert arrays['probabilities'].shape == (22, 11)
    assert list(tmp_path.glob('fit/*.pt')) == [tmp_path/'fit/selected.pt']
    with pytest.raises(FileExistsError):
        worker.fit(tmp_path/'fit', gate, tr, val, bank, device='cpu', technical=False)
    changed = dict(report, passes=29)
    with pytest.raises(ValueError):
        worker.report_valid(with_content_hash(changed), gate, bank.manifest['content_hash'], technical=False)


def test_real_parity_and_full_support_stress_cpu(monkeypatch):
    tiny(monkeypatch)
    raw = cache(coordinate='D000').batch(np.arange(4))
    assert preflight.parity(raw, node(), 'cpu')['auxiliary_backbone_gradient']
    for backward in (False, True):
        role = 'train' if backward else 'validation'
        row = preflight.stress(cache(role, coordinate='D000'), node(), 'cpu', backward=backward)
        assert row['maximum'] == 4 and row['batch'] == 128 and row['steps'] == 3


def test_actual_installed_standard_weaver_cuda():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable; genuine remote preflight still required')
    raw = cache(coordinate='D000').batch(np.arange(4))
    for bf16 in (False, True):
        assert preflight.parity(raw, node(), 'cuda', bf16=bf16)['zero_control_update']


def test_resource_admission_not_scientific_quality(tmp_path):
    gate = gate_fixture(tmp_path)
    measured = dict(peak_rss_bytes=2**30, gpu_peak_bytes=2**30, gpu=dict(total_memory_bytes=40*2**30),
        timings=dict(authentication=10., load_HLT=20., train_epoch=60., validation_epoch=10., checkpoint_reporting=5.))
    assert preflight.admission(gate, measured) < 480
    for changed in (dict(peak_rss_bytes=100*2**30), dict(gpu_peak_bytes=39*2**30),
                    dict(timings={**measured['timings'], 'train_epoch':4000.})):
        with pytest.raises(ValueError, match='Resource'):
            preflight.admission(gate, {**measured, **changed})
    assert preflight.admission(gate, dict(measured, accuracy=0., auc=.01)) == preflight.admission(gate, measured)


@pytest.mark.parametrize('partition', ['debug', 'tier3'])
def test_exact_submission_and_no_duplicates(tmp_path, monkeypatch, partition):
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    gate = gate_fixture(tmp_path, partition=partition)
    write_immutable_json(tmp_path/'gate_spec.json', gate)
    monkeypatch.setattr(campaign, 'check_gate', lambda _: None)
    monkeypatch.setattr(submission, 'check_site', lambda _: None)
    calls = []
    def scheduler(cmd, **kwargs):
        assert not any(k.startswith(('SBATCH_', 'SLURM_')) for k in kwargs['env'])
        calls.append(cmd)
        return SimpleNamespace(stdout='14569;sporc\n')
    monkeypatch.setattr(exact.subprocess, 'run', scheduler)
    dry = submission.submit(gate)
    assert not calls and dry['jobs'] == 1
    assert '--partition='+partition in dry['commands'][0]['command']
    for h, auth in [('bad', c.AUTHORIZATION), (dry['content_hash'], 'bad')]:
        with pytest.raises(PermissionError):
            submission.submit(gate, execute=True, plan_hash=h, authorization=auth)
    submission.submit(gate, execute=True, plan_hash=dry['content_hash'], authorization=c.AUTHORIZATION)
    assert submission.accepted_jobs(gate) == {'preflight':'14569'}
    submission.submit(gate, execute=True, plan_hash=dry['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 1
    spec = c.artifact('SPEC', root=str(tmp_path/'science'), gate_path=str(tmp_path/'gate_spec.json'),
        project_dir='/project', train_minutes=120)
    plan = submission.render(spec)
    assert plan['jobs'] == 1 and plan['commands'][0]['task_id'] == 'fit'
    assert '--partition='+partition in plan['commands'][0]['command']
    with pytest.raises(ValueError):
        submission.render(dict(spec, train_minutes=1441))


def test_failed_accounting_cannot_admit_science(tmp_path, monkeypatch):
    gate = gate_fixture(tmp_path)
    (tmp_path/'preflight').mkdir()
    write_immutable_json(tmp_path/'preflight/receipt.json', dict(job_id='14569'))
    monkeypatch.setattr(campaign, 'check_gate', lambda _: None)
    monkeypatch.setattr(preflight, 'validate_report', lambda *a: None)
    monkeypatch.setattr(submission, 'accepted_jobs', lambda _: dict(preflight='14569'))
    monkeypatch.setattr(campaign, 'preflight_accounting', lambda _: (_ for _ in ()).throw(ValueError('accounting FAILED')))
    with pytest.raises(ValueError, match='accounting FAILED'):
        campaign.science(gate)
    assert not (tmp_path/'science').exists()


def test_fresh_root_and_exact_source_guards(tmp_path, monkeypatch):
    parent = dict(nodes=parent_nodes(), project_dir=str(tmp_path/'old_source'), root=str(tmp_path/'old_run/science'))
    anchor = c.artifact('ANCHOR', spec=dict(path=str(tmp_path/'old_run/science/campaign_spec.json')),
        evidence=dict(prepared_root=str(tmp_path/'prepared'), raw_container=str(tmp_path/'raw')))
    snapshot = dict(content_hash='a'*64)
    monkeypatch.setattr(source, 'current', lambda *a: snapshot)
    monkeypatch.setattr(source, 'authenticate', lambda _: (anchor, parent))
    args = dict(parent_path='parent', project=tmp_path/'new_source', commit='b'*40, available_quota_gib=2)
    for root in (tmp_path/'raw/new', tmp_path/'prepared/new', tmp_path/'old_run/new', tmp_path/'new_source/new'):
        with pytest.raises(PermissionError):
            campaign.create(root, **args)
        assert not root.exists()
    with pytest.raises(ValueError):
        campaign.create(tmp_path/'lowquota', **{**args, 'available_quota_gib':1})
    gate = campaign.create(tmp_path/'fresh', **args)
    assert campaign.check_gate(gate) == parent
    with pytest.raises(FileExistsError):
        campaign.create(tmp_path/'fresh', **args)
    monkeypatch.setattr(source, 'current', lambda *a: dict(content_hash='c'*64))
    with pytest.raises(ValueError, match='source'):
        campaign.check_gate(gate)


def test_current_source_allows_only_new_namespace(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(source, '_source', lambda *a: dict(content_hash='a'*64))
    def git(cmd, **kw):
        return SimpleNamespace(stdout='\n'.join(source.REQUIRED) if 'ls-files' in cmd else
                               'src/hlt_classification/jetclass2_delphes/runner.py\n')
    monkeypatch.setattr(source.subprocess, 'run', git)
    with pytest.raises(ValueError, match='Existing scientific runtime'):
        source.current(project, 'a'*40)
    monkeypatch.setattr(source.subprocess, 'run', lambda cmd, **kw: SimpleNamespace(
        stdout='\n'.join(source.REQUIRED) if 'ls-files' in cmd else '\n'.join(source.ADDITIONS)))
    assert source.current(project, 'a'*40)['content_hash'] == 'a'*64
    for path in source.REQUIRED:
        assert (project/path).is_file(), path


def test_original_parent_native_and_input_bytes_must_match(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    assert len(source.matched_core(project, project)) == len(source.MATCHED_CORE)
    actual = source.sha256_file
    monkeypatch.setattr(source, 'sha256_file', lambda p: 'f'*64 if Path(p).is_relative_to(tmp_path) else actual(p))
    with pytest.raises(ValueError, match='original parent'):
        source.matched_core(tmp_path, project)


def test_interrupted_submission_never_retries(tmp_path, monkeypatch):
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    gate = gate_fixture(tmp_path)
    monkeypatch.setattr(campaign, 'check_gate', lambda _: None)
    monkeypatch.setattr(submission, 'check_site', lambda _: None)
    dry = submission.submit(gate)
    calls = []
    def uncertain(cmd, **kw):
        calls.append(cmd)
        raise RuntimeError('connection lost after possible sbatch')
    monkeypatch.setattr(exact.subprocess, 'run', uncertain)
    with pytest.raises(RuntimeError):
        submission.submit(gate, execute=True, plan_hash=dry['content_hash'], authorization=c.AUTHORIZATION)
    with pytest.raises((FileExistsError, PermissionError, ValueError)):
        submission.submit(gate, execute=True, plan_hash=dry['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 1


def test_payload_semantic_corruption_and_no_partial_publication(tmp_path, monkeypatch):
    tiny(monkeypatch)
    tr, _, ref = bank_fixture(tmp_path)
    payload = tmp_path/'bank/targets.npz'
    arrays = load_npz_arrays(payload)
    arrays['targets'][0, 0] = np.nan
    payload.write_bytes(deterministic_npz_bytes(arrays))
    manifest_path = tmp_path/'bank/manifest.json'
    manifest = with_content_hash(dict(load_json(manifest_path), payload=file_ref(payload, root=payload.parent)))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Invalid targets'):
        targets.Bank(tmp_path/'bank', tr, teacher_ref=ref)
    monkeypatch.setattr(targets, 'head_input', lambda *a, **k: (_ for _ in ()).throw(ValueError('bad teacher')))
    with pytest.raises(ValueError, match='bad teacher'):
        targets.export(tmp_path/'partial', DelphesParticleTransformer(), cache(coordinate='OFFLINE'), teacher_ref=ref, device='cpu')
    assert not (tmp_path/'partial/manifest.json').exists()
    with pytest.raises(FileNotFoundError):
        targets.Bank(tmp_path/'partial', tr, teacher_ref=ref)


def test_parent_replay_is_ce_and_offline_only():
    assert "('M0HLT','OFFLINE')" in source.PARENT_CHECK
    assert 'COARSE_U100' not in source.PARENT_CHECK and 'attention' not in source.PARENT_CHECK
    assert "w.completed" in source.PARENT_CHECK and 'metadata(root)' in source.PARENT_CHECK
    assert 'predictions=' in source.PARENT_CHECK


def test_full_gpu_preflight_and_science_workers(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('Real local CUDA required for production worker fixture')
    # Actual unmodified production architecture, only the population/site is a fixture.
    tr, val, off = cache(coordinate='D000'), cache('validation', coordinate='D000'), cache(coordinate='OFFLINE')
    root = tmp_path/'gate'; root.mkdir()
    gate = gate_fixture(root)
    save_state(tmp_path/'teacher.pt', DelphesParticleTransformer().state_dict())
    teacher_ref = file_ref(tmp_path/'teacher.pt')
    predictions = dict(identities=val.identities, labels=val.labels,
                       probabilities=np.full((len(val), 11), 1/11, np.float32))
    atomic_publish_bytes(tmp_path/'parent_predictions.npz', deterministic_npz_bytes(predictions))
    gate['anchor']['evidence']['models'] = {n:dict(checkpoint=teacher_ref,
        predictions=file_ref(tmp_path/'parent_predictions.npz')) for n in ('M0HLT', 'OFFLINE')}
    gate['anchor'] = with_content_hash(gate['anchor'])
    gate['parents']['anchor'] = gate['anchor']['content_hash']
    gate = with_content_hash(gate)
    write_immutable_json(root/'gate_spec.json', gate)
    meta = ({}, dict(summaries={r:dict(D000=dict(maximum=4)) for r in ('train', 'validation')}), {}, {})
    monkeypatch.setattr(campaign, 'check_gate', lambda _: None)
    monkeypatch.setattr(worker, 'execution', lambda *a: '77')
    monkeypatch.setattr(worker, 'data', lambda g, r, co, meta: tr if r == 'train' else val)
    monkeypatch.setattr(preflight, 'installed_environment', lambda: gate['anchor']['evidence']['environment'])
    monkeypatch.setattr(source, 'current', lambda *a: gate['source_snapshot'])
    monkeypatch.setattr(preflight.regular_cache, 'metadata', lambda _: meta)
    monkeypatch.setattr(targets, 'offline_train', lambda *a, **k: off)
    monkeypatch.setattr(worker.alljets_data, 'comparison_indices', lambda *a: np.arange(len(val), dtype=np.int64))
    def memory(_torch):
        return dict(rss_bytes=2**30, peak_rss_bytes=2**30, gpu_peak_bytes=int(torch.cuda.max_memory_allocated()))
    monkeypatch.setattr(preflight, 'memory_snapshot', memory)
    monkeypatch.setattr(worker, 'memory_snapshot', memory)
    accepted = preflight.run(gate)
    assert accepted['technical_checks_passed'] and accepted['train_minutes'] <= 1440
    assert (root/'targets/manifest.json').is_file()
    monkeypatch.setattr(submission, 'accepted_jobs', lambda _: dict(preflight='77'))
    monkeypatch.setattr(campaign, 'preflight_accounting', lambda _: dict(state='COMPLETED', exit_code='0:0'))
    spec = campaign.science(gate)
    result = worker.run(spec)
    assert [r['model'] for r in result['rows']] == ['M0HLT', 'OFFLINE', 'HLT_EMBED_AUX']
    assert not result['independent_test_available']
    assert worker.results(spec) == result
    # Reading final comparison needs no old teacher/caches/raw data or model loader.
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('Results must not load models'))
    monkeypatch.setattr(source, 'authenticate', lambda *a: pytest.fail('Results must be compact/self-contained'))
    assert worker.results(spec) == result
    # Do not accept signed-but-incomplete acceptance evidence.
    bad = deepcopy(accepted)
    bad['outputs'] = [r for r in bad['outputs'] if r['path'] != 'preflight/references/M0HLT.npz']
    with pytest.raises(ValueError, match='coverage'):
        preflight.validate_report(gate, with_content_hash(bad))
    bad = with_content_hash(dict(accepted, technical_checks_passed=False))
    with pytest.raises(ValueError, match='acceptance'):
        preflight.validate_report(gate, bad)
    path = root/'science/fit/validation_predictions.npz'
    path.write_bytes(path.read_bytes()+b'corrupted')
    with pytest.raises(ValueError, match='bytes differ'):
        worker.results(spec)


# Real ROOT fixture rebuild: no durable OFFLINE payload is read or written.
from test_luka_fullsim_stage2 import dataset, prepared
from test_luka_fullsim_alljets import all_prepared


def test_rebuild_offline_train_exact_without_durable_cache(all_prepared, monkeypatch):
    old, manifest, p, inv, splits = all_prepared
    root = Path(old['root'])/'prepared'
    meta = (manifest, p, inv, splits)
    from hlt_classification.luka_fullsim import regular_cache
    expected = regular_cache.load(root, role='train', coordinate='OFFLINE', max_ram_bytes=2**30)
    gate = gate_fixture(root.parent/'unused_new_root')
    gate['anchor']['evidence'].update(prepared_root=str(root), raw_container=old['input_container'],
                                    prepared=p['content_hash'], cache=manifest['content_hash'])
    gate['counts'] = p['counts']
    selected = regular_cache.selected_file_rows
    seen = []
    def track(container, inventory, member):
        assert member in splits['memberships']['train']
        seen.append(member['file_index'])
        yield from selected(container, inventory, member)
    monkeypatch.setattr(regular_cache, 'selected_file_rows', track)
    monkeypatch.setattr(regular_cache, 'load', lambda *a, **kw: pytest.fail('Do not require deleted OFFLINE cache'))
    actual = targets.offline_train(gate, authenticated=meta)
    for k, v in expected.batch(np.arange(len(expected))).items():
        np.testing.assert_array_equal(v, actual.batch(np.arange(len(actual)))[k])
    assert seen == [x['file_index'] for x in splits['memberships']['train']]
    bad = deepcopy(meta); bad[1]['summaries']['train']['OFFLINE']['maximum'] += 1
    with pytest.raises(ValueError, match='fingerprint'):
        targets.offline_train(gate, authenticated=bad)

"""FullStat final-rung fixtures, including real installed-Weaver CUDA execution."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import numpy as np
import pytest
import torch
from test_luka_attention import cache, parent_nodes
from test_luka_embedding_aux import tiny, threads
from test_luka_fullstat import fullstat, durable
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.data.cache_contracts import (load_json, write_immutable_json, with_content_hash,
    atomic_publish_bytes, deterministic_npz_bytes, load_npz_arrays)
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.banks import publish_bank
from hlt_classification.luka_fullstat_embedding import (
    contracts as c, banks, campaign, source, submission, worker, preflight, training, checkpoints)


def node():
    return c.node(dict(nodes=parent_nodes()))


def fixture(root, *, export=True):
    root.mkdir()
    tr, val, off = cache(coordinate='D000'), cache('validation', coordinate='D000'), cache(coordinate='OFFLINE')
    teacher = DelphesParticleTransformer()
    save_state(root/'teacher.pt', teacher.state_dict())
    a = dict(identities=val.identities, labels=val.labels,
             probabilities=np.full((len(val), 11), 1/11, np.float32))
    atomic_publish_bytes(root/'parent_predictions.npz', deterministic_npz_bytes(a))
    report = with_content_hash(dict(validation=evaluate_probabilities(a['labels'], a['probabilities']), selected_pass=17))
    write_immutable_json(root/'parent_report.json', report)
    refs = dict(checkpoint=file_ref(root/'teacher.pt'), report=file_ref(root/'parent_report.json'),
                predictions=file_ref(root/'parent_predictions.npz'))
    publish_bank(root/'original_logits', foundation_sha256=tr.foundation_sha256,
        teacher_report_sha256=report['content_hash'], teacher_node='COARSE_D033', role='train',
        identities=tr.identities, probabilities=np.full((len(tr), 11), 1/11, np.float32), shard_rows=7)
    anchor = c.artifact('ANCHOR', spec=dict(path='/parent/campaign_spec.json'), evidence=dict(
        prepared=tr.foundation_sha256, cache_root='/cache', foundation_root='/foundation',
        raw_container='/raw', models={n: refs for n in c.REFERENCES},
        environment=dict(fixture=True), cudnn_v8_disabled='unset'),
        logits=file_ref(root/'original_logits/manifest.json'))
    snapshot = dict(content_hash='c'*64)
    gate = c.artifact('GATE', parents=dict(source=snapshot['content_hash'], anchor=anchor['content_hash']),
        root=str(root), project_dir='/project', source_commit='b'*40, source_snapshot=snapshot,
        anchor=anchor, available_result_bytes=12*2**30,
        **{**campaign.registration(dict(nodes=parent_nodes())),
           'counts':dict(train=len(tr), validation=len(val))})
    write_immutable_json(root/'gate_spec.json', gate)
    if export:
        banks.export(root/'targets', teacher, off, refs=refs, device='cpu')
    return gate, tr, val, off


def test_policy_fullstat_not_trial_or_attention():
    from hlt_classification.luka_fullstat.recipe import recipe
    old = next(n for n in parent_nodes() if n['node_id'] == 'COARSE_D000')
    actual = node()
    assert actual == dict(old, node_id='EMBED_D000', branch='EMBED', embedding_coefficient=.3)
    assert c.recipe()['base'] == recipe(128)
    assert c.recipe()['epochs'] == 20 and not c.recipe()['attention']
    assert c.recipe()['embedding_teacher'] == 'OFFLINE'
    assert c.recipe()['teacher'] == 'COARSE_D033'
    assert c.COUNTS == dict(train=3000000, validation=500000, final_test=1500000, unused=105325)
    assert c.recipe()['final_test_sealed'] and not c.recipe()['final_test_accessed']
    assert len(c.tasks()) == 2
    assert [x['task_id'] for x in c.tasks()] == ['train_EMBED_D000', 'aggregate']
    assert 'historical_test_repurposed_for_training' not in c.recipe()


def test_bank_export_join_and_no_forbidden_access(tmp_path, monkeypatch):
    tiny(monkeypatch)
    g, tr, val, off = fixture(tmp_path/'g')
    bank = banks.load(g, tr)
    banks.check_join(bank, tr)
    assert bank.targets.shape == (22, 128) and bank.targets.dtype == np.float32
    assert not bank.targets.flags.writeable and not bank.probabilities.flags.writeable
    np.testing.assert_allclose(bank.targets.mean(0), 0., atol=1e-6)
    for data in (val, off):
        with pytest.raises(PermissionError):
            banks.load(g, data)
    changed = deepcopy(tr)
    changed.identities = changed.identities[::-1].copy()
    with pytest.raises(ValueError, match='join'):
        banks.load(g, changed)
    for data in (val, tr):
        with pytest.raises(PermissionError):
            banks.export(tmp_path/'bad', DelphesParticleTransformer(), data,
                         refs=g['anchor']['evidence']['models']['OFFLINE'], device='cpu')
    assert not (tmp_path/'bad').exists()
    path = Path(g['root'])/'original_logits/000000000.npz'
    path.write_bytes(path.read_bytes()+b'bad')
    with pytest.raises(ValueError, match='checksum'):
        banks.load(g, tr)


@pytest.mark.parametrize('key,value', [('teacher', 'COARSE_D033'), ('coordinate', 'U100'), ('role', 'validation'),
                                     ('width', 127), ('rows', 21), ('final_test_accessed', True)])
def test_resigned_bank_rejected(tmp_path, monkeypatch, key, value):
    tiny(monkeypatch)
    g, tr, *_ = fixture(tmp_path/'g')
    path = Path(g['root'])/'targets/manifest.json'
    path.write_text(json.dumps(with_content_hash(dict(load_json(path), **{key:value}))))
    with pytest.raises((ValueError, PermissionError)):
        banks.load(g, tr)


def test_normalization_corrupt_payload_and_wrong_teacher(tmp_path, monkeypatch):
    tiny(monkeypatch)
    g, tr, _, off = fixture(tmp_path/'g')
    root = Path(g['root'])
    path = root/'targets/targets.npz'
    a = load_npz_arrays(path)
    a['scale'][0] = 0.
    path.write_bytes(deterministic_npz_bytes(a))
    manifest = load_json(root/'targets/manifest.json')
    (root/'targets/manifest.json').write_text(json.dumps(with_content_hash(dict(manifest,
        payload=file_ref(path, root=path.parent)))))
    with pytest.raises(ValueError, match='normalization'):
        banks.load(g, tr)
    teacher = DelphesParticleTransformer()
    with pytest.raises(ValueError, match='weights differ'):
        banks.export(root/'wrong_teacher', teacher, off,
            refs=g['anchor']['evidence']['models']['OFFLINE'], device='cpu')
    assert not (root/'wrong_teacher').exists()


def test_zero_aux_native_kd_exact_and_fixed20_fit(tmp_path, monkeypatch):
    tiny(monkeypatch)
    g, tr, val, _ = fixture(tmp_path/'g')
    bank = banks.load(g, tr)
    assert preflight.zero_parity(tr, val, bank, node(), 'cpu')
    with pytest.raises(PermissionError):
        training.train(tr, val, node=node(), bank=bank, device='cpu', zero_control=True)
    monkeypatch.setattr(worker, 'validation_arrays', lambda _: dict(identities=val.identities, labels=val.labels))
    store = checkpoints.EpochStore(tmp_path/'epochs', binding=dict(gate=g['content_hash'], bank=bank.manifest['content_hash']))
    worker.fit(tmp_path/'fit', g, tr, val, bank, device='cpu', store=store)
    report = worker.validate_fit(tmp_path/'fit', g, bank.manifest['content_hash'])
    assert report['passes'] == len(report['validation_history']) == 20
    assert report['validation_history'][-1]['learning_rate'] == pytest.approx(1.5e-5)
    assert report['validation_history'][-1]['update'] == 20
    assert all(r['losses']['auxiliary'] > 0 for r in report['validation_history'])
    assert len(list((tmp_path/'epochs').glob('*.json'))) == 20
    epoch = checkpoints.check(tmp_path/'epochs/epoch-20.json', store.binding)
    state = torch.load(epoch['payload']['path'], weights_only=True)
    assert state['epoch'] == 20 and state['history'] == report['validation_history']
    assert any(k.startswith('predictor.') for k in state['model'])
    assert not any(k.startswith('predictor.') for k in state['best_state'])
    for key, value in [('passes', 30), ('technical', True), ('selected_pass', 0), ('final_test_accessed', True)]:
        with pytest.raises((ValueError, PermissionError)):
            worker.report_valid(with_content_hash(dict(report, **{key:value})), g, bank.manifest['content_hash'])
    with pytest.raises(FileExistsError):
        worker.fit(tmp_path/'fit', g, tr, val, bank, device='cpu')
    with pytest.raises(ValueError):
        checkpoints.check(tmp_path/'epochs/epoch-20.json', dict(gate='wrong'))


def test_stress_all_views_cpu(tmp_path, monkeypatch):
    tiny(monkeypatch)
    for role, co, backward, off in [('train', 'D000', True, False),
                                   ('validation', 'D000', False, False), ('train', 'OFFLINE', False, True)]:
        row = preflight.stress(cache(role, coordinate=co), node(), 'cpu', backward=backward, offline=off)
        assert row == dict(role=role, coordinate=co, maximum=4, batch=128,
                           steps=3, backward=backward, fp32_export=off)


def test_resources_not_accuracy(tmp_path, monkeypatch):
    tiny(monkeypatch)
    g, *_ = fixture(tmp_path/'g')
    m = dict(peak_rss_bytes=2**30, gpu_peak_bytes=2**30, epoch_checkpoint_bytes=10*2**20,
        gpu=dict(total_memory_bytes=48*2**30), timings=dict(authentication=10., load_HLT_targets=20.,
            train_epoch=1200., validation_epoch=60., checkpoint_reporting=10.))
    request = preflight.admission(g, m)
    assert 720 <= request <= 2880
    assert preflight.admission(g, dict(m, accuracy=0., auc=0.)) == request
    for change in (dict(peak_rss_bytes=170*2**30), dict(gpu_peak_bytes=47*2**30),
                   dict(epoch_checkpoint_bytes=2**30),
                   dict(timings={**m['timings'], 'train_epoch':5000.}),
                   dict(timings={**m['timings'], 'validation_epoch':float('nan')})):
        with pytest.raises(ValueError):
            preflight.admission(g, {**m, **change})


def test_exact_dry_submission_and_no_duplicates(tmp_path, monkeypatch):
    tiny(monkeypatch)
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
    command = plan['commands'][0]['command']
    assert '--time=08:00:00' in command and '--gres=gpu:l40s:1' in command
    for h, a in [('bad', c.AUTHORIZATION), (plan['content_hash'], 'bad')]:
        with pytest.raises(PermissionError):
            submission.submit(g, execute=True, plan_hash=h, authorization=a)
    for _ in range(2):
        submission.submit(g, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 1 and submission.accepted_jobs(g) == {'preflight':'123456'}
    spec = c.artifact('SPEC', root=str(tmp_path/'science'), gate_path=str(Path(g['root'])/'gate_spec.json'),
                      project_dir='/project', train_minutes=900)
    science = submission.render(spec)
    assert science['jobs'] == 2
    assert '--dependency=afterok:${JOB_train_EMBED_D000}' in science['commands'][1]['command']
    assert '--partition=batch' in science['commands'][1]['command']
    assert '--time=15:00:00' in science['commands'][0]['command']
    with pytest.raises(ValueError):
        submission.render(dict(spec, train_minutes=2881))


def test_uncertain_submit_no_retry(tmp_path, monkeypatch):
    tiny(monkeypatch)
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    g, *_ = fixture(tmp_path/'g')
    monkeypatch.setattr(campaign, 'check_gate', lambda _: None)
    monkeypatch.setattr(submission, 'check_site', lambda _: None)
    plan = submission.submit(g)
    calls = []
    def uncertain(*a, **kw):
        calls.append(a)
        raise RuntimeError('lost connection')
    monkeypatch.setattr(exact.subprocess, 'run', uncertain)
    with pytest.raises(RuntimeError):
        submission.submit(g, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    with pytest.raises((FileExistsError, ValueError, PermissionError)):
        submission.submit(g, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 1


def test_failed_preflight_never_creates_science(tmp_path, monkeypatch):
    tiny(monkeypatch)
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


def test_fresh_gate_source_quota_and_protected_roots(tmp_path, monkeypatch):
    parent_root = tmp_path/'original/science'
    parent_root.mkdir(parents=True)
    parent = dict(root=str(parent_root), project_dir=str(tmp_path/'old_source'), nodes=parent_nodes())
    path = parent_root/'campaign_spec.json'
    write_immutable_json(path, parent)
    anchor = c.artifact('ANCHOR', original=dict(spec=file_ref(path)), spec=file_ref(path),
        evidence=dict(cache_root=str(tmp_path/'cache'), foundation_root=str(tmp_path/'foundation'),
                      raw_container=str(tmp_path/'raw')))
    snap = dict(content_hash='c'*64)
    monkeypatch.setattr(source, 'current', lambda *a: snap)
    monkeypatch.setattr(source, 'authenticate', lambda *a: (anchor, parent))
    project = tmp_path/'new_source'
    for root in (project/'output', parent_root/'output', tmp_path/'raw/new',
                 tmp_path/'cache/new', tmp_path/'foundation/new', tmp_path):
        with pytest.raises(PermissionError):
            campaign.create(root, path, project, 'b'*40, quota_gib=12)
    with pytest.raises(ValueError, match='12 GiB'):
        campaign.create(tmp_path/'too_small', path, project, 'b'*40, quota_gib=2)
    gate = campaign.create(tmp_path/'new', path, project, 'b'*40, quota_gib=12)
    assert campaign.check_gate(gate) == parent
    with pytest.raises(FileExistsError):
        campaign.create(tmp_path/'new', path, project, 'b'*40, quota_gib=12)
    monkeypatch.setattr(source, 'current', lambda *a: dict(content_hash='d'*64))
    with pytest.raises(ValueError, match='snapshot'):
        campaign.check_gate(gate)


def test_original_d033_bank_binding(tmp_path, monkeypatch):
    tiny(monkeypatch)
    g, tr, *_ = fixture(tmp_path/'g')
    root = Path(g['root'])
    report = load_json(g['anchor']['evidence']['models']['COARSE_D033']['report']['path'])
    manifest = publish_bank(root/'reduce_COARSE_D033/train', foundation_sha256=tr.foundation_sha256,
        teacher_report_sha256=report['content_hash'], teacher_node='COARSE_D033', role='train',
        identities=tr.identities, probabilities=np.full((len(tr), 11), 1/11, np.float32))
    parent = dict(root=str(root), prepared_sha256=tr.foundation_sha256, counts=dict(train=len(tr)))
    original = c.artifact('FAKE_PARENT', spec=g['anchor']['spec'], evidence=g['anchor']['evidence'])
    monkeypatch.setattr(source, 'parent_authenticate', lambda *a: (original, parent))
    anchor, returned = source.authenticate('fixture', 'project')
    assert returned == parent and anchor['logits'] == file_ref(root/'reduce_COARSE_D033/train/manifest.json')
    mpath = root/'reduce_COARSE_D033/train/manifest.json'
    for key, value in [('role', 'validation'), ('temperature', 1.), ('teacher_node', 'COARSE_U100'),
                       ('teacher_report_sha256', 'b'*64), ('foundation_sha256', 'b'*64)]:
        mpath.write_text(json.dumps(with_content_hash(dict(manifest, **{key:value}))))
        with pytest.raises(ValueError):
            source.authenticate('fixture', 'project')


@pytest.mark.parametrize('role,coordinate', [('final_test', 'D000'), ('unused', 'D000'),
    ('validation', 'OFFLINE'), ('train', 'D033'), ('train', 'U100')])
def test_no_forbidden_cache_access(role, coordinate, monkeypatch):
    monkeypatch.setattr(worker, 'metadata', lambda _: pytest.fail('Forbidden read reached metadata'))
    with pytest.raises(PermissionError):
        worker.data({}, role, coordinate)


def test_technical_supervision_and_wrong_labels_blocked(tmp_path, monkeypatch):
    tiny(monkeypatch)
    g, tr, val, _ = fixture(tmp_path/'g')
    bank = banks.load(g, tr)
    original = bank.labels
    bank.labels = original[::-1].copy()
    with pytest.raises(ValueError, match='join'):
        banks.check_join(bank, tr)
    bank.labels = original
    bank.manifest = c.artifact('TECHNICAL_SUPERVISION', parents=bank.manifest['parents'],
                              prepared=tr.foundation_sha256, role='train', rows=len(tr))
    with pytest.raises(PermissionError, match='Technical'):
        training.train(tr, val, node=node(), bank=bank, device='cpu')


def test_real_root_prepared_cache_to_embedding_bank(durable, tmp_path, monkeypatch):
    tiny(monkeypatch)
    output, meta = durable
    g = dict(memory_mb=180000, counts=meta[1]['counts'],
        anchor=dict(evidence=dict(prepared=meta[0]['content_hash'], cache_root=str(output))))
    monkeypatch.setattr(worker, 'metadata', lambda _: meta)
    tr = worker.data(g, 'train', 'D000')
    off = worker.data(g, 'train', 'OFFLINE')
    val = worker.data(g, 'validation', 'D000')
    assert len(tr) == len(off) == 264 and len(val) == 66
    np.testing.assert_array_equal(tr.identities, off.identities)
    np.testing.assert_array_equal(tr.labels, off.labels)
    teacher = DelphesParticleTransformer()
    save_state(tmp_path/'teacher.pt', teacher.state_dict())
    write_immutable_json(tmp_path/'report.json', with_content_hash(dict(selected_pass=14)))
    refs = dict(checkpoint=file_ref(tmp_path/'teacher.pt'), report=file_ref(tmp_path/'report.json'))
    banks.export(tmp_path/'targets', teacher, off, refs=refs, device='cpu')
    bank = banks.Bank(tmp_path/'targets', tr, refs=refs)
    assert bank.targets.shape == (264, 128)
    np.testing.assert_allclose(bank.targets.mean(0), 0., atol=1e-6)


def test_source_complete_additive_only(monkeypatch):
    project = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(source, '_source', lambda *a: dict(content_hash='a'*64))
    def git(cmd, **kw):
        return SimpleNamespace(stdout='\n'.join(source.REQUIRED) if 'ls-files' in cmd else
                               'src/hlt_classification/jetclass2_delphes/runner.py')
    monkeypatch.setattr(source.subprocess, 'run', git)
    with pytest.raises(ValueError, match='frozen'):
        source.current(project, 'b'*40)
    monkeypatch.setattr(source.subprocess, 'run', lambda cmd, **kw: SimpleNamespace(
        stdout='\n'.join(source.REQUIRED) if 'ls-files' in cmd else '\n'.join(source.RUNTIME)))
    assert source.current(project, 'b'*40)['content_hash'] == 'a'*64
    for path in source.REQUIRED:
        assert (project/path).is_file(), path


def test_entrypoint_and_shell_syntax():
    project = Path(__file__).resolve().parents[1]
    bash = Path('C:/Program Files/Git/bin/bash.exe')
    if not bash.is_file():
        import shutil
        bash = shutil.which('bash')
    if not bash:
        pytest.skip('Bash unavailable')
    for file in ('scripts/queue_luka_fullstat_embedding.sh', 'sbatch/run_luka_fullstat_embedding.sh'):
        subprocess.run([str(bash), '-n', file], cwd=project, check=True, capture_output=True)
    text = (project/'sbatch/run_luka_fullstat_embedding.sh').read_text()
    assert '${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh' in text
    assert 'CUBLAS_WORKSPACE_CONFIG=:4096:8' in text


def test_actual_installed_weaver_cuda_preflight_and_full20_workers(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable; real Oscar preflight still required')
    # Unmodified production architecture. Only population/site/lineage replay use fixtures.
    g, tr, val, off = fixture(tmp_path/'g', export=False)
    monkeypatch.setattr(campaign, 'check_gate', lambda _: None)
    monkeypatch.setattr(worker, 'execution', lambda *a: '77')
    monkeypatch.setattr(worker, 'data', lambda g, r, co, **kw: off if co == 'OFFLINE' else tr if r == 'train' else val)
    meta = (dict(content_hash=tr.foundation_sha256,
        summaries=dict(train=dict(D000=dict(maximum=4), OFFLINE=dict(maximum=4)),
                       validation=dict(D000=dict(maximum=4)))), None, None)
    monkeypatch.setattr(worker, 'metadata', lambda _: meta)
    monkeypatch.setattr(preflight, 'installed_environment', lambda: g['anchor']['evidence']['environment'])
    monkeypatch.setattr(preflight, 'gpu_identity', lambda: dict(name='FIXTURE NVIDIA L40S',
                        total_memory_bytes=48*2**30, compute_capability=[8,9]))
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    monkeypatch.delenv('TORCH_CUDNN_V8_API_DISABLED', raising=False)
    def memory(_torch):
        return dict(rss_bytes=2**30, peak_rss_bytes=2**30, gpu_peak_bytes=int(torch.cuda.max_memory_allocated()))
    monkeypatch.setattr(preflight, 'memory_snapshot', memory)
    monkeypatch.setattr(worker, 'memory_snapshot', memory)
    report = preflight.run(g)
    assert report['technical_checks_passed'] and report['zero_aux_native_kd_parity']
    assert report['full_population_training_measured'] and report['full_validation_measured']
    monkeypatch.setattr(submission, 'accepted_jobs', lambda _: dict(preflight='77'))
    monkeypatch.setattr(campaign, 'preflight_accounting', lambda _: None)
    spec = campaign.science(g)
    fit = worker.run(spec, 'train_EMBED_D000')
    assert worker.run(spec, 'train_EMBED_D000') == fit
    result = worker.run(spec, 'aggregate')
    assert worker.run(spec, 'aggregate') == result
    assert result['result'] == worker.results(spec, g)
    assert [r['node_id'] for r in result['result']['rows']] == [*c.REFERENCES, 'EMBED_D000']
    assert result['result']['final_test_sealed'] and not result['result']['final_test_accessed']
    for change in (dict(technical_checks_passed=False), dict(full_population_training_measured=False),
                   dict(train_minutes=2879)):
        with pytest.raises(ValueError):
            preflight.validate_report(g, with_content_hash(dict(report, **change)))
    bad = deepcopy(report)
    bad['outputs'] = [r for r in bad['outputs'] if r['path'] != 'preflight/runtime.json']
    with pytest.raises(ValueError, match='coverage'):
        preflight.validate_report(g, with_content_hash(bad))
    pred = Path(spec['root'])/'train_EMBED_D000/validation_predictions.npz'
    pred.write_bytes(pred.read_bytes()+b'corrupt')
    with pytest.raises(ValueError):
        worker.results(spec, g)

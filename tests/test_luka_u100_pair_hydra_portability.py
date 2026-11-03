"""V2 admission reuses authenticated inference, not a substitute resource gate."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, deterministic_npz_bytes, write_immutable_json, load_json, load_npz_arrays)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_u100_pair_hydra import (
    contracts as c, portability as p, precision, campaign, worker)
from test_luka_u100_pair_correction import fixture, threads


@pytest.fixture
def evidence_fixture(tmp_path, monkeypatch):
    _, _, val, *_ = fixture(tmp_path)
    root = tmp_path/'diagnostic'; folder = root/'precision'; folder.mkdir(parents=True)
    (root/'submission').mkdir(); (root/'submission/ledger.json').write_text('fixture')
    bundle_root = tmp_path/'bundle'; bundle_root.mkdir()
    saved = np.full((len(val), 11), 1/11, dtype=np.float32)
    full = saved.copy(); full[:, 0] += .01; full[:, 1] -= .01
    def pack(probs): return dict(identities=val.identities.copy(), labels=val.labels.copy(), probabilities=probs)
    paths = {'reference/matched/checkpoint.pt':bundle_root/'checkpoint.pt',
             'reference/matched/predictions.npz':bundle_root/'predictions.npz'}
    paths['reference/matched/checkpoint.pt'].write_bytes(b'authenticated synthetic checkpoint')
    atomic_publish_bytes(paths['reference/matched/predictions.npz'], deterministic_npz_bytes(pack(saved)))
    reference = dict(job_id='21993056', environment={}, content_hash='e'*64)
    original = c.artifact('PRECISION_SPEC', parents=dict(transfer=c.MANIFEST_HASH),
        root=str(root), project_dir=str(tmp_path/'original_code'), bundle_root=str(bundle_root),
        source_commit=p.DIAGNOSTIC_COMMIT, source_snapshot=dict(content_hash='a'*64),
        reference=reference, counts=dict(train=22, validation=22), execution_site=c.SITE)
    write_immutable_json(root/'campaign_spec.json', original)
    monkeypatch.setattr(precision, '_predict_batches', lambda *a, **k: full.copy())
    probe, arrays = precision.probe(None, val, full, saved)
    metrics = precision.full_comparison(val.labels, full, saved); metrics['settings'] = precision.backend_settings()
    atomic_publish_bytes(folder/'fixed_batch_predictions.npz', deterministic_npz_bytes(arrays))
    atomic_publish_bytes(folder/'hydra_validation_predictions.npz', deterministic_npz_bytes(pack(full)))
    write_immutable_json(folder/'full_validation.json', c.artifact('PRECISION_FULL',
        parents=dict(spec=original['content_hash']), **metrics))
    report = c.artifact('PRECISION_REPORT', parents=dict(spec=original['content_hash']),
        checkpoint=file_ref(paths['reference/matched/checkpoint.pt']),
        saved_predictions=file_ref(paths['reference/matched/predictions.npz']),
        environment={}, gpu=dict(name=c.SITE['gpu_name']), full_validation=metrics, probe=probe,
        admission=False, scientific_fits_released=False, reference_retrained=False,
        outputs=[file_ref(folder/n, root=folder) for n in (
            'hydra_validation_predictions.npz', 'full_validation.json', 'fixed_batch_predictions.npz')])
    write_immutable_json(folder/'report.json', report)
    receipt = c.artifact('PRECISION_RECEIPT', parents=dict(spec=original['content_hash'], report=report['content_hash']),
        job_id=p.DIAGNOSTIC_JOB, report=file_ref(folder/'report.json', root=folder))
    write_immutable_json(folder/'receipt.json', receipt)
    bundle = SimpleNamespace(root=bundle_root, path=lambda name: paths[name])
    monkeypatch.setattr(p, 'Bundle', lambda _: bundle)
    monkeypatch.setattr(p, '_source', lambda *a: original['source_snapshot'])
    calls = []
    def replay(path, project, program, marker):
        calls.append((path, project, program, marker))
        return file_ref(path), dict(job=p.DIAGNOSTIC_JOB, report_hash=report['content_hash'])
    monkeypatch.setattr(p, 'replay', replay)
    evidence = p.authenticate(root/'campaign_spec.json', bundle_root, reference, original['counts'])
    return SimpleNamespace(root=root, folder=folder, bundle=bundle, paths=paths, reference=reference,
        original=original, report=report, arrays=arrays, full=pack(full), saved=pack(saved), val=val,
        evidence=evidence, calls=calls)


def test_original_replay_and_qualified_bf16_shift(evidence_fixture):
    f = evidence_fixture
    assert len(f.calls) == 1 and 'results(spec)' in f.calls[0][2]
    report, summary = p.validate_evidence(f.evidence, f.bundle.root, f.reference, f.original['counts'])
    assert report == f.report and summary['cpu_gpu_fp32']['max_abs'] == 0
    assert summary['full_validation']['differences']['max_abs'] > c.PARITY['max_abs']
    assert summary['full_validation']['differences']['rms'] > c.PARITY['rms']
    assert f.evidence['schema_version'] == 2 and f.evidence['policy'] == p.policy()
    assert c.PARITY == dict(max_abs=.005, rms=.0002, auc_abs=.00005)


def test_setup_binds_same_gpu_backend_and_population(evidence_fixture, monkeypatch):
    from hlt_classification.jetclass2_delphes import execution
    f = evidence_fixture
    spec = dict(content_hash='b'*64, portability_evidence=f.evidence, bundle_root=str(f.bundle.root),
                reference=f.reference, counts=f.original['counts'])
    monkeypatch.setattr(execution, 'gpu_identity', lambda: f.report['gpu'])
    value = p.setup_record(spec, f.val)
    assert value['qualified'] and value['control_inference_reused'] and not value['resource_acceptance']
    assert p.validate_setup(spec, value) == value
    bad = deepcopy(value); bad['resource_acceptance'] = True
    with pytest.raises(ValueError): p.validate_setup(spec, bad)
    monkeypatch.setattr(execution, 'gpu_identity', lambda: dict(name='wrong GPU'))
    with pytest.raises(ValueError, match='GPU/backend'): p.setup_record(spec, f.val)
    monkeypatch.setattr(execution, 'gpu_identity', lambda: f.report['gpu'])
    flags = precision.backend_settings(); flags['flash_sdp'] = not flags['flash_sdp']
    monkeypatch.setattr(precision, 'backend_settings', lambda: flags)
    with pytest.raises(ValueError, match='GPU/backend'): p.setup_record(spec, f.val)


@pytest.mark.parametrize('case', ['fp32', 'repeat', 'nan', 'identities', 'settings', 'missing_mode'])
def test_qualification_rejects_invalid_numeric_or_population_evidence(evidence_fixture, case):
    f = evidence_fixture; arrays = deepcopy(f.arrays); report = deepcopy(f.report)
    if case == 'fp32':
        arrays['gpu_fp32_math'][:, 0] += .001; arrays['gpu_fp32_math'][:, 1] -= .001
    if case == 'repeat':
        arrays['gpu_bf16_default'][:, 0] += .001; arrays['gpu_bf16_default'][:, 1] -= .001
    if case == 'nan': arrays['cpu_fp32_math'][0, 0] = np.nan
    if case == 'identities': arrays['identities'] = arrays['identities'][::-1]
    if case == 'settings': report['probe']['settings']['cpu_fp32_math']['matmul_tf32'] = True
    if case == 'missing_mode': del report['probe']['settings']['cpu_fp32_math']
    if case in ('fp32', 'repeat'):
        for name in report['probe']['comparisons']:
            a, b = name.split('_vs_'); report['probe']['comparisons'][name] = precision.differences(arrays[a], arrays[b])
    with pytest.raises(ValueError): p.qualify(report, arrays, f.full, f.saved, 22)


@pytest.mark.parametrize('ref', ['spec', 'report', 'receipt', 'full', 'probe', 'predictions'])
def test_reuse_refuses_corrupted_artifacts(evidence_fixture, ref):
    f = evidence_fixture
    with Path(f.evidence['refs'][ref]['path']).open('ab') as handle: handle.write(b'corruption')
    with pytest.raises(ValueError): p.validate_evidence(f.evidence, f.bundle.root, f.reference, f.original['counts'])


def test_wrong_source_or_job_not_imported(evidence_fixture, monkeypatch):
    f = evidence_fixture; path = f.root/'campaign_spec.json'
    monkeypatch.setattr(p, '_source', lambda *a: dict(content_hash='wrong'))
    with pytest.raises(ValueError, match='source'): p.authenticate(path, f.bundle.root, f.reference, f.original['counts'])
    monkeypatch.setattr(p, '_source', lambda *a: f.original['source_snapshot'])
    monkeypatch.setattr(p, 'replay', lambda *a: (file_ref(path), dict(job='other', report_hash=f.report['content_hash'])))
    with pytest.raises(ValueError, match='reviewed'): p.authenticate(path, f.bundle.root, f.reference, f.original['counts'])


def test_cross_site_auc_drift_guard_is_retained(evidence_fixture):
    f = evidence_fixture; arrays = deepcopy(f.arrays); report = deepcopy(f.report); hydra = deepcopy(f.full)
    probs = np.full((22, 11), .001, np.float32); probs[np.arange(22), hydra['labels']] = .99
    hydra['probabilities'] = probs
    for mode in (*precision.MODES, 'full_gpu_bf16'): arrays[mode] = probs.copy()
    for name in report['probe']['comparisons']:
        a, b = name.split('_vs_'); report['probe']['comparisons'][name] = precision.differences(arrays[a], arrays[b])
    settings = report['full_validation']['settings']
    report['full_validation'] = precision.full_comparison(hydra['labels'], probs, f.saved['probabilities'])
    report['full_validation']['settings'] = settings
    assert report['full_validation']['differences']['auc_abs'] > .1
    with pytest.raises(ValueError, match='qualification'): p.qualify(report, arrays, hydra, f.saved, 22)


def test_metric_roundoff_preserves_original_scalars_but_rejects_real_drift(evidence_fixture):
    f = evidence_fixture; report = deepcopy(f.report)
    report['full_validation']['differences']['rms'] += 1e-15
    summary = p.qualify(report, f.arrays, f.full, f.saved, 22)
    assert summary['full_validation']['differences']['rms'] == report['full_validation']['differences']['rms']
    report['full_validation']['differences']['rms'] += 1e-5
    with pytest.raises(ValueError, match='full metrics'): p.qualify(report, f.arrays, f.full, f.saved, 22)
    assert not p.same_metrics({'rows':22}, {'rows':22.})


def test_v2_spec_creation_and_immutable_qualification(evidence_fixture, tmp_path, monkeypatch):
    f = evidence_fixture
    origin = dict(content_hash='c'*64, reference=f.reference, counts=f.original['counts'], source_commit='d'*40,
        recipe=c.recipe(), imported={}, anchor={})
    f.bundle.manifest = dict(content_hash=c.MANIFEST_HASH, origin=dict(campaign=origin['content_hash'],
        reference=f.reference['content_hash'], environment={}, source_commit=origin['source_commit']))
    f.bundle.json = lambda _: origin
    f.bundle.verify_all = lambda: None
    f.paths['reference/M0HLT/predictions.npz'] = f.paths['reference/matched/predictions.npz']
    f.paths['reference/OFFLINE/predictions.npz'] = f.paths['reference/matched/predictions.npz']
    monkeypatch.setattr(campaign, 'Bundle', lambda _: f.bundle)
    monkeypatch.setattr(campaign, 'COUNTS', f.original['counts'])
    monkeypatch.setattr(campaign, 'current', lambda *a: dict(content_hash='s'*64))
    monkeypatch.setattr(campaign.site, 'environment', lambda e: e)
    monkeypatch.setattr(campaign, 'storage', lambda s: None)
    project = tmp_path/'source'; project.mkdir()
    spec = campaign.create(tmp_path/'campaign', f.bundle.root, project, 'b'*40, 8,
                           precision_spec=f.root/'campaign_spec.json')
    assert spec['schema_version'] == 2 and spec['portability_evidence'] == f.evidence
    assert campaign.check(spec) == spec and spec['parents']['portability'] == f.evidence['content_hash']
    assert c.authorization(spec) == c.AUTHORIZATION_V2
    with pytest.raises(ValueError): c.validate(spec, 'SPEC')
    bad = deepcopy(spec); bad['portability_policy']['cross_site_auc_limit'] = 1
    from hlt_classification.data.cache_contracts import with_content_hash
    with pytest.raises(ValueError): campaign.check(with_content_hash(bad))


def test_v2_setup_reuses_control_and_still_measures_resources(evidence_fixture, tmp_path, monkeypatch):
    f = evidence_fixture; root = tmp_path/'campaign'; root.mkdir()
    spec = dict(campaign.registration(2), schema_version=2, root=str(root), content_hash='f'*64,
        reference=f.reference, bundle_root=str(f.bundle.root), portability_evidence=f.evidence)
    spec['counts'] = f.original['counts']
    from hlt_classification.jetclass2_delphes import execution
    monkeypatch.setattr(execution, 'gpu_identity', lambda: f.report['gpu'])
    monkeypatch.setattr(worker, 'installed_environment', lambda: {})
    monkeypatch.setattr(worker.targets, 'load_view', lambda *a: f.val)
    monkeypatch.setattr(worker.targets, 'embedding_bank', lambda *a: 'original bank')
    monkeypatch.setattr(worker.targets, 'particle_bank', lambda *a: 'exported particles')
    monkeypatch.setattr(worker, 'control_parity', lambda *a, **k: pytest.fail('Control inference rerun'))
    def export(*a, **k):
        folder = root/'setup/particles'; folder.mkdir()
        write_immutable_json(folder/'manifest.json', {'fixture':True})
    monkeypatch.setattr(worker.targets, 'export', export)
    monkeypatch.setattr(worker, 'Bundle', lambda _: SimpleNamespace(path=lambda _: f.root/'campaign_spec.json'))
    calls = []
    monkeypatch.setattr(worker.resources, 'measure', lambda *a, **k: calls.append((a, k)))
    worker.setup(spec, device='cpu')
    assert len(calls) == 1 and calls[0][0][3:5] == ('original bank', 'exported particles')
    assert (root/'setup/portability.json').exists() and not (root/'setup/control_parity.json').exists()
    assert not (root/'setup/acceptance.json').exists()


def test_v2_results_use_hydra_delta_and_historical_anchors(evidence_fixture, tmp_path, monkeypatch):
    f = evidence_fixture; spec = dict(campaign.registration(2), schema_version=2, root=str(tmp_path),
        content_hash='f'*64, bundle_root=str(f.bundle.root), portability_evidence=f.evidence)
    spec.update(counts=f.original['counts'], original50k_rows=11)
    monkeypatch.setattr(worker.campaign, 'check', lambda *a, **k: None)
    subset = tmp_path/'subset.npz'; atomic_publish_bytes(subset, deterministic_npz_bytes(dict(indices=np.arange(11, dtype=np.int64))))
    monkeypatch.setattr(worker, 'Bundle', lambda _: SimpleNamespace(path=lambda name:
        subset if name.endswith('subset.npz') else f.paths['reference/matched/predictions.npz']))
    result = worker.results(spec)
    assert result['schema_version'] == 2 and result['comparison'] == 'U100_W5_HYDRA'
    assert result['recovery_anchors'] == 'historical_sporc_M0HLT_OFFLINE'
    rows = {r['model']:r for r in result['rows']}
    assert 'U100_W5_MATCHED' in rows and 'U100_W5_HYDRA' in rows
    assert rows['U100_W5_HYDRA']['validation_delta_matched'] == 0
    assert result['pending'] == list(c.ARMS)

"""Inference-only diagnostic: precision isolation, saved evidence and one-job DAG."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from hlt_classification.data.cache_contracts import (
    load_json, load_npz_arrays, write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_u100_pair_hydra import precision as p, precision_campaign as pc, contracts as c
from test_luka_u100_pair_correction import compact, fixture, threads


def test_probe_keeps_original_batches_padding_and_tail():
    batches = p.fixed_batches(74895)
    assert [len(x) for x in batches] == [128]*4+[15]
    assert [int(x[0]) for x in batches] == [0, 24832, 49792, 74752, 74880]
    rows = np.concatenate(batches)
    assert len(rows) == 527 and len(np.unique(rows)) == 527 and rows[-1] == 74894
    for batch in batches:
        assert batch[0] % 128 == 0
        np.testing.assert_array_equal(batch, np.arange(batch[0], batch[-1]+1))
    assert [len(x) for x in p.fixed_batches(256)] == [128, 128]
    assert [len(x) for x in p.fixed_batches(22)] == [22]
    with pytest.raises(ValueError): p.fixed_batches(0)
    with pytest.raises(ValueError): p.fixed_batches(74895, 256)


def test_backend_flags_restore_even_on_failure():
    before = p.backend_settings()
    with pytest.raises(RuntimeError, match='fixture'):
        with p.strict_math():
            flags = p.backend_settings()
            assert not flags['matmul_tf32'] and not flags['cudnn_tf32']
            assert not flags['bf16_reduced_precision_reduction']
            assert flags['math_sdp'] and not flags['flash_sdp'] and not flags['memory_efficient_sdp']
            assert not flags['cudnn_sdp']
            raise RuntimeError('fixture')
    assert p.backend_settings() == before


@pytest.mark.parametrize('bad', ['nan', 'negative', 'unnormalized', 'shape', 'empty'])
def test_invalid_probabilities_fail(bad):
    a = np.full((22, 11), 1/11)
    if bad == 'nan': a[0, 0] = np.nan
    if bad == 'negative': a[0, 0] = -.1
    if bad == 'unnormalized': a *= .9
    if bad == 'shape': a = a[:, :10]
    if bad == 'empty': a = a[:0]
    with pytest.raises(ValueError, match='probabilities'): p.differences(a, a)


def test_finite_large_differences_are_reported_not_admitted(capsys):
    labels = np.arange(220) % 11
    saved = np.full((220, 11), .01); saved[np.arange(220), labels] = .9
    actual = np.roll(saved, 1, axis=1)
    result = p.full_comparison(labels, actual, saved)
    assert not result['admission'] and not result['scientific_fits_released']
    assert all(not x['within_limit'] for x in result['original_gate'].values())
    assert result['differences']['top1_disagreements'] == 220
    assert result['differences']['rms'] == pytest.approx(np.sqrt(2*.89**2/11))
    p.print_report(dict(full_validation=result, probe=dict(rows=22, comparisons={})))
    output = capsys.readouterr().out
    assert 'censored' in output and 'QCD rejection @30%' in output and 'QCD rejection @50%' in output
    # No historical production tolerance was loosened.
    assert c.PARITY == dict(max_abs=.005, rms=.0002, auc_abs=.00005)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_native_fp32_replay_is_read_only(tmp_path, monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
    compact(monkeypatch); _, _, val, *_ = fixture(tmp_path)
    model = DelphesParticleTransformer().to(device).eval()
    state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    batches = [val.batch(x) for x in p.fixed_batches(len(val))]
    before = p.backend_settings()
    with p.strict_math():
        a = p._predict_batches(model, batches, device=device, bf16=False)
        b = p._predict_batches(model, batches, device=device, bf16=False)
    np.testing.assert_array_equal(a, b)
    assert p.backend_settings() == before
    assert all(torch.equal(v.cpu(), state[k]) for k,v in model.state_dict().items())
    assert all(v.grad is None for v in model.parameters())


def diagnostic_spec(root):
    return c.artifact('PRECISION_SPEC', root=str(root), project_dir=str(root/'code'),
        bundle_root=str(root/'inputs'), source_commit='a'*40, reference=dict(environment={}), **pc.registration())


def test_single_job_submission_is_exact_and_does_not_dispatch_science(tmp_path, monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as dag
    from hlt_classification.luka_u100_pair_hydra import submission
    calls = []
    def submit(args, **kw):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout='131999\n', stderr='')
    monkeypatch.setattr(dag.subprocess, 'run', submit)
    monkeypatch.setattr(pc, 'check', lambda *a, **k: None)
    monkeypatch.setattr(pc.site, 'check_site', lambda *a: None)
    monkeypatch.setattr(submission, 'dispatch', lambda *a, **k: pytest.fail('science dispatch'))
    spec = diagnostic_spec(tmp_path); plan = pc.review(spec)
    assert calls == [] and len(plan['commands']) == 1 and plan['scientific_fits'] == 0
    row = plan['commands'][0]
    assert row['task_id'] == 'precision' and row['dependencies'] == []
    for arg in ('--time=01:00:00', '--gres=gpu:nvidia_l40:1', '--partition=gpus', '--mem=120000M'):
        assert arg in row['command']
    assert row['command'][-1] == 'precision'
    with pytest.raises(PermissionError): pc.submit(spec, plan['content_hash'], 'wrong')
    with pytest.raises(PermissionError): pc.submit(spec, '0'*64, pc.AUTHORIZATION)
    assert calls == []
    ledger = pc.submit(spec, plan['content_hash'], pc.AUTHORIZATION)
    assert ledger['jobs'] == {'precision':'131999'} and pc.accepted_job(spec) == '131999'
    assert len(calls) == 1 and not any('--dependency' in a for a in calls[0])
    with pytest.raises(FileExistsError): pc.submit(spec, plan['content_hash'], pc.AUTHORIZATION)
    assert len(calls) == 1


@pytest.fixture
def registered(tmp_path, monkeypatch):
    bundle_root = tmp_path/'inputs'; bundle_root.mkdir()
    project = tmp_path/'code'; project.mkdir()
    reference = dict(content_hash='r'*64, environment={}, job_id='21993056')
    origin = dict(content_hash='o'*64, reference=reference, counts=c.COUNTS,
                  source_commit='d'*40, recipe=c.recipe())
    bundle = SimpleNamespace(root=bundle_root, manifest=dict(content_hash='m'*64, origin=dict(
        campaign=origin['content_hash'], reference=reference['content_hash'], environment={},
        source_commit=origin['source_commit'])), json=lambda name: origin)
    monkeypatch.setattr(pc.inputs, 'Bundle', lambda root: bundle)
    monkeypatch.setattr(pc.campaign, 'current', lambda *a: dict(content_hash='s'*64))
    monkeypatch.setattr(pc.site, 'environment', lambda e: e)
    monkeypatch.setattr(pc, 'storage', lambda s: None)
    spec = pc.create(tmp_path/'run', bundle_root, project, 'a'*40)
    return spec, bundle


def test_registration_rejects_science_or_policy_changes(registered):
    spec, _ = registered
    assert pc.check(spec) == spec
    for key, value in [('scientific_fits', 4), ('admission', True), ('modes', ['gpu_bf16_default']),
                       ('batch_size', 256), ('original_parity_limits', dict(max_abs=1., rms=1., auc_abs=1.))]:
        changed = deepcopy(spec); changed[key] = value
        # Even a correctly rehashed changed spec may not redefine this diagnostic.
        from hlt_classification.data.cache_contracts import with_content_hash
        changed = with_content_hash(changed)
        with pytest.raises(ValueError): pc.check(changed)


def test_fresh_root_and_origin_enforced(registered):
    spec, bundle = registered
    with pytest.raises(PermissionError): pc.create(spec['root'], bundle.root, spec['project_dir'], spec['source_commit'])
    bundle.manifest['origin']['environment'] = {'wrong': True}
    with pytest.raises(ValueError, match='origin'): pc.check(spec)


def test_wrong_accepted_job_never_opens_model(tmp_path, monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    monkeypatch.setattr(pc, 'check', lambda *a, **k: None)
    monkeypatch.setattr(pc.site, 'allocation', lambda s: '555')
    monkeypatch.setattr(pc, 'accepted_job', lambda s: '556')
    monkeypatch.setattr(pc.time, 'sleep', lambda s: None)
    monkeypatch.setattr(pc, 'evaluate', lambda *a: pytest.fail('unaccepted evaluation'))
    with pytest.raises(PermissionError, match='exact submitted'): pc.run(diagnostic_spec(tmp_path))
    assert not (tmp_path/'precision').exists()


def test_real_four_mode_diagnostic_outputs_and_reuse(tmp_path, monkeypatch):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer, installed_environment
    from hlt_classification.jetclass2_delphes.runner import predict
    compact(monkeypatch); _, _, val, *_ = fixture(tmp_path)
    native = DelphesParticleTransformer().eval()
    bundle_root = tmp_path/'inputs'; bundle_root.mkdir()
    checkpoint = bundle_root/'checkpoint.pt'; torch.save(native.state_dict(), checkpoint)
    # Synthetic original probabilities, deliberately very different; no quality gate.
    saved = np.full((len(val), 11), .001, np.float32)
    saved[np.arange(len(val)), val.labels] = .99
    predictions = bundle_root/'predictions.npz'
    atomic_publish_bytes(predictions, deterministic_npz_bytes(dict(identities=val.identities, labels=val.labels, probabilities=saved)))
    refs = {'reference/matched/checkpoint.pt':checkpoint, 'reference/matched/predictions.npz':predictions}
    class Bundle:
        def __init__(self, root): assert Path(root) == bundle_root
        def path(self, name):
            assert name in refs, 'Unexpected input: '+name
            return refs[name]
    monkeypatch.setattr(pc.inputs, 'Bundle', Bundle)
    reads = []
    def load(spec, role, coord):
        reads.append((role, coord)); assert (role, coord) == ('validation', 'D000')
        return val
    monkeypatch.setattr(pc.inputs, 'load_view', load)
    monkeypatch.setattr(pc, 'check', lambda *a, **k: None)
    monkeypatch.setattr(pc.site, 'allocation', lambda s: '131999')
    monkeypatch.setattr(pc, 'accepted_job', lambda s: '131999')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    spec = diagnostic_spec(tmp_path); spec['reference']['environment'] = installed_environment()
    before = p.backend_settings(); digest = file_ref(checkpoint)
    report = pc.run(spec)
    assert p.backend_settings() == before and file_ref(checkpoint) == digest
    assert reads == [('validation', 'D000')]
    assert report == pc.results(spec)
    assert set(report['probe']['settings']) == set(p.MODES)
    assert report['probe']['settings']['gpu_bf16_default'] == before
    assert not report['full_validation']['original_gate']['max_abs']['within_limit']
    arrays = load_npz_arrays(tmp_path/'precision/fixed_batch_predictions.npz')
    assert set(p.MODES) <= set(arrays) and all(arrays[m].shape == (22, 11) for m in p.MODES)
    np.testing.assert_array_equal(arrays['gpu_bf16_default'], arrays['full_gpu_bf16'])
    assert p.differences(arrays['gpu_fp32_math'], arrays['cpu_fp32_math'])['max_abs'] < 1e-5
    assert not (tmp_path/'setup/acceptance.json').exists()
    with pytest.raises(FileExistsError): pc.run(spec)
    # Receipts do not bless changed saved evidence.
    with (tmp_path/'precision/fixed_batch_predictions.npz').open('ab') as handle: handle.write(b'corrupt')
    with pytest.raises(ValueError): pc.results(spec)


def test_identity_mismatch_fails_before_checkpoint_open(tmp_path, monkeypatch):
    _, _, val, *_ = fixture(tmp_path)
    saved = tmp_path/'predictions.npz'
    atomic_publish_bytes(saved, deterministic_npz_bytes(dict(identities=val.identities[::-1],
        labels=val.labels, probabilities=np.full((len(val), 11), 1/11, np.float32))))
    monkeypatch.setattr(pc.inputs, 'Bundle', lambda _: SimpleNamespace(path=lambda name: saved))
    monkeypatch.setattr(pc.inputs, 'load_view', lambda *a: val)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('checkpoint opened before identity check'))
    with pytest.raises(ValueError, match='population/order'): pc.evaluate(diagnostic_spec(tmp_path), tmp_path, device='cpu')

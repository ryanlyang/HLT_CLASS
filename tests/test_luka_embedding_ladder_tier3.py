"""Tier3 transfer guards and unchanged real five-rung kernels on tiny populations."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from test_luka_embedding_ladder import gate_fixture, cache, bank_fixture
from test_luka_embedding_aux import threads
from hlt_classification.data.cache_contracts import (
    load_json, write_immutable_json, with_content_hash, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.luka_embedding_ladder import tier3 as t, worker, source, preflight, campaign, contracts
from hlt_classification.luka_embedding_aux import worker as old_worker


def fixture(tmp_path, monkeypatch):
    root = tmp_path/'original'; root.mkdir()
    gate = gate_fixture(root)
    write_immutable_json(root/'gate_spec.json', gate)
    (root/'preflight').mkdir()
    report = dict(content_hash='d'*64, job_id='21929903', train_minutes=dict(EMBED=700, LOGIT=650))
    write_immutable_json(root/'preflight/receipt.json', report)
    snapshot = dict(content_hash='c'*64, git_commit='e'*40)
    monkeypatch.setattr(t, 'current', lambda *a:snapshot)
    monkeypatch.setattr(t, 'authenticate', lambda *a:(
        gate, report, file_ref(root/'gate_spec.json'), file_ref(root/'preflight/receipt.json')))
    spec = t.create(root/'gate_spec.json', 'EMBED', tmp_path/'source', 'e'*40)
    return gate, report, spec


def test_create_reuses_original_gate_unchanged_and_idempotent(tmp_path, monkeypatch):
    gate, report, spec = fixture(tmp_path, monkeypatch)
    assert spec['execution_site'] == execution_site('sporc_a100')
    assert (spec['cpus'], spec['memory_mb'], spec['train_minutes']) == (6, 90000, 700)
    assert spec['parents']['gate'] == gate['content_hash']
    assert spec['parents']['preflight'] == report['content_hash']
    assert load_json(Path(gate['root'])/'gate_spec.json') == gate
    assert t.create(spec['gate_path'], 'EMBED', spec['project_dir'], spec['source_commit']) == spec
    assert t.check_spec(spec) == (gate, report)
    other = t.create(spec['gate_path'], 'LOGIT', spec['project_dir'], spec['source_commit'])
    assert other['root'] != spec['root'] and other['train_minutes'] == 650
    assert not (Path(gate['root'])/'science').exists()


@pytest.mark.parametrize('changed', [
    dict(execution_site=execution_site('sporc_a100_debug')),
    dict(memory_mb=180000), dict(cpus=8), dict(train_minutes=600),
    dict(source_commit='0'*40), dict(transfer_policy='anything'),
])
def test_rehashed_changes_rejected(tmp_path, monkeypatch, changed):
    gate, report, spec = fixture(tmp_path, monkeypatch)
    bad = with_content_hash(dict(spec, **changed))
    with pytest.raises(ValueError):
        t.validate_spec(bad, gate, report)


def test_duplicate_debug_or_different_transfer_cannot_be_submitted(tmp_path, monkeypatch):
    gate, report, spec = fixture(tmp_path, monkeypatch)
    debug = Path(gate['root'])/'science/EMBED'; debug.mkdir(parents=True)
    with pytest.raises(FileExistsError, match='debug science'):
        t.check_spec(spec)
    with pytest.raises(FileExistsError):
        t.create(spec['gate_path'], 'EMBED', spec['project_dir'], spec['source_commit'])
    # A different source also cannot claim an existing canonical tier3 arm.
    monkeypatch.setattr(t, 'debug_conflict', lambda *a:None)
    with pytest.raises(FileExistsError, match='differs'):
        t.create(spec['gate_path'], 'EMBED', tmp_path/'different', spec['source_commit'])


def test_original_evidence_failure_blocks_creation_without_output(tmp_path, monkeypatch):
    monkeypatch.setattr(t, 'current', lambda *a:dict(content_hash='a'*64))
    monkeypatch.setattr(t, 'authenticate', lambda *a:(_ for _ in ()).throw(ValueError('FAILED accounting')))
    with pytest.raises(ValueError, match='FAILED'):
        t.create(tmp_path/'gate_spec.json', 'EMBED', tmp_path/'source', 'e'*40)
    assert not list(tmp_path.iterdir())


def test_original_source_replay_and_unchanged_scientific_runtime(tmp_path, monkeypatch):
    root = tmp_path/'gate'; root.mkdir()
    gate = gate_fixture(root)
    write_immutable_json(root/'gate_spec.json', gate)
    (root/'preflight').mkdir()
    report = dict(job_id='21929903', content_hash='a'*64)
    write_immutable_json(root/'preflight/receipt.json', report)
    monkeypatch.setattr(t, 'validate_source_snapshot', lambda *a, **kw:None)
    monkeypatch.setattr(t, 'git', lambda *a:['A\t'+t.MODULE])
    calls = []
    def replay(path, project, code, marker):
        calls.append((path, project, code, marker))
        return file_ref(path), report
    monkeypatch.setattr(source, 'replay', replay)
    actual = t.authenticate(root/'gate_spec.json', tmp_path/'source')
    assert actual[:2] == (gate, report)
    assert calls[0][1] == gate['project_dir']
    assert 'evidence(load_json' in calls[0][2]
    monkeypatch.setattr(t, 'git', lambda *a:['M\tsrc/hlt_classification/luka_embedding_ladder/training.py'])
    with pytest.raises(ValueError, match='Runtime differs'):
        t.authenticate(root/'gate_spec.json', tmp_path/'source')


def test_source_guard_and_additive_namespaces(monkeypatch):
    project = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(t, '_source', lambda *a:dict(content_hash='a'*64))
    monkeypatch.setattr(t, 'git', lambda p,*a:list(t.REQUIRED) if 'ls-files' in a else [t.MODULE])
    assert t.current(project, 'e'*40)['content_hash'] == 'a'*64
    for name in t.REQUIRED:
        assert (project/name).is_file(), name
    monkeypatch.setattr(t, 'git', lambda p,*a:list(t.REQUIRED) if 'ls-files' in a else ['src/changed.py'])
    with pytest.raises(ValueError, match='scientific runtime'):
        t.current(project, 'e'*40)
    for status in ('M', 'D', 'R100'):
        monkeypatch.setattr(t, 'git', lambda *a:[status+'\tsrc/hlt_classification/luka_embedding_weight/source.py'])
        with pytest.raises(ValueError):
            t.matched_runtime(project, 'f'*40)
    monkeypatch.setattr(t, 'git', lambda *a:['A\tsrc/hlt_classification/luka_embedding_weight/source.py'])
    t.matched_runtime(project, 'f'*40)


def test_exact_tier3_submission_and_uncertain_claim(tmp_path, monkeypatch):
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    _, _, spec = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(t, 'check_site', lambda _:None)
    calls = []
    def scheduler(cmd, **kw):
        assert '--partition=tier3' in cmd and '--partition=debug' not in cmd
        assert '--cpus-per-task=6' in cmd and '--mem=90000M' in cmd
        assert '--gres=gpu:a100:1' in cmd and '--time=11:40:00' in cmd
        assert not any(k.startswith(('SBATCH_', 'SLURM_')) for k in kw['env'])
        calls.append(cmd)
        return SimpleNamespace(stdout='123;sporc\n')
    monkeypatch.setattr(exact.subprocess, 'run', scheduler)
    plan = t.submit(spec)
    assert plan['jobs'] == 1 and not calls
    with pytest.raises(PermissionError):
        t.submit(spec, execute=True, plan_hash='0'*64, authorization=t.AUTHORIZATION)
    for _ in range(2):
        t.submit(spec, execute=True, plan_hash=plan['content_hash'], authorization=t.AUTHORIZATION)
    assert len(calls) == 1
    other = t.create(spec['gate_path'], 'LOGIT', spec['project_dir'], spec['source_commit'])
    plan = t.submit(other)
    def uncertain(*a, **kw):
        calls.append(a)
        raise RuntimeError('possible accepted sbatch then disconnect')
    monkeypatch.setattr(exact.subprocess, 'run', uncertain)
    for error in (RuntimeError, FileExistsError):
        with pytest.raises(error):
            t.submit(other, execute=True, plan_hash=plan['content_hash'], authorization=t.AUTHORIZATION)
    assert len(calls) == 2


def test_corrupt_original_reference_rejected(tmp_path, monkeypatch):
    gate, report, spec = fixture(tmp_path, monkeypatch)
    path = Path(gate['root'])/'preflight/receipt.json'
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):
        t.validate_spec(spec, gate, report)


def test_redirected_output_root_rejected(tmp_path, monkeypatch):
    t.canonical_root(tmp_path)
    monkeypatch.setattr(Path, 'resolve', lambda *a, **kw:tmp_path/'redirected')
    with pytest.raises(PermissionError, match='redirected'):
        t.canonical_root(tmp_path)


def test_worker_allocation_still_rejects_different_gpu_environment(monkeypatch):
    from hlt_classification.luka_attention import worker as allocation
    gate = gate_fixture('/fixture')
    gate['execution_site'] = execution_site('sporc_a100')
    monkeypatch.setattr(allocation, 'require_backend', lambda:None)
    def check(site):
        assert site['partition'] == 'tier3'
        return '22', 6, 90000
    monkeypatch.setattr(allocation, 'allocation', check)
    monkeypatch.setattr(allocation, 'gpu_identity', lambda:dict(name='A100', total_memory_bytes=40*2**30))
    monkeypatch.setattr(allocation, 'installed_environment', lambda:dict(env='same'))
    measured = dict(gpu=allocation.gpu_identity(), environment=allocation.installed_environment())
    assert worker.execution(gate, measured) == '22'
    for changed in (dict(gpu=dict(name='A100-other')), dict(environment=dict(env='different'))):
        with pytest.raises(PermissionError, match='changed'):
            worker.execution(gate, dict(measured, **changed))


def test_real_cuda_preflight_transfer_and_both_complete_ladders(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('Local CUDA required; does not replace genuine gate replay')
    root = tmp_path/'gate'; root.mkdir()
    gate = gate_fixture(root)
    train = {co:cache(coordinate=co) for co in contracts.VIEWS}
    val = {co:cache('validation', coordinate=co) for co in contracts.VIEWS}
    _, emb, teacher = bank_fixture(tmp_path, train['D000'])
    write_immutable_json(tmp_path/'teacher_report.json', dict(content_hash='e'*64))
    predictions = dict(identities=val['D000'].identities, labels=val['D000'].labels,
                       probabilities=np.full((22,11), 1/11, np.float32))
    atomic_publish_bytes(tmp_path/'references.npz', deterministic_npz_bytes(predictions))
    gate['anchor']['evidence']['models'] = {n:dict(checkpoint=teacher,
        report=file_ref(tmp_path/'teacher_report.json'), predictions=file_ref(tmp_path/'references.npz'))
        for n in ('M0HLT', 'OFFLINE', 'U000')}
    gate['embedding_import'].update(bank=file_ref(tmp_path/'bank/manifest.json'),
        payload=file_ref(tmp_path/'bank/targets.npz'), predictions=file_ref(tmp_path/'references.npz'))
    for key in ('anchor', 'embedding_import'):
        gate[key] = with_content_hash(gate[key])
    gate['parents'].update(anchor=gate['anchor']['content_hash'], embeddings=gate['embedding_import']['content_hash'])
    gate = with_content_hash(gate)
    write_immutable_json(root/'gate_spec.json', gate)
    meta = ({}, dict(summaries={r:{co:dict(maximum=4) for co in contracts.VIEWS} for r in ('train','validation')}), {}, {})
    monkeypatch.setattr(campaign, 'check_gate', lambda _:None)
    monkeypatch.setattr(worker, 'execution', lambda *a:'77')
    monkeypatch.setattr(worker.data, 'load_views', lambda g,r,**kw:train if r=='train' else val)
    monkeypatch.setattr(preflight, 'installed_environment', lambda:gate['anchor']['evidence']['environment'])
    monkeypatch.setattr(source, 'current', lambda *a:gate['source_snapshot'])
    monkeypatch.setattr(worker.regular_cache, 'metadata', lambda _:meta)
    monkeypatch.setattr(old_worker.alljets_data, 'comparison_indices', lambda *a:np.arange(22, dtype=np.int64))
    def memory(_):
        return dict(rss_bytes=2**30, peak_rss_bytes=2**30, gpu_peak_bytes=int(torch.cuda.max_memory_allocated()))
    monkeypatch.setattr(preflight, 'memory_snapshot', memory)
    monkeypatch.setattr(worker, 'memory_snapshot', memory)
    measured = preflight.run(gate)
    snapshot = dict(content_hash='a'*64, git_commit='f'*40)
    monkeypatch.setattr(t, 'current', lambda *a:snapshot)
    monkeypatch.setattr(t, 'validate_source_snapshot', lambda *a, **kw:None)
    monkeypatch.setattr(t, 'authenticate', lambda *a:(
        gate, measured, file_ref(root/'gate_spec.json'), file_ref(root/'preflight/receipt.json')))
    def allocation(runtime, evidence):
        assert runtime['execution_site']['partition'] == 'tier3' and evidence == measured
        return '88'
    monkeypatch.setattr(worker, 'execution', allocation)
    specs = []
    for arm in contracts.ARMS:
        spec = t.create(root/'gate_spec.json', arm, tmp_path/'source', 'f'*40)
        specs.append(spec)
        value = t.run(spec)
        assert len(value['rows']) == 8
        assert t.results(spec) == value
        for node in gate['nodes'][arm]:
            folder = Path(spec['root'])/'ladder'/node['node_id']
            report = load_json(folder/'training_report.json')
            assert len(report['validation_history']) == 30
            assert report['parents']['embeddings'] == emb.manifest['content_hash']
            assert (folder/'selected.pt').exists() == (node['coordinate'] == 'D000')
        state = torch.load(Path(spec['root'])/'ladder'/(arm+'_D000')/'selected.pt', weights_only=True)
        assert not any(k.startswith('predictor.') for k in state)
        with pytest.raises(FileExistsError):
            t.run(spec)
    assert load_json(root/'gate_spec.json') == gate
    assert load_json(root/'preflight/receipt.json') == measured
    monkeypatch.setattr(torch, 'load', lambda *a,**kw:pytest.fail('Results must not load model weights'))
    for spec in specs:
        t.results(spec)
    assert (tmp_path/'teacher.pt').exists()
    assert (tmp_path/'bank/targets.npz').exists()

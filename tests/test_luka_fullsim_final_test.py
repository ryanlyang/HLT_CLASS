"""Synthetic final-test unlock/inference; never open real local test particles."""
from copy import deepcopy
from pathlib import Path
import os
import subprocess

import numpy as np
import pytest
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
import uproot

from test_luka_fullsim_stage2 import dataset, prepared, bind
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullsim import (
    final_test_campaign as c, final_test_data as d, final_test_worker as w, final_test_submission as sub,
    science_campaign as science, stage2 as s,
)
from hlt_classification.luka_fullsim.particles import ParticleReader, physical
from hlt_classification.jetclass2_delphes.cache import RamCache
from hlt_classification.context_fusion.inputs import PairedCache


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(17, 11)

    def forward(self, features, vectors, mask):
        x = (features*mask).sum(-1)/mask.sum(-1).clamp_min(1)
        if x.ndim == 3:
            x = x.mean(1)
        return self.linear(x)


@pytest.fixture
def locked(prepared, tmp_path, monkeypatch):
    old_root, p, f, inventory, splits = prepared
    bind(prepared, monkeypatch)
    root, science_root = tmp_path/'test_run', tmp_path/'science_run'
    root.mkdir()
    science_root.mkdir()
    models = []
    for node in science.nodes():
        path = science_root/(node['node_id']+'.pt')
        torch.save(Tiny().state_dict(), path)
        models.append(dict(node=node, checkpoint=file_ref(path), selected_pass=8,
            validation=dict(accuracy=.5, macro_ovr_auc=.6)))
    parent_spec, evidence = science_root/'campaign_spec.json', science_root/'evidence.json'
    write_immutable_json(parent_spec, {})
    write_immutable_json(evidence, dict(gpu=dict(total_memory_bytes=40*2**30)))
    parent = dict(root=str(science_root), prepared_root=str(old_root), models=models,
        spec=dict(path=str(parent_spec)), preflight=dict(path=str(evidence)))
    protocol = c.protocol()
    protocol.update(counts=splits['selected_counts']['final_test'],
        files=sum(g['role'] == 'final_test' for g in splits['groups']),
        split_sha256=splits['content_hash'])
    protocol['bootstrap']['draws'] = 8
    lock = c.artifact('FINALIST_LOCK', models=models, protocol=protocol, final_test_accessed=False)
    execution = c.execution_lock(root, tmp_path/'project', dict(content_hash='a'*64), parent,
        lock, dict(content_hash='b'*64))
    execution = with_content_hash({**execution, 'protocol': protocol})
    write_immutable_json(root/'finalist_lock.json', lock)
    path = root/'execution_lock.json'
    write_immutable_json(path, execution)
    monkeypatch.setattr(c, 'protocol', lambda: deepcopy(protocol))
    original = c.authenticate
    def authenticate(value, full=True):
        e, l = original(value, full=False)
        assert e == execution
        return (e, l, p, f, inventory, splits) if full else (e, l)
    monkeypatch.setattr(c, 'authenticate', authenticate)
    monkeypatch.setenv('SLURM_JOB_ID', '90001')
    monkeypatch.setattr(d.science_worker, 'execution', lambda *a: '90001')
    monkeypatch.setattr(w, 'new_model', lambda node: Tiny())
    return path, execution, p, f, inventory, splits


def fake_smoke(execution):
    return c.artifact('TRAIN_SMOKE', parents=dict(execution=execution['content_hash']),
        installed_parity=dict(passed=True), native_mask_parity=True, final_test_accessed=False,
        models=[dict(node_id=m['node']['node_id'], rows=64,
                     checkpoint_sha256=m['checkpoint']['sha256'], exact_probability_replay=True)
                for m in execution['parent']['models']])


def unlock(path, execution):
    directory = Path(execution['root'])/'evaluate'
    directory.mkdir(exist_ok=False)
    write_immutable_json(directory/'train_smoke.json', fake_smoke(execution))
    return d.consume(path)


def test_protocol_twelve_frozen_models_and_exact_population():
    p = c.protocol()
    assert p['counts'] == 149553 and p['files'] == 337
    assert len(p['nodes']) == 12 and not p['retraining'] and not p['test_selected_winner']
    assert p['primary_contrast'] == ['FINAL_BRIDGE_D000', 'DIRECT_D000']
    assert p['temperature'] == 1. and p['batch_size'] == 256
    assert p['bootstrap'] == dict(unit='source_file', draws=200, seed=20261012, interval=[.025, .975])


def test_no_capability_or_wrong_job_before_open(locked, monkeypatch):
    path, execution, *_ = locked
    monkeypatch.setattr(uproot, 'open', lambda *a, **kw: pytest.fail('Forbidden ROOT open'))
    with pytest.raises(PermissionError):
        list(d.test_jets(None))
    with pytest.raises(PermissionError):
        d.TestCache([], 'D000', None)
    with pytest.raises(PermissionError):
        d.Access(None, None, None, None, None, None, None)
    access = unlock(path, execution)
    assert Path(execution['claim_dir'], 'claim.json').is_file()
    with pytest.raises(PermissionError):
        d.consume(path)
    monkeypatch.setenv('SLURM_JOB_ID', '90002')
    with pytest.raises(PermissionError):
        list(d.test_jets(access))


def test_changed_locks_and_missing_claim_fail_before_read(locked, monkeypatch):
    path, execution, *_ = locked
    access = unlock(path, execution)
    claim = Path(execution['claim_dir'])/'claim.json'
    monkeypatch.setattr(d, 'load_json', lambda p: {'contract': 'bad'} if Path(p) == claim else load_json(p))
    with pytest.raises((ValueError, KeyError)):
        list(d.test_jets(access))


def test_reader_only_exact_test_entries_and_ordinary_firewalls(locked, monkeypatch):
    path, execution, _, f, inventory, splits = locked
    access = unlock(path, execution)
    expected = d.population_arrays(inventory, splits)
    allowed = {r['path']: set(sum(r['entries_by_class'], [])) for r, g in zip(inventory['files'], splits['groups']) if g['role'] == 'final_test'}
    original = uproot.behaviors.TBranch.HasBranches.arrays
    calls = []
    def guarded(tree, expressions, **kw):
        name = Path(tree.file.file_path).relative_to(Path(f['input_container'])/'raw').as_posix()
        assert name in allowed and set(range(kw['entry_start'], kw['entry_stop'])) <= allowed[name]
        calls.append(name)
        return original(tree, expressions, **kw)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, 'arrays', guarded)
    jets = list(d.test_jets(access))
    assert calls and len(jets) == len(expected['labels'])
    assert [j.identity for j in jets] == [bytes(x).hex() for x in expected['identities']]
    assert [j.label for j in jets] == expected['labels'].tolist()
    with pytest.raises(PermissionError):
        ParticleReader('absent', {}, {}, role='final_test')
    with pytest.raises(PermissionError):
        RamCache([], role='final_test', foundation_sha256='a'*64, coordinate_name='D000')


def test_all_views_cache_pair_bytes_native_hlt_and_budget(locked):
    path, execution, p, *_ = locked
    access = unlock(path, execution)
    with pytest.raises(MemoryError):
        d.build(access, 1)
    bundle, arrays, summaries = d.build(access, 2**30)
    assert set(bundle) == set(s.COORDINATES)
    for co, cache in bundle.items():
        ordinary = RamCache(cache.blocks, role='validation', foundation_sha256=p['content_hash'], coordinate_name=co)
        for name, actual in cache.batch(np.arange(len(cache))).items():
            np.testing.assert_array_equal(actual, ordinary.batch(np.arange(len(cache)))[name])
        assert summaries[co]['rows'] == len(arrays['labels'])
    pair = d.for_node(bundle, science.nodes()[10], access)
    assert pair.primary is pair.context
    left = RamCache(pair.primary.blocks, role='validation', foundation_sha256=p['content_hash'], coordinate_name='D000')
    expected = PairedCache(left, left).batch(np.arange(len(left)))
    for key, actual in pair.batch(np.arange(len(pair))).items():
        np.testing.assert_array_equal(actual, expected[key])
    for jet, row in zip(d.test_jets(access), range(len(pair))):
        native = s.build_inputs(s.views.build_view(identity=jet.identity,
            proxy=physical(jet.hlt, p['conventions'], 'hlt'), offline=None, coordinate='D000', mapping=None), capacity=512)
        batch = pair.primary.batch(np.asarray([row]))
        np.testing.assert_array_equal(batch['features'][0, :, :len(native.features)], native.features.T)


def test_train_smoke_all_checkpoints_no_test_reads(locked, monkeypatch):
    path, execution, p, f, inv, splits = locked
    allowed = {g['path'] for g in splits['groups'] if g['role'] == 'train'}
    original = uproot.open
    def guarded(value, *a, **kw):
        assert Path(value).relative_to(Path(f['input_container'])/'raw').as_posix() in allowed
        return original(value, *a, **kw)
    monkeypatch.setattr(uproot, 'open', guarded)
    monkeypatch.setattr(w, 'installed_parity', lambda *a, **kw: dict(passed=True))
    monkeypatch.setattr(w, 'native_mask_parity', lambda *a, **kw: True)
    result = w.smoke(execution, p, f, inv, splits, device='cpu')
    assert len(result['models']) == 12 and result['final_test_accessed'] is False
    assert not Path(execution['claim_dir']).exists()


def test_auc_ties_and_file_weights_match_sklearn():
    rng = np.random.default_rng(27)
    labels = np.tile(np.arange(11), 15)
    scores = rng.integers(1, 5, size=(len(labels), 11)).astype(np.float64)
    scores /= scores.sum(1, keepdims=True)
    weights = rng.integers(0, 5, size=len(labels)).astype(float)
    expected = roc_auc_score(labels, scores/scores.sum(1, keepdims=True), multi_class='ovr', sample_weight=weights)
    assert w.weighted_auc(w.auc_order(labels, scores), weights) == pytest.approx(expected, abs=1e-14)
    assert w.weighted_auc(w.auc_order(labels, scores), (labels == 0).astype(float)) is None
    predictions = {name: scores for name in ('FINAL_BRIDGE_D000', 'DIRECT_D000', 'M0HLT')}
    p = dict(seed=19, draws=12, interval=[.025, .975])
    a = w.paired_intervals(labels, predictions, np.arange(len(labels))//11, p)
    assert a == w.paired_intervals(labels, predictions, np.arange(len(labels))//11, p)
    assert all(row['interval_95'] == [0., 0.] for m in a['contrasts'].values() for row in m.values())


def test_complete_twelve_model_inference_and_report_without_training(locked, monkeypatch):
    path, execution, *_ = locked
    original_probabilities = w.probabilities
    monkeypatch.setattr(w, 'probabilities', lambda model, cache, device: original_probabilities(model, cache, device='cpu'))
    monkeypatch.setattr(w, 'installed_parity', lambda *a, **kw: dict(passed=True))
    monkeypatch.setattr(w, 'native_mask_parity', lambda *a, **kw: True)
    monkeypatch.setattr(w, 'memory_snapshot', lambda _: dict(peak_rss_bytes=1, gpu_peak_bytes=1))
    import hlt_classification.jetclass2_delphes.runner as runner
    monkeypatch.setattr(runner, 'train_kernel', lambda *a, **kw: pytest.fail('No test-stage fitting'))
    first = w.evaluate(path)
    assert len(first['result']['models']) == 12
    assert w.evaluate(path) == first
    report = w.report(path)
    assert len(report['result']['rows']) == 12 and report['result']['final_test_accessed']
    assert w.report(path) == report
    w.print_results(path)
    pred = Path(execution['root'])/first['result']['models'][0]['prediction']['path']
    with pred.open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='bytes differ'):
        w.completed(execution, 'report')


def test_lock_authentication_rejects_relocated_or_changed_protocol(locked, monkeypatch):
    path, execution, *_ = locked
    bad = deepcopy(execution)
    bad['protocol']['counts'] += 1
    bad = with_content_hash(bad)
    monkeypatch.setattr(c, 'load_json', lambda p: bad if Path(p) == path else load_json(p))
    with pytest.raises(ValueError, match='protocol'):
        c.authenticate(path, full=False)


def test_exact_two_job_submission_and_no_repeat(locked, monkeypatch):
    path, execution, *_ = locked
    monkeypatch.setattr(sub, 'check_site', lambda value: None)
    value = sub.submit(path)
    assert value['jobs'] == 2 and value['commands'][1]['dependencies'] == ['evaluate']
    assert '--mem=256000M' in value['commands'][0]['command']
    assert '--time=23:06:00' in value['commands'][0]['command']
    assert not Path(execution['claim_dir']).exists()
    with pytest.raises(PermissionError):
        sub.submit(path, execute=True, plan_hash='0'*64, authorization=c.AUTHORIZATION)
    calls, original = [], subprocess.run
    def run(argv, **kw):
        if argv[0] == 'sbatch':
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, str(400+len(calls))+'\n', '')
        return original(argv, **kw)
    monkeypatch.setattr(subprocess, 'run', run)
    live = sub.submit(path, execute=True, plan_hash=value['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 2 and '--dependency=afterok:401' in calls[1]
    assert sub.submit(path, execute=True, plan_hash=value['content_hash'], authorization=c.AUTHORIZATION) == live
    sub.submit(path)
    assert len(calls) == 2


def test_failed_live_submission_preserves_exclusive_claim(locked, monkeypatch):
    path, execution, *_ = locked
    monkeypatch.setattr(sub, 'check_site', lambda _: None)
    value = sub.submit(path)
    original = subprocess.run
    def run(argv, **kw):
        if argv[0] == 'sbatch':
            raise RuntimeError('ambiguous scheduler response')
        return original(argv, **kw)
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(RuntimeError):
        sub.submit(path, execute=True, plan_hash=value['content_hash'], authorization=c.AUTHORIZATION)
    with pytest.raises(PermissionError, match='Submission active'):
        sub.submit(path, execute=True, plan_hash=value['content_hash'], authorization=c.AUTHORIZATION)
    assert not Path(execution['claim_dir']).exists()


def test_scripts_and_old_sources_remain_sealed():
    root = Path(__file__).resolve().parents[1]
    changed = subprocess.run(['git', '-C', str(root), 'diff', '--name-only', c.SCIENCE_COMMIT, '--', 'src'],
                             check=True, text=True, capture_output=True).stdout.splitlines()
    assert set(changed) <= c.ADDITIONS
    worker = (root/'sbatch/run_luka_fullsim_final_test.sh').read_text()
    assert 'source "${PROJECT_DIR}/sbatch/common.sh"' in worker
    assert 'PYTHONNOUSERSITE=1' in worker and '${CONDA_PREFIX}/lib' in worker
    assert worker.index('CUBLAS_WORKSPACE_CONFIG=:4096:8') < worker.index('exec python')
    assert all((root/p).is_file() for p in c.REQUIRED)


def test_missing_or_failed_train_miniature_cannot_consume_test(locked, monkeypatch):
    path, execution, *_ = locked
    monkeypatch.setattr(uproot, 'open', lambda *a, **kw: pytest.fail('Forbidden test read'))
    with pytest.raises(FileNotFoundError):
        d.consume(path)
    directory = Path(execution['root'])/'evaluate'
    directory.mkdir()
    value = fake_smoke(execution)
    value['models'][0]['exact_probability_replay'] = False
    write_immutable_json(directory/'train_smoke.json', with_content_hash(value))
    with pytest.raises(ValueError, match='miniature'):
        d.consume(path)
    assert not Path(execution['claim_dir']).exists()


def test_failed_smoke_never_unseals_and_failed_test_never_retries(locked, monkeypatch):
    path, execution, *_ = locked
    monkeypatch.setattr(w, 'memory_snapshot', lambda _: dict(peak_rss_bytes=1, gpu_peak_bytes=1))
    monkeypatch.setattr(w, 'smoke', lambda *a: fake_smoke(execution))
    monkeypatch.setattr(w, 'probabilities', lambda *a, **kw: (_ for _ in ()).throw(ValueError('nonfinite prediction')))
    with pytest.raises(ValueError, match='nonfinite'):
        w.evaluate(path)
    assert Path(execution['claim_dir'], 'claim.json').is_file()
    assert (Path(execution['root'])/'evaluate/population.npz').is_file()
    assert w.completed(execution, 'evaluate') is None
    monkeypatch.setattr(uproot, 'open', lambda *a, **kw: pytest.fail('No automatic test reread'))
    with pytest.raises(PermissionError, match='consumed'):
        w.evaluate(path)
    with pytest.raises(ValueError, match='must commit'):
        w.report(path)


def test_smoke_failure_before_claim_preserves_diagnostics(locked, monkeypatch):
    path, execution, *_ = locked
    monkeypatch.setattr(w, 'memory_snapshot', lambda _: dict(peak_rss_bytes=1, gpu_peak_bytes=1))
    monkeypatch.setattr(w, 'smoke', lambda *a: (_ for _ in ()).throw(ValueError('TRAIN parity failed')))
    monkeypatch.setattr(uproot, 'open', lambda *a, **kw: pytest.fail('Forbidden test read'))
    with pytest.raises(ValueError, match='TRAIN parity'):
        w.evaluate(path)
    assert not Path(execution['claim_dir']).exists()
    assert (Path(execution['root'])/'evaluate').is_dir()


def test_source_reuse_requires_exact_old_bytes_and_only_four_new_modules(tmp_path):
    from hlt_classification.provenance import capture_source_snapshot
    old, new = tmp_path/'old', tmp_path/'new'
    def commit(project):
        subprocess.run(['git', '-C', str(project), 'add', '.'], check=True, capture_output=True)
        subprocess.run(['git', '-C', str(project), '-c', 'user.name=Fixture', '-c',
                        'user.email=fixture@example.invalid', 'commit', '-m', 'fixture'],
                       check=True, capture_output=True)
        return capture_source_snapshot(project, require_clean=True)
    for project in (old, new):
        (project/'src').mkdir(parents=True)
        (project/'src/frozen.py').write_bytes(b'frozen = 1\n')
        subprocess.run(['git', '-C', str(project), 'init'], check=True, capture_output=True)
    parent = dict(project_dir=str(old), source_snapshot=commit(old))
    for relative in c.ADDITIONS:
        path = new/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'# additive execution\n')
    result = c.reuse(parent, commit(new), new)
    assert result['old_src_files'] == 1 and set(result['added_modules']) == c.ADDITIONS
    (new/'src/frozen.py').write_bytes(b'frozen = 2\n')
    with pytest.raises(ValueError, match='scientific source changed'):
        c.reuse(parent, commit(new), new)
    (new/'src/extra.py').write_bytes(b'# unauthorized\n')
    with pytest.raises(ValueError, match='additions/deletions'):
        c.reuse(parent, commit(new), new)


def test_parent_authentication_uses_original_pinned_imports(tmp_path, monkeypatch):
    path = tmp_path/'campaign_spec.json'
    value = dict(source_commit=c.SCIENCE_COMMIT, project_dir=str(tmp_path/'old_project'))
    write_immutable_json(path, value)
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        assert kwargs['cwd'] == value['project_dir']
        assert kwargs['env']['PYTHONPATH'] == str(Path(value['project_dir'])/'src')
        assert kwargs['env']['PYTHONDONTWRITEBYTECODE'] == '1'
        assert argv[-1] == str(path.resolve())
        assert 'w.completed(spec' in argv[-2]
        return subprocess.CompletedProcess(argv, 0, '{"authenticated": true}', '')
    monkeypatch.setattr(subprocess, 'run', run)
    assert c.parent_snapshot(path) == dict(authenticated=True)
    def failed(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv, stderr='Corrupt parent checkpoint')
    monkeypatch.setattr(subprocess, 'run', failed)
    with pytest.raises(ValueError, match='Corrupt parent checkpoint'):
        c.parent_snapshot(path)
    monkeypatch.setattr(c, 'load_json', lambda _: {**value, 'source_commit': '0'*40})
    with pytest.raises(ValueError, match='Wrong completed'):
        c.parent_snapshot(path)
    assert len(calls) == 1


def test_population_rejects_changed_split_counts_or_file_reservoir(monkeypatch):
    foundation = dict(counts=dict(train=100000, validation=50000, final_test=149553))
    splits = dict(content_hash=c.SPLIT_HASH, groups=[dict(role='final_test') for _ in range(337)])
    monkeypatch.setattr(s, 'load_prepared', lambda _: ({}, foundation, {}, splits))
    assert c.population(dict(prepared_root='unused'))[1] == foundation
    for field in ('counts', 'files', 'hash'):
        f, sp = deepcopy(foundation), deepcopy(splits)
        if field == 'counts':
            f['counts']['final_test'] += 1
        elif field == 'files':
            sp['groups'].pop()
        else:
            sp['content_hash'] = '0'*64
        monkeypatch.setattr(s, 'load_prepared', lambda _: ({}, f, {}, sp))
        with pytest.raises(ValueError, match='population differs'):
            c.population(dict(prepared_root='unused'))


def test_second_output_root_cannot_reopen_same_science_claim(locked, tmp_path, monkeypatch):
    path, execution, *_ = locked
    unlock(path, execution)
    assert Path(execution['claim_dir']).exists()
    parent = {**execution['parent'], 'project_dir': str(tmp_path/'old_project')}
    monkeypatch.setattr(c, 'source', lambda *a: {})
    monkeypatch.setattr(c, 'parent_snapshot', lambda _: parent)
    monkeypatch.setattr(c, 'population', lambda _: ({}, {}, {}, {}))
    monkeypatch.setattr(c, 'reuse', lambda *a: {})
    target = tmp_path/'second_test_run'
    with pytest.raises(PermissionError, match='already has'):
        c.create(path, target, tmp_path/'new_project', 'a'*40)
    assert not target.exists()


def test_create_authenticate_and_changed_finalist_fail_closed(prepared, tmp_path, monkeypatch):
    prepared_root, p, f, inventory, splits = prepared
    parent_root, project = tmp_path/'parent', tmp_path/'project'
    parent_root.mkdir()
    parent_spec = parent_root/'campaign_spec.json'
    completion = parent_root/'complete.json'
    checkpoint = parent_root/'selected.pt'
    report = parent_root/'training_report.json'
    write_immutable_json(parent_spec, c.artifact('FIXTURE'))
    write_immutable_json(completion, c.artifact('FIXTURE'))
    write_immutable_json(report, c.artifact('FIXTURE'))
    checkpoint.write_bytes(b'fixture weights; no particle decode')
    active = dict(content_hash='a'*64, git_commit='b'*40)
    reuse_record = c.artifact('SOURCE_REUSE')
    parent = dict(root=str(parent_root), project_dir=str(tmp_path/'old_project'),
        prepared_root=str(prepared_root), source_snapshot=dict(content_hash='c'*64),
        preflight=dict(path=str(tmp_path/'old_preflight/preflight.json')),
        spec=file_ref(parent_spec), completion=file_ref(completion), models=[
            dict(node=n, receipt=file_ref(completion), report=file_ref(report),
                 checkpoint=file_ref(checkpoint), selected_pass=14, validation={}) for n in science.nodes()])
    monkeypatch.setattr(c, 'source', lambda *a: deepcopy(active))
    monkeypatch.setattr(c, 'parent_snapshot', lambda _: deepcopy(parent))
    monkeypatch.setattr(c, 'population', lambda _: (p, f, inventory, splits))
    monkeypatch.setattr(c, 'reuse', lambda *a: deepcopy(reuse_record))
    monkeypatch.setattr(uproot, 'open', lambda *a, **kw: pytest.fail('Locks never open particles'))
    root = tmp_path/'created'
    execution = c.create(parent_spec, root, project, active['git_commit'])
    path = root/'execution_lock.json'
    assert c.authenticate(path)[0] == execution
    assert not Path(execution['claim_dir']).exists()
    parent['models'][0]['selected_pass'] += 1
    with pytest.raises(ValueError, match='drifted'):
        c.authenticate(path)
    parent['models'][0]['selected_pass'] -= 1
    checkpoint.write_bytes(b'corrupt frozen weights')
    with pytest.raises(ValueError, match='bytes differ'):
        c.authenticate(path)

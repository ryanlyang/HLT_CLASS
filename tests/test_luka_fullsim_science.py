"""CPU/synthetic stage-3 integration, not a claim of new Slurm/GPU execution."""
from copy import deepcopy
from pathlib import Path
import math
import subprocess

import numpy as np
import pytest
import torch
from torch import nn
import uproot

from test_luka_fullsim_stage2 import dataset, prepared, bind
from test_luka_fullsim_preflight_v2 import _git, _commit
from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json
from hlt_classification.luka_fullsim import (
    science_campaign as c, science_admission as a, science_cache as cache,
    science_submission as sub, science_worker as w, preflight_v2 as gpu)
from hlt_classification.jetclass2_delphes import runner
from hlt_classification.jetclass2_delphes.contracts import artifact as kernel_artifact
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import (
    PAIR_OFFLOAD_POLICY, PARITY_CHECKS, PARITY_BACKEND, PARITY_TOLERANCES)
from hlt_classification.jetclass2_delphes.model import model_contract
from hlt_classification.luka_fullsim.stage2_cache import prepare_cache


def fixture_spec(tmp_path, prepared_value=None):
    p = prepared_value or dict(content_hash='f'*64, counts=a.COUNTS)
    root = tmp_path/'science'
    root.mkdir()
    spec = c.make_artifact('SPEC', root=str(root), project_dir=str(tmp_path/'project'), source_commit='a'*40,
        prepared_sha256=p['content_hash'], preflight_sha256=a.PREFLIGHT_HASH,
        admission=dict(prepared_root='fixture', preflight=dict(path=str(tmp_path/'preflight.json'))),
        train_minutes=1386, reduce_minutes=131, **{**c.registration(), 'counts': p['counts']})
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def test_graph_no_imported_teachers_and_paired_endpoints():
    nodes, tasks = c.nodes(), c.tasks()
    assert len(nodes) == 12 and len(tasks) == 21
    assert sum(t['kind'] == 'reduce' for t in tasks) == 7
    assert c.registration()['training_recipe'] == runner.recipe()
    assert c.registration()['training_recipe']['patience_clock'] == 'starts_at_first_validation_not_pass60'
    assert c.registration()['counts'] == dict(train=100000, validation=50000)
    assert not c.registration()['imported_models']
    assert not any('DENSE' in t['task_id'] for t in tasks)
    seen = set()
    for task in tasks:
        assert set(task['dependencies']) <= seen
        seen.add(task['task_id'])
    named = {n['node_id']: n for n in nodes}
    for name in ('DIRECT_D000', 'FINAL_DIRECT_D000', 'FINAL_BRIDGE_D000'):
        assert named[name]['sampler_seed'] == named['M0HLT']['sampler_seed']
        assert named[name]['initialization_seed'] == named['M0HLT']['initialization_seed']
        assert named[name]['deployable'] and named[name]['context_coordinate'] is None
    assert not named['FUSION_D000']['deployable']
    assert named['FUSION_D000_D000']['deployable']
    assert {t['task_id']:t for t in tasks}['train_DIRECT_D000']['dependencies'] == ['reduce_U000']


@pytest.mark.parametrize('role', ['train', 'validation'])
def test_all_science_views_exact_v1_and_single_pass(prepared, monkeypatch, role):
    root, p = bind(prepared, monkeypatch)
    calls, original = [], cache.ParticleReader
    def counted(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(cache, 'ParticleReader', counted)
    for coords in (('OFFLINE',), ('U000', 'U050'), ('U100', 'D066'), ('D033', 'D000')):
        calls.clear()
        values = cache.build(root, role=role, coordinates=coords, max_ram_bytes=2**30)
        assert len(calls) == 1
        for coord, actual in values.items():
            expected = prepare_cache(root, role=role, coordinate=coord, max_ram_bytes=2**30)
            for key, array in expected.batch(np.arange(len(expected))).items():
                np.testing.assert_array_equal(actual.batch(np.arange(len(actual)))[key], array)


def test_native_hlt_no_offline_shared_pair_and_firewall(prepared, monkeypatch):
    root, p = bind(prepared, monkeypatch)
    original, calls = uproot.behaviors.TBranch.HasBranches.arrays, []
    def guarded(tree, expressions, *args, **kwargs):
        assert not any(x.startswith('part_') or x == 'jet_nparticles' for x in expressions)
        calls.append(expressions)
        return original(tree, expressions, *args, **kwargs)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, 'arrays', guarded)
    with pytest.raises(MemoryError):
        cache.build(root, role='train', coordinates=('D000',), max_ram_bytes=1)
    assert not calls
    pair = cache.cache(dict(memory_mb=256000, admission=dict(prepared_root=str(root))), p, c.nodes()[10], 'train')
    assert pair.primary is pair.context
    assert calls
    for role in ('final_test', 'unused', 'test'):
        with pytest.raises(PermissionError):
            cache.build('absent', role=role, coordinates=('D000',), max_ram_bytes=2**30)
    with pytest.raises(ValueError):
        cache.build(root, role='train', coordinates=('D000', 'D000'), max_ram_bytes=2**30)


def test_cache_rejects_resealed_bad_input_fingerprint(prepared, monkeypatch):
    root, p = bind(prepared, monkeypatch)
    original = cache.s.load_prepared(root)
    bad = deepcopy(p)
    bad['summaries']['train']['U100']['input_sha256'] = '0'*64
    monkeypatch.setattr(cache.s, 'load_prepared', lambda _: (bad, *original[1:]))
    with pytest.raises(ValueError, match='replay'):
        cache.build(root, role='train', coordinates=('U100',), max_ram_bytes=2**30)


def evidence():
    p = dict(content_hash='f'*64, counts=dict(a.COUNTS), source_snapshot=dict(content_hash='d'*64),
        summaries={r: {co: dict(rows=n, maximum=10 if co == 'U000' else 8, resident_bytes=100)
                        for co in cache.s.COORDINATES} for r, n in a.COUNTS.items()})
    stats = {name: dict(calls=3, saved_cuda_tensors=1, saved_cuda_bytes=100, restored_cuda_tensors=1)
             for name in PAIR_OFFLOAD_POLICY['scope']}
    parities = dict(installed=kernel_artifact('WEAVER_PARITY', passed=True, device='cuda', model=model_contract(),
        forward_and_feature_and_parameter_gradients=True), native_mask=True)
    for name in ('FUSION_probe', 'HLT_PAIR_probe'):
        for bf16 in (False, True):
            precision = 'bf16' if bf16 else 'fp32'
            parities[f'{name}_{bf16}'] = dict(passed=True, precision=precision, device_type='cuda', steps=3,
                checks=PARITY_CHECKS, parity_backend=PARITY_BACKEND, saved_tensor_storage=PAIR_OFFLOAD_POLICY,
                tolerance=PARITY_TOLERANCES[precision], offload_stats=stats)
    measurements = []
    for index in range(4):
        node = gpu._node(index)
        report = kernel_artifact('KERNEL_TRAINING_REPORT', node=node, foundation_sha256=p['content_hash'],
            recipe_sha256=runner.recipe()['content_hash'], scientific_fit=False, acceptance_only=True, passes=1,
            selected_weights_restored=True, final_test_accessed=False, runtime_seconds=10.)
        measurements.append(dict(name=node['node_id'], node=node, training=report,
            cache_seconds=10., inference_seconds=5., stress_steps=3, stress_batch=256, stress_offload=stats))
    r = with_content_hash(dict(contract='LUKA_FULLSIM_GPU_PREFLIGHT/v2', schema_version=2,
        counts=dict(a.COUNTS), recipe=runner.recipe(), site=c.registration()['execution_site'], cpus=6, memory_mb=256000,
        technical_checks_passed=True, checkpoint_probability_roundtrip=True, resource_envelope_ok=True, fits_debug_24h=True,
        scientific_fit=False, scientific_metrics_for_selection=False, final_test_accessed=False,
        production_submission_authorized=False, automatic_followup=False, preparation_source_snapshot=p['source_snapshot'],
        environment=kernel_artifact('INSTALLED_ENVIRONMENT', version=2, architecture='x86_64'),
        gpu=dict(name='A100', total_memory_bytes=40*2**30), gpu_peak_bytes=20*2**30, peak_rss_bytes=150*2**30,
        parities=parities, measurements=measurements, projected_train_minutes=math.ceil(1.75*1010/60),
        projected_reduce_minutes=1))
    return p, r


def test_admission_facts_fail_closed():
    p, r = evidence()
    assert a.validate_facts(r, p) == r
    for key, value in [('counts', dict(train=10, validation=5)), ('technical_checks_passed', False),
            ('checkpoint_probability_roundtrip', False), ('final_test_accessed', True),
            ('scientific_fit', True), ('memory_mb', 90000), ('cpus', 16), ('gpu_peak_bytes', 40*2**30),
            ('peak_rss_bytes', 256000*1024**2), ('projected_train_minutes', 1441), ('measurements', [])]:
        bad = with_content_hash(dict(r, **{key:value}))
        with pytest.raises((ValueError, PermissionError)):
            a.validate_facts(bad, p)
    bad = deepcopy(r)
    bad['parities']['FUSION_probe_True']['offload_stats'] = {}
    with pytest.raises(ValueError):
        a.validate_facts(with_content_hash(bad), p)
    bad = deepcopy(p)
    bad['summaries']['train']['D033']['maximum'] = 11
    with pytest.raises(ValueError, match='support'):
        a.validate_facts(r, bad)


def test_authentication_pins_report_and_all_output_bytes(tmp_path, monkeypatch):
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    p, r = evidence()
    old_reuse = dict(content_hash='b'*64)
    active = dict(content_hash='c'*64)
    r.update(source_snapshot=dict(git_commit=a.PREFLIGHT_COMMIT, content_hash='e'*64),
        parents=dict(prepared=p['content_hash'], source='e'*64, reuse='b'*64))
    write_immutable_json(tmp_path/'prepared_reuse.json', old_reuse)
    write_immutable_json(tmp_path/'runtime.json', dict(job_id='21835630'))
    r['outputs'] = {name: {k:v for k,v in file_ref(tmp_path/name).items() if k != 'path'}
                    for name in ('prepared_reuse.json', 'runtime.json')}
    r = with_content_hash(r)
    write_immutable_json(tmp_path/'preflight.json', r)
    monkeypatch.setattr(a.s, 'load_prepared', lambda _: (p, {}, {}, {}))
    monkeypatch.setattr(a.preflight_reuse, 'authenticate', lambda *args, **kwargs: old_reuse)
    monkeypatch.setattr(a, 'compatible_source', lambda *args, **kwargs: dict(content_hash='d'*64))
    args = dict(prepared_root=tmp_path/'prepared', preflight_path=tmp_path/'preflight.json',
        prepared_project=tmp_path/'old', preflight_project=tmp_path/'preflight_src', project=tmp_path/'science_src', active=active)
    with pytest.raises(ValueError, match='reviewed'):
        a.authenticate(**args)
    monkeypatch.setattr(a, 'PREFLIGHT_HASH', r['content_hash'])  # Synthetic fixture only.
    admitted, *_ = a.authenticate(**args)
    assert admitted['imported_models'] == [] and not admitted['probe_models_used_for_science']
    (tmp_path/'runtime.json').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='bytes differ'):
        a.authenticate(**args)


def test_create_validate_and_protected_root(tmp_path, monkeypatch):
    active = dict(content_hash='a'*64)
    admitted = dict(content_hash='b'*64, prepared_root=str(tmp_path/'prepared'),
        prepared_project=str(tmp_path/'old'), preflight_project=str(tmp_path/'preflight_source'),
        preflight=dict(path=str(tmp_path/'evidence/preflight.json')))
    p = dict(content_hash='c'*64, foundation_root=str(tmp_path/'foundation'))
    f = dict(input_container=str(tmp_path/'raw'))
    r = dict(content_hash=a.PREFLIGHT_HASH, projected_train_minutes=1386, projected_reduce_minutes=131, job_id='21835630')
    monkeypatch.setattr(a, 'source', lambda *args: active)
    monkeypatch.setattr(a, 'authenticate', lambda **kwargs: (admitted,p,f,r))
    monkeypatch.setattr(c, 'preflight_accounting', lambda job: dict(job_id=job, state='COMPLETED', exit_code='0:0'))
    args = dict(prepared_root=admitted['prepared_root'], preflight_path=admitted['preflight']['path'],
        prepared_project=admitted['prepared_project'], preflight_project=admitted['preflight_project'],
        project=tmp_path/'project', commit='e'*40, root=tmp_path/'science')
    spec = c.create(**args)
    assert c.validate_spec(spec) == (p,r)
    with pytest.raises(FileExistsError):
        c.create(**args)
    bad = with_content_hash(dict(spec, counts=dict(train=200000, validation=50000)))
    with pytest.raises(ValueError, match='registration'):
        c.validate_spec(bad)
    for target in (tmp_path, tmp_path/'raw/child', tmp_path/'project', tmp_path/'prepared/subdir'):
        with pytest.raises(PermissionError, match='overlaps'):
            c.check_location(target, args['project'], admitted, p, f)


def test_runtime_gpu_environment_and_cudnn_guards(tmp_path, monkeypatch):
    spec = fixture_spec(tmp_path)
    expected = dict(gpu=dict(name='A100'), environment=dict(version='fixture'))
    write_immutable_json(tmp_path/'runtime.json', dict(cudnn_v8_disabled='unset'))
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    monkeypatch.delenv('TORCH_CUDNN_V8_API_DISABLED', raising=False)
    monkeypatch.setattr(w, 'allocation', lambda _: ('21840000', 6, 256000))
    monkeypatch.setattr(w, 'gpu_identity', lambda: expected['gpu'])
    monkeypatch.setattr(w, 'installed_environment', lambda: expected['environment'])
    assert w.execution(spec, expected) == '21840000'
    monkeypatch.setenv('TORCH_CUDNN_V8_API_DISABLED', '1')
    with pytest.raises(PermissionError, match='cuDNN'):
        w.execution(spec, expected)
    monkeypatch.delenv('TORCH_CUDNN_V8_API_DISABLED')
    monkeypatch.setattr(w, 'allocation', lambda _: ('21840000', 6, 90000))
    with pytest.raises(PermissionError, match='allocation'):
        w.execution(spec, expected)


def test_new_source_requires_all_old_bytes_and_exact_additions(tmp_path):
    old, new = tmp_path/'old', tmp_path/'new'
    old.mkdir()
    _git(old, 'init')
    _git(old, 'config', 'core.autocrlf', 'false')
    (old/'src').mkdir()
    (old/'src/science.py').write_bytes(b'# frozen science\n')
    measured = _commit(old)
    _git(tmp_path, '-c', 'core.autocrlf=false', 'clone', str(old), str(new))
    for name in a.ADDITIONS:
        target = new/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'# stage3 addition\n')
    active = _commit(new)
    result = a.compatible_source(measured, active, preflight_project=old, project=new)
    assert result['parents']['preflight_source'] != result['parents']['science_source']
    (new/'src/science.py').write_bytes(b'# unauthorized science change\n')
    active = _commit(new)
    with pytest.raises(ValueError, match='scientific source changed'):
        a.compatible_source(measured, active, preflight_project=old, project=new)
    (new/'src/science.py').write_bytes((old/'src/science.py').read_bytes())
    (new/'src/unregistered.py').write_bytes(b'# unknown\n')
    active = _commit(new)
    with pytest.raises(ValueError, match='allowlist'):
        a.compatible_source(measured, active, preflight_project=old, project=new)


def test_accounting_exact_job_and_success(monkeypatch):
    def output(value):
        monkeypatch.setattr(c.subprocess, 'run', lambda *args, **kwargs: subprocess.CompletedProcess(args,0,value,''))
    output('21835630|COMPLETED|0:0|\n')
    assert c.preflight_accounting('21835630')['state'] == 'COMPLETED'
    for value in ('', '21835631|COMPLETED|0:0|\n', '21835630|RUNNING|0:0|\n', '21835630|COMPLETED|1:0|\n'):
        output(value)
        with pytest.raises(ValueError):
            c.preflight_accounting('21835630')


def test_exact_plan_dry_live_journal_and_environment(tmp_path, monkeypatch):
    spec = fixture_spec(tmp_path)
    monkeypatch.setattr(c, 'validate_spec', lambda _: None)
    monkeypatch.setattr(sub, 'check_site', lambda _: None)
    calls = []
    def scheduler(args, **kwargs):
        assert args[0] == 'sbatch' and not any('${JOB_' in x for x in args)
        assert not any(k.startswith(('SBATCH_', 'SLURM_')) for k in kwargs['env'])
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, str(800+len(calls))+';sporc\n', '')
    monkeypatch.setattr(sub.subprocess, 'run', scheduler)
    monkeypatch.setenv('SBATCH_GRES', 'gpu:bad:8')
    monkeypatch.setenv('SLURM_JOB_ID', '99')
    value = sub.submit(spec)
    assert not calls and value['jobs'] == 21
    for row in value['commands']:
        cmd = row['command']
        assert '--partition=debug' in cmd and '--export=NONE' in cmd
        if row['task_id'].startswith(('train_', 'reduce_')):
            assert '--gres=gpu:a100:1' in cmd and '--mem=256000M' in cmd
    with pytest.raises(PermissionError):
        sub.submit(spec, execute=True, plan_hash='0'*64, authorization=c.AUTHORIZATION)
    ledger = sub.submit(spec, execute=True, plan_hash=value['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 21 and len(ledger['jobs']) == 21
    assert any('--dependency=afterok:803' in x for x in calls[3])  # U000 reducer.
    assert sub.submit(spec, execute=True, plan_hash=value['content_hash'], authorization=c.AUTHORIZATION) == ledger
    assert len(calls) == 21


def test_ambiguous_submit_is_not_automatically_retried(tmp_path, monkeypatch):
    spec = fixture_spec(tmp_path)
    monkeypatch.setattr(c, 'validate_spec', lambda _: None)
    monkeypatch.setattr(sub, 'check_site', lambda _: None)
    value = sub.submit(spec)
    calls = []
    def fail(args, **kwargs):
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError('Lost sbatch reply')
        return subprocess.CompletedProcess(args, 0, '801\n', '')
    monkeypatch.setattr(sub.subprocess, 'run', fail)
    with pytest.raises(RuntimeError):
        sub.submit(spec, execute=True, plan_hash=value['content_hash'], authorization=c.AUTHORIZATION)
    assert len(list((Path(spec['root'])/'submission/submission_ledger_journal').glob('*.json'))) == 1
    with pytest.raises(PermissionError, match='interrupted'):
        sub.submit(spec, execute=True, plan_hash=value['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 2


def test_cluster_test_only_checks_and_no_submission(tmp_path, monkeypatch):
    spec = fixture_spec(tmp_path)
    monkeypatch.setattr(c, 'validate_spec', lambda _: None)
    value, calls = sub.plan(spec), []
    def run(args, **kwargs):
        calls.append(args)
        if args[0] == 'scontrol':
            return subprocess.CompletedProcess(args,0,'ClusterName = sporc\n','')
        assert '--test-only' in args and '--wrap=true' in args
        assert not any('${JOB_' in x for x in args)
        return subprocess.CompletedProcess(args,0,'hypothetical start','')
    monkeypatch.setattr(sub.subprocess, 'run', run)
    sub.check_site(value)
    assert len(calls) == 4  # cluster, train, reduce, CPU summary
    monkeypatch.setattr(sub.subprocess, 'run', lambda *args, **kwargs: subprocess.CompletedProcess(args,0,'ClusterName = tigris',''))
    with pytest.raises(PermissionError, match='SPORC'):
        sub.check_site(value)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(17, 11)
    def forward(self, features, vectors, mask):
        mask = mask.float()
        if features.ndim == 4:
            features, mask = features.mean(1), mask.mean(1)
        return self.linear((features*mask).sum(-1)/mask.sum(-1).clamp_min(1))


def cpu_worker(prepared, tmp_path, monkeypatch):
    root, p = bind(prepared, monkeypatch)
    spec = fixture_spec(tmp_path, p)
    spec['admission']['prepared_root'] = str(root)
    spec = with_content_hash(spec)
    monkeypatch.setattr(c, 'validate_spec', lambda _: (p, {}))
    monkeypatch.setattr(w, 'execution', lambda *args: 'fixture-not-slurm')
    monkeypatch.setattr(w, 'memory_snapshot', lambda _: dict(rss_bytes=1, peak_rss_bytes=1, gpu_peak_bytes=0))
    monkeypatch.setattr(w, 'new_model', lambda n: (torch.manual_seed(n['initialization_seed']), TinyModel())[1])
    monkeypatch.setattr(w, 'train_kernel', lambda model, train, val, **kwargs:
        runner.train_kernel(model, train, val, **{**kwargs, 'device':'cpu'}))
    monkeypatch.setattr(w, 'predict', lambda model, train, **kwargs:
        runner.predict(model, train, **{**kwargs, 'device':'cpu'}))
    # The reducer's .to('cuda') is a hardware-only substitution, not bank math.
    monkeypatch.setattr(TinyModel, 'to', lambda self, *args, **kwargs: self)
    return spec


def test_complete_scientific_graph_real_cpu_kd_and_readbacks(prepared, tmp_path, monkeypatch):
    spec = cpu_worker(prepared, tmp_path, monkeypatch)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for task in c.tasks():
            receipt = w.run(spec, task['task_id'])
            assert receipt['final_test_accessed'] is False
            assert w.run(spec, task['task_id']) == receipt  # no rerun
        rows = w.result_rows(spec)
        assert len(rows) == 12 and all(r['state'] == 'COMPLETE' for r in rows)
        assert all(60 <= r['passes'] <= 100 for r in rows)
        assert w.completed(spec, 'complete')['result']['fits'] == 12
        train = cache.cache(spec, c.validate_spec(spec)[0], c.nodes()[3], 'train')
        with pytest.raises(ValueError, match='population'):
            w.teacher(spec, 'U000', train.identities[::-1].copy())
        checkpoint = Path(spec['root'])/'train_U000/selected.pt'
        checkpoint.write_bytes(checkpoint.read_bytes()+b'corrupt')
        with pytest.raises(ValueError, match='bytes differ'):
            w.completed(spec, 'train_DIRECT_D000')
    finally:
        torch.set_num_threads(threads)


def test_failed_fit_keeps_partial_and_missing_teacher_stops(prepared, tmp_path, monkeypatch):
    spec = cpu_worker(prepared, tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='Uncommitted dependency'):
        w.run(spec, 'train_DIRECT_D000')
    assert not (Path(spec['root'])/'train_DIRECT_D000').exists()
    def fail(*args, **kwargs):
        raise ValueError('Nonfinite fixture')
    monkeypatch.setattr(w, 'train_kernel', fail)
    with pytest.raises(ValueError, match='Nonfinite'):
        w.run(spec, 'train_M0HLT')
    assert not (Path(spec['root'])/'train_M0HLT/receipt.json').exists()
    with pytest.raises(FileExistsError):
        w.run(spec, 'train_M0HLT')


def test_technical_probes_cannot_be_scientific_fits(tmp_path):
    spec = fixture_spec(tmp_path)
    node = c.nodes()[0]
    report = kernel_artifact('KERNEL_TRAINING_REPORT', node=node,
        foundation_sha256=spec['prepared_sha256'], recipe_sha256=spec['training_recipe']['content_hash'],
        scientific_fit=True, acceptance_only=False, final_test_accessed=False, selected_weights_restored=True,
        passes=60, selected_pass=1, validation_history=[{}]*60)
    w.validate_training(spec, node, report)
    for changes in (dict(scientific_fit=False, acceptance_only=True, passes=1, validation_history=[{}]),
            dict(passes=59, validation_history=[{}]*59), dict(final_test_accessed=True),
            dict(foundation_sha256='1'*64), dict(selected_weights_restored=False)):
        with pytest.raises(ValueError, match='Scientific fit'):
            w.validate_training(spec, node, with_content_hash(dict(report, **changes)))


def test_scripts_and_old_science_unchanged():
    root = Path(__file__).resolve().parents[1]
    worker = (root/'sbatch/run_luka_fullsim_science.sh').read_text()
    assert worker.index('CUBLAS_WORKSPACE_CONFIG=:4096:8') < worker.index('exec python')
    assert '${PROJECT_DIR}/sbatch/common.sh' in worker and 'PYTHONNOUSERSITE=1' in worker
    assert '${CONDA_PREFIX}/lib' in worker and 'atlas_kd_sporc' in worker
    delta = subprocess.run(['git','diff','--name-status',a.PREFLIGHT_COMMIT,'--','src'],
        cwd=root,check=True,capture_output=True,text=True).stdout.splitlines()
    assert all(line.startswith('A\t') and line.split('\t')[1] in a.ADDITIONS for line in delta)


def test_downloaded_train_science_cache_all_coordinates(tmp_path, monkeypatch):
    """Real 128 TRAIN-row input diagnostic, never synthetic production admission."""
    from test_luka_fullsim_preflight_v2 import test_downloaded_train_rows_local_diagnostic
    from hlt_classification.luka_fullsim import preflight_cache as donor, stage2_cache
    # Donor authenticates eight real files before/after reading only selected
    # TRAIN ranges. It leaves its deliberately ephemeral diagnostic loader bound.
    test_downloaded_train_rows_local_diagnostic(monkeypatch, tmp_path)
    p, foundation, inventory, splits = cache.s.load_prepared(tmp_path)
    p = deepcopy(p)
    jets = list(donor.ParticleReader())
    assert len(jets) == 128
    refs = {r['file_index']: r for r in p['assignments']['train']}
    meters = {co: cache.s.ViewMeter() for co in cache.s.COORDINATES}
    from itertools import groupby
    for index, group in groupby(jets, key=lambda j: j.file_index):
        assigned = cache.s.load_assignment(tmp_path, refs[index])
        for row, jet in enumerate(group):
            for co, value in donor._values(jet, assigned, row, p, cache.s.COORDINATES).items():
                meters[co].add(jet, value)
    p['summaries']['train'] = {co: m.report() for co, m in meters.items()}
    monkeypatch.setattr(cache.s, 'load_prepared', lambda _: (p, foundation, inventory, splits))
    monkeypatch.setattr(cache, 'ParticleReader', lambda *args, **kwargs: iter(jets))
    monkeypatch.setattr(stage2_cache, 'ParticleReader', lambda *args, **kwargs: iter(jets))
    for coords in (('OFFLINE',), ('U000', 'U050'), ('U100', 'D066'), ('D033', 'D000')):
        bundle = cache.build(tmp_path, role='train', coordinates=coords, max_ram_bytes=2**30)
        for co in coords:
            reference = prepare_cache(tmp_path, role='train', coordinate=co, max_ram_bytes=2**30)
            for key, expected in reference.batch(np.arange(128)).items():
                np.testing.assert_array_equal(bundle[co].batch(np.arange(128))[key], expected)
    print('Exact science/v1 bytes on 128 downloaded TRAIN jets, seven views; no validation/test particles.')

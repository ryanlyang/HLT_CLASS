"""Process isolation is tested separately from real A100 resource acceptance."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys

import pytest
import torch

from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullsim import alljets_fusion_campaign as c, alljets_fusion_preflight as pre
from hlt_classification.luka_fullsim import alljets_fusion_worker as w, alljets_fusion_submission as sub
from hlt_classification.luka_fullsim import alljets_fusion_sync as sync
from hlt_classification.luka_fullsim import alljets_fusion_isolated_preflight as iso
from hlt_classification.luka_fullsim import alljets_fusion_isolated_recovery as recovery
from hlt_classification.jetclass2_delphes import runner
from test_luka_fullsim_alljets_fusion import fixture_spec, evidence_fixture, TinyModel
from test_luka_fullsim_stage2 import dataset, prepared, config
from test_luka_fullsim_alljets import all_prepared


class Measurement:
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def report(self): return dict(sampled_peak_tree_rss_bytes=2**30)


@pytest.fixture
def local_gate(all_prepared, tmp_path, monkeypatch):
    spec, parent, p = fixture_spec(all_prepared, tmp_path)
    monkeypatch.setattr(c, 'COUNTS', p['counts'])
    template = evidence_fixture()[0]
    gate = c.artifact('GATE', version=3, root=str(tmp_path/'isolated'), project_dir=str(tmp_path/'source'),
        source_commit='a'*40, source_snapshot=dict(content_hash='e'*64), parent_import=spec['parent_import'],
        prepared_sha256=p['content_hash'], **c.registration(3))
    Path(gate['root']).mkdir()
    runtime = {k:template[k] for k in iso.RUNTIME_KEYS}
    write_immutable_json(tmp_path/'prior_runtime.json', runtime)
    gate = with_content_hash(dict(gate, recovery=dict(original_runtime=file_ref(tmp_path/'prior_runtime.json'))))
    write_immutable_json(Path(gate['root'])/'gate_spec.json', gate)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    monkeypatch.setattr(c, 'validate_gate', lambda _:parent)
    monkeypatch.setattr(w, 'execution', lambda *a:'123')
    monkeypatch.setattr(pre, 'gpu_identity', lambda:runtime['gpu'])
    monkeypatch.setattr(pre, 'installed_environment', lambda:runtime['environment'])
    monkeypatch.setattr(pre, 'installed_parity', lambda *a, **kw:template['parities']['installed'])
    monkeypatch.setattr(pre, 'native_mask_parity', lambda *a, **kw:True)
    def parity(*a, **kw):
        return dict(deepcopy(template['parities']['FUSION_U050_probe_'+('bf16' if kw['bf16'] else 'fp32')]),
                    saved_tensor_storage=sync.PAIR_STORAGE)
    monkeypatch.setattr(sync, 'native_offload_parity', parity)
    monkeypatch.setattr(pre, 'memory_snapshot', lambda _:dict(rss_bytes=2**30, peak_rss_bytes=2**30, gpu_peak_bytes=2**30))
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda:None)
    monkeypatch.setattr(pre, 'clear', lambda:None)
    monkeypatch.setattr(iso, 'Measurement', Measurement)
    monkeypatch.setattr(sync, 'new_model', lambda n:(torch.manual_seed(n['initialization_seed']), TinyModel())[1])
    stats = template['parities']['FUSION_U050_probe_fp32']['offload_stats']
    events = []
    def stress(model, cache):
        events.append('train_stress')
        assert cache.role == 'train'
        return deepcopy(stats) if isinstance(cache, pre.PairedCache) else None
    monkeypatch.setattr(pre, 'stress', stress)
    validation = pre.validation_stress
    def val_stress(model, cache):
        events.append('validation_stress')
        validation(model, cache, device='cpu')
    monkeypatch.setattr(pre, 'validation_stress', val_stress)
    def fit(model, train, val, **kw):
        events.append('fit')
        report, state = runner.train_kernel(model, train, val, **{**kw, 'device':'cpu'})
        return with_content_hash(dict(report, runtime_seconds=max(.001, report['runtime_seconds']))), state
    monkeypatch.setattr(pre, 'train_kernel', fit)
    monkeypatch.setattr(pre, 'predict', lambda model, cache, **kw:runner.predict(model, cache, **{**kw, 'device':'cpu'}))
    launched = []
    def launch(g, kind, name, log):
        launched.append((kind, name))
        log.write_text('mocked subprocess; real synthetic CPU payloads')
        pid = 50000+len(launched)
        with monkeypatch.context() as local:
            local.setattr(iso.os, 'getpid', lambda:pid)
            iso.run_child(g, kind, name)
        return 0, pid
    monkeypatch.setattr(iso, '_launch', launch)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield gate, parent, p, events, launched
    torch.set_num_threads(threads)


def test_all_stresses_first_and_real_cpu_payload_admission(local_gate, monkeypatch):
    gate, parent, p, events, launched = local_gate
    report = pre.run(gate)
    assert launched == iso.sequence()
    assert launched[:2] == [('parity', 'FUSION_D000_D000_probe'), ('stress', 'FUSION_D000_D000_probe')]
    assert events == ['train_stress', 'validation_stress']*3 + ['fit']*3
    assert report['schema_version'] == 3 and len(report['isolated_children']) == 9
    assert c.evidence(gate) == (parent, p, report)
    monkeypatch.setattr(sub, 'accepted_jobs', lambda _:{'preflight':'123'})
    monkeypatch.setattr(c, 'preflight_accounting', lambda job:dict(job_id=job, state='COMPLETED', exit_code='0:0'))
    science = c.create_science(gate)
    assert science['schema_version'] == 3 and science['jobs'] == 16
    assert c.validate_spec(science) == (parent, p, report)
    assert science['pair_storage'] == sync.PAIR_STORAGE
    bad = deepcopy(report)
    bad['isolated_children'][0]['pid'] += 1
    with pytest.raises(ValueError, match='processes'):
        pre.validate_evidence(with_content_hash(bad), gate, p)
    bad = deepcopy(report)
    bad['isolated_children'].reverse()
    with pytest.raises(ValueError, match='stresses'):
        pre.validate_evidence(with_content_hash(bad), gate, p)
    bad = deepcopy(report)
    bad['parities']['FUSION_D000_D000_probe_fp32']['passed'] = False
    with pytest.raises(ValueError):
        pre.validate_evidence(with_content_hash(bad), gate, p)
    path = Path(gate['root'])/'preflight/children/parity_FUSION_D000_D000_probe/result.json'
    path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError):
        c.evidence(gate)


@pytest.mark.parametrize('failed_index', [0, 1, 3, 5, 6, 8])
def test_child_failure_stops_without_retry_or_admission(local_gate, monkeypatch, failed_index):
    gate, _, _, events, launched = local_gate
    calls = []
    real = iso._launch
    def launch(g, kind, name, log):
        index = len(calls)
        calls.append((kind, name))
        if index == failed_index:
            log.write_text('CUDA failure fixture')
            return 1, 9999
        return real(g, kind, name, log)
    monkeypatch.setattr(iso, '_launch', launch)
    with pytest.raises(RuntimeError, match='stopped before later probes'):
        pre.run(gate)
    assert len(calls) == failed_index+1
    assert not (Path(gate['root'])/'preflight/preflight.json').exists()
    assert load_json(Path(gate['root'])/'preflight/failure.json')['exit_code'] == 1
    if failed_index < 6:
        assert 'fit' not in events
    with pytest.raises(FileExistsError):
        pre.run(gate)


def test_runtime_gate_preserves_measurement_without_admission(local_gate, monkeypatch):
    gate, _, _, _, _ = local_gate
    original = pre.train_kernel
    def slow(*a, **kw):
        r, s = original(*a, **kw)
        return with_content_hash(dict(r, runtime_seconds=1264.)), s
    monkeypatch.setattr(pre, 'train_kernel', slow)
    with pytest.raises(ValueError, match='resource admission'):
        pre.run(gate)
    root = Path(gate['root'])/'preflight'
    r = load_json(root/'measurement.json')
    assert not r['fits_debug_24h'] and r['projected_train_minutes'] > 1440
    assert not (root/'preflight.json').exists()


def test_stress_child_never_runs_parity_or_population_fit(local_gate, monkeypatch):
    gate, _, _, events, _ = local_gate
    def forbidden(*a, **kw):
        pytest.fail('Stress process must start without earlier parity/model fits')
    monkeypatch.setattr(pre, 'installed_parity', forbidden)
    monkeypatch.setattr(pre, 'native_mask_parity', forbidden)
    monkeypatch.setattr(sync, 'native_offload_parity', forbidden)
    monkeypatch.setattr(pre, 'train_kernel', forbidden)
    row = iso.run_child(gate, 'stress', iso.ORDER[0])
    assert row['completed'] and row['stress_batch'] == 256 and row['stress_steps'] == 3
    assert events == ['train_stress', 'validation_stress']
    assert 'parities' not in row and 'training' not in row


def test_no_cuda_query_after_exception(tmp_path):
    calls = []
    def sample():
        calls.append('sample')
        if len(calls) > 1:
            pytest.fail('CUDA sampled after failure')
        return {'fixture':1}
    log = iso.SafePhaseLog(tmp_path, sample)
    with pytest.raises(RuntimeError, match='illegal'):
        with log.phase('backward'):
            raise RuntimeError('CUDA illegal access fixture')
    assert calls == ['sample']
    assert load_json(tmp_path/'phases/00-end.json')['status'] == 'failed'


def test_launcher_creates_actual_fresh_processes(tmp_path):
    # Stub only the GPU child entry point: exercise Popen/env/logs/PID boundaries.
    scripts = tmp_path/'scripts'
    scripts.mkdir()
    (scripts/'luka_fullsim_alljets_fusion.py').write_text(
        'import os,json\nprint(json.dumps({k:os.environ[k] for k in '
        '["PYTHONNOUSERSITE","PYTHONDONTWRITEBYTECODE","CUDA_LAUNCH_BLOCKING","PYTHONPATH"]}))\n')
    gate = dict(project_dir=str(tmp_path), root=str(tmp_path/'gate'))
    pids = []
    for i in range(2):
        path = tmp_path/f'child{i}.out'
        code, pid = iso._launch(gate, 'stress', iso.ORDER[0], path)
        assert code == 0
        pids.append(pid)
        env = json.loads(path.read_text())
        assert env == dict(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', CUDA_LAUNCH_BLOCKING='0',
                           PYTHONPATH=str(tmp_path/'src'))
    assert len(set(pids)) == 2


def test_v3_submission_exact_one_preflight_and_no_duplicate(tmp_path, monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as exact
    gate = c.artifact('GATE', version=3, root=str(tmp_path), project_dir=str(tmp_path/'source'), **c.registration(3))
    monkeypatch.setattr(c, 'validate_gate', lambda _:None)
    monkeypatch.setattr(sub, 'check_site', lambda _:None)
    calls = []
    def submit(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, '888', '')
    monkeypatch.setattr(exact.subprocess, 'run', submit)
    plan = sub.submit(gate)
    assert plan['jobs'] == 1 and not calls
    assert '--time=08:00:00' in plan['commands'][0]['command']
    with pytest.raises(PermissionError):
        sub.submit(gate, execute=True, plan_hash='f'*64, authorization=c.AUTHORIZATION)
    ledger = sub.submit(gate, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    assert sub.submit(gate, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION) == ledger
    assert len(calls) == 1 and sub.accepted_jobs(gate) == {'preflight':'888'}


def test_scientific_source_and_recipe_unchanged():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(['git', 'diff', '--name-only', recovery.FAILED_COMMIT, '--', 'src'],
                            cwd=root, check=True, text=True, capture_output=True)
    from hlt_classification.luka_fullsim import alljets_batch128 as batch128
    from hlt_classification.luka_fullsim import alljets_report_recovery as report_recovery
    from hlt_classification.luka_fullsim import alljets_30epoch as thirty
    assert set(result.stdout.splitlines()) <= (recovery.CHANGES | recovery.ADDITIONS | batch128.CHANGES | batch128.ADDITIONS
                                               | report_recovery.CHANGES | report_recovery.ADDITIONS
                                               | thirty.CHANGES | thirty.ADDITIONS | thirty.FULLSTAT)
    assert {k:v for k,v in c.registration(3).items() if k != 'preflight_execution'} == c.registration(2)
    assert iso.ORDER[0] == 'FUSION_D000_D000_probe'
    assert c.registration(3)['preflight_execution']['automatic_retry'] is False
    helper = (root/'scripts/queue_luka_fusion_isolated_preflight.sh').read_text()
    assert 'create-isolated-recovery' in helper and 'create-science' not in helper
    assert 'PYTHONNOUSERSITE=1' in helper and '${CONDA_PREFIX}/lib' in helper


def test_isolated_source_guard_rejects_shared_model_changes(tmp_path, monkeypatch):
    before, after = tmp_path/'before', tmp_path/'after'
    names = recovery.CHANGES | {'src/hlt_classification/luka_fullsim/alljets_fusion_sync.py'}
    for root in (before, after):
        for name in names:
            (root/name).parent.mkdir(parents=True, exist_ok=True)
            (root/name).write_text('unchanged')
    monkeypatch.setattr(recovery, 'validate_source_snapshot', lambda *a, **kw:None)
    monkeypatch.setattr(recovery, '_files', lambda root:names if root == before else names|recovery.ADDITIONS)
    old = dict(project_dir=str(before), source_snapshot={})
    recovery.compatible_source(old, {}, after)
    (after/'src/hlt_classification/luka_fullsim/alljets_fusion_sync.py').write_text('changed')
    with pytest.raises(ValueError, match='source changed'):
        recovery.compatible_source(old, {}, after)


@pytest.fixture
def failed_gate(tmp_path, monkeypatch):
    root = tmp_path/'failed'
    (root/'preflight/phases').mkdir(parents=True)
    parent = tmp_path/'parent.json'
    write_immutable_json(parent, dict(root='parent'))
    old = c.artifact('GATE', version=2, root=str(root), project_dir=str(tmp_path/'old_source'),
        source_commit=recovery.FAILED_COMMIT, source_snapshot=dict(git_commit=recovery.FAILED_COMMIT),
        parent_import=dict(spec=file_ref(parent)), prepared_sha256='e'*64, recovery={}, **c.registration(2))
    write_immutable_json(root/'gate_spec.json', old)
    write_immutable_json(root/'preflight/runtime.json', dict(job_id=recovery.FAILED_JOB))
    write_immutable_json(root/'preflight/phases/19-end.json', dict(
        phase='FUSION_D000_D000_probe_train_stress256_steps3', status='failed', error='CUDA illegal memory access'))
    (root/f'preflight-{recovery.FAILED_JOB}.out').write_text('original failed Slurm log')
    monkeypatch.setattr(recovery, 'compatible_source', lambda *a:dict(fixture=True))
    def original(argv, **kw):
        assert kw['cwd'] == old['project_dir']
        assert kw['env']['PYTHONPATH'] == str(Path(old['project_dir'])/'src')
        assert argv[-2:] == [recovery.ORIGINAL_CHECK, str(root/'gate_spec.json')]
        return subprocess.CompletedProcess(argv, 0, 'LUKA_ISOLATED_AUTH='+json.dumps(
            dict(gate=old['content_hash'], jobs=dict(preflight=recovery.FAILED_JOB))), '')
    monkeypatch.setattr(recovery.subprocess, 'run', original)
    monkeypatch.setattr(recovery.sources, 'source', lambda *a:dict(content_hash='a'*64))
    monkeypatch.setattr(c, 'protect', lambda *a:None)
    monkeypatch.setattr(recovery, 'failed_accounting', lambda job:dict(job_id=job, state='FAILED', exit_code='1:0'))
    return root, old


def test_recovery_authenticates_failed_source_preserves_cache_and_rejects_tampering(failed_gate, tmp_path, monkeypatch):
    root, old = failed_gate
    before = (root/'gate_spec.json').read_bytes()
    gate = recovery.create(failed_gate=root/'gate_spec.json', root=tmp_path/'new',
                           project=tmp_path/'new_source', commit='a'*40)
    assert gate['schema_version'] == 3 and gate['parent_import'] == old['parent_import']
    assert gate['prepared_sha256'] == old['prepared_sha256']
    assert c.validate_gate(gate) == {'root':'parent'}
    assert (root/'gate_spec.json').read_bytes() == before
    assert not (Path(gate['root'])/'prepared').exists()
    with pytest.raises(FileExistsError):
        recovery.create(failed_gate=root/'gate_spec.json', root=tmp_path/'new',
                        project=tmp_path/'new_source', commit='a'*40)
    with pytest.raises(PermissionError, match='overlaps'):
        recovery.create(failed_gate=root/'gate_spec.json', root=root/'bad',
                        project=tmp_path/'new_source', commit='a'*40)
    (root/f'preflight-{recovery.FAILED_JOB}.out').write_text('changed')
    with pytest.raises(ValueError):
        c.validate_gate(gate)


def test_recovery_rejects_live_job_and_wrong_failure(failed_gate, tmp_path, monkeypatch):
    root, old = failed_gate
    monkeypatch.setattr(recovery, 'failed_accounting', lambda job:dict(job_id=job, state='RUNNING', exit_code='0:0'))
    with pytest.raises(ValueError, match='accounting'):
        recovery.create(failed_gate=root/'gate_spec.json', root=tmp_path/'new',
                        project=tmp_path/'new_source', commit='a'*40)
    assert not (tmp_path/'new').exists()
    (root/'preflight/phases/19-end.json').write_text(json.dumps(dict(
        phase='FUSION_D000_D000_probe_train_stress256_steps3', status='failed', error='Not the reviewed error')))
    with pytest.raises(ValueError, match='Expected D000'):
        recovery.authenticate(root/'gate_spec.json', {}, tmp_path/'new_source')


def test_earlier_admission_cannot_admit_v3(tmp_path):
    report, gate, p = evidence_fixture()
    gate.update(root=str(tmp_path), schema_version=3, **c.registration(3))
    with pytest.raises(ValueError, match='versions'):
        pre.validate_evidence(report, gate, p)


@pytest.mark.parametrize('same', [True, False])
def test_fresh_untraced_production_cuda_stress(same):
    pytest.importorskip('weaver')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    # Full batch and production code, but SMALL support on the local 8GiB GPU.
    # This is not a claim of full-support A100 acceptance.
    program = '''
import sys
from types import SimpleNamespace
import numpy as np
import torch
from hlt_classification.luka_fullsim import alljets_fusion_sync as sync, alljets_fusion_campaign as c
from hlt_classification.context_fusion.inputs import PairedCache
from hlt_classification.context_fusion.worker import stress
from test_jetclass2_dzfix_fusion_offload import inputs
raw,other=inputs()
class Cache:
    role='train'; foundation_sha256='f'*64; coordinate_name='D000'; nbytes=1000
    def __init__(self,value):
        self.value=value; self.labels=value['labels']
        self.identities=np.arange(128,dtype=np.uint8).reshape(4,32)
        self.blocks=[SimpleNamespace(offsets=np.arange(5)*value['mask'].shape[-1])]
    def __len__(self): return 4
    def batch(self,ix): return {k:v[ix] for k,v in self.value.items()}
same=sys.argv[1]=='True'
a=Cache(raw); views=PairedCache(a,a if same else Cache(other))
node=next(n for n in c.nodes() if n['node_id']==('FUSION_D000_D000' if same else 'FUSION_U050'))
sync.old.validate_offload_stats(stress(sync.new_model(node),views),calls=3)
print('UNTRACED_PRODUCTION_STRESS_PASS')
'''
    root = Path(__file__).resolve().parents[1]
    import os
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root/'tests'),os.environ.get('PYTHONPATH','')]),
               CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_LAUNCH_BLOCKING='0', PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', '-c', program, str(same)], env=env,
                            cwd=root, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'UNTRACED_PRODUCTION_STRESS_PASS' in result.stdout

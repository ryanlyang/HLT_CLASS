"""v2 recovery/factory isolation; local tests do not admit SPORC science."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess

import numpy as np
import pytest
import torch

from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullsim import alljets_fusion_sync as sync
from hlt_classification.luka_fullsim import alljets_fusion_recovery as recovery
from hlt_classification.luka_fullsim import alljets_fusion_campaign as c, alljets_fusion_preflight as pre
from hlt_classification.luka_fullsim import alljets_fusion_worker as w, alljets_fusion_submission as sub
from hlt_classification.luka_fullsim import alljets_fusion_diagnostic as diagnostic
from test_jetclass2_dzfix_fusion_offload import inputs, pair_native
from test_jetclass2_dzfix_fusion_chain import fake_native
from test_luka_fullsim_alljets_fusion import evidence_fixture, fixture_spec, cpu_worker
from test_luka_fullsim_stage2 import dataset, prepared, config
from test_luka_fullsim_alljets import all_prepared


def test_unpack_waits_and_preserves_physical_layout():
    calls = []
    class Buffer:
        def to(self, device, *, non_blocking):
            calls.append((device, non_blocking))
            return self
        def as_strided(self, shape, strides):
            calls.append((shape, strides))
            return self
    buffer = Buffer()
    assert sync.unpack(('cuda', buffer, (3,4), (1,8))) is buffer
    assert calls == [('cuda', False), ((3,4), (1,8))]
    assert sync.unpack(('cpu', buffer, None, None)) is buffer


@pytest.mark.parametrize('bf16', [False, True])
@pytest.mark.parametrize('same', [False, True])
def test_actual_production_factory_installed_weaver_parity(bf16, same):
    pytest.importorskip('weaver')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    raw, other = inputs()
    before = sync.old.DzfixFusionParticleTransformer
    report = sync.native_offload_parity(raw, raw if same else other, device='cuda', bf16=bf16)
    assert report['passed'] and report['saved_tensor_storage'] == sync.PAIR_STORAGE
    assert sync.old.DzfixFusionParticleTransformer is before
    sync.old.validate_offload_stats(report['offload_stats'], calls=3)


def test_parity_constructor_restored_on_error(monkeypatch):
    before = sync.old.DzfixFusionParticleTransformer
    def fail(*a, **kw):
        assert sync.old.DzfixFusionParticleTransformer is sync.SynchronousFusionModel
        raise RuntimeError('injected')
    monkeypatch.setattr(sync.old, 'native_offload_parity', fail)
    with pytest.raises(RuntimeError):
        sync.native_offload_parity(None, None, device='cpu')
    assert sync.old.DzfixFusionParticleTransformer is before


def test_factories_preserve_seeds_and_state_layout_without_global_patch(pair_native):
    from hlt_classification.context_fusion.inputs import new_model as old_factory
    for node in c.nodes():
        a, b = old_factory(node), sync.new_model(node)
        assert a.state_dict().keys() == b.state_dict().keys()
        assert all(torch.equal(value, b.state_dict()[key]) for key, value in a.state_dict().items())
        assert isinstance(b, sync.SynchronousFusionModel) == (node['context_coordinate'] is not None)
        a.load_state_dict(b.state_dict(), strict=True)
    assert sync.old._unpack_pair_tensor is diagnostic.fusion._unpack_pair_tensor


@pytest.mark.parametrize('same', [False, True])
def test_production_batch256_three_steps_real_weaver_small_support(same):
    from hlt_classification.context_fusion.inputs import PairedCache
    pytest.importorskip('weaver')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    raw, other = inputs()
    class Cache:
        role = 'train'
        foundation_sha256 = 'f'*64
        coordinate_name = 'D000'
        nbytes = 1000
        def __init__(self, values):
            self.values = values
            self.labels = values['labels']
            self.identities = np.arange(128, dtype=np.uint8).reshape(4,32)
            self.blocks = [SimpleNamespace(offsets=np.arange(5)*values['mask'].shape[-1])]
        def __len__(self): return 4
        def batch(self, ix): return {k:v[ix] for k,v in self.values.items()}
    a = Cache(raw)
    batch = PairedCache(a, a if same else Cache(other))
    node = next(n for n in c.nodes() if n['node_id']==('FUSION_D000_D000' if same else 'FUSION_U050'))
    model = sync.new_model(node)
    assert isinstance(model, sync.SynchronousFusionModel)
    stats = diagnostic.stress(model, batch)
    sync.old.validate_offload_stats(stats, calls=3)
    del model
    torch.cuda.empty_cache()


def test_v1_registration_unchanged_and_v2_cannot_relabel_old_evidence():
    r, gate, p = evidence_fixture()
    assert pre.validate_evidence(r, gate, p) == r
    assert {k:v for k,v in c.registration(2).items() if k != 'pair_storage'} == c.registration()
    assert c.registration(2)['pair_storage'] == sync.PAIR_STORAGE
    gate = dict(gate, schema_version=2, pair_storage=sync.PAIR_STORAGE)
    with pytest.raises(ValueError, match='versions'):
        pre.validate_evidence(r, gate, p)
    r = with_content_hash(dict(r, schema_version=2, contract='LUKA_ALLJETS_FUSION_PREFLIGHT/v2',
                              pair_storage=sync.PAIR_STORAGE))
    with pytest.raises(ValueError, match='offload parity'):
        pre.validate_evidence(r, gate, p)
    for key, row in r['parities'].items():
        if key not in ('installed', 'native_mask'):
            row['saved_tensor_storage'] = deepcopy(sync.PAIR_STORAGE)
    r = with_content_hash(r)
    assert pre.validate_evidence(r, gate, p) == r
    for m in r['measurements']:
        m['training'] = with_content_hash(dict(m['training'], runtime_seconds=1240.))
    r['projected_train_minutes'], r['projected_reduce_minutes'] = pre.projections(r['measurements'])
    with pytest.raises(ValueError, match='debug24h'):
        pre.validate_evidence(with_content_hash(r), gate, p)


def test_v2_one_preflight_exact_submission_and_dedup(tmp_path, monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as exact
    root = tmp_path/'recovery'
    root.mkdir()
    gate = c.artifact('GATE', version=2, root=str(root), project_dir=str(tmp_path/'project'), **c.registration(2))
    monkeypatch.setattr(c, 'validate_gate', lambda _: None)
    monkeypatch.setattr(sub, 'check_site', lambda _: None)
    calls = []
    def scheduler(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, '321', '')
    monkeypatch.setattr(exact.subprocess, 'run', scheduler)
    plan = sub.submit(gate)
    assert not calls and plan['jobs'] == 1 and plan['schema_version'] == 2
    argv = plan['commands'][0]['command']
    assert '--time=08:00:00' in argv and '--mem=256000M' in argv and '--partition=debug' in argv
    assert not plan['commands'][0]['dependencies']
    with pytest.raises(PermissionError):
        sub.submit(gate, execute=True, plan_hash='f'*64, authorization=c.AUTHORIZATION)
    first = sub.submit(gate, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    assert sub.submit(gate, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION) == first
    assert sub.accepted_jobs(gate) == {'preflight':'321'} and len(calls) == 1


def test_interrupted_v2_submit_is_not_retried(tmp_path, monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as exact
    root = tmp_path/'recovery'
    root.mkdir()
    gate = c.artifact('GATE', version=2, root=str(root), project_dir=str(tmp_path/'project'), **c.registration(2))
    monkeypatch.setattr(c, 'validate_gate', lambda _: None)
    monkeypatch.setattr(sub, 'check_site', lambda _: None)
    plan = sub.submit(gate)
    def fail(*a, **kw): raise RuntimeError('lost scheduler reply')
    monkeypatch.setattr(exact.subprocess, 'run', fail)
    with pytest.raises(RuntimeError):
        sub.submit(gate, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    with pytest.raises(PermissionError, match='interrupted'):
        sub.submit(gate, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)


def test_v2_worker_uses_repaired_factory_for_both_routes(all_prepared, tmp_path, monkeypatch):
    spec, parent, p = fixture_spec(all_prepared, tmp_path)
    spec = with_content_hash(dict(spec, schema_version=2, contract='LUKA_ALLJETS_FUSION_SPEC/v2',
                                 pair_storage=sync.PAIR_STORAGE))
    cpu_worker(monkeypatch, p, parent)
    factory, called = w.new_model, []
    def new(node):
        called.append(node['node_id'])
        return factory(node)
    monkeypatch.setattr(sync, 'new_model', new)
    monkeypatch.setattr(w, 'new_model', lambda _: (_ for _ in ()).throw(AssertionError('old factory used')))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for task in c.tasks():
            receipt = w.run(spec, task['task_id'])
            assert receipt['schema_version'] == 2
        assert all(node['node_id'] in called for node in c.nodes())
        assert len(w.result_rows(spec)) == 17
        assert w.completed(spec, 'complete')['result']['fits'] == 8
    finally:
        torch.set_num_threads(threads)


def diagnostic_fixture(tmp_path, monkeypatch):
    root = tmp_path/'diagnostic'
    root.mkdir()
    spec = c.artifact('CUDA_DIAGNOSTIC_SPEC', root=str(root))
    gate = dict(prepared_sha256='b'*64)
    runtime = dict(environment={'installed':'original'}, gpu={'name':'A100'})
    template = evidence_fixture()[0]['parities']['FUSION_U050_probe_fp32']
    rows = []
    for name, node, candidate, blocking in diagnostic.CASES:
        output = root/'cases'/name
        output.mkdir(parents=True)
        witness = dict(node=next(n for n in c.nodes() if n['node_id']==node),
            prepared_sha256='b'*64, train_rows=c.COUNTS['train'], batch_size=256, candidate=candidate)
        write_immutable_json(output/'witness.json', witness)
        write_immutable_json(output/'runtime.json', dict(runtime, job_id='123', cuda_launch_blocking='1' if blocking else '0'))
        fields = dict(name=name, completed=candidate, science_admissible=False)
        if candidate:
            parities = {}
            for key, precision in [('False','fp32'), ('True','bf16')]:
                parities[key] = dict(report=dict(template, precision=precision, tolerance=sync.old.PARITY_TOLERANCES[precision]),
                    candidate=diagnostic.POLICY['candidate'], cuda_restores=5)
            fields.update(witness=witness, parities=parities, offload=template['offload_stats'],
                          synchronous_restore=dict(cuda_restores=441))
        else:
            fields['error'] = 'RuntimeError: CUDA illegal memory access'
        result = c.artifact('CUDA_DIAGNOSTIC_CASE' if candidate else 'CUDA_DIAGNOSTIC_FAILURE',
            parents=dict(spec=spec['content_hash']), **fields)
        write_immutable_json(output/('result.json' if candidate else 'failure.json'), result)
        (root/(name+'.out')).write_text('fixture log')
        rows.append(dict(name=name, completed=candidate, returncode=0 if candidate else 1,
            log=file_ref(root/(name+'.out'), root=root),
            outputs=[file_ref(path, root=root) for path in sorted(output.iterdir())]))
    report = c.artifact('CUDA_DIAGNOSTIC_REPORT', parents=dict(spec=spec['content_hash']), cases=rows,
                       all_cases_completed=False, science_admissible=False)
    monkeypatch.setattr(recovery, 'REPORT_HASH', report['content_hash'])
    return spec, report, gate, runtime


def test_diagnostic_evidence_full_hashes_and_candidate_parity(tmp_path, monkeypatch):
    spec, report, gate, runtime = diagnostic_fixture(tmp_path, monkeypatch)
    recovery.check_report(spec, report, gate, runtime, '123')
    with pytest.raises(ValueError, match='environment/job'):
        recovery.check_report(spec, report, gate, runtime, '999')
    bad = with_content_hash(dict(report, science_admissible=True))
    with pytest.raises(ValueError):
        recovery.check_report(spec, bad, gate, runtime, '123')
    path = Path(spec['root'])/'cases/sync_restore_d000/result.json'
    path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError):
        recovery.check_report(spec, report, gate, runtime, '123')


def test_source_guard_rejects_data_or_shared_model_changes(tmp_path, monkeypatch):
    old, new = tmp_path/'old', tmp_path/'new'
    names = {'src/hlt_classification/jetclass2_delphes/dzfix_fusion_model.py'} | recovery.CHANGES
    for root in (old, new):
        for name in names:
            (root/name).parent.mkdir(parents=True, exist_ok=True)
            (root/name).write_text('unchanged')
    monkeypatch.setattr(recovery, 'validate_source_snapshot', lambda *a, **k: None)
    monkeypatch.setattr(recovery, '_files', lambda path: names if Path(path)==old else names|recovery.ADDITIONS)
    spec = dict(project_dir=str(old), source_snapshot={})
    recovery.compatible_source(spec, {}, new)
    (new/'src/hlt_classification/jetclass2_delphes/dzfix_fusion_model.py').write_text('modified')
    with pytest.raises(ValueError, match='scientific source'):
        recovery.compatible_source(spec, {}, new)


def test_production_execution_rejects_diagnostic_backend(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    monkeypatch.setenv('CUDA_LAUNCH_BLOCKING', '1')
    with pytest.raises(PermissionError, match='launch blocking'):
        w.execution(dict(schema_version=2, pair_storage=sync.PAIR_STORAGE))


def test_recovery_create_validate_parent_preservation_and_live_rejection(tmp_path, monkeypatch):
    diagnostic_root, old_root = tmp_path/'diagnostic', tmp_path/'original'
    diagnostic_root.mkdir()
    old_root.mkdir()
    parent_path = tmp_path/'parent.json'
    write_immutable_json(parent_path, dict(root='parent'))
    old = c.artifact('GATE', root=str(old_root), project_dir=str(tmp_path/'oldsource'),
        parent_import=dict(spec=file_ref(parent_path)), prepared_sha256='e'*64, **c.registration())
    diagnostic_spec = dict(root=str(diagnostic_root), project_dir=str(tmp_path/'diagnostic_source'))
    for root, filename, payload in ((old_root, 'gate_spec.json', old),
                                    (old_root, 'runtime.json', dict(job_id='1')),
                                    (diagnostic_root, 'diagnostic_spec.json', diagnostic_spec),
                                    (diagnostic_root, 'diagnostic_report.json', dict(fixture=True))):
        write_immutable_json(root/filename, payload)
    reuse = dict(diagnostic=file_ref(diagnostic_root/'diagnostic_spec.json'),
        report=file_ref(diagnostic_root/'diagnostic_report.json'), original_gate=file_ref(old_root/'gate_spec.json'),
        original_runtime=file_ref(old_root/'runtime.json'), source_reuse={'fixture':True},
        accounting={name:dict(job_id=job, state='FAILED', exit_code='1:0')
                    for name,job in [('original','1'),('diagnostic','2')]})
    monkeypatch.setattr(recovery.sources, 'source', lambda *a:dict(content_hash='a'*64))
    monkeypatch.setattr(recovery, 'authenticate', lambda *a:(old,diagnostic_spec,reuse))
    monkeypatch.setattr(c, 'protect', lambda *a: None)
    monkeypatch.setattr(recovery, 'failed_accounting', lambda job:dict(job_id=job, state='FAILED', exit_code='1:0'))
    before = (old_root/'gate_spec.json').read_bytes()
    gate = recovery.create(diagnostic_path=reuse['diagnostic']['path'], root=tmp_path/'new',
                           project=tmp_path/'new_source', commit='a'*40)
    assert gate['schema_version'] == 2 and gate['parent_import'] == old['parent_import']
    assert c.validate_gate(gate) == {'root':'parent'}
    assert (old_root/'gate_spec.json').read_bytes() == before
    assert not (Path(gate['root'])/'prepared').exists()
    assert sub.render(gate)['jobs'] == 1
    with pytest.raises(FileExistsError):
        recovery.create(diagnostic_path=reuse['diagnostic']['path'], root=tmp_path/'new',
                        project=tmp_path/'new_source', commit='a'*40)
    with pytest.raises(PermissionError, match='overlaps'):
        recovery.create(diagnostic_path=reuse['diagnostic']['path'], root=old_root/'bad',
                        project=tmp_path/'new_source', commit='a'*40)
    def live(job):
        raise PermissionError('must be terminal')
    monkeypatch.setattr(recovery, 'failed_accounting', live)
    with pytest.raises(PermissionError, match='terminal'):
        recovery.create(diagnostic_path=reuse['diagnostic']['path'], root=tmp_path/'live',
                        project=tmp_path/'new_source', commit='a'*40)
    assert not (tmp_path/'live').exists()
    (diagnostic_root/'diagnostic_report.json').write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        c.validate_gate(gate)


def test_authentication_uses_original_diagnostic_interpreter_path(tmp_path, monkeypatch):
    root = tmp_path/'diagnostic'
    root.mkdir()
    project = tmp_path/'pinned_diagnostic'
    spec = c.artifact('CUDA_DIAGNOSTIC_SPEC', root=str(root), project_dir=str(project),
        source_commit=recovery.DIAGNOSTIC_COMMIT, source_snapshot=dict(git_commit=recovery.DIAGNOSTIC_COMMIT),
        policy=diagnostic.POLICY)
    old_root = tmp_path/'original'
    old_root.mkdir()
    gate = c.artifact('GATE', root=str(old_root), source_commit=recovery.ORIGINAL_COMMIT, **c.registration())
    write_immutable_json(old_root/'gate_spec.json', gate)
    write_immutable_json(old_root/'runtime.json', dict(job_id='1'))
    spec = with_content_hash(dict(spec, original_gate=file_ref(old_root/'gate_spec.json'),
                                 original_runtime=file_ref(old_root/'runtime.json')))
    path = root/'diagnostic_spec.json'
    write_immutable_json(path, spec)
    write_immutable_json(root/'diagnostic_report.json', dict(fixture=True))
    monkeypatch.setattr(recovery, 'compatible_source', lambda *a: {})
    monkeypatch.setattr(recovery, 'check_report', lambda *a: None)
    def original(argv, **kw):
        assert kw['cwd'] == str(project) and kw['env']['PYTHONPATH'] == str(project/'src')
        assert kw['env']['PYTHONDONTWRITEBYTECODE'] == '1' and kw['env']['PYTHONNOUSERSITE'] == '1'
        assert argv[-2:] == [recovery.ORIGINAL_CHECK, str(path)]
        return subprocess.CompletedProcess(argv, 0, 'LUKA_SYNC_AUTH='+json.dumps(dict(
            gate=gate['content_hash'], diagnostic=spec['content_hash'], jobs={'diagnostic':'2'})), '')
    monkeypatch.setattr(recovery.subprocess, 'run', original)
    old, _, reuse = recovery.authenticate(path, {}, tmp_path/'new')
    assert old == gate and reuse['accounting']['diagnostic']['job_id'] == '2'


def test_original_model_diagnostic_and_science_declared_helpers_unchanged():
    root = Path(__file__).resolve().parents[1]
    paths = ['jetclass2_delphes/dzfix_fusion_model.py', 'context_fusion/inputs.py', 'context_fusion/worker.py',
             'luka_fullsim/regular_cache.py', 'luka_fullsim/alljets_campaign.py',
             'luka_fullsim/science_campaign.py', 'luka_fullsim/alljets_fusion_diagnostic.py']
    for path in paths:
        relative = 'src/hlt_classification/'+path
        pinned = subprocess.run(['git', 'show', recovery.DIAGNOSTIC_COMMIT+':'+relative],
            cwd=root, check=True, capture_output=True).stdout.decode().replace('\r\n', '\n')
        assert (root/relative).read_text() == pinned
    helper = (root/'scripts/queue_luka_fusion_sync_recovery.sh').read_text()
    assert 'create-recovery' in helper and 'create-science' not in helper
    assert 'PYTHONNOUSERSITE=1' in helper and '${CONDA_PREFIX}/lib' in helper
    assert 'NUMEXPR_NUM_THREADS=1' in helper

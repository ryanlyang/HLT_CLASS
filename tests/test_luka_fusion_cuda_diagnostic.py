"""Local diagnostic contracts and candidate parity, not A100 admission."""
from pathlib import Path
from types import SimpleNamespace
import subprocess
import runpy
import sys

import numpy as np
import pytest
import torch

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.luka_fullsim import alljets_fusion_diagnostic as d
from test_jetclass2_dzfix_fusion_offload import pair_native, inputs
from test_jetclass2_dzfix_fusion_chain import fake_native
from test_luka_fullsim_stage2 import dataset, prepared, config
from test_luka_fullsim_alljets import all_prepared
from test_luka_fullsim_alljets_fusion import fixture_spec, evidence_fixture


def test_sync_restore_waits_preserves_layout_and_restores_on_failure():
    calls=[]
    class Values:
        def to(self,device,*,non_blocking):
            calls.append((device,non_blocking))
            return self
        def as_strided(self,shape,strides):
            calls.append((shape,strides))
            return 'restored'
    original=d.fusion._unpack_pair_tensor
    with pytest.raises(RuntimeError):
        with d.synchronous_restore() as stats:
            assert d.fusion._unpack_pair_tensor(('cuda',Values(),(3,4),(1,8)))=='restored'
            assert stats['cuda_restores']==1
            assert calls==[('cuda',False),((3,4),(1,8))]
            assert d.fusion._unpack_pair_tensor(('cpu','unchanged',None,None))=='unchanged'
            raise RuntimeError('injected')
    assert d.fusion._unpack_pair_tensor is original


@pytest.mark.parametrize('bf16',[False,True])
def test_synthetic_candidate_training_parity(pair_native,bf16,monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    # Include nontrivial masked, transposed pair activations.
    raw,other=inputs()
    for item in (raw,other):
        for key in ('features','vectors','mask'):
            item[key]=np.tile(item[key],(1,1,3))
    report=d.candidate_parity(raw,other,bf16=bf16)
    assert report['report']['passed'] and report['cuda_restores']>0


@pytest.mark.parametrize('bf16',[False,True])
@pytest.mark.parametrize('same',[False,True])
def test_installed_weaver_candidate_parity(bf16,same,monkeypatch):
    pytest.importorskip('weaver')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    raw,other=inputs()
    for item in (raw,other):
        for key in ('features','vectors','mask'):
            item[key]=np.tile(item[key],(1,1,2))
    report=d.candidate_parity(raw,raw if same else other,bf16=bf16)
    assert report['report']['passed'] and report['cuda_restores']>0


def test_kernel_trace_preserves_values_and_records_failing_op():
    rows=[]
    with d.KernelTrace(rows.append):
        result=torch.mm(torch.ones(2,3),torch.ones(3,4))
        torch.testing.assert_close(result,torch.full((2,4),3.))
        with pytest.raises(RuntimeError):
            torch.mm(torch.ones(2,3),torch.ones(2,4))
    assert rows[-1]['status']=='failed'
    assert rows[-1]['operation']=='aten.mm.default'
    assert rows[-1]['args'][0]['shape']==[2,3]


@pytest.mark.parametrize('synchronous',[False,True])
def test_installed_weaver_batch256_traced_stress_small_support(synchronous,monkeypatch):
    """Actual installed model/optimizer/offload/trace; NOT production support."""
    from contextlib import nullcontext
    from hlt_classification.context_fusion.inputs import PairedCache
    pytest.importorskip('weaver')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    raw,_=inputs()
    class Cache:
        role='train'
        foundation_sha256='f'*64
        coordinate_name='D000'
        labels=raw['labels']
        identities=np.arange(4*32,dtype=np.uint8).reshape(4,32)
        blocks=[SimpleNamespace(offsets=np.arange(5)*4)]
        nbytes=1000
        def __len__(self): return 4
        def batch(self,ix): return {k:v[ix] for k,v in raw.items()}
    cache=Cache()
    paired=PairedCache(cache,cache)
    node=next(n for n in d.c.nodes() if n['node_id']=='FUSION_D000_D000')
    model=d.new_model(node)
    rows=[]
    with (d.synchronous_restore() if synchronous else nullcontext()),d.KernelTrace(rows.append):
        stats=d.stress(model,paired)
    d.fusion.validate_offload_stats(stats,calls=3)
    assert any('convolution_backward' in row.get('operation','') for row in rows)
    assert not any(row['status']=='failed' for row in rows)
    del model
    torch.cuda.empty_cache()


def spec_fixture(tmp_path):
    root=tmp_path/'diag'
    root.mkdir()
    return d.c.artifact('CUDA_DIAGNOSTIC_SPEC',root=str(root),project_dir=str(tmp_path/'project'),
        source_commit='a'*40,source_snapshot={},policy=d.POLICY,
        execution_site=d.c.registration()['execution_site'],cpus=6,memory_mb=256000,minutes=60)


def test_exact_one_job_dry_live_and_duplicate_prevention(tmp_path,monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as exact
    from hlt_classification.luka_fullsim import regular_submission as sub
    spec=spec_fixture(tmp_path)
    monkeypatch.setattr(d,'validate',lambda *a,**kw:None)
    monkeypatch.setattr(sub,'check_site',lambda _:None)
    calls=[]
    def scheduler(args,**kw):
        calls.append(args)
        return subprocess.CompletedProcess(args,0,'123','')
    monkeypatch.setattr(exact.subprocess,'run',scheduler)
    plan=d.submit(spec)
    assert not calls and plan['jobs']==1
    argv=plan['commands'][0]['command']
    assert '--time=01:00:00' in argv and '--mem=256000M' in argv
    assert not plan['commands'][0]['dependencies']
    assert spec['policy']['science_admissible'] is False
    with pytest.raises(PermissionError):
        d.submit(spec,execute=True,plan_hash='0'*64,authorization=d.AUTHORIZATION)
    first=d.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=d.AUTHORIZATION)
    assert d.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=d.AUTHORIZATION)==first
    assert len(calls)==1


def test_ambiguous_submission_does_not_resubmit(tmp_path,monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as exact
    from hlt_classification.luka_fullsim import regular_submission as sub
    spec=spec_fixture(tmp_path)
    monkeypatch.setattr(d,'validate',lambda *a,**kw:None)
    monkeypatch.setattr(sub,'check_site',lambda _:None)
    plan=d.submit(spec)
    def lost(*a,**k): raise RuntimeError('lost reply')
    monkeypatch.setattr(exact.subprocess,'run',lost)
    with pytest.raises(RuntimeError,match='lost reply'):
        d.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=d.AUTHORIZATION)
    with pytest.raises(PermissionError,match='interrupted'):
        d.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=d.AUTHORIZATION)


def test_original_source_guard_does_not_allow_model_changes(tmp_path,monkeypatch):
    old,new=tmp_path/'old',tmp_path/'new'
    original={'src/hlt_classification/jetclass2_delphes/dzfix_fusion_model.py',
              'src/hlt_classification/luka_fullsim/alljets_fusion_source.py'}
    extra={'src/hlt_classification/luka_fullsim/alljets_fusion_diagnostic.py'}
    for root in (old,new):
        for name in original:
            path=root/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('original')
    monkeypatch.setattr(d,'_files',lambda path:original if Path(path)==old else original|extra)
    d.source_compatibility(new,dict(project_dir=str(old)))
    (new/'src/hlt_classification/jetclass2_delphes/dzfix_fusion_model.py').write_text('different')
    with pytest.raises(ValueError,match='scientific execution'):
        d.source_compatibility(new,dict(project_dir=str(old)))


def test_runtime_rejects_environment_or_blocking_drift(monkeypatch):
    monkeypatch.setattr(d.w,'execution',lambda spec:'123')
    monkeypatch.setattr(d,'installed_environment',lambda:{'torch':'old'})
    monkeypatch.setattr(d,'gpu_identity',lambda:{'name':'A100'})
    monkeypatch.delenv('TORCH_CUDNN_V8_API_DISABLED',raising=False)
    runtime=dict(environment={'torch':'old'},gpu={'name':'A100'},cudnn_v8_disabled='unset')
    monkeypatch.setenv('CUDA_LAUNCH_BLOCKING','1')
    assert d._runtime({},runtime,launch_blocking=True)['job_id']=='123'
    with pytest.raises(PermissionError,match='before starting Python'):
        d._runtime({},runtime,launch_blocking=False)
    monkeypatch.setattr(d,'installed_environment',lambda:{'torch':'new'})
    with pytest.raises(PermissionError,match='original software'):
        d._runtime({},runtime,launch_blocking=True)


@pytest.mark.parametrize('fail',[False,True])
def test_case_publication_and_poisoned_cuda_failure_no_gpu_queries(all_prepared,tmp_path,monkeypatch,fail):
    gate,parent,p=fixture_spec(all_prepared,tmp_path)
    spec=spec_fixture(tmp_path)
    monkeypatch.setattr(d,'validate',lambda *a,**kw:(gate,{}))
    monkeypatch.setattr(d,'_runtime',lambda *a,**kw:dict(job_id='123'))
    monkeypatch.setattr(d,'new_model',lambda node:object())
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    poisoned=[]
    def peak():
        assert not poisoned, 'Never query CUDA after failure'
        return 1024
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',peak)
    evidence,_,_=evidence_fixture()
    def stress(model,train):
        assert train.role=='train' and train.primary is train.context
        if fail:
            poisoned.append(True)
            raise RuntimeError('CUDA error: injected illegal memory access')
        return evidence['measurements'][0]['stress_offload']
    monkeypatch.setattr(d,'stress',stress)
    name='native_async_d000'
    directory=Path(spec['root'])/'cases'/name
    if fail:
        with pytest.raises(RuntimeError,match='illegal memory'):
            d.run_case(spec,name)
        assert not (directory/'result.json').exists()
        failure=load_json(directory/'failure.json')
        assert not failure['completed'] and not failure['science_admissible']
    else:
        report=d.run_case(spec,name)
        assert report['completed'] and not report['science_admissible']
        assert report['witness']['train_rows']==p['counts']['train']
    assert (directory/'operations.jsonl').is_file()
    assert (directory/'witness.json').is_file()
    with pytest.raises(FileExistsError):
        d.run_case(spec,name)


def test_failed_cuda_child_does_not_poison_or_skip_candidate_children(tmp_path,monkeypatch,capsys):
    spec=spec_fixture(tmp_path)
    monkeypatch.setattr(d,'validate',lambda *a,**kw:({},{}))
    monkeypatch.setattr(d,'_runtime',lambda *a,**k:None)
    names=[]
    def child(argv,**kw):
        name=argv[-1]
        names.append(name)
        case=Path(spec['root'])/'cases'/name
        case.mkdir()
        case_policy=next(row for row in d.CASES if row[0]==name)
        assert kw['env']['CUDA_LAUNCH_BLOCKING']==('1' if case_policy[3] else '0')
        if name=='native_async_d000':
            return subprocess.CompletedProcess(argv,1)
        write_immutable_json(case/'result.json',d.c.artifact('CUDA_DIAGNOSTIC_CASE',
            parents=dict(spec=spec['content_hash']),completed=True,science_admissible=False,name=name))
        return subprocess.CompletedProcess(argv,0)
    monkeypatch.setattr(d.subprocess,'run',child)
    report=d.run(spec)
    assert names==[row[0] for row in d.CASES]
    assert not report['all_cases_completed'] and report['science_admissible'] is False
    assert [r['completed'] for r in report['cases']]==[False,True,True,True]
    assert (Path(spec['root'])/'diagnostic_report.json').is_file()
    assert not (Path(spec['root'])/'preflight.json').exists()
    with pytest.raises(FileExistsError):
        d.run(spec)
    path=Path(spec['root'])/'diagnostic_spec.json'
    write_immutable_json(path,spec)
    cli=Path(__file__).resolve().parents[1]/'scripts/luka_fusion_cuda_diagnostic.py'
    monkeypatch.setattr(sys,'argv',[str(cli),'results','--spec',str(path)])
    runpy.run_path(str(cli),run_name='__main__')
    assert 'Science admission: NO.' in capsys.readouterr().out


def test_no_validation_reads_no_science_factory_change_and_source_guard():
    root=Path(__file__).resolve().parents[1]
    worker=(root/'sbatch/run_luka_fusion_cuda_diagnostic.sh').read_text()
    assert worker.index('CUDA_LAUNCH_BLOCKING=1') < worker.index('exec python')
    assert '${PROJECT_DIR}/sbatch/common.sh' in worker
    text=Path(d.__file__).read_text()
    assert "w.cache(gate, p, node, 'train'" in text
    assert "w.cache(gate, p, node, 'validation'" not in text
    names=subprocess.run(['git','diff','--name-only','e9b0b560e60b60bf4e779cd69756e1e3163a7fb6','--','src'],
        cwd=root,check=True,capture_output=True,text=True).stdout.splitlines()
    assert set(names)<={'src/hlt_classification/luka_fullsim/alljets_fusion_source.py',
                       'src/hlt_classification/luka_fullsim/alljets_fusion_diagnostic.py'}

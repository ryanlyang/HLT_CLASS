"""Fail before full decoding; keep CPU control collectives off the GPU group."""
from pathlib import Path
from datetime import timedelta
import os
import subprocess
import numpy as np
import pytest
import torch
from hlt_classification.data.cache_contracts import load_json,write_immutable_json,with_content_hash
from hlt_classification.operations.hydra_reconstruction_gpu_diagnostic import context,SyntheticCache
from hlt_classification.luka_balanced_reconstruction_hydra import (
    contracts as c,launch,startup,worker,kernel,campaign,population,inputs,resources,artifacts,transport)
from hlt_classification.luka_subclass_ddp_hydra.distributed import Context


def test_transport_policy_records_operator_evidence_without_changing_science():
    from hlt_classification.data.cache_contracts import sha256_file,validate_content_hash
    assert transport.policy()['environment']=={'NCCL_P2P_DISABLE':'1'}
    evidence=transport.policy()['evidence'];assert evidence['passing_job']=='219052'
    helper=Path(__file__).resolve().parents[1]/'src/hlt_classification/operations/hydra_reconstruction_gpu_diagnostic.py'
    assert sha256_file(helper)==evidence['helper_sha256']
    assert evidence['scope']=='known_tensor_collectives_and_OFFLINE_s10_synthetic_four_rank_parity'
    validate_content_hash(launch.policy(),expected_contract=c.PREFIX+'_WORKER_LAUNCH_POLICY/v3',expected_schema_version=3)
    validate_content_hash(startup.policy(),expected_contract=c.PREFIX+'_EARLY_GPU_POLICY/v2',expected_schema_version=2)
    assert launch.policy()['gpu_transport']==startup.policy()['gpu_transport']==transport.policy()
    assert c.recipe()['normalization']=='SyncBatchNorm_global128'


@pytest.mark.parametrize('value',[None,'0','true'])
def test_unqualified_transport_stops_rank_before_cuda_initialization(monkeypatch,value):
    for key,setting in dict(RANK='0',LOCAL_RANK='0',WORLD_SIZE='4').items():monkeypatch.setenv(key,setting)
    if value is None:monkeypatch.delenv('NCCL_P2P_DISABLE',raising=False)
    else:monkeypatch.setenv('NCCL_P2P_DISABLE',value)
    monkeypatch.setattr(torch.cuda,'set_device',lambda *a:pytest.fail('CUDA initialized before transport check'))
    with pytest.raises(PermissionError,match='NCCL_P2P_DISABLE=1'):
        worker.rank({},'qualify','/dev/shm/unused',startup_only=True)


def test_shell_pins_no_p2p_before_first_python_even_if_inherited_value_is_zero(tmp_path):
    bash=Path('C:/Program Files/Git/bin/bash.exe') if os.name=='nt' else Path('/bin/bash')
    if not bash.is_file():pytest.skip('Bash required')
    (tmp_path/'bin').mkdir();(tmp_path/'env/bin').mkdir(parents=True)
    python=tmp_path/'env/bin/python'
    python.write_bytes(b'#!/bin/sh\ntest "$NCCL_P2P_DISABLE" = 1 || exit 97\ncat >/dev/null\nprintf "QUALIFIED_TRANSPORT=%s\\n" "$NCCL_P2P_DISABLE"\n')
    timeout=tmp_path/'bin/timeout';timeout.write_bytes(b'#!/bin/sh\nshift 3\nexec "$@"\n')
    python.chmod(0o755);timeout.chmod(0o755)
    root=Path(__file__).resolve().parents[1]
    command=('set -euo pipefail; task_root="$1"; '
        'if command -v cygpath >/dev/null 2>&1; then task_root="$(cygpath -u "$task_root")"; fi; '
        'export PROJECT_DIR="$task_root" ENV_DIR="$task_root/env"; export PATH="$task_root/bin:$PATH"; '
        'source "$2"; test "$NCCL_P2P_DISABLE" = 1')
    result=subprocess.run([str(bash),'-c',command,'test',tmp_path.as_posix(),
        (root/'sbatch/luka_balanced_reconstruction_hydra_common.sh').as_posix()],
        capture_output=True,text=True,timeout=20,env=dict(os.environ,NCCL_P2P_DISABLE='0'))
    assert result.returncode==0,result.stdout+result.stderr
    assert 'QUALIFIED_TRANSPORT=1' in result.stdout


def test_startup_step_uses_actual_parser_four_ranks_and_five_minute_slurm_boundary(tmp_path):
    argv=launch.startup_command(tmp_path/'project with spaces',tmp_path/'study','qualify',tmp_path/'node')
    assert argv[0]=='srun' and '--time=00:05:00' in argv and '--kill-on-bad-exit=1' in argv
    assert '--ntasks=1' in argv and '--gres=gpu:geforce_gtx_1080_ti:4' in argv and '--export=ALL' in argv
    assert launch.validate(argv)['training_script_args'][-1]=='--startup-only'
    assert launch.policy()['early_gpu_policy']==startup.policy()


def test_rank_startup_uses_separate_groups_and_returns_before_loading_views(tmp_path,monkeypatch):
    monkeypatch.setenv('NCCL_P2P_DISABLE','1')
    spec=dict(root=str(tmp_path),content_hash='a'*64);startup.folder(spec,'123').mkdir(parents=True)
    for key,value in dict(RANK='0',LOCAL_RANK='0',WORLD_SIZE='4').items():monkeypatch.setenv(key,value)
    monkeypatch.setattr(torch.cuda,'set_device',lambda *a:None)
    monkeypatch.setattr(kernel,'configure',lambda *a:None);monkeypatch.setattr(campaign,'check',lambda *a:None)
    monkeypatch.setattr(worker,'allocation',lambda *a:'123')
    calls=[];control=object()
    monkeypatch.setattr(worker.dist,'init_process_group',lambda backend,**kw:calls.append((backend,kw['timeout'])))
    def group(**kw):calls.append((kw['backend'],kw['timeout']));return control
    monkeypatch.setattr(worker.dist,'new_group',group)
    monkeypatch.setattr(worker.dist,'is_initialized',lambda:True)
    monkeypatch.setattr(worker.dist,'destroy_process_group',lambda:calls.append('destroy'))
    monkeypatch.setattr(inputs,'load_node',lambda *a:pytest.fail('Startup must not load particle views'))
    monkeypatch.setattr(population,'load',lambda *a:pytest.fail('Startup must not replay populations'))
    def probe(spec,task,ctx,job):
        assert ctx.world==4 and ctx.rank==0 and job=='123' and task=='qualify'
        return dict(probed=True)
    monkeypatch.setattr(startup,'run',probe)
    assert worker.rank(spec,'qualify',str(Path('/dev/shm')/'rlyang_lukabrec_123'),startup_only=True)==dict(probed=True)
    assert calls==[('nccl',timedelta(seconds=45)),('gloo',timedelta(seconds=45)),'destroy']


@pytest.mark.parametrize('failure',['process_exit','missing_receipt','bad_receipt'])
def test_early_gpu_failure_precedes_every_population_or_particle_read(tmp_path,monkeypatch,failure):
    spec=dict(root=str(tmp_path),project='/project',dataset='/dataset',content_hash='a'*64)
    node=tmp_path/'node';node.mkdir();cleaned=[];calls=[]
    monkeypatch.setattr(kernel,'configure',lambda *a:None);monkeypatch.setattr(campaign,'check',lambda *a:None)
    monkeypatch.setattr(worker,'allocation',lambda *a:'123');monkeypatch.setattr(worker,'task_receipt',lambda *a:{})
    monkeypatch.setattr(worker,'scratch',lambda *a:node);monkeypatch.setattr(worker,'cleanup',lambda *a:cleaned.append(1))
    monkeypatch.setattr(population,'load',lambda *a:pytest.fail('Population replay started before GPU acceptance'))
    monkeypatch.setattr(inputs,'prepare_node',lambda *a,**kw:pytest.fail('Particle decoding started before GPU acceptance'))
    def execute(argv,**kwargs):
        calls.append(argv)
        if failure=='process_exit':raise subprocess.CalledProcessError(124,argv)
        if failure=='bad_receipt':write_immutable_json(startup.folder(spec,'123')/'complete.json',c.artifact('EARLY_GPU'))
    monkeypatch.setattr(worker.subprocess,'run',execute)
    with pytest.raises((subprocess.CalledProcessError,FileNotFoundError,ValueError,KeyError)):
        worker.run(spec,'qualify')
    assert len(calls)==1 and calls[0][0]=='srun' and cleaned==[1]
    assert not (tmp_path/'qualification.json').exists()


@pytest.mark.parametrize('task',['qualify','R24_s11','OFFLINE_s10'])
def test_early_driver_uses_all_required_native_paths_and_authenticates_receipts(tmp_path,monkeypatch,task):
    monkeypatch.setenv('NCCL_P2P_DISABLE','1')
    from hlt_classification.luka_sixclass_balanced_oscar import model as native
    from hlt_classification.luka_next_supervision_hydra.reconstruction_batched import pack
    spec=dict(root=str(tmp_path),content_hash='a'*64,maximum=316);job='123'
    startup.folder(spec,job).mkdir(parents=True);calls=[]
    class Ctx:
        rank=0;world=4;device=torch.device('cpu')
        def sync(self):pass
        def main(self,fn):return fn()
        def all(self,value):return [value]*4
    monkeypatch.setattr(startup.dist,'all_reduce',lambda t:t.fill_(10))
    monkeypatch.setattr(startup.dist,'broadcast',lambda t,src:None)
    monkeypatch.setattr(native,'installed_parity',lambda *a,**kw:dict(content_hash='b'*64,passed=True))
    def parity(spec,task,cache,targets,bank,ctx,position=8.):
        assert isinstance(cache,SyntheticCache) and len(cache)==16 and bank is None
        raw=cache.batch(np.arange(16));assert raw['features'].shape==(16,17,8)
        assert (targets is not None)==c.cell(task)['reconstruction']
        if targets is not None:
            _,counts=pack(targets.batch(np.arange(16)),316,'cpu')
            assert counts.tolist()==[1,2,3,4]*4
        calls.append((task,position))
        return dict(position=position,ranks=[dict(passed=True)]*4,tolerances=resources.TOLERANCES)
    monkeypatch.setattr(resources,'parity',parity)
    monkeypatch.setattr(population,'load',lambda *a:pytest.fail('No population access'))
    monkeypatch.setattr(inputs,'prepare_node',lambda *a,**kw:pytest.fail('No particle decoding'))
    result=startup.run(spec,task,Ctx(),job)
    assert startup.check(spec,task,job)==result
    expected=[(t,p) for t in startup.probe_tasks(task) for p in ((8.,2.) if c.cell(t)['reconstruction'] else (8.,))]
    assert calls==expected and result['synthetic_only'] and result['discarded']
    with pytest.raises(ValueError,match='receipt differs'):startup.check(spec,'M0HLT_s11',job)
    if task=='qualify':
        path=startup.folder(spec,job)/'OFFLINE_s10.json';row=load_json(path)
        row['proof']['ranks'][2]['passed']=False;write_immutable_json(tmp_path/'changed.json',with_content_hash(row))
        path.write_bytes((tmp_path/'changed.json').read_bytes())
        with pytest.raises(ValueError):startup.check(spec,task,job)


def _host_control_worker(rank,init):
    import torch.distributed as dist
    from datetime import timedelta
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=init,rank=rank,world_size=4,timeout=timedelta(seconds=30))
    control=dist.new_group(backend='gloo',timeout=timedelta(seconds=30))
    ctx=context(Context,control,dist)(rank,4,torch.device('cpu'))
    try:
        for i in range(8):
            assert ctx.main(lambda:dict(epoch=i))==dict(epoch=i)
            t=torch.tensor(float(rank+1));dist.all_reduce(t);assert t.item()==10.
            assert ctx.all(rank)==[0,1,2,3]
        def failed():raise OSError('root callback failed')
        with pytest.raises(RuntimeError,match='root callback failed'):ctx.main(failed)
        ctx.barrier()
    finally:dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(),reason='Actual Gloo required')
def test_actual_four_process_control_group_and_tensor_group_ordering(tmp_path):
    torch.multiprocessing.spawn(_host_control_worker,args=((tmp_path/'control_store').as_uri(),),nprocs=4,join=True)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual CUDA/Weaver required')
def test_actual_weaver_and_auxiliary_paths_with_new_synthetic_inputs(tmp_path,monkeypatch):
    from hlt_classification.luka_sixclass_balanced_oscar.model import installed_parity
    previous=torch.get_num_threads();torch.set_num_threads(1);kernel.configure(1,12)
    try:
        spec=dict(root=str(tmp_path),content_hash='a'*64,population=dict(content_hash='b'*64),maximum=16)
        artifacts.prepare_starts(spec);cache=SyntheticCache()
        assert installed_parity(cache,device='cuda')['passed']
        class Targets:
            def batch(self,ix):return [startup.target(i,spec['maximum']) for i in ix]
        ctx=Context(0,1,torch.device('cuda'))
        for task in startup.probe_tasks('qualify'):
            target=Targets() if c.cell(task)['reconstruction'] else None
            for position in ((8.,2.) if target else (8.,)):
                assert resources.parity(spec,task,cache,target,None,ctx,position)['ranks'][0]['passed']
    finally:torch.set_num_threads(previous)

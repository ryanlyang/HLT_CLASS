"""Bounded GPU diagnosis submission and host-control transport; no real Slurm."""
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
from types import SimpleNamespace
import pytest
from hlt_classification.operations import hydra_reconstruction_gpu_diagnostic as op


@pytest.fixture
def setup(tmp_path,monkeypatch):
    base=tmp_path/'workspace';base.mkdir()
    monkeypatch.setattr(op,'BASE',base)
    spec_path=base/'runs/balanced_reco_600k_a2b634c7_r1/campaign_spec.json'
    spec_path.parent.mkdir(parents=True);monkeypatch.setattr(op,'SPEC',spec_path)
    body=dict(contract='LUKA_BALANCED_RECONSTRUCTION_HYDRA_SPEC/v1',schema_version=1,
        root=str(spec_path.parent),commit=op.COMMIT,project=str(base/'code_balanced_reco_a2b634c7'),
        environment_dir=str(base/'environment_pair_hydra_v1/venv'),population=dict(counts=dict(train=600000,validation=150000)))
    import hashlib
    body['content_hash']=hashlib.sha256(op.canonical(body)).hexdigest();spec_path.write_text(json.dumps(body))
    helper=base/'helper.py';helper.write_text('technical helper')
    captured=[];counter=[]
    def query(args,**kwargs):
        if args[0]=='git':return op.COMMIT if args[-1]=='HEAD' else ''
        return 'ClusterName = hydra\n'
    monkeypatch.setattr(op.subprocess,'check_output',query)
    def run(args,**kwargs):
        captured.append((args,kwargs))
        if '--test-only' in args:return SimpleNamespace(returncode=0)
        counter.append(1);return SimpleNamespace(returncode=0,stdout=str(100+len(counter)),stderr='')
    monkeypatch.setattr(op.subprocess,'run',run)
    return base,helper,body,captured


def launch(setup):
    op.submit(setup[1],'a'*64,'gpu1905',op.AUTH)
    return setup[0]/'reports'/('reconstruction_gpu_diagnostic_'+'a'*12)


def test_only_three_short_diagnostic_jobs_with_afterany_ordering(setup):
    output=launch(setup);calls=setup[-1]
    assert len(calls)==6 and sum('--test-only' in a for a,_ in calls)==3
    actual=[a for a,_ in calls if '--test-only' not in a]
    for args in actual:
        assert '--time=00:05:00' in args and '--nodelist=gpu1905' in args
        assert '--gres=gpu:geforce_gtx_1080_ti:4' in args and '--cpus-per-task=12' in args
        assert '--mem=64000M' in args and '--export=NONE' in args
        assert not any('prepare' in a or 'train_' in a or 'release' in a for a in args)
    assert '--dependency=afterany:101' in actual[1] and '--dependency=afterany:102' in actual[2]
    assert all((output/(p+'.submitted.json')).is_file() for p in op.PROFILES)


def test_profiles_change_only_explicit_debug_transport_and_preserve_quoting(setup):
    _,helper,value,_=setup;helper=helper.parent/'helper with spaces.py'
    rows=op.commands(value,helper,helper.parent/'output with spaces','b'*64,'gpu1905')
    for row in rows:
        wrapped=next(a.split('=',1)[1] for a in row['command'] if a.startswith('--wrap='))
        tokens=shlex.split(wrapped)
        assert str(helper) in tokens and 'supervise' in tokens
        assert ('NCCL_P2P_DISABLE=1' in tokens)==(row['profile']=='gloo_no_p2p')
        assert 'CUBLAS_WORKSPACE_CONFIG=:4096:8' in tokens
        assert 'PYTHONNOUSERSITE=1' in tokens


def test_failed_dry_review_submits_nothing_and_creates_no_claim(setup,monkeypatch):
    calls=[]
    def fail(args,**kwargs):
        calls.append(args);raise RuntimeError('scheduler dry review failed')
    monkeypatch.setattr(op.subprocess,'run',fail)
    with pytest.raises(RuntimeError,match='dry review'):launch(setup)
    assert len(calls)==1 and '--test-only' in calls[0]
    assert not (setup[0]/'reports').exists()


def test_ambiguous_submission_preserves_intent_and_refuses_repeat(setup,monkeypatch):
    original=op.subprocess.run
    def uncertain(args,**kwargs):
        if '--test-only' in args:return original(args,**kwargs)
        return SimpleNamespace(returncode=1,stdout='',stderr='connection lost')
    monkeypatch.setattr(op.subprocess,'run',uncertain)
    with pytest.raises(ValueError,match='Uncertain submission'):launch(setup)
    output=setup[0]/'reports'/('reconstruction_gpu_diagnostic_'+'a'*12)
    assert (output/'original.intent.json').exists() and not (output/'original.submitted.json').exists()
    with pytest.raises(ValueError,match='already exists'):launch(setup)


def test_explicit_authentication_and_failed_source_guard(setup):
    with pytest.raises(ValueError,match='authorization'):op.submit(setup[1],'a'*64,'gpu1905','WRONG')
    p=op.SPEC;value=json.loads(p.read_text());value['commit']='0'*40;p.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='checksum'):launch(setup)
    assert setup[-1]==[]


def test_successful_submission_does_not_touch_original_spec_or_source(setup):
    before=op.SPEC.read_bytes();launch(setup)
    assert op.SPEC.read_bytes()==before
    assert not (setup[0]/'code_balanced_reco_a2b634c7').exists()


def test_saved_job_binding_checks_profile_parent_source_and_short_allocation(setup,monkeypatch):
    output=launch(setup);monkeypatch.setenv('SLURM_JOB_ID','101')
    fields='JobState=RUNNING Comment='+('a'*64)+':original Partition=gpus NumNodes=1 TimeLimit=00:05:00 '
    fields+='NodeList=gpu1905 CPUs/Task=12 NumTasks=1 AllocTRES=cpu=12,mem=64000M,gres/gpu=4,gres/gpu:geforce_gtx_1080_ti=4'
    monkeypatch.setattr(op.subprocess,'check_output',lambda *a,**kw:fields)
    assert op.bound(setup[2],output,'original','a'*64)=='101'
    monkeypatch.setenv('SLURM_JOB_ID','999')
    with pytest.raises(ValueError,match='Saved diagnostic job'):op.bound(setup[2],output,'original','a'*64)


def test_host_control_group_is_used_without_changing_data_methods():
    @dataclass
    class Parent:
        rank:int;world:int;device:object
        def average(self,tensor):return ('original_data_reduce',tensor)
    calls=[];marker=object()
    class Dist:
        def broadcast_object_list(self,box,src,group):calls.append(('broadcast',group))
        def all_gather_object(self,values,value,group):
            calls.append(('gather',group));values[:]=[value]*4
        def barrier(self,group):calls.append(('barrier',group))
    ctx=op.context(Parent,marker,Dist())(0,4,'device')
    assert ctx.main(lambda:dict(ok=True))==dict(ok=True)
    assert ctx.all(17)==[17]*4
    ctx.barrier();assert calls==[('broadcast',marker),('barrier',marker),('gather',marker),('barrier',marker)]
    assert ctx.average(5)==('original_data_reduce',5)


def test_host_control_propagates_rank_zero_errors():
    @dataclass
    class Parent:
        rank:int;world:int;device:object
    ctx=op.context(Parent,None,None)(0,1,'cpu')
    def failed():raise OSError('technical failure')
    with pytest.raises(RuntimeError,match='technical failure'):ctx.main(failed)


def test_no_gpu_probe_claim_is_made_by_missing_rank_records(setup,monkeypatch):
    output=launch(setup)
    monkeypatch.setattr(op,'bound',lambda *a:'101')
    monkeypatch.setattr(op.subprocess,'run',lambda args,**kwargs:SimpleNamespace(returncode=0))
    with pytest.raises(FileNotFoundError):op.supervise(setup[2],output,'original','a'*64,setup[1])
    assert not (output/'original/complete.json').exists()


def test_failed_torchrun_preserves_failure_and_never_publishes_success(setup,monkeypatch):
    output=launch(setup)
    monkeypatch.setattr(op,'bound',lambda *a:'101')
    observed=[]
    def failed(args,**kwargs):observed.append(args);return SimpleNamespace(returncode=1)
    monkeypatch.setattr(op.subprocess,'run',failed)
    with pytest.raises(RuntimeError,match='Diagnostic failed'):
        op.supervise(setup[2],output,'original','a'*64,setup[1])
    assert (output/'original/failed.json').exists() and not (output/'original/complete.json').exists()
    assert observed[0][observed[0].index('--')+1]==str(setup[1])
    assert '--max-restarts=0' in observed[0]
    from torch.distributed.run import get_args_parser
    argv=observed[0];parsed=get_args_parser().parse_args(argv[argv.index('torch.distributed.run')+1:])
    assert parsed.training_script==str(setup[1]) and parsed.training_script_args[0]=='rank'


@pytest.mark.parametrize('profile',op.PROFILES)
def test_rank_driver_publishes_synthetic_probe_only_with_mocked_cuda(setup,monkeypatch,profile):
    import numpy as np
    import torch
    import torch.distributed as dist
    from hlt_classification.luka_balanced_reconstruction_hydra import campaign,kernel,resources,population,inputs,training
    output=launch(setup);groups=[]
    monkeypatch.setenv('RANK','0');monkeypatch.setenv('WORLD_SIZE','4');monkeypatch.setenv('SLURM_JOB_ID','101')
    monkeypatch.setattr(campaign,'check',lambda value:None);monkeypatch.setattr(kernel,'configure',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'set_device',lambda rank:None)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:4)
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda rank:(6,1))
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda rank:'NVIDIA GeForce GTX 1080 Ti')
    monkeypatch.setattr(torch.cuda.nccl,'version',lambda:(2,21,5))
    monkeypatch.setattr(op,'device_for_rank',lambda *a:torch.device('cpu'))
    def initialized(backend,**kwargs):assert backend=='nccl' and kwargs['timeout'].total_seconds()==45
    monkeypatch.setattr(dist,'init_process_group',initialized)
    marker=object();monkeypatch.setattr(dist,'new_group',lambda **kwargs:marker)
    monkeypatch.setattr(dist,'all_reduce',lambda tensor:tensor.fill_(10))
    monkeypatch.setattr(dist,'broadcast',lambda tensor,**kwargs:tensor.fill_(42))
    monkeypatch.setattr(dist,'broadcast_object_list',lambda box,**kwargs:groups.append(kwargs.get('group')))
    def gather(values,value,**kwargs):values[:]=[dict(rank=r) for r in range(4)]
    monkeypatch.setattr(dist,'all_gather_object',gather);monkeypatch.setattr(dist,'barrier',lambda **kwargs:None)
    monkeypatch.setattr(dist,'is_initialized',lambda:True);monkeypatch.setattr(dist,'destroy_process_group',lambda:None)
    monkeypatch.setattr(population,'load',lambda *a:pytest.fail('No population/data reads'))
    monkeypatch.setattr(inputs,'prepare_node',lambda *a,**kwargs:pytest.fail('No input decoding'))
    monkeypatch.setattr(training,'fit',lambda *a,**kwargs:pytest.fail('No scientific training'))
    def parity(value,task,cache,targets,bank,ctx):
        raw=cache.batch(np.arange(16))
        assert task=='OFFLINE_s10' and targets is None and bank is None
        assert raw['features'].shape==(16,17,8) and raw['vectors'].shape==(16,4,8)
        assert np.isfinite(raw['features']).all() and set(raw['labels'])==set(range(6))
        return dict(ranks=[dict(passed=True)]*4)
    monkeypatch.setattr(resources,'parity',parity)
    op.rank(setup[2],output,profile,'a'*64)
    result=op.read(output/profile/'rank_0.json')
    assert result['passed'] and result['raw_particle_reads'] is False and result['scientific_training'] is False
    assert groups==[None if profile=='original' else marker]

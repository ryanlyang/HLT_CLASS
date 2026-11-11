"""Two-rank production semantics, native replay, exact release and fresh controls."""
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import copy
import numpy as np
import pytest
import torch
from torch import nn
import torch.distributed as dist
from test_luka_subclass_hydra import spec as old_spec, banks as old_banks, small, threads, reseal
from test_luka_u100_pretrain import cache
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.luka_subclass_hydra import contracts as old_contracts
from hlt_classification.luka_subclass_hydra.benchmark_training import parity_snapshot, compare
from hlt_classification.luka_subclass_ddp_hydra import (
    campaign, contracts as c, distributed as d, resources, training, model, inputs,
    targets, submission, worker, reporting, execution)


def spec(root):
    s=old_spec(root);s.update(campaign.registration());s['counts']=dict(train=22,validation=22);s['original50k_rows']=22
    s['nodes']={t:dict(initialization_seed=23,sampler_seed=29,node_id=t) for t in c.FITS}
    return s


def banks(s,tr=None):
    bb=old_banks(s,tr)
    bb['subclass'].manifest=c.artifact('SUBCLASS_BANK',parents=targets.parents(s),role='train',coordinate='OFFLINE')
    return bb


def devices():
    return [dict(name='GeForce GTX 1080 Ti',compute_capability=[6,1],total_memory_bytes=11*2**30) for _ in range(2)]


def diagnostic(s):
    return c.artifact('RESOURCES',parents=dict(spec=s['content_hash']),status='complete',devices=devices(),
        environment=s['reference']['environment'],profiles={t:dict(real_targets=True,miniature_complete=True,
        deployment_reload=True,largest_support_backward=True,partial_batch_backward=True,
        train_rows=s['counts']['train'],validation_rows=s['counts']['validation'],step_samples=[.2]*12,
        step_seconds=float(np.mean([.2]*12)),validation_seconds=20.,load_seconds=120.,probe_seconds=1.,startup_seconds=30.,
        memory=dict(host_peak_bytes=12*2**30,ranks=[dict(peak_rss_bytes=5*2**30,gpu_peak_bytes=7*2**30) for _ in range(2)])) for t in c.FITS})


def test_new_execution_preserves_target_recipe_and_has_fresh_control():
    assert c.target_recipe()==old_contracts.target_recipe()
    assert c.target_recipe()['content_hash']=='2794d55556ae6d031f1347c00437df918e4261bea5f0c10016d38af3c980e60c'
    assert len(c.FITS)==11 and c.FITS[0]=='C0'
    assert c.WEIGHTS==dict(C0=0.,**old_contracts.WEIGHTS)
    assert c.recipe()['precision']=='fp32' and c.recipe()['normalization']=='SyncBatchNorm'
    with pytest.raises(ValueError):c.validate(old_contracts.artifact('SPEC'),'SPEC')
    from hlt_classification.luka_subclass_hydra import benchmark
    with pytest.raises(ValueError):c.validate(benchmark.artifact('REPORT'),'ACCEPTANCE')
    for p in ('a6000','l40','1080ti','auto'):
        with pytest.raises(ValueError):campaign.registration(p)


def test_dag_has_two_bootstrap_eleven_independent_fits_and_one_report(tmp_path):
    s=spec(tmp_path);boot=submission.plan(s,'bootstrap');envelope=submission.plan(s,'envelope')
    assert len(boot['commands'])==2 and len(envelope['commands'])==12
    assert boot['commands'][1]['dependencies']==['setup']
    assert '--dependency=afterok:${JOB_setup}' in boot['commands'][1]['command']
    assert boot['upper_bound_commands']==envelope['commands']
    for row in envelope['commands'][:-1]:
        assert row['dependencies']==[]
        assert '--gres=gpu:geforce_gtx_1080_ti:2' in row['command']
        assert '--mem=48000M' in row['command'] and '--ntasks=1' in row['command']
    assert envelope['commands'][0]['task_id']=='C0'
    assert envelope['commands'][-1]['dependencies']==list(c.FITS)
    assert any(x.startswith('--dependency=afterany:') for x in envelope['commands'][-1]['command'])
    measured=submission.plan(s,'science',resources.accept(s,diagnostic(s)))
    assert submission.within_envelope(envelope,measured)
    measured['commands'][0]['command'].append('--gres=gpu:3')
    with pytest.raises(PermissionError):submission.within_envelope(envelope,measured)


@pytest.mark.parametrize('bad',('host','gpu','missing','time','negative','samples','targets','devices','nan'))
def test_resources_fail_closed_without_quality_gates(tmp_path,bad):
    s=spec(tmp_path);value=diagnostic(s);assert set(resources.accept(s,value)['minutes'])==set(c.FITS)
    p=value['profiles']['C0']
    if bad=='host':p['memory']['host_peak_bytes']=48*2**30
    elif bad=='gpu':p['memory']['ranks'][1]['gpu_peak_bytes']=11*2**30
    elif bad=='missing':del value['profiles']['S100']
    elif bad=='time':p.update(step_samples=[1e6]*12,step_seconds=1e6)
    elif bad=='negative':p['load_seconds']=-1
    elif bad=='samples':p['step_samples']=[.2]
    elif bad=='targets':p['real_targets']=False
    elif bad=='devices':value['devices'][1]['name']='NVIDIA TITAN RTX'
    else:p['memory']['ranks'][1]['peak_rss_bytes']=float('nan')
    with pytest.raises((ValueError,MemoryError,RuntimeError)):resources.accept(s,reseal(value))


@pytest.mark.parametrize('task',('C0','S0p5','S100'))
def test_probe_does_not_change_rng_buffers_or_optimizer_update(tmp_path,monkeypatch,task):
    small(monkeypatch);s=spec(tmp_path);tr=cache();bb=banks(s,tr);ctx=d.Context(0,1,torch.device('cpu'))
    outcomes=[]
    for diagnose in (False,True):
        m,w,o=d.build(s,task,ctx)
        loss,probe,_=d.step(m,w,o,task,tr.batch(np.arange(15)),bb,np.arange(15),1.,ctx,diagnose=diagnose)
        outcomes.append((parity_snapshot(m,o,loss['total']),torch.get_rng_state()))
        if diagnose:assert probe['ce_norm']>0 and probe['weighted_subclass_norm']>=0
    result=compare(outcomes[0][0],outcomes[1][0],dict(loss_abs=0,gradient_relative_l2=0,buffer_abs=0))
    assert result['passed'] and result['adamw_moments_relative_l2']==0 and result['parameters_relative_l2']==0
    assert torch.equal(outcomes[0][1],outcomes[1][1])


@pytest.mark.parametrize('device',('cpu','cuda'))
def test_control_is_original_fp32_u100_update(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    s=spec(tmp_path);tr=cache();bb=banks(s,tr);ix=np.arange(16)
    assert resources.zero_control(s,tr.batch(ix),bb,ix,torch.device(device))==dict(model_update_equal=True,adamw_equal=True)


@pytest.mark.parametrize('task',('C0','S100'))
def test_real_native_miniature_reload_and_lineage(tmp_path,monkeypatch,task):
    small(monkeypatch);s=spec(tmp_path);tr=cache();va=cache('validation');bb=banks(s,tr)
    ctx=d.Context(0,1,torch.device('cpu'))
    r=training.fit(tmp_path/task,s,task,tr,va,bb,ctx,technical=True)
    assert r['native_reload'] and len(r['history'])==2
    state=torch.load(tmp_path/task/'selected.pt',weights_only=True)
    assert all(k.startswith('mod.') for k in state)
    assert training.validate_fit(tmp_path/task,s,task,technical=True)[0]==r
    bad=copy.deepcopy(bb);bad['u100'].manifest['role']='validation'
    with pytest.raises(PermissionError):training.check_inputs(s,task,tr,va,bad,True)
    with pytest.raises(PermissionError):training.fit(tmp_path/'bad',s,task,tr,va,bb,ctx)
    p=tmp_path/task/'selected.pt';p.write_bytes(p.read_bytes()+b'corrupt')
    with pytest.raises(ValueError):training.validate_fit(tmp_path/task,s,task,technical=True)


@pytest.mark.parametrize('bad',('duplicate','missing','nonfinite','normalization'))
def test_predictions_are_reassembled_once_in_global_order(bad):
    ix=np.arange(17,dtype=np.int64);p=np.full((17,11),1/11,np.float32)
    shards=[(ix[1::2],p[1::2]),(ix[::2],p[::2])]
    np.testing.assert_array_equal(d.merge_predictions(shards,17),p)
    if bad=='duplicate':shards[0][0][0]=0
    elif bad=='missing':shards=shards[:1]
    elif bad=='nonfinite':shards[0][1][0,0]=np.nan
    else:shards[0][1][0,0]=10
    with pytest.raises(ValueError):d.merge_predictions(shards,17)


class TinyNative(nn.Module):
    def __init__(self):
        super().__init__();self.mod=nn.Module();self.mod.embed=nn.Linear(17,128);self.mod.fc=nn.Linear(128,11)
    def embedding(self,features,mask):
        return self.mod.embed((features*mask).sum(2)/mask.sum(2).clamp_min(1)).tanh()
    def forward(self,features,vectors,mask):return self.mod.fc(self.embedding(features,mask))
    def no_weight_decay(self):return set()


class TinyStudent(TinyNative):
    def __init__(self):
        super().__init__();self.predictor=nn.Linear(128,128);self.subclass=nn.Linear(128,60);self.capture=False;self.clear()
    def clear(self):self.prediction=None;self.subclass_logits=None
    def forward(self,features,vectors,mask):
        h=self.embedding(features,mask)
        if self.capture:self.prediction=self.predictor(h);self.subclass_logits=self.subclass(h)
        return self.mod.fc(h)
    def native_state(self):return {k:v for k,v in self.state_dict().items() if k.startswith('mod.')}


def full_fixture(root):
    from test_luka_attention import cache as make_cache
    s=spec(Path(root));tr=make_cache('train','D000',267);va=make_cache('validation','D000',129)
    va.identities[:,-1]=1
    for block in va.blocks:block.identities[:,-1]=1
    s['counts']=dict(train=len(tr),validation=len(va))
    return s,tr,va


def _two_cpu_fit(rank,init_method,root):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=init_method,rank=rank,world_size=2,timeout=timedelta(seconds=100))
    try:
        model.Student=TinyStudent;training.Native=TinyNative
        s,tr,va=full_fixture(root);bb=banks(s,tr);ctx=d.Context(rank,2,torch.device('cpu'))
        assert resources.probe_neutrality(s,tr,bb,ctx)['passed']
        # Compare the production update (including its diagnostic) against a
        # global mean reference on an odd batch, then exercise all30 epochs.
        def reference():
            m,w,o=d.build(s,'S100',d.Context(0,1,ctx.device),parity=True)
            values,_,_=d.step(m,w,o,'S100',tr.batch(np.arange(15)),bb,np.arange(15),3.,d.Context(0,1,ctx.device))
            return parity_snapshot(m,o,values['total'])
        expected=ctx.main(reference)
        m,w,o=d.build(s,'S100',ctx,parity=True)
        values,diag,_=d.step(m,w,o,'S100',tr.batch(np.arange(15)),bb,np.arange(15),3.,ctx,diagnose=True)
        observed=ctx.main(lambda:parity_snapshot(m,o,values['total']))
        assert compare(expected,observed,dict(loss_abs=2e-5,gradient_relative_l2=2e-5,buffer_abs=0))['passed']
        assert diag['weighted_subclass_norm']>0
        del m,w,o
        for task in ('C0','S100'):
            r=training.fit(Path(root)/task,s,task,tr,va,bb,ctx)
            assert r['world_size']==2 and r['epochs']==30
            counts=[r['history'][e-1]['gradient_diagnostic'] is not None for e in range(1,31)]
            assert [e+1 for e,x in enumerate(counts) if x]==[1,3,10,20,30]
    finally:dist.destroy_process_group()


def test_two_process_full30_and_uneven_gradient_mean(tmp_path):
    # Real CPU collectives test orchestration/weighting; CUDA SyncBN is checked
    # separately on actual GPUs and by every Brown setup job.
    if not dist.is_gloo_available():pytest.skip('Gloo unavailable')
    banks(spec(tmp_path)) # Publish shared fixture before either rank loads it.
    torch.multiprocessing.spawn(_two_cpu_fit,args=((tmp_path/'store').as_uri(),str(tmp_path)),nprocs=2,join=True)
    for task in ('C0','S100'):
        report,p=training.validate_fit(tmp_path/task,full_fixture(tmp_path)[0],task)
        assert len(report['history'])==30 and len(p['identities'])==129


def fake_scheduler(monkeypatch):
    calls=[];reviews=[]
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(submission,'scheduler_review',lambda p:reviews.append(p))
    import hlt_classification.scouting.hcwdl_exact_dag_submission as dag
    def run(command,**kwargs):
        calls.append(command);return SimpleNamespace(stdout=str(9000+len(calls))+'\n',returncode=0)
    monkeypatch.setattr(dag.subprocess,'run',run)
    return calls,reviews


def test_auto_release_exact_ids_and_no_duplicate_campaign(tmp_path,monkeypatch):
    s=spec(tmp_path);(tmp_path/'setup').mkdir();a=resources.accept(s,diagnostic(s))
    write_immutable_json(tmp_path/'setup/acceptance.json',a)
    calls,reviews=fake_scheduler(monkeypatch);submission.launch(s)
    p=submission.plan(s,'bootstrap')
    with pytest.raises(PermissionError):submission.launch(s,execute=True,plan_hash='bad',authorization=c.AUTHORIZATION)
    b=submission.launch(s,execute=True,plan_hash=p['content_hash'],authorization=c.AUTHORIZATION)
    assert b['jobs']==dict(setup='9001',release='9002')
    monkeypatch.setattr(submission,'accounting',lambda j:['COMPLETED','0:0'])
    monkeypatch.setattr(worker,'completed_setup',lambda s:a)
    result=submission.finish(s,'9002');assert len(result['jobs'])==12 and len(calls)==14
    assert submission.finish(s,'9002')==result and len(calls)==14
    assert [p['stage'] for p in reviews]==['bootstrap','envelope','science']


def test_failed_setup_never_releases_science(tmp_path,monkeypatch):
    s=spec(tmp_path);calls,_=fake_scheduler(monkeypatch);submission.launch(s)
    p=submission.plan(s,'bootstrap');submission.launch(s,execute=True,plan_hash=p['content_hash'],authorization=c.AUTHORIZATION)
    monkeypatch.setattr(submission,'accounting',lambda j:['FAILED','1:0'])
    with pytest.raises(RuntimeError,match='setup failed'):submission.finish(s,'9002')
    assert len(calls)==2


def test_ambiguous_submission_preserves_claim_and_never_retries(tmp_path,monkeypatch):
    s=spec(tmp_path);calls,_=fake_scheduler(monkeypatch);submission.launch(s)
    folder=tmp_path/'submission_bootstrap';(folder/'live_claim').mkdir()
    p=submission.plan(s,'bootstrap')
    with pytest.raises(RuntimeError,match='ambiguous/partial'):
        submission.launch(s,execute=True,plan_hash=p['content_hash'],authorization=c.AUTHORIZATION)
    assert not calls and (folder/'live_claim').is_dir()


def test_worker_runs_disposable_preparation_before_torchrun(tmp_path,monkeypatch):
    s=spec(tmp_path);calls=[]
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(worker,'require_job',lambda *a:'123')
    monkeypatch.setattr(execution,'allocation',lambda s:devices())
    monkeypatch.setattr(worker,'installed_environment',lambda:s['reference']['environment'])
    def run(cmd,**kw):calls.append(cmd);assert kw['check'] is True
    monkeypatch.setattr(worker.subprocess,'run',run)
    worker.run(s,'setup')
    assert calls[0][-3]=='prepare'
    assert 'torch.distributed.run' in calls[1] and '--nproc-per-node=2' in calls[1]
    assert calls[1][-1]=='setup'
    with pytest.raises(FileExistsError):worker.run(s,'setup')


def test_historical_control_never_substitutes_for_fresh_control(tmp_path,monkeypatch,capsys):
    s=spec(tmp_path);labels=cache('validation').labels;ids=cache('validation').identities
    prob=np.full((22,11),.01,np.float32);prob[np.arange(22),labels]=.9
    # Change a few class rankings so the two anchors have a nonzero gap.
    worse=prob.copy();worse[:11]=np.roll(worse[:11],1,axis=1)
    arrays={k:dict(labels=labels,identities=ids,probabilities=p) for k,p in
            (('M0HLT',worse),('OFFLINE',prob),('C0_BF16_HISTORICAL',prob))}
    monkeypatch.setattr(reporting,'references',lambda s:(SimpleNamespace(path=lambda p:p),arrays))
    monkeypatch.setattr(reporting,'load_npz_arrays',lambda p:dict(indices=np.arange(22)))
    monkeypatch.setattr(reporting,'states',lambda s:{})
    result=reporting.results(s)
    assert all(r['delta_C0'] is None for r in result['rows'])
    assert 'C0_BF16_HISTORICAL' in capsys.readouterr().out


@pytest.mark.parametrize('task',('C0','S100'))
def test_full_depth_fp32_cuda_global_and_partial_batches(tmp_path,task):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    s=spec(tmp_path);tr=cache();bb=banks(s,tr);ctx=d.Context(0,1,torch.device('cuda',0))
    m,w,o=d.build(s,task,ctx)
    assert len(m.mod.blocks)==8 and len(m.mod.cls_blocks)==2
    for n in (128,98,11):
        ix=np.arange(n)%len(tr)
        parts,probe,_=d.step(m,w,o,task,tr.batch(ix),bb,ix,3.,ctx,diagnose=n==11)
        assert np.isfinite(list(parts.values())).all()
        assert all(p.dtype==torch.float32 for p in m.parameters())
    assert probe['ce_norm']>0
    if task=='C0':assert probe['weighted_subclass_norm']==0
    else:assert probe['weighted_subclass_norm']>0


def _cuda_worker(rank,root,init_method):
    torch.set_num_threads(1);torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    dist.init_process_group('nccl',init_method=init_method,rank=rank,world_size=2,timeout=timedelta(seconds=180))
    try:
        s=spec(Path(root));tr=cache();va=cache('validation');bb=banks(s,tr);ctx=d.Context(rank,2,torch.device('cuda',rank))
        result=resources.qualification(s,tr,bb,ctx)
        assert result['probe_neutrality']['passed']
        for task in ('C0','S100'):
            r=resources.miniature(Path(root)/task,s,task,tr,va,bb,ctx)
            assert r['world_size']==2 and r['native_reload']
    finally:dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count()<2 or not dist.is_nccl_available(),reason='Two CUDA GPUs and NCCL required')
def test_real_two_gpu_syncbn_parity_diagnostics_and_native_reload(tmp_path):
    banks(spec(tmp_path));(tmp_path/'setup').mkdir()
    torch.multiprocessing.spawn(_cuda_worker,args=(str(tmp_path),(tmp_path/'store').as_uri()),nprocs=2,join=True)


def test_source_guard_requires_complete_additive_runtime(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1];assert all((root/p).is_file() for p in campaign.REQUIRED)
    tracked=set(campaign.REQUIRED);changes=['A\t'+p for p in campaign.RUNTIME]
    monkeypatch.setattr(campaign,'_source',lambda *a:dict(content_hash='s'*64))
    def git(args,**kwargs):
        if 'ls-files' in args:out='\n'.join(tracked)
        elif 'diff' in args:out='\n'.join(changes)
        else:out=''
        return SimpleNamespace(stdout=out,returncode=0)
    monkeypatch.setattr(campaign.subprocess,'run',git)
    assert campaign.current(root,'a'*40)==dict(content_hash='s'*64)
    changes.append('M\tsrc/hlt_classification/luka_subclass_hydra/model.py')
    with pytest.raises(ValueError):campaign.current(root,'a'*40)
    changes.pop();tracked.remove('tests/test_luka_subclass_ddp_hydra.py')
    with pytest.raises(ValueError):campaign.current(root,'a'*40)


def test_rank0_failure_is_shared_without_publishing(tmp_path):
    ctx=d.Context(0,1,torch.device('cpu'))
    with pytest.raises(RuntimeError,match='broken'):
        ctx.main(lambda:(_ for _ in ()).throw(ValueError('broken')))


def test_allocation_checks_typed_two_gpu_shape(tmp_path,monkeypatch):
    from hlt_classification.luka_subclass_hydra import benchmark_site as site
    import sys
    s=spec(tmp_path);s['environment_dir']=str(Path(sys.prefix).resolve())
    monkeypatch.setenv('SLURM_JOB_ID','123');monkeypatch.setenv('PYTHONNOUSERSITE','1')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    for key in ('SLURM_CLUSTER_NAME','SLURM_NTASKS','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE'):
        monkeypatch.delenv(key,raising=False)
    fields=dict(JobId='123',JobState='RUNNING',Account='default',Partition='gpus',QOS='normal',
        NumNodes='1',NumCPUs='6',NumTasks='1',**{'CPUs/Task':'6'},
        AllocTRES='cpu=6,mem=48000M,gres/gpu=2,gres/gpu:geforce_gtx_1080_ti=2')
    def controller(cmd,**kwargs):return SimpleNamespace(stdout='ClusterName = hydra\n' if cmd[-1]=='config'
        else ' '.join(k+'='+v for k,v in fields.items()))
    monkeypatch.setattr(site.subprocess,'run',controller);monkeypatch.setattr(site.platform,'machine',lambda:'x86_64')
    monkeypatch.setattr(site,'environment',lambda e:e)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True);monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'GeForce GTX 1080 Ti')
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda i:(6,1))
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda i:SimpleNamespace(total_memory=11*2**30))
    assert execution.allocation(s)==devices()
    fields['AllocTRES']=fields['AllocTRES'].replace('gres/gpu=2','gres/gpu=1')
    with pytest.raises(PermissionError):execution.allocation(s)

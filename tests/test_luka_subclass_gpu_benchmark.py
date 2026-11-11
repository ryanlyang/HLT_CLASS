"""Benchmark isolation, global-batch semantics, exact submission and CUDA kernels."""
import copy
import os
from pathlib import Path
from types import SimpleNamespace
import time
from datetime import timedelta
import numpy as np
import pytest
import torch
from torch import nn
from test_luka_u100_pretrain import cache
from test_luka_subclass_hydra import spec as science_spec, small, threads
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_subclass_hydra import (benchmark as b, benchmark_training as t,
    benchmark_submission as s, benchmark_site as site, contracts, campaign)


def spec(root):
    result=science_spec(root); result.update(b.registration())
    result['root']=str(root); result['reference']={'environment':{'fixture':True}}
    return result


def batch(n=16):
    data=cache(); ix=np.arange(n)
    raw=data.batch(ix); q=np.zeros((n,10),np.float32)
    q[raw['labels']==0]=.1; q[raw['labels']!=0,:5]=.2
    raw.update(indices=ix.astype(np.int64),u100=np.ones((n,128),np.float32),subclass=q)
    return raw


def test_benchmark_cannot_be_used_as_scientific_acceptance(tmp_path):
    cfg=b.registration(); p=s.plan(spec(tmp_path))
    assert cfg['science_release'] is False and cfg['precision']=='fp32'
    assert cfg['global_batch']==128 and cfg['tasks']==['S0p5','S100']
    assert len(p['commands'])==1 and not p['commands'][0]['dependencies']
    cmd=p['commands'][0]['command']
    assert '--gres=gpu:geforce_gtx_1080_ti:2' in cmd and '--time=02:00:00' in cmd and '--nodes=1' in cmd
    with pytest.raises(ValueError): contracts.validate(b.artifact('SPEC'),'SPEC')
    with pytest.raises(ValueError): contracts.validate(b.artifact('REPORT'),'ACCEPTANCE')


@pytest.mark.parametrize('n',[16,15,11])
def test_global_padding_and_uneven_gradient_weights(n):
    raw=batch(n); parts=[t.shard(raw,r,2) for r in range(2)]
    assert np.array_equal(np.concatenate([x[0]['identities'] for x in parts]),raw['identities'])
    assert all(x[0]['features'].shape[2:]==raw['features'].shape[2:] for x in parts)
    # DDP averages rank gradients. This weighting recovers a mean over jets,
    # including odd final minibatches with unequal local row counts.
    global_grad=torch.arange(n,dtype=torch.float64).mean()
    distributed=sum(torch.from_numpy(x[0]['indices']).double().mean()*x[1] for x in parts)/2
    assert torch.equal(global_grad,distributed)
    with pytest.raises(ValueError): t.shard(raw,2,2)


@pytest.mark.parametrize('bad',['labels','ids','targets','fields','finite'])
def test_benchmark_batches_reject_invalid_data(bad):
    raw=batch(); b.check_batch(raw,16,'train')
    if bad=='labels': raw['labels'][0]=11
    elif bad=='ids': raw['identities'][0]=raw['identities'][1]
    elif bad=='targets': raw['subclass'][0]=0
    elif bad=='fields': raw['construction_index']=np.zeros(16)
    else: raw['features'][0,0,0]=np.nan
    with pytest.raises(ValueError): b.check_batch(raw,16,'train')


def test_real_batch_publication_and_corruption_rejection(tmp_path,monkeypatch):
    cfg=spec(tmp_path)
    class Cache:
        def __init__(self,n,role):
            self.role=role; self.labels=np.arange(n,dtype=np.int64)%11
            self.identities=np.zeros((n,32),np.uint8)
            self.identities[:,:4]=np.arange(n,dtype=np.uint32).view(np.uint8).reshape(n,4)
            self.identities[:,-1]=int(role=='validation')
            self.blocks=[SimpleNamespace(offsets=np.arange(n+1)*16)]
        def __len__(self):return len(self.labels)
        def batch(self,ix):
            a=cache().batch(ix%22); a['labels']=self.labels[ix]; a['identities']=self.identities[ix]
            return a
    tr,va=Cache(1931,'train'),Cache(1024,'validation'); cfg['counts']=dict(train=len(tr),validation=len(va))
    q=np.zeros((len(tr),10),np.float32);q[tr.labels==0]=.1;q[tr.labels!=0,:5]=.2
    banks=dict(u100=SimpleNamespace(targets=np.zeros((len(tr),128),np.float32)),
        subclass=SimpleNamespace(targets=q,manifest={'content_hash':'q'*64}))
    monkeypatch.setattr(b.targets,'check_manifest',lambda s:banks['subclass'].manifest)
    m=b.publish_batches(cfg,tr,va,banks,preparation_started=time.perf_counter())
    actual,arrays=b.read_batches(cfg); assert actual==m and len(arrays)==28
    assert len(arrays['tail']['labels'])==11 and arrays['val_00']['identities'][0,-1]==1
    path=tmp_path/'batches/train_00.npz'; path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError):b.read_batches(cfg)


def test_parity_reports_real_disagreement_without_releasing_science():
    a=dict(loss=1.,gradients=torch.tensor([1.,2.]),parameters=torch.ones(2),
        moments=torch.ones(4),buffers={'bn.running_mean':torch.zeros(2)})
    assert t.compare(a,copy.deepcopy(a),b.registration()['parity_tolerances'])['passed']
    bad=copy.deepcopy(a);bad['gradients'][0]+=1
    assert not t.compare(a,bad,b.registration()['parity_tolerances'])['passed']


@pytest.mark.parametrize('task',['S0p5','S100'])
def test_fp32_wrapper_and_optimizer_match_production_objective(tmp_path,monkeypatch,task):
    small(monkeypatch); cfg=spec(tmp_path); raw=batch()
    student,wrapped,opt=t.build(cfg,task,0,1,'cpu',parity=True)
    assert all(not isinstance(x,nn.Dropout) or x.p==0 for x in student.modules())
    loss=t.update(wrapped,student,opt,raw,task,0,1,'cpu')
    snap=t.parity_snapshot(student,opt,loss)
    assert np.isfinite(loss) and torch.isfinite(snap['gradients']).all() and snap['gradients'].norm()>0
    assert any('running_mean' in k for k in snap['buffers'])
    assert any(p in opt.state for p in student.predictor.parameters())
    wrapped.eval()
    with torch.no_grad():
        logits,pred,sub=wrapped(**{k:torch.from_numpy(raw[k]) for k in ('features','vectors','mask')})
    assert logits.shape==(16,11) and pred is None and sub is None


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_single_gpu_fp32_measurement_full_and_partial(tmp_path,monkeypatch):
    small(monkeypatch); cfg=spec(tmp_path); cfg['counts']=dict(train=22,validation=22)
    raw={k:np.concatenate([v]*8) for k,v in batch().items()}
    batches={f'train_{i:02}':raw for i in range(15)}
    batches.update({f'val_{i:02}':raw for i in range(8)})
    batches.update(longest=raw,partial11=batch(11),tail=batch(15))
    row=t.measure_task(cfg,'S100',batches,0,1,torch.device('cuda'))
    assert len(row['step_seconds'])==12 and min(row['step_seconds'])>0
    assert row['longest_support_pass'] and row['peak_gpu_gib'][0]>0


def test_single_job_submission_is_exact_and_idempotent(tmp_path,monkeypatch):
    cfg=spec(tmp_path); calls=[]
    monkeypatch.setattr(b,'check',lambda *a,**k:None)
    monkeypatch.setattr(s,'scheduler_review',lambda p:None)
    import hlt_classification.scouting.hcwdl_exact_dag_submission as dag
    monkeypatch.setattr(dag.subprocess,'run',lambda args,**kw:(calls.append(args) or SimpleNamespace(stdout='12345\n')))
    s.submit(cfg);p=s.plan(cfg)
    with pytest.raises(PermissionError):s.submit(cfg,execute=True,plan_hash='wrong',authorization=s.AUTHORIZATION)
    result=s.submit(cfg,execute=True,plan_hash=p['content_hash'],authorization=s.AUTHORIZATION)
    assert result['jobs']=={'probe':'12345'} and s.jobs(cfg)==result['jobs']
    assert s.submit(cfg,execute=True,plan_hash=p['content_hash'],authorization=s.AUTHORIZATION)==result
    assert len(calls)==1
    monkeypatch.setenv('SLURM_JOB_ID','12345'); assert s.require_job(cfg)=='12345'


def test_partial_submission_is_never_blindly_retried(tmp_path,monkeypatch):
    cfg=spec(tmp_path);monkeypatch.setattr(b,'check',lambda *a,**k:None)
    monkeypatch.setattr(s,'scheduler_review',lambda p:None)
    s.submit(cfg);(tmp_path/'submission/live_claim').mkdir()
    with pytest.raises(RuntimeError,match='Ambiguous'):
        s.submit(cfg,execute=True,plan_hash=s.plan(cfg)['content_hash'],authorization=s.AUTHORIZATION)


def test_benchmark_registration_has_own_identity(tmp_path,monkeypatch):
    import sys
    parent=tmp_path/'parent';parent.mkdir();project=tmp_path/'project';project.mkdir()
    old=science_spec(parent);old.update(project_dir=str(parent),environment_dir=str(Path(sys.prefix).resolve()),
        offline_root=str(tmp_path/'offline'),qualification={'fixture':True})
    source=parent/'campaign_spec.json';write_immutable_json(source,old)
    monkeypatch.setattr(b.reuse,'original',lambda p:load_json(p))
    monkeypatch.setattr(b.reuse,'references',lambda s:{'fixture':True})
    monkeypatch.setattr(campaign,'current',lambda *a:{'content_hash':'f'*64})
    monkeypatch.setattr(campaign.site,'environment',lambda *a:None)
    monkeypatch.setattr(b.inputs,'validate_sources',lambda s:None)
    cfg=b.create(tmp_path/'bench',source,project,'c'*40);b.check(cfg)
    assert not (tmp_path/'bench/campaign_spec.json').exists() and cfg['benchmark_only']
    with pytest.raises(ValueError):b.create(tmp_path/'bench',source,project,'c'*40)


def test_two_card_scheduler_and_gpu_identity(tmp_path,monkeypatch):
    import sys
    cfg=spec(tmp_path);cfg['environment_dir']=str(Path(sys.prefix).resolve())
    monkeypatch.setenv('SLURM_JOB_ID','123');monkeypatch.setenv('PYTHONNOUSERSITE','1')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    for key in ('SLURM_NTASKS','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE','SLURM_CLUSTER_NAME'):
        monkeypatch.delenv(key,raising=False)
    fields='JobId=123 JobState=RUNNING Account=default QOS=normal Partition=gpus NumNodes=1 NumCPUs=6 NumTasks=1 CPUs/Task=6 AllocTRES=cpu=6,mem=32000M,gres/gpu=2,gres/gpu:geforce_gtx_1080_ti=2'
    monkeypatch.setattr(site.subprocess,'run',lambda cmd,**kw:SimpleNamespace(stdout='ClusterName = hydra' if cmd[-1]=='config' else fields))
    monkeypatch.setattr(site,'environment',lambda e:e);monkeypatch.setattr(site.platform,'machine',lambda:'x86_64')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True);monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'GeForce GTX 1080 Ti')
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda i:(6,1))
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda i:SimpleNamespace(total_memory=11*2**30))
    assert len(site.allocation(cfg))==2
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    with pytest.raises(PermissionError):site.allocation(cfg)


def test_report_compares_campaign_throughput_and_keeps_failures_visible(tmp_path):
    cfg=spec(tmp_path)
    for world,seconds in ((1,2.),(2,1.25)):
        row=dict(mean_step_seconds=seconds,longest_support_pass=True)
        write_immutable_json(tmp_path/f'world{world}.json',b.artifact('PHASE',parents=dict(spec=cfg['content_hash']),
            world_size=world,precision='fp32',global_batch=128,tasks={x:row for x in cfg['tasks']}))
    report=b.report(cfg,{'1':0,'2':0})
    assert report['comparisons']['S100']['two_gpu_speedup']==1.6
    assert report['comparisons']['S100']['two_independent_one_gpu_runs_relative_throughput']==1.25
    assert not report['science_qualified'] and not report['science_submitted']
    failed=b.report(cfg,{'1':0,'2':1})
    assert not failed['comparisons'] and '2' not in failed['phases']


@pytest.mark.parametrize('single_exit',[0,1])
def test_worker_always_attempts_second_phase_and_preserves_failures(tmp_path,monkeypatch,single_exit):
    cfg=spec(tmp_path);calls=[]
    monkeypatch.setattr(b,'check',lambda *a,**k:None)
    monkeypatch.setattr(site,'allocation',lambda s:[])
    monkeypatch.setattr(s,'require_job',lambda s:'123')
    def process(cmd,**kw):
        calls.append(cmd)
        if 'prepare' in cmd: return SimpleNamespace(returncode=0)
        world=int(next(x.split('=')[1] for x in cmd if x.startswith('--nproc-per-node=')))
        code=single_exit if world==1 else 0
        if not code:
            row=dict(mean_step_seconds=1.,longest_support_pass=True,train_only_30_epoch_hours=20.,peak_gpu_gib=[1.]*world)
            write_immutable_json(tmp_path/f'world{world}.json',b.artifact('PHASE',parents=dict(spec=cfg['content_hash']),
                world_size=world,precision='fp32',global_batch=128,tasks={x:row for x in cfg['tasks']},parity={}))
        return SimpleNamespace(returncode=code)
    monkeypatch.setattr(b.subprocess,'run',process)
    if single_exit:
        with pytest.raises(RuntimeError,match='phase failed'):b.run(cfg)
    else:b.run(cfg)
    report=load_json(tmp_path/'report.json')
    assert report['phase_exit_codes']=={'1':single_exit,'2':0} and not report['science_submitted']
    assert len(calls)==3 and 'prepare' in calls[0]
    assert '--nproc-per-node=1' in calls[1] and '--nproc-per-node=2' in calls[2]
    assert (tmp_path/'run_claim/intent.json').is_file() and (tmp_path/'world2.log').is_file()


def _cpu_ddp_worker(rank,root,init_method):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=init_method,rank=rank,world_size=2,timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(12); model=nn.Linear(17,11).double()
        wrapped=DistributedDataParallel(model)
        raw=batch(15); local,weight=t.shard(raw,rank,2)
        loss=nn.functional.cross_entropy(wrapped(torch.from_numpy(local['features'][:,:,0]).double()),torch.from_numpy(local['labels']))
        (loss*weight).backward()
        if rank==0: torch.save({k:p.grad for k,p in model.named_parameters()},Path(root)/'ddp_gradients.pt')
    finally:dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(),reason='Gloo unavailable')
def test_actual_two_process_ddp_preserves_partial_global_gradient(tmp_path):
    torch.multiprocessing.spawn(_cpu_ddp_worker,args=(str(tmp_path),(tmp_path/'store').as_uri()),nprocs=2,join=True)
    torch.manual_seed(12); model=nn.Linear(17,11).double();raw=batch(15)
    nn.functional.cross_entropy(model(torch.from_numpy(raw['features'][:,:,0]).double()),torch.from_numpy(raw['labels'])).backward()
    actual=torch.load(tmp_path/'ddp_gradients.pt',weights_only=True)
    for k,p in model.named_parameters():torch.testing.assert_close(actual[k],p.grad,atol=1e-12,rtol=1e-12)


def _cuda_ddp_worker(rank,root,init_method):
    import torch.distributed as dist
    torch.set_num_threads(1);torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    dist.init_process_group('nccl',init_method=init_method,rank=rank,world_size=2,timeout=timedelta(seconds=90))
    try:
        with pytest.MonkeyPatch.context() as patch:
            small(patch);cfg=spec(Path(root))
            checks=t.parity(cfg,{f'parity{n}':batch(n) for n in (16,15)},rank,2,torch.device('cuda',rank))
            if rank==0: assert all(row['passed'] for row in checks.values()),checks
    finally:dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count()<2 or not torch.distributed.is_nccl_available(),reason='Two CUDA GPUs and NCCL required')
def test_actual_two_gpu_sync_batchnorm_parity(tmp_path,monkeypatch):
    small(monkeypatch);cfg=spec(tmp_path)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    t.parity(cfg,{f'parity{n}':batch(n) for n in (16,15)},0,1,torch.device('cuda',0))
    torch.multiprocessing.spawn(_cuda_ddp_worker,args=(str(tmp_path),(tmp_path/'store').as_uri()),nprocs=2,join=True)

"""Bounded R24 profiling, real gradients, uneven shards and immutable submission."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from hlt_classification.data.cache_contracts import (load_json, write_immutable_json,
    deterministic_npz_bytes, atomic_publish_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_reconstruction_benchmark import campaign as c,data,kernel as k,submission as s,runner
from hlt_classification.luka_next_supervision_hydra import contracts, reconstruction_model as rm, reconstruction_batched as loss
from hlt_classification.luka_next_supervision_hydra import reconstruction_training as original
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from test_luka_direct_reconstruction import target,fixture,compact_native


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads(); workers=loss.ASSIGNMENT_WORKERS; torch.set_num_threads(1)
    yield
    if loss.ASSIGNMENT_WORKERS!=workers:
        if loss.solver_pool.cache_info().currsize: loss.solver_pool().shutdown(wait=True)
        loss.solver_pool.cache_clear(); loss.ASSIGNMENT_WORKERS=workers
    torch.set_num_threads(old)


def spec(root):
    return dict(c.policy(),content_hash='a'*64,root=str(root),project_dir=str(root/'code'),maximum=4,
                counts=dict(train=299234,validation=74895))


def batch(n=11):
    rng=np.random.default_rng(42); ids=np.zeros((n,32),np.uint8); ids[:,0]=np.arange(n)
    raw=dict(features=rng.normal(size=(n,6,3)).astype(np.float32),vectors=np.zeros((n,4,3),np.float32),
        mask=np.ones((n,1,3),bool),identities=ids,labels=np.arange(n,dtype=np.int64)%11)
    return raw,rng.normal(size=(n,128)).astype(np.float32),[target(1+i%4) for i in range(n)]


class TinyNative(nn.Module):
    """No normalization: isolate exact VJP and uneven global loss reduction."""
    def __init__(self):
        super().__init__(); self.cfg=contracts.setting('R24'); self.encoder=nn.Linear(6,128)
        self.predictor=nn.Linear(128,128); self.classifier=nn.Linear(128,11); self.aux=rm.Decoder(self.cfg,4)
    def forward(self,features,vectors,mask):
        self.particle_tokens=self.encoder(features.transpose(1,2)); self.particle_mask=mask[:,0]
        self.jet=self.particle_tokens.mean(1); self.ce_representation=self.jet
        self.prediction=self.predictor(self.jet); return self.classifier(self.jet)
    def no_weight_decay(self): return set()


@pytest.mark.parametrize('world',[2,3,4])
@pytest.mark.parametrize('n',[11,15,16,128])
def test_shards_keep_global_padding_and_equal_jet_weight(n,world):
    b=batch(n); pieces=[k.shard(b,r,world) for r in range(world)]
    assert sum(f for _,f in pieces)==pytest.approx(1.)
    assert [len(p[0]['features']) for p,f in pieces]==[len(x) for x in np.array_split(np.arange(n),world)]
    assert all(p[0]['features'].shape[-1]==b[0]['features'].shape[-1] for p,f in pieces)
    assert np.array_equal(np.concatenate([p[0]['identities'] for p,f in pieces]),b[0]['identities'])
    expected=np.arange(n).mean()
    got=sum(p[0]['identities'][:,0].mean()*f for p,f in pieces)
    assert got==pytest.approx(expected)


@pytest.mark.parametrize('world',[2,3,4])
@pytest.mark.parametrize('position',[3.,4.5,7.])
def test_sample_weighted_VJP_matches_independent_full_graph_and_adamw(world,position,monkeypatch):
    torch.manual_seed(4); reference=TinyNative(); template=deepcopy(reference); opt=k.optimizer(reference)
    b=batch(); raw,emb,targets=b; logits=k.native(reference,raw,'cpu')
    ce=F.cross_entropy(logits,torch.from_numpy(raw['labels'])); e=F.smooth_l1_loss(reference.prediction,torch.from_numpy(emb))
    values,_=loss.rows(reference.aux(reference.jet,reference.particle_tokens,reference.particle_mask),targets)
    w=contracts.coefficient(reference.cfg,position); total=ce+5*e+w*values.mean(); total.backward()
    for group in opt.param_groups: group['lr']=learning_rate(position)
    opt.step(); expected=k.snapshot(reference,opt,float(total.detach()))
    accumulated=deepcopy(template); total_loss=0.; gradients=[]
    monkeypatch.setattr(k,'reduce_gradients',lambda *a,**kw:None)
    monkeypatch.setattr(k.dist,'all_reduce',lambda *a,**kw:None)
    for rank in range(world):
        local=deepcopy(template); optimizer=k.optimizer(local); monkeypatch.setattr(optimizer,'step',lambda:None)
        scalar,_=k.update(local,optimizer,b,rank,world,'cpu',position); total_loss+=scalar
        gradients.append([p.grad.detach().clone() for p in local.parameters()]); del local,optimizer
    for i,p in enumerate(accumulated.parameters()): p.grad=sum(x[i] for x in gradients)
    optimizer=k.optimizer(accumulated)
    for group in optimizer.param_groups: group['lr']=learning_rate(position)
    optimizer.step(); actual=k.snapshot(accumulated,optimizer,total_loss)
    metrics=k.compare(expected,actual,c.policy()['tolerances']); assert metrics['passed'],metrics


@pytest.mark.parametrize('position',[3.,4.5,7.])
def test_real_native_update_matches_original_FP32_execution(tmp_path,monkeypatch,position):
    old,tr,_,bank=fixture(tmp_path,monkeypatch,rows=11)
    source=original_targets(old,tr); ix=np.arange(11); targets,_=source.batch(ix,1,contracts.setting('R24'))
    model,opt=k.build(old,12,0,1,'cpu',parity=True); ref=deepcopy(model); ropt=k.optimizer(ref)
    expected,_=original.step(ref,ropt,tr.batch(ix),bank.targets,'cpu',candidates=source,indices=ix,
        epoch=1,position=position,probe=False)
    got,_=k.update(model,opt,(tr.batch(ix),bank.targets,targets),0,1,'cpu',position)
    metrics=k.compare(k.snapshot(ref,ropt,float(expected[-1])),k.snapshot(model,opt,got),c.policy()['tolerances'])
    assert metrics['passed'],metrics; source.close()


def original_targets(old,tr):
    from hlt_classification.luka_next_supervision_hydra.reconstruction_targets import Targets
    return Targets(old,tr,'OFFLINE')


def test_ragged_serialization_does_not_put_targets_in_native_inputs():
    b=batch(); a=data.pack(*b); raw,emb,targets=data.unpack(a)
    assert set(raw)=={'features','vectors','mask','labels','identities'}
    np.testing.assert_array_equal(emb,b[1])
    for x,y in zip(targets,b[2]):
        for key in x: np.testing.assert_array_equal(x[key],y[key])
    native={key:b[0][key] for key in data.NATIVE}; assert data.unpack(native)[1:]==(None,None)
    with pytest.raises(ValueError):data.unpack(dict(native,labels=b[0]['labels']))
    with pytest.raises(ValueError):data.unpack(dict(a,source_index=np.arange(11)))
    a['labels'][0]=11
    with pytest.raises(ValueError):data.unpack(a)


def test_corrupt_batch_is_rejected_before_use(tmp_path):
    cfg=spec(tmp_path); folder=tmp_path/'batches'; folder.mkdir()
    names=[f'train_{i:02}' for i in range(27)]+[f'val_{i:02}' for i in range(32)]
    names+=['parity_0','parity_1','parity_2','longest_hlt','largest_offline','partial11','tail']
    outputs={}; b=batch(); b[0]['features']=np.zeros((11,17,3),np.float32)
    for name in names:
        val=name.startswith('val_'); a={key:b[0][key] for key in data.NATIVE} if val else data.pack(*b)
        p=folder/(name+'.npz'); atomic_publish_bytes(p,deterministic_npz_bytes(a))
        outputs[name]=dict(file=file_ref(p,root=folder),role='validation' if val else 'train',rows=11)
    m=c.artifact('BATCHES',parents=dict(spec=cfg['content_hash']),outputs=outputs,maximum=4,native_validation_only=True)
    write_immutable_json(folder/'manifest.json',m); assert len(data.read(cfg)[1])==len(names)
    p=folder/'train_00.npz'; p.write_bytes(p.read_bytes()+b'corrupt')
    with pytest.raises(ValueError):data.read(cfg)


def test_projection_contains_full_validation_diagnostics_and_preparation(tmp_path):
    result=c.projection(spec(tmp_path),mean_step=2.,mean_validation_batch=.1,probe_seconds=10.,input_seconds=.01,startup_seconds=100.)
    parts=result['component_seconds']
    assert parts==dict(training=140280.,input_construction=701.4,validation=1758.,diagnostics=300.,startup=100.)
    assert result['estimated_hours']==pytest.approx(sum(parts.values())/3600)
    for bad in (-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):c.projection(spec(tmp_path),mean_step=bad,mean_validation_batch=.1,probe_seconds=1.,input_seconds=1.,startup_seconds=1.)


def test_four_card_single_job_exact_idempotent_submission(tmp_path,monkeypatch):
    cfg=spec(tmp_path); calls=[]; monkeypatch.setattr(c,'check',lambda *a,**kw:None)
    monkeypatch.setattr(s,'scheduler_review',lambda p:None)
    import hlt_classification.scouting.hcwdl_exact_dag_submission as dag
    monkeypatch.setattr(dag.subprocess,'run',lambda args,**kw:(calls.append(args) or SimpleNamespace(stdout='12345\n')))
    s.submit(cfg); p=s.plan(cfg); command=p['commands'][0]['command']
    assert len(p['commands'])==1 and not p['science_release'] and not p['commands'][0]['dependencies']
    assert '--gres=gpu:geforce_gtx_1080_ti:4' in command and '--cpus-per-task=12' in command
    assert not any(x.startswith(('--dependency','--nodelist','--array')) for x in command)
    with pytest.raises(PermissionError):s.submit(cfg,execute=True,plan_hash='wrong',authorization=s.AUTHORIZATION)
    value=s.submit(cfg,execute=True,plan_hash=p['content_hash'],authorization=s.AUTHORIZATION)
    assert value['jobs']=={'probe':'12345'} and s.jobs(cfg)==value['jobs']
    assert s.submit(cfg,execute=True,plan_hash=p['content_hash'],authorization=s.AUTHORIZATION)==value
    assert len(calls)==1


def test_ambiguous_submission_is_not_repeated(tmp_path,monkeypatch):
    cfg=spec(tmp_path); monkeypatch.setattr(c,'check',lambda *a,**kw:None); monkeypatch.setattr(s,'scheduler_review',lambda p:None)
    s.submit(cfg); (tmp_path/'submission/live_claim').mkdir()
    with pytest.raises(RuntimeError,match='Ambiguous'):
        s.submit(cfg,execute=True,plan_hash=s.plan(cfg)['content_hash'],authorization=s.AUTHORIZATION)


def test_registration_pins_both_sources_and_preserves_original(tmp_path,monkeypatch):
    import sys
    oldroot=tmp_path/'original';oldroot.mkdir(); oldproject=tmp_path/'oldcode';oldproject.mkdir()
    newproject=tmp_path/'newcode';newproject.mkdir()
    oldsnap={'content_hash':'e'*64}; newsnap={'content_hash':'f'*64}
    old=dict(source_commit=c.DONOR,campaign_kind='R8_FOLLOWUP_V2',execution_version='R5',
        content_hash='b'*64,source_snapshot=oldsnap,root=str(oldroot),project_dir=str(oldproject),
        environment_dir=str(Path(sys.prefix).resolve()),counts=dict(train=299234,validation=74895),
        target_inventory=dict(maximum=298),**{key:str(tmp_path/key) for key in ('bundle_root','raw_root','delta_root')})
    original=oldroot/'campaign_spec.json';write_immutable_json(original,old); before=original.read_bytes()
    reference=dict(report_hash='d'*64,files=[])
    monkeypatch.setattr(c,'original',lambda p:old);monkeypatch.setattr(c,'reference',lambda o:reference)
    monkeypatch.setattr(c,'source',lambda *a:newsnap);monkeypatch.setattr(c,'capture_source_snapshot',lambda *a,**kw:oldsnap)
    monkeypatch.setattr(c.donor,'__file__',str(oldproject/'src/hlt_classification/luka_next_supervision_hydra/campaign.py'))
    cfg=c.create(tmp_path/'bench',original,newproject,'c'*40,3); assert c.check(cfg,full=True)==old
    assert original.read_bytes()==before and cfg['maximum']==298 and cfg['science_release'] is False
    with pytest.raises(ValueError):c.create(tmp_path/'bench',original,newproject,'c'*40,3)
    with pytest.raises(ValueError):c.create(oldroot/'nested',original,newproject,'c'*40,3)
    from hlt_classification.data.cache_contracts import with_content_hash
    bad=with_content_hash(dict(cfg,worlds=[2]))
    with pytest.raises(ValueError):c.check(bad)
    monkeypatch.setattr(c,'capture_source_snapshot',lambda *a,**kw:{'content_hash':'0'*64})
    with pytest.raises(ValueError,match='runtime changed'):c.check(cfg)


def test_batch_preparation_samples_epoch_and_full_support_without_mutating_donor(tmp_path,monkeypatch):
    oldroot=tmp_path/'old';oldroot.mkdir();newroot=tmp_path/'new';newroot.mkdir()
    old,tr,va,bank=fixture(oldroot,monkeypatch,rows=384);cfg=spec(newroot);cfg['maximum']=12
    cfg['counts']=old['counts'];monkeypatch.setattr(c,'check',lambda *a,**kw:old)
    monkeypatch.setattr(data.inputs,'hlt',lambda o,role:tr if role=='train' else va)
    monkeypatch.setattr(data.inputs,'embedding_bank',lambda *a:bank)
    before={p.name:p.read_bytes() for p in (oldroot/'setup').iterdir()}
    m=data.prepare(cfg); actual,batches=data.read(cfg)
    assert m==actual and batches['val_00'][1:] == (None,None)
    assert m['outputs']['train_00']['indices']!=m['outputs']['train_26']['indices']
    assert m['outputs']['tail']['rows']==128 and m['outputs']['partial11']['rows']==11
    assert before=={p.name:p.read_bytes() for p in (oldroot/'setup').iterdir()}


def test_allocation_requires_four_correct_cards_and_twelve_cpus(tmp_path,monkeypatch):
    cfg=spec(tmp_path);monkeypatch.setattr(s,'require_job',lambda _: '123')
    for key in ('SLURM_NTASKS','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE','SLURM_CLUSTER_NAME'):
        monkeypatch.delenv(key,raising=False)
    monkeypatch.setenv('PYTHONNOUSERSITE','1');monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    fields='JobId=123 JobState=RUNNING Account=default QOS=normal Partition=gpus NumNodes=1 NumCPUs=12 NumTasks=1 CPUs/Task=12 AllocTRES=cpu=12,mem=64000M,gres/gpu=4,gres/gpu:geforce_gtx_1080_ti=4'
    monkeypatch.setattr(s,'controller',lambda args:'ClusterName = hydra' if args[-1]=='config' else fields)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True);monkeypatch.setattr(torch.cuda,'device_count',lambda:4)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'GeForce GTX 1080 Ti')
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda i:(6,1))
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda i:SimpleNamespace(total_memory=11*2**30))
    assert len(s.allocation(cfg)[1])==4
    fields=fields.replace('NumCPUs=12','NumCPUs=6')
    with pytest.raises(PermissionError):s.allocation(cfg)


def test_failed_world_does_not_skip_later_worlds(tmp_path,monkeypatch):
    cfg=spec(tmp_path);cfg['reference']=dict(job_id='168731');called=[]
    monkeypatch.setattr(c,'check',lambda *a,**kw:{})
    monkeypatch.setattr(s,'allocation',lambda *a:('123',[]))
    monkeypatch.setattr(runner,'accounting',lambda *a:None)
    monkeypatch.setattr(runner,'results',lambda *a:None)
    def child(cfg,name,args,timeout):
        called.append(name);path=tmp_path/(name+'.log');path.write_text('log')
        if name=='prepare':
            (tmp_path/'batches').mkdir()
            write_immutable_json(tmp_path/'batches/manifest.json',c.artifact('BATCHES',parents=dict(spec=cfg['content_hash'])))
        return dict(name=name,exit_code=1 if name.startswith('world') else 0,elapsed_seconds=1.,log=file_ref(path,root=tmp_path))
    monkeypatch.setattr(runner,'child',child);runner.run(cfg)
    assert called==['prepare','reference','world2','world3','world4']
    result=load_json(tmp_path/'benchmark_report.json');assert not result['measurements'] and len(result['phases'])==3
    assert result['science_release'] is False and (tmp_path/'receipt.json').is_file()


def test_parity_disagreement_is_retained_and_zero_loss_not_accuracy_gate():
    ref=dict(loss=0.,gradients=torch.ones(2),parameters=torch.ones(2),moments=torch.ones(4),buffers={})
    assert k.compare(ref,deepcopy(ref),c.policy()['tolerances'])['passed']
    bad=deepcopy(ref);bad['gradients'][0]=2.
    assert not k.compare(ref,bad,c.policy()['tolerances'])['passed']


def _sum_worker(rank,world,root,init):
    import torch.distributed as dist
    torch.set_num_threads(1); dist.init_process_group('gloo',init_method=init,rank=rank,world_size=world,timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(42); model=nn.Sequential(nn.Linear(17,11),nn.Linear(11,11)).double()
        raw=batch(11)[0]; x=np.random.default_rng(0).normal(size=(11,17)); raw['features']=x
        (local,_,_),fraction=k.shard((raw,None,None),rank,world)
        (F.cross_entropy(model(torch.from_numpy(local['features'])),torch.from_numpy(local['labels']))*fraction).backward()
        k.reduce_gradients(model,world,bucket_elements=23)
        if rank==0: torch.save({n:p.grad for n,p in model.named_parameters()},Path(root)/'grad.pt')
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('world',[2,3,4])
@pytest.mark.skipif(not torch.distributed.is_gloo_available(),reason='Gloo unavailable')
def test_actual_process_gradient_sum_handles_uneven_shards_and_split_parameters(tmp_path,world):
    torch.multiprocessing.spawn(_sum_worker,args=(world,str(tmp_path),(tmp_path/'store').as_uri()),nprocs=world,join=True)
    torch.manual_seed(42); model=nn.Sequential(nn.Linear(17,11),nn.Linear(11,11)).double()
    x=np.random.default_rng(0).normal(size=(11,17)); y=batch(11)[0]['labels']
    F.cross_entropy(model(torch.from_numpy(x)),torch.from_numpy(y)).backward()
    actual=torch.load(tmp_path/'grad.pt',weights_only=True)
    for n,p in model.named_parameters():torch.testing.assert_close(actual[n],p.grad,atol=1e-12,rtol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_full_production_native_FP32_and_decoder_on_local_cuda(tmp_path,monkeypatch):
    import importlib
    from hlt_classification.jetclass2_delphes import model as native
    production=importlib.reload(native).model_config; monkeypatch.setattr(native,'model_config',production)
    old,tr,_,bank=fixture(tmp_path,monkeypatch,rows=16); source=original_targets(old,tr)
    targets,_=source.batch(np.arange(16),1,contracts.setting('R24')); device=torch.device('cuda')
    k.configure(1,12); model,opt=k.build(old,12,0,1,device,parity=True)
    result,detail=k.update(model,opt,(tr.batch(np.arange(16)),bank.targets,targets),0,1,device,7.)
    assert np.isfinite(result) and detail['weight']==.3
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.aux.parameters())
    model.eval();model.capture=False
    with torch.inference_mode(): output=k.native(model,tr.batch(np.arange(16)),device)
    assert output.shape==(16,11) and model.prediction is None and model.particle_tokens is None
    source.close(); del model,opt; torch.cuda.empty_cache()


def _cuda_worker(rank,world,root,init,old,batches):
    import torch.distributed as dist
    torch.cuda.set_device(rank); k.configure(world,12)
    dist.init_process_group('nccl',init_method=init,rank=rank,world_size=world,timeout=timedelta(seconds=180))
    try:
        cfg=spec(Path(root)); cfg['maximum']=12
        checks=runner.parity(cfg,old,batches,rank,world,torch.device('cuda',rank))
        if rank==0: assert all(x['passed'] for x in checks.values()),checks
    finally: dist.destroy_process_group()


@pytest.mark.parametrize('world',[2,3,4])
def test_real_cuda_syncbn_VJP_adamw_parity_when_hardware_available(tmp_path,monkeypatch,world):
    if torch.cuda.device_count()<world or not torch.distributed.is_nccl_available(): pytest.skip(f'{world} GPUs and NCCL required')
    import importlib
    from hlt_classification.jetclass2_delphes import model as native
    monkeypatch.setattr(native,'model_config',importlib.reload(native).model_config)
    cfg=spec(tmp_path);cfg['maximum']=12
    old,tr,_,bank=fixture(tmp_path,monkeypatch,rows=16); source=original_targets(old,tr); batches={}
    for i,case in enumerate(cfg['parity_cases']):
        ix=np.arange(case['rows']);targets,_=source.batch(ix,1,contracts.setting('R24'))
        batches[f'parity_{i}']=(tr.batch(ix),bank.targets[ix],targets)
    k.configure(1,12); runner.parity(cfg,old,batches,0,1,torch.device('cuda',0)); source.close()
    torch.multiprocessing.spawn(_cuda_worker,args=(world,str(tmp_path),(tmp_path/'store').as_uri(),old,batches),nprocs=world,join=True)

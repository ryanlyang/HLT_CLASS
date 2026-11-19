"""Balanced small-study science, exact auxiliary gradients and safe execution."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import math
import os
import subprocess
import numpy as np
import pytest
import torch
from torch import nn
from hlt_classification.data.cache_contracts import load_json,write_immutable_json,with_content_hash,sha256_file
from hlt_classification.luka_balanced_reconstruction_hydra import (
    contracts as c,population,inputs,kernel,model,artifacts,readouts,resources,training,storage,campaign,banks,reporting)
from hlt_classification.luka_next_supervision_hydra.reconstruction_model import Decoder
from hlt_classification.luka_fullstat_readout.training import optimizer
from hlt_classification.luka_subclass_ddp_hydra.distributed import Context
from test_luka_direct_reconstruction import target
from test_luka_fullstat import write_root
from test_luka_sixclass_balanced_oscar import cache


@pytest.fixture(autouse=True)
def threads():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Tiny(nn.Module):
    def __init__(self,task,maximum):
        super().__init__();self.task=task;self.cfg=c.decoder_config(task);self.mod=nn.Module()
        self.mod.encoder=nn.Linear(17,128);self.mod.fc=nn.Linear(128,6)
        self.predictor=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,128))
        self.aux=Decoder(self.cfg,maximum) if c.cell(task)['reconstruction'] else None
        self.capture=False;self.clear()
    def clear(self):self.jet=self.prediction=self.particle_tokens=self.particle_mask=None
    def forward(self,features,vectors,mask):
        self.particle_tokens=self.mod.encoder(features.transpose(1,2));self.particle_mask=mask[:,0]
        self.jet=(self.particle_tokens*self.particle_mask[...,None]).sum(1)/self.particle_mask.sum(1)[:,None]
        self.prediction=self.predictor(self.jet) if self.capture else None
        return self.mod.fc(self.jet)
    def no_weight_decay(self):return set()
    def deployment_state(self):return {k:v.detach().cpu().clone() for k,v in self.state_dict().items() if k.startswith('mod.')}


def test_matrix_clock_teacher_semantics_and_balancing():
    assert len(c.FITS)==18 and len(set(c.FITS))==18
    assert c.SEEDS==(10,11) and c.COUNTS==dict(train=600000,validation=150000)
    assert math.ceil(600000/c.BATCH)==4688 and 600000%c.BATCH==64
    assert c.weights('R24_s10',3)==(0.,0.) and c.weights('R24_s10',6)==(.3,.3)
    assert c.weights('D20_s10',2)==(0.,.3) and c.weights('D20_s10',9)==(.3,.3)
    assert c.weights('R20_s10',3)==(2.,2.) and c.weights('R22_s10',3)==(10.,10.)
    assert c.cell('R28_s10')['token_layers']==[4,8]
    assert c.cell('U100_s11')['teacher']=='OFFLINE_s11'
    assert c.cell('R24_s11')['teacher']=='U100_s11'
    assert c.cell('M0HLT_s10')['teacher'] is None
    assert c.recipe()['primary']=='EMA05'
    with pytest.raises(ValueError):c.cell('R24_s0')


def test_proportional_cells_preserve_the_expected_exposure():
    p=load_json(Path(__file__).resolve().parents[1]/'artifacts/audits/sixclass_balanced_oscar_v1/population.json')
    counts=np.asarray(p['class_bin_counts']);small=population.quotas(counts.ravel(),600000)
    assert small.sum()==600000 and small.min()==1234
    assert small.reshape(6,20).sum(1).tolist()==[427186,35806,34645,33939,33774,34650]
    assert 600000/(120*small.min())==pytest.approx(4.051863857)


def test_subset_is_deterministic_nested_and_keeps_parent_order():
    n=1200;ids=np.zeros((n,32),np.uint8);ids[:,:2]=np.array([[i%256,i//256] for i in range(n)],np.uint8)
    y=np.tile(np.arange(6),200);pt=np.tile(np.repeat(np.arange(650,2600,100),6),10)
    a,cells=population.select(ids,y,pt,600,'train');b,_=population.select(ids,y,pt,600,'train')
    assert np.array_equal(a,b) and np.all(np.diff(a)>0) and np.all(cells==5)
    assert len(a)==600
    with pytest.raises(PermissionError):population.select(ids,y,pt,600,'final_test')
    with pytest.raises(ValueError):population.select(ids,np.full(n,6),pt,600,'train')


@pytest.mark.parametrize('task',c.FITS[:9])
@pytest.mark.parametrize('position',[2.,8.])
def test_chunked_update_matches_independent_full_graph(task,position):
    torch.manual_seed(71);actual=Tiny(task,4);reference=deepcopy(actual)
    ctx=Context(0,1,torch.device('cpu'));ao=optimizer(actual,c.recipe()['optimizer']);ro=optimizer(reference,c.recipe()['optimizer'])
    tr=cache(rows=8);raw=tr.batch(np.arange(8));rng=np.random.default_rng(19)
    x=c.cell(task);emb=rng.normal(size=(8,128)).astype(np.float32) if x['embedding_weight'] else None
    q=np.full((8,6),1/6,np.float32) if x['main_kd_weight'] else None
    targets=[target(1+i%4) for i in range(8)] if x['reconstruction'] else None
    expected=resources.independent_step(reference,ro,raw,emb,q,targets,task,'cpu',position)
    losses=kernel.update(actual,ao,raw,emb,q,targets,task,ctx,position)
    measured=resources.snapshot(actual,ao,losses['total'])
    result=resources.compare(expected,measured,resources.TOLERANCES)
    assert result['passed'],result
    if not x['embedding_weight']:assert all(p.grad is None and p not in ao.state for p in actual.predictor.parameters())
    if task.startswith('R24') and position==2:
        assert all(float(ao.state[p]['step'])==1 and torch.count_nonzero(p.grad)==0 for p in actual.aux.parameters())


def test_ce_only_capabilities_and_detached_decoder_preparation():
    ctx=Context(0,1,torch.device('cpu'));raw=cache(rows=8).batch(np.arange(8))
    net=Tiny('M0HLT_s10',4);opt=optimizer(net,c.recipe()['optimizer'])
    with pytest.raises(PermissionError):kernel.update(net,opt,raw,np.zeros((8,128),np.float32),None,None,'M0HLT_s10',ctx,1.)
    with pytest.raises(PermissionError):kernel.update(net,opt,raw,None,None,[target()]*8,'M0HLT_s10',ctx,1.)
    torch.manual_seed(5);warm=Tiny('D20_s10',4);control=Tiny('U100_W5_s10',4)
    control.load_state_dict({k:v for k,v in warm.state_dict().items() if k in control.state_dict()})
    emb=np.zeros((8,128),np.float32);a=optimizer(warm,c.recipe()['optimizer']);b=optimizer(control,c.recipe()['optimizer'])
    before={k:v.clone() for k,v in warm.aux.state_dict().items()}
    kernel.update(warm,a,raw,emb,None,[target()]*8,'D20_s10',ctx,2.)
    kernel.update(control,b,raw,emb,None,None,'U100_W5_s10',ctx,2.)
    for k,v in control.state_dict().items():torch.testing.assert_close(v,warm.state_dict()[k],rtol=0,atol=0)
    assert any(not torch.equal(v,warm.aux.state_dict()[k]) for k,v in before.items())


def test_ema_uses_new_draw_clock_and_balanced_train_only_membership(monkeypatch):
    tr=cache(rows=120);pt=np.repeat(np.arange(650,2600,100),6)
    monkeypatch.setattr(c,'BN_ROWS',8192)
    m=readouts.membership(tr,pt,'a'*64)
    assert len(m['indices'])==8192 and m['sampling'].startswith('fixed_balanced')
    val=deepcopy(tr);val.role='validation'
    with pytest.raises(PermissionError):readouts.membership(val,pt,'a'*64)
    net=Tiny('M0HLT_s10',4);ema=readouts.Averages();ema.update(net,2.,128);assert not ema.values
    ema.update(net,3.,128);before=ema.values['EMA05']['mod.fc.weight'].clone()
    with torch.no_grad():net.mod.fc.weight.add_(1)
    ema.update(net,3.5,300000)
    torch.testing.assert_close(ema.values['EMA05']['mod.fc.weight'],before*.5+net.mod.fc.weight*.5)


@pytest.mark.parametrize('coordinate,reco',[('D000',False),('OFFLINE',False),('U100',False),('D000',True)])
def test_new_root_reader_preserves_roles_and_complete_support(tmp_path,monkeypatch,coordinate,reco):
    import uproot
    path=tmp_path/'raw/test.root';write_root(path,[1]*492)
    record=dict(path='raw/test.root',sha256=sha256_file(path),entries=492,tree_key='tree',category='signal')
    labels=np.tile(np.arange(6),82);pt=np.tile(np.repeat(np.arange(650,2600,100),6),5)[:492]
    roles=np.r_[np.zeros(240,np.uint8),np.ones(240,np.uint8),np.full(12,2,np.uint8)]
    member=dict(labels=labels,jet_pt=pt,jet_nparticles=np.full(492,3),hlt_jet_nparticles=np.full(492,2))
    parent_arrays={}
    for role,ix in [('train',np.arange(240)),('validation',np.arange(240,480))]:
        parent_arrays[role+'_identities']=np.array([np.frombuffer(bytes.fromhex(population.donor.original.identity(record,int(i))),np.uint8) for i in ix])
        parent_arrays[role+'_labels']=labels[ix];parent_arrays[role+'_pt']=pt[ix]
    parent=(dict(content_hash='a'*64,counts=dict(train=240,validation=240)),dict(files=[record]),[member],[roles],parent_arrays,None)
    monkeypatch.setattr(c,'COUNTS',dict(train=120,validation=120))
    arrays,fields=population.describe(parent)
    p=c.artifact('POPULATION',parents=dict(parent='a'*64),**fields)
    spec=dict(root=str(tmp_path/'study'),dataset=str(tmp_path),content_hash='b'*64,population=p,maximum=3)
    original=uproot.behaviors.TBranch.HasBranches.arrays;reads=[]
    def guarded(tree,expressions,*args,**kw):
        lo,hi=kw['entry_start'],kw['entry_stop'];assert np.all(roles[lo:hi]!=2)
        if coordinate=='D000' and (not reco or roles[lo]==1):assert all(k.startswith('hlt_') for k in expressions)
        if reco and any(not k.startswith('hlt_') for k in expressions):assert np.all(roles[lo:hi]==0)
        reads.append((lo,hi));return original(tree,expressions,*args,**kw)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches,'arrays',guarded)
    inputs.prepare_node(spec,tmp_path/'node',coordinate,targets=reco,workers=1,loaded=(p,parent,arrays))
    caches,targets,m=inputs.load_node(spec,tmp_path/'node')
    assert reads and all(len(v)==120 for v in caches.values())
    for role,v in caches.items():
        assert np.array_equal(v.identities,arrays[role+'_identities'])
        assert np.all(np.diff(v.blocks[0].offsets)==(3 if coordinate=='OFFLINE' else 2))
    if reco:
        t=targets.batch([0,0,119]);assert [len(v['values']) for v in t]==[3,3,3]
        assert m['normalization']['rows']==120 and m['normalization']['tokens']==360
    else:assert targets is None


def test_serial_dag_has_only_teacher_quality_dependencies():
    spec=dict(root='/study',project='/project',content_hash='a'*64)
    rows=campaign.commands(spec,[*c.FITS,'report'])
    first=rows[0]['command'];assert not any(x.startswith('--dependency=') for x in first)
    assert '--dependency=afterany:{OFFLINE_s10}' in rows[1]['command']
    u=next(r['command'] for r in rows if r['task']=='U100_s10')
    assert '--dependency=afterok:{OFFLINE_s10},afterany:{M0HLT_s10}' in u
    for row in rows[:-1]:
        assert '--gres=gpu:geforce_gtx_1080_ti:4' in row['command'] and '--time=2880' in row['command']
    assert any(a.startswith('--dependency=afterany:') for a in rows[-1]['command'])


def test_quota_respects_existing_tail_and_never_mutates_it(tmp_path,monkeypatch):
    spec=dict(root=str(tmp_path/'new'))
    monkeypatch.setattr(storage,'other_remaining',lambda s:14*2**30)
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda p:SimpleNamespace(free=27*2**30))
    q=storage.check(spec,declared_gib=27);assert q['needed_free_bytes']==26*2**30
    with pytest.raises(OSError):storage.check(spec,declared_gib=25)
    Path(spec['root']).mkdir();(Path(spec['root'])/'own.bin').write_bytes(b'x'*1024)
    assert storage.check(spec)['needed_free_bytes']==26*2**30-1024
    monkeypatch.setenv('LUKABREC_INPUT_LAUNCHER_PID','991')
    parent=os.getppid()
    monkeypatch.setattr(resources,'host_snapshot',lambda pid=None:dict(Pss=100 if pid is None else 20,Rss=110))
    monkeypatch.setattr(resources.d,'host_usage',lambda:120)
    monkeypatch.setattr(resources.d,'memory',lambda ctx:dict(ranks=[dict(gpu_peak_bytes=0)]))
    measured=resources.memory(Context(0,1,torch.device('cpu')))
    assert measured['host_peak_bytes']==110+20*len({parent,991})


def test_storage_admission_is_explicit_execution_policy_and_old_specs_fail_closed():
    p=storage.policy();c.validate(p,'STORAGE_POLICY')
    assert p['automatic_home_rechecks'] is False and p['runtime_output_limit'] is False
    assert p['admission']=='registration_and_explicit_resume_only'
    pop=c.artifact('POPULATION',parents=dict(parent=c.parent.POPULATION_HASH))
    old=c.artifact('SPEC',population=pop)
    with pytest.raises(ValueError,match='Execution storage policy differs'):campaign.check(old,live=False)


def test_gpu_job_wrapper_does_not_recheck_home_space(tmp_path,monkeypatch):
    from hlt_classification.luka_balanced_reconstruction_hydra import worker
    spec=dict(root=str(tmp_path),project='/project',dataset='/data',content_hash='a'*64)
    node=tmp_path/'node';node.mkdir()
    monkeypatch.setattr(kernel,'configure',lambda *a:None);monkeypatch.setattr(campaign,'check',lambda *a:None)
    monkeypatch.setattr(worker,'allocation',lambda *a:'123');monkeypatch.setattr(worker,'task_receipt',lambda *a:{})
    monkeypatch.setattr(storage,'check',lambda *a,**kw:pytest.fail('No HOME recheck at GPU startup'))
    monkeypatch.setattr(worker,'scratch',lambda job:node);monkeypatch.setattr(worker,'cleanup',lambda *a:None)
    monkeypatch.setattr(population,'load',lambda *a:None);monkeypatch.setattr(inputs,'prepare_node',lambda *a,**kw:None)
    def run(*a,**kw):
        assert kw['env']['LUKABREC_INPUT_LAUNCHER_PID']==str(os.getpid())
        write_immutable_json(tmp_path/'qualification.json',c.artifact('QUALIFICATION'))
    monkeypatch.setattr(worker.subprocess,'run',run)
    assert worker.run(spec,'qualify')['task']=='qualify'


class ToyModule(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(nn.Linear(17,128),nn.Dropout(.1));self.fc=nn.Sequential(nn.Linear(128,6))
    def forward(self,features,v=None,mask=None):
        tokens=self.encoder(features.transpose(1,2));h=(tokens*mask[:,0,:,None]).sum(1)/mask.sum((1,2))[:,None]
        return self.fc(h)


class ToyNative(nn.Module):
    def __init__(self):super().__init__();self.mod=ToyModule()
    def forward(self,features,vectors,mask):return self.mod(features,v=vectors,mask=mask)


class ToyStudent(Tiny):
    def __init__(self,task,maximum):
        super().__init__(task,maximum);self.mod=ToyModule()
    def forward(self,features,vectors,mask):
        self.particle_tokens=self.mod.encoder(features.transpose(1,2));self.particle_mask=mask[:,0]
        self.jet=(self.particle_tokens*self.particle_mask[...,None]).sum(1)/self.particle_mask.sum(1)[:,None]
        self.prediction=self.predictor(self.jet) if self.capture else None
        return self.mod.fc(self.jet)


def fit_fixture(root,monkeypatch):
    monkeypatch.setattr(c,'COUNTS',dict(train=120,validation=12));monkeypatch.setattr(c,'EPOCHS',4)
    monkeypatch.setattr(c,'BN_ROWS',128);monkeypatch.setattr(c,'SEEDS',(10,))
    monkeypatch.setattr(c,'FITS',tuple(f'{a}_s10' for a in c.ARMS));monkeypatch.setitem(c.RESOURCES,'gpus',1)
    monkeypatch.setattr(model,'Student',ToyStudent);monkeypatch.setattr(model,'NativeParticleTransformer',ToyNative)
    monkeypatch.setattr(storage,'check',lambda *a,**k:pytest.fail('No automatic HOME checks during science'))
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    jobs=[dict(task=task,job=str(100+i)) for i,task in enumerate(c.FITS)]
    monkeypatch.setattr(campaign,'accepted_jobs',lambda *a,**k:jobs)
    root.mkdir(parents=True,exist_ok=True)
    spec=dict(root=str(root),content_hash='a'*64,population=dict(content_hash='b'*64),maximum=4)
    artifacts.prepare_starts(spec)
    tr=cache(coordinate='D000',rows=120);val=cache(role='validation',coordinate='D000',rows=12)
    val.identities[:,20]=1;tr.foundation_sha256=val.foundation_sha256='b'*64
    pt=np.repeat(np.arange(650,2600,100),6).astype(float)
    normal=c.artifact('TARGET_NORMALIZATION',parents=dict(population='b'*64),role='train')
    class Targets:
        normalization=normal
        def batch(self,ix):return [target(1+int(i)%4) for i in ix]
    return spec,tr,val,pt,Targets(),jobs


def run_fit(spec,task,tr,val,pt,targets,job):
    tr=deepcopy(tr);val=deepcopy(val);tr.coordinate_name=val.coordinate_name=c.cell(task)['coordinate']
    bank=banks.Bank(spec,task,tr) if c.cell(task)['teacher'] else None
    ctx=Context(0,1,torch.device('cpu'))
    return training.fit(spec,task,dict(train=tr,validation=val),targets if c.cell(task)['reconstruction'] else None,
        bank,pt,ctx,artifacts.fit_root(spec,task)/'runs'/job/'fit',job,validation_pt=np.full(len(val),650.))


def test_all_arm_publications_fresh_teachers_readouts_and_retirement(tmp_path,monkeypatch):
    spec,tr,val,pt,targets,jobs=fit_fixture(tmp_path/'study',monkeypatch)
    initial=[]
    for row in jobs:
        task,job=row['task'],row['job'];receipt=run_fit(spec,task,tr,val,pt,targets,job)
        completed=artifacts.completed(spec,task);report=completed['report']
        assert receipt==completed['receipt'] and report['epochs']==4
        initial.append(load_json(receipt['files']['initialization']['path'])['components'])
        assert not list(artifacts.fit_root(spec,task).rglob('state_*.pt'))
        assert len(list((artifacts.fit_root(spec,task)/'retired').glob('*.intent.json')))==4
        for name in c.READOUTS:
            state=torch.load(receipt['files'][name+'_checkpoint']['path'],map_location='cpu',weights_only=True)
            assert state and all(k.startswith('mod.') for k in state)
        if task.startswith(('OFFLINE','U100_s')):
            assert 'bank' in receipt['files']
    assert len({v['native'] for v in initial})==len({v['predictor'] for v in initial})==1
    result=reporting.results(spec)
    assert len(result['models'])==9 and not result['missing']
    assert result['primary']=='EMA05'
    reporting.table(spec,rejections=True)
    assert result['models']['M0HLT_s10']['EMA05']['recovery']['macro_ovr_auc'] in (None,0.)
    bad=deepcopy(tr);bad.identities=bad.identities[::-1]
    with pytest.raises(ValueError,match='order'):banks.Bank(spec,'R24_s10',bad)
    report_path=Path(artifacts.completed(spec,'R24_s10')['receipt']['files']['report']['path'])
    report_path.write_bytes(report_path.read_bytes()+b'corruption')
    with pytest.raises(ValueError):artifacts.completed(spec,'R24_s10')


def test_epoch_resume_restores_optimizer_rng_and_selected_readouts(tmp_path,monkeypatch):
    spec,tr,val,pt,targets,jobs=fit_fixture(tmp_path/'reference',monkeypatch)
    expected=run_fit(spec,'M0HLT_s10',tr,val,pt,targets,'101')
    spec,tr,val,pt,targets,jobs=fit_fixture(tmp_path/'restart',monkeypatch)
    real=artifacts.commit_epoch
    def stop(spec,task,folder,row,state,parents,job):
        if row['epoch']==2:raise OSError(28,'No space left on device')
        return real(spec,task,folder,row,state,parents,job)
    monkeypatch.setattr(artifacts,'commit_epoch',stop)
    with pytest.raises(RuntimeError,match='No space left'):run_fit(spec,'M0HLT_s10',tr,val,pt,targets,'101')
    assert len(list((artifacts.fit_root(spec,'M0HLT_s10')/'epochs').glob('*.json')))==1
    jobs.append(dict(task='M0HLT_s10',job='201'));monkeypatch.setattr(artifacts,'commit_epoch',real)
    actual=run_fit(spec,'M0HLT_s10',tr,val,pt,targets,'201')
    for readout in c.READOUTS:
        a=torch.load(expected['files'][readout+'_checkpoint']['path'],map_location='cpu',weights_only=True)
        b=torch.load(actual['files'][readout+'_checkpoint']['path'],map_location='cpu',weights_only=True)
        for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)


def test_teacher_export_failure_resumes_without_retraining(tmp_path,monkeypatch):
    spec,tr,val,pt,targets,jobs=fit_fixture(tmp_path/'study',monkeypatch)
    export=banks.export
    def fail(*a,**k):raise RuntimeError('export interrupted')
    monkeypatch.setattr(banks,'export',fail)
    with pytest.raises(RuntimeError,match='export interrupted'):run_fit(spec,'OFFLINE_s10',tr,val,pt,targets,'100')
    assert len(list(artifacts.fit_root(spec,'OFFLINE_s10').rglob('state_*.pt')))==1
    monkeypatch.setattr(artifacts,'commit_epoch',lambda *a,**k:pytest.fail('Do not repeat completed updates'))
    monkeypatch.setattr(banks,'export',export);jobs.append(dict(task='OFFLINE_s10',job='200'))
    receipt=run_fit(spec,'OFFLINE_s10',tr,val,pt,targets,'200')
    assert receipt['job_id']=='200' and not list(artifacts.fit_root(spec,'OFFLINE_s10').rglob('state_*.pt'))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual installed-Weaver CUDA acceptance')
def test_actual_weaver_six_output_paths_decoder_taps_and_native_exports(tmp_path,monkeypatch):
    from hlt_classification.luka_sixclass_balanced_oscar.model import installed_parity
    kernel.configure(1,3);tr=cache(rows=120);tr.coordinate_name='D000'
    assert installed_parity(tr,device='cuda')['passed']
    spec=dict(root=str(tmp_path),content_hash='b'*64,maximum=4,population=dict(content_hash='a'*64))
    artifacts.prepare_starts(spec);ctx=Context(0,1,torch.device('cuda'))
    class Targets:
        def batch(self,ix):return [target(1+int(i)%4) for i in ix]
    for arm in c.ARMS:
        task=arm+'_s10';t=Targets() if c.cell(task)['reconstruction'] else None
        for position in (2.,8.):
            assert resources.parity(spec,task,tr,t,None,ctx,position)['ranks'][0]['passed']
        state,_=artifacts.start(spec,task);net,opt=kernel.build(task,4,ctx,state)
        raw=tr.batch(np.arange(8));emb,q=resources.target_values(task,np.arange(8))
        kernel.update(net,opt,raw,emb,q,t.batch(np.arange(8)) if t else None,task,ctx,8.)
        net.eval();p=kernel.native(net,raw,ctx.device).detach()
        native=training.readouts_native(net.deployment_state(),ctx)
        torch.testing.assert_close(kernel.native(native,raw,ctx.device),p,rtol=0,atol=0)
        del net,opt,native
        torch.cuda.empty_cache()


def _four_rank_updates(rank,init):
    from datetime import timedelta
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=init,rank=rank,world_size=4,timeout=timedelta(minutes=5))
    ctx=Context(rank,4,torch.device('cpu'))
    try:
        for task in ('M0HLT_s10','U100_s10','U100_W5_s10','R24_s10','D20_s10','R28_s10'):
            # Unequal shards exercise sample weighting, not an average of means.
            torch.manual_seed(51);net=Tiny(task,4);reference=deepcopy(net)
            opt=optimizer(net,c.recipe()['optimizer']);ro=optimizer(reference,c.recipe()['optimizer'])
            raw=cache(rows=15).batch(np.arange(15));emb,q=resources.target_values(task,np.arange(15))
            t=[target(1+i%4) for i in range(15)] if c.cell(task)['reconstruction'] else None
            for position in (2.,8.):
                expected=resources.independent_step(reference,ro,raw,emb,q,t,task,'cpu',position)
                loss=kernel.update(net,opt,raw,emb,q,t,task,ctx,position)
                measured=resources.snapshot(net,opt,loss['total'])
                proof=resources.compare(expected,measured,resources.TOLERANCES)
                assert proof['passed'],(task,proof)
    finally:dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(),reason='Four-rank Gloo required')
def test_four_rank_weighted_gradients_optimizer_and_partial_batch(tmp_path):
    torch.multiprocessing.spawn(_four_rank_updates,args=((tmp_path/'gloo_store').as_uri(),),nprocs=4,join=True)


def test_submission_intents_duplicate_guard_dependencies_and_uncertain_ids(tmp_path,monkeypatch):
    spec=dict(root=str(tmp_path),project='/p',content_hash='a'*64)
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    monkeypatch.setattr(storage,'check',lambda *a,**kw:pytest.fail('No HOME recheck during automatic submission/release'))
    monkeypatch.setattr(campaign,'site',lambda *a:None)
    commands=[]
    def submit(args,**kw):
        assert args[0]=='sbatch';commands.append(args)
        return SimpleNamespace(returncode=0,stdout=str(100+len(commands)),stderr='')
    monkeypatch.setattr(campaign.subprocess,'run',submit)
    with pytest.raises(PermissionError):campaign.submit(spec,['prepare'],execute=True)
    jobs=campaign.submit(spec,['prepare','qualify','release'],execute=True,authorization=c.AUTH)
    assert jobs==dict(prepare='101',qualify='102',release='103')
    assert '--dependency=afterok:101' in commands[1] and '--dependency=afterok:102' in commands[2]
    with pytest.raises(FileExistsError):campaign.submit(spec,['prepare'],execute=True,authorization=c.AUTH)
    assert len(commands)==3
    def uncertain(args,**kw):return SimpleNamespace(returncode=1,stdout='',stderr='connection lost')
    monkeypatch.setattr(campaign.subprocess,'run',uncertain)
    with pytest.raises(RuntimeError,match='Uncertain'):campaign.submit(spec,['OFFLINE_s10'],execute=True,authorization=c.AUTH)
    with pytest.raises(ValueError,match='Ambiguous'):campaign.accepted_jobs(spec)


def test_worker_allocation_rejects_wrong_gpu_or_request(monkeypatch):
    from hlt_classification.luka_balanced_reconstruction_hydra import worker
    spec=dict(root='/r',project='/p',environment_dir='/e',content_hash='a'*64)
    monkeypatch.setenv('SLURM_JOB_ID','123');monkeypatch.setenv('PYTHONNOUSERSITE','1')
    monkeypatch.setenv('LD_LIBRARY_PATH',str(Path('/e')/'lib'))
    monkeypatch.setattr(campaign,'accepted_jobs',lambda *a,**kw:[dict(job='123',task='R24_s10')])
    fields=dict(JobId='123',JobState='RUNNING',Account='default',QOS='normal',Partition='gpus',JobName='lukabrec_R24_s10',
        Comment='a'*64+':R24_s10',WorkDir='/p',NumNodes='1',NumCPUs='12',NumTasks='1',TimeLimit='2-00:00:00',
        AllocTRES='cpu=12,mem=64000M,gres/gpu=4,gres/gpu:geforce_gtx_1080_ti=4',**{'CPUs/Task':'12'})
    monkeypatch.setattr(worker,'controller',lambda args:'ClusterName = hydra' if args[-1]=='config' else ' '.join(k+'='+v for k,v in fields.items()))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True);monkeypatch.setattr(torch.cuda,'device_count',lambda:4)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'NVIDIA GeForce GTX 1080 Ti')
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda i:(6,1))
    assert worker.allocation(spec,'R24_s10')=='123'
    fields['AllocTRES']='cpu=12,mem=64000M,gres/gpu=2,gres/gpu:geforce_gtx_1080_ti=2'
    with pytest.raises(PermissionError):worker.allocation(spec,'R24_s10')


def test_retirement_cannot_escape_the_fit(tmp_path):
    outside=tmp_path/'important.pt';outside.write_bytes(b'keep me')
    spec=dict(root=str(tmp_path/'campaign'),content_hash='a'*64)
    commit=dict(content_hash='b'*64,epoch=1,state=dict(path=str(outside),sha256=sha256_file(outside),bytes=7))
    with pytest.raises(PermissionError):artifacts.retire(spec,'R24_s10',commit,'newer_epoch_committed')
    commit['state']['path']=str(artifacts.fit_root(spec,'R24_s10')/'runs/../../state_01.pt')
    with pytest.raises(PermissionError):artifacts.retire(spec,'R24_s10',commit,'newer_epoch_committed')
    assert outside.read_bytes()==b'keep me'


def test_afterany_report_attempt_does_not_block_a_later_complete_report(tmp_path,monkeypatch):
    from hlt_classification.luka_balanced_reconstruction_hydra import worker
    spec=dict(root=str(tmp_path),content_hash='a'*64)
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    monkeypatch.setattr(kernel,'configure',lambda *a:None)
    job=['123'];monkeypatch.setattr(worker,'allocation',lambda *a:job[0])
    value=[c.artifact('RESULTS',parents=dict(spec=spec['content_hash']),missing=['R24_s10'])]
    monkeypatch.setattr(reporting,'results',lambda *a:value[0])
    worker.run(spec,'report')
    assert (tmp_path/'reports/123.json').exists() and not (tmp_path/'results.json').exists()
    job[0]='124';value[0]=c.artifact('RESULTS',parents=dict(spec=spec['content_hash']),missing=[])
    worker.run(spec,'report')
    assert load_json(tmp_path/'results.json')==value[0]
    assert (tmp_path/'reports/124.task.json').exists()


def test_donor_pins_match_canonical_committed_bytes():
    import hashlib
    root=Path(__file__).resolve().parents[1]
    pins=load_json(root/'src/hlt_classification/luka_balanced_reconstruction_hydra/donors.json')
    assert len(pins['files'])>100 and len(pins['commit'])==40
    for name,digest in pins['files'].items():
        # Git's canonical text uses LF; the Windows checkout may use CRLF.
        data=(root/name).read_bytes().replace(b'\r\n',b'\n')
        assert hashlib.sha256(data).hexdigest()==digest,name


def _four_cuda_parity(rank,root,init):
    import torch.distributed as dist
    from datetime import timedelta
    torch.cuda.set_device(rank);kernel.configure(4,12)
    dist.init_process_group('nccl',init_method=init,rank=rank,world_size=4,timeout=timedelta(minutes=10))
    ctx=Context(rank,4,torch.device('cuda',rank));tr=cache(rows=120);pt=np.repeat(np.arange(650,2600,100),6)
    spec=dict(root=root,content_hash='b'*64,maximum=4,population=dict(content_hash='a'*64))
    ctx.main(lambda:artifacts.prepare_starts(spec))
    class Targets:
        def batch(self,ix):return [target(1+int(i)%4) for i in ix]
    try:
        for arm in c.ARMS:
            task=arm+'_s10';t=Targets() if c.cell(task)['reconstruction'] else None
            for position in (2.,8.):resources.parity(spec,task,tr,t,None,ctx,position)
            resources.miniature(spec,task,dict(train=tr,validation=tr),t,None,pt,ctx)
    finally:dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count()<4 or not torch.distributed.is_nccl_available(),
    reason='Four real CUDA/NCCL GPUs unavailable locally; mandatory Hydra qualification')
def test_real_four_cuda_syncbn_parity_and_readouts(tmp_path):
    torch.multiprocessing.spawn(_four_cuda_parity,args=(str(tmp_path),(tmp_path/'nccl_store').as_uri()),nprocs=4,join=True)

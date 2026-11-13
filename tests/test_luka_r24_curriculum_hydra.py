"""Independent gradient reference, real export, exact DAG and four-rank qualification."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.luka_r24_curriculum_hydra import contracts as c,campaign,kernel as k,training,resources,submission,worker,reporting
from hlt_classification.luka_next_supervision_hydra import reconstruction_batched as loss, reconstruction_targets, model as original_model
from hlt_classification.luka_next_supervision_hydra.contracts import seed
from hlt_classification.luka_subclass_ddp_hydra import distributed as d
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from test_luka_r24_gpu_benchmark import TinyNative,batch
from test_luka_direct_reconstruction import fixture,compact_native


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def spec(root):
    return dict(campaign.registration(),root=str(root),project_dir=str(root/'code'),content_hash='a'*64,
        parents=dict(original='b'*64),counts=dict(train=299234,validation=74895),maximum=4,
        imported=dict(node=dict(initialization_seed=1,sampler_seed=2)),environment={},
        recovery_anchors=dict(M0HLT=.96056122,OFFLINE=.96452569))


def tiny(task):
    m=TinyNative();m.cfg=c.scientific_cfg(task);m.aux.cfg=m.cfg
    if c.cell(task)['control']:m.aux=None
    return m


def test_twenty_cells_exact_design_and_controls():
    assert len(set(c.FITS))==20 and c.FITS[:6]==('D02','D01','D18','D13','D07','D15')
    assert sum(c.cell(t)['control'] for t in c.FITS)==2
    assert {t for t in c.FITS if c.cell(t)['seed_index']}=={'D03','D04'}
    assert {t for t in c.FITS if c.cell(t)['decoder_warm']}=={'D18','D19','D20'}
    assert all(c.control(t)==('D03' if t in ('D03','D04') else 'D01') for t in c.FITS)
    assert len({tuple(c.cell(t).items()) for t in c.FITS})==20
    assert c.recipe()['embedding_weight']==5 and c.recipe()['epochs']==30
    assert campaign.registration()['world_size']==4


@pytest.mark.parametrize('task',c.FITS)
def test_schedule_boundaries_and_warm_decoder(task):
    row=c.cell(task);delay=row['delay'];ramp=row['ramp'];maximum=row['maximum']
    assert c.weights(task,delay)[0]==0
    assert c.weights(task,delay+ramp)[0]==pytest.approx(maximum)
    assert c.weights(task,delay+.5*ramp)[0]==pytest.approx(.5*maximum)
    if row['decoder_warm']:assert c.weights(task,0)==(0.,maximum)
    else:assert c.weights(task,delay)==(0.,0.)
    if row['fade']:assert c.weights(task,30)==pytest.approx((.1,.1))
    else:assert c.weights(task,30)==pytest.approx((maximum,maximum))
    for bad in (-1.,30.1,float('nan'),float('inf')):
        with pytest.raises(ValueError):c.weights(task,bad)


@pytest.mark.parametrize('task,position',[('D02',3.),('D02',4.5),('D02',7.),('D18',1.),('D18',4.5),('D20',5.),('D17',24.),('D16',7.)])
def test_chunked_vjp_matches_independent_full_graph_and_adamw(task,position):
    torch.manual_seed(4);reference=tiny(task);candidate=deepcopy(reference);b=batch(11);raw,emb,targets=b
    opt=k.optimizer(reference);other=k.optimizer(candidate);we,wd=c.weights(task,position)
    logits=k.native(reference,raw,'cpu');ratio=we/wd if wd else 0.
    # Independent autograd graph: straight-through condition scales only its
    # encoder derivative, while decoder parameter gradients retain weight wd.
    h=reference.jet.detach()+ratio*(reference.jet-reference.jet.detach())
    t=reference.particle_tokens.detach()+ratio*(reference.particle_tokens-reference.particle_tokens.detach())
    rows,_=loss.rows(reference.aux(h,t,reference.particle_mask),targets)
    ce=F.cross_entropy(logits,torch.from_numpy(raw['labels']));emb_loss=F.smooth_l1_loss(reference.prediction,torch.from_numpy(emb))
    objective=ce+5*emb_loss+wd*rows.mean();objective.backward()
    for group in opt.param_groups:group['lr']=learning_rate(position)
    opt.step();expected=k.snapshot(reference,opt,float(objective.detach()))
    actual_losses,_=k.update(candidate,other,b,task,0,1,'cpu',position)
    actual=k.snapshot(candidate,other,resources.scalar(actual_losses));comparison=k.compare(expected,actual,resources.TOLERANCES)
    assert comparison['passed'],comparison


def test_control_update_matches_original_u100_and_warmup_leaves_encoder_unchanged(tmp_path,monkeypatch,compact_native):
    old,tr,_,bank=fixture(tmp_path,monkeypatch,rows=16);cfg=spec(tmp_path);cfg.update(imported=old['imported'],maximum=12)
    targets=reconstruction_targets.Targets(old,tr,'OFFLINE');b=training.batch(tr,bank,targets,np.arange(15),1,'D18')
    result=resources.identity_and_warmup(cfg,b,torch.device('cpu'))
    assert result['decoder_changed'] and result['native_and_adamw_unchanged_by_decoder']
    for task,original in (('D01','C0'),('D03','C1'),('D02','R24')):
        actual,_=k.build(cfg,task,0,1,'cpu')
        expected=original_model.new_model(old['imported'],original)
        common={n:v for n,v in actual.state_dict().items() if not n.startswith('aux.')}
        for n,v in expected.state_dict().items():
            if not n.startswith('aux.'):torch.testing.assert_close(common[n],v,rtol=0,atol=0)
    targets.close()


@pytest.mark.parametrize('task',['D01','D02','D18','D20'])
def test_real_two_epoch_fit_export_progress_and_tamper(tmp_path,monkeypatch,compact_native,task):
    old,tr,va,bank=fixture(tmp_path,monkeypatch,rows=22);cfg=spec(tmp_path);cfg.update(imported=old['imported'],maximum=12)
    cfg.update(foundation_hash=tr.foundation_sha256,embedding_hash=bank.manifest['content_hash'])
    va.identities[:,-1]=1
    for block in va.blocks:block.identities[:,-1]=1
    targets=reconstruction_targets.Targets(old,tr,'OFFLINE');ctx=d.Context(0,1,torch.device('cpu'))
    folder=tmp_path/task/'fit'
    report=training.fit(folder,cfg,task,tr,va,bank,targets,ctx,technical=True)
    actual,arrays=training.validate_fit(folder,cfg,task,technical=True)
    assert actual==report and arrays['probabilities'].shape==(22,11)
    state=torch.load(folder/'selected.pt',weights_only=True)
    assert state and all(n.startswith('mod.') for n in state)
    with pytest.raises(ValueError):training.validate_fit(folder,cfg,task,technical=False)
    (folder/'selected.pt').write_bytes(b'corrupt')
    with pytest.raises((ValueError,RuntimeError)):training.validate_fit(folder,cfg,task,technical=True)
    targets.close()


def devices():return [dict(name='GeForce GTX 1080 Ti',compute_capability=[6,1],total_memory_bytes=11*2**30) for _ in range(4)]


def diagnostic(cfg):
    profiles={t:dict(step_samples=[.6]*12,step_seconds=float(np.mean([.6]*12)),validation_seconds=50.,probe_seconds=5.,
        load_seconds=200.,export_seconds=10.,miniature_seconds=10.,miniature='b'*64,
        memory=dict(ranks=[dict(gpu_peak_bytes=5*2**30,peak_rss_bytes=5*2**30) for _ in range(4)],host_peak_bytes=21*2**30),
        position=c.cell(t)['delay']+c.cell(t)['ramp']+.5,full_hlt_support=True,full_offline_support=True,partial11=True,
        tail_rows=cfg['counts']['train']%128,train_rows=cfg['counts']['train'],validation_rows=cfg['counts']['validation']) for t in c.FITS}
    return c.artifact('RESOURCES',parents=dict(spec=cfg['content_hash']),profiles=profiles,devices=devices(),environment={},status='complete')


def test_durable_dag_preserves_order_four_gpu_fits_and_only_afterany_report(tmp_path):
    cfg=spec(tmp_path);boot=submission.plan(cfg,'bootstrap');envelope=submission.plan(cfg,'envelope')
    assert len(boot['commands'])==2 and len(envelope['commands'])==21
    assert '--dependency=afterok:${JOB_setup}' in boot['commands'][1]['command']
    assert boot['upper_bound_commands']==envelope['commands']
    assert [r['task_id'] for r in envelope['commands'][:-1]]==list(c.FITS)
    for row in envelope['commands'][:-1]:
        assert row['dependencies']==[]
        assert '--gres=gpu:geforce_gtx_1080_ti:4' in row['command']
        assert '--cpus-per-task=12' in row['command'] and '--mem=64000M' in row['command']
    assert envelope['commands'][-1]['dependencies']==list(c.FITS)
    assert any(arg.startswith('--dependency=afterany:') for arg in envelope['commands'][-1]['command'])
    measured=submission.plan(cfg,'science',resources.accept(cfg,diagnostic(cfg)))
    assert submission.within_envelope(envelope,measured)
    measured['commands'][0]['command'].append('--gres=gpu:3')
    with pytest.raises(PermissionError):submission.within_envelope(envelope,measured)


@pytest.mark.parametrize('bad',['missing','time','host','gpu','devices','samples','nan','partial','position'])
def test_resources_fail_closed_and_never_quality_gate(tmp_path,bad):
    cfg=spec(tmp_path);v=diagnostic(cfg);assert set(resources.accept(cfg,v)['minutes'])==set(c.FITS)
    p=v['profiles']['D02']
    if bad=='missing':v['profiles'].pop('D20')
    elif bad=='time':p['step_samples']=[20.]*12;p['step_seconds']=20.
    elif bad=='host':p['memory']['host_peak_bytes']=64*2**30
    elif bad=='gpu':p['memory']['ranks'][1]['gpu_peak_bytes']=11*2**30
    elif bad=='devices':v['devices'][0]['compute_capability']=[8,6]
    elif bad=='samples':p['step_samples']=[]
    elif bad=='nan':p['validation_seconds']=float('inf')
    elif bad=='partial':p['partial11']=False
    elif bad=='position':p['position']=1.
    # Rehash a structurally valid record to exercise semantic validation.
    with pytest.raises((ValueError,RuntimeError,MemoryError)):
        v=c.artifact('RESOURCES',parents=v['parents'],**{k:x for k,x in v.items() if k not in ('contract','schema_version','content_hash','parents') and k not in c.FLAGS})
        resources.accept(cfg,v)


def test_probe_does_not_change_gradients_update_or_rng():
    b=batch(11);states=[];rng=[]
    for probe in (False,True):
        torch.manual_seed(4);m=tiny('D18');o=k.optimizer(m)
        losses,detail=k.update(m,o,b,'D18',0,1,'cpu',4.5,probe=probe)
        states.append(k.snapshot(m,o,resources.scalar(losses)));rng.append(torch.get_rng_state())
    assert torch.equal(*rng)
    assert k.compare(*states,resources.TOLERANCES)['passed']
    assert detail['diagnostics']['decoder_weight']==.3 and detail['diagnostics']['encoder_weight']==.15


def _sum_worker(rank,root,init):
    import torch.distributed as dist
    torch.set_num_threads(1);dist.init_process_group('gloo',init_method=init,rank=rank,world_size=4,timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(4);m=tiny('D20');o=k.optimizer(m)
        values,_=k.update(m,o,batch(11),'D20',rank,4,'cpu',5.)
        if rank==0:torch.save(k.snapshot(m,o,resources.scalar(values)),Path(root)/'snapshot.pt')
    finally:dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(),reason='Gloo unavailable')
def test_actual_four_process_uneven_warm_vjp_and_adamw(tmp_path):
    torch.multiprocessing.spawn(_sum_worker,args=(str(tmp_path),(tmp_path/'store').as_uri()),nprocs=4,join=True)
    torch.manual_seed(4);m=tiny('D20');o=k.optimizer(m);values,_=k.update(m,o,batch(11),'D20',0,1,'cpu',5.)
    expected=k.snapshot(m,o,resources.scalar(values));actual=torch.load(tmp_path/'snapshot.pt',weights_only=True)
    assert k.compare(expected,actual,resources.TOLERANCES)['passed']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_full_installed_weaver_cuda_warm_and_export(tmp_path,monkeypatch):
    old,tr,va,bank=fixture(tmp_path,monkeypatch,rows=16);cfg=spec(tmp_path);cfg.update(imported=old['imported'],maximum=12)
    target=reconstruction_targets.Targets(old,tr,'OFFLINE');data=training.batch(tr,bank,target,np.arange(15),1,'D18')
    result=resources.identity_and_warmup(cfg,data,torch.device('cuda'))
    assert result['factory'] and result['decoder_changed'];target.close();torch.cuda.empty_cache()


def test_cli_isolation_and_wrapper_are_explicit():
    cli=Path('scripts/luka_r24_curriculum_hydra.py').read_text()
    wrapper=Path('sbatch/run_luka_r24_curriculum_hydra.sh').read_text()
    assert "old['project_dir']" in cli and 'submodule_search_locations' in cli
    assert 'unset PYTHONPATH' in wrapper and 'PYTHONNOUSERSITE=1' in wrapper
    assert '${ENV_DIR}/lib' in wrapper and '${PROJECT_DIR}/scripts/' in wrapper


def fake_scheduler(monkeypatch):
    calls=[];reviews=[]
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    monkeypatch.setattr(submission,'scheduler_review',lambda p:reviews.append(p))
    import hlt_classification.scouting.hcwdl_exact_dag_submission as dag
    def run(command,**kw):
        calls.append(command);return SimpleNamespace(stdout=str(9000+len(calls))+'\n',returncode=0)
    monkeypatch.setattr(dag.subprocess,'run',run)
    return calls,reviews


def test_auto_release_twenty_fits_exact_ids_and_no_duplicate_submits(tmp_path,monkeypatch):
    cfg=spec(tmp_path);(tmp_path/'setup').mkdir();a=resources.accept(cfg,diagnostic(cfg))
    write_immutable_json(tmp_path/'setup/acceptance.json',a)
    calls,reviews=fake_scheduler(monkeypatch);submission.launch(cfg);plan=submission.plan(cfg,'bootstrap')
    with pytest.raises(PermissionError):submission.launch(cfg,execute=True,plan_hash='bad',authorization=c.AUTHORIZATION)
    boot=submission.launch(cfg,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert boot['jobs']==dict(setup='9001',release='9002')
    monkeypatch.setattr(submission,'accounting',lambda j:['COMPLETED','0:0'])
    monkeypatch.setattr(worker,'completed_setup',lambda s:a)
    result=submission.finish(cfg,'9002')
    assert len(result['jobs'])==21 and len(calls)==23
    assert submission.finish(cfg,'9002')==result and len(calls)==23
    assert [p['stage'] for p in reviews]==['bootstrap','envelope','science']
    assert all(not any(arg.startswith('--dependency=') for arg in cmd) for cmd in calls[2:-1])


def test_failed_setup_and_ambiguous_submission_do_not_release(tmp_path,monkeypatch):
    cfg=spec(tmp_path);calls,_=fake_scheduler(monkeypatch);submission.launch(cfg);p=submission.plan(cfg,'bootstrap')
    folder=tmp_path/'submission_bootstrap';(folder/'live_claim').mkdir()
    with pytest.raises(RuntimeError,match='ambiguous/partial'):
        submission.launch(cfg,execute=True,plan_hash=p['content_hash'],authorization=c.AUTHORIZATION)
    assert not calls
    (folder/'live_claim').rmdir()  # Empty test-owned fixture only.
    submission.launch(cfg,execute=True,plan_hash=p['content_hash'],authorization=c.AUTHORIZATION)
    monkeypatch.setattr(submission,'accounting',lambda j:['FAILED','1:0'])
    with pytest.raises(RuntimeError,match='setup failed'):submission.finish(cfg,'9002')
    assert len(calls)==2


def test_launcher_uses_four_ranks_and_rejects_duplicate_claim(tmp_path,monkeypatch):
    cfg=spec(tmp_path);calls=[]
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:{})
    monkeypatch.setattr(worker,'allocation',lambda *a:('123',devices()))
    monkeypatch.setattr(worker,'require_backend',lambda:None)
    monkeypatch.setattr(worker.reconstruction_targets,'authenticate',lambda s:None)
    monkeypatch.setattr(worker.subprocess,'run',lambda args,**kw:calls.append(args))
    worker.run(cfg,'setup')
    assert '--nproc-per-node=4' in calls[0] and '--max-restarts=0' in calls[0]
    assert 'torch.distributed.run' in calls[0] and calls[0][-1]=='setup'
    with pytest.raises(FileExistsError):worker.run(cfg,'setup')
    assert len(calls)==1


def test_controller_retries_reads_only_and_preserves_error(monkeypatch):
    calls=[]
    def run(args,**kw):
        calls.append(args);return SimpleNamespace(returncode=1,stderr='controller unavailable')
    monkeypatch.setattr(worker.subprocess,'run',run);monkeypatch.setattr(worker.time,'sleep',lambda n:None)
    with pytest.raises(RuntimeError,match='controller unavailable'):worker.controller(['show','job','-o','123'])
    assert len(calls)==3 and all(args[0]=='scontrol' for args in calls)


class PortableNative(torch.nn.Module):
    """Small graph for full30 orchestration; CUDA tests cover production SyncBN."""
    def __init__(self):
        super().__init__();self.mod=torch.nn.Module()
        self.mod.encoder=torch.nn.Linear(17,128);self.mod.classifier=torch.nn.Linear(128,11)
    def forward(self,features,vectors,mask):
        tokens=self.mod.encoder(features.transpose(1,2));h=(tokens*mask.transpose(1,2)).sum(1)/mask.sum(2)
        return self.mod.classifier(h)
    def no_weight_decay(self):return set()


class PortableStudent(PortableNative):
    def __init__(self,task):
        super().__init__();self.cfg=c.scientific_cfg(task);self.predictor=torch.nn.Linear(128,128)
        self.aux=None if c.cell(task)['control'] else original_model.reconstruction_model.Decoder(self.cfg,4)
    def forward(self,features,vectors,mask):
        self.particle_tokens=self.mod.encoder(features.transpose(1,2));self.particle_mask=mask[:,0]
        self.jet=(self.particle_tokens*mask.transpose(1,2)).sum(1)/mask.sum(2);self.ce_representation=self.jet
        self.prediction=self.predictor(self.jet);return self.mod.classifier(self.jet)
    def deployment_state(self):return {n:v for n,v in self.state_dict().items() if n.startswith('mod.')}


def portable_fixture(root):
    from test_luka_attention import cache
    cfg=spec(Path(root));train=cache('train','D000',139);val=cache('validation','D000',33)
    val.identities[:,-1]=1
    for block in val.blocks:block.identities[:,-1]=1
    cfg.update(counts=dict(train=len(train),validation=len(val)),foundation_hash=train.foundation_sha256,embedding_hash='b'*64)
    bank=SimpleNamespace(targets=np.zeros((len(train),128),np.float32),manifest={'content_hash':'b'*64})
    return cfg,train,val,bank


def _full_worker(rank,root,init):
    import torch.distributed as dist
    from hlt_classification.luka_next_supervision_hydra import reconstruction_model
    from test_luka_direct_reconstruction import target
    torch.set_num_threads(1);dist.init_process_group('gloo',init_method=init,rank=rank,world_size=4,timeout=timedelta(seconds=180))
    try:
        original_model.reconstruction_model=reconstruction_model
        def build(cfg,task,rank,world,device,**kw):
            torch.manual_seed(1);m=PortableStudent(task).to(device);return m,k.optimizer(m)
        k.build=build;training.DelphesParticleTransformer=PortableNative
        cfg,train,val,bank=portable_fixture(root)
        targets=SimpleNamespace(batch=lambda ix,epoch,cfg:([target(1+int(i)%4) for i in ix],[]))
        ctx=d.Context(rank,4,torch.device('cpu'))
        training.fit(Path(root)/'D18/fit',cfg,'D18',train,val,bank,targets,ctx)
    finally:dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(),reason='Gloo unavailable')
def test_four_process_full30_with_actual_export_and_epoch_progress(tmp_path,monkeypatch):
    torch.multiprocessing.spawn(_full_worker,args=(str(tmp_path),(tmp_path/'store').as_uri()),nprocs=4,join=True)
    cfg,_,_,_=portable_fixture(tmp_path);report,_=training.validate_fit(tmp_path/'D18/fit',cfg,'D18')
    assert len(report['history'])==30 and report['history'][2]['encoder_weight']==0
    assert report['history'][3]['encoder_weight']==pytest.approx(.1) and report['history'][5]['encoder_weight']==.3
    monkeypatch.setattr(reporting,'states',lambda s:{})
    value=reporting.progress(cfg);assert value['best']['D18']['epoch']==report['selected_epoch']
    assert value['best']['D01'] is None


def _cuda_qualification(rank,root,init,cfg,old,train,bank):
    import torch.distributed as dist
    torch.cuda.set_device(rank);k.configure(4,12)
    dist.init_process_group('nccl',init_method=init,rank=rank,world_size=4,timeout=timedelta(minutes=5))
    try:
        targets=reconstruction_targets.Targets(old,train,'OFFLINE')
        resources.qualification(cfg,train,bank,targets,d.Context(rank,4,torch.device('cuda',rank)));targets.close()
    finally:dist.destroy_process_group()


def test_real_four_cuda_syncbn_production_qualification_when_available(tmp_path,monkeypatch):
    if torch.cuda.device_count()<4 or not torch.distributed.is_nccl_available():pytest.skip('Four CUDA GPUs and NCCL required; enforced in Hydra setup')
    old,train,_,bank=fixture(tmp_path,monkeypatch,rows=22);cfg=spec(tmp_path);cfg.update(imported=old['imported'],maximum=12)
    torch.multiprocessing.spawn(_cuda_qualification,args=(str(tmp_path),(tmp_path/'store').as_uri(),cfg,old,train,bank),nprocs=4,join=True)


def test_registration_pins_all_three_sources_and_preserves_original(tmp_path,monkeypatch):
    import sys
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    oldroot=tmp_path/'original';oldroot.mkdir();benchroot=tmp_path/'benchmark';benchroot.mkdir()
    oldcode=tmp_path/'oldcode';benchcode=tmp_path/'benchcode';newcode=tmp_path/'newcode'
    oldsnap={'content_hash':'d'*64};bsnap={'content_hash':'e'*64};newsnap={'content_hash':'f'*64}
    old=dict(content_hash='b'*64,source_commit=c.DONOR,source_snapshot=oldsnap,project_dir=str(oldcode),root=str(oldroot),
        environment_dir=str(Path(sys.prefix).resolve()),reference=dict(environment={}),counts=dict(train=299234,validation=74895),
        imported=dict(node=dict(initialization_seed=1,sampler_seed=2)),target_inventory=dict(maximum=298),
        anchor=dict(evidence=dict(prepared='a'*64)),**{n:str(tmp_path/n) for n in ('bundle_root','raw_root','delta_root')})
    oldpath=oldroot/'campaign_spec.json';write_immutable_json(oldpath,old);before=oldpath.read_bytes()
    b=dict(content_hash='c'*64,source_commit=c.BENCHMARK,source_snapshot=bsnap,project_dir=str(benchcode),root=str(benchroot),original_spec=file_ref(oldpath))
    bpath=benchroot/'benchmark_spec.json';write_immutable_json(bpath,b)
    report=dict(content_hash='1'*64,measurements={'4':dict(projection=dict(estimated_hours=12.04))})
    write_immutable_json(benchroot/'benchmark_report.json',report);write_immutable_json(benchroot/'receipt.json',{'saved':True})
    monkeypatch.setattr(campaign,'benchmark',lambda *a,**kw:(b,old,report))
    monkeypatch.setattr(campaign,'source',lambda *a:newsnap)
    monkeypatch.setattr(campaign,'capture_source_snapshot',lambda p,**kw:oldsnap if str(p)==str(oldcode) else bsnap)
    monkeypatch.setattr(campaign.original_campaign,'__file__',str(oldcode/'src/hlt_classification/luka_next_supervision_hydra/campaign.py'))
    monkeypatch.setattr(campaign,'installed_environment',lambda:{})
    monkeypatch.setattr(campaign,'Bundle',lambda p:SimpleNamespace(json=lambda key:dict(recovery_anchors=dict(M0HLT=.2,OFFLINE=.9),content_hash='2'*64)))
    cfg=campaign.create(tmp_path/'newrun',bpath,newcode,'a'*40,9)
    assert campaign.check(cfg,full=True)==old and oldpath.read_bytes()==before
    assert cfg['maximum']==298 and cfg['embedding_hash']=='2'*64
    with pytest.raises(ValueError):campaign.create(tmp_path/'newrun',bpath,newcode,'a'*40,9)
    with pytest.raises(ValueError):campaign.create(oldroot/'nested',bpath,newcode,'a'*40,9)
    with pytest.raises(ValueError):campaign.create(tmp_path/'small',bpath,newcode,'a'*40,8)
    bad=deepcopy(cfg);bad['world_size']=3
    with pytest.raises(ValueError):campaign.check(bad)
    monkeypatch.setattr(campaign,'capture_source_snapshot',lambda *a,**kw:{'content_hash':'0'*64})
    with pytest.raises(ValueError,match='identity differs'):campaign.check(cfg)


def test_complete_file_inventory_and_no_donor_module_changes():
    assert all(Path(p).is_file() for p in campaign.REQUIRED)
    assert campaign.REQUIRED[2].endswith('/campaign.py')
    import ast
    for path in campaign.REQUIRED:
        if not path.endswith('.py') or path.startswith('tests/'):continue
        tree=ast.parse(Path(path).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):assert all(not a.name.startswith('Fresh_check') for a in node.names)
            if isinstance(node,ast.ImportFrom):assert not (node.module or '').startswith('Fresh_check')

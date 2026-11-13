"""Matched RNG factors, genuine model paths, mixed-lane release and resource gates."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from datetime import timedelta
import numpy as np
import pytest
import torch
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.luka_r24_diagnostic_hydra import contracts as c,campaign,kernel as k,training,resources,submission,worker,identity,original,reporting
from hlt_classification.luka_next_supervision_hydra import reconstruction_targets,model as old_model
from hlt_classification.luka_subclass_ddp_hydra import distributed as d
from test_luka_direct_reconstruction import fixture,compact_native
from test_luka_r24_gpu_benchmark import TinyNative,batch


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


def test_design_has_eight_pascal_six_single_a6000_and_paired_controls():
    assert len(c.FITS)==14 and len(c.tasks('pascal'))==8 and len(c.tasks('a6000'))==6
    assert c.FITS[:2]==('P0_R','P0_C')
    for t in c.FITS:
        assert c.control(t)==t.split('_')[0]+'_C'
        assert c.scientific_cfg(t)['seed_index']==0
        assert c.weights(t,3)==(0.,0.)
        assert c.weights(t,4.5)==((0.,0.) if c.cell(t)['control'] else (.15,.15))
        assert c.weights(t,6)==((0.,0.) if c.cell(t)['control'] else (.3,.3))
    assert campaign.hardware('a6000')['world_size']==1
    assert c.cell('OB_R')['worker']=='original' and c.cell('NB_R')['precision']=='bf16_native'


def test_independent_rng_offsets_and_initial_components(tmp_path,compact_native):
    cfg=spec(tmp_path);records={}
    for task in ('P0_R','PD_R','PO_R','PI_R','P0_C'):
        m,_=k.build(cfg,task,0,1,'cpu');before=torch.get_rng_state().clone()
        records[task]=identity.initial_record(cfg,task,m,1)
        identity.validate_initial(cfg,task,records[task],1)
        assert torch.equal(before,torch.get_rng_state())
    base=records['P0_R']
    assert records['PD_R']['components']==base['components']==records['PO_R']['components']
    assert records['PI_R']['components']['native']!=base['components']['native']
    for part in ('predictor','decoder'):assert records['PI_R']['components'][part]==base['components'][part]
    for task in ('P0_R','PD_R','PI_R'):assert records[task]['epoch1_order_hash']==base['epoch1_order_hash']
    assert records['PO_R']['epoch1_order_hash']!=base['epoch1_order_hash']
    assert records['PD_R']['seeds']['dropout_base']!=base['seeds']['dropout_base']
    for part in ('native','predictor'):assert records['P0_C']['components'][part]==base['components'][part]


@pytest.mark.parametrize('task',['P0_C','P0_R','PD_R','PO_R','PI_R','NF_R'])
def test_actual_two_epoch_fit_native_export_and_tamper(tmp_path,monkeypatch,compact_native,task):
    old,tr,va,bank=fixture(tmp_path,monkeypatch,rows=22);cfg=spec(tmp_path)
    cfg.update(imported=old['imported'],maximum=12,foundation_hash=tr.foundation_sha256,embedding_hash=bank.manifest['content_hash'])
    va.identities[:,-1]=1
    for block in va.blocks:block.identities[:,-1]=1
    targets=reconstruction_targets.Targets(old,tr,'OFFLINE');folder=tmp_path/task/'fit'
    report=training.fit(folder,cfg,task,tr,va,bank,targets,d.Context(0,1,torch.device('cpu')),technical=True)
    actual,arrays=training.validate_fit(folder,cfg,task,technical=True)
    assert actual==report and arrays['probabilities'].shape==(22,11)
    assert all(n.startswith('mod.') for n in torch.load(folder/'selected.pt',weights_only=True))
    assert report['initialization']==load_json(folder/'initialization.json')['content_hash']
    (folder/'initialization.json').write_text('{}')
    with pytest.raises((ValueError,KeyError)):training.validate_fit(folder,cfg,task,technical=True)
    targets.close()


def devices(lane):
    return [dict(name='GeForce GTX 1080 Ti' if lane=='pascal' else 'NVIDIA RTX A6000',
        compute_capability=[6,1] if lane=='pascal' else [8,6],total_memory_bytes=(11 if lane=='pascal' else 48)*2**30)
        for _ in range(campaign.hardware(lane)['world_size'])]


def diagnostic(cfg,lane):
    world=campaign.hardware(lane)['world_size']
    profiles={t:dict(step_samples=[.6]*12,step_seconds=float(np.mean([.6]*12)),validation_seconds=50.,probe_seconds=5.,
        load_seconds=200.,miniature_seconds=10.,miniature='b'*64,
        memory=dict(ranks=[dict(gpu_peak_bytes=5*2**30,peak_rss_bytes=3*2**30) for _ in range(world)],host_peak_bytes=(3*world+1)*2**30),
        full_hlt_support=True,full_offline_support=True,partial11=True,precision=c.cell(t)['precision'],worker=c.cell(t)['worker'],
        tail_rows=cfg['counts']['train']%128,train_rows=cfg['counts']['train'],validation_rows=cfg['counts']['validation']) for t in c.tasks(lane)}
    return c.artifact('RESOURCES',parents=dict(spec=cfg['content_hash']),lane=lane,profiles=profiles,devices=devices(lane),environment={},status='complete')


def test_two_independent_dags_and_no_science_dependencies(tmp_path):
    cfg=spec(tmp_path);boot=submission.plan(cfg,'bootstrap')
    assert [r['task_id'] for r in boot['commands']]==['setup_pascal','release_pascal','setup_a6000','release_a6000']
    for i,lane in enumerate(c.LANES):
        assert boot['commands'][i*2+1]['dependencies']==['setup_'+lane]
        envelope=submission.plan(cfg,'envelope_'+lane)
        measured=submission.plan(cfg,'science_'+lane,resources.accept(cfg,diagnostic(cfg,lane),lane))
        assert submission.within_envelope(envelope,measured)
        for r in measured['commands'][:-1]:
            assert r['dependencies']==[]
            assert '--gres='+campaign.hardware(lane)['execution_site']['gres'] in r['command']
        assert measured['commands'][-1]['dependencies']==list(c.tasks(lane))
        assert any(x.startswith('--dependency=afterany:') for x in measured['commands'][-1]['command'])
        measured['commands'][0]['command'].append('--gres=gpu:4')
        with pytest.raises(PermissionError):submission.within_envelope(envelope,measured)


@pytest.mark.parametrize('lane',c.LANES)
@pytest.mark.parametrize('bad',['missing','time','host','gpu','devices','nan','partial','precision'])
def test_resource_fail_closed_without_quality_gate(tmp_path,lane,bad):
    cfg=spec(tmp_path);v=diagnostic(cfg,lane);assert set(resources.accept(cfg,v,lane)['minutes'])==set(c.tasks(lane))
    p=v['profiles'][c.tasks(lane)[0]]
    if bad=='missing':v['profiles'].pop(c.tasks(lane)[-1])
    elif bad=='time':p['step_samples']=[20.]*12;p['step_seconds']=20.
    elif bad=='host':p['memory']['host_peak_bytes']=64*2**30
    elif bad=='gpu':p['memory']['ranks'][0]['gpu_peak_bytes']=50*2**30
    elif bad=='devices':v['devices'][0]['compute_capability']=[9,0]
    elif bad=='nan':p['validation_seconds']=float('inf')
    elif bad=='partial':p['partial11']=False
    elif bad=='precision':p['precision']='fp16'
    with pytest.raises((ValueError,RuntimeError,MemoryError)):
        v=c.artifact('RESOURCES',parents=v['parents'],**{k:x for k,x in v.items() if k not in ('contract','schema_version','content_hash','parents') and k not in c.FLAGS})
        resources.accept(cfg,v,lane)


def fake_scheduler(monkeypatch):
    calls=[];reviews=[]
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    monkeypatch.setattr(submission,'scheduler_review',lambda p:reviews.append(p))
    import hlt_classification.scouting.hcwdl_exact_dag_submission as dag
    def run(command,**kw):
        calls.append(command);return SimpleNamespace(stdout=str(9000+len(calls))+'\n',returncode=0)
    monkeypatch.setattr(dag.subprocess,'run',run)
    return calls,reviews


def test_independent_auto_release_and_no_duplicate_submissions(tmp_path,monkeypatch):
    cfg=spec(tmp_path);acceptances={}
    for lane in c.LANES:
        root=resources.setup_root(cfg,lane);root.mkdir();a=resources.accept(cfg,diagnostic(cfg,lane),lane)
        write_immutable_json(root/'acceptance.json',a);acceptances[lane]=a
    calls,reviews=fake_scheduler(monkeypatch);submission.launch(cfg);plan=submission.plan(cfg,'bootstrap')
    boot=submission.launch(cfg,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert boot['jobs']==dict(setup_pascal='9001',release_pascal='9002',setup_a6000='9003',release_a6000='9004')
    seen=[]
    def accounting(j):
        seen.append(j)
        return ['COMPLETED','0:0'] if j=='9001' else ['PENDING','0:0']
    monkeypatch.setattr(submission,'accounting',accounting)
    monkeypatch.setattr(worker,'completed_setup',lambda s,l:acceptances[l])
    result=submission.finish(cfg,'9002','pascal')
    assert len(result['jobs'])==9 and len(calls)==13 and seen==['9001']
    assert submission.finish(cfg,'9002','pascal')==result and len(calls)==13
    assert not (tmp_path/'submission_science_a6000').exists()
    monkeypatch.setattr(submission,'accounting',lambda j:['COMPLETED','0:0'])
    assert len(submission.finish(cfg,'9004','a6000')['jobs'])==7 and len(calls)==20


def test_failed_setup_never_releases(tmp_path,monkeypatch):
    cfg=spec(tmp_path);calls,_=fake_scheduler(monkeypatch);submission.launch(cfg);p=submission.plan(cfg,'bootstrap')
    submission.launch(cfg,execute=True,plan_hash=p['content_hash'],authorization=c.AUTHORIZATION)
    monkeypatch.setattr(submission,'accounting',lambda j:['FAILED','1:0'])
    with pytest.raises(RuntimeError,match='setup failed'):submission.finish(cfg,'9002','pascal')
    assert len(calls)==4


@pytest.mark.parametrize('lane',c.LANES)
def test_launcher_world_and_duplicate_claim(tmp_path,monkeypatch,lane):
    cfg=spec(tmp_path);calls=[]
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:{})
    monkeypatch.setattr(worker,'allocation',lambda *a:('123',devices(lane)))
    monkeypatch.setattr(worker,'require_backend',lambda:None)
    monkeypatch.setattr(worker.reconstruction_targets,'authenticate',lambda s:None)
    monkeypatch.setattr(worker.subprocess,'run',lambda args,**kw:calls.append(args))
    worker.run(cfg,'setup_'+lane)
    assert '--nproc-per-node='+str(campaign.hardware(lane)['world_size']) in calls[0]
    with pytest.raises(FileExistsError):worker.run(cfg,'setup_'+lane)


def _four_worker(rank,root,init):
    import torch.distributed as dist
    torch.set_num_threads(1);dist.init_process_group('gloo',init_method=init,rank=rank,world_size=4,timeout=timedelta(seconds=180))
    try:
        torch.manual_seed(4);m=tiny('P0_R');o=k.optimizer(m)
        for i,pos in enumerate(resources.POSITIONS):
            losses,_=k.update(m,o,batch(15 if i%2 else 16),'P0_R',rank,4,'cpu',pos)
            if rank==0:torch.save(k.snapshot(m,o,resources.scalar(losses)),Path(root)/f'step_{i}.pt')
    finally:dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(),reason='Gloo unavailable')
def test_twelve_update_four_process_uneven_transition(tmp_path):
    torch.multiprocessing.spawn(_four_worker,args=(str(tmp_path),(tmp_path/'store').as_uri()),nprocs=4,join=True)
    torch.manual_seed(4);m=tiny('P0_R');o=k.optimizer(m)
    for i,pos in enumerate(resources.POSITIONS):
        losses,_=k.update(m,o,batch(15 if i%2 else 16),'P0_R',0,1,'cpu',pos)
        actual=torch.load(tmp_path/f'step_{i}.pt',weights_only=True)
        comparison=k.compare(k.snapshot(m,o,resources.scalar(losses)),actual,resources.TOLERANCES)
        assert comparison['passed'],(i,comparison)


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),reason='BF16 CUDA required')
def test_installed_weaver_bf16_native_route_and_original_fit(tmp_path,monkeypatch):
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    old,tr,va,bank=fixture(tmp_path,monkeypatch,rows=22);cfg=spec(tmp_path)
    cfg.update(imported=old['imported'],maximum=12,foundation_hash=tr.foundation_sha256,embedding_hash=bank.manifest['content_hash'])
    oldfile=tmp_path/'original_spec.json';write_immutable_json(oldfile,old);cfg['original_spec']=file_ref(oldfile)
    cfg['parents']['original']=old['content_hash']
    for block in va.blocks:block.identities[:,-1]=1
    va.identities[:,-1]=1
    targets=reconstruction_targets.Targets(old,tr,'OFFLINE');ctx=d.Context(0,1,torch.device('cuda'))
    resources.setup_root(cfg,'a6000').mkdir()
    q=resources.qualification(cfg,tr,bank,targets,ctx,'a6000');assert q['checks']['native_BF16_donor_forward']['passed']
    for task in ('OB_C','OB_R','NB_C','NB_R','NF_R'):
        folder=tmp_path/task
        if task.startswith('OB'):
            report,b=original.fit(folder,cfg,task,tr,va,bank,targets,ctx,technical=True)
            assert original.validate_fit(folder,cfg,task,technical=True)[0]==report
        else:
            k.configure(1,6);report=training.fit(folder,cfg,task,tr,va,bank,targets,ctx,technical=True)
            assert training.validate_fit(folder,cfg,task,technical=True)[0]==report
    targets.close();torch.cuda.empty_cache()


def test_no_bf16_on_pascal_and_original_not_routed_to_new_fit(tmp_path):
    cfg=spec(tmp_path)
    assert all(c.cell(t)['precision']=='fp32' for t in c.tasks('pascal'))
    m=tiny('P0_C');m._diagnostic_precision='bf16_native'
    with pytest.raises(PermissionError):k.native(m,batch(4)[0],'cpu')


def test_cli_and_wrappers_isolate_pinned_donor():
    cli=Path('scripts/luka_r24_diagnostic_hydra.py').read_text();wrapper=Path('sbatch/run_luka_r24_diagnostic_hydra.sh').read_text()
    assert "old['project_dir']" in cli and 'submodule_search_locations' in cli
    assert 'unset PYTHONPATH' in wrapper and 'PYTHONNOUSERSITE=1' in wrapper and '${ENV_DIR}/lib' in wrapper


def test_source_inventory_and_no_mutations_of_donor_modules():
    import ast
    assert all(Path(p).is_file() for p in campaign.REQUIRED)
    assert campaign.REQUIRED[2].endswith('/campaign.py')
    # The only donor mutation is process-local assignment pool sizing, inherited
    # from the qualified benchmark. No source files or scientific functions change.
    for name in campaign.MODULES:
        path=Path('src/hlt_classification/luka_r24_diagnostic_hydra')/(name+'.py')
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                assert 'Fresh_check' not in ast.get_source_segment(path.read_text(),node)


def test_probe_neutrality_and_exact_curriculum_update(tmp_path,compact_native):
    from hlt_classification.luka_r24_curriculum_hydra import kernel as previous
    cfg=spec(tmp_path);states=[];rng=[];data=batch(11)
    # Same complete model initialization and original D02 schedule; run several
    # positions to verify the diagnostic instrumentation is observational.
    for probe in (False,True):
        torch.manual_seed(4);m=tiny('P0_R');o=k.optimizer(m)
        for pos in (2.,3.1,4.5,7.):losses,_=k.update(m,o,data,'P0_R',0,1,'cpu',pos,probe=probe)
        states.append(k.snapshot(m,o,resources.scalar(losses)));rng.append(torch.get_rng_state())
    assert torch.equal(*rng) and k.compare(*states,resources.TOLERANCES)['passed']
    torch.manual_seed(4);m=tiny('P0_R');o=previous.optimizer(m)
    for pos in (2.,3.1,4.5,7.):losses,_=previous.update(m,o,data,'D02',0,1,'cpu',pos)
    ref=previous.snapshot(m,o,resources.scalar(losses))
    assert k.compare(ref,states[0],{key:0. for key in resources.TOLERANCES})['passed']


from test_luka_r24_curriculum_hydra import PortableNative


class PortableStudent(PortableNative):
    def __init__(self,task):
        from hlt_classification.luka_next_supervision_hydra.reconstruction_model import Decoder
        super().__init__();self.cfg=c.scientific_cfg(task);self.predictor=torch.nn.Linear(128,128)
        self.aux=None if c.cell(task)['control'] else Decoder(self.cfg,4)
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
    from test_luka_direct_reconstruction import target
    torch.set_num_threads(1);dist.init_process_group('gloo',init_method=init,rank=rank,world_size=4,timeout=timedelta(seconds=180))
    try:
        def build(cfg,task,rank,world,device,**kw):
            torch.manual_seed(1);m=PortableStudent(task).to(device);return m,k.optimizer(m)
        k.build=build;training.DelphesParticleTransformer=PortableNative
        cfg,train,val,bank=portable_fixture(root)
        targets=SimpleNamespace(batch=lambda ix,epoch,cfg:([target(1+int(i)%4) for i in ix],[]))
        ctx=d.Context(rank,4,torch.device('cpu'))
        for task in ('P0_R','P0_C'):training.fit(Path(root)/task/'fit',cfg,task,train,val,bank,targets,ctx)
    finally:dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(),reason='Gloo unavailable')
def test_full30_paired_four_rank_export_and_progress(tmp_path,monkeypatch):
    torch.multiprocessing.spawn(_full_worker,args=(str(tmp_path),(tmp_path/'store').as_uri()),nprocs=4,join=True)
    cfg,_,_,_=portable_fixture(tmp_path)
    for task in ('P0_R','P0_C'):
        report,_=training.validate_fit(tmp_path/task/'fit',cfg,task)
        assert len(report['history'])==30 and report['history'][2]['encoder_weight']==0
        assert report['history'][5]['encoder_weight']==(0. if task.endswith('C') else .3)
    monkeypatch.setattr(reporting,'states',lambda s:{})
    result=reporting.progress(cfg)
    assert result['best']['P0_R'] and result['best']['P0_C'] and result['best']['OB_R'] is None


@pytest.mark.parametrize('lane',c.LANES)
def test_allocation_uses_lane_count_and_exact_typed_cards(tmp_path,monkeypatch,lane):
    cfg=spec(tmp_path);hw=campaign.hardware(lane);n=hw['world_size']
    monkeypatch.setattr(worker,'require_job',lambda *args:'123')
    fields=f'JobId=123 JobState=RUNNING Partition=gpus Account=default QOS=normal NumNodes=1 NumCPUs={hw["cpus"]} NumTasks=1 CPUs/Task={hw["cpus"]} AllocTRES=cpu={hw["cpus"]},mem={hw["memory_mb"]}M,node=1,gres/gpu={n}'
    monkeypatch.setattr(worker,'controller',lambda args:'ClusterName = hydra\n' if args==['show','config'] else fields)
    monkeypatch.setenv('PYTHONNOUSERSITE','1');monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    for key in ('SLURM_CLUSTER_NAME','SLURM_NTASKS','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE'):monkeypatch.delenv(key,raising=False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True);monkeypatch.setattr(torch.cuda,'device_count',lambda:n)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:devices(lane)[i]['name'])
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda i:tuple(devices(lane)[i]['compute_capability']))
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda i:SimpleNamespace(total_memory=devices(lane)[i]['total_memory_bytes']))
    assert worker.allocation(cfg,'setup_'+lane)==('123',devices(lane))
    monkeypatch.setattr(torch.cuda,'device_count',lambda:n+1)
    with pytest.raises(PermissionError):worker.allocation(cfg,'setup_'+lane)


def _protocol_worker(rank,root,init):
    import torch.distributed as dist
    from test_luka_attention import cache
    from test_luka_direct_reconstruction import target
    torch.set_num_threads(1);dist.init_process_group('gloo',init_method=init,rank=rank,world_size=4,timeout=timedelta(seconds=180))
    try:
        def build(cfg,task,rank,world,device,**kw):
            torch.manual_seed(1);m=PortableStudent(task).to(device);return m,k.optimizer(m)
        k.build=build;cfg=spec(Path(root));train=cache('train','D000',220)
        bank=SimpleNamespace(targets=np.zeros((len(train),128),np.float32))
        targets=SimpleNamespace(batch=lambda ix,epoch,cfg:([target(1+int(i)%4) for i in ix],[]))
        resources.qualification(cfg,train,bank,targets,d.Context(rank,4,torch.device('cpu')),'pascal')
    finally:dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(),reason='Gloo unavailable')
def test_complete_qualification_protocol_and_reference_tamper(tmp_path):
    cfg=spec(tmp_path);resources.setup_root(cfg,'pascal').mkdir()
    torch.multiprocessing.spawn(_protocol_worker,args=(str(tmp_path),(tmp_path/'store').as_uri()),nprocs=4,join=True)
    q=load_json(resources.setup_root(cfg,'pascal')/'qualification.json')
    resources.check_qualification(cfg,'pascal',q)
    assert len(q['checks'])==24 and len(q['reference_snapshots'])==24
    path=resources.setup_root(cfg,'pascal')/q['reference_snapshots'][0]['path'];path.write_bytes(b'corrupt')
    with pytest.raises(ValueError):resources.check_qualification(cfg,'pascal',q)


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
    cfg=campaign.create(tmp_path/'newrun',bpath,newcode,'a'*40,10)
    assert campaign.check(cfg,full=True)==old and oldpath.read_bytes()==before
    assert cfg['maximum']==298 and cfg['embedding_hash']=='2'*64
    with pytest.raises(ValueError):campaign.create(tmp_path/'newrun',bpath,newcode,'a'*40,10)
    with pytest.raises(ValueError):campaign.create(oldroot/'nested',bpath,newcode,'a'*40,10)
    with pytest.raises(ValueError):campaign.create(tmp_path/'small',bpath,newcode,'a'*40,9)
    bad=deepcopy(cfg);bad['hardware']['pascal']['world_size']=3
    with pytest.raises(ValueError):campaign.check(bad)
    monkeypatch.setattr(campaign,'capture_source_snapshot',lambda *a,**kw:{'content_hash':'0'*64})
    with pytest.raises(ValueError,match='identity differs'):campaign.check(cfg)

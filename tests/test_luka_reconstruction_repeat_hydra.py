"""Six scientific recipes, paired starts, production exports and automatic release."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from datetime import timedelta
import numpy as np
import pytest
import torch
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.luka_reconstruction_repeat_hydra import contracts as c,campaign,kernel as k,training,resources,submission,worker,identity,reporting
from hlt_classification.luka_next_supervision_hydra import reconstruction_targets,model as old_model
from hlt_classification.luka_subclass_ddp_hydra import distributed as d
from test_luka_direct_reconstruction import fixture,compact_native
from test_luka_r24_gpu_benchmark import TinyNative,batch
from test_luka_r24_curriculum_hydra import PortableNative


def test_exact_recipes_schedules_and_fresh_paired_seeds():
    from hlt_classification.luka_next_supervision_hydra import reconstruction_followup as old
    from hlt_classification.luka_next_supervision_hydra.contracts import setting as native_setting
    from hlt_classification.luka_r24_curriculum_hydra import contracts as curricula
    from hlt_classification.luka_next_supervision_hydra import reconstruction_training as rt
    assert len(c.FITS)==18 and c.SEED_INDICES==(4,5,6)
    assert c.FITS[:6]==('C_s4','R24_s4','R20_s4','R22_s4','R28_s4','D20_s4')
    for task in c.FITS:
        row=c.cell(task);cfg=c.scientific_cfg(task)
        # C0 is dispatched through the original general U100 control registry;
        # reconstruction_followup's unused C0 metadata is not that runtime path.
        expected=deepcopy(native_setting('C0') if row['control'] else old.setting(c.original_task(task)))
        expected['seed_index']=row['seed_index']
        assert cfg==expected
        assert c.control(task)==f'C_s{row["seed_index"]}'
        for pos in (0.,.001,1.5,3.,3.01,4.5,6.,9.,20.,30.):
            if row['recipe']=='D20':assert c.weights(task,pos)==curricula.weights('D20',pos)
            elif row['control']:assert c.weights(task,pos)==(0.,0.)
            else:
                w=rt.coefficient(cfg,pos)
                assert c.weights(task,pos)==(w,w)
    for pos in (-1.,float('nan'),31.):
        with pytest.raises(ValueError):c.weights('R24_s4',pos)
    for lanes in ([],['a6000'],['pascal','pascal']):
        with pytest.raises(ValueError):campaign.registration(lanes)


def test_same_seed_model_decoder_order_and_dropout_matching(tmp_path,compact_native):
    cfg=spec(tmp_path);records={}
    for task in c.FITS:
        model,_=k.build(cfg,task,0,1,'cpu');state=torch.get_rng_state().clone()
        records[task]=identity.initial_record(cfg,task,model,4)
        assert torch.equal(state,torch.get_rng_state())
    identity.validate_pairs(cfg,records,complete=True)
    assert len({records[f'C_s{s}']['components']['native'] for s in c.SEED_INDICES})==3
    assert len({records[f'C_s{s}']['epoch1_order_hash'] for s in c.SEED_INDICES})==3
    assert len({records[f'R24_s{s}']['components']['decoder'] for s in c.SEED_INDICES})==3
    bad=deepcopy(records);v=bad['R22_s4'];v['components']['native']='0'*64
    bad['R22_s4']=c.artifact('INITIALIZATION',parents=v['parents'],**{k:x for k,x in v.items()
        if k not in ('contract','schema_version','content_hash','parents') and k not in c.FLAGS})
    with pytest.raises(ValueError,match='pairing'):identity.validate_pairs(cfg,bad,complete=True)
    with pytest.raises(ValueError,match='coverage'):identity.validate_pairs(cfg,{},complete=True)


@pytest.mark.parametrize('task',['R24_s4','R20_s4','R22_s4','R28_s4','D20_s4'])
def test_chunked_vjp_against_independent_full_graph(task):
    from torch.nn import functional as F
    from hlt_classification.luka_next_supervision_hydra import reconstruction_batched as b
    from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
    raw,embedding,targets=batch(11);torch.manual_seed(17)
    reference=tiny(task);actual=deepcopy(reference)
    a=k.optimizer(actual);o=k.optimizer(reference)
    # Warm decoder exercises zero native reconstruction gradient, then full weight.
    for pos in (2.,4.5,20.):
        reference.train();o.zero_grad(set_to_none=True)
        for group in o.param_groups:group['lr']=learning_rate(pos)
        logits=reference(**{key:torch.as_tensor(raw[key]) for key in ('features','vectors','mask')})
        encoder,decoder=c.weights(task,pos)
        ce=F.cross_entropy(logits,torch.as_tensor(raw['labels']).long())
        base=ce+5*F.smooth_l1_loss(reference.prediction,torch.as_tensor(embedding))
        # Independent differentiable full graph with explicit condition-gradient scaling.
        ratio=encoder/decoder if decoder else 0.
        scale=lambda x:x.detach()+ratio*(x-x.detach())
        output=reference.aux(scale(reference.jet),scale(reference.particle_tokens),reference.particle_mask)
        loss,_=b.rows(output,targets)
        objective=base+decoder*loss.mean()
        objective.backward();o.step()
        losses,_=k.update(actual,a,(raw,embedding,targets),task,0,1,'cpu',pos)
        compared=k.compare(k.snapshot(reference,o,float(objective.detach())),
            k.snapshot(actual,a,resources.scalar(losses)),resources.TOLERANCES)
        assert compared['passed'],compared


def test_automatic_release_all_eighteen_and_idempotence(tmp_path,monkeypatch):
    cfg=spec(tmp_path);folder=resources.setup_root(cfg,'pascal');folder.mkdir()
    acceptance=resources.accept(cfg,diagnostic(cfg,'pascal'),'pascal')
    write_immutable_json(folder/'acceptance.json',acceptance)
    calls,reviews=fake_scheduler(monkeypatch);submission.launch(cfg)
    plan=submission.plan(cfg,'bootstrap')
    assert [r['task_id'] for r in plan['commands']]==['setup_pascal','release_pascal']
    boot=submission.launch(cfg,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert boot['jobs']==dict(setup_pascal='9001',release_pascal='9002')
    assert {p['stage'] for p in reviews}=={'bootstrap','envelope_pascal'}
    monkeypatch.setattr(submission,'accounting',lambda j:['COMPLETED','0:0'])
    monkeypatch.setattr(worker,'completed_setup',lambda s,l:acceptance)
    result=submission.finish(cfg,'9002','pascal')
    assert set(result['jobs'])==set(c.FITS)|{'report_pascal'} and len(calls)==21
    assert submission.finish(cfg,'9002','pascal')==result and len(calls)==21
    science=submission.plan(cfg,'science_pascal',acceptance)
    assert [r['task_id'] for r in science['commands'][:-1]]==list(c.FITS)
    assert all(r['dependencies']==[] for r in science['commands'][:-1])
    assert all('--gres=gpu:geforce_gtx_1080_ti:4' in r['command'] for r in science['commands'][:-1])
    assert science['commands'][-1]['dependencies']==list(c.FITS)
    assert '--dependency=afterany:' in ' '.join(science['commands'][-1]['command'])
    assert submission.within_envelope(submission.plan(cfg,'envelope_pascal'),science)
    science['commands'][0]['command'].append('--gres=gpu:8')
    with pytest.raises(PermissionError):submission.within_envelope(submission.plan(cfg,'envelope_pascal'),science)


def test_failed_setup_and_partial_submission_do_not_release(tmp_path,monkeypatch):
    cfg=spec(tmp_path);calls,_=fake_scheduler(monkeypatch);submission.launch(cfg)
    p=submission.plan(cfg,'bootstrap')
    with pytest.raises(PermissionError):submission.launch(cfg,execute=True,plan_hash='0'*64,authorization=c.AUTHORIZATION)
    submission.launch(cfg,execute=True,plan_hash=p['content_hash'],authorization=c.AUTHORIZATION)
    monkeypatch.setattr(submission,'accounting',lambda j:['FAILED','1:0'])
    with pytest.raises(RuntimeError,match='setup failed'):submission.finish(cfg,'9002','pascal')
    assert len(calls)==2
    root=tmp_path/'submission_science_pascal';root.mkdir();(root/'live_claim').mkdir()
    monkeypatch.setattr(submission,'accounting',lambda j:['COMPLETED','0:0'])
    monkeypatch.setattr(worker,'completed_setup',lambda s,l:resources.accept(cfg,diagnostic(cfg,l),l))
    with pytest.raises(RuntimeError,match='Ambiguous partial'):submission.finish(cfg,'9002','pascal')
    assert len(calls)==2


def test_stability_and_partial_seed_coverage():
    anchors=dict(M0HLT=.9,OFFLINE=1.)
    def row(e,r):return dict(epoch=e,validation=dict(macro_ovr_auc=.9+.001*r))
    rows=[row(e,54. if e in (21,22,23,25,26) else 50.) for e in range(1,31)]
    s=reporting.stability(rows,anchors)
    assert s['complete'] and s['mean_recovery']==pytest.approx(52.)
    assert s['epochs_at_threshold']==5 and s['longest_streak']==3
    partial=reporting.stability(rows[:24],anchors)
    assert not partial['complete'] and partial['mean_recovery'] is None and partial['observed_epochs']==4
    values=[dict(task=f'R24_s{i}',recovery=r,delta=delta,late=s) for i,r,delta in ((4,55.,.001),(5,49.,None))]
    summary=reporting.summarize(values)['R24']
    assert summary['completed_seeds']==2 and not summary['complete'] and summary['matched_pairs']==1
    assert summary['all_three_selected_at_threshold'] is None and summary['minimum_recovery']==49.
    assert reporting.summarize([])['C']['mean_recovery'] is None


@pytest.mark.skipif(not torch.distributed.is_gloo_available(),reason='Gloo unavailable')
def test_full30_paired_four_rank_export_reporting_and_tamper(tmp_path,monkeypatch):
    torch.multiprocessing.spawn(_full_worker,args=(str(tmp_path),(tmp_path/'store').as_uri()),nprocs=4,join=True)
    cfg,_,_,_=portable_fixture(tmp_path)
    for task in ('R24_s4','C_s4'):
        report,_=training.validate_fit(tmp_path/task/'fit',cfg,task)
        assert len(report['history'])==30 and report['history'][2]['encoder_weight']==0
        assert report['history'][5]['encoder_weight']==(0. if task=='C_s4' else .3)
        (tmp_path/task/'receipt.json').write_text('{}')
    monkeypatch.setattr(reporting,'states',lambda s:{})
    monkeypatch.setattr(worker,'receipt',lambda *args:None)
    result=reporting.results(cfg);progress=reporting.progress(cfg)
    assert len(result['rows'])==2 and len(result['pending'])==16
    assert result['summary']['R24']['completed_seeds']==1 and not result['summary']['R24']['complete']
    assert progress['best']['R24_s4'] and progress['best']['R20_s4'] is None
    path=tmp_path/'R24_s4/fit/epoch-01.json';path.write_text('{}')
    with pytest.raises((ValueError,KeyError)):reporting.results(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Installed-Weaver CUDA required')
def test_installed_weaver_cuda_two_depth_memory_and_native_exports(tmp_path,monkeypatch):
    old,tr,va,bank=fixture(tmp_path,monkeypatch,rows=22);cfg=spec(tmp_path)
    cfg.update(imported=old['imported'],maximum=12,foundation_hash=tr.foundation_sha256,embedding_hash=bank.manifest['content_hash'])
    va.identities[:,-1]=1
    for block in va.blocks:block.identities[:,-1]=1
    targets=reconstruction_targets.Targets(old,tr,'OFFLINE');k.configure(1,3)
    # No compact model fixture: execute installed Weaver's eight native blocks.
    ctx=d.Context(0,1,torch.device('cuda'));token_counts={}
    for task in ('R24_s4','R28_s4'):
        m,_=k.build(cfg,task,0,1,'cuda',parity=True);m.capture=True
        k.native(m,tr.batch(np.arange(11)),ctx.device)
        token_counts[task]=m.particle_tokens.shape[1]
        assert m.particle_mask.shape[1]==token_counts[task]
        del m;torch.cuda.empty_cache()
    assert token_counts['R28_s4']==2*token_counts['R24_s4']
    for task in c.QUALIFICATION_TASKS:
        training.fit(tmp_path/task/'fit',cfg,task,tr,va,bank,targets,ctx,technical=True)
        training.validate_fit(tmp_path/task/'fit',cfg,task,technical=True)
    targets.close();torch.cuda.empty_cache()


def _cuda_qualification(rank,root,init,cfg,old,train,bank):
    import torch.distributed as dist
    torch.cuda.set_device(rank);k.configure(4,12)
    dist.init_process_group('nccl',init_method=init,rank=rank,world_size=4,timeout=timedelta(minutes=5))
    try:
        targets=reconstruction_targets.Targets(old,train,'OFFLINE')
        resources.qualification(cfg,train,bank,targets,d.Context(rank,4,torch.device('cuda',rank)),'pascal')
        targets.close()
    finally:dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count()<4 or not torch.distributed.is_nccl_available(),reason='Four CUDA GPUs/NCCL required; mandatory Hydra setup')
def test_real_four_cuda_syncbn_qualification(tmp_path,monkeypatch):
    old,train,_,bank=fixture(tmp_path,monkeypatch,rows=88);cfg=spec(tmp_path)
    cfg.update(imported=old['imported'],maximum=12)
    resources.setup_root(cfg,'pascal').mkdir()
    torch.multiprocessing.spawn(_cuda_qualification,args=(str(tmp_path),(tmp_path/'store').as_uri(),cfg,old,train,bank),nprocs=4,join=True)



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


def fake_scheduler(monkeypatch):
    calls=[];reviews=[]
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    monkeypatch.setattr(submission,'scheduler_review',lambda p:reviews.append(p))
    import hlt_classification.scouting.hcwdl_exact_dag_submission as dag
    def run(command,**kw):
        calls.append(command);return SimpleNamespace(stdout=str(9000+len(calls))+'\n',returncode=0)
    monkeypatch.setattr(dag.subprocess,'run',run)
    return calls,reviews


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


def test_cli_and_wrappers_isolate_pinned_donor():
    cli=Path('scripts/luka_reconstruction_repeat_hydra.py').read_text();wrapper=Path('sbatch/run_luka_reconstruction_repeat_hydra.sh').read_text()
    assert "old['project_dir']" in cli and 'submodule_search_locations' in cli
    assert 'unset PYTHONPATH' in wrapper and 'PYTHONNOUSERSITE=1' in wrapper and '${ENV_DIR}/lib' in wrapper


def test_source_inventory_and_no_mutations_of_donor_modules():
    import ast
    assert all(Path(p).is_file() for p in campaign.REQUIRED)
    assert campaign.REQUIRED[2].endswith('/campaign.py')
    # The only donor mutation is process-local assignment pool sizing, inherited
    # from the qualified benchmark. No source files or scientific functions change.
    for name in campaign.MODULES:
        path=Path('src/hlt_classification/luka_reconstruction_repeat_hydra')/(name+'.py')
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                assert 'Fresh_check' not in ast.get_source_segment(path.read_text(),node)


def test_same_start_restore_includes_bn_statistics_and_optimizer_history():
    torch.manual_seed(17)
    m=torch.nn.Sequential(torch.nn.Linear(4,6),torch.nn.BatchNorm1d(6),torch.nn.Linear(6,3))
    o=torch.optim.AdamW(m.parameters(),lr=.003)
    x=torch.randn(15,4)
    def step(model,opt):
        model.train();opt.zero_grad(set_to_none=True)
        loss=model(x).square().mean();loss.backward();opt.step()
        return k.snapshot(model,opt,float(loss.detach()))
    for _ in range(3):step(m,o)
    before=k.qualification_state(m,o)
    expected=step(m,o)
    assert int(before['model']['1.num_batches_tracked'])==3
    assert all(float(s['step'])==3 for s in before['optimizer']['state'].values())
    assert int(m[1].num_batches_tracked)==4
    other=deepcopy(m);other_opt=torch.optim.AdamW(other.parameters(),lr=.1)
    # The checkpoint's buffers and moments must not alias the moving model.
    for _ in range(2):step(other,other_opt)
    k.restore_qualification_state(other,other_opt,before)
    actual=step(other,other_opt)
    assert k.compare(expected,actual,{key:0. for key in resources.TOLERANCES})['passed']
    assert int(before['model']['1.num_batches_tracked'])==3
    assert all(float(s['step'])==3 for s in before['optimizer']['state'].values())


@pytest.mark.parametrize('task',list(c.QUALIFICATION_TASKS))
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
            torch.manual_seed(c.seeds(cfg,task)['native_initialization']);m=PortableStudent(task).to(device)
            torch.manual_seed(c.seeds(cfg,task)['dropout_base']+rank);return m,k.optimizer(m)
        k.build=build;training.DelphesParticleTransformer=PortableNative
        cfg,train,val,bank=portable_fixture(root)
        targets=SimpleNamespace(batch=lambda ix,epoch,cfg:([target(1+int(i)%4) for i in ix],[]))
        ctx=d.Context(rank,4,torch.device('cpu'))
        for task in ('R24_s4','C_s4'):training.fit(Path(root)/task/'fit',cfg,task,train,val,bank,targets,ctx)
    finally:dist.destroy_process_group()


def _protocol_worker(rank,root,init):
    import torch.distributed as dist
    from test_luka_attention import cache
    from test_luka_direct_reconstruction import target
    torch.set_num_threads(1);dist.init_process_group('gloo',init_method=init,rank=rank,world_size=4,timeout=timedelta(seconds=180))
    try:
        def build(cfg,task,rank,world,device,**kw):
            torch.manual_seed(c.seeds(cfg,task)['native_initialization']);m=PortableStudent(task).to(device)
            torch.manual_seed(c.seeds(cfg,task)['dropout_base']+rank);return m,k.optimizer(m)
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
    assert q['schema_version']==1 and q['protocol']==c.QUALIFICATION_PROTOCOL
    ref=torch.load(resources.setup_root(cfg,'pascal')/q['reference_snapshots'][1]['path'],weights_only=True)
    assert set(ref)=={'before','after','probabilities'}
    assert any(float(s['step'])==1 for s in ref['before']['optimizer']['state'].values())
    path=resources.setup_root(cfg,'pascal')/q['reference_snapshots'][0]['path'];path.write_bytes(b'corrupt')
    with pytest.raises(ValueError):resources.check_qualification(cfg,'pascal',q)


@pytest.mark.parametrize('lanes',[c.LANES])
def test_registration_pins_all_three_sources_and_preserves_original(tmp_path,monkeypatch,lanes):
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
    cfg=campaign.create(tmp_path/'newrun',bpath,newcode,'a'*40,13,lanes=lanes)
    assert campaign.check(cfg,full=True)==old and oldpath.read_bytes()==before
    assert cfg['enabled_lanes']==list(lanes) and cfg['schema_version']==1
    assert cfg['maximum']==298 and cfg['embedding_hash']=='2'*64
    with pytest.raises(ValueError):campaign.create(tmp_path/'newrun',bpath,newcode,'a'*40,13)
    with pytest.raises(ValueError):campaign.create(oldroot/'nested',bpath,newcode,'a'*40,13)
    with pytest.raises(ValueError):campaign.create(tmp_path/'small',bpath,newcode,'a'*40,12)
    bad=deepcopy(cfg);bad['hardware']['pascal']['world_size']=3
    with pytest.raises(ValueError):campaign.check(bad)
    monkeypatch.setattr(campaign,'capture_source_snapshot',lambda *a,**kw:{'content_hash':'0'*64})
    with pytest.raises(ValueError,match='identity differs'):campaign.check(cfg)

"""Objective isolation, paired starts, readouts and exact twelve-job release."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np
import pytest
import torch
from torch import nn
from hlt_classification.luka_ema_control_hydra import contracts as c, campaign, kernel as k
from hlt_classification.luka_ema_control_hydra import optimization as opt, resources, identity, readouts, training
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash
from hlt_classification.luka_subclass_ddp_hydra.distributed import Context
from hlt_classification.luka_next_supervision_hydra import reconstruction_model as rm
from test_luka_direct_reconstruction import fixture, compact_native, target
from test_luka_r24_gpu_benchmark import batch


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def spec_at(root):
    return dict(campaign.registration(),root=str(root),project_dir=str(root/'code'),content_hash='a'*64,
        parents=dict(original='b'*64),counts=dict(train=299234,validation=74895),maximum=4,
        imported=dict(node=dict(initialization_seed=1,sampler_seed=2)),environment={},output_budget_bytes=20*2**30,
        recovery_anchors=dict(M0HLT=.96056122,OFFLINE=.96452569))


class Tiny(nn.Module):
    def __init__(self,task):
        super().__init__();self.cfg=c.scientific_cfg(task);self.mod=nn.Module()
        self.mod.encoder=nn.Linear(6,128);self.mod.fc=nn.Linear(128,11)
        self.predictor=nn.Linear(128,128);self.aux=None if c.cell(task)['control'] else rm.Decoder(self.cfg,4)
    def forward(self,features,vectors,mask):
        self.particle_tokens=self.mod.encoder(features.transpose(1,2));self.particle_mask=mask[:,0]
        self.jet=self.particle_tokens.mean(1);self.prediction=self.predictor(self.jet)
        return self.mod.fc(self.jet)
    def no_weight_decay(self):return set()
    def deployment_state(self):return {n:v.detach().cpu().clone() for n,v in self.state_dict().items() if n.startswith('mod.')}


def test_matrix_objectives_and_prospective_primary():
    assert c.FITS==tuple(f'{r}_s{s}' for s in (7,8,9) for r in ('CE','U','R24'))
    assert c.recipe()['primary_readout']=='EMA05'
    assert c.recipe()['statistical_scope']=='three_previously_used_development_seeds'
    assert c.control('R24_s9')=='U_s9' and c.control('U_s9')=='CE_s9'
    assert c.weights('R24_s7',3)==(0.,0.) and c.weights('R24_s7',4.5)==(.15,.15)
    assert c.weights('R24_s7',6)==(.3,.3)
    assert c.cell('CE_s7')['embedding_weight']==0
    assert campaign.hardware('a6000')['world_size']==1
    with pytest.raises(ValueError):campaign.registration(['pascal'])


@pytest.mark.parametrize('task',c.QUALIFICATION_TASKS)
def test_chunked_production_matches_independent_full_graph(task):
    torch.manual_seed(41);ref=Tiny(task);actual=deepcopy(ref)
    k.disable_dropout(ref);k.disable_dropout(actual)
    ro=opt.optimizer(ref);ao=opt.optimizer(actual);ao.stability_state.update(decoder_rng={},decoder_seed=99)
    raw,emb,targets=batch(11)
    pieces=[(raw,None if task.startswith('CE_') else emb,None if c.cell(task)['control'] else targets)]
    for pos in (3.,3.01,4.5,9.,20.):
        expected=resources.reference_update(ref,ro,pieces,task,'cpu',pos)
        actual_loss,_=k.update_group(actual,ao,pieces,task,0,1,'cpu',pos)
        result=k.compare(k.snapshot(ref,ro,resources.scalar(expected)),k.snapshot(actual,ao,resources.scalar(actual_loss)),resources.TOLERANCES)
        assert result['passed'],result
    if task.startswith('CE_'):
        assert all(p.grad is None and p not in ao.state for p in actual.predictor.parameters())
        assert actual_loss['embedding']==actual_loss['weighted_embedding']==actual_loss['reconstruction']==0


def test_ce_rejects_privileged_targets_and_loader_does_not_touch_them():
    class Forbidden:
        def __getattr__(self,n):raise AssertionError('Privileged target touched: '+n)
    raw,emb,targets=batch(11);train=SimpleNamespace(batch=lambda ix:raw)
    pieces=training.batch(train,Forbidden(),Forbidden(),np.arange(11),1,'CE_s7')
    assert pieces[1:]==(None,None)
    model=Tiny('CE_s7');optimizer=opt.optimizer(model)
    with pytest.raises(PermissionError):k.update(model,optimizer,(raw,emb,None),'CE_s7',0,1,'cpu',5.)


def test_canonical_starts_all_nine_matched_and_fail_closed(tmp_path,compact_native):
    spec=spec_at(tmp_path);(tmp_path/'setup_a6000').mkdir();identity.prepare(spec)
    records={}
    for task in c.FITS:
        model,_=k.build(spec,task,0,1,'cpu');rng=torch.get_rng_state().clone()
        records[task]=identity.initial_record(spec,task,model,1)
        assert torch.equal(rng,torch.get_rng_state())
        identity.check_start(spec,task,records[task]);del model
    identity.validate_pairs(spec,records,complete=True)
    assert len({records[f'CE_s{s}']['components']['native'] for s in c.SEED_INDICES})==3
    assert len({records[f'CE_s{s}']['epoch1_order_hash'] for s in c.SEED_INDICES})==3
    # Simulated changed host construction is overridden by the frozen untrained bytes.
    with torch.no_grad():
        model,_=k.build(spec,'U_s7',0,1,'cpu',fresh=True)
        next(model.mod.parameters()).add_(.01)
    identity.load_start(spec,'U_s7',model)
    assert identity.initial_record(spec,'U_s7',model,1)['components']==records['U_s7']['components']
    (tmp_path/'setup_a6000/initial_s7.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError):k.build(spec,'CE_s7',0,1,'cpu')


def mini_fixture(tmp_path,monkeypatch):
    old,tr,va,bank=fixture(tmp_path,monkeypatch,rows=22)
    va.identities[:,-1]=1
    for block in va.blocks:block.identities[:,-1]=1
    spec=spec_at(tmp_path);spec.update(maximum=12,counts=dict(train=len(tr),validation=len(va)),imported=old['imported'],
        foundation_hash=tr.foundation_sha256,embedding_hash=bank.manifest['content_hash'])
    (tmp_path/'setup_a6000').mkdir();identity.prepare(spec)
    return spec,old,tr,va,bank


@pytest.mark.parametrize('task',c.QUALIFICATION_TASKS)
def test_two_epoch_production_and_four_exports(tmp_path,monkeypatch,compact_native,task):
    from hlt_classification.luka_next_supervision_hydra import reconstruction_targets
    spec,old,tr,va,bank=mini_fixture(tmp_path,monkeypatch)
    targets=reconstruction_targets.Targets(old,tr,'OFFLINE') if task.startswith('R24') else None
    try:
        report=training.fit(tmp_path/task,spec,task,tr,va,None if task.startswith('CE') else bank,targets,
            Context(0,1,torch.device('cpu')),technical=True)
        training.validate_fit(tmp_path/task,spec,task,technical=True)
        assert report['primary_readout']=='EMA05' and report['validation']==report['selected_readouts']['EMA05']['validation']
        assert report['target_hash']==(None if task.startswith('CE') else spec['embedding_hash'])
        (tmp_path/task/'EMA05/deployment.json').write_text('{}')
        with pytest.raises(ValueError):training.validate_fit(tmp_path/task,spec,task,technical=True)
    finally:
        if targets is not None:targets.close()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Installed Weaver CUDA required')
def test_installed_weaver_cuda_three_objectives_qualification_and_exports(tmp_path,monkeypatch):
    from hlt_classification.luka_next_supervision_hydra import reconstruction_targets
    k.configure(1,3);spec,old,tr,va,bank=mini_fixture(tmp_path,monkeypatch)
    targets=reconstruction_targets.Targets(old,tr,'OFFLINE');ctx=Context(0,1,torch.device('cuda'))
    try:
        resources.qualification(spec,tr,bank,targets,ctx,'a6000')
        for task in c.QUALIFICATION_TASKS:
            training.fit(tmp_path/task,spec,task,tr,va,None if task.startswith('CE') else bank,
                None if c.cell(task)['control'] else targets,ctx,technical=True)
            training.validate_fit(tmp_path/task,spec,task,technical=True)
    finally:targets.close();torch.cuda.empty_cache()


def test_ema_clock_only_native_and_bn_copy_isolation(tmp_path,compact_native):
    from test_luka_attention import cache
    from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
    spec=spec_at(tmp_path);(tmp_path/'setup_a6000').mkdir();identity.prepare(spec)
    m,_=k.build(spec,'U_s7',0,1,'cpu');avg=readouts.Averages()
    avg.update(m,2.99,128);assert not avg.values
    avg.update(m,3.,128);name=next(iter(avg.values['EMA05']));initial=avg.values['EMA05'][name].clone()
    with torch.no_grad():dict(m.named_parameters())[name].add_(1.)
    avg.update(m,3.5,c.TRAIN_ROWS//2)
    torch.testing.assert_close(avg.values['EMA05'][name],initial+.5)
    assert all(n.startswith('mod.') for values in avg.values.values() for n in values)
    tr,va=cache('train','D000',22),cache('validation','D000',22)
    ctx=Context(0,1,torch.device('cpu'));manifest=readouts.membership(spec,tr,technical=True)
    buffers={n:v.clone() for n,v in m.named_buffers()};rng=torch.get_rng_state().clone()
    outputs=readouts.evaluate_all(m,avg,tr,va,manifest,ctx,training.evaluate,k.native,DelphesParticleTransformer)
    assert set(outputs)==set(c.READOUTS) and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,buffers[n]) for n,v in m.named_buffers())
    with pytest.raises(PermissionError):readouts.membership(spec,va,technical=True)


def resource_record(spec):
    memory=dict(ranks=[dict(gpu_peak_bytes=8*2**30,peak_rss_bytes=4*2**30)],host_peak_bytes=5*2**30)
    devices=[dict(name='NVIDIA RTX A6000',compute_capability=[8,6],total_memory_bytes=48*2**30)]
    profiles={r:dict(phase_samples={p:[.6]*6 for p in ('4.5','20.0')},step_seconds=float(np.mean([.6]*6)),
        validation_seconds=80.,probe_seconds=2.,load_seconds=50.,memory=memory,
        full_hlt_support=True,full_offline_support=True,partial11=True,tail_rows=spec['counts']['train']%128,
        effective_batch=128,train_rows=spec['counts']['train'],validation_rows=spec['counts']['validation'],readouts=list(c.READOUTS)) for r in c.RECIPES}
    return c.artifact('RESOURCES',parents=dict(spec=spec['content_hash'],bn_membership='b'*64,initial_states='c'*64),
        lane='a6000',profiles=profiles,miniatures={t:dict(content_hash='d'*64,seconds=15.,fit_memory=memory) for t in c.FITS},
        devices=devices,environment={},status='complete',storage=dict(setup_bytes=2**30,scientific_bytes=5*2**30,
            required_bytes=math.ceil(1.25*6*2**30)+2**30))


def fake_scheduler(monkeypatch):
    from hlt_classification.luka_ema_control_hydra import submission
    from hlt_classification.scouting import hcwdl_exact_dag_submission as dag
    calls=[];reviews=[]
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    monkeypatch.setattr(submission,'scheduler_review',lambda plan:reviews.append(plan))
    def run(command,**kwargs):calls.append(command);return SimpleNamespace(stdout=str(9000+len(calls))+'\n',returncode=0)
    monkeypatch.setattr(dag.subprocess,'run',run)
    return calls,reviews


def test_exact_twelve_jobs_automatic_release_and_no_duplicates(tmp_path,monkeypatch):
    from hlt_classification.luka_ema_control_hydra import submission,worker
    spec=spec_at(tmp_path);folder=resources.setup_root(spec,'a6000');folder.mkdir()
    accepted=resources.accept(spec,resource_record(spec),'a6000');write_immutable_json(folder/'acceptance.json',accepted)
    calls,reviews=fake_scheduler(monkeypatch);submission.launch(spec);plan=submission.plan(spec,'bootstrap')
    boot=submission.launch(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert boot['jobs']==dict(setup_a6000='9001',release_a6000='9002')
    assert {v['stage'] for v in reviews}=={'bootstrap','envelope_a6000'}
    monkeypatch.setattr(submission,'accounting',lambda _:['COMPLETED','0:0'])
    monkeypatch.setattr(worker,'completed_setup',lambda *args:accepted)
    result=submission.finish(spec,'9002','a6000');assert len(calls)==12
    assert submission.finish(spec,'9002','a6000')==result and len(calls)==12
    science=submission.plan(spec,'science_a6000',accepted)
    assert [r['task_id'] for r in science['commands'][:-1]]==list(c.FITS)
    assert all(not r['dependencies'] and '--gres=gpu:nvidia_rtx_a6000:1' in r['command'] for r in science['commands'][:-1])
    assert science['commands'][-1]['dependencies']==list(c.FITS)
    assert '--dependency=afterany:' in ' '.join(science['commands'][-1]['command'])
    assert submission.within_envelope(submission.plan(spec,'envelope_a6000'),science)
    science['commands'][0]['command'].append('--gres=gpu:8')
    with pytest.raises(PermissionError):submission.within_envelope(submission.plan(spec,'envelope_a6000'),science)


def test_failed_setup_and_partial_release_stop(tmp_path,monkeypatch):
    from hlt_classification.luka_ema_control_hydra import submission,worker
    spec=spec_at(tmp_path);calls,_=fake_scheduler(monkeypatch);submission.launch(spec)
    plan=submission.plan(spec,'bootstrap')
    with pytest.raises(PermissionError):submission.launch(spec,execute=True,plan_hash='0'*64,authorization=c.AUTHORIZATION)
    submission.launch(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    monkeypatch.setattr(submission,'accounting',lambda _:['FAILED','1:0'])
    with pytest.raises(RuntimeError,match='setup failed'):submission.finish(spec,'9002','a6000')
    assert len(calls)==2
    monkeypatch.setattr(submission,'accounting',lambda _:['COMPLETED','0:0'])
    monkeypatch.setattr(worker,'completed_setup',lambda *args:resources.accept(spec,resource_record(spec),'a6000'))
    folder=tmp_path/'submission_science_a6000';folder.mkdir();(folder/'live_claim').mkdir()
    with pytest.raises(RuntimeError,match='Ambiguous partial'):submission.finish(spec,'9002','a6000')
    assert len(calls)==2


@pytest.mark.parametrize('failure',['missing','time','memory','storage','readout','phase','nan','gpu'])
def test_resource_failures_stop_release(tmp_path,failure):
    spec=spec_at(tmp_path);value=resource_record(spec)
    assert len(resources.accept(spec,value,'a6000')['minutes'])==9
    if failure=='missing':value['miniatures'].pop('CE_s9')
    if failure=='time':value['profiles']['R24']['phase_samples']={p:[10.]*6 for p in ('4.5','20.0')};value['profiles']['R24']['step_seconds']=10.
    if failure=='memory':value['profiles']['R24']['memory']['ranks'][0]['gpu_peak_bytes']=49*2**30
    if failure=='storage':spec['output_budget_bytes']=2**30
    if failure=='readout':value['profiles']['R24']['readouts']=['RAW']
    if failure=='phase':value['profiles']['R24']['phase_samples'].pop('4.5')
    if failure=='nan':value['profiles']['R24']['validation_seconds']=float('inf')
    if failure=='gpu':value['devices'][0]['name']='NVIDIA GeForce GTX 1080 Ti'
    with pytest.raises((ValueError,RuntimeError,MemoryError,OSError)):
        resources.accept(spec,with_content_hash(value),'a6000')


class PortableNative(nn.Module):
    """Small graph for complete orchestration; real Weaver is tested separately."""
    def __init__(self):
        super().__init__();self.mod=nn.Module();self.mod.encoder=nn.Linear(17,128);self.mod.fc=nn.Linear(128,11)
    def forward(self,features,vectors,mask):
        tokens=self.mod.encoder(features.transpose(1,2));h=(tokens*mask.transpose(1,2)).sum(1)/mask.sum(2)
        return self.mod.fc(h)
    def no_weight_decay(self):return set()


class PortableStudent(PortableNative):
    def __init__(self,task):
        super().__init__();self.cfg={};self.predictor=nn.Linear(128,128);self.aux=None
    def forward(self,features,vectors,mask):
        self.particle_tokens=self.mod.encoder(features.transpose(1,2));self.particle_mask=mask[:,0]
        self.jet=(self.particle_tokens*mask.transpose(1,2)).sum(1)/mask.sum(2)
        self.prediction=self.predictor(self.jet);return self.mod.fc(self.jet)
    def deployment_state(self):return {n:v.detach().cpu().clone() for n,v in self.state_dict().items() if n.startswith('mod.')}


def test_full30_partial_batch_exports_primary_deltas_and_tamper(tmp_path,monkeypatch):
    from test_luka_attention import cache
    from hlt_classification.luka_ema_control_hydra import reporting,worker
    spec=spec_at(tmp_path);tr,va=cache('train','D000',139),cache('validation','D000',33)
    va.identities[:,-1]=1
    for block in va.blocks:block.identities[:,-1]=1
    spec.update(counts=dict(train=len(tr),validation=len(va)),foundation_hash=tr.foundation_sha256,embedding_hash='b'*64)
    bank=SimpleNamespace(targets=np.zeros((len(tr),128),np.float32),manifest={'content_hash':'b'*64})
    targets=SimpleNamespace(batch=lambda ix,epoch,cfg:([target(1+int(i)%4) for i in ix],[]))
    monkeypatch.setattr(k,'Student',PortableStudent)
    monkeypatch.setattr(k.reconstruction_model,'attach',lambda m,n:setattr(m,'aux',rm.Decoder(m.cfg,n)))
    monkeypatch.setattr(training,'DelphesParticleTransformer',PortableNative)
    monkeypatch.setattr(readouts,'load_membership',lambda spec,train,**kw:readouts.membership(spec,train,technical=True))
    (tmp_path/'setup_a6000').mkdir();identity.prepare(spec)
    for task in c.QUALIFICATION_TASKS:
        training.fit(tmp_path/task/'fit',spec,task,tr,va,None if task.startswith('CE') else bank,
            None if c.cell(task)['control'] else targets,Context(0,1,torch.device('cpu')))
        report,_=training.validate_fit(tmp_path/task/'fit',spec,task)
        assert len(report['history'])==30 and set(report['history'][0]['readouts'])=={'RAW','RAW_BN'}
        assert set(report['history'][2]['readouts'])==set(c.READOUTS)
        assert all(r['optimizer_updates_this_epoch']==2 for r in report['history'])
        (tmp_path/task/'receipt.json').write_text('{}')
    monkeypatch.setattr(reporting,'states',lambda spec:{})
    monkeypatch.setattr(worker,'receipt',lambda *a:None)
    result=reporting.results(spec,readout='all');progress=reporting.progress(spec,readout='all')
    assert result['primary']==progress['primary']=='EMA05' and len(result['pending'])==6
    assert len(result['rows'])==12
    for name in c.READOUTS:
        rows={r['task']:r for r in result['rows'] if r['readout']==name}
        assert rows['U_s7']['delta']==pytest.approx(rows['U_s7']['auc']-rows['CE_s7']['auc'])
        assert rows['R24_s7']['delta']==pytest.approx(rows['R24_s7']['auc']-rows['U_s7']['auc'])
        assert result['summary'][name]['R24']['completed_seeds']==1
        assert not result['summary'][name]['R24']['complete']
        assert result['summary'][name]['R24']['late_mean']['count']==1
    assert progress['best']['EMA05']['R24_s7'] and progress['best']['EMA05']['R24_s8'] is None
    (tmp_path/'R24_s7/fit/epoch-01.json').write_text('{}')
    with pytest.raises((ValueError,KeyError)):reporting.results(spec)


def test_registration_lineage_low_space_reader_and_immutable_parent(tmp_path,monkeypatch,compact_native):
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    oldroot=tmp_path/'old';oldroot.mkdir();parentroot=tmp_path/'parent';(parentroot/'setup_pascal').mkdir(parents=True)
    old=dict(root=str(oldroot),content_hash='b'*64,
        **{n:str(tmp_path/n) for n in ('project_dir','environment_dir','bundle_root','raw_root','delta_root')})
    original=oldroot/'campaign_spec.json';write_immutable_json(original,old)
    p=dict(spec_at(parentroot),source_commit=c.PARENT_COMMIT,original_spec=file_ref(original),
        environment_dir=old['environment_dir'],embedding_hash='d'*64,foundation_hash='e'*64)
    parentpath=parentroot/'campaign_spec.json';write_immutable_json(parentpath,p)
    write_immutable_json(parentroot/'setup_pascal/bn_membership.json',dict(old=True))
    before=parentpath.read_bytes();snap=dict(content_hash='f'*64)
    monkeypatch.setattr(campaign,'parent',lambda *a,**kw:(p,old))
    monkeypatch.setattr(campaign,'source',lambda *a:snap)
    monkeypatch.setattr(campaign.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*2**30))
    value=campaign.create(tmp_path/'new',parentpath,tmp_path/'newcode','1'*40,30)
    assert campaign.check(value,full=True)==old and parentpath.read_bytes()==before
    assert value['reused_membership']['path'].endswith('bn_membership.json')
    with pytest.raises(ValueError):campaign.create(tmp_path/'new',parentpath,tmp_path/'newcode','1'*40,30)
    with pytest.raises(ValueError):campaign.create(oldroot/'nested',parentpath,tmp_path/'newcode','1'*40,30)
    with pytest.raises(ValueError):campaign.create(tmp_path/'small',parentpath,tmp_path/'newcode','1'*40,1)
    bad=deepcopy(value);bad['hardware']['a6000']['world_size']=4
    with pytest.raises(ValueError):campaign.check(bad)
    with monkeypatch.context() as low:
        low.setattr(campaign.shutil,'disk_usage',lambda p:SimpleNamespace(free=0))
        with pytest.raises(OSError):campaign.check(value)
        assert campaign.check(value,check_storage=False)==old
        with pytest.raises(ValueError):campaign.check(bad,check_storage=False)
    monkeypatch.setattr(campaign,'source',lambda *a:dict(content_hash='0'*64))
    with pytest.raises(ValueError,match='identity differs'):campaign.check(value,check_storage=False)


def test_summary_missing_pairs_are_not_counted_and_no_best_readout_selection():
    from hlt_classification.luka_ema_control_hydra import reporting
    late=dict(complete=True,mean_recovery=40.)
    rows=[dict(task=f'R24_s{s}',readout='EMA05',auc=.96,recovery=r,delta=d,delta_ce=None,delta_u=d,late=late)
        for s,r,d in ((7,55.,.001),(8,49.,None))]
    summary=reporting.summarize(rows)
    assert summary['EMA05']['R24']['completed_seeds']==2
    assert summary['EMA05']['R24']['paired_delta']['count']==1
    assert summary['EMA05']['R24']['recovery']['sample_sd']==pytest.approx(math.sqrt(18))
    assert summary['RAW']['R24']['completed_seeds']==0
    assert reporting.stability([dict(epoch=21,validation=dict(macro_ovr_auc=.96))],dict(M0HLT=.9,OFFLINE=1.))['mean_recovery'] is None


def test_setup_membership_is_exactly_the_parent_train_subset(tmp_path):
    from test_luka_attention import cache
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    train=cache('train','D000',22);spec=spec_at(tmp_path);value=readouts.membership(spec,train,technical=True)
    path=tmp_path/'parent_bn.json';write_immutable_json(path,value);spec['reused_membership']=file_ref(path)
    readouts.check_parent_membership(spec,value)
    other=deepcopy(value);other['indices']=list(reversed(other['indices']))
    with pytest.raises(ValueError):readouts.check_parent_membership(spec,other)


def test_source_inventory_and_cli_wrappers():
    import ast
    assert all(Path(p).is_file() for p in campaign.REQUIRED)
    assert campaign.REQUIRED[2].endswith('/campaign.py')
    for name in campaign.MODULES:
        source=(Path('src/hlt_classification/luka_ema_control_hydra')/(name+'.py')).read_text()
        ast.parse(source);assert 'Fresh_check' not in source
    cli=Path('scripts/luka_ema_control_hydra.py').read_text()
    assert "default='EMA05'" in cli and "check_storage=False" in cli
    wrapper=Path('sbatch/run_luka_ema_control_hydra.sh').read_text()
    assert 'PYTHONNOUSERSITE=1' in wrapper and 'CUBLAS_WORKSPACE_CONFIG=:4096:8' in wrapper
    assert '${ENV_DIR}/lib' in wrapper and '${PROJECT_DIR}/scripts/' in wrapper


def test_single_gpu_launcher_preserves_existing_claim(tmp_path,monkeypatch):
    from hlt_classification.luka_ema_control_hydra import worker
    spec=spec_at(tmp_path);calls=[]
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:{})
    monkeypatch.setattr(worker,'allocation',lambda *a:('123',resource_record(spec)['devices']))
    monkeypatch.setattr(worker,'require_backend',lambda:None)
    monkeypatch.setattr(worker.reconstruction_targets,'authenticate',lambda spec:None)
    monkeypatch.setattr(worker.subprocess,'run',lambda args,**kw:calls.append(args))
    worker.run(spec,'setup_a6000')
    assert '--nproc-per-node=1' in calls[0] and '--max-restarts=0' in calls[0]
    with pytest.raises(FileExistsError):worker.run(spec,'setup_a6000')


def test_allocation_requires_single_typed_a6000(tmp_path,monkeypatch):
    from hlt_classification.luka_ema_control_hydra import worker
    spec=spec_at(tmp_path);devices=resource_record(spec)['devices']
    monkeypatch.setattr(worker,'require_job',lambda *a:'123')
    fields='JobId=123 JobState=RUNNING Partition=gpus Account=default QOS=normal NumNodes=1 NumCPUs=6 NumTasks=1 CPUs/Task=6 AllocTRES=cpu=6,mem=32000M,node=1,gres/gpu=1,gres/gpu:nvidia_rtx_a6000=1'
    monkeypatch.setattr(worker,'controller',lambda args:'ClusterName = hydra\n' if args==['show','config'] else fields)
    monkeypatch.setenv('PYTHONNOUSERSITE','1');monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    for key in ('SLURM_CLUSTER_NAME','SLURM_NTASKS','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE'):monkeypatch.delenv(key,raising=False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True);monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:devices[i]['name'])
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda i:tuple(devices[i]['compute_capability']))
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda i:SimpleNamespace(total_memory=devices[i]['total_memory_bytes']))
    assert worker.allocation(spec,'setup_a6000')==('123',devices)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:4)
    with pytest.raises(PermissionError):worker.allocation(spec,'setup_a6000')


def test_ce_worker_does_not_open_privileged_banks(tmp_path,monkeypatch):
    from hlt_classification.luka_ema_control_hydra import worker
    spec=spec_at(tmp_path);task='CE_s7';(tmp_path/task).mkdir()
    claim=tmp_path/'claims'/task;claim.mkdir(parents=True)
    write_immutable_json(claim/'intent.json',c.artifact('RUN_CLAIM',parents=dict(spec=spec['content_hash']),task=task,job_id='123'))
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:{})
    monkeypatch.setattr(worker,'allocation',lambda *a:('123',resource_record(spec)['devices']))
    monkeypatch.setattr(torch.cuda,'set_device',lambda _:None)
    monkeypatch.setattr(k,'configure',lambda *a:None)
    monkeypatch.setattr(worker,'require_backend',lambda:None)
    monkeypatch.setattr(worker.inputs,'hlt',lambda old,role:object())
    def forbidden(*a,**kw):raise AssertionError('CE opened a privileged target bank')
    monkeypatch.setattr(worker.inputs,'embedding_bank',forbidden)
    monkeypatch.setattr(worker.reconstruction_targets,'Targets',forbidden)
    calls=[]
    def fit(folder,spec,task,tr,va,bank,targets,ctx,**kw):
        assert bank is targets is None;calls.append(task)
        return dict(content_hash='b'*64,fit_memory=resource_record(spec)['miniatures'][task]['fit_memory'])
    monkeypatch.setattr(resources,'miniature',fit)
    monkeypatch.setattr(training,'fit',fit)
    for name,value in dict(RANK='0',LOCAL_RANK='0',WORLD_SIZE='1',LOCAL_WORLD_SIZE='1',GROUP_RANK='0').items():monkeypatch.setenv(name,value)
    worker.rank(spec,task)
    assert calls==[task,task]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA full-support decoder qualification required')
def test_complete_298_slot_cuda_chunked_decoder_and_empty_targets():
    torch.manual_seed(52);task='R24_s7';ref=Tiny(task)
    ref.aux=rm.Decoder(c.scientific_cfg(task),298);ref=ref.cuda();actual=deepcopy(ref)
    k.configure(1,3);k.disable_dropout(ref);k.disable_dropout(actual)
    ro=opt.optimizer(ref);ao=opt.optimizer(actual);ao.stability_state.update(decoder_rng={},decoder_seed=99)
    raw,emb,_=batch(11)
    targets=[target(n) for n in (0,1,170,298,4,2,7,80,298,3,0)]
    pieces=[(raw,emb,targets)]
    expected=resources.reference_update(ref,ro,pieces,task,'cuda',4.5)
    actual_loss,_=k.update_group(actual,ao,pieces,task,0,1,'cuda',4.5)
    result=k.compare(k.snapshot(ref,ro,resources.scalar(expected)),k.snapshot(actual,ao,resources.scalar(actual_loss)),resources.TOLERANCES)
    assert result['passed'],result
    del ref,actual,ro,ao;torch.cuda.empty_cache()


def test_r24_zero_weight_matches_u100_native_rng_buffers_and_updates(tmp_path,monkeypatch,compact_native):
    from hlt_classification.luka_next_supervision_hydra import reconstruction_targets
    spec,old,tr,va,bank=mini_fixture(tmp_path,monkeypatch);targets=reconstruction_targets.Targets(old,tr,'OFFLINE')
    u,uo=k.build(spec,'U_s7',0,1,'cpu');r,ro=k.build(spec,'R24_s7',0,1,'cpu')
    ru=torch.get_rng_state().clone();rr=ru.clone();ix=np.arange(len(tr))
    try:
        for pos in (1.,2.,3.):
            torch.set_rng_state(ru);k.update(u,uo,training.batch(tr,bank,None,ix,1,'U_s7'),'U_s7',0,1,'cpu',pos);ru=torch.get_rng_state().clone()
            torch.set_rng_state(rr);k.update(r,ro,training.batch(tr,bank,targets,ix,1,'R24_s7'),'R24_s7',0,1,'cpu',pos);rr=torch.get_rng_state().clone()
            assert torch.equal(ru,rr)
            for name,value in u.deployment_state().items():torch.testing.assert_close(value,r.deployment_state()[name],atol=1e-7,rtol=1e-6)
        for (un,up),(rn,rp) in zip(u.named_parameters(),r.named_parameters()):
            assert un==rn
            for name in ('exp_avg','exp_avg_sq','step'):torch.testing.assert_close(uo.state[up][name],ro.state[rp][name],atol=1e-7,rtol=1e-6)
    finally:targets.close()

"""Numerical semantics, independent references and immutable automatic release."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from hlt_classification.luka_reconstruction_stability_hydra import contracts as c, kernel as k, optimization as opt, transport
from hlt_classification.luka_next_supervision_hydra import reconstruction_model as rm, reconstruction_batched as hard
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from test_luka_r24_gpu_benchmark import batch
from test_luka_direct_reconstruction import fixture, compact_native, target


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class Tiny(nn.Module):
    def __init__(self,task):
        super().__init__();self.cfg=c.scientific_cfg(task);self.mod=nn.Module()
        self.mod.encoder=nn.Linear(6,128);self.mod.fc=nn.Linear(128,11)
        self.predictor=nn.Linear(128,128);self.aux=None if c.cell(task)['control'] else rm.Decoder(self.cfg,4)
    def forward(self,features,vectors,mask):
        self.particle_tokens=self.mod.encoder(features.transpose(1,2));self.particle_mask=mask[:,0]
        self.jet=self.particle_tokens.mean(1);self.ce_representation=self.jet
        self.prediction=self.predictor(self.jet);return self.mod.fc(self.jet)
    def no_weight_decay(self):return set()
    def deployment_state(self):return {n:v.detach().cpu().clone() for n,v in self.state_dict().items() if n.startswith('mod.')}


def test_exact_matrix_clock_controls_and_order():
    assert len(c.FITS)==len(set(c.FITS))==42 and c.SEED_INDICES==(7,8,9)
    assert c.FITS[:6]==('C128_s7','R24_s7','C128_s8','R24_s8','C128_s9','R24_s9')
    assert sum(c.cell(t)['control'] for t in c.FITS)==15
    assert c.control('R_L25_s8')=='C_L25_s8' and c.control('R512_s9')=='C512_s9'
    assert c.control('PROJ_s7')=='C128_s7'
    for factor,steps in ((1,2338),(2,1169),(4,585)):
        assert math.ceil(299234/(128*factor))==steps
    assert c.encoder_multiplier('R_L25_s7',3)==1
    assert c.encoder_multiplier('R_L25_s7',3.5)==.625
    assert c.encoder_multiplier('R_L25_s7',4)==.25
    assert c.weights('R24_s9',3)==(0.,0.) and c.weights('R24_s9',6)==(.3,.3)


@pytest.mark.parametrize('task',['C128_s7','R24_s7','R_L25_s7','C256_s7','R256_s7','C512_s7','R512_s7'])
def test_microbatch_accumulation_matches_independent_full_graph(task):
    torch.manual_seed(31);ref=Tiny(task);actual=deepcopy(ref);ro=opt.optimizer(ref);ao=opt.optimizer(actual)
    k.disable_dropout(ref);k.disable_dropout(actual)
    batches=[batch(n) for n in ([11,7,15,5][:c.cell(task)['accumulation']])]
    for position in (3.,4.5,20.):
        ro.zero_grad(set_to_none=True);opt.set_lr(ro,task,position,learning_rate(position));total=sum(len(b[0]['labels']) for b in batches)
        for raw,emb,targets in batches:
            logits=k.native(ref,raw,'cpu');loss=F.cross_entropy(logits,torch.from_numpy(raw['labels']))+5*F.smooth_l1_loss(ref.prediction,torch.from_numpy(emb))
            if ref.aux is not None:
                scores=ref.aux(ref.jet,ref.particle_tokens,ref.particle_mask)
                values,_=hard.rows(scores,targets);loss=loss+c.weights(task,position)[0]*values.mean()
            (len(raw['labels'])/total*loss).backward()
        ro.step();k.update_group(actual,ao,batches,task,0,1,'cpu',position)
        for a,b in zip(ref.parameters(),actual.parameters()):torch.testing.assert_close(a,b,rtol=1e-4,atol=2e-6)
        for a,b in zip(ref.parameters(),actual.parameters()):
            for name in ('exp_avg','exp_avg_sq','step'):torch.testing.assert_close(ro.state[a][name],ao.state[b][name],rtol=1e-4,atol=2e-7)


def test_global_gradient_cap_projection_and_zero_norm():
    b=[torch.tensor([3.,0.])];a=[torch.tensor([-4.,8.])]
    g,d=opt.transform(b,a,.3,'PROJ',{},128)
    torch.testing.assert_close(g[0],torch.tensor([3.,2.4]));assert d['projected']
    g,d=opt.transform(b,a,.3,'G30',{},128)
    assert d['capped'] and d['injected_to_base_norm']<=.3+1e-8
    torch.testing.assert_close((g[0]-b[0]).norm(),torch.tensor(.9))
    g,d=opt.transform([torch.zeros(2)],a,.3,'G30',{},128)
    assert torch.equal(g[0],torch.zeros(2))


@pytest.mark.parametrize('task',['G30_s7','PROJ_s7'])
def test_parameter_gradient_transform_against_independent_graph(task):
    torch.manual_seed(51);ref=Tiny(task);actual=deepcopy(ref);k.disable_dropout(ref);k.disable_dropout(actual)
    ro=opt.optimizer(ref);ao=opt.optimizer(actual);raw,emb,targets=batch(11);position=4.5;w=c.weights(task,position)[0]
    logits=k.native(ref,raw,'cpu');base=F.cross_entropy(logits,torch.from_numpy(raw['labels']))+5*F.smooth_l1_loss(ref.prediction,torch.from_numpy(emb))
    values,_=hard.rows(ref.aux(ref.jet,ref.particle_tokens,ref.particle_mask),targets)
    names,ps=zip(*ref.named_parameters());bg=torch.autograd.grad(base,ps,retain_graph=True,allow_unused=True)
    ag=torch.autograd.grad(values.mean(),ps,allow_unused=True)
    bi=[torch.zeros_like(p) if g is None else g for p,g in zip(ps,bg)];ai=[torch.zeros_like(p) if g is None else g for p,g in zip(ps,ag)]
    shared=[i for i,n in enumerate(names) if opt.owner(n)=='encoder']
    transformed,_=opt.transform([bi[i] for i in shared],[ai[i] for i in shared],w,c.cell(task)['gradient_rule'],{},11)
    for p,b,a in zip(ps,bi,ai):p.grad=b+w*a
    for i,g in zip(shared,transformed):ps[i].grad=g
    opt.set_lr(ro,task,position,learning_rate(position));ro.step()
    k.update(actual,ao,(raw,emb,targets),task,0,1,'cpu',position,probe=True)
    for a,b in zip(ref.parameters(),actual.parameters()):torch.testing.assert_close(a,b,rtol=1e-4,atol=2e-6)


def test_decoder_only_steps_do_not_advance_native_adam_or_rng():
    torch.manual_seed(71);m=Tiny('DEC2_s7');o=opt.optimizer(m);k.disable_dropout(m)
    state=torch.get_rng_state().clone();_,details=k.update(m,o,batch(),'DEC2_s7',0,1,'cpu',4.5)
    assert torch.equal(state,torch.get_rng_state())
    assert details['diagnostics']['extra_decoder_steps']==2
    for n,p in m.named_parameters():assert int(o.state[p]['step'])==(3 if opt.owner(n)=='decoder' else 1)
    # At the boundary, the same ownership groups reduce exactly to ordinary R24.
    torch.manual_seed(71);a=Tiny('DEC2_s7');b=deepcopy(a);k.disable_dropout(a);k.disable_dropout(b)
    oa,ob=opt.optimizer(a),opt.optimizer(b)
    k.update(a,oa,batch(),'DEC2_s7',0,1,'cpu',3.)
    k.update(b,ob,batch(),'R24_s7',0,1,'cpu',3.)
    for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_transport_mass_gap_envelope_gradient_and_hard_limit():
    cost=torch.tensor([[.1,1.,.4],[1.,.1,.3],[.6,.5,.1]],dtype=torch.float64,requires_grad=True)
    mass=torch.ones(3,dtype=torch.float64);tau=.08
    plan,info=transport.solve(cost,mass,tau)
    torch.testing.assert_close(plan.sum(0),mass,atol=1e-4,rtol=0)
    torch.testing.assert_close(plan.sum(1),mass,atol=1e-4,rtol=0)
    assert info['normalized_gap']<=1e-4
    loss=(plan*cost).sum()+tau*info['entropy'];g=torch.autograd.grad(loss,cost)[0]
    eps=1e-5
    for i,j in ((0,0),(0,1),(2,2)):
        perturb=torch.zeros_like(cost);perturb[i,j]=eps
        hi=transport.solve(cost+perturb,mass,tau)[1]['primal'];lo=transport.solve(cost-perturb,mass,tau)[1]['primal']
        assert g[i,j]==pytest.approx((hi-lo)/(2*eps),abs=1e-4)
    cold,record=transport.solve(cost,mass,.001)
    assert float((cold*cost).sum())==pytest.approx(.3,abs=1e-5)
    with pytest.raises(RuntimeError,match='no hard fallback'):
        transport.solve(torch.tensor([[0.,0.],[0.,2.]]),torch.ones(2),.001,iterations=1)


def test_soft_empty_full_and_padded_masses():
    m=Tiny('SOFT_s7');raw,emb,targets=batch(3);targets=[target(0),target(4),target(2)]
    k.native(m,raw,'cpu');scores=m.aux(m.jet,m.particle_tokens,m.particle_mask)
    matrices,masses,_=transport.costs(scores,targets)
    assert matrices[0] is None and masses[1].tolist()==[1.]*4 and masses[2].tolist()==[1.,1.,2.]
    values,diag=transport.rows(scores,targets,1.)
    values.mean().backward();assert torch.isfinite(values).all() and diag['transport_gap_max']<=1e-4
    hard_empty,_=hard.rows({n:v[:1] for n,v in scores.items()},targets[:1])
    torch.testing.assert_close(values[:1].float(),hard_empty)


def test_ema_example_clock_and_training_buffer_isolation():
    from hlt_classification.luka_reconstruction_stability_hydra.readouts import Averages
    m=Tiny('R24_s7');ema=Averages();ema.update(m,2.99,128);assert ema.values=={}
    ema.update(m,3.,128);initial=ema.values['EMA05']['mod.encoder.weight'].clone()
    with torch.no_grad():m.mod.encoder.weight.add_(1)
    ema.update(m,3.5,c.TRAIN_ROWS//2)
    decay=2**(-(c.TRAIN_ROWS//2)/(.5*c.TRAIN_ROWS))
    torch.testing.assert_close(ema.values['EMA05']['mod.encoder.weight'],initial+1-decay)
    assert all(n.startswith('mod.') for values in ema.values.values() for n in values)


@pytest.mark.parametrize('task',['C128_s7','R24_s7','G30_s7','PROJ_s7','DEC2_s7','R512_s7'])
def test_actual_production_two_epoch_fit_and_four_native_exports(tmp_path,monkeypatch,compact_native,task):
    from hlt_classification.luka_reconstruction_stability_hydra import training,campaign
    from hlt_classification.luka_next_supervision_hydra import reconstruction_targets
    from hlt_classification.luka_subclass_ddp_hydra import distributed as d
    old,tr,va,bank=fixture(tmp_path,monkeypatch,rows=22)
    va.identities[:,-1]=1
    for block in va.blocks:block.identities[:,-1]=1
    spec=dict(campaign.registration(),root=str(tmp_path),content_hash='a'*64,parents=dict(original='b'*64),
        maximum=12,counts=dict(train=len(tr),validation=len(va)),imported=old['imported'],
        foundation_hash=tr.foundation_sha256,embedding_hash=bank.manifest['content_hash'],
        recovery_anchors=dict(M0HLT=.96056122,OFFLINE=.96452569))
    targets=reconstruction_targets.Targets(old,tr,'OFFLINE');ctx=d.Context(0,1,torch.device('cpu'));k.configure(1,3)
    try:
        report=training.fit(tmp_path/task,spec,task,tr,va,bank,targets,ctx,technical=True)
        assert set(report['selected_readouts'])==set(c.READOUTS)
        training.validate_fit(tmp_path/task,spec,task,technical=True)
        (tmp_path/task/'EMA05/deployment.json').write_text('{}')
        with pytest.raises(ValueError):training.validate_fit(tmp_path/task,spec,task,technical=True)
    finally:targets.close()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Installed Weaver CUDA required')
def test_installed_weaver_cuda_all_fourteen_paths_and_exports(tmp_path,monkeypatch):
    from hlt_classification.luka_reconstruction_stability_hydra import training,campaign,calibration,readouts
    from hlt_classification.luka_next_supervision_hydra import reconstruction_targets
    from hlt_classification.luka_subclass_ddp_hydra import distributed as d
    old,tr,va,bank=fixture(tmp_path,monkeypatch,rows=22)
    va.identities[:,-1]=1
    for block in va.blocks:block.identities[:,-1]=1
    spec=dict(campaign.registration(),root=str(tmp_path),content_hash='a'*64,parents=dict(original='b'*64),
        maximum=12,counts=dict(train=len(tr),validation=len(va)),imported=old['imported'],
        foundation_hash=tr.foundation_sha256,embedding_hash=bank.manifest['content_hash'],
        recovery_anchors=dict(M0HLT=.96056122,OFFLINE=.96452569))
    # This is a synthetic single-device numerical miniature, not the full-data
    # 512-jet calibration or four-card qualification performed by Hydra setup.
    monkeypatch.setattr(calibration,'load',lambda spec:dict(scale=.1))
    targets=reconstruction_targets.Targets(old,tr,'OFFLINE');ctx=d.Context(0,1,torch.device('cuda'));k.configure(1,3)
    try:
        for task in c.QUALIFICATION_TASKS:
            training.fit(tmp_path/task,spec,task,tr,va,bank,targets,ctx,technical=True)
            training.validate_fit(tmp_path/task,spec,task,technical=True)
    finally:targets.close();torch.cuda.empty_cache()


def _four_rank_numerics(rank,init):
    import torch.distributed as dist
    from datetime import timedelta
    from hlt_classification.luka_subclass_ddp_hydra.distributed import Context
    from hlt_classification.luka_reconstruction_stability_hydra import resources
    torch.set_num_threads(1);dist.init_process_group('gloo',init_method=init,rank=rank,world_size=4,timeout=timedelta(minutes=5))
    ctx=Context(rank,4,torch.device('cpu'))
    try:
        for task in ('G30_s7','PROJ_s7','R512_s7','DEC2_s7','SOFT_s7'):
            torch.manual_seed(91);model=Tiny(task);k.disable_dropout(model);optimizer=opt.optimizer(model)
            optimizer.stability_state['soft_scale']=1.
            pieces=[batch(n) for n in (16,15,12,11)[:c.cell(task)['accumulation']]]
            def serial():
                ref=deepcopy(model);ro=opt.optimizer(ref);ro.stability_state['soft_scale']=1.
                loss,_=k.update_group(ref,ro,pieces,task,0,1,'cpu',4.5)
                return k.snapshot(ref,ro,resources.scalar(loss))
            reference=ctx.main(serial)
            losses,_=k.update_group(model,optimizer,pieces,task,rank,4,'cpu',4.5)
            actual=k.snapshot(model,optimizer,resources.scalar(losses))
            comparison=k.compare(reference,actual,resources.TOLERANCES)
            assert comparison['passed'],(task,comparison)
            for parameter in model.parameters():
                shared=parameter.detach().clone();dist.broadcast(shared,0)
                torch.testing.assert_close(parameter,shared,rtol=0,atol=0)
    finally:dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(),reason='Gloo required')
def test_four_rank_global_transforms_accumulation_and_decoder_steps(tmp_path):
    torch.multiprocessing.spawn(_four_rank_numerics,args=((tmp_path/'distributed_store').as_uri(),),nprocs=4,join=True)


def test_bn_calibration_changes_only_copies_and_is_train_only(tmp_path,monkeypatch,compact_native):
    from hlt_classification.luka_reconstruction_stability_hydra import readouts,training
    from hlt_classification.luka_subclass_ddp_hydra.distributed import Context
    old,tr,va,bank=fixture(tmp_path,monkeypatch,rows=22)
    spec=dict(imported=old['imported'],maximum=12,content_hash='a'*64,counts=dict(train=22,validation=22))
    model,_=k.build(spec,'R24_s7',0,1,'cpu');state={n:v.clone() for n,v in model.state_dict().items()}
    ctx=Context(0,1,torch.device('cpu'));manifest=readouts.membership(spec,tr,technical=True)
    a=readouts.Averages();a.update(model,3.,128);rng=torch.get_rng_state().clone()
    results=readouts.evaluate_all(model,a,tr,va,manifest,ctx,training.evaluate,k.native,training.DelphesParticleTransformer)
    assert set(results)==set(c.READOUTS) and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(state[n],v) for n,v in model.state_dict().items())
    assert any(not torch.equal(results['RAW'][2][n],results['RAW_BN'][2][n]) for n in state if n.startswith('mod.') and 'running_' in n)
    with pytest.raises(PermissionError):readouts.membership(spec,va,technical=True)


def submission_spec(root):
    from hlt_classification.luka_reconstruction_stability_hydra import campaign
    return dict(campaign.registration(),root=str(root),project_dir=str(root/'code'),content_hash='a'*64,
        parents=dict(original='b'*64),counts=dict(train=299234,validation=74895),maximum=4,
        imported=dict(node=dict(initialization_seed=1,sampler_seed=2)),environment={},output_budget_bytes=29*2**30,
        recovery_anchors=dict(M0HLT=.96056122,OFFLINE=.96452569))


def resource_record(spec):
    profiles={};memory=dict(ranks=[dict(gpu_peak_bytes=5*2**30,peak_rss_bytes=3*2**30) for _ in range(4)],host_peak_bytes=13*2**30)
    devices=[dict(name='GeForce GTX 1080 Ti',compute_capability=[6,1],total_memory_bytes=11*2**30) for _ in range(4)]
    for name in c.RECIPES:
        task=f'{name}_s7';effective=128*c.cell(task)['accumulation'];times=[.6*c.cell(task)['accumulation']]*6
        phases={pos:times for pos in (('4.5','8.99','20.0') if name=='SOFT' else ('4.5','20.0'))}
        profiles[name]=dict(phase_samples=phases,step_seconds=float(np.mean(times)),validation_seconds=80.,probe_seconds=2.,load_seconds=50.,
            memory=memory,full_hlt_support=True,full_offline_support=True,partial11=True,tail_rows=spec['counts']['train']%effective,
            effective_batch=effective,train_rows=spec['counts']['train'],validation_rows=spec['counts']['validation'],readouts=list(c.READOUTS))
    return c.artifact('RESOURCES',parents=dict(spec=spec['content_hash'],bn_membership='b'*64,soft_calibration='c'*64),
        lane='pascal',profiles=profiles,miniatures={t:dict(content_hash='d'*64,seconds=15.,fit_memory=memory) for t in c.FITS},
        devices=devices,environment={},status='complete',storage=dict(setup_bytes=2**30,scientific_bytes=8*2**30,required_bytes=math.ceil(1.25*9*2**30)+2**30))


def fake_scheduler(monkeypatch):
    from hlt_classification.luka_reconstruction_stability_hydra import campaign,submission
    from hlt_classification.scouting import hcwdl_exact_dag_submission as dag
    calls=[];reviews=[]
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    monkeypatch.setattr(submission,'scheduler_review',lambda plan:reviews.append(plan))
    def run(command,**kwargs):calls.append(command);return SimpleNamespace(stdout=str(9000+len(calls))+'\n',returncode=0)
    monkeypatch.setattr(dag.subprocess,'run',run)
    return calls,reviews


def test_exact_45_job_auto_release_order_idempotence_and_envelope(tmp_path,monkeypatch):
    from hlt_classification.luka_reconstruction_stability_hydra import submission,resources,worker
    from hlt_classification.data.cache_contracts import write_immutable_json
    spec=submission_spec(tmp_path);folder=resources.setup_root(spec,'pascal');folder.mkdir()
    accepted=resources.accept(spec,resource_record(spec),'pascal');write_immutable_json(folder/'acceptance.json',accepted)
    calls,reviews=fake_scheduler(monkeypatch);submission.launch(spec);plan=submission.plan(spec,'bootstrap')
    boot=submission.launch(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert boot['jobs']==dict(setup_pascal='9001',release_pascal='9002')
    assert {v['stage'] for v in reviews}=={'bootstrap','envelope_pascal'}
    monkeypatch.setattr(submission,'accounting',lambda _:['COMPLETED','0:0'])
    monkeypatch.setattr(worker,'completed_setup',lambda *args:accepted)
    result=submission.finish(spec,'9002','pascal');assert len(calls)==45
    assert submission.finish(spec,'9002','pascal')==result and len(calls)==45
    assert set(result['jobs'])==set(c.FITS)|{'report_pascal'}
    science=submission.plan(spec,'science_pascal',accepted)
    assert [r['task_id'] for r in science['commands'][:-1]]==list(c.FITS)
    assert all(not r['dependencies'] and '--gres=gpu:geforce_gtx_1080_ti:4' in r['command'] for r in science['commands'][:-1])
    assert science['commands'][-1]['dependencies']==list(c.FITS)
    assert '--dependency=afterany:' in ' '.join(science['commands'][-1]['command'])
    assert submission.within_envelope(submission.plan(spec,'envelope_pascal'),science)
    science['commands'][0]['command'].append('--gres=gpu:8')
    with pytest.raises(PermissionError):submission.within_envelope(submission.plan(spec,'envelope_pascal'),science)


def test_failed_setup_and_partial_release_never_blindly_resubmit(tmp_path,monkeypatch):
    from hlt_classification.luka_reconstruction_stability_hydra import submission,resources,worker
    spec=submission_spec(tmp_path);calls,_=fake_scheduler(monkeypatch);submission.launch(spec)
    plan=submission.plan(spec,'bootstrap')
    with pytest.raises(PermissionError):submission.launch(spec,execute=True,plan_hash='0'*64,authorization=c.AUTHORIZATION)
    submission.launch(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    monkeypatch.setattr(submission,'accounting',lambda _:['FAILED','1:0'])
    with pytest.raises(RuntimeError,match='setup failed'):submission.finish(spec,'9002','pascal')
    assert len(calls)==2
    monkeypatch.setattr(submission,'accounting',lambda _:['COMPLETED','0:0'])
    monkeypatch.setattr(worker,'completed_setup',lambda *args:resources.accept(spec,resource_record(spec),'pascal'))
    folder=tmp_path/'submission_science_pascal';folder.mkdir();(folder/'live_claim').mkdir()
    with pytest.raises(RuntimeError,match='Ambiguous partial'):submission.finish(spec,'9002','pascal')
    assert len(calls)==2


@pytest.mark.parametrize('failure',['missing','time','memory','storage','readout','phase','nan'])
def test_resource_acceptance_includes_new_paths_and_storage(tmp_path,failure):
    from hlt_classification.luka_reconstruction_stability_hydra import resources
    spec=submission_spec(tmp_path);value=resource_record(spec)
    assert len(resources.accept(spec,value,'pascal')['minutes'])==42
    if failure=='missing':value['miniatures'].pop('SOFT_s9')
    if failure=='time':value['profiles']['R24']['phase_samples']={p:[10.]*6 for p in ('4.5','20.0')};value['profiles']['R24']['step_seconds']=10.
    if failure=='memory':value['profiles']['R24']['memory']['ranks'][0]['gpu_peak_bytes']=12*2**30
    if failure=='storage':spec['output_budget_bytes']=2**30
    if failure=='readout':value['profiles']['R24']['readouts']=['RAW']
    if failure=='phase':value['profiles']['SOFT']['phase_samples'].pop('8.99')
    if failure=='nan':value['profiles']['R24']['validation_seconds']=float('inf')
    with pytest.raises((ValueError,RuntimeError,MemoryError,OSError)):
        value=c.artifact('RESOURCES',parents=value['parents'],**{n:v for n,v in value.items() if n not in ('contract','schema_version','content_hash','parents') and n not in c.FLAGS})
        resources.accept(spec,value,'pascal')


def test_reporting_keeps_readouts_pairs_seeds_and_late_windows_separate():
    from hlt_classification.luka_reconstruction_stability_hydra import reporting
    anchors=dict(M0HLT=.9,OFFLINE=1.)
    history=[dict(epoch=i,validation=dict(macro_ovr_auc=.954 if i in (21,22,23,25,26) else .95)) for i in range(1,31)]
    late=reporting.stability(history,anchors)
    assert late['thresholds']['52']==dict(count=5,longest_streak=3)
    assert late['mean_recovery']==pytest.approx(52.) and late['minimum_recovery']==pytest.approx(50.)
    assert reporting.stability(history[:24],anchors)['mean_recovery'] is None
    rows=[dict(task=f'R24_s{s}',readout='RAW',auc=.95,recovery=r,delta=delta,delta_r24=0.,late=late)
        for s,r,delta in ((7,55.,.001),(8,49.,None))]
    summary=reporting.summarize(rows)
    assert summary['RAW']['R24']['completed_seeds']==2 and not summary['RAW']['R24']['complete']
    assert summary['RAW']['R24']['paired_delta']['count']==1
    assert summary['RAW']['R24']['recovery']['sample_sd']==pytest.approx(math.sqrt(18))
    assert summary['EMA05']['R24']['completed_seeds']==0


def test_all_42_initializations_are_fresh_and_pair_matched(tmp_path,compact_native):
    from hlt_classification.luka_reconstruction_stability_hydra import identity,calibration
    from unittest.mock import patch
    spec=submission_spec(tmp_path);records={}
    with patch.object(calibration,'load',return_value=dict(scale=.1)):
        for task in c.FITS:
            model,_=k.build(spec,task,0,1,'cpu');rng=torch.get_rng_state().clone()
            records[task]=identity.initial_record(spec,task,model,4)
            assert torch.equal(rng,torch.get_rng_state())
    identity.validate_pairs(spec,records,complete=True)
    assert len({records[f'C128_s{s}']['components']['native'] for s in c.SEED_INDICES})==3
    assert len({records[f'C128_s{s}']['epoch1_order_hash'] for s in c.SEED_INDICES})==3


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_full_298_slot_soft_cost_scale_and_annealing():
    torch.manual_seed(2026111501);cfg=c.scientific_cfg('SOFT_s7');decoder=rm.Decoder(cfg,298).cuda()
    jet=torch.randn(4,128,device='cuda');tokens=torch.randn(4,16,128,device='cuda');mask=torch.ones(4,16,device='cuda',dtype=torch.bool)
    targets=[target(n) for n in (0,1,170,298)]
    scores=decoder(jet,tokens,mask);matrices,_,_=transport.costs(scores,targets)
    scale=float(np.median([float((m-m.mean(1,keepdim=True)).abs().mean()) for m in matrices if m is not None]))
    for position in (3.01,4.5,8.99):
        decoder.zero_grad(set_to_none=True);scores=decoder(jet,tokens,mask)
        values,diagnostics=transport.rows(scores,targets,transport.temperature_at(scale,position))
        values.mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in decoder.parameters() if p.grad is not None)
        assert diagnostics['transport_residual_max']<=1e-4 and diagnostics['transport_gap_max']<=1e-4


def _protocol_worker(rank,root,init):
    import torch.distributed as dist
    from datetime import timedelta
    from test_luka_attention import cache
    from hlt_classification.luka_reconstruction_stability_hydra import resources
    from hlt_classification.luka_subclass_ddp_hydra.distributed import Context
    torch.set_num_threads(1);dist.init_process_group('gloo',init_method=init,rank=rank,world_size=4,timeout=timedelta(minutes=5))
    try:
        def build(spec,task,rank,world,device,**kwargs):
            torch.manual_seed(c.seeds(spec,task)['native_initialization']);model=Tiny(task)
            model.mod.encoder=nn.Linear(17,128);k.disable_dropout(model)
            optimizer=opt.optimizer(model);optimizer.stability_state['soft_scale']=1.
            return model,optimizer
        k.build=build;spec=submission_spec(Path(root));train=cache('train','D000',220)
        bank=SimpleNamespace(targets=np.zeros((len(train),128),np.float32))
        targets=SimpleNamespace(batch=lambda ix,epoch,cfg:([target(1+int(i)%4) for i in ix],[]))
        resources.qualification(spec,train,bank,targets,Context(rank,4,torch.device('cpu')),'pascal')
    finally:dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(),reason='Gloo required')
def test_complete_seventy_step_qualification_and_report_tamper(tmp_path):
    from hlt_classification.luka_reconstruction_stability_hydra import resources
    from hlt_classification.data.cache_contracts import load_json
    spec=submission_spec(tmp_path);root=resources.setup_root(spec,'pascal');root.mkdir()
    torch.multiprocessing.spawn(_protocol_worker,args=(str(tmp_path),(tmp_path/'protocol_store').as_uri()),nprocs=4,join=True)
    record=load_json(root/'qualification.json');resources.check_qualification(spec,'pascal',record)
    assert len(record['checks'])==70 and len(record['reference_digests'])==70
    assert not list(root.glob('*.pt'))
    (root/'parity_R24_s7_01.json').write_text('{}')
    with pytest.raises(ValueError):resources.check_qualification(spec,'pascal',record)


def test_source_inventory_and_pinned_wrappers():
    import ast
    from hlt_classification.luka_reconstruction_stability_hydra import campaign
    assert all(Path(p).is_file() for p in campaign.REQUIRED)
    assert campaign.REQUIRED[2].endswith('/campaign.py')
    for name in campaign.MODULES:
        path=Path('src/hlt_classification/luka_reconstruction_stability_hydra')/(name+'.py');source=path.read_text();ast.parse(source)
        assert 'Fresh_check' not in source
    cli=Path('scripts/luka_reconstruction_stability_hydra.py').read_text()
    wrapper=Path('sbatch/run_luka_reconstruction_stability_hydra.sh').read_text()
    assert "old['project_dir']" in cli and 'submodule_search_locations' in cli
    assert 'check_storage=False' in cli and 'unset PYTHONPATH' in wrapper
    assert 'PYTHONNOUSERSITE=1' in wrapper and '${ENV_DIR}/lib' in wrapper


def test_registration_pins_all_three_sources_and_preserves_original(tmp_path,monkeypatch):
    import sys
    from hlt_classification.luka_reconstruction_stability_hydra import campaign,storage
    from hlt_classification.data.cache_contracts import write_immutable_json
    lanes=c.LANES
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
    monkeypatch.setattr(campaign.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*2**30))
    cfg=campaign.create(tmp_path/'newrun',bpath,newcode,'a'*40,60,lanes=lanes)
    assert campaign.check(cfg,full=True)==old and oldpath.read_bytes()==before
    assert cfg['enabled_lanes']==list(lanes) and cfg['schema_version']==1
    assert cfg['maximum']==298 and cfg['embedding_hash']=='2'*64
    with pytest.raises(ValueError):campaign.create(tmp_path/'newrun',bpath,newcode,'a'*40,60)
    with pytest.raises(ValueError):campaign.create(oldroot/'nested',bpath,newcode,'a'*40,60)
    with pytest.raises(ValueError):campaign.create(tmp_path/'small',bpath,newcode,'a'*40,1)
    bad=deepcopy(cfg);bad['hardware']['pascal']['world_size']=3
    with pytest.raises(ValueError):campaign.check(bad)
    with monkeypatch.context() as low_disk:
        low_disk.setattr(campaign.shutil,'disk_usage',lambda p:SimpleNamespace(free=0))
        with pytest.raises(OSError,match='output allowance'):campaign.check(cfg)
        assert campaign.check(cfg,check_storage=False)==old
        with pytest.raises(ValueError):campaign.check(bad,check_storage=False)
    monkeypatch.setattr(campaign,'capture_source_snapshot',lambda *a,**kw:{'content_hash':'0'*64})
    with pytest.raises(ValueError,match='identity differs'):campaign.check(cfg)
    with pytest.raises(ValueError,match='identity differs'):campaign.check(cfg,check_storage=False)


def test_frozen_512_train_temperature_calibration_and_tamper(tmp_path,monkeypatch,compact_native):
    from hlt_classification.luka_reconstruction_stability_hydra import calibration,readouts
    from hlt_classification.luka_subclass_ddp_hydra.distributed import Context
    from hlt_classification.luka_next_supervision_hydra.reconstruction_targets import Targets
    from hlt_classification.data.cache_contracts import with_content_hash
    old,tr,va,bank=fixture(tmp_path,monkeypatch,rows=528);spec=submission_spec(tmp_path)
    spec.update(counts=dict(train=528,validation=528),maximum=12)
    (tmp_path/'setup_pascal').mkdir();targets=Targets(old,tr,'OFFLINE')
    ctx=Context(0,1,torch.device('cpu'));rng=torch.get_rng_state().clone()
    try:
        value=calibration.prepare(spec,tr,targets,ctx)
        assert calibration.load(spec)==value and value['scale']>0
        assert value['indices']==readouts.fixed_indices(tr,512,'stability-soft-calibration-v1').tolist()
        assert torch.equal(rng,torch.get_rng_state())
        with pytest.raises(PermissionError):calibration.prepare(spec,va,targets,ctx)
        bad=deepcopy(value);bad['indices'][0]=528
        import json
        (tmp_path/'setup_pascal/soft_calibration.json').write_text(json.dumps(with_content_hash(bad)))
        with pytest.raises(ValueError,match='calibration differs'):calibration.load(spec)
    finally:targets.close()


class PortableNative(nn.Module):
    """Small graph for full30 orchestration; installed CUDA covers real ParT."""
    def __init__(self):
        super().__init__();self.mod=nn.Module();self.mod.encoder=nn.Linear(17,128);self.mod.fc=nn.Linear(128,11)
    def forward(self,features,vectors,mask):
        tokens=self.mod.encoder(features.transpose(1,2));h=(tokens*mask.transpose(1,2)).sum(1)/mask.sum(2)
        return self.mod.fc(h)
    def no_weight_decay(self):return set()


class PortableStudent(PortableNative):
    def __init__(self,task):
        super().__init__();self.cfg=c.scientific_cfg(task);self.predictor=nn.Linear(128,128)
        self.aux=None if c.cell(task)['control'] else rm.Decoder(self.cfg,4)
    def forward(self,features,vectors,mask):
        self.particle_tokens=self.mod.encoder(features.transpose(1,2));self.particle_mask=mask[:,0]
        self.jet=(self.particle_tokens*mask.transpose(1,2)).sum(1)/mask.sum(2);self.ce_representation=self.jet
        self.prediction=self.predictor(self.jet);return self.mod.fc(self.jet)
    def deployment_state(self):return {n:v.detach().cpu().clone() for n,v in self.state_dict().items() if n.startswith('mod.')}


def portable_fixture(root):
    from test_luka_attention import cache
    cfg=submission_spec(Path(root));tr=cache('train','D000',139);va=cache('validation','D000',33)
    va.identities[:,-1]=1
    for block in va.blocks:block.identities[:,-1]=1
    cfg.update(counts=dict(train=len(tr),validation=len(va)),foundation_hash=tr.foundation_sha256,embedding_hash='b'*64)
    bank=SimpleNamespace(targets=np.zeros((len(tr),128),np.float32),manifest={'content_hash':'b'*64})
    return cfg,tr,va,bank


def _full_worker(rank,root,init):
    import torch.distributed as dist
    from datetime import timedelta
    from hlt_classification.luka_reconstruction_stability_hydra import training,readouts
    from hlt_classification.luka_subclass_ddp_hydra.distributed import Context
    torch.set_num_threads(1);dist.init_process_group('gloo',init_method=init,rank=rank,world_size=4,timeout=timedelta(minutes=8))
    try:
        def build(spec,task,rank,world,device,**kwargs):
            torch.manual_seed(c.seeds(spec,task)['native_initialization']);model=PortableStudent(task)
            torch.manual_seed(c.seeds(spec,task)['dropout_base']+rank);return model,opt.optimizer(model)
        k.build=build;training.DelphesParticleTransformer=PortableNative
        # Only fixture population/model are reduced; keep production30 epochs,
        # four ranks, accumulation, selection, serialization and authentication.
        readouts.load_membership=lambda spec,train,**kw:readouts.membership(spec,train,technical=True)
        spec,tr,va,bank=portable_fixture(root);ctx=Context(rank,4,torch.device('cpu'))
        targets=SimpleNamespace(batch=lambda ix,epoch,cfg:([target(1+int(i)%4) for i in ix],[]))
        for task in ('R24_s7','C128_s7'):training.fit(Path(root)/task/'fit',spec,task,tr,va,bank,targets,ctx)
    finally:dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(),reason='Gloo required')
def test_full30_four_rank_paired_exports_results_progress_and_tamper(tmp_path,monkeypatch):
    from hlt_classification.luka_reconstruction_stability_hydra import training,reporting,worker
    torch.multiprocessing.spawn(_full_worker,args=(str(tmp_path),(tmp_path/'full_store').as_uri()),nprocs=4,join=True)
    spec,_,_,_=portable_fixture(tmp_path)
    for task in ('R24_s7','C128_s7'):
        report,_=training.validate_fit(tmp_path/task/'fit',spec,task)
        assert len(report['history'])==30 and set(report['history'][0]['readouts'])=={'RAW','RAW_BN'}
        assert set(report['history'][2]['readouts'])==set(c.READOUTS)
        (tmp_path/task/'receipt.json').write_text('{}')
    monkeypatch.setattr(reporting,'states',lambda spec:{})
    monkeypatch.setattr(worker,'receipt',lambda *a:None)
    results=reporting.results(spec,readout='all');progress=reporting.progress(spec,readout='all')
    assert len(results['rows'])==8 and len(results['pending'])==40
    assert results['summary']['EMA05']['R24']['completed_seeds']==1
    assert all(results['summary'][n]['R24']['late_mean']['count']==1 for n in c.READOUTS)
    assert progress['best']['RAW']['R24_s7'] and progress['best']['RAW']['R24_s8'] is None
    (tmp_path/'R24_s7/fit/epoch-01.json').write_text('{}')
    with pytest.raises((ValueError,KeyError)):reporting.results(spec)


def _cuda_qualification(rank,root,init,spec,old,train,bank):
    import torch.distributed as dist
    from datetime import timedelta
    from hlt_classification.luka_reconstruction_stability_hydra import calibration,resources
    from hlt_classification.luka_subclass_ddp_hydra.distributed import Context
    from hlt_classification.luka_next_supervision_hydra.reconstruction_targets import Targets
    torch.cuda.set_device(rank);k.configure(4,12)
    dist.init_process_group('nccl',init_method=init,rank=rank,world_size=4,timeout=timedelta(minutes=10))
    try:
        targets=Targets(old,train,'OFFLINE');ctx=Context(rank,4,torch.device('cuda',rank))
        calibration.prepare(spec,train,targets,ctx)
        resources.qualification(spec,train,bank,targets,ctx,'pascal');targets.close()
    finally:dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count()<4 or not torch.distributed.is_nccl_available(),reason='Four CUDA GPUs/NCCL required; mandatory Hydra setup')
def test_real_four_cuda_syncbn_seventy_comparisons_and_calibration(tmp_path,monkeypatch):
    old,tr,va,bank=fixture(tmp_path,monkeypatch,rows=528);spec=submission_spec(tmp_path)
    spec.update(imported=old['imported'],maximum=12,counts=dict(train=528,validation=528))
    (tmp_path/'setup_pascal').mkdir()
    torch.multiprocessing.spawn(_cuda_qualification,args=(str(tmp_path),(tmp_path/'cuda_store').as_uri(),spec,old,tr,bank),nprocs=4,join=True)


def test_exact_four_gpu_launcher_and_duplicate_claim(tmp_path,monkeypatch):
    from hlt_classification.luka_reconstruction_stability_hydra import campaign,worker
    spec=submission_spec(tmp_path);calls=[]
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:{})
    monkeypatch.setattr(worker,'allocation',lambda *a:('123',resource_record(spec)['devices']))
    monkeypatch.setattr(worker,'require_backend',lambda:None)
    monkeypatch.setattr(worker.reconstruction_targets,'authenticate',lambda spec:None)
    monkeypatch.setattr(worker.subprocess,'run',lambda args,**kw:calls.append(args))
    worker.run(spec,'setup_pascal')
    assert '--nproc-per-node=4' in calls[0] and '--max-restarts=0' in calls[0]
    with pytest.raises(FileExistsError):worker.run(spec,'setup_pascal')


def test_allocation_checks_exact_pascal_shape(tmp_path,monkeypatch):
    from hlt_classification.luka_reconstruction_stability_hydra import worker
    spec=submission_spec(tmp_path);devices=resource_record(spec)['devices']
    monkeypatch.setattr(worker,'require_job',lambda *a:'123')
    fields='JobId=123 JobState=RUNNING Partition=gpus Account=default QOS=normal NumNodes=1 NumCPUs=12 NumTasks=1 CPUs/Task=12 AllocTRES=cpu=12,mem=64000M,node=1,gres/gpu=4'
    monkeypatch.setattr(worker,'controller',lambda args:'ClusterName = hydra\n' if args==['show','config'] else fields)
    monkeypatch.setenv('PYTHONNOUSERSITE','1');monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    for key in ('SLURM_CLUSTER_NAME','SLURM_NTASKS','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE'):monkeypatch.delenv(key,raising=False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True);monkeypatch.setattr(torch.cuda,'device_count',lambda:4)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:devices[i]['name'])
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda i:tuple(devices[i]['compute_capability']))
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda i:SimpleNamespace(total_memory=devices[i]['total_memory_bytes']))
    assert worker.allocation(spec,'setup_pascal')==('123',devices)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:3)
    with pytest.raises(PermissionError):worker.allocation(spec,'setup_pascal')


def test_gradient_buckets_are_bounded_and_reduce_all_elements(monkeypatch):
    values=[torch.ones(800000),torch.full((1100000,),2.),torch.full((17,),3.)];sizes=[]
    def reduce(tensor):sizes.append(tensor.numel());tensor.mul_(4)
    monkeypatch.setattr(opt.dist,'all_reduce',reduce);opt.reduce_values(values,4)
    assert max(sizes)<=1048576 and sum(sizes)==1900017
    for value,expected in zip(values,(4.,8.,12.)):assert torch.all(value==expected)


def test_chunked_transport_diagnostics_preserve_worst_residual(monkeypatch):
    from hlt_classification.luka_reconstruction_stability_hydra import decoder
    m=Tiny('SOFT_s7');raw,emb,targets=batch(11);k.native(m,raw,'cpu');original=transport.rows;count=0
    def measured(scores,targets,temperature):
        nonlocal count
        losses,parts=original(scores,targets,temperature);count+=1
        parts['transport_residual_max']=.09 if count==1 else .01
        return losses,parts
    monkeypatch.setattr(transport,'rows',measured)
    _,parts,_,_,_=decoder.backward(m.aux,m.jet,m.particle_tokens,m.particle_mask,targets,.3,temperature=1.)
    assert count==2 and parts['transport_residual_max']==.09

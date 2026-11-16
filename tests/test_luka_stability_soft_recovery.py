from copy import deepcopy
import numpy as np
import pytest
import torch
from hlt_classification.luka_reconstruction_stability_hydra import transport,transport_v2 as v2
from hlt_classification.luka_reconstruction_stability_hydra import contracts as c,recovery,submission,repeat_audit


def problem(k,n,scale):
    generator=torch.Generator().manual_seed(21)
    cost=torch.rand(k,n,generator=generator,dtype=torch.float64)*scale
    mass=torch.ones(n,dtype=torch.float64);mass[-1]=k-n+1
    return cost,mass


@pytest.mark.parametrize('k,n,scale',[(32,8,10),(32,32,10),(298,30,10),(298,298,10),(64,12,100),(298,20,100)])
def test_polishes_problems_that_old_solver_rejects(k,n,scale):
    torch.set_num_threads(1);cost,mass=problem(k,n,scale)
    with pytest.raises(transport.ScalingConvergenceError):transport.solve(cost,mass,.01)
    p,r=v2.solve(cost,mass,.01)
    assert r['newton_iterations']>0 and r['marginal_residual']<=1e-4 and r['normalized_gap']<=1e-4
    torch.testing.assert_close(p.sum(1),torch.ones(k,dtype=torch.float64),atol=1e-4,rtol=0)
    torch.testing.assert_close(p.sum(0),mass,atol=1e-4,rtol=0)


def test_fast_path_exact_and_envelope_gradient():
    cost=torch.tensor([[.1,1.],[1.,.1],[.8,.2]],dtype=torch.float64);mass=torch.tensor([1.,2.])
    p,r=transport.solve(cost,mass,.1);actual,record=v2.solve(cost,mass,.1)
    assert torch.equal(p,actual) and record['primal']==r['primal']
    cost,mass=problem(32,8,10);p,r=v2.solve(cost,mass,.01)
    for i,j in ((0,0),(5,7),(25,3)):
        delta=torch.zeros_like(cost);delta[i,j]=1e-3
        fd=(v2.solve(cost+delta,mass,.01)[1]['primal']-v2.solve(cost-delta,mass,.01)[1]['primal'])/.002
        assert abs(fd-p[i,j].item())<2e-3


def test_invalid_and_unconverged_fail_closed(monkeypatch):
    cost,mass=problem(32,8,10)
    with pytest.raises(ValueError):v2.solve(cost,mass,-1.)
    with pytest.raises(ValueError):v2.solve(cost*float('nan'),mass,.01)
    monkeypatch.setitem(v2.PROTOCOL,'newton_iterations',0)
    with pytest.raises(v2.ConvergenceError) as raised:v2.solve(cost,mass,.01)
    assert torch.equal(raised.value.problem['cost'],cost)


def test_recovery_is_only_three_fits_with_auto_release(tmp_path):
    from test_luka_reconstruction_stability_hydra import submission_spec
    spec=submission_spec(tmp_path);spec.update(recovery.registration(spec))
    spec['soft_solver']=v2.PROTOCOL
    assert c.tasks('pascal',spec)==recovery.TASKS
    assert c.qualification_tasks(spec)==('SOFT_s7',)
    boot=submission.plan(spec,'bootstrap');science=submission.plan(spec,'envelope_pascal')
    assert [r['task_id'] for r in boot['commands']]==['setup_pascal','release_pascal']
    assert boot['commands'][1]['dependencies']==['setup_pascal']
    assert [r['task_id'] for r in science['commands']]==[*recovery.TASKS,'report_pascal']
    assert science['commands'][-1]['dependencies']==list(recovery.TASKS)
    assert all('--gres=gpu:geforce_gtx_1080_ti:4' in r['command'] for r in science['commands'][:-1])
    assert all(not r['dependencies'] for r in science['commands'][:-1])


def test_audit_withholds_only_mismatched_pair():
    initial=dict(components=dict(native='a',predictor='b',decoder='c'),seeds={'x':1},
        dropout_rank_seeds=[1,2,3,4],epoch1_order_hash='d',world_size=4)
    records={'C_s4':initial,'R24_s4':deepcopy(initial),'R20_s4':deepcopy(initial)}
    records['R20_s4']['components']['native']='different'
    assert repeat_audit.pair_status(records,'R24_s4','C_s4')['status']=='matched'
    mismatch=repeat_audit.pair_status(records,'R20_s4','C_s4')
    assert mismatch['status']=='mismatch' and set(mismatch['differences'])=={'native'}
    assert repeat_audit.pair_status(records,'R22_s4','C_s4')['status']=='incomplete'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cuda_newton_returns_same_plan_and_correct_envelope():
    cost,mass=problem(298,30,10)
    expected,r=v2.solve(cost,mass,.01)
    cost=cost.cuda().requires_grad_(True)
    actual,record=v2.solve(cost,mass.cuda(),.01)
    assert actual.device==cost.device and record['newton_iterations']>0
    torch.testing.assert_close(actual.cpu(),expected,rtol=1e-5,atol=1e-6)
    value=(actual*cost).sum()+.01*record['entropy'];value.backward()
    torch.testing.assert_close(cost.grad,actual,rtol=0,atol=0)


def test_recovery_initialization_is_exact_and_failure_refs_authenticated(tmp_path):
    from hlt_classification.data.cache_contracts import write_immutable_json
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    value=dict(components={'native':'a','predictor':'b','decoder':'c'},seeds={'s':7},dropout_rank_seeds=[7,8,9,10],
        epoch1_order_hash='d',world_size=4)
    path=tmp_path/'initialization.json';write_immutable_json(path,value)
    spec={'failures':{'SOFT_s7':{'initialization':file_ref(path)}}}
    recovery.check_initial(spec,'SOFT_s7',value)
    bad=deepcopy(value);bad['components']['native']='different'
    with pytest.raises(ValueError,match='starting state'):recovery.check_initial(spec,'SOFT_s7',bad)
    path.write_text('{}')
    with pytest.raises(ValueError):recovery.check_initial(spec,'SOFT_s7',value)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Installed Weaver CUDA required')
def test_recovery_installed_weaver_native_exports(tmp_path,monkeypatch):
    from test_luka_direct_reconstruction import fixture
    from hlt_classification.luka_next_supervision_hydra.reconstruction_targets import Targets
    from hlt_classification.luka_reconstruction_stability_hydra import campaign,calibration,training,kernel,identity
    from hlt_classification.luka_subclass_ddp_hydra.distributed import Context
    from hlt_classification.data.cache_contracts import write_immutable_json
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    old,tr,va,bank=fixture(tmp_path,monkeypatch,rows=22)
    va.identities[:,-1]=1
    for block in va.blocks:block.identities[:,-1]=1
    spec=dict(campaign.registration(),root=str(tmp_path),content_hash='a'*64,parents=dict(original='b'*64),
        maximum=12,counts=dict(train=len(tr),validation=len(va)),imported=old['imported'],
        foundation_hash=tr.foundation_sha256,embedding_hash=bank.manifest['content_hash'],
        recovery_anchors=dict(M0HLT=.96056122,OFFLINE=.96452569),soft_solver=v2.PROTOCOL)
    monkeypatch.setattr(calibration,'load',lambda spec:dict(scale=.1))
    kernel.configure(1,3);model,_=kernel.build(spec,'SOFT_s7',0,1,'cuda')
    initial=identity.initial_record(spec,'SOFT_s7',model,1);del model
    path=tmp_path/'old_initialization.json';write_immutable_json(path,initial)
    spec.update(recovery_parent={},failures={'SOFT_s7':{'initialization':file_ref(path)}})
    ctx=Context(0,1,torch.device('cuda'));targets=Targets(old,tr,'OFFLINE')
    try:
        training.fit(tmp_path/'replacement',spec,'SOFT_s7',tr,va,bank,targets,ctx,technical=True)
        training.validate_fit(tmp_path/'replacement',spec,'SOFT_s7',technical=True)
    finally:targets.close();torch.cuda.empty_cache()


def test_six_job_release_is_idempotent_and_requires_recovery_authorization(tmp_path,monkeypatch):
    from test_luka_reconstruction_stability_hydra import submission_spec,fake_scheduler
    from hlt_classification.luka_reconstruction_stability_hydra import worker
    from hlt_classification.data.cache_contracts import write_immutable_json
    spec=submission_spec(tmp_path);spec.update(recovery.registration(spec));spec['recovery_parent']={}
    accepted=c.artifact('ACCEPTANCE',parents=dict(spec=spec['content_hash']),minutes={t:1200 for t in recovery.TASKS})
    folder=tmp_path/'setup_pascal';folder.mkdir();write_immutable_json(folder/'acceptance.json',accepted)
    calls,_=fake_scheduler(monkeypatch);submission.launch(spec);plan=submission.plan(spec,'bootstrap')
    with pytest.raises(PermissionError):submission.launch(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    submission.launch(spec,execute=True,plan_hash=plan['content_hash'],authorization=plan['authorization'])
    monkeypatch.setattr(submission,'accounting',lambda _:['FAILED','1:0'])
    with pytest.raises(RuntimeError):submission.finish(spec,'9002','pascal')
    assert len(calls)==2
    monkeypatch.setattr(submission,'accounting',lambda _:['COMPLETED','0:0'])
    monkeypatch.setattr(worker,'completed_setup',lambda *args:accepted)
    value=submission.finish(spec,'9002','pascal')
    assert set(value['jobs'])==set(recovery.TASKS)|{'report_pascal'} and len(calls)==6
    assert submission.finish(spec,'9002','pascal')==value and len(calls)==6


def test_recovery_gate_requires_evolved_probe_and_conservative_timing(tmp_path):
    import math
    from test_luka_reconstruction_stability_hydra import submission_spec,resource_record
    from hlt_classification.luka_reconstruction_stability_hydra import resources
    from hlt_classification.data.cache_contracts import with_content_hash,write_immutable_json
    spec=submission_spec(tmp_path);spec.update(recovery.registration(spec));spec['recovery_parent']={}
    value=resource_record(spec);value['profiles']={'SOFT':value['profiles']['SOFT']}
    value['miniatures']={t:value['miniatures'][t] for t in recovery.TASKS};value=with_content_hash(value)
    with pytest.raises(FileNotFoundError):resources.accept(spec,value,'pascal')
    probe=c.artifact('EVOLVED_SOFT_PROBE',parents=dict(spec=spec['content_hash']),task='SOFT_s8',
        train_passes=4,train_rows=spec['counts']['train'],updates=4*math.ceil(spec['counts']['train']/128),
        discarded=True,validation_accessed=False,phase_samples={k:[.5]*6 for k in ('4.5','8.99','20.0')},
        memory=value['profiles']['SOFT']['memory'],seconds=100.,solver=v2.PROTOCOL,
        numerical_cases=[[dict(marginal_residual=1e-5,normalized_gap=1e-5) for _ in range(3)] for _ in range(4)])
    folder=tmp_path/'setup_pascal';folder.mkdir();write_immutable_json(folder/'evolved_soft_probe.json',probe)
    assert set(resources.accept(spec,value,'pascal')['minutes'])==set(recovery.TASKS)
    value['profiles']['SOFT']['phase_samples']['8.99']=[.1]*6;value=with_content_hash(value)
    with pytest.raises(ValueError,match='evolved solver cost'):resources.accept(spec,value,'pascal')

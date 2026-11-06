"""Group C likelihood, measured-set targets, native export and independent release."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from test_luka_next_supervision_hydra import compact_native,particles,spec_fixture
from test_luka_next_supervision_group_b import fake_raw
from hlt_classification.luka_next_supervision_hydra import contracts as c,probabilistic_contracts as pc
from hlt_classification.luka_next_supervision_hydra import probabilistic_model as pm,probabilistic_targets as pt,group_c
from hlt_classification.luka_next_supervision_hydra import model,training,inputs,resources,worker,submission,campaign,reporting
from hlt_classification.data.cache_contracts import load_json,write_immutable_json,with_content_hash
from hlt_classification.luka_embedding_aux.model import AuxiliaryParT


def fixture(tmp_path,monkeypatch,rows=22):
    spec,tr,va,bank=spec_fixture(tmp_path,rows)
    spec.update(group_c.registration(),counts=dict(train=rows,validation=rows),root=str(tmp_path),remaining=list(pc.TASKS),
        reference=dict(environment={}),qualification=dict(acceptance=dict(gpu=dict(name='fixture',total_memory_bytes=8*2**30))))
    (tmp_path/'setup').mkdir(); raw=fake_raw(spec,tr); monkeypatch.setattr(inputs,'raw',lambda s:raw)
    monkeypatch.setattr(inputs,'embedding_bank',lambda s,t:bank)
    pt.prepare(spec,tr,bank)
    return spec,tr,va,bank


def test_registry_seven_and_frozen_A_B_semantics():
    from hlt_classification.luka_next_supervision_hydra import structured_contracts as sc
    assert pc.TASKS==('D1','S1','D2','S2','D5','S5','D3')
    assert c.recipe()['content_hash']=='d39c6ee09d803622385834756bb57c131ed8a8142659b1d99aec248ec6b6bddc'
    assert len(set(c.TASKS)|set(sc.TASKS)|set(pc.TASKS))==40
    assert c.setting('D5')['components']==1 and c.setting('D1')['components']==4
    assert c.setting('S5')['source']=='HLT' and c.setting('D3')['coefficient']==.1
    assert c.setting('S2')['seed_index']==c.setting('D2')['seed_index']==1
    assert len(group_c.registration()['tasks'])==9 and pc.recipe()['embedding_weight']==5
    with pytest.raises(NotImplementedError): c.setting('D4')


def test_density_exact_joint_likelihood_and_negative_values():
    z=torch.full((2,128),.1); mu=torch.zeros(2,2,128,requires_grad=True)
    mu.data[:,1,:]=1.; std=torch.ones_like(mu)*.5; logits=torch.tensor([[0.,1.],[1.,0.]],requires_grad=True)
    score=dict(mean=mu,std=std,logits=logits)
    component=torch.distributions.Normal(mu,std).log_prob(z[:,None]).sum(-1)+logits.log_softmax(-1)
    expected=-torch.logsumexp(component,-1).mean()/128
    actual=pm.loss(score,dict(targets=z), 'D')
    torch.testing.assert_close(actual,expected); actual.backward()
    assert mu.grad.abs().sum()>0 and logits.grad.abs().sum()>0
    narrow=dict(mean=torch.zeros(2,1,128),std=torch.full((2,1,128),.1),logits=torch.zeros(2,1))
    assert pm.loss(narrow,dict(targets=z),'D')<0


def test_density_whole_vector_component_not_coordinate_mixtures():
    mu=torch.zeros(1,2,128); mu[:,0,:64]=-2.; mu[:,1,:64]=2.
    mu[:,0,64:]=2.; mu[:,1,64:]=-2.
    score=dict(mean=mu,std=torch.ones_like(mu),logits=torch.zeros(1,2)); z=torch.full((1,128),2.)
    joint=pm.loss(score,dict(targets=z),'D')
    marginal=-torch.logsumexp(torch.distributions.Normal(mu,1).log_prob(z[:,None])-math.log(2),1).mean()
    assert joint>marginal+2


@pytest.mark.parametrize('k',[1,4])
def test_density_scale_bounds_FP32_gradient_and_diagnostics(k):
    head=pm.DensityHead(k); h=torch.randn(3,128,requires_grad=True); target=torch.randn(3,128,requires_grad=True)
    batch=dict(targets=target); score=head(h,batch)
    torch.testing.assert_close(score['std'],torch.ones_like(score['std']))
    assert (score['logits']==0).all()
    loss=pm.loss(score,batch,'D'); loss.backward()
    assert h.grad.norm()>0 and target.grad is None
    with torch.no_grad():
        d=pm.diagnostics(score,batch,'D'); assert len(d['responsibilities'])==k
        head.scale.bias.fill_(100); assert (head(h,batch)['std']<=2).all()
        head.scale.bias.fill_(-100); assert (head(h,batch)['std']>=.1).all()
    bad=dict(score,std=torch.zeros_like(score['std']))
    with pytest.raises(ValueError): pm.loss(bad,batch,'D')


def test_measured_HLT_reference_signed_fields_and_missing_errors():
    raw=particles(n=1,counts=3).values; raw[:,6]=[-1.,2.,3.]; raw[:,10:]=1
    raw[0,8]=0.; raw[1,10]=0; raw[1,6]=1e300
    x,valid=pt.measured(raw,np.array([.4,.3,10.]))
    assert x.shape==(3,22) and not valid[0,16] and not valid[1,14]
    assert x[1,14]==0 and x[0,14]<0 and x[2,14]>0
    np.testing.assert_allclose(x[:,0],np.log(np.hypot(raw[:,0],raw[:,1])/10))
    np.testing.assert_allclose(x[:,2],np.arcsinh(raw[:,2]/np.hypot(raw[:,0],raw[:,1]))-.4)
    assert valid[:,18:].all() and x[0,20]==0
    assert (x[:,5:11].sum(-1)==1).all() and (x[:,11:14].sum(-1)==1).all()
    with pytest.raises(ValueError): pt.measured(raw,np.array([0.,0.,0.]))


def test_target_preparation_join_masks_hashes_full_collection_and_frozen_centers(tmp_path,monkeypatch):
    spec,tr,va,bank=fixture(tmp_path,monkeypatch)
    prepared=pt.authenticate(spec)
    assert prepared['role']=='train' and not prepared['quality_gate']
    source=pt.candidates(spec,'S1',tr); assert isinstance(source.values,np.memmap)
    assert source.capacity['full_collection'] and source.capacity['rows']==22
    batch,_=source.batch(np.array([2,4]),3,c.setting('S1'))
    assert np.array_equal(batch['count'],source.counts[[2,4]])
    assert np.array_equal(batch['real'].sum(-1),batch['count'])
    assert batch['noisy'].shape[-1]==22
    density=pt.candidates(spec,'D1',tr); m=model.new_model(spec['imported'],'D1'); density.initialize(m)
    np.testing.assert_array_equal(m.aux.mean.bias.detach().reshape(4,128),density.centers)
    with pytest.raises(PermissionError): pt.candidates(spec,'S1',va)
    path=tmp_path/'setup/density_centers.npz'; path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError): pt.authenticate(spec)


def test_noise_velocity_identity_and_isolated_reproducibility(tmp_path,monkeypatch):
    spec,tr,_,_=fixture(tmp_path,monkeypatch); source=pt.candidates(spec,'S1',tr); ix=np.array([0,3,5])
    torch_state=torch.get_rng_state(); np_state=np.random.get_state()
    a,_=source.batch(ix,2,c.setting('S1')); b,_=source.batch(ix,2,c.setting('S1'))
    assert torch.equal(torch_state,torch.get_rng_state())
    assert all(np.array_equal(x,y) for x,y in zip(np_state,np.random.get_state()))
    for key in a: np.testing.assert_array_equal(a[key],b[key])
    other,_=source.batch(ix[::-1],2,c.setting('S1'))
    np.testing.assert_array_equal(other['noisy'],a['noisy'][::-1])
    again,_=source.batch(ix,3,c.setting('S1')); assert not np.array_equal(a['noisy'],again['noisy'])
    for j,i in enumerate(ix):
        phase=np.pi*a['time'][j]/2; clean=np.cos(phase)*a['noisy'][j]-np.sin(phase)*a['velocity'][j]
        start,stop=source.offsets[i:i+2]
        # A row permutation preserves the clean set, without HLT particle matches.
        np.testing.assert_allclose(np.sort(clean[:stop-start],axis=0),np.sort(source.values[start:stop],axis=0),atol=5e-6)


def set_batch(n=3,length=5):
    count=torch.tensor([length-j%2 for j in range(n)]); real=torch.arange(length)[None]<count[:,None]
    return dict(noisy=torch.randn(n,length,22),velocity=torch.randn(n,length,22),valid=real[...,None].expand(-1,-1,22).clone(),
        real=real,time=torch.full((n,),.7),count=count)


def test_set_exact_per_jet_group_reduction_and_count_normalization():
    b=set_batch(2,4); b['velocity'].zero_(); b['valid'][0,:,14:18]=False
    pred=torch.ones_like(b['velocity']); pred[...,:5]*=2.; pred[...,5:14]*=3.
    scores=dict(velocity=pred,count_logits=torch.zeros(2,5))
    rows,groups,ce=pm.set_rows(scores,b)
    torch.testing.assert_close(groups,torch.tensor([[4.,9.,1.],[4.,9.,1.]]))
    torch.testing.assert_close(ce,torch.ones(2)); torch.testing.assert_close(rows,groups.mean(-1)+.1)


def test_set_equivariance_padding_invalid_target_and_conditioning_gradients():
    head=pm.SetHead(); head.configure(7); h=torch.randn(3,128,requires_grad=True); b=set_batch()
    b['valid'][:,:,14]=False; b['velocity'][:,:,14]=float('nan')
    out=head(h,b); loss=pm.loss(out,b,'S'); loss.backward()
    assert h.grad.norm()>0
    perm=torch.tensor([2,0,4,1,3]); changed={k:v[:,perm] if v.ndim>=2 and v.shape[1]==5 else v for k,v in b.items()}
    shifted=head(h,changed)
    torch.testing.assert_close(shifted['velocity'],out['velocity'][:,perm],atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(pm.loss(shifted,changed,'S'),loss)
    padded={k:F.pad(v,(0,0,0,3)) if k in ('noisy','velocity','valid') else F.pad(v,(0,3)) if k=='real' else v for k,v in b.items()}
    padded['noisy'][:,5:]=float('nan'); padded['velocity'][:,5:]=float('nan')
    result=head(h,padded)
    torch.testing.assert_close(result['velocity'][:,:5],out['velocity'],atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(pm.loss(result,padded,'S'),loss)
    altered=dict(b,velocity=torch.full_like(b['velocity'],999),valid=~b['valid'],count=b['count']+1)
    # Forward only reads noisy rows, padding and noise level. Clean values,
    # supervision masks and true count never condition the prediction.
    torch.testing.assert_close(head(h,altered)['velocity'],out['velocity'])
    with pytest.raises(ValueError): pm.loss(out,altered,'S')


def test_chunked_set_path_equal_and_all_particles_above64():
    head=pm.SetHead(); head.configure(91); h=torch.randn(10,128,requires_grad=True); b=set_batch(10,91)
    out=head(h,b); split=[]
    for i in range(10): split.append(head(h[i:i+1],{k:v[i:i+1] for k,v in b.items()}))
    torch.testing.assert_close(out['velocity'],torch.cat([x['velocity'] for x in split]),atol=2e-6,rtol=1e-5)
    grad=torch.autograd.grad(pm.loss(out,b,'S'),h)[0]; assert grad.norm()>0
    b['velocity'][:,80:,0]+=10
    assert pm.loss(out,b,'S')>pm.loss(out,dict(b,velocity=torch.zeros_like(b['velocity'])),'S')


@pytest.mark.parametrize('task',pc.TASKS)
def test_every_C_path_native_parity_training_replay_and_no_targets_at_deployment(tmp_path,monkeypatch,task):
    spec,tr,va,bank=fixture(tmp_path,monkeypatch)
    source=pt.candidates(spec,task,tr); original_seed=c.seed(spec['imported'],c.setting(task),'initialization_seed')
    torch.manual_seed(original_seed); original=AuxiliaryParT(); native_state=original.native_state(); predictor=deepcopy(original.predictor.state_dict())
    after=torch.get_rng_state(); m=model.new_model(spec['imported'],task)
    assert torch.equal(after,torch.get_rng_state())
    assert all(torch.equal(v,m.deployment_state()[k]) for k,v in native_state.items())
    assert all(torch.equal(v,m.predictor.state_dict()[k]) for k,v in predictor.items())
    source.initialize(m); opt=training.optimizer(m)
    before={k:v.clone() for k,v in m.state_dict().items()}; b=tr.batch(np.arange(22))
    losses,probe=training.step(m,opt,b,bank.targets,'cpu',candidates=source,indices=np.arange(22),epoch=1,position=7.)
    assert np.isfinite(losses).all() and probe['probes']['auxiliary_jet_gradient_norm']>=0
    assert any(not torch.equal(before[k],v) for k,v in m.state_dict().items() if k.startswith('mod.'))
    assert all(k.startswith('mod.') for k in m.deployment_state())
    report=training.fit(tmp_path/('fit_'+task),spec,task,tr,va,bank,candidates=source,device='cpu',technical=True)
    training.validate_fit(tmp_path/('fit_'+task),spec,task,technical=True)
    assert report['parents']['probabilistic_targets']==pt.authenticate(spec)['content_hash']
    restored=model.deployment(m.deployment_state(),'cpu'); assert not hasattr(restored,'aux')
    assert report['parameter_counts']['deployed']==sum(v.numel() for v in restored.parameters())


def test_negative_density_history_is_valid_and_full30_not_quality_gated(tmp_path,monkeypatch):
    spec,tr,va,bank=fixture(tmp_path,monkeypatch)
    source=pt.candidates(spec,'D5',tr); old=pm.DensityHead.forward
    def narrow(self,h,batch):
        out=old(self,h,batch); out['std']=out['std']*0+.1; out['mean']=out['mean']*0
        return out
    monkeypatch.setattr(pm.DensityHead,'forward',narrow)
    report=training.fit(tmp_path/'negative',spec,'D5',tr,va,bank,candidates=source,device='cpu')
    assert report['epochs']==30 and len(report['validation_history'])==30
    assert all(r['auxiliary_loss']<0 for r in report['validation_history'])
    training.validate_fit(tmp_path/'negative',spec,'D5')


def test_set_miniature_includes_target_tail_and_real_partial(tmp_path,monkeypatch):
    spec,tr,_,_=fixture(tmp_path,monkeypatch,rows=270); source=pt.candidates(spec,'S1',tr)
    source.counts[12]=1000
    ix=pt.miniature_indices(source,np.array([1,2,3]))
    assert 12 in ix and len(np.unique(ix))==len(ix) and len(ix)%128>=2


def test_scope_specific_independent_queue_order_and_no_reference_rerun(tmp_path):
    spec,_,_,_=spec_fixture(tmp_path); spec.update(group_c.registration(),remaining=list(pc.TASKS),project_dir=str(tmp_path),reused={'C0':{},'C1':{}})
    acceptance=dict(content_hash='e'*64,minutes={t:400 for t in pc.TASKS},probabilistic_review={'reviewed':True})
    plan=submission.plan(spec,'science',acceptance)
    assert [r['task_id'] for r in plan['commands']]==list(pc.TASKS)
    assert c.authorization(spec)==pc.AUTHORIZATION and plan['scope']=='group_C_only'
    assert plan['probabilistic_target_review']=={'reviewed':True}
    for command in plan['commands']:
        assert command['dependencies']==[]
        assert '--job-name=lukanc_'+command['task_id'] in command['command']
        assert not any(x.startswith(('--dependency','--nice','--array','--hold')) for x in command['command'])
    assert resources.path_key('D1')==resources.path_key('D2')==resources.path_key('D3')
    assert len({resources.path_key(t) for t in pc.TASKS})==4


def test_conditioning_shuffle_respects_class_and_coarse_observables():
    labels=np.array([0,0,1,1,0]); vectors=np.ones((5,4,3),np.float32); mask=np.ones((5,1,3),bool)
    vectors[4]*=1e5; ix=pt.conditioning_permutation(dict(labels=labels,vectors=vectors,mask=mask))
    np.testing.assert_array_equal(ix,[1,0,3,2,4])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('task',['D1','D5','S1','S5'])
def test_CUDA_full128_partial2_with_production_width(tmp_path,monkeypatch,task):
    import importlib
    from hlt_classification.jetclass2_delphes import model as native
    production=importlib.reload(native).model_config
    monkeypatch.setattr(native,'model_config',production)
    spec,tr,_,bank=fixture(tmp_path,monkeypatch,rows=130); source=pt.candidates(spec,task,tr)
    m=model.new_model(spec['imported'],task).cuda(); source.initialize(m); opt=training.optimizer(m)
    for ix in (np.arange(128),np.arange(128,130)):
        losses,_=training.step(m,opt,tr.batch(ix),bank.targets[ix],'cuda',candidates=source,indices=ix,epoch=1,position=7.,probe=False)
        assert np.isfinite(losses).all()
    del m,opt; torch.cuda.empty_cache()


def test_C_resource_acceptance_covers_four_paths_and_frozen_target_review(tmp_path,monkeypatch):
    spec,tr,va,bank=fixture(tmp_path,monkeypatch,rows=130)
    monkeypatch.setattr(resources,'miniature_rows',lambda x:np.arange(130))
    monkeypatch.setattr(resources,'peaks',lambda d:dict(gpu_peak_bytes=1,peak_rss_bytes=1))
    monkeypatch.setattr(resources,'installed_environment',lambda:{})
    monkeypatch.setattr(resources,'gpu_identity',lambda:spec['qualification']['acceptance']['gpu'])
    write_immutable_json(tmp_path/'setup/baseline_parity.json',resources.baseline(spec,tr,'cpu'))
    acceptance=resources.measure(spec,tr,va,bank,'cpu')
    assert len(acceptance['measurements'])==4 and set(acceptance['minutes'])==set(pc.TASKS)
    assert acceptance['probabilistic_review']['sets']['OFFLINE']['maximum_count']==12
    assert acceptance['probabilistic_targets']==pt.authenticate(spec)['content_hash']
    assert resources.accepted(spec)==acceptance
    changed=deepcopy(acceptance); changed['probabilistic_targets']='wrong'
    import json
    (tmp_path/'setup/acceptance.json').write_text(json.dumps(with_content_hash(changed)))
    with pytest.raises(ValueError,match='target parent'): resources.accepted(spec)


def test_C_create_check_parent_immutability_and_quota(tmp_path,monkeypatch):
    import sys,json
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    _,tr,_,_=spec_fixture(tmp_path)
    parent=tmp_path/'native_parent.json'; parent.write_text('{}')
    old=dict(scope='group_A_only',content_hash='parent',root=str(tmp_path/'old'),project_dir=str(tmp_path/'oldcode'),
        environment_dir=sys.prefix,bundle_root=str(tmp_path/'bundle'),reference=dict(environment={}),
        imported=dict(node={'initialization_seed':1,'sampler_seed':2}),anchor=dict(evidence=dict(prepared=tr.foundation_sha256)),
        parent_spec=file_ref(parent),qualification={},raw_root=str(tmp_path/'raw'),raw_hash='r',
        delta_root=str(tmp_path/'delta'),delta_hash='d',embedding_hash='e',key_hash='k',
        reused={'C0':{'proof':0},'C1':{'proof':1},'J5':{'proof':5}})
    original=tmp_path/'group_a.json'; original.write_text(json.dumps(old)); initial=original.read_bytes()
    monkeypatch.setattr(group_c,'original',lambda *a:old)
    monkeypatch.setattr(campaign,'current',lambda *a:dict(content_hash='source'))
    monkeypatch.setattr(group_c.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='',stderr=''))
    monkeypatch.setattr(group_c.reuse,'check',lambda *a,**k:None); monkeypatch.setattr(inputs,'raw',lambda *a:None)
    with pytest.raises(ValueError,match='eight'): group_c.create(tmp_path/'C',original,tmp_path/'code','f'*40,4)
    spec=group_c.create(tmp_path/'C',original,tmp_path/'code','f'*40,8)
    assert spec['scope']=='group_C_only' and spec['remaining']==list(pc.TASKS)
    assert set(spec['reused'])=={'C0','C1'} and spec['contract'].endswith('PROBABILISTIC_SPEC/v1')
    group_c.check(spec); assert original.read_bytes()==initial
    with pytest.raises(ValueError,match='fresh'): group_c.create(tmp_path/'C',original,tmp_path/'code','f'*40,8)
    spec['remaining'].append('Q1')
    with pytest.raises(ValueError): group_c.check(spec)


def test_C_submission_journals_tokens_exact_setup_and_idempotence(tmp_path,monkeypatch):
    import json
    from hlt_classification.scouting import hcwdl_exact_dag_submission as dag
    spec,_,_,_=fixture(tmp_path,monkeypatch); spec['project_dir']='/code'; spec['reused']={'C0':{},'C1':{}}
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None); monkeypatch.setattr(submission,'check_site',lambda *a:None)
    commands=[]
    def sbatch(args,**kwargs):
        commands.append(args); return SimpleNamespace(returncode=0,stdout=str(9000+len(commands))+'\n',stderr='')
    monkeypatch.setattr(dag.subprocess,'run',sbatch)
    submission.submit(spec,'setup'); plan=submission.plan(spec,'setup')
    with pytest.raises(PermissionError): submission.submit(spec,'setup',execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    submission.submit(spec,'setup',execute=True,plan_hash=plan['content_hash'],authorization=pc.AUTHORIZATION)
    accepted=dict(content_hash='accepted',minutes={t:60 for t in pc.TASKS},probabilistic_review=dict(targets='frozen'))
    (tmp_path/'setup/acceptance.json').write_text(json.dumps(accepted)); monkeypatch.setattr(worker,'completed_setup',lambda *a:accepted)
    monkeypatch.setattr(submission,'accounting',lambda *a:['RUNNING','0:0'])
    with pytest.raises(PermissionError,match='COMPLETED'): submission.submit(spec,'science')
    monkeypatch.setattr(submission,'accounting',lambda *a:['COMPLETED','0:0'])
    submission.submit(spec,'science'); expected=submission.plan(spec,'science',accepted)
    one=submission.submit(spec,'science',execute=True,plan_hash=expected['content_hash'],authorization=pc.AUTHORIZATION)
    two=submission.submit(spec,'science',execute=True,plan_hash=expected['content_hash'],authorization=pc.AUTHORIZATION)
    assert one==two and len(commands)==8
    assert list(submission.accepted_jobs(spec,'science'))==list(pc.TASKS)
    assert all(not any(x.startswith('--dependency') for x in command) for command in commands)


def test_C_worker_runs_only_registered_C_rows(tmp_path,monkeypatch):
    spec,tr,va,bank=fixture(tmp_path,monkeypatch); spec['remaining']=['S1']
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(worker,'require_backend',lambda:None); monkeypatch.setattr(worker,'allocation',lambda *a:'444')
    monkeypatch.setattr(worker,'installed_environment',lambda:{})
    monkeypatch.setattr(worker,'gpu_identity',lambda:spec['qualification']['acceptance']['gpu'])
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:{'S1':'444'}); monkeypatch.setattr(worker,'completed_setup',lambda *a:None)
    monkeypatch.setattr(inputs,'hlt',lambda s,role:tr if role=='train' else va)
    value=worker.run(spec,'S1')
    assert value['job_id']=='444' and (tmp_path/'S1/fit/epoch_30.json').exists()
    assert not (tmp_path/'P1').exists() and not (tmp_path/'Q1').exists()
    with pytest.raises(ValueError): worker.run(spec,'Q1')


def test_C_setup_prepares_once_and_has_no_A_B_target_preparation(tmp_path,monkeypatch):
    spec,tr,va,bank=spec_fixture(tmp_path); spec.update(group_c.registration()); calls=[]
    monkeypatch.setattr(inputs,'Bundle',lambda *a:SimpleNamespace(verify_all=lambda:calls.append('inputs')))
    monkeypatch.setattr(inputs,'hlt',lambda s,role:tr if role=='train' else va); monkeypatch.setattr(inputs,'embedding_bank',lambda *a:bank)
    monkeypatch.setattr(pt,'prepare',lambda *a:calls.append('prepare') or dict(preparation_seconds=9999.))
    monkeypatch.setattr(resources,'measure',lambda *a,**kw:calls.append(kw))
    worker.setup(spec,'cpu')
    assert calls[:2]==['inputs','prepare'] and 0<=calls[2]['shared_preparation_seconds']<9999
    assert not (tmp_path/'setup/hlt_keys.json').exists() and not (tmp_path/'setup/structured_targets.json').exists()


def test_preparation_rejects_changed_embedding_lineage_and_insufficient_disk(tmp_path,monkeypatch):
    spec,tr,_,bank=spec_fixture(tmp_path); spec.update(group_c.registration(),counts=dict(train=22,validation=22)); (tmp_path/'setup').mkdir()
    raw=fake_raw(spec,tr); monkeypatch.setattr(inputs,'raw',lambda *a:raw)
    bad=deepcopy(bank); bad.manifest['content_hash']='wrong'
    with pytest.raises(ValueError,match='lineage'): pt.prepare(spec,tr,bad)
    monkeypatch.setattr(pt.shutil,'disk_usage',lambda *a:SimpleNamespace(free=0))
    with pytest.raises(OSError,match='headroom'): pt.prepare(spec,tr,bank)
    assert not (tmp_path/'setup/probabilistic_targets.json').exists()


@pytest.mark.parametrize('change',['mean','std','target'])
def test_density_nonfinite_required_values_fail_closed(change):
    score=dict(mean=torch.zeros(1,1,128),std=torch.ones(1,1,128),logits=torch.zeros(1,1)); target=torch.zeros(1,128)
    if change=='target': target[0,0]=float('nan')
    else: score[change].flatten()[0]=float('nan')
    with pytest.raises(ValueError): pm.loss(score,dict(targets=target),'D')


def test_set_nonfinite_real_inputs_fail_and_all_absent_group_is_zero():
    head=pm.SetHead(); head.configure(5); b=set_batch(2,4); h=torch.randn(2,128)
    b['valid'][...,:5]=False; scores=head(h,b); _,groups,_=pm.set_rows(scores,b)
    assert (groups[:,0]==0).all()
    b['noisy'][0,0,0]=float('nan')
    with pytest.raises(ValueError): head(h,b)


def test_source_normalization_uses_valid_full_TRAIN_continuous_only(tmp_path,monkeypatch):
    spec,tr,_,_=fixture(tmp_path,monkeypatch); source=pt.candidates(spec,'S1',tr)
    values=np.asarray(source.values); cols=np.array(pc.CONTINUOUS)
    np.testing.assert_allclose(values[:,cols].mean(0),0.,atol=1e-6)
    assert np.isin(values[:,5:14],[0.,1.]).all() and np.isin(values[:,18:],[0.,1.]).all()
    norm=load_json(tmp_path/'setup/set_OFFLINE.json')
    assert norm['support']==[len(values)]*22 and norm['tokens']==len(values)


def test_target_bank_retains_tail_particle_and_no_fixed_crop(tmp_path,monkeypatch):
    spec,tr,_,bank=spec_fixture(tmp_path); spec.update(group_c.registration(),counts=dict(train=22,validation=22)); (tmp_path/'setup').mkdir()
    p=particles(n=22,counts=[87]+[3]*21); p.identities=tr.identities; p.labels=tr.labels
    class Raw:
        def shards(self,source,hlt):
            yield dict(identities=p.identities,labels=p.labels,offsets=p.offsets,values=p.values)
    monkeypatch.setattr(inputs,'raw',lambda *a:Raw()); pt.prepare(spec,tr,bank)
    source=pt.SetTargets(spec,tr,'OFFLINE'); batch,_=source.batch(np.array([0,1]),1,c.setting('S1'))
    assert source.capacity['maximum_count']==87 and batch['real'][0].sum()==87
    assert batch['valid'][0,86].all() and batch['noisy'][0,86].any()

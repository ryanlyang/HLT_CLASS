"""Group B target semantics, frozen lineage, native parity and independent release."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from test_luka_next_supervision_hydra import compact_native,particles,spec_fixture,InlineExecutor
from hlt_classification.luka_next_supervision_hydra import contracts as c,structured_contracts as sc
from hlt_classification.luka_next_supervision_hydra import questions as q,vocabulary as v,structured_model as sm
from hlt_classification.luka_next_supervision_hydra import structured_targets as st,group_b,model,training,inputs,resources,submission,worker,campaign
from hlt_classification.luka_next_supervision_hydra import reporting
from hlt_classification.luka_next_supervision_hydra.parallel import ParallelCandidates
from hlt_classification.data.cache_contracts import load_json,write_immutable_json,with_content_hash
from hlt_classification.luka_embedding_aux.model import AuxiliaryParT


def geometry(n=22,count=12,source='OFFLINE'):
    p=particles(n=n,counts=count); rows=[]
    for a,b in zip(p.offsets[:-1],p.offsets[1:]): rows.append(q.measured(p.values[a:b],np.array([0.,0.])))
    return q.Geometry(p.identities,p.labels,p.offsets,**{k:np.concatenate([x[k] for x in rows]) for k in rows[0]},source=source).check()


def catalog(g,per_family=16):
    proposal,scales,_=q.propose(g,per_family=per_family,design_jets=8)
    stats,_=st.statistics(g,proposal)
    arrays,coverage=q.freeze(proposal,scales,stats,goal=per_family)
    return arrays,coverage


def fixture(tmp_path,rows=22):
    spec,tr,va,bank=spec_fixture(tmp_path,rows)
    spec.update(group_b.registration(),counts=dict(train=rows,validation=rows),root=str(tmp_path),
        remaining=list(sc.TASKS),
        reference=dict(environment={}),qualification=dict(acceptance=dict(gpu=dict(name='fixture',total_memory_bytes=8*2**30))))
    return spec,tr,va,bank


class FakeStructured:
    def batch(self,ix,epoch,cfg,mine=None):
        rng=np.random.default_rng(11); n=len(ix)
        if cfg['family']=='V':
            values=rng.random((n,1 if cfg['count_only'] else 2,128)).astype(np.float32); values/=values.sum(-1,keepdims=True)
            return dict(targets=values,prior=np.full_like(values,1/128)),[]
        families=np.repeat(cfg['question_families'],64//len(cfg['question_families']))
        return dict(questions=rng.normal(size=(n,64,31)).astype(np.float32),targets=rng.random((n,64)).astype(np.float32),
            families=np.broadcast_to(families,(n,64)).copy(),applicable=np.ones((n,64),bool),prior=np.full((n,64),.5,np.float32)),[]


def test_registry_exact_nine_rows_and_shared_recipe_unchanged():
    assert sc.TASKS==('Q1','V1','Q2','V2','Q5','V6','Q3','V7','Q6')
    assert len(c.ORDER)==40 and len(sc.ARMS)==9
    assert c.recipe()['implemented_group']=='A' and sc.recipe()['embedding_weight']==5.
    assert c.recipe()['content_hash']=='d39c6ee09d803622385834756bb57c131ed8a8142659b1d99aec248ec6b6bddc'
    assert c.setting('Q5')['source']=='HLT' and c.setting('V6')['source']=='HLT'
    assert c.setting('Q6')['question_families']==[0,1] and c.setting('V7')['count_only']
    assert c.setting('Q2')['seed_index']==c.setting('V2')['seed_index']==1
    assert c.setting('Q3')['coefficient']==.1
    with pytest.raises(NotImplementedError): c.setting('S1')


def test_geometry_tracking_validity_and_hlt_axis():
    p=particles(n=1,counts=3); raw=p.values.copy()
    raw[:,6]=[-.2,.3,.4]; raw[:,8]=[.1,0.,.2]; raw[:,10:]=1
    raw[2,10]=0
    a=q.measured(raw,np.array([.15,.3]))
    np.testing.assert_array_equal(a['valid'][:,2],[True,False,False])
    np.testing.assert_allclose(a['displacement'][:,2],[2.,0.,0.])
    np.testing.assert_allclose(a['eta'],np.arcsinh(raw[:,2]/np.hypot(raw[:,0],raw[:,1]))-.15)
    assert a['fraction'].sum()==pytest.approx(1.)
    g=geometry(); g.role='validation'
    with pytest.raises(ValueError): g.check()


def test_hand_computed_four_question_families_and_pair_charge():
    ids=np.zeros((1,32),np.uint8)
    g=q.Geometry(ids,np.array([0],np.int64),np.array([0,3],np.int64),np.array([.5,.3,.2]),
        np.array([0.,.1,.3]),np.zeros(3),np.array([1,-1,0],np.int8),np.array([0,0,2],np.int8),
        np.array([[.2,0.,2.,0.],[.4,0.,4.,0.],[0.,0.,0.,0.]]),
        np.array([[1,1,1,1],[1,1,1,1],[0,0,0,0]],bool),'OFFLINE').check()
    qs=np.array([
        [0,0,3,4,0,.2,0,1.01,0,0,0,0,0],
        [1,6,3,0,0,1,0,1.01,.1,2,0,0,0],
        [2,6,3,0,0,1,0,1.01,.3,0,0,0,0],
        [3,6,3,0,0,1,0,1.01,.1,0,.05,.2,2],
        [3,6,3,0,0,1,0,1.01,.1,0,.05,.2,1],
    ],np.float64)
    np.testing.assert_allclose(q.answers(g,0,qs,pair_chunk=1,question_chunk=2),[.8,1.,.3,1.,0.])
    perm=np.array([2,0,1]); changed=deepcopy(g)
    for key in ('fraction','eta','phi','charge','category','displacement','valid'): setattr(changed,key,getattr(g,key)[perm])
    np.testing.assert_array_equal(q.answers(g,0,qs),q.answers(changed,0,qs,pair_chunk=1))


def test_semantic_queries_have_no_catalog_index_and_known_width():
    g=geometry(); proposed,scales,_=q.propose(g,per_family=16,design_jets=8)
    values=q.semantic(proposed,scales)
    assert values.shape==(64,31) and np.isfinite(values).all()
    np.testing.assert_array_equal(q.semantic(proposed[::-1],scales),values[::-1])
    assert len(np.unique(proposed,axis=0))==64


def test_catalog_keeps_single_rare_positive_and_reports_constant_shortfall():
    proposed=np.array([[f,6,3,4 if f==0 else 0,0,1,0,1.01,0,0,0,1,0] for f in range(4)],np.float64)
    statistics=dict(rows=10000,sum=np.array([1.,0.,9999.,10000.]),square_sum=np.array([1.,0.,9999.,10000.]),
        minimum=np.array([0.,0.,0.,1.]),maximum=np.array([1.,0.,1.,1.]))
    arrays,coverage=q.freeze(proposed,dict(radius=1.,separation=2.,displacement=[1.]*4),statistics)
    np.testing.assert_array_equal(arrays['applicable'],[True,False,True,False])
    assert arrays['prior'][0]==pytest.approx(.0001) and coverage[0]['nonconstant']==1
    assert all(r['retained']==1 for r in coverage)


def test_query_stream_is_source_answer_label_and_chunk_independent():
    g=geometry(n=22); arrays,_=catalog(g); a=q.QuestionCandidates(g,arrays,'h')
    h=deepcopy(g); h.source='HLT'; h.labels=h.labels[::-1].copy(); h.displacement*=0
    b=q.QuestionCandidates(h,arrays,'h'); indices=np.array([2,0,9])
    one,_=a.batch(indices,2,c.setting('Q1')); other,_=b.batch(indices,2,c.setting('Q5'))
    np.testing.assert_array_equal(one['questions'],other['questions'])
    split=[a.batch([int(i)],2,c.setting('Q1'))[0] for i in indices]
    for key in one: np.testing.assert_array_equal(one[key],np.concatenate([x[key] for x in split]))
    next_epoch,_=a.batch(indices,3,c.setting('Q1'))
    assert not np.array_equal(one['questions'],next_epoch['questions'])


def test_parallel_question_targets_equal_serial_transport():
    g=geometry(); arrays,_=catalog(g); source=q.QuestionCandidates(g,arrays,'h')
    indices=np.array([0,4,7,11,19,2,3,5,6,8]); direct,evidence=source.batch(indices,1,c.setting('Q1'))
    pool=ParallelCandidates(source,executor=InlineExecutor()); actual,other=pool.batch(indices,1,c.setting('Q1'))
    for key in direct: np.testing.assert_array_equal(direct[key],actual[key])
    assert evidence==other; pool.close()


def test_dictionary_transform_has_equal_block_energy_and_assignment_ties():
    p=particles(n=22); scales=v.block_scales(p.features); x=v.transform(p.features,scales)
    for block in v.BLOCKS: assert np.mean((x[:,block].astype(np.float64)**2).sum(-1))==pytest.approx(1/3,rel=1e-6)
    centers=np.zeros((2,22),np.float32); codes,distance=v.assign(p.features,centers,scales,chunk=7)
    assert not codes.any() and np.isfinite(distance).all()
    other,_=v.assign(p.features,centers,scales,chunk=128); np.testing.assert_array_equal(codes,other)


def test_dictionary_ignores_class_labels_and_remains_frozen():
    p=particles(n=22); before=torch.get_rng_state().clone(); a,_=v.fit(p,size=4,design_jets=8)
    p.labels=p.labels[::-1].copy(); b,_=v.fit(p,size=4,design_jets=8)
    for key in a: np.testing.assert_array_equal(a[key],b[key])
    assert torch.equal(before,torch.get_rng_state())
    center=a['centers'].copy(); v.assign(p.features,center,a['block_scales']); np.testing.assert_array_equal(center,a['centers'])
    p.role='validation'
    with pytest.raises(PermissionError): v.fit(p,size=4)


def test_duplicate_dictionary_centers_are_a_diagnostic_not_a_quality_failure():
    import warnings
    p=particles(n=22); p.features[:]=0.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dictionary,diagnostic=v.fit(p,size=4,design_jets=8)
    assert diagnostic['duplicate_centers']==3 and np.isfinite(dictionary['centers']).all()
    codes,distances=v.assign(p.features,dictionary['centers'],dictionary['block_scales'])
    assert not codes.any() and not distances.any()


def test_frequency_count_and_momentum_hand_calculation():
    p=particles(n=2,counts=3); codes=np.array([0,0,1,1,2,2]); pt=np.array([1.,2.,3.,2.,3.,5.])
    f=v.frequencies(p,codes,pt,size=4)
    np.testing.assert_allclose(f[0,0],[2/3,1/3,0,0]); np.testing.assert_allclose(f[0,1],[.5,.5,0,0])
    np.testing.assert_allclose(f[1,1],[0,.2,.8,0]); assert np.allclose(f.sum(-1),1)
    with pytest.raises(ValueError): v.frequencies(p,codes,np.zeros(6),size=4)


def test_structured_losses_have_declared_reductions_and_detached_targets():
    logits=torch.zeros(2,64,requires_grad=True); targets=torch.rand(2,64,requires_grad=True)
    batch=dict(targets=targets,families=torch.arange(4).repeat_interleave(16)[None].expand(2,-1),applicable=torch.ones(2,64,dtype=torch.bool))
    value=sm.loss(logits,batch,'Q'); assert value.item()==pytest.approx(np.log(2))
    value.backward(); assert targets.grad is None and logits.grad is not None
    f=torch.full((2,2,128),1/128,requires_grad=True); scores=torch.zeros(2,2,128,requires_grad=True)
    loss=sm.loss(scores,dict(targets=f),'V'); assert loss.item()==pytest.approx(np.log(128))
    loss.backward(); assert f.grad is None
    batch['applicable'][:]=False
    assert sm.loss(logits,batch,'Q').item()==0.


@pytest.mark.parametrize('task',list(sc.TASKS))
def test_native_initialization_rng_target_independence_and_gradient(tmp_path,task):
    spec,tr,_,bank=fixture(tmp_path); cfg=c.setting(task)
    torch.manual_seed(c.seed(spec['imported'],cfg,'initialization_seed')); old=AuxiliaryParT(); state=torch.get_rng_state().clone()
    student=model.new_model(spec['imported'],task)
    assert torch.equal(state,torch.get_rng_state())
    assert all(torch.equal(x,old.native_state()[k]) for k,x in student.deployment_state().items())
    assert all(torch.equal(x,old.predictor.state_dict()[k]) for k,x in student.predictor.state_dict().items())
    raw=tr.batch(np.arange(3)); student.capture=True; student(**{k:torch.from_numpy(raw[k]) for k in ('features','vectors','mask')})
    batch,_=FakeStructured().batch([0,1,2],1,cfg); batch={k:torch.as_tensor(x) for k,x in batch.items()}
    before=student.auxiliary(batch); changed=dict(batch,targets=torch.ones_like(batch['targets']))
    torch.testing.assert_close(before,student.auxiliary(changed),rtol=0,atol=0)
    model.auxiliary_loss(before,batch,cfg['family']).backward()
    assert student.jet.grad_fn is not None
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in student.mod.parameters())
    assert all(k.startswith('mod.') for k in student.deployment_state())


@pytest.mark.parametrize('task',list(sc.TASKS))
def test_all_nine_production_fit_paths_export_native_and_replay(tmp_path,task):
    spec,tr,va,bank=fixture(tmp_path); source=FakeStructured()
    report=training.fit(tmp_path/task,spec,task,tr,va,bank,candidates=source,device='cpu',technical=True)
    training.validate_fit(tmp_path/task,spec,task,technical=True)
    assert report['parents']['recipe']==sc.recipe()['content_hash']
    assert report['parameter_counts']['trainable']>report['parameter_counts']['deployed']
    assert report['epochs']==1 and report['deployable_hlt_only']


def test_group_b_plan_has_only_nine_independent_interleaved_science_jobs(tmp_path):
    spec,_,_,_=fixture(tmp_path); spec.update(project_dir='/code',remaining=list(sc.TASKS),reused={'C0':{},'C1':{}})
    a=dict(content_hash='accept',minutes={t:60 for t in sc.TASKS},structured_review=dict(catalog='frozen'))
    plan=submission.plan(spec,'science',a)
    assert plan['scope']=='group_B_only' and plan['authorization']==sc.AUTHORIZATION
    assert [x['task_id'] for x in plan['commands']]==list(sc.TASKS)
    for row in plan['commands']:
        assert row['dependencies']==[]
        assert '--job-name=lukanb_'+row['task_id'] in row['command']
        assert not any(s.startswith(('--dependency','--hold','--array','--nice','--nodelist')) for s in row['command'])
    assert len({resources.path_key(t) for t in sc.TASKS})==6


def fake_raw(spec,train):
    sources={}
    for source in ('HLT','OFFLINE'):
        p=particles(n=len(train.labels),counts=12,source=source)
        p.identities=train.identities; p.labels=train.labels
        if source=='HLT': p.values[:,6]*=.5
        sources[source]=p
    class Raw:
        def shards(self,source,hlt):
            p=sources[source]
            for start,end in ((0,len(p.labels)//2),(len(p.labels)//2,len(p.labels))):
                if start==end: continue
                a,b=p.offsets[start],p.offsets[end]
                yield dict(identities=p.identities[start:end],labels=p.labels[start:end],offsets=p.offsets[start:end+1]-a,values=p.values[a:b].copy())
        def load(self,source,hlt):
            p=sources[source]
            from hlt_classification.luka_next_supervision_hydra.data import Particles
            return Particles(dict(identities=p.identities,labels=p.labels,offsets=p.offsets,values=p.values.copy()),source=source,parent_hash=spec['raw_hash'])
    return Raw()


def test_real_target_preparation_streams_fits_freezes_and_rejects_corruption(tmp_path,monkeypatch):
    spec,tr,_,_=fixture(tmp_path); (tmp_path/'setup').mkdir()
    monkeypatch.setattr(inputs,'raw',lambda *a:fake_raw(spec,tr))
    record=st.prepare(spec,tr); assert record['role']=='train'
    assert st.authenticate(spec)==record
    qsource=st.candidates(spec,'Q1',tr); vsource=st.candidates(spec,'V1',tr)
    batch,_=qsource.batch([0,1],1,c.setting('Q1')); assert batch['questions'].shape==(2,64,31)
    values,_=vsource.batch([0,1],1,c.setting('V1')); assert values['targets'].shape==(2,2,128)
    assert record['dictionaries']['HLT']!=record['dictionaries']['OFFLINE']
    path=tmp_path/'setup/dictionary_OFFLINE.npz'
    with path.open('ab') as handle: handle.write(b'corrupt')
    with pytest.raises(ValueError): st.authenticate(spec)


def test_measured_group_b_setup_acceptance_covers_all_six_paths(tmp_path,monkeypatch):
    spec,tr,va,bank=fixture(tmp_path,rows=130); (tmp_path/'setup').mkdir()
    monkeypatch.setattr(inputs,'raw',lambda *a:fake_raw(spec,tr))
    # Bound the catalog in this synthetic execution test; the production
    # proposal budget and full TRAIN algorithm are separately registered/tested.
    propose=q.propose
    monkeypatch.setattr(q,'propose',lambda g:propose(g,per_family=16,design_jets=8))
    st.prepare(spec,tr)
    def candidates(spec,task,train,device):
        value=st.candidates(spec,task,train)
        return ParallelCandidates(value,executor=InlineExecutor()) if c.setting(task)['family']=='Q' else value
    monkeypatch.setattr(inputs,'candidates',candidates)
    monkeypatch.setattr(resources,'miniature_rows',lambda x:np.arange(130))
    monkeypatch.setattr(resources,'peaks',lambda d:dict(gpu_peak_bytes=1,peak_rss_bytes=1))
    monkeypatch.setattr(resources,'installed_environment',lambda:{})
    monkeypatch.setattr(resources,'gpu_identity',lambda:spec['qualification']['acceptance']['gpu'])
    write_immutable_json(tmp_path/'setup/baseline_parity.json',resources.baseline(spec,tr,'cpu'))
    acceptance=resources.measure(spec,tr,va,bank,'cpu')
    assert len(acceptance['measurements'])==6 and set(acceptance['minutes'])==set(sc.TASKS)
    assert resources.accepted(spec)==acceptance
    changed=deepcopy(acceptance); changed['structured_targets']='wrong'
    (tmp_path/'setup/acceptance.json').write_text(json.dumps(with_content_hash(changed)))
    with pytest.raises(ValueError,match='target parent'): resources.accepted(spec)


@pytest.mark.parametrize('task',['Q1','V1'])
def test_full30_budget_with_finite_poor_targets(tmp_path,task):
    spec,tr,va,bank=fixture(tmp_path)
    class Constant(FakeStructured):
        def batch(self,*args,**kwargs):
            result,info=super().batch(*args,**kwargs)
            if task=='Q1': result['applicable'][:]=False
            else: result['targets'][:]=1/128
            return result,info
    report=training.fit(tmp_path/task,spec,task,tr,va,bank,candidates=Constant(),device='cpu')
    assert len(report['validation_history'])==30
    assert all(np.isfinite(r['total_loss']) for r in report['validation_history'])


def test_group_b_submission_requires_its_own_authorization_and_exact_setup(tmp_path,monkeypatch):
    spec,_,_,_=fixture(tmp_path); spec['project_dir']='/code'
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None); monkeypatch.setattr(submission,'check_site',lambda *a:None)
    value=submission.plan(spec,'setup'); submission.submit(spec,'setup')
    with pytest.raises(PermissionError): submission.submit(spec,'setup',execute=True,plan_hash=value['content_hash'],authorization=c.AUTHORIZATION)
    monkeypatch.setattr(worker,'completed_setup',lambda *a:dict(content_hash='accept',minutes={t:60 for t in sc.TASKS}))
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:{'setup':'123'})
    monkeypatch.setattr(submission,'accounting',lambda *a:['RUNNING','0:0'])
    with pytest.raises(PermissionError,match='COMPLETED'): submission.submit(spec,'science')
    assert not (tmp_path/'submission_science').exists()


def test_frozen_catalog_rejects_semantic_encoding_and_prior_tampering():
    g=geometry(); arrays,coverage=catalog(g); proposals,scales,_=q.propose(g,per_family=16,design_jets=8)
    record=dict(role='train',source='OFFLINE',quality_gate=False,rows=len(g.labels),coverage=coverage,scales=scales)
    q.validate_catalog(arrays,record)
    bad=deepcopy(arrays); bad['encoded'][0,0]+=1
    with pytest.raises(ValueError,match='encoding'): q.validate_catalog(bad,record)
    bad=deepcopy(arrays); bad['prior'][0]=.7
    with pytest.raises(ValueError,match='prior'): q.validate_catalog(bad,record)


def test_catalog_proposals_avoid_globally_impossible_charge_categories():
    g=geometry(); proposed,_,_=q.propose(g,per_family=128,design_jets=8)
    assert np.isin(proposed[:,1],[0,6]).all()
    assert np.isin(proposed[:,2],[2,3]).all()  # Fixture has positively charged category0 only.


def test_nearby_query_audit_handles_numpy_inverse_shape_and_constant_answers():
    g=geometry(n=8,count=1); arrays,_=catalog(g,per_family=8)
    result=q.audit(g,arrays,np.arange(8))
    assert len(result['families'])==4 and result['nearby_query_pairs']
    assert all(row['correlation'] is None or np.isfinite(row['correlation']) for row in result['nearby_query_pairs'])


def test_question_resource_rows_include_target_tails_and_native_extremes():
    g=geometry(n=300,count=3)
    # Make the last sources longer without changing native cache ordering.
    last=deepcopy(g); sizes=np.full(300,3,np.int64); sizes[-5:]=20
    last.offsets=np.r_[0,sizes.cumsum()]
    for name in ('fraction','eta','phi','charge','category','displacement','valid'):
        base=getattr(g,name); chunks=[]
        for i,size in enumerate(sizes):
            values=np.resize(base[g.offsets[i]:g.offsets[i+1]],(int(size),*base.shape[1:]))
            if name=='fraction': values[:]=1/size
            chunks.append(values)
        setattr(last,name,np.concatenate(chunks))
    last.check(); arrays,_=catalog(g); source=q.QuestionCandidates(last,arrays,'h')
    selected=st.miniature_indices(source,np.arange(130))
    assert set(range(130))<=set(selected) and set(range(295,300))<=set(selected[:128])
    assert len(selected)==len(set(selected)) and 2<=len(selected)%128<128


def test_group_b_create_check_and_group_a_parent_are_immutable(tmp_path,monkeypatch):
    import sys
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    _,tr,_,_=fixture(tmp_path)
    parent=tmp_path/'native_parent.json'; parent.write_text('{}')
    old=dict(scope='group_A_only',content_hash='parent',root=str(tmp_path/'old'),project_dir=str(tmp_path/'oldcode'),
        environment_dir=sys.prefix,bundle_root=str(tmp_path/'bundle'),reference=dict(environment={}),
        imported=dict(node={'initialization_seed':1,'sampler_seed':2}),anchor=dict(evidence=dict(prepared=tr.foundation_sha256)),
        parent_spec=file_ref(parent),qualification={},raw_root=str(tmp_path/'raw'),raw_hash='r',
        delta_root=str(tmp_path/'delta'),delta_hash='d',embedding_hash='e',key_hash='k',
        reused={'C0':{'proof':0},'C1':{'proof':1},'J5':{'proof':5}})
    original=tmp_path/'group_a.json'; original.write_text(json.dumps(old)); initial=original.read_bytes()
    monkeypatch.setattr(group_b,'original',lambda *a:old)
    monkeypatch.setattr(campaign,'current',lambda *a:dict(content_hash='source'))
    monkeypatch.setattr(group_b.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='',stderr=''))
    monkeypatch.setattr(group_b.reuse,'check',lambda *a,**k:None)
    monkeypatch.setattr(inputs,'raw',lambda *a:None)
    root=tmp_path/'B'; spec=group_b.create(root,original,tmp_path/'code','f'*40,4)
    assert spec['scope']=='group_B_only' and spec['remaining']==list(sc.TASKS)
    assert set(spec['reused'])=={'C0','C1'} and spec['contract'].endswith('STRUCTURED_SPEC/v1')
    group_b.check(spec)
    assert original.read_bytes()==initial
    with pytest.raises(ValueError,match='fresh'): group_b.create(root,original,tmp_path/'code','f'*40,4)
    spec['remaining'].append('P1')
    with pytest.raises(ValueError): group_b.check(spec)


def test_group_b_worker_publishes_only_its_new_fit(tmp_path,monkeypatch):
    spec,tr,va,bank=fixture(tmp_path); spec['remaining']=['Q1']
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(worker,'require_backend',lambda:None); monkeypatch.setattr(worker,'allocation',lambda *a:'321')
    monkeypatch.setattr(worker,'installed_environment',lambda:{})
    monkeypatch.setattr(worker,'gpu_identity',lambda:spec['qualification']['acceptance']['gpu'])
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:{'Q1':'321'})
    monkeypatch.setattr(worker,'completed_setup',lambda *a:None)
    monkeypatch.setattr(inputs,'hlt',lambda s,role:tr if role=='train' else va)
    monkeypatch.setattr(inputs,'embedding_bank',lambda *a:bank)
    monkeypatch.setattr(inputs,'candidates',lambda *a:FakeStructured())
    value=worker.run(spec,'Q1')
    assert value['job_id']=='321' and (tmp_path/'Q1/fit/epoch_30.json').exists()
    assert not (tmp_path/'P1').exists()
    with pytest.raises(ValueError): worker.run(spec,'P1')


def test_group_b_setup_prepares_once_and_does_not_charge_full_preparation_per_fit(tmp_path,monkeypatch):
    spec,tr,va,bank=fixture(tmp_path); calls=[]
    monkeypatch.setattr(inputs,'Bundle',lambda *a:SimpleNamespace(verify_all=lambda:calls.append('inputs')))
    monkeypatch.setattr(inputs,'hlt',lambda s,role:tr if role=='train' else va)
    monkeypatch.setattr(inputs,'embedding_bank',lambda *a:bank)
    monkeypatch.setattr(st,'prepare',lambda *a:calls.append('prepare') or dict(preparation_seconds=9999.))
    monkeypatch.setattr(resources,'measure',lambda *a,**kw:calls.append(kw))
    worker.setup(spec,'cpu')
    assert calls[:2]==['inputs','prepare']
    assert 0<=calls[2]['shared_preparation_seconds']<9999
    assert not (tmp_path/'setup/hlt_keys.json').exists()


def test_group_b_real_submission_journals_are_idempotent_without_fit_dependencies(tmp_path,monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as dag
    spec,_,_,_=fixture(tmp_path); spec['project_dir']='/code'; spec['reused']={'C0':{},'C1':{}}
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None); monkeypatch.setattr(submission,'check_site',lambda *a:None)
    commands=[]
    def sbatch(args,**kwargs):
        commands.append(args); return SimpleNamespace(returncode=0,stdout=str(8000+len(commands))+'\n',stderr='')
    monkeypatch.setattr(dag.subprocess,'run',sbatch)
    submission.submit(spec,'setup'); plan=submission.plan(spec,'setup')
    submission.submit(spec,'setup',execute=True,plan_hash=plan['content_hash'],authorization=sc.AUTHORIZATION)
    accepted=dict(content_hash='accepted',minutes={t:60 for t in sc.TASKS},structured_review=dict(catalog='frozen'))
    (tmp_path/'setup').mkdir(); (tmp_path/'setup/acceptance.json').write_text(json.dumps(accepted))
    monkeypatch.setattr(worker,'completed_setup',lambda *a:accepted)
    monkeypatch.setattr(submission,'accounting',lambda *a:['COMPLETED','0:0'])
    plan=submission.submit(spec,'science'); expected=submission.plan(spec,'science',accepted)
    one=submission.submit(spec,'science',execute=True,plan_hash=expected['content_hash'],authorization=sc.AUTHORIZATION)
    two=submission.submit(spec,'science',execute=True,plan_hash=expected['content_hash'],authorization=sc.AUTHORIZATION)
    assert one==two and len(commands)==10
    assert list(submission.accepted_jobs(spec,'science'))==list(sc.TASKS)
    assert all(not any(x.startswith('--dependency') for x in command) for command in commands)


def test_group_b_partial_results_are_read_only_and_never_substitute_seed0_control(tmp_path,monkeypatch,capsys):
    from hlt_classification.cms_proxy_ladder.contracts import sha256_file
    n=50000; labels=np.arange(n,dtype=np.int64)%11; ids=np.zeros((n,32),np.uint8)
    ids[:,:4]=np.arange(n,dtype=np.int32).view(np.uint8).reshape(n,4)
    noise=np.random.default_rng(18).normal(size=(n,11)); arrays={}; paths={}
    for name,boost in (('M0HLT',.2),('OFFLINE',.8),('matched',.4),('C0',.4),('Q1',.5),('Q2',.5)):
        scores=noise.copy(); scores[np.arange(n),labels]+=boost; values=np.exp(scores-scores.max(-1,keepdims=True)); values/=values.sum(-1,keepdims=True)
        a=dict(identities=ids,labels=labels,probabilities=values.astype(np.float32)); path=tmp_path/(name+'.npz')
        np.savez(path,**a); arrays[name]=a; paths[name]=path
    subset=tmp_path/'subset.npz'; np.savez(subset,indices=np.arange(n,dtype=np.int64))
    def path(name):
        if name=='reference/historical/subset.npz': return subset
        return paths[name.split('/')[1]]
    monkeypatch.setattr(inputs,'Bundle',lambda *a:SimpleNamespace(path=path))
    monkeypatch.setattr(reporting.reuse,'check',lambda *a:paths['C0'])
    monkeypatch.setattr(worker,'receipt',lambda *a:None)
    monkeypatch.setattr(training,'validate_fit',lambda folder,spec,task:(dict(selected_pass=3,content_hash=task),arrays[task]))
    monkeypatch.setattr(reporting,'scheduler_states',lambda *a:{'Q3':['FAILED','1:0']})
    for task in ('Q1','Q2'):
        (tmp_path/task).mkdir(); (tmp_path/task/'receipt.json').write_text('{}')
    spec=dict(group_b.registration(),root=str(tmp_path),bundle_root='unused',counts=dict(train=299234,validation=n),content_hash='s',
        reused=dict(C0=dict(report=dict(selected_pass=2),content_hash='c0',source_mode='pinned_original_worktree')))
    before={str(p):sha256_file(p) for p in tmp_path.rglob('*') if p.is_file()}
    result=reporting.results(spec); output=capsys.readouterr().out
    rows={r['model']:r for r in result['rows'] if r['role']=='validation'}
    assert rows['Q1']['delta_matched']>0 and rows['Q2']['delta_matched'] is None
    assert 'C1' in result['incomplete'] and 'P1' not in result['incomplete'] and 'Group B' in output
    assert before=={str(p):sha256_file(p) for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('task',['Q1','Q6','V1','V7'])
def test_cuda_production_width_structured_full_and_partial(tmp_path,monkeypatch,task):
    import importlib
    from hlt_classification.jetclass2_delphes import model as native
    production=importlib.reload(native).model_config
    monkeypatch.setattr(native,'model_config',production)
    spec,tr,_,bank=fixture(tmp_path,rows=130); student=model.new_model(spec['imported'],task).cuda()
    opt=training.optimizer(student)
    for ix in (np.arange(128),np.arange(128,130)):
        losses,details=training.step(student,opt,tr.batch(ix),bank.targets[ix],'cuda',candidates=FakeStructured(),
            indices=ix,epoch=1,position=7.,probe=ix[0]==0,measure=True)
        assert np.isfinite(losses).all() and set(details['phase_seconds'])==set(training.PHASE_KEYS)
    del student,opt; torch.cuda.empty_cache()

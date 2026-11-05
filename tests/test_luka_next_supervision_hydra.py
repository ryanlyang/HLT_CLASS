"""Group A scientific kernels, native parity, provenance and independent release."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from test_luka_attention import cache,parent_nodes
from hlt_classification.jetclass2_delphes import model as native
from hlt_classification.luka_embedding_aux.model import AuxiliaryParT
from hlt_classification.luka_next_supervision_hydra import contracts as c,data,sampling,model as m,training as t
from hlt_classification.luka_next_supervision_hydra import inputs,resources,submission,worker,campaign,reporting,reuse
from hlt_classification.luka_next_supervision_hydra.parallel import ParallelCandidates
from hlt_classification.data.cache_contracts import load_json,write_immutable_json,with_content_hash


@pytest.fixture(autouse=True)
def compact_native(monkeypatch):
    old=torch.get_num_threads(); torch.set_num_threads(1)
    cfg=native.model_config(); cfg.update(embed_dims=[128],pair_embed_dims=[4],num_heads=8,num_layers=8,num_cls_layers=1,fc_params=[])
    monkeypatch.setattr(native,'model_config',lambda:deepcopy(cfg))
    yield
    torch.set_num_threads(old)


def particles(n=440,source='OFFLINE',counts=40):
    ids=np.zeros((n,32),np.uint8); ids[:,:8]=np.arange(n,dtype=np.int64).view(np.uint8).reshape(n,8)
    labels=np.arange(n,dtype=np.int64)%11
    sizes=np.full(n,counts,np.int64) if np.ndim(counts)==0 else np.asarray(counts,np.int64)
    offsets=np.r_[0,sizes.cumsum()].astype(np.int64); values=[]
    for i,size in enumerate(sizes):
        rng=np.random.default_rng(i); pt=rng.uniform(.5,10,size); phi=rng.uniform(-.2,.2,size)
        p4=np.column_stack((pt*np.cos(phi),pt*np.sin(phi),pt*rng.normal(0,.1,size),pt*1.1))
        track=rng.normal(0,.1,(size,4)); track[:,2:]=abs(track[:,2:])+.01
        values.append(np.column_stack((p4,np.ones(size),np.zeros(size),track,np.ones((size,4)))))
    return data.Particles(dict(identities=ids,labels=labels,offsets=offsets,values=np.concatenate(values)),source=source,parent_hash='b'*64)


def spec_fixture(tmp_path,rows=22):
    tr=cache(coordinate='D000',rows=rows); va=cache(coordinate='D000',role='validation',rows=rows)
    bank=SimpleNamespace(identities=tr.identities,labels=tr.labels,targets=np.zeros((rows,128),np.float32),manifest={'content_hash':'b'*64})
    spec=dict(campaign.registration(),root=str(tmp_path),content_hash='a'*64,imported=dict(node=parent_nodes()[0]),
        bundle_root=str(tmp_path/'bundle'),
        counts=dict(train=rows,validation=rows),embedding_hash='b'*64,raw_hash='c'*64,key_hash='d'*64,
        anchor=dict(evidence=dict(prepared=tr.foundation_sha256)),reused={},remaining=['P1'])
    return spec,tr,va,bank


def test_registry_global_order_scope_and_weights():
    assert len(c.ORDER)==40 and len(set(c.ORDER))==40 and len(c.TASKS)==24
    assert c.TASKS[:3]==('P1','G1','J1')
    assert sum(t.startswith('P') for t in c.TASKS)==10
    assert sum(t.startswith('G') for t in c.TASKS)==6
    assert sum(t.startswith('J') for t in c.TASKS)==6
    assert c.recipe()['embedding_weight']==5
    assert c.coefficient(c.setting('P4'),1.5)==.5
    for value in (-1,float('nan'),float('inf')):
        with pytest.raises(ValueError): c.coefficient(c.setting('P1'),value)
    with pytest.raises(NotImplementedError): c.setting('Q1')


def test_physical_columns_and_parent_local_invariance():
    p=particles(); raw=p.values[:40].copy(); x=data.descriptors(raw)
    angle=1.7; rotated=raw.copy()
    rotated[:,0]=raw[:,0]*np.cos(angle)-raw[:,1]*np.sin(angle)
    rotated[:,1]=raw[:,0]*np.sin(angle)+raw[:,1]*np.cos(angle)
    np.testing.assert_allclose(x,data.descriptors(rotated),atol=1e-6)
    scaled=raw.copy(); scaled[:,:4]*=3
    np.testing.assert_allclose(x,data.descriptors(scaled),atol=1e-6)
    assert x.shape==(40,22) and np.isfinite(p.features).all()
    assert np.all(p.scale>=.001) and p.role=='train'
    raw[:,10:]=0; raw[:,6:10]=999
    missing=data.descriptors(raw)
    assert not missing[:,12:].any()
    with pytest.raises(PermissionError): data.Particles(dict(identities=p.identities,labels=p.labels,offsets=p.offsets,values=p.values),role='validation',source='OFFLINE',parent_hash='x')


@pytest.mark.parametrize('law',['M','G','S','K','F'])
def test_particle_pool_full_quota_exclusion_caps_and_replay(law):
    p=particles(); sampler=sampling.MembershipSampler(p,np.arange(440)+1,np.diff(p.offsets))
    a=sampler.particle_pool(3,1,law); b=sampler.particle_pool(3,1,law)
    np.testing.assert_array_equal(a['indices'],b['indices'])
    assert a['labels'].sum()==32 and len(a['indices'])==500 and len(np.unique(a['indices']))==500
    negative=a['indices'][~a['labels']]; parents=p.parents[negative]
    assert 3 not in parents and np.unique(parents,return_counts=True)[1].max()<=16
    if law=='S': assert np.all(p.labels[parents]==p.labels[3])
    assert not np.array_equal(a['indices'],sampler.particle_pool(3,2,law)['indices'])


def test_authenticated_event_exclusion_and_low_multiplicity():
    sizes=np.full(440,40); sizes[0]=1; p=particles(counts=sizes)
    events=np.arange(440)//2; sampler=sampling.MembershipSampler(p,np.ones(440),sizes,event_groups=events)
    a=sampler.particle_pool(0,1,'M'); assert a['labels'].sum()==1
    assert not set(p.parents[a['indices'][~a['labels']]])&{0,1}
    g=sampler.group_pool(0,1,3); assert g['indices'].shape==(128,1) and g['labels'].sum()==1


@pytest.mark.parametrize('g',[2,3])
def test_intact_group_pool_uniform_members_dedup_and_permutation(g):
    p=particles(counts=12); s=sampling.MembershipSampler(p,np.ones(440),np.diff(p.offsets))
    pool=s.group_pool(3,1,g); rows=pool['indices']
    assert rows.shape==(128,g) and pool['labels'].sum()==16
    assert len({tuple(v) for v in rows})==128
    assert all(len(set(p.parents[v]))==1 for v in rows)
    neg=p.parents[rows[~pool['labels'],0]]
    assert 3 not in neg and np.unique(neg,return_counts=True)[1].max()<=8
    np.testing.assert_array_equal(rows,s.group_pool(3,1,g)['indices'])
    batch,_=s.batch([3],1,c.setting('G4' if g==3 else 'G1'))
    assert batch['features'].shape==(1,128,g,22)
    r=data.relations(p.local[rows],np.ones(rows.shape,bool))
    np.testing.assert_allclose(r,data.relations(p.local[rows[:,::-1]],np.ones(rows.shape,bool)),atol=1e-6)


def test_semi_hard_ranks_detached_scores_and_parent_cap():
    p=particles(n=880,counts=40); s=sampling.MembershipSampler(p,np.ones(880),np.diff(p.offsets)); calls=[]
    def score(features): calls.append(len(features)); return features[:,0]
    a=s.particle_pool(0,1,'Hsemi',score)
    assert calls==[4096] and len(a['indices'])==500 and a['evidence']['band']==[257,1024]
    assert np.unique(p.parents[a['indices'][~a['labels']]],return_counts=True)[1].max()<=16


def test_exact_original_jet_candidates_even_hlt_control():
    n=5500; rng=np.random.default_rng(7); keys=rng.normal(size=(n,128)).astype(np.float32)
    keys/=np.linalg.norm(keys,axis=1,keepdims=True)
    ids=np.zeros((n,32),np.uint8); ids[:,:8]=np.arange(n,dtype=np.int64).view(np.uint8).reshape(n,8)
    labels=np.arange(n,dtype=np.int64)%11; pt=np.ones(n); count=np.full(n,3)
    a=sampling.JetCandidates(keys,ids,labels,pt,count)
    old=sampling.JetSampler(keys,ids,labels,pt=pt,count=count)
    ix,_=old.sample([0,1],1,'mix'); b,_=a.batch([0,1],1,c.setting('J1'))
    np.testing.assert_array_equal(b['features'],keys[ix])
    changed=sampling.JetCandidates(keys,ids,labels,pt,count,source_keys=-keys)
    h,_=changed.batch([0,1],1,c.setting('J6')); np.testing.assert_array_equal(h['features'],-keys[ix])
    assert not torch.as_tensor(b['features']).requires_grad


def test_population_capacity_proof_and_group_support():
    p=particles(n=2200); s=sampling.MembershipSampler(p,np.ones(2200),np.diff(p.offsets))
    assert s.capacity_proof('P')['bounds']['1']['minimum_class_capacity']>=2547
    assert s.capacity_proof('G')['bounds']['3']['minimum_class_capacity']>=127
    tiny=particles(counts=1); weak=sampling.MembershipSampler(tiny,np.ones(440),np.ones(440))
    with pytest.raises(ValueError,match='capacity'): weak.capacity_proof('P')


class InlineExecutor:
    """Exercise chunk transport on Windows; real Linux fork remains remote qualification."""
    def submit(self,fn,*args):
        from concurrent.futures import Future
        result=Future(); result.set_result(fn(*args)); return result
    def map(self,fn,jobs): return map(fn,jobs)
    def shutdown(self,**kw): self.closed=True


@pytest.mark.parametrize('task',['P1','G1','G4','P9'])
def test_chunked_cpu_targets_equal_serial_and_batched_mining_order(task):
    p=particles(n=2200,counts=40); source=sampling.MembershipSampler(p,np.arange(2200)+1,np.diff(p.offsets))
    indices=np.array([0,42,15,8,64,301,98,401,73,2100]); cfg=c.setting(task)
    before=torch.get_rng_state().clone()
    direct,info=source.batch(indices,2,cfg,mine=lambda j,f:f[:,0]*(j+1))
    executor=InlineExecutor(); pool=ParallelCandidates(source,executor=executor)
    def mine(js,features):
        assert features.shape==(len(js),4096,22)
        return features[:,:,0]*(js[:,None]+1)
    chunked,evidence=pool.batch(indices,2,cfg,mine=mine)
    for key in direct: np.testing.assert_array_equal(direct[key],chunked[key])
    assert info==evidence and torch.equal(before,torch.get_rng_state())
    pool.close(); assert executor.closed


def test_raw_transfer_train_only_relocation_and_corrupt_bytes(tmp_path,monkeypatch):
    from hlt_classification.luka_u100_pair_transfer import bundle as transfer
    from hlt_classification.luka_fullsim.particles import RawJet
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    p=particles(n=22,counts=4); rows=[]
    # Stored ROOT field order differs from canonical physical columns.
    for i in range(22):
        physical=p.values[p.offsets[i]:p.offsets[i+1]]
        fields=np.zeros((4,16)); fields[:,:5]=physical[:,:5]; fields[:,5]=1
        fields[:,10:14]=physical[:,6:10]
        rows.append(RawJet(bytes(p.identities[i]).hex(),int(p.labels[i]),i//11,i,fields,fields.copy()))
    checkpoint=tmp_path/'M0.pt'; checkpoint.write_bytes(b'opaque-native-state')
    old=with_content_hash(dict(anchor=dict(evidence=dict(prepared='p',models=dict(M0HLT=dict(checkpoint=file_ref(checkpoint))))),
        root=str(tmp_path/'original'),source_snapshot=dict(content_hash='s')))
    origin=tmp_path/'origin.json'; origin.write_text(json.dumps(old))
    counts=dict(train=22,validation=11); conventions=dict(content_hash='c')
    monkeypatch.setattr(data,'COUNTS',counts)
    monkeypatch.setattr(transfer,'authenticate',lambda *a:dict(prepared='p'))
    monkeypatch.setattr(data.alljets_data,'load_prepared',lambda *a:(dict(content_hash='p',counts=counts,conventions=conventions,
        summaries=dict(train={coord:dict(particles=88) for coord in ('D000','OFFLINE')})),
        dict(input_container=str(tmp_path/'container')),{},dict(memberships=dict(train=[1,2]))))
    def reader(*args,**kwargs):
        assert kwargs['role']=='train' and kwargs['include_offline'] is True
        return iter(rows)
    monkeypatch.setattr(data.alljets_data,'ParticleReader',reader)
    from hlt_classification.cms2jc2_response.bridge import Particles
    def convert(raw,*args):
        return Particles(raw[:,:4],raw[:,4],np.zeros(4),raw[:,10:14],np.ones((4,4),bool),tuple(map(str,range(4))))
    monkeypatch.setattr(data,'physical',convert)
    out=tmp_path/'transfer'; result=data.export(tmp_path/'prepared',origin,out)
    bundle=data.RawBundle(out,result['manifest_hash'])
    source=bundle.load('OFFLINE',SimpleNamespace(role='train',identities=p.identities,labels=p.labels))
    np.testing.assert_array_equal(source.values,p.values)
    assert result['payload_bytes']>0 and len(bundle.rows)==5
    target=bundle.path('OFFLINE/0000.npz'); target.write_bytes(b'changed')
    with pytest.raises(ValueError): bundle.path('OFFLINE/0000.npz')


def test_group_independent_member_control_same_exact_targets():
    p=particles(counts=12); s=sampling.MembershipSampler(p,np.ones(440),np.diff(p.offsets))
    a,_=s.batch([0],1,c.setting('G1')); b,_=s.batch([0],1,c.setting('G5'))
    for key in a: np.testing.assert_array_equal(a[key],b[key])


@pytest.mark.parametrize('task',['P1','G1','G4','G5','J1','J5','C0','C1'])
def test_native_initialization_rng_gradients_and_head_removal(task):
    imported=dict(node=parent_nodes()[0]); torch.manual_seed(c.seed(imported,c.setting(task),'initialization_seed'))
    old=AuxiliaryParT().eval(); rng=torch.get_rng_state().clone()
    student=m.new_model(imported,task).eval()
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,old.native_state()[k]) for k,v in student.deployment_state().items())
    assert all(torch.equal(v,old.predictor.state_dict()[k]) for k,v in student.predictor.state_dict().items())
    raw=cache(coordinate='D000').batch(np.arange(4)); values={k:torch.from_numpy(raw[k]) for k in ('features','vectors','mask')}
    student.capture=old.capture=True
    assert torch.equal(student(**values),old(**values))
    assert torch.equal(student.prediction,old.prediction)
    restored=m.deployment(student.deployment_state(),'cpu')
    assert torch.equal(restored(**values),old(**values))
    assert all(k.startswith('mod.') for k in student.deployment_state())
    if task.startswith('C'): return
    cfg=c.setting(task); n=7; g=cfg['group_size']
    shape=(4,n,128) if cfg['family']=='J' else (4,n,g,22) if cfg['family']=='G' else (4,n,22)
    features=torch.randn(shape,requires_grad=True)
    batch=dict(features=features)
    if cfg['family']=='G': batch.update(member_mask=torch.ones((4,n,g),dtype=torch.bool),relations=torch.randn(4,n,g*(g-1)//2,10))
    scores=student.auxiliary(batch)
    scores.square().mean().backward()
    assert features.grad is None
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in student.mod.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in student.aux.parameters())
    perm=torch.randperm(n); other={k:v[:,perm] for k,v in batch.items()}
    torch.testing.assert_close(student.auxiliary(other),scores[:,perm],rtol=1e-6,atol=1e-6)
    if cfg['family']=='G':
        other=dict(batch); other['features']=features.flip(-2)
        torch.testing.assert_close(student.auxiliary(other),scores,rtol=1e-6,atol=1e-6)


def test_balanced_membership_loss_and_mask_reductions():
    scores=torch.tensor([[0.,1.,2.],[-1.,0.,1.]],requires_grad=True); labels=torch.tensor([[1,0,0],[1,1,0]],dtype=torch.bool)
    individual=F.binary_cross_entropy_with_logits(scores,labels.float(),reduction='none')
    expected=(.5*individual[0,0]+.5*individual[0,1:].mean()+.5*individual[1,:2].mean()+.5*individual[1,2])/2
    assert torch.allclose(m.membership_loss(scores,labels),expected)
    with pytest.raises(ValueError): m.membership_loss(scores,torch.ones_like(labels))
    with pytest.raises(ValueError): m.membership_loss(scores*float('nan'),labels)


@pytest.mark.parametrize('task',['G1','G5'])
def test_singleton_masked_padding_has_no_output_or_gradient_effect(task):
    head=m.Compatibility(c.setting(task)); h=torch.randn(2,128,requires_grad=True)
    features=torch.randn(2,4,2,22); mask=torch.zeros(2,4,2,dtype=torch.bool); mask[:,:,0]=True
    relations=torch.zeros(2,4,1,10)
    clean=head(h,features,mask,relations)
    features[:,:,1]=float('nan'); relations[:]=float('nan')
    other=head(h,features,mask,relations)
    torch.testing.assert_close(clean,other,rtol=0,atol=0)
    other.square().mean().backward()
    assert torch.isfinite(h.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)


def test_original_u100_one_update_exact_parity(tmp_path):
    from hlt_classification.luka_u100_embedding.training import objective
    from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
    spec,tr,_,bank=spec_fixture(tmp_path)
    torch.manual_seed(spec['imported']['node']['initialization_seed'])
    old=AuxiliaryParT(); old.capture=True; old.train(); original_opt=t.optimizer(old)
    student=m.new_model(spec['imported'],'C0'); opt=t.optimizer(student)
    ix=np.arange(4); raw=tr.batch(ix); state=torch.get_rng_state().clone()
    for group in original_opt.param_groups: group['lr']=learning_rate(.5)
    logits=t.forward(old,raw,'cpu'); labels=torch.as_tensor(raw['labels'])
    total,_,_=objective(logits,labels,old.prediction,torch.as_tensor(bank.targets[ix]),coefficient=5.)
    total.backward(); original_opt.step()
    torch.set_rng_state(state)
    losses,_=t.step(student,opt,raw,bank.targets[ix],'cpu',candidates=None,indices=ix,epoch=1,position=.5)
    assert losses[-1]==float(total.detach())
    assert student.state_dict().keys()==old.state_dict().keys()
    assert all(torch.equal(v,old.state_dict()[k]) for k,v in student.state_dict().items())


class FakeCandidates:
    def batch(self,ix,epoch,cfg,mine=None):
        rng=np.random.default_rng(9); n=len(ix)
        if cfg['family']=='J': return dict(features=rng.normal(size=(n,1025,128)).astype(np.float32),positive=np.zeros(n,np.int64)),[]
        size=128 if cfg['family']=='G' else 500
        labels=np.zeros((n,size),bool); labels[:,:min(16,size)]=1
        if cfg['family']=='G':
            g=cfg['group_size']; batch=dict(features=rng.normal(size=(n,size,g,22)).astype(np.float32),labels=labels,
                member_mask=np.ones((n,size,g),bool),relations=np.zeros((n,size,g*(g-1)//2,10),np.float32))
        else: batch=dict(features=rng.normal(size=(n,size,22)).astype(np.float32),labels=labels)
        if mine is not None:
            values=mine(0,np.zeros((4096,22),np.float32)); assert values.shape==(4096,)
        return batch,[]


@pytest.mark.parametrize('task',['C0','P1','P9','G1','G4','G5','J1','J5'])
def test_production_objective_fit_native_replay_and_tamper(tmp_path,task):
    spec,tr,va,bank=spec_fixture(tmp_path)
    report=t.fit(tmp_path/'fit',spec,task,tr,va,bank,device='cpu',technical=True,
        candidates=None if task=='C0' else FakeCandidates())
    assert report['epochs']==1 and report['deployable_hlt_only']
    assert report['validation_history'][0]['total_loss']>=0
    t.validate_fit(tmp_path/'fit',spec,task,technical=True)
    epoch=load_json(tmp_path/'fit/epoch_01.json'); epoch['auxiliary_loss']=999
    (tmp_path/'fit/epoch_01.json').write_text(json.dumps(with_content_hash(epoch)))
    with pytest.raises(ValueError): t.validate_fit(tmp_path/'fit',spec,task,technical=True)


def test_full30_budget_and_poor_finite_metrics_not_gate(tmp_path):
    spec,tr,va,bank=spec_fixture(tmp_path)
    report=t.fit(tmp_path/'fit',spec,'C0',tr,va,bank,device='cpu')
    assert len(report['validation_history'])==30
    assert report['selected_pass'] in range(1,31)
    t.validate_fit(tmp_path/'fit',spec,'C0')


def test_resources_do_not_admit_quality_or_unwarmed_timing(tmp_path):
    spec,tr,va,bank=spec_fixture(tmp_path)
    r=dict(task='P1',technical=True,epochs=1,parents=dict(spec=spec['content_hash']),content_hash='r',
        counts=dict(train=130,validation=22),validation_history=[dict(batch_seconds=[.2,.1],batch_sizes=[128,2],validation_seconds=.01)])
    mem=dict(gpu_peak_bytes=1,peak_rss_bytes=1); gpu=dict(total_memory_bytes=8*2**30)
    a=resources.resource_record(spec,'P1',r,r,mem,gpu,1)
    assert a['passed'] and a['quality_gate'] is False
    r['validation_history'][0]['batch_sizes']=[128,1]
    with pytest.raises(ValueError): resources.resource_record(spec,'P1',r,r,mem,gpu,1)
    r['validation_history'][0]['batch_sizes']=[128,2]; r['validation_history'][0]['batch_seconds']=[10000.,1000.]
    assert not resources.resource_record(spec,'P1',r,r,mem,gpu,1)['passed']
    with pytest.raises(ValueError,match='resource measurement'):
        resources.resource_record(spec,'P1',r,r,dict(gpu_peak_bytes=float('nan')),gpu,1)


def test_measured_production_path_acceptance_and_tampering(tmp_path,monkeypatch):
    spec,tr,va,bank=spec_fixture(tmp_path,rows=130)
    spec['remaining']=['P1','P2','P3']; spec['reference']=dict(environment={'fixture':True})
    gpu=dict(name='fixture',total_memory_bytes=8*2**30); spec['qualification']=dict(acceptance=dict(gpu=gpu))
    (tmp_path/'setup').mkdir()
    monkeypatch.setattr(resources,'gpu_identity',lambda:gpu)
    monkeypatch.setattr(resources,'installed_environment',lambda:{'fixture':True})
    monkeypatch.setattr(resources,'miniature_rows',lambda x:np.arange(130))
    monkeypatch.setattr(resources,'peaks',lambda d:dict(gpu_peak_bytes=1,peak_rss_bytes=1))
    source=FakeCandidates(); source.capacity=dict(rows=130,class_counts=np.bincount(tr.labels,minlength=11).tolist(),sampler_workers=4)
    monkeypatch.setattr(inputs,'candidates',lambda *a:source)
    write_immutable_json(tmp_path/'setup/baseline_parity.json',resources.baseline(spec,tr,'cpu'))
    a=resources.measure(spec,tr,va,bank,'cpu')
    assert a['paths']=={'P1':'P1','P2':'P1','P3':'P1'} and len(a['measurements'])==1
    assert resources.accepted(spec)==a
    parity=load_json(tmp_path/'setup/baseline_parity.json'); parity['checks']={}
    (tmp_path/'setup/baseline_parity.json').write_text(json.dumps(with_content_hash(parity)))
    with pytest.raises(ValueError,match='parity'): resources.accepted(spec)


def test_independent_filtered_queue_and_deferred_groups_fail(tmp_path):
    spec,_,_,_=spec_fixture(tmp_path); spec.update(project_dir='/code',reused={'C0':{},'C1':{},'J5':{}})
    spec['remaining']=[t for t in c.TASKS if t not in spec['reused']]
    accepted=dict(content_hash='b',minutes={t:60 for t in spec['remaining']})
    value=submission.plan(spec,'science',accepted)
    assert [r['task_id'] for r in value['commands']]==spec['remaining']
    assert [r['task_id'] for r in value['commands']][:3]==['P1','G1','J1']
    for row in value['commands']:
        assert not row['dependencies']
        assert '--gres=gpu:nvidia_rtx_a6000:1' in row['command']
        assert not any(x.startswith(('--dependency','--nice','--hold','--array','--nodelist')) for x in row['command'])
    assert value['scope']=='group_A_only' and value['deferred_groups']==['B','C']
    with pytest.raises(ValueError): submission.plan(spec,'full40',accepted)


def test_exact_submission_idempotence_and_ambiguous_claim(tmp_path,monkeypatch):
    spec,_,_,_=spec_fixture(tmp_path); spec['project_dir']='/code'
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(submission,'check_site',lambda *a:None)
    value=submission.plan(spec,'setup'); counter=[]
    def submit(**kwargs):
        counter.append(kwargs['execute']); return dict(dry_run=not kwargs['execute'])
    monkeypatch.setattr(submission,'submit_exact_dag',submit)
    submission.submit(spec,'setup')
    assert counter==[False]
    with pytest.raises(PermissionError): submission.submit(spec,'setup',execute=True,authorization='wrong')
    submission.submit(spec,'setup',execute=True,plan_hash=value['content_hash'],authorization=c.AUTHORIZATION)
    assert counter==[False,True]
    with pytest.raises(RuntimeError,match='partial'):
        submission.submit(spec,'setup',execute=True,plan_hash=value['content_hash'],authorization=c.AUTHORIZATION)


def test_complete_submission_is_idempotent_and_science_needs_setup(tmp_path,monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as dag
    spec,_,_,_=spec_fixture(tmp_path); spec['project_dir']='/code'
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None); monkeypatch.setattr(submission,'check_site',lambda *a:None)
    commands=[]
    def sbatch(args,**kw):
        commands.append(args); return SimpleNamespace(returncode=0,stdout='123\n',stderr='')
    monkeypatch.setattr(dag.subprocess,'run',sbatch)
    submission.submit(spec,'setup'); value=submission.plan(spec,'setup')
    one=submission.submit(spec,'setup',execute=True,plan_hash=value['content_hash'],authorization=c.AUTHORIZATION)
    two=submission.submit(spec,'setup',execute=True,plan_hash=value['content_hash'],authorization=c.AUTHORIZATION)
    assert one==two and len(commands)==1 and submission.accepted_jobs(spec,'setup')=={'setup':'123'}
    def unready(*a): raise ValueError('setup incomplete')
    monkeypatch.setattr(worker,'completed_setup',unready)
    with pytest.raises(ValueError,match='incomplete'): submission.submit(spec,'science')


@pytest.mark.parametrize('state',[['FAILED','1:0'],['RUNNING','0:0'],None])
def test_science_rejects_noncompleted_setup_even_with_saved_acceptance(tmp_path,monkeypatch,state):
    spec,_,_,_=spec_fixture(tmp_path); spec['project_dir']='/code'
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(worker,'completed_setup',lambda *a:dict(content_hash='acceptance',minutes={'P1':60}))
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:{'setup':'123'})
    monkeypatch.setattr(submission,'accounting',lambda job:state)
    with pytest.raises(PermissionError,match='COMPLETED'): submission.submit(spec,'science')
    assert not (tmp_path/'submission_science').exists()


def test_create_registration_reuses_controls_and_does_not_publish_on_bad_map(tmp_path,monkeypatch):
    import sys
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    parent=dict(counts=c.COUNTS,root=str(tmp_path/'old'),project_dir=str(tmp_path/'oldcode'),environment_dir=sys.prefix,
        bundle_root=str(tmp_path/'bundle'),reference=dict(environment={}),imported=dict(node=parent_nodes()[0]),
        anchor=dict(evidence=dict(prepared='p',models=dict(M0HLT=dict(checkpoint={'opaque':'ref'})))),content_hash='o')
    parent_path=tmp_path/'parent.json'; parent_path.write_text(json.dumps(parent))
    qualification={'acceptance':{}}; source={'content_hash':'s'}
    monkeypatch.setattr(campaign,'authenticate_parent',lambda *a:(parent,qualification))
    monkeypatch.setattr(campaign,'current',lambda *a:source); monkeypatch.setattr(campaign.site,'environment',lambda *a:None)
    monkeypatch.setattr(campaign,'Bundle',lambda *a:SimpleNamespace(json=lambda *a:dict(content_hash='b')))
    key=tmp_path/'keys.json'; key.write_text(json.dumps(dict(content_hash='k')))
    monkeypatch.setattr(campaign.contrast,'delta',lambda *a:SimpleNamespace(path=lambda *a:key))
    monkeypatch.setattr(campaign.contrast,'bind',lambda *a:None)
    monkeypatch.setattr(inputs,'raw',lambda *a:SimpleNamespace(manifest=dict(original_m0={'opaque':'ref'})))
    monkeypatch.setattr(reuse,'inspect',lambda row,*a:dict(row=row,content_hash=row))
    monkeypatch.setattr(reuse,'transferred_dot',lambda *a:dict(row='J5',content_hash='J5'))
    monkeypatch.setattr(reuse,'check',lambda *a,**k:None)
    args=(parent_path,tmp_path/'raw','r',tmp_path/'delta','d')
    with pytest.raises(ValueError,match='C0/C1'):
        campaign.create(tmp_path/'invalid',*args,{},tmp_path/'code','f'*40,4)
    assert not (tmp_path/'invalid').exists()
    spec=campaign.create(tmp_path/'new',*args,{'C0':str(parent_path),'C1':str(parent_path)},tmp_path/'code','f'*40,4)
    assert set(spec['reused'])=={'C0','C1','J5'} and len(spec['remaining'])==21
    campaign.check(spec)
    changed=deepcopy(spec); changed['science_dependencies']='afterok'
    with pytest.raises(ValueError): campaign.check(with_content_hash(changed))


def test_reuse_completed_dot_from_authenticated_transfer(tmp_path,monkeypatch):
    from hlt_classification.luka_u100_contrastive_hydra import transfer
    from hlt_classification.luka_u100_contrastive import contracts as old
    from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
    from hlt_classification.jetclass2_delphes.runner import selection_key
    from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
    from hlt_classification.data.cache_contracts import atomic_publish_bytes,deterministic_npz_bytes
    spec,tr,_,_=spec_fixture(tmp_path)
    spec.update(delta_root=str(tmp_path),delta_hash='delta',bundle_root='bundle')
    original=dict(content_hash='original',counts=spec['counts'],imported=spec['imported'])
    probabilities=np.full((22,11),1/11,np.float32)
    arrays=dict(identities=tr.identities,labels=tr.labels,probabilities=probabilities)
    prediction=tmp_path/'predictions.npz'; atomic_publish_bytes(prediction,deterministic_npz_bytes(arrays))
    metrics=evaluate_probabilities(tr.labels,probabilities); history=[]; arm='mix_l0p3_s0'
    for e in range(1,31):
        weighted=old.ramp(.3,e,1)*6
        history.append(dict(pass_number=e,update=e,validation=metrics,ce_loss=1.,auxiliary_loss=.1,
            matching_loss=6.,weighted_matching_loss=weighted,weighted_auxiliary_loss=.5,total_loss=1.+5*.1+weighted,
            learning_rate=learning_rate(e),matching_coefficient_end=old.ramp(.3,e,1),train_seconds=.1,validation_seconds=.1))
    selected=max(history,key=lambda r:selection_key(r['validation'],r['update']))['pass_number']
    report=old.artifact('TRAINING_REPORT',parents=dict(spec='original',recipe=old.recipe()['content_hash'],
        prepared=spec['anchor']['evidence']['prepared'],u100=spec['embedding_hash'],offline=spec['key_hash']),
        arm=arm,settings=old.ARMS[arm],technical=False,epochs=30,counts=spec['counts'],resolved_seeds=old.seeds(spec['imported'],arm),
        node=old.node(spec['imported'],arm),final_test_access=False,selected_state='native_HLT_only',
        selection='entire_validation_native_AUC',selected_pass=selected,validation=metrics,validation_history=history)
    (tmp_path/'report.json').write_text(json.dumps(report))
    for name in ('manifest.json','task_receipt.json','fit_receipt.json'): (tmp_path/name).write_text('{}')
    manifest=dict(completed=[arm],origin_spec=original,content_hash='delta',original_replay=dict(completed={arm:dict(job_id='123')}))
    d=SimpleNamespace(root=tmp_path,manifest=manifest,path=lambda name:tmp_path/Path(name).name)
    monkeypatch.setattr(transfer,'Delta',lambda *a:d)
    monkeypatch.setattr(inputs,'Bundle',lambda *a:SimpleNamespace(path=lambda *a:prediction))
    # The reuse function deliberately imports the canonical original Bundle reader.
    from hlt_classification.luka_u100_pair_hydra import inputs as old_inputs
    monkeypatch.setattr(old_inputs,'Bundle',lambda *a:SimpleNamespace(path=lambda *a:prediction))
    proof=reuse.transferred_dot(spec)
    assert proof['row']=='J5' and proof['source_mode']=='transferred_original_SPORC_result'
    assert proof['selected_checkpoint_transferred'] is False and reuse.check(proof)==prediction
    changed=dict(spec,key_hash='different')
    with pytest.raises(ValueError,match='differ'): reuse.transferred_dot(changed)
    manifest['completed']=[]; assert reuse.transferred_dot(spec) is None


def test_baseline_installed_weaver_parity(tmp_path):
    spec,tr,_,_=spec_fixture(tmp_path)
    result=resources.baseline(spec,tr,'cpu')
    assert result['installed_weaver'] and len(result['checks'])==6


def test_exact_science_worker_publishes_native_fit(tmp_path,monkeypatch):
    spec,tr,va,bank=spec_fixture(tmp_path)
    spec['qualification']=dict(acceptance=dict(gpu={'name':'fixture'})); spec['reference']=dict(environment={'fixture':True})
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:spec)
    monkeypatch.setattr(worker,'require_backend',lambda:None)
    monkeypatch.setattr(worker,'allocation',lambda *a:'123')
    monkeypatch.setattr(worker,'gpu_identity',lambda:{'name':'fixture'})
    monkeypatch.setattr(worker,'installed_environment',lambda:{'fixture':True})
    monkeypatch.setattr(worker,'completed_setup',lambda *a:None)
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:{'P1':'123'})
    monkeypatch.setattr(inputs,'hlt',lambda s,role:tr if role=='train' else va)
    monkeypatch.setattr(inputs,'embedding_bank',lambda *a:bank)
    monkeypatch.setattr(inputs,'candidates',lambda *a:FakeCandidates())
    original=t.fit
    monkeypatch.setattr(t,'fit',lambda *a,**kw:original(*a,**dict(kw,device='cpu')))
    r=worker.run(spec,'P1')
    assert r['job_id']=='123' and r['task']=='P1'
    worker.receipt(spec,'P1'); report,a=t.validate_fit(tmp_path/'P1/fit',spec,'P1')
    assert len(report['validation_history'])==30 and len(a['labels'])==22
    with pytest.raises(FileExistsError): worker.run(spec,'P1')
    with pytest.raises(ValueError): worker.run(spec,'Q1')


def test_frozen_hlt_train_keys_exact_rebuild_and_role_guard(tmp_path,monkeypatch):
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    spec,tr,va,_=spec_fixture(tmp_path)
    state=tmp_path/'M0.pt'; torch.save(native.DelphesParticleTransformer().state_dict(),state)
    b=SimpleNamespace(path=lambda n:state,manifest=dict(original_m0=file_ref(state)))
    monkeypatch.setattr(inputs,'raw',lambda *a:b)
    (tmp_path/'setup').mkdir()
    record=inputs.build_hlt_keys(spec,tr,'cpu'); keys=inputs.hlt_keys(spec,tr)
    assert record['role']=='train' and keys.shape==(22,128) and not np.isnan(keys).any()
    for fn in (lambda:inputs.build_hlt_keys(spec,va,'cpu'),lambda:inputs.hlt_keys(spec,va)):
        with pytest.raises(PermissionError): fn()
    other=deepcopy(spec); other['root']=str(tmp_path/'other'); (tmp_path/'other/setup').mkdir(parents=True)
    repeated=inputs.build_hlt_keys(other,tr,'cpu')
    assert repeated==record
    np.testing.assert_array_equal(keys,inputs.hlt_keys(other,tr))


def test_exact_setup_worker_prepares_keys_normalizations_and_receipt(tmp_path,monkeypatch):
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    spec,tr,va,bank=spec_fixture(tmp_path)
    state=tmp_path/'M0.pt'; torch.save(native.DelphesParticleTransformer().state_dict(),state)
    p=particles(n=22,counts=4)
    b=SimpleNamespace(path=lambda n:state,manifest=dict(original_m0=file_ref(state)),load=lambda source,hlt:particles(n=22,counts=4,source=source))
    monkeypatch.setattr(inputs,'raw',lambda *a:b)
    monkeypatch.setattr(inputs,'Bundle',lambda *a:SimpleNamespace(verify_all=lambda:None))
    monkeypatch.setattr(inputs,'hlt',lambda s,role:tr if role=='train' else va)
    monkeypatch.setattr(inputs,'embedding_bank',lambda *a:bank)
    measured=[]
    def measure(s,*args,**kw):
        assert kw['shared_preparation_seconds']>=0
        measured.append(True); write_immutable_json(tmp_path/'setup/acceptance.json',c.artifact('ACCEPTANCE'))
    monkeypatch.setattr(resources,'measure',measure)
    spec['qualification']=dict(acceptance=dict(gpu={'fixture':True})); spec['reference']=dict(environment={'fixture':True})
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:spec)
    monkeypatch.setattr(worker,'require_backend',lambda:None); monkeypatch.setattr(worker,'allocation',lambda *a:'123')
    monkeypatch.setattr(worker,'installed_environment',lambda:{'fixture':True}); monkeypatch.setattr(worker,'gpu_identity',lambda:{'fixture':True})
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:{'setup':'123'})
    original=worker.setup; monkeypatch.setattr(worker,'setup',lambda s:original(s,'cpu'))
    result=worker.run(spec,'setup'); worker.receipt(spec,'setup')
    names={r['path'] for r in result['outputs']}
    assert measured==[True] and {'hlt_keys.json','hlt_keys.npz','baseline_parity.json','normalization_HLT.json','normalization_OFFLINE.json','acceptance.json'}<=names


def test_source_and_task_target_capabilities_fail_closed(tmp_path):
    spec,tr,va,bank=spec_fixture(tmp_path)
    with pytest.raises(PermissionError): inputs.candidates(spec,'P1',va)
    student=m.new_model(spec['imported'],'C0'); opt=t.optimizer(student); raw=tr.batch(np.arange(2))
    raw['source_indices']=np.ones(2)
    with pytest.raises(PermissionError): t.step(student,opt,raw,bank.targets[:2],'cpu',candidates=None,indices=[0,1],epoch=1,position=1)
    with pytest.raises(PermissionError): t.fit(tmp_path/'bad',spec,'C0',va,tr,bank,device='cpu')


def test_progress_read_only_recovery_and_same_seed_control(tmp_path,monkeypatch,capsys):
    spec,tr,va,bank=spec_fixture(tmp_path)
    t.fit(tmp_path/'P1/fit',spec,'P1',tr,va,bank,candidates=FakeCandidates(),device='cpu',technical=True)
    low,high=.2,.9
    monkeypatch.setattr(inputs,'Bundle',lambda *a:SimpleNamespace(json=lambda *a:dict(recovery_anchors={'M0HLT':low,'OFFLINE':high})))
    spec['bundle_root']='unused'
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result=reporting.progress(spec)
    output=capsys.readouterr().out
    assert 'Provisional' in output and 'delta_selected_C0=--' in output
    auc=result['best']['P1']['auc']; assert result['best']['P1']['recovery']==pytest.approx(100*(auc-low)/(high-low))
    after={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert before==after


def test_partial_results_recompute_metrics_keep_failures_and_do_not_write(tmp_path,monkeypatch,capsys):
    from hlt_classification.data.cache_contracts import atomic_publish_bytes,deterministic_npz_bytes,sha256_file
    n=50000; ids=np.zeros((n,32),np.uint8); ids[:,:8]=np.arange(n,dtype=np.int64).view(np.uint8).reshape(n,8)
    labels=np.arange(n,dtype=np.int64)%11; rng=np.random.default_rng(6); logits=rng.normal(size=(n,11))
    arrays={}; files={}
    for name,margin in (('M0HLT',.2),('OFFLINE',1.),('C0',.5),('P1',.6)):
        v=logits.copy(); v[np.arange(n),labels]+=margin; p=np.exp(v-v.max(-1,keepdims=True)); p/=p.sum(-1,keepdims=True)
        arrays[name]=dict(identities=ids,labels=labels,probabilities=p.astype(np.float32))
        files[name]=tmp_path/(name+'.npz'); atomic_publish_bytes(files[name],deterministic_npz_bytes(arrays[name]))
    subset=tmp_path/'subset.npz'; atomic_publish_bytes(subset,deterministic_npz_bytes(dict(indices=np.arange(n,dtype=np.int64))))
    def path(name):
        if 'subset' in name: return subset
        key=name.split('/')[1]; return files['C0' if key=='matched' else key]
    monkeypatch.setattr(inputs,'Bundle',lambda *a:SimpleNamespace(path=path))
    monkeypatch.setattr(reuse,'check',lambda *a:files['C0'])
    monkeypatch.setattr(worker,'receipt',lambda *a:{})
    monkeypatch.setattr(t,'validate_fit',lambda *a:(dict(selected_pass=2,content_hash='p1'),arrays['P1']))
    monkeypatch.setattr(reporting,'scheduler_states',lambda *a:{'P2':['FAILED','1:0']})
    (tmp_path/'P1').mkdir(); (tmp_path/'P1/receipt.json').write_text('{}')
    spec=dict(root=str(tmp_path),bundle_root='unused',counts=dict(train=299234,validation=n),content_hash='s',
        reused=dict(C0=dict(report=dict(selected_pass=3),content_hash='c0',source_mode='pinned_original_worktree')))
    before={str(p):sha256_file(p) for p in tmp_path.rglob('*') if p.is_file()}
    result=reporting.results(spec); output=capsys.readouterr().out
    row=next(r for r in result['rows'] if r['model']=='P1' and r['role']=='validation')
    assert row['delta_matched']>0 and row['recovery']>0
    assert result['states']['P2']==['FAILED','1:0'] and 'P2' in result['incomplete'] and 'FAILED' in output
    assert before=={str(p):sha256_file(p) for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('task',['P1','G4','J1'])
def test_cuda_production_width_full_and_partial(tmp_path,monkeypatch,task):
    # Restore production model config, not the compact autouse fixture.
    import importlib
    original=importlib.reload(native).model_config
    monkeypatch.setattr(native,'model_config',original)
    spec,tr,_,bank=spec_fixture(tmp_path,rows=130)
    student=m.new_model(spec['imported'],task).cuda(); opt=t.optimizer(student)
    for ix in (np.arange(128),np.arange(128,130)):
        loss,_=t.step(student,opt,tr.batch(ix),bank.targets[ix],'cuda',candidates=FakeCandidates(),indices=ix,epoch=1,position=7.)
        assert np.isfinite(loss).all()
    del student,opt; torch.cuda.empty_cache()

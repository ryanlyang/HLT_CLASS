"""Complete measured jets, exact assignment, bounded VJP, native export and Hydra release."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from itertools import permutations
import math
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from test_luka_next_supervision_hydra import compact_native,particles,spec_fixture
from test_luka_next_supervision_group_b import fake_raw
from hlt_classification.luka_embedding_aux.model import AuxiliaryParT
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.luka_next_supervision_hydra import reconstruction_contracts as rc
from hlt_classification.luka_next_supervision_hydra import reconstruction_model as rm,reconstruction_loss as loss
from hlt_classification.luka_next_supervision_hydra import reconstruction_targets as rt,reconstruction_training as kernel,reconstruction
from hlt_classification.luka_next_supervision_hydra import contracts as c,model,training,inputs,resources,submission,worker,campaign


def fixture(tmp_path,monkeypatch,rows=22):
    spec,tr,va,bank=spec_fixture(tmp_path,rows)
    sources={s:dict(rows=rows,tokens=rows*12,maximum_count=12) for s in ('OFFLINE','HLT')}
    inventory=rc.artifact('INVENTORY',parents=dict(raw=spec['raw_hash']),sources=sources,maximum=12,
        storage_budget_bytes=rt.storage_bytes(sources,rows),quality_gate=False)
    spec.update(reconstruction.registration(),counts=dict(train=rows,validation=rows),remaining=list(rc.TASKS),
        target_inventory=inventory,storage_budget_bytes=inventory['storage_budget_bytes'],
        reference=dict(environment={}),qualification=dict(acceptance=dict(gpu=dict(name='fixture',total_memory_bytes=8*2**30))))
    (tmp_path/'setup').mkdir(); raw=fake_raw(spec,tr); monkeypatch.setattr(inputs,'raw',lambda s:raw)
    rt.prepare(spec,tr)
    return spec,tr,va,bank


def target(n=3):
    raw=particles(n=1,counts=max(n,1)).values[:n]
    frame=np.array([.2,.3,20.]); x,valid=rt.measured(raw,frame) if n else (np.empty((0,22)),np.empty((0,22),bool))
    return dict(values=x.astype(np.float32),valid=valid[:,14:18],mean=np.zeros(22),scale=np.ones(22),frame=frame,
        aggregate=np.r_[raw[:,:4].sum(0),raw[raw[:,4]!=0,3].sum()])


def score(k=4):
    g=torch.Generator().manual_seed(13)
    result={name:torch.randn((k,width),generator=g,requires_grad=True) for name,width in
        (('kinematic',3),('category',6),('charge',3),('track',4),('validity',4))}
    result.update(angle=torch.randn(k,generator=g,requires_grad=True),presence=torch.randn(k,generator=g,requires_grad=True),
        count=torch.randn(k+1,generator=g,requires_grad=True))
    return result


def test_registry_is_additive_exact_twelve_weight5_and_six_paths():
    assert len(rc.TASKS)==12 and rc.TASKS[:6]==('R1','R2','R3','R4','R11','R12')
    assert len(c.ORDER)==40 and not any(t.startswith('R') for t in c.ORDER)
    assert c.recipe()['content_hash']=='d39c6ee09d803622385834756bb57c131ed8a8142659b1d99aec248ec6b6bddc'
    assert len({resources.path_key(t) for t in rc.TASKS})==6
    assert rc.recipe()['embedding_weight']==5 and c.authorization(reconstruction.registration())==rc.AUTHORIZATION
    assert c.setting('R3')['seed_index']==c.setting('R4')['seed_index']==1
    assert c.setting('R11')['source']==c.setting('R12')['source']=='HLT'
    assert c.coefficient(c.setting('R1'),1.5)==.05
    with pytest.raises(NotImplementedError): c.setting('R99')


@pytest.mark.parametrize('n',[0,1,3,4])
def test_exact_presence_aware_assignment_equals_exhaustive_full_loss(n):
    s=score(); t=loss.target(target(n),'cpu'); selected,columns=loss.match(s,t)
    positive=F.softplus(-s['presence']); negative=F.softplus(s['presence']); denominator=n+.1*(4-n)
    fields,_=loss.fields(s,t,pairwise=True)
    values=[]
    for slots in permutations(range(4),n):
        present=torch.zeros(4,dtype=torch.bool); present[list(slots)]=True
        field=fields[list(slots),list(range(n))].mean() if n else fields.sum()*0
        values.append(float((field+(positive[present].sum()+.1*negative[~present].sum())/denominator).detach()))
    got,parts=loss.row(s,t)
    assert float((parts['fields']+parts['presence']).detach())==pytest.approx(min(values),abs=2e-6)
    assert len(set(selected))==len(set(columns))==n
    got.backward(); assert s['presence'].grad is not None and torch.isfinite(s['presence'].grad).all()


def test_permutation_duplicate_ties_invalid_sentinels_and_no_cropping():
    s=score(); t=target(); base,_=loss.row(s,loss.target(t,'cpu'))
    changed=dict(t,values=t['values'][[2,0,1]],valid=t['valid'][[2,0,1]])
    v,_=loss.row(s,loss.target(changed,'cpu')); torch.testing.assert_close(v,base)
    permutation=torch.tensor([2,0,3,1]); ss={k:v if k=='count' else v[permutation] for k,v in s.items()}
    v,_=loss.row(ss,loss.target(t,'cpu')); torch.testing.assert_close(v,base)
    duplicate=dict(t,values=np.repeat(t['values'][:1],3,0),valid=np.repeat(t['valid'][:1],3,0))
    assert len(set(loss.match(s,loss.target(duplicate,'cpu'))[0]))==3
    t['valid'][:]=False; t['values'][:,18:22]=0; t['values'][:,14:18]=float('nan')
    got,_=loss.row(s,loss.target(t,'cpu')); assert torch.isfinite(got)
    with pytest.raises(ValueError,match='never crop'): loss.row(s,loss.target(target(5),'cpu'))
    s['presence'].data[0]=float('nan')
    with pytest.raises(ValueError): loss.row(s,loss.target(target(),'cpu'))


def test_matched_fields_equal_group_weighting_and_zero_valid_track():
    t=loss.target(target(1),'cpu'); t['values'][:]=0; t['values'][:,5]=1; t['values'][:,12]=1
    t['valid'][:]=False; s=score(1)
    s={k:torch.zeros_like(v) for k,v in s.items()}; s['angle'].fill_(math.pi/2)
    _,parts=loss.fields(s,t,pairwise=False)
    torch.testing.assert_close(parts['composition'],torch.ones(1))
    torch.testing.assert_close(parts['track'],torch.full((1,),.5*math.log(2)))
    torch.testing.assert_close(parts['kinematic'],torch.full((1,),.1))


def test_consistency_physical_inverse_angular_wrap_signed_mass_and_gradient():
    t=loss.target(target(1),'cpu'); s=score(1)
    s={k:torch.zeros_like(v) for k,v in s.items()}
    s['kinematic']=t['values'][:,:3].clone().requires_grad_(True)
    s['angle']=torch.atan2(t['values'][:,3],t['values'][:,4]).requires_grad_(True)
    s['presence'].fill_(30); s['charge'][:,2]=30
    p=loss.physical(s,t); torch.testing.assert_close(p[0],t['aggregate'][:4],atol=2e-6,rtol=2e-6)
    shifted=dict(s,angle=s['angle']+2*math.pi); torch.testing.assert_close(loss.physical(shifted,t),p,atol=2e-6,rtol=2e-6)
    assert float(loss.consistency(s,t))<1e-10
    s['kinematic']=s['kinematic']+torch.tensor([[.2,-.5,.2]])
    value=loss.consistency(s,t); value.backward(); assert torch.isfinite(value) and s['angle'].grad is not None


@pytest.mark.parametrize('task',['R1','R2'])
def test_decoder_target_blind_fixed_support_padding_and_rng(task):
    state=torch.get_rng_state().clone(); decoder=rm.Decoder(c.setting(task),17)
    assert torch.equal(state,torch.get_rng_state())
    assert not any(isinstance(m,(torch.nn.BatchNorm1d,torch.nn.Dropout)) and getattr(m,'p',0)!=0 for m in decoder.modules())
    h=torch.randn(2,128); tokens=torch.randn(2,5,128); mask=torch.ones(2,5,dtype=torch.bool); mask[:,-1]=False
    a=decoder(h,tokens,mask); bad=tokens.clone(); bad[:,-1]=float('nan'); b=decoder(h,bad,mask)
    assert a['presence'].shape==(2,17) and a['count'].shape==(2,18)
    assert (a['category']==0).all() and (a['charge']==0).all() and (a['count']==0).all()
    torch.testing.assert_close(a['presence'].sigmoid(),torch.full_like(a['presence'],.1))
    for k in a: torch.testing.assert_close(a[k],b[k])
    if task=='R2':
        perm=torch.tensor([2,0,4,1,3]); b=decoder(h,tokens[:,perm],mask[:,perm])
        for k in a: torch.testing.assert_close(a[k],b[k],rtol=2e-5,atol=2e-6)
    with pytest.raises(TypeError): decoder(h,targets=target())


@pytest.mark.parametrize('task',['R1','R2','R9','R10'])
def test_chunked_decoder_backward_matches_full_graph_including_encoder_VJP(task):
    torch.manual_seed(11); decoder=rm.Decoder(c.setting(task),4); reference=deepcopy(decoder)
    encoder=torch.nn.Linear(6,128); ref_encoder=deepcopy(encoder)
    x=torch.randn(3,2,6); ts=encoder(x); h=ts.mean(1); rs=ref_encoder(x); rh=rs.mean(1)
    mask=torch.ones(3,2,dtype=torch.bool); targets=[target(3) for _ in range(3)]
    predictions=reference(rh,rs,mask); values,_=loss.rows(predictions,targets,c.setting(task)['consistency'])
    base=rh.square().mean(); (base+.1*values.mean()).backward()
    actual,_,gh,gt,_=kernel.decode_backward(decoder,h,ts,mask,targets,.1,chunk=2,consistency=c.setting(task)['consistency'])
    outs=[h.square().mean(),h]; grads=[torch.ones(()),gh]
    if gt is not None: outs.append(ts); grads.append(gt)
    torch.autograd.backward(outs,grads)
    assert actual==pytest.approx(float(values.detach().mean()),abs=2e-6)
    torch.testing.assert_close(encoder.weight.grad,ref_encoder.weight.grad,rtol=2e-4,atol=2e-6)
    for (name,a),(_,b) in zip(decoder.named_parameters(),reference.named_parameters()):
        assert a.grad is not None, name
        torch.testing.assert_close(a.grad,b.grad,rtol=3e-4,atol=3e-6,msg=name)


def test_streamed_preparation_source_join_support_aggregates_and_corruption(tmp_path,monkeypatch):
    spec,tr,va,_=fixture(tmp_path,monkeypatch); record=rt.authenticate(spec)
    assert record['maximum']==12 and not (tmp_path/'setup/density_centers.npz').exists()
    source=rt.candidates(spec,'R2',tr); assert isinstance(source.values,np.memmap)
    rows,_=source.batch([0,2],1,c.setting('R2')); assert len(rows[0]['values'])==12
    assert rows[0]['aggregate'].dtype==np.float64 and source.capacity['full_collection']
    with pytest.raises(PermissionError): rt.candidates(spec,'R1',va)
    source.close()
    path=tmp_path/'setup/reconstruction_HLT_valid.npy'; path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError): rt.authenticate(spec)


@pytest.mark.parametrize('task',rc.TASKS)
def test_native_parity_live_particle_gradients_training_and_strict_export(tmp_path,monkeypatch,task):
    spec,tr,va,bank=fixture(tmp_path,monkeypatch); source=rt.candidates(spec,task,tr)
    torch.manual_seed(c.seed(spec['imported'],c.setting(task),'initialization_seed')); original=AuxiliaryParT()
    state=torch.get_rng_state().clone(); m=model.new_model(spec['imported'],task); source.initialize(m)
    assert torch.equal(state,torch.get_rng_state())
    assert all(torch.equal(v,m.deployment_state()[k]) for k,v in original.native_state().items())
    assert all(torch.equal(v,m.predictor.state_dict()[k]) for k,v in original.predictor.state_dict().items())
    opt=training.optimizer(m); native_before={k:v.clone() for k,v in m.mod.state_dict().items()}
    losses,details=training.step(m,opt,tr.batch(np.arange(22)),bank.targets,'cpu',candidates=source,indices=np.arange(22),epoch=1,position=7.)
    assert np.isfinite(losses).all() and details['probes']['auxiliary_jet_gradient_norm']>0
    if c.setting(task)['memory']=='tokens': assert details['probes']['auxiliary_token_gradient_norm']>0
    assert not m.mod.blocks[7]._forward_hooks and not m.mod.blocks[7]._forward_pre_hooks
    assert any(not torch.equal(v,native_before[k]) for k,v in m.mod.state_dict().items())
    report=training.fit(tmp_path/('fit_'+task),spec,task,tr,va,bank,candidates=source,device='cpu',technical=True)
    training.validate_fit(tmp_path/('fit_'+task),spec,task,technical=True)
    assert report['parents']['reconstruction_targets']==rt.authenticate(spec)['content_hash']
    assert all(k.startswith('mod.') for k in m.deployment_state())
    assert not hasattr(model.deployment(m.deployment_state(),'cpu'),'aux')
    source.close()


def test_independent_queue_order_and_review_full_support(tmp_path):
    spec,_,_,_=spec_fixture(tmp_path); spec.update(reconstruction.registration(),remaining=list(rc.TASKS),
        project_dir=str(tmp_path),reused={'C0':{},'C1':{}})
    a=dict(content_hash='e'*64,minutes={t:600 for t in rc.TASKS},reconstruction_review=dict(fixed_output_slots=183))
    p=submission.plan(spec,'science',a)
    assert [r['task_id'] for r in p['commands']]==list(rc.TASKS)
    assert p['reconstruction_target_review']['fixed_output_slots']==183 and p['authorization']==rc.AUTHORIZATION
    for r in p['commands']:
        assert r['dependencies']==[] and '--job-name=lukanr_'+r['task_id'] in r['command']
        assert '--gres=gpu:nvidia_rtx_a6000:1' in r['command']
        assert not any(v.startswith(('--dependency','--hold','--nice','--array')) for v in r['command'])
    with pytest.raises(ValueError): submission.plan(spec,'science')


def test_full30_poor_finite_fit_and_component_validation(tmp_path,monkeypatch):
    spec,tr,va,bank=fixture(tmp_path,monkeypatch,rows=11); source=rt.candidates(spec,'R1',tr)
    va.labels[:]=np.roll(va.labels.copy(),1)
    report=training.fit(tmp_path/'full',spec,'R1',tr,va,bank,candidates=source,device='cpu')
    assert report['epochs']==30 and len(report['validation_history'])==30
    assert report['validation']['macro_ovr_auc']<.99
    training.validate_fit(tmp_path/'full',spec,'R1'); source.close()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('task',['R1','R2','R9','R10','R11','R12'])
def test_CUDA_full128_partial2_production_native_and_decoder(tmp_path,monkeypatch,task):
    import importlib
    from hlt_classification.jetclass2_delphes import model as native
    production=importlib.reload(native).model_config
    monkeypatch.setattr(native,'model_config',production)
    spec,tr,_,bank=fixture(tmp_path,monkeypatch,rows=130); source=rt.candidates(spec,task,tr)
    m=model.new_model(spec['imported'],task).cuda(); source.initialize(m); opt=training.optimizer(m)
    for ix in (np.arange(128),np.arange(128,130)):
        values,_=training.step(m,opt,tr.batch(ix),bank.targets[ix],'cuda',candidates=source,indices=ix,epoch=1,position=7.,probe=False)
        assert np.isfinite(values).all()
    source.close(); del m,opt; torch.cuda.empty_cache()


def test_resource_acceptance_covers_six_paths_and_binds_full_support(tmp_path,monkeypatch):
    spec,tr,va,bank=fixture(tmp_path,monkeypatch,rows=130)
    monkeypatch.setattr(resources,'miniature_rows',lambda x:np.arange(130))
    monkeypatch.setattr(resources,'peaks',lambda d:dict(gpu_peak_bytes=1,peak_rss_bytes=1))
    monkeypatch.setattr(resources,'installed_environment',lambda:{})
    monkeypatch.setattr(resources,'gpu_identity',lambda:spec['qualification']['acceptance']['gpu'])
    write_immutable_json(tmp_path/'setup/baseline_parity.json',resources.baseline(spec,tr,'cpu'))
    acceptance=resources.measure(spec,tr,va,bank,'cpu')
    assert len(acceptance['measurements'])==6 and set(acceptance['minutes'])==set(rc.TASKS)
    assert acceptance['reconstruction_review']['fixed_output_slots']==12
    assert acceptance['reconstruction_targets']==rt.authenticate(spec)['content_hash']
    assert resources.accepted(spec)==acceptance
    import json
    from hlt_classification.data.cache_contracts import with_content_hash
    changed=deepcopy(acceptance); changed['reconstruction_targets']='wrong'
    (tmp_path/'setup/acceptance.json').write_text(json.dumps(with_content_hash(changed)))
    with pytest.raises(ValueError,match='target parent'): resources.accepted(spec)


def test_create_check_inventory_quota_and_original_immutability(tmp_path,monkeypatch):
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
    monkeypatch.setattr(reconstruction,'original',lambda *a:old)
    monkeypatch.setattr(campaign,'current',lambda *a:dict(content_hash='source'))
    monkeypatch.setattr(reconstruction.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='',stderr=''))
    monkeypatch.setattr(reconstruction.reuse,'check',lambda *a,**k:None); monkeypatch.setattr(inputs,'raw',lambda *a:None)
    def inventory(spec):
        n=spec['counts']['train']; sources={s:dict(rows=n,tokens=n*12,maximum_count=12) for s in ('OFFLINE','HLT')}
        return rc.artifact('INVENTORY',parents=dict(raw=spec['raw_hash']),sources=sources,maximum=12,
            storage_budget_bytes=rt.storage_bytes(sources,n),quality_gate=False)
    monkeypatch.setattr(rt,'inventory',inventory)
    with pytest.raises(OSError,match='headroom'): reconstruction.create(tmp_path/'R',original,tmp_path/'code','f'*40,1)
    assert not (tmp_path/'R').exists()
    spec=reconstruction.create(tmp_path/'R',original,tmp_path/'code','f'*40,8)
    assert spec['scope']==rc.SCOPE and spec['remaining']==list(rc.TASKS)
    assert set(spec['reused'])=={'C0','C1'} and spec['contract'].endswith('DIRECT_RECONSTRUCTION_SPEC/v3')
    reconstruction.check(spec); assert original.read_bytes()==initial
    with pytest.raises(ValueError,match='fresh'): reconstruction.create(tmp_path/'R',original,tmp_path/'code','f'*40,8)
    spec['remaining'].append('S1')
    with pytest.raises(ValueError): reconstruction.check(spec)


def test_submission_exact_setup_authorization_idempotence_and_no_dependencies(tmp_path,monkeypatch):
    import json
    from hlt_classification.scouting import hcwdl_exact_dag_submission as dag
    spec,_,_,_=fixture(tmp_path,monkeypatch); spec['project_dir']='/code'; spec['reused']={'C0':{},'C1':{}}
    from hlt_classification.luka_next_supervision_hydra import reconstruction_release
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None); monkeypatch.setattr(submission,'check_site',lambda *a:None)
    monkeypatch.setattr(reconstruction_release,'scheduler_review',lambda *a:None)
    commands=[]
    def sbatch(args,**kwargs):
        commands.append(args); return SimpleNamespace(returncode=0,stdout=str(9000+len(commands))+'\n',stderr='')
    monkeypatch.setattr(dag.subprocess,'run',sbatch)
    submission.submit(spec,'bootstrap'); plan=submission.plan(spec,'bootstrap')
    with pytest.raises(PermissionError): submission.submit(spec,'bootstrap',execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    submission.submit(spec,'bootstrap',execute=True,plan_hash=plan['content_hash'],authorization=rc.AUTHORIZATION)
    accepted=dict(content_hash='accepted',minutes={t:60 for t in rc.TASKS},reconstruction_review=dict(fixed_output_slots=12))
    (tmp_path/'setup/acceptance.json').write_text(json.dumps(accepted)); monkeypatch.setattr(worker,'completed_setup',lambda *a:accepted)
    monkeypatch.setattr(submission,'accounting',lambda *a:['RUNNING','0:0'])
    with pytest.raises(PermissionError,match='COMPLETED'): submission.submit(spec,'science')
    monkeypatch.setattr(submission,'accounting',lambda *a:['COMPLETED','0:0'])
    submission.submit(spec,'science'); expected=submission.plan(spec,'science',accepted)
    one=submission.submit(spec,'science',execute=True,plan_hash=expected['content_hash'],authorization=rc.AUTHORIZATION)
    two=submission.submit(spec,'science',execute=True,plan_hash=expected['content_hash'],authorization=rc.AUTHORIZATION)
    assert one==two and len(commands)==14
    assert list(submission.accepted_jobs(spec,'science'))==list(rc.TASKS)
    assert '--dependency=afterok:9001' in commands[1]
    assert all(not any(x.startswith('--dependency') for x in command) for command in commands[2:])


def test_complete_raw_tail_retained_and_common_K_across_source_controls(tmp_path,monkeypatch):
    spec,tr,_,_=spec_fixture(tmp_path); spec.update(reconstruction.registration(),counts=dict(train=22,validation=22))
    p=particles(n=22,counts=[87]+[3]*21); p.identities=tr.identities; p.labels=tr.labels
    h=particles(n=22,counts=4); h.identities=tr.identities; h.labels=tr.labels
    class Raw:
        def shards(self,source,hlt):
            v=p if source=='OFFLINE' else h
            yield dict(identities=v.identities,labels=v.labels,offsets=v.offsets,values=v.values)
    sources={s:dict(rows=22,tokens=len(v.values),maximum_count=int(np.diff(v.offsets).max())) for s,v in (('OFFLINE',p),('HLT',h))}
    spec['target_inventory']=dict(sources=sources,maximum=87); spec['storage_budget_bytes']=rt.storage_bytes(sources,22)
    (tmp_path/'setup').mkdir(); monkeypatch.setattr(inputs,'raw',lambda *a:Raw()); rt.prepare(spec,tr)
    for task in ('R1','R11'):
        source=rt.candidates(spec,task,tr); batch,_=source.batch([0],1,c.setting(task))
        assert source.maximum==87
        assert len(batch[0]['values'])==(87 if task=='R1' else 4)
        if task=='R1': assert batch[0]['valid'][86].all()
        source.close()


def test_setup_uses_only_R_preparation_and_preserves_allocation_inputs(tmp_path,monkeypatch):
    spec,tr,va,bank=spec_fixture(tmp_path); spec.update(reconstruction.registration()); calls=[]
    monkeypatch.setattr(inputs,'Bundle',lambda *a:SimpleNamespace(verify_all=lambda:calls.append('inputs')))
    monkeypatch.setattr(inputs,'hlt',lambda s,role:tr if role=='train' else va); monkeypatch.setattr(inputs,'embedding_bank',lambda *a:bank)
    monkeypatch.setattr(rt,'prepare',lambda *a:calls.append('prepare'))
    monkeypatch.setattr(resources,'measure',lambda *a,**kw:calls.append(kw))
    worker.setup(spec,'cpu')
    assert calls[:2]==['inputs','prepare']
    assert not (tmp_path/'setup/density_centers.npz').exists() and not (tmp_path/'setup/hlt_keys.json').exists()


def test_assignment_cannot_collapse_targets_onto_one_nearest_slot():
    t=loss.target(target(2),'cpu'); t['values'][1]=t['values'][0]; t['valid'][1]=t['valid'][0]
    s=score(2); s['kinematic'].data[:]=t['values'][0,:3]; s['kinematic'].data[1]+=20
    f,_=loss.fields(s,t,pairwise=True)
    a,b=loss.match(s,t)
    assert len(set(a))==2 and f[a,b].mean()>f.min(0).values.mean()


def test_R_progress_reports_selected_and_same_epoch_same_seed_without_writes(tmp_path,monkeypatch,capsys):
    from hlt_classification.luka_next_supervision_hydra import reporting
    spec,_,_,_=spec_fixture(tmp_path); spec.update(reconstruction.registration())
    def row(ep,auc): return dict(pass_number=ep,validation=dict(macro_ovr_auc=auc))
    spec['reused']={name:dict(report=dict(selected_pass=2,validation_history=[row(1,auc),row(2,auc+.02)]))
        for name,auc in (('C0',.5),('C1',.6))}
    path=tmp_path/'R3/fit'; path.mkdir(parents=True)
    write_immutable_json(path/'epoch_01.json',c.artifact('EPOCH',parents=dict(spec=spec['content_hash']),task='R3',**row(1,.63)))
    monkeypatch.setattr(inputs,'Bundle',lambda *a:SimpleNamespace(json=lambda *a:dict(recovery_anchors={'M0HLT':.2,'OFFLINE':.9})))
    monkeypatch.setattr(reporting,'scheduler_states',lambda *a:{'R4':['FAILED','1:0']})
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    value=reporting.progress(spec); output=capsys.readouterr().out
    assert 'delta_selected_C1=+0.01000000' in output and 'delta_same_epoch_C1=+0.03000000' in output
    assert value['states']['R4']==['FAILED','1:0']
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_CUDA_298_fixed_slots_full128_decoder_backward_without_target_count_input():
    decoder=rm.Decoder(c.setting('R10'),298).cuda()
    h=torch.randn(128,128,device='cuda',requires_grad=True)
    tokens=torch.randn(128,16,128,device='cuda',requires_grad=True); mask=torch.ones(128,16,device='cuda',dtype=torch.bool)
    targets=[target((298,87,3)[i%3]) for i in range(128)]
    value,_,gh,gt,_=kernel.decode_backward(decoder,h,tokens,mask,targets,.1,consistency=True)
    assert math.isfinite(value) and gh.shape==h.shape and gt.shape==tokens.shape
    assert gh.norm()>0 and gt.norm()>0 and decoder.queries.shape==(298,256)
    del decoder,h,tokens,gh,gt; torch.cuda.empty_cache()

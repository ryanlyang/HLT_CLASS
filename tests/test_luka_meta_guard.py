"""Guard isolation, full-state rollback, real meta derivatives and exact queueing."""
from copy import deepcopy
from types import SimpleNamespace
import hashlib
import math
import numpy as np
import pytest
import torch
from hlt_classification.data.cache_contracts import with_content_hash, load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.luka_meta_guard import contracts as c, training as t, policy, campaign, submission, source, worker
from hlt_classification.luka_meta_embedding import model as models, meta


@pytest.fixture(autouse=True)
def threads():
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def identities(n=3200):
    return np.array([list(hashlib.sha256(str(i).encode()).digest()) for i in range(n)],dtype=np.uint8)


def test_split_and_half_blocks_are_identity_stable_disjoint_and_exhaustive():
    ids=identities(); inner,query,guard=t.split(ids)
    old_inner,old_query=t.old.split(ids)
    np.testing.assert_array_equal(inner,old_inner)
    assert set(query)|set(guard)==set(old_query) and not set(query)&set(guard)
    assert len(query)==len(guard)==80
    perm=np.random.default_rng(2).permutation(len(ids)); reordered=t.split(ids[perm])
    for a,b in zip((inner,query,guard),reordered):
        assert {tuple(row) for row in ids[a]}=={tuple(row) for row in ids[perm][b]}
    for b in range(1,21,2):
        one,two=t.order_for(inner,4,b),t.order_for(inner,4,b+1)
        assert not set(one)&set(two) and set(one)|set(two)==set(inner)
    with pytest.raises(ValueError): t.split(ids[:200])


def test_recipe_and_event_budget():
    r=c.recipe(); assert r['ce_weight']==1 and r['embedding_weight']==5 and r['logit_kd'] is False
    assert c.INITIAL_LR==5e-6 and r['warmup'] is False
    assert sum(c.events('meta',b) for b in range(1,21))==475
    assert sum(c.events('fixed',b) for b in range(1,21))==0
    assert [c.events('meta',b) for b in range(17,21)]==[0]*4
    for arm,b in [('bad',1),('fixed',0),('meta',21)]:
        with pytest.raises(ValueError): c.events(arm,b)


def test_guard_tolerance_patience_cap_and_no_failure_for_poor_performance():
    g=policy.Guard(.96)
    assert g.observe(1,.9601)['action']=='keep'
    assert g.observe(2,.9605)['action']=='accept'
    assert g.observe(3,.959)['action']=='keep'
    assert g.observe(4,.9604)['action']=='keep'  # tolerance resets patience
    assert g.observe(5,.959)['action']=='keep'
    r=g.observe(6,.959); assert r['action']=='rollback' and r['restore_block']==2
    assert g.lr==2.5e-6
    for b in range(7,21): g.observe(b,.5)
    assert g.rollbacks==3 and g.lr==6.25e-7
    assert g.observe(21,.2)['action']=='budget_exhausted_continue'
    with pytest.raises(ValueError): g.observe(22,float('nan'))


class Mini(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=torch.nn.Linear(3,8); self.bn=torch.nn.BatchNorm1d(8)
        self.classifier=torch.nn.Linear(8,11); self.predictor=torch.nn.Linear(8,128)
    def forward(self,features,vectors=None,mask=None,*,auxiliary=False):
        h=torch.tanh(self.bn(self.encoder(features))); logits=self.classifier(h)
        return (logits,self.predictor(h)) if auxiliary else logits
    def no_weight_decay(self): return set()
    def native_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items() if not k.startswith('predictor.')}


def test_rollback_restores_student_buffers_predictor_moments_not_adapter():
    m=Mini(); opt=t.old.optimizer_for(m); adapter=models.Adapter(); ao=torch.optim.Adam(adapter.parameters())
    for _ in range(2):
        opt.zero_grad(); m(torch.randn(8,3),auxiliary=True)[1].square().mean().backward(); opt.step()
    anchor=t.old.cpu_state(dict(student=m.state_dict(),optimizer=opt.state_dict()))
    with torch.no_grad():
        for p in m.parameters(): p.add_(1)
        m.bn.running_mean.add_(4)
    ao.zero_grad(); adapter(torch.randn(3,128)).square().mean().backward(); ao.step()
    adapter_before=t.old.cpu_state(adapter.state_dict()); ao_before=t.old.cpu_state(ao.state_dict())
    t.restore_student(m,opt,anchor,2.5e-6)
    for k,v in anchor['student'].items(): torch.testing.assert_close(v,m.state_dict()[k],rtol=0,atol=0)
    for k,v in adapter_before.items(): torch.testing.assert_close(v,adapter.state_dict()[k],rtol=0,atol=0)
    assert ao.state_dict()['param_groups']==ao_before['param_groups']
    for key,state in anchor['optimizer']['state'].items():
        for name,value in state.items(): torch.testing.assert_close(value,opt.state_dict()['state'][key][name])
    assert all(g['lr']==2.5e-6 for g in opt.param_groups)


def test_actual_five_step_diagnostic_and_meta_update_do_not_change_real_state():
    torch.manual_seed(44); m=Mini(); opt=t.old.optimizer_for(m); adapter=models.Adapter()
    supports=[dict(inputs=dict(features=torch.randn(16,3)),labels=torch.arange(16)%11,
        target=torch.randn(16,128)) for _ in range(5)]
    query=dict(inputs=dict(features=torch.randn(64,3)),labels=torch.arange(64)%11)
    before=t.old.cpu_state(m.state_dict()); rng=torch.get_rng_state().clone()
    d=t.diagnostics(m,opt,adapter,supports,query,c.INITIAL_LR,3)
    assert d['query_ce_delta']==0 and d['residual_max']==0
    classifier=torch.nn.Linear(128,11)
    for p in classifier.parameters(): p.requires_grad_(False)
    rng=torch.get_rng_state().clone()
    result=meta.update(m,opt,adapter,torch.optim.Adam(adapter.parameters(),lr=1e-4),classifier,
        torch.zeros(128),torch.ones(128),supports,query,[c.INITIAL_LR]*5,seed=3)
    assert result['grad_norm']>0
    assert t.diagnostics(m,opt,adapter,supports,query,c.INITIAL_LR,3)['residual_max']>0
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in before.items(): torch.testing.assert_close(v,m.state_dict()[k],rtol=0,atol=0)
    assert not opt.state and all(p.grad is None for p in m.parameters())
    assert all(p.grad is None for p in classifier.parameters())


class Cache:
    def __init__(self,role='train',n=3200):
        self.role=role; self.coordinate_name='D000'; self.foundation_sha256='a'*64
        self.identities=identities(n); self.labels=np.arange(n,dtype=np.int64)%11
        if role=='validation': self.identities=self.identities.copy(); self.identities[:,-1]^=255
        self.features=np.random.default_rng(15).normal(size=(n,3)).astype(np.float32)
    def __len__(self): return len(self.labels)
    def batch(self,ix):
        return dict(features=self.features[ix],vectors=self.features[ix],mask=np.ones((len(ix),1),bool),labels=self.labels[ix])


def fixture_spec(root,train_rows=3200):
    from test_luka_attention import parent_nodes
    from hlt_classification.luka_meta_embedding.contracts import node
    parent=dict(node=node(dict(nodes=parent_nodes())),files=dict(state=dict(sha256='d'*64)),bank_hash='b'*64)
    return c.artifact('SPEC',root=str(root),project_dir='/project',source_commit='f'*40,
        source_snapshot=dict(content_hash='c'*64),anchor=dict(evidence=dict(prepared='a'*64,environment={})),
        parent=parent,**dict(campaign.registration(parent),counts=dict(train=train_rows,validation=22),original50k_rows=22))


@pytest.mark.parametrize('arm',['fixed','meta'])
def test_all_twenty_blocks_real_meta_objective_durable_selection_and_role_isolation(tmp_path,monkeypatch,arm):
    train=Cache(); val=Cache('validation',22); spec=fixture_spec(tmp_path); m=Mini()
    bank=SimpleNamespace(identities=train.identities,labels=train.labels,
        targets=np.random.default_rng(16).normal(size=(len(train),128)).astype(np.float32))
    inner,query,guard=t.split(train.identities); seen=[]
    original=train.batch
    def read(ix):
        seen.append(set(ix)); return original(ix)
    monkeypatch.setattr(train,'batch',read)
    # Real production predictor/CE and five-step virtual objective; only model
    # size and reporting forward wrapper are miniature here.
    def mini_predict(model,cache,device,batch_size=128):
        model.eval(); out=[]
        with torch.no_grad():
            for start in range(0,len(cache),batch_size):
                raw=cache.batch(np.arange(start,min(start+batch_size,len(cache))))
                out.append(torch.softmax(model(torch.from_numpy(raw['features'])),dim=1).numpy())
        return np.concatenate(out)
    monkeypatch.setattr(t,'predict',mini_predict)
    original_meta=t.old.meta_batches
    def checked_meta(cache,bank,inside,q,device,seed):
        assert set(inside)==set(inner) and set(q)==set(query) and not set(q)&set(guard)
        return original_meta(cache,bank,inside,q,device,seed)
    monkeypatch.setattr(t.old,'meta_batches',checked_meta)
    snapshots={}; histories={}
    def publish(kind,b,value):
        if kind=='selected':
            snapshots[b]=value; path=tmp_path/'snapshots'/f'selected-{b:03d}.pt'; save_state(path,value)
        else:
            histories[b]=value; path=tmp_path/'history'/f'block-{b:03d}.json'; write_immutable_json(path,value)
        return file_ref(path,root=tmp_path)
    (tmp_path/'snapshots').mkdir(); (tmp_path/'history').mkdir()
    report,native,state=t.train(m,train,val,bank,spec['node'],arm,'cpu',initial=deepcopy(m.state_dict()),
        classifier=torch.nn.Linear(128,11),mean=torch.zeros(128),scale=torch.ones(128),
        parents=worker.fit_parents(spec),publish=publish)
    t.validate_report(report,spec,worker.fit_parents(spec),arm)
    assert len(histories)==20 and 0 in snapshots
    assert sum(len(r['meta_updates']) for r in report['history'])==(475 if arm=='meta' else 0)
    assert all(s<=set(inner) or s<=set(query) or s<=set(guard) for s in seen)
    assert state['block']==report['selected']['block']
    for k,v in native.items(): torch.testing.assert_close(v,state['student'][k],rtol=0,atol=0)
    for mutate in ('decision','learning_rate','meta_updates','selected','snapshots','rows'):
        bad=deepcopy(report)
        if mutate=='selected': bad['selected']['validation']['macro_ovr_auc']=.9999
        elif mutate=='snapshots': bad['snapshots']=[]
        elif mutate=='decision': bad['history'][0]['decision']['anchor_auc']=0.
        elif mutate=='meta_updates': bad['history'][0]['meta_updates']=[dict(event=999)]
        else: bad['history'][0][mutate]=999
        with pytest.raises(ValueError): t.validate_report(with_content_hash(bad),spec,worker.fit_parents(spec),arm)


def test_block_zero_wins_when_validation_worsens_and_guard_not_validation(tmp_path,monkeypatch):
    train=Cache(); val=Cache('validation',22); spec=fixture_spec(tmp_path); m=Mini()
    bank=SimpleNamespace(identities=train.identities,labels=train.labels,targets=np.zeros((len(train),128),np.float32))
    def metrics(auc,rows): return dict(macro_ovr_auc=auc,cross_entropy=1.,macro_mean_log_qcd_rejection_at_50pct_signal=None,rows=rows)
    seen=[]
    def scored(model,cache,device):
        seen.append(cache)
        return metrics(.7 if isinstance(cache,t.Subset) else .95 if len(seen)==2 else .5,len(cache))
    monkeypatch.setattr(t,'score',scored)
    monkeypatch.setattr(t,'diagnostics',lambda *a:dict(identity_query_ce=1.,adapted_query_ce=1.,query_ce_delta=0.,residual_rms=0.,residual_max=0.))
    initial=deepcopy(m.state_dict())
    def publish(kind,b,value): return dict(path=f'snapshots/selected-{b:03d}.pt',bytes=1,sha256='a'*64)
    report,_,state=t.train(m,train,val,bank,spec['node'],'fixed','cpu',initial=initial,
        classifier=torch.nn.Linear(128,11),mean=torch.zeros(128),scale=torch.ones(128),parents=worker.fit_parents(spec),publish=publish)
    assert report['selected']['block']==0
    assert all(r['decision']['action']=='keep' for r in report['history'])
    for k,v in initial.items(): torch.testing.assert_close(v,state['student'][k],rtol=0,atol=0)
    t.validate_report(report,spec,worker.fit_parents(spec),'fixed')


def test_dry_plan_exact_dependencies_repeated_submission_and_lost_reply(tmp_path,monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as exact
    spec=fixture_spec(tmp_path/'good'); (tmp_path/'good').mkdir()
    monkeypatch.setattr(campaign,'check',lambda _:None); monkeypatch.setattr(submission,'check_site',lambda _:None)
    monkeypatch.setattr(exact.subprocess,'run',lambda *a,**kw:pytest.fail('No scheduler on dry run'))
    plan=submission.submit(spec)
    assert plan['jobs']==3 and [r['dependencies'] for r in plan['commands']]==[[],['setup'],['setup']]
    for row in plan['commands']:
        assert '--partition=debug' in row['command'] and '--no-requeue' in row['command']
        assert '--time=12:00:00' in row['command']
    with pytest.raises(PermissionError): submission.submit(spec,execute=True,plan_hash='0'*64,authorization=c.AUTHORIZATION)
    calls=[]
    def scheduler(cmd,**kwargs):
        calls.append(cmd); return SimpleNamespace(stdout=str(800+len(calls))+'\n',returncode=0,stderr='')
    monkeypatch.setattr(exact.subprocess,'run',scheduler)
    for _ in range(2): submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==3 and submission.accepted_jobs(spec)==dict(setup='801',fixed='802',meta='803')
    assert all('--dependency=afterok:801' in cmd for cmd in calls[1:])
    other=fixture_spec(tmp_path/'lost'); (tmp_path/'lost').mkdir(); plan=submission.submit(other)
    def lost(*a,**kw): raise RuntimeError('possible accepted job; lost reply')
    monkeypatch.setattr(exact.subprocess,'run',lost)
    with pytest.raises(RuntimeError): submission.submit(other,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    with pytest.raises(FileExistsError): submission.submit(other,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)


def test_resource_admission_fail_closed_and_fixed_budget(tmp_path,monkeypatch):
    spec=fixture_spec(tmp_path)
    measured=dict(environment={},gpu=dict(total_memory_bytes=40*2**30),peak_rss_bytes=2**30,gpu_peak_bytes=2**30,
        timings={a:dict(authentication=1.,load=1.,readback=1.,initialization=1.,headstart=4.,later_phase=1.,block=10.) for a in c.ARMS})
    assert max(worker.admission(spec,measured).values())<=720
    for key,value in [('peak_rss_bytes',100*2**30),('gpu_peak_bytes',40*2**30)]:
        with pytest.raises(MemoryError): worker.admission(spec,dict(measured,**{key:value}))
    bad=deepcopy(measured); bad['timings']['meta']['block']=10000.
    with pytest.raises(ValueError,match='walltime'): worker.admission(spec,bad)
    bad=deepcopy(measured); del bad['timings']['meta']['headstart']
    with pytest.raises(ValueError): worker.admission(spec,bad)
    assert spec['storage_budget_bytes']==4*2**30


def test_source_refuses_missing_completed_fixed_before_replay(tmp_path,monkeypatch):
    path=tmp_path/'campaign_spec.json'
    write_immutable_json(path,dict(root=str(tmp_path),source_commit=source.PARENT_COMMIT))
    monkeypatch.setattr(source,'replay',lambda *a:pytest.fail('Must stop before replay'))
    with pytest.raises(ValueError,match='must finish'): source.authenticate(path)


def test_training_rejects_reporting_validation_overlap_before_model_use():
    train=Cache(); val=Cache('validation',22)
    val.identities=train.identities[:22].copy()
    spec=fixture_spec('/fixture')
    bank=SimpleNamespace(identities=train.identities,labels=train.labels)
    with pytest.raises(PermissionError,match='overlap'):
        t.train(None,train,val,bank,spec['node'],'fixed','cpu',initial=None,classifier=None,
            mean=None,scale=None,parents={},publish=None)


def test_source_rejects_existing_runtime_mutation(tmp_path,monkeypatch):
    monkeypatch.setattr(source,'_source',lambda *a:{})
    def git(cmd,**kwargs):
        return SimpleNamespace(stdout='\n'.join(sorted(source.REQUIRED)) if 'ls-files' in cmd else 'M\tsrc/old.py\n')
    monkeypatch.setattr(source.subprocess,'run',git)
    with pytest.raises(ValueError,match='runtime changed'): source.current(tmp_path,'f'*40)


def test_original_replay_requires_fixed_and_setup_not_meta(tmp_path,monkeypatch):
    import io,json,sys
    from contextlib import redirect_stdout
    from hlt_classification.luka_meta_embedding import campaign as oc, worker as ow, submission as os
    from hlt_classification.luka_fullsim import science_campaign
    spec=fixture_spec(tmp_path); write_immutable_json(tmp_path/'campaign_spec.json',spec)
    (tmp_path/'targets').mkdir(); write_immutable_json(tmp_path/'targets/manifest.json',dict(content_hash='b'*64))
    validated=[]; accounted=[]
    monkeypatch.setattr(oc,'check',lambda _:None)
    monkeypatch.setattr(os,'accepted_jobs',lambda _:dict(setup='1',fixed='2',meta='3'))
    def receipt(s,task):
        validated.append(task); return dict(job_id={'setup':'1','fixed':'2'}[task])
    monkeypatch.setattr(ow,'validate_task',receipt)
    monkeypatch.setattr(ow,'validate_fit',lambda *a:(dict(report=dict(selected=dict(pass_number=2))),{}))
    monkeypatch.setattr(science_campaign,'preflight_accounting',lambda job:accounted.append(job))
    import hlt_classification.cms_proxy_ladder.contracts as fc
    monkeypatch.setattr(fc,'file_ref',lambda path:dict(path=str(path),bytes=1,sha256='a'*64))
    monkeypatch.setattr(sys,'argv',['replay',str(tmp_path/'campaign_spec.json')])
    captured=io.StringIO()
    with redirect_stdout(captured): exec(compile(source.REPLAY,'parent-replay','exec'),{})
    value=json.loads(captured.getvalue().split('META_GUARD_PARENT=')[1])
    assert validated==['setup','fixed'] and accounted==['1','2']
    assert value['selected']['pass_number']==2 and 'state' in value['files']


def test_full_size_weaver_cuda_hypergradient_at_guarded_rate():
    if not torch.cuda.is_available(): pytest.skip('CUDA absent')
    pytest.importorskip('weaver.nn.model.ParticleTransformer')
    from test_luka_attention import cache
    from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
    spec=fixture_spec('/fixture'); student=models.new_model(spec['node']).cuda().eval()
    native=DelphesParticleTransformer().cuda().eval(); native.load_state_dict(student.native_state(),strict=True)
    raw=cache('train','D000').batch(np.arange(16))
    inputs={k:torch.from_numpy(raw[k]).cuda() for k in ('features','vectors','mask')}
    with torch.no_grad(): torch.testing.assert_close(student(**inputs),native(**inputs),rtol=0,atol=0)
    labels=torch.from_numpy(raw['labels']).cuda(); adapter=models.Adapter().cuda()
    optimizer=t.old.optimizer_for(student); before=t.old.cpu_state(student.state_dict())
    supports=[dict(inputs=inputs,labels=labels,target=torch.randn(16,128,device='cuda')) for _ in range(5)]
    loss=meta.virtual_query(student,optimizer,adapter,supports,dict(inputs=inputs,labels=labels),[c.INITIAL_LR]*5,seed=24)
    grads=torch.autograd.grad(loss,tuple(adapter.parameters()))
    assert all(torch.isfinite(g).all() for g in grads) and any(torch.count_nonzero(g) for g in grads)
    for k,v in before.items(): torch.testing.assert_close(v,student.state_dict()[k].cpu(),rtol=0,atol=0)
    assert not optimizer.state and all(p.grad is None for p in student.parameters())


def test_cuda_installed_weaver_production_workers_setup_and_both_twenty_blocks(tmp_path,monkeypatch):
    """Synthetic populations, real installed Weaver/production loops, no Slurm claims."""
    if not torch.cuda.is_available(): pytest.skip('CUDA absent')
    pytest.importorskip('weaver.nn.model.ParticleTransformer')
    from test_luka_embedding_aux import tiny
    from test_luka_attention import cache
    from hlt_classification.luka_embedding_aux import targets
    from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
    from hlt_classification.jetclass2_delphes.runner import predict
    from hlt_classification.data.cache_contracts import atomic_publish_bytes, deterministic_npz_bytes
    tiny(monkeypatch)
    from hlt_classification.jetclass2_delphes import acceptance, model as native_model
    monkeypatch.setattr(acceptance,'model_config',native_model.model_config)
    root=tmp_path/'new'; root.mkdir(); parent_root=tmp_path/'parent'; parent_root.mkdir()
    train=cache('train','D000',rows=3200); val=cache('validation','D000')
    val.identities[:,-1]=1
    for block in val.blocks: block.identities[:,-1]=1
    offline=cache('train','OFFLINE',rows=3200)
    spec=fixture_spec(root); student=models.new_model(spec['node']); teacher=DelphesParticleTransformer()
    save_state(parent_root/'teacher.pt',teacher.state_dict()); teacher_ref=file_ref(parent_root/'teacher.pt')
    manifest=targets.export(parent_root/'targets',teacher,offline,teacher_ref=teacher_ref,device='cuda')
    bank=targets.Bank(parent_root/'targets',train,teacher_ref=teacher_ref)
    save_state(parent_root/'state.pt',dict(student=student.state_dict(),epoch=2))
    save_state(parent_root/'native.pt',student.native_state())
    save_state(parent_root/'classifier.pt',teacher.mod.fc[0].state_dict())
    actual=predict(student.cuda(),val,device='cuda',batch_size=128)
    arrays=dict(identities=val.identities,labels=val.labels,probabilities=actual)
    atomic_publish_bytes(parent_root/'predictions.npz',deterministic_npz_bytes(arrays))
    atomic_publish_bytes(parent_root/'original50k.npz',deterministic_npz_bytes(dict(indices=np.arange(22,dtype=np.int64))))
    files={k:file_ref(parent_root/name) for k,name in dict(state='state.pt',native='native.pt',classifier='classifier.pt',
        predictions='predictions.npz',bank='targets/targets.npz',bank_manifest='targets/manifest.json',original50k='original50k.npz').items()}
    files.update({k:files['predictions'] for k in ('M0HLT','OFFLINE','OFFLINE_EMBED_W5')})
    spec['parent'].update(files=files,bank_hash=manifest['content_hash'],selected=dict(pass_number=2))
    spec['anchor']['evidence'].update(environment=worker.installed_environment(),prepared_root='/fixture')
    spec=with_content_hash(spec); write_immutable_json(root/'campaign_spec.json',spec)
    monkeypatch.setattr(worker,'load_inputs',lambda *a:(train,val,bank))
    monkeypatch.setattr(campaign,'check',lambda _:None); monkeypatch.setattr(campaign,'validate_spec',lambda _:None)
    monkeypatch.setattr(source,'current',lambda *a:spec['source_snapshot']); monkeypatch.setattr(source,'validate_parent',lambda *a:None)
    jobs=dict(setup='1',fixed='2',meta='3')
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a,**kw:jobs)
    monkeypatch.setattr(worker,'preflight_accounting',lambda *a:None)
    metadata=({},dict(summaries={r:dict(D000=dict(maximum=4)) for r in ('train','validation')}),{}, {})
    monkeypatch.setattr(worker.regular_cache,'metadata',lambda *a:metadata)
    monkeypatch.setattr(worker,'memory_snapshot',lambda _:dict(rss_bytes=2**30,peak_rss_bytes=2**30,
        gpu_peak_bytes=int(torch.cuda.max_memory_allocated())))
    for task in c.TASKS:
        monkeypatch.setattr(worker,'execution',lambda *a,task=task:jobs[task])
        worker.run(spec,task)
    assert not (root/'targets').exists() and not (root/'initial').exists()
    for arm in c.ARMS:
        record,_=worker.validate_fit(root/arm/'fit',spec,arm)
        assert len(record['report']['history'])==20
    with pytest.raises(FileExistsError): worker.run(spec,'meta')
    monkeypatch.setattr(torch,'load',lambda *a,**kw:pytest.fail('Results must not deserialize models'))
    result=worker.results(spec); assert not result['pending'] and len(result['rows'])==6
    path=root/'meta/fit/history/block-001.json'; bad=load_json(path); bad['row']['decision']['anchor_auc']=0
    path.write_text(__import__('json').dumps(bad))  # only this test's scratch artifact
    with pytest.raises(ValueError): worker.results(spec)

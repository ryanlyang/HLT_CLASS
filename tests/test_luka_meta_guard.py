"""Guard isolation, full-state rollback, real meta derivatives and exact queueing."""
from copy import deepcopy
from pathlib import Path
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


def fixture_spec(root,train_rows=3200,initializer_source='completed_fixed'):
    from test_luka_attention import parent_nodes
    from hlt_classification.luka_meta_embedding.contracts import node
    parent=dict(node=node(dict(nodes=parent_nodes())),files=dict(state=dict(sha256='d'*64)),bank_hash='b'*64,
        initializer_source=initializer_source)
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


def test_shared_initializer_is_explicit_and_still_requires_completed_setup(tmp_path,monkeypatch):
    path=tmp_path/'campaign_spec.json'
    write_immutable_json(path,dict(root=str(tmp_path),source_commit=source.PARENT_COMMIT))
    monkeypatch.setattr(source,'replay',lambda *a,**kw:pytest.fail('Must stop before replay'))
    with pytest.raises(ValueError,match='setup receipt'): source.authenticate(path,'shared_w5')
    with pytest.raises(ValueError,match='no automatic fallback'): source.authenticate(path,'auto')
    assert c.recipe('shared_w5')['content_hash']!=c.recipe('completed_fixed')['content_hash']
    assert c.recipe('shared_w5')['schema_version']==2
    with pytest.raises(ValueError): c.validate(dict(c.recipe(),contract='LUKA_META_GUARD_RECIPE/v1'),'RECIPE')
    spec=fixture_spec(tmp_path,initializer_source='shared_w5')
    assert spec['initializer_source']=='shared_w5' and spec['recipe']==c.recipe('shared_w5')


@pytest.mark.parametrize('mode',['completed_fixed','shared_w5'])
def test_full_initializer_readback_preserves_predictor_and_rejects_native_only(tmp_path,monkeypatch,mode):
    from hlt_classification.data.cache_contracts import atomic_publish_bytes,deterministic_npz_bytes
    torch.manual_seed(123); model=Mini(); native=model.native_state()
    with torch.no_grad(): model.predictor.weight.fill_(.12345)
    state=deepcopy(model.state_dict())
    full=dict(student=state,epoch=2) if mode=='completed_fixed' else state
    save_state(tmp_path/'state.pt',full); save_state(tmp_path/'native.pt',native)
    save_state(tmp_path/'classifier.pt',torch.nn.Linear(128,11).state_dict())
    atomic_publish_bytes(tmp_path/'bank.npz',deterministic_npz_bytes(dict(mean=np.zeros(128,np.float32),scale=np.ones(128,np.float32))))
    spec=fixture_spec(tmp_path,initializer_source=mode)
    files={k:file_ref(tmp_path/(k+'.pt' if k!='bank' else 'bank.npz')) for k in ('state','native','classifier','bank')}
    spec['parent'].update(files=files,selected=dict(pass_number=2) if mode=='completed_fixed' else None)
    monkeypatch.setattr(models,'new_model',lambda _:Mini())
    initial,classifier,mean,scale=worker.read_shared(spec)
    assert not any(p.requires_grad for p in classifier.parameters())
    torch.testing.assert_close(initial['predictor.weight'],state['predictor.weight'],rtol=0,atol=0)
    for key,v in state.items(): torch.testing.assert_close(v,initial[key],rtol=0,atol=0)
    save_state(tmp_path/'missing.pt',dict(student=native,epoch=2) if mode=='completed_fixed' else native)
    files['state']=file_ref(tmp_path/'missing.pt')
    with pytest.raises(RuntimeError,match='Missing key'): worker.read_shared(spec)
    if mode=='completed_fixed':
        save_state(tmp_path/'epoch.pt',dict(student=state,epoch=8)); files['state']=file_ref(tmp_path/'epoch.pt')
        with pytest.raises(ValueError,match='selected FIXED epoch'): worker.read_shared(spec)
    bad=deepcopy(state); bad['predictor.weight'][0,0]=float('nan')
    save_state(tmp_path/'nonfinite.pt',dict(student=bad,epoch=2) if mode=='completed_fixed' else bad)
    files['state']=file_ref(tmp_path/'nonfinite.pt')
    with pytest.raises(ValueError,match='nonfinite'): worker.read_shared(spec)


def test_shared_parent_authentication_binds_inventory_and_rejects_changes(tmp_path,monkeypatch):
    root=tmp_path/'original'; root.mkdir()
    names=dict(setup_receipt='setup/receipt.json',bank_manifest='targets/manifest.json',
        bank='targets/targets.npz',classifier='initial/classifier.pt',original50k='references/original50k.npz',
        state='initial/student.pt',initializer='setup/initializer.json',predictions='references/OFFLINE_EMBED_W5.npz')
    names.update({n:'references/'+n+'.npz' for n in ('M0HLT','OFFLINE','OFFLINE_EMBED_W5')})
    for name in set(names.values()):
        path=root/name; path.parent.mkdir(parents=True,exist_ok=True)
        write_immutable_json(path,dict(content_hash='b'*64))
    native=root/'w5.pt'; write_immutable_json(native,dict(dummy=True))
    old=dict(source_commit=source.PARENT_COMMIT,root=str(root),project_dir=str(tmp_path/'project'),
        source_snapshot={},anchor={},node={},counts={},initial_checkpoint=file_ref(native))
    specpath=root/'campaign_spec.json'; write_immutable_json(specpath,old)
    files={k:file_ref(root/n) for k,n in names.items()}; files['native']=old['initial_checkpoint']
    evidence=dict(files=files,initializer_source='shared_w5',anchor={},node={},counts={},jobs={'setup':'1'},
        selected=None,bank_hash='b'*64)
    monkeypatch.setattr(source,'validate_source_snapshot',lambda *a,**kw:None)
    calls=[]
    def replay(*args,extra):
        calls.append(extra); return file_ref(specpath),deepcopy(evidence)
    monkeypatch.setattr(source,'replay',replay)
    parent=source.authenticate(specpath,'shared_w5')
    assert calls==[('shared_w5',)] and source.validate_parent(parent)==old
    bad=deepcopy(parent); bad['selected']=dict(pass_number=2)
    with pytest.raises(ValueError,match='cannot claim'): source.validate_parent(with_content_hash(bad))
    bad=deepcopy(parent); bad['files']['state']=files['native']
    with pytest.raises(ValueError,match='lineage'): source.validate_parent(with_content_hash(bad))
    bad=deepcopy(parent); bad['jobs']['fixed']='2'
    with pytest.raises(ValueError,match='lineage'): source.validate_parent(with_content_hash(bad))
    # Simulate mutation after registration without updating the authenticated file ref.
    (root/'initial/student.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError): source.validate_parent(parent)


def test_queue_helper_requires_explicit_mode_and_v2_authorization():
    root=Path(__file__).resolve().parents[1]
    script=(root/'scripts/queue_luka_meta_guard.sh').read_text()
    assert '--initializer-source "${INITIALIZER_SOURCE}"' in script
    assert '[[ "${INITIALIZER_SOURCE}" == completed_fixed ]]' in script
    assert 'AUTHORIZE_LUKA_META_GUARD_DEBUG_V2' in script
    assert '${COMMIT:0:8}_${INITIALIZER_SOURCE}_debug_r1' in script
    assert 'shared_w5|completed_fixed)' in script


def test_create_and_spec_check_bind_explicit_mode_before_claiming_root(tmp_path,monkeypatch):
    project=tmp_path/'project'; project.mkdir()
    oldroot=tmp_path/'parent'; oldroot.mkdir()
    oldspec=oldroot/'campaign_spec.json'
    write_immutable_json(oldspec,dict(root=str(oldroot),project_dir=str(tmp_path/'old_project')))
    parent=c.artifact('PARENT',initializer_source='shared_w5',spec=file_ref(oldspec),counts=c.COUNTS,
        node={},anchor=dict(evidence=dict(prepared_root=str(tmp_path/'prepared'),raw_container=str(tmp_path/'raw'))))
    monkeypatch.setattr(source,'current',lambda *a:dict(content_hash='c'*64))
    requests=[]
    def authenticate(path,mode):
        assert Path(path)==oldspec; requests.append(mode); return parent
    monkeypatch.setattr(source,'authenticate',authenticate)
    monkeypatch.setattr(source,'validate_parent',lambda _:None)
    checked=[]
    def storage(spec):
        assert not Path(spec['root']).exists()
        checked.append(spec['storage_budget_bytes'])
    monkeypatch.setattr(campaign,'storage',storage)
    root=tmp_path/'new'
    spec=campaign.create(root,oldspec,project,'f'*40,22,initializer_source='shared_w5')
    assert requests==['shared_w5'] and checked==[4*2**30]
    assert load_json(root/'campaign_spec.json')==spec and spec['initializer_source']=='shared_w5'
    campaign.check(spec); assert requests==['shared_w5','shared_w5']
    bad=dict(spec,initializer_source='completed_fixed')
    with pytest.raises(ValueError,match='recipe/parent'): campaign.validate_spec(with_content_hash(bad))
    bad=dict(spec,recipe=c.recipe('completed_fixed'))
    with pytest.raises(ValueError,match='recipe/parent'): campaign.validate_spec(with_content_hash(bad))


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


def test_source_allows_only_guard_revision_on_pinned_compatibility_base(monkeypatch):
    project=Path(source.__file__).resolve().parents[3]; calls=[]
    monkeypatch.setattr(source,'_source',lambda *a:{'snapshot':'checked'})
    changed=['M\tsrc/hlt_classification/luka_meta_guard/source.py']
    def git(cmd,**kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout='\n'.join(sorted(source.REQUIRED)) if 'ls-files' in cmd
            else '\n'.join(changed) if 'diff' in cmd else '')
    monkeypatch.setattr(source.subprocess,'run',git)
    assert source.current(project,'f'*40)=={'snapshot':'checked'}
    assert any(cmd[-4:]==['merge-base','--is-ancestor',source.BASE,'f'*40] for cmd in calls)
    for row in ('M\tsrc/hlt_classification/luka_meta_embedding/worker.py',
                'M\tsrc/hlt_classification/luka_fullstat_finetune/source.py',
                'D\tsrc/hlt_classification/luka_meta_guard/source.py'):
        changed[:]=[row]
        with pytest.raises(ValueError,match='runtime changed'): source.current(project,'f'*40)


@pytest.mark.parametrize('mode',['completed_fixed','shared_w5'])
def test_original_replay_requires_only_selected_dependencies(tmp_path,monkeypatch,mode):
    import io,json,sys
    from contextlib import redirect_stdout
    from hlt_classification.luka_meta_embedding import campaign as oc, worker as ow, submission as os
    from hlt_classification.luka_fullsim import science_campaign
    spec=fixture_spec(tmp_path)
    spec['initial_checkpoint']=dict(path='/historical/w5.pt',bytes=1,sha256='e'*64)
    write_immutable_json(tmp_path/'campaign_spec.json',spec)
    (tmp_path/'targets').mkdir(); write_immutable_json(tmp_path/'targets/manifest.json',dict(content_hash='b'*64))
    validated=[]; accounted=[]; shared=[]
    monkeypatch.setattr(oc,'check',lambda _:None)
    monkeypatch.setattr(os,'accepted_jobs',lambda _:dict(setup='1',fixed='2',meta='3'))
    def receipt(s,task):
        if mode=='shared_w5': assert task=='setup', 'Cancelled fits must not be required'
        validated.append(task); return dict(job_id={'setup':'1','fixed':'2'}[task])
    monkeypatch.setattr(ow,'validate_task',receipt)
    def fit(*a):
        assert mode=='completed_fixed', 'Cancelled fit must not be read'
        return dict(report=dict(selected=dict(pass_number=2))),{}
    monkeypatch.setattr(ow,'validate_fit',fit)
    monkeypatch.setattr(ow,'read_shared',lambda s:shared.append(s))
    monkeypatch.setattr(science_campaign,'preflight_accounting',lambda job:accounted.append(job))
    import hlt_classification.cms_proxy_ladder.contracts as fc
    monkeypatch.setattr(fc,'file_ref',lambda path:dict(path=str(path),bytes=1,sha256='a'*64))
    monkeypatch.setattr(sys,'argv',['replay',str(tmp_path/'campaign_spec.json'),mode])
    captured=io.StringIO()
    with redirect_stdout(captured): exec(compile(source.REPLAY,'parent-replay','exec'),{})
    value=json.loads(captured.getvalue().split('META_GUARD_PARENT=')[1])
    assert value['initializer_source']==mode and 'state' in value['files']
    if mode=='completed_fixed':
        assert validated==['setup','fixed'] and accounted==['1','2'] and not shared
        assert value['selected']['pass_number']==2
    else:
        assert validated==['setup'] and accounted==['1'] and len(shared)==1
        assert value['selected'] is None and value['jobs']=={'setup':'1'}
        assert value['files']['native']==spec['initial_checkpoint']
        assert Path(value['files']['state']['path'])==tmp_path/'initial/student.pt'
        assert not {'fixed_receipt','report'} & value['files'].keys()


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


@pytest.mark.parametrize('mode',['completed_fixed','shared_w5'])
def test_cuda_installed_weaver_production_workers_setup_and_both_twenty_blocks(tmp_path,monkeypatch,mode):
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
    spec=fixture_spec(root,initializer_source=mode); student=models.new_model(spec['node']); teacher=DelphesParticleTransformer()
    save_state(parent_root/'teacher.pt',teacher.state_dict()); teacher_ref=file_ref(parent_root/'teacher.pt')
    manifest=targets.export(parent_root/'targets',teacher,offline,teacher_ref=teacher_ref,device='cuda')
    bank=targets.Bank(parent_root/'targets',train,teacher_ref=teacher_ref)
    save_state(parent_root/'state.pt',dict(student=student.state_dict(),epoch=2) if mode=='completed_fixed' else student.state_dict())
    save_state(parent_root/'native.pt',student.native_state())
    save_state(parent_root/'classifier.pt',teacher.mod.fc[0].state_dict())
    actual=predict(student.cuda(),val,device='cuda',batch_size=128)
    arrays=dict(identities=val.identities,labels=val.labels,probabilities=actual)
    atomic_publish_bytes(parent_root/'predictions.npz',deterministic_npz_bytes(arrays))
    atomic_publish_bytes(parent_root/'original50k.npz',deterministic_npz_bytes(dict(indices=np.arange(22,dtype=np.int64))))
    files={k:file_ref(parent_root/name) for k,name in dict(state='state.pt',native='native.pt',classifier='classifier.pt',
        predictions='predictions.npz',bank='targets/targets.npz',bank_manifest='targets/manifest.json',original50k='original50k.npz').items()}
    files.update({k:files['predictions'] for k in ('M0HLT','OFFLINE','OFFLINE_EMBED_W5')})
    spec['parent'].update(files=files,bank_hash=manifest['content_hash'],selected=dict(pass_number=2) if mode=='completed_fixed' else None)
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
    assert result['rows'][3]['model']==('PARENT_FIXED' if mode=='completed_fixed' else 'PARENT_SHARED_W5')
    path=root/'meta/fit/history/block-001.json'; bad=load_json(path); bad['row']['decision']['anchor_auc']=0
    path.write_text(__import__('json').dumps(bad))  # only this test's scratch artifact
    with pytest.raises(ValueError): worker.results(spec)

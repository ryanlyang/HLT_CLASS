"""Loss wiring, differentiable optimization, invariants and authenticated DAG."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from hlt_classification.luka_meta_embedding import contracts as c, model as m, meta, training as t


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_recipe_and_split():
    assert c.TASKS=={'setup':[],'fixed':['setup'],'meta':['setup']}
    assert c.recipe()['meta_steps']==5 and not c.recipe()['logit_kd']
    assert c.learning_rate(3)==pytest.approx(3e-5)
    assert c.learning_rate(30)==pytest.approx(1.5e-6)
    assert [e for e in range(1,31) if c.meta_due(e,100)]==list(range(4,26))
    ids=np.random.default_rng(123).integers(0,256,(200,32),dtype=np.uint8)
    inner,query=t.split(ids)
    assert len(query)==10 and len(inner)==190 and not set(inner)&set(query)
    order=np.random.default_rng(42).permutation(len(ids)); _,other=t.split(ids[order])
    assert {tuple(i) for i in ids[query]}=={tuple(i) for i in ids[order][other]}


def test_targets_detached_only_in_real_objective_and_adapter_bounded():
    logits=torch.randn(6,11,requires_grad=True); p=torch.randn(6,128,requires_grad=True)
    z=torch.randn(6,128,requires_grad=True); y=torch.arange(6)
    loss,ce,aux=m.objective(logits,y,p,z); loss.backward()
    assert z.grad is None and p.grad is not None
    torch.testing.assert_close(loss,ce+5*aux)
    m.objective(logits,y,p,z,differentiable_target=True)[0].backward()
    assert z.grad is not None
    a=m.Adapter(); torch.testing.assert_close(a(z),z,atol=0,rtol=0)
    with torch.no_grad(): a.net[-1].bias.fill_(1e4)
    assert (a(z)-z).abs().max()<.100001


def test_functional_adam_matches_torch_with_and_without_moments():
    for warm in (0,3):
        layer=torch.nn.Linear(3,2).double(); opt=torch.optim.AdamW(layer.parameters(),lr=.002,weight_decay=.05)
        for _ in range(warm):
            opt.zero_grad(); layer(torch.ones(4,3,dtype=torch.float64)).square().mean().backward(); opt.step()
        states,configs=meta.clone_optimizer(layer,opt)
        ps={n:p.detach().clone().requires_grad_() for n,p in layer.named_parameters()}
        for _ in range(5):
            opt.zero_grad(); layer(torch.ones(4,3,dtype=torch.float64)).square().mean().backward()
            grads=tuple(p.grad.clone() for p in layer.parameters())
            ps,states=meta.adam_step(ps,grads,states,configs,.002); opt.step()
            for n,p in layer.named_parameters(): torch.testing.assert_close(ps[n],p,rtol=1e-12,atol=1e-12)


class Mini(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=torch.nn.Linear(3,8); self.classifier=torch.nn.Linear(8,11)
        self.predictor=torch.nn.Linear(8,128)
    def forward(self,x,*,auxiliary=False):
        h=torch.tanh(self.encoder(x)); logits=self.classifier(h)
        return (logits,self.predictor(h)) if auxiliary else logits


def test_five_step_meta_gradient_matches_finite_difference_and_no_mutation():
    torch.manual_seed(54); student=Mini().double(); adapter=m.Adapter().double()
    opt=torch.optim.AdamW(student.parameters(),lr=.002)
    support=[dict(inputs=dict(x=torch.randn(4,3,dtype=torch.float64)),labels=torch.arange(4),
                  target=torch.randn(4,128,dtype=torch.float64)) for _ in range(5)]
    query=dict(inputs=dict(x=torch.randn(4,3,dtype=torch.float64)),labels=torch.arange(4))
    state=deepcopy(student.state_dict()); rng=torch.get_rng_state().clone()
    rates=[.002]*5
    loss=meta.virtual_query(student,opt,adapter,support,query,rates,seed=12)
    g=torch.autograd.grad(loss,adapter.net[-1].bias)[0]
    assert torch.isfinite(g).all() and g.norm()>0
    idx=int(g.abs().argmax()); eps=2e-3
    values=[]
    for sign in (1,-1):
        with torch.no_grad(): adapter.net[-1].bias[idx]=sign*eps
        values.append(float(meta.virtual_query(student,opt,adapter,support,query,rates,seed=12).detach()))
    with torch.no_grad(): adapter.net[-1].bias[idx]=0
    assert float(g[idx])==pytest.approx((values[0]-values[1])/(2*eps),rel=.1,abs=5e-5)
    assert torch.equal(rng,torch.get_rng_state()) and not opt.state
    for k,v in state.items(): torch.testing.assert_close(v,student.state_dict()[k],rtol=0,atol=0)
    assert all(p.grad is None for p in student.parameters())


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_installed_weaver_five_step_full_encoder_hypergradient(monkeypatch,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA absent')
    pytest.importorskip('weaver.nn.model.ParticleTransformer')
    from test_luka_embedding_aux import tiny
    from test_luka_u100_pretrain import cache
    from test_luka_attention import parent_nodes
    tiny(monkeypatch); node=c.node(dict(nodes=parent_nodes())); student=m.new_model(node).to(device)
    raw=cache().batch(np.arange(4))
    inputs={k:torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')}
    y=torch.from_numpy(raw['labels']).to(device)
    adapter=m.Adapter().to(device); opt=t.optimizer_for(student)
    support=[dict(inputs=inputs,labels=y,target=torch.randn(4,128,device=device)) for _ in range(5)]
    state=deepcopy(student.state_dict())
    rng=torch.get_rng_state().clone(); cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    loss=meta.virtual_query(student,opt,adapter,support,dict(inputs=inputs,labels=y),[3e-5]*5,seed=17)
    grads=torch.autograd.grad(loss,tuple(adapter.parameters()))
    assert all(torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads)>0
    for k,v in state.items(): torch.testing.assert_close(v,student.state_dict()[k],rtol=0,atol=0)
    assert all(p.grad is None for p in student.parameters()) and not opt.state
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))


def test_full_size_native_weaver_cuda_five_step_and_inference_parity():
    if not torch.cuda.is_available(): pytest.skip('CUDA absent')
    pytest.importorskip('weaver.nn.model.ParticleTransformer')
    from test_luka_u100_pretrain import cache
    from test_luka_attention import parent_nodes
    from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
    node=c.node(dict(nodes=parent_nodes())); student=m.new_model(node).cuda()
    native=DelphesParticleTransformer().cuda(); native.load_state_dict(student.native_state(),strict=True)
    raw=cache().batch(np.arange(4)); native.eval(); student.eval()
    inputs={k:torch.from_numpy(raw[k]).cuda() for k in ('features','vectors','mask')}
    with torch.no_grad():
        a=student(**inputs); b=native(**inputs)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        with m.virtual_backend(): actual=student(**inputs)
        torch.testing.assert_close(a,actual,rtol=2e-5,atol=2e-6)
    adapter=m.Adapter().cuda(); opt=t.optimizer_for(student)
    labels=torch.from_numpy(raw['labels']).cuda()
    supports=[dict(inputs=inputs,labels=labels,target=torch.randn(4,128,device='cuda')) for _ in range(5)]
    loss=meta.virtual_query(student,opt,adapter,supports,dict(inputs=inputs,labels=labels),[3e-5]*5,seed=1)
    grads=torch.autograd.grad(loss,tuple(adapter.parameters()))
    assert all(torch.isfinite(g).all() for g in grads) and any(torch.count_nonzero(g) for g in grads)


def fixture_spec(root,rows=22):
    from test_luka_attention import parent_nodes
    from hlt_classification.luka_meta_embedding import campaign
    return c.artifact('SPEC',root=str(root),project_dir='/project',source_commit='f'*40,
        source_snapshot=dict(content_hash='c'*64),anchor=dict(evidence=dict(prepared='a'*64,environment={})),
        initial_checkpoint=dict(sha256='e'*64),reference=dict(),
        **dict(campaign.registration(dict(nodes=parent_nodes())),counts=dict(train=rows,validation=22),original50k_rows=22))


def test_exact_dry_dag_authorization_no_requeue_or_serial_science(tmp_path,monkeypatch):
    from hlt_classification.luka_meta_embedding import campaign,submission
    from hlt_classification.scouting import hcwdl_exact_dag_submission as exact
    spec=fixture_spec(tmp_path); monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(submission,'check_site',lambda _:None)
    monkeypatch.setattr(exact.subprocess,'run',lambda *a,**kw:pytest.fail('Dry review must not invoke scheduler'))
    plan=submission.submit(spec)
    assert plan==submission.render(spec) and plan['jobs']==3
    assert [r['dependencies'] for r in plan['commands']]==[[],['setup'],['setup']]
    for r in plan['commands']:
        assert '--partition=debug' in r['command'] and '--no-requeue' in r['command']
        assert '--gres=gpu:a100:1' in r['command'] and '--time=12:00:00' in r['command']
        assert not any('preflight' in x for x in r['command'])
    with pytest.raises(PermissionError): submission.submit(spec,execute=True,plan_hash='0'*64,authorization=c.AUTHORIZATION)
    assert not (tmp_path/'submission/live_claim').exists()


def test_initialization_calibration_native_unchanged_and_no_query_training(monkeypatch):
    pytest.importorskip('weaver.nn.model.ParticleTransformer')
    from types import SimpleNamespace
    from test_luka_embedding_aux import tiny
    from test_luka_u100_pretrain import cache
    from test_luka_attention import parent_nodes
    tiny(monkeypatch); train=cache(); node=c.node(dict(nodes=parent_nodes())); model=m.new_model(node)
    native=model.native_state(); native['mod.fc.0.bias'].add_(.5)
    m.initialize(model,native)
    bank=SimpleNamespace(identities=train.identities,labels=train.labels,
        targets=np.random.default_rng(1).normal(size=(len(train),128)).astype(np.float32))
    inner,query=t.split(train.identities); seen=[]; original=train.batch
    def checked(ix):
        assert not set(ix)&set(query); seen.extend(ix); return original(ix)
    monkeypatch.setattr(train,'batch',checked)
    before=deepcopy(model.predictor.state_dict())
    state,record=t.calibrate(model,train,bank,node,'cpu')
    for k,v in native.items(): torch.testing.assert_close(v,state[k],rtol=0,atol=0)
    assert record['native_unchanged'] and len(seen)==3*len(inner)
    assert any(not torch.equal(v,model.predictor.state_dict()[k]) for k,v in before.items())
    assert all(p.requires_grad for p in model.parameters())
    with pytest.raises(ValueError): m.initialize(model,state)


@pytest.mark.parametrize('arm',c.ARMS)
def test_full30_selection_bad_metrics_do_not_gate_and_roundtrip_report(monkeypatch,arm):
    pytest.importorskip('weaver.nn.model.ParticleTransformer')
    from types import SimpleNamespace
    from test_luka_embedding_aux import tiny
    from test_luka_u100_pretrain import cache
    tiny(monkeypatch); spec=fixture_spec('/fixture'); train=cache(); val=cache('validation')
    bank=SimpleNamespace(identities=train.identities,labels=train.labels,
        targets=np.random.default_rng(42).normal(size=(len(train),128)).astype(np.float32))
    model=m.new_model(spec['node']); initial=t.cpu_state(model.state_dict())
    classifier=torch.nn.Linear(128,11); parents={'bank':'a'*64}
    original=t.evaluate_probabilities; calls=[]
    def poor(labels,prob):
        calls.append(1); return dict(original(labels,prob),macro_ovr_auc=.4 if len(calls)==1 else .1,accuracy=0.)
    monkeypatch.setattr(t,'evaluate_probabilities',poor)
    report,native,full=t.train(model,train,val,bank,spec['node'],arm,'cpu',initial=initial,
        classifier=classifier,mean=torch.zeros(128),scale=torch.ones(128),parents=parents)
    assert len(calls)==30 and report['selected_pass']==1
    t.validate_report(report,spec,parents,arm)
    assert full['epoch']==1 and any(k.startswith('predictor.') for k in full['student'])
    assert not any(k.startswith('predictor.') for k in native)
    assert all(torch.equal(v,full['student'][k]) for k,v in native.items())
    from hlt_classification.data.cache_contracts import with_content_hash
    bad=deepcopy(report); bad['history'][0]['total_loss']+=.1
    with pytest.raises(ValueError): t.validate_report(with_content_hash(bad),spec,parents,arm)


def test_storage_budget_is_not_user_total_and_failure_before_root_claim(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from hlt_classification.luka_meta_embedding import campaign
    spec=fixture_spec(tmp_path/'fresh'); assert spec['storage_budget_bytes']==4*2**30
    monkeypatch.setattr(campaign.shutil,'disk_usage',lambda _:SimpleNamespace(free=3*2**30))
    with pytest.raises(OSError,match='4 GiB'): campaign.storage(spec)
    assert not (tmp_path/'fresh').exists()


def test_required_files_and_bash_syntax():
    from pathlib import Path
    import subprocess
    from hlt_classification.luka_meta_embedding import source
    root=Path(__file__).resolve().parents[1]
    assert all((root/p).is_file() for p in source.REQUIRED)
    compile(source.REPLAY,'original-w5-replay','exec')
    assert 'worker.validate_fit' in source.REPLAY and 'debug.results' not in source.REPLAY
    script=(root/'scripts/queue_luka_meta_embedding.sh').read_text()
    assert not any(x in script for x in ('scancel','scontrol','rm -'))
    assert 'train_OFFLINE/selected.pt' in script and 'w5/fit/selected.pt' in script
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    if bash.is_file():
        for p in ('scripts/queue_luka_meta_embedding.sh','sbatch/run_luka_meta_embedding.sh'):
            subprocess.run([str(bash),'-n',p],cwd=root,check=True,capture_output=True)


def test_cuda_production_workers_setup_and_both30_continuations(tmp_path,monkeypatch):
    """Real installed-Weaver kernels, reduced depth/population; no fake Slurm claims."""
    if not torch.cuda.is_available(): pytest.skip('CUDA absent')
    pytest.importorskip('weaver.nn.model.ParticleTransformer')
    from pathlib import Path
    from test_luka_embedding_aux import tiny
    from test_luka_attention import cache
    from hlt_classification.luka_meta_embedding import worker as w,campaign,source,submission
    from hlt_classification.luka_embedding_aux import targets
    from hlt_classification.context_fusion.worker import save_state
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    from hlt_classification.data.cache_contracts import (with_content_hash,write_immutable_json,load_json,
        atomic_publish_bytes,deterministic_npz_bytes)
    from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
    from hlt_classification.jetclass2_delphes.runner import predict
    tiny(monkeypatch)
    from hlt_classification.jetclass2_delphes import acceptance,model as native_model
    monkeypatch.setattr(acceptance,'model_config',native_model.model_config)
    # Default cadence100, five real differentiable steps,16/64 batches unchanged.
    rows=13475; train=cache(coordinate='D000',rows=rows); offline=cache(coordinate='OFFLINE',rows=rows)
    val=cache('validation','D000'); val.identities[:,-1]=1
    for block in val.blocks: block.identities[:,-1]=1
    root=tmp_path/'study'; root.mkdir(); spec=fixture_spec(root,rows)
    teacher=DelphesParticleTransformer().cuda().eval(); teacher_path=tmp_path/'teacher.pt'
    save_state(teacher_path,t.cpu_state(teacher.state_dict())); teacher_ref=file_ref(teacher_path)
    original_bank=targets.export(tmp_path/'original_bank',teacher,offline,teacher_ref=teacher_ref,device='cuda')
    student=m.new_model(spec['node']).cuda().eval(); native_path=tmp_path/'w5.pt'
    save_state(native_path,student.native_state())
    for name,net in (('M0HLT',student),('OFFLINE',teacher),('OFFLINE_EMBED_W5',student)):
        atomic_publish_bytes(tmp_path/(name+'.npz'),deterministic_npz_bytes(dict(identities=val.identities,
            labels=val.labels,probabilities=predict(net,val,device='cuda',batch_size=128))))
    spec['initial_checkpoint']=file_ref(native_path)
    spec['reference']=dict(bank_hash=original_bank['content_hash'],files=dict(predictions=file_ref(tmp_path/'OFFLINE_EMBED_W5.npz')))
    spec['anchor']['evidence'].update(prepared_root='/fixture',cache='b'*64,environment=w.installed_environment(),
        models={n:dict(predictions=file_ref(tmp_path/(n+'.npz')),checkpoint=teacher_ref) for n in ('M0HLT','OFFLINE')})
    spec=with_content_hash(spec); write_immutable_json(root/'campaign_spec.json',spec)
    monkeypatch.setattr(campaign,'check',lambda _:None); monkeypatch.setattr(campaign,'validate_spec',lambda _:None)
    monkeypatch.setattr(source,'current',lambda *a:spec['source_snapshot'])
    jobs=dict(setup='1',fixed='2',meta='3')
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a,**kw:jobs)
    monkeypatch.setattr(w,'execution',lambda *a:'1'); monkeypatch.setattr(w,'preflight_accounting',lambda *a:None)
    monkeypatch.setattr(w,'data',lambda s,r,co,meta:train if r=='train' else val)
    monkeypatch.setattr(targets,'offline_train',lambda *a,**kw:offline)
    monkeypatch.setattr(w.regular_cache,'metadata',lambda _:({},dict(summaries={r:dict(D000=dict(maximum=4)) for r in ('train','validation')}),{},{}))
    monkeypatch.setattr(w.alljets_data,'comparison_indices',lambda *a:np.arange(22,dtype=np.int64))
    monkeypatch.setattr(w,'memory_snapshot',lambda _:dict(rss_bytes=2**30,peak_rss_bytes=2**30,
        gpu_peak_bytes=int(torch.cuda.max_memory_allocated())))
    w.run(spec,'setup')
    assert w.results(spec)['pending']==list(c.ARMS)
    probe=load_json(root/'setup/probe_meta/training_report.json')['report']
    assert len(probe['history'][0]['meta_updates'])==1
    assert not (root/'targets/validation.npz').exists()
    for arm in c.ARMS:
        monkeypatch.setattr(w,'execution',lambda *a,arm=arm:jobs[arm]); w.run(spec,arm)
    records={a:load_json(root/a/'fit/training_report.json')['report'] for a in c.ARMS}
    assert records['fixed']['parents']==records['meta']['parents']
    assert len(records['meta']['history'])==30
    assert sum(len(r['meta_updates']) for r in records['meta']['history'])==22
    assert sum(len(r['meta_updates']) for r in records['fixed']['history'])==0
    with pytest.raises(FileExistsError): w.run(spec,'meta')
    monkeypatch.setattr(torch,'load',lambda *a,**kw:pytest.fail('Results must not open model tensors'))
    result=w.results(spec); assert not result['pending'] and len(result['rows'])==5
    assert result['rows'][-2]['validation_delta_fixed']==0
    # Tamper detection before publication/reuse, not mere path presence.
    path=root/'meta/fit/validation_predictions.npz'
    path.write_bytes(b'corrupt fixture')
    with pytest.raises(ValueError): w.results(spec)


def test_original_w5_replay_uses_real_fit_validator_without_retired_bank(tmp_path,monkeypatch):
    """Execute the actual replay program on real historical-format miniature artifacts."""
    pytest.importorskip('weaver.nn.model.ParticleTransformer')
    import io,json,sys
    from contextlib import redirect_stdout
    from pathlib import Path
    from test_luka_embedding_aux import tiny,bank_fixture
    from test_luka_embedding_sweep import fixture_gate,small_counts
    from test_luka_embedding_sweep_debug import fixture
    from test_luka_attention import cache
    from hlt_classification.luka_meta_embedding import source
    from hlt_classification.luka_embedding_sweep import worker,contracts,debug
    from hlt_classification.data.cache_contracts import with_content_hash,write_immutable_json
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    tiny(monkeypatch); small_counts(monkeypatch)
    train,bank,teacher_ref=bank_fixture(tmp_path)
    root=tmp_path/'original'; root.mkdir(); gate=fixture_gate(root)
    gate['anchor']['evidence']['models']={'OFFLINE':dict(checkpoint=teacher_ref)}
    gate['anchor']=with_content_hash(gate['anchor']); gate['parents']['anchor']=gate['anchor']['content_hash']
    gate['embedding']['parents']['anchor']=gate['anchor']['content_hash']; gate['embedding']=with_content_hash(gate['embedding'])
    gate['parents']['embedding']=gate['embedding']['content_hash']; gate=with_content_hash(gate)
    report=contracts.artifact('PREFLIGHT',parents=dict(gate=gate['content_hash'],bank=bank.manifest['content_hash']),
        bank_hash=bank.manifest['content_hash'],train_minutes={a:200 for a in contracts.WEIGHTS})
    gate,report,spec=fixture(tmp_path,monkeypatch,gate=gate,report=report)
    write_immutable_json(root/'preflight/target_manifest.json',bank.manifest)
    folder=Path(spec['root'])/'w5'
    fit=worker.fit(folder/'fit',contracts.arm_gate(gate,'w5'),train,cache('validation','D000'),bank,device='cpu',technical=False)
    result=debug.artifact('RESULTS',parents=dict(spec=spec['content_hash'],fit=fit['content_hash']))
    write_immutable_json(folder/'results.json',result)
    receipt=debug.artifact('SCIENCE',parents=dict(spec=spec['content_hash'],preflight=report['content_hash']),
        arm='w5',weight=5.,job_id='123',outputs=[file_ref(folder/n,root=folder) for n in
            ('results.json','fit/receipt.json','fit/selected.pt','fit/training_report.json','fit/validation_predictions.npz')])
    write_immutable_json(folder/'receipt.json',receipt)
    # This scratch fixture removal models user's already-retired original bank;
    # no real campaign files are touched.
    (tmp_path/'bank/targets.npz').unlink()
    monkeypatch.setattr(sys,'argv',['replay',str(Path(spec['root'])/'campaign_spec.json')])
    captured=io.StringIO()
    with redirect_stdout(captured): exec(compile(source.REPLAY,'w5-replay','exec'),{})
    evidence=json.loads(next(s.split('=',1)[1] for s in captured.getvalue().splitlines() if s.startswith('META_W5=')))
    assert evidence['files']['checkpoint']==file_ref(folder/'fit/selected.pt')
    assert evidence['bank_hash']==bank.manifest['content_hash']
    (folder/'fit/selected.pt').write_bytes(b'corrupt fixture')
    with pytest.raises(ValueError): exec(compile(source.REPLAY,'w5-replay','exec'),{})


def test_meta_update_frozen_teacher_and_real_student_invariant():
    torch.manual_seed(54); student=Mini(); opt=torch.optim.AdamW(student.parameters(),lr=.002)
    adapter=m.Adapter(); ao=torch.optim.Adam(adapter.parameters(),lr=1e-4)
    classifier=torch.nn.Linear(128,11)
    for p in classifier.parameters(): p.requires_grad_(False)
    supports=[dict(inputs=dict(x=torch.randn(4,3)),labels=torch.arange(4),target=torch.randn(4,128)) for _ in range(5)]
    query=dict(inputs=dict(x=torch.randn(4,3)),labels=torch.arange(4))
    before=deepcopy(student.state_dict()); teacher_before=deepcopy(classifier.state_dict())
    result=meta.update(student,opt,adapter,ao,classifier,torch.randn(128),torch.rand(128)+.1,
                       supports,query,[.002]*5,seed=3)
    assert result['grad_norm']>0 and result['teacher_kl']<1e-6 and result['anchor']==0
    assert torch.count_nonzero(adapter.net[-1].weight)>0
    assert all(p.grad is None for p in student.parameters()) and not opt.state
    for k,v in before.items(): torch.testing.assert_close(v,student.state_dict()[k],rtol=0,atol=0)
    for k,v in teacher_before.items(): torch.testing.assert_close(v,classifier.state_dict()[k],rtol=0,atol=0)
    assert all(p.grad is None for p in classifier.parameters())


def test_submission_exact_dependencies_once_and_lost_reply_stops(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from hlt_classification.luka_meta_embedding import campaign,submission
    from hlt_classification.scouting import hcwdl_exact_dag_submission as exact
    spec=fixture_spec(tmp_path/'good'); (tmp_path/'good').mkdir()
    monkeypatch.setattr(campaign,'check',lambda _:None); monkeypatch.setattr(submission,'check_site',lambda _:None)
    plan=submission.submit(spec); calls=[]
    def scheduler(cmd,**kwargs):
        calls.append(cmd); return SimpleNamespace(stdout=str(800+len(calls))+'\n',returncode=0,stderr='')
    monkeypatch.setattr(exact.subprocess,'run',scheduler)
    submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert submission.accepted_jobs(spec)==dict(setup='801',fixed='802',meta='803')
    assert all('--dependency=afterok:801' in cmd for cmd in calls[1:])
    assert not any('${JOB_' in part for cmd in calls for part in cmd)
    submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==3
    other=fixture_spec(tmp_path/'ambiguous'); (tmp_path/'ambiguous').mkdir(); reviewed=submission.submit(other)
    def lost(*a,**kw): raise RuntimeError('reply lost after possible acceptance')
    monkeypatch.setattr(exact.subprocess,'run',lost)
    with pytest.raises(RuntimeError): submission.submit(other,execute=True,plan_hash=reviewed['content_hash'],authorization=c.AUTHORIZATION)
    with pytest.raises(FileExistsError): submission.submit(other,execute=True,plan_hash=reviewed['content_hash'],authorization=c.AUTHORIZATION)


def test_admission_memory_walltime_and_incomplete_measurement_rejected():
    from hlt_classification.luka_meta_embedding import worker
    spec=fixture_spec('/fixture')
    measured=dict(environment={},gpu=dict(total_memory_bytes=40*2**30),peak_rss_bytes=2**30,gpu_peak_bytes=2**30,
        timings=dict(authentication=1.,load_HLT=1.,fixed_train=10.,fixed_validation=1.,meta_train=20.,meta_validation=1.,readback=1.))
    assert worker.admission(spec,measured)['meta']<=720
    for bad in (dict(measured,peak_rss_bytes=100*2**30),dict(measured,gpu_peak_bytes=40*2**30)):
        with pytest.raises(MemoryError): worker.admission(spec,bad)
    bad=deepcopy(measured); bad['timings']['meta_train']=10000.
    with pytest.raises(ValueError,match='walltime'): worker.admission(spec,bad)
    bad=deepcopy(measured); bad['timings']['meta_train']=0.
    with pytest.raises(ValueError,match='timings'): worker.admission(spec,bad)


def test_current_source_rejects_donor_mutation_or_missing_files(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from hlt_classification.luka_meta_embedding import source
    monkeypatch.setattr(source,'_source',lambda *a:dict(content_hash='a'*64))
    def git(cmd,**kw):
        return SimpleNamespace(stdout='\n'.join(sorted(source.REQUIRED)) if 'ls-files' in cmd else 'M\tsrc/shared.py\n')
    monkeypatch.setattr(source.subprocess,'run',git)
    with pytest.raises(ValueError,match='runtime changed'): source.current(tmp_path,'f'*40)
    monkeypatch.setattr(source.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=''))
    with pytest.raises(ValueError,match='complete'): source.current(tmp_path,'f'*40)


def test_calibration_and_meta_sampler_reject_validation_and_overlap(monkeypatch):
    from types import SimpleNamespace
    from test_luka_u100_pretrain import cache
    val=cache('validation'); bank=SimpleNamespace(identities=val.identities,labels=val.labels)
    with pytest.raises(PermissionError): t.calibrate(None,val,bank,{},'cpu')
    with pytest.raises(PermissionError): t.meta_batches(val,bank,np.arange(3),np.arange(3,4),'cpu',1)
    with pytest.raises(PermissionError): t.meta_batches(cache(),bank,np.arange(3),np.arange(2,4),'cpu',1)

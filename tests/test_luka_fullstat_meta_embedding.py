"""FullStat-specific continuation, authentic bank reuse and complete worker flow."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import numpy as np
import pytest
import torch
from test_luka_embedding_aux import tiny, threads
from hlt_classification.data.cache_contracts import (
    load_json, write_immutable_json, with_content_hash, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullstat_meta_embedding import (
    contracts as c, training as t, model as m, campaign, source, worker as w, submission)


def fixture_spec(root,rows=22):
    from test_luka_attention import parent_nodes
    from hlt_classification.jetclass2_delphes.execution import execution_site
    base=dict(next(n for n in parent_nodes() if n['node_id']=='M0HLT'),
              node_id='U100_EMBED_W5',branch='u100_w5',auxiliary_weight=5.)
    reuse=c.artifact('REUSE',counts=dict(c.COUNTS),source_commit=source.DONOR,
        base_node=base,execution_site=execution_site('oscar_l40s'),
        gpu=dict(name='FIXTURE L40S',total_memory_bytes=48*2**30),
        teacher_node=dict(node_id='COARSE_U100',coordinate='U100',context_coordinate=None),
        jobs=dict(export_U100='11',train_w5='12'),anchor=dict(evidence=dict(prepared='a'*64,environment={})),
        bank_manifest={},reference=dict(checkpoint=dict(sha256='e'*64)),teacher_refs={},
        summaries={r:dict(D000=dict(maximum=4)) for r in ('train','validation')})
    return c.artifact('SPEC',root=str(root),project_dir='/project',source_commit='f'*40,
        source_snapshot=dict(content_hash='c'*64),reuse=reuse,available_quota_gib=8,
        **dict(campaign.registration(reuse),counts=dict(train=rows,validation=22,final_test=1500000,unused=105325)))


def test_recipe_fullstat_counts_meta_window_and_original_bank():
    r=c.recipe(); assert r['epochs']==20 and r['teacher']=='COARSE_U100'
    assert r['ce_weight']==1 and r['embedding_weight']==5 and not r['logit_kd']
    assert c.COUNTS==dict(train=3000000,validation=500000,final_test=1500000,unused=105325)
    assert c.learning_rate(2)==pytest.approx(3e-5) and c.learning_rate(20)==pytest.approx(1.5e-6)
    assert [e for e in range(1,21) if c.meta_due(e,100)]==list(range(3,18))
    assert not c.meta_due(3,99) and not c.meta_due(3,0)
    s=fixture_spec('/fixture'); assert s['preflight_jobs']==0 and s['scientific_fits']==2
    assert s['final_test_sealed'] and not s['final_test_accessed']
    assert s['storage_budget_bytes']==8*2**30 and s['minutes']==c.MINUTES
    from hlt_classification.luka_meta_embedding.model import Student
    assert m.Student is Student  # Frozen, already parity-tested native/auxiliary wiring.


def test_split_permutation_independent_five_percent_and_constant_time_sampling(monkeypatch):
    ids=np.random.default_rng(12).integers(0,256,(2000,32),dtype=np.uint8)
    inner,query=t.split(ids)
    assert len(inner)==1900 and len(query)==100 and not np.intersect1d(inner,query).size
    assert not inner.flags.writeable and not query.flags.writeable
    order=np.random.default_rng(3).permutation(len(ids)); _,q2=t.split(ids[order])
    assert set(map(bytes,ids[query]))==set(map(bytes,ids[order][q2]))
    duplicate=ids.copy();duplicate[-1]=duplicate[0]
    with pytest.raises(ValueError): t.split(duplicate)
    seen=[]
    monkeypatch.setattr(t,'batch',lambda cache,bank,ix,device,*,target:seen.append((ix.copy(),target)))
    monkeypatch.setattr(np,'intersect1d',lambda *a:pytest.fail('No 3M-row overlap sort in hot sampling path'))
    t._sample_batches(None,None,inner,query,'cpu',81)
    support=np.concatenate([r[0] for r in seen[:-1]])
    assert len(support)==80 and len(np.unique(support))==80 and set(support)<=set(inner)
    assert len(seen[-1][0])==64 and not seen[-1][1] and set(seen[-1][0])<=set(query)


@pytest.mark.parametrize('arm',c.ARMS)
def test_full20_selection_poor_metrics_and_native_state(monkeypatch,arm):
    pytest.importorskip('weaver.nn.model.ParticleTransformer')
    from test_luka_u100_pretrain import cache
    tiny(monkeypatch); s=fixture_spec('/fixture'); tr=cache();val=cache('validation')
    bank=SimpleNamespace(identities=tr.identities,labels=tr.labels,
        targets=np.random.default_rng(5).normal(size=(len(tr),128)).astype(np.float32))
    model=m.new_model(s['node']); initial=t.cpu_state(model.state_dict()); original=t.evaluate_probabilities; calls=[]
    def poor(y,p):
        calls.append(1);return dict(original(y,p),macro_ovr_auc=.4 if len(calls)==1 else .1)
    monkeypatch.setattr(t,'evaluate_probabilities',poor)
    report,native,full=t.train(model,tr,val,bank,s['node'],arm,'cpu',initial=initial,
        classifier=torch.nn.Linear(128,11),mean=torch.zeros(128),scale=torch.ones(128),parents={})
    assert len(calls)==20 and full['epoch']==1 and report['selected_pass']==1
    assert all(torch.equal(v,full['student'][k]) for k,v in native.items())
    assert not any(k.startswith('predictor.') for k in native)
    t.validate_report(report,s,{},arm)
    bad=deepcopy(report);bad['history'][3]['learning_rate']=1.
    with pytest.raises(ValueError): t.validate_report(with_content_hash(bad),s,{},arm)


def test_calibration_restores_every_native_tensor_excludes_query(monkeypatch):
    pytest.importorskip('weaver.nn.model.ParticleTransformer')
    from test_luka_u100_pretrain import cache
    tiny(monkeypatch);s=fixture_spec('/fixture');tr=cache();model=m.new_model(s['node'])
    native=model.native_state();native['mod.fc.0.bias'].add_(.1);m.initialize(model,native)
    bank=SimpleNamespace(identities=tr.identities,labels=tr.labels,targets=np.ones((len(tr),128),np.float32))
    inner,query=t.split(tr.identities);seen=[];original=tr.batch
    def checked(ix):
        assert not set(ix)&set(query);seen.extend(ix);return original(ix)
    monkeypatch.setattr(tr,'batch',checked)
    state,cal=t.calibrate(model,tr,bank,s['node'],'cpu')
    assert cal['native_unchanged'] and len(seen)==3*len(inner)
    assert all(torch.equal(v,state[k]) for k,v in native.items())
    with pytest.raises(ValueError): m.initialize(model,state)


@pytest.mark.parametrize('role,coordinate',[('final_test','D000'),('unused','D000'),('validation','U100'),('train','OFFLINE')])
def test_forbidden_data_roles_never_reach_metadata(role,coordinate,monkeypatch):
    from hlt_classification.luka_fullstat_u100_embedding import worker as old
    monkeypatch.setattr(old,'metadata',lambda *_:pytest.fail('Forbidden role reached data'))
    with pytest.raises(PermissionError): w.data({},role,coordinate)


def test_resource_admission_real_full_epoch_and_bounds():
    s=fixture_spec('/fixture')
    measured=dict(environment={},gpu=dict(total_memory_bytes=48*2**30),peak_rss_bytes=2**30,gpu_peak_bytes=2**30,
        timings=dict(authentication=1.,load_HLT=1.,fixed_train=1000.,fixed_validation=60.,meta_train=1600.,meta_validation=60.,readback=10.))
    out=w.admission(s,measured); assert out['meta']>out['fixed'] and out['meta']<=2880
    for change in (dict(peak_rss_bytes=180000*1024**2),dict(gpu_peak_bytes=48*2**30)):
        with pytest.raises(MemoryError): w.admission(s,dict(measured,**change))
    for seconds in (0.,float('nan'),10000.):
        bad=deepcopy(measured);bad['timings']['meta_train']=seconds
        with pytest.raises(ValueError): w.admission(s,bad)
    with pytest.raises(ValueError): w.admission(s,dict(measured,environment={'changed':True}))


def test_storage_is_fixed_budget_and_does_not_delete(tmp_path,monkeypatch):
    s=fixture_spec(tmp_path/'fresh');s['available_quota_gib']=80
    monkeypatch.setattr(campaign.shutil,'disk_usage',lambda _:SimpleNamespace(free=7*2**30))
    with pytest.raises(OSError,match='8 GiB'): campaign.storage(s)
    assert not Path(s['root']).exists()


def test_dry_live_dag_requires_exact_review_and_ambiguous_reply_stops(tmp_path,monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as exact
    s=fixture_spec(tmp_path);monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(submission,'check_site',lambda _:None)
    plan=submission.submit(s); assert plan['jobs']==3 and plan['preflight_jobs']==0
    assert [r['dependencies'] for r in plan['commands']]==[[],['setup'],['setup']]
    for r in plan['commands']:
        assert '--partition=gpu' in r['command'] and '--gres=gpu:l40s:1' in r['command']
        assert '--no-requeue' in r['command'] and '--export=NONE' in r['command']
    with pytest.raises(PermissionError): submission.submit(s,execute=True,plan_hash='bad',authorization=c.AUTHORIZATION)
    calls=[]
    def scheduler(cmd,**kwargs):
        calls.append(cmd);return SimpleNamespace(stdout=str(800+len(calls))+'\n',returncode=0,stderr='')
    monkeypatch.setattr(exact.subprocess,'run',scheduler)
    submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert submission.accepted_jobs(s)==dict(setup='801',fixed='802',meta='803')
    assert all('--dependency=afterok:801' in cmd for cmd in calls[1:])
    with pytest.raises(FileExistsError): submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    other=tmp_path/'other';other.mkdir();s2=fixture_spec(other);p2=submission.submit(s2)
    def lost(*a,**kw):raise RuntimeError('possibly submitted; reply lost')
    monkeypatch.setattr(exact.subprocess,'run',lost)
    with pytest.raises(RuntimeError): submission.submit(s2,execute=True,plan_hash=p2['content_hash'],authorization=c.AUTHORIZATION)
    with pytest.raises(FileExistsError): submission.submit(s2,execute=True,plan_hash=p2['content_hash'],authorization=c.AUTHORIZATION)


def test_cuda_production_setup_and_two_full20_continuations(tmp_path,monkeypatch):
    if not torch.cuda.is_available(): pytest.skip('CUDA absent')
    pytest.importorskip('weaver.nn.model.ParticleTransformer')
    from test_luka_attention import cache
    from hlt_classification.luka_fullstat_u100_embedding import banks
    from hlt_classification.context_fusion.worker import save_state
    from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
    from hlt_classification.jetclass2_delphes.runner import predict
    from hlt_classification.jetclass2_delphes import acceptance,model as native_model
    tiny(monkeypatch);monkeypatch.setattr(acceptance,'model_config',native_model.model_config)
    rows=13475;tr=cache(coordinate='D000',rows=rows);u=cache(coordinate='U100',rows=rows)
    val=cache('validation','D000');val.identities[:,-1]=1
    for block in val.blocks:block.identities[:,-1]=1
    root=tmp_path/'study';root.mkdir();s=fixture_spec(root,rows)
    teacher=DelphesParticleTransformer().cuda().eval();teacher_path=tmp_path/'teacher.pt'
    save_state(teacher_path,t.cpu_state(teacher.state_dict()))
    teacher_report=tmp_path/'teacher.json';write_immutable_json(teacher_report,dict(selected_pass=18))
    refs=dict(checkpoint=file_ref(teacher_path),report=file_ref(teacher_report))
    bank=banks.export(tmp_path/'original_bank',teacher,u,refs=refs,device='cuda')
    student=m.new_model(s['node']).cuda().eval();native_path=tmp_path/'w5.pt';save_state(native_path,student.native_state())
    for name,net in (('M0HLT',student),('OFFLINE',teacher),('U100_EMBED_W5',student)):
        atomic_publish_bytes(tmp_path/(name+'.npz'),deterministic_npz_bytes(dict(identities=val.identities,
            labels=val.labels,probabilities=predict(net,val,device='cuda',batch_size=128))))
    s['initial_checkpoint']=file_ref(native_path);s['teacher_refs']=refs
    s['bank_manifest']=file_ref(tmp_path/'original_bank/manifest.json')
    s['reuse'].update(bank_hash=bank['content_hash'],reference=dict(
        predictions=file_ref(tmp_path/'U100_EMBED_W5.npz'),report=file_ref(teacher_report)))
    s['anchor']['evidence'].update(environment=w.installed_environment(),models={n:dict(
        predictions=file_ref(tmp_path/(n+'.npz')),report=file_ref(teacher_report)) for n in ('M0HLT','OFFLINE')})
    s=with_content_hash(s);write_immutable_json(root/'campaign_spec.json',s)
    monkeypatch.setattr(campaign,'check',lambda _:None);monkeypatch.setattr(campaign,'validate_spec',lambda _:None)
    monkeypatch.setattr(source,'current',lambda *a:s['source_snapshot'])
    jobs=dict(setup='1',fixed='2',meta='3')
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a,**kw:jobs)
    monkeypatch.setattr(submission,'successful',lambda *a:None)
    monkeypatch.setattr(w,'execution',lambda *a:'1')
    monkeypatch.setattr(w,'data',lambda s,r,co,**kw:tr if r=='train' else val)
    monkeypatch.setattr(w,'cache_metadata',lambda _:(dict(summaries={r:dict(D000=dict(maximum=4)) for r in ('train','validation')}),{},{}))
    monkeypatch.setattr(w,'memory_snapshot',lambda _:dict(rss_bytes=2**30,peak_rss_bytes=2**30,gpu_peak_bytes=int(torch.cuda.max_memory_allocated())))
    w.run(s,'setup')
    assert w.results(s)['pending']==list(c.ARMS)
    assert not (root/'targets').exists()  # Original bank only, no duplicate/export.
    for arm in c.ARMS:
        monkeypatch.setattr(w,'execution',lambda *a,arm=arm:jobs[arm]);w.run(s,arm)
    reports={a:load_json(root/a/'fit/training_report.json')['report'] for a in c.ARMS}
    assert reports['fixed']['parents']==reports['meta']['parents']
    assert sum(len(r['meta_updates']) for r in reports['meta']['history'])==15
    assert sum(len(r['meta_updates']) for r in reports['fixed']['history'])==0
    assert len(reports['meta']['history'])==20
    with pytest.raises(FileExistsError):w.run(s,'meta')
    monkeypatch.setattr(torch,'load',lambda *a,**kw:pytest.fail('Saved reporting must not open model tensors'))
    result=w.results(s);assert len(result['rows'])==5 and not result['pending']
    assert result['rows'][-2]['validation_delta_fixed']==0.
    assert not result['final_test_accessed'] and result['final_test_sealed']
    path=root/'meta/fit/validation_predictions.npz';path.write_bytes(b'corrupt fixture')
    with pytest.raises(ValueError):w.results(s)


def test_publication_failure_keeps_completed_native_and_full_state(tmp_path,monkeypatch):
    from test_luka_u100_pretrain import cache
    tiny(monkeypatch);s=fixture_spec(tmp_path);tr=cache();val=cache('validation')
    native={'native':torch.tensor([1.])};full={'student':native};report=c.artifact('REPORT',fixture=True)
    monkeypatch.setattr(w,'read_shared',lambda _:(None,None,None,None))
    monkeypatch.setattr(w,'fit_parents',lambda _: {})
    monkeypatch.setattr(t,'train',lambda *a,**kw:(report,native,full))
    def reject(*a,**kw):raise ValueError('injected publication validation failure')
    monkeypatch.setattr(t,'validate_report',reject)
    folder=tmp_path/'fit'
    with pytest.raises(ValueError,match='publication'):w.fit(folder,s,tr,val,None,'fixed','cpu')
    assert load_json(folder/'completed_training.json')==report
    assert torch.equal(torch.load(folder/'selected.pt',weights_only=True)['native'],native['native'])
    assert (folder/'training_state.pt').is_file() and not (folder/'training_report.json').exists()


def test_missing_setup_or_failed_dependency_never_starts_fit(tmp_path,monkeypatch):
    s=fixture_spec(tmp_path);monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a,**kw:dict(setup='1',fixed='2',meta='3'))
    monkeypatch.setattr(w,'fit',lambda *a,**kw:pytest.fail('Unadmitted fit'))
    with pytest.raises(FileNotFoundError):w.run(s,'fixed')
    monkeypatch.setattr(w,'validate_task',lambda *a:dict(job_id='1',content_hash='a'*64))
    def fail(*a):raise ValueError('Dependency FAILED')
    monkeypatch.setattr(submission,'successful',fail)
    with pytest.raises(ValueError,match='FAILED'):w.run(s,'meta')
    assert not (tmp_path/'meta').exists() and not (tmp_path/'fixed').exists()


def test_original_U100_replay_executes_real_fit_receipt_checks(tmp_path,monkeypatch,capsys):
    if not torch.cuda.is_available():pytest.skip('CUDA absent')
    import sys,os
    from test_luka_fullstat_u100_embedding import fixture
    from hlt_classification.luka_fullstat_u100_embedding import worker as old, campaign as oldc, submission as olds
    from hlt_classification.luka_fullsim import science_campaign
    tiny(monkeypatch);s,tr,val,u=fixture(tmp_path,export=False)
    parent=tmp_path/'parent.json'
    write_immutable_json(parent,dict(root='/original',project_dir='/oldsource',nodes=[dict(
        node_id='COARSE_U100',coordinate='U100',context_coordinate=None)]))
    s['anchor']['spec']=file_ref(parent);s['reuse'].update(old_root='/oldgate',old_project='/oldproject')
    s=with_content_hash(s);spec_path=tmp_path/'replay_spec.json';write_immutable_json(spec_path,s)
    monkeypatch.setattr(oldc,'check',lambda _:None)
    monkeypatch.setattr(old,'execution',lambda *a:os.environ['SLURM_JOB_ID'])
    monkeypatch.setattr(old,'metadata',lambda _:(dict(content_hash=tr.foundation_sha256,summaries=s['reuse']['summaries']),None,None))
    monkeypatch.setattr(old,'data',lambda s,r,co,**kw:u if co=='U100' else tr if r=='train' else val)
    monkeypatch.setattr(olds,'accepted_jobs',lambda _:dict(export_U100='101',train_w5='102'))
    monkeypatch.setattr(olds,'successful_export',lambda *a:None)
    monkeypatch.setattr(old,'memory_snapshot',lambda _:dict(peak_rss_bytes=2**30,rss_bytes=2**30,gpu_peak_bytes=max(1,int(torch.cuda.max_memory_allocated()))))
    monkeypatch.setenv('SLURM_JOB_ID','101');old.run(s,'export_U100')
    monkeypatch.setenv('SLURM_JOB_ID','102');old.run(s,'train_w5')
    checked=[];monkeypatch.setattr(science_campaign,'preflight_accounting',lambda job:checked.append(job))
    monkeypatch.setattr(sys,'argv',['replay',str(spec_path)])
    exec(compile(source.REPLAY,'original-U100-replay','exec'),{})
    output=capsys.readouterr().out
    e=json.loads(next(line.split('=',1)[1] for line in output.splitlines() if line.startswith('FULLSTAT_META_REUSE=')))
    assert e['teacher_refs']==s['teacher_refs'] and checked==['102']
    assert e['reference']['checkpoint']==file_ref(Path(s['root'])/'train_w5/selected.pt')
    assert e['bank_manifest']==file_ref(Path(s['root'])/'targets/manifest.json')
    checkpoint=Path(s['root'])/'train_w5/selected.pt';checkpoint.write_bytes(b'corrupt fixture')
    with pytest.raises(ValueError):exec(compile(source.REPLAY,'original-U100-replay','exec'),{})


def test_source_additive_inventory_and_all_shell_examples(monkeypatch):
    import re,sys
    project=Path(__file__).resolve().parents[1]
    assert all((project/p).is_file() for p in source.REQUIRED)
    compile(source.REPLAY,'replay','exec')
    assert "worker.results(s)" in source.REPLAY and "preflight_accounting(jobs['train_w5'])" in source.REPLAY
    bash='C:/Program Files/Git/bin/bash.exe' if sys.platform=='win32' else 'bash'
    for path in source.ENTRYPOINTS:
        if path.endswith('.sh'):subprocess.run([bash,'-n',path],cwd=project,check=True,capture_output=True)
    for doc in ('docs/LUKA_FULLSTAT_META_EMBEDDING_RUNBOOK.md',):
        for block in re.findall(r'^```bash\n(.*?)^```',(project/doc).read_text(),re.M|re.S):
            subprocess.run([bash,'-n'],input=block,text=True,capture_output=True,check=True)
    monkeypatch.setattr(source,'_source',lambda *a:dict(content_hash='a'*64))
    def git(cmd,**kw):
        return SimpleNamespace(stdout='\n'.join(source.REQUIRED) if 'ls-files' in cmd else
            '\n'.join(source.ADDITIONS|source.ENTRYPOINTS) if 'diff' in cmd else '')
    monkeypatch.setattr(subprocess,'run',git)
    assert source.current(project,'f'*40)['content_hash']=='a'*64
    assert not source.compatible_changes({'src/old.py'},{'src/old.py'})
    assert not source.compatible_changes(source.ADDITIONS,set())


def test_registration_rejects_wrong_teacher_scope_and_schedule():
    s=fixture_spec('/fixture')
    for patch in (dict(counts={}),dict(source_commit='0'*40),dict(teacher_node=dict(
            node_id='UE100',coordinate='U100',context_coordinate=None))):
        with pytest.raises(ValueError):campaign.registration(with_content_hash(dict(s['reuse'],**patch)))
    bad=deepcopy(s['reuse']);bad['base_node']['auxiliary_weight']=50
    with pytest.raises(ValueError):campaign.registration(with_content_hash(bad))


def test_create_persists_exact_registration_and_rejects_policy_drift(tmp_path,monkeypatch):
    root=tmp_path/'luka_fullstat_meta_fixture'
    reuse=fixture_spec(root)['reuse']; active=dict(content_hash='c'*64)
    monkeypatch.setattr(source,'current',lambda *a:active)
    monkeypatch.setattr(source,'authenticate',lambda *a:reuse)
    checked=[]
    monkeypatch.setattr(campaign,'protected',lambda *args:checked.append(args))
    monkeypatch.setattr(campaign.shutil,'disk_usage',lambda _:SimpleNamespace(free=30*2**30))
    project=tmp_path/'source';project.mkdir()
    s=campaign.create(root,'parent.json',project,'f'*40,quota_gib=8)
    assert load_json(root/'campaign_spec.json')==s
    assert s['parents']==dict(source=active['content_hash'],reuse=reuse['content_hash'])
    assert checked==[(root,reuse,str(project.resolve()))]
    campaign.validate_spec(s)
    for patch in (dict(preflight_jobs=1),dict(counts={}),dict(available_quota_gib=True),
                  dict(available_quota_gib=7),dict(recipe=dict(s['recipe'],epochs=30))):
        with pytest.raises(ValueError):campaign.validate_spec(with_content_hash(dict(s,**patch)))
    with pytest.raises(FileExistsError):campaign.create(root,'parent.json',project,'f'*40,quota_gib=8)
    assert load_json(root/'campaign_spec.json')==s

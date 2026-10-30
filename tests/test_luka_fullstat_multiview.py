"""FullStat union, fusion, native deployment and complete three-fit workers."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_luka_embedding_aux import tiny as native_tiny, threads
from test_luka_fullstat import fullstat, durable
from test_luka_fullstat_retained import fixture as retained_fixture, fixture_spec as retained_spec
from test_luka_u100_multiview import particles
from hlt_classification.data.cache_contracts import load_json,with_content_hash,write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullstat import campaign as original_campaign
from hlt_classification.luka_fullstat_multiview import contracts as c, campaign, model, training, targets, views, startup, worker, submission, source


def tiny(monkeypatch):
    native_tiny(monkeypatch)
    from hlt_classification.jetclass2_delphes import model as native, salience_learned_model as fusion
    monkeypatch.setattr(fusion,'model_config',native.model_config)
    monkeypatch.setattr(fusion,'INJECTION_BLOCKS',(1,))


def new_reuse(old):
    return c.artifact('REUSE',**{k:v for k,v in old.items() if k not in ('contract','content_hash','schema_version','parents',*c.FLAGS)},
        oracle_node=next(n for n in original_campaign.nodes() if n['node_id']=='OFFLINE'))


def fixture_spec(root):
    old=retained_spec(root);r=new_reuse(old['reuse'])
    return c.artifact('SPEC',parents=dict(source='c'*64,reuse=r['content_hash']),root=str(root),project_dir='/project',
        source_commit='f'*40,source_snapshot=dict(content_hash='c'*64),reuse=r,available_quota_gib=22,**campaign.registration(r))


def fixture(tmp_path,device='cpu'):
    old,tr,val,u,bank=retained_fixture(tmp_path,device);r=new_reuse(old['reuse'])
    root=tmp_path/'luka_fullstat_multiview_fixture';root.mkdir()
    s=c.artifact('SPEC',parents=dict(source='c'*64,reuse=r['content_hash']),root=str(root),project_dir='/project',
        source_commit='f'*40,source_snapshot=dict(content_hash='c'*64),reuse=r,available_quota_gib=22,**campaign.registration(r))
    s=with_content_hash(dict(s,counts=dict(c.COUNTS,train=len(tr),validation=len(val))))
    return s,tr,val,u,bank


def inputs(tr,val,task):
    if task=='train_multi3':return tr,val
    if task=='teacher_FUSION':return views.PairedCache(tr,tr),views.PairedCache(val,val)
    a,b=deepcopy(tr),deepcopy(val)
    a.coordinate_name=b.coordinate_name='CONCAT'
    return a,b


def worker_fixture(tmp_path,monkeypatch,device):
    s,tr,val,u,bank=fixture(tmp_path,device)
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    monkeypatch.setattr(worker,'execution',lambda spec,task,*a:str(100+list(c.TASKS).index(task)))
    monkeypatch.setattr(views,'load',lambda spec,task:(*inputs(tr,val,task),None))
    monkeypatch.setattr(startup.rw,'data',lambda spec,r,co,**kw:u)
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a,**kw:{t:str(100+i) for i,t in enumerate(c.TASKS)})
    monkeypatch.setattr(submission,'successful',lambda *a:None)
    mem=lambda _:dict(peak_rss_bytes=2**30,rss_bytes=2**30,
        gpu_peak_bytes=max(1,int(torch.cuda.max_memory_allocated())) if device=='cuda' else 1)
    monkeypatch.setattr(startup,'memory_snapshot',mem);monkeypatch.setattr(worker,'memory_snapshot',mem)
    return s,tr,val,u,bank


def test_registered_exact_science():
    s=fixture_spec('/fixture');r=c.recipe()
    assert s['counts']==dict(train=3000000,validation=500000,final_test=1500000,unused=105325)
    assert (r['epochs'],r['batch_size'],r['ce_weight'],r['total_embedding_weight'])==(20,128,1,10)
    assert r['weights']=={n:[10,3] for n in c.TEACHERS}
    assert r['base']['maximum_passes']==20 and r['base']['warmup_passes']==2
    assert not any(r[k] for k in ('retained','warm_start','teacher_readout','logit_kd','relational'))
    assert s['scientific_fits']==3 and s['preflight_jobs']==s['setup_jobs']==0
    assert s['nodes']['train_multi3']['deployable'] and s['nodes']['train_multi3']['coordinate']=='D000'
    assert all(not s['nodes'][t]['deployable'] for t in ('teacher_CONCAT','teacher_FUSION'))


@pytest.mark.parametrize('bad',['epochs','weight','retained','coordinate','order','quota','memory','flag','teacher'])
def test_definition_changes_fail_closed(tmp_path,bad):
    s=fixture_spec(tmp_path/'luka_fullstat_multiview_fixture')
    if bad=='epochs':s['recipe']['epochs']=30
    elif bad=='weight':s['recipe']['total_embedding_weight']=5
    elif bad=='retained':s['nodes']['train_multi3']['retained_predictor']=True
    elif bad=='coordinate':s['nodes']['train_multi3']['coordinate']='U100'
    elif bad=='order':s['teacher_order']='afterok:teacher_CONCAT'
    elif bad=='quota':s['available_quota_gib']=21
    elif bad=='memory':s['memory_mb']=180000
    elif bad=='flag':s['final_test_accessed']=True
    else:s['reuse']['oracle_node']['node_id']='UE100'
    with pytest.raises((ValueError,PermissionError)):campaign.validate_spec(with_content_hash(s))


def test_weight_gradients_no_prediction_classifier_and_exact_deployment(tmp_path,monkeypatch):
    tiny(monkeypatch);s,tr,val,_,bank=fixture(tmp_path)
    task='train_multi3';net=model.new_model(s['nodes'][task],task)
    assert not net.retained and not hasattr(net,'combined')
    from hlt_classification.luka_embedding_aux.model import new_model as original_model
    original=original_model(s['reuse']['u100']['base_node'])
    for k,v in original.mod.state_dict().items():
        torch.testing.assert_close(v,net.mod.state_dict()[k],rtol=0,atol=0)
    for k,v in original.predictor.state_dict().items():
        torch.testing.assert_close(v,net.predictors['U100'].state_dict()[k],rtol=0,atol=0)
    banks={n:bank for n in c.TEACHERS};ix=np.arange(len(tr));raw=tr.batch(ix)
    loss,parts=model.loss_for(net,task,raw,ix,banks,'cpu')
    expected=parts['ce']
    for name in c.TEACHERS:expected=expected+(10./3)*parts[name]
    torch.testing.assert_close(loss,expected,rtol=0,atol=0)
    grads=torch.autograd.grad(parts['ce'],tuple(net.predictors.parameters()),allow_unused=True,retain_graph=True)
    assert all(g is None for g in grads)
    main=torch.autograd.grad(parts['ce'],tuple(net.mod.fc.parameters()),retain_graph=True)
    loss.backward();training.finite_gradients(net)
    for p,g in zip(net.mod.fc.parameters(),main):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
    assert all(any(torch.any(p.grad!=0) for p in h.parameters()) for h in net.predictors.values())
    net.eval();net.capture=False;deploy=model.deploy(net,task)
    with torch.no_grad():
        a=model.teachers.forward(net,raw,'cpu',bf16=False);b=model.teachers.forward(deploy,raw,'cpu',bf16=False)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert all(k.startswith('mod.') for k in deploy.state_dict())
    with pytest.raises(PermissionError):model.loss_for(net,task,raw,ix,{'U100':bank},'cpu')


@pytest.mark.parametrize('task',c.TASKS)
def test_all_twenty_epochs_and_poor_finite_results(tmp_path,monkeypatch,task):
    tiny(monkeypatch);s,tr,val,_,bank=fixture(tmp_path);tr,val=inputs(tr,val,task)
    evaluator=training.evaluate_probabilities;calls=[];saved=[]
    def poor(y,p):
        calls.append(1);return dict(evaluator(y,p),macro_ovr_auc=.4 if len(calls)==1 else .1)
    monkeypatch.setattr(training,'evaluate_probabilities',poor)
    net,report,full=training.train(tr,val,banks={} if model.teacher_name(task) else {n:bank for n in c.TEACHERS},
        node=s['nodes'][task],task=task,parents={},device='cpu',save_epoch=lambda e,r,f:saved.append(f))
    assert len(calls)==len(saved)==20 and report['selected_pass']==full['epoch']==1
    assert full['optimizer'] and 'cuda_rng' in full and 'cpu_rng' in full
    training.validate_report(report,s,task,{})
    bad=deepcopy(report);bad['validation_history'][1]['losses']['total']+=.1
    with pytest.raises(ValueError):training.validate_report(with_content_hash(bad),s,task,{})


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_complete_three_production_workers_and_target_banks(tmp_path,monkeypatch,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable; not Oscar acceptance')
    if device=='cpu':
        tiny(monkeypatch);monkeypatch.setattr(startup,'installed_parity',lambda *a,**kw:None)
    s,*_=worker_fixture(tmp_path,monkeypatch,device)
    for i,task in enumerate(c.TASKS):
        worker.run(s,task,device=device)
        worker.completed(s,task)
        folder=Path(s['root'])/task
        assert len(list((folder/'epochs').glob('*/state.pt')))==20
        result=worker.results(s)
        assert len(result['rows'])==4+i and len(result['pending'])==2-i
        assert not result['final_test_accessed']
    state=torch.load(Path(s['root'])/'train_multi3/selected.pt',weights_only=True)
    assert all(k.startswith('mod.') for k in state)
    assert not result['rows'][-1]['privileged']


def test_publication_failure_keeps_training(tmp_path,monkeypatch):
    tiny(monkeypatch);s,*_=worker_fixture(tmp_path,monkeypatch,'cpu')
    monkeypatch.setattr(startup,'installed_parity',lambda *a,**kw:None)
    def fail(*a):raise RuntimeError('Final publication failure')
    monkeypatch.setattr(worker,'validate_fit',fail)
    with pytest.raises(RuntimeError,match='Final publication'):worker.run(s,'teacher_CONCAT',device='cpu')
    folder=Path(s['root'])/'teacher_CONCAT'
    assert len(list((folder/'epochs').glob('*/state.pt')))==20
    assert load_json(folder/'completed_training.json')['passes']==20
    assert not (folder/'task_receipt.json').exists()


def test_concat_physical_union_full_support_and_order():
    a,b=particles(300),particles(301)
    x=views.union(a,b);y=views.union(b.take(np.arange(301)[::-1]),a)
    assert len(x)==601
    np.testing.assert_array_equal(views.build_inputs(x,capacity=601).features,views.build_inputs(y,capacity=601).features)
    assert len(views.union(a,a))==600


@pytest.mark.parametrize('workers',[1,4])
def test_real_root_concat_fullstat_population_and_no_sealed_access(fullstat,durable,workers):
    container,_,f=fullstat;output,meta=durable
    spec=dict(memory_mb=192000,anchor=dict(evidence=dict(raw_container=str(container),cache_root=str(output))))
    before=sorted(p for p in container.rglob('*'))
    tr,val=views.concat(spec,meta,workers=workers)
    for role,v in [('train',tr),('validation',val)]:
        old=views.plain(spec,role,'D000',meta)
        np.testing.assert_array_equal(v.identities,old.identities)
        np.testing.assert_array_equal(v.labels,old.labels)
        assert len(v)==f['counts'][role] and np.all(views.lengths(v)==5)
    assert sorted(p for p in container.rglob('*'))==before
    for role in ('final_test','unused'):
        with pytest.raises(PermissionError):views.plain(spec,role,'D000',meta)


def test_fusion_stress_covers_both_padding_maxima(tmp_path,monkeypatch):
    tiny(monkeypatch);_,tr,val,_,_=fixture(tmp_path)
    paired=views.PairedCache(tr,tr);ix=startup.stress_ids(paired)
    assert len(ix)==128
    raw=paired.batch(ix)
    assert raw['mask'].sum((1,2)).max()==views.lengths(tr).max()


def test_timing_includes_teacher_export_and_no_fallback():
    s=fixture_spec('/fixture')
    a=startup.timing(s,'teacher_CONCAT',.05,.02,.04,50)
    b=startup.timing(s,'teacher_CONCAT',.05,.02,.08,50)
    assert b['conservative_minutes']>a['conservative_minutes']
    with pytest.raises(ValueError):startup.timing(s,'teacher_FUSION',50,20,40,10)
    with pytest.raises(ValueError):startup.timing(s,'train_multi3',.05,.02,.01,10)


def test_new_bank_selected_checkpoint_row_order_and_tamper_guards(tmp_path,monkeypatch):
    from hlt_classification.context_fusion.worker import save_state
    from hlt_classification.data.cache_contracts import load_npz_arrays,deterministic_npz_bytes
    tiny(monkeypatch);s,tr,val,_,_=fixture(tmp_path)
    tr,val=inputs(tr,val,'teacher_CONCAT')
    net=model.new_model(s['nodes']['teacher_CONCAT'],'teacher_CONCAT').eval()
    root=tmp_path/'selected';root.mkdir();save_state(root/'selected.pt',net.state_dict())
    write_immutable_json(root/'report.json',dict(selected_pass=1))
    refs=dict(checkpoint=file_ref(root/'selected.pt'),report=file_ref(root/'report.json'))
    folder=root/'bank';targets.export(folder,net,tr,'CONCAT',refs,s,'cpu')
    bank=targets.Bank(folder,tr,'CONCAT',refs,s)
    assert bank.targets.shape==(len(tr),128) and not bank.targets.flags.writeable
    with pytest.raises(PermissionError):targets.Bank(folder,val,'CONCAT',refs,s)
    with pytest.raises((PermissionError,ValueError)):targets.Bank(folder,tr,'FUSION',refs,s)
    # Even if the file ref and outer hash are updated, reordered TRAIN rows fail.
    m=load_json(folder/'manifest.json');a=load_npz_arrays(folder/'targets.npz')
    a['identities']=a['identities'][::-1].copy()
    (folder/'targets.npz').write_bytes(deterministic_npz_bytes(a))
    with pytest.raises(ValueError):targets.Bank(folder,tr,'CONCAT',refs,s)
    m['payload']=file_ref(folder/'targets.npz',root=folder)
    (folder/'manifest.json').write_text(__import__('json').dumps(with_content_hash(m)))
    with pytest.raises(ValueError,match='join'):targets.Bank(folder,tr,'CONCAT',refs,s)
    with torch.no_grad():next(net.parameters()).add_(.01)
    with pytest.raises(ValueError,match='selected checkpoint'):
        targets.export(root/'wrong_bank',net,tr,'CONCAT',refs,s,'cpu')


def test_registration_creation_preserves_roots_and_pins_source(tmp_path,monkeypatch):
    r=new_reuse(retained_spec(tmp_path)['reuse']);root=tmp_path/'luka_fullstat_multiview_new'
    parent=tmp_path/'old.json'
    write_immutable_json(parent,dict(project_dir=str(tmp_path/'old_source'),root=str(tmp_path/'old_results')))
    r['u100']['anchor']['spec']=file_ref(parent)
    r['u100']['anchor']['evidence'].update({k:str(tmp_path/k) for k in ('cache_root','foundation_root','raw_container')})
    r['parent_spec']=file_ref(parent)
    r['u100']=with_content_hash(r['u100']);r=with_content_hash(r)
    snapshot=dict(content_hash='c'*64)
    monkeypatch.setattr(source,'current',lambda *a:snapshot)
    monkeypatch.setattr(source,'authenticate',lambda *a:r)
    monkeypatch.setattr(source,'check_registered_reuse',lambda *a:None)
    s=campaign.create(root,'parent',tmp_path/'project','f'*40,quota_gib=22)
    assert campaign.check(s,full_reuse=True)==s
    with pytest.raises(FileExistsError):campaign.create(root,'parent',tmp_path/'project','f'*40,quota_gib=22)
    before=(root/'campaign_spec.json').read_bytes()
    monkeypatch.setattr(source,'current',lambda *a:dict(content_hash='d'*64))
    with pytest.raises(ValueError,match='Pinned source'):campaign.check(s)
    assert (root/'campaign_spec.json').read_bytes()==before


def test_concat_bounds_and_prepared_summary_authentication(fullstat,durable,monkeypatch):
    container,_,_=fullstat;output,meta=durable
    s=dict(memory_mb=1,anchor=dict(evidence=dict(raw_container=str(container),cache_root=str(output))))
    with pytest.raises(MemoryError):views.concat(s,meta)
    s['reuse']=dict(u100=dict(summaries={}))
    monkeypatch.setattr(views,'metadata',lambda spec:meta)
    with pytest.raises(ValueError,match='summaries'):views.load(s,'teacher_CONCAT')

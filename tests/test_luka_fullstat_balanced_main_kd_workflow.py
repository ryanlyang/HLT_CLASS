"""One fresh worker with frozen original teachers and complete epoch recovery."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import numpy as np
import pytest
import torch
from test_luka_fullstat_balanced_ladder import threads
from test_luka_fullstat_balanced_ladder_workflow import spec_fixture as old_spec, worker_fixture as old_worker_fixture
from hlt_classification.operations import fullstat_pt_metadata as m
from hlt_classification.luka_fullstat_balanced_ladder import contracts as oc,worker as ow,artifacts as oa
from hlt_classification.luka_fullstat_balanced_main_kd import (
    contracts as c,source,campaign,model,training,worker,startup,artifacts,submission,execution,recovery,inspect)


def spec_fixture(root, parent=None, reuse=None):
    parent=parent or old_spec(root/'parent')
    reuse=reuse or c.artifact('REUSE',references={})
    fields={k:v for k,v in parent.items() if k not in ('content_hash','contract','schema_version','parents',*c.FLAGS)}
    fields.update(root=str(root),reuse=reuse,recipe=c.recipe(),nodes={'DE000':c.node('DE000')},
                  tasks=c.tasks(),available_result_bytes=8*2**30,**campaign.resources())
    return c.artifact('SPEC',**fields)


def test_one_job_exact_dry_submission_and_duplicates(tmp_path,monkeypatch):
    parent=old_spec(tmp_path/'parent');s=spec_fixture(tmp_path,parent)
    monkeypatch.setattr(source,'check',lambda *a,**kw:parent)
    monkeypatch.setattr(source,'current',lambda *a:s['source'])
    campaign.check(s)
    monkeypatch.setattr(submission,'check_site',lambda _:None)
    plan=submission.submit(s)
    assert plan['jobs']==plan['scientific_fits']==1 and plan['preflight_jobs']==plan['setup_jobs']==0
    assert plan['commands'][0]['task_id']=='train_DE000' and plan['commands'][0]['dependencies']==[]
    with pytest.raises(PermissionError):submission.submit(s,execute=True,plan_hash=plan['content_hash'])
    calls=[]
    def fake(args,**kw):calls.append(args);return SimpleNamespace(returncode=0,stdout='123\n',stderr='')
    monkeypatch.setattr(subprocess,'run',fake)
    submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert submission.accepted_jobs(s)=={'train_DE000':'123'} and len(calls)==1
    with pytest.raises(FileExistsError):submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==1


@pytest.mark.parametrize('key,value',[('recipe',{}),('nodes',{}),('tasks',{}),('memory_mb',100000),('final_test_accessed',True),('population_root','other')])
def test_registration_rejects_semantic_changes(tmp_path,monkeypatch,key,value):
    parent=old_spec(tmp_path/'parent');s=spec_fixture(tmp_path,parent)
    monkeypatch.setattr(source,'check',lambda *a,**kw:parent)
    s[key]=value;s=m.with_content_hash(s)
    with pytest.raises((ValueError,PermissionError)):campaign.check(s,live_source=False)


def reference_fixture(tmp_path):
    root=tmp_path/'original';root.mkdir()
    parent=old_spec(root);parent['source_commit']=source.PARENT_COMMIT
    parent['inputs']['evidence'].update(raw_container=str(tmp_path/'raw'),cache_root=str(tmp_path/'cache'),
                                      foundation_root=str(tmp_path/'foundation'))
    parent['inputs']=m.with_content_hash(parent['inputs']);parent=m.with_content_hash(parent)
    path=root/'campaign_spec.json';m.write_immutable_json(path,parent);refs={}
    for stage in c.REFERENCES:
        folder=root/('train_'+stage);folder.mkdir()
        record=oc.artifact('TASK_RECEIPT',parents=dict(spec=parent['content_hash']),
            stage=stage,population=parent['population']['content_hash'],files={})
        p=folder/'task_receipt.json';m.write_immutable_json(p,record)
        refs[stage]=dict(receipt=m.file_ref(p),files={})
    return parent,path,refs


def test_original_authentication_runs_pinned_source_and_rejects_corruption(tmp_path,monkeypatch):
    parent,path,refs=reference_fixture(tmp_path);project=tmp_path/'newproject';project.mkdir()
    monkeypatch.setattr(source,'original_source',lambda *a:(Path(parent['project_dir']),{'source':'witness'}))
    calls=[]
    def original(args,**kw):
        calls.append((args,kw))
        return SimpleNamespace(returncode=0,stdout='BALANCED_MAIN_KD_REUSE='+json.dumps(refs),stderr='')
    monkeypatch.setattr(subprocess,'run',original)
    reuse=source.authenticate(path,project)
    assert source.check(reuse,project)==parent
    args,kw=calls[0]
    assert args[-2:]==[source.REPLAY,str(path.resolve())]
    assert kw['env']['PYTHONPATH']==str(Path(parent['project_dir'])/'src')
    assert kw['env']['PYTHONDONTWRITEBYTECODE']=='1'
    bad=deepcopy(reuse);del bad['references']['DE050']
    with pytest.raises(ValueError):source.check(m.with_content_hash(bad),project)
    # Rehashed but misbound receipt paths cannot substitute other teachers.
    bad=deepcopy(reuse);bad['references']['DE050']=bad['references']['UE100']
    with pytest.raises(ValueError):source.check(m.with_content_hash(bad),project)
    p=Path(refs['DE050']['receipt']['path']);p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError):source.check(reuse,project)
    parent['source_commit']='0'*40
    with pytest.raises(ValueError):source.boundary(m.with_content_hash(parent),path)


def test_registration_authenticates_five_teachers_without_original_de000(tmp_path,monkeypatch):
    parent,path,refs=reference_fixture(tmp_path);project=tmp_path/'project';project.mkdir()
    reuse=c.artifact('REUSE',parents=dict(spec=parent['content_hash']),spec=m.file_ref(path),references=refs,compatible_source={})
    monkeypatch.setattr(source,'current',lambda *a:dict(files={}))
    monkeypatch.setattr(source,'authenticate',lambda *a:reuse)
    monkeypatch.setattr(source,'original_source',lambda *a:(Path(parent['project_dir']),{}))
    s=campaign.create(tmp_path/'study',path,project,'f'*40,8)
    assert s['tasks']=={'train_DE000':[]} and s['continuations']=={} and s['population_root']==parent['population_root']
    assert not (path.parent/'train_DE000').exists()
    with pytest.raises(FileExistsError):campaign.create(tmp_path/'study',path,project,'f'*40,8)
    with pytest.raises(ValueError):campaign.create(tmp_path/'study2',path,project,'f'*40,7)
    with pytest.raises(ValueError):campaign.create(path.parent/'nested',path,project,'f'*40,8)


def fixture(tmp_path,monkeypatch):
    parent,tr,val,sampler=old_worker_fixture(tmp_path/'parent',monkeypatch)
    # Parent baseline fixtures exercise real optimizer/export/bank code. Only
    # scheduler, source/container admission and synthetic candidate pool are mocked.
    for stage in c.REFERENCES:ow.run(parent,'train_'+stage,device='cpu')
    refs={}
    for stage in c.REFERENCES:
        r=oa.completed(parent,stage)
        refs[stage]=dict(receipt=m.file_ref(r['folder']/'task_receipt.json'),files=r['receipt']['files'])
    reuse=c.artifact('REUSE',parents=dict(spec=parent['content_hash']),spec=m.file_ref(Path(parent['root'])/'campaign_spec.json'),
                     references=refs,compatible_source={})
    monkeypatch.setattr(c,'COUNTS',oc.COUNTS);monkeypatch.setattr(c,'DRAWS',oc.DRAWS)
    root=tmp_path/'new';root.mkdir();s=spec_fixture(root,parent,reuse);m.write_immutable_json(root/'campaign_spec.json',s)
    monkeypatch.setattr(campaign,'check',lambda s,**kw:s)
    monkeypatch.setattr(source,'check',lambda *a,**kw:parent)
    monkeypatch.setattr(source,'replay',lambda *a:parent)
    monkeypatch.setattr(source,'current',lambda *a:s['source'])
    monkeypatch.setattr(execution,'validate',lambda *a:'500')
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a,**kw:{'train_DE000':'500'})
    monkeypatch.setattr(submission,'accounting',lambda *a,**kw:None)
    monkeypatch.setattr(worker,'memory_snapshot',lambda _:dict(rss_bytes=1024,peak_rss_bytes=1024,gpu_peak_bytes=0))
    monkeypatch.setattr(startup,'run',lambda spec,stage,*a,**kw:c.artifact('STARTUP',parents=dict(spec=spec['content_hash']),stage=stage,
        installed_weaver_parity=True,finite_connected_gradients=True,classifier_ce_only=False,classifier_ce_kd=True,
        native_deployment_parity=True,full_support=True,probe_discarded=True,matching_serial_parity=True))
    return s,parent,tr,val,sampler


def test_production_worker_restart_publication_and_native_metrics(tmp_path,monkeypatch,capsys):
    s,parent,tr,val,sampler=fixture(tmp_path,monkeypatch)
    before={str(p):m.sha256_file(p) for p in Path(parent['root']).rglob('*') if p.is_file()}
    receipt=worker.run(s,'train_DE000',device='cpu');completed=artifacts.completed(s)
    assert completed['receipt']==receipt and len(receipt['new_epochs'])==20
    assert not (Path(s['root'])/'train_DE000/bank').exists()
    assert before=={str(p):m.sha256_file(p) for p in Path(parent['root']).rglob('*') if p.is_file()}
    state=torch.load(receipt['files']['checkpoint']['path'],weights_only=True)
    assert not any(k.startswith(('predictors.','logit_heads.','matching.')) for k in state)
    report=completed['report'];parents=receipt['fit_parents'];folder=completed['folder']
    refs=artifacts.epoch_refs(folder,parents,'DE000',len(val),sampler)
    restored=artifacts.restore(refs[:10],parents,'DE000',len(val),sampler)
    deps=worker.dependencies_for(s,parent)
    banks={t:worker.banks.Bank(r['folder']/'bank',tr,t,r['refs'],parent) for t,r in deps.items()}
    match=worker.contrastive.Sampler(banks['OFFLINE'].keys(),tr.identities,tr.labels)
    tr.coordinate_name=val.coordinate_name='D000';saved=[]
    net,replayed,best=training.train(tr,val,banks=banks,sampler=sampler,stage='DE000',parents=parents,
        device='cpu',matching=match,resume=restored,save_epoch=lambda e,r,s:saved.append((e,r,s)))
    assert [e for e,_,_ in saved]==list(range(11,21))
    assert replayed['validation']==report['validation'] and replayed['selected_pass']==report['selected_pass']
    for epoch,row,st in saved:
        original=torch.load(folder/'epochs'/f'{epoch:02d}'/'state.pt',weights_only=True)
        assert all(torch.equal(v,st['student'][k]) for k,v in original['student'].items())
        assert row['losses']==report['validation_history'][epoch-1]['losses']
    continuation=recovery.history(s,'DE000');assert continuation['completed_epochs']==20
    fresh=tmp_path/'finalize';fresh.mkdir()
    fields={k:v for k,v in s.items() if k not in ('content_hash','contract','schema_version','parents',*c.FLAGS)}
    fields.update(root=str(fresh),continuations={'DE000':continuation});recovered=c.artifact('SPEC',**fields)
    monkeypatch.setattr(artifacts,'save_epoch',lambda *a,**kw:pytest.fail('No training on completed history'))
    out=worker.run(recovered,'train_DE000',device='cpu')
    assert len(out['inherited_epochs'])==20 and out['new_epochs']==[]
    result=inspect.results(s);assert result['selected_epoch']==report['selected_pass']
    assert result['populations']['natural']['metrics']['summary']==report['validation']
    inspect.progress(s);assert 'Completed validation epochs: 20/20' in capsys.readouterr().out
    corrupted=deepcopy(report);corrupted['validation_history'][0]['losses']['main_kd']+=1
    with pytest.raises(ValueError):training.validate_report(m.with_content_hash(corrupted),'DE000',parents,len(val),sampler)
    # A same-shaped original CE-only report can never become a valid continuation.
    with pytest.raises(ValueError):c.validate(oc.artifact('EPOCH_COMMIT',parents=parents),'EPOCH_COMMIT',parents)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Real integrated objective probe requires CUDA')
def test_real_full_graph_integrated_startup(tmp_path,monkeypatch):
    import time
    from test_luka_attention import cache
    from hlt_classification.luka_fullstat_balanced_ladder import contrastive
    from hlt_classification.luka_fullstat_balanced_ladder.sampling import BalancedSampler
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();n=5500
    tr=cache(coordinate='D000',rows=n);val=cache(role='validation',coordinate='D000',rows=22)
    sampler=BalancedSampler(tr.labels,250+100*((np.arange(n)//11)%18));rng=np.random.default_rng(71)
    banks={t:SimpleNamespace(targets=rng.normal(size=(n,128)).astype(np.float32),
                            probabilities=np.full((n,11),1/11,np.float32)) for t in c.TEACHERS['DE000']}
    keys=banks['OFFLINE'].targets.copy();keys/=np.linalg.norm(keys,axis=1,keepdims=True)
    match=contrastive.Sampler(keys,tr.identities,tr.labels,device='cuda')
    monkeypatch.setattr(startup,'memory_snapshot',lambda _:dict(rss_bytes=2**30,peak_rss_bytes=2**30,
        gpu_peak_bytes=torch.cuda.max_memory_allocated()))
    proof=startup.run(spec_fixture(tmp_path),'DE000',tr,val,banks,sampler,match,'cuda',tmp_path,time.monotonic(),remaining_epochs=0)
    assert proof['classifier_ce_kd'] and not proof['classifier_ce_only'] and proof['probe_discarded']
    assert proof['matching_serial_parity'] and proof['timing']['train_step']>0

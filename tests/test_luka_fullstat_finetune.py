"""Low-LR warm start, frozen-native calibration and complete publication tests."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import math
import numpy as np
import pytest
import torch
from test_luka_embedding_aux import tiny,threads
from test_luka_u100_pretrain import cache
from test_luka_attention import parent_nodes
from hlt_classification.data.cache_contracts import (load_json,write_immutable_json,with_content_hash,
    atomic_publish_bytes,deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.luka_fullstat_fine_tail import contracts as oldc,model as oldm
from hlt_classification.luka_fullstat_finetune import (
    contracts as c,training as t,worker as w,campaign,source,submission)


def fixture_spec(root):
    base=next(n for n in parent_nodes() if n['node_id']=='M0HLT')
    node=oldc.node(base,'DE000')
    reuse=c.artifact('REUSE',counts=dict(c.COUNTS),source_commit=source.DONOR,
        parent_recipe=oldc.recipe('DE000'),base_node=node,imported_banks={k:{} for k in oldc.teachers('DE000')},
        teacher_refs={k:{} for k in oldc.teachers('DE000')},execution_site=execution_site('oscar_l40s'),
        gpu=dict(name='FIXTURE L40S',total_memory_bytes=48*2**30),environment={},
        anchor=dict(evidence=dict(prepared='a'*64)),reference=dict(checkpoint={}),references={},
        summaries={r:dict(D000=dict(maximum=4,resident_bytes=1000)) for r in ('train','validation')},
        jobs=dict(train_DE022='201',train_DE011='202',train_DE000='203'),protected_roots=[],parent_validation={})
    return c.artifact('SPEC',parents=dict(source='c'*64,reuse=reuse['content_hash']),
        root=str(root),project_dir='/project',source_commit='f'*40,source_snapshot=dict(content_hash='c'*64),
        reuse=reuse,available_quota_gib=4,**campaign.registration(reuse))


def data_fixture():
    tr,val=cache(),cache('validation'); ts=oldc.teachers('DE000'); rng=np.random.default_rng(7)
    manifest=oldc.artifact('BUNDLE',prepared=tr.foundation_sha256,stage='DE000',role='train',
        rows=len(tr),teacher_order=ts,temperature=2.)
    bank=SimpleNamespace(manifest=manifest,identities=tr.identities,labels=tr.labels,
        targets={k:rng.normal(size=(len(tr),128)).astype(np.float32) for k in ts},
        probabilities={k:np.full((len(tr),11),1/11,np.float32) for k in ts})
    return tr,val,bank


def test_registration_objective_no_new_teachers_or_preflight():
    s=fixture_spec('/fixture');r=c.recipe()
    assert r['objective']==oldc.recipe('DE000')
    assert r['objective']['teacher_order']==['OFFLINE','UE100','DE066','DE033','M0HLT','DE022','DE011']
    assert (r['epochs'],r['calibration_epochs'],r['batch_size'])==(10,3,128)
    assert s['preflight_jobs']==s['setup_jobs']==0 and s['scientific_fits']==1
    assert s['storage_budget_bytes']==4*2**30 and s['minutes']=={'finetune':1440}
    assert r['objective']['ce_coefficient']==.5 and r['objective']['logit_total']==1 and r['objective']['auxiliary_total']==40
    assert not r['exact_resume'] and not r['objective']['relational']
    assert c.learning_rate(0)==pytest.approx(1.5e-5) and c.learning_rate(10)==pytest.approx(1.5e-6)
    assert all(c.learning_rate(i/100)>=c.learning_rate((i+1)/100) for i in range(1000))
    for p in (-1,11,float('nan'),float('inf')):
        with pytest.raises(ValueError):c.learning_rate(p)


def test_same_objective_and_ce_only_deployed_head(monkeypatch):
    tiny(monkeypatch);s=fixture_spec('/fixture');tr,_,bank=data_fixture()
    model=oldm.new_model(s['node']);model.train();model.capture=True
    ix=np.arange(len(tr));loss,ce,*_=t.old.batch_loss(model,tr.batch(ix),bank,ix,'DE000','cpu')
    expected=torch.autograd.grad(.5*ce,tuple(model.mod.fc.parameters()),retain_graph=True)
    loss.backward()
    for p,g in zip(model.mod.fc.parameters(),expected):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_calibration_native_buffers_frozen_heads_learn_and_repeat(monkeypatch):
    tiny(monkeypatch);s=fixture_spec('/fixture');tr,val,bank=data_fixture()
    native=oldm.new_model(s['node']).native_state(); states=[]
    for _ in range(2):
        model=t.restore(s['node'],native,'cpu');before=t.cpu_state(model.state_dict())
        initial,record=t.calibrate(model,tr,bank,s['node'],'cpu')
        t.validate_calibration(record,len(tr));states.append(initial)
        assert all(torch.equal(v,initial[k]) for k,v in native.items())
        assert any(not torch.equal(v,initial[k]) for k,v in before.items() if k.startswith('predictors.'))
        assert any(not torch.equal(v,initial[k]) for k,v in before.items() if k.startswith('logit_heads.'))
        assert all(p.requires_grad for p in model.parameters())
    assert all(torch.equal(v,states[1][k]) for k,v in states[0].items())
    wrong=deepcopy(tr);wrong.role='validation'
    with pytest.raises(ValueError):t.calibrate(model,wrong,bank,s['node'],'cpu')


@pytest.mark.parametrize('winner',[0,3,10])
def test_all_ten_epochs_baseline_selection_schedule_and_state(monkeypatch,winner):
    tiny(monkeypatch);s=fixture_spec('/fixture');tr,val,bank=data_fixture()
    model=oldm.new_model(s['node']);initial=t.cpu_state(model.state_dict())
    metrics=evaluate_probabilities(val.labels,predict(model,val,device='cpu',batch_size=128))
    baseline=dict(metrics,macro_ovr_auc=.9 if winner==0 else .1)
    s['counts']=dict(s['counts'],train=len(tr),validation=len(val));s['reuse']['parent_validation']=baseline
    real=t.evaluate_probabilities;calls=[];saved=[];orders=[];original_batch=tr.batch
    def metric(y,p):
        calls.append(1);return dict(real(y,p),macro_ovr_auc=.8 if len(calls)==winner else .2)
    monkeypatch.setattr(t,'evaluate_probabilities',metric)
    def batch(ix):orders.append(ix.copy());return original_batch(ix)
    monkeypatch.setattr(tr,'batch',batch)
    report,native,full=t.train(tr,val,bank,s['node'],initial,baseline,{},'cpu',
        save_epoch=lambda e,r,v:saved.append((e,deepcopy(r),v)))
    t.validate_report(json.loads(json.dumps(report,sort_keys=True)),s,{})
    assert len(calls)==10 and [r[0] for r in saved]==list(range(11))
    assert report['selected_pass']==winner and full['epoch']==winner
    assert report['baseline_retained'] is (winner==0)
    assert set(native)==set(model.native_state())
    expected=np.random.default_rng(np.random.SeedSequence([s['node']['sampler_seed'],21])).permutation(len(tr))
    np.testing.assert_array_equal(orders[0],expected)
    for badfield in ('learning_rate','total_loss','absolute_epoch'):
        bad=deepcopy(report);bad['validation_history'][0][badfield]+=1
        with pytest.raises(ValueError):t.validate_report(with_content_hash(bad),s,{})
    if winner==0:assert all(torch.equal(v,initial[k]) for k,v in native.items())


@pytest.mark.parametrize('bad',['teacher','source','count','node','site','memory'])
def test_registration_rejects_inapplicable_reuse(bad):
    s=fixture_spec('/fixture');r=deepcopy(s['reuse'])
    if bad=='teacher':r['imported_banks'].pop('DE011')
    elif bad=='source':r['source_commit']='0'*40
    elif bad=='count':r['counts']['train']=299234
    elif bad=='node':r['base_node']['coordinate']='D011'
    elif bad=='site':r['execution_site']['partition']='other'
    else:r['summaries']['train']['D000']['resident_bytes']=200*2**30
    with pytest.raises((ValueError,MemoryError)):campaign.registration(with_content_hash(r))


def test_resource_timing_and_memory_fail_closed():
    s=fixture_spec('/fixture')
    a=w.timing(s,.05,.01,100)
    assert 0<a['conservative_minutes']<1440 and not a['full_epoch_measured']
    for vals in ((1.,1.,0.),(float('nan'),1.,0.),(.1,.1,-1)):
        with pytest.raises(ValueError):w.timing(s,*vals)
    with pytest.raises(MemoryError):w.check_memory(s,dict(peak_rss_bytes=170000*1024**2,gpu_peak_bytes=1))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Genuine local installed-Weaver CUDA worker')
def test_complete_worker_calibration_ten_epochs_recovery_and_results(tmp_path,monkeypatch):
    # Full contracted Weaver architecture; only population and cluster boundaries are fixtures.
    root=tmp_path/'luka_fullstat_finetune_fixture';root.mkdir()
    s=fixture_spec(root);tr,val,bank=data_fixture();node=s['node']
    initial=oldm.new_model(node).native_state();oldroot=tmp_path/'old';oldroot.mkdir()
    save_state(oldroot/'selected.pt',initial)
    model=t.restore(node,initial,'cuda')
    probabilities=predict(model,val,device='cuda',batch_size=128)
    metrics=evaluate_probabilities(val.labels,probabilities)
    report=oldc.artifact('TRAINING_REPORT',selected_pass=20,validation=metrics,node=node)
    write_immutable_json(oldroot/'training_report.json',report)
    atomic_publish_bytes(oldroot/'validation_predictions.npz',deterministic_npz_bytes(dict(
        identities=val.identities,labels=val.labels,probabilities=probabilities)))
    write_immutable_json(oldroot/'bundle.json',bank.manifest)
    refs={k:file_ref(oldroot/f) for k,f in [('checkpoint','selected.pt'),('report','training_report.json'),
        ('predictions','validation_predictions.npz')]}
    s['counts']=dict(s['counts'],train=len(tr),validation=len(val))
    s['initial_checkpoint']=refs['checkpoint'];s['references']={n:refs for n in ('M0HLT','OFFLINE','FINE_TAIL_DE000')}
    s['reuse']['reference']=refs;s['reuse']['parent_validation']=metrics
    s['reuse']['bundle']=file_ref(oldroot/'bundle.json')
    s['anchor']['evidence']['prepared']=tr.foundation_sha256
    for role,cache_ in [('train',tr),('validation',val)]:
        s['reuse']['summaries'][role]['D000']['maximum']=w.longest(cache_)[1]
    s=with_content_hash(s);write_immutable_json(root/'campaign_spec.json',s)
    original_refs={p.name:file_ref(p) for p in oldroot.iterdir()}
    monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(w,'execution',lambda *a:'901')
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:dict(finetune='901'))
    monkeypatch.setattr(submission,'successful',lambda *a:None)
    monkeypatch.setattr(w,'load_inputs',lambda _:(tr,val,bank))
    monkeypatch.setattr(w,'memory_snapshot',lambda _:dict(peak_rss_bytes=2**30,gpu_peak_bytes=int(torch.cuda.max_memory_allocated())))
    result=w.run(s,'finetune');assert result==w.completed(s)
    final,arrays=w.validate_fit(s)
    assert final['passes']==10 and len(list((root/'finetune/epochs').iterdir()))==11
    assert final['validation']['macro_ovr_auc']>=metrics['macro_ovr_auc']
    selected=torch.load(root/'finetune/selected.pt',weights_only=True,map_location='cpu')
    assert not any(k.startswith(('predictors.','logit_heads.')) for k in selected)
    assert len(w.results(s)['rows'])==4
    assert original_refs=={p.name:file_ref(p) for p in oldroot.iterdir()}
    with pytest.raises(FileExistsError):w.run(s,'finetune')
    path=root/'finetune/epochs/04/state.pt';path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError):w.completed(s)


def test_source_scope_and_cli_bash_syntax():
    import ast,subprocess,sys
    project=Path(__file__).resolve().parents[1]
    assert source.compatible_changes(source.ADDITIONS,source.ADDITIONS)
    assert not source.compatible_changes({'src/old.py'},{'src/old.py'})
    assert not source.compatible_changes(source.ADDITIONS,set())
    bash='C:/Program Files/Git/bin/bash.exe' if sys.platform=='win32' else 'bash'
    for path in source.ADDITIONS|source.ENTRYPOINTS:
        if path.endswith('.py'):ast.parse((project/path).read_text())
        else:subprocess.run([bash,'-n',path],cwd=project,check=True,capture_output=True)

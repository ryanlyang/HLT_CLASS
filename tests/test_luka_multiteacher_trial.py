"""Five small-data arms: exact science, durable joins, direct DAG and real Weaver."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import re
import shutil
import subprocess
import sys
import numpy as np
import pytest
import torch
from test_luka_embedding_aux import tiny, threads
from test_luka_attention import cache, parent_nodes
from test_luka_attention_reference_archive import saved, rebind_training
from test_luka_u100_embedding import inherited_fixture
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays, with_content_hash,
    write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import _forward, predict
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_embedding_aux.model import new_model as single_model
from hlt_classification.luka_embedding_aux import contracts as anchor_contracts
from hlt_classification.luka_d066_embedding import contracts as reference_contracts
from hlt_classification.luka_multiteacher_trial import (
    contracts as c, source, campaign, banks, model, training, submission, worker)


def fixture(path, *, rows=22, export=False):
    path.mkdir(exist_ok=True)
    parent=dict(nodes=parent_nodes()); write_immutable_json(path/'parent.json',parent)
    root=path/'trial'; root.mkdir()
    anchor=dict(content_hash='a'*64,spec=file_ref(path/'parent.json'),evidence=dict(prepared='a'*64,models={}))
    teachers={}; views={}
    for i,(name,co) in enumerate(c.TEACHERS.items()):
        torch.manual_seed(i+10); teacher=DelphesParticleTransformer()
        save_state(path/(name+'.pt'),teacher.state_dict())
        t=c.artifact('TEACHER',parents=dict(anchor=anchor['content_hash']),model_id=name,
            coordinate=co,role='train',prepared='a'*64,files=dict(checkpoint=file_ref(path/(name+'.pt'))))
        teachers[name]=t; views[co]=cache(coordinate=co,rows=rows)
        if export: banks.export(root/'targets'/name,teacher,views[co],t,device='cpu')
    fields=dict(root=str(root),project_dir='/project',source_commit='b'*40,source_snapshot=dict(content_hash='c'*64),
        anchor=anchor,teachers=teachers,reference=dict(content_hash='d'*64,files={}),u100_reference=None,
        resources=dict(content_hash='e'*64,measured_w5_minutes=190),available_quota_gib=4,
        **campaign.policy(parent,dict(measured_w5_minutes=190)))
    fields.update(counts=dict(train=rows,validation=22),original50k_rows=22)
    spec=c.artifact('SPEC',parents=campaign.parents(fields),**fields)
    write_immutable_json(root/'campaign_spec.json',spec)
    tr=cache(coordinate='D000',rows=rows); val=cache('validation',coordinate='D000')
    val.identities[:,-1]=1
    for b in val.blocks: b.identities[:,-1]=1
    bank=banks.load(spec,tr,publish=True) if export else None
    return spec,tr,val,views,bank


def test_exact_five_scientific_recipes_and_six_tier3_jobs():
    policy=campaign.policy(dict(nodes=parent_nodes()),dict(measured_w5_minutes=190))
    assert policy['counts']==dict(train=299234,validation=74895)
    assert policy['scientific_fits']==5 and policy['preflight_jobs']==policy['technical_training_epochs']==0
    assert policy['export_minutes']==360 and policy['train_minutes']==440
    assert policy['execution_site']['partition']=='tier3' and policy['memory_mb']==90000
    assert policy['storage_budget_bytes']==4*2**30 and not policy['automatic_retry']
    assert c.TEACHERS==dict(OFFLINE='OFFLINE',COARSE_U100='U100',LOGIT_D066='D066',LOGIT_D033='D033')
    for arm,weight in c.ARMS.items():
        r=policy['recipes'][arm]; n=policy['nodes'][arm]
        assert r['epochs']==30 and r['batch_size']==128 and r['total_weight']==weight
        assert sum(r['weights'].values())==weight and set(r['weights'].values())=={weight/4}
        assert not r['logit_kd'] and not r['warm_start']
        assert n['initialization_seed']==parent_nodes()[0]['initialization_seed']
        assert n['teacher'] is None and n['context_coordinate'] is None and n['deployable']
        assert n['coordinate']=='D000'
        assert r['targets']=={t:'COARSE_U100' if arm=='u100x4_w5' else t for t in c.TEACHERS}
        assert r['relation']['enabled']==(arm=='multi_w5_rel')
    for minutes in (0,True,1.5,691):
        with pytest.raises(ValueError):campaign.policy(dict(nodes=parent_nodes()),dict(measured_w5_minutes=minutes))
    with pytest.raises(ValueError):c.recipe('extra')
    with pytest.raises(PermissionError):c.validate(with_content_hash(dict(c.artifact('BANK'),final_test_accessed=True)),'BANK')


@pytest.mark.parametrize('arm',list(c.ARMS))
def test_pointwise_exact_registered_weights_detach_and_head_control(arm):
    logits=torch.randn(7,11,requires_grad=True); labels=torch.arange(7)
    p={t:torch.randn(7,128,requires_grad=True) for t in c.TEACHERS}
    targets={t:torch.randn(7,128,requires_grad=True) for t in c.TEACHERS}
    loss,ce,aux,rel=training.objective(logits,labels,p,targets,arm)
    r=c.recipe(arm)
    torch.testing.assert_close(loss,ce+sum(r['weights'][t]*aux[t] for t in c.TEACHERS),rtol=0,atol=0)
    assert rel==0; loss.backward()
    for t,target in r['targets'].items():
        assert targets[t].grad is None
        expected=r['weights'][t]*(p[t].detach()-targets[target].detach()).clamp(-1,1)/p[t].numel()
        torch.testing.assert_close(p[t].grad,expected)
    with pytest.raises(ValueError):training.objective(logits,labels,{},targets,arm)
    bad=dict(targets,COARSE_U100=torch.full((7,128),float('nan')))
    with pytest.raises(ValueError):training.objective(logits,labels,p,bad,arm)
    if arm!='multi_w5_rel':
        with pytest.raises(ValueError):training.objective(logits,labels,p,targets,arm,beta=.1)


def test_relation_exact_kl_self_mask_coordinate_separation_and_fp32_gradients():
    p=torch.randn(3,128,requires_grad=True); t=torch.randn(3,128,requires_grad=True)
    a=torch.randn(2048,128,requires_grad=True); valid=torch.ones(3,2048,dtype=torch.bool)
    valid[0,1]=False
    loss=training.reference_kl(p,t,a,valid)
    pp=torch.nn.functional.normalize(p,dim=-1); tt=torch.nn.functional.normalize(t,dim=-1)
    aa=torch.nn.functional.normalize(a,dim=-1)
    lp=torch.log_softmax((pp@aa.T/.2).masked_fill(~valid,-torch.inf),-1)
    lt=torch.log_softmax((tt@aa.T/.2).masked_fill(~valid,-torch.inf),-1)
    expected=(lt.exp()*torch.where(valid,lt-lp,torch.zeros_like(lt))).sum(-1).mean()
    torch.testing.assert_close(loss,expected)
    loss.backward(); assert p.grad.abs().sum()>0 and t.grad is None and a.grad is None
    assert training.reference_kl(t,t,a,valid).item()==pytest.approx(0,abs=1e-7)
    old=training.reference_kl(p[:1],t[:1],a,valid[:1]); changed=a.detach().clone();changed[1]=100
    torch.testing.assert_close(old,training.reference_kl(p[:1],t[:1],changed,valid[:1]))
    with pytest.raises(ValueError):training.reference_kl(p,t,a,torch.zeros_like(valid))
    preds={name:p for name in c.TEACHERS};targets={name:t+idx for idx,name in enumerate(c.TEACHERS)}
    anchors={name:a+idx for idx,name in enumerate(c.TEACHERS)}
    logits=torch.randn(3,11,requires_grad=True)
    value,ce,aux,relation=training.objective(logits,torch.arange(3),preds,targets,'multi_w5_rel',
        beta=.25,anchors=anchors,valid=valid)
    expected=sum(training.reference_kl(p,targets[n],anchors[n],valid) for n in c.TEACHERS)/4
    torch.testing.assert_close(relation,expected)
    torch.testing.assert_close(value,ce+1.25*sum(aux.values())+.25*relation)


def test_sampler_balanced_unique_rotating_deterministic_and_isolated():
    labels=np.repeat(np.arange(11,dtype=np.int64),188)
    sampler=training.AnchorSampler(labels); state=np.random.get_state()
    for s in range(11):
        rows=sampler.sample(1,s,s)
        assert len(rows)==len(np.unique(rows))==2048
        expected=np.full(11,186); expected[[(2*s)%11,(2*s+1)%11]]+=1
        np.testing.assert_array_equal(np.bincount(labels[rows]),expected)
        np.testing.assert_array_equal(rows,sampler.sample(1,s,s))
    np.testing.assert_array_equal(state[1],np.random.get_state()[1])
    assert not np.array_equal(sampler.sample(1,0,0),sampler.sample(2,0,0))
    with pytest.raises(ValueError):training.AnchorSampler(np.tile(np.arange(11,dtype=np.int64),186))
    with pytest.raises(ValueError):training.AnchorSampler(labels.astype(np.float32))


@pytest.mark.parametrize('bf16',[False,True])
def test_full_installed_native_parity_head_removal_rng_and_gradients(monkeypatch,bf16):
    if bf16 and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    device='cuda' if bf16 else 'cpu'; n=c.node(dict(nodes=parent_nodes()),'multi_w5')
    old=single_model(n); rng=torch.get_rng_state().clone(); new=model.new_model(n)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,old.native_state()[k]) for k,v in new.native_state().items())
    assert all(torch.equal(v,old.predictor.state_dict()[k]) for k,v in new.predictors['COARSE_U100'].state_dict().items())
    raw=cache(coordinate='D000').batch(np.arange(4)); old.to(device).eval();new.to(device).eval()
    with parity_backend(device),torch.no_grad():
        a=_forward(old,raw,device,bf16=bf16);b=_forward(new,raw,device,bf16=bf16)
        new.capture=True;d=_forward(new,raw,device,bf16=bf16)
        torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(a,d,rtol=0,atol=0)
    new.train(); logits=_forward(new,raw,device,bf16=bf16)
    loss,*_=training.objective(logits,torch.from_numpy(raw['labels']).to(device),new.predictions,
        {t:torch.randn(4,128,device=device) for t in c.TEACHERS},'multi_w5')
    loss.backward()
    for prefix in ('mod.',*[f'predictors.{n}.' for n in c.TEACHERS]):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
                   for n,p in new.named_parameters() if n.startswith(prefix))
    native=DelphesParticleTransformer().to(device).eval();native.load_state_dict(new.native_state(),strict=True)
    new.capture=False;new.eval()
    for h in new.predictors.values():monkeypatch.setattr(h,'forward',lambda *a:pytest.fail('Inference head executed'))
    with parity_backend(device),torch.no_grad():
        torch.testing.assert_close(_forward(native,raw,device,bf16=bf16),_forward(new,raw,device,bf16=bf16),rtol=0,atol=0)


def test_four_banks_train_normalization_identity_binding_and_corruption(tmp_path,monkeypatch):
    tiny(monkeypatch);s,tr,val,views,b=fixture(tmp_path,export=True)
    for name in c.TEACHERS:
        np.testing.assert_allclose(b.targets[name].mean(0),0,atol=1e-6)
        assert b.targets[name].dtype==np.float32 and np.max(b.targets[name].std(0))<=1.00001
        wrong=deepcopy(tr);wrong.identities=tr.identities[::-1].copy()
        with pytest.raises(ValueError):banks.load_one(Path(s['root'])/'targets'/name,wrong,s['teachers'][name])
    assert not np.array_equal(b.targets['OFFLINE'],b.targets['COARSE_U100'])
    t=s['teachers']['LOGIT_D033']; teacher=worker.load_native(t['files']['checkpoint'],'cpu')
    with pytest.raises(PermissionError):banks.export(tmp_path/'invalid',teacher,val,t,device='cpu')
    with pytest.raises(ValueError,match='weights'):
        banks.export(tmp_path/'wrong',DelphesParticleTransformer(),views['D033'],t,device='cpu')
    with pytest.raises(PermissionError):banks.data(s,'test','D000',None)
    (Path(s['root'])/'targets/LOGIT_D033/targets.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError):banks.load(s,tr)


@pytest.mark.parametrize('field,value',[('rows',21),('role','validation'),('coordinate','U100'),('width',127)])
def test_resigned_wrong_bank_rejected(tmp_path,monkeypatch,field,value):
    tiny(monkeypatch);s,tr,*_=fixture(tmp_path,export=True)
    p=Path(s['root'])/'targets/OFFLINE/manifest.json'
    p.write_text(json.dumps(with_content_hash(dict(load_json(p),**{field:value}))))
    with pytest.raises(ValueError):banks.load_one(p.parent,tr,s['teachers']['OFFLINE'])


def test_full30_fit_selected_native_readback_and_poor_finite_metric_publication(tmp_path,monkeypatch):
    tiny(monkeypatch);s,tr,val,_,b=fixture(tmp_path,export=True);arm='multi_w5'
    with parity_backend('cpu'):worker.fit(Path(s['root'])/arm,s,arm,tr,val,b,'cpu')
    report,_=worker.validate_fit(s,arm,b.manifest['content_hash'])
    assert report['passes']==len(report['validation_history'])==30
    assert not any(k.startswith('predictors.') for k in torch.load(Path(s['root'])/arm/'selected.pt',weights_only=True))
    bad=deepcopy(report);bad['validation_history'][0]['weighted_auxiliary_loss']=0
    with pytest.raises(ValueError):training.validate_report(with_content_hash(bad),s,arm,b.manifest['content_hash'])
    original=training.evaluate_probabilities; calls=[]
    def poor(labels,p):
        calls.append(1);return dict(original(labels,p),macro_ovr_auc=.4 if len(calls)==1 else .1,accuracy=0.)
    monkeypatch.setattr(training,'evaluate_probabilities',poor)
    report,_=training.train(model.new_model(s['nodes'][arm]),tr,val,bank=b,node=s['nodes'][arm],device='cpu')
    assert report['passes']==30 and report['selected_pass']==1
    training.validate_report(report,s,arm,b.manifest['content_hash'])


def test_exact_dag_no_preflight_then_six_dependencies_and_no_retry(tmp_path,monkeypatch):
    tiny(monkeypatch);s,*_=fixture(tmp_path)
    monkeypatch.setattr(campaign,'check',lambda _:None);monkeypatch.setattr(submission,'check_site',lambda _:None)
    plan=submission.submit(s)
    assert plan['jobs']==6 and plan['preflight_jobs']==0
    assert [r['task_id'] for r in plan['commands']]==['export_targets',*c.ARMS]
    for row in plan['commands']:
        assert '--partition=tier3' in row['command'] and '--gres=gpu:a100:1' in row['command']
        assert '--kill-on-invalid-dep=yes' in row['command'] and '--no-requeue' in row['command']
    assert all('--dependency=afterok:${JOB_export_targets}' in row['command'] for row in plan['commands'][1:])
    with pytest.raises(PermissionError):submission.submit(s,execute=True,plan_hash='bad',authorization=c.AUTHORIZATION)
    calls=[]
    def fake(args,**kw):
        calls.append(args);return SimpleNamespace(returncode=0,stdout=str(100+len(calls))+'\n',stderr='')
    monkeypatch.setattr(subprocess,'run',fake)
    submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert submission.accepted_jobs(s)==dict(zip(['export_targets',*c.ARMS],map(str,range(101,107))))
    assert all('--dependency=afterok:101' in args for args in calls[1:])
    with pytest.raises(FileExistsError):submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==6


def test_exact_export_accounting_and_no_science_on_missing_export(tmp_path,monkeypatch):
    tiny(monkeypatch);s,*_=fixture(tmp_path)
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:dict(export_targets='101',multi_w5='102'))
    monkeypatch.setattr(submission.getpass,'getuser',lambda:'ryreu')
    valid='101|lukamt_export_targets|ryreu|reu-aisocial|COMPLETED|0:0\n'
    for text in ('',valid.replace('COMPLETED','FAILED'),valid.replace('0:0','1:0'),valid.replace('ryreu','other')):
        monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=text))
        with pytest.raises(ValueError):submission.successful_export(s,'101')
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=valid))
    submission.successful_export(s,'101')
    monkeypatch.setattr(campaign,'check',lambda _:None);monkeypatch.setattr(worker,'execution',lambda *a:'102')
    monkeypatch.setattr(worker,'fit',lambda *a,**kw:pytest.fail('No fit without completed export'))
    with pytest.raises(ValueError,match='incomplete'):worker.run(s,'multi_w5')
    assert not (Path(s['root'])/'multi_w5_attempt').exists()


def test_source_guards_and_original_archive_explicit_checkpoint_binding(monkeypatch):
    project=Path(__file__).resolve().parents[1]
    monkeypatch.setattr(source,'_source',lambda *a:dict(content_hash='a'*64))
    monkeypatch.setattr(subprocess,'run',lambda cmd,**kw:SimpleNamespace(stdout='\n'.join(source.REQUIRED)
        if 'ls-files' in cmd else '\n'.join('A\t'+p for p in source.ADDITIONS)))
    assert source.current(project,'b'*40)['content_hash']=='a'*64
    monkeypatch.setattr(subprocess,'run',lambda cmd,**kw:SimpleNamespace(stdout='\n'.join(source.REQUIRED)
        if 'ls-files' in cmd else 'M\tsrc/hlt_classification/jetclass2_delphes/runner.py'))
    with pytest.raises(ValueError):source.current(project,'b'*40)
    compile(source.ARCHIVE_CHECK,'original-archive','exec');compile(source.U100_REFERENCE,'u100','exec')
    assert 'read_archive' in source.ARCHIVE_CHECK and 'validate_file_ref(matches[0],root=root)' in source.ARCHIVE_CHECK
    assert "a['archive']['training_receipts'][task]" in source.ARCHIVE_CHECK
    assert "'LOGIT_D066','D066'" in source.ARCHIVE_CHECK and "'LOGIT_D033','D033'" in source.ARCHIVE_CHECK
    assert 'preflight/export_ANCHOR' not in source.ARCHIVE_CHECK


def authenticated_fixture(path):
    """Real hashes/resource-parent checks; scientific replay itself has its own test."""
    path.mkdir(exist_ok=True); parent=dict(nodes=parent_nodes(),project_dir=str(path/'old_project'),root=str(path/'parent/science'))
    write_immutable_json(path/'parent/science/campaign_spec.json',parent)
    write_immutable_json(path/'old_payload.json',dict(fixture=True))
    write_immutable_json(path/'reference/science/campaign_spec.json',dict(fixture=True))
    ref=file_ref(path/'old_payload.json')
    files={k:ref for k in ('checkpoint','report','predictions','receipt','diagnostics')}
    anchor=anchor_contracts.artifact('ANCHOR',spec=file_ref(path/'parent/science/campaign_spec.json'),
        evidence=dict(prepared='a'*64,cache='b'*64,prepared_root=str(path/'prepared'),raw_container=str(path/'raw'),
            environment=dict(fixture=True),models={n:files for n in ('M0HLT','OFFLINE')}))
    reference=reference_contracts.artifact('REFERENCE',parents=dict(anchor=anchor['content_hash']),
        spec=file_ref(path/'reference/science/campaign_spec.json'),
        node=source.donor_node(parent),files={k:ref for k in ('receipt','result','report','predictions')})
    reference,resources=inherited_fixture(path/'inherited',dict(anchor=anchor,reference=reference))
    teachers={n:c.artifact('TEACHER',parents=dict(anchor=anchor['content_hash']),model_id=n,
        coordinate=co,node=dict(node_id=n,coordinate=co,context_coordinate=None),role='train',
        prepared='a'*64,spec=anchor['spec'],selected_native_checkpoint_revalidated=True,
        files={k:ref for k in ('checkpoint','report','predictions','receipt')}) for n,co in c.TEACHERS.items()}
    return anchor,parent,reference,resources,teachers


def test_create_import_validation_no_extra_preflight_and_fail_before_new_root(tmp_path,monkeypatch):
    anchor,parent,ref,res,teachers=authenticated_fixture(tmp_path/'donor')
    monkeypatch.setattr(source,'current',lambda *a:dict(content_hash='c'*64))
    monkeypatch.setattr(source,'authenticate',lambda _: (anchor,parent))
    monkeypatch.setattr(source,'teachers',lambda *a: teachers)
    monkeypatch.setattr(source,'reference',lambda *a:ref)
    monkeypatch.setattr(source,'resources',lambda *a:res)
    monkeypatch.setattr(source,'optional_u100',lambda *a:None)
    args=[anchor['spec']['path'],tmp_path/'attention/science/spec.json',ref['spec']['path'],tmp_path/'project','f'*40]
    # Ref location would protect the donor folder; distinct study must be used.
    for quota in (0,3,True):
        root=tmp_path/('bad_'+str(quota))
        with pytest.raises(ValueError):campaign.create(root,*args,available_quota_gib=quota)
        assert not root.exists()
    root=tmp_path/'run';spec=campaign.create(root,*args,available_quota_gib=4)
    campaign.check(spec);assert spec['preflight_jobs']==0
    assert len(spec['tasks'])==6 and not list(root.glob('*preflight*'))
    with pytest.raises(FileExistsError):campaign.create(root,*args,available_quota_gib=4)
    bad=deepcopy(spec);bad['teachers']['LOGIT_D033']=with_content_hash(dict(bad['teachers']['LOGIT_D033'],coordinate='D066'))
    with pytest.raises(ValueError,match='Teacher identity'):source.imports(bad)
    bad=deepcopy(spec);bad['reference']['node']['auxiliary_weight']=10
    with pytest.raises(ValueError):source.imports(bad)
    bad=deepcopy(spec);bad['resources']['gpu']={}
    with pytest.raises(ValueError):source.imports(bad)


def test_ambiguous_sbatch_failure_never_automatically_retried(tmp_path,monkeypatch):
    tiny(monkeypatch);s,*_=fixture(tmp_path)
    monkeypatch.setattr(campaign,'check',lambda _:None);monkeypatch.setattr(submission,'check_site',lambda _:None)
    plan=submission.submit(s);calls=[]
    def lost(args,**kw):
        calls.append(args)
        return SimpleNamespace(returncode=1,stdout='',stderr='simulated lost acknowledgement')
    monkeypatch.setattr(subprocess,'run',lost)
    with pytest.raises(Exception):submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==1 and (Path(s['root'])/'submission/live_claim/intent.json').is_file()
    with pytest.raises(FileExistsError):submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==1


def test_real_archive_replay_both_checkpoint_payloads_required(saved,monkeypatch,capsys):
    # Real retained-result validators/joins/metrics. Old source+inventory checks
    # are fixture stand-ins; discarded attention banks must stay unnecessary.
    for name in ('LOGIT_D066','LOGIT_D033'):
        checkpoint=saved.root/('train_'+name)/'selected.pt'
        checkpoint.write_bytes(b'bound checkpoint payload fixture')
        receipt=load_json(checkpoint.parent/'receipt.json')
        receipt['outputs']=[file_ref(checkpoint,root=saved.root) if r['path'].endswith('/selected.pt') else r
                            for r in receipt['outputs']]
        (checkpoint.parent/'receipt.json').write_text(json.dumps(with_content_hash(receipt)))
        rebind_training(saved,name)
    helper=Path(source.__file__).parents[1]/'luka_attention_weight/reference_archive.py'
    monkeypatch.setattr(sys,'argv',['replay',str(saved.root/'campaign_spec.json'),str(helper)])
    exec(compile(source.ARCHIVE_CHECK,'original-replay','exec'),{})
    line=next(x for x in capsys.readouterr().out.splitlines() if x.startswith('MULTI_TEACHERS='))
    evidence=json.loads(line.split('=',1)[1])
    assert set(evidence['teachers'])=={'LOGIT_D066','LOGIT_D033'}
    assert not (saved.root.parent/'preflight/export_ANCHOR').exists()
    (saved.root/'train_LOGIT_D033/selected.pt').write_bytes(b'wrong weights')
    with pytest.raises(ValueError):exec(compile(source.ARCHIVE_CHECK,'original-replay','exec'),{})


def test_real_installed_cuda_all_five_full30_production_workers(tmp_path,monkeypatch,capsys):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable; not SPORC certification')
    tiny(monkeypatch)
    # Real production2048-anchor sampler, >=187/class; reduced particle lengths
    # and transformer blocks only. No reduced epochs/objectives/reference count.
    s,tr,val,views,_=fixture(tmp_path,rows=2068)
    refs={}
    for i,name in enumerate(('M0HLT','OFFLINE','OFFLINE_EMBED_W5')):
        torch.manual_seed(i+45);native=DelphesParticleTransformer().eval()
        p=predict(native,val,device='cpu',batch_size=128)
        directory=tmp_path/'refs'/name
        write_immutable_json(directory/'report.json',dict(validation=evaluate_probabilities(val.labels,p)))
        atomic_publish_bytes(directory/'predictions.npz',deterministic_npz_bytes(dict(
            identities=val.identities,labels=val.labels,probabilities=p)))
        refs[name]=dict(report=file_ref(directory/'report.json'),predictions=file_ref(directory/'predictions.npz'))
    s['anchor']['evidence']['models']={n:refs[n] for n in ('M0HLT','OFFLINE')}
    s['reference']['files']=refs['OFFLINE_EMBED_W5'];s=with_content_hash(s)
    (Path(s['root'])/'campaign_spec.json').write_text(json.dumps(s))
    jobs=dict(zip(['export_targets',*c.ARMS],map(str,range(101,107))));active=['export_targets']
    monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(campaign,'validate_spec',lambda _:None)
    monkeypatch.setattr(worker,'metadata',lambda _: (None,None,None,None))
    monkeypatch.setattr(banks,'data',lambda s,role,co,meta:views[co] if co!='D000' else tr if role=='train' else val)
    monkeypatch.setattr(worker.alljets_data,'comparison_indices',lambda *a:np.arange(len(val),dtype=np.int64))
    monkeypatch.setattr(worker,'execution',lambda *a:jobs[active[0]])
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:jobs)
    monkeypatch.setattr(submission,'successful_export',lambda *a:None)
    # Synthetic Linux-host counters, genuine local CUDA/model execution.
    monkeypatch.setattr(worker,'memory_snapshot',lambda _:dict(peak_rss_bytes=2**30,rss_bytes=2**30,
        gpu_peak_bytes=max(1,int(torch.cuda.max_memory_allocated()))))
    receipt=worker.run(s,'export_targets')
    assert receipt['preflight_performed'] is False
    initial=worker.results(s)
    assert len(initial['rows'])==3 and initial['pending']==list(c.ARMS)
    for arm in c.ARMS:
        active[0]=arm;record=worker.run(s,arm)
        assert record['dependencies']==dict(export_targets=receipt['content_hash'])
        report=load_json(Path(s['root'])/arm/'training_report.json')
        assert report['passes']==30 and report['validation_history'][-1]['update']==510
        if arm=='multi_w5_rel':
            assert report['validation_history'][0]['relation_beta']==.5
            assert report['validation_history'][-1]['relation_beta']==1.
            assert all(r['relation_loss']>0 for r in report['validation_history'])
        torch.cuda.empty_cache()
    # All reports remain readable without models executing or particle loads.
    monkeypatch.setattr(torch,'load',lambda *a,**kw:pytest.fail('Reporting must not load models'))
    monkeypatch.setattr(banks,'data',lambda *a,**kw:pytest.fail('Reporting must not read particles'))
    final=worker.results(s);assert len(final['rows'])==8 and not final['pending']
    assert not final['final_test_accessed'] and final['adaptive_validation']
    worker.print_results(final);assert 'r30:' in capsys.readouterr().out
    before={p:p.read_bytes() for p in Path(s['root']).rglob('*') if p.is_file()}
    assert worker.results(s)==final
    assert before=={p:p.read_bytes() for p in before}
    path=Path(s['root'])/'multi_w5_receipt.json'
    path.write_text(json.dumps(with_content_hash(dict(load_json(path),job_id='999'))))
    with pytest.raises(ValueError,match='job'):worker.results(s)


def test_shell_cli_and_required_documentation():
    project=Path(__file__).resolve().parents[1];bash=Path('C:/Program Files/Git/bin/bash.exe')
    if not bash.is_file():bash=shutil.which('bash')
    if not bash:pytest.skip('Bash unavailable')
    for path in ('scripts/queue_luka_multiteacher_trial.sh','sbatch/run_luka_multiteacher_trial.sh'):
        subprocess.run([str(bash),'-n',path],cwd=project,check=True,capture_output=True,timeout=30)
    subprocess.run([sys.executable,'-B','scripts/luka_multiteacher_trial.py','--help'],
        cwd=project,check=True,capture_output=True,timeout=60)
    for path in source.REQUIRED:
        text=(project/path).read_text();assert all(line==line.rstrip() for line in text.splitlines()),path
        if path.endswith('.md'):
            for link in re.findall(r'\]\(([^)#]+)(?:#[^)]*)?\)',text):
                assert (project/path).parent.joinpath(link).is_file(),(path,link)
            for block in re.findall(r'^```bash\n(.*?)^```',text,re.M|re.S):
                subprocess.run([str(bash),'-n'],input=block,check=True,text=True,capture_output=True,timeout=30)

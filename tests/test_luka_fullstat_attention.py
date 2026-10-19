from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch

from test_luka_attention import cache, tiny_config, threads
from test_luka_fullstat import fullstat, durable
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.luka_fullsim.regular_campaign import nodes as parent_nodes
from hlt_classification.luka_attention.model import new_model
from hlt_classification.luka_fullstat_attention import contracts as c, banks, training, campaign, worker


def models():
    return c.nodes(parent_nodes())


def make_bank(root, monkeypatch):
    tiny_config(monkeypatch)
    parents = dict(checkpoint='c'*64,report='d'*64)
    m = banks.export(root,new_model(models()[0]),cache(coordinate='D033'),
        parents=parents,teacher=c.TEACHER,attention=True,device='cpu')
    tr = cache(coordinate='D000')
    return banks.Bank(root,tr,parents=parents,teacher=c.TEACHER,require_attention=True), tr, m


def test_registered_two_fresh_final_rungs_and_unchanged_recipes():
    from hlt_classification.luka_fullstat.recipe import recipe
    from hlt_classification.luka_attention.contracts import recipe as trial_recipe
    a,b = models()
    original = next(n for n in parent_nodes() if n['node_id']=='COARSE_D000')
    assert a['initialization_seed']==b['initialization_seed']==original['initialization_seed']
    assert a['sampler_seed']==b['sampler_seed']==original['sampler_seed']
    assert a['teacher']==b['teacher']=='COARSE_D033'
    assert a['attention_coefficient']==0 and b['attention_coefficient']==.1
    assert a['coordinate']==b['coordinate']=='D000'
    assert a['context_coordinate'] is None and b['context_coordinate'] is None
    assert c.recipe()['base']==recipe(128) and c.recipe()['epochs']==20
    assert c.recipe()['final_test_sealed'] and not c.recipe()['final_test_accessed']
    assert 'historical_test_repurposed_for_training' not in c.recipe()
    assert trial_recipe()['epochs']==30 and len(c.tasks())==4
    assert c.COUNTS==dict(train=3000000,validation=500000,final_test=1500000,unused=105325)


def test_bank_only_selected_D033_TRAIN_and_exact_population(tmp_path,monkeypatch):
    bank,tr,m = make_bank(tmp_path/'bank',monkeypatch)
    assert m['contract']=='LUKA_FULLSTAT_LAST_ATTENTION_BANK/v1'
    assert m['alignment']==banks.alignment(tr)
    with pytest.raises(PermissionError):
        banks.export(tmp_path/'bad',new_model(models()[0]),cache('validation','D033'),
                     parents=m['parents'],teacher=c.TEACHER,attention=True,device='cpu')
    with pytest.raises(PermissionError):
        banks.export(tmp_path/'bad',new_model(models()[0]),cache(coordinate='U100'),
                     parents=m['parents'],teacher=c.TEACHER,attention=True,device='cpu')
    path=tmp_path/'bank/manifest.json'
    for key,value in [('coordinate','U100'),('teacher','COARSE_U100'),('attention',False),
                      ('surface',dict(m['surface'],block_index=6)),('shards',[])]:
        path.write_text(json.dumps(with_content_hash(dict(m,**{key:value}))))
        with pytest.raises(ValueError):
            banks.Bank(tmp_path/'bank',tr,parents=m['parents'],teacher=c.TEACHER,require_attention=True)
    path.write_text(json.dumps(m))
    reordered=cache(coordinate='D000'); reordered.identities=reordered.identities[::-1].copy()
    with pytest.raises(ValueError):
        banks.Bank(tmp_path/'bank',reordered,parents=m['parents'],teacher=c.TEACHER,require_attention=True)
    with (tmp_path/'bank'/m['shards'][0]['payload']['path']).open('ab') as f:
        f.write(b'corrupt')
    with pytest.raises(ValueError):
        banks.Bank(tmp_path/'bank',tr,parents=m['parents'],teacher=c.TEACHER,require_attention=True)


@pytest.mark.parametrize('arm',[0,1])
def test_scientific_twenty_epochs_and_report_guard(tmp_path,monkeypatch,arm):
    bank,tr,_ = make_bank(tmp_path/'bank',monkeypatch)
    val,node=cache('validation','D000'),models()[arm]
    report,_=training.train(new_model(node),tr,val,bank=bank,node=node,device='cpu')
    worker.report_valid(report,node,bank.manifest['content_hash'],'a'*64,len(val))
    assert report['passes']==20 and report['scientific_fit'] is True
    assert report['validation_history'][-1]['learning_rate']==pytest.approx(1.5e-5)
    assert report['validation_history'][-1]['update']==20
    if arm:
        assert report['validation_history'][0]['losses']['weighted_attention'] > 0
    else:
        assert all(r['losses']['weighted_attention']==0 for r in report['validation_history'])
    for key,value in [('scientific_fit',False),('passes',30),('selected_pass',0),('final_test_accessed',True)]:
        with pytest.raises((ValueError,PermissionError)):
            worker.report_valid(with_content_hash(dict(report,**{key:value})),node,bank.manifest['content_hash'],'a'*64,len(val))


def test_zero_weight_matches_original_fullstat20_kernel(tmp_path,monkeypatch):
    from hlt_classification.luka_fullstat_attention.preflight import zero_weight_check
    bank,tr,_=make_bank(tmp_path/'bank',monkeypatch)
    assert zero_weight_check(models()[0],tr,cache('validation','D000'),bank,'cpu')


def gate_fixture(tmp_path):
    return c.artifact('GATE',root=str(tmp_path),bank_root=str(tmp_path/'bank_store'),
        project_dir='/project',source_commit='a'*40,available_bank_bytes=20*2**30,
        available_result_bytes=8*2**30,**campaign.registration(dict(nodes=parent_nodes())))


def test_exact_oscar_gate_and_four_job_science(tmp_path,monkeypatch):
    from hlt_classification.luka_fullstat_attention import submission as s
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    gate=gate_fixture(tmp_path)
    monkeypatch.setattr(campaign,'check_gate',lambda g:None)
    monkeypatch.setattr(s,'check_site',lambda p:None)
    calls=[]
    monkeypatch.setattr(exact.subprocess,'run',lambda cmd,**kw:(calls.append(cmd) or SimpleNamespace(stdout='12345\n')))
    dry=s.submit(gate)
    assert not calls and dry['jobs']==1
    command=dry['commands'][0]['command']
    assert '--partition=gpu' in command and '--gres=gpu:l40s:1' in command and '--mem=180000M' in command
    for digest,auth in [('bad',campaign.AUTHORIZATION),(dry['content_hash'],'bad')]:
        with pytest.raises(PermissionError): s.submit(gate,execute=True,plan_hash=digest,authorization=auth)
    for _ in range(2): s.submit(gate,execute=True,plan_hash=dry['content_hash'],authorization=campaign.AUTHORIZATION)
    assert len(calls)==1 and s.accepted_jobs(gate)==dict(preflight='12345')
    spec=c.artifact('SPEC',root=str(tmp_path/'science'),project_dir='/project',train_minutes=600,export_minutes=60)
    plan=s.render(spec,gate)
    rows={r['task_id']:r for r in plan['commands']}
    assert plan['jobs']==4
    assert rows['train_LOGIT_D000']['dependencies']==rows['train_ATTN_D000']['dependencies']==['export_D033']
    assert '--partition=batch' in rows['aggregate']['command']
    assert not any(a.startswith('--gres=') or a.startswith('--qos=') for a in rows['aggregate']['command'])
    with pytest.raises(ValueError): s.render(dict(spec,train_minutes=2881),gate)


def test_ambiguous_submission_refuses_retry(tmp_path,monkeypatch):
    from hlt_classification.luka_fullstat_attention import submission as s
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    gate=gate_fixture(tmp_path)
    monkeypatch.setattr(campaign,'check_gate',lambda g:None)
    monkeypatch.setattr(s,'check_site',lambda p:None)
    dry=s.submit(gate); calls=[]
    def fail(*args,**kwargs):
        calls.append(args)
        raise RuntimeError('Unknown sbatch outcome')
    monkeypatch.setattr(exact.subprocess,'run',fail)
    for error in (RuntimeError,FileExistsError):
        with pytest.raises(error): s.submit(gate,execute=True,plan_hash=dry['content_hash'],authorization=campaign.AUTHORIZATION)
    assert len(calls)==1


def test_budget_storage_and_scope_fail_closed(tmp_path):
    from hlt_classification.luka_fullstat_attention.preflight import budgets
    m=[dict(counts=dict(train=1000,validation=100),train_seconds=1.,validation_seconds=1.,readback_seconds=1.)]*2
    assert budgets(m,dict(train=1000,validation=100),1.,1.,1.)==(30,15)
    with pytest.raises(ValueError): budgets([dict(m[0],train_seconds=float('nan'))]*2,m[0]['counts'],1.,1.,1.)
    with pytest.raises(ValueError): budgets([dict(m[0],train_seconds=10000)]*2,m[0]['counts'],1.,1.,1.)
    g=gate_fixture(tmp_path)
    (tmp_path/'bank_store').mkdir()
    value=campaign.storage(g,3000000*100,3000000)
    assert value['attention_bytes']==4*3000000*100
    with pytest.raises(OSError): campaign.storage(dict(g,available_bank_bytes=1),3000000*100,3000000)
    with pytest.raises(PermissionError): campaign.separate(tmp_path/'parent/new',tmp_path/'bank',[tmp_path/'parent'])
    with pytest.raises(PermissionError): worker.data({},'final_test','D000')
    with pytest.raises(PermissionError): worker.data({},'train','U100')


def local_fixture(tmp_path,monkeypatch,device):
    tiny_config(monkeypatch)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    from hlt_classification.jetclass2_delphes.runner import predict
    from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
    from hlt_classification.data.cache_contracts import atomic_publish_bytes,deterministic_npz_bytes
    parent=tmp_path/'parent'; parent.mkdir()
    model=new_model(models()[0]).to(device)
    val=cache('validation','D000')
    probabilities=predict(model,val,device=device,batch_size=128)
    metrics=evaluate_probabilities(val.labels,probabilities)
    save_state(parent/'selected.pt',model.state_dict())
    write_immutable_json(parent/'report.json',dict(validation=metrics,selected_pass=17))
    write_immutable_json(parent/'receipt.json',dict(complete=True))
    atomic_publish_bytes(parent/'predictions.npz',deterministic_npz_bytes(dict(
        probabilities=probabilities,labels=val.labels,identities=val.identities)))
    refs=dict(checkpoint=file_ref(parent/'selected.pt'),report=file_ref(parent/'report.json'),
              receipt=file_ref(parent/'receipt.json'),predictions=file_ref(parent/'predictions.npz'))
    gate=gate_fixture(tmp_path)
    gate.update(counts=dict(train=22,validation=22,final_test=1500000,unused=105325),
        anchor=dict(evidence=dict(environment={},prepared='a'*64,cudnn_v8_disabled='unset',
                                 models={n:refs for n in c.REFERENCES})))
    (tmp_path/'bank_store').mkdir()
    meta=dict(content_hash='a'*64,summaries={r:{co:dict(maximum=4,resident_bytes=cache(r,co).nbytes)
        for co in ('D033','D000')} for r in ('train','validation')})
    monkeypatch.setattr(worker,'metadata',lambda g:(meta,None,None))
    monkeypatch.setattr(worker,'data',lambda g,r,co,**kw:cache(r,co))
    monkeypatch.setattr(worker,'execution',lambda *a,**kw:'123')
    monkeypatch.setattr(worker,'memory_snapshot',lambda t:dict(peak_rss_bytes=2**30,rss_bytes=2**30,gpu_peak_bytes=1))
    monkeypatch.setattr(campaign,'check_gate',lambda g:None)
    audit=dict(train_alignment=banks.alignment(cache(coordinate='D000')),
               validation_alignment=banks.alignment(val),sum_squared_counts=int((banks.counts(val)**2).sum()),
               maxima=dict(train=4,validation=4),prepared='a'*64)
    return gate,meta,audit


@pytest.mark.skipif(not torch.cuda.is_available(),reason='local CUDA optional; real Oscar gate remains mandatory')
def test_complete_four_task_production_flow(tmp_path,monkeypatch):
    gate,meta,audit=local_fixture(tmp_path,monkeypatch,'cuda')
    root=tmp_path/'science'; root.mkdir()
    spec=c.artifact('SPEC',root=str(root),project_dir='/project',source_commit='a'*40)
    monkeypatch.setattr(campaign,'check_spec',lambda s:(gate,dict(audit=audit)))
    for task in gate['tasks']:
        result=worker.run(spec,task['task_id'])
        assert worker.completed(spec,gate,task['task_id'])==result
    result=load_json(root/'aggregate/results.json')
    assert result['primary_contrast']=='ATTN_D000 minus LOGIT_D000' and len(result['rows'])==7
    assert result['final_test_sealed'] and not result['final_test_accessed']
    for n in models():
        report=load_json(root/('train_'+n['node_id'])/'training_report.json')
        assert report['passes']==20 and report['scientific_fit']
        assert (root/('train_'+n['node_id'])/'selected.pt').is_file()
    assert worker.run(spec,'aggregate')==worker.completed(spec,gate,'aggregate')
    with (root/'train_ATTN_D000/selected.pt').open('ab') as f: f.write(b'bad')
    with pytest.raises(ValueError): worker.completed(spec,gate,'aggregate')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='local CUDA optional; no Oscar resource claim')
def test_production_preflight_and_admission_failure_injection(tmp_path,monkeypatch):
    from hlt_classification.luka_fullstat_attention import preflight as p
    gate,meta,audit=local_fixture(tmp_path,monkeypatch,'cuda')
    monkeypatch.setattr(p,'audit_payloads',lambda g,m:audit)
    monkeypatch.setattr(p,'installed_environment',lambda:{})
    monkeypatch.setattr(p,'gpu_identity',lambda:dict(name='NVIDIA L40S',total_memory_bytes=48*2**30))
    monkeypatch.setattr(p,'memory_snapshot',lambda t:dict(peak_rss_bytes=2**30,rss_bytes=2**30,
                                                        gpu_peak_bytes=t.cuda.max_memory_allocated()))
    report=p.run(gate)
    p.validate_report(gate,report)
    assert report['zero_weight_kernel_parity'] and len(report['stress'])==3
    assert report['full_population_training_measured'] is False
    for key,value in [('technical_checks_passed',False),('scientific_fit',True),('zero_weight_kernel_parity',False),
                      ('stress',report['stress'][:-1]),('train_minutes',1),('final_test_accessed',True),
                      ('reserved_host_peak_bytes',gate['memory_mb']*1024**2),
                      ('storage',dict(report['storage'],attention_bytes=1))]:
        with pytest.raises((ValueError,PermissionError)):
            p.validate_report(gate,with_content_hash(dict(report,**{key:value})))
    with pytest.raises(FileExistsError): p.run(gate)


def test_science_requires_exact_successful_gate(tmp_path,monkeypatch):
    from hlt_classification.luka_fullstat_attention import preflight,submission
    gate=gate_fixture(tmp_path); (tmp_path/'preflight').mkdir()
    write_immutable_json(tmp_path/'preflight/receipt.json',dict(job_id='123'))
    monkeypatch.setattr(campaign,'check_gate',lambda g:None)
    monkeypatch.setattr(preflight,'validate_report',lambda *a:None)
    monkeypatch.setattr(submission,'accepted_jobs',lambda g:dict(preflight='124'))
    with pytest.raises(ValueError): campaign.science(gate)
    monkeypatch.setattr(submission,'accepted_jobs',lambda g:dict(preflight='123'))
    def fail(job): raise PermissionError('FAILED gate')
    monkeypatch.setattr(campaign,'preflight_accounting',fail)
    with pytest.raises(PermissionError): campaign.science(gate)
    assert not (tmp_path/'science').exists()


def test_source_parent_type_and_frozen_code(tmp_path,monkeypatch):
    from hlt_classification.luka_fullstat_attention import source
    old,new=tmp_path/'old',tmp_path/'new'
    name='src/hlt_classification/example.py'
    for root in (old,new):
        f=root/name; f.parent.mkdir(parents=True); f.write_bytes(b'original')
    monkeypatch.setattr(source,'validate_source_snapshot',lambda *a,**kw:None)
    monkeypatch.setattr(source,'_files',lambda p:{name})
    parent=dict(project_dir=str(old),source_snapshot={})
    assert name in source.compatible(parent,new)
    (new/name).write_bytes(b'modified')
    with pytest.raises(ValueError,match='Frozen'): source.compatible(parent,new)
    write_immutable_json(tmp_path/'trial.json',dict(contract='LUKA_FULLSIM_REGULAR_SPEC/v1'))
    with pytest.raises(ValueError): source.authenticate(tmp_path/'trial.json',new)


def test_real_prepared_payload_audit_without_sealed_or_root_reads(durable,monkeypatch):
    from hlt_classification.luka_fullstat_attention import preflight as p
    import uproot
    output,meta=durable
    g=dict(counts=meta[1]['counts'],anchor=dict(evidence=dict(cache_root=str(output))))
    tr=worker.cache.load(output,meta,role='train',coordinate='D000',max_ram_bytes=2**30)
    val=worker.cache.load(output,meta,role='validation',coordinate='D000',max_ram_bytes=2**30)
    monkeypatch.setattr(worker,'validation_arrays',lambda gate:dict(identities=val.identities,labels=val.labels))
    def forbidden(*a,**kw):
        raise AssertionError('No ROOT/test/unused decoding is needed for the attention audit')
    monkeypatch.setattr(uproot,'open',forbidden)
    result=p.audit_payloads(g,meta)
    assert result['train_alignment']==banks.alignment(tr)
    assert result['validation_alignment']==banks.alignment(val)
    assert result['sum_squared_counts']==sum(int(n)**2 for n in banks.counts(tr))
    assert result['maxima']==dict(train=2,validation=2)
    bad=deepcopy(meta[0]); bad['chunks'][0]['role']='final_test'
    with pytest.raises(PermissionError): p.audit_payloads(g,(bad,meta[1],meta[2]))
    original=worker.cache.read_block
    def reorder(*args,**kw):
        block=original(*args,**kw)
        if args[2]=='D033': block.identities=block.identities[::-1].copy()
        return block
    monkeypatch.setattr(worker.cache,'read_block',reorder)
    with pytest.raises(ValueError,match='ordered-slot alignment'): p.audit_payloads(g,meta)

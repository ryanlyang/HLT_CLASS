"""Real model/ROOT paths with tiny populations; not remote acceptance evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from test_luka_attention import cache, parent_nodes
from test_luka_embedding_aux import tiny, bank_fixture, threads
from hlt_classification.data.cache_contracts import (load_json, with_content_hash,
    write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import train_kernel
from hlt_classification.luka_embedding_aux import worker as old_worker
from hlt_classification.luka_embedding_ladder import (
    contracts as c, source, campaign, submission, data, banks, training, worker, preflight)


def gate_fixture(root, rows=22):
    anchor = c.artifact('ANCHOR', spec=dict(path='/parent/campaign_spec.json'), evidence=dict(
        prepared_root='/prepared', raw_container='/raw', prepared='a'*64, cache='b'*64,
        environment=dict(fixture=True), models={}))
    imported = c.artifact('EMBEDDING_IMPORT', spec=dict(path='/embedding/science/campaign_spec.json'))
    snapshot = dict(content_hash='c'*64)
    return c.artifact('GATE', parents=dict(source=snapshot['content_hash'],anchor=anchor['content_hash'],
        embeddings=imported['content_hash']), root=str(root),project_dir='/project',source_commit='f'*40,
        source_snapshot=snapshot,anchor=anchor,embedding_import=imported,available_quota_gib=2,
        support={r:{co:4 for co in c.VIEWS} for r in ('train','validation')},
        **{**campaign.registration(dict(nodes=parent_nodes())),
           'counts':dict(train=rows,validation=rows),'original50k_rows':rows})


def bank(tr=None, *, technical=True):
    tr = cache(coordinate='U000') if tr is None else tr
    return banks.export(DelphesParticleTransformer(),tr,teacher='U000',checkpoint='d'*64,
                        report='e'*64,device='cpu',technical=technical)


def test_fixed_offline_and_immediate_logit_registration():
    r = c.recipe()
    assert (r['ce'],r['kd'],r['temperature'],r['embedding_weight']) == (.25,.75,2.,.3)
    assert r['embedding_teacher']=='OFFLINE' and r['starting_teacher']=='U000'
    assert not r['attention_kd'] and not r['warm_start']
    assert r['schedule']['maximum_passes']==30 and r['schedule']['batch_size']==128
    gate = gate_fixture('/new')
    for arm in c.ARMS:
        nodes = gate['nodes'][arm]
        assert [n['coordinate'] for n in nodes]==list(c.COORDINATES)
        assert [n['teacher'] for n in nodes]==['U000']+[n['node_id'] for n in nodes[:-1]]
        assert all(n['context_coordinate'] is None for n in nodes)
        assert all(n['embedding_coefficient']==(.3 if arm=='EMBED' else 0.) for n in nodes)
    assert [n['initialization_seed'] for n in gate['nodes']['LOGIT']]==[
        n['initialization_seed'] for n in gate['nodes']['EMBED']]
    assert not gate['automatic_followup'] and gate['execution_site']['partition']=='debug'
    with pytest.raises(ValueError): c.nodes(dict(nodes=parent_nodes()),'OTHER')
    with pytest.raises(PermissionError): c.validate(with_content_hash(dict(gate,final_test_accessed=True)),'GATE')


def test_objective_detaches_targets_and_zero_has_no_head_gradient():
    for weight in (0.,.3):
        logits = torch.randn(4,11,requires_grad=True)
        labels = torch.tensor([0,1,2,3])
        q = torch.full((4,11),1/11,requires_grad=True)
        prediction = torch.randn(4,128,requires_grad=True)
        target = torch.randn(4,128,requires_grad=True)
        loss,ce,kd,aux = training.objective(logits,labels,q,prediction,target,weight)
        torch.testing.assert_close(loss,.25*ce+.75*kd+weight*aux)
        loss.backward()
        assert target.grad is None and q.grad is None
        assert logits.grad is not None
        assert (prediction.grad is None)==(weight==0.)


def test_zero_aux_matches_native_kd_epoch(tmp_path,monkeypatch):
    tiny(monkeypatch)
    tr,emb,_ = bank_fixture(tmp_path,cache(coordinate='D000'))
    tr = cache(coordinate='U050'); val = cache('validation',coordinate='U050')
    incoming = bank()
    node = gate_fixture(tmp_path)['nodes']['LOGIT'][0]
    torch.manual_seed(node['initialization_seed'])
    expected,state = train_kernel(DelphesParticleTransformer(),tr,val,node=node,device='cpu',acceptance_passes=1,
        training_recipe=c.recipe()['schedule'],teacher_probabilities=incoming.probabilities,teacher_identities=incoming.identities)
    actual,produced = training.train(tr,val,node=node,bank=incoming,embeddings=emb,device='cpu',technical=True)
    assert actual['validation']==expected['validation']
    for key in state: torch.testing.assert_close(state[key],produced[key],rtol=0,atol=0)


def test_bank_exact_population_and_chain_binding(tmp_path,monkeypatch):
    tiny(monkeypatch)
    incoming=bank(); tr=cache(coordinate='U050')
    node=gate_fixture(tmp_path)['nodes']['EMBED'][0]
    banks.check(incoming,tr,node,technical=True)
    bad=deepcopy(incoming); bad.probabilities[0,0]+=.001
    with pytest.raises(ValueError): banks.check(bad,tr,node,technical=True)
    with pytest.raises(ValueError,match='preceding'): banks.check(incoming,tr,dict(node,teacher='OFFLINE'))
    with pytest.raises(ValueError,match='contamination'): banks.check(incoming,tr,node,technical=False)
    bad=deepcopy(tr); bad.identities=tr.identities[::-1].copy()
    with pytest.raises(ValueError): banks.check(incoming,bad)
    with pytest.raises(PermissionError): bank(cache('validation',coordinate='U000'))
    _,emb,_=bank_fixture(tmp_path)
    # Whole-jet targets do not demand particle-count or constituent matching.
    data.check_join(emb,tr)
    with pytest.raises(ValueError): data.check_join(emb,bad)
    with pytest.raises(ValueError): data.check_join(emb,cache('validation',coordinate='D000'))


def test_scientific_step_retirement_and_native_final(tmp_path,monkeypatch):
    tiny(monkeypatch)
    _,emb,_=bank_fixture(tmp_path)
    gate=gate_fixture(tmp_path); node=gate['nodes']['EMBED'][0]
    incoming=bank(technical=False)
    step,outgoing=worker.fit(tmp_path/'first',gate,node,cache(coordinate='U050'),cache('validation',coordinate='U050'),
        incoming,emb,device='cpu',technical=False)
    report,a,descriptor=worker.validate_step(tmp_path/'first',gate,node,incoming.manifest,emb.manifest['content_hash'],technical=False)
    assert report['passes']==len(report['validation_history'])==30
    assert step['retired'] and not (tmp_path/'first/selected.pt').exists()
    assert (tmp_path/'first/retirement/receipt.json').is_file()
    assert descriptor==outgoing.manifest and a['probabilities'].shape==(22,11)
    assert (tmp_path/'teacher.pt').exists() # imported shared teacher is untouched
    with pytest.raises((ValueError,FileExistsError)):
        worker.fit(tmp_path/'first',gate,node,cache(coordinate='U050'),cache('validation',coordinate='U050'),
                   incoming,emb,device='cpu',technical=False)


def measurement(gate):
    return dict(peak_rss_bytes=2**30,gpu_peak_bytes=2**30,gpu=dict(total_memory_bytes=40*2**30),
        authentication=10.,setup=20.,fits={n['node_id']:dict(train_epoch=60.,validation_epoch=10.,other=5.)
                                        for a in c.ARMS for n in gate['nodes'][a]})


def test_resource_admission_is_whole_ladder_not_accuracy(tmp_path):
    gate=gate_fixture(tmp_path); m=measurement(gate)
    assert set(preflight.admission(gate,m))==set(c.ARMS)
    assert preflight.admission(gate,{**m,'auc':0.})==preflight.admission(gate,m)
    for fields in (dict(peak_rss_bytes=90*2**30),dict(gpu_peak_bytes=39*2**30),dict(setup=float('nan')),
                   dict(fits={}),dict(fits={n:dict(train_epoch=5000.,validation_epoch=10.,other=5.) for n in m['fits']})):
        with pytest.raises(ValueError): preflight.admission(gate,{**m,**fields})


def test_exact_submission_and_accounting_gate(tmp_path,monkeypatch):
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    gate=gate_fixture(tmp_path); write_immutable_json(tmp_path/'gate_spec.json',gate)
    monkeypatch.setattr(campaign,'check_gate',lambda _:None)
    monkeypatch.setattr(submission,'check_site',lambda _:None)
    calls=[]
    def scheduler(cmd,**kw):
        assert not any(k.startswith(('SBATCH_','SLURM_')) for k in kw['env'])
        calls.append(cmd); return SimpleNamespace(stdout='77;sporc\n')
    monkeypatch.setattr(exact.subprocess,'run',scheduler)
    plan=submission.submit(gate)
    assert not calls and plan['jobs']==1
    with pytest.raises(PermissionError): submission.submit(gate,execute=True,plan_hash=plan['content_hash'],authorization='bad')
    for _ in range(2): submission.submit(gate,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==1 and submission.accepted_jobs(gate)==dict(preflight='77')
    (tmp_path/'preflight').mkdir()
    write_immutable_json(tmp_path/'preflight/receipt.json',dict(job_id='77'))
    monkeypatch.setattr(preflight,'validate_report',lambda *a:None)
    monkeypatch.setattr(campaign,'preflight_accounting',lambda _:(_ for _ in ()).throw(ValueError('FAILED')))
    with pytest.raises(ValueError,match='FAILED'): campaign.science(gate,'EMBED')
    assert not (tmp_path/'science').exists()
    spec=c.artifact('SPEC',root=str(tmp_path/'science/EMBED'),gate_path=str(tmp_path/'gate_spec.json'),
                    project_dir='/project',arm='EMBED',train_minutes=120)
    plan=submission.render(spec)
    assert plan['commands'][0]['task_id']=='ladder' and plan['jobs']==1
    assert '--partition=debug' in plan['commands'][0]['command']
    with pytest.raises(ValueError): submission.render(dict(spec,train_minutes=1441))


def test_source_guard_and_no_attention_dependencies(monkeypatch):
    project=Path(__file__).resolve().parents[1]
    monkeypatch.setattr(source,'_source',lambda *a:dict(content_hash='a'*64))
    def git(cmd,**kw):
        return SimpleNamespace(stdout='\n'.join(source.REQUIRED) if 'ls-files' in cmd else
            'src/hlt_classification/jetclass2_delphes/runner.py\n')
    monkeypatch.setattr(source.subprocess,'run',git)
    with pytest.raises(ValueError): source.current(project,'a'*40)
    monkeypatch.setattr(source.subprocess,'run',lambda cmd,**kw:SimpleNamespace(
        stdout='\n'.join(source.REQUIRED) if 'ls-files' in cmd else '\n'.join(source.ADDITIONS)))
    assert source.current(project,'a'*40)['content_hash']=='a'*64
    for path in source.REQUIRED: assert (project/path).is_file(),path
    assert "('M0HLT','OFFLINE','U000')" in source.PARENT_CHECK
    assert 'COARSE_U100' not in source.PARENT_CHECK and 'attention' not in source.PARENT_CHECK


def test_fresh_root_source_and_registration_guards(tmp_path,monkeypatch):
    parent=dict(nodes=parent_nodes(),project_dir=str(tmp_path/'oldsource'),root=str(tmp_path/'old/science'))
    anchor=c.artifact('ANCHOR',spec=dict(path=str(tmp_path/'old/science/campaign_spec.json')),
        evidence=dict(prepared_root=str(tmp_path/'prepared'),raw_container=str(tmp_path/'raw')))
    imported=c.artifact('EMBEDDING_IMPORT',spec=dict(path=str(tmp_path/'embed/science/campaign_spec.json')))
    snapshot=dict(content_hash='a'*64)
    monkeypatch.setattr(source,'current',lambda *a:snapshot)
    monkeypatch.setattr(source,'authenticate',lambda *a:(anchor,parent))
    monkeypatch.setattr(source,'import_embedding',lambda *a:imported)
    monkeypatch.setattr(campaign,'support',lambda *a:{r:{co:4 for co in c.VIEWS} for r in ('train','validation')})
    args=dict(parent_path='parent',embedding_path='embedding',project=tmp_path/'source',commit='b'*40,available_quota_gib=2)
    for p in ('raw/new','prepared/new','old/new','embed/new','source/new'):
        with pytest.raises(PermissionError): campaign.create(tmp_path/p,**args)
        assert not (tmp_path/p).exists()
    with pytest.raises(ValueError): campaign.create(tmp_path/'low',**{**args,'available_quota_gib':1})
    gate=campaign.create(tmp_path/'fresh',**args)
    campaign.check_gate(gate)
    with pytest.raises(FileExistsError): campaign.create(tmp_path/'fresh',**args)
    monkeypatch.setattr(source,'current',lambda *a:dict(content_hash='c'*64))
    with pytest.raises(ValueError,match='source'): campaign.check_gate(gate)


def test_interrupted_submit_does_not_retry(tmp_path,monkeypatch):
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    gate=gate_fixture(tmp_path)
    monkeypatch.setattr(campaign,'check_gate',lambda _:None)
    monkeypatch.setattr(submission,'check_site',lambda _:None)
    plan=submission.submit(gate); calls=[]
    def uncertain(*a,**kw):
        calls.append(a); raise RuntimeError('possible sbatch followed by disconnect')
    monkeypatch.setattr(exact.subprocess,'run',uncertain)
    for error in (RuntimeError,(FileExistsError,PermissionError,ValueError)):
        with pytest.raises(error):
            submission.submit(gate,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==1


@pytest.mark.parametrize('mismatch',['teacher','population','prepared'])
def test_import_requires_same_pure_offline_teacher_and_population(tmp_path,monkeypatch,mismatch):
    from hlt_classification.luka_embedding_aux import contracts as old_contracts
    root=tmp_path/'embedding'; (root/'science').mkdir(parents=True)
    path=root/'science/campaign_spec.json'
    spec=old_contracts.artifact('SPEC',root=str(path.parent),gate_path=str(root/'gate_spec.json'),
                                project_dir='/old/source',source_commit='b'*40)
    write_immutable_json(path,spec)
    original=c.artifact('ANCHOR',spec=dict(path='/parent/spec',sha256='a'*64),evidence=dict(prepared='c'*64,
        cache='d'*64,prepared_root='/prepared',raw_container='/raw',environment={},
        models={n:dict(checkpoint=dict(sha256=n)) for n in ('M0HLT','OFFLINE','U000')}))
    old_anchor=deepcopy(original); old_anchor['evidence']['models'].pop('U000')
    gate=dict(source_snapshot={},source_commit=spec['source_commit'],anchor=old_anchor,counts=c.COUNTS)
    write_immutable_json(root/'gate_spec.json',gate)
    monkeypatch.setattr(source,'validate_source_snapshot',lambda *a,**kw:None)
    imported=dict(gate=gate,bank={},payload={},predictions={},result={},preflight='e'*64)
    def replay(*a): return file_ref(path),deepcopy(imported)
    monkeypatch.setattr(source,'replay',replay)
    assert source.import_embedding(path,original)['parents']['spec']==spec['content_hash']
    if mismatch=='teacher': gate['anchor']['evidence']['models']['OFFLINE']['checkpoint']['sha256']='wrong'
    elif mismatch=='prepared': gate['anchor']['evidence']['prepared']='wrong'
    else: gate['counts']=dict(train=100000,validation=50000)
    with pytest.raises(ValueError,match='teacher/population'): source.import_embedding(path,original)


def test_full_preflight_both_science_arms_real_kernels(tmp_path,monkeypatch):
    if not torch.cuda.is_available(): pytest.skip('Local CUDA required; remote preflight still mandatory')
    # Production architecture, reduced population only. Native installed parity
    # constructs its own model, so do not monkeypatch the architecture here.
    root=tmp_path/'gate'; root.mkdir(); gate=gate_fixture(root)
    train={co:cache(coordinate=co) for co in c.VIEWS}
    val={co:cache('validation',coordinate=co) for co in c.VIEWS}
    _,emb,teacher_ref=bank_fixture(tmp_path,train['D000'])
    write_immutable_json(tmp_path/'teacher_report.json',dict(content_hash='e'*64))
    predictions=dict(identities=val['D000'].identities,labels=val['D000'].labels,
                     probabilities=np.full((22,11),1/11,np.float32))
    atomic_publish_bytes(tmp_path/'references.npz',deterministic_npz_bytes(predictions))
    gate['anchor']['evidence']['models']={n:dict(checkpoint=teacher_ref,report=file_ref(tmp_path/'teacher_report.json'),
        predictions=file_ref(tmp_path/'references.npz')) for n in ('M0HLT','OFFLINE','U000')}
    gate['embedding_import'].update(bank=file_ref(tmp_path/'bank/manifest.json'),payload=file_ref(tmp_path/'bank/targets.npz'),
                                   predictions=file_ref(tmp_path/'references.npz'))
    for k in ('anchor','embedding_import'): gate[k]=with_content_hash(gate[k])
    gate['parents'].update(anchor=gate['anchor']['content_hash'],embeddings=gate['embedding_import']['content_hash'])
    gate=with_content_hash(gate); write_immutable_json(root/'gate_spec.json',gate)
    meta=({},dict(summaries={r:{co:dict(maximum=4) for co in c.VIEWS} for r in ('train','validation')}),{}, {})
    monkeypatch.setattr(campaign,'check_gate',lambda _:None)
    monkeypatch.setattr(worker,'execution',lambda *a:'77')
    monkeypatch.setattr(data,'load_views',lambda g,r,**kw:train if r=='train' else val)
    monkeypatch.setattr(preflight,'installed_environment',lambda:gate['anchor']['evidence']['environment'])
    monkeypatch.setattr(source,'current',lambda *a:gate['source_snapshot'])
    monkeypatch.setattr(preflight.regular_cache,'metadata',lambda _:meta)
    monkeypatch.setattr(old_worker.alljets_data,'comparison_indices',lambda *a:np.arange(22,dtype=np.int64))
    def memory(_):
        return dict(rss_bytes=2**30,peak_rss_bytes=2**30,gpu_peak_bytes=int(torch.cuda.max_memory_allocated()))
    monkeypatch.setattr(preflight,'memory_snapshot',memory)
    monkeypatch.setattr(worker,'memory_snapshot',memory)
    accepted=preflight.run(gate)
    assert accepted['technical_checks_passed'] and len(accepted['fits'])==10 and len(accepted['stress'])==20
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:dict(preflight='77'))
    monkeypatch.setattr(campaign,'preflight_accounting',lambda _:dict(state='COMPLETED',exit_code='0:0'))
    specs=[]
    for arm in c.ARMS:
        spec=campaign.science(gate,arm); specs.append(spec)
        result=worker.run(spec)
        assert len(result['rows'])==8 and not result['independent_test_available']
        assert worker.results(spec)==result
        for node in gate['nodes'][arm]:
            folder=Path(spec['root'])/'ladder'/node['node_id']
            report=load_json(folder/'training_report.json')
            assert len(report['validation_history'])==30 and report['parents']['embeddings']==emb.manifest['content_hash']
            assert (folder/'selected.pt').exists()==(node['coordinate']=='D000')
        state=torch.load(Path(spec['root'])/'ladder'/(arm+'_D000')/'selected.pt',weights_only=True)
        assert not any(k.startswith('predictor.') for k in state)
        with pytest.raises(FileExistsError): worker.run(spec)
    # Results authenticate compact artifacts, never reconstruct or load a model.
    monkeypatch.setattr(torch,'load',lambda *a,**kw:pytest.fail('No model loading for results'))
    for spec in specs: worker.results(spec)
    bad=deepcopy(accepted); bad['stress'][0]['maximum']+=1
    with pytest.raises(ValueError,match='support'): preflight.validate_report(gate,with_content_hash(bad))
    bad=deepcopy(accepted); bad['outputs']=[x for x in bad['outputs'] if x['path']!='references/SINGLE_EMBED.npz']
    with pytest.raises(ValueError,match='coverage'): preflight.validate_report(gate,with_content_hash(bad))
    p=Path(specs[0]['root'])/'ladder/LOGIT_D000/validation_predictions.npz'
    p.write_bytes(p.read_bytes()+b'bad')
    with pytest.raises(ValueError): worker.results(specs[0])
    assert (tmp_path/'teacher.pt').exists() and (tmp_path/'bank/targets.npz').exists()


from test_luka_fullsim_stage2 import dataset, prepared
from test_luka_fullsim_alljets import all_prepared


def test_real_root_rebuild_deleted_U_only_exact(all_prepared,monkeypatch):
    from hlt_classification.luka_fullsim import regular_cache
    old,manifest,p,inventory,splits=all_prepared
    root=Path(old['root'])/'prepared'; meta=(manifest,p,inventory,splits)
    gate=gate_fixture(root.parent/'new'); gate['counts']=p['counts']
    gate['anchor']['evidence'].update(prepared_root=str(root),raw_container=old['input_container'],
                                    prepared=p['content_hash'],cache=manifest['content_hash'])
    expected=data.load_views(gate,'train',authenticated=meta)
    for co in ('U000','U050'):
        for ref in manifest['blocks']['train'][co]: (root/ref['path']).unlink()
    actual=data.load_views(gate,'train',authenticated=meta)
    for co in c.VIEWS:
        for k,v in expected[co].batch(np.arange(len(expected[co]))).items():
            np.testing.assert_array_equal(v,actual[co].batch(np.arange(len(actual[co])))[k])
    for co in ('U000','U050'):
        assert not any((root/ref['path']).exists() for ref in manifest['blocks']['train'][co])
    with pytest.raises(PermissionError): data.load_views(gate,'final_test',authenticated=meta)
    bad=deepcopy(meta); bad[1]['summaries']['train']['U000']['maximum']+=1
    with pytest.raises(ValueError,match='fingerprint'): data.load_views(gate,'train',authenticated=bad)
    (root/manifest['blocks']['train']['D000'][0]['path']).unlink()
    with pytest.raises(ValueError,match='nonregistered'): data.load_views(gate,'train',authenticated=meta)

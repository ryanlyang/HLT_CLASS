"""Native, not predicted, predecessor targets; exact fresh four-stage execution."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import numpy as np
import pytest
import torch
from test_luka_u100_pretrain import cache
from test_luka_attention import parent_nodes
from test_luka_embedding_aux import tiny
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays, with_content_hash,
    write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.luka_embedding_aux.model import new_model, head_input
from hlt_classification.luka_embedding_ladder import contracts as anchor_contracts
from hlt_classification.luka_embedding_prediction import contracts as c, campaign, source, submission, data, targets, training, worker


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def node(co='U100'): return c.nodes(dict(nodes=parent_nodes()))[c.COORDINATES.index(co)]


def spec_fixture(root):
    root.mkdir(exist_ok=True)
    anchor=anchor_contracts.artifact('ANCHOR',spec=dict(path='/parent/spec.json'), evidence=dict(
        environment={'fixture':True}, prepared='a'*64, cache='b'*64, prepared_root='/prepared', raw_container='/raw',
        models={n:{} for n in worker.REFS}))
    snapshot=dict(content_hash='c'*64)
    spec=c.artifact('SPEC',parents=dict(source=snapshot['content_hash'],anchor=anchor['content_hash']),
        root=str(root),project_dir='/project',source_commit='f'*40,source_snapshot=snapshot,anchor=anchor,
        available_quota_gib=2,support={r:{co:4 for co in c.VIEWS} for r in ('train','validation')},
        **dict(campaign.registration(dict(nodes=parent_nodes())),counts=dict(train=22,validation=22),original50k_rows=22))
    write_immutable_json(root/'campaign_spec.json',spec)
    return spec


def teacher_fixture(root,coordinate='U000',technical=False):
    root.mkdir(exist_ok=True)
    native=DelphesParticleTransformer(); save_state(root/'selected.pt',native.state_dict())
    write_immutable_json(root/'report.json',dict(fixture=True))
    files=dict(checkpoint=file_ref(root/'selected.pt'),report=file_ref(root/'report.json'),predictions=file_ref(root/'report.json'))
    record=c.artifact('TEACHER',parents=dict(spec='f'*64,prepared='a'*64,**{k:v['sha256'] for k,v in files.items()}),
        spec='f'*64,prepared='a'*64,coordinate=coordinate,model_id=c.teacher_id(coordinate),
        target_role='train',technical=technical,tap='native_preclassifier',
        origin='original_anchor' if coordinate=='U000' else 'this_campaign',files=files)
    return native,record


def bank_fixture(root,co='U100',technical=False):
    native,record=teacher_fixture(root,c.PREVIOUS[co],technical and co!='U100')
    targets.export(root/'targets',native,cache(coordinate=c.PREVIOUS[co]),teacher_record=record,device='cpu')
    tr=cache(coordinate=co)
    return tr,targets.Bank(root/'targets',tr,teacher_record=record,node=node(co),technical=technical),record


def test_recipe_is_immediate_native_not_fixed_offline_or_predictions():
    r=c.recipe(); assert r['starting_teacher']=='U000' and r['auxiliary_coefficient']==5
    assert r['target']=='immediate_predecessor_selected_native_preclassifier_embedding'
    assert r['target_roles']==['train'] and not r['logit_kd'] and not r['warm_start'] and not r['attention_kd']
    assert r['base']['peak_lr']==3e-4 and training.learning_rate(30)==pytest.approx(1.5e-5)
    original={n['node_id']:n for n in parent_nodes()}
    for co,previous in zip(c.COORDINATES,('U000','U100','D066','D033')):
        n=node(co)
        assert n['embedding_teacher']==c.teacher_id(previous) and n['embedding_teacher_coordinate']==previous
        assert n['teacher'] is n['teacher_coordinate'] is n['context_coordinate'] is None
        for k in ('initialization_seed','sampler_seed'): assert n[k]==original['COARSE_'+co][k]
    from hlt_classification.luka_embedding_ladder import source as original_source
    assert source.authenticate is original_source.authenticate
    assert "('M0HLT','OFFLINE','U000')" in original_source.PARENT_CHECK


def test_native_bank_values_freeze_normalization_and_no_predictor(tmp_path,monkeypatch):
    tiny(monkeypatch); native,record=teacher_fixture(tmp_path); tr=cache(coordinate='U000')
    raw=tr.batch(np.arange(len(tr)))
    native.eval()
    for p in native.parameters(): p.requires_grad_(False)
    with torch.inference_mode():
        _,expected=head_input(native,**{k:torch.from_numpy(raw[k]) for k in ('features','vectors','mask')})
    normalized,mean,scale=targets.standardize(expected.numpy())
    targets.export(tmp_path/'targets',native,tr,teacher_record=record,device='cpu')
    a=load_npz_arrays(tmp_path/'targets/targets.npz')
    np.testing.assert_array_equal(a['targets'],normalized)
    np.testing.assert_array_equal(a['mean'],mean); np.testing.assert_array_equal(a['scale'],scale)
    assert not any(p.requires_grad for p in native.parameters()) and not native.training
    assert a['targets'].shape==(22,128) and a['targets'].dtype==np.float32
    augmented=new_model(node())
    state=augmented.state_dict(); state.update(native.state_dict()); augmented.load_state_dict(state)
    with pytest.raises(ValueError,match='NATIVE'):
        targets.export(tmp_path/'wrong',augmented,tr,teacher_record=record,device='cpu')
    with pytest.raises(PermissionError):
        targets.export(tmp_path/'validation',native,cache('validation','U000'),teacher_record=record,device='cpu')


@pytest.mark.parametrize('co',c.COORDINATES)
def test_bank_exact_teacher_identity_labels_and_technical_boundary(tmp_path,monkeypatch,co):
    tiny(monkeypatch); tr,bank,record=bank_fixture(tmp_path,co)
    assert bank.manifest['coordinate']==c.PREVIOUS[co]
    for attr in ('identities','labels'):
        bad=deepcopy(tr); setattr(bad,attr,getattr(tr,attr)[::-1].copy())
        with pytest.raises(ValueError,match='join'):
            targets.Bank(tmp_path/'targets',bad,teacher_record=record,node=node(co))
    wrong=dict(node(co),embedding_teacher='OLD_PREDECESSOR')
    with pytest.raises(ValueError): targets.Bank(tmp_path/'targets',tr,teacher_record=record,node=wrong)
    if co!='U100':
        with pytest.raises(ValueError): targets.Bank(tmp_path/'targets',tr,teacher_record=record,node=node(co),technical=True)
    (tmp_path/'targets/targets.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError): targets.Bank(tmp_path/'targets',tr,teacher_record=record,node=node(co))


@pytest.mark.parametrize('field,value',[('role','validation'),('tap','predictor_output'),('coordinate','OFFLINE'),('width',127)])
def test_resigned_manifest_cannot_change_science(tmp_path,monkeypatch,field,value):
    tiny(monkeypatch); tr,bank,record=bank_fixture(tmp_path)
    p=tmp_path/'targets/manifest.json'; p.write_text(json.dumps(with_content_hash(dict(bank.manifest,**{field:value}))))
    with pytest.raises(ValueError): targets.Bank(tmp_path/'targets',tr,teacher_record=record,node=node())


def test_objective_is_ce_plus5_and_target_detached():
    logits=torch.randn(4,11,requires_grad=True); pred=torch.randn(4,128,requires_grad=True)
    target=torch.randn(4,128,requires_grad=True)
    loss,ce,aux=training.objective(logits,torch.arange(4),pred,target,coefficient=5.)
    torch.testing.assert_close(loss,ce+5*aux); loss.backward()
    assert target.grad is None and pred.grad is not None and logits.grad is not None
    for bad in (True,.3,10,float('nan')):
        with pytest.raises(ValueError): training.objective(logits,torch.arange(4),pred,target,coefficient=bad)
    with pytest.raises(ValueError): training.objective(logits,torch.arange(4),pred*float('nan'),target,coefficient=5)


def test_fresh_kernel_matches_existing_weight5_kernel(tmp_path,monkeypatch):
    tiny(monkeypatch)
    from hlt_classification.luka_u100_embedding import training as old, contracts as oldc
    tr,bank,_=bank_fixture(tmp_path,'D000',technical=True); val=cache('validation')
    donor_bank=SimpleNamespace(targets=bank.targets,identities=bank.identities,labels=bank.labels,
        manifest=oldc.artifact('BANK',role='train',coordinate='U100'))
    r,state=training.train(new_model(node('D000')),tr,val,bank=bank,node=node('D000'),device='cpu',technical=True)
    previous,expected=old.train(new_model(node('D000')),tr,val,bank=donor_bank,node=node('D000'),device='cpu',technical=True)
    for k,v in state.items(): torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
    for k in ('ce_loss','auxiliary_loss','learning_rate','validation'):
        assert r['validation_history'][0][k]==previous['validation_history'][0][k]


def test_full30_poor_scores_do_not_skip_and_selected_native_is_restored(tmp_path,monkeypatch):
    tiny(monkeypatch); tr,bank,_=bank_fixture(tmp_path); val=cache('validation','U100')
    original=training.evaluate_probabilities; calls=[]
    def poor(labels,p):
        calls.append(1); return dict(original(labels,p),accuracy=0.,macro_ovr_auc=.4 if len(calls)==1 else .1)
    monkeypatch.setattr(training,'evaluate_probabilities',poor)
    model=new_model(node()); initial=model.native_state()
    report,state=training.train(model,tr,val,bank=bank,node=node(),device='cpu')
    spec=dict(counts=dict(train=22,validation=22),anchor=dict(evidence=dict(prepared='a'*64)))
    training.validate_report(report,spec,bank.manifest['content_hash'],node())
    assert report['passes']==30 and report['selected_pass']==1 and len(calls)==30
    assert not any(k.startswith('predictor.') for k in state)
    for k,v in state.items(): torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
    assert any(not torch.equal(v,state[k]) for k,v in initial.items())
    with pytest.raises(ValueError,match='selected'):
        training.validate_report(with_content_hash(dict(report,selected_pass=30)),spec,bank.manifest['content_hash'],node())


def test_exact_five_job_dag_dry_authorized_submit_and_idempotence(tmp_path,monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as exact
    spec=spec_fixture(tmp_path); monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(submission,'check_site',lambda _:None); calls=[]
    def submit(cmd,**kw): calls.append(cmd); return SimpleNamespace(stdout=str(100+len(calls)))
    monkeypatch.setattr(exact.subprocess,'run',submit)
    plan=submission.submit(spec); assert not calls and plan['jobs']==5
    assert [r['dependencies'] for r in plan['commands']]==[[],['setup'],['setup','U100'],['setup','D066'],['setup','D033']]
    for row in plan['commands']:
        assert '--partition=debug' in row['command'] and '--mem=90000M' in row['command'] and '--gres=gpu:a100:1' in row['command']
        assert row['command'][-3:]==['/project',str(tmp_path/'campaign_spec.json'),row['task_id']]
    with pytest.raises(PermissionError): submission.submit(spec,execute=True,plan_hash='wrong',authorization=c.AUTHORIZATION)
    ledger=submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert ledger['jobs']==dict(setup='101',U100='102',D066='103',D033='104',D000='105')
    for command,dep in zip(calls[1:],('101','101:102','101:103','101:104')):
        assert '--dependency=afterok:'+dep in command and '--kill-on-invalid-dep=yes' in command
    assert submission.accepted_jobs(spec)==ledger['jobs']
    assert submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)==ledger
    assert len(calls)==5


def test_ambiguous_submit_not_retried(tmp_path,monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as exact
    spec=spec_fixture(tmp_path); monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(submission,'check_site',lambda _:None); plan=submission.submit(spec); calls=[]
    def lost(*a,**kw): calls.append(1); raise RuntimeError('lost response')
    monkeypatch.setattr(exact.subprocess,'run',lost)
    with pytest.raises(RuntimeError): submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    with pytest.raises(FileExistsError): submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==1


@pytest.mark.parametrize('co',c.COORDINATES)
def test_failed_setup_blocks_every_stage(tmp_path,monkeypatch,co):
    spec=spec_fixture(tmp_path); monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a,**kw:{n:str(i) for i,n in enumerate(c.TASKS)})
    monkeypatch.setattr(worker,'validate_task',lambda *a:dict(job_id='0',content_hash='a'*64))
    def failed(*a): raise ValueError('setup FAILED')
    monkeypatch.setattr(worker,'preflight_accounting',failed)
    monkeypatch.setattr(worker,'fit',lambda *a,**kw:pytest.fail('No fit allowed'))
    with pytest.raises(ValueError,match='setup FAILED'): worker.run(spec,co)
    assert not (tmp_path/co).exists()


def test_resource_admission_requires_every_stage_and_no_quality_gate(tmp_path):
    spec=spec_fixture(tmp_path)
    m=dict(environment=spec['anchor']['evidence']['environment'],gpu=dict(total_memory_bytes=40*2**30),
        authentication=10.,peak_rss_bytes=2**30,gpu_peak_bytes=2**30,
        fits={co:dict(load=10.,train_epoch=20.,validation_epoch=10.,readback_export=10.) for co in c.COORDINATES})
    assert set(worker.admission(spec,m))==set(c.COORDINATES)
    for k,v in (('peak_rss_bytes',90*2**30),('gpu_peak_bytes',40*2**30),('peak_rss_bytes',0),('environment',{})):
        with pytest.raises((ValueError,MemoryError)): worker.admission(spec,dict(m,**{k:v}))
    bad=deepcopy(m); bad['fits']['D000']['train_epoch']=100000
    with pytest.raises(ValueError,match='walltime'): worker.admission(spec,bad)
    del bad['fits']['D000']
    with pytest.raises(ValueError,match='Incomplete'): worker.admission(spec,bad)


def test_create_separate_root_and_original_anchor_contract(tmp_path,monkeypatch):
    parent=dict(nodes=parent_nodes(),project_dir=str(tmp_path/'oldproject'),root=str(tmp_path/'old/science'))
    fixture=spec_fixture(tmp_path/'fixture')
    monkeypatch.setattr(source,'current',lambda *a:fixture['source_snapshot'])
    monkeypatch.setattr(source,'authenticate',lambda *a:(fixture['anchor'],parent))
    monkeypatch.setattr(campaign,'support',lambda _:fixture['support'])
    spec=campaign.create(tmp_path/'new','original',tmp_path/'project','f'*40,2)
    campaign.validate_spec(spec,payloads=False)
    assert spec['anchor']['contract']=='LUKA_EMBEDDING_LADDER_ANCHOR/v1'
    assert len(spec['nodes'])==4 and spec['counts']==c.COUNTS
    with pytest.raises(PermissionError): campaign.create(tmp_path/'old/overlap','original',tmp_path/'project','f'*40,2)
    with pytest.raises(ValueError): campaign.create(tmp_path/'noquota','original',tmp_path/'project','f'*40,1)
    with pytest.raises(FileExistsError): campaign.create(tmp_path/'new','original',tmp_path/'project','f'*40,2)
    for key,value in [('memory_mb',1000),('minutes',{}),('recipe',{})]:
        with pytest.raises(ValueError): campaign.validate_spec(with_content_hash(dict(spec,**{key:value})),payloads=False)


def test_source_guard_requires_complete_addition_and_preserves_donors(tmp_path,monkeypatch):
    monkeypatch.setattr(source,'_source',lambda *a:dict(content_hash='a'*64))
    monkeypatch.setattr(source,'__file__',str(tmp_path/'src/hlt_classification/luka_embedding_prediction/source.py'))
    def git(cmd,**kw): return SimpleNamespace(stdout='\n'.join(sorted(source.REQUIRED)) if 'ls-files' in cmd else 'M\tsrc/shared.py\n')
    monkeypatch.setattr(source.subprocess,'run',git)
    with pytest.raises(ValueError,match='donor source'): source.current(tmp_path,'f'*40)
    monkeypatch.setattr(source.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=''))
    with pytest.raises(ValueError,match='required'): source.current(tmp_path,'f'*40)


def test_reviewed_integration_preserves_all_existing_scientific_source():
    root=Path(__file__).resolve().parents[1]
    rows=subprocess.run(['git','-C',str(root),'diff','--name-status','--no-renames',
                         source.DONOR_BASE,source.BASE,'--','src'],
                        check=True,capture_output=True,text=True).stdout.splitlines()
    assert len(rows)==9
    assert all(row.startswith('A\tsrc/hlt_classification/luka_fullstat_multiteacher/') for row in rows)


from test_luka_fullsim_stage2 import dataset, prepared
from test_luka_fullsim_alljets import all_prepared


def test_real_root_deleted_U000_rebuilt_exactly_in_RAM_only(all_prepared):
    old,manifest,p,inventory,splits=all_prepared; root=Path(old['root'])/'prepared'; meta=(manifest,p,inventory,splits)
    spec=dict(memory_mb=90000,counts=p['counts'],anchor=dict(evidence=dict(prepared_root=str(root),
        raw_container=old['input_container'],prepared=p['content_hash'],cache=manifest['content_hash'])))
    expected=data.load_views(spec,'train',['U000'],authenticated=meta)['U000']
    for ref in manifest['blocks']['train']['U000']: (root/ref['path']).unlink()
    actual=data.load_views(spec,'train',['U000'],authenticated=meta)['U000']
    for k,v in expected.batch(np.arange(len(expected))).items():
        np.testing.assert_array_equal(v,actual.batch(np.arange(len(actual)))[k])
    assert not any((root/r['path']).exists() for r in manifest['blocks']['train']['U000'])
    bad=deepcopy(meta); bad[1]['summaries']['train']['U000']['maximum']+=1
    with pytest.raises(ValueError,match='fingerprint'): data.load_views(spec,'train',['U000'],authenticated=bad)
    with pytest.raises(PermissionError): data.load_views(spec,'final_test',['U000'],authenticated=meta)
    with pytest.raises(ValueError): data.load_views(spec,'train',['OFFLINE'],authenticated=meta)
    (root/manifest['blocks']['train']['D000'][0]['path']).unlink()
    with pytest.raises(ValueError,match='nonregistered'): data.load_views(spec,'train',['D000'],authenticated=meta)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Installed-Weaver CUDA required')
def test_real_cuda_setup_and_all_four30_epoch_production_workers(tmp_path,monkeypatch):
    tiny(monkeypatch)
    from hlt_classification.jetclass2_delphes import acceptance, model as native_model
    monkeypatch.setattr(acceptance,'model_config',native_model.model_config)
    spec=spec_fixture(tmp_path/'campaign'); reg=campaign.registration
    monkeypatch.setattr(campaign,'registration',lambda p:dict(reg(p),counts=dict(train=22,validation=22),original50k_rows=22))
    native,record=teacher_fixture(tmp_path/'teacher')
    native.to('cuda'); val=cache('validation','U000')
    predictions=dict(identities=val.identities,labels=val.labels,probabilities=predict(native,val,device='cuda',batch_size=128))
    atomic_publish_bytes(tmp_path/'references.npz',deterministic_npz_bytes(predictions))
    refs=dict(record['files'],predictions=file_ref(tmp_path/'references.npz'))
    spec['anchor']['evidence']['models']={n:refs for n in worker.REFS}
    spec['anchor']['evidence']['environment']=worker.installed_environment()
    spec['anchor']=with_content_hash(spec['anchor']); spec['parents']['anchor']=spec['anchor']['content_hash']; spec=with_content_hash(spec)
    (Path(spec['root'])/'campaign_spec.json').write_text(json.dumps(spec))
    monkeypatch.setattr(campaign,'check',lambda _:None); monkeypatch.setattr(source,'current',lambda *a:spec['source_snapshot'])
    jobs={n:str(i) for i,n in enumerate(c.TASKS)}
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a,**kw:jobs)
    monkeypatch.setattr(worker,'execution',lambda *a:jobs['setup']); monkeypatch.setattr(worker,'preflight_accounting',lambda *a:None)
    seen_loads=[]
    def load(s,r,coordinates,**kw): seen_loads.append((r,coordinates)); return {co:cache(r,co) for co in coordinates}
    monkeypatch.setattr(data,'load_views',load); monkeypatch.setattr(worker.regular_cache,'metadata',lambda _:({},{},{},{}))
    monkeypatch.setattr(worker.alljets_data,'comparison_indices',lambda *a:np.arange(22,dtype=np.int64))
    monkeypatch.setattr(worker,'memory_snapshot',lambda _:dict(rss_bytes=2**30,peak_rss_bytes=2**30,
        gpu_peak_bytes=int(torch.cuda.max_memory_allocated())))
    worker.run(spec,'setup'); assert len(worker.results(spec)['pending'])==4
    starts=[]; original_model=worker.new_model
    def fresh(n):
        m=original_model(n); starts.append((n['node_id'],m.native_state())); return m
    monkeypatch.setattr(worker,'new_model',fresh)
    for co in c.COORDINATES:
        monkeypatch.setattr(worker,'execution',lambda *a:jobs[co]); worker.run(spec,co)
        report=load_json(Path(spec['root'])/co/'fit/training_report.json')['report']
        assert report['passes']==30 and not report['technical'] and report['initialization']=='fresh_original_coordinate_seed'
        incoming=load_json(worker.incoming_dir(spec,co)/'manifest.json')
        assert incoming['teacher']==worker.teacher_for(spec,c.PREVIOUS[co])
        assert not incoming['teacher']['technical']
        if co!='U100':
            assert incoming['teacher']['files']['checkpoint']==file_ref(Path(spec['root'])/c.PREVIOUS[co]/'fit/selected.pt')
            technical_teacher=worker.teacher_for(spec,c.PREVIOUS[co],True)
            assert technical_teacher!=incoming['teacher']
            with pytest.raises(ValueError):
                targets.Bank(worker.incoming_dir(spec,co,True),cache(coordinate=co),teacher_record=technical_teacher,node=node(co))
        selected=torch.load(Path(spec['root'])/co/'fit/selected.pt',weights_only=True)
        assert not any(k.startswith('predictor.') for k in selected)
    # Every stage freshly initialized, not restored from its teacher or probe.
    assert len(starts)==4
    for name,state in starts:
        expected=original_model(node(name.removeprefix('PRED_'))).native_state()
        for k,v in state.items(): torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
    assert all(r in ('train','validation') and len(cs)==1 for r,cs in seen_loads)
    assert not (Path(spec['root'])/'D000/targets').exists()
    with pytest.raises(FileExistsError): worker.run(spec,'D000')
    monkeypatch.setattr(torch,'load',lambda *a,**kw:pytest.fail('Results may not decode models'))
    result=worker.results(spec); assert len(result['rows'])==7 and not result['pending']
    assert result['rows'][-1]['input']=='single HLT' and result['rows'][3]['input']=='ORACLE'
    worker.print_results(result)
    (Path(spec['root'])/'U100/fit/validation_predictions.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError): worker.results(spec)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Installed-Weaver CUDA required')
def test_full_architecture_installed_native_and_predictor_parity():
    with worker.parity_backend('cuda'): assert worker.installed_parity(cache(),device='cuda')['passed']
    for bf16 in (False,True): assert worker.parity(cache().batch(np.arange(4)),node(),'cuda',bf16=bf16)['zero_control_update']


def test_scripts_required_files_and_bash_syntax():
    root=Path(__file__).resolve().parents[1]; assert all((root/p).is_file() for p in source.REQUIRED)
    script=(root/'sbatch/run_luka_embedding_prediction.sh').read_text()
    for token in ('${PROJECT_DIR}/sbatch/common.sh','PYTHONNOUSERSITE=1','${CONDA_PREFIX}/lib','CUBLAS_WORKSPACE_CONFIG=:4096:8'):
        assert token in script
    queue=(root/'scripts/queue_luka_embedding_prediction.sh').read_text()
    assert not any(s in queue for s in ('scancel','scontrol update','rm -','train_COARSE_U100'))
    assert queue.count('</dev/tty')==3 and 'train_U000/selected.pt' in queue
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    if bash.is_file():
        for p in ('scripts/queue_luka_embedding_prediction.sh','sbatch/run_luka_embedding_prediction.sh'):
            subprocess.run([str(bash),'-n',p],cwd=root,check=True,capture_output=True)

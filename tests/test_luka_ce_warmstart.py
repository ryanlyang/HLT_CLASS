"""Matched initialization, objective isolation, exact DAG and real CUDA worker miniature."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import numpy as np
import pytest
import torch
from test_luka_u100_pretrain import cache
from test_luka_u100_embedding import bank_fixture
from test_luka_attention import parent_nodes
from test_luka_embedding_aux import tiny
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays, with_content_hash,
    write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import predict, _forward
from hlt_classification.luka_embedding_aux.model import new_model
from hlt_classification.luka_embedding_aux import contracts as ac
from hlt_classification.luka_d066_embedding import contracts as rc
from hlt_classification.luka_u100_embedding import contracts as uc
from hlt_classification.luka_ce_warmstart import contracts as c, training, source, campaign, submission, worker, targets


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def node(): return c.node(dict(nodes=parent_nodes()))


def spec_fixture(root):
    root.mkdir(exist_ok=True)
    initial = dict(path='/parent/selected.pt', bytes=10, sha256='e'*64)
    anchor = ac.artifact('ANCHOR', spec=dict(path='/parent/spec.json'), evidence=dict(environment={'fixture': True},
        prepared='a'*64, cache='b'*64, prepared_root='/prepared', raw_container='/raw',
        models={'M0HLT': dict(checkpoint=initial)}))
    teacher = uc.artifact('TEACHER', parents=dict(anchor=anchor['content_hash']), prepared='a'*64,
        coordinate='U100', target_role='train', spec=anchor['spec'],
        node=dict(node_id='COARSE_U100', coordinate='U100', context_coordinate=None),
        files={key: {} for key in ('checkpoint', 'report', 'predictions', 'receipt')})
    ref = rc.artifact('REFERENCE', parents=dict(anchor=anchor['content_hash']),
        spec=dict(path='/reference/spec.json'), node=node(), files={})
    snapshot = dict(content_hash='c'*64)
    spec = c.artifact('SPEC', parents=dict(source=snapshot['content_hash'], anchor=anchor['content_hash'],
        teacher=teacher['content_hash'], reference=ref['content_hash']), root=str(root), project_dir='/project',
        source_commit='f'*40, source_snapshot=snapshot, anchor=anchor, teacher=teacher, reference=ref,
        initial_checkpoint=initial, available_quota_gib=2,
        **dict(campaign.registration(dict(nodes=parent_nodes())), counts=dict(train=22, validation=22), original50k_rows=22))
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def initial_state():
    # Different from normal initialization; also exercise classifier and BN buffers.
    torch.manual_seed(123456)
    m = DelphesParticleTransformer()
    for name, value in m.state_dict().items():
        if name.endswith('running_mean'): value.add_(.25)
    return {k: v.clone() for k, v in m.state_dict().items()}


def test_registered_matched_protocol():
    from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
    r = c.recipe()
    assert r['teacher'] == 'COARSE_U100' and r['epochs'] == 30 and r['target_roles'] == ['train']
    assert r['arms'] == {'ce': dict(ce=1., embedding=0.), 'embed': dict(ce=1., embedding=5.)}
    assert not r['logit_kd'] and not r['attention_kd']
    assert c.TASKS == {'setup': [], 'ce': ['setup'], 'embed': ['setup']}
    for p in np.linspace(.01, 30, 100): assert c.learning_rate(p) == .1*learning_rate(p)
    assert c.learning_rate(3) == pytest.approx(3e-5)
    assert c.learning_rate(30) == pytest.approx(1.5e-6)
    for p in (0,31,float('nan')):
        with pytest.raises(ValueError): c.learning_rate(p)


@pytest.mark.parametrize('arm', c.ARMS)
def test_restores_all_native_weights_buffers_classifier_fresh_head_and_optimizer(monkeypatch, arm):
    tiny(monkeypatch); state = initial_state(); model = new_model(node())
    fresh_head = {k: v.clone() for k, v in model.predictor.state_dict().items()}
    training.initialize(model, state, arm)
    for k, v in state.items(): torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
    for k, v in fresh_head.items(): torch.testing.assert_close(model.predictor.state_dict()[k], v, rtol=0, atol=0)
    assert all(p.requires_grad for p in model.mod.parameters())
    assert all(p.requires_grad == (arm == 'embed') for p in model.predictor.parameters())
    optimizer = training.optimizer_for(model); assert not optimizer.state
    assert all(g['lr'] == 3e-5 for g in optimizer.param_groups)
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in model.parameters() if p.requires_grad}
    with pytest.raises(ValueError, match='all and only'): training.initialize(model, model.state_dict(), arm)
    with pytest.raises(ValueError): training.initialize(model, {k:v for k,v in state.items() if 'fc' not in k}, arm)


def test_loss_isolation_and_detached_target():
    logits=torch.randn(4,11,requires_grad=True); prediction=torch.randn(4,128,requires_grad=True)
    target=torch.randn(4,128,requires_grad=True); labels=torch.arange(4)
    loss,ce,aux=training.objective(logits,labels,prediction,target,arm='embed')
    torch.testing.assert_close(loss,ce+5*aux); loss.backward()
    assert prediction.grad is not None and target.grad is None and logits.grad is not None
    with pytest.raises(PermissionError): training.objective(logits,labels,prediction,target,arm='ce')
    loss,ce,aux=training.objective(logits,labels,arm='ce')
    assert aux.item()==0; torch.testing.assert_close(loss,ce)
    with pytest.raises(ValueError): training.objective(logits,labels,prediction*float('nan'),target,arm='embed')


def test_ce_update_matches_native_continuation_exactly(tmp_path, monkeypatch):
    tiny(monkeypatch); tr=cache(); val=cache('validation'); initial=initial_state()
    augmented = new_model(node())
    report, actual = training.train(augmented,tr,val,None,node(),'ce','cpu',
        initial_state=initial,initial_sha256='e'*64,technical=True)
    torch.manual_seed(node()['initialization_seed'])
    native=DelphesParticleTransformer(); native.load_state_dict(initial)
    optimizer=training.optimizer_for(native)
    for group in optimizer.param_groups: group['lr']=c.learning_rate(1)
    ix=np.random.default_rng(np.random.SeedSequence([node()['sampler_seed'],1])).permutation(len(tr))
    raw=tr.batch(ix); native.train()
    loss=torch.nn.functional.cross_entropy(_forward(native,raw,'cpu',bf16=False),torch.from_numpy(raw['labels']))
    loss.backward(); optimizer.step()
    for k,v in native.state_dict().items(): torch.testing.assert_close(actual[k],v,rtol=0,atol=0)
    assert report['history'][0]['auxiliary_loss']==0 and report['parents']['bank'] is None
    with pytest.raises(PermissionError): training.train(new_model(node()),tr,val,object(),node(),'ce','cpu',
        initial_state=initial,initial_sha256='e'*64,technical=True)


@pytest.mark.parametrize('arm', c.ARMS)
def test_full30_poor_metrics_are_results_and_selection_is_authenticated(tmp_path, monkeypatch, arm):
    tiny(monkeypatch); tr,bank,_=bank_fixture(tmp_path); val=cache('validation'); initial=initial_state()
    original=training.evaluate_probabilities; calls=[]
    def poor(labels,probs):
        calls.append(1); return dict(original(labels,probs),macro_ovr_auc=.4 if len(calls)==1 else .1,accuracy=0.)
    monkeypatch.setattr(training,'evaluate_probabilities',poor)
    report,state=training.train(new_model(node()),tr,val,bank if arm=='embed' else None,node(),arm,'cpu',
        initial_state=initial,initial_sha256='e'*64)
    spec=dict(node=node(),counts=dict(train=22,validation=22),initial_checkpoint=dict(sha256='e'*64),
              anchor=dict(evidence=dict(prepared='a'*64)))
    training.validate_report(report,spec,bank.manifest['content_hash'],arm)
    assert report['passes']==30 and report['selected_pass']==1
    assert not any(k.startswith('predictor.') for k in state)
    assert any(not torch.equal(v,state[k]) for k,v in initial.items() if k.startswith('mod.fc.'))
    changed=with_content_hash(dict(report,selected_pass=30,selected=report['history'][-1]))
    with pytest.raises(ValueError,match='selected'): training.validate_report(changed,spec,bank.manifest['content_hash'],arm)


def test_exact_debug_dag_and_dry_submit(tmp_path, monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as exact
    spec=spec_fixture(tmp_path); monkeypatch.setattr(campaign,'check',lambda _: None)
    monkeypatch.setattr(submission,'check_site',lambda _: None); calls=[]
    def submit(cmd,**kw): calls.append(cmd); return SimpleNamespace(stdout=str(100+len(calls)))
    monkeypatch.setattr(exact.subprocess,'run',submit)
    plan=submission.submit(spec); assert not calls and plan['jobs']==3
    assert [r['dependencies'] for r in plan['commands']]==[[],['setup'],['setup']]
    for row in plan['commands']:
        assert '--partition=debug' in row['command'] and '--mem=90000M' in row['command']
        assert row['command'][-3:]==['/project',str(tmp_path/'campaign_spec.json'),row['task_id']]
    with pytest.raises(PermissionError): submission.submit(spec,execute=True,plan_hash='wrong',authorization=c.AUTHORIZATION)
    ledger=submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert ledger['jobs']==dict(setup='101',ce='102',embed='103')
    assert all('--dependency=afterok:101' in cmd and '--kill-on-invalid-dep=yes' in cmd for cmd in calls[1:])
    assert submission.accepted_jobs(spec)==ledger['jobs']
    assert submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)==ledger
    assert len(calls)==3


def test_ambiguous_submit_not_retried(tmp_path, monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as exact
    spec=spec_fixture(tmp_path); monkeypatch.setattr(campaign,'check',lambda _: None)
    monkeypatch.setattr(submission,'check_site',lambda _: None)
    plan=submission.submit(spec); calls=[]
    def lost(*a,**kw): calls.append(1); raise RuntimeError('lost response')
    monkeypatch.setattr(exact.subprocess,'run',lost)
    with pytest.raises(RuntimeError): submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    with pytest.raises(FileExistsError): submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==1


@pytest.mark.parametrize('key,value',[('peak_rss_bytes',90*2**30),('gpu_peak_bytes',40*2**30),('peak_rss_bytes',0)])
def test_resource_failures(tmp_path,key,value):
    spec=spec_fixture(tmp_path)
    m=dict(environment=spec['anchor']['evidence']['environment'],gpu=dict(total_memory_bytes=40*2**30),
        peak_rss_bytes=2**30,gpu_peak_bytes=2**30,timings=dict(authentication=10.,load_HLT=10.,ce_train=20.,
        ce_validation=10.,embed_train=20.,embed_validation=10.,readback=10.))
    assert set(worker.admission(spec,m))=={'ce','embed'}
    with pytest.raises((ValueError,MemoryError)): worker.admission(spec,dict(m,**{key:value}))
    m['timings']['embed_train']=100000
    with pytest.raises(ValueError,match='walltime'): worker.admission(spec,m)


@pytest.mark.parametrize('arm', c.ARMS)
def test_failed_setup_blocks_both_science_arms(tmp_path,monkeypatch,arm):
    spec=spec_fixture(tmp_path); monkeypatch.setattr(campaign,'check',lambda _: None)
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a,**k: dict(setup='1',ce='2',embed='3'))
    monkeypatch.setattr(worker,'validate_task',lambda *a: dict(job_id='1',content_hash='a'*64))
    def failed(*a): raise ValueError('setup FAILED')
    monkeypatch.setattr(worker,'preflight_accounting',failed)
    monkeypatch.setattr(worker,'fit',lambda *a,**kw: pytest.fail('Must not train'))
    with pytest.raises(ValueError,match='setup FAILED'): worker.run(spec,arm)
    assert not (tmp_path/arm).exists()


def test_source_rejects_shared_modification_and_reports_missing(tmp_path,monkeypatch):
    monkeypatch.setattr(source,'_source',lambda *a: dict(content_hash='a'*64))
    monkeypatch.setattr(source,'__file__',str(tmp_path/'src/hlt_classification/luka_ce_warmstart/source.py'))
    def git(cmd,**kw): return SimpleNamespace(stdout='\n'.join(sorted(source.REQUIRED)) if 'ls-files' in cmd else 'M\tsrc/shared.py\n')
    monkeypatch.setattr(source.subprocess,'run',git)
    with pytest.raises(ValueError,match='donor source'): source.current(tmp_path,'f'*40)
    monkeypatch.setattr(source.subprocess,'run',lambda *a,**kw: SimpleNamespace(stdout=''))
    with pytest.raises(ValueError,match='required'): source.current(tmp_path,'f'*40)


def test_initial_checkpoint_cannot_be_substituted(tmp_path,monkeypatch):
    spec=spec_fixture(tmp_path)
    reg=campaign.registration
    monkeypatch.setattr(campaign,'registration',lambda p: dict(reg(p),counts=dict(train=22,validation=22),original50k_rows=22))
    campaign.validate_spec(spec,payloads=False)
    changed=deepcopy(spec); changed['initial_checkpoint']['sha256']='1'*64
    # Deepcopy preserves alias between anchor and spec in this fixture; sever it.
    changed['anchor']=spec['anchor']
    with pytest.raises(ValueError,match='Initial checkpoint'): campaign.validate_spec(with_content_hash(changed),payloads=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires installed-Weaver CUDA')
def test_real_cuda_setup_and_two_independent30_epoch_production_workers(tmp_path,monkeypatch):
    tiny(monkeypatch)
    from hlt_classification.jetclass2_delphes import acceptance, model as native_model
    monkeypatch.setattr(acceptance,'model_config',native_model.model_config)
    spec=spec_fixture(tmp_path/'campaign'); reg=campaign.registration
    monkeypatch.setattr(campaign,'registration',lambda p: dict(reg(p),counts=dict(train=22,validation=22),original50k_rows=22))
    teacher=DelphesParticleTransformer().to('cuda'); save_state(tmp_path/'teacher.pt',teacher.state_dict())
    initial=initial_state(); initial_model=DelphesParticleTransformer().to('cuda'); initial_model.load_state_dict(initial)
    save_state(tmp_path/'initial.pt',initial)
    for name,model,coordinate in [('teacher',teacher,'U100'),('initial',initial_model,'D000')]:
        val=cache('validation',coordinate)
        a=dict(identities=val.identities,labels=val.labels,probabilities=predict(model,val,device='cuda',batch_size=128))
        atomic_publish_bytes(tmp_path/(name+'.npz'),deterministic_npz_bytes(a))
    spec['teacher']['files']={key:file_ref(tmp_path/('teacher.pt' if key=='checkpoint' else 'teacher.npz'))
                             for key in ('checkpoint','report','predictions','receipt')}
    spec['reference']['files']=dict(predictions=file_ref(tmp_path/'teacher.npz'))
    spec['initial_checkpoint']=file_ref(tmp_path/'initial.pt')
    spec['anchor']['evidence']['models']={n:dict(predictions=file_ref(tmp_path/('initial.npz' if n=='M0HLT' else 'teacher.npz')),
        checkpoint=file_ref(tmp_path/'initial.pt')) for n in ('M0HLT','OFFLINE')}
    spec['anchor']['evidence']['environment']=worker.installed_environment(); spec['anchor']=with_content_hash(spec['anchor'])
    for key in ('teacher','reference'):
        spec[key]['parents']['anchor']=spec['anchor']['content_hash']; spec[key]=with_content_hash(spec[key])
    for key in ('teacher','reference','anchor'): spec['parents'][key]=spec[key]['content_hash']
    spec=with_content_hash(spec); (Path(spec['root'])/'campaign_spec.json').write_text(json.dumps(spec))
    monkeypatch.setattr(campaign,'check',lambda _: None); monkeypatch.setattr(source,'current',lambda *a: spec['source_snapshot'])
    jobs=dict(setup='1',ce='2',embed='3'); monkeypatch.setattr(submission,'accepted_jobs',lambda *a,**kw: jobs)
    monkeypatch.setattr(worker,'execution',lambda *a: '1'); monkeypatch.setattr(worker,'preflight_accounting',lambda *a: None)
    monkeypatch.setattr(worker,'data',lambda s,r,co,meta: cache(r,co))
    monkeypatch.setattr(worker.regular_cache,'metadata',lambda _: ({},dict(summaries={
        r:dict(D000=dict(maximum=4)) for r in ('train','validation')}),{},{}))
    monkeypatch.setattr(worker.alljets_data,'comparison_indices',lambda *a: np.arange(22,dtype=np.int64))
    monkeypatch.setattr(worker,'memory_snapshot',lambda _: dict(rss_bytes=2**30,peak_rss_bytes=2**30,
        gpu_peak_bytes=int(torch.cuda.max_memory_allocated())))
    worker.run(spec,'setup')
    assert not (Path(spec['root'])/'targets/validation.npz').exists()
    assert set(worker.results(spec)['pending'])==set(c.ARMS.values())
    # Control must not even decode the TRAIN bank.
    bank_class=targets.Bank
    monkeypatch.setattr(targets,'Bank',lambda *a,**kw: pytest.fail('CE may only hash, not decode, shared targets'))
    monkeypatch.setattr(worker,'execution',lambda *a: '2'); worker.run(spec,'ce')
    assert worker.results(spec)['pending']==[c.ARMS['embed']]
    monkeypatch.setattr(targets,'Bank',bank_class)
    monkeypatch.setattr(worker,'execution',lambda *a: '3'); worker.run(spec,'embed')
    records=[load_json(Path(spec['root'])/arm/'fit/training_report.json') for arm in c.ARMS]
    assert records[0]['initial_checkpoint']==records[1]['initial_checkpoint']
    assert all(r['initial_checkpoint']['sha256']==spec['initial_checkpoint']['sha256'] for r in records)
    assert all(r['report']['passes']==30 and not r['report']['technical'] for r in records)
    assert all(not any(k.startswith('predictor.') for k in torch.load(r['checkpoint']['path'],weights_only=True)) for r in records)
    with pytest.raises(FileExistsError): worker.run(spec,'embed')
    monkeypatch.setattr(torch,'load',lambda *a,**kw: pytest.fail('Reporting must not load a model'))
    result=worker.results(spec); assert len(result['rows'])==6 and not result['pending']
    assert result['rows'][-2]['validation_auc_delta_vs_continued_CE']==0
    worker.print_results(result)
    (Path(spec['root'])/'ce/fit/validation_predictions.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError): worker.results(spec)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires installed-Weaver CUDA')
def test_production_model_installed_weaver_and_predictor_parity():
    # Full native width/depth; not a tiny-model fixture.
    with worker.parity_backend('cuda'):
        assert worker.installed_parity(cache(),device='cuda')['passed']
    for bf16 in (False,True):
        assert worker.parity(cache().batch(np.arange(4)),node(),'cuda',bf16=bf16)['zero_control_update']


def test_scripts_required_paths_and_bash_syntax():
    root=Path(__file__).resolve().parents[1]
    assert all((root/p).is_file() for p in source.REQUIRED)
    script=(root/'sbatch/run_luka_ce_warmstart.sh').read_text()
    for token in ('${PROJECT_DIR}/sbatch/common.sh','PYTHONNOUSERSITE=1','${CONDA_PREFIX}/lib','CUBLAS_WORKSPACE_CONFIG=:4096:8'):
        assert token in script
    queue=(root/'scripts/queue_luka_ce_warmstart.sh').read_text()
    assert not any(token in queue for token in ('scancel','scontrol update','rm -'))
    assert queue.count('</dev/tty')==3 and 'train_M0HLT/selected.pt' in queue
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    if bash.is_file():
        for path in ('scripts/queue_luka_ce_warmstart.sh','sbatch/run_luka_ce_warmstart.sh'):
            subprocess.run([str(bash),'-n',path],cwd=root,check=True,capture_output=True)

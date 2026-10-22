"""Direct U100 target semantics, inherited execution, real CUDA fixed30 miniature."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import re
import subprocess
import numpy as np
import pytest
import torch
from test_luka_u100_pretrain import cache
from test_luka_attention import parent_nodes
from test_luka_embedding_aux import tiny
from hlt_classification.data.cache_contracts import (
    load_json, with_content_hash, write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.luka_embedding_aux.model import new_model
from hlt_classification.luka_embedding_aux import contracts as anchor_contracts
from hlt_classification.luka_d066_embedding import contracts as reference_contracts, preflight as old_preflight
from hlt_classification.luka_u100_embedding import direct, source, worker, targets, training, contracts as c


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def fixture_gate(root, teacher=None, reference=None, *, site_name='sporc_a100'):
    anchor = anchor_contracts.artifact('ANCHOR', spec=dict(path='/parent/spec.json'), evidence=dict(
        prepared='a'*64, cache='b'*64, prepared_root='/prepared', raw_container='/raw',
        environment=dict(fixture=True), models={}))
    teacher = teacher or c.artifact('TEACHER', coordinate='U100', target_role='train', prepared='a'*64, files={})
    teacher['spec'] = anchor['spec']
    teacher['node'] = dict(node_id='COARSE_U100', coordinate='U100', context_coordinate=None)
    for key in ('checkpoint','report','predictions','receipt'):
        teacher['files'].setdefault(key, {})
    reference = reference or reference_contracts.artifact('REFERENCE', spec=dict(path='/reference/spec.json'), files={})
    reference['node'] = c.node(dict(nodes=parent_nodes()))
    for value in (teacher, reference):
        value['parents']['anchor'] = anchor['content_hash']
    teacher, reference = map(with_content_hash, (teacher, reference))
    snapshot = dict(content_hash='c'*64)
    return dict(root=str(root), project_dir='/project', source_commit='f'*40, source_snapshot=snapshot,
        anchor=anchor, teacher=teacher, reference=reference, content_hash='d'*64, recipe=c.recipe(),
        node=c.node(dict(nodes=parent_nodes())), counts=dict(train=22, validation=22))


def teacher_fixture(root):
    model = DelphesParticleTransformer()
    save_state(root/'teacher.pt', model.state_dict())
    return model, c.artifact('TEACHER', coordinate='U100', target_role='train', prepared='a'*64,
        files=dict(checkpoint=file_ref(root/'teacher.pt')))


def bank_fixture(root):
    teacher, record = teacher_fixture(root)
    targets.export(root/'targets', teacher, cache(coordinate='U100'), teacher_record=record, device='cpu')
    tr = cache(coordinate='D000')
    return tr, targets.Bank(root/'targets', tr, teacher_record=record), record


def test_train_only_bank_normalization_and_teacher_binding(tmp_path,monkeypatch):
    tiny(monkeypatch); tr,bank,record=bank_fixture(tmp_path)
    assert bank.targets.shape==(22,128) and bank.targets.dtype==np.float32
    np.testing.assert_allclose(bank.targets.mean(0),0,atol=1e-6)
    assert np.max(bank.targets.std(0))<=1.00001
    for wrong in (cache('validation',coordinate='U100'),cache(coordinate='OFFLINE'),cache(coordinate='D000')):
        with pytest.raises(PermissionError):
            targets.export(tmp_path/'wrong',DelphesParticleTransformer(),wrong,teacher_record=record,device='cpu')
    changed=deepcopy(tr); changed.identities=tr.identities[::-1].copy()
    with pytest.raises(ValueError): targets.Bank(tmp_path/'targets',changed,teacher_record=record)
    different=with_content_hash(dict(record,prepared='b'*64))
    with pytest.raises(ValueError): targets.Bank(tmp_path/'targets',tr,teacher_record=different)
    (tmp_path/'targets/targets.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError): targets.Bank(tmp_path/'targets',tr,teacher_record=record)


@pytest.mark.parametrize('field,value',[('role','validation'),('coordinate','OFFLINE'),('width',127),('rows',21)])
def test_resigned_wrong_target_semantics(tmp_path,monkeypatch,field,value):
    tiny(monkeypatch); tr,_,record=bank_fixture(tmp_path)
    path=tmp_path/'targets/manifest.json'
    path.write_text(json.dumps(with_content_hash(dict(load_json(path),**{field:value}))))
    with pytest.raises(ValueError): targets.Bank(tmp_path/'targets',tr,teacher_record=record)


def test_loss_and_detached_teacher():
    logits=torch.randn(4,11,requires_grad=True); prediction=torch.randn(4,128,requires_grad=True)
    target=torch.randn(4,128,requires_grad=True)
    loss,ce,aux=training.objective(logits,torch.arange(4),prediction,target,coefficient=5.)
    torch.testing.assert_close(loss,ce+5*aux)
    expected=torch.autograd.grad(aux,prediction,retain_graph=True)[0]
    loss.backward(); torch.testing.assert_close(prediction.grad,5*expected)
    assert target.grad is None and torch.count_nonzero(logits.grad)
    for invalid in (True,.3,10,float('nan')):
        with pytest.raises(ValueError): training.objective(logits,torch.arange(4),prediction,target,coefficient=invalid)


def test_same_targets_exactly_match_offline_w5_kernel(tmp_path,monkeypatch):
    from hlt_classification.luka_embedding_sweep import training as donor
    from hlt_classification.luka_embedding_sweep.contracts import nodes
    tiny(monkeypatch); tr,bank,_=bank_fixture(tmp_path); val=cache('validation',coordinate='D000')
    n=c.node(dict(nodes=parent_nodes())); old_node=nodes(dict(nodes=parent_nodes()))['w5']
    first,state=training.train(new_model(n),tr,val,bank=bank,node=n,device='cpu',technical=True)
    second,expected=donor.train(new_model(old_node),tr,val,bank=bank,node=old_node,device='cpu',technical=True)
    assert first['validation']==second['validation']
    for k in state: torch.testing.assert_close(state[k],expected[k],rtol=0,atol=0)


def test_full30_native_checkpoint_and_poor_metrics_do_not_gate(tmp_path,monkeypatch,capsys):
    tiny(monkeypatch)
    tr,bank,_=bank_fixture(tmp_path); gate=fixture_gate(tmp_path)
    val=cache('validation',coordinate='D000')
    worker.fit(tmp_path/'fit',gate,tr,val,bank,device='cpu',technical=False)
    report,_=worker.validate_fit(tmp_path/'fit',gate,bank.manifest['content_hash'],technical=False)
    assert report['passes']==len(report['validation_history'])==30
    assert not any(k.startswith('predictor.') for k in torch.load(tmp_path/'fit/selected.pt',weights_only=True))
    original=training.evaluate_probabilities; calls=[]
    def poor(labels,probabilities):
        calls.append(1)
        return dict(original(labels,probabilities),macro_ovr_auc=.4 if len(calls)==1 else .1,accuracy=0.)
    monkeypatch.setattr(training,'evaluate_probabilities',poor)
    report,_=training.train(new_model(gate['node']),tr,val,bank=bank,node=gate['node'],device='cpu')
    assert report['passes']==30 and report['selected_pass']==1
    worker.report_valid(report,gate,bank.manifest['content_hash'],technical=False)
    assert 'epoch=30/30 auc=0.10000000 best_auc=0.40000000 best_epoch=1' in capsys.readouterr().out


def test_teacher_readback_rejects_wrong_join_and_probabilities(tmp_path,monkeypatch):
    tiny(monkeypatch); teacher,record=teacher_fixture(tmp_path)
    val=cache('validation',coordinate='U100')
    arrays=dict(identities=val.identities,labels=val.labels,probabilities=predict(teacher,val,device='cpu',batch_size=128))
    atomic_publish_bytes(tmp_path/'pred.npz',deterministic_npz_bytes(arrays))
    record['files']['predictions']=file_ref(tmp_path/'pred.npz'); record=with_content_hash(record)
    proof=worker.teacher_readback(teacher,val,record,'cpu')
    assert proof['native_probability_parity'] and not proof['normalization_used']
    bad=deepcopy(val); bad.identities=val.identities[::-1].copy()
    with pytest.raises(ValueError): worker.teacher_readback(teacher,bad,record,'cpu')
    wrong=DelphesParticleTransformer()
    with pytest.raises(AssertionError): worker.teacher_readback(wrong,val,record,'cpu')



def inherited_fixture(folder, gate):
    folder.mkdir()
    original = dict(execution_site=execution_site('sporc_a100'), anchor=gate['anchor'], counts=direct.COUNTS)
    report = with_content_hash(dict(train_minutes=dict(w5=190), environment=gate['anchor']['evidence']['environment'],
        gpu=dict(name='fixture GPU',total_memory_bytes=40*2**30), technical_checks_passed=True,
        checkpoint_probability_roundtrip=True))
    fit = dict(job_id='21940431')
    for name, value in zip(('gate.json','report.json','fit.json'), (original,report,fit)):
        write_immutable_json(folder/name,value)
    reference = with_content_hash(dict(gate['reference'], files=dict(gate['reference']['files'],receipt=file_ref(folder/'fit.json'))))
    proof = c.artifact('DIRECT_RESOURCES', parents=dict(reference=reference['content_hash']),
        policy=direct.POLICY, donor_preflight_hash=report['content_hash'], environment=report['environment'],
        gpu=report['gpu'], measured_w5_minutes=190, setup_allowance_minutes=120, train_minutes=360,
        donor_refs=[file_ref(folder/n) for n in ('gate.json','report.json','fit.json')],
        w5_accounting=dict(job_id='21940431',state='COMPLETED',exit_code='0:0'),new_U100_preflight_performed=False)
    return reference, proof


def direct_fixture(tmp_path, *, gate=None):
    root=tmp_path/'direct'; root.mkdir()
    gate=gate or fixture_gate(root,site_name='sporc_a100')
    reference, inherited=inherited_fixture(tmp_path/'donor',gate)
    spec=c.artifact('DIRECT_SPEC',parents=dict(source=gate['source_snapshot']['content_hash'],
        anchor=gate['anchor']['content_hash'],teacher=gate['teacher']['content_hash'],
        reference=reference['content_hash'],resources=inherited['content_hash']),root=str(root),
        project_dir=gate['project_dir'],source_commit=gate['source_commit'],source_snapshot=gate['source_snapshot'],
        anchor=gate['anchor'],teacher=gate['teacher'],reference=reference,resources=inherited,available_quota_gib=2,
        **direct.registration(dict(nodes=parent_nodes()),inherited))
    write_immutable_json(root/'campaign_spec.json',spec)
    return spec


def test_direct_one_job_not_preflight_or_probe(tmp_path):
    spec=direct_fixture(tmp_path); direct.validate_spec(spec)
    plan=direct.render(spec); cmd=plan['commands'][0]['command']
    assert plan['jobs']==1 and plan['preflight_jobs']==0 and plan['mode']=='direct_science'
    assert plan['commands'][0]['dependencies']==[] and cmd[-1]=='direct'
    for flag in ('--partition=tier3','--time=06:00:00','--mem=90000M','--no-requeue','--export=NONE'):
        assert flag in cmd
    assert spec['recipe']==c.recipe() and spec['node']['auxiliary_weight']==5
    assert spec['technical_training_epochs']==0 and not spec['recipe']['logit_kd']
    assert spec['counts']==dict(train=299234,validation=74895)
    with pytest.raises(ValueError): c.validate(spec,'SPEC')


@pytest.mark.parametrize('change',[dict(execution_site=execution_site('sporc_a100_debug')),
    dict(train_minutes=150),dict(technical_training_epochs=1),dict(preflight_jobs=1),dict(available_quota_gib=1)])
def test_resigned_direct_changes_rejected(tmp_path,change):
    spec=with_content_hash(dict(direct_fixture(tmp_path),**change))
    (Path(spec['root'])/'campaign_spec.json').write_text(json.dumps(spec))
    with pytest.raises(ValueError): direct.validate_spec(spec)


@pytest.mark.parametrize('change',[dict(new_U100_preflight_performed=True),dict(measured_w5_minutes=1321),
    dict(donor_preflight_hash='f'*64),dict(gpu=dict(name='substituted')),dict(setup_allowance_minutes=0)])
def test_inherited_resources_fail_closed(tmp_path,change):
    spec=direct_fixture(tmp_path)
    bad=with_content_hash(dict(spec['resources'],**change))
    with pytest.raises(ValueError): direct.validate_resources(bad,spec['reference'],spec['anchor'])


def test_resource_original_source_program(tmp_path,monkeypatch,capsys):
    import sys
    from hlt_classification.luka_embedding_sweep import debug
    from hlt_classification.luka_fullsim import science_campaign
    root=tmp_path/'reference'; (root/'w5').mkdir(parents=True)
    gate=dict(root=str(root),anchor=dict(a=1),counts=direct.COUNTS,nodes=dict(w5={'auxiliary_weight':5}),
        execution_site=execution_site('sporc_a100'),cpus=6,memory_mb=90000)
    write_immutable_json(root/'gate.json',gate)
    spec=dict(root=str(root),gate_path=str(root/'gate.json'))
    write_immutable_json(root/'spec.json',spec)
    (root/'preflight').mkdir(); write_immutable_json(root/'preflight/receipt.json',dict(saved=True))
    write_immutable_json(root/'w5/receipt.json',dict(job_id='123'))
    calls=[]
    def checked(s,*,retired): assert s==spec and retired; calls.append('original_check'); return gate,dict(saved=True)
    monkeypatch.setattr(debug,'check_spec',checked)
    monkeypatch.setattr(debug,'results',lambda s:dict(pending=[]))
    monkeypatch.setattr(science_campaign,'preflight_accounting',lambda j:dict(job_id=j,state='COMPLETED',exit_code='0:0'))
    monkeypatch.setattr(sys,'argv',['replay',str(root/'spec.json')])
    exec(compile(direct.RESOURCE_CHECK,'original-replay','exec'),{})
    out=capsys.readouterr().out; data=json.loads(out.split('U100_DIRECT_RESOURCES=')[1])
    assert calls==['original_check'] and data['accounting']['job_id']=='123'
    monkeypatch.setattr(debug,'results',lambda s:dict(pending=['w5']))
    with pytest.raises(ValueError,match='Completed matched'): exec(compile(direct.RESOURCE_CHECK,'replay','exec'),{})


def test_exact_direct_submission_and_ambiguous_response(tmp_path,monkeypatch):
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    spec=direct_fixture(tmp_path)
    monkeypatch.setattr(direct,'check_spec',lambda _:None)
    monkeypatch.setattr(direct,'check_site',lambda _:None)
    plan=direct.submit(spec); calls=[]
    def lost(cmd,**kwargs): calls.append(cmd); raise RuntimeError('lost reply')
    monkeypatch.setattr(exact.subprocess,'run',lost)
    with pytest.raises(PermissionError): direct.submit(spec,execute=True,plan_hash='bad',authorization=direct.AUTHORIZATION)
    with pytest.raises(RuntimeError): direct.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=direct.AUTHORIZATION)
    with pytest.raises((FileExistsError,PermissionError,ValueError)):
        direct.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=direct.AUTHORIZATION)
    assert len(calls)==1 and '--partition=tier3' in calls[0]


def test_successful_exact_direct_journal_is_idempotent(tmp_path,monkeypatch):
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    spec=direct_fixture(tmp_path)
    monkeypatch.setattr(direct,'check_spec',lambda _:None)
    monkeypatch.setattr(direct,'check_site',lambda _:None)
    calls=[]
    def submitted(cmd,**kw): calls.append(cmd); return SimpleNamespace(stdout='12345\n')
    monkeypatch.setattr(exact.subprocess,'run',submitted)
    plan=direct.submit(spec)
    assert not calls
    result=direct.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=direct.AUTHORIZATION)
    assert result['jobs']=={'direct':'12345'} and direct.accepted_job(spec)=='12345'
    assert direct.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=direct.AUTHORIZATION)==result
    assert len(calls)==1
    path=Path(spec['root'])/'submission/submission_ledger.json'
    ledger=load_json(path); ledger['jobs']['direct']='different'; path.write_text(json.dumps(with_content_hash(ledger)))
    with pytest.raises(ValueError): direct.accepted_job(spec)


def test_create_direct_authenticates_and_check_spec_detects_changed_import(tmp_path,monkeypatch):
    spec=direct_fixture(tmp_path)
    teacher=spec['teacher']
    parent=dict(nodes=parent_nodes(),project_dir='/parent-source',root='/parent/science')
    calls=[]
    monkeypatch.setattr(source,'current',lambda *a:spec['source_snapshot'])
    monkeypatch.setattr(source,'authenticate',lambda *a:(spec['anchor'],parent))
    monkeypatch.setattr(source,'teacher',lambda *a:teacher)
    monkeypatch.setattr(source,'reference',lambda *a:spec['reference'])
    monkeypatch.setattr(source,'validate_file_ref',lambda *a:None)
    monkeypatch.setattr(direct,'resources',lambda *a:spec['resources'])
    root=tmp_path/'new'
    value=direct.create(root,'/parent/science/spec.json','/reference/science/spec.json',
        '/project','f'*40,available_quota_gib=2)
    direct.check_spec(value)
    assert load_json(root/'campaign_spec.json')==value and not (root/'gate_spec.json').exists()
    monkeypatch.setattr(source,'teacher',lambda *a:dict(teacher,unexpected=True))
    with pytest.raises(ValueError,match='imported evidence'): direct.check_spec(value)


def test_resource_import_binding_and_failure(tmp_path,monkeypatch):
    spec=direct_fixture(tmp_path)
    reference=spec['reference']; path=tmp_path/'reference.json'
    write_immutable_json(path,dict(source_commit=source.REFERENCE_COMMIT,project_dir='/original'))
    reference=with_content_hash(dict(reference,spec=file_ref(path)))
    report=load_json(spec['resources']['donor_refs'][1]['path'])
    data=dict(node=spec['node'],anchor=spec['anchor'],counts=direct.COUNTS,report=report,
        cpus=6,memory_mb=90000,site=execution_site('sporc_a100'),refs=spec['resources']['donor_refs'],
        accounting=dict(job_id='21940431',state='COMPLETED',exit_code='0:0'))
    monkeypatch.setattr(source,'replay',lambda *a:(reference['spec'],data))
    value=direct.resources(reference,spec['anchor'],spec['node'])
    assert value['train_minutes']==360 and not value['new_U100_preflight_performed']
    data['counts']=dict(train=100000,validation=50000)
    with pytest.raises(ValueError,match='Inherited resource evidence'):
        direct.resources(reference,spec['anchor'],spec['node'])


def test_dry_import_failure_cannot_submit(tmp_path,monkeypatch):
    spec=direct_fixture(tmp_path)
    def invalid(_): raise ValueError('missing accepted resource receipt')
    monkeypatch.setattr(direct,'check_spec',invalid)
    monkeypatch.setattr(direct,'check_site',lambda *a:pytest.fail('No scheduler check after invalid lineage'))
    with pytest.raises(ValueError,match='missing'): direct.submit(spec)
    assert not (Path(spec['root'])/'submission').exists()


def test_actual_cuda_direct_fixed30_no_probe(tmp_path,monkeypatch,capsys):
    if not torch.cuda.is_available(): pytest.skip('Real CUDA required')
    # Real installed model, target export, all30 scientific epochs and native
    # readback. Only small population and remote state are fixtures.
    monkeypatch.setattr(direct,'COUNTS',dict(train=22,validation=22))
    original=direct.registration
    monkeypatch.setattr(direct,'registration',lambda p,r:dict(original(p,r),original50k_rows=22))
    tiny(monkeypatch)
    teacher,record=teacher_fixture(tmp_path)
    val=cache('validation',coordinate='U100')
    probabilities=predict(teacher.to('cuda'),val,device='cuda',batch_size=128)
    atomic_publish_bytes(tmp_path/'pred.npz',deterministic_npz_bytes(dict(
        identities=val.identities,labels=val.labels,probabilities=probabilities)))
    ref=file_ref(tmp_path/'pred.npz'); record=with_content_hash(dict(record,files=dict(record['files'],predictions=ref)))
    gate=fixture_gate(tmp_path/'direct',record,reference_contracts.artifact('REFERENCE',files=dict(predictions=ref)))
    gate['anchor']['evidence']['models']={n:dict(predictions=ref) for n in ('M0HLT','OFFLINE')}
    gate['anchor']=with_content_hash(gate['anchor'])
    for key in ('teacher','reference'):
        gate[key]['parents']['anchor']=gate['anchor']['content_hash']; gate[key]=with_content_hash(gate[key])
    spec=direct_fixture(tmp_path,gate=gate)
    monkeypatch.setattr(direct,'check_spec',lambda _:None)
    monkeypatch.setattr(worker,'execution',lambda *a:'77')
    monkeypatch.setattr(direct,'accepted_job',lambda *a:'77')
    monkeypatch.setattr(source,'current',lambda *a:spec['source_snapshot'])
    monkeypatch.setattr(worker,'data',lambda s,r,co,meta:cache(r,coordinate=co))
    monkeypatch.setattr(direct.regular_cache,'metadata',lambda _:({}, {}, {}, {}))
    monkeypatch.setattr(worker.alljets_data,'comparison_indices',lambda *a:np.arange(22,dtype=np.int64))
    monkeypatch.setattr(direct,'memory_snapshot',lambda _:dict(rss_bytes=2**30,peak_rss_bytes=2**30,
        gpu_peak_bytes=int(torch.cuda.max_memory_allocated())))
    monkeypatch.setattr(old_preflight,'run',lambda *a:pytest.fail('No preflight'))
    monkeypatch.setattr(old_preflight,'stress',lambda *a,**kw:pytest.fail('No stress rerun'))
    calls=[]; actual=worker.fit
    def fit(*a,**kw): calls.append(kw['technical']); return actual(*a,**kw)
    monkeypatch.setattr(worker,'fit',fit)
    result=direct.run(spec)
    assert calls==[False] and not (Path(spec['root'])/'preflight').exists()
    report=load_json(Path(spec['root'])/'fit/training_report.json')
    assert report['passes']==30 and not report['technical']
    assert len(result['rows'])==5 and result['new_preflight_performed'] is False
    assert not result['final_test_accessed'] and direct.results(spec)==result
    with pytest.raises(FileExistsError): direct.run(spec)
    monkeypatch.setattr(torch,'load',lambda *a,**kw:pytest.fail('No GPU/weights needed for results'))
    assert direct.results(spec)==result
    worker.print_results(result); assert 'AUC rec.%' in capsys.readouterr().out
    (Path(spec['root'])/'targets/targets.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError): direct.results(spec)


def test_script_and_runbook_syntax_and_no_mutations():
    import shutil
    root=Path(__file__).resolve().parents[1]
    local=Path('C:/Program Files/Git/bin/bash.exe')
    bash=str(local) if local.is_file() else shutil.which('bash')
    if not bash: pytest.skip('Bash required')
    path=root/'scripts/queue_luka_u100_embedding.sh'
    subprocess.run([bash,'-n',str(path)],check=True,capture_output=True)
    text=path.read_text()
    for required in ('PYTHONNOUSERSITE=1','${CONDA_PREFIX}/lib','CUBLAS_WORKSPACE_CONFIG=:4096:8',
        'create-direct','submit-direct','</dev/tty'):
        assert required in text
    for forbidden in ('scancel','scontrol','ORDER_PENDING',' auto --',' preflight --'):
        assert forbidden not in text
    for block in re.findall(r'```bash\n(.*?)\n```',(root/'docs/LUKA_U100_EMBEDDING_RUNBOOK.md').read_text(),re.S):
        subprocess.run([bash,'-n'],input=block,text=True,check=True,capture_output=True)


def test_single_phase_registration_distinct_from_debug_pretrain():
    from hlt_classification.luka_u100_pretrain.contracts import recipe as two_stage
    r = c.recipe()
    assert r['epochs'] == 30 and r['ce_coefficient'] == 1 and r['auxiliary_coefficient'] == 5
    assert r['teacher'] == 'COARSE_U100' and r['target_coordinate'] == 'U100' and r['target_role'] == 'train'
    assert not any(r[k] for k in ('warm_start', 'logit_kd', 'attention_kd'))
    assert 'pretrain_epochs' not in r and two_stage()['pretrain_epochs'] == 20
    with pytest.raises(ValueError): c.validate(two_stage(), 'RECIPE')
    student = c.node(dict(nodes=parent_nodes()))
    assert student['coordinate'] == 'D000' and student['teacher'] is None and student['context_coordinate'] is None


@pytest.mark.parametrize('field,value', [('coordinate', 'D066'), ('target_role', 'validation'),
    ('prepared', 'wrong'), ('node', dict(node_id='FUSION_U100', coordinate='U100', context_coordinate='U050'))])
def test_import_semantics_reject_teacher_substitution(tmp_path, field, value):
    spec = direct_fixture(tmp_path)
    spec['teacher'] = with_content_hash(dict(spec['teacher'], **{field: value}))
    with pytest.raises(ValueError): source.validate_imports(spec, payloads=False)


def test_teacher_replay_checks_completed_replacement_artifacts(tmp_path, monkeypatch, capsys):
    import sys
    from hlt_classification.luka_fullsim import alljets_fusion_campaign as campaign, alljets_fusion_worker as original
    folder = tmp_path/'science/train_COARSE_U100'; folder.mkdir(parents=True)
    for filename in ('selected.pt', 'training_report.json', 'validation_predictions.npz', 'receipt.json'):
        (folder/filename).write_bytes(b'fixture authenticated by original worker')
    path = tmp_path/'science/campaign_spec.json'
    write_immutable_json(path, dict(root=str(folder.parent), prepared_sha256='a'*64, project_dir='/original'))
    calls = []
    monkeypatch.setattr(campaign, 'validate_spec', lambda _: None)
    n = dict(node_id='COARSE_U100', coordinate='U100', context_coordinate=None)
    monkeypatch.setattr(campaign, 'scoped_nodes', lambda _: [n])
    def completed(s, task):
        calls.append(task); return dict(job_id='21910999')
    monkeypatch.setattr(original, 'completed', completed)
    monkeypatch.setattr(sys, 'argv', ['replay', str(path)])
    exec(compile(source.TEACHER_CHECK, 'U100-original-source', 'exec'), {})
    evidence = json.loads(capsys.readouterr().out.split('U100_TEACHER=')[1])
    assert calls == ['train_COARSE_U100']
    assert evidence['node'] == n and set(evidence['files']) == {'checkpoint', 'report', 'predictions', 'receipt'}
    anchor = anchor_contracts.artifact('ANCHOR', spec=file_ref(path), evidence=dict(prepared='a'*64))
    monkeypatch.setattr(source, 'replay', lambda *a: (anchor['spec'], evidence))
    record = source.teacher(anchor)
    assert record['target_role'] == 'train' and record['coordinate'] == 'U100'
    evidence['prepared'] = 'wrong'
    with pytest.raises(ValueError, match='population'): source.teacher(anchor)
    monkeypatch.setattr(original, 'completed', lambda *a: None)
    with pytest.raises(ValueError, match='incomplete'):
        exec(compile(source.TEACHER_CHECK, 'U100-original-source', 'exec'), {})
    n['context_coordinate'] = 'U050'
    with pytest.raises(ValueError, match='single-encoder'):
        exec(compile(source.TEACHER_CHECK, 'U100-original-source', 'exec'), {})


@pytest.mark.parametrize('diff', ['M\tsrc/shared.py', 'A\tsrc/unreviewed.py',
    'D\tsrc/hlt_classification/luka_u100_pretrain/source.py'])
def test_new_source_guard_preserves_all_existing_runtime(tmp_path, monkeypatch, diff):
    monkeypatch.setattr(source, '_source', lambda *a: dict(content_hash='a'*64))
    monkeypatch.setattr(source, '__file__', str(tmp_path/'src/hlt_classification/luka_u100_embedding/source.py'))
    monkeypatch.setattr(source.subprocess, 'run', lambda cmd, **kw: SimpleNamespace(
        stdout='\n'.join(sorted(source.REQUIRED)) if 'ls-files' in cmd else diff))
    with pytest.raises(ValueError, match='donor source'): source.current(tmp_path, 'f'*40)


def test_new_source_required_paths_and_reviewed_additive_baseline(monkeypatch):
    project = Path(__file__).resolve().parents[1]
    assert all((project/p).is_file() for p in source.REQUIRED)
    assert source.BASE == 'dd51e8e25770482134809dbb34d4766f8dac74b9'
    # Check the real committed baseline exists; no index or worktree mutation.
    subprocess.run(['git', '-C', str(project), 'cat-file', '-e', source.BASE+'^{commit}'], check=True)
    monkeypatch.setattr(source, '_source', lambda *a: dict(content_hash='a'*64))
    monkeypatch.setattr(source.subprocess, 'run', lambda cmd, **kw: SimpleNamespace(stdout=
        '\n'.join(sorted(source.REQUIRED)) if 'ls-files' in cmd else
        '\n'.join('A\t'+p for p in sorted(source.ADDITIONS))))
    assert source.current(project, 'f'*40) == dict(content_hash='a'*64)
    monkeypatch.setattr(source.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=''))
    with pytest.raises(ValueError, match='Commit required'): source.current(project, 'f'*40)


def test_cli_has_no_preflight_or_skip_validation_escape():
    import sys
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-B', str(project/'scripts/luka_u100_embedding.py'), '--help'],
                            check=True, capture_output=True, text=True)
    assert '{create-direct,submit-direct,run,results}' in result.stdout
    bad = subprocess.run([sys.executable, '-B', str(project/'scripts/luka_u100_embedding.py'), 'preflight'],
                         capture_output=True, text=True)
    assert bad.returncode != 0
    assert not hasattr(direct, 'cancelled_original')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires local installed-Weaver CUDA')
def test_production_model_native_and_predictor_parity():
    from hlt_classification.jetclass2_delphes.acceptance import installed_parity
    from hlt_classification.luka_embedding_aux.preflight import parity
    from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
    tr = cache()
    # Production architecture, no tiny() override. These are local tests only;
    # direct.run does not execute them or train a disposable probe remotely.
    with parity_backend('cuda'):
        installed_parity(tr, device='cuda')
    for bf16 in (False, True):
        parity(tr.batch(np.arange(4)), c.node(dict(nodes=parent_nodes())), 'cuda', bf16=bf16)

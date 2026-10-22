"""One actual fixed30 fit; no disguised new probe, no unbound resource bypass."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import re
import subprocess
import numpy as np
import pytest
import torch
from test_luka_attention import cache, parent_nodes
from test_luka_d066_embedding import fixture_gate, teacher_fixture, threads
from hlt_classification.data.cache_contracts import (
    load_json, with_content_hash, write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.luka_d066_embedding import direct, source, worker, preflight, campaign, contracts as c


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
        w5_accounting=dict(job_id='21940431',state='COMPLETED',exit_code='0:0'),new_D066_preflight_performed=False)
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
    with pytest.raises(ValueError): campaign.validate_gate_record(spec)
    with pytest.raises(ValueError): c.validate(spec,'SPEC')


@pytest.mark.parametrize('change',[dict(execution_site=execution_site('sporc_a100_debug')),
    dict(train_minutes=150),dict(technical_training_epochs=1),dict(preflight_jobs=1),dict(available_quota_gib=1)])
def test_resigned_direct_changes_rejected(tmp_path,change):
    spec=with_content_hash(dict(direct_fixture(tmp_path),**change))
    (Path(spec['root'])/'campaign_spec.json').write_text(json.dumps(spec))
    with pytest.raises(ValueError): direct.validate_spec(spec)


@pytest.mark.parametrize('change',[dict(new_D066_preflight_performed=True),dict(measured_w5_minutes=1321),
    dict(donor_preflight_hash='f'*64),dict(gpu=dict(name='substituted')),dict(setup_allowance_minutes=0)])
def test_inherited_resources_fail_closed(tmp_path,change):
    spec=direct_fixture(tmp_path)
    bad=with_content_hash(dict(spec['resources'],**change))
    with pytest.raises(ValueError): direct.validate_resources(bad,spec['reference'],spec['anchor'])


@pytest.mark.parametrize('line,passed',[
    ('21944417|lukad6e_preflight|ryreu|reu-aisocial|debug|CANCELLED by 2914834|0|',True),
    ('21944417|lukad6e_preflight|ryreu|reu-aisocial|debug|CANCELLED|0',True),
    ('21944417|lukad6e_preflight|ryreu|reu-aisocial|debug|RUNNING|1|',False),
    ('21944417|lukad6e_preflight|ryreu|reu-aisocial|debug|COMPLETED|30|',False),
    ('21944417|lukad6e_preflight|ryreu|reu-aisocial|debug|CANCELLED|1|',False),
    ('21944417|unrelated|ryreu|reu-aisocial|debug|CANCELLED|0|',False),
    ('21944417|lukad6e_preflight|other|reu-aisocial|debug|CANCELLED|0|',False),
    ('21944417|lukad6e_preflight|ryreu|reu-aisocial|tier3|CANCELLED|0|',False),
    ('',False), ('one\ntwo',False)])
def test_pending_replacement_guard(tmp_path,monkeypatch,line,passed):
    calls=[]
    def run(cmd,**kwargs): calls.append(cmd); return SimpleNamespace(stdout=line)
    monkeypatch.setattr(direct.subprocess,'run',run)
    monkeypatch.setattr(direct.getpass,'getuser',lambda:'ryreu')
    if passed: assert direct.cancelled_original()['elapsed_seconds']==0
    else:
        with pytest.raises(ValueError): direct.cancelled_original()
    assert len(calls)==1 and calls[0][0]=='sacct'


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
    out=capsys.readouterr().out; data=json.loads(out.split('D066_DIRECT_RESOURCES=')[1])
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
    teacher=with_content_hash(dict(spec['teacher'],spec=dict(path='/teacher/science/spec.json')))
    parent=dict(nodes=parent_nodes(),project_dir='/parent-source',root='/parent/science')
    calls=[]
    monkeypatch.setattr(direct,'cancelled_original',lambda:calls.append('cancelled'))
    monkeypatch.setattr(source,'current',lambda *a:spec['source_snapshot'])
    monkeypatch.setattr(source,'authenticate',lambda *a:(spec['anchor'],parent))
    monkeypatch.setattr(source,'teacher',lambda *a:teacher)
    monkeypatch.setattr(source,'reference',lambda *a:spec['reference'])
    monkeypatch.setattr(direct,'resources',lambda *a:spec['resources'])
    root=tmp_path/'new'
    value=direct.create(root,'/parent/spec.json','/teacher/science/spec.json','/reference/science/spec.json',
        '/project','f'*40,available_quota_gib=2)
    direct.check_spec(value)
    assert calls==['cancelled','cancelled']
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
    assert value['train_minutes']==360 and not value['new_D066_preflight_performed']
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
    teacher,record=teacher_fixture(tmp_path)
    val=cache('validation',coordinate='D066')
    probabilities=predict(teacher.to('cuda'),val,device='cuda',batch_size=128)
    atomic_publish_bytes(tmp_path/'pred.npz',deterministic_npz_bytes(dict(
        identities=val.identities,labels=val.labels,probabilities=probabilities)))
    ref=file_ref(tmp_path/'pred.npz'); record=with_content_hash(dict(record,files=dict(record['files'],predictions=ref)))
    gate=fixture_gate(tmp_path/'direct',record,c.artifact('REFERENCE',files=dict(predictions=ref)))
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
    monkeypatch.setattr(preflight,'run',lambda *a:pytest.fail('No preflight'))
    monkeypatch.setattr(preflight,'stress',lambda *a,**kw:pytest.fail('No stress rerun'))
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
    path=root/'scripts/queue_luka_d066_embedding_direct.sh'
    subprocess.run([bash,'-n',str(path)],check=True,capture_output=True)
    text=path.read_text()
    for required in ('PYTHONNOUSERSITE=1','${CONDA_PREFIX}/lib','CUBLAS_WORKSPACE_CONFIG=:4096:8',
        'create-direct','submit-direct','</dev/tty'):
        assert required in text
    for forbidden in ('scancel','scontrol','ORDER_PENDING',' auto --',' preflight --'):
        assert forbidden not in text
    for block in re.findall(r'```bash\n(.*?)\n```',(root/'docs/LUKA_D066_EMBEDDING_RUNBOOK.md').read_text(),re.S):
        subprocess.run([bash,'-n'],input=block,text=True,check=True,capture_output=True)

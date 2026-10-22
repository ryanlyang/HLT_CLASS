"""D066 targets and matched student math; real CUDA miniature, not SPORC admission."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import numpy as np
import pytest
import torch
from test_luka_attention import cache, parent_nodes
from test_luka_embedding_aux import tiny
from test_luka_attention_reference_archive import saved, rebind_training
from hlt_classification.data.cache_contracts import (
    with_content_hash, write_immutable_json, load_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.luka_embedding_aux.model import new_model
from hlt_classification.luka_embedding_aux import contracts as old
from hlt_classification.luka_d066_embedding import (
    contracts as c, targets, training, worker, campaign, preflight, submission, source)


@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def fixture_gate(root, teacher=None, reference=None, *, site_name='sporc_a100_debug'):
    anchor=old.artifact('ANCHOR',spec=dict(path='/parent/spec.json'),evidence=dict(
        prepared='a'*64,cache='b'*64,prepared_root='/prepared',raw_container='/raw',
        environment=dict(fixture=True),models={}))
    teacher=teacher or c.artifact('TEACHER',coordinate='D066',target_role='train',prepared='a'*64,files={})
    reference=reference or c.artifact('REFERENCE',spec=dict(path='/reference/spec.json'),files={})
    for value in (teacher,reference):
        value['parents']['anchor']=anchor['content_hash']
    teacher,reference=map(with_content_hash,(teacher,reference))
    snapshot=dict(content_hash='c'*64)
    return c.artifact('GATE',parents=dict(source=snapshot['content_hash'],anchor=anchor['content_hash'],
        teacher=teacher['content_hash'],reference=reference['content_hash']),root=str(root),project_dir='/project',
        source_commit='f'*40,source_snapshot=snapshot,anchor=anchor,teacher=teacher,reference=reference,
        available_quota_gib=2,**campaign.registration(dict(nodes=parent_nodes()),site_name))


def small_counts(monkeypatch, rows=22):
    actual=campaign.registration
    monkeypatch.setattr(campaign,'registration',lambda parent,site_name='sporc_a100_debug':{**actual(parent,site_name),
        'counts':dict(train=rows,validation=rows),'original50k_rows':rows})


def teacher_fixture(root):
    model=DelphesParticleTransformer()
    save_state(root/'teacher.pt',model.state_dict())
    record=c.artifact('TEACHER',coordinate='D066',target_role='train',prepared='a'*64,
        files=dict(checkpoint=file_ref(root/'teacher.pt')))
    return model,record


def bank_fixture(root):
    teacher,record=teacher_fixture(root)
    targets.export(root/'targets',teacher,cache(coordinate='D066'),teacher_record=record,device='cpu')
    tr=cache(coordinate='D000')
    return tr,targets.Bank(root/'targets',tr,teacher_record=record),record


def test_scientific_registration_and_hlt_only_inference():
    gate=fixture_gate(Path('/run')); campaign.validate_gate_record(gate)
    r=gate['recipe']; n=gate['node']
    assert r['teacher']=='LOGIT_D066' and r['target_coordinate']=='D066'
    assert r['auxiliary_coefficient']==5 and r['ce_coefficient']==1 and r['epochs']==30
    assert not any(r[k] for k in ('logit_kd','attention_kd','warm_start'))
    assert n['coordinate']=='D000' and n['teacher'] is None and n['context_coordinate'] is None
    assert n['initialization_seed']==parent_nodes()[0]['initialization_seed']
    assert gate['execution_site']['partition']=='debug' and gate['memory_mb']==90000
    assert old.recipe()['teacher']=='OFFLINE' and old.recipe()['auxiliary_coefficient']==.3
    with pytest.raises(ValueError): c.validate(old.recipe(),'RECIPE')
    bad=deepcopy(gate); bad['node']['auxiliary_weight']=10
    with pytest.raises(ValueError): campaign.validate_gate_record(with_content_hash(bad))


def test_train_only_bank_normalization_and_teacher_binding(tmp_path,monkeypatch):
    tiny(monkeypatch); tr,bank,record=bank_fixture(tmp_path)
    assert bank.targets.shape==(22,128) and bank.targets.dtype==np.float32
    np.testing.assert_allclose(bank.targets.mean(0),0,atol=1e-6)
    assert np.max(bank.targets.std(0))<=1.00001
    for wrong in (cache('validation',coordinate='D066'),cache(coordinate='OFFLINE'),cache(coordinate='D000')):
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
    tiny(monkeypatch); small_counts(monkeypatch)
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
    val=cache('validation',coordinate='D066')
    arrays=dict(identities=val.identities,labels=val.labels,probabilities=predict(teacher,val,device='cpu',batch_size=128))
    atomic_publish_bytes(tmp_path/'pred.npz',deterministic_npz_bytes(arrays))
    record['files']['predictions']=file_ref(tmp_path/'pred.npz'); record=with_content_hash(record)
    proof=preflight.teacher_readback(teacher,val,record,'cpu')
    assert proof['native_probability_parity'] and not proof['normalization_used']
    bad=deepcopy(val); bad.identities=val.identities[::-1].copy()
    with pytest.raises(ValueError): preflight.teacher_readback(teacher,bad,record,'cpu')
    wrong=DelphesParticleTransformer()
    with pytest.raises(AssertionError): preflight.teacher_readback(wrong,val,record,'cpu')


def test_additive_source_required_and_unchanged_donors(monkeypatch):
    project=Path(__file__).resolve().parents[1]
    assert all((project/p).is_file() for p in source.REQUIRED)
    monkeypatch.setattr(source,'_source',lambda *a:dict(content_hash='a'*64))
    def git(cmd,**kw):
        return SimpleNamespace(stdout='\n'.join(source.REQUIRED) if 'ls-files' in cmd else
                               'src/hlt_classification/luka_embedding_aux/training.py\n')
    monkeypatch.setattr(source.subprocess,'run',git)
    with pytest.raises(ValueError,match='additive'): source.current(project,'f'*40)


def test_exact_submission_and_ambiguous_retry(tmp_path,monkeypatch):
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    gate=fixture_gate(tmp_path)
    monkeypatch.setattr(campaign,'check_gate',lambda _:None)
    monkeypatch.setattr(submission,'check_site',lambda _:None)
    dry=submission.submit(gate); cmd=dry['commands'][0]['command']
    assert '--partition=debug' in cmd and '--time=08:00:00' in cmd and dry['jobs']==1
    assert '--no-requeue' in cmd and '--export=NONE' in cmd
    calls=[]
    def uncertain(*a,**kw): calls.append(a); raise RuntimeError('lost response')
    monkeypatch.setattr(exact.subprocess,'run',uncertain)
    with pytest.raises(PermissionError): submission.submit(gate,execute=True,plan_hash='bad',authorization=c.AUTHORIZATION)
    with pytest.raises(RuntimeError): submission.submit(gate,execute=True,plan_hash=dry['content_hash'],authorization=c.AUTHORIZATION)
    with pytest.raises((PermissionError,FileExistsError,ValueError)):
        submission.submit(gate,execute=True,plan_hash=dry['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==1


@pytest.mark.parametrize('site_name',campaign.SITES)
def test_complete_real_cuda_preflight_and_science(tmp_path,monkeypatch,capsys,site_name):
    if not torch.cuda.is_available(): pytest.skip('Real CUDA required')
    # No architecture/model math substitutes: installed native ParT plus real
    # auxiliary head. Only population, OS sampler and remote scheduler/source are fixtures.
    small_counts(monkeypatch)
    teacher,record=teacher_fixture(tmp_path)
    val_teacher=cache('validation',coordinate='D066')
    probabilities=predict(teacher.to('cuda'),val_teacher,device='cuda',batch_size=128)
    arrays=dict(identities=val_teacher.identities,labels=val_teacher.labels,probabilities=probabilities)
    atomic_publish_bytes(tmp_path/'pred.npz',deterministic_npz_bytes(arrays))
    ref=file_ref(tmp_path/'pred.npz'); record['files']['predictions']=ref
    record=with_content_hash(record)
    root=tmp_path/'run'; root.mkdir()
    gate=fixture_gate(root,record,c.artifact('REFERENCE',files=dict(predictions=ref)),site_name=site_name)
    gate['anchor']['evidence']['models']={n:dict(predictions=ref) for n in ('M0HLT','OFFLINE')}
    gate['anchor']=with_content_hash(gate['anchor'])
    gate['parents']['anchor']=gate['anchor']['content_hash']
    for key in ('teacher','reference'):
        gate[key]['parents']['anchor']=gate['anchor']['content_hash']; gate[key]=with_content_hash(gate[key])
        gate['parents'][key]=gate[key]['content_hash']
    gate=with_content_hash(gate); write_immutable_json(root/'gate_spec.json',gate)
    meta=({},dict(summaries={r:dict(D000=dict(maximum=4)) for r in ('train','validation')}),{}, {})
    monkeypatch.setattr(campaign,'check_gate',lambda _:None)
    monkeypatch.setattr(worker,'execution',lambda *a:'77')
    monkeypatch.setattr(worker,'data',lambda g,r,co,meta:cache(r,coordinate=co))
    monkeypatch.setattr(preflight,'installed_environment',lambda:gate['anchor']['evidence']['environment'])
    monkeypatch.setattr(source,'current',lambda *a:gate['source_snapshot'])
    monkeypatch.setattr(preflight.regular_cache,'metadata',lambda _:meta)
    monkeypatch.setattr(worker.alljets_data,'comparison_indices',lambda *a:np.arange(22,dtype=np.int64))
    def memory(_):
        return dict(rss_bytes=2**30,peak_rss_bytes=2**30,gpu_peak_bytes=int(torch.cuda.max_memory_allocated()))
    monkeypatch.setattr(preflight,'memory_snapshot',memory); monkeypatch.setattr(worker,'memory_snapshot',memory)
    report=preflight.run(gate)
    assert report['technical_checks_passed'] and report['checkpoint_probability_roundtrip']
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:dict(preflight='77'))
    monkeypatch.setattr(campaign,'preflight_accounting',lambda _:None)
    spec=campaign.science(gate); result=worker.run(spec)
    assert len(result['rows'])==5 and not result['final_test_accessed']
    assert worker.results(spec)==result
    worker.print_results(result); assert 'AUC rec.%' in capsys.readouterr().out
    monkeypatch.setattr(torch,'load',lambda *a,**kw:pytest.fail('No inference while reporting'))
    assert worker.results(spec)==result
    bad=deepcopy(report); bad['outputs']=[r for r in bad['outputs'] if r['path']!='preflight/teacher_readback.json']
    with pytest.raises(ValueError,match='coverage'): preflight.validate_report(gate,with_content_hash(bad))
    bad=with_content_hash(dict(report,technical_checks_passed=False))
    with pytest.raises(ValueError,match='acceptance'): preflight.validate_report(gate,bad)
    (root/'science/fit/validation_predictions.npz').write_bytes(b'bad')
    with pytest.raises(ValueError): worker.results(spec)


def test_shell_cli_and_runbook_syntax():
    import shutil
    import re
    project=Path(__file__).resolve().parents[1]
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    executable=str(bash) if bash.is_file() else shutil.which('bash')
    if executable is None: pytest.skip('Bash unavailable')
    for name in ('scripts/queue_luka_d066_embedding.sh','sbatch/run_luka_d066_embedding.sh'):
        subprocess.run([executable,'-n',name],cwd=project,check=True,capture_output=True)
        text=(project/name).read_text()
        for token in ('PYTHONNOUSERSITE=1','PYTHONDONTWRITEBYTECODE=1','${CONDA_PREFIX}/lib','CUBLAS_WORKSPACE_CONFIG=:4096:8'):
            assert token in text
    text=(project/'docs/LUKA_D066_EMBEDDING_RUNBOOK.md').read_text()
    for block in re.findall(r'```bash\n(.*?)\n```',text,re.S):
        subprocess.run([executable,'-n'],input=block,text=True,cwd=project,check=True,capture_output=True)
    for link in re.findall(r'\]\(([^)]+)\)',text):
        if not link.startswith('https:'): assert (project/'docs'/link).resolve().is_file()


def test_original_archive_program_requires_real_checkpoint_binding(saved,monkeypatch,capsys):
    import sys
    # Execute the actual original-source replay program with retained archive
    # fixtures. Source checkout checks only are replaced by the donor fixture.
    root=saved.root; checkpoint=root/'train_LOGIT_D066/selected.pt'
    checkpoint.write_bytes(b'checkpoint payload fixture')
    receipt=load_json(checkpoint.parent/'receipt.json')
    receipt['outputs']=[file_ref(checkpoint,root=root) if r['path'].endswith('/selected.pt') else r
                        for r in receipt['outputs']]
    (checkpoint.parent/'receipt.json').write_text(json.dumps(with_content_hash(receipt)))
    rebind_training(saved,'LOGIT_D066')
    # The old anchor omits predictions; the newer CE anchor adds that field.
    models={n:dict(saved.anchor['evidence']['models'][n],predictions=dict(path='extra-bound-reference'))
            for n in ('M0HLT','OFFLINE')}
    anchor=old.artifact('ANCHOR',spec=saved.anchor['spec'],evidence=dict(
        prepared='b'*64,models=models))
    def replay(path,project,program,marker,*,extra):
        monkeypatch.setattr(sys,'argv',['replay',str(path),str(extra[0])])
        exec(compile(program,'original-replay','exec'),{})
        lines=[line[len(marker):] for line in capsys.readouterr().out.splitlines() if line.startswith(marker)]
        assert len(lines)==1
        return file_ref(path),json.loads(lines[0])
    monkeypatch.setattr(source,'replay',replay)
    imported=source.teacher(root/'campaign_spec.json',anchor)
    assert imported['checkpoint_payload_revalidated'] and not imported['legacy_attention_banks_reused']
    assert imported['node']['node_id']=='LOGIT_D066' and imported['files']['checkpoint']==file_ref(checkpoint)
    checkpoint.write_bytes(b'wrong weights')
    with pytest.raises(ValueError): source.teacher(root/'campaign_spec.json',anchor)


def test_matched_reference_import_accepts_only_completed_exact_w5(tmp_path,monkeypatch,capsys):
    import sys
    from hlt_classification.luka_embedding_sweep import debug,contracts as sweep
    root=tmp_path/'reference'; root.mkdir()
    gate=fixture_gate(tmp_path/'run')
    original=dict(anchor=gate['anchor'],counts=c.COUNTS,nodes=sweep.nodes(dict(nodes=parent_nodes())))
    write_immutable_json(root/'gate.json',original)
    spec=dict(root=str(root),gate_path=str(root/'gate.json'),source_commit=source.REFERENCE_COMMIT,
              project_dir='/original',source_snapshot={},content_hash='d'*64)
    write_immutable_json(root/'campaign_spec.json',spec)
    for path in ('receipt.json','results.json','fit/training_report.json','fit/validation_predictions.npz'):
        write_immutable_json(root/'w5'/path,dict(fixture=True))
    monkeypatch.setattr(source,'validate_source_snapshot',lambda *a,**kw:None)
    monkeypatch.setattr(debug,'results',lambda _:dict(pending=[]))
    def replay(path,project,program,marker):
        monkeypatch.setattr(sys,'argv',['replay',str(path)])
        exec(compile(program,'original-reference','exec'),{})
        lines=[line[len(marker):] for line in capsys.readouterr().out.splitlines() if line.startswith(marker)]
        assert len(lines)==1
        return file_ref(path),json.loads(lines[0])
    monkeypatch.setattr(source,'replay',replay)
    result=source.reference(root/'campaign_spec.json',gate['anchor'],gate['node'])
    assert result['node']['auxiliary_weight']==5
    monkeypatch.setattr(debug,'results',lambda _:dict(pending=['w5']))
    with pytest.raises(ValueError,match='not complete'): source.reference(root/'campaign_spec.json',gate['anchor'],gate['node'])


def test_results_printer_recovery_already_percentage(capsys):
    row=dict(model='HLT_D066_EMBED_W5')
    for role in ('validation','original50k'):
        row[role]=dict(accuracy=.8,macro_ovr_auc=.95)
        row[role+'_recovery']=dict(macro_ovr_auc=50.)
        row[role+'_delta_vs_ce']=.01; row[role+'_delta_vs_offline_w5']=.001
    worker.print_results(dict(rows=[row]))
    text=capsys.readouterr().out
    assert '50.0' in text and '5000.0' not in text


def test_d066_resource_evidence_rejects_excesses():
    gate=fixture_gate(Path('/run'))
    measurement=dict(gpu=dict(total_memory_bytes=40*2**30),peak_rss_bytes=4*2**30,gpu_peak_bytes=2*2**30,
        timings=dict(authentication=10.,load_HLT=10.,train_epoch=60.,validation_epoch=15.,checkpoint_reporting=10.))
    assert 30<=preflight.admission(gate,measurement)<=1440
    for key,value in [('train_epoch',float('nan')),('train_epoch',100000.)]:
        bad=deepcopy(measurement); bad['timings'][key]=value
        with pytest.raises(ValueError): preflight.admission(gate,bad)
    for key,value in [('peak_rss_bytes',90000*1024**2),('gpu_peak_bytes',40*2**30)]:
        with pytest.raises(ValueError): preflight.admission(gate,dict(measurement,**{key:value}))


def test_tier3_registration_changes_only_hashed_execution_site():
    parent=dict(nodes=parent_nodes())
    legacy=campaign.registration(parent)
    assert legacy==campaign.registration(parent,'sporc_a100_debug')
    tier3=campaign.registration(parent,'sporc_a100')
    assert {k:v for k,v in legacy.items() if k!='execution_site'}=={
        k:v for k,v in tier3.items() if k!='execution_site'}
    old_gate=fixture_gate(Path('/run'))
    gate=fixture_gate(Path('/run'),site_name='sporc_a100')
    assert gate['content_hash']!=old_gate['content_hash'] and gate['recipe']==old_gate['recipe']
    campaign.validate_gate_record(old_gate); campaign.validate_gate_record(gate)
    bad=deepcopy(gate); bad['execution_site']['partition']='debug'
    bad['execution_site']=with_content_hash(bad['execution_site'])
    with pytest.raises(ValueError,match='resources differ'):
        campaign.validate_gate_record(with_content_hash(bad))
    for site in ('oscar_l40s','tigris_gh200','debug,tier3','tier3',None):
        with pytest.raises(ValueError,match='SPORC A100'):
            campaign.registration(parent,site)


def test_create_tier3_roundtrip_and_cli_option(tmp_path,monkeypatch):
    import sys
    gate=fixture_gate(tmp_path/'fixture')
    parent=dict(nodes=parent_nodes(),project_dir='/parent-source',root='/parent/science')
    monkeypatch.setattr(source,'current',lambda *a:gate['source_snapshot'])
    monkeypatch.setattr(source,'authenticate',lambda *a:(gate['anchor'],parent))
    monkeypatch.setattr(source,'teacher',lambda *a:gate['teacher'])
    monkeypatch.setattr(source,'reference',lambda *a:gate['reference'])
    root=tmp_path/'tier3'
    value=campaign.create(root,'/parent/spec.json','/teacher/science/spec.json',
        '/reference/science/spec.json','/project','f'*40,available_quota_gib=2,site_name='sporc_a100')
    assert value['execution_site']['partition']=='tier3'
    assert load_json(root/'gate_spec.json')==value
    campaign.validate_gate_record(value)
    command=submission.render(value)['commands'][0]['command']
    assert '--partition=tier3' in command and '--time=08:00:00' in command
    assert '--partition=debug' not in command
    help_text=subprocess.run([sys.executable,'-B','scripts/luka_d066_embedding.py','create','--help'],
        check=True,capture_output=True,text=True).stdout
    assert '--site {sporc_a100_debug,sporc_a100}' in help_text


def test_execution_still_rejects_live_partition_change(monkeypatch):
    from hlt_classification.jetclass2_delphes import execution
    for name,value in dict(SLURM_JOB_ID='21944417',SLURM_CPUS_PER_TASK='6',
            SLURM_MEM_PER_NODE='90000',SLURM_CLUSTER_NAME='sporc').items():
        monkeypatch.setenv(name,value)
    fields='Account=reu-aisocial Partition=tier3 QOS=qos_tier3 NumNodes=1 NumTasks=1 NumCPUs=6'
    monkeypatch.setattr(execution.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=fields))
    with pytest.raises(PermissionError,match='registered single-task'):
        execution.allocation(execution.execution_site('sporc_a100_debug'))


@pytest.mark.parametrize('change,value,ok',[
    (None,None,True),(5,'CANCELLED by 2914834',True),(5,'COMPLETED',False),
    (5,'PENDING',False),(5,'RUNNING',False),(6,'1',False),(0,'21940433',False),
    (1,'lukad6e_fit',False),(2,'other',False),(3,'other',False),(4,'tier3',False),
    (0,'',False),(0,'21944417\n21944417',False),(6,'0|extra',False),
])
def test_replacement_guard_requires_exact_never_started_cancelled_job(change,value,ok):
    import os
    import shutil
    project=Path(__file__).resolve().parents[1]
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    executable=str(bash) if bash.is_file() else shutil.which('bash')
    if not executable: pytest.skip('Bash unavailable')
    source_text=(project/'scripts/queue_luka_d066_embedding.sh').read_text()
    block=source_text.split('# Explicit replacement guard:',1)[1].split('\n',1)[1]
    block=block.split('# End replacement guard.',1)[0]
    fields=['21944417','lukad6e_preflight','ryreu','reu-aisocial','debug','CANCELLED','0']
    if change is not None: fields[change]=value
    prefix='set -euo pipefail\nPARTITION=tier3\nREPLACES_PREFLIGHT_JOB=21944417\nsacct() { printf "%s\\n" "$MOCK_ACCOUNTING"; }\n'
    result=subprocess.run([executable,'-c',prefix+block],capture_output=True,text=True,
        env=dict(os.environ,MOCK_ACCOUNTING='|'.join(fields)))
    assert (result.returncode==0)==ok,result.stderr

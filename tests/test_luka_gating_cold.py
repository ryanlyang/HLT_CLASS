"""Cold scientific controls, production-worker miniature and scratch lifecycle."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import ast
import numpy as np
import pytest
import torch
from test_luka_attention import cache, tiny_config, parent_nodes
from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_gating import banks
from hlt_classification.luka_gating.model import new_model
from hlt_classification.luka_gating_cold import contracts as c, campaign, worker, training, storage, preflight


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def gate_fixture(root):
    anchor=c.artifact('ANCHOR',spec=dict(path='/parent/spec.json'),evidence={})
    return c.artifact('GATE',parents=dict(source='b'*64,anchor=anchor['content_hash']),
        root=str(root),project_dir='/project',source_commit='a'*40,source_snapshot=dict(content_hash='b'*64),
        anchor=anchor,scratch_base='/tmp',available_quota_gib=4,**campaign.registration(dict(nodes=parent_nodes())))


def population(tmp_path,monkeypatch,device='cpu'):
    tiny_config(monkeypatch)
    parent=tmp_path/'parent'; parent.mkdir()
    def view(role,coordinate):
        data=cache(role,coordinate)
        for block in data.blocks:
            block.features[:,11]+={'U100':1.,'D066':.66,'D033':.33,'D000':0.}[coordinate]
        return data
    gate=gate_fixture(tmp_path)
    anchor=new_model(worker.anchor_node(gate))
    save_state(parent/'selected.pt',anchor.state_dict())
    val=view('validation','U100')
    metrics=evaluate_probabilities(val.labels,banks.predict(anchor,val,device=device))
    write_immutable_json(parent/'report.json',dict(validation=metrics))
    write_immutable_json(parent/'diagnostics.json',dict(original_validation=metrics))
    write_immutable_json(parent/'receipt.json',dict(complete=True))
    refs=dict(checkpoint=file_ref(parent/'selected.pt'),report=file_ref(parent/'report.json'),
        diagnostics=file_ref(parent/'diagnostics.json'),receipt=file_ref(parent/'receipt.json'))
    anchor=c.artifact('ANCHOR',spec=dict(path='/parent/spec.json'),evidence=dict(environment={},prepared='a'*64,
        prepared_root=str(parent),models={n:refs for n in ('M0HLT','OFFLINE','DIRECT_D000','COARSE_U100')}))
    gate=with_content_hash(dict(gate,anchor=anchor,parents=dict(source='b'*64,anchor=anchor['content_hash']),
                               counts=dict(train=22,validation=22)))
    write_immutable_json(tmp_path/'gate_spec.json',gate)
    root=tmp_path/'science'; root.mkdir()
    spec=c.artifact('SPEC',root=str(root),project_dir='/project',source_commit='a'*40)
    monkeypatch.setattr(campaign,'check_spec',lambda s:(gate,{}))
    monkeypatch.setattr(campaign,'check_gate',lambda s:None)
    monkeypatch.setattr(worker,'execution',lambda *a,**kw:'123')
    monkeypatch.setattr(worker,'data',lambda g,r,co:view(r,co))
    monkeypatch.setattr(worker,'original_indices',lambda g,val:np.arange(len(val)))
    from hlt_classification.luka_fullsim import preflight_v2
    sample=lambda t:dict(peak_rss_bytes=2**30,rss_bytes=2**30,gpu_peak_bytes=t.cuda.max_memory_allocated())
    monkeypatch.setattr(preflight_v2,'memory_snapshot',sample)
    monkeypatch.setattr(preflight,'memory_snapshot',sample)
    monkeypatch.setattr(preflight,'metadata',lambda p:(None,dict(summaries={
        r:{co:dict(maximum=4) for co in ('D066','D033','D000')} for r in ('train','validation')}),{},{}))
    local=tmp_path/'scratch'; local.mkdir()
    actual=storage.scratch
    monkeypatch.setattr(storage,'scratch',lambda g,j:actual(dict(g,scratch_base=str(local)),j))
    return gate,spec


def test_registration_is_cold_complete_and_tier3(tmp_path):
    gate=gate_fixture(tmp_path)
    assert len(gate['nodes'])==12 and len(gate['tasks'])==13
    assert gate['execution_site']['partition']=='tier3'
    for n in gate['nodes']:
        assert n['warm_start'] is None and n['freeze_epochs']==0
        assert n['gate_weight'] in (0.,.1)
        if n['deployable']: assert n['coordinate']=='D000' and n['context_coordinate'] is None
    assert not any(n['node_id']=='REPLACE_STAGED' for n in gate['nodes'])
    for branch in ('LOGIT','SELF','PRIV'):
        chain=[n for n in gate['nodes'] if n['branch']==branch]
        assert [n['teacher'] for n in chain]==['ANCHOR',branch+'_D066',branch+'_D033']
    assert c.recipe()['base']['batch_size']==128


def test_initialization_is_seeded_and_teacher_independent(tmp_path,monkeypatch):
    tiny_config(monkeypatch); gate=gate_fixture(tmp_path)
    rows=preflight.initialization_checks(gate)
    assert len(rows)==12 and all(r['repeatable_after_mutation'] for r in rows)
    for n in gate['nodes']:
        assert training.initial_digest(training.new_student(n))==next(r['initialization_sha256'] for r in rows if r['node_id']==n['node_id'])
        with pytest.raises(TypeError): training.new_student(n,state={})
        with pytest.raises(ValueError): training.new_student(dict(n,warm_start='ANCHOR'))
        with pytest.raises(ValueError): training.new_student(dict(n,freeze_epochs=3))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='local CUDA miniature; remote gate still required')
def test_all_12_cold_science_fits_and_post_retirement_results(tmp_path,monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    gate,spec=population(tmp_path,monkeypatch,'cuda')
    initials={r['node_id']:r for r in preflight.initialization_checks(gate)}
    for task in gate['tasks']:
        result=worker.run(spec,task['task_id'])
        assert result==worker.completed(spec,gate,task['task_id'])
    rows=worker.result_rows(spec,gate)
    assert len(rows)==16 and all(r['validation'] is not None for r in rows)
    for node in gate['nodes']:
        root=Path(spec['root'])/('train_'+node['node_id'])
        r=load_json(root/'training_report.json')
        assert r['passes']==30 and r['scientific_fit'] and r['selected_weights_restored']
        assert r['initialization_sha256']==initials[node['node_id']]['initialization_sha256']
        assert not any(h['backbone_frozen'] for h in r['validation_history'])
        assert (root/'validation_predictions.npz').is_file()
        intent=load_json(root/'retired_targets/intent.json')
        assert all(not (Path(intent['base'])/f['path']).exists() for f in intent['files'])
        if node['gate_weight']: assert max(h['losses']['gate'] for h in r['validation_history'])>0
    assert not list((tmp_path/'scratch').rglob('population.npz'))
    assert not list((tmp_path/'scratch').rglob('*.npy'))
    # An absent scratch mount must not prevent comparing durable saved results.
    for p in (tmp_path/'scratch').glob('*/owner.json'): p.unlink()
    assert worker.result_rows(spec,gate)==rows
    with (Path(spec['root'])/'train_PRIV_D033/selected.pt').open('ab') as f: f.write(b'corrupt')
    with pytest.raises(ValueError): worker.completed(spec,gate,'train_REPLACE_GATE')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='installed parity and production cold preflight')
def test_all_preflight_tasks_resources_and_tamper_guards(tmp_path,monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    gate,_=population(tmp_path,monkeypatch,'cuda')
    monkeypatch.setattr(preflight,'installed_environment',lambda:{})
    report=preflight.run(gate); preflight.validate_report(gate,report)
    assert len(report['measurements'])==12
    assert sum(len(m['stress']) for m in report['measurements'])==24
    assert len(report['fit_minutes'])==12 and not report['scientific_fit']
    spec=c.artifact('SPEC',parents=dict(gate=gate['content_hash'],preflight=report['content_hash']),
        root=str(tmp_path/'science'),gate_path=str(tmp_path/'gate_spec.json'),
        project_dir=gate['project_dir'],source_commit=gate['source_commit'],fit_minutes=report['fit_minutes'])
    write_immutable_json(tmp_path/'science/campaign_spec.json',spec)
    # Metadata/results reading needs neither live accounting nor target scratch.
    rows=worker.saved_results(spec)
    assert len(rows)==16 and all(r['validation'] is None for r in rows[4:])
    for key,value in [('technical_checks_passed',False),('scientific_fit',True),
            ('parities',report['parities'][:1]),('measurements',report['measurements'][:-1]),
            ('initialization',report['initialization'][:-1]),('fit_minutes',{}),('outputs',[])]:
        with pytest.raises((ValueError,MemoryError)):
            preflight.validate_report(gate,with_content_hash(dict(report,**{key:value})))
    for mode in ('stress','storage','initialization_sha256'):
        bad=deepcopy(report)
        if mode=='stress': bad['measurements'][0][mode]=[]
        elif mode=='storage': bad['measurements'][0][mode]['scratch_required_bytes']=1
        else: bad['measurements'][0][mode]='e'*64
        with pytest.raises(ValueError): preflight.validate_report(gate,with_content_hash(bad))
    with pytest.raises(FileExistsError): preflight.run(gate)
    with (tmp_path/'preflight/train_SELF_REDISTILL/selected.pt').open('ab') as f: f.write(b'bad')
    with pytest.raises(ValueError): preflight.validate_report(gate,report)


def test_per_fit_budget_has_no_quality_admission():
    rows=[dict(node_id='native',setup_seconds=120,epoch_seconds=60,other_seconds=60),
          dict(node_id='gate',setup_seconds=240,epoch_seconds=300,other_seconds=60)]
    assert preflight.budgets(rows,60)==dict(native=60,gate=273)
    assert preflight.budgets([dict(m,auc=0.) for m in rows],60)==dict(native=60,gate=273)
    for value in (float('nan'),-1,100000):
        bad=deepcopy(rows); bad[0]['epoch_seconds']=value
        with pytest.raises(ValueError): preflight.budgets(bad)
    with pytest.raises(ValueError): preflight.budgets(rows+rows)
    with pytest.raises(MemoryError): preflight.memory_ok(dict(memory_mb=1),dict(peak_rss_bytes=2**30,gpu_peak_bytes=1),dict(total_memory_bytes=10))


@pytest.mark.parametrize('name',['../escape','/absolute','x\\escape','x:escape','owner.json'])
def test_retirement_cannot_escape_or_remove_owner(tmp_path,name):
    owner='a'*64; root=storage.create_owned(tmp_path/'owned',owner)
    consumer=tmp_path/'consumer.json'; write_immutable_json(consumer,dict(valid=True))
    with pytest.raises((PermissionError,ValueError)):
        storage.retire(root,[dict(path=name,sha256='b'*64,bytes=1)],owner=owner,consumer=file_ref(consumer),output=tmp_path/'retirement')
    assert (root/'owner.json').is_file()


def test_storage_all_inventory_validated_before_removal_and_consumer_required(tmp_path):
    root=storage.create_owned(tmp_path/'owned','a'*64)
    write_immutable_json(root/'a.json',dict(value=1)); write_immutable_json(root/'b.json',dict(value=2))
    refs=[file_ref(root/n,root=root) for n in ('a.json','b.json')]
    consumer=tmp_path/'consumer.json'; write_immutable_json(consumer,dict(valid=True))
    bad=deepcopy(refs); bad[1]['sha256']='b'*64
    with pytest.raises(ValueError): storage.retire(root,bad,owner='a'*64,consumer=file_ref(consumer),output=tmp_path/'bad')
    assert (root/'a.json').is_file() and not (tmp_path/'bad').exists()
    receipt=storage.retire(root,refs,owner='a'*64,consumer=file_ref(consumer),output=tmp_path/'done')
    assert receipt['deleted_files']==2 and (root/'owner.json').is_file()
    storage.validate_retirement(tmp_path/'done','a'*64)
    write_immutable_json(root/'a.json',dict(value=1))
    with pytest.raises(ValueError): storage.validate_retirement(tmp_path/'done','a'*64)


def test_storage_budget_and_no_fallback(tmp_path,monkeypatch):
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda p:SimpleNamespace(free=20*2**30))
    result=storage.budget([3,4],tmp_path,tmp_path,2)
    assert result['factor_bank_bytes']==252
    assert result['persistent_required_bytes']==2*2**30
    with pytest.raises(OSError): storage.budget([3,4],tmp_path,tmp_path,1)
    with pytest.raises(PermissionError): storage.scratch(dict(scratch_base=str(tmp_path/'absent')),'x')


def test_failed_selected_validation_never_retires_scratch(tmp_path,monkeypatch):
    gate,spec=population(tmp_path,monkeypatch)
    def stop(*args,**kwargs): raise ValueError('selected output authentication failed')
    monkeypatch.setattr(worker,'validate_selected',stop)
    with pytest.raises(ValueError,match='selected output authentication'):
        worker.execute_task(spec,gate,'train_LOGIT_D066',device='cpu',technical=True)
    assert list((tmp_path/'scratch').rglob('population.npz'))
    assert not (Path(spec['root'])/'train_LOGIT_D066/retired_targets').exists()
    assert not (Path(spec['root'])/'train_LOGIT_D066/receipt.json').exists()


def test_campaign_scope_storage_and_source_guards(tmp_path,monkeypatch):
    from hlt_classification.luka_gating_cold import source
    parent=dict(nodes=parent_nodes(),root=str(tmp_path/'old/science'),project_dir=str(tmp_path/'old_source'))
    anchor=c.artifact('ANCHOR',spec=dict(path='/parent/spec.json'),evidence=dict(
        prepared_root=str(tmp_path/'prepared'),raw_container=str(tmp_path/'raw')))
    monkeypatch.setattr(source,'current',lambda *a:dict(content_hash='b'*64))
    monkeypatch.setattr(source,'authenticate',lambda *a:(anchor,parent))
    project=tmp_path/'source'
    for root in (tmp_path,tmp_path/'prepared/new',tmp_path/'raw/new',project/'new',tmp_path/'old/new'):
        with pytest.raises(PermissionError): campaign.create(root,'parent',project,'a'*40,available_quota_gib=4)
    with pytest.raises(ValueError): campaign.create(tmp_path/'new','parent',project,'a'*40,available_quota_gib=1)
    with pytest.raises(PermissionError): campaign.create(tmp_path/'new','parent',project,'a'*40,available_quota_gib=4,scratch_base='/home')
    gate=campaign.create(tmp_path/'new','parent',project,'a'*40,available_quota_gib=4)
    campaign.check_gate(gate)
    with pytest.raises(FileExistsError): campaign.create(tmp_path/'new','parent',project,'a'*40,available_quota_gib=4)
    with pytest.raises(ValueError): campaign.read_gate(with_content_hash(dict(gate,recipe={})))
    monkeypatch.setattr(source,'current',lambda *a:dict(content_hash='c'*64))
    with pytest.raises(ValueError): campaign.check_gate(gate)


def test_retirement_hardlink_and_unknown_bank_files(tmp_path):
    import os
    root=storage.create_owned(tmp_path/'owned','a'*64)
    write_immutable_json(root/'payload.json',dict(v=1))
    os.link(root/'payload.json',tmp_path/'alias.json')
    consumer=tmp_path/'proof.json'; write_immutable_json(consumer,dict(valid=True))
    with pytest.raises(PermissionError): storage.retire(root,[file_ref(root/'payload.json',root=root)],
        owner='a'*64,consumer=file_ref(consumer),output=tmp_path/'retirement')
    assert (root/'payload.json').is_file()
    with pytest.raises(ValueError): storage.bank_refs(root,dict(population=file_ref(root/'payload.json',root=root),shards=[]))


def test_shell_and_runbook_bash_syntax():
    import re
    import shutil
    import subprocess
    root=Path(__file__).resolve().parents[1]
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    executable=str(bash) if bash.is_file() else shutil.which('bash')
    if not executable: pytest.skip('bash not installed')
    scripts=[(root/f).read_text() for f in ('scripts/queue_luka_gating_cold.sh','sbatch/run_luka_gating_cold.sh')]
    scripts+=re.findall(r'```bash\n(.*?)```',(root/'docs/LUKA_COLD_GATE_TIER3_RUNBOOK.md').read_text(),re.S)
    for script in scripts:
        subprocess.run([executable,'-n'],input=script,text=True,check=True,capture_output=True)


def test_source_checker_freezes_existing_modules(monkeypatch):
    from hlt_classification.luka_gating_cold import source
    project=Path(__file__).resolve().parents[1]
    monkeypatch.setattr(source,'_source',lambda *a:dict(content_hash='a'*64))
    files=set(source.REQUIRED); changes=set(source.ADDITIONS)
    def git(args,**kwargs):
        return SimpleNamespace(stdout='\n'.join(sorted(files if 'ls-files' in args else changes)))
    monkeypatch.setattr(source.subprocess,'run',git)
    assert source.current(project,'b'*40)['content_hash']=='a'*64
    changes.add('src/hlt_classification/luka_gating/model.py')
    with pytest.raises(ValueError,match='Existing scientific source'): source.current(project,'b'*40)
    changes.clear(); files.remove('tests/test_luka_gating_cold.py')
    with pytest.raises(ValueError,match='Commit complete'): source.current(project,'b'*40)


def test_source_required_files_and_unchanged_donors():
    from hlt_classification.luka_gating_cold import source
    root=Path(__file__).resolve().parents[1]
    assert all((root/p).is_file() for p in source.REQUIRED)
    for name in source.ADDITIONS: ast.parse((root/name).read_text())
    worker_script=(root/'sbatch/run_luka_gating_cold.sh').read_text()
    assert '${PROJECT_DIR}/sbatch/common.sh' in worker_script and 'PYTHONNOUSERSITE=1' in worker_script
    assert 'CUBLAS_WORKSPACE_CONFIG=:4096:8' in worker_script
    assert '--scratch-base' in (root/'scripts/queue_luka_gating_cold.sh').read_text()

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import inspect
import numpy as np
import pytest
import torch

from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    write_immutable_json, with_content_hash, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_fullsim.regular_campaign import nodes as parent_nodes
from hlt_classification.luka_attention import banks, model
from hlt_classification.luka_particle_swap import contracts as c, worker as w, retention as r
from hlt_classification.luka_particle_swap import campaign, preflight, submission, training
from test_luka_attention import cache, tiny_config


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture(tmp_path, monkeypatch, arm='logit', *, tiny=True):
    if tiny:
        tiny_config(monkeypatch)
    gate_root = tmp_path/'campaign'
    gate_root.mkdir()
    models = c.nodes(parent_nodes(), arm)
    state = model.new_model(models[0]).state_dict()
    save_state(tmp_path/'anchor.pt', state)
    write_immutable_json(tmp_path/'anchor.json', {'scientific':True})
    gate = c.artifact('GATE', root=str(gate_root), project_dir=str(tmp_path/'project'), source_commit='a'*40,
        anchor=dict(content_hash='b'*64, evidence=dict(prepared='a'*64, cache='c'*64,
            environment={'test':True}, models={'COARSE_U100':dict(checkpoint=file_ref(tmp_path/'anchor.pt'), report=file_ref(tmp_path/'anchor.json'))})),
        available_quota_gib=1, scratch_base=str(tmp_path), **campaign.registration(dict(nodes=parent_nodes()),arm))
    gate = with_content_hash(dict(gate, counts=dict(train=22,validation=22), input_identity=c.input_identity(gate['anchor'])))
    metrics = evaluate_probabilities(np.arange(22)%11, np.full((22,11),1/11,np.float32))
    refs = c.artifact('REFERENCES', parents=dict(anchor=gate['anchor']['content_hash']),
        rows=[dict(node_id=name,validation=metrics,original_validation=metrics) for name in
              ('M0HLT','OFFLINE','DIRECT_D000','COARSE_U100','LOGIT_D000','ATTN_D000')])
    write_immutable_json(gate_root/'references.json', refs)
    caches = {(role,co):cache(role,co) for role in ('train','validation') for co in w.data.COORDINATES}
    for value in caches.values(): value.foundation_sha256 = gate['input_identity']['content_hash']
    audit = c.artifact('INPUT_AUDIT', parents=dict(inputs=gate['input_identity']['content_hash']),
        roles={role:dict(views={co:dict(rows=22) for co in w.data.COORDINATES}) for role in ('train','validation')},
        endpoint_fingerprints_replayed=True, resident_bytes=sum(v.nbytes for v in caches.values()))
    bundle = dict(caches=caches,audit=audit,original_validation=np.ones(22,bool),construction_seconds=.1)
    monkeypatch.setattr(w.data, 'build', lambda *a,**kw:bundle)
    scratch = r.create_owned(tmp_path/'scratch', gate['content_hash'])
    log = SimpleNamespace(phase=lambda name: __import__('contextlib').nullcontext())
    return gate, scratch, log


@pytest.mark.parametrize('arm,coefficient',[('logit',0.),('attn',.1)])
def test_registered_graph_and_unchanged_old_recipe(arm,coefficient):
    from hlt_classification.luka_attention.contracts import recipe as old
    reg = campaign.registration(dict(nodes=parent_nodes()),arm)
    assert reg['execution_site']['partition']=='debug'
    assert reg['execution_site']['gres']=='gpu:a100:1'
    assert reg['recipe']['coefficient']==coefficient
    assert reg['recipe']['base']==old()['base'] and reg['recipe']['surface']==old()['surface']
    assert old()['coefficient']==.1
    for node in reg['nodes']:
        original = next(n for n in parent_nodes() if n['node_id']=='COARSE_'+node['coordinate'])
        assert node['initialization_seed']==original['initialization_seed']
        assert node['sampler_seed']==original['sampler_seed']
    assert [n['teacher'] for n in reg['nodes']]==['ANCHOR','SWAP_'+arm.upper()+'_D066','SWAP_'+arm.upper()+'_D033']
    assert [n['coordinate'] for n in reg['nodes']]==list(c.COORDINATES)
    for bad in (True,.3,1.,float('nan')):
        with pytest.raises(ValueError): c.recipe(bad)
    with pytest.raises(ValueError,match='schema version'):
        c.validate(with_content_hash(dict(c.recipe(coefficient),schema_version=2)),'RECIPE')


def test_training_kernel_only_changes_registration():
    from hlt_classification.luka_attention import training as old
    # Protect the numerics, sampler, optimizer and selector against accidental rewrites.
    original = inspect.getsource(old.train)
    revised = inspect.getsource(training.train)
    assert original[original.index('    if (train_cache.role'):original.index("    report = artifact")] == revised[
        revised.index('    if (train_cache.role'):revised.index("    report = artifact")]


@pytest.mark.parametrize('arm',['logit','attn'])
@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason='Local CUDA optional; remote A100 acceptance required'))])
def test_full_fixed30_chain_and_results_after_payload_deletion(tmp_path,monkeypatch,arm,device):
    gate,scratch,log=fixture(tmp_path,monkeypatch,arm)
    output=Path(gate['root'])/'ladder'
    write_immutable_json(Path(gate['root'])/'preflight/receipt.json', dict(facts=dict(input_audit=w.data.build(gate)['audit'])))
    facts=w.run_chain(gate,output,scratch,device=device,log=log)
    assert len(facts['measurements'])==3
    assert not list(scratch.rglob('*.pt')) and not list(scratch.rglob('*.npy')) and not list(scratch.rglob('*.npz'))
    assert (tmp_path/'anchor.pt').is_file()
    rows=w.result_rows(gate,output)
    assert len(rows)==9 and all(row['checkpoint_retired'] for row in rows[-3:])
    for node in gate['nodes']:
        directory=output/node['node_id']
        report=load_json(directory/'training_report.json')
        assert report['passes']==30 and report['scientific_fit'] is True
        assert report['validation_history'][-1]['learning_rate']==pytest.approx(1.5e-5)
        for epoch in report['validation_history']:
            assert epoch['losses']['weighted_attention']==pytest.approx(gate['coefficient']*epoch['losses']['attention'])
        p=load_npz_arrays(directory/'validation_predictions.npz')
        assert p['probabilities'].shape==(22,11) and p['original_validation'].all()
        r.validate_retirement(directory/'retired_teacher',gate['content_hash'])
        r.validate_retirement(directory/'retired_checkpoint',gate['content_hash'])
    # Source files/models/old experiments are not needed to display stored new predictions.
    (tmp_path/'anchor.pt').unlink()
    assert w.result_rows(gate,output)==rows
    payload=output/gate['nodes'][-1]['node_id']/'validation_predictions.npz'
    with payload.open('ab') as stream: stream.write(b'corruption')
    with pytest.raises(ValueError): w.result_rows(gate,output)


def owned_payload(tmp_path):
    base=r.create_owned(tmp_path/'owned','a'*64)
    atomic_publish_bytes(base/'payload.bin',b'target')
    consumer=tmp_path/'consumer.json'
    write_immutable_json(consumer,{'consumer':'completed'})
    return base,[file_ref(base/'payload.bin',root=base)],file_ref(consumer)


def test_retirement_requires_validated_consumer_and_all_payloads(tmp_path):
    base,refs,consumer=owned_payload(tmp_path)
    wrong=dict(consumer,sha256='0'*64)
    with pytest.raises(ValueError): r.retire(base,refs,owner='a'*64,consumer=wrong,output=tmp_path/'retire')
    assert (base/'payload.bin').is_file()
    with pytest.raises((ValueError,PermissionError)): r.retire(base,refs+[dict(refs[0],path='missing')],owner='a'*64,consumer=consumer,output=tmp_path/'retire')
    assert (base/'payload.bin').is_file()
    result=r.retire(base,refs,owner='a'*64,consumer=consumer,output=tmp_path/'retire')
    assert result['deleted_bytes']==6
    r.validate_retirement(tmp_path/'retire','a'*64)
    atomic_publish_bytes(base/'payload.bin',b'target')
    with pytest.raises(ValueError,match='reappeared'):r.validate_retirement(tmp_path/'retire','a'*64)


@pytest.mark.parametrize('escape',['../other','/absolute','x/../../other','owner.json'])
def test_retirement_never_deletes_outside_owned_payloads(tmp_path,escape):
    base,refs,consumer=owned_payload(tmp_path)
    with pytest.raises((PermissionError,ValueError,FileNotFoundError)):
        r.retire(base,[dict(refs[0],path=escape)],owner='a'*64,consumer=consumer,output=tmp_path/'retire')
    assert (base/'payload.bin').is_file() and (base/'owner.json').is_file()


def test_retirement_rejects_symlink_and_unknown_bank_files(tmp_path):
    base,refs,consumer=owned_payload(tmp_path)
    link=base/'link'
    try: link.symlink_to(tmp_path/'consumer.json')
    except OSError: pytest.skip('Local account cannot create symlinks')
    with pytest.raises(PermissionError):r.retire(base,[dict(consumer,path='link')],owner='a'*64,consumer=consumer,output=tmp_path/'retire')
    assert (tmp_path/'consumer.json').is_file()
    with pytest.raises(ValueError,match='Unknown bank'):
        r.bank_refs(base,dict(population=refs[0],shards=[]))


def test_interrupted_cleanup_does_not_count_as_success(tmp_path,monkeypatch):
    base,refs,consumer=owned_payload(tmp_path)
    atomic_publish_bytes(base/'second.bin',b'target2')
    refs.append(file_ref(base/'second.bin',root=base))
    real=Path.unlink
    def interrupted(path,*args,**kwargs):
        if path.name=='second.bin':raise OSError('injected interruption')
        return real(path,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',interrupted)
    with pytest.raises(OSError):r.retire(base,refs,owner='a'*64,consumer=consumer,output=tmp_path/'retire')
    assert (tmp_path/'retire/intent.json').is_file()
    assert not (tmp_path/'retire/receipt.json').exists()
    with pytest.raises(FileNotFoundError):r.validate_retirement(tmp_path/'retire','a'*64)


def test_bank_peak_and_storage_no_silent_fallback(tmp_path,monkeypatch):
    monkeypatch.setattr(r.shutil,'disk_usage',lambda path:SimpleNamespace(free=20*2**30))
    result=r.budget([1,4,7],tmp_path,tmp_path,1)
    assert result['attention_bank_bytes']==4*(1+16+49)
    assert result['persistent_required_bytes']==2**30
    assert result['scratch_required_bytes']==4*66+3*128+3*2**30
    with pytest.raises(OSError):r.budget([1],tmp_path,tmp_path,0)
    with pytest.raises(PermissionError):r.scratch(dict(scratch_base=str(tmp_path/'missing')),'1')


def test_budgets_measure_whole_ladder_not_single_fit():
    f=dict(anchor_seconds=60.,construction_seconds=0.,measurements=[dict(load_seconds=10.,epoch_seconds=60.,readback_seconds=5.,export_seconds=20.)]*3)
    assert preflight.budgets(f,20.)==163
    too_long=deepcopy(f)
    too_long['measurements'][0]['epoch_seconds']=100000.
    with pytest.raises(ValueError,match='24h'):preflight.budgets(too_long,0.)
    with pytest.raises(ValueError):preflight.budgets(f,float('nan'))


@pytest.mark.parametrize('arm',['logit','attn'])
def test_exact_dry_live_partition_plans(tmp_path,monkeypatch,arm):
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    partition='debug'
    gate=c.artifact('GATE',root=str(tmp_path),project_dir='/project',source_commit='a'*40,
                    **campaign.registration(dict(nodes=parent_nodes()),arm,))
    monkeypatch.setattr(campaign,'check_gate',lambda g:None)
    monkeypatch.setattr(submission,'check_site',lambda p:None)
    calls=[]
    monkeypatch.setattr(exact.subprocess,'run',lambda cmd,**kw:(calls.append(cmd) or SimpleNamespace(stdout='12345\n')))
    plan=submission.submit(gate)
    assert calls==[] and plan['jobs']==1
    args=plan['commands'][0]['command']
    assert '--partition='+partition in args and '--gres=gpu:a100:1' in args
    assert '--mem=128000M' in args and '--time=08:00:00' in args
    with pytest.raises(PermissionError):submission.submit(gate,execute=True,plan_hash='0'*64,authorization=c.AUTHORIZATION)
    submission.submit(gate,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==1 and submission.accepted_jobs(gate)=={'preflight':'12345'}
    spec=c.artifact('SPEC',root=str(tmp_path/'science'),project_dir='/project',ladder_minutes=700)
    science=submission.render(spec,gate)
    assert science['jobs']==1 and science['commands'][0]['task_id']=='ladder'
    assert '--partition='+partition in science['commands'][0]['command']
    assert '--time=11:40:00' in science['commands'][0]['command']


def test_site_and_namespace_are_bound():
    from hlt_classification.jetclass2_delphes.execution import execution_site
    gate=c.artifact('GATE',root='/new',project_dir='/project',
        **campaign.registration(dict(nodes=parent_nodes()),'attn'))
    campaign.validate_execution(gate)
    assert submission.is_gate(gate)
    bad=with_content_hash(dict(gate,execution_site=execution_site('sporc_a100')))
    with pytest.raises(ValueError,match='debug'): submission.render(bad)
    bad=with_content_hash(dict(gate,contract='LUKA_ATTENTION_WEIGHT_GATE/v1'))
    with pytest.raises(ValueError): c.validate(bad,'GATE')
    old_report=c.artifact('PREFLIGHT',parents=dict(gate='f'*64))
    with pytest.raises(ValueError,match='Parent lineage'): preflight.validate_report(gate,old_report)


@pytest.mark.parametrize('arm',['logit','attn'])
def test_full_preflight_then_science_worker_with_retired_probes(tmp_path,monkeypatch,arm):
    gate,scratch,log=fixture(tmp_path,monkeypatch,arm)
    monkeypatch.setattr(campaign,'check_gate',lambda g:None)
    monkeypatch.setattr(w,'execution',lambda *args:'123')
    monkeypatch.setattr(preflight,'installed_environment',lambda:{'test':True})
    monkeypatch.setattr(w,'installed_environment',lambda:{'test':True})
    monkeypatch.setattr(r,'scratch',lambda g,j:scratch)
    snapshot=dict(peak_rss_bytes=2**30,gpu_peak_bytes=2**30)
    gpu=dict(total_memory_bytes=40*2**30,name='test GPU')
    monkeypatch.setattr(preflight,'memory_snapshot',lambda t:snapshot)
    monkeypatch.setattr(w,'memory_snapshot',lambda t:snapshot)
    monkeypatch.setattr(preflight,'gpu_identity',lambda:gpu)
    monkeypatch.setattr(w,'gpu_identity',lambda:gpu)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    real_parity=preflight.parity
    # Real CPU FP32 parity here; the existing installed-CUDA test covers actual BF16.
    monkeypatch.setattr(preflight,'parity',lambda raw,device,bf16:dict(real_parity(raw,'cpu'),bf16=bf16))
    real_chain=w.run_chain
    monkeypatch.setattr(w,'run_chain',lambda g,output,scratch_root,**kwargs:real_chain(
        g,output,scratch_root,device='cpu',log=log,**kwargs))
    report=preflight.run(gate)
    assert report['technical_checks_passed'] and len(report['facts']['stress'])==6
    assert all(s['coefficient']==gate['coefficient'] for s in report['facts']['stress'])
    assert not list(scratch.rglob('*.npy')) and not list(scratch.rglob('*.pt'))
    preflight.validate_report(gate,report)
    bad=deepcopy(report); bad['facts']['stress']=bad['facts']['stress'][:-1]
    with pytest.raises(ValueError,match='stress'):preflight.validate_report(gate,with_content_hash(bad))
    bad=deepcopy(report); bad['execution_site']['partition']='wrong'
    with pytest.raises(ValueError):preflight.validate_report(gate,with_content_hash(bad))
    bad=deepcopy(report); bad['memory']['gpu_peak_bytes']=40*2**30
    with pytest.raises(MemoryError):preflight.validate_report(gate,with_content_hash(bad))
    bad=deepcopy(report); bad['parities'][1]['native_attention_surface']=False
    with pytest.raises(ValueError,match='parity'):preflight.validate_report(gate,with_content_hash(bad))
    bad=deepcopy(report); bad['facts']['measurements'][0]['resident_target_bytes']+=1
    with pytest.raises(ValueError,match='target RAM'):preflight.validate_report(gate,with_content_hash(bad))
    write_immutable_json(Path(gate['root'])/'gate_spec.json',gate)
    spec=c.artifact('SPEC',root=str(Path(gate['root'])/'science'),
        gate_path=str(Path(gate['root'])/'gate_spec.json'),
        parents=dict(gate=gate['content_hash'],preflight=report['content_hash']))
    Path(spec['root']).mkdir()
    write_immutable_json(Path(spec['root'])/'campaign_spec.json',spec)
    science_scratch=r.create_owned(tmp_path/'science_scratch',gate['content_hash'])
    monkeypatch.setattr(r,'scratch',lambda g,j:science_scratch)
    monkeypatch.setattr(campaign,'check_spec',lambda s:(gate,report))
    result=w.run(spec)
    assert result['new_inference_requires_retraining'] and result['checkpoints_retired']
    assert set(result['contrasts'])=={'LOGIT_D000','ATTN_D000'}
    assert not list(science_scratch.rglob('*.pt'))
    with pytest.raises(FileExistsError):w.run(spec)
    # Archival CLI validates durable predictions without scheduler or old models.
    monkeypatch.setattr(campaign,'read_gate',lambda g:None)
    def no_remote(*args):raise AssertionError('Archival reader accessed live execution')
    monkeypatch.setattr(campaign,'check_spec',no_remote)
    monkeypatch.setattr(w,'execution',no_remote)
    (tmp_path/'anchor.pt').unlink()
    assert w.saved_results(spec)==result['rows']
    bad=deepcopy(result)
    bad['contrasts']['LOGIT_D000']['validation']['accuracy']+=.1
    (Path(spec['root'])/'results.json').write_text(__import__('json').dumps(with_content_hash(bad)))
    with pytest.raises(ValueError,match='contrasts'):w.saved_results(spec)


def test_results_reject_wrong_retirement_consumer(tmp_path,monkeypatch):
    gate,scratch,log=fixture(tmp_path,monkeypatch)
    root=Path(gate['root'])/'ladder'
    w.run_chain(gate,root,scratch,device='cpu',technical=True,log=log)
    directory=root/gate['nodes'][0]['node_id']
    step=w.validate_step(directory,gate,gate['nodes'][0],True)
    # Re-sign an otherwise valid but unrelated consumer. It must not count as dependency completion.
    intent_path=directory/'retired_checkpoint/intent.json'
    intent=load_json(intent_path)
    intent['consumer']=file_ref(directory/'receipt.json')
    intent['parents']['consumer']=intent['consumer']['sha256']
    intent=with_content_hash(intent)
    intent_path.write_text(__import__('json').dumps(intent))
    receipt_path=directory/'retired_checkpoint/receipt.json'
    receipt=load_json(receipt_path)
    receipt['parents']['intent']=intent['content_hash']
    receipt_path.write_text(__import__('json').dumps(with_content_hash(receipt)))
    with pytest.raises(ValueError,match='last checkpoint consumer'):
        w.validate_retired_rung(directory,gate,gate['nodes'][0],step)


def test_science_requires_accepted_exact_gate_journal_and_accounting(tmp_path,monkeypatch):
    gate,scratch,log=fixture(tmp_path,monkeypatch)
    report={'job_id':'123'}
    write_immutable_json(Path(gate['root'])/'preflight/receipt.json',report)
    monkeypatch.setattr(campaign,'check_gate',lambda g:None)
    monkeypatch.setattr(preflight,'validate_report',lambda g,r:None)
    monkeypatch.setattr(submission,'accepted_jobs',lambda g:{'preflight':'124'})
    called=[]
    monkeypatch.setattr(campaign,'preflight_accounting',lambda j:called.append(j))
    with pytest.raises(ValueError,match='journal'):campaign.evidence(gate)
    assert called==[]
    monkeypatch.setattr(submission,'accepted_jobs',lambda g:{'preflight':'123'})
    assert campaign.evidence(gate)==report and called==['123']
    def failed(_):raise ValueError('Failed scheduler accounting')
    monkeypatch.setattr(campaign,'preflight_accounting',failed)
    with pytest.raises(ValueError,match='accounting'):campaign.evidence(gate)


def test_symlink_guard_without_host_symlink_privileges(tmp_path,monkeypatch):
    base,refs,consumer=owned_payload(tmp_path)
    real=Path.is_symlink
    monkeypatch.setattr(Path,'is_symlink',lambda p:True if p==base/'payload.bin' else real(p))
    with pytest.raises(PermissionError,match='symlink'):
        r.retire(base,refs,owner='a'*64,consumer=consumer,output=tmp_path/'retire')
    assert (base/'payload.bin').is_file()


def test_source_pin_rejects_other_runtime_edits(monkeypatch):
    from hlt_classification.luka_particle_swap import source
    project=Path(__file__).resolve().parents[1]
    monkeypatch.setattr(source,'_source',lambda p,c:{'git_commit':c})
    changed=[]
    def git(cmd,**kwargs):
        if 'ls-files' in cmd:return SimpleNamespace(stdout='\n'.join(source.REQUIRED))
        if 'diff' in cmd:return SimpleNamespace(stdout='\n'.join(changed))
        raise AssertionError(cmd)
    monkeypatch.setattr(source.subprocess,'run',git)
    assert source.current(project,'a'*40)['git_commit']=='a'*40
    changed.append('src/hlt_classification/luka_particle_swap/worker.py')
    source.current(project,'a'*40)
    changed.append('src/hlt_classification/luka_attention/model.py')
    with pytest.raises(ValueError,match='Existing scientific source'):source.current(project,'a'*40)


def test_partial_rows_do_not_hide_missing_parents(tmp_path,monkeypatch):
    gate,scratch,log=fixture(tmp_path,monkeypatch)
    output=Path(gate['root'])/'ladder'
    output.mkdir()
    second=output/gate['nodes'][1]['node_id']
    second.mkdir()
    write_immutable_json(second/'receipt.json',{})
    with pytest.raises(ValueError,match='missing immediate parent'):w.result_rows(gate,output,partial=True)


def test_create_snapshots_references_and_refuses_overlap(tmp_path,monkeypatch):
    from hlt_classification.luka_particle_swap import source
    project=tmp_path/'project'; project.mkdir()
    oldroot=tmp_path/'old'; oldroot.mkdir()
    reference_root=tmp_path/'old_reference'; (reference_root/'science').mkdir(parents=True)
    parent=dict(project_dir=str(project),root=str(oldroot/'science'),nodes=parent_nodes())
    anchor=c.artifact('ANCHOR',spec={'path':str(oldroot/'science/campaign_spec.json')},
        evidence=dict(prepared_root=str(tmp_path/'prepared'),raw_container=str(tmp_path/'raw'), prepared='a'*64,cache='b'*64))
    payload=tmp_path/'pred.npz'
    p=dict(identities=cache('validation').identities,labels=np.arange(22)%11,probabilities=np.full((22,11),1/11,np.float32))
    atomic_publish_bytes(payload,deterministic_npz_bytes(p))
    imported=c.artifact('REFERENCES',parents=dict(anchor=anchor['content_hash']),rows=[],spec={'path':'old'},
        predictions={a+'_'+co:file_ref(payload) for a in ('LOGIT','ATTN') for co in c.COORDINATES})
    names=imported['predictions']
    proof=c.artifact('REFERENCE_ARCHIVE',parents=dict(anchor=anchor['content_hash'],spec='c'*64,results='d'*64),
        **source.SCOPE,source_commit=source.REFERENCE_COMMIT,spec='c'*64,results='d'*64,
        training_reports={n:'e'*64 for n in names},training_receipts={'train_'+n:'f'*64 for n in names},
        recorded_dependencies={'train_'+n:'f'*64 for n in names},verified_files=[imported['spec'],file_ref(payload)])
    imported=with_content_hash(dict(imported,archive=proof))
    monkeypatch.setattr(source,'current',lambda p,c:{'content_hash':'a'*64})
    monkeypatch.setattr(source,'authenticate',lambda p:(anchor,parent))
    monkeypatch.setattr(source,'references',lambda p,a:imported)
    for arm in ('logit','attn'):
        gate=campaign.create(tmp_path/arm,'old',reference_root/'science/campaign_spec.json',project,'a'*40,
            arm=arm,available_quota_gib=1,scratch_base='/tmp')
        campaign.read_gate(gate)
        campaign.check_gate(gate)
        assert gate['schema_version']==1 and gate['execution_site']['partition']=='debug'
        saved=load_json(Path(gate['root'])/'references.json')
        assert saved['predictions']['LOGIT_D000']['sha256']==file_ref(payload)['sha256']
    with pytest.raises(PermissionError,match='overlaps'):
        campaign.create(oldroot/'overwrite','old',reference_root/'science/campaign_spec.json',project,'a'*40,
            arm='logit',available_quota_gib=1,scratch_base='/tmp')
    with pytest.raises(PermissionError,match='scratch'):
        campaign.create(tmp_path/'new','old',reference_root/'science/campaign_spec.json',project,'a'*40,
            arm='logit',available_quota_gib=1,scratch_base='/home/ryreu')


def test_queue_worker_cli_and_runbook_syntax():
    import ast
    import re
    import shutil
    import subprocess
    for path in list(Path('src/hlt_classification/luka_particle_swap').glob('*.py'))+[Path('scripts/luka_particle_swap.py')]:
        text=path.read_text(encoding='utf-8')
        ast.parse(text,filename=str(path))
        assert 'import Fresh_check' not in text
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    if not bash.is_file():
        found=shutil.which('bash')
        if not found:pytest.skip('Bash unavailable for syntax checks')
        bash=Path(found)
    for path in ('scripts/queue_luka_particle_swap.sh','sbatch/run_luka_particle_swap.sh'):
        subprocess.run([str(bash),'-n',path],check=True,capture_output=True,text=True)
    text=Path('docs/LUKA_PARTICLE_SWAP_RUNBOOK.md').read_text(encoding='utf-8')
    blocks=re.findall(r'```bash\n(.*?)```',text,re.S)
    assert len(blocks)>=3
    for block in blocks:
        subprocess.run([str(bash),'-n'],input=block,check=True,capture_output=True,text=True)


@pytest.mark.parametrize('arm',['logit','attn'])
def test_queue_helper_requires_new_gate_and_exact_source(tmp_path,arm):
    import shlex
    import subprocess
    import sys
    lines=Path('scripts/queue_luka_particle_swap.sh').read_text(encoding='utf-8').splitlines()
    line=next(line for line in lines if 'python -B -s -c ' in line)
    args=shlex.split(line)
    program=args[args.index('-c')+1]
    gate=c.artifact('GATE',root=str(tmp_path),project_dir=str(tmp_path/'project'),source_commit='a'*40,
        anchor={'spec':{'path':str(tmp_path/'parent')}},
        reference_import={'spec':{'path':str(tmp_path/'reference')}},available_quota_gib=1,
        **campaign.registration(dict(nodes=parent_nodes()),arm))
    path=tmp_path/'gate_spec.json'
    write_immutable_json(path,gate)
    command=[sys.executable,'-B','-s','-c',program,str(path),str(tmp_path/'parent'),
             str(tmp_path/'reference'),arm,'1','a'*40,str(tmp_path/'project')]
    assert subprocess.run(command,capture_output=True,text=True).returncode==0
    for changed in (dict(source_commit='b'*40),dict(project_dir=str(tmp_path/'other')),
                    dict(schema_version=2,contract='LUKA_PARTICLE_SWAP_GATE/v2')):
        bad_path=tmp_path/(next(iter(changed))+'.json')
        write_immutable_json(bad_path,with_content_hash(dict(gate,**changed)))
        bad_command=command.copy(); bad_command[5]=str(bad_path)
        result=subprocess.run(bad_command,capture_output=True,text=True)
        assert result.returncode!=0 and ('AssertionError' in result.stderr or 'ValueError' in result.stderr)


@pytest.mark.parametrize('attention',[False,True])
def test_resident_bank_exact_no_minibatch_io_and_bounded(tmp_path,monkeypatch,attention):
    from hlt_classification.luka_particle_swap.banks import ResidentBank
    tiny_config(monkeypatch)
    tr=cache('train','U100')
    model_instance=model.new_model(c.nodes(parent_nodes(),'attn')[0])
    parents=dict(checkpoint='a'*64,report='b'*64)
    manifest=banks.export(tmp_path/'bank',model_instance,tr,parents=parents,teacher='ANCHOR',attention=attention,device='cpu')
    ordinary=banks.Bank(tmp_path/'bank',tr,parents=parents,teacher='ANCHOR',require_attention=attention)
    ix=np.array([21,0,8,8,15],np.int64)
    expected=ordinary.attention_batch(ix,16) if attention else None
    resident=ResidentBank(tmp_path/'bank',tr,parents=parents,teacher='ANCHOR',require_attention=attention,max_bytes=2**30)
    np.testing.assert_array_equal(resident.probabilities,ordinary.probabilities)
    assert resident.manifest==manifest and not resident.values.flags.writeable
    with pytest.raises(MemoryError,match='RAM'):
        ResidentBank(tmp_path/'bank',tr,parents=parents,teacher='ANCHOR',require_attention=attention,max_bytes=1)
    def no_io(*a,**kw):raise AssertionError('Target reread inside minibatch')
    monkeypatch.setattr(Path,'read_bytes',no_io)
    monkeypatch.setattr(np,'load',no_io)
    if attention:
        np.testing.assert_array_equal(resident.attention_batch(ix,16),expected)
        for bad,width in [(np.array([-1]),16),(np.array([22]),16),(np.array([0.5]),16),(ix,0),(ix,513)]:
            with pytest.raises(ValueError):resident.attention_batch(bad,width)
    else:
        with pytest.raises(ValueError):resident.attention_batch(ix,16)
    w.close_bank(ordinary)


def test_science_rejects_changed_swap_schedule_before_training(tmp_path,monkeypatch):
    gate,scratch,log=fixture(tmp_path,monkeypatch,'attn')
    wrong=deepcopy(w.data.build(gate)['audit'])
    wrong['endpoint_fingerprints_replayed']=False
    write_immutable_json(Path(gate['root'])/'preflight/receipt.json',dict(facts=dict(input_audit=with_content_hash(wrong))))
    monkeypatch.setattr(w,'fit',lambda *a,**kw:pytest.fail('Training before matching accepted input audit'))
    with pytest.raises(ValueError,match='schedule'):
        w.run_chain(gate,Path(gate['root'])/'ladder',scratch,device='cpu',log=log)


def test_results_reject_rehashed_wrong_validation_identity(tmp_path,monkeypatch):
    gate,scratch,log=fixture(tmp_path,monkeypatch,'logit')
    root=Path(gate['root'])/'ladder'
    w.run_chain(gate,root,scratch,device='cpu',technical=True,log=log)
    directory=root/gate['nodes'][0]['node_id']
    payload=directory/'validation_predictions.npz'
    values=load_npz_arrays(payload)
    values['identities'][0,31]^=1
    payload.write_bytes(deterministic_npz_bytes(values))
    step=load_json(directory/'receipt.json')
    step['predictions']=file_ref(payload,root=directory)
    (directory/'receipt.json').write_text(__import__('json').dumps(with_content_hash(step)))
    with pytest.raises(ValueError,match='population'):
        w.validate_step(directory,gate,gate['nodes'][0],True)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Installed CUDA optional; genuine A100 preflight still required')
def test_production_architecture_cuda_probe_and_native_parity(tmp_path,monkeypatch):
    # No architecture monkeypatch. Only population/allocation/PhaseLog are local fixtures.
    gate,scratch,log=fixture(tmp_path,monkeypatch,'attn',tiny=False)
    inputs=w.data.build(gate)
    raw=w.data.get(inputs,'train','U100').batch(np.arange(4))
    for bf16 in (False,True):
        result=preflight.parity(raw,'cuda',bf16=bf16)
        assert result['native_attention_surface'] is True and result['optimizer_updates'] is True
    root=Path(gate['root'])/'production_probe'
    facts=w.run_chain(gate,root,scratch,device='cuda',technical=True,log=log)
    assert len(facts['measurements'])==3
    assert all(n['native_checkpoint_readback'] for n in [
        w.validate_step(root/node['node_id'],gate,node,True) for node in gate['nodes']])
    w.result_rows(gate,root,technical=True)

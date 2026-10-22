"""New targets without a repeated full epoch: semantics, reuse, exact DAG and CUDA."""
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
from test_luka_fullstat_embedding_sweep import fixture as old_fixture
from test_luka_fullstat import fullstat, durable
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash
from hlt_classification.luka_fullstat_embedding_sweep import (
    contracts as old_c, banks as old_banks, training as old_training)
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.luka_fullstat_u100_embedding import (
    contracts as c, campaign, source, banks, training, worker, submission)


def fixture(path, *, export=True, device='cpu'):
    g,tr,val,off = old_fixture(path/'old', export=False)
    reuse = c.artifact('REUSE', counts=dict(c.COUNTS), cpus=6, memory_mb=180000,
        export_minutes=480, train_minutes={a:720+i for i,a in enumerate(old_c.WEIGHTS)},
        environment=g['anchor']['evidence']['environment'],
        gpu=dict(name='FIXTURE L40S', total_memory_bytes=48*2**30, compute_capability=[8,9]),
        anchor=g['anchor'], base_node=g['nodes']['w0'], execution_site=g['execution_site'],
        teacher_refs=g['anchor']['evidence']['models']['OFFLINE'],
        teacher_node=dict(node_id='COARSE_U100',coordinate='U100'),
        summaries=dict(train={co:dict(maximum=4) for co in ('D000','U100')},
                       validation=dict(D000=dict(maximum=4))))
    root = path/'new'; root.mkdir()
    spec = c.artifact('SPEC', parents=dict(source='c'*64,reuse=reuse['content_hash']),
        root=str(root), project_dir='/project', source_commit='b'*40, source_snapshot=dict(content_hash='c'*64),
        reuse=reuse, available_result_bytes=6*2**30, **campaign.policy(reuse))
    spec = with_content_hash(dict(spec, counts=dict(train=len(tr),validation=len(val))))
    write_immutable_json(root/'campaign_spec.json', spec)
    u100 = deepcopy(off); u100.coordinate_name = 'U100'
    if export:
        teacher = DelphesParticleTransformer()
        teacher.load_state_dict(torch.load(spec['teacher_refs']['checkpoint']['path'], weights_only=True))
        banks.export(root/'targets', teacher, u100, refs=spec['teacher_refs'], device=device)
    return spec,tr,val,u100


def test_scientific_policy_is_one_fresh_weight5_U100_fit(tmp_path):
    s,*_ = fixture(tmp_path,export=False)
    r = s['recipe']
    assert r['epochs']==20 and r['batch_size']==128 and r['auxiliary_coefficient']==5
    assert r['embedding_teacher']=='COARSE_U100' and r['ce_coefficient']==1 and r['kd_coefficient']==0
    assert not r['attention'] and not r['warm_start']
    assert s['node']['teacher'] is None and s['node']['context_coordinate'] is None
    assert s['node']['coordinate']=='D000' and s['node']['deployable']
    assert s['resource_reuse'] and not s['new_full_preflight']
    assert s['train_minutes']==726 and s['export_minutes']==480
    assert s['final_test_sealed'] and not s['final_test_accessed']
    assert c.target_recipe()['coordinate']=='U100' and c.target_recipe()['role']=='train'


@pytest.mark.parametrize('key,value', [('cpus',12),('memory_mb',90000),('export_minutes',60),
    ('counts',{}),('train_minutes',{'w5':720}),('teacher_node',dict(node_id='COARSE_D066',coordinate='D066'))])
def test_inapplicable_resource_reuse_rejected(tmp_path,key,value):
    s,*_=fixture(tmp_path,export=False)
    with pytest.raises(ValueError): campaign.policy(with_content_hash(dict(s['reuse'],**{key:value})))


def test_weight5_objective_and_frozen_target():
    x=torch.randn(3,11,requires_grad=True);p=torch.randn(3,128,requires_grad=True)
    t=torch.randn(3,128,requires_grad=True);y=torch.tensor([0,2,4])
    loss,ce,aux=training.objective(x,y,p,t,coefficient=5)
    torch.testing.assert_close(loss,ce+5*aux,rtol=0,atol=0)
    loss.backward()
    assert t.grad is None and p.grad is not None and x.grad is not None
    for coefficient in (0.,2.,10.,50.,True,float('nan')):
        with pytest.raises(ValueError): training.objective(x,y,p,t,coefficient=coefficient)


def test_frozen_U100_bank_new_moments_and_forbidden_roles(tmp_path,monkeypatch):
    tiny(monkeypatch)
    s,tr,val,u=fixture(tmp_path)
    bank=banks.load(s,tr)
    assert bank.manifest['teacher']=='COARSE_U100' and bank.manifest['coordinate']=='U100'
    np.testing.assert_allclose(bank.targets.mean(0),0,atol=1e-6)
    assert bank.targets.shape==(len(tr),128) and not bank.targets.flags.writeable
    banks.check_join(bank,tr)
    for role,co in [('validation','U100'),('final_test','D000'),('train','OFFLINE'),('train','D066')]:
        with pytest.raises(PermissionError): worker.data(s,role,co)
    for wrong in (val,deepcopy(u)):
        wrong.role='validation'
        with pytest.raises(PermissionError): banks.load(s,wrong)
    swapped=deepcopy(tr);swapped.identities=tr.identities[::-1]
    with pytest.raises(ValueError): banks.load(s,swapped)
    other=DelphesParticleTransformer()
    with pytest.raises(ValueError,match='weights'): banks.export(tmp_path/'bad',other,u,refs=s['teacher_refs'],device='cpu')
    with pytest.raises(PermissionError): banks.export(tmp_path/'bad',other,tr,refs=s['teacher_refs'],device='cpu')


@pytest.mark.parametrize('key,value',[('teacher','OFFLINE'),('coordinate','D066'),('role','validation'),
                                    ('width',127),('rows',1),('final_test_accessed',True)])
def test_resigned_mislabelled_bank_rejected(tmp_path,monkeypatch,key,value):
    tiny(monkeypatch)
    s,tr,*_=fixture(tmp_path)
    path=Path(s['root'])/'targets/manifest.json'
    path.write_text(json.dumps(with_content_hash(dict(load_json(path),**{key:value}))))
    with pytest.raises((ValueError,PermissionError)):banks.load(s,tr)


def test_same_targets_weight_and_seed_match_accepted_training_kernel(tmp_path,monkeypatch):
    tiny(monkeypatch)
    s,tr,val,u=fixture(tmp_path)
    teacher=DelphesParticleTransformer()
    teacher.load_state_dict(torch.load(s['teacher_refs']['checkpoint']['path'],weights_only=True))
    off=deepcopy(u);off.coordinate_name='OFFLINE'
    old_banks.export(tmp_path/'oldtargets',teacher,off,refs=s['teacher_refs'],device='cpu')
    old=old_banks.Bank(tmp_path/'oldtargets',tr,refs=s['teacher_refs'])
    new=banks.load(s,tr)
    np.testing.assert_array_equal(old.targets,new.targets)
    # Test-only extension of donor coefficient whitelist, never a runtime patch.
    monkeypatch.setitem(old_c.WEIGHTS,'test_weight5',5.)
    with parity_backend('cpu'):
        a,expected=old_training.train(tr,val,bank=old,node=s['node'],device='cpu')
        b,actual=training.train(tr,val,bank=new,node=s['node'],device='cpu')
    assert a['selected_pass']==b['selected_pass'] and a['validation']==b['validation']
    assert all(torch.equal(v,expected[k]) for k,v in actual.items())
    assert [r['validation'] for r in a['validation_history']]==[r['validation'] for r in b['validation_history']]
    worker.fit(tmp_path/'fit',s,tr,val,new,device='cpu')
    report=worker.validate_fit(tmp_path/'fit',s,new.manifest['content_hash'])
    assert report['passes']==20 and report['selected_state']=='native_backbone_only'
    assert not any(k.startswith('predictor.') for k in torch.load(tmp_path/'fit/selected.pt',weights_only=True))


def test_real_prepared_U100_cache_loading(durable,monkeypatch):
    output,meta=durable
    s=dict(memory_mb=180000,counts=meta[1]['counts'],anchor=dict(evidence=dict(
        prepared=meta[0]['content_hash'],cache_root=str(output))))
    monkeypatch.setattr(worker,'metadata',lambda _:meta)
    u=worker.data(s,'train','U100');tr=worker.data(s,'train','D000')
    assert len(u)==len(tr)==264
    np.testing.assert_array_equal(u.identities,tr.identities)
    np.testing.assert_array_equal(u.labels,tr.labels)


def test_two_job_plan_no_preflight_and_explicit_admission(tmp_path,monkeypatch):
    s,*_=fixture(tmp_path,export=False)
    monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(submission,'check_site',lambda _:None)
    plan=submission.submit(s)
    assert plan['jobs']==2 and plan['preflight_jobs']==0
    assert [r['task_id'] for r in plan['commands']]==['export_U100','train_w5']
    assert '--dependency=afterok:${JOB_export_U100}' in plan['commands'][1]['command']
    assert all('--kill-on-invalid-dep=yes' in r['command'] for r in plan['commands'])
    with pytest.raises(PermissionError): submission.submit(s,execute=True,plan_hash='bad',authorization=c.AUTHORIZATION)
    calls=[]
    def fake(args,**kw):
        calls.append(args)
        return SimpleNamespace(returncode=0,stdout=str(100+len(calls))+'\n',stderr='')
    # Exact-DAG path, real immutable journal, fake scheduler transport only.
    monkeypatch.setattr(subprocess,'run',fake)
    submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert submission.accepted_jobs(s)==dict(export_U100='101',train_w5='102')
    assert '--dependency=afterok:101' in calls[1]
    with pytest.raises(FileExistsError):submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==2


def test_export_accounting_rejects_failure_missing_or_wrong_owner(tmp_path,monkeypatch):
    s,*_=fixture(tmp_path,export=False)
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:dict(export_U100='101'))
    monkeypatch.setattr(submission.getpass,'getuser',lambda:'rlyang')
    valid='101|lukafsu_export_U100|rlyang|default|COMPLETED|0:0\n'
    for text in ('',valid.replace('COMPLETED','FAILED'),valid.replace('0:0','1:0'),valid.replace('rlyang','other')):
        monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=text))
        with pytest.raises(ValueError):submission.successful_export(s,'101')
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=valid))
    submission.successful_export(s,'101')


def test_original_source_replay_and_changed_measured_kernel_rejected(tmp_path,monkeypatch):
    original=tmp_path/'old';project=tmp_path/'new';root=tmp_path/'gate'
    root.mkdir()
    for directory in (original,project):
        (directory/'src').mkdir(parents=True)
        (directory/'src/kernel.py').write_text('# accepted scientific kernel\n')
    g=dict(project_dir=str(original),root=str(root),source_commit=source.DONOR,
        counts=c.COUNTS,anchor={},nodes=dict(w0={}),execution_site={},cpus=6,memory_mb=180000,gate_minutes=480)
    write_immutable_json(root/'gate_spec.json',g)
    calls=[]
    def replay(args,**kwargs):
        calls.append((args,kwargs))
        if args[0]=='git':return SimpleNamespace(stdout='src/kernel.py\n')
        assert kwargs['cwd']==original and kwargs['env']['PYTHONPATH']==str(original/'src')
        assert kwargs['env']['PYTHONNOUSERSITE']=='1'
        assert args[-1]==str(root/'gate_spec.json') and '-B' in args and '-s' in args
        return SimpleNamespace(returncode=0,stdout='U100_REUSE={"teacher_refs":{}}\n',stderr='')
    monkeypatch.setattr(subprocess,'run',replay)
    record=source.authenticate(root/'gate_spec.json',project)
    c.validate(record,'REUSE')
    assert set(record['compatible_source'])=={'src/kernel.py'}
    assert 'campaign.accepted(g)' in source.REPLAY
    assert "original.completed(p,'train_COARSE_U100')" in source.REPLAY
    compile(source.REPLAY,'original-replay','exec')
    (project/'src/kernel.py').write_text('# altered kernel\n')
    with pytest.raises(ValueError,match='Changed measured'):source.authenticate(root/'gate_spec.json',project)
    assert len(calls)==3  # second authentication stopped before the original interpreter


@pytest.mark.parametrize('result',[
    SimpleNamespace(returncode=1,stdout='',stderr='original evidence rejected'),
    SimpleNamespace(returncode=0,stdout='',stderr=''),
    SimpleNamespace(returncode=0,stdout='U100_REUSE={}\nU100_REUSE={}\n',stderr='')])
def test_missing_or_failed_original_acceptance_blocks_reuse(tmp_path,monkeypatch,result):
    original=tmp_path/'original';project=tmp_path/'project';root=tmp_path/'gate'
    root.mkdir()
    for directory in (original,project):
        (directory/'src').mkdir(parents=True)
        (directory/'src/kernel.py').write_text('')
    write_immutable_json(root/'gate_spec.json',dict(project_dir=str(original),root=str(root),
        source_commit=source.DONOR,counts=c.COUNTS))
    monkeypatch.setattr(subprocess,'run',lambda args,**kw:SimpleNamespace(stdout='src/kernel.py\n') if args[0]=='git' else result)
    with pytest.raises(ValueError):source.authenticate(root/'gate_spec.json',project)


@pytest.mark.parametrize('memory',[
    dict(peak_rss_bytes=0,gpu_peak_bytes=1),
    dict(peak_rss_bytes=180000*1024**2,gpu_peak_bytes=1),
    dict(peak_rss_bytes=1,gpu_peak_bytes=48*2**30),
    dict(peak_rss_bytes=float('nan'),gpu_peak_bytes=1)])
def test_resource_reuse_does_not_admit_unmeasured_or_excessive_support(tmp_path,memory):
    s,*_=fixture(tmp_path,export=False)
    with pytest.raises(MemoryError):worker.limits(s,memory)


def test_failed_export_evidence_prevents_student_fit(tmp_path,monkeypatch):
    s,*_=fixture(tmp_path,export=False)
    monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(worker,'execution',lambda *a:'102')
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:dict(export_U100='101',train_w5='102'))
    monkeypatch.setattr(worker,'fit',lambda *a,**kw:pytest.fail('Fit must not start without successful export'))
    with pytest.raises(FileNotFoundError):worker.run(s,'train_w5')
    assert not (Path(s['root'])/'train_w5').exists()


def test_real_installed_cuda_export_then_full20_worker(tmp_path,monkeypatch):
    if not torch.cuda.is_available():pytest.skip('Local CUDA unavailable; not Oscar resource certification')
    s,tr,val,u=fixture(tmp_path,export=False)
    monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(worker,'execution',lambda *a:__import__('os').environ['SLURM_JOB_ID'])
    monkeypatch.setattr(worker,'metadata',lambda _: (dict(content_hash=tr.foundation_sha256,summaries=s['reuse']['summaries']),None,None))
    monkeypatch.setattr(worker,'data',lambda s,r,co,**kw:u if co=='U100' else tr if r=='train' else val)
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:dict(export_U100='101',train_w5='102'))
    monkeypatch.setattr(submission,'successful_export',lambda *a:None)
    monkeypatch.setattr(worker,'memory_snapshot',lambda _:dict(peak_rss_bytes=2**30,rss_bytes=2**30,
        gpu_peak_bytes=max(1,int(torch.cuda.max_memory_allocated()))))
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    monkeypatch.setenv('SLURM_JOB_ID','101')
    result=worker.run(s,'export_U100')
    assert result['probe']['steps']==3 and not result['probe']['technical_epoch_run']
    assert not result['new_full_preflight']
    assert not (Path(s['root'])/'preflight').exists()
    monkeypatch.setenv('SLURM_JOB_ID','102')
    fitted=worker.run(s,'train_w5')
    assert fitted['task_id']=='train_w5'
    summary=worker.results(s)
    assert len(summary['rows'])==4 and summary['recovery_baseline'].startswith('historical')
    assert not summary['final_test_accessed'] and summary['final_test_sealed']
    saved=load_json(Path(s['root'])/'train_w5/training_report.json')
    assert saved['passes']==20
    task=Path(s['root'])/'train_w5/task_receipt.json'
    task_bytes=task.read_bytes()
    task.write_text(json.dumps(with_content_hash(dict(load_json(task),job_id='999'))))
    with pytest.raises(ValueError,match='publication identity'):worker.results(s)
    task.write_bytes(task_bytes)
    old=Path(s['root'])/'export_U100/receipt.json'
    before=old.read_bytes()
    bad=deepcopy(result);bad['probe']['steps']=0
    old.write_text(json.dumps(with_content_hash(bad)))
    with pytest.raises(ValueError):worker.validate_export(s)
    old.write_bytes(before)
    payload=Path(s['root'])/'targets/targets.npz'
    payload.write_bytes(payload.read_bytes()+b'bad')
    with pytest.raises(ValueError):worker.validate_export(s)


def test_source_guard_and_entrypoint_syntax(monkeypatch):
    project=Path(__file__).resolve().parents[1]
    monkeypatch.setattr(source,'_source',lambda *a:dict(content_hash='a'*64))
    def git(cmd,**kw):
        return SimpleNamespace(stdout='\n'.join(source.REQUIRED) if 'ls-files' in cmd else
            '\n'.join(source.ADDITIONS|source.ENTRYPOINTS) if 'diff' in cmd else '')
    monkeypatch.setattr(subprocess,'run',git)
    assert source.current(project,'b'*40)['content_hash']=='a'*64
    for path in source.REQUIRED:assert (project/path).is_file(),path
    monkeypatch.setattr(subprocess,'run',lambda cmd,**kw:SimpleNamespace(stdout=
        '\n'.join(source.REQUIRED) if 'ls-files' in cmd else 'src/old_runtime.py' if 'diff' in cmd else ''))
    with pytest.raises(ValueError):source.current(project,'b'*40)


def test_shell_cli_documentation_syntax():
    project=Path(__file__).resolve().parents[1]
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    if not bash.is_file():bash=shutil.which('bash')
    if not bash:pytest.skip('Bash unavailable')
    for path in ('scripts/queue_luka_fullstat_u100_embedding.sh','sbatch/run_luka_fullstat_u100_embedding.sh'):
        subprocess.run([str(bash),'-n',path],cwd=project,check=True,capture_output=True,timeout=30)
    book=(project/'docs/LUKA_FULLSTAT_U100_EMBEDDING_RUNBOOK.md').read_text()
    for block in re.findall(r'^```bash\n(.*?)^```',book,re.M|re.S):
        subprocess.run([str(bash),'-n'],input=block,check=True,text=True,capture_output=True,timeout=30)
    subprocess.run([sys.executable,'-B','scripts/luka_fullstat_u100_embedding.py','--help'],
                   cwd=project,check=True,capture_output=True,timeout=60)
    for path in source.REQUIRED:
        text=(project/path).read_text()
        assert all(line==line.rstrip() for line in text.splitlines()),path
        if path.endswith('.md'):
            for link in re.findall(r'\]\(([^)#]+)(?:#[^)]*)?\)',text):
                assert (project/path).parent.joinpath(link).is_file(),(path,link)

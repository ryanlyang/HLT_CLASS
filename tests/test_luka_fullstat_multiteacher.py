"""Four-target semantics, source/queue guards and actual installed-Weaver workers."""
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
from test_luka_fullstat_u100_embedding import fixture as u_fixture
from test_luka_fullstat import fullstat, durable
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import _forward
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_embedding_aux.model import new_model as single_model
from hlt_classification.luka_fullstat_embedding_sweep import banks as old_banks
from hlt_classification.luka_fullstat_multiteacher import (
    contracts as c, campaign, source, banks, model, training, worker, submission)


def fixture(path, *, export=True):
    path.mkdir(exist_ok=True)
    (path/'previous').mkdir()
    previous,tr,val,u = u_fixture(path/'previous',export=export)
    refs = previous['teacher_refs']
    off = deepcopy(u); off.coordinate_name = 'OFFLINE'
    oldtargets = path/'offline_targets'
    if export:
        teacher = DelphesParticleTransformer()
        teacher.load_state_dict(torch.load(refs['checkpoint']['path'],weights_only=True))
        old_banks.export(oldtargets,teacher,off,refs=refs,device='cpu')
    else:
        for directory in (oldtargets,Path(previous['root'])/'targets'):
            directory.mkdir(exist_ok=True)
            write_immutable_json(directory/'manifest.json',{})
    reuse = c.artifact('REUSE', counts=dict(c.COUNTS),
        teachers={t:dict(node=dict(node_id=t,coordinate=co,context_coordinate=None),refs=refs) for t,co in c.TEACHERS.items()},
        reuse_banks=dict(OFFLINE=file_ref(oldtargets/'manifest.json'),
            COARSE_U100=file_ref(Path(previous['root'])/'targets/manifest.json')),
        reference=refs, anchor=previous['anchor'], base_node=previous['node'],
        execution_site=previous['execution_site'], environment=previous['reuse']['environment'],
        gpu=previous['reuse']['gpu'],parent_train_minutes=726,
        summaries=dict(train={co:dict(maximum=4) for co in ('D000',*c.TEACHERS.values())},
                       validation=dict(D000=dict(maximum=4))),protected_roots=[previous['root'],str(oldtargets)])
    root=path/'multi'; root.mkdir()
    s=c.artifact('SPEC',parents=dict(source='c'*64,reuse=reuse['content_hash']),root=str(root),
        project_dir='/project',source_commit='b'*40,source_snapshot=dict(content_hash='c'*64),
        reuse=reuse,available_result_bytes=12*2**30,**campaign.policy(reuse))
    s=with_content_hash(dict(s,counts=dict(train=len(tr),validation=len(val))))
    write_immutable_json(root/'campaign_spec.json',s)
    views={co:deepcopy(u) for co in c.TEACHERS.values()}
    for co,v in views.items():v.coordinate_name=co
    return s,tr,val,views


def bundle(s,tr,views):
    for t in ('COARSE_D066','COARSE_D033'):
        teacher=DelphesParticleTransformer()
        refs=s['teachers'][t]['refs']
        teacher.load_state_dict(torch.load(refs['checkpoint']['path'],weights_only=True))
        banks.export(Path(s['root'])/'targets'/t,teacher,views[c.TEACHERS[t]],
            refs=refs,teacher_id=t,device='cpu')
    return banks.publish(s,tr)


def test_science_exact_four_heads_one_new_fit(tmp_path):
    s,*_=fixture(tmp_path,export=False)
    r=s['recipe']
    assert r['weights']=={t:1.25 for t in c.TEACHERS} and sum(r['weights'].values())==5
    assert r['epochs']==20 and r['batch_size']==128 and r['ce_coefficient']==1 and r['kd_coefficient']==0
    assert not any(r[k] for k in ('attention','relational','warm_start','teacher_embedding_average'))
    assert s['new_scientific_fits']==1 and s['automatic_dependency'] and not s['automatic_retry']
    assert s['node']['teacher'] is None and s['node']['context_coordinate'] is None and s['node']['deployable']
    assert s['node']['coordinate']=='D000' and s['node']['initialization_seed']==s['reuse']['base_node']['initialization_seed']
    assert s['final_test_sealed'] and not s['final_test_accessed']
    assert s['setup_minutes']==720 and s['train_minutes']==1089
    assert s['acceptance']['benchmark_steps']==512 and not s['acceptance']['full_epoch_training_measured']


@pytest.mark.parametrize('key,value',[('counts',{}),('teachers',{}),('reuse_banks',{}),
    ('parent_train_minutes',1),('parent_train_minutes',True),('gpu',dict(name='A100'))])
def test_bad_reuse_policy_rejected(tmp_path,key,value):
    s,*_=fixture(tmp_path,export=False)
    with pytest.raises(ValueError):campaign.policy(with_content_hash(dict(s['reuse'],**{key:value})))


def test_objective_exact_four_means_detached_targets_and_gradients():
    x=torch.randn(5,11,requires_grad=True); y=torch.arange(5)
    p={t:torch.randn(5,128,requires_grad=True) for t in c.TEACHERS}
    targets={t:torch.randn(5,128,requires_grad=True) for t in c.TEACHERS}
    loss,ce,aux=training.objective(x,y,p,targets)
    torch.testing.assert_close(loss,ce+sum(1.25*aux[t] for t in c.TEACHERS),rtol=0,atol=0)
    loss.backward()
    for t in c.TEACHERS:
        assert targets[t].grad is None
        torch.testing.assert_close(p[t].grad,1.25*(p[t].detach()-targets[t].detach()).clamp(-1,1)/p[t].numel())
    shared=torch.randn(5,128,requires_grad=True); target=torch.randn(5,128)
    a,ce,_=training.objective(x,y,{t:shared for t in c.TEACHERS},{t:target for t in c.TEACHERS})
    expected=ce+5*torch.nn.functional.smooth_l1_loss(shared,target)
    torch.testing.assert_close(a,expected)
    torch.testing.assert_close(torch.autograd.grad(a,shared,retain_graph=True)[0],torch.autograd.grad(expected,shared)[0])
    with pytest.raises(ValueError):training.objective(x,y,{},targets)
    bad=dict(targets);bad['OFFLINE']=torch.full((5,128),float('nan'))
    with pytest.raises(ValueError):training.objective(x,y,p,bad)


@pytest.mark.parametrize('bf16',[False,True])
def test_installed_native_parity_rng_four_gradients_and_no_inference_heads(tmp_path,monkeypatch,bf16):
    tiny(monkeypatch)
    s,tr,*_=fixture(tmp_path,export=False)
    if bf16 and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    device='cuda' if bf16 else 'cpu'
    old=single_model(s['node']); expected_rng=torch.get_rng_state().clone()
    new=model.new_model(s['node'])
    assert torch.equal(expected_rng,torch.get_rng_state())
    assert all(torch.equal(v,old.native_state()[k]) for k,v in new.native_state().items())
    assert all(torch.equal(v,old.predictor.state_dict()[k]) for k,v in new.predictors['COARSE_U100'].state_dict().items())
    raw=tr.batch(np.arange(4));new.to(device).eval();old.to(device).eval()
    with parity_backend(device),torch.no_grad():
        a=_forward(old,raw,device,bf16=bf16)
        b=_forward(new,raw,device,bf16=bf16)
        new.capture=True
        d=_forward(new,raw,device,bf16=bf16)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(a,d,rtol=0,atol=0)
    new.train()
    logits=_forward(new,raw,device,bf16=bf16)
    loss,*_=training.objective(logits,torch.from_numpy(raw['labels']).to(device),new.predictions,
        {t:torch.randn(4,128,device=device) for t in c.TEACHERS})
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in new.named_parameters() if n.startswith('mod.'))
    for t in c.TEACHERS:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.predictors[t].parameters())
    new.eval();new.capture=False
    for head in new.predictors.values():
        for p in head.parameters():p.data.fill_(float('nan'))
    with torch.no_grad():assert torch.isfinite(_forward(new,raw,device,bf16=bf16)).all()
    native=DelphesParticleTransformer();native.load_state_dict(new.native_state(),strict=True)


def test_four_banks_separate_moments_and_original_contracts(tmp_path,monkeypatch):
    tiny(monkeypatch)
    s,tr,val,views=fixture(tmp_path)
    b=bundle(s,tr,views)
    for t in c.TEACHERS:
        assert b.targets[t].shape==(len(tr),128) and not b.targets[t].flags.writeable
        np.testing.assert_allclose(b.targets[t].mean(0),0,atol=1e-6)
    assert load_json(b.manifest['records']['OFFLINE']['path'])['contract'].startswith('LUKA_FULLSTAT_EMBEDDING_SWEEP_')
    assert load_json(b.manifest['records']['COARSE_U100']['path'])['contract'].startswith('LUKA_FULLSTAT_U100_EMBEDDING_')
    assert banks.load(s,tr).manifest==b.manifest
    for role,co in [('final_test','D000'),('validation','U100'),('unused','D000'),('train','invalid')]:
        with pytest.raises(PermissionError):worker.data(s,role,co)
    with pytest.raises(PermissionError):banks.load(s,val)
    swapped=deepcopy(tr);swapped.identities=tr.identities[::-1]
    with pytest.raises(ValueError):banks.load(s,swapped)
    changed=deepcopy(tr);changed.labels=(tr.labels+1)%11
    with pytest.raises(ValueError):banks.load(s,changed)


def test_independent_teacher_axes_are_not_averaged(tmp_path,monkeypatch):
    tiny(monkeypatch)
    from hlt_classification.context_fusion.worker import save_state
    from hlt_classification.data.cache_contracts import load_npz_arrays
    s,tr,_,views=fixture(tmp_path)
    hashes={t:deepcopy(ref) for t,ref in s['reuse_banks'].items()}
    # Different frozen native weights produce genuinely different target bases.
    for seed,t in enumerate(('COARSE_D066','COARSE_D033'),501):
        torch.manual_seed(seed)
        teacher=DelphesParticleTransformer()
        selected=tmp_path/(t+'.pt');save_state(selected,teacher.state_dict())
        refs=dict(s['teachers'][t]['refs'],checkpoint=file_ref(selected))
        s['teachers'][t]['refs']=refs
        banks.export(Path(s['root'])/'targets'/t,teacher,views[c.TEACHERS[t]],
            refs=refs,teacher_id=t,device='cpu')
    b=banks.Bundle(s,tr)
    assert not np.array_equal(b.targets['COARSE_D066'],b.targets['COARSE_D033'])
    means=[]
    for t in ('COARSE_D066','COARSE_D033'):
        a=load_npz_arrays(Path(s['root'])/'targets'/t/'targets.npz')
        means.append(a['mean'])
        np.testing.assert_allclose(a['targets'].mean(0),0,atol=1e-6)
        np.testing.assert_array_equal(a['targets'],b.targets[t])
    assert not np.array_equal(*means)
    assert all(file_ref(ref['path'])==hashes[t] for t,ref in s['reuse_banks'].items())


@pytest.mark.parametrize('key,value',[('teacher','OFFLINE'),('coordinate','D033'),('role','validation'),
    ('width',127),('rows',1),('final_test_accessed',True)])
def test_resigned_bank_semantics_fail_closed(tmp_path,monkeypatch,key,value):
    tiny(monkeypatch)
    s,tr,_,v=fixture(tmp_path);bundle(s,tr,v)
    path=Path(s['root'])/'targets/COARSE_D066/manifest.json'
    path.write_text(json.dumps(with_content_hash(dict(load_json(path),**{key:value}))))
    with pytest.raises((ValueError,PermissionError)):banks.load(s,tr)


def test_payload_corruption_and_wrong_checkpoint_rejected(tmp_path,monkeypatch):
    tiny(monkeypatch)
    s,tr,_,v=fixture(tmp_path);bundle(s,tr,v)
    path=Path(s['root'])/'targets/COARSE_D033/targets.npz'
    path.write_bytes(path.read_bytes()+b'corruption')
    with pytest.raises(ValueError):banks.load(s,tr)
    with pytest.raises(ValueError,match='weights'):
        banks.export(tmp_path/'wrong',DelphesParticleTransformer(),v['D033'],
            refs=s['teachers']['COARSE_D033']['refs'],teacher_id='COARSE_D033',device='cpu')


def test_full20_production_fit_native_readback_and_losses(tmp_path,monkeypatch):
    tiny(monkeypatch)
    s,tr,val,v=fixture(tmp_path);b=bundle(s,tr,v)
    with parity_backend('cpu'):worker.fit(tmp_path/'fit',s,tr,val,b,device='cpu')
    r=worker.validate_fit(tmp_path/'fit',s,b.manifest['content_hash'])
    assert r['passes']==20 and r['scientific_fit'] and not r['technical']
    assert not any(k.startswith('predictors.') for k in torch.load(tmp_path/'fit/selected.pt',weights_only=True))
    for row in r['validation_history']:
        assert set(row['auxiliary_losses'])==set(c.TEACHERS)
    bad=deepcopy(r);bad['validation_history'][0]['weighted_auxiliary_losses']['OFFLINE']=0
    with pytest.raises(ValueError):worker.report_valid(with_content_hash(bad),s,b.manifest['content_hash'])


def test_real_prepared_four_coordinate_loading(durable,monkeypatch):
    output,meta=durable
    s=dict(memory_mb=180000,counts=meta[1]['counts'],anchor=dict(evidence=dict(
        prepared=meta[0]['content_hash'],cache_root=str(output))))
    monkeypatch.setattr(worker,'metadata',lambda _:meta)
    tr=worker.data(s,'train','D000')
    for co in c.TEACHERS.values():
        value=worker.data(s,'train',co)
        np.testing.assert_array_equal(value.identities,tr.identities)
        np.testing.assert_array_equal(value.labels,tr.labels)


def test_exact_two_job_dag_dry_review_then_single_authorized_submit(tmp_path,monkeypatch):
    s,*_=fixture(tmp_path,export=False)
    monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(submission,'check_site',lambda _:None)
    plan=submission.submit(s)
    assert plan['jobs']==2 and plan['setup_jobs']==1 and plan['scientific_fits']==1
    assert [r['task_id'] for r in plan['commands']]==['setup','train_multi4']
    assert '--dependency=afterok:${JOB_setup}' in plan['commands'][1]['command']
    assert all('--kill-on-invalid-dep=yes' in r['command'] for r in plan['commands'])
    with pytest.raises(PermissionError):submission.submit(s,execute=True,plan_hash='bad',authorization=c.AUTHORIZATION)
    calls=[]
    def fake(args,**kw):
        calls.append(args)
        return SimpleNamespace(returncode=0,stdout=str(100+len(calls))+'\n',stderr='')
    monkeypatch.setattr(subprocess,'run',fake)
    submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert submission.accepted_jobs(s)==dict(setup='101',train_multi4='102')
    assert '--dependency=afterok:101' in calls[1]
    with pytest.raises(FileExistsError):submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==2


def test_exact_success_accounting_and_failed_setup_blocks_fit(tmp_path,monkeypatch):
    s,*_=fixture(tmp_path,export=False)
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:dict(setup='101',train_multi4='102'))
    monkeypatch.setattr(submission.getpass,'getuser',lambda:'rlyang')
    valid='101|lukafsm_setup|rlyang|default|COMPLETED|0:0\n'
    for text in ('',valid.replace('COMPLETED','FAILED'),valid.replace('0:0','1:0'),valid.replace('rlyang','other')):
        monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=text))
        with pytest.raises(ValueError):submission.successful_setup(s,'101')
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=valid))
    submission.successful_setup(s,'101')
    monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(worker,'execution',lambda *a:'102')
    monkeypatch.setattr(worker,'fit',lambda *a,**kw:pytest.fail('No fit without setup admission'))
    with pytest.raises(FileNotFoundError):worker.run(s,'train_multi4')
    assert not (Path(s['root'])/'train_multi4').exists()


def test_original_source_replay_changed_imported_kernel_rejected(tmp_path,monkeypatch):
    original=tmp_path/'old';project=tmp_path/'new';root=tmp_path/'parent';root.mkdir()
    for directory in (original,project):
        (directory/'src').mkdir(parents=True);(directory/'src/kernel.py').write_text('# accepted\n')
    write_immutable_json(root/'campaign_spec.json',dict(project_dir=str(original),root=str(root),
        source_commit=source.DONOR,counts=c.COUNTS))
    calls=[]
    def replay(args,**kwargs):
        calls.append((args,kwargs))
        if args[0]=='git':return SimpleNamespace(stdout='src/kernel.py\n')
        assert kwargs['cwd']==original and kwargs['env']['PYTHONPATH']==str(original/'src')
        return SimpleNamespace(returncode=0,stdout='MULTI_REUSE={}\n',stderr='')
    monkeypatch.setattr(subprocess,'run',replay)
    record=source.authenticate(root/'campaign_spec.json',project)
    c.validate(record,'REUSE');assert set(record['compatible_source'])=={'src/kernel.py'}
    assert 'campaign.check(s); result=worker.results(s)' in source.REPLAY
    assert "original.completed(p,'train_'+name)" in source.REPLAY
    compile(source.REPLAY,'original-replay','exec')
    (project/'src/kernel.py').write_text('# altered\n')
    with pytest.raises(ValueError,match='Changed donor'):source.authenticate(root/'campaign_spec.json',project)
    assert len(calls)==3


@pytest.mark.parametrize('result',[
    SimpleNamespace(returncode=1,stdout='',stderr='original evidence rejected'),
    SimpleNamespace(returncode=0,stdout='',stderr=''),
    SimpleNamespace(returncode=0,stdout='MULTI_REUSE={}\nMULTI_REUSE={}\n',stderr='')])
def test_failed_or_missing_original_authentication_blocks_reuse(tmp_path,monkeypatch,result):
    original=tmp_path/'old';project=tmp_path/'new';root=tmp_path/'parent';root.mkdir()
    for directory in (original,project):
        (directory/'src').mkdir(parents=True);(directory/'src/kernel.py').write_text('')
    write_immutable_json(root/'campaign_spec.json',dict(project_dir=str(original),root=str(root),
        source_commit=source.DONOR,counts=c.COUNTS))
    monkeypatch.setattr(subprocess,'run',lambda args,**kw:SimpleNamespace(stdout='src/kernel.py\n') if args[0]=='git' else result)
    with pytest.raises(ValueError):source.authenticate(root/'campaign_spec.json',project)


def test_actual_donor_compatibility_exceptions_are_only_unrelated_sporc():
    project=Path(__file__).resolve().parents[1]
    changed=subprocess.run(['git','diff','--name-only','--diff-filter=MDR',source.DONOR,source.BASELINE,
        '--','src','scripts','sbatch'],cwd=project,check=True,capture_output=True,text=True).stdout.splitlines()
    assert set(changed)<=source.UNRELATED_SPORC
    assert all('luka_d066_embedding' in p for p in source.UNRELATED_SPORC)
    # Check a fresh process: unrelated tests may import SPORC modules at collection.
    subprocess.run([sys.executable,'-B','-c',
        'import sys; from hlt_classification.luka_fullstat_multiteacher import campaign,worker,submission; '
        'assert not any(n.startswith("hlt_classification.luka_d066_embedding") for n in sys.modules)'],
        cwd=project,check=True,capture_output=True,timeout=60)


@pytest.mark.parametrize('memory',[dict(peak_rss_bytes=0,gpu_peak_bytes=1),
    dict(peak_rss_bytes=180000*1024**2,gpu_peak_bytes=1),dict(peak_rss_bytes=1,gpu_peak_bytes=48*2**30)])
def test_unmeasured_or_excessive_memory_rejected(tmp_path,memory):
    s,*_=fixture(tmp_path,export=False)
    with pytest.raises(MemoryError):worker.limits(s,memory)


def test_real_installed_cuda_setup_and_full20_production_workers(tmp_path,monkeypatch):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable; not Oscar resource certification')
    s,tr,val,views=fixture(tmp_path)
    monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(worker,'execution',lambda *a:__import__('os').environ['SLURM_JOB_ID'])
    monkeypatch.setattr(worker,'metadata',lambda _: (dict(content_hash=tr.foundation_sha256,summaries=s['reuse']['summaries']),None,None))
    monkeypatch.setattr(worker,'data',lambda s,r,co,**kw:views[co] if co!='D000' else tr if r=='train' else val)
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:dict(setup='101',train_multi4='102'))
    monkeypatch.setattr(submission,'successful_setup',lambda *a:None)
    # Windows has no Linux resource counters; preserve actual CUDA peak measurement.
    monkeypatch.setattr(worker,'memory_snapshot',lambda _:dict(peak_rss_bytes=2**30,rss_bytes=2**30,
        gpu_peak_bytes=max(1,int(torch.cuda.max_memory_allocated()))))
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8');monkeypatch.setenv('SLURM_JOB_ID','101')
    admitted=worker.run(s,'setup')
    assert admitted['probe']['stress_steps']==3 and admitted['probe']['full_validation_rows']==len(val)
    assert not admitted['probe']['full_epoch_training_measured'] and admitted['technical_checks_passed']
    saved=Path(s['root'])/'setup/receipt.json';before=saved.read_bytes()
    bad=deepcopy(admitted);bad['probe']['estimated_train_minutes']=0
    saved.write_text(json.dumps(with_content_hash(bad)))
    with pytest.raises(ValueError):worker.validate_setup(s)
    saved.write_bytes(before)
    monkeypatch.setenv('SLURM_JOB_ID','102')
    assert worker.run(s,'train_multi4')['task_id']=='train_multi4'
    summary=worker.results(s)
    assert [r['node_id'] for r in summary['rows']]==['M0HLT','OFFLINE','U100_EMBED_W5','MULTI4_EMBED_W5']
    assert not summary['fresh_matched_control'] and not summary['final_test_accessed']
    report=load_json(Path(s['root'])/'train_multi4/training_report.json');assert report['passes']==20
    task=Path(s['root'])/'train_multi4/task_receipt.json'
    task.write_text(json.dumps(with_content_hash(dict(load_json(task),job_id='999'))))
    with pytest.raises(ValueError,match='publication identity'):worker.results(s)


def test_source_guard_and_required_files(monkeypatch):
    project=Path(__file__).resolve().parents[1]
    monkeypatch.setattr(source,'_source',lambda *a:dict(content_hash='a'*64))
    monkeypatch.setattr(subprocess,'run',lambda cmd,**kw:SimpleNamespace(stdout='\n'.join(source.REQUIRED)
        if 'ls-files' in cmd else '\n'.join(source.ADDITIONS|source.ENTRYPOINTS) if 'diff' in cmd else ''))
    assert source.current(project,'b'*40)['content_hash']=='a'*64
    for path in source.REQUIRED:assert (project/path).is_file(),path
    monkeypatch.setattr(subprocess,'run',lambda cmd,**kw:SimpleNamespace(stdout='\n'.join(source.REQUIRED)
        if 'ls-files' in cmd else 'src/old.py' if 'diff' in cmd else ''))
    with pytest.raises(ValueError):source.current(project,'b'*40)


def test_shell_cli_documentation_syntax():
    project=Path(__file__).resolve().parents[1]
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    if not bash.is_file():bash=shutil.which('bash')
    if not bash:pytest.skip('Bash unavailable')
    for path in ('scripts/queue_luka_fullstat_multiteacher.sh','sbatch/run_luka_fullstat_multiteacher.sh'):
        subprocess.run([str(bash),'-n',path],cwd=project,check=True,capture_output=True,timeout=30)
    book=(project/'docs/LUKA_FULLSTAT_MULTITEACHER_RUNBOOK.md').read_text()
    for block in re.findall(r'^```bash\n(.*?)^```',book,re.M|re.S):
        subprocess.run([str(bash),'-n'],input=block,check=True,text=True,capture_output=True,timeout=30)
    subprocess.run([sys.executable,'-B','scripts/luka_fullstat_multiteacher.py','--help'],
        cwd=project,check=True,capture_output=True,timeout=60)
    for path in source.REQUIRED:
        text=(project/path).read_text();assert all(line==line.rstrip() for line in text.splitlines()),path
        if path.endswith('.md'):
            for link in re.findall(r'\]\(([^)#]+)(?:#[^)]*)?\)',text):
                assert (project/path).parent.joinpath(link).is_file(),(path,link)

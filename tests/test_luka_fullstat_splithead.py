"""FullStat20 split heads: real kernels, immutable joins, three production workers."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import os
import re
import shutil
import subprocess
import sys
import numpy as np
import pytest
import torch
from test_luka_embedding_aux import tiny, threads
from test_luka_fullstat_hybrid_embedding import fixture as parent_fixture, originals
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json, with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import _forward, predict
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_fullstat_hybrid_embedding import contracts as old_c, worker as old_w
from hlt_classification.luka_fullstat_splithead import contracts as c, source, campaign, banks, kd_banks, model, training, worker, submission


def fixture(path, export=False):
    old, caches = parent_fixture(path, export=export)
    # Synthetic selected parent for local fixtures only. Real reuse replays the
    # complete old worker receipts/accounting, never these fixture shortcuts.
    directory = Path(old['root'])/'train_UE100'; directory.mkdir()
    if export:
        originals(old, caches)
        native = DelphesParticleTransformer()
        save_state(directory/'selected.pt', native.state_dict())
        write_immutable_json(directory/'training_report.json', old_c.artifact('TRAINING_REPORT',
            node=old['nodes']['UE100'], scientific_fit=True, technical=False,
            selected_state='native_backbone_only'))
        for kind in ('embedding', 'kd'):
            old_w.export_target(old, 'UE100', kind, caches['train']['U100'], device='cpu')
    refs = old_w.banks.refs_for(old, 'UE100') if export else old['offline_refs']
    root = path/'splithead'; root.mkdir()
    imported = {'OFFLINE':dict(embedding=old['offline_bank']), 'M0HLT':{}, 'UE100':{}}
    if export:
        imported['OFFLINE']['kd'] = file_ref(Path(old['root'])/'logit_targets/OFFLINE/manifest.json')
        imported['M0HLT']['embedding'] = file_ref(Path(old['root'])/'targets/M0HLT/manifest.json')
        for kind, folder in [('embedding','targets'), ('kd','logit_targets')]:
            imported['UE100'][kind] = file_ref(Path(old['root'])/folder/'UE100/manifest.json')
    else:
        imported['OFFLINE']['kd'] = {}
        imported['M0HLT']['embedding'] = {}
        imported['UE100'] = dict(embedding={}, kd={})
    r = old['reuse']
    reuse = c.artifact('REUSE', parent_spec=file_ref(Path(old['root'])/'campaign_spec.json'),
        source_commit=source.DONOR, counts=dict(c.COUNTS), accepted=r,
        anchor=old['anchor'], references=old['references'],
        external_refs=dict(OFFLINE=old['offline_refs'], M0HLT=old['references']['M0HLT'], UE100=refs),
        imported_banks=imported, base_node=r['base_node'], execution_site=r['execution_site'],
        environment=r['environment'], gpu=r['gpu'], summaries=r['summaries'],
        parent_task_minutes=old['task_minutes'], jobs=dict(train_UE000='101', train_UE100='102'),
        protected_roots=[old['root']])
    spec = c.artifact('SPEC', parents=dict(source='c'*64, reuse=reuse['content_hash']),
        root=str(root), project_dir='/project', source_commit='f'*40, source_snapshot=dict(content_hash='c'*64),
        reuse=reuse, available_result_bytes=16*2**30, **campaign.policy(reuse))
    spec = with_content_hash(dict(spec, counts=old['counts']))
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec, caches


def mock_execution(spec, caches, monkeypatch):
    monkeypatch.setattr(campaign, 'check', lambda _: None)
    monkeypatch.setattr(worker, 'execution', lambda *a: os.environ['SLURM_JOB_ID'])
    monkeypatch.setattr(worker, 'metadata', lambda _: (dict(content_hash=spec['anchor']['evidence']['prepared'],
        summaries=spec['reuse']['summaries']), None, None))
    monkeypatch.setattr(worker, 'data', lambda s,r,co,**kw: caches[r][co])
    monkeypatch.setattr(submission, 'accepted_jobs', lambda _: {'train_'+s: str(201+i) for i,s in enumerate(c.STAGES)})
    monkeypatch.setattr(submission, 'successful', lambda *a: None)
    monkeypatch.setattr(worker, 'memory_snapshot', lambda _: dict(peak_rss_bytes=2**30,
        gpu_peak_bytes=max(1,int(torch.cuda.max_memory_allocated()))))


def test_exact_fullstat_registration_and_source_isolation(tmp_path):
    s, _ = fixture(tmp_path)
    expected = dict(DE066=['OFFLINE','UE100','M0HLT'], DE033=['OFFLINE','UE100','DE066','M0HLT'],
                    DE000=['OFFLINE','UE100','DE066','DE033'])
    assert list(c.STAGES) == list(expected)
    for stage, ts in expected.items():
        r = c.recipe(stage); n = s['nodes'][stage]
        assert c.teachers(stage) == r['teacher_order'] == ts
        assert r['weights'] == {t: 5/len(ts) for t in ts}
        assert r['logit_weights'] == {t: .5/len(ts) for t in ts}
        assert r['ce_coefficient'] == r['logit_total'] == .5 and r['temperature'] == 2
        assert r['epochs'] == r['base']['maximum_passes'] == 20
        assert r['base']['warmup_passes'] == 2 and r['batch_size'] == 128
        assert n['teacher'] is n['context_coordinate'] is None
        assert n['deployable'] is (stage == 'DE000') and not r['warm_start']
    assert s['preflight_jobs'] == s['setup_jobs'] == 0 and s['new_scientific_fits'] == 3
    assert c.COUNTS['train'] == 3000000 and c.COUNTS['validation'] == 500000
    assert c.COUNTS['final_test'] == 1500000
    assert not s['final_test_accessed'] and s['final_test_sealed']
    assert source.compatible_changes(source.ADDITIONS, source.ADDITIONS)
    assert not source.compatible_changes({'src/hlt_classification/luka_fullstat/recipe.py'}, set())
    assert "worker.completed(s,'UE100')" in source.REPLAY and 'worker.results' not in source.REPLAY
    assert "train_DE066" not in source.REPLAY and "luka_auxiliary_logits" not in source.REPLAY


@pytest.mark.parametrize('stage', c.STAGES)
def test_exact_loss_main_head_isolation_and_detached_teachers(stage):
    x = torch.randn(5,11,requires_grad=True); y = torch.arange(5)
    logits = {t:torch.randn(5,11,requires_grad=True) for t in c.teachers(stage)}
    q = {t:torch.softmax(torch.randn(5,11)/2,-1).requires_grad_(True) for t in logits}
    pred = {t:torch.randn(5,128,requires_grad=True) for t in logits}
    targets = {t:torch.randn(5,128,requires_grad=True) for t in logits}
    loss, ce, kd, emb = training.objective(x,y,logits,pred,q,targets,stage=stage)
    expected = .5*ce + 5/len(logits)*sum(emb.values()) + .5/len(logits)*sum(kd.values())
    torch.testing.assert_close(loss,expected)
    main_grad = torch.autograd.grad(.5*ce,x,retain_graph=True)[0]
    loss.backward(); torch.testing.assert_close(x.grad,main_grad,rtol=0,atol=0)
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for d in (logits,pred) for v in d.values())
    assert all(v.grad is None for d in (q,targets) for v in d.values())
    for t in logits:
        expected_kd = 4*torch.nn.functional.kl_div((logits[t]/2).log_softmax(-1),q[t],reduction='batchmean')
        torch.testing.assert_close(kd[t],expected_kd)
    with pytest.raises(ValueError): training.objective(x,y,{},pred,q,targets,stage=stage)


@pytest.mark.parametrize('bf16', [False, True])
def test_full_native_parity_rng_and_both_auxiliary_losses_reach_encoder(tmp_path,bf16):
    if not torch.cuda.is_available(): pytest.skip('CUDA needed')
    s, cs = fixture(tmp_path)
    for stage in c.STAGES:
        node = s['nodes'][stage]
        torch.manual_seed(node['initialization_seed']); native = DelphesParticleTransformer().cuda().eval()
        rng = torch.get_rng_state().clone()
        student = model.new_model(node).cuda().eval(); student.capture=True
        assert torch.equal(rng,torch.get_rng_state())
        assert all(torch.equal(v,native.state_dict()[k].cpu()) for k,v in student.native_state().items())
        raw=cs['train'][c.STAGES[stage]].batch(np.arange(4)); y=torch.from_numpy(raw['labels']).cuda()
        with parity_backend('cuda'):
            out=_forward(student,raw,'cuda',bf16=bf16)
            with torch.no_grad(): reference=_forward(native,raw,'cuda',bf16=bf16)
            torch.testing.assert_close(out,reference,rtol=0,atol=0)
            ps=student.auxiliary_logits; es=student.predictions
            q={t:torch.softmax(torch.randn(4,11,device='cuda'),-1) for t in ps}
            z={t:torch.randn(4,128,device='cuda') for t in ps}
            loss,ce,kd,emb=training.objective(out,y,ps,es,q,z,stage=stage)
            head=tuple(student.mod.fc.parameters())
            head_grad=torch.autograd.grad(.5*ce,head,retain_graph=True)
            encoder=[p for name,p in student.named_parameters() if name.startswith('mod.') and not name.startswith('mod.fc.')]
            for term in (sum(kd.values()),sum(emb.values())):
                grads=torch.autograd.grad(term,encoder,retain_graph=True,allow_unused=True)
                assert any(g is not None and g.abs().sum()>0 for g in grads)
            loss.backward()
            for p,g in zip(head,head_grad): torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
        assert not any(k.startswith(('predictors.','logit_heads.')) for k in student.native_state())
        del native,student,out,reference,loss,ce,kd,emb,ps,es,head,encoder,grads,head_grad


def test_imported_banks_exact_roles_corruption_and_new_m0_logits(tmp_path,monkeypatch):
    tiny(monkeypatch); s,cs=fixture(tmp_path,export=True)
    worker.export_target(s,'M0HLT','kd',cs['train']['D000'],device='cpu')
    tr=cs['train']['D066']; bank=banks.publish(s,'DE066',tr)
    assert set(bank.targets)==set(bank.probabilities)==set(c.teachers('DE066'))
    for t in ('OFFLINE','UE100'):
        ref=s['imported_banks'][t]['kd']; saved=load_json(ref['path'])
        a=load_npz_arrays(Path(ref['path']).parent/'probabilities.npz')
        assert saved['contract'].startswith('LUKA_FULLSTAT_HYBRID_EMBEDDING_')
        np.testing.assert_array_equal(bank.probabilities[t],a['probabilities'])
    for role in ('validation','final_test'):
        bad=deepcopy(tr); bad.role=role
        with pytest.raises(PermissionError): banks.Bundle(s,'DE066',bad)
    bad=deepcopy(tr); bad.identities=bad.identities[::-1]
    with pytest.raises(ValueError): banks.Bundle(s,'DE066',bad)
    bad=deepcopy(tr); bad.labels=bad.labels[::-1]
    with pytest.raises(ValueError): banks.Bundle(s,'DE066',bad)
    target=Path(s['imported_banks']['UE100']['kd']['path']).parent/'probabilities.npz'
    target.write_bytes(target.read_bytes()+b'corrupt')
    with pytest.raises(ValueError): banks.Bundle(s,'DE066',tr)


def test_three_job_dry_live_authorization_dependencies_and_no_duplicate(tmp_path,monkeypatch):
    s,_=fixture(tmp_path)
    monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(submission,'check_site',lambda _:None)
    plan=submission.submit(s)
    assert plan['jobs']==plan['scientific_fits']==3 and plan['setup_jobs']==plan['preflight_jobs']==0
    assert [x['task_id'] for x in plan['commands']]==['train_'+x for x in c.STAGES]
    with pytest.raises(PermissionError): submission.submit(s,execute=True,plan_hash=plan['content_hash'])
    calls=[]
    def fake(args,**kw):
        calls.append(args); return SimpleNamespace(returncode=0,stdout=str(200+len(calls))+'\n',stderr='')
    monkeypatch.setattr(subprocess,'run',fake)
    submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert submission.accepted_jobs(s)==dict(train_DE066='201',train_DE033='202',train_DE000='203')
    assert '--dependency=afterok:201:202' in calls[2]
    assert all('--no-requeue' in x and '--kill-on-invalid-dep=yes' in x for x in calls)
    with pytest.raises(FileExistsError): submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==3


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual installed-Weaver CUDA required')
def test_full_architecture_three_fixed20_workers_and_native_descendants(tmp_path,monkeypatch):
    s,cs=fixture(tmp_path,export=True); mock_execution(s,cs,monkeypatch)
    inherited={t:{k:file_ref(v['path']) for k,v in kinds.items()} for t,kinds in s['imported_banks'].items()}
    for i,stage in enumerate(c.STAGES):
        monkeypatch.setenv('SLURM_JOB_ID',str(201+i))
        assert worker.completed(s,stage) is None
        rec=worker.run(s,'train_'+stage)
        assert rec==worker.completed(s,stage)
        report=load_json(Path(s['root'])/('train_'+stage)/'training_report.json')
        assert len(report['validation_history'])==20
        state=torch.load(Path(s['root'])/('train_'+stage)/'selected.pt',weights_only=True)
        assert not any(k.startswith(('predictors.','logit_heads.')) for k in state)
        native=DelphesParticleTransformer().cuda(); native.load_state_dict(state,strict=True)
        if stage!='DE000':
            a=load_npz_arrays(Path(s['root'])/'logit_targets'/stage/'probabilities.npz')
            np.testing.assert_array_equal(a['probabilities'],predict(native,cs['train'][c.STAGES[stage]],device='cuda',temperature=2.,batch_size=128))
        del native
        summary=worker.results(s)
        assert summary['pending']==list(c.STAGES)[i+1:]
        with pytest.raises(FileExistsError): worker.run(s,'train_'+stage)
    assert summary['rows'][-1]['input']=='native_HLT' and not summary['final_test_accessed']
    assert {t:{k:file_ref(v['path']) for k,v in kinds.items()} for t,kinds in s['imported_banks'].items()}==inherited
    assert not (Path(s['root'])/'targets/M0HLT').exists()
    assert not (Path(s['root'])/'targets/UE100').exists()
    assert not (Path(s['root'])/'logit_targets/DE000').exists()
    bad=Path(s['root'])/'targets/DE066/targets.npz'; bad.write_bytes(bad.read_bytes()+b'corrupt')
    with pytest.raises(ValueError): worker.completed(s,'DE000')


def test_poor_finite_science_does_not_gate_and_rng_restarts(tmp_path,monkeypatch):
    tiny(monkeypatch); s,cs=fixture(tmp_path,export=True)
    worker.export_target(s,'M0HLT','kd',cs['train']['D000'],device='cpu')
    tr,val=cs['train']['D066'],cs['validation']['D066']; bank=banks.publish(s,'DE066',tr)
    monkeypatch.setattr(training,'predict',lambda *a,**kw:np.full((len(val),11),1/11,np.float32))
    torch.manual_seed(777); report,state=training.train(tr,val,bank=bank,node=s['nodes']['DE066'],device='cpu')
    worker.report_valid(report,s,'DE066',bank.manifest['content_hash'])
    assert report['selected_pass']==1 and all(x['validation']['macro_ovr_auc']==.5 for x in report['validation_history'])
    torch.manual_seed(888); again,other=training.train(tr,val,bank=bank,node=s['nodes']['DE066'],device='cpu')
    assert all(torch.equal(v,other[k]) for k,v in state.items())
    for field in ('weighted_ce_loss','total_loss'):
        bad=deepcopy(report); bad['validation_history'][0][field]+=.1
        with pytest.raises(ValueError): worker.report_valid(with_content_hash(bad),s,'DE066',bank.manifest['content_hash'])


@pytest.mark.parametrize('key,value', [('counts',{}), ('external_refs',{}), ('jobs',{}), ('parent_task_minutes',{'DE066':2880})])
def test_bad_reuse_fails_closed(tmp_path,key,value):
    s,_=fixture(tmp_path); bad=with_content_hash(dict(s['reuse'],**{key:value}))
    with pytest.raises((ValueError,KeyError)): campaign.policy(bad)


def test_source_replay_in_original_tree_and_changed_runtime_refused(tmp_path,monkeypatch):
    original=tmp_path/'old'; project=tmp_path/'new'; root=tmp_path/'parent'; root.mkdir()
    for d in (original,project):
        (d/'src').mkdir(parents=True); (d/'src/kernel.py').write_text('# native\n')
    write_immutable_json(root/'campaign_spec.json',dict(project_dir=str(original),root=str(root),source_commit=source.DONOR,counts=c.COUNTS))
    def replay(args,**kw):
        if args[0]=='git': return SimpleNamespace(stdout='src/kernel.py\n')
        assert kw['cwd']==original and kw['env']['PYTHONPATH']==str(original/'src')
        assert args[-2]==source.REPLAY
        return SimpleNamespace(returncode=0,stdout='SPLITHEAD_REUSE={}\n',stderr='')
    monkeypatch.setattr(subprocess,'run',replay)
    c.validate(source.authenticate(root/'campaign_spec.json',project),'REUSE')
    (project/'src/kernel.py').write_text('# changed\n')
    with pytest.raises(ValueError,match='Changed donor'): source.authenticate(root/'campaign_spec.json',project)


@pytest.mark.parametrize('out,code',[('',1),('',0),('SPLITHEAD_REUSE={}\nSPLITHEAD_REUSE={}\n',0)])
def test_missing_failed_or_ambiguous_replay_blocks_reuse(tmp_path,monkeypatch,out,code):
    original=tmp_path/'old'; project=tmp_path/'new'; root=tmp_path/'parent'; root.mkdir()
    for d in (original,project):
        (d/'src').mkdir(parents=True); (d/'src/kernel.py').write_text('')
    write_immutable_json(root/'campaign_spec.json',dict(project_dir=str(original),root=str(root),source_commit=source.DONOR,counts=c.COUNTS))
    monkeypatch.setattr(subprocess,'run',lambda args,**kw:SimpleNamespace(stdout='src/kernel.py\n')
        if args[0]=='git' else SimpleNamespace(returncode=code,stdout=out,stderr='rejected'))
    with pytest.raises(ValueError): source.authenticate(root/'campaign_spec.json',project)


def test_additive_source_guard_and_existing_donor_unchanged(monkeypatch):
    project=Path(__file__).resolve().parents[1]
    changed=subprocess.run(['git','diff','--name-only','--diff-filter=MDR',source.DONOR,source.BASELINE,
        '--','src','scripts','sbatch'],cwd=project,check=True,capture_output=True,text=True).stdout
    assert not changed
    compile(source.REPLAY,'replay','exec')
    monkeypatch.setattr(source,'_source',lambda *a:dict(content_hash='a'*64))
    monkeypatch.setattr(subprocess,'run',lambda args,**kw:SimpleNamespace(stdout='\n'.join(source.REQUIRED)
        if 'ls-files' in args else '\n'.join(source.ADDITIONS|source.ENTRYPOINTS) if 'diff' in args else ''))
    assert source.current(project,'b'*40)['content_hash']=='a'*64
    monkeypatch.setattr(subprocess,'run',lambda args,**kw:SimpleNamespace(stdout='\n'.join(source.REQUIRED)
        if 'ls-files' in args else 'src/old.py' if 'diff' in args else ''))
    with pytest.raises(ValueError): source.current(project,'b'*40)


def test_ambiguous_submission_no_retry(tmp_path,monkeypatch):
    s,_=fixture(tmp_path)
    monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(submission,'check_site',lambda _:None)
    p=submission.submit(s); calls=[]
    def ambiguous(args,**kw):
        calls.append(args); return SimpleNamespace(returncode=0,stdout='ambiguous\n',stderr='')
    monkeypatch.setattr(subprocess,'run',ambiguous)
    with pytest.raises(Exception): submission.submit(s,execute=True,plan_hash=p['content_hash'],authorization=c.AUTHORIZATION)
    with pytest.raises(FileExistsError): submission.submit(s,execute=True,plan_hash=p['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==1


def test_failed_accounting_or_missing_receipt_blocks_descendant(tmp_path,monkeypatch):
    s,_=fixture(tmp_path)
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:dict(train_DE066='201',train_DE033='202'))
    monkeypatch.setattr(submission.getpass,'getuser',lambda:'rlyang')
    valid='201|lukafss_train_DE066|rlyang|default|COMPLETED|0:0\n'
    for text in ('',valid.replace('COMPLETED','FAILED'),valid.replace('0:0','1:0'),valid.replace('rlyang','other')):
        monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=text))
        with pytest.raises(ValueError): submission.successful(s,'train_DE066','201')
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=valid))
    submission.successful(s,'train_DE066','201')
    monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(worker,'execution',lambda *a:'202')
    monkeypatch.setattr(worker,'fit',lambda *a,**kw:pytest.fail('No fit without predecessor'))
    with pytest.raises(ValueError,match='incomplete'): worker.run(s,'train_DE033')
    assert not (Path(s['root'])/'train_DE033_attempt').exists()


@pytest.mark.parametrize('damage',['negative','nonfinite','dtype','normalization','width'])
def test_probability_validation_fail_closed(damage):
    p=np.full((3,11),1/11,np.float32)
    if damage=='negative': p[0,0]=-.1
    if damage=='nonfinite': p[0,0]=np.nan
    if damage=='dtype': p=p.astype(np.float64)
    if damage=='normalization': p*=2
    if damage=='width': p=p[:,:-1]
    with pytest.raises(ValueError): kd_banks.validate_probabilities(p,3)


@pytest.mark.parametrize('stage',['DE066','DE033'])
def test_old_same_named_teacher_report_cannot_replace_new_selected_model(tmp_path,stage):
    s,_=fixture(tmp_path)
    refs=dict(s['external_refs']['OFFLINE'])
    path=tmp_path/'old_same_name.json'
    write_immutable_json(path,old_c.artifact('TRAINING_REPORT',node=dict(node_id=stage),
        scientific_fit=True,technical=False,selected_state='native_backbone_only'))
    refs['report']=file_ref(path)
    with pytest.raises(ValueError): banks.check_selected(DelphesParticleTransformer(),refs,stage)


@pytest.mark.parametrize('memory',[dict(peak_rss_bytes=0,gpu_peak_bytes=1),
    dict(peak_rss_bytes=180000*1024**2,gpu_peak_bytes=1),dict(peak_rss_bytes=1,gpu_peak_bytes=48*2**30)])
def test_no_memory_fallback(tmp_path,memory):
    s,_=fixture(tmp_path)
    with pytest.raises(MemoryError): worker.limits(s,memory)


def test_shell_cli_docs_syntax_and_new_runtime_inventory():
    project=Path(__file__).resolve().parents[1]
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    if not bash.is_file(): bash=shutil.which('bash')
    if not bash: pytest.skip('Bash unavailable')
    for path in ('scripts/queue_luka_fullstat_splithead.sh','sbatch/run_luka_fullstat_splithead.sh'):
        subprocess.run([str(bash),'-n',path],cwd=project,check=True,capture_output=True,timeout=30)
    book=(project/'docs/LUKA_FULLSTAT_SPLITHEAD_RUNBOOK.md').read_text()
    for block in re.findall(r'^```bash\n(.*?)^```',book,re.M|re.S):
        subprocess.run([str(bash),'-n'],input=block,check=True,text=True,capture_output=True,timeout=30)
    subprocess.run([sys.executable,'-B','scripts/luka_fullstat_splithead.py','--help'],
        cwd=project,check=True,capture_output=True,timeout=60)
    for path in source.REQUIRED:
        text=(project/path).read_text()
        assert all(line==line.rstrip() for line in text.splitlines()),path
        if path.endswith('.md'):
            for link in re.findall(r'\]\(([^)#]+)(?:#[^)]*)?\)',text):
                assert (project/path).parent.joinpath(link).is_file(),(path,link)
        elif path.startswith('src/'):
            assert 'Fresh_check' not in text and 'luka_auxiliary_logits' not in text

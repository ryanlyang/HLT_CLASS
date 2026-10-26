"""Post-export failure: native readback, no repeat fit/export, exact DE000 DAG."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import ast
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
from test_luka_fullstat_splithead_recovery import saved_fit, create_recovery
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json, with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullstat_splithead import contracts as c, worker, training, banks, kd_banks, campaign, submission
from hlt_classification.luka_fullstat_splithead import recovery as r, finalize as f, finalize_source as fs


def exported_failure(tmp_path, monkeypatch, *, gpu=False):
    """Real saved models and exports; scheduler/source are synthetic local fixtures."""
    device = 'cuda' if gpu else 'cpu'
    initial, _, caches = saved_fit(tmp_path, monkeypatch, device=device)
    old = create_recovery(tmp_path, initial)
    r.claim(old, submission.render(old))
    monkeypatch.setattr(submission, 'accepted_jobs', lambda spec: fs.JOBS if spec['root']==old['root'] else
        dict(finalize_DE033='8200001', train_DE000='8200002'))
    monkeypatch.setattr(submission, 'successful', lambda *a: None)
    monkeypatch.setenv('SLURM_JOB_ID', fs.JOBS['recover_DE066'])
    if gpu:
        de066 = worker.run(old, 'recover_DE066')
    else:
        # CPU fixture of a completed preceding recovery; the GPU integration
        # below executes the actual production recovery, including its readback.
        root = Path(old['root']); (root/'recovery_DE066').mkdir()
        parents = dict(spec=old['content_hash'], original_fit=old['recovery']['evidence']['parents']['fit'],
            evidence=old['recovery']['evidence']['content_hash'])
        write_immutable_json(root/'recovery_DE066/readback.json', c.artifact('RECOVERY_READBACK',
            parents=parents, rows=old['counts']['validation'], selected_refs=old['external_refs']['DE066'],
            native_prediction_bytes_equal=True))
        for kind in ('embedding', 'kd'):
            worker.export_target(old, 'DE066', kind, caches['train']['D066'], device=device)
        (root/'train_DE066').mkdir()
        write_immutable_json(root/'train_DE066/task_receipt.json', c.artifact('RECOVERY_TASK', parents=parents,
            task_id='recover_DE066', stage='DE066', source_commit=old['source_commit'],
            job_id=fs.JOBS['recover_DE066'], training_repeated=False,
            outputs=[file_ref(root/p, root=root) for p in r.recovered_outputs()]))
        de066 = r.completed_de066(old)
    root = Path(old['root'])
    monkeypatch.setenv('SLURM_JOB_ID', fs.JOBS['train_DE033'])
    if gpu:
        # Reproduce the precise failure boundary: fit + both exports finished,
        # but the final source/environment replay prevents TASK publication.
        real_check = campaign.check
        def fail_after_exports(s):
            if s['root']==old['root'] and (root/'logit_targets/DE033/manifest.json').is_file():
                raise ValueError('synthetic post-export Python startup failure')
            return real_check(s)
        with monkeypatch.context() as patch:
            patch.setattr(campaign, 'check', fail_after_exports)
            with pytest.raises(ValueError, match='post-export Python'):
                worker.run(old, 'train_DE033')
    else:
        tr, val = caches['train']['D033'], caches['validation']['D033']
        bank = banks.publish(old, 'DE033', tr)
        attempt = root/'train_DE033_attempt'; attempt.mkdir()
        write_immutable_json(attempt/'startup.json', worker.startup(old, 'DE033', tr, val, bank, device=device))
        worker.fit(root/'train_DE033', old, 'DE033', tr, val, bank, device=device)
        for kind in ('embedding', 'kd'):
            worker.export_target(old, 'DE033', kind, tr, device=device)
    (root/'train_DE033-7938792.out').write_text(
        "stage=DE033 epoch=20/20 auc=0.9\nOriginal campaign authentication failed:\n"
        "Fatal Python error: init_fs_encoding: can't find encoding\n")
    monkeypatch.setattr(fs, 'original', lambda _: (old, fs.JOBS, de066))
    monkeypatch.setattr(fs, 'current', lambda *a: dict(content_hash='f'*64))
    monkeypatch.setattr(f, 'accounting', lambda *a: dict(failed='7938792', cancelled='7938793'))
    monkeypatch.setattr(f, 'memory_snapshot', worker.memory_snapshot)
    return old, de066, caches


@pytest.fixture
def failed(tmp_path, monkeypatch):
    tiny(monkeypatch)
    return exported_failure(tmp_path, monkeypatch)


def create(tmp_path, old):
    return f.create(tmp_path/'finalized', Path(old['root'])/'campaign_spec.json',
        tmp_path/'finalsource', 'f'*40, quota_gib=2)


def test_readonly_audit_and_exact_recipe_reuse(failed, tmp_path):
    old, de066, _ = failed
    original = {p: file_ref(p) for p in Path(old['root']).rglob('*') if p.is_file()}
    s = create(tmp_path, old)
    assert campaign.check(s) == old
    assert s['recipes'] == old['recipes'] and s['nodes']==old['nodes']
    assert s['counts']==old['counts'] and s['preflight_jobs']==0
    assert s['new_scientific_fits']==1 and s['reused_scientific_fits']==2
    assert s['runtime_prefix']==fs.RUNTIME and s['available_result_bytes']==2*2**30
    for stage in ('DE066', 'DE033'):
        for module, kind in ((banks, 'embedding'), (kd_banks, 'kd')):
            assert module.manifest_ref(s, stage)['path']==str(Path(old['root'])/
                ('targets' if kind=='embedding' else 'logit_targets')/stage/'manifest.json')
    assert original == {p: file_ref(p) for p in original}
    assert worker.completed(s, 'DE033') is None
    assert not (Path(old['root'])/'finalize_DE033_claim_v1').exists()


@pytest.mark.parametrize('relative', ['train_DE033/selected.pt', 'train_DE033/training_report.json',
    'train_DE033/validation_predictions.npz', 'train_DE033/receipt.json', 'bundles/DE033.json',
    'train_DE033_attempt/startup.json', 'targets/DE033/targets.npz', 'targets/DE033/manifest.json',
    'logit_targets/DE033/probabilities.npz', 'logit_targets/DE033/manifest.json'])
def test_corrupt_preexisting_evidence_blocks_finalization(failed, relative):
    old, done, _ = failed
    p = Path(old['root'])/relative
    p.write_bytes(p.read_bytes()+b'corrupt')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        f.audit(old, fs.JOBS, done)


@pytest.mark.parametrize('relative', ['train_DE033/task_receipt.json', 'train_DE000',
    'train_DE000_attempt', 'bundles/DE000.json'])
def test_already_started_or_completed_stage_blocks_finalization(failed, relative):
    old, done, _ = failed
    (Path(old['root'])/relative).mkdir(parents=True)
    with pytest.raises(ValueError, match='Unexpected'):
        f.audit(old, fs.JOBS, done)


def test_failure_signature_and_quota_cannot_be_bypassed(failed, tmp_path):
    old, done, _ = failed
    with pytest.raises(ValueError, match='at least 2 GiB'):
        f.create(tmp_path/'blocked', Path(old['root'])/'campaign_spec.json', tmp_path/'source', 'f'*40, quota_gib=1)
    assert not (tmp_path/'blocked').exists()
    (Path(old['root'])/'train_DE033-7938792.out').write_text('ordinary training failure')
    with pytest.raises(ValueError, match='post-export'):
        f.audit(old, fs.JOBS, done)


@pytest.mark.parametrize('field', ['runtime_prefix', 'recipes', 'counts', 'task_minutes'])
def test_policy_tampering_remains_rejected_after_rehash(failed, tmp_path, field):
    old, _, _ = failed
    s=create(tmp_path,old)
    changed=deepcopy(s)
    if field=='runtime_prefix': changed[field]='/wrong/runtime'
    elif field=='recipes': changed[field]['DE000']['ce_coefficient']=1.
    elif field=='counts': changed[field]['validation']+=1
    else: changed[field]['DE000']+=1
    changed=with_content_hash(changed)
    (Path(s['root'])/'campaign_spec.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='recipe/runtime changed'): campaign.check(changed)


def test_storage_exception_is_scoped_to_finalization(monkeypatch, tmp_path):
    monkeypatch.setattr(campaign.shutil,'disk_usage',lambda _:SimpleNamespace(free=3*2**30))
    s=dict(root=str(tmp_path), available_result_bytes=2*2**30, contract=f.CONTRACT)
    campaign.storage(s)
    for contract in (r.RECOVERY_CONTRACT,'LUKA_FULLSTAT_SPLITHEAD_SPEC/v1'):
        with pytest.raises(OSError,match='16 GiB'): campaign.storage(dict(s,contract=contract))
    monkeypatch.setattr(campaign.shutil,'disk_usage',lambda _:SimpleNamespace(free=2**30))
    with pytest.raises(OSError,match='2 GiB'): campaign.storage(s)


def test_only_two_jobs_authorization_and_duplicate_claim(failed, tmp_path, monkeypatch):
    old, _, _ = failed
    s = create(tmp_path, old)
    monkeypatch.setattr(submission, 'check_site', lambda _: None)
    calls = []
    def fake(args, **kw):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=str(8200000+len(calls))+'\n', stderr='')
    monkeypatch.setattr(subprocess, 'run', fake)
    plan = submission.submit(s)
    assert plan['jobs']==2 and plan['scientific_fits']==1 and plan['preflight_jobs']==0
    assert [x['task_id'] for x in plan['commands']]==['finalize_DE033', 'train_DE000']
    assert not calls
    with pytest.raises(PermissionError):
        submission.submit(s, execute=True, plan_hash=plan['content_hash'], authorization=r.AUTHORIZATION)
    submission.submit(s, execute=True, plan_hash=plan['content_hash'], authorization=f.AUTHORIZATION)
    assert len(calls)==2 and '--job-name=lukafssf_finalize_DE033' in calls[0]
    assert '--dependency=afterok:8200001' in calls[1]
    assert '--job-name=lukafssf_train_DE000' in calls[1]
    assert all('793879' not in ' '.join(call) for call in calls)
    f.verify_claim(s)
    with pytest.raises(FileExistsError): f.claim(dict(s, root=str(tmp_path/'different')), plan)
    with pytest.raises(FileExistsError):
        submission.submit(s, execute=True, plan_hash=plan['content_hash'], authorization=f.AUTHORIZATION)


def test_old_submission_plans_remain_byte_semantically_identical(failed):
    old, _, _=failed
    project=Path(__file__).resolve().parents[1]
    previous=subprocess.run(['git','show',fs.BASELINE+':src/hlt_classification/luka_fullstat_splithead/submission.py'],
        cwd=project,check=True,capture_output=True,text=True).stdout
    render=next(n for n in ast.parse(previous).body if isinstance(n,ast.FunctionDef) and n.name=='render')
    namespace=dict(submission.__dict__)
    exec(compile(ast.Module(body=[render],type_ignores=[]),'original_render','exec'),namespace)
    for spec in (old,load_json(old['recovery']['parent_spec']['path'])):
        assert submission.render(spec)==namespace['render'](spec)


@pytest.mark.parametrize('bad', ['none', 'owner', 'account', 'name', 'state', 'exit', 'started', 'duplicate', 'missing', 'jobs'])
def test_exact_old_accounting(monkeypatch, bad):
    rows = [[j, 'lukafssr_'+t, 'rlyang', 'default',
        'COMPLETED' if t=='recover_DE066' else 'FAILED' if t=='train_DE033' else 'CANCELLED',
        '1:0' if t=='train_DE033' else '0:0', '00:00:00' if t=='train_DE000' else '01:00:00']
        for t,j in fs.JOBS.items()]
    if bad in ('owner','account','name','state','exit','started'):
        rows[2][dict(owner=2,account=3,name=1,state=4,exit=5,started=6)[bad]]='wrong'
    if bad=='duplicate': rows.append(rows[0])
    if bad=='missing': rows.pop()
    monkeypatch.setattr(f.getpass, 'getuser', lambda:'rlyang')
    monkeypatch.setattr(subprocess, 'run', lambda *a,**kw:SimpleNamespace(stdout='\n'.join('|'.join(x) for x in rows)))
    jobs = dict(fs.JOBS, train_DE033='wrong') if bad=='jobs' else fs.JOBS
    if bad=='none':
        assert len(f.accounting(dict(execution_site=dict(account='default')), jobs))==3
    else:
        with pytest.raises(ValueError): f.accounting(dict(execution_site=dict(account='default')), jobs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual installed-Weaver GPU required')
def test_actual_installed_weaver_post_export_recovery_and_full20_DE000(tmp_path, monkeypatch):
    old, done, caches = exported_failure(tmp_path, monkeypatch, gpu=True)
    original = {p: file_ref(p) for p in Path(old['root']).rglob('*') if p.is_file()}
    s = create(tmp_path, old)
    f.claim(s, submission.render(s))
    monkeypatch.setenv('SLURM_JOB_ID', '8200001')
    def forbidden(*a, **kw): pytest.fail('Finalization must not train, initialize a student or regenerate targets')
    with monkeypatch.context() as patch:
        for module, name in ((worker,'fit'), (worker,'startup'), (training,'train'), (worker,'export_target'),
                             (banks,'export'), (kd_banks,'export')):
            patch.setattr(module, name, forbidden)
        result = worker.run(s, 'finalize_DE033')
    assert result == worker.completed(s, 'DE033')
    assert result['training_repeated'] is False and result['targets_regenerated'] is False
    assert worker.completed(s, 'DE066')
    assert not (Path(s['root'])/'targets').exists()
    assert not (Path(s['root'])/'train_DE033/selected.pt').exists()
    assert original == {p:file_ref(p) for p in original}
    with pytest.raises(FileExistsError): worker.run(s, 'finalize_DE033')
    monkeypatch.setenv('SLURM_JOB_ID', '8200002')
    worker.run(s, 'train_DE000')
    assert worker.completed(s, 'DE000')
    assert worker.results(s)['pending']==[]
    assert original == {p:file_ref(p) for p in original}
    p = Path(old['root'])/'logit_targets/DE033/probabilities.npz'
    p.write_bytes(p.read_bytes()+b'corrupt')
    with pytest.raises(ValueError): worker.completed(s, 'DE000')


@pytest.mark.parametrize('failure', ['none','subprocess','missing','duplicate','spec','jobs','ancestor'])
def test_pinned_parent_replay_uses_replacement_interpreter(tmp_path, monkeypatch, failure):
    project=tmp_path/'parent'; project.mkdir()
    old=c.artifact('RECOVERY_SPEC', root=str(tmp_path), project_dir=str(project),
        source_commit=fs.PARENT_COMMIT, counts=dict(c.COUNTS), reuse=dict(environment={'fingerprint':'expected'}))
    path=tmp_path/'campaign_spec.json'; write_immutable_json(path, old)
    monkeypatch.setattr(fs,'runtime',lambda env: None if env==old['reuse']['environment'] else pytest.fail('Wrong fingerprint'))
    replay=dict(spec=deepcopy(old), jobs=dict(fs.JOBS), de066={'content_hash':'done'})
    if failure=='spec': replay['spec']['root']='wrong'
    if failure=='jobs': replay['jobs'].pop('train_DE000')
    if failure=='ancestor': replay['de066']=None
    line='FINALIZE_PARENT='+json.dumps(replay)+'\n'
    def run(args, **kw):
        assert args[0]==sys.executable and kw['cwd']==project
        assert kw['env']['PYTHONPATH']==str(project/'src')
        assert kw['env']['PYTHONDONTWRITEBYTECODE']==kw['env']['PYTHONNOUSERSITE']=='1'
        return SimpleNamespace(returncode=int(failure=='subprocess'), stderr='failure',
            stdout='' if failure=='missing' else line*2 if failure=='duplicate' else line)
    monkeypatch.setattr(subprocess,'run',run)
    if failure=='none': assert fs.original(path)==(old,fs.JOBS,replay['de066'])
    else:
        with pytest.raises(ValueError): fs.original(path)


def test_runtime_source_boundaries_and_unchanged_science(tmp_path, monkeypatch):
    monkeypatch.setattr(fs,'RUNTIME',str(tmp_path))
    monkeypatch.setattr(fs.sys,'prefix',str(tmp_path/'wrong'))
    with pytest.raises(PermissionError): fs.runtime({})
    monkeypatch.setattr(fs.sys,'prefix',str(tmp_path))
    monkeypatch.setattr(fs,'installed_environment',lambda: {'verified':True})
    fs.runtime({'verified':True})
    with pytest.raises(ValueError): fs.runtime({'verified':False})
    assert fs.compatible(fs.MODIFIED|fs.ADDED,fs.ADDED)
    assert not fs.compatible(fs.MODIFIED|{'src/scientific_change.py'},set())
    assert not fs.compatible(fs.ADDED,set())
    project=Path(__file__).resolve().parents[1]
    for name in ('training.py','model.py','contracts.py'):
        relative='src/hlt_classification/luka_fullstat_splithead/'+name
        before=subprocess.run(['git','show',fs.BASELINE+':'+relative],cwd=project,
            check=True,capture_output=True,text=True).stdout
        assert ast.dump(ast.parse(before))==ast.dump(ast.parse((project/relative).read_text()))
    monkeypatch.setattr(fs,'_source',lambda *a: {'content_hash':'s'})
    monkeypatch.setattr(subprocess,'run',lambda args,**kw:SimpleNamespace(stdout='\n'.join(
        fs.REQUIRED if 'ls-files' in args else fs.ADDED if '--diff-filter=A' in args else fs.MODIFIED|fs.ADDED)))
    assert fs.current(project,'f'*40)=={'content_hash':'s'}
    compile(fs.REPLAY,'replay','exec')


def test_finalizer_ancestor_must_be_successful_registered_job(monkeypatch):
    s=dict(contract=f.CONTRACT, execution_site=dict(account='default'))
    monkeypatch.setattr(submission,'accepted_jobs',lambda _: {'finalize_DE033':'8200001'})
    monkeypatch.setattr(submission.getpass,'getuser',lambda:'rlyang')
    row=['8200001','lukafssf_finalize_DE033','rlyang','default','COMPLETED','0:0']
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='|'.join(row)))
    submission.successful(s,'finalize_DE033','8200001')
    for index, bad in [(1,'lukafssr_finalize_DE033'),(2,'other'),(3,'other'),(4,'RUNNING'),(5,'1:0')]:
        good=row[index]; row[index]=bad
        with pytest.raises(ValueError): submission.successful(s,'finalize_DE033','8200001')
        row[index]=good


def test_shell_helpers_docs_and_cli():
    project=Path(__file__).resolve().parents[1]
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    if not bash.is_file(): bash=shutil.which('bash')
    if not bash: pytest.skip('Bash unavailable')
    for name in fs.ADDED:
        if name.endswith('.sh'):
            subprocess.run([str(bash),'-n',name],cwd=project,check=True,capture_output=True)
    common=(project/'sbatch/luka_splithead_finalize_common.sh').read_text()
    assert fs.RUNTIME in common and 'hlt_activate' in common
    assert 'jetclass2_delphes_common.sh' not in common
    for name in fs.REQUIRED:
        text=(project/name).read_text()
        assert all(line==line.rstrip() for line in text.splitlines()),name
        if name.endswith('.md'):
            for block in re.findall(r'^```bash\n(.*?)^```',text,re.M|re.S):
                subprocess.run([str(bash),'-n'],input=block,text=True,check=True,capture_output=True)
            for link in re.findall(r'\]\(([^)#]+)(?:#[^)]*)?\)',text):
                assert (project/name).parent.joinpath(link).is_file(),(name,link)
    subprocess.run([sys.executable,'-B','scripts/luka_fullstat_splithead_finalize.py','--help'],
        cwd=project,check=True,capture_output=True)

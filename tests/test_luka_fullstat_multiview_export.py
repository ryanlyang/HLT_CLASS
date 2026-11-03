"""Finished-fit recovery: exact targets, unchanged student, fail-closed queue."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import sys
import time
import numpy as np
import pytest
import torch
from test_luka_embedding_aux import threads
from test_luka_fullstat_multiview import fixture_spec, worker_fixture, tiny
from test_luka_fullstat_multiview_recovery import setup, evidence, recovered
from test_luka_fullstat_multiview_launch import allocation
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullstat_multiview import (
    worker as original, recovery as prior, submission as oldsubmit, campaign as oldcampaign,
    targets, views, startup, model, training)
from hlt_classification.luka_fullstat_multiview_export import (
    contracts as c, campaign, submission, worker, execution, launch, bootstrap)


def new_spec(old, root, reuse=None):
    if reuse is None:
        reuse = c.artifact('REUSE',parents=dict(spec=old['content_hash']),
            parent_spec=file_ref(Path(old['root'])/'campaign_spec.json'),u100=old['reuse']['u100'],
            jobs=dict(c.JOBS),outputs={},selected_pass=14,report_hash='d'*64)
    return c.artifact('SPEC',parents=dict(source=old['source_snapshot']['content_hash'],reuse=reuse['content_hash']),
        root=str(root),project_dir=old['project_dir'],source_commit=old['source_commit'],
        source_snapshot=old['source_snapshot'],reuse=reuse,available_quota_gib=22,**campaign.registration(old))


def fixture(tmp_path, monkeypatch):
    old,_ = setup(tmp_path,monkeypatch)
    old = with_content_hash(dict(old,source_commit=c.PARENT_COMMIT))
    Path(old['root']).mkdir(); write_immutable_json(Path(old['root'])/'campaign_spec.json',old)
    return new_spec(old,tmp_path/'luka_fullstat_multiview_export_fixture'),old


def test_registration_and_two_job_science(tmp_path,monkeypatch):
    s,old = fixture(tmp_path,monkeypatch)
    campaign.validate_spec(s)
    assert s['recipe'] == old['recipe'] and s['nodes'] == old['nodes']
    assert (s['scientific_fits'],s['export_jobs'],s['preflight_jobs']) == (1,1,0)
    p = submission.render(s)
    assert [r['task_id'] for r in p['commands']] == ['export_FUSION','train_multi3']
    assert '--time=06:00:00' in p['commands'][0]['command']
    assert '--time=1-00:00:00' in p['commands'][1]['command']
    assert p['commands'][1]['dependencies'] == ['export_FUSION']
    assert s['recipe']['total_embedding_weight'] == 10 and s['recipe']['ce_weight'] == 1
    assert s['recipe']['epochs'] == 20 and s['recipe']['batch_size'] == 128
    for task in ('teacher_FUSION','teacher_CONCAT'):
        with pytest.raises(PermissionError): worker.run(s,task,device='cpu')


@pytest.mark.parametrize('change',['weight','epochs','quota','source','u100','tasks','walltime','sealed','overlap'])
def test_changed_registration_rejected(tmp_path,monkeypatch,change):
    s,old = fixture(tmp_path,monkeypatch)
    if change == 'weight': s['recipe']['total_embedding_weight'] = 5
    elif change == 'epochs': s['recipe']['epochs'] = 19
    elif change == 'quota': s['available_quota_gib'] = 21
    elif change == 'source': monkeypatch.setattr(campaign,'PARENT_COMMIT','9'*40)
    elif change == 'u100': s['reuse']['u100']['bank_hash'] = '0'*64; s['reuse'] = with_content_hash(s['reuse'])
    elif change == 'tasks': s['tasks']['export_FUSION'] = ['teacher_FUSION']
    elif change == 'walltime': s['minutes']['export_FUSION'] = 4320
    elif change == 'sealed': s['final_test_accessed'] = True
    else: s['root'] = str(Path(old['root'])/'luka_fullstat_multiview_export_nested')
    s['parents']['reuse'] = s['reuse']['content_hash']
    with pytest.raises((ValueError,PermissionError)): campaign.validate_spec(with_content_hash(s))


@pytest.mark.parametrize('bad',['none','owner','state','code','duplicate','missing','job'])
def test_original_failed_job_accounting(tmp_path,monkeypatch,bad):
    _,old = fixture(tmp_path,monkeypatch)
    rows = [[job,'lukafsmv_'+task,campaign.getpass.getuser(),'default',state,code]
        for (task,job),(state,code) in zip(c.JOBS.items(),[('FAILED','0:7'),('CANCELLED','0:0')])]
    jobs = dict(c.JOBS)
    if bad == 'owner': rows[0][2] = 'other'
    elif bad == 'state': rows[0][4] = 'RUNNING'
    elif bad == 'code': rows[0][5] = '1:0'
    elif bad == 'duplicate': rows[1] = rows[0]
    elif bad == 'missing': rows.pop()
    elif bad == 'job': jobs['teacher_FUSION'] = '99'
    monkeypatch.setattr(oldsubmit,'accepted_jobs',lambda *a:jobs)
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='\n'.join('|'.join(r) for r in rows)))
    if bad == 'none': assert campaign.accounting(old) == sorted(rows)
    else:
        with pytest.raises(ValueError): campaign.accounting(old)


@pytest.mark.parametrize('failure',['none','scheduler','first','second','authorization'])
def test_detached_submission_two_jobs_and_partial_failures(tmp_path,monkeypatch,failure):
    s,old = fixture(tmp_path,monkeypatch); root = Path(s['root']); directory = Path(str(root)+'.launch'); directory.mkdir()
    def create(*a,**kw): root.mkdir(); write_immutable_json(root/'campaign_spec.json',s); return s
    monkeypatch.setattr(campaign,'create',create); monkeypatch.setattr(campaign,'check',lambda *a,**kw:old)
    monkeypatch.setattr(campaign,'accounting',lambda *a:[])
    def site(plan):
        if failure == 'scheduler': raise PermissionError('site rejected')
    monkeypatch.setattr(submission,'check_site',site)
    calls = []
    def sbatch(args,**kw):
        assert (directory/'approval.json').is_file()
        assert (Path(old['root'])/c.CLAIM/'intent.json').is_file()
        calls.append(args)
        return SimpleNamespace(returncode=0,stdout='ambiguous\n' if failure == 'first' or (failure == 'second' and len(calls) == 2)
            else str(8400000+len(calls))+'\n',stderr='')
    monkeypatch.setattr(subprocess,'run',sbatch)
    params = dict(root=root,path=s['reuse']['parent_spec']['path'],project='/project',commit='f'*40,
        quota_gib=22,authorization=c.LAUNCH_AUTHORIZATION if failure != 'authorization' else 'wrong')
    if failure == 'none':
        result = launch.run(**params)
        assert result['jobs'] == {'export_FUSION':'8400001','train_multi3':'8400002'}
        assert '--dependency=afterok:8400001' in calls[1]
        assert not any('teacher_FUSION' in a or 'teacher_CONCAT' in a for call in calls for a in call)
        assert submission.accepted_jobs(s) == result['jobs']
    else:
        with pytest.raises((ValueError,PermissionError)): launch.run(**params)
        assert len(calls) == {'scheduler':0,'first':1,'second':2,'authorization':0}[failure]
        if failure == 'authorization':
            assert not (directory/'process_claim').exists(); return
        assert (directory/'failure.json').is_file()
        if failure == 'second': assert submission.accepted_jobs(s,partial=True) == {'export_FUSION':'8400001'}
    count = len(calls)
    with pytest.raises(FileExistsError): launch.run(**params)
    assert len(calls) == count


def test_current_source_allows_adapter_only(monkeypatch):
    project = Path(campaign.__file__).resolve().parents[3]
    monkeypatch.setattr(campaign,'capture_source_snapshot',lambda *a,**kw:dict(git_commit='f'*40,content_hash='a'*64))
    changes = [campaign.PACKAGE+'worker.py']
    def git(args,**kw):
        return SimpleNamespace(stdout='\n'.join(campaign.REQUIRED) if 'ls-files' in args else '\n'.join(changes) if 'diff' in args else '')
    monkeypatch.setattr(subprocess,'run',git)
    campaign.current(project,'f'*40)
    changes.append('src/hlt_classification/luka_fullstat_multiview/training.py')
    with pytest.raises(ValueError,match='Unexpected runtime'): campaign.current(project,'f'*40)


def test_exact_export_allocation(tmp_path,monkeypatch):
    s,measured,fields = allocation(tmp_path,monkeypatch)
    s.update(contract='LUKA_FULLSTAT_MULTIVIEW_EXPORT_SPEC/v1',minutes=dict(c.MINUTES))
    monkeypatch.setattr(campaign,'parent',lambda _:dict(reuse=dict(u100=measured)))
    monkeypatch.setattr(execution,'installed_environment',lambda:measured['environment'])
    monkeypatch.setattr(execution,'gpu_identity',lambda:measured['gpu'])
    with pytest.raises(PermissionError): execution.execution(s,'export_FUSION')
    fields['TimeLimit'] = '06:00:00'
    assert execution.execution(s,'export_FUSION') == '8300001'
    with pytest.raises(PermissionError): execution.execution(s,'teacher_FUSION')
    fields['TimeLimit'] = '1-00:00:00'
    assert execution.execution(s,'train_multi3') == '8300001'
    monkeypatch.setenv('SLURM_MEM_PER_NODE','180000')
    with pytest.raises(PermissionError): execution.execution(s,'train_multi3')


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_finished_teacher_export_only_then_twenty_epoch_student(tmp_path,monkeypatch,device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    tiny(monkeypatch)
    old,tr,val,*_ = worker_fixture(tmp_path,monkeypatch,device)
    monkeypatch.setattr(startup,'installed_parity',lambda *a,**kw:None)
    original.run(old,'teacher_CONCAT',device=device)
    monkeypatch.setattr(prior,'JOBS',dict(teacher_CONCAT='100',teacher_FUSION='101',train_multi3='102'))
    r = evidence(old,monkeypatch)
    parent = recovered(old,r,tmp_path/'luka_fullstat_multiview_recovery_finished')
    project = Path(oldcampaign.__file__).resolve().parents[3]
    parent = with_content_hash(dict(parent,counts=old['counts'],project_dir=str(project)))
    folder = Path(parent['root']); folder.mkdir(); write_immutable_json(folder/'campaign_spec.json',parent)
    real_export = targets.export
    def fail(*a,**kw): raise RuntimeError('synthetic late export crash')
    monkeypatch.setattr(targets,'export',fail)
    with pytest.raises(RuntimeError,match='late export crash'): original.run(parent,'teacher_FUSION',device=device)
    monkeypatch.setattr(targets,'export',real_export)
    report,_ = original.validate_fit(parent,'teacher_FUSION')
    assert report['passes'] == 20 and not (folder/'teacher_FUSION/task_receipt.json').exists()
    monkeypatch.setattr(campaign,'PARENT_COMMIT',parent['source_commit'])
    monkeypatch.setattr(campaign,'accounting',lambda *a:[])
    _,reuse = campaign.inspect_parent(folder/'campaign_spec.json')
    s = new_spec(parent,tmp_path/'luka_fullstat_multiview_export_workers',reuse)
    Path(s['root']).mkdir(); write_immutable_json(Path(s['root'])/'campaign_spec.json',s)
    before = {str(p):file_ref(p)['sha256'] for base in (folder,Path(old['root'])) for p in base.rglob('*') if p.is_file()}
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:parent)
    monkeypatch.setattr(worker,'execution',lambda spec,task:'200' if task == 'export_FUSION' else '201')
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a,**kw:dict(export_FUSION='200',train_multi3='201'))
    monkeypatch.setattr(submission,'successful',lambda *a:None)
    monkeypatch.setattr(worker,'memory_snapshot',original.memory_snapshot)
    monkeypatch.setattr(views,'metadata',lambda _:({'summaries':parent['reuse']['u100']['summaries']},None,None))
    def plain(spec,role,co,meta):
        assert role == 'train' and co in ('D000','OFFLINE'); return tr
    monkeypatch.setattr(views,'plain',plain)
    real_train = training.train
    monkeypatch.setattr(training,'train',lambda *a,**kw:pytest.fail('Export recovery must NEVER train'))
    worker.run(s,'export_FUSION',device=device)
    worker.completed(s,'export_FUSION')
    proof = load_json(Path(s['root'])/'export_FUSION/export_check.json')
    for key,value in (('precision','BF16'),('batch_size',64),('seconds_per_batch',float('nan')),
                      ('selected_pass',0),('conservative_minutes',9999)):
        bad = dict(proof,**{key:value})
        with pytest.raises(ValueError): worker.check_export(s,with_content_hash(bad))
    net = model.new_model(parent['nodes']['teacher_FUSION'],'teacher_FUSION').to(device)
    net.load_state_dict(torch.load(folder/'teacher_FUSION/selected.pt',map_location=device,weights_only=True))
    real_export(tmp_path/'golden_bank',net,views.PairedCache(tr,tr),'FUSION',original.teacher_refs(parent,'FUSION'),parent,device)
    assert (tmp_path/'golden_bank/targets.npz').read_bytes() == (worker.fusion_bank(s)/'targets.npz').read_bytes()
    del net
    # The new student must still use the ordinary production twenty-epoch trainer.
    monkeypatch.setattr(training,'train',real_train)
    worker.run(s,'train_multi3',device=device); worker.completed(s,'train_multi3')
    result,_ = worker.validate_fit(s)
    assert len(result['validation_history']) == 20
    state = torch.load(Path(s['root'])/'train_multi3/selected.pt',map_location='cpu',weights_only=True)
    assert not any('predictor' in k for k in state)
    assert before == {str(p):file_ref(p)['sha256'] for base in (folder,Path(old['root'])) for p in base.rglob('*') if p.is_file()}
    assert campaign.inspect_parent(folder/'campaign_spec.json')[1] == reuse
    assert worker.results(s)['pending'] == []
    # The failed teacher still has no fabricated receipt; corrupt dependencies fail closed.
    assert not (folder/'teacher_FUSION/task_receipt.json').exists()
    with (worker.fusion_bank(s)/'targets.npz').open('ab') as stream: stream.write(b'corruption')
    with pytest.raises(ValueError): worker.completed(s,'export_FUSION')
    with (folder/'teacher_FUSION/selected_training_state.pt').open('ab') as stream: stream.write(b'corruption')
    with pytest.raises(ValueError,match='Selected training state'): campaign.inspect_parent(folder/'campaign_spec.json')


def test_bootstrap_resolves_historical_modules_before_adapter(tmp_path):
    old = tmp_path/'old'; new = tmp_path/'new'
    for project in (old,new):
        pkg = project/'src/hlt_classification'; pkg.mkdir(parents=True)
        (pkg/'__init__.py').write_text('')
        (pkg/'kernel.py').write_text('ORIGIN = '+repr(project.name))
    pkg = new/'src/hlt_classification/luka_fullstat_multiview_export'; pkg.mkdir()
    (pkg/'__init__.py').write_text('ORIGIN = "new"')
    (new/'src/hlt_classification/unrelated_new.py').write_text('raise AssertionError("new source leaked")')
    spec = tmp_path/'parent.json'; spec.write_text(json.dumps(dict(project_dir=str(old),source_commit=c.PARENT_COMMIT)))
    code = '''
import importlib.util,sys
from types import SimpleNamespace
m=importlib.util.spec_from_file_location('boot',sys.argv[1]);b=importlib.util.module_from_spec(m);m.loader.exec_module(b)
b.subprocess.run=lambda args,**kw:SimpleNamespace(stdout=b.PARENT_COMMIT if 'rev-parse' in args else '')
b.configure(sys.argv[2],sys.argv[3])
from hlt_classification import kernel,luka_fullstat_multiview_export as adapter
assert kernel.ORIGIN=='old' and adapter.ORIGIN=='new'
try:
    import hlt_classification.unrelated_new
except ModuleNotFoundError: pass
else: raise AssertionError('Unrelated new modules must not become fallbacks')
'''
    subprocess.run([sys.executable,'-I','-B','-c',code,bootstrap.__file__,str(new),str(spec)],check=True,capture_output=True,text=True)


def test_atomic_original_claim_blocks_second_recovery(tmp_path,monkeypatch):
    s,old = fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(campaign,'accounting',lambda *a:[])
    campaign.claim(s,submission.render(s))
    other = with_content_hash(dict(s,root=s['root']+'_another'))
    with pytest.raises(FileExistsError): campaign.claim(other,submission.render(other))
    assert load_json(Path(old['root'])/c.CLAIM/'intent.json')['root'] == s['root']


@pytest.mark.parametrize('path',['teacher_FUSION/task_receipt.json','teacher_FUSION/bank/manifest.json','train_multi3'])
def test_unexpected_published_or_attempted_outputs_block_recovery(tmp_path,monkeypatch,path):
    s,old = fixture(tmp_path,monkeypatch)
    old = with_content_hash(dict(old,project_dir=str(Path(oldcampaign.__file__).resolve().parents[3])))
    p = Path(old['root'])/'campaign_spec.json'; p.write_text(json.dumps(old))
    target = Path(old['root'])/path; target.parent.mkdir(parents=True,exist_ok=True)
    target.mkdir() if path == 'train_multi3' else target.write_text('{}')
    monkeypatch.setattr(oldcampaign,'check',lambda *a,**kw:None)
    monkeypatch.setattr(original,'validate_fit',lambda *a:pytest.fail('Unexpected artifacts must stop before reading fit'))
    with pytest.raises(ValueError): campaign.inspect_parent(p)


def test_shell_tail_can_exit_while_detached_launcher_finishes(tmp_path):
    # Real shell process orchestration; setsid is stubbed on Windows, so this
    # proves asynchronous completion, not Linux SSH-session survival.
    bash = 'C:/Program Files/Git/bin/bash.exe' if sys.platform == 'win32' else 'bash'
    def shellpath(p):
        p = Path(p).resolve()
        return '/'+p.drive[0].lower()+p.as_posix()[2:] if sys.platform == 'win32' else str(p)
    repo = tmp_path/'project'; (repo/'scripts').mkdir(parents=True); (repo/'sbatch').mkdir()
    name = 'launch_luka_fullstat_multiview_export.sh'
    script = repo/'scripts'/name
    script.write_bytes((Path(__file__).resolve().parents[1]/'scripts'/name).read_bytes())
    (repo/'sbatch/luka_splithead_finalize_common.sh').write_text('# fixture runtime\n')
    binpath = tmp_path/'bin'; binpath.mkdir()
    bodies = {
        'git':'case "$*" in\n*--show-toplevel*) printf "%s\\n" "$FIXTURE_PROJECT";;\n*"rev-parse HEAD"*) printf "%040d\\n" 1;;\nesac\n',
        'setsid':'exec "$@"\n', 'tail':'printf "TAIL_ATTACHED\\n"\n',
        'python':'sleep 1\nprintf "DETACHED_FINISHED\\n"\nprintf done > "$FIXTURE_DONE"\n'}
    for name,body in bodies.items():
        p = binpath/name; p.write_bytes(('#!/usr/bin/env bash\n'+body).encode()); p.chmod(0o755)
    parent = tmp_path/'parent.json'; parent.write_text('{}')
    root = tmp_path/'output'; done = tmp_path/'done'
    env = dict(os.environ,PATH=str(binpath)+os.pathsep+os.environ['PATH'],
        FIXTURE_PROJECT=shellpath(repo),FIXTURE_DONE=shellpath(done))
    startup_env = tmp_path/'env.sh'; startup_env.write_bytes(b'export PATH="${FIXTURE_BIN}:${PATH}"\n')
    env.update(BASH_ENV=shellpath(startup_env),FIXTURE_BIN=shellpath(binpath))
    args = [bash,str(script),'0'*39+'1',shellpath(root),shellpath(parent),'22','--authorize',c.LAUNCH_AUTHORIZATION]
    result = subprocess.run(args,env=env,capture_output=True,text=True,timeout=10)
    assert result.returncode == 0,result.stderr
    assert 'TAIL_ATTACHED' in result.stdout and 'DETACHED:' in result.stdout
    deadline = time.monotonic()+10
    while not done.exists() and time.monotonic() < deadline: time.sleep(.05)
    log = Path(str(root)+'.launch/launcher.log')
    assert done.read_text() == 'done',log.read_text()
    assert 'DETACHED_FINISHED' in log.read_text()
    assert subprocess.run(args,env=env,capture_output=True,text=True,timeout=5).returncode != 0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_full_installed_weaver_selected_FP32_export(tmp_path,monkeypatch):
    # Full production architecture, without the tiny() block-count replacement.
    s,tr,_,*_ = worker_fixture(tmp_path,monkeypatch,'cuda')
    from hlt_classification.jetclass2_delphes.acceptance import installed_parity
    from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
    from hlt_classification.context_fusion.worker import save_state
    with parity_backend('cuda'): installed_parity(tr,device='cuda')
    paired = views.PairedCache(tr,tr)
    folder = Path(s['root'])/'teacher_FUSION'; folder.mkdir()
    net = model.new_model(s['nodes']['teacher_FUSION'],'teacher_FUSION').cuda().eval()
    save_state(folder/'selected.pt',net.state_dict())
    report = with_content_hash(dict(selected_pass=14)); write_immutable_json(folder/'training_report.json',report)
    write_immutable_json(folder/'startup.json',dict(train=views.describe(paired)))
    write_immutable_json(Path(s['root'])/'campaign_spec.json',s)
    reuse = c.artifact('REUSE',parent_spec=file_ref(Path(s['root'])/'campaign_spec.json'),
        u100=s['reuse']['u100'],selected_pass=14,report_hash=report['content_hash'])
    new = new_spec(s,tmp_path/'luka_fullstat_multiview_export_native',reuse)
    directory = Path(new['root'])/'export_FUSION'; directory.mkdir(parents=True)
    monkeypatch.setattr(views,'metadata',lambda _:({'summaries':s['reuse']['u100']['summaries']},None,None))
    monkeypatch.setattr(views,'plain',lambda spec,role,co,meta:tr if role == 'train' else pytest.fail('Only TRAIN may load'))
    monkeypatch.setattr(worker,'memory_snapshot',original.memory_snapshot)
    monkeypatch.setattr(training,'train',lambda *a,**kw:pytest.fail('Export cannot train'))
    log = worker.PhaseLog(directory,lambda:worker.memory_snapshot(torch))
    worker.export_fit(new,directory,log,'cuda',time.monotonic())
    targets.export(tmp_path/'golden_native',net,paired,'FUSION',original.teacher_refs(s,'FUSION'),s,'cuda')
    assert (directory/'bank/targets.npz').read_bytes() == (tmp_path/'golden_native/targets.npz').read_bytes()

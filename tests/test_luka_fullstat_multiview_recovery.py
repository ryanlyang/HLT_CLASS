"""Exact timing-rejection recovery, immutable CONCAT reuse and two-job launch."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import subprocess
import pytest
import torch
from test_luka_fullstat_multiview import fixture_spec, worker_fixture, tiny
from test_luka_fullstat_multiview_launch import allocation
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullstat_multiview import (
    contracts as c, recovery, campaign, resources, startup, worker, submission, launch, source)


TIMING = dict(train_step=.3329635614354629,val_step=.044565155170857906,
              export_step=.06421695626340806,preparation=465.17860917933285)


def evidence(old, monkeypatch):
    # The inherited tiny fixtures do not materialize the historical raw-data
    # directory tree. Keep the new old-root protection real, stub only that
    # separately tested pre-existing historical-location check.
    from hlt_classification.luka_fullstat_multiteacher import campaign as multi_campaign
    monkeypatch.setattr(multi_campaign,'separate',lambda *a:None)
    monkeypatch.setattr(recovery,'PARENT_COMMIT',old['source_commit'])
    monkeypatch.setattr(recovery,'PARENT_HASH',old['content_hash'])
    monkeypatch.setattr(recovery,'COUNTS',old['counts'])
    root=Path(old['root']); root.mkdir(exist_ok=True)
    write_immutable_json(root/'campaign_spec.json',old)
    args=dict(TIMING)
    if old['counts']['train'] != 3000000:
        args.update(train_step=6000.,val_step=.01,export_step=.01,preparation=1.)
    t=startup.timing(old,'teacher_FUSION',**args,enforce=False)
    assert 2880<t['conservative_minutes']<=4320
    folder=root/'teacher_FUSION/startup_diagnostics'
    mem=dict(gpu_peak_bytes=2**30,peak_rss_bytes=2**30,rss_bytes=2**30)
    for index,stage,fields in (
        (9,'measurements',dict(timing=t,measured_memory=mem,allocated_minutes=2880)),
        (10,'failure',dict(failed_stage='measurements',error_type='ValueError',
          message=f'Measured full20 plus export requires {t["conservative_minutes"]} minutes; allocation is 2880 minutes. See startup_diagnostics; no fallback.'))):
        write_immutable_json(folder/f'{index:02d}-{stage}.json',c.artifact('STARTUP_DIAGNOSTIC',
            parents=dict(spec=old['content_hash']),stage=stage,task_id='teacher_FUSION',accepted=False,
            runtime_policy=resources.policy(),memory=mem,**fields))
    for name in ('task_receipt.json','bank/manifest.json'):
        path=root/'teacher_CONCAT'/name
        if not path.exists(): write_immutable_json(path,c.artifact('FIXTURE'))
    return c.artifact('RECOVERY_REUSE',parents=dict(spec=old['content_hash']),parent_spec=file_ref(root/'campaign_spec.json'),
        receipt=file_ref(root/'teacher_CONCAT/task_receipt.json'),bank=file_ref(root/'teacher_CONCAT/bank/manifest.json'),
        original_source={'src/frozen.py':'a'*64},accounting=[],jobs=dict(recovery.JOBS),**recovery.failed_parent(old))


def recovered(old, r, root):
    return c.artifact('RECOVERY_SPEC',parents=dict(source=old['source_snapshot']['content_hash'],
        reuse=old['reuse']['content_hash'],recovery=r['content_hash']),root=str(root),project_dir=old['project_dir'],
        source_commit=old['source_commit'],source_snapshot=old['source_snapshot'],reuse=old['reuse'],recovery=r,
        available_quota_gib=22,**campaign.registration(old['reuse'],recovered=True))


def setup(tmp_path,monkeypatch):
    old=fixture_spec(tmp_path/'luka_fullstat_multiview_original')
    r=evidence(old,monkeypatch)
    s=recovered(old,r,tmp_path/'luka_fullstat_multiview_recovery_fixture')
    return s,old


def test_real_witness_accepted_at_72h_without_relaxing_guards(tmp_path,monkeypatch):
    s,old=setup(tmp_path,monkeypatch)
    campaign.validate_spec(s)
    with pytest.raises(ValueError,match='4164 minutes'):
        startup.timing(old,'teacher_FUSION',**TIMING)
    assert startup.timing(s,'teacher_FUSION',**TIMING)['conservative_minutes']==4164
    assert resources.limits(s)==resources.limits(old)
    assert s['recipe']==old['recipe'] and s['nodes']==old['nodes']
    with pytest.raises(ValueError): startup.timing(s,'teacher_FUSION',1.,.04,.06,500.)
    with pytest.raises(MemoryError): resources.check_memory(s,dict(gpu_peak_bytes=1e12,peak_rss_bytes=2**30))
    assert s['scientific_fits']==2 and s['preflight_jobs']==s['setup_jobs']==0


@pytest.mark.parametrize('bad',['recipe','nodes','walltime','runtime','reference','parent','diagnostic','attempt','student','claim'])
def test_wrong_recovery_or_changed_parent_fails_closed(tmp_path,monkeypatch,bad):
    s,old=setup(tmp_path,monkeypatch)
    if bad=='recipe': s['recipe']['epochs']=19
    elif bad=='nodes': s['nodes']['train_multi3']['coordinate']='OFFLINE'
    elif bad=='walltime': s['minutes']['teacher_FUSION']=9999
    elif bad=='runtime': s['runtime_policy']['timing_factor']=1.
    elif bad=='reference': s['recovery']['bank']['sha256']='f'*64
    elif bad=='parent': Path(s['recovery']['parent_spec']['path']).write_text('{}')
    elif bad=='diagnostic': Path(s['recovery']['failure']['path']).write_text('{}')
    elif bad=='attempt': (Path(old['root'])/'teacher_FUSION/epochs').mkdir()
    elif bad=='student': (Path(old['root'])/'train_multi3').mkdir()
    else:
        monkeypatch.setattr(recovery,'accounting',lambda *a:[])
        recovery.claim(s,submission.render(s))
        with pytest.raises(FileExistsError): recovery.claim(s,submission.render(s))
        return
    with pytest.raises((ValueError,PermissionError,KeyError)): campaign.validate_spec(with_content_hash(s))


def test_current_source_freezes_scientific_files(monkeypatch):
    project=Path(source.__file__).resolve().parents[3]
    monkeypatch.setattr(recovery,'_source',lambda *a:dict(content_hash='a'*64))
    changes={'src/hlt_classification/luka_fullstat_multiview/recovery.py'}
    def git(args,**kw):
        if 'ls-files' in args: out='\n'.join(source.REQUIRED)
        elif 'diff' in args: out='\n'.join(changes)
        else: out=''
        return SimpleNamespace(stdout=out)
    monkeypatch.setattr(subprocess,'run',git)
    assert recovery.current(project,'f'*40)['content_hash']=='a'*64
    changes.add('src/hlt_classification/luka_fullstat_multiview/training.py')
    with pytest.raises(ValueError,match='frozen scientific'): recovery.current(project,'f'*40)


@pytest.mark.parametrize('bad',['owner','state','missing','duplicate','jobs','none'])
def test_exact_terminal_scheduler_accounting(tmp_path,monkeypatch,bad):
    _,old=setup(tmp_path,monkeypatch)
    rows=[[job,'lukafsmv_'+task,recovery.getpass.getuser(),'default',state,code]
          for (task,job),(state,code) in zip(recovery.JOBS.items(),[('COMPLETED','0:0'),('FAILED','1:0'),('CANCELLED','0:0')])]
    jobs=dict(recovery.JOBS)
    if bad=='owner': rows[1][2]='somebody_else'
    elif bad=='state': rows[1][4]='RUNNING'
    elif bad=='missing': rows.pop()
    elif bad=='duplicate': rows[2]=rows[1]
    elif bad=='jobs': jobs['teacher_FUSION']='99'
    def sacct(args,**kw):
        assert args[0]=='sacct' and '2026-10-30' in args
        return SimpleNamespace(stdout='\n'.join('|'.join(r) for r in rows))
    monkeypatch.setattr(subprocess,'run',sacct)
    if bad=='none': assert recovery.accounting(old,jobs)==sorted(rows)
    else:
        with pytest.raises(ValueError): recovery.accounting(old,jobs)


def test_two_job_detached_launch_and_original_claim(tmp_path,monkeypatch):
    s,old=setup(tmp_path,monkeypatch); root=Path(s['root']); directory=Path(str(root)+'.launch');directory.mkdir()
    def create(*a,**kw):
        root.mkdir();write_immutable_json(root/'campaign_spec.json',s);return s
    monkeypatch.setattr(recovery,'create',create)
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    monkeypatch.setattr(recovery,'accounting',lambda *a:[])
    checks=[]; calls=[]
    monkeypatch.setattr(submission,'check_site',lambda plan:checks.append(plan))
    def sbatch(args,**kw):
        assert (directory/'approval.json').is_file()
        assert (Path(old['root'])/recovery.CLAIM/'intent.json').is_file()
        calls.append(args);return SimpleNamespace(returncode=0,stdout=str(8400000+len(calls))+'\n',stderr='')
    monkeypatch.setattr(subprocess,'run',sbatch)
    params=dict(root=root,previous_spec=s['recovery']['parent_spec']['path'],project='/project',commit='f'*40,
                quota_gib=22,authorization=c.RECOVERY_LAUNCH_AUTHORIZATION,recover=True)
    result=launch.run(**params)
    assert len(calls)==2 and len(checks)==2
    assert '--time=3-00:00:00' in calls[0] and '--time=1-00:00:00' in calls[1]
    assert not any(a.startswith('--dependency=') for a in calls[0])
    assert '--dependency=afterok:8400001' in calls[1]
    assert not any('teacher_CONCAT' in a for call in calls for a in call)
    assert set(result['jobs'])=={'teacher_FUSION','train_multi3'}
    assert submission.accepted_jobs(s)==result['jobs']
    with pytest.raises(FileExistsError): launch.run(**params)
    with pytest.raises(FileExistsError): recovery.claim(s,submission.render(s))


@pytest.mark.parametrize('failure',['scheduler','first_sbatch','second_sbatch'])
def test_failed_or_partial_submit_preserves_claims_and_never_retries(tmp_path,monkeypatch,failure):
    s,old=setup(tmp_path,monkeypatch)
    root=Path(s['root']);root.mkdir();write_immutable_json(root/'campaign_spec.json',s)
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    monkeypatch.setattr(recovery,'accounting',lambda *a:[])
    def site(plan):
        if failure=='scheduler': raise PermissionError('72h rejected by scheduler fixture')
    monkeypatch.setattr(submission,'check_site',site)
    calls=[]
    def sbatch(args,**kw):
        calls.append(args)
        return SimpleNamespace(returncode=0,stdout='ambiguous\n' if failure=='first_sbatch' or len(calls)==2 else '8400001\n',stderr='')
    monkeypatch.setattr(subprocess,'run',sbatch)
    if failure=='scheduler':
        with pytest.raises(PermissionError): submission.submit(s)
        assert not calls and not (Path(old['root'])/recovery.CLAIM).exists()
        return
    plan=submission.submit(s)
    with pytest.raises(Exception):
        submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=c.RECOVERY_AUTHORIZATION)
    assert len(calls)==(1 if failure=='first_sbatch' else 2)
    assert (Path(old['root'])/recovery.CLAIM/'intent.json').is_file()
    assert (root/'submission/live_claim/intent.json').is_file()
    if failure=='second_sbatch': assert submission.accepted_jobs(s,partial=True)=={'teacher_FUSION':'8400001'}
    with pytest.raises(FileExistsError):
        submission.submit(s,execute=True,plan_hash=plan['content_hash'],authorization=c.RECOVERY_AUTHORIZATION)
    assert len(calls)==(1 if failure=='first_sbatch' else 2)


def test_create_fresh_recovery_and_reject_overlapping_or_claimed_roots(tmp_path,monkeypatch):
    s,old=setup(tmp_path,monkeypatch)
    monkeypatch.setattr(recovery,'current',lambda *a:old['source_snapshot'])
    monkeypatch.setattr(recovery,'authenticate',lambda *a:s['recovery'])
    path=s['recovery']['parent_spec']['path']; root=Path(s['root'])
    # Resolve the test project like production create() does.
    monkeypatch.setitem(old,'project_dir',str(Path(old['project_dir']).resolve()))
    made=recovery.create(root,path,'/project','f'*40,quota_gib=22)
    assert load_json(root/'campaign_spec.json')==made
    assert made['recipe']==s['recipe'] and made['recovery']==s['recovery']
    with pytest.raises(FileExistsError): recovery.create(root,path,'/project','f'*40,quota_gib=22)
    with pytest.raises(PermissionError): recovery.create(Path(old['root'])/'luka_fullstat_multiview_nested',path,'/project','f'*40,quota_gib=22)
    (Path(old['root'])/recovery.CLAIM).mkdir()
    with pytest.raises(FileExistsError): recovery.create(root.with_name(root.name+'_other'),path,'/project','f'*40,quota_gib=22)


def test_original_source_replay_is_pinned_and_failure_does_not_publish(tmp_path,monkeypatch):
    original=tmp_path/'oldsource'; project=tmp_path/'newsource'
    name='src/hlt_classification/luka_fullstat_multiview/model.py'
    for folder in (original,project):
        path=folder/name;path.parent.mkdir(parents=True);path.write_text('# frozen model\n')
    old=with_content_hash(dict(fixture_spec(tmp_path/'luka_fullstat_multiview_original'),project_dir=str(original)))
    old['reuse']['u100']['environment']={}
    old['reuse']['u100']=with_content_hash(old['reuse']['u100'])
    old['reuse']=with_content_hash(old['reuse'])
    old=with_content_hash(old)
    r=evidence(old,monkeypatch); replay=dict(jobs=r['jobs'],receipt=r['receipt'],bank=r['bank'])
    monkeypatch.setattr(source,'runtime',lambda *a:None)
    monkeypatch.setattr(recovery,'accounting',lambda *a:[])
    def run(args,**kw):
        if args[0]=='git': return SimpleNamespace(stdout=name+'\n')
        assert kw['cwd']==original and kw['env']['PYTHONPATH']==str(original/'src')
        assert args[-2]==recovery.REPLAY and args[-1]==r['parent_spec']['path']
        return SimpleNamespace(returncode=0,stdout='MULTIVIEW_RECOVERY='+__import__('json').dumps(replay),stderr='')
    monkeypatch.setattr(subprocess,'run',run)
    value=recovery.authenticate(r['parent_spec']['path'],project)
    assert value['receipt']==r['receipt'] and name in value['original_source']
    (project/name).write_text('# changed model\n')
    with pytest.raises(ValueError,match='Frozen scientific runtime'): recovery.authenticate(r['parent_spec']['path'],project)
    assert not (Path(old['root'])/recovery.CLAIM).exists()


def test_recovery_allocation_requires_72h_and_refuses_concat(tmp_path,monkeypatch):
    gate,measured,fields=allocation(tmp_path,monkeypatch)
    gate['contract']='LUKA_FULLSTAT_MULTIVIEW_RECOVERY_SPEC/v1'
    gate['minutes']=dict(teacher_FUSION=4320,train_multi3=1440)
    with pytest.raises(PermissionError,match='walltime'): worker.execution(gate,'teacher_FUSION',measured)
    fields['TimeLimit']='3-00:00:00'
    assert worker.execution(gate,'teacher_FUSION',measured)=='8300001'
    with pytest.raises(PermissionError): worker.execution(gate,'teacher_CONCAT',measured)
    with pytest.raises(PermissionError,match='read-only'): worker.run(gate,'teacher_CONCAT')


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_completed_concat_reused_by_real_two_worker_recovery(tmp_path,monkeypatch,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    tiny(monkeypatch)
    old,*_=worker_fixture(tmp_path,monkeypatch,device)
    monkeypatch.setattr(startup,'installed_parity',lambda *a,**kw:None)
    worker.run(old,'teacher_CONCAT',device=device)
    # Bind the fixture's recorded original CONCAT job to its receipt.
    monkeypatch.setattr(recovery,'JOBS',dict(teacher_CONCAT='100',teacher_FUSION='101',train_multi3='102'))
    r=evidence(old,monkeypatch)
    s=recovered(old,r,tmp_path/'luka_fullstat_multiview_recovery_workers')
    s=with_content_hash(dict(s,counts=old['counts']))
    Path(s['root']).mkdir(); write_immutable_json(Path(s['root'])/'campaign_spec.json',s)
    before={p.relative_to(old['root']).as_posix():file_ref(p) for p in (Path(old['root'])/'teacher_CONCAT').rglob('*') if p.is_file()}
    worker.run(s,'teacher_FUSION',device=device)
    worker.run(s,'train_multi3',device=device)
    assert worker.completed(s,'teacher_CONCAT')['job_id']=='100'
    worker.completed(s,'train_multi3')
    assert not (Path(s['root'])/'teacher_CONCAT').exists()
    after={p.relative_to(old['root']).as_posix():file_ref(p) for p in (Path(old['root'])/'teacher_CONCAT').rglob('*') if p.is_file()}
    assert before==after
    report=load_json(Path(s['root'])/'train_multi3/training_report.json')
    assert len(report['validation_history'])==20
    bank=load_json(Path(old['root'])/'teacher_CONCAT/bank/manifest.json')
    assert bank['parents']['spec']==old['content_hash']
    assert worker.fit_parents(s,'train_multi3')['CONCAT']==bank['content_hash']
    assert worker.results(s)['pending']==[]
    bank_path=Path(old['root'])/'teacher_CONCAT/bank/targets.npz'
    with bank_path.open('ab') as stream: stream.write(b'corrupt')
    with pytest.raises(ValueError): worker.completed(s,'teacher_CONCAT')

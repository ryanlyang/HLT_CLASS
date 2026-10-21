"""Exact debug transfer, cancellation boundaries and unchanged scientific workers."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import subprocess
import numpy as np
import pytest
import torch

from test_luka_embedding_sweep import fixture_gate, small_counts, threads
from test_luka_embedding_aux import bank_fixture
from test_luka_attention import cache
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.data.cache_contracts import (
    load_json, write_immutable_json, with_content_hash, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.luka_embedding_sweep import (
    debug as d, debug_retirement as r, contracts as c, campaign, submission, worker, preflight)


def fixture(tmp_path, monkeypatch, gate=None, report=None):
    root=tmp_path/'original'
    root.mkdir(exist_ok=True)
    gate=gate or fixture_gate(root)
    write_immutable_json(root/'gate_spec.json',gate)
    (root/'preflight').mkdir(exist_ok=True)
    report=report or dict(content_hash='a'*64,train_minutes={arm:200+i for i,arm in enumerate(c.WEIGHTS)})
    write_immutable_json(root/'preflight/receipt.json',report)
    original=c.artifact('SPEC',parents=dict(gate=gate['content_hash'],preflight=report['content_hash']),
        root=str(root/'science'),gate_path=str(root/'gate_spec.json'),project_dir=gate['project_dir'],
        source_commit=gate['source_commit'],train_minutes=report['train_minutes'],arms=list(c.WEIGHTS))
    (root/'science').mkdir(exist_ok=True)
    write_immutable_json(root/'science/campaign_spec.json',original)
    evidence=dict(original_ref=file_ref(root/'science/campaign_spec.json'),
        gate_ref=file_ref(root/'gate_spec.json'),preflight_ref=file_ref(root/'preflight/receipt.json'),
        original_jobs={a:str(21937191+i) for i,a in enumerate(c.WEIGHTS)},
        original_plan=submission.render(original,gate),original_refs=[])
    snapshot=dict(content_hash='d'*64,git_commit='e'*40)
    monkeypatch.setattr(d,'current',lambda *a:snapshot)
    monkeypatch.setattr(d,'authenticate',lambda *a:(original,gate,report,evidence))
    spec=d.create(root/'science/campaign_spec.json',tmp_path/'source','e'*40)
    return gate,report,spec


def test_identical_eight_settings_and_fresh_idempotent_spec(tmp_path,monkeypatch):
    gate,report,spec=fixture(tmp_path,monkeypatch)
    assert d.check_spec(spec)==(gate,report)
    assert d.create(spec['original_path'],spec['project_dir'],spec['source_commit'])==spec
    plan=d.render(spec)
    assert plan['jobs']==8 and spec['train_minutes']==report['train_minutes']
    assert gate['execution_site']['partition']=='tier3'
    for row in plan['commands']:
        assert '--partition=debug' in row['command']
        assert '--mem=90000M' in row['command'] and '--gres=gpu:a100:1' in row['command']
        assert row['dependencies']==[]
    assert load_json(spec['gate_path'])==gate
    assert not list((Path(spec['original_path']).parent).glob('w*'))


@pytest.mark.parametrize('change',[
    dict(memory_mb=180000),dict(cpus=8),dict(execution_site=execution_site('sporc_a100')),
    dict(transfer_policy='anything'),dict(source_commit='f'*40),dict(arms=['w10']),
    dict(train_minutes={a:1 for a in c.WEIGHTS}),
])
def test_rehashed_registration_changes_rejected(tmp_path,monkeypatch,change):
    gate,report,spec=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError):
        d.validate_spec(with_content_hash(dict(spec,**change)),gate,report)


def test_corrupt_preflight_source_and_original_outputs_rejected(tmp_path,monkeypatch):
    gate,report,spec=fixture(tmp_path,monkeypatch)
    path=Path(gate['root'])/'preflight/receipt.json'
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError): d.validate_spec(spec,gate,report)
    original=load_json(spec['original_path'])
    (Path(original['root'])/'w3').mkdir()
    with pytest.raises(FileExistsError,match='progress'): d.no_original_outputs(original)
    monkeypatch.setattr(d,'current',lambda *a:{})
    with pytest.raises(ValueError,match='source'): d.check_spec(spec)


def test_matched_runtime_and_current_source_guard(monkeypatch):
    project=Path(__file__).resolve().parents[1]
    monkeypatch.setattr(d,'_source',lambda *a:dict(content_hash='a'*64))
    def git(_, *a):
        if 'ls-files' in a: return list(d.REQUIRED)
        if '--name-status' in a: return ['A\t'+p for p in d.ADDITIONS]
        return list(d.ADDITIONS)
    monkeypatch.setattr(d,'git',git)
    assert d.current(project,'e'*40)['content_hash']=='a'*64
    for p in d.REQUIRED: assert (project/p).is_file(),p
    monkeypatch.setattr(d,'git',lambda *a:['M\tsrc/hlt_classification/luka_embedding_sweep/training.py'])
    with pytest.raises(ValueError,match='runtime changed'): d.matched_runtime(project)


def test_authentication_uses_original_worktree_and_full_gate_replay(tmp_path,monkeypatch):
    original_auth=d.authenticate
    gate,report,spec=fixture(tmp_path,monkeypatch)
    original=load_json(spec['original_path'])
    monkeypatch.setattr(d,'MEASURED_COMMIT',gate['source_commit'])
    monkeypatch.setattr(d,'matched_runtime',lambda *a:None)
    monkeypatch.setattr(d,'validate_source_snapshot',lambda *a,**kw:None)
    evidence=dict(report=report,jobs=spec['original_jobs'],plan=spec['original_plan'],refs=[])
    def replay(path,project,code,marker):
        assert project==original['project_dir']
        assert 'campaign.check_spec(s)' in code and 'followup.science_jobs' in code
        assert "f/'complete.json'" in code
        return file_ref(path),evidence
    monkeypatch.setattr(d,'replay',replay)
    assert original_auth(spec['original_path'],spec['project_dir'])[:3]==(original,gate,report)
    evidence['report']={}
    with pytest.raises(ValueError,match='evidence changed'):
        original_auth(spec['original_path'],spec['project_dir'])


def test_submit_exact_eight_once_and_ambiguous_reply_never_retries(tmp_path,monkeypatch):
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    _,_,spec=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(d,'check_site',lambda *a:None)
    confirmed=[]
    monkeypatch.setattr(r,'confirm_retired',lambda s:confirmed.append(s['content_hash']))
    calls=[]
    def scheduler(command,**kw):
        assert '--partition=debug' in command
        calls.append(command)
        return SimpleNamespace(stdout=str(300000+len(calls))+';sporc\n')
    monkeypatch.setattr(exact.subprocess,'run',scheduler)
    reviewed=d.submit(spec)
    assert not calls and not confirmed
    for _ in range(2):
        d.submit(spec,execute=True,plan_hash=reviewed['content_hash'],authorization=d.AUTHORIZATION)
    assert len(calls)==8 and len(confirmed)==2
    # A fresh isolated fixture tests the permanent claim after an ambiguous response.
    second=tmp_path/'other'; second.mkdir()
    _,_,other=fixture(second,monkeypatch)
    reviewed=d.submit(other)
    def ambiguous(*a,**kw): raise RuntimeError('lost response')
    monkeypatch.setattr(exact.subprocess,'run',ambiguous)
    for error in (RuntimeError,FileExistsError):
        with pytest.raises(error):
            d.submit(other,execute=True,plan_hash=reviewed['content_hash'],authorization=d.AUTHORIZATION)


def job_fixture(spec):
    original=load_json(spec['original_path'])
    rows=r.bound_rows(original,spec['original_plan'],spec['original_jobs'],sweep=True)
    alt=dict(rows[0],job_id='21935431',task_id='train_GATE_TEACH_D066',sweep=False,
        partition='debug',job_name='lukaag_train_GATE_TEACH_D066',spec_hash=r.ALTERNATING_HASH)
    return rows+[alt]


def state(row, status='PENDING'):
    return dict(name=row['job_name'],user='ryreu',account=row['account'],partition=row['partition'],
        state=status,elapsed=0,start='Unknown')


@pytest.mark.parametrize('change',[
    dict(state='RUNNING'),dict(state='COMPLETED'),dict(elapsed=1),
    dict(start='2026-10-21T12:00:00'),dict(name='evmsg_train'),dict(user='someone'),
    dict(account='wrong'),dict(partition='debug'),
])
def test_started_or_foreign_original_never_admitted(tmp_path,monkeypatch,change):
    _,_,spec=fixture(tmp_path,monkeypatch); row=job_fixture(spec)[0]
    with pytest.raises((ValueError,PermissionError)):
        r.check_account(row,dict(state(row),**change),'ryreu')


def test_cancelled_never_started_only_for_submission(tmp_path,monkeypatch):
    _,_,spec=fixture(tmp_path,monkeypatch); row=job_fixture(spec)[0]
    r.check_account(row,state(row,'CANCELLED'),'ryreu',retired=True)
    with pytest.raises(ValueError): r.check_account(row,state(row),'ryreu',retired=True)


@pytest.mark.parametrize('field',['JobId','JobName','Account','QOS','Partition','Comment','WorkDir','Command','StdOut','UserId','Restarts'])
def test_live_identity_guards(tmp_path,monkeypatch,field):
    _,_,spec=fixture(tmp_path,monkeypatch); row=job_fixture(spec)[0]
    record=dict(JobId=row['job_id'],JobName=row['job_name'],Account=row['account'],QOS=row['qos'],
        Partition=row['partition'],Comment=row['comment'],WorkDir=row['workdir'],Command=row['command'],
        StdOut=row['stdout'],UserId='ryreu(2914834)',Requeue='0',Restarts='0',JobState='PENDING',RunTime='00:00:00')
    monkeypatch.setattr(r.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=' '.join(k+'='+v for k,v in record.items())))
    r.check_live(row,'ryreu')
    record[field]='FOREIGN'
    with pytest.raises(PermissionError): r.check_live(row,'ryreu')


def test_exact_cancellation_preserves_terminal_jobs_and_does_not_submit(tmp_path,monkeypatch):
    _,_,spec=fixture(tmp_path,monkeypatch); rows=job_fixture(spec)
    reviewed=d.artifact('RETIRE_PLAN',parents=dict(spec=spec['content_hash']),rows=rows,user='ryreu',
        alternating_path='/authorized/alt',alternating_refs=[])
    monkeypatch.setattr(r,'plan',lambda *a:reviewed)
    monkeypatch.setattr(r,'check_live',lambda *a:None)
    states={row['job_id']:state(row) for row in rows}
    states[rows[1]['job_id']]['state']='CANCELLED'
    states[rows[-1]['job_id']]['state']='COMPLETED'
    monkeypatch.setattr(r,'accounting',lambda rs:{row['job_id']:states[row['job_id']] for row in rs})
    calls=[]
    def scheduler(cmd,**kw):
        assert cmd[:3]==['scancel','--ctld','--state=PENDING']
        assert not any(k.startswith('SCANCEL_') for k in kw['env'])
        calls.extend(cmd[3:])
        for job in cmd[3:]: states[job]['state']='CANCELLED'
        return SimpleNamespace(stdout='')
    monkeypatch.setattr(r.subprocess,'run',scheduler)
    assert r.retire(spec,'/authorized/alt')==reviewed and not calls
    with pytest.raises(PermissionError):
        r.retire(spec,'/authorized/alt',execute=True,plan_hash='wrong',authorization=r.AUTHORIZATION)
    r.retire(spec,'/authorized/alt',execute=True,plan_hash=reviewed['content_hash'],authorization=r.AUTHORIZATION)
    assert len(calls)==7 and all(j in spec['original_jobs'].values() for j in calls)
    assert r.confirm_retired(spec)
    with pytest.raises(FileExistsError):
        r.retire(spec,'/authorized/alt',execute=True,plan_hash=reviewed['content_hash'],authorization=r.AUTHORIZATION)


def test_cancel_race_and_partial_failure_block_replacement(tmp_path,monkeypatch):
    _,_,spec=fixture(tmp_path,monkeypatch); rows=job_fixture(spec)
    reviewed=d.artifact('RETIRE_PLAN',parents=dict(spec=spec['content_hash']),rows=rows,user='ryreu')
    monkeypatch.setattr(r,'plan',lambda *a:reviewed)
    monkeypatch.setattr(r,'check_live',lambda *a:None)
    monkeypatch.setattr(r,'accounting',lambda rs:{row['job_id']:state(row) for row in rs})
    r.retire(spec,'/alt')
    calls=[]
    def fail(cmd,**kw):
        calls.append(cmd)
        raise RuntimeError('connection lost after cancellation request')
    monkeypatch.setattr(r.subprocess,'run',fail)
    with pytest.raises(RuntimeError):
        r.retire(spec,'/alt',execute=True,plan_hash=reviewed['content_hash'],authorization=r.AUTHORIZATION)
    assert len(calls)==1 and not (Path(spec['root'])/'retirement/receipt.json').exists()
    with pytest.raises(FileExistsError):
        r.retire(spec,'/alt',execute=True,plan_hash=reviewed['content_hash'],authorization=r.AUTHORIZATION)


def test_accounting_parser_requires_all_ids_and_owner_checks(tmp_path,monkeypatch):
    _,_,spec=fixture(tmp_path,monkeypatch); rows=job_fixture(spec)[:2]
    text='\n'.join('|'.join([x['job_id'],x['job_name'],'ryreu',x['account'],x['partition'],
                           'CANCELLED by 2914834','0','Unknown']) for x in rows)
    monkeypatch.setattr(r.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=text))
    assert len(r.accounting(rows))==2
    text=text.splitlines()[0]
    with pytest.raises(ValueError,match='Missing exact'): r.accounting(rows)


def test_retirement_rejects_unrequested_campaign_before_scheduler_calls(tmp_path,monkeypatch):
    _,_,spec=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(r,'alternating',lambda *a:pytest.fail('Wrong sweep cannot inspect another campaign'))
    with pytest.raises(PermissionError,match='selected pending sweep'):
        r.plan(spec,'/some/other/campaign_spec.json')


def test_start_between_review_and_cancel_never_triggers_scancel(tmp_path,monkeypatch):
    _,_,spec=fixture(tmp_path,monkeypatch); rows=job_fixture(spec)
    reviewed=d.artifact('RETIRE_PLAN',parents=dict(spec=spec['content_hash']),rows=rows,user='ryreu')
    monkeypatch.setattr(r,'plan',lambda *a:reviewed)
    monkeypatch.setattr(r,'check_live',lambda *a:None)
    def accounting(rs):
        return {row['job_id']:state(row,'RUNNING' if len(rs)==8 else 'PENDING') for row in rs}
    monkeypatch.setattr(r,'accounting',accounting)
    r.retire(spec,'/alt')
    monkeypatch.setattr(r.subprocess,'run',lambda *a,**kw:pytest.fail('No mutation after raced start'))
    with pytest.raises(ValueError,match='started'):
        r.retire(spec,'/alt',execute=True,plan_hash=reviewed['content_hash'],authorization=r.AUTHORIZATION)
    assert not (Path(spec['root'])/'retirement/receipt.json').exists()


def test_debug_worker_resource_identity_is_still_enforced(monkeypatch):
    from hlt_classification.luka_attention import worker as allocation
    gate=dict(execution_site=execution_site('sporc_a100_debug'),cpus=6,memory_mb=90000)
    monkeypatch.setattr(allocation,'require_backend',lambda:None)
    monkeypatch.setattr(allocation,'allocation',lambda site:('77',6,90000))
    monkeypatch.setattr(allocation,'gpu_identity',lambda:dict(name='A100',capacity=40))
    monkeypatch.setattr(allocation,'installed_environment',lambda:dict(torch='same'))
    measured=dict(gpu=allocation.gpu_identity(),environment=allocation.installed_environment())
    assert d.execution(gate,measured)=='77'
    for changes in (dict(gpu={}),dict(environment={})):
        with pytest.raises(PermissionError,match='changed'): d.execution(gate,dict(measured,**changes))
    with pytest.raises(PermissionError,match='allocation'):
        d.execution(dict(gate,cpus=8),measured)


def test_real_installed_cuda_debug_worker_uses_exact_original_fit(tmp_path,monkeypatch):
    if not torch.cuda.is_available(): pytest.skip('Local CUDA fixture; not remote acceptance')
    small_counts(monkeypatch)
    train,bank,teacher=bank_fixture(tmp_path)
    val=cache('validation',coordinate='D000')
    root=tmp_path/'original'; root.mkdir()
    gate=fixture_gate(root)
    arrays=dict(identities=val.identities,labels=val.labels,
                probabilities=np.full((len(val),11),1/11,np.float32))
    atomic_publish_bytes(tmp_path/'reference.npz',deterministic_npz_bytes(arrays))
    ref=file_ref(tmp_path/'reference.npz')
    gate['anchor']['evidence']['models']={n:dict(checkpoint=teacher,predictions=ref) for n in ('M0HLT','OFFLINE')}
    gate['anchor']=with_content_hash(gate['anchor'])
    gate['embedding'].update(bank=file_ref(tmp_path/'bank/manifest.json'),
        payload=file_ref(tmp_path/'bank/targets.npz'),predictions=ref)
    gate['embedding']['parents']['anchor']=gate['anchor']['content_hash']
    gate['embedding']=with_content_hash(gate['embedding'])
    gate['parents'].update(anchor=gate['anchor']['content_hash'],embedding=gate['embedding']['content_hash'])
    gate=with_content_hash(gate)
    report=dict(content_hash='a'*64,bank_hash=bank.manifest['content_hash'],
                train_minutes={a:200 for a in c.WEIGHTS})
    gate,report,spec=fixture(tmp_path,monkeypatch,gate,report)
    monkeypatch.setattr(r,'confirm_retired',lambda *a:None)
    monkeypatch.setattr(d,'validate_source_snapshot',lambda *a,**kw:None)
    monkeypatch.setattr(d,'execution',lambda g,m:'22' if g['execution_site']['partition']=='debug' else pytest.fail('wrong site'))
    monkeypatch.setattr(d.regular_cache,'metadata',lambda *a:None)
    monkeypatch.setattr(d,'data',lambda g,role,*a:train if role=='train' else val)
    # Linux RSS accounting is not available on this Windows fixture host.
    monkeypatch.setattr(d,'memory_snapshot',lambda t:dict(rss_bytes=2**30,peak_rss_bytes=2**30,
        gpu_peak_bytes=int(t.cuda.max_memory_allocated())))
    monkeypatch.setattr(worker.alljets_data,'comparison_indices',lambda *a:np.arange(len(val),dtype=np.int64))
    worker.snapshot_references(gate,val,({}, {}, {}, {}))
    # Only remote acceptance is a fixture; production ParT, 30-epoch fit/readback and reporting are real.
    monkeypatch.setattr(preflight,'validate_report',lambda *a:None)
    assert d.results(spec)['pending']==list(c.WEIGHTS)
    for arm in ('w3','w1000'):
        result=d.run(spec,arm)
        assert result['weight']==c.WEIGHTS[arm]
        scoped=c.arm_gate(gate,arm)
        worker.fit(tmp_path/('donor_'+arm),scoped,train,val,bank,device='cuda',technical=False)
        actual=worker.validate_fit(Path(spec['root'])/arm/'fit',scoped,bank.manifest['content_hash'],technical=False)
        expected=worker.validate_fit(tmp_path/('donor_'+arm),scoped,bank.manifest['content_hash'],technical=False)
        np.testing.assert_allclose(actual[1]['probabilities'],expected[1]['probabilities'],rtol=1e-5,atol=1e-6)
        assert actual[0]['selected_pass']==expected[0]['selected_pass']
    monkeypatch.setattr(torch,'load',lambda *a,**kw:pytest.fail('No model loading for results'))
    assert len(d.results(spec)['rows'])==5
    pred=Path(spec['root'])/'w3/fit/validation_predictions.npz'
    pred.write_bytes(pred.read_bytes()+b'corrupt')
    with pytest.raises(ValueError): d.results(spec)


def test_shell_cli_and_runbook_syntax():
    import re
    project=Path(__file__).resolve().parents[1]
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    if not bash.is_file(): pytest.skip('Bash not available')
    for name in ('scripts/queue_luka_embedding_sweep_debug.sh','sbatch/run_luka_embedding_sweep_debug.sh'):
        subprocess.run([str(bash),'-n',name],cwd=project,check=True,capture_output=True)
        text=(project/name).read_text()
        for token in ('PYTHONNOUSERSITE=1','PYTHONDONTWRITEBYTECODE=1','CUBLAS_WORKSPACE_CONFIG=:4096:8','${CONDA_PREFIX}/lib'):
            assert token in text
    text=(project/'docs/LUKA_EMBEDDING_SWEEP_DEBUG_RUNBOOK.md').read_text()
    for block in re.findall(r'```bash\n(.*?)\n```',text,re.S):
        subprocess.run([str(bash),'-n'],input=block,text=True,cwd=project,check=True,capture_output=True)
    for name in re.findall(r'\]\(([^)]+)\)',text):
        if not name.startswith('https://'): assert (project/'docs'/name).is_file()

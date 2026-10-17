"""Local recovery/controller tests; scheduler and GPU acceptance are simulated."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import subprocess

import pytest

from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash
from hlt_classification.luka_fullstat import recovery as r, campaign as c, submission as s, worker
from hlt_classification.luka_fullstat.contracts import artifact
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag


@pytest.fixture
def gate(tmp_path):
    return artifact('RECOVERY_GATE', parents={}, root=str(tmp_path), project_dir='/oscar/pinned',
        retry_indices=r.RETRY, excluded_node=r.EXCLUDED_NODE, **{**c.policy(), 'automatic_followup': True})


def test_sparse_plan_does_not_rebuild_or_change_old_policy(gate):
    plan = s.render(gate)
    assert (plan['jobs'],plan['scheduled_tasks']) == (3,5)
    assert [x['task_id'] for x in plan['commands']] == ['prepare','assemble','preflight']
    prepare, assemble, preflight = plan['commands']
    assert prepare['dependencies'] == []
    assert '--array=12-14%3' in prepare['command']
    assert '--exclude=node1804' in prepare['command']
    assert not any(a.startswith('--dependency=') for a in prepare['command'])
    assert '--dependency=afterok:${JOB_prepare}' in assemble['command']
    assert '--dependency=afterok:${JOB_assemble}' in preflight['command']
    assert '--gres=gpu:l40s:1' in preflight['command']
    assert '--mem=180000M' in preflight['command']
    assert c.policy()['automatic_followup'] is False
    assert gate['training_recipe']['maximum_passes'] == 20


def test_accounting_identity_array_and_exit_guards():
    expected = r.expectations(dict(prepare='20',assemble='21',preflight='22'), r.RETRY)
    text = '\n'.join(f'{job}|{name}|rlyang|default|COMPLETED|0:0|' for job,name in expected.items())
    states = r.read_states(text+'\n20|lukafs_prepare|rlyang|default|COMPLETED|0:0|',expected,'rlyang')
    assert r.gate_ready(states,expected)
    for job in expected:
        assert not r.gate_ready({k:v for k,v in states.items() if k != job},expected)
        assert not r.gate_ready({**states,job:('RUNNING','0:0')},expected)
        for bad in [('FAILED','1:0'),('CANCELLED','0:0'),('COMPLETED','1:0'),('OUT_OF_MEMORY','0:125')]:
            with pytest.raises(PermissionError):r.gate_ready({**states,job:bad},expected)
    with pytest.raises(PermissionError):r.read_states(text.replace('rlyang','someone_else'),expected,'rlyang')
    with pytest.raises(PermissionError):r.read_states(text.replace('default','wrong'),expected,'rlyang')
    with pytest.raises(ValueError):r.read_states(text+'\n'+text,expected,'rlyang')


def test_original_accounting_requires_exact_old_outcome(monkeypatch):
    expected = r.expectations(r.ORIGINAL_JOBS, range(16), foundation=True)
    states = {j:('COMPLETED','0:0') for j in expected}
    for j in ('7486669','7486670','7486668_12','7486668_13','7486668_14'):
        states[j] = ('CANCELLED','0:0')
    monkeypatch.setattr(r,'accounting',lambda _:states)
    assert r.check_original() == states
    states['7486668_12'] = ('PENDING','0:0')
    with pytest.raises(PermissionError):r.check_original()


def test_fresh_missing_outputs_and_no_overwrite(tmp_path):
    origin = dict(root=str(tmp_path/'old'),cache_root=str(tmp_path/'cache'))
    r.missing_outputs(origin)
    partial = tmp_path/'cache/chunks/f13'
    partial.mkdir(parents=True)
    with pytest.raises(FileExistsError,match='partial'):r.missing_outputs(origin)
    assert partial.is_dir()


def test_cross_root_claim_and_ambiguous_claim_refused(tmp_path,monkeypatch,gate):
    old = tmp_path/'old'; old.mkdir()
    monkeypatch.setattr(c,'foundation_gate',lambda _:dict(root=str(old),content_hash='a'*64))
    r.claim_original(gate)
    r.claim_original(gate)
    with pytest.raises(PermissionError):r.claim_original(with_content_hash({**gate,'root':str(tmp_path/'different')}))


def test_exact_ledger_checks_plan_journal_and_replies(gate,tmp_path,monkeypatch):
    monkeypatch.setattr(c,'validate_gate',lambda _:None)
    monkeypatch.setattr(s,'check_site',lambda _:None)
    monkeypatch.setattr(r,'check_original',lambda:None)
    monkeypatch.setattr(r,'missing_outputs',lambda _:None)
    monkeypatch.setattr(r,'claim_original',lambda _:None)
    monkeypatch.setattr(c,'foundation_gate',lambda _:None)
    calls=[]
    def scheduler(args,**kw):
        calls.append(args)
        return subprocess.CompletedProcess(args,0,str(100+len(calls))+';slurmctld\n','')
    monkeypatch.setattr(s.subprocess,'run',scheduler)
    plan=s.submit(gate)
    ledger=s.submit(gate,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert r.exact_ledger(gate)==ledger
    assert s.submit(gate,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)==ledger
    assert len(calls)==3
    journal=tmp_path/'gate_submission/submission_ledger_journal/0001_assemble.json'
    # A well-hashed but alien event must not be accepted.
    bad=with_content_hash({**load_json(journal),'job_id':'999'})
    from hlt_classification.data.cache_contracts import canonical_json_bytes
    journal.write_bytes(canonical_json_bytes(bad))
    with pytest.raises(ValueError):r.exact_ledger(gate)


def test_science_scope_and_all_sixteen_template_commands(gate,monkeypatch):
    monkeypatch.setattr(r,'validate_gate',lambda _:None)
    monkeypatch.setattr(c,'validate_gate',lambda _:None)
    monkeypatch.setattr(c,'validate_spec',lambda _:None)
    monkeypatch.setattr(s,'check_site',lambda _:None)
    reviewed=r.review(gate)
    assert reviewed['recovery_plan']==s.render(gate)
    assert len(reviewed['science_template']['commands'])==16
    spec=artifact('RECOVERY_SPEC',parents=dict(gate=gate['content_hash']),root=str(Path(gate['root'])/'science'),
                  project_dir=gate['project_dir'],train_minutes=400,reduce_minutes=40,**c.policy())
    plan=s.render(spec)
    r.verify_science(gate,spec,plan,reviewed)
    for name, value in [('train_minutes',2881),('reduce_minutes',0),('memory_mb',250000)]:
        bad=with_content_hash({**spec,name:value})
        with pytest.raises(PermissionError):r.verify_science(gate,bad,s.render(bad),reviewed)
    bad=deepcopy(plan); bad['commands'][3]['dependencies']=[]
    with pytest.raises(PermissionError):r.verify_science(gate,spec,with_content_hash(bad),reviewed)


@contextmanager
def fake_lock(path):
    path.mkdir(exist_ok=True)
    yield


def controller(gate,monkeypatch,*,failure=None,missing=False):
    """Real durable submission DAG/claims/controller; mock remote checks only."""
    from hlt_classification import literature_ladder_followup
    monkeypatch.setattr(literature_ladder_followup,'lock',fake_lock)
    monkeypatch.setattr(r,'validate_gate',lambda _:None)
    monkeypatch.setattr(c,'validate_gate',lambda _:None)
    monkeypatch.setattr(c,'validate_spec',lambda _:None)
    monkeypatch.setattr(s,'check_site',lambda _:None)
    monkeypatch.setattr(r,'check_original',lambda:None)
    monkeypatch.setattr(r,'missing_outputs',lambda _:None)
    monkeypatch.setattr(r,'claim_original',lambda _:None)
    monkeypatch.setattr(c,'foundation_gate',lambda _:dict(root='unused'))
    monkeypatch.setattr(r.time,'sleep',lambda _:None)
    calls=[]
    def scheduler(args,**kw):
        calls.append(args)
        return subprocess.CompletedProcess(args,0,str(200+len(calls))+'\n','')
    monkeypatch.setattr(s.subprocess,'run',scheduler)
    def account(expected):
        states={j:('COMPLETED','0:0') for j in expected}
        if failure:
            target=next(j for j in expected if failure in expected[j])
            states[target]=('FAILED','1:0')
        return {} if missing else states
    monkeypatch.setattr(r,'accounting',account)
    def admit(current):
        r.completed_gate(current)
        spec=artifact('RECOVERY_SPEC',parents=dict(gate=current['content_hash']),
            root=str(Path(current['root'])/'science'),project_dir=current['project_dir'],
            train_minutes=500,reduce_minutes=50,**c.policy())
        Path(spec['root']).mkdir(exist_ok=False)
        write_immutable_json(Path(spec['root'])/'campaign_spec.json',spec)
        return spec
    monkeypatch.setattr(c,'admit',admit)
    reviewed=r.review(gate)
    return calls,reviewed


@pytest.mark.parametrize('failure',['prepare','assemble','preflight'])
def test_no_science_on_any_gate_failure(gate,monkeypatch,failure):
    calls,reviewed=controller(gate,monkeypatch,failure=failure)
    with pytest.raises(PermissionError,match='Gate failed'):
        r.run(gate,review_hash=reviewed['content_hash'],authorization=r.AUTHORIZE)
    assert len(calls)==3
    assert not (Path(gate['root'])/'science').exists()


def test_accounting_absence_timeout_does_not_submit_science(gate,monkeypatch):
    calls,reviewed=controller(gate,monkeypatch,missing=True)
    with pytest.raises(TimeoutError):
        r.run(gate,review_hash=reviewed['content_hash'],authorization=r.AUTHORIZE,timeout_seconds=0)
    assert len(calls)==3


def test_controller_success_and_completed_replay_are_duplicate_free(gate,monkeypatch):
    calls,reviewed=controller(gate,monkeypatch)
    kwargs=dict(review_hash=reviewed['content_hash'],authorization=r.AUTHORIZE)
    result=r.run(gate,**kwargs)
    assert len(calls)==19 and len(result['jobs'])==16
    assert r.run(gate,**kwargs)==result
    assert len(calls)==19
    assert (Path(gate['root'])/'followup/submitted.json').is_file()


@pytest.mark.parametrize('field',['hash','authorization'])
def test_wrong_authorization_submits_nothing(gate,monkeypatch,field):
    calls,reviewed=controller(gate,monkeypatch)
    with pytest.raises(PermissionError):
        r.run(gate,review_hash='0'*64 if field=='hash' else reviewed['content_hash'],
              authorization='wrong' if field=='authorization' else r.AUTHORIZE)
    assert calls==[]


def test_ambiguous_science_reply_is_not_retried(gate,monkeypatch):
    calls,reviewed=controller(gate,monkeypatch)
    original=s.subprocess.run
    def lose_reply(args,**kw):
        if len(calls)>=4:raise RuntimeError('scheduler reply lost')
        return original(args,**kw)
    monkeypatch.setattr(s.subprocess,'run',lose_reply)
    kwargs=dict(review_hash=reviewed['content_hash'],authorization=r.AUTHORIZE)
    with pytest.raises(RuntimeError):r.run(gate,**kwargs)
    with pytest.raises(PermissionError,match='interrupted'):r.run(gate,**kwargs)
    assert len(calls)==4


def test_successful_accounting_without_valid_preflight_does_not_admit(gate,monkeypatch):
    calls,reviewed=controller(gate,monkeypatch)
    def invalid(_):raise ValueError('corrupt/missing preflight evidence')
    monkeypatch.setattr(c,'admit',invalid)
    with pytest.raises(ValueError,match='preflight'):
        r.run(gate,review_hash=reviewed['content_hash'],authorization=r.AUTHORIZE)
    assert len(calls)==3


def test_ambiguous_gate_submission_is_not_retried(gate,monkeypatch):
    calls,reviewed=controller(gate,monkeypatch)
    original=s.subprocess.run
    def lose_reply(args,**kw):
        if calls:raise RuntimeError('gate reply lost')
        return original(args,**kw)
    monkeypatch.setattr(s.subprocess,'run',lose_reply)
    kwargs=dict(review_hash=reviewed['content_hash'],authorization=r.AUTHORIZE)
    with pytest.raises(RuntimeError):r.run(gate,**kwargs)
    with pytest.raises(PermissionError,match='interrupted'):r.run(gate,**kwargs)
    assert len(calls)==1


def test_recovery_worker_refuses_other_indices_and_foundation(gate,monkeypatch):
    monkeypatch.setattr(r,'validate_gate',lambda _:dict())
    for index in [0,11,15,True,None]:
        with pytest.raises(PermissionError):worker.gate_task(gate,'prepare',index)
    with pytest.raises(PermissionError):worker.gate_task(gate,'foundation')


def test_compatibility_refuses_unreviewed_transform(monkeypatch):
    monkeypatch.setattr(r,'git',lambda project,*args: 'src/hlt_classification/luka_fullstat/cache.py' if args[0]=='diff' else '')
    with pytest.raises(ValueError,match='unreviewed'):r.compatibility('/fixture')


# Reuse real tiny ROOT/membership fixtures; this does NOT certify remote data.
from test_luka_fullstat import fullstat


def test_real_cache_reuse_recovery_admission_and_corruption(fullstat,tmp_path,monkeypatch):
    import shutil
    from hlt_classification.luka_fullstat import cache, preflight
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    container, foundation_path, foundation = fullstat
    old = tmp_path/'old'; old.mkdir()
    shutil.copytree(foundation_path,old/'foundation')
    cache_root=tmp_path/'cache'; cache_root.mkdir()
    project=tmp_path/'project'; project.mkdir()
    output=tmp_path/'recovery'
    base=c.policy()
    local_policy={**base,'counts':foundation['counts'],'expected_files':3,
                  'expected_raw_rows':foundation['raw_entries'],'preparation_workers':1}
    monkeypatch.setattr(c,'policy',lambda batch=128:deepcopy(local_policy))
    monkeypatch.setattr(r,'RETRY',[1]); monkeypatch.setattr(r,'REUSED',[0,2])
    snapshot=dict(content_hash='b'*64)
    origin=dict(content_hash='a'*64,root=str(old),container=str(container),cache_root=str(cache_root),
                source_snapshot=snapshot,source_commit=r.PRODUCER,transfer=foundation['transfer'],
                source_records=[{k:f[k] for k in ('path','sha256','category')} for f in foundation['files']],
                **local_policy)
    write_immutable_json(old/'gate_spec.json',origin)
    old_ledger=dict(content_hash='c'*64,jobs=r.ORIGINAL_JOBS)
    write_immutable_json(old/'gate_submission/submission_ledger.json',old_ledger)
    monkeypatch.setattr(r,'original',lambda path:(origin,old_ledger))
    monkeypatch.setattr(c,'source',lambda *args:snapshot)
    monkeypatch.setattr(r,'compatibility',lambda _:dict(fixture=True))
    monkeypatch.setattr(r,'check_original',lambda:None)
    for i in (0,2):cache.prepare_file(container,old/'foundation',cache_root,i,workers=1)
    before={i:(cache_root/f'files/{i:02d}/receipt.json').read_bytes() for i in (0,2)}
    gate=r.create(old/'gate_spec.json',output,project=project,commit='d'*40)
    assert r.validate_gate(gate)==origin
    with pytest.raises(FileExistsError):r.create(old/'gate_spec.json',output,project=project,commit='d'*40)
    worker.gate_task(gate,'prepare',1)
    manifest=worker.gate_task(gate,'assemble')
    assert manifest['parents']['gate']==origin['content_hash']
    assert c.authenticated_cache(gate)[0]==manifest
    assert before=={i:(cache_root/f'files/{i:02d}/receipt.json').read_bytes() for i in (0,2)}
    evidence=artifact('PREFLIGHT',parents=dict(gate=gate['content_hash'],cache=manifest['content_hash']),
                      job_id='999',train_minutes=400,reduce_minutes=40)
    write_immutable_json(output/'preflight/preflight.json',evidence)
    ledger=dict(jobs=dict(preflight='999'))
    write_immutable_json(output/'gate_submission/submission_ledger.json',ledger)
    monkeypatch.setattr(c,'accounting',lambda _:ledger)
    monkeypatch.setattr(c,'gate_ledger',lambda _:ledger)
    def evidence_check(value,current,prepared):
        assert value==evidence and current==gate and prepared==manifest
    monkeypatch.setattr(preflight,'validate_evidence',evidence_check)
    spec=c.admit(gate)
    assert spec['contract']=='LUKA_FULLSTAT_RECOVERY_SPEC/v1'
    assert spec['foundation_root']==str(old/'foundation')
    assert c.validate_spec(spec)[2]==evidence
    assert len(s.render(spec)['commands'])==16
    with pytest.raises(ValueError,match='registration'):
        c.validate_spec(with_content_hash({**spec,'contract':'LUKA_FULLSTAT_SPEC/v1'}))
    ledger['jobs']['preflight']='998'
    with pytest.raises(ValueError,match='evidence'):c.validate_spec(spec)
    ledger['jobs']['preflight']='999'
    # Full physical assembly catches changed reused bytes even with an intact
    # receipt; receipt-only inspection is never represented as payload validation.
    payload=cache_root/manifest['chunks'][0]['views']['D000']['path']
    with payload.open('ab') as stream:stream.write(b'corrupt')
    with pytest.raises(ValueError):worker.gate_task(gate,'assemble')
    row=cache_root/'files/00/receipt.json'
    with row.open('ab') as stream:stream.write(b' ')
    with pytest.raises(ValueError):r.validate_gate(gate)

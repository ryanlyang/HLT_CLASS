"""Exact submit journals, bounded ordering and failure/ambiguous hold recovery."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_luka_d066_embedding import fixture_gate
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.luka_d066_embedding import contracts as c, campaign, preflight, submission as s, followup as f, priority as p


@pytest.fixture
def controller(tmp_path,monkeypatch):
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    gate=fixture_gate(tmp_path); write_immutable_json(tmp_path/'gate_spec.json',gate)
    monkeypatch.setattr(campaign,'check_gate',lambda _:None)
    monkeypatch.setattr(s,'check_site',lambda _:None)
    monkeypatch.setattr(f,'controller_lock',lambda _:nullcontext())
    monkeypatch.setattr(f.getpass,'getuser',lambda:'owner')
    control=dict(gate_state='COMPLETED',fit_state='COMPLETED',bad_receipt=False,fail_submit=None,
                 fail_hold=None,hold_race=None,science_validated=False)
    calls=[]; holds=[]; releases=[]
    rows=[dict(job_id=job,task_id=arm,job_name='lukaesd_'+arm,partition='debug') for arm,job in p.SELECTED.items()]
    ordering=c.artifact('PRIORITY_PLAN',parents=dict(gate=gate['content_hash']),rows=rows,user='owner')
    monkeypatch.setattr(p,'plan',lambda _:ordering)
    states={row['job_id']:dict(JobState='PENDING',RunTime='00:00:00',Priority='123',Reason='Priority') for row in rows}
    monkeypatch.setattr(p,'live',lambda row,user:deepcopy(states[row['job_id']]))
    def scheduler(args,**kw):
        if args[0]=='scontrol':
            job=args[2]
            if args[1]=='hold':
                holds.append(job); states[job].update(Priority='0',Reason='JobHeldUser')
                if control['hold_race']==job: states[job]['JobState']='RUNNING'
                if control['fail_hold']==job: raise RuntimeError('lost hold response')
            elif args[1]=='release':
                releases.append(job); states[job].update(Priority='123',Reason='Priority')
            else: pytest.fail('Unexpected mutation')
            return SimpleNamespace(stdout='')
        if args[0]=='sacct':
            job=args[args.index('-j')+1]; task='preflight' if job=='12345' else 'fit'
            state=control['gate_state'] if task=='preflight' else control['fit_state']
            if task=='preflight' and state=='COMPLETED' and not (tmp_path/'preflight/receipt.json').exists():
                write_immutable_json(tmp_path/'preflight/receipt.json',c.artifact('PREFLIGHT',
                    job_id=job,train_minutes=96,technical_checks_passed=True))
            return SimpleNamespace(stdout=f'{job}|lukad6e_{task}|owner|reu-aisocial|{state}|0:0\n')
        assert args[0]=='sbatch' and '--partition=debug' in args
        calls.append(args)
        if len(calls)>1: assert control['science_validated']
        if len(calls)==control['fail_submit']: raise RuntimeError('lost sbatch response')
        return SimpleNamespace(stdout=str(12344+len(calls))+'\n')
    monkeypatch.setattr(exact.subprocess,'run',scheduler)
    def valid(gate,report):
        if control['bad_receipt']: raise ValueError('Invalid resource receipt')
        control['science_validated']=True
    monkeypatch.setattr(preflight,'validate_report',valid)
    monkeypatch.setattr(campaign,'preflight_accounting',lambda _:None)
    return SimpleNamespace(gate=gate,root=tmp_path,calls=calls,holds=holds,releases=releases,control=control,states=states)


def execute(t,*,ordered=True,review=None):
    review=review or f.run(t.gate,order_pending=ordered)
    return f.run(t.gate,order_pending=ordered,execute=True,plan_hash=review['content_hash'],
                 authorization=f.AUTHORIZATION,priority_authorization=p.AUTHORIZATION if ordered else None)


def test_dry_authorization_and_success_exact_two_jobs(controller):
    t=controller; review=f.run(t.gate,order_pending=True)
    assert not t.calls and not t.holds and review['science_jobs']==1 and review['max_total_jobs']==2
    assert '--time=1-00:00:00' in review['science_ceiling_plan']['commands'][0]['command']
    with pytest.raises(PermissionError):
        f.run(t.gate,order_pending=True,execute=True,plan_hash=review['content_hash'],authorization=f.AUTHORIZATION)
    assert not t.holds
    done=execute(t,review=review)
    assert done['fit_job']=='12346' and len(t.calls)==2
    assert t.holds==t.releases==list(p.SELECTED.values())
    assert '--time=01:36:00' in t.calls[1]
    with pytest.raises(FileExistsError): execute(t,review=review)
    assert len(t.calls)==2


def test_existing_gate_not_submitted_again_and_no_order_option(controller):
    t=controller; review=s.submit(t.gate)
    s.submit(t.gate,execute=True,plan_hash=review['content_hash'],authorization=c.AUTHORIZATION)
    execute(t,ordered=False)
    assert len(t.calls)==2 and not t.holds


@pytest.mark.parametrize('state',['FAILED','OUT_OF_MEMORY','CANCELLED by 1234','TIMEOUT'])
def test_gate_failure_releases_holds_and_never_creates_science(controller,state):
    t=controller; t.control['gate_state']=state
    with pytest.raises(RuntimeError): execute(t)
    assert len(t.calls)==1 and not (t.root/'science').exists()
    assert t.holds==t.releases


def test_bad_receipt_and_fit_failure_release(controller):
    t=controller; t.control['bad_receipt']=True
    with pytest.raises(ValueError,match='resource'): execute(t)
    assert len(t.calls)==1 and t.holds==t.releases


def test_fit_failed_after_submission_releases(controller):
    t=controller; t.control['fit_state']='FAILED'
    with pytest.raises(RuntimeError): execute(t)
    assert len(t.calls)==2 and t.holds==t.releases


@pytest.mark.parametrize('which',[1,2])
def test_lost_sbatch_reply_no_automatic_retry(controller,which):
    t=controller; t.control['fail_submit']=which; review=f.run(t.gate,order_pending=True)
    with pytest.raises(RuntimeError): execute(t,review=review)
    with pytest.raises((PermissionError,ValueError,FileExistsError)): execute(t,review=review)
    assert len(t.calls)==which and t.holds==t.releases


def test_no_takeover_of_preheld_jobs_and_leave_running_unchanged(controller):
    t=controller
    t.states['21940433']['JobState']='RUNNING'
    execute(t)
    assert '21940433' not in t.holds and '21940433' not in t.releases
    assert set(t.holds)==set(p.SELECTED.values())-{'21940433'}


def test_preexisting_user_hold_fails_before_mutation(controller):
    t=controller; t.states['21940434'].update(Priority='0',Reason='JobHeldUser')
    with pytest.raises(PermissionError,match='Already held'): execute(t)
    assert not t.calls and not t.holds


def test_hold_start_race_is_released_without_preempting(controller):
    t=controller; t.control['hold_race']='21940434'
    with pytest.raises(RuntimeError,match='changed during hold'): execute(t)
    assert not t.calls and t.holds==t.releases==['21940433','21940434']
    assert t.states['21940434']['JobState']=='RUNNING'


def test_ambiguous_hold_not_retried_or_claimed(controller):
    t=controller; t.control['fail_hold']='21940434'
    with pytest.raises(RuntimeError,match='ambiguous hold'): execute(t)
    assert not t.calls and t.holds==['21940433','21940434'] and t.releases==['21940433']
    with pytest.raises(RuntimeError,match='ambiguous hold'): f.recover_holds(t.gate)
    assert t.releases==['21940433']


def test_live_scheduler_identity_guard(monkeypatch):
    row=dict(job_id='21940433',job_name='lukaesd_w50',partition='debug',account='reu-aisocial',
        qos='qos_tier3',comment='bound',workdir='/project',command='/script',stdout='/log')
    text='JobId=21940433 JobName=lukaesd_w50 UserId=owner(123) Partition=debug Account=reu-aisocial '
    text+='QOS=qos_tier3 Comment=bound WorkDir=/project Command=/script StdOut=/log Requeue=0 Restarts=0 '
    text+='JobState=PENDING RunTime=00:00:00 Priority=123 Reason=Priority'
    monkeypatch.setattr(p.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=text))
    assert p.pending(p.live(row,'owner'))
    for field,value in [('partition','tier3'),('job_id','21940431'),('comment','other'),('command','/Fresh_check')]:
        with pytest.raises(PermissionError): p.live(dict(row,**{field:value}),'owner')
    with pytest.raises(PermissionError): p.live(row,'another_user')


@pytest.mark.parametrize('raw,error',[
    ('12345|lukad6e_preflight|owner|reu-aisocial|COMPLETED|1:0',RuntimeError),
    ('12345|lukad6e_preflight|other|reu-aisocial|COMPLETED|0:0',PermissionError),
    ('12345|other|owner|reu-aisocial|COMPLETED|0:0',PermissionError),
    ('garbled',ValueError),
])
def test_accounting_and_missing_not_success(raw,error):
    with pytest.raises(error): f.accounting(raw,job='12345',task='preflight',user='owner',account='reu-aisocial')
    assert f.accounting('',job='12345',task='preflight',user='owner',account='reu-aisocial')=='ACCOUNTING_PENDING'


def test_source_or_graph_change_cannot_expand_science(controller,monkeypatch):
    t=controller; review=f.run(t.gate,order_pending=False)
    spec=dict(f.ceiling_spec(t.gate),content_hash='a'*64,train_minutes=90)
    monkeypatch.setattr(campaign,'check_spec',lambda _:(t.gate,{}))
    assert f.review_science(t.gate,spec,review)['jobs']==1
    for key,value in [('root','/wrong'),('project_dir','/other'),('train_minutes',1441),('train_minutes',True)]:
        with pytest.raises(ValueError): f.review_science(t.gate,dict(spec,**{key:value}),review)


def test_manual_recovery_cannot_race_live_controller(controller,monkeypatch):
    t=controller
    def locked(_): raise BlockingIOError('Controller still owns lock')
    monkeypatch.setattr(f,'controller_lock',locked)
    with pytest.raises(BlockingIOError): f.recover_holds(t.gate)
    assert not t.releases


def test_exact_old_journal_binds_only_requested_five_ids(tmp_path,monkeypatch):
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    from hlt_classification.jetclass2_delphes.execution import execution_site
    from hlt_classification.luka_embedding_sweep.contracts import WEIGHTS
    root=tmp_path/'old'; root.mkdir()
    old=p.debug.artifact('SPEC',parents=dict(original=p.SWEEP_HASH),root=str(root),project_dir='/old-source',
        execution_site=execution_site('sporc_a100_debug'),cpus=6,memory_mb=90000,
        train_minutes={a:120 for a in WEIGHTS},arms=list(WEIGHTS))
    path=root/'campaign_spec.json'; write_immutable_json(path,old)
    plan=p.debug.render(old); folder=root/'submission'; folder.mkdir()
    write_immutable_json(folder/'command_plan.json',plan)
    dry=folder/'dry.json'; ledger=folder/'submission_ledger.json'
    ids=iter(str(i) for i in range(21940430,21940438))
    monkeypatch.setattr(exact.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=next(ids)))
    exact.submit_exact_dag(identity=old['content_hash'],plan=plan,output=dry,canonical_dry_run=dry,execute=False)
    exact.submit_exact_dag(identity=old['content_hash'],plan=plan,output=ledger,canonical_dry_run=dry,execute=True)
    gate=dict(content_hash='a'*64,reference=dict(spec=file_ref(path)))
    bound=p.plan(gate)
    assert {r['task_id']:r['job_id'] for r in bound['rows']}==p.SELECTED
    assert not {'21940430','21940431','21940432'} & {r['job_id'] for r in bound['rows']}
    ledger.write_bytes(b'corrupted ledger')
    with pytest.raises(ValueError): p.plan(gate)


def test_real_linux_controller_lock_excludes_concurrent_recovery(tmp_path):
    import sys
    if sys.platform!='linux': pytest.skip('Real Linux flock exercised separately on WSL')
    (tmp_path/'followup').mkdir()
    gate=dict(root=str(tmp_path))
    with f.controller_lock(gate):
        with pytest.raises(BlockingIOError):
            with f.controller_lock(gate): pass
    with f.controller_lock(gate): pass

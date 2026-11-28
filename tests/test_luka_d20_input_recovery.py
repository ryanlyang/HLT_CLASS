"""Native-reader equality, bounded retries and selective live-campaign recovery."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import errno
import multiprocessing
import subprocess
import threading
import time
import numpy as np
import pytest
import uproot
from hlt_classification.luka_d20_input_recovery import contracts as c,reader,decode,runtime,controller,probe
from hlt_classification.luka_d20_full_hydra import basket_reader as original_reader,inputs,population
from hlt_classification.luka_d20_full_hydra import campaign as campaign,contracts as science
from hlt_classification.data.cache_contracts import sha256_file,with_content_hash
from test_luka_fullstat import write_root


def arguments(tmp_path,n=6):
    path=tmp_path/'raw.bin';path.write_bytes(b'authenticated source')
    record=dict(path=path.name,sha256=c.digest(path))
    return [(str(tmp_path),record,0,i,i+1,0,np.ones(1,bool),'D000',False,'a'*64,{}) for i in range(n)]


def singles(args):return ([arg] for arg in args)


def echo(batch):return [arg[3] for arg in batch]


class FakeContext:
    def __init__(self,outcome):self.pools=[];self.outcome=outcome;self.calls=[]
    def Pool(self,*args,**kwargs):
        owner=self;generation=len(self.pools)
        class Pool:
            terminated=False
            def apply_async(self,function,args):
                batch=args[1];owner.calls.append((generation,batch[0][3],args[2]))
                class Future:
                    def get(self,timeout):
                        failure=owner.outcome(generation,batch[0][3])
                        if failure is not None:raise failure
                        return function(*args)
                return Future()
            def terminate(self):self.terminated=True
            def join(self):pass
        pool=Pool();self.pools.append(pool);return pool


def test_retry_preserves_accepted_prefix_order_and_checksums(tmp_path,monkeypatch):
    args=arguments(tmp_path);ctx=FakeContext(lambda g,i:multiprocessing.TimeoutError() if g==0 and i==1 else None)
    hashes=[];native=c.digest
    monkeypatch.setattr(c,'digest',lambda p:(hashes.append(str(p)) or native(p)))
    events=[]
    actual=list(decode.decoded(args,workers=3,function=echo,batcher=singles,context=ctx,on_retry=events.append))
    assert [value for _,value in actual]==list(range(6))
    assert sum(i==0 for _,i,_ in ctx.calls)==1  # Accepted prefix never replays.
    assert all(recheck for generation,i,recheck in ctx.calls if generation==1 and i in (1,2,3))
    assert len(hashes)==3 and len(events)==1 and events[0]['replayed_batches']==3
    assert len(ctx.pools)==2 and all(p.terminated for p in ctx.pools)


@pytest.mark.parametrize('error',[ValueError('invalid data'),MemoryError('RAM'),
    PermissionError(errno.EACCES,'forbidden'),FileNotFoundError(errno.ENOENT,'missing')])
def test_invalid_inputs_are_never_retried(tmp_path,error):
    ctx=FakeContext(lambda g,i:error)
    with pytest.raises(type(error)):
        list(decode.decoded(arguments(tmp_path),workers=2,function=echo,batcher=singles,context=ctx))
    assert len(ctx.pools)==1 and ctx.pools[0].terminated


def test_io_retry_is_bounded_and_checksum_mismatch_fails_closed(tmp_path):
    args=arguments(tmp_path)
    def fail(g,i):
        if g==0:
            (tmp_path/'raw.bin').write_bytes(b'changed')
            return OSError(errno.EIO,'temporary filesystem error')
    ctx=FakeContext(fail)
    with pytest.raises(ValueError,match='checksum differs'):
        list(decode.decoded(args,workers=2,function=echo,batcher=singles,context=ctx))
    assert len(ctx.pools)==2 and all(p.terminated for p in ctx.pools)


def test_repeated_timeouts_exhaust_exactly_two_restarts(tmp_path):
    ctx=FakeContext(lambda g,i:multiprocessing.TimeoutError())
    with pytest.raises(TimeoutError,match='2 pool restarts.*no rows skipped'):
        list(decode.decoded(arguments(tmp_path),workers=2,function=echo,batcher=singles,context=ctx))
    assert len(ctx.pools)==3 and all(p.terminated for p in ctx.pools)


def test_unreapable_pool_never_retries_or_blocks_indefinitely():
    event=threading.Event()
    pool=SimpleNamespace(terminate=lambda:event.wait(5),join=lambda:None)
    try:
        begun=time.monotonic()
        with pytest.raises(decode.UnreapedPoolError):decode.stop(pool,timeout=.02)
        assert time.monotonic()-begun<1
    finally:event.set()


def test_spawned_numeric_batches_and_timeout_cleanup(tmp_path):
    args=arguments(tmp_path,4);before={p.pid for p in multiprocessing.active_children()}
    result=list(decode.decoded(args,workers=2,function=list,batcher=singles,task_timeout=60,max_restarts=0))
    assert [arg[3] for _,arg in result]==list(range(4))
    with pytest.raises(TimeoutError,match='no rows skipped'):
        list(decode.decoded(args,workers=2,function=list,batcher=singles,task_timeout=.00001,max_restarts=0))
    assert not ({p.pid for p in multiprocessing.active_children()}-before)


@pytest.mark.parametrize('coordinate,targets',[('D000',True),('D000',False),('OFFLINE',False),('U100',False)])
def test_native_source_full_builder_is_byte_equal_and_role_bounded(tmp_path,monkeypatch,coordinate,targets):
    path=tmp_path/'raw/a.root';write_root(path,[1]*6)
    record=dict(path='raw/a.root',sha256=sha256_file(path),tree_key='tree',entries=6)
    roles=np.array([0,1,2,3,3,0],np.uint8)
    member=dict(labels=np.array([1,2,0,3,5,4]),jet_pt=np.full(6,250.),roles=roles,
        jet_nparticles=np.full(6,3),hlt_jet_nparticles=np.full(6,2))
    counts,outside,particles,maxima,arrays,classes=population.donor.inventory(dict(files=[record]),[member],[roles])
    p=with_content_hash(dict(counts=counts,role_particles=particles,maximum_support=maxima))
    parent=(p,dict(files=[record]),[member],[roles],arrays)
    spec=dict(root=str(tmp_path/'study'),dataset=str(tmp_path),population=p,maximum=3,content_hash='a'*64)
    original=original_reader.Reader;calls=[];native_arrays=original.arrays
    def guarded(self,keys,**kw):
        lo,hi=kw['entry_start'],kw['entry_stop']
        assert roles[lo] in (0,1) and np.all(roles[lo:hi]==roles[lo])
        if coordinate=='D000' and (roles[lo]==1 or not targets):assert all(k.startswith('hlt_') for k in keys)
        calls.append((lo,hi));return native_arrays(self,keys,**kw)
    monkeypatch.setattr(original,'arrays',guarded)
    try:
        old=inputs.prepare_node(spec,tmp_path/'baseline',coordinate,targets=targets,workers=1,loaded=(p,parent,arrays))
        original_reader.close_reader()
        Native=reader.make_reader(original);monkeypatch.setattr(original_reader,'Reader',Native)
        with _owned_reader(Native,path) as value:
            assert isinstance(value.handle.file.source,uproot.source.file.MultithreadedFileSource)
        new=inputs.prepare_node(spec,tmp_path/'native',coordinate,targets=targets,workers=1,loaded=(p,parent,arrays))
        for file in (tmp_path/'baseline').glob('*.npy'):
            assert file.read_bytes()==(tmp_path/'native'/file.name).read_bytes(),file.name
        assert old['normalization']==new['normalization'] and calls
    finally:original_reader.close_reader()


from contextlib import contextmanager
@contextmanager
def _owned_reader(cls,path):
    value=cls(path,'tree')
    try:yield value
    finally:value.close()


def fixture_old(tmp_path,monkeypatch):
    root=tmp_path/'original';project=tmp_path/'old_code';root.mkdir();script=project/'sbatch/run_luka_d20_full_hydra.sh'
    script.parent.mkdir(parents=True);script.write_bytes(b'#!/usr/bin/env bash\nexit 0\n')
    spec=dict(root=str(root),project=str(project),content_hash='a'*64)
    records=[dict(task=t,job=str(300000+i)) for i,t in enumerate(science.tasks())]
    latest={r['task']:r for r in records};latest['M0HLT_s22_p01']['job']='243576';latest['report']['job']='243624'
    states={r['job']:('PENDING','0:0') for r in records}
    for phase in ('prepare','qualify','release'):states[latest[phase]['job']]=('COMPLETED','0:0')
    states['243576']=('FAILED','1:0')
    for part in (2,3,4):states[latest[f'M0HLT_s22_p{part:02d}']['job']]=('CANCELLED','0:0')
    for name in ('OFFLINE_s20_p01','OFFLINE_s21_p01','OFFLINE_s22_p01','M0HLT_s20_p02','M0HLT_s21_p02'):
        states[latest[name]['job']]=('RUNNING','0:0')
    old=SimpleNamespace(spec=spec,contracts=science,campaign=SimpleNamespace(
        accepted_jobs=lambda s:records,commands=campaign.commands,script_index=campaign.script_index),
        worker=SimpleNamespace(task_receipt=lambda s,t:dict(job=latest['qualify']['job']) if t=='qualify' else None),
        resources=SimpleNamespace(qualification=lambda s:dict(content_hash='c'*64)))
    repair=dict(root=str(tmp_path/'repair'),project=str(tmp_path/'new_code'),content_hash='b'*64,
        qualification_hash='c'*64,qualification_job=latest['qualify']['job'],
        requested_tasks={'M0HLT_s22_p01':'243576'})
    Path(repair['root']).mkdir();c.publish(Path(repair['root'])/'repair_spec.json',repair)
    monkeypatch.setattr(controller,'states',lambda jobs:{j:states[j] for j in jobs})
    return repair,old,records,latest,states


def test_only_failed_chain_and_report_are_replaced_while_other_jobs_run(tmp_path,monkeypatch):
    spec,old,records,latest,status=fixture_old(tmp_path,monkeypatch)
    plan=controller.replacements(spec,old)
    assert [r['task'] for r in plan['commands']]==[f'M0HLT_s22_p{i:02d}' for i in range(1,5)]+['report']
    assert [r['task'] for r in plan['cancel']]==['report']
    for name in ('OFFLINE_s20_p01','M0HLT_s20_p02','M0HLT_s21_p02'):
        assert plan['preserved'][name]==latest[name]['job']
    command=plan['commands'][1]['command']
    assert '--dependency=afterok:{M0HLT_s22_p01}' in command
    assert '--gres=gpu:geforce_gtx_1080_ti:4' in command and '--time=2880' in command
    assert '--chdir='+old.spec['project'] in command
    report=plan['commands'][-1]['command'];dep=next(v for v in report if v.startswith('--dependency='))
    assert '{M0HLT_s22_p04}' in dep and latest['OFFLINE_s20_p04']['job'] in dep


@pytest.mark.parametrize('change',['root_running','root_completed','new_attempt','descendant_running'])
def test_active_or_changed_recovery_targets_are_preserved(tmp_path,monkeypatch,change):
    spec,old,records,latest,status=fixture_old(tmp_path,monkeypatch)
    if change=='root_running':status['243576']=('RUNNING','0:0')
    elif change=='root_completed':status['243576']=('COMPLETED','0:0')
    elif change=='new_attempt':latest['M0HLT_s22_p01']['job']='777';status['777']=('FAILED','1:0')
    else:status[latest['M0HLT_s22_p02']['job']]=('RUNNING','0:0')
    with pytest.raises(ValueError):controller.replacements(spec,old)


def test_cancel_uses_exact_bound_pending_id_and_scheduler_state_filter(tmp_path,monkeypatch):
    spec,old,records,latest,status=fixture_old(tmp_path,monkeypatch);calls=[];row=latest['report']
    fields=f"JobId={row['job']} JobState=PENDING JobName=lukadfull_report Comment={old.spec['content_hash']}:report WorkDir={old.spec['project']}"
    monkeypatch.setattr(controller.subprocess,'check_output',lambda *a,**kw:fields)
    def cancel(command,**kw):
        calls.append(command);status[row['job']]=('CANCELLED','0:0')
    monkeypatch.setattr(controller.subprocess,'run',cancel)
    controller.cancel_pending(spec,old,[row])
    assert calls==[['scancel','--state=PENDING','243624']]
    assert (Path(spec['root'])/'cancellations/243624.intent.json').exists()


def test_cancel_refuses_job_that_started_or_has_foreign_binding(tmp_path,monkeypatch):
    spec,old,records,latest,status=fixture_old(tmp_path,monkeypatch);row=latest['report']
    monkeypatch.setattr(controller.subprocess,'run',lambda *a,**kw:pytest.fail('Running/foreign job cancelled'))
    status[row['job']]=('RUNNING','0:0')
    with pytest.raises(ValueError):controller.cancel_pending(spec,old,[row])
    status[row['job']]=('PENDING','0:0')
    monkeypatch.setattr(controller.subprocess,'check_output',lambda *a,**kw:'JobId=243624 JobState=PENDING JobName=foreign')
    with pytest.raises(ValueError):controller.cancel_pending(spec,old,[row])


def test_actual_replacement_journal_is_accepted_by_original_campaign(tmp_path,monkeypatch):
    spec,old,records,latest,status=fixture_old(tmp_path,monkeypatch)
    rows=controller.replacements(spec,old)['commands'];calls=[]
    def submit(command,**kw):
        calls.append(command);return SimpleNamespace(returncode=0,stdout=str(500+len(calls))+'\n',stderr='')
    monkeypatch.setattr(controller.subprocess,'run',submit)
    result=controller.submit(spec,rows,Path(old.spec['root'])/'submission/attempt_001',old=old)
    accepted=campaign.accepted_jobs(old.spec)
    assert len(accepted)==5 and result['M0HLT_s22_p01']=='501'
    assert all(r['execution_recovery']==spec['content_hash'] for r in accepted)
    assert '--dependency=afterok:501' in calls[1]
    assert all('{' not in arg for command in calls for arg in command)


def test_uncertain_submission_stays_blocked_without_duplicate_retry(tmp_path,monkeypatch):
    spec,old,records,latest,status=fixture_old(tmp_path,monkeypatch);calls=[]
    monkeypatch.setattr(controller.subprocess,'run',lambda args,**kw:(calls.append(args) or SimpleNamespace(returncode=1,stdout='',stderr='uncertain')))
    rows=controller.replacements(spec,old)['commands'];folder=Path(old.spec['root'])/'submission/attempt_001'
    with pytest.raises(ValueError,match='Uncertain'):controller.submit(spec,rows,folder,old=old)
    assert len(calls)==1 and (folder/'M0HLT_s22_p01.intent.json').exists()
    with pytest.raises(ValueError,match='Ambiguous'):campaign.accepted_jobs(old.spec)


def test_native_source_probe_does_not_accept_changed_numeric_payloads():
    value=SimpleNamespace(features=np.ones((2,17),np.float32),vectors=np.ones((2,4),np.float32))
    left=([('a'*64,value)],[]);probe.exact(left,deepcopy(left))
    changed=deepcopy(left);changed[0][0][1].features[0,0]=2
    with pytest.raises(ValueError,match='numeric bytes'):probe.exact(left,changed)


def test_release_cannot_cancel_or_submit_without_successful_reader_probe(tmp_path,monkeypatch):
    spec,old,*_=fixture_old(tmp_path,monkeypatch)
    monkeypatch.setattr(runtime,'check',lambda *a:old)
    monkeypatch.setattr(controller,'bound',lambda *a:'111')
    monkeypatch.setattr(runtime,'parity',lambda *a:(_ for _ in ()).throw(FileNotFoundError('probe absent')))
    monkeypatch.setattr(controller,'cancel_pending',lambda *a:pytest.fail('Cancelled before parity'))
    monkeypatch.setattr(controller,'submit',lambda *a,**kw:pytest.fail('Submitted before parity'))
    with pytest.raises(FileNotFoundError,match='probe absent'):controller.release(spec)


def test_release_preserves_active_science_and_orders_probe_review_cancel_submit(tmp_path,monkeypatch):
    spec,old,records,latest,status=fixture_old(tmp_path,monkeypatch);events=[]
    old.campaign.submission_lock=__import__('hlt_classification.luka_sixclass_tail_hydra.campaign',fromlist=['submission_lock']).submission_lock
    monkeypatch.setattr(runtime,'check',lambda *a:old)
    monkeypatch.setattr(controller,'bound',lambda *a:'111')
    status['222']=('COMPLETED','0:0')
    monkeypatch.setattr(runtime,'parity',lambda *a:(events.append('parity') or dict(job='222')))
    monkeypatch.setattr(controller,'dry_review',lambda rows:events.append('dry review'))
    def cancel(spec,old,rows):
        assert [r['job'] for r in rows]==['243624'];events.append('cancel pending report')
    monkeypatch.setattr(controller,'cancel_pending',cancel)
    def submit(command,**kw):
        events.append('submit');return SimpleNamespace(returncode=0,stdout=str(800+len(events))+'\n',stderr='')
    monkeypatch.setattr(controller.subprocess,'run',submit)
    jobs=controller.release(spec)
    assert events[:3]==['parity','dry review','cancel pending report']
    assert len(jobs)==5 and all(status[latest[name]['job']][0]=='RUNNING' for name in ('OFFLINE_s20_p01','M0HLT_s20_p02'))
    assert c.read(Path(spec['root'])/'released.json')['preserved']['M0HLT_s20_p02']==latest['M0HLT_s20_p02']['job']
    with pytest.raises(ValueError,match='Release already'):controller.release(spec)


def test_real_torchrun_parser_accepts_recovery_script_and_arguments(tmp_path,monkeypatch):
    from hlt_classification.luka_d20_full_hydra import launch
    from hlt_classification.luka_d20_full_hydra import basket_reader,worker,resources,training,artifacts,kernel
    root=Path(__file__).resolve().parents[1]
    spec=dict(root=str(tmp_path),project=str(root),content_hash='b'*64)
    old=SimpleNamespace(spec=dict(project=str(root),root=str(tmp_path/'parent')),
        inputs=inputs,basket_reader=basket_reader,launch=launch)
    for obj,name in [(inputs,'decoded'),(basket_reader,'Reader'),(launch,'command')]:
        monkeypatch.setattr(obj,name,getattr(obj,name))
    monkeypatch.setattr(runtime,'_INSTALLED',None)
    monkeypatch.setattr(runtime,'check',lambda *a,**kw:old)
    runtime.install(spec)
    argv=launch.command(str(root),old.spec['root'],'M0HLT_s22_p01','/dev/shm/rlyang_d20full_123')
    parsed=launch.validate(argv)
    assert Path(parsed['training_script']).name=='recover_luka_d20_inputs.py'
    assert parsed['training_script_args'][:3]==['rank','--repair-spec',str(tmp_path/'repair_spec.json')]
    assert '--time=00:15:00' in launch.startup_command(str(root),old.spec['root'],'M0HLT_s22_p01','/dev/shm/rlyang_d20full_123')


def test_clean_repair_deployment_and_stale_source_detection(tmp_path):
    import sys
    project=tmp_path/'deploy';project.mkdir();root=Path(__file__).resolve().parents[1]
    for name in runtime.required_files():
        target=project/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((root/name).read_bytes().replace(b'\r\n',b'\n'))
    def git(*args):return subprocess.check_output(['git','-C',str(project),*args],text=True,stderr=subprocess.STDOUT).strip()
    git('init');git('add','--','.');git('-c','commit.gpgsign=false','-c','user.name=InputRecoveryFixture',
        '-c','user.email=fixture@example.invalid','commit','--no-verify','-m','Isolated repair fixture')
    commit=git('rev-parse','HEAD');git('update-ref','refs/remotes/origin/main',commit)
    code='''import sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])/'src'))
from hlt_classification.luka_d20_input_recovery.runtime import source
assert source(sys.argv[1])['commit']==sys.argv[2]
p=Path(sys.argv[1])/'src/hlt_classification/luka_d20_input_recovery/reader.py'
p.write_text(p.read_text()+'\\n# drift\\n')
try:source(sys.argv[1])
except ValueError:print('PASS clean deployment and stale source rejection')
else:raise AssertionError('Stale source accepted')
'''
    result=subprocess.run([sys.executable,'-I','-B','-c',code,str(project),commit],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'PASS clean deployment' in result.stdout


def test_original_successful_qualification_is_required_before_reuse(tmp_path,monkeypatch):
    spec,old,records,latest,status=fixture_old(tmp_path,monkeypatch)
    controller.qualified(spec,old)
    status[spec['qualification_job']]=('FAILED','1:0')
    with pytest.raises(ValueError,match='did not finish'):controller.qualified(spec,old)
    status[spec['qualification_job']]=('COMPLETED','0:0');spec['qualification_hash']='d'*64
    with pytest.raises(ValueError,match='qualification or accepted'):controller.qualified(spec,old)


def test_bootstrap_journals_are_bound_to_exact_cpu_commands(tmp_path,monkeypatch):
    spec,old,*_=fixture_old(tmp_path,monkeypatch);calls=[]
    def submitted(command,**kw):
        calls.append(command);return SimpleNamespace(returncode=0,stdout=str(100+len(calls))+'\n',stderr='')
    monkeypatch.setattr(controller.subprocess,'run',submitted)
    rows=controller.bootstrap_commands(spec)
    assert all('--partition=compute' in row['command'] and not any('--gres=' in v for v in row['command']) for row in rows)
    controller.submit(spec,rows,Path(spec['root'])/'submission/bootstrap')
    assert c.bootstrap_receipt(spec,'probe')['job']=='101'
    assert c.bootstrap_receipt(spec,'release')['job']=='102'
    assert '--dependency=afterok:101' in calls[1]
    path=Path(spec['root'])/'submission/bootstrap/probe.submitted.json';value=c.read(path)
    value['command'][2]='--partition=gpus'
    path.unlink();c.publish(path,c.artifact('SUBMITTED',**{k:v for k,v in value.items()
        if k not in ('content_hash','contract','schema_version',*c.FLAGS)}))
    with pytest.raises(ValueError,match='provenance'):c.bootstrap_receipt(spec,'probe')


@pytest.mark.parametrize('damage',[None,'missing_view','test_role','missing_bad_file','wrong_job'])
def test_reader_probe_requires_complete_authorized_matrix_and_job_binding(tmp_path,monkeypatch,damage):
    spec,old,*_=fixture_old(tmp_path,monkeypatch)
    records=[dict(path=path,sha256='a'*64,role=role,start=role,stop=role+1,
        coordinate=coordinate,targets=role==0 and coordinate=='D000',byte_equal=True)
        for path in (c.BAD_FILE,'raw/signal/a.root') for role in (0,1)
        for coordinate in ('D000','OFFLINE','U100')]
    if damage=='missing_view':records.pop()
    elif damage=='test_role':records[0]['role']=2
    elif damage=='missing_bad_file':
        for row in records:
            if row['path']==c.BAD_FILE:row['path']='raw/qcd/another.root'
    proof=c.artifact('READER_PARITY',parents=dict(repair=spec['content_hash'],qualification=spec['qualification_hash']),
        job='404' if damage=='wrong_job' else '303',records=records,cases=len(records),passed=True,models_updated=False)
    c.publish(Path(spec['root'])/'reader_parity.json',proof)
    monkeypatch.setattr(c,'bootstrap_receipt',lambda *a:dict(job='303'))
    if damage is None:assert runtime.parity(spec,old)==proof
    else:
        with pytest.raises(ValueError):runtime.parity(spec,old)

"""Durable R bootstrap: real journal orchestration with a simulated Slurm boundary."""
from types import SimpleNamespace
import pytest
from hlt_classification.data.cache_contracts import write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.scouting import hcwdl_exact_dag_submission as dag
from hlt_classification.luka_next_supervision_hydra import campaign,contracts,submission,worker,reconstruction
from hlt_classification.luka_next_supervision_hydra import reconstruction_contracts as rc,reconstruction_release as release


def spec_for(root):
    return dict(reconstruction.registration(),root=str(root),project_dir='/code',content_hash='a'*64,
        remaining=list(rc.TASKS),reused={'C0':{},'C1':{}},reference=dict(environment={}))


@pytest.fixture
def scheduler(tmp_path,monkeypatch):
    spec=spec_for(tmp_path); commands=[]; reviews=[]
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    monkeypatch.setattr(release,'scheduler_review',lambda p:reviews.append(p['stage']))
    monkeypatch.setattr(submission,'check_site',lambda p:reviews.append(p['stage']))
    monkeypatch.setattr(release.time,'sleep',lambda *a:None)
    def sbatch(cmd,**kw):
        commands.append(cmd)
        return SimpleNamespace(stdout=str(8000+len(commands)),stderr='',returncode=0)
    monkeypatch.setattr(dag.subprocess,'run',sbatch)
    submission.submit(spec,'bootstrap')
    plan=submission.plan(spec,'bootstrap')
    ledger=submission.submit(spec,'bootstrap',execute=True,plan_hash=plan['content_hash'],authorization=rc.AUTHORIZATION)
    accepted=dict(content_hash='b'*64,minutes={t:120+i for i,t in enumerate(rc.TASKS)},
        reconstruction_review=dict(fixed_output_slots=257))
    write_immutable_json(tmp_path/'setup/acceptance.json',accepted)
    monkeypatch.setattr(worker,'completed_setup',lambda *a:accepted)
    monkeypatch.setattr(submission,'accounting',lambda *a:['COMPLETED','0:0'])
    return spec,commands,reviews,ledger,accepted


def test_one_authorization_queues_setup_cpu_release_then_twelve_measured_independent_fits(scheduler):
    spec,commands,reviews,bootstrap,accepted=scheduler
    assert bootstrap['jobs']==dict(setup='8001',release='8002') and len(commands)==2
    assert '--dependency=afterok:8001' in commands[1] and '--kill-on-invalid-dep=yes' in commands[1]
    assert '--partition=compute' in commands[1] and not any(x.startswith('--gres=') for x in commands[1])
    # Initial review commits the recipe and complete finite resource envelope.
    preview=submission.plan(spec,'bootstrap')['automatic_science']
    assert preview['tasks']==list(rc.TASKS) and preview['fit_max_minutes']==2880
    assert preview['recipe']['content_hash']=='90b582b9f37955f13217ec36f3f724c83ccecd7c88c9d9983ce8bce6bda345cc'
    result=release.finish(spec,'8002')
    assert len(commands)==14 and set(result['jobs'])==set(rc.TASKS)
    assert reviews==['bootstrap','science']
    assert [cmd[-1] for cmd in commands[2:]]==list(rc.TASKS)
    assert all('--time=02:'+str(i).zfill(2)+':00' in cmd for i,cmd in enumerate(commands[2:]))
    assert all(not any(x.startswith('--dependency=') for x in cmd) for cmd in commands[2:])
    assert release.finish(spec,'8002')==result and len(commands)==14
    assert submission.submit(spec,'bootstrap',execute=True,
        plan_hash=submission.plan(spec,'bootstrap')['content_hash'],authorization=rc.AUTHORIZATION)==bootstrap
    assert len(commands)==14


def test_accounting_delay_is_bounded_and_requires_the_exact_setup(scheduler,monkeypatch):
    spec,commands,_,_,_=scheduler; observed=[]; states=iter([None,['RUNNING','0:0'],['COMPLETED','0:0']])
    def accounting(job):
        observed.append(job); return next(states,['COMPLETED','0:0'])
    monkeypatch.setattr(submission,'accounting',accounting)
    release.finish(spec,'8002')
    assert set(observed)=={'8001'} and len(observed)>=3 and len(commands)==14


@pytest.mark.parametrize('record',[['FAILED','1:0'],['CANCELLED','0:0'],['COMPLETED','1:0'],None])
def test_failed_or_unconfirmed_setup_submits_nothing(scheduler,monkeypatch,record):
    spec,commands,_,_,_=scheduler
    monkeypatch.setattr(submission,'accounting',lambda *a:record)
    with pytest.raises(RuntimeError,match='no science submitted'): release.finish(spec,'8002')
    assert len(commands)==2


def test_invalid_setup_receipts_and_stale_source_submits_nothing(scheduler,monkeypatch):
    spec,commands,_,_,_=scheduler
    def broken(*a,**kw): raise ValueError('Receipt or source mismatch')
    monkeypatch.setattr(worker,'completed_setup',broken)
    with pytest.raises(ValueError,match='mismatch'): release.finish(spec,'8002')
    monkeypatch.setattr(campaign,'check',broken)
    with pytest.raises(ValueError,match='mismatch'): release.finish(spec,'8002')
    assert len(commands)==2


def test_wrong_release_job_or_modified_authorization_cannot_submit(scheduler,tmp_path):
    spec,commands,_,_,_=scheduler
    with pytest.raises(PermissionError,match='exact accepted'): release.finish(spec,'9999')
    claim=tmp_path/'submission_bootstrap/live_claim/intent.json'
    claim.write_text('{}')
    with pytest.raises(PermissionError,match='authorization'): release.finish(spec,'8002')
    assert len(commands)==2


def test_lost_sbatch_response_cannot_create_duplicates_on_release_retry(scheduler,monkeypatch,tmp_path):
    spec,commands,_,_,_=scheduler
    def lost(cmd,**kw):
        commands.append(cmd); raise OSError('Lost response after request reached scheduler')
    monkeypatch.setattr(dag.subprocess,'run',lost)
    with pytest.raises(OSError,match='Lost response'): release.finish(spec,'8002')
    assert len(commands)==3 and (tmp_path/'submission_science/live_claim').is_dir()
    with pytest.raises(RuntimeError,match='partial'): release.finish(spec,'8002')
    assert len(commands)==3


def test_setup_receipt_uses_bootstrap_identity_and_authenticates_payload(scheduler,tmp_path,monkeypatch):
    spec,_,_,_,_=scheduler; path=tmp_path/'setup/acceptance.json'
    write_immutable_json(tmp_path/'setup/receipt.json',contracts.artifact('TASK',
        parents=dict(spec=spec['content_hash']),task='setup',job_id='8001',outputs=[file_ref(path,root=path.parent)]))
    assert worker.receipt(spec,'setup')['job_id']=='8001'
    path.write_text('{}')
    with pytest.raises(ValueError): worker.receipt(spec,'setup')


def test_existing_manual_campaigns_never_gain_automatic_authorization(tmp_path):
    for scope,version in [('group_A_only',2),('group_B_only','B1'),('group_C_only','C1'),('direct_reconstruction_only','R1')]:
        spec=spec_for(tmp_path); spec.update(scope=scope,execution_version=version)
        assert submission.setup_stage(spec)=='setup'
        assert len(submission.plan(spec,'setup')['commands'])==1
        with pytest.raises(ValueError,match='fresh R2'): submission.plan(spec,'bootstrap')
    with pytest.raises(ValueError,match='bootstrap'): submission.plan(spec_for(tmp_path),'setup')


def test_release_worker_never_calls_gpu_backend_or_inference(tmp_path,monkeypatch):
    spec=spec_for(tmp_path); calls=[]
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    monkeypatch.setattr(release,'installed_environment',lambda:{})
    monkeypatch.setattr(release,'cpu_allocation',lambda s:'8002')
    monkeypatch.setattr(release,'finish',lambda s,j:calls.append(j) or {'jobs':{}})
    monkeypatch.setattr(worker,'require_backend',lambda:pytest.fail('CPU release cannot initialize GPU'))
    worker.run(spec,'release'); assert calls==['8002']
    monkeypatch.setattr(release,'installed_environment',lambda:{'wrong':True})
    with pytest.raises(PermissionError,match='software'): worker.run(spec,'release')
    assert calls==['8002']


def test_bootstrap_dry_review_checks_both_shapes_without_submitting(tmp_path,monkeypatch):
    spec=spec_for(tmp_path); calls=[]
    def run(cmd,**kw):
        calls.append(cmd); return SimpleNamespace(stdout='ClusterName = hydra\n',stderr='',returncode=0)
    monkeypatch.setattr(release.subprocess,'run',run)
    release.scheduler_review(submission.plan(spec,'bootstrap'))
    assert len(calls)==3
    assert all('--test-only' in c and '--wrap=true' in c for c in calls[1:])
    assert all(not any(x.startswith('--dependency=') for x in c) for c in calls[1:])
    assert '--gres=gpu:nvidia_rtx_a6000:1' in calls[1]
    assert not any(x.startswith('--gres=') for x in calls[2])


@pytest.fixture
def cpu_controller(tmp_path,monkeypatch):
    spec=spec_for(tmp_path)
    monkeypatch.setenv('SLURM_JOB_ID','8002'); monkeypatch.setenv('SLURM_CLUSTER_NAME','(null)')
    for key,value in dict(SLURM_NTASKS='1',SLURM_CPUS_PER_TASK='1',SLURM_MEM_PER_NODE='4000').items():
        monkeypatch.setenv(key,value)
    fields=dict(JobId='8002',JobState='RUNNING',Partition='compute',Account='default',QOS='normal',
        NumCPUs='2',NumNodes='1',NumTasks='(null)',**{'CPUs/Task':'1'},
        ReqTRES='cpu=1,mem=4000M,node=1',AllocTRES='cpu=2,mem=4000M,node=1')
    def run(cmd,**kw):
        return SimpleNamespace(stdout='ClusterName = hydra\n' if cmd[-1]=='config' else ' '.join(k+'='+v for k,v in fields.items()))
    monkeypatch.setattr(release.subprocess,'run',run)
    return spec,fields


def test_observed_hydra_cpu_rounding_and_null_task_field_are_accepted(cpu_controller):
    spec,fields=cpu_controller; assert release.cpu_allocation(spec)=='8002'
    fields.update(NumCPUs='1',AllocTRES='cpu=1,mem=4000M,node=1')
    assert release.cpu_allocation(spec)=='8002'


@pytest.mark.parametrize('key,value',[
    ('JobId','9999'),('JobState','PENDING'),('Partition','gpus'),('Account','other'),
    ('QOS','other'),('NumNodes','2'),('NumTasks','2'),('CPUs/Task','2'),
    ('ReqTRES','cpu=2,mem=4000M,node=1'),('AllocTRES','cpu=2,mem=4000M,node=1,gres/gpu=1'),
    ('AllocTRES','cpu=3,mem=4000M,node=1'),('AllocTRES','cpu=2,mem=8000M,node=1')])
def test_other_cpu_allocation_mismatches_still_reject(cpu_controller,key,value):
    spec,fields=cpu_controller; fields[key]=value
    with pytest.raises(PermissionError): release.cpu_allocation(spec)

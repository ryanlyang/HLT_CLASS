"""Reconstruct saved CPU-fixture probes; never manufacture remote acceptance."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess

import pytest

from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.cms2jc2_response.contracts import artifact as response_artifact
from hlt_classification.luka_fullsim import alljets_report_recovery as r, alljets_report_replay as replay
from hlt_classification.luka_fullsim import alljets_fusion_campaign as c, alljets_fusion_worker as w
from hlt_classification.luka_fullsim import alljets_fusion_submission as sub, alljets_fusion_isolated_preflight as iso
from test_luka_alljets_batch128 import gate128
from test_luka_fusion_isolated_preflight import local_gate, Measurement
from test_luka_fullsim_stage2 import dataset, prepared, config
from test_luka_fullsim_alljets import all_prepared

ACCOUNT = (
    '21898881|lukab_preflight|debug|reu-aisocial|FAILED|1:0|6|1||250Gn|\n'
    '21898881.batch|batch|debug|reu-aisocial|FAILED|1:0|6|1|1||74695592K\n'
    '21898881.extern|extern|debug|reu-aisocial|COMPLETED|0:0|6|1|1||\n')
TAIL = '''Traceback (most recent call last):
  File "alljets_fusion_isolated_preflight.py", line 285, in run
    with Measurement() as measured:
  File "measurement.py", line 89, in __exit__
    raise self.error
  File "measurement.py", line 72, in _loop
    self.sample()
ProcessLookupError: [Errno 3] No such process
'''


@pytest.fixture
def finished(gate128,monkeypatch):
    gate,parent,p,events,launched=gate128
    gate=with_content_hash(dict(gate,source_commit=r.FAILED_COMMIT,
        source_snapshot=dict(git_commit=r.FAILED_COMMIT,content_hash='e'*64)))
    root=Path(gate['root'])
    (root/'gate_spec.json').write_text(json.dumps(gate))
    monkeypatch.setattr(w,'execution',lambda *a:r.FAILED_JOB)
    class FailedCoordinator(Measurement):
        instances=0
        def __init__(self):
            self.outer=FailedCoordinator.instances==0
            FailedCoordinator.instances+=1
        def __exit__(self,*a):
            if self.outer: raise ProcessLookupError(3,'No such process')
    monkeypatch.setattr(iso,'Measurement',FailedCoordinator)
    with pytest.raises(ProcessLookupError): iso.run(gate)
    text=''
    for kind,name in iso.sequence(gate):
        text+=f'LUKA isolated {kind}: {name}; fresh process, no automatic retry\n'
        text+=json.dumps(load_json(root/'preflight/children'/(kind+'_'+name)/'result.json'),indent=2)+'\n'
    (root/f'preflight-{r.FAILED_JOB}.out').write_text(text+TAIL)
    monkeypatch.setattr(c,'__file__',str(Path(gate['project_dir'])/'src/hlt_classification/luka_fullsim/alljets_fusion_campaign.py'))
    monkeypatch.setattr(sub,'accepted_jobs',lambda _:dict(preflight=r.FAILED_JOB))
    return gate,parent,p


def test_sampler_failure_replay_keeps_original_and_all_checks(finished):
    gate,_,_=finished
    root=Path(gate['root'])
    before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    data=replay.replay(gate,r.parse_accounting(ACCOUNT))
    assert data['admission_error'] is None and data['original_job_status']=='FAILED'
    assert len(data['candidate']['isolated_children'])==12
    assert data['candidate']['job_id']==r.FAILED_JOB
    assert data['memory_evidence']['conservative_sum_bytes']==74695592*1024+2**30
    assert data['candidate']['peak_rss_bytes']==data['memory_evidence']['conservative_sum_bytes']
    assert before=={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert not (root/'preflight/preflight.json').exists()


def test_replay_rejects_missing_corrupt_or_unlogged_evidence(finished):
    gate,_,_=finished
    root=Path(gate['root']); account=r.parse_accounting(ACCOUNT)
    target=root/'preflight/children/stress_FUSION_D000_D000_probe/result.json'
    original=target.read_bytes()
    target.write_bytes(original+b'broken')
    with pytest.raises(ValueError): replay.replay(gate,account)
    target.write_bytes(original)
    log=root/f'preflight-{r.FAILED_JOB}.out'
    original_log=log.read_text()
    for text in (original_log.replace('ProcessLookupError','RuntimeError'),
                 original_log.replace('LUKA isolated stress:','LUKA isolated other:'),
                 original_log.replace('"completed": true','"completed": false',1)):
        log.write_text(text)
        with pytest.raises(ValueError): replay.replay(gate,account)
    log.write_text(original_log)
    failure=root/'preflight/children/stress_U000_probe/failure.json'
    failure.write_text('{}')
    with pytest.raises(ValueError,match='probe failed'): replay.replay(gate,account)


def test_replay_retains_resource_and_schedule_admission(finished,monkeypatch):
    gate,_,_=finished
    account=r.parse_accounting(ACCOUNT)
    too_big=dict(account,batch_maxrss_bytes=250*2**30)
    data=replay.replay(gate,too_big)
    assert not data['candidate']['resource_envelope_ok'] and data['admission_error']
    monkeypatch.setattr(replay.pre,'projections',lambda _: (1500,30))
    data=replay.replay(gate,account)
    assert not data['candidate']['fits_debug_24h'] and data['admission_error']


@pytest.mark.parametrize('before,after',[
    ('FAILED','RUNNING'),('1:0','0:0'),('74695592K',''),('74695592K','0K'),('250Gn','90Gn'),
    ('lukab_preflight','another_job'),('|6|1|1|','|6|1|2|'),('debug','tier3'),('reu-aisocial','wrong')])
def test_accounting_fails_closed(before,after):
    with pytest.raises(ValueError): r.parse_accounting(ACCOUNT.replace(before,after))


def test_memory_units_duplicates_and_cpu_render(tmp_path):
    assert r.size_bytes('1.5G')==int(1.5*2**30)
    assert r.size_bytes('256000Mn')==250*2**30
    for text in ('0K','nanG','1024','1Gcgarbage'):
        with pytest.raises(ValueError): r.size_bytes(text)
    with pytest.raises(ValueError): r.parse_accounting(ACCOUNT+ACCOUNT.splitlines()[0]+'\n')
    gate=c.artifact('GATE',version=5,root=str(tmp_path),project_dir='project',**c.registration(5))
    plan=sub.render(gate)
    command=plan['commands'][0]['command']
    assert plan['jobs']==1 and plan['authorization']==r.AUTHORIZATION
    assert '--mem=32768M' in command and '--cpus-per-task=1' in command and '--time=04:00:00' in command
    assert not any(x.startswith('--gres=') for x in command)
    for key in ('nodes','tasks','training_recipe','counts','pair_storage'):
        assert c.registration(5)[key]==c.registration(4)[key]
    spec=c.artifact('SPEC',version=5,root='science',project_dir='project',train_minutes=1000,reduce_minutes=30,**c.registration(5))
    plan=sub.render(spec)
    assert plan['jobs']==30 and c.completion(spec)['fits']==17
    tasks={row['task_id']:row for row in plan['commands']}
    assert '--dependency=afterok:${JOB_reduce_U000}' in tasks['train_FUSION_U050']['command']
    with pytest.raises(ValueError): c.artifact('PROBE',version=5)


def test_create_run_admit_and_submit_report_only(finished,tmp_path,monkeypatch):
    old,parent,p=finished
    real_validate=c.validate_gate
    account=r.parse_accounting(ACCOUNT)
    active=dict(content_hash='a'*64,git_commit='a'*40)
    monkeypatch.setattr(r,'source',lambda *a:active)
    monkeypatch.setattr(r,'compatible',lambda *a:dict(fixture=True))
    monkeypatch.setattr(r,'accounting',lambda:account)
    monkeypatch.setattr(r.batch128,'location',lambda *a:parent)
    def local_replay(path,account,project):
        return replay.replay(load_json(path),account)
    monkeypatch.setattr(r,'replay',local_replay)
    gate=r.create(failed_gate=Path(old['root'])/'gate_spec.json',root=tmp_path/'recovered',project=tmp_path/'newsource',commit='a'*40)
    assert gate['schema_version']==5
    monkeypatch.setattr(c,'validate_gate',lambda g:r.validate_gate(g) if g['schema_version']==5 else real_validate(g))
    monkeypatch.setattr(sub,'check_site',lambda _:None)
    from hlt_classification.scouting import hcwdl_exact_dag_submission as exact
    calls=[]
    def submit(argv,**kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv,0,str(200000+len(calls)),'')
    monkeypatch.setattr(exact.subprocess,'run',submit)
    gate_plan=sub.submit(gate)
    assert not calls
    with pytest.raises(PermissionError):
        sub.submit(gate,execute=True,plan_hash='0'*64,authorization=r.AUTHORIZATION)
    sub.submit(gate,execute=True,plan_hash=gate_plan['content_hash'],authorization=r.AUTHORIZATION)
    sub.submit(gate,execute=True,plan_hash=gate_plan['content_hash'],authorization=r.AUTHORIZATION)
    assert len(calls)==1 and not any(x.startswith('--gres=') for x in calls[0])
    calls.clear()
    from hlt_classification.cms2jc2_response import measurement
    allocation=response_artifact('ALLOCATION',job_id='99999',cpus=1,gpus=0,partition='debug',account='reu-aisocial')
    monkeypatch.setattr(measurement,'allocation',lambda **kw:allocation)
    monkeypatch.setenv('SLURM_MEM_PER_NODE','32768')
    value=r.run(gate)
    assert value['job_id']=='99999' and value['original_gpu_job_id']==r.FAILED_JOB
    assert value['original_gpu_accounting']['state']=='FAILED'
    assert c.evidence(gate)==(parent,p,value)
    monkeypatch.setattr(sub,'accepted_jobs',lambda g:{'preflight':'99999' if g['schema_version']==5 else r.FAILED_JOB})
    monkeypatch.setattr(c,'preflight_accounting',lambda job:dict(job_id=job,state='COMPLETED',exit_code='0:0'))
    spec=c.create_science(gate)
    assert c.validate_spec(spec)==(parent,p,value)
    assert spec['jobs']==30 and spec['preflight_accounting']['job_id']=='99999'
    assert not (Path(old['root'])/'preflight/preflight.json').exists()
    with pytest.raises(FileExistsError): r.run(gate)
    plan=sub.submit(spec)
    assert not calls
    with pytest.raises(PermissionError): sub.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.authorization({'schema_version':4}))
    sub.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=r.AUTHORIZATION)
    assert len(calls)==30


def test_recovery_never_publishes_admission_on_failed_projection(finished,tmp_path,monkeypatch):
    old,_,_=finished
    monkeypatch.setattr(replay.pre,'projections',lambda _: (1500,30))
    account=r.parse_accounting(ACCOUNT)
    data=replay.replay(old,account)
    gate=c.artifact('GATE',version=5,root=str(tmp_path/'report_only'),source_snapshot={'fixture':True},
        prepared_sha256=old['prepared_sha256'],report_recovery=dict(accounting=account),**c.registration(5))
    Path(gate['root']).mkdir()
    from hlt_classification.cms2jc2_response import measurement
    allocation=response_artifact('ALLOCATION',job_id='99999',cpus=1,gpus=0,partition='debug',account='reu-aisocial')
    monkeypatch.setattr(measurement,'allocation',lambda **kw:allocation)
    monkeypatch.setenv('SLURM_MEM_PER_NODE','32768')
    monkeypatch.setattr(r,'verified_replay',lambda _:data)
    monkeypatch.setattr(r,'accounting',lambda:account)
    with pytest.raises(ValueError,match='science remains blocked'): r.run(gate)
    output=Path(gate['root'])/'preflight'
    value=load_json(output/'measurement.json')
    assert not value['science_admissible'] and value['projected_train_minutes']==1500
    assert not (output/'preflight.json').exists()
    with pytest.raises(FileNotFoundError): r.evidence(gate)


def test_compatibility_preserves_all_model_data_and_recipe_bytes(tmp_path,monkeypatch):
    before,after=tmp_path/'before',tmp_path/'after'
    protected='src/hlt_classification/jetclass2_delphes/runner.py'
    allowed='src/hlt_classification/cms2jc2_response/measurement.py'
    for root in (before,after):
        for name in (protected,allowed):
            path=root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('original')
    for name in r.ADDITIONS:
        path=after/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('new')
    monkeypatch.setattr(r,'validate_source_snapshot',lambda *a,**kw:None)
    monkeypatch.setattr(r,'_files',lambda root:{p.relative_to(root).as_posix() for p in root.rglob('*.py')})
    old=dict(project_dir=str(before),source_snapshot={})
    (after/allowed).write_text('sampler repair')
    record=r.compatible(old,{},after)
    assert record['modules'][protected]['original']==record['modules'][protected]['execution']
    (after/protected).write_text('changed model')
    with pytest.raises(ValueError,match='model/data/recipe'): r.compatible(old,{},after)
    (after/protected).write_text('original')
    (after/'extra.py').write_text('unexpected')
    with pytest.raises(ValueError,match='additions/deletions'): r.compatible(old,{},after)


def test_recovery_required_files_and_helper():
    root=Path(__file__).resolve().parents[1]
    assert all((root/name).is_file() for name in r.REQUIRED)
    helper=(root/'scripts/queue_luka_batch128_report_recovery.sh').read_text()
    assert r.AUTHORIZATION in helper and 'PYTHONNOUSERSITE=1' in helper
    assert 'LD_LIBRARY_PATH="${CONDA_PREFIX}/lib' in helper


def test_replay_subprocess_uses_original_package_not_current(finished,monkeypatch,tmp_path):
    gate,_,_=finished
    wanted={'gate_sha256':gate['content_hash']}
    def launch(argv,**kw):
        assert kw['env']['PYTHONPATH']==str(Path(gate['project_dir'])/'src')
        assert kw['env']['CUDA_VISIBLE_DEVICES']==''
        assert kw['cwd']==gate['project_dir']
        assert 'alljets_report_replay.py' in argv[-2]
        assert json.loads(kw['input'])==r.parse_accounting(ACCOUNT)
        return subprocess.CompletedProcess(argv,0,'LUKA_REPORT_REPLAY='+json.dumps(wanted),'')
    monkeypatch.setattr(r.subprocess,'run',launch)
    assert r.replay(Path(gate['root'])/'gate_spec.json',r.parse_accounting(ACCOUNT),tmp_path)==wanted

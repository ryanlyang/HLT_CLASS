"""Completed fits survive storage guards; only qualified, unstarted SOFT fits run."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import math
import importlib
import numpy as np
import pytest
import torch
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_stability_completion_hydra import core, audit, submission, runtime, reporting
from hlt_classification.luka_reconstruction_stability_hydra import contracts as c, resources, recovery
from hlt_classification.luka_stability_storage_recovery import core as storage_core


def original_fixture(root):
    root.mkdir()
    p = dict(root=str(root), content_hash='a'*64, tasks=list(c.FITS), environment={},
             environment_dir='env', recovery_anchors=dict(M0HLT=.96056122, OFFLINE=.96452569))
    modules = dict(contracts=c, submission=SimpleNamespace(accepted_jobs=lambda *a:dict(core.SALVAGE)),
        worker=SimpleNamespace(completed_setup=lambda *a:None),
        training=SimpleNamespace(validate_fit=lambda *a,technical=False:({'content_hash':('c' if technical else 'd')*64}, {})))
    for task, job in core.SALVAGE.items():
        (root/f'{task}-{job}.out').write_text('OSError: Insufficient remaining stability output allowance plus one GiB reserve\n')
        for name in ('fit/receipt.json','startup/receipt.json'):
            path=root/task/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
        write_immutable_json(root/task/'startup_memory.json',c.artifact('STARTUP',parents=dict(spec=p['content_hash']),task=task,miniature='c'*64,discarded=True))
        write_immutable_json(root/'claims'/task/'intent.json',c.artifact('RUN_CLAIM',parents=dict(spec=p['content_hash']),task=task,job_id=job))
    return p,lambda part:modules[part],modules


def test_salvage_validates_four_finished_fits_without_writing_old_receipts(tmp_path):
    p,r,_=original_fixture(tmp_path/'old')
    result=audit.salvage(p,r,lambda jobs:{t:['FAILED','1:0'] for t in jobs})
    assert set(result)==set(core.SALVAGE)
    assert not any((Path(p['root'])/t/'receipt.json').exists() for t in result)
    assert all(v['report']=='d'*64 and len(v['references'])==5 for v in result.values())


@pytest.mark.parametrize('damage',['live','wrong_job','additional_error','corrupt_fit','startup','claim','task_receipt'])
def test_salvage_rejects_unverified_or_live_work(tmp_path,damage):
    p,r,m=original_fixture(tmp_path/'old');task=next(iter(core.SALVAGE));root=Path(p['root'])
    states={t:['FAILED','1:0'] for t in core.SALVAGE}
    if damage=='live':states[task]=['RUNNING','0:0']
    if damage=='wrong_job':m['submission'].accepted_jobs=lambda *a:{**core.SALVAGE,task:'999'}
    if damage=='additional_error':(root/f'{task}-{core.SALVAGE[task]}.out').write_text('ValueError: bad data\n')
    if damage=='corrupt_fit':m['training'].validate_fit=lambda *a,**kw:(_ for _ in ()).throw(ValueError('corrupt'))
    if damage=='startup':(root/task/'startup_memory.json').write_text('{}')
    if damage=='claim':(root/'claims'/task/'intent.json').write_text('{}')
    if damage=='task_receipt':(root/task/'receipt.json').write_text('{}')
    with pytest.raises(ValueError):audit.salvage(p,r,lambda _:states)


def failed_soft_fixture(root):
    root.mkdir();p=dict(root=str(root),content_hash='b'*64)
    (root/f'setup_pascal-{core.SETUP_JOB}.out').write_text('[rank0]: '+audit.CAP_ERROR+'\nsubprocess.CalledProcessError: wrapper\n')
    write_immutable_json(root/'claims/setup_pascal/intent.json',c.artifact('RUN_CLAIM',parents=dict(spec=p['content_hash']),task='setup_pascal',job_id=core.SETUP_JOB))
    states={core.SETUP_JOB:['FAILED','1:0'],core.RELEASE_JOB:['CANCELLED','0:0']}
    module=SimpleNamespace(accepted_jobs=lambda *a:dict(setup_pascal=core.SETUP_JOB,release_pascal=core.RELEASE_JOB),accounting=lambda job:states[job])
    return p,lambda part:c if part=='contracts' else module,states


@pytest.mark.parametrize('damage',[None,'live_release','live_setup','science_directory','science_claim','setup_receipt','other_error'])
def test_old_soft_failure_must_be_only_time_cap_and_no_science(tmp_path,damage):
    p,fr,states=failed_soft_fixture(tmp_path/'soft');root=Path(p['root'])
    if damage=='live_release':states[core.RELEASE_JOB]=['RUNNING','0:0']
    if damage=='live_setup':states[core.SETUP_JOB]=['RUNNING','0:0']
    if damage=='science_directory':(root/'submission_science_pascal').mkdir()
    if damage=='science_claim':(root/'claims/SOFT_s7').mkdir()
    if damage=='setup_receipt':write_immutable_json(root/'setup_pascal/receipt.json',{})
    if damage=='other_error':(root/f'setup_pascal-{core.SETUP_JOB}.out').write_text(audit.CAP_ERROR+'\nValueError: NaN\n')
    if damage is None:assert len(audit.failed_setup(p,fr))==2
    else:
        with pytest.raises(ValueError):audit.failed_setup(p,fr)


def timing_fixture(tmp_path):
    from test_luka_reconstruction_stability_hydra import submission_spec,resource_record
    from hlt_classification.luka_reconstruction_stability_hydra.transport_v2 import PROTOCOL
    p=submission_spec(tmp_path);p.update(recovery.registration(p));p['recovery_parent']={}
    value=resource_record(p);value['profiles']={'SOFT':value['profiles']['SOFT']}
    value['miniatures']={t:value['miniatures'][t] for t in core.TASKS}
    profile=value['profiles']['SOFT']
    profile['phase_samples']={'4.5':[11.03784811423005]*6,'8.99':[12.599750670798434]*6,'20.0':[.6208435070002452]*6}
    profile.update(step_seconds=max(float(np.mean(v)) for v in profile['phase_samples'].values()),validation_seconds=112.46087781200185,
                   probe_seconds=.9165465869009495,load_seconds=72.78469586605206)
    for mini in value['miniatures'].values():mini['seconds']=28.06172468746081
    value=with_content_hash(value)
    probe=c.artifact('EVOLVED_SOFT_PROBE',parents=dict(spec=p['content_hash']),task='SOFT_s8',train_passes=4,
        train_rows=p['counts']['train'],updates=4*math.ceil(p['counts']['train']/128),discarded=True,validation_accessed=False,
        phase_samples=profile['phase_samples'],memory=profile['memory'],seconds=16898.5,solver=PROTOCOL,
        numerical_cases=[[dict(marginal_residual=1e-5,normalized_gap=1e-5) for _ in range(3)] for _ in range(4)])
    write_immutable_json(tmp_path/'setup_pascal/evolved_soft_probe.json',probe)
    return p,value


def test_actual_resource_validator_preserves_safety_margin_and_bounds(tmp_path):
    p,value=timing_fixture(tmp_path);original=deepcopy(p)
    with pytest.raises(RuntimeError,match='48h cap'):resources.accept(p,value,'pascal')
    result=resources.accept(dict(p,fit_max_minutes=core.MAX_MINUTES),value,'pascal')
    assert set(result['minutes'])==set(core.TASKS)
    assert set(result['minutes'].values())=={7209}
    assert p==original and not (tmp_path/'setup_pascal/acceptance.json').exists()
    # We preserve the safety factor and all six active passes, not 30 soft passes.
    assert 120 < result['minutes']['SOFT_s7']/60 < 121
    bad=deepcopy(value);bad['profiles']['SOFT']['phase_samples']['8.99']=[100.]*6
    bad['profiles']['SOFT']['step_seconds']=100.;bad=with_content_hash(bad)
    with pytest.raises(RuntimeError):resources.accept(dict(p,fit_max_minutes=core.MAX_MINUTES),bad,'pascal')


@pytest.mark.parametrize('damage',['memory','evolved_cost','residual','missing_probe'])
def test_larger_walltime_does_not_waive_qualification(tmp_path,damage):
    p,value=timing_fixture(tmp_path);p=dict(p,fit_max_minutes=core.MAX_MINUTES)
    if damage=='memory':value['profiles']['SOFT']['memory']['host_peak_bytes']=100*2**30;value=with_content_hash(value)
    if damage=='evolved_cost':value['profiles']['SOFT']['phase_samples']['4.5']=[.1]*6;value=with_content_hash(value)
    if damage=='residual':
        path=tmp_path/'setup_pascal/evolved_soft_probe.json';probe=load_json(path)
        probe['numerical_cases'][0][0]['normalized_gap']=.01
        import json
        path.write_text(json.dumps(with_content_hash(probe)))
    if damage=='missing_probe':(tmp_path/'setup_pascal/evolved_soft_probe.json').unlink()
    with pytest.raises((ValueError,MemoryError,FileNotFoundError)):resources.accept(p,value,'pascal')


def queue_fixture(tmp_path,monkeypatch):
    tmp_path.mkdir(exist_ok=True)
    original=dict(root=str(tmp_path/'original'),content_hash='a'*64)
    qualification=core.artifact('QUALIFICATION',minutes={t:7208 for t in core.TASKS})
    write_immutable_json(tmp_path/'qualification.json',qualification)
    spec=dict(root=str(tmp_path),project_dir=str(tmp_path/'code'),content_hash='b'*64,
              minutes=qualification['minutes'],qualification=file_ref(tmp_path/'qualification.json'))
    monkeypatch.setattr(core,'check',lambda *a,**kw:(original,None,None,None,None,None,None))
    monkeypatch.setattr(audit,'qualify',lambda _:qualification)
    from hlt_classification.scouting import hcwdl_exact_dag_submission as dag
    calls=[];reviews=[]
    monkeypatch.setattr(submission,'scheduler_review',lambda p:reviews.append(p))
    def run(command,**kw):calls.append(command);return SimpleNamespace(stdout=str(9000+len(calls))+'\n',returncode=0)
    monkeypatch.setattr(dag.subprocess,'run',run)
    return spec,calls,reviews


def test_only_three_soft_jobs_and_afterany_report_idempotent(tmp_path,monkeypatch):
    spec,calls,reviews=queue_fixture(tmp_path,monkeypatch);plan=submission.plan(spec)
    with pytest.raises(PermissionError):submission.launch(spec,execute=True,plan_hash=plan['content_hash'],authorization=core.AUTHORIZATION)
    assert calls==[]
    submission.launch(spec)
    result=submission.launch(spec,execute=True,plan_hash=plan['content_hash'],authorization=core.AUTHORIZATION)
    assert list(result['jobs'])==[*core.TASKS,'report_pascal']
    assert len(calls)==4 and len(reviews)==1
    assert all('--gres=gpu:geforce_gtx_1080_ti:4' in row for row in calls[:3])
    assert all('--time=5-00:08:00' in row for row in calls[:3])
    assert '--dependency=afterany:9001:9002:9003' in calls[-1]
    assert not any('setup_pascal' in str(row) or any(t in str(row) for t in core.SALVAGE) for row in calls)
    assert submission.launch(spec,execute=True,plan_hash=plan['content_hash'],authorization=core.AUTHORIZATION)==result
    assert len(calls)==4


def test_alternative_root_cannot_duplicate_soft_jobs(tmp_path,monkeypatch):
    first,calls,_=queue_fixture(tmp_path/'first',monkeypatch)
    original=dict(root=str(tmp_path/'original'),content_hash='a'*64)
    monkeypatch.setattr(core,'check',lambda *a,**kw:(original,None,None,None,None,None,None))
    submission.launch(first);plan=submission.plan(first)
    submission.launch(first,execute=True,plan_hash=plan['content_hash'],authorization=core.AUTHORIZATION)
    second=dict(first,root=str(tmp_path/'second'),content_hash='c'*64);Path(second['root']).mkdir()
    submission.launch(second);plan=submission.plan(second)
    with pytest.raises(RuntimeError,match='already owns'):
        submission.launch(second,execute=True,plan_hash=plan['content_hash'],authorization=core.AUTHORIZATION)
    assert len(calls)==4


def test_scheduler_rejection_submits_nothing(tmp_path,monkeypatch):
    spec,calls,_=queue_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(submission,'scheduler_review',lambda _:(_ for _ in ()).throw(RuntimeError('QOS time cap')))
    with pytest.raises(RuntimeError,match='QOS'):submission.launch(spec)
    assert calls==[]


def test_no_replacement_route_for_live_existing_or_salvaged_tasks(tmp_path):
    p=dict(root='original');s=dict(root='storage');f=dict(root='soft')
    r=lambda _:None;sr=lambda _:storage_core;fr=lambda _:None
    values=(p,s,f,None,r,sr,fr);spec=dict(root='new')
    for task in core.SALVAGE:
        assert reporting.location(spec,values,task)[0]==Path('original')/task
    for task in storage_core.RESTART:
        assert reporting.location(spec,values,task)[0]==Path('storage')/task
    for task in core.TASKS:
        folder,scientific,worker,kind=reporting.location(spec,values,task)
        assert folder==Path('new')/task and scientific is f and worker is fr
    with pytest.raises(ValueError,match='Only the three'):runtime.rank(spec,'DEC2_s9')


def test_allocation_retains_original_validator_and_restores_lookup(monkeypatch):
    p={'science':'unchanged'};spec={};seen=[]
    old=lambda *a:'old'
    worker=SimpleNamespace(require_job=old)
    def allocate(actual,task):
        assert actual is p;seen.append(worker.require_job(actual,task));return seen[-1],['devices']
    worker.allocation=allocate
    monkeypatch.setattr(submission,'require_job',lambda *a:'123')
    assert runtime.allocation(spec,p,lambda _:worker,'SOFT_s7')==('123',['devices'])
    assert worker.require_job is old and seen==['123']


def registration_fixture(tmp_path,monkeypatch):
    p,r,modules=original_fixture(tmp_path/'original')
    p.update(source_commit=core.ORIGINAL,project_dir=str(tmp_path/'original_code'))
    storage=dict(content_hash='e'*64,root=str(tmp_path/'storage'),project_dir=str(tmp_path/'storage_code'))
    soft=dict(content_hash='f'*64,root=str(tmp_path/'soft'),project_dir=str(tmp_path/'soft_code'),environment={},environment_dir='env')
    old={k:str(tmp_path/('old_'+k)) for k in ('root','project_dir','environment_dir','bundle_root','raw_root','delta_root')}
    sr=lambda _:storage_core;values=(p,storage,soft,old,r,sr,None)
    records=audit.salvage(p,r,lambda jobs:{t:['FAILED','1:0'] for t in jobs})
    qualification=core.artifact('QUALIFICATION',parents=dict(original=p['content_hash'],storage=storage['content_hash'],soft=soft['content_hash']),
        salvaged=records,failure_references=[],setup_references=[],resource_overrides=dict(fit_max_minutes=core.MAX_MINUTES),
        walltime_policy=core.POLICY,minutes={t:7209 for t in core.TASKS},devices=[],environment={},
        storage=dict(scientific_bytes=1000),diagnostic='1'*64,technical_weights_discarded=True)
    storage_path=tmp_path/'storage.json';soft_path=tmp_path/'soft.json'
    write_immutable_json(storage_path,storage);write_immutable_json(soft_path,soft)
    monkeypatch.setattr(core,'parents',lambda *a,**kw:values)
    monkeypatch.setattr(core,'source',lambda *a:dict(content_hash='2'*64))
    monkeypatch.setattr(audit,'qualify',lambda *a:qualification)
    monkeypatch.setattr(storage_core.shutil,'disk_usage',lambda _:SimpleNamespace(free=0))
    spec=core.create(tmp_path/'new',storage_path,soft_path,tmp_path/'code','3'*40)
    return spec,values


def test_registration_keeps_science_immutable_and_low_space_advisory(tmp_path,monkeypatch):
    spec,values=registration_fixture(tmp_path,monkeypatch)
    assert core.check(spec)==values
    assert spec['tasks']==list(core.TASKS) and spec['salvaged']==core.SALVAGE
    assert spec['storage_observation']['available_bytes']==0
    assert not any((Path(spec['root'])/t).exists() for t in (*core.SALVAGE,*core.TASKS))
    with pytest.raises(ValueError):core.create(values[0]['root'],spec['storage_spec']['path'],spec['soft_spec']['path'],tmp_path/'code','3'*40)
    path=Path(values[0]['root'])/'DEC2_s7/fit/receipt.json';path.write_text('corrupt')
    with pytest.raises(ValueError):core.check(spec)


def test_changed_resource_policy_cannot_reuse_registration(tmp_path,monkeypatch):
    spec,_=registration_fixture(tmp_path,monkeypatch)
    bad=with_content_hash(dict(spec,max_minutes=999999))
    with pytest.raises(ValueError,match='registration'):core.check(bad)


def test_combined_results_retain_mismatches_and_all_recovery_routes(tmp_path,monkeypatch):
    p,r,_=original_fixture(tmp_path/'old')
    p['tasks']=['C128_s7','PROJ_s7','DEC2_s8','SOFT_s7']
    storage=dict(root='storage',content_hash='s'*64)
    oldreport=importlib.import_module('hlt_classification.luka_reconstruction_stability_hydra.reporting')
    oldstorage=importlib.import_module('hlt_classification.luka_stability_storage_recovery.reporting')
    modules=dict(contracts=c,reporting=SimpleNamespace(states=lambda _: {},stability=oldreport.stability,
        for_readout=oldreport.for_readout,summarize=oldreport.summarize))
    srmodules=dict(core=SimpleNamespace(accounting=lambda _:{}),submission=SimpleNamespace(accepted_jobs=lambda _:{}),reporting=oldstorage)
    values=(p,storage,{},None,lambda n:modules[n],lambda n:srmodules[n],None)
    qpath=tmp_path/'qualification.json';write_immutable_json(qpath,{})
    spec=dict(content_hash='x'*64,qualification=file_ref(qpath))
    monkeypatch.setattr(core,'check',lambda _:values)
    monkeypatch.setattr(submission,'accepted_jobs',lambda _: {})
    initial=dict(components=dict(native='n',predictor='p'),seeds={},dropout_rank_seeds=[1],epoch1_order_hash='o')
    def result(spec,values,task,q):
        if task=='SOFT_s7':return None
        auc=.962 if task=='C128_s7' else .9625
        readouts={name:dict(epoch=12,validation=dict(macro_ovr_auc=auc)) for name in c.READOUTS}
        report=dict(selected_readouts=readouts,history=[dict(epoch=e,readouts={name:dict(macro_ovr_auc=auc) for name in c.READOUTS}) for e in range(1,31)])
        start=deepcopy(initial)
        if task=='PROJ_s7':start['components']['native']='different'
        return report,dict(identities=np.arange(11),labels=np.arange(11)),start,dict(kind=task)
    monkeypatch.setattr(reporting,'completed',result)
    value=reporting.results(spec,'all')
    assert value['pending']==['SOFT_s7'] and len(value['rows'])==12
    assert all(row['delta'] is None for row in value['rows'] if row['task']=='PROJ_s7')
    assert all(row['recovery']>0 for row in value['rows'])
    assert value['summary']['RAW']['PROJ']['completed_seeds']==1


def test_actual_setup_evidence_validation_does_not_require_fake_receipt(tmp_path,monkeypatch):
    p,value=timing_fixture(tmp_path);root=tmp_path/'setup_pascal';original=deepcopy(p)
    write_immutable_json(root/'resource_diagnostic.json',value)
    write_immutable_json(root/'qualification.json',{})
    write_immutable_json(root/'bn_membership.json',c.artifact('BN_MEMBERSHIP',parents=dict(spec=p['content_hash'])))
    # Match content hash parents to genuine membership artifact.
    membership=load_json(root/'bn_membership.json');value['parents']['bn_membership']=membership['content_hash']
    import json
    (root/'resource_diagnostic.json').write_text(json.dumps(with_content_hash(value)))
    for task in core.TASKS:write_immutable_json(root/('mini_'+task)/'initialization.json',{})
    calls=[]
    resource=SimpleNamespace(accept=resources.accept,check_qualification=lambda *a:calls.append('parity'),check_replay=lambda *a:calls.append('replay'))
    modules=dict(resources=resource,calibration=SimpleNamespace(load=lambda _:dict(content_hash='c'*64)),contracts=c,
        recovery=SimpleNamespace(check_membership=lambda *a:calls.append('membership'),check_initial=lambda *a:calls.append('initial')),
        identity=SimpleNamespace(validate_pairs=lambda *a,**kw:calls.append('pairs')),
        training=SimpleNamespace(validate_fit=lambda *a,**kw:(dict(content_hash='d'*64,bn_membership=membership['content_hash'],world_size=4),{})))
    accepted,refs=audit.soft_qualification(p,{},lambda n:modules[n])
    assert set(accepted['minutes'].values())=={7209} and p==original
    assert calls.count('initial')==3 and all(name in calls for name in ('parity','replay','membership','pairs'))
    assert refs and not (root/'receipt.json').exists() and not (root/'acceptance.json').exists()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Installed Weaver CUDA required')
def test_installed_weaver_soft_startup_exports_for_completion(tmp_path,monkeypatch):
    # Unchanged pinned worker path: test its real two-epoch miniature with
    # numerical-v2 configuration, all readouts and exact initialization check.
    from test_luka_stability_soft_recovery import test_recovery_installed_weaver_native_exports
    test_recovery_installed_weaver_native_exports(tmp_path,monkeypatch)


def test_files_and_isolated_wrappers():
    import ast
    assert all(Path(p).is_file() for p in core.REQUIRED)
    for name in core.MODULES:ast.parse(Path('src/hlt_classification/luka_stability_completion_hydra',name+'.py').read_text())
    wrapper=Path('sbatch/run_luka_stability_completion_hydra.sh').read_text()
    assert 'PYTHONNOUSERSITE=1' in wrapper and 'CUBLAS_WORKSPACE_CONFIG=:4096:8' in wrapper
    assert '${ENV_DIR}/lib' in wrapper and '${PROJECT_DIR}/scripts/' in wrapper

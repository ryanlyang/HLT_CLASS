"""Storage-only failures must not erase valid science or duplicate a failed fit."""
import importlib
from pathlib import Path
from types import SimpleNamespace
import errno
import numpy as np
import pytest
import torch
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_stability_storage_recovery import core, submission, runtime, reporting
from hlt_classification.luka_reconstruction_stability_hydra import contracts as c


def fake_parent(root):
    root.mkdir()
    p = dict(root=str(root), content_hash='a'*64, source_commit=core.PARENT_COMMIT, project_dir=str(root/'code'),
             environment_dir=str(root/'env'), tasks=list(c.FITS), storage_inventory=dict(estimated_fit_bytes=500*1024**2),
             recovery_anchors=dict(M0HLT=.96056122, OFFLINE=.96452569))
    modules = {}
    def fit(folder, p, task, technical=False):
        return {'content_hash': ('c' if technical else 'd')*64}, {}
    accepted = c.artifact('ACCEPTANCE', parents=dict(spec=p['content_hash']), minutes={t:1200 for t in core.RESTART})
    modules.update(contracts=c, training=SimpleNamespace(validate_fit=fit),
        submission=SimpleNamespace(accepted_jobs=lambda *a:dict(core.FAILED)),
        worker=SimpleNamespace(completed_setup=lambda *a:accepted))
    for task, job in core.FAILED.items():
        (root/f'{task}-{job}.out').write_text('Traceback\n'+core.MESSAGE+'\n')
        if task in core.SALVAGE:
            for name in ('fit/receipt.json','startup/receipt.json'):
                path=root/task/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
            write_immutable_json(root/task/'startup_memory.json',c.artifact('STARTUP',parents=dict(spec=p['content_hash']),
                task=task,miniature='c'*64,discarded=True))
            write_immutable_json(root/'claims'/task/'intent.json',c.artifact('RUN_CLAIM',parents=dict(spec=p['content_hash']),task=task,job_id=job))
    write_immutable_json(root/'setup_pascal/acceptance.json',accepted)
    write_immutable_json(root/'campaign_spec.json',p)
    return p,lambda part:modules[part],modules


def fixture(tmp_path,monkeypatch):
    p,r,modules=fake_parent(tmp_path/'parent')
    old={k:str(tmp_path/('old_'+k)) for k in ('root','project_dir','environment_dir','bundle_root','raw_root','delta_root')}
    monkeypatch.setattr(core,'parent',lambda *a,**kw:(p,old,r))
    snap=dict(content_hash='b'*64,git_commit='e'*40)
    monkeypatch.setattr(core,'source',lambda *a:snap)
    monkeypatch.setattr(core,'accounting',lambda jobs:{t:['FAILED','1:0'] for t in jobs})
    spec=core.create(tmp_path/'new',tmp_path/'parent/campaign_spec.json',tmp_path/'newcode','e'*40)
    return spec,p,r,modules


def test_space_observation_never_vetoes_low_space(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(core.shutil,'disk_usage',lambda _:SimpleNamespace(free=0))
    assert core.observe_space(tmp_path,29*2**30)['available_bytes']==0
    assert 'no predictive admission gate' in capsys.readouterr().out
    spec,_,_,_=fixture(tmp_path,monkeypatch)
    assert spec['storage_policy']==core.POLICY and spec['storage_observation']['available_bytes']==0
    assert core.check(spec)


def test_actual_write_error_is_not_swallowed(tmp_path,monkeypatch):
    spec,p,r,_=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(core,'write_immutable_json',lambda *a:(_ for _ in ()).throw(OSError(errno.ENOSPC,'No space left on device')))
    with pytest.raises(OSError,match='No space'):
        core.create(tmp_path/'second',tmp_path/'parent/campaign_spec.json',tmp_path/'code','e'*40)


@pytest.mark.parametrize('damage',['active','wrong_job','other_error','old_receipt','partial','corrupt_fit','bad_startup','bad_claim'])
def test_failure_audit_rejects_unsafe_recovery(tmp_path,damage):
    p,r,m=fake_parent(tmp_path/'p');states={t:['FAILED','1:0'] for t in core.FAILED}
    task=next(iter(core.RESTART));saved=next(iter(core.SALVAGE));root=Path(p['root'])
    if damage=='active':states[task]=['RUNNING','0:0']
    if damage=='wrong_job':m['submission'].accepted_jobs=lambda *a:{**core.FAILED,task:'999'}
    if damage=='other_error':(root/f'{task}-{core.RESTART[task]}.out').write_text(core.MESSAGE+'\nValueError: invalid data\n')
    if damage=='old_receipt':(root/saved/'receipt.json').write_text('{}')
    if damage=='partial':(root/task).mkdir()
    if damage=='corrupt_fit':m['training'].validate_fit=lambda *a,**kw:(_ for _ in ()).throw(ValueError('corrupt fit'))
    if damage=='bad_startup':(root/saved/'startup_memory.json').write_text('{}')
    if damage=='bad_claim':(root/'claims'/saved/'intent.json').write_text('{}')
    with pytest.raises((ValueError,KeyError)):core.evidence(p,r,states=states)


def test_three_salvages_only_seven_restarts_and_preserved_inputs(tmp_path,monkeypatch):
    spec,p,r,_=fixture(tmp_path,monkeypatch)
    assert len(spec['salvaged'])==3 and len(spec['tasks'])==7
    assert set(spec['salvaged']).isdisjoint(spec['tasks'])
    assert all(not (Path(p['root'])/t/'receipt.json').exists() for t in core.SALVAGE)
    assert not any((Path(spec['root'])/t).exists() for t in core.SALVAGE)
    core.check(spec)
    (Path(p['root'])/'DEC2_s8-180559.out').write_text('changed')
    with pytest.raises(ValueError):core.check(spec)


def scheduler(monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as dag
    calls=[];reviews=[]
    monkeypatch.setattr(submission,'scheduler_review',lambda p:reviews.append(p))
    def run(args,**kwargs):calls.append(args);return SimpleNamespace(stdout=str(9000+len(calls))+'\n')
    monkeypatch.setattr(dag.subprocess,'run',run)
    return calls,reviews


def test_exact_eight_jobs_and_idempotence(tmp_path,monkeypatch):
    spec,_,_,_=fixture(tmp_path,monkeypatch);calls,reviews=scheduler(monkeypatch)
    submission.launch(spec);plan=submission.plan(spec)
    assert len(reviews)==1 and len(plan['commands'])==8
    assert [v['task_id'] for v in plan['commands'][:-1]]==list(core.RESTART)
    assert all('--gres=gpu:geforce_gtx_1080_ti:4' in v['command'] and not v['dependencies'] for v in plan['commands'][:-1])
    assert plan['commands'][-1]['dependencies']==list(core.RESTART)
    result=submission.launch(spec,execute=True,plan_hash=plan['content_hash'],authorization=core.AUTHORIZATION)
    assert len(calls)==8 and submission.accepted_jobs(spec)==result['jobs']
    assert '--dependency=afterany:9001:9002:9003:9004:9005:9006:9007' in calls[-1]
    assert submission.launch(spec,execute=True,plan_hash=plan['content_hash'],authorization=core.AUTHORIZATION)==result
    assert len(calls)==8


@pytest.mark.parametrize('case',['no_review','authorization','partial','other_root'])
def test_submission_rejects_ambiguity(tmp_path,monkeypatch,case):
    spec,p,_,_=fixture(tmp_path,monkeypatch);calls,_=scheduler(monkeypatch);plan=submission.plan(spec)
    if case!='no_review':submission.launch(spec)
    if case=='partial':write_immutable_json(Path(spec['root'])/'submission/live_claim.json',{})
    if case=='other_root':write_immutable_json(Path(p['root']).parent/('stability_storage_'+p['content_hash']+'.json'),{})
    with pytest.raises((PermissionError,RuntimeError)):
        submission.launch(spec,execute=True,plan_hash=plan['content_hash'],authorization='wrong' if case=='authorization' else core.AUTHORIZATION)
    assert not calls


def test_allocation_reuses_exact_site_checks_and_restores_lookup(tmp_path,monkeypatch):
    spec,p,r,m=fixture(tmp_path,monkeypatch);original=lambda *a:'old'
    m['worker'].require_job=original
    m['worker'].allocation=lambda old,task:(m['worker'].require_job(old,task),['validated'])
    monkeypatch.setattr(submission,'require_job',lambda *a:'new')
    assert runtime.allocation(spec,p,r,'PROJ_s9')==('new',['validated'])
    assert m['worker'].require_job is original


def test_worker_completion_ignores_forecast_and_keeps_exact_claim(tmp_path,monkeypatch):
    spec,p,r,m=fixture(tmp_path,monkeypatch);task='C512_s9';calls=[]
    monkeypatch.setattr(runtime,'allocation',lambda *a:('9001',[]))
    monkeypatch.setattr(runtime,'require_backend',lambda:None)
    monkeypatch.setattr(runtime.reconstruction_targets,'authenticate',lambda *a:None)
    monkeypatch.setattr(runtime.subprocess,'run',lambda args,**kw:calls.append(args))
    monkeypatch.setattr(core.shutil,'disk_usage',lambda _:SimpleNamespace(free=0))
    value=runtime.run(spec,task)
    assert '--nproc-per-node=4' in calls[0] and value['job_id']=='9001'
    assert (Path(spec['root'])/task/'receipt.json').is_file()
    with pytest.raises(FileExistsError):runtime.run(spec,task)


def test_pairing_differences_are_visible():
    a=dict(components=dict(native='a',predictor='b'),seeds={},dropout_rank_seeds=[1],epoch1_order_hash='c')
    b={**a,'components':dict(native='other',predictor='b')}
    assert reporting.differences(a,a)=={}
    assert reporting.differences(a,b)=={'native':['a','other']}


def test_unsubmitted_results_need_no_accounting_query(monkeypatch):
    def forbidden(*a,**kw): raise AssertionError('No empty sacct query')
    monkeypatch.setattr(core.subprocess,'run',forbidden)
    assert core.accounting({})=={}


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Installed Weaver CUDA required')
def test_installed_weaver_cuda_seven_restart_paths(tmp_path,monkeypatch):
    from test_luka_direct_reconstruction import fixture as data_fixture
    from hlt_classification.luka_reconstruction_stability_hydra import training,campaign,kernel,calibration
    from hlt_classification.luka_next_supervision_hydra import reconstruction_targets
    from hlt_classification.luka_subclass_ddp_hydra.distributed import Context
    old,tr,va,bank=data_fixture(tmp_path,monkeypatch,rows=22)
    va.identities[:,-1]=1
    for block in va.blocks:block.identities[:,-1]=1
    spec=dict(campaign.registration(),root=str(tmp_path),content_hash='a'*64,parents=dict(original='b'*64),
        maximum=12,counts=dict(train=len(tr),validation=len(va)),imported=old['imported'],
        foundation_hash=tr.foundation_sha256,embedding_hash=bank.manifest['content_hash'],
        recovery_anchors=dict(M0HLT=.96056122,OFFLINE=.96452569))
    # The seven restart arms use hard reconstruction or no decoder. This local
    # synthetic single-CUDA miniature does not requalify four Pascal GPUs.
    monkeypatch.setattr(calibration,'load',lambda spec:dict(scale=.1))
    targets=reconstruction_targets.Targets(old,tr,'OFFLINE');ctx=Context(0,1,torch.device('cuda'))
    previous=torch.get_num_threads();kernel.configure(1,3)
    try:
        for task in core.RESTART:
            report=training.fit(tmp_path/task,spec,task,tr,va,bank,targets,ctx,technical=True)
            assert set(report['selected_readouts'])==set(c.READOUTS)
            training.validate_fit(tmp_path/task,spec,task,technical=True)
    finally:
        targets.close();torch.cuda.empty_cache();torch.set_num_threads(previous)


def _portable_worker(rank,root,init):
    import torch.distributed as dist
    from datetime import timedelta
    from test_luka_reconstruction_stability_hydra import portable_fixture,PortableStudent,PortableNative
    from test_luka_direct_reconstruction import target
    from hlt_classification.luka_reconstruction_stability_hydra import training,readouts,kernel,optimization
    from hlt_classification.luka_subclass_ddp_hydra.distributed import Context
    torch.set_num_threads(1);dist.init_process_group('gloo',init_method=init,rank=rank,world_size=4,timeout=timedelta(minutes=8))
    try:
        def build(spec,task,rank,world,device,**kw):
            torch.manual_seed(c.seeds(spec,task)['native_initialization']);model=PortableStudent(task)
            torch.manual_seed(c.seeds(spec,task)['dropout_base']+rank);return model,optimization.optimizer(model)
        kernel.build=build;training.DelphesParticleTransformer=PortableNative
        readouts.load_membership=lambda spec,tr,**kw:readouts.membership(spec,tr,technical=True)
        p,tr,va,bank=portable_fixture(Path(root)/'parent');ctx=Context(rank,4,torch.device('cpu'))
        targets=SimpleNamespace(batch=lambda ix,epoch,cfg:([target(1+int(i)%4) for i in ix],[]))
        training.fit(Path(root)/'parent/C512_s8/fit',p,'C512_s8',tr,va,bank,targets,ctx)
        training.fit(Path(root)/'new/C512_s9/fit',p,'C512_s9',tr,va,bank,targets,ctx)
    finally:dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(),reason='Gloo required')
def test_actual_full30_salvage_and_replacement_reports(tmp_path,monkeypatch):
    from test_luka_reconstruction_stability_hydra import portable_fixture
    torch.multiprocessing.spawn(_portable_worker,args=(str(tmp_path),(tmp_path/'store').as_uri()),nprocs=4,join=True)
    p,_,_,_=portable_fixture(tmp_path/'parent');p['tasks']=['C512_s8','C512_s9']
    r=lambda name:importlib.import_module('hlt_classification.luka_reconstruction_stability_hydra.'+name)
    report,_=r('training').validate_fit(tmp_path/'parent/C512_s8/fit',p,'C512_s8')
    spec=dict(root=str(tmp_path/'new'),content_hash='e'*64,failures={'C512_s8':{'report':report['content_hash']}})
    monkeypatch.setattr(core,'check',lambda *a,**kw:(p,{},r))
    monkeypatch.setattr(r('reporting'),'states',lambda *a:{})
    monkeypatch.setattr(core,'accounting',lambda *a:{})
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:{'C512_s9':'9001'})
    startup=tmp_path/'new/C512_s9/startup';startup.mkdir();(startup/'receipt.json').write_text('{}')
    (tmp_path/'new/C512_s9/startup_memory.json').write_text('{}')
    folder=tmp_path/'new/C512_s9'
    receipt=core.artifact('TASK',parents=dict(spec=spec['content_hash'],original=p['content_hash']),task='C512_s9',job_id='9001',
        outputs=[file_ref(folder/name,root=folder) for name in ('startup/receipt.json','startup_memory.json','fit/receipt.json','fit/initialization.json')])
    write_immutable_json(folder/'receipt.json',receipt)
    actual_validate=r('training').validate_fit
    monkeypatch.setattr(r('training'),'validate_fit',lambda path,*a,**kw:({}, {}) if kw.get('technical') else actual_validate(path,*a,**kw))
    value=reporting.results(spec,readout='all');progress=reporting.progress(spec,readout='all')
    assert len(value['rows'])==8 and not value['pending']
    assert value['summary']['EMA05']['C512']['completed_seeds']==2
    assert all(row['delta']==0 for row in value['rows'])
    assert all(len(rows)==30 for rows in progress['epochs'].values())
    assert not (tmp_path/'parent/C512_s8/receipt.json').exists()
    (tmp_path/'parent/C512_s8/fit/RAW/selected.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError):reporting.results(spec)


def test_source_inventory_and_wrappers():
    import ast
    assert all(Path(p).is_file() for p in core.REQUIRED)
    for name in core.MODULES:
        ast.parse(Path('src/hlt_classification/luka_stability_storage_recovery',name+'.py').read_text())
    wrapper=Path('sbatch/run_luka_stability_storage_recovery.sh').read_text()
    assert 'PYTHONNOUSERSITE=1' in wrapper and 'CUBLAS_WORKSPACE_CONFIG=:4096:8' in wrapper
    assert '${ENV_DIR}/lib' in wrapper and '${PROJECT_DIR}/scripts/' in wrapper

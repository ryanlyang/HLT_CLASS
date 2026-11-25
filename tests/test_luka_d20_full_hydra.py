"""Full-population science, five-epoch continuation, real inputs and Slurm gates."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import math
import subprocess
import numpy as np
import pytest
import torch
from torch import nn
from hlt_classification.data.cache_contracts import load_json,write_immutable_json,sha256_file,with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.jetclass2_delphes.cache import RamCache,RamBlock
from hlt_classification.luka_subclass_ddp_hydra.distributed import Context
from hlt_classification.luka_next_supervision_hydra.reconstruction_model import Decoder
from hlt_classification.luka_next_supervision_hydra import reconstruction_batched as objective
from hlt_classification.luka_fullstat_readout.training import optimizer
from hlt_classification.luka_d20_full_hydra import (
    contracts as c,campaign,worker,training,artifacts,kernel,resources,model,banks,readouts,
    inputs,population,normalization,storage,launch,startup,reporting)
from hlt_classification.luka_d20_full_hydra.sampling import BalancedSampler
from test_luka_balanced_reconstruction_hydra import ToyModule,ToyNative
from test_luka_direct_reconstruction import target
from test_luka_sixclass_balanced_oscar import cache
from test_luka_fullstat import write_root


@pytest.fixture(autouse=True)
def single_thread():
    prior=torch.get_num_threads();torch.set_num_threads(1);workers=objective.ASSIGNMENT_WORKERS
    yield
    if objective.solver_pool.cache_info().currsize:objective.solver_pool().shutdown(wait=True)
    objective.solver_pool.cache_clear();objective.ASSIGNMENT_WORKERS=workers;torch.set_num_threads(prior)


class Tiny(nn.Module):
    def __init__(self,task,maximum):
        super().__init__();self.mod=ToyModule();self.capture=False
        self.predictor=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,128))
        self.aux=Decoder(c.decoder_config(task),maximum) if c.cell(task)['reconstruction'] else None
        self.clear()
    def clear(self):self.jet=self.prediction=self.particle_tokens=self.particle_mask=None
    def forward(self,features,vectors,mask):
        self.particle_tokens=self.mod.encoder(features.transpose(1,2));self.particle_mask=mask[:,0]
        self.jet=(self.particle_tokens*self.particle_mask[...,None]).sum(1)/self.particle_mask.sum(1)[:,None]
        self.prediction=self.predictor(self.jet) if self.capture else None
        return self.mod.fc(self.jet)
    def no_weight_decay(self):return set()
    def deployment_state(self):return {k:v.detach().cpu().clone() for k,v in self.state_dict().items() if k.startswith('mod.')}


def test_registered_science_exact_full_population_and_23_fit_matrix():
    assert c.COUNTS==dict(train=2488148,validation=497954)
    assert c.BATCH==256 and c.EPOCHS==20 and math.ceil(c.COUNTS['train']/c.BATCH)==9720
    assert c.COUNTS['train']%c.BATCH==84 and len(c.FITS)==23 and len(c.GPU_TASKS)==92
    assert c.SEEDS==(20,21,22) and c.recipe()['primary']=='RAW'
    assert c.learning_rate(2)==1e-4 and c.learning_rate(20)==5e-6
    assert c.recipe()['optimizer']['peak_lr']==1e-4 and 'precision' not in c.recipe()['optimizer']
    assert c.readouts('D20_s20')==('RAW',) and c.readouts('OFFLINE_s20')==('RAW','EMA05')
    for arm,change in c.VARIANTS.items():
        base=c.cell('D20_s20');actual=c.cell(arm+'_s20')
        differences={k:v for k,v in actual.items() if v!=base[k] and k!='arm'}
        assert differences==change
        assert c.seeds(arm+'_s20')==c.seeds('D20_s20')
    assert c.weights('D20_E06_s20',10)==(.6,.3)
    assert c.weights('D20_E01_s20',10)==(.1,.3)
    assert c.weights('D20_NO_PREP_s20',2)==(0.,0.)
    assert c.cell('U100_s21')['teacher']=='OFFLINE_s21'
    assert c.cell('D20_EMB25_s20')['teacher']=='U100_s20'


def test_dag_has_only_teacher_and_same_fit_segment_dependencies():
    tasks=c.tasks();assert len(tasks)==96
    assert tasks['M0HLT_s20_p01']==[] and tasks['OFFLINE_s21_p01']==[]
    assert tasks['U100_s20_p01']==['OFFLINE_s20_p04']
    assert tasks['D20_s22_p01']==['U100_s22_p04']
    assert tasks['D20_E06_s20_p01']==['U100_s20_p04']
    assert tasks['D20_s20_p03']==['D20_s20_p02']
    for fit in c.FITS:
        assert [c.segment(fit+f'_p{i:02d}')[2:] for i in range(1,5)]==[(1,5),(6,10),(11,15),(16,20)]
    seen=set()
    for task,deps in tasks.items():assert set(deps)<=seen;seen.add(task)


@pytest.mark.parametrize('arm',c.ARMS)
@pytest.mark.parametrize('position',[2.,6.,10.])
def test_all_objectives_match_independent_full_graph(arm,position):
    task=arm+'_s20';torch.manual_seed(41);net=Tiny(task,4);kernel.disable_dropout(net);reference=deepcopy(net)
    raw=cache(rows=7).batch(np.arange(7));emb,q=resources.target_values(task,np.arange(7))
    targets=[target(1+i%4) for i in range(7)] if c.cell(task)['reconstruction'] else None
    opt=optimizer(net,c.recipe()['optimizer']);ropt=optimizer(reference,c.recipe()['optimizer'])
    expected=resources.independent_step(reference,ropt,raw,emb,q,targets,task,'cpu',position)
    losses=kernel.update(net,opt,raw,emb,q,targets,task,Context(0,1,torch.device('cpu')),position)
    proof=resources.compare(expected,resources.snapshot(net,opt,losses['total']),resources.TOLERANCES)
    assert proof['passed'],proof


def test_real_root_requests_are_role_bounded_and_validation_has_no_offline(tmp_path,monkeypatch):
    path=tmp_path/'raw/a.root';write_root(path,[1]*6)
    record=dict(path='raw/a.root',sha256=sha256_file(path),tree_key='tree',entries=6)
    a=dict(labels=np.zeros(6,np.int64),jet_pt=np.full(6,250.),hlt_jet_nparticles=np.full(6,2),jet_nparticles=np.full(6,3))
    rs=np.array([0,1,2,3,4,0],np.uint8);parent=(dict(content_hash='b'*64),dict(files=[record]),[a],[rs],{})
    spans=list(inputs.arguments(dict(dataset=str(tmp_path)),parent,{},'D000',True))
    assert all(np.all(rs[arg[3]:arg[4]]==arg[5]) for arg in spans)
    original=inputs.reader;branches=[]
    class Guard:
        def __init__(self,inner):self.inner=inner
        def __enter__(self):self.tree=self.inner.__enter__();self.num_entries=self.tree.num_entries;return self
        def __exit__(self,*a):return self.inner.__exit__(*a)
        def arrays(self,keys,**kw):
            lo,hi=kw['entry_start'],kw['entry_stop'];assert np.all(rs[lo:hi]==rs[lo]) and rs[lo] in (0,1)
            if rs[lo]==1:assert all(k.startswith('hlt_') for k in keys)
            branches.append(keys);return self.tree.arrays(keys,**kw)
    monkeypatch.setattr(inputs,'reader',lambda *a,**kw:Guard(original(*a,**kw)))
    try:
        results=[inputs.work(arg) for arg in spans]
        assert branches and sum(len(targets) for _,targets in results)==2
    finally:inputs.close_reader()


def small_fixture(root,monkeypatch):
    root.mkdir(parents=True);monkeypatch.setattr(c,'COUNTS',dict(train=144,validation=12));monkeypatch.setattr(c,'SEEDS',(20,))
    monkeypatch.setattr(c,'BN_ROWS',128);monkeypatch.setitem(c.RESOURCES,'gpus',1)
    monkeypatch.setattr(model,'Student',Tiny);monkeypatch.setattr(model,'NativeParticleTransformer',ToyNative)
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    jobs=[dict(task=t,job=str(100+i)) for i,t in enumerate(c.GPU_TASKS)]
    monkeypatch.setattr(campaign,'accepted_jobs',lambda *a,**kw:jobs)
    spec=dict(root=str(root),content_hash='a'*64,population=dict(content_hash='b'*64),maximum=4)
    artifacts.prepare_starts(spec)
    tr=cache(rows=144);val=cache(rows=12,role='validation');val.identities[:,20]=1
    tr.foundation_sha256=val.foundation_sha256='b'*64
    pt=np.repeat(np.arange(250,2600,100),6).astype(float)
    norm=c.artifact('TARGET_NORMALIZATION',parents=dict(population='b'*64),role='train')
    class Targets:
        normalization=norm
        def batch(self,ix):return [target(1+int(i)%4) for i in ix]
    return spec,tr,val,pt,Targets(),jobs


def run_fit(spec,task,tr,val,pt,targets,job,end):
    tr=deepcopy(tr);val=deepcopy(val);tr.coordinate_name=val.coordinate_name=c.cell(task)['coordinate']
    bank=banks.Bank(spec,task,tr) if c.cell(task)['teacher'] else None
    return training.fit(spec,task,dict(train=tr,validation=val),targets if c.cell(task)['reconstruction'] else None,
        bank,pt,Context(0,1,torch.device('cpu')),artifacts.fit_root(spec,task)/'runs'/job/'fit',job,
        validation_pt=np.full(len(val),650.),end_epoch=end)


def test_four_segments_restore_exact_raw_checkpoints_and_preserve_epoch_history(tmp_path,monkeypatch):
    spec,tr,val,pt,targets,jobs=small_fixture(tmp_path/'one',monkeypatch)
    task='M0HLT_s20';job=next(r['job'] for r in jobs if r['task']==task+'_p04')
    expected=run_fit(spec,task,tr,val,pt,targets,job,20)
    spec,tr,val,pt,targets,jobs=small_fixture(tmp_path/'segmented',monkeypatch)
    for part,end in enumerate((5,10,15,20),1):
        job=next(r['job'] for r in jobs if r['task']==task+f'_p{part:02d}')
        actual=run_fit(spec,task,tr,val,pt,targets,job,end)
        assert len(list((artifacts.fit_root(spec,task)/'epochs').glob('*.json')))==end
        if end<20:assert actual['scientific_complete'] is False and not (artifacts.fit_root(spec,task)/'result.json').exists()
    a=torch.load(expected['files']['RAW_checkpoint']['path'],weights_only=True)
    b=torch.load(actual['files']['RAW_checkpoint']['path'],weights_only=True)
    assert all(torch.equal(a[k],b[k]) for k in a)
    assert not list(artifacts.fit_root(spec,task).rglob('state_*.pt'))
    assert worker.segment_files(spec,task+'_p04')['result']['path'].endswith('result.json')


def test_teacher_export_and_student_bank_use_all_unique_train_and_raw_primary(tmp_path,monkeypatch):
    spec,tr,val,pt,targets,jobs=small_fixture(tmp_path/'teachers',monkeypatch)
    for task in ('OFFLINE_s20','U100_s20','U100_W5_s20'):
        job=next(r['job'] for r in jobs if r['task']==task+'_p04')
        result=run_fit(spec,task,tr,val,pt,targets,job,20)
        report=artifacts.completed(spec,task)['report'];assert report['primary']=='RAW'
        assert set(report['selected'])==set(c.readouts(task))
        if 'bank' in result['files']:
            manifest=load_json(validate_file_ref(result['files']['bank']))
            assert manifest['rows']==144 and manifest['readout']=='EMA05' and manifest['role']=='train'
    measured=reporting.results(spec)
    assert 'U100_W5_s20' in measured['models']
    assert measured['primary']=='RAW'


def test_d20_resume_probe_serializes_decoder_optimizer_and_rng(tmp_path,monkeypatch):
    spec,tr,val,pt,targets,jobs=small_fixture(tmp_path/'decoder',monkeypatch)
    proof=resources.resume_probe(spec,'D20_s20',dict(train=tr,validation=val),targets,None,pt,Context(0,1,torch.device('cpu')),tmp_path)
    assert proof['ranks'][0]['exact'] and proof['ranks'][0]['passed']
    state=torch.load(proof['state']['path'],weights_only=True)
    assert any(k.startswith('aux.') for k in state['model']) and state['optimizer']['state'] and len(state['rng'])==1


def test_disk_error_leaves_last_committed_epoch_and_recovery_replays_identically(tmp_path,monkeypatch):
    spec,tr,val,pt,targets,jobs=small_fixture(tmp_path/'expected',monkeypatch)
    task='M0HLT_s20';job=next(r['job'] for r in jobs if r['task']==task+'_p04')
    expected=run_fit(spec,task,tr,val,pt,targets,job,20)
    spec,tr,val,pt,targets,jobs=small_fixture(tmp_path/'interrupted',monkeypatch)
    real=artifacts.commit_epoch
    def fail(spec,task,folder,row,state,parents,job):
        if row['epoch']==3:raise OSError(28,'No space left on device')
        return real(spec,task,folder,row,state,parents,job)
    monkeypatch.setattr(artifacts,'commit_epoch',fail)
    with pytest.raises(RuntimeError,match='No space left'):run_fit(spec,task,tr,val,pt,targets,job,20)
    assert len(list((artifacts.fit_root(spec,task)/'epochs').glob('*.json')))==2
    jobs.append(dict(task=task+'_p04',job='999'));monkeypatch.setattr(artifacts,'commit_epoch',real)
    actual=run_fit(spec,task,tr,val,pt,targets,'999',20)
    a=torch.load(expected['files']['RAW_checkpoint']['path'],weights_only=True)
    b=torch.load(actual['files']['RAW_checkpoint']['path'],weights_only=True)
    assert all(torch.equal(a[k],b[k]) for k in a)


def test_export_failure_resumes_epoch20_without_retraining(tmp_path,monkeypatch):
    spec,tr,val,pt,targets,jobs=small_fixture(tmp_path/'export',monkeypatch)
    task='OFFLINE_s20';job=next(r['job'] for r in jobs if r['task']==task+'_p04')
    export=banks.export
    monkeypatch.setattr(banks,'export',lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('export interrupted')))
    with pytest.raises(RuntimeError,match='export interrupted'):run_fit(spec,task,tr,val,pt,targets,job,20)
    jobs.append(dict(task=task+'_p04',job='999'));monkeypatch.setattr(banks,'export',export)
    monkeypatch.setattr(artifacts,'commit_epoch',lambda *a,**kw:pytest.fail('Completed scientific epochs must not repeat'))
    result=run_fit(spec,task,tr,val,pt,targets,'999',20)
    assert artifacts.completed(spec,task)['receipt']==result


def test_full_audit_and_semantic_spec_reject_rehashed_recipe_changes(tmp_path):
    pop=load_json(Path(c.__file__).with_name('population_audit.json'))
    c.population_parent.validate(pop,'POPULATION')
    assert pop['content_hash']==c.POPULATION_HASH
    assert np.asarray(pop['class_bin_counts']).shape==(6,24) and np.asarray(pop['class_bin_counts']).min()==805
    assert c.COUNTS['train']/(144*805)==pytest.approx(21.4643547274)
    assert max(pop['maximum_support']['train'][s] for s in ('hlt','offline'))==316
    spec=c.artifact('SPEC',root=str(tmp_path),population=pop,recipe=c.recipe(),tasks=c.tasks(),resources=c.RESOURCES,
        execution_site=c.SITE,maximum=316,output_allowance_bytes=c.RESULT_BYTES,reserve_bytes=c.RESERVE_BYTES,
        transfer_manifest_sha256=c.population_parent.TRANSFER_HASH,protected_campaign=None,
        storage_policy=storage.policy(),input_execution_policy=inputs.policy(),worker_launch_policy=launch.policy())
    campaign.check(spec,live=False)
    bad=deepcopy(spec);bad['recipe']['batch']=512;bad=with_content_hash(bad)
    with pytest.raises(ValueError,match='Scientific/resource'):campaign.check(bad,live=False)


def test_full_support_output_inventory_accounts_for_all_concurrent_restarts():
    measured=storage.estimate(316)
    assert measured['concurrent_restart_allowance']>2*measured['maximum_restart_bytes']
    assert measured['conservative_output_bytes']<c.RESULT_BYTES


def test_late_variants_reach_full_coefficients_and_match_independent_gradients():
    for arm in c.VARIANTS:
        assert c.weights(arm+'_s20',20.)==(c.cell(arm+'_s20')['maximum'],.3)
        test_all_objectives_match_independent_full_graph(arm,20.)


def test_missing_full_normalizer_cannot_authorize_science(tmp_path):
    spec=dict(root=str(tmp_path),content_hash='a'*64)
    write_immutable_json(tmp_path/'qualification.json',c.artifact('QUALIFICATION',parents=dict(spec='a'*64)))
    with pytest.raises(ValueError,match='normalizer'):resources.qualification(spec)


def test_all_view_scratch_peak_is_checked_before_partial_input_builds():
    pop=load_json(Path(c.__file__).with_name('population_audit.json'))
    single=inputs.planned_peak(pop,[('D000',True)])
    qualification=inputs.planned_peak(pop,[('D000',True),('OFFLINE',False),('U100',False)])
    assert 50*2**30<single<qualification<60*2**30
    with pytest.raises(ValueError):inputs.planned_peak(pop,[('OFFLINE',True)])


def test_invalid_batch_header_is_rejected_before_any_scheduler_call(tmp_path,monkeypatch):
    spec=command_spec(tmp_path);script=Path(spec['project'])/'sbatch/run_luka_d20_full_hydra.sh'
    script.write_bytes(b'#!/usr/bin/env bash\r\nexit 0\r\n')
    monkeypatch.setattr(campaign.subprocess,'run',lambda *a,**kw:pytest.fail('Do not consult scheduler for a malformed script'))
    with pytest.raises(ValueError,match='shebang'):campaign.site(campaign.commands(spec,['prepare']))


def test_rejected_submission_preserves_response_and_does_not_release_children(tmp_path,monkeypatch):
    spec=command_spec(tmp_path);monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    def run(args,**kw):
        if args[0]=='scontrol':return SimpleNamespace(returncode=0,stdout='ClusterName = hydra\n',stderr='')
        return SimpleNamespace(returncode=0 if '--test-only' in args else 1,stdout='',stderr='definitive scheduler rejection')
    monkeypatch.setattr(campaign.subprocess,'run',run)
    with pytest.raises(RuntimeError,match='Uncertain'):campaign.submit(spec,['prepare','qualify'],execute=True,authorization=c.AUTH)
    journal=Path(spec['root'])/'submission/attempt_001'
    assert load_json(journal/'prepare.response.json')['returncode']==1
    assert not (journal/'qualify.intent.json').exists()
    with pytest.raises(ValueError,match='Ambiguous'):campaign.accepted_jobs(spec)


def test_bound_reader_can_finish_while_an_unrelated_submission_is_unresolved(tmp_path,monkeypatch):
    spec=command_spec(tmp_path);monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    def run(args,**kw):
        if args[0]=='scontrol':return SimpleNamespace(returncode=0,stdout='ClusterName = hydra\n',stderr='')
        ok='--test-only' in args or '--job-name=lukadfull_prepare' in args
        return SimpleNamespace(returncode=0 if ok else 1,stdout='123\n' if ok else '',stderr='')
    monkeypatch.setattr(campaign.subprocess,'run',run)
    with pytest.raises(RuntimeError):campaign.submit(spec,['prepare','qualify'],execute=True,authorization=c.AUTH)
    path=Path(spec['root'])/'starts.json';write_immutable_json(path,dict(technical=True))
    receipt=c.artifact('TASK',parents=dict(spec=spec['content_hash']),task='prepare',job='123',files=dict(starts=file_ref(path)))
    write_immutable_json(Path(spec['root'])/'tasks/prepare.json',receipt)
    assert worker.task_receipt(spec,'prepare')==receipt
    with pytest.raises(ValueError,match='Ambiguous'):campaign.accepted_jobs(spec)


def test_streaming_standardization_matches_population_moments():
    values=np.random.default_rng(7).normal(size=(65539,128)).astype(np.float32)
    mean,scale=banks.moments(values)
    np.testing.assert_allclose(mean,values.mean(0,dtype=np.float64),rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(scale,values.std(0,dtype=np.float64),rtol=1e-12,atol=1e-14)


def command_spec(tmp_path):
    project=tmp_path/'code';script=project/'sbatch/run_luka_d20_full_hydra.sh';script.parent.mkdir(parents=True)
    script.write_bytes(b'#!/usr/bin/env bash\nexit 0\n');root=tmp_path/'study';root.mkdir()
    return dict(root=str(root),project=str(project),content_hash='a'*64)


def test_actual_batch_script_review_precedes_submission_and_binds_dependencies(tmp_path,monkeypatch):
    spec=command_spec(tmp_path);calls=[];monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    monkeypatch.setattr(campaign.subprocess,'run',lambda args,**kw:SimpleNamespace(returncode=0,stdout='ClusterName = hydra\n' if args[0]=='scontrol' else (calls.append(args) or ('123\n' if len(calls)<5 else '124\n')),stderr=''))
    jobs=campaign.submit(spec,['prepare','qualify'],execute=True,authorization=c.AUTH)
    assert len(calls)==4 and all('--wrap=true' not in args and 'bash' not in args for args in calls)
    assert all(Path(args[campaign.script_index(args)]).name=='run_luka_d20_full_hydra.sh' for args in calls)
    # Exact accepted IDs are preserved even if a synthetic scheduler returned duplicates.
    assert '--dependency=afterok:123' in calls[-1]
    assert (Path(spec['root'])/'submission/attempt_001/qualify.response.json').exists()
    with pytest.raises(ValueError,match='Duplicate'):campaign.accepted_jobs(spec)


def test_transport_and_torchrun_contracts_require_four_pascal_ranks(tmp_path):
    spec=command_spec(tmp_path);argv=launch.command(spec['project'],spec['root'],'D20_s20_p02','/dev/shm/node')
    proof=launch.validate(argv);assert proof['processes']=='4' and proof['training_script_args'][-1]=='/dev/shm/node'
    assert startup.probe_tasks('qualify')==[arm+'_s20' for arm in c.BASE_ARMS]
    rows=campaign.commands(spec,list(c.tasks()))
    assert len(rows)==96 and sum('--gres=gpu:geforce_gtx_1080_ti:4' in r['command'] for r in rows)==93
    for row in rows:
        if row['task'] in c.GPU_TASKS:assert '--mem=96000M' in row['command'] and '--time=2880' in row['command']


def test_clean_deployment_closure_and_stale_source_rejection(tmp_path):
    """A separate local Git fixture, no working-repository commit or remote call."""
    import sys,importlib
    root=Path(__file__).resolve().parents[1];project=tmp_path/'deployment';project.mkdir()
    owned=root/'src/hlt_classification/luka_d20_full_hydra'
    pins=load_json(owned/'donors.json');files=set(pins['files'])
    files|={p.relative_to(root).as_posix() for p in owned.iterdir() if p.suffix in ('.py','.json')}
    files|={'scripts/luka_d20_full_hydra.py','scripts/launch_luka_d20_full_hydra.sh',
        'scripts/queue_luka_d20_full_hydra.sh','sbatch/run_luka_d20_full_hydra.sh',
        'sbatch/luka_d20_full_hydra_common.sh','tests/test_luka_d20_full_hydra.py',
        'docs/plans/LUKA_D20_FULL_HYDRA_PLAN.md','docs/contracts/LUKA_D20_FULL_HYDRA.md',
        'docs/LUKA_D20_FULL_HYDRA_RUNBOOK.md','docs/LUKA_D20_FULL_HYDRA_VALIDATION.md'}
    for name in files:
        target_path=project/name;target_path.parent.mkdir(parents=True,exist_ok=True)
        target_path.write_bytes((root/name).read_bytes().replace(b'\r\n',b'\n'))
        if name in pins['files']:assert sha256_file(target_path)==pins['files'][name],name
    def git(*args):return subprocess.check_output(['git','-C',str(project),*args],text=True,stderr=subprocess.STDOUT).strip()
    git('init');git('add','--','.');git('-c','commit.gpgsign=false','-c','user.name=D20Fixture','-c','user.email=d20-fixture@example.invalid','commit','--no-verify','-m','Isolated source validation fixture')
    commit=git('rev-parse','HEAD');git('update-ref','refs/remotes/origin/main',commit)
    weaver=Path(importlib.import_module('weaver').__file__).resolve().parent.parent
    code='''import sys
from pathlib import Path
sys.path[:0]=[str(Path(sys.argv[1])/'src'),sys.argv[3]]
from hlt_classification.luka_d20_full_hydra.campaign import source
assert source(sys.argv[1],sys.argv[2])['git_commit']==sys.argv[2]
path=Path(sys.argv[1])/'src/hlt_classification/luka_d20_full_hydra/kernel.py'
path.write_text(path.read_text()+'\\n# injected drift\\n')
try:source(sys.argv[1],sys.argv[2])
except ValueError:print('PASS: complete clean closure and stale source rejection')
else:raise AssertionError('Stale source accepted')
'''
    result=subprocess.run([sys.executable,'-I','-B','-c',code,str(project),commit,str(weaver)],capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'stale source rejection' in result.stdout


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Installed-Weaver CUDA device unavailable')
def test_installed_weaver_all_13_scientific_arms_and_asymmetric_gradients(tmp_path):
    from hlt_classification.luka_sixclass_balanced_oscar.model import installed_parity
    kernel.configure(1,3);tr=cache(rows=144);spec=dict(root=str(tmp_path),content_hash='a'*64,population=dict(content_hash='b'*64),maximum=4)
    assert installed_parity(tr,device='cuda')['passed']
    artifacts.prepare_starts(spec)
    class Targets:
        def batch(self,ix):return [target(1+int(i)%4) for i in ix]
    ctx=Context(0,1,torch.device('cuda'))
    for arm in c.ARMS:
        task=arm+'_s20';targets=Targets() if c.cell(task)['reconstruction'] else None
        for position in (2.,10.):assert resources.parity(spec,task,tr,targets,None,ctx,position)['ranks'][0]['passed']
    pt=np.repeat(np.arange(250,2600,100),6).astype(float)
    probe=resources.resume_probe(spec,'D20_s20',dict(train=tr,validation=tr),Targets(),None,pt,ctx,tmp_path)
    assert probe['ranks'][0]['exact']

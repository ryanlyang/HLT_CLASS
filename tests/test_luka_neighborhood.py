"""Physical targets, immutable banks, native parity and two-lane production tests."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys
import numpy as np
import pytest
import torch
from hlt_classification.cms2jc2_response.bridge import Particles,p4_from_coordinates
from hlt_classification.luka_fullsim import stage2
from hlt_classification.luka_fullsim.particles import RawJet
from hlt_classification.luka_fullsim.preflight_cache import _block
from hlt_classification.jetclass2_delphes.cache import RamCache
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer,installed_environment
from hlt_classification.jetclass2_delphes.runner import _forward
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.data.cache_contracts import load_json,with_content_hash,write_immutable_json,atomic_publish_bytes,deterministic_npz_bytes,load_npz_arrays
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_neighborhood import contracts as c,targets,banks,model,training,campaign,submission,worker,startup,source,retention
from hlt_classification.luka_supervision_depth import model as oldmodel,training as oldtraining
from test_luka_supervision_depth import bare as oldspec,fake_bank,science_fixture
from test_luka_fullsim_stage2 import dataset,prepared


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def particles(pt=(2.,3.,4.),eta=(0.,.04,.11),phi=(0.,0.,0.),category=(0,2,5)):
    cat=np.asarray(category,np.int8); n=len(cat)
    return Particles(p4_from_coordinates(pt,eta,phi,0.),np.isin(cat,(0,3,4)).astype(np.int8),cat,
        np.zeros((n,4)),np.zeros((n,4),bool),tuple(str(i) for i in range(n)))


def population(role='train'):
    rows=[]; paired=[]
    for i in range(22):
        h=particles(pt=(2.+i*.1,3.,4.)); o=particles(pt=(2.+i*.12,3.4,5.),eta=(.005,.04,.14))
        jet=RawJet(f'{i+(100 if role=="validation" else 0):064x}',i%11,0,i,None,None)
        rows.append((jet,stage2.build_inputs(h,capacity=512))); paired.append((h,o))
    return RamCache([_block(0,rows)],role=role,coordinate_name='D000',foundation_sha256='a'*64),paired


def bare(tmp_path):
    p=oldspec(tmp_path)
    s=dict(root=str(tmp_path/'neighborhood'),project_dir='/project',source_commit='a'*40,
        source_snapshot=dict(content_hash='b'*64),content_hash='c'*64,anchor=p['anchor'],reuse=p['reuse'],**campaign.policy(p,8))
    return s


def test_registration_and_failure_isolated_two_lane_plan(tmp_path):
    spec=bare(tmp_path); plan=submission.render(spec)
    assert len(c.ARMS)==14 and plan['jobs']==16 and plan['preflight_jobs']==plan['teacher_fits']==0
    assert list(c.ARMS)[:4]==['smooth_w3','smooth_w10','ann_w3','ann_w10']
    for i,(arm,row) in enumerate(zip(c.ARMS,plan['commands'][1:15])):
        r=c.recipe(arm)
        assert r['ce_coefficient']==.5 and r['embedding_total']==40 and r['logit_total']==1
        assert r['epochs']==30 and r['batch_size']==128 and not r['relation'] and not r['depths']
        assert row['dependencies']==['targets',*list(c.ARMS)[i%2:i:2]]
        dep=next(x for x in row['command'] if x.startswith('--dependency='))
        assert dep.startswith('--dependency=afterok:${JOB_targets}')
        assert '--partition=debug' in row['command'] and '--no-requeue' in row['command']
        if i>=2: assert ',afterany:' in dep
    assert plan['commands'][-1]['command'][13].startswith('--comment=') or 'report' in plan['commands'][-1]['task_id']
    assert not any('preflight' in r['task_id'] for r in plan['commands'])


def test_hand_calculated_counts_momentum_unknown_and_smooth():
    h=particles(pt=(10.,),eta=(0.,),phi=(0.,),category=(0,))
    o=particles(pt=(1.,2.,3.),eta=(0.,.07,.15),phi=(0.,0.,0.),category=(0,2,5))
    a=targets.descriptors(h,o,smooth=False)
    expected=np.zeros(36,np.float32)
    expected[[0,8,17]]=np.log(2); expected[[18,26,35]]=np.log1p([10,20,30])
    np.testing.assert_allclose(a[0],expected,rtol=1e-7)
    b=targets.descriptors(h,o,smooth=True)
    assert b[0,0]==pytest.approx(np.log(2))
    assert b[0,8]==pytest.approx(np.log1p(np.exp(-.5*(.07/.05)**2)),rel=1e-6)


def test_context_permutation_query_equivariance_and_chunking():
    h=particles(); o=particles(eta=(.02,-.03,.4))
    for smooth in (False,True):
        a=targets.descriptors(h,o,smooth=smooth)
        np.testing.assert_array_equal(a,targets.descriptors(h,o.take([2,0,1]),smooth=smooth,chunk=1))
        np.testing.assert_array_equal(a[[2,0,1]],targets.descriptors(h.take([2,0,1]),o,smooth=smooth,chunk=2))


def test_phi_wrap_and_unmatched_context():
    h=particles(pt=(2.,),eta=(0.,),phi=(np.pi-.01,),category=(0,))
    o=particles(pt=(1.,1.),eta=(0.,0.),phi=(-np.pi+.01,0.),category=(2,5))
    x=targets.descriptors(h,o,smooth=False)
    assert x[0,2]==pytest.approx(np.log(2)) and x[0,5]==0
    assert np.count_nonzero(x)==2


def test_equal_jet_moments_and_scale_floor():
    m=targets.Moments(); m.add(np.zeros((1,36),np.float32)); m.add(np.full((9,36),2.,np.float32))
    mean,scale=m.result(); np.testing.assert_array_equal(mean,np.ones(36)); np.testing.assert_array_equal(scale,np.ones(36))
    m=targets.Moments(); m.add(np.zeros((2,36))); assert np.all(m.result()[1]==.1)


def test_loss_equal_jet_padding_and_target_detached():
    mask=torch.tensor([[True,False],[True,True]])
    p=torch.zeros((2,2,36),requires_grad=True); t=torch.tensor([[[2.]*36,[float('nan')]*36],[[0.]*36,[0.]*36]],requires_grad=True)
    loss,_=model.neighborhood_loss(p,t,mask); assert float(loss)==.75
    loss.backward(); assert t.grad is None and torch.count_nonzero(p.grad[0,1])==0
    with pytest.raises(ValueError): model.neighborhood_loss(p,t,torch.zeros_like(mask))


def test_bank_roundtrip_control_correction_and_role_guards(tmp_path):
    tr,rows=population(); root=tmp_path/'bank'; m=banks.build(root,tr,rows,source_hash='b'*64,available_bytes=8*2**30)
    bank=banks.NeighborhoodBank(root,tr,'b'*64)
    for kind in (*c.FAMILIES,'correction'):
        a=bank.batch(np.array([0,2],np.int64),16,kind)
        assert a.shape==(2,16,36) and not a[:,3:].any()
    expected=(bank.payload['ann'][:3]-bank.payload['hlt_ann'][:3])/bank.index['ann_scale']
    np.testing.assert_allclose(bank.batch(np.array([0],np.int64),3,'correction')[0],expected,rtol=1e-6,atol=1e-7)
    with pytest.raises(ValueError): bank.batch(np.array([0],np.int64),2,'ann')
    bank.close()
    with pytest.raises((ValueError,PermissionError)): banks.NeighborhoodBank(root,population('validation')[0],'b'*64)
    with pytest.raises(ValueError): banks.NeighborhoodBank(root,tr,'c'*64)
    with (root/'ann.npy').open('r+b') as f: f.seek(-1,2); f.write(b'x')
    with pytest.raises(ValueError): banks.NeighborhoodBank(root,tr,'b'*64)
    with pytest.raises(PermissionError): banks.build(tmp_path/'val',population('validation')[0],rows,source_hash='b'*64,available_bytes=8*2**30)
    with pytest.raises(OSError): banks.build(tmp_path/'quota',tr,rows,source_hash='b'*64,available_bytes=1)


@pytest.mark.parametrize('key',['role','recipe','alignment','width','tokens'])
def test_rehashed_bank_metadata_rejected(tmp_path,key):
    tr,rows=population(); root=tmp_path/'bank'; m=banks.build(root,tr,rows,source_hash='b'*64,available_bytes=8*2**30)
    m[key]=None; m=with_content_hash(m)
    with pytest.raises((ValueError,TypeError)): banks.validate_index(m,load_npz_arrays(root/'index.npz'),tr,'b'*64)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_native_parity_and_rng(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    spec=bare(tmp_path); node=spec['nodes']['smooth_w3']; tr,_=population(); raw=tr.batch(np.arange(4))
    old=oldmodel.new_model(dict(node,arm='baseline')).to(device).eval(); old.capture=True
    rng=torch.random.get_rng_state(); new=model.new_model(node).to(device).eval(); new.capture=True
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(torch.equal(v,new.native_state()[k]) for k,v in old.native_state().items())
    with parity_backend(device):
        for bf16 in (False,True):
            torch.testing.assert_close(_forward(old,raw,device,bf16=bf16),_forward(new,raw,device,bf16=bf16),rtol=0,atol=0)
    native=DelphesParticleTransformer(); native.load_state_dict(new.native_state(),strict=True)


@pytest.mark.parametrize('arm',list(c.ARMS))
def test_all_objectives_ce_only_main_and_connected_heads(tmp_path,arm):
    spec=bare(tmp_path); tr,_=population(); bank=fake_bank(); bank.identities=tr.identities; bank.labels=tr.labels
    bank.neighborhood=SimpleNamespace(batch=lambda ix,width,kind:np.ones((len(ix),width,36),np.float32))
    student=model.new_model(spec['nodes'][arm]).train(); student.capture=True
    loss,terms,weights=training.batch_loss(student,tr,bank,np.arange(4),'cpu')
    head=tuple(student.mod.fc.parameters()); expected=torch.autograd.grad(.5*terms['ce'],head,retain_graph=True)
    loss.backward()
    for p,g in zip(head,expected): torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
    for name in ('predictors','logit_heads','neighborhood_head'):
        assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in getattr(student,name).parameters())
    assert weights==training.expected_weights(arm)


def test_complete30_zero_neighborhood_parity(tmp_path,monkeypatch):
    tr,_=population(); va,_=population('validation'); spec=bare(tmp_path); node=spec['nodes']['smooth_w3']; bank=fake_bank()
    bank.identities=tr.identities; bank.labels=tr.labels
    device='cuda' if torch.cuda.is_available() else 'cpu'
    with parity_backend(device):
        _,old_report,old_state=oldtraining.train(tr,va,node=dict(node,arm='baseline'),bank=bank,device=device)
        make=model.new_model
        def zero(n):
            student=make(n); student.registered['neighborhood_weight']=0.; return student
        monkeypatch.setattr(training,'new_model',zero)
        _,report,state=training.train(tr,va,node=node,bank=bank,device=device)
    assert report['selected_pass']==old_report['selected_pass']
    for a,b in zip(report['validation_history'],old_report['validation_history']):
        for k in ('validation','losses','weights','learning_rate','update'): assert a[k]==b[k]
    assert all(torch.equal(v,state[k]) for k,v in old_state.items())


def test_source_inventory_cli_and_shell():
    root=Path(__file__).resolve().parents[1]
    assert all((root/p).is_file() for p in source.REQUIRED)
    subprocess.run([sys.executable,'scripts/luka_neighborhood.py','--help'],check=True,capture_output=True)
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    if bash.exists():
        for name in ('scripts/queue_luka_neighborhood.sh','sbatch/run_luka_neighborhood.sh'):
            subprocess.run([str(bash),'-n',name],check=True,capture_output=True)


def test_production_workers_all14_native_results_and_retirement(tmp_path,monkeypatch):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    from hlt_classification.luka_supervision_depth import worker as donor_worker
    original,tr,va=science_fixture(tmp_path,monkeypatch)
    for task in ('targets','baseline'): donor_worker.run(original,task,device='cuda')
    oldroot=Path(original['root']); root=tmp_path/'neighborhood'; root.mkdir()
    refs={n:file_ref(oldroot/'targets/references'/(n+'.npz')) for n in c.REFERENCES if n!='BASELINE_REPEAT'}
    refs['BASELINE_REPEAT']=file_ref(oldroot/'baseline/fit/validation_predictions.npz')
    imported=dict(references=refs,subset=file_ref(oldroot/'targets/references/original50k.npz'),
        baseline_report=file_ref(oldroot/'baseline/fit/training_report.json'),
        banks={t:file_ref(oldroot/'targets/banks'/t/'manifest.json') for t in c.TEACHERS})
    spec=dict(root=str(root),project_dir='/project',source_commit='a'*40,source_snapshot=dict(content_hash='b'*64),
        anchor=original['anchor'],reuse=original['reuse'],imported=imported,**campaign.policy(dict(original,counts=c.COUNTS),8))
    spec['counts']=original['counts']; spec['original50k_rows']=22
    spec=with_content_hash(spec); write_immutable_json(root/'campaign_spec.json',spec)
    jobs={row['task_id']:str(900+i) for i,row in enumerate(spec['tasks'])}
    monkeypatch.setattr(campaign,'check',lambda _:None); monkeypatch.setattr(campaign,'validate_spec',lambda *a,**kw:None)
    monkeypatch.setattr(source,'current',lambda *a:spec['source_snapshot'])
    monkeypatch.setattr(worker,'execution',lambda _,t:jobs[t]); monkeypatch.setattr(submission,'accepted_jobs',lambda _:jobs)
    monkeypatch.setattr(submission,'successful',lambda *a:None); monkeypatch.setattr(worker,'storage',lambda _:0)
    mem=lambda _:dict(rss_bytes=2**30,peak_rss_bytes=2**30,gpu_peak_bytes=2**30)
    monkeypatch.setattr(worker,'memory_snapshot',mem); monkeypatch.setattr(startup,'memory_snapshot',mem)
    def raw_rows(s,cache,meta):
        for n in banks.counts(cache):
            p=particles(pt=np.ones(n),eta=np.arange(n)*.02,phi=np.zeros(n),category=np.full(n,5))
            yield p,p
    monkeypatch.setattr(banks,'physical_rows',raw_rows)
    for row in spec['tasks']: worker.run(spec,row['task_id'],device='cuda')
    result=worker.results(spec)
    assert not result['pending'] and len(result['rows'])==len(c.REFERENCES)+14
    for task in c.ARMS:
        folder=root/task/'fit'; report=load_json(folder/'training_report.json')
        assert report['passes']==30
        assert report['validation_history'][0]['weights']==training.expected_weights(task)
        assert len(report['validation_history'][0]['neighborhood_channel_errors'])==36
        native=DelphesParticleTransformer(); native.load_state_dict(torch.load(folder/'selected.pt',weights_only=True,map_location='cpu'),strict=True)
    changed=deepcopy(report); changed['validation_history'][0]['neighborhood_channel_errors'][0]+=1
    with pytest.raises(ValueError,match='diagnostics'):
        training.validate_report(with_content_hash(changed),spec,task,report['parents']['bank'])
    changed=deepcopy(report); changed['validation_history'][0]['delta_repeat']+=.1
    with pytest.raises(ValueError,match='comparison'):
        training.validate_report(with_content_hash(changed),spec,task,report['parents']['bank'])
    assert load_json(root/'report/results.json')==result
    monkeypatch.setattr(retention.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=''))
    review=retention.plan(spec)
    with pytest.raises(PermissionError): retention.retire(spec,plan_hash='0'*64,authorization=retention.AUTHORIZATION)
    retention.retire(spec,plan_hash=review['content_hash'],authorization=retention.AUTHORIZATION)
    assert result==worker.results(spec)
    assert all(not (root/p).exists() for p in retention.names())
    assert all((oldroot/'targets/banks'/t/'targets.npz').is_file() for t in c.TEACHERS)
    with pytest.raises(ValueError): worker.run(spec,'ann_w3',device='cuda')


def test_submit_requires_review_and_rejects_existing_outputs(tmp_path,monkeypatch):
    spec=bare(tmp_path); root=Path(spec['root']); root.mkdir()
    monkeypatch.setattr(campaign,'check',lambda _:None); monkeypatch.setattr(submission,'check_site',lambda _:None)
    def submit(**kwargs):
        write_immutable_json(kwargs['output'],dict(dry_run=not kwargs['execute']))
        return dict(accepted=True)
    monkeypatch.setattr(submission,'submit_exact_dag',submit)
    with pytest.raises(PermissionError): submission.submit(spec,execute=True,authorization=c.AUTHORIZATION)
    plan=submission.submit(spec)
    with pytest.raises(PermissionError): submission.submit(spec,execute=True,plan_hash='0'*64,authorization=c.AUTHORIZATION)
    assert submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)==dict(accepted=True)
    with pytest.raises(FileExistsError): submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    (root/'smooth_w3').mkdir()
    with pytest.raises(FileExistsError): submission.submit(spec)


def test_partial_bank_never_publishes_manifest(tmp_path):
    tr,rows=population(); root=tmp_path/'partial'
    with pytest.raises(ValueError): banks.build(root,tr,rows[:-1],source_hash='b'*64,available_bytes=8*2**30)
    assert not (root/'manifest.json').exists()
    with pytest.raises(FileExistsError): banks.build(root,tr,rows,source_hash='b'*64,available_bytes=8*2**30)


def test_correction_zero_for_identical_views(tmp_path):
    tr,rows=population(); root=tmp_path/'bank'
    banks.build(root,tr,[(h,h) for h,_ in rows],source_hash='b'*64,available_bytes=8*2**30)
    bank=banks.NeighborhoodBank(root,tr,'b'*64)
    assert not bank.batch(np.arange(len(tr)),16,'correction').any()
    bank.close()


def test_selected_root_train_only_native_input_replay(dataset,prepared,monkeypatch):
    import uproot
    container,inventory,splits=dataset; _,p,*_=prepared
    spec={'anchor':{'evidence':dict(raw_container=str(container),cache='b'*64)}}
    rows=[]
    for member in splits['memberships']['train']:
        for jet in banks.regular_cache.selected_file_rows(container,inventory,member):
            rows.append((jet,stage2.build_inputs(banks.physical(jet.hlt,p['conventions'],'hlt'),capacity=512)))
    cache=RamCache([_block(0,rows)],role='train',coordinate_name='D000',foundation_sha256=p['content_hash'])
    meta=(dict(content_hash='b'*64),p,inventory,splits)
    allowed={inventory['files'][m['file_index']]['path']:set(sum(m['entries_by_class'],[]))
             for m in splits['memberships']['train']}
    original=uproot.behaviors.TBranch.HasBranches.arrays; calls=[]
    def guarded(tree,expressions,*args,**kwargs):
        path=Path(tree.file.file_path).relative_to(container/'raw').as_posix()
        assert path in allowed and set(range(kwargs['entry_start'],kwargs['entry_stop']))<=allowed[path]
        calls.append(path); return original(tree,expressions,*args,**kwargs)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches,'arrays',guarded)
    values=list(banks.physical_rows(spec,cache,meta))
    assert len(values)==66 and calls and all(len(h)==2 and len(o)==3 for h,o in values)
    assert all(targets.all_descriptors(h,o)['ann'].shape==(2,36) for h,o in values)
    cache.blocks[0].features[0,0]+=1
    with pytest.raises(ValueError,match='fingerprint'): list(banks.physical_rows(spec,cache,meta))
    cache.role='validation'
    with pytest.raises(PermissionError): list(banks.physical_rows(spec,cache,meta))


def test_annular_boundaries_and_common_axis(monkeypatch):
    # Exact geometry values isolate the half-open kernel from p4 roundoff.
    h=particles(pt=(1.,),eta=(0.,),phi=(0.,),category=(0,)); o=particles(category=(0,0,0))
    original=targets.geometry
    def exact(p):
        if p is h: return np.array([1.]),np.array([0.]),np.array([0.])
        return np.ones(3),np.array([.05,.10,.20]),np.zeros(3)
    monkeypatch.setattr(targets,'geometry',exact)
    value=targets.descriptors(h,o,smooth=False)
    assert value[0,0]==0 and value[0,6]==pytest.approx(np.log(2)) and value[0,12]==pytest.approx(np.log(2))
    monkeypatch.setattr(targets,'geometry',original)
    o=particles()
    base=targets.descriptors(h,o,smooth=True)
    shifted_h=particles(pt=(1.,),eta=(.7,),phi=(2.,),category=(0,))
    shifted_o=particles(eta=(.7,.74,.81),phi=(2.,2.,2.))
    np.testing.assert_allclose(base,targets.descriptors(shifted_h,shifted_o,smooth=True),rtol=2e-6,atol=1e-7)


def test_startup_projection_uses_population_timing_not_repeated_longest():
    tr=[.1]*8; va=[.02]*8
    expected=2*(30*np.ceil(299234/128)*.1+33*np.ceil(74895/128)*.02)+1200
    assert startup.projection(299234,74895,tr,va)==expected


def test_source_scope_preserves_existing_runtime(monkeypatch,tmp_path):
    project=Path(source.__file__).resolve().parents[3]
    monkeypatch.setattr(source,'_source',lambda *a:{'content_hash':'a'*64})
    def git(args,**kwargs):
        if 'ls-files' in args: return SimpleNamespace(stdout='\n'.join(source.REQUIRED))
        return SimpleNamespace(stdout='src/hlt_classification/luka_supervision_depth/training.py\n')
    monkeypatch.setattr(source.subprocess,'run',git)
    with pytest.raises(ValueError,match='runtime changed'): source.current(project,'b'*40)


def test_retirement_rejects_partial_completion_and_live_jobs(tmp_path,monkeypatch):
    spec=bare(tmp_path); root=Path(spec['root']); root.mkdir()
    monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(worker,'results',lambda _:dict(pending=['smooth_w3']))
    with pytest.raises(ValueError,match='All rows'): retention.plan(spec)
    (root/'report').mkdir(); write_immutable_json(root/'report/results.json',dict(pending=[]))
    monkeypatch.setattr(worker,'results',lambda _:dict(pending=[]))
    monkeypatch.setattr(worker,'validate_task',lambda *a:dict(job_id='900'))
    monkeypatch.setattr(submission,'successful',lambda *a:None)
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:dict(report='900'))
    monkeypatch.setattr(retention.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='900\n'))
    with pytest.raises(ValueError,match='still active'): retention.plan(spec)
    (root/'retirement').mkdir()
    with pytest.raises(ValueError,match='Partial retirement'): retention.retired_evidence(spec)


@pytest.mark.parametrize('field',['projected_seconds','train_samples','timing_seed','updates','native_parity'])
def test_rehashed_startup_tampering_rejected(tmp_path,field):
    spec=bare(tmp_path); tr=[.1]*8; va=[.02]*8
    value=c.artifact('STARTUP',parents=dict(spec=spec['content_hash'],bundle='d'*64),task='smooth_w3',
        passed=True,native_parity=True,main_head_CE_only=True,discarded=True,head_connected=True,
        batch_size=128,updates=9,maximum_train=3,maximum_validation=3,step_seconds=.1,validation_seconds=.01,
        projected_seconds=startup.projection(c.COUNTS['train'],c.COUNTS['validation'],tr,va),
        train_samples=tr,validation_samples=va,elapsed_seconds=1.,timing_seed=94272,projection_factor=2,publication_reserve_seconds=1200)
    startup.check(value,spec,'smooth_w3','d'*64)
    value[field]=dict(projected_seconds=1.,train_samples=[.1]*7,timing_seed=1,updates=1,native_parity=False)[field]
    with pytest.raises(ValueError): startup.check(with_content_hash(value),spec,'smooth_w3','d'*64)


def test_allocation_identity_and_walltime_are_exact(tmp_path,monkeypatch):
    spec=bare(tmp_path); task='smooth_w3'; monkeypatch.setenv('SLURM_JOB_ID','123')
    fields=dict(JobName='lukanb_'+task,Comment='lukanb:'+spec['content_hash']+':'+task,
                WorkDir=spec['project_dir'],TimeLimit='12:00:00')
    monkeypatch.setattr(worker.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=' '.join(k+'='+v for k,v in fields.items())))
    monkeypatch.setattr(worker,'gpu_execution',lambda *a:'123')
    assert worker.execution(spec,task)=='123'
    for key in fields:
        saved=fields[key]; fields[key]='different'
        with pytest.raises(PermissionError): worker.execution(spec,task)
        fields[key]=saved

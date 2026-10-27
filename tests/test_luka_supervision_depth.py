"""Scientific invariants, exact DAG, native parity and production miniature."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from test_luka_u100_pretrain import cache
from test_luka_trial_hybrid import bare_spec as hybrid_spec
from hlt_classification.data.cache_contracts import load_json,load_npz_arrays,write_immutable_json,with_content_hash,atomic_publish_bytes,deterministic_npz_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer,installed_environment
from hlt_classification.jetclass2_delphes.runner import _forward
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_auxiliary_grid import model as grid_model,training as grid_training
from hlt_classification.luka_trial_hybrid import contracts as old
from hlt_classification.luka_supervision_depth import contracts as c,model,training,banks,campaign,submission,source,worker,startup,data


@pytest.fixture(autouse=True)
def threads():
    n=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def parent(tmp_path):
    p=hybrid_spec(Path('/parent')); report=tmp_path/'original_offline_report.json'
    write_immutable_json(report,dict(node=dict(initialization_seed=23,sampler_seed=29,coordinate='OFFLINE',node_id='OFFLINE')))
    p['anchor']['evidence']['models']['OFFLINE']=dict(report=file_ref(report))
    return p


def bare(tmp_path):
    p=parent(tmp_path)
    return dict(root=str(tmp_path/'new'),project_dir='/project',source_commit='a'*40,source_snapshot=dict(content_hash='b'*64),
        content_hash='c'*64,anchor=p['anchor'],reuse=dict(acceptance=p['acceptance'],teachers={}),available_quota_gib=8,**campaign.policy(p))


def fake_bank(n=22,depths=()):
    rng=np.random.default_rng(42)
    class Tokens:
        def batch(self,ix,width): return {d:np.zeros((len(ix),width,128),np.float32) for d in depths}
    return SimpleNamespace(probabilities={t:np.full((n,11),1/11,np.float32) for t in c.TEACHERS},
        targets={t:rng.normal(size=(n,128)).astype(np.float32) for t in (*c.TEACHERS,*c.TEACHER_TASKS)},tokens=Tokens(),
        manifest=c.artifact('BUNDLE',parents={}),identities=cache().identities,labels=cache().labels)


def test_exact_twenty_recipes_and_ordered_debug_plan(tmp_path):
    spec=bare(tmp_path); plan=submission.render(spec)
    assert len(c.ARMS)==20 and len(plan['commands'])==25 and plan['preflight_jobs']==0 and plan['teacher_fit_jobs']==2
    assert spec['counts']==dict(train=299234,validation=74895)
    previous=None; fitted=[]
    for row in plan['commands']:
        assert '--partition=debug' in row['command'] and '--no-requeue' in row['command']
        assert row['dependencies']==([] if previous is None else [previous])
        previous=row['task_id']
        if previous in c.ARMS: fitted.append(previous)
    assert fitted==list(c.ARMS)
    for arm in c.ARMS:
        r=c.recipe(arm)
        assert r['ce_coefficient']==.5 and sum(r['logit_weights'].values())==1 and sum(r['embedding_weights'].values())==40
        assert r['epochs']==30 and r['batch_size']==128 and not r['relation']
        assert sum(v for k,v in training.expected_weights(arm).items() if k.startswith('token_'))==pytest.approx(r['token_weight'])
    assert c.recipe('duplicate_offline')['target_names']['OFFLINE_ALT']=='OFFLINE'
    assert c.recipe('supcon_low')['embedding_weights']['OFFLINE']==8
    assert c.recipe('supcon_replace')['target_names']['OFFLINE']=='teacher_supcon'
    assert c.recipe('supcon_replace')['embedding_seeds']==c.recipe('baseline')['embedding_seeds']
    assert spec['nodes']['baseline_seed1']['initialization_seed']-spec['nodes']['baseline']['initialization_seed']==100000
    a,b=(spec['nodes'][t] for t in c.TEACHER_TASKS)
    assert a['initialization_seed']==b['initialization_seed'] and a['sampler_seed']==b['sampler_seed']


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_native_parity_all_heads_rng_and_baseline_gradients(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    spec=bare(tmp_path); raw=cache().batch(np.arange(4)); node=spec['nodes']['baseline']
    oldnode=dict(node,arm='k1_e40_plain'); oldmodel=grid_model.new_model(oldnode).to(device).eval(); oldmodel.capture=True
    expected_rng=torch.random.get_rng_state()
    expected_native=oldmodel.native_state()
    for arm in c.ARMS:
        student=model.new_model(spec['nodes'][arm]).to(device).eval(); student.capture=True
        if c.ARMS[arm]['seed']==0:
            assert torch.equal(torch.random.get_rng_state(),expected_rng)
            assert all(torch.equal(v,expected_native[k]) for k,v in student.native_state().items())
            with parity_backend(device):
                for bf16 in (False,True):
                    torch.testing.assert_close(_forward(student,raw,device,bf16=bf16),_forward(oldmodel,raw,device,bf16=bf16),rtol=0,atol=0)
        native=DelphesParticleTransformer().to(device); native.load_state_dict(student.native_state(),strict=True)
        assert set(native.state_dict())==set(student.native_state())
        del student,native
    student=model.new_model(node).to(device).eval(); student.capture=True
    bank=fake_bank(); ix=np.arange(4,dtype=np.int64)
    with parity_backend(device):
        first,_,_=training.batch_loss(student,cache(),bank,ix,device)
        second,*_=grid_training.batch_loss(oldmodel,raw,bank,ix,'k1_e40_plain',device)
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        first.backward(); second.backward()
        for (n,p),(n2,q) in zip(student.named_parameters(),oldmodel.named_parameters()):
            assert n==n2
            if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)


@pytest.mark.parametrize('arm',list(c.ARMS))
def test_every_objective_keeps_main_head_ce_only_and_aux_connected(tmp_path,arm):
    spec=bare(tmp_path); student=model.new_model(spec['nodes'][arm]); student.capture=True; student.eval()
    bank=fake_bank(depths=c.ARMS[arm]['depths']); ix=np.arange(4,dtype=np.int64)
    loss,terms,weights=training.batch_loss(student,cache(),bank,ix,'cpu')
    assert weights==training.expected_weights(arm)
    expected=torch.autograd.grad(.5*terms['ce'],tuple(student.mod.fc.parameters()),retain_graph=True)
    loss.backward(); training.finite_gradients(student)
    for p,g in zip(student.mod.fc.parameters(),expected): torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
    assert all(p.grad is not None for p in student.predictors.parameters())
    assert all(p.grad is not None for p in student.logit_heads.parameters())
    assert all(p.grad is not None for p in student.token_heads.parameters())


def test_shared_hungarian_permutation_padding_and_depth_reduction():
    p={d:torch.randn(2,3,128,requires_grad=True) for d in (2,4,8)}
    mask=torch.tensor([[True,True,False],[True,True,True]])
    targets={d:p[d].detach().clone()[:,[1,0,2]].requires_grad_() for d in p}
    for d in p:
        with torch.no_grad(): p[d][0,2]=float('nan'); targets[d][0,2]=float('nan')
    value,parts=model.token_loss(p,targets,mask,'set'); assert value==0
    value.backward(); assert all(t.grad is None for t in targets.values())
    # Contradictory assignments across two depths cannot both be satisfied.
    a=torch.stack([torch.zeros(128),torch.ones(128)])[None]
    value,_=model.token_loss({4:a,8:a},{4:a,8:a[:,[1,0]]},torch.ones(1,2,dtype=torch.bool),'set')
    assert value==.25
    aligned,_=model.token_loss({4:a},{4:a[:,[1,0]]},torch.ones(1,2,dtype=torch.bool),'aligned')
    assert aligned==.5
    with pytest.raises(ValueError): model.token_loss({4:a},{4:a},torch.zeros(1,2,dtype=torch.bool),'set')


def test_supcon_reference_positives_and_singletons():
    z=torch.randn(5,128,requires_grad=True); labels=torch.tensor([0,0,1,1,2])
    normalized=F.normalize(z,dim=-1); scores=normalized@normalized.T/.1
    expected=[]
    for i in range(4):
        positive=1-i if i<2 else 5-i
        denominator=torch.logsumexp(scores[i,torch.arange(5)!=i],0)
        expected.append(denominator-scores[i,positive])
    actual=model.supcon(z,labels); torch.testing.assert_close(actual,torch.stack(expected).mean())
    actual.backward(); assert torch.isfinite(z.grad).all()
    singleton=torch.randn(1,128,requires_grad=True); value=model.supcon(singleton,torch.tensor([0])); value.backward()
    assert value==0 and torch.count_nonzero(singleton.grad)==0


def science_fixture(tmp_path,monkeypatch):
    spec=bare(tmp_path); root=Path(spec['root']); root.mkdir(); env=installed_environment()
    spec['counts']=dict(train=22,validation=22); spec['original50k_rows']=22
    spec['anchor']['evidence']['environment']=env; spec['reuse']['acceptance']['preflight']['environment']=env
    tr={co:cache('train',co) for co in ('OFFLINE','U100','D066','D033','D000')}
    va={co:cache('validation',co) for co in tr}
    for i,t in enumerate(c.TEACHERS):
        folder=tmp_path/t; folder.mkdir(); torch.manual_seed(100+i); native=DelphesParticleTransformer()
        save_state(folder/'selected.pt',native.state_dict()); write_immutable_json(folder/'report.json',dict(fixture=True))
        refs=dict(checkpoint=file_ref(folder/'selected.pt'),report=file_ref(folder/'report.json'))
        spec['reuse']['teachers'][t]=old.artifact('TEACHER',parents=dict(spec='d'*64,prepared=tr['D000'].foundation_sha256,**{k:r['sha256'] for k,r in refs.items()}),
            scope='shared' if t=='OFFLINE' else 'c50',teacher=t,coordinate=dict(OFFLINE='OFFLINE',UE100='U100',DE066='D066',DE033='D033')[t],files=refs)
    spec=with_content_hash(spec); write_immutable_json(root/'campaign_spec.json',spec)
    jobs={row['task_id']:str(100+i) for i,row in enumerate(spec['tasks'])}
    monkeypatch.setattr(campaign,'check',lambda _:None); monkeypatch.setattr(campaign,'validate_spec',lambda *a,**k:None)
    monkeypatch.setattr(source,'current',lambda *a:spec['source_snapshot']); monkeypatch.setattr(worker,'execution',lambda _,t:jobs[t])
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:jobs); monkeypatch.setattr(submission,'successful',lambda *a:None)
    monkeypatch.setattr(worker,'storage',lambda _:0); monkeypatch.setattr(worker.regular_cache,'metadata',lambda _:None)
    monkeypatch.setattr(data,'load',lambda _,r,co,meta:(tr if r=='train' else va)[co])
    scratch=tmp_path/'scratch'; scratch.mkdir(); monkeypatch.setattr(worker,'scratch_directory',lambda _:scratch)
    mem=lambda _:dict(rss_bytes=2**30,peak_rss_bytes=2**30,gpu_peak_bytes=2**30)
    monkeypatch.setattr(worker,'memory_snapshot',mem); monkeypatch.setattr(startup,'memory_snapshot',mem)
    def references(s,val):
        folder=Path(s['root'])/'targets/references'; folder.mkdir()
        for name in c.REFERENCES:
            p=np.full((len(val),11),1/11,np.float32)
            if name=='OFFLINE': p[:,0]+=.02; p/=p.sum(-1,keepdims=True)
            atomic_publish_bytes(folder/(name+'.npz'),deterministic_npz_bytes(dict(identities=val.identities,labels=val.labels,probabilities=p)))
        atomic_publish_bytes(folder/'original50k.npz',deterministic_npz_bytes(dict(indices=np.arange(len(val),dtype=np.int64))))
        write_immutable_json(folder.parent/'baseline_report.json',dict(fixture=True))
    monkeypatch.setattr(worker,'snapshot_references',references)
    return spec,tr,va


def test_native_multidepth_bank_lineage_padding_and_corruption(tmp_path,monkeypatch):
    spec,tr,_=science_fixture(tmp_path,monkeypatch); teacher=banks.record(spec,'UE100')
    native=worker.native(teacher['files']['checkpoint'],'cpu'); root=tmp_path/'tokens'
    m=banks.export_tokens(root,native,tr['U100'],tr['D000'],teacher,[2,4,8],'cpu')
    bank=banks.TokenBank(root,tr['D000'],teacher,[2,4,8])
    values=bank.batch(np.arange(22),5); assert all(v.shape==(22,5,128) for v in values.values())
    for d in (2,4,8):
        real=(bank.payload[d].astype(np.float64)-bank.index[f'mean{d}'])/bank.index[f'scale{d}']
        np.testing.assert_allclose(real.mean(0),0,atol=1e-6)
    with pytest.raises(ValueError): banks.TokenBank(root,tr['D000'],teacher,[4,8])
    bank.close()
    with (root/'tokens4.npy').open('r+b') as handle:
        handle.seek(-1,2); oldbyte=handle.read(1); handle.seek(-1,2); handle.write(bytes([oldbyte[0]^1]))
    with pytest.raises(ValueError): banks.TokenBank(root,tr['D000'],teacher,[2,4,8])


def test_all_production_workers_fixed30_native_readback_and_retirement(tmp_path,monkeypatch):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    spec,tr,va=science_fixture(tmp_path,monkeypatch)
    for row in spec['tasks']:
        worker.run(spec,row['task_id'],device='cuda')
    result=worker.results(spec)
    assert not result['pending'] and len(result['rows'])==28
    assert not list((tmp_path/'scratch').iterdir())
    assert all(not (Path(spec['root'])/name).exists() for name in worker.target_payloads())
    for task in (*c.TEACHER_TASKS,*c.ARMS):
        folder=Path(spec['root'])/task/'fit'; report=load_json(folder/'training_report.json')
        assert report['passes']==30 and report['selected_state']=='native_backbone_only'
        native=DelphesParticleTransformer(); native.load_state_dict(torch.load(folder/'selected.pt',map_location='cpu',weights_only=True),strict=True)
    assert result==load_json(Path(spec['root'])/'report/results.json')


def test_no_positive_teacher_and_poor_student_still_finish30(tmp_path,monkeypatch):
    spec=bare(tmp_path); tr=cache(); val=cache('validation','D000'); bank=fake_bank()
    spec['counts']=dict(train=22,validation=22)
    original=training.evaluate_probabilities
    def weak(labels,p):
        r=original(labels,p); r['macro_ovr_auc']=.001; return r
    monkeypatch.setattr(training,'evaluate_probabilities',weak)
    _,report,_=training.train(tr,val,node=spec['nodes']['baseline'],bank=bank,device='cpu')
    training.validate_report(report,spec,'baseline',bank.manifest['content_hash']); assert report['passes']==30
    broken=deepcopy(report); broken['validation_history'][3]['weights']['ce']=1.; broken=with_content_hash(broken)
    with pytest.raises(ValueError): training.validate_report(broken,spec,'baseline',bank.manifest['content_hash'])


def test_cli_source_inventory_and_shell_syntax():
    root=Path(__file__).resolve().parents[1]
    assert all((root/p).is_file() for p in source.REQUIRED)
    subprocess.run([sys.executable,'scripts/luka_supervision_depth.py','--help'],check=True,capture_output=True)
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    if bash.exists():
        for name in ('scripts/queue_luka_supervision_depth.sh','sbatch/run_luka_supervision_depth.sh'):
            subprocess.run([str(bash),'-n',name],check=True,capture_output=True)


def test_submit_needs_full_dry_hash_and_no_duplicate_claim(tmp_path,monkeypatch):
    spec=bare(tmp_path); Path(spec['root']).mkdir()
    monkeypatch.setattr(campaign,'check',lambda _:None); monkeypatch.setattr(submission,'check_site',lambda _:None)
    calls=[]
    def submit(**kwargs):
        calls.append(kwargs['execute'])
        if not kwargs['execute']: write_immutable_json(kwargs['output'],dict(dry=True))
        return dict(ok=True)
    monkeypatch.setattr(submission,'submit_exact_dag',submit)
    with pytest.raises(PermissionError): submission.submit(spec,execute=True,plan_hash='a'*64,authorization=c.AUTHORIZATION)
    plan=submission.submit(spec)
    with pytest.raises(PermissionError): submission.submit(spec,execute=True,plan_hash='a'*64,authorization=c.AUTHORIZATION)
    submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    with pytest.raises(FileExistsError): submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert calls==[False,True]


@pytest.mark.parametrize('field',['teacher','depths','alignment','role','rows','width','dtype'])
def test_rehashed_token_metadata_semantics_rejected(tmp_path,monkeypatch,field):
    spec,tr,_=science_fixture(tmp_path,monkeypatch); teacher=banks.record(spec,'UE100')
    n=worker.native(teacher['files']['checkpoint'],'cpu'); folder=tmp_path/'token'
    m=banks.export_tokens(folder,n,tr['U100'],tr['D000'],teacher,[4,8],'cpu')
    a=load_npz_arrays(folder/'index.npz')
    bad=deepcopy(m)
    bad[field]={'teacher':{},'depths':[8],'alignment':{},'role':'validation','rows':23,'width':64,'dtype':'float16'}[field]
    bad=with_content_hash(bad)
    with pytest.raises((ValueError,KeyError)):
        banks.validate_token_index(bad,a,tr['D000'],teacher,[4,8])


@pytest.mark.parametrize('fault',['identities','offsets','mean4','scale8'])
def test_token_index_rejects_wrong_rows_counts_and_statistics(tmp_path,monkeypatch,fault):
    spec,tr,_=science_fixture(tmp_path,monkeypatch); teacher=banks.record(spec,'UE100')
    n=worker.native(teacher['files']['checkpoint'],'cpu'); folder=tmp_path/'token'
    m=banks.export_tokens(folder,n,tr['U100'],tr['D000'],teacher,[4,8],'cpu'); a=load_npz_arrays(folder/'index.npz')
    if fault=='identities': a[fault]=a[fault][::-1].copy()
    elif fault=='offsets': a[fault][1]+=1
    elif fault=='mean4': a[fault][0]=np.nan
    else: a[fault][0]=0.
    with pytest.raises(ValueError): banks.validate_token_index(m,a,tr['D000'],teacher,[4,8])


def test_source_compatibility_exception_is_exact_and_not_trial_runtime():
    root=Path(__file__).resolve().parents[1]
    changed=subprocess.run(['git','diff','--name-only','--diff-filter=MDRT','028c3314',source.BASE,'--','src'],
        check=True,capture_output=True,text=True).stdout.splitlines()
    assert set(changed)==source.ISOLATED_FULLSTAT
    assert all('/luka_fullstat_splithead/' in name for name in changed)
    modules=('luka_supervision_depth','luka_auxiliary_grid','luka_trial_hybrid','luka_particle_prediction','luka_joint_prediction')
    for module in modules:
        for path in (root/'src/hlt_classification'/module).glob('*.py'):
            if path.name!='source.py': assert 'luka_fullstat_splithead' not in path.read_text()


@pytest.mark.parametrize('role',['test','final_test'])
def test_data_forbids_test(role):
    with pytest.raises(PermissionError): data.load({},role,'OFFLINE',None)


def test_failed_startup_never_publishes_target_success(tmp_path,monkeypatch):
    spec,_,_=science_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(startup,'run',lambda *a,**kw:(_ for _ in ()).throw(MemoryError('injected')))
    with pytest.raises(MemoryError): worker.run(spec,'targets',device='cpu')
    assert not (Path(spec['root'])/'targets/receipt.json').exists()
    assert all(not (Path(spec['root'])/arm).exists() for arm in c.ARMS)


def test_cleanup_refuses_incomplete_campaign_without_deletion(tmp_path,monkeypatch):
    spec=bare(tmp_path); root=Path(spec['root']); (root/'cleanup').mkdir(parents=True)
    payload=root/'targets/banks/OFFLINE/targets.npz'; payload.parent.mkdir(parents=True)
    atomic_publish_bytes(payload,b'owned payload')
    monkeypatch.setattr(worker,'results',lambda _:dict(pending=['set8_p3']))
    with pytest.raises(ValueError): worker.retire(spec,'101')
    assert payload.read_bytes()==b'owned payload' and not (root/'cleanup/intent.json').exists()


def test_partial_cleanup_fails_closed(tmp_path):
    spec=bare(tmp_path); folder=Path(spec['root'])/'cleanup'; folder.mkdir(parents=True)
    write_immutable_json(folder/'intent.json',dict(partial=True))
    with pytest.raises(ValueError,match='Partial cleanup'): worker.retired_evidence(spec)


def test_teacher_pair_identical_initialization_projection_not_exported(tmp_path):
    spec=bare(tmp_path); a=model.new_model(spec['nodes']['teacher_ce']); rng=torch.random.get_rng_state()
    b=model.new_model(spec['nodes']['teacher_supcon'])
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    raw=cache('train','OFFLINE'); ix=np.arange(22,dtype=np.int64)
    for task,teacher in (('teacher_ce',a),('teacher_supcon',b)):
        teacher.train(); teacher.capture=True
        loss,terms,weights=training.batch_loss(teacher,raw,None,ix,'cpu')
        grads=torch.autograd.grad(terms['ce'],tuple(teacher.mod.fc.parameters()),retain_graph=True)
        loss.backward()
        for p,g in zip(teacher.mod.fc.parameters(),grads): torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
        assert all((p.grad is None)==(task=='teacher_ce') for p in teacher.projector.parameters())
        native=DelphesParticleTransformer(); native.load_state_dict(teacher.native_state(),strict=True)


def test_offline_ram_rebuild_is_role_scoped_and_never_partial(tmp_path,monkeypatch):
    spec=bare(tmp_path); root=tmp_path/'prepared'; root.mkdir(); spec['anchor']['evidence']['prepared_root']=str(root)
    manifest={'content_hash':spec['anchor']['evidence']['cache'],'blocks':{'validation':{'OFFLINE':[{'path':'a'},{'path':'b'}]}}}
    prepared={'content_hash':spec['anchor']['evidence']['prepared'],'summaries':{'validation':{'OFFLINE':{'resident_bytes':1}}}}
    atomic_publish_bytes(root/'a',b'partial')
    with pytest.raises(ValueError,match='Partially missing'): data.load(spec,'validation','OFFLINE',(manifest,prepared,{},{}))


def test_cleanup_rejects_symlink_payload_before_any_delete(tmp_path,monkeypatch):
    spec=bare(tmp_path); root=Path(spec['root']); (root/'cleanup').mkdir(parents=True); (root/'report').mkdir()
    result={'pending':[]}; write_immutable_json(root/'report/results.json',result)
    monkeypatch.setattr(worker,'results',lambda _:result)
    outside=tmp_path/'outside.npz'; atomic_publish_bytes(outside,b'preserve')
    paths=[]
    for name in worker.target_payloads():
        path=root/name; path.parent.mkdir(parents=True,exist_ok=True); paths.append(path)
    try: paths[0].symlink_to(outside)
    except OSError: pytest.skip('Symlink privilege unavailable')
    for path in paths[1:]: atomic_publish_bytes(path,b'owned')
    with pytest.raises(PermissionError): worker.retire(spec,'100')
    assert outside.read_bytes()==b'preserve' and all(p.exists() for p in paths)


def test_source_current_rejects_runtime_modification(tmp_path,monkeypatch):
    monkeypatch.setattr(source,'_source',lambda *a:dict(content_hash='a'*64))
    monkeypatch.setattr(source,'__file__',str(tmp_path/'src/hlt_classification/luka_supervision_depth/source.py'))
    def git(args,**kwargs):
        output='\n'.join(source.REQUIRED) if 'ls-files' in args else 'M\tsrc/hlt_classification/luka_auxiliary_grid/model.py'
        return SimpleNamespace(stdout=output)
    monkeypatch.setattr(source.subprocess,'run',git)
    with pytest.raises(ValueError,match='Existing scientific runtime changed'): source.current(tmp_path,'a'*40)


def test_debug_plan_has_no_gpu_request_for_report_cleanup(tmp_path):
    spec=bare(tmp_path); rows=submission.render(spec)['commands']
    for row in rows:
        gpu=any(c.startswith('--gres=') for c in row['command'])
        assert gpu==(row['task_id'] not in ('report','cleanup'))
        if gpu: assert '--time=12:00:00' in row['command']


def test_baseline_complete30_matches_original_grid(tmp_path):
    spec=bare(tmp_path); tr=cache(); val=cache('validation','D000'); bank=fake_bank(); node=spec['nodes']['baseline']
    device='cuda' if torch.cuda.is_available() else 'cpu'
    with parity_backend(device):
        _,first,state=training.train(tr,val,node=node,bank=bank,device=device)
        _,second,oldstate=grid_training.train(tr,val,node=dict(node,arm='k1_e40_plain'),bank=bank,device=device)
    assert state.keys()==oldstate.keys() and all(torch.equal(state[k],oldstate[k]) for k in state)
    assert first['validation']==second['validation'] and first['selected_pass']==second['selected_pass']
    assert [r['validation'] for r in first['validation_history']]==[r['validation'] for r in second['validation_history']]

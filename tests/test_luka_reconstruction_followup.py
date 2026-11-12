"""R8 v2: immutable 18-fit screen, production gradients, qualification and durable release."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_luka_direct_reconstruction import fixture, target, compact_native
from test_luka_reconstruction_batched import examples
from hlt_classification.data.cache_contracts import write_immutable_json
from hlt_classification.luka_next_supervision_hydra import contracts,submission,campaign,worker,resources,training,model
from hlt_classification.luka_next_supervision_hydra import reconstruction_contracts as base
from hlt_classification.luka_next_supervision_hydra import reconstruction_followup as f
from hlt_classification.luka_next_supervision_hydra import reconstruction_loss as scalar,reconstruction_batched as batch
from hlt_classification.luka_next_supervision_hydra import reconstruction_targets as targets,reconstruction_model as decoder
from hlt_classification.luka_next_supervision_hydra import reconstruction_release as release
from hlt_classification.scouting import hcwdl_exact_dag_submission as dag


def test_registry_exact_replicas_weights_and_one_factor_strategies():
    assert len(f.TASKS)==len(set(f.TASKS))==18
    assert f.TASKS[:4]==('C2','R13','C3','R14')
    for control,replica,seed in (('C2','R13',2),('C3','R14',3)):
        assert f.setting(control)['family']=='C' and f.setting(control)['coefficient']==0
        assert f.setting(control)['seed_index']==seed
        expected=dict(base.setting('R8'),seed_index=seed)
        assert f.setting(replica)==expected
    assert [f.setting('R'+str(n))['coefficient'] for n in range(15,23)]==[.15,.2,.5,.7,1.,2.,5.,10.]
    changes={'R23':{'ramp_epochs'},'R24':{'start_after_epochs'},'R25':{'fade_start','fade_end'},
        'R26':{'consistency','consistency_weight'},'R27':{'field_weights'},'R28':{'token_layers'}}
    anchor=base.setting('R8')
    for task,keys in changes.items():
        cfg=f.setting(task)
        assert {k for k in cfg if cfg[k]!=anchor.get(k)}==keys
    assert f.setting('R27')['field_weights']==[1.,1.,2.]
    assert f.setting('R28')['token_layers']==[4,8]
    assert contracts.setting('C0')==contracts.ARMS['C0']  # old A/B/C controls unchanged
    assert f.recipe()['schema_version']==2 and f.recipe()['decoder']==base.recipe()['decoder']
    assert f.registration()['execution_version']=='R5'
    with pytest.raises(ValueError): f.validate(f.recipe(),'FOLLOWUP_SPEC')


def test_schedules_at_boundaries_and_during_fade():
    assert [contracts.coefficient(f.setting('R13'),p) for p in (0,1.5,3,30)]==[0,.15,.3,.3]
    assert [contracts.coefficient(f.setting('R23'),p) for p in (0,3,6)]==[0,.15,.3]
    assert [contracts.coefficient(f.setting('R24'),p) for p in (2,3,4.5,6)]==[0,0,.15,.3]
    assert [contracts.coefficient(f.setting('R25'),p) for p in (3,15,20,25,30)]==[.3,.3,.15,0,0]
    for patch in ({'fade_end':15.},{'fade_start':float('nan')},{'fade_start':1.}):
        with pytest.raises(ValueError): contracts.coefficient(dict(f.setting('R25'),**patch),5)


def test_qualification_separates_high_weights_new_paths_and_budgets_all_fits():
    assert len({resources.path_key(t) for t in f.TASKS})==13
    assert resources.path_key('R13')==resources.path_key('R14')==resources.path_key('R25')
    assert resources.path_key('C2')==resources.path_key('C3')
    assert resources.path_key('R21')!=resources.path_key('R22')!=resources.path_key('R13')
    for t in ('R26','R27','R28'): assert resources.path_key(t)!=resources.path_key('R13')
    sources={s:{'tokens':10000} for s in ('HLT','OFFLINE')}
    assert f.storage_bytes(sources,100)==targets.storage_bytes(sources,100,fits=44)
    assert f.storage_bytes(sources,100)>targets.storage_bytes(sources,100)


@pytest.mark.parametrize('counts',[(0,0),(0,1,4),(40,3,12)])
def test_track_reweighting_assignment_loss_and_gradients_match_scalar(counts):
    a,ts=examples(counts); b={k:v.detach().clone().requires_grad_() for k,v in a.items()}
    expected,ep=scalar.rows(a,ts,field_weights=(1,1,2)); actual,ap=batch.rows(b,ts,field_weights=(1,1,2))
    torch.testing.assert_close(actual,expected,rtol=3e-6,atol=3e-6)
    torch.testing.assert_close(ap['fields'],(ap['kinematic']+ap['composition']+2*ap['track'])/4)
    expected.sum().backward(); actual.sum().backward()
    for name in a:
        ga=a[name].grad; gb=b[name].grad
        if ga is not None: torch.testing.assert_close(gb,ga,rtol=3e-5,atol=3e-6)
    packed,ns=batch.pack(ts,a['presence'].shape[1],'cpu')
    slots,_=batch.assignments(b,packed,ns,field_weights=(1,1,2))
    for i,n in enumerate(ns):
        ix,col=scalar.match({k:v[i] for k,v in b.items()},scalar.target(ts[i],'cpu'),field_weights=(1,1,2))
        expected=np.empty(n,dtype=np.int64); expected[col]=ix
        np.testing.assert_array_equal(slots[i,:n].numpy(),expected)


@pytest.mark.parametrize('task',['C2','C3','R13','R14','R21','R22','R23','R24','R25','R26','R27','R28'])
def test_production_miniature_new_paths_and_selected_native_export(tmp_path,monkeypatch,task):
    spec,tr,va,bank=fixture(tmp_path,monkeypatch)
    spec.update(f.registration(),counts=dict(train=len(tr),validation=len(va)))
    source=None if task.startswith('C') else targets.candidates(spec,task,tr)
    report=training.fit(tmp_path/('fit_'+task),spec,task,tr,va,bank,candidates=source,device='cpu',technical=True)
    training.validate_fit(tmp_path/('fit_'+task),spec,task,technical=True)
    assert report['settings']==f.setting(task) and report['deployable_hlt_only']
    if source is not None: source.close()


def test_two_depth_taps_have_live_gradients_and_no_extra_parameters(tmp_path,monkeypatch):
    spec,tr,va,bank=fixture(tmp_path,monkeypatch)
    m=model.new_model(spec['imported'],'R28'); decoder.attach(m,12); m.capture=True
    outputs={}
    def retain(i):
        def hook(mod,args,out): outputs[i]=out; out.retain_grad()
        return hook
    handles=[m.mod.blocks[i-1].register_forward_hook(retain(i)) for i in (4,8)]
    raw=tr.batch(np.arange(3)); logits=training.forward(m,raw,'cpu')
    for h in handles: h.remove()
    n=outputs[8].shape[1]
    assert m.particle_tokens.shape==(3,2*n,128)
    torch.testing.assert_close(m.particle_tokens[:,:n],outputs[4])
    torch.testing.assert_close(m.particle_tokens[:,n:],outputs[8])
    # Isolate each half of the concatenated memory: both receive direct gradients.
    m.particle_tokens[:,:n].square().mean().backward(retain_graph=True)
    assert outputs[4].grad.abs().sum()>0 and (outputs[8].grad is None or not outputs[8].grad.any())
    m.particle_tokens[:,n:].square().mean().backward()
    assert outputs[8].grad.abs().sum()>0
    assert all(not block._forward_hooks and not block._forward_pre_hooks for block in m.mod.blocks)
    ref=decoder.Decoder(base.setting('R8'),12)
    assert sum(p.numel() for p in ref.parameters())==sum(p.numel() for p in m.aux.parameters())
    assert all(k.startswith('mod.') for k in m.deployment_state())


def test_durable_bootstrap_releases_18_independent_fits_and_afterany_report(tmp_path,monkeypatch):
    spec=dict(f.registration(),root=str(tmp_path),project_dir='/code',content_hash='a'*64,reused={'C0':{},'C1':{}},reference={'environment':{}})
    commands=[]
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    monkeypatch.setattr(release,'scheduler_review',lambda *a:None)
    monkeypatch.setattr(release.time,'sleep',lambda *a:None)
    def sbatch(cmd,**kw):
        commands.append(cmd); return SimpleNamespace(stdout=str(8000+len(commands)),stderr='',returncode=0)
    monkeypatch.setattr(dag.subprocess,'run',sbatch)
    submission.submit(spec,'bootstrap'); plan=submission.plan(spec,'bootstrap')
    assert plan['automatic_science']['tasks']==list(f.TASKS)
    submission.submit(spec,'bootstrap',execute=True,plan_hash=plan['content_hash'],authorization=f.AUTHORIZATION)
    assert '--dependency=afterok:8001' in commands[1]
    accepted=dict(content_hash='b'*64,minutes={t:120 for t in f.TASKS},reconstruction_review={})
    write_immutable_json(tmp_path/'setup/acceptance.json',accepted)
    monkeypatch.setattr(worker,'completed_setup',lambda *a:accepted)
    monkeypatch.setattr(submission,'accounting',lambda *a:['FAILED','1:0'])
    with pytest.raises(RuntimeError): release.finish(spec,'8002')
    assert len(commands)==2
    monkeypatch.setattr(submission,'accounting',lambda *a:['COMPLETED','0:0'])
    result=release.finish(spec,'8002')
    assert len(commands)==21 and set(result['jobs'])==set(f.TASKS)|{'report'}
    assert [cmd[-1] for cmd in commands[2:-1]]==list(f.TASKS)
    assert all(not any(x.startswith('--dependency=') for x in cmd) for cmd in commands[2:-1])
    assert '--dependency=afterany:'+':'.join(str(i) for i in range(8003,8021)) in commands[-1]
    assert '--partition=compute' in commands[-1] and commands[-1][-1]=='report'
    assert release.finish(spec,'8002')==result and len(commands)==21


def test_measured_setup_and_acceptance_cover_controls_high_weights_and_two_depth(tmp_path,monkeypatch):
    spec,tr,va,bank=fixture(tmp_path,monkeypatch,rows=130)
    spec.update(f.registration(),counts=dict(train=130,validation=130))
    monkeypatch.setattr(resources,'miniature_rows',lambda x:np.arange(130))
    monkeypatch.setattr(resources,'peaks',lambda d:dict(gpu_peak_bytes=1,peak_rss_bytes=1))
    monkeypatch.setattr(resources,'installed_environment',lambda:{})
    monkeypatch.setattr(resources,'gpu_identity',lambda:spec['qualification']['acceptance']['gpu'])
    write_immutable_json(tmp_path/'setup/baseline_parity.json',resources.baseline(spec,tr,'cpu'))
    acceptance=resources.measure(spec,tr,va,bank,'cpu')
    assert len(acceptance['measurements'])==13 and set(acceptance['minutes'])==set(f.TASKS)
    assert {'C2','R21','R22','R26','R27','R28'}<=set(acceptance['measurements'])
    assert resources.accepted(spec)==acceptance
    assert not (tmp_path/'setup/capacity_C2.json').exists()
    assert 'batch_reconstruction_seconds' not in acceptance['measurements']['C2']


def test_followup_registration_quota_version_and_parent_immutability(tmp_path,monkeypatch):
    import sys,json
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    from hlt_classification.luka_next_supervision_hydra import group_b,inputs,reuse
    from test_luka_next_supervision_hydra import spec_fixture
    _,tr,_,_=spec_fixture(tmp_path)
    parent=tmp_path/'native_parent.json'; parent.write_text('{}')
    old=dict(scope='group_A_only',content_hash='parent',root=str(tmp_path/'old'),project_dir=str(tmp_path/'oldcode'),
        environment_dir=sys.prefix,bundle_root=str(tmp_path/'bundle'),reference=dict(environment={}),
        imported=dict(node={'initialization_seed':1,'sampler_seed':2}),anchor=dict(evidence=dict(prepared=tr.foundation_sha256)),
        parent_spec=file_ref(parent),qualification={},raw_root=str(tmp_path/'raw'),raw_hash='r',
        delta_root=str(tmp_path/'delta'),delta_hash='d',embedding_hash='e',key_hash='k',
        reused={'C0':{'proof':0},'C1':{'proof':1}})
    original=tmp_path/'group_a.json'; original.write_text(json.dumps(old)); before=original.read_bytes()
    monkeypatch.setattr(group_b,'original',lambda *a:old)
    monkeypatch.setattr(campaign,'current',lambda *a:dict(content_hash='source'))
    monkeypatch.setattr(f.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='',stderr=''))
    monkeypatch.setattr(reuse,'check',lambda *a,**k:None); monkeypatch.setattr(inputs,'raw',lambda *a:None)
    def inventory(spec):
        n=spec['counts']['train']; sources={s:dict(rows=n,tokens=n*12,maximum_count=12) for s in ('OFFLINE','HLT')}
        return base.artifact('INVENTORY',parents=dict(raw=spec['raw_hash']),sources=sources,maximum=12,
            storage_budget_bytes=targets.storage_bytes(sources,n),quality_gate=False)
    monkeypatch.setattr(targets,'inventory',inventory)
    with pytest.raises(OSError,match='headroom'): f.create(tmp_path/'new',original,tmp_path/'code','f'*40,1)
    assert not (tmp_path/'new').exists()
    spec=f.create(tmp_path/'new',original,tmp_path/'code','f'*40,10)
    assert spec['contract'].endswith('FOLLOWUP_SPEC/v2')
    assert spec['storage_budget_bytes']>spec['target_inventory']['storage_budget_bytes']
    f.check(spec); assert original.read_bytes()==before
    with pytest.raises(ValueError): f.create(tmp_path/'new',original,tmp_path/'code','f'*40,10)
    spec['remaining'].pop()
    with pytest.raises(ValueError): f.check(spec)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('task',['R22','R28'])
def test_full_depth_installed_weaver_CUDA_full_and_partial_updates(tmp_path,monkeypatch,task):
    import importlib
    from hlt_classification.jetclass2_delphes import model as native
    production=importlib.reload(native).model_config
    monkeypatch.setattr(native,'model_config',production)
    spec,tr,va,bank=fixture(tmp_path,monkeypatch,rows=130)
    source=targets.candidates(spec,task,tr)
    m=model.new_model(spec['imported'],task).cuda()
    # Stress the historically observed298 slots, even for this small synthetic set.
    decoder.attach(m,298); opt=training.optimizer(m)
    for ix in (np.arange(128),np.arange(128,130)):
        values,details=training.step(m,opt,tr.batch(ix),bank.targets[ix],'cuda',candidates=source,
            indices=ix,epoch=1,position=7.,probe=True,measure=True)
        assert np.isfinite(values).all()
        assert details['probes']['weighted_auxiliary_token_gradient_norm']>0
        assert np.isfinite(details['probes']['ce_jet_gradient_norm'])
    assert not hasattr(model.deployment(m.deployment_state(),'cpu'),'aux')
    source.close(); del m,opt; torch.cuda.empty_cache()



def test_progress_labels_variants_and_pairs_new_seeds_without_writes(tmp_path,monkeypatch,capsys):
    from hlt_classification.luka_next_supervision_hydra import reporting,inputs
    spec=dict(f.registration(),root=str(tmp_path),content_hash='a'*64,bundle_root='unused',reused={})
    for task,auc in (('C2',.96),('R13',.961),('R22',.962)):
        write_immutable_json(tmp_path/task/'fit/epoch_01.json',contracts.artifact('EPOCH',
            parents=dict(spec=spec['content_hash']),task=task,pass_number=1,validation=dict(macro_ovr_auc=auc)))
    monkeypatch.setattr(inputs,'Bundle',lambda *a:SimpleNamespace(json=lambda *a:dict(recovery_anchors={'M0HLT':.95,'OFFLINE':.97})))
    monkeypatch.setattr(reporting,'scheduler_states',lambda *a:{})
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    reporting.progress(spec); output=capsys.readouterr().out
    assert 'R13: exact R8 weight0.3, seed 2' in output
    assert 'delta_selected_C2=+0.00100000' in output
    assert 'R22: R8 weight10.0, seed0' in output
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}

"""Shared pooling semantics, teacher routing, native parity and exact durable release."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import subprocess
import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score
from test_luka_u100_pretrain import cache
from hlt_classification.data.cache_contracts import (load_json, write_immutable_json, with_content_hash,
    deterministic_npz_bytes, atomic_publish_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.jetclass2_delphes.runner import _forward
from hlt_classification.luka_summary_tokens_hydra import (contracts as c, model, inputs, training,
    resources, submission, campaign, worker, reporting, reuse)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def spec(root):
    s = dict(campaign.registration(), root=str(root), project_dir='/project', content_hash='c'*64,
        anchor=dict(evidence=dict(prepared='a'*64)), reference=dict(environment={'fixture': True}),
        bundle_root='/bundle', environment_dir='/env', counts=dict(train=22, validation=22),
        parent_hash='d'*64, recovery_anchors=dict(M0HLT=.95, OFFLINE=.98), original50k_rows=22)
    s['nodes'] = campaign.nodes(dict(imported=dict(node=dict(initialization_seed=23, sampler_seed=29))))
    return s


def banks(train, task):
    rng = np.random.default_rng(8)
    return {k: SimpleNamespace(targets=rng.normal(size=(len(train), 128)).astype(np.float32),
        identities=train.identities, labels=train.labels,
        manifest=c.artifact('TEST_BANK', role='train', coordinate='IT' if k == 'it' else 'U100')) for k in c.weights(task)}


def small(monkeypatch):
    original = c.model_config
    def config(width=128):
        return dict(original(width), num_layers=1, num_cls_layers=1, pair_embed_dims=[8, 8, 8])
    monkeypatch.setattr(model, 'model_config', config)


def diagnostic(s):
    return c.artifact('RESOURCES', parents=dict(spec=s['content_hash']), status='complete', error=None,
        environment=s['reference']['environment'], gpu=dict(name='NVIDIA RTX A6000', compute_capability=[8, 6], total_memory_bytes=48*2**30),
        profiles={t: dict(initial_factory_fp32_parity=True, miniature_complete=True, deployment_reload=True,
            initial_factory_max_abs=0., fp32_equivalence=c.recipe()['fp32_equivalence'],
            shared_pooling_verified=True, architecture=dict(summaries=c.cell(t)['summaries'], native_parameters=10000,
                parameters=10000+(3072 if c.cell(t)['summaries']==3 else 0), additional_parameters=3072 if c.cell(t)['summaries']==3 else 0,
                batched_pooling_max_abs=0.),
            largest_support_backward=True, partial_batch_backward=True, real_targets=True,
            train_rows=s['counts']['train'], validation_rows=s['counts']['validation'], step_samples=[.3]*12,
            step_seconds=float(np.mean([.3]*12)), validation_seconds=1., probe_seconds=1., load_seconds=1.,
            memory=dict(peak_rss_bytes=4*2**30, gpu_peak_bytes=5*2**30)) for t in c.FITS})


def reseal(d):
    return with_content_hash({k: v for k, v in d.items() if k != 'content_hash'})


@pytest.mark.parametrize('task', c.FITS)
def test_full_thirty_epoch_width_specific_fit(tmp_path, monkeypatch, task):
    small(monkeypatch); s = spec(tmp_path); tr, va = cache(), cache('validation')
    native, report = training.fit(tmp_path/task, s, task, tr, va, banks(tr, task), device='cpu')
    assert len(report['history']) == 30 and report['native_reload'] and not report['oracle']
    assert native.width == c.cell(task)['width']
    assert all(k.startswith('mod.') for k in torch.load(tmp_path/task/'selected.pt', weights_only=True))
    training.validate_fit(tmp_path/task, s, task)
    with pytest.raises(RuntimeError):
        model.Native(4-native.summary_count).load_state_dict(native.state_dict(), strict=True)
    assert report['history'][0]['gradient_diagnostic']['weighted_encoder_norms']['ce'] > 0
    p = tmp_path/task/'validation_predictions.npz'; p.write_bytes(p.read_bytes()+b'bad')
    with pytest.raises(ValueError): training.validate_fit(tmp_path/task, s, task)


@pytest.mark.parametrize('task', c.FITS[:5])
def test_full_depth_128_and_partial11_cuda(tmp_path, task):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    s = spec(tmp_path); tr = cache(); bb = banks(tr, task)
    m = model.new_model(s, task).cuda(); opt = training.optimizer(m)
    assert len(m.mod.blocks) == 8 and len(m.mod.cls_blocks) == 2
    for size in (128,11):
        ix = np.arange(size)%len(tr)
        values, _ = training.step(m, task, opt, tr.batch(ix), bb, ix, 1., 'cuda')
        assert set(values) == c.term_names(task) and np.isfinite(list(values.values())).all()
    m.eval(); m.capture = False; native = model.Native(c.cell(task)['summaries']).cuda().eval()
    native.load_state_dict(m.native_state(), strict=True)
    with torch.no_grad(): torch.testing.assert_close(_forward(m,tr.batch(np.arange(11)),'cuda',bf16=True),
        _forward(native,tr.batch(np.arange(11)),'cuda',bf16=True), rtol=0, atol=0)


@pytest.mark.parametrize('task', ('A_s0', 'D_s0', 'E_s0'))
def test_cuda_production_miniature(tmp_path, monkeypatch, task):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    small(monkeypatch); s = spec(tmp_path); tr, va = cache(), cache('validation')
    resources.miniature(tmp_path/task, s, task, tr, va, banks(tr, task), 'cuda')
    training.validate_fit(tmp_path/task, s, task, technical=True)


def test_ce_only_never_opens_target_bank(tmp_path, monkeypatch):
    def forbidden(*a, **k): raise AssertionError('Read forbidden target')
    monkeypatch.setattr(inputs, 'embedding_bank', forbidden); monkeypatch.setattr(inputs, 'Bank', forbidden)
    monkeypatch.setattr(inputs, 'validate_file_ref', forbidden); monkeypatch.setattr(inputs, 'Bundle', forbidden)
    s = spec(tmp_path); assert inputs.banks(s, 'C_s0', cache()) == {}
    assert inputs.fit_parents(s, 'C_s0') == dict(spec=s['content_hash'], recipe=s['recipe']['content_hash'])
    training.check_inputs(s, 'C_s0', cache(), cache('validation'), {}, False)
    with pytest.raises(PermissionError): training.check_inputs(s,'B_s0',cache(),cache('validation'),{},False)
    with pytest.raises(PermissionError): training.check_inputs(s,'A_s0',cache(),cache(),{},False)
    val = cache('validation'); val.identities[:] = cache().identities
    with pytest.raises(ValueError, match='overlapping'): training.check_inputs(s,'A_s0',cache(),val,{},False)


@pytest.mark.parametrize('bad', ('memory','time','missing','gpu','negative','targets','samples'))
def test_resource_failure_does_not_use_quality(tmp_path, bad):
    s = spec(tmp_path); d = diagnostic(s); t = c.FITS[0]
    assert resources.accept(s, d)['minutes'].keys() == set(c.FITS)
    if bad == 'memory': d['profiles'][t]['memory']['peak_rss_bytes'] = 32*2**30
    elif bad == 'time': d['profiles'][t].update(step_samples=[1e6]*12, step_seconds=1e6)
    elif bad == 'missing': del d['profiles'][t]
    elif bad == 'gpu': d['gpu']['name'] = 'NVIDIA L40'
    elif bad == 'negative': d['profiles'][t]['load_seconds'] = -1
    elif bad == 'targets': d['profiles'][t]['real_targets'] = False
    else: d['profiles'][t]['step_samples'] = [.3]
    with pytest.raises((ValueError,MemoryError,RuntimeError)): resources.accept(s, reseal(d))


def fake_scheduler(monkeypatch):
    calls = []; reviews = []
    monkeypatch.setattr(campaign, 'check', lambda *a, **k: None)
    monkeypatch.setattr(submission, 'scheduler_review', lambda p: reviews.append(p))
    import hlt_classification.scouting.hcwdl_exact_dag_submission as dag
    def run(command, **kwargs):
        calls.append(command); return SimpleNamespace(stdout=str(8000+len(calls))+'\n', returncode=0)
    monkeypatch.setattr(dag.subprocess, 'run', run)
    return calls, reviews


def test_durable_release_exact_ids_independent_science_and_no_duplicates(tmp_path, monkeypatch):
    s = spec(tmp_path); (tmp_path/'setup').mkdir(); a = resources.accept(s, diagnostic(s))
    write_immutable_json(tmp_path/'setup/acceptance.json', a)
    calls, reviews = fake_scheduler(monkeypatch); submission.launch(s); p = submission.plan(s, 'bootstrap')
    with pytest.raises(PermissionError): submission.launch(s, execute=True, plan_hash='bad', authorization=c.AUTHORIZATION)
    ledger = submission.launch(s, execute=True, plan_hash=p['content_hash'], authorization=c.AUTHORIZATION)
    assert ledger['jobs'] == dict(setup='8001', release='8002')
    monkeypatch.setattr(submission, 'accounting', lambda job: ['COMPLETED','0:0'])
    monkeypatch.setattr(worker, 'completed_setup', lambda s: a)
    result = submission.finish(s, '8002')
    assert len(result['jobs']) == 11 and len(calls) == 13
    assert all(not any(x.startswith('--dependency=') for x in result['commands'][t]) for t in c.FITS)
    assert submission.finish(s, '8002') == result and len(calls) == 13
    assert [x['stage'] for x in reviews] == ['bootstrap','envelope','science']


def test_failed_setup_and_partial_submission_fail_closed(tmp_path, monkeypatch):
    s = spec(tmp_path); calls, _ = fake_scheduler(monkeypatch); submission.launch(s)
    p = submission.plan(s, 'bootstrap')
    submission.launch(s, execute=True, plan_hash=p['content_hash'], authorization=c.AUTHORIZATION)
    monkeypatch.setattr(submission, 'accounting', lambda job: ['FAILED','1:0'])
    with pytest.raises(RuntimeError, match='setup failed'): submission.finish(s, '8002')
    assert len(calls) == 2 and not (tmp_path/'submission_science').exists()
    other = tmp_path/'other'; other.mkdir(); s2 = spec(other); submission.launch(s2)
    (other/'submission_bootstrap/live_claim').mkdir()
    with pytest.raises(RuntimeError, match='ambiguous'):
        submission.launch(s2, execute=True, plan_hash=submission.plan(s2,'bootstrap')['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 2


def test_runtime_scripts_and_cli_parse():
    root = Path(__file__).resolve().parents[1]
    for p in campaign.RUNTIME:
        source = (root/p).read_text(); assert 'Fresh_check' not in source
        if p.endswith('.py'): compile(source,p,'exec')
    bash = Path('C:/Program Files/Git/bin/bash.exe')
    if bash.exists():
        for p in ('scripts/queue_luka_summary_tokens_hydra.sh','sbatch/run_luka_summary_tokens_hydra.sh'):
            subprocess.run([str(bash),'-n',p],check=True)


def test_native_dual_predictor_exact_donor_initialization_and_update(tmp_path):
    from hlt_classification.luka_selective_teacher_hydra import model as donor, training as donor_training
    s = spec(tmp_path); dspec = dict(s, nodes=dict(S2IT=s['nodes']['B_s0']))
    old = donor.new_model(dspec, 'S2IT'); fresh = model.new_model(s, 'B_s0')
    mapping = lambda k: k.replace('predictor.', 'heads.u100.').replace('extra.it.', 'heads.it.')
    for k, v in old.state_dict().items(): torch.testing.assert_close(v, fresh.state_dict()[mapping(k)], rtol=0, atol=0)
    tr = cache(); bb = banks(tr, 'B_s0'); ix = np.arange(22); raw = tr.batch(ix)
    torch.manual_seed(33)
    v1, _ = donor_training.step(old, 'S2IT', donor_training.optimizer(old), raw, bb, ix, 1., 'cpu')
    torch.manual_seed(33)
    v2, _ = training.step(fresh, 'B_s0', training.optimizer(fresh), raw, bb, ix, 1., 'cpu')
    assert v1 == v2
    for k, v in old.state_dict().items(): torch.testing.assert_close(v, fresh.state_dict()[mapping(k)], rtol=0, atol=0)


def test_reused_selected_it_bank_and_role_corruption_firewall(tmp_path, monkeypatch):
    from hlt_classification.luka_selective_teacher_hydra import targets
    from hlt_classification.context_fusion.worker import save_state
    small(monkeypatch); tr = cache(); s = spec(tmp_path); teacher = model.Native(1)
    checkpoint = tmp_path/'selected.pt'; save_state(checkpoint, teacher.state_dict()); ref = file_ref(checkpoint)
    target_dir = tmp_path/'bank'
    targets.export(target_dir, teacher, cache(coordinate='IT'), teacher_ref=ref, device='cpu', spec_hash=s['parent_hash'])
    task_receipt = tmp_path/'receipt.json'; write_immutable_json(task_receipt, {'test':'placeholder task receipt; parent authentication tested separately'})
    s['reused'] = dict(it_checkpoint=ref, it_bank=file_ref(target_dir/'manifest.json'),
        it_targets=file_ref(target_dir/'targets.npz'), it_receipt=file_ref(task_receipt))
    monkeypatch.setattr(inputs,'embedding_bank',lambda s,train:banks(train,'B_s0')['u100'])
    bb = inputs.banks(s,'E_s0',tr)
    training.check_inputs(s,'E_s0',tr,cache('validation'),bb,False)
    assert bb['it'].manifest['parents']['teacher'] == ref['sha256']
    with pytest.raises(ValueError): inputs.banks(s,'E_s0',cache('validation'))
    altered = cache(); altered.identities[:] = altered.identities[::-1]
    with pytest.raises(ValueError): inputs.banks(s,'E_s0',altered)
    p=target_dir/'targets.npz'; p.write_bytes(p.read_bytes()+b'corrupt')
    with pytest.raises(ValueError): inputs.banks(s,'E_s0',tr)


def test_real_cuda_setup_path_all_profiles_with_simulated_site(tmp_path, monkeypatch):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    from test_luka_attention import cache as raw_cache
    from hlt_classification.context_fusion.worker import save_state
    from hlt_classification.jetclass2_delphes.runner import predict
    small(monkeypatch); monkeypatch.setattr(resources,'model_config',model.model_config)
    s=spec(tmp_path); tr=raw_cache(coordinate='D000',rows=1804); va=cache('validation')
    s['counts']['train']=len(tr)
    monkeypatch.setattr(inputs,'views',lambda *a:(tr,va)); monkeypatch.setattr(inputs,'banks',lambda _,t,train:banks(train,t))
    monkeypatch.setattr(inputs,'fit_parents',lambda _,t:dict(spec=s['content_hash'],recipe=s['recipe']['content_hash'],
        **{k:b.manifest['content_hash'] for k,b in banks(tr,t).items()}))
    fixture=tmp_path/'fixture'; fixture.mkdir(); native=model.Native(1).cuda().eval()
    save_state(fixture/'checkpoint.pt',native.state_dict())
    atomic_publish_bytes(fixture/'predictions.npz',deterministic_npz_bytes(dict(identities=va.identities,
        labels=va.labels,probabilities=predict(native,va,device='cuda',batch_size=128))))
    del native
    class Bundle:
        def __init__(self,*a): pass
        def verify_all(self): pass
        def path(self,key): return fixture/key.split('/')[-1]
    monkeypatch.setattr(inputs,'Bundle',Bundle)
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(worker.site,'allocation',lambda s:'902'); monkeypatch.setattr(worker,'require_backend',lambda:None)
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:dict(setup='902'))
    for module in (worker,resources):
        monkeypatch.setattr(module,'installed_environment',lambda:s['reference']['environment'])
        monkeypatch.setattr(module,'gpu_identity',lambda:diagnostic(s)['gpu'])
    # Windows lacks Linux process peak RSS. This site/accounting envelope is simulated;
    # the GPU model, replay, forward/backward, timings and file validators below are real.
    monkeypatch.setattr(resources,'memory_snapshot',lambda _:dict(peak_rss_bytes=4*2**30,gpu_peak_bytes=torch.cuda.max_memory_allocated()))
    worker.run(s,'setup'); a=worker.completed_setup(s)
    assert set(a['minutes'])==set(c.FITS)
    d=load_json(tmp_path/'setup/resource_diagnostic.json')
    assert all(len(p['step_samples'])==12 for p in d['profiles'].values())
    for t in c.FITS:
        assert (tmp_path/'setup'/('mini_'+t)/'deployment.json').is_file()


def test_original_authentication_uses_own_project_and_exact_sentinel(tmp_path, monkeypatch):
    from hlt_classification.luka_selective_teacher_hydra.contracts import artifact
    old=artifact('SPEC',root='/old',project_dir='/old_code'); path=tmp_path/'old.json';write_immutable_json(path,old)
    calls=[]
    def run(command,**kwargs):
        calls.append((command,kwargs));return SimpleNamespace(returncode=0,stdout='SUMMARY_TOKENS_PARENT='+old['content_hash']+'\n',stderr='')
    monkeypatch.setattr(reuse.subprocess,'run',run)
    assert reuse.original(path)==old
    command,kwargs=calls[0]
    assert kwargs['cwd']=='/old_code' and kwargs['env']['PYTHONPATH']==str(Path('/old_code')/'src')
    assert 'worker.validate_teacher' in command[4] and command[-1]=='/old_code'
    monkeypatch.setattr(reuse.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='bad',stderr=''))
    with pytest.raises(ValueError,match='authentication failed'):reuse.original(path)


def test_source_guard_rejects_changes_to_existing_runtime(monkeypatch):
    root=Path(__file__).resolve().parents[1]; monkeypatch.setattr(campaign,'_source',lambda *a:dict(content_hash='a'*64))
    def git(command,**kwargs):
        if 'ls-files' in command:return SimpleNamespace(stdout='\n'.join(campaign.REQUIRED))
        if 'diff' in command:return SimpleNamespace(stdout='A\tsrc/hlt_classification/luka_summary_tokens_hydra/model.py\n')
        return SimpleNamespace(stdout='')
    monkeypatch.setattr(campaign.subprocess,'run',git)
    assert campaign.current(root,'b'*40)['content_hash']=='a'*64
    def bad(command,**kwargs):
        if 'diff' in command:return SimpleNamespace(stdout='M\tsrc/hlt_classification/luka_selective_teacher_hydra/model.py\n')
        return git(command,**kwargs)
    monkeypatch.setattr(campaign.subprocess,'run',bad)
    with pytest.raises(ValueError):campaign.current(root,'b'*40)


def test_immutable_registration_round_trip_and_changed_target_rejected(tmp_path, monkeypatch):
    import sys
    from hlt_classification.luka_selective_teacher_hydra.contracts import artifact
    parent_dir=tmp_path/'parent'; parent_dir.mkdir(); old_code=tmp_path/'old_code'; old_code.mkdir()
    project=tmp_path/'code'; project.mkdir(); bundle=tmp_path/'bundle'; bundle.mkdir()
    old=artifact('SPEC',root=str(parent_dir),project_dir=str(old_code),environment_dir=str(Path(sys.prefix).resolve()),
        bundle_root=str(bundle), reference=dict(environment={'fixture':True}),
        imported=dict(node=dict(initialization_seed=23,sampler_seed=29)),anchor=dict(evidence=dict(prepared='a'*64)),
        embedding_hash='e'*64,qualification={'test':'parent proof'},recovery_anchors=dict(M0HLT=.95,OFFLINE=.98))
    path=parent_dir/'campaign_spec.json'; write_immutable_json(path,old)
    sentinel=parent_dir/'target';sentinel.write_bytes(b'original immutable payload')
    refs=dict(it_checkpoint=file_ref(sentinel))
    monkeypatch.setattr(campaign,'current',lambda *a:dict(content_hash='f'*64))
    monkeypatch.setattr(reuse,'original',lambda p:load_json(p))
    monkeypatch.setattr(reuse,'references',lambda p:dict(it_checkpoint=file_ref(sentinel)))
    monkeypatch.setattr(campaign.site,'environment',lambda e:None)
    monkeypatch.setattr(campaign,'storage',lambda s:None)
    root=tmp_path/'new'
    s=campaign.create(root,path,project,'b'*40,9)
    campaign.check(s)
    assert s['reused']==refs and len(s['nodes'])==10
    for seed in (0,1):
        assert len({n['initialization_seed'] for t,n in s['nodes'].items() if c.cell(t)['seed']==seed})==1
        assert len({n['sampler_seed'] for t,n in s['nodes'].items() if c.cell(t)['seed']==seed})==1
    with pytest.raises(ValueError,match='fresh separate'):campaign.create(root,path,project,'b'*40,9)
    with pytest.raises(ValueError):campaign.create(tmp_path/'another',path,project,'b'*40,8)
    sentinel.write_bytes(b'substitution')
    with pytest.raises(ValueError,match='Immutable'):campaign.check(s)


@pytest.mark.parametrize('allocated', ('1','2','3','gpu'))
def test_cpu_allocation_keeps_observed_hydra_rounding(tmp_path, monkeypatch, allocated):
    s=spec(tmp_path)
    for k in ('SLURM_NTASKS','SLURM_CPUS_PER_TASK'):monkeypatch.setenv(k,'1')
    monkeypatch.setenv('SLURM_JOB_ID','901');monkeypatch.setenv('SLURM_MEM_PER_NODE','4000')
    count='1' if allocated=='gpu' else allocated
    gpu=',gres/gpu=1' if allocated=='gpu' else ''
    fields=('JobId=901 JobState=RUNNING Partition=compute Account=default QOS=normal '
        f'NumNodes=1 NumTasks=1 CPUs/Task=1 NumCPUs={count} '
        f'ReqTRES=cpu=1,mem=4000M,node=1 AllocTRES=cpu={count},mem=4000M,node=1{gpu}')
    def run(command,**kwargs):return SimpleNamespace(stdout='ClusterName = hydra\n' if 'config' in command else fields)
    monkeypatch.setattr(subprocess,'run',run)
    if allocated in ('1','2'):assert worker.site.cpu_allocation(s)=='901'
    else:
        with pytest.raises(PermissionError):worker.site.cpu_allocation(s)


def test_batch_accounting_uses_exact_science_ids(tmp_path,monkeypatch):
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:dict(A_s0='901',B_s0='902'))
    calls=[]
    def run(command,**kwargs):
        calls.append(command);return SimpleNamespace(stdout='901|COMPLETED|0:0\n999|RUNNING|0:0\n')
    monkeypatch.setattr(subprocess,'run',run)
    assert reporting.states(spec(tmp_path))==dict(A_s0=['COMPLETED','0:0'],B_s0=['ACCOUNTING_PENDING',''])
    assert len(calls)==1 and '901,902' in calls[0]


def test_registered_matrix_and_bounded_dag(tmp_path):
    s=spec(tmp_path); boot=submission.plan(s,'bootstrap'); env=submission.plan(s,'envelope')
    assert len(c.FITS)==len(set(c.FITS))==10
    assert c.FITS[:5]==('E_s0','B_s0','D_s0','C_s0','A_s0')
    assert [r['task_id'] for r in boot['commands']]==['setup','release']
    assert '--dependency=afterok:${JOB_setup}' in boot['commands'][1]['command']
    assert boot['upper_bound_commands']==env['commands']
    for row in env['commands'][:-1]:
        assert not row['dependencies'] and not any(x.startswith('--dependency=') for x in row['command'])
        assert '--gres=gpu:nvidia_rtx_a6000:1' in row['command']
    assert env['commands'][-1]['dependencies']==list(c.FITS)
    assert any(x.startswith('--dependency=afterany:') for x in env['commands'][-1]['command'])
    a=resources.accept(s,diagnostic(s)); measured=submission.plan(s,'science',a)
    assert submission.within_envelope(env,measured)
    measured['commands'][0]['command'].append('--gres=gpu:2')
    with pytest.raises(PermissionError):submission.within_envelope(env,measured)
    a['minutes'][c.FITS[0]]=2881
    with pytest.raises(ValueError):submission.plan(s,'science',a)
    assert c.model_config()['num_layers']==8 and c.model_config()['num_cls_layers']==2
    with pytest.raises(ValueError):c.model_config(256)


@pytest.mark.parametrize('seed',(0,1))
def test_initialization_parameter_counts_and_routes(tmp_path,monkeypatch,seed):
    small(monkeypatch); s=spec(tmp_path); raw=cache().batch(np.arange(22))
    models={name:model.new_model(s,f'{name}_s{seed}').eval() for name in 'ABCDE'}
    a=models['A']; expected=_forward(a,raw,'cpu',bf16=False)
    for name,m in models.items():
        m.capture=True; logits=_forward(m,raw,'cpu',bf16=False)
        torch.testing.assert_close(logits,expected,rtol=1e-5,atol=1e-6)
        extra=0 if name in 'AB' else 3072
        assert sum(p.numel() for p in m.mod.parameters())==a.native_parameters+extra
        assert len(m.mod.cls_blocks)==len(a.mod.cls_blocks)
        tt={k:torch.randn(22,128,requires_grad=True) for k in c.weights(f'{name}_s{seed}')}
        loss,terms=model.objective(m,f'{name}_s{seed}',logits,torch.from_numpy(raw['labels']),**tt)
        diag=training.summary_diagnostic(terms,m)
        for k in tt:
            norms=diag['weighted_loss_gradient_norms'][k]; source=c.cell(f'{name}_s{seed}')['prediction_sources'][k]
            assert norms[source]>0 and all(v==0 for i,v in enumerate(norms) if i!=source)
            gs=torch.autograd.grad(terms[k],list(m.mod.fc.parameters()),retain_graph=True,allow_unused=True)
            assert all(g is None for g in gs)
        loss.backward(); assert all(t.grad is None for t in tt.values())
        if name in 'CDE':assert m.mod.fc[0].weight.grad[:,128:].norm()>0
        m.capture=False
        with torch.no_grad():torch.testing.assert_close(_forward(m,raw,'cpu',bf16=False),_forward(a,raw,'cpu',bf16=False),rtol=1e-5,atol=1e-6)
        with pytest.raises(PermissionError):model.objective(m,f'{name}_s{seed}',logits,torch.from_numpy(raw['labels']),forbidden=torch.zeros(22,128))
    for name in 'DE':
        for k,v in models['C'].native_state().items():torch.testing.assert_close(models[name].native_state()[k],v,rtol=0,atol=0)
    for k,v in models['D'].state_dict().items():torch.testing.assert_close(models['E'].state_dict()[k],v,rtol=0,atol=0)
    for name in 'ABCDE':
        torch.manual_seed(s['nodes'][f'{name}_s{seed}']['initialization_seed']); model.Native(1)
        rng=torch.random.get_rng_state(); model.new_model(s,f'{name}_s{seed}')
        assert torch.equal(torch.random.get_rng_state(),rng)


def test_independent_pooling_reference_gradients_and_single_encoder(tmp_path,monkeypatch):
    small(monkeypatch); m=model.new_model(spec(tmp_path),'E_s0').eval(); m.capture=True
    raw=cache().batch(np.arange(11)); count=[]
    hook=m.mod.blocks[0].register_forward_hook(lambda *args:count.append(1))
    _forward(m,raw,'cpu',bf16=False); hook.remove(); assert len(count)==1
    actual=torch.stack(m.summaries,dim=1)
    x,pad=m.mod._forward_encoder(torch.from_numpy(raw['features']),v=torch.from_numpy(raw['vectors']),mask=torch.from_numpy(raw['mask']))
    b=len(x); x=x.repeat_interleave(3,0); pad=pad.repeat_interleave(3,0)
    q=torch.cat((m.mod.cls_token,m.mod.extra_cls_tokens),dim=1).expand(b,-1,-1).reshape(b*3,1,128)
    for block in m.mod.cls_blocks:q=block(x,x_cls=q,padding_mask=pad)
    reference=m.mod.norm(q).reshape(b,3,128)
    torch.testing.assert_close(actual,reference,rtol=1e-5,atol=1e-6)
    probe=torch.randn_like(actual)
    params=list(m.mod.blocks.parameters())+list(m.mod.cls_blocks.parameters())+[m.mod.cls_token,m.mod.extra_cls_tokens]
    ga=torch.autograd.grad((actual*probe).sum(),params,retain_graph=True)
    gr=torch.autograd.grad((reference*probe).sum(),params)
    for a,r in zip(ga,gr):torch.testing.assert_close(a,r,rtol=2e-4,atol=2e-4)
    with torch.no_grad():
        _forward(m,raw,'cpu',bf16=False); before=[h.clone() for h in m.summaries]
        m.mod.extra_cls_tokens[:,0].add_(torch.randn(1,128)); _forward(m,raw,'cpu',bf16=False)
        torch.testing.assert_close(before[0],m.summaries[0],rtol=0,atol=0)
        torch.testing.assert_close(before[2],m.summaries[2],rtol=0,atol=0)
        assert not torch.allclose(before[1],m.summaries[1])
    assert resources.qualify_pooling(m,raw,'cpu')['additional_parameters']==3072


def test_padding_invariance(tmp_path,monkeypatch):
    small(monkeypatch); raw=cache().batch(np.arange(22)); m=model.new_model(spec(tmp_path),'E_s0').eval()
    changed={k:v.copy() for k,v in raw.items()}
    for key in ('features','vectors','mask'):
        changed[key]=np.concatenate((raw[key],np.zeros((*raw[key].shape[:2],5),raw[key].dtype)),axis=-1)
    with torch.no_grad():torch.testing.assert_close(_forward(m,raw,'cpu',bf16=False),_forward(m,changed,'cpu',bf16=False),rtol=1e-5,atol=1e-6)


def test_matched_contrasts_missing_controls():
    a=dict(A_s0=.9,B_s0=.92,C_s0=.91,D_s0=.93,E_s0=.95); r=reporting.contrasts(a)
    assert r['per_seed']['0']['E-D']==pytest.approx(.02)
    assert r['per_seed']['0']['(E-C)-(B-A)']==pytest.approx(.02)
    assert not r['two_seed_descriptive']
    a.update({k.replace('_s0','_s1'):v+.001 for k,v in list(a.items())})
    assert reporting.contrasts(a)['two_seed_descriptive']['E-D']['mean']==pytest.approx(.02)
    del a['D_s0']; assert reporting.contrasts(a)['per_seed']['0']['E-D'] is None


def test_native_ce_exact_donor_update(tmp_path):
    from hlt_classification.luka_capacity_hydra import model as donor, training as dt
    s=spec(tmp_path); ds=dict(s,nodes=dict(N_CE_s0=s['nodes']['A_s0']))
    old=donor.new_model(ds,'N_CE_s0'); fresh=model.new_model(s,'A_s0'); tr=cache(); ix=np.arange(22); raw=tr.batch(ix)
    for k,v in old.state_dict().items():torch.testing.assert_close(v,fresh.state_dict()[k],rtol=0,atol=0)
    torch.manual_seed(33); v1,_=dt.step(old,'N_CE_s0',dt.optimizer(old),raw,{},ix,1.,'cpu')
    torch.manual_seed(33); v2,_=training.step(fresh,'A_s0',training.optimizer(fresh),raw,{},ix,1.,'cpu')
    assert v1==v2
    for k,v in old.state_dict().items():torch.testing.assert_close(v,fresh.state_dict()[k],rtol=0,atol=0)


def test_completed_and_provisional_reports_are_read_only(tmp_path,monkeypatch):
    small(monkeypatch); s=spec(tmp_path); tr,va=cache(),cache('validation')
    arrays=dict(identities=va.identities,labels=va.labels,probabilities=np.full((22,11),1/11,np.float32))
    for name in ('M0HLT','OFFLINE','matched'):
        a=copy.deepcopy(arrays)
        if name=='OFFLINE':a['probabilities']=.1*a['probabilities']+.9*np.eye(11,dtype=np.float32)[va.labels]
        atomic_publish_bytes(tmp_path/(name+'.npz'),deterministic_npz_bytes(a))
    atomic_publish_bytes(tmp_path/'subset.npz',deterministic_npz_bytes(dict(indices=np.arange(22,dtype=np.int64))))
    class Bundle:
        def __init__(self,*a):pass
        def path(self,k):return tmp_path/('subset.npz' if 'subset' in k else k.split('/')[1]+'.npz')
    monkeypatch.setattr(inputs,'Bundle',Bundle);monkeypatch.setattr(submission,'accepted_jobs',lambda *a:{})
    pending=reporting.results(s);assert pending['pending']==list(c.FITS) and not pending['deployments']
    task='C_s0';folder=tmp_path/task;folder.mkdir()
    _,r=training.fit(folder/'fit',s,task,tr,va,{},device='cpu')
    write_immutable_json(folder/'receipt.json',{'simulated':'allocation receipt'})
    monkeypatch.setattr(worker,'receipt',lambda *a:None)
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result=reporting.results(s);progress=reporting.progress(s)
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert task not in result['pending'] and progress['best'][task]['epoch']==r['selected_epoch']
    selected=[row for row in result['rows'] if row['model']==task]
    assert len(selected)==2 and all(row['delta_matched_CE']==0 and row['delta_matched_B'] is None for row in selected)
    assert result['deployments'][task]['profile']['additional_parameters']==3072
    p=folder/'fit/epoch-01.json';p.write_text('{}')
    with pytest.raises(ValueError):reporting.progress(s)


@pytest.mark.parametrize('seed',(0,1))
def test_production_depth_fp32_setup_equivalence_cuda(tmp_path,seed):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    s=spec(tmp_path);task=f'E_s{seed}';m=model.new_model(s,task).cuda().eval()
    tr=cache();raw=tr.batch(np.arange(128)%len(tr))
    torch.manual_seed(s['nodes'][task]['initialization_seed']);native=model.Native(1).cuda().eval()
    with torch.no_grad():
        torch.testing.assert_close(_forward(m,raw,'cuda',bf16=False),_forward(native,raw,'cuda',bf16=False),**c.recipe()['fp32_equivalence'])
    proof=resources.qualify_pooling(m,raw,'cuda')
    assert proof['summaries']==3 and proof['additional_parameters']==3072
    assert len(m.mod.blocks)==8 and len(m.mod.cls_blocks)==2

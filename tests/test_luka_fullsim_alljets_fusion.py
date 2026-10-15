"""Synthetic/CPU continuation acceptance, never claimed as real A100 evidence."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess

import numpy as np
import pytest
import torch

from test_luka_fullsim_stage2 import dataset, prepared, config
from test_luka_fullsim_alljets import all_prepared
from test_luka_fullsim_science import TinyModel
from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.jetclass2_delphes import runner
from hlt_classification.jetclass2_delphes.banks import publish_bank
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.contracts import artifact as kernel_artifact
from hlt_classification.luka_fullsim import (alljets_fusion_campaign as c, alljets_fusion_source as source,
    alljets_fusion_submission as sub, alljets_fusion_worker as w, alljets_fusion_preflight as pre,
    alljets_campaign as old, regular_cache)


def test_graph_both_routes_cold_start_and_paired_seeds():
    nodes = {n['node_id']:n for n in c.nodes()}
    tasks = {t['task_id']:t for t in c.tasks()}
    assert len(nodes) == 8 and len(tasks) == 16
    assert sum(t['kind']=='reduce' for t in tasks.values()) == 6
    assert tasks['train_FUSION_U050']['dependencies'] == []
    assert nodes['FUSION_U050']['teacher'] == 'U000'
    assert tasks['train_FINAL_DIRECT_D000']['dependencies'] == ['reduce_FUSION_D000']
    assert tasks['train_FUSION_D000_D000']['dependencies'] == ['reduce_FUSION_D000']
    assert tasks['train_FINAL_BRIDGE_D000']['dependencies'] == ['reduce_FUSION_D000_D000']
    seen = set()
    for name, task in tasks.items():
        assert set(task['dependencies']) <= seen
        seen.add(name)
    assert not any('M0HLT' in n or n=='reduce_U000' for n in tasks)
    for n in ('FINAL_DIRECT_D000','FINAL_BRIDGE_D000'):
        for field in ('initialization_seed','sampler_seed'):
            assert nodes[n][field] == old.nodes()[0][field]
        assert w.input_label(nodes[n]) == 'single HLT'
    assert w.input_label(nodes['FUSION_D000']) == 'OFFLINE ORACLE'
    assert w.input_label(nodes['FUSION_D000_D000']) == 'two HLT encoders'
    assert c.COUNTS == old.COUNTS == dict(train=299234, validation=74895)
    assert c.registration()['training_recipe'] == old.recipe()
    assert c.registration()['imported_weights'] == []
    assert c.registration()['imported_train_banks'] == ['U000']


def test_scope_and_unchanged_parent_sources():
    for key, value in [('final_test_accessed',True), ('historical_test_repurposed_for_training',False),
                       ('independent_test_available',True)]:
        with pytest.raises(PermissionError):
            c.validate(with_content_hash(dict(c.artifact('GATE'), **{key:value})), 'GATE')
    root = Path(__file__).resolve().parents[1]
    for name in source.REQUIRED:
        assert (root/name).is_file(), name
    # This continuation must not edit any old scientific source, unlike older
    # historical allowlists spanning unrelated intervening campaigns.
    changed = subprocess.run(['git','diff','--name-only','9e591c7a73e1fcf24d2e2032be441dd3ee965052','--','src'],
        cwd=root, check=True, capture_output=True, text=True).stdout.splitlines()
    assert set(changed) <= source.ADDITIONS
    worker = (root/'sbatch/run_luka_fullsim_alljets_fusion.sh').read_text()
    for text in ('${PROJECT_DIR}/sbatch/common.sh', 'PYTHONNOUSERSITE=1', '${CONDA_PREFIX}/lib', 'atlas_kd_sporc'):
        assert text in worker
    assert worker.index('CUBLAS_WORKSPACE_CONFIG') < worker.index('exec python')


def make_gate(tmp_path):
    root = tmp_path/'new'
    root.mkdir()
    return c.artifact('GATE', root=str(root), project_dir=str(tmp_path/'project'),
        source_commit='a'*40, source_snapshot=dict(content_hash='b'*64), parent_import={}, **c.registration())


def test_dry_live_exact_submission_and_gate_binding(tmp_path, monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as exact
    gate = make_gate(tmp_path)
    monkeypatch.setattr(c, 'validate_gate', lambda _:None)
    monkeypatch.setattr(sub, 'check_site', lambda _:None)
    calls = []
    def scheduler(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args,0,str(900+len(calls)),'')
    monkeypatch.setattr(exact.subprocess, 'run', scheduler)
    plan = sub.submit(gate)
    assert not calls and plan['jobs']==1
    args = plan['commands'][0]['command']
    assert '--mem=256000M' in args and '--time=08:00:00' in args and '--partition=debug' in args
    assert not any(a.startswith('--dependency') for a in args)
    with pytest.raises(PermissionError):
        sub.submit(gate, execute=True, plan_hash='0'*64, authorization=c.AUTHORIZATION)
    ledger = sub.submit(gate, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    assert sub.accepted_jobs(gate) == {'preflight':'901'}
    assert sub.submit(gate, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION) == ledger
    assert len(calls)==1
    spec = c.artifact('SPEC', root=str(tmp_path/'science'), project_dir=gate['project_dir'],
        source_commit=gate['source_commit'], train_minutes=1200, reduce_minutes=90, **c.registration())
    plan = sub.render(spec)
    assert plan['jobs']==16
    assert not plan['commands'][0]['dependencies']
    assert all('U000' not in dep for cmd in plan['commands'] for dep in cmd['dependencies'])
    assert '--time=20:00:00' in plan['commands'][0]['command']
    assert all('--gres=gpu:a100:1' in r['command'] for r in plan['commands'][:-2])


def test_interrupted_claim_never_silently_retries(tmp_path, monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as exact
    gate = make_gate(tmp_path)
    monkeypatch.setattr(c,'validate_gate',lambda _:None)
    monkeypatch.setattr(sub,'check_site',lambda _:None)
    plan = sub.submit(gate)
    monkeypatch.setattr(exact.subprocess,'run',lambda *a,**k: (_ for _ in ()).throw(RuntimeError('lost reply')))
    with pytest.raises(RuntimeError):
        sub.submit(gate,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    with pytest.raises(PermissionError, match='interrupted'):
        sub.submit(gate,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)


def fixture_spec(all_prepared, tmp_path):
    gate, manifest, p, inv, splits = all_prepared
    parent_root = tmp_path/'parent'
    parent_root.mkdir()
    parent = dict(root=str(parent_root), admission=dict(prepared_root=str(Path(gate['root'])/'prepared')))
    write_immutable_json(parent_root/'campaign_spec.json', parent)
    train = regular_cache.load(parent['admission']['prepared_root'], role='train', coordinate='U000', max_ram_bytes=2**30)
    val = regular_cache.load(parent['admission']['prepared_root'], role='validation', coordinate='D000', max_ram_bytes=2**30)
    probs = np.full((len(train),11),1/11,np.float32)
    args = dict(foundation_sha256=p['content_hash'], teacher_report_sha256='b'*64, teacher_node='U000', role='train')
    publish_bank(parent_root/'reduce_U000/train', identities=train.identities, probabilities=probs, **args)
    metrics = evaluate_probabilities(val.labels,np.full((len(val),11),1/11,np.float32))
    oldrows = [dict(node_id=n['node_id'], deployable=n['deployable'], state='COMPLETE',
        validation=metrics, original_validation=metrics, selected_pass=1, passes=60) for n in old.nodes()]
    imported = dict(spec=file_ref(parent_root/'campaign_spec.json'), evidence=dict(cache=manifest['content_hash'],
        tasks=dict(reduce_U000='d'*64), rows=oldrows,
        u000_bank=dict(train_bank='reduce_U000/train', teacher_report_sha256='b'*64)))
    root=tmp_path/'science'
    root.mkdir()
    spec=c.artifact('SPEC',root=str(root),project_dir=str(tmp_path/'project'),source_commit='a'*40,
        prepared_sha256=p['content_hash'],preflight_sha256='c'*64,parent_import=imported,
        **{**c.registration(),'counts':p['counts']})
    return spec,parent,p


def test_cached_pairs_exact_alias_and_role_firewall(all_prepared, tmp_path, monkeypatch):
    spec,parent,p=fixture_spec(all_prepared,tmp_path)
    original, calls = regular_cache.load, []
    def load(*a,**kw):
        calls.append((kw['role'],kw['coordinate']))
        return original(*a,**kw)
    monkeypatch.setattr(regular_cache,'load',load)
    for role in ('train','validation'):
        for n in c.nodes():
            calls.clear()
            value=w.cache(spec,p,n,role)
            required={n['coordinate'],n['context_coordinate']}-{None}
            assert calls==[(role,co) for co in dict.fromkeys((n['coordinate'],n['context_coordinate'])) if co is not None]
            if n['context_coordinate'] == n['coordinate']:
                assert value.primary is value.context
            if n['context_coordinate'] is None:
                assert required == {'D000'}
            else:
                np.testing.assert_array_equal(value.primary.identities,value.context.identities)
    for role in ('final_test','unused','test'):
        with pytest.raises(PermissionError):
            w.cache(spec,p,c.nodes()[0],role)
    with pytest.raises(ValueError):
        w.teacher(spec,'U000',np.zeros((1,32),np.uint8))


def cpu_worker(monkeypatch,p,parent):
    monkeypatch.setattr(c,'validate_spec',lambda _:(parent,p,{}))
    monkeypatch.setattr(w,'execution',lambda *a:'fixture-not-slurm')
    monkeypatch.setattr(w,'memory_snapshot',lambda _:dict(rss_bytes=1,peak_rss_bytes=1,gpu_peak_bytes=0))
    monkeypatch.setattr(w,'new_model',lambda n:(torch.manual_seed(n['initialization_seed']),TinyModel())[1])
    monkeypatch.setattr(TinyModel,'to',lambda self,*a,**k:self)
    monkeypatch.setattr(w,'train_kernel',lambda model,train,val,**kw:runner.train_kernel(model,train,val,**{**kw,'device':'cpu'}))
    monkeypatch.setattr(w,'predict',lambda model,train,**kw:runner.predict(model,train,**{**kw,'device':'cpu'}))


def test_full_eight_fit_graph_real_banks_diagnostics_and_corruption(all_prepared,tmp_path,monkeypatch):
    spec,parent,p=fixture_spec(all_prepared,tmp_path)
    cpu_worker(monkeypatch,p,parent)
    before=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert len(w.result_rows(spec))==17
        for task in c.tasks():
            receipt=w.run(spec,task['task_id'])
            assert w.run(spec,task['task_id'])==receipt
        assert w.completed(spec,'complete')['result']['fits']==8
        rows=w.result_rows(spec)
        assert len(rows)==17 and all(r['state']=='COMPLETE' for r in rows)
        assert all(r['original_validation'] is not None for r in rows)
        first=w.completed(spec,'train_FUSION_U050')
        assert first['dependencies']=={'external:reduce_U000':'d'*64}
        for node in c.nodes():
            report=load_json(Path(spec['root'])/('train_'+node['node_id'])/'training_report.json')
            assert 60<=report['passes']<=100
        w.print_results(spec)
        target=Path(spec['root'])/'train_FINAL_BRIDGE_D000/validation_predictions.npz'
        target.write_bytes(target.read_bytes()+b'corrupt')
        with pytest.raises(ValueError):
            w.completed(spec,'complete')
    finally:
        torch.set_num_threads(before)


def test_preserve_partial_task_and_reject_probe_teacher(all_prepared,tmp_path,monkeypatch):
    spec,parent,p=fixture_spec(all_prepared,tmp_path)
    cpu_worker(monkeypatch,p,parent)
    (Path(spec['root'])/'train_FUSION_U050').mkdir()
    with pytest.raises(FileExistsError):
        w.run(spec,'train_FUSION_U050')
    with pytest.raises(ValueError):
        w.teacher(spec,'FUSION_U050_probe',np.zeros((1,32),np.uint8))


def evidence_fixture():
    gate=dict(content_hash='a'*64,source_snapshot=dict(content_hash='b'*64),**c.registration())
    p=dict(content_hash='c'*64,source_snapshot=dict(content_hash='d'*64),summaries={r:{co:dict(maximum=n,rows=c.COUNTS[r])
        for co in regular_cache.s.COORDINATES} for r,n in [('train',5),('validation',9)]})
    from test_luka_fullsim_science import evidence
    _,old_report=evidence()
    # Synthetic parity facts only; no claim of real GPU acceptance.
    pars=old_report['parities']
    parities=dict(installed=pars['installed'],native_mask=True)
    measurements=[]
    for n in pre.probe_nodes():
        for precision,key in [('fp32','False'),('bf16','True')]:
            if n['context_coordinate'] is not None:
                parities[n['node_id']+'_'+precision]=deepcopy(pars['FUSION_probe_'+key])
        training=kernel_artifact('KERNEL_TRAINING_REPORT',node=n,foundation_sha256=p['content_hash'],
            recipe_sha256=gate['training_recipe']['content_hash'],passes=1,scientific_fit=False,acceptance_only=True,
            selected_weights_restored=True,final_test_accessed=False,validation=dict(rows=c.COUNTS['validation']),runtime_seconds=10.)
        measurements.append(dict(node=n,training=training,counts=c.COUNTS,cache_seconds=1.,inference_seconds=2.,
            stress_batch=256,stress_steps=3,validation_stress_batch=256,validation_stress_steps=3,validation_stress_inference_only=True,
            stress_offload=deepcopy(pars['FUSION_probe_True']['offload_stats'])))
    tr,red=pre.projections(measurements)
    r=c.artifact('PREFLIGHT',parents=dict(gate=gate['content_hash'],prepared=p['content_hash']),
        counts=c.COUNTS,cpus=6,memory_mb=c.MEMORY_MB,source_snapshot=gate['source_snapshot'],preparation_source_snapshot=p['source_snapshot'],
        site=gate['execution_site'],recipe=gate['training_recipe'],scientific_fit=False,scientific_metrics_for_selection=False,
        checkpoint_probability_roundtrip=True,technical_checks_passed=True,resource_envelope_ok=True,fits_debug_24h=True,
        cublas_workspace_config=':4096:8',cudnn_v8_disabled='unset',support_maxima=pre.support(p),technical_teacher='authenticated_parent_U000_train_bank',
        job_id='123',environment=old_report['environment'],gpu=old_report['gpu'],parities=parities,measurements=measurements,
        peak_rss_bytes=2**30,gpu_peak_bytes=2**30,projected_train_minutes=tr,projected_reduce_minutes=red,
        outputs=[dict(path=name,bytes=1,sha256='f'*64) for name in ['runtime.json',*(n['node_id']+'.pt' for n in pre.probe_nodes()),
            *(n['node_id']+'_bank/manifest.json' for n in pre.probe_nodes())]])
    return r,gate,p


def test_strict_fusion_resource_and_parity_admission():
    r,gate,p=evidence_fixture()
    assert pre.validate_evidence(r,gate,p)==r
    for key,value in [('counts',dict(train=100000,validation=50000)),('memory_mb',90000),('scientific_fit',True),
        ('fits_debug_24h',False),('peak_rss_bytes',256*2**30),('gpu_peak_bytes',39*2**30),('projected_train_minutes',1),
        ('outputs',[]),('measurements',[]),('source_snapshot',{})]:
        with pytest.raises((ValueError,PermissionError)):
            pre.validate_evidence(with_content_hash(dict(r,**{key:value})),gate,p)
    bad=deepcopy(r)
    bad['parities']['FUSION_U050_probe_fp32']['passed']=False
    with pytest.raises(ValueError):
        pre.validate_evidence(with_content_hash(bad),gate,p)
    bad=deepcopy(r)
    for m in bad['measurements']:
        m['training']=with_content_hash(dict(m['training'],runtime_seconds=1000.))
    bad['projected_train_minutes'],bad['projected_reduce_minutes']=pre.projections(bad['measurements'])
    with pytest.raises(ValueError,match='debug24h'):
        pre.validate_evidence(with_content_hash(bad),gate,p)


def test_validation_stress_has_no_labels_loss_or_gradients(all_prepared,tmp_path,monkeypatch):
    spec,parent,p=fixture_spec(all_prepared,tmp_path)
    batch=w.cache(spec,p,c.nodes()[0],'validation')
    calls=[]
    def forward(model,raw,device,bf16):
        assert set(raw)=={'features','vectors','mask'}
        assert not torch.is_grad_enabled() and not model.training
        calls.append(raw['features'].shape)
        return torch.zeros((256,11))
    monkeypatch.setattr(pre,'_forward',forward)
    model=TinyModel()
    pre.validation_stress(model,batch,device='cpu')
    assert len(calls)==3 and all(p.grad is None for p in model.parameters())
    with pytest.raises(PermissionError):
        pre.validation_stress(model,w.cache(spec,p,c.nodes()[0],'train'),device='cpu')


@pytest.mark.parametrize('failure', [None, 'train', 'validation'])
def test_cpu_preflight_payloads_admission_and_stress_failure(all_prepared,tmp_path,monkeypatch,failure):
    """Real cached populations, technical passes and banks; GPU facts mocked."""
    spec,parent,p=fixture_spec(all_prepared,tmp_path)
    monkeypatch.setattr(c,'COUNTS',p['counts'])
    _,_,_=pre.probe_nodes()
    synthetic,_,_=evidence_fixture()
    gate=c.artifact('GATE',root=str(tmp_path/'gate'),project_dir=str(tmp_path/'project'),source_commit='a'*40,
        source_snapshot=dict(content_hash='e'*64),parent_import=spec['parent_import'],
        prepared_sha256=p['content_hash'],**c.registration())
    Path(gate['root']).mkdir()
    write_immutable_json(Path(gate['root'])/'gate_spec.json',gate)
    monkeypatch.setattr(c,'validate_gate',lambda _ :parent)
    monkeypatch.setattr(w,'execution',lambda *a:'123')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    monkeypatch.setattr(pre,'gpu_identity',lambda:synthetic['gpu'])
    monkeypatch.setattr(pre,'installed_environment',lambda:synthetic['environment'])
    monkeypatch.setattr(pre,'installed_parity',lambda *a,**k:synthetic['parities']['installed'])
    monkeypatch.setattr(pre,'native_mask_parity',lambda *a,**k:True)
    monkeypatch.setattr(pre,'native_offload_parity',lambda *a,**k:synthetic['parities'][
        'FUSION_U050_probe_'+('bf16' if k['bf16'] else 'fp32')])
    monkeypatch.setattr(pre,'memory_snapshot',lambda _ :dict(rss_bytes=2**30,peak_rss_bytes=2**30,gpu_peak_bytes=2**30))
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(pre,'clear',lambda:None)
    monkeypatch.setattr(pre,'new_model',lambda n:(torch.manual_seed(n['initialization_seed']),TinyModel())[1])
    events=[]
    def train_stress(model,train):
        events.append('train')
        assert train.role=='train'
        if failure=='train':
            raise MemoryError('injected train stress')
        return deepcopy(synthetic['parities']['FUSION_U050_probe_fp32']['offload_stats'])
    validation=pre.validation_stress
    def validation_stress(model,val):
        events.append('validation')
        if failure=='validation':
            raise MemoryError('injected validation stress')
        validation(model,val,device='cpu')
    monkeypatch.setattr(pre,'stress',train_stress)
    monkeypatch.setattr(pre,'validation_stress',validation_stress)
    def fit(model,train,val,**kw):
        events.append('fit')
        report,state=runner.train_kernel(model,train,val,**{**kw,'device':'cpu'})
        return with_content_hash(dict(report,runtime_seconds=max(.001,report['runtime_seconds']))),state
    monkeypatch.setattr(pre,'train_kernel',fit)
    monkeypatch.setattr(pre,'predict',lambda model,cache,**kw:runner.predict(model,cache,**{**kw,'device':'cpu'}))
    class Measurement:
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def report(self): return dict(sampled_peak_tree_rss_bytes=2**30)
    monkeypatch.setattr(pre,'Measurement',Measurement)
    threads=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        if failure:
            with pytest.raises(MemoryError,match='injected'):
                pre.run(gate)
            assert 'fit' not in events
            assert not (Path(gate['root'])/'preflight/preflight.json').exists()
            with pytest.raises(FileNotFoundError):
                c.create_science(gate)
            return
        r=pre.run(gate)
        assert events==['train','validation','fit']*3
        assert r['technical_checks_passed'] and len(r['measurements'])==3
        assert c.evidence(gate)==(parent,p,r)
        monkeypatch.setattr(sub,'accepted_jobs',lambda _ :{'preflight':'123'})
        monkeypatch.setattr(c,'preflight_accounting',lambda job:dict(job_id=str(job),state='COMPLETED',exit_code='0:0'))
        science=c.create_science(gate)
        assert c.validate_spec(science)==(parent,p,r)
        assert len(science['nodes'])==8 and science['jobs']==16
        with pytest.raises(ValueError):
            c.validate_spec(with_content_hash(dict(science,train_minutes=1)))
        monkeypatch.setattr(sub,'accepted_jobs',lambda _ :{'preflight':'999'})
        with pytest.raises(ValueError,match='journal'):
            c.create_science(gate)
        # No technical bank can substitute for the imported scientific bank.
        bad=deepcopy(r)
        bad['measurements'][0]['training']=with_content_hash(dict(bad['measurements'][0]['training'],scientific_fit=True))
        with pytest.raises(ValueError):
            pre.validate_evidence(with_content_hash(bad),gate,p)
        runtime=Path(gate['root'])/'preflight/runtime.json'
        runtime.write_bytes(runtime.read_bytes()+b'corrupt')
        with pytest.raises(ValueError):
            c.evidence(gate)
    finally:
        torch.set_num_threads(threads)


def test_parent_source_bytes_and_original_process_are_bound(tmp_path,monkeypatch):
    oldpath,newpath=tmp_path/'original',tmp_path/'new'
    relative='src/hlt_classification/fixture.py'
    for root in (oldpath,newpath):
        (root/relative).parent.mkdir(parents=True)
        (root/relative).write_text('original scientific bytes')
    parent=dict(project_dir=str(oldpath),source_snapshot=dict(content_hash='a'*64))
    new=dict(content_hash='b'*64)
    checked=[]
    monkeypatch.setattr(source,'validate_source_snapshot',lambda snap,**kw:checked.append(kw))
    monkeypatch.setattr(source,'_files',lambda root:{relative} if root==oldpath else {relative}|source.ADDITIONS)
    records=source.compatible(parent,new,newpath)
    assert len(checked)==2 and all(x['require_clean'] for x in checked)
    (newpath/relative).write_text('changed matching semantics')
    with pytest.raises(ValueError,match='scientific source'):
        source.compatible(parent,new,newpath)
    (newpath/relative).write_text('original scientific bytes')
    monkeypatch.setattr(source,'_files',lambda root:{relative})
    with pytest.raises(ValueError,match='additions'):
        source.compatible(parent,new,newpath)
    monkeypatch.setattr(source,'compatible',lambda *a,**kw:records)
    parent=old.make_artifact('SPEC',root=str(oldpath),project_dir=str(oldpath),source_snapshot=parent['source_snapshot'],
        counts=old.COUNTS,nodes=old.nodes(),training_recipe=old.recipe(),prepared_sha256='c'*64)
    path=oldpath/'campaign_spec.json'
    write_immutable_json(path,parent)
    evidence=dict(spec=parent['content_hash'],prepared=parent['prepared_sha256'],
        rows=[dict(state='COMPLETE') for _ in range(9)])
    calls=[]
    def original_process(argv,**kw):
        assert kw['env']['PYTHONPATH']==str(oldpath/'src')
        assert kw['env']['PYTHONNOUSERSITE']=='1' and kw['env']['PYTHONDONTWRITEBYTECODE']=='1'
        assert kw['cwd']==str(oldpath) and argv[-1]==str(path)
        assert argv[-2]==source.PARENT_CHECK
        calls.append(argv)
        return subprocess.CompletedProcess(argv,0,'LUKA_PARENT_AUTH='+json.dumps(evidence),'')
    monkeypatch.setattr(source.subprocess,'run',original_process)
    imported,replayed=source.authenticate(path,active=new,project=newpath)
    assert replayed==parent and imported['identical_source_files']==records and len(calls)==1
    evidence['prepared']='d'*64
    with pytest.raises(ValueError,match='parent comparison'):
        source.authenticate(path,active=new,project=newpath)


def test_validation_identity_swap_and_pair_join_are_rejected(all_prepared,tmp_path):
    from hlt_classification.context_fusion.inputs import PairedCache
    spec,parent,p=fixture_spec(all_prepared,tmp_path)
    node=c.nodes()[0]
    pair=w.cache(spec,p,node,'validation')
    pair.context.identities=pair.context.identities[::-1].copy()
    with pytest.raises(ValueError,match='join'):
        PairedCache(pair.primary,pair.context)
    arrays=dict(identities=pair.context.identities,labels=pair.primary.labels,
        probabilities=np.full((len(pair),11),1/11,np.float32))
    with pytest.raises(ValueError):
        w.diagnostic(spec,'train_FUSION_U050',dict(content_hash='f'*64),arrays)

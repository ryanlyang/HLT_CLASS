"""Matched batch128 contracts and real CPU payloads; not A100 admission."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import math
import os
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.jetclass2_delphes import runner
from hlt_classification.jetclass2_delphes.batch128_recipe import recipe
from hlt_classification.luka_fullsim import alljets_batch128 as b
from hlt_classification.luka_fullsim import alljets_fusion_campaign as c, alljets_fusion_worker as w
from hlt_classification.luka_fullsim import alljets_fusion_submission as sub, alljets_fusion_preflight as pre
from hlt_classification.luka_fullsim import alljets_fusion_sync as sync, alljets_fusion_isolated_preflight as iso
from test_luka_fullsim_stage2 import dataset, prepared, config
from test_luka_fullsim_alljets import all_prepared
from test_luka_fullsim_alljets_fusion import fixture_spec, cpu_worker, TinyModel
from test_luka_fusion_isolated_preflight import local_gate

VALIDATION_STRESS = pre.validation_stress


def test_recipe_graph_matched_fresh_teachers_and_old_versions_unchanged():
    r = recipe()
    base = runner.recipe()
    assert r['batch_size'] == r['inference_batch_size'] == 128 and r['accumulation'] == 1
    changed = {'content_hash', 'contract', 'parents', 'batch_size', 'inference_batch_size'}
    assert {k:v for k,v in r.items() if k not in changed} == {k:v for k,v in base.items() if k not in changed}
    assert r['parents'] == {'original_recipe':base['content_hash']}
    scope = dict(schema_version=4)
    reg = c.registration(4)
    assert (reg['fits'], reg['reducers'], reg['jobs'], reg['comparison_parent_fits']) == (17,11,30,0)
    assert reg['imported_weights'] == reg['imported_train_banks'] == []
    tasks = {t['task_id']:t for t in c.scoped_tasks(scope)}
    seen = set()
    for name, row in tasks.items():
        assert set(row['dependencies']) <= seen
        seen.add(name)
    for n in ('DIRECT_D000', 'COARSE_U050', 'FUSION_U050'):
        assert tasks['train_'+n]['dependencies'] == ['reduce_U000']
    assert tasks['train_FINAL_DIRECT_D000']['dependencies'] == ['reduce_FUSION_D000']
    assert tasks['train_FINAL_BRIDGE_D000']['dependencies'] == ['reduce_FUSION_D000_D000']
    for v in (1,2,3):
        assert c.registration(v)['training_recipe'] == base
        assert c.registration(v)['jobs'] == 16 and c.registration(v)['imported_train_banks'] == ['U000']
    assert len(iso.sequence(scope)) == 12 and len(iso.sequence()) == 9
    assert [n['node_id'] for n in pre.probe_nodes(scope)][-1] == 'U000_probe'
    assert pre.probe_nodes(scope)[-1]['teacher'] is None


class CountingCache:
    foundation_sha256 = 'f'*64
    coordinate_name = 'D000'
    def __init__(self, role, rows=259):
        self.role = role
        self.labels = np.arange(rows, dtype=np.int64) % 11
        self.identities = np.zeros((rows,32), np.uint8)
        self.calls = []
        self.blocks = [SimpleNamespace(offsets=np.arange(rows+1)*16)]
    def __len__(self): return len(self.labels)
    def batch(self, ix):
        self.calls.append(np.asarray(ix).copy())
        return dict(features=np.ones((len(ix),17,16), np.float32),
                    vectors=np.ones((len(ix),4,16), np.float32),
                    mask=np.ones((len(ix),1,16), bool), labels=self.labels[ix])


@pytest.mark.parametrize('kd', [False,True])
def test_real_kernel_batch_updates_inference_and_no_accumulation(kd):
    train, val = CountingCache('train'), CountingCache('validation')
    node = dict(c.old.nodes()[0], teacher='teacher' if kd else None)
    q = np.full((len(train),11), 1/11, np.float32) if kd else None
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        report, _ = runner.train_kernel(TinyModel(), train, val, node=node, device='cpu',
            training_recipe=recipe(), acceptance_passes=1, teacher_probabilities=q,
            teacher_identities=train.identities if kd else None)
    finally:
        torch.set_num_threads(threads)
    assert [len(ix) for ix in train.calls] == [128,128,3]
    assert [len(ix) for ix in val.calls] == [128,128,3]
    assert report['validation_history'][0]['update'] == 3
    assert report['recipe_sha256'] == recipe()['content_hash']
    np.testing.assert_array_equal(np.sort(np.concatenate(train.calls)),np.arange(259))
    for bad in (with_content_hash(dict(runner.recipe(),batch_size=128)),
                with_content_hash(dict(recipe(),accumulation=2)),
                with_content_hash(dict(recipe(),inference_batch_size=256)),
                with_content_hash(dict(recipe(),peak_lr=6e-4))):
        with pytest.raises(ValueError):
            runner.train_kernel(TinyModel(),train,val,node=node,device='cpu',training_recipe=bad,acceptance_passes=1)


def test_real_train_and_validation_stresses_are_128_and_role_safe():
    train, val = CountingCache('train'), CountingCache('validation')
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        b.stress(TinyModel(),train,device='cpu')
        model = TinyModel()
        VALIDATION_STRESS(model,val,device='cpu',batch_size=128)
    finally:
        torch.set_num_threads(threads)
    assert all(len(ix)==128 for ix in train.calls+val.calls)
    assert all(p.grad is None for p in model.parameters())
    with pytest.raises(PermissionError):
        b.stress(model,val,device='cpu')
    with pytest.raises(PermissionError):
        VALIDATION_STRESS(model,train,device='cpu',batch_size=128)


@pytest.fixture
def gate128(local_gate, tmp_path, monkeypatch):
    old, parent, p, events, launched = local_gate
    root = tmp_path/'batch128'
    root.mkdir()
    gate = with_content_hash(dict(old, contract='LUKA_ALLJETS_FUSION_GATE/v4', schema_version=4,
        root=str(root), **c.registration(4)))
    write_immutable_json(root/'gate_spec.json', gate)
    monkeypatch.setattr(b, 'stress', pre.stress)
    def validation(model, cache, **kw):
        assert kw == dict(batch_size=128)
        events.append('validation_stress')
        VALIDATION_STRESS(model,cache,device='cpu',**kw)
    monkeypatch.setattr(pre, 'validation_stress', validation)
    monkeypatch.setattr(w, 'teacher', lambda *a:pytest.fail('Batch128 preflight must not load old teachers'))
    return gate,parent,p,events,launched


def test_full_v4_preflight_admission_real_cpu_payloads_and_30_job_spec(gate128,monkeypatch):
    gate,parent,p,events,launched = gate128
    report = pre.run(gate)
    assert launched == iso.sequence(gate)
    assert events == ['train_stress','validation_stress']*4 + ['fit']*4
    assert report['schema_version']==4 and len(report['isolated_children'])==12
    assert report['technical_teacher']=='uniform_11_class_technical_only'
    for m in report['measurements']:
        assert m['stress_batch']==m['validation_stress_batch']==128
        assert m['training']['recipe_sha256']==recipe()['content_hash']
    assert c.evidence(gate)==(parent,p,report)
    monkeypatch.setattr(sub,'accepted_jobs',lambda _:{'preflight':'123'})
    monkeypatch.setattr(c,'preflight_accounting',lambda job:dict(job_id=job,state='COMPLETED',exit_code='0:0'))
    spec=c.create_science(gate)
    assert c.validate_spec(spec)==(parent,p,report)
    assert sub.render(spec)['jobs']==30
    for key,val in [('schema_version',3),('technical_teacher','authenticated_parent_U000_train_bank')]:
        with pytest.raises(ValueError):
            pre.validate_evidence(with_content_hash(dict(report,**{key:val})),gate,p)
    bad=deepcopy(report)
    bad['measurements'][0]['stress_batch']=256
    with pytest.raises(ValueError):
        pre.validate_evidence(with_content_hash(bad),gate,p)
    bad=deepcopy(report)
    bad['parities']['FUSION_D000_D000_probe_bf16']['passed']=False
    with pytest.raises(ValueError):
        pre.validate_evidence(with_content_hash(bad),gate,p)


def test_batch128_failure_stops_before_population_or_science(gate128,monkeypatch):
    gate,_,_,events,launched=gate128
    def fail(*a,**kw): raise RuntimeError('CUDA failure fixture')
    monkeypatch.setattr(b,'stress',fail)
    with pytest.raises(RuntimeError):
        pre.run(gate)
    assert 'fit' not in events
    assert launched == iso.sequence(gate)[:2]
    assert not (Path(gate['root'])/'preflight/preflight.json').exists()


def test_batch128_keeps_runtime_admission(gate128,monkeypatch):
    gate,_,_,_,_=gate128
    original=pre.train_kernel
    def slow(*a,**kw):
        report,state=original(*a,**kw)
        return with_content_hash(dict(report,runtime_seconds=1264.)),state
    monkeypatch.setattr(pre,'train_kernel',slow)
    with pytest.raises(ValueError,match='resource admission'):
        pre.run(gate)
    assert (Path(gate['root'])/'preflight/measurement.json').is_file()
    assert not (Path(gate['root'])/'preflight/preflight.json').exists()


@pytest.mark.parametrize('version',[4,5,6])
def test_all17_fresh_scientific_fits_and_11_local_banks(all_prepared,tmp_path,monkeypatch,version):
    old,parent,p=fixture_spec(all_prepared,tmp_path)
    spec=with_content_hash(dict(old,contract=f'LUKA_ALLJETS_FUSION_SPEC/v{version}',schema_version=version,
        **{**c.registration(version),'counts':p['counts']}))
    cpu_worker(monkeypatch,p,parent)
    monkeypatch.setattr(sync,'new_model',lambda n:(torch.manual_seed(n['initialization_seed']),TinyModel())[1])
    original=w.load_bank
    banks=[]
    def read(path,**kw):
        assert Path(path).is_relative_to(Path(spec['root']))
        banks.append(kw['teacher_node'])
        return original(path,**kw)
    monkeypatch.setattr(w,'load_bank',read)
    threads=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert len(w.result_rows(spec))==17 and all(r['state']=='UNFINISHED' for r in w.result_rows(spec))
        for task in c.scoped_tasks(spec):
            receipt=w.run(spec,task['task_id'])
            assert w.completed(spec,task['task_id'])==receipt
        rows=w.result_rows(spec)
        assert all(r['state']=='COMPLETE' and r['origin']=='new' for r in rows)
        assert w.completed(spec,'complete')['result']==c.completion(spec)
        assert w.completed(spec,'train_FUSION_U050')['dependencies'].keys()=={'reduce_U000'}
        for n in c.scoped_nodes(spec):
            report=load_json(Path(spec['root'])/('train_'+n['node_id'])/'training_report.json')
            assert report['recipe_sha256']==spec['training_recipe']['content_hash']
            assert (report['passes']==30 if version==6 else 60<=report['passes']<=100)
            assert report['validation_history'][-1]['update']==report['passes']*math.ceil(p['counts']['train']/128)
        w.print_results(spec)
        assert 'U000' in banks and 'FUSION_D000_D000' in banks
        # A technical or batch256 report cannot become a scientific teacher.
        node=c.scoped_nodes(spec)[0]
        report=load_json(Path(spec['root'])/'train_M0HLT/training_report.json')
        for changes in (dict(recipe_sha256=runner.recipe()['content_hash']),dict(scientific_fit=False,acceptance_only=True)):
            with pytest.raises(ValueError):
                w.validate_training(spec,node,with_content_hash(dict(report,**changes)))
        target=Path(spec['root'])/'train_FINAL_BRIDGE_D000/validation_predictions.npz'
        target.write_bytes(target.read_bytes()+b'corrupt')
        with pytest.raises(ValueError): w.completed(spec,'complete')
    finally:
        torch.set_num_threads(threads)


def test_exact_v4_plan_and_no_older_authorization(tmp_path,monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as exact
    gate=c.artifact('GATE',version=4,root=str(tmp_path),project_dir=str(tmp_path/'source'),**c.registration(4))
    monkeypatch.setattr(c,'validate_gate',lambda _:None)
    monkeypatch.setattr(sub,'check_site',lambda _:None)
    calls=[]
    def submit(args,**kw):
        calls.append(args)
        return subprocess.CompletedProcess(args,0,'999','')
    monkeypatch.setattr(exact.subprocess,'run',submit)
    plan=sub.submit(gate)
    assert not calls and plan['jobs']==1
    argv=plan['commands'][0]['command']
    assert '--time=08:00:00' in argv and '--job-name=lukab_preflight' in argv
    with pytest.raises(PermissionError):
        sub.submit(gate,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    ledger=sub.submit(gate,execute=True,plan_hash=plan['content_hash'],authorization=c.authorization(gate))
    assert sub.submit(gate,execute=True,plan_hash=plan['content_hash'],authorization=c.authorization(gate))==ledger
    assert len(calls)==1
    spec=c.artifact('SPEC',version=4,root=str(tmp_path/'science'),project_dir=gate['project_dir'],
        train_minutes=1000,reduce_minutes=50,**c.registration(4))
    plan=sub.render(spec)
    assert plan['jobs']==30
    named={row['task_id']:row for row in plan['commands']}
    for n in ('DIRECT_D000','COARSE_U050','FUSION_U050'):
        assert '--dependency=afterok:${JOB_reduce_U000}' in named['train_'+n]['command']
    assert all('lukab' in ' '.join(row['command']) for row in plan['commands'])


def test_source_allowlist_and_runtime_files_exist():
    root=Path(__file__).resolve().parents[1]
    for name in b.REQUIRED: assert (root/name).is_file(),name
    changed=subprocess.run(['git','diff','--name-only',b.FAILED_COMMIT,'--','src'],cwd=root,
        check=True,capture_output=True,text=True).stdout.splitlines()
    from hlt_classification.luka_fullsim import alljets_report_recovery as report_recovery
    from hlt_classification.luka_fullsim import alljets_30epoch as thirty
    assert set(changed)<=b.CHANGES|b.ADDITIONS|report_recovery.CHANGES|report_recovery.ADDITIONS|thirty.CHANGES|thirty.ADDITIONS|thirty.FULLSTAT
    # Production model, shared stress, original recipe, split/cache bytes untouched.
    assert 'src/hlt_classification/luka_fullsim/alljets_fusion_sync.py' not in changed
    assert 'src/hlt_classification/context_fusion/worker.py' not in changed
    helper=(root/'scripts/queue_luka_alljets_batch128.sh').read_text()
    for text in ('create-batch128','create-science','--execute','PYTHONNOUSERSITE=1','${CONDA_PREFIX}/lib'):
        assert text in helper


@pytest.fixture
def failed128(tmp_path,monkeypatch):
    root=tmp_path/'failed'
    (root/'preflight').mkdir(parents=True)
    parent=tmp_path/'parent.json'
    write_immutable_json(parent,dict(root='parent'))
    old=c.artifact('GATE',version=3,root=str(root),project_dir=str(tmp_path/'old_source'),
        source_commit=b.FAILED_COMMIT,source_snapshot=dict(git_commit=b.FAILED_COMMIT),
        parent_import=dict(spec=file_ref(parent)),prepared_sha256='e'*64,recovery={},**c.registration(3))
    write_immutable_json(root/'gate_spec.json',old)
    write_immutable_json(root/'preflight/runtime.json',dict(job_id=b.FAILED_JOB))
    write_immutable_json(root/'preflight/failure.json',dict(phase='stress',node=b.ORDER[0],exit_code=1,science_admissible=False))
    (root/f'preflight-{b.FAILED_JOB}.out').write_text('original failed log')
    monkeypatch.setattr(b,'compatible',lambda *a:dict(fixture=True))
    def original(argv,**kw):
        assert kw['cwd']==old['project_dir']
        assert kw['env']['PYTHONPATH']==str(Path(old['project_dir'])/'src')
        assert argv[-2:]==[b.ORIGINAL_CHECK,str(root/'gate_spec.json')]
        return subprocess.CompletedProcess(argv,0,'LUKA_BATCH128_AUTH='+json.dumps(
            dict(gate=old['content_hash'],jobs=dict(preflight=b.FAILED_JOB))),'')
    monkeypatch.setattr(b.subprocess,'run',original)
    monkeypatch.setattr(b,'source',lambda *a:dict(content_hash='a'*64))
    monkeypatch.setattr(c,'protect',lambda *a:None)
    monkeypatch.setattr(b,'failed_accounting',lambda job:dict(job_id=job,state='FAILED',exit_code='1:0'))
    return root,old


def test_batch128_original_source_auth_fresh_root_and_immutable_reuse(failed128,tmp_path):
    root,old=failed128
    before=(root/'gate_spec.json').read_bytes()
    args=dict(failed_gate=root/'gate_spec.json',root=tmp_path/'new',project=tmp_path/'source',commit='a'*40)
    gate=b.create(**args)
    assert gate['schema_version']==4 and gate['jobs']==30
    assert gate['prepared_sha256']==old['prepared_sha256']
    assert c.validate_gate(gate)=={'root':'parent'}
    assert (root/'gate_spec.json').read_bytes()==before
    assert not (Path(gate['root'])/'prepared').exists()
    with pytest.raises(FileExistsError): b.create(**args)
    with pytest.raises(PermissionError,match='overlaps'):
        b.create(**dict(args,root=root/'bad'))
    (root/f'preflight-{b.FAILED_JOB}.out').write_text('tampered')
    with pytest.raises(ValueError): c.validate_gate(gate)


def test_batch128_refuses_live_failure_job_and_wrong_source(failed128,tmp_path,monkeypatch):
    root,old=failed128
    monkeypatch.setattr(b,'failed_accounting',lambda job:dict(job_id=job,state='RUNNING',exit_code='0:0'))
    with pytest.raises(ValueError,match='accounting'):
        b.create(failed_gate=root/'gate_spec.json',root=tmp_path/'new',project=tmp_path/'source',commit='a'*40)
    assert not (tmp_path/'new').exists()
    (root/'gate_spec.json').write_text(json.dumps(with_content_hash(dict(old,source_commit='f'*40))))
    with pytest.raises(ValueError,match='5cab1830'):
        b.authenticate(root/'gate_spec.json',{},tmp_path/'source')


def test_batch128_source_guard_preserves_models_and_caches(tmp_path,monkeypatch):
    before,after=tmp_path/'before',tmp_path/'after'
    protected='src/hlt_classification/luka_fullsim/alljets_fusion_sync.py'
    names=b.CHANGES|{protected}
    for root in (before,after):
        for name in names:
            (root/name).parent.mkdir(parents=True,exist_ok=True)
            (root/name).write_text('unchanged')
    monkeypatch.setattr(b,'validate_source_snapshot',lambda *a,**kw:None)
    monkeypatch.setattr(b,'_files',lambda root:names if root==before else names|b.ADDITIONS)
    old=dict(project_dir=str(before),source_snapshot={})
    b.compatible(old,{},after)
    (after/protected).write_text('changed')
    with pytest.raises(ValueError,match='model/data/source'):
        b.compatible(old,{},after)


@pytest.mark.parametrize('same',[True,False])
def test_true128_production_cuda_small_support_fresh_process(same):
    pytest.importorskip('weaver')
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    # Real production model/three full128 updates, NOT full-support A100 evidence.
    program='''
import sys
from types import SimpleNamespace
import numpy as np
from hlt_classification.context_fusion.inputs import PairedCache
from hlt_classification.luka_fullsim import alljets_batch128 as b, alljets_fusion_sync as sync, alljets_fusion_campaign as c
from test_jetclass2_dzfix_fusion_offload import inputs
raw,other=inputs()
class Cache:
    role='train'; foundation_sha256='f'*64; coordinate_name='D000'; nbytes=1000
    def __init__(self,value):
        self.value=value; self.labels=value['labels']
        self.identities=np.arange(128,dtype=np.uint8).reshape(4,32)
        self.blocks=[SimpleNamespace(offsets=np.arange(5)*value['mask'].shape[-1])]
    def __len__(self): return 4
    def batch(self,ix):
        assert len(ix)==128
        return {k:v[ix] for k,v in self.value.items()}
same=sys.argv[1]=='True'
a=Cache(raw); train=PairedCache(a,a if same else Cache(other))
node=next(n for n in c.nodes() if n['node_id']==('FUSION_D000_D000' if same else 'FUSION_U050'))
sync.old.validate_offload_stats(b.stress(sync.new_model(node),train),calls=3)
print('TRUE128_PRODUCTION_STRESS_PASS')
'''
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root/'tests'),os.environ.get('PYTHONPATH','')]),
        CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_LAUNCH_BLOCKING='0',PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([sys.executable,'-B','-c',program,str(same)],env=env,cwd=root,
        capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'TRUE128_PRODUCTION_STRESS_PASS' in result.stdout

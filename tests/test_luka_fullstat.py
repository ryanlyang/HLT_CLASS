"""FullStat fixtures exercise real ROOT/cache bytes, not remote acceptance."""
from copy import deepcopy
from pathlib import Path
import subprocess

import awkward as ak
import numpy as np
import pytest
import torch
import uproot

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, deterministic_npz_bytes, load_json, load_npz_arrays,
    sha256_file, with_content_hash, write_immutable_json,
)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.jetclass2_delphes.schema import SIGNAL_IDS
from hlt_classification.luka_fullsim.particles import FIELDS
from hlt_classification.luka_fullstat import data, cache, campaign as c, submission as sub, worker as w, preflight as pre
from hlt_classification.luka_fullstat.contracts import SOURCE, COORDINATES, ORDINARY, ROLES, artifact, identity
from hlt_classification.luka_fullstat.recipe import recipe, learning_rate


def write_root(path, labels):
    n = len(labels)
    arrays = dict(jet_label=np.asarray(labels,np.int32), hlt_matched=np.ones(n,bool),
        jet_nparticles=np.full(n,3,np.int32), hlt_jet_nparticles=np.full(n,2,np.int32), jet_pt=np.full(n,199.,np.float32))
    schema = dict(jet_label='int32',hlt_matched='bool',jet_nparticles='int32',hlt_jet_nparticles='int32',jet_pt='float32')
    for prefix, count in [('part_',3), ('hlt_part_',2)]:
        raw = np.zeros((count,len(FIELDS)))
        raw[:,0] = np.arange(count)+1.
        raw[:,3] = raw[:,0]+.1
        raw[:,4:6] = 1
        raw[:,10:14] = [.02,-.03,.004,.005]
        raw[0,12] = 0
        for j,field in enumerate(FIELDS):
            arrays[prefix+field] = ak.Array([raw[:,j].tolist() for _ in labels])
            schema[prefix+field] = 'var * float32'
    path.parent.mkdir(parents=True,exist_ok=True)
    with uproot.recreate(path) as handle:
        handle.mktree('tree',schema)
        handle['tree'].extend(arrays)


@pytest.fixture(scope='module')
def fullstat(tmp_path_factory):
    root = tmp_path_factory.mktemp('fullstat')
    container = root/'snapshot'
    write_root(container/'raw/jetclass2/train_higgs2p/merged/ntuple_merged_0.root', list(SIGNAL_IDS)*16+[161]*16)
    write_root(container/'raw/jetclass2sb/QCD_PT-200to300/merged/ntuple_merged_0.root', [161]*160)
    write_root(container/'raw/jetclass2sb/QCD_PT-300to470/merged/ntuple_merged_0.root', [187]*80)
    (container/'source.txt').write_text(SOURCE+'\n')
    (container/'source.sha256').write_text(''.join(f'{sha256_file(p)}  {p.relative_to(container/"raw").as_posix()}\n'
        for p in sorted((container/'raw').rglob('*.root'))))
    counts = dict(train=264,validation=66,final_test=66,unused=4)
    foundation = data.build(container,root/'foundation',gate_hash='a'*64,counts=counts,expected_files=3)
    return container,root/'foundation',foundation


@pytest.fixture(scope='module')
def durable(fullstat,tmp_path_factory):
    container,foundation_root,foundation = fullstat
    output = tmp_path_factory.mktemp('fullstat_cache')
    for index in range(3):
        cache.prepare_file(container,foundation_root,output,index,workers=1)
    manifest = cache.assemble(foundation_root,output,gate_hash='a'*64)
    authenticated = cache.metadata(foundation_root,output,gate_hash='a'*64)
    return output,authenticated


def test_recipe_schedule_and_legacy_unchanged():
    from hlt_classification.jetclass2_delphes.campaign import recipe as legacy, learning_rate as old_lr
    assert legacy()['minimum_passes']==60 and old_lr(20)==3e-4
    assert recipe()['minimum_passes']==recipe()['maximum_passes']==20
    assert recipe()['batch_size']==recipe()['inference_batch_size']==128
    assert recipe(256)['batch_size']==256
    assert learning_rate(1)==1.5e-4 and learning_rate(2)==3e-4 and learning_rate(20)==1.5e-5
    assert learning_rate(11)==pytest.approx((3e-4+1.5e-5)/2)
    for x in (0,21,np.nan,np.inf):
        with pytest.raises(ValueError): learning_rate(x)
    for x in (1,64,True):
        with pytest.raises(ValueError): recipe(x)
    assert len(c.nodes())==9 and len(c.tasks())==16
    assert sum(t['kind']=='reduce' for t in c.tasks())==5
    assert all(n['context_coordinate'] is None for n in c.nodes())
    endpoints=[n for n in c.nodes() if n['coordinate']=='D000']
    assert len({(n['initialization_seed'],n['sampler_seed']) for n in endpoints})==1


def test_foundation_selection_ratios_and_sealed_memberships(fullstat):
    _,root,f = fullstat
    replay, payloads=data.load(root)
    assert replay==f and f['raw_entries']==416
    assert f['files'][0]['cutflow']['excluded_qcd_source']==16
    assert f['policy']['additional_pt_cut'] is None
    assert not f['policy']['event_independence_verified']
    assert sum(r['cutflow']['selected'] for r in f['files'])==400  # includes pT=199; no extra cut
    assert all(sum(f['quotas'][r].values())==f['counts'][r] for r in ROLES)
    for role in ORDINARY:
        labels=np.concatenate([a['labels'][data.ordinary_indices(a,role)] for a in payloads])
        assert set(labels)==set(range(11)) and .58 < np.mean(labels==0) < .62
        q=f['quotas'][role]
        assert abs(q['QCD_PT-200to300']-2*q['QCD_PT-300to470'])<=2
    all_ids=[]
    for record,a in zip(f['files'],payloads):
        by_role=[set(np.flatnonzero(a['roles']==j)) for j in range(4)]
        assert not any(by_role[i]&by_role[j] for i in range(4) for j in range(i))
        all_ids.extend(identity(record,e) for e in np.flatnonzero(a['roles']!=255))
    assert len(set(all_ids))==400
    for role in ('final_test','unused','test'):
        with pytest.raises(PermissionError): data.ordinary_indices({},role)
        with pytest.raises(PermissionError): cache.load('absent',None,role=role,coordinate='D000',max_ram_bytes=1)


def test_foundation_reads_scalars_only(fullstat,tmp_path,monkeypatch):
    container,_,f=fullstat
    original=uproot.behaviors.TBranch.HasBranches.iterate
    def guarded(tree,expressions,*args,**kwargs):
        assert set(expressions)==set(data.SCALARS)
        return original(tree,expressions,*args,**kwargs)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches,'iterate',guarded)
    again=data.build(container,tmp_path/'f',gate_hash='a'*64,counts=f['counts'],expected_files=3)
    assert again==f


def test_particle_reader_never_requests_test_or_unused(fullstat,monkeypatch):
    container,root,f=fullstat
    _,members=data.load(root)
    original=uproot.behaviors.TBranch.HasBranches.arrays
    lookup={r['path']:a for r,a in zip(f['files'],members)}
    calls=[]
    def guarded(tree,expressions,*args,**kwargs):
        name=Path(tree.file.file_path).relative_to(container/'raw').as_posix()
        roles=lookup[name]['roles'][kwargs['entry_start']:kwargs['entry_stop']]
        assert not np.isin(roles,(2,3)).any()
        calls.append(name)
        return original(tree,expressions,*args,**kwargs)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches,'arrays',guarded)
    rows=[]
    for record,a in zip(f['files'],members):
        rows.extend(cache.read_selected_chunk(container/'raw'/record['path'],record,a,0,record['entries']))
    assert len(rows)==330 and calls


def test_serial_parallel_bytes_and_shared_miniature(fullstat,durable,tmp_path):
    container,root,f=fullstat
    output,auth=durable
    parallel=cache.prepare_file(container,root,tmp_path,0,workers=2)
    serial=load_json(output/'files/00/receipt.json')
    assert parallel['chunks']==serial['chunks']
    for role in ORDINARY:
        ids=None
        for co in COORDINATES:
            bundle=cache.load(output,auth,role=role,coordinate=co,max_ram_bytes=2**30)
            mini=cache.load(output,auth,role=role,coordinate=co,max_ram_bytes=2**30,miniature=True)
            assert len(bundle)==f['counts'][role]
            if ids is None: ids=bundle.identities
            np.testing.assert_array_equal(ids,bundle.identities)
            assert set(bundle.labels)==set(range(11))
            assert bundle.batch(np.array([0,1],np.int64))['mask'].shape==(2,1,16)
            np.testing.assert_array_equal(ids,mini.identities)
    with pytest.raises(MemoryError): cache.load(output,auth,role='train',coordinate='D000',max_ram_bytes=1)


def test_tampered_manifest_and_block_rejected(fullstat,durable,tmp_path):
    output,auth=durable
    chunk=deepcopy(auth[0]['chunks'][0]); co='D000'
    ref=chunk['views'][co]
    a=load_npz_arrays(output/ref['path']); a['labels'][0]+=1
    path=tmp_path/'wrong.npz'
    atomic_publish_bytes(path,deterministic_npz_bytes(a))
    chunk['views'][co]={**ref,**file_ref(path,root=tmp_path)}
    with pytest.raises(ValueError,match='identity/label'):
        cache.read_block(tmp_path,chunk,co,auth[1]['files'][chunk['file_index']],auth[2][chunk['file_index']])


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.linear=torch.nn.Linear(17,11)
    def forward(self,features,vectors,mask):
        return self.linear((features*mask).sum(-1)/mask.sum(-1).clamp_min(1))


def test_twenty_actual_epochs_lr_and_invalid_recipe(durable):
    from hlt_classification.jetclass2_delphes.runner import train_kernel
    output,auth=durable
    bundles=[cache.load(output,auth,role=r,coordinate='D000',max_ram_bytes=2**30) for r in ORDINARY]
    node=c.nodes()[0]
    torch.set_num_threads(1)
    report,state=train_kernel(Tiny(),*bundles,node=node,device='cpu',training_recipe=recipe())
    assert report['passes']==20 and report['scientific_fit'] and len(state)==2
    assert [h['learning_rate'] for h in report['validation_history']]==[learning_rate(i) for i in range(1,21)]
    assert report['validation_history'][-1]['update']==60
    bad=with_content_hash({**recipe(),'maximum_passes':21})
    with pytest.raises(ValueError,match='exact registered'):
        train_kernel(Tiny(),*bundles,node=node,device='cpu',training_recipe=bad)


def fixture_spec(tmp_path,durable):
    output,auth=durable
    root=tmp_path/'science'; root.mkdir()
    return artifact('SPEC',parents={},root=str(root),project_dir=str(tmp_path/'project'),source_commit='b'*40,
        cache_root=str(output),prepared_sha256=auth[0]['content_hash'],train_minutes=120,reduce_minutes=20,
        **{**c.policy(),'counts':auth[1]['counts']})


def test_entire_nine_fit_graph_and_predictions(durable,tmp_path,monkeypatch):
    from hlt_classification.context_fusion import inputs
    spec=fixture_spec(tmp_path,durable)
    monkeypatch.setattr(c,'validate_spec',lambda _: ({},durable[1],{}))
    monkeypatch.setattr(w,'execution',lambda *args:'12345')
    monkeypatch.setattr(inputs,'new_model',lambda node:Tiny())
    actual=w.fit_or_reduce
    monkeypatch.setattr(w,'fit_or_reduce',lambda *args:actual(*args,device='cpu'))
    torch.set_num_threads(1)
    for task in c.tasks():
        row=w.run(spec,task['task_id'])
        assert row==w.completed(spec,task['task_id'])
    rows=w.result_rows(spec)
    assert len(rows)==9 and all(r['selected_epoch'] in range(1,21) for r in rows)
    assert all(r['recovery'] is not None for r in rows)
    assert w.run(spec,'complete')['result']==dict(fits=9,reducers=5,final_test_accessed=False)
    with pytest.raises(ValueError): w.run(spec,'final_test')
    predictions=Path(spec['root'])/'train_COARSE_D000/validation_predictions.npz'
    assert load_npz_arrays(predictions)['probabilities'].shape==(66,11)
    with predictions.open('ab') as handle: handle.write(b'corrupt')
    with pytest.raises(ValueError): w.completed(spec,'train_COARSE_D000')


@pytest.mark.parametrize('gate',[True,False])
def test_exact_oscar_submission_and_idempotence(tmp_path,monkeypatch,gate):
    spec=artifact('GATE' if gate else 'SPEC',parents={},root=str(tmp_path),project_dir='/oscar/source/pinned',
                  train_minutes=900,reduce_minutes=60,**c.policy())
    monkeypatch.setattr(c,'validate_gate',lambda _:None)
    monkeypatch.setattr(c,'validate_spec',lambda _:None)
    monkeypatch.setattr(sub,'check_site',lambda _:None)
    calls=[]
    def scheduler(args,**kwargs):
        assert not any('${JOB_' in a for a in args)
        assert not any(k.startswith(('SBATCH_','SLURM_')) for k in kwargs['env'])
        calls.append(args)
        return subprocess.CompletedProcess(args,0,str(1000+len(calls))+';slurmctld\n','')
    monkeypatch.setattr(sub.subprocess,'run',scheduler)
    plan=sub.submit(spec)
    assert len(plan['commands'])==(4 if gate else 16) and not calls
    assert all('--account=default' in r['command'] for r in plan['commands'])
    if gate:
        assert '--array=0-15%4' in plan['commands'][1]['command']
        assert '--partition=batch' in plan['commands'][1]['command']
        assert '--gres=gpu:l40s:1' in plan['commands'][-1]['command']
    with pytest.raises(PermissionError):sub.submit(spec,execute=True,plan_hash='0'*64,authorization=c.AUTHORIZATION)
    ledger=sub.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert sub.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)==ledger
    assert len(calls)==len(plan['commands'])


def test_interrupted_submission_is_not_retried(tmp_path,monkeypatch):
    spec=artifact('GATE',parents={},root=str(tmp_path),project_dir='/oscar/source/pinned',**c.policy())
    monkeypatch.setattr(c,'validate_gate',lambda _:None)
    monkeypatch.setattr(sub,'check_site',lambda _:None)
    plan=sub.submit(spec)
    def fail(*args,**kwargs):raise RuntimeError('reply lost')
    monkeypatch.setattr(sub.subprocess,'run',fail)
    with pytest.raises(RuntimeError):sub.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    with pytest.raises(PermissionError,match='interrupted'):
        sub.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)


def test_row_split_acknowledgement_required_before_any_write(tmp_path):
    with pytest.raises(PermissionError):
        c.create(tmp_path/'raw',tmp_path/'out',tmp_path/'cache',project=tmp_path/'project',commit='a'*40,quota_gib=400)
    assert not (tmp_path/'out').exists()


def test_role_specific_stress_no_validation_labels(durable):
    output,auth=durable
    for role in ORDINARY:
        bundle=cache.load(output,auth,role=role,coordinate='U000',max_ram_bytes=2**30)
        assert pre.stress(Tiny(),bundle,batch=8,training=role=='train',device='cpu')==auth[0]['summaries'][role]['U000']['maximum']
    with pytest.raises(PermissionError):pre.stress(Tiny(),bundle,batch=8,training=True,device='cpu')


def test_validation_longer_than_training_has_separate_inference_stress():
    from hlt_classification.jetclass2_delphes.cache import RamCache, RamBlock
    for role,length in [('train',3),('validation',27)]:
        block=RamBlock(0,np.array([0,length],np.int64),np.ones((length,17),np.float32),
            np.ones((length,4),np.float32),np.zeros((1,32),np.uint8),np.array([999],np.int64))
        bundle=RamCache([block],role=role,coordinate_name='U000',foundation_sha256='f'*64)
        # Invalid stored labels are deliberately unusable as stress targets.
        # The stress extracts inputs only, training zeros / inference no targets.
        assert pre.stress(Tiny(),bundle,batch=8,training=role=='train',device='cpu')==length


def test_full_preflight_evidence_and_rejection(durable,tmp_path,monkeypatch):
    """Execute genuine local model/bank/phase logic; mock ONLY remote hardware/site.

    This is a unit test, never an admissible real Oscar acceptance receipt.
    """
    from hlt_classification.context_fusion import inputs
    from hlt_classification.jetclass2_delphes import acceptance, execution, model, runner
    from hlt_classification.jetclass2_delphes.contracts import artifact as ka
    from hlt_classification.luka_fullsim import preflight_v2
    import time
    # Tiny CPU fixture steps can fall below Windows monotonic clock resolution.
    monkeypatch.setattr(runner.time,'monotonic',time.perf_counter)
    output,auth=durable
    gate=artifact('GATE',parents={},root=str(tmp_path),cache_root=str(output),source_snapshot={'fixture':True},
                  project_dir=str(tmp_path/'project'),source_commit='b'*40,
                  **{**c.policy(),'counts':auth[1]['counts']})
    auth=list(deepcopy(auth))
    auth[0]=with_content_hash({**auth[0],'parents':{'gate':gate['content_hash'],'foundation':auth[1]['content_hash']}})
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    monkeypatch.setattr(c,'validate_gate',lambda _:None)
    monkeypatch.setattr(c,'authenticated_cache',lambda _:auth)
    monkeypatch.setattr(execution,'allocation',lambda site:('12345',gate['cpus'],gate['memory_mb']))
    monkeypatch.setattr(execution,'gpu_identity',lambda:{'name':'TEST FIXTURE L40S','total_memory_bytes':48*2**30})
    monkeypatch.setattr(model,'installed_environment',lambda:{'test_fixture':True})
    monkeypatch.setattr(inputs,'new_model',lambda node:Tiny())
    monkeypatch.setattr(acceptance,'installed_parity',lambda *a,**k:ka('WEAVER_PARITY',model=model.model_contract(),
        device='cuda',passed=True,forward_and_feature_and_parameter_gradients=True,final_test_accessed=False))
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    phase=preflight_v2.PhaseLog
    monkeypatch.setattr(preflight_v2,'PhaseLog',lambda output,sample:phase(output,lambda:{'fixture':True}))
    monkeypatch.setattr(preflight_v2,'memory_snapshot',lambda _:dict(peak_rss_bytes=2**30,gpu_peak_bytes=2**30))
    kernel,predict,stress=runner.train_kernel,runner.predict,pre.stress
    monkeypatch.setattr(runner,'train_kernel',lambda *a,**k:kernel(*a,**{**k,'device':'cpu'}))
    monkeypatch.setattr(runner,'predict',lambda *a,**k:predict(*a,**{**k,'device':'cpu'}))
    monkeypatch.setattr(pre,'stress',lambda *a,**k:stress(*a,**{**k,'device':'cpu'}))
    result=pre.run(gate)
    pre.validate_evidence(result,gate,auth[0])
    assert len(result['measurements'])==7 and result['full_population_training_measured'] is False
    assert result['train_minutes']>=30 and result['reduce_minutes']>=15
    for field,value in [('final_test_accessed',True),('full_population_training_measured',True),
                        ('train_minutes',1),('gpu_peak_bytes',48*2**30),('outputs',[])]:
        bad=with_content_hash({**result,field:value})
        with pytest.raises((ValueError,KeyError)):pre.validate_evidence(bad,gate,auth[0])
    bad=deepcopy(result); bad['measurements'][1]['maxima']['validation']+=1
    with pytest.raises(ValueError):pre.validate_evidence(with_content_hash(bad),gate,auth[0])
    # Exercise real admission/spec/ledger binding with completed fixture accounting.
    from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger
    from hlt_classification.scouting.hcwdl_exact_dag_submission import _resolved
    write_immutable_json(tmp_path/'gate_spec.json',gate)
    plan=sub.render(gate)
    jobs=dict(foundation='100',prepare='101',assemble='102',preflight='12345')
    ledger=build_submission_ledger(campaign_spec_sha256=gate['content_hash'],jobs=jobs,
        commands={row['task_id']:_resolved(row,jobs) for row in plan['commands']},dry_run=False)
    write_immutable_json(tmp_path/'gate_submission/submission_ledger.json',ledger)
    job_ids=['100','102','12345']+['101_'+str(i) for i in range(16)]
    monkeypatch.setattr(c.subprocess,'run',lambda args,**kwargs:subprocess.CompletedProcess(args,0,
        ''.join(job+'|COMPLETED|0:0|\n' for job in job_ids),''))
    spec=c.admit(gate)
    assert c.validate_spec(spec)[2]==result
    assert len(sub.render(spec)['commands'])==16
    with pytest.raises(FileExistsError):c.admit(gate)
    with (tmp_path/'preflight/U000.pt').open('ab') as handle:handle.write(b'corrupt')
    with pytest.raises(ValueError):pre.validate_evidence(result,gate,auth[0])


def test_rehashed_metadata_cannot_hide_missing_views_or_wrong_population(fullstat,durable):
    _,_,foundation=fullstat
    gate=artifact('GATE',parents={},counts=foundation['counts'],transfer=foundation['transfer'],
        expected_raw_rows=foundation['raw_entries'],source_records=[{k:r[k] for k in ('path','sha256','category')}
        for r in foundation['files']])
    # The gate parent is authenticated independently of the payload's own hash.
    bound=with_content_hash({**foundation,'parents':dict(gate=gate['content_hash'])})
    c.validate_foundation(gate,bound)
    bad=with_content_hash({**bound,'raw_entries':bound['raw_entries']+1})
    with pytest.raises(ValueError):c.validate_foundation(gate,bad)
    output,auth=durable
    row=load_json(output/'files/00/receipt.json')
    bad=deepcopy(row); bad['chunks'].pop()
    with pytest.raises(ValueError):cache.file_metadata(with_content_hash(bad),foundation,auth[2][0])
    bad=deepcopy(row); bad['chunks'][0]['views']['D000']['resident_bytes']+=1
    with pytest.raises(ValueError):cache.file_metadata(with_content_hash(bad),foundation,auth[2][0])


def test_missing_gate_jobs_and_wrong_scheduler_fail_closed(tmp_path,monkeypatch):
    gate=artifact('GATE',parents={},root=str(tmp_path),project_dir='/oscar/pinned',**c.policy())
    monkeypatch.setattr(c,'gate_ledger',lambda _:dict(jobs=dict(foundation='1',prepare='2',assemble='3',preflight='4')))
    def accounting(args,**kwargs):
        return subprocess.CompletedProcess(args,0,'1|COMPLETED|0:0|\n3|COMPLETED|0:0|\n4|COMPLETED|0:0|\n','')
    monkeypatch.setattr(c.subprocess,'run',accounting)
    with pytest.raises(PermissionError,match='Every exact gate'):c.accounting(gate)
    monkeypatch.setattr(sub.subprocess,'run',lambda args,**kwargs:subprocess.CompletedProcess(args,0,'ClusterName = tigris\n',''))
    with pytest.raises(PermissionError,match='Oscar cluster'):sub.check_site(sub.render(gate))

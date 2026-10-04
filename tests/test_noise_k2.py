"""Synthetic local semantics and queue safety; never remote acceptance evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hlt_classification.data.cache_contracts import load_json, sha256_file, with_content_hash
from hlt_classification.noise_k2 import contracts as c, views, campaign as q, data, runtime as run
from hlt_classification.cms2jc2_response.bridge import Particles, p4_from_coordinates
from hlt_classification.cms_proxy_ladder.inputs import build_inputs
from hlt_classification.jetclass2_delphes.concat_k2_campaign import artifact as donor_artifact
from hlt_classification.jetclass2_delphes.concat_k2_views import view_contract as donor_views
from hlt_classification.scouting.hcwdl_fullcard_salience_contracts import CANDIDATES
from test_jetclass2_dzfix_fusion_chain import make_cache


def physical(n, *, unknown=False):
    category = np.arange(n, dtype=np.int8) % 6 if unknown else np.zeros(n, np.int8)
    charge = np.where(np.isin(category, (0,3,4)), 1, 0).astype(np.int8)
    valid = np.repeat((charge != 0)[:,None],4,axis=1)
    tracking = np.tile([.3,-.2,.01,.02], (n,1))*valid
    return Particles(p4_from_coordinates(np.arange(n)+1., np.linspace(-.2,.3,n), np.linspace(-3.13,3.13,n), .1),
        charge, category, tracking, valid, tuple(map(str,range(n))))


def rehash(value, **fields):
    return with_content_hash({k:v for k,v in dict(value,**fields).items() if k != "content_hash"})


def donor(path, candidate=CANDIDATES[1]):
    source = donor_artifact("SOURCE_IMPORT", selected_candidate=candidate, salience_formula_only=True,
        assignments_imported=False, models_imported=[], final_test_accessed=False)
    f = donor_artifact("FOUNDATION_SPEC", candidate=candidate, views=donor_views(candidate),
        source_import_sha256=source["content_hash"], final_test_accessed=False)
    value = donor_artifact("CAMPAIGN_SPEC", foundation=f, source_import=source, source_commit="d"*40, final_test_accessed=False)
    c.publish(path,value)
    return path,sha256_file(path)


def toy_spec(tmp_path):
    settings=c.registration()
    shards=[dict(index=0,role="train",source={"shard_id":"s0"}),dict(index=1,role="validation",source={"shard_id":"s1"})]
    return c.artifact("CAMPAIGN_SPEC", **settings, candidate=CANDIDATES[0],
        campaign_root=str(tmp_path/"campaign"), project_dir=str(tmp_path/"project"), source_commit="a"*40,
        shards=shards,tasks=q.graph(shards,settings["nodes"]),manifest_sha256="b"*64)


@pytest.mark.parametrize("candidate",CANDIDATES)
@pytest.mark.parametrize("nh,no",[(1,3),(2,3),(2,6),(3,2)])
def test_physical_exact_solver_retention_and_every_rung(candidate,nh,no):
    p,o=physical(nh,unknown=True),physical(no,unknown=True)
    m,base=views.match(p,o,candidate)
    np.testing.assert_array_equal(m,views.match(p,o,candidate,reference=True)[0])
    kept=sorted(sorted(range(no),key=lambda j:(-int(base['offline_salience'][j]),j))[:2*nh])
    assert sorted(m[m>=0].tolist()) == kept
    for coordinate in views.STRENGTHS:
        v=views.build_view(p,coordinate=coordinate,offline=o,identity='a'*64,candidate=candidate,mapping=m)
        assert len(v)==3*nh
        for key in ('p4','charge','category','tracking','valid'):
            np.testing.assert_array_equal(getattr(v,key)[::3],getattr(p,key))
        assert np.isfinite(build_inputs(v,capacity=32).features).all()
        if coordinate=='D100':
            for owner in range(nh):
                for slot in range(2):
                    expected,idx=(o,m[owner,slot]) if m[owner,slot]>=0 else (p,owner)
                    for key in ('p4','charge','category','tracking','valid'):
                        np.testing.assert_array_equal(getattr(v,key)[3*owner+slot+1],getattr(expected,key)[idx])


def test_tracking_validity_unknown_mm_and_deployment_no_privileged_access():
    p=physical(6,unknown=True)
    o=physical(6,unknown=True)
    valid=o.valid.copy(); valid[0,2:]=False
    tracking=o.tracking.copy(); tracking[~valid]=0
    o=Particles(o.p4,o.charge,o.category,tracking,valid,o.keys)
    m,_=views.match(p,o,CANDIDATES[0])
    for coord in views.STRENGTHS:
        v=views.build_view(p,coordinate=coord,offline=o,identity='f'*64,candidate=CANDIDATES[0],mapping=m)
        x=build_inputs(v,capacity=32)
        assert x.features[0,11] == pytest.approx(np.tanh(.3))
        assert np.all(x.features[v.category==5,6:11]==0)
        assert np.all(v.tracking[~v.valid]==0)
    class Forbidden:
        def __getattribute__(self,name): raise AssertionError('privileged read')
    for coord,copies in [('HLT_X1',1),('D000',3),('HLT_X3',3)]:
        v=views.build_view(p,coordinate=coord,offline=Forbidden(),identity=Forbidden(),candidate=Forbidden(),mapping=Forbidden())
        np.testing.assert_array_equal(v.p4,np.repeat(p.p4,copies,axis=0))
        x=views.deployment_inputs(p,copies=copies,capacity=32)
        np.testing.assert_array_equal(x.features,build_inputs(v,capacity=32).features)
    with pytest.raises(ValueError): views.repeat(p,True)
    with pytest.raises(ValueError,match='truncation'): views.deployment_inputs(physical(10),copies=3,capacity=16)


def test_nested_switch_hashes_reproducible():
    from fractions import Fraction
    for owner in range(30):
        hits=[views.choose('a'*64,CANDIDATES[0],owner,1,'d0',Fraction(n,4)) for n in range(5)]
        assert hits==sorted(hits) and hits[0] is False and hits[-1] is True


def test_formula_portable_copy_authenticates_exact_same_formula(tmp_path,monkeypatch):
    path,digest=donor(tmp_path/'old.json')
    original=Path.open
    def restricted(p,*a,**k):
        assert p==path
        return original(p,*a,**k)
    monkeypatch.setattr(Path,'open',restricted)
    value=q.import_formula(path,digest)
    assert value['candidate']==CANDIDATES[1] and value['original_formula']==donor_views(CANDIDATES[1])['salience']
    assert not value['assignments_imported'] and not value['trained_models_imported']
    with pytest.raises(ValueError,match='byte hash'): q.import_formula(path,'0'*64)


def test_formula_does_not_accept_rehashed_scientific_or_lineage_changes(tmp_path):
    path,digest=donor(tmp_path/'donor.json')
    v=load_json(path)
    for change in ('formula','parent','pilot'):
        a=deepcopy(v)
        if change=='formula':
            a['foundation']=rehash(a['foundation'],candidate=CANDIDATES[0])
        elif change=='parent':
            a['foundation']=rehash(a['foundation'],source_import_sha256='0'*64)
        else: a['contract']='JETCLASS2_DELPHES_CONCAT_K2_CAMPAIGN_SPEC/v8'
        p=tmp_path/(change+'.json'); c.publish(p,rehash(a))
        with pytest.raises(ValueError): q.import_formula(p,sha256_file(p))


def test_graph_ten_fits_five_reducers_full_original_recipe_and_oscar(tmp_path):
    spec=toy_spec(tmp_path)
    assert spec['counts']==dict(train=100000,validation=50000)
    assert spec['training']['maximum_passes']==100 and spec['training']['batch_size']==128
    assert [n['primary_coordinate'] for n in spec['nodes'][5:]]==['D075','D050','D025','D000','HLT_X1']
    assert len(q.plan(spec,'science')['commands'])==17
    assert sum(t['kind']=='train' for t in spec['tasks'])==10
    assert sum(t['kind']=='reduce' for t in spec['tasks'])==5
    byid={r['task_id']:r for r in spec['tasks']}
    assert byid['train_HLT_X1_COMPRESSED']['dependencies']==['reduce_CONCAT_K2_D000']
    assert byid['train_DIRECT_HLT_X3_KD']['dependencies']==['reduce_CONCAT_K2_D100']
    seen=set()
    for r in q.plan(spec,'all')['commands']:
        assert set(r['dependencies'])<=seen
        seen.add(r['task_id'])
        cmd=r['command']
        assert '--account=default' in cmd and '--no-requeue' in cmd
        if byid[r['task_id']]['resource'] in ('train','reduce','preflight'):
            assert '--partition=gpu' in cmd and '--qos=norm-gpu' in cmd and '--gres=gpu:l40s:1' in cmd
        else:
            assert '--partition=batch' in cmd and not any(x.startswith('--gres=') for x in cmd)
        assert not any('afterany' in x or 'sporc' in x or 'tigris' in x for x in cmd)
    for r in q.plan(spec,'science')['commands']:
        assert 'preflight' not in r['dependencies']  # Artifact gate replaces expired scheduler ID.


@pytest.fixture
def created(tmp_path,monkeypatch):
    path,digest=donor(tmp_path/'old.json')
    fake=SimpleNamespace(study={'shards':[{'shard_id':'t','role':'train'},{'shard_id':'v','role':'validation'},
                                             {'shard_id':'TEST','role':'final_test'}]},describe=lambda:{'metadata':'toy'})
    monkeypatch.setattr(q.RelocatedDataset,'from_oscar_copy',lambda *a,**k:fake)
    monkeypatch.setattr(q,'_source',lambda *a:None)
    project=tmp_path/'code'
    monkeypatch.setattr(q,'__file__',str(project/'src/hlt_classification/noise_k2/campaign.py'))
    return q.create(dataset_root=tmp_path/'data',formula_spec=path,formula_sha256=digest,
        project_dir=project,source_commit='a'*40,campaign_root=tmp_path/'campaign')


def test_create_dry_by_default_source_and_population_scopes(created,monkeypatch):
    root=Path(created['campaign_root'])
    assert len(created['shards'])==2
    for stage in ('all','gate','science'):
        ledger=load_json(root/f'{stage}_dry_run.json')
        assert ledger['dry_run'] and set(ledger['jobs'].values())=={f'DRY_RUN_{i:04d}' for i in range(len(ledger['jobs']))}
    assert not list(root.glob('*submission*'))
    assert q.validate_campaign(created)==created['content_hash']
    bad=rehash(created,counts={'train':500000,'validation':50000})
    with pytest.raises(ValueError,match='registration'): q.validate_campaign(bad)
    with pytest.raises(ValueError,match='full-DAG'): q.submit(created,stage='all',execute=True,authorization_phrase=c.AUTHORIZE)
    with pytest.raises(ValueError,match='phrase'): q.submit(created,stage='gate',execute=True)
    with pytest.raises(ValueError,match='preflight'): q.submit(created,stage='science',execute=True,authorization_phrase=c.AUTHORIZE)


def test_native_preflight_cpu_cannot_admit(created,tmp_path):
    with pytest.raises(ValueError,match='Genuine'): run.preflight(created,tmp_path,'1','cpu')


def test_live_staged_submission_exact_journal_no_duplicate_and_sanitized_environment(created,monkeypatch):
    calls=[]
    def command(cmd,**kw):
        assert all(not k.startswith(('SLURM_','SBATCH_')) for k in kw['env'])
        calls.append(cmd)
        return SimpleNamespace(returncode=0,stdout=str(1000+len(calls))+'\n',stderr='')
    monkeypatch.setenv('SBATCH_PARTITION','debug'); monkeypatch.setenv('SLURM_JOB_ID','999')
    monkeypatch.setattr(q.subprocess,'run',command)
    ledger=q.submit(created,stage='gate',execute=True,authorization_phrase=c.AUTHORIZE,auto_science=True)
    assert not ledger['dry_run'] and 'preflight' in ledger['jobs'] and 'train_HLT_X1_CE' not in ledger['jobs']
    assert any('--test-only' in cmd for cmd in calls)
    before=sum('--test-only' not in cmd for cmd in calls)
    again=q.submit(created,stage='gate',execute=True,authorization_phrase=c.AUTHORIZE,auto_science=True)
    assert again==ledger and sum('--test-only' not in cmd for cmd in calls)==before
    assert load_json(Path(created['campaign_root'])/'authorization.json')['auto_science']


def test_submission_ambiguous_ack_requires_review(created,monkeypatch):
    def command(cmd,**kw):
        if '--test-only' in cmd: return SimpleNamespace(returncode=0,stdout='',stderr='')
        raise OSError('ambiguous acknowledgement')
    monkeypatch.setattr(q.subprocess,'run',command)
    with pytest.raises(OSError): q.submit(created,stage='gate',execute=True,authorization_phrase=c.AUTHORIZE)
    with pytest.raises(FileExistsError): q.submit(created,stage='gate',execute=True,authorization_phrase=c.AUTHORIZE)


def test_worker_requires_exact_journal_id_and_oscar_allocation(created,monkeypatch):
    counter=iter(range(12000,12100))
    monkeypatch.setattr(q.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=str(next(counter)),stderr=''))
    ledger=q.submit(created,stage='gate',execute=True,authorization_phrase=c.AUTHORIZE)
    row=q.task(created,'authenticate'); job=ledger['jobs']['authenticate']
    site=created['execution_site']
    for key,value in dict(SLURM_JOB_ID=job,SLURM_CLUSTER_NAME=site['cluster'],SLURM_CPUS_PER_TASK='1',
        SLURM_MEM_PER_NODE='8192',PYTHONNOUSERSITE='1',CONDA_PREFIX=site['conda_base']+'/envs/'+site['conda_env']).items():
        monkeypatch.setenv(key,value)
    monkeypatch.setattr(q.sys,'prefix',site['conda_base']+'/envs/'+site['conda_env'])
    monkeypatch.setattr(q.platform,'machine',lambda:site['architecture'])
    result=SimpleNamespace(stdout='Account=default Partition=batch NumNodes=1 NumTasks=1 NumCPUs=1')
    monkeypatch.setattr(q.subprocess,'run',lambda *a,**k:result)
    assert q.authenticate_job(created,row)==job
    result.stdout='Account=default Partition=debug NumNodes=1 NumTasks=1 NumCPUs=1'
    with pytest.raises(ValueError,match='allocation'): q.authenticate_job(created,row)
    monkeypatch.setenv('SLURM_JOB_ID','99999')
    with pytest.raises(ValueError,match='exact submitted'): q.authenticate_job(created,row)


def test_slurm_test_only_rejection_publishes_no_claim(created,monkeypatch):
    monkeypatch.setattr(q.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout='',stderr='account limit'))
    with pytest.raises(ValueError,match='account limit'): q.submit(created,stage='gate',execute=True,authorization_phrase=c.AUTHORIZE)
    assert not list(Path(created['campaign_root']).glob('*.claim'))


def test_validation_partition_is_disjoint_and_exactly_replayable():
    cache=make_cache('validation')
    codes=data.partition_codes(cache.identities,cache.labels)
    assert np.bincount(codes).tolist()==[44,22,22]
    order=np.arange(len(cache))[::-1]
    np.testing.assert_array_equal(data.partition_codes(cache.identities[order],cache.labels[order]),codes[order])


def test_science_native_evidence_cannot_be_faked_by_single_passed_boolean(tmp_path,monkeypatch):
    spec=toy_spec(tmp_path)
    monkeypatch.setattr(data,'get_foundation',lambda s:{'content_hash':'f'*64,'capacity':32})
    with pytest.raises((KeyError,ValueError)):
        run.validate_acceptance(spec,c.artifact('ACCEPTANCE',passed=True))
    assert run.result_rows(spec)==[dict(node_id=n['node_id'],deployable=n['deployable'],validation=None,
        recovery=None,selected_pass=None,passes=None) for n in spec['nodes']]


def test_final_test_cache_and_selection_rejected_before_io(tmp_path,monkeypatch):
    spec=toy_spec(tmp_path)
    monkeypatch.setattr(Path,'open',lambda *a,**k:pytest.fail('final-test access'))
    with pytest.raises(PermissionError): data.cache(spec,'final_test','D100')
    with pytest.raises(PermissionError): data.selected(spec,'final_test')
    with pytest.raises(PermissionError): list(data.iter_selected(None,{'role':'final_test'},None,paired=True))


def test_native_kernel_ce_kd_checkpoint_bank_roundtrip_partial_results(tmp_path,monkeypatch):
    spec=toy_spec(tmp_path)
    root=Path(spec['campaign_root']); root.mkdir()
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.fc=torch.nn.Linear(17,11)
        def forward(self,features,vectors,mask):
            return self.fc((features*mask).sum(-1)/mask.sum(-1))
    def model(node):
        torch.manual_seed(node['initialization_seed']); return Tiny()
    monkeypatch.setattr(run,'new_model',model); torch.set_num_threads(1)
    train, val=make_cache('train'),make_cache('validation')
    codes=data.partition_codes(val.identities,val.labels)
    from hlt_classification.jetclass2_delphes.salience_learned_data import IndexedRamCache
    subsets={n:IndexedRamCache(val,np.flatnonzero(codes==i),role='validation') for i,n in enumerate(('checkpoint','diagnostic','report'))}
    monkeypatch.setattr(run,'caches',lambda *a:(train,subsets,{'content_hash':'p'*64}))
    monkeypatch.setattr(data,'cache',lambda s,role,coord:train if role=='train' else val)
    monkeypatch.setattr(data,'get_foundation',lambda s:{'capacity':32})
    done={}
    monkeypatch.setattr(run,'completed',lambda s,n,**kw:done.get(n))
    def record(name,result):
        result={k:v.relative_to(root).as_posix() if isinstance(v,Path) else v for k,v in result.items()}
        done[name]={'content_hash':'e'*64,'result':result}
    teachers={n['teacher_distribution'] for n in spec['nodes']}-{None}
    for name in (n['node_id'] for n in spec['nodes']):
        node=run.node_for(spec,name); directory=root/name; directory.mkdir()
        result=run.fit(spec,node,directory,'cpu'); record('train_'+name,result)
        report=load_json(result['training_report'])
        assert report['kernel_report']['scientific_fit'] and report['kernel_report']['passes']>=75
        assert report['report_validation']['rows']==22
        assert bool(report['teacher_lineage'])==(node['teacher_distribution'] is not None)
        if node['deployable']:
            export=load_json(directory/'deployment.json')
            assert not export['offline_inputs'] and not export['assignment_inputs']
        if name in teachers:
            dest=root/(name+'_reduce'); dest.mkdir()
            result=run.reduce(spec,node,dest,'cpu'); record('reduce_'+name,result)
    rows=run.result_rows(spec)
    assert sum(r['validation'] is not None for r in rows)==10
    assert all(r['recovery'] is not None for r in rows)
    if rows[2]['validation']['macro_ovr_auc']!=rows[0]['validation']['macro_ovr_auc']:
        assert rows[0]['recovery']['macro_ovr_auc']==pytest.approx(0.)
        assert rows[2]['recovery']['macro_ovr_auc']==pytest.approx(100.)
    else:
        assert rows[0]['recovery']['macro_ovr_auc'] is None
        assert rows[2]['recovery']['macro_ovr_auc'] is None


def test_task_receipt_byte_and_parent_tampering_closed(tmp_path):
    spec=toy_spec(tmp_path); root=Path(spec['campaign_root']); root.mkdir()
    output=root/'ok.json'; c.publish(output,c.artifact('DIAGNOSTIC'))
    receipt=c.artifact('TASK',campaign_sha256=spec['content_hash'],source_commit=spec['source_commit'],
        task_id='authenticate',parents={},outputs=[c.reference(root,output)],result={})
    c.publish(root/'tasks/authenticate.json',receipt)
    assert q.completed(spec,'authenticate')==receipt
    # No destructive overwrite needed to demonstrate a corrupt immutable reference.
    malformed=rehash(receipt,task_id='select_population',parents={'authenticate':'0'*64})
    c.publish(root/'tasks/select_population.json',malformed)
    with pytest.raises(ValueError,match='Dependency'): q.completed(spec,'select_population')
    wrong=deepcopy(receipt['outputs'][0]); wrong['sha256']='0'*64
    with pytest.raises(ValueError,match='bytes'): c.checked(root,wrong)


def test_path_traversal_fails(tmp_path):
    for path in ('../outside',str(tmp_path/'absolute')):
        with pytest.raises(ValueError): c.safe(tmp_path,path)


def native_evidence_fixture(spec):
    """Validation-unit fixture only; never write this into a task receipt."""
    from hlt_classification.jetclass2_delphes.salience_learned_contracts import artifact as ka
    stats=dict(calls=3,saved_cuda_tensors=9,saved_cuda_bytes=4096,restored_cuda_tensors=9)
    evidence=[]
    for name in run.CASES:
        kernel=ka('TRAINING_REPORT',node=dict(run.node_for(spec,name),node_id='ACCEPTANCE_'+name),
            passes=2,validation_history=[{},{}],acceptance_only=True,scientific_fit=False,
            selected_weights_restored=True,final_test_accessed=False,
            batching=dict(training_batch_size=128,inference_batch_size=128,gradient_accumulation_steps=1))
        evidence.append(dict(node_id=name,kernel_report=kernel,checkpoint_roundtrip=True,bank_roundtrip=True,
            train_rows=100000,validation_rows=50000,train_seconds_per_row=.001,validation_seconds_per_row=.0001,
            stress=dict(step_seconds=[1.,1.,1.],validation_seconds=.1,storage_stats=stats)))
    parity=[]
    for n in ('CONCAT_K2_D100','CONCAT_K2_D000'):
        for p in ('fp32','bf16'):
            parity.append(dict(node_id=n,native=dict(passed=True,device='cuda',model=spec['model'],
                forward_and_feature_and_parameter_gradients=True,final_test_accessed=False),storage=dict(
                passed=True,device_type='cuda',steps=3,checks=run.PARITY_CHECKS,pair_storage=spec['pair_storage'],
                parity_backend=run.PARITY_BACKEND,precision=p,tolerance=run.PARITY_TOLERANCES[p],storage_stats=stats)))
    return c.artifact('ACCEPTANCE',campaign_sha256=spec['content_hash'],foundation_sha256='f'*64,
        capacity=32,site=spec['execution_site'],ordinary_rows=spec['counts'],batch_size=128,inference_batch_size=128,
        pair_storage=spec['pair_storage'],acceptance_only=True,job_id='12345',gpu=dict(name='L40S',total_memory_bytes=48*2**30),
        environment={'fixture_only':True},peak_cuda_bytes=20*2**30,peak_rss_bytes=20*2**30,
        projected_max_fit_seconds=60000.,evidence=evidence,parity=parity)


@pytest.mark.parametrize('change',['batch','cpu','cuda','walltime','site','counts','cases','parity','kernel','source','nan'])
def test_native_gate_rejects_rehashed_invalid_evidence(tmp_path,monkeypatch,change):
    spec=toy_spec(tmp_path)
    monkeypatch.setattr(data,'get_foundation',lambda s:dict(content_hash='f'*64,capacity=32))
    a=native_evidence_fixture(spec)
    assert run.validate_acceptance(spec,a)==a
    a=deepcopy(a)
    if change=='batch': a['batch_size']=256
    elif change=='cpu': a['peak_rss_bytes']=1e15
    elif change=='cuda': a['peak_cuda_bytes']=48*2**30
    elif change=='walltime': a['projected_max_fit_seconds']=24*3600
    elif change=='site': a['site']['partition']='debug'
    elif change=='counts': a['ordinary_rows']['validation']=1000000
    elif change=='cases': a['evidence'].pop()
    elif change=='parity': a['parity'][0]['storage']['storage_stats']['calls']=0
    elif change=='kernel':
        a['evidence'][0]['kernel_report']=rehash(a['evidence'][0]['kernel_report'],scientific_fit=True)
    elif change=='source': a['campaign_sha256']='0'*64
    elif change=='nan': a['evidence'][0]['train_seconds_per_row']=-1.
    with pytest.raises(ValueError): run.validate_acceptance(spec,rehash(a))

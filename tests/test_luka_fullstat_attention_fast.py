from copy import deepcopy
from io import BytesIO
import json
import numpy as np
import pytest
import torch

from test_luka_attention import cache, tiny_config, threads
from test_luka_fullstat_attention import make_bank, models
from hlt_classification.luka_attention.model import new_model
from hlt_classification.luka_fullstat_attention import training as original
from hlt_classification.luka_fullstat_attention import banks as old_banks
from hlt_classification.luka_fullstat_attention_fast import banks, training, checkpoints
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.data.cache_contracts import with_content_hash


def resident(root, tr, m, **kw):
    return banks.ResidentBank(root, tr, parents=m['parents'], teacher='COARSE_D033',
                              require_attention=True, max_bytes=kw.get('max_bytes', 2**30))


def clean_report(report):
    report = deepcopy(report)
    for key in ('content_hash','runtime_seconds'):
        report.pop(key)
    for row in report['validation_history']:
        row.pop('train_seconds'); row.pop('validation_seconds')
    return report


def test_resident_bytes_order_zero_padding_and_no_hot_path_io(tmp_path, monkeypatch):
    old, tr, manifest = make_bank(tmp_path/'bank', monkeypatch)
    fast = resident(tmp_path/'bank', tr, manifest)
    indices = np.array([21, 0, 5, 5, 13, 1])
    expected = old.attention_batch(indices, 8)
    def forbidden(*a, **kw):
        raise AssertionError('Target filesystem access in a training batch')
    monkeypatch.setattr(banks, 'safe', forbidden)
    monkeypatch.setattr(np, 'load', forbidden)
    np.testing.assert_array_equal(fast.attention_batch(indices, 8), expected)
    assert not fast.values.flags.writeable
    with pytest.raises(ValueError): fast.attention_batch(indices, 1)
    with pytest.raises(ValueError): fast.attention_batch(np.array([-1]), 4)


def test_resident_budget_and_corruption_fail_closed(tmp_path, monkeypatch):
    _, tr, m = make_bank(tmp_path/'bank', monkeypatch)
    with pytest.raises(MemoryError): resident(tmp_path/'bank', tr, m, max_bytes=1)
    path = tmp_path/'bank'/m['shards'][0]['payload']['path']
    with path.open('ab') as f: f.write(b'bad')
    with pytest.raises(ValueError): resident(tmp_path/'bank', tr, m)


@pytest.mark.parametrize('depth', [0, 2])
def test_prefetch_is_exact_order_and_does_not_advance_rng(tmp_path, monkeypatch, depth):
    _, tr, m = make_bank(tmp_path/'bank', monkeypatch)
    fast = resident(tmp_path/'bank', tr, m)
    order = np.random.default_rng(17).permutation(len(tr))
    state = checkpoints.rng_state()
    for ix, raw, target in banks.batches(tr, fast, order, depth=depth):
        np.testing.assert_array_equal(ix, order)
        for key, value in tr.batch(ix).items():
            np.testing.assert_array_equal(raw[key], value)
        np.testing.assert_array_equal(target, fast.attention_batch(ix, raw['mask'].shape[-1]))
    after = checkpoints.rng_state()
    assert torch.equal(after['torch'], state['torch'])
    assert after['numpy'] == state['numpy'] and after['python'] == state['python']


@pytest.mark.parametrize('device', ['cpu','cuda'])
@pytest.mark.parametrize('rows',[22,153])
def test_twenty_epochs_identical_to_original_and_resume(tmp_path, monkeypatch, device, rows):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    tiny_config(monkeypatch)
    tr=cache(coordinate='D000',rows=rows)
    parents=dict(checkpoint='c'*64,report='d'*64)
    m=old_banks.export(tmp_path/'bank',new_model(models()[0]),cache(coordinate='D033',rows=rows),
        parents=parents,teacher='COARSE_D033',attention=True,device='cpu')
    old=old_banks.Bank(tmp_path/'bank',tr,parents=parents,teacher='COARSE_D033',require_attention=True)
    fast = resident(tmp_path/'bank', tr, m)
    val, node = cache('validation','D000'), models()[1]
    with parity_backend(device):
        expected, state = original.train(new_model(node), tr, val, bank=old, node=node, device=device)
        actual, produced = training.train(new_model(node), tr, val, bank=fast, node=node, device=device)
        assert clean_report(actual) == clean_report(expected)
        for key in state: torch.testing.assert_close(produced[key], state[key], rtol=0, atol=0)
        binding = dict(spec='a'*64, bank=m['content_hash'])
        store = checkpoints.EpochStore(tmp_path/'checkpoints', binding=binding)
        with pytest.raises(InterruptedError):
            training.train(new_model(node), tr, val, bank=fast, node=node, device=device,
                           store=store, stop_after=7)
        restored = checkpoints.load(tmp_path/'checkpoints/epoch-07.json', binding=binding)
        resumed, final = training.train(new_model(node), tr, val, bank=fast, node=node,
                                       device=device, resume=restored)
        assert clean_report(resumed) == clean_report(expected)
        for key in state: torch.testing.assert_close(final[key], state[key], rtol=0, atol=0)
        with pytest.raises(ValueError): checkpoints.load(tmp_path/'checkpoints/epoch-07.json', binding={})
        with (tmp_path/'checkpoints/epoch-07.pt').open('ab') as f: f.write(b'bad')
        with pytest.raises(ValueError): checkpoints.load(tmp_path/'checkpoints/epoch-07.json', binding=binding)


def test_resident_many_shards_retains_original_random_targets(tmp_path, monkeypatch):
    tiny_config(monkeypatch)
    tr=cache(coordinate='D000',rows=70)
    parents=dict(checkpoint='c'*64,report='d'*64)
    m=old_banks.export(tmp_path/'bank',new_model(models()[0]),cache(coordinate='D033',rows=70),
        parents=parents,teacher='COARSE_D033',attention=True,device='cpu')
    #Exercise original LRU churn with >64 authenticated shards.
    n = len(tr)
    shards=[]
    original_data=np.load(tmp_path/'bank'/m['shards'][0]['payload']['path'])
    offsets=m['shards'][0]['offsets']
    for i in range(n):
        path=tmp_path/'bank'/f'row-{i}.npy'
        buf=BytesIO(); np.save(buf, original_data[offsets[i]:offsets[i+1]], allow_pickle=False)
        path.write_bytes(buf.getvalue())
        shards.append(dict(start=i,stop=i+1,offsets=[0,offsets[i+1]-offsets[i]],payload=file_ref(path,root=tmp_path/'bank')))
    m=with_content_hash(dict(m,shards=shards))
    (tmp_path/'bank/manifest.json').write_text(json.dumps(m))
    old=old_banks.Bank(tmp_path/'bank',tr,parents=m['parents'],teacher='COARSE_D033',require_attention=True)
    fast=resident(tmp_path/'bank',tr,m)
    real_load=np.load; opens=[]
    def counted(*args,**kw):
        opens.append(args[0]); return real_load(*args,**kw)
    monkeypatch.setattr(np,'load',counted)
    for _ in range(3):
        ix=np.random.default_rng(8).permutation(n)
        np.testing.assert_array_equal(fast.attention_batch(ix,4),old.attention_batch(ix,4))
    assert len(opens)==3*n and len(old._lru)==64


def test_prefetch_multiple_batches_and_error_propagation():
    tr=cache(coordinate='D000',rows=273)
    from types import SimpleNamespace
    targets=SimpleNamespace(attention_batch=lambda ix,width:np.zeros((len(ix),width,width),np.float32))
    order=np.random.default_rng(7).permutation(len(tr))
    got=[]
    for ix,raw,target in banks.batches(tr,targets,order):
        got.extend(ix.tolist())
        for key,value in tr.batch(ix).items(): np.testing.assert_array_equal(raw[key],value)
    np.testing.assert_array_equal(got,order)
    def bad(*a,**kw): raise ValueError('Producer error')
    with pytest.raises(ValueError,match='Producer error'):
        list(banks.batches(tr,SimpleNamespace(attention_batch=bad),order))


def test_budgets_are_finite_conservative_and_bounded():
    from hlt_classification.luka_fullstat_attention_fast.preflight import budget
    counts=dict(train=3000000,validation=500000)
    assert budget(train_step_seconds=.03,validation_seconds=60,auth_seconds=100,load_seconds=200,counts=counts)==720
    with pytest.raises(ValueError): budget(train_step_seconds=1,validation_seconds=60,auth_seconds=100,load_seconds=200,counts=counts)
    with pytest.raises(ValueError): budget(train_step_seconds=float('nan'),validation_seconds=60,auth_seconds=100,load_seconds=200,counts=counts)


def fast_fixture(tmp_path,monkeypatch):
    from test_luka_fullstat_attention import local_fixture
    from hlt_classification.luka_fullstat_attention import worker as old_worker
    from hlt_classification.luka_fullstat_attention_fast import contracts as c
    parent,_,audit=local_fixture(tmp_path,monkeypatch,'cuda')
    _,parents=old_worker.checkpoint(parent)
    m=old_banks.export(old_worker.bank_path(parent),new_model(models()[0]),cache(coordinate='D033'),
        parents=parents,teacher='COARSE_D033',attention=True,device='cuda')
    root=tmp_path/'fast'; root.mkdir()
    g=c.artifact('GATE',root=str(root),project_dir='/project',source_commit='a'*40,
        source_snapshot=dict(content_hash='b'*64),policy=c.policy(),
        reuse=dict(gate=parent,preflight=dict(audit=audit),export=dict(result=dict(bank_hash=m['content_hash']))))
    return g


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Local CUDA optional; Oscar performance gate remains required')
def test_attention_only_worker_and_historical_comparison(tmp_path,monkeypatch):
    from hlt_classification.luka_fullstat_attention_fast import worker,campaign,contracts as c
    g=fast_fixture(tmp_path,monkeypatch)
    root=tmp_path/'fast/science'; root.mkdir()
    s=c.artifact('SPEC',root=str(root),project_dir='/project',source_commit='a'*40)
    monkeypatch.setattr(campaign,'check_spec',lambda s:(g,{}))
    monkeypatch.setattr(campaign,'storage',lambda g:None)
    monkeypatch.setattr(worker,'memory_snapshot',lambda t:dict(rss_bytes=2**30))
    #Windows has no fcntl; a mock only for this LOCAL workflow, real worker is Linux-only.
    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules,'fcntl',SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *a:None))
    receipt=worker.run(s,'train_ATTN_D000')
    assert worker.completed(s,g,'train_ATTN_D000')==receipt
    aggregate=worker.run(s,'aggregate')['result']
    assert aggregate['primary_contrast']=='ATTN_D000 minus original COARSE_D000'
    assert len(aggregate['rows'])==6 and aggregate['new_logit_fit'] is False
    assert not aggregate['final_test_accessed']
    assert len(list((root/'attempts/123/checkpoints').glob('epoch-*.json')))==20
    assert worker.run(s,'aggregate')['result']==aggregate
    with pytest.raises(ValueError): worker.run(s,'export_D033')
    with (root/'train_ATTN_D000/selected.pt').open('ab') as f: f.write(b'bad')
    with pytest.raises(ValueError): worker.completed(s,g,'aggregate')


def test_exact_one_gate_two_science_jobs_no_export_or_logit(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from test_luka_fullstat_attention import gate_fixture
    from hlt_classification.luka_fullstat_attention_fast import campaign,submission as s,contracts as c
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    old=gate_fixture(tmp_path/'old')
    g=c.artifact('GATE',root=str(tmp_path),project_dir='/project',source_commit='a'*40,policy=c.policy(),reuse=dict(gate=old))
    monkeypatch.setattr(campaign,'check_gate',lambda g:None)
    monkeypatch.setattr(s,'check_site',lambda p:None)
    calls=[]
    monkeypatch.setattr(exact.subprocess,'run',lambda cmd,**kw:(calls.append(cmd) or SimpleNamespace(stdout='12345\n')))
    dry=s.submit(g)
    assert dry['jobs']==1 and not calls
    for digest,auth in [('bad',c.AUTHORIZATION),(dry['content_hash'],'bad')]:
        with pytest.raises(PermissionError): s.submit(g,execute=True,plan_hash=digest,authorization=auth)
    for _ in range(2): s.submit(g,execute=True,plan_hash=dry['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==1 and s.accepted_jobs(g)==dict(preflight='12345')
    spec=c.artifact('SPEC',root=str(tmp_path/'science'),project_dir='/project',source_commit='a'*40,train_minutes=720,
        tasks=[dict(task_id='train_ATTN_D000',dependencies=[]),dict(task_id='aggregate',dependencies=['train_ATTN_D000'])])
    plan=s.render(spec,g)
    assert plan['jobs']==2 and [r['task_id'] for r in plan['commands']]==['train_ATTN_D000','aggregate']
    assert '--gres=gpu:l40s:1' in plan['commands'][0]['command']
    assert '--mem=180000M' in plan['commands'][0]['command']
    assert '--partition=batch' in plan['commands'][1]['command']
    assert '--dependency=afterok:${JOB_train_ATTN_D000}' in plan['commands'][1]['command']
    with pytest.raises(ValueError): s.render(dict(spec,train_minutes=2881),g)


def test_science_requires_registered_successful_performance_gate(tmp_path,monkeypatch):
    from hlt_classification.data.cache_contracts import write_immutable_json
    from hlt_classification.luka_fullstat_attention_fast import campaign,submission,preflight
    g=dict(root=str(tmp_path)); (tmp_path/'preflight').mkdir()
    write_immutable_json(tmp_path/'preflight/receipt.json',dict(job_id='123'))
    monkeypatch.setattr(campaign,'check_gate',lambda g:None)
    monkeypatch.setattr(preflight,'validate_report',lambda *a:None)
    monkeypatch.setattr(submission,'accepted_jobs',lambda g:dict(preflight='124'))
    with pytest.raises(ValueError): campaign.science(g)
    monkeypatch.setattr(submission,'accepted_jobs',lambda g:dict(preflight='123'))
    def fail(job): raise ValueError('Preflight not COMPLETED')
    monkeypatch.setattr(campaign,'preflight_accounting',fail)
    with pytest.raises(ValueError): campaign.science(g)
    assert not (tmp_path/'science').exists()


def test_ambiguous_submission_never_retries(tmp_path,monkeypatch):
    from test_luka_fullstat_attention import gate_fixture
    from hlt_classification.luka_fullstat_attention_fast import campaign,submission as s,contracts as c
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    g=c.artifact('GATE',root=str(tmp_path),project_dir='/project',source_commit='a'*40,
                 policy=c.policy(),reuse=dict(gate=gate_fixture(tmp_path/'old')))
    monkeypatch.setattr(campaign,'check_gate',lambda g:None)
    monkeypatch.setattr(s,'check_site',lambda p:None)
    dry=s.submit(g); calls=[]
    def fail(*a,**kw):
        calls.append(a); raise RuntimeError('Unknown sbatch outcome')
    monkeypatch.setattr(exact.subprocess,'run',fail)
    for error in (RuntimeError,FileExistsError):
        with pytest.raises(error): s.submit(g,execute=True,plan_hash=dry['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==1


def test_recovery_rejects_active_or_unknown_previous_jobs(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from hlt_classification.luka_fullstat_attention_fast import campaign,submission
    monkeypatch.setattr(submission,'registered_jobs',lambda s:dict(train_ATTN_D000='123',aggregate='124'))
    spec=dict(root=str(tmp_path))
    monkeypatch.setattr(campaign.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='123|RUNNING|0:0\n124|PENDING|0:0\n'))
    with pytest.raises(ValueError,match='active/unknown'): campaign.check_stopped(spec)
    monkeypatch.setattr(campaign.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=''))
    with pytest.raises(ValueError,match='active/unknown'): campaign.check_stopped(spec)
    monkeypatch.setattr(campaign.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='123|TIMEOUT|0:0\n124|CANCELLED by 1|0:0\n'))
    campaign.check_stopped(spec)


def test_readonly_original_source_replay_and_no_unscoped_actions():
    from hlt_classification.luka_fullstat_attention_fast import campaign
    assert "worker.completed(s,g,'export_D033')" in campaign.REPLAY
    assert 'train_LOGIT_D000' not in campaign.REPLAY
    assert all('luka_fullstat_attention_fast' in path for path in campaign.RUNTIME)
    package=__import__('pathlib').Path(campaign.__file__).parent
    for path in package.glob('*.py'):
        text=path.read_text()
        assert 'Fresh_check' not in text and 'scancel' not in text and 'rmtree' not in text


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Local CUDA optional; this is NOT Oscar validation')
def test_full_preflight_full_bank_benchmark_and_fail_closed(tmp_path,monkeypatch):
    from hlt_classification.luka_fullstat_attention_fast import preflight as p,worker,campaign
    from hlt_classification.jetclass2_delphes.cache import RamCache,RamBlock
    g=fast_fixture(tmp_path,monkeypatch)
    #Real bank and real installed model, synthetic34kpopulation. This is a local
    #production-code exercise, not a full3M performance or resource claim.
    sample=cache(coordinate='D000').batch(np.array([0]))
    n=34000; length=int(sample['mask'].sum()); labels=np.arange(n,dtype=np.int64)%11
    ids=np.zeros((n,32),np.uint8); ids[:,:8]=np.arange(n,dtype=np.int64).view(np.uint8).reshape(n,8)
    block=RamBlock(0,np.arange(n+1,dtype=np.int64)*length,
        np.tile(sample['features'][0,:,:length].T,(n,1)),np.tile(sample['vectors'][0,:,:length].T,(n,1)),ids,labels)
    tr=RamCache([block],role='train',foundation_sha256='a'*64,coordinate_name='D000')
    teacher=RamCache([block],role='train',foundation_sha256='a'*64,coordinate_name='D033')
    val=cache('validation','D000')
    parents=dict(checkpoint='c'*64,report='d'*64)
    m=old_banks.export(tmp_path/'large',new_model(models()[0]),teacher,parents=parents,teacher='COARSE_D033',attention=True,device='cuda')
    b=banks.ResidentBank(tmp_path/'large',tr,parents=parents,teacher='COARSE_D033',require_attention=True,max_bytes=2**30)
    g['reuse']['gate']['counts']['train']=n
    g['reuse']['gate']['counts']['validation']=len(val)
    g['reuse']['export']['result']['bank_hash']=m['content_hash']
    g['reuse']['preflight']['audit']['maxima']=dict(train=length,validation=4)
    g['reuse']['preflight']['audit']['sum_squared_counts']=int((b.counts**2).sum())
    monkeypatch.setattr(campaign,'check_gate',lambda g:g['reuse']['gate'])
    monkeypatch.setattr(campaign,'storage',lambda g:None)
    monkeypatch.setattr(worker,'load_inputs',lambda g:(tr,val,b))
    monkeypatch.setattr(p,'installed_environment',lambda:{})
    monkeypatch.setattr(p,'gpu_identity',lambda:dict(name='local test GPU',total_memory_bytes=48*2**30))
    monkeypatch.setattr(p,'memory_snapshot',lambda t:dict(rss_bytes=2**30,peak_rss_bytes=2**30,gpu_peak_bytes=t.cuda.max_memory_allocated()))
    report=p.run(g); p.validate_report(g,report)
    assert report['benchmark_batches']==256 and report['loader_benchmark']['target_bytes_equal']
    assert report['full_validation_measured'] and not report['full_epoch_training_measured']
    for key,value in [('technical_checks_passed',False),('scientific_fit',True),('benchmark_batches',128),
                      ('final_test_accessed',True),('train_minutes',1),('reserved_host_peak_bytes',180000*1024**2)]:
        with pytest.raises((ValueError,PermissionError)): p.validate_report(g,with_content_hash(dict(report,**{key:value})))

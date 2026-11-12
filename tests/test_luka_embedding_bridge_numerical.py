"""Historical export tolerance never replaces exact lineage or local parity."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from hlt_classification.data.cache_contracts import with_content_hash
from hlt_classification.luka_embedding_bridge_hydra import contracts as c, numerical as n, qualification as q


def stats(maximum=0., rms=0.):
    return dict(max_abs=maximum, rms=rms, mean_abs=rms/2)


def history():
    return dict(standardized=stats(.0014574527740478516,5.6815192022950564e-5),
                raw=stats(.00044439450412658843,2.9057899678110172e-5))


def probe_record(rows=299234):
    batches=n.fixed_batches(rows);p=c.u100_numerical_policy()
    return dict(rows=sum(map(len,batches)),batch_starts=[int(a[0]) for a in batches],batch_lengths=list(map(len,batches)),
        selection=p['probe_selection'],cpu_threads=p['cpu_threads'],precision='fp32_autocast_disabled',
        modes={mode:dict(seconds=1.,historical=history()) for mode in p['probe_modes']},
        comparisons={a+'_vs_'+b:stats(5.364418029785156e-6,5.161255827756094e-7) for a,b in n.PAIRS})


def witness():
    fields=dict(rows=4096,numerical_policy=c.u100_numerical_policy(),local_probe=probe_record(),historical=history(),
        maximum_error=.0014574527740478516,rms_error=5.6815192022950564e-5,mean_absolute_error=5.6815192022950564e-5/2,
        atol=1e-4,rtol=1e-4,exceeding_coordinates=13629,legacy_elementwise_passed=False,
        legacy_elementwise_diagnostic_only=True,extraction_policy=c.U100_EXTRACTION)
    fields['checks']=n.decisions(fields,299234)
    return c.artifact('U100_REPLAY',parents=dict(spec='a'*64,teacher='b'*64,original='c'*64),**fields,passed=True)


def test_diagnostic_supports_bounded_qualification_without_legacy_pass(monkeypatch):
    value=witness();assert all(value['checks'].values()) and not value['legacy_elementwise_passed']
    monkeypatch.setattr(q.inputs,'transferred',lambda _:SimpleNamespace(manifest=dict(teacher=dict(sha256='b'*64))))
    spec=dict(content_hash='a'*64,embedding_hash='c'*64,counts=dict(train=299234),u100_numerical_policy=c.u100_numerical_policy())
    assert q.check_u100_witness(value,spec)==value
    assert value['local_probe']['batch_lengths']==[128,128,128,128,98]


@pytest.mark.parametrize('location,field,value',[
    ('historical_std','max_abs',.00201),('historical_std','rms',.000101),
    ('historical_raw','max_abs',.00201),('historical_raw','rms',.000101),
    ('local','max_abs',1.001e-5),('local','rms',2.001e-6),
    ('cpu_historical','max_abs',.01),('gpu_historical','rms',.0002),
])
def test_each_numerical_bound_fails_even_if_saved_passed_true(monkeypatch,location,field,value):
    w=witness()
    target={'historical_std':w['historical']['standardized'],'historical_raw':w['historical']['raw'],
        'local':w['local_probe']['comparisons']['gpu_default_vs_cpu_math'],
        'cpu_historical':w['local_probe']['modes']['cpu_math']['historical']['standardized'],
        'gpu_historical':w['local_probe']['modes']['gpu_math']['historical']['standardized']}[location]
    target[field]=value
    assert not all(n.decisions(w,299234).values())
    # Even resealing a deliberately dishonest passed=True / old checks cannot admit it.
    monkeypatch.setattr(q.inputs,'transferred',lambda _:SimpleNamespace(manifest=dict(teacher=dict(sha256='b'*64))))
    with pytest.raises(ValueError):q.check_u100_witness(with_content_hash(w),dict(content_hash='a'*64,embedding_hash='c'*64,
        counts=dict(train=299234),u100_numerical_policy=c.u100_numerical_policy()))


@pytest.mark.parametrize('change',['policy','missing_cpu','missing_pair','selection','tail','rows','precision','nan','negative','impossible'])
def test_incomplete_or_invalid_evidence_is_rejected(change):
    w=witness();p=w['local_probe']
    if change=='policy':w['numerical_policy']['historical_limits']['standardized']['max_abs']=1.
    elif change=='missing_cpu':del p['modes']['cpu_math']
    elif change=='missing_pair':p['comparisons'].pop('gpu_math_vs_cpu_math')
    elif change=='selection':p['batch_starts'][1]+=1
    elif change=='tail':p['batch_lengths'][-1]=128
    elif change=='rows':p['rows']-=1
    elif change=='precision':p['precision']='bf16'
    elif change=='nan':w['historical']['raw']['rms']=float('nan')
    elif change=='negative':w['historical']['raw']['mean_abs']=-1.
    else:w['historical']['raw']['mean_abs']=1.
    with pytest.raises(ValueError):n.decisions(w,299234)


def test_sparse_and_distributed_errors_are_separately_bounded():
    zeros=np.zeros((4096,128),np.float32);spike=zeros.copy();spike[0,0]=.003
    spread=zeros+.0002;limits=c.u100_numerical_policy()['historical_limits']['standardized']
    assert not n.within(n.metrics(spike,zeros),limits)
    assert not n.within(n.metrics(spread,zeros),limits)
    small=zeros.copy();small[0,0]=.0015
    assert n.within(n.metrics(small,zeros),limits)
    with pytest.raises(ValueError):n.metrics(zeros[:,:127],zeros[:,:127])
    small[0,0]=np.nan
    with pytest.raises(ValueError):n.metrics(small,zeros)


def test_real_cpu_gpu_probe_preserves_teacher_rng_threads_and_backend(monkeypatch):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from test_luka_embedding_bridge_hydra import synthetic_dataset
    from hlt_classification.luka_embedding_bridge_hydra.model import Model
    from hlt_classification.luka_u100_pair_hydra.precision import backend_settings
    # Match worker.rank explicitly; an isolated CUDA run otherwise inherits
    # cuDNN's TF32 default, unlike the production FP32 allocation.
    monkeypatch.setattr(torch.backends.cuda.matmul,'allow_tf32',False)
    monkeypatch.setattr(torch.backends.cudnn,'allow_tf32',False)
    monkeypatch.setattr(torch.backends.cudnn,'benchmark',False)
    monkeypatch.setattr(torch.backends.cudnn,'deterministic',True)
    torch.manual_seed(23)
    ds=synthetic_dataset();view=ds.view('train','U100');m=Model('TR').cuda().eval()
    state={k:v.clone() for k,v in m.state_dict().items()};before=backend_settings()
    batches=[view.batch(ix) for ix in n.fixed_batches(len(view))]
    z=n._encode(m,batches,torch.device('cuda'))
    u=dict(targets=z.copy(),mean=np.zeros(128,np.float64),scale=np.ones(128,np.float64))
    cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
    result=n.local_probe(m,view,u,torch.device('cuda'))
    assert result['modes']['gpu_default']['historical']['raw']['max_abs']==0
    assert result['batch_lengths']==[33]
    assert all(n.within(x,c.u100_numerical_policy()['local_limits']) for x in result['comparisons'].values())
    assert backend_settings()==before
    assert torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),gpu_rng)
    assert all(torch.equal(v,state[k]) for k,v in m.state_dict().items())
    np.testing.assert_array_equal(u['targets'],z)
    encode=n._encode
    def fail_on_cpu(model,batches,device):
        if device.type=='cpu':raise RuntimeError('injected CPU failure')
        return encode(model,batches,device)
    monkeypatch.setattr(n,'_encode',fail_on_cpu)
    with pytest.raises(RuntimeError,match='injected'):n.local_probe(m,view,u,torch.device('cuda'))
    assert backend_settings()==before
    assert torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),gpu_rng)


def test_native_teacher_complete_witness_publish_and_readback(tmp_path,monkeypatch):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from test_luka_embedding_bridge_hydra import synthetic_dataset
    from hlt_classification.luka_embedding_bridge_hydra import banks, distributed as d
    from hlt_classification.luka_embedding_bridge_hydra.model import Model
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    from hlt_classification.data.cache_contracts import load_json
    ds=synthetic_dataset();ctx=d.Context(0,1,torch.device('cuda'));m=Model('TR').cuda().eval()
    with n.strict_math():
        z=d.original_u100(m,ds.view('train','U100'),ctx)
    u=ds.u100('train');u['mean'][:]=0.;u['scale'][:]=1.;u['targets'][:]=z
    saved=u['targets'].copy();path=tmp_path/'teacher.pt';torch.save(m.state_dict(),path)
    (tmp_path/'setup').mkdir();spec=dict(ds.spec,root=str(tmp_path),u100_numerical_policy=c.u100_numerical_policy())
    monkeypatch.setattr(banks.inputs,'transferred',lambda _:SimpleNamespace(path=lambda _:path,
        manifest=dict(teacher=file_ref(path))))
    with n.strict_math():banks.check_u100(spec,ds,ctx)
    w=load_json(tmp_path/'setup/u100_replay.json')
    assert q.check_u100_witness(w,spec)==w and all(w['checks'].values())
    np.testing.assert_array_equal(u['targets'],saved)


def test_schema_separates_old_witnesses_and_policy_is_not_scientific_recipe():
    assert c.artifact('SPEC')['schema_version']==7
    assert c.artifact('U100_REPLAY')['schema_version']==3
    assert c.recipe()['schema_version']==1 and c.u100_numerical_policy()['schema_version']==1
    old=copy.deepcopy(witness());old['schema_version']=2;old['contract']=old['contract'].replace('/v3','/v2')
    with pytest.raises(ValueError):c.validate(with_content_hash(old),'U100_REPLAY')

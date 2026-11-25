"""Pointwise input gradients, unchanged BN populations and second-recovery lineage."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from hlt_classification.luka_sixclass_wide_b256_oscar import model, pair_storage, pair_convolution
from test_luka_sixclass_wide_b256_oscar import cache
from test_luka_fullstat_balanced_ladder import threads
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json,write_immutable_json,with_content_hash
from hlt_classification.luka_sixclass_wide_b256_oscar import (
    contracts as c,campaign,recovery,gradient_recovery,artifacts,early_memory,startup)
from test_luka_sixclass_wide_b256_recovery import registered


@pytest.mark.parametrize('bf16',[False,True])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='Real installed Weaver CUDA parity')
def test_forced_fallback_native_forward_backward_bn_adamw_parity(bf16):
    raw=cache(coordinate='CONCAT',rows=12).batch(np.arange(6))
    proof=pair_storage.parity(raw,'cuda',bf16=bf16,pointwise_dgrad=True)
    assert proof['passed'] and proof['forced_fallback_reference']
    assert proof['pointwise_statistics']['input_gradient_calls']>0
    torch.cuda.empty_cache()


def second_recovery(tmp_path,monkeypatch):
    old,previous,oldjobs=registered(tmp_path,monkeypatch)
    monkeypatch.setattr(gradient_recovery,'PREVIOUS_HASH',previous['content_hash'])
    monkeypatch.setattr(gradient_recovery,'PREVIOUS_COMMIT',previous['commit'])
    monkeypatch.setattr(gradient_recovery,'PREVIOUS_ROOT',previous['root'])
    write_immutable_json(Path(previous['root'])/'campaign_spec.json',previous)
    recovery.claim(previous)
    write_immutable_json(Path(previous['root'])/'train_CONCAT/runs'/gradient_recovery.FAILED_JOB/'failure.json',
        c.artifact('FAILURE',parents=dict(spec=previous['content_hash']),stage='CONCAT',job_id=gradient_recovery.FAILED_JOB,
            error_type='ValueError',message='Nonfinite/disconnected native gradient'))
    log=tmp_path/'diagnostic.out'
    log.write_text('Job ID : '+gradient_recovery.DIAGNOSTIC_JOB+'\n'+
        str(Path(previous['project'])/'src/hlt_classification/luka_sixclass_wide_b256_oscar/early_memory.py')+
        "\nFunction 'ConvolutionBackward0' returned nan values in its 0th output.")
    monkeypatch.setattr(gradient_recovery,'DIAGNOSTIC_LOG',str(log))
    monkeypatch.setattr(gradient_recovery,'capture_source_snapshot',lambda *a,**kw:previous['source'])
    jobs=[dict(task='train_'+stage,job=str(int(gradient_recovery.FAILED_JOB)+i)) for i,stage in enumerate(c.STAGES[2:])]
    monkeypatch.setattr(campaign,'accepted_jobs',lambda s,**kw:oldjobs if s==old else jobs if s==previous else [])
    original,record,evidence=gradient_recovery.authenticate(Path(previous['root'])/'campaign_spec.json',payloads=False)
    fields={k:v for k,v in previous.items() if k not in ('contract','schema_version','content_hash','parents',*c.FLAGS)}
    fields.update(root=str(tmp_path/'gradient'),project='/new/gradient_project',commit='b'*40,
                  population_root=str(tmp_path/'gradient/population'),gradient_recovery=evidence)
    return old,previous,c.artifact('SPEC',version=3,**fields)


def test_second_recovery_authentication_claims_and_science_are_separate(tmp_path,monkeypatch):
    old,previous,new=second_recovery(tmp_path,monkeypatch)
    campaign.check(new,live=False)
    paths=[Path(old['root'])/'concat_memory_recovery_claim_v1.json',Path(previous['root'])/'campaign_spec.json']
    before={p:p.read_bytes() for p in paths}
    assert recovery.origin(new,'OFFLINE')==old
    assert recovery.model_options(new,'CONCAT')==dict(pair_storage=True,pointwise_dgrad=True)
    assert recovery.model_options(new,'DE000')=={}
    assert new['recipe']==old['recipe'] and new['population']==old['population']
    gradient_recovery.claim(new);gradient_recovery.claim(new)
    with pytest.raises(FileExistsError):gradient_recovery.claim(with_content_hash(dict(new,root=str(tmp_path/'other'))))
    assert {p:p.read_bytes() for p in paths}==before
    bad=deepcopy(new);bad['gradient_recovery']['pointwise_policy']['forward']='changed'
    bad['gradient_recovery']=with_content_hash(bad['gradient_recovery'])
    with pytest.raises(ValueError):campaign.check(with_content_hash(bad),live=False)
    with pytest.raises(ValueError,match='superseded'):campaign.remaining(previous,True)


def test_second_recovery_rejects_missing_evidence_live_jobs_or_committed_epochs(tmp_path,monkeypatch):
    old,previous,new=second_recovery(tmp_path,monkeypatch);path=Path(previous['root'])/'campaign_spec.json'
    with monkeypatch.context() as m:
        m.setattr(campaign,'terminal',lambda job:(_ for _ in ()).throw(ValueError('live job')))
        with pytest.raises(ValueError,match='live job'):gradient_recovery.authenticate(path,payloads=False)
    marker=Path(previous['root'])/'train_UE100/epochs/01.json';write_immutable_json(marker,dict(committed=True))
    with pytest.raises(ValueError,match='committed progress'):gradient_recovery.authenticate(path,payloads=False)
    Path(gradient_recovery.DIAGNOSTIC_LOG).write_text('unrelated diagnostic')
    with pytest.raises(ValueError,match='diagnostic witness'):gradient_recovery.validate(new)


def test_second_recovery_only_queues_four_tail_stages(tmp_path,monkeypatch):
    old,previous,new=second_recovery(tmp_path,monkeypatch)
    auth=gradient_recovery.authenticate
    monkeypatch.setattr(gradient_recovery,'authenticate',lambda path:auth(path,payloads=False))
    monkeypatch.setattr(artifacts,'completed',lambda s,stage:dict(receipt={'content_hash':stage}))
    tasks,done=campaign.remaining(new,False)
    assert tasks==['train_'+s for s in c.STAGES[2:]] and set(done)==set(recovery.BASELINES)
    commands=campaign.commands(new,tasks)
    assert commands[0]['dependencies']==[] and commands[1]['dependencies']==['train_CONCAT']
    assert 'train_UE100' in commands[2]['dependencies'] and 'train_DE050' in commands[3]['dependencies']


def test_v3_registration_copies_population_and_preserves_both_prior_specs(tmp_path,monkeypatch):
    import sys
    from hlt_classification.data.cache_contracts import canonical_sha256
    old,previous,new=second_recovery(tmp_path,monkeypatch)
    dataset=tmp_path/'dataset';dataset.mkdir()
    manifest=dict(foundation_sha256=c.PARENT_HASH)
    monkeypatch.setattr(c,'TRANSFER_HASH',canonical_sha256(manifest))
    write_immutable_json(dataset/'transfer_manifest.json',manifest)
    write_immutable_json(dataset/'transfer_complete.json',dict(manifest_sha256=c.TRANSFER_HASH,all_files_verified=True))
    # Register() mechanics are tested here; exact-source authentication is covered
    # independently above. Keep its outputs bound to the same scientific recipe.
    old=dict(old,dataset=str(dataset),environment_dir=str(Path(sys.prefix).resolve()),transfer_manifest_sha256=c.TRANSFER_HASH)
    pop=Path(old['population_root']);write_immutable_json(pop/'population.json',old['population'])
    (pop/'role_parent').mkdir();(pop/'role_parent/scalar').write_bytes(b'population only')
    monkeypatch.setattr(gradient_recovery,'authenticate',lambda path:(old,previous['execution_recovery'],new['gradient_recovery']))
    monkeypatch.setattr(campaign,'locations',lambda *a:None)
    monkeypatch.setattr(campaign,'source',lambda *a:{})
    monkeypatch.setattr(campaign,'runtime',lambda:{})
    monkeypatch.setattr(campaign,'backend',lambda:old['backend'])
    monkeypatch.setattr(campaign.lineage,'verify',lambda *a:{})
    monkeypatch.setattr(campaign,'storage',lambda *a,**kw:old['storage'])
    from hlt_classification.luka_sixclass_wide_b256_oscar import population
    calls=[];monkeypatch.setattr(population,'load',lambda *a,**kw:calls.append(a))
    checks=[];monkeypatch.setattr(campaign,'check',lambda s,**kw:checks.append(s))
    paths=[Path(previous['root'])/'campaign_spec.json',Path(old['root'])/'campaign_spec.json']
    before={p:p.read_bytes() for p in paths}
    root=tmp_path/'registered_v3'
    result=campaign.create(dataset,root,tmp_path/'source_v3','c'*40,sys.prefix,available_gib=18,
                           recover=paths[0])
    assert result['schema_version']==3 and result['gradient_recovery']==new['gradient_recovery']
    assert result['execution_recovery']==previous['execution_recovery']
    assert result['recipe']==previous['recipe'] and result['population']==previous['population']
    assert calls==[(dataset.resolve(),old['population_root'])] and checks==[result]
    assert (root/'population/role_parent/scalar').read_bytes()==b'population only'
    assert not (root/'train_OFFLINE').exists() and {p:p.read_bytes() for p in paths}==before


def force_small_pointwise(monkeypatch):
    original=pair_convolution.PointwiseInputGradient.__init__
    def forced(self,pair):original(self,pair);self.force=True
    monkeypatch.setattr(pair_convolution.PointwiseInputGradient,'__init__',forced)


def test_threshold_dispatch_preserves_small_and_inference_paths(monkeypatch):
    net=model.new_model('CONCAT',pair_storage=True,pointwise_dgrad=True).train()
    guard=net.pair_convolution;layer=net.mod.pair_embed.embed[10]
    monkeypatch.setattr(pair_convolution,'INDEX_LIMIT',6400)
    assert guard.force is False
    layer(torch.randn(1,64,100,requires_grad=True)).sum().backward()
    assert guard.stats['fallback_calls']==0
    layer(torch.randn(1,64,101,requires_grad=True)).sum().backward()
    assert guard.stats['fallback_calls']==guard.stats['input_gradient_calls']==1
    layer.eval()(torch.randn(1,64,101,requires_grad=True))
    assert guard.stats['fallback_calls']==1
    with pytest.raises(ValueError,match='sparse pair layout'):layer(torch.randn(2,64,101))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual CUDA early check; forced small support')
def test_v3_early_memory_records_both_parities_and_actual_backward(tmp_path,monkeypatch):
    from test_luka_sixclass_wide_b256_oscar import spec_fixture
    force_small_pointwise(monkeypatch)
    tr=cache(coordinate='CONCAT',rows=12);val=cache(role='validation',coordinate='CONCAT',rows=12)
    s=spec_fixture(tmp_path);s['schema_version']=3
    monkeypatch.setattr(early_memory,'witnesses',lambda *a:({'train':tr,'validation':val,'outside_validation':val},{}))
    monkeypatch.setattr(early_memory,'memory_snapshot',lambda _:dict(rss_bytes=2**30,peak_rss_bytes=2**30,
                                                                gpu_peak_bytes=torch.cuda.max_memory_allocated()))
    proof=early_memory.run(s,'CONCAT',None,'cuda',tmp_path)
    assert proof['passed'] and len(proof['parity'])==4
    assert proof['pointwise_statistics']['input_gradient_calls']==12
    pair_convolution.validate_parities(proof['parity'])
    bad=deepcopy(proof['parity']);bad[2]['tolerance']['atol']=1.
    with pytest.raises(ValueError,match='parity contract'):pair_convolution.validate_parities(bad)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual CUDA full-objective miniature; forced small support')
def test_v3_integrated_acceptance_and_native_export(tmp_path,monkeypatch):
    from test_luka_sixclass_wide_b256_oscar import test_real_weaver_integrated_qualification
    force_small_pointwise(monkeypatch)
    original=startup.run;proofs=[]
    def qualified(*args,**kwargs):
        proof=original(*args,**kwargs,pair_storage=True,pointwise_dgrad=True);proofs.append(proof);return proof
    monkeypatch.setattr(startup,'run',qualified)
    test_real_weaver_integrated_qualification('CONCAT',tmp_path,monkeypatch)
    pair_convolution.validate_stats(proofs[0]['pointwise_statistics'],require_fallback=True)


def test_pointwise_does_not_split_batchnorm_or_change_state_keys():
    native=model.new_model('CONCAT').train()
    fixed=model.new_model('CONCAT',pair_storage=True,pointwise_dgrad=True).train()
    assert native.state_dict().keys()==fixed.state_dict().keys()
    fixed.load_state_dict(native.state_dict(),strict=True)
    guard=fixed.pair_convolution;guard.force=True
    raw=cache(coordinate='CONCAT',rows=12).batch(np.arange(6));seen=[]
    handles=[m.register_forward_pre_hook(lambda module,args:seen.append(tuple(args[0].shape)))
             for m in fixed.mod.pair_embed.modules() if isinstance(m,torch.nn.BatchNorm1d)]
    torch.manual_seed(133);a=model.forward(native,raw,'cpu',bf16=False)
    torch.manual_seed(133);b=model.forward(fixed,raw,'cpu',bf16=False)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    b.square().mean().backward()
    for h in handles:h.remove()
    assert len(seen)==5 and len({s[-1] for s in seen})==1
    for name,value in native.named_buffers():
        torch.testing.assert_close(value,dict(fixed.named_buffers())[name],rtol=0,atol=0)
    pair_convolution.validate_stats(guard.stats,require_fallback=True)
    deployed=fixed.deployed();deployed.pair_convolution.force=True
    before=deployed.pair_convolution.stats['fallback_calls']
    model.forward(deployed,raw,'cpu',bf16=False)
    assert deployed.pair_convolution.stats['fallback_calls']==before
    assert deployed.state_dict().keys()==native.state_dict().keys()
    with pytest.raises(ValueError):model.new_model('DE000',pair_storage=True,pointwise_dgrad=True)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='Native fallback GEMM parity on CUDA')
def test_input_gradient_and_native_parameter_gradients_match_fallback_exactly(dtype):
    native=model.new_model('CONCAT').cuda().mod.pair_embed.embed[10]
    fixed_net=model.new_model('CONCAT',pair_storage=True,pointwise_dgrad=True).cuda()
    fixed_net.pair_convolution.force=True
    fixed=fixed_net.mod.pair_embed.embed[10];fixed.load_state_dict(native.state_dict())
    native.to(dtype);fixed.to(dtype)
    x=torch.randn(1,64,10001,device='cuda',dtype=dtype,requires_grad=True)
    z=x.detach().clone().requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False):
        a=native(x);b=fixed(z);dy=torch.randn_like(a)
        ga=torch.autograd.grad(a,(x,native.weight,native.bias),dy)
        gb=torch.autograd.grad(b,(z,fixed.weight,fixed.bias),dy)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    for left,right in zip(ga,gb):torch.testing.assert_close(left,right,rtol=0,atol=0)
    torch.cuda.empty_cache()

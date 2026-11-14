"""Balanced scientific semantics and production adapters; no remote acceptance claims."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from hlt_classification.luka_fullstat_balanced_ladder import contracts as c, sampling, population, model, contrastive


@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def sampler_fixture():
    # Deliberately unequal cell occupancy, including sharply different priors.
    count=(np.arange(198)%7+1)*3
    codes=np.repeat(np.arange(198),count)
    return sampling.BalancedSampler(codes//18,250+100*(codes%18))


def test_frozen_budget_targets_and_occupancy():
    assert population.occupancy().sum()==c.ELIGIBLE==2425536
    assert c.STEPS==18950 and c.DRAWS%128==64 and c.EPOCHS*c.DRAWS==48510720
    assert len(c.STAGES)==6 and c.TEACHERS['DE000']==('OFFLINE','CONCAT','UE100','DE050')
    assert c.TEACHERS['UE100']==('OFFLINE','CONCAT','M0HLT')
    assert c.TEACHERS['DE050']==('OFFLINE','CONCAT','UE100','M0HLT')
    assert c.recipe()['loss_class_weights'] is None
    assert c.coefficient('DE000',3*c.STEPS)==.3
    assert c.coefficient('DE050',3*c.STEPS)==0
    assert c.recipe()['original_counts']['train']==3000000


def test_exact_flat_probability_boundaries_repeat_and_replay():
    s=sampler_fixture()
    a=s.draw(1,draws=198000);b=s.draw(1,draws=198000)
    np.testing.assert_array_equal(a,b)
    assert not np.array_equal(a,s.draw(2,draws=len(a)))
    actual=np.bincount(s.codes[a],minlength=198)
    assert np.max(np.abs(actual-1000))<140
    assert len(np.unique(a))<len(a)
    before=np.random.get_state();s.draw(3,draws=10);after=np.random.get_state()
    assert before[0]==after[0] and np.array_equal(before[1],after[1])
    labels=np.r_[s.labels,[0,0,0]];pt=np.r_[s.pt,[199.,200.,2000.]]
    edge=sampling.BalancedSampler(labels,pt)
    assert len(edge.indices)==len(s.labels)+1 and len(s.labels)+1 in edge.indices
    assert len(s.labels) not in edge.indices and len(s.labels)+2 not in edge.indices
    with pytest.raises(ValueError):sampling.BalancedSampler(labels,pt,role='validation')
    with pytest.raises(ValueError):sampling.BalancedSampler(np.array([0]),np.array([250.]))
    with pytest.raises(ValueError):sampling.BalancedSampler(labels,pt,expected_counts=population.occupancy())
    evidence=s.evidence(a)
    assert sum(map(sum,evidence['class_bin_counts']))==len(a)
    order=s.draw(5,draws=320)
    batches=list(s.batches(order))
    assert [len(b) for _,b in batches]==[128,128,64]
    np.testing.assert_array_equal(np.concatenate([b for _,b in s.batches(order,start_update=1)]),order[128:])


@pytest.mark.parametrize('stage',c.STAGES)
def test_native_parity_direct_main_ce_and_auxiliary_gradients(monkeypatch,stage):
    from test_luka_embedding_aux import tiny
    from test_luka_attention import cache
    from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
    from hlt_classification.luka_u100_multiview.teachers import forward
    tiny(monkeypatch)
    torch.manual_seed(c.INITIALIZATION_SEED);baseline=DelphesParticleTransformer()
    state=torch.get_rng_state().clone();net=model.new_model(stage)
    assert torch.equal(state,torch.get_rng_state())
    assert all(torch.equal(v,baseline.state_dict()[k]) for k,v in net.native_state().items())
    data=cache(coordinate=c.COORDINATES[stage],rows=22);ids=np.array([1,1,2,3,4])
    rng=np.random.default_rng(3)
    banks={t:SimpleNamespace(targets=rng.normal(size=(22,128)).astype(np.float32),
                            probabilities=np.full((22,11),1/11,np.float32)) for t in c.TEACHERS[stage]}
    sampler=None;candidates=None
    if stage=='DE000':
        keys=torch.nn.functional.normalize(torch.randn(22,128),dim=-1)
        sampler=SimpleNamespace(candidate_keys=lambda ix:keys[ix])
        candidates=np.tile(np.arange(1025)%22,(len(ids),1));candidates[:,0]=ids
    net.eval()
    loss,parts=model.objective(net,data.batch(ids),ids,banks,update=1,steps=2,device='cpu',
                               sampler=sampler,candidates=candidates)
    expected=parts['ce']
    for t in c.TEACHERS[stage]:
        expected=expected+10/len(banks)*parts['embedding_'+t]+1/len(banks)*parts['kd_'+t]
    if stage=='DE000':expected=expected+c.coefficient(stage,1,2)*parts['matching']
    torch.testing.assert_close(loss,expected)
    main=list(net.mod.fc.parameters());cegrad=torch.autograd.grad(parts['ce'],main,retain_graph=True)
    loss.backward()
    for p,g in zip(main,cegrad):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    for group in (net.predictors,net.logit_heads):
        for head in group.values():assert any(p.grad.count_nonzero() for p in head.parameters())
    native=net.deployed();net.capture=False
    with torch.no_grad():
        expected=forward(net,data.batch(ids),'cpu',bf16=False)
        for head in (*net.predictors.values(),*net.logit_heads.values()):
            for p in head.parameters():p.fill_(float('nan'))
        torch.testing.assert_close(expected,forward(net,data.batch(ids),'cpu',bf16=False),rtol=0,atol=0)
        torch.testing.assert_close(expected,forward(native,data.batch(ids),'cpu',bf16=False),rtol=0,atol=0)


@pytest.mark.parametrize('ties',[False,True])
def test_occurrence_matching_serial_optimized_prefetch_and_duplicates(ties):
    from test_luka_u100_contrastive import sampler_fixture as donor_fixture
    r=donor_fixture()
    keys=r.keys.numpy().copy()
    if ties:keys[:]=0
    args=(keys,r.identities,r.labels)
    serial=contrastive.SerialReference(*args,root_seed=c.NEGATIVE_SEED)
    fast=contrastive.Sampler(*args)
    anchors=np.array([0,0,1,500,1,2,21,20,19]);positions=np.arange(len(anchors))
    expected,proof=serial.occurrences(anchors,2,positions)
    actual,evidence=fast.sample(anchors,2,positions)
    np.testing.assert_array_equal(actual,expected);assert proof==evidence
    assert not np.array_equal(actual[0],actual[1])
    for a,row in zip(anchors,actual):
        assert row[0]==a and a not in row[1:] and len(np.unique(row))==1025
    with contrastive.Lookahead(fast,anchors,2,batch_size=3) as batches:
        np.testing.assert_array_equal(np.concatenate([v for _,v in batches]),expected)
    with contrastive.CPUWorkers(fast,workers=2):
        actual,evidence=fast.sample(anchors,2,positions)
        np.testing.assert_array_equal(actual,expected);assert proof==evidence


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Full architecture native CUDA parity requires a local GPU')
@pytest.mark.parametrize('stage',c.STAGES)
def test_full_native_cuda_installed_weaver_and_removable_heads(stage):
    from test_luka_attention import cache
    from hlt_classification.jetclass2_delphes.acceptance import installed_parity
    from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
    from hlt_classification.luka_u100_multiview.teachers import forward
    from hlt_classification.luka_fullstat_readout.training import finite_gradients
    torch.cuda.empty_cache();data=cache(coordinate=c.COORDINATES[stage]);ix=np.arange(4);device='cuda'
    with parity_backend(device):installed_parity(data,device=device)
    net=model.new_model(stage).to(device);banks={t:SimpleNamespace(targets=np.zeros((22,128),np.float32),
        probabilities=np.full((22,11),1/11,np.float32)) for t in c.TEACHERS[stage]}
    matching=None;candidates=None
    if stage=='DE000':
        keys=torch.nn.functional.normalize(torch.randn(22,128,device=device),dim=-1)
        matching=SimpleNamespace(candidate_keys=lambda ids:keys[ids])
        candidates=np.tile(np.arange(1025)%22,(4,1));candidates[:,0]=ix
    net.train()
    loss,parts=model.objective(net,data.batch(ix),ix,banks,update=3*c.STEPS,steps=c.STEPS,
        device=device,sampler=matching,candidates=candidates)
    grads=torch.autograd.grad(parts['ce'],tuple(net.mod.fc.parameters()),retain_graph=True)
    loss.backward();finite_gradients(net)
    for p,g in zip(net.mod.fc.parameters(),grads):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
    native=net.deployed().to(device).eval();net.eval();net.capture=False
    with parity_backend(device),torch.inference_mode():
        for bf16 in (False,True):
            torch.testing.assert_close(forward(net,data.batch(ix),device,bf16=bf16),
                                       forward(native,data.batch(ix),device,bf16=bf16),rtol=0,atol=0)

"""Batched loss/gradient parity against the independently retained scalar objective."""
import numpy as np
import pytest
import torch
from test_luka_direct_reconstruction import target,score
from hlt_classification.luka_next_supervision_hydra import reconstruction_loss as scalar,reconstruction_batched as batch
from hlt_classification.luka_next_supervision_hydra import reconstruction,contracts,submission


def examples(counts,device='cpu'):
    k=max(max(counts),4); predictions={name:torch.stack([score(k)[name].detach() for _ in counts]).to(device).requires_grad_()
        for name in score(k)}
    rng=np.random.default_rng(12); targets=[]
    for n in counts:
        t=target(n); x=t['values']
        t['valid']=rng.random((n,4))>.5; x[:,18:22]=t['valid']; x[:,14:18][~t['valid']]=np.nan
        x[:,5:11]=0; x[np.arange(n),5+rng.integers(0,6,n)]=1
        x[:,11:14]=0; x[np.arange(n),11+rng.integers(0,3,n)]=1
        t['mean']=rng.normal(0,.1,22); t['scale']=rng.uniform(.8,1.2,22)
        targets.append(t)
    return predictions,targets


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('extra',[False,True])
@pytest.mark.parametrize('counts',[(0,0),(0,1,4),(64,3,40)])
def test_scalar_batched_losses_components_and_all_output_gradients(device,extra,counts):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    a,targets=examples(counts,device); b={k:v.detach().clone().requires_grad_() for k,v in a.items()}
    expected,ep=scalar.rows(a,targets,extra); actual,ap=batch.rows(b,targets,extra)
    torch.testing.assert_close(actual,expected,rtol=3e-6,atol=2e-6)
    assert ep.keys()==ap.keys()
    for name in ep: torch.testing.assert_close(ap[name],ep[name],rtol=3e-6,atol=2e-6,msg=name)
    expected.sum().backward(); actual.sum().backward()
    for name in a:
        ag=a[name].grad if a[name].grad is not None else torch.zeros_like(a[name])
        bg=b[name].grad if b[name].grad is not None else torch.zeros_like(b[name])
        torch.testing.assert_close(bg,ag,rtol=3e-5,atol=3e-6,msg=name)


def test_canonical_target_permutation_duplicate_ties_empty_and_parallel_assignment():
    scores,targets=examples([40,0,30]); targets[0]['values'][1]=targets[0]['values'][0]; targets[0]['valid'][1]=targets[0]['valid'][0]
    t,counts=batch.pack(targets,40,'cpu'); slots,chosen=batch.assignments(scores,t,counts)
    for i,n in enumerate(counts):
        s={k:v[i] for k,v in scores.items()}; st=scalar.target(targets[i],'cpu'); a,c=scalar.match(s,st)
        expected=np.empty(n,np.int64); expected[c]=a
        np.testing.assert_array_equal(slots[i,:n].numpy(),expected)
        assert int(chosen[i].sum())==n and len(set(slots[i,:n].tolist()))==n
    permuted=[dict(t,values=t['values'][::-1].copy(),valid=t['valid'][::-1].copy()) for t in targets]
    a,_=batch.rows(scores,targets); b,_=batch.rows(scores,permuted)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert batch.solver_pool()._max_workers==6


@pytest.mark.parametrize('change', ['normalization','valid_field','category','validity','frame','aggregate','count'])
def test_invalid_targets_fail_before_any_solver(change,monkeypatch):
    scores,targets=examples([2]); t=targets[0]
    if change=='normalization': t['scale'][0]=0
    if change=='valid_field': t['values'][0,0]=np.nan
    if change=='category': t['values'][0,5:11]=0
    if change=='validity': t['values'][0,18:22]=1-t['valid'][0]
    if change=='frame': t['frame'][2]=0
    if change=='aggregate': t['aggregate'][0]=np.nan
    if change=='count': targets=[target(5)]
    monkeypatch.setattr(batch,'solve',lambda *a:pytest.fail('Invalid targets reached solver'))
    with pytest.raises(ValueError): batch.rows(scores,targets)


def test_nonfinite_predictions_and_inverse_are_not_hidden():
    scores,targets=examples([2]); scores['count'].data[0,0]=float('nan')
    with pytest.raises(ValueError,match='prediction'): batch.rows(scores,targets)
    scores,targets=examples([2]); scores['kinematic'].data[0,:,0]=1000
    with pytest.raises(ValueError,match='four-vector'): batch.rows(scores,targets,True)


def test_R3_is_fresh_execution_with_unchanged_recipe_and_resource_limits():
    s=reconstruction.registration()
    assert s['execution_version']=='R3' and s['reconstruction_kernel']=='batched_exact_chunk_v1'
    assert s['assignment_workers']==batch.ASSIGNMENT_WORKERS==s['cpus']==6
    assert s['fit_max_minutes']==2880 and s['resource_policy']['safety_multiplier']==2
    assert s['recipe']['content_hash']=='90b582b9f37955f13217ec36f3f724c83ccecd7c88c9d9983ce8bce6bda345cc'
    assert submission.automatic(s) and contracts.authorization(s)=='AUTHORIZE_DIRECT_RECONSTRUCTION_AUTO_V3'

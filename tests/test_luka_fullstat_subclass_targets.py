"""TRAIN-only target fitting, conditional losses and frozen-bank recovery."""
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
import torch
from hlt_classification.data.cache_contracts import (write_immutable_json,load_json,
    deterministic_npz_bytes,atomic_publish_bytes,with_content_hash)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullstat_subclass import targets,model,contracts as c,banks
from test_luka_attention import cache
from test_luka_fullstat_multi3_contrastive import bank_records


def arrays():
    rng=np.random.default_rng(91);labels=np.repeat(np.arange(11,dtype=np.int64),30)
    embedding=rng.normal(size=(len(labels),128)).astype(np.float32)
    return embedding,labels,np.zeros(128),np.ones(128),rng.normal(size=(11,128)),rng.normal(size=11)


def test_fit_reproducibility_statistics_softness_and_no_global_rng():
    a=arrays();before=np.random.get_state();state=torch.get_rng_state().clone()
    q,p,d=targets.fit(*a);q2,p2,d2=targets.fit(*a)
    np.testing.assert_array_equal(q,q2)
    assert d==d2 and all(np.array_equal(p[k],p2[k]) for k in p)
    np.testing.assert_array_equal(before[1],np.random.get_state()[1]);assert torch.equal(state,torch.get_rng_state())
    np.testing.assert_allclose(q.sum(1),1,atol=1e-6);assert np.all(q[a[1]!=0,5:]==0)
    for label,count in enumerate(c.COUNTS_PER_CLASS):
        ix=np.flatnonzero(a[1]==label);raw=a[0][ix].astype(np.float64)*a[3]+a[2]
        logits=raw@a[4].T+a[5];other=np.delete(np.arange(11),label)
        margins=logits[:,other]-logits[:,label,None]
        np.testing.assert_allclose(p['margin_mean'][label],margins.mean(0))
        np.testing.assert_allclose(p['margin_scale'][label],np.maximum(margins.std(0),.001))
        assert d[label]['median_max_membership']==pytest.approx(.8,abs=1e-5)
        assert sum(d[label]['hard_occupancy'])==len(ix)
        x=np.concatenate((a[0][ix]*np.sqrt(.75/128),
            ((margins-p['margin_mean'][label])/p['margin_scale'][label]*np.sqrt(.25/10)).astype(np.float32)),1)
        recomputed=targets.probabilities(targets.distances(x,p['centers'][label,:count]),p['temperatures'][label])
        np.testing.assert_array_equal(q[ix,:count],recomputed.astype(np.float32))


def test_softness_degeneracy_does_not_fail_science():
    d=np.zeros((40,5));t,m=targets.calibrate(d)
    assert t==1e-8 and m==.2
    np.testing.assert_allclose(targets.probabilities(d,t),.2)
    a=list(arrays());a[0].fill(0);a[4].fill(0);a[5].fill(0)
    q,_,diagnostics=targets.fit(*a)
    assert np.isfinite(q).all() and all(not x['softness_target_reached'] for x in diagnostics)


def test_distance_scaling_and_conditional_gradient_direction():
    x=np.r_[np.ones(128)*np.sqrt(.75/128),np.ones(10)*np.sqrt(.25/10)][None]
    assert targets.distances(x,np.zeros((1,138)))[0,0]==pytest.approx(1.)
    labels=torch.tensor([0,1,10]);logits=torch.randn(3,60,requires_grad=True)
    q=torch.zeros(3,10);q[0]=.1;q[1:,:5]=.2;q.requires_grad_(True)
    loss=model.conditional_kl(logits,labels,q)
    explicit=[]
    for i,(offset,k) in enumerate(((0,10),(10,5),(55,5))):
        explicit.append((q[i,:k].detach()*(q[i,:k].detach().log()-logits[i,offset:offset+k].log_softmax(0))).sum())
    torch.testing.assert_close(loss,torch.stack(explicit).mean())
    loss.backward();assert q.grad is None
    for i,(offset,k) in enumerate(((0,10),(10,5),(55,5))):
        assert logits.grad[i,offset:offset+k].abs().sum()>0
        logits.grad[i,offset:offset+k]=0
    assert logits.grad.count_nonzero()==0


def target_spec(tmp_path,rows=330):
    tr=cache(coordinate='D000',rows=rows);records,raw=bank_records(tmp_path,tr)
    off=records['OFFLINE'];path=Path(off['refs']['checkpoint']['path'])
    rng=np.random.default_rng(14)
    torch.save({'mod.fc.0.weight':torch.from_numpy(rng.normal(size=(11,128)).astype(np.float32)),
                'mod.fc.0.bias':torch.from_numpy(rng.normal(size=11).astype(np.float32))},path)
    off['refs']['checkpoint']=file_ref(path)
    m=load_json(off['manifest']['path']);m['checkpoint']=off['refs']['checkpoint']
    m['parents']['checkpoint']=off['refs']['checkpoint']['sha256'];m=with_content_hash(m)
    # Fixture is still under construction, before the immutable production reader.
    import json
    Path(off['manifest']['path']).write_text(json.dumps(m))
    off['manifest']=file_ref(off['manifest']['path']);off['manifest_hash']=m['content_hash']
    spec=dict(root=str(tmp_path/'study'),counts=dict(train=len(tr)),reuse=dict(banks=records),
        anchor=dict(evidence=dict(prepared=tr.foundation_sha256)),resume=None)
    Path(spec['root']).mkdir()
    return spec,tr


def test_real_bank_readers_build_once_role_guards_and_recovery(tmp_path):
    spec,tr=target_spec(tmp_path)
    from hlt_classification.luka_fullstat_embedding_sweep.banks import Bank as Offline
    rec=spec['reuse']['banks']['OFFLINE'];offline=Offline(Path(rec['manifest']['path']).parent,tr,refs=rec['refs'])
    bank=targets.prepare(spec,tr,offline)
    assert bank.targets.shape==(330,10) and not bank.targets.flags.writeable
    wrong=deepcopy(tr);wrong.role='validation'
    with pytest.raises(PermissionError):targets.prepare(spec,wrong,offline)
    with pytest.raises(PermissionError):targets.Bank(spec,wrong)
    # Same frozen bank on explicit recovery, no reclustering.
    original=Path(spec['root'])/'campaign_spec.json';write_immutable_json(original,spec)
    resumed=dict(spec,root=str(tmp_path/'retry'),resume=dict(spec=file_ref(original)))
    Path(resumed['root']).mkdir();got=targets.prepare(resumed,tr,offline)
    assert got.manifest==bank.manifest
    for n in targets.FILES:assert (Path(spec['root'])/'subclasses'/n).read_bytes()==(Path(resumed['root'])/'subclasses'/n).read_bytes()
    with pytest.raises(FileExistsError):targets.prepare(resumed,tr,offline)
    (Path(resumed['root'])/'subclasses/targets.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError):targets.Bank(resumed,tr)


def test_nonfinite_and_bad_labels_rejected():
    a=list(arrays());a[0][0,0]=np.nan
    with pytest.raises(ValueError):targets.fit(*a)
    a=list(arrays());a[1][0]=11
    with pytest.raises(ValueError):targets.fit(*a)


def test_real_four_bank_pipeline_with_subclasses(tmp_path):
    spec,tr=target_spec(tmp_path,rows=6006)
    loaded,sampler=banks.load(spec,tr,'cpu')
    assert set(loaded)==set(c.TEACHERS)
    assert sampler.subclass_bank.targets.shape==(len(tr),10)
    np.testing.assert_array_equal(sampler.subclass_bank.identities,tr.identities)
    np.testing.assert_allclose(sampler.keys.norm(dim=1),1.,atol=2e-7)
    assert targets.check_manifest(spec)['parents']==targets.parents(spec)
    # Metadata can be freshly checksummed yet still violate the science contract.
    path=Path(spec['root'])/'subclasses/manifest.json';m=load_json(path)
    m['recipe']['coefficient']=1.
    import json
    path.write_text(json.dumps(with_content_hash(m)))
    with pytest.raises(ValueError,match='semantics'):targets.check_manifest(spec)

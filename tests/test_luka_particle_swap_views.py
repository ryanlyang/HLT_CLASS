"""Physical construction and real ROOT readback, not remote acceptance evidence."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest

from hlt_classification.cms2jc2_response.bridge import Particles, p4_from_coordinates
from hlt_classification.cms_proxy_ladder.views import build_view as old_view
from hlt_classification.luka_fullsim import stage2, regular_cache
from hlt_classification.luka_particle_swap import views, contracts as c, data
from test_luka_fullsim_stage2 import dataset, prepared
from test_luka_fullsim_alljets import all_prepared


def particles(n=15, richer=False):
    i = np.arange(n)
    charge = np.where(i%3 == 0, 0, 1).astype(np.int8)
    category = np.where(charge == 0, 1, 0).astype(np.int8)
    valid = np.repeat((charge != 0)[:, None], 4, axis=1)
    if richer:
        valid[(i%4)==1, 2:] = False
    tracking = np.tile([.02, -.03, .004, .005], (n,1))*(2 if richer else 1)
    tracking[~valid] = 0
    p4 = p4_from_coordinates(5+i*(1.5 if richer else 1), .03*i,
                             .08*i+(0.02 if richer else 0), .1)
    return Particles(p4, charge, category, tracking, valid,
                     tuple(('off' if richer else 'hlt')+str(k) for k in i))


@pytest.mark.parametrize('n',[0,1,2,3,4,5,7,15,101,512])
def test_stable_nested_balanced_counts_and_unmatched(n):
    hlt = particles(n)
    mapping = np.arange(n, dtype=np.int64)
    if n > 4:
        mapping[::4] = -1
        mapping[mapping>=0] = np.arange(np.sum(mapping>=0))
    g = views.replacement_groups(hlt, mapping, identity='a'*64)
    np.random.seed(349)
    np.random.random(100)
    np.testing.assert_array_equal(g, views.replacement_groups(hlt, mapping, identity='a'*64))
    assert g.dtype == np.int8 and np.array_equal(g == -1, mapping == -1)
    sizes = np.bincount(g[g>=0], minlength=3)
    assert max(sizes)-min(sizes) <= 1
    off = particles(int(np.sum(mapping>=0)), True)
    values, actual = views.all_views(identity='a'*64,hlt=hlt,offline=off,mapping=mapping)
    np.testing.assert_array_equal(actual,g)
    assert {i for i in range(n) if g[i]>=2} <= {i for i in range(n) if g[i]>=1}
    for coordinate, threshold in [('D066',1),('D033',2)]:
        for field in ('p4','charge','category','tracking','valid'):
            expected = np.array(getattr(hlt,field), copy=True)
            keep = np.flatnonzero(g>=threshold)
            expected[keep] = getattr(off,field)[mapping[keep]]
            np.testing.assert_array_equal(getattr(values[coordinate],field),expected)
        assert len(values[coordinate])==n
        assert values[coordinate].keys==tuple('proxy:'+s for s in hlt.keys)
    assert values['D000'] is hlt
    expected = old_view(identity='a'*64,proxy=hlt,offline=off,mapping=mapping,coordinate='U100')
    for field in ('p4','charge','category','tracking','valid'):
        np.testing.assert_array_equal(getattr(values['U100'],field),getattr(expected,field))


def test_endpoint_native_inference_needs_no_offline_or_assignment():
    hlt=particles()
    assert views.build_view(identity=None,hlt=hlt,offline=None,mapping=None,coordinate='D000') is hlt


def test_features_recomputed_from_hybrid_jet_not_spliced_columns():
    hlt,off=particles(),particles(richer=True)
    mapping=np.arange(len(hlt),dtype=np.int64)
    values,groups=views.all_views(identity='b'*64,hlt=hlt,offline=off,mapping=mapping)
    converted={k:stage2.build_inputs(v,capacity=512) for k,v in values.items()}
    for coordinate,cut in [('D066',1),('D033',2)]:
        # p4 is atomic; jet-relative/normalized features depend on the entire hybrid.
        assert converted[coordinate].features.shape==converted['D000'].features.shape
        spliced=converted['D000'].features.copy()
        keep=groups>=cut
        spliced[keep]=converted['U100'].features[keep]
        assert not np.array_equal(converted[coordinate].features,spliced)
        assert set(vars(converted[coordinate]))=={'features','vectors'}
        assert converted[coordinate].features.shape[-1]==17
    assert not np.array_equal(values['D066'].p4,old_view(
        identity='b'*64,proxy=hlt,offline=off,mapping=mapping,coordinate='D066').p4)


def test_schedule_has_no_offline_label_or_global_rng_dependency():
    import inspect
    assert set(inspect.signature(views.replacement_groups).parameters)=={'hlt','mapping','identity'}
    hlt=particles(60); mapping=np.arange(60)
    g=views.replacement_groups(hlt,mapping,identity='c'*64)
    # Same shell/assignment, radically changed offline quality -> same schedule.
    off=particles(60,True)
    other=replace(off,p4=off.p4*10,tracking=off.tracking*50)
    _,a=views.all_views(identity='c'*64,hlt=hlt,offline=off,mapping=mapping)
    _,b=views.all_views(identity='c'*64,hlt=hlt,offline=other,mapping=mapping)
    np.testing.assert_array_equal(g,a); np.testing.assert_array_equal(g,b)
    # A broad monotone spectrum is spread across all groups, not highest pT last.
    shares=np.array([hlt.pt[g==b].sum()/hlt.pt.sum() for b in range(3)])
    assert max(shares)-min(shares)<.15


def test_invalid_mapping_identity_and_tampered_schedule_fail():
    hlt,off=particles(5),particles(5,True)
    for mapping in ([0,0,1,2,3],[0,1,2,3,5],[0,1,-2,3,4],[0,1,2,3],np.arange(5)+.5):
        with pytest.raises((ValueError,IndexError)):
            views.all_views(identity='a'*64,hlt=hlt,offline=off,mapping=mapping)
    with pytest.raises(ValueError,match='identity'):
        views.replacement_groups(hlt,np.arange(5),identity='label=1')
    with pytest.raises(ValueError,match='schedule'):
        views.build_view(identity='a'*64,hlt=hlt,offline=off,mapping=np.arange(5),coordinate='D066',groups=np.zeros(5))


def fixture_gate(all_prepared):
    old,manifest,p,inv,splits=all_prepared
    anchor=c.artifact('ANCHOR',evidence=dict(prepared_root=str(Path(old['root'])/'prepared'),
        raw_container=old['input_container'],prepared=p['content_hash'],cache=manifest['content_hash']))
    gate=c.artifact('GATE',anchor=anchor,input_identity=c.input_identity(anchor),counts=p['counts'],memory_mb=128000)
    return gate,(manifest,p,inv,splits)


def test_real_root_rebuild_exact_endpoints_no_persistent_cache_or_test(all_prepared,monkeypatch):
    gate,meta=fixture_gate(all_prepared)
    old={ (r,co):regular_cache.load(gate['anchor']['evidence']['prepared_root'],role=r,coordinate=co,max_ram_bytes=2**30)
          for r in ('train','validation') for co in ('U100','D000') }
    seen=[]; selected=regular_cache.selected_file_rows
    def rows(container,inv,member):
        seen.append(member)
        assert member in meta[3]['memberships']['train']+meta[3]['memberships']['validation']
        yield from selected(container,inv,member)
    monkeypatch.setattr(regular_cache,'selected_file_rows',rows)
    monkeypatch.setattr(regular_cache,'load',lambda *a,**kw:pytest.fail('No durable feature cache read'))
    bundle=data.build(gate,authenticated=meta)
    assert seen==meta[3]['memberships']['train']+meta[3]['memberships']['validation']
    for key,expected in old.items():
        actual=data.get(bundle,*key)
        assert actual.foundation_sha256==gate['input_identity']['content_hash']
        for k,v in expected.batch(np.arange(len(expected))).items():
            np.testing.assert_array_equal(v,actual.batch(np.arange(len(actual)))[k])
    assert sum(bundle['original_validation'])==meta[3]['comparison_rows']
    c.validate(bundle['audit'],'INPUT_AUDIT',dict(inputs=gate['input_identity']['content_hash']))
    for role in ('test','final_test','unused'):
        with pytest.raises(PermissionError):data.get(bundle,role,'D000')
    assert bundle['audit']==data.build(gate,authenticated=meta)['audit']


def test_rebuild_rejects_changed_endpoints_identity_or_ram(all_prepared):
    gate,meta=fixture_gate(all_prepared)
    bad=deepcopy(meta); bad[1]['summaries']['train']['U100']['maximum']+=1
    with pytest.raises(ValueError,match='endpoint'):data.build(gate,authenticated=bad)
    bad=deepcopy(gate); bad['input_identity']['content_hash']='e'*64
    with pytest.raises(ValueError,match='identity'):data.build(bad,authenticated=meta)
    bad=dict(gate,memory_mb=1)
    with pytest.raises(MemoryError,match='construction'):data.build(bad,authenticated=meta)

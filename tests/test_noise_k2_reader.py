"""Real synthetic relocated ROOT/banks exercise the new K2 consumer adapter."""
from pathlib import Path

import numpy as np
import pytest

from test_literature_proxy_consumer import (
    parent, parent_v2, final_pilot, study, relocated, opened,
)
from hlt_classification.noise_k2 import contracts as c, data, campaign, views
from hlt_classification.scouting.hcwdl_fullcard_salience_contracts import CANDIDATES
from hlt_classification.data.cache_contracts import sha256_file


@pytest.fixture
def prepared(relocated,tmp_path,monkeypatch):
    root,digest,_=relocated
    reader=opened(relocated)
    settings=c.registration()
    settings['counts']=dict(train=22,validation=11)
    spec=c.artifact('CAMPAIGN_SPEC',**settings,dataset_root=str(root),manifest_sha256=digest,
        candidate=CANDIDATES[1],campaign_root=str(tmp_path/'consumer'),shards=campaign.shards(reader))
    # Existing synthetic producer has one val row/class. This test exercises
    # relocation/selection/maps/cache, not production scoring. Partition unit
    # tests separately exercise the unmodified >=4/class guard and 50/25/25.
    monkeypatch.setattr(data,'partition_codes',lambda ids,labels:np.zeros(len(labels),np.uint8))
    return spec,reader


def test_real_relocated_selection_matching_cache_and_hlt_only_reads(prepared,monkeypatch):
    spec,reader=prepared
    root=Path(spec['campaign_root'])
    before={p:sha256_file(p) for p in Path(spec['dataset_root']).rglob('*') if p.is_file()}
    report=data.select_population(spec)
    assert report['roles']['train']['rows']==22 and report['roles']['validation']['rows']==11
    train=data.selected(spec,'train')[1]; val=data.selected(spec,'validation')[1]
    assert not {bytes(v) for v in train['identities']} & {bytes(v) for v in val['identities']}
    assert np.bincount(train['labels']).tolist()==[2]*11
    # Same spec replay must produce byte-identical immutable selection.
    assert data.select_population(spec)==report
    for shard in spec['shards']:
        r=data.assignment(spec,shard)
        assert r['all_coordinate_invariants']
    foundation=data.foundation(spec)
    assert foundation['counts']['jets']==33
    assert foundation['capacity']>=foundation['maximum_particles']['expanded']
    assert foundation['assignments']
    rich=data.cache(spec,'train','D100')
    middle=data.cache(spec,'train','D050')
    assert len(rich)==len(middle)==22
    assert rich.nbytes <= foundation['ram_upper_bytes']
    assert np.isfinite(rich.batch(np.arange(22))['features']).all()
    assert before=={p:sha256_file(p) for p in before}
    # Corrupting (or even asking for) assignment/offline capabilities is forbidden
    # for either deployable endpoint. Proxy-bank authentication remains enabled.
    monkeypatch.setattr(data,'authenticated_open',lambda *a,**k:pytest.fail('D000 opened ROOT'))
    monkeypatch.setattr(data,'load_assignment',lambda *a,**k:pytest.fail('D000 loaded assignment'))
    hlt=data.cache(spec,'train','HLT_X1')
    x3=data.cache(spec,'train','D000')
    assert len(hlt)==len(x3)==22
    np.testing.assert_array_equal(hlt.identities,rich.identities)
    np.testing.assert_array_equal(x3.identities,rich.identities)
    assert sum(b.features.shape[0] for b in x3.blocks)==3*sum(b.features.shape[0] for b in hlt.blocks)
    assert (root/'foundation.json').is_file()


def test_selected_reader_matches_public_reader_raw_identity_and_no_test(prepared,monkeypatch):
    spec,reader=prepared
    data.select_population(spec)
    selected=data.selected(spec,'train')[1]
    actual={r.identity:r for r in reader.iter_pairs('train',labels=True)}
    before=Path.open
    final_paths={p.resolve() for p in (reader.proxy_root/'attempts').rglob('block_*.npz')
                 if any(s['shard_id'] in p.parts for s in reader.study['shards'] if s['role']=='final_test')}
    def guarded(path,*args,**kwargs):
        assert path.resolve() not in final_paths
        return before(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    matched=0
    for shard in spec['shards']:
        if shard['role']!='train': continue
        rows=data.shard_rows(selected,shard['index'])
        for _,identity,label,proxy,offline in data.iter_selected(reader,shard,rows,paired=True):
            expected=actual[identity]
            assert label==expected.label
            for field in ('p4','charge','category','tracking','valid'):
                np.testing.assert_array_equal(getattr(proxy,field),getattr(expected.proxy,field))
                np.testing.assert_array_equal(getattr(offline,field),getattr(expected.offline,field))
            matched+=1
    assert matched==22


def test_population_crossrole_and_order_changes_fail(prepared):
    spec,reader=prepared
    data.select_population(spec)
    selected=data.selected(spec,'train')[1]
    s=next(s for s in spec['shards'] if s['role']=='train' and np.count_nonzero(selected['shards']==s['index'])>=2)
    rows=data.shard_rows(selected,s['index'])
    bad={k:v[::-1] for k,v in rows.items()}
    with pytest.raises(ValueError,match='order'): list(data.iter_selected(reader,s,bad,paired=True))
    bad={k:v.copy() for k,v in rows.items()}; bad['identities'][0,0]^=1
    with pytest.raises(ValueError,match='identity'): list(data.iter_selected(reader,s,bad,paired=False))

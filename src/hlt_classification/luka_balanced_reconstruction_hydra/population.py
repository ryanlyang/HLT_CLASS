"""Proportional nested sampling without moving frozen TRAIN/validation/test roles."""
from pathlib import Path
import hashlib
import numpy as np
from hlt_classification.data.cache_contracts import (load_json,load_npz_arrays,write_immutable_json,
    atomic_publish_bytes,deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_sixclass_balanced_oscar import population as donor
from hlt_classification.luka_sixclass_balanced_oscar.sampling import BalancedSampler
from . import contracts as c


def quotas(counts,total):
    counts=np.asarray(counts,np.int64)
    if counts.shape!=(120,) or np.any(counts<0) or type(total)!=int or not 0<total<=int(counts.sum()):
        raise ValueError('Invalid proportional subset budget')
    # Integer arithmetic removes floating-point tie ambiguity.
    numerator=counts*total;result=numerator//int(counts.sum());remainders=numerator%int(counts.sum())
    order=np.argsort(-remainders,kind='stable');result[order[:total-int(result.sum())]]+=1
    if np.any((counts>0)&(result==0)):raise ValueError('Subset budget empties a supported cell')
    return result


def select(identities,labels,pt,total,role):
    if role not in ('train','validation'):raise PermissionError('Test subset selection is sealed')
    ids=np.asarray(identities);labels=np.asarray(labels);pt=np.asarray(pt)
    if ids.shape!=(len(labels),32) or ids.dtype!=np.uint8 or pt.shape!=labels.shape:
        raise ValueError('Malformed parent rows')
    if np.any((labels<0)|(labels>=6)) or not np.isfinite(pt).all() or np.any((pt<600)|(pt>=2600)):
        raise ValueError('Wrong six-class in-range population')
    codes=labels*20+np.searchsorted(c.EDGES,pt,side='right')-1
    counts=np.bincount(codes,minlength=120);wanted=quotas(counts,total);chosen=[]
    domain=f'{c.PREFIX}_SUBSET/v1/{c.SUBSET_SEED}/{role}'.encode()
    for cell in range(120):
        indices=np.flatnonzero(codes==cell)
        ranked=sorted(indices,key=lambda i:(hashlib.sha256(domain+b'\0'+bytes(ids[i])).digest(),bytes(ids[i])))
        chosen.extend(ranked[:int(wanted[cell])])
    return np.asarray(sorted(chosen),np.int64),wanted.reshape(6,20)


def describe(parent,counts=None):
    arrays=parent[4];counts=c.COUNTS if counts is None else counts;selected={};out={}
    for role in ('train','validation'):
        ix,cells=select(*(arrays[role+'_'+k] for k in ('identities','labels','pt')),counts[role],role)
        selected[role]=ix
        out[role+'_parent_indices']=ix
        for k in ('identities','labels','pt'):out[role+'_'+k]=arrays[role+'_'+k][ix]
    sampler=BalancedSampler(out['train_labels'],out['train_pt'])
    if len(sampler.indices)!=counts['train']:raise ValueError('Subset lost TRAIN rows')
    maxima={r:dict(hlt=0,offline=0) for r in counts};particles={r:dict(hlt=0,offline=0) for r in counts}
    cursors=dict(train=0,validation=0)
    for record,member,roles in zip(parent[1]['files'],parent[2],parent[3],strict=True):
        for role,mask in list(donor.masks(member,roles).items())[:2]:
            entries=np.flatnonzero(mask);start=cursors[role];cursors[role]+=len(entries)
            a,b=np.searchsorted(selected[role],[start,cursors[role]])
            chosen=entries[selected[role][a:b]-start]
            for side,field in [('hlt','hlt_jet_nparticles'),('offline','jet_nparticles')]:
                n=member[field][chosen]
                maxima[role][side]=max(maxima[role][side],int(n.max()) if len(n) else 0)
                particles[role][side]+=int(n.sum())
    fields=dict(counts=dict(counts),selection='nested_identity_hash_proportional_cells_v1',subset_seed=c.SUBSET_SEED,
        parent_counts=parent[0]['counts'],protected_test_counts=dict(in_range=c.parent.COUNTS['final_test'],outside=c.parent.OUTSIDE['final_test']),
        class_bin_counts=sampler.counts.tolist(),maximum_support=maxima,particles=particles,
        maximum_expected_appearances=counts['train']/(120*int(sampler.counts.min())),
        withheld=dict(train=parent[0]['counts']['train']-counts['train'],validation=parent[0]['counts']['validation']-counts['validation']))
    return out,fields


def build(dataset,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    donor.build(dataset,folder/'parent')
    parent=donor.load(dataset,folder/'parent');arrays,fields=describe(parent)
    atomic_publish_bytes(folder/'rows.npz',deterministic_npz_bytes(arrays))
    result=c.artifact('POPULATION',parents=dict(parent=parent[0]['content_hash']),
        parent=file_ref(folder/'parent/population.json',root=folder),rows=file_ref(folder/'rows.npz',root=folder),**fields)
    write_immutable_json(folder/'population.json',result)
    return result


def load(dataset,folder):
    folder=Path(folder);parent=donor.load(dataset,folder/'parent');p=load_json(folder/'population.json')
    c.validate(p,'POPULATION',dict(parent=c.parent.POPULATION_HASH))
    if p['content_hash']!=c.POPULATION_HASH:raise ValueError('Frozen 600k/150k population differs')
    if p['parent']!=file_ref(folder/'parent/population.json',root=folder):raise ValueError('Population parent differs')
    arrays,fields=describe(parent)
    if any(p[k]!=v for k,v in fields.items()):raise ValueError('Subset summary replay differs')
    saved=load_npz_arrays(validate_file_ref(p['rows'],root=folder))
    if set(saved)!=set(arrays) or any(saved[k].dtype!=v.dtype or not np.array_equal(saved[k],v) for k,v in arrays.items()):
        raise ValueError('Subset membership differs from identity replay')
    return p,parent,saved


def selections(parent,arrays):
    """Original authorized I/O codes and reduced learning rows for each source."""
    cursors=dict(train=0,validation=0)
    for index,(record,a,roles) in enumerate(zip(parent[1]['files'],parent[2],parent[3],strict=True)):
        codes=np.full(len(roles),255,np.uint8);keep=np.zeros(len(roles),bool)
        for code,role in enumerate(('train','validation')):
            entries=np.flatnonzero(donor.masks(a,roles)[role]);start=cursors[role];cursors[role]+=len(entries)
            ids=arrays[role+'_parent_indices'];lo,hi=np.searchsorted(ids,[start,cursors[role]])
            selected=entries[ids[lo:hi]-start];keep[selected]=True;codes[entries]=code
        if keep.any():yield index,record,a,codes,keep

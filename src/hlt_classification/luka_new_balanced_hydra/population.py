"""Unused-only split extension; parent/test particles are never needed here."""
from pathlib import Path
import hashlib
import numpy as np
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    deterministic_npz_bytes, atomic_publish_bytes, write_immutable_json)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_new_fullstat_hydra import population as original
from hlt_classification.luka_fullstat.data import apportion
from hlt_classification.luka_fullstat_balanced_ladder.sampling import BalancedSampler
from . import contracts as c

ROLES = ('train','validation','final_test')


def extend(records, members, additions=None):
    additions=dict(c.ADDITIONS if additions is None else additions)
    if set(additions)!=set(ROLES) or any(type(x)!=int or x<0 for x in additions.values()):
        raise ValueError('Invalid extension budgets')
    groups={}; result=[a['roles'].copy() for a in members]
    for i,(r,a) in enumerate(zip(records,members,strict=True)):
        eligible=(a['roles']==3)&(a['jet_pt']>=200)&(a['jet_pt']<2000)
        bins=np.searchsorted(c.EDGES,a['jet_pt'],side='right')-1
        for label in range(11):
            for b in range(18):
                es=np.flatnonzero(eligible&(a['labels']==label)&(bins==b))
                if len(es):groups.setdefault(f'{r["category"]}:{label:02d}:{b:02d}',[]).append((i,es))
    capacity={k:sum(len(es) for _,es in v) for k,v in groups.items()}
    if sum(additions.values())>sum(capacity.values()):raise ValueError('Insufficient unused in-range rows')
    quotas={};left=capacity.copy()
    for role in ROLES:
        quotas[role]=apportion(additions[role],left)
        left={k:n-quotas[role][k] for k,n in left.items()}
    quotas['unused']=left
    for key,items in sorted(groups.items()):
        chunks=[]
        for i,es in items:
            for part in np.split(es,np.flatnonzero(np.diff(es//c.BLOCK))+1):
                token=f'{c.PREFIX}_EXTENSION/v1/{c.EXTENSION_SEED}/{records[i]["sha256"]}/{int(part[0]//c.BLOCK)}'
                chunks.append((hashlib.sha256(token.encode()).digest(),i,part))
        chunks.sort(key=lambda x:(x[0],x[1],int(x[2][0])))
        role=0;remaining=quotas[ROLES[0]][key]
        for _,i,es in chunks:
            start=0
            while start<len(es):
                while remaining==0:
                    role+=1
                    remaining=quotas[(*ROLES,'unused')[role]][key]
                n=min(remaining,len(es)-start)
                result[i][es[start:start+n]]=role
                remaining-=n;start+=n
    for old,new in zip(members,result,strict=True):
        changed=new!=old['roles']
        if np.any(changed & ((old['roles']!=3)|(old['jet_pt']<200)|(old['jet_pt']>=2000))):
            raise PermissionError('Attempted reassignment of a protected row')
    return result,quotas


def masks(member, extended_roles):
    inside=(member['jet_pt']>=200)&(member['jet_pt']<2000)
    return dict(train=(extended_roles==0)&inside,validation=(extended_roles==1)&inside,
                outside_validation=(member['roles']==1)&~inside)


def inventory(f,members,roles):
    counts={r:0 for r in (*ROLES,'unused')};outside={r:0 for r in counts}
    arrays={r:{k:[] for k in ('identities','labels','pt')} for r in ('train','validation','outside_validation')}
    particles={r:dict(hlt=0,offline=0) for r in arrays};maxima={r:dict(hlt=0,offline=0,concat=0) for r in arrays}
    for record,a,rs in zip(f['files'],members,roles,strict=True):
        inside=(a['jet_pt']>=200)&(a['jet_pt']<2000)
        for code,role in enumerate(counts):
            counts[role]+=int(((rs==code)&inside).sum());outside[role]+=int(((rs==code)&~inside).sum())
        for role,mask in masks(a,rs).items():
            es=np.flatnonzero(mask)
            ids=np.asarray([np.frombuffer(bytes.fromhex(original.identity(record,int(e))),np.uint8) for e in es],np.uint8).reshape(-1,32)
            for k,v in [('identities',ids),('labels',a['labels'][es].astype(np.int64)),('pt',a['jet_pt'][es].astype(np.float64))]:
                arrays[role][k].append(v)
            nh,no=a['hlt_jet_nparticles'][es],a['jet_nparticles'][es]
            particles[role]['hlt']+=int(nh.sum());particles[role]['offline']+=int(no.sum())
            for k,v in [('hlt',nh),('offline',no),('concat',nh+no)]:
                maxima[role][k]=max(maxima[role][k],int(v.max()) if len(v) else 0)
    return counts,outside,particles,maxima,{f'{r}_{k}':np.concatenate(v) for r,fields in arrays.items() for k,v in fields.items()}


def build(dataset,folder,*,production=True):
    dataset,folder=Path(dataset),Path(folder)
    f,members=original.load(dataset/'foundation',production=production)
    if production and f['content_hash']!=c.PARENT_HASH:raise ValueError('Wrong parent foundation')
    roles,quotas=extend(f['files'],members)
    counts,outside,particles,maxima,arrays=inventory(f,members,roles)
    if production and (counts!=dict(c.COUNTS,unused=689) or outside!=c.OUTSIDE):raise ValueError('Extension population differs')
    sampler=BalancedSampler(arrays['train_labels'],arrays['train_pt'])
    folder.mkdir(parents=True,exist_ok=False);refs=[]
    for i,rs in enumerate(roles):
        path=folder/'roles'/f'{i:05d}.npz';atomic_publish_bytes(path,deterministic_npz_bytes(dict(roles=rs)))
        refs.append(file_ref(path,root=folder))
    atomic_publish_bytes(folder/'rows.npz',deterministic_npz_bytes(arrays))
    report=c.artifact('POPULATION',parents=dict(foundation=f['content_hash']),policy=c.population_policy(),
        counts=counts,outside_counts=outside,role_particles=particles,maximum_support=maxima,
        class_bin_counts=sampler.counts.tolist(),quotas=quotas,memberships=refs,
        rows=file_ref(folder/'rows.npz',root=folder),production=production)
    if production and report['content_hash']!=c.POPULATION_HASH:raise ValueError('New population differs from the authenticated local audit')
    write_immutable_json(folder/'population.json',report)
    return report


def load(dataset,folder,*,production=True):
    dataset,folder=Path(dataset),Path(folder)
    f,members=original.load(dataset/'foundation',production=production)
    p=load_json(folder/'population.json');c.validate(p,'POPULATION',dict(foundation=f['content_hash']))
    if p['policy']!=c.population_policy() or (production and (p['production'] is not True or f['content_hash']!=c.PARENT_HASH or p['content_hash']!=c.POPULATION_HASH)):
        raise ValueError('Extension policy differs')
    expected,quotas=extend(f['files'],members)
    if len(p['memberships'])!=len(members) or p['quotas']!=quotas:raise ValueError('Extension quota/coverage differs')
    for i,(ref,rs) in enumerate(zip(p['memberships'],expected,strict=True)):
        a=load_npz_arrays(validate_file_ref(ref,root=folder))
        if ref['path']!=f'roles/{i:05d}.npz' or set(a)!={'roles'} or a['roles'].dtype!=np.uint8 or not np.array_equal(a['roles'],rs):
            raise ValueError('Unused-only extension replay differs')
    counts,outside,parts,maxima,arrays=inventory(f,members,expected)
    if p['counts']!=counts or p['outside_counts']!=outside or p['role_particles']!=parts or p['maximum_support']!=maxima:
        raise ValueError('Population summary differs')
    if production and (counts!=dict(c.COUNTS,unused=689) or outside!=c.OUTSIDE):raise ValueError('Population budgets differ')
    saved=load_npz_arrays(validate_file_ref(p['rows'],root=folder))
    if set(saved)!=set(arrays) or any(saved[k].dtype!=v.dtype or not np.array_equal(saved[k],v) for k,v in arrays.items()):
        raise ValueError('Canonical TRAIN/validation row join differs')
    sampler=BalancedSampler(arrays['train_labels'],arrays['train_pt'],expected_counts=p['class_bin_counts'])
    return p,f,members,expected,arrays,sampler

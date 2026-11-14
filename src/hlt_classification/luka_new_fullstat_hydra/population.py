"""Scalar-only new-source selection, proportional split and portable memberships."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
import hashlib
import re
import numpy as np
import uproot
from hlt_classification.data.cache_contracts import (atomic_publish_bytes, canonical_sha256,
    deterministic_npz_bytes, load_json, load_npz_arrays, sha256_file, write_immutable_json)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.schema import CLASS_NAMES, RAW_TO_CLASS, validate_tree_schema
from hlt_classification.jetclass2_delphes.inventory import latest_tree
from hlt_classification.luka_fullstat.data import apportion
from . import contracts as c


def inventory(root, *, production=True):
    root=Path(root); manifest=root/'source.sha256'
    digest=sha256_file(manifest)
    if production and digest!=c.MANIFEST_SHA: raise ValueError('Wrong CERN snapshot manifest')
    records=[]
    for line in manifest.read_text(encoding='utf-8-sig').splitlines():
        m=re.fullmatch(r'([0-9a-f]{64}) [ *](.+)',line)
        if m is None: raise ValueError('Malformed source manifest')
        sha,remote=m.groups()
        found=[(prefix,side) for prefix,side in c.SOURCES.items() if remote.startswith(prefix)]
        if len(found)!=1: raise ValueError('Unregistered source')
        prefix,side=found[0]; rel='raw/'+side+'/'+remote[len(prefix):]
        path=c.safe(root,rel)
        parts=rel.split('/')
        if side=='qcd':
            if len(parts)!=7 or parts[2]!='jetclass2sb' or not re.fullmatch(r'QCD_PT-[A-Za-z0-9]+',parts[3]):
                raise ValueError('Unregistered QCD source path')
            category=parts[3]
        else:
            if parts[2:5]!=['jetclass2sb','train_higgs2p','merged'] or len(parts)!=6:
                raise ValueError('Unregistered signal source path')
            category='signal'
        records.append(dict(path=rel,remote=remote,sha256=sha,bytes=path.stat().st_size,category=category))
    records.sort(key=lambda r:r['path'])
    if (len({r['path'].casefold() for r in records})!=len(records)
            or len({r['sha256'] for r in records})!=len(records)
            or sorted(p.relative_to(root).as_posix() for p in (root/'raw').rglob('*.root'))!=[r['path'] for r in records]):
        raise ValueError('Missing/extra/duplicate source files')
    if production and Counter('nonqcd' if r['category']=='signal' else 'qcd' for r in records)!=dict(qcd=6990,nonqcd=6):
        raise ValueError('Wrong source counts')
    return digest,records


def select(a, category):
    if set(a)!=set(c.SCALARS): raise ValueError('Scalar fields differ')
    raw=a['jet_label']; n=len(raw)
    if raw.dtype.kind not in 'iu' or any(v.shape!=(n,) for v in a.values()): raise ValueError('Scalar types/shapes differ')
    if a['hlt_matched'].dtype.kind!='b': raise ValueError('Invalid matched flag')
    for k in ('jet_nparticles','hlt_jet_nparticles'):
        if a[k].dtype.kind not in 'iu' or np.any(a[k]<0): raise ValueError('Invalid particle count')
    if not np.isfinite(a['jet_pt']).all() or np.any(a['jet_pt']<=0): raise ValueError('Invalid pT')
    unknown=set(map(int,np.unique(raw)))-set(RAW_TO_CLASS)
    if unknown-({206} if category=='signal' else set()): raise ValueError('Unregistered raw label')
    labels=np.full(n,-1,np.int8)
    for code in np.unique(raw):
        if int(code) in RAW_TO_CLASS: labels[raw==code]=RAW_TO_CLASS[int(code)]
    if category!='signal' and np.any(labels!=0): raise ValueError('Non-QCD in dedicated QCD source')
    paired=a['hlt_matched'] & (a['jet_nparticles']>0) & (a['hlt_jet_nparticles']>0)
    keep=paired & ((labels>0) if category=='signal' else labels==0)
    return keep,labels,dict(raw=n,paired=int(paired.sum()),selected=int(keep.sum()),
        quarantined206=int((raw==206).sum()),quarantined206_paired=int(((raw==206)&paired).sum()),
        signal_source_qcd=int((labels==0).sum()) if category=='signal' else 0)


def partition(records, arrays, counts):
    if set(counts)!=set(c.ROLES) or any(type(v)!=int or v<0 for v in counts.values()): raise ValueError('Role budget differs')
    groups={}; labels=[]; roles=[]
    for i,(r,a) in enumerate(zip(records,arrays)):
        keep,labs,_=select(a,r['category']);labels.append(labs);roles.append(np.full(len(labs),255,np.uint8))
        for label in range(11):
            entries=np.flatnonzero(keep & (labs==label))
            if not len(entries):continue
            key=f'signal:{label:02d}' if r['category']=='signal' else r['category']
            groups.setdefault(key,[]).append((i,entries))
    capacity={k:sum(len(es) for _,es in v) for k,v in groups.items()}
    if sum(capacity.values())!=sum(counts.values()):raise ValueError('Budget must exhaust eligible pool, including unused')
    remaining=capacity.copy();quotas={}
    for role in c.ROLES:
        quotas[role]=apportion(counts[role],remaining)
        remaining={k:v-quotas[role][k] for k,v in remaining.items()}
    for key,items in sorted(groups.items()):
        chunks=[]
        for i,entries in items:
            for es in np.split(entries,np.flatnonzero(np.diff(entries//c.BLOCK))+1):
                digest=hashlib.sha256(f'NEW_FULLSTAT_SPLIT/{c.SEED}/{records[i]["sha256"]}/{int(es[0]//c.BLOCK)}'.encode()).digest()
                chunks.append((digest,i,es))
        chunks.sort(key=lambda x:(x[0],x[1],int(x[2][0])))
        role_index=0;left=quotas[c.ROLES[0]][key]
        for _,i,entries in chunks:
            start=0
            while start<len(entries):
                while left==0:
                    role_index+=1;left=quotas[c.ROLES[role_index]][key]
                take=min(left,len(entries)-start)
                roles[i][entries[start:start+take]]=role_index
                start+=take;left-=take
    return roles,labels,quotas


def identity(record, entry):
    return canonical_sha256(dict(domain='LUKA_NEW_FULLSTAT_ROW/v1',path=record['path'],
        sha256=record['sha256'],tree=record['tree_key'],entry=int(entry)))


def scan(args):
    root,r=args;path=c.safe(root,r['path']);before=path.stat()
    if sha256_file(path)!=r['sha256']:raise ValueError('Raw source checksum differs: '+r['path'])
    with uproot.open(path,object_cache=None,array_cache=None) as f:
        key,tree=latest_tree(f);schema=validate_tree_schema(tree)
        a=tree.arrays(list(c.SCALARS),library='np')
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source changed while scanning')
    _,_,flow=select(a,r['category'])
    return dict(r,tree_key=key,entries=len(a['jet_label']),cutflow=flow),a,schema


def build(root, output, *, acknowledge_row_split=False, production=True, counts=None):
    if not acknowledge_row_split:raise PermissionError('Acknowledge row-disjoint, event-independence-unverified split')
    root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError('Preserve existing foundation')
    manifest,records=inventory(root,production=production)
    scanned=[];arrays=[];schema=None
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(r,a,s) in enumerate(pool.map(scan,((root,r) for r in records)),1):
            if schema is not None and s!=schema:raise ValueError('Source schemas differ')
            schema=s;scanned.append(r);arrays.append(a)
            if i%500==0 or i==len(records):print(f'Scalar foundation {i}/{len(records)}',flush=True)
    counts=dict(c.COUNTS if counts is None else counts)
    if production and counts!=c.COUNTS:raise ValueError('Production split counts differ')
    roles,labels,quotas=partition(scanned,arrays,counts)
    output.mkdir(parents=True)
    for i,(r,a,rs,ls) in enumerate(zip(scanned,arrays,roles,labels)):
        target=output/'memberships'/f'{i:05d}.npz'
        atomic_publish_bytes(target,deterministic_npz_bytes(dict(**a,roles=rs,labels=ls)))
        r['membership']=file_ref(target,root=output)
        r['role_counts']={role:int((rs==j).sum()) for j,role in enumerate(c.ROLES)}
        r['role_particles']={role:{side:int(a[k][rs==j].sum()) for side,k in
            [('offline','jet_nparticles'),('hlt','hlt_jet_nparticles')]} for j,role in enumerate(c.ROLES)}
    result=c.artifact('FOUNDATION',source_manifest_sha256=manifest,files=scanned,counts=counts,quotas=quotas,
        schema=schema,schema_sha256=canonical_sha256(schema),policy=c.policy(),class_names=list(CLASS_NAMES),
        raw_rows=sum(r['entries'] for r in scanned),production=production,particle_arrays_decoded=False)
    write_immutable_json(output/'foundation.json',result)
    load(output,production=production)
    return result


def load(root, *, production=True, replay=True):
    root=Path(root);f=load_json(root/'foundation.json');c.validate(f,'FOUNDATION')
    if (f['policy']!=c.policy() or f['class_names']!=list(CLASS_NAMES) or f['particle_arrays_decoded'] is not False
            or f['schema_sha256']!=canonical_sha256(f['schema'])):raise ValueError('Foundation semantics differ')
    if production and (f['production'] is not True or f['counts']!=c.COUNTS or f['source_manifest_sha256']!=c.MANIFEST_SHA):
        raise ValueError('Not the registered production foundation')
    records=f['files']
    if [r['path'] for r in records]!=sorted({r['path'] for r in records}) or len({r['sha256'] for r in records})!=len(records):
        raise ValueError('Duplicate/source order differs')
    payloads=[]
    for i,r in enumerate(records):
        if r['membership']['path']!=f'memberships/{i:05d}.npz':raise ValueError('Membership path differs')
        a=load_npz_arrays(validate_file_ref(r['membership'],root=root))
        if set(a)!=set(c.SCALARS)|{'roles','labels'}:raise ValueError('Membership fields differ')
        keep,ls,flow=select({k:a[k] for k in c.SCALARS},r['category'])
        if (not np.array_equal(ls,a['labels']) or not np.array_equal(keep,a['roles']!=255)
                or a['roles'].dtype!=np.uint8 or not np.isin(a['roles'],[0,1,2,3,255]).all()
                or flow!=r['cutflow'] or len(ls)!=r['entries']):raise ValueError('Membership scalar binding differs')
        for j,role in enumerate(c.ROLES):
            if r['role_counts'][role]!=int((a['roles']==j).sum()):raise ValueError('Role counts differ')
            for side,k in [('offline','jet_nparticles'),('hlt','hlt_jet_nparticles')]:
                if r['role_particles'][role][side]!=int(a[k][a['roles']==j].sum()):raise ValueError('Particle budget differs')
        payloads.append(a)
    if any(sum(r['role_counts'][role] for r in records)!=f['counts'][role] for role in c.ROLES):raise ValueError('Total budget differs')
    if replay:
        rs,_,q=partition(records,[{k:a[k] for k in c.SCALARS} for a in payloads],f['counts'])
        if q!=f['quotas'] or any(not np.array_equal(r,a['roles']) for r,a in zip(rs,payloads)):raise ValueError('Split replay differs')
    return f,payloads

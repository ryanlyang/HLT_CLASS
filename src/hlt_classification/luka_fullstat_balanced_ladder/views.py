"""Eligible-only TRAIN views, untouched validation, full-support physical D050/CONCAT."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from types import SimpleNamespace
import time
import numpy as np
from hlt_classification.jetclass2_delphes.cache import RamBlock, RamCache, _limit_worker_threads
from hlt_classification.luka_fullstat import cache, data
from hlt_classification.luka_fullstat.contracts import BLOCK_ROWS, identity
from hlt_classification.luka_fullstat_fine_tail.views import assignments, digest
from hlt_classification.luka_fullsim.particles import physical
from hlt_classification.luka_fullsim.preflight_cache import _block
from hlt_classification.literature_context.transform import build_inputs
from hlt_classification.cms_proxy_ladder.views import build_view
from hlt_classification.luka_u100_multiview.views import union
from .contracts import COORDINATES, ELIGIBLE, COUNTS, artifact


def selected_entries(member,start,role):
    code={'train':0,'validation':1}[role]
    entries=np.flatnonzero(member['roles'][start:start+BLOCK_ROWS]==code)+start
    if role=='train':
        p=member['jet_pt'][entries];entries=entries[(p>=200)&(p<2000)]
    return entries


def take(block,indices):
    indices=np.asarray(indices,dtype=np.int64)
    if indices.ndim!=1 or np.any(indices<0) or np.any(indices>=len(block.labels)):
        raise ValueError('Invalid block subset')
    lengths=np.diff(block.offsets)[indices]
    parts=[np.arange(block.offsets[i],block.offsets[i+1]) for i in indices]
    positions=np.concatenate(parts) if parts else np.empty(0,np.int64)
    return RamBlock(block.file_index,np.r_[0,np.cumsum(lengths)],block.features[positions],
                    block.vectors[positions],block.identities[indices],block.labels[indices])


def ordinary(metadata,root,coordinate,*,memory_bytes):
    manifest,foundation,memberships=metadata
    blocks={'train':[],'validation':[]};used=0
    for chunk in manifest['chunks']:
        i=chunk['file_index'];role=chunk['role'];member=memberships[i]
        value=cache.read_block(root,chunk,coordinate,foundation['files'][i],member)
        if role=='train':
            start=chunk['block_index']*BLOCK_ROWS
            entries=np.flatnonzero(member['roles'][start:start+BLOCK_ROWS]==0)+start
            p=member['jet_pt'][entries];value=take(value,np.flatnonzero((p>=200)&(p<2000)))
        if len(value.labels):blocks[role].append(value);used+=value.nbytes+40*len(value.labels)
        if used+4*2**30>memory_bytes:raise MemoryError('Eligible input resident budget exceeded')
    return {r:RamCache(bb,role=r,coordinate_name=coordinate,foundation_sha256=manifest['content_hash'])
            for r,bb in blocks.items()}


_WORK=None


def _init(path,record,member,config,root,chunks,coordinate):
    global _WORK
    _limit_worker_threads()
    _WORK=(path,record,member,config,root,chunks,coordinate)


def _work(index):
    path,record,member,config,root,chunks,coordinate=_WORK
    selected=[c for c in chunks if c['block_index']==index]
    maps={c['role']:assignments(c,record,member,root) for c in selected} if coordinate=='D050' else {}
    lookups={r:{int(e):j for j,e in enumerate(a['entries'])} for r,a in maps.items()}
    rows={r:[] for r in ('train','validation')};endpoints={r:[] for r in rows}
    start=index*BLOCK_ROWS
    for role,entry,rh,ro in cache.read_selected_chunk(path,record,member,start,min(start+BLOCK_ROWS,record['entries'])):
        if role=='train' and not 200<=member['jet_pt'][entry]<2000:continue
        h,o=physical(rh,config,'hlt'),physical(ro,config,'offline')
        row=SimpleNamespace(identity=identity(record,entry),label=int(member['labels'][entry]))
        if coordinate=='CONCAT':
            particles=union(h,o)
            if len(particles)!=len(h)+len(o):raise ValueError('Union lost particles')
        else:
            a=maps[role];j=lookups[role][entry];lo,hi=a['offsets'][j:j+2]
            particles=build_view(identity=row.identity,proxy=h,offline=o,coordinate='D050',mapping=a['mapping'][lo:hi])
            endpoints[role].append((row,build_inputs(h,capacity=max(16,len(h)))))
        rows[role].append((row,build_inputs(particles,capacity=max(16,len(particles)))))
    products=[]
    for chunk in selected:
        role=chunk['role'];entries=selected_entries(member,start,role)
        if not len(entries):continue
        value=_block(chunk['file_index'],rows[role])
        expected=np.asarray([np.frombuffer(bytes.fromhex(identity(record,int(e))),np.uint8) for e in entries])
        lengths=member['hlt_jet_nparticles'][entries]
        if coordinate=='CONCAT':lengths=lengths+member['jet_nparticles'][entries]
        if (not np.array_equal(value.identities,expected) or not np.array_equal(value.labels,member['labels'][entries])
                or not np.array_equal(np.diff(value.offsets),lengths)):
            raise ValueError('New view identity/label/cardinality differs')
        if coordinate=='D050':
            original=cache.read_block(root,chunk,'D000',record,member)
            allentries=np.flatnonzero(member['roles'][start:start+BLOCK_ROWS]==(0 if role=='train' else 1))+start
            reference=take(original,np.searchsorted(allentries,entries))
            if digest(_block(chunk['file_index'],endpoints[role]))!=digest(reference):
                raise ValueError('Physical HLT endpoint replay differs')
        products.append((role,value,dict(file_index=chunk['file_index'],block_index=index,role=role,
            rows=len(value.labels),arrays_sha256=digest(value),endpoint_replay=coordinate=='D050')))
    return products


def constructed(metadata,evidence,coordinate,*,workers,memory_bytes):
    manifest,foundation,memberships=metadata
    if coordinate not in ('D050','CONCAT') or workers not in (1,4):raise ValueError('Unregistered view construction')
    blocks={'train':[],'validation':[]};records=[];used=0
    for i,(record,member) in enumerate(zip(foundation['files'],memberships,strict=True)):
        path=data.authenticate_source(evidence['raw_container'],record)
        chunks=[c for c in manifest['chunks'] if c['file_index']==i]
        indices=sorted({c['block_index'] for c in chunks})
        args=(str(path),record,member,manifest['conventions'],evidence['cache_root'],chunks,coordinate)
        def accept(products):
            nonlocal used
            for role,value,info in products:
                blocks[role].append(value);records.append(info);used+=value.nbytes+40*len(value.labels)
            if used+8*2**30>memory_bytes:raise MemoryError('Constructed view host envelope exceeded')
        if workers==1:
            _init(*args)
            for index in indices:accept(_work(index))
        else:
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                     initializer=_init,initargs=args) as pool:
                pending=deque();iterator=iter(indices)
                for _ in range(workers):
                    index=next(iterator,None)
                    if index is not None:pending.append(pool.submit(_work,index))
                while pending:
                    accept(pending.popleft().result())
                    index=next(iterator,None)
                    if index is not None:pending.append(pool.submit(_work,index))
        data.authenticate_source(evidence['raw_container'],record)
        print(f'BALANCED view={coordinate} file={i+1}/{len(memberships)} resident_GiB={used/2**30:.2f}',flush=True)
    return {r:RamCache(bb,role=r,coordinate_name=coordinate,foundation_sha256=manifest['content_hash'])
            for r,bb in blocks.items()},records


def load(spec,stage,metadata,population):
    begun=time.monotonic();coordinate=COORDINATES[stage];evidence=spec['inputs']['evidence']
    budget=int(spec['memory_mb']*1024**2*.70)
    if coordinate in ('CONCAT','D050'):
        values,records=constructed(metadata,evidence,coordinate,workers=spec['preparation_workers'],memory_bytes=budget)
    else:
        values=ordinary(metadata,evidence['cache_root'],coordinate,memory_bytes=budget);records=[]
    for role,value in values.items():
        ix=population['eligible_indices'] if role=='train' else slice(None)
        if (len(value)!=(ELIGIBLE if role=='train' else COUNTS[role])
                or not np.array_equal(value.identities,population[role+'_identities'][ix])
                or not np.array_equal(value.labels,population[role+'_labels'][ix])):
            raise ValueError('Stage input population differs')
    proof=artifact('VIEW_BUILD',parents=dict(spec=spec['content_hash'],population=spec['population']['content_hash']),
        stage=stage,coordinate=coordinate,rows={r:len(v) for r,v in values.items()},
        chunks=records,seconds=time.monotonic()-begun,full_particle_support=True,
        summaries={r:dict(resident_bytes=v.nbytes,maximum=int(lengths(v).max())) for r,v in values.items()})
    return values['train'],values['validation'],proof


def lengths(value):
    return np.concatenate([np.diff(b.offsets) for b in value.blocks])

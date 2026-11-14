"""Bounded raw reads into one resident view; no persistent particle-cache copies."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext, contextmanager
from pathlib import Path
from types import SimpleNamespace
import hashlib
import multiprocessing
import time
import numpy as np
import uproot
from hlt_classification.data.cache_contracts import sha256_file
from hlt_classification.jetclass2_delphes.cache import RamCache, _limit_worker_threads
from hlt_classification.luka_fullsim.particles import FIELDS, _matrix, _ranges, physical
from hlt_classification.luka_fullsim.preflight_cache import _block
from hlt_classification.luka_fullstat.cache import conventions
from hlt_classification.cms_proxy_ladder.views import match_particles, build_view
from hlt_classification.literature_context.transform import build_inputs
from hlt_classification.luka_u100_multiview.views import union
from hlt_classification.luka_new_fullstat_hydra.population import identity
from hlt_classification.luka_fullstat_balanced_ladder.views import lengths
from . import contracts as c, population

_OPEN=None


def close_reader():
    global _OPEN
    if _OPEN is not None:
        _OPEN[1].close();_OPEN=None


def initialize_worker():
    import atexit
    _limit_worker_threads();atexit.register(close_reader)


@contextmanager
def reader(path,tree_key,*,retain=False):
    """One metadata handle per worker; no decoded array or basket cache."""
    global _OPEN
    if not retain:
        with uproot.open(path,object_cache=None,array_cache=None) as handle:yield handle[tree_key]
        return
    key=(str(path),tree_key)
    if _OPEN is None or _OPEN[0]!=key:
        close_reader()
        handle=uproot.open(path,object_cache=None,array_cache=None)
        try:tree=handle[tree_key]
        except BaseException:
            handle.close();raise
        _OPEN=(key,handle,tree)
    yield _OPEN[2]


def selected_spans(codes):
    """Codes 0/1/2 here are authorized TRAIN/in-range-val/outside-val only.

    Unlike parent role codes, 2 is NOT test. Codes are produced solely from
    population.masks; every other raw row is 255 and splits the read interval.
    ROOT may internally decompress baskets beyond these requested slices.
    """
    return _ranges(np.flatnonzero(codes!=255).tolist(),c.BLOCK)


def work(arg,*,retain_reader=False):
    dataset,record,index,start,member,codes,coordinate,parent=arg
    path=Path(dataset)/record['path'];config=conventions(parent)
    sides=('hlt',) if coordinate=='D000' else ('offline',) if coordinate=='OFFLINE' else ('hlt','offline')
    branches=[]
    for side in sides:
        prefix='hlt_' if side=='hlt' else ''
        branches += [prefix+'jet_nparticles']+[prefix+'part_'+k for k in FIELDS]
    rows={r:[] for r in ('train','validation','outside_validation')};digests={r:hashlib.sha256() for r in rows}
    with reader(path,record['tree_key'],retain=retain_reader) as tree:
        if tree.num_entries!=record['entries']:raise ValueError('Source row count changed')
        for lo,hi in selected_spans(codes):
            if np.any(codes[lo:hi]==255):raise PermissionError('Forbidden particle span')
            arrays=tree.arrays(branches,entry_start=start+lo,entry_stop=start+hi,library='ak',how=dict)
            for offset,j in enumerate(range(lo,hi)):
                endpoints={}
                for side in sides:
                    prefix='hlt_' if side=='hlt' else ''
                    n=int(arrays[prefix+'jet_nparticles'][offset])
                    if n!=int(member[prefix+'jet_nparticles'][j]):raise ValueError('Particle/scalar count differs')
                    endpoints[side]=physical(_matrix(arrays,prefix+'part_',offset,n),config,side)
                role=tuple(rows)[codes[j]];rid=identity(record,start+j)
                if coordinate=='OFFLINE':particles=endpoints['offline']
                elif coordinate=='D000':particles=endpoints['hlt']
                elif coordinate=='CONCAT':particles=union(endpoints['hlt'],endpoints['offline'])
                else:
                    mapping=match_particles(endpoints['hlt'],endpoints['offline'])
                    digests[role].update(bytes.fromhex(rid));digests[role].update(np.asarray(mapping,dtype='<i4').tobytes())
                    particles=build_view(identity=rid,proxy=endpoints['hlt'],offline=endpoints['offline'],coordinate=coordinate,mapping=mapping)
                expected=(member['jet_nparticles'][j] if coordinate=='OFFLINE' else
                          member['jet_nparticles'][j]+member['hlt_jet_nparticles'][j] if coordinate=='CONCAT' else member['hlt_jet_nparticles'][j])
                if len(particles)!=expected:raise ValueError('Particle support changed')
                rows[role].append((SimpleNamespace(identity=rid,label=int(member['labels'][j])),
                                   build_inputs(particles,capacity=max(16,len(particles)))))
    blocks={r:_block(index,v) for r,v in rows.items() if v}
    return blocks,{r:digests[r].hexdigest() for r in blocks} if coordinate in ('U100','D050') else {}


def process_work(arg):
    return work(arg,retain_reader=True)


def arguments(dataset,f,members,roles,coordinate,parent):
    for i,(record,a,rs) in enumerate(zip(f['files'],members,roles,strict=True)):
        codes=np.full(len(rs),255,np.uint8)
        for code,mask in enumerate(population.masks(a,rs).values()):codes[mask]=code
        if not np.any(codes!=255):continue
        path=Path(dataset)/record['path']
        if sha256_file(path)!=record['sha256']:raise ValueError('Raw source hash differs: '+record['path'])
        for start in range(0,len(rs),c.BLOCK):
            stop=min(start+c.BLOCK,len(rs))
            if not np.any(codes[start:stop]!=255):continue
            member={k:a[k][start:stop] for k in ('jet_nparticles','hlt_jet_nparticles','labels')}
            yield (str(dataset),record,i,start,member,codes[start:stop],coordinate,parent)


def load(spec,stage,loaded,*,workers=None):
    begun=time.monotonic();p,f,members,roles,arrays,_=loaded
    coordinate=c.COORDINATES[stage];workers=spec['preparation_workers'] if workers is None else workers
    if workers not in (1,4):raise ValueError('Use one or four bounded workers')
    budget=int(spec['memory_mb']*1024**2*.65)
    blocks={r:[] for r in ('train','validation','outside_validation')};mapping=[];used=0;count=0
    def accept(arg,result):
        nonlocal used,count
        built,hashes=result
        for r,b in built.items():blocks[r].append(b);used+=b.nbytes+len(b.labels)*40
        if hashes:mapping.append(dict(file_index=arg[2],start=arg[3],hashes=hashes))
        count+=1
        if used+2*2**30>budget:raise MemoryError('Resident view exceeds input allocation envelope')
        if count%100==0:print(f'HYDRA_BALANCED view={coordinate} chunks={count} resident_GiB={used/2**30:.2f}',flush=True)
    jobs=arguments(spec['dataset'],f,members,roles,coordinate,p['content_hash'])
    context=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker) if workers>1 else nullcontext()
    with context as pool:
        if pool is None:
            try:
                for arg in jobs:accept(arg,work(arg,retain_reader=True))
            finally:close_reader()
        else:
            pending=deque()
            for _ in range(workers):
                arg=next(jobs,None)
                if arg is not None:pending.append((arg,pool.submit(process_work,arg)))
            while pending:
                arg,future=pending.popleft();accept(arg,future.result())
                nxt=next(jobs,None)
                if nxt is not None:pending.append((nxt,pool.submit(process_work,nxt)))
    # The file hash is checked again after its final future has completed.
    touched={b.file_index for bb in blocks.values() for b in bb}
    for i in sorted(touched):
        r=f['files'][i]
        if sha256_file(Path(spec['dataset'])/r['path'])!=r['sha256']:raise ValueError('Raw source changed during view build')
    caches={r:RamCache(bb,role='train' if r=='train' else 'validation',coordinate_name=coordinate,
                       foundation_sha256=p['content_hash']) for r,bb in blocks.items()}
    for role,cache in caches.items():
        if not np.array_equal(cache.identities,arrays[role+'_identities']) or not np.array_equal(cache.labels,arrays[role+'_labels']):
            raise ValueError('Resident input canonical row order differs')
    proof=c.artifact('VIEW_BUILD',parents=dict(spec=spec['content_hash'],population=p['content_hash']),
        stage=stage,coordinate=coordinate,rows={r:len(v) for r,v in caches.items()},
        mapping=mapping,seconds=time.monotonic()-begun,full_particle_support=True,
        summaries={r:dict(resident_bytes=v.nbytes,maximum=int(lengths(v).max())) for r,v in caches.items()})
    return caches,proof

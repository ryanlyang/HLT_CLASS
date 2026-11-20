"""Bounded ROOT decoding into allocation-local shared read-only mmap arrays."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import gc
import multiprocessing
import os
import shutil
import time
import numpy as np
from hlt_classification.data.cache_contracts import load_json,write_immutable_json,sha256_file
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.jetclass2_delphes.cache import RamBlock,RamCache
from hlt_classification.luka_fullsim.preflight_cache import _block
from hlt_classification.luka_fullsim.particles import FIELDS,_matrix,_ranges,physical
from hlt_classification.luka_fullstat.cache import conventions
from hlt_classification.literature_context.transform import build_inputs
from hlt_classification.cms_proxy_ladder.views import match_particles,build_view
from hlt_classification.luka_new_fullstat_hydra.population import identity
from hlt_classification.luka_sixclass_balanced_oscar.views import reader,initialize_worker,close_reader
from hlt_classification.luka_next_supervision_hydra.probabilistic_targets import measured
from hlt_classification.luka_next_supervision_hydra.probabilistic_contracts import CONTINUOUS
from . import contracts as c, population, diagnostics as log


def work(arg):
    dataset,record,index,lo,hi,role,keep,coordinate,reco,parent,counts=arg
    # Caller creates a span wholly inside one original authorized role. Keep
    # discards reduced-out rows before construction/statistics, not before I/O.
    sides=('offline',) if coordinate=='OFFLINE' else ('hlt','offline') if coordinate=='U100' or reco else ('hlt',)
    branches=[]
    for side in sides:
        prefix='hlt_' if side=='hlt' else ''
        branches += [prefix+'jet_nparticles']+[prefix+'part_'+f for f in FIELDS]
    rows=[];targets=[];config=conventions(parent)
    with reader(Path(dataset)/record['path'],record['tree_key'],retain=True) as tree:
        if tree.num_entries!=record['entries']:raise ValueError('ROOT row count differs')
        arrays=tree.arrays(branches,entry_start=lo,entry_stop=hi,library='ak',how=dict)
        for offset in np.flatnonzero(keep):
            endpoints={}
            for side in sides:
                prefix='hlt_' if side=='hlt' else '';n=int(arrays[prefix+'jet_nparticles'][offset])
                if n!=int(counts[side][offset]):raise ValueError('Particle/scalar count differs')
                endpoints[side]=physical(_matrix(arrays,prefix+'part_',offset,n),config,side)
            entry=lo+int(offset);rid=identity(record,entry)
            if coordinate=='OFFLINE':particles=endpoints['offline']
            elif coordinate=='U100':
                mapping=match_particles(endpoints['hlt'],endpoints['offline'])
                particles=build_view(identity=rid,proxy=endpoints['hlt'],offline=endpoints['offline'],coordinate='U100',mapping=mapping)
            else:particles=endpoints['hlt']
            # Labels are scalar metadata passed separately; never inferred from particles.
            rows.append((rid,build_inputs(particles,capacity=max(16,len(particles)))))
            if reco:
                h=endpoints['hlt'].p4.sum(0);pt=max(float(np.hypot(h[0],h[1])),1e-12)
                frame=np.array([np.arcsinh(h[2]/pt),np.arctan2(h[1],h[0]),pt],np.float64)
                p=endpoints['offline'];raw=np.column_stack((p.p4,p.charge,p.category,p.tracking,p.valid)).astype(np.float64)
                values,valid=measured(raw,frame)
                aggregate=np.r_[p.p4.sum(0),p.p4[p.charge!=0,3].sum()]
                targets.append((values,valid,frame,aggregate))
    return rows,targets


def arguments(spec,parent,arrays,coordinate,targets):
    last_report=0.
    for index,record,a,codes,keep in population.selections(parent,arrays):
        path=Path(spec['dataset'])/record['path']
        now=time.monotonic()
        if now-last_report>=60:
            log.mark(f'input source authentication coordinate={coordinate} file_index={index}')
            last_report=now
        if sha256_file(path)!=record['sha256']:raise ValueError('Raw source checksum differs')
        for role in (0,1):
            entries=np.flatnonzero(codes==role)
            if not len(entries):continue
            for lo,hi in _ranges(entries.tolist(),1024):
                if np.any(codes[lo:hi]!=role):raise PermissionError('Unauthorized ROOT span')
                subset=keep[lo:hi]
                if subset.any():yield (spec['dataset'],record,index,lo,hi,role,subset,coordinate,targets and role==0,parent[0]['content_hash'],
                    dict(hlt=a['hlt_jet_nparticles'][lo:hi],offline=a['jet_nparticles'][lo:hi]))


def shape_bytes(pop,coordinate,targets):
    side='offline' if coordinate=='OFFLINE' else 'hlt'
    value=sum(pop['particles'][r][side]*84+(pop['counts'][r]+1)*8+pop['counts'][r]*40 for r in c.COUNTS)
    if targets:value+=pop['particles']['train']['offline']*(22*12+1)+pop['counts']['train']*80
    return int(value*1.05)+2**30


def prepare_node(spec,node,coordinate,*,targets=False,workers=4,loaded=None):
    if coordinate not in ('D000','OFFLINE','U100') or (targets and coordinate!='D000'):
        raise PermissionError('Unregistered view/target capability')
    p,parent,arrays=population.load(spec['dataset'],Path(spec['root'])/'population') if loaded is None else loaded
    node=Path(node);node.mkdir(parents=True,exist_ok=False);begun=time.monotonic()
    if shutil.disk_usage(node).free<shape_bytes(p,coordinate,targets):raise OSError('Insufficient allocation-local shared scratch')
    side='offline' if coordinate=='OFFLINE' else 'hlt';values={};files=[];cursors={r:[0,0] for r in c.COUNTS}
    def create(name,shape,dtype):
        value=np.lib.format.open_memmap(node/(name+'.npy'),mode='w+',dtype=dtype,shape=shape)
        values[name]=value;return value
    for role in c.COUNTS:
        n=p['counts'][role];tokens=p['particles'][role][side]
        for name,shape,dtype in [('features',(tokens,17),np.float32),('vectors',(tokens,4),np.float32),
            ('offsets',(n+1,),np.int64),('identities',(n,32),np.uint8),('labels',(n,),np.int64)]:
            create(role+'_'+name,shape,dtype)
        values[role+'_offsets'][0]=0
    if targets:
        n=p['counts']['train'];tokens=p['particles']['train']['offline']
        for name,shape,dtype in [('target_raw',(tokens,22),np.float64),('target_bits',(tokens,),np.uint8),
            ('target_offsets',(n+1,),np.int64),('target_frame',(n,3),np.float64),('target_aggregate',(n,5),np.float64)]:create(name,shape,dtype)
        values['target_offsets'][0]=0
    log.mark(f'input arrays allocated coordinate={coordinate} targets={targets}')
    total=np.zeros(22);squares=np.zeros(22);support=np.zeros(22,np.int64);target_cursor=0
    chunks=0;last_report=time.monotonic()
    def accept(arg,result):
        nonlocal target_cursor,chunks,last_report
        role=('train','validation')[arg[5]];row,token=cursors[role];rows,tg=result
        n=len(rows);wanted=arrays[role+'_identities'][row:row+n]
        if not np.array_equal(np.asarray([np.frombuffer(bytes.fromhex(rid),np.uint8) for rid,_ in rows]),wanted):
            raise ValueError('ROOT/student canonical identity ordering differs')
        items=[(SimpleNamespace(identity=rid,label=int(arrays[role+'_labels'][row+i])),v) for i,(rid,v) in enumerate(rows)]
        block=_block(arg[2],items)
        for name in ('features','vectors'):values[role+'_'+name][token:token+len(block.features)]=getattr(block,name)
        for name in ('identities','labels'):values[role+'_'+name][row:row+n]=getattr(block,name)
        values[role+'_offsets'][row:row+n+1]=block.offsets+token
        if tg:
            if role!='train' or len(tg)!=n:raise PermissionError('TRAIN-only target join differs')
            for i,(x,mask,frame,aggregate) in enumerate(tg):
                end=target_cursor+len(x);values['target_raw'][target_cursor:end]=x
                values['target_bits'][target_cursor:end]=np.packbits(mask[:,14:18],axis=-1,bitorder='little')[:,0]
                values['target_offsets'][row+i+1]=end;values['target_frame'][row+i]=frame;values['target_aggregate'][row+i]=aggregate
                safe=np.where(mask,x,0.);total[:]+=safe.sum(0);squares[:]+=(safe*safe).sum(0);support[:]+=mask.sum(0)
                target_cursor=end
        cursors[role]=[row+n,token+len(block.features)]
        chunks+=1;now=time.monotonic()
        if chunks==1 or now-last_report>=60:
            log.mark(f'input decoding coordinate={coordinate} chunks={chunks} '
                f'train_rows={cursors["train"][0]}/{p["counts"]["train"]} '
                f'validation_rows={cursors["validation"][0]}/{p["counts"]["validation"]} '
                f'elapsed_s={now-begun:.1f}')
            last_report=now
    jobs=arguments(spec,parent,arrays,coordinate,targets)
    context=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker) if workers>1 else nullcontext()
    with context as pool:
        if pool is None:
            try:
                for arg in jobs:accept(arg,work(arg))
            finally:close_reader()
        else:
            pending=deque()
            for _ in range(workers):
                arg=next(jobs,None)
                if arg is not None:pending.append((arg,pool.submit(work,arg)))
            while pending:
                arg,future=pending.popleft();accept(arg,future.result());nxt=next(jobs,None)
                if nxt is not None:pending.append((nxt,pool.submit(work,nxt)))
    log.mark(f'input decoding complete coordinate={coordinate}; rechecking source hashes')
    for _,record,_,_,_ in population.selections(parent,arrays):
        if sha256_file(Path(spec['dataset'])/record['path'])!=record['sha256']:raise ValueError('Source changed during construction')
    normalization=None
    for role in c.COUNTS:
        if cursors[role]!=[p['counts'][role],p['particles'][role][side]]:raise ValueError('Full input support/coverage differs')
    if targets:
        log.mark('TRAIN reconstruction normalization and target publication')
        if target_cursor!=p['particles']['train']['offline']:raise ValueError('Incomplete reconstruction targets')
        mean=np.zeros(22);scale=np.ones(22);cols=np.asarray(CONTINUOUS)
        mean[cols]=total[cols]/np.maximum(support[cols],1)
        scale[cols]=np.maximum(np.sqrt(np.maximum(squares[cols]/np.maximum(support[cols],1)-mean[cols]**2,0)),.001)
        target=create('target_values',(target_cursor,22),np.float32)
        for start in range(0,target_cursor,65536):
            x=values['target_raw'][start:start+65536];target[start:start+len(x)]=((x-mean)/scale).astype(np.float32)
        normalization=c.artifact('TARGET_NORMALIZATION',parents=dict(population=p['content_hash']),
            role='train',mean=mean.tolist(),scale=scale.tolist(),support=support.tolist(),maximum=spec['maximum'],
            rows=p['counts']['train'],tokens=target_cursor,frame='paired_HLT_sum_p4_axis_and_vector_pt')
        raw=values.pop('target_raw');raw.flush();raw._mmap.close();del raw
        (node/'target_raw.npy').unlink()
    log.mark(f'input flush and payload authentication coordinate={coordinate}')
    for name,value in values.items():
        value.flush();value._mmap.close();files.append(file_ref(node/(name+'.npy'),root=node))
    values.clear();gc.collect()
    record=c.artifact('NODE_INPUTS',parents=dict(spec=spec['content_hash'],population=p['content_hash']),
        coordinate=coordinate,targets=targets,files=files,normalization=normalization,build_seconds=time.monotonic()-begun)
    write_immutable_json(node/'manifest.json',record)
    log.mark(f'input publication complete coordinate={coordinate} elapsed_s={record["build_seconds"]:.1f}')
    return record


def load_node(spec,node):
    node=Path(node);m=load_json(node/'manifest.json');c.validate(m,'NODE_INPUTS',dict(spec=spec['content_hash'],population=spec['population']['content_hash']))
    values={}
    for ref in m['files']:
        p=validate_file_ref(ref,root=node);a=np.load(p,mmap_mode='r',allow_pickle=False)
        if a.dtype.kind=='f':
            for start in range(0,len(a),65536):
                if not np.isfinite(a[start:start+65536]).all():raise ValueError('Nonfinite node payload')
        values[p.stem]=a
    caches={}
    for role in c.COUNTS:
        fields={k:values[role+'_'+k] for k in ('offsets','features','vectors','identities','labels')}
        caches[role]=RamCache([RamBlock(0,**fields)],role=role,foundation_sha256=spec['population']['content_hash'],coordinate_name=m['coordinate'])
    targets=Targets(values,m['normalization'],caches['train']) if m['targets'] else None
    return caches,targets,m


class Targets:
    def __init__(self,values,normalization,train):
        if train.role!='train':raise PermissionError('TRAIN-only target capability')
        self.values=values['target_values'];self.bits=values['target_bits'];self.offsets=values['target_offsets']
        self.frame=values['target_frame'];self.aggregate=values['target_aggregate'];self.normalization=normalization
        self.mean=np.asarray(normalization['mean']);self.scale=np.asarray(normalization['scale'])
        if (self.offsets.shape!=(len(train)+1,) or self.offsets[0]!=0 or self.offsets[-1]!=len(self.values)
                or np.any(np.diff(self.offsets)<=0) or max(np.diff(self.offsets))>normalization['maximum']):
            raise ValueError('Full TRAIN set support differs')
    def batch(self,indices):
        return [dict(values=np.asarray(self.values[a:b]),valid=((self.bits[a:b,None]>>np.arange(4))&1).astype(bool),
            mean=self.mean,scale=self.scale,frame=self.frame[i],aggregate=self.aggregate[i])
            for i in indices for a,b in [self.offsets[i:i+2]]]

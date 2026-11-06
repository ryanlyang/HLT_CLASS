"""TRAIN-only frozen mixture centers and streamed, memory-mapped full-set banks."""
from pathlib import Path
import gc
import math
import os
import shutil
import time
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from hlt_classification.data.cache_contracts import (load_json,load_npz_arrays,write_immutable_json,
    atomic_publish_bytes,deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_u100_contrastive.training import peaks
from .contracts import artifact,validate,setting
from .probabilistic_contracts import recipe,WIDTH,CONTINUOUS
from . import inputs


def parents(spec):
    return dict(spec=spec['content_hash'],raw=spec['raw_hash'],embedding=spec['embedding_hash'],recipe=recipe()['content_hash'])


def measured(raw,reference):
    """Physical14 -> mixed22, in the paired observable HLT frame (never source-local)."""
    a=np.asarray(raw,np.float64); ref=np.asarray(reference,np.float64)
    if (a.ndim!=2 or a.shape[1]!=14 or not len(a) or not np.isfinite(a).all() or ref.shape!=(3,)
            or not np.isfinite(ref).all() or ref[2]<=0 or not np.isin(a[:,4],[-1,0,1]).all()
            or not np.isin(a[:,5],range(6)).all() or not np.isin(a[:,10:],[0,1]).all()
            or (a[:,3]<0).any() or (a[:,8:10]<0).any()): raise ValueError('Invalid physical set/HLT frame')
    pt=np.hypot(a[:,0],a[:,1]); phi=np.arctan2(a[:,1],a[:,0])-ref[1]
    flags=a[:,10:].astype(bool); valid=flags.copy(); valid[:,2:]&=a[:,8:10]>0
    values=np.zeros((len(a),WIDTH),np.float64); mask=np.ones_like(values,bool)
    values[:,:5]=np.column_stack((np.log(np.maximum(pt/ref[2],1e-12)),np.log(np.maximum(a[:,3]/ref[2],1e-12)),
        np.arcsinh(a[:,2]/np.maximum(pt,1e-12))-ref[0],np.sin(phi),np.cos(phi)))
    values[:,5:11]=np.eye(6)[a[:,5].astype(int)]; values[:,11:14]=np.eye(3)[a[:,4].astype(int)+1]
    values[:,14:16]=np.arcsinh(np.where(valid[:,:2],a[:,6:8],0.))
    values[:,16:18]=np.log(np.maximum(np.where(valid[:,2:],a[:,8:10],1.),1e-12))
    values[:,18:22]=valid; mask[:,14:18]=valid
    if not np.isfinite(values).all(): raise ValueError('Nonfinite transformed set')
    return values,mask


def hlt_reference(spec,train):
    refs=np.empty((len(train.labels),3),np.float64); row=0
    for shard in inputs.raw(spec).shards('HLT',train):
        total=np.add.reduceat(shard['values'][:,:4],shard['offsets'][:-1],axis=0)
        pt=np.maximum(np.hypot(total[:,0],total[:,1]),1e-12); n=len(total)
        refs[row:row+n]=np.column_stack((np.arcsinh(total[:,2]/pt),np.arctan2(total[:,1],total[:,0]),pt)); row+=n
    if row!=len(refs): raise ValueError('HLT frame coverage differs')
    return refs


def transformed(spec,source,train,refs):
    row=0
    for shard in inputs.raw(spec).shards(source,train):
        blocks=[]; masks=[]
        for a,b in zip(shard['offsets'][:-1],shard['offsets'][1:]):
            x,m=measured(shard['values'][a:b],refs[row]); row+=1; blocks.append(x); masks.append(m)
        yield shard,np.concatenate(blocks),np.concatenate(masks)
    if row!=len(train.labels): raise ValueError('Incomplete full-set TRAIN coverage')


def prepare_source(spec,source,train,refs):
    folder=Path(spec['root'])/'setup'; counts=[]; sums=np.zeros(WIDTH); squares=sums.copy(); support=sums.copy()
    for shard,x,mask in transformed(spec,source,train,refs):
        counts.extend(np.diff(shard['offsets']).tolist()); support+=mask.sum(0)
        sums+=np.where(mask,x,0.).sum(0); squares+=np.where(mask,x*x,0.).sum(0)
    mean=np.zeros(WIDTH); scale=np.ones(WIDTH); cols=np.array(CONTINUOUS)
    mean[cols]=sums[cols]/np.maximum(support[cols],1)
    scale[cols]=np.maximum(np.sqrt(np.maximum(squares[cols]/np.maximum(support[cols],1)-mean[cols]**2,0)),.001)
    offsets=np.r_[0,np.cumsum(counts)].astype(np.int64); total=int(offsets[-1]); staged=[]; outputs=[]
    needed=total*(WIDTH*4+1)+offsets.nbytes
    used=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
    if used+needed>spec['storage_budget_bytes']-2**30 or shutil.disk_usage(folder).free<needed+2**30:
        raise OSError('Full-set target bank exceeds registered output allowance/headroom; preserve preparation')
    # These arrays stay on disk; no full raw collection or duplicate normalized
    # collection is retained in a worker. Master manifest is published last.
    for suffix,dtype,shape in (('values',np.float32,(total,WIDTH)),('valid',np.uint8,(total,))):
        path=folder/f'set_{source}_{suffix}.npy'; tmp=path.with_suffix('.npy.partial')
        if path.exists() or tmp.exists(): raise FileExistsError('Preserve prior target preparation')
        array=np.lib.format.open_memmap(tmp,mode='w+',dtype=dtype,shape=shape); staged.append((path,tmp,array))
    start=0
    for shard,x,mask in transformed(spec,source,train,refs):
        n=len(x); staged[0][2][start:start+n]=np.where(mask,(x-mean)/scale,0.).astype(np.float32)
        staged[1][2][start:start+n]=np.packbits(mask[:,14:18],axis=-1,bitorder='little')[:,0]; start+=n
    if start!=total: raise ValueError('Full collection changed between preparation passes')
    for path,tmp,array in staged:
        array.flush(); array._mmap.close()
        # Exclusive setup claim and fresh-root check prevent writer races.
        if path.exists(): raise FileExistsError(path)
        os.replace(tmp,path); outputs.append(file_ref(path,root=folder))
    staged.clear(); gc.collect()
    path=folder/f'set_{source}_index.npz'
    atomic_publish_bytes(path,deterministic_npz_bytes(dict(identities=train.identities,labels=train.labels,offsets=offsets)))
    outputs.append(file_ref(path,root=folder))
    record=artifact('SET_BANK',parents=parents(spec),source=source,role='train',rows=len(train.labels),tokens=total,
        maximum_count=max(counts),count_quantiles=np.quantile(counts,[0,.5,.9,.99,1]).tolist(),
        mean=mean.tolist(),scale=scale.tolist(),support=support.astype(np.int64).tolist(),
        width=WIDTH,full_collection=True,frame='paired_HLT_sum_p4_axis_and_vector_pt',outputs=outputs)
    write_immutable_json(folder/f'set_{source}.json',record)
    return file_ref(folder/f'set_{source}.json',root=folder)


def prepare(spec,train,bank):
    if train.role!='train' or len(train.labels)!=spec['counts']['train']: raise PermissionError('Full TRAIN targets only')
    if any(not np.array_equal(getattr(bank,k),getattr(train,k)) for k in ('identities','labels')) or bank.manifest['content_hash']!=spec['embedding_hash']:
        raise ValueError('Density TRAIN join/embedding lineage differs')
    began=time.perf_counter(); root=Path(spec['root'])/'setup'; outputs=[]
    z=np.asarray(bank.targets)
    if z.dtype!=np.float32 or z.shape!=(len(train.labels),128) or not np.isfinite(z).all(): raise ValueError('Invalid standardized U100 bank')
    centers={}
    for k in (1,4):
        estimator=MiniBatchKMeans(n_clusters=k,random_state=2026110509,n_init=3,batch_size=4096,
            max_iter=100,max_no_improvement=20,reassignment_ratio=0.,tol=0.)
        centers['k'+str(k)]=estimator.fit(z).cluster_centers_.astype(np.float32)
    payload=root/'density_centers.npz'; atomic_publish_bytes(payload,deterministic_npz_bytes(centers)); outputs.append(file_ref(payload,root=root))
    refs=hlt_reference(spec,train)
    for source in ('OFFLINE','HLT'):
        print('GROUP_C preparing full '+source+' TRAIN set',flush=True)
        outputs.append(prepare_source(spec,source,train,refs))
    record=artifact('PROBABILISTIC_TARGETS',parents=parents(spec),role='train',rows=len(train.labels),
        outputs=outputs,preparation_seconds=time.perf_counter()-began,statistics_peak_host_bytes=peaks('cpu').get('peak_rss_bytes'),
        density_initialization=recipe()['density']['initialization'],quality_gate=False)
    write_immutable_json(root/'probabilistic_targets.json',record)
    return record


def authenticate(spec):
    root=Path(spec['root'])/'setup'; record=load_json(root/'probabilistic_targets.json'); validate(record,'PROBABILISTIC_TARGETS',parents(spec))
    expected=['density_centers.npz','set_OFFLINE.json','set_HLT.json']
    if (record['role']!='train' or record['rows']!=spec['counts']['train'] or record['quality_gate'] is not False
            or [r['path'] for r in record['outputs']]!=expected
            or record['density_initialization']!=recipe()['density']['initialization']
            or not math.isfinite(record['preparation_seconds']) or record['preparation_seconds']<0): raise ValueError('Group C target scope/recipe differs')
    for r in record['outputs']: validate_file_ref(r,root=root)
    centers=load_npz_arrays(root/'density_centers.npz')
    if set(centers)!={'k1','k4'} or any(v.dtype!=np.float32 or v.shape!=(int(k[1:]),128) or not np.isfinite(v).all() for k,v in centers.items()):
        raise ValueError('Invalid frozen mixture initialization')
    for source in ('OFFLINE','HLT'):
        bank=load_json(root/f'set_{source}.json'); validate(bank,'SET_BANK',parents(spec))
        if (bank['source']!=source or bank['role']!='train' or bank['rows']!=spec['counts']['train']
                or bank['width']!=WIDTH or bank['full_collection'] is not True
                or bank['frame']!='paired_HLT_sum_p4_axis_and_vector_pt'
                or [r['path'] for r in bank['outputs']]!=[f'set_{source}_{n}' for n in ('values.npy','valid.npy','index.npz')]
                or len(bank['mean'])!=WIDTH or len(bank['scale'])!=WIDTH or len(bank['support'])!=WIDTH
                or not np.isfinite(bank['mean']+bank['scale']).all() or min(bank['scale'])<.001): raise ValueError('Invalid set bank semantics')
        for r in bank['outputs']: validate_file_ref(r,root=root)
    return record


class DensityTargets:
    def __init__(self,spec,train,bank,centers):
        self.targets=bank.targets; self.centers=centers
        self.capacity=dict(rows=len(train.labels),class_counts=np.bincount(train.labels,minlength=11).tolist(),source='U100',width=128)
    def initialize(self,model):
        import torch
        with torch.no_grad(): model.aux.mean.bias.copy_(torch.as_tensor(self.centers,device=model.aux.mean.bias.device).flatten())
    def batch(self,indices,epoch,cfg,mine=None): return dict(targets=np.asarray(self.targets[indices],np.float32)),[]


class SetTargets:
    def __init__(self,spec,train,source):
        self.source=source; self.root=Path(spec['root'])/'setup'; record=load_json(self.root/f'set_{source}.json')
        self.values=np.load(self.root/f'set_{source}_values.npy',mmap_mode='r',allow_pickle=False)
        self.bits=np.load(self.root/f'set_{source}_valid.npy',mmap_mode='r',allow_pickle=False)
        index=load_npz_arrays(self.root/f'set_{source}_index.npz'); self.offsets=index['offsets']; self.counts=np.diff(self.offsets)
        if (set(index)!={'identities','labels','offsets'} or any(not np.array_equal(index[k],getattr(train,k)) for k in ('identities','labels'))
                or self.offsets.dtype!=np.int64 or self.offsets.shape!=(len(train.labels)+1,) or self.offsets[0]!=0
                or (self.counts<1).any() or self.values.shape!=(self.offsets[-1],WIDTH) or self.values.dtype!=np.float32
                or self.bits.shape!=(self.offsets[-1],) or self.bits.dtype!=np.uint8 or int(self.counts.max())!=record['maximum_count']
                or self.offsets[-1]!=record['tokens']): raise ValueError('Set bank TRAIN join/cardinality differs')
        # Full payload finite scan is bounded in memory; no silent nonfinite rows.
        for start in range(0,len(self.values),65536):
            if not np.isfinite(self.values[start:start+65536]).all() or (self.bits[start:start+65536]>15).any(): raise ValueError('Corrupt set values/validity')
        self.capacity=dict(rows=len(train.labels),class_counts=np.bincount(train.labels,minlength=11).tolist(),
            source=source,maximum_count=int(self.counts.max()),tokens=int(self.offsets[-1]),full_collection=True)
    def initialize(self,model): model.aux.configure(int(self.counts.max()))
    def batch(self,indices,epoch,cfg,mine=None):
        ix=np.asarray(indices,np.int64); counts=self.counts[ix]; n=int(counts.max()); b=len(ix)
        noisy=np.zeros((b,n,WIDTH),np.float32); velocity=np.zeros_like(noisy); valid=np.zeros_like(noisy,bool)
        real=np.zeros((b,n),bool); levels=np.empty(b,np.float32)
        for j,i in enumerate(ix):
            rng=np.random.default_rng(np.random.SeedSequence([2026110510,1000003*cfg['seed_index'],int(epoch),int(i)]))
            a,z=self.offsets[i:i+2]; permutation=rng.permutation(z-a)
            clean=np.asarray(self.values[a:z])[permutation]; flags=((self.bits[a:z,None]>>np.arange(4))&1).astype(bool)[permutation]
            levels[j]=rng.uniform(.2,.95); alpha=np.float32(np.cos(np.pi*levels[j]/2)); sigma=np.float32(np.sin(np.pi*levels[j]/2))
            noise=rng.standard_normal(clean.shape).astype(np.float32)
            noisy[j,:z-a]=alpha*clean+sigma*noise; velocity[j,:z-a]=alpha*noise-sigma*clean
            real[j,:z-a]=True; valid[j,:z-a]=True; valid[j,:z-a,14:18]=flags
        return dict(noisy=noisy,velocity=velocity,valid=valid,real=real,time=levels,count=counts.astype(np.int64)),[]


def candidates(spec,task,train):
    if train.role!='train' or len(train.labels)!=spec['counts']['train']: raise PermissionError('Full TRAIN targets only')
    authenticate(spec); cfg=setting(task)
    if cfg['family']=='D':
        bank=inputs.embedding_bank(spec,train)
        centers=load_npz_arrays(Path(spec['root'])/'setup/density_centers.npz')['k'+str(cfg['components'])]
        return DensityTargets(spec,train,bank,centers)
    return SetTargets(spec,train,cfg['source'])


def miniature_indices(source,indices):
    if not isinstance(source,SetTargets): return indices
    # Include 128 largest target sets, 128 most valid tracking rows, native
    # extremes, and a real partial batch. No anchor or target-particle crop.
    good=np.empty(len(source.counts),np.int64)
    for i,(a,b) in enumerate(zip(source.offsets[:-1],source.offsets[1:])): good[i]=np.count_nonzero(source.bits[a:b])
    ix=list(dict.fromkeys([*np.argsort(-source.counts,kind='stable')[:128],*np.argsort(-good,kind='stable')[:128],*indices]))
    for i in range(len(source.counts)):
        if len(ix)%128 not in (0,1): break
        if i not in ix: ix.append(i)
    return np.asarray(ix,np.int64)


def conditioning_permutation(raw):
    labels=np.asarray(raw['labels']); vectors=np.asarray(raw['vectors']); real=np.asarray(raw['mask']).reshape(len(labels),-1).astype(bool)
    total=np.where(real[:,None,:],vectors,0.).sum(-1); pt=np.hypot(total[:,0],total[:,1])
    keys=np.column_stack((labels,np.floor(np.log(np.maximum(pt,1e-12))),real.sum(-1)//16))
    permutation=np.arange(len(labels))
    _,group=np.unique(keys,axis=0,return_inverse=True)
    for key in np.unique(group):
        ix=np.where(group==key)[0]
        if len(ix)>1: permutation[ix]=np.roll(ix,1)
    return permutation

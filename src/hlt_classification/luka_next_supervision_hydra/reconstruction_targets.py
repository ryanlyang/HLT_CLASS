"""Authenticated, ragged, complete TRAIN targets; no conditional target input."""
from pathlib import Path
import gc
import math
import os
import shutil
import time
import numpy as np
from hlt_classification.data.cache_contracts import load_json,load_npz_arrays,write_immutable_json,atomic_publish_bytes,deterministic_npz_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_u100_contrastive.training import peaks
from . import inputs
from .reconstruction_contracts import artifact,validate,recipe,WIDTH,CONTINUOUS
from .probabilistic_targets import measured,hlt_reference,transformed


def parents(spec):
    return dict(spec=spec['content_hash'],raw=spec['raw_hash'],recipe=recipe()['content_hash'])


def storage_bytes(sources,rows,*,fits=24):
    # Actual token inventory, two compact index/frame/aggregate banks, twelve
    # science + twelve warm/measured native exports, full predictions and logs.
    banks=sum(s['tokens'] for s in sources.values())*89+2*rows*120
    fit=16*2**20+74895*84+30*2**20
    return int(banks+fits*fit+2*2**30)


def inventory(spec):
    raw=inputs.raw(spec); sources={}
    for source in ('OFFLINE','HLT'):
        tokens=rows=maximum=0
        for name in sorted(raw.rows):
            if not name.startswith(source+'/'): continue
            a=load_npz_arrays(raw.path(name)); offsets=a['offsets']; counts=np.diff(offsets)
            if (offsets.dtype!=np.int64 or offsets[0]!=0 or len(offsets)!=len(a['labels'])+1
                    or (counts<1).any() or a['values'].shape!=(offsets[-1],14)):
                raise ValueError('Malformed raw reconstruction inventory')
            rows+=len(counts); tokens+=int(offsets[-1]); maximum=max(maximum,int(counts.max()))
        if rows!=spec['counts']['train']: raise ValueError('Incomplete TRAIN reconstruction inventory')
        sources[source]=dict(rows=rows,tokens=tokens,maximum_count=maximum)
    return artifact('INVENTORY',parents=dict(raw=spec['raw_hash']),sources=sources,
        maximum=max(1,*(s['maximum_count'] for s in sources.values())),
        storage_budget_bytes=storage_bytes(sources,rows),quality_gate=False)


def prepare_source(spec,source,train,refs):
    folder=Path(spec['root'])/'setup'; counts=[]; sums=np.zeros(WIDTH); squares=sums.copy(); support=sums.copy(); aggregates=[]
    for shard,x,mask in transformed(spec,source,train,refs):
        counts.extend(np.diff(shard['offsets']).tolist()); support+=mask.sum(0)
        sums+=np.where(mask,x,0.).sum(0); squares+=np.where(mask,x*x,0.).sum(0)
        for a,b in zip(shard['offsets'][:-1],shard['offsets'][1:]):
            v=shard['values'][a:b]; aggregates.append(np.r_[v[:,:4].sum(0),v[v[:,4]!=0,3].sum()])
    mean=np.zeros(WIDTH); scale=np.ones(WIDTH); cols=np.array(CONTINUOUS)
    mean[cols]=sums[cols]/np.maximum(support[cols],1)
    scale[cols]=np.maximum(np.sqrt(np.maximum(squares[cols]/np.maximum(support[cols],1)-mean[cols]**2,0)),.001)
    offsets=np.r_[0,np.cumsum(counts)].astype(np.int64); total=int(offsets[-1])
    actual=dict(rows=len(counts),tokens=total,maximum_count=max(counts))
    if actual!=spec['target_inventory']['sources'][source]: raise ValueError('TRAIN inventory changed')
    needed=total*89+len(counts)*120
    used=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
    if used+needed>spec['storage_budget_bytes']-2**30 or shutil.disk_usage(folder).free<needed+2**30:
        raise OSError('Full reconstruction targets exceed registered storage; preserve files')
    staged=[]; outputs=[]
    for suffix,dtype,shape in (('values',np.float32,(total,WIDTH)),('valid',np.uint8,(total,))):
        path=folder/f'reconstruction_{source}_{suffix}.npy'; tmp=path.with_suffix('.npy.partial')
        if path.exists() or tmp.exists(): raise FileExistsError('Preserve previous reconstruction preparation')
        array=np.lib.format.open_memmap(tmp,mode='w+',dtype=dtype,shape=shape); staged.append((path,tmp,array))
    start=0
    for shard,x,mask in transformed(spec,source,train,refs):
        n=len(x); staged[0][2][start:start+n]=np.where(mask,(x-mean)/scale,0.).astype(np.float32)
        staged[1][2][start:start+n]=np.packbits(mask[:,14:18],axis=-1,bitorder='little')[:,0]; start+=n
    if start!=total: raise ValueError('Incomplete reconstruction bank')
    for path,tmp,array in staged:
        array.flush(); array._mmap.close(); os.replace(tmp,path); outputs.append(file_ref(path,root=folder))
    staged.clear(); gc.collect()
    path=folder/f'reconstruction_{source}_index.npz'
    atomic_publish_bytes(path,deterministic_npz_bytes(dict(identities=train.identities,labels=train.labels,offsets=offsets,
        frame=refs,aggregate=np.asarray(aggregates,np.float64))))
    outputs.append(file_ref(path,root=folder))
    record=artifact('BANK',parents=parents(spec),source=source,role='train',**actual,
        count_quantiles=np.quantile(counts,[0,.5,.9,.99,1]).tolist(),
        mean=mean.tolist(),scale=scale.tolist(),support=support.astype(np.int64).tolist(),
        width=WIDTH,full_collection=True,frame='paired_HLT_sum_p4_axis_and_vector_pt',outputs=outputs)
    path=folder/f'reconstruction_{source}.json'; write_immutable_json(path,record)
    return file_ref(path,root=folder)


def prepare(spec,train):
    if train.role!='train' or len(train.labels)!=spec['counts']['train']: raise PermissionError('Full TRAIN targets only')
    began=time.perf_counter(); root=Path(spec['root'])/'setup'; refs=hlt_reference(spec,train); outputs=[]
    for source in ('OFFLINE','HLT'):
        print('DIRECT_RECONSTRUCTION preparing complete '+source+' TRAIN particles',flush=True)
        outputs.append(prepare_source(spec,source,train,refs))
    result=artifact('TARGETS',parents=parents(spec),role='train',rows=len(train.labels),outputs=outputs,
        maximum=spec['target_inventory']['maximum'],preparation_seconds=time.perf_counter()-began,
        statistics_peak_host_bytes=peaks('cpu').get('peak_rss_bytes'),quality_gate=False)
    write_immutable_json(root/'reconstruction_targets.json',result)
    return result


def authenticate(spec):
    root=Path(spec['root'])/'setup'; result=load_json(root/'reconstruction_targets.json'); validate(result,'TARGETS',parents(spec))
    if (result['role']!='train' or result['rows']!=spec['counts']['train'] or result['quality_gate'] is not False
            or result['maximum']!=spec['target_inventory']['maximum']
            or [r['path'] for r in result['outputs']]!=['reconstruction_OFFLINE.json','reconstruction_HLT.json']
            or not math.isfinite(result['preparation_seconds']) or result['preparation_seconds']<0):
        raise ValueError('Invalid reconstruction target scope/support')
    for ref in result['outputs']:
        validate_file_ref(ref,root=root); bank=load_json(root/ref['path']); validate(bank,'BANK',parents(spec)); source=bank['source']
        if (source not in ('OFFLINE','HLT') or ref['path']!=f'reconstruction_{source}.json' or bank['role']!='train'
                or any(bank[k]!=v for k,v in spec['target_inventory']['sources'][source].items())
                or bank['width']!=WIDTH or bank['full_collection'] is not True
                or bank['frame']!='paired_HLT_sum_p4_axis_and_vector_pt'
                or [r['path'] for r in bank['outputs']]!=[f'reconstruction_{source}_{n}' for n in ('values.npy','valid.npy','index.npz')]
                or len(bank['mean'])!=WIDTH or len(bank['scale'])!=WIDTH or len(bank['support'])!=WIDTH
                or not np.isfinite(bank['mean']+bank['scale']).all() or min(bank['scale'])<.001):
            raise ValueError('Invalid reconstruction bank')
        for r in bank['outputs']: validate_file_ref(r,root=root)
    return result


class Targets:
    def __init__(self,spec,train,source):
        self.root=Path(spec['root'])/'setup'; self.source=source; self.maximum=spec['target_inventory']['maximum']
        record=load_json(self.root/f'reconstruction_{source}.json'); self.mean=np.array(record['mean']); self.scale=np.array(record['scale'])
        self.values=np.load(self.root/f'reconstruction_{source}_values.npy',mmap_mode='r',allow_pickle=False)
        self.bits=np.load(self.root/f'reconstruction_{source}_valid.npy',mmap_mode='r',allow_pickle=False)
        index=load_npz_arrays(self.root/f'reconstruction_{source}_index.npz')
        self.offsets=index['offsets']; self.counts=np.diff(self.offsets); self.frame=index['frame']; self.aggregate=index['aggregate']
        n=len(train.labels)
        if (set(index)!={'identities','labels','offsets','frame','aggregate'}
                or any(not np.array_equal(index[k],getattr(train,k)) for k in ('identities','labels'))
                or self.offsets.dtype!=np.int64 or self.offsets.shape!=(n+1,) or self.offsets[0]!=0 or (self.counts<1).any()
                or self.values.shape!=(self.offsets[-1],22) or self.values.dtype!=np.float32
                or self.bits.shape!=(self.offsets[-1],) or self.bits.dtype!=np.uint8
                or self.frame.shape!=(n,3) or self.aggregate.shape!=(n,5) or (self.frame[:,2]<=0).any()
                or not np.isfinite(self.frame).all() or not np.isfinite(self.aggregate).all()
                or self.counts.max()!=record['maximum_count'] or self.offsets[-1]!=record['tokens']
                or self.counts.max()>self.maximum): raise ValueError('Invalid TRAIN reconstruction join/support')
        for start in range(0,len(self.values),65536):
            if not np.isfinite(self.values[start:start+65536]).all() or (self.bits[start:start+65536]>15).any():
                raise ValueError('Corrupt reconstruction payload')
        self.capacity=dict(rows=n,class_counts=np.bincount(train.labels,minlength=11).tolist(),source=source,
            maximum_count=int(self.counts.max()),fixed_output_slots=self.maximum,tokens=int(self.offsets[-1]),full_collection=True)
    def initialize(self,model):
        from .reconstruction_model import attach
        attach(model,self.maximum)
    def batch(self,indices,epoch,cfg,mine=None):
        if cfg['family']!='R' or cfg['source']!=self.source: raise ValueError('Wrong reconstruction source')
        result=[]
        for i in np.asarray(indices,np.int64):
            a,b=self.offsets[i:i+2]
            result.append(dict(values=np.asarray(self.values[a:b]),valid=((self.bits[a:b,None]>>np.arange(4))&1).astype(bool),
                mean=self.mean,scale=self.scale,frame=self.frame[i],aggregate=self.aggregate[i]))
        return result,[]
    def close(self):
        self.values._mmap.close(); self.bits._mmap.close()


def candidates(spec,task,train):
    from .contracts import setting
    if train.role!='train' or len(train.labels)!=spec['counts']['train']: raise PermissionError('Full TRAIN targets only')
    authenticate(spec); return Targets(spec,train,setting(task)['source'])


def miniature_indices(source,indices):
    good=np.empty(len(source.counts),np.int64)
    for i,(a,b) in enumerate(zip(source.offsets[:-1],source.offsets[1:])): good[i]=np.count_nonzero(source.bits[a:b])
    ix=list(dict.fromkeys([*np.argsort(-source.counts,kind='stable')[:128],*np.argsort(-good,kind='stable')[:128],*indices]))
    for i in range(len(source.counts)):
        if len(ix)%128 not in (0,1): break
        if i not in ix: ix.append(i)
    return np.asarray(ix,np.int64)

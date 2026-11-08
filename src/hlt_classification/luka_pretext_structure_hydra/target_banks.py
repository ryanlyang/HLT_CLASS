"""Immutable memmaps, full physical support, and TRAIN-only normalizers."""
from pathlib import Path
import os
import numpy as np
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from .contracts import artifact, validate, TEACHERS
from . import structure as st, corruption as co
from .normalization import Moments


def publish_array(path,array):
    """Publish a complete NPY without a second population-sized bytes copy."""
    path=Path(path); temp=path.with_suffix(path.suffix+'.partial')
    with temp.open('xb') as f:
        np.save(f,array,allow_pickle=False); f.flush(); os.fsync(f.fileno())
    os.link(temp,path); temp.unlink()


def finish_memmap(path,array):
    array.flush(); array._mmap.close(); del array
    path=Path(path); temp=path.with_suffix(path.suffix+'.partial')
    with temp.open('r+b') as f: os.fsync(f.fileno())
    os.link(temp,path); temp.unlink()


def open_staging(path,dtype,shape):
    path=Path(path); temp=path.with_suffix(path.suffix+'.partial')
    # Exclusive claim, including across processes; partial files are never reused.
    with temp.open('xb'): pass
    return np.lib.format.open_memmap(temp,mode='w+',dtype=dtype,shape=shape)


def prepare_raw(folder,shards,cache,source,parent_hash,*,reserve=None):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=False)
    counts=[]
    for a in shards(): counts.extend(np.diff(a['offsets']).tolist())
    if len(counts)!=len(cache): raise ValueError('Physical row coverage differs')
    offsets=np.r_[0,np.cumsum(counts)].astype(np.int64)
    if reserve: reserve(int(offsets[-1])*14*8+offsets.nbytes+cache.identities.nbytes+cache.labels.nbytes+4096)
    dst=folder/'values.npy'; out=open_staging(dst,np.float64,(int(offsets[-1]),14)); row=0
    for a in shards():
        n=len(a['labels']); out[offsets[row]:offsets[row+n]]=a['values']; row+=n
    finish_memmap(dst,out); del out
    for name,value in dict(offsets=offsets,identities=cache.identities,labels=cache.labels).items():
        publish_array(folder/(name+'.npy'),value)
    value=artifact('INVENTORY',parents=dict(raw=parent_hash,prepared=cache.foundation_sha256),role=cache.role,
        source=source,rows=len(cache),tokens=int(offsets[-1]),maximum=max(counts),
        files=[file_ref(folder/(k+'.npy'),root=folder) for k in ('values','offsets','identities','labels')])
    write_immutable_json(folder/'manifest.json',value)
    return PhysicalBank(folder,cache,source)


class PhysicalBank:
    def __init__(self,folder,cache,source):
        self.folder=Path(folder); self.manifest=load_json(self.folder/'manifest.json'); m=self.manifest
        validate(m,'INVENTORY')
        if (cache.role not in ('train','validation') or m['role']!=cache.role or m['source']!=source
                or m['rows']!=len(cache) or m['parents']['prepared']!=cache.foundation_sha256
                or [r['path'] for r in m['files']]!=[k+'.npy' for k in ('values','offsets','identities','labels')]):
            raise PermissionError('Physical bank role/source/population differs')
        for ref in m['files']: validate_file_ref(ref,root=self.folder)
        for k in ('values','offsets','identities','labels'):
            setattr(self,k,np.load(self.folder/(k+'.npy'),mmap_mode='r',allow_pickle=False))
        if (any(not np.array_equal(getattr(self,k),getattr(cache,k)) for k in ('identities','labels'))
                or self.values.dtype!=np.float64 or self.values.shape!=(m['tokens'],14)
                or self.offsets.dtype!=np.int64 or self.offsets.shape!=(len(cache)+1,)
                or self.offsets[0]!=0 or self.offsets[-1]!=m['tokens'] or (np.diff(self.offsets)<1).any()
                or int(np.diff(self.offsets).max())!=m['maximum']):
            raise ValueError('Physical bank metadata differs')
        self.role=cache.role; self.source=source
    def __len__(self): return len(self.labels)
    def jet(self,i): return self.values[self.offsets[i]:self.offsets[i+1]]


def prepare_structure(folder,physical,*,normalizers=None,reserve=None):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=False)
    if (physical.role=='train') != (normalizers is None):
        raise PermissionError('Only TRAIN may fit target normalization')
    example=st.all_families(physical.jet(0)); arrays={}; masks={}; moments={}
    if reserve: reserve(sum(len(t.values) for t in example.values())*len(physical)*5+8192)
    for name,t in example.items():
        arrays[name]=open_staging(folder/(name+'.npy'),np.float32,(len(physical),len(t.values)))
        masks[name]=open_staging(folder/(name+'_valid.npy'),np.bool_,arrays[name].shape)
        moments[name]=Moments(len(t.values))
    for i in range(len(physical)):
        targets=st.all_families(physical.jet(i))
        for name,t in targets.items():
            arrays[name][i]=t.values; masks[name][i]=t.valid
            if normalizers is None: moments[name].add(t.values,t.valid)
        if i%5000==0: print(f'PRETEXT structure {physical.source}/{physical.role}: {i+1}/{len(physical)}',flush=True)
    parents=dict(physical=physical.manifest['content_hash'])
    norms=normalizers or {name:moments[name].finish(parents=parents,binary=t.binary,blocks=t.blocks) for name,t in example.items()}
    outputs=[]
    for name in example:
        finish_memmap(folder/(name+'.npy'),arrays.pop(name)); finish_memmap(folder/(name+'_valid.npy'),masks.pop(name))
        outputs.extend(file_ref(folder/f,root=folder) for f in (name+'.npy',name+'_valid.npy'))
    value=artifact('STRUCTURE_BANK',parents=parents,source=physical.source,role=physical.role,rows=len(physical),
                   normalizers=norms,files=outputs)
    write_immutable_json(folder/'manifest.json',value)
    return StructureBank(folder,physical)


class StructureBank:
    def __init__(self,folder,physical):
        self.folder=Path(folder); self.manifest=load_json(self.folder/'manifest.json'); m=self.manifest
        validate(m,'STRUCTURE_BANK',dict(physical=physical.manifest['content_hash']))
        if m['role']!=physical.role or m['source']!=physical.source or m['rows']!=len(physical):
            raise PermissionError('Structure capability differs')
        if {r['path'] for r in m['files']}!={n+s+'.npy' for n in ('C2','C3','B','R') for s in ('','_valid')}:
            raise ValueError('Structure inventory differs')
        for ref in m['files']: validate_file_ref(ref,root=self.folder)
        self.normalizers=m['normalizers']; self.values={}; self.valid={}
        for name,norm in self.normalizers.items():
            validate(norm,'NORMALIZATION')
            if norm['role']!='train': raise PermissionError('Validation-fitted moments forbidden')
            self.values[name]=np.load(self.folder/(name+'.npy'),mmap_mode='r',allow_pickle=False)
            self.valid[name]=np.load(self.folder/(name+'_valid.npy'),mmap_mode='r',allow_pickle=False)
            if self.values[name].shape!=(len(physical),len(norm['mean'])) or self.valid[name].shape!=self.values[name].shape:
                raise ValueError('Structure support differs')
        self.identities=physical.identities; self.labels=physical.labels; self.role=physical.role
    def target(self,i,name):
        norm=self.normalizers[name]
        return st.summary(self.values[name][i],self.valid[name][i],norm['binary'],norm['blocks'])


def primary(task,clean,visible,removed,structure,i):
    if task in ('T01','T02','T03','T08'):
        return co.set_target(clean if task=='T08' else clean[removed],visible)
    if task=='T04': return {'map':co.patch_map(clean,removed,visible)}
    if task=='T05': return {'pairs':co.missing_pairs(clean,removed)}
    return {n:structure.target(i,n) for n in (('C2','C3') if task=='T06' else ('B',))}


def prepare_pretext_statistics(folder,physical,structures):
    if physical.role!='train' or structures.role!='train': raise PermissionError('TRAIN moments only')
    result={}; parents=dict(raw=physical.manifest['content_hash'],structure=structures.manifest['content_hash'])
    for task in TEACHERS:
        if task in ('T06','T07'):
            result[task]={n:structures.normalizers[n] for n in (('C2','C3') if task=='T06' else ('B',))}; continue
        moments={}; templates={}
        for i in range(len(physical)):
            for draw in range(2):
                visible,clean,removed=co.corrupt(physical.jet(i),physical.identities[i],task,0,draw)
                target=primary(task,clean,visible,removed,structures,i)
                if 'kinematic' in target:
                    data={'set':st.summary(np.zeros(10))}
                    x=np.concatenate((target['kinematic'],target['track']),axis=-1)
                    ok=np.concatenate((np.ones_like(target['kinematic'],bool),target['validity']),axis=-1)
                    moments.setdefault('set',Moments(10)).add(x,ok); templates.update(data)
                else:
                    for name,t in target.items():
                        moments.setdefault(name,Moments(len(t.values))).add(t.values,t.valid); templates[name]=t
            if i%10000==0: print(f'PRETEXT TRAIN moments {task}: {i+1}/{len(physical)}',flush=True)
        result[task]={name:m.finish(parents=dict(parents,task=task),binary=templates[name].binary,blocks=templates[name].blocks)
                      for name,m in moments.items()}
    value=artifact('STATISTICS',parents=parents,role='train',draws=2,teachers=result)
    write_immutable_json(Path(folder)/'statistics.json',value)
    return value

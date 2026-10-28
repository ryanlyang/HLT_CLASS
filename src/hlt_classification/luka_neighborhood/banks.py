"""Once-built ragged TRAIN neighborhoods, borrowed compact teacher targets."""
from pathlib import Path
import os
import shutil
import numpy as np
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_attention.banks import counts, alignment
from hlt_classification.luka_trial_hybrid import banks as compact
from hlt_classification.luka_fullsim import regular_cache, stage2
from hlt_classification.luka_fullsim.particles import physical
from .contracts import artifact, validate, target_recipe, recipe, FAMILIES, TEACHERS
from .targets import all_descriptors, Moments


def required_bytes(cache):
    return int(counts(cache).sum())*432 + (len(cache)+1)*8 + len(cache)*40 + 2*2**30


def physical_rows(spec, cache, meta):
    """Decode selected TRAIN only and replay the EXACT native HLT arrays."""
    manifest,prepared,inventory,splits=meta
    if (cache.role!='train' or cache.coordinate_name!='D000' or prepared['content_hash']!=cache.foundation_sha256
            or manifest['content_hash']!=spec['anchor']['evidence']['cache']): raise PermissionError('Only authenticated D000 TRAIN')
    regular_cache.validate_splits(splits,inventory)
    position=0
    for member in splits['memberships']['train']:
        for jet in regular_cache.selected_file_rows(spec['anchor']['evidence']['raw_container'],inventory,member):
            if position>=len(cache): raise ValueError('Extra selected TRAIN rows')
            hlt=physical(jet.hlt,prepared['conventions'],'hlt'); offline=physical(jet.offline,prepared['conventions'],'offline')
            rebuilt=stage2.build_inputs(hlt,capacity=prepared['inputs']['capacity'])
            raw=cache.batch(np.asarray([position],np.int64)); n=int(raw['mask'][0].sum())
            if (bytes(cache.identities[position]).hex()!=jet.identity or cache.labels[position]!=jet.label
                    or n!=len(hlt) or not np.array_equal(rebuilt.features.T,raw['features'][0,:,:n])
                    or not np.array_equal(rebuilt.vectors.T,raw['vectors'][0,:,:n])):
                raise ValueError('Native HLT input/order/identity fingerprint differs')
            yield hlt,offline
            position+=1
        print(f'NEIGHBOR targets selected TRAIN rows={position}/{len(cache)}',flush=True)
    if position!=len(cache): raise ValueError('Incomplete selected TRAIN coverage')


def build(root, cache, rows, *, source_hash, available_bytes):
    if cache.role!='train' or cache.coordinate_name!='D000': raise PermissionError('TRAIN-only targets')
    root=Path(root); needed=required_bytes(cache)
    if available_bytes<needed or shutil.disk_usage(root.parent).free<needed: raise OSError(f'Need {needed/2**30:.2f} GiB including reserve')
    root.mkdir(exist_ok=False); offsets=np.r_[0,np.cumsum(counts(cache))].astype(np.int64)
    moments={name:Moments() for name in FAMILIES}; mapped={}; seen=0
    try:
        for name in FAMILIES:
            mapped[name]=np.lib.format.open_memmap(root/(name+'.partial.npy'),mode='w+',dtype=np.float32,shape=(int(offsets[-1]),36))
        for hlt,offline in rows:
            if seen>=len(cache) or len(hlt)!=offsets[seen+1]-offsets[seen]: raise ValueError('Target population/support differs')
            values=all_descriptors(hlt,offline)
            for name in FAMILIES:
                mapped[name][offsets[seen]:offsets[seen+1]]=values[name]; moments[name].add(values[name])
            seen+=1
        if seen!=len(cache): raise ValueError('Missing target jets')
        for a in mapped.values(): a.flush()
    finally:
        for a in mapped.values(): a._mmap.close()
    index=dict(identities=cache.identities,labels=cache.labels,offsets=offsets)
    for name in FAMILIES:
        mean,scale=moments[name].result(); index[name+'_mean']=mean; index[name+'_scale']=scale
        path=root/(name+'.npy')
        if path.exists() or path.is_symlink(): raise FileExistsError(path)
        os.rename(root/(name+'.partial.npy'),path)
    atomic_publish_bytes(root/'index.npz',deterministic_npz_bytes(index))
    result=artifact('BANK',parents=dict(source=source_hash,prepared=cache.foundation_sha256,recipe=target_recipe()['content_hash']),
        role='train',coordinate='D000',rows=len(cache),tokens=int(offsets[-1]),width=36,dtype='float32',
        alignment=alignment(cache),recipe=target_recipe(),
        outputs=[file_ref(root/name,root=root) for name in ('index.npz',*(f+'.npy' for f in FAMILIES))])
    write_immutable_json(root/'manifest.json',result)
    bank=NeighborhoodBank(root,cache,source_hash); bank.close()
    return result


def validate_index(m,index,cache,source_hash):
    validate(m,'BANK',dict(source=source_hash,prepared=cache.foundation_sha256,recipe=target_recipe()['content_hash']))
    if (cache.role!='train' or cache.coordinate_name!='D000' or m['recipe']!=target_recipe() or m['alignment']!=alignment(cache)
            or (m['role'],m['coordinate'],m['rows'],m['width'],m['dtype'])!=('train','D000',len(cache),36,'float32')
            or [r['path'] for r in m['outputs']]!=['index.npz',*(f+'.npy' for f in FAMILIES)]): raise ValueError('Neighborhood semantics differ')
    if set(index)!={'identities','labels','offsets',*(f+'_'+s for f in FAMILIES for s in ('mean','scale'))}: raise ValueError('Index inventory differs')
    offsets=index['offsets']
    if (index['identities'].dtype!=np.uint8 or index['labels'].dtype!=np.int64
            or not np.array_equal(index['identities'],cache.identities) or not np.array_equal(index['labels'],cache.labels)
            or offsets.dtype!=np.int64 or offsets.shape!=(len(cache)+1,) or offsets[0]!=0 or offsets[-1]!=m['tokens']
            or not np.array_equal(np.diff(offsets),counts(cache))): raise ValueError('Neighborhood join differs')
    for f in FAMILIES:
        for s in ('mean','scale'):
            a=index[f+'_'+s]
            if a.dtype!=np.float64 or a.shape!=(36,) or not np.isfinite(a).all(): raise ValueError('Invalid normalization')
        if np.any(index[f+'_scale']<.1): raise ValueError('Neighborhood scale floor differs')


class NeighborhoodBank:
    def __init__(self, root, cache, source_hash):
        self.root=Path(root); self.manifest=load_json(self.root/'manifest.json'); self.payload={}
        for ref in self.manifest['outputs']: validate_file_ref(ref,root=self.root)
        self.index=load_npz_arrays(self.root/'index.npz'); validate_index(self.manifest,self.index,cache,source_hash)
        try:
            for f in FAMILIES:
                a=np.load(self.root/(f+'.npy'),mmap_mode='r',allow_pickle=False); self.payload[f]=a
                if a.dtype!=np.float32 or a.shape!=(self.manifest['tokens'],36): raise ValueError('Neighborhood payload shape differs')
        except BaseException:
            self.close(); raise

    def batch(self,indices,width,kind):
        ix=np.asarray(indices); offsets=self.index['offsets']
        if (kind not in (*FAMILIES,'correction') or ix.dtype!=np.int64 or ix.ndim!=1 or not 1<=width<=512
                or np.any(ix<0) or np.any(ix>=len(offsets)-1)): raise ValueError('Invalid target request')
        result=np.zeros((len(ix),width,36),np.float32)
        for j,i in enumerate(ix):
            lo,hi=offsets[i:i+2]
            if hi-lo>width: raise ValueError('Student support truncated')
            if kind=='correction':
                z=(self.payload['ann'][lo:hi].astype(np.float64)-self.payload['hlt_ann'][lo:hi].astype(np.float64))/self.index['ann_scale']
            else: z=(self.payload[kind][lo:hi].astype(np.float64)-self.index[kind+'_mean'])/self.index[kind+'_scale']
            if not np.isfinite(z).all(): raise ValueError('Nonfinite target')
            result[j,:hi-lo]=z
        return result

    def close(self):
        for a in self.payload.values(): a._mmap.close()
        self.payload.clear()


class Bundle:
    def __init__(self,spec,cache,arm,neighborhood):
        if cache.role!='train' or cache.coordinate_name!='D000': raise PermissionError('Only TRAIN HLT joins')
        self.identities,self.labels=cache.identities,cache.labels
        self.targets={}; self.probabilities={}; self.neighborhood=neighborhood
        records={}
        for t in TEACHERS:
            ref=spec['imported']['banks'][t]; validate_file_ref(ref)
            record=spec['reuse']['teachers'][t]
            m,a=compact.load(Path(ref['path']).parent,cache,record)
            if m!=load_json(ref['path']): raise ValueError('Borrowed bank changed')
            records[t]=m; self.targets[t]=a['targets']; self.probabilities[t]=a['probabilities']
        self.manifest=artifact('BUNDLE',parents=dict(spec=spec['content_hash'],recipe=recipe(arm)['content_hash'],
            neighborhoods=neighborhood.manifest['content_hash'],**{t:m['content_hash'] for t,m in records.items()}),
            arm=arm,banks=records,neighborhoods=neighborhood.manifest)

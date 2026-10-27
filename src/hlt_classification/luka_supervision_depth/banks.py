"""Compact shared jet targets and independently owned multi-depth scratch banks."""
from pathlib import Path
import gc
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json,load_npz_arrays,write_immutable_json,atomic_publish_bytes,deterministic_npz_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_auxiliary_grid import banks as grid
from hlt_classification.luka_trial_hybrid import banks as old
from hlt_classification.luka_embedding_aux import targets as embedding
from hlt_classification.luka_attention.banks import counts,alignment
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from .contracts import artifact,validate,recipe,TEACHERS
from .model import tapped_forward


def record(spec,teacher):
    return grid.record(spec,teacher)


def export_shared(spec,teacher,model,cache,device):
    return grid.export(spec,teacher,model,cache,device=device)


class Bundle:
    def __init__(self,spec,cache,arm,token_bank=None):
        r=recipe(arm)
        if cache.role!='train' or cache.coordinate_name!='D000': raise PermissionError('Only HLT TRAIN consumes targets')
        self.identities,self.labels=cache.identities,cache.labels
        self.targets={}; self.probabilities={}; records={}; self.tokens=token_bank
        for t in TEACHERS:
            records[t],a=old.load(grid.directory(spec,t),cache,record(spec,t))
            self.targets[t]=a['targets']; self.probabilities[t]=a['probabilities']
        for t in sorted(set(r['target_names'].values())-set(TEACHERS)):
            folder=Path(spec['root'])/t
            ref=file_ref(folder/'fit/selected.pt')
            bank=embedding.Bank(folder/'bank',cache,teacher_ref=ref)
            records[t]=bank.manifest; self.targets[t]=bank.targets
        if bool(token_bank)!=bool(r['depths']): raise ValueError('Missing/unexpected token targets')
        if token_bank:
            if list(token_bank.depths)!=r['depths']: raise ValueError('Wrong depth bank')
            records['tokens']=token_bank.manifest
        self.manifest=artifact('BUNDLE',parents=dict(spec=spec['content_hash'],recipe=r['content_hash'],
            **{t:m['content_hash'] for t,m in records.items()}),arm=arm,records=records)


def export_tokens(folder,model,rich,hlt,teacher,depths,device):
    depths=tuple(depths)
    if (not depths or len(set(depths))!=len(depths) or any(d not in (2,4,8) for d in depths)
            or rich.role!='train' or hlt.role!='train' or rich.coordinate_name!='U100' or hlt.coordinate_name!='D000'
            or alignment(rich)!=alignment(hlt) or teacher['teacher']!='UE100' or teacher['scope']!='c50'):
        raise ValueError('Invalid TRAIN/c50 persistent-slot token target join')
    old.check_selected(model,teacher)
    root=Path(folder); root.mkdir(exist_ok=False)
    lengths=counts(rich); offsets=np.concatenate(([0],np.cumsum(lengths))).astype(np.int64)
    payloads={}; means={d:np.zeros(128) for d in depths}; m2={d:np.zeros(128) for d in depths}; seen=0
    try:
        payloads={d:np.lib.format.open_memmap(root/f'tokens{d}.npy',mode='w+',dtype=np.float32,shape=(int(offsets[-1]),128)) for d in depths}
        model.eval().requires_grad_(False)
        with parity_backend(device),torch.inference_mode():
            for start in range(0,len(rich),128):
                ix=np.arange(start,min(start+128,len(rich))); raw=rich.batch(ix)
                x,v,m=[torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')]
                _,_,taps=tapped_forward(model,x,v,m,depths)
                n=int(offsets[start+len(ix)]-offsets[start]); total=seen+n
                for d in depths:
                    real=taps[d][m[:,0]].float().cpu().numpy()
                    if real.shape!=(n,128) or not np.isfinite(real).all(): raise ValueError('Invalid real teacher tokens')
                    payloads[d][offsets[start]:offsets[start+len(ix)]]=real
                    values=real.astype(np.float64); avg=values.mean(0); delta=avg-means[d]
                    m2[d]+=((values-avg)**2).sum(0)+delta**2*seen*n/total
                    means[d]+=delta*n/total
                seen=total
                if start%(128*128)==0: print(f'DEPTH token export {start+len(ix)}/{len(rich)} depths={depths}',flush=True)
    finally:
        for payload in payloads.values():
            payload.flush(); payload._mmap.close()
        payloads.clear()
    old.check_selected(model,teacher)
    index=dict(identities=rich.identities,labels=rich.labels,offsets=offsets)
    for d in depths: index.update({f'mean{d}':means[d],f'scale{d}':np.maximum(np.sqrt(m2[d]/seen),1e-3)})
    atomic_publish_bytes(root/'index.npz',deterministic_npz_bytes(index))
    manifest=artifact('TOKEN_BANK',parents=dict(teacher=teacher['content_hash'],prepared=rich.foundation_sha256),
        teacher=teacher,depths=list(depths),alignment=alignment(rich),role='train',rows=len(rich),tokens=seen,width=128,dtype='float32',
        outputs=[file_ref(root/n,root=root) for n in ('index.npz',*(f'tokens{d}.npy' for d in depths))])
    write_immutable_json(root/'manifest.json',manifest)
    return manifest


def validate_token_index(m,index,cache,teacher,depths):
    validate(m,'TOKEN_BANK',dict(teacher=teacher['content_hash'],prepared=cache.foundation_sha256))
    if (m['teacher']!=teacher or m['depths']!=list(depths) or m['alignment']!=alignment(cache) or cache.role!='train'
            or cache.coordinate_name!='D000' or (m['role'],m['rows'],m['width'],m['dtype'])!=('train',len(cache),128,'float32')
            or [r['path'] for r in m['outputs']]!=['index.npz',*(f'tokens{d}.npy' for d in depths)]):
        raise ValueError('Token bank lineage/shape differs')
    expected={'identities','labels','offsets',*(f'mean{d}' for d in depths),*(f'scale{d}' for d in depths)}
    if set(index)!=expected: raise ValueError('Token index inventory differs')
    offsets=index['offsets']
    if (index['identities'].dtype!=np.uint8 or index['labels'].dtype!=np.int64
            or not np.array_equal(index['identities'],cache.identities) or not np.array_equal(index['labels'],cache.labels)
            or offsets.dtype!=np.int64 or offsets.shape!=(len(cache)+1,) or offsets[0]!=0 or offsets[-1]!=m['tokens']
            or not np.array_equal(np.diff(offsets),counts(cache))): raise ValueError('Token identity/support join differs')
    for d in depths:
        for name in (f'mean{d}',f'scale{d}'):
            a=index[name]
            if a.dtype!=np.float64 or a.shape!=(128,) or not np.isfinite(a).all(): raise ValueError('Invalid TRAIN moments')
        if np.any(index[f'scale{d}']<1e-3): raise ValueError('Invalid TRAIN scale')


class TokenBank:
    def __init__(self,root,cache,teacher,depths):
        root=Path(root); self.depths=tuple(depths); self.manifest=load_json(root/'manifest.json'); self.payload={}
        for ref in self.manifest['outputs']: validate_file_ref(ref,root=root)
        self.index=load_npz_arrays(root/'index.npz'); validate_token_index(self.manifest,self.index,cache,teacher,self.depths)
        try:
            for d in self.depths:
                a=np.load(root/f'tokens{d}.npy',mmap_mode='r',allow_pickle=False); self.payload[d]=a
                if a.dtype!=np.float32 or a.shape!=(self.manifest['tokens'],128): raise ValueError('Token payload differs')
        except BaseException:
            self.close(); raise

    def batch(self,indices,width):
        indices=np.asarray(indices); offsets=self.index['offsets']
        if indices.dtype!=np.int64 or indices.ndim!=1 or np.any(indices<0) or np.any(indices>=len(offsets)-1) or not 1<=width<=512:
            raise ValueError('Invalid token request')
        result={d:np.zeros((len(indices),width,128),np.float32) for d in self.depths}
        for row,i in enumerate(indices):
            lo,hi=offsets[i:i+2]
            if hi-lo>width: raise ValueError('Student truncated teacher support')
            for d in self.depths:
                values=(self.payload[d][lo:hi].astype(np.float64)-self.index[f'mean{d}'])/self.index[f'scale{d}']
                if not np.isfinite(values).all(): raise ValueError('Nonfinite real target')
                result[d][row,:hi-lo]=values.astype(np.float32)
        return result

    def close(self):
        for a in self.payload.values(): a._mmap.close()
        self.payload.clear(); gc.collect()

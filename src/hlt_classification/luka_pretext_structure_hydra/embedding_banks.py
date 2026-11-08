"""Selected clean OFFLINE embeddings; each teacher keeps its own coordinates."""
from pathlib import Path
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from .contracts import artifact, validate, TEACHERS
from .target_banks import open_staging, finish_memmap, publish_array
from .corruption import batch
from .normalization import Moments
from hlt_classification.luka_capacity_hydra.model import Native
from hlt_classification.luka_embedding_aux.model import head_input


def export(folder,model,physical,spec,task,teacher_ref,*,device,technical=False):
    if physical.role!='train' or physical.source!='OFFLINE' or task not in TEACHERS:
        raise PermissionError('Only clean OFFLINE TRAIN may produce targets')
    validate_file_ref(teacher_ref); folder=Path(folder); folder.mkdir(parents=True,exist_ok=False)
    model.eval()
    with torch.random.fork_rng(devices=[]): native=Native(128).to(device).eval()
    native.load_state_dict(torch.load(teacher_ref['path'],map_location='cpu',weights_only=True),strict=True)
    path=folder/'targets.npy'; out=open_staging(path,np.float32,(len(physical),128)); moments=Moments(128)
    with torch.inference_mode():
        for start in range(0,len(physical),128):
            stop=min(start+128,len(physical)); raw=batch([physical.jet(i) for i in range(start,stop)])
            tensors={k:torch.as_tensor(v,device=device) for k,v in raw.items()}
            _,h=head_input(native,**tensors)
            if start==0:
                torch.testing.assert_close(h,model.embedding(raw,device,bf16=False),rtol=0,atol=0)
            value=h.float().cpu().numpy().astype(np.float32)
            if not np.isfinite(value).all(): raise ValueError('Nonfinite clean embedding')
            out[start:stop]=value; moments.add(value,np.ones_like(value,bool))
    mean=moments.mean; scale=np.maximum(np.sqrt(moments.m2/np.maximum(moments.count,1)),.001)
    # Rank/variance diagnostics are descriptive, never quality admission gates.
    sample=np.array(out[:min(8192,len(out))],np.float64); centred=sample-sample.mean(0)
    singular=np.linalg.svd(centred,compute_uv=False)
    energy=singular**2; total=energy.sum(); nonzero=energy[energy>0]
    effective_rank=float(np.exp(-np.sum((nonzero/total)*np.log(nonzero/total)))) if total>0 else 0.
    for start in range(0,len(out),4096): out[start:start+4096]=(out[start:start+4096].astype(np.float64)-mean)/scale
    finish_memmap(path,out); del out
    for k,a in dict(identities=physical.identities,labels=physical.labels,mean=mean,scale=scale).items(): publish_array(folder/(k+'.npy'),a)
    value=artifact('EMBEDDING_BANK',parents=dict(spec=spec['content_hash'],physical=physical.manifest['content_hash'],teacher=teacher_ref['sha256']),
        teacher=teacher_ref,task=task,role='train',coordinate='PRETEXT_'+task,technical=technical,rows=len(physical),width=128,
        clean_variance=(moments.m2/np.maximum(moments.count,1)).tolist(),singular_values=singular.tolist(),
        effective_rank=effective_rank,rank_sample_rows=len(sample),rank_sample='first_ordered_TRAIN_rows',native_reload=True,
        files=[file_ref(folder/(k+'.npy'),root=folder) for k in ('targets','identities','labels','mean','scale')])
    validate_file_ref(teacher_ref); write_immutable_json(folder/'manifest.json',value)
    return value


class Bank:
    def __init__(self,folder,train,spec,task,teacher_ref,*,technical=False):
        folder=Path(folder); self.manifest=load_json(folder/'manifest.json'); m=self.manifest; validate(m,'EMBEDDING_BANK')
        if (train.role!='train' or task not in TEACHERS or m['task']!=task or m['teacher']!=teacher_ref or m['technical']!=technical
                or m['parents']['teacher']!=teacher_ref['sha256'] or m['parents']['spec']!=spec['content_hash']
                or m['role']!='train' or m['coordinate']!='PRETEXT_'+task or m['rows']!=len(train) or m['width']!=128
                or [r['path'] for r in m['files']]!=[k+'.npy' for k in ('targets','identities','labels','mean','scale')]):
            raise ValueError('Selected clean embedding bank meaning differs')
        validate_file_ref(teacher_ref)
        for ref in m['files']: validate_file_ref(ref,root=folder)
        for k in ('targets','identities','labels','mean','scale'): setattr(self,k,np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False))
        if (self.targets.shape!=(len(train),128) or self.targets.dtype!=np.float32
                or any(not np.array_equal(getattr(self,k),getattr(train,k)) for k in ('identities','labels'))
                or self.mean.shape!=(128,) or self.scale.shape!=(128,) or np.any(self.scale<.001)
                or not all(np.isfinite(getattr(self,k)).all() for k in ('targets','mean','scale'))):
            raise ValueError('Embedding normalization/identity join differs')

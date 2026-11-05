"""Authenticated TRAIN-only HLT-slot targets; dense arrays stay in allocation RAM."""
from contextlib import nullcontext
import gc
import hashlib
import time
import numpy as np
import torch
from hlt_classification.luka_shared_message_hydra.inputs import load_hlt,bank,frozen_teacher
from hlt_classification.luka_cross_message_hydra import inputs as donor
from hlt_classification.luka_particle_prediction.model import token_forward
from hlt_classification.data.cache_contracts import load_json
from .contracts import artifact,validate,recipe


def array_hash(array):
    if not array.flags.c_contiguous: raise ValueError('Target array must be contiguous')
    digest=hashlib.sha256(); view=memoryview(array).cast('B')
    for start in range(0,len(view),8*2**20): digest.update(view[start:start+8*2**20])
    return digest.hexdigest()


def parents(spec):
    return dict(spec=spec['content_hash'],teacher=spec['teacher_proof']['report_hash'],
        encoder=spec['encoder_hash'],prepared=spec['anchor']['evidence']['prepared'])


def validate_metadata(value,spec):
    validate(value,'TARGETS',parents(spec))
    if (value['rows']!=spec['counts']['train'] or value['depths']!=[6,8] or value['channels']!=128
            or value['role']!='train' or value['coordinate']!='D000' or value['batch_size']!=128
            or value['normalization']!=recipe()['normalization'] or value['tokens']<value['rows']
            or set(value['statistics'])!={'6','8'} or set(value['array_hashes'])!={'6','8'}):
        raise ValueError('Teacher target registration differs')
    for stats in value['statistics'].values():
        mean=np.asarray(stats['mean']); scale=np.asarray(stats['scale'])
        if mean.shape!=(128,) or scale.shape!=(128,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<recipe()['std_floor']).any():
            raise ValueError('Invalid frozen TRAIN normalization')
    return value


class Targets:
    def __init__(self,hlt,offsets,arrays,metadata,seconds=0.):
        self.identities=hlt.identities; self.labels=hlt.labels; self.role=hlt.role
        self.offsets=offsets; self.arrays=arrays; self.metadata=metadata; self.build_seconds=seconds
        self.nbytes=offsets.nbytes+sum(a.nbytes for a in arrays.values())
        self.statistics={int(d):(np.asarray(s['mean'],np.float32),np.asarray(s['scale'],np.float32))
            for d,s in metadata['statistics'].items()}

    def batch(self,indices,mask,depths):
        if self.role!='train' or mask.dtype!=np.bool_ or mask.ndim!=2 or len(indices)!=len(mask):
            raise ValueError('Invalid target batch')
        result={d:np.zeros((*mask.shape,128),np.float32) for d in depths}
        for j,i in enumerate(indices):
            if not 0<=i<len(self.identities): raise IndexError(i)
            a,b=self.offsets[i:i+2]
            if b-a!=mask[j].sum(): raise ValueError('Target/HLT slots differ')
            for d,value in result.items():
                mean,scale=self.statistics[d]
                value[j,mask[j]]=(self.arrays[d][a:b]-mean)/scale
        return result


class TargetSubset:
    def __init__(self,source,indices):
        self.source=source; self.indices=np.asarray(indices,dtype=np.int64)
        self.identities=source.identities[self.indices]; self.labels=source.labels[self.indices]
        self.role=source.role; self.metadata=source.metadata

    def batch(self,indices,mask,depths):
        return self.source.batch(self.indices[indices],mask,depths)


def build_targets(spec,hlt,u100,encoder,teacher,device='cuda',*,budget_bytes):
    if hlt.role!='train' or u100.role!='train': raise PermissionError('Teacher targets are TRAIN-only')
    donor.aligned(hlt,u100)
    if (len(hlt)!=spec['counts']['train'] or hlt.foundation_sha256!=spec['anchor']['evidence']['prepared']
            or any(p.requires_grad for model in (encoder,teacher) for p in model.parameters())
            or getattr(teacher,'mode',None)!='teacher' or getattr(teacher,'width',None)!=16
            or not teacher.message_enabled):
        raise ValueError('Need authenticated population and frozen active m16 oracle')
    started=time.perf_counter()
    offsets=np.concatenate(([0],np.cumsum(np.concatenate([np.diff(b.offsets) for b in hlt.blocks]),dtype=np.int64)))
    required=int(offsets[-1])*128*4*2+offsets.nbytes
    # Reserve batch/normalization workspace before allocating dense targets.
    if required+hlt.nbytes+u100.nbytes+512*2**20>budget_bytes:
        raise MemoryError(f'Two-depth TRAIN targets plus views exceed RAM budget: {required} target bytes')
    arrays={d:np.empty((offsets[-1],128),np.float32) for d in (6,8)}
    sums={d:np.zeros(128,np.float64) for d in arrays}; squares={d:np.zeros(128,np.float64) for d in arrays}
    encoder.eval(); teacher.eval(); captured={}; handles=[]
    def hook(depth):
        def capture(_m,_a,out):
            if depth in captured: raise ValueError('Teacher particle tap fired twice')
            captured[depth]=out
        return capture
    try:
        for d in arrays: handles.append(teacher.mod.blocks[d-1].register_forward_hook(hook(d)))
        with torch.no_grad():
            for start in range(0,len(hlt),128):
                ix=np.arange(start,min(start+128,len(hlt))); h=hlt.batch(ix); u=u100.batch(ix)
                if any(not np.array_equal(h[k],u[k]) for k in ('identities','labels','mask')):
                    raise ValueError('Oracle batch identities or slots differ')
                def tensors(raw): return {k:torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')}
                captured.clear(); ctx=torch.autocast('cuda',dtype=torch.bfloat16) if str(device).startswith('cuda') else nullcontext()
                with ctx:
                    _,memory=token_forward(encoder,**tensors(u))
                    logits=teacher.oracle(**tensors(h),u_tokens=memory.detach())
                mask=h['mask'][:,0]
                if set(captured)!={6,8} or logits.shape!=(len(ix),11) or not torch.isfinite(logits).all():
                    raise ValueError('Oracle forward or taps differ')
                for d,out in captured.items():
                    if out.shape!=(*mask.shape,128): raise ValueError('Teacher discarded HLT particle slots')
                    values=out.float().cpu().numpy()[mask]
                    if not np.isfinite(values).all(): raise ValueError('Nonfinite real teacher token')
                    arrays[d][offsets[ix[0]]:offsets[ix[-1]+1]]=values
                    v=values.astype(np.float64); sums[d]+=v.sum(0); squares[d]+=(v*v).sum(0)
                if start%(128*100)==0 or ix[-1]==len(hlt)-1:
                    print(f'PARTICLE_TEACHER targets: {ix[-1]+1}/{len(hlt)} TRAIN jets',flush=True)
    finally:
        for handle in handles: handle.remove()
    statistics={}
    for d in arrays:
        mean=sums[d]/offsets[-1]; variance=np.maximum(squares[d]/offsets[-1]-mean*mean,0.)
        statistics[str(d)]=dict(mean=mean.tolist(),scale=np.maximum(np.sqrt(variance),recipe()['std_floor']).tolist())
    value=artifact('TARGETS',parents=parents(spec),role='train',coordinate='D000',rows=len(hlt),
        tokens=int(offsets[-1]),channels=128,depths=[6,8],batch_size=128,normalization=recipe()['normalization'],
        identities_hash=array_hash(hlt.identities),offsets_hash=array_hash(offsets),
        array_hashes={str(d):array_hash(a) for d,a in arrays.items()},statistics=statistics)
    validate_metadata(value,spec)
    return Targets(hlt,offsets,arrays,value,time.perf_counter()-started)


def targets(spec,train,device='cuda'):
    if train.role!='train': raise PermissionError('Teacher targets are TRAIN-only')
    u=donor.load_view(spec,'train','U100'); encoder=donor.encoder(spec,device); teacher=frozen_teacher(spec,device)
    result=build_targets(spec,train,u,encoder,teacher,device,budget_bytes=int(spec['memory_mb']*1024**2*.75))
    del teacher,encoder,u; gc.collect()
    if str(device).startswith('cuda'): torch.cuda.empty_cache()
    return result


def registered_metadata(spec):
    from pathlib import Path
    return validate_metadata(load_json(Path(spec['root'])/'setup/targets.json'),spec)

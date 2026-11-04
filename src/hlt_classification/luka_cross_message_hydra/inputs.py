"""Minimal U100 encoder transfer and bounded allocation-local token caching."""
from contextlib import nullcontext
from pathlib import Path
import os
import shutil
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_u100_pair_transfer import bundle as transfer
from hlt_classification.luka_frozen_adapter_hydra.inputs import (
    Bundle,Delta,delta,load_view,metadata,embedding_bank,oracle_validation,aligned)
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.luka_particle_prediction.model import token_forward
from .contracts import artifact,validate,COUNTS,MANIFEST_HASH


def teacher_files(origin):
    teacher=origin['imported']['teacher']
    transfer.checked(teacher)
    if (teacher['coordinate']!='U100' or teacher['node']['node_id']!='COARSE_U100'
            or teacher['node'].get('context_coordinate') is not None
            or teacher['prepared']!=origin['anchor']['evidence']['prepared']):
        raise ValueError('Expected original selected single-encoder COARSE_U100')
    return [dict(teacher['files'][key],path=name) for key,name in
        (('checkpoint','U100.pt'),('predictions','validation_predictions.npz'))]


def origin_proof(old):
    anchor=old['anchor']['evidence']; ref=old['reference']
    return dict(campaign=old['content_hash'],reference=ref['content_hash'],prepared=anchor['prepared'],
        cache=anchor['cache'],source_commit=old['source_commit'],control_job='21993056',counts=COUNTS,
        environment=ref['environment'],original_source_replay=True)


def export(original_spec,output):
    path=transfer.regular(original_spec); old=transfer.checked(load_json(path))
    if old['contract']!='LUKA_U100_PAIR_CORRECTION_SPEC/v1' or old['counts']!=COUNTS:
        raise ValueError('Use the original SPORC pair campaign')
    proof=transfer.authenticate(path,old)
    if proof!=origin_proof(old): raise ValueError('Original-source replay differs')
    rows=teacher_files(old); teacher=old['imported']['teacher']
    sources=[transfer.regular(teacher['files'][k]['path']) for k in ('checkpoint','predictions')]
    for row,src in zip(rows,sources):
        actual=file_ref(src)
        if any(actual[k]!=row[k] for k in ('sha256','bytes')): raise ValueError('U100 source bytes changed')
    total=sum(r['bytes'] for r in rows); output=Path(output).absolute()
    if (output.exists() or not output.parent.is_dir() or any(p.is_symlink() for p in (output,*output.parents))
            or any(output.is_relative_to(p.parent) or p.is_relative_to(output) for p in sources)
            or not 0<total<=128*2**20): raise ValueError('Use a fresh, separate small encoder delta')
    if shutil.disk_usage(output.parent).free<total+2**30: raise OSError('Need encoder bytes plus one GiB headroom')
    output.mkdir()
    for row,src in zip(rows,sources):
        with src.open('rb') as source,(output/row['path']).open('xb') as dest:
            shutil.copyfileobj(source,dest,2**20); dest.flush(); os.fsync(dest.fileno())
        validate_file_ref(row,root=output)
    manifest=artifact('ENCODER_TRANSFER',parents=dict(origin=old['content_hash'],transfer=MANIFEST_HASH,
        teacher=teacher['content_hash']),original_replay=proof,files=rows,payload_bytes=total,
        roles=['frozen_U100_encoder','U100_validation_replay'])
    write_immutable_json(output/'manifest.json',manifest)
    return dict(directory=str(output),manifest_hash=manifest['content_hash'],payload_bytes=total,files=len(rows))


class EncoderDelta:
    def __init__(self,root,expected_hash,bundle):
        self.root=Path(root).resolve(strict=True); old=bundle.json('origin/campaign_spec.json')
        value=load_json(transfer.regular(self.root/'manifest.json'))
        validate(value,'ENCODER_TRANSFER',dict(origin=old['content_hash'],transfer=MANIFEST_HASH,
            teacher=old['imported']['teacher']['content_hash']))
        if (value['content_hash']!=expected_hash or value['files']!=teacher_files(old)
                or value['original_replay']!=origin_proof(old)
                or value['payload_bytes']!=sum(r['bytes'] for r in value['files'])
                or not 0<value['payload_bytes']<=128*2**20
                or value['roles']!=['frozen_U100_encoder','U100_validation_replay']):
            raise ValueError('Wrong U100 encoder transfer')
        self.manifest=value; self.rows={r['path']:r for r in value['files']}

    def path(self,name):
        transfer.relative(name)
        if name not in self.rows: raise PermissionError('Not an encoder delta member')
        p=transfer.regular(self.root/name); validate_file_ref(self.rows[name],root=self.root)
        return p

    def verify_all(self):
        for name in self.rows: self.path(name)


def encoder_delta(spec):
    return EncoderDelta(spec['encoder_root'],spec['encoder_hash'],Bundle(spec['bundle_root']))


def encoder(spec,device):
    state=torch.load(encoder_delta(spec).path('U100.pt'),map_location='cpu',weights_only=True)
    model=DelphesParticleTransformer(); model.load_state_dict(state,strict=True)
    return model.to(device).eval().requires_grad_(False)


class TokenCache:
    """An explicit privileged wrapper; HLT student forwards reject its extra fields."""
    def __init__(self,hlt,offsets,tokens,logits):
        self.hlt,self.offsets,self.tokens,self.logits=hlt,offsets,tokens,logits
        if (offsets.dtype!=np.int64 or offsets.shape!=(len(hlt)+1,) or offsets[0]!=0
                or (np.diff(offsets)<1).any() or tokens.shape!=(offsets[-1],128)
                or tokens.dtype!=np.float32 or logits.shape!=(len(hlt),11) or logits.dtype!=np.float32
                or not np.isfinite(tokens).all() or not np.isfinite(logits).all()):
            raise ValueError('Invalid frozen feature cache')
        self.nbytes=hlt.nbytes+offsets.nbytes+tokens.nbytes+logits.nbytes

    def __len__(self): return len(self.hlt)
    def __getattr__(self,key): return getattr(self.hlt,key)

    def batch(self,indices):
        indices=np.asarray(indices,dtype=np.int64); raw=self.hlt.batch(indices)
        memory=np.zeros((len(indices),raw['features'].shape[-1],128),np.float32)
        for j,i in enumerate(indices):
            valid=raw['mask'][j,0]; a,b=self.offsets[i:i+2]
            if valid.sum()!=b-a: raise ValueError('HLT/U100 token support differs')
            memory[j,valid]=self.tokens[a:b]
        return dict(raw,u_tokens=memory,u_logits=self.logits[indices])


def build_cache(hlt,u100,source,device,*,budget_bytes):
    aligned(hlt,u100)
    counts=np.concatenate([np.diff(b.offsets) for b in hlt.blocks])
    offsets=np.concatenate(([0],np.cumsum(counts,dtype=np.int64)))
    required=int(offsets[-1])*128*4+len(hlt)*11*4+offsets.nbytes
    if required+hlt.nbytes+u100.nbytes+128*512*128*4>budget_bytes:
        raise MemoryError('Frozen-token cache exceeds registered RAM allowance')
    tokens=np.empty((offsets[-1],128),np.float32); logits=np.empty((len(hlt),11),np.float32)
    source.eval().requires_grad_(False); started=time.perf_counter()
    with torch.no_grad():
        for start in range(0,len(hlt),128):
            ix=np.arange(start,min(start+128,len(hlt))); raw=u100.batch(ix)
            data={k:torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')}
            ctx=torch.autocast('cuda',dtype=torch.bfloat16) if str(device).startswith('cuda') else nullcontext()
            with ctx: z,h=token_forward(source,**data)
            h=h.float().cpu().numpy(); z=z.float().cpu().numpy()
            valid=raw['mask'][:,0]
            if not np.isfinite(h[valid]).all() or not np.isfinite(z).all(): raise ValueError('Nonfinite U100 features')
            for j,i in enumerate(ix): tokens[offsets[i]:offsets[i+1]]=h[j,valid[j]]
            logits[ix]=z
            if start%(128*100)==0 or ix[-1]==len(hlt)-1:
                print(f'CROSS_MESSAGE frozen {hlt.role} U100 tokens: {ix[-1]+1}/{len(hlt)}',flush=True)
    cache=TokenCache(hlt,offsets,tokens,logits)
    cache.build_seconds=time.perf_counter()-started
    return cache


def oracle_cache(spec,hlt,device='cuda'):
    if hlt.role not in ('train','validation'): raise PermissionError('No test feature extraction')
    u=load_view(spec,'train','U100') if hlt.role=='train' else oracle_validation(spec)
    source=encoder(spec,device)
    result=build_cache(hlt,u,source,device,budget_bytes=int(spec['memory_mb']*1024**2*.35))
    del source,u
    if str(device).startswith('cuda'): torch.cuda.empty_cache()
    return result

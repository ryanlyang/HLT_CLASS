"""Allowlisted original views/banks plus the additive TRAIN measurement bundle."""
from pathlib import Path
import numpy as np
import torch
from hlt_classification.luka_u100_pair_hydra.inputs import Bundle, load_view, embedding_bank
from hlt_classification.luka_u100_contrastive_hydra.inputs import key_bank
from hlt_classification.luka_u100_contrastive.targets import summaries, normalized_keys
from hlt_classification.luka_embedding_aux.model import head_input
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.data.cache_contracts import (load_json,load_npz_arrays,write_immutable_json,
    atomic_publish_bytes,deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from .contracts import artifact,validate,setting
from .data import RawBundle
from .sampling import MembershipSampler,JetCandidates


def hlt(spec,role):
    if role not in ('train','validation'): raise PermissionError('No test capability')
    return load_view(spec,role,'D000')


def raw(spec):
    result=RawBundle(spec['raw_root'],spec['raw_hash'])
    bundle=Bundle(spec['bundle_root']); original=bundle.json('origin/campaign_spec.json')
    prepared=bundle.json('prepared/prepared.json')
    from hlt_classification.luka_cross_message_hydra.inputs import origin_proof
    if (result.manifest['parents']['prepared']!=spec['anchor']['evidence']['prepared']
            or result.manifest['parents']['origin']!=original['content_hash']
            or result.manifest['origin_proof']!=origin_proof(original)
            or result.manifest['conventions']!=prepared['conventions']
            or any(result.rows['M0HLT.pt'][k]!=result.manifest['original_m0'][k] for k in ('bytes','sha256'))):
        raise ValueError('Raw source prepared parent differs')
    return result


def build_hlt_keys(spec,train,device):
    if (train.role!='train' or train.coordinate_name!='D000' or len(train)!=spec['counts']['train']
            or train.foundation_sha256!=spec['anchor']['evidence']['prepared']):
        raise PermissionError('HLT matching keys are full TRAIN only')
    b=raw(spec); model=DelphesParticleTransformer().to(device)
    model.load_state_dict(torch.load(b.path('M0HLT.pt'),map_location='cpu',weights_only=True),strict=True)
    model.eval().requires_grad_(False); values=np.empty((len(train),128),np.float32)
    with torch.inference_mode(),torch.autocast(torch.device(device).type,enabled=False):
        for start in range(0,len(train),128):
            indices=np.arange(start,min(start+128,len(train))); data=train.batch(indices)
            _,h=head_input(model,**{k:torch.from_numpy(data[k]).to(device) for k in ('features','vectors','mask')})
            values[indices]=h.float().cpu().numpy()
    if not np.isfinite(values).all(): raise ValueError('Nonfinite frozen HLT key')
    keys,mean,scale=normalized_keys(values); folder=Path(spec['root'])/'setup'
    atomic_publish_bytes(folder/'hlt_keys.npz',deterministic_npz_bytes(dict(identities=train.identities,
        labels=train.labels,keys=keys,mean=mean,scale=scale)))
    record=artifact('HLT_KEYS',parents=dict(spec=spec['content_hash'],raw=spec['raw_hash']),
        role='train',rows=len(train),width=128,normalization='TRAIN_population_std_floor1e-3_then_L2_eps1e-12',
        teacher=b.manifest['original_m0'],payload=file_ref(folder/'hlt_keys.npz',root=folder))
    write_immutable_json(folder/'hlt_keys.json',record)
    return record


def hlt_keys(spec,train):
    if (train.role!='train' or train.coordinate_name!='D000' or len(train)!=spec['counts']['train']
            or train.foundation_sha256!=spec['anchor']['evidence']['prepared']):
        raise PermissionError('HLT matching keys are full TRAIN only')
    folder=Path(spec['root'])/'setup'; record=load_json(folder/'hlt_keys.json')
    validate(record,'HLT_KEYS',dict(spec=spec['content_hash'],raw=spec['raw_hash']))
    if (record['role']!='train' or record['rows']!=len(train) or record['width']!=128
            or record['normalization']!='TRAIN_population_std_floor1e-3_then_L2_eps1e-12'
            or record['teacher']!=raw(spec).manifest['original_m0'] or record['payload']['path']!='hlt_keys.npz'):
        raise ValueError('Wrong HLT-key teacher or role')
    validate_file_ref(record['payload'],root=folder); a=load_npz_arrays(folder/'hlt_keys.npz')
    if (set(a)!={'identities','labels','keys','mean','scale'}
            or any(not np.array_equal(a[k],getattr(train,k)) for k in ('identities','labels'))
            or a['keys'].shape!=(len(train),128) or a['keys'].dtype!=np.float32
            or any(a[k].shape!=(128,) or a[k].dtype!=np.float64 for k in ('mean','scale'))
            or any(not np.isfinite(v).all() for v in a.values()) or (a['scale']<.001).any()):
        raise ValueError('HLT-key payload or join differs')
    norms=np.linalg.norm(a['keys'],axis=-1)
    if not np.all((norms<1e-8)|np.isclose(norms,1.,atol=2e-6)): raise ValueError('Non-unit frozen HLT key')
    return a['keys']


def candidates(spec,task,train,device='cpu'):
    cfg=setting(task)
    if train.role!='train': raise PermissionError('Candidates are TRAIN-only')
    if cfg['family']=='C': return None
    pt,count=summaries(train)
    if cfg['family']=='J':
        keys=key_bank(spec,train)
        return JetCandidates(keys.keys,train.identities,train.labels,pt,count,
            source_keys=hlt_keys(spec,train) if cfg['source']=='HLT' else None,
            replicate=cfg['seed_index'],device=device)
    source=raw(spec).load(cfg['source'],train)
    # Membership consumes the authenticated local/standardized descriptors.
    # Retaining a second physical TRAIN collection in every fork is unnecessary.
    source.values=None
    result=MembershipSampler(source,pt,count,replicate=cfg['seed_index'])
    result.capacity=result.capacity_proof(cfg['family'])
    record=Path(spec['root'])/'setup'/('normalization_'+cfg['source']+'.json')
    if record.exists() and load_json(record)!=source.normalization: raise ValueError('TRAIN descriptor normalization changed')
    import os
    if os.name=='posix':
        from .parallel import ParallelCandidates
        return ParallelCandidates(result,spec['candidate_workers'])
    return result


class SubsetCandidates:
    def __init__(self,source,indices): self.source=source; self.indices=np.asarray(indices,np.int64)
    def batch(self,indices,epoch,cfg,mine=None): return self.source.batch(self.indices[indices],epoch,cfg,mine)

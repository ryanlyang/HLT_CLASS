"""Evaluation-only validation embeddings with frozen original TRAIN normalization."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json,load_npz_arrays,write_immutable_json,
    atomic_publish_bytes,deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_u100_pair_hydra.inputs import Bundle,load_view,embedding_bank
from hlt_classification.luka_cross_message_hydra import inputs as donor
from hlt_classification.luka_embedding_aux.model import head_input
from .contracts import artifact,validate


def hlt(spec,role):
    if role not in ('train','validation'): raise PermissionError('No test access')
    return load_view(spec,role,'D000')


def validate_sources(spec):
    tc=spec['teacher_context']; bundle=Bundle(spec['bundle_root'])
    if tc['bundle_root']!=spec['bundle_root'] or tc['counts']!=spec['counts']: raise ValueError('Teacher population differs')
    donor.delta(tc); enc=donor.encoder_delta(tc); enc.path('U100.pt')
    manifest=bundle.json('targets/u100/manifest.json')
    if (manifest['content_hash']!=spec['embedding_hash'] or manifest['parents']['teacher']!=spec['imported']['teacher']['content_hash']
            or manifest['parents']['prepared']!=spec['anchor']['evidence']['prepared']): raise ValueError('TRAIN target lineage differs')


def moments(spec):
    a=load_npz_arrays(Bundle(spec['bundle_root']).path('targets/u100/targets.npz'))
    mean,scale=a['mean'],a['scale']
    if (mean.shape!=(128,) or scale.shape!=(128,) or mean.dtype!=np.float64 or scale.dtype!=np.float64
            or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<.001).any()): raise ValueError('Invalid TRAIN moments')
    return mean,scale


def parents(spec):
    return dict(spec=spec['content_hash'],embedding=spec['embedding_hash'],
        teacher=spec['imported']['teacher']['content_hash'],encoder=spec['teacher_context']['encoder_hash'],
        validation_source=spec['teacher_context']['delta_hash'],prepared=spec['anchor']['evidence']['prepared'])


def extract(teacher,u100,hlt_validation,mean,scale,device):
    if u100.role!='validation' or hlt_validation.role!='validation': raise PermissionError('Evaluation targets require validation')
    donor.aligned(hlt_validation,u100)
    if any(p.requires_grad for p in teacher.parameters()) or teacher.training: raise ValueError('Teacher must be frozen in eval mode')
    values=np.empty((len(u100),128),np.float32)
    with torch.inference_mode(),torch.autocast(torch.device(device).type,enabled=False):
        for start in range(0,len(u100),128):
            ix=np.arange(start,min(start+128,len(u100))); raw=u100.batch(ix)
            _,z=head_input(teacher,**{k:torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')})
            v=z.float().cpu().numpy()
            # Match the original TRAIN standardize implementation's FP64 arithmetic.
            values[ix]=((v.astype(np.float64)-mean)/scale).astype(np.float32)
            if start%(128*100)==0 or ix[-1]==len(u100)-1:
                print(f'EMBEDDING_OBJECTIVE validation targets: {ix[-1]+1}/{len(u100)}',flush=True)
    if not np.isfinite(values).all(): raise ValueError('Nonfinite validation embeddings')
    return values


def build(spec,val,device='cuda'):
    tc=spec['teacher_context']; u=donor.oracle_validation(tc); model=donor.encoder(tc,device)
    mean,scale=moments(spec); targets=extract(model,u,val,mean,scale,device)
    folder=Path(spec['root'])/'setup'; path=folder/'validation_embeddings.npz'
    atomic_publish_bytes(path,deterministic_npz_bytes(dict(identities=val.identities,labels=val.labels,
        targets=targets,mean=mean,scale=scale)))
    value=artifact('VALIDATION_BANK',parents=parents(spec),role='validation',usage='evaluation_only',coordinate='U100',
        rows=len(val),width=128,precision='FP32',batch_size=128,statistics='original_TRAIN_frozen',
        payload=file_ref(path,root=folder))
    write_immutable_json(folder/'validation_embeddings.json',value)
    del model,u
    if str(device).startswith('cuda'): torch.cuda.empty_cache()
    return validation_bank(spec,val)


def validation_bank(spec,val):
    if val.role!='validation' or val.coordinate_name!='D000' or val.foundation_sha256!=spec['anchor']['evidence']['prepared']:
        raise PermissionError('Validation targets are not optimization targets')
    folder=Path(spec['root'])/'setup'; m=load_json(folder/'validation_embeddings.json'); validate(m,'VALIDATION_BANK',parents(spec))
    if (m['role']!='validation' or m['usage']!='evaluation_only' or m['coordinate']!='U100' or m['rows']!=len(val)
            or len(val)!=spec['counts']['validation'] or m['width']!=128 or m['precision']!='FP32'
            or m['batch_size']!=128 or m['statistics']!='original_TRAIN_frozen' or m['payload']['path']!='validation_embeddings.npz'):
        raise ValueError('Invalid validation bank contract')
    validate_file_ref(m['payload'],root=folder); a=load_npz_arrays(folder/m['payload']['path']); mean,scale=moments(spec)
    if (set(a)!={'identities','labels','targets','mean','scale'} or a['targets'].shape!=(len(val),128)
            or a['targets'].dtype!=np.float32 or not np.isfinite(a['targets']).all()
            or a['mean'].dtype!=np.float64 or a['scale'].dtype!=np.float64
            or any(not np.array_equal(a[k],getattr(val,k)) for k in ('identities','labels'))
            or not np.array_equal(a['mean'],mean) or not np.array_equal(a['scale'],scale)):
        raise ValueError('Validation join or frozen TRAIN moments differ')
    return SimpleNamespace(**a,manifest=m,role='validation')


def train_bank(spec,train):
    result=embedding_bank(spec,train); result.role='train'
    if result.manifest['content_hash']!=spec['embedding_hash']: raise ValueError('TRAIN bank differs')
    return result


def join(view, bank, role):
    if (role not in ('train', 'validation') or view.role != role or bank.manifest['role'] != role
            or view.coordinate_name != 'D000' or bank.manifest['coordinate'] != 'U100'
            or any(not np.array_equal(getattr(view, k), getattr(bank, k)) for k in ('identities', 'labels'))
            or bank.targets.shape != (len(view), 128) or bank.targets.dtype != np.float32
            or not np.isfinite(bank.targets).all()):
        raise PermissionError('Embedding role/coordinate/ordered identity join differs')


def views(spec, task):
    from .contracts import cell
    cell(task)
    train, val = hlt(spec, 'train'), hlt(spec, 'validation')
    reference = load_npz_arrays(Bundle(spec['bundle_root']).path('reference/matched/predictions.npz'))
    if any(not np.array_equal(reference[k], getattr(val, k)) for k in ('identities', 'labels')):
        raise ValueError('Validation population differs')
    return train, val


def banks(spec, task, train):
    from .contracts import cell
    cell(task)
    return {'u100': train_bank(spec, train)}


def fit_parents(spec, task):
    from .contracts import cell
    from . import geometry
    cell(task)
    folder = Path(spec['root'])/'setup'
    g = load_json(folder/'geometry.json'); v = load_json(folder/'validation_embeddings.json')
    validate(g, 'GEOMETRY', geometry.parents(spec)); validate(v, 'VALIDATION_BANK', parents(spec))
    validate_file_ref(g['payload'], root=folder); validate_file_ref(v['payload'], root=folder)
    return dict(spec=spec['content_hash'], recipe=spec['recipe']['content_hash'], u100=spec['embedding_hash'],
                geometry=g['content_hash'], validation_bank=v['content_hash'])

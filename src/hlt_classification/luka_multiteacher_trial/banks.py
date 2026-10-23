"""Compact independent TRAIN targets, shared by all five fits."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    atomic_publish_bytes, deterministic_npz_bytes, write_immutable_json)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_embedding_aux.targets import standardize, offline_train
from hlt_classification.luka_embedding_aux.model import head_input
from hlt_classification.luka_attention.worker import data as ordinary_data
from .contracts import artifact, validate, target_recipe, TEACHERS


def data(spec, role, coordinate, authenticated):
    if coordinate == 'OFFLINE' and role == 'train':
        return offline_train(spec, authenticated=authenticated)
    if role not in ('train','validation') or coordinate not in ('D000','U100','D066','D033'):
        raise PermissionError('No unregistered particle roles/views')
    return ordinary_data(spec, role, coordinate, authenticated)


def export(directory, teacher, cache, record, *, device):
    validate(record, 'TEACHER')
    name = record['model_id']; registered = target_recipe(name)
    if (cache.role != 'train' or cache.coordinate_name != TEACHERS[name]
            or record['coordinate'] != TEACHERS[name] or record['role'] != 'train'
            or cache.foundation_sha256 != record['prepared']):
        raise PermissionError('Target export requires exact teacher TRAIN coordinate')
    ref = record['files']['checkpoint']; validate_file_ref(ref)
    saved = torch.load(ref['path'], map_location='cpu', weights_only=True)
    state = teacher.state_dict()
    if set(saved) != set(state) or any(not torch.equal(saved[k], state[k].detach().cpu()) for k in saved):
        raise ValueError('In-memory teacher weights differ from selected native checkpoint')
    del saved, state
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    teacher.to(device).eval().requires_grad_(False)
    values = np.empty((len(cache),128), np.float32)
    with torch.inference_mode(), torch.autocast(torch.device(device).type, enabled=False):
        for start in range(0,len(cache),128):
            raw = cache.batch(np.arange(start,min(start+128,len(cache))))
            _, z = head_input(teacher, **{k:torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')})
            if not torch.isfinite(z).all(): raise ValueError('Nonfinite teacher vector')
            values[start:start+len(z)] = z.cpu().numpy()
            if start % 16384 == 0: print(f'MULTI5 export {name}: {start+len(z)}/{len(cache)}',flush=True)
    normalized, mean, scale = standardize(values)
    atomic_publish_bytes(directory/'targets.npz', deterministic_npz_bytes(dict(identities=cache.identities,
        labels=cache.labels, targets=normalized, mean=mean, scale=scale)))
    validate_file_ref(ref)
    m = artifact('BANK', parents=dict(recipe=registered['content_hash'], prepared=cache.foundation_sha256,
        teacher=record['content_hash']), role='train', coordinate=TEACHERS[name], model_id=name,
        rows=len(cache), width=128, checkpoint=ref, payload=file_ref(directory/'targets.npz',root=directory))
    write_immutable_json(directory/'manifest.json',m)
    return m


def load_one(directory, cache, teacher):
    validate(teacher, 'TEACHER')
    directory = Path(directory); m = load_json(directory/'manifest.json'); name=teacher['model_id']
    validate(m,'BANK',dict(recipe=target_recipe(name)['content_hash'],prepared=cache.foundation_sha256,
                           teacher=teacher['content_hash']))
    if (cache.role != 'train' or cache.coordinate_name != 'D000' or m['role'] != 'train'
            or m['model_id'] != name or m['coordinate'] != TEACHERS[name] or m['rows'] != len(cache)
            or m['width'] != 128 or m['checkpoint'] != teacher['files']['checkpoint']
            or m['payload']['path'] != 'targets.npz'):
        raise ValueError('Wrong target role, coordinate, teacher or population')
    a = load_npz_arrays(validate_file_ref(m['payload'],root=directory))
    if (set(a) != {'identities','labels','targets','mean','scale'}
            or a['identities'].dtype != np.uint8 or a['identities'].shape != (len(cache),32)
            or len(np.unique(a['identities'],axis=0)) != len(cache)
            or a['labels'].dtype != np.int64 or a['labels'].shape != (len(cache),)
            or not np.array_equal(a['identities'],cache.identities) or not np.array_equal(a['labels'],cache.labels)
            or a['targets'].dtype != np.float32 or a['targets'].shape != (len(cache),128)
            or any(a[k].dtype != np.float64 or a[k].shape != (128,) for k in ('mean','scale'))
            or np.any(a['scale']<1e-3) or any(not np.isfinite(a[k]).all() for k in ('targets','mean','scale'))):
        raise ValueError('Malformed bank or wrong ordered TRAIN join')
    return m,a


def load(spec, cache, *, publish=False):
    records, targets = {}, {}
    root=Path(spec['root'])/'targets'
    for name in TEACHERS:
        m,a=load_one(root/name,cache,spec['teachers'][name])
        records[name]=file_ref(root/name/'manifest.json')
        targets[name]=a['targets']
    bundle=artifact('BUNDLE',parents=dict(spec=spec['content_hash'],prepared=cache.foundation_sha256),
        records=records,rows=len(cache),role='train')
    if publish: write_immutable_json(root/'bundle.json',bundle)
    elif load_json(root/'bundle.json')!=bundle: raise ValueError('Shared target bundle changed')
    return SimpleNamespace(manifest=bundle,targets=targets,identities=cache.identities,labels=cache.labels)

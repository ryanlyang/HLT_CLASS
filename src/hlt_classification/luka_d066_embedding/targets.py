"""TRAIN-only D066 targets in a distinct, teacher-bound bank."""
from pathlib import Path
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    atomic_publish_bytes, deterministic_npz_bytes, write_immutable_json)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_embedding_aux.targets import standardize
from hlt_classification.luka_embedding_aux.model import head_input
from .contracts import artifact, validate, recipe, WIDTH


def export(directory, teacher, cache, *, teacher_record, device):
    validate(teacher_record, 'TEACHER')
    if (cache.role != 'train' or cache.coordinate_name != 'D066'
            or teacher_record['coordinate'] != 'D066' or teacher_record['target_role'] != 'train'
            or cache.foundation_sha256 != teacher_record['prepared']):
        raise PermissionError('Only paired TRAIN D066 may generate D066 embedding targets')
    ref = teacher_record['files']['checkpoint']; validate_file_ref(ref)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    teacher.to(device).eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    values = np.empty((len(cache), WIDTH), np.float32)
    with torch.inference_mode(), torch.autocast(torch.device(device).type, enabled=False):
        for start in range(0, len(cache), 128):
            raw = cache.batch(np.arange(start, min(start+128, len(cache))))
            inputs = {k: torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')}
            _, value = head_input(teacher, **inputs)
            if not torch.isfinite(value).all(): raise ValueError('Nonfinite D066 embedding')
            values[start:start+len(value)] = value.float().cpu().numpy()
            if start % (128*128) == 0:
                print(f'D066 target export: {start+len(value)}/{len(cache)}', flush=True)
    normalized, mean, scale = standardize(values)
    atomic_publish_bytes(directory/'targets.npz', deterministic_npz_bytes(dict(
        identities=cache.identities, labels=cache.labels, targets=normalized, mean=mean, scale=scale)))
    validate_file_ref(ref)
    manifest = artifact('BANK', parents=dict(recipe=recipe()['content_hash'],
        prepared=cache.foundation_sha256, teacher=teacher_record['content_hash']),
        role='train', coordinate='D066', width=WIDTH, rows=len(cache), teacher=ref,
        payload=file_ref(directory/'targets.npz', root=directory))
    write_immutable_json(directory/'manifest.json', manifest)
    return manifest


class Bank:
    def __init__(self, directory, cache, *, teacher_record):
        directory = Path(directory); self.manifest = load_json(directory/'manifest.json')
        m = self.manifest
        validate(teacher_record, 'TEACHER')
        validate(m, 'BANK', dict(recipe=recipe()['content_hash'], prepared=cache.foundation_sha256,
                                teacher=teacher_record['content_hash']))
        if (cache.role != 'train' or cache.coordinate_name != 'D000'
                or m['teacher'] != teacher_record['files']['checkpoint']
                or m['role'] != 'train' or m['coordinate'] != 'D066' or m['width'] != WIDTH
                or m['rows'] != len(cache) or m['payload']['path'] != 'targets.npz'):
            raise ValueError('Wrong student role, teacher, target coordinate or population')
        validate_file_ref(m['payload'], root=directory)
        a = load_npz_arrays(directory/'targets.npz')
        if (set(a) != {'identities','labels','targets','mean','scale'}
                or a['identities'].dtype != np.uint8 or a['identities'].shape != (len(cache),32)
                or len(np.unique(a['identities'], axis=0)) != len(cache)
                or a['labels'].dtype != np.int64 or a['labels'].shape != (len(cache),)
                or not np.array_equal(a['identities'], cache.identities)
                or not np.array_equal(a['labels'], cache.labels)
                or a['targets'].dtype != np.float32 or a['targets'].shape != (len(cache),WIDTH)
                or any(a[k].dtype != np.float64 or a[k].shape != (WIDTH,) for k in ('mean','scale'))
                or np.any(a['scale'] < recipe()['minimum_scale'])
                or any(not np.isfinite(a[k]).all() for k in ('targets','mean','scale'))):
            raise ValueError('Invalid D066 target bank or ordered TRAIN join')
        self.targets, self.identities, self.labels = a['targets'], a['identities'], a['labels']

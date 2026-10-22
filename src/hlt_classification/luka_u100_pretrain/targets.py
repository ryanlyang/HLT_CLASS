"""Two disjoint target roles, one frozen TRAIN-only normalization."""
from pathlib import Path
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_embedding_aux.model import head_input
from hlt_classification.luka_embedding_aux.targets import standardize
from .contracts import artifact, validate, recipe


def check_cache(cache, role, coordinate):
    if role not in ('train', 'validation') or cache.role != role or cache.coordinate_name != coordinate:
        raise PermissionError('Forbidden target role or input coordinate')
    if (cache.identities.dtype != np.uint8 or cache.identities.shape != (len(cache), 32)
            or len(np.unique(cache.identities, axis=0)) != len(cache)):
        raise ValueError('Invalid or duplicate jet identities')


def disjoint(train_ids, val_ids):
    if set(map(bytes, train_ids)) & set(map(bytes, val_ids)):
        raise ValueError('TRAIN and validation identities overlap')


def extract(teacher, cache, device):
    if cache.role not in ('train', 'validation') or cache.coordinate_name != 'U100':
        raise PermissionError('Only registered U100 roles can be exported')
    result = np.empty((len(cache), 128), np.float32)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    with torch.inference_mode(), torch.autocast(torch.device(device).type, enabled=False):
        for start in range(0, len(cache), 128):
            raw = cache.batch(np.arange(start, min(start+128, len(cache))))
            _, value = head_input(teacher, **{k: torch.from_numpy(raw[k]).to(device)
                                           for k in ('features', 'vectors', 'mask')})
            if value.shape != (len(raw['labels']), 128) or not torch.isfinite(value).all():
                raise ValueError('Invalid frozen U100 embedding')
            result[start:start+len(value)] = value.float().cpu().numpy()
            if start % (128*128) == 0:
                print(f'U100 targets role={cache.role} rows={start+len(value)}/{len(cache)}', flush=True)
    return result


def export(directory, teacher, train, val, teacher_record, device):
    validate(teacher_record, 'TEACHER')
    if (teacher_record['coordinate'] != 'U100' or teacher_record['target_roles'] != ['train', 'validation']
            or teacher_record['prepared'] != train.foundation_sha256
            or train.foundation_sha256 != val.foundation_sha256):
        raise ValueError('Frozen teacher/population differs')
    check_cache(train, 'train', 'U100'); check_cache(val, 'validation', 'U100')
    disjoint(train.identities, val.identities)
    ref = teacher_record['files']['checkpoint']; validate_file_ref(ref)
    directory = Path(directory); directory.mkdir(exist_ok=False)
    raw = extract(teacher, train, device)
    normalized, mean, scale = standardize(raw)
    del raw
    for role, cache in (('train', train), ('validation', val)):
        values = normalized if role == 'train' else ((extract(teacher, val, device)-mean)/scale).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite normalized targets')
        atomic_publish_bytes(directory/(role+'.npz'), deterministic_npz_bytes(dict(
            identities=cache.identities, labels=cache.labels, targets=values)))
    atomic_publish_bytes(directory/'normalization.npz', deterministic_npz_bytes(dict(mean=mean, scale=scale)))
    validate_file_ref(ref)
    manifest = artifact('BANK', parents=dict(recipe=recipe()['content_hash'], teacher=teacher_record['content_hash'],
        prepared=train.foundation_sha256), teacher=ref, coordinate='U100', width=128,
        normalization_fit_role='train', counts=dict(train=len(train), validation=len(val)),
        outputs=[file_ref(directory/name, root=directory) for name in
                 ('train.npz', 'validation.npz', 'normalization.npz')])
    write_immutable_json(directory/'manifest.json', manifest)
    return manifest


def validate_manifest(directory, teacher_record, prepared, counts):
    directory = Path(directory); m = load_json(directory/'manifest.json')
    validate(teacher_record, 'TEACHER')
    validate(m, 'BANK', dict(recipe=recipe()['content_hash'], teacher=teacher_record['content_hash'],
                            prepared=prepared))
    if (m['coordinate'] != 'U100' or m['width'] != 128 or m['normalization_fit_role'] != 'train'
            or m['teacher'] != teacher_record['files']['checkpoint'] or m['counts'] != counts
            or [r['path'] for r in m['outputs']] != ['train.npz', 'validation.npz', 'normalization.npz']):
        raise ValueError('Wrong target bank semantics')
    for ref in m['outputs']:
        validate_file_ref(ref, root=directory)
    return m


class Bank:
    def __init__(self, directory, train, val, teacher_record):
        directory = Path(directory)
        self.manifest = validate_manifest(directory, teacher_record, train.foundation_sha256,
                                         dict(train=len(train), validation=len(val)))
        check_cache(train, 'train', 'D000'); check_cache(val, 'validation', 'D000')
        disjoint(train.identities, val.identities)
        if val.foundation_sha256 != train.foundation_sha256:
            raise ValueError('Different target populations')
        moments = load_npz_arrays(directory/'normalization.npz')
        if (set(moments) != {'mean', 'scale'} or any(a.dtype != np.float64 or a.shape != (128,)
                or not np.isfinite(a).all() for a in moments.values()) or np.any(moments['scale'] < 1e-3)):
            raise ValueError('Invalid TRAIN normalization')
        self.payloads = {}
        for role, cache in (('train', train), ('validation', val)):
            a = load_npz_arrays(directory/(role+'.npz'))
            if (set(a) != {'identities', 'labels', 'targets'}
                    or not np.array_equal(a['identities'], cache.identities)
                    or a['identities'].dtype != np.uint8 or a['labels'].dtype != np.int64
                    or not np.array_equal(a['labels'], cache.labels)
                    or a['targets'].dtype != np.float32 or a['targets'].shape != (len(cache), 128)
                    or not np.isfinite(a['targets']).all()):
                raise ValueError('Target role/identity/label/shape join differs')
            self.payloads[role] = a

    def for_cache(self, cache):
        check_cache(cache, cache.role, 'D000')
        if cache.foundation_sha256 != self.manifest['parents']['prepared']:
            raise ValueError('Prepared population differs')
        a = self.payloads[cache.role]
        if not np.array_equal(a['identities'], cache.identities) or not np.array_equal(a['labels'], cache.labels):
            raise ValueError('Ordered target identity join differs')
        return a['targets']

"""TRAIN-only native predecessor targets in distinct, selected-teacher-bound banks."""
from pathlib import Path
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    atomic_publish_bytes, deterministic_npz_bytes, write_immutable_json)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_embedding_aux.targets import standardize
from hlt_classification.luka_embedding_aux.model import head_input
from .contracts import artifact, validate, recipe, WIDTH, VIEWS, teacher_id


def export(directory, teacher, cache, *, teacher_record, device):
    validate_teacher(teacher_record)
    if (cache.role != 'train' or cache.coordinate_name != teacher_record['coordinate']
            or cache.foundation_sha256 != teacher_record['prepared']):
        raise PermissionError('Only paired TRAIN predecessor views may generate native targets')
    for reference in teacher_record['files'].values(): validate_file_ref(reference)
    ref = teacher_record['files']['checkpoint']
    expected = torch.load(ref['path'], map_location='cpu', weights_only=True)
    actual = teacher.state_dict()
    if (expected.keys() != actual.keys() or any(k.startswith('predictor.') for k in actual)
            or any(not torch.equal(expected[k].cpu(), actual[k].detach().cpu()) for k in actual)):
        raise ValueError('Teacher must be the exact selected NATIVE checkpoint, without predictor')
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    teacher.to(device).eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    values = np.empty((len(cache), WIDTH), np.float32)
    with torch.inference_mode(), torch.autocast(torch.device(device).type, enabled=False):
        for start in range(0, len(cache), 128):
            raw = cache.batch(np.arange(start, min(start+128, len(cache))))
            inputs = {k: torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')}
            _, value = head_input(teacher, **inputs)
            if not torch.isfinite(value).all(): raise ValueError('Nonfinite native teacher embedding')
            values[start:start+len(value)] = value.float().cpu().numpy()
            if start % (128*128) == 0:
                print(f'NATIVE_EMBED target export: {start+len(value)}/{len(cache)}', flush=True)
    normalized, mean, scale = standardize(values)
    atomic_publish_bytes(directory/'targets.npz', deterministic_npz_bytes(dict(
        identities=cache.identities, labels=cache.labels, targets=normalized, mean=mean, scale=scale)))
    for reference in teacher_record['files'].values(): validate_file_ref(reference)
    manifest = artifact('BANK', parents=dict(recipe=recipe()['content_hash'],
        prepared=cache.foundation_sha256, teacher=teacher_record['content_hash']),
        role='train', coordinate=teacher_record['coordinate'], width=WIDTH, rows=len(cache), teacher=teacher_record,
        tap='native_preclassifier',
        payload=file_ref(directory/'targets.npz', root=directory))
    write_immutable_json(directory/'manifest.json', manifest)
    return manifest


class Bank:
    def __init__(self, directory, cache, *, teacher_record, node, technical=False):
        directory = Path(directory); self.manifest = load_json(directory/'manifest.json')
        m = self.manifest
        validate_teacher(teacher_record)
        validate(m, 'BANK', dict(recipe=recipe()['content_hash'], prepared=cache.foundation_sha256,
                                teacher=teacher_record['content_hash']))
        if (cache.role != 'train' or cache.coordinate_name != node['coordinate']
                or m['teacher'] != teacher_record or m['tap'] != 'native_preclassifier'
                or m['role'] != 'train' or m['coordinate'] != node['embedding_teacher_coordinate'] or m['width'] != WIDTH
                or teacher_record['model_id'] != node['embedding_teacher']
                or (teacher_record['coordinate'] != 'U000' and teacher_record['technical'] is not technical)
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
            raise ValueError('Invalid native predecessor bank or ordered TRAIN join')
        self.targets, self.identities, self.labels = a['targets'], a['identities'], a['labels']

def validate_teacher(record):
    validate(record, 'TEACHER', dict(spec=record['spec'], prepared=record['prepared'],
        checkpoint=record['files']['checkpoint']['sha256'], report=record['files']['report']['sha256'],
        predictions=record['files']['predictions']['sha256']))
    co = record['coordinate']
    if (co not in VIEWS[:-1] or record['model_id'] != teacher_id(co) or record['target_role'] != 'train'
            or record['origin'] != ('original_anchor' if co == 'U000' else 'this_campaign')
            or type(record['technical']) is not bool or (co == 'U000' and record['technical'])
            or record['tap'] != 'native_preclassifier'
            or set(record['files']) != {'checkpoint', 'report', 'predictions'}):
        raise ValueError('Invalid selected native teacher descriptor')


def validate_manifest(directory, teacher, prepared, counts):
    directory = Path(directory); m = load_json(directory/'manifest.json')
    validate_teacher(teacher)
    validate(m, 'BANK', dict(recipe=recipe()['content_hash'], prepared=prepared, teacher=teacher['content_hash']))
    if (m['role'] != 'train' or m['coordinate'] != teacher['coordinate'] or m['width'] != WIDTH
            or m['rows'] != counts['train'] or m['teacher'] != teacher or m['tap'] != 'native_preclassifier'
            or m['payload']['path'] != 'targets.npz'):
        raise ValueError('Wrong native TRAIN bank binding')
    validate_file_ref(m['payload'], root=directory)
    return m

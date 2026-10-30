"""Shared TRAIN-only targets with exact frozen-classifier reconstruction."""
from pathlib import Path
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from .contracts import artifact, validate, recipe, TEACHERS
from . import teachers


def roundtrip(a):
    reconstructed = a['mean'].astype(np.float32) + a['scale'].astype(np.float32)*a['targets']
    logits = reconstructed @ a['weight'].T + a['bias']
    np.testing.assert_allclose(logits, a['logits'], rtol=2e-5, atol=2e-5)
    return float(np.max(np.abs(logits-a['logits'])))


def export(folder, model, cache, name, teacher_ref, *, device, source_hash):
    from hlt_classification.data.cache_contracts import require_sha256
    require_sha256(source_hash, name='target source')
    if cache.role != 'train' or cache.coordinate_name != name or name not in TEACHERS:
        raise PermissionError('Export only registered teacher TRAIN embeddings')
    validate_file_ref(teacher_ref)
    model.eval().requires_grad_(False)
    embeddings, logits = [], []
    with torch.no_grad():
        for start in range(0, len(cache), 128):
            raw = cache.batch(np.arange(start, min(start+128, len(cache))))
            out, h = teachers.forward(model, raw, device, bf16=False, name=name, capture=True)
            embeddings.append(h.float().cpu().numpy()); logits.append(out.float().cpu().numpy())
    raw = np.concatenate(embeddings)
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite teacher embeddings')
    mean = raw.astype(np.float64).mean(0)
    scale = np.maximum(raw.astype(np.float64).std(0), .001)
    fc = teachers.classifier(model, name)[0]
    arrays = dict(identities=cache.identities, labels=cache.labels,
        targets=((raw-mean)/scale).astype(np.float32), mean=mean, scale=scale,
        logits=np.concatenate(logits), weight=fc.weight.detach().float().cpu().numpy(),
        bias=fc.bias.detach().float().cpu().numpy())
    error = roundtrip(arrays)
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    atomic_publish_bytes(folder/'targets.npz', deterministic_npz_bytes(arrays))
    value = artifact('BANK', parents=dict(source=source_hash, recipe=recipe()['content_hash'], prepared=cache.foundation_sha256,
        checkpoint=teacher_ref['sha256']), teacher=name, checkpoint=teacher_ref,
        rows=len(cache), role='train', width=128, payload=file_ref(folder/'targets.npz', root=folder),
        maximum_readout_error=error, readout_rtol=2e-5, readout_atol=2e-5)
    write_immutable_json(folder/'manifest.json', value)
    validate_file_ref(teacher_ref)
    return value


class Bank:
    def __init__(self, folder, cache, name, teacher_ref, *, source_hash):
        if cache.role != 'train' or cache.coordinate_name != 'D000':
            raise PermissionError('Banks are training supervision only')
        folder = Path(folder); record = load_json(folder/'manifest.json')
        validate(record, 'BANK', dict(source=source_hash, recipe=recipe()['content_hash'], prepared=cache.foundation_sha256,
                                     checkpoint=teacher_ref['sha256']))
        if (record['teacher'] != name or record['checkpoint'] != teacher_ref or record['role'] != 'train'
                or record['rows'] != len(cache) or record['width'] != 128
                or record['payload']['path'] != 'targets.npz'
                or (record['readout_rtol'], record['readout_atol']) != (2e-5, 2e-5)):
            raise ValueError('Target bank teacher/view/role differs')
        validate_file_ref(record['payload'], root=folder)
        a = load_npz_arrays(folder/'targets.npz')
        shapes = dict(identities=(len(cache), 32), labels=(len(cache),), targets=(len(cache), 128),
                      mean=(128,), scale=(128,), logits=(len(cache), 11), weight=(11, 128), bias=(11,))
        if set(a) != set(shapes):
            raise ValueError('Target array inventory differs')
        for k, shape in shapes.items():
            dtype = {'identities': np.uint8, 'labels': np.int64, 'mean': np.float64, 'scale': np.float64}.get(k, np.float32)
            if a[k].shape != shape or a[k].dtype != dtype or not np.isfinite(a[k]).all():
                raise ValueError('Target shape/dtype/finiteness differs: '+k)
        if (not np.array_equal(a['identities'], cache.identities) or not np.array_equal(a['labels'], cache.labels)
                or np.any(a['scale'] < .001) or np.max(np.abs(a['targets'].mean(0, dtype=np.float64))) > 1e-5
                or np.max(a['targets'].std(0, dtype=np.float64)) > 1.00001):
            raise ValueError('Target join/normalization differs')
        error = roundtrip(a)
        if error != record['maximum_readout_error']:
            # BLAS reduction differences are allowed only within the explicit FP32 tolerance.
            if abs(error-record['maximum_readout_error']) > 2e-5:
                raise ValueError('Readout roundtrip evidence differs')
        self.arrays, self.manifest = a, record

"""Owned scratch-only FP32 token targets; immutable retained lineage, no padded bank."""
from pathlib import Path
import gc
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (atomic_publish_bytes, deterministic_npz_bytes,
    load_npz_arrays, write_immutable_json, load_json)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_attention.banks import counts, alignment
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from .contracts import artifact, validate, recipe
from .model import token_forward


def check_pair(teacher, student):
    if teacher.role != 'train' or student.role != 'train':
        raise PermissionError('Token targets are TRAIN only')
    if teacher.coordinate_name != 'U100' or student.coordinate_name != 'D000':
        raise PermissionError('Only original U100 to HLT is registered')
    if alignment(teacher) != alignment(student):
        raise ValueError('Persistent-slot identities/labels/counts/prepared binding differs')


def export(root, model, teacher_cache, student_cache, teacher_record, *, device):
    check_pair(teacher_cache, student_cache)
    if (teacher_record['coordinate'] != 'U100' or teacher_record['target_role'] != 'train'
            or teacher_record['prepared'] != teacher_cache.foundation_sha256):
        raise ValueError('Teacher target meaning differs')
    validate_file_ref(teacher_record['files']['checkpoint'])
    root = Path(root); root.mkdir(exist_ok=False)
    lengths = counts(teacher_cache)
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    payload = np.lib.format.open_memmap(root/'tokens.npy', mode='w+', dtype=np.float32,
                                       shape=(int(offsets[-1]), 128))
    mean, m2, seen = np.zeros(128), np.zeros(128), 0
    model.eval()
    # Reproducible teacher targets across the independent jobs. Restore normal
    # production backend settings before any student training.
    with parity_backend(device), torch.inference_mode():
        for start in range(0, len(teacher_cache), 128):
            ix = np.arange(start, min(start+128, len(teacher_cache)))
            raw = teacher_cache.batch(ix)
            x, v, m = [torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')]
            _, vectors = token_forward(model, x, v, m)
            real = vectors[m[:, 0]].float().cpu().numpy()
            if not np.isfinite(real).all():
                raise ValueError('Nonfinite teacher token targets')
            lo, hi = offsets[start], offsets[start+len(ix)]
            if len(real) != hi-lo:
                raise ValueError('Teacher token order/support changed')
            payload[lo:hi] = real
            values = real.astype(np.float64); batch_mean = values.mean(0)
            batch_m2 = ((values-batch_mean)**2).sum(0)
            delta = batch_mean-mean; total = seen+len(values)
            m2 += batch_m2+delta**2*seen*len(values)/total
            mean += delta*len(values)/total; seen = total
            if start % (128*100) == 0:
                print(f'TOKEN export rows={start+len(ix)}/{len(teacher_cache)} particles={seen}', flush=True)
    payload.flush(); del payload
    scale = np.maximum(np.sqrt(m2/seen), 1e-3)
    atomic_publish_bytes(root/'index.npz', deterministic_npz_bytes(dict(offsets=offsets,
        identities=teacher_cache.identities, labels=teacher_cache.labels, mean=mean, scale=scale)))
    record = artifact('BANK', parents=dict(recipe=recipe()['content_hash'], teacher=teacher_record['content_hash'],
        prepared=teacher_cache.foundation_sha256), alignment=alignment(teacher_cache), role='train',
        coordinate='U100', width=128, rows=len(teacher_cache), tokens=seen,
        checkpoint=teacher_record['files']['checkpoint'], dtype='float32',
        outputs=[file_ref(root/n, root=root) for n in ('tokens.npy','index.npz')])
    write_immutable_json(root/'manifest.json', record)
    return record


class Bank:
    def __init__(self, root, cache, teacher_record):
        root = Path(root); self.manifest = load_json(root/'manifest.json')
        validate(self.manifest, 'BANK', dict(recipe=recipe()['content_hash'], teacher=teacher_record['content_hash'],
                                          prepared=cache.foundation_sha256))
        m = self.manifest
        if (cache.role != 'train' or cache.coordinate_name != 'D000' or m['alignment'] != alignment(cache)
                or (m['role'],m['coordinate'],m['width'],m['dtype'],m['rows']) != ('train','U100',128,'float32',len(cache))
                or m['checkpoint'] != teacher_record['files']['checkpoint']
                or [r['path'] for r in m['outputs']] != ['tokens.npy','index.npz']):
            raise ValueError('Token bank surface/teacher/population differs')
        for ref in m['outputs']: validate_file_ref(ref, root=root)
        a = load_npz_arrays(root/'index.npz')
        self.identities, self.labels = a['identities'], a['labels']
        self.offsets, self.mean, self.scale = a['offsets'], a['mean'], a['scale']
        if (not np.array_equal(self.identities, cache.identities) or not np.array_equal(self.labels, cache.labels)
                or self.offsets.dtype != np.int64 or self.offsets.shape != (len(cache)+1,) or self.offsets[0] != 0
                or not np.array_equal(np.diff(self.offsets), counts(cache))
                or any(v.dtype != np.float64 or v.shape != (128,) or not np.isfinite(v).all()
                       for v in (self.mean,self.scale)) or np.any(self.scale < 1e-3)):
            raise ValueError('Token bank index/normalization differs')
        self.payload = np.load(root/'tokens.npy', mmap_mode='r', allow_pickle=False)
        if self.payload.dtype != np.float32 or self.payload.shape != (int(self.offsets[-1]),128) or m['tokens'] != self.offsets[-1]:
            raise ValueError('Token payload dimensions differ')

    def batch(self, indices, width):
        indices=np.asarray(indices)
        if (indices.dtype!=np.int64 or indices.ndim!=1 or np.any(indices<0)
                or np.any(indices>=len(self.labels)) or not 1<=width<=512):
            raise ValueError('Invalid bank batch request')
        output = np.zeros((len(indices), width, 128), np.float32)
        for row, i in enumerate(indices):
            start, end = self.offsets[i:i+2]
            if end-start > width: raise ValueError('Student truncated a real token')
            values = (self.payload[start:end].astype(np.float64)-self.mean)/self.scale
            if not np.isfinite(values).all(): raise ValueError('Nonfinite normalized targets')
            output[row, :end-start] = values.astype(np.float32)
        return output

    def close(self):
        # Drop our only owned read-only mmap before TemporaryDirectory retirement.
        del self.payload
        gc.collect()

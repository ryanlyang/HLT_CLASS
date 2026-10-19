"""Compact TRAIN-only frozen-teacher targets, including deleted-cache recovery."""
from pathlib import Path
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    atomic_publish_bytes, deterministic_npz_bytes, write_immutable_json)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.cache import RamCache
from hlt_classification.luka_fullsim import regular_cache, stage2
from hlt_classification.luka_fullsim.particles import physical, RawJet
from hlt_classification.luka_fullsim.preflight_cache import _block
from .contracts import artifact, validate, recipe, WIDTH
from .model import head_input


def offline_train(gate, *, authenticated=None):
    """Reconstruct only OFFLINE TRAIN, preserving original prepared semantics."""
    root = Path(gate['anchor']['evidence']['prepared_root'])
    meta = regular_cache.metadata(root) if authenticated is None else authenticated
    manifest, prepared, inventory, splits = meta
    if (manifest['content_hash'] != gate['anchor']['evidence']['cache']
            or prepared['content_hash'] != gate['anchor']['evidence']['prepared']):
        raise ValueError('Offline reconstruction metadata differs')
    budget = int(gate['memory_mb'] * 1024**2 * .25)
    if 2*prepared['summaries']['train']['OFFLINE']['resident_bytes'] + 64*2**20 > budget:
        raise MemoryError('Offline TRAIN reconstruction exceeds RAM bound')
    # No durable OFFLINE cache or matching is required; raw source bytes are
    # authenticated before and after each bounded selected-file iteration.
    blocks, meter = [], stage2.ViewMeter()
    for member in splits['memberships']['train']:
        rows = []
        for jet in regular_cache.selected_file_rows(gate['anchor']['evidence']['raw_container'], inventory, member):
            value = stage2.build_inputs(physical(jet.offline, prepared['conventions'], 'offline'),
                                       capacity=prepared['inputs']['capacity'])
            meter.add(jet, value)
            rows.append((RawJet(jet.identity, jet.label, jet.file_index, jet.entry, None, None), value))
        blocks.append(_block(member['file_index'], rows))
        print(f'EMBED OFFLINE TRAIN files={len(blocks)}/{len(splits["memberships"]["train"])} rows={meter.rows}', flush=True)
    if meter.report() != prepared['summaries']['train']['OFFLINE']:
        raise ValueError('Rebuilt OFFLINE TRAIN fingerprint differs')
    result = RamCache(blocks, role='train', coordinate_name='OFFLINE', foundation_sha256=prepared['content_hash'])
    if result.nbytes > budget or len(result) != gate['counts']['train']:
        raise ValueError('Offline TRAIN population or memory differs')
    return result


def standardize(values):
    if values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != WIDTH or not np.isfinite(values).all():
        raise ValueError('Invalid teacher embeddings')
    mean = np.mean(values, axis=0, dtype=np.float64)
    scale = np.maximum(np.std(values, axis=0, dtype=np.float64), recipe()['minimum_scale'])
    return ((values.astype(np.float64)-mean)/scale).astype(np.float32), mean, scale


def export(directory, teacher, cache, *, teacher_ref, device):
    if cache.role != 'train' or cache.coordinate_name != 'OFFLINE':
        raise PermissionError('Targets and normalization require TRAIN OFFLINE only')
    validate_file_ref(teacher_ref)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    values = np.empty((len(cache), WIDTH), np.float32)
    with torch.inference_mode():
        for start in range(0, len(cache), 128):
            ix = np.arange(start, min(start+128, len(cache)))
            raw = cache.batch(ix)
            inputs = {k: torch.from_numpy(raw[k]).to(device) for k in ('features', 'vectors', 'mask')}
            _, embedding = head_input(teacher, **inputs)
            if not torch.isfinite(embedding).all():
                raise ValueError('Nonfinite offline embedding')
            values[ix] = embedding.float().cpu().numpy()
    normalized, mean, scale = standardize(values)
    atomic_publish_bytes(directory/'targets.npz', deterministic_npz_bytes(dict(
        identities=cache.identities, labels=cache.labels, targets=normalized, mean=mean, scale=scale)))
    validate_file_ref(teacher_ref)
    manifest = artifact('BANK', parents=dict(recipe=recipe()['content_hash'], prepared=cache.foundation_sha256,
        teacher=teacher_ref['sha256']), role='train', coordinate='OFFLINE', rows=len(cache), width=WIDTH,
        teacher=teacher_ref, payload=file_ref(directory/'targets.npz', root=directory))
    write_immutable_json(directory/'manifest.json', manifest)
    return manifest


class Bank:
    def __init__(self, directory, cache, *, teacher_ref):
        directory = Path(directory)
        self.manifest = load_json(directory/'manifest.json')
        validate(self.manifest, 'BANK', dict(recipe=recipe()['content_hash'],
            prepared=cache.foundation_sha256, teacher=teacher_ref['sha256']))
        m = self.manifest
        if (cache.role != 'train' or cache.coordinate_name not in ('D000', 'OFFLINE')
                or m['teacher'] != teacher_ref or m['role'] != 'train' or m['coordinate'] != 'OFFLINE'
                or m['rows'] != len(cache) or m['width'] != WIDTH or m['payload']['path'] != 'targets.npz'):
            raise ValueError('Target bank meaning differs')
        validate_file_ref(m['payload'], root=directory)
        a = load_npz_arrays(directory/'targets.npz')
        if (set(a) != {'identities', 'labels', 'targets', 'mean', 'scale'}
                or a['identities'].dtype != np.uint8 or a['identities'].shape != (len(cache), 32)
                or len(np.unique(a['identities'], axis=0)) != len(cache)
                or not np.array_equal(a['identities'], cache.identities)
                or a['labels'].dtype != np.int64 or not np.array_equal(a['labels'], cache.labels)
                or a['targets'].shape != (len(cache), WIDTH) or a['targets'].dtype != np.float32
                or any(a[k].dtype != np.float64 or a[k].shape != (WIDTH,) for k in ('mean', 'scale'))
                or np.any(a['scale'] < recipe()['minimum_scale'])
                or any(not np.isfinite(a[k]).all() for k in ('targets', 'mean', 'scale'))):
            raise ValueError('Invalid targets, normalization or ordered TRAIN identity join')
        self.targets, self.identities, self.labels = a['targets'], a['identities'], a['labels']

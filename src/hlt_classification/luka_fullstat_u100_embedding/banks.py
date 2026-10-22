"""One frozen TRAIN-only U100 embedding bank, no probability supervision."""
from pathlib import Path
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    atomic_publish_bytes, deterministic_npz_bytes, write_immutable_json)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_embedding_aux.model import head_input
from hlt_classification.luka_embedding_aux.targets import standardize
from .contracts import artifact, validate, target_recipe


def parents(prepared, refs):
    return dict(recipe=target_recipe()['content_hash'], prepared=prepared,
                checkpoint=refs['checkpoint']['sha256'], report=refs['report']['sha256'])


def export(directory, teacher, cache, *, refs, device):
    if cache.role != 'train' or cache.coordinate_name != 'U100':
        raise PermissionError('Embedding export/normalization requires U100 TRAIN')
    for key in ('checkpoint', 'report'):
        validate_file_ref(refs[key])
    expected = torch.load(refs['checkpoint']['path'], map_location='cpu', weights_only=True)
    if expected.keys() != teacher.state_dict().keys() or any(
            not torch.equal(v.detach().cpu(), expected[k]) for k, v in teacher.state_dict().items()):
        raise ValueError('Teacher weights differ from selected U100 checkpoint')
    del expected
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    teacher.to(device).eval().requires_grad_(False)
    values = np.empty((len(cache), 128), np.float32)
    with torch.inference_mode(), torch.autocast(torch.device(device).type, enabled=False):
        for start in range(0, len(cache), 128):
            ix = np.arange(start, min(start+128, len(cache)))
            raw = cache.batch(ix)
            tensors = {k: torch.from_numpy(raw[k]).to(device) for k in ('features', 'vectors', 'mask')}
            _, embedding = head_input(teacher, **tensors)
            if not torch.isfinite(embedding).all():
                raise ValueError('Nonfinite U100 target')
            values[ix] = embedding.float().cpu().numpy()
            if start % (128*256) == 0:
                print(f'FULLSTAT EMBED export rows={start+len(ix)}/{len(cache)}', flush=True)
    targets, mean, scale = standardize(values)
    del values
    atomic_publish_bytes(directory/'targets.npz', deterministic_npz_bytes(dict(
        identities=cache.identities, labels=cache.labels, targets=targets, mean=mean, scale=scale)))
    for key in ('checkpoint', 'report'):
        validate_file_ref(refs[key])
    record = artifact('BANK', parents=parents(cache.foundation_sha256, refs),
        role='train', coordinate='U100', teacher='COARSE_U100', rows=len(cache), width=128,
        checkpoint=refs['checkpoint'], report=refs['report'],
        payload=file_ref(directory/'targets.npz', root=directory))
    write_immutable_json(directory/'manifest.json', record)
    return record


class Bank:
    def __init__(self, directory, cache, *, refs):
        directory = Path(directory)
        m = self.manifest = load_json(directory/'manifest.json')
        validate(m, 'BANK', parents(cache.foundation_sha256, refs))
        if (cache.role != 'train' or cache.coordinate_name not in ('D000', 'U100')
                or any(m.get(k) != v for k, v in dict(role='train', coordinate='U100',
                    teacher='COARSE_U100', rows=len(cache), width=128,
                    checkpoint=refs['checkpoint'], report=refs['report']).items())
                or m['payload']['path'] != 'targets.npz'):
            raise ValueError('Embedding bank semantics differ')
        a = load_npz_arrays(validate_file_ref(m['payload'], root=directory))
        if (set(a) != {'identities', 'labels', 'targets', 'mean', 'scale'}
                or a['identities'].dtype != np.uint8 or a['identities'].shape != (len(cache), 32)
                or not np.array_equal(a['identities'], cache.identities)
                or len(np.unique(a['identities'], axis=0)) != len(cache)
                or a['labels'].dtype != np.int64 or not np.array_equal(a['labels'], cache.labels)
                or a['targets'].dtype != np.float32 or a['targets'].shape != (len(cache), 128)
                or any(a[k].dtype != np.float64 or a[k].shape != (128,) for k in ('mean', 'scale'))
                or np.any(a['scale'] < 1e-3)
                or any(not np.isfinite(a[k]).all() for k in ('targets', 'mean', 'scale'))):
            raise ValueError('Embedding normalization or ordered TRAIN join differs')
        self.targets, self.identities, self.labels = a['targets'], a['identities'], a['labels']
        for value in a.values():
            value.flags.writeable = False


def load(gate, train):
    if train.role != 'train' or train.coordinate_name != 'D000':
        raise PermissionError('Only native HLT TRAIN consumes supervision')
    return Bank(Path(gate['root'])/'targets', train,
                refs=gate['teacher_refs'])


def check_join(bank, train):
    validate(bank.manifest, 'BANK')
    if (train.role != 'train' or train.coordinate_name != 'D000'
            or bank.manifest['parents']['prepared'] != train.foundation_sha256
            or bank.manifest['role'] != 'train' or bank.manifest['rows'] != len(train)
            or bank.manifest['teacher'] != 'COARSE_U100' or bank.manifest['coordinate'] != 'U100'
            or bank.manifest['parents']['recipe'] != target_recipe()['content_hash']
            or not np.array_equal(bank.identities, train.identities)
            or not np.array_equal(bank.labels, train.labels)
            or bank.targets.shape != (len(train), 128) or bank.targets.dtype != np.float32
            or not np.isfinite(bank.targets).all()):
        raise ValueError('Embedding TRAIN identity/label/shape/finite join differs')


"""Compact TRAIN-only pure-OFFLINE vectors and original D033 probabilities."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    atomic_publish_bytes, deterministic_npz_bytes, write_immutable_json)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.banks import load_bank
from hlt_classification.luka_embedding_aux.model import head_input
from hlt_classification.luka_embedding_aux.targets import standardize
from .contracts import artifact, validate, recipe, TEACHER


def parents(prepared, refs):
    return dict(recipe=recipe()['content_hash'], prepared=prepared,
                checkpoint=refs['checkpoint']['sha256'], report=refs['report']['sha256'])


def export(directory, teacher, cache, *, refs, device):
    if cache.role != 'train' or cache.coordinate_name != 'OFFLINE':
        raise PermissionError('Embedding export/normalization requires pure OFFLINE TRAIN')
    for key in ('checkpoint', 'report'):
        validate_file_ref(refs[key])
    expected = torch.load(refs['checkpoint']['path'], map_location='cpu', weights_only=True)
    if expected.keys() != teacher.state_dict().keys() or any(
            not torch.equal(v.detach().cpu(), expected[k]) for k, v in teacher.state_dict().items()):
        raise ValueError('Teacher weights differ from selected OFFLINE checkpoint')
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
                raise ValueError('Nonfinite pure-OFFLINE target')
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
        role='train', coordinate='OFFLINE', teacher='OFFLINE', rows=len(cache), width=128,
        checkpoint=refs['checkpoint'], report=refs['report'],
        payload=file_ref(directory/'targets.npz', root=directory))
    write_immutable_json(directory/'manifest.json', record)
    return record


class Bank:
    def __init__(self, directory, cache, *, refs):
        directory = Path(directory)
        m = self.manifest = load_json(directory/'manifest.json')
        validate(m, 'BANK', parents(cache.foundation_sha256, refs))
        if (cache.role != 'train' or cache.coordinate_name not in ('D000', 'OFFLINE')
                or any(m.get(k) != v for k, v in dict(role='train', coordinate='OFFLINE',
                    teacher='OFFLINE', rows=len(cache), width=128,
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
    refs = gate['anchor']['evidence']['models']
    embeddings = Bank(Path(gate['root'])/'targets', train, refs=refs['OFFLINE'])
    ref = gate['anchor']['logits']
    directory = Path(validate_file_ref(ref)).parent
    report = load_json(validate_file_ref(refs[TEACHER]['report']))
    q = load_bank(directory, foundation_sha256=train.foundation_sha256,
        teacher_report_sha256=report['content_hash'], teacher_node=TEACHER,
        role='train', expected_identities=train.identities)
    q.flags.writeable = False
    manifest = artifact('SUPERVISION', parents=dict(embeddings=embeddings.manifest['content_hash'],
        logits=load_json(directory/'manifest.json')['content_hash']),
        prepared=train.foundation_sha256, rows=len(train), role='train')
    return SimpleNamespace(manifest=manifest, targets=embeddings.targets, probabilities=q,
                           identities=embeddings.identities, labels=embeddings.labels)


def check_join(bank, train):
    kind = 'TECHNICAL_SUPERVISION' if bank.manifest.get('contract') == (
        'LUKA_FULLSTAT_LAST_EMBEDDING_TECHNICAL_SUPERVISION/v1') else 'SUPERVISION'
    validate(bank.manifest, kind)
    if (train.role != 'train' or train.coordinate_name != 'D000'
            or bank.manifest['prepared'] != train.foundation_sha256 or bank.manifest['role'] != 'train'
            or bank.manifest['rows'] != len(train)
            or not np.array_equal(bank.identities, train.identities)
            or not np.array_equal(bank.labels, train.labels)
            or bank.targets.shape != (len(train), 128) or bank.targets.dtype != np.float32
            or bank.probabilities.shape != (len(train), 11) or bank.probabilities.dtype != np.float32
            or not np.isfinite(bank.targets).all() or not np.isfinite(bank.probabilities).all()
            or np.any(bank.probabilities < 0)
            or not np.allclose(bank.probabilities.sum(-1), 1., atol=2e-6, rtol=0)):
        raise ValueError('Supervision identity/label/shape/finite TRAIN join differs')

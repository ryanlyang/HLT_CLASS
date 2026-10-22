"""Teacher-specific banks and exact joins; imported banks keep their original contracts."""
from pathlib import Path
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    atomic_publish_bytes, deterministic_npz_bytes, write_immutable_json)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_embedding_aux.model import head_input
from hlt_classification.luka_embedding_aux.targets import standardize
from .contracts import artifact, validate, target_recipe, TEACHERS


def parents(prepared, refs, teacher_id):
    return dict(recipe=target_recipe(teacher_id)['content_hash'], prepared=prepared,
                checkpoint=refs['checkpoint']['sha256'], report=refs['report']['sha256'])


def export(directory, teacher, cache, *, refs, device, teacher_id):
    if teacher_id not in TEACHERS or cache.role != 'train' or cache.coordinate_name != TEACHERS[teacher_id]:
        raise PermissionError('Embedding export/normalization requires teacher TRAIN')
    for key in ('checkpoint', 'report'):
        validate_file_ref(refs[key])
    expected = torch.load(refs['checkpoint']['path'], map_location='cpu', weights_only=True)
    if expected.keys() != teacher.state_dict().keys() or any(
            not torch.equal(v.detach().cpu(), expected[k]) for k, v in teacher.state_dict().items()):
        raise ValueError('Teacher weights differ from selected teacher checkpoint')
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
                raise ValueError('Nonfinite teacher target')
            values[ix] = embedding.float().cpu().numpy()
            if start % (128*256) == 0:
                print(f'FULLSTAT EMBED export rows={start+len(ix)}/{len(cache)}', flush=True)
    targets, mean, scale = standardize(values)
    del values
    atomic_publish_bytes(directory/'targets.npz', deterministic_npz_bytes(dict(
        identities=cache.identities, labels=cache.labels, targets=targets, mean=mean, scale=scale)))
    for key in ('checkpoint', 'report'):
        validate_file_ref(refs[key])
    record = artifact('BANK', parents=parents(cache.foundation_sha256, refs, teacher_id),
        role='train', coordinate=TEACHERS[teacher_id], teacher=teacher_id, rows=len(cache), width=128,
        checkpoint=refs['checkpoint'], report=refs['report'],
        payload=file_ref(directory/'targets.npz', root=directory))
    write_immutable_json(directory/'manifest.json', record)
    return record


class Bank:
    def __init__(self, directory, cache, *, refs, teacher_id):
        if teacher_id not in TEACHERS:
            raise ValueError('Unknown teacher')
        directory = Path(directory)
        m = self.manifest = load_json(directory/'manifest.json')
        validate(m, 'BANK', parents(cache.foundation_sha256, refs, teacher_id))
        if (cache.role != 'train' or cache.coordinate_name not in ('D000', TEACHERS[teacher_id])
                or any(m.get(k) != v for k, v in dict(role='train', coordinate=TEACHERS[teacher_id],
                    teacher=teacher_id, rows=len(cache), width=128,
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


def records(spec):
    if set(spec['reuse_banks']) != {'OFFLINE', 'COARSE_U100'}:
        raise ValueError('Only original OFFLINE and U100 banks may be reused')
    result = {}
    for teacher in TEACHERS:
        if teacher in spec['reuse_banks']:
            ref = spec['reuse_banks'][teacher]
            validate_file_ref(ref)
        else:
            ref = file_ref(Path(spec['root'])/'targets'/teacher/'manifest.json')
        result[teacher] = ref
    return result


def bundle_record(spec):
    refs = records(spec)
    return artifact('BUNDLE', parents=dict(spec=spec['content_hash'],
        **{t: load_json(validate_file_ref(refs[t]))['content_hash'] for t in TEACHERS}),
        role='train', prepared=spec['anchor']['evidence']['prepared'],
        rows=spec['counts']['train'], teacher_order=list(TEACHERS),
        teachers=spec['teachers'], records=refs)


class Bundle:
    def __init__(self, spec, train):
        if train.role != 'train' or train.coordinate_name != 'D000':
            raise PermissionError('Only HLT TRAIN can consume the four target banks')
        from hlt_classification.luka_fullstat_u100_embedding import banks as u100
        from hlt_classification.luka_fullstat_embedding_sweep import banks as offline
        self.manifest = bundle_record(spec)
        self.targets = {}
        self.identities, self.labels = train.identities, train.labels
        for teacher in TEACHERS:
            directory = Path(self.manifest['records'][teacher]['path']).parent
            refs = spec['teachers'][teacher]['refs']
            if teacher in spec['reuse_banks']:
                cls = offline.Bank if teacher == 'OFFLINE' else u100.Bank
                value = cls(directory, train, refs=refs)
            else:
                value = Bank(directory, train, refs=refs, teacher_id=teacher)
            self.targets[teacher] = value.targets
        check_join(self, train)


def check_join(bank, train):
    validate(bank.manifest, 'BUNDLE')
    if (train.role != 'train' or train.coordinate_name != 'D000'
            or bank.manifest['prepared'] != train.foundation_sha256
            or bank.manifest['role'] != 'train' or bank.manifest['rows'] != len(train)
            or bank.manifest['teacher_order'] != list(TEACHERS)
            or set(bank.targets) != set(TEACHERS)
            or not np.array_equal(bank.identities, train.identities)
            or not np.array_equal(bank.labels, train.labels)):
        raise ValueError('Four-teacher TRAIN join differs')
    for teacher in TEACHERS:
        a = bank.targets[teacher]
        if a.dtype != np.float32 or a.shape != (len(train), 128) or not np.isfinite(a).all():
            raise ValueError('Invalid targets for '+teacher)


def publish(spec, train):
    value = Bundle(spec, train)
    write_immutable_json(Path(spec['root'])/'targets/bundle.json', value.manifest)
    return value


def load(spec, train):
    value = Bundle(spec, train)
    if load_json(Path(spec['root'])/'targets/bundle.json') != value.manifest:
        raise ValueError('Frozen bundle identity changed')
    return value

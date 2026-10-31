"""One shared OFFLINE key bank; original U100 regression targets remain unchanged."""
from pathlib import Path
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays, atomic_publish_bytes,
    deterministic_npz_bytes, write_immutable_json)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_embedding_aux.model import head_input
from hlt_classification.luka_embedding_aux.targets import standardize
from hlt_classification.luka_u100_multiview.views import join
from hlt_classification.luka_u100_embedding.targets import Bank as U100Bank
from .contracts import artifact, validate, sampler_recipe

EXCLUSION_POLICY = dict(event_groups_available=False, alias_groups_available=False,
    reason='Historical authenticated RamCache exposes unique source-row identities, not event/replica groups',
    inferred_distinct_events=False, duplicate_row_id_policy='reject_bank')


def summaries(hlt):
    if hlt.role != 'train' or hlt.coordinate_name != 'D000':
        raise PermissionError('Kinematics must come from real TRAIN HLT tokens')
    pt = np.empty(len(hlt), np.float64); count = np.empty(len(hlt), np.int64)
    for start in range(0, len(hlt), 128):
        ix = np.arange(start, min(start+128, len(hlt)))
        raw = hlt.batch(ix); mask = raw['mask'][:, 0]
        xy = (raw['vectors'][:, :2].astype(np.float64)*mask[:, None]).sum(-1)
        pt[ix] = np.hypot(xy[:, 0], xy[:, 1]); count[ix] = mask.sum(-1)
    return pt, count


def normalized_keys(raw):
    standardized, mean, scale = standardize(raw)
    norms = np.linalg.norm(standardized, axis=1, keepdims=True)
    return (standardized/np.maximum(norms, 1e-12)).astype(np.float32), mean, scale


def publish(directory, raw, hlt, teacher_ref, *, spec_hash, imported_hash):
    """Shared writer used by export and synthetic production-path tests."""
    if hlt.role != 'train' or hlt.coordinate_name != 'D000':
        raise PermissionError('Only TRAIN HLT identities may index an OFFLINE bank')
    validate_file_ref(teacher_ref)
    keys, mean, scale = normalized_keys(raw)
    if keys.shape != (len(hlt), 128):
        raise ValueError('Wrong pre-classifier embedding shape')
    pt, count = summaries(hlt)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    atomic_publish_bytes(directory/'keys.npz', deterministic_npz_bytes(dict(identities=hlt.identities,
        labels=hlt.labels, keys=keys, mean=mean, scale=scale, hlt_pt=pt, hlt_count=count)))
    record = artifact('KEYS', parents=dict(spec=spec_hash, imported=imported_hash,
        prepared=hlt.foundation_sha256, sampler=sampler_recipe()['content_hash']),
        teacher=teacher_ref, role='train', coordinate='OFFLINE', tap='native_preclassifier_128',
        normalization='TRAIN_population_std_floor1e-3_then_L2_eps1e-12',
        rows=len(hlt), width=128, exclusions=EXCLUSION_POLICY,
        payload=file_ref(directory/'keys.npz', root=directory))
    write_immutable_json(directory/'manifest.json', record)
    return record


def export(directory, teacher, offline, hlt, teacher_ref, *, spec_hash, imported_hash, device):
    join(offline, hlt)
    if offline.role != 'train' or offline.coordinate_name != 'OFFLINE':
        raise PermissionError('No validation/test retrieval bank')
    validate_file_ref(teacher_ref)
    teacher.to(device).eval().requires_grad_(False)
    values = np.empty((len(offline), 128), np.float32)
    with torch.inference_mode(), torch.autocast(torch.device(device).type, enabled=False):
        for start in range(0, len(offline), 128):
            raw = offline.batch(np.arange(start, min(start+128, len(offline))))
            inputs = {k: torch.from_numpy(raw[k]).to(device) for k in ('features', 'vectors', 'mask')}
            _, embedding = head_input(teacher, **inputs)
            if not torch.isfinite(embedding).all():
                raise ValueError('Nonfinite OFFLINE embedding')
            values[start:start+len(embedding)] = embedding.float().cpu().numpy()
            if start % (128*128) == 0:
                print(f'OFFLINE keys {start+len(embedding)}/{len(offline)}', flush=True)
    validate_file_ref(teacher_ref)
    return publish(directory, values, hlt, teacher_ref, spec_hash=spec_hash, imported_hash=imported_hash)


class KeyBank:
    def __init__(self, directory, hlt, teacher_ref, *, spec_hash, imported_hash):
        directory = Path(directory); self.manifest = m = load_json(directory/'manifest.json')
        validate(m, 'KEYS', dict(spec=spec_hash, imported=imported_hash, prepared=hlt.foundation_sha256,
                                sampler=sampler_recipe()['content_hash']))
        if (hlt.role != 'train' or hlt.coordinate_name != 'D000' or m['teacher'] != teacher_ref
                or m['role'] != 'train' or m['coordinate'] != 'OFFLINE' or m['width'] != 128
                or m['rows'] != len(hlt) or m['tap'] != 'native_preclassifier_128'
                or m['normalization'] != 'TRAIN_population_std_floor1e-3_then_L2_eps1e-12'
                or m['exclusions'] != EXCLUSION_POLICY or m['payload']['path'] != 'keys.npz'):
            raise ValueError('Wrong OFFLINE key lineage or semantics')
        validate_file_ref(m['payload'], root=directory)
        a = load_npz_arrays(directory/'keys.npz')
        shape = dict(identities=((len(hlt), 32), np.uint8), labels=((len(hlt),), np.int64),
            keys=((len(hlt), 128), np.float32), mean=((128,), np.float64), scale=((128,), np.float64),
            hlt_pt=((len(hlt),), np.float64), hlt_count=((len(hlt),), np.int64))
        if set(a) != set(shape) or any(a[k].shape != s or a[k].dtype != d or not np.isfinite(a[k]).all()
                                     for k, (s, d) in shape.items()):
            raise ValueError('Invalid key array format/values')
        if (not np.array_equal(a['identities'], hlt.identities) or not np.array_equal(a['labels'], hlt.labels)
                or len(np.unique(a['identities'], axis=0)) != len(hlt) or np.any(a['scale'] < 1e-3)
                or np.any(a['hlt_pt'] < 0) or np.any(a['hlt_count'] < 0)):
            raise ValueError('Key ordered TRAIN join or normalization differs')
        norms = np.linalg.norm(a['keys'], axis=1)
        if not np.all((norms < 1e-8) | np.isclose(norms, 1., atol=2e-6)):
            raise ValueError('Non-unit matching key')
        for k, v in a.items():
            setattr(self, k, v)


def u100_bank(spec, train):
    bundle = load_json(Path(spec['root'])/'setup/bundle.json')
    validate(bundle, 'BUNDLE', dict(spec=spec['content_hash']))
    location = Path(bundle['u100_manifest']['path'])
    validate_file_ref(bundle['u100_manifest'])
    return U100Bank(location.parent, train, teacher_record=spec['imported']['teacher'])


def key_bank(spec, train):
    return KeyBank(Path(spec['root'])/'setup/OFFLINE', train,
        spec['anchor']['evidence']['models']['OFFLINE']['checkpoint'],
        spec_hash=spec['content_hash'], imported_hash=spec['imported']['content_hash'])

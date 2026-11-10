"""Authenticated HLT inputs, original U100 regression and TRAIN-only subclasses."""
from pathlib import Path
import numpy as np
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays
from hlt_classification.luka_u100_pair_hydra.inputs import Bundle, load_view, embedding_bank
from hlt_classification.luka_embedding_aux import targets as offline_targets
from .contracts import cell
from . import reuse


def validate_sources(spec):
    reuse.check_refs(spec['teacher_refs']); refs = spec['teacher_refs']
    m = load_json(refs['bank']['path'])
    offline_targets.validate(m, 'BANK', dict(recipe=offline_targets.recipe()['content_hash'],
        prepared=spec['anchor']['evidence']['prepared'], teacher=refs['checkpoint']['sha256']))
    if (m['role'] != 'train' or m['coordinate'] != 'OFFLINE' or m['width'] != 128
            or m['rows'] != spec['counts']['train'] or m['teacher'] != refs['checkpoint']
            or m['payload']['path'] != 'targets.npz'
            or m['payload']['sha256'] != refs['targets']['sha256']
            or Path(refs['bank']['path']).parent/'targets.npz' != Path(refs['targets']['path'])):
        raise ValueError('T00 target population or ownership differs')
    m = Bundle(spec['bundle_root']).json('targets/u100/manifest.json')
    if (m['content_hash'] != spec['embedding_hash'] or m['role'] != 'train' or m['coordinate'] != 'U100'
            or m['parents']['prepared'] != spec['anchor']['evidence']['prepared']
            or m['parents']['teacher'] != spec['imported']['teacher']['content_hash']):
        raise ValueError('Original U100 TRAIN lineage differs')


def offline_bank(spec, train):
    validate_sources(spec)
    refs = spec['teacher_refs']
    return offline_targets.Bank(Path(refs['bank']['path']).parent, train, teacher_ref=refs['checkpoint'])


def views(spec, task):
    cell(task)
    train, val = (load_view(spec, role, 'D000') for role in ('train','validation'))
    reference = load_npz_arrays(Bundle(spec['bundle_root']).path('reference/matched/predictions.npz'))
    if any(not np.array_equal(reference[k], getattr(val,k)) for k in ('identities','labels')):
        raise ValueError('Validation population differs')
    return train, val


def banks(spec, task, train):
    from .targets import Bank
    cell(task)
    result = dict(u100=embedding_bank(spec, train), subclass=Bank(spec, train))
    if result['u100'].manifest['content_hash'] != spec['embedding_hash']:
        raise ValueError('U100 bank differs')
    return result


def join(train, banks):
    if train.role != 'train' or train.coordinate_name != 'D000' or set(banks) != {'u100','subclass'}:
        raise PermissionError('Only D000 TRAIN with U100 and subclass targets is authorized')
    for key, width, coordinate in (('u100',128,'U100'),('subclass',10,'OFFLINE')):
        b = banks[key]
        if (b.manifest['role'] != 'train' or b.manifest['coordinate'] != coordinate
                or b.targets.shape != (len(train),width) or b.targets.dtype != np.float32
                or any(not np.array_equal(getattr(train,k),getattr(b,k)) for k in ('identities','labels'))
                or not np.isfinite(b.targets).all()):
            raise PermissionError('TRAIN target role/coordinate/identity join differs')
    q = banks['subclass'].targets
    if (np.any(q < 0) or not np.allclose(q.sum(1),1,rtol=0,atol=2e-6)
            or np.any(q[train.labels!=0,5:] != 0)):
        raise ValueError('Invalid conditional subclass probabilities')


def fit_parents(spec, task):
    from .targets import check_manifest
    cell(task)
    return dict(spec=spec['content_hash'], recipe=spec['recipe']['content_hash'],
        u100=spec['embedding_hash'], subclass=check_manifest(spec)['content_hash'])

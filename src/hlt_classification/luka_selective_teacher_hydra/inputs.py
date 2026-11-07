"""Ordered native input joins and role-restricted frozen target banks."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.luka_u100_pair_hydra.inputs import Bundle, load_view, embedding_bank
from hlt_classification.luka_embedding_aux import targets as offline_targets
from .transfer import selected
from .contracts import TEACHERS, STUDENTS, artifact
from . import views as selective, targets


def delta(spec):
    return selected(spec['validation_kind'], spec['validation_root'], spec['validation_hash'], spec['bundle_root'])


def views(spec, task, *, publish=False):
    if task not in (*TEACHERS, *STUDENTS): raise ValueError('Unknown input task')
    caches = []
    for role in ('train', 'validation'):
        hlt = load_view(spec, role, 'D000')
        if task in TEACHERS:
            u100 = load_view(spec, role, 'U100') if role == 'train' else delta(spec).load(spec, role)
            cache = selective.hybrid(hlt, u100, task)
            del hlt, u100
            proof = selective.fingerprint(cache)
            path = Path(spec['root'])/'setup'/f'view_{task}_{role}.json'
            if publish: write_immutable_json(path, proof)
            elif load_json(path) != proof: raise ValueError('Selective input fingerprint changed after setup')
        else:
            cache = hlt
        if len(cache) != spec['counts'][role]: raise ValueError('Input population differs')
        caches.append(cache)
    return tuple(caches)


def banks(spec, task, train, *, technical=False):
    if task in TEACHERS: return {}
    if task not in STUDENTS: raise ValueError('Unknown student')
    from .worker import validate_teacher
    result = dict(u100=embedding_bank(spec, train))
    for name in STUDENTS[task]:
        if name == 'OFFLINE':
            ref = spec['reused']['offline_checkpoint']
            result['offline'] = offline_targets.Bank(Path(spec['reused']['offline_bank']['path']).parent,
                                                     train, teacher_ref=ref)
        elif technical:
            result[name.lower()] = SimpleNamespace(targets=np.zeros((len(train), 128), np.float32),
                identities=train.identities, labels=train.labels,
                manifest=artifact('TECHNICAL_BANK', role='train', coordinate=name, real_teacher=False))
        else:
            validate_teacher(spec, name)
            folder = Path(spec['root'])/name
            result[name.lower()] = targets.Bank(folder/'bank', train, coordinate=name,
                teacher_ref=file_ref(folder/'fit/selected.pt'), spec_hash=spec['content_hash'])
    return result

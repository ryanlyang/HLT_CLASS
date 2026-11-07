"""HLT-only student inputs; TRAIN-only target joins; native OFFLINE teacher views."""
from pathlib import Path
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_u100_pair_hydra.inputs import Bundle, load_view, embedding_bank
from hlt_classification.luka_embedding_aux import targets as offline_targets
from .transfer import Delta
from .contracts import TEACHERS, STUDENTS


def delta(spec):
    return Delta(spec['offline_root'], spec['offline_hash'], spec['bundle_root'])


def views(spec, teacher):
    if teacher:
        d = delta(spec)
        return tuple(d.load(spec, role) for role in ('train', 'validation'))
    return tuple(load_view(spec, role, 'D000') for role in ('train', 'validation'))


def banks(spec, task, train):
    if task in TEACHERS:
        return {}
    if task not in STUDENTS:
        raise ValueError('Unknown student')
    from .worker import validate_teacher
    teacher = STUDENTS[task]; validate_teacher(spec, teacher)
    folder = Path(spec['root'])/teacher
    return dict(u100=embedding_bank(spec, train), offline=offline_targets.Bank(folder/'bank', train,
        teacher_ref=file_ref(folder/'fit/selected.pt')))

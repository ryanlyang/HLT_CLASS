"""Authenticated old HLT/U100 reuse and native OFFLINE TRAIN/validation views."""
from pathlib import Path
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_u100_pair_correction.targets import load_view, metadata, embedding_bank
from hlt_classification.luka_u100_multiview import offline_cache
from hlt_classification.luka_embedding_aux import targets as offline_targets
from .contracts import TEACHERS, STUDENTS


def views(spec, teacher):
    if teacher:
        meta = metadata(spec)
        return tuple(offline_cache.load(spec['anchor']['evidence'], role, meta,
            int(spec['memory_mb']*1024**2*.30)) for role in ('train', 'validation'))
    return tuple(load_view(spec, role, 'D000') for role in ('train', 'validation'))


def banks(spec, task, train):
    if task in TEACHERS:
        return {}
    if task not in STUDENTS:
        raise ValueError('Unknown student')
    from .worker import receipt, validate_teacher
    teacher = STUDENTS[task]
    receipt(spec, teacher); validate_teacher(spec, teacher)
    folder = Path(spec['root'])/teacher
    return dict(u100=embedding_bank(spec, train), offline=offline_targets.Bank(folder/'bank', train,
        teacher_ref=file_ref(folder/'fit/selected.pt')))

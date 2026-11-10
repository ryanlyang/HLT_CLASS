"""Authenticated D000 inputs; CE-only never opens teacher target banks."""
from pathlib import Path
import numpy as np
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
from hlt_classification.luka_u100_pair_hydra.inputs import Bundle, load_view, embedding_bank
from hlt_classification.luka_selective_teacher_hydra.targets import Bank
from .contracts import cell, weights


def views(spec, task):
    cell(task)
    train, val = (load_view(spec, r, 'D000') for r in ('train', 'validation'))
    reference = load_npz_arrays(Bundle(spec['bundle_root']).path('reference/matched/predictions.npz'))
    if any(not np.array_equal(reference[k], getattr(val, k)) for k in ('identities', 'labels')):
        raise ValueError('Validation population differs from original references')
    return train, val


def banks(spec, task, train):
    names = weights(task)
    result = {}
    if 'u100' in names:
        result['u100'] = embedding_bank(spec, train)
    if 'it' in names:
        for key in ('it_checkpoint', 'it_bank', 'it_targets', 'it_receipt'):
            validate_file_ref(spec['reused'][key])
        result['it'] = Bank(Path(spec['reused']['it_bank']['path']).parent, train,
                            teacher_ref=spec['reused']['it_checkpoint'], coordinate='IT', spec_hash=spec['parent_hash'])
    return result


def fit_parents(spec, task):
    parents = dict(spec=spec['content_hash'], recipe=spec['recipe']['content_hash'])
    if 'u100' in weights(task):
        parents['u100'] = Bundle(spec['bundle_root']).json('targets/u100/manifest.json')['content_hash']
    if 'it' in weights(task):
        validate_file_ref(spec['reused']['it_bank'])
        parents['it'] = load_json(spec['reused']['it_bank']['path'])['content_hash']
    return parents

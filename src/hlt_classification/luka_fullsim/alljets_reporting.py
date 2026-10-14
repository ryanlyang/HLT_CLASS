"""Full-validation selected checkpoints, plus a fixed historical 50k comparison."""
from pathlib import Path

import numpy as np

from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, deterministic_npz_bytes, load_json, load_npz_arrays, write_immutable_json,
)
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from . import alljets_data as data


def population(spec):
    from . import alljets_campaign as c
    foundation, inventory, splits = data.load_foundation(Path(spec['gate_path']).parent/'foundation')
    c.check_population(inventory, splits)
    if foundation['source_snapshot'] != spec['source_snapshot']:
        raise ValueError('Validation diagnostic foundation source differs')
    return inventory, splits


def describe(spec, task_id, report, arrays, *, authenticated_population=None):
    if task_id != 'train_'+report['node']['node_id']:
        raise ValueError('Validation diagnostic task differs')
    inventory, splits = population(spec) if authenticated_population is None else authenticated_population
    if (set(arrays) != {'identities', 'labels', 'probabilities'}
            or arrays['identities'].dtype != np.uint8 or arrays['labels'].dtype != np.int64
            or arrays['probabilities'].dtype != np.float32
            or arrays['probabilities'].shape != (spec['counts']['validation'], 11)):
        raise ValueError('Validation probability payload schema differs')
    indices = data.comparison_indices(inventory, splits, arrays['identities'], arrays['labels'])
    directory = Path(spec['root'])/task_id
    return data.artifact('VALIDATION_DIAGNOSTICS',
        parents=dict(spec=spec['content_hash'], training_report=report['content_hash'], splits=splits['content_hash']),
        task_id=task_id, checkpoint=file_ref(directory/'selected.pt', root=Path(spec['root'])),
        predictions=file_ref(directory/'validation_predictions.npz', root=Path(spec['root'])),
        selection_population='entire_validation', secondary_selection_used=False, temperature=1.,
        validation=evaluate_probabilities(arrays['labels'], arrays['probabilities']),
        original_validation=evaluate_probabilities(arrays['labels'][indices], arrays['probabilities'][indices]))


def publish(spec, task_id, report, validation, probabilities):
    arrays = dict(identities=validation.identities, labels=validation.labels, probabilities=probabilities)
    directory = Path(spec['root'])/task_id
    # Validate identities before publishing. No predictions may enter TRAIN banks.
    inventory, splits = population(spec)
    data.comparison_indices(inventory, splits, arrays['identities'], arrays['labels'])
    atomic_publish_bytes(directory/'validation_predictions.npz', deterministic_npz_bytes(arrays))
    result = describe(spec, task_id, report, arrays, authenticated_population=(inventory, splits))
    write_immutable_json(directory/'validation_diagnostics.json', result)
    return validate(spec, task_id, report, authenticated_population=(inventory, splits))


def validate(spec, task_id, report, *, authenticated_population=None):
    root = Path(spec['root'])
    result = load_json(root/task_id/'validation_diagnostics.json')
    data.validate(result, 'VALIDATION_DIAGNOSTICS')
    for key, name in (('checkpoint', 'selected.pt'), ('predictions', 'validation_predictions.npz')):
        if result[key]['path'] != task_id+'/'+name:
            raise ValueError('Validation diagnostic output scope differs')
        validate_file_ref(result[key], root=root)
    arrays = load_npz_arrays(root/task_id/'validation_predictions.npz')
    if describe(spec, task_id, report, arrays, authenticated_population=authenticated_population) != result:
        raise ValueError('Validation diagnostic metrics/lineage replay differs')
    return result

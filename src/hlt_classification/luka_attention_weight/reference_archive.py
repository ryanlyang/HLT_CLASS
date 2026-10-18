"""Read-only comparison recovery, executed against the ORIGINAL pinned runtime.

This file is loaded by absolute path with runpy. Imports deliberately resolve
to the original attention worktree, not the new weight-sweep package. It does
not certify old checkpoints, target banks, or preflight admission for reuse.
"""
from pathlib import Path, PurePosixPath
import re

import numpy as np

from hlt_classification.data.cache_contracts import load_json, load_npz_arrays
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from hlt_classification.luka_attention import campaign, worker
from hlt_classification.luka_attention.contracts import validate
from hlt_classification.luka_fullsim import regular_cache, alljets_data


REFERENCE_COMMIT = '94f560259dd04253c0aa47b31dec3aedc105ac14'
SCOPE = dict(reuse_scope='archival_comparison_only', metrics_recomputed=True,
             legacy_preflight_reusable=False, model_or_target_reuse_authorized=False,
             checkpoint_and_target_payloads_revalidated=False)


def _digest(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Invalid archived lineage hash')
    return value


def _outputs(receipt, task):
    """Validate descriptors without touching discarded payload paths."""
    result = {}
    for ref in receipt['outputs']:
        name = ref['path']
        parts = name.split('/')
        if (PurePosixPath(name).is_absolute() or parts[0] != task
                or any(p in ('', '.', '..') for p in parts) or '\\' in name or ':' in name
                or name in result or type(ref['bytes']) is not int or ref['bytes'] < 0):
            raise ValueError('Invalid archived output inventory')
        _digest(ref['sha256'])
        result[name] = ref
    return result


def read_archive(path):
    """Verify retained results only; never call check_spec/completed/evidence."""
    checked = {}

    def remember(path):
        ref = file_ref(path)
        if ref['path'] in checked and checked[ref['path']] != ref:
            raise ValueError('Reference changed during archival verification')
        checked[ref['path']] = ref
        return ref

    def read(path):
        remember(path)
        return load_json(path)

    spec = read(path)
    gate = read(spec['gate_path'])
    # Still authenticate original source, recipe, shared U100/controls and data
    # metadata. This API does NOT read the old technical banks.
    campaign.check_gate(gate)
    root = Path(spec['root'])
    preflight = read(Path(gate['root'])/'preflight/receipt.json')
    validate(preflight, 'PREFLIGHT', dict(gate=gate['content_hash']))
    validate(spec, 'SPEC', dict(gate=gate['content_hash'], preflight=preflight['content_hash']))
    if (Path(path).resolve() != (root/'campaign_spec.json').resolve()
            or root != Path(gate['root'])/'science'
            or Path(spec['gate_path']) != Path(gate['root'])/'gate_spec.json'
            or spec['project_dir'] != gate['project_dir']
            or spec['source_commit'] != gate['source_commit']
            or spec['source_commit'] != REFERENCE_COMMIT
            or spec['train_minutes'] != preflight['train_minutes']
            or spec['export_minutes'] != preflight['export_minutes']
            or preflight['technical_checks_passed'] is not True
            or preflight['scientific_fit'] is not False):
        raise ValueError('Archived science/preflight registration differs')

    meta = regular_cache.metadata(gate['anchor']['evidence']['prepared_root'])
    if (meta[0]['content_hash'] != gate['anchor']['evidence']['cache']
            or meta[1]['content_hash'] != gate['anchor']['evidence']['prepared']):
        raise ValueError('Archived reference population differs')

    aggregate = read(root/'aggregate/receipt.json')
    validate(aggregate, 'TASK', dict(spec=spec['content_hash']))
    task = next(t for t in gate['tasks'] if t['task_id'] == 'aggregate')
    if (aggregate['task_id'] != 'aggregate'
            or set(aggregate['dependencies']) != set(task['dependencies'])):
        raise ValueError('Archived aggregate dependency coverage differs')
    for digest in aggregate['dependencies'].values():
        _digest(digest)

    def bound(outputs, name):
        if name not in outputs:
            raise ValueError('Missing bound archival output: '+name)
        target = validate_file_ref(outputs[name], root=root)
        remember(target)
        return target

    result = read(bound(_outputs(aggregate, 'aggregate'), 'aggregate/results.json'))
    validate(result, 'RESULTS', dict(spec=spec['content_hash']))
    rows = []
    for name in ('M0HLT', 'OFFLINE', 'DIRECT_D000', 'COARSE_U100'):
        refs = gate['anchor']['evidence']['models'][name]
        # These controls are still live dependencies, authenticated by check_gate.
        for ref in refs.values():
            validate_file_ref(ref)
        report = read(refs['report']['path'])
        diag = read(refs['diagnostics']['path'])
        rows.append(dict(node_id=name, origin='parent_fixed30', validation=report['validation'],
                         original_validation=diag['original_validation']))

    predictions, reports, receipts = {}, {}, {}
    for node in gate['nodes']:
        name = node['node_id']
        task_name = 'train_'+name
        receipt = read(root/task_name/'receipt.json')
        validate(receipt, 'TASK', dict(spec=spec['content_hash']))
        task = next(t for t in gate['tasks'] if t['task_id'] == task_name)
        if (receipt['task_id'] != task_name
                or receipt['content_hash'] != aggregate['dependencies'][task_name]
                or receipt['dependencies'] != {
                    dep: aggregate['dependencies'][dep] for dep in task['dependencies']}):
            raise ValueError('Archived training/dependency lineage differs')
        outputs = _outputs(receipt, task_name)
        if task_name+'/selected.pt' not in outputs:
            raise ValueError('Missing recorded selected checkpoint identity')
        report = read(bound(outputs, task_name+'/training_report.json'))
        # This bank identity is RECORDED lineage, not a claim its payload survives.
        bank_hash = _digest(report['parents']['bank'])
        worker.report_valid(report, node, bank_hash, meta[1]['content_hash'], gate['counts']['validation'])
        pred_path = bound(outputs, task_name+'/validation_predictions.npz')
        p = load_npz_arrays(pred_path)
        if (set(p) != {'identities', 'labels', 'probabilities'}
                or p['probabilities'].shape != (gate['counts']['validation'], 11)
                or p['labels'].dtype != np.int64 or p['identities'].dtype != np.uint8
                or p['identities'].shape != (gate['counts']['validation'], 32)):
            raise ValueError('Archived prediction schema differs')
        # Checks exact FULL validation order/labels against the registered split,
        # then returns the fixed original50k report-only indices. No particles.
        ix = alljets_data.comparison_indices(meta[2], meta[3], p['identities'], p['labels'])
        metrics = evaluate_probabilities(p['labels'], p['probabilities'])
        if metrics != report['validation']:
            raise ValueError('Archived predictions disagree with selected report')
        rows.append(dict(node_id=name, origin='new', validation=metrics,
            original_validation=evaluate_probabilities(p['labels'][ix], p['probabilities'][ix]),
            costs=read(bound(outputs, task_name+'/costs.json'))))
        predictions[name] = remember(pred_path)
        reports[name] = report['content_hash']
        receipts[task_name] = receipt['content_hash']

    for row in rows:
        row['recovery'] = {field: recovery(row[field], rows[0][field], rows[1][field])
                           for field in ('validation', 'original_validation')}
    by_name = {r['node_id']: r for r in rows}
    contrasts = {field: {metric: by_name['ATTN_D000'][field][metric]-by_name['LOGIT_D000'][field][metric]
                        for metric in ('accuracy', 'macro_ovr_auc', 'cross_entropy')}
                 for field in ('validation', 'original_validation')}
    if (result['rows'] != rows or result['contrasts'] != contrasts
            or result['primary_contrast'] != 'ATTN_D000 minus LOGIT_D000'
            or result['single_seed'] is not True):
        raise ValueError('Archived aggregate differs from retained predictions/reports')
    for ref in checked.values():
        validate_file_ref(ref)
    return dict(rows=rows, predictions=predictions, parent=gate['anchor']['content_hash'],
        prepared=meta[1]['content_hash'], anchor=gate['anchor'],
        archive=dict(**SCOPE, source_commit=spec['source_commit'], spec=spec['content_hash'],
            gate=gate['content_hash'], preflight=preflight['content_hash'],
            aggregate=aggregate['content_hash'], results=result['content_hash'],
            training_reports=reports, training_receipts=receipts,
            recorded_dependencies=aggregate['dependencies'], verified_files=list(checked.values()),
            not_revalidated=['legacy_preflight_outputs', 'legacy_science_checkpoints',
                             'legacy_target_banks_and_export_payloads', 'legacy_phase_logs']))

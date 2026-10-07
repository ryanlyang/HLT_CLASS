"""Authenticate historical pilot evidence without substituting its source root."""
from pathlib import Path

from hlt_classification.correlated_tracking import campaign as pilot, worker
from hlt_classification.correlated_tracking.contracts import validate
from hlt_classification.correlated_tracking.kernel import recipe
from hlt_classification.literature_proxy_v2.inputs import authenticate
from .contracts import load_json, sha256_file, safe

APPROVED_PILOT_COMMIT = '0667355d8c4d72e48e32c03f2d1540595fe654b7'
APPROVED_PILOT_JOB = '228972'


def require_frozen_code(current, original):
    prefixes = ('src/hlt_classification/correlated_tracking/',
                'src/hlt_classification/literature_proxy/',
                'src/hlt_classification/literature_proxy_v2/')
    exact = {'src/hlt_classification/cms2jc2_response/bridge.py',
             'src/hlt_classification/literature_context/transform.py',
             'src/hlt_classification/cms_proxy_ladder/inputs.py',
             'src/hlt_classification/cms_proxy_ladder/contracts.py',
             'src/hlt_classification/jetclass2_delphes/inputs.py'}
    for name in set(current['file_sha256']) | set(original['file_sha256']):
        if name.startswith(prefixes) or name in exact:
            if current['file_sha256'].get(name) != original['file_sha256'].get(name):
                raise ValueError('Frozen pilot implementation changed: '+name)


def authenticate_pilot(path):
    path = Path(path).resolve(strict=True)
    spec = load_json(path)
    validate(spec, 'SPEC')
    root = Path(spec['root']).resolve(strict=True)
    if path != root/'study_spec.json' or spec['source']['commit'] != APPROVED_PILOT_COMMIT:
        raise PermissionError('Only the explicitly approved correlated pilot may be frozen')
    # Its immutable submission binds the known real pilot job to the exact
    # spec/plan hash; no guessed hash or caller-supplied approval override.
    ledger = load_json(root/'submission_ledger.json')
    validate(ledger, 'LEDGER')
    plan = pilot.plan(spec)
    if (ledger.get('job') != APPROVED_PILOT_JOB or ledger.get('final_test_accessed') is not False
            or ledger['parents'] != dict(spec=spec['content_hash'], plan=plan['content_hash'])
            or load_json(root/'command_plan.json') != plan):
        raise PermissionError('Approved pilot job/plan binding differs')
    if (spec['recipe'] != recipe() or spec['resources'] != pilot.RESOURCES
            or any(spec.get(k) is not False for k in pilot.FLAGS)
            or pilot.source(spec['project_dir'], spec['source']['commit']) != spec['source']):
        raise ValueError('Historical pilot source/recipe/scope differs')
    original, records = authenticate(spec['parent_spec'], spec['parent_receipt'])
    if (spec['inputs'] != records or spec['population'] != original['population']
            or not 1 <= spec['population']['jets'] <= 20_000):
        raise ValueError('Historical training population differs')
    receipt = load_json(root/'receipt.json')
    validate(receipt, 'RECEIPT')
    if (receipt['parents'] != dict(spec=spec['content_hash'])
            or set(receipt['outputs']) != worker.required_outputs(spec)
            or receipt.get('final_test_accessed') is not False):
        raise ValueError('Historical completion receipt differs')
    for name, digest in receipt['outputs'].items():
        if sha256_file(safe(root, name)) != digest:
            raise ValueError('Historical output checksum differs: '+name)
    report = load_json(root/'report.json')
    validate(report, 'REPORT')
    if (report['parents'] != dict(spec=spec['content_hash']) or report['recipe'] != recipe()
            or report['jets'] != spec['population']['jets'] or report['structure_exact'] is not True
            or report['replay']['exact'] is not True or report['no_classifier_fit'] is not True
            or any(report.get(k) is not False for k in pilot.FLAGS)):
        raise ValueError('Historical report coverage/replay/scope differs')
    files = []
    for record in records:
        key = worker.file_key(record)
        row = load_json(root/'files'/f'{key}.json')
        validate(row, 'FILE_REPORT')
        block = f'blocks/{key}.npz'
        if (row['input'] != record or row['parents'] != dict(spec=spec['content_hash'])
                or row['jets'] != len(record['selected']['identities'])
                or row['structure_exact'] is not True
                or row['outputs'] != {block: receipt['outputs'][block]}):
            raise ValueError('Historical per-file evidence differs')
        files.append(row)
    if (sum(r['jets'] for r in files) != report['jets']
            or sum(r['particles'] for r in files) != report['particles']):
        raise ValueError('Historical aggregate coverage differs')
    return spec, report, original

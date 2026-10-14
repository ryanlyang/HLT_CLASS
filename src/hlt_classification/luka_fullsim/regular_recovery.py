"""Preflight-only recovery of the exact a54c60e5 preparation; no source/data rewrite."""
from pathlib import Path
import ast
import re
import subprocess

from hlt_classification.data.cache_contracts import load_json, sha256_file, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from . import regular_campaign as c, regular_cache as cache
from .preflight_reuse import _files

ORIGINAL_COMMIT = 'a54c60e5354f61b4c065f9322d89dbb4f6cf8dc3'
ADDITIONS = {'src/hlt_classification/luka_fullsim/regular_recovery.py'}
CHANGES = {'src/hlt_classification/luka_fullsim/regular_'+n+'.py' for n in ('campaign', 'preflight', 'submission')}
FROZEN_DECLARATIONS = {'COUNTS', 'INVENTORY', 'SPLITS', 'nodes', 'tasks', 'registration',
                       'conventions', 'check_population', 'gate_policy'}


def scientific_declarations(project):
    tree = ast.parse((Path(project)/'src/hlt_classification/luka_fullsim/regular_campaign.py').read_text())
    result = {}
    for node in tree.body:
        names = ([node.name] if isinstance(node, ast.FunctionDef) else
                 [target.id for target in node.targets if isinstance(target, ast.Name)] if isinstance(node, ast.Assign) else [])
        for name in set(names) & FROZEN_DECLARATIONS:
            result[name] = ast.dump(node, include_attributes=False)
    if set(result) != FROZEN_DECLARATIONS:
        raise ValueError('Missing frozen scientific declarations')
    return result


def compatible_source(old, active, project):
    before_root, after_root = Path(old['project_dir']), Path(project)
    validate_source_snapshot(old['source_snapshot'], repository=before_root, require_clean=True)
    validate_source_snapshot(active, repository=after_root, require_clean=True)
    before, after = _files(before_root), _files(after_root)
    if not before or not before <= after or after-before != ADDITIONS:
        raise ValueError('Unregistered recovery source additions/deletions')
    if scientific_declarations(before_root) != scientific_declarations(after_root):
        raise ValueError('Registered science/population/conventions changed')
    records = {}
    for name in sorted(before):
        prior, current = sha256_file(before_root/name), sha256_file(after_root/name)
        if prior != current and name not in CHANGES:
            raise ValueError('Prepared/scientific source changed: '+name)
        records[name] = dict(preparation=prior, execution=current)
    return c.make_artifact('SOURCE_REUSE', parents=dict(preparation=old['source_snapshot']['content_hash'],
        execution=active['content_hash']), modules=records, added_modules=sorted(ADDITIONS), allowed_changes=sorted(CHANGES))


def original(gate_path, active, project):
    path = Path(gate_path).resolve(strict=True)
    old = load_json(path)
    c.validate(old, 'GATE', dict(source=old['source_snapshot']['content_hash']))
    if (old['source_commit'] != ORIGINAL_COMMIT or old['source_snapshot']['git_commit'] != ORIGINAL_COMMIT
            or path != Path(old['root']).resolve()/'gate_spec.json'
            or any(old.get(k) != v for k, v in c.gate_policy().items())):
        raise ValueError('Recovery requires the original a54c60e5 gate and unchanged policy')
    reuse = compatible_source(old, active, project)
    root = Path(old['root'])
    if (root/'science').exists() or (root/'preflight/preflight.json').exists():
        raise PermissionError('Original campaign already has science/accepted preflight; do not duplicate')
    ledger = c.gate_ledger(old, historical=True)
    manifest, p, inventory, splits = cache.metadata(root/'prepared')
    f, _, _ = c.load_foundation(root/'foundation')
    c.check_population(inventory, splits)
    if (manifest['parents'] != dict(prepared=p['content_hash'], gate=old['content_hash'])
            or p['source_snapshot'] != old['source_snapshot'] or f['source_snapshot'] != old['source_snapshot']
            or p['foundation_root'] != str(root/'foundation') or f['input_container'] != old['input_container']
            or p['conventions'] != c.conventions(p['audit']['content_hash'])):
        raise ValueError('Reused preparation lineage differs')
    return old, ledger, manifest, p, reuse


def failed_accounting(job):
    if re.fullmatch(r'[1-9][0-9]*', str(job)) is None:
        raise ValueError('Invalid original preflight job ID')
    result = subprocess.run(['sacct', '-j', str(job), '-X', '-n', '-P', '-o', 'JobIDRaw,State,ExitCode'],
        check=True, capture_output=True, text=True)
    rows = [line.strip().split('|') for line in result.stdout.splitlines() if line.strip()]
    matches = [r for r in rows if r[0] == str(job)]
    if len(matches) != 1 or matches[0][1:3] != ['FAILED', '1:0']:
        raise PermissionError('Original preflight must be terminal FAILED/1:0; never replace a live job')
    return dict(job_id=str(job), state='FAILED', exit_code='1:0')


def check_location(root, old, project):
    root = Path(root).resolve()
    for protected in (old['root'], old['input_container'], old['project_dir'], project):
        path = Path(protected).resolve()
        if root.is_relative_to(path) or path.is_relative_to(root):
            raise PermissionError('Recovery root overlaps original data/artifacts/source')


def create(*, original_gate, root, project, commit):
    active = c.source(project, commit)
    old, ledger, manifest, p, source_reuse = original(original_gate, active, project)
    accounting = {k: c.preflight_accounting(ledger['jobs'][k]) for k in ('foundation', 'prepare')}
    accounting['preflight'] = failed_accounting(ledger['jobs']['preflight'])
    log = Path(old['root'])/('preflight-'+ledger['jobs']['preflight']+'.out')
    if 'Validation support exceeds TRAIN stress; register a new envelope' not in log.read_text():
        raise ValueError('This recovery requires the diagnosed validation-support failure')
    root, project = Path(root).resolve(), Path(project).resolve()
    check_location(root, old, project)
    value = c.make_artifact('RECOVERY_GATE', parents=dict(source=active['content_hash'], original_gate=old['content_hash'],
        prepared=p['content_hash'], cache=manifest['content_hash'], source_reuse=source_reuse['content_hash']),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=active, input_container=old['input_container'],
        reuse=dict(original_gate=file_ref(Path(original_gate).resolve()), prepared_root=str(Path(old['root'])/'prepared'),
            prepared_sha256=p['content_hash'], cache_sha256=manifest['content_hash'], source_reuse=source_reuse,
            accounting=accounting, failed_log=file_ref(log)), **c.gate_policy())
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root/'gate_spec.json', value)
    return value


def validate_gate(gate):
    c.validate(gate, 'RECOVERY_GATE')
    if (any(gate.get(k) != v for k, v in c.gate_policy().items())
            or c.source(gate['project_dir'], gate['source_commit']) != gate['source_snapshot']
            or load_json(Path(gate['root'])/'gate_spec.json') != gate):
        raise ValueError('Recovery source/spec/policy differs')
    reuse = gate['reuse']
    validate_file_ref(reuse['original_gate'])
    old, ledger, manifest, p, source_reuse = original(reuse['original_gate']['path'], gate['source_snapshot'], gate['project_dir'])
    check_location(gate['root'], old, gate['project_dir'])
    expected_parents = dict(source=gate['source_snapshot']['content_hash'], original_gate=old['content_hash'],
        prepared=p['content_hash'], cache=manifest['content_hash'], source_reuse=source_reuse['content_hash'])
    accounting = {k: dict(job_id=ledger['jobs'][k], state='COMPLETED', exit_code='0:0') for k in ('foundation', 'prepare')}
    accounting['preflight'] = dict(job_id=ledger['jobs']['preflight'], state='FAILED', exit_code='1:0')
    log = Path(old['root'])/('preflight-'+ledger['jobs']['preflight']+'.out')
    if (gate['parents'] != expected_parents or gate['input_container'] != old['input_container']
            or reuse != dict(original_gate=file_ref(Path(old['root'])/'gate_spec.json'), prepared_root=str(Path(old['root'])/'prepared'),
                prepared_sha256=p['content_hash'], cache_sha256=manifest['content_hash'], source_reuse=source_reuse,
                accounting=accounting, failed_log=file_ref(log))):
        raise ValueError('Recovery artifact/source/accounting bindings differ')
    return gate

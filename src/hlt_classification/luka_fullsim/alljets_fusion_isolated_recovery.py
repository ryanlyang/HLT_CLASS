"""Immutable v3 process-isolation recovery from failed synchronous gate21898311."""
from pathlib import Path
import json
import os
import subprocess
import sys

from hlt_classification.data.cache_contracts import load_json, sha256_file, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from . import alljets_fusion_campaign as c, alljets_fusion_source as sources
from .preflight_reuse import _files
from .regular_recovery import failed_accounting

FAILED_COMMIT = 'f979171c40c0280ec307c4ed585889328b86732a'
FAILED_JOB = '21898311'
ADDITIONS = {'src/hlt_classification/luka_fullsim/alljets_fusion_isolated_'+n+'.py'
             for n in ('preflight', 'recovery')}
CHANGES = {'src/hlt_classification/luka_fullsim/alljets_fusion_'+n+'.py'
           for n in ('source', 'campaign', 'worker', 'preflight', 'submission')}
ORIGINAL_CHECK = '''
import json,sys
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullsim import alljets_fusion_campaign as c
from hlt_classification.luka_fullsim.alljets_fusion_submission import accepted_jobs
g=load_json(sys.argv[1]); c.validate_gate(g)
print('LUKA_ISOLATED_AUTH='+json.dumps(dict(gate=g['content_hash'],jobs=accepted_jobs(g))))
'''


def compatible_source(old, active, project):
    before_root, after_root = Path(old['project_dir']), Path(project)
    validate_source_snapshot(old['source_snapshot'], repository=before_root, require_clean=True)
    validate_source_snapshot(active, repository=after_root, require_clean=True)
    before, after = _files(before_root), _files(after_root)
    if not before or not before <= after or after-before != ADDITIONS:
        raise ValueError('Unregistered isolated-recovery source additions/deletions')
    records = {}
    for name in sorted(before):
        prior, current = sha256_file(before_root/name), sha256_file(after_root/name)
        if prior != current and name not in CHANGES:
            raise ValueError('Scientific/model/cache source changed: '+name)
        records[name] = dict(original=prior, execution=current)
    return dict(modules=records, additions=sorted(ADDITIONS), allowed_changes=sorted(CHANGES))


def authenticate(path, active, project):
    path = Path(path).resolve(strict=True)
    old = load_json(path)
    c.validate(old, 'GATE')
    if (old['schema_version'] != 2 or old['source_commit'] != FAILED_COMMIT
            or old['source_snapshot']['git_commit'] != FAILED_COMMIT
            or path != Path(old['root'])/'gate_spec.json'
            or any(old.get(k) != v for k,v in c.registration(2).items())):
        raise ValueError('Expected original failed f979171c synchronous gate')
    reuse = compatible_source(old, active, project)
    root = Path(old['root'])
    if (root/'preflight/preflight.json').exists() or (root/'science').exists():
        raise PermissionError('Recovery requires an unadmitted failed gate')
    env = dict(os.environ, PYTHONPATH=str(Path(old['project_dir'])/'src'),
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    print('Authenticating failed synchronous gate in its ORIGINAL pinned source...', flush=True)
    proc = subprocess.run([sys.executable, '-u', '-s', '-c', ORIGINAL_CHECK, str(path)],
        cwd=old['project_dir'], env=env, capture_output=True, text=True)
    if proc.returncode:
        raise ValueError('Original gate authentication failed:\n'+proc.stderr)
    records = [json.loads(v.removeprefix('LUKA_ISOLATED_AUTH=')) for v in proc.stdout.splitlines()
               if v.startswith('LUKA_ISOLATED_AUTH=')]
    if records != [dict(gate=old['content_hash'], jobs=dict(preflight=FAILED_JOB))]:
        raise ValueError('Failed gate differs from exact submission journal')
    runtime_path = root/'preflight/runtime.json'
    runtime = load_json(runtime_path)
    if str(runtime['job_id']) != FAILED_JOB:
        raise ValueError('Failed runtime job differs')
    failures = []
    for phase_path in sorted((root/'preflight/phases').glob('*-end.json')):
        row = load_json(phase_path)
        if row.get('phase') == 'FUSION_D000_D000_probe_train_stress256_steps3':
            if row.get('status') != 'failed' or 'illegal memory access' not in row.get('error', ''):
                raise ValueError('Expected D000/D000 CUDA failure, not scientific selection')
            failures.append(phase_path)
    if len(failures) != 1:
        raise ValueError('Exactly one failed D000/D000 stress record required')
    return old, dict(failed_gate=file_ref(path), original_runtime=file_ref(runtime_path),
        failure=file_ref(failures[0]), log=file_ref(root/f'preflight-{FAILED_JOB}.out'),
        source_reuse=reuse, accounting=dict(job_id=FAILED_JOB, state='FAILED', exit_code='1:0'))


def location(root, old, project):
    root = Path(root).resolve()
    parent = load_json(old['parent_import']['spec']['path'])
    c.protect(root, parent, Path(project).resolve())
    protected = [old['root'], old['project_dir']]
    # Also preserve earlier diagnostic/failed source roots in the lineage.
    for key in ('diagnostic', 'original_gate'):
        ref = old['recovery'].get(key)
        if ref:
            prior = load_json(ref['path'])
            protected += [prior['root'], prior['project_dir']]
    for name in protected:
        other = Path(name).resolve()
        if root.is_relative_to(other) or other.is_relative_to(root):
            raise PermissionError('Isolated recovery overlaps original artifacts/source')
    return parent


def create(*, failed_gate, root, project, commit):
    root, project = Path(root).resolve(), Path(project).resolve()
    active = sources.source(project, commit)
    old, reuse = authenticate(failed_gate, active, project)
    location(root, old, project)
    if failed_accounting(FAILED_JOB) != reuse['accounting']:
        raise ValueError('Terminal failed accounting differs')
    gate = c.artifact('GATE', version=3, parents=dict(source=active['content_hash'], failed_gate=old['content_hash']),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=active,
        parent_import=old['parent_import'], prepared_sha256=old['prepared_sha256'], recovery=reuse,
        **c.registration(3))
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root/'gate_spec.json', gate)
    return gate


def validate_gate(gate):
    c.validate(gate, 'GATE')
    if (gate['schema_version'] != 3 or any(gate.get(k) != v for k,v in c.registration(3).items())
            or sources.source(gate['project_dir'], gate['source_commit']) != gate['source_snapshot']
            or load_json(Path(gate['root'])/'gate_spec.json') != gate):
        raise ValueError('Isolated recovery source/registration differs')
    for key in ('failed_gate', 'original_runtime', 'failure', 'log'):
        validate_file_ref(gate['recovery'][key])
    old, reuse = authenticate(gate['recovery']['failed_gate']['path'], gate['source_snapshot'], gate['project_dir'])
    parent = location(gate['root'], old, gate['project_dir'])
    if (gate['recovery'] != reuse or gate['parent_import'] != old['parent_import']
            or gate['prepared_sha256'] != old['prepared_sha256']
            or gate['parents'] != dict(source=gate['source_snapshot']['content_hash'], failed_gate=old['content_hash'])):
        raise ValueError('Isolated recovery lineage changed')
    return parent

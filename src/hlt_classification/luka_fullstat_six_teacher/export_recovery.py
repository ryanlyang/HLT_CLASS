"""Narrow changed-source recovery of job 8821154's finished TRAIN exports.

This route imports no student state. The original source authenticates its own
spec, exports, scheduler records and failed startup before the new source may
reuse bytes. All training/target-generation code must remain identical.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json
from .contracts import artifact, validate, recipe, COUNTS, TASK

ORIGINAL_COMMIT = '15808976126fb87cdf69ffae7a7a406cb865afb1'
ORIGINAL_JOB = '8821154'
KIND = 'EXPORT_RECOVERY'
PREFIX = 'src/hlt_classification/luka_fullstat_six_teacher/'
ALLOWED = {PREFIX+n+'.py' for n in
           ('sampling', 'startup', 'campaign', 'recovery', 'export_recovery', 'kd_banks', 'source')}


def source_delta(project, commit):
    from .source import git
    git(project, 'merge-base', '--is-ancestor', ORIGINAL_COMMIT, commit)
    changed = git(project, 'diff', '--name-only', ORIGINAL_COMMIT, commit,
                  '--', 'src', 'scripts', 'sbatch').splitlines()
    if not changed or not set(changed) <= ALLOWED:
        raise ValueError('Export recovery permits only the reviewed memory/recovery patch')


def old_boundary(old):
    validate(old, 'SPEC')
    if (old['source_commit'] != ORIGINAL_COMMIT or old.get('resume') is not None
            or old['counts'] != COUNTS or old['recipe'] != recipe()):
        raise ValueError('Expected original pre-training six-teacher failure')
    folder = Path(old['root'])/TASK
    if ((folder/'epochs').exists() or any((folder/n).exists() for n in
            ('startup.json', 'completed_training.json', 'training_report.json', 'task_receipt.json'))):
        raise ValueError('Changed-source recovery is export-only, before student training')
    failure = load_json(folder/'startup_failure.json');validate(failure, 'FAILURE')
    proof = load_json(folder/'startup_measurements.json')
    validate(proof, 'STARTUP', dict(spec=old['content_hash']))
    if (failure['error_type'] != 'MemoryError'
            or not failure['message'].startswith('Multiview peak_rss_bytes: measured ')
            or any(proof.get(k) is not True for k in ('installed_weaver_parity', 'exact_sampler_parity',
                'connected_gradients', 'classifier_ce_only', 'native_reload_parity', 'probe_discarded'))):
        raise ValueError('Expected reviewed host-memory admission failure')
    return {n:file_ref(folder/n) for n in ('startup_failure.json', 'startup_measurements.json')}


REPLAY = r'''
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);s=json.loads(p.read_text());sys.path.insert(0,str(Path(s['project_dir'])/'src'))
from hlt_classification.luka_fullstat_six_teacher import recovery,startup
value=recovery.inspect(p,s['project_dir'],s['source_commit'])
proof=json.loads((Path(s['root'])/'train_six/startup_measurements.json').read_text())
try:startup.check(s,proof)
except MemoryError as exc:
    if not str(exc).startswith('Multiview peak_rss_bytes: measured '):raise
else:raise ValueError('Original startup no longer rejects host memory')
print('SIX_EXPORT_RECOVERY='+json.dumps(value,sort_keys=True,allow_nan=False))
'''


def replay(path, old):
    from .source import git
    original = Path(old['project_dir'])
    if git(original, 'rev-parse', 'HEAD') != ORIGINAL_COMMIT:
        raise ValueError('Original six-teacher source is not pinned')
    env = dict(os.environ, PYTHONPATH='', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', '-s', '-c', REPLAY, str(path)],
                            cwd=original, env=env, capture_output=True, text=True)
    if result.returncode:
        raise ValueError('Original export recovery authentication failed:\n'+result.stderr)
    rows = [line.split('=', 1)[1] for line in result.stdout.splitlines()
            if line.startswith('SIX_EXPORT_RECOVERY=')]
    if len(rows) != 1:
        raise ValueError('Missing original export recovery evidence')
    return json.loads(rows[0])


def inspect(path, project, commit):
    old = load_json(path)
    source_delta(project, commit);diagnostics = old_boundary(old)
    print('Authenticating completed exports from failed job 8821154 in original source...', flush=True)
    original = replay(path, old)
    result = artifact(KIND, parents=dict(spec=old['content_hash'], recipe=recipe()['content_hash']),
        spec=file_ref(Path(path)), accounting=original['accounting'], exports=original['exports'],
        epochs=[], original_resume=original, diagnostics=diagnostics)
    check_evidence(result, old)
    return result


def check_evidence(value, old):
    from .recovery import export_records
    validate(value, KIND, dict(spec=old['content_hash'], recipe=recipe()['content_hash']))
    original = value['original_resume']
    validate(original, 'RESUME', dict(spec=old['content_hash'], recipe=recipe()['content_hash']))
    state = value['accounting']
    if (len(state) != 6 or state[0] != ORIGINAL_JOB or state[1] != 'lukafs6_'+TASK
            or state[4:] != ['FAILED', '1:0'] or state != original['accounting']
            or value['spec'] != original['spec'] or value['epochs'] != [] or original['epochs'] != []
            or set(value['exports']) != {'CONCAT', 'FUSION'}
            or value['exports'] != original['exports'] or value['exports'] != export_records(old)
            or value['diagnostics'] != old_boundary(old)):
        raise ValueError('Reviewed pre-training export recovery evidence changed')


def check(spec):
    value = spec['resume'];old = load_json(validate_file_ref(value['spec']))
    source_delta(spec['project_dir'], spec['source_commit'])
    check_evidence(value, old)
    if (old['recipe'] != spec['recipe'] or old['reuse'] != spec['reuse']
            or old['nodes'] != spec['nodes'] or old['counts'] != spec['counts']
            or old['root'] == spec['root']):
        raise ValueError('Export recovery changes science or target identities')
    from .source import git
    from hlt_classification.provenance import capture_source_snapshot
    if (git(old['project_dir'], 'rev-parse', 'HEAD') != ORIGINAL_COMMIT
            or capture_source_snapshot(Path(old['project_dir']), require_clean=True) != old['source_snapshot']):
        raise ValueError('Original source changed after recovery review')
    git(old['project_dir'], 'merge-base', '--is-ancestor', ORIGINAL_COMMIT, 'origin/main')

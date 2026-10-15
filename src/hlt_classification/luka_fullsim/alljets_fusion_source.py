"""Authenticate completed regular science using its original pinned interpreter path."""
from pathlib import Path
import json
import os
import subprocess
import sys

from hlt_classification.data.cache_contracts import load_json, sha256_file
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from .foundation import _source
from .preflight_reuse import _files
from . import alljets_campaign as parent_policy

ADDITIONS = frozenset('src/hlt_classification/luka_fullsim/alljets_fusion_'+name+'.py'
    for name in ('source', 'campaign', 'worker', 'preflight', 'submission'))
REQUIRED = sorted(ADDITIONS) + ['scripts/luka_fullsim_alljets_fusion.py',
    'scripts/queue_luka_fullsim_alljets_fusion.sh', 'sbatch/run_luka_fullsim_alljets_fusion.sh',
    'tests/test_luka_fullsim_alljets_fusion.py', 'docs/plans/LUKA_FULLSIM_ALLJETS_FUSION_PLAN.md',
    'docs/contracts/LUKA_FULLSIM_ALLJETS_FUSION.md', 'docs/LUKA_FULLSIM_ALLJETS_FUSION_RUNBOOK.md']

PARENT_CHECK = '''
import json, sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullsim import alljets_campaign as c, regular_worker as w
from hlt_classification.luka_fullsim.regular_cache import metadata
p = Path(sys.argv[1]).resolve(strict=True)
s = load_json(p)
c.validate(s, 'SPEC')
if p != Path(s['root'])/'campaign_spec.json':
    raise ValueError('Noncanonical parent specification')
c.validate_spec(s)
seen = {}
if w.completed(s, 'complete', seen) is None:
    raise ValueError('Complete regular ALLJETS campaign required')
receipts = {t['task_id']: w.completed(s, t['task_id'], seen) for t in c.tasks()}
if any(v is None for v in receipts.values()):
    raise ValueError('Missing parent task')
m, prepared, _, _ = metadata(s['admission']['prepared_root'])
out = dict(spec=s['content_hash'], prepared=prepared['content_hash'], cache=m['content_hash'],
    tasks={k: v['content_hash'] for k,v in receipts.items()}, rows=w.result_rows(s),
    u000_bank=receipts['reduce_U000']['result'])
print('LUKA_PARENT_AUTH=' + json.dumps(out, sort_keys=True, allow_nan=False))
'''


def source(project, commit):
    value = _source(Path(project), commit)
    tracked = subprocess.run(['git', '-C', str(project), 'ls-files'], check=True,
                             capture_output=True, text=True).stdout.splitlines()
    if not set(REQUIRED) <= set(tracked):
        raise ValueError('Commit all fusion code/tests/contracts before queueing')
    if Path(__file__).resolve() != Path(project).resolve()/'src/hlt_classification/luka_fullsim/alljets_fusion_source.py':
        raise ValueError('Fusion import escapes pinned project')
    return value


def compatible(parent, active, project):
    old, new = Path(parent['project_dir']).resolve(), Path(project).resolve()
    validate_source_snapshot(parent['source_snapshot'], repository=old, require_clean=True)
    validate_source_snapshot(active, repository=new, require_clean=True)
    before, after = _files(old), _files(new)
    if not before or not before <= after or after-before != ADDITIONS:
        raise ValueError('Unexpected fusion source additions/deletions')
    records = {}
    for name in sorted(before):
        records[name] = sha256_file(old/name)
        if records[name] != sha256_file(new/name):
            raise ValueError('Parent scientific source changed: '+name)
    return records


def authenticate(path, *, active, project):
    path = Path(path).resolve(strict=True)
    ref, parent = file_ref(path), load_json(path)
    parent_policy.validate(parent, 'SPEC')
    if (parent['counts'] != parent_policy.COUNTS or parent['nodes'] != parent_policy.nodes()
            or parent['training_recipe'] != parent_policy.recipe()
            or path != Path(parent['root'])/'campaign_spec.json'):
        raise ValueError('Expected original completed ALLJETS regular comparison')
    records = compatible(parent, active, project)
    env = dict(os.environ, PYTHONPATH=str(Path(parent['project_dir'])/'src'),
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    print('Authenticating completed regular parent in its original pinned worktree...', flush=True)
    try:
        result = subprocess.run([sys.executable, '-u', '-s', '-c', PARENT_CHECK, str(path)],
            cwd=parent['project_dir'], env=env, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise ValueError('Original-source parent authentication failed:\n'+(exc.stderr or str(exc))) from exc
    lines = [line[len('LUKA_PARENT_AUTH='):] for line in result.stdout.splitlines()
             if line.startswith('LUKA_PARENT_AUTH=')]
    if len(lines) != 1:
        raise ValueError('Missing original-source parent authentication result')
    value = json.loads(lines[0])
    if (value['spec'] != parent['content_hash'] or value['prepared'] != parent['prepared_sha256']
            or len(value['rows']) != 9 or any(r['state'] != 'COMPLETE' for r in value['rows'])):
        raise ValueError('Incomplete/wrong parent comparison')
    validate_file_ref(ref)
    return dict(spec=ref, evidence=value, identical_source_files=records), parent

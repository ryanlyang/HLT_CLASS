"""Exact old-source replay and byte-identical scientific kernels."""
from pathlib import Path
import subprocess
from hlt_classification.data.cache_contracts import load_json, sha256_file
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_fullsim.preflight_reuse import _files
from hlt_classification.luka_attention_weight.source import replay
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.luka_supervision_depth import contracts as old

BASE = '04d1f0f5945ab9af38d83e192eff48e124ad8dc5'
MODULES = ('__init__', 'contracts', 'source', 'scheduler', 'campaign', 'submission', 'worker', 'scratch', 'interrupted')
REQUIRED = {'src/hlt_classification/luka_depth_recovery/' + n + '.py' for n in MODULES} | {
    'scripts/luka_depth_recovery.py', 'scripts/queue_luka_depth_recovery.sh',
    'sbatch/run_luka_depth_recovery.sh', 'tests/test_luka_depth_recovery.py',
    'docs/plans/LUKA_DEPTH_RECOVERY_PLAN.md', 'docs/contracts/LUKA_DEPTH_RECOVERY.md',
    'docs/LUKA_DEPTH_RECOVERY_RUNBOOK.md'}


def current(project, commit):
    project = Path(project).resolve(); snap = _source(project, commit)
    def git(*args):
        return subprocess.run(['git', '-C', str(project), *args], check=True, capture_output=True, text=True).stdout.splitlines()
    if not REQUIRED <= set(git('ls-files')) or Path(__file__).resolve() != project/'src/hlt_classification/luka_depth_recovery/source.py':
        raise ValueError('Commit the complete additive recovery implementation')
    if git('diff', '--name-only', '--diff-filter=MDRT', BASE, commit, '--', 'src'):
        raise ValueError('Recovery must not change existing scientific runtime')
    return snap


REPLAY = '''
import json,sys
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_supervision_depth import campaign,submission
s=load_json(sys.argv[1]); campaign.check(s)
print('DEPTH_RECOVERY_SOURCE='+json.dumps(submission.accepted_jobs(s),sort_keys=True))
'''


def authenticate(path, project):
    path = Path(path).resolve(strict=True); spec = load_json(path); old.validate(spec, 'SPEC')
    if path != Path(spec['root'])/'campaign_spec.json':
        raise ValueError('Canonical original spec required')
    validate_source_snapshot(spec['source_snapshot'], repository=spec['project_dir'], require_clean=True)
    before = _files(spec['project_dir'])
    if not before or not before <= _files(project):
        raise ValueError('Original runtime missing')
    for name in sorted(before):
        if sha256_file(Path(spec['project_dir'])/name) != sha256_file(Path(project)/name):
            raise ValueError('Original runtime changed: ' + name)
    ref, jobs = replay(path, spec['project_dir'], REPLAY, 'DEPTH_RECOVERY_SOURCE=')
    if set(jobs) != {t['task_id'] for t in spec['tasks']}:
        raise ValueError('Require fully journaled original submission')
    return ref, spec, jobs


def original(ref):
    validate_file_ref(ref); spec = load_json(ref['path']); old.validate(spec, 'SPEC')
    if Path(ref['path']) != Path(spec['root'])/'campaign_spec.json':
        raise ValueError('Wrong original specification location')
    return spec

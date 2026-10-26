"""Authenticate completed c50 only, in its original source, including retirement."""
from pathlib import Path
import subprocess
from hlt_classification.data.cache_contracts import load_json, sha256_file
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_fullsim.preflight_reuse import _files
from hlt_classification.luka_attention_weight.source import replay
from hlt_classification.luka_trial_hybrid import banks as old_banks, contracts as old
from .contracts import artifact, validate, COUNTS, TEACHERS, REFERENCES

BASE = '72490a9d6de4acabe9cbea518c5b100d893e6866'
SOURCE_BASE = '0c4a71a55e8bc163ea3c4805b72f06d18e640cb5'
MODULES = ('__init__', 'contracts', 'source', 'model', 'banks', 'training', 'campaign', 'submission', 'worker', 'startup', 'reference')
REQUIRED = {'src/hlt_classification/luka_auxiliary_grid/'+n+'.py' for n in MODULES} | {
    'scripts/luka_auxiliary_grid.py', 'scripts/queue_luka_auxiliary_grid.sh', 'sbatch/run_luka_auxiliary_grid.sh',
    'tests/test_luka_auxiliary_grid.py', 'docs/plans/LUKA_AUXILIARY_GRID_PLAN.md',
    'docs/contracts/LUKA_AUXILIARY_GRID.md', 'docs/LUKA_AUXILIARY_GRID_RUNBOOK.md'}


def current(project, commit):
    project = Path(project).resolve(); value = _source(project, commit)
    def git(*args):
        return subprocess.run(['git', '-C', str(project), *args], check=True, capture_output=True, text=True).stdout.splitlines()
    if not REQUIRED <= set(git('ls-files')):
        raise ValueError('Commit complete auxiliary-grid implementation, tests and docs')
    changed = git('diff', '--name-status', '--no-renames', '--diff-filter=MDRT', SOURCE_BASE, commit, '--', 'src')
    if changed:
        raise ValueError('Existing scientific runtime changed: '+repr(changed))
    if Path(__file__).resolve() != project/'src/hlt_classification/luka_auxiliary_grid/source.py':
        raise ValueError('Imported source escapes pinned project')
    return value


REPLAY = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_trial_hybrid import campaign,worker,banks,submission
s=load_json(sys.argv[1]); campaign.check(s)
root=Path(s['root']); retired=worker.retired_evidence(s); seen={}
worker.validate_task(s,'c50_DE000',seen,retired=retired)
for task,rec in seen.items(): submission.successful(s,task,rec['job_id'])
teachers={t:banks.teacher_record(s,'c50',t) for t in ('OFFLINE','UE100','DE066','DE033')}
refs={t:file_ref(root/'targets/references'/(t+'.npz')) for t in ('M0HLT','OFFLINE','OFFLINE_EMBED_W5','U100_EMBED_W5')}
refs['C50_DE000']=file_ref(root/'c50_DE000/fit/validation_predictions.npz')
print('AUXGRID_REUSE='+json.dumps(dict(anchor=s['anchor'],acceptance=s['acceptance'],
    teachers=teachers,references=refs,subset=file_ref(root/'targets/references/original50k.npz'),
    tasks={t:file_ref(root/t/'receipt.json') for t in seen},
    costs=file_ref(root/'c50_DE000/costs.json'),
    comparison_report=file_ref(root/'c50_DE000/fit/training_report.json')),
    sort_keys=True,allow_nan=False))
'''


def parent(path):
    path = Path(path).resolve(strict=True); s = load_json(path); old.validate(s, 'SPEC')
    if (path != Path(s['root'])/'campaign_spec.json' or s['counts'] != COUNTS or s['source_commit'] != BASE
            or s['recipes']['c50'] != {stage: old.recipe('c50', stage) for stage in old.STAGES}):
        raise ValueError('Expected original 72490a9d trial c50 ladder, not another teacher chain')
    return s


def import_parent(path, project):
    s = parent(path)
    validate_source_snapshot(s['source_snapshot'], repository=s['project_dir'], require_clean=True)
    before = _files(s['project_dir'])
    if not before or not before <= _files(project):
        raise ValueError('Missing donor runtime')
    for name in sorted(before):
        if sha256_file(Path(s['project_dir'])/name) != sha256_file(Path(project)/name):
            raise ValueError('Parent runtime bytes changed: '+name)
    print('Authenticating completed c50 teachers and DE000 in original source; other arms need not finish...', flush=True)
    ref, evidence = replay(path, s['project_dir'], REPLAY, 'AUXGRID_REUSE=')
    value = artifact('REUSE', parents=dict(spec=s['content_hash']), spec=ref, **evidence)
    validate_reuse(value)
    return value, s


def validate_reuse(value):
    validate_file_ref(value['spec']); s = parent(value['spec']['path'])
    validate(value, 'REUSE', dict(spec=s['content_hash']))
    validate_source_snapshot(s['source_snapshot'], repository=s['project_dir'], require_clean=True)
    if (value['anchor'] != s['anchor'] or value['acceptance'] != s['acceptance']
            or set(value['teachers']) != set(TEACHERS) or set(value['references']) != set(REFERENCES[:-1])
            or set(value['tasks']) != {'targets', *('c50_'+stage for stage in old.STAGES)}):
        raise ValueError('Frozen c50 import inventory differs')
    root = Path(s['root'])
    for t in TEACHERS:
        if value['teachers'][t] != old_banks.teacher_record(s, 'c50', t):
            raise ValueError('Wrong branch/selected teacher identity')
    for name, ref in value['references'].items():
        expected = root/('c50_DE000/fit/validation_predictions.npz' if name == 'C50_DE000'
                         else 'targets/references/'+name+'.npz')
        validate_file_ref(ref)
        if ref != file_ref(expected):
            raise ValueError('Reference path is not bound to original c50')
    paths = dict(subset=root/'targets/references/original50k.npz', costs=root/'c50_DE000/costs.json',
                 comparison_report=root/'c50_DE000/fit/training_report.json')
    for key, path in paths.items():
        validate_file_ref(value[key])
        if value[key] != file_ref(path):
            raise ValueError('Imported evidence location differs')
    for task, ref in value['tasks'].items():
        validate_file_ref(ref)
        if ref != file_ref(root/task/'receipt.json'):
            raise ValueError('Parent task evidence differs')
    return s

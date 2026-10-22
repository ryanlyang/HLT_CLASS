"""Original-source replay of real accepted resources and selected U100 lineage."""
from pathlib import Path
import json
import os
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json, sha256_file
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullsim.foundation import _source
from .contracts import artifact, COUNTS

DONOR = 'fd687181a60c1557f98d7498f577e26d6dbb35a7'
BASELINE = '5bf9a00d722c867f652588572cb489451bb1a12e'
MODULES = ('__init__', 'contracts', 'source', 'campaign', 'banks', 'training', 'worker', 'submission')
ADDITIONS = {f'src/hlt_classification/luka_fullstat_u100_embedding/{n}.py' for n in MODULES}
ENTRYPOINTS = {'scripts/luka_fullstat_u100_embedding.py', 'scripts/queue_luka_fullstat_u100_embedding.sh',
              'sbatch/run_luka_fullstat_u100_embedding.sh'}
REQUIRED = ADDITIONS | ENTRYPOINTS | {'tests/test_luka_fullstat_u100_embedding.py',
    'docs/plans/LUKA_FULLSTAT_U100_EMBEDDING_PLAN.md',
    'docs/contracts/LUKA_FULLSTAT_U100_EMBEDDING.md', 'docs/LUKA_FULLSTAT_U100_EMBEDDING_RUNBOOK.md'}

REPLAY = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullstat_embedding_sweep import campaign,worker
from hlt_classification.luka_fullstat import worker as original
g=load_json(sys.argv[1]); accepted=campaign.accepted(g)
p=load_json(g['anchor']['spec']['path'])
task=original.completed(p,'train_COARSE_U100')
if task is None: raise ValueError('Missing selected COARSE_U100')
directory=Path(p['root'])/'train_COARSE_U100'
report=load_json(directory/'training_report.json')
node=next(n for n in p['nodes'] if n['node_id']=='COARSE_U100')
if node['coordinate']!='U100' or node['context_coordinate'] is not None or report['node']!=node:
    raise ValueError('Not the original single-encoder COARSE_U100')
refs={key:file_ref(directory/name) for key,name in (
    ('checkpoint','selected.pt'),('report','training_report.json'),
    ('receipt','receipt.json'),('predictions','validation_predictions.npz'))}
meta=worker.metadata(g)[0]
print('U100_REUSE='+json.dumps(dict(teacher_refs=refs,teacher_node=node,
    summaries=meta['summaries'],preflight=file_ref(Path(g['root'])/'preflight/receipt.json'),
    gpu=accepted['gpu'],environment=accepted['environment'],train_minutes=accepted['train_minutes'],
    measurements=accepted['measurements']),sort_keys=True,allow_nan=False))
'''


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project, commit)
    subprocess.run(['git', '-C', str(project), 'merge-base', '--is-ancestor', commit, 'origin/main'],
                   check=True, capture_output=True, text=True)
    tracked = set(subprocess.run(['git', '-C', str(project), 'ls-files'],
        check=True, capture_output=True, text=True).stdout.splitlines())
    changed = set(subprocess.run(['git', '-C', str(project), 'diff', '--name-only',
        BASELINE, commit, '--', 'src', 'scripts', 'sbatch'],
        check=True, capture_output=True, text=True).stdout.splitlines())
    if (not REQUIRED <= tracked or not changed <= ADDITIONS | ENTRYPOINTS
            or Path(__file__).resolve() != project/'src/hlt_classification/luka_fullstat_u100_embedding/source.py'):
        raise ValueError('Commit the additive U100 implementation without changing old runtime')
    return snapshot


def authenticate(path, project):
    path = Path(path).resolve(strict=True)
    g = load_json(path)
    original = Path(g['project_dir'])
    if (g['source_commit'] != DONOR or g['counts'] != COUNTS
            or path != Path(g['root'])/'gate_spec.json'):
        raise ValueError('Expected original fd687181 FullStat embedding gate')
    # All donor scientific runtime stays byte-identical, including transitive kernels.
    names = subprocess.run(['git', '-C', str(original), 'ls-files', 'src', 'sbatch', 'scripts'],
        check=True, capture_output=True, text=True).stdout.splitlines()
    hashes = {}
    for name in names:
        if not name.endswith(('.py', '.sh')):
            continue
        before, after = original/name, Path(project)/name
        if not after.is_file() or sha256_file(before) != sha256_file(after):
            raise ValueError('Changed measured scientific source: '+name)
        hashes[name] = sha256_file(after)
    if not hashes:
        raise ValueError('Empty donor scientific source inventory')
    env = dict(os.environ, PYTHONPATH=str(original/'src'), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    print('Reusing authenticated Oscar preflight and frozen COARSE_U100; no new full preflight.', flush=True)
    run = subprocess.run([sys.executable, '-B', '-s', '-c', REPLAY, str(path)],
        cwd=original, env=env, capture_output=True, text=True)
    if run.returncode:
        raise ValueError('Original-source reuse rejected:\n'+run.stderr)
    lines = [s.split('=',1)[1] for s in run.stdout.splitlines() if s.startswith('U100_REUSE=')]
    if len(lines) != 1:
        raise ValueError('Missing original-source reuse evidence')
    evidence = json.loads(lines[0])
    return artifact('REUSE', gate=file_ref(path), source_commit=DONOR,
        anchor=g['anchor'], old_root=g['root'], old_project=str(original),
        base_node=g['nodes']['w0'], counts=g['counts'], execution_site=g['execution_site'],
        cpus=g['cpus'], memory_mb=g['memory_mb'], export_minutes=g['gate_minutes'],
        compatible_source=hashes, **evidence)

"""Replay only completed hybrid UE000/UE100 in the original pinned source."""
from pathlib import Path
import json
import os
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json, sha256_file
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullsim.foundation import _source
from .contracts import artifact, COUNTS

DONOR = 'e129075e9c3eaa9fd2e49460b79ecf5f56ba4be7'
BASELINE = 'a4596cb915252672ab42c7a8724d6a105a0bc56e'
MODULES = ('__init__', 'contracts', 'source', 'campaign', 'banks', 'model', 'training', 'worker', 'submission', 'kd_banks')
ADDITIONS = {f'src/hlt_classification/luka_fullstat_splithead/{n}.py' for n in MODULES}
ENTRYPOINTS = {'scripts/luka_fullstat_splithead.py',
    'scripts/queue_luka_fullstat_splithead.sh', 'sbatch/run_luka_fullstat_splithead.sh'}
REQUIRED = ADDITIONS | ENTRYPOINTS | {'tests/test_luka_fullstat_splithead.py',
    'docs/plans/LUKA_FULLSTAT_SPLITHEAD_PLAN.md',
    'docs/contracts/LUKA_FULLSTAT_SPLITHEAD.md',
    'docs/LUKA_FULLSTAT_SPLITHEAD_RUNBOOK.md'}

def compatible_changes(changed, added):
    return changed <= ADDITIONS | ENTRYPOINTS and changed <= added


REPLAY = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullstat_hybrid_embedding import campaign,worker,submission
s=load_json(sys.argv[1]); campaign.check(s)
if worker.completed(s,'UE100') is None:
    raise ValueError('Completed hybrid UE100 and UE000 ancestry required')
def refs(stage):
    d=Path(s['root'])/('train_'+stage)
    return {k:file_ref(d/f) for k,f in [('checkpoint','selected.pt'),
        ('report','training_report.json'),('predictions','validation_predictions.npz'),('receipt','task_receipt.json')]}
r=dict(s['references'], UE000=refs('UE000'), UE100=refs('UE100'))
banks={'OFFLINE':dict(embedding=s['offline_bank'],
    kd=file_ref(Path(s['root'])/'logit_targets/OFFLINE/manifest.json')),
    'M0HLT':dict(embedding=file_ref(Path(s['root'])/'targets/M0HLT/manifest.json')),
    'UE100':{kind:file_ref(Path(s['root'])/directory/'UE100/manifest.json')
             for kind,directory in [('embedding','targets'),('kd','logit_targets')]}}
e=dict(accepted=s['reuse'],anchor=s['anchor'], references=r,
    external_refs=dict(OFFLINE=s['offline_refs'],M0HLT=s['references']['M0HLT'],UE100=r['UE100']),
    imported_banks=banks, base_node=s['reuse']['base_node'], execution_site=s['execution_site'],
    environment=s['reuse']['environment'],gpu=s['reuse']['gpu'],summaries=s['reuse']['summaries'],
    parent_task_minutes=s['task_minutes'],
    jobs={k:v for k,v in submission.accepted_jobs(s).items() if k in ('train_UE000','train_UE100')},
    protected_roots=s['reuse']['protected_roots']+[s['root'],s['project_dir']])
print('SPLITHEAD_REUSE='+json.dumps(e,sort_keys=True,allow_nan=False))
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
    added = set(subprocess.run(['git', '-C', str(project), 'diff', '--name-only', '--diff-filter=A',
        BASELINE, commit, '--', 'src', 'scripts', 'sbatch'],
        check=True, capture_output=True, text=True).stdout.splitlines())
    if (not REQUIRED <= tracked or not compatible_changes(changed, added)
            or Path(__file__).resolve() != project/'src/hlt_classification/luka_fullstat_splithead/source.py'):
        raise ValueError('Commit additive split-head files without changing existing runtime')
    return snapshot


def authenticate(path, project):
    path = Path(path).resolve(strict=True)
    s = load_json(path)
    original = Path(s['project_dir'])
    if (s['source_commit'] != DONOR or s['counts'] != COUNTS
            or path != Path(s['root'])/'campaign_spec.json'):
        raise ValueError('Expected completed FullStat hybrid UE100 specification')
    names = subprocess.run(['git', '-C', str(original), 'ls-files', 'src', 'sbatch', 'scripts'],
        check=True, capture_output=True, text=True).stdout.splitlines()
    hashes = {}
    for name in names:
        if not name.endswith(('.py', '.sh')):
            continue
        before, after = original/name, Path(project)/name
        if not after.is_file() or sha256_file(before) != sha256_file(after):
            raise ValueError('Changed donor runtime: '+name)
        hashes[name] = sha256_file(after)
    if not hashes:
        raise ValueError('Empty donor runtime inventory')
    env = dict(os.environ, PYTHONPATH=str(original/'src'), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    print('Authenticating completed FullStat UE100/UE000 and original Oscar acceptance; no new preflight.', flush=True)
    run = subprocess.run([sys.executable, '-B', '-s', '-c', REPLAY, str(path)],
        cwd=original, env=env, capture_output=True, text=True)
    if run.returncode:
        raise ValueError('Original-source authentication failed:\n'+run.stderr)
    lines = [line.split('=', 1)[1] for line in run.stdout.splitlines() if line.startswith('SPLITHEAD_REUSE=')]
    if len(lines) != 1:
        raise ValueError('Missing exact original-source evidence')
    evidence = json.loads(lines[0])
    # Original worker.completed checks exact UE100 and UE000 Slurm identities,
    # native checkpoints, TRAIN banks and recursively authenticated lineage.
    return artifact('REUSE', parent_spec=file_ref(path), source_commit=DONOR,
        counts=dict(COUNTS), compatible_source=hashes, **evidence)

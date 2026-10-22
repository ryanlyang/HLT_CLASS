"""Replay completed U100/reference evidence in its own clean pinned worktree."""
from pathlib import Path
import json
import os
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json, sha256_file
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullsim.foundation import _source
from .contracts import artifact, COUNTS

DONOR = '7ea8b0c693734a26ced3820354e047f4af26c979'
BASELINE = '118aa375eea423e5e99338cd9a6eda198225ef21'
MODULES = ('__init__', 'contracts', 'source', 'campaign', 'banks', 'model', 'training', 'worker', 'submission')
ADDITIONS = {f'src/hlt_classification/luka_fullstat_multiteacher/{n}.py' for n in MODULES}
ENTRYPOINTS = {'scripts/luka_fullstat_multiteacher.py', 'scripts/queue_luka_fullstat_multiteacher.sh',
              'sbatch/run_luka_fullstat_multiteacher.sh'}
REQUIRED = ADDITIONS | ENTRYPOINTS | {'tests/test_luka_fullstat_multiteacher.py',
    'docs/plans/LUKA_FULLSTAT_MULTITEACHER_PLAN.md',
    'docs/contracts/LUKA_FULLSTAT_MULTITEACHER.md', 'docs/LUKA_FULLSTAT_MULTITEACHER_RUNBOOK.md'}

# These unrelated SPORC-trial entrypoints changed after the Oscar U100 run.
# None is imported by the FullStat data/model/training/replay path. Keep the
# exception explicit; every other pre-existing runtime byte remains compared.
UNRELATED_SPORC = {'scripts/luka_d066_embedding.py', 'scripts/queue_luka_d066_embedding.sh',
    *{'src/hlt_classification/luka_d066_embedding/'+n+'.py'
      for n in ('campaign','followup','source','worker')}}

REPLAY = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_fullstat_u100_embedding import campaign,worker
from hlt_classification.luka_fullstat import worker as original
s=load_json(sys.argv[1]); campaign.check(s); result=worker.results(s)
p=load_json(s['anchor']['spec']['path'])
teachers={}
for name,co in [('OFFLINE','OFFLINE'),('COARSE_U100','U100'),('COARSE_D066','D066'),('COARSE_D033','D033')]:
    task=original.completed(p,'train_'+name)
    if task is None: raise ValueError('Missing completed teacher '+name)
    d=Path(p['root'])/('train_'+name)
    node=next(n for n in p['nodes'] if n['node_id']==name)
    report=load_json(d/'training_report.json')
    if node['coordinate']!=co or node['context_coordinate'] is not None or report['node']!=node:
        raise ValueError('Wrong single-encoder teacher coordinate')
    refs={k:file_ref(d/f) for k,f in [('checkpoint','selected.pt'),('report','training_report.json'),
        ('receipt','receipt.json'),('predictions','validation_predictions.npz')]}
    teachers[name]=dict(node=node,refs=refs)
old_gate=load_json(s['reuse']['gate']['path'])
reuse_banks={'OFFLINE':file_ref(Path(old_gate['root'])/'targets/manifest.json'),
             'COARSE_U100':file_ref(Path(s['root'])/'targets/manifest.json')}
for ref in reuse_banks.values():
    m=load_json(ref['path']); validate_file_ref(m['payload'],root=Path(ref['path']).parent)
reference={k:file_ref(Path(s['root'])/'train_w5'/f) for k,f in
    [('checkpoint','selected.pt'),('report','training_report.json'),
     ('predictions','validation_predictions.npz'),('receipt','task_receipt.json')]}
print('MULTI_REUSE='+json.dumps(dict(teachers=teachers,reuse_banks=reuse_banks,
    reference=reference,reference_result=result['content_hash'],
    summaries=s['reuse']['summaries'],anchor=s['anchor'],base_node=s['node'],
    execution_site=s['execution_site'],environment=s['reuse']['environment'],gpu=s['reuse']['gpu'],
    parent_train_minutes=s['train_minutes'],protected_roots=[s['root'],old_gate['root'],p['root'],
    s['project_dir'],old_gate['project_dir'],p['project_dir']]),sort_keys=True,allow_nan=False))
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
            or Path(__file__).resolve() != project/'src/hlt_classification/luka_fullstat_multiteacher/source.py'):
        raise ValueError('Commit additive multi-teacher files without changing existing runtime')
    return snapshot


def authenticate(path, project):
    path = Path(path).resolve(strict=True)
    s = load_json(path)
    original = Path(s['project_dir'])
    if (s['source_commit'] != DONOR or s['counts'] != COUNTS
            or path != Path(s['root'])/'campaign_spec.json'):
        raise ValueError('Expected completed original FullStat U100-w5 specification')
    names = subprocess.run(['git', '-C', str(original), 'ls-files', 'src', 'sbatch', 'scripts'],
        check=True, capture_output=True, text=True).stdout.splitlines()
    hashes = {}
    for name in names:
        if not name.endswith(('.py', '.sh')) or name in UNRELATED_SPORC:
            continue
        before, after = original/name, Path(project)/name
        if not after.is_file() or sha256_file(before) != sha256_file(after):
            raise ValueError('Changed donor runtime: '+name)
        hashes[name] = sha256_file(after)
    if not hashes:
        raise ValueError('Empty donor scientific inventory')
    env = dict(os.environ, PYTHONPATH=str(original/'src'), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    print('Authenticating completed U100-w5, four frozen teachers and reusable TRAIN banks...', flush=True)
    run = subprocess.run([sys.executable, '-B', '-s', '-c', REPLAY, str(path)],
        cwd=original, env=env, capture_output=True, text=True)
    if run.returncode:
        raise ValueError('Original-source authentication failed:\n'+run.stderr)
    lines = [s.split('=', 1)[1] for s in run.stdout.splitlines() if s.startswith('MULTI_REUSE=')]
    if len(lines) != 1:
        raise ValueError('Missing exact original-source evidence')
    return artifact('REUSE', parent_spec=file_ref(path), source_commit=DONOR,
        counts=dict(COUNTS), compatible_source=hashes, **json.loads(lines[0]))

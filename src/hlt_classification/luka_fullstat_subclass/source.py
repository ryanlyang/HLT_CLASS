"""Replay historical completion in its own source; never reinterpret old receipts."""
from pathlib import Path
import json
import os
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json, sha256_file
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_fullsim.foundation import _source
from .contracts import artifact, validate, COUNTS

# Execution ancestry includes the separate complementary-teacher Hydra additions.
# Those additions do not modify any existing Oscar scientific runtime.
BASELINE = 'd88eb82c6569312b72a3a3bccb1c39f9e357b4a4'
PARENT_COMMIT = 'a7249f49b6f522f7df3b150d27f53613452b7fb0'
PACKAGE = 'src/hlt_classification/luka_fullstat_subclass/'
ENTRYPOINTS = {'scripts/luka_fullstat_subclass.py',
    'scripts/launch_luka_fullstat_subclass.sh', 'sbatch/run_luka_fullstat_subclass.sh'}


def git(project, *args):
    return subprocess.run(['git','-C',str(project),*args],check=True,capture_output=True,text=True).stdout.strip()


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project,commit)
    git(project,'merge-base','--is-ancestor',commit,'origin/main')
    git(project,'merge-base','--is-ancestor',BASELINE,commit)
    changed = git(project,'diff','--name-only',BASELINE,commit,'--','src','scripts','sbatch').splitlines()
    if (any(not(n.startswith(PACKAGE) or n in ENTRYPOINTS) for n in changed)
            or Path(__file__).resolve() != project/(PACKAGE+'source.py')):
        raise ValueError('Only additive subclass runtime changes are registered')
    required = ENTRYPOINTS | {PACKAGE+n+'.py' for n in
        ('__init__','contracts','source','campaign','model','sampling','training','banks','startup',
         'worker','submission','launch','execution','recovery','targets')}
    required |= {'tests/test_luka_fullstat_subclass.py','tests/test_luka_fullstat_subclass_targets.py',
                 'tests/test_luka_fullstat_subclass_lifecycle.py',
                 'docs/contracts/LUKA_FULLSTAT_SUBCLASS.md',
                 'docs/LUKA_FULLSTAT_SUBCLASS_RUNBOOK.md',
                 'docs/plans/LUKA_FULLSTAT_SUBCLASS_PLAN.md'}
    if not required <= set(git(project,'ls-files').splitlines()):
        raise ValueError('Required runtime/tests/docs must be committed')
    return snapshot


REPLAY = r'''
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); s=json.loads(p.read_text()); project=Path(s['project_dir'])
sys.path.insert(0,str(project/'src'))
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullstat_multi3_contrastive import campaign,worker
campaign.check(s); receipt=worker.completed(s)
report,_=worker.validate_fit(s)
refs={k:file_ref(Path(s['root'])/'train_multi3_mix'/f) for k,f in
 [('checkpoint','selected.pt'),('report','training_report.json'),('predictions','validation_predictions.npz'),('receipt','task_receipt.json')]}
print('SUBCLASS_REUSE='+json.dumps(dict(reference=refs,banks=s['reuse']['banks'],job=receipt['job_id'],
 report_hash=report['content_hash'],u100=s['reuse']['u100']),sort_keys=True,allow_nan=False))
'''


def authenticate(path, project):
    path = Path(path).resolve(strict=True); parent = load_json(path)
    if (parent['contract'] != 'LUKA_FULLSTAT_MULTI3_CONTRASTIVE_SPEC/v1'
            or parent['source_commit'] != PARENT_COMMIT or parent['counts'] != COUNTS
            or path != Path(parent['root'])/'campaign_spec.json'):
        raise ValueError('Expected completed a7249f49 Multi3 contrastive campaign')
    original = Path(parent['project_dir'])
    if git(original,'rev-parse','HEAD') != PARENT_COMMIT or git(original,'status','--porcelain','--untracked-files=all'):
        raise ValueError('Historical source must remain clean and pinned')
    env = dict(os.environ,PYTHONPATH='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    print('Authenticating completed Multi3 contrastive and frozen TRAIN banks in original source...',flush=True)
    run = subprocess.run([sys.executable,'-B','-s','-c',REPLAY,str(path)],cwd=original,env=env,capture_output=True,text=True)
    if run.returncode: raise ValueError('Original-source replay failed:\n'+run.stderr)
    rows = [v.split('=',1)[1] for v in run.stdout.splitlines() if v.startswith('SUBCLASS_REUSE=')]
    if len(rows) != 1: raise ValueError('Missing original-source evidence')
    evidence = json.loads(rows[0])
    if str(evidence['job']) != '8361672':
        raise ValueError('Expected completed historical Multi3 contrastive job8361672')
    # Pin every imported historical runtime byte as well as the new source snapshot.
    inventory = dict(parent['reuse']['historical_source'])
    for tree in (original,):
        for name in git(tree,'ls-files','src','scripts','sbatch').splitlines():
            if name.endswith(('.py','.sh')): inventory[str(tree/name)] = sha256_file(tree/name)
    return artifact('REUSE',parents=dict(spec=parent['content_hash']),parent_spec=file_ref(path),
        historical_source=inventory,**evidence)


def check(reuse):
    validate(reuse,'REUSE')
    p = load_json(validate_file_ref(reuse['parent_spec']))
    if (p['content_hash'] != reuse['parents']['spec'] or p['source_commit'] != PARENT_COMMIT
            or p['counts'] != COUNTS or p['reuse']['u100'] != reuse['u100']
            or p['reuse']['banks'] != reuse['banks']):
        raise ValueError('Parent population/source changed')
    for path,digest in reuse['historical_source'].items():
        if sha256_file(path) != digest: raise ValueError('Historical runtime changed: '+path)
    for refs in [reuse['reference'],*p['references'].values()]:
        for key in ('report','receipt'): validate_file_ref(refs[key])
    for bank in reuse['banks'].values():
        m = load_json(validate_file_ref(bank['manifest']))
        if m['content_hash'] != bank['manifest_hash']: raise ValueError('Bank identity changed')
        for key in ('checkpoint','report'):validate_file_ref(bank['refs'][key])
        for key in ('receipt','predictions'):
            if key in bank['refs']:validate_file_ref(bank['refs'][key])
        if 'owner' in bank: validate_file_ref(bank['owner'])
    return p

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

# Current repository ancestry; completed Oscar donors remain byte-identical.
BASELINE = 'ffaf2997b3fa76a45e53105dae4c8ed68be3d56a'
PARENT_COMMIT = '16546c0e0196ab85cd69e23de005dc5f6f29feb5'
PACKAGE = 'src/hlt_classification/luka_fullstat_subclass_c06/'
ENTRYPOINTS = {'scripts/luka_fullstat_subclass_c06.py',
    'scripts/launch_luka_fullstat_subclass_c06.sh', 'sbatch/run_luka_fullstat_subclass_c06.sh'}


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
    required |= {'tests/test_luka_fullstat_subclass_c06.py','tests/test_luka_fullstat_subclass_c06_targets.py',
                 'tests/test_luka_fullstat_subclass_c06_lifecycle.py',
                 'docs/contracts/LUKA_FULLSTAT_SUBCLASS_C06.md',
                 'docs/LUKA_FULLSTAT_SUBCLASS_C06_RUNBOOK.md',
                 'docs/plans/LUKA_FULLSTAT_SUBCLASS_C06_PLAN.md'}
    if not required <= set(git(project,'ls-files').splitlines()):
        raise ValueError('Required runtime/tests/docs must be committed')
    return snapshot


REPLAY = r'''
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); s=json.loads(p.read_text()); project=Path(s['project_dir'])
sys.path.insert(0,str(project/'src'))
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullstat_subclass import campaign,worker,targets
campaign.check(s); receipt=worker.completed(s)
report,_=worker.validate_fit(s)
refs={k:file_ref(Path(s['root'])/'train_subclass'/f) for k,f in
 [('checkpoint','selected.pt'),('report','training_report.json'),('predictions','validation_predictions.npz'),('receipt','task_receipt.json')]}
f=Path(s['root'])/'subclasses'; m=targets.check_manifest(s)
subclasses=dict(manifest_hash=m['content_hash'],files=[file_ref(f/n) for n in targets.FILES])
print('SUBCLASS_C06_REUSE='+json.dumps(dict(reference=refs,banks=s['reuse']['banks'],job=receipt['job_id'],
 report_hash=report['content_hash'],u100=s['reuse']['u100'],subclasses=subclasses),sort_keys=True,allow_nan=False))
'''


def authenticate(path, project):
    path = Path(path).resolve(strict=True); parent = load_json(path)
    if (parent['contract'] != 'LUKA_FULLSTAT_SUBCLASS_SPEC/v1'
            or parent['source_commit'] != PARENT_COMMIT or parent['counts'] != COUNTS
            or path != Path(parent['root'])/'campaign_spec.json'):
        raise ValueError('Expected completed 16546c0e contrastive0.3/subclass0.5 campaign')
    original = Path(parent['project_dir'])
    if git(original,'rev-parse','HEAD') != PARENT_COMMIT or git(original,'status','--porcelain','--untracked-files=all'):
        raise ValueError('Historical source must remain clean and pinned')
    env = dict(os.environ,PYTHONPATH='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    print('Authenticating completed subclass0.5 reference and exact frozen targets in original source...',flush=True)
    run = subprocess.run([sys.executable,'-B','-s','-c',REPLAY,str(path)],cwd=original,env=env,capture_output=True,text=True)
    if run.returncode: raise ValueError('Original-source replay failed:\n'+run.stderr)
    rows = [v.split('=',1)[1] for v in run.stdout.splitlines() if v.startswith('SUBCLASS_C06_REUSE=')]
    if len(rows) != 1: raise ValueError('Missing original-source evidence')
    evidence = json.loads(rows[0])
    if str(evidence['job']) != '8761518':
        raise ValueError('Expected completed historical subclass job8761518')
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
    from hlt_classification.luka_fullstat_subclass.contracts import validate as old_validate, recipe as old_recipe
    old_validate(p,'SPEC')
    if (p['content_hash'] != reuse['parents']['spec'] or p['source_commit'] != PARENT_COMMIT
            or p['counts'] != COUNTS or p['reuse']['u100'] != reuse['u100']
            or p['reuse']['banks'] != reuse['banks'] or p['recipe'] != old_recipe()
            or str(reuse['job']) != '8761518'):
        raise ValueError('Parent population/source changed')
    for path,digest in reuse['historical_source'].items():
        if sha256_file(path) != digest: raise ValueError('Historical runtime changed: '+path)
    for refs in [reuse['reference'],*p['references'].values()]:
        for key in ('report','receipt'): validate_file_ref(refs[key])
    rec=load_json(validate_file_ref(reuse['reference']['receipt'])); old_validate(rec,'TASK',dict(spec=p['content_hash']))
    report=load_json(validate_file_ref(reuse['reference']['report']));old_validate(report,'TRAINING_REPORT')
    if (rec['job_id'] != reuse['job'] or rec['task_id'] != 'train_subclass'
            or rec['source_commit'] != PARENT_COMMIT or rec['report_hash'] != report['content_hash']
            or report['content_hash'] != reuse['report_hash']):
        raise ValueError('Completed historical report/receipt binding differs')
    from .targets import check_reuse
    check_reuse(reuse)
    for bank in reuse['banks'].values():
        m = load_json(validate_file_ref(bank['manifest']))
        if m['content_hash'] != bank['manifest_hash']: raise ValueError('Bank identity changed')
        for key in ('checkpoint','report'):validate_file_ref(bank['refs'][key])
        for key in ('receipt','predictions'):
            if key in bank['refs']:validate_file_ref(bank['refs'][key])
        if 'owner' in bank: validate_file_ref(bank['owner'])
    return p

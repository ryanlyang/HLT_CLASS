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
BASELINE = '0a00441fee7b7974f6e7b2692e79519a65faabd3'
PARENT_COMMIT = '16546c0e0196ab85cd69e23de005dc5f6f29feb5'
PACKAGE = 'src/hlt_classification/luka_fullstat_six_teacher/'
ENTRYPOINTS = {'scripts/luka_fullstat_six_teacher.py',
    'scripts/launch_luka_fullstat_six_teacher.sh', 'sbatch/run_luka_fullstat_six_teacher.sh'}


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
         'worker','submission','launch','execution','recovery','export_recovery','targets','kd_banks')}
    required |= {'tests/test_luka_fullstat_six_teacher.py','tests/test_luka_fullstat_six_teacher_targets.py',
                 'tests/test_luka_fullstat_six_teacher_lifecycle.py',
                 'tests/test_luka_fullstat_six_teacher_memory.py',
                 'tests/fixtures/luka_six_teacher_memory_smoke.py',
                 'docs/contracts/LUKA_FULLSTAT_SIX_TEACHER.md',
                 'docs/LUKA_FULLSTAT_SIX_TEACHER_RUNBOOK.md',
                 'docs/plans/LUKA_FULLSTAT_SIX_TEACHER_PLAN.md'}
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
print('SIX_TEACHER_REUSE='+json.dumps(dict(reference=refs,banks=s['reuse']['banks'],job=receipt['job_id'],
 report_hash=report['content_hash'],u100=s['reuse']['u100'],subclasses=subclasses),sort_keys=True,allow_nan=False))
'''


def authenticate(path, project, ladder_path):
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
    rows = [v.split('=',1)[1] for v in run.stdout.splitlines() if v.startswith('SIX_TEACHER_REUSE=')]
    if len(rows) != 1: raise ValueError('Missing original-source evidence')
    evidence = json.loads(rows[0])
    if str(evidence['job']) != '8761518':
        raise ValueError('Expected completed historical subclass job8761518')
    # Pin every imported historical runtime byte as well as the new source snapshot.
    inventory = dict(parent['reuse']['historical_source'])
    for tree in (original,):
        for name in git(tree,'ls-files','src','scripts','sbatch').splitlines():
            if name.endswith(('.py','.sh')): inventory[str(tree/name)] = sha256_file(tree/name)
    ladder = authenticate_ladder(ladder_path)
    for root in sorted({ladder['project_dir'],*ladder['protected_roots']}):
        tree=Path(root)
        if (tree/'.git').exists():
            for name in git(tree,'ls-files','src','scripts','sbatch').splitlines():
                if name.endswith(('.py','.sh')):
                    p=tree/name;inventory[str(p)]=sha256_file(p)
    same_population(parent,evidence,ladder)
    return artifact('REUSE',parents=dict(spec=parent['content_hash'],ladder=ladder['spec_hash']),parent_spec=file_ref(path),
        historical_source=inventory,ladder=ladder,**evidence)


def authenticate_ladder(path):
    from hlt_classification.luka_fullstat_finetune.source import REPLAY as replay, DONOR
    from hlt_classification.luka_fullstat_fine_tail.contracts import validate as fine_validate
    path=Path(path).resolve(strict=True);s=load_json(path);fine_validate(s,'SPEC')
    original=Path(s['project_dir'])
    if (s['source_commit']!=DONOR or s['counts']!=COUNTS or path!=Path(s['root'])/'campaign_spec.json'
            or git(original,'rev-parse','HEAD')!=DONOR or git(original,'status','--porcelain','--untracked-files=all')):
        raise ValueError('Expected completed clean original 04d1f0f5 fine-tail donor')
    print('Authenticating hybrid UE100 and previous split-head DE066/DE033 through original fine-tail source...',flush=True)
    run=subprocess.run([sys.executable,'-B','-s','-c',replay,str(path)],cwd=original,
        env=dict(os.environ,PYTHONPATH=str(original/'src'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True,text=True)
    if run.returncode:raise ValueError('Original ladder replay failed:\n'+run.stderr)
    rows=[v.split('=',1)[1] for v in run.stdout.splitlines() if v.startswith('FINETUNE_REUSE=')]
    if len(rows)!=1:raise ValueError('Missing ladder evidence')
    e=json.loads(rows[0]);names=('OFFLINE','UE100','DE066','DE033')
    return dict(spec=file_ref(path),spec_hash=s['content_hash'],project_dir=str(original),
        source_commit=DONOR,anchor=e['anchor'],execution_site=e['execution_site'],
        environment=e['environment'],gpu=e['gpu'],summaries=e['summaries'],
        protected_roots=e['protected_roots'],reference=e['reference'],
        banks={t:e['imported_banks'][t] for t in names},refs={t:e['teacher_refs'][t] for t in names})


def same_population(parent,evidence,ladder):
    u=evidence['u100'];a=parent['anchor']['evidence'];b=ladder['anchor']['evidence']
    if (a['prepared']!=b['prepared'] or parent['execution_site']!=ladder['execution_site']
            or any(u[k]!=ladder[k] for k in ('environment','gpu','summaries'))
            or any(evidence['banks']['OFFLINE']['refs'][k]['sha256']!=ladder['refs']['OFFLINE'][k]['sha256']
                   for k in ('checkpoint','report'))):
        raise ValueError('Teacher populations, OFFLINE identity or runtime differ')


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
    ladder=reuse['ladder'];same_population(p,reuse,ladder)
    from hlt_classification.luka_fullstat_finetune.source import DONOR
    from hlt_classification.luka_fullstat_fine_tail import banks as fb, kd_banks as fk
    from hlt_classification.luka_fullstat_fine_tail.contracts import validate as fv
    s=load_json(validate_file_ref(ladder['spec']));fv(s,'SPEC')
    if (s['source_commit']!=DONOR or s['counts']!=COUNTS or s['content_hash']!=ladder['spec_hash']
            or reuse['parents']!=dict(spec=p['content_hash'],ladder=s['content_hash'])
            or ladder['project_dir']!=s['project_dir'] or ladder['source_commit']!=DONOR
            or set(ladder['banks'])!={'OFFLINE','UE100','DE066','DE033'}):
        raise ValueError('Ladder identity changed')
    for t in ladder['banks']:
        if (ladder['refs'][t]!=fb.refs_for(s,t)
                or ladder['banks'][t]!=dict(embedding=fb.manifest_ref(s,t),kd=fk.manifest_ref(s,t))):
            raise ValueError('Selected ladder teacher changed')
        for key in ('checkpoint','report','receipt','predictions'):
            if key in ladder['refs'][t]:validate_file_ref(ladder['refs'][t][key])
        for ref in ladder['banks'][t].values():validate_file_ref(ref)
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

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

# The newer committed Hydra package adds no changes to the scientific donors.
# Pin it as execution ancestry so a normal follow-up commit can be queued.
BASELINE = '73abecfd95281246cea8e50bba3887a2737bac82'
PARENT_COMMIT = 'ceb2146149d2b1508a36b2cf91b274e0b58a26e6'
PACKAGE = 'src/hlt_classification/luka_fullstat_multi3_contrastive/'
ENTRYPOINTS = {'scripts/luka_fullstat_multi3_contrastive.py',
    'scripts/launch_luka_fullstat_multi3_contrastive.sh', 'sbatch/run_luka_fullstat_multi3_contrastive.sh'}


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
        raise ValueError('Only additive Multi3 contrastive runtime changes are registered')
    required = ENTRYPOINTS | {PACKAGE+n+'.py' for n in
        ('__init__','contracts','source','campaign','model','sampling','training','banks','startup',
         'worker','submission','launch','execution','recovery')}
    required |= {'tests/test_luka_fullstat_multi3_contrastive.py',
                 'docs/contracts/LUKA_FULLSTAT_MULTI3_CONTRASTIVE.md',
                 'docs/LUKA_FULLSTAT_MULTI3_CONTRASTIVE_RUNBOOK.md',
                 'docs/plans/LUKA_FULLSTAT_MULTI3_CONTRASTIVE_PLAN.md'}
    if not required <= set(git(project,'ls-files').splitlines()):
        raise ValueError('Required runtime/tests/docs must be committed')
    return snapshot


REPLAY = r'''
import importlib.util,json,sys
from pathlib import Path
p=Path(sys.argv[1]); s=json.loads(p.read_text()); project=Path(s['project_dir'])
boot=importlib.util.spec_from_file_location('original_bootstrap',project/'src/hlt_classification/luka_fullstat_multiview_export/bootstrap.py')
b=importlib.util.module_from_spec(boot); boot.loader.exec_module(b)
b.configure(project,p,registered=True)
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_fullstat_multiview_export import campaign,worker
from hlt_classification.luka_fullstat_multiview import worker as original
campaign.check(s,full=True); receipt=worker.completed(s,'train_multi3')
report,_=worker.validate_fit(s); old=campaign.parent(s)
refs={k:file_ref(Path(s['root'])/'train_multi3'/f) for k,f in
 [('checkpoint','selected.pt'),('report','training_report.json'),('predictions','validation_predictions.npz'),('receipt','task_receipt.json')]}
u=load_json(Path(s['references']['U100_W5']['report']['path']).parent.parent/'campaign_spec.json')
gate=load_json(u['reuse']['gate']['path'])
offline=file_ref(Path(gate['root'])/'targets/manifest.json')
m=load_json(offline['path']); validate_file_ref(m['payload'],root=Path(offline['path']).parent)
banks={'U100':dict(manifest=s['bank_manifest'],refs=s['teacher_refs']),
 'CONCAT':dict(manifest=file_ref(original.teacher_folder(old,'CONCAT')/'bank/manifest.json'),
   refs=original.teacher_refs(old,'CONCAT'),owner=file_ref(Path(original.teacher_spec(old,'CONCAT')['root'])/'campaign_spec.json')),
 'FUSION':dict(manifest=file_ref(worker.fusion_bank(s)/'manifest.json'),refs=original.teacher_refs(old,'FUSION'),
   owner=s['reuse']['parent_spec']),
 'OFFLINE':dict(manifest=offline,refs=s['anchor']['evidence']['models']['OFFLINE'])}
for item in banks.values():
 m=load_json(item['manifest']['path'])
 item['payload']=file_ref(validate_file_ref(m['payload'],root=Path(item['manifest']['path']).parent))
 item['manifest_hash']=m['content_hash']
print('MULTI3_MATCH_REUSE='+json.dumps(dict(reference=refs,banks=banks,job=receipt['job_id'],
 report_hash=report['content_hash'],u100=s['reuse']['u100']),sort_keys=True,allow_nan=False))
'''


def authenticate(path, project):
    path = Path(path).resolve(strict=True); parent = load_json(path)
    if (parent['contract'] != 'LUKA_FULLSTAT_MULTIVIEW_EXPORT_SPEC/v1'
            or parent['source_commit'] != PARENT_COMMIT or parent['counts'] != COUNTS
            or path != Path(parent['root'])/'campaign_spec.json'):
        raise ValueError('Expected completed ceb21461 Multi3 export-recovery campaign')
    original = Path(parent['project_dir'])
    if git(original,'rev-parse','HEAD') != PARENT_COMMIT or git(original,'status','--porcelain','--untracked-files=all'):
        raise ValueError('Historical source must remain clean and pinned')
    env = dict(os.environ,PYTHONPATH='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    print('Authenticating completed Multi3 and frozen TRAIN banks in original source...',flush=True)
    run = subprocess.run([sys.executable,'-B','-s','-c',REPLAY,str(path)],cwd=original,env=env,capture_output=True,text=True)
    if run.returncode: raise ValueError('Original-source replay failed:\n'+run.stderr)
    rows = [v.split('=',1)[1] for v in run.stdout.splitlines() if v.startswith('MULTI3_MATCH_REUSE=')]
    if len(rows) != 1: raise ValueError('Missing original-source evidence')
    # Pin every imported historical runtime byte as well as the new source snapshot.
    scientific = Path(load_json(parent['reuse']['parent_spec']['path'])['project_dir'])
    inventory = {}
    for tree in (original,scientific):
        for name in git(tree,'ls-files','src','scripts','sbatch').splitlines():
            if name.endswith(('.py','.sh')): inventory[str(tree/name)] = sha256_file(tree/name)
    return artifact('REUSE',parents=dict(spec=parent['content_hash']),parent_spec=file_ref(path),
        historical_source=inventory,**json.loads(rows[0]))


def check(reuse):
    validate(reuse,'REUSE')
    p = load_json(validate_file_ref(reuse['parent_spec']))
    if (p['content_hash'] != reuse['parents']['spec'] or p['source_commit'] != PARENT_COMMIT
            or p['counts'] != COUNTS or p['reuse']['u100'] != reuse['u100']):
        raise ValueError('Parent population/source changed')
    for path,digest in reuse['historical_source'].items():
        if sha256_file(path) != digest: raise ValueError('Historical runtime changed: '+path)
    for refs in [reuse['reference'],p['references']['M0HLT'],p['references']['OFFLINE'],p['references']['U100_W5']]:
        for key in ('report','receipt'): validate_file_ref(refs[key])
    for bank in reuse['banks'].values():
        m = load_json(validate_file_ref(bank['manifest']))
        if m['content_hash'] != bank['manifest_hash']: raise ValueError('Bank identity changed')
        for key in ('checkpoint','report'):validate_file_ref(bank['refs'][key])
        for key in ('receipt','predictions'):
            if key in bank['refs']:validate_file_ref(bank['refs'][key])
        if 'owner' in bank: validate_file_ref(bank['owner'])
    return p

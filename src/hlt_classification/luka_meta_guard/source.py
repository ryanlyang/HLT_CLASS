"""Read-only original-source authentication of the completed FIXED initializer."""
from pathlib import Path
import subprocess
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_attention_weight.source import replay
from .contracts import artifact, validate

BASE = 'c10d2205677bae4769d45da447c338165bc6724f'
PARENT_COMMIT = '7a6dc268800b0788e0e89d7ec56db37583397946'
MODULES = ('__init__','contracts','policy','training','source','campaign','worker','submission')
ADDITIONS = {'src/hlt_classification/luka_meta_guard/'+n+'.py' for n in MODULES} | {
    'scripts/luka_meta_guard.py','scripts/queue_luka_meta_guard.sh','sbatch/run_luka_meta_guard.sh'}
REQUIRED = ADDITIONS | {'tests/test_luka_meta_guard.py','docs/plans/LUKA_META_GUARD_PLAN.md',
    'docs/contracts/LUKA_META_GUARD.md','docs/LUKA_META_GUARD_RUNBOOK.md'}


def current(project, commit):
    project = Path(project).resolve(); snapshot = _source(project, commit)
    def git(*args):
        return subprocess.run(['git','-C',str(project),*args], check=True,capture_output=True,text=True).stdout.splitlines()
    missing = REQUIRED-set(git('ls-files'))
    if missing: raise ValueError('Commit complete guarded implementation: '+str(sorted(missing)))
    changed = git('diff','--name-status','--no-renames',BASE,commit,'--','src','scripts','sbatch')
    if any(not row.startswith('A\t') or row[2:] not in ADDITIONS for row in changed):
        raise ValueError('Existing runtime changed; use an additive guarded implementation')
    git('merge-base','--is-ancestor',commit,'origin/main')
    if Path(__file__).resolve() != project/'src/hlt_classification/luka_meta_guard/source.py':
        raise ValueError('Imported source escapes pinned project')
    return snapshot


REPLAY = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_meta_embedding import campaign,worker,submission
from hlt_classification.luka_fullsim.science_campaign import preflight_accounting
s=load_json(sys.argv[1]); campaign.check(s)
root=Path(s['root']); jobs=submission.accepted_jobs(s)
for task in ('setup','fixed'):
    row=worker.validate_task(s,task)
    if row['job_id']!=jobs[task]: raise ValueError('Parent job not bound to journal')
    preflight_accounting(jobs[task])
record,arrays=worker.validate_fit(root/'fixed/fit',s,'fixed')
names=dict(state='fixed/fit/training_state.pt',native='fixed/fit/selected.pt',
    report='fixed/fit/training_report.json',predictions='fixed/fit/validation_predictions.npz',
    fixed_receipt='fixed/receipt.json',setup_receipt='setup/receipt.json',
    bank_manifest='targets/manifest.json',bank='targets/targets.npz',classifier='initial/classifier.pt',
    original50k='references/original50k.npz')
names.update({n:'references/'+n+'.npz' for n in worker.REFS})
print('META_GUARD_PARENT='+json.dumps(dict(files={k:file_ref(root/n) for k,n in names.items()},
    anchor=s['anchor'],node=s['node'],counts=s['counts'],jobs={k:jobs[k] for k in ('setup','fixed')},
    selected=record['report']['selected'],bank_hash=load_json(root/'targets/manifest.json')['content_hash']),
    sort_keys=True,allow_nan=False))
'''


def authenticate(path):
    path = Path(path).resolve(strict=True); original = load_json(path)
    if original['source_commit'] != PARENT_COMMIT or path != Path(original['root'])/'campaign_spec.json':
        raise ValueError('Expected the original 7a6dc268 FIXED/META campaign')
    if not (path.parent/'fixed/receipt.json').is_file():
        raise ValueError('Original FIXED must finish and publish its selected full state; do not cancel it or substitute native-only w5')
    validate_source_snapshot(original['source_snapshot'], repository=original['project_dir'],require_clean=True)
    print('Authenticating completed FIXED in its original source; META need not finish...', flush=True)
    ref, evidence = replay(path, original['project_dir'], REPLAY, 'META_GUARD_PARENT=')
    for f in evidence['files'].values(): validate_file_ref(f)
    return artifact('PARENT', spec=ref, **evidence)


def validate_parent(value):
    validate(value,'PARENT'); validate_file_ref(value['spec'])
    old = load_json(value['spec']['path'])
    if (old['source_commit'] != PARENT_COMMIT or old['anchor'] != value['anchor']
            or old['node'] != value['node'] or old['counts'] != value['counts']):
        raise ValueError('Original source/teacher/population differs')
    for f in value['files'].values(): validate_file_ref(f)
    return old

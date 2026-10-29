"""Explicit completed-FIXED or completed-setup reuse; never partial-fit recovery."""
from pathlib import Path
import subprocess
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_attention_weight.source import replay
from .contracts import artifact, validate, INITIALIZERS

BASE = 'b9cf63b0bf4a037e11ece91da21db465d2347b96'
PARENT_COMMIT = '7a6dc268800b0788e0e89d7ec56db37583397946'
MODULES = ('__init__','contracts','policy','training','source','campaign','worker','submission')
OWNED_RUNTIME = {'src/hlt_classification/luka_meta_guard/'+n+'.py' for n in MODULES} | {
    'scripts/luka_meta_guard.py','scripts/queue_luka_meta_guard.sh','sbatch/run_luka_meta_guard.sh'}
REQUIRED = OWNED_RUNTIME | {'tests/test_luka_meta_guard.py','docs/plans/LUKA_META_GUARD_PLAN.md',
    'docs/contracts/LUKA_META_GUARD.md','docs/LUKA_META_GUARD_RUNBOOK.md'}


def current(project, commit):
    project = Path(project).resolve(); snapshot = _source(project, commit)
    def git(*args):
        return subprocess.run(['git','-C',str(project),*args], check=True,capture_output=True,text=True).stdout.splitlines()
    missing = REQUIRED-set(git('ls-files'))
    if missing: raise ValueError('Commit complete guarded implementation: '+str(sorted(missing)))
    git('merge-base','--is-ancestor',BASE,commit)
    changed = git('diff','--name-status','--no-renames',BASE,commit,'--','src','scripts','sbatch')
    if any(row[:2] not in ('A\t','M\t') or row[2:] not in OWNED_RUNTIME for row in changed):
        raise ValueError('Existing runtime changed; only the guarded implementation may change')
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
mode=sys.argv[2]
if mode not in ('completed_fixed','shared_w5'): raise ValueError('Unknown initializer source')
root=Path(s['root']); jobs=submission.accepted_jobs(s)
tasks=('setup','fixed') if mode=='completed_fixed' else ('setup',)
for task in tasks:
    row=worker.validate_task(s,task)
    if row['job_id']!=jobs[task]: raise ValueError('Parent job not bound to journal')
    preflight_accounting(jobs[task])
names=dict(setup_receipt='setup/receipt.json',
    bank_manifest='targets/manifest.json',bank='targets/targets.npz',classifier='initial/classifier.pt',
    original50k='references/original50k.npz')
names.update({n:'references/'+n+'.npz' for n in worker.REFS})
if mode=='completed_fixed':
    record,arrays=worker.validate_fit(root/'fixed/fit',s,'fixed')
    selected=record['report']['selected']
    names.update(state='fixed/fit/training_state.pt',native='fixed/fit/selected.pt',
        report='fixed/fit/training_report.json',predictions='fixed/fit/validation_predictions.npz',
        fixed_receipt='fixed/receipt.json')
else:
    # Authenticates both full-state outputs and the original three-pass head
    # calibration. No fixed/meta fit file, receipt, or completion is consulted.
    worker.read_shared(s)
    selected=None
    names.update(state='initial/student.pt',initializer='setup/initializer.json',
        predictions='references/OFFLINE_EMBED_W5.npz')
files={k:file_ref(root/n) for k,n in names.items()}
if mode=='shared_w5': files['native']=s['initial_checkpoint']
print('META_GUARD_PARENT='+json.dumps(dict(files=files,initializer_source=mode,
    anchor=s['anchor'],node=s['node'],counts=s['counts'],jobs={k:jobs[k] for k in tasks},
    selected=selected,bank_hash=load_json(root/'targets/manifest.json')['content_hash']),
    sort_keys=True,allow_nan=False))
'''


def authenticate(path, initializer_source='completed_fixed'):
    if initializer_source not in INITIALIZERS:
        raise ValueError('Unknown initializer source; no automatic fallback')
    path = Path(path).resolve(strict=True); original = load_json(path)
    if original['source_commit'] != PARENT_COMMIT or path != Path(original['root'])/'campaign_spec.json':
        raise ValueError('Expected the original 7a6dc268 FIXED/META campaign')
    if initializer_source=='completed_fixed' and not (path.parent/'fixed/receipt.json').is_file():
        raise ValueError('Original FIXED must finish for completed_fixed; explicitly choose shared_w5 to reuse completed setup, not cancelled fit outputs')
    if not (path.parent/'setup/receipt.json').is_file():
        raise ValueError('Completed original setup receipt required; cannot reuse unauthenticated initializer or targets')
    validate_source_snapshot(original['source_snapshot'], repository=original['project_dir'],require_clean=True)
    print('Authenticating '+initializer_source+' in original source; no cancelled fit outputs reused...', flush=True)
    ref, evidence = replay(path, original['project_dir'], REPLAY, 'META_GUARD_PARENT=',extra=(initializer_source,))
    if evidence['initializer_source']!=initializer_source:
        raise ValueError('Replayed initializer source differs')
    for f in evidence['files'].values(): validate_file_ref(f)
    value=artifact('PARENT', spec=ref, **evidence)
    validate_parent(value)
    return value


def validate_parent(value):
    validate(value,'PARENT'); validate_file_ref(value['spec'])
    old = load_json(value['spec']['path'])
    mode=value['initializer_source']
    if mode not in INITIALIZERS:
        raise ValueError('Unknown initializer source')
    if (old['source_commit'] != PARENT_COMMIT or old['anchor'] != value['anchor']
            or old['node'] != value['node'] or old['counts'] != value['counts']):
        raise ValueError('Original source/teacher/population differs')
    root=Path(old['root'])
    names=dict(setup_receipt='setup/receipt.json',bank_manifest='targets/manifest.json',
        bank='targets/targets.npz',classifier='initial/classifier.pt',original50k='references/original50k.npz')
    names.update({n:'references/'+n+'.npz' for n in ('M0HLT','OFFLINE','OFFLINE_EMBED_W5')})
    if mode=='completed_fixed':
        names.update(state='fixed/fit/training_state.pt',native='fixed/fit/selected.pt',
            report='fixed/fit/training_report.json',predictions='fixed/fit/validation_predictions.npz',
            fixed_receipt='fixed/receipt.json')
        jobs={'setup','fixed'}
        if value['selected']!=load_json(root/names['report'])['report']['selected']:
            raise ValueError('Parent selected FIXED evidence differs')
    else:
        names.update(state='initial/student.pt',initializer='setup/initializer.json',
            predictions='references/OFFLINE_EMBED_W5.npz')
        jobs={'setup'}
        if value['selected'] is not None:
            raise ValueError('Shared w5 initializer cannot claim a selected continuation epoch')
    expected={k:file_ref(root/n) for k,n in names.items()}
    if mode=='shared_w5': expected['native']=old['initial_checkpoint']
    if (value['files']!=expected or set(value['jobs'])!=jobs
            or value['bank_hash']!=load_json(root/'targets/manifest.json')['content_hash']):
        raise ValueError('Initializer inventory/lineage differs')
    for f in value['files'].values(): validate_file_ref(f)
    return old

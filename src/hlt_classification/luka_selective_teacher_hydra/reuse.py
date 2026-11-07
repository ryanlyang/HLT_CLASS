"""Authenticate completed T00/S00 with their own pinned implementation."""
from pathlib import Path
import subprocess
import sys
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullsim.regular_submission import clean_environment


def original(path):
    path = Path(path).resolve(strict=True)
    old = load_json(path)
    if old.get('contract') != 'LUKA_TEACHER_SWEEP_SPEC/v1':
        raise ValueError('Use the original teacher-sweep spec with completed T00 and S00')
    code = '''import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[2])/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_teacher_sweep_hydra import campaign,worker,training
s=load_json(sys.argv[1]); campaign.check(s,check_storage=False)
for t in ('T00','S00'):
    worker.receipt(s,t)
    r,a=worker.validate_teacher(s,t) if t=='T00' else training.validate_fit(Path(s['root'])/t/'fit',s,t)
    if r['parents']!=worker.fit_parents(s,t): raise ValueError('Original fit parents differ')
print('SELECTIVE_PARENT='+s['content_hash'])
'''
    env = clean_environment()
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(Path(old['project_dir'])/'src'))
    run = subprocess.run([sys.executable, '-B', '-s', '-c', code, str(path), old['project_dir']],
        cwd=old['project_dir'], env=env, capture_output=True, text=True)
    if run.returncode or run.stdout.splitlines().count('SELECTIVE_PARENT='+old['content_hash']) != 1:
        raise ValueError('Original T00/S00 authentication failed:\n'+run.stdout+run.stderr)
    return old


def references(old):
    root = Path(old['root'])
    return {key: file_ref(root/name) for key, name in dict(
        offline_checkpoint='T00/fit/selected.pt', offline_bank='T00/bank/manifest.json',
        offline_targets='T00/bank/targets.npz', offline_predictions='T00/fit/validation_predictions.npz',
        extra_control_predictions='S00/fit/validation_predictions.npz',
        teacher_receipt='T00/receipt.json', student_receipt='S00/receipt.json').items()}


def check_refs(refs):
    for ref in refs.values(): validate_file_ref(ref)

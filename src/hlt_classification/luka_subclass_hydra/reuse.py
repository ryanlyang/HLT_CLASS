"""Read-only authentication of the small-data T00 teacher in its original code."""
from pathlib import Path
import subprocess
import sys
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullsim.regular_submission import clean_environment


def original(path):
    path = Path(path).resolve(strict=True); old = load_json(path)
    if old.get('contract') != 'LUKA_TEACHER_SWEEP_SPEC/v1':
        raise ValueError('Use the small-data teacher-sweep spec with completed T00')
    code = '''import sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[2])/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_teacher_sweep_hydra import campaign,worker,contracts
s=load_json(sys.argv[1]); campaign.check(s,check_storage=False)
r,a=worker.validate_teacher(s,'T00')
if r['parents']!=worker.fit_parents(s,'T00'): raise ValueError('Teacher fit parents differ')
if contracts.recipe()['teachers']['T00']!='ce': raise ValueError('T00 is not ordinary CE')
print('SUBCLASS_PARENT='+s['content_hash'])
'''
    env = clean_environment()
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(Path(old['project_dir'])/'src'))
    run = subprocess.run([sys.executable, '-B', '-s', '-c', code, str(path), old['project_dir']],
        cwd=old['project_dir'], env=env, capture_output=True, text=True)
    if run.returncode or run.stdout.splitlines().count('SUBCLASS_PARENT='+old['content_hash']) != 1:
        raise ValueError('Original small-data T00 authentication failed:\n'+run.stdout+run.stderr)
    return old


def references(old):
    root = Path(old['root'])/'T00'
    return {k: file_ref(root/p) for k,p in dict(checkpoint='fit/selected.pt',
        bank='bank/manifest.json', targets='bank/targets.npz', report='fit/training_report.json',
        receipt='receipt.json').items()}


def check_refs(refs):
    if set(refs) != {'checkpoint','bank','targets','report','receipt'}:
        raise ValueError('T00 reference inventory differs')
    for ref in refs.values():
        validate_file_ref(ref)

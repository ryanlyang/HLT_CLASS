"""Read-only original-worktree authentication; never import an old source guard here."""
from pathlib import Path
import subprocess
import sys
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullsim.regular_submission import clean_environment
from hlt_classification.luka_selective_teacher_hydra.contracts import validate
from .contracts import MEMBERS


def original(path):
    path = Path(path).resolve(strict=True)
    old = load_json(path)
    validate(old, 'SPEC')
    code = '''import sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[2])/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_selective_teacher_hydra import campaign,worker,training
s=load_json(sys.argv[1]); campaign.check(s,check_storage=False)
worker.validate_teacher(s,'IT')
for t in ('S2IT','S4','S6'):
    worker.receipt(s,t)
    r,a=training.validate_fit(Path(s['root'])/t/'fit',s,t)
    if r['parents']!=worker.fit_parents(s,t): raise ValueError('Original target lineage differs')
print('CAPACITY_PARENT='+s['content_hash'])
'''
    env = clean_environment()
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(Path(old['project_dir'])/'src'))
    run = subprocess.run([sys.executable, '-B', '-s', '-c', code, str(path), old['project_dir']],
                         cwd=old['project_dir'], env=env, capture_output=True, text=True)
    if run.returncode or run.stdout.splitlines().count('CAPACITY_PARENT='+old['content_hash']) != 1:
        raise ValueError('Original selective campaign authentication failed:\n'+run.stdout+run.stderr)
    return old


def references(old):
    root = Path(old['root'])
    names = dict(it_checkpoint='IT/fit/selected.pt', it_bank='IT/bank/manifest.json',
                 it_targets='IT/bank/targets.npz', it_receipt='IT/receipt.json')
    for t in MEMBERS:
        names[t+'_predictions'] = t+'/fit/validation_predictions.npz'
        names[t+'_receipt'] = t+'/receipt.json'
        names[t+'_report'] = t+'/fit/training_report.json'
    return {k: file_ref(root/p) for k, p in names.items()}


def check_refs(refs):
    for ref in refs.values():
        validate_file_ref(ref)

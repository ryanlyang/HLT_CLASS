"""Authenticate input ancestry inside the original pinned teacher-sweep worktree."""
from pathlib import Path
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json,validate_content_hash
from hlt_classification.luka_fullsim.regular_submission import clean_environment


def original(path):
    path=Path(path).resolve(strict=True); old=load_json(path)
    validate_content_hash(old,expected_contract='LUKA_TEACHER_SWEEP_SPEC/v1',expected_schema_version=1)
    code='''import sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[2])/'src'))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_teacher_sweep_hydra import campaign,inputs
s=load_json(sys.argv[1]); campaign.check(s,check_storage=False)
inputs.delta(s).verify_all()
print('COMPLEMENTARY_PARENT='+s['content_hash'])
'''
    env=clean_environment(); env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(Path(old['project_dir'])/'src'))
    run=subprocess.run([sys.executable,'-B','-s','-c',code,str(path),old['project_dir']],cwd=old['project_dir'],env=env,capture_output=True,text=True)
    if run.returncode or run.stdout.splitlines().count('COMPLEMENTARY_PARENT='+old['content_hash'])!=1:
        raise ValueError('Original teacher-sweep input authentication failed:\n'+run.stdout+run.stderr)
    return old

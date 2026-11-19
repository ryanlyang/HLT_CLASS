"""One-file authenticated cleanup of unused old Hydra contrastive TRAIN keys."""
from pathlib import Path
import argparse
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hlt_classification.operations import hydra_contrast_key_retirement as operation


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--authorize')
    a=p.parse_args();project=Path(__file__).resolve().parents[1]
    paths=['scripts/retire_luka_hydra_contrast_keys.py','src/hlt_classification/operations/hydra_contrast_key_retirement.py']
    tracked=set(subprocess.check_output(['git','-C',str(project),'ls-files','--',*paths],text=True).splitlines())
    if set(paths)!=tracked:raise ValueError('Commit both cleanup helper files before execution')
    status=subprocess.check_output(['git','-C',str(project),'status','--porcelain','--untracked-files=all','--',*paths],text=True)
    if status:raise ValueError('Use committed unchanged cleanup helpers')
    commit=subprocess.check_output(['git','-C',str(project),'rev-parse','HEAD'],text=True).strip()
    operation.retire(execute=a.execute,authorization=a.authorize,source_commit=commit)


if __name__=='__main__':main()

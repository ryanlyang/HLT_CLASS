"""Load original immutable science first; adapt only declared I/O/launch hooks."""
from functools import partial
import importlib
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace
from . import contracts as c

_INSTALLED=None


def required_files():
    return [*(f'src/hlt_classification/luka_d20_input_recovery/{name}.py' for name in
        ('__init__','contracts','reader','decode','runtime','controller','probe')),
        'scripts/recover_luka_d20_inputs.py','scripts/launch_luka_d20_input_recovery.sh',
        'sbatch/run_luka_d20_input_recovery.sh','tests/test_luka_d20_input_recovery.py',
        'src/hlt_classification/luka_d20_input_recovery/donors.json',
        'docs/LUKA_D20_INPUT_RECOVERY.md','docs/contracts/LUKA_D20_INPUT_RECOVERY.md',
        'docs/plans/LUKA_D20_FULL_HYDRA_PLAN.md']


def source(project):
    project=Path(project).resolve(strict=True)
    c.require(Path(__file__).resolve()==project/'src/hlt_classification/luka_d20_input_recovery/runtime.py',
        'Repair imported outside the declared project')
    def git(*args):return subprocess.check_output(['git','-C',str(project),*args],timeout=60)
    c.require(not git('status','--porcelain','--untracked-files=all').strip(),'Repair source is dirty')
    commit=git('rev-parse','HEAD').decode().strip()
    c.require(re.fullmatch('[0-9a-f]{40}',commit),'Exact repair commit required')
    subprocess.run(['git','-C',str(project),'merge-base','--is-ancestor',commit,'origin/main'],check=True,timeout=30)
    tracked=set(git('ls-files').decode().splitlines());files=required_files()
    c.require(set(files)<=tracked,'Commit the complete repair, wrappers, tests and documents')
    return dict(commit=commit,files={name:c.digest(project/name) for name in files})


def original(spec,*,verify=True):
    original_spec=c.read(c.referenced(spec['parent_spec']))
    c.require(original_spec['root']==str(c.PARENT) and original_spec['commit']==c.PARENT_COMMIT,
        'Unexpected original full D20 execution')
    original_src=Path(original_spec['project'])/'src';repair_src=Path(spec['project'])/'src'
    # Keep the recovery namespace available to spawn while all science resolves
    # from its original worktree. No original source file is modified.
    import hlt_classification
    hlt_classification.__path__=[str(original_src/'hlt_classification'),str(repair_src/'hlt_classification')]
    sys.path[:0]=[str(original_src),str(repair_src)]
    names=('campaign','contracts','resources','inputs','basket_reader','worker','launch','training','population','artifacts','kernel')
    modules={name:importlib.import_module('hlt_classification.luka_d20_full_hydra.'+name) for name in names}
    for name,module in modules.items():
        c.require(Path(module.__file__).resolve()==original_src/'hlt_classification/luka_d20_full_hydra'/(name+'.py'),
            'Original science preloaded from another source: '+name)
    old=SimpleNamespace(spec=original_spec,**modules)
    if verify:
        import json
        pins=json.loads((Path(spec['project'])/'src/hlt_classification/luka_d20_input_recovery/donors.json').read_bytes())
        c.require(pins['commit']==c.PARENT_COMMIT,'Repair donor commit differs')
        for name,digest in pins['files'].items():
            c.require(c.digest(Path(original_spec['project'])/name)==digest,'Original repair donor differs: '+name)
        old.kernel.configure(1,12);old.campaign.check(original_spec)
    return old


def check(spec,*,light=False):
    c.validate(spec,'SPEC')
    c.require(spec['policy']==c.policy(),'Repair execution policy changed')
    c.require(c.read(Path(spec['root'])/'repair_spec.json')==spec,'Canonical repair spec differs')
    root,project=Path(spec['root']),Path(spec['project'])
    c.require(root.is_relative_to(c.BASE/'runs') and project.is_relative_to(c.BASE)
        and root.resolve()==root and project.resolve()==project
        and not any(p.is_symlink() for path in (root,project) for p in (path,*path.parents)),
        'Unexpected recovery location')
    if light:
        for name,digest in spec['source']['files'].items():
            c.require(c.digest(Path(spec['project'])/name)==digest,'Repair file changed: '+name)
    else:c.require(source(spec['project'])==spec['source'],'Repair source changed')
    c.referenced(spec['qualification']);c.referenced(spec['qualification_task'])
    return original(spec,verify=not light)


def parity(spec,old):
    record=c.read(Path(spec['root'])/'reader_parity.json');c.validate(record,'READER_PARITY')
    c.require(record['parents']==dict(repair=spec['content_hash'],qualification=spec['qualification_hash'])
        and record['passed'] is True and record['cases']==len(record['records'])==12 and record['models_updated'] is False,
        'Reader equivalence proof differs')
    files={r['path'] for r in record['records']}
    c.require(len(files)==2 and c.BAD_FILE in files and any(p.startswith('raw/') and not p.startswith('raw/qcd/') for p in files),
        'Reader proof must cover the timed-out file and a signal file')
    for path in files:
        rows=[r for r in record['records'] if r['path']==path]
        c.require(len(rows)==6 and {(r['role'],r['coordinate']) for r in rows}==
            {(role,coordinate) for role in (0,1) for coordinate in ('D000','OFFLINE','U100')}
            and len({r['sha256'] for r in rows})==1
            and all(re.fullmatch('[0-9a-f]{64}',r['sha256']) and r['byte_equal'] is True
                and type(r['start']) is int and type(r['stop']) is int and 0<=r['start']<r['stop']<=r['start']+4
                and r['targets'] is (r['role']==0 and r['coordinate']=='D000') for r in rows),
            'Reader proof role/view/target coverage differs')
    accepted=c.bootstrap_receipt(spec,'probe')
    c.require(record['job']==accepted['job'],'Reader probe job binding differs')
    return record


def science_binding(spec,old,task):
    job=os.environ.get('SLURM_JOB_ID','')
    c.require(re.fullmatch('[1-9][0-9]*',job),'A registered replacement allocation is required')
    for _ in range(30):
        rows=[r for r in old.campaign.accepted_jobs(old.spec,unambiguous=False) if r['job']==job and r['task']==task]
        if len(rows)==1:break
        time.sleep(1)
    c.require(len(rows)==1 and rows[0].get('execution_recovery')==spec['content_hash']
        and rows[0].get('execution_spec')==c.ref(Path(spec['root'])/'repair_spec.json'),
        'Replacement execution/spec binding differs')
    return job


def install(spec,*,light=False):
    global _INSTALLED
    if _INSTALLED is not None:
        c.require(_INSTALLED[0]==spec['content_hash'],'Another repair is installed in this process')
        return _INSTALLED[1]
    old=check(spec,light=light)
    from .reader import make_reader
    from . import decode
    old.basket_reader.close_reader()
    old.basket_reader._d20_original_reader=old.basket_reader.Reader
    old.basket_reader.Reader=make_reader(old.basket_reader.Reader)
    repair_path=str(Path(spec['root'])/'repair_spec.json')
    def journal(row):
        job=os.environ.get('SLURM_JOB_ID','probe');task=os.environ.get('D20_INPUT_TASK','probe')
        path=Path(spec['root'])/'retries'/job/f'{task}-{row["restart"]:02d}.json'
        c.publish(path,c.artifact('INPUT_RETRY',parents=dict(repair=spec['content_hash']),job=job,task=task,**row))
    old.inputs.decoded=partial(decode.decoded,repair_path=repair_path,on_retry=journal)
    native=old.launch.command
    def command(project,root,task,node,*,startup_only=False):
        c.require(project==old.spec['project'] and str(root)==old.spec['root'],'Launch escaped original execution')
        argv=native(project,root,task,node,startup_only=startup_only);cut=argv.index('--')+1
        return argv[:cut]+[str(Path(spec['project'])/'scripts/recover_luka_d20_inputs.py'),
            'rank','--repair-spec',repair_path,'--task',task,'--node',str(node)]+(['--startup-only'] if startup_only else [])
    old.launch.command=command
    _INSTALLED=(spec['content_hash'],old)
    return old


def run(spec,task,*,node=None,startup_only=False):
    os.environ['D20_INPUT_TASK']=task
    old=install(spec,light=node is not None);proof=parity(spec,old);job=science_binding(spec,old,task)
    c.require(task in old.contracts.GPU_TASKS or task=='report','Not a recoverable scientific task')
    if node is not None:return old.worker.rank(old.spec,task,node,startup_only=startup_only)
    evidence=c.artifact('EXECUTION_OVERRIDE',parents=dict(spec=old.spec['content_hash'],
        qualification=spec['qualification_hash'],repair=spec['content_hash'],reader_parity=proof['content_hash']),
        task=task,job=job,policy=spec['policy'],scientific_parents='unchanged_authenticated_original_study')
    c.publish(Path(spec['root'])/'executions'/(job+'.json'),evidence)
    return old.worker.run(old.spec,task)

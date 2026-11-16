"""Reuse completed balanced teachers through their authenticated original source."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from hlt_classification.operations import fullstat_pt_metadata as meta
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_fullstat_balanced_ladder import contracts as old, source as donor
from . import contracts as c

PARENT_COMMIT='ff22b20f27fdc67d0fbd1a1df7005eb4626d19b7'
RUNTIME=donor.RUNTIME
runtime=donor.runtime
PACKAGE='src/hlt_classification/luka_fullstat_balanced_main_kd/'
MODULES=('__init__','contracts','source','campaign','model','training','startup','worker',
         'artifacts','submission','execution','launch','recovery','inspect')
REQUIRED={PACKAGE+n+'.py' for n in MODULES}|{
    'scripts/luka_fullstat_balanced_main_kd.py','scripts/launch_luka_fullstat_balanced_main_kd.sh',
    'scripts/queue_luka_fullstat_balanced_main_kd.sh','sbatch/run_luka_fullstat_balanced_main_kd.sh',
    'tests/test_luka_fullstat_balanced_main_kd.py','tests/test_luka_fullstat_balanced_main_kd_workflow.py',
    'tests/test_luka_fullstat_balanced_main_kd_launch.py',
    'docs/plans/LUKA_FULLSTAT_BALANCED_MAIN_KD_PLAN.md','docs/contracts/LUKA_FULLSTAT_BALANCED_MAIN_KD.md',
    'docs/LUKA_FULLSTAT_BALANCED_MAIN_KD_RUNBOOK.md','docs/LUKA_FULLSTAT_BALANCED_MAIN_KD_VALIDATION.md'}


def current(project,commit):
    project=Path(project).resolve(strict=True);snapshot=_source(project,commit)
    if Path(__file__).resolve()!=project/(PACKAGE+'source.py'):
        raise ValueError('Imported main-KD source outside pinned project')
    names=subprocess.run(['git','-C',str(project),'ls-files'],check=True,capture_output=True,text=True).stdout.splitlines()
    if not REQUIRED<=set(names):raise ValueError('Commit the complete main-KD implementation')
    files={n:meta.file_ref(project/n,root=project) for n in names
           if n.startswith(('src/','scripts/','sbatch/','configs/')) or n in REQUIRED}
    return c.artifact('SOURCE',snapshot=snapshot,commit=commit,files=files)


def closure(project):
    """Conservative static local-import closure, including imports inside functions."""
    project=Path(project);pending=['hlt_classification.luka_fullstat_balanced_ladder.worker',
        'hlt_classification.luka_fullstat_balanced_ladder.inspect',
        'hlt_classification.luka_fullstat_balanced_ladder.source']
    seen=set();paths=set()
    while pending:
        name=pending.pop()
        if name in seen or not name.startswith('hlt_classification'):continue
        seen.add(name);stem=project/'src'/name.replace('.','/')
        path=stem.with_suffix('.py') if stem.with_suffix('.py').is_file() else stem/'__init__.py'
        if not path.is_file():continue
        paths.add(path.relative_to(project).as_posix())
        package=name if path.name=='__init__.py' else name.rpartition('.')[0]
        parts=name.split('.')
        pending.extend('.'.join(parts[:i]) for i in range(1,len(parts)))
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node,ast.Import):pending.extend(a.name for a in node.names)
            elif isinstance(node,ast.ImportFrom):
                parent=package.split('.')[:len(package.split('.'))-node.level+1] if node.level else []
                module='.'.join(parent+([node.module] if node.module else []))
                pending.append(module);pending.extend(module+'.'+a.name for a in node.names if a.name!='*')
    paths.add('src/hlt_classification/luka_fullstat_balanced_ladder/occupancy.json')
    return sorted(paths)


def boundary(parent,path):
    old.validate(parent,'SPEC')
    if (parent['source_commit']!=PARENT_COMMIT or parent['counts']!=c.COUNTS
            or parent['recipe']!=old.recipe() or parent['runtime_prefix']!=RUNTIME
            or parent['completed'] or parent['continuations']
            or Path(path)!=Path(parent['root'])/'campaign_spec.json'):
        raise ValueError('Expected original ff22b20f balanced campaign')


def original_source(parent,project):
    original=Path(parent['project_dir'])
    validate_source_snapshot(parent['source']['snapshot'],repository=original,require_clean=True)
    commit=subprocess.run(['git','-C',str(original),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    if commit!=PARENT_COMMIT:raise ValueError('Original balanced source commit differs')
    hashes={}
    for name in closure(original):
        ref=parent['source']['files'][name]
        meta.validate_file_ref(ref,root=original)
        meta.validate_file_ref(ref,root=project)
        hashes[name]=ref
    return original,hashes


REPLAY=r'''
import json,sys
from pathlib import Path
from hlt_classification.operations import fullstat_pt_metadata as m
from hlt_classification.luka_fullstat_balanced_ladder import campaign,artifacts,source,population,banks,contracts
p=Path(sys.argv[1]);s=m.load_json(p);campaign.check(s)
source.replay(s['inputs'])
pop,_,_=population.load(s['population_root'],input_hash=s['inputs']['content_hash'],prepared_hash=s['inputs']['evidence']['prepared'])
if pop!=s['population']:raise ValueError('Registered population changed')
refs={}
for stage in ('OFFLINE','M0HLT','CONCAT','UE100','DE050'):
    result=artifacts.completed(s,stage)
    bank=m.load_json(m.validate_file_ref(result['receipt']['files']['bank']))
    contracts.validate(bank,'BANK',banks.parents(s,stage,result['refs']))
    if (bank['stage']!=stage or bank['role']!='train' or bank['temperature']!=2.
            or bank['rows']!=s['eligible_train_rows'] or bank['refs']!=result['refs']):
        raise ValueError('Frozen bank scientific binding differs')
    for ref in bank['payloads'].values():m.validate_file_ref(ref,root=result['folder']/'bank')
    refs[stage]=dict(receipt=m.file_ref(result['folder']/'task_receipt.json'),files=result['receipt']['files'])
print('BALANCED_MAIN_KD_REUSE='+json.dumps(refs,sort_keys=True,allow_nan=False))
'''


def authenticate(path,project):
    path=Path(path).resolve(strict=True);parent=meta.load_json(path);boundary(parent,path)
    original,hashes=original_source(parent,project)
    env=dict(os.environ,PYTHONPATH=str(original/'src'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    print('Authenticating five completed balanced references and frozen TRAIN banks; no teacher fits or preflight.',flush=True)
    run=subprocess.run([sys.executable,'-B','-s','-c',REPLAY,str(path)],cwd=original,env=env,capture_output=True,text=True)
    if run.returncode:raise ValueError('Original balanced authentication failed:\n'+run.stderr)
    lines=[x.split('=',1)[1] for x in run.stdout.splitlines() if x.startswith('BALANCED_MAIN_KD_REUSE=')]
    if len(lines)!=1:raise ValueError('Missing original-source reuse evidence')
    value=c.artifact('REUSE',parents=dict(spec=parent['content_hash']),spec=meta.file_ref(path),
                     references=json.loads(lines[0]),compatible_source=hashes)
    check(value,project,source_check=False)
    return value


def check(value,project,*,source_check=True):
    c.validate(value,'REUSE')
    parent=meta.load_json(meta.validate_file_ref(value['spec']));boundary(parent,Path(value['spec']['path']))
    c.validate(value,'REUSE',dict(spec=parent['content_hash']))
    if set(value['references'])!=set(c.REFERENCES):raise ValueError('Five exact balanced references required')
    if source_check:
        _,hashes=original_source(parent,project)
        if hashes!=value['compatible_source']:raise ValueError('Compatible original runtime changed')
    for stage,refs in value['references'].items():
        expected=Path(parent['root'])/('train_'+stage)/'task_receipt.json'
        if Path(refs['receipt']['path'])!=expected:raise ValueError('Reference path differs')
        receipt=meta.load_json(meta.validate_file_ref(refs['receipt']));old.validate(receipt,'TASK_RECEIPT')
        if (receipt['stage']!=stage or receipt['parents']['spec']!=parent['content_hash']
                or receipt['population']!=parent['population']['content_hash'] or receipt['files']!=refs['files']):
            raise ValueError('Reference receipt binding differs')
    return parent


def replay(value,project):
    actual=authenticate(value['spec']['path'],project)
    if actual!=value:raise ValueError('Frozen balanced references changed')
    return check(value,project,source_check=False)

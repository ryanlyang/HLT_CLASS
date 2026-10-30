"""Authenticate the completed corrected Oscar readout and original U100 lineage."""
from pathlib import Path
import json
import os
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json, sha256_file
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_fullstat_readout.source import runtime, RUNTIME
from .contracts import artifact, COUNTS

DONOR = 'db9afe8b87cdef6f3489f9b85603f52cf7d2084f'
BASELINE = '74cba41f4938df8aedcee4f931ef14bfc79b839d'
MODULES = ('__init__','contracts','source','campaign','model','training','worker','execution','submission','launch','views','targets','startup')
ADDITIONS = {f'src/hlt_classification/luka_fullstat_multiview/{n}.py' for n in MODULES}
ENTRYPOINTS = {'scripts/luka_fullstat_multiview.py','scripts/launch_luka_fullstat_multiview.sh',
              'scripts/queue_luka_fullstat_multiview.sh','sbatch/run_luka_fullstat_multiview.sh'}
REQUIRED = ADDITIONS | ENTRYPOINTS | {'tests/test_luka_fullstat_multiview.py',
    'tests/test_luka_fullstat_multiview_launch.py','docs/plans/LUKA_FULLSTAT_MULTIVIEW_PLAN.md',
    'docs/contracts/LUKA_FULLSTAT_MULTIVIEW.md','docs/LUKA_FULLSTAT_MULTIVIEW_RUNBOOK.md'}


def compatible_changes(changed,added):
    return changed<=ADDITIONS|ENTRYPOINTS and added==changed


def current(project,commit):
    project=Path(project).resolve(); snapshot=_source(project,commit)
    def git(*args):
        return subprocess.run(['git','-C',str(project),*args],check=True,capture_output=True,text=True).stdout
    git('merge-base','--is-ancestor',commit,'origin/main')
    tracked=set(git('ls-files').splitlines())
    changed=set(git('diff','--name-only',BASELINE,commit,'--','src','scripts','sbatch').splitlines())
    added=set(git('diff','--name-only','--diff-filter=A',BASELINE,commit,'--','src','scripts','sbatch').splitlines())
    if (not REQUIRED<=tracked or not compatible_changes(changed,added)
            or Path(__file__).resolve()!=project/'src/hlt_classification/luka_fullstat_multiview/source.py'):
        raise ValueError('Source changes outside additive FullStat multiview namespace')
    return snapshot


REPLAY = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullstat_readout import campaign,worker,submission
s=load_json(sys.argv[1]); campaign.check(s); rec=worker.completed(s)
report,arrays=worker.validate_fit(s)
root=Path(s['root']); previous=s['reuse']
p=load_json(previous['anchor']['spec']['path'])
oracle=next(n for n in p['nodes'] if n['node_id']=='OFFLINE')
e=dict(u100=previous, oracle_node=oracle, readout_reference={k:file_ref(root/'train_readout'/f) for k,f in
    [('checkpoint','selected.pt'),('report','training_report.json'),('predictions','validation_predictions.npz'),('receipt','task_receipt.json')]},
    accepted_job=rec['job_id'], startup=load_json(root/'train_readout/startup.json'),
    startup_ref=file_ref(root/'train_readout/startup.json'),
    protected_roots=[s['root'],s['project_dir']]+previous['protected_roots'])
print('FULLSTAT_MULTIVIEW_REUSE='+json.dumps(e,sort_keys=True,allow_nan=False))
'''


def authenticate(path,project):
    path=Path(path).resolve(strict=True); s=load_json(path); original=Path(s['project_dir'])
    if (s['contract']!='LUKA_FULLSTAT_READOUT_SPEC/v2' or s['source_commit']!=DONOR
            or s['counts']!=COUNTS or path!=Path(s['root'])/'campaign_spec.json'):
        raise ValueError('Expected completed corrected FullStat readout at db9afe8b')
    runtime(s['reuse']['environment'])
    names=subprocess.run(['git','-C',str(original),'ls-files','src','scripts','sbatch'],
        check=True,capture_output=True,text=True).stdout.splitlines()
    hashes={}; original_hashes={}
    # Only these old SPORC files changed since corrected Oscar readout.
    # New bytes are pinned independently by BASELINE and the current source snapshot.
    permitted={"src/hlt_classification/luka_u100_multiview/"+n+".py"
               for n in ("campaign","contracts","source","views")}
    for name in names:
        if name.endswith(('.py','.sh')):
            before,after=original/name,Path(project)/name
            if not after.is_file() or (name not in permitted and sha256_file(before)!=sha256_file(after)):
                raise ValueError('Changed retained donor runtime: '+name)
            hashes[name]=sha256_file(after)
            original_hashes[name]=sha256_file(before)
    if not hashes: raise ValueError('Empty donor runtime inventory')
    print('Authenticating completed Oscar readout and original COARSE_U100 bank in pinned source...',flush=True)
    env=dict(os.environ,PYTHONPATH=str(original/'src'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    run=subprocess.run([sys.executable,'-B','-s','-c',REPLAY,str(path)],cwd=original,env=env,capture_output=True,text=True)
    if run.returncode: raise ValueError('Readout source replay failed:\n'+run.stderr)
    records=[x.split('=',1)[1] for x in run.stdout.splitlines() if x.startswith('FULLSTAT_MULTIVIEW_REUSE=')]
    if len(records)!=1: raise ValueError('Missing exact completed-readout evidence')
    return artifact('REUSE',parent_spec=file_ref(path),source_commit=DONOR,counts=dict(COUNTS),
        compatible_source=hashes, original_source=original_hashes,**json.loads(records[0]))


def check_registered_reuse(reuse,project):
    """Workers recheck pinned source/metadata; the student loads bank payloads."""
    validate_path=Path(reuse['parent_spec']['path'])
    from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
    validate_file_ref(reuse['parent_spec']); validate_file_ref(reuse['startup_ref'])
    if load_json(reuse['startup_ref']['path'])!=reuse['startup']: raise ValueError('Parent startup changed')
    s=load_json(validate_path)
    if s['source_commit']!=DONOR or s['reuse']!=reuse['u100']: raise ValueError('Parent reuse changed')
    for name,digest in reuse['compatible_source'].items():
        if sha256_file(Path(project)/name)!=digest or sha256_file(Path(s['project_dir'])/name)!=reuse['original_source'][name]:
            raise ValueError('Pinned donor source changed: '+name)
    # Receipt and report bytes bind the historical acceptance without loading
    # teacher supervision into the oracle workers.
    for refs in (reuse['readout_reference'],reuse['u100']['reference']):
        for key in ('report','receipt'): validate_file_ref(refs[key])

    validate_file_ref(reuse['u100']['anchor']['spec'])
    p=load_json(reuse['u100']['anchor']['spec']['path'])
    if reuse['oracle_node']!=next(n for n in p['nodes'] if n['node_id']=='OFFLINE'):
        raise ValueError('Original OFFLINE node changed')

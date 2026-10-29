"""Authenticate completed fine-tail DE000 in its unchanged original source."""
from pathlib import Path
import json
import os
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json, sha256_file
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_fullstat_fine_tail.source import runtime, RUNTIME
from .contracts import artifact, COUNTS

DONOR = '04d1f0f5945ab9af38d83e192eff48e124ad8dc5'
BASELINE = 'a65caa78913e5e31a89ff7855b6f184605881d22'
MODULES = ('__init__', 'contracts', 'source', 'campaign', 'training', 'worker',
           'submission', 'execution', 'launch')
ADDITIONS = {f'src/hlt_classification/luka_fullstat_finetune/{n}.py' for n in MODULES}
ENTRYPOINTS = {'scripts/luka_fullstat_finetune.py', 'scripts/queue_luka_fullstat_finetune.sh',
              'scripts/launch_luka_fullstat_finetune.sh', 'sbatch/run_luka_fullstat_finetune.sh'}
REQUIRED = ADDITIONS | ENTRYPOINTS | {'tests/test_luka_fullstat_finetune.py',
    'tests/test_luka_fullstat_finetune_launch.py', 'docs/plans/LUKA_FULLSTAT_FINETUNE_PLAN.md',
    'docs/contracts/LUKA_FULLSTAT_FINETUNE.md', 'docs/LUKA_FULLSTAT_FINETUNE_RUNBOOK.md'}

REPLAY = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullstat_fine_tail import campaign,worker,banks,kd_banks,submission
s=load_json(sys.argv[1]); campaign.check(s)
done=worker.completed(s,'DE000')
if done is None: raise ValueError('Completed fine-tail DE000 required')
d=Path(s['root'])/'train_DE000'
reference={k:file_ref(d/f) for k,f in [('checkpoint','selected.pt'),('report','training_report.json'),
    ('predictions','validation_predictions.npz'),('receipt','task_receipt.json')]}
report=load_json(reference['report']['path'])
order=s['recipes']['DE000']['teacher_order']; refs=dict(s['references'],FINE_TAIL_DE000=reference)
e=dict(anchor=s['anchor'],references=refs,reference=reference,base_node=s['nodes']['DE000'],
    parent_recipe=s['recipes']['DE000'],parent_selected_pass=report['selected_pass'],
    parent_validation=report['validation'],parent_train_minutes=s['task_minutes']['DE000'],
    bundle=file_ref(Path(s['root'])/'bundles/DE000.json'),
    imported_banks={t:dict(embedding=banks.manifest_ref(s,t),kd=kd_banks.manifest_ref(s,t)) for t in order},
    teacher_refs={t:banks.refs_for(s,t) for t in order},
    execution_site=s['execution_site'],environment=s['reuse']['environment'],gpu=s['reuse']['gpu'],
    summaries=s['reuse']['summaries'],jobs=submission.accepted_jobs(s),
    protected_roots=s['reuse']['protected_roots']+[s['root'],s['project_dir'],s['runtime_prefix']])
print('FINETUNE_REUSE='+json.dumps(e,sort_keys=True,allow_nan=False))
'''


def compatible_changes(changed, added):
    return changed <= ADDITIONS | ENTRYPOINTS and added == changed


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project, commit)
    subprocess.run(['git', '-C', str(project), 'merge-base', '--is-ancestor', commit, 'origin/main'],
                   check=True, capture_output=True, text=True)
    def git(*args):
        return set(subprocess.run(['git','-C',str(project),*args],check=True,
                                 capture_output=True,text=True).stdout.splitlines())
    changed = git('diff','--name-only',BASELINE,commit,'--','src','scripts','sbatch')
    added = git('diff','--name-only','--diff-filter=A',BASELINE,commit,'--','src','scripts','sbatch')
    if (not REQUIRED <= git('ls-files') or not compatible_changes(changed,added)
            or Path(__file__).resolve() != project/'src/hlt_classification/luka_fullstat_finetune/source.py'):
        raise ValueError('Source differs outside the additive fine-tuning namespace')
    return snapshot


def authenticate(path, project):
    path = Path(path).resolve(strict=True); s = load_json(path)
    if (s.get('contract') != 'LUKA_FULLSTAT_FINE_TAIL_SPEC/v1' or s['counts'] != COUNTS
            or s['source_commit'] != DONOR or s.get('runtime_prefix') != RUNTIME
            or path != Path(s['root'])/'campaign_spec.json'):
        raise ValueError('Expected the exact completed original fine-tail campaign')
    runtime(s['reuse']['environment']); original = Path(s['project_dir'])
    names = subprocess.run(['git','-C',str(original),'ls-files','src','scripts','sbatch'],
        check=True,capture_output=True,text=True).stdout.splitlines()
    hashes = {}
    for name in names:
        if name.endswith(('.py','.sh')):
            if not (Path(project)/name).is_file() or sha256_file(original/name) != sha256_file(Path(project)/name):
                raise ValueError('Changed parent runtime: '+name)
            hashes[name] = sha256_file(Path(project)/name)
    if not hashes: raise ValueError('Empty parent runtime inventory')
    env = dict(os.environ,PYTHONPATH=str(original/'src'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    print('Authenticating completed fine-tail DE000 and all seven frozen banks; no new preflight.',flush=True)
    run = subprocess.run([sys.executable,'-B','-s','-c',REPLAY,str(path)],cwd=original,
                         env=env,capture_output=True,text=True)
    if run.returncode: raise ValueError('Original-source authentication failed:\n'+run.stderr)
    rows = [v.split('=',1)[1] for v in run.stdout.splitlines() if v.startswith('FINETUNE_REUSE=')]
    if len(rows) != 1: raise ValueError('Missing exact original-source evidence')
    return artifact('REUSE',parent_spec=file_ref(path),source_commit=s['source_commit'],
                    counts=dict(COUNTS),compatible_source=hashes,**json.loads(rows[0]))

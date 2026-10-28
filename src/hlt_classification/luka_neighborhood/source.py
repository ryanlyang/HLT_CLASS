"""Scoped original-source import: completed targets/baseline, not other arms."""
from pathlib import Path
import subprocess
from hlt_classification.data.cache_contracts import load_json,sha256_file
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_fullsim.preflight_reuse import _files
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.luka_attention_weight.source import replay
from hlt_classification.luka_supervision_depth import contracts as depth
from hlt_classification.luka_trial_hybrid import banks as compact
from .contracts import artifact,validate,BASE,COUNTS,TEACHERS,REFERENCES

MODULES=('__init__','contracts','targets','banks','model','training','startup','source','campaign','submission','worker','retention')
REQUIRED={'src/hlt_classification/luka_neighborhood/'+n+'.py' for n in MODULES}|{
    'scripts/luka_neighborhood.py','scripts/queue_luka_neighborhood.sh','sbatch/run_luka_neighborhood.sh',
    'tests/test_luka_neighborhood.py','docs/plans/LUKA_NEIGHBORHOOD_PLAN.md',
    'docs/contracts/LUKA_NEIGHBORHOOD.md','docs/LUKA_NEIGHBORHOOD_RUNBOOK.md',
    'docs/LUKA_OFFLINE_NEIGHBORHOOD_STRATEGY_HANDOFF.md'}


def current(project,commit):
    project=Path(project).resolve(); snap=_source(project,commit)
    def git(*args): return subprocess.run(['git','-C',str(project),*args],check=True,capture_output=True,text=True).stdout.splitlines()
    if not REQUIRED<=set(git('ls-files')) or Path(__file__).resolve()!=project/'src/hlt_classification/luka_neighborhood/source.py':
        raise ValueError('Commit the complete additive neighborhood implementation first')
    if git('diff','--name-only','--diff-filter=MDRT',BASE,commit,'--','src'):
        raise ValueError('Existing scientific runtime changed')
    return snap


REPLAY='''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_supervision_depth import campaign,worker,submission
s=load_json(sys.argv[1]); campaign.check(s)
seen={}; worker.validate_task(s,'baseline',seen)
for task,row in seen.items(): submission.successful(s,task,row['job_id'])
root=Path(s['root']); refs={n:file_ref(root/'targets/references'/(n+'.npz')) for n in worker.REFERENCES}
refs['BASELINE_REPEAT']=file_ref(root/'baseline/fit/validation_predictions.npz')
print('NEIGHBOR_PARENT='+json.dumps(dict(reuse=s['reuse'],anchor=s['anchor'],references=refs,
    subset=file_ref(root/'targets/references/original50k.npz'),
    baseline_report=file_ref(root/'baseline/fit/training_report.json'),
    receipts={t:file_ref(root/t/'receipt.json') for t in seen},jobs={t:r['job_id'] for t,r in seen.items()},
    banks={t:file_ref(root/'targets/banks'/t/'manifest.json') for t in worker.TEACHERS}),sort_keys=True,allow_nan=False))
'''


def import_parent(path,project):
    path=Path(path).resolve(strict=True); parent=load_json(path); depth.validate(parent,'SPEC')
    if path!=Path(parent['root'])/'campaign_spec.json' or parent['counts']!=COUNTS: raise ValueError('Original depth parent required')
    validate_source_snapshot(parent['source_snapshot'],repository=parent['project_dir'],require_clean=True)
    before=_files(parent['project_dir'])
    if not before or not before<=_files(project): raise ValueError('Missing inherited runtime')
    for name in before:
        if sha256_file(Path(parent['project_dir'])/name)!=sha256_file(Path(project)/name): raise ValueError('Parent runtime differs: '+name)
    print('Authenticating original completed targets and baseline; no other depth fit required...',flush=True)
    ref,e=replay(path,parent['project_dir'],REPLAY,'NEIGHBOR_PARENT=')
    value=artifact('IMPORT',parents=dict(spec=parent['content_hash']),spec=ref,**e)
    validate_import(value)
    return value


def validate_import(value,*,live=True):
    validate_file_ref(value['spec']); parent=load_json(value['spec']['path']); depth.validate(parent,'SPEC')
    validate(value,'IMPORT',dict(spec=parent['content_hash']))
    if (parent['counts']!=COUNTS or parent['recipes']['baseline']!=depth.recipe('baseline') or value['reuse']!=parent['reuse']
            or value['anchor']!=parent['anchor'] or set(value['references'])!=set(REFERENCES)
            or set(value['banks'])!=set(TEACHERS) or set(value['receipts'])!={'targets','baseline'}
            or set(value['jobs'])!={'targets','baseline'}): raise ValueError('Import scope differs')
    root=Path(parent['root']); expected={n:root/('baseline/fit/validation_predictions.npz' if n=='BASELINE_REPEAT'
        else 'targets/references/'+n+'.npz') for n in REFERENCES}
    expected.update(subset=root/'targets/references/original50k.npz',baseline_report=root/'baseline/fit/training_report.json')
    for key,path in expected.items():
        ref=value['references'][key] if key in REFERENCES else value[key]
        if Path(ref['path'])!=path: raise ValueError('Imported reference path differs')
        if live: validate_file_ref(ref)
    for t,ref in value['receipts'].items():
        if Path(ref['path'])!=root/t/'receipt.json': raise ValueError('Parent receipt path differs')
        if live:
            validate_file_ref(ref); row=load_json(ref['path']); depth.validate(row,'TASK')
            if row['job_id']!=value['jobs'][t]: raise ValueError('Imported accepted job differs')
    for t,ref in value['banks'].items():
        if Path(ref['path'])!=root/'targets/banks'/t/'manifest.json': raise ValueError('Borrowed bank location differs')
        if live:
            validate_file_ref(ref)
            compact.manifest(Path(ref['path']).parent,value['reuse']['teachers'][t],value['anchor']['evidence']['prepared'],COUNTS['train'])
    if live: validate_source_snapshot(parent['source_snapshot'],repository=parent['project_dir'],require_clean=True)
    return parent

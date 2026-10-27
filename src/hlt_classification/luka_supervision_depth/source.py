"""Read-only completed-grid replay; no old runtime patching or artifact relabeling."""
from pathlib import Path
import subprocess
from hlt_classification.data.cache_contracts import load_json,sha256_file
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_fullsim.preflight_reuse import _files
from hlt_classification.luka_attention_weight.source import replay
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.luka_auxiliary_grid import contracts as grid,source as grid_source
from .contracts import artifact,validate,COUNTS,REFERENCES

BASE='e21aa91463a8f313bf51e36d6bcac97ee7942bfb'
# Unrelated FullStat revisions after the original grid are explicitly pinned.
# Original-source replay uses its own old copies, not these replacements.
ISOLATED_FULLSTAT={'src/hlt_classification/luka_fullstat_splithead/'+n+'.py'
                  for n in ('campaign','kd_banks','submission','worker')}
MODULES=('__init__','contracts','model','training','banks','data','source','campaign','submission','worker','startup')
REQUIRED={'src/hlt_classification/luka_supervision_depth/'+n+'.py' for n in MODULES}|{
    'scripts/luka_supervision_depth.py','scripts/queue_luka_supervision_depth.sh','sbatch/run_luka_supervision_depth.sh',
    'tests/test_luka_supervision_depth.py','docs/plans/LUKA_SUPERVISION_DEPTH_PLAN.md',
    'docs/contracts/LUKA_SUPERVISION_DEPTH.md','docs/LUKA_SUPERVISION_DEPTH_RUNBOOK.md'}


def current(project,commit):
    project=Path(project).resolve(); snap=_source(project,commit)
    def git(*args):
        return subprocess.run(['git','-C',str(project),*args],check=True,capture_output=True,text=True).stdout.splitlines()
    if not REQUIRED<=set(git('ls-files')) or Path(__file__).resolve()!=project/'src/hlt_classification/luka_supervision_depth/source.py':
        raise ValueError('Commit complete additive depth/diversity implementation')
    changed=git('diff','--name-status','--no-renames','--diff-filter=MDRT',BASE,commit,'--','src')
    if changed: raise ValueError('Existing scientific runtime changed: '+repr(changed))
    return snap


REPLAY='''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_auxiliary_grid import campaign,worker,submission
s=load_json(sys.argv[1]); campaign.check(s)
seen={}; retired=worker.retired_evidence(s)
worker.validate_task(s,'k1_e40_plain',seen,retired=retired)
for task,row in seen.items(): submission.successful(s,task,row['job_id'])
root=Path(s['root']); references={n:file_ref(root/'targets/references'/(n+'.npz'))
    for n in ('M0HLT','OFFLINE','OFFLINE_EMBED_W5','U100_EMBED_W5','C50_DE000')}
references['K1_E40_PLAIN']=file_ref(root/'k1_e40_plain/fit/validation_predictions.npz')
print('DEPTH_PARENT='+json.dumps(dict(reuse=s['reuse'],anchor=s['anchor'],references=references,
    subset=file_ref(root/'targets/references/original50k.npz'),
    baseline_report=file_ref(root/'k1_e40_plain/fit/training_report.json'),
    baseline_checkpoint=file_ref(root/'k1_e40_plain/fit/selected.pt'),
    baseline_receipt=file_ref(root/'k1_e40_plain/receipt.json'),
    jobs={t:r['job_id'] for t,r in seen.items()}),sort_keys=True,allow_nan=False))
'''


def import_parent(path,project):
    path=Path(path).resolve(strict=True); parent=load_json(path); grid.validate(parent,'SPEC')
    if path!=Path(parent['root'])/'campaign_spec.json' or parent['counts']!=COUNTS or parent['recipes']['k1_e40_plain']!=grid.recipe('k1_e40_plain'):
        raise ValueError('Expected completed original grid k1_e40_plain')
    validate_source_snapshot(parent['source_snapshot'],repository=parent['project_dir'],require_clean=True)
    before=_files(parent['project_dir'])
    if not before or not before<=_files(project): raise ValueError('Missing parent runtime')
    for name in before:
        if name in ISOLATED_FULLSTAT:
            expected=subprocess.run(['git','-C',str(project),'show',BASE+':'+name],check=True,capture_output=True).stdout
            if (Path(project)/name).read_bytes()!=expected: raise ValueError('Registered isolated FullStat revision differs: '+name)
        elif sha256_file(Path(parent['project_dir'])/name)!=sha256_file(Path(project)/name):
            raise ValueError('Parent runtime differs: '+name)
    print('Authenticating completed k1_e40_plain and frozen c50 ancestry in original source...',flush=True)
    ref,evidence=replay(path,parent['project_dir'],REPLAY,'DEPTH_PARENT=')
    imported=artifact('IMPORT',parents=dict(spec=parent['content_hash']),spec=ref,**evidence)
    validate_import(imported)
    return imported


def validate_import(value,*,live=True):
    validate_file_ref(value['spec']); parent=load_json(value['spec']['path']); grid.validate(parent,'SPEC')
    validate(value,'IMPORT',dict(spec=parent['content_hash']))
    if (parent['counts']!=COUNTS or parent['recipes']['k1_e40_plain']!=grid.recipe('k1_e40_plain')
            or value['reuse']!=parent['reuse'] or value['anchor']!=parent['anchor'] or set(value['references'])!=set(REFERENCES)):
        raise ValueError('Parent import scope differs')
    root=Path(parent['root'])
    expected={n:root/('k1_e40_plain/fit/validation_predictions.npz' if n=='K1_E40_PLAIN' else 'targets/references/'+n+'.npz') for n in REFERENCES}
    expected.update(subset=root/'targets/references/original50k.npz',baseline_report=root/'k1_e40_plain/fit/training_report.json',
        baseline_checkpoint=root/'k1_e40_plain/fit/selected.pt',baseline_receipt=root/'k1_e40_plain/receipt.json')
    for key,path in expected.items():
        ref=value['references'][key] if key in REFERENCES else value[key]
        if Path(ref['path'])!=path: raise ValueError('Imported file is not the registered grid output')
    if live:
        validate_source_snapshot(parent['source_snapshot'],repository=parent['project_dir'],require_clean=True)
        grid_source.validate_reuse(value['reuse'])
        for ref in (*value['references'].values(),value['subset'],value['baseline_report'],value['baseline_checkpoint'],value['baseline_receipt']):
            validate_file_ref(ref)
        report=load_json(value['baseline_report']['path']); grid.validate(report,'TRAINING_REPORT')
        receipt=load_json(value['baseline_receipt']['path']); grid.validate(receipt,'TASK')
        if (report['node']!=parent['nodes']['k1_e40_plain'] or report['passes']!=30
                or receipt['job_id']!=value['jobs'].get('k1_e40_plain')): raise ValueError('Baseline import identity differs')
    return load_json(value['reuse']['spec']['path'])

"""Authenticate the finished subclass-on control in its own pinned source."""
from pathlib import Path
import json
import os
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.luka_fullstat_six_teacher import contracts as old_contracts
from hlt_classification.luka_fullstat_six_teacher.source import git
from .contracts import artifact, validate, COUNTS, TASK

BASELINE = '27a8341210ca686b74d1c538a3ed258bb7fe947d'
PARENT_COMMIT = '2fb09f38a3a1da85cdceff9df311963fdb3aa872'
PARENT_JOB = '8848995'
PACKAGE = 'src/hlt_classification/luka_fullstat_six_teacher_s0/'
ENTRYPOINTS = {'scripts/luka_fullstat_six_teacher_s0.py',
    'scripts/launch_luka_fullstat_six_teacher_s0.sh','sbatch/run_luka_fullstat_six_teacher_s0.sh'}
REQUIRED = ENTRYPOINTS | {PACKAGE+n+'.py' for n in
    ('__init__','contracts','source','campaign','model','sampling','training','banks','startup',
     'worker','submission','launch','execution','recovery','targets','kd_banks')}
REQUIRED |= {'tests/test_luka_fullstat_six_teacher_s0.py',
    'tests/test_luka_fullstat_six_teacher_s0_lifecycle.py',
    'tests/test_luka_fullstat_six_teacher_s0_source.py',
    'docs/plans/LUKA_FULLSTAT_SIX_TEACHER_S0_PLAN.md',
    'docs/contracts/LUKA_FULLSTAT_SIX_TEACHER_S0.md',
    'docs/LUKA_FULLSTAT_SIX_TEACHER_S0_RUNBOOK.md',
    'docs/LUKA_FULLSTAT_SIX_TEACHER_S0_VALIDATION.md'}


def current(project, commit):
    project=Path(project).resolve();snapshot=_source(project,commit)
    git(project,'merge-base','--is-ancestor',commit,'origin/main')
    git(project,'merge-base','--is-ancestor',BASELINE,commit)
    changed=git(project,'diff','--name-only',BASELINE,commit,'--','src','scripts','sbatch').splitlines()
    if (any(not(n.startswith(PACKAGE) or n in ENTRYPOINTS) for n in changed)
            or Path(__file__).resolve()!=project/(PACKAGE+'source.py')):
        raise ValueError('Only additive six-teacher S0 runtime changes are registered')
    if not REQUIRED <= set(git(project,'ls-files').splitlines()):
        raise ValueError('Required S0 runtime/tests/docs must be committed')
    return snapshot


def boundary(parent, path):
    old_contracts.validate(parent,'SPEC')
    if (parent['source_commit']!=PARENT_COMMIT or parent['counts']!=COUNTS
            or parent['recipe']!=old_contracts.recipe()
            or Path(path)!=Path(parent['root'])/'campaign_spec.json'):
        raise ValueError('Expected original 2fb09f38 six-teacher control')


def original_source(parent):
    # Foreign worktrees must use the generic snapshot checker, not the active-
    # import-location guard in foundation._source.
    original=Path(parent['project_dir'])
    if git(original,'rev-parse','HEAD')!=PARENT_COMMIT:
        raise ValueError('Control source commit differs')
    validate_source_snapshot(parent['source_snapshot'],repository=original,require_clean=True)
    return original


REPLAY = r'''
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);s=json.loads(p.read_text())
sys.path.insert(0,str(Path(s['project_dir'])/'src'))
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullstat_six_teacher import campaign,worker,submission,recovery
campaign.check(s);receipt=worker.completed(s)
f=Path(s['root'])/'train_six'
reference={k:file_ref(f/n) for k,n in [('checkpoint','selected.pt'),
 ('report','training_report.json'),('predictions','validation_predictions.npz'),('receipt','task_receipt.json')]}
print('SIX_S0_BASELINE='+json.dumps(dict(reference=reference,exports=recovery.export_records(s),
 accounting=submission.accounting(s),job_id=receipt['job_id'],report_hash=receipt['report_hash']),
 sort_keys=True,allow_nan=False))
'''


def authenticate(path):
    path=Path(path).resolve(strict=True);parent=load_json(path);boundary(parent,path)
    original=original_source(parent)
    env=dict(os.environ,PYTHONPATH='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    print('Authenticating completed six-teacher job8848995 and reusing its frozen targets; no teacher exports...',flush=True)
    result=subprocess.run([sys.executable,'-B','-s','-c',REPLAY,str(path)],
        cwd=original,env=env,capture_output=True,text=True)
    if result.returncode:raise ValueError('Original control authentication failed:\n'+result.stderr)
    rows=[line.split('=',1)[1] for line in result.stdout.splitlines() if line.startswith('SIX_S0_BASELINE=')]
    if len(rows)!=1:raise ValueError('Missing original control evidence')
    evidence=json.loads(rows[0])
    value=artifact('BASELINE',parents=dict(spec=parent['content_hash']),spec=file_ref(path),**evidence)
    check_evidence(value)
    return value


def check_evidence(value):
    parent=load_json(validate_file_ref(value['spec']));boundary(parent,Path(value['spec']['path']))
    validate(value,'BASELINE',dict(spec=parent['content_hash']))
    state=value['accounting']
    if (value['job_id']!=PARENT_JOB or len(state)!=6 or state[:2]!=[PARENT_JOB,'lukafs6_train_six']
            or state[3]!=parent['execution_site']['account'] or state[4:]!=['COMPLETED','0:0']):
        raise ValueError('Expected successful completed six-teacher job8848995')
    expected={'checkpoint':'selected.pt','report':'training_report.json',
              'predictions':'validation_predictions.npz','receipt':'task_receipt.json'}
    if value['reference']!={k:file_ref(Path(parent['root'])/TASK/n) for k,n in expected.items()}:
        raise ValueError('Control reference files differ')
    receipt=load_json(value['reference']['receipt']['path']);old_contracts.validate(receipt,'TASK',dict(spec=parent['content_hash']))
    report=load_json(value['reference']['report']['path']);old_contracts.validate(report,'TRAINING_REPORT')
    from hlt_classification.luka_fullstat_six_teacher import worker,training,recovery
    training.validate_report(report,parent,worker.parents(parent))
    if (receipt['job_id']!=PARENT_JOB or receipt['source_commit']!=PARENT_COMMIT or receipt['task_id']!=TASK
            or receipt['report_hash']!=report['content_hash'] or value['report_hash']!=report['content_hash']
            or value['exports']!=recovery.export_records(parent) or set(value['exports'])!={'CONCAT','FUSION'}):
        raise ValueError('Control report, task or frozen exports changed')
    outputs={r['path']:r for r in receipt['outputs']}
    # The task receipt binds its published files, not a recursive hash of itself.
    for name in (n for n in expected.values() if n!='task_receipt.json'):
        if outputs.get(TASK+'/'+name)!=file_ref(Path(parent['root'])/TASK/name,root=parent['root']):
            raise ValueError('Control publication binding differs')
    return parent


def check(value):
    parent=check_evidence(value);original_source(parent)
    from hlt_classification.luka_fullstat_six_teacher import source as old_source
    old_source.check(parent['reuse'])
    return parent

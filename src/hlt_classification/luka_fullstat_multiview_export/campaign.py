"""Authenticate the finished fit without declaring the failed job successful."""
from pathlib import Path
import getpass
import math
import os
import subprocess
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import capture_source_snapshot
from hlt_classification.luka_fullstat_multiview import campaign as old_campaign, worker as old_worker
from hlt_classification.luka_fullstat_multiview import recovery as old_recovery, resources
from hlt_classification.luka_fullstat_multiteacher.campaign import protected
from .contracts import artifact, validate, PARENT_COMMIT, BASELINE, JOBS, TASKS, MINUTES, CLAIM

PACKAGE = 'src/hlt_classification/luka_fullstat_multiview_export/'
ENTRYPOINTS = {'scripts/luka_fullstat_multiview_export.py',
              'scripts/launch_luka_fullstat_multiview_export.sh',
              'sbatch/run_luka_fullstat_multiview_export.sh'}
REQUIRED = ENTRYPOINTS | {PACKAGE+n+'.py' for n in
    ('__init__','bootstrap','contracts','campaign','submission','worker','launch','execution')}
REQUIRED |= {'tests/test_luka_fullstat_multiview_export.py',
             'docs/LUKA_FULLSTAT_MULTIVIEW_EXPORT_RUNBOOK.md',
             'docs/contracts/LUKA_FULLSTAT_MULTIVIEW_EXPORT.md'}


def current(project, commit):
    project = Path(project).resolve()
    snapshot = capture_source_snapshot(project, require_clean=True)
    def git(*args):
        return subprocess.run(['git','-C',str(project),*args],check=True,capture_output=True,text=True).stdout
    if (snapshot['git_commit'] != commit or Path(__file__).resolve() != project/(PACKAGE+'campaign.py')):
        raise ValueError('Wrong adapter source')
    git('merge-base','--is-ancestor',commit,'origin/main')
    git('merge-base','--is-ancestor',BASELINE,commit)
    changed = git('diff','--name-only',BASELINE,commit,'--','src','scripts','sbatch').splitlines()
    if (any(not (n.startswith(PACKAGE) or n in ENTRYPOINTS) for n in changed)
            or not REQUIRED <= set(git('ls-files').splitlines())):
        raise ValueError('Unexpected runtime changes or missing recovery files')
    return snapshot


def parent(spec):
    return load_json(validate_file_ref(spec['reuse']['parent_spec']))


def accounting(old):
    from hlt_classification.luka_fullstat_multiview import submission
    if submission.accepted_jobs(old) != JOBS:
        raise ValueError('Expected exact 8220911/8220912 job binding')
    out = subprocess.run(['sacct','-S','2026-11-01','-j',','.join(JOBS.values()),'-X','-n','-P',
        '-o','JobIDRaw,JobName%80,User,Account,State%40,ExitCode'],
        check=True,capture_output=True,text=True).stdout
    rows = [line.strip().split('|')[:6] for line in out.splitlines() if line.strip()]
    expected = [(JOBS['teacher_FUSION'],'teacher_FUSION','FAILED','0:7'),
                (JOBS['train_multi3'],'train_multi3','CANCELLED','0:0')]
    if len(rows) != 2 or len({r[0] for r in rows}) != 2:
        raise ValueError('Missing or ambiguous failed-job accounting')
    for job, task, state, code in expected:
        row = next((r for r in rows if r[0] == job), None)
        if (row is None or row[1:4] != ['lukafsmv_'+task,getpass.getuser(),old['execution_site']['account']]
                or row[4].split(' by ')[0] != state or row[5] != code):
            raise ValueError('Wrong original owner/account/terminal state')
    return sorted(rows)


def inspect_parent(path, *, full=False):
    path = Path(path).resolve(strict=True)
    old = load_json(path)
    if (old['contract'] != 'LUKA_FULLSTAT_MULTIVIEW_RECOVERY_SPEC/v1'
            or old['source_commit'] != PARENT_COMMIT or path != Path(old['root'])/'campaign_spec.json'):
        raise ValueError('Expected the original a1eb1a74 recovery campaign')
    # This module is deliberately imported from that original source checkout.
    if Path(old_campaign.__file__).resolve() != Path(old['project_dir'])/'src/hlt_classification/luka_fullstat_multiview/campaign.py':
        raise ValueError('Original-source bootstrap required')
    old_campaign.check(old, full_reuse=full)
    root = Path(old['root']); folder = root/'teacher_FUSION'
    if os.path.lexists(folder/'task_receipt.json') or os.path.lexists(root/'train_multi3'):
        raise ValueError('Teacher already complete or student already attempted; inspect instead')
    # Bank partials, if any, stay untouched; the recovery exports to a fresh root.
    if os.path.lexists(folder/'bank/manifest.json'):
        raise ValueError('Unexpected published bank requires separate inspection')
    report, _ = old_worker.validate_fit(old,'teacher_FUSION')
    old_worker.completed(old,'teacher_CONCAT')
    if report['passes'] != 20:
        raise ValueError('Only a completely finished twenty-epoch fit can be recovered')
    inventory = {n:file_ref(root/n) for n in old_worker.output_names('teacher_FUSION')
                 if n != 'teacher_FUSION/task_receipt.json' and '/bank/' not in n}
    proof = load_json(folder/'startup.json')
    estimate = math.ceil(proof['timing']['preparation']/60+120+
        1.5*math.ceil(old['counts']['train']/128)*proof['timing']['export_step']/60)
    if estimate > MINUTES['export_FUSION']:
        raise ValueError('Saved export timing exceeds the six-hour recovery allocation')
    return old, artifact('REUSE',parents=dict(spec=old['content_hash']),
        parent_spec=file_ref(path),jobs=dict(JOBS),accounting=accounting(old),outputs=inventory,
        u100=old['reuse']['u100'],
        concat=old['recovery']['receipt'],selected_pass=report['selected_pass'],
        report_hash=report['content_hash'],export_estimate_minutes=estimate,
        no_new_teacher_fit=True)


def registration(old):
    keys = ('counts','nodes','recipe','anchor','references','bank_manifest','teacher_refs',
            'teacher_replay','runtime_prefix','execution_site','cpus','memory_mb')
    return dict({k:old[k] for k in keys},minutes=dict(MINUTES),tasks=dict(TASKS),
        runtime_policy=resources.policy(),storage_budget_bytes=20*2**30,filesystem_headroom_bytes=2*2**30,
        scientific_fits=1,export_jobs=1,preflight_jobs=0,automatic_retry=False,
        export_policy=dict(batch_size=128,precision='FP32_no_autocast',timing_factor=1.5,
            fixed_margin_minutes=120,train_only=True,original_selected_checkpoint=True))


def locations(root, old, project):
    protected(root,dict(anchor=old['anchor'],protected_roots=[old['root'],old['project_dir'],
        old_recovery.parent(old)['root'],old_recovery.parent(old)['project_dir'],
        *old['reuse']['protected_roots']]),project)
    root = Path(root)
    if root.resolve() != root or root.is_symlink() or not root.name.startswith('luka_fullstat_multiview_export_'):
        raise ValueError('Fresh dedicated unredirected export-recovery root required')


def validate_spec(spec):
    old = parent(spec)
    validate(spec['reuse'],'REUSE',dict(spec=old['content_hash']))
    validate(spec,'SPEC',dict(source=spec['source_snapshot']['content_hash'],reuse=spec['reuse']['content_hash']))
    if (old['source_commit'] != PARENT_COMMIT or spec['reuse']['jobs'] != JOBS
            or spec['reuse']['u100'] != old['reuse']['u100']
            or any(spec.get(k) != v for k,v in registration(old).items())
            or type(spec['available_quota_gib']) is not int or spec['available_quota_gib'] < 22):
        raise ValueError('Export recovery definition differs')
    locations(spec['root'],old,spec['project_dir'])
    return old


def check(spec, *, full=False):
    old = validate_spec(spec)
    if (load_json(Path(spec['root'])/'campaign_spec.json') != spec
            or current(spec['project_dir'],spec['source_commit']) != spec['source_snapshot']):
        raise ValueError('Pinned recovery source/spec changed')
    if inspect_parent(spec['reuse']['parent_spec']['path'],full=full)[1] != spec['reuse']:
        raise ValueError('Saved teacher evidence changed')
    claim = Path(old['root'])/CLAIM
    if claim.exists():
        from .submission import render
        value = load_json(claim/'intent.json'); validate(value,'CLAIM',dict(spec=spec['content_hash']))
        if value['root'] != spec['root'] or value['plan_hash'] != render(spec)['content_hash']:
            raise ValueError('Another recovery owns the failed campaign')
    old_campaign.storage(spec)
    return old


def create(root, path, project, commit, *, quota_gib):
    root, project = Path(root).absolute(), Path(project).resolve()
    snapshot = current(project,commit)
    print('Authenticating completed FUSION fit and original CONCAT in pinned source; no teacher inference.',flush=True)
    old, reuse = inspect_parent(path,full=True)
    if os.path.lexists(Path(old['root'])/CLAIM):
        raise FileExistsError('Original already claimed; preserve journals and inspect')
    spec = artifact('SPEC',parents=dict(source=snapshot['content_hash'],reuse=reuse['content_hash']),
        root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=snapshot,
        reuse=reuse,available_quota_gib=quota_gib,**registration(old))
    validate_spec(spec); old_campaign.storage(spec)
    root.mkdir(exist_ok=False); write_immutable_json(root/'campaign_spec.json',spec)
    return spec


def claim(spec, plan):
    old = parent(spec); accounting(old)
    folder = Path(old['root'])/CLAIM
    folder.mkdir(exist_ok=False)
    write_immutable_json(folder/'intent.json',artifact('CLAIM',parents=dict(spec=spec['content_hash']),
        root=spec['root'],plan_hash=plan['content_hash']))

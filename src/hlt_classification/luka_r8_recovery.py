"""Exact recovery of two pre-claim Slurm failures using the original R8 runtime.

Loaded by the new CLI from its pinned path, with hlt_classification imports
resolved exclusively from the original, clean scientific worktree.
"""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np
from hlt_classification.data.cache_contracts import (
    load_json, load_npz_arrays, write_immutable_json, with_content_hash, validate_content_hash,
)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import capture_source_snapshot
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_fullsim.regular_submission import clean_environment
from hlt_classification.luka_fullsim.preflight_v2 import require_backend
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_cross_message_hydra.execution import allocation
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag, load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger
from hlt_classification.luka_next_supervision_hydra import (
    campaign, worker, submission, training, inputs, reporting, contracts, reconstruction_release,
)

DONOR = '58d0c0edf5f41cff2398fb168fc16fc92e169294'
FAILED = {'C3': '168726', 'R14': '168727'}
BOOTSTRAP = {'setup': '168256', 'release': '168257'}
AUTHORIZATION = 'AUTHORIZE_R8_PRECLAIM_RECOVERY_C3_R14_V1'
REQUIRED = (
    'src/hlt_classification/luka_r8_recovery.py', 'scripts/luka_r8_recovery.py',
    'sbatch/run_luka_r8_recovery.sh', 'scripts/queue_luka_r8_recovery.sh',
    'tests/test_luka_r8_recovery.py', 'docs/contracts/LUKA_R8_RECOVERY.md',
    'docs/LUKA_R8_RECOVERY_RUNBOOK.md', 'docs/LUKA_R8_RECOVERY_VALIDATION.md',
)
POLICY = dict(attempts=3, delays_seconds=[5, 10], operations=['scontrol show job', 'scontrol show config'],
    scientific_changes=False, automatic_fit_retry=False, output_budget_bytes=2*2**30)


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_R8_RECOVERY_{kind}/v1', schema_version=1,
        parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind):
    validate_content_hash(value, expected_contract=f'LUKA_R8_RECOVERY_{kind}/v1')
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise ValueError('Recovery scientific scope differs')


def source(project, commit):
    project = Path(project).resolve()
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Full pushed recovery commit required')
    if Path(__file__).resolve() != project / REQUIRED[0]:
        raise ValueError('Recovery driver imported outside its pinned worktree')
    snapshot = capture_source_snapshot(project, require_clean=True)
    if snapshot['git_commit'] != commit:
        raise ValueError('Recovery source commit differs')
    tracked = subprocess.run(['git', '-C', str(project), 'ls-files'], check=True,
        capture_output=True, text=True).stdout.splitlines()
    if not set(REQUIRED) <= set(tracked):
        raise ValueError('Commit the complete recovery implementation')
    subprocess.run(['git', '-C', str(project), 'merge-base', '--is-ancestor', commit, 'origin/main'],
        check=True, capture_output=True)
    return snapshot


def original(path):
    old = load_json(path)
    if (old.get('source_commit') != DONOR or old.get('campaign_kind') != 'R8_FOLLOWUP_V2'
            or old.get('execution_version') != 'R5'):
        raise ValueError('This recovery is only for the completed 58d0c0ed R8 setup')
    if Path(campaign.__file__).resolve() != Path(old['project_dir']) / 'src/hlt_classification/luka_next_supervision_hydra/campaign.py':
        raise ValueError('Scientific runtime must come from the original pinned worktree')
    campaign.check(old, check_storage=False)
    return old


def no_original_attempt(old):
    root = Path(old['root'])
    for task in FAILED:
        for p in (root / 'claims' / task, root / task):
            if p.exists() or p.is_symlink():
                raise FileExistsError(f'Original attempt has a claim or output: {p}')


def qualified(old, *, require_failed=True):
    no_original_attempt(old)
    if submission.accepted_jobs(old, 'bootstrap') != BOOTSTRAP:
        raise ValueError('Original bootstrap jobs differ')
    jobs = submission.accepted_jobs(old, 'science')
    if any(jobs.get(t) != j for t, j in FAILED.items()) or jobs.get('report') != '168742':
        raise ValueError('Original science job bindings differ')
    if require_failed:
        for job in BOOTSTRAP.values():
            if submission.accounting(job) != ['COMPLETED', '0:0']:
                raise ValueError('Successful exact setup and release required')
        for task, job in FAILED.items():
            if submission.accounting(job) != ['FAILED', '1:0']:
                raise ValueError(f'{task} must still be the exact FAILED/1:0 original attempt')
    return worker.completed_setup(old)


def storage(spec):
    root = Path(spec['root'])
    used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    remaining = max(0, POLICY['output_budget_bytes'] - used)
    if used > POLICY['output_budget_bytes'] or shutil.disk_usage(root if root.exists() else root.parent).free < remaining + 2**30:
        raise OSError('Recovery needs its remaining two-GiB allowance plus one GiB headroom')


def create(root, original_spec, project, commit, quota_gib):
    old = original(original_spec)
    acceptance = qualified(old)
    root = Path(root).absolute(); project = Path(project).resolve()
    protected = [Path(old[k]).resolve() for k in ('root', 'project_dir', 'environment_dir', 'bundle_root', 'raw_root', 'delta_root')] + [project]
    if (root.exists() or not root.parent.is_dir() or root.resolve() != root
            or any(p.is_symlink() for p in (root, *root.parents))
            or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected)):
        raise ValueError('Use a fresh, separate recovery root')
    if type(quota_gib) is not int or quota_gib < 3:
        raise ValueError('Reserve at least three personally available GiB for recovery')
    snapshot = source(project, commit)
    logs = {}
    for task, job in FAILED.items():
        p = Path(old['root']) / f'{task}-{job}.out'
        text = p.read_text(errors='replace')
        if (f"['scontrol', 'show', 'job', '-o', '{job}']" not in text
                or 'subprocess.CalledProcessError' not in text or 'NEXT_SUPERVISION' in text):
            raise ValueError('Expected the recorded pre-training controller-query failure')
        logs[task] = file_ref(p)
    spec = artifact('SPEC', parents=dict(original=old['content_hash'], acceptance=acceptance['content_hash'], source=snapshot['content_hash']),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=snapshot,
        original_spec=file_ref(Path(original_spec).resolve()), original_logs=logs,
        original_ledger=file_ref(Path(old['root'])/'submission_science/submission_ledger.json'),
        original_setup_receipt=file_ref(Path(old['root'])/'setup/receipt.json'),
        environment_dir=old['environment_dir'], superseded_jobs=FAILED, tasks=list(FAILED),
        minutes={t:acceptance['minutes'][t] + 15 for t in FAILED},
        policy=POLICY, quota_gib=quota_gib, authorization=AUTHORIZATION)
    if max(spec['minutes'].values()) > old['fit_max_minutes']:
        raise ValueError('Recovery request exceeds the original reviewed runtime ceiling')
    storage(spec); root.mkdir()
    write_immutable_json(root/'recovery_spec.json', spec)
    return spec


def check(spec, *, check_storage=True):
    validate(spec, 'SPEC')
    for ref in (spec['original_spec'], spec['original_ledger'], spec['original_setup_receipt'], *spec['original_logs'].values()):
        validate_file_ref(ref)
    old = original(spec['original_spec']['path'])
    acceptance = qualified(old, require_failed=False)
    snapshot = source(spec['project_dir'], spec['source_commit'])
    if (spec['parents'] != dict(original=old['content_hash'], acceptance=acceptance['content_hash'], source=snapshot['content_hash'])
            or spec['source_snapshot'] != snapshot or spec['policy'] != POLICY
            or spec['tasks'] != list(FAILED) or spec['superseded_jobs'] != FAILED
            or spec['environment_dir'] != old['environment_dir'] or spec['authorization'] != AUTHORIZATION
            or spec['minutes'] != {t:acceptance['minutes'][t] + 15 for t in FAILED}
            or load_json(Path(spec['root'])/'recovery_spec.json') != spec):
        raise ValueError('Immutable recovery registration differs')
    if check_storage: storage(spec)
    return old


def retry_controller(call, *args):
    """Retry failed read-only RPCs, never allocation mismatches or training."""
    for attempt in range(POLICY['attempts']):
        try:
            return call(*args)
        except subprocess.CalledProcessError as exc:
            cmd = exc.cmd
            if (not isinstance(cmd, (list, tuple)) or len(cmd) < 3 or list(cmd[:2]) != ['scontrol', 'show']
                    or cmd[2] not in ('job', 'config')):
                raise
            print(f'Controller query failed ({attempt+1}/{POLICY["attempts"]}): {cmd!r}\n'
                  f'stdout: {exc.stdout!r}\nstderr: {exc.stderr!r}', flush=True)
            if attempt + 1 == POLICY['attempts']: raise
            time.sleep(POLICY['delays_seconds'][attempt])


def plan(spec, old):
    rows = []
    for task in (*FAILED, 'report'):
        row = (reconstruction_release.report_command(old) if task == 'report'
               else submission.command(old, task, spec['minutes'][task]))
        replacements = {'--job-name=':f'--job-name=lukanfix_{task}',
            '--comment=':f'--comment=lukanfix:{spec["content_hash"]}:{task}',
            '--chdir=':'--chdir='+spec['project_dir'],
            '--output=':'--output='+str(Path(spec['root'])/f'{task}-%j.out')}
        cmd = [next((v for k,v in replacements.items() if x.startswith(k)),x)
               for x in row['command'][:-4] if not x.startswith('--dependency=')]
        dependencies = list(FAILED) if task == 'report' else []
        if dependencies: cmd.append('--dependency=afterany:'+':'.join('${JOB_'+t+'}' for t in dependencies))
        cmd += [spec['project_dir']+'/sbatch/run_luka_r8_recovery.sh', spec['project_dir'],
            str(Path(spec['root'])/'recovery_spec.json'), task]
        rows.append(dict(task_id=task, dependencies=dependencies, command=cmd))
    return artifact('PLAN', parents=dict(recovery=spec['content_hash']), commands=rows,
        superseded_jobs=FAILED, authorization=AUTHORIZATION,
        report_scope='combined_snapshot_after_recovery_fits_other_original_fits_may_be_incomplete')


def jobs(spec, old):
    folder=Path(spec['root'])/'submission'; value=plan(spec,old)
    saved=folder/'command_plan.json'
    if not saved.exists(): return {}
    if load_json(saved) != value: raise ValueError('Recovery command plan differs')
    events, accepted=load_exact_dag_journal(folder/'submission_ledger_journal', identity=value['content_hash'], plan=value)
    if (folder/'submission_ledger.json').exists():
        if load_json(folder/'submission_ledger.json') != assemble_submission_ledger(events,campaign_spec_sha256=value['content_hash']):
            raise ValueError('Recovery ledger/journal differs')
    return accepted


def submit(spec, *, execute=False, plan_hash=None, authorization=None):
    old=check(spec); qualified(old)
    value=plan(spec,old); folder=Path(spec['root'])/'submission'; folder.mkdir(exist_ok=True)
    saved=folder/'command_plan.json'; dry=folder/'dry_run_submission_ledger.json'; live=folder/'submission_ledger.json'
    if execute:
        if (authorization != AUTHORIZATION or plan_hash != value['content_hash']
                or not saved.exists() or load_json(saved) != value):
            raise PermissionError('Authorize the exact dry-reviewed recovery plan')
        if live.exists():
            if set(jobs(spec,old)) != {*FAILED,'report'}: raise ValueError('Incomplete recovery ledger')
            return load_json(live)
        if (folder/'live_claim').exists() or (folder/'submission_ledger_journal').exists():
            raise RuntimeError('Ambiguous partial recovery submission; inspect exact IDs, do not retry blindly')
    write_immutable_json(saved,value)
    if execute:
        (folder/'live_claim').mkdir()
        write_immutable_json(folder/'live_claim/intent.json',artifact('SUBMIT_CLAIM',parents=dict(plan=value['content_hash']),authorization=AUTHORIZATION))
    else:
        reconstruction_release.scheduler_review(value)
    return submit_exact_dag(identity=value['content_hash'],plan=value,output=live if execute else dry,
        canonical_dry_run=dry,execute=execute,environment=clean_environment())


def exact_job(spec, old, task, job):
    for attempt in range(12):
        if jobs(spec,old).get(task)==job: return
        time.sleep(5)
    raise PermissionError('Worker is not the exact accepted recovery job')


def fit(spec, old, task, *, device='cuda'):
    """The donor fit receives the unmodified original scientific spec/targets."""
    root=Path(spec['root'])/task; root.mkdir(exist_ok=False)
    train=inputs.hlt(old,'train'); val=inputs.hlt(old,'validation'); bank=inputs.embedding_bank(old,train)
    candidates=inputs.candidates(old,task,train,device)
    try:
        return training.fit(root/'fit',old,task,train,val,bank,candidates=candidates,device=device)
    finally:
        if hasattr(candidates,'close'): candidates.close()


def run(spec, task):
    if task not in (*FAILED,'report'): raise ValueError('Unregistered recovery task')
    old=check(spec)
    if installed_environment()!=old['reference']['environment']: raise PermissionError('Qualified software differs')
    if task=='report':
        job=retry_controller(reconstruction_release.cpu_allocation,old)
    else:
        parent=campaign.check(old,check_storage=False); require_backend()
        job=retry_controller(allocation,parent)
        if gpu_identity()!=old['qualification']['acceptance']['gpu']: raise PermissionError('Qualified GPU differs')
    exact_job(spec,old,task,job)
    claim=Path(spec['root'])/'claims'/task; claim.mkdir(parents=True,exist_ok=False)
    write_immutable_json(claim/'intent.json',artifact('RUN_CLAIM',parents=dict(recovery=spec['content_hash']),task=task,job_id=job))
    if task=='report':
        result=results(spec)
        write_immutable_json(Path(spec['root'])/'report.json',result)
        return result
    report=fit(spec,old,task)
    check(spec)
    root=Path(spec['root'])/task
    receipt=artifact('TASK',parents=dict(recovery=spec['content_hash'],original=old['content_hash'],report=report['content_hash']),
        task=task,job_id=job,supersedes_job=FAILED[task],outputs=[file_ref(p,root=root) for p in sorted(root.rglob('*')) if p.is_file()])
    write_immutable_json(root/'receipt.json',receipt)
    return receipt


def recovered(spec,old,task):
    folder=Path(spec['root'])/task; receipt=load_json(folder/'receipt.json'); validate(receipt,'TASK')
    report, arrays=training.validate_fit(folder/'fit',old,task)
    if (receipt['parents']!=dict(recovery=spec['content_hash'],original=old['content_hash'],report=report['content_hash'])
            or receipt['task']!=task or receipt['job_id']!=jobs(spec,old).get(task)
            or receipt['supersedes_job']!=FAILED[task]
            or receipt['outputs']!=[file_ref(p,root=folder) for p in sorted(folder.rglob('*')) if p.is_file() and p!=folder/'receipt.json']):
        raise ValueError('Recovered result ownership or output inventory differs')
    return report,arrays,receipt


def results(spec):
    old=check(spec,check_storage=False)
    with redirect_stdout(StringIO()): base=reporting.results(old)
    rows=list(base['rows']); incomplete=list(base['incomplete']); hashes={}
    bundle=inputs.Bundle(old['bundle_root'])
    reference=load_npz_arrays(bundle.path('reference/M0HLT/predictions.npz'))
    subset=load_npz_arrays(bundle.path('reference/historical/subset.npz'))['indices']
    for task in FAILED:
        if not (Path(spec['root'])/task/'receipt.json').exists(): continue
        report,arrays,receipt=recovered(spec,old,task); hashes[task]=receipt['content_hash']
        if any(not np.array_equal(reference[k],arrays[k]) for k in ('identities','labels')):
            raise ValueError('Recovered validation identities/labels differ')
        if task in incomplete: incomplete.remove(task)
        for role,ix in (('validation',slice(None)),('original50k',subset)):
            rows.append(dict(model=task,role=role,metrics=evaluate_probabilities(arrays['labels'][ix],arrays['probabilities'][ix]),
                epoch=report['selected_pass'],reused=False,location='Hydra recovery',recovered_job=receipt['job_id'],supersedes_job=FAILED[task]))
    print('Combined original R8 campaign and exact C3/R14 recovery; exploratory validation, no independent test.')
    for role in ('validation','original50k'):
        selected=[r for r in rows if r['role']==role]; by={r['model']:r for r in selected}
        if len(by)!=len(selected): raise ValueError('Duplicate original/recovered row')
        lo=by['M0HLT']['metrics']['macro_ovr_auc']; hi=by['OFFLINE']['metrics']['macro_ovr_auc']
        print('\n'+role+(' (overlaps validation)' if role=='original50k' else ''))
        print(f'{"model":16} {"ep":>3} {"AUC":>12} {"recovery%":>10} {"delta matched":>15}')
        for r in selected:
            task=r['model']; auc=r['metrics']['macro_ovr_auc']; ctrl='C'+str(contracts.setting(task)['seed_index']) if task in old['tasks'] else None
            r['recovery']=reporting.recovery(auc,lo,hi)
            r['delta_matched']=None if ctrl not in by else auc-by[ctrl]['metrics']['macro_ovr_auc']
            label=task+(' [recovered]' if 'recovered_job' in r else '')
            print(f'{label:16} {str(r.get("epoch") or "-"):>3} {auc:12.8f} {r["recovery"]:10.1f} '+('--' if r['delta_matched'] is None else f'{r["delta_matched"]:+.8f}').rjust(15))
    print('Pending/incomplete:',', '.join(incomplete) or 'none')
    return artifact('RESULTS',parents=dict(recovery=spec['content_hash'],original_results=base['content_hash'],**hashes),
        rows=rows,incomplete=incomplete,original_states=base['states'],recovery_jobs=jobs(spec,old))


def progress(spec):
    old=check(spec,check_storage=False)
    anchors=inputs.Bundle(old['bundle_root']).json('origin/campaign_spec.json')['recovery_anchors']
    rows={}
    for task in FAILED:
        rows[task]=[]
        for path in sorted((Path(spec['root'])/task/'fit').glob('epoch_*.json')):
            row=load_json(path); contracts.validate(row,'EPOCH',dict(spec=old['content_hash']))
            if row['task']!=task: raise ValueError('Epoch task differs')
            rows[task].append(row)
    control=max(rows['C3'],key=lambda r:r['validation']['macro_ovr_auc'],default=None)
    for task,history in rows.items():
        print('\n'+task+' recovery')
        for row in history:
            auc=row['validation']['macro_ovr_auc']; delta='--' if control is None else f'{auc-control["validation"]["macro_ovr_auc"]:+.8f}'
            print(f'epoch {row["pass_number"]:2d} AUC={auc:.8f} recovery={reporting.recovery(auc,anchors["M0HLT"],anchors["OFFLINE"]):.1f}% delta_selected_C3={delta}')
        if not history: print('No published validation epoch')
    return dict(recovery_jobs=jobs(spec,old),epochs={t:len(v) for t,v in rows.items()})

"""Same-resource debug execution from the untouched accepted tier3 sweep."""
from pathlib import Path
import subprocess
import torch
from hlt_classification.data.cache_contracts import (
    load_json, write_immutable_json, with_content_hash, validate_content_hash)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_fullsim.regular_submission import check_site, clean_environment
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger
from hlt_classification.luka_embedding_ladder.source import replay
from . import contracts, campaign
from .contracts import WEIGHTS, arm_gate
from .worker import (execution, storage, PhaseLog, memory_snapshot, regular_cache,
                     data, load_bank, fit, validate_fit, comparison)

BASE_COMMIT = 'fd687181a60c1557f98d7498f577e26d6dbb35a7'
MEASURED_COMMIT = '007885f7360f4cd28e9f91a690e04f0f4ea0afff'
MODULE = 'src/hlt_classification/luka_embedding_sweep/debug.py'
ADDITIONS = {MODULE, 'src/hlt_classification/luka_embedding_sweep/debug_retirement.py'}
REQUIRED = ADDITIONS | {
    'scripts/luka_embedding_sweep_debug.py', 'scripts/queue_luka_embedding_sweep_debug.sh',
    'sbatch/run_luka_embedding_sweep_debug.sh', 'tests/test_luka_embedding_sweep_debug.py',
    'docs/plans/LUKA_EMBEDDING_SWEEP_DEBUG_PLAN.md',
    'docs/contracts/LUKA_EMBEDDING_SWEEP_DEBUG.md', 'docs/LUKA_EMBEDDING_SWEEP_DEBUG_RUNBOOK.md'}
POLICY = 'sporc_tier3_to_debug_same_a100_environment_resources_v1'
AUTHORIZATION = 'AUTHORIZE_LUKA_EMBEDDING_SWEEP_DEBUG_V1'
REPLAY = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_embedding_sweep import campaign,submission,followup,contracts
s=load_json(sys.argv[1]); g,r=campaign.check_spec(s)
p=submission.render(s); folder=Path(s['root'])/'submission'
ledger=load_json(folder/'submission_ledger.json')
jobs=followup.science_jobs(s,p,ledger)
refs=[file_ref(folder/'submission_ledger.json'),file_ref(folder/'command_plan.json')]
if load_json(folder/'command_plan.json') != p: raise ValueError('Original exact plan differs')
f=Path(g['root'])/'followup'
if f.exists():
    done=load_json(f/'complete.json'); auth=load_json(f/'authorization.json')
    contracts.validate(auth,'FOLLOWUP_AUTHORIZATION')
    contracts.validate(done,'FOLLOWUP_COMPLETE',dict(authorization=auth['content_hash'],
        plan=p['content_hash'],ledger=ledger['content_hash']))
    if done['jobs'] != jobs or done['training_complete'] is not False:
        raise ValueError('Original controller has not finished submitting')
    refs += [file_ref(f/'complete.json'),file_ref(f/'authorization.json')]
print('SWEEP_DEBUG_ACCEPTED='+json.dumps(dict(report=r,jobs=jobs,plan=p,refs=refs),
    sort_keys=True,allow_nan=False))
'''


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_EMBEDDING_SWEEP_DEBUG_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'LUKA_EMBEDDING_SWEEP_DEBUG_{kind}/v1',
                          expected_schema_version=1)
    if any(value.get(k) != v for k,v in FLAGS.items()):
        raise PermissionError('Exploratory scope differs')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Debug transfer parents differ')


def git(project, *args):
    return subprocess.run(['git','-C',str(project),*args], check=True,
                          capture_output=True,text=True).stdout.splitlines()


def matched_runtime(project):
    for row in git(project,'diff','--name-status','--no-renames',MEASURED_COMMIT,'--','src'):
        status,path=row.split('\t',1)
        if status != 'A' or not (path in ADDITIONS or path.startswith(
                'src/hlt_classification/luka_fullstat_embedding_sweep/')):
            raise ValueError('Measured scientific runtime changed: '+row)


def current(project, commit):
    project=Path(project).resolve()
    snapshot=_source(project,commit)
    if (Path(__file__).resolve() != project/MODULE or not REQUIRED <= set(git(project,'ls-files'))
            or not set(git(project,'diff','--name-only',BASE_COMMIT,'--','src')) <= ADDITIONS):
        raise ValueError('Commit complete additive transfer; preserve existing runtime')
    matched_runtime(project)
    return snapshot


def canonical_root(path):
    path=Path(path)
    if path.is_symlink() or path.resolve() != path:
        raise PermissionError('Debug output must not be redirected through a symlink')


def no_original_outputs(original):
    for arm in WEIGHTS:
        path=Path(original['root'])/arm
        if path.exists() or path.is_symlink():
            raise FileExistsError('Original sweep has output; preserve progress and inspect: '+str(path))


def authenticate(path, project):
    path=Path(path).resolve(strict=True)
    original=load_json(path)
    contracts.validate(original,'SPEC')
    gate=load_json(original['gate_path'])
    campaign.validate_gate_record(gate)
    if (original['source_commit'] != MEASURED_COMMIT or gate['source_commit'] != MEASURED_COMMIT
            or path != Path(original['root'])/'campaign_spec.json'):
        raise ValueError('Expected the original measured eight-weight sweep')
    validate_source_snapshot(gate['source_snapshot'],repository=gate['project_dir'],require_clean=True)
    matched_runtime(project)
    print('Authenticating accepted tier3 gate and eight exact jobs in original source...',flush=True)
    ref,evidence=replay(path,original['project_dir'],REPLAY,'SWEEP_DEBUG_ACCEPTED=')
    report=load_json(Path(gate['root'])/'preflight/receipt.json')
    if report != evidence['report'] or set(evidence['jobs']) != set(WEIGHTS):
        raise ValueError('Original evidence changed or incomplete')
    campaign.validate_spec_record(original,gate,report)
    for item in evidence['refs']: validate_file_ref(item)
    no_original_outputs(original)
    return original,gate,report,dict(original_ref=ref,gate_ref=file_ref(original['gate_path']),
        preflight_ref=file_ref(Path(gate['root'])/'preflight/receipt.json'),
        original_jobs=evidence['jobs'],original_plan=evidence['plan'],original_refs=evidence['refs'])


def fields(original, gate, report, project, commit, snapshot, evidence):
    return dict(root=str(Path(gate['root'])/'science_debug'),gate_path=original['gate_path'],
        original_path=str(Path(original['root'])/'campaign_spec.json'),**evidence,
        project_dir=str(Path(project).resolve()),source_commit=commit,source_snapshot=snapshot,
        transfer_policy=POLICY,execution_site=execution_site('sporc_a100_debug'),
        cpus=gate['cpus'],memory_mb=gate['memory_mb'],train_minutes=report['train_minutes'],arms=list(WEIGHTS))


def create(original_path, project, commit):
    snapshot=current(project,commit)
    original,gate,report,evidence=authenticate(original_path,project)
    spec=artifact('SPEC',parents=dict(original=original['content_hash'],gate=gate['content_hash'],
        preflight=report['content_hash'],source=snapshot['content_hash']),
        **fields(original,gate,report,project,commit,snapshot,evidence))
    root=Path(spec['root']); canonical_root(root)
    if root.exists():
        if load_json(root/'campaign_spec.json') != spec:
            raise FileExistsError('Existing debug transfer differs; preserve it')
    else:
        root.mkdir(exist_ok=False)
        write_immutable_json(root/'campaign_spec.json',spec)
    validate_spec(spec,gate,report)
    return spec


def validate_spec(spec, gate, report):
    original=load_json(spec['original_path'])
    campaign.validate_spec_record(original,gate,report)
    validate(spec,'SPEC',dict(original=original['content_hash'],gate=gate['content_hash'],
        preflight=report['content_hash'],source=spec['source_snapshot']['content_hash']))
    for ref in [spec['original_ref'],spec['gate_ref'],spec['preflight_ref'],*spec['original_refs']]:
        validate_file_ref(ref)
    from .submission import render as original_render
    if spec['original_plan'] != original_render(original,gate):
        raise ValueError('Original commands differ')
    evidence={k:spec[k] for k in ('original_ref','gate_ref','preflight_ref','original_jobs','original_plan','original_refs')}
    expected=fields(original,gate,report,spec['project_dir'],spec['source_commit'],spec['source_snapshot'],evidence)
    if (any(spec.get(k) != v for k,v in expected.items())
            or spec['source_snapshot']['git_commit'] != spec['source_commit']
            or set(spec['original_jobs']) != set(WEIGHTS) or len(set(spec['original_jobs'].values())) != 8
            or any(not str(j).isdigit() for j in spec['original_jobs'].values())
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec):
        raise ValueError('Debug science registration differs')
    for minutes in spec['train_minutes'].values():
        if type(minutes) is not int or not 1 <= minutes <= 1440:
            raise ValueError('Measured request exceeds unchanged 24h ceiling')
    canonical_root(spec['root'])


def check_spec(spec, *, retired=False):
    if current(spec['project_dir'],spec['source_commit']) != spec['source_snapshot']:
        raise ValueError('Debug source changed')
    original,gate,report,evidence=authenticate(spec['original_path'],spec['project_dir'])
    validate_spec(spec,gate,report)
    if any(spec[k] != v for k,v in evidence.items()):
        raise ValueError('Original journal/evidence changed')
    if retired:
        from .debug_retirement import confirm_retired
        confirm_retired(spec)
    return gate,report


def render(spec):
    validate(spec,'SPEC')
    if (spec['execution_site'] != execution_site('sporc_a100_debug')
            or (spec['cpus'],spec['memory_mb']) != (6,90000) or spec['arms'] != list(WEIGHTS)):
        raise ValueError('Only the registered debug resource transfer is allowed')
    commands=[]
    for arm in WEIGHTS:
        minutes=spec['train_minutes'][arm]
        if type(minutes) is not int or not 1 <= minutes <= 1440: raise ValueError('Invalid measured walltime')
        command=['sbatch','--parsable','--no-requeue','--nodes=1','--ntasks=1','--export=NONE',
            '--partition=debug','--account=reu-aisocial','--qos=qos_tier3',
            '--cpus-per-task=6','--mem=90000M','--time='+_walltime(minutes),'--gres=gpu:a100:1',
            '--job-name=lukaesd_'+arm,'--comment=lukaesd:'+spec['content_hash']+':'+arm,
            '--chdir='+spec['project_dir'],'--output='+str(Path(spec['root'])/(arm+'-%j.out')),
            spec['project_dir']+'/sbatch/run_luka_embedding_sweep_debug.sh',spec['project_dir'],
            str(Path(spec['root'])/'campaign_spec.json'),arm]
        commands.append(dict(task_id=arm,dependencies=[],command=command))
    return artifact('PLAN',parents=dict(spec=spec['content_hash']),jobs=8,mode='science',
                    authorization=AUTHORIZATION,commands=commands)


def submit(spec, *, execute=False, plan_hash=None, authorization=None):
    check_spec(spec,retired=execute)
    plan=render(spec)
    if execute and (plan_hash != plan['content_hash'] or authorization != AUTHORIZATION):
        raise PermissionError('Exact reviewed debug plan hash and authorization required')
    root=Path(spec['root'])/'submission'; root.mkdir(exist_ok=True)
    saved,dry,ledger=root/'command_plan.json',root/'dry_run_submission_ledger.json',root/'submission_ledger.json'
    if execute and (not saved.is_file() or load_json(saved) != plan):
        raise ValueError('Saved unchanged dry review required')
    write_immutable_json(saved,plan)
    check_site(plan)
    if execute:
        expected=build_submission_ledger(campaign_spec_sha256=spec['content_hash'],
            jobs={arm:'1' for arm in WEIGHTS},commands={r['task_id']:r['command'] for r in plan['commands']},dry_run=True)
        if not dry.is_file() or load_json(dry) != expected: raise ValueError('Canonical dry ledger required')
        if not ledger.is_file():
            (root/'live_claim').mkdir(exist_ok=False)
            write_immutable_json(root/'live_claim/intent.json',artifact('SUBMISSION_CLAIM',
                parents=dict(spec=spec['content_hash'],plan=plan['content_hash'])))
    value=submit_exact_dag(identity=spec['content_hash'],plan=plan,output=ledger if execute else dry,
        canonical_dry_run=dry,execute=execute,environment=clean_environment())
    return value if execute else plan


def run(spec, arm):
    gate, measured = check_spec(spec, retired=True)
    scoped = arm_gate(gate, arm)
    job = execution(dict(gate, execution_site=spec['execution_site']), measured)
    storage(gate)
    root = Path(spec['root'])/arm
    root.mkdir(exist_ok=False)
    log = PhaseLog(root, lambda: memory_snapshot(torch))
    with log.phase('load_authenticated_HLT_and_shared_TRAIN_targets'):
        meta = regular_cache.metadata(gate['anchor']['evidence']['prepared_root'])
        train = data(gate, 'train', 'D000', meta)
        val = data(gate, 'validation', 'D000', meta)
        bank = load_bank(gate, train)
        if bank.manifest['content_hash'] != measured['bank_hash']:
            raise ValueError('Measured bank changed')
    with log.phase('scientific_fixed30_fit'):
        record = fit(root/'fit', scoped, train, val, bank, device='cuda', technical=False)
    with log.phase('selected_comparison'):
        _, student = validate_fit(root/'fit', scoped, bank.manifest['content_hash'], technical=False)
        result = artifact('RESULTS', parents=dict(spec=spec['content_hash'], fit=record['content_hash']),
            arm=arm, weight=WEIGHTS[arm], counts=gate['counts'], single_training_seed=True,
            selection_population='entire_validation', original50k_selection=False,
            rows=comparison(scoped, student))
        write_immutable_json(root/'results.json', result)
    storage(gate)
    if current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']:
        raise ValueError('Source changed during science')
    validate_source_snapshot(gate['source_snapshot'], repository=gate['project_dir'], require_clean=True)
    outputs = [file_ref(root/f, root=root) for f in ('results.json', 'fit/receipt.json',
               'fit/selected.pt', 'fit/training_report.json', 'fit/validation_predictions.npz')]
    receipt = artifact('SCIENCE', parents=dict(spec=spec['content_hash'], preflight=measured['content_hash']),
                       arm=arm, weight=WEIGHTS[arm], job_id=job, outputs=outputs)
    write_immutable_json(root/'receipt.json', receipt)
    return result


def results(spec):
    """Read partial or complete comparisons, without loading a model/old cache."""
    gate = load_json(spec['gate_path'])
    from .preflight import validate_report
    measured = load_json(Path(gate['root'])/'preflight/receipt.json')
    validate_report(gate, measured)
    validate_spec(spec, gate, measured)
    rows, pending = [], []
    for arm in WEIGHTS:
        root = Path(spec['root'])/arm
        if not (root/'receipt.json').is_file():
            pending.append(arm)
            continue
        receipt = load_json(root/'receipt.json')
        validate(receipt, 'SCIENCE', dict(spec=spec['content_hash'], preflight=measured['content_hash']))
        required = {'results.json', 'fit/receipt.json', 'fit/selected.pt',
                    'fit/training_report.json', 'fit/validation_predictions.npz'}
        if (receipt['arm'] != arm or receipt['weight'] != WEIGHTS[arm]
                or len(receipt['outputs']) != len(required)
                or {r['path'] for r in receipt['outputs']} != required):
            raise ValueError('Science receipt differs')
        for ref in receipt['outputs']:
            validate_file_ref(ref, root=root)
        scoped = arm_gate(gate, arm)
        _, student = validate_fit(root/'fit', scoped, measured['bank_hash'], technical=False)
        result = load_json(root/'results.json')
        fit_record = load_json(root/'fit/receipt.json')
        validate(result, 'RESULTS', dict(spec=spec['content_hash'], fit=fit_record['content_hash']))
        if (result['rows'] != comparison(scoped, student) or result['counts'] != gate['counts']
                or result['arm'] != arm or result['weight'] != WEIGHTS[arm]
                or result['selection_population'] != 'entire_validation'
                or result['original50k_selection'] is not False or result['single_training_seed'] is not True):
            raise ValueError('Saved comparison replay differs')
        if not rows:
            rows.extend(result['rows'][:3])
        elif rows[:3] != result['rows'][:3]:
            raise ValueError('Comparison reference rows differ')
        rows.append(result['rows'][-1])
    return dict(rows=rows, pending=pending, independent_test_available=False)

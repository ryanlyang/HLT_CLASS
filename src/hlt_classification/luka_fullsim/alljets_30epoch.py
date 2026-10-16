"""New fixed30 experiment, reusing immutable batch128 technical evidence only."""
from pathlib import Path
import hashlib
import math
import os
import re
import subprocess

from hlt_classification.data.cache_contracts import load_json, sha256_file, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe
from .foundation import _source
from .preflight_reuse import _files
from . import alljets_fusion_campaign as c, alljets_report_recovery as recovery
from . import alljets_fusion_preflight as pre

AUTHORIZATION = 'AUTHORIZE LUKA MATCHED BATCH128 FIXED30 EXACT PLAN'
POLICY = dict(version='fixed30_from_authenticated_batch128_resources_v1',
    original_job=recovery.FAILED_JOB, original_source=recovery.FAILED_COMMIT,
    original_recipe_preserved=True, gpu_work_repeated=False, technical_only_reuse=True,
    epochs=30, train_margin=1.75, reduce_margin=2., debug_limit_minutes=1440,
    cache_counted_once=True, validation_in_measured_pass=True,
    selected_inference_readback_included=True, automatic_followup=False)
FULLSTAT_COMMIT = '592a6dace8adc662903d3da0d409bcab5fefb9a7'
FULLSTAT = {'src/hlt_classification/luka_fullstat/'+n+'.py' for n in (
    '__init__', 'cache', 'campaign', 'contracts', 'data', 'preflight', 'recipe', 'submission', 'worker')}
ADDITIONS = recovery.ADDITIONS | {
    'src/hlt_classification/luka_fullsim/alljets_30epoch.py',
    'src/hlt_classification/jetclass2_delphes/batch128_30e_recipe.py'}
CHANGES = recovery.CHANGES | {'src/hlt_classification/luka_fullsim/alljets_fusion_preflight.py',
    'src/hlt_classification/jetclass2_delphes/runner.py'}
REQUIRED = ADDITIONS | CHANGES | {
    'scripts/queue_luka_alljets_30epoch.sh', 'scripts/luka_fullsim_alljets_fusion.py',
    'sbatch/run_luka_fullsim_alljets_fusion.sh', 'tests/test_luka_alljets_30epoch.py',
    'docs/plans/LUKA_ALLJETS_30E_PLAN.md', 'docs/contracts/LUKA_ALLJETS_30E.md',
    'docs/LUKA_ALLJETS_30E_RUNBOOK.md'}


def source(project, commit):
    project = Path(project).resolve()
    active = _source(project, commit)
    tracked = set(subprocess.run(['git','-C',str(project),'ls-files'],check=True,
        capture_output=True,text=True).stdout.splitlines())
    if not REQUIRED <= tracked or Path(__file__).resolve() != project/'src/hlt_classification/luka_fullsim/alljets_30epoch.py':
        raise ValueError('Commit the complete fixed30 implementation in this worktree')
    return active


def compatible(old, active, project):
    before_root, after_root = Path(old['project_dir']), Path(project)
    validate_source_snapshot(old['source_snapshot'],repository=before_root,require_clean=True)
    validate_source_snapshot(active,repository=after_root,require_clean=True)
    before, after = _files(before_root), _files(after_root)
    # FullStat is a registered concurrent namespace, never trial data/model code.
    if not before or not before <= after or after-before != ADDITIONS | FULLSTAT:
        raise ValueError('Unregistered fixed30 source additions/deletions')
    records = {}
    for name in sorted(before):
        prior, current = sha256_file(before_root/name), sha256_file(after_root/name)
        if prior != current and name not in CHANGES:
            raise ValueError('Unregistered fixed30 model/data/source change: '+name)
        records[name] = dict(original=prior,execution=current)
    fullstat = {}
    for name in sorted(FULLSTAT):
        expected = subprocess.run(['git','-C',str(after_root),'show',FULLSTAT_COMMIT+':'+name],
                                  check=True,capture_output=True).stdout
        digest = hashlib.sha256(expected).hexdigest()
        # Git content may be LF while a Windows checkout uses CRLF. Source
        # snapshot still binds exact physical bytes; normalize only this comparison.
        if (after_root/name).read_bytes().replace(b'\r\n',b'\n') != expected.replace(b'\r\n',b'\n'):
            raise ValueError('Concurrent FullStat source differs from pinned donor: '+name)
        fullstat[name] = dict(git_sha256=digest,execution=sha256_file(after_root/name))
    return dict(modules=records,additions=sorted(ADDITIONS),allowed_changes=sorted(CHANGES),
                concurrent_fullstat_commit=FULLSTAT_COMMIT,concurrent_fullstat=fullstat)


def projections(measurements):
    if not measurements:
        raise ValueError('Missing measured families')
    for m in measurements:
        if any(type(x) not in (int,float) or not math.isfinite(x) or x <= 0 for x in
               (m['cache_seconds'],m['training']['runtime_seconds'],m['inference_seconds'])):
            raise ValueError('Invalid measured timing')
    train = max(30,math.ceil(max(1.75*(m['cache_seconds']+30*m['training']['runtime_seconds']+
                                     m['inference_seconds']) for m in measurements)/60))
    reduce = max(15,math.ceil(max(2*(m['cache_seconds']+m['inference_seconds']) for m in measurements)/60))
    return train, reduce


def assess(data, old):
    # Never alter old candidate fields or treat a failed 100-pass budget as passed.
    pre.validate_resource_reuse(data['candidate'],old,data['prepared'])
    times = projections(data['candidate']['measurements'])
    if max(times) > POLICY['debug_limit_minutes']:
        raise ValueError(f'Fixed30 still exceeds debug24h: train={times[0]}min reduce={times[1]}min')
    return times


def create(*, failed_gate, root, project, commit):
    path, root, project = Path(failed_gate).resolve(strict=True),Path(root).resolve(),Path(project).resolve()
    active = source(project,commit)
    old = load_json(path)
    reuse = compatible(old,active,project)
    recovery.batch128.location(root,old,project)
    account = recovery.accounting()
    print('Authenticating saved batch128 probes; assessing NEW fixed30 budget; no GPU work...',flush=True)
    data = recovery.replay(path,account,project)
    times = assess(data,old)  # Fail before creating a gate or queueing a CPU job.
    gate = c.artifact('GATE',version=6,parents=dict(source=active['content_hash'],failed_gate=old['content_hash']),
        root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=active,
        parent_import=old['parent_import'],prepared_sha256=old['prepared_sha256'],
        resource_reuse=dict(original_gate=file_ref(path),accounting=account,source_reuse=reuse,
            inputs=data['inputs'],candidate_sha256=data['candidate']['content_hash']),**c.registration(6))
    root.mkdir(parents=True,exist_ok=False)
    write_immutable_json(root/'gate_spec.json',gate)
    print(f'Fixed30 admissible estimate: train={times[0]}min reduce={times[1]}min; original100 budget unchanged.',flush=True)
    return gate


def validate_gate(gate):
    c.validate(gate,'GATE')
    if (gate['schema_version'] != 6 or any(gate.get(k)!=v for k,v in c.registration(6).items())
            or source(gate['project_dir'],gate['source_commit'])!=gate['source_snapshot']
            or load_json(Path(gate['root'])/'gate_spec.json')!=gate):
        raise ValueError('Fixed30 source/registration differs')
    reuse = gate['resource_reuse']
    path = validate_file_ref(reuse['original_gate'])
    old = load_json(path)
    if (old['schema_version']!=4 or old['source_commit']!=recovery.FAILED_COMMIT
            or gate['parents']!=dict(source=gate['source_snapshot']['content_hash'],failed_gate=old['content_hash'])
            or gate['parent_import']!=old['parent_import'] or gate['prepared_sha256']!=old['prepared_sha256']
            or compatible(old,gate['source_snapshot'],gate['project_dir'])!=reuse['source_reuse']):
        raise ValueError('Fixed30 resource/source lineage changed')
    parent = recovery.batch128.location(gate['root'],old,gate['project_dir'])
    expected = recovery.parse_accounting('\n'.join('|'.join(row[k] for k in recovery.ACCOUNT_FIELDS)
                                                 for row in reuse['accounting']['records']))
    if expected != reuse['accounting']:
        raise ValueError('Original accounting payload differs')
    for ref in reuse['inputs']:
        validate_file_ref(ref)
    return parent


def verified_replay(gate):
    parent = validate_gate(gate)
    reuse = gate['resource_reuse']
    data = recovery.replay(reuse['original_gate']['path'],reuse['accounting'],gate['project_dir'])
    if (data['inputs']!=reuse['inputs'] or data['candidate']['content_hash']!=reuse['candidate_sha256']
            or data['prepared']['content_hash']!=gate['prepared_sha256'] or data['parent']!=parent):
        raise ValueError('Fixed30 original resource evidence changed')
    assess(data,load_json(reuse['original_gate']['path']))
    return data


def report(gate,data,allocated):
    old = data['candidate']
    train, reduce = projections(old['measurements'])
    return c.bound_artifact(gate,'PREFLIGHT',parents=dict(gate=gate['content_hash'],prepared=gate['prepared_sha256']),
        job_id=allocated['job_id'],recovery_allocation=allocated,source_snapshot=gate['source_snapshot'],
        original_gpu_job_id=recovery.FAILED_JOB,original_candidate=old,
        original_gpu_accounting=gate['resource_reuse']['accounting'],original_admission_error=data['admission_error'],
        inputs=data['inputs'],memory_evidence=data['memory_evidence'],exit_evidence=data['exit_evidence'],
        recipe=recipe(),resource_policy=POLICY,projected_train_minutes=train,projected_reduce_minutes=reduce,
        fits_debug_24h=max(train,reduce)<=1440,science_admissible=True,
        **{k:old[k] for k in ('environment','gpu','cublas_workspace_config','cudnn_v8_disabled','counts',
            'measurements','peak_rss_bytes','gpu_peak_bytes','resource_envelope_ok',
            'checkpoint_probability_roundtrip','technical_checks_passed')})


def run(gate):
    from hlt_classification.cms2jc2_response.measurement import allocation
    allocated = allocation(cpus=1)
    if os.environ.get('SLURM_MEM_PER_NODE')!='32768':
        raise PermissionError('Fixed30 admission requires the registered32GiB CPU allocation')
    data = verified_replay(gate)
    if recovery.accounting()!=gate['resource_reuse']['accounting']:
        raise ValueError('Original Slurm accounting changed')
    output = Path(gate['root'])/'preflight'
    output.mkdir(exist_ok=False)
    value = report(gate,data,allocated)
    write_immutable_json(output/'measurement.json',value)
    validate_gate(gate)
    write_immutable_json(output/'preflight.json',value)
    return value


def evidence(gate):
    data = verified_replay(gate)
    value = load_json(Path(gate['root'])/'preflight/preflight.json')
    c.validate_bound(value,gate,'PREFLIGHT',dict(gate=gate['content_hash'],prepared=gate['prepared_sha256']))
    allocated = value['recovery_allocation']
    from hlt_classification.cms2jc2_response.contracts import validate as validate_response
    validate_response(allocated,'ALLOCATION')
    if (any(allocated.get(k)!=v for k,v in dict(cpus=1,gpus=0,partition='debug',account='reu-aisocial').items())
            or not re.fullmatch('[1-9][0-9]*',str(allocated['job_id'])) or allocated['job_id']==recovery.FAILED_JOB
            or value!=report(gate,data,allocated)
            or load_json(Path(gate['root'])/'preflight/measurement.json')!=value):
        raise ValueError('Fixed30 admission/allocation differs')
    return data['parent'],data['prepared'],value

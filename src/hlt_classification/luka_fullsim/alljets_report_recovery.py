"""CPU-only authenticated recovery of the completed batch128 GPU probes."""
from pathlib import Path
from decimal import Decimal, ROUND_CEILING
import json
import os
import re
import subprocess
import sys

from hlt_classification.data.cache_contracts import load_json, sha256_file, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from .foundation import _source
from .preflight_reuse import _files
from . import alljets_fusion_campaign as c, alljets_batch128 as batch128

FAILED_COMMIT = '922bd972ef9ec52079e9e750e6e7896e857aba8b'
FAILED_JOB = '21898881'
AUTHORIZATION = 'AUTHORIZE LUKA BATCH128 REPORT RECOVERY EXACT PLAN'
POLICY = dict(version='batch128_report_only_recovery_v1', original_job=FAILED_JOB,
    original_state='FAILED', original_exit_code='1:0', gpu_work_repeated=False,
    memory_method='slurm_single_task_batch_maxrss_plus_largest_sequential_child_tree_peak_v1',
    incomplete_coordinator_sampler_not_used=True, automatic_followup=False)
ACCOUNT_FIELDS = ['JobIDRaw','JobName','Partition','Account','State','ExitCode','AllocCPUS','NNodes','NTasks','ReqMem','MaxRSS']
ADDITIONS = {'src/hlt_classification/luka_fullsim/alljets_report_'+n+'.py' for n in ('recovery','replay')}
CHANGES = {'src/hlt_classification/cms2jc2_response/measurement.py'} | {
    'src/hlt_classification/luka_fullsim/alljets_fusion_'+n+'.py' for n in ('campaign','worker','submission')}
REQUIRED = ADDITIONS | CHANGES | {'scripts/luka_fullsim_alljets_fusion.py',
    'scripts/queue_luka_batch128_report_recovery.sh', 'tests/test_luka_batch128_report_recovery.py',
    'tests/test_process_measurement.py', 'docs/plans/LUKA_BATCH128_REPORT_RECOVERY_PLAN.md',
    'docs/contracts/LUKA_BATCH128_REPORT_RECOVERY.md', 'docs/LUKA_BATCH128_REPORT_RECOVERY_RUNBOOK.md'}


def size_bytes(value):
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT])(?:[nc])?', value)
    if not match:
        raise ValueError('Missing/invalid Slurm memory value: '+value)
    result = int((Decimal(match[1]) * 1024**('KMGT'.index(match[2])+1)).to_integral_value(rounding=ROUND_CEILING))
    if result <= 0:
        raise ValueError('Slurm memory must be positive')
    return result


def accounting():
    fields = ACCOUNT_FIELDS
    result = subprocess.run(['sacct','-j',FAILED_JOB,'-n','-P','--units=K','-o',
        ','.join('JobName%64' if f == 'JobName' else f for f in fields)],
        check=True,capture_output=True,text=True)
    return parse_accounting(result.stdout, fields)


def parse_accounting(text, fields=None):
    fields = fields or ACCOUNT_FIELDS
    rows = []
    for line in text.splitlines():
        if not line.strip(): continue
        values = [v.strip() for v in line.strip().split('|')]
        if len(values) == len(fields)+1 and not values[-1]: values.pop()
        if len(values) != len(fields): raise ValueError('Malformed sacct record')
        rows.append(dict(zip(fields, values)))
    names = [r['JobIDRaw'] for r in rows]
    if len(set(names)) != len(names) or not {FAILED_JOB,FAILED_JOB+'.batch'} <= set(names) or not set(names) <= {FAILED_JOB,FAILED_JOB+'.batch',FAILED_JOB+'.extern'}:
        raise ValueError('Expected exact single batch step accounting')
    job = next(r for r in rows if r['JobIDRaw']==FAILED_JOB)
    batch = next(r for r in rows if r['JobIDRaw']==FAILED_JOB+'.batch')
    if (any(r['State']!='FAILED' or r['ExitCode']!='1:0' or r['AllocCPUS']!='6' or r['NNodes']!='1' for r in (job,batch))
            or batch['NTasks']!='1' or job['JobName']!='lukab_preflight' or job['Partition']!='debug'
            or job['Account']!='reu-aisocial' or size_bytes(job['ReqMem']) != c.MEMORY_MB*1024**2):
        raise ValueError('Original single-task failed allocation differs')
    return dict(job_id=FAILED_JOB,state='FAILED',exit_code='1:0',batch_maxrss_bytes=size_bytes(batch['MaxRSS']),
                records=sorted(rows,key=lambda r:r['JobIDRaw']))


def source(project, commit):
    active = _source(Path(project), commit)
    tracked = set(subprocess.run(['git','-C',str(project),'ls-files'],check=True,capture_output=True,text=True).stdout.splitlines())
    if not REQUIRED <= tracked or Path(__file__).resolve() != Path(project).resolve()/'src/hlt_classification/luka_fullsim/alljets_report_recovery.py':
        raise ValueError('Commit the complete report-recovery implementation in this worktree')
    return active


def compatible(old, active, project):
    before_root, after_root = Path(old['project_dir']), Path(project)
    validate_source_snapshot(old['source_snapshot'],repository=before_root,require_clean=True)
    validate_source_snapshot(active,repository=after_root,require_clean=True)
    before,after = _files(before_root),_files(after_root)
    if not before or not before<=after or after-before != ADDITIONS:
        raise ValueError('Unregistered recovery source additions/deletions')
    records={}
    for name in sorted(before):
        prior,current = sha256_file(before_root/name),sha256_file(after_root/name)
        if prior != current and name not in CHANGES:
            raise ValueError('Unregistered model/data/recipe source change: '+name)
        records[name]=dict(original=prior,execution=current)
    return dict(modules=records,additions=sorted(ADDITIONS),allowed_changes=sorted(CHANGES))


def replay(path, account, project):
    old=load_json(path)
    c.validate(old,'GATE')
    if (old['schema_version']!=4 or old['source_commit']!=FAILED_COMMIT
            or old['source_snapshot']['git_commit']!=FAILED_COMMIT
            or Path(path).resolve()!=Path(old['root'])/'gate_spec.json'):
        raise ValueError('Expected original failed 922bd972 batch128 gate')
    env=dict(os.environ,PYTHONPATH=str(Path(old['project_dir'])/'src'),PYTHONNOUSERSITE='1',
             PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
    script=Path(project)/'src/hlt_classification/luka_fullsim/alljets_report_replay.py'
    proc=subprocess.run([sys.executable,'-u','-B','-s',str(script),str(path)],cwd=old['project_dir'],env=env,
                        input=json.dumps(account,allow_nan=False),check=False,capture_output=True,text=True)
    if proc.returncode:
        raise ValueError('Original-source read-only replay failed:\n'+proc.stderr)
    rows=[json.loads(line.removeprefix('LUKA_REPORT_REPLAY=')) for line in proc.stdout.splitlines()
          if line.startswith('LUKA_REPORT_REPLAY=')]
    if len(rows)!=1 or rows[0]['gate_sha256']!=old['content_hash']:
        raise ValueError('Missing original replay result')
    return rows[0]


def create(*, failed_gate, root, project, commit):
    path,root,project=Path(failed_gate).resolve(strict=True),Path(root).resolve(),Path(project).resolve()
    active=source(project,commit)
    old=load_json(path)
    reuse=compatible(old,active,project)
    batch128.location(root,old,project)
    account=accounting()
    print('Authenticating twelve saved probes in their original source; no GPU work...',flush=True)
    data=replay(path,account,project)
    gate=c.artifact('GATE',version=5,parents=dict(source=active['content_hash'],failed_gate=old['content_hash']),
        root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=active,
        parent_import=old['parent_import'],prepared_sha256=old['prepared_sha256'],
        report_recovery=dict(original_gate=file_ref(path),accounting=account,source_reuse=reuse,
                             inputs=data['inputs'],candidate_sha256=data['candidate']['content_hash']),**c.registration(5))
    root.mkdir(parents=True,exist_ok=False)
    write_immutable_json(root/'gate_spec.json',gate)
    show(data)
    return gate


def validate_gate(gate):
    c.validate(gate,'GATE')
    if (gate['schema_version']!=5 or any(gate.get(k)!=v for k,v in c.registration(5).items())
            or source(gate['project_dir'],gate['source_commit'])!=gate['source_snapshot']
            or load_json(Path(gate['root'])/'gate_spec.json')!=gate):
        raise ValueError('Report-recovery source/registration differs')
    reuse=gate['report_recovery']
    path=validate_file_ref(reuse['original_gate'])
    old=load_json(path)
    if (old['schema_version']!=4 or old['source_commit']!=FAILED_COMMIT
            or gate['parents']!=dict(source=gate['source_snapshot']['content_hash'],failed_gate=old['content_hash'])
            or gate['parent_import']!=old['parent_import'] or gate['prepared_sha256']!=old['prepared_sha256']
            or compatible(old,gate['source_snapshot'],gate['project_dir'])!=reuse['source_reuse']):
        raise ValueError('Recovery lineage/source changed')
    parent=batch128.location(Path(gate['root']).resolve(),old,gate['project_dir'])
    expected=parse_accounting('\n'.join('|'.join(r[k] for k in ACCOUNT_FIELDS) for r in reuse['accounting']['records']))
    if expected!=reuse['accounting']:
        raise ValueError('Recovery accounting payload differs')
    for ref in reuse['inputs']: validate_file_ref(ref)
    return parent


def verified_replay(gate):
    parent=validate_gate(gate)
    reuse=gate['report_recovery']
    data=replay(reuse['original_gate']['path'],reuse['accounting'],gate['project_dir'])
    if (data['inputs']!=reuse['inputs'] or data['candidate']['content_hash']!=reuse['candidate_sha256']
            or data['prepared']['content_hash']!=gate['prepared_sha256'] or data['parent']!=parent):
        raise ValueError('Original evidence changed during recovery')
    return data


def show(data):
    r=data['candidate']
    print(f"Report-only replay: train request={r['projected_train_minutes']}min; reduce={r['projected_reduce_minutes']}min; "
          f"conservative host={r['peak_rss_bytes']/2**30:.2f}GiB; GPU={r['gpu_peak_bytes']/2**30:.2f}GiB",flush=True)
    print('Admission: '+(data['admission_error'] or 'all original checks passed'),flush=True)


def report(gate,data,allocation):
    old=data['candidate']
    return c.bound_artifact(gate,'PREFLIGHT',parents=dict(gate=gate['content_hash'],prepared=gate['prepared_sha256']),
        job_id=allocation['job_id'],recovery_allocation=allocation,source_snapshot=gate['source_snapshot'],
        original_gpu_job_id=FAILED_JOB,original_gpu_accounting=gate['report_recovery']['accounting'],
        original_execution_source=old['source_snapshot'],original_candidate=old,
        inputs=data['inputs'],memory_evidence=data['memory_evidence'],exit_evidence=data['exit_evidence'],
        admission_error=data['admission_error'],science_admissible=data['admission_error'] is None,**{k:old[k] for k in (
            'environment','gpu','cublas_workspace_config','cudnn_v8_disabled','counts','recipe','measurements',
            'projected_train_minutes','projected_reduce_minutes','peak_rss_bytes','gpu_peak_bytes',
            'resource_envelope_ok','fits_debug_24h','checkpoint_probability_roundtrip','technical_checks_passed')})


def run(gate):
    from hlt_classification.cms2jc2_response.measurement import allocation
    allocated=allocation(cpus=1)
    if os.environ.get('SLURM_MEM_PER_NODE')!='32768':
        raise PermissionError('Recovery requires the registered 32GiB CPU allocation')
    data=verified_replay(gate)
    if accounting()!=gate['report_recovery']['accounting']:
        raise ValueError('Original Slurm accounting changed')
    output=Path(gate['root'])/'preflight'
    output.mkdir(exist_ok=False)
    value=report(gate,data,allocated)
    write_immutable_json(output/'measurement.json',value)
    show(data)
    if data['admission_error'] is not None:
        raise ValueError('Report recovered, but science remains blocked: '+data['admission_error'])
    validate_gate(gate)
    write_immutable_json(output/'preflight.json',value)
    return value


def evidence(gate):
    data=verified_replay(gate)
    value=load_json(Path(gate['root'])/'preflight/preflight.json')
    c.validate_bound(value,gate,'PREFLIGHT',dict(gate=gate['content_hash'],prepared=gate['prepared_sha256']))
    allocation=value['recovery_allocation']
    from hlt_classification.cms2jc2_response.contracts import validate as validate_response
    validate_response(allocation,'ALLOCATION')
    if (any(allocation.get(k)!=v for k,v in dict(cpus=1,gpus=0,partition='debug',account='reu-aisocial').items())
            or not re.fullmatch('[1-9][0-9]*',str(allocation['job_id'])) or allocation['job_id']==FAILED_JOB
            or data['admission_error'] is not None or value!=report(gate,data,allocation)
            or load_json(Path(gate['root'])/'preflight/measurement.json')!=value):
        raise ValueError('Recovered admission/allocation differs')
    return data['parent'],data['prepared'],value

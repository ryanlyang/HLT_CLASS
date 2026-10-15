"""Isolated CUDA diagnosis; candidate execution is NEVER science admission.

Original and candidate cases run in separate processes because an illegal CUDA
access poisons its process. Only cached TRAIN data are loaded. The production
factory, offload, BN populations, batch size, and training recipe are untouched.
"""
from contextlib import contextmanager, nullcontext
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from hlt_classification.data.cache_contracts import load_json, write_immutable_json, sha256_file
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.context_fusion.inputs import new_model
from hlt_classification.context_fusion.worker import largest_raw, stress
from hlt_classification.jetclass2_delphes import dzfix_fusion_model as fusion
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from . import alljets_fusion_campaign as c, alljets_fusion_source as sources
from . import alljets_fusion_worker as w, regular_cache
from .preflight_reuse import _files

AUTHORIZATION = 'AUTHORIZE LUKA FUSION CUDA DIAGNOSTIC EXACT PLAN'
CASES = [('native_async_d000', 'FUSION_D000_D000', False, False),
         ('native_blocking_d000', 'FUSION_D000_D000', False, True),
         ('sync_restore_d000', 'FUSION_D000_D000', True, False),
         ('sync_restore_u050', 'FUSION_U050', True, False)]
POLICY = dict(scientific_fit=False, science_admissible=False, train_only=True,
              automatic_followup=False, automatic_retry=False, batch_size=256,
              stress_steps=3,
              candidate='synchronous_pair_tensor_restore_v1',
              batchnorm_population='unchanged_full_pair_population',
              cases=[dict(name=a, node=b, synchronous_restore=d, launch_blocking=e) for a,b,d,e in CASES])
ORIGINAL_CHECK = '''
import json, sys
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullsim import alljets_fusion_campaign as c
from hlt_classification.luka_fullsim.alljets_fusion_submission import accepted_jobs
g=load_json(sys.argv[1]); c.validate_gate(g)
print('LUKA_DIAG_PARENT='+json.dumps(dict(gate=g['content_hash'],jobs=accepted_jobs(g))))
'''


@contextmanager
def synchronous_restore():
    """Diagnostic only: retain exact storage/layout, wait for H2D completion.

    No convolution tiling, changed BN/reduction, detach, dtype change or data crop.
    Restore the original function even on failure. Only used in a fresh child.
    """
    original = fusion._unpack_pair_tensor
    stats = dict(cuda_restores=0)
    def unpack(packed):
        device, values, shape, strides = packed
        if shape is None:
            return values
        stats['cuda_restores'] += 1
        return values.to(device, non_blocking=False).as_strided(shape, strides)
    try:
        fusion._unpack_pair_tensor = unpack
        yield stats
    finally:
        fusion._unpack_pair_tensor = original


def candidate_parity(primary, context, *, device='cuda', bf16=False):
    """Three-update reference vs synchronous offload, unchanged tolerances."""
    with synchronous_restore() as stats:
        report = fusion.native_offload_parity(primary, context, device=device, bf16=bf16)
    if str(device).startswith('cuda') and stats['cuda_restores'] == 0:
        raise ValueError('Candidate parity did not exercise CUDA restoration')
    return dict(report=report, candidate=POLICY['candidate'], **stats)


def _tensor_metadata(value):
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), stride=list(value.stride()),
                    dtype=str(value.dtype), device=str(value.device), numel=value.numel(),
                    storage_offset=value.storage_offset())
    if isinstance(value, (tuple, list)):
        return [_tensor_metadata(v) for v in value]
    if isinstance(value, dict):
        return {k: _tensor_metadata(v) for k,v in value.items()}
    return value if isinstance(value, (str, int, float, bool, type(None))) else type(value).__name__


class KernelTrace(TorchDispatchMode):
    """Flush operation/shape evidence BEFORE dispatch; no GPU reads on failure."""
    def __init__(self, emit):
        super().__init__()
        self.emit, self.index = emit, 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.index += 1
        name = str(func)
        details = dict(index=self.index, operation=name, args=_tensor_metadata(args),
                       kwargs=_tensor_metadata(kwargs or {}))
        tracked = any(s in name for s in ('convolution', 'batch_norm', 'bmm', 'addmm',
                                          'scaled_dot_product', '_to_copy'))
        if tracked:
            self.emit(dict(status='running', **details))
        try:
            result = func(*args, **(kwargs or {}))
        except BaseException as exc:
            self.emit(dict(status='failed', error=str(exc), **details))
            raise
        if tracked:
            self.emit(dict(status='completed', index=self.index, operation=name))
        return result


def authenticate_original(path):
    """Authenticate the failed gate using ITS unchanged original validator."""
    path = Path(path).resolve(strict=True)
    gate = load_json(path)
    c.validate(gate, 'GATE')
    if path != Path(gate['root'])/'gate_spec.json':
        raise ValueError('Noncanonical original gate path')
    old = Path(gate['project_dir']).resolve(strict=True)
    validate_source_snapshot(gate['source_snapshot'], repository=old, require_clean=True)
    env = dict(os.environ, PYTHONPATH=str(old/'src'), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    print('Authenticating original gate, parent and submission journal read-only...', flush=True)
    result = subprocess.run([sys.executable,'-u','-s','-c',ORIGINAL_CHECK,str(path)],
        cwd=old, env=env, capture_output=True, text=True)
    if result.returncode:
        raise ValueError('Original gate authentication failed:\n'+result.stderr)
    rows = [json.loads(s.removeprefix('LUKA_DIAG_PARENT=')) for s in result.stdout.splitlines()
            if s.startswith('LUKA_DIAG_PARENT=')]
    runtime = load_json(Path(gate['root'])/'preflight/runtime.json')
    if (len(rows) != 1 or rows[0] != dict(gate=gate['content_hash'], jobs={'preflight':str(runtime['job_id'])})):
        raise ValueError('Runtime differs from submitted original gate')
    if (Path(gate['root'])/'preflight/preflight.json').exists() or (Path(gate['root'])/'science').exists():
        raise PermissionError('Use only an unadmitted gate; no science campaign recovery here')
    return gate, runtime


def source_compatibility(project, gate):
    old = Path(gate['project_dir'])
    before, after = _files(old), _files(project)
    added = {'src/hlt_classification/luka_fullsim/alljets_fusion_diagnostic.py'}
    changed_allowed = {'src/hlt_classification/luka_fullsim/alljets_fusion_source.py'}
    if after-before != added or not before <= after:
        raise ValueError('Unexpected diagnostic source additions/deletions')
    for name in before-changed_allowed:
        if sha256_file(old/name) != sha256_file(Path(project)/name):
            raise ValueError('Diagnostic changes existing scientific execution: '+name)


def create(*, root, original_gate, project, commit):
    project, root = Path(project).resolve(), Path(root).resolve()
    active = sources.source(project, commit)
    gate, runtime = authenticate_original(original_gate)
    source_compatibility(project, gate)
    parent = load_json(gate['parent_import']['spec']['path'])
    c.protect(root, parent, project)
    for protected in (Path(gate['root']), Path(gate['project_dir'])):
        if root.is_relative_to(protected) or protected.is_relative_to(root):
            raise PermissionError('Diagnostic overlaps original gate/source')
    spec = c.artifact('CUDA_DIAGNOSTIC_SPEC', parents=dict(gate=gate['content_hash'],source=active['content_hash']),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=active,
        original_gate=file_ref(Path(original_gate).resolve()),
        original_runtime=file_ref(Path(gate['root'])/'preflight/runtime.json'), policy=POLICY,
        execution_site=gate['execution_site'], cpus=6, memory_mb=c.MEMORY_MB, minutes=60)
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root/'diagnostic_spec.json', spec)
    return spec


def validate(spec, *, original=False):
    c.validate(spec, 'CUDA_DIAGNOSTIC_SPEC')
    if (spec['policy'] != POLICY or (spec['cpus'],spec['memory_mb'],spec['minutes']) != (6,c.MEMORY_MB,60)
            or sources.source(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']
            or load_json(Path(spec['root'])/'diagnostic_spec.json') != spec):
        raise ValueError('Diagnostic source/policy/spec differs')
    for name in ('original_gate','original_runtime'):
        validate_file_ref(spec[name])
    gate, runtime = load_json(spec['original_gate']['path']), load_json(spec['original_runtime']['path'])
    c.validate(gate, 'GATE')
    if (spec['parents'] != dict(gate=gate['content_hash'],source=spec['source_snapshot']['content_hash'])
            or spec['execution_site'] != gate['execution_site']
            or Path(spec['original_gate']['path']) != Path(gate['root'])/'gate_spec.json'
            or Path(spec['original_runtime']['path']) != Path(gate['root'])/'preflight/runtime.json'):
        raise ValueError('Diagnostic parent/site differs')
    parent = load_json(gate['parent_import']['spec']['path'])
    root = Path(spec['root']).resolve()
    c.protect(root,parent,Path(spec['project_dir']).resolve())
    for protected in (Path(gate['root']).resolve(),Path(gate['project_dir']).resolve()):
        if root.is_relative_to(protected) or protected.is_relative_to(root):
            raise PermissionError('Diagnostic overlaps original gate/source')
    source_compatibility(spec['project_dir'], gate)
    if original:
        if (gate, runtime) != authenticate_original(spec['original_gate']['path']):
            raise ValueError('Original evidence changed')
    return gate, runtime


def _runtime(spec, original_runtime, *, launch_blocking):
    # Validate BEFORE reading cache arrays or creating a CUDA model.
    blocking = '1' if launch_blocking else '0'
    if os.environ.get('CUDA_LAUNCH_BLOCKING') != blocking:
        raise PermissionError('Set the exact case CUDA_LAUNCH_BLOCKING before starting Python')
    job = w.execution(spec)
    environment, gpu = installed_environment(), gpu_identity()
    if (environment != original_runtime['environment'] or gpu != original_runtime['gpu']
            or os.environ.get('TORCH_CUDNN_V8_API_DISABLED','unset') != original_runtime['cudnn_v8_disabled']):
        raise PermissionError('Diagnostic must first reproduce in the original software/GPU environment')
    return dict(job_id=job, environment=environment, gpu=gpu, cuda_launch_blocking=blocking)


def run_case(spec, name):
    gate, prior_runtime = validate(spec)
    matches = [case for case in CASES if case[0] == name]
    if len(matches) != 1:
        raise ValueError('Unregistered diagnostic case')
    _, node_name, sync_restore, launch_blocking = matches[0]
    output = Path(spec['root'])/'cases'/name
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    # Keep the trace on disk even when CUDA is poisoned or process aborts.
    with (output/'operations.jsonl').open('x', encoding='utf-8') as stream:
        def emit(row):
            stream.write(json.dumps(dict(seconds=time.monotonic()-started, **row), sort_keys=True)+'\n')
            stream.flush()
        try:
            runtime = _runtime(spec, prior_runtime, launch_blocking=launch_blocking)
            write_immutable_json(output/'runtime.json', runtime)
            parent = load_json(gate['parent_import']['spec']['path'])
            auth = regular_cache.metadata(parent['admission']['prepared_root'])
            p = auth[1]
            if p['content_hash'] != gate['prepared_sha256']:
                raise ValueError('Prepared cache differs')
            node = next(n for n in c.nodes() if n['node_id']==node_name)
            train = w.cache(gate, p, node, 'train', authenticated=auth)
            raw = [largest_raw(v) for v in (train.primary, train.context)]
            witness = dict(primary_length=raw[0]['mask'].shape[-1], context_length=raw[1]['mask'].shape[-1],
                           train_rows=len(train), batch_size=256, same_cache=train.primary is train.context,
                           prepared_sha256=p['content_hash'], node=node, candidate=sync_restore)
            write_immutable_json(output/'witness.json', witness)
            print('LUKA DIAGNOSTIC '+name+': cached TRAIN loaded; '+json.dumps(witness,sort_keys=True),flush=True)
            del raw
            parities = {}
            if sync_restore:
                # Small parity witness only; the actual stress below is NEVER cropped.
                samples = [v.batch(np.arange(4)) for v in (train.primary, train.context)]
                for sample in samples:
                    for key in ('features','vectors','mask'):
                        sample[key] = sample[key][..., :8].copy()
                for bf16 in (False, True):
                    print('LUKA DIAGNOSTIC '+name+': parity bf16='+str(bf16),flush=True)
                    parities[str(bf16)] = candidate_parity(*samples, bf16=bf16)
                write_immutable_json(output/'parity.json', parities)
                torch.cuda.empty_cache()
            model = new_model(node)
            torch.cuda.reset_peak_memory_stats()
            emit(dict(phase='batch256_steps3', status='running'))
            print('LUKA DIAGNOSTIC '+name+': batch256, three longest-TRAIN steps', flush=True)
            with (synchronous_restore() if sync_restore else nullcontext(None)) as restored, KernelTrace(emit):
                stats = stress(model, train)
            fusion.validate_offload_stats(stats, calls=3)
            if sync_restore and not restored['cuda_restores']:
                raise ValueError('Full stress did not exercise synchronous restoration')
            result = c.artifact('CUDA_DIAGNOSTIC_CASE', parents=dict(spec=spec['content_hash']), name=name,
                completed=True, science_admissible=False, seconds=time.monotonic()-started,
                witness=witness, parities=parities, offload=stats, synchronous_restore=restored,
                gpu_peak_bytes=torch.cuda.max_memory_allocated())
            write_immutable_json(output/'result.json', result)
            emit(dict(phase='batch256_steps3', status='completed'))
            return result
        except BaseException as exc:
            # No synchronization or memory sampling after an illegal access.
            emit(dict(status='failed', error=str(exc)))
            write_immutable_json(output/'failure.json', c.artifact('CUDA_DIAGNOSTIC_FAILURE',
                parents=dict(spec=spec['content_hash']), name=name, completed=False, science_admissible=False,
                error=repr(exc), traceback=traceback.format_exc()))
            raise


def run(spec):
    gate, prior_runtime = validate(spec, original=True)
    _runtime(spec, prior_runtime, launch_blocking=True)
    root = Path(spec['root'])
    (root/'cases').mkdir(exist_ok=False)
    rows = []
    for name, _, _, blocking in CASES:
        log = root/(name+'.out')
        print('LUKA DIAGNOSTIC starting '+name+' in a fresh process', flush=True)
        with log.open('x', encoding='utf-8') as stream:
            child = subprocess.run([sys.executable,'-u','-s',
                str(Path(spec['project_dir'])/'scripts/luka_fusion_cuda_diagnostic.py'),
                'case','--spec',str(root/'diagnostic_spec.json'),'--case',name],
                cwd=spec['project_dir'], stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_LAUNCH_BLOCKING='1' if blocking else '0'))
        outputs = [file_ref(path, root=root) for path in sorted((root/'cases'/name).rglob('*')) if path.is_file()]
        success = root/'cases'/name/'result.json'
        completed = child.returncode==0 and success.is_file()
        if completed:
            result = load_json(success)
            c.validate(result, 'CUDA_DIAGNOSTIC_CASE', dict(spec=spec['content_hash']))
            if result.get('completed') is not True or result.get('science_admissible') is not False or result.get('name') != name:
                raise ValueError('Child result differs')
        rows.append(dict(name=name, returncode=child.returncode, completed=completed,
                         log=file_ref(log, root=root), outputs=outputs))
        print('LUKA DIAGNOSTIC '+name+': '+('PASS' if completed else 'FAILED; evidence preserved'), flush=True)
    report = c.artifact('CUDA_DIAGNOSTIC_REPORT', parents=dict(spec=spec['content_hash']),
        cases=rows, all_cases_completed=all(r['completed'] for r in rows), science_admissible=False,
        interpretation='A/B kernel diagnosis only. No scientific checkpoints, metrics or admission.',
        timing_scope='instrumented stress; NOT a production throughput measurement')
    write_immutable_json(root/'diagnostic_report.json', report)
    return report


def plan(spec):
    root, site = Path(spec['root']), spec['execution_site']
    args = ['sbatch','--parsable','--no-requeue','--nodes=1','--ntasks=1','--export=NONE',
        '--partition='+site['partition'],'--account='+site['account'],'--qos='+site['qos'],
        '--cpus-per-task=6','--mem=256000M','--time=01:00:00','--gres='+site['gres'],
        '--job-name=lukaaf_cuda_diag','--comment=lukaafdiag:'+spec['content_hash'],
        '--chdir='+spec['project_dir'],'--output='+str(root/'diagnostic-%j.out'),
        spec['project_dir']+'/sbatch/run_luka_fusion_cuda_diagnostic.sh', spec['project_dir'],
        str(root/'diagnostic_spec.json'),'diagnostic']
    return c.artifact('CUDA_DIAGNOSTIC_PLAN',parents=dict(spec=spec['content_hash']),jobs=1,
        commands=[dict(task_id='diagnostic',dependencies=[],command=args)],authorization=AUTHORIZATION)


def submit(spec, *, execute=False, plan_hash=None, authorization=None):
    from .regular_submission import check_site, clean_environment
    from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
    from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger
    validate(spec, original=True)
    value = plan(spec)
    if execute and (plan_hash != value['content_hash'] or authorization != AUTHORIZATION):
        raise PermissionError('Exact diagnostic plan hash and authorization required')
    root = Path(spec['root'])/'submission'
    root.mkdir(exist_ok=True)
    saved, dry, ledger = root/'command_plan.json',root/'dry_run_submission_ledger.json',root/'submission_ledger.json'
    if execute and (not saved.is_file() or load_json(saved)!=value):
        raise PermissionError('Saved exact dry review required')
    write_immutable_json(saved,value)
    check_site(value)
    if execute:
        expected = build_submission_ledger(campaign_spec_sha256=spec['content_hash'],jobs={'diagnostic':'1'},
            commands={'diagnostic':value['commands'][0]['command']},dry_run=True)
        if not dry.is_file() or load_json(dry)!=expected:
            raise PermissionError('Canonical dry ledger required')
        if not ledger.is_file():
            try:
                (root/'live_claim').mkdir(exist_ok=False)
            except FileExistsError as exc:
                raise PermissionError('Active/interrupted submission; preserve journal, do not retry') from exc
    result = submit_exact_dag(identity=spec['content_hash'],plan=value,output=ledger if execute else dry,
        canonical_dry_run=dry,execute=execute,environment=clean_environment())
    return result if execute else value

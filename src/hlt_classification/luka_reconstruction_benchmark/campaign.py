"""Immutable benchmark identity, original R24 evidence, and runtime projections."""
from pathlib import Path
import math
import re
import shutil
import subprocess
import sys

from hlt_classification.data.cache_contracts import load_json, with_content_hash, validate_content_hash, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import capture_source_snapshot
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_next_supervision_hydra import campaign as donor, worker, training, submission
from hlt_classification.jetclass2_delphes.model import installed_environment

DONOR = '58d0c0edf5f41cff2398fb168fc16fc92e169294'
FILES = ('campaign.py', 'data.py', 'kernel.py', 'runner.py', 'submission.py', '__init__.py')
REQUIRED = tuple('src/hlt_classification/luka_reconstruction_benchmark/'+n for n in FILES) + (
    'scripts/luka_r24_gpu_benchmark.py', 'scripts/queue_luka_r24_gpu_benchmark.sh',
    'sbatch/run_luka_r24_gpu_benchmark.sh', 'docs/plans/LUKA_R24_GPU_BENCHMARK_PLAN.md',
    'docs/contracts/LUKA_R24_GPU_BENCHMARK.md', 'tests/test_luka_r24_gpu_benchmark.py',
    'docs/LUKA_R24_GPU_BENCHMARK_RUNBOOK.md','docs/LUKA_R24_GPU_BENCHMARK_VALIDATION.md')


def artifact(kind, *, parents=None, **values):
    return with_content_hash(dict(contract=f'LUKA_R24_GPU_BENCHMARK_{kind}/v1', schema_version=1,
                                  parents=dict(parents or {}), **FLAGS, **values))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'LUKA_R24_GPU_BENCHMARK_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k,v in FLAGS.items()) or (parents is not None and value['parents'] != parents):
        raise ValueError('Benchmark scope or parents differ')


def policy():
    return dict(benchmark_only=True, science_release=False, worlds=[2,3,4], allocated_gpus=4,
        cpus=12, memory_mb=64000, minutes=120, global_batch=128, precision='fp32', tf32=False,
        task='R24', warmup_batches=3, measured_batches=24, validation_batches=32,
        parity_cases=[dict(rows=15,position=3.),dict(rows=16,position=4.5),dict(rows=15,position=7.)],
        position=7., epochs=30, solver_threads='cpus//world', gradient_bucket_elements=1048576,
        synchronization='SyncBatchNorm_global_padding_then_sample_weighted_gradient_sum',
        tolerances=dict(loss_abs=1e-3,gradient_relative_l2=5e-4,buffer_abs=1e-4,
                        parameters_relative_l2=5e-4,adamw_moments_relative_l2=5e-4),
        output_budget_bytes=2*2**30, automatic_retry=False)


def source(project, commit):
    project=Path(project).resolve()
    if not re.fullmatch('[0-9a-f]{40}',commit) or Path(__file__).resolve()!=project/REQUIRED[0]:
        raise ValueError('Use the benchmark driver from its exact pushed worktree')
    snap=capture_source_snapshot(project,require_clean=True)
    names=subprocess.check_output(['git','-C',str(project),'ls-files'],text=True).splitlines()
    if snap['git_commit']!=commit or not set(REQUIRED)<=set(names):
        raise ValueError('Commit the complete benchmark implementation')
    subprocess.run(['git','-C',str(project),'merge-base','--is-ancestor',commit,'origin/main'],check=True,capture_output=True)
    return snap


def original(path):
    old=load_json(path)
    if (old.get('source_commit')!=DONOR or old.get('campaign_kind')!='R8_FOLLOWUP_V2'
            or old.get('execution_version')!='R5'
            or Path(donor.__file__).resolve()!=Path(old['project_dir'])/'src/hlt_classification/luka_next_supervision_hydra/campaign.py'):
        raise ValueError('Import scientific runtime from the original 58d0c0ed worktree')
    donor.check(old,check_storage=False)
    if installed_environment()!=old['reference']['environment'] or Path(sys.prefix).resolve()!=Path(old['environment_dir']):
        raise ValueError('Pinned original environment required')
    return old


def reference(old):
    acceptance=worker.completed_setup(old)
    worker.receipt(old,'R24')
    report,_=training.validate_fit(Path(old['root'])/'R24/fit',old,'R24')
    history=report['validation_history']
    if report['epochs']!=30 or len(history)!=30 or report['technical']:
        raise ValueError('Completed scientific A6000 R24 required')
    paths=['setup/receipt.json','R24/receipt.json','R24/fit/training_report.json',
           'submission_science/submission_ledger.json']
    return dict(acceptance_hash=acceptance['content_hash'], report_hash=report['content_hash'],
        job_id=submission.accepted_jobs(old,'science')['R24'],
        train_seconds=sum(r['train_seconds'] for r in history),
        validation_seconds=sum(r['validation_seconds'] for r in history),
        epoch_train_seconds=[r['train_seconds'] for r in history],
        epoch_validation_seconds=[r['validation_seconds'] for r in history],
        precision='A6000_BF16_native_FP32_decoder',
        files=[file_ref(Path(old['root'])/p) for p in paths])


def storage(spec):
    root=Path(spec['root']); used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    remaining=max(0,spec['output_budget_bytes']-used)
    if used>spec['output_budget_bytes'] or shutil.disk_usage(root if root.exists() else root.parent).free<remaining+2**30:
        raise OSError('Benchmark requires remaining two-GiB allowance plus one GiB reserve')


def create(root, original_spec, project, commit, quota_gib):
    old=original(original_spec); ref=reference(old); root=Path(root).absolute(); project=Path(project).resolve()
    protected=[Path(old[k]).resolve() for k in ('root','project_dir','environment_dir','bundle_root','raw_root','delta_root')]+[project]
    if (root.exists() or not root.parent.is_dir() or root.resolve()!=root
            or any(p.is_symlink() for p in (root,*root.parents))
            or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected)):
        raise ValueError('Use a fresh separate benchmark output root')
    if type(quota_gib) is not int or quota_gib<3: raise ValueError('Reserve at least three personally available GiB')
    snap=source(project,commit)
    spec=artifact('SPEC',parents=dict(original=old['content_hash'],source=snap['content_hash'],reference=ref['report_hash']),
        **policy(),root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=snap,
        original_spec=file_ref(Path(original_spec).resolve()),environment_dir=old['environment_dir'],
        quota_gib=quota_gib,reference=ref,counts=old['counts'],maximum=old['target_inventory']['maximum'])
    storage(spec); root.mkdir(); write_immutable_json(root/'benchmark_spec.json',spec)
    return spec


def check(spec, *, full=False):
    validate(spec,'SPEC'); validate_file_ref(spec['original_spec'])
    old=load_json(spec['original_spec']['path'])
    if full: old=original(spec['original_spec']['path'])
    if (old.get('source_commit')!=DONOR or old.get('campaign_kind')!='R8_FOLLOWUP_V2'
            or old.get('execution_version')!='R5'
            or Path(donor.__file__).resolve()!=Path(old['project_dir'])/'src/hlt_classification/luka_next_supervision_hydra/campaign.py'
            or capture_source_snapshot(old['project_dir'],require_clean=True)!=old['source_snapshot']):
        raise ValueError('Original scientific runtime changed')
    for ref in spec['reference']['files']: validate_file_ref(ref)
    snap=source(spec['project_dir'],spec['source_commit'])
    if (any(spec.get(k)!=v for k,v in policy().items()) or spec['source_snapshot']!=snap
            or spec['parents']!=dict(original=old['content_hash'],source=snap['content_hash'],reference=spec['reference']['report_hash'])
            or spec['counts']!=old['counts'] or spec['maximum']!=old['target_inventory']['maximum']
            or spec['environment_dir']!=old['environment_dir'] or Path(sys.prefix).resolve()!=Path(spec['environment_dir'])
            or load_json(Path(spec['root'])/'benchmark_spec.json')!=spec):
        raise ValueError('Immutable benchmark identity differs')
    if full and reference(old)!=spec['reference']: raise ValueError('A6000 reference changed')
    storage(spec)
    return old


def projection(spec, *, mean_step, mean_validation_batch, probe_seconds, input_seconds, startup_seconds):
    values=(mean_step,mean_validation_batch,probe_seconds,input_seconds,startup_seconds)
    if not all(math.isfinite(x) and x>=0 for x in values) or min(mean_step,mean_validation_batch)<=0:
        raise ValueError('Invalid measured durations')
    train_steps=math.ceil(spec['counts']['train']/128); val_steps=math.ceil(spec['counts']['validation']/128)
    parts=dict(training=30*train_steps*mean_step,input_construction=30*train_steps*input_seconds,
        validation=30*val_steps*mean_validation_batch,diagnostics=30*probe_seconds,startup=startup_seconds)
    return dict(component_seconds=parts,estimated_hours=sum(parts.values())/3600,
        exclusions=['queue_wait','metric_calculation','production_checkpoint_io','additional_source_audits'],
        estimated_not_production_walltime=True)

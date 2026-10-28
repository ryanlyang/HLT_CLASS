"""New execution receipts around unchanged original scientific functions.

Scientific outputs keep the original SPEC identity; RECOVERY_TASK binds their
actual new source/job and only real target parents. No old receipt is rewritten.
"""
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import os
import platform
import re
import subprocess
import sys
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_supervision_depth import worker as old, contracts as science
from .contracts import artifact, validate, ALL_FITS, scientific_dependencies
from . import campaign, source, submission, scratch


def execution(spec, original, task):
    job = os.environ.get('SLURM_JOB_ID', '')
    if not re.fullmatch('[1-9][0-9]*', job):
        raise PermissionError('Real Slurm allocation required')
    p = subprocess.run(['scontrol', 'show', 'job', '-o', job], check=True, capture_output=True, text=True)
    fields = dict(s.split('=', 1) for s in p.stdout.split() if '=' in s)
    if (fields.get('JobName') != 'lukasr_'+task or fields.get('Comment') != 'lukasr:'+spec['content_hash']+':'+task
            or fields.get('WorkDir') != spec['project_dir']):
        raise PermissionError('Recovery execution identity differs')
    if task != 'report':
        return old.gpu_execution(original, original['reuse']['acceptance']['preflight'])
    site = original['execution_site']; prefix = site['conda_base']+'/envs/'+site['conda_env']
    if (fields.get('Account') != site['account'] or fields.get('Partition') != site['partition'] or fields.get('QOS') != site['qos']
            or fields.get('NumNodes') != '1' or fields.get('NumTasks') != '1' or fields.get('NumCPUs') != str(original['report_cpus'])
            or os.environ.get('SLURM_CLUSTER_NAME') != site['cluster'] or platform.machine() != site['architecture']
            or os.environ.get('SLURM_CPUS_PER_TASK') != str(original['report_cpus'])
            or os.environ.get('SLURM_MEM_PER_NODE') != str(original['report_memory_mb'])
            or sys.prefix != prefix or os.environ.get('CONDA_PREFIX') != prefix or os.environ.get('PYTHONNOUSERSITE') != '1'):
        raise PermissionError('Recovery report allocation differs')
    return job


def receipt(spec, original, task, seen=None, *, accounting=True):
    seen = {} if seen is None else seen
    if task in seen:
        return seen[task]
    if task in spec['reused']:
        validate_file_ref(spec['reused'][task])
        row = old.validate_task(original, task)
        if accounting:
            old.submission.successful(original, task, row['job_id'])
    else:
        if task not in ALL_FITS or task not in {t['task_id'] for t in spec['tasks']}:
            raise ValueError('Not a registered replacement fit')
        row = load_json(Path(spec['root'])/task/'receipt.json')
        parents = dict(recovery=spec['content_hash'], original=original['content_hash'])
        for dep in scientific_dependencies(task):
            parents[dep] = receipt(spec, original, dep, seen, accounting=accounting)['content_hash']
        validate(row, 'TASK', parents)
        if (row['task'] != task or row['job_id'] != submission.accepted_jobs(spec).get(task)
                or [r['path'] for r in row['outputs']] != old.output_names(task)):
            raise ValueError('Recovery task output/journal binding differs')
        for ref in row['outputs']:
            validate_file_ref(ref, root=original['root'])
        cost = load_json(Path(original['root'])/task/'costs.json')
        science.validate(cost, 'COSTS', dict(spec=original['content_hash']))
        if (cost['task'] != task or cost['job_id'] != row['job_id'] or cost['execution_recovery'] != spec['content_hash']
                or cost['environment'] != original['anchor']['evidence']['environment'] or cost['separate_preflight_performed'] is not False):
            raise ValueError('Recovery cost/source evidence differs')
        old.validate_fit(original, task)
        if task in science.ARMS and science.ARMS[task]['depths']:
            old.retained_tokens(original, task)
        if task in science.TEACHER_TASKS:
            m = load_json(Path(original['root'])/task/'bank/manifest.json')
            teacher = file_ref(Path(original['root'])/task/'fit/selected.pt')
            old.embedding.validate(m, 'BANK', dict(recipe=old.embedding.recipe()['content_hash'],
                prepared=original['anchor']['evidence']['prepared'], teacher=teacher['sha256']))
            a = load_npz_arrays(Path(original['root'])/task/'bank/targets.npz')
            norm = load_npz_arrays(Path(original['root'])/task/'bank/normalization.npz')
            if (m['teacher'] != teacher or m['coordinate'] != 'OFFLINE' or m['role'] != 'train'
                    or m['rows'] != original['counts']['train'] or m['width'] != 128
                    or set(norm) != {'identities', 'labels', 'mean', 'scale'}
                    or any(not np.array_equal(v, a[k]) for k, v in norm.items())):
                raise ValueError('Recovery teacher normalization differs')
            validate_file_ref(m['payload'], root=Path(original['root'])/task/'bank')
        if accounting:
            submission.successful(spec, task, row['job_id'])
    seen[task] = row
    return row


def run(spec, task, *, device='cuda'):
    original = campaign.check(spec)
    if task not in {t['task_id'] for t in spec['tasks']}:
        raise ValueError('Task is reused or not registered; never rerun it')
    job = execution(spec, original, task)
    if submission.accepted_jobs(spec).get(task) != job:
        raise ValueError('Job not accepted in recovery journal')
    if old.installed_environment() != original['anchor']['evidence']['environment']:
        raise ValueError('Installed environment differs')
    out = Path(spec['root'])/task
    if task == 'report':
        out.mkdir(exist_ok=False); value = results(spec)
        write_immutable_json(out/'results.json', value); old.print_results(value)
        # A failed independent fit remains an explicit pending row, not a reason
        # to prevent publication of all other results. No cleanup follows.
        return value
    parents = dict(recovery=spec['content_hash'], original=original['content_hash']); seen = {}
    for dep in scientific_dependencies(task):
        parents[dep] = receipt(spec, original, dep, seen)['content_hash']
    old.storage(original)
    root = Path(original['root']); folder = root/task
    if root != root.resolve(strict=True) or folder.is_symlink():
        raise PermissionError('Noncanonical original output namespace')
    folder.mkdir(exist_ok=False); out.mkdir(exist_ok=False)
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    log = old.PhaseLog(folder, lambda: old.memory_snapshot(torch))
    meta = old.regular_cache.metadata(original['anchor']['evidence']['prepared_root'])
    if task in science.TEACHER_TASKS:
        with log.phase('OFFLINE_teacher_fixed30_and_compact_export'):
            tr, val = (old.data.load(original, role, 'OFFLINE', meta) for role in ('train', 'validation'))
            bank = SimpleNamespace(manifest=science.artifact('BUNDLE',
                parents=dict(spec=original['content_hash'], recipe=old.training.registered(task)['content_hash']), arm=task, records={}))
            write_immutable_json(folder/'bundle.json', bank.manifest)
            proof = old.startup.run(original, tr, val, bank, original['nodes'][task], device)
            write_immutable_json(folder/'startup.json', proof); old.release()
            old.fit(original, task, tr, val, bank, device)
            ref = file_ref(folder/'fit/selected.pt'); frozen = old.native(ref, device)
            old.embedding.export(folder/'bank', frozen, tr, teacher_ref=ref, device=device)
            a = load_npz_arrays(folder/'bank/targets.npz')
            atomic_publish_bytes(folder/'bank/normalization.npz', deterministic_npz_bytes({k:a[k] for k in ('identities', 'labels', 'mean', 'scale')}))
            del tr, val, frozen, a; old.release()
    else:
        with log.phase('HLT_student_fixed30_owned_scratch'):
            tr, val = (old.data.load(original, role, 'D000', meta) for role in ('train', 'validation'))
            depths = science.ARMS[task]['depths']
            required = int(old.counts(tr).sum()) * 128 * 4 * len(depths) + 512 * 2**20
            lease = scratch.reserve(old.scratch_directory(original), required) if depths else nullcontext()
            with lease:
                old.student_run(original, task, tr, val, meta, device)
            del tr, val; old.release()
    costs = science.artifact('COSTS', parents=dict(spec=original['content_hash']), task=task, job_id=job,
        environment=old.installed_environment(), memory=old.memory_snapshot(torch), separate_preflight_performed=False,
        execution_recovery=spec['content_hash'])
    write_immutable_json(folder/'costs.json', costs)
    if source.current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']:
        raise ValueError('Source changed during recovery worker')
    row = artifact('TASK', parents=parents, task=task, job_id=job,
        outputs=[file_ref(root/n, root=root) for n in old.output_names(task)])
    write_immutable_json(out/'receipt.json', row)
    receipt(spec, original, task, accounting=False)
    return row


def results(spec):
    original = campaign.check(spec, live=False); seen = {}
    receipt(spec, original, 'targets', seen)
    root = Path(original['root']); refs = root/'targets/references'
    arrays = {n:load_npz_arrays(refs/(n+'.npz')) for n in science.REFERENCES}
    ix = load_npz_arrays(refs/'original50k.npz')['indices']; anchor = arrays['M0HLT']
    if (ix.dtype != np.int64 or ix.shape != (original['original50k_rows'],) or len(np.unique(ix)) != len(ix)
            or np.any(ix < 0) or np.any(ix >= original['counts']['validation'])):
        raise ValueError('Invalid original50k report-only subset')
    picks = {}; pending = []
    token_refs = {}
    for task in (*science.TEACHER_TASKS, *science.ARMS):
        ref = Path(spec['reused'][task]['path']) if task in spec['reused'] else Path(spec['root'])/task/'receipt.json'
        if not ref.is_file():
            pending.append(task); continue
        receipt(spec, original, task, seen)
        report, arrays[task] = old.validate_fit(original, task); picks[task] = report['selected_pass']
        if task in science.ARMS and science.ARMS[task]['depths']:
            m = old.retained_tokens(original, task); a = load_npz_arrays(root/task/'token_index.npz')
            for d, r in zip(m['depths'], m['outputs'][1:]):
                evidence = (r, a[f'mean{d}'].tolist(), a[f'scale{d}'].tolist())
                if d in token_refs and token_refs[d] != evidence:
                    raise ValueError('Cross-fit token target mismatch')
                token_refs[d] = evidence
    rows = []
    for name, a in arrays.items():
        old.validate_predictions(a, original['counts']['validation'])
        if not np.array_equal(a['identities'], anchor['identities']) or not np.array_equal(a['labels'], anchor['labels']):
            raise ValueError('Comparison population mismatch')
        rows.append(dict(model=name, selected_epoch=picks.get(name),
            input='ORACLE' if name == 'OFFLINE' or name in science.TEACHER_TASKS else 'single HLT',
            validation=old.evaluate_probabilities(a['labels'], a['probabilities']),
            original50k=old.evaluate_probabilities(a['labels'][ix], a['probabilities'][ix])))
    by = {r['model']:r for r in rows}
    for row in rows:
        for role in ('validation', 'original50k'):
            row[role+'_recovery'] = old.recovery(row[role], by['M0HLT'][role], by['OFFLINE'][role])
            row[role+'_delta_baseline'] = row[role]['macro_ovr_auc'] - by['K1_E40_PLAIN'][role]['macro_ovr_auc']
    return artifact('RESULTS', parents=dict(recovery=spec['content_hash'], **{t:r['content_hash'] for t,r in seen.items()}),
        rows=rows, pending=pending, independent_test=False, original50k_selection=False,
        shared_targets_retained=True, completed_original_fits_reused=list(spec['reused']),
        interrupted_original_attempts=spec.get('restarted', {}))

"""Retained native replacements followed by unchanged multi-head science."""
from pathlib import Path
import gc
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from hlt_classification.luka_attention import worker as logit, banks as logit_banks, contracts as logit_contracts
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from hlt_classification.luka_multiteacher_trial import worker as donor, banks, contracts as multi
from .contracts import artifact, validate, ARMS, REPLACEMENTS
from . import campaign, submission

TEACHER_FILES = ('selected.pt', 'training_report.json', 'validation_predictions.npz', 'costs.json',
                 'teacher_bank/manifest.json', 'teacher_bank/population.npz')


def teacher_node(spec, name):
    return next(n for n in spec['teacher_nodes'] if n['node_id'] == name)


def teacher_files(spec, name):
    root = Path(spec['root']); folder = root/('train_'+name)
    return dict(checkpoint=file_ref(folder/'selected.pt'), report=file_ref(folder/'training_report.json'),
        predictions=file_ref(folder/'validation_predictions.npz'), receipt=file_ref(root/('train_'+name+'_receipt.json')))


def predecessor(spec, name):
    if name == 'LOGIT_D066':
        return spec['survivors']['COARSE_U100']['files']
    if name == 'LOGIT_D033':
        return teacher_files(spec, 'LOGIT_D066')
    raise ValueError('Unregistered replacement teacher')


def bank_parents(files):
    return dict(checkpoint=files['checkpoint']['sha256'], report=files['report']['sha256'])


def context(spec, *, publish=False):
    """Resolved actual weights, separately hashed and bound to the approved plan."""
    teachers = dict(spec['survivors']); receipts = {}
    for name in REPLACEMENTS:
        receipt = completed(spec, 'train_'+name)
        if receipt is None:
            raise ValueError('Replacement teacher incomplete: '+name)
        receipts[name] = receipt['content_hash']
        node = teacher_node(spec, name)
        teachers[name] = multi.artifact('TEACHER', parents=dict(anchor=spec['anchor']['content_hash']),
            model_id=name, coordinate=node['coordinate'], node=node, prepared=spec['anchor']['evidence']['prepared'],
            role='train', spec=file_ref(Path(spec['root'])/'campaign_spec.json'), files=teacher_files(spec, name),
            selected_native_checkpoint_revalidated=True, origin='new_LOGIT_replacement',
            recovery_receipt=receipt['content_hash'],
            historical_checkpoint_descriptor=spec['archive']['recorded_checkpoints'][name])
    keys = ('root','anchor','counts','original50k_rows','memory_mb','nodes','recipes','reference','u100_reference')
    value = artifact('CONTEXT', parents=dict(spec=spec['content_hash'], **receipts),
                     teachers=teachers, **{k:spec[k] for k in keys})
    path = Path(spec['root'])/'execution_context.json'
    if publish:
        write_immutable_json(path, value)
    elif load_json(path) != value:
        raise ValueError('Resolved teacher registry changed')
    return value


def output_names(spec, task):
    if task.startswith('train_'):
        return [task+'/'+n for n in TEACHER_FILES]
    if task == 'export_targets':
        return ['execution_context.json', *donor.output_names(spec, task)]
    return donor.output_names(spec, task)


def validate_teacher(spec, name):
    root = Path(spec['root'])/('train_'+name); node = teacher_node(spec, name)
    files = predecessor(spec, name)
    for ref in files.values(): validate_file_ref(ref)
    manifest = load_json(root/'teacher_bank/manifest.json')
    logit_contracts.validate(manifest, 'BANK', bank_parents(files))
    if (manifest['teacher'] != node['teacher'] or manifest['coordinate'] != ('U100' if name == 'LOGIT_D066' else 'D066')
            or manifest['attention'] is not False or manifest['shards'] != [] or manifest['surface'] is not None
            or manifest['temperature'] != 2. or manifest['alignment']['role'] != 'train'
            or manifest['alignment']['prepared'] != spec['anchor']['evidence']['prepared']
            or manifest['alignment']['rows'] != spec['counts']['train']
            or manifest['population']['path'] != 'population.npz'):
        raise ValueError('Replacement input bank differs')
    validate_file_ref(manifest['population'], root=root/'teacher_bank')
    report = load_json(root/'training_report.json')
    logit.report_valid(report, node, manifest['content_hash'], spec['anchor']['evidence']['prepared'],
                       spec['counts']['validation'])
    prediction = load_npz_arrays(root/'validation_predictions.npz')
    donor.validate_predictions(prediction, spec['counts']['validation'])
    baseline = load_npz_arrays(validate_file_ref(spec['anchor']['evidence']['models']['M0HLT']['predictions']))
    if (not np.array_equal(prediction['identities'], baseline['identities'])
            or not np.array_equal(prediction['labels'], baseline['labels'])
            or evaluate_probabilities(prediction['labels'], prediction['probabilities']) != report['validation']):
        raise ValueError('Replacement validation population/metrics differ')
    return report, prediction


def completed(spec, task, *, seen=None):
    seen = {} if seen is None else seen
    if task in seen: return seen[task]
    registered = next((t for t in spec['tasks'] if t['task_id'] == task), None)
    if registered is None: raise ValueError('Unregistered recovery task')
    root = Path(spec['root']); path = root/(task+'_receipt.json')
    if not path.is_file(): return None
    record = load_json(path); validate(record, 'TASK', dict(spec=spec['content_hash']))
    deps = {}
    for dep in registered['dependencies']:
        r = completed(spec, dep, seen=seen)
        if r is None: raise ValueError('Missing completed dependency')
        deps[dep] = r['content_hash']
    if (record['task_id'] != task or record['dependencies'] != deps
            or record['source_commit'] != spec['source_commit'] or record['preflight_performed'] is not False
            or record['job_id'] != submission.accepted_jobs(spec).get(task)
            or [r['path'] for r in record['outputs']] != output_names(spec, task)):
        raise ValueError('Completed task/job/lineage/output set differs')
    for ref in record['outputs']: validate_file_ref(ref, root=root)
    if task.startswith('train_'):
        if record['execution_context'] is not None: raise ValueError('Teacher cannot consume student context')
        validate_teacher(spec, task.removeprefix('train_'))
    else:
        ctx = context(spec)
        if record['execution_context'] != ctx['content_hash']:
            raise ValueError('Completed task uses different resolved teachers')
        bundle = load_json(root/'targets/bundle.json')
        multi.validate(bundle, 'BUNDLE', dict(spec=ctx['content_hash'], prepared=spec['anchor']['evidence']['prepared']))
        if bundle['role'] != 'train' or bundle['rows'] != spec['counts']['train'] or set(bundle['records']) != set(multi.TEACHERS):
            raise ValueError('Target bundle registration differs')
        for name, ref in bundle['records'].items():
            if ref != file_ref(root/'targets'/name/'manifest.json'):
                raise ValueError('Target bank binding changed')
        if task in ARMS: donor.validate_fit(ctx, task, bundle['content_hash'])
    seen[task] = record
    return record


def train_teacher(spec, name, device):
    root = Path(spec['root'])/('train_'+name); root.mkdir(exist_ok=False)
    node = teacher_node(spec, name); files = predecessor(spec, name)
    meta = donor.metadata(spec)
    co = 'U100' if name == 'LOGIT_D066' else 'D066'
    teacher_cache = banks.data(spec, 'train', co, meta)
    teacher = donor.load_native(files['checkpoint'], device)
    # T2 class targets only. No historical attention payload is reconstructed.
    logit_banks.export(root/'teacher_bank', teacher, teacher_cache, parents=bank_parents(files),
        teacher=node['teacher'], attention=False, device=device)
    del teacher, teacher_cache; gc.collect()
    if str(device).startswith('cuda'): torch.cuda.empty_cache()
    tr = banks.data(spec, 'train', node['coordinate'], meta)
    val = banks.data(spec, 'validation', node['coordinate'], meta)
    bank = logit_banks.Bank(root/'teacher_bank', tr, parents=bank_parents(files),
                           teacher=node['teacher'], require_attention=False)
    # Original unmodified LOGIT trainer/readout path, seeds, C25/P75 T2 and30 epochs.
    logit.fit(root, spec, node, tr, val, bank, device=device, technical=False)


def run(spec, task, *, device='cuda'):
    campaign.check(spec)
    registered = next((t for t in spec['tasks'] if t['task_id'] == task), None)
    if registered is None: raise PermissionError('No unregistered recovery task')
    job = logit.execution(spec, spec['resources'])
    if job != submission.accepted_jobs(spec).get(task):
        raise PermissionError('Not the exact submitted allocation')
    deps = {}
    for dep in registered['dependencies']:
        record = completed(spec, dep)
        if record is None: raise ValueError('Dependency incomplete: '+dep)
        submission.successful(spec, dep, record['job_id'])
        deps[dep] = record['content_hash']
    root = Path(spec['root']); attempt = root/(task+'_attempt'); attempt.mkdir(exist_ok=False)
    campaign.storage(spec); log = PhaseLog(attempt, lambda:memory_snapshot(torch)); ctx = None
    if task.startswith('train_'):
        with log.phase('replacement_LOGIT_teacher_and_compact_T2_targets'):
            train_teacher(spec, task.removeprefix('train_'), device)
    elif task == 'export_targets':
        with log.phase('bind_new_teachers_and_export_actual_TRAIN_embeddings'):
            ctx = context(spec, publish=True)
            donor.export_targets(ctx, device)
    else:
        ctx = context(spec)
        with log.phase('load_shared_targets_and_HLT'):
            meta = donor.metadata(ctx)
            tr = banks.data(ctx, 'train', 'D000', meta)
            val = banks.data(ctx, 'validation', 'D000', meta)
            bank = banks.load(ctx, tr)
        with log.phase('unchanged_fresh_thirty_epoch_student'):
            donor.fit(root/task, ctx, task, tr, val, bank, device)
    campaign.storage(spec); campaign.check(spec)
    if logit.execution(spec, spec['resources']) != job:
        raise PermissionError('Allocation changed')
    for dep, digest in deps.items():
        if completed(spec, dep)['content_hash'] != digest:
            raise ValueError('Dependency changed during execution')
    record = artifact('TASK', parents=dict(spec=spec['content_hash']), task_id=task, job_id=job,
        source_commit=spec['source_commit'], dependencies=deps, preflight_performed=False,
        execution_context=None if ctx is None else ctx['content_hash'],
        outputs=[file_ref(root/p, root=root) for p in output_names(spec, task)])
    write_immutable_json(root/(task+'_receipt.json'), record)
    return completed(spec, task)


def results(spec):
    """Saved-only partial teacher/student results; never load a model or particles."""
    campaign.validate_spec(spec)
    root = Path(spec['root']); seen = {}; pending = []; rows = []
    baseline = load_npz_arrays(validate_file_ref(spec['anchor']['evidence']['models']['M0HLT']['predictions']))
    meta = donor.metadata(spec)
    ix = donor.alljets_data.comparison_indices(meta[2], meta[3], baseline['identities'], baseline['labels'])
    if len(ix) != spec['original50k_rows']: raise ValueError('Report-only subset differs')
    inputs = [(n, load_npz_arrays(validate_file_ref(ref['predictions'])), None, 'historical')
              for n, ref in donor.reference_files(spec).items()]
    for name in REPLACEMENTS:
        if completed(spec, 'train_'+name, seen=seen) is None:
            pending.append('train_'+name)
        else:
            report, a = validate_teacher(spec, name)
            inputs.append(('NEW_'+name, a, report['selected_pass'], 'replacement_teacher'))
    exported = completed(spec, 'export_targets', seen=seen)
    ctx = context(spec) if exported else None
    bundle = load_json(root/'targets/bundle.json') if exported else None
    if not exported: pending.append('export_targets')
    for arm in ARMS:
        if completed(spec, arm, seen=seen) is None:
            pending.append(arm)
        else:
            report, a = donor.validate_fit(ctx, arm, bundle['content_hash'])
            inputs.append((arm, a, report['selected_pass'], 'new_student'))
    for name, a, epoch, origin in inputs:
        donor.validate_predictions(a, spec['counts']['validation'])
        if not np.array_equal(a['identities'], baseline['identities']) or not np.array_equal(a['labels'], baseline['labels']):
            raise ValueError('Result populations differ')
        row = dict(model=name, selected_epoch=epoch, origin=origin)
        for role, ids in (('validation', np.arange(len(a['labels']))), ('original50k', ix)):
            y, p = a['labels'][ids], a['probabilities'][ids]
            row[role] = evaluate_probabilities(y, p)
            row[role+'_r30'] = donor.rejection(y, p, .3); row[role+'_r50'] = donor.rejection(y, p, .5)
        rows.append(row)
    by = {r['model']:r for r in rows}
    for row in rows:
        for role in ('validation', 'original50k'):
            row[role+'_recovery'] = recovery(row[role], by['M0HLT'][role], by['OFFLINE'][role])
            row[role+'_delta_vs_multi_w5'] = (row[role]['macro_ovr_auc']-by['multi_w5'][role]['macro_ovr_auc']
                                             if 'multi_w5' in by else None)
    return artifact('RESULTS', parents=dict(spec=spec['content_hash']), rows=rows, pending=pending,
        single_seed=True, adaptive_validation=True, original50k_selection=False,
        replacement_teachers=list(REPLACEMENTS), new_preflight_performed=False)


def print_results(result):
    print('NEW LOGIT_D066/D033 replacements: not restored historical checkpoints.')
    print('Teacher fits use logit KD; the FIVE HLT student arms have NO logit KD.')
    donor.print_results(result)

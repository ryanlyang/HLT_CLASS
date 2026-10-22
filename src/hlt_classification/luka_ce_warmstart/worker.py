"""Measured setup, isolated science stages and authenticated compact reporting."""
from pathlib import Path
import gc
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays, write_immutable_json,
    atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer, installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.runner import predict, _forward
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from hlt_classification.jetclass2_delphes.acceptance import installed_parity
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_fullsim import regular_cache, alljets_data
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from hlt_classification.luka_fullsim.science_campaign import preflight_accounting
from hlt_classification.luka_attention.worker import data, execution
from hlt_classification.luka_embedding_weight.worker import storage, validate_predictions
from hlt_classification.luka_embedding_aux.model import new_model
from hlt_classification.luka_embedding_aux.preflight import parity
from .contracts import artifact, validate, TASKS, ARMS
from . import campaign, source, training, targets, submission

REFS = ('M0HLT', 'OFFLINE', 'OFFLINE_EMBED_W5', 'COARSE_U100')
SETUP_FILES = ['targets/manifest.json', 'targets/targets.npz', 'initial/selected.pt',
    'setup/measurement.json', 'setup/teacher_readback.json', 'setup/initial_readback.json', 'setup/checks.json',
    'references/original50k.npz'] + ['references/'+name+'.npz' for name in REFS] + [
    'setup/probe_'+arm+'/'+name for arm in ARMS
    for name in ('selected.pt', 'training_report.json', 'validation_predictions.npz')]


def output_names(task):
    return SETUP_FILES if task == 'setup' else [
        task+'/fit/'+name for name in ('selected.pt', 'training_report.json', 'validation_predictions.npz')]


def snapshot_references(spec, val, meta):
    folder = Path(spec['root'])/'references'; folder.mkdir(exist_ok=False)
    ix = alljets_data.comparison_indices(meta[2], meta[3], val.identities, val.labels)
    atomic_publish_bytes(folder/'original50k.npz', deterministic_npz_bytes(dict(indices=ix)))
    refs = {n: spec['anchor']['evidence']['models'][n]['predictions'] for n in ('M0HLT', 'OFFLINE')}
    refs.update(OFFLINE_EMBED_W5=spec['reference']['files']['predictions'], COARSE_U100=spec['teacher']['files']['predictions'])
    for name, ref in refs.items():
        validate_file_ref(ref); a = load_npz_arrays(ref['path']); validate_predictions(a, len(val))
        if not np.array_equal(a['identities'], val.identities) or not np.array_equal(a['labels'], val.labels):
            raise ValueError('Reference validation populations differ')
        atomic_publish_bytes(folder/(name+'.npz'), deterministic_npz_bytes(a))
        validate_file_ref(ref)


def comparison(spec, students):
    folder = Path(spec['root'])/'references'; ix = load_npz_arrays(folder/'original50k.npz')['indices']
    arrays = {n: load_npz_arrays(folder/(n+'.npz')) for n in REFS}
    reference = arrays['M0HLT']
    if (ix.dtype != np.int64 or ix.shape != (spec['original50k_rows'],) or len(np.unique(ix)) != len(ix)
            or np.any(ix < 0) or np.any(ix >= len(reference['labels']))):
        raise ValueError('Report-only subset differs')
    arrays.update({ARMS[arm]: a for arm, a in students.items()}); rows = []
    for name, a in arrays.items():
        validate_predictions(a, spec['counts']['validation'])
        if not np.array_equal(a['identities'], reference['identities']) or not np.array_equal(a['labels'], reference['labels']):
            raise ValueError('Comparison identities/labels differ')
        rows.append(dict(model=name, input='ORACLE' if name in ('OFFLINE', 'COARSE_U100') else 'single HLT',
            validation=evaluate_probabilities(a['labels'], a['probabilities']),
            original50k=evaluate_probabilities(a['labels'][ix], a['probabilities'][ix])))
    control = next((r for r in rows if r['model'] == ARMS['ce']), None)
    for row in rows:
        for role in ('validation', 'original50k'):
            row[role+'_recovery'] = recovery(row[role], rows[0][role], rows[1][role])
            row[role+'_auc_delta_vs_continued_CE'] = None if control is None else (
                row[role]['macro_ovr_auc']-control[role]['macro_ovr_auc'])
    return rows


def fit(directory, spec, train, val, bank, arm, device, *, technical=False):
    directory = Path(directory); directory.mkdir(exist_ok=False)
    initial_path = Path(spec['root'])/'initial/selected.pt'
    initial_ref = file_ref(initial_path)
    if any(initial_ref[k] != spec['initial_checkpoint'][k] for k in ('sha256', 'bytes')):
        raise ValueError('Local warm start differs from selected M0HLT')
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    model = new_model(spec['node'])
    report, state = training.train(model, train, val, bank, spec['node'], arm, device,
        initial_state=initial, initial_sha256=initial_ref['sha256'], technical=technical)
    bank_hash = None if bank is None else bank.manifest['content_hash']
    training.validate_report(report, spec, bank_hash, arm, technical=technical)
    validate_file_ref(initial_ref)
    save_state(directory/'selected.pt', state)
    restored = torch.load(directory/'selected.pt', map_location='cpu', weights_only=True)
    if state.keys() != restored.keys() or any(not torch.equal(state[k], restored[k]) for k in state):
        raise ValueError('Checkpoint state roundtrip differs')
    native = DelphesParticleTransformer().to(device).eval(); native.load_state_dict(restored, strict=True)
    expected = predict(model, val, device=device, batch_size=128)
    actual = predict(native, val, device=device, batch_size=128)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    if evaluate_probabilities(val.labels, actual) != report['selected']['validation']:
        raise ValueError('Native HLT-only readback differs')
    atomic_publish_bytes(directory/'validation_predictions.npz', deterministic_npz_bytes(dict(
        identities=val.identities, labels=val.labels, probabilities=actual)))
    report = artifact('FIT_REPORT', parents=dict(report=report['content_hash']), report=report,
        initial_checkpoint=initial_ref, checkpoint=file_ref(directory/'selected.pt'), checkpoint_readback=True)
    write_immutable_json(directory/'training_report.json', report)
    return report


def validate_fit(directory, spec, bank_hash, arm, *, technical=False):
    directory = Path(directory); record = load_json(directory/'training_report.json')
    report = record['report']; validate(record, 'FIT_REPORT', dict(report=report['content_hash']))
    training.validate_report(report, spec, bank_hash, arm, technical=technical)
    if (record['checkpoint'] != file_ref(directory/'selected.pt') or record['checkpoint_readback'] is not True
            or record['initial_checkpoint'] != file_ref(Path(spec['root'])/'initial/selected.pt')):
        raise ValueError('Checkpoint/initialization binding differs')
    a = load_npz_arrays(directory/'validation_predictions.npz'); validate_predictions(a, spec['counts']['validation'])
    if evaluate_probabilities(a['labels'], a['probabilities']) != report['selected']['validation']:
        raise ValueError('Saved native probabilities differ')
    return record


def stress(cache, bank, node, initial, arm, device):
    lengths = np.concatenate([np.diff(b.offsets) for b in cache.blocks])
    ix = np.repeat(int(np.argmax(lengths)), 128); raw = cache.batch(ix)
    model = new_model(node); training.initialize(model, initial, arm); model.to(device)
    backward = cache.role == 'train'; model.train(backward); model.capture = backward and arm == 'embed'
    optimizer = training.optimizer_for(model)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(backward):
            logits = _forward(model, raw, device, bf16=str(device).startswith('cuda'))
            if not torch.isfinite(logits).all(): raise ValueError('Nonfinite stress logits')
            if backward:
                target = torch.from_numpy(bank.targets[ix]).to(device) if arm == 'embed' else None
                loss, _, _ = training.objective(logits, torch.from_numpy(raw['labels']).to(device),
                    model.prediction, target, arm=arm)
                loss.backward()
                finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
                if finite and not torch.stack(finite).all(): raise ValueError('Nonfinite stress gradients')
                optimizer.step()
    if str(device).startswith('cuda'): torch.cuda.synchronize()
    return dict(role=cache.role, arm=arm, maximum=int(lengths.max()), batch=128, steps=3, backward=backward,
                predictor_enabled=backward and arm == 'embed')


def admission(spec, measurement):
    if measurement['environment'] != spec['anchor']['evidence']['environment']:
        raise ValueError('Original installed environment differs')
    for k in ('peak_rss_bytes', 'gpu_peak_bytes'):
        if type(measurement[k]) is not int or measurement[k] <= 0:
            raise ValueError('Missing memory high-water evidence')
    if (measurement['peak_rss_bytes'] >= .85*spec['memory_mb']*1024**2
            or measurement['gpu_peak_bytes'] >= .9*measurement['gpu']['total_memory_bytes']):
        raise MemoryError('Measured resource envelope exceeded')
    times = measurement['timings']
    expected = {'authentication', 'load_HLT', 'ce_train', 'ce_validation',
                'embed_train', 'embed_validation', 'readback'}
    if set(times) != expected or any(not math.isfinite(v) or v < 0 for v in times.values()):
        raise ValueError('Invalid timing measurement')
    projected = {}
    for stage, epochs in (('ce', 30), ('embed', 30)):
        if min(times[stage+'_train'], times[stage+'_validation']) <= 0:
            raise ValueError('Missing whole-population measurement')
        projected[stage] = max(30, math.ceil(1.75*(times['authentication']+times['load_HLT']+times['readback']+
            epochs*(times[stage+'_train']+times[stage+'_validation'])+300)/60))
        if projected[stage] > spec['minutes'][stage]:
            raise ValueError('Measured stage exceeds registered walltime: '+stage)
    return projected


def setup(spec, folder, log, meta, authentication, device):
    root = Path(spec['root'])
    with log.phase('authenticate_frozen_U100_and_export_TRAIN_targets'):
        teacher = DelphesParticleTransformer().to(device).eval()
        teacher.load_state_dict(torch.load(spec['teacher']['files']['checkpoint']['path'], map_location='cpu', weights_only=True), strict=True)
        train_teacher, val_teacher = (data(spec, role, 'U100', meta) for role in ('train', 'validation'))
        ref = spec['teacher']['files']['predictions']; validate_file_ref(ref)
        saved = load_npz_arrays(ref['path']); validate_predictions(saved, len(val_teacher))
        if not np.array_equal(saved['identities'], val_teacher.identities) or not np.array_equal(saved['labels'], val_teacher.labels):
            raise ValueError('U100 teacher readback population differs')
        actual = predict(teacher, val_teacher, device=device, batch_size=128)
        np.testing.assert_allclose(actual, saved['probabilities'], rtol=1e-5, atol=1e-6)
        write_immutable_json(folder/'teacher_readback.json', artifact('READBACK',
            parents=dict(teacher=spec['teacher']['content_hash']), passed=True, rows=len(val_teacher),
            max_error=float(np.max(np.abs(actual-saved['probabilities'])))))
        targets.export(root/'targets', teacher, train_teacher, teacher_record=spec['teacher'], device=device)
        del teacher, train_teacher, val_teacher
        gc.collect(); torch.cuda.empty_cache()
    with log.phase('load_HLT_and_compact_targets') as loaded:
        train, val = (data(spec, role, 'D000', meta) for role in ('train', 'validation'))
        bank = targets.Bank(root/'targets', train, teacher_record=spec['teacher'])
        snapshot_references(spec, val, meta)
    with log.phase('authenticate_and_copy_exact_CE_initialization'):
        ref = spec['initial_checkpoint']; validate_file_ref(ref)
        initial = torch.load(ref['path'], map_location='cpu', weights_only=True)
        model = new_model(spec['node']); training.initialize(model, initial, 'ce')
        model.to(device).eval()
        actual = predict(model, val, device=device, batch_size=128)
        saved = load_npz_arrays(root/'references/M0HLT.npz')
        np.testing.assert_allclose(actual, saved['probabilities'], rtol=1e-5, atol=1e-6)
        (root/'initial').mkdir(exist_ok=False)
        atomic_publish_bytes(root/'initial/selected.pt', Path(ref['path']).read_bytes())
        validate_file_ref(ref)
        local = file_ref(root/'initial/selected.pt')
        if any(local[k] != ref[k] for k in ('sha256', 'bytes')): raise ValueError('Initial checkpoint copy differs')
        write_immutable_json(folder/'initial_readback.json', artifact('READBACK',
            parents=dict(initial_checkpoint=ref['sha256']), passed=True, rows=len(val),
            max_error=float(np.max(np.abs(actual-saved['probabilities'])))))
        del model
    with log.phase('native_predictor_parity_and_maximum_support'):
        with parity_backend(device): native = installed_parity(train, device=device)
        parities = [parity(train.batch(np.arange(min(4, len(train)))), spec['node'], device, bf16=b) for b in (False, True)]
        stresses = [stress(cache, bank if arm == 'embed' else None, spec['node'], initial, arm, device)
                    for arm in ARMS for cache in (train, val)]
        maxima = {role: meta[1]['summaries'][role]['D000']['maximum'] for role in ('train', 'validation')}
        if any(s['maximum'] != maxima[s['role']] for s in stresses):
            raise ValueError('Stress support differs from prepared metadata')
        gc.collect(); torch.cuda.empty_cache()
        write_immutable_json(folder/'checks.json', artifact('CHECKS', native=native, parities=parities, stress=stresses, maxima=maxima))
    with log.phase('disposable_full_population_matched_continuations') as probe:
        probes = {arm: fit(folder/('probe_'+arm), spec, train, val, bank if arm == 'embed' else None,
                          arm, device, technical=True) for arm in ARMS}
    times = dict(authentication=authentication, load_HLT=loaded['seconds'])
    for arm, record in probes.items():
        row = record['report']['history'][0]
        times[arm+'_train'], times[arm+'_validation'] = row['train_seconds'], row['validation_seconds']
    times['readback'] = max(0., probe['seconds']-sum(v for k, v in times.items() if k.endswith(('_train', '_validation'))))
    memory = memory_snapshot(torch)
    measurement = dict(environment=installed_environment(), gpu=gpu_identity(), timings=times,
        peak_rss_bytes=memory['peak_rss_bytes'], gpu_peak_bytes=memory['gpu_peak_bytes'])
    measurement = artifact('MEASUREMENT', **measurement, projected_minutes=admission(spec, measurement))
    write_immutable_json(folder/'measurement.json', measurement)


def validate_task(spec, task, seen=None):
    seen = {} if seen is None else seen
    if task in seen: return seen[task]
    root = Path(spec['root']); row = load_json(root/task/'receipt.json')
    deps = {name: validate_task(spec, name, seen)['content_hash'] for name in TASKS[task]}
    validate(row, 'TASK', dict(spec=spec['content_hash'], **deps))
    if row['task'] != task or [r['path'] for r in row['outputs']] != output_names(task):
        raise ValueError('Task output inventory differs')
    for ref in row['outputs']: validate_file_ref(ref, root=root)
    manifest = targets.validate_manifest(root/'targets', spec['teacher'], spec['anchor']['evidence']['prepared'], spec['counts'])
    local = file_ref(root/'initial/selected.pt')
    if any(local[k] != spec['initial_checkpoint'][k] for k in ('sha256', 'bytes')):
        raise ValueError('Warm start differs from authenticated M0HLT')
    if task == 'setup':
        measured = load_json(root/'setup/measurement.json'); validate(measured, 'MEASUREMENT')
        if measured['projected_minutes'] != admission(spec, measured): raise ValueError('Measured projection differs')
        checks = load_json(root/'setup/checks.json'); validate(checks, 'CHECKS')
        for name, parent in (('teacher', dict(teacher=spec['teacher']['content_hash'])),
                             ('initial', dict(initial_checkpoint=spec['initial_checkpoint']['sha256']))):
            proof = load_json(root/'setup'/(name+'_readback.json')); validate(proof, 'READBACK', parent)
            if proof['passed'] is not True or proof['rows'] != spec['counts']['validation'] or not 0 <= proof['max_error'] <= 1.1e-5:
                raise ValueError('Saved probability readback differs')
        if (checks['native']['passed'] is not True or [p['bf16'] for p in checks['parities']] != [False, True]
                or any(p[k] is not True for p in checks['parities'] for k in
                       ('native_forward', 'zero_control_update', 'auxiliary_backbone_gradient', 'auxiliary_head_gradient'))
                or [(s['arm'], s['role'], s['batch'], s['steps'], s['backward'], s['predictor_enabled']) for s in checks['stress']] !=
                   [(a, r, 128, 3, r == 'train', a == 'embed' and r == 'train') for a in ARMS for r in ('train', 'validation')]
                or any(s['maximum'] != checks['maxima'][s['role']] for s in checks['stress'])):
            raise ValueError('Acceptance evidence differs')
        for arm in ARMS:
            validate_fit(root/'setup'/('probe_'+arm), spec, manifest['content_hash'], arm, technical=True)
    else:
        validate_fit(root/task/'fit', spec, manifest['content_hash'], task)
    seen[task] = row
    return row


def run(spec, task):
    if task not in TASKS:
        raise ValueError('Unknown task')
    started = time.monotonic(); campaign.check(spec); authentication = time.monotonic()-started
    jobs = submission.accepted_jobs(spec, partial=task == 'setup')
    measured = None; deps = {}
    for dep in TASKS[task]:
        record = validate_task(spec, dep)
        if record['job_id'] != jobs[dep]:
            raise ValueError('Dependency job differs from exact journal')
        preflight_accounting(jobs[dep]); deps[dep] = record['content_hash']
    if task != 'setup':
        measured = load_json(Path(spec['root'])/'setup/measurement.json')
    job = execution(spec, measured)
    if jobs.get(task) != job or installed_environment() != spec['anchor']['evidence']['environment']:
        raise ValueError('Actual job/environment differs')
    storage(spec)
    root = Path(spec['root']); folder = root/task; folder.mkdir(exist_ok=False)
    log = PhaseLog(folder, lambda: memory_snapshot(torch)); torch.cuda.reset_peak_memory_stats()
    meta = regular_cache.metadata(spec['anchor']['evidence']['prepared_root'])
    if task == 'setup':
        setup(spec, folder, log, meta, authentication, 'cuda')
    else:
        with log.phase('load_authenticated_HLT_and_paired_targets'):
            train, val = (data(spec, role, 'D000', meta) for role in ('train', 'validation'))
            bank = targets.Bank(root/'targets', train, teacher_record=spec['teacher']) if task == 'embed' else None
        with log.phase(task+'_scientific_epochs'):
            fit(folder/'fit', spec, train, val, bank, task, 'cuda')
    storage(spec)
    if source.current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']:
        raise ValueError('Source changed during task')
    record = artifact('TASK', parents=dict(spec=spec['content_hash'], **deps), task=task, job_id=job,
        outputs=[file_ref(root/name, root=root) for name in output_names(task)])
    write_immutable_json(folder/'receipt.json', record); validate_task(spec, task)
    return record


def results(spec):
    campaign.validate_spec(spec, payloads=False); root = Path(spec['root'])
    setup_record = validate_task(spec, 'setup'); parents = dict(spec=spec['content_hash'], setup=setup_record['content_hash'])
    students = {}; pending = []
    for arm in ARMS:
        if not (root/arm/'receipt.json').is_file():
            pending.append(ARMS[arm]); continue
        record = validate_task(spec, arm); parents[arm] = record['content_hash']
        students[arm] = load_npz_arrays(root/arm/'fit/validation_predictions.npz')
    return artifact('RESULTS', parents=parents, counts=spec['counts'], comparison=spec['comparison'],
                    rows=comparison(spec, students), pending=pending)


def print_results(result):
    print('Matched selected-CE warm starts: CE vs CE+5*U100 embedding, 30 epochs, 0.1x LR.')
    print('Exploratory development; no independent test. Original50k is report-only.')
    for role in ('validation', 'original50k'):
        print('\n'+role)
        print(f'{"model":25} {"accuracy":>10} {"AUC":>10} {"recovery%":>10} {"AUC delta vs continued CE":>26}')
        for row in result['rows']:
            m = row[role]; r = row[role+'_recovery']['macro_ovr_auc']; d = row[role+'_auc_delta_vs_continued_CE']
            rec = 'n/a' if r is None else f'{r:.1f}'; delta = 'pending' if d is None else f'{d:+.8f}'
            print(f'{row["model"]:25} {m["accuracy"]:10.6f} {m["macro_ovr_auc"]:10.6f} {rec:>10} {delta:>26}')
    for name in result['pending']: print(name+': PENDING/INCOMPLETE')

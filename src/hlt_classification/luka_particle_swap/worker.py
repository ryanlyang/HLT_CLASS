"""Sequential production ladder with rolling scratch targets and durable results."""
import gc
import time
from pathlib import Path
import numpy as np
import torch

from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    atomic_publish_bytes, deterministic_npz_bytes, write_immutable_json)
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer, installed_environment
from hlt_classification.jetclass2_delphes.runner import predict, _forward, selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.luka_fullsim import regular_cache, alljets_data
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from hlt_classification.luka_attention import banks
from hlt_classification.luka_attention.model import new_model
from hlt_classification.luka_attention.worker import execution
from . import data
from .banks import ResidentBank
from . import training, retention
from .contracts import artifact, validate, recipe


def original_indices(gate, cache):
    meta = regular_cache.metadata(gate['anchor']['evidence']['prepared_root'])
    return alljets_data.comparison_indices(meta[2], meta[3], cache.identities, cache.labels)


def report_valid(report, node, bank_hash, prepared, rows, technical):
    validate(report, 'TRAINING_REPORT', dict(recipe=recipe(node['attention_coefficient'])['content_hash'],
        prepared=prepared, bank=bank_hash))
    history = report['validation_history']
    epochs = 1 if technical else 30
    if (report['node'] != node or report['scientific_fit'] is not (not technical)
            or report['passes'] != epochs or len(history) != epochs
            or [r['pass_number'] for r in history] != list(range(1, epochs+1))
            or report['selected_weights_restored'] is not True
            or any(r['validation']['rows'] != rows for r in history)):
        raise ValueError('Training report contract differs')
    best = max(history, key=lambda r: selection_key(r['validation'], r['update']))
    if report['selected_pass'] != best['pass_number'] or report['validation'] != best['validation']:
        raise ValueError('Wrong validation-selected checkpoint')


def fit(directory, checkpoint, gate, node, tr, val, bank, device, technical, original_mask):
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    model = new_model(node)
    report, state = training.train(model, tr, val, bank=bank, node=node, device=device, technical=technical)
    report_valid(report, node, bank.manifest['content_hash'], tr.foundation_sha256, len(val), technical)
    save_state(checkpoint, state)
    restored = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if restored.keys() != state.keys() or any(not torch.equal(restored[k], state[k]) for k in state):
        raise ValueError('Checkpoint bytes failed readback')
    model.load_state_dict(restored, strict=True)
    native = DelphesParticleTransformer().to(device)
    native.load_state_dict(restored, strict=True)
    native.eval(); model.eval(); model.capture_attention = False
    raw = val.batch(np.arange(min(4, len(val))))
    with torch.inference_mode():
        torch.testing.assert_close(_forward(native, raw, device, bf16=str(device).startswith('cuda')),
            _forward(model, raw, device, bf16=str(device).startswith('cuda')), rtol=0, atol=0)
    del native, restored, state
    p = predict(model, val, device=device, batch_size=128)
    if evaluate_probabilities(val.labels, p) != report['validation']:
        raise ValueError('Selected prediction/report mismatch')
    original = np.array(original_mask, dtype=bool, copy=True)
    if original.shape != (len(val),) or not original.any():
        raise ValueError('Original-validation membership differs')
    arrays = dict(identities=val.identities, labels=val.labels, probabilities=p, original_validation=original)
    atomic_publish_bytes(directory/'validation_predictions.npz', deterministic_npz_bytes(arrays))
    write_immutable_json(directory/'training_report.json', report)
    return model, report


def validate_step(directory, gate, node, technical):
    directory = Path(directory)
    step = load_json(directory/'receipt.json')
    validate(step, 'STEP', dict(gate=gate['content_hash']))
    if (step['node'] != node or step['technical'] is not technical
            or step['native_checkpoint_readback'] is not True):
        raise ValueError('Step identity differs')
    for field in ('report', 'predictions', 'bank_manifest'):
        validate_file_ref(step[field], root=directory)
    if (step['input_audit']['path'] != 'input_audit.json'
            or step['validation_population']['path'] != 'validation_population.npz'):
        raise ValueError('Swap evidence path differs')
    validate_file_ref(step['input_audit'], root=directory.parent)
    validate_file_ref(step['validation_population'], root=directory.parent)
    audit = load_json(directory.parent/'input_audit.json')
    validate(audit, 'INPUT_AUDIT', dict(inputs=gate['input_identity']['content_hash']))
    for role in ('train','validation'):
        alignment = step[role+'_alignment']
        if (alignment['prepared'] != gate['input_identity']['content_hash']
                or alignment['role'] != role or alignment['rows'] != gate['counts'][role]):
            raise ValueError('Swap step input alignment differs')
    bank = load_json(directory/step['bank_manifest']['path'])
    from hlt_classification.luka_attention.contracts import validate as old_validate, SURFACE
    old_validate(bank, 'BANK', step['teacher_parents'])
    attention = bool(node['attention_coefficient'])
    if (bank['teacher'] != node['teacher'] or bank['attention'] is not attention
            or bank['surface'] != (SURFACE if attention else None) or bank['temperature'] != 2.
            or bank['alignment'] != step['train_alignment']
            or bank['coordinate'] != {'D066':'U100', 'D033':'D066', 'D000':'D033'}[node['coordinate']]):
        raise ValueError('Target lineage/surface differs')
    report = load_json(directory/step['report']['path'])
    report_valid(report, node, bank['content_hash'], gate['input_identity']['content_hash'],
                 gate['counts']['validation'], technical)
    p = load_npz_arrays(directory/step['predictions']['path'])
    if (set(p) != {'identities', 'labels', 'probabilities', 'original_validation'}
            or p['identities'].dtype != np.uint8 or p['identities'].shape != (len(p['labels']), 32)
            or len(np.unique(p['identities'], axis=0)) != len(p['labels'])
            or p['labels'].dtype != np.int64 or p['original_validation'].dtype != np.bool_
            or p['original_validation'].shape != p['labels'].shape
            or not p['original_validation'].any()
            or evaluate_probabilities(p['labels'], p['probabilities']) != report['validation']):
        raise ValueError('Saved validation predictions differ')
    expected = evaluate_probabilities(p['labels'][p['original_validation']], p['probabilities'][p['original_validation']])
    if expected != step['original_validation']:
        raise ValueError('Original50k diagnostics differ')
    # Bind the saved full population, order, labels and subset to the canonical
    # views. The compact per-role population is retained independently of caches.
    canonical = load_npz_arrays(directory.parent/'validation_population.npz')
    for field in ('identities','labels','original_validation'):
        if not np.array_equal(p[field], canonical[field]):
            raise ValueError('Validation output population differs from swap inputs')
    return step


def validate_retired_rung(directory, gate, node, step):
    """Retirement records bind exactly this payload and its declared last consumer."""
    directory = Path(directory)
    for name in ('retired_teacher','retired_checkpoint'):
        retention.validate_retirement(directory/name, gate['content_hash'])
    target = load_json(directory/'retired_teacher/intent.json')
    checkpoint = load_json(directory/'retired_checkpoint/intent.json')
    manifest = load_json(directory/'teacher_bank.json')
    prefix = 'bank_'+node['teacher']+'/'
    expected = [dict(r, path=prefix+r['path']) for r in [manifest['population']]+[s['payload'] for s in manifest['shards']]]
    expected.append(dict(file_ref(directory/'teacher_bank.json'), path=prefix+'manifest.json'))
    if (target['files'] != expected or target['consumer'] != file_ref(directory/'receipt.json')
            or checkpoint['files'] != [step['checkpoint']] or target['base'] != checkpoint['base']):
        raise ValueError('Retirement does not match this rung inventory/consumer')
    consumer = directory/'receipt.json'
    if node['coordinate'] != 'D000':
        consumer = directory/'exported_bank.json'
        exported = load_json(consumer)
        from hlt_classification.luka_attention.contracts import validate as old_validate, SURFACE
        old_validate(exported, 'BANK', dict(checkpoint=step['checkpoint']['sha256'], report=step['report']['sha256']))
        if (exported['teacher'] != node['node_id'] or exported['alignment'] != step['train_alignment']
                or exported['coordinate'] != node['coordinate']
                or exported['surface'] != (SURFACE if node['attention_coefficient'] else None)
                or exported['attention'] is not bool(node['attention_coefficient']) or exported['temperature'] != 2.):
            raise ValueError('Checkpoint retired before its registered export')
    if checkpoint['consumer'] != file_ref(consumer):
        raise ValueError('Wrong last checkpoint consumer')


def close_bank(bank):
    for array in bank._lru.values():
        array._mmap.close()
    bank._lru.clear()


def export_bank(root, durable, model, tr, parents, teacher, device, attention):
    manifest = banks.export(root, model, tr, parents=parents, teacher=teacher, attention=attention, device=device)
    checked = banks.Bank(root, tr, parents=parents, teacher=teacher, require_attention=attention)
    close_bank(checked)
    write_immutable_json(durable, manifest)
    return manifest


def run_chain(gate, output, scratch_root, *, device='cuda', technical=False, stress_fn=None, log=None, inputs=None):
    """Same code for technical and science; only registered epoch count differs."""
    root, scratch_root = Path(output), Path(scratch_root)
    root.mkdir(exist_ok=False)
    log = log or PhaseLog(root, lambda: memory_snapshot(torch))
    with log.phase('construct_swap_views_once'):
        inputs = data.build(gate) if inputs is None else inputs
    write_immutable_json(root/'input_audit.json', inputs['audit'])
    val_pop = data.get(inputs, 'validation', 'D000')
    atomic_publish_bytes(root/'validation_population.npz', deterministic_npz_bytes(dict(
        identities=val_pop.identities, labels=val_pop.labels, original_validation=inputs['original_validation'])))
    population_ref = file_ref(root/'validation_population.npz', root=root)
    audit_ref = file_ref(root/'input_audit.json', root=root)
    if not technical:
        measured = load_json(Path(gate['root'])/'preflight/receipt.json')
        if measured['facts']['input_audit'] != inputs['audit']:
            raise ValueError('Science swap inputs/schedule differ from admitted preflight')
    attention = bool(gate['coefficient'])
    measurements, stresses = [], []
    with log.phase('anchor_export_and_storage'):
        started = time.monotonic()
        tr = data.get(inputs, 'train', 'U100')
        storage = retention.budget(banks.counts(tr), root, scratch_root, gate['available_quota_gib'])
        train_alignment = banks.alignment(tr)
        refs = gate['anchor']['evidence']['models']['COARSE_U100']
        validate_file_ref(refs['checkpoint']); validate_file_ref(refs['report'])
        teacher_parents = dict(checkpoint=refs['checkpoint']['sha256'], report=refs['report']['sha256'])
        model = new_model(dict(gate['nodes'][0], coordinate='U100'))
        model.load_state_dict(torch.load(refs['checkpoint']['path'], map_location='cpu', weights_only=True), strict=True)
        teacher = 'ANCHOR'
        bank_root = scratch_root/('bank_'+teacher)
        manifest = export_bank(bank_root, root/'anchor_bank.json', model, tr, teacher_parents, teacher, device, attention)
        anchor_seconds = time.monotonic()-started
        del model, tr
    validation_alignment = None
    for index, node in enumerate(gate['nodes']):
        gc.collect()
        if str(device).startswith('cuda'): torch.cuda.empty_cache()
        with log.phase('rung_'+node['node_id']):
            started = time.monotonic()
            tr, val = data.get(inputs, 'train', node['coordinate']), data.get(inputs, 'validation', node['coordinate'])
            if banks.alignment(tr) != train_alignment:
                raise ValueError('TRAIN persistent-slot alignment differs')
            if validation_alignment is None: validation_alignment = banks.alignment(val)
            if banks.alignment(val) != validation_alignment:
                raise ValueError('Validation persistent-slot alignment differs')
            bank = ResidentBank(bank_root, tr, parents=teacher_parents, teacher=node['teacher'],
                require_attention=attention, max_bytes=int(gate['memory_mb']*1024**2*.15))
            resident_target_bytes = bank.loaded_bytes
            load_seconds = time.monotonic()-started
            if technical and stress_fn:
                stresses.extend([stress_fn(tr, node, device, backward=True), stress_fn(val, node, device, backward=False)])
            directory = root/node['node_id']
            checkpoint = scratch_root/(node['node_id']+'.pt')
            started = time.monotonic()
            model, report = fit(directory, checkpoint, gate, node, tr, val, bank, device, technical, inputs['original_validation'])
            fit_seconds = time.monotonic()-started
            write_immutable_json(directory/'teacher_bank.json', manifest)
            p = load_npz_arrays(directory/'validation_predictions.npz')
            subset = p['original_validation']
            step = artifact('STEP', parents=dict(gate=gate['content_hash']), node=node, technical=technical,
                train_alignment=train_alignment, validation_alignment=validation_alignment,
                teacher_parents=teacher_parents, checkpoint=file_ref(checkpoint, root=scratch_root),
                report=file_ref(directory/'training_report.json', root=directory),
                predictions=file_ref(directory/'validation_predictions.npz', root=directory),
                bank_manifest=file_ref(directory/'teacher_bank.json', root=directory),
                input_audit=audit_ref, validation_population=population_ref,
                original_validation=evaluate_probabilities(p['labels'][subset], p['probabilities'][subset]),
                native_checkpoint_readback=True)
            write_immutable_json(directory/'receipt.json', step)
            validate_step(directory, gate, node, technical)
            close_bank(bank); del bank
            # Last consumer is durable and authenticated BEFORE old targets disappear.
            target_refs = [dict(r, path=bank_root.name+'/'+r['path']) for r in retention.bank_refs(bank_root, manifest)]
            retention.retire(scratch_root, target_refs, owner=gate['content_hash'],
                consumer=file_ref(directory/'receipt.json'), output=directory/'retired_teacher')
            started = time.monotonic()
            consumer = directory/'receipt.json'
            if index < len(gate['nodes'])-1:
                teacher, teacher_parents = node['node_id'], dict(checkpoint=step['checkpoint']['sha256'], report=step['report']['sha256'])
                bank_root = scratch_root/('bank_'+teacher)
                manifest = export_bank(bank_root, directory/'exported_bank.json', model, tr, teacher_parents, teacher, device, attention)
                consumer = directory/'exported_bank.json'
            export_seconds = time.monotonic()-started
            retention.retire(scratch_root, [step['checkpoint']], owner=gate['content_hash'],
                consumer=file_ref(consumer), output=directory/'retired_checkpoint')
            epoch_seconds = sum(report['validation_history'][0][k] for k in ('train_seconds','validation_seconds'))
            measurements.append(dict(node_id=node['node_id'], load_seconds=load_seconds,
                epoch_seconds=epoch_seconds, readback_seconds=max(0., fit_seconds-epoch_seconds) if technical else 0.,
                export_seconds=export_seconds, fit_seconds=fit_seconds, resident_target_bytes=resident_target_bytes))
            del model, tr, val
    return dict(measurements=measurements, stress=stresses, storage=storage,
        input_audit=inputs['audit'], construction_seconds=inputs['construction_seconds'],
        anchor_seconds=anchor_seconds, train_alignment=train_alignment, validation_alignment=validation_alignment)


def result_rows(gate, root, *, technical=False, partial=False):
    root = Path(root)
    references = load_json(Path(gate['root'])/'references.json')
    validate(references, 'REFERENCES', dict(anchor=gate['anchor']['content_hash']))
    rows = list(references['rows'])
    previous = None
    gap = False
    for node in gate['nodes']:
        directory = root/node['node_id']
        if not (directory/'receipt.json').is_file():
            if not partial: raise ValueError('Missing registered scientific rung')
            gap = True
            rows.append(dict(node_id=node['node_id'], validation=None, original_validation=None))
            continue
        if gap:
            raise ValueError('Completed descendant has a missing immediate parent')
        step = validate_step(directory, gate, node, technical)
        if previous is None:
            refs = gate['anchor']['evidence']['models']['COARSE_U100']
            expected = dict(checkpoint=refs['checkpoint']['sha256'], report=refs['report']['sha256'])
        else:
            expected = dict(checkpoint=previous['checkpoint']['sha256'], report=previous['report']['sha256'])
        if step['teacher_parents'] != expected:
            raise ValueError('Rung does not descend from its selected immediate parent')
        # A partial live display can precede retirement; completion cannot.
        if not partial or (directory/'retired_checkpoint/receipt.json').is_file():
            validate_retired_rung(directory, gate, node, step)
        report = load_json(directory/'training_report.json')
        rows.append(dict(node_id=node['node_id'], origin='whole_particle_swap_ladder',
            validation=report['validation'], original_validation=step['original_validation'],
            selected_pass=report['selected_pass'], checkpoint_retired=(directory/'retired_checkpoint/receipt.json').is_file()))
        previous = step
    by_name = {r['node_id']:r for r in rows}
    for row in rows:
        row['recovery'] = {field: None if row[field] is None else recovery(row[field],
            by_name['M0HLT'][field], by_name['OFFLINE'][field]) for field in ('validation','original_validation')}
    return rows


def run(spec):
    from . import campaign
    gate, measured = campaign.check_spec(spec)
    job = execution(gate, measured)
    root = Path(spec['root'])
    (root/'live_claim').mkdir(exist_ok=False)
    scratch_root = retention.scratch(gate, job)
    write_immutable_json(root/'scratch_location.json', artifact('SCRATCH',
        parents=dict(spec=spec['content_hash']), path=str(scratch_root), job_id=job))
    torch.cuda.reset_peak_memory_stats()
    facts = run_chain(gate, root/'ladder', scratch_root)
    from .preflight import memory_ok
    memory_ok(gate, memory_snapshot(torch), gpu_identity())
    rows = result_rows(gate, root/'ladder')
    endpoint = next(r for r in rows if r['node_id'] == gate['nodes'][-1]['node_id'])
    reference = {r['node_id']:r for r in rows}
    contrasts = {name: {field: {metric: endpoint[field][metric]-reference[name][field][metric]
        for metric in ('accuracy','macro_ovr_auc','cross_entropy')}
        for field in ('validation','original_validation')} for name in ('LOGIT_D000','ATTN_D000')}
    campaign.check_spec(spec)
    result = artifact('RESULTS', parents=dict(spec=spec['content_hash']), rows=rows,
        contrasts=contrasts, single_seed=True, facts=facts, checkpoints_retired=True,
        new_inference_requires_retraining=True, environment=installed_environment(), job_id=job)
    write_immutable_json(root/'results.json', result)
    return result


def saved_results(spec):
    """Archival readback needs no retired model, scratch payload or remote job query."""
    from . import campaign
    root = Path(spec['root'])
    gate = load_json(spec['gate_path'])
    campaign.read_gate(gate)
    measured = load_json(Path(gate['root'])/'preflight/receipt.json')
    validate(measured, 'PREFLIGHT', dict(gate=gate['content_hash']))
    validate(spec, 'SPEC', dict(gate=gate['content_hash'], preflight=measured['content_hash']))
    if root != Path(gate['root'])/'science' or load_json(root/'campaign_spec.json') != spec:
        raise ValueError('Saved result spec/root differs')
    finished = (root/'results.json').is_file()
    rows = result_rows(gate, root/'ladder', partial=not finished)
    if finished:
        result = load_json(root/'results.json')
        validate(result, 'RESULTS', dict(spec=spec['content_hash']))
        if result['rows'] != rows or result['checkpoints_retired'] is not True or result['new_inference_requires_retraining'] is not True:
            raise ValueError('Final aggregate differs from retained outputs')
        endpoint = rows[-1]
        by_name = {r['node_id']:r for r in rows}
        expected = {name:{field:{metric:endpoint[field][metric]-by_name[name][field][metric]
            for metric in ('accuracy','macro_ovr_auc','cross_entropy')}
            for field in ('validation','original_validation')} for name in ('LOGIT_D000','ATTN_D000')}
        if result['contrasts'] != expected:
            raise ValueError('Saved contrasts differ')
    return rows

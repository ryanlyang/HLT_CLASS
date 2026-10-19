"""FullStat final-rung workers; parent artifacts and sealed test stay read-only."""
from pathlib import Path
import os
import time
import numpy as np
import torch

from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.runner import predict, selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import allocation, gpu_identity
from hlt_classification.luka_fullstat import cache
from hlt_classification.luka_fullstat.recipe import learning_rate
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot, require_backend
from hlt_classification.luka_attention.model import new_model
from . import campaign, banks, training
from .contracts import artifact, validate, recipe, REFERENCES, TEACHER


def metadata(gate):
    e = gate['anchor']['evidence']
    meta = cache.metadata(e['foundation_root'], e['cache_root'], gate_hash=e['foundation_gate'])
    if meta[0]['content_hash'] != e['prepared']:
        raise ValueError('Prepared FullStat metadata differs')
    return meta


def data(gate, role, coordinate, *, miniature=False, authenticated=None):
    if role not in ('train', 'validation') or coordinate not in ('D033', 'D000'):
        raise PermissionError('Only registered ordinary-role final-rung views allowed')
    meta = metadata(gate) if authenticated is None else authenticated
    if meta[0]['content_hash'] != gate['anchor']['evidence']['prepared']:
        raise ValueError('Cache parent differs')
    result = cache.load(gate['anchor']['evidence']['cache_root'], meta, role=role,
        coordinate=coordinate, max_ram_bytes=int(gate['memory_mb']*1024**2*.35), miniature=miniature)
    if not miniature and len(result) != gate['counts'][role]:
        raise ValueError('FullStat population differs')
    return result


def execution(gate, measured=None):
    require_backend()
    job, cpus, memory = allocation(gate['execution_site'])
    e = gate['anchor']['evidence']
    if (cpus, memory) != (gate['cpus'], gate['memory_mb']):
        raise PermissionError('Oscar allocation differs')
    if installed_environment() != e['environment']:
        raise PermissionError('Installed environment differs from original FullStat')
    if os.environ.get('TORCH_CUDNN_V8_API_DISABLED', 'unset') != e['cudnn_v8_disabled']:
        raise PermissionError('Backend setting differs from original FullStat')
    if measured and (measured['gpu'] != gpu_identity() or measured['environment'] != installed_environment()):
        raise PermissionError('Runtime differs from accepted new Oscar gate')
    return job


def checkpoint(gate):
    refs = gate['anchor']['evidence']['models'][TEACHER]
    for key in ('checkpoint', 'report'):
        validate_file_ref(refs[key])
    return Path(refs['checkpoint']['path']), dict(checkpoint=refs['checkpoint']['sha256'],
                                                 report=refs['report']['sha256'])


def bank_path(gate):
    return Path(gate['bank_root'])/'D033_train'


def report_valid(report, node, bank_hash, prepared, validation_rows, *, technical=False):
    validate(report, 'TRAINING_REPORT', dict(recipe=recipe()['content_hash'], prepared=prepared, bank=bank_hash))
    history = report['validation_history']
    if (report['node'] != node or report['scientific_fit'] is not (not technical)
            or report['passes'] != (1 if technical else 20) or len(history) != report['passes']
            or [r['pass_number'] for r in history] != list(range(1, len(history)+1))
            or report['selected_weights_restored'] is not True
            or any(r['validation']['rows'] != validation_rows for r in history)
            or any(r['learning_rate'] != learning_rate(r['pass_number']) for r in history)):
        raise ValueError('FullStat20 report population/recipe/status differs')
    best = max(history, key=lambda r: selection_key(r['validation'], r['update']))
    if report['selected_pass'] != best['pass_number'] or report['validation'] != best['validation']:
        raise ValueError('Wrong selected checkpoint report')


def fit(directory, gate, node, train_cache, val, bank, *, device, technical=False):
    model=new_model(node)
    report,state=training.train(model,train_cache,val,bank=bank,node=node,device=device,technical=technical)
    report_valid(report,node,bank.manifest['content_hash'],train_cache.foundation_sha256,len(val),technical=technical)
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    if any((directory/f).exists() for f in ('selected.pt','training_report.json','validation_predictions.npz')):
        raise FileExistsError('Preserve partial fit; no automatic restart')
    save_state(directory/'selected.pt',state)
    restored=torch.load(directory/'selected.pt',map_location='cpu',weights_only=True)
    if set(restored)!=set(state) or any(not torch.equal(restored[n],state[n]) for n in state):
        raise ValueError('Checkpoint roundtrip differs')
    model.load_state_dict(restored,strict=True)
    # Native HLT deployment has no readout/target capability.
    from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
    native=DelphesParticleTransformer().to(device)
    native.load_state_dict(restored,strict=True)
    native.eval(); model.eval(); model.capture_attention=False
    raw=val.batch(np.arange(min(4,len(val))))
    from hlt_classification.jetclass2_delphes.runner import _forward
    with torch.inference_mode():
        torch.testing.assert_close(_forward(native,raw,device,bf16=str(device).startswith('cuda')),
                                   _forward(model,raw,device,bf16=str(device).startswith('cuda')),rtol=0,atol=0)
    native_parameters=sum(p.numel() for p in native.parameters())
    with torch.inference_mode():
        for _ in range(3): _forward(native,raw,device,bf16=str(device).startswith('cuda'))
        if str(device).startswith('cuda'): torch.cuda.synchronize()
        latency_start=time.monotonic()
        for _ in range(20): _forward(native,raw,device,bf16=str(device).startswith('cuda'))
        if str(device).startswith('cuda'): torch.cuda.synchronize()
        batch_ms=(time.monotonic()-latency_start)*1000/20
    del native
    prediction_start=time.monotonic()
    arrays=dict(identities=val.identities,labels=val.labels,
                probabilities=predict(model,val,device=device,batch_size=128))
    costs=dict(parameters=native_parameters,device=str(device),training_validation_seconds=report['runtime_seconds'],
               training_validation_gpu_hours=report['runtime_seconds']/3600 if str(device).startswith('cuda') else None,
               selected_prediction_seconds=time.monotonic()-prediction_start,selected_prediction_jets=len(val),
               native_batch_latency_ms=batch_ms,latency_batch_rows=len(raw['labels']),
               latency_padded_width=raw['mask'].shape[-1],latency_scope='20 warmed label-free forwards; input transfer included')
    recomputed=evaluate_probabilities(arrays['labels'],arrays['probabilities'])
    if recomputed!=report['validation']:
        raise ValueError('Selected validation checkpoint readback differs')
    atomic_publish_bytes(directory/'validation_predictions.npz',deterministic_npz_bytes(arrays))
    write_immutable_json(directory/'training_report.json',report)
    write_immutable_json(directory/'costs.json',costs)
    return report,model


def check_bank(gate, expected_hash=None, expected_alignment=None):
    root = bank_path(gate)
    m = load_json(root/'manifest.json')
    _, parents = checkpoint(gate)
    validate(m, 'BANK', parents)
    if (m['teacher'] != TEACHER or m['coordinate'] != 'D033' or m['attention'] is not True
            or m['temperature'] != 2. or m['surface'] != recipe()['surface']
            or m['alignment']['prepared'] != gate['anchor']['evidence']['prepared']
            or m['alignment']['role'] != 'train' or m['alignment']['rows'] != gate['counts']['train']
            or (expected_hash is not None and m['content_hash'] != expected_hash)
            or (expected_alignment is not None and m['alignment'] != expected_alignment)):
        raise ValueError('Shared scientific bank registration differs')
    for ref in [m['population']]+[s['payload'] for s in m['shards']]:
        validate_file_ref(ref, root=root)
    return m


def validation_arrays(gate):
    ref = gate['anchor']['evidence']['models']['M0HLT']['predictions']
    return load_npz_arrays(validate_file_ref(ref))


def completed(spec, gate, name, seen=None):
    seen = {} if seen is None else seen
    if name in seen:
        return seen[name]
    task = next((t for t in gate['tasks'] if t['task_id'] == name), None)
    if task is None:
        raise ValueError('Unregistered task')
    root = Path(spec['root'])
    path = root/name/'receipt.json'
    if not path.is_file():
        return None
    row = load_json(path)
    validate(row, 'TASK', dict(spec=spec['content_hash']))
    deps = {}
    for dep in task['dependencies']:
        value = completed(spec, gate, dep, seen)
        if value is None:
            raise ValueError('Committed child has missing parent')
        deps[dep] = value['content_hash']
    if row['task_id'] != name or row['dependencies'] != deps or row['source_commit'] != spec['source_commit']:
        raise ValueError('Task/dependency/source differs')
    paths = [ref['path'] for ref in row['outputs']]
    if len(paths) != len(set(paths)) or any(not p.startswith(name+'/') for p in paths):
        raise ValueError('Task output scope differs')
    for ref in row['outputs']:
        validate_file_ref(ref, root=root)
    if task['kind'] == 'export':
        descriptor = load_json(root/name/'bank.json')
        if name+'/bank.json' not in paths or descriptor != row['result']:
            raise ValueError('Missing bank descriptor')
        validate_file_ref(descriptor['manifest'])
        if descriptor['manifest']['path'] != str(bank_path(gate)/'manifest.json'):
            raise ValueError('Shared bank escaped registered root')
        check_bank(gate, descriptor['bank_hash'])
    elif task['kind'] == 'train':
        required = {name+'/'+f for f in ('selected.pt','training_report.json','validation_predictions.npz','costs.json')}
        if not required <= set(paths):
            raise ValueError('Missing selected scientific outputs')
        node = next(n for n in gate['nodes'] if n['node_id'] == task['node_id'])
        report = load_json(root/name/'training_report.json')
        report_valid(report, node, seen['export_D033']['result']['bank_hash'],
                     gate['anchor']['evidence']['prepared'], gate['counts']['validation'])
        p, original = load_npz_arrays(root/name/'validation_predictions.npz'), validation_arrays(gate)
        if (set(p) != {'identities','labels','probabilities'}
                or not np.array_equal(p['identities'], original['identities'])
                or not np.array_equal(p['labels'], original['labels'])
                or evaluate_probabilities(p['labels'], p['probabilities']) != report['validation']
                or row['result'] != dict(report_hash=report['content_hash'])):
            raise ValueError('Selected predictions/population/report differ')
    else:
        result = load_json(root/name/'results.json')
        validate(result, 'RESULTS', dict(spec=spec['content_hash']))
        if name+'/results.json' not in paths or row['result'] != result:
            raise ValueError('Missing aggregate result binding')
        expected = results(spec, gate, seen)
        if result != expected:
            raise ValueError('Aggregate comparison differs')
    seen[name] = row
    return row


def result_rows(spec, gate, seen=None):
    rows = []
    seen = {} if seen is None else seen
    for name in REFERENCES:
        refs = gate['anchor']['evidence']['models'][name]
        report = load_json(validate_file_ref(refs['report']))
        p = load_npz_arrays(validate_file_ref(refs['predictions']))
        metrics = evaluate_probabilities(p['labels'], p['probabilities'])
        if metrics != report['validation']:
            raise ValueError('Reference metrics differ')
        input_role = ('offline_oracle' if name == 'OFFLINE' else
                      'offline_containing_oracle' if name == TEACHER else 'single_hlt')
        rows.append(dict(node_id=name, origin='completed_parent_reference', input_role=input_role,
                         selected_epoch=report['selected_pass'], validation=metrics))
    for node in gate['nodes']:
        name = node['node_id']
        done = completed(spec, gate, 'train_'+name, seen)
        report = None if done is None else load_json(Path(spec['root'])/('train_'+name)/'training_report.json')
        rows.append(dict(node_id=name, origin='new_matched_fit', input_role='single_hlt',
            selected_epoch=None if report is None else report['selected_pass'],
            validation=None if report is None else report['validation']))
    for row in rows:
        row['recovery'] = None if row['validation'] is None else recovery(row['validation'], rows[0]['validation'], rows[1]['validation'])
    return rows


def results(spec, gate, seen=None):
    rows = result_rows(spec, gate, seen)
    if any(r['validation'] is None for r in rows):
        raise ValueError('Both fits required regardless of quality')
    a, b = [next(r['validation'] for r in rows if r['node_id'] == n) for n in ('ATTN_D000','LOGIT_D000')]
    return artifact('RESULTS', parents=dict(spec=spec['content_hash']), rows=rows, counts=gate['counts'],
        primary_contrast='ATTN_D000 minus LOGIT_D000', single_seed=True,
        contrasts={k:a[k]-b[k] for k in ('accuracy','macro_ovr_auc','cross_entropy')})


def run(spec, name):
    gate, measured = campaign.check_spec(spec)
    task = next((t for t in gate['tasks'] if t['task_id'] == name), None)
    if task is None:
        raise ValueError('Unknown task; no test operation exists')
    seen = {}
    old = completed(spec, gate, name, seen)
    if old:
        return old
    deps = {}
    for dep in task['dependencies']:
        value = completed(spec, gate, dep, seen)
        if value is None:
            raise ValueError('Missing completed dependency')
        deps[dep] = value['content_hash']
    job = execution(gate, measured) if task['kind'] != 'aggregate' else os.environ.get('SLURM_JOB_ID')
    root, directory = Path(spec['root']), Path(spec['root'])/name
    directory.mkdir(exist_ok=False)
    log = PhaseLog(directory, lambda: memory_snapshot(torch))
    if task['kind'] == 'export':
        with log.phase('load_D033_and_export_shared_TRAIN_targets'):
            tr = data(gate, 'train', 'D033')
            if banks.alignment(tr) != measured['audit']['train_alignment']:
                raise ValueError('D033 slot alignment changed')
            campaign.storage(gate, measured['audit']['sum_squared_counts'], len(tr))
            path, parents = checkpoint(gate)
            teacher = new_model(gate['nodes'][0])
            teacher.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
            teacher.requires_grad_(False)
            manifest = banks.export(bank_path(gate), teacher, tr, parents=parents,
                                    teacher=TEACHER, attention=True, device='cuda')
            banks.Bank(bank_path(gate), tr, parents=parents, teacher=TEACHER, require_attention=True)
            result = dict(manifest=file_ref(bank_path(gate)/'manifest.json'), bank_hash=manifest['content_hash'])
            write_immutable_json(directory/'bank.json', result)
            del tr, teacher
    elif task['kind'] == 'train':
        with log.phase('load_exact_HLT_views_and_shared_teacher_targets'):
            tr, val = data(gate, 'train', 'D000'), data(gate, 'validation', 'D000')
            if (banks.alignment(tr) != measured['audit']['train_alignment']
                    or banks.alignment(val) != measured['audit']['validation_alignment']):
                raise ValueError('HLT population/slot transport changed')
            _, parents = checkpoint(gate)
            bank = banks.Bank(bank_path(gate), tr, parents=parents, teacher=TEACHER, require_attention=True)
            node = next(n for n in gate['nodes'] if n['node_id'] == task['node_id'])
        with log.phase('scientific_twenty_epoch_fit'):
            report, model = fit(directory, gate, node, tr, val, bank, device='cuda')
            result = dict(report_hash=report['content_hash'])
            del tr, val, bank, model
    else:
        result = results(spec, gate, seen)
        write_immutable_json(directory/'results.json', result)
    if task['kind'] != 'aggregate':
        execution(gate, measured)
    campaign.check_spec(spec)
    if any(completed(spec, gate, d)['content_hash'] != h for d,h in deps.items()):
        raise ValueError('Dependency changed during execution')
    receipt = artifact('TASK', parents=dict(spec=spec['content_hash']), task_id=name,
        source_commit=spec['source_commit'], dependencies=deps, job_id=job, result=result,
        outputs=[file_ref(p, root=root) for p in sorted(directory.rglob('*')) if p.is_file()])
    write_immutable_json(directory/'receipt.json', receipt)
    return completed(spec, gate, name)

"""Measured four-rung chain with native-only teacher handoffs and compact records."""
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
from hlt_classification.luka_attention.worker import execution
from hlt_classification.luka_embedding_weight.worker import storage, validate_predictions
from hlt_classification.luka_embedding_aux.model import new_model
from hlt_classification.luka_embedding_aux.preflight import parity
from .contracts import artifact, validate, COORDINATES, PREVIOUS, TASKS, stage_node, teacher_id
from . import campaign, source, training, targets, data, submission

REFS = ('M0HLT','OFFLINE','U000')
FIT_FILES = ('selected.pt','training_report.json','validation_predictions.npz')
BANK_FILES = ('manifest.json','targets.npz')


def stage_dir(spec, co, technical=False):
    root = Path(spec['root'])
    return root/'setup'/('probe_'+co) if technical else root/co


def incoming_dir(spec, co, technical=False):
    previous = PREVIOUS[co]
    return Path(spec['root'])/'targets/U000' if previous == 'U000' else stage_dir(spec, previous, technical)/'targets'


def teacher_for(spec, coordinate, technical=False):
    if coordinate == 'U000':
        refs = spec['anchor']['evidence']['models']['U000']
        files = {k: refs[k] for k in ('checkpoint','report','predictions')}
        technical = False
    else:
        root = stage_dir(spec, coordinate, technical)/'fit'
        files = {k: file_ref(root/name) for k, name in
                 dict(checkpoint='selected.pt',report='training_report.json',predictions='validation_predictions.npz').items()}
    value = artifact('TEACHER', parents=dict(spec=spec['content_hash'], prepared=spec['anchor']['evidence']['prepared'],
        **{k: files[k]['sha256'] for k in ('checkpoint','report','predictions')}),
        spec=spec['content_hash'], prepared=spec['anchor']['evidence']['prepared'], coordinate=coordinate,
        model_id=teacher_id(coordinate), target_role='train', tap='native_preclassifier', technical=technical,
        origin='original_anchor' if coordinate == 'U000' else 'this_campaign', files=files)
    targets.validate_teacher(value)
    return value


def load_native(ref, device):
    validate_file_ref(ref)
    native = DelphesParticleTransformer().to(device).eval()
    native.load_state_dict(torch.load(ref['path'], map_location='cpu', weights_only=True), strict=True)
    validate_file_ref(ref)
    return native


def output_names(task):
    if task != 'setup':
        return [task+'/fit/'+n for n in FIT_FILES] + ([] if task == 'D000' else [task+'/targets/'+n for n in BANK_FILES])
    return (['targets/U000/'+n for n in BANK_FILES] + ['references/'+n+'.npz' for n in REFS] +
        ['references/original50k.npz', 'setup/readback.json','setup/checks.json','setup/measurement.json'] +
        ['setup/probe_'+co+'/fit/'+n for co in COORDINATES for n in FIT_FILES] +
        ['setup/probe_'+co+'/targets/'+n for co in COORDINATES[:-1] for n in BANK_FILES])


def snapshot_references(spec, val, meta):
    folder = Path(spec['root'])/'references'; folder.mkdir(exist_ok=False)
    ix = alljets_data.comparison_indices(meta[2], meta[3], val.identities, val.labels)
    atomic_publish_bytes(folder/'original50k.npz', deterministic_npz_bytes(dict(indices=ix)))
    for name in REFS:
        ref = spec['anchor']['evidence']['models'][name]['predictions']; validate_file_ref(ref)
        a = load_npz_arrays(ref['path']); validate_predictions(a, len(val))
        if not np.array_equal(a['identities'], val.identities) or not np.array_equal(a['labels'], val.labels):
            raise ValueError('Reference population differs')
        atomic_publish_bytes(folder/(name+'.npz'), deterministic_npz_bytes(a)); validate_file_ref(ref)


def fit(directory, spec, node, train, val, bank, device, *, technical=False):
    directory = Path(directory); directory.mkdir(exist_ok=False)
    # Fresh initialization and predictor on EVERY rung, including all probes.
    model = new_model(node)
    report, state = training.train(model, train, val, bank=bank, node=node, device=device, technical=technical)
    training.validate_report(report, spec, bank.manifest['content_hash'], node, technical=technical)
    save_state(directory/'selected.pt', state)
    native = load_native(file_ref(directory/'selected.pt'), device)
    expected = predict(model, val, device=device, batch_size=128)
    actual = predict(native, val, device=device, batch_size=128)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    if evaluate_probabilities(val.labels, actual) != report['validation']:
        raise ValueError('Selected native probability readback differs')
    atomic_publish_bytes(directory/'validation_predictions.npz', deterministic_npz_bytes(dict(
        identities=val.identities, labels=val.labels, probabilities=actual)))
    record = artifact('FIT_REPORT', parents=dict(report=report['content_hash']), report=report,
        checkpoint=file_ref(directory/'selected.pt'), checkpoint_readback=True)
    write_immutable_json(directory/'training_report.json', record)
    return record


def validate_fit(directory, spec, bank_hash, node, *, technical=False):
    directory = Path(directory); record = load_json(directory/'training_report.json')
    report = record['report']; validate(record, 'FIT_REPORT', dict(report=report['content_hash']))
    training.validate_report(report, spec, bank_hash, node, technical=technical)
    if record['checkpoint'] != file_ref(directory/'selected.pt') or record['checkpoint_readback'] is not True:
        raise ValueError('Native selected checkpoint binding differs')
    a = load_npz_arrays(directory/'validation_predictions.npz'); validate_predictions(a, spec['counts']['validation'])
    if evaluate_probabilities(a['labels'], a['probabilities']) != report['validation']:
        raise ValueError('Saved selected metrics differ')
    return record


def stress(cache, bank, node, device):
    lengths = np.concatenate([np.diff(b.offsets) for b in cache.blocks]); ix = np.repeat(int(np.argmax(lengths)),128)
    raw = cache.batch(ix); backward = cache.role == 'train'
    model = new_model(node).to(device); model.train(backward); model.capture = backward
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(backward):
            logits = _forward(model, raw, device, bf16=str(device).startswith('cuda'))
            if not torch.isfinite(logits).all(): raise ValueError('Nonfinite stress logits')
            if backward:
                loss, _, _ = training.objective(logits, torch.from_numpy(raw['labels']).to(device),
                    model.prediction, torch.from_numpy(bank.targets[ix]).to(device), coefficient=5.)
                loss.backward()
                finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
                if finite and not torch.stack(finite).all(): raise ValueError('Nonfinite stress gradients')
                optimizer.step()
    if str(device).startswith('cuda'): torch.cuda.synchronize()
    return dict(role=cache.role, coordinate=cache.coordinate_name, maximum=int(lengths.max()),
                batch=128, steps=3, backward=backward)


def execute_stage(spec, co, folder, meta, device, *, technical=False, checks=None):
    start = time.perf_counter(); node = stage_node(spec, co)
    train, val = (data.load_views(spec, role, [co], authenticated=meta)[co] for role in ('train','validation'))
    incoming = teacher_for(spec, PREVIOUS[co], technical)
    bank = targets.Bank(incoming_dir(spec,co,technical), train, teacher_record=incoming, node=node, technical=technical)
    load_seconds = time.perf_counter()-start
    if technical:
        if co == 'U100':
            with parity_backend(device): checks['native'] = installed_parity(train, device=device)
            checks['parities'] = [parity(train.batch(np.arange(min(4,len(train)))), node, device, bf16=b) for b in (False,True)]
        for cache in (train,val):
            witness = stress(cache,bank,node,device)
            if witness['maximum'] != spec['support'][cache.role][co]: raise ValueError('Stress support differs')
            checks['stress'].append(witness)
        gc.collect(); torch.cuda.empty_cache()
    start = time.perf_counter()
    record = fit(folder/'fit',spec,node,train,val,bank,device,technical=technical)
    if co != 'D000':
        outgoing = teacher_for(spec,co,technical)
        native = load_native(outgoing['files']['checkpoint'],device)
        targets.export(folder/'targets',native,train,teacher_record=outgoing,device=device)
    elapsed = time.perf_counter()-start
    row = record['report']['validation_history'][0]
    return dict(load=load_seconds, train_epoch=row['train_seconds'], validation_epoch=row['validation_seconds'],
                readback_export=max(0.,elapsed-row['train_seconds']-row['validation_seconds']))


def admission(spec, m):
    if m['environment'] != spec['anchor']['evidence']['environment']: raise ValueError('Installed environment differs')
    if any(type(m[k]) is not int or m[k] <= 0 for k in ('peak_rss_bytes','gpu_peak_bytes')):
        raise ValueError('Missing memory high-water evidence')
    if (m['peak_rss_bytes'] >= .85*spec['memory_mb']*1024**2 or m['gpu_peak_bytes'] >= .9*m['gpu']['total_memory_bytes']):
        raise MemoryError('Resource memory admission failed')
    if set(m['fits']) != set(COORDINATES) or not math.isfinite(m['authentication']) or m['authentication'] < 0:
        raise ValueError('Incomplete stage measurement')
    projected = {}
    for co,t in m['fits'].items():
        if (set(t) != {'load','train_epoch','validation_epoch','readback_export'}
                or any(not math.isfinite(v) or v < 0 for v in t.values())
                or min(t['train_epoch'],t['validation_epoch']) <= 0): raise ValueError('Invalid stage measurement')
        projected[co] = max(30, math.ceil(1.75*(m['authentication']+t['load']+
            30*(t['train_epoch']+t['validation_epoch'])+t['readback_export']+300)/60))
        if projected[co] > spec['minutes'][co]: raise ValueError('Measured stage exceeds registered walltime: '+co)
    return projected


def setup(spec, folder, log, meta, authentication, device):
    root = Path(spec['root']); checks = dict(stress=[])
    with log.phase('original_U000_readback_and_TRAIN_native_embedding_export'):
        incoming = teacher_for(spec,'U000'); native = load_native(incoming['files']['checkpoint'],device)
        train, val = (data.load_views(spec,role,['U000'],authenticated=meta)['U000'] for role in ('train','validation'))
        snapshot_references(spec,val,meta)
        saved = load_npz_arrays(root/'references/U000.npz')
        actual = predict(native,val,device=device,batch_size=128)
        np.testing.assert_allclose(actual,saved['probabilities'],rtol=1e-5,atol=1e-6)
        write_immutable_json(folder/'readback.json',artifact('READBACK',parents=dict(teacher=incoming['content_hash']),
            passed=True, rows=len(val), max_error=float(np.max(np.abs(actual-saved['probabilities'])))))
        targets.export(root/'targets/U000',native,train,teacher_record=incoming,device=device)
        del train,val,native
        gc.collect(); torch.cuda.empty_cache()
    fits = {}
    for co in COORDINATES:
        with log.phase('technical_full_population_'+co):
            directory = stage_dir(spec,co,True); directory.mkdir(exist_ok=False)
            fits[co] = execute_stage(spec,co,directory,meta,device,technical=True,checks=checks)
            gc.collect(); torch.cuda.empty_cache()
    write_immutable_json(folder/'checks.json',artifact('CHECKS',**checks))
    memory = memory_snapshot(torch)
    m = dict(environment=installed_environment(),gpu=gpu_identity(),authentication=authentication,fits=fits,
        peak_rss_bytes=memory['peak_rss_bytes'],gpu_peak_bytes=memory['gpu_peak_bytes'])
    write_immutable_json(folder/'measurement.json',artifact('MEASUREMENT',**m,projected_minutes=admission(spec,m)))


def validate_stage(spec,co,*,technical=False):
    teacher = teacher_for(spec,PREVIOUS[co],technical)
    bank = targets.validate_manifest(incoming_dir(spec,co,technical),teacher,spec['anchor']['evidence']['prepared'],spec['counts'])
    folder = stage_dir(spec,co,technical)
    record = validate_fit(folder/'fit',spec,bank['content_hash'],stage_node(spec,co),technical=technical)
    if co != 'D000':
        targets.validate_manifest(folder/'targets',teacher_for(spec,co,technical),spec['anchor']['evidence']['prepared'],spec['counts'])
    return record


def validate_task(spec,task,seen=None):
    seen = {} if seen is None else seen
    if task in seen: return seen[task]
    root = Path(spec['root']); row = load_json(root/task/'receipt.json')
    deps = {n:validate_task(spec,n,seen)['content_hash'] for n in TASKS[task]}
    validate(row,'TASK',dict(spec=spec['content_hash'],**deps))
    if row['task'] != task or [r['path'] for r in row['outputs']] != output_names(task):
        raise ValueError('Task output inventory differs')
    for ref in row['outputs']: validate_file_ref(ref,root=root)
    if task == 'setup':
        m = load_json(root/'setup/measurement.json'); validate(m,'MEASUREMENT')
        if m['projected_minutes'] != admission(spec,m): raise ValueError('Admission differs')
        proof = load_json(root/'setup/readback.json'); validate(proof,'READBACK',dict(teacher=teacher_for(spec,'U000')['content_hash']))
        checks = load_json(root/'setup/checks.json'); validate(checks,'CHECKS')
        if (proof['passed'] is not True or proof['rows'] != spec['counts']['validation'] or not 0 <= proof['max_error'] <= 1.1e-5
                or checks['native']['passed'] is not True or [p['bf16'] for p in checks['parities']] != [False,True]
                or any(p[k] is not True for p in checks['parities'] for k in
                    ('native_forward','zero_control_update','auxiliary_backbone_gradient','auxiliary_head_gradient'))
                or [(s['coordinate'],s['role'],s['batch'],s['steps'],s['backward']) for s in checks['stress']] !=
                   [(co,r,128,3,r=='train') for co in COORDINATES for r in ('train','validation')]
                or any(s['maximum'] != spec['support'][s['role']][s['coordinate']] for s in checks['stress'])):
            raise ValueError('Setup technical evidence differs')
        for co in COORDINATES: validate_stage(spec,co,technical=True)
    else: validate_stage(spec,task)
    seen[task] = row
    return row


def run(spec,task):
    if task not in TASKS: raise ValueError('Unknown task')
    start = time.perf_counter(); campaign.check(spec); authentication = time.perf_counter()-start
    jobs = submission.accepted_jobs(spec,partial=task=='setup'); deps = {}; seen = {}
    for dep in TASKS[task]:
        receipt = validate_task(spec,dep,seen)
        if receipt['job_id'] != jobs[dep]: raise ValueError('Dependency differs from exact journal')
        preflight_accounting(jobs[dep]); deps[dep] = receipt['content_hash']
    measured = None if task == 'setup' else load_json(Path(spec['root'])/'setup/measurement.json')
    job = execution(spec,measured)
    if job != jobs[task] or installed_environment() != spec['anchor']['evidence']['environment']:
        raise ValueError('Actual job/environment differs')
    storage(spec); root = Path(spec['root']); folder = root/task; folder.mkdir(exist_ok=False)
    log = PhaseLog(folder,lambda:memory_snapshot(torch)); torch.cuda.reset_peak_memory_stats()
    meta = regular_cache.metadata(spec['anchor']['evidence']['prepared_root'])
    if task == 'setup': setup(spec,folder,log,meta,authentication,'cuda')
    else:
        with log.phase('scientific_'+task+'_30_epochs_and_native_handoff'):
            execute_stage(spec,task,folder,meta,'cuda')
    storage(spec)
    if source.current(spec['project_dir'],spec['source_commit']) != spec['source_snapshot']: raise ValueError('Source changed')
    record = artifact('TASK',parents=dict(spec=spec['content_hash'],**deps),task=task,job_id=job,
        outputs=[file_ref(root/n,root=root) for n in output_names(task)])
    write_immutable_json(folder/'receipt.json',record); validate_task(spec,task)
    return record


def results(spec):
    campaign.validate_spec(spec,payloads=False); root = Path(spec['root']); seen = {}
    parents = dict(spec=spec['content_hash'],setup=validate_task(spec,'setup',seen)['content_hash'])
    arrays = {n:load_npz_arrays(root/'references'/(n+'.npz')) for n in REFS}; pending = []
    for co in COORDINATES:
        if not (root/co/'receipt.json').is_file(): pending.append(teacher_id(co)); continue
        parents[co] = validate_task(spec,co,seen)['content_hash']
        arrays[teacher_id(co)] = load_npz_arrays(root/co/'fit/validation_predictions.npz')
    ix = load_npz_arrays(root/'references/original50k.npz')['indices']; anchor = arrays['M0HLT']
    if (ix.dtype != np.int64 or ix.shape != (spec['original50k_rows'],) or len(np.unique(ix)) != len(ix)
            or np.any(ix<0) or np.any(ix>=len(anchor['labels']))): raise ValueError('Report-only subset differs')
    rows = []
    for name,a in arrays.items():
        validate_predictions(a,spec['counts']['validation'])
        if not np.array_equal(a['identities'],anchor['identities']) or not np.array_equal(a['labels'],anchor['labels']):
            raise ValueError('Comparison population differs')
        rows.append(dict(model=name,input='single HLT' if name in ('M0HLT','PRED_D000') else 'ORACLE',
            validation=evaluate_probabilities(a['labels'],a['probabilities']),
            original50k=evaluate_probabilities(a['labels'][ix],a['probabilities'][ix])))
    for row in rows:
        for role in ('validation','original50k'): row[role+'_recovery'] = recovery(row[role],rows[0][role],rows[1][role])
    return artifact('RESULTS',parents=parents,counts=spec['counts'],rows=rows,pending=pending)


def print_results(result):
    print('Native embedding prediction ladder: CE+5*embedding, no logit KD. Historical references; NO independent test.')
    print('Original50k is report-only. Intermediate richer-view students are ORACLE; PRED_D000 is single HLT.')
    for role in ('validation','original50k'):
        print('\n'+role)
        print(f'{"model":20} {"accuracy":>10} {"AUC":>10} {"AUC recovery%":>14} input')
        for row in result['rows']:
            m=row[role]; rec=row[role+'_recovery']['macro_ovr_auc']; display='n/a' if rec is None else f'{rec:.1f}'
            print(f'{row["model"]:20} {m["accuracy"]:10.6f} {m["macro_ovr_auc"]:10.6f} {display:>14} {row["input"]}')
            values = [name+'='+('censored (0 passing)' if v['rejection_at_50pct'] is None else f'{v["rejection_at_50pct"]:.1f}')
                      for name,v in m['per_class'].items()]
            print('  QCD rejection @50%: '+', '.join(values))
    for name in result['pending']: print(name+': PENDING/INCOMPLETE')

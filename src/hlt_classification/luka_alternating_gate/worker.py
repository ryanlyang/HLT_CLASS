"""Per-fit scratch targets and permanent selected outputs, no inherited weights."""
from pathlib import Path
from hashlib import sha256
import gc
import os
import re
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from hlt_classification.luka_attention.worker import data, execution
from hlt_classification.luka_attention_weight.worker import original_indices
from . import banks
from hlt_classification.luka_gating.model import new_model
from .contracts import validate as validate_bank, teacher_node as registered_teacher
from . import training, storage
from .contracts import artifact, validate, recipe, input_label

PERSISTENT = ('selected.pt','training_report.json','validation_predictions.npz','teacher_bank.json',
              'selected_outputs.json','retired_targets/intent.json','retired_targets/receipt.json')


def anchor_node(gate):
    return registered_teacher(gate['nodes'], gate['nodes'][0])


def teacher_refs(spec, gate, name, *, technical=False):
    if name == 'ANCHOR':
        refs = gate['anchor']['evidence']['models']['COARSE_U100']
        for key in ('checkpoint','report'): validate_file_ref(refs[key])
        return refs['checkpoint'], refs['report'], gate['anchor']['content_hash'], anchor_node(gate)
    parent = completed(spec,gate,'train_'+name,technical=technical)
    if parent is None: raise ValueError('Missing selected cold teacher')
    root = Path(spec['root'])/('train_'+name)
    return (file_ref(root/'selected.pt'),file_ref(root/'training_report.json'),parent['content_hash'],
            next(n for n in gate['nodes'] if n['node_id']==name))


def report_valid(report,node,manifest,gate,technical):
    validate(report,'TRAINING_REPORT',dict(recipe=recipe()['content_hash'],
        prepared=gate['anchor']['evidence']['prepared'],bank=manifest['content_hash']))
    history = report['validation_history']; epochs = 1 if technical else 30
    if (report['node'] != node or report['scientific_fit'] is not (not technical)
            or report['passes'] != epochs or len(history) != epochs
            or [r['pass_number'] for r in history] != list(range(1,epochs+1))
            or report['selected_weights_restored'] is not True
            or report['initialization_policy'] != recipe()['initialization']
            or not re.fullmatch('[0-9a-f]{64}',report['initialization_sha256'])
            or node['warm_start'] is not None or node['freeze_epochs'] != 0
            or any(r['backbone_frozen'] or r['validation']['rows']!=gate['counts']['validation']
                   or r['update'] != r['pass_number']*((gate['counts']['train']+127)//128) for r in history)):
        raise ValueError('Cold initialization/population/budget differs')
    selected = max(history,key=lambda r:selection_key(r['validation'],r['update']))
    if report['selected_pass'] != selected['pass_number'] or report['validation'] != selected['validation']:
        raise ValueError('Selection differs from full validation')


def validate_selected(directory,gate,node,technical):
    directory = Path(directory)
    proof = load_json(directory/'selected_outputs.json')
    validate(proof,'SELECTED_OUTPUTS',dict(gate=gate['content_hash']))
    if proof['node'] != node or proof['technical'] is not technical or proof['checkpoint_readback'] is not True:
        raise ValueError('Selected-output identity differs')
    for key in ('checkpoint','report','predictions','teacher_manifest'):
        validate_file_ref(proof[key],root=directory)
    if {proof[k]['path'] for k in ('checkpoint','report','predictions','teacher_manifest')} != set(PERSISTENT[:4]):
        raise ValueError('Selected-output filenames differ')
    manifest = load_json(directory/'teacher_bank.json')
    validate_bank(manifest,'BANK',proof['teacher_parents'])
    teacher = registered_teacher(gate['nodes'], node)
    if (manifest['teacher'] != node['teacher'] or manifest['coordinate'] != node['teacher_coordinate']
            or manifest['teacher_node'] != teacher
            or manifest['context_coordinate'] != teacher['context_coordinate']
            or manifest['temperature'] != 2.
            or manifest['factors'] is not bool(node['gate_weight'])
            or manifest['alignment'] != proof['train_alignment']):
        raise ValueError('Teacher bank meaning differs')
    for role in ('train','validation'):
        alignment = proof[role+'_alignment']
        if (alignment['rows'] != gate['counts'][role] or alignment['role'] != role
                or alignment['prepared'] != gate['anchor']['evidence']['prepared']):
            raise ValueError('Ordered population binding differs')
    report = load_json(directory/'training_report.json')
    report_valid(report,node,manifest,gate,technical)
    arrays = load_npz_arrays(directory/'validation_predictions.npz')
    count = gate['counts']['validation']
    if (set(arrays) != {'identities','labels','counts','probabilities','original_validation'}
            or arrays['identities'].shape != (count,32) or arrays['identities'].dtype != np.uint8
            or len(np.unique(arrays['identities'],axis=0)) != count
            or arrays['labels'].shape != (count,) or arrays['labels'].dtype != np.int64
            or arrays['counts'].shape != (count,) or arrays['counts'].dtype != np.int64
            or np.any((arrays['counts']<1)|(arrays['counts']>512))
            or arrays['original_validation'].shape != (count,) or arrays['original_validation'].dtype != np.bool_
            or not arrays['original_validation'].any()):
        raise ValueError('Prediction identities or subset invalid')
    digest=sha256()
    for a in (arrays['identities'],arrays['labels'].astype('<i8'),arrays['counts'].astype('<i8')):
        digest.update(a.tobytes())
    if digest.hexdigest()!=proof['validation_alignment']['ordered_slots_sha256']:
        raise ValueError('Predictions differ from authenticated validation slots')
    banks.check_probabilities(arrays['probabilities'],count)
    if evaluate_probabilities(arrays['labels'],arrays['probabilities']) != report['validation']:
        raise ValueError('Saved metrics/predictions differ')
    mask = arrays['original_validation']
    if evaluate_probabilities(arrays['labels'][mask],arrays['probabilities'][mask]) != proof['original_validation']:
        raise ValueError('Original50k report-only metrics differ')
    return proof


def completed(spec,gate,task_id,seen=None,*,technical=False):
    seen = {} if seen is None else seen
    if task_id in seen: return seen[task_id]
    task = next((t for t in gate['tasks'] if t['task_id']==task_id),None)
    if task is None: raise ValueError('Unregistered cold task')
    directory = Path(spec['root'])/task_id
    if not (directory/'receipt.json').is_file(): return None
    receipt = load_json(directory/'receipt.json')
    validate(receipt,'TASK',dict(spec=spec['content_hash']))
    dependencies = {}
    for dep in task['dependencies']:
        parent = completed(spec,gate,dep,seen,technical=technical)
        if parent is None: raise ValueError('Completed descendant lacks authenticated parent')
        dependencies[dep] = parent['content_hash']
    if receipt['task_id'] != task_id or receipt['technical'] is not technical or receipt['dependencies'] != dependencies:
        raise ValueError('Task dependency lineage differs')
    paths = [r['path'] for r in receipt['outputs']]
    required = set(PERSISTENT) if task['kind']=='train' else {'results.json'}
    if len(paths)!=len(set(paths)) or set(paths)!=required:
        raise ValueError('Incomplete selected output coverage')
    for ref in receipt['outputs']: validate_file_ref(ref,root=directory)
    if task['kind']=='train':
        node = next(n for n in gate['nodes'] if n['node_id']==task['node_id'])
        proof = validate_selected(directory,gate,node,technical)
        if node['teacher']=='ANCHOR':
            refs = gate['anchor']['evidence']['models']['COARSE_U100']
            parents = {k:refs[k]['sha256'] for k in ('checkpoint','report')}
            expected_teacher_receipt = gate['anchor']['content_hash']
        else:
            teacher_dir = Path(spec['root'])/('train_'+node['teacher'])
            parents = dict(checkpoint=file_ref(teacher_dir/'selected.pt')['sha256'],
                           report=file_ref(teacher_dir/'training_report.json')['sha256'])
            expected_teacher_receipt = dependencies['train_'+node['teacher']]
        if proof['teacher_parents'] != parents or proof['teacher_receipt'] != expected_teacher_receipt:
            raise ValueError('Selected teacher lineage differs')
        storage.validate_retirement(directory/'retired_targets',gate['content_hash'])
        intent = load_json(directory/'retired_targets/intent.json')
        manifest = load_json(directory/'teacher_bank.json')
        expected = [dict(r,path='targets/'+r['path']) for r in [manifest['population']]+[s['payload'] for s in manifest['shards']]]
        expected.append(dict(file_ref(directory/'teacher_bank.json'),path='targets/manifest.json'))
        if intent['files'] != expected or intent['consumer'] != file_ref(directory/'selected_outputs.json'):
            raise ValueError('Retirement inventory or last consumer differs')
    seen[task_id] = receipt
    return receipt


def close_bank(bank):
    bank.factors = None


def execute_task(spec,gate,task_id,*,device='cuda',technical=False,stress_fn=None,before_publish=None):
    task = next(t for t in gate['tasks'] if t['task_id']==task_id)
    if task['kind']!='train': raise ValueError('Fit worker cannot aggregate')
    node = next(n for n in gate['nodes'] if n['node_id']==task['node_id'])
    cp, rp, teacher_receipt, teacher_node = teacher_refs(spec,gate,node['teacher'],technical=technical)
    dependencies = {} if node['teacher']=='ANCHOR' else {'train_'+node['teacher']:teacher_receipt}
    root = Path(spec['root'])/task_id
    root.mkdir(exist_ok=False)
    scratch = storage.scratch(gate,os.environ.get('SLURM_JOB_ID','local'))
    begin = time.monotonic()
    tr, val = data(gate,'train',node['coordinate']), data(gate,'validation',node['coordinate'])
    context = None if node['context_coordinate'] is None else data(gate,'train',node['context_coordinate'])
    val_context = None if context is None else data(gate,'validation',node['context_coordinate'])
    banks.paired(tr,context); banks.paired(val,val_context)
    budget = storage.budget(banks.counts(tr),Path(gate['root']),scratch,gate['available_quota_gib'])
    stresses = []
    stress_seconds = 0.
    if technical and stress_fn:
        stress_start = time.monotonic()
        stresses = [stress_fn(tr,context,node,device,backward=True),stress_fn(val,val_context,node,device,backward=False)]
        stress_seconds = time.monotonic()-stress_start
        gc.collect()
        if str(device).startswith('cuda'): torch.cuda.empty_cache()
    teacher_tr = data(gate,'train',teacher_node['coordinate'])
    teacher_context = None if teacher_node['context_coordinate'] is None else data(gate,'train',teacher_node['context_coordinate'])
    if banks.alignment(teacher_tr) != banks.alignment(tr): raise ValueError('Teacher/student slot alignment differs')
    teacher = new_model(teacher_node,torch.load(cp['path'],map_location='cpu',weights_only=True))
    teacher.requires_grad_(False); teacher.eval()
    parents = dict(checkpoint=cp['sha256'],report=rp['sha256'])
    manifest = banks.export(scratch/'targets',teacher,teacher_tr,teacher_context,
                            parents=parents,node=teacher_node,device=device)
    del teacher,teacher_tr,teacher_context
    gc.collect()
    if str(device).startswith('cuda'): torch.cuda.empty_cache()
    bank = banks.Bank(scratch/'targets',tr,parents=parents,teacher_node=teacher_node,require_factors=bool(node['gate_weight']))
    write_immutable_json(root/'teacher_bank.json',manifest)
    setup_seconds = time.monotonic()-begin-stress_seconds
    started = time.monotonic()
    report,state,model = training.train(tr,val,context,val_context,bank=bank,node=node,device=device,technical=technical)
    fit_seconds = time.monotonic()-started
    save_state(root/'selected.pt',state)
    restored = torch.load(root/'selected.pt',map_location='cpu',weights_only=True)
    if restored.keys()!=state.keys() or any(not torch.equal(v,restored[k]) for k,v in state.items()):
        raise ValueError('Selected checkpoint roundtrip differs')
    deployed = new_model(node,restored).to(device).eval()
    predictions = banks.predict(deployed,val,val_context,device=device)
    if evaluate_probabilities(val.labels,predictions) != report['validation']:
        raise ValueError('Selected deployment readback differs')
    subset = np.zeros(len(val),dtype=bool); subset[original_indices(gate,val)] = True
    atomic_publish_bytes(root/'validation_predictions.npz',deterministic_npz_bytes(dict(
        identities=val.identities,labels=val.labels,counts=banks.counts(val),
        probabilities=predictions,original_validation=subset)))
    write_immutable_json(root/'training_report.json',report)
    proof = artifact('SELECTED_OUTPUTS',parents=dict(gate=gate['content_hash']),node=node,technical=technical,
        teacher_parents=parents,teacher_receipt=teacher_receipt,checkpoint_readback=True,
        checkpoint=file_ref(root/'selected.pt',root=root),report=file_ref(root/'training_report.json',root=root),
        predictions=file_ref(root/'validation_predictions.npz',root=root),teacher_manifest=file_ref(root/'teacher_bank.json',root=root),
        train_alignment=banks.alignment(tr),validation_alignment=banks.alignment(val),
        original_validation=evaluate_probabilities(val.labels[subset],predictions[subset]))
    write_immutable_json(root/'selected_outputs.json',proof)
    validate_selected(root,gate,node,technical)
    close_bank(bank); del bank
    refs = [dict(r,path='targets/'+r['path']) for r in storage.bank_refs(scratch/'targets',manifest)]
    storage.retire(scratch,refs,owner=gate['content_hash'],consumer=file_ref(root/'selected_outputs.json'),output=root/'retired_targets')
    epoch_seconds = sum(report['validation_history'][0][k] for k in ('train_seconds','validation_seconds'))
    measurement = dict(node_id=node['node_id'],setup_seconds=setup_seconds,epoch_seconds=epoch_seconds,
        other_seconds=max(0.,time.monotonic()-started-fit_seconds),storage=budget,stress=stresses,
        initialization_sha256=report['initialization_sha256'],checkpoint_bytes=(root/'selected.pt').stat().st_size)
    del model,deployed,state,restored,tr,val,context,val_context
    if sum(p.stat().st_size for p in Path(gate['root']).rglob('*') if p.is_file()) >= 2*2**30:
        raise OSError('Persistent selected outputs exceed registered 2GiB envelope')
    if before_publish: before_publish()
    receipt = artifact('TASK',parents=dict(spec=spec['content_hash']),task_id=task_id,technical=technical,
        dependencies=dependencies,measurement=measurement,job_id=os.environ.get('SLURM_JOB_ID','local'),
        outputs=[file_ref(root/name,root=root) for name in PERSISTENT])
    write_immutable_json(root/'receipt.json',receipt)
    return completed(spec,gate,task_id,technical=technical)


def result_rows(spec,gate,*,technical=False):
    rows = []; seen = {}; population = None
    for name in ('M0HLT','OFFLINE','DIRECT_D000','COARSE_U100'):
        refs = gate['anchor']['evidence']['models'][name]
        for ref in refs.values(): validate_file_ref(ref)
        report, diag = load_json(refs['report']['path']), load_json(refs['diagnostics']['path'])
        rows.append(dict(node_id=name,origin='fixed30_reference',validation=report['validation'],
            original_validation=diag['original_validation'],input='HLT ONLY' if name in ('M0HLT','DIRECT_D000') else 'OFFLINE ORACLE'))
    for node in gate['nodes']:
        receipt = completed(spec,gate,'train_'+node['node_id'],seen,technical=technical)
        row = dict(node_id=node['node_id'],origin='alternating_gate',branch=node['branch'],stage=node['stage'],
                   coordinate=node['coordinate'],input=input_label(node),validation=None,original_validation=None)
        if receipt:
            root = Path(spec['root'])/('train_'+node['node_id'])
            p = load_npz_arrays(root/'validation_predictions.npz')
            current = (p['identities'],p['labels'],p['original_validation'])
            if population is None: population = current
            elif any(not np.array_equal(a,b) for a,b in zip(population,current)):
                raise ValueError('Compared validation populations differ')
            report = load_json(root/'training_report.json')
            proof = load_json(root/'selected_outputs.json')
            row.update(validation=report['validation'],original_validation=proof['original_validation'],
                       selected_pass=report['selected_pass'])
        rows.append(row)
    by_name = {r['node_id']:r for r in rows}
    for row in rows:
        row['recovery'] = {k:None if row[k] is None else recovery(row[k],by_name['M0HLT'][k],by_name['OFFLINE'][k])
                           for k in ('validation','original_validation')}
    return rows


def run(spec,task_id):
    from . import campaign
    gate, measured = campaign.check_spec(spec)
    done = completed(spec,gate,task_id)
    if done: return done
    task = next(t for t in gate['tasks'] if t['task_id']==task_id)
    if task['kind']=='train':
        execution(gate,measured)
        torch.cuda.reset_peak_memory_stats()
        def final_check():
            from .preflight import memory_ok
            from hlt_classification.luka_fullsim.preflight_v2 import memory_snapshot
            from hlt_classification.jetclass2_delphes.execution import gpu_identity
            memory_ok(gate,memory_snapshot(torch),gpu_identity())
            campaign.check_spec(spec)
        return execute_task(spec,gate,task_id,before_publish=final_check)
    dependencies = {}
    for dep in task['dependencies']:
        receipt = completed(spec,gate,dep)
        if receipt is None: raise ValueError('All scientific rows required regardless of quality')
        dependencies[dep] = receipt['content_hash']
    rows = result_rows(spec,gate)
    root = Path(spec['root'])/'aggregate'; root.mkdir(exist_ok=False)
    write_immutable_json(root/'results.json',artifact('RESULTS',parents=dict(spec=spec['content_hash']),rows=rows,
        single_seed=True,initialization='cold',target_payloads_retired=True,selected_checkpoints_retained=True))
    campaign.check_spec(spec)
    receipt = artifact('TASK',parents=dict(spec=spec['content_hash']),task_id=task_id,technical=False,
        dependencies=dependencies,outputs=[file_ref(root/'results.json',root=root)])
    write_immutable_json(root/'receipt.json',receipt)
    return receipt


def saved_results(spec):
    from . import campaign
    gate = load_json(spec['gate_path']); campaign.read_gate(gate)
    report = load_json(Path(gate['root'])/'preflight/receipt.json')
    from .preflight import validate_report
    validate_report(gate,report)
    campaign.validate_spec(spec,gate,report)
    rows = result_rows(spec,gate)
    if completed(spec,gate,'aggregate'):
        result = load_json(Path(spec['root'])/'aggregate/results.json')
        validate(result,'RESULTS',dict(spec=spec['content_hash']))
        if result['rows'] != rows: raise ValueError('Aggregate differs from retained predictions')
    return rows

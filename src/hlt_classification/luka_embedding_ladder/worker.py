"""Five selected native teachers, compact durable results, owned retirement."""
from pathlib import Path
import gc
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import predict, selection_key
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from hlt_classification.luka_embedding_aux.worker import storage, validate_predictions, snapshot_references
from hlt_classification.luka_attention.worker import execution
from hlt_classification.luka_attention_weight import retention
from hlt_classification.luka_fullsim import regular_cache
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from .contracts import artifact, validate, recipe
from . import training, banks, data, campaign, source


def report_valid(report, gate, node, incoming, embedding_hash, *, technical):
    validate(report, 'TRAINING_REPORT', dict(recipe=recipe()['content_hash'],
        prepared=gate['anchor']['evidence']['prepared'], logits=incoming['content_hash'], embeddings=embedding_hash))
    epochs = 1 if technical else 30
    history = report['validation_history']
    steps = math.ceil(gate['counts']['train']/128)
    if (report['node'] != node or report['technical'] is not technical
            or report['scientific_fit'] is not (not technical) or report['passes'] != epochs
            or report['selected_state'] != 'native_backbone_only' or report['initialization'] != 'fresh_coordinate_seed'
            or len(history) != epochs or [r['pass_number'] for r in history] != list(range(1,epochs+1))
            or any(r['update'] != r['pass_number']*steps or r['learning_rate'] != learning_rate(r['pass_number'])
                or r['validation']['rows'] != gate['counts']['validation']
                or set(r['losses']) != {'ce','logit_kd','auxiliary'}
                or not all(math.isfinite(x) for x in (*r['losses'].values(),r['train_seconds'],r['validation_seconds'])) for r in history)):
        raise ValueError('Fit science/history differs')
    best = max(history, key=lambda r:selection_key(r['validation'],r['update']))
    if report['selected_pass'] != best['pass_number'] or report['validation'] != best['validation']:
        raise ValueError('Wrong validation-selected checkpoint')


def fit(directory, gate, node, train, val, incoming, embeddings, *, device, technical):
    directory = retention.create_owned(Path(directory), gate['content_hash'])
    begun = time.monotonic()
    report, state = training.train(train, val, node=node, bank=incoming, embeddings=embeddings,
                                   device=device, technical=technical)
    report_valid(report, gate, node, incoming.manifest, embeddings.manifest['content_hash'], technical=technical)
    save_state(directory/'selected.pt', state)
    restored = torch.load(directory/'selected.pt', map_location='cpu', weights_only=True)
    if state.keys() != restored.keys() or any(not torch.equal(state[k], restored[k]) for k in state):
        raise ValueError('Checkpoint roundtrip differs')
    native = DelphesParticleTransformer().to(device).eval()
    native.load_state_dict(restored, strict=True)
    predictions = dict(identities=val.identities, labels=val.labels,
                      probabilities=predict(native,val,device=device,batch_size=128))
    if evaluate_probabilities(val.labels,predictions['probabilities']) != report['validation']:
        raise ValueError('Selected native model inference differs')
    atomic_publish_bytes(directory/'validation_predictions.npz',deterministic_npz_bytes(predictions))
    write_immutable_json(directory/'training_report.json',report)
    checkpoint = file_ref(directory/'selected.pt',root=directory)
    outgoing = None
    if node['coordinate'] != 'D000':
        outgoing = banks.export(native,train,teacher=node['node_id'],checkpoint=checkpoint['sha256'],
                                report=report['content_hash'],device=device,technical=technical)
        write_immutable_json(directory/'logits.json',outgoing.manifest)
    outputs = [file_ref(directory/f,root=directory) for f in ('training_report.json','validation_predictions.npz')]
    if outgoing is not None:
        outputs.append(file_ref(directory/'logits.json',root=directory))
    record = artifact('STEP',parents=dict(gate=gate['content_hash'],incoming=incoming.manifest['content_hash'],
        embeddings=embeddings.manifest['content_hash']),node=node,technical=technical,incoming=incoming.manifest,
        checkpoint=checkpoint,outputs=outputs,checkpoint_probability_roundtrip=True,
        seconds=time.monotonic()-begun,retired=outgoing is not None)
    write_immutable_json(directory/'receipt.json',record)
    if outgoing is not None:
        retention.retire(directory,[checkpoint],owner=gate['content_hash'],
                         consumer=file_ref(directory/'receipt.json'),output=directory/'retirement')
    validate_step(directory,gate,node,incoming.manifest,embeddings.manifest['content_hash'],technical=technical)
    del native, state, restored
    gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return record, outgoing


def validate_step(directory,gate,node,incoming,embedding_hash,*,technical):
    directory = Path(directory)
    record = load_json(directory/'receipt.json')
    validate(incoming,'LOGITS')
    if (incoming['teacher'] != node['teacher'] or incoming['coordinate'] != node['teacher_coordinate']
            or incoming['technical'] is not technical or incoming['role'] != 'train' or incoming['temperature'] != 2.
            or incoming['parents']['prepared'] != gate['anchor']['evidence']['prepared']
            or incoming['rows'] != gate['counts']['train']):
        raise ValueError('Incoming teacher lineage differs')
    validate(record,'STEP',dict(gate=gate['content_hash'],incoming=incoming['content_hash'],embeddings=embedding_hash))
    retired = node['coordinate'] != 'D000'
    expected = ['training_report.json','validation_predictions.npz']+(['logits.json'] if retired else [])
    if (record['node'] != node or record['incoming'] != incoming or record['technical'] is not technical
            or record['checkpoint_probability_roundtrip'] is not True or record['retired'] is not retired
            or record['checkpoint']['path'] != 'selected.pt' or [r['path'] for r in record['outputs']] != expected
            or not math.isfinite(record['seconds']) or record['seconds'] <= 0):
        raise ValueError('Step output meaning differs')
    for ref in record['outputs']:
        validate_file_ref(ref,root=directory)
    if retired:
        retention.validate_retirement(directory/'retirement',gate['content_hash'])
        intent = load_json(directory/'retirement/intent.json')
        if intent['files'] != [record['checkpoint']] or intent['consumer'] != file_ref(directory/'receipt.json'):
            raise ValueError('Retired checkpoint binding differs')
    else:
        validate_file_ref(record['checkpoint'],root=directory)
    report = load_json(directory/'training_report.json')
    report_valid(report,gate,node,incoming,embedding_hash,technical=technical)
    arrays = load_npz_arrays(directory/'validation_predictions.npz')
    validate_predictions(arrays,gate['counts']['validation'])
    if evaluate_probabilities(arrays['labels'],arrays['probabilities']) != report['validation']:
        raise ValueError('Stored predictions/metrics differ')
    outgoing = None
    if retired:
        outgoing = load_json(directory/'logits.json')
        validate(outgoing,'LOGITS',dict(checkpoint=record['checkpoint']['sha256'],
            report=report['content_hash'],prepared=gate['anchor']['evidence']['prepared']))
        if (outgoing['teacher'] != node['node_id'] or outgoing['coordinate'] != node['coordinate']
                or outgoing['role'] != 'train' or outgoing['temperature'] != 2. or outgoing['technical'] is not technical
                or any(outgoing[k] != incoming[k] for k in ('rows','identities_sha256','labels_sha256'))):
            raise ValueError('Successor target descriptor differs')
    return report,arrays,outgoing


def anchor_bank(gate,train,*,device,technical):
    refs = gate['anchor']['evidence']['models']['U000']
    for ref in refs.values():
        validate_file_ref(ref)
    model = DelphesParticleTransformer().to(device).eval()
    model.load_state_dict(torch.load(refs['checkpoint']['path'],map_location='cpu',weights_only=True),strict=True)
    report = load_json(refs['report']['path'])
    result = banks.export(model,train,teacher='U000',checkpoint=refs['checkpoint']['sha256'],
                          report=report['content_hash'],device=device,technical=technical)
    del model
    return result


def comparison(gate,arrays_by_name):
    directory = Path(gate['root'])/'preflight/references'
    ix = load_npz_arrays(directory/'original50k.npz')['indices']
    if (ix.dtype != np.int64 or ix.ndim != 1 or len(ix) != gate['original50k_rows']
            or len(np.unique(ix)) != len(ix) or np.any(ix < 0) or np.any(ix >= gate['counts']['validation'])):
        raise ValueError('Original50k report-only indices differ')
    arrays = {n:load_npz_arrays(directory/(n+'.npz')) for n in ('M0HLT','OFFLINE','SINGLE_EMBED')}
    arrays.update(arrays_by_name)
    base = arrays['M0HLT']
    rows = []
    for name,a in arrays.items():
        validate_predictions(a,gate['counts']['validation'])
        if not np.array_equal(a['identities'],base['identities']) or not np.array_equal(a['labels'],base['labels']):
            raise ValueError('Comparison population differs')
        rows.append(dict(model=name,validation=evaluate_probabilities(a['labels'],a['probabilities']),
            original50k=evaluate_probabilities(a['labels'][ix],a['probabilities'][ix]),
            input='single HLT' if name in ('M0HLT','SINGLE_EMBED') or name.endswith('D000') else 'OFFLINE-CONTAINING ORACLE'))
    for row in rows:
        for role in ('validation','original50k'):
            row[role+'_recovery'] = recovery(row[role],rows[0][role],rows[1][role])
    return rows


def run(spec):
    gate,measured = campaign.check_spec(spec)
    job = execution(gate,measured)
    root = Path(spec['root'])
    if (root/'ladder').exists() or (root/'receipt.json').exists():
        raise FileExistsError('Preserve partial/completed ladder; no automatic restart')
    storage(gate)
    log = PhaseLog(root,lambda:memory_snapshot(torch))
    with log.phase('load_views_and_fixed_OFFLINE_targets'):
        meta = regular_cache.metadata(gate['anchor']['evidence']['prepared_root'])
        train = data.load_views(gate,'train',authenticated=meta)
        val = data.load_views(gate,'validation',authenticated=meta)
        embeddings = data.embedding_bank(gate,train['D000'])
        if embeddings.manifest['content_hash'] != measured['embedding_hash']:
            raise ValueError('Measured embedding target changed')
        incoming = anchor_bank(gate,train['U000'],device='cuda',technical=False)
        write_immutable_json(root/'anchor_logits.json',incoming.manifest)
    results = {}
    for node in gate['nodes'][spec['arm']]:
        co = node['coordinate']
        with log.phase('fit_'+node['node_id']):
            directory = root/'ladder'/node['node_id']
            previous = incoming.manifest
            record,incoming = fit(directory,gate,node,train[co],val[co],incoming,embeddings,device='cuda',technical=False)
            _,a,_ = validate_step(directory,gate,node,previous,embeddings.manifest['content_hash'],technical=False)
            results[node['node_id']] = a
            storage(gate)
    value = artifact('RESULTS',parents=dict(spec=spec['content_hash']),rows=comparison(gate,results),counts=gate['counts'])
    write_immutable_json(root/'results.json',value)
    if source.current(gate['project_dir'],gate['source_commit']) != gate['source_snapshot']:
        raise ValueError('Source changed during execution')
    outputs = [file_ref(p,root=root) for p in sorted(root.rglob('*')) if p.is_file()
        and 'submission' not in p.relative_to(root).parts and not p.name.endswith('.out') and p.name != 'campaign_spec.json']
    write_immutable_json(root/'receipt.json',artifact('SCIENCE',parents=dict(spec=spec['content_hash']),job_id=job,outputs=outputs))
    return value


def results(spec):
    gate = load_json(spec['gate_path'])
    measured = load_json(Path(gate['root'])/'preflight/receipt.json')
    from .preflight import validate_report
    validate_report(gate,measured)
    campaign.validate_spec(spec,gate,measured)
    root = Path(spec['root'])
    receipt = load_json(root/'receipt.json')
    validate(receipt,'SCIENCE',dict(spec=spec['content_hash']))
    expected = {'results.json','anchor_logits.json'}
    for node in gate['nodes'][spec['arm']]:
        prefix = 'ladder/'+node['node_id']+'/'
        expected |= {prefix+f for f in ('receipt.json','training_report.json','validation_predictions.npz')}
    names = [r['path'] for r in receipt['outputs']]
    if len(names) != len(set(names)) or not expected <= set(names):
        raise ValueError('Incomplete science receipt coverage')
    for ref in receipt['outputs']:
        validate_file_ref(ref,root=root)
    incoming = load_json(root/'anchor_logits.json')
    refs = gate['anchor']['evidence']['models']['U000']
    if incoming['parents']['checkpoint'] != refs['checkpoint']['sha256']:
        raise ValueError('Starting U000 checkpoint differs')
    arrays = {}
    for node in gate['nodes'][spec['arm']]:
        _,a,incoming = validate_step(root/'ladder'/node['node_id'],gate,node,incoming,measured['embedding_hash'],technical=False)
        arrays[node['node_id']] = a
    result = load_json(root/'results.json')
    validate(result,'RESULTS',dict(spec=spec['content_hash']))
    if result['counts'] != gate['counts'] or result['rows'] != comparison(gate,arrays):
        raise ValueError('Comparison replay differs')
    return result

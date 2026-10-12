"""Inference-only frozen comparison, completion-last results and paired errors."""
from pathlib import Path
import gc
import hashlib
import shutil

import numpy as np
import torch

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, deterministic_npz_bytes, load_json, load_npz_arrays, write_immutable_json,
)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.context_fusion.inputs import new_model
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from hlt_classification.jetclass2_delphes.acceptance import installed_parity
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import native_mask_parity
from . import final_test_campaign as c, final_test_data as data, science_worker
from .preflight_v2 import PhaseLog, memory_snapshot


def probabilities(model_row, cache, *, device):
    checkpoint = validate_file_ref(model_row['checkpoint'])
    model = new_model(model_row['node'])
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.to(device).eval()
    try:
        values = predict(model, cache, device=device, temperature=1., batch_size=256)
        validate_file_ref(model_row['checkpoint'])
        return values
    finally:
        del model
        gc.collect()
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()


def smoke(execution, p, f, inventory, splits, *, device='cuda'):
    bundle = data.train_smoke(execution['parent'], p, f, inventory, splits)
    parity = installed_parity(bundle['U000'], device=device)
    native = native_mask_parity(bundle['U000'].batch(np.arange(4)), device=device)
    if not parity.get('passed') or native is not True:
        raise ValueError('Installed/native TRAIN parity failed before test access')
    results = []
    for row in execution['parent']['models']:
        cache = data.for_node(bundle, row['node'])
        first = probabilities(row, cache, device=device)
        second = probabilities(row, cache, device=device)
        if not np.array_equal(first, second):
            raise ValueError('Selected checkpoint TRAIN inference replay differs')
        results.append(dict(node_id=row['node']['node_id'], rows=len(cache),
            checkpoint_sha256=row['checkpoint']['sha256'], exact_probability_replay=True))
    return c.artifact('TRAIN_SMOKE', parents=dict(execution=execution['content_hash']),
        installed_parity=parity, native_mask_parity=native, models=results, final_test_accessed=False)


def bindings(execution):
    return dict(execution=execution['content_hash'], finalists=execution['parents']['finalists'])


def claim(execution):
    value = load_json(Path(execution['claim_dir'])/'claim.json')
    c.validate(value, 'ACCESS_CLAIM', bindings(execution))
    if value['final_test_accessed'] is not True:
        raise ValueError('Test execution claim is not consumed')
    witness_path = validate_file_ref(value['train_smoke'])
    if witness_path.resolve() != (Path(execution['root'])/'evaluate/train_smoke.json').resolve():
        raise ValueError('Test claim TRAIN miniature locator differs')
    c.validate_smoke(load_json(witness_path), execution)
    return value


def completed(execution, task):
    if task not in ('evaluate', 'report'):
        raise ValueError('Unknown final-test task')
    root = Path(execution['root'])
    path = root/task/'receipt.json'
    if not path.is_file():
        return None
    value = load_json(path)
    c.validate(value, 'TASK', bindings(execution))
    if (value['task'] != task or value['claim_sha256'] != claim(execution)['content_hash']
            or value['final_test_accessed'] is not True):
        raise ValueError('Final-test task/claim semantics differ')
    paths = [r['path'] for r in value['outputs']]
    if len(paths) != len(set(paths)) or any(not p.startswith(task+'/') for p in paths):
        raise ValueError('Final-test output scope differs')
    for ref in value['outputs']:
        validate_file_ref(ref, root=root)
    if task == 'evaluate':
        result = value['result']
        if (len(result['models']) != 12 or result['rows'] != execution['protocol']['counts']
                or [r['node'] for r in result['models']] != execution['protocol']['nodes']
                or result['population'] not in value['outputs']):
            raise ValueError('Frozen all-model test coverage differs')
        for row, model in zip(result['models'], execution['parent']['models']):
            if (row['prediction'] not in value['outputs'] or row['checkpoint_sha256'] != model['checkpoint']['sha256']
                    or row['selected_pass'] != model['selected_pass'] or row['validation'] != model['validation']):
                raise ValueError('Test predictions/checkpoint binding differs')
        if task+'/train_smoke.json' not in paths:
            raise ValueError('Missing pre-unsealing TRAIN miniature')
        witness = load_json(root/task/'train_smoke.json')
        c.validate_smoke(witness, execution)
    else:
        parent = completed(execution, 'evaluate')
        if (parent is None or value['evaluation_sha256'] != parent['content_hash']
                or task+'/results.json' not in paths or load_json(root/task/'results.json') != value['result']):
            raise ValueError('Final-test report/evaluation lineage differs')
        result = value['result']
        c.validate(result, 'RESULTS', dict(evaluation=parent['content_hash'], **bindings(execution)))
        if (result['jets'] != execution['protocol']['counts'] or result['final_test_accessed'] is not True
                or result['test_selected_winner'] is not False or len(result['rows']) != 12
                or result['identity_sha256'] != parent['result']['identity_sha256']):
            raise ValueError('Final-test report scope differs')
        metrics = {r['node']['node_id']: r['test'] for r in parent['result']['models']}
        for row, frozen in zip(result['rows'], parent['result']['models']):
            if (any(row[k] != frozen[k] for k in ('node', 'selected_pass', 'validation', 'test'))
                    or row['recovery'] != recovery(row['test'], metrics['M0HLT'], metrics['OFFLINE'])):
                raise ValueError('Final-test report rows differ from evaluated checkpoints')
    return value


def publish(execution, task, result, **fields):
    root = Path(execution['root'])
    value = c.artifact('TASK', parents=bindings(execution), task=task, result=result,
        claim_sha256=claim(execution)['content_hash'], final_test_accessed=True,
        outputs=[file_ref(p, root=root) for p in sorted((root/task).rglob('*')) if p.is_file()], **fields)
    write_immutable_json(root/task/'receipt.json', value)
    return completed(execution, task)


def evaluate(path):
    execution, _, p, f, inventory, splits = c.authenticate(path)
    if completed(execution, 'evaluate') is not None:
        return completed(execution, 'evaluate')
    parent, evidence = load_json(execution['parent']['spec']['path']), load_json(execution['parent']['preflight']['path'])
    job = science_worker.execution(parent, evidence)
    root = Path(execution['root'])
    if Path(execution['claim_dir']).exists():
        raise PermissionError('Already consumed test claim; no duplicate execution')
    if shutil.disk_usage(root).free < 2*2**30:
        raise OSError('Need at least 2 GiB free for immutable test predictions and reports')
    directory = root/'evaluate'
    directory.mkdir(exist_ok=False)
    log = PhaseLog(directory, lambda: memory_snapshot(torch))
    with log.phase('train_only_checkpoint_and_view_smoke'):
        witness = smoke(execution, p, f, inventory, splits)
        write_immutable_json(directory/'train_smoke.json', witness)
    check_headroom(execution, evidence)
    access = data.consume(path)  # Atomic, global to the science run, before test read.
    budget = int(.15*execution['resources']['memory_mb']*1024**2)
    with log.phase('construct_locked_test_views_once'):
        bundle, arrays, meters = data.build(access, budget)
        atomic_publish_bytes(directory/'population.npz', deterministic_npz_bytes(arrays))
    check_headroom(execution, evidence)
    identity_hash = hashlib.sha256(arrays['identities'].tobytes()).hexdigest()
    rows = []
    for model in execution['parent']['models']:
        name = model['node']['node_id']
        with log.phase('inference_'+name):
            data.require(access)
            cache = data.for_node(bundle, model['node'], access)
            if not np.array_equal(cache.identities, arrays['identities']) or not np.array_equal(cache.labels, arrays['labels']):
                raise ValueError('Test model population differs')
            scores = probabilities(model, cache, device='cuda')
            metric = evaluate_probabilities(arrays['labels'], scores)
            output = directory/(name+'.npz')
            atomic_publish_bytes(output, deterministic_npz_bytes(dict(probabilities=scores)))
            if not np.array_equal(load_npz_arrays(output)['probabilities'], scores):
                raise ValueError('Test probability roundtrip differs')
            rows.append(dict(node=model['node'], selected_pass=model['selected_pass'],
                validation=model['validation'], test=metric, prediction=file_ref(output, root=root),
                checkpoint_sha256=model['checkpoint']['sha256']))
            check_headroom(execution, evidence)
    del bundle
    c.authenticate(path)
    science_worker.execution(parent, evidence)
    return publish(execution, 'evaluate', dict(models=rows, rows=len(arrays['labels']),
        identity_sha256=identity_hash, population=file_ref(directory/'population.npz', root=root),
        view_summaries=meters, job_id=job))


def check_headroom(execution, evidence):
    snap = memory_snapshot(torch)
    if (snap['peak_rss_bytes'] >= .85*execution['resources']['memory_mb']*1024**2
            or snap['gpu_peak_bytes'] >= .90*evidence['gpu']['total_memory_bytes']):
        raise MemoryError('Final-test runtime exceeded accepted headroom')


def auc_order(labels, probabilities):
    """Prepare tied ranks once; weighted AUC reuses them across file bootstraps."""
    p = np.asarray(probabilities, np.float64)
    p = p/p.sum(axis=1, keepdims=True)
    result = []
    for cls in range(11):
        order = np.argsort(p[:, cls], kind='stable')
        scores = p[order, cls]
        starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1])+1]
        result.append((order, starts, labels[order] == cls))
    return result


def weighted_auc(orderings, weights):
    aucs = []
    for order, starts, positive in orderings:
        w = weights[order]
        pos = np.add.reduceat(w*positive, starts)
        neg = np.add.reduceat(w*~positive, starts)
        if pos.sum() == 0 or neg.sum() == 0:
            return None
        aucs.append(float(np.sum(pos*(np.cumsum(neg)-.5*neg))/(pos.sum()*neg.sum())))
    return float(np.mean(aucs))


def paired_intervals(labels, predictions, files, protocol):
    groups, index = np.unique(files, return_inverse=True)
    names = ('FINAL_BRIDGE_D000', 'DIRECT_D000', 'M0HLT')
    ranks = {name: auc_order(labels, predictions[name]) for name in names}
    correct = {name: predictions[name].argmax(1) == labels for name in names}
    rng = np.random.default_rng(protocol['seed'])
    draws = {ref: dict(accuracy=[], macro_ovr_auc=[]) for ref in names[1:]}
    for _ in range(protocol['draws']):
        weights = np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))[index].astype(np.float64)
        auc = {name: weighted_auc(ranks[name], weights) for name in names}
        acc = {name: float(np.dot(weights, correct[name])/weights.sum()) for name in names}
        for ref, metrics in draws.items():
            metrics['accuracy'].append(acc[names[0]]-acc[ref])
            if auc[names[0]] is not None and auc[ref] is not None:
                metrics['macro_ovr_auc'].append(auc[names[0]]-auc[ref])
    return dict(unit='source_file', groups=len(groups), draws=protocol['draws'], seed=protocol['seed'],
        training_seed_uncertainty_included=False, event_independence_verified=False,
        contrasts={names[0]+' minus '+ref: {metric: dict(valid_draws=len(values),
            interval_95=None if not values else np.quantile(values, protocol['interval']).tolist())
            for metric, values in rows.items()} for ref, rows in draws.items()})


def report(path):
    execution, _, _, _, inventory, splits = c.authenticate(path)
    if completed(execution, 'report') is not None:
        return completed(execution, 'report')
    evaluated = completed(execution, 'evaluate')
    if evaluated is None:
        raise ValueError('Full all-model test evaluation must commit before report')
    root = Path(execution['root'])
    directory = root/'report'
    directory.mkdir(exist_ok=False)
    log = PhaseLog(directory, lambda: memory_snapshot(torch))
    result = evaluated['result']
    with log.phase('verify_predictions_and_recompute_all_metrics'):
        population = load_npz_arrays(validate_file_ref(result['population'], root=root))
        expected = data.population_arrays(inventory, splits)
        if set(population) != set(expected) or any(not np.array_equal(population[k], expected[k]) for k in expected):
            raise ValueError('Published test identities/labels differ from locked split')
        labels, ids, files = (population[k] for k in ('labels', 'identities', 'files'))
        if (len(labels) != execution['protocol']['counts'] or ids.shape != (len(labels), 32)
                or files.shape != labels.shape or len(np.unique(files)) != execution['protocol']['files']
                or hashlib.sha256(ids.tobytes()).hexdigest() != result['identity_sha256']):
            raise ValueError('Test prediction population differs')
        predictions, metrics = {}, {}
        for row in result['models']:
            name = row['node']['node_id']
            scores = load_npz_arrays(validate_file_ref(row['prediction'], root=root))['probabilities']
            metrics[name] = evaluate_probabilities(labels, scores)
            if metrics[name] != row['test']:
                raise ValueError('Test metric replay differs')
            if name in ('FINAL_BRIDGE_D000', 'DIRECT_D000', 'M0HLT'):
                predictions[name] = scores
    with log.phase('paired_source_file_bootstrap'):
        intervals = paired_intervals(labels, predictions, files, execution['protocol']['bootstrap'])
    rows = [dict(node=row['node'], selected_pass=row['selected_pass'], validation=row['validation'],
                 test=row['test'], recovery=recovery(row['test'], metrics['M0HLT'], metrics['OFFLINE']))
            for row in result['models']]
    value = c.artifact('RESULTS', parents=dict(evaluation=evaluated['content_hash'], **bindings(execution)),
        rows=rows, jets=len(labels), identity_sha256=result['identity_sha256'], paired_intervals=intervals,
        final_test_accessed=True, test_selected_winner=False,
        caveats=['frozen all-model holdout comparison', 'single training seed',
                 'cross-file event independence unverified', 'offline oracles and two-encoder models marked',
                 'provisional mm and zero-error convention', 'future test-driven tuning is exploratory'])
    write_immutable_json(directory/'results.json', value)
    c.authenticate(path)
    return publish(execution, 'report', value, evaluation_sha256=evaluated['content_hash'])


def print_results(path):
    execution, _ = c.authenticate(path)[:2]
    done = completed(execution, 'report')
    if done is None:
        raise ValueError('Final-test report is not yet committed')
    value = done['result']
    print(f"Locked Luka final test: {value['jets']:,} jets; all 12 validation-selected models; no retraining.")
    print(f"{'node':25} {'val AUC':>9} {'test acc':>10} {'test AUC':>10} {'acc rec%':>9} {'AUC rec%':>9} input")
    def fmt(x, digits=6):
        return 'n/a' if x is None else f'{x:.{digits}f}'
    for row in value['rows']:
        node, m, rec = row['node'], row['test'], row['recovery']
        category = ('single HLT' if node['context_coordinate'] is None else 'two HLT encoders') if node['deployable'] else 'OFFLINE ORACLE'
        print(f"{node['node_id']:25} {row['validation']['macro_ovr_auc']:9.6f} {m['accuracy']:10.6f} {m['macro_ovr_auc']:10.6f} "
              f"{fmt(rec['accuracy'],1):>9} {fmt(rec['macro_ovr_auc'],1):>9} {category}")
        print('  QCD rejection @50%: '+', '.join(k+'='+('censored (0 passing)' if v['zero_fpr_censored'] else
            fmt(v['rejection_at_50pct'],1)) for k, v in m['per_class'].items()))
    print('Paired 95% source-file intervals:', value['paired_intervals'])
    print('All-model comparison, not a test-selected winner. This holdout is now consumed.')

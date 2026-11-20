"""Read-only checkpoint replay against the original six-class science source.

Loaded by file location so production imports resolve to the frozen science
checkout, not to the newer checkout containing this analysis adapter.
Donors: views/training/model/artifacts at 8a25f12653d0bcfefe170273d3f3e0748b4b508f.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, nullcontext
import hashlib
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

SPEC_HASH = 'a68c689360012272a1703f67e6da160a617a2d7937450dd5d83ac0a473a5c7a4'
SCIENCE_COMMIT = '8a25f12653d0bcfefe170273d3f3e0748b4b508f'
MODELS = ('M0HLT', 'OFFLINE', 'DE000')
ROLES = ('validation', 'outside_validation')
CLASS_COUNTS = [432708, 26937, 26592, 26483, 26449, 26863]
JOBS = dict(OFFLINE='9198207', M0HLT='9198208', DE000='9198212')
POLICY = dict(population='full_validation_no_pt_cut', rows=566032,
              score='p_X_bb/(p_X_bb+p_QCD)', positive=1, negative=0,
              selection='maximum_binary_auc_then_earliest_epoch', epochs=20,
              original_batching_preserved=True, exploratory_reselection=True,
              final_test_accessed=False, final_test_sealed=True)


def check(ok, message):
    if not ok:
        raise ValueError(message)


def artifact(kind, parents, **fields):
    from hlt_classification.data.cache_contracts import with_content_hash
    return with_content_hash(dict(contract=f'LUKA_SIXCLASS_HBB_RESELECTION_{kind}/v1',
        schema_version=1, parents=parents, policy=POLICY, **fields))


def validate(value, kind, parents):
    from hlt_classification.data.cache_contracts import validate_content_hash
    validate_content_hash(value, expected_contract=f'LUKA_SIXCLASS_HBB_RESELECTION_{kind}/v1')
    check(value['parents'] == parents and value['policy'] == POLICY, 'Analysis identity differs')
    return value


def binary_metrics(labels, probabilities):
    """Pairwise AUC, not X_bb-vs-all AUC and not a mean of regional AUCs."""
    from sklearn.metrics import roc_auc_score
    labels, p = np.asarray(labels), np.asarray(probabilities, dtype=np.float64)
    check(p.shape == (len(labels), 6) and labels.ndim == 1, 'Six-class probability shape differs')
    check(labels.dtype.kind in 'iu' and np.all((0 <= labels) & (labels < 6)), 'Invalid labels')
    check(np.isfinite(p).all() and np.all((0 <= p) & (p <= 1))
          and np.allclose(p.sum(1), 1, rtol=0, atol=1e-6), 'Invalid probabilities')
    keep = (labels == 0) | (labels == 1)
    pair = p[keep, :2]
    check(np.all(pair.sum(1) > 0), 'Undefined Hbb/QCD score')
    scores = pair[:, 1] / pair.sum(1)
    positive = labels[keep] == 1
    check(positive.any() and (~positive).any(), 'Both Hbb and QCD are required')
    signal = np.sort(scores[positive])[::-1]
    background = scores[~positive]
    result = dict(auc=float(roc_auc_score(positive, scores)),
                  signal_count=len(signal), background_count=len(background))
    for key, efficiency in (('r50', .5), ('r30', .3)):
        threshold = float(signal[int(np.ceil(efficiency * len(signal))) - 1])
        passing = int(np.count_nonzero(background >= threshold))
        result[key] = dict(rejection=len(background) / passing if passing else None,
            background_passing=passing, threshold=threshold,
            signal_passing=int(np.count_nonzero(signal >= threshold)),
            censored=passing == 0)
    return result


def selected(rows):
    check(len(rows) == 20 and sorted(r['epoch'] for r in rows) == list(range(1, 21)),
          'Selection requires every epoch exactly once')
    check(all(np.isfinite(r['metrics']['auc']) and 0 <= r['metrics']['auc'] <= 1 for r in rows),
          'Invalid selection AUC')
    return max(rows, key=lambda r: (r['metrics']['auc'], -r['epoch']))


def recovery(values):
    gap = values['OFFLINE'] - values['M0HLT']
    return 100 * (values['DE000'] - values['M0HLT']) / gap if gap > 0 else None


def validation_arguments(spec, loaded, stage):
    """Never pass TRAIN, test or unused rows to the particle reader."""
    from hlt_classification.luka_sixclass_balanced_oscar import contracts as c, views
    population, foundation, members, roles, _, _ = loaded
    val_only_roles = [np.where(r == 1, 1, 255).astype(np.uint8) for r in roles]
    for arg in views.arguments(spec['dataset'], foundation, members, val_only_roles,
                               c.COORDINATES[stage], population['content_hash']):
        check(np.isin(arg[5], (1, 2, 255)).all(), 'Forbidden particle role')
        yield arg


def load_validation(spec, loaded, stage, *, workers=4):
    """Original ordered, bounded reader; only the two validation caches exist."""
    from hlt_classification.luka_sixclass_balanced_oscar import views, contracts as c
    from hlt_classification.jetclass2_delphes.cache import RamCache
    from hlt_classification.data.cache_contracts import sha256_file
    check(workers in (1, 4), 'Use one or four preparation workers')
    blocks = {r: [] for r in ROLES}
    used = count = 0

    def accept(result):
        nonlocal used, count
        built, mapping = result
        check(set(built) <= set(ROLES) and not mapping, 'Unexpected input role/coordinate')
        for role, block in built.items():
            blocks[role].append(block)
            used += block.nbytes + len(block.labels) * 40
        count += 1
        check(used + 2 * 2**30 < spec['memory_mb'] * 1024**2 * .65, 'Input memory budget exceeded')
        if count % 100 == 0:
            print(f'HBB validation view={stage} chunks={count} resident_GiB={used/2**30:.2f}', flush=True)

    jobs = validation_arguments(spec, loaded, stage)
    context = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
        initializer=views.initialize_worker) if workers > 1 else nullcontext()
    with context as pool:
        if pool is None:
            try:
                for arg in jobs:
                    accept(views.work(arg, retain_reader=True))
            finally:
                views.close_reader()
        else:
            pending = deque()
            for _ in range(workers):
                arg = next(jobs, None)
                if arg is not None:
                    pending.append(pool.submit(views.process_work, arg))
            while pending:
                accept(pending.popleft().result())
                arg = next(jobs, None)
                if arg is not None:
                    pending.append(pool.submit(views.process_work, arg))
    for index in sorted({b.file_index for bb in blocks.values() for b in bb}):
        record = loaded[1]['files'][index]
        check(sha256_file(Path(spec['dataset']) / record['path']) == record['sha256'],
              'Raw source changed during input build')
    caches = {r: RamCache(bb, role='validation', coordinate_name=c.COORDINATES[stage],
                        foundation_sha256=loaded[0]['content_hash']) for r, bb in blocks.items()}
    for role, cache in caches.items():
        check(np.array_equal(cache.identities, loaded[4][role+'_identities'])
              and np.array_equal(cache.labels, loaded[4][role+'_labels']), 'Validation row order differs')
    return caches


def check_population(arrays):
    ids = np.concatenate([arrays[r+'_identities'] for r in ROLES])
    labels = np.concatenate([arrays[r+'_labels'] for r in ROLES])
    check([len(arrays[r+'_labels']) for r in ROLES] == [328589, 237443], 'Wrong validation populations')
    check(ids.shape == (566032, 32) and ids.dtype == np.uint8, 'Invalid validation IDs')
    check(len(np.unique(ids, axis=0)) == len(ids), 'Overlapping/duplicate validation rows')
    check(np.bincount(labels, minlength=6).tolist() == CLASS_COUNTS, 'Validation class counts differ')
    return dict(identities_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                class_counts=CLASS_COUNTS, rows=len(ids))


def native_from_epoch(stage, receipt, *, device):
    import torch
    from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
    from hlt_classification.luka_sixclass_balanced_oscar import model, contracts as c
    state = torch.load(validate_file_ref(receipt['state']), map_location='cpu', weights_only=True)
    check(state['epoch'] == receipt['epoch'] and state['update'] == receipt['epoch'] * c.STEPS,
          'Checkpoint epoch/update binding differs')
    student = model.new_model(stage)
    student.load_state_dict(state['student'], strict=True)
    return student.deployed().to(device).eval()


def check_saved_predictions(result, caches, probabilities, arrays):
    from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
    for role, key in zip(ROLES, ('predictions', 'outside_predictions')):
        with np.load(validate_file_ref(result['receipt']['files'][key]), allow_pickle=False) as saved:
            check(set(saved.files) == {'identities', 'labels', 'pt', 'probabilities'}, 'Prediction inventory differs')
            check(np.array_equal(saved['identities'], caches[role].identities)
                  and np.array_equal(saved['labels'], caches[role].labels)
                  and np.array_equal(saved['pt'], arrays[role+'_pt']), 'Published prediction population differs')
            check(np.array_equal(saved['probabilities'], probabilities[role]),
                  'Selected-checkpoint replay differs from published probabilities: '+role)


@contextmanager
def execution_lock(output):
    """OS releases the advisory lock on exit, including timeout/crash."""
    import fcntl  # Oscar/Linux only; no persistent claim blocks a safe resume.
    with (output / 'evaluation.lock').open('a') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run(spec_path, output, audit_source):
    import torch
    from hlt_classification.data.cache_contracts import load_json, write_immutable_json
    from hlt_classification.provenance import capture_source_snapshot
    from hlt_classification.luka_new_fullstat_hydra import lineage
    from hlt_classification.luka_sixclass_balanced_oscar import campaign, population, artifacts

    spec = load_json(spec_path)
    check(spec['content_hash'] == SPEC_HASH and spec['commit'] == SCIENCE_COMMIT, 'Wrong science campaign')
    check(Path(spec_path).resolve() == Path(spec['root'])/'campaign_spec.json', 'Wrong spec locator')
    campaign.check(spec)
    check(torch.cuda.is_available() and torch.cuda.device_count() == 1
          and torch.cuda.is_bf16_supported() and torch.cuda.get_device_name(0) == 'NVIDIA L40S',
          'Use one L40S with the registered BF16 runtime')
    check(os.environ.get('SLURM_JOB_ID', '').isdigit(), 'Run inside the GPU analysis allocation')
    check(os.environ.get('PYTHONNOUSERSITE') == '1' and
          os.environ.get('LD_LIBRARY_PATH', '').split(':')[0] == spec['environment_dir']+'/lib',
          'Use the registered runtime helper')
    source = capture_source_snapshot(audit_source, require_clean=True)
    output = Path(output).resolve()
    check(output.is_relative_to(Path('/oscar/home/rlyang/hlt_classification/reports')),
          'Analysis output must be separate from checkpoints/data/source')
    output.mkdir(parents=True, exist_ok=True)
    with execution_lock(output):
        if (output/'report.json').exists():
            raise FileExistsError('Analysis already published; read report.json')
        torch.set_num_threads(1)
        print('Authenticating original source, results and 20-epoch checkpoint histories...', flush=True)
        memo = {}
        results = {s: artifacts.completed(spec, s, memo=memo) for s in MODELS}
        for stage, result in results.items():
            check(result['receipt']['job_id'] == JOBS[stage], 'Unexpected original training job')
        check(lineage.verify(spec['dataset'], spec['project'], spec['population']['parents']['foundation'])
              == spec['foundation_producer'], 'Population producer drift')
        loaded = population.load(spec['dataset'], spec['population_root'])
        check(loaded[0] == spec['population'], 'Registered population differs')
        rows_proof = check_population(loaded[4])
        audit = artifact('SPEC', dict(campaign=SPEC_HASH, source=source['content_hash']),
            source=source, original_source=spec['source'], runtime=spec['environment'],
            population=rows_proof, results={s:r['receipt']['content_hash'] for s,r in results.items()})
        spec_file = output/'analysis_spec.json'
        if spec_file.exists():
            check(load_json(spec_file) == audit, 'Resume analysis specification differs')
        else:
            write_immutable_json(spec_file, audit)
        histories = {}
        # The two HLT-only models share exactly the same resident inputs.
        for group in (('M0HLT', 'DE000'), ('OFFLINE',)):
            caches = load_validation(spec, loaded, group[0])
            for stage in group:
                histories[stage] = evaluate_stage(spec, audit, stage, results[stage], caches, loaded[4], output)
            del caches
        chosen = {s: selected(histories[s]) for s in MODELS}
        original = {s: next(r for r in histories[s] if r['epoch'] == results[s]['report']['selected_pass'])
                    for s in MODELS}
        for folder in (spec['project'], audit_source):
            want = spec['source'] if str(folder) == spec['project'] else source
            check(capture_source_snapshot(folder, require_clean=True) == want, 'Source changed during evaluation')
        report = artifact('REPORT', dict(analysis=audit['content_hash']),
            epoch_history=histories, original_selection=original, hbb_selection=chosen,
            original_recovery=recovery({s:r['metrics']['auc'] for s,r in original.items()}),
            hbb_recovery=recovery({s:r['metrics']['auc'] for s,r in chosen.items()}))
        write_immutable_json(output/'report.json', report)
        print_report(report)
        print('Saved:', output/'report.json', flush=True)


def evaluate_stage(spec, audit, stage, result, caches, arrays, output):
    from hlt_classification.data.cache_contracts import load_json, write_immutable_json
    from hlt_classification.luka_sixclass_balanced_oscar import artifacts, training, reporting
    _, commits = artifacts.history(spec, stage, result['receipt']['fit_parents'], states=False)
    check(len(commits) == 20, 'All 20 epoch checkpoints are required')
    selected_epoch = result['report']['selected_pass']
    order = [selected_epoch] + [e for e in range(1, 21) if e != selected_epoch]
    rows = []
    labels = np.concatenate([caches[r].labels for r in ROLES])
    for epoch in order:
        receipt = commits[epoch-1]
        parents = dict(analysis=audit['content_hash'], epoch_commit=receipt['content_hash'])
        target = output/stage/f'{epoch:02d}.json'
        # Always replay the original selection on resume to recheck this runtime/view.
        if target.exists() and epoch != selected_epoch:
            row = validate(load_json(target), 'EPOCH', parents)
            check(row['stage'] == stage and row['epoch'] == epoch, 'Resume epoch differs')
        else:
            begun = time.monotonic()
            native = native_from_epoch(stage, receipt, device='cuda:0')
            probabilities = {r: training.predict(native, caches[r], 'cuda:0') for r in ROLES}
            check(reporting.evaluate_probabilities(caches['validation'].labels, probabilities['validation'])
                  == receipt['row']['validation'], 'In-range epoch metrics did not reproduce')
            if epoch == selected_epoch:
                check_saved_predictions(result, caches, probabilities, arrays)
            metric = binary_metrics(labels, np.concatenate([probabilities[r] for r in ROLES]))
            row = artifact('EPOCH', parents, stage=stage, epoch=epoch, metrics=metric,
                           original_selection_reproduced=epoch == selected_epoch,
                           in_range_metrics_reproduced=True,
                           analysis_job=os.environ['SLURM_JOB_ID'], seconds=time.monotonic()-begun)
            if target.exists():
                saved = validate(load_json(target), 'EPOCH', parents)
                check(saved['metrics'] == metric and saved['stage'] == stage and saved['epoch'] == epoch,
                      'Resumed metric replay differs')
                row = saved
            else:
                target.parent.mkdir(exist_ok=True)
                write_immutable_json(target, row)
            del native, probabilities
        rows.append(row)
        print(f"{stage:7s} epoch {epoch:2d}/20 full-validation Hbb/QCD AUC {row['metrics']['auc']:.8f}", flush=True)
    return sorted(rows, key=lambda r:r['epoch'])


def print_report(report):
    print('\nFull validation: 566,032 jets; Hbb/QCD AUC uses 26,937 Hbb + 432,708 QCD.')
    for title, key, rec in [('Original macro-AUC-selected epochs', 'original_selection', 'original_recovery'),
                            ('New full-validation Hbb-AUC-selected epochs', 'hbb_selection', 'hbb_recovery')]:
        print('\n'+title)
        print(f"{'Model':<9} {'Epoch':>5} {'Hbb AUC':>12} {'Hbb R50':>12} {'Hbb R30':>12}")
        for stage in MODELS:
            row=report[key][stage];m=row['metrics']
            r=[f"{m[k]['rejection']:,.1f}" if m[k]['rejection'] is not None else 'censored' for k in ('r50','r30')]
            print(f"{stage:<9} {row['epoch']:5d} {m['auc']:12.8f} {r[0]:>12} {r[1]:>12}")
        text=f"{report[rec]:.2f}%" if report[rec] is not None else 'unresolved (nonpositive anchor gap)'
        print('DE000 Hbb AUC recovery:',text)
    print('\nExploratory validation reselection; original results unchanged. No training or final-test inference.')

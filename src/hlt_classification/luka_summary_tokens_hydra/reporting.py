"""Read-only matched summary-token contrasts, deployment costs and provisional epochs."""
from pathlib import Path
import math
import subprocess
import numpy as np
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.luka_selective_teacher_hydra.reporting import recovery
from .contracts import FITS, artifact, validate, term_names, cell, recipe
from . import inputs, submission, training
from hlt_classification.luka_capacity_hydra.complementarity import align


def states(spec):
    jobs = submission.accepted_jobs(spec, 'science')
    if not jobs:
        return {}
    run = subprocess.run(['sacct', '-X', '-n', '-P', '-j', ','.join(jobs.values()), '-o', 'JobIDRaw,State,ExitCode'],
                         check=True, capture_output=True, text=True)
    rows = {line.split('|')[0]: line.split('|')[1:3] for line in run.stdout.splitlines() if len(line.split('|')) >= 3}
    return {task: rows.get(job, ['ACCOUNTING_PENDING', '']) for task, job in jobs.items()}


def references(spec):
    b = inputs.Bundle(spec['bundle_root'])
    arrays = {k: load_npz_arrays(b.path(f'reference/{k}/predictions.npz')) for k in ('M0HLT', 'OFFLINE')}
    arrays['U100_SPORC'] = load_npz_arrays(b.path('reference/matched/predictions.npz'))
    return b, arrays


def contrasts(aucs):
    """Missing controls stay missing, never substituted with historical fits."""
    result = {}
    for seed in (0, 1):
        def difference(a, b):
            return aucs[a]-aucs[b] if a in aucs and b in aucs else None
        row = {}
        for a, b in (('E','D'), ('C','A'), ('D','B'), ('E','B'), ('B','A'), ('E','C')):
            row[a+'-'+b] = difference(a+f'_s{seed}', b+f'_s{seed}')
        a, b = row['E-C'], row['B-A']
        row['(E-C)-(B-A)'] = a-b if a is not None and b is not None else None
        result[str(seed)] = row
    summary = {}
    for key in result['0']:
        values = [result[str(s)][key] for s in (0, 1)]
        if all(v is not None for v in values):
            summary[key] = dict(mean=float(np.mean(values)), minimum=min(values), maximum=max(values), seeds=2)
    return dict(per_seed=result, two_seed_descriptive=summary)


def results(spec):
    from .worker import receipt
    bundle, arrays = references(spec); epochs = {}; deployments = {}; pending = []
    for task in FITS:
        folder = Path(spec['root'])/task
        if not (folder/'receipt.json').exists():
            pending.append(task); continue
        receipt(spec, task)
        r, arrays[task] = training.validate_fit(folder/'fit', spec, task)
        if r['parents'] != inputs.fit_parents(spec, task):
            raise ValueError('Completed target lineage differs')
        epochs[task] = r['selected_epoch']
        deployments[task] = dict(profile=load_json(folder/'fit/deployment.json'),
                                 full_train_seconds=sum(e['train_seconds'] for e in r['history']))
    base = arrays['M0HLT']
    if len(base['labels']) != spec['counts']['validation']:
        raise ValueError('Full validation population differs')
    arrays = {k: align(base, a) for k, a in arrays.items()}
    ix = load_npz_arrays(bundle.path('reference/historical/subset.npz'))['indices']
    if (ix.dtype != np.int64 or ix.shape != (spec['original50k_rows'],) or len(np.unique(ix)) != len(ix)
            or ix.min() < 0 or ix.max() >= len(base['labels'])):
        raise ValueError('Invalid overlapping historical subset')
    rows = []; comparisons = {}
    print('HLT-only summary-token study: A/B one summary; C/D/E three shared-weight summaries; seeds0/1.')
    print('Historical recovery anchors. Exploratory selected-checkpoint validation; original50k overlaps; no independent test.')
    for role, subset in (('validation', slice(None)), ('original50k', ix)):
        m = {n: evaluate_probabilities(a['labels'][subset], a['probabilities'][subset])['macro_ovr_auc'] for n, a in arrays.items()}
        print('\n'+role)
        print(f'{"model":16} {"ep":>3} {"AUC":>12} {"recovery%":>10} {"delta CE":>12} {"delta B":>12}')
        for name, auc in m.items():
            dc = du = None
            if name in FITS:
                c = cell(name); suffix = '_s'+str(c['seed'])
                ce, u = ('A' if c['summaries'] == 1 else 'C')+suffix, 'B'+suffix
                dc = auc-m[ce] if ce in m else None; du = auc-m[u] if u in m else None
            row = dict(model=name, role=role, epoch=epochs.get(name), auc=auc,
                recovery=recovery(auc, m['M0HLT'], m['OFFLINE']), delta_matched_CE=dc, delta_matched_B=du)
            rows.append(row)
            fmt = lambda x: '--' if x is None else f'{x:+.8f}'
            print(f'{name:16} {str(row["epoch"] or "-"):>3} {auc:12.8f} {row["recovery"]:10.1f} {fmt(dc):>12} {fmt(du):>12}')
        comparisons[role] = contrasts(m)
        for seed, differences in comparisons[role]['per_seed'].items():
            print('Matched contrasts seed'+seed+': '+str(differences))
        print('Two-seed descriptive summary: '+str(comparisons[role]['two_seed_descriptive']))
    for task, value in deployments.items():
        profile = value['profile']
        print(f'{task}: deployed_parameters={profile["parameters"]:,} batch={profile["batch_size"]} '
              f'mean_forward_ms={1000*profile["mean_latency_seconds"]:.3f} train_hours={value["full_train_seconds"]/3600:.3f}')
    print('Latency includes transfer and current shared-node conditions; not a production latency certification.')
    status = states(spec)
    for task in pending:
        print(task+': '+str(status.get(task, ['NOT_SUBMITTED_OR_UNCONFIRMED', ''])))
    return artifact('RESULTS', parents=dict(spec=spec['content_hash']), rows=rows, contrasts=comparisons,
        deployments=deployments, pending=pending, states=status)


def progress(spec):
    _, arrays = references(spec)
    base = arrays['M0HLT']; arrays = {k: align(base, v) for k, v in arrays.items()}
    m = {k: evaluate_probabilities(a['labels'], a['probabilities'])['macro_ovr_auc'] for k, a in arrays.items()}
    best = {}; status = states(spec)
    print('Provisional full-validation epochs. Historical recovery anchors; selected checkpoints are exploratory.')
    for task in FITS:
        paths = sorted((Path(spec['root'])/task/'fit').glob('epoch-*.json'))
        print('\n'+task+' '+str(cell(task)))
        if not paths:
            print(str(status.get(task, ['NO_PUBLISHED_EPOCH', '']))); continue
        parents = inputs.fit_parents(spec, task); rows = []
        for e, path in enumerate(paths, 1):
            row = load_json(path); validate(row, 'EPOCH', parents)
            if (e > 30 or row['task'] != task or row['epoch'] != e or row['technical'] is not False
                    or row['validation']['rows'] != spec['counts']['validation'] or set(row['weighted_losses']) != term_names(task)
                    or row['update'] != e*math.ceil(spec['counts']['train']/128) or row['learning_rate'] != learning_rate(e)
                    or not np.isfinite(list(row['weighted_losses'].values())).all()):
                raise ValueError('Epoch meaning differs')
            rows.append(row); auc = row['validation']['macro_ovr_auc']
            rec = recovery(auc, m['M0HLT'], m['OFFLINE'])
            print(f'  epoch {e:2} AUC={auc:.8f} recovery={rec:.1f}% delta_historical_U100={auc-m["U100_SPORC"]:+.8f} losses={row["weighted_losses"]}')
        selected = max(rows, key=lambda r: selection_key(r['validation'], r['update']))
        best[task] = dict(epoch=selected['epoch'], auc=selected['validation']['macro_ovr_auc'])
        print('  BEST '+str(best[task]))
    print('Provisional best-epoch matched contrasts: '+str(contrasts({k: v['auc'] for k, v in best.items()})))
    return artifact('PROGRESS', parents=dict(spec=spec['content_hash']), best=best,
                    provisional_contrasts=contrasts({k: v['auc'] for k, v in best.items()}), states=status)

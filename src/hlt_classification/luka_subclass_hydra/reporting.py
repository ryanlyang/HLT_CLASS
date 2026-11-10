"""Read-only matched subclass contrasts, deployment costs and provisional epochs."""
from pathlib import Path
import math
import subprocess
import numpy as np
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.luka_selective_teacher_hydra.reporting import recovery
from .contracts import FITS, artifact, validate, term_names, cell
from hlt_classification.luka_next_supervision_hydra.reuse import check as reuse_check
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
    arrays['C0_U100_HYDRA'] = load_npz_arrays(reuse_check(spec['control'], imported=spec['imported']))
    return b, arrays


def results(spec):
    from .worker import receipt
    bundle, arrays = references(spec); epochs = {'C0_U100_HYDRA': spec['control']['report']['selected_pass']}; deployments = {}; pending = []
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
    rows = []
    print('Ten native HLT fits: CE + U100-w5 + subclass KL; existing same-seed U100-w5 control reused.')
    print('Historical recovery anchors. Exploratory selected-checkpoint validation; original50k overlaps; no independent test.')
    for role, subset in (('validation', slice(None)), ('original50k', ix)):
        m = {n: evaluate_probabilities(a['labels'][subset], a['probabilities'][subset])['macro_ovr_auc'] for n, a in arrays.items()}
        print('\n'+role)
        print(f'{"model":16} {"ep":>3} {"AUC":>12} {"recovery%":>10} {"delta C0":>12} {"recipe":>18}')
        for name, auc in m.items():
            delta = auc-m['C0_U100_HYDRA']
            row = dict(model=name, role=role, epoch=epochs.get(name), auc=auc,
                recovery=recovery(auc, m['M0HLT'], m['OFFLINE']), delta_C0=delta)
            rows.append(row)
            desc = 'subclass='+str(cell(name)['subclass_weight']) if name in FITS else 'reference'
            print(f'{name:16} {str(row["epoch"] or "-"):>3} {auc:12.8f} {row["recovery"]:10.1f} {delta:+12.8f} {desc}')
    for task, value in deployments.items():
        profile = value['profile']
        print(f'{task}: deployed_parameters={profile["parameters"]:,} batch={profile["batch_size"]} '
              f'mean_forward_ms={1000*profile["mean_latency_seconds"]:.3f} train_hours={value["full_train_seconds"]/3600:.3f}')
    print('Latency includes transfer and current shared-node conditions; not a production latency certification.')
    status = states(spec)
    for task in pending:
        print(task+': '+str(status.get(task, ['NOT_SUBMITTED_OR_UNCONFIRMED', ''])))
    return artifact('RESULTS', parents=dict(spec=spec['content_hash']), rows=rows,
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
            training.validate_diagnostics(row, task, math.ceil(spec['counts']['train']/128))
            rows.append(row); auc = row['validation']['macro_ovr_auc']
            reference = next((r['validation']['macro_ovr_auc'] for r in spec['control']['report']['validation_history']
                              if r['pass_number'] == e), None)
            same = '' if reference is None else f' delta_same_epoch={auc-reference:+.8f}'
            rec = recovery(auc, m['M0HLT'], m['OFFLINE'])
            print(f'  epoch {e:2} AUC={auc:.8f} recovery={rec:.1f}% delta_selected_C0={auc-m["C0_U100_HYDRA"]:+.8f}{same} lambda={row["subclass_coefficient"]:g} KL={row["weighted_losses"]["subclass"]:.5f} weighted_KL={row["weighted_losses"]["weighted_subclass"]:.5f} U100={row["weighted_losses"]["weighted_embedding"]:.5f}')
        selected = max(rows, key=lambda r: selection_key(r['validation'], r['update']))
        best[task] = dict(epoch=selected['epoch'], auc=selected['validation']['macro_ovr_auc'])
        print('  BEST '+str(best[task]))
    return artifact('PROGRESS', parents=dict(spec=spec['content_hash']), best=best, states=status)

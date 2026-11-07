"""Authenticated completed results and provisional epoch-by-epoch recovery."""
from pathlib import Path
import subprocess
import numpy as np
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.luka_next_supervision_hydra.reuse import check as reuse_check
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from .contracts import FITS, TEACHERS, STUDENTS, NAMES, artifact, validate, term_names
from . import inputs, submission, training


def states(spec):
    jobs = submission.accepted_jobs(spec, 'science')
    if not jobs: return {}
    r = subprocess.run(['sacct', '-X', '-n', '-P', '-j', ','.join(jobs.values()), '-o', 'JobIDRaw,State,ExitCode'],
        check=True, capture_output=True, text=True)
    rows = {x.split('|')[0]: x.split('|')[1:3] for x in r.stdout.splitlines() if len(x.split('|')) >= 3}
    return {t: rows.get(j, ['ACCOUNTING_PENDING', '']) for t, j in jobs.items()}


def references(spec):
    b = inputs.Bundle(spec['bundle_root'])
    arrays = {n: load_npz_arrays(b.path(f'reference/{n}/predictions.npz')) for n in ('M0HLT', 'OFFLINE')}
    arrays['U100_SPORC'] = load_npz_arrays(b.path('reference/matched/predictions.npz'))
    arrays['C0_U100_HYDRA'] = load_npz_arrays(reuse_check(spec['control'], imported=spec['imported']))
    return b, arrays


def recovery(auc, lo, hi):
    if not hi > lo: raise ValueError('Invalid historical recovery anchors')
    return 100*(auc-lo)/(hi-lo)


def results(spec):
    from .worker import receipt, validate_teacher, fit_parents
    bundle, arrays = references(spec); selected = {}; pending = []
    for task in FITS:
        folder = Path(spec['root'])/task
        if not (folder/'receipt.json').exists():
            pending.append(task); continue
        receipt(spec, task)
        r, arrays[task] = validate_teacher(spec, task) if task in TEACHERS else training.validate_fit(folder/'fit', spec, task)
        if r['parents'] != fit_parents(spec, task): raise ValueError('Completed target lineage differs')
        selected[task] = r['selected_epoch']
    base = arrays['M0HLT']
    for a in arrays.values():
        validate_predictions(a, spec['counts']['validation'])
        if any(not np.array_equal(a[k], base[k]) for k in ('identities', 'labels')):
            raise ValueError('Comparison populations differ')
    subset = load_npz_arrays(bundle.path('reference/historical/subset.npz'))['indices']
    if (subset.dtype != np.int64 or subset.shape != (spec['original50k_rows'],) or len(np.unique(subset)) != len(subset)
            or subset.min() < 0 or subset.max() >= len(base['labels'])):
        raise ValueError('Invalid overlapping historical subset')
    rows = []
    print('12 OFFLINE oracle teachers; 12 HLT-only CE+U100-w5+new-teacher-w5 students.')
    print('Single-seed exploratory validation; original50k overlaps validation. No independent test.')
    for role, ix in (('validation', slice(None)), ('original50k', subset)):
        m = {n: evaluate_probabilities(a['labels'][ix], a['probabilities'][ix])['macro_ovr_auc'] for n, a in arrays.items()}
        print('\n'+role)
        print(f'{"model":18} {"ep":>3} {"AUC":>12} {"recovery%":>10} {"delta C0":>12} {"delta S00":>12} role / recipe')
        for name, auc in m.items():
            oracle = name == 'OFFLINE' or name in TEACHERS
            rec = recovery(auc, m['M0HLT'], m['OFFLINE']); dc = auc-m['C0_U100_HYDRA']
            ds = auc-m['S00'] if name in STUDENTS and 'S00' in m else None
            delta_s = '--' if ds is None else f'{ds:+.8f}'
            label = NAMES[int(name[1:])] if name in FITS else 'reference'
            print(f'{name:18} {str(selected.get(name,"-")):>3} {auc:12.8f} {rec:10.1f} {dc:+12.8f} {delta_s:>12} '
                  f'{"OFFLINE oracle" if oracle else "HLT"} / {label}')
            rows.append(dict(model=name, role=role, epoch=selected.get(name), auc=auc, recovery=rec,
                             delta_u100=dc, delta_extra_teacher_control=ds, oracle=oracle))
    status = states(spec)
    for task in pending: print(task+': '+str(status.get(task, ['NOT_SUBMITTED_OR_UNCONFIRMED', ''])))
    return artifact('RESULTS', parents=dict(spec=spec['content_hash']), rows=rows, pending=pending, states=status)


def progress(spec):
    from .worker import fit_parents
    _, arrays = references(spec)
    m = {n: evaluate_probabilities(a['labels'], a['probabilities'])['macro_ovr_auc'] for n, a in arrays.items()}
    best = {}; status = states(spec)
    print('Provisional full validation. T rows are OFFLINE oracles; S rows are HLT-only students.')
    print('Recovery uses historical M0HLT/OFFLINE anchors; no inference or job changes.')
    for task in FITS:
        paths = sorted((Path(spec['root'])/task/'fit').glob('epoch-*.json'))
        print('\n'+task+' '+NAMES[int(task[1:])])
        if not paths:
            print(str(status.get(task, ['NO_PUBLISHED_EPOCH', '']))); continue
        parents = fit_parents(spec, task); rows = []
        for e, path in enumerate(paths, 1):
            row = load_json(path); validate(row, 'EPOCH', parents)
            if (row['task'] != task or row['epoch'] != e or row['technical'] is not False
                    or row['validation']['rows'] != spec['counts']['validation'] or set(row['weighted_losses']) != term_names(task)):
                raise ValueError('Epoch meaning differs')
            rows.append(row); auc = row['validation']['macro_ovr_auc']
            print(f'  epoch {e:2} AUC={auc:.8f} recovery={recovery(auc,m["M0HLT"],m["OFFLINE"]):.1f}% '
                  f'delta_C0={auc-m["C0_U100_HYDRA"]:+.8f} losses={row["weighted_losses"]}')
        row = max(rows, key=lambda r: selection_key(r['validation'], r['update']))
        best[task] = dict(epoch=row['epoch'], auc=row['validation']['macro_ovr_auc'])
        print('  BEST '+str(best[task]))
    return artifact('PROGRESS', parents=dict(spec=spec['content_hash']), best=best, states=status)

"""Fixed HLT probability ensembles and conditional paired bootstrap diagnostics."""
from itertools import combinations
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from hlt_classification.data.cache_contracts import load_npz_arrays
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.schema import CLASS_NAMES
from .contracts import MEMBERS, ENSEMBLES, artifact, recipe
from . import inputs, reuse


def align(reference, arrays):
    """Reject duplicates/missing/extra identities; reorder without silently dropping jets."""
    n = len(reference['labels'])
    validate_predictions(reference, n); validate_predictions(arrays, n)
    reference_ids = list(map(bytes, reference['identities']))
    ids = list(map(bytes, arrays['identities']))
    if len(set(ids)) != n or len(set(reference_ids)) != n or set(ids) != set(reference_ids):
        raise ValueError('Prediction identities differ or repeat')
    lookup = {k: i for i, k in enumerate(ids)}
    order = np.array([lookup[k] for k in reference_ids], np.int64)
    result = {k: v[order] for k, v in arrays.items()}
    if not np.array_equal(result['labels'], reference['labels']):
        raise ValueError('Identity-aligned labels differ')
    return result


def metrics(labels, p):
    value = evaluate_probabilities(labels, p)
    normalized = p.astype(np.float64)
    normalized /= normalized.sum(axis=1, keepdims=True)
    value['ovr_auc_per_class'] = {name: float(roc_auc_score(labels == k, normalized[:, k])) for k, name in enumerate(CLASS_NAMES)}
    value['confusion_matrix_true_rows'] = confusion_matrix(labels, p.argmax(1), labels=np.arange(11)).tolist()
    return value


class RankedAUC:
    """Exact weighted OVR AUC with cached score ordering and half-credit ties."""
    def __init__(self, labels, probabilities):
        p = probabilities.astype(np.float64)
        p /= p.sum(axis=1, keepdims=True)
        self.tables = []
        for k in range(11):
            order = np.argsort(p[:, k], kind='stable')
            sorted_scores = p[order, k]
            starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1])+1]
            self.tables.append((order, starts, labels[order] == k))

    def __call__(self, weights):
        aucs = []
        for order, starts, positive in self.tables:
            w = weights[order]
            pos = np.add.reduceat(w*positive, starts).astype(np.float64)
            neg = np.add.reduceat(w*~positive, starts).astype(np.float64)
            if pos.sum() <= 0 or neg.sum() <= 0:
                raise ValueError('Bootstrap dropped a required class')
            aucs.append(float(np.dot(pos, np.cumsum(neg)-.5*neg)/(pos.sum()*neg.sum())))
        return float(np.mean(aucs))


def bootstrap(labels, probabilities, *, replicates, seed):
    if type(replicates) is not int or replicates < 2:
        raise ValueError('At least two bootstrap replicates required')
    tables = {k: RankedAUC(labels, p) for k, p in probabilities.items()}
    groups = [np.flatnonzero(labels == k) for k in range(11)]
    if any(len(g) == 0 for g in groups):
        raise ValueError('All classes required for stratified bootstrap')
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in tables}
    for repetition in range(replicates):
        weights = np.zeros(len(labels), np.int64)
        for group in groups:
            weights[group] = np.bincount(rng.integers(len(group), size=len(group)), minlength=len(group))
        for k, table in tables.items():
            samples[k].append(table(weights))
        if (repetition+1) % 100 == 0:
            print(f'CAPACITY paired jet bootstrap {repetition+1}/{replicates}', flush=True)
    return {k: np.asarray(v) for k, v in samples.items()}


def mistakes(labels, probabilities):
    result = {}
    for a, b in combinations(MEMBERS, 2):
        pa, pb = probabilities[a].argmax(1), probabilities[b].argmax(1)
        ca, cb = pa == labels, pb == labels
        def counts(ix):
            return dict(rows=int(np.count_nonzero(ix)), both_correct=int(np.sum(ca[ix]&cb[ix])),
                only_first_correct=int(np.sum(ca[ix]&~cb[ix])), only_second_correct=int(np.sum(~ca[ix]&cb[ix])),
                both_wrong=int(np.sum(~ca[ix]&~cb[ix])), predicted_class_disagreements=int(np.sum(pa[ix] != pb[ix])))
        result[a+'_'+b] = dict(first=a, second=b, all=counts(np.ones(len(labels), bool)),
            per_class={n: counts(labels == k) for k, n in enumerate(CLASS_NAMES)})
    return result


def compute(arrays, anchors, subset, *, replicates=500, seed=2026110709):
    if set(arrays) != set(MEMBERS) or set(anchors) != {'M0HLT', 'OFFLINE'}:
        raise ValueError('Fixed members and historical anchors required')
    base = anchors['M0HLT']; n = len(base['labels'])
    models = {k: align(base, v)['probabilities'].astype(np.float64) for k, v in arrays.items()}
    refs = {k: align(base, v)['probabilities'] for k, v in anchors.items()}
    if (subset.dtype != np.int64 or subset.ndim != 1 or not len(subset) or len(np.unique(subset)) != len(subset)
            or subset.min() < 0 or subset.max() >= n):
        raise ValueError('Invalid original50k subset')
    models.update({name: np.mean([models[k] for k in members], axis=0) for name, members in ENSEMBLES.items()})
    populations = {}
    for name, ix in (('validation', np.arange(n)), ('original50k', subset)):
        labels = base['labels'][ix]
        lo, hi = [metrics(labels, refs[k][ix])['macro_ovr_auc'] for k in ('M0HLT','OFFLINE')]
        if hi <= lo:
            raise ValueError('Historical recovery denominator is nonpositive')
        rows = {}
        for k, p in models.items():
            m = metrics(labels, p[ix])
            rows[k] = dict(metrics=m, recovery=100*(m['macro_ovr_auc']-lo)/(hi-lo))
        for k, members in ENSEMBLES.items():
            strongest = max(members, key=lambda x: rows[x]['metrics']['macro_ovr_auc'])
            rows[k].update(members=list(members), strongest_member=strongest,
                delta_each_member={x: rows[k]['metrics']['macro_ovr_auc']-rows[x]['metrics']['macro_ovr_auc'] for x in members},
                delta_strongest_member=rows[k]['metrics']['macro_ovr_auc']-rows[strongest]['metrics']['macro_ovr_auc'])
        populations[name] = dict(rows=rows, anchors=dict(M0HLT=lo, OFFLINE=hi),
                                mistakes=mistakes(labels, {k: p[ix] for k, p in models.items()}))
    samples = bootstrap(base['labels'], models, replicates=replicates, seed=seed)
    denominator = populations['validation']['anchors']['OFFLINE']-populations['validation']['anchors']['M0HLT']
    for k, members in ENSEMBLES.items():
        populations['validation']['rows'][k]['paired_intervals_vs_each_member'] = {
            member: dict(auc_delta_95pct=np.quantile(samples[k]-samples[member], [.025,.975]).tolist(),
                         recovery_point_delta_95pct=(100*np.quantile(samples[k]-samples[member], [.025,.975])/denominator).tolist())
            for member in members}
    return dict(populations=populations, bootstrap=dict(replicates=replicates, seed=seed,
        scope='full_validation_only', unit='class_stratified_jet_IID_sensitivity',
        event_independence='unverified', selection_corrected=False, training_seed_uncertainty=False,
        anchors='fixed_historical_denominator'), fitted_weights=False)


def run(spec):
    reuse.check_refs(spec['reused'])
    b = inputs.Bundle(spec['bundle_root'])
    arrays = {k: load_npz_arrays(spec['reused'][k+'_predictions']['path']) for k in MEMBERS}
    anchors = {k: load_npz_arrays(b.path(f'reference/{k}/predictions.npz')) for k in ('M0HLT','OFFLINE')}
    if any(len(v['labels']) != spec['counts']['validation'] for v in [*arrays.values(), *anchors.values()]):
        raise ValueError('Complementarity population differs')
    subset = load_npz_arrays(b.path('reference/historical/subset.npz'))['indices']
    if subset.shape != (spec['original50k_rows'],):
        raise ValueError('Historical subset size differs')
    settings = recipe()['bootstrap']
    result = artifact('COMPLEMENTARITY', parents=dict(spec=spec['content_hash'], selective=spec['parent_hash']),
        prediction_refs={k: spec['reused'][k+'_predictions'] for k in MEMBERS},
        **compute(arrays, anchors, subset, replicates=settings['replicates'], seed=settings['seed']))
    display(result)
    return result


def display(result):
    print('Fixed equal-weight HLT ensembles. Exploratory, selected-checkpoint validation; original50k overlaps.')
    print('Intervals: paired jet-IID sensitivity only; event independence unverified; no selection/seed correction.')
    for role, value in result['populations'].items():
        print('\n'+role)
        print(f'{"model":18} {"AUC":>12} {"recovery%":>10} {"accuracy":>10} {"CE":>10} {"delta best member":>18}')
        for name, row in value['rows'].items():
            delta = '--' if name in MEMBERS else f'{row["delta_strongest_member"]:+.8f}'
            m = row['metrics']
            print(f'{name:18} {m["macro_ovr_auc"]:12.8f} {row["recovery"]:10.1f} {m["accuracy"]:10.6f} {m["cross_entropy"]:10.6f} {delta:>18}')
            for member, interval in row.get('paired_intervals_vs_each_member', {}).items():
                print('  vs '+member+' paired AUC delta 95%: '+str(interval['auc_delta_95pct']))
        for name, row in value['mistakes'].items():
            print(name+' top1 correctness: '+str(row['all']))
        print('QCD rejection @50% (each model has its own signal threshold; passing counts in parentheses):')
        print(f'{"class":16} '+''.join(f'{name:>21}' for name in value['rows']))
        for cls in CLASS_NAMES[1:]:
            cells = []
            for row in value['rows'].values():
                entry = row['metrics']['per_class'][cls]; r = entry['rejection_at_50pct']
                cells.append(('censored' if r is None else f'{r:.2f}')+f' ({entry["background_passing"]})')
            print(f'{cls:16} '+''.join(f'{x:>21}' for x in cells))

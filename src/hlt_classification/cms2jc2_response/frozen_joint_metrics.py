"""Predeclared collection checks and paired FILE bootstrap, never a new selection."""
import math
import numpy as np

from . import bdz_audit_metrics as audit
from .dev_diagnostics import CONDITIONS, merge_payloads

SIDES = ('real', 'proxy0', 'proxy1', 'proxy2')
CANDIDATES = ('B_DZ', 'JOINT')
CORE = ('jet_multiplicity', 'jet_vector_pt', 'jet_mass', 'jet_width',
        'jet_charged_fraction', 'jet_pt_ratio', 'jet_axis_dr', 'particle_pt', 'particle_energy',
        'particle_d0', 'particle_dz', 'particle_d0err', 'particle_dzerr',
        'particle_d0_significance', 'particle_dz_significance')
TRACKING = tuple(n for n in CORE if n.startswith('particle_d'))
VALID = tuple('particle_valid_'+n for n in ('d0', 'dz', 'd0err', 'dzerr'))
DRAWS = 200
SEED = 20260928
MIN_GROUPS = 4
MIN_CONDITIONAL_JETS = 1000


def rules():
    return dict(core=list(CORE), overall_tv=.10, conditional_tv=.25,
        validity_tv=.05, pid_tv=.25, mean_bias=.05, correlation_difference=.10,
        pt_response_sd_ratio=[.75, 1.25], tracking_sd_ratio=[.5, 2.],
        conditional_minimum_jets=MIN_CONDITIONAL_JETS, minimum_source_files=MIN_GROUPS,
        bootstrap=dict(draws=DRAWS, seed=SEED, unit='source_file_paired', interval=[.025, .975]),
        missing_required='inconclusive', no_reselection=True)


def moments(rows):
    count = sum(r['count'] for r in rows)
    if not count:
        return dict(count=0, mean=None, sd=None)
    mean = sum(r['sum'] for r in rows)/count
    return dict(count=count, mean=mean,
                sd=math.sqrt(max(0., sum(r['sumsq'] for r in rows)/count-mean**2)))


def summaries(by_file):
    return {n: merge_payloads([by_file[f][n] for f in sorted(by_file)]) for n in CANDIDATES}


def _decide(value, interval, bounds, groups):
    lo, hi = bounds
    if value is None:
        return 'inconclusive'
    if groups < MIN_GROUPS:
        return 'rejected' if not lo <= value <= hi else 'inconclusive'
    if interval is None:
        return 'inconclusive'
    if interval[1] < lo or interval[0] > hi:
        return 'rejected'
    return 'supported' if interval[0] >= lo and interval[1] <= hi else 'inconclusive'


def assess(by_file, payload):
    """Vectorize bootstrap over additive per-file bins/moments; no raw arrays."""
    files = sorted(by_file)
    if files != sorted(payload) or not files:
        raise ValueError('Missing per-file confirmation diagnostics')
    rng = np.random.default_rng(SEED)
    draws = np.asarray([np.bincount(rng.integers(len(files), size=len(files)), minlength=len(files))
                        for _ in range(DRAWS)], float)
    weights = np.vstack((np.ones(len(files)), draws))
    checks, sparse = {}, {}

    def add(key, values, bounds, contributors, reference=None):
        a = np.asarray(values, float)
        if a.shape != (DRAWS+1,) or np.isinf(a).any():
            raise ValueError('Invalid confirmation statistic')
        point = float(a[0]) if np.isfinite(a[0]) else None
        enough = contributors >= MIN_GROUPS
        interval = (np.quantile(a[1:], [.025, .975]).tolist()
                    if enough and np.isfinite(a[1:]).all() else None)
        checks[key] = dict(value=point, interval_95=interval, bounds=list(bounds),
            contributing_files=contributors, point_pass=None if point is None else bounds[0] <= point <= bounds[1],
            status=_decide(point, interval, bounds, contributors), reference_B_DZ=reference,
            missing_bootstrap_draws=int((~np.isfinite(a[1:])).sum()))

    def stats(rows):
        # Shape: side x (point + resamples) x moments/bins. Missing groups add zero.
        bins = next((len(r['bins']) for side in rows for r in side if r is not None), 1)
        blocks = []
        support = []
        for side in rows:
            x = np.asarray([[r['count'], r['sum'], r['sumsq'], *r['bins']] if r else [0.]*(3+bins) for r in side], float)
            blocks.append(weights@x)
            support.append([r.get('covered_jets', r.get('jets', 0)) if r else 0 for r in side])
        return np.asarray(blocks), np.asarray(support)

    def values(blocks, kind):
        with np.errstate(divide='ignore', invalid='ignore'):
            count = blocks[:, :, 0]
            mean = blocks[:, :, 1]/count
            sd = np.sqrt(np.maximum(0., blocks[:, :, 2]/count-mean**2))
            if kind == 'tv':
                probabilities = blocks[:, :, 3:]/count[:, :, None]
                v = np.mean(.5*np.abs(probabilities[1:]-probabilities[0]).sum(axis=2), axis=0)
            elif kind == 'bias':
                v = np.mean(np.abs(mean[1:]-mean[0])/np.abs(mean[0]), axis=0)
                v[mean[0] == 0] = np.nan
            else:
                v = np.mean(sd[1:]/sd[0], axis=0)
                v[sd[0] == 0] = np.nan
            v[~np.isfinite(v)] = np.nan
        return v

    def rows_for(n, cohort, name):
        return [[by_file[f][n]['cells'].get(side+'/'+cohort, {}).get('variables', {}).get(name)
                 for f in files] for side in SIDES]

    for cohort in CONDITIONS:
        for name in (*CORE, *VALID) if cohort == 'all' else CORE:
            blocks, support = stats(rows_for('JOINT', cohort, name))
            eligible = np.min(support.sum(axis=1)) >= MIN_CONDITIONAL_JETS
            if cohort != 'all' and not eligible:
                sparse[f'{cohort}/{name}'] = support.sum(axis=1).tolist()
                continue
            contributors = int(np.all(support > 0, axis=0).sum())
            reference = values(stats(rows_for('B_DZ', cohort, name))[0], 'tv')[0]
            reference = float(reference) if np.isfinite(reference) else None
            limit = .25 if cohort != 'all' else .05 if name in VALID else .10
            add(f'tv/{cohort}/{name}', values(blocks, 'tv'), [0., limit], contributors, reference)
            if cohort == 'all' and name in ('jet_multiplicity', 'jet_vector_pt', 'jet_mass'):
                add('bias/'+name, values(blocks, 'bias'), [0., .05], contributors)
            if cohort == 'all' and name in ('jet_pt_ratio', *TRACKING):
                bounds = [.75, 1.25] if name == 'jet_pt_ratio' else [.5, 2.]
                add('sd_ratio/'+name, values(blocks, 'sd'), bounds, contributors)

    for pid in map(str, range(6)):
        for name in audit.NAMES:
            rows = [[payload[f]['real' if side == 'real' else 'JOINT/'+side][pid]['variables'][name]
                     for f in files] for side in SIDES]
            blocks, support = stats(rows)
            if np.min(support.sum(axis=1)) < MIN_CONDITIONAL_JETS:
                sparse['pid/'+pid+'/'+name] = support.sum(axis=1).tolist()
            else:
                add('tv/pid/'+pid+'/'+name, values(blocks, 'tv'), [0., .25],
                    int(np.all(support > 0, axis=0).sum()))

    # Registered Pearson jet-vector pairs, with undefined values retained.
    from .metrics import JET_JOINT
    k = len(JET_JOINT)
    blocks = []
    supports = []
    for side in SIDES:
        rows = [by_file[f]['JOINT']['correlations'].get(side) for f in files]
        x = np.asarray([[r['count'], *r['sum'], *np.asarray(r['cross']).ravel()] if r
                        else [0.]*(1+k+k*k) for r in rows], float)
        blocks.append(weights@x); supports.append(x[:, 0])
    blocks = np.asarray(blocks)
    with np.errstate(divide='ignore', invalid='ignore'):
        means = blocks[:, :, 1:1+k]/blocks[:, :, :1]
        cov = blocks[:, :, 1+k:].reshape(4, -1, k, k)/blocks[:, :, :1, None]-means[:, :, :, None]*means[:, :, None, :]
        sd = np.sqrt(np.maximum(0., np.diagonal(cov, axis1=2, axis2=3)))
        corr = cov/(sd[:, :, :, None]*sd[:, :, None, :])
        corr = np.clip(corr, -1, 1)
        for i in range(k):
            for j in range(i+1, k):
                v = np.mean(abs(corr[1:, :, i, j]-corr[0, :, i, j]), axis=0)
                v[~np.isfinite(v)] = np.nan
                add(f'correlation/{JET_JOINT[i]}/{JET_JOINT[j]}', v, [0., .10],
                    int(np.all(np.asarray(supports) > 1, axis=0).sum()))
    statuses = {r['status'] for r in checks.values()}
    status = 'rejected' if 'rejected' in statuses else 'supported' if statuses == {'supported'} else 'inconclusive'
    return dict(status=status, checks=checks, sparse_cells=sparse, source_files=len(files),
        bootstrap=rules()['bootstrap'], independent_groups_sufficient=len(files) >= MIN_GROUPS,
        frozen_candidate='JOINT', reselection=False, production_qualified=False, transfer_authorized=False)


def chart(histogram):
    real = histogram['cells']['real/all']['variables']
    proxies = [histogram['cells'][s+'/all']['variables'] for s in SIDES[1:]]
    result = {}
    for name in sorted(set(real).union(*(p.keys() for p in proxies))):
        r, ps = real.get(name), [p.get(name) for p in proxies]
        tvs = [audit.tv(r['bins'], p['bins']) if r and p else None for p in ps]
        result[name] = dict(real=moments([r] if r else []),
            proxy=moments([p for p in ps if p]), tv=None if any(v is None for v in tvs) else sum(tvs)/3)
    return result

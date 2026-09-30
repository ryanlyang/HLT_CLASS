"""Descriptive, explicitly unpaired cross-dataset comparisons and paired counts."""
from collections import Counter
from io import BytesIO, StringIO
import csv
import math

import numpy as np

from hlt_classification.cms2jc2_response import dev_diagnostics as d

SIDES = ('CMS_offline', 'CMS_HLT', 'JC2_offline', 'JC2_proxy')


def paired_counts(offline, proxy, hist):
    values = dict(offline=len(offline), proxy=len(proxy), delta=len(proxy)-len(offline))
    for pid in range(6):
        a, b = int((offline.category == pid).sum()), int((proxy.category == pid).sum())
        values[f'pid_{pid}_delta'] = b-a
    for name, value in values.items():
        row = hist.setdefault(name, {})
        row[str(value)] = row.get(str(value), 0)+1


def merge_counts(rows):
    result = {}
    for hist in rows:
        for name, counts in hist.items():
            result.setdefault(name, Counter()).update(counts)
    return {n: dict(sorted(v.items(), key=lambda kv: int(kv[0]))) for n, v in result.items()}


def count_summary(hist):
    result = {}
    for name, counts in hist.items():
        values = sorted((int(k), v) for k, v in counts.items())
        n = sum(v for _, v in values)
        mean = sum(k*v for k, v in values)/n
        def quantile(q):
            rank, total = max(1, math.ceil(n*q)), 0
            for k, v in values:
                total += v
                if total >= rank:
                    return k
        result[name] = dict(jets=n, mean=mean,
            sd=math.sqrt(max(0., sum(k*k*v for k, v in values)/n-mean*mean)),
            minimum=values[0][0], maximum=values[-1][0],
            quantiles={str(q): quantile(q) for q in (.5, .9, .95, .99)},
            negative_fraction=sum(v for k, v in values if k < 0)/n,
            zero_fraction=sum(v for k, v in values if k == 0)/n,
            positive_fraction=sum(v for k, v in values if k > 0)/n)
    return result


def moments(row):
    if row is None or not row['count']:
        return dict(count=0, mean=None, sd=None)
    mean = row['sum']/row['count']
    return dict(count=row['count'], covered_jets=row['covered_jets'], mean=mean,
        sd=math.sqrt(max(0., row['sumsq']/row['count']-mean*mean)),
        minimum=row['minimum'], maximum=row['maximum'],
        underflow_fraction=row['bins'][0]/row['count'], overflow_fraction=row['bins'][-1]/row['count'])


def tv(a, b):
    if a is None or b is None or not a['count'] or not b['count']:
        return None
    if len(a['bins']) != len(b['bins']):
        raise ValueError('Different frozen histogram definitions')
    return float(.5*np.abs(np.asarray(a['bins'])/a['count']-np.asarray(b['bins'])/b['count']).sum())


def joined(cms, jc2):
    payloads = []
    for payload, mapping in ((cms, {'offline': 'CMS_offline', 'real': 'CMS_HLT'}),
                             (jc2, {'offline': 'JC2_offline', 'proxy': 'JC2_proxy'})):
        payloads.append(dict(cells={mapping[k.split('/')[0]]+'/'+k.split('/')[1]: v
                            for k, v in payload['cells'].items() if k.split('/')[0] in mapping},
            correlations={mapping[k]: v for k, v in payload['correlations'].items() if k in mapping}))
    return d.merge_payloads(payloads)


def tables(payload):
    result = {}
    for cohort in d.CONDITIONS:
        table = {}
        for name in d.NAMES:
            raw = {side: payload['cells'].get(side+'/'+cohort, {}).get('variables', {}).get(name)
                   for side in SIDES}
            stats = {s: moments(r) for s, r in raw.items()}
            def shift(a, b):
                return stats[b]['mean']-stats[a]['mean'] if all(stats[s]['mean'] is not None for s in (a, b)) else None
            table[name] = dict(sides=stats, offline_population_tv=tv(raw['CMS_offline'], raw['JC2_offline']),
                hlt_population_tv=tv(raw['CMS_HLT'], raw['JC2_proxy']),
                cms_offline_to_hlt_tv=tv(raw['CMS_offline'], raw['CMS_HLT']),
                jc2_offline_to_proxy_tv=tv(raw['JC2_offline'], raw['JC2_proxy']),
                cms_mean_shift=shift('CMS_offline', 'CMS_HLT'), jc2_mean_shift=shift('JC2_offline', 'JC2_proxy'))
        result[cohort] = table
    return result


def text_table(report):
    lines = [f"TRAIN-ONLY PROXY AUDIT: {report['jets']:,} paired JetClass2 jets",
             f"CMS reference: {report['cms_reference']['cms_jets']:,} jets; saved histograms only.",
             f"Retained CMS confirmation status: {report['cms_reference'].get('qualification_status', 'see reference report')}",
             'Proxy is controlled CMS-calibrated simulation, NOT genuine detector HLT.',
             'Raw populations differ; cross-dataset TV is descriptive, not mapping closure.',
             'SD is distribution width, not uncertainty. Particle rows are particle-weighted.',
             'Particle mean deltas compare distributions, not matched individual particles.',
             'Tracking values/errors: mm; momenta: GeV; significances: dimensionless.',
             'PID: 0 charged hadron; 1 neutral hadron; 2 photon; 3 electron; 4 muon; 5 unknown.',
             '', f"{'observable':<29}"+''.join(f'{s:>25}' for s in SIDES),
             f"{'mean +/- SD':<29}"+' (valid counts and histogram flow fractions in CSV/JSON)']
    for name in d.NAMES:
        row = report['tables']['all'][name]
        vals = []
        for side in SIDES:
            r = row['sides'][side]
            vals.append('n/a' if r['mean'] is None else f"{r['mean']:.5g} +/- {r['sd']:.5g}")
        lines.append(f'{name:<29}'+''.join(f'{v:>25}' for v in vals))
    lines += ['', 'MARGINAL POPULATION DISTANCES AND WITHIN-DATASET MEAN CHANGES',
              f"{'observable':<29} {'offline TV':>12} {'HLT TV':>12} {'CMS mean delta':>16} {'JC2 mean delta':>16}"]
    for name in d.NAMES:
        row = report['tables']['all'][name]
        values = [row[n] for n in ('offline_population_tv', 'hlt_population_tv', 'cms_mean_shift', 'jc2_mean_shift')]
        text = ['n/a' if v is None else f'{v:.5g}' for v in values]
        lines.append(f'{name:<29} {text[0]:>12} {text[1]:>12} {text[2]:>16} {text[3]:>16}')
    count = report['paired_count_summary']
    a, b, change = (count[n] for n in ('offline', 'proxy', 'delta'))
    lines += ['', f"Total particles: offline={report['particle_totals']['offline']:,}; proxy={report['particle_totals']['proxy']:,}",
        f"Mean particles: offline={a['mean']:.6f}; proxy={b['mean']:.6f}; paired delta={change['mean']:+.6f}",
        f"Count decreases/unchanged/increases: {change['negative_fraction']:.2%} / {change['zero_fraction']:.2%} / {change['positive_fraction']:.2%}",
        'CMS paired count-change histograms are unavailable; only its difference of means is reported.',
        'No refit, no new selection, no validation/test particles, no native Delphes HLT or labels read.',
        'See distributions.pdf, statistics.csv and report.json for tails, conditions and counts.']
    return '\n'.join(lines)+'\n'


def csv_table(report):
    stream = StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['cohort', 'observable', 'side', 'count', 'covered_jets', 'mean', 'sd',
                     'minimum', 'maximum', 'underflow_fraction', 'overflow_fraction',
                     'offline_population_tv', 'hlt_population_tv', 'cms_mean_shift', 'jc2_mean_shift'])
    for cohort in d.CONDITIONS:
        table = report['tables'][cohort]
        for name in d.NAMES:
            row = table[name]
            for side in SIDES:
                stats = row['sides'][side]
                writer.writerow([cohort, name, side, *(stats.get(n) for n in
                    ('count', 'covered_jets', 'mean', 'sd', 'minimum', 'maximum', 'underflow_fraction', 'overflow_fraction')),
                    *(row[n] for n in ('offline_population_tv', 'hlt_population_tv', 'cms_mean_shift', 'jc2_mean_shift'))])
    return stream.getvalue()


def pdf(payload, ranges, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    stream = BytesIO()
    with PdfPages(stream, metadata={'CreationDate': None, 'ModDate': None}) as pages:
        for start in range(0, len(d.NAMES), 6):
            fig, axes = plt.subplots(3, 2, figsize=(13, 12))
            for ax, name in zip(axes.flat, d.NAMES[start:start+6]):
                edges = ranges['definitions'][name]['edges']
                flow = []
                for side in SIDES:
                    row = payload['cells'].get(side+'/all', {}).get('variables', {}).get(name)
                    if row and row['count']:
                        ax.stairs(np.asarray(row['bins'][1:-1])/row['count'], edges, label=side)
                        flow.append(f"{side}: {row['bins'][0]/row['count']:.2%}/{row['bins'][-1]/row['count']:.2%}")
                ax.set_title(name)
                ax.set_xlabel(d.transform_name(name), fontsize=7)
                ax.set_ylabel('Probability per bin (flow not renormalized)')
                if flow:
                    ax.legend(fontsize=6)
                ax.text(.02, .98, 'Under/overflow:\n'+'\n'.join(flow), transform=ax.transAxes,
                        va='top', fontsize=6)
            for ax in list(axes.flat)[len(d.NAMES[start:start+6]):]:
                ax.set_visible(False)
            fig.suptitle(f"CMS reference vs JetClass2 TRAIN ({report['jets']:,} jets)\nDifferent populations; descriptive comparison only")
            fig.tight_layout(rect=(0, 0, 1, .95)); pages.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(2, 1, figsize=(11, 8))
        for name in ('offline', 'proxy'):
            values = sorted((int(k), v) for k, v in report['paired_count_histograms'][name].items())
            axes[0].step([k for k, _ in values], [v/report['jets'] for _, v in values], where='mid', label=name)
        values = sorted((int(k), v) for k, v in report['paired_count_histograms']['delta'].items())
        axes[1].bar([k for k, _ in values], [v/report['jets'] for _, v in values])
        axes[0].set_title('Exact saved JetClass2 particle multiplicities'); axes[0].legend()
        axes[1].set_title('Exact paired proxy-HLT minus offline particle count (not CMS)')
        for ax in axes:
            ax.set_ylabel('Jet fraction'); ax.set_xlabel('Particles')
        fig.tight_layout(); pages.savefig(fig); plt.close(fig)
    return stream.getvalue()

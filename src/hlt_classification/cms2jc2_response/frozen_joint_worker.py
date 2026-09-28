"""Bounded two-candidate replay; frozen response, no fitting API."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import hashlib
from itertools import islice
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np

from . import frozen_joint_campaign as campaign, frozen_joint_data as data, frozen_joint_metrics as metrics
from . import bdz_joint_worker as jw, bdz_joint_maps as maps, bdz_worker as old
from . import bdz_audit_metrics as audit, dev_campaign as dev
from .contracts import artifact, validate, sha256_file
from .dev_data import sample_stream
from .dev_diagnostics import Histograms, conditions, merge_payloads, CONDITIONS
from .dev_parallel import _child_init, identity_digest
from .c_diagnostic_worker import historical_replay_equal
from .response import Generator
from .measurement import Measurement
from .storage import publish_bytes, GIB


def inputs(spec):
    donor = campaign.donor(spec)
    _, model, ranges, historical = jw.inputs(donor)
    joint = campaign.joint.registry(donor)['mapping']
    return model, historical['mapping'], joint, ranges


def chunk(pairs, model, historical, joint, ranges, gen=None):
    gen = gen if gen is not None else Generator(model['runtime_response'])
    result = dict(old.empty(), jets=len(pairs), identity=identity_digest(pairs), audit={})
    for pair in pairs:
        f, cohorts = pair.source_group, conditions(pair.offline)
        hs = result['by_file'].setdefault(f, {n: Histograms(ranges) for n in metrics.CANDIDATES})
        ts = result['tracking'].setdefault(f, {n: {} for n in metrics.CANDIDATES})
        ds = result['audit'].setdefault(f, {})
        audit.merge(ds, {'real': audit.particles(pair.hlt, jet=pair.identity, source_group=f, side='real')})
        real = old.metrics.tracking_row(pair.hlt)
        for n in metrics.CANDIDATES:
            hs[n].add('offline', cohorts, pair.offline, pair.offline)
            hs[n].add('real', cohorts, pair.offline, pair.hlt)
            ts[n]['real'] = old.metrics.merge(ts[n].get('real'), real)
        for replica in range(3):
            base, info = gen(pair.offline, jet=pair.identity, replica=replica, trace=True)
            for n in metrics.CANDIDATES:
                output, counters = maps.apply(base, historical, joint, n)
                maps.old.check_invariants(base, output)
                side = f'proxy{replica}'
                hs[n].add(side, cohorts, pair.offline, output)
                ts[n][side] = old.metrics.merge(ts[n].get(side), old.metrics.tracking_row(output))
                audit.merge(ds, {n+'/'+side: jw.audit_particles(output, base, historical, joint, n, pair, replica)})
                old.add_counters(result['corrections'].setdefault(n, {}), counters)
            old.add_counters(result['flags'], {k: int(bool(v)) for k, v in info['flags'].items()})
    result['by_file'] = {f: {n: h.payload() for n, h in hs.items()} for f, hs in result['by_file'].items()}
    return result


def initialize(*args):
    _child_init()
    global _args
    _args = (*args, Generator(args[0]['runtime_response']))


def process(pairs):
    return chunk(pairs, *_args)


def run_pairs(pairs, model, historical, joint, ranges, *, workers, count):
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError('Unregistered frozen worker count')
    maps.old.validate_map(historical); maps.validate_map(joint)
    result = dict(old.empty(), audit={})
    seen, digest = set(), hashlib.sha256()
    started = last = time.monotonic()
    args = model, historical, joint, ranges

    def progress():
        print(f"CMS2JC2-FROZEN jets={result['jets']}/{count} workers={workers} seconds={time.monotonic()-started:.1f}", flush=True)

    def batches():
        iterator = iter(pairs)
        while True:
            batch = tuple(islice(iterator, 8))
            if not batch:
                break
            for pair in batch:
                if pair.identity in seen:
                    raise ValueError('Duplicate confirmation jet')
                seen.add(pair.identity); digest.update(pair.identity.encode())
            yield (len(batch), identity_digest(batch)), batch

    def merge(key, row):
        nonlocal last
        if key != (row['jets'], row['identity']):
            raise ValueError('Confirmation process identity/count differs')
        old.merge_result(result, row); audit.merge(result['audit'], row['audit'])
        if time.monotonic()-last >= 15:
            progress(); last = time.monotonic()

    progress()
    if workers == 1:
        gen = Generator(model['runtime_response'])
        for key, batch in batches():
            merge(key, chunk(batch, *args, gen))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize, initargs=args) as pool:
            jobs, pending, exhausted = iter(batches()), deque(), False
            while pending or not exhausted:
                while not exhausted and len(pending) < 2*workers:
                    try:
                        key, batch = next(jobs)
                    except StopIteration:
                        exhausted = True; break
                    pending.append((key, pool.submit(process, batch)))
                if pending:
                    key, future = pending[0]
                    try:
                        row = future.result(timeout=15)
                    except TimeoutError:
                        progress(); continue
                    pending.popleft(); merge(key, row)
    if result['jets'] != count or len(seen) != count:
        raise ValueError('Incomplete confirmation shard')
    progress()
    return dict(result, ordered_identities=digest.hexdigest(), source_groups=sorted(result['by_file']))


def acceptance(ctx, spec):
    args = inputs(spec)
    with sample_stream(ctx, 'residual') as pairs:
        rows = tuple(islice(pairs, 64))
    serial = run_pairs(rows, *args, workers=1, count=len(rows))
    with Measurement() as measurement:
        parallel = run_pairs(rows, *args, workers=8, count=len(rows))
    if not historical_replay_equal(serial, parallel):
        raise ValueError('Confirmation serial/process replay differs')
    # Same fitted maps/random streams as the original five-candidate kernel.
    original = jw.run_pairs(rows, *args, mode='evaluate', workers=1, count=len(rows))
    for key in ('by_file', 'tracking'):
        restricted = {f: {n: row[n] for n in metrics.CANDIDATES} for f, row in original[key].items()}
        if not historical_replay_equal(serial[key], restricted):
            raise ValueError('Reduced worker changed frozen candidate: '+key)
    restricted = {f: {s: v for s, v in rows.items() if s == 'real' or s.split('/')[0] in metrics.CANDIDATES}
                  for f, rows in original['audit'].items()}
    if (not historical_replay_equal(serial['audit'], restricted) or serial['flags'] != original['flags']
            or not historical_replay_equal(serial['corrections'], {n: original['corrections'][n] for n in metrics.CANDIDATES})):
        raise ValueError('Reduced worker changed frozen diagnostics')
    measured = measurement.report()
    size = len(json.dumps(parallel, allow_nan=False).encode())
    return artifact('FROZEN_ACCEPTANCE', parents={'stage': spec['content_hash'],
        'reuse': spec['reuse']['content_hash'], 'protocol': campaign.protocol()['content_hash']},
        jets=len(rows), measurement=measured, result_bytes=size,
        projection=campaign.projection(measured, size, len(rows), spec['membership']),
        serial_process_parity=True, frozen_worker_replay=True, confirmation_accessed=False)


def evaluate(ctx, spec, task):
    study = campaign.validate_stage(spec, source=False)
    shard = task['params']['shard']
    membership = spec['membership']['shards'][shard]
    with data.stream(spec, study, shard=shard) as pairs:
        result = run_pairs(pairs, *inputs(spec), workers=task['cpus'], count=membership['jets'])
    if result['ordered_identities'] != membership['ordered_identities'] or result['source_groups'] != [membership['sha256']]:
        raise ValueError('Frozen shard consumed wrong identities')
    return artifact('FROZEN_SHARD', parents=campaign.parents(spec), **result, shard=shard,
        confirmation_accessed=True, selected='JOINT', reference='B_DZ', replicas=[0, 1, 2])


def figures(charts):
    """Small complete moments/width plot; full histograms retained in the report."""
    import io
    from matplotlib.figure import Figure
    import matplotlib
    names = metrics.CORE
    for first in range(0, len(names), 6):
        fig = Figure(figsize=(12, 8))
        for i, name in enumerate(names[first:first+6]):
            ax = fig.add_subplot(2, 3, i+1)
            entries = [charts['JOINT'][name]['real'], *[charts[n][name]['proxy'] for n in metrics.CANDIDATES]]
            for x, entry in enumerate(entries):
                if entry['mean'] is not None:
                    ax.errorbar(x, entry['mean'], yerr=entry['sd'], fmt='o', capsize=3)
            ax.set_xticks([0, 1, 2], ['CMS', 'B_DZ', 'JOINT'])
            ax.set_title(name, fontsize=9)
        fig.suptitle('Frozen CMS confirmation: means and distribution SD (not uncertainty)')
        fig.tight_layout()
        buf = io.BytesIO()
        with matplotlib.rc_context({'svg.hashsalt': 'cms2jc2_frozen_v1'}):
            fig.savefig(buf, format='svg', metadata={'Date': None})
        yield f'moments_{first//6}.svg', buf.getvalue()


def distributions(histograms, ranges):
    """Frozen bins including disclosed out-of-range probability; no refitting."""
    import io
    from matplotlib.figure import Figure
    import matplotlib
    names = ('jet_multiplicity', *[f'jet_count_{i}' for i in range(6)], *metrics.CORE[1:])
    for first in range(0, len(names), 6):
        fig = Figure(figsize=(14, 10))
        for i, name in enumerate(names[first:first+6]):
            ax = fig.add_subplot(2, 3, i+1)
            definition = ranges['definitions'][name]
            edges = np.asarray(definition['edges'])
            curves = [('CMS', histograms['JOINT'], ['real'])]
            curves += [(n, histograms[n], list(metrics.SIDES[1:])) for n in metrics.CANDIDATES]
            for label, payload, sides in curves:
                bins = []
                for side in sides:
                    row = payload['cells'][side+'/all']['variables'].get(name)
                    if row and row['count']:
                        bins.append(np.asarray(row['bins'], float)/row['count'])
                if len(bins) != len(sides):
                    continue
                p = np.mean(bins, axis=0)
                label += f' (under {p[0]:.2%}, over {p[-1]:.2%})'
                ax.stairs(p[1:-1], edges, label=label)
            ax.set_title(name, fontsize=10)
            ax.set_xlabel(definition['transform'], fontsize=7)
            ax.set_ylabel('probability / frozen bin')
            ax.legend(fontsize=6)
            if name.startswith('particle_d'):
                ax.set_yscale('log')
        fig.suptitle('Frozen CMS confirmation: replica-mean histograms; no independent-replica error bars')
        fig.tight_layout()
        buf = io.BytesIO()
        with matplotlib.rc_context({'svg.hashsalt': 'cms2jc2_frozen_v1'}):
            fig.savefig(buf, format='svg', metadata={'Date': None})
        yield f'distributions_{first//6}.svg', buf.getvalue()


def report(ctx, spec):
    total = dict(old.empty(), audit={})
    hashes = []
    for expected in spec['membership']['shards']:
        i = expected['index']
        row = dev.product(spec, f'jf_eval_{i:04d}', 'result')
        validate(row, 'FROZEN_SHARD', parents=campaign.parents(spec))
        if (row['shard'] != i or row['jets'] != expected['jets']
                or row['ordered_identities'] != expected['ordered_identities']
                or row['source_groups'] != [expected['sha256']] or sorted(row['by_file']) != row['source_groups']
                or row['selected'] != 'JOINT' or row['reference'] != 'B_DZ'
                or row['replicas'] != [0, 1, 2] or row['confirmation_accessed'] is not True):
            raise ValueError('Frozen shard registration differs')
        old.merge_result(total, row); audit.merge(total['audit'], row['audit']); hashes.append(row['content_hash'])
        print(f"CMS2JC2-FROZEN phase=report shards={i+1}/{len(spec['membership']['shards'])}", flush=True)
    if total['jets'] != spec['membership']['jets']:
        raise ValueError('Confirmation population incomplete')
    for f in spec['membership']['files']:
        for name in metrics.CANDIDATES:
            for side in metrics.SIDES:
                if total['by_file'][f['sha256']][name]['cells'][side+'/all']['jets'] != f['jets']:
                    raise ValueError('Per-file confirmation coverage differs')
    histograms = metrics.summaries(total['by_file'])
    charts = {n: metrics.chart(p) for n, p in histograms.items()}
    decision = metrics.assess(total['by_file'], total['audit'])
    ranges = dev.product(dev.preparation_stage(spec), 'prepare', 'ranges')
    validate(ranges, 'DEV_RANGES')
    if ranges['content_hash'] != spec['reuse']['parents']['ranges']:
        raise ValueError('Frozen plotting ranges changed')
    plots = {}
    from itertools import chain
    for name, blob in chain(figures(charts), distributions(histograms, ranges)):
        relative = f"figures/{spec['name']}/{name}"
        path = publish_bytes(Path(spec['root']), relative, blob, remaining_bytes=GIB)
        plots[name] = dict(relative=relative, sha256=sha256_file(path), bytes=len(blob))
    return artifact('FROZEN_REPORT', parents=campaign.parents(spec), **total, shard_hashes=hashes,
        decision=decision, charts=charts, figures=plots, diagnostics=audit.diagnostics(total['audit']),
        selected='JOINT', confirmation_accessed=True, production_qualified=False, transfer_authorized=False,
        reselection=False, replicas_are_independent=False, class_labels_accessed=False,
        production_six_block_qualification_performed=False, original_association_requirement_waived=False)


def dispatch(ctx, spec, task):
    if task['action'] == 'jf_acceptance':
        return acceptance(ctx, spec)
    if task['action'] == 'jf_evaluate':
        return evaluate(ctx, spec, task)
    if task['action'] == 'jf_report':
        return report(ctx, spec)
    raise ValueError('Unknown frozen confirmation task')

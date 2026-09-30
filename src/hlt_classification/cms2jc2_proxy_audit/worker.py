"""Bounded read-only shard analysis. No generation, labels or test reader."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import zip_longest
from pathlib import Path
import hashlib
import multiprocessing
import os
import time

from hlt_classification.cms2jc2_production import contracts as k, output, population
from hlt_classification.cms2jc2_response import dev_diagnostics as d
from hlt_classification.cms2jc2_response.generation_benchmark_engine import physical_digest
from hlt_classification.cms2jc2_response.provenance import environment
from . import campaign as c, diagnostics as stats


def analyze_shard(spec, study, ranges, row):
    started = time.monotonic()
    shard = next(s for s in study['shards'] if s['shard_id'] == row['shard_id'])
    if shard['role'] != 'train':
        raise PermissionError('Only train shards may enter this audit')
    receipt = k.load_json(k.checked(row['receipt']))
    target = k.safe(spec['root'], f"shards/{shard['shard_id']}.json")
    parents = {'spec': spec['content_hash'], 'receipt': receipt['content_hash']}
    if target.exists():
        saved = k.load_json(target)
        c.validate(saved, 'SHARD', parents=parents)
        if saved['jets'] != shard['jets']:
            raise ValueError('Saved shard audit count differs')
        return k.ref(target)
    # Authenticate lineage, exact registered membership, bytes, dtype and shape.
    # A second bounded traversal computes physical digest while collecting stats.
    print(f"C2JPA shard={shard['shard_id']} phase=authenticate jets={shard['jets']}", flush=True)
    output.verify_shard(study, receipt, physical=False)
    def proxies():
        for block in receipt['blocks']:
            path = k.safe(study['root'], block['relative'])
            if k.sha256_file(path) != block['sha256']:
                raise ValueError('Physical block changed during audit')
            yield from output.particles(output.arrays(path))
    offline = population.iterate(study['data_root'], study['population'], shard, study['review'])
    hist, counts, digest, jets = d.Histograms(ranges), {}, hashlib.sha256(), 0
    for pair, proxy in zip_longest(offline, proxies()):
        if pair is None or proxy is None or pair.identity != proxy[0]:
            raise ValueError('Offline/proxy identities or lengths differ')
        identity, particles = proxy
        digest.update(bytes.fromhex(physical_digest(identity, particles)))
        cohorts = d.conditions(pair.offline)
        hist.add('offline', cohorts, pair.offline, pair.offline)
        hist.add('proxy', cohorts, pair.offline, particles)
        stats.paired_counts(pair.offline, particles, counts)
        jets += 1
        if jets % 1000 == 0:
            print(f"C2JPA shard={shard['shard_id']} jets={jets}/{shard['jets']} seconds={time.monotonic()-started:.1f}", flush=True)
    if jets != shard['jets'] or digest.hexdigest() != receipt['physical_digest']:
        raise ValueError('Audit physical population/digest differs')
    result = c.artifact('SHARD', parents=parents, shard_id=shard['shard_id'], jets=jets,
        histograms=hist.payload(), paired_count_histograms=counts, seconds=time.monotonic()-started,
        physical_digest=receipt['physical_digest'], source_file=shard['path'])
    k.write(target, result)
    return k.ref(target)


def run(spec):
    started = time.monotonic()
    study, reference, ranges = c.load_inputs(spec, source=True)
    cpus = spec['resources']['cpus']
    if not os.environ.get('SLURM_JOB_ID') or int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) < cpus:
        raise PermissionError('Run this full audit inside its allocated CPU job, not on a login node')
    print(f"C2JPA start jets={spec['jets']} train_shards={len(spec['train'])} workers={cpus}", flush=True)
    root = Path(spec['root'])
    root.mkdir(parents=True, exist_ok=True)
    # Prevent simultaneous workers publishing competing diagnostics.
    import fcntl
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'outputs.json').exists():
            return read_report(spec)
        refs, done_jets = {}, 0
        sizes = {s['shard_id']: s['jets'] for s in study['shards'] if s['role'] == 'train'}
        with ProcessPoolExecutor(max_workers=cpus, mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = {pool.submit(analyze_shard, spec, study, ranges, row): row for row in spec['train']}
            for future in as_completed(pending):
                row = pending[future]; refs[row['shard_id']] = future.result()
                done_jets += sizes[row['shard_id']]
                elapsed = time.monotonic()-started
                remaining = elapsed*(spec['jets']-done_jets)/done_jets
                print(f"C2JPA committed={len(refs)}/{len(pending)} jets={done_jets}/{spec['jets']} "
                      f"seconds={elapsed:.1f} rough_remaining_minutes={remaining/60:.1f} "
                      'eta_excludes_report=true', flush=True)
        return report(spec, reference, ranges, [refs[r['shard_id']] for r in spec['train']],
                      seconds=time.monotonic()-started)


def report(spec, reference, ranges, refs, *, seconds):
    if len(refs) != len(spec['train']):
        raise ValueError('Incomplete audit shard set')
    payloads, count_rows, hashes, jets = [], [], [], 0
    for ref, expected in zip(refs, spec['train']):
        row = k.load_json(k.checked(ref))
        receipt = k.load_json(k.checked(expected['receipt']))
        c.validate(row, 'SHARD', parents={'spec': spec['content_hash'], 'receipt': receipt['content_hash']})
        if row['shard_id'] != expected['shard_id'] or row['jets'] != receipt['jets']:
            raise ValueError('Audit report shard order/count differs')
        jets += row['jets']; hashes.append(row['content_hash'])
        payloads.append(row['histograms']); count_rows.append(row['paired_count_histograms'])
    if jets != spec['jets']:
        raise ValueError('Audit full training population differs')
    jc2 = d.merge_payloads(payloads)
    payload = stats.joined(reference['histograms'], jc2)
    counts = stats.merge_counts(count_rows)
    result = c.artifact('REPORT', parents={'spec': spec['content_hash'], 'shards': k.canonical_sha256(hashes)},
        jets=jets, shard_artifacts=refs, histograms=payload, tables=stats.tables(payload),
        paired_count_histograms=counts, paired_count_summary=stats.count_summary(counts),
        particle_totals={side: sum(int(n)*frequency for n, frequency in counts[side].items())
                         for side in ('offline', 'proxy')},
        correlations=d.correlation_report(payload), cms_reference={k_: v for k_, v in reference.items() if k_ != 'histograms'},
        runtime_seconds=seconds, analysis_environment=environment(),
        physics_production_qualified=False, descriptive_only=True, independent_proxy_replicas=False,
        raw_cms_particles_accessed=False, final_test_generation_completion_claimed=False)
    root = Path(spec['root'])
    if (root/'report.json').exists():
        # A plotting interruption may leave the authenticated numerical result.
        # Retain its original timing/environment; do not overwrite it on retry.
        saved = k.load_json(root/'report.json')
        c.validate(saved, 'REPORT', parents=result['parents'])
        for name in ('jets', 'histograms', 'tables', 'paired_count_histograms', 'shard_artifacts', 'cms_reference'):
            if saved[name] != result[name]:
                raise ValueError('Saved numerical audit changed: '+name)
        result = saved
    # Publish payloads before the completion receipt. Missing receipt is not completion.
    outputs = {'report': k.write(root/'report.json', result)}
    for name, data in (('statistics.csv', stats.csv_table(result).encode()),
                       ('summary.txt', stats.text_table(result).encode()),
                       ('distributions.pdf', stats.pdf(payload, ranges, result))):
        path = root/name
        k.atomic_publish_bytes(path, data); outputs[name] = k.ref(path)
    k.write(root/'outputs.json', c.artifact('OUTPUTS', parents={'spec': spec['content_hash'],
                    'report': result['content_hash']}, outputs=outputs))
    print(stats.text_table(result), flush=True)
    return result


def read_report(spec):
    c.validate(spec, 'SPEC')
    receipt = k.load_json(Path(spec['root'])/'outputs.json')
    c.validate(receipt, 'OUTPUTS')
    if set(receipt['outputs']) != {'report', 'statistics.csv', 'summary.txt', 'distributions.pdf'}:
        raise ValueError('Incomplete audit report outputs')
    for record in receipt['outputs'].values():
        k.checked(record)
    result = k.load_json(k.checked(receipt['outputs']['report']))
    c.validate(result, 'REPORT')
    if (receipt['parents'] != {'spec': spec['content_hash'], 'report': result['content_hash']}
            or result['parents']['spec'] != spec['content_hash'] or result['jets'] != spec['jets']):
        raise ValueError('Audit report lineage differs')
    return result

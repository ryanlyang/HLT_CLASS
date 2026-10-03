"""Bounded ordered processes; the only generator is frozen literature v3."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import hashlib
from itertools import islice
import multiprocessing
import os
from pathlib import Path
import time

from hlt_classification.literature_proxy_v3.kernel import generate
from . import codec, storage
from .contracts import artifact, digest_ids, sha256_file, safe, write, canonical_sha256

THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
_rates = None


def initialize(rates):
    global _rates
    if any(os.environ.get(k, '1') != '1' for k in THREADS):
        raise ValueError('One native thread per process is required')
    _rates = (rates['p_drop'], rates['p_merge'])


def generate_chunk(batch):
    if _rates is None:
        raise RuntimeError('Frozen calibration not initialized')
    result, counts = [], {}
    for identity, offline in batch:
        out = generate(offline, identity, *_rates)
        # Keys/ancestry are deliberately not returned or persisted.
        result.append((identity, out.particles, None))
        for key, n in out.counts.items():
            counts[key] = counts.get(key, 0)+n
    return result, counts


def chunks(stream, size):
    stream = iter(stream)
    while batch := list(islice(stream, size)):
        yield batch


def parallel(stream, rates, workers, *, chunk=32):
    if type(workers) is not int or not 1 <= workers <= 36 or not 1 <= chunk <= 128:
        raise ValueError('Invalid bounded process profile')
    if workers == 1:
        initialize(rates)
        for batch in chunks(stream, chunk):
            yield generate_chunk(batch)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=(rates,)) as pool:
        pending, source = deque(), iter(chunks(stream, chunk))
        try:
            for batch in islice(source, 2*workers):
                pending.append(pool.submit(generate_chunk, batch))
            while pending:
                future = pending.popleft()
                while True:
                    try:
                        result = future.result(timeout=15)
                        break
                    except TimeoutError:
                        print('JC2-LIT-PROD phase=generate heartbeat=true', flush=True)
                yield result
                batch = next(source, None)
                if batch is not None:
                    pending.append(pool.submit(generate_chunk, batch))
        finally:
            for future in pending:
                future.cancel()


def process(stream, rates, workers, publish, *, chunk=32):
    """Shared production/gate writer, at most 1000 physical rows per block."""
    started = time.monotonic()
    buffered, blocks, identities, totals = [], [], [], {}
    physical, particles = hashlib.sha256(), 0

    def flush():
        nonlocal particles
        packed = codec.pack(buffered)
        blob = codec.encode(packed, 'stored')
        path = publish(f'block_{len(blocks):05d}.npz', blob)
        digests = codec.readback(path, packed)
        for (identity, p, _), digest in zip(buffered, digests):
            identities.append(identity)
            physical.update(bytes.fromhex(digest))
            particles += len(p)
        blocks.append(dict(path=str(path), bytes=len(blob), sha256=sha256_file(path), jets=len(buffered)))
        buffered.clear()
        print(f'JC2-LIT-PROD phase=write jets={len(identities)} seconds={time.monotonic()-started:.1f}', flush=True)

    for rows, counts in parallel(stream, rates, workers, chunk=chunk):
        for key, count in counts.items():
            totals[key] = totals.get(key, 0)+count
        for row in rows:
            buffered.append(row)
            if len(buffered) == 1000:
                flush()
    if buffered:
        flush()
    if (not identities or totals['output_particles'] != particles or totals['jets'] != len(identities)
            or totals['input_particles']-totals['dropped_particles']-totals['merged_pairs'] != particles):
        raise ValueError('Production particle/count accounting differs')
    return dict(blocks=blocks, jets=len(identities), particles=particles,
        ordered_identities=digest_ids(identities), physical_digest=physical.hexdigest(),
        output_bytes=sum(b['bytes'] for b in blocks), seconds=time.monotonic()-started)


def generate_shard(study, attempt, shard, rates):
    from . import population as pop, campaign as c
    c.require_preflight(study, attempt)
    lock = c.test_lock(study) if shard['role'] == 'final_test' else None
    root = Path(study['root'])
    # Failed claims survive. A new immutable attempt is required to retry.
    from hlt_classification.literature_proxy.campaign import exclusive
    with exclusive(safe(root, f'attempts/{attempt["name"]}/claims/{shard["shard_id"]}.lock')):
        if safe(root, f'shards/{shard["shard_id"]}.json').exists():
            raise PermissionError('Completed shard is immutable; do not generate twice')
        total = sum(study['counts'].values())
        allowance = int((study['storage']['budget_bytes']-storage.METADATA_ALLOWANCE)*shard['jets']/total)
        reservation = storage.reserve(study, attempt['name'], shard['shard_id'], allowance)
        result = process(pop.iterate(study['data_root'], study['population'], shard, test_lock=lock),
            rates, study['resources']['cpus'], lambda name, blob: storage.publish_block(study, reservation, name, blob))
        if result['jets'] != shard['jets'] or result['ordered_identities'] != shard['ordered_identities']:
            raise ValueError('Generated shard membership differs')
        for block in result['blocks']:
            block['relative'] = Path(block.pop('path')).relative_to(root).as_posix()
        # No test distributions; only construction integrity counts are committed.
        result = artifact('SHARD', parents=dict(study=study['content_hash'], attempt=attempt['content_hash'],
            shard=canonical_sha256(shard)), test=shard['role'] == 'final_test',
            shard_id=shard['shard_id'], role=shard['role'], attempt=attempt['name'],
            physical_schema_verified=True, test_lock=None if lock is None else lock['content_hash'], **result)
        write(safe(root, f'shards/{shard["shard_id"]}.json'), result)
    return result

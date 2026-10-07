"""Bounded ordered processes; apply the unchanged frozen CORR_MID kernel."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import hashlib
from itertools import islice
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
from hlt_classification.correlated_tracking.kernel import generate_all, assert_structure
from .recipe import recipe
from . import codec, storage
from .contracts import artifact, digest_ids, sha256_file, safe, write, canonical_sha256

THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
_recipe = None


def initialize(frozen_recipe):
    global _recipe
    if any(os.environ.get(k, '1') != '1' for k in THREADS):
        raise ValueError('One native thread per process is required')
    if frozen_recipe != recipe():
        raise ValueError('Only the frozen CORR_MID endpoint is authorized')
    _recipe = frozen_recipe


def generate_chunk(batch):
    if _recipe is None:
        raise RuntimeError('Frozen recipe not initialized')
    result, counts = [], {}
    for identity, offline in batch:
        sides, geometry = generate_all(offline, identity)
        out = sides['CORR_MID']
        assert_structure(offline, out, geometry.eligible)
        # No keys, ancestry, random draws or oracle fields are persisted.
        result.append((identity, out, None))
        for key, n in dict(jets=1, input_particles=len(offline), output_particles=len(out),
                           structure_verified_jets=1).items():
            counts[key] = counts.get(key, 0)+n
    return result, counts


def chunks(stream, size):
    stream = iter(stream)
    while batch := list(islice(stream, size)):
        yield batch


def parallel(stream, frozen_recipe, workers, *, chunk=32):
    if type(workers) is not int or not 1 <= workers <= 36 or not 1 <= chunk <= 128:
        raise ValueError('Invalid bounded process profile')
    if workers == 1:
        initialize(frozen_recipe)
        for batch in chunks(stream, chunk):
            yield generate_chunk(batch)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=(frozen_recipe,)) as pool:
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
                        print('JC2-CORR-PROD phase=generate heartbeat=true', flush=True)
                yield result
                batch = next(source, None)
                if batch is not None:
                    pending.append(pool.submit(generate_chunk, batch))
        finally:
            for future in pending:
                future.cancel()


def process(stream, frozen_recipe, workers, publish, *, chunk=32):
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
        print(f'JC2-CORR-PROD phase=write jets={len(identities)} seconds={time.monotonic()-started:.1f}', flush=True)

    for rows, counts in parallel(stream, frozen_recipe, workers, chunk=chunk):
        for key, count in counts.items():
            totals[key] = totals.get(key, 0)+count
        for row in rows:
            buffered.append(row)
            if len(buffered) == 1000:
                flush()
    if buffered:
        flush()
    if (not identities or totals['output_particles'] != particles or totals['jets'] != len(identities)
            or totals['input_particles'] != particles or totals['structure_verified_jets'] != len(identities)):
        raise ValueError('Production particle/count accounting differs')
    return dict(blocks=blocks, jets=len(identities), particles=particles,
        ordered_identities=digest_ids(identities), physical_digest=physical.hexdigest(),
        output_bytes=sum(b['bytes'] for b in blocks), seconds=time.monotonic()-started,
        structure_exact=True)


def generate_shard(study, attempt, shard, frozen_recipe):
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
            frozen_recipe, study['resources']['cpus'], lambda name, blob: storage.publish_block(study, reservation, name, blob))
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

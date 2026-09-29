"""Ordered bounded processes and lossless, source-key-free benchmark blocks."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import hashlib
import io
from itertools import islice
import multiprocessing
import os
from pathlib import Path
import time
import zipfile

import numpy as np

from . import bdz_joint_maps as maps
from .bridge import Particles
from .contracts import canonical_sha256, sha256_file, validate
from .dev_parallel import _child_init
from .response import Generator
from .storage import publish_bytes, GIB

FIELDS = ('p4', 'charge', 'category', 'tracking', 'valid')
DTYPES = ('<f8', 'i1', 'i1', '<f8', '?')
_runtime = None


def initialize(bundle):
    global _runtime
    _child_init()
    validate(bundle, 'GEN_BUNDLE')
    maps.validate_map(bundle['joint'])
    maps.old.validate_map(bundle['historical'])
    _runtime = (Generator(bundle['response']), bundle['historical'], bundle['joint'])


def physical_digest(identity, particle):
    digest = hashlib.sha256(bytes.fromhex(identity))
    for name, dtype in zip(FIELDS, DTYPES):
        array = np.asarray(getattr(particle, name), dtype=dtype)
        digest.update(np.asarray(array.shape, dtype='<i8').tobytes())
        digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def generate_chunk(pairs, trace=False):
    if _runtime is None:
        raise RuntimeError('Initialize immutable response once per worker')
    gen, historical, joint = _runtime
    cpu, start = time.process_time(), time.perf_counter()
    rows, flags = [], {}
    for pair in pairs:
        if pair.hlt is not None:
            raise PermissionError('Native HLT is not a generation input')
        base, info = gen(pair.offline, jet=pair.identity, replica=0, trace=trace)
        out, _ = maps.apply(base, historical, joint, 'JOINT')
        maps.old.check_invariants(base, out)
        rows.append((pair.identity, out, canonical_sha256(list(out.keys))))
        for name, active in info['flags'].items():
            flags[name] = flags.get(name, 0)+int(bool(active))
    return dict(rows=rows, cpu_seconds=time.process_time()-cpu,
                worker_wall_seconds=time.perf_counter()-start, flags=flags)


def chunks(stream, size):
    while batch := list(islice(stream, size)):
        yield batch


def parallel(stream, bundle, workers, chunk, trace=False):
    # PORT_STAGE registers the larger Tigris profiles; GEN_STAGE's task registry
    # still permits only its original 4/8/16-worker screen.
    if type(workers) is not int or workers not in (1, 4, 8, 16, 36, 72, 144) or chunk not in (8, 32, 128):
        raise ValueError('Unregistered generation execution profile')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        if os.environ.get(key, '1') != '1':
            raise ValueError('Generation requires one native thread per process')
    if workers == 1:
        initialize(bundle)
        for batch in chunks(stream, chunk):
            yield generate_chunk(batch, trace)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=(bundle,)) as pool:
        pending, source = deque(), iter(chunks(stream, chunk))
        try:
            for _ in range(2*workers):
                batch = next(source, None)
                if batch is None:
                    break
                pending.append(pool.submit(generate_chunk, batch, trace))
            while pending:
                # Bound both waiting work and completed-but-not-yet-ordered RAM.
                future = pending.popleft()
                while True:
                    try:
                        result = future.result(timeout=15)
                        break
                    except TimeoutError:
                        print('CMS2JC2-GEN phase=generating heartbeat=true', flush=True)
                yield result
                batch = next(source, None)
                if batch is not None:
                    pending.append(pool.submit(generate_chunk, batch, trace))
        finally:
            for future in pending:
                future.cancel()


def pack(rows):
    offsets = np.concatenate(([0], np.cumsum([len(p) for _, p, _ in rows], dtype=np.int64)))
    arrays = {name: np.concatenate([np.asarray(getattr(p, name), dtype=dtype) for _, p, _ in rows])
              for name, dtype in zip(FIELDS, DTYPES)}
    return dict(offsets=np.asarray(offsets, dtype='<i8'),
                jet_identity=np.asarray([list(bytes.fromhex(i)) for i, _, _ in rows], dtype='u1'), **arrays)


def encode(arrays, compression):
    if compression not in ('stored', 'deflate'):
        raise ValueError('Only lossless encodings are registered')
    stream = io.BytesIO()
    method = zipfile.ZIP_STORED if compression == 'stored' else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(stream, 'w', compression=method, compresslevel=6 if method else None) as archive:
        for name, array in sorted(arrays.items()):
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(name+'.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = method
            archive.writestr(info, buffer.getvalue(), compresslevel=6 if method else None)
    return stream.getvalue()


def readback(path, expected):
    decoded = {}
    with np.load(path, allow_pickle=False) as handle:
        if set(handle.files) != set(expected):
            raise ValueError('Output fields changed')
        for name, array in expected.items():
            actual = handle[name]
            if actual.dtype != array.dtype or actual.shape != array.shape or actual.tobytes() != array.tobytes():
                raise ValueError('Output readback changed: '+name)
            decoded[name] = actual
    # Validate the cache through the same physical schema, not just its bytes.
    offsets = decoded['offsets']
    if offsets[0] != 0 or np.any(np.diff(offsets) < 0) or offsets[-1] != len(decoded['p4']):
        raise ValueError('Invalid output offsets')
    digests = []
    for i, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        p = Particles(*(decoded[k][lo:hi] for k in FIELDS), tuple(str(j) for j in range(hi-lo)))
        digests.append(physical_digest(bytes(decoded['jet_identity'][i]).hex(), p))
    return digests


def process(stream, bundle, *, root, relative, workers, chunk, compression, expected_count,
            trace=False, block_jets=1000):
    start = time.perf_counter()
    pending, blocks, identity, physical, keys = [], [], hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    flags, count, cpu, worker_wall, io_seconds, particles = {}, 0, 0., 0., 0., 0

    def flush(rows):
        nonlocal io_seconds
        begin = time.perf_counter()
        arrays = pack(rows)
        blob = encode(arrays, compression)
        path = publish_bytes(Path(root), f'{relative}/block_{len(blocks):05d}.npz', blob, remaining_bytes=2*GIB)
        file_hash = sha256_file(path)
        if file_hash != hashlib.sha256(blob).hexdigest():
            raise ValueError('Published output checksum differs')
        digests = readback(path, arrays)
        for (jet, p, key), digest in zip(rows, digests):
            if digest != physical_digest(jet, p):
                raise ValueError('Physical output digest differs')
            identity.update(bytes.fromhex(jet)); physical.update(bytes.fromhex(digest)); keys.update(bytes.fromhex(key))
        blocks.append(dict(relative=path.relative_to(root).as_posix(), sha256=file_hash,
                           bytes=len(blob), jets=len(rows)))
        io_seconds += time.perf_counter()-begin

    for result in parallel(stream, bundle, workers, chunk, trace):
        count += len(result['rows'])
        if count > expected_count:
            raise ValueError('Reader exceeded frozen population')
        cpu += result['cpu_seconds']; worker_wall += result['worker_wall_seconds']
        for k, v in result['flags'].items():
            flags[k] = flags.get(k, 0)+v
        particles += sum(len(p) for _, p, _ in result['rows'])
        pending.extend(result['rows'])
        while len(pending) >= block_jets:
            flush(pending[:block_jets]); del pending[:block_jets]
            print(f'CMS2JC2-GEN phase=written jets={count}/{expected_count} seconds={time.perf_counter()-start:.1f}', flush=True)
    if count != expected_count:
        raise ValueError('Reader omitted registered jets')
    if pending:
        flush(pending)
    return dict(jets=count, particles=particles, ordered_identities=identity.hexdigest(),
        physical_digest=physical.hexdigest(), generation_key_digest=keys.hexdigest(), blocks=blocks,
        flags=flags, output_bytes=sum(b['bytes'] for b in blocks),
        processing_seconds=time.perf_counter()-start, output_io_seconds=io_seconds,
        generation_cpu_seconds=cpu, summed_generation_wall_seconds=worker_wall)


def parity(rows):
    keys = ('jets', 'particles', 'ordered_identities', 'physical_digest', 'generation_key_digest', 'flags')
    if not rows or any(any(r[k] != rows[0][k] for k in keys) for r in rows[1:]):
        raise ValueError('Generation changed across serial/process/chunk/trace/encoding settings')
    return True

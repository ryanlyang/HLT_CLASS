"""Scheduled production execution using the unchanged frozen process kernel."""
import hashlib
import itertools
import math
import os
from pathlib import Path
import time

from . import campaign as c, population as pop, storage, output
from .contracts import artifact, write, safe, sha256_file, load_json
from hlt_classification.cms2jc2_response import generation_benchmark_engine as engine
from hlt_classification.cms2jc2_response.measurement import Measurement
from hlt_classification.cms2jc2_response.provenance import numerical_environment


def generate(study, attempt, shard, reservation, *, bundle, measurement_factory=Measurement):
    begin = time.perf_counter()
    lock = c.test_lock(study) if shard['role'] == 'final_test' else None
    iterator = pop.iterate(study['data_root'], study['population'], shard, study['review'], test_lock=lock)
    # Only the TRAIN pilot is serially replayed. Never tune on test generation.
    prefix = list(itertools.islice(iterator, 64)) if shard['pilot'] else []
    stream = itertools.chain(prefix, iterator)
    blocks, pending, parallel_prefix = [], [], []
    physical, identities, keys = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    count, particle_count, cpu = 0, 0, 0.

    def flush(rows):
        values = engine.pack(rows)
        blob = engine.encode(values, 'stored')
        path = storage.publish_block(study, reservation, f'block_{len(blocks):05d}.npz', blob)
        digests = engine.readback(path, values)
        checksum = sha256_file(path)
        if checksum != hashlib.sha256(blob).hexdigest():
            raise ValueError('Production block readback checksum differs')
        for (identity, p, key), digest in zip(rows, digests):
            if digest != engine.physical_digest(identity, p):
                raise ValueError('Physical readback differs')
            identities.update(bytes.fromhex(identity))
            physical.update(bytes.fromhex(digest))
            keys.update(bytes.fromhex(key))
        blocks.append(dict(relative=path.relative_to(study['root']).as_posix(),
                           sha256=checksum, bytes=len(blob), jets=len(rows)))

    try:
        with measurement_factory() as measured:
            for result in engine.parallel(stream, bundle, 36, 32, False):
                count += len(result['rows'])
                if count > shard['jets']:
                    raise ValueError('Generation exceeded frozen shard membership')
                cpu += result['cpu_seconds']
                particle_count += sum(len(p) for _, p, _ in result['rows'])
                if shard['pilot'] and len(parallel_prefix) < 64:
                    parallel_prefix.extend(result['rows'][:64-len(parallel_prefix)])
                pending.extend(result['rows'])
                while len(pending) >= 1000:
                    flush(pending[:1000]); del pending[:1000]
                    print(f'CMS2JC2-PRODUCTION shard={shard["shard_id"]} written={sum(b["jets"] for b in blocks)}/{shard["jets"]} seconds={time.perf_counter()-begin:.1f}', flush=True)
            if count != shard['jets']:
                raise ValueError('Generation omitted frozen rows')
            if pending:
                flush(pending)
            exact = None
            if shard['pilot']:
                engine.initialize(bundle)
                serial = engine.generate_chunk(prefix, False)['rows']
                a = [(i, engine.physical_digest(i, p), k) for i, p, k in serial]
                b = [(i, engine.physical_digest(i, p), k) for i, p, k in parallel_prefix]
                if len(a) != 64 or a != b:
                    raise ValueError('Retained production pilot serial/process parity failed')
                exact = True
    finally:
        iterator.close()
    if identities.hexdigest() != shard['ordered_identities']:
        raise ValueError('Generated membership order differs')
    return dict(shard_id=shard['shard_id'], attempt=attempt['name'], role=shard['role'], jets=count,
        particles=particle_count, blocks=blocks, ordered_identities=identities.hexdigest(),
        physical_digest=physical.hexdigest(), generation_key_digest=keys.hexdigest(),
        output_bytes=sum(b['bytes'] for b in blocks), processing_seconds=time.perf_counter()-begin,
        generation_cpu_seconds=cpu, serial_process_exact=exact, measurement=measured.report(),
        physical_schema_verified=True,
        test_lock=None if lock is None else lock['content_hash'])


def run(attempt_path, task, *, index=None):
    from .submission import worker_identity
    from .recovery import validate_execution
    start = time.perf_counter()
    print(f'CMS2JC2-PRODUCTION phase=authenticate task={task} index={index}', flush=True)
    attempt = load_json(attempt_path)
    study = load_json(c.checked(attempt['study']))
    c.validate_attempt(attempt, study)
    repair = validate_execution(study, attempt.get('execution_repair'))
    if attempt['kind'] != 'pilot':
        c.require_admission(study, attempt)
    allocation = worker_identity(study, attempt, task, index)
    if numerical_environment() != study['numerical_environment']:
        raise ValueError('Allocated Tigris numerical environment differs')
    name = f'{task}_{index:05d}' if index is not None else task
    claim_path = safe(study['root'], f'attempts/{attempt["name"]}/claims/{name}')
    claim_path.mkdir(parents=True, exist_ok=False)
    write(claim_path/'claim.json', artifact('CLAIM', parents={'attempt': attempt['content_hash']},
        task=task, index=index, allocation=allocation))
    if task == 'preflight':
        print('CMS2JC2-PRODUCTION phase=full_historical_authentication', flush=True)
        result = c.preflight(study)
        c.validate_study(study, source=True)
        write(safe(study['root'], 'preflight.json'), result)
    elif task == 'generate':
        c.require_preflight(study)
        shard = next(s for s in study['shards'] if s['shard_id'] == attempt['shards'][index])
        receipt_path = safe(study['root'], f'shards/{shard["shard_id"]}.json')
        if receipt_path.exists():
            raise PermissionError('Completed shard must not be regenerated')
        allowance = math.ceil(shard['jets']*attempt['resources']['bytes_per_jet'])+64*2**20
        reservation = storage.reserve(study, attempt['name'], shard['shard_id'], allowance)
        auth_seconds = time.perf_counter()-start
        print(f'CMS2JC2-PRODUCTION phase=generate shard={shard["shard_id"]} jets={shard["jets"]} workers=36 authentication_seconds={auth_seconds:.1f}', flush=True)
        row = generate(study, attempt, shard, reservation, bundle=c.imported(study, 'bundle'))
        extra = dict(execution_repair=attempt['execution_repair'],
                     scientific_source=study['source']['content_hash']) if repair else {}
        result = artifact('SHARD_EXECUTION_REPAIR' if repair else 'SHARD',
            parents={'study': study['content_hash'],
            'population': study['population']['content_hash']}, test=shard['role'] == 'final_test',
            **row, **extra, source=(repair['source'] if repair else study['source'])['content_hash'],
            environment=study['numerical_environment']['content_hash'],
            authentication_seconds=auth_seconds, allocation=allocation)
        output.verify_shard(study, result)
        validate_execution(study, attempt.get('execution_repair'))
        if numerical_environment() != study['numerical_environment']:
            raise ValueError('Numerical environment changed during generation')
        # Receipt is the commit marker; any earlier interrupted blocks remain uncommitted.
        write(receipt_path, result)
    elif task == 'finalize':
        result = output.finalize(study)
    else:
        raise ValueError('Unknown production task')
    print(f'CMS2JC2-PRODUCTION task={name} complete=true content_hash={result["content_hash"]}', flush=True)
    return result

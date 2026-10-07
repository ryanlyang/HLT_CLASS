"""Ordinary-role materialization on the exact completed CORR_MID population."""
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
import multiprocessing
from pathlib import Path
import shutil
import numpy as np

from hlt_classification.cms_proxy_ladder.contracts import (
    artifact, validate, file_ref, validate_file_ref, write_json,
)
from hlt_classification.data.cache_contracts import atomic_publish_bytes, load_json
from hlt_classification.literature_proxy_production import codec
from hlt_classification.literature_proxy_v3 import contracts as noise_contracts
from hlt_classification.literature_proxy_v3.kernel import recipe as noise_recipe
from .kernel import recipe, generate
from hlt_classification.literature_proxy import diagnostics

COUNTS = dict(train=100000, validation=50000)
KIND = 'controlled_corr_high_topology_synthetic_proxy'


def noise_calibration(spec_path, *, full=False):
    spec = load_json(spec_path)
    noise_contracts.validate(spec, 'SPEC')
    if spec['recipe'] != noise_recipe():
        raise ValueError('Expected unchanged NOISE_V3 pilot recipe')
    root = Path(spec['root'])
    receipt, report = load_json(root/'receipt.json'), load_json(root/'report.json')
    noise_contracts.validate(receipt, 'RECEIPT'); noise_contracts.validate(report, 'REPORT')
    from hlt_classification.data.cache_contracts import sha256_file
    rates = load_json(root/'calibration.json')
    if (receipt['parents'] != dict(spec=spec['content_hash'])
            or report['parents'] != dict(spec=spec['content_hash'], calibration=rates['content_hash'])
            or any(sha256_file(root/name) != receipt['outputs'][name]
                   for name in ('report.json', 'calibration.json'))
            or rates != spec['calibration'] or report['calibration'] != rates
            or report['structure_exact'] is not True or report['replay']['exact'] is not True
            or report['calibration_refitted'] is not False):
        raise ValueError('NOISE_V3 completed calibration lineage differs')
    recipe(rates)  # Validate its original training-only counts/probabilities.
    if full:
        from hlt_classification.literature_proxy_v3.worker import results
        results(spec)
    return rates


def request(original_release, noise_spec):
    from hlt_classification.cms_proxy_ladder.release import validate_release
    original = load_json(original_release)
    validate_release(original, root=Path(original_release).parent)
    if original['schema_version'] != 4 or original['counts'] != COUNTS:
        raise ValueError('Expected completed CORR_MID 100k/50k release')
    rates = noise_calibration(noise_spec)
    return artifact('RELEASE_REQUEST', version=7, original_release=file_ref(Path(original_release)),
        noise_spec=file_ref(Path(noise_spec)), recipe=recipe(rates), counts=dict(COUNTS),
        study_root=original['study_root'], offline_root=original['offline_root'],
        selection_domain=original['selection_domain'], allowed_roles=list(COUNTS),
        labels_read=False, selection_depends_on_labels=False, dataset_kind=KIND)


def validate_request(value):
    digest = validate(value, 'RELEASE_REQUEST', version=7)
    original = validate_file_ref(value['original_release'])
    noise = validate_file_ref(value['noise_spec'])
    if value != request(original, noise):
        raise ValueError('CORR_HIGH_TOPO request/recipe differs')
    return digest


def _chunk(args):
    rows, frozen = args
    result, counts = [], {}
    collector = diagnostics.Collector()
    replay = {}
    for row in rows:
        response = generate(row.offline, row.identity, frozen, historical_mid=row.proxy)
        result.append((row.identity, response.particles, None))
        for k, v in response.counts.items():
            counts[k] = counts.get(k, 0) + v
        merge_replay(replay, response.historical_replay)
        if row.role == 'train':
            collector.observe('OFFLINE', row.offline)
            collector.observe('CORR_HIGH_TOPO', response.particles)
    return result, counts, collector.finish(), replay


def merge_replay(target, row):
    for key, value in row.items():
        target[key] = max(target.get(key, 0), value) if key.startswith('max_') else target.get(key, 0)+value


def _chunks(rows, frozen):
    while batch := list(islice(rows, 128)):
        yield batch, frozen


def build_release(value, *, output_root, workers=4):
    from hlt_classification.cms_proxy_ladder import release as r, data
    from hlt_classification.cms_proxy_ladder.cache_full import ordered_results
    validate_request(value)
    if type(workers) is not int or not 1 <= workers <= 6:
        raise ValueError('Expected 1..6 generation processes')
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError('Fresh topology output root required; preserve partial output')
    original_path = validate_file_ref(value['original_release'])
    original = load_json(original_path)
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in
           (original_path.parent.resolve(), Path(original['study_root']).resolve(), Path(original['offline_root']).resolve())):
        raise PermissionError('New release overlaps protected input')
    noise_calibration(validate_file_ref(value['noise_spec']), full=True)
    if shutil.disk_usage(root.parent).free < 10*2**30:
        raise OSError('Require 10 GiB filesystem headroom for study-sized release')
    bank = r.load_bank(original, root=original_path.parent)
    new_bank = {k: v.copy() for k, v in bank.items()}
    root.mkdir()
    blocks, totals, consumed, statistics, replays = [], {}, 0, {}, {}
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                               initializer=data._limit_worker_threads)
    try:
        for role in COUNTS:
            count, buffer = 0, []
            rows = iter(data.iter_paired(original, release_root=original_path.parent, role=role))
            chunks = iter(_chunks(rows, value['recipe']))
            first = next(chunks, None)
            if first is None:
                raise ValueError('Empty ordinary role')
            serial = _chunk(first)
            parallel = pool.submit(_chunk, first).result()
            if (codec.encode(codec.pack(serial[0]), 'stored') != codec.encode(codec.pack(parallel[0]), 'stored')
                    or serial[1:] != parallel[1:]):
                raise ValueError('Generation process replay differs')
            def flush():
                nonlocal consumed
                if not buffer:
                    return
                packed = codec.pack(buffer)
                ids = bank['identity'][consumed:consumed+len(buffer)]
                if not np.array_equal(ids, packed['jet_identity']):
                    raise ValueError('Generation changed the frozen ordered population')
                index = len(blocks)
                path = root/f'block_{index:05d}.npz'
                blob = codec.encode(packed, 'deflate')
                if sum(b['bytes'] for b in blocks)+len(blob) > 5*2**30:
                    raise OSError('Study-sized dataset exceeded the 5 GiB payload budget')
                atomic_publish_bytes(path, blob)
                codec.readback(path, packed)
                blocks.append(file_ref(path, root=root))
                sl = slice(consumed, consumed+len(buffer))
                new_bank['proxy_block'][sl] = index
                new_bank['proxy_row'][sl] = np.arange(len(buffer), dtype=np.int32)
                consumed += len(buffer)
                print(f'CORR-HIGH-TOPO materialized={consumed} role={role}', flush=True)
                buffer.clear()
            from itertools import chain
            for generated, counts, summary, audit in chain([parallel], ordered_results(pool, _chunk, chunks, workers)):
                statistics = diagnostics.merge((statistics, summary))
                merge_replay(replays.setdefault(role, {}), audit)
                for k, n in counts.items():
                    totals.setdefault(role, {})[k] = totals.setdefault(role, {}).get(k, 0)+n
                for row in generated:
                    buffer.append(row); count += 1
                    if len(buffer) == 1000:
                        flush()
            flush()
            if count != COUNTS[role]:
                raise ValueError('Generated ordinary role count differs')
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    if consumed != original['total_rows']:
        raise ValueError('Incomplete topology dataset')
    r._save_npz(root/'release_index.npz', new_bank)
    report = artifact('CORR_HIGH_TOPO_DIAGNOSTICS', parents=dict(request=value['content_hash']),
        role='train', jets=COUNTS['train'], statistics=statistics, mechanism_counts=totals['train'],
        sd_is_distribution_width=True, used_for_tuning=False)
    write_json(root/'diagnostics.json', report)
    result = artifact('RELEASE', version=7,
        parents=dict(request=value['content_hash'], study=original['content_hash']), request=value,
        study_contract='JC2_CORR_HIGH_TOPO_ORDINARY_DATASET/v1', study_root=str(root),
        offline_root=original['offline_root'], scope='committed_ordinary_receipt_snapshot',
        counts=COUNTS, selection_domain=value['selection_domain'], labels_read=True,
        labels_used_by_generator=False, selection_depends_on_labels=False, receipts=[],
        source_files=original['source_files'], proxy_blocks=blocks,
        bank=file_ref(root/'release_index.npz', root=root), identity_sha256=original['identity_sha256'],
        total_rows=consumed, mechanism_counts=totals, exact_process_replay=True,
        historical_mid_replay=replays,
        diagnostics=file_ref(root/'diagnostics.json', root=root),
        materialized_final_test=False, payload_bytes=sum(b['bytes'] for b in blocks))
    validate_release_source(result)
    write_json(root/'release.json', result)
    r.validate_release(result, root=root)
    return result


def validate_release_source(value):
    from hlt_classification.cms_proxy_ladder import release as r
    original_path = validate_file_ref(value['request']['original_release'])
    original = load_json(original_path)
    if (value['parents']['study'] != original['content_hash'] or value['source_files'] != original['source_files']
            or value['identity_sha256'] != original['identity_sha256'] or value['offline_root'] != original['offline_root']
            or value['study_contract'] != 'JC2_CORR_HIGH_TOPO_ORDINARY_DATASET/v1'
            or value['exact_process_replay'] is not True or value['materialized_final_test'] is not False
            or value['labels_used_by_generator'] is not False or value['receipts'] != []
            or value['payload_bytes'] != sum(b['bytes'] for b in value['proxy_blocks'])
            or value['payload_bytes'] > 5*2**30):
        raise ValueError('Topology release lineage/evidence differs')
    old = r.load_bank(original, root=original_path.parent)
    report = load_json(validate_file_ref(value['diagnostics'], root=Path(value['study_root'])))
    validate(report, 'CORR_HIGH_TOPO_DIAGNOSTICS', parents=dict(request=value['request']['content_hash']))
    if (report['role'] != 'train' or report['jets'] != COUNTS['train'] or report['used_for_tuning'] is not False
            or report['mechanism_counts'] != value['mechanism_counts']['train']
            or any(report['statistics'][f'{side}/jet/multiplicity']['count'] != COUNTS['train']
                   for side in ('OFFLINE', 'CORR_HIGH_TOPO'))):
        raise ValueError('Topology train-only diagnostics differ')
    path = validate_file_ref(value['bank'], root=Path(value['study_root']))
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != r.BANK_FIELDS:
            raise ValueError('Topology bank field set differs')
        for name in ('identity', 'role', 'source_file', 'entry'):
            if not np.array_equal(old[name], archive[name]):
                raise ValueError('Topology release changed original selection')
        if (np.any(archive['proxy_block'] < 0) or np.any(archive['proxy_block'] >= len(value['proxy_blocks']))
                or np.any(archive['proxy_row'] < 0)):
            raise ValueError('Topology bank block/row index differs')
    for role in COUNTS:
        n = value['mechanism_counts'][role]
        if n['jets'] != COUNTS[role] or n['output_particles'] != n['input_particles']-n['dropped_particles']-n['merged_pairs']:
            raise ValueError('Topology mechanism accounting differs')
        audit = value['historical_mid_replay'][role]
        if (audit['jets'] != COUNTS[role] or not 0 <= audit['bitwise_exact_jets'] <= audit['jets']
                or not 0 <= audit['max_tolerance_fraction'] <= 1
                or not 0 <= audit['max_abs_tracking_difference_mm'] < float('inf')):
            raise ValueError('Historical MID portability evidence differs')


def iter_paired(value, *, release_root, role, source_file_index=None):
    from hlt_classification.cms_proxy_ladder import release as r, data
    data._role_code(role)  # Reject test before opening any artifact.
    r.validate_release(value, root=release_root)
    original_path = validate_file_ref(value['request']['original_release'])
    original = load_json(original_path)
    bank = r.load_bank(value, root=release_root)
    start = 0 if role == 'train' else COUNTS['train']
    current, arrays = None, None
    for row in data.iter_paired(original, release_root=original_path.parent,
                                role=role, source_file_index=source_file_index):
        absolute = start+row.ordinal
        block_id = int(bank['proxy_block'][absolute])
        if block_id != current:
            path = validate_file_ref(value['proxy_blocks'][block_id], root=Path(value['study_root']))
            from hlt_classification.cms2jc2_production.output import arrays as read_arrays
            arrays = read_arrays(path)
            validate_file_ref(value['proxy_blocks'][block_id], root=Path(value['study_root']))
            current = block_id
        index = int(bank['proxy_row'][absolute])
        if not 0 <= index < len(arrays['jet_identity']):
            raise ValueError('Topology paired row outside block')
        if bytes(arrays['jet_identity'][index]).hex() != row.identity:
            raise ValueError('Topology paired identity differs')
        yield data.PairedRow(row.ordinal, row.identity, role, row.label,
                             data._proxy_particles(arrays, index), row.offline)

"""Authenticated scalar-only foundation and reproducible sealed memberships."""
from pathlib import Path
import hashlib
import re

import numpy as np
import uproot

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, canonical_sha256, deterministic_npz_bytes, load_json,
    load_npz_arrays, sha256_file, write_immutable_json,
)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.inventory import latest_tree
from hlt_classification.jetclass2_delphes.schema import CLASS_NAMES, map_labels, validate_tree_schema
from hlt_classification.luka_fullsim.contracts import SCALARS
from .contracts import (SOURCE, COUNTS, ROLES, ORDINARY, SEED, BLOCK_ROWS, artifact,
                        validate, source_path, split_policy)


def transfer(container, expected_files=16):
    container = Path(container).resolve(strict=True)
    marker, manifest = container/'source.txt', container/'source.sha256'
    if marker.read_text(encoding='utf-8').strip() != SOURCE:
        raise ValueError('FullStat source marker differs')
    records = []
    for line in manifest.read_text(encoding='utf-8').splitlines():
        match = re.fullmatch(r'([0-9a-f]{64}) [ *](.+)', line)
        if match is None:
            raise ValueError('Malformed source SHA256 manifest')
        digest, name = match.groups()
        name = name.removeprefix('./')
        _, category = source_path(container/'raw', name)
        records.append(dict(path=name, sha256=digest, category=category))
    records.sort(key=lambda r: r['path'])
    if (len(records) != expected_files or len({r['path'].casefold() for r in records}) != expected_files
            or len({r['sha256'] for r in records}) != expected_files
            or sorted(p.relative_to(container/'raw').as_posix() for p in (container/'raw').rglob('*.root'))
            != [r['path'] for r in records]
            or not any(r['category'] == 'signal' for r in records)
            or not any(r['category'] != 'signal' for r in records)):
        raise ValueError('Missing/extra/duplicate ROOT source or source categories')
    return dict(source=SOURCE, marker_sha256=sha256_file(marker), manifest_sha256=sha256_file(manifest)), records


def eligible(arrays, category):
    if set(arrays) != set(SCALARS):
        raise ValueError('Scalar field coverage differs')
    labels = map_labels(arrays['jet_label'])
    if any(a.ndim != 1 or len(a) != len(labels) for a in arrays.values()):
        raise ValueError('Scalar shapes differ')
    if arrays['hlt_matched'].dtype.kind != 'b':
        raise ValueError('Matching flag is not Boolean')
    for name in ('jet_nparticles', 'hlt_jet_nparticles'):
        if arrays[name].dtype.kind not in 'iu' or np.any(arrays[name] < 0):
            raise ValueError('Negative/invalid particle count')
    if arrays['jet_pt'].dtype.kind != 'f' or not np.isfinite(arrays['jet_pt']).all():
        raise ValueError('Nonfinite/invalid offline jet pT')
    if category != 'signal' and np.any(labels != 0):
        raise ValueError('Unexpected signal label in QCD source')
    keep, flow = np.ones(len(labels), bool), {}
    for name, mask in (
        ('excluded_qcd_source', (labels != 0) | (category != 'signal')),
        ('excluded_unmatched', arrays['hlt_matched']),
        ('excluded_empty', (arrays['jet_nparticles'] > 0) & (arrays['hlt_jet_nparticles'] > 0)),
    ):
        flow[name] = int(np.count_nonzero(keep & ~mask))
        keep &= mask
    flow['selected'] = int(keep.sum())
    return keep, labels.astype(np.int8), flow


def apportion(total, capacities):
    """Largest remainder with integer arithmetic and canonical stratum tie breaks."""
    denominator = sum(capacities.values())
    if type(total) is not int or total < 0 or total > denominator:
        raise ValueError('Insufficient split capacity')
    if denominator == 0:
        return {k: 0 for k in capacities}
    result = {k: total*n//denominator for k, n in capacities.items()}
    order = sorted(capacities, key=lambda k: (-(total*capacities[k] % denominator), k))
    for k in order[:total-sum(result.values())]:
        result[k] += 1
    return result


def partition(records, scalars, counts):
    """Metadata only. Shared block order keeps selected-only particle I/O bounded."""
    if set(counts) != set(ROLES) or any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError('Four explicit nonnegative role counts required')
    groups, labels_by_file, eligible_by_file = {}, [], []
    for index, (record, arrays) in enumerate(zip(records, scalars)):
        keep, labels, _ = eligible(arrays, record['category'])
        labels_by_file.append(labels)
        eligible_by_file.append(keep)
        for label in range(11):
            entries = np.flatnonzero(keep & (labels == label))
            if not len(entries):
                continue
            stratum = f'signal:{label:02d}' if record['category'] == 'signal' else record['category']
            groups.setdefault(stratum, []).append((index, entries))
    capacity = {key: sum(len(es) for _, es in members) for key, members in groups.items()}
    if sum(capacity.values()) != sum(counts.values()):
        raise ValueError('Role counts must exhaust eligible rows, including unused')
    remaining, quotas = capacity.copy(), {}
    for role in ROLES:
        quotas[role] = apportion(counts[role], remaining)
        remaining = {k: n-quotas[role][k] for k, n in remaining.items()}
    assigned = [np.full(len(a['jet_label']), 255, np.uint8) for a in scalars]
    for key in sorted(groups):
        chunks = []
        for index, entries in groups[key]:
            boundaries = np.flatnonzero(np.diff(entries//BLOCK_ROWS))+1
            for es in np.split(entries, boundaries):
                block = int(es[0]//BLOCK_ROWS)
                digest = hashlib.sha256(f'FULLSTAT_SPLIT/{SEED}/{records[index]["sha256"]}/{block}'.encode()).digest()
                chunks.append((digest, index, es))
        chunks.sort(key=lambda row: (row[0], row[1], int(row[2][0])))
        role_index, left = 0, quotas[ROLES[0]][key]
        for _, index, entries in chunks:
            start = 0
            while start < len(entries):
                while left == 0:
                    role_index += 1
                    left = quotas[ROLES[role_index]][key]
                take = min(left, len(entries)-start)
                assigned[index][entries[start:start+take]] = role_index
                left -= take
                start += take
    for keep, roles in zip(eligible_by_file, assigned):
        if not np.array_equal(roles != 255, keep):
            raise ValueError('Split eligible coverage differs')
    return assigned, labels_by_file, quotas


def build(container, output, *, gate_hash, counts=COUNTS, expected_files=16):
    """Low-level core supports small test fixtures; production wrapper pins counts."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    parent, records = transfer(container, expected_files)
    all_arrays, schema = [], None
    for record in records:
        path, _ = source_path(Path(container)/'raw', record['path'])
        if sha256_file(path) != record['sha256']:
            raise ValueError('Source checksum differs')
        with uproot.open(path) as handle:
            key, tree = latest_tree(handle)
            current = validate_tree_schema(tree)
            if schema is not None and current != schema:
                raise ValueError('FullStat ROOT schemas differ')
            schema = current
            pieces = {k: [] for k in SCALARS}
            for arrays in tree.iterate(list(SCALARS), step_size=100000, library='np'):
                eligible(arrays, record['category'])
                for name in SCALARS:
                    pieces[name].append(arrays[name])
            arrays = {k: np.concatenate(v) for k, v in pieces.items()}
            _, labels, flow = eligible(arrays, record['category'])
            record.update(tree_key=key, entries=len(labels), bytes=path.stat().st_size, cutflow=flow)
            all_arrays.append(arrays)
        if sha256_file(path) != record['sha256']:
            raise ValueError('Source changed during scalar scan')
        print(f'FullStat scalar scan {len(all_arrays)}/{len(records)}: {record["path"]}', flush=True)
    assignments, labels, quotas = partition(records, all_arrays, counts)
    for index, (record, arrays, roles, mapped) in enumerate(zip(records, all_arrays, assignments, labels)):
        path = output/'memberships'/f'{index:02d}.npz'
        atomic_publish_bytes(path, deterministic_npz_bytes(dict(**arrays, roles=roles, labels=mapped)))
        record['membership'] = file_ref(path, root=output)
        record['role_counts'] = {r: int(np.count_nonzero(roles == j)) for j, r in enumerate(ROLES)}
        record['role_particles'] = {r: {side: int(arrays[name][roles == j].sum())
            for side, name in [('hlt', 'hlt_jet_nparticles'), ('offline', 'jet_nparticles')]}
            for j, r in enumerate(ROLES)}
    if transfer(container, expected_files)[0] != parent:
        raise ValueError('Transfer metadata changed during scan')
    result = artifact('FOUNDATION', parents=dict(gate=gate_hash), transfer=parent, files=records,
        schema=schema, schema_sha256=canonical_sha256(schema), counts=counts, quotas=quotas,
        policy=split_policy(), class_names=list(CLASS_NAMES), particle_arrays_decoded=False,
        raw_entries=sum(r['entries'] for r in records))
    write_immutable_json(output/'foundation.json', result)
    load(output)
    return result


def load(root, *, expected_gate=None):
    root = Path(root)
    value = load_json(root/'foundation.json')
    validate(value, 'FOUNDATION', None if expected_gate is None else dict(gate=expected_gate))
    if (value['policy'] != split_policy() or value['particle_arrays_decoded'] is not False
            or value['transfer']['source'] != SOURCE or value['class_names'] != list(CLASS_NAMES)
            or value['schema_sha256'] != canonical_sha256(value['schema'])):
        raise ValueError('FullStat foundation policy differs')
    records = value['files']
    if ([r['path'] for r in records] != sorted({r['path'] for r in records})
            or len({r['sha256'] for r in records}) != len(records)):
        raise ValueError('Source duplicate/order differs')
    payloads, scalars = [], []
    for index, record in enumerate(records):
        _, category = source_path(Path.cwd(), record['path'])
        if record['category'] != category or record['membership']['path'] != f'memberships/{index:02d}.npz':
            raise ValueError('Source/category/membership path differs')
        arrays = load_npz_arrays(validate_file_ref(record['membership'], root=root))
        if set(arrays) != set(SCALARS) | {'labels', 'roles'}:
            raise ValueError('Membership fields differ')
        current = {k: arrays[k] for k in SCALARS}
        keep, labels, flow = eligible(current, category)
        if (len(keep) != record['entries'] or flow != record['cutflow']
                or arrays['roles'].dtype != np.uint8 or arrays['labels'].dtype != np.int8
                or not np.array_equal(arrays['labels'], labels)):
            raise ValueError('Membership/scalar consistency differs')
        payloads.append(arrays)
        scalars.append(current)
    roles, _, quotas = partition(records, scalars, value['counts'])
    if quotas != value['quotas'] or sum(r['entries'] for r in records) != value['raw_entries']:
        raise ValueError('Foundation totals/quotas differ')
    for record, arrays, expected in zip(records, payloads, roles):
        if not np.array_equal(arrays['roles'], expected):
            raise ValueError('Deterministic split replay differs')
        for j, role in enumerate(ROLES):
            selected = expected == j
            particles = {side: int(arrays[name][selected].sum()) for side, name in
                         [('hlt', 'hlt_jet_nparticles'), ('offline', 'jet_nparticles')]}
            if int(selected.sum()) != record['role_counts'][role] or particles != record['role_particles'][role]:
                raise ValueError('Role counts/particle reservation differ')
    return value, payloads


def ordinary_indices(membership, role):
    if role not in ORDINARY:
        raise PermissionError('Final-test/unused particle access is sealed')
    return np.flatnonzero(membership['roles'] == ROLES.index(role))


def authenticate_source(container, record):
    path, _ = source_path(Path(container)/'raw', record['path'])
    if path.stat().st_size != record['bytes'] or sha256_file(path) != record['sha256']:
        raise ValueError('Raw source bytes differ')
    return path

"""Filter frozen extended roles; never refill splits or export test row arrays."""
from pathlib import Path
import numpy as np
from hlt_classification.data.cache_contracts import (
    load_json, load_npz_arrays, deterministic_npz_bytes, atomic_publish_bytes, write_immutable_json)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_new_balanced_hydra import population as parent
from . import contracts as c
from .sampling import BalancedSampler

original, extend, ROLES = parent.original, parent.extend, parent.ROLES


def selected(member):
    return (member['labels'] >= 0) & (member['labels'] < 6)


def masks(member, extended_roles):
    six = selected(member)
    inside = (member['jet_pt'] >= c.EDGES[0]) & (member['jet_pt'] < c.EDGES[-1])
    return dict(train=(extended_roles == 0) & six & inside,
                validation=(extended_roles == 1) & six & inside,
                outside_validation=(extended_roles == 1) & six & ~inside)


def inventory(f, members, roles):
    counts = {r: 0 for r in (*ROLES, 'unused')}
    outside = dict(counts)
    arrays = {r: {k: [] for k in ('identities', 'labels', 'pt')}
              for r in ('train', 'validation', 'outside_validation')}
    particles = {r: dict(hlt=0, offline=0) for r in arrays}
    maxima = {r: dict(hlt=0, offline=0, concat=0) for r in arrays}
    class_counts = {r: np.zeros(6, np.int64) for r in counts}
    for record, a, rs in zip(f['files'], members, roles, strict=True):
        if rs.shape != a['roles'].shape:
            raise ValueError('Role coverage differs')
        six = selected(a)
        inside = (a['jet_pt'] >= c.EDGES[0]) & (a['jet_pt'] < c.EDGES[-1])
        for code, role in enumerate(counts):
            mask = (rs == code) & six & inside
            counts[role] += int(mask.sum())
            outside[role] += int(((rs == code) & six & ~inside).sum())
            class_counts[role] += np.bincount(a['labels'][mask], minlength=6)
        for role, mask in masks(a, rs).items():
            es = np.flatnonzero(mask)
            ids = np.asarray([np.frombuffer(bytes.fromhex(original.identity(record, int(e))), np.uint8)
                              for e in es], np.uint8).reshape(-1, 32)
            for k, v in [('identities', ids), ('labels', a['labels'][es].astype(np.int64)),
                         ('pt', a['jet_pt'][es].astype(np.float64))]:
                arrays[role][k].append(v)
            nh, no = a['hlt_jet_nparticles'][es], a['jet_nparticles'][es]
            particles[role]['hlt'] += int(nh.sum()); particles[role]['offline'] += int(no.sum())
            for k, v in [('hlt', nh), ('offline', no), ('concat', nh+no)]:
                maxima[role][k] = max(maxima[role][k], int(v.max()) if len(v) else 0)
    joined = {f'{r}_{k}': np.concatenate(v) for r, fields in arrays.items() for k, v in fields.items()}
    return counts, outside, particles, maxima, joined, {k:v.tolist() for k,v in class_counts.items()}


def describe(f, members, roles, folder, production):
    counts, outside, particles, maxima, arrays, class_counts = inventory(f, members, roles)
    if production and (counts != dict(c.COUNTS, unused=c.UNUSED) or outside != c.OUTSIDE):
        raise ValueError('Six-class audited population differs')
    sampler = BalancedSampler(arrays['train_labels'], arrays['train_pt'])
    return arrays, dict(policy=c.population_policy(), counts=counts, outside_counts=outside,
        role_particles=particles, maximum_support=maxima, class_counts=class_counts,
        class_bin_counts=sampler.counts.tolist(), production=production,
        role_population=file_ref(folder/'role_parent/population.json', root=folder))


def build(dataset, folder, *, production=True):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    # Keep the donor domain/algorithm: its unused-only extension is unchanged.
    print('SIXCLASS population: replaying frozen foundation and extended split roles (scalar metadata only).', flush=True)
    parent.build(dataset, folder/'role_parent', production=production)
    p, f, members, roles, _, _ = parent.load(dataset, folder/'role_parent', production=production)
    if production and p['content_hash'] != c.ROLE_POPULATION_HASH:
        raise ValueError('Wrong role parent')
    arrays, fields = describe(f, members, roles, folder, production)
    atomic_publish_bytes(folder/'rows.npz', deterministic_npz_bytes(arrays))
    report = c.artifact('POPULATION', parents=dict(foundation=f['content_hash'], role_population=p['content_hash']),
                        rows=file_ref(folder/'rows.npz', root=folder), **fields)
    if production and report['content_hash'] != c.POPULATION_HASH:
        raise ValueError('Population differs from authenticated local audit')
    write_immutable_json(folder/'population.json', report)
    print('SIXCLASS population authenticated:', report['content_hash'], report['counts'], flush=True)
    return report


def load(dataset, folder, *, production=True):
    folder = Path(folder)
    p, f, members, roles, _, _ = parent.load(dataset, folder/'role_parent', production=production)
    report = load_json(folder/'population.json')
    c.validate(report, 'POPULATION', dict(foundation=f['content_hash'], role_population=p['content_hash']))
    if (production and (p['content_hash'] != c.ROLE_POPULATION_HASH or
                        report['content_hash'] != c.POPULATION_HASH or report['production'] is not True)):
        raise ValueError('Frozen population identity differs')
    arrays, fields = describe(f, members, roles, folder, production)
    if any(report[k] != v for k,v in fields.items()):
        raise ValueError('Population policy/summary replay differs')
    validate_file_ref(report['role_population'], root=folder)
    saved = load_npz_arrays(validate_file_ref(report['rows'], root=folder))
    if set(saved) != set(arrays) or any(saved[k].dtype != v.dtype or not np.array_equal(saved[k],v)
                                      for k,v in arrays.items()):
        raise ValueError('Canonical TRAIN/validation rows differ')
    sampler = BalancedSampler(arrays['train_labels'], arrays['train_pt'], expected_counts=report['class_bin_counts'])
    return report, f, members, roles, arrays, sampler

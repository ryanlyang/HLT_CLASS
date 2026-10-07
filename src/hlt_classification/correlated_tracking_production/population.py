"""Same existing 2.25M memberships; raw literature-compatible offline reader."""
import awkward as ak
import numpy as np

from hlt_classification.cms2jc2_production import population as registered
from hlt_classification.cms2jc2_production.contracts import validate as validate_population
from hlt_classification.cms2jc2_response.readers import authenticated_open
from hlt_classification.jetclass2_delphes.inventory import latest_tree
from hlt_classification.jetclass2_delphes.split_registry import unpack_entries
from hlt_classification.literature_proxy.population import BRANCHES, JC2_FIELDS, from_columns
from .contracts import digest_ids, safe, validate

COUNTS = registered.COUNTS
build = registered.build  # Metadata only: unchanged registry and validation subset.
entries_for = registered.entries_for
ids = registered.ids
SHARD_JETS = 25_000


def validate_shards(population, specs):
    """Cheap complete metadata partition check; per-shard reader verifies ID digests."""
    index = 0
    for role in COUNTS:
        for row in population['files']:
            if row['role'] != role:
                continue
            for start in range(0, row['jets'], SHARD_JETS):
                stop = min(start+SHARD_JETS, row['jets'])
                if index >= len(specs):
                    raise ValueError('Missing registered shard')
                actual = specs[index]
                expected = dict(shard_id=f's{index:05d}', role=role, path=row['path'],
                                start=start, stop=stop, jets=stop-start)
                if ({k: v for k, v in actual.items() if k != 'ordered_identities'} != expected
                        or len(actual.get('ordered_identities', '')) != 64):
                    raise ValueError('Shard coverage/order differs')
                index += 1
    if index != len(specs):
        raise ValueError('Extra registered shards')


def shards(population):
    validate_population(population, 'POPULATION', test=False)
    out = []
    for role in COUNTS:
        for row in population['files']:
            if row['role'] != role:
                continue
            entries = unpack_entries(row['entry_mask'], row['raw_entries'])
            for start in range(0, len(entries), SHARD_JETS):
                stop = min(start+SHARD_JETS, len(entries))
                out.append(dict(shard_id=f's{len(out):05d}', role=role, path=row['path'],
                    start=start, stop=stop, jets=stop-start,
                    ordered_identities=digest_ids(ids(population['parents']['inventory'], row, entries[start:stop]))))
    return out


def iterate(data_root, population, shard, *, test_lock=None, selected=None):
    """Authenticate file and membership BEFORE opening; selected is train gate only."""
    if shard['role'] == 'final_test':
        if test_lock is None:
            raise PermissionError('Sealed test requires materialization lock before opening')
        validate(test_lock, 'TEST_BUILD_LOCK', test=False)
        if (test_lock['parents'].get('population') != population['content_hash']
                or test_lock['evaluate'] is not False or test_lock['materialize'] is not True):
            raise PermissionError('Wrong sealed test materialization lock')
    row, entries = entries_for(population, shard)
    if selected is not None:
        chosen = np.asarray(selected)
        if (shard['role'] != 'train' or chosen.dtype.kind not in 'iu' or chosen.ndim != 1
                or len(chosen) == 0 or not np.array_equal(chosen, np.unique(chosen))
                or not np.isin(chosen, entries).all()):
            raise PermissionError('Preflight selection must be a registered training subset')
        entries = chosen
    if not len(entries):
        raise ValueError('Empty production shard')
    with authenticated_open(safe(data_root, row['path']), row['sha256']) as handle:
        key, tree = latest_tree(handle)
        if key != row['tree_key'] or tree.num_entries != row['raw_entries']:
            raise ValueError('Offline ROOT tree identity differs')
        buckets = {}
        for entry, identity in zip(entries, ids(population['parents']['inventory'], row, entries)):
            buckets.setdefault(int(entry)//512, []).append((int(entry), identity))
        for bucket, rows in buckets.items():
            start = bucket*512
            values = tree.arrays(list(BRANCHES), entry_start=start,
                entry_stop=min(start+512, row['raw_entries']), library='ak', how=dict)
            for entry, identity in rows:
                index = entry-start
                n = int(values['jet_nparticles'][index])
                columns = {f: ak.to_numpy(values['part_'+f][index]) for f in JC2_FIELDS}
                if n < 0 or any(len(a) != n for a in columns.values()):
                    raise ValueError('Offline particle count differs')
                yield identity, from_columns(columns)

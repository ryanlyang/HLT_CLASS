"""At most two required views in one read, replaying the unchanged stage-2 bytes."""
from pathlib import Path
import itertools

from hlt_classification.context_fusion.inputs import PairedCache
from hlt_classification.jetclass2_delphes.cache import RamCache
from . import stage2 as s
from .particles import ParticleReader, RawJet
from .preflight_cache import _values, _block, resident_bound


def build(root, *, role, coordinates, max_ram_bytes):
    if role not in ('train', 'validation'):
        raise PermissionError('Final-test science cache access is sealed')
    coordinates = tuple(coordinates)
    if (not 1 <= len(coordinates) <= 2 or len(set(coordinates)) != len(coordinates)
            or not set(coordinates) <= set(s.COORDINATES)
            or type(max_ram_bytes) is not int or max_ram_bytes <= 0):
        raise ValueError('Unregistered science views/budget')
    root = Path(root)
    p, f, inventory, splits = s.load_prepared(root)
    if resident_bound(p, role, coordinates) > max_ram_bytes:
        raise MemoryError('Science cache exceeds explicit budget')
    refs = {r['file_index']: r for r in p['assignments'][role]}
    meters, blocks = {c: s.ViewMeter() for c in coordinates}, {c: [] for c in coordinates}
    reader = ParticleReader(f['input_container'], inventory, splits, role=role,
        include_offline=any(c != 'D000' for c in coordinates))
    for index, jets in itertools.groupby(reader, key=lambda j: j.file_index):
        assigned, rows = s.load_assignment(root, refs[index]), {c: [] for c in coordinates}
        for i, jet in enumerate(jets):
            for c, value in _values(jet, assigned, i, p, coordinates).items():
                meters[c].add(jet, value)
                rows[c].append((RawJet(jet.identity, jet.label, index, jet.entry, None, None), value))
        if len(rows[coordinates[0]]) != refs[index]['rows']:
            raise ValueError('Assignment coverage differs')
        for c in coordinates:
            blocks[c].append(_block(index, rows[c]))
        print(f'LUKA science-cache role={role} files={len(blocks[coordinates[0]])}/{len(refs)} '
              f'rows={meters[coordinates[0]].rows} views={coordinates}', flush=True)
    result = {}
    for c in coordinates:
        if meters[c].report() != p['summaries'][role][c]:
            raise ValueError(f'Prepared input replay differs: {role}/{c}')
        result[c] = RamCache(blocks[c], role=role, coordinate_name=c, foundation_sha256=p['content_hash'])
        if len(result[c]) != p['counts'][role]:
            raise ValueError('Science population differs')
    if sum(cache.nbytes for cache in result.values()) > max_ram_bytes:
        raise MemoryError('Science cache residency differs')
    return result


def cache(spec, p, node, role):
    coords = tuple(dict.fromkeys(c for c in (node['coordinate'], node['context_coordinate']) if c is not None))
    bounds = {r: resident_bound(p, r, coords) for r in ('train', 'validation')}
    if sum(bounds.values()) > .15*spec['memory_mb']*1024**2:
        raise MemoryError('Science caches exceed reserved host envelope')
    bundle = build(spec['admission']['prepared_root'], role=role, coordinates=coords, max_ram_bytes=bounds[role])
    primary = bundle[node['coordinate']]
    return primary if node['context_coordinate'] is None else PairedCache(primary, bundle[node['context_coordinate']])

"""One selected-file read per role; physical swaps and authenticated RAM views."""
from hashlib import sha256
from pathlib import Path
import time
import numpy as np

from hlt_classification.jetclass2_delphes.cache import RamCache
from hlt_classification.luka_fullsim import regular_cache, stage2, alljets_data
from hlt_classification.luka_fullsim.particles import physical, RawJet
from hlt_classification.luka_fullsim.preflight_cache import _block
from . import views
from .contracts import artifact, input_identity

COORDINATES = ('U100', 'D066', 'D033', 'D000')


def build(gate, *, authenticated=None):
    start = time.monotonic()
    root = Path(gate['anchor']['evidence']['prepared_root'])
    meta = regular_cache.metadata(root) if authenticated is None else authenticated
    manifest, prepared, inventory, splits = meta
    if (manifest['content_hash'] != gate['anchor']['evidence']['cache']
            or prepared['content_hash'] != gate['anchor']['evidence']['prepared']
            or gate['input_identity'] != input_identity(gate['anchor'])):
        raise ValueError('Swap parent/input identity differs')
    budget = int(gate['memory_mb']*1024**2*.30)
    bound = 8*sum(prepared['summaries'][r]['D000']['resident_bytes'] for r in ('train','validation'))+64*2**20
    if bound > budget:
        raise MemoryError('All-view RAM construction bound exceeds registered budget')
    caches, audits = {}, {}
    original = None
    for role in ('train','validation'):
        refs = {r['file_index']:r for r in prepared['assignments'][role]}
        meters = {c:stage2.ViewMeter() for c in COORDINATES}
        blocks = {c:[] for c in COORDINATES}
        schedule_hash = sha256()
        group_counts = np.zeros(3, np.int64)
        unmatched, max_group_pt_share = 0, 0.
        for member in splits['memberships'][role]:
            index = member['file_index']
            assigned = stage2.load_assignment(root, refs[index])
            rows = {c:[] for c in COORDINATES}
            count = 0
            for jet in regular_cache.selected_file_rows(gate['anchor']['evidence']['raw_container'], inventory, member):
                if (count >= len(assigned['labels']) or bytes(assigned['identities'][count]).hex() != jet.identity
                        or int(assigned['labels'][count]) != jet.label):
                    raise ValueError('Swap assignment identity/label order differs')
                lo, hi = assigned['offsets'][count:count+2]
                mapping = assigned['mapping'][lo:hi]
                hlt = physical(jet.hlt, prepared['conventions'], 'hlt')
                off = physical(jet.offline, prepared['conventions'], 'offline')
                if len(mapping) != len(hlt):
                    raise ValueError('Swap HLT support differs')
                values, groups = views.all_views(identity=jet.identity, hlt=hlt, offline=off, mapping=mapping)
                schedule_hash.update(bytes.fromhex(jet.identity))
                schedule_hash.update(np.asarray([len(groups)], '<i8').tobytes())
                schedule_hash.update(groups.tobytes())
                total_pt = float(hlt.pt[groups >= 0].sum())
                for b in range(3):
                    group_counts[b] += int(np.sum(groups == b))
                    if total_pt:
                        max_group_pt_share = max(max_group_pt_share, float(hlt.pt[groups == b].sum())/total_pt)
                unmatched += int(np.sum(groups < 0))
                light = RawJet(jet.identity, jet.label, index, jet.entry, None, None)
                for c, particles in values.items():
                    value = stage2.build_inputs(particles, capacity=prepared['inputs']['capacity'])
                    meters[c].add(light, value)
                    rows[c].append((light, value))
                count += 1
            if count != refs[index]['rows'] or count != len(assigned['labels']):
                raise ValueError('Swap selected file coverage differs')
            for c in COORDINATES:
                blocks[c].append(_block(index, rows[c]))
            print(f'LUKA swap inputs role={role} files={len(blocks["D000"])}/{len(refs)} rows={meters["D000"].rows}', flush=True)
        reports = {c:m.report() for c,m in meters.items()}
        for c in ('U100','D000'):
            if reports[c] != prepared['summaries'][role][c]:
                raise ValueError(f'Exact original endpoint input replay differs: {role}/{c}')
        for c in COORDINATES:
            cache = RamCache(blocks[c], role=role, coordinate_name=c,
                             foundation_sha256=gate['input_identity']['content_hash'])
            if len(cache) != gate['counts'][role]:
                raise ValueError('Swap population differs')
            caches[role,c] = cache
        if role == 'validation':
            v = caches[role,'D000']
            original = np.zeros(len(v), bool)
            original[alljets_data.comparison_indices(inventory, splits, v.identities, v.labels)] = True
        audits[role] = dict(views=reports, schedule_sha256=schedule_hash.hexdigest(),
            replacement_counts=group_counts.tolist(), unmatched_hlt_slots=unmatched,
            maximum_single_batch_matched_pt_fraction=max_group_pt_share)
    resident = sum(c.nbytes for c in caches.values())
    if resident > budget:
        raise MemoryError('Actual swap cache RAM exceeds budget')
    audit = artifact('INPUT_AUDIT', parents=dict(inputs=gate['input_identity']['content_hash']),
        roles=audits, endpoint_fingerprints_replayed=True, resident_bytes=resident)
    return dict(caches=caches, audit=audit, original_validation=original,
                construction_seconds=time.monotonic()-start, max_ram_bytes=budget)


def get(bundle, role, coordinate):
    if role not in ('train','validation') or coordinate not in COORDINATES:
        raise PermissionError('Only registered ordinary-role swap views')
    return bundle['caches'][role,coordinate]

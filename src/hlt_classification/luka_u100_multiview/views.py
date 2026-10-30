"""Full-support physical union; existing student views remain byte-identical."""
import numpy as np
from hlt_classification.cms2jc2_response.bridge import Particles
from hlt_classification.jetclass2_delphes.cache import RamCache
from hlt_classification.literature_context.transform import build_inputs
from hlt_classification.luka_fullsim import regular_cache
from hlt_classification.luka_fullsim.particles import physical
from hlt_classification.luka_fullsim.preflight_cache import _block


def union(hlt, offline):
    """Canonical ordering depends only on physical fields, never source/keys."""
    fields = [np.concatenate((getattr(hlt, k), getattr(offline, k)))
              for k in ('p4', 'charge', 'category', 'tracking', 'valid')]
    p4, charge, category, tracking, valid = fields
    tie = np.column_stack((p4, charge, category, tracking, valid))
    order = np.lexsort(tuple(tie[:, j] for j in reversed(range(tie.shape[1]))) +
                       (-np.hypot(p4[:, 0], p4[:, 1]),))
    # New arbitrary keys are metadata only; neither source names nor keys reach inputs.
    return Particles(*(f[order] for f in fields), tuple(map(str, range(len(order)))))


def join(primary, context):
    if (primary.role != context.role or primary.foundation_sha256 != context.foundation_sha256
            or not np.array_equal(primary.identities, context.identities)
            or not np.array_equal(primary.labels, context.labels)):
        raise ValueError('Paired views have different populations/order/labels')


class PairedCache:
    def __init__(self, primary, context):
        join(primary, context)
        self.primary, self.context = primary, context
        for key in ('role', 'foundation_sha256', 'labels', 'identities'):
            setattr(self, key, getattr(primary, key))
        self.coordinate_name = 'FUSION'
        self.nbytes = primary.nbytes + context.nbytes

    def __len__(self):
        return len(self.primary)

    def batch(self, indices):
        result = self.primary.batch(indices)
        context = self.context.batch(indices)
        result.update({'context_'+k: context[k] for k in ('features', 'vectors', 'mask')})
        return result


def load(spec, role, coordinate, meta=None):
    if role not in ('train', 'validation'):
        raise PermissionError('No final-test view')
    anchor = spec['anchor']['evidence']
    meta = meta or regular_cache.metadata(anchor['prepared_root'])
    manifest, prepared, inventory, splits = meta
    if (manifest['content_hash'] != anchor['cache'] or prepared['content_hash'] != anchor['prepared']):
        raise ValueError('Historical prepared lineage differs')
    budget = int(spec['memory_mb']*1024**2*.22)
    if coordinate == 'FUSION':
        result = PairedCache(load(spec, role, 'D000', meta), load(spec, role, 'OFFLINE', meta))
    elif coordinate == 'CONCAT':
        # Conservative allocation bound includes both supports, per-file transient copies.
        summaries = prepared['summaries'][role]
        bound = 2*sum(summaries[c]['resident_bytes'] for c in ('D000', 'OFFLINE')) + 2**30
        if bound > budget:
            raise MemoryError('Union RAM bound exceeds allocation; no truncation or spilling')
        blocks = []
        for member in splits['memberships'][role]:
            rows = []
            for jet in regular_cache.selected_file_rows(anchor['raw_container'], inventory, member):
                hlt = physical(jet.hlt, prepared['conventions'], 'hlt')
                off = physical(jet.offline, prepared['conventions'], 'offline')
                combined = union(hlt, off)
                value = build_inputs(combined, capacity=max(16, len(combined)))
                rows.append((jet, value))
            blocks.append(_block(member['file_index'], rows))
            if sum(b.nbytes for b in blocks) > budget:
                raise MemoryError('Union resident bound exceeded')
            print(f'MULTIVIEW cache={coordinate}/{role} files={len(blocks)}', flush=True)
        result = RamCache(blocks, role=role, foundation_sha256=prepared['content_hash'], coordinate_name=coordinate)
        from hlt_classification.luka_fullsim.alljets_data import row_arrays
        ids, labels = row_arrays(inventory, splits['memberships'][role])
        if not np.array_equal(ids, result.identities) or not np.array_equal(labels, result.labels):
            raise ValueError('Union row coverage differs')
    elif coordinate == 'OFFLINE':
        from .offline_cache import load as load_offline
        result = load_offline(anchor, role, meta, budget)
    elif coordinate in ('D000', 'U100'):
        result = regular_cache.load(anchor['prepared_root'], role=role, coordinate=coordinate,
                                    authenticated=meta, max_ram_bytes=budget)
    else:
        raise ValueError('Unregistered multiview coordinate')
    if len(result) != spec['counts'][role]:
        raise ValueError('Population count differs')
    return result


def lengths(cache):
    if hasattr(cache, 'cache') and hasattr(cache, 'indices'):
        return lengths(cache.cache)[cache.indices]
    if isinstance(cache, PairedCache):
        return lengths(cache.primary) + lengths(cache.context)
    return np.concatenate([np.diff(b.offsets) for b in cache.blocks])

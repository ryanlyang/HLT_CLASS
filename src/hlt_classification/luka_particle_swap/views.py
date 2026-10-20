"""Label-free, nested whole-record replacement on persistent HLT slots."""
from functools import lru_cache
from hashlib import sha256
import re
import numpy as np

from hlt_classification.cms2jc2_response.bridge import Particles
from hlt_classification.cms_proxy_ladder.views import build_view as original_view, validate_pairing
from .contracts import views


@lru_cache(maxsize=1)
def _domain():
    return views()['content_hash']


def _tie(identity, *parts):
    return sha256((':'.join((_domain(), identity, *map(str, parts)))).encode()).digest()


def replacement_groups(hlt, mapping, *, identity):
    """-1 for unmatched slots; 0/1/2 for the three nested replacement batches.

    All ranking uses HLT-only pT/charge. No offline quality, label, model or
    validation quantity can be supplied to this function.
    """
    if not isinstance(identity, str) or re.fullmatch('[0-9a-f]{64}', identity) is None:
        raise ValueError('Canonical jet identity required')
    mapping = np.asarray(mapping)
    if mapping.dtype.kind not in 'iu' or mapping.shape != (len(hlt),) or np.any(mapping < -1):
        raise ValueError('Invalid HLT assignment shape')
    if not np.isfinite(hlt.pt).all() or np.any(hlt.pt <= 0):
        raise ValueError('Invalid native HLT momentum')
    slots = np.flatnonzero(mapping >= 0)
    result = np.full(len(hlt), -1, np.int8)
    n = len(slots)
    if not n:
        return result
    scale = views()['quantization']
    shares = np.floor(hlt.pt[slots]/float(np.sum(hlt.pt[slots]))*scale+.5).astype(np.int64)
    order = sorted(range(n), key=lambda i: (-int(shares[i]), _tie(identity, 'slot', int(slots[i]))))
    # Rank bins have equal counts up to rounding; keep native slot order in outputs.
    stratum = {i: 2*min(2, 3*rank//n)+int(hlt.charge[slots[i]] != 0) for rank, i in enumerate(order)}
    sizes = np.bincount(list(stratum.values()), minlength=6)
    priority = sorted(range(3), key=lambda b: _tie(identity, 'batch', b))
    capacity = [n//3]*3
    for b in priority[:n % 3]:
        capacity[b] += 1
    count, momentum, types = [0]*3, [0]*3, [[0]*6 for _ in range(3)]
    count_unit = (2*scale+n)//(2*n)
    for i in order:
        s, p = stratum[i], int(shares[i])
        type_unit = (2*scale+int(sizes[s]))//(2*int(sizes[s]))
        def cost(b):
            return (2*count[b]+1)*count_unit**2 + 2*momentum[b]*p+p*p + (2*types[b][s]+1)*type_unit**2
        feasible = [b for b in priority if count[b] < capacity[b]]
        chosen = min(feasible, key=lambda b: (cost(b), priority.index(b)))
        result[slots[i]] = chosen
        count[chosen] += 1
        momentum[chosen] += p
        types[chosen][s] += 1
    if count != capacity:
        raise RuntimeError('Replacement capacity invariant failed')
    return result


def build_view(*, identity, hlt, offline, mapping, coordinate, groups=None):
    if coordinate not in ('U100', 'D066', 'D033', 'D000'):
        raise ValueError('Unregistered swap coordinate')
    # Native deployment has no offline/matching requirement.
    if coordinate == 'D000':
        return hlt
    mapping = np.asarray(mapping)
    validate_pairing(mapping, nh=len(hlt), no=len(offline))
    expected = replacement_groups(hlt, mapping, identity=identity)
    if groups is not None and not np.array_equal(groups, expected):
        raise ValueError('Replacement schedule differs')
    return _assemble(identity, hlt, offline, mapping, coordinate, expected)


def _assemble(identity, hlt, offline, mapping, coordinate, groups):
    if coordinate == 'D000':
        return hlt
    if coordinate == 'U100':
        return original_view(identity=identity, proxy=hlt, offline=offline, mapping=mapping, coordinate='U100')
    swapped = {'D066': 1, 'D033': 2}[coordinate]
    slots = np.flatnonzero((mapping >= 0) & (groups >= swapped))
    fields = {}
    for name in ('p4', 'charge', 'category', 'tracking', 'valid'):
        value = np.array(getattr(hlt, name), copy=True)
        value[slots] = getattr(offline, name)[mapping[slots]]
        fields[name] = value
    return Particles(**fields, keys=tuple('proxy:'+key for key in hlt.keys))


def all_views(*, identity, hlt, offline, mapping):
    """One schedule calculation per jet; no stochastic state or label access."""
    mapping = np.asarray(mapping)
    validate_pairing(mapping, nh=len(hlt), no=len(offline))
    groups = replacement_groups(hlt, mapping, identity=identity)
    return {c: _assemble(identity, hlt, offline, mapping, c, groups)
            for c in ('U100', 'D066', 'D033', 'D000')}, groups

"""Exact field substitution, retaining the authenticated HLT particle slots."""
import hashlib
import numpy as np
from hlt_classification.jetclass2_delphes.cache import RamBlock, RamCache
from .contracts import GROUPS, TEACHERS, view_recipe, artifact


def splice(hlt, u100, fields):
    if fields not in (*TEACHERS, '', 'KIT'):
        raise ValueError('Unknown field substitution')
    if (hlt.file_index != u100.file_index or any(not np.array_equal(getattr(hlt, k), getattr(u100, k))
            for k in ('identities', 'labels', 'offsets'))
            or hlt.features.shape != u100.features.shape or hlt.vectors.shape != u100.vectors.shape
            or hlt.features.shape[1:] != (17,) or hlt.vectors.shape[1:] != (4,)):
        raise ValueError('D000/U100 particle support/order differs')
    features = hlt.features.copy()
    for group in fields:
        features[:, GROUPS[group]] = u100.features[:, GROUPS[group]]
    vectors = (u100.vectors if 'K' in fields else hlt.vectors).copy()
    if not np.isfinite(features).all() or not np.isfinite(vectors).all():
        raise ValueError('Nonfinite selective input')
    return RamBlock(hlt.file_index, hlt.offsets, features, vectors, hlt.identities, hlt.labels)


def hybrid(hlt, u100, task):
    if (task not in TEACHERS or hlt.coordinate_name != 'D000' or u100.coordinate_name != 'U100'
            or hlt.role != u100.role or hlt.role not in ('train', 'validation')
            or hlt.foundation_sha256 != u100.foundation_sha256 or len(hlt.blocks) != len(u100.blocks)):
        raise PermissionError('Selective view source differs')
    blocks = [splice(a, b, task) for a, b in zip(hlt.blocks, u100.blocks)]
    return RamCache(blocks, role=hlt.role, coordinate_name=task, foundation_sha256=hlt.foundation_sha256)


def fingerprint(cache):
    h = hashlib.sha256()
    for block in cache.blocks:
        for name, dtype in (('identities', 'u1'), ('labels', '<i8'), ('offsets', '<i8'),
                            ('features', '<f4'), ('vectors', '<f4')):
            a = np.ascontiguousarray(getattr(block, name), dtype=dtype)
            h.update(name.encode()+str(a.shape).encode()+a.tobytes())
    return artifact('VIEW', parents=dict(prepared=cache.foundation_sha256, recipe=view_recipe()['content_hash']),
        role=cache.role, coordinate=cache.coordinate_name, rows=len(cache), sha256=h.hexdigest())

"""Bounded temporary ROOT basket buffers; finalize only authorized row spans.

Uproot normally reinterprets a complete basket for every tiny ``tree.arrays``
request. Cache its temporary numeric buffers, then use Uproot's public final
interpretation step to trim to the caller's original authorized span. No full
file/role array, particle object or scientific cache is constructed here.
"""
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import uproot

from hlt_classification.luka_fullsim.particles import FIELDS

CACHE_BYTES = 128 * 2**20
_OPEN = None


def buffer_bytes(value):
    if isinstance(value, np.ndarray) and not value.dtype.hasobject:
        return value.nbytes
    if isinstance(value, uproot.interpretation.jagged.JaggedArray):
        return buffer_bytes(value.offsets) + buffer_bytes(value.content)
    raise TypeError('Only numeric and numeric-jagged ROOT buffers are authorized')


class Reader:
    def __init__(self, path, tree_key, *, cache_bytes=CACHE_BYTES):
        if type(cache_bytes) is not int or cache_bytes < 0:
            raise ValueError('Nonnegative integer basket-cache allowance required')
        self.handle = uproot.open(path, object_cache=None, array_cache=None)
        try:
            self.tree = self.handle[tree_key]
        except BaseException:
            self.handle.close()
            raise
        self.cache_bytes = cache_bytes
        self.cache = OrderedDict()
        self.resident_bytes = self.peak_bytes = self.reads = self.hits = 0
        self.library = uproot.interpretation.library.Awkward()
        self.numpy_library = uproot.interpretation.library.NumPy()
        self.num_entries = self.tree.num_entries

    def close(self):
        self.cache.clear()
        self.resident_bytes = 0
        self.handle.close()

    def temporary(self, branch, basket_num):
        key = (branch.cache_key, branch.interpretation.cache_key, basket_num)
        if key in self.cache:
            self.hits += 1
            value = self.cache.pop(key)
            self.cache[key] = value
            return value[0]
        basket = branch.basket(basket_num)
        interpretation = branch.interpretation
        if not isinstance(interpretation, (uproot.interpretation.numerical.AsDtype,
                                           uproot.interpretation.jagged.AsJagged)):
            raise TypeError('Unsupported ROOT particle/scalar interpretation')
        value = interpretation.basket_array(
            basket.data, basket.byte_offsets, basket, branch, dict(branch.context),
            basket.member('fKeylen'), self.library, {'ak_add_doc': False})
        if len(value) != basket.num_entries:
            raise ValueError('ROOT basket entry coverage differs')
        size = buffer_bytes(value)
        self.reads += 1
        # Oversize baskets are processed once without retention, never clipped.
        if size <= self.cache_bytes:
            while self.cache and self.resident_bytes + size > self.cache_bytes:
                _, (_, retired) = self.cache.popitem(last=False)
                self.resident_bytes -= retired
            self.cache[key] = (value, size)
            self.resident_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.resident_bytes)
        return value

    def arrays(self, branches, *, entry_start, entry_stop, library='ak', how=dict):
        if (library not in ('ak','np') or how is not dict or type(entry_start) is not int
                or type(entry_stop) is not int or not 0 <= entry_start < entry_stop <= self.num_entries):
            raise ValueError('Explicit bounded numeric/dict row request required')
        allowed = {prefix+'jet_nparticles' for prefix in ('', 'hlt_')}
        allowed |= {prefix+'part_'+field for prefix in ('', 'hlt_') for field in FIELDS}
        if not branches or len(set(branches)) != len(branches) or not set(branches) <= allowed:
            raise PermissionError('Undeclared particle/scalar branch request')
        result = {}
        for name in branches:
            branch = self.tree[name]
            offsets = np.asarray(branch.entry_offsets)
            first = max(0, int(np.searchsorted(offsets, entry_start, side='right'))-1)
            stop = int(np.searchsorted(offsets, entry_stop, side='left'))
            baskets = {i: self.temporary(branch, i) for i in range(first, stop)}
            result[name] = branch.interpretation.final_array(
                baskets, entry_start, entry_stop, branch.entry_offsets,
                self.library if library=='ak' else self.numpy_library, branch, {'ak_add_doc': False})
            if len(result[name]) != entry_stop-entry_start:
                raise ValueError('Authorized ROOT span coverage differs')
        return result


def close_reader():
    global _OPEN
    if _OPEN is not None:
        _OPEN[1].close()
        _OPEN = None


@contextmanager
def reader(path, tree_key, *, retain=True):
    global _OPEN
    key = (str(Path(path)), tree_key)
    if not retain:
        value = Reader(path, tree_key)
        try:
            yield value
        finally:
            value.close()
        return
    if _OPEN is None or _OPEN[0] != key:
        close_reader()
        _OPEN = (key, Reader(path, tree_key))
    yield _OPEN[1]

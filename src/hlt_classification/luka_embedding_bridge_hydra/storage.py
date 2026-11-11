"""Bounded memory-mapped arrays with exclusive atomic publication and typed manifests."""
from pathlib import Path
import os
import numpy as np
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from .contracts import artifact, validate


def regular(root, name):
    root = Path(root).resolve(strict=True); path = root/name
    if (Path(name).is_absolute() or '..' in Path(name).parts or not path.resolve().is_relative_to(root)
            or any(p.is_symlink() for p in (path,*path.parents))):
        raise ValueError('Unsafe or redirected artifact path')
    return path


class Writer:
    """NPY arrays are private until flush/fsync/link and completion manifest."""
    def __init__(self, root, shapes):
        self.root = Path(root); self.root.mkdir(parents=True,exist_ok=False)
        self.arrays = {}
        for key,(dtype,shape) in shapes.items():
            if not key.isidentifier(): raise ValueError('Array names must be identifiers')
            self.arrays[key] = np.lib.format.open_memmap(self.root/(key+'.partial.npy'),mode='w+',dtype=dtype,shape=shape)

    def finish(self, kind, **fields):
        inventory = {}
        for key,a in self.arrays.items():
            a.flush(); path = self.root/(key+'.partial.npy')
            info = dict(dtype=a.dtype.str,shape=list(a.shape))
            a._mmap.close()
            with path.open('r+b') as handle: os.fsync(handle.fileno())
            final = self.root/(key+'.npy'); os.link(path,final); path.unlink()
            inventory[key] = dict(file=file_ref(final,root=self.root),**info)
        self.arrays.clear()
        value = artifact(kind,arrays=inventory,**fields)
        write_immutable_json(self.root/'manifest.json',value)
        return value


class Store:
    def __init__(self, root, kind, *, parents=None, digest=None):
        self.root = Path(root).resolve(strict=True)
        self.manifest = load_json(regular(self.root,'manifest.json'))
        validate(self.manifest,kind,parents)
        if digest is not None and self.manifest['content_hash'] != digest:
            raise ValueError('Store content identity differs')
        self.arrays = {}
        for key,info in self.manifest['arrays'].items():
            if not key.isidentifier() or info['file']['path'] != key+'.npy':
                raise ValueError('Store inventory differs')
            path=regular(self.root,info['file']['path']); validate_file_ref(info['file'],root=self.root)
            a=np.load(path,mmap_mode='r',allow_pickle=False)
            if a.dtype.str != info['dtype'] or list(a.shape) != info['shape'] or a.dtype.hasobject:
                raise ValueError('Store array declaration differs')
            self.arrays[key]=a

    def __getitem__(self,key): return self.arrays[key]


def validate_population(ids, labels):
    if (ids.dtype != np.uint8 or ids.shape != (len(labels),32) or labels.dtype != np.int64
            or labels.ndim != 1 or not np.isin(labels,range(11)).all()
            or len(np.unique(ids,axis=0)) != len(labels)):
        raise ValueError('Invalid authenticated population')


def join(left, right):
    for key in ('identities','labels'):
        a = getattr(left,key) if hasattr(left,key) else left[key]
        b = getattr(right,key) if hasattr(right,key) else right[key]
        if not np.array_equal(a,b): raise ValueError('Ordered identity/label join differs')

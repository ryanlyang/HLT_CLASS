"""One immutable smooth ragged bank and borrowed four-teacher jet targets."""
from pathlib import Path
import os
import shutil
import numpy as np
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    atomic_publish_bytes, deterministic_npz_bytes, write_immutable_json)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_attention.banks import counts, alignment
from hlt_classification.luka_fullstat_fine_tail import banks as old, kd_banks as old_kd
from .contracts import artifact, validate, target_recipe, recipe, teachers
from .targets import Moments


def size(rows, tokens):
    return int(tokens)*144 + (int(rows)+1)*8 + int(rows)*40 + 2*2**30


def support(manifest):
    chunks = [c['views']['D000'] for c in manifest['chunks'] if c['role'] == 'train']
    return sum(c['rows'] for c in chunks), sum(c['particles'] for c in chunks)


def build(root, chunks, *, prepared, source_hash, rows, tokens, available_bytes):
    root = Path(root)
    needed = size(rows, tokens)
    if available_bytes < needed or shutil.disk_usage(root.parent).free < needed:
        raise OSError(f'Need {needed/2**30:.2f} GiB personal target quota including reserve')
    root.mkdir(exist_ok=False)
    offsets = np.empty(rows+1, np.int64); offsets[0] = 0
    ids = np.empty((rows, 32), np.uint8); labels = np.empty(rows, np.int64)
    mapped = np.lib.format.open_memmap(root/'smooth.partial.npy', mode='w+', dtype=np.float32, shape=(tokens, 36))
    moments = Moments(); n = token = 0
    try:
        for block in chunks:
            ii, yy, oo, vv = (block[k] for k in ('identities', 'labels', 'offsets', 'values'))
            length, nt = len(yy), len(vv)
            if (not length or n+length > rows or token+nt > tokens
                    or ii.dtype != np.uint8 or ii.shape != (length, 32) or yy.dtype != np.int64
                    or oo.dtype != np.int64 or oo.shape != (length+1,) or oo[0] != 0 or oo[-1] != nt
                    or np.any(np.diff(oo) < 1) or np.any(np.diff(oo) > 512)
                    or vv.dtype != np.float32 or vv.shape != (nt, 36) or not np.isfinite(vv).all()
                    or np.any(yy < 0) or np.any(yy > 10)):
                raise ValueError('Invalid TRAIN target chunk')
            ids[n:n+length], labels[n:n+length] = ii, yy
            offsets[n+1:n+length+1] = token+oo[1:]
            mapped[token:token+nt] = vv
            for lo, hi in zip(oo[:-1], oo[1:]):
                moments.add(vv[lo:hi])
            n += length; token += nt
        if n != rows or token != tokens or len(np.unique(ids, axis=0)) != rows:
            raise ValueError('Missing/duplicate TRAIN targets')
        mapped.flush()
    finally:
        mapped._mmap.close()
    mean, scale = moments.result()
    os.rename(root/'smooth.partial.npy', root/'smooth.npy')
    atomic_publish_bytes(root/'index.npz', deterministic_npz_bytes(dict(
        identities=ids, labels=labels, offsets=offsets, mean=mean, scale=scale)))
    record = artifact('BANK', parents=dict(prepared=prepared, source=source_hash, recipe=target_recipe()['content_hash']),
        recipe=target_recipe(), role='train', coordinate='D000', rows=rows, tokens=tokens, width=36, dtype='float32',
        outputs=[file_ref(root/f, root=root) for f in ('index.npz', 'smooth.npy')])
    write_immutable_json(root/'manifest.json', record)
    return record


def validate_bank(root, *, prepared, source_hash, rows, tokens, payload=True):
    root = Path(root); m = load_json(root/'manifest.json')
    validate(m, 'BANK', dict(prepared=prepared, source=source_hash, recipe=target_recipe()['content_hash']))
    expected = dict(recipe=target_recipe(), role='train', coordinate='D000', rows=rows, tokens=tokens, width=36, dtype='float32')
    if (any(m.get(k) != v for k, v in expected.items())
            or [r['path'] for r in m['outputs']] != ['index.npz', 'smooth.npy']):
        raise ValueError('Neighborhood semantics or file inventory differs')
    for ref in m['outputs']:
        if payload or ref['path'] == 'index.npz':
            validate_file_ref(ref, root=root)
    a = load_npz_arrays(root/'index.npz')
    if (set(a) != {'identities', 'labels', 'offsets', 'mean', 'scale'}
            or a['identities'].dtype != np.uint8 or a['identities'].shape != (rows, 32)
            or a['labels'].dtype != np.int64 or a['labels'].shape != (rows,)
            or a['offsets'].dtype != np.int64 or a['offsets'].shape != (rows+1,)
            or a['offsets'][0] != 0 or a['offsets'][-1] != tokens
            or np.any(np.diff(a['offsets']) < 1) or np.any(np.diff(a['offsets']) > 512)
            or any(a[k].dtype != np.float64 or a[k].shape != (36,) or not np.isfinite(a[k]).all() for k in ('mean', 'scale'))
            or np.any(a['scale'] < .1)):
        raise ValueError('Invalid ragged index or TRAIN normalization')
    return m, a


class NeighborhoodBank:
    def __init__(self, spec, cache):
        if cache.role != 'train' or cache.coordinate_name != 'D000':
            raise PermissionError('Only native TRAIN HLT may join neighborhood targets')
        m, a = validate_bank(spec['target_root'], prepared=cache.foundation_sha256,
            source_hash=spec['content_hash'], rows=len(cache), tokens=int(counts(cache).sum()))
        if (not np.array_equal(a['identities'], cache.identities) or not np.array_equal(a['labels'], cache.labels)
                or not np.array_equal(np.diff(a['offsets']), counts(cache))):
            raise ValueError('Native TRAIN identity/label/support join differs')
        self.manifest, self.index = m, a
        # One RAM copy: no random filesystem reads during GPU training.
        self.payload = np.load(Path(spec['target_root'])/'smooth.npy', allow_pickle=False)
        if self.payload.dtype != np.float32 or self.payload.shape != (m['tokens'], 36):
            raise ValueError('Target payload shape/dtype differs')
        for start in range(0, len(self.payload), 65536):
            if not np.isfinite(self.payload[start:start+65536]).all():
                raise ValueError('Nonfinite target payload')
        self.payload.flags.writeable = False

    def batch(self, indices, width):
        ix = np.asarray(indices); offsets = self.index['offsets']
        if (ix.dtype != np.int64 or ix.ndim != 1 or not 1 <= width <= 512
                or np.any(ix < 0) or np.any(ix >= len(offsets)-1)):
            raise ValueError('Invalid target batch request')
        lengths = offsets[ix+1]-offsets[ix]
        if np.any(lengths > width):
            raise ValueError('Student support truncated')
        columns = np.arange(width)[None, :]
        real = columns < lengths[:, None]
        positions = np.minimum(offsets[ix, None]+columns, offsets[ix+1, None]-1)
        out = ((self.payload[positions].astype(np.float64)-self.index['mean'])/self.index['scale']).astype(np.float32)
        out[~real] = 0
        if not np.isfinite(out).all():
            raise ValueError('Nonfinite standardized neighborhoods')
        return out


def bundle_record(spec, stage):
    order = teachers(stage)
    record = load_json(Path(spec['target_root'])/'manifest.json')
    return artifact('BUNDLE', parents=dict(spec=spec['content_hash'], recipe=recipe(stage)['content_hash'],
        neighborhood=record['content_hash']), stage=stage, role='train', prepared=spec['anchor']['evidence']['prepared'],
        rows=spec['counts']['train'], teacher_order=order, temperature=2.,
        neighborhood=file_ref(Path(spec['target_root'])/'manifest.json'),
        records={t:spec['imported_banks'][t] for t in order})


class Bundle:
    def __init__(self, spec, stage, train):
        self.manifest = bundle_record(spec, stage)
        self.identities, self.labels = train.identities, train.labels
        self.targets, self.probabilities = {}, {}
        for t in teachers(stage):
            refs = spec['external_refs'][t]
            _, a = old.arrays(spec['imported_banks'][t]['embedding'], train, refs=refs, teacher_id=t)
            self.targets[t] = a['targets']
            _, a = old_kd.arrays(spec['imported_banks'][t]['kd'], train, refs=refs, teacher_id=t)
            self.probabilities[t] = a['probabilities']
        self.neighborhood = NeighborhoodBank(spec, train)
        check_join(self, train, stage)


def check_join(bank, train, stage):
    validate(bank.manifest, 'BUNDLE')
    if (train.role != 'train' or train.coordinate_name != 'D000'
            or bank.manifest['prepared'] != train.foundation_sha256 or bank.manifest['stage'] != stage
            or bank.manifest['teacher_order'] != teachers(stage) or bank.manifest['rows'] != len(train)
            or set(bank.targets) != set(teachers(stage)) or set(bank.probabilities) != set(teachers(stage))
            or not np.array_equal(bank.identities, train.identities) or not np.array_equal(bank.labels, train.labels)):
        raise ValueError('Four-teacher TRAIN join differs')
    for t in teachers(stage):
        old_kd.validate_probabilities(bank.probabilities[t], len(train))
        z = bank.targets[t]
        if z.dtype != np.float32 or z.shape != (len(train), 128) or not np.isfinite(z).all():
            raise ValueError('Invalid teacher embedding')


def publish(spec, stage, train):
    bank = Bundle(spec, stage, train)
    root = Path(spec['root'])/'bundles'; root.mkdir(exist_ok=True)
    write_immutable_json(root/(stage+'.json'), bank.manifest)
    return bank

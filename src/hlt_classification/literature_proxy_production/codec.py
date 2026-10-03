"""Lossless physical bank codec; no fitted response or generation imports."""
import hashlib
import io
import zipfile
import numpy as np
from hlt_classification.cms2jc2_response.bridge import Particles

FIELDS = ('p4', 'charge', 'category', 'tracking', 'valid')
DTYPES = ('<f8', 'i1', 'i1', '<f8', '?')

def physical_digest(identity, particle):
    digest = hashlib.sha256(bytes.fromhex(identity))
    for name, dtype in zip(FIELDS, DTYPES):
        array = np.asarray(getattr(particle, name), dtype=dtype)
        digest.update(np.asarray(array.shape, dtype='<i8').tobytes())
        digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def pack(rows):
    offsets = np.concatenate(([0], np.cumsum([len(p) for _, p, _ in rows], dtype=np.int64)))
    arrays = {name: np.concatenate([np.asarray(getattr(p, name), dtype=dtype) for _, p, _ in rows])
              for name, dtype in zip(FIELDS, DTYPES)}
    return dict(offsets=np.asarray(offsets, dtype='<i8'),
                jet_identity=np.asarray([list(bytes.fromhex(i)) for i, _, _ in rows], dtype='u1'), **arrays)


def encode(arrays, compression):
    if compression not in ('stored', 'deflate'):
        raise ValueError('Only lossless encodings are registered')
    stream = io.BytesIO()
    method = zipfile.ZIP_STORED if compression == 'stored' else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(stream, 'w', compression=method, compresslevel=6 if method else None) as archive:
        for name, array in sorted(arrays.items()):
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(name+'.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = method
            archive.writestr(info, buffer.getvalue(), compresslevel=6 if method else None)
    return stream.getvalue()


def readback(path, expected):
    decoded = {}
    with np.load(path, allow_pickle=False) as handle:
        if set(handle.files) != set(expected):
            raise ValueError('Output fields changed')
        for name, array in expected.items():
            actual = handle[name]
            if actual.dtype != array.dtype or actual.shape != array.shape or actual.tobytes() != array.tobytes():
                raise ValueError('Output readback changed: '+name)
            decoded[name] = actual
    # Validate the cache through the same physical schema, not just its bytes.
    offsets = decoded['offsets']
    if offsets[0] != 0 or np.any(np.diff(offsets) < 0) or offsets[-1] != len(decoded['p4']):
        raise ValueError('Invalid output offsets')
    digests = []
    for i, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        p = Particles(*(decoded[k][lo:hi] for k in FIELDS), tuple(str(j) for j in range(hi-lo)))
        digests.append(physical_digest(bytes(decoded['jet_identity'][i]).hex(), p))
    return digests




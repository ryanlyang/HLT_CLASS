"""Differences have meaning only under one shared encoder and one normalization."""
import numpy as np
from .registry import CONDITIONS, QUERY_NAMES


def moments(chunks, width=128):
    count = 0; mean = np.zeros(width,np.float64); m2 = mean.copy()
    for chunk in chunks:
        a = np.asarray(chunk,np.float64).reshape(-1,width)
        if not np.isfinite(a).all(): raise ValueError('Nonfinite normalization input')
        if not len(a): continue
        cm = a.mean(0); delta = cm-mean; n = len(a)
        m2 += ((a-cm)**2).sum(0)+delta**2*(count*n/(count+n))
        mean += delta*(n/(count+n)); count += n
    return mean, np.maximum(np.sqrt(m2/max(count,1)),.001), count


def normalize(values, mean, scale):
    if np.shape(mean) != (128,) or np.shape(scale) != (128,) or np.any(np.asarray(scale) < .001):
        raise ValueError('Invalid common TRAIN moments')
    result = ((np.asarray(values,np.float64)-mean)/scale).astype(np.float32)
    if not np.isfinite(result).all(): raise ValueError('Nonfinite normalized target')
    return result


def differences(values):
    """Values: batch x six conditions x 128, common standardized axes."""
    a = np.asarray(values)
    if a.ndim != 3 or a.shape[1:] != (6,128) or not np.isfinite(a).all():
        raise ValueError('Shared six-condition bank required')
    out = {k:a[:,CONDITIONS.index(k)]-a[:,0] for k in ('K','I','T','IT')}
    out['J'] = out['IT']-out['I']-out['T']
    return out


def scales(chunks):
    total = {k:0. for k in ('K','I','T','IT')}; count = 0
    for a in chunks:
        d = differences(np.asarray(a,np.float64)); count += len(a)*128
        for k in total: total[k] += float(np.square(d[k]).sum())
    return {k:max(.1,float(np.sqrt(v/max(count,1)))) for k,v in total.items()}


def targets(values, scale, queries, *, absolute=False):
    if set(scale) != {'K','I','T','IT'} or any(not np.isfinite(v) or v < .1 for v in scale.values()):
        raise ValueError('Invalid response TRAIN scales')
    d = differences(values)
    if not queries or any(q not in QUERY_NAMES for q in queries) or (absolute and 'J' in queries):
        raise ValueError('Unknown response query')
    return np.stack([(values[:,CONDITIONS.index(q)] if absolute else d[q])/scale['IT' if q == 'J' else q]
                     for q in queries],axis=1).astype(np.float32)

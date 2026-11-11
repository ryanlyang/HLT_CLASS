"""HLT-defined nested partitions. Crop encoding never sees its complement."""
import math
import numpy as np
from hlt_classification.cms2jc2_response.bridge import Particles
from hlt_classification.literature_context.transform import build_inputs
from .registry import SCALES, SLICES


def catalog():
    rows = []
    for level, count in enumerate((1, 2, 4)):
        for band in range(2):
            for angle in range(count):
                center = -np.pi+(angle+.5)*2*np.pi/count
                # Full circle is represented canonically, not with sin(pi) roundoff.
                angular = [0., 1., 1.] if count == 1 else [math.sin(center), math.cos(center), 1./count]
                rows.append(dict(index=len(rows), scale=SCALES[level], band=band, angular_bin=angle,
                    bins=count, descriptor=[0., *[float(i == level) for i in range(3)],
                        float(band == 0), float(band == 1), *angular]))
    return rows


def descriptors(scales):
    if tuple(scales) == ('whole',):
        return np.array([[1.,0.,0.,0.,0.,0.,0.,1.,1.]], np.float32), [14]
    if not scales or any(s not in SCALES for s in scales) or len(set(scales)) != len(scales):
        raise ValueError('Invalid regional queries')
    selected = [r for r in catalog() if r['scale'] in scales]
    return np.array([r['descriptor'] for r in selected], np.float32), [r['index'] for r in selected]


def physical(values):
    a = np.asarray(values)
    if (a.dtype != np.float64 or a.ndim != 2 or a.shape[1] != 14 or not np.isfinite(a).all()
            or not np.isin(a[:,4], [-1,0,1]).all() or not np.isin(a[:,5], range(6)).all()
            or not np.isin(a[:,10:], [0,1]).all() or (a[:,8:10] < 0).any() or (a[:,3] < 0).any()):
        raise ValueError('Invalid physical columns: p4,charge,category,tracking_mm,validity')
    return Particles(a[:,:4], a[:,4].astype(np.int8), a[:,5].astype(np.int8),
                     a[:,6:10], a[:,10:].astype(bool), tuple(str(i) for i in range(len(a))))


def frame(hlt):
    p = physical(hlt).p4
    # Sorting stabilizes FP64 sums under permutations, without exposing indices.
    p = p[np.lexsort(tuple(p[:,j] for j in (3,2,1,0)))]
    summed = p.sum(0); pt = np.hypot(p[:,0], p[:,1]); total = float(pt.sum())
    axis_pt = float(np.hypot(summed[0], summed[1]))
    if not len(p) or total <= 0 or axis_pt <= 1e-12:
        return 0., 0., .05
    eta = float(np.arcsinh(summed[2]/axis_pt)); phi = float(np.arctan2(summed[1], summed[0]))
    radii, _ = polar(p, (eta, phi, 1.))
    order = np.argsort(radii, kind='stable')
    pos = min(np.searchsorted(np.cumsum(pt[order]), .9*total, side='left'), len(p)-1)
    return eta, phi, max(.05, float(radii[order[pos]]))


def polar(p4, axis):
    eta, phi, _ = axis
    de = np.arcsinh(p4[:,2]/np.maximum(np.hypot(p4[:,0], p4[:,1]), 1e-12))-eta
    dp = (np.arctan2(p4[:,1],p4[:,0])-phi+np.pi)%(2*np.pi)-np.pi
    return np.hypot(de,dp), (np.arctan2(dp,de)+np.pi)%(2*np.pi)-np.pi


def partitions(hlt, source):
    p = physical(source); axis = frame(hlt); r, theta = polar(p.p4, axis)
    band = (r >= axis[2]/2).astype(np.int64)
    result = []
    for count in (1,2,4):
        angular = np.minimum(((theta+np.pi)*count/(2*np.pi)).astype(np.int64), count-1)
        result.append(band*count+angular)
    return dict(zip(SCALES, result))


def encode(values):
    p = physical(values)
    return build_inputs(p, capacity=max(512,len(p)))


def padded(inputs, labels, identities=None):
    n = len(inputs); width = max(16, max((len(x.features) for x in inputs), default=0))
    result = dict(features=np.zeros((n,17,width),np.float32), vectors=np.zeros((n,4,width),np.float32),
                  mask=np.zeros((n,1,width),bool), labels=np.asarray(labels,np.int64))
    for i, x in enumerate(inputs):
        length = len(x.features)
        result['features'][i,:,:length] = x.features.T
        result['vectors'][i,:,:length] = x.vectors.T
        result['mask'][i,:,:length] = True
        if length == 0:
            # A dummy visible zero-feature token avoids all-masked attention NaNs.
            # Its output is excluded by the separate target/aggregation validity mask.
            result['mask'][i,0,0] = True
            result['vectors'][i,3,0] = 1e-8
    if identities is not None:
        result['identities'] = identities
    return result


def crops(hlt_rows, source_rows, labels, scale):
    if scale not in SCALES or len(hlt_rows) != len(source_rows):
        raise ValueError('Invalid crop sources')
    count = SLICES[scale][1]-SLICES[scale][0]
    assignments = [partitions(h,s)[scale] for h,s in zip(hlt_rows,source_rows)]
    valid = np.zeros((len(source_rows),count),bool); batches = []
    for j in range(count):
        parts = [s[a == j] for s,a in zip(source_rows,assignments)]
        valid[:,j] = [len(p) > 0 for p in parts]
        batches.append(padded([encode(p) for p in parts], labels))
    return batches, valid

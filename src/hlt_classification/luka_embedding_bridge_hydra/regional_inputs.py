"""Batch the crop frontend while retaining each crop's own frame and row order.

The scalar frontend in regions.crops remains the parity authority.
Validation is applied to the complete physical sources before taking subsets;
no particle matching, complement features or labels enter the crop encoder.
"""
import numpy as np
from hlt_classification.cms_proxy_ladder.inputs import RaggedInputs
from hlt_classification.literature_context.transform import VALUE_SCALES, ERROR_SCALES
from . import regions
from .registry import SCALES, SLICES


def encode_many(parts):
    """Input parts have already passed the full Particles physical validation."""
    lengths = np.array([len(a) for a in parts], np.int64)
    offsets = np.r_[0, np.cumsum(lengths)]
    a = np.concatenate(parts)
    # Keep precisely the original per-crop reduction order. A segmented sum,
    # padded reduction or float32 sum could change historical input bits.
    summed = np.array([p[:, :4].copy().sum(axis=0) for p in parts])
    group = np.repeat(np.arange(len(parts)), lengths)
    jet_pt = np.maximum(np.hypot(summed[:, 0], summed[:, 1]), 1e-8)
    jet_energy = np.maximum(summed[:, 3], 1e-8)
    jet_eta = np.arcsinh(summed[:, 2] / jet_pt)
    jet_phi = np.arctan2(summed[:, 1], summed[:, 0])
    pt = np.hypot(a[:, 0], a[:, 1])
    eta = np.arcsinh(a[:, 2] / np.maximum(pt, 1e-8))
    phi = np.arctan2(a[:, 1], a[:, 0])
    deta = (eta - jet_eta[group]) * np.where(jet_eta[group] < 0, -1, 1)
    dphi = (phi - jet_phi[group] + np.pi) % (2*np.pi) - np.pi
    pt = np.maximum(pt, 1e-8)
    energy = np.maximum(a[:, 3], 1e-8)
    one_hot = np.zeros((len(a), 5), np.float64)
    category = a[:, 5].astype(np.int64)
    known = category < 5
    one_hot[np.flatnonzero(known), category[known]] = 1.
    values = np.column_stack((
        np.clip((np.log(pt)-1.7)*.7, -5, 5),
        np.clip((np.log(energy)-2.)*.7, -5, 5),
        np.clip((np.log(pt/jet_pt[group])+4.7)*.7, -5, 5),
        np.clip((np.log(energy/jet_energy[group])+4.7)*.7, -5, 5),
        np.clip((np.hypot(deta, dphi)-.2)*4., -5, 5),
        a[:, 4], one_hot, np.zeros((len(a), 4)), deta, dphi,
    )).astype(np.float32)
    values[:, [11, 13]] = (np.arcsinh(a[:, 6:8]/VALUE_SCALES)/4).astype(np.float32)
    values[:, [12, 14]] = (np.log1p(a[:, 8:10]/ERROR_SCALES)/2).astype(np.float32)
    if values.shape != (len(a), 17) or not np.isfinite(values).all():
        raise ValueError('Nonfinite batched crop inputs')
    vectors = a[:, :4].astype(np.float32)
    return [RaggedInputs(values[lo:hi], vectors[lo:hi]) for lo, hi in zip(offsets[:-1], offsets[1:])]


def crops(hlt_rows, source_rows, labels, scale):
    if (scale not in SCALES or not len(hlt_rows) or len(hlt_rows) != len(source_rows)
            or len(labels) != len(hlt_rows)):
        raise ValueError('Invalid crop sources')
    # Validate once per global batch rather than creating/validating thousands
    # of immutable Particles objects for already validated disjoint subsets.
    # Include each original array's dtype/shape check before concatenation.
    for rows in (hlt_rows, source_rows):
        if any(a.dtype != np.float64 or a.ndim != 2 or a.shape[1] != 14 for a in rows):
            raise ValueError('Invalid physical source shape/dtype')
        regions.physical(np.concatenate(rows))
    count = SLICES[scale][1] - SLICES[scale][0]
    assignments = []
    for h, s in zip(hlt_rows, source_rows):
        p = h[:, :4]
        p = p[np.lexsort(tuple(p[:, j] for j in (3, 2, 1, 0)))]
        summed = p.sum(0)
        pt = np.hypot(p[:, 0], p[:, 1]); total = float(pt.sum())
        axis_pt = float(np.hypot(summed[0], summed[1]))
        if not len(p) or total <= 0 or axis_pt <= 1e-12:
            axis = (0., 0., .05)
        else:
            eta = float(np.arcsinh(summed[2]/axis_pt))
            phi = float(np.arctan2(summed[1], summed[0]))
            radii, _ = regions.polar(p, (eta, phi, 1.))
            order = np.argsort(radii, kind='stable')
            pos = min(np.searchsorted(np.cumsum(pt[order]), .9*total, side='left'), len(p)-1)
            axis = (eta, phi, max(.05, float(radii[order[pos]])))
        radius, theta = regions.polar(s[:, :4], axis)
        bins = count // 2
        angle = np.minimum(((theta+np.pi)*bins/(2*np.pi)).astype(np.int64), bins-1)
        assignments.append((radius >= axis[2]/2).astype(np.int64)*bins + angle)
    parts = [s[a == j] for j in range(count) for s, a in zip(source_rows, assignments)]
    encoded = encode_many(parts)
    n = len(source_rows)
    valid = np.array([len(p) > 0 for p in parts], bool).reshape(count, n).T.copy()
    return [regions.padded(encoded[j*n:(j+1)*n], labels) for j in range(count)], valid


def qualify(spec, ds):
    """Real inputs: every class, long-support tails and random rows, both roles."""
    from pathlib import Path
    from .contracts import artifact, crop_execution_policy
    from .resources import selected_indices
    from hlt_classification.data.cache_contracts import write_immutable_json
    selected = selected_indices(ds, wide=True); checks = []
    for role, ix in selected.items():
        physical = ds.physical(role)
        for source in ('HLT', 'OFFLINE'):
            for scale in SCALES:
                for start in range(0, len(ix), 128):
                    indices = ix[start:start+128]
                    def rows(name):
                        offsets = physical[name+'_offsets']
                        return [physical[name+'_values'][offsets[i]:offsets[i+1]] for i in indices]
                    labels = physical['labels'][indices]
                    args = (rows('HLT'), rows(source), labels, scale)
                    before, valid = regions.crops(*args)
                    after, actual = crops(*args)
                    if valid.tobytes()!=actual.tobytes() or valid.shape!=actual.shape:
                        raise ValueError('Batched crop validity differs')
                    for a,b in zip(before,after):
                        if set(a)!=set(b) or any(a[k].shape!=b[k].shape or a[k].dtype!=b[k].dtype
                                or a[k].tobytes()!=b[k].tobytes() for k in a):
                            raise ValueError(f'Batched crop bytes differ: {role}/{source}/{scale}')
                checks.append(dict(role=role,source=source,scale=scale,rows=len(ix),byte_exact=True))
                print(f'EMBEDDING_BRIDGE crop frontend exact: {role}/{source}/{scale}; {len(ix)} rows',flush=True)
    value=artifact('CROP_FRONTEND',parents=dict(spec=spec['content_hash']),policy=crop_execution_policy(),
        checks=checks,indices={k:v.tolist() for k,v in selected.items()})
    write_immutable_json(Path(spec['root'])/'setup/crop_frontend.json',value)
    check(value,spec)
    return value


def check(value,spec):
    from .contracts import validate,crop_execution_policy
    validate(value,'CROP_FRONTEND',dict(spec=spec['content_hash']))
    expected=[(role,source,scale) for role in ('train','validation') for source in ('HLT','OFFLINE') for scale in SCALES]
    if (spec['crop_execution_policy']!=crop_execution_policy() or value['policy']!=crop_execution_policy()
            or [(c['role'],c['source'],c['scale']) for c in value['checks']]!=expected
            or set(value['indices'])!={'train','validation'}):raise ValueError('Incomplete crop frontend parity')
    for c in value['checks']:
        ix=value['indices'][c['role']]
        if (c['byte_exact']is not True or c['rows']!=len(ix) or not ix
                or sorted(set(ix))!=ix or min(ix)<0 or max(ix)>=spec['counts'][c['role']]):
            raise ValueError('Invalid crop frontend parity rows')
    return value

"""TRAIN-only physical-summary and masked native-particle normalization."""
from pathlib import Path
import numpy as np
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from .contracts import artifact, validate


def summaries(raw):
    f = np.asarray(raw['features'], np.float64).transpose(0, 2, 1)
    v = np.asarray(raw['vectors'], np.float64).transpose(0, 2, 1)
    mask = raw['mask'][:, 0].astype(bool); rows = []
    for ff, vv, valid in zip(f, v, mask):
        ff, vv = ff[valid], vv[valid]
        if not len(ff):
            raise ValueError('Empty physical jet')
        total = vv.sum(0); pt = np.hypot(vv[:, 0], vv[:, 1]); w = pt/max(pt.sum(), 1e-8)
        jetpt = np.hypot(total[0], total[1]); mass = np.sqrt(max(total[3]**2-(total[:3]**2).sum(), 0))
        r = np.hypot(ff[:, 15], ff[:, 16]); q = ff[:, 5]; cat = ff[:, 6:11]
        lead = np.zeros(4); ranked = np.sort(w)[::-1]; lead[:min(4, len(w))] = ranked[:4]
        rows.append(np.r_[np.log1p([len(ff), jetpt, total[3], mass]),
            abs(np.arcsinh(total[2]/max(jetpt, 1e-8))), np.sqrt((w*w).sum()),
            [(w*r**p).sum() for p in (.5, 1, 2, 3)], cat.mean(0), (w[:, None]*cat).sum(0),
            np.abs(q).mean(), (w*q).sum(), (w*q*r).sum(), (w*q*r*r).sum(),
            np.abs(ff[:, 11:15]).mean(0), (w[:, None]*np.abs(ff[:, 11:15])).sum(0), lead])
    result = np.asarray(rows, np.float32)
    if result.shape != (len(f), 36) or not np.isfinite(result).all():
        raise ValueError('Invalid measured summaries')
    return result


def build(spec, train):
    if train.role != 'train' or train.coordinate_name != 'OFFLINE' or train.foundation_sha256 != spec['anchor']['evidence']['prepared']:
        raise PermissionError('Normalization requires authenticated OFFLINE TRAIN')
    totals = {k: [0, np.zeros(n), np.zeros(n)] for k, n in (('particle', 21), ('summary', 36))}
    maximum = 0
    for start in range(0, len(train), 128):
        raw = train.batch(np.arange(start, min(start+128, len(train))))
        mask = raw['mask'][:, 0].astype(bool)
        maximum = max(maximum, int(mask.sum(1).max()))
        p = np.concatenate((raw['features'], raw['vectors']), axis=1).transpose(0, 2, 1)[mask]
        for kind, values in (('particle', p), ('summary', summaries(raw))):
            a = values.astype(np.float64); t = totals[kind]
            t[0] += len(a); t[1] += a.sum(0); t[2] += (a*a).sum(0)
    moments = {}
    for k, (n, sums, squares) in totals.items():
        mean = sums/n; scale = np.maximum(np.sqrt(np.maximum(squares/n-mean*mean, 0)), .001)
        moments[k] = dict(mean=mean.tolist(), scale=scale.tolist(), rows=n)
    value = artifact('STATISTICS', parents=dict(spec=spec['content_hash'], transfer=spec['offline_hash']),
        role='train', coordinate='OFFLINE', maximum=maximum, rows=len(train), moments=moments)
    write_immutable_json(Path(spec['root'])/'setup/statistics.json', value)
    return load(spec)


def load(spec):
    value = load_json(Path(spec['root'])/'setup/statistics.json')
    validate(value, 'STATISTICS', dict(spec=spec['content_hash'], transfer=spec['offline_hash']))
    if (value['role'] != 'train' or value['coordinate'] != 'OFFLINE' or value['rows'] != spec['counts']['train']
            or type(value['maximum']) is not int or not 1 <= value['maximum'] <= 512
            or set(value['moments']) != {'particle', 'summary'}):
        raise ValueError('TRAIN statistics meaning differs')
    for k, width in (('particle', 21), ('summary', 36)):
        m = value['moments'][k]
        if (np.shape(m['mean']) != (width,) or np.shape(m['scale']) != (width,)
                or not np.isfinite([m['mean'], m['scale']]).all() or min(m['scale']) < .001
                or m['rows'] < value['rows']):
            raise ValueError('Invalid TRAIN statistics')
    return value

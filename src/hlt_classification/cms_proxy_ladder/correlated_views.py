"""Verified one-to-one tracking-only views, with no runtime randomness."""
from fractions import Fraction
import numpy as np
from hlt_classification.cms2jc2_response.bridge import Particles
from hlt_classification.correlated_tracking_production.recipe import build_inputs, input_contract as encoding
from .inputs import input_contract as legacy_contract
from .contracts import artifact


def input_contract(*, capacity=512):
    old = legacy_contract(capacity=capacity)
    fields = {k: v for k, v in old.items() if k not in (
        'contract', 'schema_version', 'content_hash', 'parents', 'final_test_accessed')}
    fields.update(normalization='correlated_asinh_log1p_17_v1',
        error_transform='log1p_error_over_scale_no_clipping', encoding=encoding())
    return artifact('INPUTS', version=3, **fields)


def matcher_contract():
    return artifact('IDENTITY_MATCHER', physical_equality=['p4', 'charge', 'category', 'valid'],
        pairing='verified_original_particle_order', construction_features=False)


def view_contract():
    return artifact('VIEWS', version=4, matcher=matcher_contract(),
        support='same_original_particle_rows', D='linear_added_residual_quadrature_added_variance',
        D_coordinate='Dxxx_retained_offline_fraction', ascending_views_are_duplicates=True,
        endpoints={'OFFLINE': 'exact_original_offline', 'D000': 'exact_frozen_CORR_MID'},
        source_indices_model_visible=False, label_dependent=False)


def match_particles(proxy, offline):
    if len(proxy) != len(offline) or any(not np.array_equal(getattr(proxy, n), getattr(offline, n))
            for n in ('p4', 'charge', 'category', 'valid')):
        raise ValueError('CORR_MID physical row correspondence differs')
    if np.any(proxy.tracking[:, 2:] < offline.tracking[:, 2:]):
        raise ValueError('CORR_MID uncertainty smaller than original error')
    eligible = offline.valid[:, :2] & offline.valid[:, 2:]
    mask = np.column_stack((eligible, eligible))
    if not np.array_equal(proxy.tracking[~mask], offline.tracking[~mask]):
        raise ValueError('CORR_MID ineligible tracking changed')
    return np.arange(len(proxy), dtype=np.int32)


def build_view(*, identity, proxy, offline, coordinate, mapping):
    expected = match_particles(proxy, offline)
    if not np.array_equal(mapping, expected):
        raise ValueError('CORR_MID identity mapping differs')
    if coordinate in ('OFFLINE', 'U000', 'U100'):
        return offline
    if coordinate == 'D000':
        return proxy
    fraction = {'D066': Fraction(1, 3), 'D033': Fraction(2, 3), 'D050': Fraction(1, 2)}.get(coordinate)
    if fraction is None:
        raise ValueError('Unregistered correlated tracking coordinate')
    f = float(fraction)
    tr = offline.tracking.copy()
    eligible = offline.valid[:, :2] & offline.valid[:, 2:]
    tr[:, :2] = np.where(eligible,
        tr[:, :2]+f*(proxy.tracking[:, :2]-tr[:, :2]), tr[:, :2])
    tr[:, 2:] = np.where(eligible, np.sqrt(tr[:, 2:]**2 + f*f*(
        proxy.tracking[:, 2:]**2-tr[:, 2:]**2)), tr[:, 2:])
    return Particles(offline.p4, offline.charge, offline.category, tr, offline.valid, offline.keys)

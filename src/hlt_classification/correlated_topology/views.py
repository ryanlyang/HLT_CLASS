"""Fixed proxy topology; oracle tracking interpolation never restores tokens."""
import numpy as np
from hlt_classification.cms2jc2_response.bridge import Particles
from hlt_classification.cms_proxy_ladder.contracts import artifact
from hlt_classification.cms_proxy_ladder.correlated_views import input_contract, build_inputs
from .kernel import generate, equal


def matcher_contract():
    return artifact('CONSTRUCTION_MATCHER', recipe='CORR_HIGH_TOPO',
        pairing='verified_full_replay_then_proxy_to_offline_representative', construction_features=False)


def view_contract():
    return artifact('VIEWS', version=7, matcher=matcher_contract(),
        support='fixed_degraded_topology_all_D_original_OFFLINE',
        D='linear_HIGH_residual_quadrature_added_variance',
        endpoints=dict(OFFLINE='exact_original_offline', D000='exact_frozen_CORR_HIGH_TOPO'),
        source_indices_model_visible=False, label_dependent=False)


def match_particles(proxy, offline, *, identity, frozen):
    expected = generate(offline, identity, frozen)
    if not equal(proxy, expected.particles):
        raise ValueError('CORR_HIGH_TOPO endpoint replay differs')
    return expected.mapping


def build_view(*, identity, proxy, offline, coordinate, mapping):
    mapping = np.asarray(mapping)
    if (mapping.dtype != np.int32 or mapping.shape != (len(proxy),)
            or np.any(mapping < 0) or np.any(mapping >= len(offline))
            or len(set(map(int, mapping))) != len(mapping)):
        raise ValueError('Invalid topology representative mapping')
    if coordinate in ('OFFLINE', 'U000'):
        return offline
    if coordinate == 'D000':
        return proxy
    fraction = {'U100': 0., 'D066': 1/3, 'D033': 2/3, 'D050': .5}.get(coordinate)
    if fraction is None:
        raise ValueError('Unregistered topology coordinate')
    original = offline.tracking[mapping]
    eligible = proxy.valid[:, :2] & proxy.valid[:, 2:]
    variance = proxy.tracking[:, 2:]**2 - original[:, 2:]**2
    if np.any(variance[eligible] < -1e-12):
        raise ValueError('Added tracking variance is negative')
    tr = proxy.tracking.copy()
    tr[:, :2] = np.where(eligible, original[:, :2] + fraction*(tr[:, :2]-original[:, :2]), tr[:, :2])
    tr[:, 2:] = np.where(eligible, np.sqrt(original[:, 2:]**2 + fraction**2*np.maximum(variance, 0)), tr[:, 2:])
    return Particles(proxy.p4, proxy.charge, proxy.category, tr, proxy.valid, proxy.keys)

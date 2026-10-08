"""Fixed S3 support/PID/masks, oracle interpolation of surviving tracking only."""
import numpy as np
from hlt_classification.gap_sweep.kernel import generate, recipe
from hlt_classification.correlated_topology.views import build_view as interpolate, build_inputs
from .contracts import artifact

COORDINATES = ('D066', 'D033', 'D000')


def contract():
    return artifact('VIEWS', parents=dict(endpoint=recipe()['content_hash']), candidate='S3',
        coordinates=list(COORDINATES), support='fixed_S3_support_p4_PID_charge_validity',
        context='full_frozen_CORR_HIGH_TOPO_before_interpolation',
        tracking='offline_plus_f_residual_errors_quadrature_f_squared',
        fractions=dict(D066=[1, 3], D033=[2, 3], D000=[1, 1]),
        restored_tokens=False, restored_measurements=False, source_indices_model_visible=False)


def build_view(*, proxy, identity, coordinate, offline=None, mapping=None):
    if coordinate not in COORDINATES:
        raise ValueError('Unknown S3 coordinate')
    endpoint, _ = generate(proxy, identity, 'S3')
    if coordinate == 'D000':
        return endpoint  # Endpoint has no offline/mapping dependency.
    if offline is None or mapping is None:
        raise ValueError('Intermediate oracle views need authenticated original mapping')
    mapping = np.asarray(mapping)
    if (mapping.dtype != np.int32 or mapping.shape != (len(proxy),)
            or np.any(mapping < 0) or np.any(mapping >= len(offline))
            or len(set(map(int, mapping))) != len(mapping)):
        raise ValueError('Invalid original representative mapping')
    positions = {key: i for i, key in enumerate(proxy.keys)}
    selected = np.asarray([positions[key] for key in endpoint.keys], np.int64)
    return interpolate(identity=identity, proxy=endpoint, offline=offline,
                       coordinate=coordinate, mapping=mapping[selected])

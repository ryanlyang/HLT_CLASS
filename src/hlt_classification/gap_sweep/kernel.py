"""Frozen irreversible losses on saved CORR_HIGH_TOPO; label-blind and nested."""
import hashlib
import numpy as np
from hlt_classification.cms2jc2_response.bridge import Particles, wrap_phi
from hlt_classification.data.cache_contracts import canonical_json_bytes
from .contracts import artifact

CANDIDATES = {'S1': .5, 'S2': 1., 'S3': 1.5}


def recipe():
    return artifact('RECIPE', candidates=dict(CANDIDATES), base='CORR_HIGH_TOPO',
        context=dict(radius=.08, displacement_scales_mm=[.2, .5], log_divisor=4., mixture=[.5, .5]),
        drop=dict(categories=[0, 1, 2], pt_max_gev=10., coefficient=.10, context_base=.5),
        neutralize=dict(category=0, base=.08, context=.16),
        tracking_loss=dict(base=.15, context=.30, all_charged=True),
        stream='JC2_GAP_SWEEP_RANDOM/v1', seed=20261007, common_draws=True,
        unchanged_retained_p4=True, extra_merges=False, per_epoch_redraw=False,
        classification='development_tuned_synthetic_not_CMS_HLT')


def _draw(identity, domain, n):
    if not isinstance(identity, str) or not identity:
        raise ValueError('Jet identity is required')
    seed = hashlib.sha256(canonical_json_bytes(['JC2_GAP_SWEEP_RANDOM/v1', 20261007, identity, domain])).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(seed, 'big'))).random(n)


def generate(p, identity, candidate):
    if candidate not in CANDIDATES:
        raise ValueError('Unregistered strength; no implicit scan')
    # Canonical key order before random draws, output order is also canonical.
    p = p.take(sorted(range(len(p)), key=lambda i: p.keys[i]))
    n, strength = len(p), CANDIDATES[candidate]
    if n > 1:
        dr = np.hypot(p.eta[:, None]-p.eta, wrap_phi(p.phi[:, None]-p.phi))
        np.fill_diagonal(dr, np.inf)
        crowding = np.maximum(0., 1.-dr.min(axis=1)/.08)
    else:
        crowding = np.zeros(n)
    displacement = np.tanh(np.log1p(np.abs(p.tracking[:, :2])/np.array([.2, .5])).sum(axis=1)/4.)
    context = .5*(crowding+displacement)
    drop = (np.isin(p.category, [0, 1, 2]) & (p.pt < 10.) &
            (_draw(identity, 'drop', n) < strength*.10*(.5+.5*context)*np.maximum(0., 1.-p.pt/10.)))
    neutral = (p.category == 0) & ~drop & (_draw(identity, 'neutral', n) < strength*(.08+.16*context))
    lost = (p.charge != 0) & ~drop & ~neutral & (_draw(identity, 'tracking', n) < strength*(.15+.30*context))
    category, charge, tracking, valid = p.category.copy(), p.charge.copy(), p.tracking.copy(), p.valid.copy()
    category[neutral], charge[neutral] = 1, 0
    removed = neutral | lost
    tracking[removed], valid[removed] = 0., False
    keep = np.flatnonzero(~drop)
    output = Particles(p.p4[keep], charge[keep], category[keep], tracking[keep], valid[keep], tuple(p.keys[i] for i in keep))
    counts = dict(jets=1, input_particles=n, output_particles=len(output),
        dropped_particles=int(drop.sum()), charged_to_neutral=int(neutral.sum()),
        tracking_erased_particles=int((removed & p.valid.any(axis=1)).sum()),
        empty_output=int(len(output) == 0))
    return output, counts

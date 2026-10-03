"""Raw CMS endpoints -> paired physical views -> explicit feature interfaces.

No inverse of clipped model features and no response-model bridge filtering.
"""
from dataclasses import dataclass
from fractions import Fraction
import numpy as np

from hlt_classification.cms2jc2_response.bridge import Particles
from hlt_classification.cms_proxy_ladder.inputs import build_inputs, _eta_phi, _wrap_phi
from hlt_classification.cms_proxy_ladder.views import build_view, match_particles, _choose_offline
from hlt_classification.scouting.hcwdl_homotopy import prepare_hlt_endpoints, prepare_offline_endpoints
from hlt_classification.scouting.inputs import transform_hlt_endpoint_features
from .contracts import ARMS, CAPACITY

PID_CHANNELS = [4, 6, 5, 2, 3]  # common order: CH, NH, photon, electron, muon


@dataclass(frozen=True)
class Endpoint:
    particles: Particles
    raw: np.ndarray
    valid: np.ndarray


def endpoint(p4, raw, valid, *, prefix):
    raw = np.asarray(raw, np.float64)
    valid = np.asarray(valid, bool) & np.isfinite(raw) & (np.abs(raw) <= 1.e32)
    if raw.ndim != 2 or raw.shape[1] != 21 or valid.shape != raw.shape:
        raise ValueError("CMS raw endpoint must have 21 fields")
    n = len(raw)
    flags, charge = raw[:, PID_CHANNELS], raw[:, 1]
    if (not valid[:, [1, *PID_CHANNELS]].all() or not np.isin(flags, (0, 1)).all()
        or not np.isin(charge, (-1, 0, 1)).all()):
        raise ValueError("Invalid CMS raw identity flags/charge")
    category = np.where(flags.sum(1) == 1, flags.argmax(1), 5)
    coherent = (charge != 0) == np.isin(category, (0, 3, 4))
    category = np.where(coherent, category, 5)
    tracking, tv = np.zeros((n, 4)), np.zeros((n, 4), bool)
    for i, (value, significance) in enumerate(((13, 14), (12, 18))):
        ok = valid[:, value] & (charge != 0)
        error_ok = ok & valid[:, significance] & (raw[:, value] != 0) & (raw[:, significance] != 0)
        tracking[ok, i] = raw[ok, value] * 10.
        tv[:, i] = ok
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            error = np.abs(np.divide(raw[:, value], raw[:, significance],
                out=np.zeros(n), where=error_ok)) * 10.
        error_ok &= np.isfinite(error) & (error > 0)
        tracking[error_ok, i + 2] = error[error_ok]
        tv[:, i + 2] = error_ok
    physical = Particles(np.asarray(p4, np.float64), charge, category, tracking, tv,
                         tuple(f"{prefix}:{i}" for i in range(n)))
    return Endpoint(physical, np.where(valid, raw, 0.), valid)


def endpoints(raw):
    """Chunk conversion retains every cpfcandlt entry, including lost tracks."""
    h, o = prepare_hlt_endpoints(raw), prepare_offline_endpoints(raw)
    if h.rows != o.rows:
        raise ValueError("CMS paired endpoint row count differs")
    for i in range(h.rows):
        yield (endpoint(h.p4[i], h.raw_features[i], np.ones_like(h.raw_features[i], bool), prefix="hlt"),
               endpoint(o.p4[i], o.raw_features[i], o.validity[i], prefix="offline"))


def view(*, identity, hlt, offline, coordinate, mapping=None):
    if coordinate == "D000":
        return hlt
    if coordinate == "OFFLINE":
        if offline is None:
            raise PermissionError("Offline view requires offline inputs")
        return offline
    if coordinate not in ("U000", "U050", "U100", "D066", "D033") or offline is None:
        raise ValueError("Unregistered CMS view or absent offline endpoint")
    mapping = match_particles(hlt.particles, offline.particles) if mapping is None else np.asarray(mapping)
    physical = build_view(identity=identity, proxy=hlt.particles, offline=offline.particles,
                          coordinate=coordinate, mapping=mapping)
    raw, valid = hlt.raw.copy(), hlt.valid.copy()
    alpha = Fraction(1) if coordinate[0] == "U" else {"D066": Fraction(2, 3), "D033": Fraction(1, 3)}[coordinate]
    for slot in np.flatnonzero(mapping >= 0):
        source = int(mapping[slot])
        if coordinate[0] == "U":
            raw[slot], valid[slot] = offline.raw[source], offline.valid[source]
            continue
        take = _choose_offline(identity, int(slot), "identity", alpha)
        chosen, index = (offline, source) if take else (hlt, slot)
        raw[slot, 1:7], valid[slot, 1:7] = chosen.raw[index, 1:7], chosen.valid[index, 1:7]
        charged_changed = (hlt.particles.charge[slot] != 0) != (offline.particles.charge[source] != 0)
        for channel in [0, *range(7, 21)]:
            if channel in (0, 20) or (charged_changed and channel in (0, *range(11, 19), 20)):
                take_channel = take if charged_changed else _choose_offline(identity, int(slot), f"cms_field_{channel}", alpha)
                side, k = (offline, source) if take_channel else (hlt, slot)
                raw[slot, channel], valid[slot, channel] = side.raw[k, channel], side.valid[k, channel]
            elif hlt.valid[slot, channel] and offline.valid[source, channel]:
                raw[slot, channel] = (1. - float(alpha)) * hlt.raw[slot, channel] + float(alpha) * offline.raw[source, channel]
                valid[slot, channel] = True
            else:
                take_channel = _choose_offline(identity, int(slot), f"cms_valid_{channel}", alpha)
                side, k = (offline, source) if take_channel else (hlt, slot)
                raw[slot, channel], valid[slot, channel] = side.raw[k, channel], side.valid[k, channel]
    tail_lookup = {"offline:" + key: i for i, key in enumerate(offline.particles.keys)}
    tail = [tail_lookup[key] for key in physical.keys[len(hlt.particles):]]
    if tail:
        raw, valid = np.concatenate((raw, offline.raw[tail])), np.concatenate((valid, offline.valid[tail]))
    return Endpoint(physical, raw, valid)


def features(value, arm):
    if arm not in ARMS:
        raise ValueError("Unknown feature interface")
    p = value.particles
    if len(p) > CAPACITY:
        raise ValueError("CMS view exceeds 512 tokens; truncation forbidden")
    if arm == "SHARED17":
        result = build_inputs(p, capacity=CAPACITY)
        f = result.features.copy()
        # Unlike a single category, raw binary flags preserve CMS zero/multi-hot
        # identities without inventing a species. Flags switch atomically.
        f[:, 6:11] = value.raw[:, PID_CHANNELS]
        return f, result.vectors
    raw = value.raw.copy()
    eta, phi = _eta_phi(p.p4)
    axis_eta, axis_phi = _eta_phi(p.p4.sum(0))
    raw[:, 7] = _wrap_phi(phi - axis_phi)
    raw[:, 8] = (eta - axis_eta) * (-1 if axis_eta < 0 else 1)
    raw[:, 9] = np.abs(eta)
    raw[:, 10] = np.log(np.maximum(p.pt, 1e-8))
    raw[:, 19] = np.log(np.maximum(p.p4[:, 3], 1e-8))
    f = transform_hlt_endpoint_features(raw)
    if not np.isfinite(f).all():
        raise ValueError("Nonfinite CMS input tensor")
    return f, p.p4.astype(np.float32)


def audit(value):
    p = value.particles
    flags = value.raw[:, PID_CHANNELS]
    exactly_one = flags.sum(1) == 1
    charged_pid = np.isin(flags.argmax(1), (0, 3, 4))
    return dict(particles=len(p), charged=int((p.charge != 0).sum()),
        ambiguous_pid=int((~exactly_one).sum()),
        incoherent_pid_charge=int((exactly_one & (charged_pid != (p.charge != 0))).sum()),
        valid_d0=int(p.valid[:, 0].sum()), valid_dz=int(p.valid[:, 1].sum()),
        recovered_d0err=int(p.valid[:, 2].sum()), recovered_dzerr=int(p.valid[:, 3].sum()))

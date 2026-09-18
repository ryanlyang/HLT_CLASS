"""Common ordered topology exposures and physical emission coordinates.

Calibration and generation share precisely the same merge proposal/order.
Associations are hypotheses; unresolved jets remain available to evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .association import groups, _axis
from .bridge import Particles, p4_from_coordinates, wrap_phi
from .contracts import validate
from .features import FEATURE_NAMES, features, group_features
from .rng import group_key

COORDINATES = ("log_pt_ratio", "delta_eta", "delta_phi", "log_mass",
               "d0", "dz", "log_d0err", "log_dzerr")


@dataclass(frozen=True)
class Record:
    module: str
    key: str
    x: np.ndarray
    target: np.ndarray | tuple


def jet_features(p: Particles, x: np.ndarray | None = None) -> np.ndarray:
    """Allowed particle summaries only: no mass, label or target jet axis."""
    if not len(p):
        result = np.zeros(len(FEATURE_NAMES))
        result[0] = 5
        return result
    return group_features(p, tuple(range(len(p))), x)


def reference(p: Particles, indices) -> tuple[float, float, float]:
    axis = _axis(p.p4[list(indices)].sum(axis=0)) if len(indices) else None
    # The coordinate origin for a directionless/empty source is a declared
    # convention, not an inferred HLT axis. Count this in the generation audit.
    return axis if axis is not None else (1., 0., 0.)


def emission_state(output: Particles) -> tuple[int, ...]:
    state = []
    for i in range(len(output)):
        mask = sum(int(output.valid[i, j]) << j for j in range(4))
        state.extend((int(output.category[i]), int(output.charge[i]), mask,
                      int(output.mass[i] > 1e-6)))
    return tuple(state)


def condition_on_state(x: np.ndarray, state: tuple[int, ...]) -> np.ndarray:
    if len(state) not in (4, 8):
        raise ValueError("Only singleton or paired emission states are supported")
    row = np.array(x, dtype=np.float64, copy=True)
    row[35:] = 0
    row[35:35+len(state)] = state
    return row


def emission_coordinates(output: Particles, origin) -> np.ndarray:
    if len(output) not in (1, 2):
        raise ValueError("Invalid emission cardinality")
    pt, eta, phi = origin
    rows = []
    for i in range(len(output)):
        rows.append((np.log(output.pt[i]/pt), output.eta[i]-eta,
                     float(wrap_phi(output.phi[i]-phi)),
                     np.log(output.mass[i]) if output.mass[i] > 1e-6 else 0.,
                     *output.tracking[i, :2],
                     *np.where(output.valid[i, 2:],
                               np.log(np.maximum(output.tracking[i, 2:], 1e-300)), 0.)))
    return np.asarray(rows, np.float64).ravel()


def decode_emission(coordinates, state, origin, *, key: str) -> Particles:
    state = tuple(int(v) for v in state)
    n = len(state)//4
    values = np.asarray(coordinates, np.float64).reshape(n, len(COORDINATES))
    if n not in (1, 2) or len(state) != n*4 or not np.isfinite(values).all():
        raise ValueError("Invalid generated emission")
    cat, charge, tracking, valid, p4 = [], [], [], [], []
    for i, row in enumerate(values):
        c, q, mask, positive_mass = state[i*4:i*4+4]
        flags = np.array([(mask >> j) & 1 for j in range(4)], bool)
        if mask not in range(16) or positive_mass not in (0, 1):
            raise ValueError("Invalid generated measurement state")
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            pt = origin[0]*np.exp(row[0])
            mass = np.exp(row[3]) if positive_mass else 0.
            tr = np.array([row[4], row[5],
                           np.exp(row[6]) if flags[2] else 0.,
                           np.exp(row[7]) if flags[3] else 0.])
        tr[~flags] = 0.
        p4.append(p4_from_coordinates(pt, origin[1]+row[1],
                                     float(wrap_phi(origin[2]+row[2])), mass))
        cat.append(c); charge.append(q); tracking.append(tr); valid.append(flags)
    return Particles(np.asarray(p4), np.asarray(charge), np.asarray(cat),
                     np.asarray(tracking), np.asarray(valid),
                     tuple(f"{key}:child:{i}" for i in range(n)))


def calibration_records(offline: Particles, hlt: Particles, association: dict,
                        rules: dict) -> tuple[list[Record], dict]:
    validate(association, "ASSOCIATION", parents={"policy": rules["content_hash"]})
    if association["offline_keys"] != list(offline.keys) or association["hlt_keys"] != list(hlt.keys):
        raise ValueError("Association object identities differ")
    if not association["resolved"]:
        return [], dict(resolved=False, jets=1, offline=len(offline), hlt=len(hlt))
    x = features(offline)
    selected = {frozenset(h["offline"]): h for h in association["hypotheses"] if h["offline"]}
    consumed, records = set(), []

    def emit(indices, targets, module):
        order = sorted(targets, key=lambda j: (-float(hlt.pt[j]), hlt.keys[j]))
        out = hlt.take(order)
        key = group_key([offline.keys[i] for i in indices]) if indices else "additional"
        row = group_features(offline, tuple(indices), x) if indices else jet_features(offline, x)
        state = emission_state(out)
        suffix = ":" + ":".join(hlt.keys[j] for j in order)
        records.append(Record(module+"_state", key+suffix, row, state))
        records.append(Record(module+"_value", key+suffix, condition_on_state(row, state),
                              emission_coordinates(out, reference(offline, indices or tuple(range(len(offline)))))))

    for indices in groups(offline, rules["same_side_radius"]):
        if consumed.intersection(indices):
            continue
        h = selected.get(frozenset(indices))
        accepted = h is not None and len(h["hlt"]) == 1
        key = group_key([offline.keys[i] for i in indices])
        records.append(Record("merge", key, group_features(offline, indices, x), (int(accepted),)))
        if accepted:
            consumed.update(indices)
            emit(indices, h["hlt"], "emission1")
    for i in sorted(set(range(len(offline)))-consumed, key=lambda i: offline.keys[i]):
        h = selected.get(frozenset((i,)))
        if h is None:
            raise ValueError("Resolved topology cannot be replayed by the common proposal")
        count = len(h["hlt"])
        if count not in (0, 1, 2):
            raise ValueError("Unregistered split cardinality")
        records.append(Record("singleton", offline.keys[i], x[i], (count,)))
        if count:
            emit((i,), h["hlt"], f"emission{count}")
    additional = [j for h in association["hypotheses"] if not h["offline"] for j in h["hlt"]]
    records.append(Record("additional_count", "jet", jet_features(offline, x), (len(additional),)))
    for j in sorted(additional, key=lambda j: hlt.keys[j]):
        emit((), (j,), "additional")
    return records, dict(resolved=True, jets=1, offline=len(offline), hlt=len(hlt),
                         merges=sum(len(h["offline"]) > 1 for h in association["hypotheses"]),
                         splits=sum(len(h["hlt"]) > 1 for h in association["hypotheses"]),
                         additional=len(additional))

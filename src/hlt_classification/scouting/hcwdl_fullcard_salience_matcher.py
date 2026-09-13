"""Exact full-cardinality salience-weighted angular pairing.

This module implements a forced training coordinate.  It does not estimate a
physical correspondence probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable

import numpy as np

from .hcwdl_fullcard_bottleneck_matcher import (
    REFERENCE_ENUMERATOR_MAX_SIDE,
    _integer_hungarian,
    _native_indices,
    canonical_qabs_log_pt_response,
    canonical_qdr,
    validate_pairing,
)
from .hcwdl_fullcard_salience_contracts import (
    CANDIDATES,
    DR_QUANTUM,
    PAIR_DR_CAP,
    PT_SHARE_SCALE,
    SALIENCE_FLOOR,
    SALIENCE_PT_LINEAR,
    SALIENCE_PT_QUADRATIC,
    SALIENCE_PT_QUADRATIC_CORE25,
    SOLVER,
)
from .highcov_data import Particles
from .highcov_features import edge_matrices, kinematics, wrap_phi


QCAP = int(np.rint(PAIR_DR_CAP / DR_QUANTUM))


@dataclass(frozen=True)
class SaliencePairingResult:
    """HLT-oriented assignment with exact salience diagnostics."""

    concatenated_offline_index: np.ndarray
    native_offline_index: np.ndarray
    pairing_validity: np.ndarray
    selected_qdr: np.ndarray
    selected_qabs_log_pt_response: np.ndarray
    hlt_salience: np.ndarray
    offline_salience: np.ndarray
    selected_utility: np.ndarray
    candidate: str
    solver: str

    @property
    def selected_count(self) -> int:
        return int(np.count_nonzero(self.pairing_validity))


def _round_ratio_ties_to_even(numerator: int, denominator: int) -> int:
    """Round a nonnegative exact rational using ties-to-even."""

    if numerator < 0 or denominator <= 0:
        raise ValueError("canonical ratio must be nonnegative with positive denominator")
    quotient, remainder = divmod(int(numerator), int(denominator))
    doubled = 2 * remainder
    if doubled > denominator or (doubled == denominator and quotient % 2):
        quotient += 1
    return quotient


def _quantized_pt_shares(pt: np.ndarray) -> np.ndarray:
    values = np.asarray(pt, np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("salience requires finite positive particle pT")
    total = float(np.sum(values, dtype=np.float64))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("salience endpoint scalar-pT sum is invalid")
    scaled = values / total * np.float64(PT_SHARE_SCALE)
    if not np.isfinite(scaled).all():
        raise ValueError("salience pT shares are nonfinite")
    return np.rint(scaled).astype(np.int64)


def _axis_qdr(p4: np.ndarray) -> np.ndarray:
    value = np.asarray(p4, np.float64)
    if value.ndim != 2 or value.shape[1] != 4 or not np.isfinite(value).all():
        raise ValueError("salience endpoint four-vectors are invalid")
    if len(value) == 0:
        return np.empty(0, np.int64)
    pt, eta, phi, _ = kinematics(value)
    if np.any(pt <= 0):
        raise ValueError("salience requires positive endpoint particle pT")
    jet = np.sum(value, axis=0, dtype=np.float64)
    jet_pt = float(np.hypot(jet[0], jet[1]))
    if not np.isfinite(jet).all() or jet_pt <= 0:
        raise ValueError("salience own-endpoint jet axis is invalid")
    jet_eta = float(np.arcsinh(jet[2] / jet_pt))
    jet_phi = float(np.arctan2(jet[1], jet[0]))
    dr = np.hypot(eta - jet_eta, wrap_phi(phi - jet_phi))
    return canonical_qdr(dr)


def particle_salience(particles: Particles, candidate: str) -> np.ndarray:
    """Compute one candidate's canonical positive integer particle weights."""

    if candidate not in CANDIDATES:
        raise ValueError("unknown full-cardinality salience candidate")
    pt, _, _, _ = kinematics(particles.p4)
    share = _quantized_pt_shares(pt)
    if candidate == SALIENCE_PT_LINEAR:
        return (SALIENCE_FLOOR + share).astype(np.int64)
    power = np.asarray([
        _round_ratio_ties_to_even(int(value) * int(value), PT_SHARE_SCALE)
        for value in share
    ], dtype=np.int64)
    base = SALIENCE_FLOOR + power
    if candidate == SALIENCE_PT_QUADRATIC:
        return base.astype(np.int64)
    axis = _axis_qdr(particles.p4)
    core = np.maximum(QCAP - np.minimum(axis, QCAP), 0)
    bonus = np.asarray([
        _round_ratio_ties_to_even(int(weight) * int(core_value), 4 * QCAP)
        for weight, core_value in zip(base, core, strict=True)
    ], dtype=np.int64)
    return (base + bonus).astype(np.int64)


def pair_utility(
    qdr: np.ndarray, hlt_salience: np.ndarray, offline_salience: np.ndarray,
) -> np.ndarray:
    distance = np.asarray(qdr, np.int64)
    hweight = np.asarray(hlt_salience, np.int64)
    oweight = np.asarray(offline_salience, np.int64)
    if distance.shape != (len(hweight), len(oweight)) or np.any(distance < 0):
        raise ValueError("salience edge matrices differ")
    if np.any(hweight <= 0) or np.any(oweight <= 0):
        raise ValueError("particle salience must be positive")
    closeness = np.maximum(QCAP - np.minimum(distance, QCAP), 0).astype(object)
    weights = hweight[:, None].astype(object) + oweight[None, :].astype(object)
    return weights * closeness


def _mapping_from_oriented_permutation(
    chosen_right: Iterable[int], *, nh: int, no: int,
) -> np.ndarray:
    mapping = np.full(nh, -1, np.int32)
    if nh <= no:
        mapping[:] = np.fromiter(chosen_right, dtype=np.int32, count=nh)
    else:
        for offline_index, hlt_index in enumerate(chosen_right):
            mapping[int(hlt_index)] = offline_index
    return mapping


def assignment_signature(
    mapping: np.ndarray, *, utility: np.ndarray, qdr: np.ndarray,
    qresponse: np.ndarray, hlt_salience: np.ndarray,
    offline_salience: np.ndarray, hlt_category: np.ndarray,
    offline_category: np.ndarray, hlt_charge: np.ndarray,
    offline_charge: np.ndarray, native_offline_index: np.ndarray,
) -> tuple[object, ...]:
    """Return the frozen hierarchy for an HLT-oriented assignment."""

    value = np.asarray(mapping, np.int64)
    rows = np.flatnonzero(value >= 0)
    columns = value[rows]
    nh, no = np.shape(qdr)
    larger_salience = (
        np.asarray(offline_salience, np.int64)[columns]
        if nh <= no else np.asarray(hlt_salience, np.int64)[rows]
    )
    hcharge = np.asarray(hlt_charge)[rows]
    ocharge = np.asarray(offline_charge)[columns]
    valid_charge = np.isin(hcharge, (-1, 0, 1)) & np.isin(ocharge, (-1, 0, 1))
    native = np.asarray(native_offline_index, np.int64)
    sentinel = int(np.max(native, initial=-1)) + 1
    native_tuple = tuple(
        sentinel if column < 0 else int(native[column]) for column in value
    )
    return (
        -sum(int(utility[row, column]) for row, column in zip(rows, columns, strict=True)),
        sum(int(qdr[row, column]) for row, column in zip(rows, columns, strict=True)),
        -sum(int(item) for item in larger_salience),
        sum(int(qresponse[row, column]) for row, column in zip(rows, columns, strict=True)),
        int(np.count_nonzero(
            np.asarray(hlt_category)[rows] != np.asarray(offline_category)[columns]
        )),
        int(np.count_nonzero(valid_charge & (hcharge != ocharge))),
        native_tuple,
    )


def reference_pairing_from_matrices(
    *, utility: np.ndarray, qdr: np.ndarray, qresponse: np.ndarray,
    hlt_salience: np.ndarray, offline_salience: np.ndarray,
    hlt_category: np.ndarray, offline_category: np.ndarray,
    hlt_charge: np.ndarray, offline_charge: np.ndarray,
    native_offline_index: np.ndarray,
) -> np.ndarray:
    distance = np.asarray(qdr, np.int64)
    if distance.ndim != 2 or np.shape(utility) != distance.shape:
        raise ValueError("salience reference edge matrices differ")
    nh, no = distance.shape
    if max(nh, no) > REFERENCE_ENUMERATOR_MAX_SIDE:
        raise ValueError("salience reference requires both cardinalities <= 8")
    if nh == 0 or no == 0:
        return np.full(nh, -1, np.int32)
    right_count, left_count = (no, nh) if nh <= no else (nh, no)
    best_mapping = None
    best_signature = None
    kwargs = dict(
        utility=utility, qdr=distance, qresponse=qresponse,
        hlt_salience=hlt_salience, offline_salience=offline_salience,
        hlt_category=hlt_category, offline_category=offline_category,
        hlt_charge=hlt_charge, offline_charge=offline_charge,
        native_offline_index=native_offline_index,
    )
    for chosen in permutations(range(right_count), left_count):
        mapping = _mapping_from_oriented_permutation(chosen, nh=nh, no=no)
        signature = assignment_signature(mapping, **kwargs)
        if best_signature is None or signature < best_signature:
            best_mapping, best_signature = mapping, signature
    if best_mapping is None:  # pragma: no cover
        raise RuntimeError("salience reference enumeration produced no assignment")
    return best_mapping


def _validated_arrays(**arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    utility = np.asarray(arrays["utility"], dtype=object)
    qdr = np.asarray(arrays["qdr"], np.int64)
    qresponse = np.asarray(arrays["qresponse"], np.int64)
    if qdr.ndim != 2 or utility.shape != qdr.shape or qresponse.shape != qdr.shape:
        raise ValueError("salience production edge matrices differ")
    nh, no = qdr.shape
    vectors = (
        np.asarray(arrays["hlt_salience"], np.int64),
        np.asarray(arrays["offline_salience"], np.int64),
        np.asarray(arrays["hlt_category"]), np.asarray(arrays["offline_category"]),
        np.asarray(arrays["hlt_charge"]), np.asarray(arrays["offline_charge"]),
        np.asarray(arrays["native_offline_index"], np.int64),
    )
    shapes = ((nh,), (no,), (nh,), (no,), (nh,), (no,), (no,))
    if any(value.shape != shape for value, shape in zip(vectors, shapes, strict=True)):
        raise ValueError("salience particle vectors differ from edge matrices")
    if np.any(qdr < 0) or np.any(qresponse < 0):
        raise ValueError("canonical salience edge matrices must be nonnegative")
    if any(int(value) < 0 for value in utility.flat):
        raise ValueError("salience utility must be nonnegative")
    hweight, oweight, _, _, _, _, native = vectors
    if np.any(hweight <= 0) or np.any(oweight <= 0):
        raise ValueError("salience weights must be positive")
    if np.any(native < 0) or len(np.unique(native)) != no:
        raise ValueError("native offline indices must be unique and nonnegative")
    return (utility, qdr, qresponse, *vectors)


def production_pairing_from_matrices(**arrays: np.ndarray) -> np.ndarray:
    """Solve every registered objective exactly with arbitrary integers."""

    (
        utility, qdr, qresponse, hweight, oweight, hcat, ocat,
        hcharge, ocharge, native,
    ) = _validated_arrays(**arrays)
    nh, no = qdr.shape
    if nh == 0 or no == 0:
        return np.full(nh, -1, np.int32)
    pair_count = min(nh, no)
    sentinel = int(np.max(native)) + 1
    native_base = sentinel + 1
    native_places = [pow(native_base, nh - 1 - row) for row in range(nh)]
    native_max = sentinel * sum(native_places)

    lower_max = native_max
    charge_unit = lower_max + 1
    lower_max += pair_count * charge_unit
    category_unit = lower_max + 1
    lower_max += pair_count * category_unit
    response_unit = lower_max + 1
    lower_max += pair_count * int(np.max(qresponse, initial=0)) * response_unit
    larger_max = int(max(np.max(hweight, initial=0), np.max(oweight, initial=0)))
    salience_unit = lower_max + 1
    lower_max += pair_count * larger_max * salience_unit
    qdr_unit = lower_max + 1
    lower_max += pair_count * int(np.max(qdr, initial=0)) * qdr_unit
    primary_unit = lower_max + 1
    utility_max = max(int(value) for value in utility.flat)

    valid_hcharge = np.isin(hcharge, (-1, 0, 1))
    valid_ocharge = np.isin(ocharge, (-1, 0, 1))
    real_cost: list[list[int]] = []
    for i in range(nh):
        row = []
        for j in range(no):
            larger = int(oweight[j] if nh <= no else hweight[i])
            row.append(
                (utility_max - int(utility[i, j])) * primary_unit
                + int(qdr[i, j]) * qdr_unit
                + (larger_max - larger) * salience_unit
                + int(qresponse[i, j]) * response_unit
                + int(hcat[i] != ocat[j]) * category_unit
                + int(valid_hcharge[i] and valid_ocharge[j] and hcharge[i] != ocharge[j])
                * charge_unit
                + int(native[j]) * native_places[i]
            )
        real_cost.append(row)

    if nh <= no:
        return np.asarray(_integer_hungarian(real_cost), np.int32)

    dummy_count = nh - no
    cost = [
        row + [sentinel * native_places[i]] * dummy_count
        for i, row in enumerate(real_cost)
    ]
    selected = _integer_hungarian(cost)
    return np.asarray([column if column < no else -1 for column in selected], np.int32)


class FullCardinalitySalienceMatcher:
    """Stateless matcher for one preregistered salience candidate."""

    def __init__(self, candidate: str) -> None:
        if candidate not in CANDIDATES:
            raise ValueError("unknown full-cardinality salience candidate")
        self.candidate = candidate

    def match(self, hlt: Particles, offline: Particles) -> SaliencePairingResult:
        matrices = edge_matrices(hlt, offline)
        qdr = canonical_qdr(matrices.dr)
        qresponse = canonical_qabs_log_pt_response(matrices.log_pt)
        hweight = particle_salience(hlt, self.candidate)
        oweight = particle_salience(offline, self.candidate)
        utility = pair_utility(qdr, hweight, oweight)
        native = _native_indices(offline)
        arguments = dict(
            utility=utility, qdr=qdr, qresponse=qresponse,
            hlt_salience=hweight, offline_salience=oweight,
            hlt_category=hlt.category, offline_category=offline.category,
            hlt_charge=hlt.charge, offline_charge=offline.charge,
            native_offline_index=native,
        )
        mapping = production_pairing_from_matrices(**arguments)
        validate_pairing(mapping, nh=len(hlt.p4), no=len(offline.p4))
        accepted = mapping >= 0
        rows = np.flatnonzero(accepted)
        columns = mapping[accepted]
        native_mapping = np.full(len(mapping), -1, np.int32)
        selected_qdr = np.full(len(mapping), -1, np.int64)
        selected_response = np.full(len(mapping), -1, np.int64)
        selected_utility = np.full(len(mapping), -1, dtype=object)
        native_mapping[rows] = native[columns].astype(np.int32)
        selected_qdr[rows] = qdr[rows, columns]
        selected_response[rows] = qresponse[rows, columns]
        selected_utility[rows] = utility[rows, columns]
        return SaliencePairingResult(
            concatenated_offline_index=mapping,
            native_offline_index=native_mapping,
            pairing_validity=accepted,
            selected_qdr=selected_qdr,
            selected_qabs_log_pt_response=selected_response,
            hlt_salience=hweight,
            offline_salience=oweight,
            selected_utility=selected_utility,
            candidate=self.candidate,
            solver=SOLVER,
        )


__all__ = [
    "FullCardinalitySalienceMatcher", "QCAP", "REFERENCE_ENUMERATOR_MAX_SIDE",
    "SaliencePairingResult", "assignment_signature", "pair_utility",
    "particle_salience", "production_pairing_from_matrices",
    "reference_pairing_from_matrices", "validate_pairing",
]

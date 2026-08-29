"""Exact forced full-cardinality lexicographic bottleneck pairing.

This is a geometry-first control coordinate, not a physical truth matcher and
not a calibrated correspondence probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable

import numpy as np

from .hcwdl_fullcard_bottleneck_contracts import (
    ABS_LOG_PT_RESPONSE_QUANTUM,
    DR_QUANTUM,
    SOLVER,
)
from .highcov_data import Particles
from .highcov_features import edge_matrices


@dataclass(frozen=True)
class PairingResult:
    """HLT-oriented forced assignment and its exact canonical diagnostics."""

    concatenated_offline_index: np.ndarray
    native_offline_index: np.ndarray
    pairing_validity: np.ndarray
    selected_qdr: np.ndarray
    selected_qabs_log_pt_response: np.ndarray
    solver: str

    @property
    def selected_count(self) -> int:
        return int(np.count_nonzero(self.pairing_validity))


def _canonical_nonnegative(value: np.ndarray, *, quantum: float, name: str) -> np.ndarray:
    array = np.asarray(value, np.float64)
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    scaled = array / np.float64(quantum)
    if np.any(scaled > np.iinfo(np.int64).max):
        raise OverflowError(f"{name} exceeds canonical int64 range")
    # np.rint follows IEEE-754 round-to-nearest, ties-to-even semantics.
    return np.rint(scaled).astype(np.int64)


def canonical_qdr(dr: np.ndarray) -> np.ndarray:
    return _canonical_nonnegative(dr, quantum=DR_QUANTUM, name="delta-R")


def canonical_qabs_log_pt_response(log_pt: np.ndarray) -> np.ndarray:
    return _canonical_nonnegative(
        np.abs(np.asarray(log_pt, np.float64)),
        quantum=ABS_LOG_PT_RESPONSE_QUANTUM,
        name="absolute log-pT response",
    )


def _native_indices(offline: Particles) -> np.ndarray:
    if offline.native_index is None:
        native = np.arange(len(offline.p4), dtype=np.int64)
    else:
        native = np.asarray(offline.native_index, np.int64)
    if native.shape != (len(offline.p4),) or np.any(native < 0):
        raise ValueError("offline native indices must be nonnegative [particles]")
    if len(np.unique(native)) != len(native):
        raise ValueError("offline native indices must be one-to-one")
    return native


def assignment_signature(
    mapping: np.ndarray,
    *,
    qdr: np.ndarray,
    qresponse: np.ndarray,
    hlt_category: np.ndarray,
    offline_category: np.ndarray,
    hlt_charge: np.ndarray,
    offline_charge: np.ndarray,
    native_offline_index: np.ndarray,
) -> tuple[tuple[int, ...], tuple[int, ...], int, int, tuple[int, ...]]:
    """Return the frozen complete comparison signature for one assignment."""

    selected = np.flatnonzero(np.asarray(mapping) >= 0)
    columns = np.asarray(mapping, np.int64)[selected]
    primary = tuple(sorted((int(x) for x in qdr[selected, columns]), reverse=True))
    response = tuple(sorted((int(x) for x in qresponse[selected, columns]), reverse=True))
    category = int(np.count_nonzero(
        np.asarray(hlt_category)[selected] != np.asarray(offline_category)[columns]
    ))
    hcharge = np.asarray(hlt_charge)[selected]
    ocharge = np.asarray(offline_charge)[columns]
    valid = np.isin(hcharge, (-1, 0, 1)) & np.isin(ocharge, (-1, 0, 1))
    charge = int(np.count_nonzero(valid & (hcharge != ocharge)))
    native = np.asarray(native_offline_index, np.int64)
    sentinel = int(np.max(native, initial=-1)) + 1
    native_tuple = tuple(
        sentinel if int(column) < 0 else int(native[int(column)])
        for column in np.asarray(mapping, np.int64)
    )
    return primary, response, category, charge, native_tuple


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


def reference_pairing_from_matrices(
    *,
    qdr: np.ndarray,
    qresponse: np.ndarray,
    hlt_category: np.ndarray,
    offline_category: np.ndarray,
    hlt_charge: np.ndarray,
    offline_charge: np.ndarray,
    native_offline_index: np.ndarray,
) -> np.ndarray:
    """Exhaustive exact reference for bounded test and audit multiplicities."""

    qdr = np.asarray(qdr, np.int64)
    qresponse = np.asarray(qresponse, np.int64)
    if qdr.ndim != 2 or qresponse.shape != qdr.shape:
        raise ValueError("canonical edge matrices differ")
    nh, no = qdr.shape
    if min(nh, no) > 9:
        raise ValueError("reference enumerator is restricted to min cardinality <= 9")
    if nh == 0 or no == 0:
        return np.full(nh, -1, np.int32)
    right_count = no if nh <= no else nh
    left_count = nh if nh <= no else no
    best_mapping: np.ndarray | None = None
    best_signature: tuple[object, ...] | None = None
    for chosen in permutations(range(right_count), left_count):
        mapping = _mapping_from_oriented_permutation(chosen, nh=nh, no=no)
        signature = assignment_signature(
            mapping,
            qdr=qdr,
            qresponse=qresponse,
            hlt_category=hlt_category,
            offline_category=offline_category,
            hlt_charge=hlt_charge,
            offline_charge=offline_charge,
            native_offline_index=native_offline_index,
        )
        if best_signature is None or signature < best_signature:
            best_signature = signature
            best_mapping = mapping
    if best_mapping is None:  # pragma: no cover - guarded by nonempty dimensions
        raise RuntimeError("reference assignment enumeration produced no candidate")
    return best_mapping


def _rank_units(values: np.ndarray, *, base: int) -> tuple[list[int], int]:
    unique = np.unique(np.asarray(values, np.int64))
    rank = {int(value): index for index, value in enumerate(unique.tolist())}
    units = [pow(base, index) for index in range(len(unique))]
    return [units[rank[int(value)]] for value in np.asarray(values).ravel()], pow(base, len(unique))


def _bottleneck_threshold(qdr: np.ndarray) -> int:
    """Find the exact smallest feasible maximum edge level in C-backed scans."""

    from scipy.optimize import linear_sum_assignment

    nh, no = qdr.shape
    oriented = qdr if nh <= no else qdr.T
    levels = np.unique(oriented)
    low = 0
    high = len(levels) - 1
    while low < high:
        middle = (low + high) // 2
        forbidden = (oriented > levels[middle]).astype(np.int8)
        rows, columns = linear_sum_assignment(forbidden)
        if int(forbidden[rows, columns].sum()) == 0:
            high = middle
        else:
            low = middle + 1
    return int(levels[low])


def _rank_matrix(
    values: np.ndarray, *, allowed: np.ndarray, base: int,
) -> tuple[np.ndarray, int]:
    selected = np.asarray(values, np.int64)[allowed]
    units, bound = _rank_units(selected, base=base)
    result = np.zeros(np.shape(values), dtype=object)
    result[allowed] = np.asarray(units, dtype=object)
    return result, bound


def _integer_hungarian(cost: list[list[int]]) -> list[int]:
    """Exact rectangular Hungarian solver over arbitrary-precision integers."""

    n = len(cost)
    m = len(cost[0]) if n else 0
    if n > m or any(len(row) != m for row in cost):
        raise ValueError("integer Hungarian requires a rectangular n <= m matrix")
    if n == 0:
        return []
    u = [0] * (n + 1)
    v = [0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv: list[int | None] = [None] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta: int | None = None
            j1 = 0
            row = cost[i0 - 1]
            for j in range(1, m + 1):
                if used[j]:
                    continue
                current = row[j - 1] - u[i0] - v[j]
                if minv[j] is None or current < minv[j]:
                    minv[j] = current
                    way[j] = j0
                if delta is None or minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if delta is None:
                raise RuntimeError("integer Hungarian found no augmenting column")
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                elif minv[j] is not None:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    if any(value < 0 for value in assignment):
        raise RuntimeError("integer Hungarian returned an incomplete assignment")
    return assignment


def production_pairing_from_matrices(
    *,
    qdr: np.ndarray,
    qresponse: np.ndarray,
    hlt_category: np.ndarray,
    offline_category: np.ndarray,
    hlt_charge: np.ndarray,
    offline_charge: np.ndarray,
    native_offline_index: np.ndarray,
) -> np.ndarray:
    """Solve the complete frozen hierarchy exactly with integer arithmetic."""

    qdr = np.asarray(qdr, np.int64)
    qresponse = np.asarray(qresponse, np.int64)
    if qdr.ndim != 2 or qresponse.shape != qdr.shape:
        raise ValueError("canonical edge matrices differ")
    nh, no = qdr.shape
    native = np.asarray(native_offline_index, np.int64)
    hcat = np.asarray(hlt_category)
    ocat = np.asarray(offline_category)
    hcharge = np.asarray(hlt_charge)
    ocharge = np.asarray(offline_charge)
    if (
        hcat.shape != (nh,) or hcharge.shape != (nh,)
        or ocat.shape != (no,) or ocharge.shape != (no,)
        or native.shape != (no,)
    ):
        raise ValueError("particle identity arrays differ from edge matrices")
    if np.any(qdr < 0) or np.any(qresponse < 0):
        raise ValueError("canonical edge matrices must be nonnegative")
    if nh == 0 or no == 0:
        return np.full(nh, -1, np.int32)
    if np.any(native < 0) or len(np.unique(native)) != no:
        raise ValueError("native offline indices must be unique and nonnegative")

    pair_count = min(nh, no)
    count_base = pair_count + 1
    threshold = _bottleneck_threshold(qdr)
    allowed = qdr <= threshold
    qdr_unit, qdr_bound = _rank_matrix(qdr, allowed=allowed, base=count_base)
    response_unit, response_bound = _rank_matrix(
        qresponse, allowed=allowed, base=count_base,
    )

    native_base = int(np.max(native)) + 2
    native_bound = pow(native_base, nh)
    charge_weight = native_bound
    category_weight = (pair_count + 1) * charge_weight
    response_weight = (pair_count + 1) * category_weight
    primary_weight = response_bound * response_weight
    # qdr_bound is intentionally not needed as a multiplier: it is the bound
    # on the complete primary code rather than on a lower-priority block.
    del qdr_bound

    valid_hcharge = np.isin(hcharge, (-1, 0, 1))
    valid_ocharge = np.isin(ocharge, (-1, 0, 1))
    real_cost: list[list[int]] = []
    for i in range(nh):
        native_place = pow(native_base, nh - 1 - i)
        row: list[int] = []
        for j in range(no):
            category_mismatch = int(hcat[i] != ocat[j])
            charge_mismatch = int(
                valid_hcharge[i] and valid_ocharge[j] and hcharge[i] != ocharge[j]
            )
            row.append(
                int(qdr_unit[i, j]) * primary_weight
                + int(response_unit[i, j]) * response_weight
                + category_mismatch * category_weight
                + charge_mismatch * charge_weight
                + int(native[j]) * native_place
            )
        real_cost.append(row)

    maximum_allowed_cost = max(
        real_cost[i][j]
        for i in range(nh) for j in range(no) if allowed[i, j]
    )
    forbidden_cost = (maximum_allowed_cost + 1) * (pair_count + 1)
    for i in range(nh):
        for j in range(no):
            if not allowed[i, j]:
                real_cost[i][j] = forbidden_cost

    if nh <= no:
        selected = _integer_hungarian(real_cost)
        mapping = np.asarray(selected, np.int32)
        if np.any(~allowed[np.arange(nh), mapping]):
            raise RuntimeError("bottleneck-pruned assignment crossed its exact threshold")
        return mapping

    # Exactly nh-no interchangeable private dummy endpoints represent the
    # unavoidable unmatched HLT particles. They add no pair objective term;
    # their native sentinel is larger than every real native index.
    dummy_count = nh - no
    sentinel = int(np.max(native)) + 1
    cost: list[list[int]] = []
    for i, row in enumerate(real_cost):
        native_place = pow(native_base, nh - 1 - i)
        cost.append(row + [sentinel * native_place] * dummy_count)
    selected = _integer_hungarian(cost)
    mapping = np.asarray([value if value < no else -1 for value in selected], np.int32)
    real_rows = np.flatnonzero(mapping >= 0)
    if np.any(~allowed[real_rows, mapping[real_rows]]):
        raise RuntimeError("bottleneck-pruned assignment crossed its exact threshold")
    return mapping


def validate_pairing(mapping: np.ndarray, *, nh: int, no: int) -> None:
    value = np.asarray(mapping)
    if value.dtype.kind not in "iu" or value.shape != (nh,):
        raise ValueError("full-cardinality mapping must be integer [HLT particles]")
    if np.any((value < -1) | (value >= no)):
        raise ValueError("full-cardinality mapping contains an out-of-bounds endpoint")
    accepted = value[value >= 0]
    if len(accepted) != min(nh, no):
        raise ValueError("full-cardinality mapping does not cover the smaller side")
    if len(np.unique(accepted)) != len(accepted):
        raise ValueError("full-cardinality mapping reuses an offline endpoint")
    if nh <= no and np.any(value < 0):
        raise ValueError("full-cardinality mapping abstained despite HLT being smaller")


class FullCardinalityBottleneckMatcher:
    """Stateless exact matcher for the registered forced-pairing control."""

    def match(self, hlt: Particles, offline: Particles) -> PairingResult:
        matrices = edge_matrices(hlt, offline)
        qdr = canonical_qdr(matrices.dr)
        qresponse = canonical_qabs_log_pt_response(matrices.log_pt)
        native = _native_indices(offline)
        mapping = production_pairing_from_matrices(
            qdr=qdr,
            qresponse=qresponse,
            hlt_category=hlt.category,
            offline_category=offline.category,
            hlt_charge=hlt.charge,
            offline_charge=offline.charge,
            native_offline_index=native,
        )
        validate_pairing(mapping, nh=len(hlt.p4), no=len(offline.p4))
        accepted = mapping >= 0
        rows = np.flatnonzero(accepted)
        columns = mapping[accepted]
        native_mapping = np.full(len(mapping), -1, np.int32)
        selected_qdr = np.full(len(mapping), -1, np.int64)
        selected_response = np.full(len(mapping), -1, np.int64)
        native_mapping[rows] = native[columns].astype(np.int32)
        selected_qdr[rows] = qdr[rows, columns]
        selected_response[rows] = qresponse[rows, columns]
        return PairingResult(
            concatenated_offline_index=mapping,
            native_offline_index=native_mapping,
            pairing_validity=accepted,
            selected_qdr=selected_qdr,
            selected_qabs_log_pt_response=selected_response,
            solver=SOLVER,
        )


__all__ = [
    "FullCardinalityBottleneckMatcher", "PairingResult", "assignment_signature",
    "canonical_qabs_log_pt_response", "canonical_qdr",
    "production_pairing_from_matrices", "reference_pairing_from_matrices",
    "validate_pairing",
]

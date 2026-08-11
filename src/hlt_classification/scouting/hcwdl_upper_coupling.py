"""Label-free residual-shell coupling for the HCWDL U/J homotopy."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    require_sha256, validate_content_hash, with_content_hash,
)

from .hcwdl_homotopy_contracts import (
    COUPLING_HASH_DOMAIN, EDIT_INSERTION, EDIT_REMOVAL, EDIT_SUBSTITUTION,
    SCALE_CALIBRATION_CONTRACT, SWITCH_CALIBRATION_CONTRACT,
    TARGET_ASSIGNED_OFFLINE, TARGET_HLT_DUSTBIN,
)
from .matching import p4_kinematics, physical_p4_mask, wrapped_delta_phi
from .repair import FULL_VALIDITY_GROUPS, transform_endpoint_features


HISTOGRAM_BINS: Final = 65_536
SWITCH_BINS: Final = 4096
COST_QUANTUM: Final = 1_000_000
MASS_QUANTUM: Final = 1_000_000
P4_EPSILON: Final = 1.0e-6
FIELD_CHANNELS: Final = (0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
FIELD_FLOORS: Final = {
    0: .20, 7: .05, 8: .05, 9: .25, 10: .25,
    11: .25, 12: .25, 13: .25, 14: .25, 15: .25,
    16: .25, 17: .25, 18: .25, 19: .25, 20: 1.0,
}
FIELD_GROUPS: Final = {
    "quality": (0,), "relative": (7, 8, 9), "scale": (10, 19),
    "lost": (20,),
}
TRACK_GROUPS: Final = {
    "fit": (11,), "dz": (12, 18), "dxy": (13, 14),
    "btag": (15, 16, 17),
}


@dataclass(frozen=True)
class EndpointRecord:
    """One real source or target endpoint before canonical normalization."""

    raw_features: np.ndarray
    validity: np.ndarray
    p4: np.ndarray
    native_index: int
    hlt_slot: int
    target_kind: int

    def __post_init__(self) -> None:
        features = np.asarray(self.raw_features, np.float64)
        validity = np.asarray(self.validity, np.bool_)
        p4 = np.asarray(self.p4, np.float64)
        if features.shape != (21,) or validity.shape != (21,) or p4.shape != (4,):
            raise ValueError("HCWDL-UJ endpoint record shape differs")
        if not physical_p4_mask(p4[None, :], tolerance=1.0e-5)[0]:
            raise ValueError("HCWDL-UJ endpoint p4 is nonphysical")
        if np.any(validity & ~np.isfinite(features)):
            raise FloatingPointError("valid endpoint field is nonfinite")
        object.__setattr__(self, "raw_features", features)
        object.__setattr__(self, "validity", validity)
        object.__setattr__(self, "p4", p4)

    @property
    def source_key(self) -> int:
        return int(self.native_index)

    @property
    def target_key(self) -> tuple[int, int, int]:
        return int(self.hlt_slot), int(self.target_kind), int(self.native_index)


@dataclass(frozen=True)
class CommonPair:
    source: EndpointRecord
    target: EndpointRecord


@dataclass(frozen=True)
class EndpointPartition:
    p0: tuple[EndpointRecord, ...]
    d100: tuple[EndpointRecord, ...]
    common: tuple[CommonPair, ...]
    source_only: tuple[EndpointRecord, ...]
    target_only: tuple[EndpointRecord, ...]
    raw_hlt_length: int

    @property
    def r_hlt(self) -> tuple[EndpointRecord, ...]:
        return tuple(row for row in self.target_only if row.target_kind == TARGET_HLT_DUSTBIN)

    @property
    def r_off(self) -> tuple[EndpointRecord, ...]:
        return tuple(row for row in self.target_only if row.target_kind == TARGET_ASSIGNED_OFFLINE)


@dataclass(frozen=True)
class ResidualEdit:
    edit_kind: int
    source_native_index: int
    target_hlt_slot: int
    target_kind: int
    target_native_index: int
    cost_q: int
    mass_q: int = 0
    switch_u16: int | None = None

    @property
    def key(self) -> tuple[int, int, int, int, int]:
        return (
            int(self.edit_kind), int(self.source_native_index),
            int(self.target_hlt_slot), int(self.target_kind),
            int(self.target_native_index),
        )


def _validity(raw: np.ndarray) -> np.ndarray:
    value = np.asarray(raw, np.float64)
    return np.isfinite(value) & (np.abs(value) <= 1.0e32)


def build_endpoint_partition(
    *, offline_features: np.ndarray, offline_validity: np.ndarray,
    offline_p4: np.ndarray, charged_count: int, neutral_count: int,
    hlt_features: np.ndarray, hlt_p4: np.ndarray, assignment: np.ndarray,
    raw_hlt_length: int,
) -> EndpointPartition:
    """Build exact typed A/B/K/O/R collections for one selected jet."""

    off_f = np.asarray(offline_features, np.float64)
    off_v = np.asarray(offline_validity, np.bool_)
    off_p4 = np.asarray(offline_p4, np.float64)
    hlt_f = np.asarray(hlt_features, np.float64)
    hlt_p4_value = np.asarray(hlt_p4, np.float64)
    mapping = np.asarray(assignment, np.int64)
    if charged_count < 0 or neutral_count < 0 or charged_count + neutral_count != len(off_f):
        raise ValueError("offline endpoint collection counts differ")
    if off_f.shape != (len(off_f), 21) or off_v.shape != off_f.shape or off_p4.shape != (len(off_f), 4):
        raise ValueError("offline endpoint arrays differ")
    visible = min(int(raw_hlt_length), 200)
    if raw_hlt_length < 0 or hlt_f.shape[0] < visible or hlt_f.shape[1:] != (21,):
        raise ValueError("HLT endpoint feature shape differs")
    if hlt_p4_value.shape[0] < visible or hlt_p4_value.shape[1:] != (4,) or len(mapping) < visible:
        raise ValueError("HLT endpoint p4/assignment shape differs")
    p0_indices = [*range(min(charged_count, 90))]
    p0_indices.extend(
        charged_count + index for index in range(min(neutral_count, 60))
    )
    p0 = tuple(EndpointRecord(
        off_f[index], off_v[index], off_p4[index], index, -1,
        TARGET_ASSIGNED_OFFLINE,
    ) for index in p0_indices)
    p0_by_native = {row.native_index: row for row in p0}
    if len(p0_by_native) != len(p0):
        raise RuntimeError("P0 native identities are not unique")

    nonnegative = mapping[:visible][mapping[:visible] >= 0]
    if len(set(map(int, nonnegative))) != len(nonnegative):
        raise ValueError("imported D100 assignment is not injective")
    required_offline = sorted(set(p0_indices) | set(map(int, nonnegative)))
    if (
        required_offline
        and not physical_p4_mask(off_p4[required_offline], tolerance=1.0e-5).all()
    ) or not physical_p4_mask(hlt_p4_value[:visible], tolerance=1.0e-5).all():
        raise ValueError("HCWDL-UJ visible endpoint contains nonphysical p4")
    d100: list[EndpointRecord] = []
    common: list[CommonPair] = []
    common_native: set[int] = set()
    for slot in range(visible):
        native = int(mapping[slot])
        if native >= len(off_f):
            raise ValueError("D100 assignment references an absent native endpoint")
        if native >= 0:
            target = EndpointRecord(
                off_f[native], off_v[native], off_p4[native], native, slot,
                TARGET_ASSIGNED_OFFLINE,
            )
            if native in p0_by_native:
                source = p0_by_native[native]
                if not (
                    np.array_equal(source.raw_features, target.raw_features)
                    and np.array_equal(source.validity, target.validity)
                    and np.array_equal(source.p4, target.p4)
                ):
                    raise ValueError("K pair endpoint payload differs")
                common.append(CommonPair(source, target)); common_native.add(native)
        else:
            validity = _validity(hlt_f[slot])
            target = EndpointRecord(
                np.where(validity, hlt_f[slot], 0.0), validity,
                hlt_p4_value[slot], -1, slot, TARGET_HLT_DUSTBIN,
            )
        d100.append(target)
    common_slots = {pair.target.hlt_slot for pair in common}
    source_only = tuple(row for row in p0 if row.native_index not in common_native)
    target_only = tuple(row for row in d100 if row.hlt_slot not in common_slots)
    result = EndpointPartition(
        p0=p0, d100=tuple(d100), common=tuple(common),
        source_only=source_only, target_only=target_only,
        raw_hlt_length=int(raw_hlt_length),
    )
    if len(result.common) + len(result.source_only) != len(result.p0):
        raise RuntimeError("P0 partition conservation failed")
    if len(result.common) + len(result.target_only) != len(result.d100):
        raise RuntimeError("D100 partition conservation failed")
    if len(result.common) + max(len(result.source_only), len(result.target_only)) > 200:
        raise ValueError("HCWDL-UJ carrier exceeds 200 visible tokens")
    return result


class ScaleAccumulator:
    """Deterministic fixed-bin train-only Cartesian-edge calibration."""

    def __init__(self) -> None:
        self.delta_r = np.zeros(HISTOGRAM_BINS, np.uint64)
        self.log_pt = np.zeros(HISTOGRAM_BINS, np.uint64)
        self.log_energy = np.zeros(HISTOGRAM_BINS, np.uint64)
        self.fields = {channel: np.zeros(HISTOGRAM_BINS, np.uint64) for channel in FIELD_CHANNELS}
        self.edges = 0
        self.floor_pt = 0
        self.floor_energy = 0

    @staticmethod
    def _bin(value: float, upper: float) -> int:
        if not math.isfinite(value) or value < 0:
            raise FloatingPointError("HCWDL-UJ calibration primitive is invalid")
        return min(HISTOGRAM_BINS - 1, int(math.floor(value / upper * HISTOGRAM_BINS)))

    @staticmethod
    def _kinematics(endpoint: EndpointRecord) -> tuple[float, float, float, float]:
        pt, eta, phi, energy = p4_kinematics(endpoint.p4[None, :])
        return float(pt[0]), float(eta[0]), float(phi[0]), float(energy[0])

    @staticmethod
    def _transformed(endpoint: EndpointRecord) -> np.ndarray:
        return transform_endpoint_features(
            endpoint.raw_features[None, :], endpoint.validity[None, :],
        )[0].astype(np.float64)

    def update(self, source: EndpointRecord, target: EndpointRecord) -> None:
        opt, oeta, ophi, oe = self._kinematics(source)
        rpt, reta, rphi, re = self._kinematics(target)
        self.floor_pt += int(opt <= P4_EPSILON) + int(rpt <= P4_EPSILON)
        self.floor_energy += int(oe <= P4_EPSILON) + int(re <= P4_EPSILON)
        dr = math.hypot(oeta - reta, float(wrapped_delta_phi(ophi, rphi)))
        lp = abs(math.log(max(opt, P4_EPSILON) / max(rpt, P4_EPSILON)))
        le = abs(math.log(max(oe, P4_EPSILON) / max(re, P4_EPSILON)))
        self.delta_r[self._bin(dr, 5.0)] += 1
        self.log_pt[self._bin(lp, 8.0)] += 1
        self.log_energy[self._bin(le, 8.0)] += 1
        left, right = self._transformed(source), self._transformed(target)
        for channel in FIELD_CHANNELS:
            if not (source.validity[channel] and target.validity[channel]):
                continue
            delta = abs(float(left[channel]) - float(right[channel]))
            if channel == 7:
                delta = abs(float(wrapped_delta_phi(left[channel], right[channel])))
            self.fields[channel][self._bin(delta, 64.0)] += 1
        self.edges += 1

    def update_partition(self, partition: EndpointPartition) -> None:
        for source in partition.source_only:
            for target in partition.target_only:
                self.update(source, target)

    @staticmethod
    def _quantile(histogram: np.ndarray, *, upper: float, floor: float) -> float:
        total = int(sum(map(int, histogram)))
        if total == 0:
            return float(floor)
        threshold = (90 * total + 99) // 100
        cumulative = 0
        for index, count in enumerate(histogram):
            cumulative += int(count)
            if cumulative >= threshold:
                return max(float(floor), (index + 1) / HISTOGRAM_BINS * upper)
        raise RuntimeError("calibration histogram quantile is unreachable")

    def payload(self, *, coupling_config_sha256: str, train_identity_sha256: str) -> dict[str, Any]:
        field_scales = {
            str(channel): self._quantile(
                self.fields[channel], upper=64.0, floor=FIELD_FLOORS[channel],
            ) for channel in FIELD_CHANNELS
        }
        return with_content_hash({
            "contract": SCALE_CALIBRATION_CONTRACT,
            "schema_version": 1,
            "coupling_config_sha256": require_sha256(coupling_config_sha256, name="coupling config"),
            "train_identity_sha256": require_sha256(train_identity_sha256, name="train identity"),
            "cartesian_edge_count": int(self.edges),
            "scales": {
                "delta_r": self._quantile(self.delta_r, upper=5.0, floor=.02),
                "log_pt": self._quantile(self.log_pt, upper=8.0, floor=.25),
                "log_energy": self._quantile(self.log_energy, upper=8.0, floor=.25),
                "fields": field_scales,
            },
            "histogram_hashes": {
                "delta_r": hashlib.sha256(self.delta_r.astype("<u8").tobytes()).hexdigest(),
                "log_pt": hashlib.sha256(self.log_pt.astype("<u8").tobytes()).hexdigest(),
                "log_energy": hashlib.sha256(self.log_energy.astype("<u8").tobytes()).hexdigest(),
                "fields": {str(k): hashlib.sha256(v.astype("<u8").tobytes()).hexdigest()
                           for k, v in self.fields.items()},
            },
            "p4_floor_hits": {"pt": self.floor_pt, "energy": self.floor_energy},
            "labels_read": False,
            "final_test_accessed": False,
        })


def validate_scale_calibration(
    value: Mapping[str, Any], *, coupling_config_sha256: str | None = None,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=SCALE_CALIBRATION_CONTRACT,
        expected_schema_version=1,
    )
    config = require_sha256(value.get("coupling_config_sha256"), name="coupling config")
    require_sha256(value.get("train_identity_sha256"), name="train identity")
    if coupling_config_sha256 is not None and config != require_sha256(
        coupling_config_sha256, name="expected coupling config",
    ):
        raise ValueError("coupling scale calibration config differs")
    scales = value.get("scales")
    if not isinstance(scales, Mapping) or set(scales.get("fields", {})) != {
        str(channel) for channel in FIELD_CHANNELS
    }:
        raise ValueError("coupling scale calibration field scales differ")
    numeric = [scales.get("delta_r"), scales.get("log_pt"), scales.get("log_energy"), *scales["fields"].values()]
    if any(not math.isfinite(float(item)) or float(item) <= 0 for item in numeric):
        raise ValueError("coupling scale calibration contains invalid scales")
    if value.get("labels_read") is not False or value.get("final_test_accessed") is not False:
        raise PermissionError("coupling scale calibration access differs")
    return digest


def _category(endpoint: EndpointRecord) -> int:
    flags = endpoint.raw_features[2:7]
    valid = endpoint.validity[2:7]
    if valid.all() and np.all((flags == 0) | (flags == 1)) and float(flags.sum()) == 1.0:
        return int(np.argmax(flags))
    return -1


def _charge(endpoint: EndpointRecord) -> int | None:
    if not endpoint.validity[1]:
        return None
    value = float(endpoint.raw_features[1])
    if math.isfinite(value) and value in {-1.0, 0.0, 1.0}:
        return int(value)
    return None


def _charged_state(endpoint: EndpointRecord) -> str:
    category = _category(endpoint)
    charge = _charge(endpoint)
    if category < 0 or charge is None:
        return "unknown"
    if category < 3:
        return "charged"
    return "not_charged"


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return 0.0 if not items else math.fsum(items) / len(items)


def endpoint_cost(
    source: EndpointRecord, target: EndpointRecord, scales: Mapping[str, Any],
) -> tuple[float, int, dict[str, float]]:
    """Evaluate the exact bounded v1 cost and half-up integer quantum."""

    opt, oeta, ophi, oe = ScaleAccumulator._kinematics(source)
    rpt, reta, rphi, re = ScaleAccumulator._kinematics(target)
    dr = math.hypot(oeta - reta, float(wrapped_delta_phi(ophi, rphi)))
    d_kin = _mean((
        min(dr / float(scales["delta_r"]), 1.0),
        min(abs(math.log(max(opt, P4_EPSILON) / max(rpt, P4_EPSILON))) /
            float(scales["log_pt"]), 1.0),
        min(abs(math.log(max(oe, P4_EPSILON) / max(re, P4_EPSILON))) /
            float(scales["log_energy"]), 1.0),
    ))
    source_charge, target_charge = _charge(source), _charge(target)
    d_id = _mean((
        float(_category(source) != _category(target)),
        float(source_charge != target_charge),
    ))
    d_valid = _mean(
        float(np.any(source.validity[list(channels)] != target.validity[list(channels)]))
        for channels in FULL_VALIDITY_GROUPS.values()
    )
    left = ScaleAccumulator._transformed(source)
    right = ScaleAccumulator._transformed(target)

    def channel_delta(channel: int) -> float | None:
        if not (source.validity[channel] and target.validity[channel]):
            return None
        raw = abs(float(left[channel]) - float(right[channel]))
        if channel == 7:
            raw = abs(float(wrapped_delta_phi(left[channel], right[channel])))
        return min(raw / float(scales["fields"][str(channel)]), 1.0)

    source_app, target_app = _charged_state(source), _charged_state(target)
    if source_app != target_app:
        d_track = 1.0
    elif source_app != "charged":
        d_track = 0.0
    else:
        d_track = _mean(
            _mean(value for channel in channels
                  if (value := channel_delta(channel)) is not None)
            for channels in TRACK_GROUPS.values()
        )
    d_field = _mean(
        _mean(value for channel in channels
              if (value := channel_delta(channel)) is not None)
        for channels in FIELD_GROUPS.values()
    )
    groups = {
        "kinematics": d_kin, "identity": d_id, "validity": d_valid,
        "track": d_track, "field": d_field,
    }
    cost = math.fsum((.30 * d_kin, .20 * d_id, .15 * d_valid,
                      .20 * d_track, .15 * d_field))
    if not math.isfinite(cost) or not 0.0 <= cost <= 1.0 + 1e-12:
        raise FloatingPointError("HCWDL-UJ endpoint cost is outside [0,1]")
    cost = min(max(cost, 0.0), 1.0)
    return cost, int(math.floor(cost * COST_QUANTUM + .5)), groups


def _assignment_optimum(costs: np.ndarray, forced: Sequence[tuple[int, int]] = ()) -> tuple[int, tuple[tuple[int, int], ...]] | None:
    from scipy.optimize import linear_sum_assignment

    matrix = np.asarray(costs, np.int64)
    n_source, n_target = matrix.shape
    k = min(n_source, n_target)
    used_source = {row for row, _ in forced}
    used_target = {column for _, column in forced}
    if len(used_source) != len(forced) or len(used_target) != len(forced):
        return None
    if any(not (0 <= row < n_source and 0 <= column < n_target) for row, column in forced):
        return None
    remaining_source = [row for row in range(n_source) if row not in used_source]
    remaining_target = [column for column in range(n_target) if column not in used_target]
    needed = k - len(forced)
    if needed < 0 or min(len(remaining_source), len(remaining_target)) < needed:
        return None
    selected = list(forced)
    total = sum(int(matrix[row, column]) for row, column in forced)
    if needed:
        sub = matrix[np.ix_(remaining_source, remaining_target)]
        rows, columns = linear_sum_assignment(sub)
        if len(rows) < needed:
            return None
        pairs = sorted(
            ((remaining_source[int(row)], remaining_target[int(column)])
             for row, column in zip(rows, columns, strict=True)),
            key=lambda pair: (int(matrix[pair]), pair),
        )[:needed]
        # For a rectangular assignment scipy returns exactly min dimensions,
        # which is exactly `needed` after any conflict-free forced prefix.
        if len(pairs) != needed:
            return None
        selected.extend(pairs)
        total += sum(int(matrix[pair]) for pair in pairs)
    return total, tuple(sorted(selected))


def lexicographic_minimum_assignment(
    costs: np.ndarray, source_keys: Sequence[int], target_keys: Sequence[tuple[int, int, int]],
) -> tuple[tuple[int, int], ...]:
    """Exact optimum followed by canonical feasibility-based edge fixing."""

    matrix = np.asarray(costs, np.int64)
    if matrix.shape != (len(source_keys), len(target_keys)):
        raise ValueError("residual coupling cost matrix shape differs")
    if matrix.ndim != 2 or np.any(matrix < 0):
        raise ValueError("residual coupling costs must be a nonnegative matrix")
    if not min(matrix.shape, default=0):
        return ()
    optimum = _assignment_optimum(matrix)
    if optimum is None:
        raise RuntimeError("residual coupling assignment is infeasible")
    optimum_total = optimum[0]
    edges = sorted(
        ((row, column) for row in range(matrix.shape[0]) for column in range(matrix.shape[1])),
        key=lambda pair: (int(source_keys[pair[0]]), *target_keys[pair[1]]),
    )
    forced: list[tuple[int, int]] = []
    k = min(matrix.shape)
    for edge in edges:
        if len(forced) == k:
            break
        if any(edge[0] == row or edge[1] == column for row, column in forced):
            continue
        candidate = _assignment_optimum(matrix, (*forced, edge))
        if candidate is not None and candidate[0] == optimum_total:
            forced.append(edge)
    if len(forced) != k or sum(int(matrix[pair]) for pair in forced) != optimum_total:
        raise RuntimeError("canonical residual coupling edge fixing failed")
    return tuple(sorted(forced, key=lambda pair: (source_keys[pair[0]], *target_keys[pair[1]])))


def couple_partition(
    partition: EndpointPartition, scale_calibration: Mapping[str, Any],
) -> tuple[ResidualEdit, ...]:
    scales = scale_calibration.get("scales")
    if not isinstance(scales, Mapping):
        raise ValueError("residual coupling scale calibration differs")
    source = tuple(sorted(partition.source_only, key=lambda row: row.source_key))
    target = tuple(sorted(partition.target_only, key=lambda row: row.target_key))
    costs = np.empty((len(source), len(target)), np.int64)
    for i, left in enumerate(source):
        for j, right in enumerate(target):
            costs[i, j] = endpoint_cost(left, right, scales)[1]
    selected = lexicographic_minimum_assignment(
        costs, [row.source_key for row in source], [row.target_key for row in target],
    ) if len(source) and len(target) else ()
    used_source = {i for i, _ in selected}; used_target = {j for _, j in selected}
    edits = [ResidualEdit(
        EDIT_SUBSTITUTION, source[i].native_index, target[j].hlt_slot,
        target[j].target_kind, target[j].native_index, int(costs[i, j]),
    ) for i, j in selected]
    edits.extend(ResidualEdit(
        EDIT_REMOVAL, row.native_index, 65535, TARGET_ASSIGNED_OFFLINE, -1,
        COST_QUANTUM,
    ) for i, row in enumerate(source) if i not in used_source)
    edits.extend(ResidualEdit(
        EDIT_INSERTION, -1, row.hlt_slot, row.target_kind, row.native_index,
        COST_QUANTUM,
    ) for j, row in enumerate(target) if j not in used_target)
    edits.sort(key=lambda row: row.key)
    if len(edits) != max(len(source), len(target)):
        raise RuntimeError("residual edit count differs from cardinality equation")
    return tuple(edits)


def assign_edit_masses(
    edits: Sequence[ResidualEdit], partition: EndpointPartition,
) -> tuple[ResidualEdit, ...]:
    source = {row.native_index: row for row in partition.source_only}
    target = {row.hlt_slot: row for row in partition.target_only}
    ordered = sorted(edits, key=lambda row: row.key)
    pt_terms: list[float] = []
    energy_terms: list[float] = []
    for edit in ordered:
        left = source.get(edit.source_native_index)
        right = target.get(edit.target_hlt_slot)
        lpt, _, _, le = (0.0, 0.0, 0.0, 0.0) if left is None else ScaleAccumulator._kinematics(left)
        rpt, _, _, re = (0.0, 0.0, 0.0, 0.0) if right is None else ScaleAccumulator._kinematics(right)
        pt_terms.append(max(lpt, rpt)); energy_terms.append(max(le, re))
    pt_total, energy_total = math.fsum(pt_terms), math.fsum(energy_terms)
    result = []
    for edit, pt, energy in zip(ordered, pt_terms, energy_terms, strict=True):
        mass = math.fsum((
            1.0,
            0.0 if pt_total == 0 else 4.0 * pt / pt_total,
            0.0 if energy_total == 0 else 2.0 * energy / energy_total,
            2.0 * edit.cost_q / COST_QUANTUM,
        ))
        mass_q = int(math.floor(mass * MASS_QUANTUM + .5))
        if not 0 < mass_q <= np.iinfo(np.uint32).max:
            raise OverflowError("residual edit mass is outside uint32")
        result.append(ResidualEdit(
            edit.edit_kind, edit.source_native_index, edit.target_hlt_slot,
            edit.target_kind, edit.target_native_index, edit.cost_q, mass_q,
            edit.switch_u16,
        ))
    return tuple(result)


def build_switch_calibration(
    edits: Iterable[ResidualEdit], *, coupling_config_sha256: str,
    train_base_manifest_sha256: str,
) -> dict[str, Any]:
    bins = [0] * SWITCH_BINS
    count = 0
    for edit in edits:
        if edit.mass_q <= 0 or not 0 <= edit.cost_q <= COST_QUANTUM:
            raise ValueError("switch calibration edit quantum differs")
        index = min(SWITCH_BINS - 1, edit.cost_q * SWITCH_BINS // (COST_QUANTUM + 1))
        bins[index] += int(edit.mass_q); count += 1
        if bins[index] > np.iinfo(np.uint64).max:
            raise OverflowError("switch calibration mass overflow")
    total = sum(bins)
    if total > np.iinfo(np.uint64).max:
        raise OverflowError("switch calibration total mass overflow")
    return with_content_hash({
        "contract": SWITCH_CALIBRATION_CONTRACT,
        "schema_version": 1,
        "coupling_config_sha256": require_sha256(coupling_config_sha256, name="coupling config"),
        "train_base_manifest_sha256": require_sha256(train_base_manifest_sha256, name="train base manifest"),
        "edit_count": count,
        "bin_mass_q": bins,
        "total_mass_q": total,
        "degenerate": total == 0,
        "final_test_accessed": False,
    })


def validate_switch_calibration(
    value: Mapping[str, Any], *, coupling_config_sha256: str | None = None,
    train_base_manifest_sha256: str | None = None,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=SWITCH_CALIBRATION_CONTRACT,
        expected_schema_version=1,
    )
    config = require_sha256(value.get("coupling_config_sha256"), name="coupling config")
    base = require_sha256(value.get("train_base_manifest_sha256"), name="train base manifest")
    bins = value.get("bin_mass_q")
    if not isinstance(bins, list) or len(bins) != SWITCH_BINS or any(
        not isinstance(item, int) or item < 0 for item in bins
    ):
        raise ValueError("coupling switch histogram differs")
    if int(value.get("total_mass_q", -1)) != sum(bins):
        raise ValueError("coupling switch total mass differs")
    if bool(value.get("degenerate")) != (sum(bins) == 0):
        raise ValueError("coupling switch degeneracy differs")
    if coupling_config_sha256 is not None and config != coupling_config_sha256:
        raise ValueError("coupling switch config differs")
    if train_base_manifest_sha256 is not None and base != train_base_manifest_sha256:
        raise ValueError("coupling switch train-base lineage differs")
    if value.get("final_test_accessed") is not False:
        raise PermissionError("coupling switch calibration accessed final test")
    return digest


def switch_coordinate(
    edit: ResidualEdit, *, identity_key: str,
    coupling_config_sha256: str, calibration: Mapping[str, Any],
) -> int:
    bins = [int(value) for value in calibration.get("bin_mass_q", ())]
    total = int(calibration.get("total_mass_q", -1))
    if len(bins) != SWITCH_BINS or total != sum(bins) or total < 0:
        raise ValueError("switch calibration histogram differs")
    if total == 0:
        return (int(edit.cost_q) * 65535 + 500_000) // 1_000_000
    index = min(SWITCH_BINS - 1, int(edit.cost_q) * SWITCH_BINS // 1_000_001)
    cumulative = sum(bins[:index])
    fields = (
        COUPLING_HASH_DOMAIN.encode("utf-8"),
        require_sha256(coupling_config_sha256, name="coupling config").encode("ascii"),
        str(identity_key).encode("utf-8"),
        int(edit.edit_kind).to_bytes(1, "little", signed=False),
        int(edit.source_native_index).to_bytes(4, "little", signed=True),
        int(edit.target_hlt_slot).to_bytes(2, "little", signed=False),
        int(edit.target_kind).to_bytes(1, "little", signed=False),
        int(edit.target_native_index).to_bytes(4, "little", signed=True),
    )
    encoded = b"".join(len(field).to_bytes(4, "little") + field for field in fields)
    digest = hashlib.sha256(encoded).digest()
    h = int.from_bytes(digest[:8], "big")
    H = 1 << 64
    numerator = 2 * H * cumulative + (2 * h + 1) * bins[index]
    denominator = 2 * H * total
    return min(65535, (numerator * 65535 + denominator // 2) // denominator)


def attach_switches(
    edits: Sequence[ResidualEdit], *, identity_key: str,
    coupling_config_sha256: str, calibration: Mapping[str, Any],
) -> tuple[ResidualEdit, ...]:
    return tuple(ResidualEdit(
        edit.edit_kind, edit.source_native_index, edit.target_hlt_slot,
        edit.target_kind, edit.target_native_index, edit.cost_q, edit.mass_q,
        switch_coordinate(
            edit, identity_key=identity_key,
            coupling_config_sha256=coupling_config_sha256,
            calibration=calibration,
        ),
    ) for edit in edits)


def edit_is_active(edit: ResidualEdit, *, numerator: int, denominator: int) -> bool:
    if edit.switch_u16 is None or denominator <= 0 or not 0 <= numerator <= denominator:
        raise ValueError("structural switch request differs")
    if numerator == 0:
        return False
    if numerator == denominator:
        return True
    threshold = (2 * numerator * 65535 + denominator) // (2 * denominator)
    return int(edit.switch_u16) <= threshold


__all__ = [
    "COST_QUANTUM", "EndpointPartition", "EndpointRecord", "CommonPair",
    "ResidualEdit", "ScaleAccumulator", "assign_edit_masses",
    "attach_switches", "build_endpoint_partition", "build_switch_calibration",
    "couple_partition", "edit_is_active", "endpoint_cost",
    "lexicographic_minimum_assignment", "switch_coordinate",
    "validate_scale_calibration", "validate_switch_calibration",
]

"""Scientific kernels for the HCWDL unified-root balanced homotopy."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Final, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import require_sha256

from .hcwdl_homotopy_contracts import (
    EDIT_INSERTION, EDIT_REMOVAL, EDIT_SUBSTITUTION,
)
from .hcwdl_upper_coupling import EndpointPartition, EndpointRecord, ResidualEdit
from .repair import FULL_VALIDITY_GROUPS


BALANCED_SWITCH_CONTRACT: Final = "HCWDL_BALANCED_STRUCTURAL_SWITCH_CONFIG/v1"
BALANCED_ORDER_DOMAIN: Final = "HCWDL-UB/v1/balanced-order"
BALANCED_PHASE_DOMAIN: Final = "HCWDL-UB/v1/balanced-phase"
UINT64_RANGE: Final = 1 << 64

CATEGORY_DUMMY: Final = -2
CATEGORY_UNKNOWN: Final = -1
APPLICABILITY_DUMMY: Final = 0
APPLICABILITY_UNKNOWN: Final = 1
APPLICABILITY_NONCHARGED: Final = 2
APPLICABILITY_CHARGED: Final = 3


@dataclass(frozen=True, order=True)
class BalanceStratum:
    """One label-free operational stratum used to spread atomic U edits."""

    edit_kind: int
    source_category: int
    target_category: int
    charged_applicability_state: int
    validity_change_mask: int

    def __post_init__(self) -> None:
        if self.edit_kind not in {EDIT_SUBSTITUTION, EDIT_REMOVAL, EDIT_INSERTION}:
            raise ValueError("balanced structural edit kind differs")
        if self.source_category not in {-2, -1, 0, 1, 2, 3, 4}:
            raise ValueError("balanced source category differs")
        if self.target_category not in {-2, -1, 0, 1, 2, 3, 4}:
            raise ValueError("balanced target category differs")
        if not 0 <= self.charged_applicability_state < 16:
            raise ValueError("balanced charged-applicability state differs")
        if not 0 <= self.validity_change_mask < 256:
            raise ValueError("balanced validity-change mask differs")

    @property
    def key(self) -> tuple[int, int, int, int, int]:
        return (
            self.edit_kind, self.source_category, self.target_category,
            self.charged_applicability_state, self.validity_change_mask,
        )

    def bytes(self) -> bytes:
        return bytes((
            self.edit_kind,
            self.source_category + 2,
            self.target_category + 2,
            self.charged_applicability_state,
            self.validity_change_mask,
        ))


@dataclass(frozen=True)
class BalancedPlacement:
    edit: ResidualEdit
    stratum: BalanceStratum
    order_sha256: str
    phase_u64: int
    preceding_mass_q: int
    stratum_mass_q: int
    switch_u16: int


def _category(endpoint: EndpointRecord | None) -> int:
    if endpoint is None:
        return CATEGORY_DUMMY
    flags = np.asarray(endpoint.raw_features[2:7], np.float64)
    valid = np.asarray(endpoint.validity[2:7], np.bool_)
    if not valid.all() or not np.all((flags == 0) | (flags == 1)) or flags.sum() != 1:
        return CATEGORY_UNKNOWN
    return int(np.argmax(flags))


def _applicability(endpoint: EndpointRecord | None) -> int:
    category = _category(endpoint)
    if category == CATEGORY_DUMMY:
        return APPLICABILITY_DUMMY
    if category == CATEGORY_UNKNOWN:
        return APPLICABILITY_UNKNOWN
    return APPLICABILITY_CHARGED if category < 3 else APPLICABILITY_NONCHARGED


def _validity_mask(
    source: EndpointRecord | None, target: EndpointRecord | None,
) -> int:
    left = np.zeros(21, np.bool_) if source is None else source.validity
    right = np.zeros(21, np.bool_) if target is None else target.validity
    mask = 0
    for bit, channels in enumerate(FULL_VALIDITY_GROUPS.values()):
        if np.any(left[list(channels)] != right[list(channels)]):
            mask |= 1 << bit
    return mask


def balance_stratum(
    edit: ResidualEdit, partition: EndpointPartition,
) -> BalanceStratum:
    source_by_native = {row.native_index: row for row in partition.source_only}
    target_by_slot = {row.hlt_slot: row for row in partition.target_only}
    source = None if edit.edit_kind == EDIT_INSERTION else source_by_native.get(
        edit.source_native_index,
    )
    target = None if edit.edit_kind == EDIT_REMOVAL else target_by_slot.get(
        edit.target_hlt_slot,
    )
    if edit.edit_kind != EDIT_INSERTION and source is None:
        raise ValueError("balanced edit source is absent from its endpoint partition")
    if edit.edit_kind != EDIT_REMOVAL and target is None:
        raise ValueError("balanced edit target is absent from its endpoint partition")
    applicability = 4 * _applicability(source) + _applicability(target)
    return BalanceStratum(
        edit_kind=edit.edit_kind,
        source_category=_category(source), target_category=_category(target),
        charged_applicability_state=applicability,
        validity_change_mask=_validity_mask(source, target),
    )


def _framed_hash(domain: str, fields: Sequence[bytes]) -> bytes:
    payload = [domain.encode("utf-8")]
    payload.extend(fields)
    encoded = b"".join(len(field).to_bytes(4, "little") + field for field in payload)
    return hashlib.sha256(encoded).digest()


def _edit_key_bytes(edit: ResidualEdit) -> bytes:
    return b"".join((
        int(edit.edit_kind).to_bytes(1, "little", signed=False),
        int(edit.source_native_index).to_bytes(4, "little", signed=True),
        int(edit.target_hlt_slot).to_bytes(2, "little", signed=False),
        int(edit.target_kind).to_bytes(1, "little", signed=False),
        int(edit.target_native_index).to_bytes(4, "little", signed=True),
    ))


def _round_rational_to_u16(numerator: int, denominator: int) -> int:
    if denominator <= 0 or not 0 <= numerator < denominator:
        raise ValueError("balanced circular coordinate is outside [0,1)")
    value = (2 * numerator * 65535 + denominator) // (2 * denominator)
    if not 0 <= value <= 65535:
        raise RuntimeError("balanced uint16 rounding escaped its range")
    return value


def balanced_switch_placements(
    edits: Sequence[ResidualEdit], *, partition: EndpointPartition,
    identity_key: str, switch_config_sha256: str,
) -> tuple[BalancedPlacement, ...]:
    """Assign immutable mass-balanced circular coordinates within every stratum."""

    config_hash = require_sha256(switch_config_sha256, name="balanced switch config")
    if not identity_key:
        raise ValueError("balanced switch requires a canonical jet identity")
    if len({edit.key for edit in edits}) != len(edits):
        raise ValueError("balanced switch edit keys are not unique")
    if any(edit.mass_q <= 0 for edit in edits):
        raise ValueError("balanced switch requires positive integer edit masses")

    grouped: dict[BalanceStratum, list[tuple[bytes, ResidualEdit]]] = {}
    for edit in edits:
        stratum = balance_stratum(edit, partition)
        order_digest = _framed_hash(
            BALANCED_ORDER_DOMAIN,
            (config_hash.encode("ascii"), str(identity_key).encode("utf-8"),
             stratum.bytes(), _edit_key_bytes(edit)),
        )
        grouped.setdefault(stratum, []).append((order_digest, edit))

    placements: list[BalancedPlacement] = []
    for stratum in sorted(grouped):
        rows = sorted(grouped[stratum], key=lambda row: (row[0], row[1].key))
        total_mass = sum(int(edit.mass_q) for _, edit in rows)
        if total_mass <= 0:
            raise RuntimeError("balanced stratum has no positive mass")
        phase_digest = _framed_hash(
            BALANCED_PHASE_DOMAIN,
            (config_hash.encode("ascii"), str(identity_key).encode("utf-8"),
             stratum.bytes()),
        )
        phase_u64 = int.from_bytes(phase_digest[:8], "big")
        preceding = 0
        denominator = 2 * UINT64_RANGE * total_mass
        for order_digest, edit in rows:
            mass = int(edit.mass_q)
            numerator = (
                phase_u64 * 2 * total_mass
                + (2 * preceding + mass) * UINT64_RANGE
            ) % denominator
            switch = _round_rational_to_u16(numerator, denominator)
            placements.append(BalancedPlacement(
                edit=replace(edit, switch_u16=switch), stratum=stratum,
                order_sha256=order_digest.hex(), phase_u64=phase_u64,
                preceding_mass_q=preceding, stratum_mass_q=total_mass,
                switch_u16=switch,
            ))
            preceding += mass
        if preceding != total_mass:
            raise RuntimeError("balanced stratum mass conservation differs")
    placements.sort(key=lambda row: row.edit.key)
    return tuple(placements)


def attach_balanced_switches(
    edits: Sequence[ResidualEdit], *, partition: EndpointPartition,
    identity_key: str, switch_config_sha256: str,
) -> tuple[ResidualEdit, ...]:
    return tuple(
        row.edit for row in balanced_switch_placements(
            edits, partition=partition, identity_key=identity_key,
            switch_config_sha256=switch_config_sha256,
        )
    )


def balanced_edit_is_active(
    edit: ResidualEdit, *, numerator: int, denominator: int,
) -> bool:
    """Sample one frozen coordinate on any rational rung grid."""

    if edit.switch_u16 is None or denominator <= 0 or not 0 <= numerator <= denominator:
        raise ValueError("balanced structural switch request differs")
    if numerator == 0:
        return False
    if numerator == denominator:
        return True
    threshold = (2 * numerator * 65535 + denominator) // (2 * denominator)
    return int(edit.switch_u16) <= threshold


__all__ = [
    "APPLICABILITY_CHARGED", "APPLICABILITY_DUMMY",
    "APPLICABILITY_NONCHARGED", "APPLICABILITY_UNKNOWN",
    "BALANCED_SWITCH_CONTRACT", "BalanceStratum", "BalancedPlacement",
    "attach_balanced_switches", "balance_stratum", "balanced_edit_is_active",
    "balanced_switch_placements",
]

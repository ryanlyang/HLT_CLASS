"""Generalized variable-support HCWDL view V(s,f) with exact endpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Callable, Final

import numpy as np

from .hcwdl_homotopy_contracts import (
    EDIT_INSERTION, EDIT_REMOVAL, EDIT_SUBSTITUTION, TARGET_HLT_DUSTBIN,
)
from .hcwdl_upper_coupling import (
    EndpointPartition, ResidualEdit, build_endpoint_partition, edit_is_active,
)
from .hcwdl_unified_balanced import balanced_edit_is_active
from .inputs import ParticleInputs, build_hlt_inputs
from .hcwdl_representation_data import (
    CHARGED_FAMILY, DIRECT_CHARGED_REASON, DIRECT_NEUTRAL_REASON,
    HCWDLParticleInputs, HCWDLTokenMetadata, NEUTRAL_FAMILY,
    PADDED_FAMILY, PADDED_REASON, attach_hcwdl_token_metadata,
    derive_hcwdl_token_metadata,
)
from .repair import (
    HIGHCOV_SHELL_EXACT_FAMILY, _combined_endpoint_features,
    build_alpha_repaired_inputs, build_uniform_shell_exact_inputs,
    full_endpoint_required_branches,
    transform_endpoint_features,
)
from .schema import HLT_FEATURE_SPECS, HLT_VECTOR_BRANCHES


HOMOTOPY_VIEW_CONTRACT: Final = "HCWDL_STRUCTURAL_FEATURE_VIEW/v1"


@dataclass(frozen=True)
class HomotopyCoordinate:
    structural_numerator: int
    structural_denominator: int
    feature_numerator: int
    feature_denominator: int

    def __post_init__(self) -> None:
        for numerator, denominator in (
            (self.structural_numerator, self.structural_denominator),
            (self.feature_numerator, self.feature_denominator),
        ):
            if denominator <= 0 or not 0 <= numerator <= denominator:
                raise ValueError("HCWDL-UJ coordinate must lie in [0,1]")

    @property
    def s(self) -> float:
        return self.structural_numerator / self.structural_denominator

    @property
    def f(self) -> float:
        return self.feature_numerator / self.feature_denominator

    @property
    def alpha(self) -> float:
        return 1.0 - self.f

    def payload(self) -> dict[str, object]:
        return {
            "structural": [self.structural_numerator, self.structural_denominator],
            "feature": [self.feature_numerator, self.feature_denominator],
            "s_hex": float(self.s).hex(), "f_hex": float(self.f).hex(),
            "alpha_hex": float(self.alpha).hex(),
        }


@dataclass(frozen=True)
class PreparedOfflineEndpoints:
    """One canonical conversion of every offline branch in a source chunk."""

    rows_by_branch: Mapping[str, Sequence[np.ndarray]]
    raw_features: tuple[np.ndarray, ...]
    validity: tuple[np.ndarray, ...]
    p4: tuple[np.ndarray, ...]
    charged_counts: np.ndarray
    neutral_counts: np.ndarray

    @property
    def rows(self) -> int:
        return len(self.raw_features)


@dataclass(frozen=True)
class PreparedHltEndpoints:
    """One canonical conversion of every raw HLT endpoint branch in a chunk."""

    raw_features: tuple[np.ndarray, ...]
    p4: tuple[np.ndarray, ...]
    raw_lengths: np.ndarray

    @property
    def rows(self) -> int:
        return len(self.raw_features)


def _rows(value: object) -> list[np.ndarray]:
    try:
        import awkward as ak
        if isinstance(value, ak.Array):
            return [np.asarray(row) for row in ak.to_list(value)]
    except ImportError:
        pass
    return [np.asarray(row) for row in value]  # type: ignore[arg-type]


def prepare_offline_endpoints(
    arrays: Mapping[str, object],
) -> PreparedOfflineEndpoints:
    """Convert offline ragged branches once, then prepare every row linearly."""

    required = full_endpoint_required_branches()
    missing = sorted(required - set(arrays))
    if missing:
        raise KeyError(f"full offline endpoint is missing branches: {missing}")
    rows_by_branch = {branch: _rows(arrays[branch]) for branch in required}
    rows = len(next(iter(rows_by_branch.values()))) if rows_by_branch else 0
    if rows <= 0 or any(len(values) != rows for values in rows_by_branch.values()):
        raise ValueError("full offline endpoint branch row counts differ")

    count_rows = {
        branch: _rows(arrays[branch])
        for branch in ("n_cpfcands", "n_lts", "n_npfcands")
    }
    if any(len(values) != rows for values in count_rows.values()):
        raise ValueError("offline endpoint count row counts differ")
    raw_features: list[np.ndarray] = []
    validity: list[np.ndarray] = []
    p4_rows: list[np.ndarray] = []
    charged_counts = np.empty(rows, np.int32)
    neutral_counts = np.empty(rows, np.int32)
    for row in range(rows):
        charged = len(rows_by_branch["cpfcandlt_px"][row])
        neutral = len(rows_by_branch["npfcand_px"][row])
        scalar_counts = {}
        for branch, values in count_rows.items():
            value = np.asarray(values[row])
            if (
                value.ndim != 0 or not np.isfinite(value)
                or int(value) != value or int(value) < 0
            ):
                raise ValueError(f"{branch} must be a nonnegative integer scalar")
            scalar_counts[branch] = int(value)
        if scalar_counts["n_cpfcands"] + scalar_counts["n_lts"] != charged:
            raise ValueError("cpfcandlt endpoint collection differs from n_cpfcands + n_lts")
        if scalar_counts["n_npfcands"] != neutral:
            raise ValueError("npfcand endpoint collection differs from n_npfcands")
        charged_counts[row] = charged; neutral_counts[row] = neutral
        features, valid = _combined_endpoint_features(
            rows_by_branch, row=row, charged_count=charged, neutral_count=neutral,
        )
        charged_p4 = np.stack([
            rows_by_branch[f"cpfcandlt_{name}"][row]
            for name in ("px", "py", "pz", "energy")
        ], axis=1).astype(np.float32, copy=False)
        neutral_p4 = np.stack([
            rows_by_branch[f"npfcand_{name}"][row]
            for name in ("px", "py", "pz", "energy")
        ], axis=1).astype(np.float32, copy=False)
        if len(charged_p4) != charged or len(neutral_p4) != neutral:
            raise ValueError("offline endpoint p4 collection lengths differ")
        combined_p4 = np.concatenate((charged_p4, neutral_p4), axis=0).astype(
            np.float64, copy=False,
        )
        if len(combined_p4) != len(features):
            raise ValueError("offline p4 and projected endpoint lengths differ")
        raw_features.append(features); validity.append(valid); p4_rows.append(combined_p4)
    return PreparedOfflineEndpoints(
        rows_by_branch=rows_by_branch,
        raw_features=tuple(raw_features), validity=tuple(validity), p4=tuple(p4_rows),
        charged_counts=charged_counts, neutral_counts=neutral_counts,
    )


def prepare_hlt_endpoints(arrays: Mapping[str, object]) -> PreparedHltEndpoints:
    """Convert raw HLT ragged branches once for all endpoint partitions."""

    feature_rows = {spec.branch: _rows(arrays[spec.branch]) for spec in HLT_FEATURE_SPECS}
    vector_rows = {branch: _rows(arrays[branch]) for branch in HLT_VECTOR_BRANCHES}
    count_rows = _rows(arrays["n_scoutpfcands"])
    rows = len(count_rows)
    if rows <= 0 or any(
        len(values) != rows for values in (*feature_rows.values(), *vector_rows.values())
    ):
        raise ValueError("HLT endpoint branch row counts differ")
    raw_features: list[np.ndarray] = []
    p4_rows: list[np.ndarray] = []
    raw_lengths = np.empty(rows, np.int32)
    for row in range(rows):
        features = np.stack([
            feature_rows[spec.branch][row] for spec in HLT_FEATURE_SPECS
        ], axis=1).astype(np.float64, copy=False)
        vectors = np.stack([
            vector_rows[branch][row] for branch in HLT_VECTOR_BRANCHES
        ], axis=1).astype(np.float64, copy=False)
        count_value = np.asarray(count_rows[row])
        if (
            count_value.ndim != 0 or not np.isfinite(count_value)
            or int(count_value) != count_value or int(count_value) < 0
        ):
            raise ValueError("n_scoutpfcands must be a nonnegative integer scalar")
        count = int(count_value)
        if len(features) != len(vectors) or len(features) != count:
            raise ValueError("HLT endpoint collection differs from n_scoutpfcands")
        raw_features.append(features); p4_rows.append(vectors); raw_lengths[row] = count
    return PreparedHltEndpoints(
        raw_features=tuple(raw_features), p4=tuple(p4_rows), raw_lengths=raw_lengths,
    )


def _partition_for_row(
    *, row: int, assignment: np.ndarray,
    offline: PreparedOfflineEndpoints, hlt: PreparedHltEndpoints,
) -> tuple[EndpointPartition, np.ndarray, np.ndarray]:
    charged = int(offline.charged_counts[row])
    neutral = int(offline.neutral_counts[row])
    offline_features = offline.raw_features[row]
    offline_validity = offline.validity[row]
    offline_p4 = offline.p4[row]
    hlt_features = hlt.raw_features[row]
    hlt_p4 = hlt.p4[row]
    raw_hlt = int(hlt.raw_lengths[row])
    partition = build_endpoint_partition(
        offline_features=offline_features, offline_validity=offline_validity,
        offline_p4=offline_p4, charged_count=charged, neutral_count=neutral,
        hlt_features=hlt_features[:200], hlt_p4=hlt_p4[:200],
        assignment=assignment, raw_hlt_length=raw_hlt,
    )
    return partition, transform_endpoint_features(offline_features, offline_validity), offline_p4


def build_partition_from_arrays(
    arrays: Mapping[str, object], *, row: int, assignment: np.ndarray,
    prepared_offline: PreparedOfflineEndpoints | None = None,
    prepared_hlt: PreparedHltEndpoints | None = None,
) -> EndpointPartition:
    """Public one-row endpoint partition used by coupling production/audits."""

    return _partition_for_row(
        row=row, assignment=assignment,
        offline=(
            prepare_offline_endpoints(arrays)
            if prepared_offline is None else prepared_offline
        ),
        hlt=prepare_hlt_endpoints(arrays) if prepared_hlt is None else prepared_hlt,
    )[0]


def build_p0_inputs(
    arrays: Mapping[str, object], *, prepared: PreparedOfflineEndpoints | None = None,
) -> ParticleInputs:
    """Build the independent projected-native P0 endpoint in native order."""

    prepared = prepare_offline_endpoints(arrays) if prepared is None else prepared
    rows = prepared.rows
    features = np.zeros((rows, 21, 200), np.float32)
    vectors = np.zeros((rows, 4, 200), np.float32)
    mask = np.zeros((rows, 1, 200), np.bool_)
    lengths = np.zeros(rows, np.int32)
    for row in range(rows):
        charged = int(prepared.charged_counts[row])
        neutral = int(prepared.neutral_counts[row])
        indices = [*range(min(charged, 90))]
        indices.extend(charged + index for index in range(min(neutral, 60)))
        projected = transform_endpoint_features(
            prepared.raw_features[row], prepared.validity[row],
        )
        length = len(indices)
        if length > 150:
            raise RuntimeError("P0 visible population exceeds the 90/60 bound")
        if length:
            features[row, :, :length] = projected[indices].T
            vectors[row, :, :length] = np.asarray(prepared.p4[row], np.float32)[indices].T
            mask[row, 0, :length] = True
        lengths[row] = length
    return ParticleInputs(features, vectors, mask, lengths)


def _validate_edit_partition(partition: EndpointPartition, edits: Sequence[ResidualEdit]) -> None:
    source_expected = {row.native_index for row in partition.source_only}
    target_expected = {row.hlt_slot for row in partition.target_only}
    source_seen: set[int] = set(); target_seen: set[int] = set()
    for edit in edits:
        if edit.edit_kind != EDIT_INSERTION:
            if edit.source_native_index not in source_expected or edit.source_native_index in source_seen:
                raise ValueError("coupling edit source conservation differs")
            source_seen.add(edit.source_native_index)
        if edit.edit_kind != EDIT_REMOVAL:
            if edit.target_hlt_slot not in target_expected or edit.target_hlt_slot in target_seen:
                raise ValueError("coupling edit target conservation differs")
            target_seen.add(edit.target_hlt_slot)
    if source_seen != source_expected or target_seen != target_expected:
        raise ValueError("coupling edit endpoint coverage differs")


def _assemble_support_view(
    *, canonical: ParticleInputs, shell: ParticleInputs,
    mapping: np.ndarray, coupling_rows: Sequence[Sequence[ResidualEdit]],
    prepared_offline: PreparedOfflineEndpoints,
    prepared_hlt: PreparedHltEndpoints, active_edit: Callable[[ResidualEdit], bool],
    contract_label: str,
) -> ParticleInputs:
    features = np.zeros_like(canonical.features)
    vectors = np.zeros_like(canonical.vectors)
    mask = np.zeros_like(canonical.mask)
    lengths = np.zeros_like(canonical.raw_lengths)
    for row in range(len(canonical.raw_lengths)):
        partition, projected, projected_p4 = _partition_for_row(
            row=row, assignment=mapping[row],
            offline=prepared_offline, hlt=prepared_hlt,
        )
        edits = tuple(coupling_rows[row]); _validate_edit_partition(partition, edits)
        common_by_slot = {pair.target.hlt_slot: pair for pair in partition.common}

        # Target HLT slots precede native-index tail slots. This is the frozen
        # carrier order shared by old U/J and HCWDL-UB.
        active: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        for slot in sorted(common_by_slot):
            active.append((0, slot, shell.features[row, :, slot], shell.vectors[row, :, slot]))
        for edit in edits:
            switched = active_edit(edit)
            if edit.edit_kind == EDIT_SUBSTITUTION:
                if switched:
                    slot = edit.target_hlt_slot
                    active.append((0, slot, shell.features[row, :, slot], shell.vectors[row, :, slot]))
                else:
                    native = edit.source_native_index
                    active.append((0, edit.target_hlt_slot, projected[native],
                                   projected_p4[native].astype(np.float32)))
            elif edit.edit_kind == EDIT_INSERTION:
                if switched:
                    slot = edit.target_hlt_slot
                    active.append((0, slot, shell.features[row, :, slot], shell.vectors[row, :, slot]))
            elif edit.edit_kind == EDIT_REMOVAL:
                if not switched:
                    native = edit.source_native_index
                    active.append((1, native, projected[native],
                                   projected_p4[native].astype(np.float32)))
            else:
                raise ValueError("unknown coupling edit kind")
        active.sort(key=lambda item: (item[0], item[1]))
        if len(active) != len({(kind, key) for kind, key, _, _ in active}):
            raise ValueError(f"{contract_label} carrier contains duplicate logical slots")
        if len(active) > 200:
            raise ValueError(f"{contract_label} carrier requires hidden truncation")
        lengths[row] = len(active)
        mask[row, 0, :len(active)] = True
        for index, (_, _, feature, vector) in enumerate(active):
            features[row, :, index] = feature
            vectors[row, :, index] = vector
    if not np.isfinite(features).all() or not np.isfinite(vectors).all():
        raise FloatingPointError(f"{contract_label} active view became nonfinite")
    if np.any(features[~np.repeat(mask, features.shape[1], axis=1)]) or np.any(
        vectors[~np.repeat(mask, vectors.shape[1], axis=1)]
    ):
        raise RuntimeError(f"{contract_label} view has nonzero padding")
    return ParticleInputs(features, vectors, mask, lengths)


def build_homotopy_inputs(
    arrays: Mapping[str, object], *, assignments: np.ndarray,
    confidence: np.ndarray, coupling_rows: Sequence[Sequence[ResidualEdit]],
    coordinate: HomotopyCoordinate, identity_keys: Sequence[str],
    discrete_seed: int, prepared_offline: PreparedOfflineEndpoints | None = None,
    prepared_hlt: PreparedHltEndpoints | None = None,
) -> ParticleInputs:
    """Build canonical V(s,f) once without persisting repaired particle data."""

    canonical = build_hlt_inputs(arrays)
    mapping = np.asarray(assignments)
    weights = np.asarray(confidence, np.float32)
    rows = len(canonical.raw_lengths)
    if mapping.shape != (rows, 200) or weights.shape != mapping.shape:
        raise ValueError("HCWDL-UJ assignment/confidence shape differs")
    if len(coupling_rows) != rows or len(identity_keys) != rows or len(set(map(str, identity_keys))) != rows:
        raise ValueError("HCWDL-UJ coupling/identity rows differ")
    if coordinate.s == 0.0 and coordinate.f == 0.0:
        prepared_offline = (
            prepare_offline_endpoints(arrays) if prepared_offline is None else prepared_offline
        )
        return build_p0_inputs(arrays, prepared=prepared_offline)
    if coordinate.s == 1.0 and coordinate.f == 1.0:
        return canonical
    prepared_offline = (
        prepare_offline_endpoints(arrays) if prepared_offline is None else prepared_offline
    )
    if prepared_offline.rows != rows:
        raise ValueError("prepared offline endpoint row count differs")
    offline_p4 = prepared_offline.p4
    shell = build_alpha_repaired_inputs(
        arrays, offline_p4, mapping, alpha=coordinate.alpha,
        repair_family=HIGHCOV_SHELL_EXACT_FAMILY,
        confidence_weights=weights, offline_arrays=arrays,
        identity_keys=identity_keys, discrete_seed=discrete_seed,
    )
    # The exact s=1 branch delegates all tensor assembly and raw-length metadata
    # to the public Shell Exact implementation.
    if coordinate.s == 1.0:
        return shell

    prepared_hlt = prepare_hlt_endpoints(arrays) if prepared_hlt is None else prepared_hlt
    if prepared_hlt.rows != rows:
        raise ValueError("prepared HLT endpoint row count differs")

    return _assemble_support_view(
        canonical=canonical, shell=shell, mapping=mapping,
        coupling_rows=coupling_rows, prepared_offline=prepared_offline,
        prepared_hlt=prepared_hlt,
        active_edit=lambda edit: edit_is_active(
            edit, numerator=coordinate.structural_numerator,
            denominator=coordinate.structural_denominator,
        ),
        contract_label="HCWDL-UJ",
    )


def _attach_balanced_training_metadata(
    view: ParticleInputs, *, arrays: Mapping[str, object], mapping: np.ndarray,
    coupling_rows: Sequence[Sequence[ResidualEdit]],
    coordinate: HomotopyCoordinate,
    prepared_offline: PreparedOfflineEndpoints,
    prepared_hlt: PreparedHltEndpoints | None,
) -> HCWDLParticleInputs:
    """Attach nondeployable token IDs/families in the exact carrier order."""

    rows, _, tokens = view.features.shape
    visible_ids = np.full((rows, tokens), -1, np.int64)
    families = np.full((rows, tokens), PADDED_FAMILY, np.int8)
    reasons = np.full((rows, tokens), PADDED_REASON, np.int8)
    if coordinate.structural_numerator == 0 and coordinate.feature_numerator == 0:
        for row in range(rows):
            charged = int(prepared_offline.charged_counts[row])
            neutral = int(prepared_offline.neutral_counts[row])
            native = [*range(min(charged, 90))]
            native.extend(charged + index for index in range(min(neutral, 60)))
            count = len(native)
            visible_ids[row, :count] = 200 + np.asarray(native, np.int64)
            charged_mask = np.asarray(native) < charged
            families[row, :count] = np.where(
                charged_mask, CHARGED_FAMILY, NEUTRAL_FAMILY,
            )
            reasons[row, :count] = np.where(
                charged_mask, DIRECT_CHARGED_REASON, DIRECT_NEUTRAL_REASON,
            )
        return attach_hcwdl_token_metadata(
            view, HCWDLTokenMetadata(visible_ids, families, reasons),
        )

    prepared_hlt = (
        prepare_hlt_endpoints(arrays) if prepared_hlt is None else prepared_hlt
    )
    hlt_metadata = derive_hcwdl_token_metadata(arrays)
    for row in range(rows):
        partition, _, _ = _partition_for_row(
            row=row, assignment=mapping[row], offline=prepared_offline,
            hlt=prepared_hlt,
        )
        charged = int(prepared_offline.charged_counts[row])
        target_by_slot = {target.hlt_slot: target for target in partition.d100}

        def target_metadata(slot: int) -> tuple[int, int]:
            target = target_by_slot[slot]
            if target.native_index >= 0:
                is_charged = target.native_index < charged
                return (
                    int(CHARGED_FAMILY if is_charged else NEUTRAL_FAMILY),
                    int(DIRECT_CHARGED_REASON if is_charged else DIRECT_NEUTRAL_REASON),
                )
            return (
                int(hlt_metadata.family_codes[row, slot]),
                int(hlt_metadata.family_reason_codes[row, slot]),
            )

        if coordinate.structural_numerator == coordinate.structural_denominator:
            count = int(np.asarray(view.mask[row, 0], np.bool_).sum())
            for slot in range(count):
                visible_ids[row, slot] = slot
                families[row, slot], reasons[row, slot] = target_metadata(slot)
            continue

        common_by_slot = {pair.target.hlt_slot: pair for pair in partition.common}
        active: list[tuple[int, int, int, int]] = []
        for slot, pair in common_by_slot.items():
            native = pair.source.native_index
            is_charged = native < charged
            active.append((
                0, slot,
                int(CHARGED_FAMILY if is_charged else NEUTRAL_FAMILY),
                int(DIRECT_CHARGED_REASON if is_charged else DIRECT_NEUTRAL_REASON),
            ))
        for edit in coupling_rows[row]:
            switched = balanced_edit_is_active(
                edit, numerator=coordinate.structural_numerator,
                denominator=coordinate.structural_denominator,
            )
            if edit.edit_kind == EDIT_SUBSTITUTION:
                if switched:
                    family, reason = target_metadata(edit.target_hlt_slot)
                    active.append((0, edit.target_hlt_slot, family, reason))
                else:
                    native = edit.source_native_index
                    is_charged = native < charged
                    active.append((
                        0, edit.target_hlt_slot,
                        int(CHARGED_FAMILY if is_charged else NEUTRAL_FAMILY),
                        int(DIRECT_CHARGED_REASON if is_charged else DIRECT_NEUTRAL_REASON),
                    ))
            elif edit.edit_kind == EDIT_INSERTION and switched:
                family, reason = target_metadata(edit.target_hlt_slot)
                active.append((0, edit.target_hlt_slot, family, reason))
            elif edit.edit_kind == EDIT_REMOVAL and not switched:
                native = edit.source_native_index
                is_charged = native < charged
                active.append((
                    1, native,
                    int(CHARGED_FAMILY if is_charged else NEUTRAL_FAMILY),
                    int(DIRECT_CHARGED_REASON if is_charged else DIRECT_NEUTRAL_REASON),
                ))
        active.sort(key=lambda item: (item[0], item[1]))
        if len(active) != int(np.asarray(view.mask[row, 0], np.bool_).sum()):
            raise ValueError("HCWDL-UB training metadata carrier count differs")
        for index, (kind, key, family, reason) in enumerate(active):
            visible_ids[row, index] = key if kind == 0 else 200 + key
            families[row, index] = family
            reasons[row, index] = reason
    return attach_hcwdl_token_metadata(
        view, HCWDLTokenMetadata(visible_ids, families, reasons),
    )


def _build_unified_balanced_inputs(
    arrays: Mapping[str, object], *, assignments: np.ndarray,
    pairing_provenance: np.ndarray, provenance_kind: str,
    coupling_rows: Sequence[Sequence[ResidualEdit]],
    coordinate: HomotopyCoordinate, identity_keys: Sequence[str],
    discrete_seed: int, prepared_offline: PreparedOfflineEndpoints | None = None,
    prepared_hlt: PreparedHltEndpoints | None = None,
    include_training_metadata: bool = False,
) -> ParticleInputs:
    """Build HCWDL-UB V_UB(s,f) with balanced U and uniform rational D."""

    canonical = build_hlt_inputs(arrays)
    mapping = np.asarray(assignments)
    provenance = np.asarray(pairing_provenance)
    rows = len(canonical.raw_lengths)
    if mapping.shape != (rows, 200) or provenance.shape != mapping.shape:
        raise ValueError(f"HCWDL-UB assignment/{provenance_kind} shape differs")
    if provenance_kind == "correspondence_confidence":
        numeric = np.asarray(provenance, np.float32)
        if not np.isfinite(numeric).all() or np.any((numeric < 0) | (numeric > 1)):
            raise ValueError("HCWDL-UB confidence provenance differs")
    elif provenance_kind == "pairing_validity":
        if provenance.dtype != np.bool_ and not np.all((provenance == 0) | (provenance == 1)):
            raise ValueError("HCWDL-UB pairing validity must be boolean")
        if not np.array_equal(provenance.astype(bool), mapping >= 0):
            raise ValueError("HCWDL-UB pairing validity differs from assignment presence")
    else:
        raise ValueError("HCWDL-UB pairing provenance kind differs")
    if (
        len(coupling_rows) != rows or len(identity_keys) != rows
        or len(set(map(str, identity_keys))) != rows
    ):
        raise ValueError("HCWDL-UB coupling/identity rows differ")
    if coordinate.structural_numerator == 0 and coordinate.feature_numerator == 0:
        prepared_offline = (
            prepare_offline_endpoints(arrays) if prepared_offline is None else prepared_offline
        )
        view = build_p0_inputs(arrays, prepared=prepared_offline)
        return _attach_balanced_training_metadata(
            view, arrays=arrays, mapping=mapping, coupling_rows=coupling_rows,
            coordinate=coordinate, prepared_offline=prepared_offline,
            prepared_hlt=prepared_hlt,
        ) if include_training_metadata else view
    if (
        coordinate.structural_numerator == coordinate.structural_denominator
        and coordinate.feature_numerator == coordinate.feature_denominator
    ):
        return attach_hcwdl_token_metadata(
            canonical, derive_hcwdl_token_metadata(arrays),
        ) if include_training_metadata else canonical
    prepared_offline = (
        prepare_offline_endpoints(arrays) if prepared_offline is None else prepared_offline
    )
    if prepared_offline.rows != rows:
        raise ValueError("prepared offline endpoint row count differs")
    prepared_hlt = prepare_hlt_endpoints(arrays) if prepared_hlt is None else prepared_hlt
    if prepared_hlt.rows != rows:
        raise ValueError("prepared HLT endpoint row count differs")
    shell = build_uniform_shell_exact_inputs(
        arrays, prepared_offline.p4, mapping,
        offline_numerator=(
            coordinate.feature_denominator - coordinate.feature_numerator
        ),
        offline_denominator=coordinate.feature_denominator,
        offline_arrays=arrays, identity_keys=identity_keys,
        discrete_seed=discrete_seed,
        canonical_inputs=canonical,
        prepared_hlt_features=prepared_hlt.raw_features,
        prepared_hlt_p4=prepared_hlt.p4,
        prepared_offline_features=prepared_offline.raw_features,
        prepared_offline_validity=prepared_offline.validity,
        prepared_charged_counts=prepared_offline.charged_counts,
        prepared_neutral_counts=prepared_offline.neutral_counts,
    )
    if coordinate.structural_numerator == coordinate.structural_denominator:
        return _attach_balanced_training_metadata(
            shell, arrays=arrays, mapping=mapping, coupling_rows=coupling_rows,
            coordinate=coordinate, prepared_offline=prepared_offline,
            prepared_hlt=prepared_hlt,
        ) if include_training_metadata else shell
    view = _assemble_support_view(
        canonical=canonical, shell=shell, mapping=mapping,
        coupling_rows=coupling_rows, prepared_offline=prepared_offline,
        prepared_hlt=prepared_hlt,
        active_edit=lambda edit: balanced_edit_is_active(
            edit, numerator=coordinate.structural_numerator,
            denominator=coordinate.structural_denominator,
        ),
        contract_label="HCWDL-UB",
    )
    return _attach_balanced_training_metadata(
        view, arrays=arrays, mapping=mapping, coupling_rows=coupling_rows,
        coordinate=coordinate, prepared_offline=prepared_offline,
        prepared_hlt=prepared_hlt,
    ) if include_training_metadata else view


def build_unified_balanced_inputs(
    arrays: Mapping[str, object], *, assignments: np.ndarray,
    confidence: np.ndarray, coupling_rows: Sequence[Sequence[ResidualEdit]],
    coordinate: HomotopyCoordinate, identity_keys: Sequence[str],
    discrete_seed: int, prepared_offline: PreparedOfflineEndpoints | None = None,
    prepared_hlt: PreparedHltEndpoints | None = None,
    include_training_metadata: bool = False,
) -> ParticleInputs:
    """Build HCWDL-UB using established calibrated-confidence provenance."""

    return _build_unified_balanced_inputs(
        arrays, assignments=assignments, pairing_provenance=confidence,
        provenance_kind="correspondence_confidence", coupling_rows=coupling_rows,
        coordinate=coordinate, identity_keys=identity_keys,
        discrete_seed=discrete_seed, prepared_offline=prepared_offline,
        prepared_hlt=prepared_hlt,
        include_training_metadata=include_training_metadata,
    )


def build_unified_balanced_pairing_inputs(
    arrays: Mapping[str, object], *, assignments: np.ndarray,
    pairing_validity: np.ndarray,
    coupling_rows: Sequence[Sequence[ResidualEdit]],
    coordinate: HomotopyCoordinate, identity_keys: Sequence[str],
    discrete_seed: int, prepared_offline: PreparedOfflineEndpoints | None = None,
    prepared_hlt: PreparedHltEndpoints | None = None,
    include_training_metadata: bool = False,
) -> ParticleInputs:
    """Build HCWDL-UB using neutral validity, never fake match confidence."""

    return _build_unified_balanced_inputs(
        arrays, assignments=assignments, pairing_provenance=pairing_validity,
        provenance_kind="pairing_validity", coupling_rows=coupling_rows,
        coordinate=coordinate, identity_keys=identity_keys,
        discrete_seed=discrete_seed, prepared_offline=prepared_offline,
        prepared_hlt=prepared_hlt,
        include_training_metadata=include_training_metadata,
    )


def particle_inputs_sha256(view: ParticleInputs) -> str:
    digest = hashlib.sha256()
    for name in ("features", "vectors", "mask", "raw_lengths"):
        value = np.ascontiguousarray(getattr(view, name))
        digest.update(name.encode()); digest.update(value.dtype.str.encode())
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def assert_particle_inputs_equal(left: ParticleInputs, right: ParticleInputs, *, endpoint: str) -> None:
    for name in ("features", "vectors", "mask", "raw_lengths"):
        if not np.array_equal(getattr(left, name), getattr(right, name)):
            raise ValueError(f"HCWDL-UJ {endpoint} {name} equality differs")


__all__ = [
    "HOMOTOPY_VIEW_CONTRACT", "HomotopyCoordinate", "assert_particle_inputs_equal",
    "build_homotopy_inputs", "build_unified_balanced_inputs",
    "build_unified_balanced_pairing_inputs",
    "build_p0_inputs", "build_partition_from_arrays",
    "particle_inputs_sha256", "PreparedHltEndpoints", "PreparedOfflineEndpoints",
    "prepare_hlt_endpoints", "prepare_offline_endpoints",
]

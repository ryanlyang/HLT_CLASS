"""Generalized variable-support HCWDL view V(s,f) with exact endpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Final

import numpy as np

from .hcwdl_homotopy_contracts import (
    EDIT_INSERTION, EDIT_REMOVAL, EDIT_SUBSTITUTION, TARGET_HLT_DUSTBIN,
)
from .hcwdl_upper_coupling import (
    EndpointPartition, ResidualEdit, build_endpoint_partition, edit_is_active,
)
from .inputs import ParticleInputs, build_hlt_inputs
from .hcwdl_representation_data import (
    CHARGED_FAMILY, DIRECT_CHARGED_REASON, DIRECT_NEUTRAL_REASON,
    HCWDLParticleInputs, HCWDLTokenMetadata, NEUTRAL_FAMILY,
    PADDED_FAMILY, PADDED_REASON, attach_hcwdl_token_metadata,
    derive_hcwdl_token_metadata,
)
from .repair import (
    HIGHCOV_SHELL_EXACT_FAMILY, _combined_endpoint_features,
    build_alpha_repaired_inputs, full_endpoint_required_branches,
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
    projected_features: tuple[np.ndarray, ...]
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
    projected_features: list[np.ndarray] = []
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
        charged_counts[row] = charged
        neutral_counts[row] = neutral
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
        raw_features.append(features)
        validity.append(valid)
        projected_features.append(transform_endpoint_features(features, valid))
        p4_rows.append(combined_p4)
    return PreparedOfflineEndpoints(
        rows_by_branch=rows_by_branch,
        raw_features=tuple(raw_features),
        validity=tuple(validity),
        projected_features=tuple(projected_features),
        p4=tuple(p4_rows),
        charged_counts=charged_counts,
        neutral_counts=neutral_counts,
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
        raw_features.append(features)
        p4_rows.append(vectors)
        raw_lengths[row] = count
    return PreparedHltEndpoints(
        raw_features=tuple(raw_features),
        p4=tuple(p4_rows),
        raw_lengths=raw_lengths,
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
    return partition, offline.projected_features[row], offline_p4


def build_partition_from_arrays(
    arrays: Mapping[str, object], *, row: int, assignment: np.ndarray,
    prepared_offline: PreparedOfflineEndpoints | None = None,
    prepared_hlt: PreparedHltEndpoints | None = None,
) -> EndpointPartition:
    """Public one-row endpoint partition used by coupling production/audits."""

    return _partition_for_row(
        row=row,
        assignment=assignment,
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
        projected = prepared.projected_features[row]
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


def build_homotopy_inputs(
    arrays: Mapping[str, object], *, assignments: np.ndarray,
    confidence: np.ndarray, coupling_rows: Sequence[Sequence[ResidualEdit]],
    coordinate: HomotopyCoordinate, identity_keys: Sequence[str],
    discrete_seed: int, prepared_offline: PreparedOfflineEndpoints | None = None,
    prepared_hlt: PreparedHltEndpoints | None = None,
    include_training_metadata: bool = False,
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
            prepare_offline_endpoints(arrays)
            if prepared_offline is None else prepared_offline
        )
        p0 = build_p0_inputs(arrays, prepared=prepared_offline)
        if not include_training_metadata:
            return p0
        visible = np.asarray(p0.mask)[:, 0]
        ids = np.full(visible.shape, -1, np.int64)
        family = np.full(visible.shape, PADDED_FAMILY, np.int8)
        reason = np.full(visible.shape, PADDED_REASON, np.int8)
        for row in range(rows):
            charged = int(prepared_offline.charged_counts[row])
            neutral = int(prepared_offline.neutral_counts[row])
            native = [*range(min(charged, 90))]
            native.extend(charged + index for index in range(min(neutral, 60)))
            count = len(native)
            ids[row, :count] = 200 + np.asarray(native, np.int64)
            charged_mask = np.asarray(native) < charged
            family[row, :count] = np.where(
                charged_mask, CHARGED_FAMILY, NEUTRAL_FAMILY,
            )
            reason[row, :count] = np.where(
                charged_mask, DIRECT_CHARGED_REASON, DIRECT_NEUTRAL_REASON,
            )
        return attach_hcwdl_token_metadata(
            p0, HCWDLTokenMetadata(ids, family, reason),
        )
    if coordinate.s == 1.0 and coordinate.f == 1.0:
        if not include_training_metadata:
            return canonical
        return attach_hcwdl_token_metadata(
            canonical, derive_hcwdl_token_metadata(arrays),
        )
    prepared_offline = (
        prepare_offline_endpoints(arrays)
        if prepared_offline is None else prepared_offline
    )
    if prepared_offline.rows != rows:
        raise ValueError("prepared offline endpoint row count differs")
    offline_p4 = prepared_offline.p4
    shell = build_alpha_repaired_inputs(
        arrays, offline_p4, mapping, alpha=coordinate.alpha,
        repair_family=HIGHCOV_SHELL_EXACT_FAMILY,
        confidence_weights=weights, offline_arrays=arrays,
        identity_keys=identity_keys, discrete_seed=discrete_seed,
        offline_endpoint_rows=prepared_offline.rows_by_branch,
        offline_endpoint_features=prepared_offline.raw_features,
        offline_endpoint_validity=prepared_offline.validity,
    )
    # The exact s=1 branch delegates all tensor assembly and raw-length metadata
    # to the public Shell Exact implementation.
    if coordinate.s == 1.0:
        if not include_training_metadata:
            return shell
        ids = np.full((rows, 200), -1, np.int64)
        family = np.full((rows, 200), PADDED_FAMILY, np.int8)
        reason = np.full((rows, 200), PADDED_REASON, np.int8)
        hlt_metadata = derive_hcwdl_token_metadata(arrays)
        prepared_hlt = (
            prepare_hlt_endpoints(arrays) if prepared_hlt is None else prepared_hlt
        )
        if prepared_hlt.rows != rows:
            raise ValueError("prepared HLT endpoint row count differs")
        for row in range(rows):
            partition, _, _ = _partition_for_row(
                row=row, assignment=mapping[row],
                offline=prepared_offline, hlt=prepared_hlt,
            )
            charged_count = int(prepared_offline.charged_counts[row])
            for target in partition.d100:
                slot = target.hlt_slot
                ids[row, slot] = slot
                if target.native_index >= 0:
                    charged = target.native_index < charged_count
                    family[row, slot] = CHARGED_FAMILY if charged else NEUTRAL_FAMILY
                    reason[row, slot] = (
                        DIRECT_CHARGED_REASON if charged else DIRECT_NEUTRAL_REASON
                    )
                else:
                    family[row, slot] = hlt_metadata.family_codes[row, slot]
                    reason[row, slot] = hlt_metadata.family_reason_codes[row, slot]
        return attach_hcwdl_token_metadata(
            shell, HCWDLTokenMetadata(ids, family, reason),
        )

    prepared_hlt = prepare_hlt_endpoints(arrays) if prepared_hlt is None else prepared_hlt
    if prepared_hlt.rows != rows:
        raise ValueError("prepared HLT endpoint row count differs")

    features = np.zeros_like(canonical.features)
    vectors = np.zeros_like(canonical.vectors)
    mask = np.zeros_like(canonical.mask)
    lengths = np.zeros_like(canonical.raw_lengths)
    visible_ids = np.full((rows, 200), -1, np.int64)
    family_codes = np.full((rows, 200), PADDED_FAMILY, np.int8)
    family_reasons = np.full((rows, 200), PADDED_REASON, np.int8)
    hlt_metadata = derive_hcwdl_token_metadata(arrays)
    for row in range(rows):
        partition, projected, projected_p4 = _partition_for_row(
            row=row, assignment=mapping[row],
            offline=prepared_offline, hlt=prepared_hlt,
        )
        charged_count = int(prepared_offline.charged_counts[row])
        edits = tuple(coupling_rows[row]); _validate_edit_partition(partition, edits)
        target_by_slot = {record.hlt_slot: record for record in partition.d100}
        common_by_slot = {pair.target.hlt_slot: pair for pair in partition.common}
        source_by_native = {record.native_index: record for record in partition.source_only}

        # Entries are (carrier kind, carrier key, feature, vector).  Target
        # slots sort before tail native slots, exactly as the v1 carrier says.
        active: list[tuple[int, int, np.ndarray, np.ndarray, int, int]] = []
        for slot in sorted(common_by_slot):
            native = common_by_slot[slot].source.native_index
            charged_family = native < charged_count
            active.append((
                0, slot, shell.features[row, :, slot], shell.vectors[row, :, slot],
                int(CHARGED_FAMILY if charged_family else NEUTRAL_FAMILY),
                int(DIRECT_CHARGED_REASON if charged_family else DIRECT_NEUTRAL_REASON),
            ))
        for edit in edits:
            switched = edit_is_active(
                edit, numerator=coordinate.structural_numerator,
                denominator=coordinate.structural_denominator,
            )
            if edit.edit_kind == EDIT_SUBSTITUTION:
                if switched:
                    slot = edit.target_hlt_slot
                    target = target_by_slot[slot]
                    if target.native_index >= 0:
                        charged_family = target.native_index < charged_count
                        fam = CHARGED_FAMILY if charged_family else NEUTRAL_FAMILY
                        why = DIRECT_CHARGED_REASON if charged_family else DIRECT_NEUTRAL_REASON
                    else:
                        fam = hlt_metadata.family_codes[row, slot]
                        why = hlt_metadata.family_reason_codes[row, slot]
                    active.append((0, slot, shell.features[row, :, slot], shell.vectors[row, :, slot], int(fam), int(why)))
                else:
                    native = edit.source_native_index
                    active.append((0, edit.target_hlt_slot, projected[native],
                                   projected_p4[native].astype(np.float32),
                                   int(CHARGED_FAMILY if native < charged_count else NEUTRAL_FAMILY),
                                   int(DIRECT_CHARGED_REASON if native < charged_count else DIRECT_NEUTRAL_REASON)))
            elif edit.edit_kind == EDIT_INSERTION:
                if switched:
                    slot = edit.target_hlt_slot
                    target = target_by_slot[slot]
                    if target.native_index >= 0:
                        charged_family = target.native_index < charged_count
                        fam = CHARGED_FAMILY if charged_family else NEUTRAL_FAMILY
                        why = DIRECT_CHARGED_REASON if charged_family else DIRECT_NEUTRAL_REASON
                    else:
                        fam = hlt_metadata.family_codes[row, slot]
                        why = hlt_metadata.family_reason_codes[row, slot]
                    active.append((0, slot, shell.features[row, :, slot], shell.vectors[row, :, slot], int(fam), int(why)))
            elif edit.edit_kind == EDIT_REMOVAL:
                if not switched:
                    native = edit.source_native_index
                    active.append((1, native, projected[native],
                                   projected_p4[native].astype(np.float32),
                                   int(CHARGED_FAMILY if native < charged_count else NEUTRAL_FAMILY),
                                   int(DIRECT_CHARGED_REASON if native < charged_count else DIRECT_NEUTRAL_REASON)))
            else:
                raise ValueError("unknown coupling edit kind")
        active.sort(key=lambda item: (item[0], item[1]))
        if len(active) != len({(kind, key) for kind, key, *_ in active}):
            raise ValueError("HCWDL-UJ carrier contains duplicate logical slots")
        if len(active) > 200:
            raise ValueError("HCWDL-UJ carrier requires hidden truncation")
        lengths[row] = len(active)
        mask[row, 0, :len(active)] = True
        for index, (kind, key, feature, vector, family, reason) in enumerate(active):
            features[row, :, index] = feature
            vectors[row, :, index] = vector
            visible_ids[row, index] = key if kind == 0 else 200 + key
            family_codes[row, index] = family
            family_reasons[row, index] = reason
    if not np.isfinite(features).all() or not np.isfinite(vectors).all():
        raise FloatingPointError("HCWDL-UJ active view became nonfinite")
    if np.any(features[~np.repeat(mask, features.shape[1], axis=1)]) or np.any(
        vectors[~np.repeat(mask, vectors.shape[1], axis=1)]
    ):
        raise RuntimeError("HCWDL-UJ view has nonzero padding")
    result = ParticleInputs(features, vectors, mask, lengths)
    if not include_training_metadata:
        return result
    return attach_hcwdl_token_metadata(
        result, HCWDLTokenMetadata(visible_ids, family_codes, family_reasons),
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
    "build_homotopy_inputs", "build_p0_inputs", "build_partition_from_arrays",
    "particle_inputs_sha256", "PreparedHltEndpoints", "PreparedOfflineEndpoints",
    "prepare_hlt_endpoints", "prepare_offline_endpoints",
]

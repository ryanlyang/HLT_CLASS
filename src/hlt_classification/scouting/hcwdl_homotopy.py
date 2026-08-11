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
from .repair import (
    HIGHCOV_SHELL_EXACT_FAMILY, build_alpha_repaired_inputs,
    project_offline_endpoint_records, transform_endpoint_features,
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


def _rows(value: object) -> list[np.ndarray]:
    try:
        import awkward as ak
        if isinstance(value, ak.Array):
            return [np.asarray(row) for row in ak.to_list(value)]
    except ImportError:
        pass
    return [np.asarray(row) for row in value]  # type: ignore[arg-type]


def _raw_hlt_row(arrays: Mapping[str, object], row: int) -> tuple[np.ndarray, np.ndarray, int]:
    features = np.stack([
        _rows(arrays[spec.branch])[row] for spec in HLT_FEATURE_SPECS
    ], axis=1).astype(np.float64, copy=False)
    vectors = np.stack([
        _rows(arrays[branch])[row] for branch in HLT_VECTOR_BRANCHES
    ], axis=1).astype(np.float64, copy=False)
    if len(features) != len(vectors):
        raise ValueError("HLT raw feature/vector row lengths differ")
    raw_count = _scalar_count(arrays, "n_scoutpfcands", row)
    if len(features) != raw_count:
        raise ValueError("HLT endpoint collection differs from n_scoutpfcands")
    return features, vectors, raw_count


def _scalar_count(arrays: Mapping[str, object], branch: str, row: int) -> int:
    if branch not in arrays:
        raise KeyError(f"HCWDL-UJ endpoint input lacks {branch}")
    value = np.asarray(_rows(arrays[branch])[row])
    if value.ndim != 0 or not np.isfinite(value) or int(value) != value or int(value) < 0:
        raise ValueError(f"{branch} must be a nonnegative integer scalar")
    return int(value)


def _offline_counts(arrays: Mapping[str, object], row: int) -> tuple[int, int]:
    charged = _scalar_count(arrays, "n_cpfcands", row) + _scalar_count(
        arrays, "n_lts", row,
    )
    neutral = _scalar_count(arrays, "n_npfcands", row)
    if len(_rows(arrays["cpfcandlt_px"])[row]) != charged:
        raise ValueError("cpfcandlt endpoint collection differs from n_cpfcands + n_lts")
    if len(_rows(arrays["npfcand_px"])[row]) != neutral:
        raise ValueError("npfcand endpoint collection differs from n_npfcands")
    return charged, neutral


def _partition_for_row(
    arrays: Mapping[str, object], *, row: int, assignment: np.ndarray,
) -> tuple[EndpointPartition, np.ndarray, np.ndarray]:
    offline_features, offline_validity, offline_p4 = project_offline_endpoint_records(
        arrays, row=row,
    )
    charged, neutral = _offline_counts(arrays, row)
    hlt_features, hlt_p4, raw_hlt = _raw_hlt_row(arrays, row)
    partition = build_endpoint_partition(
        offline_features=offline_features, offline_validity=offline_validity,
        offline_p4=offline_p4, charged_count=charged, neutral_count=neutral,
        hlt_features=hlt_features[:200], hlt_p4=hlt_p4[:200],
        assignment=assignment, raw_hlt_length=raw_hlt,
    )
    return partition, transform_endpoint_features(offline_features, offline_validity), offline_p4


def build_partition_from_arrays(
    arrays: Mapping[str, object], *, row: int, assignment: np.ndarray,
) -> EndpointPartition:
    """Public one-row endpoint partition used by coupling production/audits."""

    return _partition_for_row(arrays, row=row, assignment=assignment)[0]


def build_p0_inputs(arrays: Mapping[str, object]) -> ParticleInputs:
    """Build the independent projected-native P0 endpoint in native order."""

    rows = len(_rows(arrays["cpfcandlt_px"]))
    features = np.zeros((rows, 21, 200), np.float32)
    vectors = np.zeros((rows, 4, 200), np.float32)
    mask = np.zeros((rows, 1, 200), np.bool_)
    lengths = np.zeros(rows, np.int32)
    for row in range(rows):
        raw, validity, p4 = project_offline_endpoint_records(arrays, row=row)
        charged, neutral = _offline_counts(arrays, row)
        indices = [*range(min(charged, 90))]
        indices.extend(charged + index for index in range(min(neutral, 60)))
        projected = transform_endpoint_features(raw, validity)
        length = len(indices)
        if length > 150:
            raise RuntimeError("P0 visible population exceeds the 90/60 bound")
        if length:
            features[row, :, :length] = projected[indices].T
            vectors[row, :, :length] = np.asarray(p4, np.float32)[indices].T
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
    discrete_seed: int,
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
        return build_p0_inputs(arrays)
    if coordinate.s == 1.0 and coordinate.f == 1.0:
        return canonical
    offline_p4 = [project_offline_endpoint_records(arrays, row=row)[2] for row in range(rows)]
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

    features = np.zeros_like(canonical.features)
    vectors = np.zeros_like(canonical.vectors)
    mask = np.zeros_like(canonical.mask)
    lengths = np.zeros_like(canonical.raw_lengths)
    for row in range(rows):
        partition, projected, projected_p4 = _partition_for_row(
            arrays, row=row, assignment=mapping[row],
        )
        edits = tuple(coupling_rows[row]); _validate_edit_partition(partition, edits)
        target_by_slot = {record.hlt_slot: record for record in partition.d100}
        common_by_slot = {pair.target.hlt_slot: pair for pair in partition.common}
        source_by_native = {record.native_index: record for record in partition.source_only}

        # Entries are (carrier kind, carrier key, feature, vector).  Target
        # slots sort before tail native slots, exactly as the v1 carrier says.
        active: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        for slot in sorted(common_by_slot):
            active.append((0, slot, shell.features[row, :, slot], shell.vectors[row, :, slot]))
        for edit in edits:
            switched = edit_is_active(
                edit, numerator=coordinate.structural_numerator,
                denominator=coordinate.structural_denominator,
            )
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
            raise ValueError("HCWDL-UJ carrier contains duplicate logical slots")
        if len(active) > 200:
            raise ValueError("HCWDL-UJ carrier requires hidden truncation")
        lengths[row] = len(active)
        mask[row, 0, :len(active)] = True
        for index, (_, _, feature, vector) in enumerate(active):
            features[row, :, index] = feature
            vectors[row, :, index] = vector
    if not np.isfinite(features).all() or not np.isfinite(vectors).all():
        raise FloatingPointError("HCWDL-UJ active view became nonfinite")
    if np.any(features[~np.repeat(mask, features.shape[1], axis=1)]) or np.any(
        vectors[~np.repeat(mask, vectors.shape[1], axis=1)]
    ):
        raise RuntimeError("HCWDL-UJ view has nonzero padding")
    return ParticleInputs(features, vectors, mask, lengths)


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
    "particle_inputs_sha256",
]

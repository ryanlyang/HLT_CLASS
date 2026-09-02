"""Strict training-only HLT token metadata for HCWDL representation KD.

The public ParT input remains the repository's 21 floating feature channels,
four-vectors and mask.  This module derives canonical token identities and the
M6 charged/neutral bookkeeping from the *raw* charge/PID branches before the
ordinary input builder can sanitize invalid values.  The metadata is carried
through the model trimmer as auxiliary channels and is never a deployable
feature.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from hlt_classification.data.cache_contracts import canonical_sha256

from .inputs import ParticleInputs, build_hlt_inputs
from .schema import HLT_MAX_LENGTH


CHARGED_FAMILY: Final = np.int8(0)
NEUTRAL_FAMILY: Final = np.int8(1)
CONTRADICTION_FAMILY: Final = np.int8(2)
MALFORMED_FAMILY: Final = np.int8(3)
PADDED_FAMILY: Final = np.int8(-1)

DIRECT_CHARGED_REASON: Final = np.int8(0)
DIRECT_NEUTRAL_REASON: Final = np.int8(1)
CHARGE_ONLY_CHARGED_REASON: Final = np.int8(2)
CHARGE_ONLY_NEUTRAL_REASON: Final = np.int8(3)
CONTRADICTION_REASON: Final = np.int8(4)
MALFORMED_REASON: Final = np.int8(5)
PADDED_REASON: Final = np.int8(-1)

RAW_CHARGE_BRANCH: Final = "scoutpfcand_charge"
RAW_PID_BRANCHES: Final = (
    "scoutpfcand_isEl",
    "scoutpfcand_isMu",
    "scoutpfcand_isChargedHad",
    "scoutpfcand_isGamma",
    "scoutpfcand_isNeutralHad",
)


@dataclass(frozen=True)
class HCWDLParticleInputs(ParticleInputs):
    """Ordinary HLT tensors plus nondeployable pre-transform metadata."""

    visible_indices: np.ndarray
    family_codes: np.ndarray
    family_reason_codes: np.ndarray


@dataclass(frozen=True)
class HCWDLTaggedParticleInputs(HCWDLParticleInputs):
    """Training-only particle tensors with an explicit content-source tag.

    ``content_source_codes`` is bookkeeping transported through the token
    trimmer.  It is never one of the 21 normalized physics features.
    """

    content_source_codes: np.ndarray


@dataclass(frozen=True)
class HCWDLTokenMetadata:
    visible_indices: np.ndarray
    family_codes: np.ndarray
    family_reason_codes: np.ndarray


def _rows(value: object) -> list[np.ndarray]:
    try:
        import awkward as ak

        if isinstance(value, ak.Array):
            return [np.asarray(row) for row in ak.to_list(value)]
    except ImportError:
        ak = None
    if isinstance(value, np.ndarray) and value.ndim == 2:
        return [np.asarray(row) for row in value]
    return [np.asarray(row) for row in value]  # type: ignore[arg-type]


def _derive_hcwdl_token_metadata(
    arrays: Mapping[str, object], *, max_length: int,
) -> HCWDLTokenMetadata:
    """Classify raw HLT tokens exactly as Section 9.4 requires.

    All native raw rows are validated, including tokens beyond the model's
    fixed maximum.  Invalid data may not disappear merely because truncation
    would otherwise hide it.
    """

    if max_length <= 0:
        raise ValueError("HCWDL HLT metadata length must be positive")
    missing = [name for name in (RAW_CHARGE_BRANCH, *RAW_PID_BRANCHES) if name not in arrays]
    if missing:
        raise KeyError(f"HCWDL raw token metadata branches are absent: {missing}")
    charge_rows = _rows(arrays[RAW_CHARGE_BRANCH])
    flag_rows = {name: _rows(arrays[name]) for name in RAW_PID_BRANCHES}
    row_count = len(charge_rows)
    if any(len(rows) != row_count for rows in flag_rows.values()):
        raise ValueError("HCWDL raw token metadata row counts differ")

    visible_indices = np.full((row_count, max_length), -1, dtype=np.int64)
    family = np.full((row_count, max_length), PADDED_FAMILY, dtype=np.int8)
    reason = np.full((row_count, max_length), PADDED_REASON, dtype=np.int8)
    for row_index, raw_charge in enumerate(charge_rows):
        charge = np.asarray(raw_charge)
        flags_by_name = [np.asarray(flag_rows[name][row_index]) for name in RAW_PID_BRANCHES]
        lengths = [len(charge), *(len(values) for values in flags_by_name)]
        if len(set(lengths)) != 1:
            raise ValueError(f"HCWDL raw token metadata length mismatch in row {row_index}")
        if len(charge) and not np.isfinite(charge).all():
            raise ValueError(f"visible raw charge is nonfinite in row {row_index}")
        if len(charge) and not np.isin(charge, (-1, 0, 1)).all():
            raise ValueError(
                f"visible raw charge lies outside {{-1,0,+1}} in row {row_index}"
            )
        if flags_by_name:
            flags = np.stack(flags_by_name, axis=-1)
        else:  # pragma: no cover - the frozen five-branch registry is nonempty
            flags = np.empty((len(charge), 0), dtype=np.float32)
        if flags.size and not np.isfinite(flags).all():
            raise ValueError(f"visible raw PID flags are nonfinite in row {row_index}")

        binary = ((flags == 0) | (flags == 1)).all(axis=-1)
        flag_count = flags.sum(axis=-1)
        known = binary & (flag_count == 1)
        unknown = binary & (flag_count == 0)
        malformed = ~(known | unknown)
        charge_family = np.where(charge == 0, NEUTRAL_FAMILY, CHARGED_FAMILY).astype(np.int8)
        pid_index = np.argmax(flags, axis=-1) if len(flags) else np.empty(0, dtype=np.int64)
        pid_family = np.where(pid_index <= 2, CHARGED_FAMILY, NEUTRAL_FAMILY).astype(np.int8)
        agree = known & (pid_family == charge_family)
        contradiction = known & ~agree

        native_family = np.full(len(charge), MALFORMED_FAMILY, dtype=np.int8)
        native_reason = np.full(len(charge), MALFORMED_REASON, dtype=np.int8)
        native_family[agree] = pid_family[agree]
        native_reason[agree & (pid_family == CHARGED_FAMILY)] = DIRECT_CHARGED_REASON
        native_reason[agree & (pid_family == NEUTRAL_FAMILY)] = DIRECT_NEUTRAL_REASON
        native_family[unknown] = charge_family[unknown]
        native_reason[unknown & (charge_family == CHARGED_FAMILY)] = CHARGE_ONLY_CHARGED_REASON
        native_reason[unknown & (charge_family == NEUTRAL_FAMILY)] = CHARGE_ONLY_NEUTRAL_REASON
        native_family[contradiction] = CONTRADICTION_FAMILY
        native_reason[contradiction] = CONTRADICTION_REASON
        native_family[malformed] = MALFORMED_FAMILY
        native_reason[malformed] = MALFORMED_REASON

        visible = min(len(charge), max_length)
        visible_indices[row_index, :visible] = np.arange(visible, dtype=np.int64)
        family[row_index, :visible] = native_family[:visible]
        reason[row_index, :visible] = native_reason[:visible]

    return HCWDLTokenMetadata(
        np.ascontiguousarray(visible_indices),
        np.ascontiguousarray(family),
        np.ascontiguousarray(reason),
    )


def derive_hcwdl_token_metadata(
    arrays: Mapping[str, object], *, max_length: int = HLT_MAX_LENGTH,
) -> HCWDLTokenMetadata:
    """Derive ordinary HLT metadata under the deployable 200-token bound."""

    if max_length <= 0 or max_length > HLT_MAX_LENGTH:
        raise ValueError(f"HCWDL HLT max_length must lie in [1,{HLT_MAX_LENGTH}]")
    return _derive_hcwdl_token_metadata(arrays, max_length=max_length)


def derive_extended_training_hlt_token_metadata(
    arrays: Mapping[str, object], *, max_length: int,
) -> HCWDLTokenMetadata:
    """Derive metadata beyond 200 only for an explicit privileged view.

    The ordinary public helper remains capped by ``HLT_MAX_LENGTH``.  This
    adapter reuses its exact classification semantics while authorizing a
    larger transport tensor for nondeployable training/oracle inputs.
    """

    if max_length <= HLT_MAX_LENGTH:
        raise ValueError("extended HLT metadata requires a length above 200")
    # The classification implementation is length-agnostic. Temporarily use
    # the requested transport length through the private shared body rather
    # than weakening the deployable helper's public bound.
    return _derive_hcwdl_token_metadata(arrays, max_length=max_length)


def attach_hcwdl_token_metadata(
    inputs: ParticleInputs, metadata: HCWDLTokenMetadata,
) -> HCWDLParticleInputs:
    rows, _, tokens = inputs.features.shape
    expected = (rows, tokens)
    if (
        metadata.visible_indices.shape != expected
        or metadata.family_codes.shape != expected
        or metadata.family_reason_codes.shape != expected
    ):
        raise ValueError("HCWDL token metadata shape differs from particle inputs")
    visible = np.asarray(inputs.mask)[:, 0].astype(np.bool_, copy=False)
    if not np.array_equal(metadata.visible_indices >= 0, visible):
        raise ValueError("HCWDL token metadata visibility differs from particle mask")
    if np.any(metadata.family_codes[~visible] != PADDED_FAMILY):
        raise ValueError("HCWDL padded family code differs")
    if np.any(metadata.family_reason_codes[~visible] != PADDED_REASON):
        raise ValueError("HCWDL padded family reason differs")
    return HCWDLParticleInputs(
        features=inputs.features,
        vectors=inputs.vectors,
        mask=inputs.mask,
        raw_lengths=inputs.raw_lengths,
        visible_indices=metadata.visible_indices,
        family_codes=metadata.family_codes,
        family_reason_codes=metadata.family_reason_codes,
    )


def build_hcwdl_hlt_inputs(
    arrays: Mapping[str, object], *, max_length: int = HLT_MAX_LENGTH,
) -> HCWDLParticleInputs:
    metadata = derive_hcwdl_token_metadata(arrays, max_length=max_length)
    return attach_hcwdl_token_metadata(
        build_hlt_inputs(arrays, max_length=max_length), metadata,
    )


def canonical_identity_digest(identity_key: str) -> np.ndarray:
    if not isinstance(identity_key, str) or not identity_key:
        raise ValueError("HCWDL canonical identity key must be a nonempty string")
    return np.frombuffer(
        bytes.fromhex(canonical_sha256({
            "contract": "HCWDL_REPRESENTATION_CANONICAL_IDENTITY/v1",
            "identity_key": identity_key,
        })),
        dtype=np.uint8,
    ).copy()


def canonical_identity_digests(identity_keys: Sequence[str]) -> np.ndarray:
    keys = tuple(str(value) for value in identity_keys)
    if not keys or any(not value for value in keys) or len(keys) != len(set(keys)):
        raise ValueError("HCWDL canonical identity keys must be nonempty and unique")
    rows = np.stack([canonical_identity_digest(value) for value in keys], axis=0)
    if len({bytes(row) for row in rows}) != len(rows):
        raise RuntimeError("HCWDL canonical identity digest collision")
    return np.ascontiguousarray(rows, dtype=np.uint8)


def training_batch_from_parent(
    batch: Mapping[str, object], *, student_view: str = "hlt",
) -> dict[str, object]:
    """Project a parent stream onto the explicitly registered student view.

    Dense-descent intermediates D100--D5 intentionally consume authenticated
    repaired views and are nondeployable.  D0 and M1 pass ``student_view='hlt'``
    and remain the only HLT-deployable results.
    """

    if set(batch) - {"hlt", "labels", "identity_keys", "privileged", "toff", "observers"}:
        raise ValueError("parent batch contains an unknown field")
    if student_view not in {"hlt", "privileged"}:
        raise ValueError("HCWDL-RKD student view differs")
    hlt = batch.get(student_view)
    if not isinstance(hlt, HCWDLParticleInputs):
        raise TypeError("parent batch lacks the registered HCWDL student metadata")
    keys = np.asarray(batch.get("identity_keys"))
    if keys.ndim != 1 or len(keys) != len(hlt.features):
        raise ValueError("parent batch canonical identities differ")
    return {
        "hlt": hlt,
        "labels": np.asarray(batch["labels"]),
        "identity_digests": canonical_identity_digests(tuple(map(str, keys.tolist()))),
    }


def iterate_hcwdl_model_batches(*args, **kwargs) -> Iterator[dict[str, object]]:
    """Adapt the canonical HLT reader to the strict representation trainer."""

    from .dataset import iterate_model_batches

    if "include_hcwdl_metadata" in kwargs:
        raise TypeError("HCWDL metadata mode is owned by this adapter")
    if kwargs.get("input_mode", "hlt") not in {"hlt", "paired"}:
        raise ValueError("HCWDL student batches require an HLT input view")
    for batch in iterate_model_batches(*args, **kwargs, include_hcwdl_metadata=True):
        yield training_batch_from_parent(batch)


def iterate_hcwdl_pmard_batches(*args, teacher_view: bool = False, **kwargs):
    """Adapt the parent repaired-view stream without exposing it to students.

    ``teacher_view=True`` is reserved for target construction and returns the
    parent batch with authenticated metadata on both HLT and privileged views.
    The default returns only the HLT student boundary.
    """

    from .pmard_stream import iterate_pmard_batches

    if "include_hcwdl_metadata" in kwargs:
        raise TypeError("HCWDL metadata mode is owned by this adapter")
    for batch in iterate_pmard_batches(*args, **kwargs, include_hcwdl_metadata=True):
        if teacher_view:
            yield batch
        else:
            yield training_batch_from_parent(batch)


__all__ = [
    "CHARGED_FAMILY", "CHARGE_ONLY_CHARGED_REASON",
    "CHARGE_ONLY_NEUTRAL_REASON", "CONTRADICTION_FAMILY",
    "CONTRADICTION_REASON", "DIRECT_CHARGED_REASON", "DIRECT_NEUTRAL_REASON",
    "HCWDLParticleInputs", "HCWDLTaggedParticleInputs", "HCWDLTokenMetadata",
    "MALFORMED_FAMILY",
    "MALFORMED_REASON", "NEUTRAL_FAMILY", "PADDED_FAMILY", "PADDED_REASON",
    "RAW_CHARGE_BRANCH", "RAW_PID_BRANCHES", "attach_hcwdl_token_metadata",
    "build_hcwdl_hlt_inputs", "canonical_identity_digest",
    "canonical_identity_digests", "derive_hcwdl_token_metadata",
    "derive_extended_training_hlt_token_metadata",
    "iterate_hcwdl_model_batches", "iterate_hcwdl_pmard_batches",
    "training_batch_from_parent",
]

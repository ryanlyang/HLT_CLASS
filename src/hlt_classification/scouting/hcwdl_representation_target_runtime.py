"""Deterministic one-forward construction for HCWDL-RKD target banks.

The runtime deliberately accepts a batch iterator plus one teacher-surface
callback.  It owns every operation after that single callback: strict FP32
surface validation, weighted token and latent-relation sketches, canonical
batch audits, staged publication, sentinel replay, and release of live token
surfaces before the next batch.  No particle-sized teacher data are published.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
import os
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Final
import zipfile

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    load_json,
    require_sha256,
)

from .hcwdl_representation_kernels import SpectralKernelResources
from .hcwdl_representation_losses import (
    CHARGED_FAMILY,
    CHARGE_ONLY_CLASSIFICATION,
    CONTRADICTION_CLASSIFICATION,
    DIRECT_CLASSIFICATION,
    MALFORMED_CLASSIFICATION,
    NEUTRAL_FAMILY,
    build_ordinary_token_targets,
    build_teacher_relation_targets,
    classify_hlt_token_families,
)
from .hcwdl_representation_artifacts import FailureHook
from .hcwdl_representation_targets import (
    ORDINARY_BANK,
    KERNEL_RESOURCE_NAMES,
    TARGET_FORWARD_BATCH_SIZE,
    TOFF_BANK,
    TargetGenerationContext,
    TargetPopulationHasher,
    finalize_target_generation,
    identity_order_sha256,
    identity_set_sha256,
    load_staged_target_shard,
    stage_target_shard,
    target_array_schema,
    validate_target_arrays,
    validate_target_generation,
)
from .hcwdl_representation_contracts import (
    TARGET_EXECUTION_ATTESTATION_CONTRACT,
    logical_array_sha256,
    validate_versioned_artifact,
)
@dataclass(frozen=True)
class TeacherModelInputs:
    """The only object visible to a teacher callback: allow-listed model arrays."""

    arrays: Mapping[str, Any]


TeacherSurfaceForward = Callable[[TeacherModelInputs], Any]
BatchFactory = Callable[[], Iterable["TargetForwardBatch"]]
_HASHED_BATCH_ARRAYS: Final = ("logits", "surfaces", "sketches")


@dataclass(frozen=True)
class TargetForwardBatch:
    """Audit identity plus model inputs; audit fields never reach the callback."""

    source_partition: str
    source_file_id: np.ndarray
    source_entry: np.ndarray
    identity_digest: np.ndarray
    label: np.ndarray
    teacher_inputs: Mapping[str, Any]
    companion_hlt_charge: np.ndarray | None = None
    companion_hlt_pid_flags: np.ndarray | None = None
    companion_hlt_visible_mask: np.ndarray | None = None

    @property
    def rows(self) -> int:
        return int(len(self.source_entry))


@dataclass(frozen=True)
class TargetRuntimeResult:
    manifest: Mapping[str, Any]
    execution_attestation: Mapping[str, Any]
    teacher_forward_calls: int
    sentinel_replay_calls: int
    published_partitions: tuple[str, ...]
    reused_partitions: tuple[str, ...]


@dataclass(frozen=True)
class PreparedTargetPartition:
    """One compact source shard held in RAM after its sole source/teacher pass."""

    arrays: Mapping[str, np.ndarray]
    runtime_audit: Mapping[str, Any]
    teacher_forward_calls: int


@dataclass(frozen=True)
class PreparedTargetGeneration:
    """Process-local compact targets and the exact population they attest."""

    bank_kind: str
    partitions: Mapping[str, PreparedTargetPartition]
    partition_specs: Mapping[str, Mapping[str, int]]
    class_counts: tuple[int, ...]
    identity_order_sha256: str
    identity_set_sha256: str
    population_rows_sha256: str
    canonical_batches: tuple[Mapping[str, Any], ...]
    teacher_forward_calls: int
    construction_seconds: float


@dataclass(frozen=True)
class PredecessorLogitBank:
    identity_digest: np.ndarray
    logits: np.ndarray
    identity_order_sha256: str
    identity_set_sha256: str
    logits_sha256: str
    logical_sha256: str
    _lookup: Mapping[bytes, int]

    def join(self, identity_digest: np.ndarray) -> np.ndarray:
        requested = np.asarray(identity_digest)
        if requested.dtype != np.uint8 or requested.ndim != 2 or requested.shape[1] != 32:
            raise ValueError("HCWDL-RKD predecessor join identities must be uint8 [rows,32]")
        keys = [bytes(row) for row in requested]
        if len(keys) != len(set(keys)):
            raise ValueError("HCWDL-RKD predecessor join repeats an identity")
        try:
            indexes = np.asarray([self._lookup[key] for key in keys], dtype=np.int64)
        except KeyError as error:
            raise KeyError("HCWDL-RKD predecessor-logit identity join is incomplete") from error
        return np.ascontiguousarray(self.logits[indexes])


def _as_numpy(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "requires_grad") and bool(value.requires_grad):
        raise ValueError(f"HCWDL-RKD teacher surface retains gradients: {name}")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if not result.flags.c_contiguous:
        result = np.ascontiguousarray(result)
    return result


def _float32(value: Any, *, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    result = _as_numpy(value, name=name)
    if result.dtype != np.float32 or (shape is not None and result.shape != shape):
        raise ValueError(f"HCWDL-RKD teacher FP32 surface shape/dtype differs: {name}")
    if not np.isfinite(result).all():
        raise FloatingPointError(f"HCWDL-RKD teacher surface is nonfinite: {name}")
    return result


def _integer_array(
    value: Any, *, name: str, dtype: np.dtype[Any], shape: tuple[int, ...],
) -> np.ndarray:
    result = _as_numpy(value, name=name)
    if result.dtype != dtype or result.shape != shape:
        raise ValueError(f"HCWDL-RKD teacher metadata shape/dtype differs: {name}")
    return result


def _validate_batch_identity(
    batch: TargetForwardBatch, *, partition: str, expected_source_file_id: int,
) -> None:
    rows = batch.rows
    if batch.source_partition != partition or not 1 <= rows <= TARGET_FORWARD_BATCH_SIZE:
        raise ValueError("HCWDL-RKD canonical target batch partition/size differs")
    source_ids = _integer_array(
        batch.source_file_id, name="source_file_id", dtype=np.dtype("<u4"), shape=(rows,),
    )
    entries = _integer_array(
        batch.source_entry, name="source_entry", dtype=np.dtype("<u8"), shape=(rows,),
    )
    identities = _integer_array(
        batch.identity_digest, name="identity_digest", dtype=np.dtype("u1"),
        shape=(rows, 32),
    )
    labels = _integer_array(
        batch.label, name="label", dtype=np.dtype("u1"), shape=(rows,),
    )
    if (
        np.any(source_ids != expected_source_file_id)
        or np.any(entries[1:] <= entries[:-1])
        or len({bytes(row) for row in identities}) != rows
        or np.any(labels > 14)
    ):
        raise ValueError("HCWDL-RKD canonical target batch identity/label differs")
    if not isinstance(batch.teacher_inputs, Mapping):
        raise ValueError("HCWDL-RKD target teacher inputs are not a mapping")


def _teacher_model_inputs(
    batch: TargetForwardBatch, *, allowed_fields: tuple[str, ...],
) -> TeacherModelInputs:
    if set(batch.teacher_inputs) != set(allowed_fields):
        raise PermissionError("HCWDL-RKD target teacher inputs differ from the allow-list")
    # Copy the registry and expose it read-only.  Labels, row identities,
    # family bookkeeping, assignment fields, and source coordinates are not
    # members of this object and therefore cannot accidentally enter a model
    # callback through the runtime API.
    for name, value in batch.teacher_inputs.items():
        if isinstance(value, Mapping):
            raise PermissionError(f"nested HCWDL-RKD teacher input is forbidden: {name}")
        dtype = getattr(value, "dtype", None)
        if dtype is not None:
            dtype_name = str(dtype).lower()
            if any(marker in dtype_name for marker in ("float", "half", "bfloat")) and (
                "float32" not in dtype_name
            ):
                raise ValueError(f"HCWDL-RKD teacher floating input is not FP32: {name}")
    return TeacherModelInputs(MappingProxyType(dict(batch.teacher_inputs)))


def _validate_teacher_model_runtime(model: Any) -> None:
    import torch

    if model is None or not hasattr(model, "parameters") or not hasattr(model, "training"):
        raise ValueError("HCWDL-RKD target builder lacks an inspectable teacher model")
    if bool(model.training):
        raise ValueError("HCWDL-RKD target teacher is not in evaluation mode")
    parameters = tuple(model.parameters())
    if not parameters or any(parameter.dtype != torch.float32 for parameter in parameters):
        raise ValueError("HCWDL-RKD target teacher parameters are not all FP32")


def _validate_process_backend() -> None:
    import torch

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("HCWDL-RKD deterministic CUBLAS workspace is not active")
    if not torch.are_deterministic_algorithms_enabled():
        raise ValueError("HCWDL-RKD deterministic algorithms are not active")
    if not bool(torch.backends.cudnn.deterministic) or bool(torch.backends.cudnn.benchmark):
        raise ValueError("HCWDL-RKD cuDNN deterministic flags differ")
    if bool(torch.backends.cuda.matmul.allow_tf32) or bool(torch.backends.cudnn.allow_tf32):
        raise ValueError("HCWDL-RKD TF32 is active during target construction")


def _companion_hlt_reason_counts(batch: TargetForwardBatch) -> np.ndarray:
    """Derive the six exact Section-9.4 reason counts after canonical trimming."""

    if (
        batch.companion_hlt_charge is None
        or batch.companion_hlt_pid_flags is None
        or batch.companion_hlt_visible_mask is None
    ):
        raise ValueError("HCWDL-RKD TOFF batch lacks raw companion-HLT family inputs")
    charge = _float32(
        batch.companion_hlt_charge,
        name="companion_hlt_charge",
    )
    flags = _float32(
        batch.companion_hlt_pid_flags,
        name="companion_hlt_pid_flags",
    )
    mask = _as_numpy(
        batch.companion_hlt_visible_mask,
        name="companion_hlt_visible_mask",
    )
    if (
        charge.ndim != 2 or charge.shape[0] != batch.rows
        or flags.shape != (*charge.shape, 5)
        or mask.dtype != np.bool_ or mask.shape != charge.shape
    ):
        raise ValueError("HCWDL-RKD companion-HLT family input shapes differ")
    classification = classify_hlt_token_families(charge, flags, mask)
    family = _as_numpy(classification.family_codes, name="companion_hlt_family_codes")
    reason = _as_numpy(classification.reason_codes, name="companion_hlt_reason_codes")
    result = np.zeros((batch.rows, 6), dtype=np.uint16)
    for row in range(batch.rows):
        visible = mask[row]
        direct_charged = visible & (reason[row] == DIRECT_CLASSIFICATION) & (
            family[row] == CHARGED_FAMILY
        )
        direct_neutral = visible & (reason[row] == DIRECT_CLASSIFICATION) & (
            family[row] == NEUTRAL_FAMILY
        )
        charge_only_charged = visible & (
            reason[row] == CHARGE_ONLY_CLASSIFICATION
        ) & (family[row] == CHARGED_FAMILY)
        charge_only_neutral = visible & (
            reason[row] == CHARGE_ONLY_CLASSIFICATION
        ) & (family[row] == NEUTRAL_FAMILY)
        contradiction = visible & (reason[row] == CONTRADICTION_CLASSIFICATION)
        malformed = visible & (reason[row] == MALFORMED_CLASSIFICATION)
        counts = tuple(
            int(np.count_nonzero(value)) for value in (
                direct_charged, direct_neutral, charge_only_charged,
                charge_only_neutral, contradiction, malformed,
            )
        )
        if sum(counts) != int(np.count_nonzero(visible)):
            raise ValueError("HCWDL-RKD companion-HLT family reasons do not conserve tokens")
        result[row] = counts
    return result


def _surface_field(surface: Any, name: str) -> Any:
    if isinstance(surface, Mapping):
        if name not in surface:
            raise ValueError(f"HCWDL-RKD teacher surface lacks {name}")
        return surface[name]
    if not hasattr(surface, name):
        raise ValueError(f"HCWDL-RKD teacher surface lacks {name}")
    return getattr(surface, name)


def _visible_cloud(
    hidden: Any, mask: Any, vectors: Any, visible_indices: Any,
    *, rows: int, prefix: str, allow_empty: bool = False,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    hidden_array = _float32(hidden, name=f"{prefix}.hidden")
    if hidden_array.ndim != 3 or hidden_array.shape[0] != rows or hidden_array.shape[2] != 128:
        raise ValueError(f"HCWDL-RKD {prefix} hidden surface shape differs")
    tokens = hidden_array.shape[1]
    mask_array = _as_numpy(mask, name=f"{prefix}.mask")
    if mask_array.dtype != np.bool_ or mask_array.shape != (rows, tokens):
        raise ValueError(f"HCWDL-RKD {prefix} mask shape/dtype differs")
    vectors_array = _float32(
        vectors, name=f"{prefix}.vectors", shape=(rows, 4, tokens),
    )
    ids = _integer_array(
        visible_indices, name=f"{prefix}.visible_indices", dtype=np.dtype("<i8"),
        shape=(rows, tokens),
    )
    result = []
    for row in range(rows):
        visible = mask_array[row]
        states = np.ascontiguousarray(hidden_array[row, visible])
        p4 = np.ascontiguousarray(vectors_array[row][:, visible].T)
        token_ids = np.ascontiguousarray(ids[row, visible])
        if (
            (len(states) == 0 and not allow_empty)
            or np.any(token_ids < 0)
            or len(set(token_ids.astype(int).tolist())) != len(token_ids)
        ):
            raise ValueError(f"HCWDL-RKD {prefix} visible-token metadata differs")
        result.append((states, p4, token_ids))
    return result


def _cloud_sketch(
    states: np.ndarray, p4: np.ndarray, token_ids: np.ndarray,
    *, token_resources: SpectralKernelResources,
    relation_resources: SpectralKernelResources,
) -> tuple[np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray, np.ndarray]:
    import torch

    if states.dtype != np.float32 or p4.dtype != np.float32 or token_ids.dtype != np.int64:
        raise ValueError("HCWDL-RKD target cloud dtype differs")
    rows = len(states)
    if rows <= 0:
        raise ValueError("HCWDL-RKD target cloud is empty")
    state_tensor = torch.from_numpy(np.ascontiguousarray(states[None]))
    vector_tensor = torch.from_numpy(np.ascontiguousarray(p4.T[None]))
    mask_tensor = torch.ones((1, rows), dtype=torch.bool)
    identity_tensor = torch.from_numpy(np.ascontiguousarray(token_ids[None]))
    token_targets = build_ordinary_token_targets(
        state_tensor, vector_tensor, mask_tensor, token_resources,
    )
    relation_targets = build_teacher_relation_targets(
        state_tensor, vector_tensor, mask_tensor, identity_tensor,
        relation_resources,
    )
    pt = torch.sqrt(
        vector_tensor[:, 0].float().square() + vector_tensor[:, 1].float().square()
    )
    if not bool(torch.isfinite(pt).all()) or not bool((pt > 0).all()):
        raise ValueError("HCWDL-RKD visible target four-vectors have invalid pT")
    token_mean_array = np.ascontiguousarray(
        token_targets.means[0].cpu().numpy().astype(np.float32, copy=False),
    )
    relation_means = np.ascontiguousarray(
        relation_targets.means[0, 0].cpu().numpy().astype(np.float32, copy=False),
    )
    pair_counts = np.ascontiguousarray(
        relation_targets.pair_counts[0, 0].cpu().numpy().astype(np.uint16),
    )
    effective_samples = np.ascontiguousarray(
        relation_targets.effective_sample_sizes[0, 0].cpu().numpy().astype(np.float32),
    )
    eligible = np.ascontiguousarray(
        relation_targets.eligible[0, 0].cpu().numpy().astype(np.uint8),
    )
    return (
        token_mean_array, relation_means, rows, float(pt.sum(dtype=torch.float32).cpu()),
        pair_counts, effective_samples, eligible,
    )


def _hash_array_registry(namespace: str, arrays: Mapping[str, np.ndarray]) -> str:
    return canonical_sha256({
        name: logical_array_sha256(f"{namespace}.{name}", np.asarray(value))
        for name, value in sorted(arrays.items())
    })


def _ordinary_batch_targets(
    batch: TargetForwardBatch, surface: Any, *, token_resources: SpectralKernelResources,
    relation_resources: SpectralKernelResources,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    rows = batch.rows
    logits = _float32(_surface_field(surface, "logits"), name="logits", shape=(rows, 15))
    jet = _float32(
        _surface_field(surface, "jet_penultimate"), name="jet_penultimate",
        shape=(rows, 128),
    )
    clouds = _visible_cloud(
        _surface_field(surface, "particle_block_2"),
        _surface_field(surface, "particle_mask"),
        _surface_field(surface, "vectors"),
        _surface_field(surface, "visible_indices"), rows=rows, prefix="ordinary",
    )
    token_slots = _as_numpy(
        _surface_field(surface, "particle_mask"), name="ordinary.particle_mask",
    ).shape[1]
    family_codes = _integer_array(
        _surface_field(surface, "family_codes"), name="ordinary.family_codes",
        dtype=np.dtype("i1"), shape=(rows, token_slots),
    )
    arrays = {
        name: np.zeros(shape, dtype=dtype)
        for name, (dtype, shape) in target_array_schema(ORDINARY_BANK, rows).items()
    }
    for name in ("source_file_id", "source_entry", "identity_digest", "label"):
        arrays[name][...] = np.asarray(getattr(batch, name))
    arrays["logits"][...] = logits
    arrays["jet_penultimate"][...] = jet
    for row, (states, p4, token_ids) in enumerate(clouds):
        token, relation, count, pt_sum, pair_count, ess, eligible = _cloud_sketch(
            states, p4, token_ids, token_resources=token_resources,
            relation_resources=relation_resources,
        )
        arrays["token_kernel_mean"][row] = token
        arrays["relation_kernel_mean"][row] = relation
        arrays["token_family_eligibility"][row, 0] = 1
        arrays["relation_eligibility"][row, 0] = eligible
        arrays["token_count"][row, 0] = count
        arrays["token_scalar_pt_sum"][row, 0] = pt_sum
        arrays["relation_pair_count"][row, 0] = pair_count
        arrays["relation_effective_sample"][row, 0] = ess
        arrays["family_reason_counts"][row, 0] = count
    surface_arrays = {
        "particle_block_2": _as_numpy(_surface_field(surface, "particle_block_2"), name="surface"),
        "particle_mask": _as_numpy(_surface_field(surface, "particle_mask"), name="mask"),
        "vectors": _as_numpy(_surface_field(surface, "vectors"), name="vectors"),
        "visible_indices": _as_numpy(_surface_field(surface, "visible_indices"), name="ids"),
        "family_codes": family_codes,
        "jet_penultimate": jet,
    }
    sketch_arrays = {
        name: value for name, value in arrays.items()
        if name not in {
            "source_file_id", "source_entry", "identity_digest", "label", "logits",
            "jet_penultimate",
        }
    }
    validate_target_arrays(arrays, bank_kind=ORDINARY_BANK, expected_rows=rows)
    return arrays, {
        "logits": logical_array_sha256("target_batch.logits", logits),
        "surfaces": _hash_array_registry("target_batch.surfaces", surface_arrays),
        "sketches": _hash_array_registry("target_batch.sketches", sketch_arrays),
    }


def _toff_batch_targets(
    batch: TargetForwardBatch, surface: Any, *, token_resources: SpectralKernelResources,
    relation_resources: SpectralKernelResources,
    companion_reason_counts: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    rows = batch.rows
    reason_array = np.ascontiguousarray(companion_reason_counts)
    if reason_array.dtype != np.uint16 or reason_array.shape != (rows, 6):
        raise ValueError("HCWDL-RKD companion-HLT reason-count audit differs")
    logits = _float32(_surface_field(surface, "logits"), name="logits", shape=(rows, 15))
    jet = _float32(
        _surface_field(surface, "offline_jet_penultimate"), name="offline_jet_penultimate",
        shape=(rows, 128),
    )
    families = {
        "charged": _visible_cloud(
            _surface_field(surface, "charged_particle_block_2"),
            _surface_field(surface, "charged_mask"),
            _surface_field(surface, "charged_vectors"),
            _surface_field(surface, "charged_visible_indices"), rows=rows, prefix="charged",
            allow_empty=True,
        ),
        "neutral": _visible_cloud(
            _surface_field(surface, "neutral_particle_block_2"),
            _surface_field(surface, "neutral_mask"),
            _surface_field(surface, "neutral_vectors"),
            _surface_field(surface, "neutral_visible_indices"), rows=rows, prefix="neutral",
            allow_empty=True,
        ),
    }
    arrays = {
        name: np.zeros(shape, dtype=dtype)
        for name, (dtype, shape) in target_array_schema(TOFF_BANK, rows).items()
    }
    for name in ("source_file_id", "source_entry", "identity_digest", "label"):
        arrays[name][...] = np.asarray(getattr(batch, name))
    arrays["logits"][...] = logits
    arrays["jet_penultimate"][...] = jet
    arrays["family_reason_counts"][...] = reason_array
    for family_index, family in enumerate(("charged", "neutral")):
        for row, (states, p4, token_ids) in enumerate(families[family]):
            if len(states) == 0:
                continue
            token, relation, count, pt_sum, pair_count, ess, eligible = _cloud_sketch(
                states, p4, token_ids, token_resources=token_resources,
                relation_resources=relation_resources,
            )
            arrays[f"token_kernel_mean_{family}"][row] = token
            arrays[f"relation_kernel_mean_{family}"][row] = relation
            arrays["token_family_eligibility"][row, family_index] = 1
            arrays["relation_eligibility"][row, family_index] = eligible
            arrays["token_count"][row, family_index] = count
            arrays["token_scalar_pt_sum"][row, family_index] = pt_sum
            arrays["relation_pair_count"][row, family_index] = pair_count
            arrays["relation_effective_sample"][row, family_index] = ess
    surface_arrays = {
        name: _as_numpy(_surface_field(surface, name), name=name)
        for name in (
            "charged_particle_block_2", "neutral_particle_block_2", "charged_mask",
            "neutral_mask", "charged_vectors", "neutral_vectors",
            "charged_visible_indices", "neutral_visible_indices",
        )
    }
    surface_arrays["offline_jet_penultimate"] = jet
    sketch_arrays = {
        name: value for name, value in arrays.items()
        if name not in {
            "source_file_id", "source_entry", "identity_digest", "label", "logits",
            "jet_penultimate",
        }
    }
    validate_target_arrays(arrays, bank_kind=TOFF_BANK, expected_rows=rows)
    return arrays, {
        "logits": logical_array_sha256("target_batch.logits", logits),
        "surfaces": _hash_array_registry("target_batch.surfaces", surface_arrays),
        "sketches": _hash_array_registry("target_batch.sketches", sketch_arrays),
    }


def _execute_batch(
    batch: TargetForwardBatch, teacher_forward: TeacherSurfaceForward, *, bank_kind: str,
    token_resources: SpectralKernelResources, relation_resources: SpectralKernelResources,
    allowed_input_fields: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    import torch

    model_inputs = _teacher_model_inputs(batch, allowed_fields=allowed_input_fields)
    # Companion-HLT charge/PID bookkeeping is not a teacher input.  Validate
    # and classify it before opening the privileged TOFF forward so malformed
    # raw data fail closed without consuming privileged model work.
    companion_reason_counts = (
        _companion_hlt_reason_counts(batch) if bank_kind == TOFF_BANK else None
    )
    with torch.inference_mode():
        surface = teacher_forward(model_inputs)
    if bank_kind == ORDINARY_BANK:
        return _ordinary_batch_targets(
            batch, surface, token_resources=token_resources,
            relation_resources=relation_resources,
        )
    if bank_kind == TOFF_BANK:
        return _toff_batch_targets(
            batch, surface, token_resources=token_resources,
            relation_resources=relation_resources,
            companion_reason_counts=companion_reason_counts,
        )
    raise ValueError("unknown HCWDL-RKD target-bank kind")


def _batch_audit(
    batch: TargetForwardBatch, *, batch_index: int, hashes: Mapping[str, str],
) -> dict[str, Any]:
    for name in _HASHED_BATCH_ARRAYS:
        require_sha256(hashes[name], name=f"target canonical batch {name}")
    return {
        "source_partition": batch.source_partition,
        "batch_index": batch_index,
        "source_file_id": int(batch.source_file_id[0]),
        "rows": batch.rows,
        "first_source_entry": int(batch.source_entry[0]),
        "last_source_entry": int(batch.source_entry[-1]),
        "first_identity_digest": bytes(batch.identity_digest[0]).hex(),
        "last_identity_digest": bytes(batch.identity_digest[-1]).hex(),
        **dict(hashes),
    }


def _partition_hash(records: list[Mapping[str, Any]]) -> str:
    return canonical_sha256([{key: row[key] for key in (
        "source_partition", "batch_index", "source_file_id", "rows",
        "first_source_entry", "last_source_entry", "first_identity_digest",
        "last_identity_digest",
    )} for row in records])


def _iter_batches_with_final(
    factory: BatchFactory,
) -> Iterable[tuple[int, TargetForwardBatch, bool]]:
    iterator = iter(factory())
    try:
        current = next(iterator)
    except StopIteration as error:
        raise ValueError("HCWDL-RKD target partition has no canonical batch") from error
    index = 0
    while True:
        try:
            following = next(iterator)
        except StopIteration:
            yield index, current, True
            return
        yield index, current, False
        current = following
        index += 1


def _run_partition_spec(
    *,
    partition: str,
    specification: Mapping[str, Any],
    bank_kind: str,
    factory: BatchFactory,
    teacher_forward: TeacherSurfaceForward, token_resources: SpectralKernelResources,
    relation_resources: SpectralKernelResources, allowed_input_fields: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, Any], int]:
    parts: dict[str, list[np.ndarray]] = {}
    audits = []
    previous_entry: int | None = None
    for index, batch, is_final in _iter_batches_with_final(factory):
        _validate_batch_identity(
            batch, partition=partition,
            expected_source_file_id=int(specification["source_file_id"]),
        )
        if not is_final and batch.rows != TARGET_FORWARD_BATCH_SIZE:
            raise ValueError("HCWDL-RKD non-final canonical target batch is short")
        if previous_entry is not None and int(batch.source_entry[0]) <= previous_entry:
            raise ValueError("HCWDL-RKD canonical target batches overlap/reorder entries")
        previous_entry = int(batch.source_entry[-1])
        arrays, hashes = _execute_batch(
            batch, teacher_forward, bank_kind=bank_kind,
            token_resources=token_resources, relation_resources=relation_resources,
            allowed_input_fields=allowed_input_fields,
        )
        for name, value in arrays.items():
            parts.setdefault(name, []).append(value)
        audits.append(_batch_audit(batch, batch_index=index, hashes=hashes))
        del arrays, hashes
    arrays = {
        name: np.ascontiguousarray(np.concatenate(chunks, axis=0))
        for name, chunks in parts.items()
    }
    if len(arrays["label"]) != int(specification["rows"]):
        raise ValueError("HCWDL-RKD target partition canonical batches do not conserve rows")
    runtime_audit = {
        "source_partition": partition,
        "teacher_forward_calls": len(audits),
        "batch_partition_sha256": _partition_hash(audits),
        "canonical_batches": audits,
    }
    return arrays, runtime_audit, len(audits)


def _run_partition(
    context: TargetGenerationContext, partition: str, factory: BatchFactory,
    teacher_forward: TeacherSurfaceForward, *, token_resources: SpectralKernelResources,
    relation_resources: SpectralKernelResources, allowed_input_fields: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, Any], int]:
    return _run_partition_spec(
        partition=partition,
        specification=context.build_intent["payload"]["partitions"][partition],
        bank_kind=context.bank_kind,
        factory=factory,
        teacher_forward=teacher_forward,
        token_resources=token_resources,
        relation_resources=relation_resources,
        allowed_input_fields=allowed_input_fields,
    )


def _sentinels(records: list[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    if not records:
        raise ValueError("HCWDL-RKD target runtime has no canonical batches")
    chosen = {"first": records[0], "middle": records[len(records) // 2], "last": records[-1]}
    return {
        position: {name: str(record[name]) for name in _HASHED_BATCH_ARRAYS}
        for position, record in chosen.items()
    }


def _validate_runtime_environment(
    value: Mapping[str, Any], *, context: TargetGenerationContext,
) -> dict[str, Any]:
    required = {"producer", "device", "precision", "determinism"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("HCWDL-RKD target runtime environment fields differ")
    spec = context.forward_spec["payload"]
    if (
        value["producer"] != spec["producer"]
        or value["precision"] != spec["precision"]
        or value["determinism"] != spec["determinism"]
    ):
        raise ValueError("HCWDL-RKD target runtime environment differs from forward spec")
    device = dict(value["device"])
    gpu_uuid = device.pop("gpu_uuid", None)
    if device != spec["device"] or not isinstance(gpu_uuid, str) or not gpu_uuid:
        raise ValueError("HCWDL-RKD target runtime device/UUID differs from forward spec")
    return {**dict(value), "device": {**device, "gpu_uuid": gpu_uuid}}


def _validate_kernel_resources(
    context: TargetGenerationContext, token: SpectralKernelResources,
    relation: SpectralKernelResources,
) -> None:
    from .hcwdl_representation_kernels import (
        SpectralResourceBundle, spectral_resource_logical_hashes,
    )

    actual = spectral_resource_logical_hashes(
        SpectralResourceBundle(token=token, relation=relation),
    )
    if set(actual) != set(KERNEL_RESOURCE_NAMES) or actual != context.forward_spec[
        "payload"
    ]["teacher"]["kernel_array_logical_hashes"]:
        raise ValueError("HCWDL-RKD loaded spectral arrays differ from the forward spec")


def replay_prior_target_sentinels(
    context: TargetGenerationContext, *, partition_batches: Mapping[str, BatchFactory],
    teacher_forward: TeacherSurfaceForward, token_resources: SpectralKernelResources,
    relation_resources: SpectralKernelResources,
    prior_execution_attestation: Mapping[str, Any],
    allowed_input_fields: tuple[str, ...],
) -> int:
    """Dry-replay first/middle/last canonical batches before reconstruction."""

    prior = prior_execution_attestation.get("payload", {})
    validate_versioned_artifact(
        prior_execution_attestation,
        expected_contract=TARGET_EXECUTION_ATTESTATION_CONTRACT,
        required_payload_keys=("sentinel_hashes", "logical_target_sha256"),
    )
    expected = prior.get("sentinel_hashes")
    if not isinstance(expected, Mapping):
        raise ValueError("HCWDL-RKD prior target attestation lacks sentinel hashes")
    partition_specs = context.build_intent["payload"]["partitions"]
    total_batches = sum(
        math.ceil(int(specification["rows"]) / TARGET_FORWARD_BATCH_SIZE)
        for specification in partition_specs.values()
    )
    requested = {
        "first": 0, "middle": total_batches // 2, "last": total_batches - 1,
    }
    requested_by_index: dict[int, list[str]] = {}
    for name, index in requested.items():
        requested_by_index.setdefault(index, []).append(name)
    actual: dict[str, Mapping[str, str]] = {}
    global_index = 0
    for partition, specification in partition_specs.items():
        if partition not in partition_batches:
            raise ValueError("HCWDL-RKD sentinel replay lacks a source partition")
        seen_rows = 0
        for index, batch, is_final in _iter_batches_with_final(
            partition_batches[partition],
        ):
            _validate_batch_identity(
                batch, partition=partition,
                expected_source_file_id=int(specification["source_file_id"]),
            )
            if not is_final and batch.rows != TARGET_FORWARD_BATCH_SIZE:
                raise ValueError("HCWDL-RKD sentinel replay batch partition differs")
            seen_rows += batch.rows
            if global_index in requested_by_index:
                _, hashes = _execute_batch(
                    batch, teacher_forward, bank_kind=context.bank_kind,
                    token_resources=token_resources, relation_resources=relation_resources,
                    allowed_input_fields=allowed_input_fields,
                )
                for name in requested_by_index[global_index]:
                    actual[name] = hashes
            global_index += 1
        if seen_rows != int(specification["rows"]):
            raise ValueError("HCWDL-RKD sentinel replay partition rows differ")
    if global_index != total_batches or set(actual) != set(requested):
        raise ValueError("HCWDL-RKD sentinel replay canonical batch count differs")
    if actual != expected:
        raise ValueError("HCWDL-RKD target reconstruction sentinel replay differs")
    return len(requested_by_index)


def prepare_target_generation_in_memory(
    *,
    bank_kind: str,
    partition_batches: Mapping[str, BatchFactory],
    partition_specs: Mapping[str, Mapping[str, int]],
    teacher_forward: TeacherSurfaceForward,
    token_resources: SpectralKernelResources,
    relation_resources: SpectralKernelResources,
    teacher_model: Any,
    allowed_input_fields: Sequence[str],
) -> PreparedTargetGeneration:
    """Consume each source and teacher batch exactly once into compact RAM shards.

    This is the fresh-build path.  It deliberately retains only the final
    1,935/3,727-value compact target rows—not particle inputs or latent token
    surfaces—while population hashes are derived from those same rows.  The
    caller can therefore publish the immutable build intent after learning
    exact population hashes without reopening ROOT or forwarding the teacher.
    """

    started = time.perf_counter()
    if bank_kind not in {ORDINARY_BANK, TOFF_BANK}:
        raise ValueError("unknown HCWDL-RKD prepared target-bank kind")
    if token_resources.kind != "token" or token_resources.total_features != 1024:
        raise ValueError("HCWDL-RKD token spectral resources differ")
    if relation_resources.kind != "relation" or relation_resources.total_features != 256:
        raise ValueError("HCWDL-RKD relation spectral resources differ")
    if set(partition_batches) != set(partition_specs) or not partition_specs:
        raise ValueError("HCWDL-RKD prepared target partition registry differs")
    _validate_teacher_model_runtime(teacher_model)
    _validate_process_backend()
    fields = tuple(str(name) for name in allowed_input_fields)
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("HCWDL-RKD prepared teacher input fields differ")

    prepared: dict[str, PreparedTargetPartition] = {}
    identity_parts: list[np.ndarray] = []
    class_counts = np.zeros(15, dtype=np.int64)
    population = TargetPopulationHasher()
    canonical_batches: list[Mapping[str, Any]] = []
    forward_calls = 0
    normalized_specs: dict[str, dict[str, int]] = {}
    for partition in partition_specs:
        raw_spec = partition_specs[partition]
        if set(raw_spec) != {"rows", "source_file_id"}:
            raise ValueError("HCWDL-RKD prepared partition specification differs")
        specification = {
            "rows": int(raw_spec["rows"]),
            "source_file_id": int(raw_spec["source_file_id"]),
        }
        if specification["rows"] <= 0 or not 0 <= specification[
            "source_file_id"
        ] < 2**32:
            raise ValueError("HCWDL-RKD prepared partition values differ")
        arrays, audit, calls = _run_partition_spec(
            partition=partition,
            specification=specification,
            bank_kind=bank_kind,
            factory=partition_batches[partition],
            teacher_forward=teacher_forward,
            token_resources=token_resources,
            relation_resources=relation_resources,
            allowed_input_fields=fields,
        )
        normalized_specs[partition] = specification
        identity_parts.append(np.ascontiguousarray(arrays["identity_digest"]))
        class_counts += np.bincount(arrays["label"], minlength=15)
        population.update(
            source_file_id=arrays["source_file_id"],
            source_entry=arrays["source_entry"],
            identity_digest=arrays["identity_digest"],
            label=arrays["label"],
        )
        canonical_batches.extend(audit["canonical_batches"])
        forward_calls += calls
        prepared[partition] = PreparedTargetPartition(
            arrays=MappingProxyType({
                name: np.ascontiguousarray(value)
                for name, value in arrays.items()
            }),
            runtime_audit=MappingProxyType(dict(audit)),
            teacher_forward_calls=calls,
        )
    identities = np.ascontiguousarray(np.concatenate(identity_parts, axis=0))
    return PreparedTargetGeneration(
        bank_kind=bank_kind,
        partitions=MappingProxyType(prepared),
        partition_specs=MappingProxyType(normalized_specs),
        class_counts=tuple(int(value) for value in class_counts),
        identity_order_sha256=identity_order_sha256(identities),
        identity_set_sha256=identity_set_sha256(identities),
        population_rows_sha256=population.hexdigest(),
        canonical_batches=tuple(canonical_batches),
        teacher_forward_calls=forward_calls,
        construction_seconds=time.perf_counter() - started,
    )


def build_target_generation_from_prepared(
    context: TargetGenerationContext,
    *,
    prepared: PreparedTargetGeneration,
    token_resources: SpectralKernelResources,
    relation_resources: SpectralKernelResources,
    runtime_environment: Mapping[str, Any],
    prior_execution_attestation: Mapping[str, Any] | None = None,
    failure_hook: FailureHook | None = None,
) -> TargetRuntimeResult:
    """Publish a compact one-pass preparation without reopening any source."""

    _validate_kernel_resources(context, token_resources, relation_resources)
    intent = context.build_intent["payload"]
    if (
        prepared.bank_kind != context.bank_kind
        or dict(prepared.partition_specs) != intent["partitions"]
        or list(prepared.class_counts) != intent["expected_class_counts"]
        or prepared.identity_order_sha256
        != intent["expected_identity_order_sha256"]
        or prepared.identity_set_sha256 != intent["expected_identity_set_sha256"]
        or prepared.population_rows_sha256
        != intent["expected_population_rows_sha256"]
    ):
        raise ValueError("HCWDL-RKD prepared target population differs from build intent")
    if context.committed_directory.is_dir():
        manifest = validate_target_generation(context.bank_root, context.generation_id)
        attestation = load_json(
            context.committed_directory / "target_execution_attestation.json",
        )
        return TargetRuntimeResult(
            manifest=manifest, execution_attestation=attestation,
            teacher_forward_calls=0, sentinel_replay_calls=0,
            published_partitions=(), reused_partitions=tuple(intent["partitions"]),
        )
    _validate_process_backend()
    environment = _validate_runtime_environment(runtime_environment, context=context)
    publication_started = time.perf_counter()
    if failure_hook is not None:
        failure_hook("after_target_build_intent")
    replay_calls = 0
    if prior_execution_attestation is not None:
        validate_versioned_artifact(
            prior_execution_attestation,
            expected_contract=TARGET_EXECUTION_ATTESTATION_CONTRACT,
            required_payload_keys=("sentinel_hashes", "logical_target_sha256"),
        )
        if prior_execution_attestation["payload"]["logical_target_sha256"] != (
            context.logical_bank["content_hash"]
        ) or prior_execution_attestation["payload"]["sentinel_hashes"] != _sentinels(
            list(prepared.canonical_batches)
        ):
            raise ValueError("HCWDL-RKD reconstructed target sentinels differ")
        replay_calls = len({
            0,
            len(prepared.canonical_batches) // 2,
            len(prepared.canonical_batches) - 1,
        })

    published: list[str] = []
    reused: list[str] = []
    sidecars = []
    for partition in intent["partitions"]:
        item = prepared.partitions[partition]
        try:
            sidecar, _ = load_staged_target_shard(context, partition=partition)
            if sidecar["payload"]["target_runtime_audit"] != dict(
                item.runtime_audit
            ):
                raise ValueError("staged target audit differs from one-pass preparation")
            reused.append(partition)
        except (
            FileNotFoundError, KeyError, TypeError, ValueError,
            FloatingPointError, EOFError, OSError, zipfile.BadZipFile,
        ):
            stage_target_shard(
                context,
                partition=partition,
                arrays=item.arrays,
                runtime_audit=item.runtime_audit,
                failure_hook=failure_hook,
            )
            sidecar, _ = load_staged_target_shard(context, partition=partition)
            published.append(partition)
        sidecars.append(sidecar)
    canonical_batches = [
        row for sidecar in sidecars
        for row in sidecar["payload"]["target_runtime_audit"]["canonical_batches"]
    ]
    if canonical_batches != list(prepared.canonical_batches):
        raise ValueError("published target batch audit differs from one-pass preparation")
    execution_facts = {
        **environment,
        "construction_seconds": (
            prepared.construction_seconds
            + (time.perf_counter() - publication_started)
        ),
        "batch_count": len(canonical_batches),
        "batch_partition_sha256": _partition_hash(canonical_batches),
        "sentinel_hashes": _sentinels(canonical_batches),
    }
    manifest = finalize_target_generation(
        context,
        execution_facts=execution_facts,
        prior_execution_attestation=prior_execution_attestation,
        failure_hook=failure_hook,
    )
    attestation = load_json(
        context.committed_directory / "target_execution_attestation.json",
    )
    return TargetRuntimeResult(
        manifest=manifest,
        execution_attestation=attestation,
        teacher_forward_calls=prepared.teacher_forward_calls,
        sentinel_replay_calls=replay_calls,
        published_partitions=tuple(published),
        reused_partitions=tuple(reused),
    )


def build_target_generation_from_teacher(
    context: TargetGenerationContext, *, partition_batches: Mapping[str, BatchFactory],
    teacher_forward: TeacherSurfaceForward, token_resources: SpectralKernelResources,
    relation_resources: SpectralKernelResources, runtime_environment: Mapping[str, Any],
    teacher_model: Any | None = None,
    prior_execution_attestation: Mapping[str, Any] | None = None,
    failure_hook: FailureHook | None = None,
) -> TargetRuntimeResult:
    """Build/reuse every source shard and atomically commit one target generation."""

    construction_started = time.perf_counter()
    if token_resources.kind != "token" or token_resources.total_features != 1024:
        raise ValueError("HCWDL-RKD token spectral resources differ")
    if relation_resources.kind != "relation" or relation_resources.total_features != 256:
        raise ValueError("HCWDL-RKD relation spectral resources differ")
    _validate_kernel_resources(context, token_resources, relation_resources)
    partitions = context.build_intent["payload"]["partitions"]
    if set(partition_batches) != set(partitions):
        raise ValueError("HCWDL-RKD target runtime partition factories differ")
    if context.committed_directory.is_dir():
        manifest = validate_target_generation(context.bank_root, context.generation_id)
        attestation = load_json(
            context.committed_directory / "target_execution_attestation.json",
        )
        return TargetRuntimeResult(
            manifest=manifest, execution_attestation=attestation,
            teacher_forward_calls=0, sentinel_replay_calls=0,
            published_partitions=(), reused_partitions=tuple(partitions),
        )
    _validate_teacher_model_runtime(teacher_model)
    _validate_process_backend()
    environment = _validate_runtime_environment(runtime_environment, context=context)
    allowed_input_fields = tuple(
        context.forward_spec["payload"]["implementation"]["teacher_input_fields"]
    )
    if failure_hook is not None:
        failure_hook("after_target_build_intent")
    replay_calls = 0
    if prior_execution_attestation is not None:
        replay_calls = replay_prior_target_sentinels(
            context, partition_batches=partition_batches, teacher_forward=teacher_forward,
            token_resources=token_resources, relation_resources=relation_resources,
            prior_execution_attestation=prior_execution_attestation,
            allowed_input_fields=allowed_input_fields,
        )
    published = []
    reused = []
    forward_calls = 0
    sidecars = []
    for partition in partitions:
        try:
            sidecar, _ = load_staged_target_shard(context, partition=partition)
            reused.append(partition)
        except (
            FileNotFoundError, KeyError, TypeError, ValueError,
            FloatingPointError, EOFError, OSError, zipfile.BadZipFile,
        ):
            arrays, audit, calls = _run_partition(
                context, partition, partition_batches[partition], teacher_forward,
                token_resources=token_resources, relation_resources=relation_resources,
                allowed_input_fields=allowed_input_fields,
            )
            stage_target_shard(
                context, partition=partition, arrays=arrays, runtime_audit=audit,
                failure_hook=failure_hook,
            )
            sidecar, _ = load_staged_target_shard(context, partition=partition)
            published.append(partition)
            forward_calls += calls
            del arrays
        sidecars.append(sidecar)
    canonical_batches = [
        row for sidecar in sidecars
        for row in sidecar["payload"]["target_runtime_audit"]["canonical_batches"]
    ]
    execution_facts = {
        **environment,
        "construction_seconds": time.perf_counter() - construction_started,
        "batch_count": len(canonical_batches),
        "batch_partition_sha256": _partition_hash(canonical_batches),
        "sentinel_hashes": _sentinels(canonical_batches),
    }
    manifest = finalize_target_generation(
        context, execution_facts=execution_facts,
        prior_execution_attestation=prior_execution_attestation,
        failure_hook=failure_hook,
    )
    attestation = load_json(
        context.committed_directory / "target_execution_attestation.json",
    )
    return TargetRuntimeResult(
        manifest=manifest, execution_attestation=attestation,
        teacher_forward_calls=forward_calls, sentinel_replay_calls=replay_calls,
        published_partitions=tuple(published), reused_partitions=tuple(reused),
    )


def build_predecessor_logit_cache(
    *, partition_batches: Mapping[str, BatchFactory],
    predecessor_forward: Callable[[TargetForwardBatch], Any],
    release_predecessor: Callable[[], None], expected_rows: int,
    expected_identity_order_sha256: str, expected_identity_set_sha256: str,
    predecessor_checkpoint_logical_sha256: str,
    teacher_input_fields: Sequence[str],
    predecessor_model: Any,
) -> PredecessorLogitBank:
    """Run one HLT predecessor pass, materialize logits in RAM, then release it."""

    if isinstance(expected_rows, bool) or expected_rows <= 0:
        raise ValueError("HCWDL-RKD predecessor cache row count differs")
    _validate_teacher_model_runtime(predecessor_model)
    _validate_process_backend()
    identities = []
    logits = []
    previous: tuple[int, int] | None = None
    try:
        import torch

        with torch.inference_mode():
            for partition in sorted(partition_batches):
                for index, batch, is_final in _iter_batches_with_final(
                    partition_batches[partition],
                ):
                    if not is_final and batch.rows != TARGET_FORWARD_BATCH_SIZE:
                        raise ValueError("HCWDL-RKD predecessor non-final batch is short")
                    _validate_batch_identity(
                        batch, partition=partition,
                        expected_source_file_id=int(batch.source_file_id[0]),
                    )
                    first = (int(batch.source_file_id[0]), int(batch.source_entry[0]))
                    last = (int(batch.source_file_id[-1]), int(batch.source_entry[-1]))
                    if previous is not None and first <= previous:
                        raise ValueError("HCWDL-RKD predecessor cache identities reorder")
                    previous = last
                    allowed = tuple(str(name) for name in teacher_input_fields)
                    if not allowed or tuple(sorted(allowed)) != allowed:
                        raise ValueError("HCWDL-RKD predecessor teacher input allow-list differs")
                    output = predecessor_forward(
                        _teacher_model_inputs(batch, allowed_fields=allowed),
                    )
                    if hasattr(output, "logits") or (
                        isinstance(output, Mapping) and "logits" in output
                    ):
                        output = _surface_field(output, "logits")
                    logits.append(_float32(output, name="predecessor_logits", shape=(batch.rows, 15)))
                    identities.append(np.ascontiguousarray(batch.identity_digest))
    finally:
        release_predecessor()
    identity_array = np.ascontiguousarray(np.concatenate(identities, axis=0))
    logit_array = np.ascontiguousarray(np.concatenate(logits, axis=0))
    if len(identity_array) != expected_rows:
        raise ValueError("HCWDL-RKD predecessor cache row coverage differs")
    identity_hex = [bytes(row).hex() for row in identity_array]
    order_hash = canonical_sha256(identity_hex)
    set_hash = canonical_sha256(sorted(identity_hex))
    if (
        order_hash != require_sha256(
            expected_identity_order_sha256, name="predecessor expected identity-order hash",
        )
        or set_hash != require_sha256(
            expected_identity_set_sha256, name="predecessor expected identity-set hash",
        )
        or len(set(identity_hex)) != len(identity_hex)
    ):
        raise ValueError("HCWDL-RKD predecessor cache identity population differs")
    logits_hash = logical_array_sha256("predecessor_logits", logit_array)
    logical = canonical_sha256({
        "predecessor_checkpoint_logical_sha256": require_sha256(
            predecessor_checkpoint_logical_sha256,
            name="predecessor checkpoint logical SHA-256",
        ),
        "identity_order_sha256": order_hash,
        "identity_set_sha256": set_hash,
        "logits_sha256": logits_hash,
    })
    lookup = {bytes(row): index for index, row in enumerate(identity_array)}
    return PredecessorLogitBank(
        identity_array, logit_array, order_hash, set_hash, logits_hash, logical, lookup,
    )


__all__ = [
    "BatchFactory", "PredecessorLogitBank", "PreparedTargetGeneration",
    "PreparedTargetPartition", "TargetForwardBatch", "TargetRuntimeResult",
    "TeacherModelInputs",
    "build_predecessor_logit_cache", "build_target_generation_from_prepared",
    "build_target_generation_from_teacher", "prepare_target_generation_in_memory",
    "replay_prior_target_sentinels",
]

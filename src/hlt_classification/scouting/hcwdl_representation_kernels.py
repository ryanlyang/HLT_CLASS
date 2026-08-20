"""Frozen finite spectral kernels for matching-free HCWDL representation KD."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_json_bytes,
    canonical_sha256,
    deterministic_npz_bytes,
    load_npz_arrays,
)

from .hcwdl_representation_artifacts import (
    CommittedBinaryEnvelope,
    publish_binary_envelope,
    validate_binary_envelope,
)
from .hcwdl_representation_contracts import (
    KERNEL_RESOURCES_CONTRACT,
    logical_array_sha256,
)


RFF_CONTRACT: Final = "HCWDL_REP_RFF/v1"
RFF_MASTER_SEED: Final = 20260808
TOKEN_BANDWIDTHS: Final = (0.10, 0.25, 0.50, 1.00)
RELATION_BANDWIDTHS: Final = (0.05, 0.10, 0.20, 0.40)
TOKEN_FEATURES_PER_BANDWIDTH: Final = 256
RELATION_FEATURES_PER_BANDWIDTH: Final = 64
K_TOKEN: Final = 1024
K_RELATION: Final = 256


# Frozen resources are immutable for the lifetime of a worker.  Keep one
# device copy per concrete resource object/device instead of copying four
# omega/phase blocks from NumPy for every jet (and, for RREL, every stratum).
# The object itself is retained in the value so an id cannot be recycled into
# a stale entry.  This is a process-local execution cache, never artifact
# lineage or scientific state.
_DEVICE_BLOCK_CACHE: dict[
    tuple[int, str, int | None],
    tuple["SpectralKernelResources", tuple[tuple[object, object], ...]],
] = {}

_TOKEN_RESOURCE_NAMES: Final = (
    "token_rbf_sigma_0p10",
    "token_rbf_sigma_0p25",
    "token_rbf_sigma_0p50",
    "token_rbf_sigma_1p00",
)
_RELATION_RESOURCE_NAMES: Final = (
    "relation_rbf_sigma_0p05",
    "relation_rbf_sigma_0p10",
    "relation_rbf_sigma_0p20",
    "relation_rbf_sigma_0p40",
)


@dataclass(frozen=True)
class SpectralBlock:
    resource_name: str
    bandwidth_index: int
    bandwidth: float
    seed_payload: dict[str, object]
    seed_sha256: str
    seed64: int
    omega: np.ndarray
    phase: np.ndarray

    def __post_init__(self) -> None:
        omega = np.asarray(self.omega)
        phase = np.asarray(self.phase)
        if omega.dtype != np.float32 or phase.dtype != np.float32:
            raise ValueError("spectral resource arrays must be FP32")
        if not omega.flags.c_contiguous or not phase.flags.c_contiguous:
            raise ValueError("spectral resource arrays must be C contiguous")
        if omega.ndim != 2 or phase.shape != (omega.shape[0],):
            raise ValueError("spectral resource shapes differ")
        if not np.isfinite(omega).all() or not np.isfinite(phase).all():
            raise ValueError("spectral resource arrays are nonfinite")
        if self.seed_sha256 != canonical_sha256(self.seed_payload):
            raise ValueError("spectral seed payload hash differs")
        if self.seed64 != int.from_bytes(bytes.fromhex(self.seed_sha256)[:8], "big"):
            raise ValueError("spectral 64-bit seed differs")

    @property
    def logical_hashes(self) -> dict[str, str]:
        return {
            "omega": logical_array_sha256(f"{self.resource_name}.omega", self.omega),
            "phase": logical_array_sha256(f"{self.resource_name}.phase", self.phase),
        }


@dataclass(frozen=True)
class SpectralKernelResources:
    kind: Literal["token", "relation"]
    blocks: tuple[SpectralBlock, ...]
    numpy_version: str

    def __post_init__(self) -> None:
        if self.kind not in {"token", "relation"} or len(self.blocks) != 4:
            raise ValueError("spectral kernel kind/block count differs")
        expected_input = 128 if self.kind == "token" else 1
        expected_features = (
            TOKEN_FEATURES_PER_BANDWIDTH
            if self.kind == "token" else RELATION_FEATURES_PER_BANDWIDTH
        )
        for index, block in enumerate(self.blocks):
            if block.bandwidth_index != index or block.omega.shape != (
                expected_features, expected_input,
            ):
                raise ValueError("spectral block shape/order differs")

    @property
    def total_features(self) -> int:
        return sum(block.omega.shape[0] for block in self.blocks)

    @property
    def payload(self) -> dict[str, object]:
        return {
            "contract": RFF_CONTRACT,
            "schema_version": 1,
            "kind": self.kind,
            "master_seed": RFF_MASTER_SEED,
            "numpy_version": self.numpy_version,
            "total_features": self.total_features,
            "blocks": [
                {
                    "resource_name": block.resource_name,
                    "bandwidth_index": block.bandwidth_index,
                    "bandwidth_hex": float(block.bandwidth).hex(),
                    "seed_payload": block.seed_payload,
                    "seed_payload_canonical_utf8": canonical_json_bytes(
                        block.seed_payload,
                    ).decode("ascii"),
                    "seed_sha256": block.seed_sha256,
                    "seed64": block.seed64,
                    "omega_shape": list(block.omega.shape),
                    "phase_shape": list(block.phase.shape),
                    "logical_hashes": block.logical_hashes,
                }
                for block in self.blocks
            ],
        }

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.payload)


@dataclass(frozen=True)
class SpectralResourceBundle:
    """The one logical kernel resource consumed by an HCWDL-RKD recipe."""

    token: SpectralKernelResources
    relation: SpectralKernelResources

    def __post_init__(self) -> None:
        if self.token.kind != "token" or self.relation.kind != "relation":
            raise ValueError("spectral resource bundle kinds differ")
        if self.token.numpy_version != self.relation.numpy_version:
            raise ValueError("spectral resource bundle NumPy versions differ")

    @property
    def payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "rff_contract": RFF_CONTRACT,
            "master_seed": RFF_MASTER_SEED,
            "token": self.token.payload,
            "relation": self.relation.payload,
        }

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.payload)


@dataclass(frozen=True)
class PublishedSpectralResources:
    envelope: CommittedBinaryEnvelope

    @property
    def array_path(self) -> Path:
        return self.envelope.directory / "kernel_resources.npz"

    @property
    def sidecar(self) -> Mapping[str, Any]:
        return self.envelope.sidecar


def generate_spectral_resources(
    kind: Literal["token", "relation"],
) -> SpectralKernelResources:
    """Generate the exact PCG64 resources frozen by the RKD plan."""

    if kind == "token":
        names = _TOKEN_RESOURCE_NAMES
        bandwidths = TOKEN_BANDWIDTHS
        features = TOKEN_FEATURES_PER_BANDWIDTH
        input_dim = 128
    elif kind == "relation":
        names = _RELATION_RESOURCE_NAMES
        bandwidths = RELATION_BANDWIDTHS
        features = RELATION_FEATURES_PER_BANDWIDTH
        input_dim = 1
    else:
        raise ValueError("unknown spectral resource kind")
    blocks: list[SpectralBlock] = []
    for index, (name, bandwidth) in enumerate(zip(names, bandwidths, strict=True)):
        seed_payload = {
            "bandwidth_index": index,
            "contract": RFF_CONTRACT,
            "master_seed": RFF_MASTER_SEED,
            "resource_name": name,
        }
        seed_sha256 = canonical_sha256(seed_payload)
        seed64 = int.from_bytes(bytes.fromhex(seed_sha256)[:8], "big", signed=False)
        generator = np.random.Generator(np.random.PCG64(seed64))
        omega64 = generator.standard_normal((features, input_dim)) / bandwidth
        phase64 = generator.uniform(0.0, 2.0 * np.pi, size=features)
        blocks.append(SpectralBlock(
            resource_name=name,
            bandwidth_index=index,
            bandwidth=bandwidth,
            seed_payload=seed_payload,
            seed_sha256=seed_sha256,
            seed64=seed64,
            omega=np.ascontiguousarray(omega64.astype(np.float32)),
            phase=np.ascontiguousarray(phase64.astype(np.float32)),
        ))
    result = SpectralKernelResources(kind, tuple(blocks), np.__version__)
    expected = K_TOKEN if kind == "token" else K_RELATION
    if result.total_features != expected:
        raise RuntimeError("spectral resource width differs")
    return result


def generate_spectral_resource_bundle() -> SpectralResourceBundle:
    """Generate both recipe-bound kernel families from the one frozen seed."""

    return SpectralResourceBundle(
        token=generate_spectral_resources("token"),
        relation=generate_spectral_resources("relation"),
    )


def spectral_resource_logical_hashes(
    resources: SpectralResourceBundle,
) -> dict[str, str]:
    """Return the recipe-bound logical identity of every omega/phase block."""

    if not isinstance(resources, SpectralResourceBundle):
        raise TypeError("spectral resource logical hashes require the full bundle")
    return {
        block.resource_name: canonical_sha256(block.logical_hashes)
        for family in (resources.token, resources.relation)
        for block in family.blocks
    }


def _resource_arrays(resources: SpectralKernelResources) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for index, block in enumerate(resources.blocks):
        result[f"block_{index}_omega"] = block.omega
        result[f"block_{index}_phase"] = block.phase
    return result


def _bundle_arrays(resources: SpectralResourceBundle) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for kind in ("token", "relation"):
        family = getattr(resources, kind)
        for name, value in _resource_arrays(family).items():
            result[f"{kind}_{name}"] = value
    return result


def _bundle_schema(resources: SpectralResourceBundle) -> dict[str, object]:
    return {
        "schema_version": 1,
        "member": "kernel_resources.npz",
        "container": "deterministic_npz",
        "arrays": {
            name: {"dtype": value.dtype.str, "shape": list(value.shape)}
            for name, value in sorted(_bundle_arrays(resources).items())
        },
    }


def publish_spectral_resources(
    resources: SpectralResourceBundle,
    *,
    root: str | Path,
    producer_task_id: str,
    immutable_parent_hashes: Mapping[str, Any],
    registered_output_row: Mapping[str, Any],
    campaign_or_recovery_owner: Mapping[str, Any],
) -> PublishedSpectralResources:
    """Publish the complete kernel pair through the shared committed envelope.

    A naked NPZ/JSON pair is deliberately not exposed by this API: Section 21
    makes the committed directory the only publication point.
    """

    if not isinstance(resources, SpectralResourceBundle):
        raise TypeError("spectral resource publication received the wrong type")
    array_bytes = deterministic_npz_bytes(_bundle_arrays(resources))
    envelope = publish_binary_envelope(
        root,
        artifact_contract=KERNEL_RESOURCES_CONTRACT,
        producer_task_id=producer_task_id,
        schema=_bundle_schema(resources),
        immutable_parent_hashes=immutable_parent_hashes,
        registered_output_row=registered_output_row,
        campaign_or_recovery_owner=campaign_or_recovery_owner,
        payloads={"kernel_resources.npz": array_bytes},
        member_metadata={
            "kernel_resources.npz": {
                "logical_sha256": resources.content_hash,
            },
        },
        sidecar_payload={
            "resource_id": "hcwdl_rkd_fixed_multiscale_spectral_moment_v1",
            "rff_contract": RFF_CONTRACT,
            "master_seed": RFF_MASTER_SEED,
            "array_member": "kernel_resources.npz",
            "spectral_resource_sha256": resources.content_hash,
            "spectral_resource": resources.payload,
        },
    )
    loaded = load_spectral_resources(
        envelope.root,
        envelope.envelope_id,
        expected_parents=immutable_parent_hashes,
        expected_owner_id=envelope.owner_id,
    )
    if loaded.content_hash != resources.content_hash:
        raise ValueError("published spectral resource logical hash differs")
    return PublishedSpectralResources(envelope)


def load_spectral_resources(
    root: str | Path,
    envelope_id: str,
    *,
    expected_parents: Mapping[str, Any],
    expected_owner_id: str | None = None,
) -> SpectralResourceBundle:
    """Authenticate the committed envelope and reconstruct both resources."""

    envelope = validate_binary_envelope(
        root,
        envelope_id,
        expected_contract=KERNEL_RESOURCES_CONTRACT,
        expected_parents=expected_parents,
        expected_owner_id=expected_owner_id,
    )
    sidecar = envelope.sidecar.get("payload")
    if not isinstance(sidecar, Mapping):
        raise ValueError("spectral resource sidecar payload differs")
    if (
        sidecar.get("rff_contract") != RFF_CONTRACT
        or sidecar.get("master_seed") != RFF_MASTER_SEED
        or sidecar.get("array_member") != "kernel_resources.npz"
    ):
        raise ValueError("spectral resource sidecar semantics differ")
    payload = sidecar.get("spectral_resource")
    if not isinstance(payload, Mapping):
        raise ValueError("spectral resource logical payload differs")
    if (
        payload.get("schema_version") != 1
        or payload.get("rff_contract") != RFF_CONTRACT
        or payload.get("master_seed") != RFF_MASTER_SEED
    ):
        raise ValueError("spectral resource bundle contract/seed differs")
    arrays = load_npz_arrays(envelope.directory / "kernel_resources.npz")
    expected_array_names = {
        f"{kind}_block_{index}_{suffix}"
        for kind in ("token", "relation")
        for index in range(4)
        for suffix in ("omega", "phase")
    }
    if set(arrays) != expected_array_names:
        raise ValueError("spectral resource array names differ")
    families = {
        kind: _load_spectral_family(
            kind,
            payload.get(kind),
            {
                name.removeprefix(f"{kind}_"): value
                for name, value in arrays.items()
                if name.startswith(f"{kind}_")
            },
        )
        for kind in ("token", "relation")
    }
    result = SpectralResourceBundle(
        token=families["token"], relation=families["relation"],
    )
    if result.payload != dict(payload):
        raise ValueError("spectral resource reconstructed payload differs")
    if sidecar.get("spectral_resource_sha256") != result.content_hash:
        raise ValueError("spectral resource logical content hash differs")
    members = envelope.commit.get("payload", {}).get("members", [])
    resource_member = next(
        (row for row in members if row.get("path") == "kernel_resources.npz"),
        None,
    )
    if not isinstance(resource_member, Mapping) or resource_member.get(
        "logical_sha256",
    ) != result.content_hash:
        raise ValueError("spectral resource committed logical hash differs")
    return result


def _load_spectral_family(
    expected_kind: Literal["token", "relation"],
    payload: object,
    arrays: Mapping[str, np.ndarray],
) -> SpectralKernelResources:
    if not isinstance(payload, Mapping) or payload.get("contract") != RFF_CONTRACT:
        raise ValueError("spectral resource family logical payload differs")
    if payload.get("kind") != expected_kind or payload.get("master_seed") != RFF_MASTER_SEED:
        raise ValueError("spectral resource logical kind/seed differs")
    block_payloads = payload.get("blocks")
    if not isinstance(block_payloads, list) or len(block_payloads) != 4:
        raise ValueError("spectral resource block registry differs")
    expected_array_names = {
        f"block_{index}_{suffix}"
        for index in range(4) for suffix in ("omega", "phase")
    }
    if set(arrays) != expected_array_names:
        raise ValueError("spectral resource array names differ")
    blocks: list[SpectralBlock] = []
    for index, block_payload in enumerate(block_payloads):
        if not isinstance(block_payload, dict):
            raise ValueError("spectral resource block payload differs")
        seed_payload = block_payload.get("seed_payload")
        if not isinstance(seed_payload, dict):
            raise ValueError("spectral resource seed payload differs")
        block = SpectralBlock(
            resource_name=str(block_payload.get("resource_name")),
            bandwidth_index=index,
            bandwidth=float.fromhex(str(block_payload.get("bandwidth_hex"))),
            seed_payload=seed_payload,
            seed_sha256=str(block_payload.get("seed_sha256")),
            seed64=int(block_payload.get("seed64")),
            omega=np.ascontiguousarray(arrays[f"block_{index}_omega"]),
            phase=np.ascontiguousarray(arrays[f"block_{index}_phase"]),
        )
        if block_payload.get("logical_hashes") != block.logical_hashes:
            raise ValueError("spectral resource logical array hash differs")
        blocks.append(block)
    result = SpectralKernelResources(
        expected_kind, tuple(blocks), str(payload.get("numpy_version")),
    )
    if result.payload != payload:
        raise ValueError("spectral resource reconstructed payload differs")
    return result


def finite_spectral_features(values, resources: SpectralKernelResources):
    """Evaluate the exact frozen finite feature map in FP32."""

    import torch

    value = torch.as_tensor(values).float()
    expected_dim = 128 if resources.kind == "token" else 1
    if resources.kind == "relation" and value.ndim >= 1 and value.shape[-1:] != (1,):
        value = value.unsqueeze(-1)
    if value.ndim < 2 or value.shape[-1] != expected_dim:
        raise ValueError("finite-kernel input dimension differs")
    if not torch.isfinite(value).all():
        raise FloatingPointError("finite-kernel inputs are nonfinite")
    scale = math.sqrt(2.0 / resources.total_features)
    key = (id(resources), value.device.type, value.device.index)
    cached = _DEVICE_BLOCK_CACHE.get(key)
    if cached is None or cached[0] is not resources:
        device_blocks = tuple(
            (
                torch.as_tensor(
                    block.omega, dtype=torch.float32, device=value.device,
                ),
                torch.as_tensor(
                    block.phase, dtype=torch.float32, device=value.device,
                ),
            )
            for block in resources.blocks
        )
        _DEVICE_BLOCK_CACHE[key] = (resources, device_blocks)
    else:
        device_blocks = cached[1]
    outputs = []
    for omega, phase in device_blocks:
        outputs.append(scale * torch.cos(value @ omega.transpose(0, 1) + phase))
    result = torch.cat(outputs, dim=-1)
    if result.shape[-1] != resources.total_features or not torch.isfinite(result).all():
        raise FloatingPointError("finite spectral features are invalid")
    return result


def normalized_weights(weights, *, expected_rows: int | None = None):
    import torch

    value = torch.as_tensor(weights).float()
    if value.ndim != 1 or (expected_rows is not None and len(value) != expected_rows):
        raise ValueError("kernel weights must be one-dimensional")
    if not torch.isfinite(value).all() or not (value > 0).all():
        raise ValueError("kernel weights must be finite and positive")
    total = value.sum()
    if not torch.isfinite(total) or total <= 0:
        raise ValueError("kernel weight sum is invalid")
    return value / total


def weighted_feature_mean(values, weights, resources: SpectralKernelResources):
    features = finite_spectral_features(values, resources)
    normalized = normalized_weights(weights, expected_rows=features.shape[0]).to(
        features.device,
    )
    result = (normalized[:, None] * features).sum(0)
    if not result.isfinite().all():
        raise FloatingPointError("weighted spectral mean is nonfinite")
    return result


def cached_finite_mmd(
    student_values,
    student_weights,
    teacher_mean,
    resources: SpectralKernelResources,
):
    import torch

    student_mean = weighted_feature_mean(student_values, student_weights, resources)
    target = torch.as_tensor(teacher_mean, device=student_mean.device).float()
    if target.shape != student_mean.shape or target.requires_grad:
        raise ValueError("cached teacher spectral mean differs or is not detached")
    result = (student_mean - target).square().sum()
    if not torch.isfinite(result):
        raise FloatingPointError("cached finite MMD is nonfinite")
    return result


def slow_pairwise_finite_mmd(
    student_values,
    student_weights,
    teacher_values,
    teacher_weights,
    resources: SpectralKernelResources,
):
    """Exact slow finite-kernel MMD used as an independent acceptance oracle."""

    student_features = finite_spectral_features(student_values, resources)
    teacher_features = finite_spectral_features(teacher_values, resources)
    student_weight = normalized_weights(
        student_weights, expected_rows=student_features.shape[0],
    ).to(student_features.device)
    teacher_weight = normalized_weights(
        teacher_weights, expected_rows=teacher_features.shape[0],
    ).to(teacher_features.device)
    ss = student_features @ student_features.transpose(0, 1)
    tt = teacher_features @ teacher_features.transpose(0, 1)
    st = student_features @ teacher_features.transpose(0, 1)
    return (
        (student_weight[:, None] * student_weight[None, :] * ss).sum()
        + (teacher_weight[:, None] * teacher_weight[None, :] * tt).sum()
        - 2.0 * (student_weight[:, None] * teacher_weight[None, :] * st).sum()
    )


def ideal_rbf_mmd(
    student_values: np.ndarray,
    student_weights: np.ndarray,
    teacher_values: np.ndarray,
    teacher_weights: np.ndarray,
    *,
    kind: Literal["token", "relation"],
) -> float:
    """FP64 infinite-mixture value diagnostic; never used as training loss."""

    left = np.asarray(student_values, np.float64)
    right = np.asarray(teacher_values, np.float64)
    if kind == "relation":
        left = left.reshape(-1, 1); right = right.reshape(-1, 1)
        bandwidths = RELATION_BANDWIDTHS
    elif kind == "token":
        bandwidths = TOKEN_BANDWIDTHS
    else:
        raise ValueError("unknown ideal-kernel kind")
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("ideal-kernel input shapes differ")
    lw = np.asarray(student_weights, np.float64); lw /= lw.sum()
    rw = np.asarray(teacher_weights, np.float64); rw /= rw.sum()
    def kernel(a, b):
        distance = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return sum(np.exp(-distance / (2.0 * sigma * sigma)) for sigma in bandwidths) / 4.0
    return float(
        (lw[:, None] * lw[None, :] * kernel(left, left)).sum()
        + (rw[:, None] * rw[None, :] * kernel(right, right)).sum()
        - 2.0 * (lw[:, None] * rw[None, :] * kernel(left, right)).sum()
    )


def analytic_finite_mmd_gradient(
    student_values: np.ndarray,
    student_weights: np.ndarray,
    teacher_values: np.ndarray,
    teacher_weights: np.ndarray,
    resources: SpectralKernelResources,
    *,
    normalize_inputs: bool,
) -> np.ndarray:
    """Analytic FP64 student gradient for finite-kernel acceptance tests."""

    raw = np.asarray(student_values, np.float64)
    teacher_raw = np.asarray(teacher_values, np.float64)
    if resources.kind == "relation":
        raw = raw.reshape(-1, 1); teacher_raw = teacher_raw.reshape(-1, 1)
    if raw.ndim != 2 or teacher_raw.ndim != 2 or raw.shape[1] != teacher_raw.shape[1]:
        raise ValueError("analytic-gradient input shapes differ")
    if normalize_inputs:
        raw_norm = np.linalg.norm(raw, axis=1, keepdims=True)
        teacher_norm = np.linalg.norm(teacher_raw, axis=1, keepdims=True)
        if np.any(raw_norm <= 0) or np.any(teacher_norm <= 0):
            raise ValueError("analytic normalization received a zero vector")
        student = raw / raw_norm
        teacher = teacher_raw / teacher_norm
    else:
        raw_norm = np.ones((len(raw), 1), np.float64)
        student = raw
        teacher = teacher_raw
    sw = np.asarray(student_weights, np.float64); sw /= sw.sum()
    tw = np.asarray(teacher_weights, np.float64); tw /= tw.sum()
    scale = math.sqrt(2.0 / resources.total_features)
    student_features = []
    teacher_features = []
    student_sines = []
    for block in resources.blocks:
        omega = block.omega.astype(np.float64)
        phase = block.phase.astype(np.float64)
        student_argument = student @ omega.T + phase
        teacher_argument = teacher @ omega.T + phase
        student_features.append(scale * np.cos(student_argument))
        teacher_features.append(scale * np.cos(teacher_argument))
        student_sines.append(-scale * np.sin(student_argument))
    phi_s = np.concatenate(student_features, axis=1)
    phi_t = np.concatenate(teacher_features, axis=1)
    delta = (sw[:, None] * phi_s).sum(0) - (tw[:, None] * phi_t).sum(0)
    gradient_normalized = np.zeros_like(student)
    offset = 0
    for block, sine in zip(resources.blocks, student_sines, strict=True):
        width = block.omega.shape[0]
        block_delta = delta[offset:offset + width]
        gradient_normalized += (
            2.0 * sw[:, None]
            * ((sine * block_delta[None, :]) @ block.omega.astype(np.float64))
        )
        offset += width
    if not normalize_inputs:
        return gradient_normalized.reshape(np.asarray(student_values).shape)
    projection = (gradient_normalized * student).sum(1, keepdims=True)
    gradient = (gradient_normalized - student * projection) / raw_norm
    return gradient.reshape(np.asarray(student_values).shape)


__all__ = [
    "KERNEL_RESOURCES_CONTRACT", "K_RELATION", "K_TOKEN", "PublishedSpectralResources",
    "RELATION_BANDWIDTHS",
    "RELATION_FEATURES_PER_BANDWIDTH", "RFF_CONTRACT", "RFF_MASTER_SEED",
    "SpectralBlock", "SpectralKernelResources", "SpectralResourceBundle", "TOKEN_BANDWIDTHS",
    "TOKEN_FEATURES_PER_BANDWIDTH", "analytic_finite_mmd_gradient",
    "cached_finite_mmd", "finite_spectral_features", "generate_spectral_resources",
    "generate_spectral_resource_bundle",
    "ideal_rbf_mmd", "normalized_weights", "slow_pairwise_finite_mmd",
    "load_spectral_resources", "publish_spectral_resources",
    "spectral_resource_logical_hashes", "weighted_feature_mean",
]

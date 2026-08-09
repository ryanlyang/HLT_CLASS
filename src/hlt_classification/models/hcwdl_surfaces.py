"""Frozen HCWDL tap schema, FP32 parity, and architecture attestation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import importlib
import inspect
from pathlib import Path
import sys
from typing import Any, Final, Literal

import torch

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    load_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
)

from .scouting_particle_transformer import (
    NativeOfflineParticleTransformer,
    ScoutingParticleTransformer,
    build_native_offline_particle_transformer,
    build_scouting_particle_transformer,
    scouting_particle_transformer_config,
)


TAP_CONTRACT: Final = "HCWDL_REPRESENTATION_TAP/v1"
ARCHITECTURE_ATTESTATION_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_ARCHITECTURE_ATTESTATION/v1"
)
SURFACE_PARITY_CONTRACT: Final = "HCWDL_REPRESENTATION_SURFACE_PARITY/v1"
RUNTIME_SIGNATURE_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_WEAVER_RUNTIME_SIGNATURE/v1"
)
FP32_ATOL: Final = 1.0e-6
FP32_RTOL: Final = 1.0e-5
HCWDL_PARENT_ARCHITECTURE_NODES: Final = frozenset({
    "D100", "TOFF", "M0",
    *(f"D{level}{track}" for level in (0, 25, 50, 75) for track in ("c", "w")),
    *(f"M{rung}{track}" for rung in range(1, 7) for track in ("c", "w")),
})
_FILE_BACKED_ATTESTATION_TOKEN: Final = object()

HCWDL_TAP_SCHEMA: Final = {
    "contract": TAP_CONTRACT,
    "schema_version": 1,
    "ordinary": {
        "input_features": 21,
        "particle_blocks": 8,
        "particle_block_2": {
            "block_zero_based_index": 1,
            "capture": "immediately_after_particle_attention_block",
            "layout": ["batch", "post_trimmer_tokens", 128],
        },
        "jet_penultimate": {
            "capture": "aggregator_output_before_final_linear",
            "layout": ["batch", 128],
        },
        "metadata": {
            "canonical_token_id": "zero_based_hlt_skeleton_index",
            "family_code": "raw_pre_transform_HCWDL_REP_TOKEN_FAMILY/v1",
            "transport": "same_single_trimmer_call_auxiliary_channels",
            "embedding_access": False,
            "deployable_access": False,
        },
    },
    "native_offline": {
        "charged_input_features": 19,
        "neutral_input_features": 7,
        "particle_blocks_per_encoder": 8,
        "charged_and_neutral_latent_spaces": "strictly_separate",
        "particle_block_2_zero_based_index": 1,
        "offline_jet_penultimate": (
            "classifier_layernorm_then_first_linear_then_gelu_before_final_linear"
        ),
        "canonical_token_ids": "native_branch_indices",
    },
    "public_forward_compatibility": {
        "legacy_forward_unchanged": True,
        "legacy_forward_representations_unchanged": True,
        "student_surface_forward_count": 1,
        "parity_precision": "float32",
        "parity_trimming": "disabled_fixture",
    },
}


@dataclass(frozen=True)
class CheckpointArchitectureAudit:
    node_id: str
    domain: str
    model_role: str
    checkpoint_sha256: str
    state_schema_sha256: str
    strict_key_shape_match: bool
    report_path: str | None = None
    report_sha256: str | None = None
    report_byte_sha256: str | None = None
    engine_report_path: str | None = None
    engine_report_sha256: str | None = None
    engine_report_byte_sha256: str | None = None
    checkpoint_path: str | None = None
    actual_file_evidence: bool = False


def tap_schema() -> dict[str, object]:
    return copy.deepcopy(HCWDL_TAP_SCHEMA)


def tap_schema_sha256() -> str:
    return canonical_sha256(HCWDL_TAP_SCHEMA)


def _qualified_name(value: object) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _callable_signature(value: object, *, name: str) -> str:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError) as error:
        raise TypeError(f"installed Weaver callable signature is unavailable: {name}") from error


def _source_evidence(classes: Sequence[type], *, require_weaver: bool) -> list[dict[str, str]]:
    """Hash the actual Python files defining every runtime surface component."""

    modules: dict[str, Path] = {}
    for value in classes:
        module_name = value.__module__
        module = sys.modules.get(module_name)
        if module is None:
            module = importlib.import_module(module_name)
        raw_path = getattr(module, "__file__", None)
        if raw_path is None:
            if require_weaver:
                raise ValueError(f"installed Weaver module has no source file: {module_name}")
            continue
        path = Path(raw_path).resolve()
        if path.suffix in {".pyc", ".pyo"} and path.with_suffix(".py").is_file():
            path = path.with_suffix(".py")
        if not path.is_file():
            raise FileNotFoundError(f"runtime source file is absent: {path}")
        modules[module_name] = path
    if not modules:
        raise ValueError("runtime signature has no source-file evidence")
    return [
        {
            "module": module_name,
            "file_name": path.name,
            "sha256": sha256_file(path),
        }
        for module_name, path in sorted(modules.items())
    ]


def _encoder_runtime_components(encoder: torch.nn.Module) -> dict[str, object]:
    required = (
        "trimmer", "embed", "pair_embed", "blocks", "_forward_aggregator",
        "fc", "block_ids_with_attn_mask", "cls_token",
    )
    if any(not hasattr(encoder, name) for name in required):
        raise TypeError("installed Weaver runtime surface is incomplete")
    blocks = tuple(encoder.blocks)
    if len(blocks) != 8:
        raise TypeError("installed Weaver particle-block count differs")
    cls_token = encoder.cls_token
    if not isinstance(cls_token, torch.Tensor) or cls_token.shape[-1] != 128:
        raise TypeError("installed Weaver hidden width differs")
    policy = getattr(encoder, "block_ids_with_attn_mask")
    if isinstance(policy, (list, tuple)):
        normalized_policy: object = list(policy)
    else:
        try:
            normalized_policy = sorted(int(value) for value in policy)
        except TypeError as error:
            raise TypeError("installed Weaver attention-mask policy differs") from error
    return {
        "encoder_class": _qualified_name(encoder),
        "trimmer_class": _qualified_name(encoder.trimmer),
        "embed_class": _qualified_name(encoder.embed),
        "pair_embed_class": _qualified_name(encoder.pair_embed),
        "block_classes": [_qualified_name(block) for block in blocks],
        "block_count": len(blocks),
        "hidden_width": int(cls_token.shape[-1]),
        "attention_mask_policy": normalized_policy,
        "constructor_signature": _callable_signature(type(encoder).__init__, name="ParticleTransformer.__init__"),
        "forward_signature": _callable_signature(type(encoder).forward, name="ParticleTransformer.forward"),
        "trimmer_signature": _callable_signature(type(encoder.trimmer).forward, name="trimmer.forward"),
        "aggregator_signature": _callable_signature(encoder._forward_aggregator, name="_forward_aggregator"),
        "classifier_class": _qualified_name(encoder.fc),
    }


def build_runtime_signature(
    *,
    ordinary_model: ScoutingParticleTransformer,
    native_offline_model: NativeOfflineParticleTransformer,
    runtime_kind: Literal["installed_weaver", "synthetic_test_double"],
) -> dict[str, object]:
    """Derive the runtime signature from live models and defining file bytes."""

    encoders = (
        ordinary_model.mod,
        native_offline_model.charged_encoder,
        native_offline_model.neutral_encoder,
    )
    actual_installed = all(
        encoder.__class__.__module__.startswith("weaver.") for encoder in encoders
    )
    if runtime_kind == "installed_weaver" and not actual_installed:
        raise ValueError("synthetic Weaver cannot be declared installed")
    runtime_classes: set[type] = set()
    for encoder in encoders:
        runtime_classes.update({
            type(encoder), type(encoder.trimmer), type(encoder.embed),
            type(encoder.pair_embed), type(encoder.fc),
            *(type(block) for block in encoder.blocks),
        })
    source_files = _source_evidence(
        sorted(runtime_classes, key=lambda value: (value.__module__, value.__qualname__)),
        require_weaver=runtime_kind == "installed_weaver",
    )
    ordinary = _encoder_runtime_components(ordinary_model.mod)
    charged = _encoder_runtime_components(native_offline_model.charged_encoder)
    neutral = _encoder_runtime_components(native_offline_model.neutral_encoder)
    classifier = native_offline_model.classifier
    if (
        not isinstance(classifier, torch.nn.Sequential)
        or len(classifier) != 4
        or not isinstance(classifier[0], torch.nn.LayerNorm)
        or not isinstance(classifier[1], torch.nn.Linear)
        or not isinstance(classifier[2], torch.nn.GELU)
        or not isinstance(classifier[3], torch.nn.Linear)
        or classifier[1].in_features != 256
        or classifier[1].out_features != 128
        or classifier[3].in_features != 128
        or classifier[3].out_features != 15
    ):
        raise TypeError("native-offline classifier path differs")
    return with_content_hash({
        "contract": RUNTIME_SIGNATURE_CONTRACT,
        "schema_version": 1,
        "runtime_kind": runtime_kind,
        "installed_weaver_runtime_detected": actual_installed,
        "torch_version": str(torch.__version__),
        "source_files": source_files,
        "weaver_source_sha256": canonical_sha256(source_files),
        "ordinary": ordinary,
        "native_offline": {
            "charged": charged,
            "neutral": neutral,
            "classifier_path": [
                _qualified_name(module) for module in classifier
            ],
            "classifier_dimensions": [256, 128, 15],
        },
    })


def validate_runtime_signature(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value,
        expected_contract=RUNTIME_SIGNATURE_CONTRACT,
        expected_schema_version=1,
    )
    kind = value.get("runtime_kind")
    required_top_level = {
        "contract", "schema_version", "runtime_kind",
        "installed_weaver_runtime_detected", "torch_version", "source_files",
        "weaver_source_sha256", "ordinary", "native_offline", "content_hash",
    }
    if set(value) != required_top_level:
        raise ValueError("runtime signature fields differ")
    if kind not in {"installed_weaver", "synthetic_test_double"}:
        raise ValueError("runtime signature provenance differs")
    installed = value.get("installed_weaver_runtime_detected")
    if not isinstance(installed, bool) or (kind == "installed_weaver" and not installed):
        raise ValueError("runtime signature installed-Weaver detection differs")
    if not isinstance(value.get("torch_version"), str) or not value["torch_version"]:
        raise ValueError("runtime signature PyTorch version differs")
    source_files = value.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("runtime signature source-file evidence differs")
    seen: set[str] = set()
    for row in source_files:
        if not isinstance(row, Mapping) or set(row) != {"module", "file_name", "sha256"}:
            raise ValueError("runtime signature source-file row differs")
        module = row.get("module")
        if not isinstance(module, str) or not module or module in seen:
            raise ValueError("runtime signature source modules differ")
        seen.add(module)
        if not isinstance(row.get("file_name"), str) or not row["file_name"]:
            raise ValueError("runtime signature source file name differs")
        require_sha256(row.get("sha256"), name=f"{module} runtime source")
    if source_files != sorted(source_files, key=lambda row: row["module"]):
        raise ValueError("runtime signature source-file order differs")
    if kind == "installed_weaver" and not any(
        str(row["module"]).startswith("weaver.") for row in source_files
    ):
        raise ValueError("runtime signature lacks installed-Weaver source evidence")
    if value.get("weaver_source_sha256") != canonical_sha256(source_files):
        raise ValueError("runtime signature Weaver-source hash differs")
    for name in ("ordinary", "native_offline"):
        if not isinstance(value.get(name), Mapping):
            raise ValueError(f"runtime signature {name} surface differs")
    encoder_fields = {
        "encoder_class", "trimmer_class", "embed_class", "pair_embed_class",
        "block_classes", "block_count", "hidden_width", "attention_mask_policy",
        "constructor_signature", "forward_signature", "trimmer_signature",
        "aggregator_signature", "classifier_class",
    }

    def validate_encoder(component: Mapping[str, Any], *, name: str) -> None:
        if set(component) != encoder_fields:
            raise ValueError(f"runtime signature {name} fields differ")
        if component.get("block_count") != 8 or component.get("hidden_width") != 128:
            raise ValueError(f"runtime signature {name} architecture differs")
        if not isinstance(component.get("block_classes"), list) or len(component["block_classes"]) != 8:
            raise ValueError(f"runtime signature {name} block registry differs")
        for key in (
            "encoder_class", "trimmer_class", "embed_class", "pair_embed_class",
            "constructor_signature", "forward_signature", "trimmer_signature",
            "aggregator_signature", "classifier_class",
        ):
            if not isinstance(component.get(key), str) or not component[key]:
                raise ValueError(f"runtime signature {name} callable/class differs")

    ordinary = value["ordinary"]
    validate_encoder(ordinary, name="ordinary")
    native = value["native_offline"]
    if set(native) != {
        "charged", "neutral", "classifier_path", "classifier_dimensions",
    }:
        raise ValueError("runtime signature native-offline fields differ")
    for branch in ("charged", "neutral"):
        component = native.get(branch)
        if not isinstance(component, Mapping):
            raise ValueError("runtime signature native-offline architecture differs")
        validate_encoder(component, name=f"native-offline {branch}")
    if native.get("classifier_dimensions") != [256, 128, 15]:
        raise ValueError("runtime signature native-offline classifier differs")
    if not isinstance(native.get("classifier_path"), list) or len(native["classifier_path"]) != 4:
        raise ValueError("runtime signature native-offline classifier path differs")
    return digest


def audit_checkpoint_architecture(
    model,
    checkpoint_state: Mapping[str, Any],
    *,
    node_id: str,
    domain: Literal["ordinary", "native_offline"],
    model_role: str,
    checkpoint_sha256: str,
) -> CheckpointArchitectureAudit:
    """Strictly compare a checkpoint's complete key/shape/dtype schema."""

    if not node_id or not model_role or domain not in {"ordinary", "native_offline"}:
        raise ValueError("checkpoint architecture audit identity differs")
    expected = model.state_dict()
    if set(checkpoint_state) != set(expected):
        raise ValueError("checkpoint architecture keys differ")
    schema = []
    for name in sorted(expected):
        value = checkpoint_state[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError("checkpoint architecture contains a non-tensor state value")
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f"checkpoint architecture tensor differs: {name}")
        schema.append({
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        })
    return CheckpointArchitectureAudit(
        node_id=node_id,
        domain=domain,
        model_role=model_role,
        checkpoint_sha256=require_sha256(
            checkpoint_sha256, name=f"{node_id} checkpoint SHA-256",
        ),
        state_schema_sha256=canonical_sha256(schema),
        strict_key_shape_match=True,
    )


def _parent_architecture_identity(node_id: str) -> tuple[str, str]:
    if node_id == "TOFF":
        return "native_offline", "teacher"
    if node_id.startswith("D") and len(node_id) >= 2:
        return "ordinary", "teacher"
    if node_id.startswith("M") and len(node_id) >= 2:
        return "ordinary", "logit_control"
    raise ValueError(f"unknown HCWDL parent architecture identity: {node_id}")


def audit_parent_checkpoint_file(
    *, node_id: str, training_report_path: str | Path,
) -> CheckpointArchitectureAudit:
    """Derive a strict architecture audit from authenticated files.

    The wrapper report, sibling PMARD engine report, and selected checkpoint are
    all opened and validated.  No caller-supplied report or checkpoint hash is
    accepted.  The checkpoint state is then strict-loaded into a freshly built
    canonical ordinary or TOFF architecture.
    """

    from hlt_classification.scouting.engine import validate_pmard_training_report
    from hlt_classification.scouting.hcwdl_training import (
        validate_hcwdl_training_report,
    )

    report_path = Path(training_report_path).resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"parent training report is absent: {report_path}")
    wrapper = load_json(report_path)
    report_sha256 = validate_hcwdl_training_report(wrapper)
    if wrapper.get("node_id") != node_id or wrapper.get("complete") is not True:
        raise ValueError(f"parent architecture report identity differs: {node_id}")

    engine_path = report_path.parent / "training_report.json"
    if engine_path == report_path or not engine_path.is_file():
        raise FileNotFoundError(f"parent engine report is absent: {engine_path}")
    engine = load_json(engine_path)
    engine_sha256 = validate_pmard_training_report(engine)
    if (
        wrapper.get("pmard_engine_report_sha256") != engine_sha256
        or engine.get("experiment_id") != node_id
        or engine.get("complete") is not True
    ):
        raise ValueError(f"parent architecture engine-report lineage differs: {node_id}")

    selected_name = engine.get("selected_checkpoint")
    if not isinstance(selected_name, str) or not selected_name or Path(selected_name).name != selected_name:
        raise ValueError(f"parent selected-checkpoint name differs: {node_id}")
    checkpoint_path = engine_path.parent / selected_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"parent selected checkpoint is absent: {checkpoint_path}")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if (
        engine.get("selected_checkpoint_sha256") != checkpoint_sha256
        or wrapper.get("selected_checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError(f"parent selected-checkpoint byte lineage differs: {node_id}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("model"), Mapping):
        raise ValueError(f"parent selected-checkpoint payload differs: {node_id}")
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, Mapping) or checkpoint_config.get("experiment_id") != node_id:
        raise ValueError(f"parent selected-checkpoint identity differs: {node_id}")

    domain, model_role = _parent_architecture_identity(node_id)
    model = (
        build_native_offline_particle_transformer()
        if domain == "native_offline"
        else build_scouting_particle_transformer()
    )
    state = checkpoint["model"]
    audit = audit_checkpoint_architecture(
        model,
        state,
        node_id=node_id,
        domain=domain,
        model_role=model_role,
        checkpoint_sha256=checkpoint_sha256,
    )
    # The schema check above gives precise diagnostics; strict loading is a
    # separate required proof that the canonical PyTorch architecture accepts
    # the complete serialized state without remapping or dropped keys.
    model.load_state_dict(state, strict=True)
    return CheckpointArchitectureAudit(
        node_id=audit.node_id,
        domain=audit.domain,
        model_role=audit.model_role,
        checkpoint_sha256=audit.checkpoint_sha256,
        state_schema_sha256=audit.state_schema_sha256,
        strict_key_shape_match=True,
        report_path=report_path.as_posix(),
        report_sha256=report_sha256,
        report_byte_sha256=sha256_file(report_path),
        engine_report_path=engine_path.resolve().as_posix(),
        engine_report_sha256=engine_sha256,
        engine_report_byte_sha256=sha256_file(engine_path),
        checkpoint_path=checkpoint_path.resolve().as_posix(),
        actual_file_evidence=True,
    )


def _gradient_comparison(expected: Mapping[str, torch.Tensor | None], model) -> tuple[bool, float]:
    maximum = 0.0
    actual = dict(model.named_parameters())
    if set(expected) != set(actual):
        return False, float("inf")
    for name, reference in expected.items():
        candidate = actual[name].grad
        if (reference is None) != (candidate is None):
            return False, float("inf")
        if reference is not None:
            if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
                return False, float("inf")
            maximum = max(maximum, float((reference - candidate).abs().max().cpu()))
            if not torch.allclose(reference, candidate, atol=FP32_ATOL, rtol=FP32_RTOL):
                return False, maximum
    return True, maximum


def _snapshot_model(model):
    return {
        "state": copy.deepcopy(model.state_dict()),
        "modes": {name: module.training for name, module in model.named_modules()},
        "buffers": {
            name: value.detach().clone() for name, value in model.named_buffers()
        },
        "enabled": {
            name: bool(module.enabled)
            for name, module in model.named_modules()
            if hasattr(module, "enabled")
        },
        "grads": {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
        },
        "cpu_rng": torch.random.get_rng_state().clone(),
        "cuda_rng": (
            tuple(value.clone() for value in torch.cuda.get_rng_state_all())
            if torch.cuda.is_available() else None
        ),
    }


def _restore_model(model, snapshot) -> None:
    model.load_state_dict(snapshot["state"], strict=True)
    for name, value in model.named_buffers():
        if name not in snapshot["buffers"]:
            raise RuntimeError("model buffer topology changed during surface parity")
        value.copy_(snapshot["buffers"][name].to(value.device))
    for name, module in model.named_modules():
        module.training = snapshot["modes"][name]
        if name in snapshot["enabled"]:
            module.enabled = snapshot["enabled"][name]
    for name, parameter in model.named_parameters():
        value = snapshot["grads"][name]
        parameter.grad = None if value is None else value.to(parameter.device).clone()
    torch.random.set_rng_state(snapshot["cpu_rng"])
    if snapshot["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(snapshot["cuda_rng"])


def _ordinary_parity(model: ScoutingParticleTransformer, inputs) -> dict[str, object]:
    features, vectors, mask, visible_indices, family_codes = inputs
    public_features = features.detach().clone().float().requires_grad_(True)
    public_vectors = vectors.detach().clone().float().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    public = model(public_features, public_vectors, mask)
    objective_weight = torch.linspace(.2, 1.2, public.numel(), device=public.device).reshape_as(public)
    (public.float() * objective_weight).sum().backward()
    public_parameter_gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    public_feature_gradient = public_features.grad.detach().clone()
    public_vector_gradient = public_vectors.grad.detach().clone()

    surface_features = features.detach().clone().float().requires_grad_(True)
    surface_vectors = vectors.detach().clone().float().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    surface = model.forward_hcwdl_surfaces(
        surface_features, surface_vectors, mask, visible_indices, family_codes,
    )
    (surface.logits.float() * objective_weight).sum().backward()
    parameters_close, parameter_maximum = _gradient_comparison(
        public_parameter_gradients, model,
    )
    result = {
        "logit_maximum_absolute_difference": float(
            (public.float() - surface.logits.float()).abs().max().cpu()
        ),
        "feature_gradient_maximum_absolute_difference": float(
            (public_feature_gradient - surface_features.grad).abs().max().cpu()
        ),
        "vector_gradient_maximum_absolute_difference": float(
            (public_vector_gradient - surface_vectors.grad).abs().max().cpu()
        ),
        "parameter_gradient_maximum_absolute_difference": parameter_maximum,
        "parameter_gradients_close": parameters_close,
        "surface_shapes": {
            "particle_block_2": list(surface.particle_block_2.shape),
            "jet_penultimate": list(surface.jet_penultimate.shape),
            "mask": list(surface.particle_mask.shape),
        },
        "metadata_exact": bool(
            torch.equal(surface.particle_mask, mask[:, 0])
            and torch.equal(surface.visible_indices, visible_indices)
            and torch.equal(surface.family_codes, family_codes)
        ),
    }
    result["passed"] = bool(
        result["logit_maximum_absolute_difference"] <= FP32_ATOL
        and result["feature_gradient_maximum_absolute_difference"] <= FP32_ATOL
        and result["vector_gradient_maximum_absolute_difference"] <= FP32_ATOL
        and parameters_close and result["metadata_exact"]
    )
    return result


def _native_parity(model: NativeOfflineParticleTransformer, inputs) -> dict[str, object]:
    (
        charged_features, charged_vectors, charged_mask,
        neutral_features, neutral_vectors, neutral_mask,
        charged_ids, neutral_ids,
    ) = inputs
    public_float_inputs = [
        value.detach().clone().float().requires_grad_(True)
        for value in (
            charged_features, charged_vectors, neutral_features, neutral_vectors,
        )
    ]
    model.zero_grad(set_to_none=True)
    public = model(
        public_float_inputs[0], public_float_inputs[1], charged_mask,
        public_float_inputs[2], public_float_inputs[3], neutral_mask,
    )
    weight = torch.linspace(.2, 1.2, public.numel(), device=public.device).reshape_as(public)
    (public.float() * weight).sum().backward()
    public_parameter_gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    public_input_gradients = [value.grad.detach().clone() for value in public_float_inputs]
    surface_float_inputs = [
        value.detach().clone().float().requires_grad_(True)
        for value in (
            charged_features, charged_vectors, neutral_features, neutral_vectors,
        )
    ]
    model.zero_grad(set_to_none=True)
    surface = model.forward_hcwdl_surfaces(
        surface_float_inputs[0], surface_float_inputs[1], charged_mask,
        surface_float_inputs[2], surface_float_inputs[3], neutral_mask,
        charged_ids, neutral_ids,
    )
    (surface.logits.float() * weight).sum().backward()
    parameters_close, parameter_maximum = _gradient_comparison(
        public_parameter_gradients, model,
    )
    input_maximum = max(
        float((left - right.grad).abs().max().cpu())
        for left, right in zip(public_input_gradients, surface_float_inputs, strict=True)
    )
    result = {
        "logit_maximum_absolute_difference": float(
            (public.float() - surface.logits.float()).abs().max().cpu()
        ),
        "input_gradient_maximum_absolute_difference": input_maximum,
        "parameter_gradient_maximum_absolute_difference": parameter_maximum,
        "parameter_gradients_close": parameters_close,
        "surface_shapes": {
            "charged_particle_block_2": list(surface.charged_particle_block_2.shape),
            "neutral_particle_block_2": list(surface.neutral_particle_block_2.shape),
            "offline_jet_penultimate": list(surface.offline_jet_penultimate.shape),
        },
        "branches_remain_separate": (
            surface.charged_particle_block_2.shape[1]
            == charged_features.shape[2]
            and surface.neutral_particle_block_2.shape[1]
            == neutral_features.shape[2]
        ),
        "metadata_exact": bool(
            torch.equal(surface.charged_visible_indices, charged_ids)
            and torch.equal(surface.neutral_visible_indices, neutral_ids)
        ),
    }
    result["passed"] = bool(
        result["logit_maximum_absolute_difference"] <= FP32_ATOL
        and input_maximum <= FP32_ATOL and parameters_close
        and result["branches_remain_separate"] and result["metadata_exact"]
    )
    return result


def build_surface_parity_report(
    *,
    ordinary_model: ScoutingParticleTransformer,
    native_offline_model: NativeOfflineParticleTransformer,
    ordinary_inputs,
    native_offline_inputs,
    runtime_kind: Literal["installed_weaver", "synthetic_test_double"],
) -> dict[str, object]:
    """Run trimming-disabled FP32 logit/input/parameter-gradient parity."""

    if not isinstance(ordinary_model, ScoutingParticleTransformer) or not isinstance(
        native_offline_model, NativeOfflineParticleTransformer,
    ):
        raise TypeError("surface parity requires canonical ordinary/TOFF wrappers")
    actual_installed = all(
        encoder.__class__.__module__.startswith("weaver.")
        for encoder in (
            ordinary_model.mod,
            native_offline_model.charged_encoder,
            native_offline_model.neutral_encoder,
        )
    )
    if runtime_kind == "installed_weaver" and not actual_installed:
        raise ValueError("synthetic Weaver cannot be declared installed")
    runtime_signature = build_runtime_signature(
        ordinary_model=ordinary_model,
        native_offline_model=native_offline_model,
        runtime_kind=runtime_kind,
    )
    snapshots = (_snapshot_model(ordinary_model), _snapshot_model(native_offline_model))
    try:
        ordinary_model.eval(); native_offline_model.eval()
        ordinary_model.mod.trimmer.enabled = False
        native_offline_model.charged_encoder.trimmer.enabled = False
        native_offline_model.neutral_encoder.trimmer.enabled = False
        ordinary = _ordinary_parity(ordinary_model, ordinary_inputs)
        native = _native_parity(native_offline_model, native_offline_inputs)
    finally:
        _restore_model(ordinary_model, snapshots[0])
        _restore_model(native_offline_model, snapshots[1])
    passed = bool(ordinary["passed"] and native["passed"])
    authorization_capable = bool(passed and runtime_kind == "installed_weaver" and actual_installed)
    return with_content_hash({
        "contract": SURFACE_PARITY_CONTRACT,
        "schema_version": 1,
        "tap_schema_sha256": tap_schema_sha256(),
        "runtime_kind": runtime_kind,
        "installed_weaver_runtime_detected": actual_installed,
        "runtime_signature": runtime_signature,
        "runtime_signature_sha256": runtime_signature["content_hash"],
        "passed": passed,
        "authorization_capable": authorization_capable,
        "precision": "float32",
        "trimming": "disabled_fixture",
        "absolute_tolerance": FP32_ATOL,
        "relative_tolerance": FP32_RTOL,
        "ordinary": ordinary,
        "native_offline": native,
    })


def validate_surface_parity_report(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value,
        expected_contract=SURFACE_PARITY_CONTRACT,
        expected_schema_version=1,
    )
    required_fields = {
        "contract", "schema_version", "tap_schema_sha256", "runtime_kind",
        "installed_weaver_runtime_detected", "runtime_signature",
        "runtime_signature_sha256", "passed", "authorization_capable",
        "precision", "trimming", "absolute_tolerance", "relative_tolerance",
        "ordinary", "native_offline", "content_hash",
    }
    if set(value) != required_fields:
        raise ValueError("surface parity report fields differ")
    if value.get("tap_schema_sha256") != tap_schema_sha256():
        raise ValueError("surface parity tap schema differs")
    if value.get("runtime_kind") not in {"installed_weaver", "synthetic_test_double"}:
        raise ValueError("surface parity runtime provenance differs")
    runtime = value.get("runtime_signature")
    if not isinstance(runtime, Mapping):
        raise ValueError("surface parity runtime signature differs")
    runtime_hash = validate_runtime_signature(runtime)
    if (
        value.get("runtime_signature_sha256") != runtime_hash
        or runtime.get("runtime_kind") != value.get("runtime_kind")
        or runtime.get("installed_weaver_runtime_detected")
        is not value.get("installed_weaver_runtime_detected")
    ):
        raise ValueError("surface parity runtime-signature binding differs")
    if value.get("precision") != "float32" or value.get("trimming") != "disabled_fixture":
        raise ValueError("surface parity precision/trimming differs")
    if value.get("absolute_tolerance") != FP32_ATOL or value.get("relative_tolerance") != FP32_RTOL:
        raise ValueError("surface parity tolerance differs")
    ordinary = value.get("ordinary"); native = value.get("native_offline")
    if not isinstance(ordinary, Mapping) or not isinstance(native, Mapping):
        raise ValueError("surface parity component reports differ")
    ordinary_fields = {
        "logit_maximum_absolute_difference",
        "feature_gradient_maximum_absolute_difference",
        "vector_gradient_maximum_absolute_difference",
        "parameter_gradient_maximum_absolute_difference",
        "parameter_gradients_close", "surface_shapes", "metadata_exact", "passed",
    }
    native_fields = {
        "logit_maximum_absolute_difference", "input_gradient_maximum_absolute_difference",
        "parameter_gradient_maximum_absolute_difference", "parameter_gradients_close",
        "surface_shapes", "branches_remain_separate", "metadata_exact", "passed",
    }
    if set(ordinary) != ordinary_fields or set(native) != native_fields:
        raise ValueError("surface parity component fields differ")

    def finite_difference(component: Mapping[str, Any], name: str) -> float:
        result = float(component[name])
        if not torch.isfinite(torch.tensor(result)) or result < 0:
            raise ValueError("surface parity difference is nonfinite or negative")
        return result

    ordinary_passed = bool(
        finite_difference(ordinary, "logit_maximum_absolute_difference") <= FP32_ATOL
        and finite_difference(ordinary, "feature_gradient_maximum_absolute_difference") <= FP32_ATOL
        and finite_difference(ordinary, "vector_gradient_maximum_absolute_difference") <= FP32_ATOL
        and ordinary.get("parameter_gradients_close") is True
        and ordinary.get("metadata_exact") is True
    )
    native_passed = bool(
        finite_difference(native, "logit_maximum_absolute_difference") <= FP32_ATOL
        and finite_difference(native, "input_gradient_maximum_absolute_difference") <= FP32_ATOL
        and native.get("parameter_gradients_close") is True
        and native.get("branches_remain_separate") is True
        and native.get("metadata_exact") is True
    )
    # Parameter maxima are audited for finiteness even though allclose carries
    # the relative-tolerance decision used by the forward parity execution.
    finite_difference(ordinary, "parameter_gradient_maximum_absolute_difference")
    finite_difference(native, "parameter_gradient_maximum_absolute_difference")
    if ordinary.get("passed") is not ordinary_passed or native.get("passed") is not native_passed:
        raise ValueError("surface parity component result differs")
    passed = bool(ordinary_passed and native_passed)
    if value.get("passed") is not passed:
        raise ValueError("surface parity aggregate result differs")
    authorization = bool(
        passed
        and value.get("runtime_kind") == "installed_weaver"
        and value.get("installed_weaver_runtime_detected") is True
    )
    if value.get("authorization_capable") is not authorization:
        raise ValueError("surface parity authorization result differs")
    return digest


def build_architecture_attestation(
    *,
    parity_report: Mapping[str, Any],
    runtime_signature: Mapping[str, Any],
    model_source_sha256: str,
    checkpoint_audits: Sequence[CheckpointArchitectureAudit],
    tap_schema_path: str | None = None,
    tap_schema_byte_sha256: str | None = None,
    surface_parity_path: str | None = None,
    surface_parity_byte_sha256: str | None = None,
    model_source_files: Sequence[Mapping[str, Any]] = (),
    _file_backed_token: object | None = None,
) -> dict[str, object]:
    """Build an attestation; synthetic parity is retained but cannot authorize."""

    validate_surface_parity_report(parity_report)
    if parity_report.get("tap_schema_sha256") != tap_schema_sha256() or not parity_report.get("passed"):
        raise ValueError("surface parity/tap schema differs")
    if not isinstance(runtime_signature, Mapping) or not runtime_signature:
        raise ValueError("installed-Weaver runtime signature is empty")
    runtime_hash = validate_runtime_signature(runtime_signature)
    if (
        runtime_signature.get("runtime_kind") != parity_report.get("runtime_kind")
        or parity_report.get("runtime_signature_sha256") != runtime_hash
        or parity_report.get("runtime_signature") != runtime_signature
    ):
        raise ValueError("runtime signature/parity binding differs")
    weaver_source = require_sha256(
        runtime_signature.get("weaver_source_sha256"),
        name="Weaver runtime source SHA-256",
    )
    normalized_sources: list[dict[str, str]] = []
    source_names: set[str] = set()
    for raw in sorted(model_source_files, key=lambda row: str(row.get("logical_name", ""))):
        if not isinstance(raw, Mapping) or set(raw) != {"logical_name", "path", "sha256"}:
            raise ValueError("model source-file evidence fields differ")
        logical_name = raw.get("logical_name")
        path = raw.get("path")
        if not isinstance(logical_name, str) or not logical_name or logical_name in source_names:
            raise ValueError("model source-file logical names differ")
        if not isinstance(path, str) or not path:
            raise ValueError("model source-file path differs")
        source_names.add(logical_name)
        normalized_sources.append({
            "logical_name": logical_name,
            "path": path,
            "sha256": require_sha256(raw.get("sha256"), name=f"{logical_name} model source"),
        })
    source_hash = require_sha256(model_source_sha256, name="model source SHA-256")
    source_identity = [
        {"logical_name": row["logical_name"], "sha256": row["sha256"]}
        for row in normalized_sources
    ]
    if normalized_sources and source_hash != canonical_sha256(source_identity):
        raise ValueError("model source-file aggregate hash differs")
    if (surface_parity_path is None) is not (surface_parity_byte_sha256 is None):
        raise ValueError("surface-parity file evidence is incomplete")
    if (tap_schema_path is None) is not (tap_schema_byte_sha256 is None):
        raise ValueError("tap-schema file evidence is incomplete")
    tap_byte_hash = (
        None if tap_schema_byte_sha256 is None
        else require_sha256(tap_schema_byte_sha256, name="tap schema bytes")
    )
    parity_byte_hash = (
        None if surface_parity_byte_sha256 is None
        else require_sha256(surface_parity_byte_sha256, name="surface parity bytes")
    )
    normalized_audits = []
    seen: set[str] = set()
    for audit in sorted(checkpoint_audits, key=lambda row: row.node_id):
        if not isinstance(audit, CheckpointArchitectureAudit) or audit.node_id in seen:
            raise ValueError("checkpoint architecture audit registry differs")
        seen.add(audit.node_id)
        if not audit.strict_key_shape_match:
            raise ValueError("checkpoint did not strictly match architecture")
        row = {
            "node_id": audit.node_id,
            "domain": audit.domain,
            "model_role": audit.model_role,
            "checkpoint_sha256": audit.checkpoint_sha256,
            "state_schema_sha256": audit.state_schema_sha256,
            "strict_key_shape_match": True,
            "report_path": audit.report_path,
            "report_sha256": audit.report_sha256,
            "report_byte_sha256": audit.report_byte_sha256,
            "engine_report_path": audit.engine_report_path,
            "engine_report_sha256": audit.engine_report_sha256,
            "engine_report_byte_sha256": audit.engine_report_byte_sha256,
            "checkpoint_path": audit.checkpoint_path,
            "actual_file_evidence": audit.actual_file_evidence,
        }
        if audit.actual_file_evidence:
            for name in ("report_path", "engine_report_path", "checkpoint_path"):
                if not isinstance(row[name], str) or not row[name]:
                    raise ValueError(f"checkpoint architecture {name} differs")
            for name in (
                "report_sha256", "report_byte_sha256", "engine_report_sha256",
                "engine_report_byte_sha256",
            ):
                row[name] = require_sha256(row[name], name=f"{audit.node_id} {name}")
        elif any(
            row[name] is not None for name in (
                "report_path", "report_sha256", "report_byte_sha256",
                "engine_report_path", "engine_report_sha256",
                "engine_report_byte_sha256", "checkpoint_path",
            )
        ):
            raise ValueError("partial checkpoint file evidence is forbidden")
        normalized_audits.append(row)
    if not normalized_audits:
        raise ValueError("architecture attestation checkpoint registry is empty")
    exact_files = bool(
        _file_backed_token is _FILE_BACKED_ATTESTATION_TOKEN
        and tap_schema_path
        and tap_byte_hash
        and surface_parity_path
        and parity_byte_hash
        and normalized_sources
        and all(row["actual_file_evidence"] for row in normalized_audits)
    )
    complete_registry = seen == HCWDL_PARENT_ARCHITECTURE_NODES
    authorized = bool(
        parity_report.get("authorization_capable") and exact_files and complete_registry
    )
    blocker = (
        None if authorized
        else "installed_weaver_parity_required"
        if not parity_report.get("authorization_capable")
        else "complete_parent_architecture_registry_required"
        if not complete_registry
        else "exact_file_architecture_evidence_required"
    )
    return with_content_hash({
        "contract": ARCHITECTURE_ATTESTATION_CONTRACT,
        "schema_version": 1,
        "scientific_authorization": authorized,
        "authorization_blocker": blocker,
        "exact_file_evidence": exact_files,
        "parent_registry_complete": complete_registry,
        "tap_schema": tap_schema(),
        "tap_schema_sha256": tap_schema_sha256(),
        "tap_schema_path": tap_schema_path,
        "tap_schema_byte_sha256": tap_byte_hash,
        "canonical_scouting_configuration": scouting_particle_transformer_config(),
        "canonical_scouting_configuration_sha256": canonical_sha256(
            scouting_particle_transformer_config(),
        ),
        "model_source_files": normalized_sources,
        "model_source_sha256": source_hash,
        "runtime_signature": dict(runtime_signature),
        "runtime_signature_sha256": runtime_hash,
        "weaver_source_sha256": weaver_source,
        "surface_parity_path": surface_parity_path,
        "surface_parity_byte_sha256": parity_byte_hash,
        "surface_parity_sha256": parity_report["content_hash"],
        "surface_parity": dict(parity_report),
        "checkpoint_audits": normalized_audits,
    })


def build_architecture_attestation_from_files(
    *,
    tap_schema_path: str | Path,
    surface_parity_path: str | Path,
    parent_reports: Mapping[str, str | Path],
    model_source_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, object]:
    """Build the only authorization-capable architecture attestation path."""

    tap_path = Path(tap_schema_path).resolve()
    if not tap_path.is_file():
        raise FileNotFoundError(f"tap schema artifact is absent: {tap_path}")
    materialized_tap = load_json(tap_path)
    if materialized_tap != tap_schema():
        raise ValueError("materialized tap schema differs from the frozen schema")
    parity_path = Path(surface_parity_path).resolve()
    if not parity_path.is_file():
        raise FileNotFoundError(f"surface parity artifact is absent: {parity_path}")
    parity = load_json(parity_path)
    validate_surface_parity_report(parity)
    if not parent_reports:
        raise ValueError("architecture attestation parent-report registry is empty")

    # Reconstruct the canonical wrappers in the active process and prove the
    # parity artifact came from this exact installed runtime, not merely a
    # mapping that happens to contain an installed-looking provenance string.
    with torch.random.fork_rng(devices=[]):
        ordinary = build_scouting_particle_transformer()
        native = build_native_offline_particle_transformer()
    active_runtime = build_runtime_signature(
        ordinary_model=ordinary,
        native_offline_model=native,
        runtime_kind=str(parity["runtime_kind"]),
    )
    if active_runtime["content_hash"] != parity.get("runtime_signature_sha256"):
        raise ValueError("surface parity was produced by a stale Weaver runtime")

    scouting_module = sys.modules[ScoutingParticleTransformer.__module__]
    scouting_path = getattr(scouting_module, "__file__", None)
    if scouting_path is None:
        raise ValueError("canonical scouting model source path is unavailable")
    required_model_sources = {
        "hcwdl_surfaces": Path(__file__).resolve(),
        "scouting_particle_transformer": Path(scouting_path).resolve(),
    }
    if model_source_paths is None:
        model_source_paths = required_model_sources
    else:
        for logical_name, required_path in required_model_sources.items():
            supplied = model_source_paths.get(logical_name)
            if supplied is None or Path(supplied).resolve() != required_path:
                raise ValueError(
                    f"architecture attestation canonical model source differs: {logical_name}"
                )
    if not model_source_paths:
        raise ValueError("architecture attestation model-source registry is empty")
    model_sources: list[dict[str, str]] = []
    for logical_name, raw_path in sorted(model_source_paths.items()):
        if not isinstance(logical_name, str) or not logical_name:
            raise ValueError("architecture model-source logical name differs")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"architecture model source is absent: {path}")
        model_sources.append({
            "logical_name": logical_name,
            "path": path.as_posix(),
            "sha256": sha256_file(path),
        })

    audits = [
        audit_parent_checkpoint_file(
            node_id=node_id, training_report_path=report_path,
        )
        for node_id, report_path in sorted(parent_reports.items())
    ]
    return build_architecture_attestation(
        parity_report=parity,
        runtime_signature=active_runtime,
        model_source_sha256=canonical_sha256([
            {"logical_name": row["logical_name"], "sha256": row["sha256"]}
            for row in model_sources
        ]),
        checkpoint_audits=audits,
        tap_schema_path=tap_path.as_posix(),
        tap_schema_byte_sha256=sha256_file(tap_path),
        surface_parity_path=parity_path.as_posix(),
        surface_parity_byte_sha256=sha256_file(parity_path),
        model_source_files=model_sources,
        _file_backed_token=_FILE_BACKED_ATTESTATION_TOKEN,
    )


def _verify_architecture_attestation_files(value: Mapping[str, Any]) -> None:
    """Re-open every byte source bound by an exact-file attestation."""

    tap_path = Path(str(value["tap_schema_path"]))
    parity_path = Path(str(value["surface_parity_path"]))
    if sha256_file(tap_path) != value["tap_schema_byte_sha256"]:
        raise ValueError("attested tap-schema bytes are stale")
    if load_json(tap_path) != tap_schema():
        raise ValueError("attested tap-schema semantics are stale")
    if sha256_file(parity_path) != value["surface_parity_byte_sha256"]:
        raise ValueError("attested surface-parity bytes are stale")
    parity = load_json(parity_path)
    validate_surface_parity_report(parity)
    if parity != value["surface_parity"]:
        raise ValueError("attested surface-parity artifact differs")

    for row in value["model_source_files"]:
        path = Path(str(row["path"]))
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"attested model source bytes are stale: {row['logical_name']}")

    with torch.random.fork_rng(devices=[]):
        ordinary = build_scouting_particle_transformer()
        native = build_native_offline_particle_transformer()
    active_runtime = build_runtime_signature(
        ordinary_model=ordinary,
        native_offline_model=native,
        runtime_kind=str(parity["runtime_kind"]),
    )
    if active_runtime["content_hash"] != value["runtime_signature_sha256"]:
        raise ValueError("attested installed-Weaver runtime is stale")

    for expected in value["checkpoint_audits"]:
        actual = audit_parent_checkpoint_file(
            node_id=str(expected["node_id"]),
            training_report_path=str(expected["report_path"]),
        )
        comparisons = {
            "domain": actual.domain,
            "model_role": actual.model_role,
            "checkpoint_sha256": actual.checkpoint_sha256,
            "state_schema_sha256": actual.state_schema_sha256,
            "strict_key_shape_match": actual.strict_key_shape_match,
            "report_path": actual.report_path,
            "report_sha256": actual.report_sha256,
            "report_byte_sha256": actual.report_byte_sha256,
            "engine_report_path": actual.engine_report_path,
            "engine_report_sha256": actual.engine_report_sha256,
            "engine_report_byte_sha256": actual.engine_report_byte_sha256,
            "checkpoint_path": actual.checkpoint_path,
            "actual_file_evidence": actual.actual_file_evidence,
        }
        if any(expected.get(name) != actual_value for name, actual_value in comparisons.items()):
            raise ValueError(
                f"attested parent report/checkpoint bytes are stale: {actual.node_id}"
            )


def validate_architecture_attestation(
    value: Mapping[str, Any], *, require_authorized: bool = True,
    verify_files: bool | None = None,
) -> str:
    digest = validate_content_hash(
        value,
        expected_contract=ARCHITECTURE_ATTESTATION_CONTRACT,
        expected_schema_version=1,
    )
    required_fields = {
        "contract", "schema_version", "scientific_authorization",
        "authorization_blocker", "exact_file_evidence",
        "parent_registry_complete", "tap_schema", "tap_schema_sha256",
        "tap_schema_path", "tap_schema_byte_sha256",
        "canonical_scouting_configuration",
        "canonical_scouting_configuration_sha256", "model_source_files",
        "model_source_sha256", "runtime_signature", "runtime_signature_sha256",
        "weaver_source_sha256", "surface_parity_path",
        "surface_parity_byte_sha256", "surface_parity_sha256",
        "surface_parity", "checkpoint_audits", "content_hash",
    }
    if set(value) != required_fields:
        raise ValueError("architecture attestation fields differ")
    if value.get("tap_schema") != HCWDL_TAP_SCHEMA or value.get("tap_schema_sha256") != tap_schema_sha256():
        raise ValueError("architecture attestation tap schema differs")
    tap_path = value.get("tap_schema_path")
    tap_bytes = value.get("tap_schema_byte_sha256")
    if (tap_path is None) is not (tap_bytes is None):
        raise ValueError("architecture attestation tap-schema file evidence differs")
    if tap_path is not None:
        if not isinstance(tap_path, str) or not tap_path:
            raise ValueError("architecture attestation tap-schema path differs")
        require_sha256(tap_bytes, name="tap schema bytes")
    parity = value.get("surface_parity")
    if not isinstance(parity, Mapping):
        raise ValueError("architecture attestation parity report differs")
    validate_surface_parity_report(parity)
    if value.get("surface_parity_sha256") != parity.get("content_hash"):
        raise ValueError("architecture attestation parity hash differs")
    if value.get("canonical_scouting_configuration") != scouting_particle_transformer_config():
        raise ValueError("architecture attestation canonical model configuration differs")
    if value.get("canonical_scouting_configuration_sha256") != canonical_sha256(
        scouting_particle_transformer_config(),
    ):
        raise ValueError("architecture attestation model configuration hash differs")
    model_sources = value.get("model_source_files")
    if not isinstance(model_sources, list):
        raise ValueError("architecture attestation model source files differ")
    for row in model_sources:
        if not isinstance(row, Mapping) or set(row) != {"logical_name", "path", "sha256"}:
            raise ValueError("architecture attestation model source row differs")
        if not isinstance(row.get("logical_name"), str) or not row["logical_name"]:
            raise ValueError("architecture attestation model source identity differs")
        if not isinstance(row.get("path"), str) or not row["path"]:
            raise ValueError("architecture attestation model source path differs")
        require_sha256(row.get("sha256"), name=f"{row['logical_name']} model source")
    model_source_hash = require_sha256(
        value.get("model_source_sha256"), name="model source SHA-256",
    )
    source_identity = [
        {"logical_name": row["logical_name"], "sha256": row["sha256"]}
        for row in model_sources
    ]
    if model_sources and model_source_hash != canonical_sha256(source_identity):
        raise ValueError("architecture attestation model source hash differs")
    runtime = value.get("runtime_signature")
    if not isinstance(runtime, Mapping) or not runtime:
        raise ValueError("architecture attestation runtime signature differs")
    runtime_hash = validate_runtime_signature(runtime)
    if value.get("runtime_signature_sha256") != runtime_hash:
        raise ValueError("architecture attestation runtime signature hash differs")
    if (
        runtime.get("runtime_kind") != parity.get("runtime_kind")
        or parity.get("runtime_signature_sha256") != runtime_hash
        or parity.get("runtime_signature") != runtime
    ):
        raise ValueError("architecture attestation runtime provenance differs")
    if value.get("weaver_source_sha256") != require_sha256(
        runtime.get("weaver_source_sha256"), name="Weaver source SHA-256",
    ):
        raise ValueError("architecture attestation Weaver source hash differs")
    audits = value.get("checkpoint_audits")
    if not isinstance(audits, list) or not audits:
        raise ValueError("architecture attestation checkpoint audits differ")
    seen: set[str] = set()
    required_audit_fields = {
        "node_id", "domain", "model_role", "checkpoint_sha256",
        "state_schema_sha256", "strict_key_shape_match", "report_path",
        "report_sha256", "report_byte_sha256", "engine_report_path",
        "engine_report_sha256", "engine_report_byte_sha256",
        "checkpoint_path", "actual_file_evidence",
    }
    for audit in audits:
        if not isinstance(audit, Mapping) or set(audit) != required_audit_fields:
            raise ValueError("architecture attestation checkpoint audit fields differ")
        node = audit.get("node_id")
        if not isinstance(node, str) or not node or node in seen:
            raise ValueError("architecture attestation checkpoint node IDs differ")
        seen.add(node)
        if audit.get("domain") not in {"ordinary", "native_offline"}:
            raise ValueError("architecture attestation checkpoint domain differs")
        if not isinstance(audit.get("model_role"), str) or not audit.get("model_role"):
            raise ValueError("architecture attestation checkpoint role differs")
        expected_domain, expected_role = _parent_architecture_identity(node)
        if audit.get("domain") != expected_domain or audit.get("model_role") != expected_role:
            raise ValueError("architecture attestation checkpoint identity semantics differ")
        require_sha256(audit.get("checkpoint_sha256"), name=f"{node} checkpoint SHA-256")
        require_sha256(audit.get("state_schema_sha256"), name=f"{node} state schema SHA-256")
        if audit.get("strict_key_shape_match") is not True:
            raise ValueError("architecture attestation checkpoint strict audit failed")
        actual_file_evidence = audit.get("actual_file_evidence")
        if not isinstance(actual_file_evidence, bool):
            raise ValueError("architecture attestation checkpoint file-evidence flag differs")
        file_names = ("report_path", "engine_report_path", "checkpoint_path")
        hash_names = (
            "report_sha256", "report_byte_sha256", "engine_report_sha256",
            "engine_report_byte_sha256",
        )
        if actual_file_evidence:
            if any(not isinstance(audit.get(name), str) or not audit[name] for name in file_names):
                raise ValueError("architecture attestation checkpoint file path differs")
            for name in hash_names:
                require_sha256(audit.get(name), name=f"{node} {name}")
        elif any(audit.get(name) is not None for name in (*file_names, *hash_names)):
            raise ValueError("architecture attestation has partial checkpoint file evidence")
    if audits != sorted(audits, key=lambda row: row["node_id"]):
        raise ValueError("architecture attestation checkpoint audit order differs")
    complete_registry = seen == HCWDL_PARENT_ARCHITECTURE_NODES
    if value.get("parent_registry_complete") is not complete_registry:
        raise ValueError("architecture attestation parent-registry completion differs")
    parity_path = value.get("surface_parity_path")
    parity_bytes = value.get("surface_parity_byte_sha256")
    if (parity_path is None) is not (parity_bytes is None):
        raise ValueError("architecture attestation surface-parity file evidence differs")
    if parity_path is not None:
        if not isinstance(parity_path, str) or not parity_path:
            raise ValueError("architecture attestation surface-parity path differs")
        require_sha256(parity_bytes, name="surface parity bytes")
    exact_files = bool(
        tap_path and tap_bytes and parity_path and parity_bytes and model_sources
        and all(bool(audit["actual_file_evidence"]) for audit in audits)
    )
    if value.get("exact_file_evidence") is not exact_files:
        raise ValueError("architecture attestation exact-file evidence differs")
    authorized = bool(
        parity.get("authorization_capable")
        and parity.get("runtime_kind") == "installed_weaver"
        and parity.get("installed_weaver_runtime_detected")
        and exact_files
        and complete_registry
    )
    if value.get("scientific_authorization") is not authorized:
        raise ValueError("architecture attestation authorization bit differs")
    expected_blocker = (
        None if authorized
        else "installed_weaver_parity_required"
        if not parity.get("authorization_capable")
        else "complete_parent_architecture_registry_required"
        if not complete_registry
        else "exact_file_architecture_evidence_required"
    )
    if value.get("authorization_blocker") != expected_blocker:
        raise ValueError("architecture attestation authorization blocker differs")
    if require_authorized and not authorized:
        raise ValueError(
            "installed-Weaver parity and exact-file architecture evidence have not authorized this artifact"
        )
    should_verify_files = require_authorized if verify_files is None else verify_files
    if should_verify_files:
        if not exact_files:
            raise ValueError("architecture attestation has no exact files to verify")
        _verify_architecture_attestation_files(value)
    return digest


__all__ = [
    "ARCHITECTURE_ATTESTATION_CONTRACT", "CheckpointArchitectureAudit",
    "FP32_ATOL", "FP32_RTOL", "HCWDL_PARENT_ARCHITECTURE_NODES",
    "HCWDL_TAP_SCHEMA", "RUNTIME_SIGNATURE_CONTRACT",
    "SURFACE_PARITY_CONTRACT", "TAP_CONTRACT", "audit_checkpoint_architecture",
    "audit_parent_checkpoint_file", "build_architecture_attestation",
    "build_architecture_attestation_from_files", "build_runtime_signature",
    "build_surface_parity_report",
    "tap_schema", "tap_schema_sha256", "validate_architecture_attestation",
    "validate_runtime_signature", "validate_surface_parity_report",
]

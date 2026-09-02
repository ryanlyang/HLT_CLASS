"""Production workers for the complete offline/HLT fusion study."""

from __future__ import annotations

import gc
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, load_json, sha256_file, write_immutable_json,
)
from hlt_classification.models.hcwdl_offline_hlt_fusion_transformer import (
    AnchoredFusionParticleTransformer, SymmetricFusionParticleTransformer,
    UntaggedConcatParticleTransformer, _content_view,
)
from hlt_classification.models.hcwdl_tagged_concat_transformer import (
    TaggedConcatParticleTransformer,
)

from .hcwdl_mhpe_tri60_ce_control import load_control_model
from .hcwdl_mhpe_tri60_graph import COORDINATES
from .hcwdl_mhpe_tri60_runner import _configure_deterministic_backend
from .hcwdl_mhpe_tri60_training import (
    Tri60TrainingAuthority, Tri60TrainingRuntime, _BatchPrefetcher,
    _peak_cuda_bytes, _peak_rss_bytes, _torch_bytes, load_tri60_model,
    train_tri60_node,
)
from .hcwdl_offline_hlt_concat_data import CONCAT_CAPACITY
from .hcwdl_offline_hlt_concat_runner import (
    _caches as _concat_caches, _distribution, _first_batch, _identity_batch,
    _selected_counts,
)
from .hcwdl_offline_hlt_fusion_contracts import (
    ALPHA_ZERO_AUDIT_CONTRACT, CAPACITY_AUDIT_CONTRACT,
    DEPLOYABLE_CHECKPOINT_CONTRACT, EXECUTION_ACCEPTANCE_CONTRACT,
    FINAL_CHECKPOINT_CONTRACT, SELECTED_CHECKPOINT_CONTRACT,
    TRAINING_REPORT_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_offline_hlt_fusion_graph import (
    COSINE_ALPHA, GRAPH_SHA256, NODE_REGISTRY, STEP_ALPHA,
    STUDY_C_NODES, TEACHER_DISTRIBUTION, TEACHER_NODE, TRAINING_100,
    TRAINING_60,
)
from .hcwdl_offline_hlt_fusion_probability import (
    FusionProbabilityTargets, publish_lock, publish_role,
    validate_lock as validate_probability_lock,
)
from .hcwdl_tri100_spine4_bottleneck_source import validate_source_lock
from .hcwdl_unified_balanced_runner import _cache_student_views, _load_common
from .dataset import _concat_batches, _slice_batch
from .training import derive_seed


def training_authority(node_id: str) -> Tri60TrainingAuthority:
    authority = Tri60TrainingAuthority(
        node=NODE_REGISTRY[node_id], graph_sha256=GRAPH_SHA256,
        training_report_contract=TRAINING_REPORT_CONTRACT,
        selected_checkpoint_contract=SELECTED_CHECKPOINT_CONTRACT,
        final_checkpoint_contract=FINAL_CHECKPOINT_CONTRACT,
        allowed_initializations=("fresh", "warm_selected_checkpoint"),
        allowed_training_passes=(60, 100),
    )
    authority.validate()
    return authority


def _foundation(spec: Mapping[str, Any]) -> dict[str, Any]:
    source = load_json(spec["artifact_paths"]["source_lock"])
    validate_source_lock(source)
    return load_json(spec["artifact_paths"]["foundation_spec"])


def _source_models(spec: Mapping[str, Any]):
    source = load_json(spec["artifact_paths"]["source_lock"])
    validate_source_lock(source)
    m0, m0_report = load_control_model(
        spec["artifact_paths"]["m0ce60_report"], device="cpu",
    )
    u000, u000_report = load_tri60_model(
        source["u000"]["report_path"], device="cpu",
    )
    return m0, m0_report, u000, u000_report


def _source_lineage(spec: Mapping[str, Any]):
    m0, m0_report, u000, u000_report = _source_models(spec)
    result = {
        "m0_state": {name: value.detach().cpu() for name, value in m0.state_dict().items()},
        "m0_report": m0_report,
        "u000_state": {
            name: value.detach().cpu() for name, value in u000.state_dict().items()
        },
        "u000_report": u000_report,
    }
    del m0, u000
    return result


def model_factory(spec: Mapping[str, Any], node_id: str) -> Callable[[], Any]:
    sources = _source_lineage(spec)
    if node_id == "CONCAT_UNTAGGED":
        return UntaggedConcatParticleTransformer
    if node_id == "CONCAT_TAGGED":
        return TaggedConcatParticleTransformer
    if node_id.startswith("SYMMETRIC_FUSION_"):
        arm = node_id.rsplit("_", 1)[-1]
        return lambda: SymmetricFusionParticleTransformer(arm)  # type: ignore[arg-type]
    if node_id in {"HLT_WARM_CONTINUE", "FUSION_DIRECT_KD_WARM"}:
        def hlt_factory():
            from hlt_classification.models.scouting_particle_transformer import (
                ScoutingParticleTransformer,
            )
            model = ScoutingParticleTransformer()
            model.load_state_dict(sources["m0_state"], strict=True)
            return model
        return hlt_factory
    context = "H" if node_id == "ANCHORED_FUSION_HH" else "O"
    if node_id in {"FUSION_WITHDRAW_COS", "FUSION_WITHDRAW_STEP"}:
        context = "O"
        teacher_path = (
            Path(spec["campaign_root"]) / "training" / TEACHER_NODE
            / "training_report.json"
        )
        teacher_factory = model_factory(spec, TEACHER_NODE)
        teacher, _ = load_tri60_model(
            teacher_path, device="cpu", model_factory=teacher_factory,
            authority=training_authority(TEACHER_NODE),
        )
        teacher_state = {
            name: value.detach().cpu() for name, value in teacher.state_dict().items()
        }
        del teacher

        def withdrawal_factory():
            model = AnchoredFusionParticleTransformer("O")
            model.load_state_dict(teacher_state, strict=True)
            return model
        return withdrawal_factory

    def anchored_factory():
        model = AnchoredFusionParticleTransformer(context)  # type: ignore[arg-type]
        model.load_hlt_state(sources["m0_state"])
        model.load_context_state(sources["u000_state"])
        return model
    return anchored_factory


def _runtime(node_id: str) -> Tri60TrainingRuntime:
    row = TRAINING_100 if node_id in STUDY_C_NODES else TRAINING_60
    return Tri60TrainingRuntime(
        passes=int(row.get("maximum_passes", row.get("passes"))),
        batch_size=int(row["effective_batch_size"]),
        peak_learning_rate=float(row["peak_learning_rate"]),
        weight_decay=float(row["weight_decay"]), warmup_fraction=.05,
        minimum_lr_fraction=float(row["learning_rate_floor_fraction"]),
        amp_dtype=str(row["forward_precision"]),
    )


def _hlt_caches(spec: Mapping[str, Any]):
    foundation = _foundation(spec)
    split, split_hash, selection_hash, selections, assignments, balanced = _load_common(
        foundation,
    )
    node = NODE_REGISTRY["HLT_WARM_CONTINUE"]
    sampler_seed = derive_seed(
        int(spec["replicate_seed"]), node.seed_alias + "/sampler",
    )
    repair_seed = derive_seed(
        int(spec["replicate_seed"]), "fusion/repair/shared_v1",
    )
    caches, input_key = _cache_student_views(
        foundation_spec=foundation, split=split, selections=selections,
        assignments=assignments, balanced=balanced, behavior="hlt",
        coordinate=COORDINATES["D000"], batch_size=256,
        sampler_seed=sampler_seed, repair_seed=repair_seed, memory_gib=240.0,
        include_hcwdl_metadata=True,
    )
    if input_key != "hlt":
        raise PermissionError("fusion deployable cache is not exact HLT")
    return caches, split_hash, selection_hash, input_key


def build_capacity_audit(spec: Mapping[str, Any]) -> dict[str, Any]:
    foundation = _foundation(spec)
    split, split_hash, selection_hash, selections, _, _ = _load_common(foundation)
    roles = {}
    for role in ("train", "validation"):
        offline, hlt, combined, maximum_identities = _selected_counts(
            foundation, split, selections[role], role,
        )
        roles[role] = {
            "offline": _distribution(offline), "raw_hlt": _distribution(hlt),
            "combined": _distribution(combined),
            "rows_over_concat_capacity": int(np.count_nonzero(combined > 496)),
            "rows_over_anchored_hlt_capacity": int(np.count_nonzero(hlt > 200)),
            "symmetric_oo_maximum": int(2 * offline.max()),
            "symmetric_hh_maximum": int(2 * hlt.max()),
            "maximum_identities": maximum_identities,
        }
    if any(row["rows_over_concat_capacity"] for row in roles.values()):
        raise ValueError("fusion oracle capacity would truncate O+raw-H tokens")
    view_row_bytes = CONCAT_CAPACITY * (
        21 * np.dtype(np.float32).itemsize
        + 4 * np.dtype(np.float32).itemsize
        + np.dtype(np.bool_).itemsize
        + np.dtype(np.int64).itemsize
        + 3 * np.dtype(np.int8).itemsize
    )
    cache_row_bytes = (
        view_row_bytes + np.dtype(np.int32).itemsize
        + np.dtype(np.int64).itemsize + 32
    )
    projected_cache_bytes = sum(
        int(spec["role_counts"][role]) * cache_row_bytes for role in roles
    )
    locked_cache_limit = 350 * 1024**3
    if projected_cache_bytes > locked_cache_limit:
        raise MemoryError("fusion RAM cache projection exceeds its locked limit")
    return artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "source_lock": spec["parents"]["source_lock"],
            "foundation": spec["parents"]["foundation"],
            "split_manifest": split_hash, "selection_manifest": selection_hash,
        },
        "roles": roles, "concat_capacity": CONCAT_CAPACITY,
        "symmetric_dynamic_padding": True,
        "anchored_hlt_capacity": 200,
        "anchored_hlt_overflow_policy": "canonical_native_order_cap_200_v1",
        "projected_view_row_bytes": view_row_bytes,
        "projected_cache_array_row_bytes": cache_row_bytes,
        "projected_ram_cache_array_bytes": projected_cache_bytes,
        "locked_ram_cache_array_limit_bytes": locked_cache_limit,
        "training_job_requested_memory_bytes": 500 * 1024**3,
        "every_raw_hlt_token_retained_by_concat_and_symmetric": True,
        "matching_indices_read": False, "durable_particle_view_bytes": 0,
        "final_test_accessed": False,
    }, contract=CAPACITY_AUDIT_CONTRACT)


def validate_capacity_audit(
    spec: Mapping[str, Any], value: Mapping[str, Any],
) -> str:
    digest = validate_artifact(value, contract=CAPACITY_AUDIT_CONTRACT)
    if (
        value.get("parents", {}).get("campaign_spec") != spec["content_hash"]
        or value.get("parents", {}).get("source_lock")
        != spec["parents"]["source_lock"]
        or value.get("concat_capacity") != 496
        or value.get("anchored_hlt_capacity") != 200
        or value.get("every_raw_hlt_token_retained_by_concat_and_symmetric")
        is not True
        or value.get("matching_indices_read") is not False
        or value.get("durable_particle_view_bytes") != 0
        or set(value.get("roles", {})) != {"train", "validation"}
        or any(
            set(row.get("maximum_identities", {}))
            != {"offline", "hlt", "combined"}
            or not all(row["maximum_identities"].values())
            for row in value.get("roles", {}).values()
        )
        or int(value.get("projected_ram_cache_array_bytes", -1)) < 1
        or int(value.get("projected_ram_cache_array_bytes", -1))
        > int(value.get("locked_ram_cache_array_limit_bytes", -2))
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("fusion capacity audit semantics differ")
    return digest


def _maximum_length_acceptance_batch(
    spec: Mapping[str, Any], audit: Mapping[str, Any], *, batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build one production batch containing every cardinality extremum."""

    extrema = []
    seen = set()
    for role in ("train", "validation"):
        row = audit["roles"][role]
        for name in ("offline", "raw_hlt", "combined"):
            identity_key = "hlt" if name == "raw_hlt" else name
            identity = str(row["maximum_identities"][identity_key])
            if not identity or identity in seen:
                continue
            seen.add(identity)
            extrema.append({
                "role": role, "kind": name, "identity": identity,
                "length": int(row[name]["maximum"]),
            })
    if not extrema or len(extrema) >= batch_size:
        raise RuntimeError("fusion acceptance extrema registry differs")
    first = _first_batch(spec, role="train", batch_size=batch_size)
    components = [_slice_batch(first, 0, batch_size - len(extrema))]
    components.extend(
        _identity_batch(spec, role=row["role"], identity=row["identity"])
        for row in extrema
    )
    batch = _concat_batches(tuple(components))
    if len(batch["labels"]) != batch_size:
        raise RuntimeError("fusion acceptance batch size differs")
    view = batch["privileged"]
    active = view.mask[:, 0]
    source = view.content_source_codes
    offline = np.count_nonzero(active & (source == 0), axis=1)
    hlt = np.count_nonzero(active & (source == 1), axis=1)
    observed = {
        "offline": int(offline.max()), "raw_hlt": int(hlt.max()),
        "combined": int((offline + hlt).max()),
    }
    expected = {
        name: max(int(row[name]["maximum"]) for row in audit["roles"].values())
        for name in observed
    }
    if observed != expected:
        raise RuntimeError(
            f"fusion acceptance extrema differ: {observed} != {expected}"
        )
    return batch, extrema


def run_execution_acceptance(spec: Mapping[str, Any], *, device: str = "cuda"):
    import torch

    started = time.monotonic()
    audit = load_json(spec["artifact_paths"]["capacity_audit"])
    audit_hash = validate_capacity_audit(spec, audit)
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("fusion preflight requires CUDA")
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    batch, extrema = _maximum_length_acceptance_batch(
        spec, audit, batch_size=256,
    )
    view = batch["privileged"]
    args = (
        torch.as_tensor(view.features, device=target),
        torch.as_tensor(view.vectors, device=target),
        torch.as_tensor(view.mask, device=target),
        torch.as_tensor(view.content_source_codes, device=target),
    )
    labels = torch.as_tensor(batch["labels"], dtype=torch.long, device=target)
    checked = []
    architectures = {}
    for node_id in (
        "CONCAT_UNTAGGED", "CONCAT_TAGGED", "SYMMETRIC_FUSION_OO",
        "SYMMETRIC_FUSION_HH", "SYMMETRIC_FUSION_OH",
        "ANCHORED_FUSION_HH", "ANCHORED_FUSION_OH",
    ):
        model = model_factory(spec, node_id)().to(target).train()
        architectures[node_id] = {
            "parameters": [
                {
                    "name": name, "shape": list(parameter.shape),
                    "requires_grad": bool(parameter.requires_grad),
                }
                for name, parameter in model.named_parameters()
            ],
            "parameter_scalar_count": sum(
                int(parameter.numel()) for parameter in model.parameters()
            ),
            "trainable_parameter_scalar_count": sum(
                int(parameter.numel()) for parameter in model.parameters()
                if parameter.requires_grad
            ),
        }
        with torch.autocast(
            device_type=target.type, dtype=torch.bfloat16,
            enabled=target.type == "cuda",
        ):
            logits = model(*args)
            loss = torch.nn.functional.cross_entropy(logits.float(), labels)
        loss.backward()
        if logits.shape != (256, 15) or not torch.isfinite(logits).all():
            raise RuntimeError(f"fusion preflight failed for {node_id}")
        checked.append(node_id)
        del model, logits, loss
        gc.collect()
        if target.type == "cuda":
            torch.cuda.empty_cache()
    hlt_view = _content_view(*args, code=1, capacity=200)
    hlt_control = model_factory(spec, "HLT_WARM_CONTINUE")().to(target).train()
    architectures["HLT_WARM_CONTINUE"] = {
        "parameters": [
            {
                "name": name, "shape": list(parameter.shape),
                "requires_grad": bool(parameter.requires_grad),
            }
            for name, parameter in hlt_control.named_parameters()
        ],
        "parameter_scalar_count": sum(
            int(parameter.numel()) for parameter in hlt_control.parameters()
        ),
        "trainable_parameter_scalar_count": sum(
            int(parameter.numel()) for parameter in hlt_control.parameters()
            if parameter.requires_grad
        ),
    }
    with torch.autocast(
        device_type=target.type, dtype=torch.bfloat16,
        enabled=target.type == "cuda",
    ):
        hlt_logits = hlt_control(*hlt_view)
        hlt_loss = torch.nn.functional.cross_entropy(hlt_logits.float(), labels)
    hlt_loss.backward()
    if hlt_logits.shape != (256, 15) or not torch.isfinite(hlt_logits).all():
        raise RuntimeError("fusion HLT warm control preflight differs")
    checked.append("HLT_WARM_CONTINUE")
    del hlt_control, hlt_logits, hlt_loss
    gc.collect()
    if target.type == "cuda":
        torch.cuda.empty_cache()

    from .hcwdl_offline_hlt_withdrawal import withdrawal_loss
    withdrawal_model = model_factory(spec, TEACHER_NODE)().to(target).train()
    with torch.autocast(
        device_type=target.type, dtype=torch.bfloat16,
        enabled=target.type == "cuda",
    ):
        withdrawal = withdrawal_model.forward_withdrawal(*args, alpha=.5)
    teacher = torch.full(
        (len(labels), 15), 1 / 15, dtype=torch.float32, device=target,
    )
    withdrawal_losses = withdrawal_loss(withdrawal, labels, teacher)
    withdrawal_losses["total"].backward()
    residual_gradients = [
        injection.residual_projection.weight.grad
        for injection in withdrawal_model.injections
    ]
    if not all(
        value is not None and torch.isfinite(value).all()
        and bool((value.abs().sum() > 0).item())
        for value in residual_gradients
    ):
        raise RuntimeError("fusion withdrawal cross-residual gradients differ")
    # Zero initialization intentionally makes the residual path an exact HLT
    # model on update zero. Take one preflight-only projection step, then prove
    # that the now-open path carries gradients through every context family.
    with torch.no_grad():
        for injection in withdrawal_model.injections:
            injection.residual_projection.weight.add_(
                injection.residual_projection.weight.grad, alpha=-1e-3,
            )
            if injection.residual_projection.bias is not None:
                if injection.residual_projection.bias.grad is None:
                    raise RuntimeError("fusion residual bias gradient differs")
                injection.residual_projection.bias.add_(
                    injection.residual_projection.bias.grad, alpha=-1e-3,
                )
    withdrawal_model.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=target.type, dtype=torch.bfloat16,
        enabled=target.type == "cuda",
    ):
        second = withdrawal_model.forward_withdrawal(*args, alpha=.5)
    second_losses = withdrawal_loss(second, labels, teacher)
    second_losses["total"].backward()

    def active_gradient(prefix: str) -> bool:
        gradients = [
            parameter.grad for name, parameter in withdrawal_model.named_parameters()
            if name.startswith(prefix) and parameter.requires_grad
        ]
        return bool(gradients) and all(
            value is None or bool(torch.isfinite(value).all().item())
            for value in gradients
        ) and any(
            value is not None and bool((value.abs().sum() > 0).item())
            for value in gradients
        )

    gradient_families = {
        "context_encoder": active_gradient("context_mod."),
        "context_content_embedding": active_gradient(
            "context_content_embedding."
        ),
        "cross_pair_geometry": active_gradient("cross_pair_mod.pair_embed."),
        **{
            f"cross_injection_{index}": active_gradient(f"injections.{index}.")
            for index in range(len(withdrawal_model.injections))
        },
    }
    if not all(gradient_families.values()):
        raise RuntimeError(
            f"fusion active context gradient families differ: {gradient_families}"
        )
    del (
        withdrawal_model, withdrawal, withdrawal_losses, second,
        second_losses, teacher,
    )
    gc.collect()
    if target.type == "cuda":
        torch.cuda.empty_cache()
    anchored = model_factory(spec, TEACHER_NODE)().to(target).eval()
    with torch.inference_mode():
        zero = anchored.forward_zero(*args).logits.float()
        offline = args[3] == 0
        perturbed_features = args[0].masked_fill(offline[:, None], float("nan"))
        perturbed_vectors = args[1].masked_fill(offline[:, None], float("nan"))
        perturbed = anchored.forward_zero(
            perturbed_features, perturbed_vectors, args[2], args[3],
        ).logits.float()
        hlt = _content_view(*args, code=1, capacity=200)
        extracted = anchored.extract_hlt().to(target).eval()
        direct = extracted(*hlt).float()
    maximum_error = float((zero - direct).abs().max().item())
    perturbation_error = float((zero - perturbed).abs().max().item())
    if maximum_error > 2e-5 or perturbation_error != 0:
        raise RuntimeError("fusion alpha-zero extraction parity differs")
    return artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "capacity_audit": audit_hash, "graph": GRAPH_SHA256,
        },
        "device_type": target.type,
        "device_name": torch.cuda.get_device_name(target) if target.type == "cuda" else "cpu",
        "installed_weaver_forward_backward": True,
        "production_batch_size_exercised": 256,
        "oracle_nodes_exercised": checked,
        "architecture_registry": architectures,
        "matched_symmetric_parameter_registry": (
            architectures["SYMMETRIC_FUSION_OO"]["parameters"]
            == architectures["SYMMETRIC_FUSION_HH"]["parameters"]
            == architectures["SYMMETRIC_FUSION_OH"]["parameters"]
        ),
        "matched_anchored_parameter_registry": (
            architectures["ANCHORED_FUSION_HH"]["parameters"]
            == architectures["ANCHORED_FUSION_OH"]["parameters"]
        ),
        "withdrawal_transition_exercised": True,
        "cross_residual_gradients_nonzero": True,
        "post_open_context_gradient_families": gradient_families,
        "all_active_context_gradient_families_nonzero": True,
        "maximum_extrema_exercised": extrema,
        "maximum_lengths_exercised": {
            name: max(
                int(row[name]["maximum"]) for row in audit["roles"].values()
            )
            for name in ("offline", "raw_hlt", "combined")
        },
        "alpha_zero_context_dispatch_skipped": True,
        "alpha_zero_extraction_max_abs_error": maximum_error,
        "alpha_zero_offline_perturbation_max_abs_error": perturbation_error,
        "alpha_zero_offline_perturbation_invariant": True,
        "alpha_zero_extraction_parity": True,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": _peak_rss_bytes(),
        "peak_cuda_bytes": _peak_cuda_bytes(),
        "final_test_accessed": False,
    }, contract=EXECUTION_ACCEPTANCE_CONTRACT)


def validate_execution_acceptance(
    spec: Mapping[str, Any], value: Mapping[str, Any],
) -> str:
    digest = validate_artifact(value, contract=EXECUTION_ACCEPTANCE_CONTRACT)
    audit = load_json(spec["artifact_paths"]["capacity_audit"])
    validate_capacity_audit(spec, audit)
    expected_lengths = {
        name: max(
            int(row[name]["maximum"]) for row in audit["roles"].values()
        )
        for name in ("offline", "raw_hlt", "combined")
    }
    architectures = value.get("architecture_registry", {})
    required = {
        "CONCAT_UNTAGGED", "CONCAT_TAGGED", "SYMMETRIC_FUSION_OO",
        "SYMMETRIC_FUSION_HH", "SYMMETRIC_FUSION_OH",
        "HLT_WARM_CONTINUE", "ANCHORED_FUSION_HH",
        "ANCHORED_FUSION_OH",
    }
    if (
        value.get("parents", {}).get("campaign_spec") != spec["content_hash"]
        or set(value.get("oracle_nodes_exercised", ())) != required
        or not isinstance(architectures, Mapping)
        or set(architectures) != required
        or value.get("installed_weaver_forward_backward") is not True
        or value.get("production_batch_size_exercised") != 256
        or value.get("maximum_lengths_exercised") != expected_lengths
        or not isinstance(value.get("maximum_extrema_exercised"), list)
        or not value.get("maximum_extrema_exercised")
        or value.get("withdrawal_transition_exercised") is not True
        or value.get("cross_residual_gradients_nonzero") is not True
        or value.get("all_active_context_gradient_families_nonzero") is not True
        or not value.get("post_open_context_gradient_families")
        or not all(value["post_open_context_gradient_families"].values())
        or value.get("matched_symmetric_parameter_registry") is not True
        or value.get("matched_anchored_parameter_registry") is not True
        or value.get("alpha_zero_context_dispatch_skipped") is not True
        or value.get("alpha_zero_extraction_parity") is not True
        or float(value.get("alpha_zero_extraction_max_abs_error", 1.0)) > 2e-5
        or value.get("alpha_zero_offline_perturbation_invariant") is not True
        or float(value.get(
            "alpha_zero_offline_perturbation_max_abs_error", 1.0,
        )) != 0
        or int(value.get("peak_rss_bytes", -1)) < 0
        or int(value.get("peak_cuda_bytes", -1)) < 0
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("fusion execution acceptance semantics differ")
    return digest


def _initialization_lineage(spec: Mapping[str, Any], node_id: str):
    if NODE_REGISTRY[node_id].initialization == "fresh":
        return {}, {}
    sources = _source_lineage(spec)
    if node_id in {"FUSION_WITHDRAW_COS", "FUSION_WITHDRAW_STEP"}:
        report = load_json(
            Path(spec["campaign_root"]) / "training" / TEACHER_NODE
            / "training_report.json"
        )
        return {
            "source_report": report["content_hash"],
            "source_checkpoint": report["selected_checkpoint_sha256"],
        }, {
            "initialization_teacher_report": report["content_hash"],
            "initialization_teacher_checkpoint": report["selected_checkpoint_sha256"],
        }
    m0 = sources["m0_report"]
    parents = {
        "initialization_m0_report": m0["content_hash"],
        "initialization_m0_checkpoint": m0["selected_checkpoint_sha256"],
    }
    if node_id.startswith("ANCHORED_"):
        u000 = sources["u000_report"]
        parents.update({
            "initialization_u000_report": u000["content_hash"],
            "initialization_u000_checkpoint": u000["selected_checkpoint_sha256"],
        })
    return {
        "source_report": m0["content_hash"],
        "source_checkpoint": m0["selected_checkpoint_sha256"],
    }, parents


def run_fit(
    spec: Mapping[str, Any], node_id: str, *, device: str = "cuda",
    recovery_spec_sha256: str | None = None,
    execution_source_commit: str | None = None,
):
    _configure_deterministic_backend()
    if node_id not in NODE_REGISTRY:
        raise KeyError("unknown fusion fit")
    node = NODE_REGISTRY[node_id]
    use_hlt = node_id in {"HLT_WARM_CONTINUE", "FUSION_DIRECT_KD_WARM"}
    started = time.monotonic()
    if use_hlt:
        caches, split_hash, selection_hash, input_key = _hlt_caches(spec)
        protocol = "standard_hlt_v1"
    else:
        caches, split_hash, selection_hash = _concat_caches(spec)
        input_key = "privileged"
        protocol = (
            "anchored_withdrawal_v1"
            if node_id in {"FUSION_WITHDRAW_COS", "FUSION_WITHDRAW_STEP"}
            else "offline_hlt_fusion_v1"
        )
    targets = None
    if node_id in STUDY_C_NODES:
        targets = FusionProbabilityTargets.load(
            Path(spec["campaign_root"]) / "probabilities" / TEACHER_DISTRIBUTION
            / "train_manifest.json"
        )
    init, init_parents = _initialization_lineage(spec, node_id)
    parents = {
        "campaign_spec": spec["content_hash"],
        "source_lock": spec["parents"]["source_lock"],
        "foundation": spec["parents"]["foundation"],
        "graph": GRAPH_SHA256, "recipe": spec["parents"]["recipe"],
        "capacity_audit": validate_capacity_audit(
            spec, load_json(spec["artifact_paths"]["capacity_audit"]),
        ),
        "execution_acceptance": validate_execution_acceptance(
            spec, load_json(spec["artifact_paths"]["execution_acceptance"]),
        ),
        "split_manifest": split_hash, "selection_manifest": selection_hash,
        **init_parents,
    }
    if targets is not None:
        teacher_report = load_json(
            Path(spec["campaign_root"]) / "training" / TEACHER_NODE
            / "training_report.json"
        )
        lock_path = (
            Path(spec["campaign_root"]) / "probabilities" / TEACHER_DISTRIBUTION
            / "lock.json"
        )
        parents["teacher_probability_lock"] = validate_probability_lock(
            lock_path, campaign_spec_sha256=spec["content_hash"],
            teacher_report_sha256=teacher_report["content_hash"],
        )
    if recovery_spec_sha256 is not None:
        parents["recovery_spec"] = recovery_spec_sha256
    early = None
    lr = None
    if node_id in STUDY_C_NODES:
        early = {
            "kind": "macro_auc_patience_v1", "minimum_passes": 60,
            "patience_passes": 15, "minimum_auc_delta": 1e-5,
        }
        lr = {
            "kind": "warmup_hold_cosine_floor_tail_v1",
            "warmup_passes": 3, "hold_through_pass": 45,
            "decay_through_pass": 60, "minimum_lr_fraction": .05,
        }
    withdrawal = {
        "FUSION_WITHDRAW_COS": COSINE_ALPHA,
        "FUSION_WITHDRAW_STEP": STEP_ALPHA,
    }.get(node_id)
    try:
        return train_tri60_node(
            node_id=node_id, train_cache=caches["train"],
            validation_cache=caches["validation"], input_key=input_key,
            probability_targets=targets,
            output_dir=Path(spec["campaign_root"]) / "training" / node_id,
            parents=parents, campaign_spec_sha256=spec["content_hash"],
            recipe_sha256=spec["parents"]["recipe"],
            execution_source_commit=execution_source_commit or spec["source_commit"],
            replicate_seed=int(spec["replicate_seed"]), device=device,
            runtime=_runtime(node_id), model_factory=model_factory(spec, node_id),
            authority=training_authority(node_id), model_input_protocol=protocol,
            initialization_lineage=init, learning_rate_schedule=lr,
            early_stopping=early, withdrawal_schedule=withdrawal,
            preparation_metrics={
                "student_view_cache_seconds": time.monotonic() - started,
            },
        )
    finally:
        caches.clear()
        gc.collect()


def _infer_fusion(model, cache, *, sampler_seed: int, device: str):
    import torch
    identities, logits, labels = [], [], []
    model.to(device).eval()
    batches = cache.iterate_batches(epoch=0, sampler_seed=sampler_seed, batch_size=256)
    with torch.inference_mode(), _BatchPrefetcher(batches) as prefetched:
        for batch in prefetched:
            view = batch["privileged"]
            value = model(
                torch.as_tensor(view.features, device=device).float(),
                torch.as_tensor(view.vectors, device=device).float(),
                torch.as_tensor(view.mask, device=device).bool(),
                torch.as_tensor(view.content_source_codes, device=device).to(torch.int8),
            ).float().cpu().numpy()
            identities.append(np.ascontiguousarray(batch["identity_digests"], dtype=np.uint8))
            logits.append(np.ascontiguousarray(value, dtype=np.float32))
            labels.append(np.ascontiguousarray(batch["labels"], dtype=np.int64))
    return np.concatenate(identities), np.concatenate(logits), np.concatenate(labels)


def run_teacher_bank(
    spec: Mapping[str, Any], *, device: str = "cuda",
    recovery_spec_sha256: str | None = None,
    execution_source_commit: str | None = None,
):
    report_path = (
        Path(spec["campaign_root"]) / "training" / TEACHER_NODE
        / "training_report.json"
    )
    model, report = load_tri60_model(
        report_path, device=device, model_factory=model_factory(spec, TEACHER_NODE),
        authority=training_authority(TEACHER_NODE),
    )
    caches, _, _ = _concat_caches(spec)
    root = Path(spec["campaign_root"]) / "probabilities" / TEACHER_DISTRIBUTION
    seed = derive_seed(
        int(spec["replicate_seed"]), NODE_REGISTRY[TEACHER_NODE].seed_alias + "/sampler",
    )
    manifests = {}
    try:
        for role in ("train", "validation"):
            identities, logits, _ = _infer_fusion(
                model, caches[role], sampler_seed=seed, device=device,
            )
            manifests[role] = publish_role(
                root, role=role, identity_digests=identities, logits=logits,
                teacher_report_sha256=report["content_hash"],
                teacher_checkpoint_sha256=report["selected_checkpoint_sha256"],
                campaign_spec_sha256=spec["content_hash"],
                producer_commit=execution_source_commit or spec["source_commit"],
            )
        return publish_lock(
            root / "lock.json", train_manifest=manifests["train"],
            validation_manifest=manifests["validation"],
            campaign_spec_sha256=spec["content_hash"],
            teacher_report_sha256=report["content_hash"],
        )
    finally:
        caches.clear()
        del model
        gc.collect()


def run_alpha_zero_extraction(
    spec: Mapping[str, Any], node_id: str, *, device: str = "cuda",
    recovery_spec_sha256: str | None = None,
):
    import torch
    if node_id not in {"FUSION_WITHDRAW_COS", "FUSION_WITHDRAW_STEP"}:
        raise KeyError("alpha-zero extraction node differs")
    report_path = Path(spec["campaign_root"]) / "training" / node_id / "training_report.json"
    model, report = load_tri60_model(
        report_path, device=device, model_factory=model_factory(spec, node_id),
        authority=training_authority(node_id),
    )
    model.eval()
    batch = _first_batch(spec, role="validation", batch_size=256)
    view = batch["privileged"]
    args = (
        torch.as_tensor(view.features, device=device).float(),
        torch.as_tensor(view.vectors, device=device).float(),
        torch.as_tensor(view.mask, device=device).bool(),
        torch.as_tensor(view.content_source_codes, device=device).to(torch.int8),
    )
    extracted = model.extract_hlt().to(device).eval()
    with torch.inference_mode():
        expected = model.forward_zero(*args).logits.float()
        offline = args[3] == 0
        perturbed = model.forward_zero(
            args[0].masked_fill(offline[:, None], float("nan")),
            args[1].masked_fill(offline[:, None], float("nan")),
            args[2], args[3],
        ).logits.float()
        hlt = _content_view(*args, code=1, capacity=200)
        actual = extracted(*hlt).float()
    error = float((expected - actual).abs().max().item())
    perturbation_error = float((expected - perturbed).abs().max().item())
    if error > 2e-5 or perturbation_error != 0:
        raise RuntimeError("alpha-zero extracted checkpoint parity differs")
    output = Path(spec["campaign_root"]) / "deployable" / node_id
    checkpoint = {
        "contract": DEPLOYABLE_CHECKPOINT_CONTRACT, "schema_version": 1,
        "node_id": node_id, "campaign_spec_sha256": spec["content_hash"],
        "source_training_report_sha256": report["content_hash"],
        "source_selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
        "model_contract": "hlt_classification_scouting_part_v1",
        "model": {name: value.detach().cpu() for name, value in extracted.state_dict().items()},
        "input_fields": ["features", "vectors", "mask"],
        "offline_modules_present": False,
        "final_test_accessed": False,
    }
    checkpoint_path = output / "selected_hlt_model.pt"
    atomic_publish_bytes(checkpoint_path, _torch_bytes(checkpoint))
    audit = artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "training_report": report["content_hash"],
            "training_checkpoint": report["selected_checkpoint_sha256"],
            **({} if recovery_spec_sha256 is None else {
                "recovery_spec": recovery_spec_sha256,
            }),
        },
        "node_id": node_id, "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parity_rows": len(batch["labels"]), "maximum_abs_error": error,
        "exact_ordinary_hlt_state_projection": True,
        "offline_context_dispatch_skipped": True,
        "offline_perturbation_max_abs_error": perturbation_error,
        "offline_perturbation_invariant": True,
        "state_keys_are_ordinary_mod_only": all(
            name.startswith("mod.") for name in extracted.state_dict()
        ),
        "input_fields": ["features", "vectors", "mask"],
        "final_test_accessed": False,
    }, contract=ALPHA_ZERO_AUDIT_CONTRACT)
    write_immutable_json(output / "alpha_zero_audit.json", audit)
    return audit


__all__ = [
    "build_capacity_audit", "model_factory", "run_alpha_zero_extraction",
    "run_execution_acceptance", "run_fit", "run_teacher_bank",
    "training_authority", "validate_capacity_audit",
    "validate_execution_acceptance",
]

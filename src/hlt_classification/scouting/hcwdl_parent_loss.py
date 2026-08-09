"""Versioned HCWDL base-loss semantics and parent-loss attestations.

The historical PMARD helper deliberately remains untouched: completed PMARD
artifacts used class weights for every row loss.  HCWDL's authoritative
scientific plan instead weights only cross entropy and reduces both KD terms
as ordinary row means.  Keeping that distinction in a separately named,
versioned surface prevents an old report from acquiring new meaning merely
because the shared source changed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
)
from hlt_classification.provenance import (
    capture_source_snapshot,
    validate_source_snapshot_payload,
)

from .training import LossConfiguration
from .hcwdl_representation_contracts import logical_array_sha256
from .hcwdl_recipe import (
    CLASS_WEIGHT_POLICY,
    PRIMARY_RECIPE_PROFILE,
    RECIPE_CONTRACT,
    validate_recipe,
)


HCWDL_PARENT_BASE_LOSS_CONTRACT: Final = "HCWDL_PARENT_BASE_LOSS/v1"
HCWDL_PARENT_LOSS_ATTESTATION_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_PARENT_LOSS_ATTESTATION/v3"
)
PARENT_RECIPE_AUTHORITY_KEYS: Final = frozenset({
    "parent_recipe_contract",
    "parent_recipe_sha256",
    "parent_recipe_profile",
    "parent_class_weight_policy",
})
PARENT_LOSS_RUNTIME_SOURCE_FILES: Final[Mapping[str, str]] = MappingProxyType({
    "engine": "src/hlt_classification/scouting/engine.py",
    "parent_loss": "src/hlt_classification/scouting/hcwdl_parent_loss.py",
    "training": "src/hlt_classification/scouting/hcwdl_training.py",
})
HCWDL_PARENT_LOSS_SEMANTICS: Final = {
    "ce_reduction": "class_weighted_row_mean",
    "kd_reduction": "unweighted_row_mean",
    "kd_direction": "forward_kl_teacher_to_student",
    "temperature_correction": "tau_squared",
    "loss_dtype": "float32",
}
HCWDL_PARENT_LOSS_POLICY: Final = {
    "semantic_contract": HCWDL_PARENT_BASE_LOSS_CONTRACT,
    "semantics": HCWDL_PARENT_LOSS_SEMANTICS,
    "required_parent_recipe": {
        "contract": RECIPE_CONTRACT,
        "recipe_profile": PRIMARY_RECIPE_PROFILE,
        "class_weight_policy": CLASS_WEIGHT_POLICY,
        "class_weights": [1.0] * 15,
    },
    "required_parent_evidence": (
        "authenticated_parent_recipe_bytes",
        "recipe_hash_in_wrapper_engine_and_checkpoint",
        "training_report_bytes",
        "selected_checkpoint_bytes",
        "final_checkpoint_bytes",
        "exact_60_pass_execution_config",
        "exact_60_validation_records",
        "producer_runtime_source_bytes",
    ),
    "legacy_weighted_kd_is_acceptable": False,
}
HCWDL_PARENT_LOSS_POLICY_SHA256: Final = canonical_sha256(
    HCWDL_PARENT_LOSS_POLICY,
)


def _validated_parent_recipe_authority(
    parent_recipe: Mapping[str, Any],
) -> dict[str, str]:
    """Authenticate the one parent recipe that can authorize RKD import."""

    if parent_recipe.get("contract") != RECIPE_CONTRACT:
        raise ValueError(
            "parent-loss attestation requires an authenticated HCWDL_RECIPE/v4 parent"
        )
    digest = validate_recipe(
        parent_recipe,
        require_authorized=True,
        expected_profile=PRIMARY_RECIPE_PROFILE,
    )
    weighting = parent_recipe.get("class_weighting")
    if (
        not isinstance(weighting, Mapping)
        or weighting.get("policy") != CLASS_WEIGHT_POLICY
    ):
        raise ValueError("parent-loss attestation requires the v4 unweighted policy")
    weights = np.asarray(parent_recipe.get("class_weights"), dtype=np.float32)
    if weights.shape != (15,) or not np.array_equal(weights, np.ones(15, np.float32)):
        raise ValueError("parent-loss attestation requires fifteen exact one weights")
    return {
        "parent_recipe_contract": RECIPE_CONTRACT,
        "parent_recipe_sha256": digest,
        "parent_recipe_profile": PRIMARY_RECIPE_PROFILE,
        "parent_class_weight_policy": CLASS_WEIGHT_POLICY,
    }


def _validate_inputs(student_logits, labels, class_weights):
    import torch

    if student_logits.ndim != 2 or student_logits.shape[1] != 15:
        raise ValueError("HCWDL student logits must be [batch,15]")
    if labels.ndim != 1 or labels.shape[0] != student_logits.shape[0]:
        raise ValueError("HCWDL labels must be [batch]")
    labels = labels.to(device=student_logits.device, dtype=torch.long)
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= 15):
        raise ValueError("HCWDL labels lie outside the 15-class contract")
    weights = torch.as_tensor(
        class_weights, device=student_logits.device, dtype=torch.float32,
    )
    if weights.shape != (15,) or not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("HCWDL class weights must be 15 finite positive values")
    return student_logits.float(), labels, weights


def _kd_rows(student_fp32, teacher, *, temperature: float):
    import torch
    import torch.nn.functional as functional

    if teacher is None:
        raise ValueError("required HCWDL teacher logits are absent")
    if teacher.shape != student_fp32.shape or teacher.requires_grad:
        raise ValueError("HCWDL teacher logits must be shape-matched and detached")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("HCWDL KD temperature must be finite and positive")
    teacher_fp32 = teacher.float()
    if not torch.isfinite(teacher_fp32).all():
        raise FloatingPointError("HCWDL teacher logits are nonfinite")
    target = functional.softmax(teacher_fp32 / temperature, dim=-1)
    return functional.kl_div(
        functional.log_softmax(student_fp32 / temperature, dim=-1),
        target,
        reduction="none",
    ).sum(-1) * temperature * temperature


def hcwdl_base_loss_rows(
    student_logits,
    labels,
    *,
    class_weights,
    configuration: LossConfiguration,
    hlt_teacher_logits=None,
    privileged_teacher_logits=None,
) -> dict[str, Any]:
    """Return FP32 per-row HCWDL components without applying reductions.

    ``weighted_ce_rows`` is the quantity used by the ordinary training base
    loss.  ``total_rows`` is intentionally the coefficient-weighted mixture
    before reduction and exists for support-matched gradient calibration.
    Calibration applies its own declared class-weighted reduction to these
    rows; it does not change the training-time unweighted KD reductions.
    """

    import torch
    import torch.nn.functional as functional

    student_fp32, labels, weights = _validate_inputs(
        student_logits, labels, class_weights,
    )
    if not torch.isfinite(student_fp32).all():
        raise FloatingPointError("HCWDL student logits are nonfinite")
    ce_rows = functional.cross_entropy(student_fp32, labels, reduction="none")
    zero = student_fp32.sum(-1) * 0.0
    hlt_rows = (
        _kd_rows(
            student_fp32, hlt_teacher_logits,
            temperature=configuration.hlt_temperature,
        )
        if configuration.hlt_kd else zero
    )
    privileged_rows = (
        _kd_rows(
            student_fp32, privileged_teacher_logits,
            temperature=configuration.effective_privileged_temperature,
        )
        if configuration.privileged_kd else zero
    )
    total_rows = (
        configuration.ce * ce_rows
        + configuration.hlt_kd * hlt_rows
        + configuration.privileged_kd * privileged_rows
    )
    result = {
        "ce_rows": ce_rows,
        "weighted_ce_rows": weights[labels] * ce_rows,
        "hlt_kd_rows": hlt_rows,
        "privileged_kd_rows": privileged_rows,
        "total_rows": total_rows,
        "labels": labels,
        "row_class_weights": weights[labels],
    }
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("HCWDL base-loss rows are nonfinite")
    return result


def hcwdl_base_loss(
    student_logits,
    labels,
    *,
    class_weights,
    configuration: LossConfiguration,
    hlt_teacher_logits=None,
    privileged_teacher_logits=None,
) -> dict[str, Any]:
    """Compute the locked HCWDL CE/KD objective in FP32.

    CE is ``mean(class_weight[label] * CE_row)``.  Each active KD component is
    the unweighted row mean.  This function must be explicitly selected by an
    HCWDL execution; it is not the default behavior of :func:`pmard_loss`.
    """

    import torch

    rows = hcwdl_base_loss_rows(
        student_logits,
        labels,
        class_weights=class_weights,
        configuration=configuration,
        hlt_teacher_logits=hlt_teacher_logits,
        privileged_teacher_logits=privileged_teacher_logits,
    )
    zero = rows["total_rows"].sum() * 0.0
    components = {
        "ce": rows["weighted_ce_rows"].mean(),
        "hlt_kd": rows["hlt_kd_rows"].mean() if configuration.hlt_kd else zero,
        "privileged_kd": (
            rows["privileged_kd_rows"].mean()
            if configuration.privileged_kd else zero
        ),
    }
    components["total"] = (
        configuration.ce * components["ce"]
        + configuration.hlt_kd * components["hlt_kd"]
        + configuration.privileged_kd * components["privileged_kd"]
    )
    if any(not torch.isfinite(value) for value in components.values()):
        raise FloatingPointError("HCWDL base loss is nonfinite")
    return components


def _tensor_logical_sha256(name: str, value) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return logical_array_sha256(name, array)


def parent_loss_runtime_fingerprint() -> dict[str, Any]:
    """Execute a frozen nonuniform fixture that distinguishes both semantics."""

    import torch

    student = torch.linspace(-1.4, 1.8, 60, dtype=torch.float32).reshape(4, 15)
    student.requires_grad_(True)
    labels = torch.tensor([0, 3, 8, 14])
    class_weights = torch.linspace(0.35, 2.45, 15)
    hlt = torch.linspace(1.1, -0.9, 60, dtype=torch.float32).reshape(4, 15)
    privileged = torch.sin(torch.arange(60, dtype=torch.float32)).reshape(4, 15)
    configuration = LossConfiguration.for_mixture(
        arm="HCWDL_PARENT_LOSS_ATTESTATION_FIXTURE",
        ce=0.25,
        hlt_kd=0.40,
        privileged_kd=0.35,
        hlt_temperature=1.0,
        privileged_temperature=2.0,
    )
    parts = hcwdl_base_loss(
        student,
        labels,
        class_weights=class_weights,
        configuration=configuration,
        hlt_teacher_logits=hlt,
        privileged_teacher_logits=privileged,
    )
    parts["total"].backward()
    assert student.grad is not None
    return {
        "fixture_contract": "HCWDL_PARENT_BASE_LOSS_FIXTURE/v1",
        "component_hex": {
            name: float(value.detach().cpu()).hex()
            for name, value in sorted(parts.items())
        },
        "student_gradient_sha256": _tensor_logical_sha256(
            "student_gradient", student.grad,
        ),
    }


def _validate_parent_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_recipe_authority: Mapping[str, str],
    expected_runtime_source_sha256: str,
    expected_total_updates: int,
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("parent-loss attestation requires parent artifacts")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        row = dict(raw)
        required = {
            "node_id", "training_report_sha256", "checkpoint_sha256",
            "final_checkpoint_sha256", "execution_config_sha256",
            "completed_updates", "validation_record_count",
            "loss_semantics_contract", "producer_source_sha256",
        } | PARENT_RECIPE_AUTHORITY_KEYS
        if set(row) != required:
            raise ValueError("parent-loss evidence fields differ")
        node = str(row["node_id"])
        if not node or node in seen:
            raise ValueError("parent-loss evidence node IDs are invalid")
        seen.add(node)
        if row["loss_semantics_contract"] != HCWDL_PARENT_BASE_LOSS_CONTRACT:
            raise ValueError("parent artifact used incompatible loss semantics")
        if any(
            row.get(name) != expected_recipe_authority[name]
            for name in PARENT_RECIPE_AUTHORITY_KEYS
        ):
            raise ValueError("parent artifact used a different parent recipe authority")
        for name in (
            "training_report_sha256", "checkpoint_sha256",
            "final_checkpoint_sha256", "execution_config_sha256",
            "producer_source_sha256",
        ):
            row[name] = require_sha256(row[name], name=f"{node} {name}")
        if row["producer_source_sha256"] != expected_runtime_source_sha256:
            raise ValueError("parent artifact producer source differs from runtime registry")
        if (
            row["completed_updates"] != expected_total_updates
            or row["validation_record_count"] != 60
        ):
            raise ValueError("parent artifact is not an exact completed 60-pass execution")
        row["node_id"] = node
        result.append(row)
    from .hcwdl_ladder import NODE_REGISTRY

    if seen != set(NODE_REGISTRY):
        raise ValueError("parent-loss attestation must cover the exact 23-node screen")
    return sorted(result, key=lambda item: item["node_id"])


def _validated_parent_campaign_authority(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize the exact v8 prefix fields carried by the attestation."""

    from .hcwdl_authorization import (
        AUTOMATIC_ENDPOINT_CONTINUATION,
        PARENT_PREFIX_SCOPE,
    )
    from .hcwdl_campaign import (
        MODES, PARENT_PREFIX_CAMPAIGN_CONTRACT, ROLE_COUNTS,
    )

    expected_keys = {
        "parent_campaign_contract", "parent_campaign_mode",
        "parent_execution_scope", "parent_endpoint_continuation",
        "parent_training_passes", "parent_validation_every_passes",
        "parent_train_rows", "parent_terminal_task_id",
        "parent_execution_lock_authorized",
        "parent_final_test_access_authorized",
        "parent_registered_final_test_tasks",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError("parent-loss campaign authority fields differ")
    mode = value.get("parent_campaign_mode")
    train_rows = value.get("parent_train_rows")
    expected_train_rows = (
        None if mode not in MODES else ROLE_COUNTS[str(mode)]["train"]
    )
    expected = {
        "parent_campaign_contract": PARENT_PREFIX_CAMPAIGN_CONTRACT,
        "parent_execution_scope": PARENT_PREFIX_SCOPE,
        "parent_endpoint_continuation": AUTOMATIC_ENDPOINT_CONTINUATION,
        "parent_training_passes": 60,
        "parent_validation_every_passes": 1,
        "parent_terminal_task_id": "finalist_lock",
        "parent_execution_lock_authorized": False,
        "parent_final_test_access_authorized": False,
        "parent_registered_final_test_tasks": 0,
    }
    if (
        mode not in MODES
        or mode == "smoke"
        or expected_train_rows is None
        or isinstance(train_rows, bool)
        or not isinstance(train_rows, int)
        or train_rows != expected_train_rows
        or any(value.get(name) != item for name, item in expected.items())
    ):
        raise PermissionError(
            "parent-loss attestation requires the exact non-smoke v8 parent prefix"
        )
    return dict(value)


def _validated_runtime_source_registry(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("parent-loss runtime-source registry differs")
    by_name: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != {
            "logical_name", "relative_path", "sha256",
        }:
            raise ValueError("parent-loss runtime-source row differs")
        logical_name = raw.get("logical_name")
        if not isinstance(logical_name, str) or logical_name in by_name:
            raise ValueError("parent-loss runtime-source logical names differ")
        by_name[logical_name] = raw
    if set(by_name) != set(PARENT_LOSS_RUNTIME_SOURCE_FILES):
        raise ValueError("parent-loss runtime-source registry is incomplete or expanded")
    normalized: list[dict[str, str]] = []
    for logical_name, relative_path in sorted(PARENT_LOSS_RUNTIME_SOURCE_FILES.items()):
        row = by_name[logical_name]
        if row.get("relative_path") != relative_path:
            raise ValueError(
                f"parent-loss runtime-source relative path differs: {logical_name}"
            )
        normalized.append({
            "logical_name": logical_name,
            "relative_path": relative_path,
            "sha256": require_sha256(
                row.get("sha256"), name=f"{logical_name} runtime source",
            ),
        })
    return normalized


def _validated_source_authority(
    *, parent_campaign_spec_sha256: str,
    parent_campaign_authority: Mapping[str, Any],
    parent_source_snapshot: Mapping[str, Any],
    runtime_source_registry: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    campaign_sha256 = require_sha256(
        parent_campaign_spec_sha256, name="parent campaign specification",
    )
    validate_source_snapshot_payload(parent_source_snapshot)
    if parent_source_snapshot.get("worktree_clean") is not True:
        raise ValueError("parent-loss producer source checkout is not clean")
    source_commit = parent_source_snapshot.get("git_commit")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError("parent-loss producer source commit differs")
    registry = _validated_runtime_source_registry(runtime_source_registry)
    return {
        "parent_campaign_spec_sha256": campaign_sha256,
        "parent_campaign_authority": _validated_parent_campaign_authority(
            parent_campaign_authority
        ),
        "parent_source_commit": source_commit,
        "parent_source_snapshot": dict(parent_source_snapshot),
        "runtime_source_registry": registry,
        "runtime_source_sha256": canonical_sha256(registry),
    }


def _source_authority_from_files(
    *,
    parent_campaign_spec_path: str | Path,
    runtime_source_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    from .hcwdl_training import validate_hcwdl_parent_prefix_campaign

    campaign = load_json(Path(parent_campaign_spec_path))
    campaign_sha256 = validate_hcwdl_parent_prefix_campaign(
        campaign, executable=True,
    )
    if set(runtime_source_paths) != set(PARENT_LOSS_RUNTIME_SOURCE_FILES):
        raise ValueError("parent-loss runtime-source registry is incomplete or expanded")

    project_roots: set[Path] = set()
    registry: list[dict[str, str]] = []
    for logical_name, relative_text in sorted(PARENT_LOSS_RUNTIME_SOURCE_FILES.items()):
        candidate = Path(runtime_source_paths[logical_name])
        if (
            not candidate.is_absolute()
            or not candidate.is_file()
            or candidate.is_symlink()
        ):
            raise ValueError(
                f"parent-loss runtime source is not an absolute regular file: {logical_name}"
            )
        resolved = candidate.resolve()
        project_root = resolved
        for _ in Path(relative_text).parts:
            project_root = project_root.parent
        if (project_root / relative_text).resolve() != resolved:
            raise ValueError(
                f"parent-loss runtime source path differs: {logical_name}"
            )
        project_roots.add(project_root)
        registry.append({
            "logical_name": logical_name,
            "relative_path": relative_text,
            "sha256": sha256_file(resolved),
        })
    if len(project_roots) != 1:
        raise ValueError("parent-loss runtime sources do not share one project checkout")
    project_root = next(iter(project_roots))
    snapshot = capture_source_snapshot(project_root, require_clean=True)
    validate_source_snapshot_payload(snapshot)
    if snapshot.get("git_commit") != campaign.get("source_commit"):
        raise PermissionError(
            "parent-loss runtime source checkout differs from parent campaign commit"
        )
    return _validated_source_authority(
        parent_campaign_spec_sha256=campaign_sha256,
        parent_campaign_authority={
            "parent_campaign_contract": campaign["contract"],
            "parent_campaign_mode": campaign["mode"],
            "parent_execution_scope": campaign["execution_scope"],
            "parent_endpoint_continuation": campaign["endpoint_continuation"],
            "parent_training_passes": campaign["training_passes"],
            "parent_validation_every_passes": campaign[
                "validation_every_passes"
            ],
            "parent_train_rows": campaign["role_counts"]["train"],
            "parent_terminal_task_id": campaign["terminal_task_id"],
            "parent_execution_lock_authorized": campaign[
                "execution_lock_authorized"
            ],
            "parent_final_test_access_authorized": campaign[
                "final_test_access_authorized"
            ],
            "parent_registered_final_test_tasks": campaign[
                "registered_final_test_tasks"
            ],
        },
        parent_source_snapshot=snapshot,
        runtime_source_registry=registry,
    )


def parent_artifact_evidence_from_report(
    *, node_id: str, training_report_path: str | Path,
    producer_source_sha256: str, parent_recipe: Mapping[str, Any],
    train_rows: int,
) -> dict[str, Any]:
    """Open and verify one corrected parent report and selected checkpoint.

    Attestation callers may not provide bare hashes.  This function follows
    the HCWDL wrapper report into its PMARD engine report and checkpoint,
    validates both versioned semantics, re-hashes the checkpoint bytes, and
    verifies the serialized checkpoint semantic binding.
    """

    from .hcwdl_training import validate_hcwdl_full_parent_wrapper_report

    recipe_authority = _validated_parent_recipe_authority(parent_recipe)
    expected_recipe_sha256 = recipe_authority["parent_recipe_sha256"]
    path = Path(training_report_path)
    wrapper = load_json(path)
    evidence = validate_hcwdl_full_parent_wrapper_report(
        wrapper, training_report_path=path, train_rows=train_rows,
        recipe=parent_recipe, expected_node_id=node_id,
    )
    if wrapper.get("recipe_sha256") != expected_recipe_sha256:
        raise ValueError(f"parent wrapper recipe authority differs: {node_id}")
    return {
        "node_id": node_id,
        "training_report_sha256": evidence["wrapper_sha256"],
        "checkpoint_sha256": evidence["selected_checkpoint_sha256"],
        "final_checkpoint_sha256": evidence["final_checkpoint_sha256"],
        "execution_config_sha256": evidence["execution_config_sha256"],
        "completed_updates": evidence["completed_updates"],
        "validation_record_count": evidence["validation_record_count"],
        "loss_semantics_contract": HCWDL_PARENT_BASE_LOSS_CONTRACT,
        "producer_source_sha256": require_sha256(
            producer_source_sha256, name=f"{node_id} producer source",
        ),
        **recipe_authority,
    }


def build_parent_loss_attestation_from_reports(
    *, parent_recipe_path: str | Path,
    parent_campaign_spec_path: str | Path,
    parent_reports: Mapping[str, str | Path],
    runtime_source_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Construct the attestation from executable source/report/checkpoint bytes.

    The scientific policy is a versioned code constant.  Runtime workers do
    not read the implementation plan Markdown or any other concept document.
    """

    if (
        not parent_reports or not isinstance(runtime_source_paths, Mapping)
        or set(runtime_source_paths) != set(PARENT_LOSS_RUNTIME_SOURCE_FILES)
    ):
        raise ValueError("parent-loss attestation file registry is empty")
    parent_recipe = load_json(Path(parent_recipe_path))
    _validated_parent_recipe_authority(parent_recipe)
    source_authority = _source_authority_from_files(
        parent_campaign_spec_path=parent_campaign_spec_path,
        runtime_source_paths=runtime_source_paths,
    )
    runtime_source_sha256 = source_authority["runtime_source_sha256"]
    train_rows = int(
        source_authority["parent_campaign_authority"]["parent_train_rows"]
    )
    evidence = [
        parent_artifact_evidence_from_report(
            node_id=node_id, training_report_path=path,
            producer_source_sha256=runtime_source_sha256,
            parent_recipe=parent_recipe,
            train_rows=train_rows,
        )
        for node_id, path in sorted(parent_reports.items())
    ]
    return build_parent_loss_attestation(
        parent_recipe=parent_recipe,
        parent_artifacts=evidence,
        parent_campaign_spec_sha256=source_authority[
            "parent_campaign_spec_sha256"
        ],
        parent_campaign_authority=source_authority[
            "parent_campaign_authority"
        ],
        parent_source_snapshot=source_authority["parent_source_snapshot"],
        runtime_source_registry=source_authority["runtime_source_registry"],
    )


def build_parent_loss_attestation(
    *,
    parent_recipe: Mapping[str, Any],
    parent_artifacts: Sequence[Mapping[str, Any]],
    parent_campaign_spec_sha256: str,
    parent_campaign_authority: Mapping[str, Any],
    parent_source_snapshot: Mapping[str, Any],
    runtime_source_registry: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build an immutable attestation over corrected parent executions.

    Old reports cannot satisfy this function: each parent row must explicitly
    bind the versioned loss semantic, its report/checkpoint bytes, and the
    producer source.  The deterministic runtime fingerprint additionally
    prevents a matching string from attesting a different implementation.
    """

    recipe_authority = _validated_parent_recipe_authority(parent_recipe)
    source_authority = _validated_source_authority(
        parent_campaign_spec_sha256=parent_campaign_spec_sha256,
        parent_campaign_authority=parent_campaign_authority,
        parent_source_snapshot=parent_source_snapshot,
        runtime_source_registry=runtime_source_registry,
    )
    payload = with_content_hash({
        "contract": HCWDL_PARENT_LOSS_ATTESTATION_CONTRACT,
        "schema_version": 3,
        "parent_loss_policy": dict(HCWDL_PARENT_LOSS_POLICY),
        "parent_loss_policy_sha256": HCWDL_PARENT_LOSS_POLICY_SHA256,
        **source_authority,
        "loss_semantics_contract": HCWDL_PARENT_BASE_LOSS_CONTRACT,
        "loss_semantics": dict(HCWDL_PARENT_LOSS_SEMANTICS),
        "loss_semantics_sha256": canonical_sha256(HCWDL_PARENT_LOSS_SEMANTICS),
        "runtime_fingerprint": parent_loss_runtime_fingerprint(),
        **recipe_authority,
        "parent_artifacts": _validate_parent_rows(
            parent_artifacts,
            expected_recipe_authority=recipe_authority,
            expected_runtime_source_sha256=source_authority[
                "runtime_source_sha256"
            ],
            expected_total_updates=(
                60 * int(np.ceil(
                    source_authority["parent_campaign_authority"][
                        "parent_train_rows"
                    ]
                    / int(parent_recipe["batching"]["effective_batch_size"])
                ))
            ),
        ),
    })
    validate_parent_loss_attestation(payload, parent_recipe=parent_recipe)
    return payload


def validate_parent_loss_attestation(
    value: Mapping[str, Any], *, parent_recipe: Mapping[str, Any],
) -> str:
    recipe_authority = _validated_parent_recipe_authority(parent_recipe)
    digest = validate_content_hash(
        value,
        expected_contract=HCWDL_PARENT_LOSS_ATTESTATION_CONTRACT,
        expected_schema_version=3,
    )
    if (
        value.get("parent_loss_policy") != HCWDL_PARENT_LOSS_POLICY
        or value.get("parent_loss_policy_sha256")
        != HCWDL_PARENT_LOSS_POLICY_SHA256
    ):
        raise ValueError("parent-loss attestation executable policy differs")
    raw_snapshot = value.get("parent_source_snapshot")
    raw_registry = value.get("runtime_source_registry")
    if not isinstance(raw_snapshot, Mapping) or not isinstance(raw_registry, list):
        raise ValueError("parent-loss source authority differs")
    source_authority = _validated_source_authority(
        parent_campaign_spec_sha256=value.get("parent_campaign_spec_sha256"),
        parent_campaign_authority=value.get("parent_campaign_authority"),
        parent_source_snapshot=raw_snapshot,
        runtime_source_registry=raw_registry,
    )
    for name, expected in source_authority.items():
        if value.get(name) != expected:
            raise ValueError("parent-loss source authority differs")
    if value.get("loss_semantics_contract") != HCWDL_PARENT_BASE_LOSS_CONTRACT:
        raise ValueError("parent-loss attestation semantic contract differs")
    if value.get("loss_semantics") != HCWDL_PARENT_LOSS_SEMANTICS:
        raise ValueError("parent-loss attestation semantic payload differs")
    if value.get("loss_semantics_sha256") != canonical_sha256(HCWDL_PARENT_LOSS_SEMANTICS):
        raise ValueError("parent-loss attestation semantic hash differs")
    if value.get("runtime_fingerprint") != parent_loss_runtime_fingerprint():
        raise ValueError("parent-loss runtime fingerprint differs")
    if any(
        value.get(name) != recipe_authority[name]
        for name in PARENT_RECIPE_AUTHORITY_KEYS
    ):
        raise ValueError("parent-loss attestation parent recipe authority differs")
    rows = value.get("parent_artifacts")
    if not isinstance(rows, list) or rows != _validate_parent_rows(
        rows,
        expected_recipe_authority=recipe_authority,
        expected_runtime_source_sha256=source_authority[
            "runtime_source_sha256"
        ],
        expected_total_updates=(
            60 * int(np.ceil(
                source_authority["parent_campaign_authority"][
                    "parent_train_rows"
                ]
                / int(parent_recipe["batching"]["effective_batch_size"])
            ))
        ),
    ):
        raise ValueError("parent-loss artifact registry differs")
    return digest


__all__ = [
    "HCWDL_PARENT_BASE_LOSS_CONTRACT",
    "HCWDL_PARENT_LOSS_ATTESTATION_CONTRACT",
    "HCWDL_PARENT_LOSS_POLICY",
    "HCWDL_PARENT_LOSS_POLICY_SHA256",
    "HCWDL_PARENT_LOSS_SEMANTICS",
    "PARENT_LOSS_RUNTIME_SOURCE_FILES",
    "build_parent_loss_attestation",
    "build_parent_loss_attestation_from_reports",
    "hcwdl_base_loss",
    "hcwdl_base_loss_rows",
    "parent_loss_runtime_fingerprint",
    "parent_artifact_evidence_from_report",
    "validate_parent_loss_attestation",
]

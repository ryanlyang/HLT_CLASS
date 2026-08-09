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
from typing import Any, Final

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
)

from .training import LossConfiguration
from .hcwdl_representation_contracts import logical_array_sha256


HCWDL_PARENT_BASE_LOSS_CONTRACT: Final = "HCWDL_PARENT_BASE_LOSS/v1"
HCWDL_PARENT_LOSS_ATTESTATION_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_PARENT_LOSS_ATTESTATION/v1"
)
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
    "required_parent_evidence": (
        "training_report_bytes",
        "selected_checkpoint_bytes",
        "producer_runtime_source_bytes",
    ),
    "legacy_weighted_kd_is_acceptable": False,
}
HCWDL_PARENT_LOSS_POLICY_SHA256: Final = canonical_sha256(
    HCWDL_PARENT_LOSS_POLICY,
)


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


def _validate_parent_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("parent-loss attestation requires parent artifacts")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        row = dict(raw)
        required = {
            "node_id", "training_report_sha256", "checkpoint_sha256",
            "loss_semantics_contract", "producer_source_sha256",
        }
        if set(row) != required:
            raise ValueError("parent-loss evidence fields differ")
        node = str(row["node_id"])
        if not node or node in seen:
            raise ValueError("parent-loss evidence node IDs are invalid")
        seen.add(node)
        if row["loss_semantics_contract"] != HCWDL_PARENT_BASE_LOSS_CONTRACT:
            raise ValueError("parent artifact used incompatible loss semantics")
        for name in (
            "training_report_sha256", "checkpoint_sha256", "producer_source_sha256",
        ):
            row[name] = require_sha256(row[name], name=f"{node} {name}")
        row["node_id"] = node
        result.append(row)
    return sorted(result, key=lambda item: item["node_id"])


def parent_artifact_evidence_from_report(
    *, node_id: str, training_report_path: str | Path,
    producer_source_sha256: str,
) -> dict[str, Any]:
    """Open and verify one corrected parent report and selected checkpoint.

    Attestation callers may not provide bare hashes.  This function follows
    the HCWDL wrapper report into its PMARD engine report and checkpoint,
    validates both versioned semantics, re-hashes the checkpoint bytes, and
    verifies the serialized checkpoint semantic binding.
    """

    import torch

    from .engine import validate_pmard_training_report
    from .hcwdl_training import validate_hcwdl_training_report

    path = Path(training_report_path)
    wrapper = load_json(path)
    wrapper_hash = validate_hcwdl_training_report(wrapper)
    if wrapper.get("node_id") != node_id or wrapper.get("complete") is not True:
        raise ValueError(f"corrected parent report identity/completion differs: {node_id}")
    engine_path = path.parent / "training_report.json"
    engine = load_json(engine_path)
    engine_hash = validate_pmard_training_report(engine)
    if engine_hash != wrapper.get("pmard_engine_report_sha256"):
        raise ValueError(f"parent engine-report lineage differs: {node_id}")
    if any(
        engine.get(name) != expected
        for name, expected in {
            "loss_semantics_contract": HCWDL_PARENT_BASE_LOSS_CONTRACT,
            "loss_semantics": HCWDL_PARENT_LOSS_SEMANTICS,
            "loss_semantics_sha256": canonical_sha256(HCWDL_PARENT_LOSS_SEMANTICS),
            "execution_config_sha256": wrapper.get("pmard_execution_config_sha256"),
        }.items()
    ):
        raise ValueError(f"parent engine loss semantics differ: {node_id}")
    checkpoint_path = engine_path.parent / str(engine.get("selected_checkpoint", ""))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"parent selected checkpoint is absent: {checkpoint_path}")
    checkpoint_bytes = sha256_file(checkpoint_path)
    if checkpoint_bytes != engine.get("selected_checkpoint_sha256") or (
        checkpoint_bytes != wrapper.get("selected_checkpoint_sha256")
    ):
        raise ValueError(f"parent selected-checkpoint byte lineage differs: {node_id}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or any(
        checkpoint.get(name) != expected
        for name, expected in {
            "loss_semantics_contract": HCWDL_PARENT_BASE_LOSS_CONTRACT,
            "loss_semantics": HCWDL_PARENT_LOSS_SEMANTICS,
            "loss_semantics_sha256": canonical_sha256(HCWDL_PARENT_LOSS_SEMANTICS),
            "execution_config_sha256": wrapper.get("pmard_execution_config_sha256"),
        }.items()
    ):
        raise ValueError(f"parent checkpoint loss semantics differ: {node_id}")
    return {
        "node_id": node_id,
        "training_report_sha256": wrapper_hash,
        "checkpoint_sha256": checkpoint_bytes,
        "loss_semantics_contract": HCWDL_PARENT_BASE_LOSS_CONTRACT,
        "producer_source_sha256": require_sha256(
            producer_source_sha256, name=f"{node_id} producer source",
        ),
    }


def build_parent_loss_attestation_from_reports(
    *, parent_reports: Mapping[str, str | Path],
    runtime_source_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Construct the attestation from executable source/report/checkpoint bytes.

    The scientific policy is a versioned code constant.  Runtime workers do
    not read the implementation plan Markdown or any other concept document.
    """

    if not parent_reports or not runtime_source_paths:
        raise ValueError("parent-loss attestation file registry is empty")
    source_rows = [
        {"path": Path(path).as_posix(), "sha256": sha256_file(path)}
        for path in runtime_source_paths
    ]
    source_rows.sort(key=lambda row: row["path"])
    runtime_source_sha256 = canonical_sha256(source_rows)
    evidence = [
        parent_artifact_evidence_from_report(
            node_id=node_id, training_report_path=path,
            producer_source_sha256=runtime_source_sha256,
        )
        for node_id, path in sorted(parent_reports.items())
    ]
    return build_parent_loss_attestation(
        parent_artifacts=evidence,
        runtime_source_sha256=runtime_source_sha256,
    )


def build_parent_loss_attestation(
    *,
    parent_artifacts: Sequence[Mapping[str, Any]],
    runtime_source_sha256: str,
) -> dict[str, Any]:
    """Build an immutable attestation over corrected parent executions.

    Old reports cannot satisfy this function: each parent row must explicitly
    bind the versioned loss semantic, its report/checkpoint bytes, and the
    producer source.  The deterministic runtime fingerprint additionally
    prevents a matching string from attesting a different implementation.
    """

    payload = with_content_hash({
        "contract": HCWDL_PARENT_LOSS_ATTESTATION_CONTRACT,
        "schema_version": 1,
        "parent_loss_policy": dict(HCWDL_PARENT_LOSS_POLICY),
        "parent_loss_policy_sha256": HCWDL_PARENT_LOSS_POLICY_SHA256,
        "runtime_source_sha256": require_sha256(
            runtime_source_sha256, name="runtime source SHA-256",
        ),
        "loss_semantics_contract": HCWDL_PARENT_BASE_LOSS_CONTRACT,
        "loss_semantics": dict(HCWDL_PARENT_LOSS_SEMANTICS),
        "loss_semantics_sha256": canonical_sha256(HCWDL_PARENT_LOSS_SEMANTICS),
        "runtime_fingerprint": parent_loss_runtime_fingerprint(),
        "parent_artifacts": _validate_parent_rows(parent_artifacts),
    })
    validate_parent_loss_attestation(payload)
    return payload


def validate_parent_loss_attestation(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value,
        expected_contract=HCWDL_PARENT_LOSS_ATTESTATION_CONTRACT,
        expected_schema_version=1,
    )
    if (
        value.get("parent_loss_policy") != HCWDL_PARENT_LOSS_POLICY
        or value.get("parent_loss_policy_sha256")
        != HCWDL_PARENT_LOSS_POLICY_SHA256
    ):
        raise ValueError("parent-loss attestation executable policy differs")
    require_sha256(value.get("runtime_source_sha256"), name="runtime source SHA-256")
    if value.get("loss_semantics_contract") != HCWDL_PARENT_BASE_LOSS_CONTRACT:
        raise ValueError("parent-loss attestation semantic contract differs")
    if value.get("loss_semantics") != HCWDL_PARENT_LOSS_SEMANTICS:
        raise ValueError("parent-loss attestation semantic payload differs")
    if value.get("loss_semantics_sha256") != canonical_sha256(HCWDL_PARENT_LOSS_SEMANTICS):
        raise ValueError("parent-loss attestation semantic hash differs")
    if value.get("runtime_fingerprint") != parent_loss_runtime_fingerprint():
        raise ValueError("parent-loss runtime fingerprint differs")
    rows = value.get("parent_artifacts")
    if not isinstance(rows, list) or rows != _validate_parent_rows(rows):
        raise ValueError("parent-loss artifact registry differs")
    return digest


__all__ = [
    "HCWDL_PARENT_BASE_LOSS_CONTRACT",
    "HCWDL_PARENT_LOSS_ATTESTATION_CONTRACT",
    "HCWDL_PARENT_LOSS_POLICY",
    "HCWDL_PARENT_LOSS_POLICY_SHA256",
    "HCWDL_PARENT_LOSS_SEMANTICS",
    "build_parent_loss_attestation",
    "build_parent_loss_attestation_from_reports",
    "hcwdl_base_loss",
    "hcwdl_base_loss_rows",
    "parent_loss_runtime_fingerprint",
    "parent_artifact_evidence_from_report",
    "validate_parent_loss_attestation",
]

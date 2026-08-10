"""Strict, non-submitting review artifact for an executable HCWDL-RKD campaign.

A planning campaign is intentionally easy to construct and is useful for
local topology review.  It is not, by itself, evidence that the same command
plan can be submitted.  This module closes that gap without weakening the
human authorization boundary: it reopens the exact planning spec, command
plan, and runtime binding, runs every machine-verifiable execution gate, and
publishes a review artifact with ``scheduler_mutated = False``.  A later
submission authorization may bind this artifact, but this artifact can never
authorize submission on its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    load_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
)

from .hcwdl_representation_contracts import EXECUTABLE_CANDIDATE_AUDIT_CONTRACT
from .hcwdl_representation_workflow import array_indices


REFERENCE_FIELDS: Final = frozenset({"path", "sha256"})


def _reference(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _load_reference(
    reference: Mapping[str, Any], *, name: str,
) -> tuple[dict[str, Any], str]:
    if not isinstance(reference, Mapping) or set(reference) != REFERENCE_FIELDS:
        raise ValueError(f"HCWDL-RKD candidate {name} reference fields differ")
    path = Path(str(reference["path"]))
    expected = require_sha256(reference["sha256"], name=f"candidate {name} bytes")
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PermissionError(f"HCWDL-RKD candidate {name} path is unavailable or unsafe")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"HCWDL-RKD candidate {name} bytes differ")
    value = load_json(path)
    return value, actual


def _strict_gate_summary(
    *, planning_spec: Mapping[str, Any], command_plan: Mapping[str, Any],
    runtime_binding: Mapping[str, Any],
) -> dict[str, Any]:
    # Imports remain local so the campaign module can validate a candidate
    # from its executable-spec path without introducing an import cycle.
    from .hcwdl_representation_campaign import (
        DENSE_TRAINING_DISPOSITION, build_command_plan,
        validate_campaign_spec,
        validate_source_checkout,
        validate_command_plan,
        validate_tigris_acceptance,
    )
    from .hcwdl_representation_resources import (
        dense_resource_measurement_source_commit,
        validate_dense_storage_availability,
        validate_dense_storage_estimate,
        validate_dense_storage_template,
        validate_dense_measured_profile,
        validate_measured_profile,
        validate_storage_estimate,
    )
    from .hcwdl_representation_runtime_binding import validate_runtime_binding
    from .hcwdl_representation_runtime_rows import validate_bound_runtime_task_rows

    validate_campaign_spec(planning_spec, executable=False)
    if (
        planning_spec.get("planning_only") is not True
        or planning_spec.get("live_submission_authorized") is not False
        or planning_spec.get("submission_authorization") is not None
        or planning_spec.get("submission_authorization_sha256") is not None
    ):
        raise PermissionError("candidate review requires an authorization-free planning spec")
    if planning_spec.get("runtime_status") != "immutable":
        raise PermissionError("candidate review requires an immutable runtime binding")
    validate_command_plan(command_plan, spec=planning_spec)
    if command_plan.get("content_hash") != planning_spec.get("command_plan_sha256"):
        raise ValueError("candidate command plan differs from planning spec")
    runtime_hash = validate_runtime_binding(runtime_binding, spec=planning_spec)
    validate_bound_runtime_task_rows(planning_spec, runtime_binding)
    if runtime_hash != planning_spec.get("runtime_binding_sha256"):
        raise ValueError("candidate runtime binding differs from planning spec")

    source_commit = str(planning_spec.get("source_commit", ""))
    validate_source_checkout(
        planning_spec["project_dir"], expected_commit=source_commit,
    )
    profile = planning_spec.get("resource_profile")
    storage = planning_spec.get("storage_estimate")
    inventory = planning_spec.get("fixed_size_inventory")
    acceptance = planning_spec.get("tigris_acceptance")
    dense_smoke = (
        planning_spec.get("mode") == "smoke"
        and planning_spec.get("disposition") == DENSE_TRAINING_DISPOSITION
    )
    if not all(isinstance(value, Mapping) for value in (
        profile, storage, inventory,
    )) or (not dense_smoke and not isinstance(acceptance, Mapping)):
        raise PermissionError(
            "candidate review requires measured resources, storage, fixed-size "
            "inventory, and (outside the dense smoke) genuine Tigris acceptance"
        )
    if dense_smoke and acceptance is not None:
        raise PermissionError("dense smoke candidate cannot consume its future acceptance")
    profile_hash = (
        validate_dense_measured_profile(
            profile, expected_source_commit=source_commit,
        )
        if planning_spec.get("disposition") == DENSE_TRAINING_DISPOSITION else
        validate_measured_profile(
            profile, require_genuine_tigris=True,
            expected_source_commit=source_commit,
        )
    )
    if planning_spec.get("disposition") == DENSE_TRAINING_DISPOSITION:
        measurement_source_commit = dense_resource_measurement_source_commit(
            profile,
        )
        inventory_hash = validate_dense_storage_template(
            load_json(inventory["path"]),
            expected_source_commit=measurement_source_commit,
            expected_recipe_sha256=str(
                planning_spec["representation_recipe_sha256"]
            ),
            expected_graph_sha256=str(planning_spec["graph_sha256"]),
            expected_dense_teacher_import_sha256=str(
                planning_spec["parent_import_sha256"]
            ),
        )
        storage_hash = validate_dense_storage_estimate(
            storage, storage_template=inventory,
            expected_source_commit=measurement_source_commit,
            expected_recipe_sha256=str(
                planning_spec["representation_recipe_sha256"]
            ),
            expected_graph_sha256=str(planning_spec["graph_sha256"]),
            expected_dense_teacher_import_sha256=str(
                planning_spec["parent_import_sha256"]
            ),
        )
        validate_dense_storage_availability(
            storage, campaign_root=str(planning_spec["campaign_root"]),
        )
    else:
        storage_hash = validate_storage_estimate(
            storage, require_measured_fixed_sizes=True,
            fixed_size_inventory=inventory,
        )
        inventory_hash = require_sha256(
            planning_spec.get("fixed_size_inventory_sha256"),
            name="candidate fixed-size inventory",
        )
    acceptance_hash = (
        None if dense_smoke else validate_tigris_acceptance(
            acceptance,
            source_commit=source_commit,
            representation_recipe_sha256=planning_spec["representation_recipe_sha256"],
            resource_profile_sha256=profile_hash,
            storage_estimate_sha256=storage_hash,
            fixed_size_inventory_sha256=inventory_hash,
            disposition=str(planning_spec["disposition"]),
            parent_import_sha256=str(planning_spec["parent_import_sha256"]),
            graph_sha256=str(planning_spec["graph_sha256"]),
        )
    )
    expected_hashes = {
        "resource_profile_sha256": profile_hash,
        "storage_estimate_sha256": storage_hash,
        "fixed_size_inventory_sha256": inventory_hash,
        "tigris_acceptance_sha256": acceptance_hash,
    }
    if any(planning_spec.get(key) != value for key, value in expected_hashes.items()):
        raise ValueError("candidate execution evidence differs from planning spec")

    expected_plan = build_command_plan(planning_spec)
    if dict(command_plan) != expected_plan:
        raise ValueError("candidate command plan is not canonical")
    task_rows = list(planning_spec.get("tasks", ()))
    array_rows = sum(
        len(array_indices(row.get("array"))) for row in task_rows
    )
    runtime_rows = sum(len(row.get("rows", ())) for row in runtime_binding["tasks"])
    if runtime_rows != array_rows:
        raise ValueError("candidate runtime binding does not cover every task/array row")
    return {
        "campaign_identity": dict(command_plan["campaign_identity"]),
        "campaign_identity_sha256": command_plan["campaign_identity_sha256"],
        "source_commit": source_commit,
        "mode": planning_spec["mode"],
        "command_plan_sha256": command_plan["content_hash"],
        "runtime_binding_sha256": runtime_hash,
        **expected_hashes,
        "task_count": len(task_rows),
        "task_array_row_count": array_rows,
    }


def build_executable_candidate_audit(
    *, planning_spec_path: str | Path, command_plan_path: str | Path,
    runtime_binding_path: str | Path,
) -> dict[str, Any]:
    """Build the strict reviewed-but-nonauthorizing execution candidate."""

    spec_reference = _reference(planning_spec_path)
    plan_reference = _reference(command_plan_path)
    runtime_reference = _reference(runtime_binding_path)
    planning_spec, _ = _load_reference(spec_reference, name="planning spec")
    command_plan, _ = _load_reference(plan_reference, name="command plan")
    runtime_binding, _ = _load_reference(runtime_reference, name="runtime binding")
    summary = _strict_gate_summary(
        planning_spec=planning_spec,
        command_plan=command_plan,
        runtime_binding=runtime_binding,
    )
    return with_content_hash({
        "contract": EXECUTABLE_CANDIDATE_AUDIT_CONTRACT,
        "schema_version": 1,
        "planning_spec": spec_reference,
        "command_plan": plan_reference,
        "runtime_binding": runtime_reference,
        **summary,
        "all_machine_execution_gates_passed": True,
        "human_submission_authorization_present": False,
        "scheduler_mutated": False,
        "authorizes_submission": False,
    })


def validate_executable_candidate_audit(
    value: Mapping[str, Any], *, campaign_spec: Mapping[str, Any] | None = None,
) -> str:
    """Reopen all evidence and validate the candidate against an optional live spec."""

    digest = validate_content_hash(
        value, expected_contract=EXECUTABLE_CANDIDATE_AUDIT_CONTRACT,
        expected_schema_version=1,
    )
    expected_fields = {
        "contract", "schema_version", "planning_spec", "command_plan",
        "runtime_binding", "campaign_identity", "campaign_identity_sha256",
        "source_commit", "mode", "command_plan_sha256",
        "runtime_binding_sha256", "resource_profile_sha256",
        "storage_estimate_sha256", "fixed_size_inventory_sha256",
        "tigris_acceptance_sha256", "task_count", "task_array_row_count",
        "all_machine_execution_gates_passed",
        "human_submission_authorization_present", "scheduler_mutated",
        "authorizes_submission", "content_hash",
    }
    if set(value) != expected_fields:
        raise ValueError("HCWDL-RKD candidate-audit fields differ")
    planning_spec, _ = _load_reference(value["planning_spec"], name="planning spec")
    command_plan, _ = _load_reference(value["command_plan"], name="command plan")
    runtime_binding, _ = _load_reference(value["runtime_binding"], name="runtime binding")
    summary = _strict_gate_summary(
        planning_spec=planning_spec,
        command_plan=command_plan,
        runtime_binding=runtime_binding,
    )
    if any(value.get(key) != expected for key, expected in summary.items()):
        raise ValueError("HCWDL-RKD candidate-audit execution evidence differs")
    if (
        value.get("all_machine_execution_gates_passed") is not True
        or value.get("human_submission_authorization_present") is not False
        or value.get("scheduler_mutated") is not False
        or value.get("authorizes_submission") is not False
    ):
        raise PermissionError("HCWDL-RKD candidate audit crossed its authority boundary")
    if campaign_spec is not None:
        from .hcwdl_representation_campaign import build_command_plan

        live_plan = build_command_plan(campaign_spec)
        if (
            live_plan["campaign_identity"] != summary["campaign_identity"]
            or live_plan["campaign_identity_sha256"]
            != summary["campaign_identity_sha256"]
            or campaign_spec.get("command_plan_sha256")
            != summary["command_plan_sha256"]
            or campaign_spec.get("runtime_binding_sha256")
            != summary["runtime_binding_sha256"]
        ):
            raise PermissionError("candidate audit differs from executable campaign")
        for key in (
            "resource_profile_sha256", "storage_estimate_sha256",
            "fixed_size_inventory_sha256", "tigris_acceptance_sha256",
        ):
            if campaign_spec.get(key) != summary[key]:
                raise PermissionError("candidate evidence differs from executable campaign")
    return digest


__all__ = [
    "EXECUTABLE_CANDIDATE_AUDIT_CONTRACT",
    "build_executable_candidate_audit",
    "validate_executable_candidate_audit",
]

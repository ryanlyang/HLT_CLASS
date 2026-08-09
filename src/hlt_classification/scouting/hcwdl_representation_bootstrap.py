"""Bounded, nonfinal Tigris acceptance bootstrap for HCWDL-RKD.

This authority is deliberately narrower than a campaign submission
authorization.  It can execute only an initial dependency-closed prefix whose
registered tasks are acceptance prerequisites or bounded miniature probes.  It
cannot register a shared-final population, train a ladder node, read a final
role, submit a pilot, or satisfy the later executable-candidate gate by itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    load_json,
    sha256_file,
    validate_content_hash,
    with_content_hash,
)

from .hcwdl_representation_campaign import (
    CampaignTask,
    validate_campaign_spec,
)
from .hcwdl_representation_contracts import ACCEPTANCE_BOOTSTRAP_CONTRACT
from .hcwdl_representation_resources import artifact_reference, resource_table
from .hcwdl_representation_runtime_binding import validate_runtime_binding


BOOTSTRAP_AUTHORIZATION_PHRASE: Final = (
    "AUTHORIZE BOUNDED NONFINAL HCWDL-RKD ACCEPTANCE BOOTSTRAP"
)
MAX_BOOTSTRAP_ROWS_PER_ROLE: Final = 4_096
SAFE_BOOTSTRAP_TASK_PREFIX: Final = (
    "tap_schema",
    "surface_parity",
    "architecture_attestation",
    "parent_loss_attestation",
    "parent_import",
    "control_registry",
    "kernel_resources",
    "representation_recipe",
    "numerical_acceptance",
    "miniature_D100_build",
    "miniature_D100_verify_cleanup",
    "miniature_TOFF_build",
    "miniature_TOFF_verify_cleanup",
    "cache_miniature",
    "smoke_probe",
    "zero_coefficient_acceptance",
)
BOOTSTRAP_WORKER_NAMES: Final = {
    "ordinary": "run_hcwdl_representation_acceptance_bootstrap.sh",
    "deterministic": (
        "run_hcwdl_representation_acceptance_bootstrap_deterministic.sh"
    ),
}


def _reference(path: str | Path) -> dict[str, str]:
    return artifact_reference(path)


def _load_reference(
    reference: Mapping[str, Any], *, name: str,
) -> tuple[dict[str, Any], str]:
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise ValueError(f"HCWDL-RKD bootstrap {name} reference fields differ")
    path = Path(str(reference["path"]))
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PermissionError(f"HCWDL-RKD bootstrap {name} path is unsafe")
    if sha256_file(path) != reference["sha256"]:
        raise ValueError(f"HCWDL-RKD bootstrap {name} bytes differ")
    return load_json(path), str(reference["sha256"])


def _validate_file_reference(reference: object, *, name: str) -> Path:
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise ValueError(f"HCWDL-RKD bootstrap {name} reference fields differ")
    path = Path(str(reference["path"]))
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PermissionError(f"HCWDL-RKD bootstrap {name} path is unsafe")
    if sha256_file(path) != reference["sha256"]:
        raise ValueError(f"HCWDL-RKD bootstrap {name} bytes differ")
    return path


def _tasks(spec: Mapping[str, Any]) -> dict[str, CampaignTask]:
    return {
        str(row["task_key"]): CampaignTask(**{
            **row,
            "dependencies": tuple(row["dependencies"]),
            "registered_inputs": tuple(row["registered_inputs"]),
            "registered_outputs": tuple(row["registered_outputs"]),
        })
        for row in spec["tasks"]
    }


def _validate_planning_boundary(spec: Mapping[str, Any]) -> None:
    validate_campaign_spec(spec, executable=False)
    if spec.get("mode") != "smoke":
        raise PermissionError("acceptance bootstrap requires smoke campaign mode")
    if spec.get("disposition") != "validation_only_parent_claim_consumed":
        raise PermissionError("acceptance bootstrap requires validation-only disposition")
    counts = spec.get("role_counts")
    if not isinstance(counts, Mapping) or any(
        isinstance(counts.get(role), bool)
        or not isinstance(counts.get(role), int)
        or not 0 < int(counts[role]) <= MAX_BOOTSTRAP_ROWS_PER_ROLE
        for role in ("train", "validation", "final_test")
    ):
        raise PermissionError("acceptance bootstrap role population is not bounded")
    if spec.get("resources") != resource_table(mode="smoke"):
        raise PermissionError("acceptance bootstrap resources are not conservative defaults")
    if spec.get("array_concurrency_limits") != {}:
        raise PermissionError("acceptance bootstrap cannot override array concurrency")


def _validate_canonical_planning_path(
    reference: Mapping[str, Any], spec: Mapping[str, Any],
) -> None:
    root_text = str(spec["campaign_root"]).replace("\\", "/")
    # Campaign specifications keep Windows paths in the portable extended
    # spelling ``//?/C:/...``.  pathlib does not compare that spelling equal
    # to the ordinary path used by the immutable file reference, even though
    # both address the same file.  Remove only that explicit local prefix;
    # POSIX/Tigris paths are left byte-for-byte unchanged.
    if root_text.startswith("//?/"):
        root_text = root_text[4:]
    expected = Path(root_text) / "campaign_spec.json"
    if Path(str(reference["path"])).resolve() != expected.resolve():
        raise PermissionError(
            "acceptance bootstrap planning spec must occupy its canonical smoke root"
        )


def _validate_authorized_prefix(
    spec: Mapping[str, Any], authorized_tasks: object,
) -> tuple[str, ...]:
    if not isinstance(authorized_tasks, list) or not authorized_tasks:
        raise ValueError("acceptance bootstrap task prefix is empty")
    prefix = tuple(str(value) for value in authorized_tasks)
    if prefix != SAFE_BOOTSTRAP_TASK_PREFIX[: len(prefix)]:
        raise PermissionError("acceptance bootstrap tasks are not the frozen safe prefix")
    tasks = _tasks(spec)
    for position, key in enumerate(prefix):
        task = tasks.get(key)
        if task is None or task.array is not None:
            raise PermissionError("acceptance bootstrap task identity differs")
        if task.kind in {
            "reservation", "shared_final_claim", "finalist_lock",
            "final_selection", "assignment_shard", "assignment_finalize",
            "data_attestation", "prediction_shard", "prediction_finalize",
            "metric_join", "execution_lock", "final_aggregate",
            "train_node", "train_control", "confirmation",
        }:
            raise PermissionError("acceptance bootstrap contains forbidden work")
        if not set(task.dependencies) <= set(prefix[:position]):
            raise PermissionError("acceptance bootstrap prefix is not dependency closed")
        if any("final/" in route for route in (*task.registered_inputs, *task.registered_outputs)):
            raise PermissionError("acceptance bootstrap task names a final-role route")
    return prefix


def _validate_workers(value: object) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != set(BOOTSTRAP_WORKER_NAMES):
        raise ValueError("acceptance bootstrap worker registry differs")
    workers: dict[str, dict[str, str]] = {}
    for role, expected_name in BOOTSTRAP_WORKER_NAMES.items():
        reference = value[role]
        path = _validate_file_reference(reference, name=f"{role} worker")
        if path.name != expected_name:
            raise PermissionError("acceptance bootstrap worker identity differs")
        workers[role] = dict(reference)
    return workers


def build_acceptance_bootstrap(
    *, planning_spec_path: str | Path, runtime_binding_path: str | Path,
    ordinary_worker_path: str | Path, deterministic_worker_path: str | Path,
    authorized_tasks: Sequence[str], authorization_phrase: str,
) -> dict[str, Any]:
    if authorization_phrase != BOOTSTRAP_AUTHORIZATION_PHRASE:
        raise PermissionError("acceptance bootstrap authorization phrase differs")
    planning_ref = _reference(planning_spec_path)
    runtime_ref = _reference(runtime_binding_path)
    planning, _ = _load_reference(planning_ref, name="planning spec")
    runtime, _ = _load_reference(runtime_ref, name="runtime binding")
    _validate_planning_boundary(planning)
    _validate_canonical_planning_path(planning_ref, planning)
    prefix = _validate_authorized_prefix(planning, list(authorized_tasks))
    runtime_hash = validate_runtime_binding(runtime, spec=planning)
    if runtime_hash != planning.get("runtime_binding_sha256"):
        raise ValueError("acceptance bootstrap runtime binding differs")
    from .hcwdl_representation_runtime_rows import validate_bound_runtime_task_rows

    validate_bound_runtime_task_rows(planning, runtime)
    workers = _validate_workers({
        "ordinary": _reference(ordinary_worker_path),
        "deterministic": _reference(deterministic_worker_path),
    })
    return with_content_hash({
        "contract": ACCEPTANCE_BOOTSTRAP_CONTRACT,
        "schema_version": 1,
        "planning_spec": planning_ref,
        "planning_spec_sha256": planning["content_hash"],
        "runtime_binding": runtime_ref,
        "runtime_binding_sha256": runtime_hash,
        "source_commit": planning["source_commit"],
        "workers": workers,
        "resources": resource_table(mode="smoke"),
        "authorized_tasks": list(prefix),
        "maximum_rows_per_role": MAX_BOOTSTRAP_ROWS_PER_ROLE,
        "explicit_bootstrap_authorization": True,
        "bounded_acceptance_only": True,
        "final_role_access_authorized": False,
        "pilot_submission_authorized": False,
        "scheduler_mutated": False,
    })


def validate_acceptance_bootstrap(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value, expected_contract=ACCEPTANCE_BOOTSTRAP_CONTRACT,
        expected_schema_version=1,
    )
    expected_fields = {
        "contract", "schema_version", "planning_spec", "planning_spec_sha256",
        "runtime_binding", "runtime_binding_sha256", "source_commit", "workers",
        "resources", "authorized_tasks", "maximum_rows_per_role",
        "explicit_bootstrap_authorization", "bounded_acceptance_only",
        "final_role_access_authorized", "pilot_submission_authorized",
        "scheduler_mutated", "content_hash",
    }
    if set(value) != expected_fields:
        raise ValueError("acceptance bootstrap fields differ")
    planning, _ = _load_reference(value["planning_spec"], name="planning spec")
    runtime, _ = _load_reference(value["runtime_binding"], name="runtime binding")
    _validate_planning_boundary(planning)
    _validate_canonical_planning_path(value["planning_spec"], planning)
    _validate_authorized_prefix(planning, value["authorized_tasks"])
    runtime_hash = validate_runtime_binding(runtime, spec=planning)
    from .hcwdl_representation_runtime_rows import validate_bound_runtime_task_rows

    validate_bound_runtime_task_rows(planning, runtime)
    _validate_workers(value["workers"])
    expected = {
        "planning_spec_sha256": planning["content_hash"],
        "runtime_binding_sha256": runtime_hash,
        "source_commit": planning["source_commit"],
        "resources": resource_table(mode="smoke"),
        "maximum_rows_per_role": MAX_BOOTSTRAP_ROWS_PER_ROLE,
        "explicit_bootstrap_authorization": True,
        "bounded_acceptance_only": True,
        "final_role_access_authorized": False,
        "pilot_submission_authorized": False,
        "scheduler_mutated": False,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise PermissionError("acceptance bootstrap authority or lineage differs")
    if runtime_hash != planning.get("runtime_binding_sha256"):
        raise ValueError("acceptance bootstrap runtime binding differs")
    return digest


def validate_acceptance_bootstrap_task(
    value: Mapping[str, Any], *, planning_spec: Mapping[str, Any],
    task_key: str, deterministic_worker: bool,
) -> str:
    digest = validate_acceptance_bootstrap(value)
    referenced, _ = _load_reference(value["planning_spec"], name="planning spec")
    validate_campaign_spec(planning_spec, executable=False)
    if referenced["content_hash"] != planning_spec.get("content_hash"):
        raise PermissionError("acceptance bootstrap planning spec differs")
    if task_key not in value["authorized_tasks"]:
        raise PermissionError("task is not authorized by the acceptance bootstrap")
    task = _tasks(planning_spec)[task_key]
    if task.deterministic_worker is not bool(deterministic_worker):
        raise PermissionError("acceptance bootstrap worker role differs")
    return digest


def load_acceptance_bootstrap_context(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_acceptance_bootstrap(value)
    planning, _ = _load_reference(value["planning_spec"], name="planning spec")
    runtime, _ = _load_reference(value["runtime_binding"], name="runtime binding")
    return planning, runtime


__all__ = [
    "ACCEPTANCE_BOOTSTRAP_CONTRACT", "BOOTSTRAP_AUTHORIZATION_PHRASE",
    "BOOTSTRAP_WORKER_NAMES", "MAX_BOOTSTRAP_ROWS_PER_ROLE",
    "SAFE_BOOTSTRAP_TASK_PREFIX", "build_acceptance_bootstrap",
    "load_acceptance_bootstrap_context", "validate_acceptance_bootstrap",
    "validate_acceptance_bootstrap_task",
]

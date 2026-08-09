"""Bounded non-final HCWDL-RKD acceptance actions and exact proofs.

This module is deliberately separate from the representation campaign DAG.
It authorizes only ten scalar acceptance actions, never submits them, and
cannot mint a campaign-training, shared-final, final-role, reservation, or
pilot capability.  The production bridge projects only authenticated action
registry values into the reviewed adapters; CLI scientific overrides are not
accepted.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
from dataclasses import dataclass
import os
from pathlib import Path
import re
import signal
from types import MappingProxyType
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    load_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.provenance import (
    capture_source_snapshot,
    validate_source_snapshot,
    validate_source_snapshot_payload,
)

from .hcwdl_parent_loss import validate_parent_loss_attestation
from .hcwdl_recipe import validate_recipe as validate_parent_recipe
from .hcwdl_representation_bootstrap import validate_acceptance_bootstrap
from .hcwdl_representation_contracts import (
    ACCEPTANCE_BOOTSTRAP_CONTRACT,
    NONFINAL_ACCEPTANCE_ACTION_ASSEMBLY_CONTRACT,
    NONFINAL_ACCEPTANCE_ACTION_INPUTS_CONTRACT,
    NONFINAL_ACCEPTANCE_ACTION_RESULT_CONTRACT,
    NONFINAL_ACCEPTANCE_AUTHORITY_CONTRACT,
    NONFINAL_ACCEPTANCE_EXECUTION_RECEIPT_CONTRACT,
    TARGET_GENERATION_CONTRACT,
    TRAINING_REPORT_CONTRACT,
    TWO_UPDATE_ACCEPTANCE_PROOF_CONTRACT,
    USR1_DELIVERY_RECEIPT_CONTRACT,
    USR1_EXACT_RESUME_PROOF_CONTRACT,
    VALIDATION_PROXY_PROOF_CONTRACT,
)
from .hcwdl_representation_locks import validate_parent_import
from .hcwdl_representation_recipe import validate_representation_recipe
from .hcwdl_representation_resources import (
    artifact_reference,
    build_storage_estimate,
    resource_table,
    validate_nonfinal_acceptance_scheduler_evidence,
    validate_storage_estimate,
)
from .hcwdl_representation_runtime_binding import resolve_runtime_row
from .selective_assignment import build_row_selection, validate_row_selection
from .hcwdl_representation_resume import validate_resume_generation
from .hcwdl_representation_training import validate_representation_training_report
from .hcwdl_training import validate_hcwdl_parent_prefix_campaign

# Named indirection is retained for test injection and downstream callers;
# its production target is the exact v8-only validator above.
validate_parent_campaign_spec = validate_hcwdl_parent_prefix_campaign


NONFINAL_ACCEPTANCE_AUTHORIZATION_PHRASE: Final = (
    "AUTHORIZE BOUNDED NONFINAL HCWDL-RKD ACCEPTANCE ACTIONS"
)
ACCEPTANCE_TRAIN_ROWS: Final = 512
ACCEPTANCE_VALIDATION_ROWS: Final = 256
ACCEPTANCE_REPLICATE_SEED: Final = 1337
ACCEPTANCE_EFFECTIVE_BATCH_SIZE: Final = 256
ACCEPTANCE_MAXIMUM_UPDATES: Final = 2
ACCEPTANCE_FINAL_ROWS: Final = 0
USR1_EXECUTION_ID: Final = "RREL_M1c"

ACTION_IDS: Final = (
    "target_d0c",
    "target_d0w",
    "rset_m1c_two_update",
    "rset_m1w_two_update",
    "rrel_m1c_two_update",
    "rrel_m1w_two_update",
    "usr1_reference",
    "usr1_interrupt",
    "usr1_resume",
    "validation_proxy",
)
TWO_UPDATE_ACTIONS: Final = {
    "RSET_M1c": "rset_m1c_two_update",
    "RSET_M1w": "rset_m1w_two_update",
    "RREL_M1c": "rrel_m1c_two_update",
    "RREL_M1w": "rrel_m1w_two_update",
}
USR1_ACTIONS: Final = (
    "usr1_reference", "usr1_interrupt", "usr1_resume",
)
SOURCE_RUNTIME_ROW_BY_ACTION: Final = MappingProxyType({
    "target_d0c": ("target_D0c_screen", None, "target_build"),
    "target_d0w": ("target_D0w_screen", None, "target_build"),
    "rset_m1c_two_update": ("train_RSET_M1c", None, "train_node"),
    "rset_m1w_two_update": ("train_RSET_M1w", None, "train_node"),
    "rrel_m1c_two_update": ("train_RREL_M1c", None, "train_node"),
    "rrel_m1w_two_update": ("train_RREL_M1w", None, "train_node"),
    "usr1_reference": ("train_RREL_M1c", None, "train_node"),
    "usr1_interrupt": ("train_RREL_M1c", None, "train_node"),
    "usr1_resume": ("train_RREL_M1c", None, "train_node"),
    # This row authenticates the proxy's parent inputs.  The proxy separately
    # measures its deterministic gpu_final_prediction worker runtime.
    "validation_proxy": ("parent_import", None, "parent_import"),
})
WORKER_NAMES: Final = {
    "ordinary": "run_hcwdl_representation_nonfinal_acceptance.sh",
    "deterministic": (
        "run_hcwdl_representation_nonfinal_acceptance_deterministic.sh"
    ),
}
PARENT_INPUT_NAMES: Final = (
    "parent_campaign_spec",
    "parent_recipe",
    "parent_import",
    "parent_loss_attestation",
    "representation_recipe",
)
REFERENCE_FIELDS: Final = frozenset({"path", "sha256"})
_SLURM_JOB_ID = re.compile(r"^[1-9][0-9]*$")


def _action_spec(
    *, action_id: str, kind: str, dependencies: tuple[str, ...],
    worker_role: str, resource_class: str, execution_id: str | None = None,
    target_identity: str | None = None,
) -> dict[str, Any]:
    result = {
        "action_id": action_id,
        "kind": kind,
        "dependencies": list(dependencies),
        "worker_role": worker_role,
        "resource_class": resource_class,
        "scalar_only": True,
        "array": None,
        "train_rows": (
            ACCEPTANCE_TRAIN_ROWS
            if kind in {"target_prepare", "two_update", "usr1"}
            else 0
        ),
        "validation_rows": (
            ACCEPTANCE_VALIDATION_ROWS
            if kind in {"two_update", "usr1", "validation_proxy"}
            else 0
        ),
        "final_rows": ACCEPTANCE_FINAL_ROWS,
        "replicate_seed": (
            ACCEPTANCE_REPLICATE_SEED
            if kind in {"two_update", "usr1"}
            else None
        ),
        "effective_batch_size": (
            ACCEPTANCE_EFFECTIVE_BATCH_SIZE
            if kind in {"two_update", "usr1"}
            else None
        ),
        "maximum_optimizer_updates": (
            ACCEPTANCE_MAXIMUM_UPDATES
            if kind in {"two_update", "usr1"}
            else 0
        ),
        "execution_id": execution_id,
        "target_identity": target_identity,
        "mode": "smoke" if kind in {"two_update", "usr1"} else "acceptance",
        "campaign_task_kind": None,
        "final_role_access_authorized": False,
    }
    result["action_spec_sha256"] = canonical_sha256(result)
    return result


def _frozen_action_registry() -> dict[str, dict[str, Any]]:
    specs = (
        _action_spec(
            action_id="target_d0c", kind="target_prepare", dependencies=(),
            worker_role="deterministic", resource_class="gpu_target",
            target_identity="D0c",
        ),
        _action_spec(
            action_id="target_d0w", kind="target_prepare", dependencies=(),
            worker_role="deterministic", resource_class="gpu_target",
            target_identity="D0w",
        ),
        _action_spec(
            action_id="rset_m1c_two_update", kind="two_update",
            dependencies=("target_d0c",), worker_role="ordinary",
            resource_class="gpu_representation", execution_id="RSET_M1c",
            target_identity="D0c",
        ),
        _action_spec(
            action_id="rset_m1w_two_update", kind="two_update",
            dependencies=("target_d0w",), worker_role="ordinary",
            resource_class="gpu_representation", execution_id="RSET_M1w",
            target_identity="D0w",
        ),
        _action_spec(
            action_id="rrel_m1c_two_update", kind="two_update",
            dependencies=("target_d0c",), worker_role="ordinary",
            resource_class="gpu_representation", execution_id="RREL_M1c",
            target_identity="D0c",
        ),
        _action_spec(
            action_id="rrel_m1w_two_update", kind="two_update",
            dependencies=("target_d0w",), worker_role="ordinary",
            resource_class="gpu_representation", execution_id="RREL_M1w",
            target_identity="D0w",
        ),
        _action_spec(
            action_id="usr1_reference", kind="usr1",
            dependencies=("target_d0c",), worker_role="ordinary",
            resource_class="gpu_representation", execution_id=USR1_EXECUTION_ID,
            target_identity="D0c",
        ),
        _action_spec(
            action_id="usr1_interrupt", kind="usr1",
            dependencies=("target_d0c",), worker_role="ordinary",
            resource_class="gpu_representation", execution_id=USR1_EXECUTION_ID,
            target_identity="D0c",
        ),
        _action_spec(
            action_id="usr1_resume", kind="usr1",
            dependencies=("target_d0c", "usr1_interrupt"), worker_role="ordinary",
            resource_class="gpu_representation", execution_id=USR1_EXECUTION_ID,
            target_identity="D0c",
        ),
        _action_spec(
            action_id="validation_proxy", kind="validation_proxy",
            dependencies=(), worker_role="deterministic",
            resource_class="gpu_final_prediction",
        ),
    )
    return {row["action_id"]: row for row in specs}


ACTION_REGISTRY: Final = MappingProxyType(_frozen_action_registry())
ACTION_REGISTRY_SHA256: Final = canonical_sha256(dict(ACTION_REGISTRY))


def _reference(path: str | Path) -> dict[str, str]:
    return artifact_reference(path)


def _load_reference(
    reference: Mapping[str, Any], *, name: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    if not isinstance(reference, Mapping) or set(reference) != REFERENCE_FIELDS:
        raise ValueError(f"non-final acceptance {name} reference fields differ")
    path = Path(str(reference["path"]))
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PermissionError(f"non-final acceptance {name} path is unsafe")
    expected = require_sha256(reference["sha256"], name=f"{name} bytes")
    if sha256_file(path) != expected:
        raise ValueError(f"non-final acceptance {name} bytes differ")
    return load_json(path), {"path": str(path), "sha256": expected}


def _validate_file_reference(
    reference: Mapping[str, Any], *, name: str,
) -> dict[str, str]:
    if not isinstance(reference, Mapping) or set(reference) != REFERENCE_FIELDS:
        raise ValueError(f"non-final acceptance {name} reference fields differ")
    path = Path(str(reference["path"]))
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PermissionError(f"non-final acceptance {name} path is unsafe")
    expected = require_sha256(reference["sha256"], name=f"{name} bytes")
    if sha256_file(path) != expected:
        raise ValueError(f"non-final acceptance {name} bytes differ")
    return {"path": str(path), "sha256": expected}


def _reject_forbidden_action_input(value: object, *, path: str = "inputs") -> None:
    forbidden = ("final_test", "shared_final", "/final/", "\\final\\", "reservation", "pilot")
    if isinstance(value, str) and any(token in value.lower() for token in forbidden):
        raise PermissionError(f"non-final acceptance {path} names forbidden work")
    if isinstance(value, Mapping):
        for name, item in value.items():
            _reject_forbidden_action_input(name, path=f"{path}.key")
            _reject_forbidden_action_input(item, path=f"{path}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden_action_input(item, path=f"{path}[{index}]")


def _validate_generic_content_artifact(
    value: Mapping[str, Any], *, expected_contract: str,
) -> str:
    schema = value.get("schema_version")
    if isinstance(schema, bool) or not isinstance(schema, int) or schema <= 0:
        raise ValueError("non-final action result schema differs")
    return validate_content_hash(
        value, expected_contract=expected_contract, expected_schema_version=schema,
    )


def _full_source_commit(value: object) -> str:
    commit = str(value)
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("non-final acceptance source commit differs")
    return commit


def _validate_parent_inputs(
    references: Mapping[str, Any],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    if not isinstance(references, Mapping) or set(references) != set(PARENT_INPUT_NAMES):
        raise ValueError("non-final acceptance parent-input registry differs")
    loaded: dict[str, dict[str, Any]] = {}
    normalized: dict[str, dict[str, str]] = {}
    for name in PARENT_INPUT_NAMES:
        loaded[name], normalized[name] = _load_reference(references[name], name=name)

    validate_parent_campaign_spec(
        loaded["parent_campaign_spec"], executable=True,
    )
    parent_recipe_hash = validate_parent_recipe(
        loaded["parent_recipe"], require_authorized=True,
        expected_profile="primary_ladder",
    )
    parent_import_hash = validate_parent_import(loaded["parent_import"])
    parent_loss_hash = validate_parent_loss_attestation(
        loaded["parent_loss_attestation"], parent_recipe=loaded["parent_recipe"],
    )
    representation_recipe_hash = validate_representation_recipe(
        loaded["representation_recipe"],
    )
    imported_parents = loaded["parent_import"].get("parents", {})
    representation_parents = loaded["representation_recipe"].get("parents", {})
    expected_imported = {
        "parent_campaign_spec": loaded["parent_campaign_spec"].get("content_hash"),
        "parent_recipe": parent_recipe_hash,
        "parent_loss_attestation": parent_loss_hash,
    }
    if any(imported_parents.get(name) != digest for name, digest in expected_imported.items()):
        raise ValueError("non-final acceptance parent-import lineage differs")
    parent_import_payload = loaded["parent_import"].get("payload", {})
    campaign = loaded["parent_campaign_spec"]
    if any((
        parent_import_payload.get("parent_campaign_contract")
        != campaign.get("contract"),
        parent_import_payload.get("parent_campaign_mode") != campaign.get("mode"),
        parent_import_payload.get("parent_execution_scope")
        != campaign.get("execution_scope"),
        parent_import_payload.get("endpoint_continuation")
        != campaign.get("endpoint_continuation"),
        parent_import_payload.get("training_passes")
        != campaign.get("training_passes"),
        parent_import_payload.get("validation_every_passes")
        != campaign.get("validation_every_passes"),
        parent_import_payload.get("parent_train_rows")
        != campaign.get("role_counts", {}).get("train"),
        parent_import_payload.get("terminal_task_id")
        != campaign.get("terminal_task_id"),
        parent_import_payload.get("execution_lock_authorized")
        != campaign.get("execution_lock_authorized"),
        parent_import_payload.get("final_test_access_authorized")
        != campaign.get("final_test_access_authorized"),
        parent_import_payload.get("registered_final_test_tasks")
        != campaign.get("registered_final_test_tasks"),
    )):
        raise PermissionError(
            "non-final acceptance parent import differs from the exact v8 prefix"
        )
    expected_representation = {
        "parent_recipe": parent_recipe_hash,
        "parent_loss_attestation": parent_loss_hash,
        "teacher_import": parent_import_hash,
    }
    if any(
        representation_parents.get(name) != digest
        for name, digest in expected_representation.items()
    ):
        raise ValueError("non-final acceptance representation lineage differs")
    if loaded["representation_recipe"].get("content_hash") != representation_recipe_hash:
        raise ValueError("non-final acceptance representation recipe hash differs")
    return normalized, loaded


def _validate_workers(
    value: Mapping[str, Any], *, project_dir: str | Path | None = None,
) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != set(WORKER_NAMES):
        raise ValueError("non-final acceptance worker registry differs")
    normalized: dict[str, dict[str, str]] = {}
    for role, basename in WORKER_NAMES.items():
        reference = _validate_file_reference(value[role], name=f"{role} worker")
        if Path(reference["path"]).name != basename:
            raise PermissionError("non-final acceptance worker identity differs")
        if project_dir is not None and Path(reference["path"]).resolve() != (
            Path(project_dir).resolve() / "sbatch" / basename
        ):
            raise PermissionError(
                "non-final acceptance worker is not the reviewed project worker"
            )
        normalized[role] = reference
    return normalized


def _validate_registry(value: object) -> dict[str, dict[str, Any]]:
    expected = _frozen_action_registry()
    if value != expected:
        raise PermissionError("non-final acceptance action registry differs")
    resources = resource_table(mode="smoke")
    for action_id in ACTION_IDS:
        row = expected[action_id]
        if row["resource_class"] not in resources:
            raise ValueError("non-final acceptance resource class differs")
        if any(ACTION_IDS.index(parent) >= ACTION_IDS.index(action_id) for parent in row["dependencies"]):
            raise ValueError("non-final acceptance actions are not dependency ordered")
    return expected


def build_nonfinal_acceptance_action_inputs_fixture(
    *, acceptance_bootstrap_path: str | Path,
    representation_recipe_path: str | Path,
    derive_inputs: Callable[
        [str, Mapping[str, Any], Mapping[str, Any]], Mapping[str, str | Path]
    ],
) -> dict[str, Any]:
    """Build callback-driven inputs for local tests only.

    This helper is intentionally not used by the authority builder.  Genuine
    action inputs are reconstructed by :func:`build_nonfinal_acceptance_action_inputs`.
    """

    bootstrap, bootstrap_ref = _load_reference(
        _reference(acceptance_bootstrap_path), name="acceptance bootstrap",
    )
    validate_acceptance_bootstrap(bootstrap)
    runtime, runtime_ref = _load_reference(
        bootstrap["runtime_binding"], name="full smoke runtime binding",
    )
    recipe, recipe_ref = _load_reference(
        _reference(representation_recipe_path), name="representation recipe",
    )
    recipe_hash = validate_representation_recipe(recipe)
    rows: dict[str, dict[str, Any]] = {}
    for action_id, action in _frozen_action_registry().items():
        raw = derive_inputs(action_id, copy.deepcopy(action), copy.deepcopy(runtime))
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError(f"non-final action {action_id} has no derived inputs")
        if any(
            not isinstance(name, str)
            or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None
            for name in raw
        ):
            raise ValueError("non-final action input names differ")
        input_artifacts = {
            name: _reference(path) for name, path in sorted(raw.items())
        }
        _reject_forbidden_action_input(input_artifacts, path=f"actions.{action_id}")
        rows[action_id] = {
            "action_id": action_id,
            "action_spec_sha256": action["action_spec_sha256"],
            "runtime_binding_sha256": bootstrap["runtime_binding_sha256"],
            "input_artifacts": input_artifacts,
            "input_artifact_set_sha256": canonical_sha256(input_artifacts),
            "derived_from_full_smoke_runtime_binding": True,
            "caller_inline_scientific_values": False,
            "final_role_access_authorized": False,
        }
    return with_content_hash({
        "contract": NONFINAL_ACCEPTANCE_ACTION_INPUTS_CONTRACT,
        "schema_version": 1,
        "source_commit": _full_source_commit(bootstrap["source_commit"]),
        "acceptance_bootstrap": bootstrap_ref,
        "acceptance_bootstrap_sha256": bootstrap["content_hash"],
        "planning_spec": dict(bootstrap["planning_spec"]),
        "planning_spec_sha256": bootstrap["planning_spec_sha256"],
        "runtime_binding": runtime_ref,
        "runtime_binding_sha256": bootstrap["runtime_binding_sha256"],
        "representation_recipe": recipe_ref,
        "representation_recipe_sha256": recipe_hash,
        "action_registry_sha256": ACTION_REGISTRY_SHA256,
        "actions": rows,
        "bounded_nonfinal_only": True,
        "final_role_access_authorized": False,
        "campaign_training_authorized": False,
        "shared_final_authorized": False,
        "pilot_submission_authorized": False,
        "scheduler_mutated": False,
        "derivation_kind": "local_fixture_only",
    })


def _runtime_input_reference(
    row: Mapping[str, Any], logical: str, *, name: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    inputs = row.get("inputs")
    reference = inputs.get(logical) if isinstance(inputs, Mapping) else None
    if not isinstance(reference, Mapping) or "path" not in reference:
        raise ValueError(f"non-final acceptance {name} runtime input is absent")
    path = Path(str(reference["path"]))
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"non-final acceptance {name} runtime input is absent")
    observed = sha256_file(path)
    supplied = reference.get("sha256")
    if supplied is not None and observed != require_sha256(supplied, name=name):
        raise ValueError(f"non-final acceptance {name} bytes differ")
    return load_json(path), {"path": str(path), "sha256": observed}


def _publish_derived_json(path: Path, value: Mapping[str, Any]) -> dict[str, str]:
    if any(part.lower() in {"final", "final_test"} for part in path.parts):
        raise PermissionError("non-final acceptance derived path names final work")
    write_immutable_json(path, value)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _action_trajectory(action_id: str) -> str | None:
    return {
        "rset_m1c_two_update": "rset_m1c_two_update",
        "rrel_m1c_two_update": "rrel_m1c_two_update",
        "rset_m1w_two_update": "rset_m1w_two_update",
        "rrel_m1w_two_update": "rrel_m1w_two_update",
        "usr1_reference": "rrel_m1c_usr1_exact_resume",
        "usr1_interrupt": "rrel_m1c_usr1_exact_resume",
        "usr1_resume": "rrel_m1c_usr1_exact_resume",
    }.get(action_id)


def _action_workspace(root: Path, action_id: str) -> Path:
    """Return the frozen scientific workspace for one acceptance action.

    The interrupted and resumed processes intentionally share one committed
    resume-generation root.  Their result envelopes remain action-specific.
    """

    workspace_id = (
        "usr1_interrupted_trajectory"
        if action_id in {"usr1_interrupt", "usr1_resume"}
        else action_id
    )
    return root / "workspaces" / workspace_id


def _canonical_action_inputs_root(value: Mapping[str, Any]) -> Path:
    """Recover the one canonical non-final root from assembly references."""

    rows = value.get("actions")
    if not isinstance(rows, Mapping) or set(rows) != set(ACTION_IDS):
        raise ValueError("non-final action-input row registry differs")
    roots: set[Path] = set()
    assembly_paths: dict[str, Path] = {}
    for action_id in ACTION_IDS:
        row = rows[action_id]
        artifacts = row.get("input_artifacts") if isinstance(row, Mapping) else None
        assembly = artifacts.get("action_assembly") if isinstance(artifacts, Mapping) else None
        if not isinstance(assembly, Mapping):
            raise PermissionError("canonical non-final action assembly is absent")
        reference = _validate_file_reference(
            assembly, name=f"{action_id} action assembly",
        )
        path = Path(reference["path"]).resolve()
        if path.parent.name != "assemblies" or path.name != f"{action_id}.json":
            raise PermissionError("canonical non-final action assembly route differs")
        roots.add(path.parent.parent)
        assembly_paths[action_id] = path
    if len(roots) != 1:
        raise PermissionError("canonical non-final derived roots differ")
    root = next(iter(roots))
    for action_id, path in assembly_paths.items():
        assembly = load_json(path)
        validate_nonfinal_acceptance_action_assembly(assembly)
        if Path(str(assembly["workspace"])).resolve() != _action_workspace(
            root, action_id,
        ):
            raise PermissionError("canonical non-final action workspace differs")
    return root


def _require_canonical_action_inputs_route(
    value: Mapping[str, Any], reference: Mapping[str, Any],
) -> Path | None:
    """Require production action inputs at ``<root>/action_inputs.json``."""

    if value.get("derivation_kind") == "local_fixture_only":
        return None
    root = _canonical_action_inputs_root(value)
    if Path(str(reference["path"])).resolve() != root / "action_inputs.json":
        raise PermissionError("canonical non-final action-input route differs")
    return root


def _require_canonical_authority_route(
    authority: Mapping[str, Any], reference: Mapping[str, Any],
) -> Path | None:
    """Require genuine authority bytes at ``<root>/authority.json``."""

    action_inputs, action_inputs_ref = _load_reference(
        authority["action_inputs"], name="non-final action inputs",
    )
    root = _require_canonical_action_inputs_route(action_inputs, action_inputs_ref)
    if root is None:
        return None
    if Path(str(reference["path"])).resolve() != root / "authority.json":
        raise PermissionError("canonical non-final authority route differs")
    return root


def validate_nonfinal_acceptance_action_assembly(
    value: Mapping[str, Any],
) -> str:
    """Validate one authority-derived scalar action descriptor.

    This is the reusable structural boundary named by the central route
    registry.  The enclosing action-input validator additionally reconstructs
    the descriptor from the immutable full-smoke runtime binding and checks
    its exact artifact paths.
    """

    digest = validate_content_hash(
        value,
        expected_contract=NONFINAL_ACCEPTANCE_ACTION_ASSEMBLY_CONTRACT,
        expected_schema_version=1,
    )
    fields = {
        "contract", "schema_version", "action_id", "action_spec_sha256",
        "source_task_key", "source_array_index", "source_kind",
        "source_runtime_row_sha256", "source_assembly_sha256",
        "bounded_row_selection_sha256", "bounded_storage_estimate_sha256",
        "target_consumer_registry_sha256", "registered_execution_id",
        "execution_id", "target_identity", "train_rows", "validation_rows",
        "final_rows", "replicate_seed", "effective_batch_size",
        "maximum_optimizer_updates", "mode", "workspace", "dependencies",
        "campaign_task_identity_reused", "reservation_authorized",
        "pilot_submission_authorized", "final_role_access_authorized",
        "shared_final_authorized", "production_bridge_available",
        "content_hash",
    }
    if set(value) != fields:
        raise ValueError("non-final action assembly fields differ")
    action_id = str(value.get("action_id"))
    if action_id not in ACTION_REGISTRY:
        raise ValueError("non-final action assembly ID differs")
    action = ACTION_REGISTRY[action_id]
    task_key, array_index, source_kind = SOURCE_RUNTIME_ROW_BY_ACTION[action_id]
    expected = {
        "action_spec_sha256": action["action_spec_sha256"],
        "source_task_key": task_key,
        "source_array_index": array_index,
        "source_kind": source_kind,
        "execution_id": action.get("execution_id"),
        "target_identity": action.get("target_identity"),
        "train_rows": action["train_rows"],
        "validation_rows": action["validation_rows"],
        "final_rows": 0,
        "replicate_seed": action["replicate_seed"],
        "effective_batch_size": action["effective_batch_size"],
        "maximum_optimizer_updates": action["maximum_optimizer_updates"],
        "mode": action["mode"],
        "dependencies": action["dependencies"],
        "campaign_task_identity_reused": False,
        "reservation_authorized": False,
        "pilot_submission_authorized": False,
        "final_role_access_authorized": False,
        "shared_final_authorized": False,
        "production_bridge_available": True,
    }
    if any(value.get(name) != expected_value for name, expected_value in expected.items()):
        raise PermissionError("non-final action assembly semantics differ")
    for name in (
        "source_runtime_row_sha256", "source_assembly_sha256",
        "bounded_row_selection_sha256", "bounded_storage_estimate_sha256",
    ):
        require_sha256(value[name], name=f"non-final action assembly {name}")
    target_registry = value["target_consumer_registry_sha256"]
    if action.get("target_identity") is None:
        if target_registry is not None:
            raise ValueError("validation proxy unexpectedly binds a target registry")
    else:
        require_sha256(target_registry, name="non-final target consumer registry")
    trajectory = _action_trajectory(action_id)
    registered_execution_id = value["registered_execution_id"]
    if trajectory is None:
        if registered_execution_id is not None:
            raise ValueError("non-training action binds a registered execution")
    elif not isinstance(registered_execution_id, str) or not registered_execution_id:
        raise ValueError("non-final training action lacks a registered execution")
    workspace = Path(str(value["workspace"]))
    if not workspace.is_absolute() or any(
        part.lower() in {"final", "final_test", "shared_final"}
        for part in workspace.parts
    ):
        raise PermissionError("non-final action workspace is unsafe")
    return digest


def build_nonfinal_acceptance_action_inputs(
    *, acceptance_bootstrap_path: str | Path,
    representation_recipe_path: str | Path, derived_root: str | Path,
) -> dict[str, Any]:
    """Canonically project the exact smoke rows into bounded non-final inputs."""

    bootstrap, bootstrap_ref = _load_reference(
        _reference(acceptance_bootstrap_path), name="acceptance bootstrap",
    )
    validate_acceptance_bootstrap(bootstrap)
    planning, planning_ref = _load_reference(
        bootstrap["planning_spec"], name="full smoke planning spec",
    )
    runtime, runtime_ref = _load_reference(
        bootstrap["runtime_binding"], name="full smoke runtime binding",
    )
    recipe, recipe_ref = _load_reference(
        _reference(representation_recipe_path), name="representation recipe",
    )
    recipe_hash = validate_representation_recipe(recipe)
    root = Path(derived_root).resolve()
    if any(part.lower() in {"final", "final_test"} for part in root.parts):
        raise PermissionError("non-final acceptance derived root names final work")

    target_rows = {
        bank: resolve_runtime_row(
            runtime, spec=planning, task_key=f"target_{bank}_screen",
            array_index=None,
        )
        for bank in ("D0c", "D0w")
    }
    training_source = resolve_runtime_row(
        runtime, spec=planning, task_key="train_RSET_M1c", array_index=None,
    )
    split, _ = _runtime_input_reference(
        training_source, "${split_manifest}", name="split manifest",
    )
    raw_assembly = training_source.get("parameters", {}).get("assembly")
    if not isinstance(raw_assembly, Mapping):
        raise ValueError("non-final acceptance training assembly is absent")
    data_root = str(raw_assembly.get("data_root", ""))
    bounded_selection = build_row_selection(
        split, data_root=data_root,
        role_budgets={
            "train": ACCEPTANCE_TRAIN_ROWS,
            "validation": ACCEPTANCE_VALIDATION_ROWS,
        },
        seed=ACCEPTANCE_REPLICATE_SEED,
    )
    validate_row_selection(
        bounded_selection, split_manifest_sha256=split["content_hash"],
    )
    train_sources = bounded_selection["roles"]["train"]["sources"]
    if any(int(source["rows"]) <= 0 for source in train_sources):
        raise ValueError(
            "bounded acceptance selection does not cover every target source"
        )
    selection_ref = _publish_derived_json(
        root / "bounded_row_selection.json", bounded_selection,
    )

    source_storage, _ = _runtime_input_reference(
        target_rows["D0c"], "${storage_estimate}", name="storage estimate",
    )
    validate_storage_estimate(source_storage)
    bounded_storage = build_storage_estimate(
        train_rows=ACCEPTANCE_TRAIN_ROWS,
        validation_rows=ACCEPTANCE_VALIDATION_ROWS,
        final_rows=0,
        parent_import_sha256=source_storage["parent_import_sha256"],
        prediction_finalists=0,
        retained_resume_bytes=source_storage["retained_resume_bytes"],
        selected_checkpoint_bytes=source_storage["selected_checkpoint_bytes"],
        final_assignment_bytes=source_storage["final_assignment_bytes"],
        fixed_artifact_bytes=source_storage["fixed_artifact_bytes"],
        interrupted_target_reserve_bytes=source_storage[
            "interrupted_target_generation_reserve_bytes"
        ],
    )
    storage_ref = _publish_derived_json(
        root / "bounded_storage_estimate.json", bounded_storage,
    )

    from .hcwdl_representation_targets import (
        build_nonfinal_acceptance_target_consumer_registry,
    )

    registries: dict[str, tuple[dict[str, Any], dict[str, str]]] = {}
    for bank, row in target_rows.items():
        logical, _ = _runtime_input_reference(
            row, f"${{logical_bank:{bank}}}", name=f"{bank} logical bank",
        )
        registry = build_nonfinal_acceptance_target_consumer_registry(
            logical,
            acceptance_bootstrap_sha256=bootstrap["content_hash"],
            runtime_binding_sha256=bootstrap["runtime_binding_sha256"],
            source_runtime_row_sha256=canonical_sha256(dict(row)),
            action_registry_sha256=ACTION_REGISTRY_SHA256,
            recipe_sha256=recipe_hash,
        )
        registries[bank] = (
            registry,
            _publish_derived_json(
                root / f"target_consumer_registry_{bank.lower()}.json", registry,
            ),
        )

    rows: dict[str, dict[str, Any]] = {}
    for action_id, action in _frozen_action_registry().items():
        task_key, array_index, source_kind = SOURCE_RUNTIME_ROW_BY_ACTION[action_id]
        source_row = resolve_runtime_row(
            runtime, spec=planning, task_key=task_key, array_index=array_index,
        )
        if source_row.get("kind") is not None and source_row.get("kind") != source_kind:
            raise ValueError("non-final source runtime kind differs")
        source_sha = canonical_sha256(dict(source_row))
        source_assembly = source_row.get("parameters", {}).get("assembly")
        trajectory = _action_trajectory(action_id)
        bank = action.get("target_identity")
        registered_execution_id = None
        registry_ref = None
        if bank in registries:
            registry, registry_ref = registries[str(bank)]
        if trajectory is not None:
            assert registry_ref is not None
            matches = [
                consumer for consumer in registry["payload"]["consumers"]
                if consumer["execution_identity_payload"]["trajectory_id"] == trajectory
            ]
            if len(matches) != 1:
                raise ValueError("non-final target trajectory is not unique")
            registered_execution_id = matches[0]["execution_id"]
        descriptor = with_content_hash({
            "contract": NONFINAL_ACCEPTANCE_ACTION_ASSEMBLY_CONTRACT,
            "schema_version": 1,
            "action_id": action_id,
            "action_spec_sha256": action["action_spec_sha256"],
            "source_task_key": task_key,
            "source_array_index": array_index,
            "source_kind": source_kind,
            "source_runtime_row_sha256": source_sha,
            "source_assembly_sha256": canonical_sha256(source_assembly),
            "bounded_row_selection_sha256": bounded_selection["content_hash"],
            "bounded_storage_estimate_sha256": bounded_storage["content_hash"],
            "target_consumer_registry_sha256": (
                None if registry_ref is None else registry["content_hash"]
            ),
            "registered_execution_id": registered_execution_id,
            "execution_id": action.get("execution_id"),
            "target_identity": bank,
            "train_rows": action["train_rows"],
            "validation_rows": action["validation_rows"],
            "final_rows": 0,
            "replicate_seed": action["replicate_seed"],
            "effective_batch_size": action["effective_batch_size"],
            "maximum_optimizer_updates": action["maximum_optimizer_updates"],
            "mode": action["mode"],
            "workspace": str(_action_workspace(root, action_id)),
            "dependencies": list(action["dependencies"]),
            "campaign_task_identity_reused": False,
            "reservation_authorized": False,
            "pilot_submission_authorized": False,
            "final_role_access_authorized": False,
            "shared_final_authorized": False,
            "production_bridge_available": True,
        })
        descriptor_ref = _publish_derived_json(
            root / "assemblies" / f"{action_id}.json", descriptor,
        )
        artifacts = {
            "action_assembly": descriptor_ref,
            "bounded_row_selection": selection_ref,
            "bounded_storage_estimate": storage_ref,
        }
        if registry_ref is not None:
            artifacts["target_consumer_registry"] = registry_ref
        rows[action_id] = {
            "action_id": action_id,
            "action_spec_sha256": action["action_spec_sha256"],
            "runtime_binding_sha256": bootstrap["runtime_binding_sha256"],
            "input_artifacts": dict(sorted(artifacts.items())),
            "input_artifact_set_sha256": canonical_sha256(dict(sorted(artifacts.items()))),
            "derived_from_full_smoke_runtime_binding": True,
            "caller_inline_scientific_values": False,
            "final_role_access_authorized": False,
        }
    return with_content_hash({
        "contract": NONFINAL_ACCEPTANCE_ACTION_INPUTS_CONTRACT,
        "schema_version": 1,
        "source_commit": _full_source_commit(bootstrap["source_commit"]),
        "acceptance_bootstrap": bootstrap_ref,
        "acceptance_bootstrap_sha256": bootstrap["content_hash"],
        "planning_spec": planning_ref,
        "planning_spec_sha256": bootstrap["planning_spec_sha256"],
        "runtime_binding": runtime_ref,
        "runtime_binding_sha256": bootstrap["runtime_binding_sha256"],
        "representation_recipe": recipe_ref,
        "representation_recipe_sha256": recipe_hash,
        "action_registry_sha256": ACTION_REGISTRY_SHA256,
        "actions": rows,
        "bounded_nonfinal_only": True,
        "final_role_access_authorized": False,
        "campaign_training_authorized": False,
        "shared_final_authorized": False,
        "pilot_submission_authorized": False,
        "scheduler_mutated": False,
        "derivation_kind": "canonical_full_smoke_projection_v1",
    })


def validate_nonfinal_acceptance_action_inputs(
    value: Mapping[str, Any], *,
    expected_bootstrap: Mapping[str, Any] | None = None,
    expected_representation_recipe_sha256: str | None = None,
    allow_local_fixture: bool = False,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=NONFINAL_ACCEPTANCE_ACTION_INPUTS_CONTRACT,
        expected_schema_version=1,
    )
    expected_fields = {
        "contract", "schema_version", "source_commit", "acceptance_bootstrap",
        "acceptance_bootstrap_sha256", "planning_spec", "planning_spec_sha256",
        "runtime_binding", "runtime_binding_sha256", "representation_recipe",
        "representation_recipe_sha256", "action_registry_sha256", "actions",
        "bounded_nonfinal_only", "final_role_access_authorized",
        "campaign_training_authorized", "shared_final_authorized",
        "pilot_submission_authorized", "scheduler_mutated", "derivation_kind",
        "content_hash",
    }
    if set(value) != expected_fields:
        raise ValueError("non-final acceptance action-input fields differ")
    bootstrap, bootstrap_ref = _load_reference(
        value["acceptance_bootstrap"], name="acceptance bootstrap",
    )
    validate_acceptance_bootstrap(bootstrap)
    if expected_bootstrap is not None and bootstrap.get("content_hash") != expected_bootstrap.get("content_hash"):
        raise ValueError("non-final action inputs bind a different bootstrap")
    runtime, runtime_ref = _load_reference(
        value["runtime_binding"], name="full smoke runtime binding",
    )
    planning, planning_ref = _load_reference(
        value["planning_spec"], name="full smoke planning spec",
    )
    derivation_kind = str(value["derivation_kind"])
    if derivation_kind not in {
        "canonical_full_smoke_projection_v1", "local_fixture_only",
    }:
        raise ValueError("non-final action-input derivation kind differs")
    if derivation_kind == "local_fixture_only" and not allow_local_fixture:
        raise PermissionError("callback-derived action inputs are local-fixture-only")
    recipe, recipe_ref = _load_reference(
        value["representation_recipe"], name="representation recipe",
    )
    recipe_hash = validate_representation_recipe(recipe)
    if (
        expected_representation_recipe_sha256 is not None
        and recipe_hash != require_sha256(
            expected_representation_recipe_sha256,
            name="expected non-final representation recipe",
        )
    ):
        raise ValueError("non-final action inputs bind a different recipe")
    raw_rows = value["actions"]
    if not isinstance(raw_rows, Mapping) or set(raw_rows) != set(ACTION_IDS):
        raise ValueError("non-final action-input row registry differs")
    canonical_selection: dict[str, Any] | None = None
    canonical_storage: dict[str, Any] | None = None
    canonical_registries: dict[str, dict[str, Any]] = {}
    if derivation_kind == "canonical_full_smoke_projection_v1":
        training_source = resolve_runtime_row(
            runtime, spec=planning, task_key="train_RSET_M1c", array_index=None,
        )
        split, _ = _runtime_input_reference(
            training_source, "${split_manifest}", name="split manifest",
        )
        raw_assembly = training_source.get("parameters", {}).get("assembly")
        if not isinstance(raw_assembly, Mapping):
            raise PermissionError("canonical non-final training assembly is absent")
        canonical_selection = build_row_selection(
            split, data_root=str(raw_assembly.get("data_root", "")),
            role_budgets={
                "train": ACCEPTANCE_TRAIN_ROWS,
                "validation": ACCEPTANCE_VALIDATION_ROWS,
            },
            seed=ACCEPTANCE_REPLICATE_SEED,
        )
        validate_row_selection(
            canonical_selection, split_manifest_sha256=split["content_hash"],
        )
        target_d0c = resolve_runtime_row(
            runtime, spec=planning, task_key="target_D0c_screen", array_index=None,
        )
        source_storage, _ = _runtime_input_reference(
            target_d0c, "${storage_estimate}", name="storage estimate",
        )
        validate_storage_estimate(source_storage)
        canonical_storage = build_storage_estimate(
            train_rows=ACCEPTANCE_TRAIN_ROWS,
            validation_rows=ACCEPTANCE_VALIDATION_ROWS,
            final_rows=0,
            parent_import_sha256=source_storage["parent_import_sha256"],
            prediction_finalists=0,
            retained_resume_bytes=source_storage["retained_resume_bytes"],
            selected_checkpoint_bytes=source_storage["selected_checkpoint_bytes"],
            final_assignment_bytes=source_storage["final_assignment_bytes"],
            fixed_artifact_bytes=source_storage["fixed_artifact_bytes"],
            interrupted_target_reserve_bytes=source_storage[
                "interrupted_target_generation_reserve_bytes"
            ],
        )
        from .hcwdl_representation_targets import (
            build_nonfinal_acceptance_target_consumer_registry,
        )

        for bank in ("D0c", "D0w"):
            source_row = resolve_runtime_row(
                runtime, spec=planning, task_key=f"target_{bank}_screen",
                array_index=None,
            )
            logical, _ = _runtime_input_reference(
                source_row, f"${{logical_bank:{bank}}}",
                name=f"{bank} logical bank",
            )
            canonical_registries[bank] = (
                build_nonfinal_acceptance_target_consumer_registry(
                    logical,
                    acceptance_bootstrap_sha256=bootstrap["content_hash"],
                    runtime_binding_sha256=bootstrap["runtime_binding_sha256"],
                    source_runtime_row_sha256=canonical_sha256(dict(source_row)),
                    action_registry_sha256=ACTION_REGISTRY_SHA256,
                    recipe_sha256=recipe_hash,
                )
            )

    canonical_rows: dict[str, dict[str, Any]] = {}
    canonical_derived_root: Path | None = None
    for action_id, action in _frozen_action_registry().items():
        row = raw_rows[action_id]
        if not isinstance(row, Mapping) or set(row) != {
            "action_id", "action_spec_sha256", "runtime_binding_sha256",
            "input_artifacts", "input_artifact_set_sha256",
            "derived_from_full_smoke_runtime_binding",
            "caller_inline_scientific_values", "final_role_access_authorized",
        }:
            raise ValueError("non-final action-input row fields differ")
        artifacts = row["input_artifacts"]
        if not isinstance(artifacts, Mapping) or not artifacts:
            raise ValueError("non-final action input artifact registry is empty")
        if list(artifacts) != sorted(artifacts) or any(
            not isinstance(name, str)
            or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None
            for name in artifacts
        ):
            raise ValueError("non-final action input artifact names differ")
        normalized = {
            name: _validate_file_reference(reference, name=f"{action_id} input {name}")
            for name, reference in artifacts.items()
        }
        _reject_forbidden_action_input(normalized, path=f"actions.{action_id}")
        if derivation_kind == "canonical_full_smoke_projection_v1":
            expected_names = {
                "action_assembly", "bounded_row_selection",
                "bounded_storage_estimate",
            }
            if action.get("target_identity") is not None:
                expected_names.add("target_consumer_registry")
            if set(normalized) != expected_names:
                raise PermissionError("canonical non-final input artifact set differs")
            descriptor = load_json(normalized["action_assembly"]["path"])
            validate_nonfinal_acceptance_action_assembly(descriptor)
            task_key, array_index, source_kind = SOURCE_RUNTIME_ROW_BY_ACTION[action_id]
            source_row = resolve_runtime_row(
                runtime, spec=planning,
                task_key=task_key, array_index=array_index,
            )
            selection = load_json(normalized["bounded_row_selection"]["path"])
            storage = load_json(normalized["bounded_storage_estimate"]["path"])
            if (
                canonical_selection is None
                or selection != canonical_selection
                or canonical_storage is None
                or storage != canonical_storage
            ):
                raise PermissionError(
                    "canonical non-final bounded projection differs from runtime"
                )
            registry = None
            if action.get("target_identity") is not None:
                registry = load_json(normalized["target_consumer_registry"]["path"])
                expected_registry = canonical_registries[
                    str(action["target_identity"])
                ]
                if registry != expected_registry:
                    raise PermissionError(
                        "canonical non-final target registry differs from runtime"
                    )
            trajectory = _action_trajectory(action_id)
            registered_execution_id = None
            if trajectory is not None:
                assert registry is not None
                matches = [
                    consumer for consumer in registry["payload"]["consumers"]
                    if consumer["execution_identity_payload"]["trajectory_id"]
                    == trajectory
                ]
                if len(matches) != 1:
                    raise PermissionError(
                        "canonical non-final target trajectory differs"
                    )
                registered_execution_id = matches[0]["execution_id"]
            assembly_path = Path(normalized["action_assembly"]["path"])
            if (
                assembly_path.parent.name != "assemblies"
                or assembly_path.name != f"{action_id}.json"
            ):
                raise PermissionError("canonical non-final assembly route differs")
            derived_root = assembly_path.parent.parent.resolve()
            if canonical_derived_root is None:
                canonical_derived_root = derived_root
            elif canonical_derived_root != derived_root:
                raise PermissionError("canonical non-final derived roots differ")
            expected_routes = {
                "bounded_row_selection": derived_root / "bounded_row_selection.json",
                "bounded_storage_estimate": derived_root / "bounded_storage_estimate.json",
            }
            if action.get("target_identity") is not None:
                expected_routes["target_consumer_registry"] = (
                    derived_root
                    / f"target_consumer_registry_{str(action['target_identity']).lower()}.json"
                )
            if any(
                Path(normalized[name]["path"]).resolve() != route
                for name, route in expected_routes.items()
            ):
                raise PermissionError("canonical non-final artifact route differs")
            expected_descriptor = {
                "action_id": action_id,
                "action_spec_sha256": action["action_spec_sha256"],
                "source_task_key": task_key,
                "source_array_index": array_index,
                "source_kind": source_kind,
                "source_runtime_row_sha256": canonical_sha256(dict(source_row)),
                "source_assembly_sha256": canonical_sha256(
                    source_row.get("parameters", {}).get("assembly")
                ),
                "bounded_row_selection_sha256": selection["content_hash"],
                "bounded_storage_estimate_sha256": storage["content_hash"],
                "target_consumer_registry_sha256": (
                    None if registry is None else registry["content_hash"]
                ),
                "registered_execution_id": registered_execution_id,
                "execution_id": action.get("execution_id"),
                "target_identity": action.get("target_identity"),
                "train_rows": action["train_rows"],
                "validation_rows": action["validation_rows"],
                "final_rows": 0,
                "replicate_seed": action["replicate_seed"],
                "effective_batch_size": action["effective_batch_size"],
                "maximum_optimizer_updates": action["maximum_optimizer_updates"],
                "mode": action["mode"],
                "workspace": str(_action_workspace(derived_root, action_id)),
                "dependencies": list(action["dependencies"]),
                "campaign_task_identity_reused": False,
                "reservation_authorized": False,
                "pilot_submission_authorized": False,
                "final_role_access_authorized": False,
                "shared_final_authorized": False,
                "production_bridge_available": True,
            }
            if set(descriptor) != {
                "contract", "schema_version", *expected_descriptor, "content_hash",
            } or any(
                descriptor.get(name) != expected
                for name, expected in expected_descriptor.items()
            ):
                raise PermissionError("canonical non-final action assembly differs")
            validate_row_selection(
                selection, split_manifest_sha256=split["content_hash"],
            )
            if (
                selection.get("seed") != ACCEPTANCE_REPLICATE_SEED
                or set(selection.get("roles", {})) != {"train", "validation"}
                or selection["roles"]["train"].get("rows") != ACCEPTANCE_TRAIN_ROWS
                or selection["roles"]["validation"].get("rows")
                != ACCEPTANCE_VALIDATION_ROWS
                or any(
                    int(source.get("rows", 0)) <= 0
                    for source in selection["roles"]["train"].get("sources", ())
                )
            ):
                raise PermissionError("bounded non-final row selection differs")
            validate_storage_estimate(storage)
            if storage.get("row_counts") != {
                "train": ACCEPTANCE_TRAIN_ROWS,
                "validation": ACCEPTANCE_VALIDATION_ROWS,
                "final": 0,
            } or storage.get("prediction_finalists") != 0:
                raise PermissionError("bounded non-final storage estimate differs")
        canonical_rows[action_id] = {
            "action_id": action_id,
            "action_spec_sha256": action["action_spec_sha256"],
            "runtime_binding_sha256": bootstrap["runtime_binding_sha256"],
            "input_artifacts": normalized,
            "input_artifact_set_sha256": canonical_sha256(normalized),
            "derived_from_full_smoke_runtime_binding": True,
            "caller_inline_scientific_values": False,
            "final_role_access_authorized": False,
        }
        if dict(row) != canonical_rows[action_id]:
            raise PermissionError("non-final action input row differs")
    expected = {
        "source_commit": bootstrap["source_commit"],
        "acceptance_bootstrap": bootstrap_ref,
        "acceptance_bootstrap_sha256": bootstrap["content_hash"],
        "planning_spec": planning_ref,
        "planning_spec_sha256": bootstrap["planning_spec_sha256"],
        "runtime_binding": runtime_ref,
        "runtime_binding_sha256": bootstrap["runtime_binding_sha256"],
        "representation_recipe": recipe_ref,
        "representation_recipe_sha256": recipe_hash,
        "action_registry_sha256": ACTION_REGISTRY_SHA256,
        "actions": canonical_rows,
        "bounded_nonfinal_only": True,
        "final_role_access_authorized": False,
        "campaign_training_authorized": False,
        "shared_final_authorized": False,
        "pilot_submission_authorized": False,
        "scheduler_mutated": False,
        "derivation_kind": derivation_kind,
    }
    if any(value.get(name) != expected_value for name, expected_value in expected.items()):
        raise PermissionError("non-final action-input authority or lineage differs")
    return digest


def build_nonfinal_acceptance_authority(
    *, project_dir: str | Path, acceptance_bootstrap_path: str | Path,
    action_inputs_path: str | Path,
    parent_campaign_spec_path: str | Path, parent_recipe_path: str | Path,
    parent_import_path: str | Path, parent_loss_attestation_path: str | Path,
    representation_recipe_path: str | Path, ordinary_worker_path: str | Path,
    deterministic_worker_path: str | Path, authorization_phrase: str,
    local_fixture: bool = False,
) -> dict[str, Any]:
    """Build, but never execute or submit, the exact non-final action authority."""

    if authorization_phrase != NONFINAL_ACCEPTANCE_AUTHORIZATION_PHRASE:
        raise PermissionError("non-final acceptance authorization phrase differs")
    root = Path(project_dir).resolve()
    source_snapshot = capture_source_snapshot(root, require_clean=True)
    validate_source_snapshot_payload(source_snapshot)
    if source_snapshot.get("worktree_clean") is not True:
        raise PermissionError("non-final acceptance requires clean source")

    bootstrap_ref = _reference(acceptance_bootstrap_path)
    bootstrap, bootstrap_ref = _load_reference(
        bootstrap_ref, name="acceptance bootstrap",
    )
    validate_acceptance_bootstrap(bootstrap)
    if bootstrap.get("source_commit") != source_snapshot.get("git_commit"):
        raise ValueError("non-final acceptance bootstrap source differs")

    parent_refs, loaded = _validate_parent_inputs({
        "parent_campaign_spec": _reference(parent_campaign_spec_path),
        "parent_recipe": _reference(parent_recipe_path),
        "parent_import": _reference(parent_import_path),
        "parent_loss_attestation": _reference(parent_loss_attestation_path),
        "representation_recipe": _reference(representation_recipe_path),
    })
    workers = _validate_workers({
        "ordinary": _reference(ordinary_worker_path),
        "deterministic": _reference(deterministic_worker_path),
    }, project_dir=root)
    action_inputs, action_inputs_ref = _load_reference(
        _reference(action_inputs_path), name="non-final action inputs",
    )
    if action_inputs.get("derivation_kind") != "canonical_full_smoke_projection_v1" and not (
        local_fixture and action_inputs.get("derivation_kind") == "local_fixture_only"
    ):
        raise PermissionError(
            "execution authority requires canonical full-smoke action inputs"
        )
    action_inputs_hash = validate_nonfinal_acceptance_action_inputs(
        action_inputs, expected_bootstrap=bootstrap,
        expected_representation_recipe_sha256=loaded["representation_recipe"][
            "content_hash"
        ],
        allow_local_fixture=local_fixture,
    )
    _require_canonical_action_inputs_route(action_inputs, action_inputs_ref)
    if not local_fixture:
        from .hcwdl_representation_campaign import validate_source_checkout

        validate_source_checkout(
            root, expected_commit=str(source_snapshot["git_commit"]),
        )
    actions = _frozen_action_registry()
    return with_content_hash({
        "contract": NONFINAL_ACCEPTANCE_AUTHORITY_CONTRACT,
        "schema_version": 1,
        "source_snapshot": source_snapshot,
        "source_commit": source_snapshot["git_commit"],
        "acceptance_bootstrap": bootstrap_ref,
        "acceptance_bootstrap_sha256": bootstrap["content_hash"],
        "planning_spec": dict(bootstrap["planning_spec"]),
        "planning_spec_sha256": bootstrap["planning_spec_sha256"],
        "runtime_binding": dict(bootstrap["runtime_binding"]),
        "runtime_binding_sha256": bootstrap["runtime_binding_sha256"],
        "action_inputs": action_inputs_ref,
        "action_inputs_sha256": action_inputs_hash,
        "parent_inputs": parent_refs,
        "representation_recipe_sha256": loaded["representation_recipe"][
            "content_hash"
        ],
        "workers": workers,
        "actions": actions,
        "action_registry_sha256": ACTION_REGISTRY_SHA256,
        "resources": resource_table(mode="smoke"),
        "role_caps": {
            "train": ACCEPTANCE_TRAIN_ROWS,
            "validation": ACCEPTANCE_VALIDATION_ROWS,
            "final_test": ACCEPTANCE_FINAL_ROWS,
        },
        "replicate_seed": ACCEPTANCE_REPLICATE_SEED,
        "effective_batch_size": ACCEPTANCE_EFFECTIVE_BATCH_SIZE,
        "maximum_optimizer_updates": ACCEPTANCE_MAXIMUM_UPDATES,
        "bounded_action_execution_authorized": True,
        "scalar_actions_only": True,
        "arrays_authorized": False,
        "campaign_training_authorized": False,
        "reservation_authorized": False,
        "shared_final_authorized": False,
        "final_role_access_authorized": False,
        "pilot_submission_authorized": False,
        "scheduler_submission_authorized": False,
        "scheduler_mutated": False,
        "execution_authorization_phrase_verified": True,
    })


def validate_nonfinal_acceptance_authority(
    value: Mapping[str, Any], *, project_dir: str | Path | None = None,
    allow_local_fixture: bool = False,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=NONFINAL_ACCEPTANCE_AUTHORITY_CONTRACT,
        expected_schema_version=1,
    )
    expected_fields = {
        "contract", "schema_version", "source_snapshot", "source_commit",
        "acceptance_bootstrap", "acceptance_bootstrap_sha256", "planning_spec",
        "planning_spec_sha256", "runtime_binding", "runtime_binding_sha256",
        "action_inputs", "action_inputs_sha256", "parent_inputs",
        "representation_recipe_sha256", "workers", "actions",
        "action_registry_sha256", "resources", "role_caps", "replicate_seed",
        "effective_batch_size", "maximum_optimizer_updates",
        "bounded_action_execution_authorized", "scalar_actions_only",
        "arrays_authorized", "campaign_training_authorized",
        "reservation_authorized", "shared_final_authorized",
        "final_role_access_authorized", "pilot_submission_authorized",
        "scheduler_submission_authorized", "scheduler_mutated",
        "execution_authorization_phrase_verified", "content_hash",
    }
    if set(value) != expected_fields:
        raise ValueError("non-final acceptance authority fields differ")
    validate_source_snapshot_payload(value["source_snapshot"])
    if project_dir is not None:
        validate_source_snapshot(
            value["source_snapshot"], repository=project_dir, require_clean=True,
        )
    bootstrap, bootstrap_ref = _load_reference(
        value["acceptance_bootstrap"], name="acceptance bootstrap",
    )
    validate_acceptance_bootstrap(bootstrap)
    parent_refs, loaded = _validate_parent_inputs(value["parent_inputs"])
    action_inputs, action_inputs_ref = _load_reference(
        value["action_inputs"], name="non-final action inputs",
    )
    action_inputs_hash = validate_nonfinal_acceptance_action_inputs(
        action_inputs, expected_bootstrap=bootstrap,
        expected_representation_recipe_sha256=loaded["representation_recipe"][
            "content_hash"
        ],
        allow_local_fixture=allow_local_fixture,
    )
    _require_canonical_action_inputs_route(action_inputs, action_inputs_ref)
    if action_inputs.get("derivation_kind") != "canonical_full_smoke_projection_v1" and not (
        allow_local_fixture
        and action_inputs.get("derivation_kind") == "local_fixture_only"
    ):
        raise PermissionError(
            "execution authority requires canonical full-smoke action inputs"
        )
    workers = _validate_workers(value["workers"], project_dir=project_dir)
    _validate_registry(value["actions"])
    expected = {
        "source_commit": value["source_snapshot"].get("git_commit"),
        "acceptance_bootstrap": bootstrap_ref,
        "acceptance_bootstrap_sha256": bootstrap.get("content_hash"),
        "planning_spec": dict(bootstrap["planning_spec"]),
        "planning_spec_sha256": bootstrap["planning_spec_sha256"],
        "runtime_binding": dict(bootstrap["runtime_binding"]),
        "runtime_binding_sha256": bootstrap["runtime_binding_sha256"],
        "action_inputs": action_inputs_ref,
        "action_inputs_sha256": action_inputs_hash,
        "parent_inputs": parent_refs,
        "representation_recipe_sha256": loaded["representation_recipe"][
            "content_hash"
        ],
        "workers": workers,
        "action_registry_sha256": ACTION_REGISTRY_SHA256,
        "resources": resource_table(mode="smoke"),
        "role_caps": {
            "train": ACCEPTANCE_TRAIN_ROWS,
            "validation": ACCEPTANCE_VALIDATION_ROWS,
            "final_test": ACCEPTANCE_FINAL_ROWS,
        },
        "replicate_seed": ACCEPTANCE_REPLICATE_SEED,
        "effective_batch_size": ACCEPTANCE_EFFECTIVE_BATCH_SIZE,
        "maximum_optimizer_updates": ACCEPTANCE_MAXIMUM_UPDATES,
        "bounded_action_execution_authorized": True,
        "scalar_actions_only": True,
        "arrays_authorized": False,
        "campaign_training_authorized": False,
        "reservation_authorized": False,
        "shared_final_authorized": False,
        "final_role_access_authorized": False,
        "pilot_submission_authorized": False,
        "scheduler_submission_authorized": False,
        "scheduler_mutated": False,
        "execution_authorization_phrase_verified": True,
    }
    if any(value.get(name) != expected_value for name, expected_value in expected.items()):
        raise PermissionError("non-final acceptance authority or lineage differs")
    if bootstrap.get("source_commit") != value["source_commit"]:
        raise ValueError("non-final acceptance bootstrap source differs")
    return digest


_validate_nonfinal_acceptance_authority_deep = validate_nonfinal_acceptance_authority


def validate_nonfinal_acceptance_authority_static(
    value: Mapping[str, Any], *, project_dir: str | Path | None = None,
    allow_local_fixture: bool = False,
) -> str:
    """Validate immutable authority lineage without reopening scientific data.

    Workers and post-job evidence collectors use this boundary.  The offline
    authority audit retains :func:`validate_nonfinal_acceptance_authority`,
    whose canonical projection intentionally recomputes the bounded selection.
    This validator instead authenticates the already-reviewed canonical
    projection, its exact assembly files, registry, byte references and false
    authority flags.  It never opens ROOT data or labels.
    """

    digest = validate_content_hash(
        value, expected_contract=NONFINAL_ACCEPTANCE_AUTHORITY_CONTRACT,
        expected_schema_version=1,
    )
    expected_fields = {
        "contract", "schema_version", "source_snapshot", "source_commit",
        "acceptance_bootstrap", "acceptance_bootstrap_sha256", "planning_spec",
        "planning_spec_sha256", "runtime_binding", "runtime_binding_sha256",
        "action_inputs", "action_inputs_sha256", "parent_inputs",
        "representation_recipe_sha256", "workers", "actions",
        "action_registry_sha256", "resources", "role_caps", "replicate_seed",
        "effective_batch_size", "maximum_optimizer_updates",
        "bounded_action_execution_authorized", "scalar_actions_only",
        "arrays_authorized", "campaign_training_authorized",
        "reservation_authorized", "shared_final_authorized",
        "final_role_access_authorized", "pilot_submission_authorized",
        "scheduler_submission_authorized", "scheduler_mutated",
        "execution_authorization_phrase_verified", "content_hash",
    }
    if set(value) != expected_fields:
        raise ValueError("non-final acceptance static authority fields differ")
    validate_source_snapshot_payload(value["source_snapshot"])
    source_commit = _full_source_commit(value["source_commit"])
    if (
        value["source_snapshot"].get("git_commit") != source_commit
        or value["source_snapshot"].get("worktree_clean") is not True
    ):
        raise PermissionError("non-final acceptance static source differs")
    if project_dir is not None:
        from .hcwdl_representation_campaign import validate_source_checkout

        validate_source_checkout(project_dir, expected_commit=source_commit)

    action_inputs, action_inputs_ref = _load_reference(
        value["action_inputs"], name="non-final action inputs",
    )
    if action_inputs.get("derivation_kind") == "local_fixture_only":
        if not allow_local_fixture:
            raise PermissionError("static authority rejects fixture-derived action inputs")
        return _validate_nonfinal_acceptance_authority_deep(
            value, project_dir=project_dir, allow_local_fixture=True,
        )
    bootstrap, bootstrap_ref = _load_reference(
        value["acceptance_bootstrap"], name="acceptance bootstrap",
    )
    bootstrap_hash = _validate_generic_content_artifact(
        bootstrap, expected_contract=ACCEPTANCE_BOOTSTRAP_CONTRACT,
    )
    action_inputs_hash = validate_content_hash(
        action_inputs, expected_contract=NONFINAL_ACCEPTANCE_ACTION_INPUTS_CONTRACT,
        expected_schema_version=1,
    )
    if action_inputs.get("derivation_kind") != "canonical_full_smoke_projection_v1":
        raise PermissionError("static authority rejects fixture-derived action inputs")
    _require_canonical_action_inputs_route(action_inputs, action_inputs_ref)
    planning, planning_ref = _load_reference(
        action_inputs["planning_spec"], name="non-final planning spec",
    )
    planning_hash = _validate_generic_content_artifact(
        planning, expected_contract=str(planning.get("contract")),
    )
    runtime, runtime_ref = _load_reference(
        action_inputs["runtime_binding"], name="non-final runtime binding",
    )
    runtime_hash = _validate_generic_content_artifact(
        runtime, expected_contract=str(runtime.get("contract")),
    )
    recipe, recipe_ref = _load_reference(
        action_inputs["representation_recipe"], name="representation recipe",
    )
    recipe_hash = _validate_generic_content_artifact(
        recipe, expected_contract=str(recipe.get("contract")),
    )
    if not isinstance(value["parent_inputs"], Mapping) or set(
        value["parent_inputs"]
    ) != set(PARENT_INPUT_NAMES):
        raise ValueError("static non-final parent-input registry differs")
    parent_refs: dict[str, dict[str, str]] = {}
    for name in PARENT_INPUT_NAMES:
        parent, parent_ref = _load_reference(
            value["parent_inputs"][name], name=f"static parent {name}",
        )
        _validate_generic_content_artifact(
            parent, expected_contract=str(parent.get("contract")),
        )
        parent_refs[name] = parent_ref
    action_input_fields = {
        "contract", "schema_version", "source_commit", "acceptance_bootstrap",
        "acceptance_bootstrap_sha256", "planning_spec", "planning_spec_sha256",
        "runtime_binding", "runtime_binding_sha256", "representation_recipe",
        "representation_recipe_sha256", "action_registry_sha256", "actions",
        "bounded_nonfinal_only", "final_role_access_authorized",
        "campaign_training_authorized", "shared_final_authorized",
        "pilot_submission_authorized", "scheduler_mutated", "derivation_kind",
        "content_hash",
    }
    if set(action_inputs) != action_input_fields:
        raise ValueError("static non-final action-input fields differ")
    action_rows = action_inputs.get("actions")
    if not isinstance(action_rows, Mapping) or set(action_rows) != set(ACTION_IDS):
        raise ValueError("static non-final action registry differs")
    for action_id, action in _frozen_action_registry().items():
        row = action_rows[action_id]
        if not isinstance(row, Mapping) or set(row) != {
            "action_id", "action_spec_sha256", "runtime_binding_sha256",
            "input_artifacts", "input_artifact_set_sha256",
            "derived_from_full_smoke_runtime_binding",
            "caller_inline_scientific_values", "final_role_access_authorized",
        }:
            raise ValueError("static non-final action-input row fields differ")
        artifacts = row["input_artifacts"]
        if not isinstance(artifacts, Mapping) or "action_assembly" not in artifacts:
            raise ValueError("static non-final action artifacts differ")
        normalized = {
            name: _validate_file_reference(reference, name=f"{action_id} input {name}")
            for name, reference in artifacts.items()
        }
        _reject_forbidden_action_input(normalized, path=f"actions.{action_id}")
        if canonical_sha256(normalized) != row.get("input_artifact_set_sha256"):
            raise ValueError("static non-final action artifact set differs")
        assembly = load_json(normalized["action_assembly"]["path"])
        assembly_hash = validate_nonfinal_acceptance_action_assembly(assembly)
        if (
            assembly.get("content_hash") != assembly_hash
            or assembly.get("action_id") != action_id
            or assembly.get("action_spec_sha256") != action["action_spec_sha256"]
            or assembly.get("dependencies") != action["dependencies"]
            or assembly.get("train_rows") != action["train_rows"]
            or assembly.get("validation_rows") != action["validation_rows"]
            or assembly.get("final_rows") != 0
            or assembly.get("replicate_seed") != action["replicate_seed"]
            or assembly.get("maximum_optimizer_updates")
            != action["maximum_optimizer_updates"]
            or any(assembly.get(name) is not False for name in (
                "campaign_task_identity_reused", "reservation_authorized",
                "pilot_submission_authorized", "final_role_access_authorized",
                "shared_final_authorized",
            ))
            or row.get("action_id") != action_id
            or row.get("action_spec_sha256") != action["action_spec_sha256"]
            or row.get("runtime_binding_sha256") != value["runtime_binding_sha256"]
            or row.get("derived_from_full_smoke_runtime_binding") is not True
            or row.get("caller_inline_scientific_values") is not False
            or row.get("final_role_access_authorized") is not False
        ):
            raise PermissionError("static non-final action assembly differs")

    workers = _validate_workers(value["workers"], project_dir=project_dir)
    _validate_registry(value["actions"])
    false_flags = (
        "arrays_authorized", "campaign_training_authorized",
        "reservation_authorized", "shared_final_authorized",
        "final_role_access_authorized", "pilot_submission_authorized",
        "scheduler_submission_authorized", "scheduler_mutated",
    )
    if (
        value["acceptance_bootstrap"] != bootstrap_ref
        or value["acceptance_bootstrap_sha256"] != bootstrap_hash
        or value["action_inputs"] != action_inputs_ref
        or value["action_inputs_sha256"] != action_inputs_hash
        or action_inputs.get("source_commit") != source_commit
        or action_inputs.get("acceptance_bootstrap") != bootstrap_ref
        or action_inputs.get("acceptance_bootstrap_sha256") != bootstrap_hash
        or action_inputs.get("planning_spec") != planning_ref
        or action_inputs.get("planning_spec_sha256") != planning_hash
        or action_inputs.get("runtime_binding") != runtime_ref
        or action_inputs.get("runtime_binding_sha256") != runtime_hash
        or action_inputs.get("representation_recipe") != recipe_ref
        or action_inputs.get("representation_recipe_sha256")
        != recipe_hash
        or recipe_hash != value["representation_recipe_sha256"]
        or parent_refs != value["parent_inputs"]
        or parent_refs["representation_recipe"] != recipe_ref
        or bootstrap.get("source_commit") != source_commit
        or bootstrap.get("planning_spec") != planning_ref
        or bootstrap.get("planning_spec_sha256") != planning_hash
        or bootstrap.get("runtime_binding") != runtime_ref
        or bootstrap.get("runtime_binding_sha256") != runtime_hash
        or value["planning_spec"] != planning_ref
        or value["planning_spec_sha256"] != planning_hash
        or value["runtime_binding"] != runtime_ref
        or value["runtime_binding_sha256"] != runtime_hash
        or action_inputs.get("action_registry_sha256") != ACTION_REGISTRY_SHA256
        or action_inputs.get("bounded_nonfinal_only") is not True
        or any(action_inputs.get(name) is not False for name in (
            "final_role_access_authorized", "campaign_training_authorized",
            "shared_final_authorized", "pilot_submission_authorized",
            "scheduler_mutated",
        ))
        or value["workers"] != workers
        or value["action_registry_sha256"] != ACTION_REGISTRY_SHA256
        or value["resources"] != resource_table(mode="smoke")
        or value["role_caps"] != {
            "train": ACCEPTANCE_TRAIN_ROWS,
            "validation": ACCEPTANCE_VALIDATION_ROWS,
            "final_test": 0,
        }
        or value["bounded_action_execution_authorized"] is not True
        or value["scalar_actions_only"] is not True
        or value["execution_authorization_phrase_verified"] is not True
        or any(value[name] is not False for name in false_flags)
    ):
        raise PermissionError("static non-final authority or lineage differs")
    return digest


@dataclass(frozen=True)
class NonfinalActionRequest:
    authority_sha256: str
    action_id: str
    action_spec: Mapping[str, Any]
    source_commit: str
    representation_recipe_sha256: str
    worker_role: str
    worker: Mapping[str, str]
    bound_inputs: Mapping[str, Mapping[str, str]]
    dependencies: Mapping[str, Mapping[str, str]]


@dataclass(frozen=True)
class NonfinalActionResult:
    authority_sha256: str
    action_id: str
    action_spec_sha256: str
    artifact: Mapping[str, str]
    artifact_contract: str
    scheduler_job_id: str | None
    authorization_capable: bool


def _nonfinal_derived_root(
    authority: Mapping[str, Any], *, action_id: str,
) -> Path:
    action_inputs, _ = _load_reference(
        authority["action_inputs"], name="non-final action inputs",
    )
    artifacts = action_inputs["actions"][action_id]["input_artifacts"]
    assembly = artifacts.get("action_assembly")
    if not isinstance(assembly, Mapping):
        raise PermissionError("canonical non-final action assembly is absent")
    assembly_ref = _validate_file_reference(
        assembly, name=f"{action_id} action assembly",
    )
    path = Path(assembly_ref["path"]).resolve()
    if path.parent.name != "assemblies" or path.name != f"{action_id}.json":
        raise PermissionError("canonical non-final action assembly route differs")
    return path.parent.parent


def nonfinal_acceptance_execution_receipt_path(
    authority: Mapping[str, Any], *, action_id: str,
) -> Path:
    return _nonfinal_derived_root(authority, action_id=action_id) / (
        f"evidence/{action_id}/execution_receipt.json"
    )


def nonfinal_acceptance_action_result_path(
    authority: Mapping[str, Any], *, action_id: str,
) -> Path:
    return _nonfinal_derived_root(authority, action_id=action_id) / (
        f"results/{action_id}.json"
    )


def nonfinal_acceptance_scheduler_evidence_path(
    authority: Mapping[str, Any], *, action_id: str,
) -> Path:
    return _nonfinal_derived_root(authority, action_id=action_id) / (
        f"evidence/{action_id}/scheduler.json"
    )


def nonfinal_acceptance_raw_sacct_path(
    authority: Mapping[str, Any], *, action_id: str,
) -> Path:
    return _nonfinal_derived_root(authority, action_id=action_id) / (
        f"evidence/{action_id}/raw_sacct.psv"
    )


def resolve_nonfinal_acceptance_dependency_action_results(
    authority: Mapping[str, Any], *, action_id: str,
    require_genuine: bool = True,
) -> dict[str, dict[str, str]]:
    """Resolve dependencies only from their frozen action-result routes."""

    action = authority["actions"][action_id]
    result: dict[str, dict[str, str]] = {}
    for dependency in action["dependencies"]:
        path = nonfinal_acceptance_action_result_path(
            authority, action_id=dependency,
        )
        reference = _reference(path)
        artifact, _ = _load_reference(
            reference, name=f"{action_id} dependency {dependency}",
        )
        validate_nonfinal_acceptance_action_result(
            artifact, expected_action_id=dependency,
            require_genuine=require_genuine,
        )
        result[dependency] = reference
    return result


def _validate_full_loss_output(
    value: Mapping[str, Any], *, authority: Mapping[str, Any],
    action_id: str,
) -> str:
    from .hcwdl_representation_training import (
        validate_acceptance_real_batch_full_loss_record,
    )

    digest = validate_acceptance_real_batch_full_loss_record(value)
    action = authority["actions"][action_id]
    assembly = _canonical_action_assembly(authority, action_id=action_id)
    expected = {
        "authority_sha256": authority["content_hash"],
        "action_id": action_id,
        "action_spec_sha256": action["action_spec_sha256"],
        "source_commit": authority["source_commit"],
        "representation_recipe_sha256": authority[
            "representation_recipe_sha256"
        ],
        "execution_id": action["execution_id"],
        "train_rows": action["train_rows"],
        "validation_rows": action["validation_rows"],
        "replicate_seed": action["replicate_seed"],
        "maximum_optimizer_updates": action["maximum_optimizer_updates"],
        "optimizer_step_performed": False,
        "real_bounded_training_batch": True,
        "model_and_rng_restored": True,
        "finite": True,
        "scientific_authorization": False,
        "final_role_accessed": False,
    }
    if assembly is not None:
        expected["registered_execution_id"] = assembly["registered_execution_id"]
    if any(value.get(name) != expected_value for name, expected_value in expected.items()):
        raise PermissionError("acceptance full-loss action lineage differs")
    return digest


def _canonical_action_assembly(
    authority: Mapping[str, Any], *, action_id: str,
) -> dict[str, Any] | None:
    action_inputs, _ = _load_reference(
        authority["action_inputs"], name="non-final action inputs",
    )
    if action_inputs.get("derivation_kind") == "local_fixture_only":
        return None
    reference = action_inputs["actions"][action_id]["input_artifacts"][
        "action_assembly"
    ]
    assembly, _ = _load_reference(reference, name=f"{action_id} action assembly")
    validate_nonfinal_acceptance_action_assembly(assembly)
    return assembly


def _validate_target_primary_output(
    artifact: Mapping[str, Any], *, reference: Mapping[str, str],
    authority: Mapping[str, Any], action_id: str, assembly: Mapping[str, Any],
) -> dict[str, Any]:
    from .hcwdl_representation_targets import (
        NONFINAL_ACCEPTANCE_TARGET_PURPOSE,
        validate_target_generation,
    )

    generation_path = Path(reference["path"]).resolve()
    generation_id = generation_path.parent.name
    bank_root = generation_path.parent.parent.parent
    workspace = Path(str(assembly["workspace"])).resolve()
    try:
        relative = generation_path.relative_to(workspace)
    except ValueError as error:
        raise PermissionError("non-final target generation is outside its action workspace") from error
    if relative.parts != (
        "targets", str(authority["actions"][action_id]["target_identity"]),
        "generations", generation_id, "generation.json",
    ):
        raise PermissionError("non-final target generation route differs")
    manifest = validate_target_generation(bank_root, generation_id)
    generation = load_json(generation_path)
    directory = generation_path.parent
    registry = load_json(directory / "consumer_registry.json")
    logical = load_json(bank_root / "logical_bank.json")
    action_inputs, _ = _load_reference(
        authority["action_inputs"], name="non-final action inputs",
    )
    expected_registry_ref = action_inputs["actions"][action_id]["input_artifacts"][
        "target_consumer_registry"
    ]
    expected_registry, _ = _load_reference(
        expected_registry_ref, name=f"{action_id} target consumer registry",
    )
    payload = manifest["payload"]
    if (
        generation.get("content_hash") != artifact.get("content_hash")
        or registry != expected_registry
        or payload.get("purpose") != NONFINAL_ACCEPTANCE_TARGET_PURPOSE
        or payload.get("rows") != ACCEPTANCE_TRAIN_ROWS
        or payload.get("logical_bank_id")
        != logical.get("payload", {}).get("logical_bank_id")
        or generation.get("parents", {}).get("logical_bank")
        != logical.get("content_hash")
        or generation.get("parents", {}).get("consumer_registry")
        != registry.get("content_hash")
        or registry.get("content_hash")
        != assembly.get("target_consumer_registry_sha256")
    ):
        raise PermissionError("non-final target generation authority differs")
    return manifest


def _validate_training_primary_output(
    report: Mapping[str, Any], *, authority: Mapping[str, Any],
    action_id: str, assembly: Mapping[str, Any],
) -> None:
    action = authority["actions"][action_id]
    if (
        report.get("mode") != "smoke"
        or report.get("complete") is not True
        or report.get("scientific_complete") is not False
        or report.get("completed_optimizer_updates")
        != action["maximum_optimizer_updates"]
        or report.get("replicate_seed") != action["replicate_seed"]
        or report.get("campaign_sha256") != authority["content_hash"]
        or report.get("execution_id") != action["execution_id"]
        or report.get("registered_execution_id")
        != assembly.get("registered_execution_id")
        or report.get("target_cache_diagnostics", {}).get(
            "row_selection_sha256"
        ) != assembly.get("bounded_row_selection_sha256")
        or report.get("validation", {}).get("rows") != action["validation_rows"]
        or not isinstance(report.get("validation_history"), list)
        or len(report["validation_history"]) != 1
        or require_sha256(
            report.get("diagnostic_batch_sha256"), name="diagnostic batch",
        )
        != report.get("diagnostic_batch_sha256")
    ):
        raise PermissionError("non-final training report authority differs")


def _cross_validate_receipt_semantics(
    *, authority: Mapping[str, Any], action_id: str,
    loaded_outputs: Mapping[str, Mapping[str, Any]],
    dependency_rows: Mapping[str, Mapping[str, Any]],
) -> None:
    action = authority["actions"][action_id]
    assembly = _canonical_action_assembly(authority, action_id=action_id)
    if assembly is None or action["kind"] not in {"two_update", "usr1"}:
        return
    primary = loaded_outputs["primary"]
    if action_id != "usr1_interrupt":
        _validate_training_primary_output(
            primary, authority=authority, action_id=action_id, assembly=assembly,
        )
    target_dependencies = [
        dependency for dependency in action["dependencies"]
        if dependency.startswith("target_")
    ]
    if len(target_dependencies) != 1:
        raise PermissionError("non-final training target dependency differs")
    target_action = target_dependencies[0]
    dependency_result, _ = _load_reference(
        dependency_rows[target_action]["action_result"],
        name=f"{action_id} target dependency result",
    )
    target_reference = dependency_result["semantic_outputs"]["primary"]["artifact"]
    target_generation, target_normalized = _load_reference(
        target_reference, name=f"{action_id} target generation",
    )
    target_assembly = _canonical_action_assembly(
        authority, action_id=target_action,
    )
    assert target_assembly is not None
    manifest = _validate_target_primary_output(
        target_generation, reference=target_normalized, authority=authority,
        action_id=target_action, assembly=target_assembly,
    )
    expected_target = {
        "target_generation_sha256": target_generation["content_hash"],
        "target_logical_sha256": manifest["payload"]["logical_target_sha256"],
        "target_manifest_sha256": manifest["content_hash"],
    }
    if action_id != "usr1_interrupt" and any(
        primary.get(name) != expected for name, expected in expected_target.items()
    ):
        raise PermissionError("non-final training report target lineage differs")
    full_loss = loaded_outputs.get("acceptance_full_loss")
    if full_loss is not None:
        cross = {
            "execution_id": action["execution_id"],
            "registered_execution_id": assembly["registered_execution_id"],
            **expected_target,
        }
        if action_id != "usr1_interrupt":
            cross["diagnostic_batch_sha256"] = primary.get(
                "diagnostic_batch_sha256"
            )
        if any(full_loss.get(name) != expected for name, expected in cross.items()):
            raise PermissionError("full-loss record and training report differ")


def _load_semantic_outputs(
    references: Mapping[str, Any], *, authority: Mapping[str, Any], action_id: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    action = authority["actions"][action_id]
    expected_names = {"primary"}
    if action_id in set(TWO_UPDATE_ACTIONS.values()):
        expected_names.add("acceptance_full_loss")
    if not isinstance(references, Mapping) or set(references) != expected_names:
        raise PermissionError("non-final semantic output inventory differs")
    loaded: dict[str, dict[str, Any]] = {}
    normalized: dict[str, dict[str, str]] = {}
    rows: dict[str, dict[str, Any]] = {}
    assembly = _canonical_action_assembly(authority, action_id=action_id)
    for name in sorted(expected_names):
        artifact, reference = _load_reference(
            references[name], name=f"{action_id} semantic output {name}",
        )
        if name == "primary":
            digest = _validate_action_output(action, artifact)
            if assembly is not None and action["kind"] == "target_prepare":
                _validate_target_primary_output(
                    artifact, reference=reference, authority=authority,
                    action_id=action_id, assembly=assembly,
                )
        else:
            digest = _validate_full_loss_output(
                artifact, authority=authority, action_id=action_id,
            )
        if artifact.get("content_hash") != digest:
            raise ValueError("non-final semantic output content identity differs")
        loaded[name] = artifact
        normalized[name] = reference
        rows[name] = {
            "artifact": reference,
            "contract": artifact["contract"],
            "schema_version": artifact["schema_version"],
            "content_hash": digest,
        }
        if assembly is not None:
            workspace = Path(str(assembly["workspace"])).resolve()
            if workspace.parent.name != "workspaces":
                raise PermissionError("non-final action workspace route differs")
            nonfinal_root = workspace.parent.parent
            output_path = Path(reference["path"]).resolve()
            if name == "acceptance_full_loss":
                expected_path = workspace / "acceptance_real_batch_full_loss.json"
            elif action["kind"] in {"two_update", "usr1"}:
                expected_path = (
                    nonfinal_root / "usr1" / "interrupt" / "receipt.json"
                    if action_id == "usr1_interrupt"
                    else workspace / "training_report.json"
                )
            elif action["kind"] == "validation_proxy":
                expected_path = nonfinal_root / "validation_proxy" / "result.json"
            else:
                expected_path = None
            if expected_path is not None and output_path != expected_path:
                raise PermissionError("non-final semantic output route differs")
    return loaded, normalized, rows


def build_nonfinal_acceptance_execution_receipt(
    *, authority: Mapping[str, Any], action_id: str,
    semantic_outputs: Mapping[str, Mapping[str, Any]],
    dependency_action_results: Mapping[str, Mapping[str, Any]] | None = None,
    scheduler_job_id: str | None, project_dir: str | Path | None = None,
    local_fixture: bool = False,
) -> dict[str, Any]:
    """Build the immutable worker-side execution/semantic binding."""

    authority_value, authority_ref = _load_reference(
        authority, name="non-final authority",
    )
    _require_canonical_authority_route(authority_value, authority_ref)
    authority_hash = validate_nonfinal_acceptance_authority_static(
        authority_value, project_dir=project_dir,
        allow_local_fixture=local_fixture,
    )
    if action_id not in ACTION_REGISTRY:
        raise PermissionError("non-final execution receipt action is not registered")
    action = authority_value["actions"][action_id]
    action_inputs, action_inputs_ref = _load_reference(
        authority_value["action_inputs"], name="non-final action inputs",
    )
    action_input_row = action_inputs["actions"][action_id]
    action_assembly = action_input_row["input_artifacts"].get("action_assembly")
    if action_inputs.get("derivation_kind") == "canonical_full_smoke_projection_v1":
        if not isinstance(action_assembly, Mapping):
            raise PermissionError("non-final execution receipt lacks its assembly")
        assembly, assembly_ref = _load_reference(
            action_assembly, name=f"{action_id} action assembly",
        )
        assembly_hash = validate_nonfinal_acceptance_action_assembly(assembly)
        source_runtime_row_sha256 = require_sha256(
            assembly["source_runtime_row_sha256"], name="source runtime row",
        )
        source_assembly_sha256 = require_sha256(
            assembly["source_assembly_sha256"], name="source assembly",
        )
    elif local_fixture:
        assembly_ref = None
        assembly_hash = None
        source_runtime_row_sha256 = None
        source_assembly_sha256 = None
    else:
        raise PermissionError("genuine execution receipt requires canonical inputs")

    dependencies = (
        resolve_nonfinal_acceptance_dependency_action_results(
            authority_value, action_id=action_id, require_genuine=not local_fixture,
        )
        if dependency_action_results is None
        else {name: dict(reference) for name, reference in dependency_action_results.items()}
    )
    if set(dependencies) != set(action["dependencies"]):
        raise PermissionError("non-final execution receipt dependencies differ")
    dependency_rows: dict[str, dict[str, Any]] = {}
    for dependency in action["dependencies"]:
        result, result_ref = _load_reference(
            dependencies[dependency], name=f"{action_id} dependency {dependency}",
        )
        validate_nonfinal_acceptance_action_result(
            result, expected_action_id=dependency,
            require_genuine=not local_fixture,
            allow_local_fixture=local_fixture,
        )
        if result.get("authority_sha256") != authority_hash:
            raise PermissionError("non-final execution dependency authority differs")
        dependency_rows[dependency] = {
            "action_result": result_ref,
            "action_result_sha256": result["content_hash"],
            "scheduler_job_id": result["scheduler_job_id"],
            "semantic_output_set_sha256": result["semantic_output_set_sha256"],
        }

    loaded_outputs, _, semantic_rows = _load_semantic_outputs(
        semantic_outputs, authority=authority_value, action_id=action_id,
    )
    _cross_validate_receipt_semantics(
        authority=authority_value, action_id=action_id,
        loaded_outputs=loaded_outputs, dependency_rows=dependency_rows,
    )
    if local_fixture:
        if scheduler_job_id is not None:
            raise PermissionError("local execution receipt cannot claim a Slurm job")
        genuine_execution = False
    else:
        if (
            project_dir is None
            or not isinstance(scheduler_job_id, str)
            or _SLURM_JOB_ID.fullmatch(scheduler_job_id) is None
            or os.environ.get("SLURM_JOB_ID") != scheduler_job_id
            or os.name != "posix"
        ):
            raise PermissionError("execution receipt lacks the live exact Slurm worker")
        genuine_execution = True
    worker_role = action["worker_role"]
    worker = dict(authority_value["workers"][worker_role])
    return with_content_hash({
        "contract": NONFINAL_ACCEPTANCE_EXECUTION_RECEIPT_CONTRACT,
        "schema_version": 1,
        "authority": authority_ref,
        "authority_sha256": authority_hash,
        "action_inputs": action_inputs_ref,
        "action_inputs_sha256": authority_value["action_inputs_sha256"],
        "action_assembly": assembly_ref,
        "action_assembly_sha256": assembly_hash,
        "action_id": action_id,
        "action_spec_sha256": action["action_spec_sha256"],
        "source_commit": authority_value["source_commit"],
        "source_runtime_row_sha256": source_runtime_row_sha256,
        "source_assembly_sha256": source_assembly_sha256,
        "representation_recipe_sha256": authority_value[
            "representation_recipe_sha256"
        ],
        "worker_role": worker_role,
        "worker": worker,
        "worker_sha256": worker["sha256"],
        "scheduler_job_id": scheduler_job_id,
        "dependency_action_results": dependency_rows,
        "dependency_action_result_set_sha256": canonical_sha256(dependency_rows),
        "semantic_outputs": semantic_rows,
        "semantic_output_set_sha256": canonical_sha256(semantic_rows),
        "genuine_worker_execution": genuine_execution,
        "worker_execution_observed": True,
        "authorization_capable": False,
        "final_role_accessed": False,
        "scientific_training_authorized": False,
        "campaign_training_authorized": False,
        "reservation_authorized": False,
        "shared_final_authorized": False,
        "pilot_submission_authorized": False,
    })


def validate_nonfinal_acceptance_execution_receipt(
    value: Mapping[str, Any], *, expected_action_id: str | None = None,
    require_genuine: bool = False, allow_local_fixture: bool = False,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=NONFINAL_ACCEPTANCE_EXECUTION_RECEIPT_CONTRACT,
        expected_schema_version=1,
    )
    expected_fields = {
        "contract", "schema_version", "authority", "authority_sha256",
        "action_inputs", "action_inputs_sha256", "action_assembly",
        "action_assembly_sha256", "action_id", "action_spec_sha256",
        "source_commit", "source_runtime_row_sha256", "source_assembly_sha256",
        "representation_recipe_sha256", "worker_role", "worker",
        "worker_sha256", "scheduler_job_id", "dependency_action_results",
        "dependency_action_result_set_sha256", "semantic_outputs",
        "semantic_output_set_sha256", "genuine_worker_execution",
        "worker_execution_observed", "authorization_capable",
        "final_role_accessed", "scientific_training_authorized",
        "campaign_training_authorized", "reservation_authorized",
        "shared_final_authorized", "pilot_submission_authorized", "content_hash",
    }
    if set(value) != expected_fields:
        raise ValueError("non-final execution receipt fields differ")
    authority, authority_ref = _load_reference(
        value["authority"], name="non-final authority",
    )
    authority_hash = validate_nonfinal_acceptance_authority_static(
        authority, allow_local_fixture=allow_local_fixture,
    )
    action_id = str(value["action_id"])
    if action_id not in ACTION_REGISTRY or (
        expected_action_id is not None and action_id != expected_action_id
    ):
        raise PermissionError("non-final execution receipt action differs")
    action = authority["actions"][action_id]
    action_inputs, action_inputs_ref = _load_reference(
        value["action_inputs"], name="non-final action inputs",
    )
    is_fixture = action_inputs.get("derivation_kind") == "local_fixture_only"
    if is_fixture and not allow_local_fixture:
        raise PermissionError("non-final execution receipt is fixture-derived")
    if is_fixture:
        assembly_ref = None
        assembly_hash = None
        source_runtime_row_sha256 = None
        source_assembly_sha256 = None
    else:
        assembly, assembly_ref = _load_reference(
            value["action_assembly"], name=f"{action_id} action assembly",
        )
        assembly_hash = validate_nonfinal_acceptance_action_assembly(assembly)
        if value["action_assembly"] != action_inputs["actions"][action_id][
            "input_artifacts"
        ]["action_assembly"]:
            raise PermissionError("non-final execution receipt assembly differs")
        source_runtime_row_sha256 = assembly["source_runtime_row_sha256"]
        source_assembly_sha256 = assembly["source_assembly_sha256"]
    semantic_refs = {
        name: row["artifact"] for name, row in value["semantic_outputs"].items()
    }
    loaded_outputs, _, semantic_rows = _load_semantic_outputs(
        semantic_refs, authority=authority, action_id=action_id,
    )
    dependencies = value["dependency_action_results"]
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(
        action["dependencies"]
    ):
        raise PermissionError("non-final execution receipt dependency registry differs")
    canonical_dependencies: dict[str, dict[str, Any]] = {}
    for dependency in action["dependencies"]:
        row = dependencies[dependency]
        result, result_ref = _load_reference(
            row["action_result"], name=f"{action_id} dependency {dependency}",
        )
        validate_nonfinal_acceptance_action_result(
            result, expected_action_id=dependency,
            require_genuine=require_genuine,
            allow_local_fixture=allow_local_fixture,
        )
        if result.get("authority_sha256") != authority_hash:
            raise PermissionError(
                "non-final execution dependency binds another authority"
            )
        canonical_dependencies[dependency] = {
            "action_result": result_ref,
            "action_result_sha256": result["content_hash"],
            "scheduler_job_id": result["scheduler_job_id"],
            "semantic_output_set_sha256": result["semantic_output_set_sha256"],
        }
    _cross_validate_receipt_semantics(
        authority=authority, action_id=action_id,
        loaded_outputs=loaded_outputs,
        dependency_rows=canonical_dependencies,
    )
    genuine = value["genuine_worker_execution"]
    job_id = value["scheduler_job_id"]
    if genuine is True:
        if not isinstance(job_id, str) or _SLURM_JOB_ID.fullmatch(job_id) is None:
            raise PermissionError("genuine execution receipt job differs")
    elif genuine is False:
        if job_id is not None or not allow_local_fixture:
            raise PermissionError("local execution receipt authority differs")
    else:
        raise ValueError("execution receipt genuineness differs")
    if require_genuine and genuine is not True:
        raise PermissionError("non-final execution receipt is not genuine")
    expected = {
        "authority": authority_ref,
        "authority_sha256": authority_hash,
        "action_inputs": action_inputs_ref,
        "action_inputs_sha256": authority["action_inputs_sha256"],
        "action_assembly": assembly_ref,
        "action_assembly_sha256": assembly_hash,
        "action_spec_sha256": action["action_spec_sha256"],
        "source_commit": authority["source_commit"],
        "source_runtime_row_sha256": source_runtime_row_sha256,
        "source_assembly_sha256": source_assembly_sha256,
        "representation_recipe_sha256": authority[
            "representation_recipe_sha256"
        ],
        "worker_role": action["worker_role"],
        "worker": authority["workers"][action["worker_role"]],
        "worker_sha256": authority["workers"][action["worker_role"]]["sha256"],
        "dependency_action_results": canonical_dependencies,
        "dependency_action_result_set_sha256": canonical_sha256(
            canonical_dependencies
        ),
        "semantic_outputs": semantic_rows,
        "semantic_output_set_sha256": canonical_sha256(semantic_rows),
        "worker_execution_observed": True,
        "authorization_capable": False,
        "final_role_accessed": False,
        "scientific_training_authorized": False,
        "campaign_training_authorized": False,
        "reservation_authorized": False,
        "shared_final_authorized": False,
        "pilot_submission_authorized": False,
    }
    if any(value.get(name) != expected_value for name, expected_value in expected.items()):
        raise PermissionError("non-final execution receipt lineage differs")
    return digest


def _validate_action_output(action: Mapping[str, Any], artifact: Mapping[str, Any]) -> str:
    kind = action["kind"]
    if kind == "target_prepare":
        return _validate_generic_content_artifact(
            artifact, expected_contract=TARGET_GENERATION_CONTRACT,
        )
    if kind == "two_update" or action["action_id"] in {
        "usr1_reference", "usr1_resume",
    }:
        return validate_representation_training_report(
            artifact,
            expected_execution_id=action["execution_id"],
        )
    if action["action_id"] == "usr1_interrupt":
        return validate_usr1_delivery_receipt(artifact)
    if kind == "validation_proxy":
        return _validate_generic_content_artifact(
            artifact, expected_contract=VALIDATION_PROXY_PROOF_CONTRACT,
        )
    raise ValueError("non-final action output kind differs")


def execute_nonfinal_action(
    *, authority: Mapping[str, Any], action_id: str,
    dependency_artifacts: Mapping[str, Mapping[str, Any]],
    executor: Callable[[NonfinalActionRequest], str | Path],
    scheduler_job_id: str | None = None, local_fixture: bool = False,
) -> NonfinalActionResult:
    """Execute one injected scalar adapter with no scientific override surface."""

    authority_hash = validate_nonfinal_acceptance_authority(authority)
    if action_id not in ACTION_REGISTRY:
        raise PermissionError("non-final acceptance action is not registered")
    action = authority["actions"][action_id]
    dependencies = action["dependencies"]
    if not isinstance(dependency_artifacts, Mapping) or set(dependency_artifacts) != set(dependencies):
        raise PermissionError("non-final acceptance dependencies differ")
    dependency_refs: dict[str, dict[str, str]] = {}
    for name in dependencies:
        _, dependency_refs[name] = _load_reference(
            dependency_artifacts[name], name=f"{action_id} dependency {name}",
        )
    if not local_fixture:
        raise PermissionError(
            "dependency-injected non-final execution is local-fixture-only; "
            "genuine results require an authority-bound action-result envelope"
        )
    if scheduler_job_id is not None:
        raise ValueError("local action fixture cannot claim a Slurm job")
    action_inputs, _ = _load_reference(
        authority["action_inputs"], name="non-final action inputs",
    )
    validate_nonfinal_acceptance_action_inputs(
        action_inputs, expected_bootstrap=_load_reference(
            authority["acceptance_bootstrap"], name="acceptance bootstrap",
        )[0],
        expected_representation_recipe_sha256=authority[
            "representation_recipe_sha256"
        ],
        allow_local_fixture=local_fixture,
    )
    request = NonfinalActionRequest(
        authority_sha256=authority_hash,
        action_id=action_id,
        action_spec=copy.deepcopy(action),
        source_commit=authority["source_commit"],
        representation_recipe_sha256=authority["representation_recipe_sha256"],
        worker_role=action["worker_role"],
        worker=copy.deepcopy(authority["workers"][action["worker_role"]]),
        bound_inputs=copy.deepcopy(
            action_inputs["actions"][action_id]["input_artifacts"]
        ),
        dependencies=dependency_refs,
    )
    output_path = Path(executor(request)).resolve()
    output, output_ref = _load_reference(_reference(output_path), name=f"{action_id} result")
    _validate_action_output(action, output)
    return NonfinalActionResult(
        authority_sha256=authority_hash,
        action_id=action_id,
        action_spec_sha256=action["action_spec_sha256"],
        artifact=output_ref,
        artifact_contract=str(output["contract"]),
        scheduler_job_id=scheduler_job_id,
        authorization_capable=False,
    )


def _load_scheduler_evidence(
    reference: Mapping[str, Any], *, authority: Mapping[str, Any], action_id: str,
    require_genuine: bool,
) -> tuple[dict[str, Any], dict[str, str]]:
    evidence, normalized = _load_reference(reference, name=f"{action_id} scheduler evidence")
    action = authority["actions"][action_id]
    validate_nonfinal_acceptance_scheduler_evidence(
        evidence,
        expected_authority_sha256=authority["content_hash"],
        expected_action_id=action_id,
        request=authority["resources"][action["resource_class"]],
        expected_source_commit=authority["source_commit"],
        expected_recipe_sha256=authority["representation_recipe_sha256"],
        expected_worker=authority["workers"][action["worker_role"]],
        expected_resource_class=action["resource_class"],
        expected_worker_role=action["worker_role"],
        require_genuine=require_genuine,
    )
    return evidence, normalized


def build_nonfinal_acceptance_action_result(
    *, authority: Mapping[str, Any], action_id: str,
    scheduler_evidence: Mapping[str, Any], execution_receipt: Mapping[str, Any],
    require_genuine: bool = False, allow_local_fixture: bool = False,
) -> dict[str, Any]:
    """Post-job bind one worker receipt to exact raw scheduler evidence."""

    authority_value, authority_ref = _load_reference(authority, name="non-final authority")
    _require_canonical_authority_route(authority_value, authority_ref)
    authority_hash = validate_nonfinal_acceptance_authority_static(
        authority_value, allow_local_fixture=allow_local_fixture,
    )
    if action_id not in ACTION_REGISTRY:
        raise PermissionError("non-final action result identity is not registered")
    action = authority_value["actions"][action_id]
    scheduler, scheduler_ref = _load_scheduler_evidence(
        scheduler_evidence, authority=authority_value, action_id=action_id,
        require_genuine=require_genuine,
    )
    receipt, receipt_ref = _load_reference(
        execution_receipt, name=f"{action_id} execution receipt",
    )
    receipt_hash = validate_nonfinal_acceptance_execution_receipt(
        receipt, expected_action_id=action_id, require_genuine=require_genuine,
        allow_local_fixture=allow_local_fixture,
    )
    if receipt.get("genuine_worker_execution") is True and Path(
        receipt_ref["path"]
    ).resolve() != nonfinal_acceptance_execution_receipt_path(
        authority_value, action_id=action_id,
    ):
        raise PermissionError("non-final execution receipt route differs")
    semantic_rows = copy.deepcopy(receipt["semantic_outputs"])
    semantic_ref = dict(semantic_rows["primary"]["artifact"])
    semantic, _ = _load_reference(
        semantic_ref, name=f"{action_id} primary semantic result",
    )
    semantic_hash = semantic_rows["primary"]["content_hash"]
    if action_id == "validation_proxy":
        from .hcwdl_representation_validation_proxy import (
            validate_validation_proxy_proof_v2,
        )

        validate_validation_proxy_proof_v2(
            semantic, authority=authority_value,
            authority_validator=validate_nonfinal_acceptance_authority_static,
        )
    authorization_capable = (
        scheduler.get("authorization_capable") is True
        and receipt.get("genuine_worker_execution") is True
    )
    if authorization_capable:
        expected_scheduler = nonfinal_acceptance_scheduler_evidence_path(
            authority_value, action_id=action_id,
        )
        if Path(scheduler_ref["path"]).resolve() != expected_scheduler:
            raise PermissionError("non-final scheduler evidence route differs")
    if (
        receipt.get("authority_sha256") != authority_hash
        or receipt.get("action_inputs") != authority_value["action_inputs"]
        or receipt.get("action_inputs_sha256")
        != authority_value["action_inputs_sha256"]
        or receipt.get("action_spec_sha256") != action["action_spec_sha256"]
        or receipt.get("source_commit") != authority_value["source_commit"]
        or receipt.get("representation_recipe_sha256")
        != authority_value["representation_recipe_sha256"]
        or receipt.get("worker_role") != action["worker_role"]
        or receipt.get("worker") != authority_value["workers"][action["worker_role"]]
        or (
            receipt.get("genuine_worker_execution") is True
            and int(receipt["scheduler_job_id"]) != int(scheduler["job_id"])
        )
    ):
        raise PermissionError("non-final action receipt/scheduler lineage differs")
    if action_id == "usr1_interrupt" and (
        semantic.get("authority_sha256") != authority_hash
        or semantic.get("action_id") != action_id
        or semantic.get("source_commit") != authority_value["source_commit"]
        or semantic.get("representation_recipe_sha256")
        != authority_value["representation_recipe_sha256"]
        or semantic.get("authorization_capable") is not authorization_capable
        or (
            authorization_capable
            and int(semantic.get("scheduler_job_id")) != int(scheduler["job_id"])
        )
    ):
        raise ValueError("USR1 interrupt result lineage differs")
    if require_genuine and not authorization_capable:
        raise PermissionError("non-final action result requires genuine Tigris evidence")
    worker_role = action["worker_role"]
    worker = authority_value["workers"][worker_role]
    return with_content_hash({
        "contract": NONFINAL_ACCEPTANCE_ACTION_RESULT_CONTRACT,
        "schema_version": 1,
        "authority": authority_ref,
        "authority_sha256": authority_hash,
        "action_inputs": dict(authority_value["action_inputs"]),
        "action_inputs_sha256": authority_value["action_inputs_sha256"],
        "action_id": action_id,
        "action_spec_sha256": action["action_spec_sha256"],
        "source_commit": authority_value["source_commit"],
        "representation_recipe_sha256": authority_value[
            "representation_recipe_sha256"
        ],
        "worker_role": worker_role,
        "worker": dict(worker),
        "worker_sha256": worker["sha256"],
        "execution_receipt": receipt_ref,
        "execution_receipt_sha256": receipt_hash,
        "scheduler_evidence": scheduler_ref,
        "scheduler_evidence_sha256": scheduler["content_hash"],
        "scheduler_job_id": scheduler["job_id"],
        "semantic_result": semantic_ref,
        "semantic_result_contract": semantic["contract"],
        "semantic_result_schema_version": semantic["schema_version"],
        "semantic_result_sha256": semantic_hash,
        "semantic_outputs": semantic_rows,
        "semantic_output_set_sha256": receipt["semantic_output_set_sha256"],
        "dependency_action_results": copy.deepcopy(
            receipt["dependency_action_results"]
        ),
        "dependency_action_result_set_sha256": receipt[
            "dependency_action_result_set_sha256"
        ],
        "bounded_acceptance_action_completed": True,
        "authorization_capable": authorization_capable,
        "final_role_accessed": False,
        "scientific_training_authorized": False,
        "campaign_training_authorized": False,
        "reservation_authorized": False,
        "shared_final_authorized": False,
        "pilot_submission_authorized": False,
    })


def validate_nonfinal_acceptance_action_result(
    value: Mapping[str, Any], *, expected_action_id: str | None = None,
    require_genuine: bool = False, allow_local_fixture: bool = False,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=NONFINAL_ACCEPTANCE_ACTION_RESULT_CONTRACT,
        expected_schema_version=1,
    )
    action_id = str(value.get("action_id", ""))
    if expected_action_id is not None and action_id != expected_action_id:
        raise ValueError("non-final action result identity differs")
    rebuilt = build_nonfinal_acceptance_action_result(
        authority=value["authority"], action_id=action_id,
        scheduler_evidence=value["scheduler_evidence"],
        execution_receipt=value["execution_receipt"],
        require_genuine=require_genuine,
        allow_local_fixture=allow_local_fixture,
    )
    if dict(value) != rebuilt:
        raise ValueError("non-final action result is not canonically derived")
    return digest


def _load_action_result(
    reference: Mapping[str, Any], *, action_id: str, require_genuine: bool,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any], dict[str, Any]]:
    result, normalized = _load_reference(
        reference, name=f"{action_id} action result",
    )
    validate_nonfinal_acceptance_action_result(
        result, expected_action_id=action_id, require_genuine=require_genuine,
        allow_local_fixture=not require_genuine,
    )
    if result.get("authorization_capable") is True and Path(
        normalized["path"]
    ).resolve() != nonfinal_acceptance_action_result_path(
        _load_reference(result["authority"], name="non-final authority")[0],
        action_id=action_id,
    ):
        raise PermissionError("non-final action result route differs")
    semantic, _ = _load_reference(
        result["semantic_result"], name=f"{action_id} semantic result",
    )
    scheduler, _ = _load_reference(
        result["scheduler_evidence"], name=f"{action_id} scheduler evidence",
    )
    return result, normalized, semantic, scheduler


def _load_smoke_report(
    reference: Mapping[str, Any], *, execution_id: str,
    recipe_sha256: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    report, normalized = _load_reference(reference, name=f"{execution_id} smoke report")
    validate_representation_training_report(
        report, expected_execution_id=execution_id,
        expected_recipe_sha256=recipe_sha256,
    )
    if (
        report.get("contract") != TRAINING_REPORT_CONTRACT
        or report.get("mode") != "smoke"
        or report.get("complete") is not True
        or report.get("scientific_complete") is not False
        or report.get("completed_optimizer_updates") != ACCEPTANCE_MAXIMUM_UPDATES
        or report.get("replicate_seed") != ACCEPTANCE_REPLICATE_SEED
        or report.get("validation", {}).get("rows") != ACCEPTANCE_VALIDATION_ROWS
        or report.get("recipe_sha256") != recipe_sha256
    ):
        raise PermissionError("two-update acceptance report semantics differ")
    return report, normalized


def build_two_update_acceptance_proof(
    *, authority: Mapping[str, Any],
    action_results: Mapping[str, Mapping[str, Any]],
    require_genuine: bool = False,
) -> dict[str, Any]:
    """Reopen the exact four cold/warm RSET/RREL two-update smoke reports."""

    authority_value, authority_ref = _load_reference(authority, name="non-final authority")
    _require_canonical_authority_route(authority_value, authority_ref)
    authority_hash = validate_nonfinal_acceptance_authority_static(
        authority_value, allow_local_fixture=not require_genuine,
    )
    if set(action_results) != set(TWO_UPDATE_ACTIONS):
        raise ValueError("two-update acceptance action-result registry differs")
    result_rows: dict[str, dict[str, Any]] = {}
    report_rows: dict[str, dict[str, Any]] = {}
    scheduler_rows: dict[str, dict[str, Any]] = {}
    full_loss_rows: dict[str, dict[str, Any]] = {}
    supporting_results: dict[str, dict[str, Any]] = {}
    supporting_scheduler: dict[str, dict[str, Any]] = {}
    job_by_action: dict[str, int] = {}
    genuine_rows: list[bool] = []
    for execution_id, action_id in TWO_UPDATE_ACTIONS.items():
        result, result_ref, semantic, scheduler = _load_action_result(
            action_results[execution_id], action_id=action_id,
            require_genuine=require_genuine,
        )
        if result["authority_sha256"] != authority_hash:
            raise ValueError("two-update action result binds a different authority")
        report, report_ref = _load_smoke_report(
            result["semantic_result"], execution_id=execution_id,
            recipe_sha256=authority_value["representation_recipe_sha256"],
        )
        if semantic["content_hash"] != report["content_hash"]:
            raise ValueError("two-update semantic result differs")
        job_by_action[action_id] = int(scheduler["job_id"])
        genuine_rows.append(result.get("authorization_capable") is True)
        result_rows[execution_id] = {
            "action_id": action_id,
            "action_result": result_ref,
            "action_result_sha256": result["content_hash"],
        }
        report_rows[execution_id] = {
            "action_id": action_id,
            "report": report_ref,
            "report_sha256": report["content_hash"],
            "selected_checkpoint_sha256": report[
                "selected_training_checkpoint_sha256"
            ],
        }
        scheduler_rows[execution_id] = {
            "action_id": action_id,
            "scheduler_evidence": dict(result["scheduler_evidence"]),
            "scheduler_evidence_sha256": scheduler["content_hash"],
            "job_id": scheduler["job_id"],
        }
        action = authority_value["actions"][action_id]
        full_loss_row = result["semantic_outputs"].get("acceptance_full_loss")
        if not isinstance(full_loss_row, Mapping):
            raise PermissionError("two-update action lacks its real-batch full-loss output")
        full_loss, full_loss_ref = _load_reference(
            full_loss_row["artifact"], name=f"{execution_id} full-loss output",
        )
        full_loss_hash = _validate_full_loss_output(
            full_loss, authority=authority_value, action_id=action_id,
        )
        if full_loss_hash != full_loss_row["content_hash"]:
            raise ValueError("two-update full-loss semantic output differs")
        full_loss_rows[execution_id] = {
            "action_id": action_id,
            "record": full_loss_ref,
            "record_sha256": full_loss_hash,
            "execution_receipt_sha256": result["execution_receipt_sha256"],
            "scheduler_job_id": scheduler["job_id"],
        }
        target_dependencies = action["dependencies"]
        if len(target_dependencies) != 1:
            raise PermissionError("two-update target dependency differs")
        target_action = target_dependencies[0]
        dependency_row = result["dependency_action_results"][target_action]
        target_result, target_result_ref, _, target_job = _load_action_result(
            dependency_row["action_result"], action_id=target_action,
            require_genuine=require_genuine,
        )
        if target_result.get("authority_sha256") != authority_hash:
            raise PermissionError(
                "two-update target result binds a different authority"
            )
        canonical_support = {
            "action_result": target_result_ref,
            "action_result_sha256": target_result["content_hash"],
        }
        if target_action in supporting_results and supporting_results[
            target_action
        ] != canonical_support:
            raise PermissionError("two-update consumers bind different target results")
        supporting_results[target_action] = canonical_support
        supporting_scheduler[target_action] = {
            "scheduler_evidence": dict(target_result["scheduler_evidence"]),
            "scheduler_evidence_sha256": target_job["content_hash"],
            "job_id": target_job["job_id"],
        }
        job_by_action[target_action] = int(target_job["job_id"])
        genuine_rows.append(target_result.get("authorization_capable") is True)
    if any(genuine_rows) and not all(genuine_rows):
        raise PermissionError("two-update acceptance mixes genuine and fixture evidence")
    authorization_capable = all(genuine_rows)
    if require_genuine and not authorization_capable:
        raise PermissionError("two-update acceptance requires genuine Tigris evidence")
    if authorization_capable and len(set(job_by_action.values())) != len(job_by_action):
        raise ValueError("two-update acceptance distinct actions reuse one job")
    return with_content_hash({
        "contract": TWO_UPDATE_ACCEPTANCE_PROOF_CONTRACT,
        "schema_version": 1,
        "authority": authority_ref,
        "authority_sha256": authority_hash,
        "source_commit": authority_value["source_commit"],
        "representation_recipe_sha256": authority_value[
            "representation_recipe_sha256"
        ],
        "full_loss_records": full_loss_rows,
        "action_results": result_rows,
        "supporting_action_results": supporting_results,
        "reports": report_rows,
        "scheduler_evidence": scheduler_rows,
        "supporting_scheduler_evidence": supporting_scheduler,
        "execution_ids": list(TWO_UPDATE_ACTIONS),
        "train_rows": ACCEPTANCE_TRAIN_ROWS,
        "validation_rows": ACCEPTANCE_VALIDATION_ROWS,
        "replicate_seed": ACCEPTANCE_REPLICATE_SEED,
        "effective_batch_size": ACCEPTANCE_EFFECTIVE_BATCH_SIZE,
        "maximum_optimizer_updates": ACCEPTANCE_MAXIMUM_UPDATES,
        "all_four_completed": True,
        "authorization_capable": authorization_capable,
        "final_role_accessed": False,
        "campaign_training_authorized": False,
        "pilot_submission_authorized": False,
    })


def validate_two_update_acceptance_proof(
    value: Mapping[str, Any], *, require_genuine: bool = False,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=TWO_UPDATE_ACCEPTANCE_PROOF_CONTRACT,
        expected_schema_version=1,
    )
    result_refs = {
        execution_id: value["action_results"][execution_id]["action_result"]
        for execution_id in TWO_UPDATE_ACTIONS
    }
    rebuilt = build_two_update_acceptance_proof(
        authority=value["authority"], action_results=result_refs,
        require_genuine=require_genuine,
    )
    if dict(value) != rebuilt:
        raise ValueError("two-update acceptance proof is not canonically derived")
    return digest


def _generation_reference(generation) -> dict[str, Any]:
    members = {
        "state": _reference(generation.state_path),
        "sidecar": _reference(generation.sidecar_path),
        "commit": _reference(generation.commit_path),
    }
    return {
        "workspace": str(generation.commit_path.parent.resolve()),
        "sequence": generation.sequence,
        "members": members,
        "generation_sha256": canonical_sha256(members),
    }


def _validate_generation_reference(value: Mapping[str, Any]):
    if not isinstance(value, Mapping) or set(value) != {
        "workspace", "sequence", "members", "generation_sha256",
    }:
        raise ValueError("USR1 resume-generation reference fields differ")
    workspace = Path(str(value["workspace"]))
    if not workspace.is_absolute() or workspace.is_symlink():
        raise PermissionError("USR1 resume workspace is unsafe")
    sequence = value["sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("USR1 resume sequence differs")
    generation = validate_resume_generation(workspace, sequence=sequence)
    rebuilt = _generation_reference(generation)
    if dict(value) != rebuilt:
        raise ValueError("USR1 resume-generation reference differs")
    return generation


def _exact_usr1_signal_number() -> int:
    number = getattr(signal, "SIGUSR1", None)
    if number is None:
        raise RuntimeError("SIGUSR1 is unavailable on this platform")
    return int(number)


def build_usr1_delivery_receipt(
    *, authority: Mapping[str, Any], resume_state_directory: str | Path,
    resumed_sequence: int, monitor: Any, worker_pid: int,
    scheduler_job_id: str | None, final_report_path: str | Path,
    local_fixture: bool = False, local_signal_number: int | None = None,
) -> dict[str, Any]:
    """Record one exact SIGUSR1 handler observation after update-one commit."""

    authority_value, authority_ref = _load_reference(authority, name="non-final authority")
    _require_canonical_authority_route(authority_value, authority_ref)
    authority_hash = validate_nonfinal_acceptance_authority_static(
        authority_value, allow_local_fixture=local_fixture,
    )
    observed = tuple(monitor.observed_signals())
    if observed != ("SIGUSR1",) or monitor.observed_exact_usr1() is not True:
        raise PermissionError("USR1 receipt requires exactly one SIGUSR1 and no TERM")
    if isinstance(worker_pid, bool) or not isinstance(worker_pid, int) or worker_pid <= 0:
        raise ValueError("USR1 worker PID differs")
    if Path(final_report_path).exists():
        raise PermissionError("interrupted USR1 stage published a terminal report")
    generation = validate_resume_generation(
        Path(resume_state_directory).resolve(), sequence=resumed_sequence,
    )
    cursor = generation.sidecar.get("payload", {})
    exact_cursor = {
        "completed_pass": 0,
        "completed_update": 1,
        "next_canonical_batch": 1,
    }
    if any(cursor.get(name) != expected for name, expected in exact_cursor.items()):
        raise ValueError("USR1 committed cursor is not exactly after update one")
    if local_fixture:
        if scheduler_job_id is not None:
            raise ValueError("local USR1 fixture cannot claim a Slurm job")
        signal_number = (
            10 if local_signal_number is None else int(local_signal_number)
        )
        authorization_capable = False
    else:
        if (
            scheduler_job_id is None
            or _SLURM_JOB_ID.fullmatch(scheduler_job_id) is None
            or os.environ.get("SLURM_JOB_ID") != scheduler_job_id
            or worker_pid != os.getpid()
        ):
            raise PermissionError("USR1 receipt lacks the live exact Slurm process")
        signal_number = _exact_usr1_signal_number()
        authorization_capable = True
    generation_ref = _generation_reference(generation)
    return with_content_hash({
        "contract": USR1_DELIVERY_RECEIPT_CONTRACT,
        "schema_version": 1,
        "authority": authority_ref,
        "authority_sha256": authority_hash,
        "action_id": "usr1_interrupt",
        "source_commit": authority_value["source_commit"],
        "representation_recipe_sha256": authority_value[
            "representation_recipe_sha256"
        ],
        "scheduler_job_id": scheduler_job_id,
        "signal_name": "SIGUSR1",
        "signal_number": signal_number,
        "observed_signals": ["SIGUSR1"],
        "worker_pid": worker_pid,
        "resume_generation": generation_ref,
        "resumed_sequence": resumed_sequence,
        "resumed_commit_sha256": generation.commit["content_hash"],
        "resumed_state_logical_sha256": generation.commit["payload"][
            "state_logical_sha256"
        ],
        "cursor": exact_cursor,
        "interrupted": True,
        "final_report_published": False,
        "authorization_capable": authorization_capable,
        "final_role_accessed": False,
        "scientific_training_authorized": False,
        "campaign_training_authorized": False,
        "reservation_authorized": False,
        "shared_final_authorized": False,
        "pilot_submission_authorized": False,
    })


def validate_usr1_delivery_receipt(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value, expected_contract=USR1_DELIVERY_RECEIPT_CONTRACT,
        expected_schema_version=1,
    )
    expected_fields = {
        "contract", "schema_version", "authority", "authority_sha256",
        "action_id", "source_commit", "representation_recipe_sha256",
        "scheduler_job_id", "signal_name", "signal_number", "observed_signals",
        "worker_pid", "resume_generation", "resumed_sequence",
        "resumed_commit_sha256", "resumed_state_logical_sha256", "cursor",
        "interrupted", "final_report_published", "authorization_capable",
        "final_role_accessed", "scientific_training_authorized",
        "campaign_training_authorized", "reservation_authorized",
        "shared_final_authorized", "pilot_submission_authorized", "content_hash",
    }
    if set(value) != expected_fields:
        raise ValueError("USR1 delivery receipt fields differ")
    authority, authority_ref = _load_reference(value["authority"], name="non-final authority")
    authority_hash = validate_nonfinal_acceptance_authority_static(
        authority, allow_local_fixture=value.get("authorization_capable") is False,
    )
    generation = _validate_generation_reference(value["resume_generation"])
    expected_cursor = {
        "completed_pass": 0, "completed_update": 1, "next_canonical_batch": 1,
    }
    scheduler_job_id = value["scheduler_job_id"]
    authorization_capable = value["authorization_capable"]
    if authorization_capable is True:
        if not isinstance(scheduler_job_id, str) or _SLURM_JOB_ID.fullmatch(scheduler_job_id) is None:
            raise PermissionError("genuine USR1 receipt lacks an exact Slurm job ID")
        if value["signal_number"] != _exact_usr1_signal_number():
            raise PermissionError("genuine USR1 receipt signal number differs")
    elif authorization_capable is False:
        if scheduler_job_id is not None:
            raise PermissionError("local USR1 receipt claims a Slurm job")
        if isinstance(value["signal_number"], bool) or not isinstance(value["signal_number"], int):
            raise ValueError("local USR1 receipt signal number differs")
    else:
        raise ValueError("USR1 authorization capability differs")
    expected = {
        "authority": authority_ref,
        "authority_sha256": authority_hash,
        "action_id": "usr1_interrupt",
        "source_commit": authority["source_commit"],
        "representation_recipe_sha256": authority["representation_recipe_sha256"],
        "signal_name": "SIGUSR1",
        "observed_signals": ["SIGUSR1"],
        "resume_generation": _generation_reference(generation),
        "resumed_sequence": generation.sequence,
        "resumed_commit_sha256": generation.commit["content_hash"],
        "resumed_state_logical_sha256": generation.commit["payload"][
            "state_logical_sha256"
        ],
        "cursor": expected_cursor,
        "interrupted": True,
        "final_report_published": False,
        "final_role_accessed": False,
        "scientific_training_authorized": False,
        "campaign_training_authorized": False,
        "reservation_authorized": False,
        "shared_final_authorized": False,
        "pilot_submission_authorized": False,
    }
    if any(value.get(name) != expected_value for name, expected_value in expected.items()):
        raise PermissionError("USR1 delivery receipt authority or semantics differ")
    return digest


_EXACT_RESUME_EQUAL_FIELDS: Final = (
    "node_id", "execution_id", "registered_execution_id", "replicate_seed",
    "campaign_sha256", "paired_rng_streams", "graph_sha256", "recipe_sha256",
    "parent_recipe_sha256", "parent_counterpart", "strategy", "track", "rung",
    "mode", "completed_optimizer_updates", "completed_natural_population_passes",
    "validation_history", "validation", "selection_sha256",
    "selected_checkpoint_id", "selected_training_checkpoint_sha256",
    "interval_mean_history", "calibration", "target_generation_sha256",
    "target_logical_sha256", "target_manifest_sha256",
    "predecessor_logit_logical_sha256", "shuffle_map_sha256",
    "projection_diagnostics",
)


def build_usr1_exact_resume_proof_v2(
    *, authority: Mapping[str, Any],
    action_results: Mapping[str, Mapping[str, Any]],
    require_genuine: bool = False,
) -> dict[str, Any]:
    """Prove real-signal interruption and fresh-process exact two-update resume."""

    authority_value, authority_ref = _load_reference(authority, name="non-final authority")
    _require_canonical_authority_route(authority_value, authority_ref)
    authority_hash = validate_nonfinal_acceptance_authority_static(
        authority_value, allow_local_fixture=not require_genuine,
    )
    if set(action_results) != set(USR1_ACTIONS):
        raise ValueError("USR1 action-result stage registry differs")
    loaded_results = {
        action_id: _load_action_result(
            action_results[action_id], action_id=action_id,
            require_genuine=require_genuine,
        )
        for action_id in USR1_ACTIONS
    }
    if any(
        result[0]["authority_sha256"] != authority_hash
        for result in loaded_results.values()
    ):
        raise ValueError("USR1 action result binds a different authority")
    reference_result, reference_result_ref, reference_semantic, _ = loaded_results[
        "usr1_reference"
    ]
    interrupt_result, interrupt_result_ref, receipt, _ = loaded_results[
        "usr1_interrupt"
    ]
    resume_result, resume_result_ref, resumed_semantic, _ = loaded_results[
        "usr1_resume"
    ]
    reference, reference_ref = _load_smoke_report(
        reference_result["semantic_result"], execution_id=USR1_EXECUTION_ID,
        recipe_sha256=authority_value["representation_recipe_sha256"],
    )
    resumed, resumed_ref = _load_smoke_report(
        resume_result["semantic_result"], execution_id=USR1_EXECUTION_ID,
        recipe_sha256=authority_value["representation_recipe_sha256"],
    )
    if (
        reference_semantic["content_hash"] != reference["content_hash"]
        or resumed_semantic["content_hash"] != resumed["content_hash"]
    ):
        raise ValueError("USR1 action-result semantic report differs")
    receipt_ref = dict(interrupt_result["semantic_result"])
    validate_usr1_delivery_receipt(receipt)
    if (
        receipt["authority_sha256"] != authority_hash
        or receipt["source_commit"] != authority_value["source_commit"]
        or receipt["representation_recipe_sha256"]
        != authority_value["representation_recipe_sha256"]
    ):
        raise ValueError("USR1 delivery receipt lineage differs")
    result_rows: dict[str, dict[str, Any]] = {}
    scheduler_rows: dict[str, dict[str, Any]] = {}
    supporting_results: dict[str, dict[str, Any]] = {}
    supporting_scheduler: dict[str, dict[str, Any]] = {}
    job_by_action: dict[str, int] = {}
    genuine_rows: list[bool] = []
    for action_id in USR1_ACTIONS:
        result, result_ref, _, scheduler = loaded_results[action_id]
        scheduler_ref = dict(result["scheduler_evidence"])
        result_rows[action_id] = {
            "action_result": result_ref,
            "action_result_sha256": result["content_hash"],
        }
        scheduler_rows[action_id] = {
            "scheduler_evidence": scheduler_ref,
            "scheduler_evidence_sha256": scheduler["content_hash"],
            "job_id": scheduler["job_id"],
        }
        job_by_action[action_id] = int(scheduler["job_id"])
        genuine_rows.append(result.get("authorization_capable") is True)
        target_row = result["dependency_action_results"].get("target_d0c")
        if not isinstance(target_row, Mapping):
            raise PermissionError("USR1 action lacks its target dependency result")
        target_result, target_result_ref, _, target_scheduler = _load_action_result(
            target_row["action_result"], action_id="target_d0c",
            require_genuine=require_genuine,
        )
        if target_result.get("authority_sha256") != authority_hash:
            raise PermissionError("USR1 target result binds a different authority")
        canonical_target = {
            "action_result": target_result_ref,
            "action_result_sha256": target_result["content_hash"],
        }
        if "target_d0c" in supporting_results and supporting_results[
            "target_d0c"
        ] != canonical_target:
            raise PermissionError("USR1 stages bind different target results")
        supporting_results["target_d0c"] = canonical_target
        supporting_scheduler["target_d0c"] = {
            "scheduler_evidence": dict(target_result["scheduler_evidence"]),
            "scheduler_evidence_sha256": target_scheduler["content_hash"],
            "job_id": target_scheduler["job_id"],
        }
        job_by_action["target_d0c"] = int(target_scheduler["job_id"])
        genuine_rows.append(target_result.get("authorization_capable") is True)
    resume_interrupt = resume_result["dependency_action_results"].get(
        "usr1_interrupt"
    )
    if (
        not isinstance(resume_interrupt, Mapping)
        or resume_interrupt.get("action_result") != interrupt_result_ref
        or resume_interrupt.get("action_result_sha256")
        != interrupt_result["content_hash"]
    ):
        raise PermissionError("USR1 resume does not bind the exact interrupt result")
    if any(genuine_rows) and not all(genuine_rows):
        raise PermissionError("USR1 proof mixes genuine and fixture scheduler evidence")
    authorization_capable = all(genuine_rows) and receipt["authorization_capable"] is True
    if require_genuine and not authorization_capable:
        raise PermissionError("USR1 proof requires genuine Tigris signal evidence")
    if authorization_capable:
        if len(set(job_by_action.values())) != len(job_by_action):
            raise ValueError("USR1 distinct actions reuse one job")
        if int(receipt["scheduler_job_id"]) != scheduler_rows["usr1_interrupt"]["job_id"]:
            raise ValueError("USR1 delivery receipt job differs from scheduler evidence")
    elif receipt["authorization_capable"] is True:
        raise PermissionError("USR1 receipt cannot outrank scheduler evidence")

    unequal = [
        name for name in _EXACT_RESUME_EQUAL_FIELDS
        if reference.get(name) != resumed.get(name)
    ]
    if unequal:
        raise ValueError(f"USR1-resumed scientific trajectory differs: {unequal}")
    reference_audit = reference.get("resume_audit")
    resumed_audit = resumed.get("resume_audit")
    if (
        not isinstance(reference_audit, Mapping)
        or not isinstance(resumed_audit, Mapping)
        or reference_audit.get("highest_loaded_sequence") is not None
        or resumed_audit.get("highest_loaded_sequence") != receipt["resumed_sequence"]
        or resumed_audit.get("invalid_commits") != []
    ):
        raise ValueError("USR1 proof lacks an exact committed fresh-process reload")
    reference_deployable = reference.get("deployable_extraction")
    resumed_deployable = resumed.get("deployable_extraction")
    if (
        not isinstance(reference_deployable, Mapping)
        or not isinstance(resumed_deployable, Mapping)
        or reference_deployable.get("checkpoint_sha256")
        != resumed_deployable.get("checkpoint_sha256")
    ):
        raise ValueError("USR1-resumed deployable checkpoint differs")
    equality_payload = {
        name: resumed[name] for name in _EXACT_RESUME_EQUAL_FIELDS
    }
    equality_payload["deployable_checkpoint_sha256"] = resumed_deployable[
        "checkpoint_sha256"
    ]
    return with_content_hash({
        "contract": USR1_EXACT_RESUME_PROOF_CONTRACT,
        "schema_version": 1,
        "authority": authority_ref,
        "authority_sha256": authority_hash,
        "source_commit": authority_value["source_commit"],
        "representation_recipe_sha256": authority_value[
            "representation_recipe_sha256"
        ],
        "execution_id": USR1_EXECUTION_ID,
        "replicate_seed": ACCEPTANCE_REPLICATE_SEED,
        "action_results": result_rows,
        "supporting_action_results": supporting_results,
        "uninterrupted_report": reference_ref,
        "uninterrupted_report_sha256": reference["content_hash"],
        "resumed_report": resumed_ref,
        "resumed_report_sha256": resumed["content_hash"],
        "delivery_receipt": receipt_ref,
        "delivery_receipt_sha256": receipt["content_hash"],
        "scheduler_evidence": scheduler_rows,
        "supporting_scheduler_evidence": supporting_scheduler,
        "updates_before_usr1": 1,
        "total_optimizer_updates": ACCEPTANCE_MAXIMUM_UPDATES,
        "resumed_sequence": receipt["resumed_sequence"],
        "resumed_commit_sha256": receipt["resumed_commit_sha256"],
        "scientific_trajectory_sha256": canonical_sha256(equality_payload),
        "selected_checkpoint_sha256": resumed[
            "selected_training_checkpoint_sha256"
        ],
        "deployable_checkpoint_sha256": resumed_deployable["checkpoint_sha256"],
        "actual_sigusr1_observed": True,
        "fresh_process_resume": authorization_capable,
        "exact_resume": True,
        "authorization_capable": authorization_capable,
        "final_role_accessed": False,
        "campaign_training_authorized": False,
        "pilot_submission_authorized": False,
    })


def validate_usr1_exact_resume_proof_v2(
    value: Mapping[str, Any], *, require_genuine: bool = False,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=USR1_EXACT_RESUME_PROOF_CONTRACT,
        expected_schema_version=1,
    )
    result_refs = {
        action_id: value["action_results"][action_id]["action_result"]
        for action_id in USR1_ACTIONS
    }
    rebuilt = build_usr1_exact_resume_proof_v2(
        authority=value["authority"],
        action_results=result_refs,
        require_genuine=require_genuine,
    )
    if dict(value) != rebuilt:
        raise ValueError("USR1 exact-resume v2 proof is not canonically derived")
    return digest


__all__ = [
    "ACCEPTANCE_EFFECTIVE_BATCH_SIZE", "ACCEPTANCE_FINAL_ROWS",
    "ACCEPTANCE_MAXIMUM_UPDATES", "ACCEPTANCE_REPLICATE_SEED",
    "ACCEPTANCE_TRAIN_ROWS", "ACCEPTANCE_VALIDATION_ROWS", "ACTION_IDS",
    "ACTION_REGISTRY", "ACTION_REGISTRY_SHA256",
    "NONFINAL_ACCEPTANCE_AUTHORIZATION_PHRASE", "NonfinalActionRequest",
    "SOURCE_RUNTIME_ROW_BY_ACTION",
    "NonfinalActionResult", "TWO_UPDATE_ACTIONS", "USR1_ACTIONS",
    "USR1_EXECUTION_ID", "WORKER_NAMES", "build_nonfinal_acceptance_authority",
    "build_nonfinal_acceptance_action_inputs",
    "build_nonfinal_acceptance_action_inputs_fixture",
    "build_nonfinal_acceptance_action_result",
    "build_nonfinal_acceptance_execution_receipt",
    "build_two_update_acceptance_proof", "build_usr1_delivery_receipt",
    "build_usr1_exact_resume_proof_v2", "execute_nonfinal_action",
    "validate_nonfinal_acceptance_action_assembly",
    "validate_nonfinal_acceptance_action_inputs",
    "validate_nonfinal_acceptance_action_result",
    "validate_nonfinal_acceptance_authority",
    "validate_nonfinal_acceptance_authority_static",
    "validate_nonfinal_acceptance_execution_receipt",
    "nonfinal_acceptance_action_result_path",
    "nonfinal_acceptance_execution_receipt_path",
    "nonfinal_acceptance_raw_sacct_path",
    "nonfinal_acceptance_scheduler_evidence_path",
    "resolve_nonfinal_acceptance_dependency_action_results",
    "validate_two_update_acceptance_proof", "validate_usr1_delivery_receipt",
    "validate_usr1_exact_resume_proof_v2",
]

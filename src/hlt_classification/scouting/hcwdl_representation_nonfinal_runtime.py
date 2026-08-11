"""Genuine scalar runtime for bounded non-final HCWDL-RKD actions.

The bridge reuses the fixed production target/training adapters after a live
runtime measurement.  It projects one reviewed full-smoke row into the exact
authority-private population and output workspace.  Its ephemeral
``CampaignTask`` value is only the fixed adapter interface and is never a
registered campaign identity; this bridge creates no reservation, shared-final
capability, final-role reader, or scheduler submission.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import os
from pathlib import Path
import re
import signal
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    load_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
    write_immutable_json,
)
from .hcwdl_representation_campaign import CampaignTask
from .hcwdl_representation_nonfinal_acceptance import (
    ACCEPTANCE_MAXIMUM_UPDATES,
    ACCEPTANCE_REPLICATE_SEED,
    ACCEPTANCE_TRAIN_ROWS,
    ACCEPTANCE_VALIDATION_ROWS,
    ACTION_REGISTRY,
    SOURCE_RUNTIME_ROW_BY_ACTION,
    build_usr1_delivery_receipt,
    resolve_nonfinal_acceptance_dependency_action_results,
    validate_nonfinal_acceptance_action_result,
    validate_nonfinal_acceptance_action_assembly,
    validate_nonfinal_acceptance_authority,
    validate_nonfinal_acceptance_authority_static as _validate_authority_static,
    validate_usr1_delivery_receipt,
)
from .hcwdl_representation_runtime_binding import resolve_runtime_row
from .hcwdl_representation_training import (
    RepresentationTrainingInterrupted,
    validate_acceptance_real_batch_full_loss_record,
)


_SLURM_JOB_ID: Final = re.compile(r"^[1-9][0-9]*$")
_USR1_WAIT_TIMEOUT_SECONDS: Final = 900


@dataclass(frozen=True)
class NonfinalProductionActionResult:
    """Exact worker outputs consumed by the execution-receipt builder."""

    authority_sha256: str
    action_id: str
    action_spec_sha256: str
    source_task_key: str
    source_runtime_row_sha256: str
    workspace: Path
    semantic_outputs: Mapping[str, Mapping[str, str]]
    dependency_action_results: Mapping[str, Mapping[str, str]]
    scheduler_job_id: str


def _artifact_reference(path: str | Path) -> dict[str, str]:
    raw = Path(path)
    if (
        not raw.is_file()
        or any(candidate.is_symlink() for candidate in (raw, *raw.parents))
    ):
        raise FileNotFoundError(f"non-final runtime artifact is absent: {raw}")
    value = raw.resolve()
    return {"path": str(value), "sha256": sha256_file(value)}


def _load_reference(
    reference: Mapping[str, Any], *, name: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise ValueError(f"non-final runtime {name} reference fields differ")
    path = Path(str(reference["path"]))
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PermissionError(f"non-final runtime {name} path is unsafe")
    digest = require_sha256(reference["sha256"], name=f"{name} byte SHA-256")
    if sha256_file(path) != digest:
        raise ValueError(f"non-final runtime {name} bytes differ")
    value = load_json(path)
    if not isinstance(value, Mapping):
        raise TypeError(f"non-final runtime {name} is not an object")
    return dict(value), {"path": str(path), "sha256": digest}


def _authority_static_context(
    authority: Mapping[str, Any], *, project_dir: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Delegate the no-data gate to the canonical static authority validator."""

    _validate_authority_static(
        authority, project_dir=project_dir,
    )
    action_inputs, _ = _load_reference(
        authority["action_inputs"], name="action inputs",
    )
    planning, _ = _load_reference(
        authority["planning_spec"], name="planning spec",
    )
    runtime, _ = _load_reference(
        authority["runtime_binding"], name="runtime binding",
    )
    return action_inputs, planning, runtime


def validate_nonfinal_acceptance_authority_static(
    authority: Mapping[str, Any], *, project_dir: str | Path | None = None,
) -> str:
    """Re-export the canonical static validator for worker runtimes."""

    return _validate_authority_static(
        authority, project_dir=project_dir,
    )


def _source_task(planning: Mapping[str, Any], task_key: str) -> Mapping[str, Any]:
    rows = [row for row in planning.get("tasks", ()) if row.get("task_key") == task_key]
    if len(rows) != 1:
        raise KeyError("non-final source task is not one planning row")
    return rows[0]


def _campaign_task(
    source: Mapping[str, Any], *, action_id: str,
    registered_outputs: tuple[str, ...],
) -> CampaignTask:
    action = ACTION_REGISTRY[action_id]
    return CampaignTask(
        task_key=f"acceptance_nonfinal_{action_id}",
        kind=str(source["kind"]),
        dependencies=(),
        resource_class=str(action["resource_class"]),
        graph_node=action.get("execution_id"),
        logical_bank=action.get("target_identity"),
        target_purpose=(
            "nonfinal_acceptance" if action["kind"] == "target_prepare" else None
        ),
        deterministic_worker=action["worker_role"] == "deterministic",
        registered_inputs=tuple(source.get("registered_inputs", ())),
        registered_outputs=registered_outputs,
    )


def _tagged_input(assembly_value: object, *, tag: str, name: str) -> str:
    if not isinstance(assembly_value, Mapping) or set(assembly_value) != {tag}:
        raise ValueError(f"non-final source assembly {name} tag differs")
    logical = str(assembly_value[tag])
    if not logical.startswith("${") or not logical.endswith("}"):
        raise ValueError(f"non-final source assembly {name} logical input differs")
    return logical


def _replace_registered_input(
    inputs: dict[str, Any], *, logical: str, reference: Mapping[str, str],
) -> None:
    if logical not in inputs:
        raise KeyError(f"non-final projected input is absent: {logical}")
    inputs[logical] = dict(reference)


def _action_descriptor(
    action_inputs: Mapping[str, Any], *, action_id: str,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Mapping[str, str]]]:
    row = action_inputs["actions"][action_id]
    artifacts = row["input_artifacts"]
    descriptor, descriptor_ref = _load_reference(
        artifacts["action_assembly"], name=f"{action_id} action assembly",
    )
    descriptor_hash = validate_nonfinal_acceptance_action_assembly(descriptor)
    action = ACTION_REGISTRY[action_id]
    if (
        descriptor.get("content_hash") != descriptor_hash
        or descriptor.get("action_id") != action_id
        or descriptor.get("action_spec_sha256")
        != action["action_spec_sha256"]
        or descriptor.get("source_task_key")
        != SOURCE_RUNTIME_ROW_BY_ACTION[action_id][0]
        or descriptor.get("source_array_index")
        != SOURCE_RUNTIME_ROW_BY_ACTION[action_id][1]
        or descriptor.get("source_kind")
        != SOURCE_RUNTIME_ROW_BY_ACTION[action_id][2]
        or descriptor.get("train_rows") != action["train_rows"]
        or descriptor.get("validation_rows") != action["validation_rows"]
        or descriptor.get("final_rows") != 0
        or descriptor.get("dependencies") != action["dependencies"]
        or descriptor.get("workspace") is None
        or descriptor.get("campaign_task_identity_reused") is not False
        or any(
            descriptor.get(name) is not False
            for name in (
                "reservation_authorized", "pilot_submission_authorized",
                "final_role_access_authorized", "shared_final_authorized",
            )
        )
    ):
        raise PermissionError("non-final action descriptor semantics differ")
    normalized = {
        str(name): dict(reference) for name, reference in artifacts.items()
    }
    return descriptor, descriptor_ref, normalized


def _dependency_action_results(
    *, authority: Mapping[str, Any], descriptor_ref: Mapping[str, str],
    action_id: str,
) -> tuple[
    dict[str, dict[str, str]], dict[str, dict[str, Any]],
]:
    del descriptor_ref
    result_refs = resolve_nonfinal_acceptance_dependency_action_results(
        authority, action_id=action_id, require_genuine=True,
    )
    results: dict[str, dict[str, Any]] = {}
    for dependency, reference in result_refs.items():
        result = load_json(reference["path"])
        validate_nonfinal_acceptance_action_result(
            result, expected_action_id=dependency, require_genuine=True,
        )
        if result.get("authority_sha256") != authority["content_hash"]:
            raise PermissionError("non-final dependency binds another authority")
        results[dependency] = dict(result)
    return result_refs, results


def _source_assembly(source_row: Mapping[str, Any]) -> dict[str, Any]:
    parameters = source_row.get("parameters")
    assembly = parameters.get("assembly") if isinstance(parameters, Mapping) else None
    if not isinstance(assembly, Mapping):
        raise ValueError("non-final source production assembly is absent")
    return copy.deepcopy(dict(assembly))


def _owner(authority: Mapping[str, Any], *, action_id: str) -> dict[str, Any]:
    return {
        "owner_kind": "bounded_nonfinal_acceptance",
        "authority_sha256": authority["content_hash"],
        "action_id": action_id,
        "action_spec_sha256": ACTION_REGISTRY[action_id]["action_spec_sha256"],
    }


def _output_row(action_id: str, registered_output: str) -> dict[str, Any]:
    return {
        "task_key": f"acceptance_nonfinal_{action_id}",
        "array_index": None,
        "registered_output": registered_output,
        "authority_action_private": True,
    }


def _workspace_path(descriptor: Mapping[str, Any]) -> Path:
    raw = Path(str(descriptor["workspace"]))
    if not raw.is_absolute() or any(
        candidate.is_symlink() for candidate in (raw, *raw.parents)
    ):
        raise PermissionError("non-final action workspace route is unsafe")
    return raw.resolve()


def _nonfinal_root_from_workspace(workspace: Path) -> Path:
    workspace = workspace.resolve()
    if workspace.parent.name != "workspaces":
        raise PermissionError("non-final action workspace route differs")
    return workspace.parent.parent


def _target_semantic_from_dependency(
    result: Mapping[str, Any], *, target_identity: str,
) -> tuple[Path, dict[str, str], dict[str, str]]:
    semantic, semantic_ref = _load_reference(
        result["semantic_result"], name=f"{target_identity} target semantic result",
    )
    generation_directory = Path(semantic_ref["path"]).resolve().parent
    if generation_directory.name != semantic.get("payload", {}).get("generation_id"):
        raise ValueError("non-final dependency target generation path differs")
    bank_root = generation_directory.parent.parent
    from .hcwdl_representation_targets import validate_target_generation

    validated = validate_target_generation(bank_root, generation_directory.name)
    if (
        validated.get("parents", {}).get("target_generation")
        != semantic.get("content_hash")
        or validated.get("payload", {}).get("logical_bank_id") != target_identity
        or validated.get("payload", {}).get("purpose") != "nonfinal_acceptance"
    ):
        raise PermissionError("non-final dependency target lineage differs")
    from .hcwdl_representation_task_runtime import _directory_identity_candidates

    directory_ref = {
        "path": str(generation_directory),
        "sha256": next(iter(_directory_identity_candidates(generation_directory))),
    }
    target_lineage = {
        "target_generation_sha256": semantic["content_hash"],
        "target_logical_sha256": validated["payload"][
            "logical_target_sha256"
        ],
        "target_manifest_sha256": validated["content_hash"],
    }
    return generation_directory, directory_ref, target_lineage


def _execute_target(
    *, authority: Mapping[str, Any], action_id: str,
    descriptor: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, str]],
    source_task: Mapping[str, Any], source_row: Mapping[str, Any],
    live_runtime: Mapping[str, Any], scheduler_job_id: str,
    descriptor_ref: Mapping[str, str],
) -> NonfinalProductionActionResult:
    from .hcwdl_representation_task_runtime import _validate_input_bytes
    from .hcwdl_representation_production import target_build_adapter
    from .hcwdl_representation_targets import derive_target_generation_id

    assembly = _source_assembly(source_row)
    inputs = copy.deepcopy(dict(source_row["inputs"]))
    replacements = (
        (assembly["row_selection"], artifacts["bounded_row_selection"], "row selection"),
        (
            assembly["consumer_registry"],
            artifacts["target_consumer_registry"],
            "consumer registry",
        ),
        (assembly["storage_estimate"], artifacts["bounded_storage_estimate"], "storage"),
    )
    for tag_value, reference, name in replacements:
        logical = _tagged_input(
            tag_value, tag="registered_reference", name=name,
        )
        _replace_registered_input(inputs, logical=logical, reference=reference)
    projected_for_authentication = {**dict(source_row), "inputs": inputs}
    materialized = _validate_input_bytes(
        projected_for_authentication,
        spec=_load_reference(authority["planning_spec"], name="planning spec")[0],
    )
    logical_ref = materialized[
        _tagged_input(
            assembly["logical_bank"], tag="registered_reference",
            name="logical bank",
        )
    ]
    logical = load_json(logical_ref["path"])
    registry = load_json(artifacts["target_consumer_registry"]["path"])
    generation_id = derive_target_generation_id(
        logical["content_hash"], registry["content_hash"],
        purpose=registry["payload"]["purpose"],
        generation_parent_sha256=registry["payload"]["generation_parent_sha256"],
    )
    workspace = _workspace_path(descriptor)
    bank_root = workspace / "targets" / str(descriptor["target_identity"])
    generation_directory = bank_root / "generations" / generation_id
    if generation_directory.exists():
        raise FileExistsError(
            "non-final target action already has a semantic generation"
        )
    logical_output = str(source_task["registered_outputs"][0])
    task = _campaign_task(
        source_task, action_id=action_id,
        registered_outputs=(logical_output,),
    )
    assembly.update({
        "bank_root": str(bank_root),
        "build_owner": _owner(authority, action_id=action_id),
    })
    runtime_context = {
        **dict(source_row),
        "inputs": materialized,
        "outputs": {logical_output: str(generation_directory)},
        "parameters": {
            "adapter_contract": source_row["parameters"]["adapter_contract"],
            "task_kind": "target_build",
            "assembly": assembly,
        },
        "_live_worker_runtime": dict(live_runtime),
    }
    target_build_adapter(
        {"content_hash": authority["content_hash"]},
        task, None, runtime_context,
    )
    semantic_path = generation_directory / "generation.json"
    semantic_ref = _artifact_reference(semantic_path)
    semantic = load_json(semantic_path)
    if semantic.get("contract") != "HCWDL_REPRESENTATION_TARGET_GENERATION/v1":
        raise ValueError("non-final target semantic result contract differs")
    return NonfinalProductionActionResult(
        authority_sha256=authority["content_hash"],
        action_id=action_id,
        action_spec_sha256=ACTION_REGISTRY[action_id]["action_spec_sha256"],
        source_task_key=str(descriptor["source_task_key"]),
        source_runtime_row_sha256=str(descriptor["source_runtime_row_sha256"]),
        workspace=workspace,
        semantic_outputs={"primary": semantic_ref},
        dependency_action_results={},
        scheduler_job_id=scheduler_job_id,
    )


def _wait_for_real_signal(monitor: Any) -> None:
    """Wait at the committed update-one cursor for an actual POSIX signal."""

    if os.name != "posix" or not all(
        hasattr(signal, name) for name in ("pause", "SIGALRM", "setitimer")
    ):
        raise RuntimeError("the exact USR1 acceptance barrier requires POSIX signals")
    if monitor.is_requested():
        raise RuntimeError("USR1 arrived before the committed update-one boundary")
    previous = signal.getsignal(signal.SIGALRM)

    def timed_out(_signum, _frame):
        raise TimeoutError("timed out waiting for the real USR1 acceptance signal")

    signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, _USR1_WAIT_TIMEOUT_SECONDS)
    try:
        while not monitor.is_requested():
            signal.pause()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _validate_full_loss_output(
    path: Path, *, authority: Mapping[str, Any], action_id: str,
    descriptor: Mapping[str, Any], report: Mapping[str, Any] | None,
) -> dict[str, str]:
    reference = _artifact_reference(path)
    value = load_json(path)
    validate_acceptance_real_batch_full_loss_record(
        value,
        expected_authority_sha256=authority["content_hash"],
        expected_action_id=action_id,
        expected_execution_id=str(descriptor["execution_id"]),
        expected_recipe_sha256=authority["representation_recipe_sha256"],
    )
    if report is not None and (
        value["registered_execution_id"] != report["registered_execution_id"]
        or value["diagnostic_batch_sha256"] != report["diagnostic_batch_sha256"]
        or value["target_generation_sha256"]
        != report["target_generation_sha256"]
        or value["target_logical_sha256"] != report["target_logical_sha256"]
        or value["target_manifest_sha256"] != report["target_manifest_sha256"]
    ):
        raise PermissionError("non-final full-loss/report batch lineage differs")
    return reference


def _validate_usr1_resume_workspace(
    workspace: Path, *, receipt_reference: Mapping[str, str],
    execution_id: str,
) -> None:
    """Admit only the exact nonterminal state produced by USR1 interrupt."""

    if not workspace.is_dir() or workspace.is_symlink():
        raise PermissionError("USR1 resume workspace is unsafe")
    entries = {path.name: path for path in workspace.iterdir()}
    expected = {"calibration", "checkpoints", "resume"}
    if set(entries) != expected or any(path.is_symlink() for path in entries.values()):
        raise PermissionError("USR1 resume workspace inventory differs")
    expected_receipt = (
        _nonfinal_root_from_workspace(workspace)
        / "usr1" / "interrupt" / "receipt.json"
    )
    if Path(receipt_reference["path"]).resolve() != expected_receipt:
        raise PermissionError("USR1 interrupt receipt route differs")

    calibration_files = {
        path.name: path for path in entries["calibration"].iterdir()
    }
    if set(calibration_files) != {"diagnostic_batch.json", "selection.json"} or any(
        not path.is_file() or path.is_symlink()
        for path in calibration_files.values()
    ):
        raise PermissionError("USR1 resume diagnostic inventory differs")
    diagnostic = load_json(calibration_files["diagnostic_batch.json"])
    selection = load_json(calibration_files["selection.json"])
    from .hcwdl_representation_contracts import (
        CALIBRATION_SELECTION_CONTRACT,
        DIAGNOSTIC_BATCH_CONTRACT,
    )

    validate_content_hash(
        diagnostic, expected_contract=DIAGNOSTIC_BATCH_CONTRACT,
        expected_schema_version=1,
    )
    validate_content_hash(
        selection, expected_contract=CALIBRATION_SELECTION_CONTRACT,
        expected_schema_version=1,
    )
    if (
        diagnostic.get("payload", {}).get("execution_id") != execution_id
        or diagnostic.get("payload", {}).get("rows") != 256
    ):
        raise PermissionError("USR1 resume diagnostic semantics differ")

    checkpoint_root = entries["checkpoints"]
    expected_directories = {
        Path("selected"), Path("selected/staging"),
        Path("selected/staging/candidates"),
    }
    observed_directories: set[Path] = set()
    for path in checkpoint_root.rglob("*"):
        if path.is_symlink() or not path.is_dir():
            raise PermissionError("USR1 resume contains checkpoint staging output")
        observed_directories.add(path.relative_to(checkpoint_root))
    if observed_directories != expected_directories:
        raise PermissionError("USR1 resume checkpoint staging inventory differs")

    from .hcwdl_representation_resume import scan_resume_generations

    scan = scan_resume_generations(entries["resume"])
    if (
        [generation.sequence for generation in scan.valid_generations] != [0]
        or scan.invalid_commits or scan.orphan_files
    ):
        raise PermissionError("USR1 resume generation inventory differs")


def _execute_training(
    *, authority: Mapping[str, Any], authority_path: Path, action_id: str,
    descriptor: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, str]],
    source_task: Mapping[str, Any], source_row: Mapping[str, Any],
    live_runtime: Mapping[str, Any], scheduler_job_id: str,
    descriptor_ref: Mapping[str, str], monitor: Any,
) -> NonfinalProductionActionResult:
    from .hcwdl_representation_task_runtime import _validate_input_bytes
    from .hcwdl_representation_production import training_adapter

    dependency_refs, dependencies = _dependency_action_results(
        authority=authority, descriptor_ref=descriptor_ref, action_id=action_id,
    )
    target_action = f"target_{str(descriptor['target_identity']).lower()}"
    if target_action not in dependencies:
        raise PermissionError("non-final training lacks its exact target dependency")
    _, target_directory_ref, target_lineage = _target_semantic_from_dependency(
        dependencies[target_action],
        target_identity=str(descriptor["target_identity"]),
    )

    assembly = _source_assembly(source_row)
    inputs = copy.deepcopy(dict(source_row["inputs"]))
    selection_logical = _tagged_input(
        assembly["row_selection"], tag="registered_reference",
        name="training row selection",
    )
    _replace_registered_input(
        inputs, logical=selection_logical,
        reference=artifacts["bounded_row_selection"],
    )
    target_value = assembly.get("target")
    if not isinstance(target_value, Mapping) or set(target_value) != {
        "committed_directory",
    }:
        raise ValueError("non-final source training target binding differs")
    target_logical = _tagged_input(
        target_value["committed_directory"], tag="registered_path",
        name="training target generation",
    )
    _replace_registered_input(
        inputs, logical=target_logical, reference=target_directory_ref,
    )
    planning = _load_reference(authority["planning_spec"], name="planning spec")[0]
    materialized = _validate_input_bytes(
        {**dict(source_row), "inputs": inputs}, spec=planning,
    )

    workspace = _workspace_path(descriptor)
    report_path = workspace / "training_report.json"
    full_loss_path = workspace / "acceptance_real_batch_full_loss.json"
    requires_full_loss = ACTION_REGISTRY[action_id]["kind"] == "two_update"
    if report_path.exists():
        raise FileExistsError("non-final action already has a terminal report")
    if full_loss_path.exists():
        raise FileExistsError("non-final action already has full-loss evidence")
    if action_id != "usr1_resume" and workspace.exists():
        raise FileExistsError(
            "non-final training action workspace already contains state"
        )
    if action_id == "usr1_interrupt" and (workspace / "resume").exists():
        raise FileExistsError("USR1 interrupt workspace already contains resume state")
    if action_id == "usr1_resume":
        interrupt = dependencies.get("usr1_interrupt")
        if interrupt is None:
            raise PermissionError("USR1 resume lacks its interrupt action result")
        receipt, receipt_ref = _load_reference(
            interrupt["semantic_result"], name="USR1 interrupt receipt",
        )
        validate_usr1_delivery_receipt(receipt)
        if receipt.get("scheduler_job_id") == scheduler_job_id:
            raise PermissionError(
                "USR1 resume must run in a fresh Slurm process"
            )
        receipt_workspace = Path(
            str(receipt["resume_generation"]["workspace"])
        ).resolve()
        if receipt_workspace != (workspace / "resume").resolve():
            raise PermissionError("USR1 resume workspace differs from its receipt")
        _validate_usr1_resume_workspace(
            workspace, receipt_reference=receipt_ref,
            execution_id=str(descriptor["execution_id"]),
        )

    source_outputs = tuple(str(value) for value in source_task["registered_outputs"])
    if len(source_outputs) != 4:
        raise ValueError("non-final source training output registry differs")
    outputs = {
        source_outputs[0]: str(report_path),
        source_outputs[1]: str(workspace / "checkpoint_selection.json"),
        source_outputs[2]: str(workspace / "deployable_extraction.json"),
        source_outputs[3]: str(workspace),
    }
    task = _campaign_task(
        source_task, action_id=action_id,
        registered_outputs=source_outputs,
    )
    bounded_selection_sha256 = require_sha256(
        load_json(artifacts["bounded_row_selection"]["path"])["content_hash"],
        name="non-final bounded row selection",
    )
    acceptance_binding = {
        "authority_sha256": authority["content_hash"],
        "action_id": action_id,
        "action_spec_sha256": ACTION_REGISTRY[action_id]["action_spec_sha256"],
        "source_commit": authority["source_commit"],
        "representation_recipe_sha256": authority[
            "representation_recipe_sha256"
        ],
        "train_rows": ACCEPTANCE_TRAIN_ROWS,
        "validation_rows": ACCEPTANCE_VALIDATION_ROWS,
        "replicate_seed": ACCEPTANCE_REPLICATE_SEED,
        "maximum_optimizer_updates": ACCEPTANCE_MAXIMUM_UPDATES,
        **target_lineage,
    }
    assembly.update({
        "train_rows": ACCEPTANCE_TRAIN_ROWS,
        "validation_rows": ACCEPTANCE_VALIDATION_ROWS,
        "execution_id": descriptor["execution_id"],
        "registered_execution_id": descriptor["registered_execution_id"],
        "replicate_seed": ACCEPTANCE_REPLICATE_SEED,
        "mode": "smoke",
        "synthetic_passes": 1,
        "registered_output_row": _output_row(action_id, source_outputs[3]),
        "publication_owner": _owner(authority, action_id=action_id),
        "confirmation_registry": None,
        "acceptance_row_selection_sha256": bounded_selection_sha256,
    })
    if requires_full_loss:
        assembly["acceptance_full_loss_binding"] = acceptance_binding
    resume_lineage = assembly.get("resume_lineage")
    if not isinstance(resume_lineage, Mapping):
        raise ValueError("non-final source resume lineage differs")
    assembly["resume_lineage"] = {
        **dict(resume_lineage),
        "execution": descriptor["registered_execution_id"],
        "representation_recipe": authority["representation_recipe_sha256"],
    }
    runtime_context = {
        **dict(source_row),
        "inputs": materialized,
        "outputs": outputs,
        "parameters": {
            "adapter_contract": source_row["parameters"]["adapter_contract"],
            "task_kind": "train_node",
            "assembly": assembly,
        },
        "_live_worker_runtime": dict(live_runtime),
        "_preemption_requested": monitor.is_requested,
        "_preemption_wait_after_update": (
            1 if action_id == "usr1_interrupt" else None
        ),
        "_preemption_wait": (
            (lambda: _wait_for_real_signal(monitor))
            if action_id == "usr1_interrupt" else None
        ),
    }
    try:
        training_adapter(
            {"content_hash": authority["content_hash"]},
            task, None, runtime_context,
        )
    except RepresentationTrainingInterrupted:
        if action_id != "usr1_interrupt":
            raise
        from .hcwdl_representation_resume import scan_resume_generations

        scan = scan_resume_generations(workspace / "resume")
        if (
            [generation.sequence for generation in scan.valid_generations] != [0]
            or scan.invalid_commits or scan.orphan_files
        ):
            raise PermissionError("USR1 interrupt resume generation differs")
        receipt = build_usr1_delivery_receipt(
            authority=_artifact_reference(authority_path),
            resume_state_directory=workspace / "resume",
            resumed_sequence=0,
            monitor=monitor,
            worker_pid=os.getpid(),
            scheduler_job_id=scheduler_job_id,
            final_report_path=report_path,
        )
        receipt_path = (
            _nonfinal_root_from_workspace(workspace)
            / "usr1" / "interrupt" / "receipt.json"
        )
        write_immutable_json(receipt_path, receipt)
        semantic_outputs = {"primary": _artifact_reference(receipt_path)}
    else:
        if action_id == "usr1_interrupt":
            raise RuntimeError("USR1 interrupt action published a terminal report")
        report = load_json(report_path)
        from .hcwdl_representation_training import (
            validate_representation_training_report,
        )

        validate_representation_training_report(
            report,
            expected_execution_id=str(descriptor["execution_id"]),
            expected_recipe_sha256=authority["representation_recipe_sha256"],
        )
        if (
            report.get("completed_optimizer_updates") != 2
            or report.get("validation", {}).get("rows") != 256
            or report.get("replicate_seed") != 1337
            or report.get("registered_execution_id")
            != descriptor["registered_execution_id"]
            or report.get("target_cache_diagnostics", {}).get(
                "row_selection_sha256"
            ) != bounded_selection_sha256
        ):
            raise PermissionError("non-final terminal training semantics differ")
        semantic_outputs = {"primary": _artifact_reference(report_path)}
        if requires_full_loss:
            semantic_outputs["acceptance_full_loss"] = _validate_full_loss_output(
                full_loss_path, authority=authority, action_id=action_id,
                descriptor=descriptor, report=report,
            )
    return NonfinalProductionActionResult(
        authority_sha256=authority["content_hash"],
        action_id=action_id,
        action_spec_sha256=ACTION_REGISTRY[action_id]["action_spec_sha256"],
        source_task_key=str(descriptor["source_task_key"]),
        source_runtime_row_sha256=str(descriptor["source_runtime_row_sha256"]),
        workspace=workspace,
        semantic_outputs=semantic_outputs,
        dependency_action_results=dependency_refs,
        scheduler_job_id=scheduler_job_id,
    )


def _execute_target_or_training_action(
    *, authority: Mapping[str, Any], authority_path: Path, action_id: str,
    project_dir: str | Path, deterministic_worker: bool,
) -> NonfinalProductionActionResult:
    """Run one genuine target/training action after live runtime validation."""

    if action_id not in ACTION_REGISTRY or action_id == "validation_proxy":
        raise PermissionError("action is not owned by the target/training bridge")
    if os.environ.get("SLURM_ARRAY_TASK_ID") not in {None, ""}:
        raise PermissionError("non-final production actions are scalar only")
    scheduler_job_id = os.environ.get("SLURM_JOB_ID", "")
    if _SLURM_JOB_ID.fullmatch(scheduler_job_id) is None:
        raise PermissionError("non-final production action requires one Slurm job")
    action = ACTION_REGISTRY[action_id]
    expected_deterministic = action["worker_role"] == "deterministic"
    if deterministic_worker is not expected_deterministic:
        raise PermissionError("non-final production worker role differs")

    # Minimal reference loading is required to locate the exact frozen row;
    # it performs no ROOT access.  The live measurement deliberately precedes
    # both static authority validation and the canonical row-selection scan.
    planning, _ = _load_reference(authority["planning_spec"], name="planning spec")
    runtime, _ = _load_reference(authority["runtime_binding"], name="runtime binding")
    task_key, array_index, source_kind = SOURCE_RUNTIME_ROW_BY_ACTION[action_id]
    source_task = _source_task(planning, task_key)
    source_row = resolve_runtime_row(
        runtime, spec=planning, task_key=task_key, array_index=array_index,
    )
    if source_task.get("kind") != source_kind:
        raise PermissionError("non-final source task kind differs")
    from .hcwdl_representation_worker_runtime import validate_live_task_runtime

    live_runtime = validate_live_task_runtime(
        spec=planning,
        binding=runtime,
        task=source_task,
        runtime_row=source_row,
        deterministic_worker=deterministic_worker,
    )
    action_inputs, checked_planning, checked_runtime = _authority_static_context(
        authority, project_dir=project_dir,
    )
    if checked_planning != planning or checked_runtime != runtime:
        raise PermissionError("non-final runtime references changed after measurement")
    # Target/training deep validation may rescan the bounded train/validation
    # population, but it happens only after the exact live runtime gate above.
    authority_hash = validate_nonfinal_acceptance_authority(
        authority, project_dir=project_dir,
    )
    descriptor, descriptor_ref, artifacts = _action_descriptor(
        action_inputs, action_id=action_id,
    )
    if (
        descriptor["source_runtime_row_sha256"]
        != canonical_sha256(dict(source_row))
        or descriptor["source_assembly_sha256"]
        != canonical_sha256(source_row["parameters"]["assembly"])
        or descriptor["action_spec_sha256"] != action["action_spec_sha256"]
        or authority_hash != authority["content_hash"]
    ):
        raise PermissionError("non-final projected source-row lineage differs")

    from .hcwdl_representation_task_runtime import RepresentationPreemptionMonitor

    monitor = RepresentationPreemptionMonitor()
    monitor.install()
    try:
        if action["kind"] == "target_prepare":
            return _execute_target(
                authority=authority, action_id=action_id,
                descriptor=descriptor, artifacts=artifacts,
                source_task=source_task, source_row=source_row,
                live_runtime=live_runtime, scheduler_job_id=scheduler_job_id,
                descriptor_ref=descriptor_ref,
            )
        return _execute_training(
            authority=authority, authority_path=authority_path,
            action_id=action_id, descriptor=descriptor, artifacts=artifacts,
            source_task=source_task, source_row=source_row,
            live_runtime=live_runtime, scheduler_job_id=scheduler_job_id,
            descriptor_ref=descriptor_ref, monitor=monitor,
        )
    finally:
        monitor.restore()


def execute_nonfinal_production_action(
    *, authority: Mapping[str, Any], authority_path: str | Path,
    action_id: str, project_dir: str | Path,
    deterministic_worker: bool,
) -> NonfinalProductionActionResult:
    """Closed generic registry for all ten non-final production actions."""

    path = Path(authority_path).resolve()
    if _artifact_reference(path) != {
        "path": str(path), "sha256": sha256_file(path),
    } or load_json(path) != dict(authority):
        raise PermissionError("non-final runtime authority path/value differs")
    action_inputs_reference = authority.get("action_inputs")
    if not isinstance(action_inputs_reference, Mapping):
        raise PermissionError("non-final runtime action-input route is absent")
    action_inputs_path = Path(str(action_inputs_reference.get("path", ""))).resolve()
    if (
        action_inputs_path.name != "action_inputs.json"
        or path != action_inputs_path.parent / "authority.json"
    ):
        raise PermissionError("non-final runtime authority route differs")
    if action_id not in ACTION_REGISTRY:
        raise PermissionError("non-final production action is not registered")
    if os.environ.get("SLURM_ARRAY_TASK_ID") not in {None, ""}:
        raise PermissionError("non-final production actions are scalar only")
    scheduler_job_id = os.environ.get("SLURM_JOB_ID", "")
    if _SLURM_JOB_ID.fullmatch(scheduler_job_id) is None:
        raise PermissionError("non-final production action requires one Slurm job")
    expected_deterministic = (
        ACTION_REGISTRY[action_id]["worker_role"] == "deterministic"
    )
    if deterministic_worker is not expected_deterministic:
        raise PermissionError("non-final production worker role differs")
    if action_id == "validation_proxy":
        from .hcwdl_representation_validation_runtime import (
            execute_validation_proxy_production_action,
        )

        result = execute_validation_proxy_production_action(
            authority=authority,
            authority_path=path,
            project_dir=project_dir,
            deterministic_worker=deterministic_worker,
        )
        action_inputs, _, _ = _authority_static_context(
            authority, project_dir=project_dir,
        )
        descriptor, _, _ = _action_descriptor(
            action_inputs, action_id=action_id,
        )
        semantic_ref = _artifact_reference(result.semantic_result_path)
        if load_json(result.semantic_result_path) != dict(result.semantic_result):
            raise PermissionError("validation semantic path/value differs")
        return NonfinalProductionActionResult(
            authority_sha256=str(authority["content_hash"]),
            action_id=action_id,
            action_spec_sha256=ACTION_REGISTRY[action_id][
                "action_spec_sha256"
            ],
            source_task_key=str(result.source_task_key),
            source_runtime_row_sha256=str(
                descriptor["source_runtime_row_sha256"]
            ),
            workspace=Path(result.workspace).resolve(),
            semantic_outputs={"primary": semantic_ref},
            dependency_action_results=dict(result.dependencies),
            scheduler_job_id=scheduler_job_id,
        )
    return _execute_target_or_training_action(
        authority=authority,
        authority_path=path,
        action_id=action_id,
        project_dir=project_dir,
        deterministic_worker=deterministic_worker,
    )


__all__ = [
    "NonfinalProductionActionResult",
    "execute_nonfinal_production_action",
    "validate_nonfinal_acceptance_authority_static",
]

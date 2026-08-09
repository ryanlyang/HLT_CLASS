"""Fail-closed HCWDL-RKD import and campaign-lock construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, require_sha256, validate_content_hash,
    with_content_hash,
)
from hlt_classification.models.hcwdl_surfaces import (
    validate_architecture_attestation,
)

from .hcwdl_parent_loss import validate_parent_loss_attestation


LEGACY_PARENT_IMPORT_CONTRACT: Final = "HCWDL_REPRESENTATION_PARENT_IMPORT/v1"
PARENT_IMPORT_CONTRACT: Final = "HCWDL_REPRESENTATION_PARENT_IMPORT/v3"
"""Authoritative file-backed parent import.

Version 1 accepted caller-supplied lineage hashes and Boolean statements that
the original contracts had been validated.  It remains useful only for local
fixtures.  Version 2 records the hashes of every file that establishes the
parent campaign, assignment, recipe, endpoint-continuation, and finalist
authority and can therefore be reconstructed from registered files.  Version
3 narrows executable import to the exact non-smoke v8 60-pass parent prefix
and authenticates the full execution config, 60 validation records, selected
checkpoint, and completed final checkpoint of every engine report.
"""

PARENT_AUTHORITY_FILE_KEYS: Final = frozenset({
    "campaign_spec", "source_manifest", "split_manifest", "row_selection",
    "matcher_resources", "train_assignment_manifest",
    "validation_assignment_manifest", "train_recomputation_audit",
    "validation_recomputation_audit", "assignment_lock", "recipe",
    "recipe_lock", "cache_miniature", "diagnostic_authority",
    "qualification_report", "endpoint_qualification_lock",
    "screen_aggregate", "confirmation_registry_lock",
    "confirmation_aggregate", "finalist_lock", "architecture_attestation",
    "parent_loss_attestation",
})
PARENT_AUTHORITY_PARENT_KEYS: Final = frozenset({
    "parent_campaign_spec", "architecture_attestation",
    "parent_loss_attestation", "source_manifest", "split_manifest",
    "row_selection", "matcher_resources", "train_assignment_manifest",
    "validation_assignment_manifest", "train_recomputation_audit",
    "validation_recomputation_audit", "assignment_lock", "parent_recipe",
    "recipe_lock", "cache_miniature", "diagnostic_authority",
    "qualification_report", "endpoint_qualification_lock",
    "screen_aggregate", "confirmation_registry_lock",
    "confirmation_aggregate", "finalist_lock", "parent_graph",
    "qualifier_T0", "qualifier_TFS", "qualifier_THC", "qualifier_TSOFT",
    "qualifier_TSHELL", "qualifier_TOFF",
})


IMPORTED_TEACHERS: Final = (
    "D0c", "D25c", "D50c", "D75c",
    "D0w", "D25w", "D50w", "D75w",
    "D100", "TOFF",
)
IMPORTED_LOGIT_CONTROLS: Final = (
    "M0", *(f"M{rung}{track}" for track in ("c", "w") for rung in range(1, 7)),
)
PARENT_IMPORT_LINEAGE_KEYS: Final = frozenset({
    "source_manifest", "split_manifest", "row_selection",
    "train_assignment_manifest", "validation_assignment_manifest",
    "assignment_lock", "parent_recipe", "endpoint_qualification_lock",
    "parent_graph",
})
PARENT_IMPORT_VALIDATION_KEYS: Final = frozenset({
    *PARENT_IMPORT_LINEAGE_KEYS,
    "parent_campaign_spec", "teachers", "logit_controls",
})


def _registered_json_file(path: str | Path, *, name: str) -> tuple[Path, dict[str, Any]]:
    """Open one immutable registered JSON file without accepting a symlink."""

    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise FileNotFoundError(f"parent authority {name} is absent or a symlink")
    resolved = candidate.resolve()
    value = load_json(resolved)
    if not isinstance(value, Mapping):
        raise TypeError(f"parent authority {name} is not a JSON object")
    return resolved, dict(value)


def _versioned_digest(value: Mapping[str, Any], *, name: str) -> str:
    contract = value.get("contract")
    schema_version = value.get("schema_version")
    if not isinstance(contract, str) or not isinstance(schema_version, int):
        raise ValueError(f"parent authority {name} lacks a versioned contract")
    return validate_content_hash(
        value, expected_contract=contract, expected_schema_version=schema_version,
    )


def _authority_paths(value: Mapping[str, str | Path]) -> dict[str, Path]:
    if not isinstance(value, Mapping) or set(value) != PARENT_AUTHORITY_FILE_KEYS:
        raise ValueError("parent authority file registry differs")
    return {name: Path(value[name]) for name in sorted(value)}


def _qualifier_paths(value: Mapping[str, str | Path]) -> dict[str, Path]:
    from .hcwdl_qualification import QUALIFIERS

    if not isinstance(value, Mapping) or set(value) != set(QUALIFIERS):
        raise ValueError("parent qualifier report registry differs")
    return {name: Path(value[name]) for name in QUALIFIERS}


def _validate_recomputation_audit(
    value: Mapping[str, Any], *, manifest_sha256: str, name: str,
) -> str:
    digest = validate_content_hash(
        value, expected_contract="HIGHCOV_ASSIGNMENT_RECOMPUTATION_AUDIT/v1",
        expected_schema_version=1,
    )
    if (
        value.get("manifest_sha256") != manifest_sha256
        or isinstance(value.get("sample_size"), bool)
        or not isinstance(value.get("sample_size"), int)
        or int(value["sample_size"]) <= 0
        or value.get("seed") != 1337
        or value.get("exact_indices") is not True
        or value.get("exact_confidence_u16") is not True
    ):
        raise ValueError(f"parent {name} recomputation audit differs")
    require_sha256(
        value.get("sample_indices_sha256"), name=f"parent {name} sample indices",
    )
    return digest


def _same_artifact(actual: Mapping[str, Any], expected: Mapping[str, Any], *, name: str) -> None:
    if dict(actual) != dict(expected):
        raise ValueError(f"parent {name} is not the canonical derived artifact")


def _validate_lock_chain(
    *, campaign_sha256: str, assignment_lock: Mapping[str, Any],
    recipe_lock: Mapping[str, Any], endpoint_lock: Mapping[str, Any],
    confirmation_lock: Mapping[str, Any], finalist_lock: Mapping[str, Any],
) -> None:
    """Require the complete parent predecessor chain, not a valid leaf alone."""

    from .hcwdl_locks import validate_lock

    rows = (
        ("assignment", assignment_lock, None),
        ("recipe", recipe_lock, assignment_lock),
        ("shell_endpoint_qualification", endpoint_lock, recipe_lock),
        ("confirmation_registry", confirmation_lock, endpoint_lock),
        ("finalist", finalist_lock, confirmation_lock),
    )
    for level, artifact, parent in rows:
        digest = validate_lock(artifact, expected_level=level)
        if artifact.get("campaign_spec_sha256") != campaign_sha256:
            raise ValueError(f"parent {level} lock belongs to another campaign")
        expected_parent = None if parent is None else parent["content_hash"]
        if artifact.get("parent_lock_sha256") != expected_parent:
            raise ValueError(f"parent {level} lock predecessor differs")
        if digest != artifact.get("content_hash"):
            raise ValueError(f"parent {level} lock content hash differs")


def _confirmation_report_key(index: int, row: Mapping[str, Any]) -> str:
    return f"{index:03d}:{row['node_id']}:{int(row['seed'])}"


def _require_exact_finalists(
    actual: object, expected: Sequence[Mapping[str, Any]],
) -> None:
    if not isinstance(actual, list) or actual != [dict(row) for row in expected]:
        raise ValueError("parent finalist registry differs from canonical reports")


def _require_exact_confirmation_registry(
    actual: object, *, screen: Mapping[str, Any],
    include_label_only_warm_continuation: bool,
) -> list[dict[str, Any]]:
    from .hcwdl_reporting import build_confirmation_registry

    expected = build_confirmation_registry(
        screen, seeds=(11, 22, 33, 44, 55),
        include_label_only_warm_continuation=(
            include_label_only_warm_continuation
        ),
    )
    if actual != expected:
        raise ValueError("parent confirmation registry differs from screen selection")
    return expected


def _validate_primary_engine_lineage(
    report: Mapping[str, Any], *, node_id: str, replicate_seed: int,
    split_sha256: str, source_sha256: str, assignment_lock_sha256: str,
    qualification_lock_sha256: str, recipe_sha256: str,
    screening_reports: Mapping[str, Mapping[str, Any]],
) -> None:
    """Bind one primary engine report to the executable parent graph."""

    from .hcwdl_ladder import GRAPH_SHA256, NODE_REGISTRY
    from .training import derive_seed

    node = NODE_REGISTRY.get(node_id)
    if node is None:
        raise ValueError(f"parent primary report node differs: {node_id}")
    config = report.get("config")
    scientific = report.get("scientific_config")
    parents = report.get("parents")
    expected_parents = {
        "split_manifest_sha256": split_sha256,
        "source_snapshot_sha256": source_sha256,
        "assignment_lock_sha256": assignment_lock_sha256,
        "qualification_lock_sha256": qualification_lock_sha256,
        "recipe": recipe_sha256,
    }
    for teacher in node.teachers:
        teacher_report = screening_reports.get(teacher.node_id)
        if teacher_report is None:
            raise ValueError(
                f"parent graph teacher report is absent: {node_id}/{teacher.node_id}"
            )
        expected_parents[
            f"teacher_{teacher.role}_report_sha256"
        ] = require_sha256(
            teacher_report.get("content_hash"),
            name=f"parent graph teacher {teacher.node_id}",
        )
    if node.initialization_parent is not None:
        warm_parent = screening_reports.get(node.initialization_parent)
        if warm_parent is None:
            raise ValueError(
                f"parent warm-initialization report is absent: {node_id}"
            )
        expected_parents["warm_parent_report_sha256"] = require_sha256(
            warm_parent.get("content_hash"),
            name=f"parent warm initialization {node.initialization_parent}",
        )
    if (
        report.get("experiment_id") != node_id
        or report.get("complete") is not True
        or not isinstance(config, Mapping)
        or config.get("master_seed")
        != derive_seed(int(replicate_seed), f"hcwdl/{node_id}")
        or config.get("selection_policy") != "hcwdl_macro_auc"
        or not isinstance(scientific, Mapping)
        or scientific.get("campaign") != "HCWDL"
        or scientific.get("graph_sha256") != GRAPH_SHA256
        or scientific.get("recipe_sha256") != recipe_sha256
        or canonical_sha256(scientific.get("node"))
        != canonical_sha256(node.payload())
        or parents != expected_parents
    ):
        raise ValueError(f"parent primary report lineage differs: {node_id}")
    require_sha256(
        report.get("selected_checkpoint_sha256"),
        name=f"parent primary {node_id} checkpoint",
    )


def _validate_control_engine_lineage(
    report: Mapping[str, Any], *, control_id: str, replicate_seed: int,
    split_sha256: str, source_sha256: str, assignment_lock_sha256: str,
    qualification_lock_sha256: str, recipe_sha256: str,
    screening_reports: Mapping[str, Mapping[str, Any]],
) -> None:
    """Bind one null-control report to its declared screening teacher."""

    from .training import derive_seed

    teacher_nodes = {
        "NULL_M1_SELF_KD": "M0",
        "NULL_M6_PREDECESSOR_ONLY": "M5c",
        "NULL_WARM_LABEL_ONLY": "M5w",
    }
    teacher_node = teacher_nodes.get(control_id)
    teacher_report = None if teacher_node is None else screening_reports.get(teacher_node)
    if teacher_report is None:
        raise ValueError(f"parent control teacher report is absent: {control_id}")
    config = report.get("config")
    scientific = report.get("scientific_config")
    expected_parents = {
        "split_manifest_sha256": split_sha256,
        "source_snapshot_sha256": source_sha256,
        "assignment_lock_sha256": assignment_lock_sha256,
        "qualification_lock_sha256": qualification_lock_sha256,
        "recipe_sha256": recipe_sha256,
        "teacher_report_sha256": require_sha256(
            teacher_report.get("content_hash"),
            name=f"parent control teacher {teacher_node}",
        ),
    }
    expected_initialization = (
        "selected_M5w_weights_optimizer_reset"
        if control_id == "NULL_WARM_LABEL_ONLY" else "fresh"
    )
    if (
        report.get("experiment_id") != control_id
        or report.get("complete") is not True
        or not isinstance(config, Mapping)
        or config.get("master_seed")
        != derive_seed(int(replicate_seed), f"hcwdl/control/{control_id}")
        or config.get("selection_policy") != "hcwdl_macro_auc"
        or not isinstance(scientific, Mapping)
        or scientific.get("campaign") != "HCWDL"
        or scientific.get("control_id") != control_id
        or scientific.get("initialization") != expected_initialization
        or report.get("parents") != expected_parents
    ):
        raise ValueError(f"parent control report lineage differs: {control_id}")
    require_sha256(
        report.get("selected_checkpoint_sha256"),
        name=f"parent control {control_id} checkpoint",
    )


def _import_row(row: Mapping[str, Any], *, node_id: str, teacher: bool) -> dict[str, Any]:
    required = {
        "node_id", "domain", "track", "report_path", "report_sha256",
        "checkpoint_path", "checkpoint_sha256", "checkpoint_byte_sha256",
    }
    if set(row) != required or row.get("node_id") != node_id:
        raise ValueError(f"parent import row differs for {node_id}")
    domain = str(row["domain"])
    track = str(row["track"])
    if node_id.endswith("c") and track != "cold":
        raise ValueError("cold parent import track differs")
    if node_id.endswith("w") and track != "warm":
        raise ValueError("warm parent import track differs")
    if node_id in {"D100", "TOFF", "M0"} and track != "shared":
        raise ValueError("shared parent import track differs")
    expected_domain = "hlt"
    if teacher and node_id == "TOFF":
        expected_domain = "native_offline"
    elif teacher and node_id == "D100":
        expected_domain = "d100"
    elif teacher and not node_id.startswith("D0"):
        expected_domain = f"d{node_id[1:].rstrip('cw').lower()}"
    if domain != expected_domain:
        kind = "teacher" if teacher else "logit-control"
        raise ValueError(f"{kind} parent import domain differs")
    if not teacher and domain != "hlt":
        raise ValueError("logit-control parent import domain differs")
    normalized = dict(row)
    for key in ("report_sha256", "checkpoint_sha256", "checkpoint_byte_sha256"):
        normalized[key] = require_sha256(row[key], name=f"{node_id} {key}")
    if normalized["checkpoint_sha256"] != normalized["checkpoint_byte_sha256"]:
        raise ValueError(f"parent import selected-checkpoint byte proof differs: {node_id}")
    if not str(row["report_path"]) or not str(row["checkpoint_path"]):
        raise ValueError("parent import path is empty")
    return normalized


def _bind_parent_evidence(
    *,
    architecture_attestation: Mapping[str, Any],
    parent_loss_attestation: Mapping[str, Any],
    teachers: Mapping[str, Mapping[str, Any]],
    logit_controls: Mapping[str, Mapping[str, Any]],
) -> None:
    """Prove both attestations describe the exact checkpoints being imported."""

    imported = {**teachers, **logit_controls}
    expected_nodes = set(IMPORTED_TEACHERS) | set(IMPORTED_LOGIT_CONTROLS)
    if set(imported) != expected_nodes:
        raise ValueError("parent import evidence registry is incomplete")

    architecture_rows = architecture_attestation.get("checkpoint_audits")
    if not isinstance(architecture_rows, list):
        raise ValueError("architecture attestation checkpoint registry differs")
    architecture_by_node = {
        row.get("node_id"): row for row in architecture_rows
        if isinstance(row, Mapping)
    }
    if (
        len(architecture_by_node) != len(architecture_rows)
        or set(architecture_by_node) != expected_nodes
    ):
        raise ValueError("architecture attestation does not cover exact imports")

    loss_rows = parent_loss_attestation.get("parent_artifacts")
    if not isinstance(loss_rows, list):
        raise ValueError("parent-loss attestation artifact registry differs")
    loss_by_node = {
        row.get("node_id"): row for row in loss_rows
        if isinstance(row, Mapping)
    }
    if len(loss_by_node) != len(loss_rows) or set(loss_by_node) != expected_nodes:
        raise ValueError("parent-loss attestation does not cover exact imports")

    for node_id, imported_row in imported.items():
        architecture = architecture_by_node[node_id]
        if architecture.get("actual_file_evidence") is not True:
            raise ValueError(f"architecture lacks actual-file evidence for {node_id}")
        report_path = Path(str(imported_row["report_path"])).resolve().as_posix()
        checkpoint_path = Path(str(imported_row["checkpoint_path"])).resolve().as_posix()
        if (
            architecture.get("report_path") != report_path
            or architecture.get("report_sha256") != imported_row.get("report_sha256")
            or architecture.get("checkpoint_path") != checkpoint_path
            or architecture.get("checkpoint_sha256")
            != imported_row.get("checkpoint_byte_sha256")
            or imported_row.get("checkpoint_sha256")
            != imported_row.get("checkpoint_byte_sha256")
        ):
            raise ValueError(f"architecture checkpoint lineage differs for {node_id}")
        loss = loss_by_node[node_id]
        if (
            loss.get("training_report_sha256") != imported_row.get("report_sha256")
            or loss.get("checkpoint_sha256") != imported_row.get("checkpoint_sha256")
        ):
            raise ValueError(f"parent-loss checkpoint/report lineage differs for {node_id}")


def _validate_parent_loss_campaign_source(
    parent_loss_attestation: Mapping[str, Any],
    *, campaign: Mapping[str, Any], campaign_sha256: str,
) -> None:
    """Bind after-the-fact source evidence to the reopened v8 parent prefix."""

    snapshot = parent_loss_attestation.get("parent_source_snapshot")
    source_commit = campaign.get("source_commit")
    expected_authority = {
        "parent_campaign_contract": campaign.get("contract"),
        "parent_campaign_mode": campaign.get("mode"),
        "parent_execution_scope": campaign.get("execution_scope"),
        "parent_endpoint_continuation": campaign.get("endpoint_continuation"),
        "parent_training_passes": campaign.get("training_passes"),
        "parent_validation_every_passes": campaign.get(
            "validation_every_passes"
        ),
        "parent_train_rows": campaign.get("role_counts", {}).get("train"),
        "parent_terminal_task_id": campaign.get("terminal_task_id"),
        "parent_execution_lock_authorized": campaign.get(
            "execution_lock_authorized"
        ),
        "parent_final_test_access_authorized": campaign.get(
            "final_test_access_authorized"
        ),
        "parent_registered_final_test_tasks": campaign.get(
            "registered_final_test_tasks"
        ),
    }
    if (
        parent_loss_attestation.get("parent_campaign_spec_sha256")
        != campaign_sha256
        or parent_loss_attestation.get("parent_source_commit") != source_commit
        or parent_loss_attestation.get("parent_campaign_authority")
        != expected_authority
        or not isinstance(snapshot, Mapping)
        or snapshot.get("git_commit") != source_commit
    ):
        raise PermissionError(
            "parent-loss producer source differs from parent campaign authority"
        )


def derive_parent_import_rows_from_architecture(
    architecture_attestation: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Derive the closed import registries from authenticated checkpoint audits."""

    raw_audits = architecture_attestation.get("checkpoint_audits")
    if not isinstance(raw_audits, list):
        raise ValueError("architecture attestation checkpoint registry differs")
    audits = {
        str(row.get("node_id")): row
        for row in raw_audits if isinstance(row, Mapping)
    }
    expected = set(IMPORTED_TEACHERS) | set(IMPORTED_LOGIT_CONTROLS)
    if len(audits) != len(raw_audits) or set(audits) != expected:
        raise ValueError("architecture attestation does not cover exact imports")

    def derived(node_id: str, *, teacher: bool) -> dict[str, Any]:
        audit = audits[node_id]
        if audit.get("actual_file_evidence") is not True:
            raise ValueError(f"architecture lacks actual-file evidence for {node_id}")
        if node_id.endswith("c"):
            track = "cold"
        elif node_id.endswith("w"):
            track = "warm"
        else:
            track = "shared"
        if not teacher:
            domain = "hlt"
        elif node_id == "TOFF":
            domain = "native_offline"
        elif node_id.startswith("D0"):
            domain = "hlt"
        else:
            domain = f"d{node_id[1:].rstrip('cw').lower()}"
        row = {
            "node_id": node_id,
            "domain": domain,
            "track": track,
            "report_path": audit.get("report_path"),
            "report_sha256": audit.get("report_sha256"),
            "checkpoint_path": audit.get("checkpoint_path"),
            "checkpoint_sha256": audit.get("checkpoint_sha256"),
            "checkpoint_byte_sha256": audit.get("checkpoint_sha256"),
        }
        return _import_row(row, node_id=node_id, teacher=teacher)

    return (
        {
            node_id: derived(node_id, teacher=True)
            for node_id in IMPORTED_TEACHERS
        },
        {
            node_id: derived(node_id, teacher=False)
            for node_id in IMPORTED_LOGIT_CONTROLS
        },
    )


def _validated_parent_authority(
    *, authority_files: Mapping[str, str | Path],
    qualifier_report_paths: Mapping[str, str | Path],
    confirmation_report_paths: Mapping[str, str | Path] | None = None,
    teachers: Mapping[str, Mapping[str, Any]] | None = None,
    logit_controls: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Open and cross-bind the complete current HCWDL parent authority."""

    paths = _authority_paths(authority_files)
    qualifier_paths = _qualifier_paths(qualifier_report_paths)
    artifacts = {
        name: _registered_json_file(path, name=name)[1]
        for name, path in paths.items()
    }
    confirmation_report_paths = dict(confirmation_report_paths or {})

    from .audit import SOURCE_MANIFEST_CONTRACT, SOURCE_MANIFEST_VERSION
    from .hcwdl_ladder import GRAPH_SHA256, validate_ladder_graph
    from .hcwdl_locks import (
        create_assignment_lock, create_confirmation_registry_lock,
        create_finalist_lock, create_recipe_lock,
        create_shell_endpoint_qualification_lock,
    )
    from .hcwdl_qualification import (
        QUALIFICATION_CONTRACT, QUALIFIERS,
        validate_diagnostic_acknowledgement, validate_diagnostic_waiver,
        validate_endpoint_diagnostics, validate_qualification_report,
    )
    from .hcwdl_recipe import (
        CLASS_WEIGHT_POLICY, PRIMARY_RECIPE_PROFILE, RECIPE_CONTRACT,
        validate_recipe, validate_recipe_class_weight_lineage,
    )
    from .hcwdl_reporting import (
        CONFIRMATION_REPORT_CONTRACT, SCREEN_REPORT_CONTRACT,
        build_screen_aggregate,
    )
    from .hcwdl_training import (
        confirmation_control_training_config, node_training_config,
        qualifier_training_config, select_checkpoint,
        validate_hcwdl_full_parent_engine_report,
        validate_hcwdl_full_parent_wrapper_report,
        validate_hcwdl_parent_prefix_campaign,
    )
    from .highcov_cache import validate_assignment_manifest
    from .highcov_resources import RESOURCE_CONTRACT, resource_validation_report
    from .selective_assignment import validate_row_selection
    from .splits import source_file_record_from_manifest_row, validate_split_manifest

    campaign = artifacts["campaign_spec"]
    campaign_sha256 = validate_hcwdl_parent_prefix_campaign(
        campaign, executable=True,
    )
    if (teachers is None) != (logit_controls is None):
        raise ValueError("parent import row registries must be supplied together")
    if teachers is None:
        teachers, logit_controls = derive_parent_import_rows_from_architecture(
            artifacts["architecture_attestation"],
        )
    assert logit_controls is not None

    source = artifacts["source_manifest"]
    source_sha256 = validate_content_hash(
        source, expected_contract=SOURCE_MANIFEST_CONTRACT,
        expected_schema_version=SOURCE_MANIFEST_VERSION,
    )
    if source_sha256 != campaign.get("source_manifest_sha256"):
        raise ValueError("parent source manifest differs from campaign")
    source_rows = source.get("files")
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("parent source manifest inventory is empty")

    split = artifacts["split_manifest"]
    split_sha256 = validate_split_manifest(
        split, source_manifest_sha256=source_sha256,
        expected_inventory=(
            source_file_record_from_manifest_row(row) for row in source_rows
        ),
    )
    if split_sha256 != campaign.get("split_manifest_sha256"):
        raise ValueError("parent split manifest differs from campaign")

    selection = artifacts["row_selection"]
    selection_sha256 = validate_row_selection(
        selection, split_manifest_sha256=split_sha256,
    )
    roles = selection.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != {"train", "validation"}:
        raise ValueError("parent import row selection must contain train and validation only")
    for role in ("train", "validation"):
        count = campaign["role_counts"][role]
        if count is not None and roles[role].get("rows") != int(count):
            raise ValueError(f"parent {role} selection count differs from campaign")

    matcher_resources = artifacts["matcher_resources"]
    matcher_sha256 = validate_content_hash(
        matcher_resources, expected_contract=RESOURCE_CONTRACT,
        expected_schema_version=1,
    )
    if dict(matcher_resources) != resource_validation_report():
        raise ValueError("parent matcher resources are not canonical")
    assignment_parents = {
        "split_manifest_sha256": split_sha256,
        "row_selection_sha256": selection_sha256,
        "matcher_resources_sha256": matcher_sha256,
    }
    train_manifest = validate_assignment_manifest(
        paths["train_assignment_manifest"], expected_role="train",
        expected_mapped_jets=int(roles["train"]["rows"]),
        expected_parents=assignment_parents, require_sub10pct_dustbins=True,
    )
    validation_manifest = validate_assignment_manifest(
        paths["validation_assignment_manifest"], expected_role="validation",
        expected_mapped_jets=int(roles["validation"]["rows"]),
        expected_parents=assignment_parents, require_sub10pct_dustbins=True,
    )
    train_manifest_sha256 = require_sha256(
        train_manifest.get("content_hash"), name="parent train assignment manifest",
    )
    validation_manifest_sha256 = require_sha256(
        validation_manifest.get("content_hash"),
        name="parent validation assignment manifest",
    )
    train_audit_sha256 = _validate_recomputation_audit(
        artifacts["train_recomputation_audit"],
        manifest_sha256=train_manifest_sha256, name="train",
    )
    validation_audit_sha256 = _validate_recomputation_audit(
        artifacts["validation_recomputation_audit"],
        manifest_sha256=validation_manifest_sha256, name="validation",
    )

    assignment_lock = artifacts["assignment_lock"]
    expected_assignment_lock = create_assignment_lock(
        campaign_spec_sha256=campaign_sha256,
        train_manifest_path=paths["train_assignment_manifest"],
        validation_manifest_path=paths["validation_assignment_manifest"],
        expected_train_rows=int(roles["train"]["rows"]),
        expected_validation_rows=int(roles["validation"]["rows"]),
        expected_parents=assignment_parents,
        train_recomputation_sha256=train_audit_sha256,
        validation_recomputation_sha256=validation_audit_sha256,
        matcher_resources_sha256=matcher_sha256,
    )
    _same_artifact(assignment_lock, expected_assignment_lock, name="assignment lock")

    recipe = artifacts["recipe"]
    if recipe.get("contract") != RECIPE_CONTRACT:
        raise ValueError("HCWDL-RKD parent must use HCWDL_RECIPE/v4")
    recipe_sha256 = validate_recipe(
        recipe, require_authorized=True, expected_profile=PRIMARY_RECIPE_PROFILE,
    )
    weighting = recipe.get("class_weighting")
    if (
        not isinstance(weighting, Mapping)
        or weighting.get("policy") != CLASS_WEIGHT_POLICY
        or not np.array_equal(
            np.asarray(recipe.get("class_weights"), np.float32),
            np.ones(15, np.float32),
        )
    ):
        raise ValueError("parent HCWDL_RECIPE/v4 is not the exact unweighted primary")
    validate_recipe_class_weight_lineage(recipe, selection)
    if campaign.get("recipe_sha256") != recipe_sha256:
        raise ValueError("parent campaign binds a different recipe")
    if campaign.get("graph_sha256") != GRAPH_SHA256 or validate_ladder_graph() != GRAPH_SHA256:
        raise ValueError("parent campaign graph differs from HCWDL_GRAPH/v1")

    recipe_lock = artifacts["recipe_lock"]
    expected_recipe_lock = create_recipe_lock(
        campaign_spec_sha256=campaign_sha256,
        assignment_lock=assignment_lock, recipe_sha256=recipe_sha256,
        evidence_hashes=recipe["evidence"],
    )
    _same_artifact(recipe_lock, expected_recipe_lock, name="recipe lock")

    qualifier_reports: dict[str, Mapping[str, Any]] = {}
    qualifier_hashes: dict[str, str] = {}
    train_rows = int(roles["train"]["rows"])
    for name in QUALIFIERS:
        report_path, report = _registered_json_file(
            qualifier_paths[name], name=f"qualifier {name}",
        )
        qualifier_reports[name] = report
        qualifier_hashes[name] = validate_hcwdl_full_parent_engine_report(
            report, train_rows=train_rows, recipe=recipe,
            expected_experiment_id=name,
            expected_exact_config=asdict(qualifier_training_config(
                name, recipe, train_rows=train_rows, replicate_seed=1337,
            )),
            report_path=report_path,
        )
        qualifier_scientific = report.get("scientific_config")
        if (
            report.get("parents") != {
                "split_manifest_sha256": split_sha256,
                "source_snapshot_sha256": source_sha256,
                "assignment_lock_sha256": assignment_lock["content_hash"],
                "recipe_sha256": recipe_sha256,
            }
            or not isinstance(qualifier_scientific, Mapping)
            or qualifier_scientific.get("qualification_id") != name
            or qualifier_scientific.get("fixed_primary_repair")
            != "HIGHCOV_SHELL_EXACT/v1"
            or qualifier_scientific.get("selection_performed") is not False
            or qualifier_scientific.get("qualification_rng_policy")
            != "shared_trajectory_across_views_v1"
        ):
            raise ValueError(f"parent qualifier lineage differs: {name}")
    cache_miniature = artifacts["cache_miniature"]
    cache_sha256 = validate_content_hash(
        cache_miniature, expected_contract="HCWDL_CACHE_MINIATURE/v1",
        expected_schema_version=1,
    )
    endpoint_invariants = cache_miniature.get("endpoint_invariants")
    if not isinstance(endpoint_invariants, Mapping):
        raise ValueError("parent cache miniature lacks endpoint invariants")
    validate_endpoint_diagnostics(qualifier_reports, endpoint_invariants)

    diagnostic = artifacts["diagnostic_authority"]
    continuation = campaign.get("endpoint_continuation")
    diagnostic_sha256 = validate_diagnostic_waiver(
        diagnostic, campaign_spec_sha256=campaign_sha256,
        assignment_manifest_sha256=validation_manifest_sha256,
        recipe_sha256=recipe_sha256, cache_miniature_sha256=cache_sha256,
        qualifier_report_sha256=qualifier_hashes,
        submission_authorization_sha256=campaign["submission_authorization_sha256"],
    )

    qualification = artifacts["qualification_report"]
    if qualification.get("contract") != QUALIFICATION_CONTRACT:
        raise ValueError("HCWDL-RKD parent requires endpoint qualification v2")
    qualification_sha256 = validate_qualification_report(qualification)
    if any((
        qualification.get("campaign_spec_sha256") != campaign_sha256,
        qualification.get("assignment_manifest_sha256") != validation_manifest_sha256,
        qualification.get("recipe_sha256") != recipe_sha256,
        qualification.get("endpoint_continuation") != continuation,
        qualification.get("diagnostic_ack_sha256") != diagnostic_sha256,
        qualification.get("reports") != dict(sorted(qualifier_hashes.items())),
        qualification.get("endpoint_invariants") != dict(endpoint_invariants),
    )):
        raise ValueError("parent endpoint qualification lineage differs")
    endpoint_lock = artifacts["endpoint_qualification_lock"]
    expected_endpoint_lock = create_shell_endpoint_qualification_lock(
        campaign_spec_sha256=campaign_sha256, recipe_lock=recipe_lock,
        qualification_report_sha256=qualification_sha256,
        assignment_manifest_sha256=validation_manifest_sha256,
        endpoint_invariants_passed=True,
    )
    _same_artifact(
        endpoint_lock, expected_endpoint_lock, name="endpoint qualification lock",
    )

    imported_rows = {**teachers, **logit_controls}
    expected_nodes = set(IMPORTED_TEACHERS) | set(IMPORTED_LOGIT_CONTROLS)
    if set(imported_rows) != expected_nodes:
        raise ValueError("parent screen report registry is incomplete")
    engine_reports: list[Mapping[str, Any]] = []
    wrapper_reports: list[Mapping[str, Any]] = []
    engine_by_node: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for node_id in sorted(expected_nodes):
        row = _import_row(
            imported_rows[node_id], node_id=node_id,
            teacher=node_id in set(IMPORTED_TEACHERS),
        )
        wrapper_path, wrapper = _registered_json_file(
            row["report_path"], name=f"screen wrapper {node_id}",
        )
        full_evidence = validate_hcwdl_full_parent_wrapper_report(
            wrapper, training_report_path=wrapper_path,
            train_rows=train_rows, recipe=recipe,
            expected_node_id=node_id, expected_replicate_seed=1337,
        )
        wrapper_sha256 = full_evidence["wrapper_sha256"]
        if (
            wrapper_sha256 != row["report_sha256"]
            or wrapper.get("node_id") != node_id
            or wrapper.get("complete") is not True
        ):
            raise ValueError(f"parent screen wrapper differs for {node_id}")
        engine_path = Path(full_evidence["engine_path"])
        engine = full_evidence["engine"]
        engine_sha256 = full_evidence["engine_sha256"]
        if (
            wrapper.get("pmard_engine_report_sha256") != engine_sha256
            or engine.get("experiment_id") != node_id
            or engine.get("complete") is not True
            or wrapper.get("parents") != engine.get("parents")
            or wrapper.get("pmard_execution_config_sha256")
            != engine.get("execution_config_sha256")
            or wrapper.get("selection")
            != select_checkpoint(engine.get("validation_history", ()))
        ):
            raise ValueError(f"parent screen engine differs for {node_id}")
        wrapper_reports.append(wrapper)
        engine_reports.append(engine)
        engine_by_node[node_id] = (engine_path, engine)

    screening_reports = {
        node_id: report for node_id, (_, report) in engine_by_node.items()
    }
    for node_id, report in screening_reports.items():
        _validate_primary_engine_lineage(
            report, node_id=node_id, replicate_seed=1337,
            split_sha256=split_sha256, source_sha256=source_sha256,
            assignment_lock_sha256=assignment_lock["content_hash"],
            qualification_lock_sha256=endpoint_lock["content_hash"],
            recipe_sha256=recipe_sha256,
            screening_reports=screening_reports,
        )

    expected_screen = build_screen_aggregate(
        engine_reports, node_reports=wrapper_reports,
        campaign_spec_sha256=campaign_sha256, recipe_sha256=recipe_sha256,
        assignment_lock_sha256=assignment_lock["content_hash"],
    )
    screen = artifacts["screen_aggregate"]
    _same_artifact(screen, expected_screen, name="screen aggregate")
    screen_sha256 = expected_screen["content_hash"]
    confirmation_lock = artifacts["confirmation_registry_lock"]
    confirmation_payload = confirmation_lock.get("payload")
    if not isinstance(confirmation_payload, Mapping):
        raise ValueError("parent confirmation registry payload differs")
    expected_registry = _require_exact_confirmation_registry(
        confirmation_payload.get("registry"), screen=screen,
        include_label_only_warm_continuation=bool(
            campaign["include_label_only_warm_continuation"]
        ),
    )
    expected_confirmation_lock = create_confirmation_registry_lock(
        campaign_spec_sha256=campaign_sha256, qualification_lock=endpoint_lock,
        screen_aggregate_sha256=screen_sha256,
        registry=expected_registry,
    )
    _same_artifact(
        confirmation_lock, expected_confirmation_lock,
        name="confirmation registry lock",
    )

    expected_confirmation_keys = {
        _confirmation_report_key(index, row)
        for index, row in enumerate(expected_registry)
    }
    if (
        not isinstance(confirmation_report_paths, Mapping)
        or set(confirmation_report_paths) != expected_confirmation_keys
    ):
        raise ValueError("parent confirmation report path registry differs")
    confirmation_reports: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    ordered_confirmation_hashes = []
    for index, row in enumerate(expected_registry):
        key = _confirmation_report_key(index, row)
        path, report = _registered_json_file(
            confirmation_report_paths[key], name=f"confirmation {key}",
        )
        node_id = str(row["node_id"])
        expected_config = (
            node_training_config(
                node_id, recipe, train_rows=train_rows,
                replicate_seed=int(row["seed"]),
            )
            if row["kind"] == "primary"
            else confirmation_control_training_config(
                node_id, recipe, train_rows=train_rows,
                replicate_seed=int(row["seed"]),
            )
        )
        report_sha256 = validate_hcwdl_full_parent_engine_report(
            report, train_rows=train_rows, recipe=recipe,
            expected_experiment_id=node_id,
            expected_exact_config=asdict(expected_config), report_path=path,
        )
        validation_arguments = {
            "report": report, "replicate_seed": int(row["seed"]),
            "split_sha256": split_sha256, "source_sha256": source_sha256,
            "assignment_lock_sha256": assignment_lock["content_hash"],
            "qualification_lock_sha256": endpoint_lock["content_hash"],
            "recipe_sha256": recipe_sha256,
            "screening_reports": screening_reports,
        }
        if row["kind"] == "primary":
            _validate_primary_engine_lineage(
                node_id=node_id, **validation_arguments,
            )
        else:
            _validate_control_engine_lineage(
                control_id=node_id, **validation_arguments,
            )
        confirmation_reports[key] = (path, report)
        ordered_confirmation_hashes.append(report_sha256)

    expected_confirmation = with_content_hash({
        "contract": CONFIRMATION_REPORT_CONTRACT, "schema_version": 1,
        "registry_lock_sha256": confirmation_lock["content_hash"],
        "reports": ordered_confirmation_hashes,
        "finite_bad_performance_retained": True,
    })
    confirmation = artifacts["confirmation_aggregate"]
    _same_artifact(
        confirmation, expected_confirmation, name="confirmation aggregate",
    )
    confirmation_sha256 = expected_confirmation["content_hash"]
    finalist_lock = artifacts["finalist_lock"]
    finalist_payload = finalist_lock.get("payload")
    if not isinstance(finalist_payload, Mapping):
        raise ValueError("parent finalist-lock payload differs")
    finalist_nodes = {
        "M0", "M6c", "M6w", "NULL_M1_SELF_KD",
        "NULL_M6_PREDECESSOR_ONLY",
        screen["selected_intermediate_cold"]["selected_node_id"],
        screen["selected_intermediate_warm"]["selected_node_id"],
    }
    if campaign["include_label_only_warm_continuation"]:
        finalist_nodes.add("NULL_WARM_LABEL_ONLY")
    expected_finalists = []
    for index, row in enumerate(expected_registry):
        if row["node_id"] not in finalist_nodes:
            continue
        key = _confirmation_report_key(index, row)
        path, report = confirmation_reports[key]
        expected_finalists.append({
            "node_id": row["node_id"], "seed": row["seed"],
            "checkpoint_sha256": report["selected_checkpoint_sha256"],
            "report_sha256": report["content_hash"],
            "report_path": str(path),
        })
    for node_id in ("D100", "TOFF"):
        path, report = engine_by_node[node_id]
        expected_finalists.append({
            "node_id": node_id, "seed": 1337,
            "checkpoint_sha256": report["selected_checkpoint_sha256"],
            "report_sha256": report["content_hash"],
            "report_path": str(path),
        })
    _require_exact_finalists(
        finalist_payload.get("finalists"), expected_finalists,
    )
    expected_finalist_lock = create_finalist_lock(
        campaign_spec_sha256=campaign_sha256,
        confirmation_lock=confirmation_lock,
        confirmation_report_sha256=confirmation_sha256,
        finalists=expected_finalists,
    )
    _same_artifact(finalist_lock, expected_finalist_lock, name="finalist lock")
    _validate_lock_chain(
        campaign_sha256=campaign_sha256, assignment_lock=assignment_lock,
        recipe_lock=recipe_lock, endpoint_lock=endpoint_lock,
        confirmation_lock=confirmation_lock, finalist_lock=finalist_lock,
    )

    architecture = artifacts["architecture_attestation"]
    architecture_sha256 = validate_architecture_attestation(
        architecture, require_authorized=True,
    )
    parent_loss = artifacts["parent_loss_attestation"]
    parent_loss_sha256 = validate_parent_loss_attestation(
        parent_loss, parent_recipe=recipe,
    )
    _validate_parent_loss_campaign_source(
        parent_loss, campaign=campaign, campaign_sha256=campaign_sha256,
    )

    digests = {
        "parent_campaign_spec": campaign_sha256,
        "architecture_attestation": architecture_sha256,
        "parent_loss_attestation": parent_loss_sha256,
        "source_manifest": source_sha256, "split_manifest": split_sha256,
        "row_selection": selection_sha256, "matcher_resources": matcher_sha256,
        "train_assignment_manifest": train_manifest_sha256,
        "validation_assignment_manifest": validation_manifest_sha256,
        "train_recomputation_audit": train_audit_sha256,
        "validation_recomputation_audit": validation_audit_sha256,
        "assignment_lock": assignment_lock["content_hash"],
        "parent_recipe": recipe_sha256,
        "recipe_lock": recipe_lock["content_hash"],
        "cache_miniature": cache_sha256,
        "diagnostic_authority": diagnostic_sha256,
        "qualification_report": qualification_sha256,
        "endpoint_qualification_lock": endpoint_lock["content_hash"],
        "screen_aggregate": screen_sha256,
        "confirmation_registry_lock": confirmation_lock["content_hash"],
        "confirmation_aggregate": confirmation_sha256,
        "finalist_lock": finalist_lock["content_hash"],
        "parent_graph": GRAPH_SHA256,
        **{
            f"qualifier_{name}": qualifier_hashes[name] for name in QUALIFIERS
        },
    }
    if set(digests) != PARENT_AUTHORITY_PARENT_KEYS:
        raise AssertionError("parent authority digest registry is incomplete")
    return {
        "campaign": campaign, "recipe": recipe,
        "architecture_attestation": architecture,
        "parent_loss_attestation": parent_loss,
        "parents": digests,
    }


def build_parent_import_fixture(
    *,
    parent_campaign_spec_sha256: str,
    parent_source_commit: str,
    lineage_hashes: Mapping[str, str],
    architecture_attestation: Mapping[str, Any],
    parent_loss_attestation: Mapping[str, Any],
    teachers: Mapping[str, Mapping[str, Any]],
    logit_controls: Mapping[str, Mapping[str, Any]],
    original_contract_validation: Mapping[str, bool],
    nonauthorizing_fixture: bool,
) -> dict[str, Any]:
    """Build a legacy-shape local fixture that cannot authorize execution."""

    if nonauthorizing_fixture is not True:
        raise PermissionError("legacy parent-import fixture requires an explicit nonauthorizing flag")
    if set(teachers) != set(IMPORTED_TEACHERS):
        raise ValueError("parent import teacher registry is incomplete")
    if set(logit_controls) != set(IMPORTED_LOGIT_CONTROLS):
        raise ValueError("parent import logit-control registry is incomplete")
    _bind_parent_evidence(
        architecture_attestation=architecture_attestation,
        parent_loss_attestation=parent_loss_attestation,
        teachers=teachers,
        logit_controls=logit_controls,
    )
    if set(lineage_hashes) != PARENT_IMPORT_LINEAGE_KEYS:
        raise ValueError("parent import lineage registry differs")
    if (
        set(original_contract_validation) != PARENT_IMPORT_VALIDATION_KEYS
        or any(value is not True for value in original_contract_validation.values())
    ):
        raise ValueError("parent import lacks original-contract validation proof")
    if len(parent_source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in parent_source_commit
    ):
        raise ValueError("parent source commit differs")
    normalized_lineage = {
        name: require_sha256(value, name=f"parent import {name}")
        for name, value in sorted(lineage_hashes.items())
    }
    return with_content_hash({
        "contract": LEGACY_PARENT_IMPORT_CONTRACT,
        "schema_version": 1,
        "parents": {
            "parent_campaign_spec": require_sha256(
                parent_campaign_spec_sha256, name="parent campaign spec"
            ),
            "architecture_attestation": architecture_attestation["content_hash"],
            "parent_loss_attestation": parent_loss_attestation["content_hash"],
            **normalized_lineage,
        },
        "payload": {
            "parent_source_commit": parent_source_commit,
            "teachers": [
                _import_row(teachers[node_id], node_id=node_id, teacher=True)
                for node_id in IMPORTED_TEACHERS
            ],
            "logit_controls": [
                _import_row(logit_controls[node_id], node_id=node_id, teacher=False)
                for node_id in IMPORTED_LOGIT_CONTROLS
            ],
            "original_contract_validation": dict(sorted(original_contract_validation.items())),
            "complete": True,
        },
    })


def build_parent_import_from_files(
    *, authority_files: Mapping[str, str | Path],
    qualifier_report_paths: Mapping[str, str | Path],
    confirmation_report_paths: Mapping[str, str | Path] | None = None,
    teachers: Mapping[str, Mapping[str, Any]] | None = None,
    logit_controls: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the executable import only from reopened parent authority files."""

    authority = _validated_parent_authority(
        authority_files=authority_files,
        qualifier_report_paths=qualifier_report_paths,
        confirmation_report_paths=confirmation_report_paths,
        teachers=teachers, logit_controls=logit_controls,
    )
    if (teachers is None) != (logit_controls is None):
        raise ValueError("parent import row registries must be supplied together")
    if teachers is None:
        teachers, logit_controls = derive_parent_import_rows_from_architecture(
            authority["architecture_attestation"],
        )
    assert logit_controls is not None
    if set(teachers) != set(IMPORTED_TEACHERS):
        raise ValueError("parent import teacher registry is incomplete")
    if set(logit_controls) != set(IMPORTED_LOGIT_CONTROLS):
        raise ValueError("parent import logit-control registry is incomplete")
    _bind_parent_evidence(
        architecture_attestation=authority["architecture_attestation"],
        parent_loss_attestation=authority["parent_loss_attestation"],
        teachers=teachers, logit_controls=logit_controls,
    )
    campaign = authority["campaign"]
    recipe = authority["recipe"]
    artifact = with_content_hash({
        "contract": PARENT_IMPORT_CONTRACT,
        "schema_version": 2,
        "parents": dict(sorted(authority["parents"].items())),
        "payload": {
            "parent_source_commit": campaign["source_commit"],
            "parent_campaign_contract": campaign["contract"],
            "parent_campaign_mode": campaign["mode"],
            "parent_execution_scope": campaign["execution_scope"],
            "parent_recipe_contract": recipe["contract"],
            "endpoint_continuation": campaign["endpoint_continuation"],
            "training_passes": campaign["training_passes"],
            "validation_every_passes": campaign["validation_every_passes"],
            "parent_train_rows": campaign["role_counts"]["train"],
            "terminal_task_id": campaign["terminal_task_id"],
            "execution_lock_authorized": campaign["execution_lock_authorized"],
            "final_test_access_authorized": campaign[
                "final_test_access_authorized"
            ],
            "registered_final_test_tasks": campaign[
                "registered_final_test_tasks"
            ],
            "teachers": [
                _import_row(teachers[node_id], node_id=node_id, teacher=True)
                for node_id in IMPORTED_TEACHERS
            ],
            "logit_controls": [
                _import_row(logit_controls[node_id], node_id=node_id, teacher=False)
                for node_id in IMPORTED_LOGIT_CONTROLS
            ],
            "authority_derived_from_registered_files": True,
            "complete": True,
        },
    })
    validate_parent_import(artifact)
    return artifact


def validate_parent_import(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value, expected_contract=PARENT_IMPORT_CONTRACT,
        expected_schema_version=2,
    )
    if set(value) != {
        "contract", "schema_version", "parents", "payload", "content_hash",
    }:
        raise ValueError("parent import envelope fields differ")
    if set(value["parents"]) != PARENT_AUTHORITY_PARENT_KEYS:
        raise ValueError("parent import parent lineage registry differs")
    payload = value["payload"]
    if set(payload) != {
        "parent_source_commit", "parent_campaign_contract",
        "parent_campaign_mode", "parent_execution_scope",
        "parent_recipe_contract", "endpoint_continuation",
        "training_passes", "validation_every_passes", "parent_train_rows",
        "terminal_task_id", "execution_lock_authorized",
        "final_test_access_authorized", "registered_final_test_tasks", "teachers",
        "logit_controls", "authority_derived_from_registered_files", "complete",
    }:
        raise ValueError("parent import payload fields differ")
    source_commit = payload["parent_source_commit"]
    if (
        not isinstance(source_commit, str) or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError("parent source commit differs")
    from .hcwdl_authorization import (
        AUTOMATIC_ENDPOINT_CONTINUATION, PARENT_PREFIX_SCOPE,
    )
    from .hcwdl_campaign import MODES, PARENT_PREFIX_CAMPAIGN_CONTRACT, ROLE_COUNTS

    mode = payload["parent_campaign_mode"]
    if (
        payload["parent_campaign_contract"] != PARENT_PREFIX_CAMPAIGN_CONTRACT
        or mode not in MODES
        or mode == "smoke"
        or ROLE_COUNTS[mode]["train"] is None
        or payload["parent_train_rows"] != ROLE_COUNTS[mode]["train"]
        or payload["parent_execution_scope"] != PARENT_PREFIX_SCOPE
        or payload["parent_recipe_contract"] != "HCWDL_RECIPE/v4"
        or payload["endpoint_continuation"] != AUTOMATIC_ENDPOINT_CONTINUATION
        or payload["training_passes"] != 60
        or payload["validation_every_passes"] != 1
        or payload["terminal_task_id"] != "finalist_lock"
        or payload["execution_lock_authorized"] is not False
        or payload["final_test_access_authorized"] is not False
        or payload["registered_final_test_tasks"] != 0
    ):
        raise ValueError("parent import exact v8 prefix identity differs")
    raw_teachers = payload["teachers"]
    raw_controls = payload["logit_controls"]
    if not isinstance(raw_teachers, list) or not isinstance(raw_controls, list):
        raise ValueError("parent import artifact registries differ")
    if any(not isinstance(row, Mapping) or "node_id" not in row for row in raw_teachers):
        raise ValueError("parent import teacher registry differs")
    if any(not isinstance(row, Mapping) or "node_id" not in row for row in raw_controls):
        raise ValueError("parent import logit-control registry differs")
    teachers = {row["node_id"]: row for row in raw_teachers}
    controls = {row["node_id"]: row for row in raw_controls}
    if len(teachers) != len(payload["teachers"]) or set(teachers) != set(IMPORTED_TEACHERS):
        raise ValueError("parent import teacher registry differs")
    if len(controls) != len(payload["logit_controls"]) or set(controls) != set(IMPORTED_LOGIT_CONTROLS):
        raise ValueError("parent import logit-control registry differs")
    for node_id, row in teachers.items():
        _import_row(row, node_id=node_id, teacher=True)
    for node_id, row in controls.items():
        _import_row(row, node_id=node_id, teacher=False)
    if (
        payload["authority_derived_from_registered_files"] is not True
        or payload["complete"] is not True
    ):
        raise ValueError("parent import file-authority completion proof differs")
    return digest


def validate_parent_import_against_authority_files(
    value: Mapping[str, Any],
    *,
    authority_files: Mapping[str, str | Path],
    qualifier_report_paths: Mapping[str, str | Path],
    confirmation_report_paths: Mapping[str, str | Path] | None = None,
) -> str:
    """Reopen every authority file and require the exact persisted projection."""

    digest = validate_parent_import(value)
    payload = value["payload"]
    teachers = {row["node_id"]: row for row in payload["teachers"]}
    controls = {row["node_id"]: row for row in payload["logit_controls"]}
    authority = _validated_parent_authority(
        authority_files=authority_files,
        qualifier_report_paths=qualifier_report_paths,
        confirmation_report_paths=confirmation_report_paths,
        teachers=teachers, logit_controls=controls,
    )
    if value["parents"] != authority["parents"]:
        raise ValueError("parent import authority files differ from persisted lineage")
    _bind_parent_evidence(
        architecture_attestation=authority["architecture_attestation"],
        parent_loss_attestation=authority["parent_loss_attestation"],
        teachers=teachers,
        logit_controls=controls,
    )
    campaign = authority["campaign"]
    recipe = authority["recipe"]
    if any((
        payload["parent_source_commit"] != campaign["source_commit"],
        payload["parent_campaign_contract"] != campaign["contract"],
        payload["parent_campaign_mode"] != campaign["mode"],
        payload["parent_execution_scope"] != campaign["execution_scope"],
        payload["parent_recipe_contract"] != recipe["contract"],
        payload["endpoint_continuation"] != campaign["endpoint_continuation"],
        payload["training_passes"] != campaign["training_passes"],
        payload["validation_every_passes"]
        != campaign["validation_every_passes"],
        payload["parent_train_rows"] != campaign["role_counts"]["train"],
        payload["terminal_task_id"] != campaign["terminal_task_id"],
        payload["execution_lock_authorized"]
        != campaign["execution_lock_authorized"],
        payload["final_test_access_authorized"]
        != campaign["final_test_access_authorized"],
        payload["registered_final_test_tasks"]
        != campaign["registered_final_test_tasks"],
    )):
        raise ValueError("parent import payload differs from current authority files")
    return digest


def validate_parent_import_against_evidence(*args, **kwargs) -> str:
    """Reject the retired attestation-only runtime boundary.

    Executable consumers must call
    :func:`validate_parent_import_against_authority_files`; two fresh
    attestations cannot establish the parent campaign/lock chain.
    """

    del args, kwargs
    raise PermissionError(
        "attestation-only parent import validation is nonauthorizing; "
        "registered parent authority files are required"
    )


__all__ = [
    "IMPORTED_LOGIT_CONTROLS", "IMPORTED_TEACHERS",
    "PARENT_AUTHORITY_FILE_KEYS", "PARENT_AUTHORITY_PARENT_KEYS",
    "PARENT_IMPORT_CONTRACT", "build_parent_import_fixture",
    "build_parent_import_from_files",
    "derive_parent_import_rows_from_architecture",
    "PARENT_IMPORT_LINEAGE_KEYS", "PARENT_IMPORT_VALIDATION_KEYS",
    "validate_parent_import", "validate_parent_import_against_authority_files",
    "validate_parent_import_against_evidence",
]

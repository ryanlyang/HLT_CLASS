"""Fail-closed HCWDL-RKD import and campaign-lock construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from hlt_classification.data.cache_contracts import require_sha256
from hlt_classification.models.hcwdl_surfaces import (
    validate_architecture_attestation,
)

from .hcwdl_parent_loss import validate_parent_loss_attestation
from .hcwdl_representation_contracts import (
    PARENT_IMPORT_CONTRACT,
    build_versioned_artifact,
    validate_versioned_artifact,
)


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
    if teacher and node_id == "TOFF" and domain != "native_offline":
        raise ValueError("TOFF parent import domain differs")
    if teacher and node_id != "TOFF" and not domain.startswith("d"):
        raise ValueError("D teacher parent import domain differs")
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


def build_parent_import(
    *,
    parent_campaign_spec_sha256: str,
    parent_source_commit: str,
    lineage_hashes: Mapping[str, str],
    architecture_attestation: Mapping[str, Any],
    parent_loss_attestation: Mapping[str, Any],
    teachers: Mapping[str, Mapping[str, Any]],
    logit_controls: Mapping[str, Mapping[str, Any]],
    original_contract_validation: Mapping[str, bool],
) -> dict[str, Any]:
    # Architecture attestations deliberately use their own strict flat schema;
    # treating them as the generic parents/payload envelope would accept the
    # wrong artifact family and broke the required schema->parity->import gate.
    validate_architecture_attestation(
        architecture_attestation, require_authorized=True,
    )
    validate_parent_loss_attestation(parent_loss_attestation)
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
    return build_versioned_artifact(
        PARENT_IMPORT_CONTRACT,
        parents={
            "parent_campaign_spec": require_sha256(
                parent_campaign_spec_sha256, name="parent campaign spec"
            ),
            "architecture_attestation": architecture_attestation["content_hash"],
            "parent_loss_attestation": parent_loss_attestation["content_hash"],
            **normalized_lineage,
        },
        payload={
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
    )


def validate_parent_import(value: Mapping[str, Any]) -> str:
    digest = validate_versioned_artifact(
        value,
        expected_contract=PARENT_IMPORT_CONTRACT,
        required_payload_keys=(
            "parent_source_commit", "teachers", "logit_controls",
            "original_contract_validation", "complete",
        ),
    )
    if set(value) != {
        "contract", "schema_version", "parents", "payload", "content_hash",
    }:
        raise ValueError("parent import envelope fields differ")
    expected_parent_keys = {
        "parent_campaign_spec", "architecture_attestation",
        "parent_loss_attestation", *PARENT_IMPORT_LINEAGE_KEYS,
    }
    if set(value["parents"]) != expected_parent_keys:
        raise ValueError("parent import parent lineage registry differs")
    payload = value["payload"]
    if set(payload) != {
        "parent_source_commit", "teachers", "logit_controls",
        "original_contract_validation", "complete",
    }:
        raise ValueError("parent import payload fields differ")
    source_commit = payload["parent_source_commit"]
    if (
        not isinstance(source_commit, str) or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError("parent source commit differs")
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
    proof = payload["original_contract_validation"]
    if (
        not isinstance(proof, Mapping)
        or set(proof) != PARENT_IMPORT_VALIDATION_KEYS
        or any(value is not True for value in proof.values())
        or payload["complete"] is not True
    ):
        raise ValueError("parent import completion proof differs")
    return digest


def validate_parent_import_against_evidence(
    value: Mapping[str, Any],
    *,
    architecture_attestation: Mapping[str, Any],
    parent_loss_attestation: Mapping[str, Any],
) -> str:
    """Bind a persisted import to the freshly audited parent files.

    ``validate_parent_import`` proves that the import is internally complete,
    but an executable campaign must additionally prove that it is the import
    described by the architecture and corrected-loss tasks in *this* DAG.
    Those tasks reopen the registered parent report/model bundles, so this
    comparison also prevents a nearby self-consistent bundle from being used
    in place of the campaign-spec import.
    """

    digest = validate_parent_import(value)
    architecture_digest = validate_architecture_attestation(
        architecture_attestation, require_authorized=True,
    )
    loss_digest = validate_parent_loss_attestation(parent_loss_attestation)
    parents = value.get("parents")
    if not isinstance(parents, Mapping) or (
        parents.get("architecture_attestation") != architecture_digest
        or parents.get("parent_loss_attestation") != loss_digest
    ):
        raise ValueError("parent import attestation parents differ from fresh evidence")
    payload = value["payload"]
    teachers = {row["node_id"]: row for row in payload["teachers"]}
    controls = {row["node_id"]: row for row in payload["logit_controls"]}
    _bind_parent_evidence(
        architecture_attestation=architecture_attestation,
        parent_loss_attestation=parent_loss_attestation,
        teachers=teachers,
        logit_controls=controls,
    )
    return digest


__all__ = [
    "IMPORTED_LOGIT_CONTROLS", "IMPORTED_TEACHERS", "build_parent_import",
    "PARENT_IMPORT_LINEAGE_KEYS", "PARENT_IMPORT_VALIDATION_KEYS",
    "validate_parent_import", "validate_parent_import_against_evidence",
]

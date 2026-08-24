"""Read-only source authentication for the TRI60 dense extension."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import load_json, sha256_file, write_immutable_json

from .hcwdl_mhpe_tri60_campaign import validate_campaign as validate_source_campaign
from .hcwdl_mhpe_tri60_contracts import (
    CAMPAIGN_COMPLETE_CONTRACT as SOURCE_COMPLETE_CONTRACT,
    TRAINING_REPORT_CONTRACT as SOURCE_TRAINING_REPORT_CONTRACT,
    validate_artifact as validate_source_artifact,
)
from .hcwdl_mhpe_tri60_dense_contracts import (
    SOURCE_GATE_CONTRACT, SOURCE_LOCK_CONTRACT, artifact, hashes,
    validate_artifact,
)
from .hcwdl_mhpe_tri60_dense_graph import (
    EARLY_SOURCE_NODES, GRAPH_SHA256, LATE_SOURCE_NODES, SOURCE_DISTRIBUTIONS,
    SOURCE_GRAPH_SHA256, source_distribution_consumers,
)
from .hcwdl_mhpe_tri60_probability import validate_probability_lock


def _source_spec(path: str | Path) -> tuple[dict[str, Any], str]:
    spec = load_json(path)
    digest = validate_source_campaign(
        spec, executable=False, verify_source_tree=False,
    )
    if spec.get("parents", {}).get("graph") != SOURCE_GRAPH_SHA256:
        raise ValueError("dense extension source graph differs")
    if spec.get("final_test_accessed") is not False:
        raise PermissionError("dense extension source accessed final test")
    return spec, digest


def _training_lineage(
    spec: Mapping[str, Any], node_id: str,
) -> dict[str, str]:
    root = Path(spec["campaign_root"])
    report_path = root / "training" / node_id / "training_report.json"
    report = load_json(report_path)
    validate_source_artifact(report, contract=SOURCE_TRAINING_REPORT_CONTRACT)
    selected = report_path.parent / str(report.get("selected_checkpoint", ""))
    final = report_path.parent / str(report.get("final_checkpoint", ""))
    if (
        report.get("node_id") != node_id
        or report.get("campaign_spec_sha256") != spec["content_hash"]
        or report.get("graph_sha256") != spec["parents"]["graph"]
        or report.get("recipe_sha256") != spec["parents"]["recipe"]
        or report.get("passes") != 60
        or report.get("validations") != 60
        or report.get("complete") is not True
        or report.get("rolling_resume_published") is not False
        or report.get("partial_checkpoint_reuse") is not False
        or report.get("final_test_accessed") is not False
        or not selected.is_file()
        or not final.is_file()
        or sha256_file(selected) != report.get("selected_checkpoint_sha256")
        or sha256_file(final) != report.get("final_checkpoint_sha256")
    ):
        raise ValueError(f"dense source training lineage differs: {node_id}")
    return {
        "report_path": str(report_path.resolve()),
        "report_sha256": report["content_hash"],
        "selected_checkpoint_path": str(selected.resolve()),
        "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
    }


def _distribution_lineage(
    spec: Mapping[str, Any], distribution_id: str,
) -> dict[str, str]:
    root = Path(spec["campaign_root"]) / "probabilities" / distribution_id
    lock, manifests = validate_probability_lock(
        root / "lock.json", distribution_id=distribution_id,
    )
    return {
        "root": str(root.resolve()),
        "lock_path": str((root / "lock.json").resolve()),
        "lock_sha256": lock["content_hash"],
        "train_manifest_sha256": manifests["train"]["content_hash"],
        "validation_manifest_sha256": manifests["validation"]["content_hash"],
    }


def build_source_lock(
    *, source_campaign_spec: str | Path,
    source_completion_job_id: str | None,
) -> dict[str, Any]:
    spec, spec_hash = _source_spec(source_campaign_spec)
    completion_path = Path(spec["campaign_root"]) / "reports/campaign_complete.json"
    already_complete = completion_path.is_file()
    if already_complete:
        complete = load_json(completion_path)
        validate_source_artifact(complete, contract=SOURCE_COMPLETE_CONTRACT)
        if complete.get("parents", {}).get("campaign_spec") != spec_hash:
            raise ValueError("dense source completion lineage differs")
        source_completion_job_id = None
    elif not re.fullmatch(r"[1-9][0-9]*", str(source_completion_job_id or "")):
        raise ValueError(
            "incomplete dense source requires its exact campaign-complete Slurm job ID"
        )
    early_nodes = {
        node_id: _training_lineage(spec, node_id)
        for node_id in EARLY_SOURCE_NODES
    }
    distributions = {
        distribution_id: _distribution_lineage(spec, distribution_id)
        for distribution_id in SOURCE_DISTRIBUTIONS
    }
    return artifact({
        "parents": {
            "source_campaign": spec_hash,
            "source_graph": spec["parents"]["graph"],
            "source_recipe": spec["parents"]["recipe"],
            "foundation": spec["parents"]["foundation"],
            "dense_graph": GRAPH_SHA256,
        },
        "source_campaign_spec_path": str(Path(source_campaign_spec).resolve()),
        "source_campaign_root": str(Path(spec["campaign_root"]).resolve()),
        "source_completion_path": str(completion_path.resolve()),
        "source_completion_already_complete": already_complete,
        "source_completion_job_id": source_completion_job_id,
        "early_node_order": list(EARLY_SOURCE_NODES),
        "early_distribution_order": list(SOURCE_DISTRIBUTIONS),
        "early_nodes": early_nodes, "early_distributions": distributions,
        "authorized_dense_probability_consumers": {
            distribution_id: list(source_distribution_consumers(distribution_id))
            for distribution_id in SOURCE_DISTRIBUTIONS
        },
        "probability_consumer_adapter_policy": (
            "new_lock_binds_immutable_source_bank_and_exact_dense_consumers_v1"
        ),
        "late_node_ids": list(LATE_SOURCE_NODES),
        "read_only_import": True, "source_outputs_mutated": False,
        "final_test_accessed": False,
    }, contract=SOURCE_LOCK_CONTRACT)


def validate_source_lock(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=SOURCE_LOCK_CONTRACT)
    spec, spec_hash = _source_spec(value["source_campaign_spec_path"])
    if (
        value.get("parents", {}).get("source_campaign") != spec_hash
        or value.get("parents", {}).get("source_graph") != SOURCE_GRAPH_SHA256
        or value.get("parents", {}).get("dense_graph") != GRAPH_SHA256
        or value.get("early_node_order") != list(EARLY_SOURCE_NODES)
        or value.get("early_distribution_order") != list(SOURCE_DISTRIBUTIONS)
        or set(value.get("early_nodes", {})) != set(EARLY_SOURCE_NODES)
        or set(value.get("early_distributions", {})) != set(SOURCE_DISTRIBUTIONS)
        or value.get("authorized_dense_probability_consumers") != {
            distribution_id: list(source_distribution_consumers(distribution_id))
            for distribution_id in SOURCE_DISTRIBUTIONS
        }
        or value.get("probability_consumer_adapter_policy")
        != "new_lock_binds_immutable_source_bank_and_exact_dense_consumers_v1"
        or value.get("late_node_ids") != list(LATE_SOURCE_NODES)
        or value.get("read_only_import") is not True
        or value.get("source_outputs_mutated") is not False
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("dense source lock semantics differ")
    for node_id in EARLY_SOURCE_NODES:
        if _training_lineage(spec, node_id) != value["early_nodes"][node_id]:
            raise ValueError(f"dense early source changed: {node_id}")
    for distribution_id in SOURCE_DISTRIBUTIONS:
        if _distribution_lineage(spec, distribution_id) != value["early_distributions"][distribution_id]:
            raise ValueError(f"dense source distribution changed: {distribution_id}")
    complete = bool(value.get("source_completion_already_complete"))
    job = value.get("source_completion_job_id")
    if complete != (job is None):
        raise ValueError("dense source completion scheduling differs")
    if not complete and re.fullmatch(r"[1-9][0-9]*", str(job or "")) is None:
        raise ValueError("dense source completion job ID differs")
    return digest


def publish_source_gate(
    source_lock: Mapping[str, Any], *, output: str | Path,
) -> dict[str, Any]:
    lock_hash = validate_source_lock(source_lock)
    spec, spec_hash = _source_spec(source_lock["source_campaign_spec_path"])
    completion = load_json(source_lock["source_completion_path"])
    completion_hash = validate_source_artifact(
        completion, contract=SOURCE_COMPLETE_CONTRACT,
    )
    if completion.get("parents", {}).get("campaign_spec") != spec_hash:
        raise ValueError("dense source gate completion differs")
    late = {
        node_id: _training_lineage(spec, node_id)
        for node_id in LATE_SOURCE_NODES
    }
    gate = artifact({
        "parents": {
            "source_lock": lock_hash, "source_campaign": spec_hash,
            "source_completion": completion_hash,
        },
        "late_node_order": list(LATE_SOURCE_NODES),
        "late_nodes": late, "all_source_imports_authenticated": True,
        "read_only_import": True, "source_outputs_mutated": False,
        "final_test_accessed": False,
    }, contract=SOURCE_GATE_CONTRACT)
    write_immutable_json(output, gate)
    return gate


def validate_source_gate(
    value: Mapping[str, Any], *, source_lock: Mapping[str, Any],
) -> str:
    digest = validate_artifact(value, contract=SOURCE_GATE_CONTRACT)
    lock_hash = validate_source_lock(source_lock)
    spec, spec_hash = _source_spec(source_lock["source_campaign_spec_path"])
    if (
        value.get("parents", {}).get("source_lock") != lock_hash
        or value.get("parents", {}).get("source_campaign") != spec_hash
        or value.get("late_node_order") != list(LATE_SOURCE_NODES)
        or set(value.get("late_nodes", {})) != set(LATE_SOURCE_NODES)
        or value.get("all_source_imports_authenticated") is not True
        or value.get("read_only_import") is not True
        or value.get("source_outputs_mutated") is not False
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("dense source gate semantics differ")
    for node_id in LATE_SOURCE_NODES:
        if _training_lineage(spec, node_id) != value["late_nodes"][node_id]:
            raise ValueError(f"dense late source changed: {node_id}")
    return digest


def source_node_lineage(
    *, source_lock: Mapping[str, Any], source_gate: Mapping[str, Any] | None,
    node_id: str,
) -> Mapping[str, str]:
    validate_source_lock(source_lock)
    if node_id in source_lock["early_nodes"]:
        return source_lock["early_nodes"][node_id]
    if source_gate is None:
        raise FileNotFoundError("dense late source gate is absent")
    validate_source_gate(source_gate, source_lock=source_lock)
    try:
        return source_gate["late_nodes"][node_id]
    except KeyError as error:
        raise KeyError(f"unknown dense source node: {node_id}") from error


__all__ = [
    "build_source_lock", "publish_source_gate", "source_node_lineage",
    "validate_source_gate", "validate_source_lock",
]

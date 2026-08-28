"""Read-only U000 authentication for the TRI100 four-spine study."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import load_json, sha256_file

from .hcwdl_mhpe_tri60_campaign import validate_campaign as validate_source_campaign
from .hcwdl_mhpe_tri60_contracts import (
    TRAINING_REPORT_CONTRACT as SOURCE_TRAINING_REPORT_CONTRACT,
    validate_artifact as validate_source_artifact,
)
from .hcwdl_mhpe_tri60_graph import GRAPH_SHA256 as SOURCE_GRAPH_SHA256
from .hcwdl_mhpe_tri60_probability import validate_probability_lock
from .hcwdl_tri100_spine4_contracts import (
    SOURCE_LOCK_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_tri100_spine4_graph import (
    GRAPH_SHA256, NODE_REGISTRY, SOURCE_DISTRIBUTION,
)


def _source_spec(path: str | Path) -> tuple[dict[str, Any], str]:
    spec = load_json(path)
    digest = validate_source_campaign(
        spec, executable=False, verify_source_tree=False,
    )
    if (
        spec.get("parents", {}).get("graph") != SOURCE_GRAPH_SHA256
        or spec.get("ordinary_final_test_capability") is not False
        or spec.get("final_test_accessed") is not False
    ):
        raise PermissionError("TRI100 four-spine source campaign differs")
    return spec, digest


def _u000_lineage(spec: Mapping[str, Any]) -> dict[str, str]:
    report_path = Path(spec["campaign_root"]) / "training/U000/training_report.json"
    report = load_json(report_path)
    validate_source_artifact(report, contract=SOURCE_TRAINING_REPORT_CONTRACT)
    selected = report_path.parent / str(report.get("selected_checkpoint", ""))
    final = report_path.parent / str(report.get("final_checkpoint", ""))
    if (
        report.get("node_id") != "U000"
        or report.get("campaign_spec_sha256") != spec["content_hash"]
        or report.get("graph_sha256") != spec["parents"]["graph"]
        or report.get("recipe_sha256") != spec["parents"]["recipe"]
        or report.get("passes") != 60
        or report.get("validations") != 60
        or report.get("complete") is not True
        or report.get("rolling_resume_published") is not False
        or report.get("partial_checkpoint_reuse") is not False
        or report.get("final_test_accessed") is not False
        or not selected.is_file() or not final.is_file()
        or sha256_file(selected) != report.get("selected_checkpoint_sha256")
        or sha256_file(final) != report.get("final_checkpoint_sha256")
    ):
        raise ValueError("TRI100 four-spine U000 lineage differs")
    return {
        "report_path": str(report_path.resolve()),
        "report_sha256": report["content_hash"],
        "selected_checkpoint_path": str(selected.resolve()),
        "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
        "final_checkpoint_path": str(final.resolve()),
        "final_checkpoint_sha256": report["final_checkpoint_sha256"],
    }


def _u000_probability_lineage(spec: Mapping[str, Any]) -> dict[str, str]:
    root = Path(spec["campaign_root"]) / "probabilities" / SOURCE_DISTRIBUTION
    lock, manifests = validate_probability_lock(
        root / "lock.json", distribution_id=SOURCE_DISTRIBUTION,
    )
    return {
        "root": str(root.resolve()),
        "lock_path": str((root / "lock.json").resolve()),
        "lock_sha256": lock["content_hash"],
        "train_manifest_path": str((root / "train_manifest.json").resolve()),
        "train_manifest_sha256": manifests["train"]["content_hash"],
        "validation_manifest_path": str(
            (root / "validation_manifest.json").resolve()
        ),
        "validation_manifest_sha256": manifests["validation"]["content_hash"],
    }


def source_consumers() -> tuple[str, ...]:
    return tuple(
        node_id for node_id, node in NODE_REGISTRY.items()
        if node.distribution_teacher_id == SOURCE_DISTRIBUTION
    )


def build_source_lock(source_campaign_spec: str | Path) -> dict[str, Any]:
    spec, spec_hash = _source_spec(source_campaign_spec)
    u000 = _u000_lineage(spec)
    probability = _u000_probability_lineage(spec)
    return artifact({
        "parents": {
            "source_campaign": spec_hash,
            "source_graph": spec["parents"]["graph"],
            "source_recipe": spec["parents"]["recipe"],
            "foundation": spec["parents"]["foundation"],
            "four_spine_graph": GRAPH_SHA256,
            "u000_report": u000["report_sha256"],
            "u000_checkpoint": u000["selected_checkpoint_sha256"],
            "u000_probability_lock": probability["lock_sha256"],
        },
        "source_campaign_spec_path": str(Path(source_campaign_spec).resolve()),
        "source_campaign_root": str(Path(spec["campaign_root"]).resolve()),
        "foundation_spec_path": str(
            Path(spec["artifact_paths"]["foundation_spec"]).resolve()
        ),
        "u000": u000,
        "u000_probability": probability,
        "authorized_probability_consumers": list(source_consumers()),
        "probability_consumer_adapter_policy": (
            "new_lock_binds_immutable_u000_bank_and_exact_four_spine_consumers_v1"
        ),
        "replicate_seed": int(spec["replicate_seed"]),
        "role_counts": dict(spec["role_counts"]),
        "population_policy": spec["population_policy"],
        "read_only_import": True,
        "source_completion_not_required": True,
        "source_outputs_mutated": False,
        "final_test_accessed": False,
    }, contract=SOURCE_LOCK_CONTRACT)


def validate_source_lock(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=SOURCE_LOCK_CONTRACT)
    spec, spec_hash = _source_spec(value["source_campaign_spec_path"])
    if (
        value.get("parents", {}).get("source_campaign") != spec_hash
        or value.get("parents", {}).get("source_graph") != SOURCE_GRAPH_SHA256
        or value.get("parents", {}).get("four_spine_graph") != GRAPH_SHA256
        or value.get("u000") != _u000_lineage(spec)
        or value.get("u000_probability") != _u000_probability_lineage(spec)
        or value.get("authorized_probability_consumers") != list(source_consumers())
        or value.get("probability_consumer_adapter_policy")
        != "new_lock_binds_immutable_u000_bank_and_exact_four_spine_consumers_v1"
        or value.get("replicate_seed") != int(spec["replicate_seed"])
        or value.get("role_counts") != dict(spec["role_counts"])
        or value.get("population_policy") != "all_authenticated_mapped_rows_v1"
        or value.get("read_only_import") is not True
        or value.get("source_completion_not_required") is not True
        or value.get("source_outputs_mutated") is not False
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI100 four-spine source lock differs")
    return digest


__all__ = [
    "build_source_lock", "source_consumers", "validate_source_lock",
]

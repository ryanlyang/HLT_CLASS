"""Full-cardinality assignments plus read-only pure-offline oracle."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import (
    load_json,
    validate_content_hash,
)

from .hcwdl_fullcard_bottleneck_contracts import (
    ASSIGNMENT_LOCK_CONTRACT,
    DIAGNOSTIC_REPORT_CONTRACT,
    FOUNDATION_LOCK_CONTRACT,
    MATCHER_ACCEPTANCE_CONTRACT,
    SCHEMA_VERSION,
    U000_EQUIVALENCE_LOCK_CONTRACT,
)
from .hcwdl_fullcard_bottleneck_foundation_campaign import validate_foundation
from .hcwdl_tri100_spine4_bottleneck_contracts import (
    SOURCE_LOCK_CONTRACT,
    artifact,
    validate_artifact,
)
from .hcwdl_tri100_spine4_bottleneck_graph import (
    GRAPH_SHA256,
    NODE_REGISTRY,
    SOURCE_DISTRIBUTION,
)
from .hcwdl_tri100_spine4_source import validate_source_lock as validate_established_source


def source_consumers() -> tuple[str, ...]:
    # The pure-offline U000 artifacts are reporting references only.  The new
    # anchor is fitted inside this campaign and owns its own probability bank.
    return ()


def build_source_lock(foundation_spec_path: str | Path) -> dict[str, Any]:
    foundation = load_json(foundation_spec_path)
    foundation_hash = validate_foundation(foundation)
    foundation_lock = load_json(foundation["artifact_paths"]["foundation_lock"])
    foundation_lock_hash = validate_content_hash(
        foundation_lock, expected_contract=FOUNDATION_LOCK_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    equivalence = load_json(foundation["artifact_paths"]["u000_equivalence_lock"])
    equivalence_hash = validate_content_hash(
        equivalence, expected_contract=U000_EQUIVALENCE_LOCK_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    established = load_json(foundation["artifact_paths"]["source_lock"])
    established_hash = validate_established_source(established)
    assignment = load_json(Path(foundation["campaign_root"]) / "locks/assignment.json")
    assignment_hash = validate_content_hash(
        assignment, expected_contract=ASSIGNMENT_LOCK_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    acceptance = load_json(
        Path(foundation["campaign_root"]) / "locks/matcher_acceptance.json"
    )
    acceptance_hash = validate_content_hash(
        acceptance, expected_contract=MATCHER_ACCEPTANCE_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    for role in ("train", "validation"):
        diagnostic = load_json(
            Path(foundation["campaign_root"]) / f"matcher/{role}_diagnostics.json"
        )
        diagnostic_hash = validate_content_hash(
            diagnostic, expected_contract=DIAGNOSTIC_REPORT_CONTRACT,
            expected_schema_version=SCHEMA_VERSION,
        )
        if assignment.get("role_diagnostics", {}).get(role) != diagnostic_hash:
            raise ValueError("foundation assignment diagnostic lineage differs")
    if (
        foundation_lock.get("foundation_spec_sha256") != foundation_hash
        or foundation_lock.get("parents", {}).get("assignment_lock") != assignment_hash
        or foundation_lock.get("parents", {}).get("u000_equivalence_lock")
        != equivalence_hash
        or foundation_lock.get("parents", {}).get("matcher_acceptance")
        != acceptance_hash
        or foundation_lock.get("role_counts") != foundation["role_counts"]
        or foundation_lock.get("u000_reused_read_only") is not True
        or foundation_lock.get("assignment_dependent_descendants_rebuilt") is not True
        or foundation_lock.get("pairing_provenance")
        != "validity_only_not_correspondence_confidence"
        or foundation_lock.get("rolling_resume_persisted") is not False
        or foundation_lock.get("optimizer_state_persisted") is not False
        or foundation_lock.get("ordinary_final_test_capability") is not False
        or foundation_lock.get("final_test_accessed") is not False
        or equivalence.get("foundation_spec_sha256") != foundation_hash
        or equivalence.get("parents", {}).get("new_assignment_lock") != assignment_hash
        or equivalence.get("role_rows") != {
            role: int(foundation["role_counts"][role])
            for role in ("train", "validation")
        }
        or equivalence.get("identical_p0_tensors_all_rows") is not True
        or equivalence.get("identical_labels_and_identity_order") is not True
        or equivalence.get("u000_checkpoint_reused_read_only") is not True
        or equivalence.get("u000_probability_bank_reused_read_only") is not True
        or equivalence.get("u000_retrained") is not False
        or equivalence.get("final_test_accessed") is not False
        or assignment.get("foundation_spec_sha256") != foundation_hash
        or assignment.get("matcher_spec_sha256") != foundation["parents"]["matcher_spec"]
        or assignment.get("complete_smaller_side_coverage") is not True
        or assignment.get("pairing_provenance")
        != "validity_only_not_correspondence_confidence"
        or assignment.get("final_test_accessed") is not False
    ):
        raise ValueError("full-cardinality foundation completion lineage differs")
    return artifact({
        "parents": {
            "foundation_spec": foundation_hash,
            "foundation_lock": foundation_lock_hash,
            "u000_equivalence_lock": equivalence_hash,
            "established_source_lock": established_hash,
            "source_campaign": established["parents"]["source_campaign"],
            "graph": GRAPH_SHA256,
            "matcher_spec": foundation["parents"]["matcher_spec"],
            "assignment_lock": assignment_hash,
        },
        "foundation_spec_path": str(Path(foundation_spec_path).resolve()),
        "foundation_root": str(Path(foundation["campaign_root"]).resolve()),
        "u000": established["u000"],
        "u000_probability": established["u000_probability"],
        "authorized_probability_consumers": [],
        "u000_reuse_authority": "oracle_reporting_only_not_training_v1",
        "replicate_seed": int(foundation["replicate_seed"]),
        "role_counts": dict(foundation["role_counts"]),
        "population_policy": "all_authenticated_mapped_rows_v1",
        "read_only_u000_import": True,
        "pure_offline_u000_role": "oracle_reporting_reference_only",
        "persistent_anchor_retrained": True,
        "source_completion_not_required": True,
        "existing_campaign_dependencies": [],
        "source_outputs_mutated": False,
        "final_test_accessed": False,
    }, contract=SOURCE_LOCK_CONTRACT)


def validate_source_lock(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=SOURCE_LOCK_CONTRACT)
    expected = build_source_lock(value["foundation_spec_path"])
    if dict(value) != expected:
        raise ValueError("bottleneck four-spine source lock differs")
    return digest


__all__ = ["build_source_lock", "source_consumers", "validate_source_lock"]

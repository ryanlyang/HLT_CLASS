"""Explicit authentication of the full-cardinality U100 handoff source."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, validate_content_hash, with_content_hash,
)

from .hcwdl_adjacent_output_handoff_contracts import (
    CONTROL_LOCK_CONTRACT, SOURCE_LOCK_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_mhpe_tri60_ce_control_contracts import (
    TRAINING_REPORT_CONTRACT as CE60_TRAINING_REPORT_CONTRACT,
)
from .hcwdl_mhpe_tri60_contracts import (
    TRAINING_REPORT_CONTRACT as TRI60_TRAINING_REPORT_CONTRACT,
)
from .hcwdl_fullcard_bottleneck_contracts import (
    FOUNDATION_LOCK_CONTRACT,
)
from .hcwdl_fullcard_bottleneck_foundation_campaign import validate_foundation
from .hcwdl_homotopy import DEFAULT_SUPPORT_POLICY
from .hcwdl_tri100_spine4_graph import (
    BRANCH_NODES as FULLCARD_BRANCH_NODES,
    BRANCH_ORDER as FULLCARD_BRANCH_ORDER,
    BRANCH_PATHS as FULLCARD_BRANCH_PATHS,
    ENDPOINT_NODES as FULLCARD_ENDPOINT_NODES,
    FIT_ORDER as FULLCARD_FIT_ORDER,
    GRAPH_SHA256 as ESTABLISHED_GRAPH_SHA256,
    NODE_REGISTRY as FULLCARD_NODES,
    PROBABILITY_COMPONENTS as FULLCARD_PROBABILITY_COMPONENTS,
    REDUCER_ORDER as FULLCARD_REDUCER_ORDER,
    SOURCE_DISTRIBUTION as FULLCARD_SOURCE_DISTRIBUTION,
    recipe_payload as established_recipe_payload,
)
from .repair import PAIRING_VALIDITY_UNCLASSIFIED_HLT_POLICY


FULLCARD_PREFIX = "HCWDL_TRI100_FOUR_SPINE_FULLCARD_BOTTLENECK"
FULLCARD_SPEC_CONTRACT = f"{FULLCARD_PREFIX}_CAMPAIGN_SPEC/v2"
FULLCARD_GRAPH_CONTRACT = f"{FULLCARD_PREFIX}_GRAPH/v2"
FULLCARD_RECIPE_CONTRACT = f"{FULLCARD_PREFIX}_RECIPE/v2"
FULLCARD_TRAINING_REPORT_CONTRACT = f"{FULLCARD_PREFIX}_TRAINING_REPORT/v2"
FULLCARD_SELECTED_CHECKPOINT_CONTRACT = (
    f"{FULLCARD_PREFIX}_SELECTED_CHECKPOINT/v2"
)
FULLCARD_FINAL_CHECKPOINT_CONTRACT = f"{FULLCARD_PREFIX}_FINAL_CHECKPOINT/v2"
SOURCE_U100_NODE_ID = "SP4_COARSE_U100_from_U050"
SOURCE_CAMPAIGN_FAMILY = "fullcard_bottleneck_nonpersistent_v2"


def _historical_graph_payload() -> dict[str, Any]:
    """Reconstruct the exact graph published by clean commit ``965bdad6``."""

    return with_content_hash({
        "contract": FULLCARD_GRAPH_CONTRACT, "schema_version": 1,
        "campaign_label": "HCWDL-TRI100-FOUR-SPINE-FULLCARD-BOTTLENECK",
        "established_graph_sha256": ESTABLISHED_GRAPH_SHA256,
        "pairing_control": "full_cardinality_lexicographic_bottleneck_delta_r_v1",
        "matched_unclassified_hlt_policy": (
            PAIRING_VALIDITY_UNCLASSIFIED_HLT_POLICY
        ),
        "branch_order": list(FULLCARD_BRANCH_ORDER),
        "branch_paths": {
            name: list(FULLCARD_BRANCH_PATHS[name])
            for name in FULLCARD_BRANCH_ORDER
        },
        "branch_nodes": {
            name: list(FULLCARD_BRANCH_NODES[name])
            for name in FULLCARD_BRANCH_ORDER
        },
        "nodes": [
            FULLCARD_NODES[name].payload() for name in FULLCARD_FIT_ORDER
        ],
        "source_distribution": FULLCARD_SOURCE_DISTRIBUTION,
        "probability_components": {
            name: list(value)
            for name, value in FULLCARD_PROBABILITY_COMPONENTS.items()
        },
        "fit_order": list(FULLCARD_FIT_ORDER),
        "reducer_order": list(FULLCARD_REDUCER_ORDER),
        "endpoint_nodes": list(FULLCARD_ENDPOINT_NODES),
        "fresh_fit_count": len(FULLCARD_FIT_ORDER),
        "single_component_probability_banks": True,
        "immediate_parent_only": True, "ensembles": False,
        "weight_continuation": False, "final_test_accessed": False,
    })


def _historical_recipe_payload() -> dict[str, Any]:
    established = established_recipe_payload()
    return with_content_hash({
        "campaign_label": "HCWDL-TRI100-FOUR-SPINE-FULLCARD-BOTTLENECK",
        "established_recipe_sha256": established["content_hash"],
        "training": established["training"], "loss": established["loss"],
        "initialization": established["initialization"],
        "teacher_policy": established["teacher_policy"],
        "execution": established["execution"],
        "same_coordinate_seed_policy": established[
            "same_coordinate_seed_policy"
        ],
        "matched_unclassified_hlt_policy": (
            PAIRING_VALIDITY_UNCLASSIFIED_HLT_POLICY
        ),
        "only_changed_variable": "particle_pairing_foundation_lineage",
        "rolling_resume": False, "final_test_accessed": False,
        "contract": FULLCARD_RECIPE_CONTRACT, "schema_version": 1,
    })


def _validate_fullcard_campaign(
    path: Path, value: Mapping[str, Any],
) -> tuple[str, dict[str, Any], str, Path, str]:
    """Authenticate the retired, immutable non-persistent `/v2` source.

    The original module name was later versioned for persistent-HLT support,
    so importing its current validator would silently assign the wrong view
    semantics to an already-published `hcwsp4b` artifact.  This compatibility
    validator binds the historical artifact family directly and fail-closes
    on every field that controls the imported population and U100 view.
    """

    digest = validate_content_hash(
        value, expected_contract=FULLCARD_SPEC_CONTRACT,
        expected_schema_version=1,
    )
    root = Path(value.get("campaign_root", "")).resolve()
    graph_path = Path(value.get("artifact_paths", {}).get("graph", "")).resolve()
    recipe_path = Path(value.get("artifact_paths", {}).get("recipe", "")).resolve()
    foundation_path = Path(
        value.get("artifact_paths", {}).get("foundation_spec", "")
    ).resolve()
    if path != (root / "campaign_spec.json").resolve():
        raise ValueError("output-handoff fullcard source spec path differs")
    graph = load_json(graph_path)
    graph_hash = validate_content_hash(
        graph, expected_contract=FULLCARD_GRAPH_CONTRACT,
        expected_schema_version=1,
    )
    recipe = load_json(recipe_path)
    recipe_hash = validate_content_hash(
        recipe, expected_contract=FULLCARD_RECIPE_CONTRACT,
        expected_schema_version=1,
    )
    if graph != _historical_graph_payload() or recipe != _historical_recipe_payload():
        raise ValueError("output-handoff historical fullcard graph/recipe differs")
    foundation = load_json(foundation_path)
    foundation_hash = validate_foundation(foundation)
    foundation_lock_path = Path(
        foundation["artifact_paths"]["foundation_lock"]
    ).resolve()
    foundation_lock = load_json(foundation_lock_path)
    foundation_lock_hash = validate_content_hash(
        foundation_lock, expected_contract=FOUNDATION_LOCK_CONTRACT,
        expected_schema_version=1,
    )
    node_rows = {
        str(row.get("node_id")): row for row in graph.get("nodes", ())
        if isinstance(row, Mapping)
    }
    u100 = node_rows.get(SOURCE_U100_NODE_ID)
    training = recipe.get("training", {})
    loss = recipe.get("loss", {})
    parents = value.get("parents", {})
    if (
        graph.get("campaign_label")
        != "HCWDL-TRI100-FOUR-SPINE-FULLCARD-BOTTLENECK"
        or graph.get("pairing_control")
        != "full_cardinality_lexicographic_bottleneck_delta_r_v1"
        or graph.get("fresh_fit_count") != 29
        or graph.get("immediate_parent_only") is not True
        or graph.get("ensembles") is not False
        or graph.get("final_test_accessed") is not False
        or not isinstance(u100, Mapping)
        or dict(u100) != FULLCARD_NODES[SOURCE_U100_NODE_ID].payload()
        or recipe.get("campaign_label")
        != "HCWDL-TRI100-FOUR-SPINE-FULLCARD-BOTTLENECK"
        or recipe.get("only_changed_variable")
        != "particle_pairing_foundation_lineage"
        or training.get("maximum_passes") != 100
        or training.get("effective_batch_size") != 256
        or loss != {
            "kind": "constant_ce_kd_v1", "ce_weight": .25,
            "kd_weight": .75, "temperature": 2.0,
        }
        or recipe.get("rolling_resume") is not False
        or recipe.get("final_test_accessed") is not False
        or parents.get("graph") != graph_hash
        or parents.get("recipe") != recipe_hash
        or parents.get("foundation_spec") != foundation_hash
        or parents.get("foundation") != foundation_lock_hash
        or parents.get("assignment_lock")
        != foundation_lock.get("parents", {}).get("assignment_lock")
        or value.get("fresh_fit_count") != 29
        or value.get("reducer_count") != 25
        or value.get("only_changed_variable")
        != "particle_pairing_foundation_lineage"
        or value.get("population_policy") != "all_authenticated_mapped_rows_v1"
        or value.get("ordinary_access_roles") != ["train", "validation"]
        or value.get("ordinary_final_test_capability") is not False
        or value.get("existing_campaign_dependencies") != []
        or value.get("existing_campaign_outputs_mutated") is not False
        or value.get("ensembles") is not False
        or value.get("weight_continuation") is not False
        or value.get("immediate_parent_only") is not True
        or value.get("rolling_resume") is not False
        or value.get("partial_checkpoint_reuse") is not False
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("output-handoff non-persistent fullcard source differs")
    return digest, foundation, foundation_lock_hash, foundation_lock_path, graph_hash


def _report(path: str | Path, *, contract: str) -> tuple[Path, dict[str, Any], str]:
    resolved = Path(path).resolve()
    value = load_json(resolved)
    digest = validate_content_hash(
        value, expected_contract=contract,
        expected_schema_version=int(value.get("schema_version", -1)),
    )
    if value.get("final_test_accessed") is not False:
        raise PermissionError("output-handoff source report accessed final test")
    return resolved, value, digest


def build_source_lock(
    *, source_campaign_spec: str | Path, u100_training_report: str | Path,
    u100_selected_checkpoint: str | Path,
) -> dict[str, Any]:
    spec_path = Path(source_campaign_spec).resolve()
    spec = load_json(spec_path)
    (
        spec_hash, foundation, foundation_lock_hash,
        foundation_lock_path, source_graph_hash,
    ) = _validate_fullcard_campaign(spec_path, spec)
    report_path, report, report_hash = _report(
        u100_training_report, contract=FULLCARD_TRAINING_REPORT_CONTRACT,
    )
    node_id = str(report.get("node_id", ""))
    node = FULLCARD_NODES.get(node_id)
    checkpoint = Path(u100_selected_checkpoint).resolve()
    selected_name = str(report.get("selected_checkpoint", ""))
    expected_checkpoint = (report_path.parent / selected_name).resolve()
    if (
        node_id != SOURCE_U100_NODE_ID
        or node is None or node.coordinate_name != "U100"
        or report.get("complete") is not True
        or report.get("node_spec") != node.payload()
        or report.get("campaign_spec_sha256") != spec_hash
        or report.get("graph_sha256") != source_graph_hash
        or report.get("recipe_sha256") != spec["parents"]["recipe"]
        or report_path.parent.parent.parent.resolve() != Path(spec["campaign_root"]).resolve()
        or checkpoint != expected_checkpoint or not checkpoint.is_file()
        or sha256_file(checkpoint) != report.get("selected_checkpoint_sha256")
        or report.get("resume_policy") != "disabled_restart_from_zero_v1"
        or report.get("rolling_resume_published") is not False
    ):
        raise ValueError("output-handoff U100 source lineage differs")
    foundation_path = Path(spec["artifact_paths"]["foundation_spec"]).resolve()
    if not foundation_path.is_file() or not foundation_lock_path.is_file():
        raise ValueError("output-handoff full-cardinality foundation is incomplete")
    return artifact({
        "parents": {
            "source_campaign": spec_hash,
            "source_graph": spec["parents"]["graph"],
            "source_recipe": spec["parents"]["recipe"],
            "foundation": foundation_lock_hash,
            "foundation_spec": validate_content_hash(foundation),
            "assignment_lock": spec["parents"]["assignment_lock"],
            "u100_report": report_hash,
            "u100_checkpoint": report["selected_checkpoint_sha256"],
        },
        "source_campaign_spec_path": str(spec_path),
        "source_campaign_root": str(Path(spec["campaign_root"]).resolve()),
        "foundation_spec_path": str(foundation_path),
        "foundation_lock_path": str(foundation_lock_path),
        "u100_node_id": node_id, "u100_report_path": str(report_path),
        "u100_checkpoint_path": str(checkpoint),
        "u100_checkpoint_sha256": report["selected_checkpoint_sha256"],
        "replicate_seed": int(spec["replicate_seed"]),
        "role_counts": dict(spec["role_counts"]),
        "coordinate": "U100", "support_policy": DEFAULT_SUPPORT_POLICY,
        "source_campaign_family": SOURCE_CAMPAIGN_FAMILY,
        "operational_source_job_id": "98318",
        "operational_source_job_name": (
            "hcwsp4b_train_SP4_COARSE_U100_from_U050"
        ),
        "scientific_identity_uses_artifact_hashes_not_scheduler_id": True,
        "source_campaign_completion_required": False,
        "source_outputs_mutated": False, "final_test_accessed": False,
    }, contract=SOURCE_LOCK_CONTRACT)


def validate_source_lock(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=SOURCE_LOCK_CONTRACT)
    expected = build_source_lock(
        source_campaign_spec=value["source_campaign_spec_path"],
        u100_training_report=value["u100_report_path"],
        u100_selected_checkpoint=value["u100_checkpoint_path"],
    )
    if dict(value) != expected:
        raise ValueError("output-handoff source lock changed")
    return digest


def build_control_lock(
    *, m0ce60_training_report: str | Path,
    pure_offline_u000_training_report: str | Path,
) -> dict[str, Any]:
    m0_path, m0, m0_hash = _report(
        m0ce60_training_report, contract=CE60_TRAINING_REPORT_CONTRACT,
    )
    u000_path, u000, u000_hash = _report(
        pure_offline_u000_training_report, contract=TRI60_TRAINING_REPORT_CONTRACT,
    )
    m0_checkpoint = (m0_path.parent / str(m0.get("selected_checkpoint", ""))).resolve()
    u000_checkpoint = (u000_path.parent / str(u000.get("selected_checkpoint", ""))).resolve()
    u000_spec_path = (u000_path.parent.parent.parent / "campaign_spec.json").resolve()
    if not u000_spec_path.is_file():
        raise ValueError("output-handoff U000 control campaign spec is unavailable")
    u000_spec = load_json(u000_spec_path)
    u000_spec_hash = validate_content_hash(
        u000_spec, expected_contract=str(u000_spec.get("contract", "")),
        expected_schema_version=int(u000_spec.get("schema_version", -1)),
    )
    if (
        m0.get("node_id") != "M0CE60" or u000.get("node_id") != "U000"
        or not isinstance(m0.get("validation"), Mapping)
        or not isinstance(u000.get("validation"), Mapping)
        or not m0_checkpoint.is_file() or not u000_checkpoint.is_file()
        or sha256_file(m0_checkpoint) != m0.get("selected_checkpoint_sha256")
        or sha256_file(u000_checkpoint) != u000.get("selected_checkpoint_sha256")
        or u000.get("campaign_spec_sha256") != u000_spec_hash
    ):
        raise ValueError("output-handoff reporting controls differ")
    return artifact({
        "parents": {
            "m0ce60_report": m0_hash,
            "m0ce60_checkpoint": m0["selected_checkpoint_sha256"],
            "pure_offline_u000_report": u000_hash,
            "pure_offline_u000_checkpoint": u000["selected_checkpoint_sha256"],
            "pure_offline_u000_campaign": u000_spec_hash,
        },
        "m0ce60_report_path": str(m0_path),
        "m0ce60_checkpoint_path": str(m0_checkpoint),
        "pure_offline_u000_report_path": str(u000_path),
        "pure_offline_u000_checkpoint_path": str(u000_checkpoint),
        "pure_offline_u000_campaign_spec_path": str(u000_spec_path),
        "training_teacher_use": False, "reporting_only": True,
        "final_test_accessed": False,
    }, contract=CONTROL_LOCK_CONTRACT)


def validate_control_lock(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=CONTROL_LOCK_CONTRACT)
    expected = build_control_lock(
        m0ce60_training_report=value["m0ce60_report_path"],
        pure_offline_u000_training_report=value["pure_offline_u000_report_path"],
    )
    if dict(value) != expected:
        raise ValueError("output-handoff control lock changed")
    return digest


__all__ = [
    "FULLCARD_GRAPH_CONTRACT", "FULLCARD_RECIPE_CONTRACT",
    "FULLCARD_SPEC_CONTRACT",
    "FULLCARD_FINAL_CHECKPOINT_CONTRACT",
    "FULLCARD_SELECTED_CHECKPOINT_CONTRACT",
    "FULLCARD_TRAINING_REPORT_CONTRACT", "SOURCE_CAMPAIGN_FAMILY",
    "SOURCE_U100_NODE_ID",
    "build_control_lock", "build_source_lock", "validate_control_lock",
    "validate_source_lock",
]

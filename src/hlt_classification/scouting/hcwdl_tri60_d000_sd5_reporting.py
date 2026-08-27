"""Validation reporting for the TRI60 D000 SD5 ablation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import load_json, sha256_file

from .hcwdl_tri60_ce5_graph import NODE_REGISTRY as CE5_NODE_REGISTRY
from .hcwdl_tri60_d000_sd5_campaign import validate_campaign
from .hcwdl_tri60_d000_sd5_contracts import (
    AGGREGATE_CONTRACT, CAMPAIGN_COMPLETE_CONTRACT,
    ENSEMBLE_REPORT_CONTRACT, TRAINING_REPORT_CONTRACT,
    artifact, validate_artifact,
)
from .hcwdl_tri60_d000_sd5_graph import (
    ENSEMBLE_ID, FIT_ORDER, NODE_REGISTRY, SEED_MATCH, SOURCE_TEACHERS,
)
from .hcwdl_tri60_d000_sd5_runner import ensemble_report_path


def training_report(spec: Mapping[str, Any], node_id: str) -> dict[str, Any]:
    path = Path(spec["campaign_root"]) / "training" / node_id / "training_report.json"
    report = load_json(path)
    validate_artifact(report, contract=TRAINING_REPORT_CONTRACT)
    node = NODE_REGISTRY[node_id]
    source_teacher = str(node.distribution_teacher_id)
    if (
        report.get("node_id") != node_id
        or report.get("node_spec") != node.payload()
        or report.get("campaign_spec_sha256") != spec["content_hash"]
        or report.get("graph_sha256") != spec["parents"]["graph"]
        or report.get("recipe_sha256") != spec["parents"]["recipe"]
        or report.get("parents", {}).get("source_probability_lock")
        != spec["source_teacher_probability_locks"][source_teacher]
        or report.get("rng_domains", {}).get("node_seed_alias") != node.seed_alias
        or report.get("passes") != 60
        or report.get("validations") != 60
        or report.get("complete") is not True
        or report.get("rolling_resume_published") is not False
        or report.get("partial_checkpoint_reuse") is not False
        or report.get("final_test_accessed") is not False
    ):
        raise ValueError(f"TRI60 D000 SD5 training report differs: {node_id}")
    for name, hash_name in (
        ("selected_checkpoint", "selected_checkpoint_sha256"),
        ("final_checkpoint", "final_checkpoint_sha256"),
    ):
        checkpoint = path.parent / str(report.get(name, ""))
        if not checkpoint.is_file() or sha256_file(checkpoint) != report.get(hash_name):
            raise ValueError(f"TRI60 D000 SD5 checkpoint differs: {node_id}/{name}")
    return report


def ensemble_report(spec: Mapping[str, Any]) -> dict[str, Any]:
    report = load_json(ensemble_report_path(spec["campaign_root"]))
    validate_artifact(report, contract=ENSEMBLE_REPORT_CONTRACT)
    reports = {name: training_report(spec, name) for name in FIT_ORDER}
    source_stage = load_json(spec["artifact_paths"]["source_logit_d000e_stage"])
    ce5_stage = load_json(spec["artifact_paths"]["ce5_ensemble_report"])
    u000_stage = load_json(spec["artifact_paths"]["source_u000_stage"])
    expected_parents = {
        "campaign_spec": spec["content_hash"],
        "source_campaign": spec["parents"]["source_campaign"],
        "ce5_campaign": spec["parents"]["ce5_campaign"],
        "foundation": spec["parents"]["foundation"],
        "recipe": spec["parents"]["recipe"],
        "graph": spec["parents"]["graph"],
        "source_logit_d000e_stage": spec["parents"]["source_logit_d000e_stage"],
        "ce5_ensemble_report": spec["parents"]["ce5_ensemble_report"],
        **{
            f"component/{name}": report.get("component_lineage", {}).get(name, {}).get(
                "report_sha256"
            )
            for name in FIT_ORDER
        },
    }
    if (
        report.get("parents") != expected_parents
        or report.get("distribution_id") != ENSEMBLE_ID
        or report.get("component_order") != list(FIT_ORDER)
        or report.get("component_weights") != {name: .2 for name in FIT_ORDER}
        or set(report.get("component_lineage", {})) != set(FIT_ORDER)
        or set(report.get("component_metrics", {})) != set(FIT_ORDER)
        or any(
            report["component_lineage"][name].get("report_sha256")
            != reports[name]["content_hash"]
            or report["component_lineage"][name].get("checkpoint_sha256")
            != reports[name]["selected_checkpoint_sha256"]
            for name in FIT_ORDER
        )
        or report.get("comparators") != {
            "paired_seed_LOGIT_D000E": source_stage["ensemble_metrics"],
            "five_seed_CE5E": ce5_stage["ensemble_metrics"],
            "offline_U000": u000_stage["ensemble_metrics"],
        }
        or report.get("validation_rows") != int(spec["role_counts"]["validation"])
        or report.get("producer_commit") != spec["source_commit"]
        or report.get("persistent_probability_bank") is not False
        or report.get("persistent_logits") is not False
        or report.get("persistent_particle_views") is not False
        or report.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 D000 SD5 ensemble report differs")
    return report


def build_aggregate(spec: Mapping[str, Any]) -> dict[str, Any]:
    validate_campaign(spec, executable=False)
    reports = {node_id: training_report(spec, node_id) for node_id in FIT_ORDER}
    stage = ensemble_report(spec)
    rows = [
        {
            "row_id": "U000", "kind": "offline_reference",
            "metrics": stage["comparators"]["offline_U000"],
        },
        {
            "row_id": "LOGIT_D000E", "kind": "paired_seed_kd_ensemble",
            "metrics": stage["comparators"]["paired_seed_LOGIT_D000E"],
        },
        {
            "row_id": "CE5E", "kind": "five_seed_ce_ensemble",
            "metrics": stage["comparators"]["five_seed_CE5E"],
        },
    ]
    for node_id in FIT_ORDER:
        node = NODE_REGISTRY[node_id]
        report = reports[node_id]
        rows.append({
            "row_id": node_id, "kind": "matched_seed_kd_specialist",
            "metrics": stage["component_metrics"][node_id],
            "checkpoint_selection_metrics": report["validation"],
            "source_teacher_id": node.distribution_teacher_id,
            "ce5_seed_source_id": SEED_MATCH[str(node.distribution_teacher_id)],
            "seed_alias": node.seed_alias,
            "selected_pass": report["selected_pass"],
            "selected_update": report["selected_update"],
            "runtime_seconds": float(report["runtime_seconds"]),
            "preparation_seconds": dict(report.get("preparation_seconds", {})),
        })
    rows.append({
        "row_id": ENSEMBLE_ID, "kind": "matched_seed_kd_ensemble",
        "metrics": stage["ensemble_metrics"],
        "component_order": list(FIT_ORDER),
        "component_weights": {name: .2 for name in FIT_ORDER},
        "runtime_seconds": float(stage["runtime_seconds"]),
    })
    component_aucs = np.asarray([
        stage["component_metrics"][name]["macro_ovr_auc"] for name in FIT_ORDER
    ], dtype=np.float64)
    durable_bytes = sum(
        path.stat().st_size
        for path in Path(spec["campaign_root"]).rglob("*") if path.is_file()
    )
    return artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "source_campaign": spec["parents"]["source_campaign"],
            "ce5_campaign": spec["parents"]["ce5_campaign"],
            "foundation": spec["parents"]["foundation"],
            "recipe": spec["parents"]["recipe"],
            "graph": spec["parents"]["graph"],
            "ensemble_report": stage["content_hash"],
        },
        "primary_question": "does_matched_stochastic_diversity_improve_logit_d000e",
        "primary_metric": "validation_macro_ovr_auc",
        "rows": rows, "comparisons": stage["comparisons"],
        "component_summary": {
            "count": 5,
            "macro_ovr_auc_mean": float(component_aucs.mean()),
            "macro_ovr_auc_sample_std": float(component_aucs.std(ddof=1)),
            "macro_ovr_auc_min": float(component_aucs.min()),
            "macro_ovr_auc_max": float(component_aucs.max()),
        },
        "matched_seed_audit": {
            "source_teacher_order": list(SOURCE_TEACHERS),
            "fit_order": list(FIT_ORDER),
            "ce5_seed_match": dict(SEED_MATCH),
            "fit_seed_aliases": {
                name: NODE_REGISTRY[name].seed_alias for name in FIT_ORDER
            },
            "ce5_seed_aliases": {
                teacher: CE5_NODE_REGISTRY[SEED_MATCH[teacher]].seed_alias
                for teacher in SOURCE_TEACHERS
            },
            "all_five_initialization_and_sampler_domains_distinct": True,
            "same_five_seed_domains_as_CE5E": True,
        },
        "fit_count": 5, "reducer_count": 1,
        "durable_campaign_bytes_at_aggregate": durable_bytes,
        "durable_probability_bank_bytes": 0,
        "durable_logits_bytes": 0,
        "durable_particle_view_bytes": 0,
        "rolling_resume_durable_bytes": 0,
        "one_five_seed_experiment": True,
        "scientific_result_does_not_control_completion": True,
        "final_test_accessed": False,
    }, contract=AGGREGATE_CONTRACT)


def build_campaign_complete(
    spec: Mapping[str, Any], aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    validate_campaign(spec, executable=False)
    validate_artifact(aggregate, contract=AGGREGATE_CONTRACT)
    if aggregate.get("parents", {}).get("campaign_spec") != spec["content_hash"]:
        raise ValueError("TRI60 D000 SD5 aggregate belongs to another campaign")
    return artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "aggregate": aggregate["content_hash"],
        },
        "fresh_fit_count": 5, "reducer_count": 1,
        "ordinary_access_roles": ["train", "validation"],
        "final_test_accessed": False,
        "scientific_result_does_not_control_completion": True,
    }, contract=CAMPAIGN_COMPLETE_CONTRACT)


__all__ = [
    "build_aggregate", "build_campaign_complete", "ensemble_report",
    "training_report",
]

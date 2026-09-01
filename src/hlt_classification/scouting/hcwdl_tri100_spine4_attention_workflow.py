"""Fail-closed dispatch and reporting for attention-reoptimized four spines."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, validate_content_hash, write_immutable_json,
)

from .hcwdl_attention_reoptimization import DEFAULT_ATTENTION_RECIPE
from .hcwdl_tri100_spine4_attention_campaign import validate_campaign
from .hcwdl_tri100_spine4_attention_contracts import (
    AGGREGATE_CONTRACT, COMPLETE_CONTRACT, SCHEMA_VERSION,
    STAGE_REPORT_CONTRACT, TRAINING_REPORT_CONTRACT, artifact,
    validate_artifact,
)
from .hcwdl_tri100_spine4_attention_execution import (
    run_attention_execution_acceptance,
    validate_attention_execution_acceptance, validate_parameter_lock,
)
from .hcwdl_tri100_spine4_attention_graph import (
    ANCHOR_NODE_ID, BRANCH_NODES, BRANCH_ORDER, DOWNSTREAM_FIT_ORDER,
    ENDPOINT_NODES, EXECUTION, FIT_ORDER, GRAPH_SHA256, NODE_REGISTRY,
    PROBABILITY_COMPONENTS, REDUCER_ORDER, RELATIONAL_CARRIERS,
)
from .hcwdl_tri100_spine4_attention_runner import run_fit, run_reducer
from .hcwdl_tri100_spine4_bottleneck_source import validate_source_lock
from .hcwdl_tri100_spine4_bottleneck_workflow import (
    _delta, _matching_diagnostics, _recovery,
)
from .hcwdl_tri100_spine4_bottleneck_workflow import (
    _training_report as _persistent_training_report,
)
from .hcwdl_tri100_spine4_persistent_support import (
    build_support_audit, validate_support_audit,
)


def _training_report(spec: Mapping[str, Any], node_id: str) -> dict[str, Any]:
    path = Path(spec["campaign_root"]) / "training" / node_id / "training_report.json"
    report = load_json(path)
    validate_content_hash(
        report, expected_contract=TRAINING_REPORT_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    selected = path.parent / str(report.get("selected_checkpoint", ""))
    final = path.parent / str(report.get("final_checkpoint", ""))
    node = NODE_REGISTRY[node_id]
    acceptance_hash = validate_attention_execution_acceptance(
        load_json(spec["artifact_paths"]["execution_acceptance"]), spec=spec,
    )
    parameter_hash = validate_parameter_lock(
        load_json(spec["artifact_paths"]["parameter_lock"]), spec=spec,
    )
    parents = report.get("parents", {})
    common_invalid = (
        report.get("node_id") != node_id
        or report.get("node_spec") != node.payload()
        or report.get("campaign_spec_sha256") != spec["content_hash"]
        or report.get("graph_sha256") != GRAPH_SHA256
        or report.get("recipe_sha256") != spec["parents"]["recipe"]
        or report.get("validations") != report.get("passes")
        or report.get("complete") is not True
        or report.get("rolling_resume_published") is not False
        or report.get("partial_checkpoint_reuse") is not False
        or parents.get("execution_acceptance") != acceptance_hash
        or parents.get("attention_parameter_lock") != parameter_hash
        or parents.get("foundation") != spec["parents"]["foundation"]
        or report.get("distributed_execution") is not None
        or report.get("throughput_optimizations", {}).get(
            "synchronous_data_parallel_world_size"
        ) != 1
        or report.get("final_test_accessed") is not False
        or not selected.is_file() or not final.is_file()
        or sha256_file(selected) != report.get("selected_checkpoint_sha256")
        or sha256_file(final) != report.get("final_checkpoint_sha256")
    )
    if node_id == ANCHOR_NODE_ID:
        stage_invalid = (
            report.get("passes") != 60
            or report.get("attention_reoptimization") is not None
            or report.get("attention_stage_history") is not None
        )
    else:
        history = report.get("validation_history", [])
        stage_invalid = (
            report.get("passes") != 100
            or report.get("performance_early_termination") is not False
            or report.get("attention_reoptimization")
            != DEFAULT_ATTENTION_RECIPE.payload()
            or report.get("attention_parameter_registry_sha256")
            != load_json(spec["artifact_paths"]["parameter_lock"])[
                "registry_sha256"
            ]
            or report.get("dense_attention_target_durable_bytes") != 0
            or report.get("relational_target_generation")
            != "same_job_per_batch_eval_no_grad_v1"
            or report.get("selected_attention_stage")
            not in {"stage0", "stage_a", "stage_b"}
            or [row.get("attention_stage") for row in history[:60]]
            != ["stage0"] * 60
            or [row.get("attention_stage") for row in history[60:75]]
            != ["stage_a"] * 15
            or [row.get("attention_stage") for row in history[75:]]
            != ["stage_b"] * 25
            or parents.get("relational_carrier_report")
            != parents.get("teacher_report")
            or parents.get("relational_carrier_checkpoint")
            != parents.get("teacher_checkpoint")
        )
    if common_invalid or stage_invalid:
        raise ValueError(f"attention four-spine training report differs: {node_id}")
    return report


def task_outputs(spec: Mapping[str, Any], task_id: str) -> list[Path]:
    tasks = {row["task_id"]: row for row in spec["tasks"]}
    if task_id not in tasks:
        raise KeyError("unknown attention four-spine task")
    task = tasks[task_id]
    root = Path(spec["campaign_root"])
    if task["kind"] == "authenticate":
        return [Path(spec["artifact_paths"]["source_lock"])]
    if task["kind"] == "support_audit":
        return [Path(spec["artifact_paths"]["support_audit"])]
    if task["kind"] == "preflight":
        return [
            Path(spec["artifact_paths"]["parameter_lock"]),
            Path(spec["artifact_paths"]["execution_acceptance"]),
        ]
    if task["kind"] == "train":
        report = _training_report(spec, task["node_id"])
        directory = root / "training" / task["node_id"]
        return [
            directory / "training_report.json",
            directory / report["selected_checkpoint"],
            directory / report["final_checkpoint"],
        ]
    if task["kind"] == "reducer":
        distribution = task["distribution_id"]
        directory = root / "probabilities" / distribution
        outputs = [
            root / "reports/stages" / f"{distribution}.json",
            directory / "lock.json",
        ]
        for role in ("train", "validation"):
            outputs.extend((
                directory / f"{role}.npz", directory / f"{role}_shard.json",
                directory / f"{role}_manifest.json",
            ))
        return outputs
    if task["kind"] == "aggregate":
        return [root / "reports/validation_aggregate.json"]
    return [root / "reports/campaign_complete.json"]


def build_aggregate(spec: Mapping[str, Any]) -> dict[str, Any]:
    validate_campaign(spec)
    source = load_json(spec["artifact_paths"]["source_lock"])
    validate_source_lock(source)
    acceptance_hash = validate_attention_execution_acceptance(
        load_json(spec["artifact_paths"]["execution_acceptance"]), spec=spec,
    )
    parameter_hash = validate_parameter_lock(
        load_json(spec["artifact_paths"]["parameter_lock"]), spec=spec,
    )
    support_hash = validate_support_audit(
        load_json(spec["artifact_paths"]["support_audit"]), spec=spec,
    )
    oracle_report = load_json(source["u000"]["report_path"])
    baseline_report = load_json(spec["artifact_paths"]["m0ce60_report"])
    oracle = oracle_report["validation"]
    baseline = baseline_report["validation"]
    parents = {
        "campaign_spec": spec["content_hash"], "graph": GRAPH_SHA256,
        "source_campaign": spec["parents"]["source_campaign"],
        "source_u000_report": source["u000"]["report_sha256"],
        "m0ce60_report": spec["parents"]["m0ce60_report"],
        "persistent_campaign": spec["parents"]["persistent_campaign"],
        "execution_acceptance": acceptance_hash,
        "attention_parameter_lock": parameter_hash,
        "support_audit": support_hash,
    }
    rows = [{
        "artifact_id": "M0CE60", "kind": "baseline", "state": "complete",
        "metrics": baseline, "recovery": _recovery(baseline, baseline, oracle),
        "report_sha256": baseline_report["content_hash"],
    }, {
        "artifact_id": "U000", "kind": "pure_offline_oracle",
        "state": "complete", "metrics": oracle,
        "recovery": _recovery(oracle, baseline, oracle),
        "report_sha256": oracle_report["content_hash"],
    }]
    new_by_id = {}
    for node_id in FIT_ORDER:
        node = NODE_REGISTRY[node_id]
        report = _training_report(spec, node_id)
        parents[f"attention_fit/{node_id}"] = report["content_hash"]
        parent_metrics = oracle if node.parent_node_id is None else new_by_id[
            node.parent_node_id
        ]["metrics"]
        row = {
            "artifact_id": node_id,
            "kind": (
                "persistent_hlt_hybrid_anchor"
                if node_id == ANCHOR_NODE_ID else "attention_reoptimized_fit"
            ),
            "state": "complete", "branch": node.branch,
            "path_index": node.path_index, "coordinate": node.coordinate_name,
            "parent_node_id": node.parent_node_id,
            "relational_carrier_id": RELATIONAL_CARRIERS.get(node_id),
            "metrics": report["validation"],
            "recovery": _recovery(report["validation"], baseline, oracle),
            "branch_edge_delta": _delta(report["validation"], parent_metrics),
            "report_sha256": report["content_hash"],
            "selected_pass": int(report["selected_pass"]),
            "completed_passes": int(report["passes"]),
            "selected_attention_stage": report.get("selected_attention_stage"),
            "stage0_validation": report.get("attention_stage0_validation"),
        }
        new_by_id[node_id] = row
        rows.append(row)
    for distribution in REDUCER_ORDER:
        stage = load_json(
            Path(spec["campaign_root"]) / "reports/stages" / f"{distribution}.json"
        )
        validate_artifact(stage, contract=STAGE_REPORT_CONTRACT)
        if (
            stage.get("distribution_id") != distribution
            or stage.get("component_order")
            != list(PROBABILITY_COMPONENTS[distribution])
            or stage.get("durable_attention_targets") is not False
        ):
            raise ValueError("attention aggregate probability stage differs")
        parents[f"probability/{distribution}"] = stage["content_hash"]

    persistent = load_json(spec["artifact_paths"]["persistent_campaign_spec"])
    controls = []
    matched = []
    for node_id in FIT_ORDER:
        path = Path(persistent["campaign_root"]) / "training" / node_id / "training_report.json"
        if not path.is_file():
            controls.append({
                "artifact_id": node_id, "kind": "persistent_control",
                "state": "pending", "metrics": None,
            })
            continue
        report = _persistent_training_report(persistent, node_id)
        parents[f"persistent_fit/{node_id}"] = report["content_hash"]
        controls.append({
            "artifact_id": node_id, "kind": "persistent_control",
            "state": "complete", "metrics": report["validation"],
            "report_sha256": report["content_hash"],
        })
        matched.append({
            "artifact_id": node_id,
            "attention_minus_persistent": _delta(
                new_by_id[node_id]["metrics"], report["validation"],
            ),
        })
    assignment_hash, diagnostics = _matching_diagnostics(spec)
    parents["assignment_lock"] = assignment_hash
    endpoints = [{
        "branch": branch, "node_id": node_id,
        "metrics": new_by_id[node_id]["metrics"],
        "recovery": new_by_id[node_id]["recovery"],
        "report_sha256": new_by_id[node_id]["report_sha256"],
    } for branch, node_id in zip(BRANCH_ORDER, ENDPOINT_NODES, strict=True)]
    return artifact({
        "parents": dict(sorted(parents.items())),
        "rows": rows + controls, "endpoints": endpoints,
        "matched_rung_differences": matched,
        "matching_diagnostics": diagnostics,
        "persistent_rows_complete": sum(row["state"] == "complete" for row in controls),
        "persistent_rows_pending": sum(row["state"] == "pending" for row in controls),
        "persistent_rows_do_not_block": True,
        "recovery_convention": "M0CE60_zero_U000_one_v1",
        "fresh_fit_count": len(FIT_ORDER), "reducer_count": len(REDUCER_ORDER),
        "attention_reoptimization_at_every_downstream_rung": True,
        "ram_only_dense_relational_targets": True,
        "dense_relational_target_durable_bytes": 0,
        "execution": dict(EXECUTION), "ensembles": False,
        "poor_metrics_do_not_control_completion": True,
        "source_campaign_outputs_mutated": False,
        "ordinary_access_roles": ["validation"],
        "final_test_accessed": False,
    }, contract=AGGREGATE_CONTRACT)


class AttentionSpine4Workflow:
    def __init__(
        self, spec: Mapping[str, Any], *, recovery_spec_sha256: str | None = None,
        execution_source_commit: str | None = None,
    ) -> None:
        validate_campaign(spec)
        self.spec = spec
        self.root = Path(spec["campaign_root"])
        self.recovery_spec_sha256 = recovery_spec_sha256
        self.execution_source_commit = execution_source_commit
        self.tasks = {row["task_id"]: row for row in spec["tasks"]}

    def run(self, task_id: str, *, device: str = "cuda") -> dict[str, Any]:
        if task_id not in self.tasks:
            raise KeyError("unknown attention four-spine task")
        task = self.tasks[task_id]
        kind = task["kind"]
        if kind == "authenticate":
            source = load_json(self.spec["artifact_paths"]["source_lock"])
            if validate_source_lock(source) != self.spec["parents"]["source_lock"]:
                raise ValueError("attention source lock changed")
            return source
        if kind == "support_audit":
            value = build_support_audit(self.spec)
            write_immutable_json(self.spec["artifact_paths"]["support_audit"], value)
            return value
        if kind == "preflight":
            validate_support_audit(
                load_json(self.spec["artifact_paths"]["support_audit"]),
                spec=self.spec,
            )
            required = int(self.spec["minimum_free_disk_bytes"]) + int(
                self.spec["projected_durable_bytes"]
            )
            if shutil.disk_usage(self.root).free < required:
                raise OSError("attention campaign free disk cannot preserve reserve")
            if tuple(self.root.rglob("*resume*")):
                raise PermissionError("attention campaign contains resume state")
            parameter, acceptance = run_attention_execution_acceptance(
                spec=self.spec,
                source_commit=(
                    self.execution_source_commit or self.spec["source_commit"]
                ),
                device=device,
            )
            write_immutable_json(self.spec["artifact_paths"]["parameter_lock"], parameter)
            write_immutable_json(
                self.spec["artifact_paths"]["execution_acceptance"], acceptance,
            )
            return acceptance
        if kind == "train":
            return run_fit(
                spec=self.spec, node_id=task["node_id"], device=device,
                recovery_spec_sha256=self.recovery_spec_sha256,
                execution_source_commit=self.execution_source_commit,
            )
        if kind == "reducer":
            return run_reducer(
                spec=self.spec, distribution_id=task["distribution_id"],
                device=device, recovery_spec_sha256=self.recovery_spec_sha256,
                execution_source_commit=self.execution_source_commit,
            )
        if kind == "aggregate":
            value = build_aggregate(self.spec)
            write_immutable_json(self.root / "reports/validation_aggregate.json", value)
            return value
        if kind == "campaign_complete":
            aggregate = load_json(self.root / "reports/validation_aggregate.json")
            durable = tuple(path for path in self.root.rglob("*") if path.is_file())
            forbidden = [
                str(path.relative_to(self.root)) for path in durable
                if "resume" in path.name.lower()
                or "attention_target" in path.name.lower()
                or "block_delta" in path.name.lower()
            ]
            if forbidden:
                raise PermissionError(
                    f"attention campaign contains forbidden durable state: {forbidden}"
                )
            durable_bytes = sum(path.stat().st_size for path in durable)
            if durable_bytes > int(self.spec["projected_durable_bytes"]):
                raise OSError("attention campaign durable bytes exceed projection")
            complete = artifact({
                "parents": {
                    "campaign_spec": self.spec["content_hash"],
                    "aggregate": validate_artifact(
                        aggregate, contract=AGGREGATE_CONTRACT,
                    ),
                },
                "fresh_fit_count": len(FIT_ORDER),
                "reducer_count": len(REDUCER_ORDER),
                "branch_order": list(BRANCH_ORDER),
                "endpoint_nodes": list(ENDPOINT_NODES),
                "rolling_resume_durable_bytes": 0,
                "optimizer_state_durable_bytes": 0,
                "dense_relational_target_durable_bytes": 0,
                "all_registered_science_executed": True,
                "scientific_result_does_not_control_completion": True,
                "persistent_rows_do_not_control_completion": True,
                "source_campaign_outputs_mutated": False,
                "final_test_accessed": False,
            }, contract=COMPLETE_CONTRACT)
            write_immutable_json(self.root / "reports/campaign_complete.json", complete)
            return complete
        raise RuntimeError(f"unhandled attention task: {kind}")


__all__ = [
    "AttentionSpine4Workflow", "_training_report", "build_aggregate",
    "task_outputs",
]

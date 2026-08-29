"""Fail-closed task dispatch and controlled old/new reporting."""

from __future__ import annotations

import math
from pathlib import Path
import shutil
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, validate_content_hash, write_immutable_json,
)

from .hcwdl_fullcard_bottleneck_contracts import (
    ASSIGNMENT_LOCK_CONTRACT, DIAGNOSTIC_REPORT_CONTRACT, SCHEMA_VERSION,
)
from .hcwdl_tri100_spine4_bottleneck_campaign import validate_campaign
from .hcwdl_tri100_spine4_bottleneck_contracts import (
    AGGREGATE_CONTRACT, COMPLETE_CONTRACT, STAGE_REPORT_CONTRACT,
    TRAINING_REPORT_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_tri100_spine4_bottleneck_execution import (
    run_execution_acceptance, validate_execution_acceptance,
)
from .hcwdl_tri100_spine4_bottleneck_graph import (
    BRANCH_NODES, BRANCH_ORDER, ENDPOINT_NODES, EXECUTION, FIT_ORDER,
    GRAPH_SHA256, NODE_REGISTRY, PROBABILITY_COMPONENTS, REDUCER_ORDER,
)
from .hcwdl_tri100_spine4_bottleneck_runner import run_fit, run_reducer
from .hcwdl_tri100_spine4_bottleneck_source import validate_source_lock
from .hcwdl_tri100_spine4_workflow import _training_report as _established_report


def _training_report(spec: Mapping[str, Any], node_id: str) -> dict[str, Any]:
    path = Path(spec["campaign_root"]) / "training" / node_id / "training_report.json"
    report = load_json(path)
    validate_content_hash(
        report, expected_contract=TRAINING_REPORT_CONTRACT,
        expected_schema_version=1,
    )
    selected = path.parent / str(report.get("selected_checkpoint", ""))
    final = path.parent / str(report.get("final_checkpoint", ""))
    completed_passes = int(report.get("passes", -1))
    stopped_early = completed_passes < 100
    acceptance = load_json(spec["artifact_paths"]["execution_acceptance"])
    acceptance_hash = validate_execution_acceptance(acceptance, spec=spec)
    parents = report.get("parents", {})
    if (
        report.get("node_id") != node_id
        or report.get("node_spec") != NODE_REGISTRY[node_id].payload()
        or report.get("campaign_spec_sha256") != spec["content_hash"]
        or report.get("graph_sha256") != GRAPH_SHA256
        or report.get("recipe_sha256") != spec["parents"]["recipe"]
        or not 60 <= completed_passes <= 100
        or report.get("validations") != completed_passes
        or report.get("maximum_passes") != 100
        or report.get("minimum_passes") != 60
        or report.get("stopped_early") is not stopped_early
        or report.get("performance_early_termination") is not stopped_early
        or report.get("stop_reason") != (
            "macro_auc_patience_exhausted" if stopped_early
            else "maximum_passes_reached"
        )
        or report.get("complete") is not True
        or report.get("rolling_resume_published") is not False
        or report.get("partial_checkpoint_reuse") is not False
        or parents.get("execution_acceptance") != acceptance_hash
        or parents.get("foundation") != spec["parents"]["foundation"]
        or parents.get("assignment_lock") != spec["parents"]["assignment_lock"]
        or parents.get("matcher_spec") != spec["parents"]["matcher_spec"]
        or report.get("distributed_execution") is not None
        or report.get("throughput_optimizations", {}).get(
            "synchronous_data_parallel_world_size"
        ) != 1
        or report.get("final_test_accessed") is not False
        or not selected.is_file() or not final.is_file()
        or sha256_file(selected) != report.get("selected_checkpoint_sha256")
        or sha256_file(final) != report.get("final_checkpoint_sha256")
    ):
        raise ValueError(f"bottleneck training report differs: {node_id}")
    return report


def task_outputs(spec: Mapping[str, Any], task_id: str) -> list[Path]:
    tasks = {row["task_id"]: row for row in spec["tasks"]}
    if task_id not in tasks:
        raise KeyError("unknown bottleneck four-spine task")
    task = tasks[task_id]
    root = Path(spec["campaign_root"])
    if task["kind"] == "authenticate":
        return [Path(spec["artifact_paths"]["source_lock"])]
    if task["kind"] == "preflight":
        return [Path(spec["artifact_paths"]["execution_acceptance"])]
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


def _linear_r50(metrics: Mapping[str, Any]) -> float | None:
    value = metrics.get("macro_mean_log_qcd_rejection_at_50pct_signal")
    return None if value is None else math.exp(float(value))


def _ratio(value: float | None, baseline: float | None, oracle: float | None):
    if value is None or baseline is None or oracle is None or oracle == baseline:
        return None
    return (value - baseline) / (oracle - baseline)


def _recovery(
    metrics: Mapping[str, Any], baseline: Mapping[str, Any], oracle: Mapping[str, Any],
) -> dict[str, Any]:
    auc = _ratio(
        metrics.get("macro_ovr_auc"), baseline.get("macro_ovr_auc"),
        oracle.get("macro_ovr_auc"),
    )
    r50 = _ratio(_linear_r50(metrics), _linear_r50(baseline), _linear_r50(oracle))
    per_class = {}
    names = sorted(
        set(baseline.get("per_class", {}))
        & set(oracle.get("per_class", {}))
        & set(metrics.get("per_class", {}))
    )
    for name in names:
        def rejection(source):
            return source.get("per_class", {}).get(name, {}).get(
                "qcd_rejection", {}
            ).get("50pct", {}).get("rejection")
        value = rejection(metrics)
        if value is not None:
            per_class[name] = {
                "qcd_rejection_at_50pct_signal": float(value),
                "linear_recovery": _ratio(
                    float(value), rejection(baseline), rejection(oracle),
                ),
            }
    return {
        "convention": "M0CE60_zero_U000_one_v1",
        "macro_ovr_auc": auc, "macro_r50_linear": r50,
        "per_class_qcd_rejection_at_50pct_signal": per_class,
    }


def _delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": float(left["accuracy"]) - float(right["accuracy"]),
        "macro_ovr_auc": float(left["macro_ovr_auc"]) - float(right["macro_ovr_auc"]),
        "macro_r50_linear": _linear_r50(left) - _linear_r50(right),
    }


def _matching_diagnostics(spec: Mapping[str, Any]):
    foundation = load_json(spec["artifact_paths"]["foundation_spec"])
    assignment = load_json(Path(foundation["campaign_root"]) / "locks/assignment.json")
    assignment_hash = validate_content_hash(
        assignment, expected_contract=ASSIGNMENT_LOCK_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    reports = {}
    for role in ("train", "validation"):
        report = load_json(Path(foundation["campaign_root"]) / f"matcher/{role}_diagnostics.json")
        digest = validate_content_hash(
            report, expected_contract=DIAGNOSTIC_REPORT_CONTRACT,
            expected_schema_version=SCHEMA_VERSION,
        )
        if assignment["role_diagnostics"].get(role) != digest:
            raise ValueError("bottleneck diagnostic/assignment lock differs")
        reports[role] = {
            "report_sha256": digest, "summary": report["summary"],
            "claim_boundary": report["claim_boundary"],
        }
    return assignment_hash, reports


def build_aggregate(spec: Mapping[str, Any]) -> dict[str, Any]:
    validate_campaign(spec)
    source = load_json(spec["artifact_paths"]["source_lock"])
    validate_source_lock(source)
    acceptance = load_json(spec["artifact_paths"]["execution_acceptance"])
    acceptance_hash = validate_execution_acceptance(acceptance, spec=spec)
    source_report = load_json(source["u000"]["report_path"])
    baseline_report = load_json(spec["artifact_paths"]["m0ce60_report"])
    baseline = baseline_report["validation"]
    oracle = source_report["validation"]
    parents = {
        "campaign_spec": spec["content_hash"], "graph": GRAPH_SHA256,
        "source_campaign": spec["parents"]["source_campaign"],
        "source_u000_report": source["u000"]["report_sha256"],
        "m0ce60_report": spec["parents"]["m0ce60_report"],
        "established_campaign": spec["parents"]["established_campaign"],
        "execution_acceptance": acceptance_hash,
    }
    rows = [{
        "artifact_id": "M0CE60", "matcher": "shared", "kind": "baseline",
        "state": "complete", "metrics": baseline,
        "recovery": _recovery(baseline, baseline, oracle),
        "report_sha256": baseline_report["content_hash"],
    }, {
        "artifact_id": "U000", "matcher": "shared", "kind": "source_anchor",
        "state": "complete", "metrics": oracle,
        "recovery": _recovery(oracle, baseline, oracle),
        "report_sha256": source_report["content_hash"],
        "selected_checkpoint_sha256": source_report["selected_checkpoint_sha256"],
        "completed_passes": int(source_report["passes"]),
    }]
    new_by_id = {}
    for node_id in FIT_ORDER:
        node = NODE_REGISTRY[node_id]
        report = _training_report(spec, node_id)
        parents[f"new_fit/{node_id}"] = report["content_hash"]
        parent_metrics = oracle if node.parent_node_id is None else new_by_id[
            node.parent_node_id
        ]["metrics"]
        row = {
            "artifact_id": node_id, "matcher": "fullcard_bottleneck",
            "kind": "fit", "state": "complete", "branch": node.branch,
            "path_index": node.path_index, "coordinate": node.coordinate_name,
            "parent_node_id": node.parent_node_id,
            "metrics": report["validation"],
            "recovery": _recovery(report["validation"], baseline, oracle),
            "branch_edge_delta": _delta(report["validation"], parent_metrics),
            "report_sha256": report["content_hash"],
            "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
            "selected_pass": int(report["selected_pass"]),
            "completed_passes": int(report["passes"]),
            "stopped_early": bool(report["stopped_early"]),
        }
        new_by_id[node_id] = row
        rows.append(row)
    for distribution_id in REDUCER_ORDER:
        stage = load_json(
            Path(spec["campaign_root"]) / "reports/stages" / f"{distribution_id}.json"
        )
        validate_artifact(stage, contract=STAGE_REPORT_CONTRACT)
        if (
            stage.get("distribution_id") != distribution_id
            or stage.get("component_order") != list(PROBABILITY_COMPONENTS[distribution_id])
            or stage.get("single_component_selected_checkpoint") is not True
        ):
            raise ValueError("bottleneck aggregate stage differs")
        parents[f"probability/{distribution_id}"] = stage["content_hash"]

    established = load_json(spec["artifact_paths"]["established_campaign_spec"])
    established_rows = []
    matched_differences = []
    for node_id in FIT_ORDER:
        path = Path(established["campaign_root"]) / "training" / node_id / "training_report.json"
        if not path.is_file():
            established_rows.append({
                "artifact_id": node_id, "matcher": "established_high_coverage",
                "kind": "fit", "state": "pending", "metrics": None,
            })
            continue
        report = _established_report(established, node_id)
        parents[f"established_fit/{node_id}"] = report["content_hash"]
        old_metrics = report["validation"]
        established_rows.append({
            "artifact_id": node_id, "matcher": "established_high_coverage",
            "kind": "fit", "state": "complete", "metrics": old_metrics,
            "recovery": _recovery(old_metrics, baseline, oracle),
            "report_sha256": report["content_hash"],
            "selected_pass": int(report["selected_pass"]),
            "completed_passes": int(report["passes"]),
        })
        matched_differences.append({
            "artifact_id": node_id, "new_minus_established": _delta(
                new_by_id[node_id]["metrics"], old_metrics,
            ),
        })
    rows.extend(established_rows)
    assignment_hash, diagnostics = _matching_diagnostics(spec)
    parents["assignment_lock"] = assignment_hash
    for role, report in diagnostics.items():
        parents[f"matching_diagnostics/{role}"] = report["report_sha256"]
    endpoints = [{
        "branch": branch, "node_id": node_id,
        "path": ["U000", *(NODE_REGISTRY[name].coordinate_name for name in BRANCH_NODES[branch])],
        "metrics": new_by_id[node_id]["metrics"],
        "recovery": new_by_id[node_id]["recovery"],
        "selected_pass": new_by_id[node_id]["selected_pass"],
        "completed_passes": new_by_id[node_id]["completed_passes"],
        "report_sha256": new_by_id[node_id]["report_sha256"],
    } for branch, node_id in zip(BRANCH_ORDER, ENDPOINT_NODES, strict=True)]
    return artifact({
        "parents": dict(sorted(parents.items())), "rows": rows,
        "endpoints": endpoints, "matched_rung_differences": matched_differences,
        "matching_diagnostics": diagnostics,
        "matching_diagnostic_lock_sha256": assignment_hash,
        "established_rows_complete": sum(
            row["state"] == "complete" for row in established_rows
        ),
        "established_rows_pending": sum(
            row["state"] == "pending" for row in established_rows
        ),
        "established_rows_do_not_block": True,
        "recovery_convention": "M0CE60_zero_U000_one_v1",
        "fresh_fit_count": len(FIT_ORDER), "reducer_count": len(REDUCER_ORDER),
        "source_fit_reuse_count": 1, "endpoint_seed_matched": True,
        "immediate_parent_only": True, "execution": dict(EXECUTION),
        "ensembles": False, "poor_metrics_do_not_control_completion": True,
        "source_campaign_outputs_mutated": False,
        "ordinary_access_roles": ["validation"], "final_test_accessed": False,
    }, contract=AGGREGATE_CONTRACT)


class BottleneckSpine4Workflow:
    def __init__(
        self, spec: Mapping[str, Any], *, recovery_spec_sha256: str | None = None,
        execution_source_commit: str | None = None,
    ) -> None:
        validate_campaign(spec)
        self.spec = spec; self.root = Path(spec["campaign_root"])
        self.recovery_spec_sha256 = recovery_spec_sha256
        self.execution_source_commit = execution_source_commit
        self.tasks = {row["task_id"]: row for row in spec["tasks"]}

    def _authenticate(self) -> dict[str, Any]:
        source = load_json(self.spec["artifact_paths"]["source_lock"])
        if validate_source_lock(source) != self.spec["parents"]["source_lock"]:
            raise ValueError("bottleneck source lock changed")
        return source

    def _preflight(self, *, device: str) -> dict[str, Any]:
        self._authenticate()
        required = int(self.spec["minimum_free_disk_bytes"]) + int(
            self.spec["projected_durable_bytes"]
        )
        if shutil.disk_usage(self.root).free < required:
            raise OSError("bottleneck free disk cannot preserve reserve after projection")
        if tuple(self.root.rglob("*resume*")):
            raise PermissionError("bottleneck four-spine contains resume state")
        if self.spec.get("ordinary_final_test_capability") is not False:
            raise PermissionError("bottleneck four-spine has final-test capability")
        acceptance = run_execution_acceptance(
            spec=self.spec,
            source_commit=self.execution_source_commit or self.spec["source_commit"],
            device=device,
        )
        write_immutable_json(self.spec["artifact_paths"]["execution_acceptance"], acceptance)
        return acceptance

    def run(self, task_id: str, *, device: str = "cuda") -> dict[str, Any]:
        if task_id not in self.tasks:
            raise KeyError("unknown bottleneck four-spine task")
        task = self.tasks[task_id]; kind = task["kind"]
        if kind == "authenticate":
            return self._authenticate()
        if kind == "preflight":
            return self._preflight(device=device)
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
            durable_files = tuple(
                path for path in self.root.rglob("*") if path.is_file()
            )
            forbidden = tuple(
                str(path.relative_to(self.root)) for path in durable_files
                if "resume" in path.name.lower() or "optimizer" in path.name.lower()
            )
            if forbidden:
                raise PermissionError(
                    "bottleneck campaign contains forbidden durable state: "
                    + ", ".join(forbidden)
                )
            durable_bytes = sum(path.stat().st_size for path in durable_files)
            if durable_bytes > int(self.spec["projected_durable_bytes"]):
                raise OSError("bottleneck durable bytes exceed the authorized projection")
            if shutil.disk_usage(self.root).free < int(self.spec["minimum_free_disk_bytes"]):
                raise OSError("bottleneck completion cannot preserve free-space reserve")
            complete = artifact({
                "parents": {
                    "campaign_spec": self.spec["content_hash"],
                    "aggregate": validate_artifact(aggregate, contract=AGGREGATE_CONTRACT),
                },
                "fresh_fit_count": len(FIT_ORDER), "reducer_count": len(REDUCER_ORDER),
                "source_fit_reuse_count": 1, "branch_order": list(BRANCH_ORDER),
                "endpoint_nodes": list(ENDPOINT_NODES),
                "rolling_resume_durable_bytes": 0,
                "optimizer_state_durable_bytes": 0,
                "durable_file_count_before_completion": len(durable_files),
                "durable_bytes_before_completion": durable_bytes,
                "projected_durable_bytes": int(self.spec["projected_durable_bytes"]),
                "minimum_free_disk_bytes_preserved": int(
                    self.spec["minimum_free_disk_bytes"]
                ),
                "durable_dense_pair_matrices": False,
                "source_campaign_outputs_mutated": False,
                "all_registered_science_executed": True,
                "scientific_result_does_not_control_completion": True,
                "established_rows_do_not_control_completion": True,
                "final_test_accessed": False,
            }, contract=COMPLETE_CONTRACT)
            write_immutable_json(self.root / "reports/campaign_complete.json", complete)
            return complete
        raise RuntimeError(f"unhandled bottleneck task: {kind}")


__all__ = [
    "BottleneckSpine4Workflow", "build_aggregate", "task_outputs",
]

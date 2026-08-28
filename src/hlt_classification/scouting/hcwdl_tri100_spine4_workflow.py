"""Fail-closed task dispatch and reporting for TRI100 four-spine."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, validate_content_hash, write_immutable_json,
)

from .hcwdl_tri100_spine4_campaign import validate_campaign
from .hcwdl_tri100_spine4_contracts import (
    AGGREGATE_CONTRACT, COMPLETE_CONTRACT, STAGE_REPORT_CONTRACT,
    TRAINING_REPORT_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_mhpe_tri60_training import Tri60DistributedContext
from .hcwdl_tri100_spine4_distributed import (
    run_distributed_acceptance, validate_distributed_acceptance,
)
from .hcwdl_tri100_spine4_graph import (
    BRANCH_NODES, BRANCH_ORDER, DDP_EXECUTION, ENDPOINT_NODES, FIT_ORDER, GRAPH_SHA256,
    NODE_REGISTRY, PROBABILITY_COMPONENTS, REDUCER_ORDER,
)
from .hcwdl_tri100_spine4_runner import run_fit, run_reducer
from .hcwdl_tri100_spine4_source import validate_source_lock


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
    acceptance = load_json(spec["artifact_paths"]["distributed_acceptance"])
    acceptance_hash = validate_distributed_acceptance(
        acceptance, campaign_spec_sha256=spec["content_hash"],
        recipe_sha256=spec["parents"]["recipe"],
    )
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
        or report.get("parents", {}).get("distributed_acceptance")
        != acceptance_hash
        or report.get("distributed_execution") != dict(DDP_EXECUTION)
        or report.get("final_test_accessed") is not False
        or not selected.is_file() or not final.is_file()
        or sha256_file(selected) != report.get("selected_checkpoint_sha256")
        or sha256_file(final) != report.get("final_checkpoint_sha256")
    ):
        raise ValueError(f"TRI100 four-spine training report differs: {node_id}")
    return report


def task_outputs(spec: Mapping[str, Any], task_id: str) -> list[Path]:
    tasks = {row["task_id"]: row for row in spec["tasks"]}
    if task_id not in tasks:
        raise KeyError("unknown TRI100 four-spine task")
    task = tasks[task_id]
    root = Path(spec["campaign_root"])
    if task["kind"] == "authenticate":
        return [Path(spec["artifact_paths"]["source_lock"])]
    if task["kind"] == "preflight":
        return [Path(spec["artifact_paths"]["distributed_acceptance"])]
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
                directory / f"{role}.npz",
                directory / f"{role}_shard.json",
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
    distributed_acceptance = load_json(
        spec["artifact_paths"]["distributed_acceptance"]
    )
    distributed_acceptance_hash = validate_distributed_acceptance(
        distributed_acceptance, campaign_spec_sha256=spec["content_hash"],
        recipe_sha256=spec["parents"]["recipe"],
    )
    source_report = load_json(source["u000"]["report_path"])
    parents = {
        "campaign_spec": spec["content_hash"],
        "graph": GRAPH_SHA256,
        "source_campaign": spec["parents"]["source_campaign"],
        "source_u000_report": source["u000"]["report_sha256"],
        "distributed_acceptance": distributed_acceptance_hash,
    }
    rows = [{
        "artifact_id": "U000",
        "kind": "source_anchor",
        "branch": None,
        "path_index": None,
        "coordinate": "U000",
        "metrics": source_report["validation"],
        "report_sha256": source_report["content_hash"],
        "selected_checkpoint_sha256": source_report[
            "selected_checkpoint_sha256"
        ],
        "completed_passes": int(source_report["passes"]),
    }]
    for node_id in FIT_ORDER:
        node = NODE_REGISTRY[node_id]
        report = _training_report(spec, node_id)
        parents[f"fit/{node_id}"] = report["content_hash"]
        rows.append({
            "artifact_id": node_id,
            "kind": "fit",
            "branch": node.branch,
            "path_index": node.path_index,
            "coordinate": node.coordinate_name,
            "parent_node_id": node.parent_node_id,
            "metrics": report["validation"],
            "report_sha256": report["content_hash"],
            "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
            "selected_pass": int(report["selected_pass"]),
            "completed_passes": int(report["passes"]),
            "stopped_early": bool(report["stopped_early"]),
        })
    for distribution_id in REDUCER_ORDER:
        stage = load_json(
            Path(spec["campaign_root"]) / "reports/stages"
            / f"{distribution_id}.json"
        )
        validate_artifact(stage, contract=STAGE_REPORT_CONTRACT)
        if (
            stage.get("distribution_id") != distribution_id
            or stage.get("component_order")
            != list(PROBABILITY_COMPONENTS[distribution_id])
            or stage.get("single_component_selected_checkpoint") is not True
        ):
            raise ValueError("TRI100 four-spine aggregate stage differs")
        parents[f"probability/{distribution_id}"] = stage["content_hash"]
    endpoints = []
    for branch, node_id in zip(BRANCH_ORDER, ENDPOINT_NODES, strict=True):
        report = _training_report(spec, node_id)
        endpoints.append({
            "branch": branch,
            "node_id": node_id,
            "path": ["U000", *(
                NODE_REGISTRY[name].coordinate_name
                for name in BRANCH_NODES[branch]
            )],
            "metrics": report["validation"],
            "selected_pass": int(report["selected_pass"]),
            "completed_passes": int(report["passes"]),
            "report_sha256": report["content_hash"],
            "selected_checkpoint_sha256": report[
                "selected_checkpoint_sha256"
            ],
        })
    return artifact({
        "parents": dict(sorted(parents.items())),
        "rows": rows,
        "endpoints": endpoints,
        "fresh_fit_count": len(FIT_ORDER),
        "reducer_count": len(REDUCER_ORDER),
        "source_fit_reuse_count": 1,
        "endpoint_seed_matched": True,
        "immediate_parent_only": True,
        "distributed_execution": dict(DDP_EXECUTION),
        "ensembles": False,
        "poor_metrics_do_not_control_completion": True,
        "source_campaign_outputs_mutated": False,
        "ordinary_access_roles": ["validation"],
        "final_test_accessed": False,
    }, contract=AGGREGATE_CONTRACT)


class Spine4Workflow:
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

    def _authenticate(self) -> dict[str, Any]:
        source = load_json(self.spec["artifact_paths"]["source_lock"])
        if validate_source_lock(source) != self.spec["parents"]["source_lock"]:
            raise ValueError("TRI100 four-spine source lock changed")
        return source

    def _preflight(
        self, distributed_context: Tri60DistributedContext | None,
    ) -> dict[str, Any]:
        source = self._authenticate()
        if shutil.disk_usage(self.root).free < int(
            self.spec["minimum_free_disk_bytes"]
        ):
            raise OSError("TRI100 four-spine free disk is below reserve")
        if tuple(self.root.rglob("*resume*")):
            raise PermissionError("TRI100 four-spine contains resume state")
        if self.spec.get("ordinary_final_test_capability") is not False:
            raise PermissionError("TRI100 four-spine has final-test capability")
        if distributed_context is None:
            raise RuntimeError("TRI100 four-spine preflight requires four-rank DDP")
        acceptance = run_distributed_acceptance(
            campaign_spec_sha256=self.spec["content_hash"],
            recipe_sha256=self.spec["parents"]["recipe"],
            source_commit=(self.execution_source_commit or self.spec["source_commit"]),
            context=distributed_context,
        )
        if distributed_context.is_primary:
            write_immutable_json(
                self.spec["artifact_paths"]["distributed_acceptance"],
                acceptance,
            )
        return acceptance

    def run(
        self, task_id: str, *, device: str = "cuda",
        distributed_context: Tri60DistributedContext | None = None,
    ) -> dict[str, Any]:
        if task_id not in self.tasks:
            raise KeyError("unknown TRI100 four-spine task")
        task = self.tasks[task_id]
        kind = task["kind"]
        if kind == "authenticate":
            return self._authenticate()
        if kind == "preflight":
            return self._preflight(distributed_context)
        if kind == "train":
            return run_fit(
                spec=self.spec, node_id=task["node_id"], device=device,
                recovery_spec_sha256=self.recovery_spec_sha256,
                execution_source_commit=self.execution_source_commit,
                distributed_context=distributed_context,
            )
        if kind == "reducer":
            return run_reducer(
                spec=self.spec, distribution_id=task["distribution_id"],
                device=device,
                recovery_spec_sha256=self.recovery_spec_sha256,
                execution_source_commit=self.execution_source_commit,
            )
        if kind == "aggregate":
            value = build_aggregate(self.spec)
            write_immutable_json(
                self.root / "reports/validation_aggregate.json", value,
            )
            return value
        if kind == "campaign_complete":
            aggregate = load_json(
                self.root / "reports/validation_aggregate.json"
            )
            complete = artifact({
                "parents": {
                    "campaign_spec": self.spec["content_hash"],
                    "aggregate": validate_artifact(
                        aggregate, contract=AGGREGATE_CONTRACT,
                    ),
                },
                "fresh_fit_count": len(FIT_ORDER),
                "reducer_count": len(REDUCER_ORDER),
                "source_fit_reuse_count": 1,
                "branch_order": list(BRANCH_ORDER),
                "endpoint_nodes": list(ENDPOINT_NODES),
                "rolling_resume_durable_bytes": 0,
                "source_campaign_outputs_mutated": False,
                "all_registered_science_executed": True,
                "scientific_result_does_not_control_completion": True,
                "final_test_accessed": False,
            }, contract=COMPLETE_CONTRACT)
            write_immutable_json(
                self.root / "reports/campaign_complete.json", complete,
            )
            return complete
        raise RuntimeError(f"unhandled TRI100 four-spine task: {kind}")


__all__ = ["Spine4Workflow", "build_aggregate", "task_outputs"]

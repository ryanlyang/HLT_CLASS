"""Fail-closed task dispatch and reporting for the dense extension."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, validate_content_hash, write_immutable_json,
)

from .hcwdl_mhpe_tri60_dense_campaign import campaign_tasks, validate_campaign
from .hcwdl_mhpe_tri60_dense_contracts import (
    AGGREGATE_CONTRACT, COMPLETE_CONTRACT, FINALIST_LOCK_CONTRACT,
    STAGE_REPORT_CONTRACT, TRAINING_REPORT_CONTRACT, artifact,
    validate_artifact,
)
from .hcwdl_mhpe_tri60_dense_graph import (
    ENSEMBLE_COMPONENTS, FIT_ORDER, GRAPH_SHA256, NODE_REGISTRY, REDUCER_ORDER,
)
from .hcwdl_mhpe_tri60_dense_runner import run_fit, run_reducer
from .hcwdl_mhpe_tri60_dense_source import (
    publish_source_gate, validate_source_gate, validate_source_lock,
)


def _training_report(spec: Mapping[str, Any], node_id: str) -> dict[str, Any]:
    path = Path(spec["campaign_root"]) / "training" / node_id / "training_report.json"
    report = load_json(path)
    validate_content_hash(
        report, expected_contract=TRAINING_REPORT_CONTRACT,
        expected_schema_version=1,
    )
    selected = path.parent / str(report.get("selected_checkpoint", ""))
    final = path.parent / str(report.get("final_checkpoint", ""))
    if (
        report.get("node_id") != node_id
        or report.get("node_spec") != NODE_REGISTRY[node_id].payload()
        or report.get("campaign_spec_sha256") != spec["content_hash"]
        or report.get("graph_sha256") != GRAPH_SHA256
        or report.get("recipe_sha256") != spec["parents"]["source_recipe"]
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
        raise ValueError(f"dense training report differs: {node_id}")
    return report


def task_outputs(spec: Mapping[str, Any], task_id: str) -> list[Path]:
    tasks = {row["task_id"]: row for row in spec["tasks"]}
    if task_id not in tasks:
        raise KeyError("unknown dense task")
    task = tasks[task_id]
    root = Path(spec["campaign_root"])
    if task["kind"] == "authenticate":
        return [Path(spec["artifact_paths"]["source_lock"])]
    if task["kind"] == "preflight":
        return [Path(spec["artifact_paths"]["graph"])]
    if task["kind"] == "source_gate":
        return [Path(spec["artifact_paths"]["source_gate"])]
    if task["kind"] == "train":
        report = _training_report(spec, task["node_id"])
        directory = root / "training" / task["node_id"]
        outputs = [
            directory / "training_report.json",
            directory / report["selected_checkpoint"],
            directory / report["final_checkpoint"],
        ]
        audit = directory / "ephemeral_representation_audit.json"
        if audit.is_file():
            outputs.append(audit)
        calibration = directory / "calibration"
        if calibration.is_dir():
            outputs.extend(sorted(calibration.glob("*.json")))
        return outputs
    if task["kind"] == "reducer":
        distribution = task["distribution_id"]
        directory = root / "probabilities" / distribution
        outputs = [root / "reports/stages" / f"{distribution}.json", directory / "lock.json"]
        for role in ("train", "validation"):
            outputs.extend([
                directory / f"{role}.npz", directory / f"{role}_shard.json",
                directory / f"{role}_manifest.json",
            ])
        return outputs
    if task["kind"] == "aggregate":
        return [root / "reports/validation_aggregate.json"]
    if task["kind"] == "finalist_lock":
        return [root / "locks/finalist.json"]
    return [root / "reports/campaign_complete.json"]


def build_aggregate(spec: Mapping[str, Any]) -> dict[str, Any]:
    validate_campaign(spec)
    rows = []
    parents = {
        "campaign_spec": spec["content_hash"], "graph": GRAPH_SHA256,
        "source_campaign": spec["parents"]["source_campaign"],
    }
    for node_id in FIT_ORDER:
        report = _training_report(spec, node_id)
        parents[f"fit/{node_id}"] = report["content_hash"]
        rows.append({
            "artifact_id": node_id, "kind": "fit",
            "track": NODE_REGISTRY[node_id].track,
            "coordinate": NODE_REGISTRY[node_id].coordinate_name,
            "metrics": report["validation"],
            "report_sha256": report["content_hash"],
            "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
        })
    for distribution_id in REDUCER_ORDER:
        stage = load_json(
            Path(spec["campaign_root"]) / "reports/stages" / f"{distribution_id}.json"
        )
        validate_artifact(stage, contract=STAGE_REPORT_CONTRACT)
        if (
            stage.get("distribution_id") != distribution_id
            or stage.get("component_order") != list(ENSEMBLE_COMPONENTS[distribution_id])
        ):
            raise ValueError("dense aggregate stage differs")
        parents[f"ensemble/{distribution_id}"] = stage["content_hash"]
        rows.append({
            "artifact_id": distribution_id, "kind": "ensemble",
            "track": distribution_id.split("_")[1], "coordinate": distribution_id,
            "metrics": stage["ensemble_metrics"],
            "report_sha256": stage["content_hash"],
            "component_count": len(ENSEMBLE_COMPONENTS[distribution_id]),
        })
    return artifact({
        "parents": dict(sorted(parents.items())), "rows": rows,
        "fresh_fit_count": len(FIT_ORDER), "reducer_count": len(REDUCER_ORDER),
        "source_fit_reuse_count": int(spec["source_fit_count"]),
        "poor_metrics_do_not_control_completion": True,
        "source_campaign_outputs_mutated": False,
        "ordinary_access_roles": ["validation"],
        "final_test_accessed": False,
    }, contract=AGGREGATE_CONTRACT)


class DenseWorkflow:
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
            raise ValueError("dense source lock changed")
        return source

    def _preflight(self) -> dict[str, Any]:
        source = self._authenticate()
        free = shutil.disk_usage(self.root).free
        if free < int(self.spec["minimum_free_disk_bytes"]):
            raise OSError("dense extension free disk is below reserve")
        if tuple(self.root.rglob("*resume*")):
            raise PermissionError("dense extension contains forbidden resume paths")
        if self.spec.get("ordinary_final_test_capability") is not False:
            raise PermissionError("dense extension has final-test capability")
        return source

    def run(self, task_id: str, *, device: str = "cuda") -> dict[str, Any]:
        if task_id not in self.tasks:
            raise KeyError("unknown dense task")
        task = self.tasks[task_id]
        kind = task["kind"]
        if kind == "authenticate":
            return self._authenticate()
        if kind == "preflight":
            return self._preflight()
        if kind == "source_gate":
            source = self._preflight()
            return publish_source_gate(
                source, output=self.spec["artifact_paths"]["source_gate"],
            )
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
        if kind == "finalist_lock":
            aggregate = load_json(self.root / "reports/validation_aggregate.json")
            aggregate_hash = validate_artifact(aggregate, contract=AGGREGATE_CONTRACT)
            m2 = _training_report(self.spec, "DX_M2")
            lock = artifact({
                "parents": {
                    "campaign_spec": self.spec["content_hash"],
                    "aggregate": aggregate_hash,
                    "m2_report": m2["content_hash"],
                    "m2_checkpoint": m2["selected_checkpoint_sha256"],
                },
                "registered_validation_finalists": [
                    "DX_LOGIT_D000E", "DX_RSET_D000E", "DX_RREL_D000E",
                    "DX_M1_LOGIT", "DX_M1_RSET", "DX_M1_RREL", "DX_M1E", "DX_M2",
                ],
                "designated_single_model": "DX_M2",
                "deployable_input_domain": "exact_hlt",
                "checkpoint_reused_without_duplicate_copy": True,
                "unlocks_final_test": False,
                "scientific_result_does_not_control_completion": True,
                "final_test_accessed": False,
            }, contract=FINALIST_LOCK_CONTRACT)
            write_immutable_json(self.root / "locks/finalist.json", lock)
            return lock
        if kind == "campaign_complete":
            aggregate = load_json(self.root / "reports/validation_aggregate.json")
            finalist = load_json(self.root / "locks/finalist.json")
            complete = artifact({
                "parents": {
                    "campaign_spec": self.spec["content_hash"],
                    "aggregate": validate_artifact(aggregate, contract=AGGREGATE_CONTRACT),
                    "finalist_lock": validate_artifact(finalist, contract=FINALIST_LOCK_CONTRACT),
                },
                "fresh_fit_count": len(FIT_ORDER),
                "ensemble_reducer_count": len(REDUCER_ORDER),
                "source_fit_reuse_count": int(self.spec["source_fit_count"]),
                "representation_target_durable_bytes": 0,
                "rolling_resume_durable_bytes": 0,
                "source_campaign_outputs_mutated": False,
                "all_registered_science_executed": True,
                "scientific_result_does_not_control_completion": True,
                "final_test_accessed": False,
            }, contract=COMPLETE_CONTRACT)
            write_immutable_json(self.root / "reports/campaign_complete.json", complete)
            return complete
        raise RuntimeError(f"unhandled dense task kind: {kind}")


__all__ = ["DenseWorkflow", "build_aggregate", "task_outputs"]

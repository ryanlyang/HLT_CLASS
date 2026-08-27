"""Task dispatch and durable output validation for TRI60 D000 SD5."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import load_json, write_immutable_json

from .hcwdl_mhpe_tri60_campaign import validate_campaign as validate_source
from .hcwdl_mhpe_tri60_contracts import (
    ENDPOINT_RESOURCE_LOCK_CONTRACT,
    validate_artifact as validate_source_artifact,
)
from .hcwdl_tri60_ce5_campaign import validate_campaign as validate_ce5
from .hcwdl_tri60_d000_sd5_campaign import campaign_tasks, validate_campaign
from .hcwdl_tri60_d000_sd5_contracts import (
    AGGREGATE_CONTRACT, CAMPAIGN_COMPLETE_CONTRACT, validate_artifact,
)
from .hcwdl_tri60_d000_sd5_graph import ENSEMBLE_ID
from .hcwdl_tri60_d000_sd5_reporting import (
    build_aggregate, build_campaign_complete, ensemble_report, training_report,
)
from .hcwdl_tri60_d000_sd5_runner import run_fit, run_reducer


def _task(spec: Mapping[str, Any], task_id: str) -> Mapping[str, Any]:
    try:
        return next(row for row in campaign_tasks() if row["task_id"] == task_id)
    except StopIteration as error:
        raise KeyError("unknown TRI60 D000 SD5 task") from error


def task_outputs(spec: Mapping[str, Any], task_id: str) -> list[Path]:
    task = _task(spec, task_id)
    root = Path(spec["campaign_root"])
    if task["kind"] == "authenticate":
        return [
            Path(spec["artifact_paths"]["source_campaign_spec"]),
            Path(spec["artifact_paths"]["ce5_campaign_spec"]),
        ]
    if task["kind"] == "preflight":
        return [Path(spec["artifact_paths"]["endpoint_resource_lock"])]
    if task["kind"] == "train":
        report = training_report(spec, task["node_id"])
        directory = root / "training" / task["node_id"]
        return [
            directory / "training_report.json",
            directory / report["selected_checkpoint"],
            directory / report["final_checkpoint"],
        ]
    if task["kind"] == "reducer":
        ensemble_report(spec)
        return [root / "reports" / f"{ENSEMBLE_ID}.json"]
    if task["kind"] == "aggregate":
        report = load_json(root / "reports/validation_aggregate.json")
        validate_artifact(report, contract=AGGREGATE_CONTRACT)
        return [root / "reports/validation_aggregate.json"]
    report = load_json(root / "reports/campaign_complete.json")
    validate_artifact(report, contract=CAMPAIGN_COMPLETE_CONTRACT)
    return [root / "reports/campaign_complete.json"]


class D000SD5Workflow:
    def __init__(
        self, spec: Mapping[str, Any], *, execution_source_commit: str | None = None,
    ) -> None:
        self.spec = dict(spec)
        self.root = Path(spec["campaign_root"])
        self.execution_source_commit = execution_source_commit

    def run(self, task_id: str, *, device: str = "cuda") -> dict[str, Any]:
        validate_campaign(self.spec, executable=False)
        task = _task(self.spec, task_id)
        kind = task["kind"]
        if kind == "authenticate":
            source = load_json(self.spec["artifact_paths"]["source_campaign_spec"])
            if validate_source(
                source, executable=False, verify_source_tree=False,
            ) != self.spec["parents"]["source_campaign"]:
                raise ValueError("TRI60 D000 SD5 source campaign changed")
            ce5 = load_json(self.spec["artifact_paths"]["ce5_campaign_spec"])
            if validate_ce5(ce5, executable=False) != self.spec["parents"]["ce5_campaign"]:
                raise ValueError("TRI60 D000 SD5 CE5 campaign changed")
            return self.spec
        if kind == "preflight":
            if shutil.disk_usage(self.root).free < int(
                self.spec["minimum_free_disk_bytes"]
            ):
                raise OSError("TRI60 D000 SD5 free disk is below the exact reserve")
            endpoint = load_json(self.spec["artifact_paths"]["endpoint_resource_lock"])
            if validate_source_artifact(
                endpoint, contract=ENDPOINT_RESOURCE_LOCK_CONTRACT,
            ) != self.spec["parents"]["endpoint_resources"]:
                raise ValueError("TRI60 D000 SD5 endpoint evidence changed")
            return endpoint
        if kind == "train":
            return run_fit(
                spec=self.spec, node_id=task["node_id"], device=device,
                execution_source_commit=self.execution_source_commit,
            )
        if kind == "reducer":
            if task_id != f"reduce_{ENSEMBLE_ID}":
                raise ValueError("TRI60 D000 SD5 reducer identity differs")
            return run_reducer(
                spec=self.spec, device=device,
                execution_source_commit=self.execution_source_commit,
            )
        if kind == "aggregate":
            result = build_aggregate(self.spec)
            write_immutable_json(
                self.root / "reports/validation_aggregate.json", result,
            )
            return result
        if kind == "campaign_complete":
            aggregate = load_json(self.root / "reports/validation_aggregate.json")
            result = build_campaign_complete(self.spec, aggregate)
            write_immutable_json(
                self.root / "reports/campaign_complete.json", result,
            )
            return result
        raise KeyError("unknown TRI60 D000 SD5 task kind")


__all__ = ["D000SD5Workflow", "task_outputs"]

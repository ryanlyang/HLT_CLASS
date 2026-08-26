"""Task dispatch and durable outputs for the TRI60 M1 screen."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import load_json, write_immutable_json

from .hcwdl_mhpe_tri60_contracts import (
    ENDPOINT_RESOURCE_LOCK_CONTRACT,
    validate_artifact as validate_source_artifact,
)
from .hcwdl_tri60_m1_screen_campaign import campaign_tasks, validate_campaign
from .hcwdl_tri60_m1_screen_contracts import (
    AGGREGATE_CONTRACT, CAMPAIGN_COMPLETE_CONTRACT,
    validate_artifact,
)
from .hcwdl_tri60_m1_screen_reporting import (
    build_aggregate, build_campaign_complete, training_report,
)
from .hcwdl_tri60_m1_screen_runner import run_fit
from .hcwdl_tri60_m1_screen_source import validate_source_lock


def _task(task_id: str) -> Mapping[str, Any]:
    try:
        return next(row for row in campaign_tasks() if row["task_id"] == task_id)
    except StopIteration as error:
        raise KeyError("unknown TRI60 M1 screen task") from error


def task_outputs(spec: Mapping[str, Any], task_id: str) -> list[Path]:
    task = _task(task_id)
    root = Path(spec["campaign_root"])
    if task["kind"] == "authenticate":
        return [Path(spec["artifact_paths"]["source_lock"])]
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
    if task["kind"] == "aggregate":
        report = load_json(root / "reports/validation_aggregate.json")
        validate_artifact(report, contract=AGGREGATE_CONTRACT)
        return [root / "reports/validation_aggregate.json"]
    report = load_json(root / "reports/campaign_complete.json")
    validate_artifact(report, contract=CAMPAIGN_COMPLETE_CONTRACT)
    return [root / "reports/campaign_complete.json"]


class M1ScreenWorkflow:
    def __init__(
        self, spec: Mapping[str, Any], *, recovery_spec_sha256: str | None = None,
        execution_source_commit: str | None = None,
    ) -> None:
        self.spec = dict(spec)
        self.root = Path(spec["campaign_root"])
        self.recovery_spec_sha256 = recovery_spec_sha256
        self.execution_source_commit = execution_source_commit

    def run(self, task_id: str, *, device: str = "cuda") -> dict[str, Any]:
        validate_campaign(self.spec, executable=False)
        task = _task(task_id)
        if task["kind"] == "authenticate":
            source = load_json(self.spec["artifact_paths"]["source_lock"])
            if validate_source_lock(source) != self.spec["parents"]["source_lock"]:
                raise ValueError("TRI60 M1 screen source changed")
            return source
        if task["kind"] == "preflight":
            if shutil.disk_usage(self.root).free < int(self.spec["minimum_free_disk_bytes"]):
                raise OSError("TRI60 M1 screen free disk is below the exact reserve")
            endpoint = load_json(self.spec["artifact_paths"]["endpoint_resource_lock"])
            validate_source_artifact(endpoint, contract=ENDPOINT_RESOURCE_LOCK_CONTRACT)
            return endpoint
        if task["kind"] == "train":
            return run_fit(
                spec=self.spec, node_id=task["node_id"], device=device,
                recovery_spec_sha256=self.recovery_spec_sha256,
                execution_source_commit=self.execution_source_commit,
            )
        if task["kind"] == "aggregate":
            value = build_aggregate(self.spec)
            write_immutable_json(self.root / "reports/validation_aggregate.json", value)
            return value
        if task["kind"] == "campaign_complete":
            aggregate = load_json(self.root / "reports/validation_aggregate.json")
            value = build_campaign_complete(self.spec, aggregate)
            write_immutable_json(self.root / "reports/campaign_complete.json", value)
            return value
        raise KeyError("unknown TRI60 M1 screen task kind")


__all__ = ["M1ScreenWorkflow", "task_outputs"]

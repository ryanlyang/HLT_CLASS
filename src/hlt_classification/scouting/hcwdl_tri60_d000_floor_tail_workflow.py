"""Fail-closed task dispatch for the D000 floor-tail confirmation."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, write_immutable_json,
)

from .hcwdl_mhpe_tri60_contracts import (
    ENDPOINT_RESOURCE_LOCK_CONTRACT,
    validate_artifact as validate_source_artifact,
)
from .hcwdl_tri60_d000_floor_tail_campaign import (
    campaign_tasks, validate_campaign,
)
from .hcwdl_tri60_d000_floor_tail_contracts import (
    CAMPAIGN_COMPLETE_CONTRACT, TRAINING_REPORT_CONTRACT,
    artifact, validate_artifact,
)
from .hcwdl_tri60_d000_floor_tail_graph import (
    CONDITION_ID, EARLY_STOPPING, GRAPH_SHA256, LOSS_SCHEDULE, LR_SCHEDULE,
    NODE,
)
from .hcwdl_tri60_d000_floor_tail_reference import validate_reference_lock
from .hcwdl_tri60_d000_floor_tail_runner import run_fit
from .hcwdl_mhpe_tri60_training import (
    Tri60TrainingRuntime, tri60_early_stopping,
)


def _task(task_id: str) -> Mapping[str, Any]:
    try:
        return next(row for row in campaign_tasks() if row["task_id"] == task_id)
    except StopIteration as error:
        raise KeyError("unknown D000 floor-tail task") from error


def training_report(spec: Mapping[str, Any]) -> dict[str, Any]:
    directory = Path(spec["campaign_root"]) / "training" / CONDITION_ID
    report = load_json(directory / "training_report.json")
    validate_artifact(report, contract=TRAINING_REPORT_CONTRACT)
    selected = directory / str(report.get("selected_checkpoint", ""))
    final = directory / str(report.get("final_checkpoint", ""))
    passes = int(report.get("passes", -1))
    stopped = passes < 100
    parents = report.get("parents", {})
    throughput = report.get("throughput_optimizations", {})
    expected_early_stopping = tri60_early_stopping(
        Tri60TrainingRuntime(passes=100, batch_size=256),
        dict(EARLY_STOPPING),
    )
    if (
        report.get("node_id") != CONDITION_ID
        or report.get("node_spec") != NODE.payload()
        or report.get("campaign_spec_sha256") != spec["content_hash"]
        or report.get("graph_sha256") != GRAPH_SHA256
        or report.get("recipe_sha256") != spec["parents"]["recipe"]
        or report.get("execution_source_commit") != spec["source_commit"]
        or parents.get("reference_lock") != spec["parents"]["reference_lock"]
        or parents.get("reference_screen") != spec["parents"]["reference_screen"]
        or parents.get("source_campaign") != spec["parents"]["source_campaign"]
        or report.get("loss_schedule") != dict(LOSS_SCHEDULE)
        or report.get("learning_rate_schedule") != dict(LR_SCHEDULE)
        or report.get("early_stopping") != expected_early_stopping
        or float(report.get("peak_learning_rate", 0)) != 3.0e-4
        or report.get("maximum_passes") != 100
        or report.get("minimum_passes") != 60
        or not 60 <= passes <= 100
        or report.get("validations") != passes
        or report.get("stopped_early") is not stopped
        or report.get("performance_early_termination") is not stopped
        or report.get("stop_reason") != (
            "macro_auc_patience_exhausted"
            if stopped else "maximum_passes_reached"
        )
        or report.get("resume_policy") != "disabled_restart_from_zero_v1"
        or report.get("rng_domains", {}).get("replicate_seed")
        != spec["replicate_seed"]
        or report.get("rng_domains", {}).get("node_seed_alias")
        != NODE.seed_alias
        or throughput.get("synchronous_data_parallel_world_size") != 1
        or throughput.get("global_batch_size_unchanged") is not True
        or throughput.get("optimizer_update_count_unchanged") is not True
        or "distributed_execution" in report
        or report.get("complete") is not True
        or report.get("rolling_resume_published") is not False
        or report.get("partial_checkpoint_reuse") is not False
        or report.get("final_test_accessed") is not False
        or not selected.is_file() or not final.is_file()
        or sha256_file(selected) != report.get("selected_checkpoint_sha256")
        or sha256_file(final) != report.get("final_checkpoint_sha256")
    ):
        raise ValueError("D000 floor-tail training report differs")
    return report


def task_outputs(spec: Mapping[str, Any], task_id: str) -> list[Path]:
    task = _task(task_id)
    root = Path(spec["campaign_root"])
    if task["kind"] == "authenticate":
        return [Path(spec["artifact_paths"]["reference_lock"])]
    if task["kind"] == "preflight":
        return [Path(spec["artifact_paths"]["endpoint_resource_lock"])]
    if task["kind"] == "train":
        report = training_report(spec)
        directory = root / "training" / CONDITION_ID
        return [
            directory / "training_report.json",
            directory / report["selected_checkpoint"],
            directory / report["final_checkpoint"],
        ]
    return [root / "reports/campaign_complete.json"]


class FloorTailWorkflow:
    def __init__(self, spec: Mapping[str, Any]) -> None:
        validate_campaign(spec, executable=False)
        self.spec = dict(spec)
        self.root = Path(spec["campaign_root"])

    def run(self, task_id: str, *, device: str = "cuda") -> dict[str, Any]:
        task = _task(task_id)
        if task["kind"] == "authenticate":
            value = load_json(self.spec["artifact_paths"]["reference_lock"])
            if validate_reference_lock(value) != self.spec["parents"]["reference_lock"]:
                raise ValueError("D000 floor-tail reference changed")
            return value
        if task["kind"] == "preflight":
            if shutil.disk_usage(self.root).free < int(
                self.spec["minimum_free_disk_bytes"]
            ):
                raise OSError("D000 floor-tail free disk is below reserve")
            endpoint = load_json(
                self.spec["artifact_paths"]["endpoint_resource_lock"]
            )
            validate_source_artifact(
                endpoint, contract=ENDPOINT_RESOURCE_LOCK_CONTRACT,
            )
            return endpoint
        if task["kind"] == "train":
            return run_fit(spec=self.spec, device=device)
        if task["kind"] == "campaign_complete":
            report = training_report(self.spec)
            value = artifact({
                "parents": {
                    "campaign_spec": self.spec["content_hash"],
                    "training_report": report["content_hash"],
                    "reference_lock": self.spec["parents"]["reference_lock"],
                },
                "condition_id": CONDITION_ID,
                "reference_condition_id": self.spec["reference_condition_id"],
                "selected_pass": int(report["selected_pass"]),
                "completed_passes": int(report["passes"]),
                "validation": dict(report["validation"]),
                "reference_report_required_for_completion": False,
                "scientific_result_does_not_control_completion": True,
                "final_test_accessed": False,
            }, contract=CAMPAIGN_COMPLETE_CONTRACT)
            write_immutable_json(
                self.root / "reports/campaign_complete.json", value,
            )
            return value
        raise RuntimeError("unhandled D000 floor-tail task")


__all__ = ["FloorTailWorkflow", "task_outputs", "training_report"]

"""Idempotent workers for the HCWDL-UB-FULLCOARSE3 arm DAGs."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import (
    load_json,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)

from .engine import validate_pmard_training_report
from .hcwdl_unified_balanced_coarse_campaign import validate_arm_campaign
from .hcwdl_unified_balanced_coarse_contracts import (
    aggregate_payload,
    completion_payload,
    validate_aggregate,
    validate_completion,
)
from .hcwdl_unified_balanced_coarse_graph import (
    arm_registry,
    idealized_u000_ancestry,
)
from .hcwdl_unified_balanced_coarse_runner import (
    arm_node_output_dir,
    run_arm_node,
)


RUNTIME_CONTRACT = "HCWDL_UNIFIED_BALANCED_FULL_COARSE_RUNTIME/v1"


def _task(tasks, task_id: str) -> Mapping[str, Any]:
    rows = [row for row in tasks if row["task_id"] == task_id]
    if len(rows) != 1:
        raise ValueError("HCWDL-UB-FULLCOARSE3 task registry differs")
    return rows[0]


def _index(task: Mapping[str, Any], array_index: int | None) -> None:
    count = int(task["array_count"])
    if count == 1 and array_index not in (None, 0):
        raise IndexError("HCWDL-UB-FULLCOARSE3 scalar task has array index")
    if count != 1:
        raise ValueError("HCWDL-UB-FULLCOARSE3 arms contain no array tasks")


def _runtime(
    output: Path, *, scope_spec_sha256: str, canonical_node_id: str,
    training_report_sha256: str, started: float,
) -> Path:
    elapsed = max(0.0, time.monotonic() - started)
    peak_gpu = 0
    try:
        import torch
        if torch.cuda.is_available():
            peak_gpu = int(torch.cuda.max_memory_allocated())
    except ImportError:
        pass
    payload = with_content_hash({
        "contract": RUNTIME_CONTRACT,
        "schema_version": 1,
        "scope_spec_sha256": scope_spec_sha256,
        "canonical_node_id": canonical_node_id,
        "training_report_sha256": training_report_sha256,
        "elapsed_seconds": elapsed,
        "measured_gpu_hours": elapsed / 3600.0,
        "peak_gpu_memory_bytes": peak_gpu,
        "phase_boundaries_recorded": True,
        "final_test_accessed": False,
    })
    path = output / "runtime.json"
    write_immutable_json(path, payload)
    return path


class UnifiedBalancedCoarseArmWorkflow:
    def __init__(
        self, spec: Mapping[str, Any], *, producer_commit: str | None = None,
    ) -> None:
        validate_arm_campaign(
            spec, executable=False, verify_source_tree=producer_commit is None,
        )
        self.spec = dict(spec)
        self.root = Path(spec["campaign_root"])
        self.arm_id = str(spec["arm_id"])
        self.producer_commit = str(producer_commit or spec["source_commit"])

    def run(self, task_id: str, *, array_index: int | None = None) -> list[Path]:
        task = _task(self.spec["tasks"], task_id)
        _index(task, array_index)
        kind = str(task["kind"])
        if kind == "arm_node":
            started = time.monotonic()
            node_id = str(task["node_id"])
            wrapper = run_arm_node(arm_spec=self.spec, node_id=node_id)
            output = arm_node_output_dir(self.root, node_id)
            runtime = _runtime(
                output, scope_spec_sha256=self.spec["content_hash"],
                canonical_node_id=f"{self.arm_id}/{node_id}",
                training_report_sha256=wrapper["pmard_engine_report_sha256"],
                started=started,
            )
            return [
                output / "training_report.json",
                output / "hcwdl_training_report.json",
                runtime,
            ]
        if kind == "aggregate":
            rows = []
            gpu_hours = 0.0
            registry = arm_registry(self.arm_id)
            ancestry = idealized_u000_ancestry(self.arm_id)
            reuse = load_json(self.spec["reuse_lock_path"])
            foundation_root = Path(reuse["foundation_lock_path"]).parent.parent
            shared = {}
            for node_id in ("U000", "M0paired"):
                report = load_json(
                    foundation_root / f"training/{node_id}/training_report.json"
                )
                shared[node_id] = {
                    "metrics": report["validation"],
                    "report_sha256": validate_pmard_training_report(report),
                    "checkpoint_sha256": report["selected_checkpoint_sha256"],
                }
            for node_id, node in registry.items():
                output = self.root / f"training/{node_id}"
                report = load_json(output / "training_report.json")
                report_hash = validate_pmard_training_report(report)
                wrapper = load_json(output / "hcwdl_training_report.json")
                validate_content_hash(
                    wrapper,
                    expected_contract=(
                        "HCWDL_UNIFIED_BALANCED_FULL_COARSE_TRAINING_REPORT/v1"
                    ),
                    expected_schema_version=1,
                )
                if (
                    wrapper.get("pmard_engine_report_sha256") != report_hash
                    or report.get("scientific_config", {}).get("canonical_node_id")
                    != node.canonical_id
                    or report.get("scientific_config", {}).get("coordinate_exact")
                    != node.coordinate.payload()
                ):
                    raise ValueError(
                        "HCWDL-UB-FULLCOARSE3 completed node lineage differs"
                    )
                runtime = load_json(output / "runtime.json")
                validate_content_hash(
                    runtime, expected_contract=RUNTIME_CONTRACT,
                    expected_schema_version=1,
                )
                gpu_hours += float(runtime["measured_gpu_hours"])
                rows.append({
                    "node_id": node_id,
                    "canonical_id": node.canonical_id,
                    "parent_id": node.parent_id,
                    "grandparent_id": node.grandparent_id,
                    "coordinate": node.coordinate.payload(),
                    "weights": {
                        "ce": node.ce_weight,
                        "parent_kd": node.parent_kd_weight,
                        "grandparent_kd": node.grandparent_kd_weight,
                    },
                    "idealized_u000_ancestry": ancestry[node_id],
                    "metrics": report["validation"],
                    "selected_update": report["selected_update"],
                    "report_sha256": report_hash,
                    "checkpoint_sha256": report["selected_checkpoint_sha256"],
                    "runtime_sha256": runtime["content_hash"],
                })
            payload = aggregate_payload(
                arm_id=self.arm_id,
                arm_spec_sha256=self.spec["content_hash"],
                rows=rows,
                shared=shared,
                gpu_hours=gpu_hours,
            )
            validate_aggregate(payload)
            output = self.root / "reports/validation_aggregate.json"
            write_immutable_json(output, payload)
            return [output]
        if kind == "campaign_complete":
            aggregate = load_json(self.root / "reports/validation_aggregate.json")
            aggregate_hash = validate_aggregate(aggregate)
            reports = {
                node_id: validate_pmard_training_report(load_json(
                    self.root / f"training/{node_id}/training_report.json"
                ))
                for node_id in arm_registry(self.arm_id)
            }
            payload = completion_payload(
                arm_id=self.arm_id,
                arm_spec_sha256=self.spec["content_hash"],
                aggregate_sha256=aggregate_hash,
                reports=reports,
                gpu_hours=float(aggregate["gpu_hours"]),
            )
            validate_completion(payload)
            output = self.root / "reports/campaign_complete.json"
            write_immutable_json(output, payload)
            return [output]
        raise RuntimeError(f"unhandled HCWDL-UB-FULLCOARSE3 task kind {kind}")


__all__ = ["RUNTIME_CONTRACT", "UnifiedBalancedCoarseArmWorkflow"]

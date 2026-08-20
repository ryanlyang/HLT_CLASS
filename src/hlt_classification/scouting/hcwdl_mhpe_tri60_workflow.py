"""Task dispatch and fail-closed gates for the TRI60 campaign."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import (
    load_json, write_immutable_json,
)

from .hcwdl_mhpe_tri60_campaign import campaign_tasks, validate_campaign
from .hcwdl_mhpe_tri60_contracts import (
    AGGREGATE_CONTRACT, CAMPAIGN_COMPLETE_CONTRACT,
    DEPLOYABLE_CHECKPOINT_CONTRACT, ENDPOINT_RESOURCE_LOCK_CONTRACT,
    FINALIST_LOCK_CONTRACT, INTEGRATION_LOCK_CONTRACT,
    artifact, validate_artifact,
)
from .hcwdl_mhpe_tri60_reporting import build_aggregate, publish_m2_deployable
from .hcwdl_mhpe_tri60_runner import run_fit, run_reducer


def task_outputs(spec: Mapping[str, Any], task_id: str) -> list[Path]:
    """Return the complete durable output inventory for one canonical task."""

    root = Path(spec["campaign_root"])
    try:
        task = next(row for row in campaign_tasks() if row["task_id"] == task_id)
    except StopIteration as error:
        raise KeyError("unknown TRI60 task") from error
    kind = task["kind"]
    if kind == "authenticate":
        return [Path(spec["artifact_paths"]["integration_lock"])]
    if kind == "preflight":
        return [Path(spec["artifact_paths"]["endpoint_resource_lock"])]
    if kind == "train":
        directory = root / "training" / task["node_id"]
        report_path = directory / "training_report.json"
        report = load_json(report_path)
        rows = [
            report_path,
            directory / report["selected_checkpoint"],
            directory / report["final_checkpoint"],
        ]
        audit = directory / "ephemeral_representation_audit.json"
        if audit.is_file():
            rows.append(audit)
        calibration = directory / "calibration"
        if calibration.is_dir():
            rows.extend(sorted(calibration.glob("*.json")))
        return rows
    if kind == "reducer":
        distribution = task["distribution_id"]
        directory = root / "probabilities" / distribution
        rows = [root / "reports/stages" / f"{distribution}.json", directory / "lock.json"]
        for role in ("train", "validation"):
            rows.extend([
                directory / f"{role}.npz",
                directory / f"{role}_shard.json",
                directory / f"{role}_manifest.json",
            ])
        return rows
    if kind == "aggregate":
        return [root / "reports/validation_aggregate.json"]
    if kind == "finalist_lock":
        return [
            root / "locks/finalist.json",
            root / "deployment/M2.pt",
            root / "deployment/M2.json",
        ]
    return [root / "reports/campaign_complete.json"]


class Tri60Workflow:
    def __init__(
        self, spec: Mapping[str, Any], *, recovery_spec_sha256: str | None = None,
        execution_source_commit: str | None = None,
    ) -> None:
        validate_campaign(spec, executable=False)
        self.spec = spec
        self.recovery_spec_sha256 = recovery_spec_sha256
        self.execution_source_commit = execution_source_commit
        self.root = Path(spec["campaign_root"])
        self.tasks = {row["task_id"]: row for row in campaign_tasks()}

    def _authenticate(self) -> dict[str, Any]:
        integration = load_json(self.spec["artifact_paths"]["integration_lock"])
        if validate_artifact(integration, contract=INTEGRATION_LOCK_CONTRACT) != (
            self.spec["parents"]["integration"]
        ):
            raise ValueError("TRI60 integration lock changed")
        if integration.get("runtime_sibling_worktree_imports") is not False:
            raise PermissionError("TRI60 runtime sibling-worktree import is enabled")
        return integration

    def _preflight(self) -> dict[str, Any]:
        self._authenticate()
        endpoint = load_json(self.spec["artifact_paths"]["endpoint_resource_lock"])
        if validate_artifact(endpoint, contract=ENDPOINT_RESOURCE_LOCK_CONTRACT) != (
            self.spec["parents"]["endpoint_resources"]
        ):
            raise ValueError("TRI60 endpoint/resource lock changed")
        free = shutil.disk_usage(self.root).free
        if free < int(self.spec["minimum_free_disk_bytes"]):
            raise OSError(
                f"TRI60 free disk is below reserve: {free} < "
                f"{self.spec['minimum_free_disk_bytes']}"
            )
        forbidden = tuple(self.root.rglob("*resume*"))
        if forbidden:
            raise PermissionError(f"TRI60 campaign contains forbidden resume paths: {forbidden}")
        if self.spec.get("ordinary_final_test_capability") is not False:
            raise PermissionError("TRI60 ordinary final-test capability is present")
        return endpoint

    def run(self, task_id: str, *, device: str = "cuda") -> dict[str, Any]:
        if task_id not in self.tasks:
            raise KeyError("unknown TRI60 task")
        task = self.tasks[task_id]
        kind = task["kind"]
        if kind == "authenticate":
            return self._authenticate()
        if kind == "preflight":
            return self._preflight()
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
            aggregate = build_aggregate(self.spec)
            write_immutable_json(self.root / "reports/validation_aggregate.json", aggregate)
            return aggregate
        if kind == "finalist_lock":
            aggregate = load_json(self.root / "reports/validation_aggregate.json")
            aggregate_hash = validate_artifact(aggregate, contract=AGGREGATE_CONTRACT)
            deployable = publish_m2_deployable(self.spec)
            deployable_hash = validate_artifact(
                deployable, contract=DEPLOYABLE_CHECKPOINT_CONTRACT,
            )
            lock = artifact({
                "parents": {
                    "campaign_spec": self.spec["content_hash"],
                    "aggregate": aggregate_hash,
                    "m2_deployable": deployable_hash,
                },
                "registered_validation_finalists": [
                    "M0paired", "LOGIT_D000E", "RSET_D000E", "RREL_D000E",
                    "M1_LOGIT", "M1_RSET", "M1_RREL", "M1E", "M2",
                ],
                "designated_single_model": "M2",
                "unlocks_final_test": False,
                "human_execution_lock_required": True,
                "scientific_result_does_not_control_completion": True,
                "final_test_accessed": False,
            }, contract=FINALIST_LOCK_CONTRACT)
            write_immutable_json(self.root / "locks/finalist.json", lock)
            return lock
        if kind == "campaign_complete":
            aggregate = load_json(self.root / "reports/validation_aggregate.json")
            finalist = load_json(self.root / "locks/finalist.json")
            aggregate_hash = validate_artifact(aggregate, contract=AGGREGATE_CONTRACT)
            finalist_hash = validate_artifact(finalist, contract=FINALIST_LOCK_CONTRACT)
            complete = artifact({
                "parents": {
                    "campaign_spec": self.spec["content_hash"],
                    "aggregate": aggregate_hash, "finalist_lock": finalist_hash,
                },
                "fresh_fit_count": 32, "ensemble_reducer_count": 12,
                "shared_u000_probability_bank_count": 1,
                "all_registered_science_executed": True,
                "representation_target_durable_bytes": 0,
                "rolling_resume_durable_bytes": 0,
                "scientific_result_does_not_control_completion": True,
                "final_test_accessed": False,
            }, contract=CAMPAIGN_COMPLETE_CONTRACT)
            write_immutable_json(self.root / "reports/campaign_complete.json", complete)
            return complete
        raise RuntimeError(f"unhandled TRI60 task kind: {kind}")


__all__ = ["Tri60Workflow", "task_outputs"]

"""Filesystem task dispatcher for direct offline-to-HLT distillation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import (
    load_json, validate_content_hash, with_content_hash,
    write_immutable_json,
)

from .hcwdl_direct_offline_kd_campaign import build_command_plan, validate_campaign
from .hcwdl_direct_offline_kd_reporting import (
    AGGREGATE_CONTRACT, COMPLETION_CONTRACT, build_aggregate, build_completion,
)
from .hcwdl_direct_offline_kd_runner import (
    build_target_bank, node_output_dir, target_output_dir, train_base_node,
    train_representation_node,
)
from .hcwdl_direct_offline_kd_targets import (
    CONSUMERS, TARGET_CLEANUP_AUTHORIZATION_CONTRACT, TARGET_CLEANUP_CONTRACT,
    authorize_target_cleanup, complete_target_cleanup,
)
from .hcwdl_recovery import (
    build_task_attestation, task_attestation_path, validate_task_attestation,
)


AUTHENTICATION_CONTRACT = "HCWDL_DIRECT_OFFLINE_KD_AUTHENTICATION/v1"


class DirectOfflineKdWorkflow:
    def __init__(self, spec: Mapping[str, Any], *, repository: str | Path) -> None:
        validate_campaign(spec, executable=False)
        self.spec = dict(spec); self.root = Path(spec["campaign_root"])
        self.repository = Path(repository).resolve()
        if self.repository != Path(spec["project_dir"]).resolve():
            raise PermissionError("direct KD worker repository differs")

    def _task(self, task_id: str) -> Mapping[str, Any]:
        tasks = {row["task_id"]: row for row in self.spec["tasks"]}
        if task_id not in tasks:
            raise KeyError(f"unknown direct KD task {task_id!r}")
        return tasks[task_id]

    def _dependencies(self, task: Mapping[str, Any]) -> None:
        for dependency in task["dependencies"]:
            path = task_attestation_path(self.root, dependency, None)
            value = load_json(path)
            validate_task_attestation(
                value, campaign_spec_sha256=self.spec["content_hash"],
                task_id=dependency, array_index=None,
            )

    def _attest(self, task_id: str, outputs: list[Path]) -> list[Path]:
        if not outputs or any(not path.is_file() for path in outputs):
            raise FileNotFoundError("direct KD task did not publish every output")
        value = build_task_attestation(
            campaign_spec_sha256=self.spec["content_hash"], task_id=task_id,
            array_index=None, outputs=outputs,
        )
        path = task_attestation_path(self.root, task_id, None)
        if path.exists():
            if load_json(path) != value:
                raise FileExistsError("direct KD task attestation differs")
        else:
            write_immutable_json(path, value)
        return [*outputs, path]

    def run(self, task_id: str, *, device: str = "cuda") -> list[Path]:
        task = self._task(task_id); self._dependencies(task); kind = task["kind"]
        if kind == "authenticate":
            validate_campaign(self.spec, executable=True)
            plan = build_command_plan(self.spec)
            output = self.root / "runtime/authenticated.json"
            value = with_content_hash({
                "contract": AUTHENTICATION_CONTRACT, "schema_version": 1,
                "campaign_spec_sha256": self.spec["content_hash"],
                "command_plan_sha256": plan["content_hash"],
                "source_commit": self.spec["source_commit"],
                "project_dir": str(self.repository), "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
                "final_test_accessed": False,
            })
            if not output.exists(): write_immutable_json(output, value)
            elif load_json(output) != value: raise FileExistsError("direct KD authentication differs")
            return self._attest(task_id, [output])
        if kind == "train_base":
            node_id = str(task["node_id"]); train_base_node(self.spec, node_id=node_id, device=device)
            output = node_output_dir(self.root, node_id)
            return self._attest(task_id, [
                output / "training_report.json", output / "hcwdl_training_report.json",
                output / "direct_report.json",
            ])
        if kind == "target":
            build_target_bank(self.spec, device=device)
            output = target_output_dir(self.root)
            return self._attest(task_id, [output / "target_spec.json", output / "generation.json",
                                          output / "manifest.json"])
        if kind == "train_representation":
            from .hcwdl_representation_task_runtime import RepresentationPreemptionMonitor
            node_id = str(task["node_id"]); monitor = RepresentationPreemptionMonitor(); monitor.install()
            try:
                train_representation_node(
                    self.spec, node_id=node_id, device=device,
                    preemption_requested=monitor.is_requested,
                )
            finally:
                monitor.restore()
            output = node_output_dir(self.root, node_id)
            return self._attest(task_id, [output / "training_report.json", output / "direct_report.json"])
        if kind == "aggregate":
            output = self.root / "reports/validation_aggregate.json"
            if not output.exists(): write_immutable_json(output, build_aggregate(self.spec))
            validate_content_hash(load_json(output), expected_contract=AGGREGATE_CONTRACT, expected_schema_version=1)
            return self._attest(task_id, [output])
        if kind == "cleanup_targets":
            authorization_path = self.root / "targets/target_cleanup_authorization.json"
            output = self.root / "targets/target_cleanup.json"
            manifest = load_json(target_output_dir(self.root) / "manifest.json")
            if not authorization_path.exists():
                reports = {
                    node: load_json(node_output_dir(self.root, node) / "direct_report.json")["content_hash"]
                    for node in CONSUMERS
                }
                write_immutable_json(
                    authorization_path,
                    authorize_target_cleanup(manifest, consumer_reports=reports),
                )
            validate_content_hash(
                load_json(authorization_path),
                expected_contract=TARGET_CLEANUP_AUTHORIZATION_CONTRACT,
                expected_schema_version=1,
            )
            if not output.exists():
                write_immutable_json(
                    output, complete_target_cleanup(
                        manifest, authorization=load_json(authorization_path),
                    ),
                )
            validate_content_hash(load_json(output), expected_contract=TARGET_CLEANUP_CONTRACT, expected_schema_version=1)
            return self._attest(task_id, [authorization_path, output])
        if kind == "campaign_complete":
            aggregate = load_json(self.root / "reports/validation_aggregate.json")
            cleanup = load_json(self.root / "targets/target_cleanup.json")
            output = self.root / "reports/campaign_complete.json"
            value = build_completion(self.spec, aggregate=aggregate, cleanup=cleanup)
            if not output.exists(): write_immutable_json(output, value)
            elif load_json(output) != value: raise FileExistsError("direct KD completion differs")
            validate_content_hash(load_json(output), expected_contract=COMPLETION_CONTRACT, expected_schema_version=1)
            return self._attest(task_id, [output])
        raise RuntimeError(f"unhandled direct KD task kind {kind!r}")


__all__ = ["AUTHENTICATION_CONTRACT", "DirectOfflineKdWorkflow"]

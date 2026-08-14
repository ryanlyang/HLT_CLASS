"""Thin filesystem dispatcher for HCWDL homotopy representation tasks."""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, write_immutable_json,
)

from .hcwdl_homotopy_representation_campaign import (
    authenticate_parent, build_command_plan, validate_campaign,
)
from .hcwdl_homotopy_representation_contracts import (
    AGGREGATE_CONTRACT, CAMPAIGN_COMPLETE_CONTRACT,
    GRAPH_RECIPE_LOCK_CONTRACT, TASK_ATTESTATION_CONTRACT,
    RUNTIME_BINDING_CONTRACT, TRAINING_REPORT_CONTRACT, build_artifact,
    validate_artifact,
)
from .hcwdl_homotopy_representation_reporting import (
    build_aggregate, build_campaign_complete,
)
from .hcwdl_homotopy_representation_targets import validate_target_manifest
from .hcwdl_homotopy_representation_training import (
    build_target_bank, node_output_dir, target_output_dir, train_node,
)


class HomotopyRepresentationWorkflow:
    def __init__(
        self, spec: Mapping[str, Any], *, repository: str | Path,
        producer_commit: str | None = None, recovery_sha256: str | None = None,
    ) -> None:
        validate_campaign(
            spec, executable=False,
            verify_source=recovery_sha256 is None,
        )
        self.spec = dict(spec)
        self.root = Path(spec["campaign_root"])
        self.repository = Path(repository).resolve()
        self.producer_commit = str(producer_commit or spec["source_commit"])
        self.recovery_sha256 = recovery_sha256
        if self.producer_commit != spec["source_commit"] and recovery_sha256 is None:
            raise PermissionError("HCWDL-U-RKD corrected source lacks recovery authority")

    def _task(self, task_id: str) -> Mapping[str, Any]:
        rows = {row["task_id"]: row for row in self.spec["tasks"]}
        if task_id not in rows:
            raise KeyError(f"unknown HCWDL-U-RKD task {task_id!r}")
        return rows[task_id]

    def _attest(self, task: Mapping[str, Any], outputs: list[Path]) -> list[Path]:
        if not outputs or any(not path.is_file() for path in outputs):
            raise FileNotFoundError("HCWDL-U-RKD task did not publish every output")
        artifact = build_artifact(
            TASK_ATTESTATION_CONTRACT,
            parents={
                "campaign_spec": self.spec["content_hash"],
                **({"recovery": self.recovery_sha256} if self.recovery_sha256 else {}),
            },
            task_id=task["task_id"], kind=task["kind"],
            dependencies=list(task["dependencies"]),
            producer_commit=self.producer_commit,
            command_plan_sha256=self.spec["command_plan_sha256"],
            outputs=[{
                "path": str(path.resolve()), "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            } for path in outputs],
            scheduler_job_id=os.environ.get("SLURM_JOB_ID"),
        )
        path = self.root / "attestations" / f"{task['task_id']}.json"
        if path.is_file():
            existing = load_json(path)
            if existing != artifact:
                raise ValueError("HCWDL-U-RKD task attestation changed")
        else:
            write_immutable_json(path, artifact)
        return [*outputs, path]

    def run(self, task_id: str, *, device: str = "cuda") -> list[Path]:
        task = self._task(task_id)
        for dependency in task["dependencies"]:
            attestation_path = self.root / "attestations" / f"{dependency}.json"
            dependency_attestation = load_json(attestation_path)
            validate_artifact(
                dependency_attestation, contract=TASK_ATTESTATION_CONTRACT,
                required_fields=("task_id", "outputs"),
            )
            if dependency_attestation["task_id"] != dependency:
                raise ValueError("HCWDL-U-RKD dependency attestation differs")
        kind = task["kind"]
        if kind == "authenticate":
            validate_campaign(
                self.spec, executable=True,
                verify_source=self.recovery_sha256 is None,
            )
            parent = authenticate_parent(self.spec["parent_homotopy_spec_path"])
            if parent["spec_sha256"] != self.spec["parent_homotopy_spec_sha256"]:
                raise ValueError("HCWDL-U-RKD parent changed")
            if (
                parent["locks"]["coupling_lock"] != self.spec["coupling_lock_sha256"]
                or parent["locks"]["endpoint_equality_lock"] != self.spec["endpoint_lock_sha256"]
                or parent["locks"]["graph_recipe_lock"]
                != load_json(self.root / "parent_import.json")["parents"]["graph_recipe_lock"]
            ):
                raise ValueError("HCWDL-U-RKD parent training locks changed")
            plan = build_command_plan(self.spec)
            if plan["content_hash"] != self.spec["command_plan_sha256"]:
                raise ValueError("HCWDL-U-RKD command plan changed")
            output = self.root / "runtime/authenticated.json"
            artifact = build_artifact(
                RUNTIME_BINDING_CONTRACT,
                parents={
                    "campaign_spec": self.spec["content_hash"],
                    "parent_homotopy_spec": parent["spec_sha256"],
                },
                source_commit=self.producer_commit,
                project_dir=str(self.repository), no_external_worktree_imports=True,
                parent_training_ready=True,
                parent_completion_required=False,
            )
            if not output.exists():
                write_immutable_json(output, artifact)
            elif load_json(output) != artifact:
                raise ValueError("HCWDL-U-RKD runtime binding changed")
            return self._attest(task, [output])
        if kind == "graph_recipe_lock":
            output = self.root / "locks/graph_recipe_lock.json"
            lock = load_json(output)
            validate_artifact(
                lock, contract=GRAPH_RECIPE_LOCK_CONTRACT,
                required_parents=(
                    "parent_import", "integration_attestation", "graph",
                    "combined_recipe", "recipe_compatibility",
                ),
            )
            return self._attest(task, [output])
        if kind == "target":
            manifest = build_target_bank(
                self.spec, bank_id=str(task["bank_id"]), device=device,
                producer_commit=self.producer_commit,
            )
            validate_target_manifest(manifest)
            output = target_output_dir(
                self.root, str(task["bank_id"]),
            ) / "manifest.json"
            return self._attest(task, [output])
        if kind == "train":
            from .hcwdl_representation_task_runtime import RepresentationPreemptionMonitor

            node_id = str(task["node_id"])
            monitor = RepresentationPreemptionMonitor()
            monitor.install()
            try:
                train_node(
                    self.spec, node_id=node_id, device=device,
                    preemption_requested=monitor.is_requested,
                    producer_commit=self.producer_commit,
                    recovery_sha256=self.recovery_sha256,
                )
            finally:
                monitor.restore()
            output = node_output_dir(
                self.root, node_id,
            ) / "combined_training_report.json"
            validate_artifact(
                load_json(output), contract=TRAINING_REPORT_CONTRACT,
                required_parents=("campaign_spec", "engine_report"),
            )
            return self._attest(task, [output])
        if kind == "aggregate":
            output = self.root / "reports/validation_aggregate.json"
            if output.exists():
                value = load_json(output)
                validate_artifact(value, contract=AGGREGATE_CONTRACT)
            else:
                write_immutable_json(output, build_aggregate(self.spec))
            return self._attest(task, [output])
        if kind == "campaign_complete":
            output = self.root / "reports/campaign_complete.json"
            aggregate = load_json(self.root / "reports/validation_aggregate.json")
            if output.exists():
                value = load_json(output)
                validate_artifact(value, contract=CAMPAIGN_COMPLETE_CONTRACT)
            else:
                write_immutable_json(
                    output, build_campaign_complete(self.spec, aggregate),
                )
            return self._attest(task, [output])
        raise ValueError(f"unsupported HCWDL-U-RKD task kind {kind!r}")


__all__ = ["HomotopyRepresentationWorkflow"]

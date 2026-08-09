"""Filesystem-bound task dispatcher for the dense cold 300k supplement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hlt_classification.data.cache_contracts import (
    load_json, validate_content_hash, write_immutable_json,
)

from .engine import validate_pmard_training_report
from .hcwdl_dense import (
    DENSE_REPAIR_RNG_POLICY, build_dense_aggregate,
    dense_profile_for_spec, validate_dense_spec,
)
from .hcwdl_dense_runner import run_dense_node


class DenseColdWorkflow:
    def __init__(self, spec: dict[str, Any]) -> None:
        validate_dense_spec(spec, executable=False)
        self.spec = spec
        self.profile = dense_profile_for_spec(spec)
        self.root = Path(spec["campaign_root"])

    def _engine_report(self, node_id: str) -> Path:
        return self.root / f"training/{node_id}/training_report.json"

    def _node_report(self, node_id: str) -> Path:
        return self.root / f"training/{node_id}/hcwdl_training_report.json"

    def _validate_engine_node(self, engine: dict[str, Any], node_id: str) -> str:
        engine_hash = validate_pmard_training_report(engine)
        scientific = engine.get("scientific_config")
        node = None if not isinstance(scientific, dict) else scientific.get("node")
        if (
            not isinstance(scientific, dict)
            or scientific.get("campaign") != self.profile.campaign
            or scientific.get("graph_sha256") != self.profile.graph_sha256
            or scientific.get("repair_rng_policy") != DENSE_REPAIR_RNG_POLICY
            or not isinstance(node, dict)
            or node.get("node_id") != node_id
        ):
            raise ValueError(f"dense cold engine lineage differs for {node_id}")
        return engine_hash

    def _teacher_report(self, node_id: str) -> tuple[Path, str]:
        teacher_id = self.profile.registry[node_id].teachers[0].node_id
        if teacher_id in self.spec["imported_controls"]:
            record = self.spec["imported_controls"][teacher_id]
            return Path(record["report_path"]), str(record["report_sha256"])
        engine_path = self._engine_report(teacher_id)
        engine = load_json(engine_path)
        engine_hash = self._validate_engine_node(engine, teacher_id)
        node = load_json(self._node_report(teacher_id))
        validate_content_hash(
            node, expected_contract=self.profile.training_report_contract,
            expected_schema_version=1,
        )
        if (
            node.get("node_id") != teacher_id
            or node.get("graph_sha256") != self.profile.graph_sha256
            or node.get("pmard_engine_report_sha256") != engine_hash
            or node.get("complete") is not True
        ):
            raise ValueError(f"dense cold teacher lineage differs for {teacher_id}")
        return engine_path, engine_hash

    def run(self, task_id: str) -> list[Path]:
        by_task = {row["task_id"]: row for row in self.spec["tasks"]}
        if task_id not in by_task:
            raise ValueError(f"task {task_id!r} is absent from dense cold spec")
        task = by_task[task_id]
        if task["kind"] == "train_node":
            node_id = str(task["node_id"])
            output = self.root / f"training/{node_id}"
            teacher_path, teacher_hash = self._teacher_report(node_id)
            run_dense_node(
                node_id=node_id,
                recipe_path=self.spec["recipe_path"],
                split_manifest_path=self.spec["split_manifest_path"],
                selection_manifest_path=self.spec["selection_manifest_path"],
                data_root=self.spec["data_root"],
                assignment_manifests=self.spec["assignment_manifests"],
                output_dir=output,
                teacher_report_path=teacher_path,
                replicate_seed=int(self.spec["replicate_seed"]),
                source_snapshot_sha256=self.spec["source_snapshot_sha256"],
                assignment_lock_sha256=self.spec["assignment_lock_sha256"],
                qualification_lock_sha256=self.spec["qualification_lock_sha256"],
                parent_campaign_spec_sha256=self.spec["parent_campaign_spec_sha256"],
                expected_recipe_sha256=self.spec["recipe_sha256"],
                expected_split_manifest_sha256=self.spec["split_manifest_sha256"],
                expected_selection_manifest_sha256=self.spec["selection_manifest_sha256"],
                expected_assignment_manifest_sha256=self.spec[
                    "assignment_manifest_sha256"
                ],
                expected_teacher_report_sha256=teacher_hash,
                device="cuda",
                registry=self.profile.registry,
                domains=self.profile.domains,
                graph_sha256=self.profile.graph_sha256,
                training_report_contract=self.profile.training_report_contract,
                node_contract=self.profile.node_contract,
                campaign_label=self.profile.campaign,
                rung_step=self.profile.rung_step,
            )
            return [self._engine_report(node_id), self._node_report(node_id)]
        if task_id == "aggregate":
            reports = {
                node: load_json(record["report_path"])
                for node, record in self.spec["imported_controls"].items()
            }
            for node_id, record in self.spec["imported_controls"].items():
                if reports[node_id].get("content_hash") != record["report_sha256"]:
                    raise ValueError(f"dense cold imported {node_id} report hash differs")
            for node_id in self.profile.registry:
                engine = load_json(self._engine_report(node_id))
                engine_hash = self._validate_engine_node(engine, node_id)
                node = load_json(self._node_report(node_id))
                validate_content_hash(
                    node, expected_contract=self.profile.training_report_contract,
                    expected_schema_version=1,
                )
                if (
                    node.get("node_id") != node_id
                    or node.get("graph_sha256") != self.profile.graph_sha256
                    or node.get("pmard_engine_report_sha256") != engine_hash
                    or node.get("selected_checkpoint_sha256")
                    != engine.get("selected_checkpoint_sha256")
                ):
                    raise ValueError(f"dense cold node report lineage differs for {node_id}")
                reports[node_id] = engine
            aggregate = build_dense_aggregate(spec=self.spec, reports=reports)
            output = self.root / f"reports/{self.profile.aggregate_filename}"
            write_immutable_json(output, aggregate)
            return [output]
        raise RuntimeError("unreachable dense cold task")


__all__ = ["DenseColdWorkflow"]

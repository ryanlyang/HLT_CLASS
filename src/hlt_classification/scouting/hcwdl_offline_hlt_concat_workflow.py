"""Fail-closed task workflow and reporting for tagged concatenation."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, validate_content_hash, write_immutable_json,
)

from .hcwdl_mhpe_tri60_ce_control_contracts import (
    TRAINING_REPORT_CONTRACT as CE60_REPORT_CONTRACT,
)
from .hcwdl_offline_hlt_concat_campaign import (
    validate_campaign, validate_source_lock,
)
from .hcwdl_offline_hlt_concat_contracts import (
    AGGREGATE_CONTRACT, CAPACITY_AUDIT_CONTRACT, COMPLETE_CONTRACT,
    EXECUTION_ACCEPTANCE_CONTRACT, TRAINING_REPORT_CONTRACT, artifact,
    validate_artifact,
)
from .hcwdl_offline_hlt_concat_graph import (
    GRAPH_SHA256, MODEL_INPUT_PROTOCOL, NODE_ID, node,
)
from .hcwdl_offline_hlt_concat_runner import (
    build_capacity_audit, run_execution_acceptance, run_fit,
)
from .hcwdl_tri100_spine4_bottleneck_contracts import (
    TRAINING_REPORT_CONTRACT as PERSISTENT_REPORT_CONTRACT,
)
from .hcwdl_tri100_spine4_bottleneck_workflow import _delta, _recovery


def _training_report(spec: Mapping[str, Any]) -> dict[str, Any]:
    directory = Path(spec["campaign_root"]) / "training" / NODE_ID
    report = load_json(directory / "training_report.json")
    validate_content_hash(
        report, expected_contract=TRAINING_REPORT_CONTRACT,
        expected_schema_version=1,
    )
    selected = directory / str(report.get("selected_checkpoint", ""))
    final = directory / str(report.get("final_checkpoint", ""))
    if (
        report.get("node_id") != NODE_ID
        or report.get("node_spec") != node().payload()
        or report.get("campaign_spec_sha256") != spec["content_hash"]
        or report.get("graph_sha256") != GRAPH_SHA256
        or report.get("recipe_sha256") != spec["parents"]["recipe"]
        or report.get("model_input_protocol") != MODEL_INPUT_PROTOCOL
        or report.get("passes") != 60 or report.get("validations") != 60
        or report.get("complete") is not True
        or report.get("rolling_resume_published") is not False
        or report.get("partial_checkpoint_reuse") is not False
        or report.get("distributed_execution") is not None
        or report.get("final_test_accessed") is not False
        or not selected.is_file() or not final.is_file()
        or sha256_file(selected) != report.get("selected_checkpoint_sha256")
        or sha256_file(final) != report.get("final_checkpoint_sha256")
    ):
        raise ValueError("tagged concatenation training report differs")
    return report


def task_outputs(spec: Mapping[str, Any], task_id: str) -> list[Path]:
    tasks = {row["task_id"]: row for row in spec["tasks"]}
    if task_id not in tasks:
        raise KeyError("unknown tagged concatenation task")
    kind = tasks[task_id]["kind"]
    root = Path(spec["campaign_root"])
    if kind == "authenticate":
        return [Path(spec["artifact_paths"]["source_lock"])]
    if kind == "capacity_audit":
        value = load_json(spec["artifact_paths"]["capacity_audit"])
        validate_artifact(value, contract=CAPACITY_AUDIT_CONTRACT)
        return [Path(spec["artifact_paths"]["capacity_audit"])]
    if kind == "preflight":
        value = load_json(spec["artifact_paths"]["execution_acceptance"])
        validate_artifact(value, contract=EXECUTION_ACCEPTANCE_CONTRACT)
        return [Path(spec["artifact_paths"]["execution_acceptance"])]
    if kind == "train":
        report = _training_report(spec)
        directory = root / "training" / NODE_ID
        return [
            directory / "training_report.json",
            directory / report["selected_checkpoint"],
            directory / report["final_checkpoint"],
        ]
    if kind == "aggregate":
        value = load_json(root / "reports/validation_aggregate.json")
        validate_artifact(value, contract=AGGREGATE_CONTRACT)
        return [root / "reports/validation_aggregate.json"]
    value = load_json(root / "reports/campaign_complete.json")
    validate_artifact(value, contract=COMPLETE_CONTRACT)
    return [root / "reports/campaign_complete.json"]


def build_aggregate(spec: Mapping[str, Any]) -> dict[str, Any]:
    validate_campaign(spec)
    source = load_json(spec["artifact_paths"]["source_lock"])
    validate_source_lock(source)
    baseline_report = load_json(spec["artifact_paths"]["m0ce60_report"])
    validate_content_hash(
        baseline_report, expected_contract=CE60_REPORT_CONTRACT,
        expected_schema_version=1,
    )
    oracle_report = load_json(spec["artifact_paths"]["pure_offline_u000_report"])
    validate_content_hash(
        oracle_report, expected_contract=str(oracle_report["contract"]),
        expected_schema_version=int(oracle_report["schema_version"]),
    )
    persistent_report = load_json(spec["artifact_paths"]["persistent_anchor_report"])
    validate_content_hash(
        persistent_report, expected_contract=PERSISTENT_REPORT_CONTRACT,
        expected_schema_version=1,
    )
    fit = _training_report(spec)
    baseline = baseline_report["validation"]
    oracle = oracle_report["validation"]
    persistent = persistent_report["validation"]
    rows = [
        {"artifact_id": "M0CE60", "kind": "hlt_baseline", "state": "complete",
         "metrics": baseline, "recovery": _recovery(baseline, baseline, oracle),
         "report_sha256": baseline_report["content_hash"]},
        {"artifact_id": "U000", "kind": "pure_offline_oracle", "state": "complete",
         "metrics": oracle, "recovery": _recovery(oracle, baseline, oracle),
         "report_sha256": oracle_report["content_hash"]},
        {"artifact_id": "SP4P_U000", "kind": "persistent_support_oracle",
         "state": "complete", "metrics": persistent,
         "recovery": _recovery(persistent, baseline, oracle),
         "report_sha256": persistent_report["content_hash"]},
        {"artifact_id": NODE_ID, "kind": "tagged_concat_oracle",
         "state": "complete", "metrics": fit["validation"],
         "recovery": _recovery(fit["validation"], baseline, oracle),
         "report_sha256": fit["content_hash"],
         "selected_pass": int(fit["selected_pass"]),
         "completed_passes": int(fit["passes"])},
    ]
    return artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "source_lock": spec["parents"]["source_lock"],
            "capacity_audit": validate_artifact(
                load_json(spec["artifact_paths"]["capacity_audit"]),
                contract=CAPACITY_AUDIT_CONTRACT,
            ),
            "execution_acceptance": validate_artifact(
                load_json(spec["artifact_paths"]["execution_acceptance"]),
                contract=EXECUTION_ACCEPTANCE_CONTRACT,
            ),
            "m0ce60_report": baseline_report["content_hash"],
            "pure_offline_u000_report": oracle_report["content_hash"],
            "persistent_anchor_report": persistent_report["content_hash"],
            "concat_report": fit["content_hash"],
        },
        "rows": rows,
        "paired_differences": {
            "concat_minus_m0ce60": _delta(fit["validation"], baseline),
            "concat_minus_u000": _delta(fit["validation"], oracle),
            "concat_minus_sp4p_u000": _delta(fit["validation"], persistent),
        },
        "recovery_convention": "M0CE60_zero_U000_one_v1",
        "input_sequence": "offline_then_hlt_v1",
        "content_source_embedding": True,
        "matching_indices_are_model_inputs": False,
        "fresh_fit_count": 1,
        "poor_metrics_do_not_control_completion": True,
        "ordinary_access_roles": ["validation"],
        "final_test_accessed": False,
    }, contract=AGGREGATE_CONTRACT)


class TaggedConcatWorkflow:
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

    def run(self, task_id: str, *, device: str = "cuda") -> dict[str, Any]:
        if task_id not in self.tasks:
            raise KeyError("unknown tagged concatenation task")
        kind = self.tasks[task_id]["kind"]
        if kind == "authenticate":
            source = load_json(self.spec["artifact_paths"]["source_lock"])
            if validate_source_lock(source) != self.spec["parents"]["source_lock"]:
                raise ValueError("tagged concatenation source lock changed")
            return source
        if kind == "capacity_audit":
            value = build_capacity_audit(self.spec)
            write_immutable_json(self.spec["artifact_paths"]["capacity_audit"], value)
            return value
        if kind == "preflight":
            audit = load_json(self.spec["artifact_paths"]["capacity_audit"])
            validate_artifact(audit, contract=CAPACITY_AUDIT_CONTRACT)
            required = int(self.spec["minimum_free_disk_bytes"]) + int(
                self.spec["projected_durable_bytes"]
            )
            if shutil.disk_usage(self.root).free < required:
                raise OSError("tagged concatenation free disk cannot preserve reserve")
            if tuple(self.root.rglob("*resume*")):
                raise PermissionError("tagged concatenation root contains resume state")
            value = run_execution_acceptance(self.spec, device=device)
            write_immutable_json(
                self.spec["artifact_paths"]["execution_acceptance"], value,
            )
            return value
        if kind == "train":
            return run_fit(
                self.spec, device=device,
                recovery_spec_sha256=self.recovery_spec_sha256,
                execution_source_commit=self.execution_source_commit,
            )
        if kind == "aggregate":
            value = build_aggregate(self.spec)
            write_immutable_json(self.root / "reports/validation_aggregate.json", value)
            return value
        if kind == "campaign_complete":
            aggregate = load_json(self.root / "reports/validation_aggregate.json")
            aggregate_hash = validate_artifact(aggregate, contract=AGGREGATE_CONTRACT)
            durable = tuple(path for path in self.root.rglob("*") if path.is_file())
            forbidden = [
                str(path.relative_to(self.root)) for path in durable
                if "resume" in path.name.lower() or "optimizer" in path.name.lower()
            ]
            if forbidden:
                raise PermissionError("tagged concatenation durable state differs")
            durable_bytes = sum(path.stat().st_size for path in durable)
            if durable_bytes > int(self.spec["projected_durable_bytes"]):
                raise OSError("tagged concatenation durable bytes exceed projection")
            value = artifact({
                "parents": {
                    "campaign_spec": self.spec["content_hash"],
                    "aggregate": aggregate_hash,
                },
                "fresh_fit_count": 1, "science_node": NODE_ID,
                "all_registered_science_executed": True,
                "durable_particle_view_bytes": 0,
                "rolling_resume_durable_bytes": 0,
                "optimizer_state_durable_bytes": 0,
                "durable_bytes_before_completion": durable_bytes,
                "scientific_result_does_not_control_completion": True,
                "source_campaign_outputs_mutated": False,
                "final_test_accessed": False,
            }, contract=COMPLETE_CONTRACT)
            write_immutable_json(self.root / "reports/campaign_complete.json", value)
            return value
        raise RuntimeError(f"unhandled tagged concatenation task: {kind}")


__all__ = ["TaggedConcatWorkflow", "build_aggregate", "task_outputs"]

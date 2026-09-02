"""Fail-closed workflow and combined reporting for fusion withdrawal."""

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
from .hcwdl_offline_hlt_fusion_campaign import validate_campaign
from .hcwdl_offline_hlt_fusion_contracts import (
    AGGREGATE_CONTRACT, ALPHA_ZERO_AUDIT_CONTRACT, COMPLETE_CONTRACT,
    TRAINING_REPORT_CONTRACT, artifact,
    validate_artifact,
)
from .hcwdl_offline_hlt_fusion_graph import (
    FIT_ORDER, GRAPH_SHA256, NODE_REGISTRY, STUDY_C_NODES,
    TEACHER_DISTRIBUTION,
)
from .hcwdl_offline_hlt_fusion_runner import (
    build_capacity_audit, run_alpha_zero_extraction,
    run_execution_acceptance, run_fit, run_teacher_bank,
)
from .hcwdl_offline_hlt_fusion_probability import (
    validate_lock as validate_probability_lock,
)
from .hcwdl_tri100_spine4_bottleneck_source import validate_source_lock
from .hcwdl_tri100_spine4_bottleneck_workflow import _delta, _recovery


def training_report(spec: Mapping[str, Any], node_id: str):
    path = Path(spec["campaign_root"]) / "training" / node_id / "training_report.json"
    value = load_json(path)
    validate_content_hash(
        value, expected_contract=TRAINING_REPORT_CONTRACT,
        expected_schema_version=1,
    )
    selected = path.parent / str(value.get("selected_checkpoint", ""))
    final = path.parent / str(value.get("final_checkpoint", ""))
    if (
        value.get("node_id") != node_id
        or value.get("node_spec") != NODE_REGISTRY[node_id].payload()
        or value.get("campaign_spec_sha256") != spec["content_hash"]
        or value.get("graph_sha256") != GRAPH_SHA256
        or value.get("recipe_sha256") != spec["parents"]["recipe"]
        or value.get("complete") is not True
        or value.get("rolling_resume_published") is not False
        or value.get("partial_checkpoint_reuse") is not False
        or value.get("final_test_accessed") is not False
        or not selected.is_file() or not final.is_file()
        or sha256_file(selected) != value.get("selected_checkpoint_sha256")
        or sha256_file(final) != value.get("final_checkpoint_sha256")
    ):
        raise ValueError("fusion training report differs")
    if node_id in {"FUSION_WITHDRAW_COS", "FUSION_WITHDRAW_STEP"} and (
        value.get("validation_route") != "exact_alpha_zero_v1"
        or value.get("checkpoint_selection_route") != "alpha_zero_macro_auc_v1"
    ):
        raise ValueError("withdrawal validation route differs")
    return value


def build_aggregate(spec: Mapping[str, Any]):
    validate_campaign(spec)
    baseline_report = load_json(spec["artifact_paths"]["m0ce60_report"])
    validate_content_hash(
        baseline_report, expected_contract=CE60_REPORT_CONTRACT,
        expected_schema_version=1,
    )
    oracle_report = load_json(spec["artifact_paths"]["pure_offline_u000_report"])
    validate_content_hash(
        oracle_report, expected_contract=oracle_report["contract"],
        expected_schema_version=int(oracle_report["schema_version"]),
    )
    baseline = baseline_report["validation"]
    oracle = oracle_report["validation"]
    rows = [
        {"artifact_id": "M0CE60", "kind": "hlt_baseline",
         "metrics": baseline, "recovery": _recovery(baseline, baseline, oracle),
         "report_sha256": baseline_report["content_hash"]},
        {"artifact_id": "U000", "kind": "pure_offline_oracle",
         "metrics": oracle, "recovery": _recovery(oracle, baseline, oracle),
         "report_sha256": oracle_report["content_hash"]},
    ]
    reports = {}
    for node_id in FIT_ORDER:
        report = training_report(spec, node_id)
        reports[node_id] = report
        rows.append({
            "artifact_id": node_id,
            "kind": (
                "withdrawn_hlt_endpoint" if node_id.startswith("FUSION_WITHDRAW")
                else "direct_hlt_kd" if node_id == "FUSION_DIRECT_KD_WARM"
                else "privileged_oracle_control"
            ),
            "metrics": report["validation"],
            "recovery": _recovery(report["validation"], baseline, oracle),
            "report_sha256": report["content_hash"],
            "selected_pass": int(report["selected_pass"]),
            "completed_passes": int(report["passes"]),
            "resource_diagnostics": {
                "parameter_scalar_count": int(
                    report.get("parameter_scalar_count", 0)
                ),
                "trainable_parameter_scalar_count": int(
                    report.get("trainable_parameter_scalar_count", 0)
                ),
                "runtime_seconds": float(report["runtime_seconds"]),
                "preparation_seconds": dict(report["preparation_seconds"]),
                "peak_rss_bytes": int(report["peak_rss_bytes"]),
                "peak_cuda_bytes": int(report["peak_cuda_bytes"]),
            },
        })
    audits = {}
    for node_id in ("FUSION_WITHDRAW_COS", "FUSION_WITHDRAW_STEP"):
        audit = load_json(
            Path(spec["campaign_root"]) / "deployable" / node_id
            / "alpha_zero_audit.json"
        )
        validate_artifact(audit, contract=ALPHA_ZERO_AUDIT_CONTRACT)
        if audit.get("node_id") != node_id:
            raise ValueError("fusion aggregate alpha-zero audit differs")
        audits[node_id] = audit
    bank_hash = validate_probability_lock(
        Path(spec["campaign_root"]) / "probabilities" / TEACHER_DISTRIBUTION
        / "lock.json",
        campaign_spec_sha256=spec["content_hash"],
        teacher_report_sha256=reports["ANCHORED_FUSION_OH"]["content_hash"],
    )
    return artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "m0ce60_report": baseline_report["content_hash"],
            "pure_offline_u000_report": oracle_report["content_hash"],
            "teacher_probability_lock": bank_hash,
            **{f"report_{name}": report["content_hash"] for name, report in reports.items()},
            **{f"alpha_zero_{name}": audit["content_hash"] for name, audit in audits.items()},
        },
        "rows": rows,
        "paired_differences": {
            "tagged_minus_untagged": _delta(
                reports["CONCAT_TAGGED"]["validation"],
                reports["CONCAT_UNTAGGED"]["validation"],
            ),
            "symmetric_oh_minus_oo": _delta(
                reports["SYMMETRIC_FUSION_OH"]["validation"],
                reports["SYMMETRIC_FUSION_OO"]["validation"],
            ),
            "symmetric_oh_minus_hh": _delta(
                reports["SYMMETRIC_FUSION_OH"]["validation"],
                reports["SYMMETRIC_FUSION_HH"]["validation"],
            ),
            "anchored_oh_minus_hh": _delta(
                reports["ANCHORED_FUSION_OH"]["validation"],
                reports["ANCHORED_FUSION_HH"]["validation"],
            ),
            "anchored_oh_minus_hlt_warm": _delta(
                reports["ANCHORED_FUSION_OH"]["validation"],
                reports["HLT_WARM_CONTINUE"]["validation"],
            ),
            "withdraw_cos_minus_direct_kd": _delta(
                reports["FUSION_WITHDRAW_COS"]["validation"],
                reports["FUSION_DIRECT_KD_WARM"]["validation"],
            ),
            "withdraw_cos_minus_step": _delta(
                reports["FUSION_WITHDRAW_COS"]["validation"],
                reports["FUSION_WITHDRAW_STEP"]["validation"],
            ),
        },
        "recovery_convention": "M0CE60_zero_U000_one_v1",
        "validation_route_for_withdrawal": "extracted_exact_hlt_alpha_zero_v1",
        "fresh_fit_count": 11,
        "study_c_executed_regardless_of_oracle_metrics": True,
        "poor_metrics_do_not_control_completion": True,
        "final_test_accessed": False,
    }, contract=AGGREGATE_CONTRACT)


def task_outputs(spec: Mapping[str, Any], task_id: str):
    tasks = {row["task_id"]: row for row in spec["tasks"]}
    if task_id not in tasks:
        raise KeyError("unknown fusion task")
    task = tasks[task_id]
    root = Path(spec["campaign_root"])
    if task["kind"] == "authenticate":
        return [Path(spec["artifact_paths"]["source_lock"])]
    if task["kind"] == "capacity_audit":
        return [Path(spec["artifact_paths"]["capacity_audit"])]
    if task["kind"] == "preflight":
        return [Path(spec["artifact_paths"]["execution_acceptance"])]
    if task["kind"] == "train":
        report = training_report(spec, task["node_id"])
        directory = root / "training" / task["node_id"]
        return [
            directory / "training_report.json",
            directory / report["selected_checkpoint"],
            directory / report["final_checkpoint"],
        ]
    if task["kind"] == "teacher_bank":
        directory = root / "probabilities" / TEACHER_DISTRIBUTION
        return [
            directory / "lock.json", directory / "train_manifest.json",
            directory / "validation_manifest.json", directory / "train_shard.json",
            directory / "validation_shard.json", directory / "train.npz",
            directory / "validation.npz",
        ]
    if task["kind"] == "extract":
        directory = root / "deployable" / task["node_id"]
        return [directory / "alpha_zero_audit.json", directory / "selected_hlt_model.pt"]
    if task["kind"] == "aggregate":
        return [root / "reports/validation_aggregate.json"]
    return [root / "reports/campaign_complete.json"]


class FusionWithdrawalWorkflow:
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

    def run(self, task_id: str, *, device: str = "cuda"):
        if task_id not in self.tasks:
            raise KeyError("unknown fusion task")
        task = self.tasks[task_id]
        kind = task["kind"]
        if kind == "authenticate":
            source = load_json(self.spec["artifact_paths"]["source_lock"])
            if validate_source_lock(source) != self.spec["parents"]["source_lock"]:
                raise ValueError("fusion source lock changed")
            return source
        if kind == "capacity_audit":
            value = build_capacity_audit(self.spec)
            write_immutable_json(self.spec["artifact_paths"]["capacity_audit"], value)
            return value
        if kind == "preflight":
            required = int(self.spec["minimum_free_disk_bytes"]) + int(
                self.spec["projected_durable_bytes"]
            )
            if shutil.disk_usage(self.root).free < required:
                raise OSError("fusion free disk cannot preserve reserve")
            if tuple(self.root.rglob("*resume*")):
                raise PermissionError("fusion campaign contains resume state")
            value = run_execution_acceptance(self.spec, device=device)
            write_immutable_json(self.spec["artifact_paths"]["execution_acceptance"], value)
            return value
        if kind == "train":
            return run_fit(
                self.spec, task["node_id"], device=device,
                recovery_spec_sha256=self.recovery_spec_sha256,
                execution_source_commit=self.execution_source_commit,
            )
        if kind == "teacher_bank":
            return run_teacher_bank(
                self.spec, device=device,
                recovery_spec_sha256=self.recovery_spec_sha256,
                execution_source_commit=self.execution_source_commit,
            )
        if kind == "extract":
            return run_alpha_zero_extraction(
                self.spec, task["node_id"], device=device,
                recovery_spec_sha256=self.recovery_spec_sha256,
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
                raise PermissionError("fusion durable state differs")
            durable_bytes = sum(path.stat().st_size for path in durable)
            if durable_bytes > int(self.spec["projected_durable_bytes"]):
                raise OSError("fusion durable bytes exceed projection")
            value = artifact({
                "parents": {
                    "campaign_spec": self.spec["content_hash"],
                    "aggregate": aggregate_hash,
                },
                "fresh_fit_count": 11, "all_registered_science_executed": True,
                "durable_particle_view_bytes": 0,
                "durable_hidden_state_bytes": 0,
                "rolling_resume_durable_bytes": 0,
                "optimizer_state_durable_bytes": 0,
                "durable_bytes_before_completion": durable_bytes,
                "scientific_result_does_not_control_completion": True,
                "source_campaign_outputs_mutated": False,
                "final_test_accessed": False,
            }, contract=COMPLETE_CONTRACT)
            write_immutable_json(self.root / "reports/campaign_complete.json", value)
            return value
        raise RuntimeError(f"unhandled fusion task kind: {kind}")


__all__ = [
    "FusionWithdrawalWorkflow", "build_aggregate", "task_outputs",
    "training_report",
]

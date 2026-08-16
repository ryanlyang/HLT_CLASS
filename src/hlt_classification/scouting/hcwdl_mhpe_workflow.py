"""Fail-closed task dispatcher and reporting for HCWDL-MHPE profiles."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import (
    load_json, validate_content_hash, with_content_hash, write_immutable_json,
)

from .hcwdl_mhpe_campaign import campaign_tasks, validate_campaign
from .hcwdl_mhpe_contracts import (
    aggregate_contract, campaign_profile, completion_contract,
    finalist_lock_contract, finalist_lock_payload, stage_report_contract,
    training_report_contract,
)
from .hcwdl_mhpe_graph import (
    endpoint_ensemble, ensemble_components, finalists, local_teacher,
    node_registry,
)
from .hcwdl_mhpe_runner import run_ensemble, run_specialist
from .engine import validate_pmard_training_report
from .hcwdl_mhpe_targets import validate_probability_bundle


def _metrics(report: Mapping[str, Any]) -> Mapping[str, Any]:
    value = report.get("validation")
    if not isinstance(value, Mapping):
        raise ValueError("HCWDL-MHPE training report lacks validation metrics")
    return value


def build_aggregate(spec: Mapping[str, Any]) -> dict[str, Any]:
    profile = campaign_profile(spec)
    registry = node_registry(profile)
    components = ensemble_components(profile)
    root = Path(spec["campaign_root"]); rows = []
    for node_id in registry:
        report = load_json(root / "training" / node_id / "training_report.json")
        outer = load_json(root / "training" / node_id / "hcwdl_training_report.json")
        runtime = load_json(root / "reports/runtime" / f"{node_id}.json")
        report_hash = validate_pmard_training_report(report)
        validate_content_hash(
            outer, expected_contract=training_report_contract(profile),
            expected_schema_version=1,
        )
        if (outer.get("node_id") != node_id
                or outer.get("graph_sha256") != spec["graph_sha256"]
                or outer.get("recipe_overlay_sha256") != spec["recipe_sha256"]
                or outer.get("pmard_engine_report_sha256") != report_hash):
            raise ValueError("HCWDL-MHPE outer training report lineage differs")
        validate_content_hash(
            runtime,
            expected_contract="HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_RUNTIME/v1",
            expected_schema_version=1,
        )
        rows.append({
            "kind": "model", "node_id": node_id,
            "teacher_id": registry[node_id].teacher_id,
            "metrics": dict(_metrics(report)),
            "selected_update": report["selected_update"],
            "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
            "report_sha256": report_hash,
            "runtime": runtime,
        })
    for ensemble_id in components:
        report = load_json(root / "reports" / f"{ensemble_id}_stage.json")
        validate_content_hash(report, expected_contract=stage_report_contract(profile), expected_schema_version=1)
        for temperature in (1.0, 2.0):
            consumers = sorted(
                node.node_id for node in registry.values()
                if node.teacher_id == ensemble_id and node.temperature == temperature
            )
            lock_hash, manifests = validate_probability_bundle(
                root / "targets" / ensemble_id / f"T{int(temperature)}",
                ensemble_id=ensemble_id, temperature=temperature,
                consumers=consumers, profile=profile,
            )
            label = f"T{int(temperature)}"
            if (report["target_lock_sha256"][label] != lock_hash
                    or any(
                        report["manifest_sha256"][f"{label}_{role}"]
                        != manifests[role]["content_hash"]
                        for role in ("train", "validation")
                    )):
                raise ValueError("HCWDL-MHPE stage report/target bundle differs")
        rows.append({
            "kind": "ensemble", "node_id": ensemble_id, "teacher_id": None,
            "metrics": report["ensemble_metrics"], "report_sha256": report["content_hash"],
            "components": list(components[ensemble_id]),
        })
    reuse = load_json(spec["reuse_lock_path"]); foundation_root = Path(reuse["foundation_spec_path"]).parent
    for node_id, path in (
        ("U000", foundation_root / "training/U000/training_report.json"),
        ("M0paired", foundation_root / "training/M0paired/training_report.json"),
    ):
        report = load_json(path)
        report_hash = validate_pmard_training_report(report)
        rows.append({"kind": "imported", "node_id": node_id, "teacher_id": None, "metrics": dict(_metrics(report)), "report_sha256": report_hash, "selected_checkpoint_sha256": report["selected_checkpoint_sha256"]})
    by_id = {row["node_id"]: row for row in rows}
    m0 = by_id["M0paired"]["metrics"]; u000 = by_id["U000"]["metrics"]
    for row in rows:
        recovery = {}
        for metric in ("macro_ovr_auc", "cross_entropy", "macro_mean_log_qcd_rejection_at_50pct_signal"):
            denominator = float(u000[metric]) - float(m0[metric])
            recovery[metric] = None if denominator == 0 else (float(row["metrics"][metric]) - float(m0[metric])) / denominator
        row["recovery_m0paired_to_u000"] = recovery
    endpoint = endpoint_ensemble(profile)
    endpoint_values = [by_id[name]["metrics"]["macro_ovr_auc"] for name in components[endpoint]]
    local_id = f"{endpoint[:-1]}_from_{local_teacher(profile, endpoint)}"
    comparisons = {
        f"{endpoint}_minus_best_specialist_auc": by_id[endpoint]["metrics"]["macro_ovr_auc"] - max(endpoint_values),
        f"{endpoint}_minus_local_specialist_auc": by_id[endpoint]["metrics"]["macro_ovr_auc"] - by_id[local_id]["metrics"]["macro_ovr_auc"],
        f"{endpoint}_minus_M0paired_auc": by_id[endpoint]["metrics"]["macro_ovr_auc"] - m0["macro_ovr_auc"],
        f"M1_minus_{endpoint}_auc": by_id["M1"]["metrics"]["macro_ovr_auc"] - by_id[endpoint]["metrics"]["macro_ovr_auc"],
        "M1_minus_best_endpoint_specialist_auc": by_id["M1"]["metrics"]["macro_ovr_auc"] - max(endpoint_values),
        "M1_minus_M0paired_auc": by_id["M1"]["metrics"]["macro_ovr_auc"] - m0["macro_ovr_auc"],
    }
    if endpoint == "D000E":
        comparisons = {
            "D000E_minus_best_D000_auc": by_id[endpoint]["metrics"]["macro_ovr_auc"] - max(endpoint_values),
            "D000E_minus_local_D000_auc": by_id[endpoint]["metrics"]["macro_ovr_auc"] - by_id[local_id]["metrics"]["macro_ovr_auc"],
            "D000E_minus_M0paired_auc": by_id[endpoint]["metrics"]["macro_ovr_auc"] - m0["macro_ovr_auc"],
            "M1_minus_D000E_auc": by_id["M1"]["metrics"]["macro_ovr_auc"] - by_id[endpoint]["metrics"]["macro_ovr_auc"],
            "M1_minus_best_D000_auc": by_id["M1"]["metrics"]["macro_ovr_auc"] - max(endpoint_values),
            "M1_minus_M0paired_auc": by_id["M1"]["metrics"]["macro_ovr_auc"] - m0["macro_ovr_auc"],
        }
    measured_gpu_hours = sum(float(row.get("runtime", {}).get("measured_gpu_hours", 0)) for row in rows)
    measured_gpu_hours += sum(float(load_json(root / "reports" / f"{name}_stage.json")["runtime"]["target_build_seconds"]) / 3600 for name in components)
    payload = {
        "contract": aggregate_contract(profile), "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"], "rows": rows,
        "comparisons": comparisons, "fresh_fit_count": len(registry),
        "ensemble_count": len(components), "final_test_accessed": False,
        "contextual_reports": list(spec["contextual_reports"]),
        "measured_gpu_hours": measured_gpu_hours,
    }
    return with_content_hash(payload)


class MhpeWorkflow:
    def __init__(self, spec: Mapping[str, Any], *, recovery_spec_sha256: str | None = None) -> None:
        validate_campaign(spec, executable=False, verify_source_tree=recovery_spec_sha256 is None); self.spec = spec; self.root = Path(spec["campaign_root"]); self.recovery_spec_sha256 = recovery_spec_sha256

    def run(self, task_id: str, *, device: str = "cuda") -> dict[str, Any]:
        profile = campaign_profile(self.spec)
        task = next(
            (row for row in campaign_tasks(profile) if row["task_id"] == task_id),
            None,
        )
        if task is None:
            raise ValueError("unknown HCWDL-MHPE task")
        if task["kind"] == "train":
            started = time.monotonic()
            result = run_specialist(spec=self.spec, node_id=task["node_id"], device=device, recovery_spec_sha256=self.recovery_spec_sha256)
            elapsed = time.monotonic() - started
            runtime = with_content_hash({
                "contract": "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_RUNTIME/v1",
                "schema_version": 1, "node_id": task["node_id"],
                "elapsed_seconds": elapsed,
                "measured_gpu_hours": elapsed / 3600 if device.startswith("cuda") else 0.0,
                "cache_array_bytes": result.pop("_runtime_cache_array_bytes"),
                "final_test_accessed": False,
            })
            write_immutable_json(self.root / "reports/runtime" / f"{task['node_id']}.json", runtime)
            return result
        if task["kind"] == "ensemble":
            return run_ensemble(spec=self.spec, ensemble_id=task["ensemble_id"], device=device, recovery_spec_sha256=self.recovery_spec_sha256)
        if task["kind"] == "aggregate":
            output = build_aggregate(self.spec); write_immutable_json(self.root / "reports/validation_aggregate.json", output); return output
        if task["kind"] == "finalist_lock":
            aggregate = load_json(self.root / "reports/validation_aggregate.json")
            validate_content_hash(aggregate, expected_contract=aggregate_contract(profile), expected_schema_version=1)
            by_id = {row["node_id"]: row for row in aggregate["rows"]}
            entries = []
            for node_id in finalists(profile):
                row = by_id[node_id]
                entries.append({
                    "node_id": node_id,
                    "report_sha256": row["report_sha256"],
                    "checkpoint_sha256": row.get("selected_checkpoint_sha256", row["report_sha256"]),
                })
            output = finalist_lock_payload(aggregate_sha256=aggregate["content_hash"], entries=entries, profile=profile)
            write_immutable_json(self.root / "locks/finalist_lock.json", output); return output
        lock = load_json(self.root / "locks/finalist_lock.json")
        validate_content_hash(lock, expected_contract=finalist_lock_contract(profile), expected_schema_version=1)
        completion = {
            "contract": completion_contract(profile), "schema_version": 1,
            "campaign_spec_sha256": self.spec["content_hash"],
            "finalist_lock_sha256": lock["content_hash"], "fresh_fit_count": len(node_registry(profile)),
            "scientific_result_does_not_control_completion": True,
            "final_test_accessed": False,
        }
        output = with_content_hash(completion)
        write_immutable_json(self.root / "reports/campaign_complete.json", output); return output


__all__ = ["MhpeWorkflow", "build_aggregate"]

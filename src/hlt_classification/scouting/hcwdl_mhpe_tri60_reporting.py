"""Validation aggregation and HLT-only deployment extraction for TRI60."""

from __future__ import annotations

from io import BytesIO
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, load_json, sha256_file, write_immutable_json,
)
from hlt_classification.models.scouting_particle_transformer import (
    build_scouting_particle_transformer,
)

from .engine import validate_pmard_training_report
from .hcwdl_mhpe_tri60_campaign import validate_campaign
from .hcwdl_mhpe_tri60_contracts import (
    AGGREGATE_CONTRACT, CAMPAIGN_COMPLETE_CONTRACT,
    DEPLOYABLE_CHECKPOINT_CONTRACT, FINALIST_LOCK_CONTRACT,
    STAGE_REPORT_CONTRACT, TRAINING_REPORT_CONTRACT,
    artifact, validate_artifact,
)
from .hcwdl_mhpe_tri60_graph import (
    ENSEMBLE_COMPONENTS, FIT_ORDER, NODE_REGISTRY, REDUCER_ORDER,
)
from .hcwdl_mhpe_tri60_probability import load_probability_role


def _training_report(spec: Mapping[str, Any], node_id: str) -> dict[str, Any]:
    path = Path(spec["campaign_root"]) / "training" / node_id / "training_report.json"
    report = load_json(path)
    validate_artifact(report, contract=TRAINING_REPORT_CONTRACT)
    if (
        report.get("node_id") != node_id
        or report.get("campaign_spec_sha256") != spec["content_hash"]
        or report.get("passes") != 60
        or report.get("validations") != 60
        or report.get("complete") is not True
        or report.get("final_test_accessed") is not False
    ):
        raise ValueError(f"TRI60 training report differs: {node_id}")
    for name, hash_name in (
        ("selected_checkpoint", "selected_checkpoint_sha256"),
        ("final_checkpoint", "final_checkpoint_sha256"),
    ):
        checkpoint = path.parent / str(report[name])
        if not checkpoint.is_file() or sha256_file(checkpoint) != report[hash_name]:
            raise ValueError(f"TRI60 checkpoint differs: {node_id}/{name}")
    return report


def _fraction(value: float, baseline: float, oracle: float) -> float | None:
    denominator = oracle - baseline
    return None if denominator == 0 else (value - baseline) / denominator


def build_aggregate(spec: Mapping[str, Any]) -> dict[str, Any]:
    validate_campaign(spec, executable=False)
    root = Path(spec["campaign_root"])
    reports = {node_id: _training_report(spec, node_id) for node_id in FIT_ORDER}
    stages = {}
    for distribution_id in ("U000", *REDUCER_ORDER):
        path = root / "reports/stages" / f"{distribution_id}.json"
        stage = load_json(path)
        validate_artifact(stage, contract=STAGE_REPORT_CONTRACT)
        if stage.get("distribution_id") != distribution_id:
            raise ValueError("TRI60 stage report identity differs")
        stages[distribution_id] = stage
    foundation = load_json(spec["artifact_paths"]["tri60_foundation_lock"])
    m0 = load_json(foundation["contextual_m0paired_report_path"])
    validate_pmard_training_report(m0)
    m0_metrics = m0["validation"]
    u000_metrics = reports["U000"]["validation"]
    rows = []
    for node_id in FIT_ORDER:
        report = reports[node_id]
        metrics = report["validation"]
        preparation_registry = dict(report.get("preparation_seconds", {}))
        preparation_seconds = float(
            preparation_registry.get(
                "pre_training_total_seconds",
                sum(
                    float(value)
                    for name, value in preparation_registry.items()
                    if name != "pre_training_total_seconds"
                ),
            )
        )
        total_runtime = float(report["runtime_seconds"]) + preparation_seconds
        rows.append({
            "row_id": node_id, "kind": "fit", "track": NODE_REGISTRY[node_id].track,
            "coordinate": NODE_REGISTRY[node_id].coordinate_name,
            "metrics": metrics, "selected_pass": report["selected_pass"],
            "selected_update": report["selected_update"],
            "teacher_id": NODE_REGISTRY[node_id].distribution_teacher_id,
            "representation_carrier_id": NODE_REGISTRY[node_id].representation_carrier_id,
            "recovery_m0paired_to_u000": {
                name: _fraction(
                    float(metrics[name]), float(m0_metrics[name]),
                    float(u000_metrics[name]),
                )
                for name in (
                    "macro_ovr_auc", "accuracy",
                    "macro_mean_log_qcd_rejection_at_50pct_signal",
                )
            },
            "runtime_seconds": float(report["runtime_seconds"]),
            "preparation_seconds": preparation_registry,
            "total_worker_seconds": total_runtime,
            "gpu_hours": total_runtime / 3600.0,
            "peak_rss_bytes": int(report["peak_rss_bytes"]),
            "peak_cuda_bytes": int(report["peak_cuda_bytes"]),
        })
    for distribution_id in ("U000", *REDUCER_ORDER):
        stage = stages[distribution_id]
        metrics = stage["ensemble_metrics"]
        rows.append({
            "row_id": distribution_id, "kind": "probability_ensemble",
            "track": distribution_id.split("_", 1)[0] if "_" in distribution_id else "ROOT",
            "coordinate": distribution_id,
            "metrics": metrics,
            "component_order": stage["component_order"],
            "ensemble_minus_mean_component_auc": stage["ensemble_minus_mean_component_auc"],
            "ensemble_minus_best_component_auc": stage["ensemble_minus_best_component_auc"],
            "leave_one_out": stage["leave_one_out"],
            "diversity": stage["diversity"],
            "runtime_seconds": float(stage["runtime_seconds"]),
            "gpu_hours": float(stage["runtime_seconds"]) / 3600.0,
            "train_probability_bytes": int(stage["train_probability_bytes"]),
            "validation_probability_bytes": int(
                stage["validation_probability_bytes"]
            ),
        })
    metric_by_id = {row["row_id"]: row["metrics"] for row in rows}
    comparisons = {}
    for track in ("LOGIT", "RSET", "RREL"):
        endpoint = f"{track}_D000E"
        m1 = f"M1_{track}"
        components = ENSEMBLE_COMPONENTS[endpoint]
        comparisons[f"{endpoint}_minus_best_specialist_auc"] = (
            float(metric_by_id[endpoint]["macro_ovr_auc"])
            - max(float(metric_by_id[name]["macro_ovr_auc"]) for name in components)
        )
        comparisons[f"{m1}_minus_{endpoint}_auc"] = (
            float(metric_by_id[m1]["macro_ovr_auc"])
            - float(metric_by_id[endpoint]["macro_ovr_auc"])
        )
    comparisons.update({
        "M1E_minus_best_M1_auc": (
            float(metric_by_id["M1E"]["macro_ovr_auc"])
            - max(float(metric_by_id[name]["macro_ovr_auc"])
                  for name in ("M1_LOGIT", "M1_RSET", "M1_RREL"))
        ),
        "M2_minus_M1E_auc": (
            float(metric_by_id["M2"]["macro_ovr_auc"])
            - float(metric_by_id["M1E"]["macro_ovr_auc"])
        ),
        "M2_minus_contextual_M0paired_auc": (
            float(metric_by_id["M2"]["macro_ovr_auc"])
            - float(m0_metrics["macro_ovr_auc"])
        ),
    })
    probability_bytes = 0
    for distribution_id in ("U000", *REDUCER_ORDER):
        directory = root / "probabilities" / distribution_id
        for role in ("train", "validation"):
            manifest, _, _ = load_probability_role(
                directory / f"{role}_manifest.json",
                expected_distribution_id=distribution_id, expected_role=role,
            )
            shard = load_json(manifest["shards"][0]["path"])
            probability_bytes += Path(shard["data_path"]).stat().st_size
    durable_bytes = sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    return artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "graph": spec["parents"]["graph"],
            "recipe": spec["parents"]["recipe"],
            "foundation": spec["parents"]["foundation"],
        },
        "primary_metric": "validation_macro_ovr_auc",
        "rows": rows,
        "contextual_m0paired": {
            "metrics": m0_metrics,
            "pass_count": foundation["contextual_m0paired_pass_count"],
            "pass_matched_60_control": False,
        },
        "comparisons": comparisons,
        "fit_count": len(reports), "reducer_count": len(REDUCER_ORDER),
        "total_measured_gpu_hours": sum(
            float(row.get("gpu_hours", 0.0)) for row in rows
        ),
        "probability_bank_bytes": probability_bytes,
        "durable_campaign_bytes_at_aggregate": durable_bytes,
        "representation_target_durable_bytes": 0,
        "rolling_resume_durable_bytes": 0,
        "scientific_result_does_not_control_completion": True,
        "one_seed_exploratory": True,
        "final_test_accessed": False,
    }, contract=AGGREGATE_CONTRACT)


def publish_m2_deployable(spec: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    report = _training_report(spec, "M2")
    selected = Path(spec["campaign_root"]) / "training/M2" / report["selected_checkpoint"]
    payload = torch.load(selected, map_location="cpu", weights_only=False)
    state = payload.get("model")
    if not isinstance(state, Mapping) or any(name.startswith("representation_heads") for name in state):
        raise ValueError("TRI60 M2 selected state is not an ordinary HLT model")
    model = build_scouting_particle_transformer()
    loaded = model.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError("TRI60 M2 deployable state does not strict-load")
    deployable = {
        "contract": DEPLOYABLE_CHECKPOINT_CONTRACT,
        "schema_version": 1,
        "architecture": "ScoutingParticleTransformer/21-channel/200-token",
        "model": state,
        "source_selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
        "hlt_only": True,
        "training_only_projection_heads_present": False,
        "offline_inputs_present": False,
    }
    stream = BytesIO(); torch.save(deployable, stream)
    path = Path(spec["campaign_root"]) / "deployment/M2.pt"
    atomic_publish_bytes(path, stream.getvalue())
    sidecar = artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "m2_report": report["content_hash"],
            "m2_selected_checkpoint": report["selected_checkpoint_sha256"],
        },
        "checkpoint_path": str(path.resolve()),
        "checkpoint_sha256": sha256_file(path),
        "architecture": deployable["architecture"],
        "hlt_only": True, "input_channels": 21, "token_limit": 200,
        "training_only_projection_heads_present": False,
        "offline_inputs_present": False,
        "final_test_accessed": False,
    }, contract=DEPLOYABLE_CHECKPOINT_CONTRACT)
    write_immutable_json(Path(spec["campaign_root"]) / "deployment/M2.json", sidecar)
    return sidecar


__all__ = ["build_aggregate", "publish_m2_deployable"]

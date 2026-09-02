"""Task dispatch, aggregation, and completion for output handoff."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import load_json, sha256_file, write_immutable_json

from .hcwdl_adjacent_output_handoff_campaign import validate_campaign
from .hcwdl_adjacent_output_handoff_contracts import (
    AGGREGATE_CONTRACT, COMPLETE_CONTRACT, ENSEMBLE_REPORT_CONTRACT,
    SELECTED_MIXTURE_CONTRACT, STAGE_REPORT_CONTRACT,
    TRAINING_REPORT_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_adjacent_output_handoff_execution import run_execution_acceptance
from .hcwdl_adjacent_output_handoff_graph import (
    ENSEMBLE_IDS, FINAL_NODES, FIT_ORDER, GRAPH_SHA256, NODE_REGISTRY,
    SELECTION_IDS, node_distribution,
)
from .hcwdl_adjacent_output_handoff_probability import ROLES
from .hcwdl_adjacent_output_handoff_runner import (
    probability_dir, run_control_reducer, run_ensemble, run_fit,
    run_model_reducer, run_partition,
    run_selection, training_dir,
)
from .hcwdl_adjacent_output_handoff_source import (
    validate_control_lock, validate_source_lock,
)


def _probability_outputs(spec: Mapping[str, Any], distribution_id: str) -> list[Path]:
    root = probability_dir(spec, distribution_id)
    result = [root / "lock.json"]
    for role in ROLES:
        result.extend((root / f"{role}.npz", root / f"{role}_shard.json", root / f"{role}_manifest.json"))
    return result


def task_outputs(spec: Mapping[str, Any], task_id: str) -> list[Path]:
    task = {row["task_id"]: row for row in spec["tasks"]}[task_id]
    root = Path(spec["campaign_root"]); kind = task["kind"]
    if kind == "authenticate": return [root / "reports/stages/authenticate.json"]
    if kind == "partition": return [Path(spec["artifact_paths"]["validation_partition"]), Path(spec["artifact_paths"]["validation_partition"]).with_suffix(".npz")]
    if kind == "audit": return [root / "reports/stages/audit_sources_and_storage.json"]
    if kind == "preflight": return [Path(spec["artifact_paths"]["execution_acceptance"])]
    if kind == "train":
        report = load_json(training_dir(spec, task["node_id"]) / "training_report.json")
        return [
            training_dir(spec, task["node_id"]) / "training_report.json",
            training_dir(spec, task["node_id"]) / report["selected_checkpoint"],
            training_dir(spec, task["node_id"]) / report["final_checkpoint"],
        ]
    if kind == "model_reducer":
        distribution = "SOURCE_U100" if task["node_id"] == "SOURCE_U100" else node_distribution(task["node_id"])
        return _probability_outputs(spec, distribution) + [root / "reports/stages" / f"{distribution}.json"]
    if kind == "control_reducer":
        return [root / "reports/stages" / f"CONTROL_{task['control_id']}.json"]
    if kind == "selection":
        selection = task["selection_id"]
        return _probability_outputs(spec, selection) + [
            root / "reports/mixtures" / selection / "curve.json",
            root / "reports/mixtures" / selection / "selected.json",
            root / "reports/mixtures" / selection / "temperature_rich.json",
            root / "reports/mixtures" / selection / "temperature_poor.json",
            root / "reports/mixtures" / selection / "bootstrap.json",
            root / "reports/stages" / f"{selection}.json",
        ]
    if kind == "ensemble":
        ensemble = task["ensemble_id"]
        return _probability_outputs(spec, ensemble) + [root / "reports/ensembles" / f"{ensemble}.json"]
    if kind == "aggregate": return [root / "reports/validation_aggregate.json"]
    if kind == "complete": return [root / "reports/campaign_complete.json"]
    raise KeyError("unknown output-handoff task kind")


def _stage(spec: Mapping[str, Any], name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    value = artifact({
        "parents": {"campaign_spec": spec["content_hash"], "graph": GRAPH_SHA256},
        "stage": name, **dict(payload), "final_test_accessed": False,
    }, contract=STAGE_REPORT_CONTRACT)
    write_immutable_json(Path(spec["campaign_root"]) / "reports/stages" / f"{name}.json", value)
    return value


def build_aggregate(spec: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(spec["campaign_root"]); rows = []
    parents = {
        "campaign_spec": spec["content_hash"], "graph": GRAPH_SHA256,
        "recipe": spec["parents"]["recipe"], "controls": spec["parents"]["controls"],
    }
    source_stage = load_json(root / "reports/stages/SOURCE_U100.json")
    validate_artifact(source_stage, contract=STAGE_REPORT_CONTRACT)
    parents["stage/SOURCE_U100"] = source_stage["content_hash"]
    rows.append({"model_id": "SOURCE_U100", "kind": "source_anchor", "metrics": source_stage["validation_metrics"]})
    for node_id in FIT_ORDER:
        stage = load_json(root / "reports/stages" / f"{node_distribution(node_id)}.json")
        validate_artifact(stage, contract=STAGE_REPORT_CONTRACT)
        parents[f"stage/{node_id}"] = stage["content_hash"]
        rows.append({
            "model_id": node_id, "kind": NODE_REGISTRY[node_id].role,
            "coordinate": NODE_REGISTRY[node_id].coordinate_name,
            "metrics": stage["validation_metrics"],
        })
    ensembles = []
    for ensemble_id in ENSEMBLE_IDS:
        stage = load_json(root / "reports/ensembles" / f"{ensemble_id}.json")
        validate_artifact(stage, contract=ENSEMBLE_REPORT_CONTRACT)
        parents[f"ensemble/{ensemble_id}"] = stage["content_hash"]
        ensembles.append({"ensemble_id": ensemble_id, "metrics": stage["validation_metrics"]})
    mixtures = []
    for selection_id in SELECTION_IDS:
        selected = load_json(root / "reports/mixtures" / selection_id / "selected.json")
        validate_artifact(selected, contract=SELECTED_MIXTURE_CONTRACT)
        stage = load_json(root / "reports/stages" / f"{selection_id}.json")
        validate_artifact(stage, contract=STAGE_REPORT_CONTRACT)
        parents[f"mixture/{selection_id}"] = selected["content_hash"]
        parents[f"mixture_stage/{selection_id}"] = stage["content_hash"]
        mixtures.append({
            "selection_id": selection_id, "family": selected["selected_family"],
            "alpha": selected["selected_alpha_numerator"] / selected["selected_alpha_denominator"],
            "selection_candidate_on_V_blend": selected["selected_candidate"],
            "metrics": stage["validation_metrics"], "report_role": "V_report",
        })
    control_rows = []
    for name in ("M0CE60", "U000"):
        stage = load_json(root / "reports/stages" / f"CONTROL_{name}.json")
        validate_artifact(stage, contract=STAGE_REPORT_CONTRACT)
        parents[f"control_stage/{name}"] = stage["content_hash"]
        control_rows.append({
            "model_id": name, "kind": "reporting_control",
            "metrics": stage["validation_metrics"], "report_role": "V_report",
        })
    return artifact({
        "parents": parents, "report_role": "V_report",
        "controls_are_contextual_full_validation": False,
        "all_rows_share_exact_V_report_identities": True,
        "control_rows": control_rows, "model_rows": rows,
        "mixture_rows": mixtures, "ensemble_rows": ensembles,
        "primary_comparison": "FINAL_HANDOFF_D000-minus-FINAL_DIRECT_D000",
        "secondary_comparison": "FINAL_HANDOFF_D000-minus-FINAL_CE_SEED_D000",
        "poor_metrics_do_not_control_completion": True,
        "final_test_accessed": False,
    }, contract=AGGREGATE_CONTRACT)


class OutputHandoffWorkflow:
    def __init__(
        self, spec: Mapping[str, Any], *, recovery_spec_sha256: str | None = None,
        execution_source_commit: str | None = None,
    ):
        validate_campaign(spec); self.spec = spec
        self.recovery_spec_sha256 = recovery_spec_sha256
        self.execution_source_commit = execution_source_commit

    def run(self, task_id: str, *, device: str = "cuda") -> dict[str, Any]:
        tasks = {row["task_id"]: row for row in self.spec["tasks"]}
        if task_id not in tasks: raise KeyError("unknown output-handoff task")
        task = tasks[task_id]; kind = task["kind"]
        if kind == "authenticate":
            source = load_json(self.spec["artifact_paths"]["source_lock"])
            controls = load_json(self.spec["artifact_paths"]["control_lock"])
            return _stage(self.spec, "authenticate", {
                "source_lock_sha256": validate_source_lock(source),
                "control_lock_sha256": validate_control_lock(controls),
                "passed": True,
            })
        if kind == "partition": return run_partition(self.spec)
        if kind == "audit":
            free = shutil.disk_usage(self.spec["campaign_root"]).free
            if free < int(self.spec["minimum_free_disk_bytes"]):
                raise OSError("output-handoff storage floor is unavailable")
            return _stage(self.spec, "audit_sources_and_storage", {
                "free_disk_bytes": free,
                "minimum_free_disk_bytes": self.spec["minimum_free_disk_bytes"],
                "projected_durable_bytes": self.spec["projected_durable_bytes"],
                "durable_particle_views": False, "durable_hidden_states": False,
                "rolling_resume": False, "passed": True,
            })
        if kind == "preflight":
            value = run_execution_acceptance(
                spec=self.spec,
                source_commit=self.execution_source_commit or self.spec["source_commit"],
                device=device, require_production=True,
            )
            write_immutable_json(self.spec["artifact_paths"]["execution_acceptance"], value); return value
        if kind == "train":
            return run_fit(
                self.spec, task["node_id"], device=device,
                recovery_spec_sha256=self.recovery_spec_sha256,
                execution_source_commit=self.execution_source_commit,
            )
        if kind == "model_reducer":
            return run_model_reducer(
                self.spec, task["node_id"], device=device,
                execution_source_commit=self.execution_source_commit,
            )
        if kind == "control_reducer":
            return run_control_reducer(
                self.spec, task["control_id"], device=device,
            )
        if kind == "selection":
            return run_selection(
                self.spec, task["selection_id"],
                execution_source_commit=self.execution_source_commit,
            )
        if kind == "ensemble":
            return run_ensemble(
                self.spec, task["ensemble_id"],
                execution_source_commit=self.execution_source_commit,
            )
        if kind == "aggregate":
            value = build_aggregate(self.spec)
            write_immutable_json(Path(self.spec["campaign_root"]) / "reports/validation_aggregate.json", value)
            return value
        if kind == "complete":
            aggregate = load_json(Path(self.spec["campaign_root"]) / "reports/validation_aggregate.json")
            aggregate_hash = validate_artifact(aggregate, contract=AGGREGATE_CONTRACT)
            value = artifact({
                "parents": {"campaign_spec": self.spec["content_hash"], "aggregate": aggregate_hash},
                "fresh_fit_count": 26, "scientific_result_does_not_control_completion": True,
                "final_test_accessed": False,
            }, contract=COMPLETE_CONTRACT)
            write_immutable_json(Path(self.spec["campaign_root"]) / "reports/campaign_complete.json", value)
            return value
        raise KeyError("unknown output-handoff task kind")


__all__ = ["OutputHandoffWorkflow", "build_aggregate", "task_outputs"]

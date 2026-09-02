"""Task dispatch and immutable reporting for learned fusion handoff."""

from __future__ import annotations

from pathlib import Path
import math
import shutil
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, load_json, load_npz_arrays, sha256_file,
    write_immutable_json,
)
from .hcwdl_adjacent_learned_handoff_campaign import validate_campaign
from .hcwdl_adjacent_learned_handoff_contracts import (
    AGGREGATE_CONTRACT, COMPLETE_CONTRACT, DIAGNOSTIC_REPORT_CONTRACT,
    POPULATION_LOCK_CONTRACT, SEED_LOCK_CONTRACT, STAGE_REPORT_CONTRACT,
    TRAINING_REPORT_CONTRACT, artifact,
    validate_artifact,
)
from .hcwdl_adjacent_learned_handoff_graph import (
    FIT_ORDER, GRAPH_SHA256, NODE_REGISTRY, RUNG_ORDER,
    acquisition_distribution,
    distribution_consumers,
)
from .hcwdl_adjacent_learned_handoff_data import morph_context_for_pass
from .hcwdl_adjacent_learned_handoff_probability import (
    REPORT_ONLY_ROLES, ROLES, load_role,
)
from .hcwdl_adjacent_output_handoff_fusion import (
    paired_stratified_macro_auc_bootstrap,
)
from .hcwdl_adjacent_learned_handoff_partition import load_partition
from .hcwdl_adjacent_learned_handoff_runner import (
    build_capacity_audit, probability_dir, run_control_reducer,
    run_execution_acceptance, run_extract, run_extracted_reducer, run_fit,
    run_model_reducer, run_partition, run_source_reducer, training_dir,
    validate_execution_acceptance,
)
from .hcwdl_adjacent_learned_handoff_source import (
    validate_control_lock, validate_source_lock,
)
from .hcwdl_recovery import (
    task_attestation_path, validate_task_attestation,
)


SCIENCE_GATE_TASKS = (
    "authenticate", "partition_validation", "audit_sources_and_storage",
    "preflight",
)


def _probability_outputs(spec, distribution_id):
    root=probability_dir(spec,distribution_id); result=[root/"lock.json"]
    roles = ROLES if distribution_consumers(distribution_id) else REPORT_ONLY_ROLES
    for role in roles: result.extend((root/f"{role}.npz",root/f"{role}_shard.json",root/f"{role}_manifest.json"))
    return result


def _node_distribution(node_id: str) -> str:
    node=NODE_REGISTRY[node_id]
    if node.role=="fusion_acquisition": return acquisition_distribution(node.primary_coordinate)
    if node_id=="DIRECT_VIEW_MORPH_U100_TO_D000": return "MORPH_Q_D000"
    return node_id


def task_outputs(spec: Mapping[str,Any],task_id: str):
    task={row["task_id"]:row for row in spec["tasks"]}[task_id]; root=Path(spec["campaign_root"]); kind=task["kind"]
    if kind=="authenticate": return [root/"reports/stages/authenticate.json"]
    if kind=="partition": return [Path(spec["artifact_paths"]["validation_partition"]),Path(spec["artifact_paths"]["validation_partition"]).with_suffix(".npz")]
    if kind=="audit": return [Path(spec["artifact_paths"]["capacity_audit"]),root/"reports/stages/audit_sources_and_storage.json"]
    if kind=="preflight": return [Path(spec["artifact_paths"]["execution_acceptance"])]
    if kind=="train":
        report=load_json(training_dir(spec,task["node_id"])/"training_report.json")
        return [training_dir(spec,task["node_id"])/"training_report.json",training_dir(spec,task["node_id"])/report["selected_checkpoint"],training_dir(spec,task["node_id"])/report["final_checkpoint"]]
    if kind=="extract":
        base=root/"deployable"/task["distribution_id"]
        return [base/"selected_model.pt",base/"extraction.json",root/"reports/diagnostics"/f"{task['node_id']}.json"]
    if kind in {"source_reducer","model_reducer","extracted_reducer"}:
        distribution=(task.get("distribution_id") or _node_distribution(task["node_id"]))
        result=_probability_outputs(spec,distribution)+[root/"reports/stages"/f"{distribution}.json"]
        if kind=="model_reducer" and NODE_REGISTRY[task["node_id"]].input_protocol!="standard_hlt_v1":
            result.extend((
                root/"reports/diagnostics"/f"{task['node_id']}.json",
                root/"reports/diagnostics"/f"{task['node_id']}_alpha_zero_V_report.npz",
            ))
        return result
    if kind=="control_reducer": return [root/"reports/stages"/f"CONTROL_{task['control_id']}.json"]
    if kind=="aggregate": return [root/"reports/validation_aggregate.json"]
    if kind=="complete": return [root/"reports/campaign_complete.json"]
    raise KeyError("unknown learned-handoff task kind")


def validate_science_gate(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate the complete four-task gate before science submission."""

    validate_campaign(spec, executable=True)
    acceptance = load_json(spec["artifact_paths"]["execution_acceptance"])
    acceptance_hash = validate_execution_acceptance(spec, acceptance)
    attestations = {}
    for task_id in SCIENCE_GATE_TASKS:
        outputs = task_outputs(spec, task_id)
        if not outputs or not all(path.is_file() for path in outputs):
            raise FileNotFoundError(
                f"learned-handoff gate output is absent for {task_id}"
            )
        attestation = load_json(
            task_attestation_path(spec["campaign_root"], task_id, None)
        )
        attestations[task_id] = validate_task_attestation(
            attestation, campaign_spec_sha256=spec["content_hash"],
            task_id=task_id, array_index=None,
        )
    return {
        "execution_acceptance": acceptance_hash,
        "task_attestations": attestations,
    }


def _stage(spec,name,payload):
    value=artifact({"parents":{"campaign_spec":spec["content_hash"],"graph":GRAPH_SHA256},"stage":name,**payload,"final_test_accessed":False},contract=STAGE_REPORT_CONTRACT)
    write_immutable_json(Path(spec["campaign_root"])/"reports/stages"/f"{name}.json",value); return value


def _tree_bytes(*paths: Path) -> int:
    files: set[Path] = set()
    for path in paths:
        if path.is_file():
            files.add(path.resolve())
        elif path.is_dir():
            files.update(value.resolve() for value in path.rglob("*") if value.is_file())
    return sum(value.stat().st_size for value in files)


def _training_summary(spec, node_id: str, distribution_id: str) -> tuple[dict, str]:
    root = Path(spec["campaign_root"])
    report_path = training_dir(spec, node_id) / "training_report.json"
    report = load_json(report_path)
    digest = validate_artifact(report, contract=TRAINING_REPORT_CONTRACT)
    diagnostic = root / "reports/diagnostics" / f"{node_id}.json"
    deployable = root / "deployable" / distribution_id
    durable_bytes = _tree_bytes(
        training_dir(spec, node_id), probability_dir(spec, distribution_id),
        diagnostic, diagnostic.with_name(f"{node_id}_alpha_zero_V_report.npz"),
        deployable,
    )
    summary = {
        "training_report_path": str(report_path.resolve()),
        "training_report_sha256": digest,
        "selected_pass": int(report["selected_pass"]),
        "selected_update": int(report["selected_update"]),
        "checkpoint_selection_minimum_pass": int(
            report.get("checkpoint_selection_minimum_pass", 1)
        ),
        "completed_passes": int(report["validations"]),
        "validation_history": report["validation_history"],
        "training_history": report["training_history"],
        "runtime_seconds": float(report["runtime_seconds"]),
        "preparation_seconds": report["preparation_seconds"],
        "peak_cpu_rss_bytes": int(report["peak_rss_bytes"]),
        "peak_gpu_memory_bytes": int(report["peak_cuda_bytes"]),
        "parameter_scalar_count": int(report["parameter_scalar_count"]),
        "trainable_parameter_scalar_count": int(
            report["trainable_parameter_scalar_count"]
        ),
        "durable_output_bytes": durable_bytes,
        "rolling_resume_published": report["rolling_resume_published"],
        "partial_checkpoint_reuse": report["partial_checkpoint_reuse"],
    }
    if node_id == "DIRECT_VIEW_MORPH_U100_TO_D000":
        summary["context_coordinate_by_pass"] = [
            {"pass": pass_number,
             "context_coordinate": morph_context_for_pass(pass_number)[0]}
            for pass_number in range(1, int(report["validations"]) + 1)
        ]
        summary["registered_context_coordinate_by_pass"] = [
            {"pass": pass_number,
             "context_coordinate": morph_context_for_pass(pass_number)[0]}
            for pass_number in range(1, 101)
        ]
    return summary, digest


def _r50(metrics: Mapping[str, Any]) -> float:
    return math.exp(float(metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]))


def _recoveries(metrics, baseline, oracle):
    def percent(value, low, high):
        return 100.0 * (value - low) / (high - low) if high != low else None

    result = {
        "auc_pct": percent(
            float(metrics["macro_ovr_auc"]), float(baseline["macro_ovr_auc"]),
            float(oracle["macro_ovr_auc"]),
        ),
        "macro_r50_linear_pct": percent(
            _r50(metrics), _r50(baseline), _r50(oracle),
        ),
        "per_class_r50_linear_pct": {},
    }
    for class_name in sorted(
        set(metrics.get("per_class", {}))
        & set(baseline.get("per_class", {}))
        & set(oracle.get("per_class", {}))
    ):
        def class_r50(value):
            return float(
                value["per_class"][class_name]["qcd_rejection"]["50pct"][
                    "rejection"
                ]
            )
        result["per_class_r50_linear_pct"][class_name] = percent(
            class_r50(metrics), class_r50(baseline), class_r50(oracle),
        )
    return result


def _ordered_v_report(spec, distribution_id, expected_ids):
    manifest, identities, probabilities = load_role(
        probability_dir(spec, distribution_id) / "V_report_manifest.json",
        distribution_id=distribution_id, role="V_report",
    )
    lookup = {bytes(value): index for index, value in enumerate(identities)}
    try:
        order = np.asarray([lookup[bytes(value)] for value in expected_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError(
            f"learned-handoff V_report coverage differs for {distribution_id}"
        ) from error
    if len(lookup) != len(expected_ids):
        raise ValueError(
            f"learned-handoff V_report row count differs for {distribution_id}"
        )
    return manifest, np.ascontiguousarray(probabilities[order], dtype=np.float32)


def _morph_alpha_zero_v_report(spec, diagnostic, expected_ids, expected_labels):
    path = Path(diagnostic["alpha_zero_V_report_path"])
    if not path.is_file() or sha256_file(path) != diagnostic["alpha_zero_V_report_sha256"]:
        raise ValueError("learned-handoff morph alpha-zero probability bytes differ")
    arrays = load_npz_arrays(path)
    if set(arrays) != {"identity_digest", "probabilities", "label"}:
        raise ValueError("learned-handoff morph alpha-zero arrays differ")
    if {
        name: array_sha256(name, value) for name, value in arrays.items()
    } != diagnostic["alpha_zero_V_report_array_sha256"]:
        raise ValueError("learned-handoff morph alpha-zero array hashes differ")
    lookup = {bytes(value): index for index, value in enumerate(arrays["identity_digest"])}
    if len(lookup) != len(expected_ids):
        raise ValueError("learned-handoff morph alpha-zero identity count differs")
    try:
        order = np.asarray([lookup[bytes(value)] for value in expected_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError("learned-handoff morph alpha-zero coverage differs") from error
    labels = np.asarray(arrays["label"], dtype=np.int16)[order]
    if not np.array_equal(labels, expected_labels):
        raise ValueError("learned-handoff morph alpha-zero labels differ")
    return np.ascontiguousarray(arrays["probabilities"][order], dtype=np.float32)


def build_aggregate(spec):
    root=Path(spec["campaign_root"]); parents={"campaign_spec":spec["content_hash"],"graph":GRAPH_SHA256,"recipe":spec["parents"]["recipe"]}; rows=[]
    source=load_json(root/"reports/stages/SOURCE_U100.json"); validate_artifact(source,contract=STAGE_REPORT_CONTRACT); parents["stage/SOURCE_U100"]=source["content_hash"]
    rows.append({"model_id":"SOURCE_U100","kind":"source_anchor","metrics":source["validation_metrics"]})
    for node_id in FIT_ORDER:
        node=NODE_REGISTRY[node_id]
        if node.role in {"fusion_withdrawal","morph_withdrawal"}: distribution=(f"LEARNED_T_{node.primary_coordinate}" if node.role=="fusion_withdrawal" else "MORPH_T_D000")
        else: distribution=_node_distribution(node_id)
        stage=load_json(root/"reports/stages"/f"{distribution}.json"); validate_artifact(stage,contract=STAGE_REPORT_CONTRACT); parents[f"stage/{node_id}"]=stage["content_hash"]
        training, training_hash = _training_summary(spec, node_id, distribution)
        parents[f"training/{node_id}"] = training_hash
        row={"model_id":node_id,"reporting_aliases":(["DIRECT_KD_D000"] if node_id=="LEARNED_DIRECT_D000" else ["CE_FUSION_D000_D000"] if node_id=="FUSION_LOW_LOW_D000" else ["PARAMETER_MATCHED_SINGLE_D000"] if node_id=="LOW_PARAMETER_MATCHED_D000" else ["VIEW_MORPH_U100_TO_D000"] if node_id=="DIRECT_VIEW_MORPH_U100_TO_D000" else []),"kind":node.role,"primary_coordinate":node.primary_coordinate,"context_coordinate":node.context_coordinate,"selection_route":node.selection_route,"metrics":stage["validation_metrics"],"training":training}
        diagnostic=root/"reports/diagnostics"/f"{node_id}.json"
        if diagnostic.is_file():
            value=load_json(diagnostic); validate_artifact(value,contract=DIAGNOSTIC_REPORT_CONTRACT); parents[f"diagnostic/{node_id}"]=value["content_hash"]; row["diagnostics"]=value
        rows.append(row)
    controls=[]
    for name in ("M0CE60","U000"):
        stage=load_json(root/"reports/stages"/f"CONTROL_{name}.json"); validate_artifact(stage,contract=STAGE_REPORT_CONTRACT); parents[f"control/{name}"]=stage["content_hash"]; controls.append({"model_id":name,"metrics":stage["validation_metrics"]})
    control_metrics = {row["model_id"]: row["metrics"] for row in controls}
    for row in rows:
        row["recovery_m0ce60_to_u000"] = _recoveries(
            row["metrics"], control_metrics["M0CE60"], control_metrics["U000"],
        )
    for row in controls:
        row["recovery_m0ce60_to_u000"] = _recoveries(
            row["metrics"], control_metrics["M0CE60"], control_metrics["U000"],
        )
    comparison_pairs=[
        ["CE_FUSION_D000_D000","CE_SINGLE_D000"],
        ["CE_FUSION_D000_D000","PARAMETER_MATCHED_SINGLE_D000"],
        ["DIRECT_VIEW_MORPH_U100_TO_D000","CE_FUSION_D000_D000"],
        ["STATIC_U100_D000","DIRECT_VIEW_MORPH_U100_TO_D000"],
        ["alpha0(DIRECT_VIEW_MORPH_U100_TO_D000)","CE_SINGLE_D000"],
        ["DIRECT_VIEW_MORPH_WITHDRAW_D000","alpha0(DIRECT_VIEW_MORPH_U100_TO_D000)"],
        ["DIRECT_VIEW_MORPH_WITHDRAW_D000","LEARNED_T_D000"],
        ["LEARNED_T_D000","DIRECT_KD_D000"],
    ]
    metrics_by_id={row["model_id"]:row["metrics"] for row in rows}
    diagnostics_by_id = {
        row["model_id"]: row["diagnostics"]
        for row in rows if "diagnostics" in row
    }
    metrics_by_id["LEARNED_T_D000"]=metrics_by_id["LEARNED_WITHDRAW_D000"]
    for coordinate in RUNG_ORDER:
        metrics_by_id[f"LEARNED_T_{coordinate}"] = metrics_by_id[
            f"LEARNED_WITHDRAW_{coordinate}"
        ]
        metrics_by_id[f"alpha0(LEARNED_ACQUIRE_{coordinate})"] = (
            diagnostics_by_id[f"LEARNED_ACQUIRE_{coordinate}"][
                "alpha_zero_metrics"
            ]
        )
    metrics_by_id["DIRECT_KD_D000"]=metrics_by_id["LEARNED_DIRECT_D000"]
    metrics_by_id["CE_FUSION_D000_D000"]=metrics_by_id["FUSION_LOW_LOW_D000"]
    metrics_by_id["PARAMETER_MATCHED_SINGLE_D000"]=metrics_by_id["LOW_PARAMETER_MATCHED_D000"]
    morph_diagnostic=next(row["diagnostics"] for row in rows if row["model_id"]=="DIRECT_VIEW_MORPH_U100_TO_D000")
    metrics_by_id["alpha0(DIRECT_VIEW_MORPH_U100_TO_D000)"]=morph_diagnostic["alpha_zero_metrics"]
    partition, partition_arrays = load_partition(
        spec["artifact_paths"]["validation_partition"],
    )
    report_selected = partition_arrays["partition"] == 2
    report_ids = np.ascontiguousarray(
        partition_arrays["identity_digest"][report_selected], dtype=np.uint8,
    )
    report_labels = np.ascontiguousarray(
        partition_arrays["label"][report_selected], dtype=np.int16,
    )
    parents["validation_partition"] = partition["content_hash"]
    probability_cache = {}
    distribution_aliases = {
        "LEARNED_T_D000": "LEARNED_T_D000",
        "DIRECT_KD_D000": "LEARNED_DIRECT_D000",
        "CE_FUSION_D000_D000": "FUSION_LOW_LOW_D000",
        "PARAMETER_MATCHED_SINGLE_D000": "LOW_PARAMETER_MATCHED_D000",
        "CE_SINGLE_D000": "CE_SINGLE_D000",
        "STATIC_U100_D000": "STATIC_U100_D000",
        "DIRECT_VIEW_MORPH_U100_TO_D000": "MORPH_Q_D000",
        "DIRECT_VIEW_MORPH_WITHDRAW_D000": "MORPH_T_D000",
    }
    morph_alpha_id = "alpha0(DIRECT_VIEW_MORPH_U100_TO_D000)"

    def probabilities(model_id):
        if model_id in probability_cache:
            return probability_cache[model_id]
        if model_id.startswith("alpha0(") and model_id.endswith(")"):
            node_id = model_id[len("alpha0("):-1]
            diagnostic = diagnostics_by_id[node_id]
            value = _morph_alpha_zero_v_report(
                spec, diagnostic, report_ids, report_labels,
            )
        else:
            distribution = distribution_aliases.get(model_id, model_id)
            manifest, value = _ordered_v_report(spec, distribution, report_ids)
            parents[f"V_report/{model_id}"] = manifest["content_hash"]
        probability_cache[model_id] = value
        return value

    comparisons=[]
    for left,right in comparison_pairs:
        a,b=metrics_by_id[left],metrics_by_id[right]
        bootstrap = paired_stratified_macro_auc_bootstrap(
            probabilities(left), probabilities(right), report_labels,
            samples=2000, seed=int(spec["paired_bootstrap_seed"]),
        )
        comparisons.append({"left":left,"right":right,"delta_accuracy":float(a["accuracy"])-float(b["accuracy"]),"delta_macro_ovr_auc":float(a["macro_ovr_auc"])-float(b["macro_ovr_auc"]),"delta_macro_r50_linear":math.exp(float(a["macro_mean_log_qcd_rejection_at_50pct_signal"]))-math.exp(float(b["macro_mean_log_qcd_rejection_at_50pct_signal"])),"paired_macro_auc_bootstrap":bootstrap})
    adjacent = []
    learned_vs_direct = []
    withdrawal_decomposition = []
    parent = "SOURCE_U100"
    for coordinate in RUNG_ORDER:
        carrier = f"LEARNED_T_{coordinate}"
        left_metrics, right_metrics = metrics_by_id[carrier], metrics_by_id[parent]
        adjacent.append({
            "left": carrier, "right": parent,
            "delta_accuracy": float(left_metrics["accuracy"]) - float(right_metrics["accuracy"]),
            "delta_macro_ovr_auc": float(left_metrics["macro_ovr_auc"]) - float(right_metrics["macro_ovr_auc"]),
            "delta_macro_r50_linear": math.exp(float(left_metrics["macro_mean_log_qcd_rejection_at_50pct_signal"])) - math.exp(float(right_metrics["macro_mean_log_qcd_rejection_at_50pct_signal"])),
            "paired_macro_auc_bootstrap": paired_stratified_macro_auc_bootstrap(
                probabilities(carrier), probabilities(parent), report_labels,
                samples=2000, seed=int(spec["paired_bootstrap_seed"]),
            ),
        })
        direct = f"LEARNED_DIRECT_{coordinate}"
        direct_metrics = metrics_by_id[direct]
        learned_vs_direct.append({
            "left": carrier, "right": direct,
            "delta_accuracy": float(left_metrics["accuracy"])
            - float(direct_metrics["accuracy"]),
            "delta_macro_ovr_auc": float(left_metrics["macro_ovr_auc"])
            - float(direct_metrics["macro_ovr_auc"]),
            "delta_macro_r50_linear": _r50(left_metrics)
            - _r50(direct_metrics),
            "paired_macro_auc_bootstrap": (
                paired_stratified_macro_auc_bootstrap(
                    probabilities(carrier), probabilities(direct),
                    report_labels, samples=2000,
                    seed=int(spec["paired_bootstrap_seed"]),
                )
            ),
        })
        acquisition = f"LEARNED_ACQUIRE_{coordinate}"
        acquisition_both = metrics_by_id[acquisition]
        acquisition_zero = metrics_by_id[f"alpha0({acquisition})"]
        auc_context_gain = float(acquisition_both["macro_ovr_auc"])
        auc_context_gain -= float(acquisition_zero["macro_ovr_auc"])
        auc_withdrawal_gain = float(left_metrics["macro_ovr_auc"])
        auc_withdrawal_gain -= float(acquisition_zero["macro_ovr_auc"])
        r50_context_gain = _r50(acquisition_both) - _r50(acquisition_zero)
        r50_withdrawal_gain = _r50(left_metrics) - _r50(acquisition_zero)
        withdrawal_decomposition.append({
            "coordinate": coordinate, "acquisition_model": acquisition,
            "extracted_carrier": carrier,
            "acquisition_both_metrics": acquisition_both,
            "acquisition_alpha_zero_metrics": acquisition_zero,
            "extracted_metrics": left_metrics,
            "auc_context_gain_before_withdrawal": auc_context_gain,
            "auc_gain_recovered_by_withdrawal": auc_withdrawal_gain,
            "auc_context_gain_recovered_pct": (
                None if auc_context_gain == 0 else
                100.0 * auc_withdrawal_gain / auc_context_gain
            ),
            "r50_context_gain_before_withdrawal": r50_context_gain,
            "r50_gain_recovered_by_withdrawal": r50_withdrawal_gain,
            "r50_context_gain_recovered_pct": (
                None if r50_context_gain == 0 else
                100.0 * r50_withdrawal_gain / r50_context_gain
            ),
            "paired_extracted_minus_acquisition_alpha_zero_auc": (
                paired_stratified_macro_auc_bootstrap(
                    probabilities(carrier),
                    probabilities(f"alpha0({acquisition})"), report_labels,
                    samples=2000, seed=int(spec["paired_bootstrap_seed"]),
                )
            ),
        })
        parent = carrier
    ce_probabilities = probabilities("CE_SINGLE_D000")
    self_average = np.ascontiguousarray(
        .5 * ce_probabilities + .5 * ce_probabilities, dtype=np.float32,
    )
    if not np.array_equal(self_average, ce_probabilities):
        raise RuntimeError("learned-handoff self-ensemble identity differs")
    self_ensemble_identity = {
        "model_id": "CE_SINGLE_D000",
        "operation": "uniform_probability_average_of_identical_checkpoint_twice",
        "input_probability_sha256": array_sha256(
            "probabilities", ce_probabilities,
        ),
        "output_probability_sha256": array_sha256(
            "probabilities", self_average,
        ),
        "byte_identical": True,
    }
    fit_rows = [row for row in rows if "training" in row]
    return artifact({"parents":parents,"report_role":"V_report","control_rows":controls,"model_rows":rows,"required_causal_comparisons":comparisons,"adjacent_carrier_comparisons":adjacent,"learned_carrier_minus_direct_comparisons":learned_vs_direct,"rung_withdrawal_decomposition":withdrawal_decomposition,"self_ensemble_identity_control":self_ensemble_identity,"paired_bootstrap_samples":2000,"paired_bootstrap_seed":int(spec["paired_bootstrap_seed"]),"all_25_fits_reported":len(fit_rows)==25,"all_fit_histories_reported":len(fit_rows)==25 and all(len(row["training"]["validation_history"])==row["training"]["completed_passes"] for row in fit_rows),"campaign_durable_bytes_before_aggregate":_tree_bytes(root),"poor_metrics_do_not_control_completion":True,"final_test_accessed":False},contract=AGGREGATE_CONTRACT)


def validate_aggregate(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=AGGREGATE_CONTRACT)

    def valid_alpha_curve(diagnostic: Mapping[str, Any]) -> bool:
        curve = diagnostic.get("alpha_validation_curve")
        if not isinstance(curve, list) or [
            row.get("alpha") for row in curve if isinstance(row, Mapping)
        ] != [0.0, 0.25, 0.5, 0.75, 1.0]:
            return False
        return all(
            isinstance(row.get("metrics"), Mapping)
            and all(
                name in row["metrics"]
                for name in (
                    "accuracy", "macro_ovr_auc",
                    "macro_mean_log_qcd_rejection_at_50pct_signal",
                )
            )
            for row in curve
        )

    model_rows = list(value.get("model_rows", ()))
    model_ids = [row.get("model_id") for row in model_rows]
    control_ids = [row.get("model_id") for row in value.get("control_rows", ())]
    fit_rows = model_rows[1:]
    two_view_ids = {
        node_id for node_id in FIT_ORDER
        if NODE_REGISTRY[node_id].input_protocol != "standard_hlt_v1"
    }
    diagnostic_ids = {
        row.get("model_id") for row in fit_rows if "diagnostics" in row
    }
    expected_causal_pairs = [
        ["CE_FUSION_D000_D000", "CE_SINGLE_D000"],
        ["CE_FUSION_D000_D000", "PARAMETER_MATCHED_SINGLE_D000"],
        ["DIRECT_VIEW_MORPH_U100_TO_D000", "CE_FUSION_D000_D000"],
        ["STATIC_U100_D000", "DIRECT_VIEW_MORPH_U100_TO_D000"],
        ["alpha0(DIRECT_VIEW_MORPH_U100_TO_D000)", "CE_SINGLE_D000"],
        [
            "DIRECT_VIEW_MORPH_WITHDRAW_D000",
            "alpha0(DIRECT_VIEW_MORPH_U100_TO_D000)",
        ],
        ["DIRECT_VIEW_MORPH_WITHDRAW_D000", "LEARNED_T_D000"],
        ["LEARNED_T_D000", "DIRECT_KD_D000"],
    ]
    expected_adjacent_pairs = []
    expected_direct_pairs = []
    parent = "SOURCE_U100"
    for coordinate in RUNG_ORDER:
        carrier = f"LEARNED_T_{coordinate}"
        expected_adjacent_pairs.append([carrier, parent])
        expected_direct_pairs.append(
            [carrier, f"LEARNED_DIRECT_{coordinate}"]
        )
        parent = carrier
    causal_pairs = [
        [row.get("left"), row.get("right")]
        for row in value.get("required_causal_comparisons", ())
    ]
    adjacent_pairs = [
        [row.get("left"), row.get("right")]
        for row in value.get("adjacent_carrier_comparisons", ())
    ]
    direct_pairs = [
        [row.get("left"), row.get("right")]
        for row in value.get("learned_carrier_minus_direct_comparisons", ())
    ]
    withdrawal_coordinates = [
        row.get("coordinate")
        for row in value.get("rung_withdrawal_decomposition", ())
    ]
    training_valid = all(
        row.get("training", {}).get("rolling_resume_published") is False
        and row.get("training", {}).get("partial_checkpoint_reuse") is False
        and row.get("training", {}).get("selected_pass", 0)
        >= row.get("training", {}).get("checkpoint_selection_minimum_pass", 1)
        for row in fit_rows
    )
    diagnostics_valid = diagnostic_ids == two_view_ids and all(
        valid_alpha_curve(row["diagnostics"]) and (
            row["diagnostics"].get(
                "all_routes_use_identical_V_report_identities",
            ) is True
            and row["diagnostics"].get(
                "all_routes_use_identical_V_report_labels",
            ) is True
        )
        if NODE_REGISTRY[row["model_id"]].role not in {
            "fusion_withdrawal", "morph_withdrawal",
        }
        else (
            row["diagnostics"].get(
                "all_routes_use_identical_validation_identities",
            ) is True
            and row["diagnostics"].get(
                "all_routes_use_identical_validation_labels",
            ) is True
        )
        for row in fit_rows if row["model_id"] in two_view_ids
    )
    if (
        model_ids != ["SOURCE_U100", *FIT_ORDER]
        or control_ids != ["M0CE60", "U000"]
        or value.get("report_role") != "V_report"
        or value.get("all_25_fits_reported") is not True
        or value.get("all_fit_histories_reported") is not True
        or not training_valid
        or not diagnostics_valid
        or causal_pairs != expected_causal_pairs
        or adjacent_pairs != expected_adjacent_pairs
        or direct_pairs != expected_direct_pairs
        or withdrawal_coordinates != list(RUNG_ORDER)
        or value.get("paired_bootstrap_samples") != 2000
        or isinstance(value.get("paired_bootstrap_seed"), bool)
        or not isinstance(value.get("paired_bootstrap_seed"), int)
        or value.get("self_ensemble_identity_control", {}).get(
            "byte_identical"
        ) is not True
        or value.get("poor_metrics_do_not_control_completion") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("learned-handoff aggregate semantics differ")
    return digest


class LearnedHandoffWorkflow:
    def __init__(self,spec,*,recovery_spec_sha256=None,execution_source_commit=None):
        validate_campaign(spec); self.spec=spec; self.recovery_spec_sha256=recovery_spec_sha256; self.execution_source_commit=execution_source_commit

    def run(self,task_id,*,device="cuda"):
        task={row["task_id"]:row for row in self.spec["tasks"]}[task_id]; kind=task["kind"]
        if kind=="authenticate":
            return _stage(self.spec,"authenticate",{"source_lock_sha256":validate_source_lock(load_json(self.spec["artifact_paths"]["source_lock"])),"control_lock_sha256":validate_control_lock(load_json(self.spec["artifact_paths"]["control_lock"])),"population_lock_sha256":validate_artifact(load_json(self.spec["artifact_paths"]["population_lock"]),contract=POPULATION_LOCK_CONTRACT),"seed_lock_sha256":validate_artifact(load_json(self.spec["artifact_paths"]["seed_lock"]),contract=SEED_LOCK_CONTRACT),"passed":True})
        if kind=="partition": return run_partition(self.spec)
        if kind=="audit":
            free=shutil.disk_usage(self.spec["campaign_root"]).free
            if free<int(self.spec["minimum_free_disk_bytes"]): raise OSError("learned-handoff storage floor unavailable")
            audit=build_capacity_audit(self.spec); write_immutable_json(self.spec["artifact_paths"]["capacity_audit"],audit)
            return _stage(self.spec,"audit_sources_and_storage",{"free_disk_bytes":free,"capacity_audit":audit["content_hash"],"durable_particle_views":False,"durable_hidden_states":False,"rolling_resume":False,"passed":True})
        if kind=="preflight":
            value=run_execution_acceptance(self.spec,device=device); write_immutable_json(self.spec["artifact_paths"]["execution_acceptance"],value); return value
        if kind=="train": return run_fit(self.spec,task["node_id"],device=device,recovery_spec_sha256=self.recovery_spec_sha256,execution_source_commit=self.execution_source_commit)
        if kind=="source_reducer": return run_source_reducer(self.spec,device=device,execution_source_commit=self.execution_source_commit)
        if kind=="model_reducer": return run_model_reducer(self.spec,task["node_id"],device=device,execution_source_commit=self.execution_source_commit)
        if kind=="extract": return run_extract(self.spec,task["node_id"],task["distribution_id"],device=device,recovery_spec_sha256=self.recovery_spec_sha256)
        if kind=="extracted_reducer": return run_extracted_reducer(self.spec,task["node_id"],task["distribution_id"],device=device,execution_source_commit=self.execution_source_commit)
        if kind=="control_reducer": return run_control_reducer(self.spec,task["control_id"],device=device)
        if kind=="aggregate":
            value=build_aggregate(self.spec); validate_aggregate(value); write_immutable_json(Path(self.spec["campaign_root"])/"reports/validation_aggregate.json",value); return value
        if kind=="complete":
            aggregate=load_json(Path(self.spec["campaign_root"])/"reports/validation_aggregate.json"); digest=validate_aggregate(aggregate)
            value=artifact({"parents":{"campaign_spec":self.spec["content_hash"],"aggregate":digest},"fresh_fit_count":25,"scientific_result_does_not_control_completion":True,"final_test_accessed":False},contract=COMPLETE_CONTRACT); write_immutable_json(Path(self.spec["campaign_root"])/"reports/campaign_complete.json",value); return value
        raise KeyError("unknown learned-handoff task kind")


__all__=[
    "SCIENCE_GATE_TASKS", "LearnedHandoffWorkflow", "build_aggregate",
    "task_outputs", "validate_aggregate", "validate_science_gate",
]

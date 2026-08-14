"""Cross-arm ranking, locks, and sealed HLT-only final evaluation for HCWDL-UB."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Mapping
from hlt_classification.data.cache_contracts import (
    load_json, require_sha256, validate_content_hash, with_content_hash,
    write_immutable_json,
)
from .engine import evaluate_model, validate_pmard_training_report
from .hcwdl_unified_balanced_contracts import (
    CAMPAIGN_COMPLETION_CONTRACT, FINAL_EVALUATION_CONTRACT,
    RECIPE_SWEEP_AGGREGATE_CONTRACT, finalist_lock_payload,
    sweep_aggregate_payload, validate_foundation_lock,
    validate_foundation_spec, validate_arm_aggregate,
    validate_arm_completion, validate_recipe_sweep, validate_sweep_aggregate,
    validate_execution_lock, validate_finalist_lock,
)
from .hcwdl_unified_balanced_graph import ARM_IDS
from .loaders import load_pmard_model, scouting_model_factory_for_report
from .dataset import iterate_model_batches
from .selective_assignment import RowSelection


def build_sweep_aggregate(recipe_sweep_path: str|Path)->dict[str,Any]:
    sweep=load_json(recipe_sweep_path); sweep_hash=validate_recipe_sweep(sweep)
    root=Path(recipe_sweep_path).parent; completions={};aggregates={};rankings=[]
    for arm in ARM_IDS:
        completion=load_json(root/arm/"reports/campaign_complete.json"); aggregate=load_json(root/arm/"reports/validation_aggregate.json")
        completion_hash=validate_arm_completion(completion)
        aggregate_hash=validate_arm_aggregate(aggregate)
        if completion["aggregate_sha256"]!=aggregate_hash or completion["arm_spec_sha256"]!=sweep["arm_spec_sha256"][arm]: raise ValueError("HCWDL-UB sweep arm lineage differs")
        rows={row["node_id"]:row for row in aggregate["rows"]}
        rankings.append({"arm_id":arm,"d0f_macro_ovr_auc":rows["D0F"]["metrics"]["macro_ovr_auc"],"j100_macro_ovr_auc":rows["J100"]["metrics"]["macro_ovr_auc"],"m1f_macro_ovr_auc":rows["M1F"]["metrics"]["macro_ovr_auc"],"d0f_cross_entropy":rows["D0F"]["metrics"]["cross_entropy"],"gpu_hours":aggregate["gpu_hours"]})
        completions[arm]=completion_hash;aggregates[arm]=aggregate_hash
    rankings.sort(key=lambda row:(-float(row["d0f_macro_ovr_auc"]),-float(row["j100_macro_ovr_auc"]),-float(row["m1f_macro_ovr_auc"]),float(row["d0f_cross_entropy"]),row["arm_id"]))
    return sweep_aggregate_payload(recipe_sweep_sha256=sweep_hash,arm_completions=completions,arm_aggregates=aggregates,rankings=rankings)


def build_finalist_lock(*,sweep_aggregate_path:str|Path,foundation_lock_path:str|Path,arms_root:str|Path)->dict[str,Any]:
    aggregate=load_json(sweep_aggregate_path);aggregate_hash=validate_sweep_aggregate(aggregate)
    foundation=load_json(foundation_lock_path);foundation_hash=validate_foundation_lock(foundation)
    sweep = load_json(Path(arms_root) / "recipe_sweep.json")
    sweep_hash = validate_recipe_sweep(sweep)
    if (
        aggregate.get("recipe_sweep_sha256") != sweep_hash
        or sweep.get("foundation_lock_sha256") != foundation_hash
    ):
        raise ValueError("HCWDL-UB finalist sweep/foundation lineage differs")
    selected=[row["arm_id"] for row in aggregate["rankings"][:2]]; finalists=[]
    for arm in selected:
        for node in ("D0F","J100","M1F","M1J"):
            path=Path(arms_root)/arm/"training"/node/"training_report.json";report=load_json(path);report_hash=validate_pmard_training_report(report)
            _validate_hlt_candidate_report(report, f"{arm}/{node}")
            finalists.append({"canonical_id":f"{arm}/{node}","report_path":str(path.resolve()),"report_sha256":report_hash,"checkpoint_sha256":report["selected_checkpoint_sha256"]})
    return finalist_lock_payload(sweep_aggregate_sha256=aggregate_hash,foundation_lock_sha256=foundation_hash,selected_arms=selected,finalists=finalists)


def run_sealed_final_evaluation(*,foundation_spec_path:str|Path,finalist_lock_path:str|Path,execution_lock_path:str|Path,output:str|Path,device:str="cuda")->dict[str,Any]:
    foundation=load_json(foundation_spec_path);finalist=load_json(finalist_lock_path);execution=load_json(execution_lock_path)
    foundation_hash=validate_foundation_spec(foundation)
    finalist_hash=validate_finalist_lock(finalist);execution_hash=validate_execution_lock(execution)
    if execution["finalist_lock_sha256"]!=finalist_hash or execution.get("authorized") is not True: raise PermissionError("HCWDL-UB final execution is not authorized")
    foundation_lock=load_json(Path(foundation["campaign_root"])/"locks/foundation.json")
    foundation_lock_hash=validate_foundation_lock(foundation_lock)
    if finalist.get("foundation_lock_sha256")!=foundation_lock_hash:
        raise PermissionError("HCWDL-UB finalist lock has a different foundation")
    if (
        execution.get("source_commit")!=foundation["source_commit"]
        or execution.get("split_manifest_sha256")!=foundation["parents"]["split_manifest_sha256"]
        or execution.get("selection_manifest_sha256")!=foundation["parents"]["selection_manifest_sha256"]
    ):
        raise PermissionError("HCWDL-UB execution lock data/source lineage differs")
    claim=Path(output).parent/"execution_claim.json"
    selection_raw=load_json(foundation["artifact_paths"]["selection_manifest"]);split=load_json(foundation["artifact_paths"]["split_manifest"])
    selection=RowSelection(selection_raw,role="final_test",split_manifest_sha256=split["content_hash"])
    if selection.rows!=100_000: raise ValueError("HCWDL-UB sealed final-test population differs")
    candidates=[{"canonical_id":"shared/M0paired","report_path":str((Path(foundation["campaign_root"])/"training/M0paired/training_report.json").resolve()),"report_sha256":foundation_lock["m0paired_report_sha256"],"checkpoint_sha256":foundation_lock["m0paired_checkpoint_sha256"]},*finalist["finalists"]]
    rows=[]
    claim_payload=with_content_hash({"contract":"HCWDL_UNIFIED_BALANCED_EXECUTION_CLAIM/v1","schema_version":1,"execution_lock_sha256":execution_hash,"state":"claimed_once"})
    if claim.exists():
        if load_json(claim)!=claim_payload: raise PermissionError("HCWDL-UB final execution is claimed by another lock")
    else: write_immutable_json(claim,claim_payload)
    for record in candidates:
        report=load_json(record["report_path"]);report_hash=validate_pmard_training_report(report)
        _validate_hlt_candidate_report(report, record["canonical_id"])
        if "report_sha256" in record and record["report_sha256"]!=report_hash: raise ValueError("HCWDL-UB finalist report drifted")
        if record.get("checkpoint_sha256")!=report["selected_checkpoint_sha256"]: raise ValueError("HCWDL-UB finalist checkpoint drifted")
        model,_=load_pmard_model(record["report_path"],model_factory=scouting_model_factory_for_report(report),device=device)
        batches=lambda:iterate_model_batches(split,data_root=foundation["data_root"],role="final_test",input_mode="hlt",epoch=0,batch_size=256,sampler_seed=1337,row_selection=selection)
        metrics=evaluate_model(model,batches(),device=device,input_key="hlt")
        rows.append({"canonical_id":record["canonical_id"],"report_sha256":report_hash,"checkpoint_sha256":report["selected_checkpoint_sha256"],"metrics":metrics})
    payload=with_content_hash({"contract":FINAL_EVALUATION_CONTRACT,"schema_version":1,"foundation_spec_sha256":foundation_hash,"finalist_lock_sha256":finalist_hash,"execution_lock_sha256":execution_hash,"rows":rows,"final_test_rows":100_000,"final_test_accessed":True,"test_did_not_select_models":True})
    validate_final_evaluation(payload, finalist_lock=finalist)
    write_immutable_json(output,payload);return payload


def _validate_hlt_candidate_report(report: Mapping[str, Any], canonical_id: str) -> None:
    scientific = report.get("scientific_config", {})
    node = scientific.get("node", {})
    if (
        scientific.get("canonical_node_id") != canonical_id
        or node.get("canonical_id") != canonical_id
        or node.get("input_domain") != "hlt"
    ):
        raise PermissionError("HCWDL-UB sealed finalist is not the exact registered HLT-input node")


def validate_final_evaluation(
    value: Mapping[str, Any], *, finalist_lock: Mapping[str, Any] | None = None,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=FINAL_EVALUATION_CONTRACT,
        expected_schema_version=1,
    )
    rows = value.get("rows", ())
    identities = [row.get("canonical_id") for row in rows]
    if (
        value.get("final_test_rows") != 100_000
        or value.get("final_test_accessed") is not True
        or value.get("test_did_not_select_models") is not True
        or len(rows) != 9
        or len(set(identities)) != 9
        or identities[0] != "shared/M0paired"
    ):
        raise ValueError("HCWDL-UB sealed final evaluation coverage differs")
    if finalist_lock is not None:
        finalist_hash = validate_finalist_lock(finalist_lock)
        expected = ["shared/M0paired", *[row["canonical_id"] for row in finalist_lock["finalists"]]]
        if value.get("finalist_lock_sha256") != finalist_hash or identities != expected:
            raise ValueError("HCWDL-UB final evaluation finalist lineage differs")
    for row in rows:
        require_sha256(row.get("report_sha256"), name="final report")
        require_sha256(row.get("checkpoint_sha256"), name="final checkpoint")
        metrics = row.get("metrics", {})
        for key in (
            "cross_entropy", "accuracy", "macro_ovr_auc",
            "macro_mean_log_qcd_rejection_at_50pct_signal",
        ):
            if not math.isfinite(float(metrics.get(key, math.nan))):
                raise ValueError("HCWDL-UB final evaluation contains a nonfinite metric")
    return digest


def completion_payload(*,sweep_aggregate_sha256:str,finalist_lock_sha256:str,final_evaluation_sha256:str)->dict[str,Any]:
    return with_content_hash({"contract":CAMPAIGN_COMPLETION_CONTRACT,"schema_version":1,"sweep_aggregate_sha256":require_sha256(sweep_aggregate_sha256,name="sweep aggregate"),"finalist_lock_sha256":require_sha256(finalist_lock_sha256,name="finalist lock"),"final_evaluation_sha256":require_sha256(final_evaluation_sha256,name="final evaluation"),"six_arms_complete":True,"sealed_test_complete":True,"final_test_accessed":True,"scientific_result_does_not_control_completion":True})


def validate_campaign_completion(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value, expected_contract=CAMPAIGN_COMPLETION_CONTRACT,
        expected_schema_version=1,
    )
    if (
        value.get("six_arms_complete") is not True
        or value.get("sealed_test_complete") is not True
        or value.get("final_test_accessed") is not True
        or value.get("scientific_result_does_not_control_completion") is not True
    ):
        raise ValueError("HCWDL-UB campaign completion semantics differ")
    for key in ("sweep_aggregate_sha256", "finalist_lock_sha256", "final_evaluation_sha256"):
        require_sha256(value.get(key), name=key)
    return digest


__all__=["build_finalist_lock","build_sweep_aggregate","completion_payload","run_sealed_final_evaluation","validate_campaign_completion","validate_final_evaluation"]

"""Explicitly authorized sealed HLT-only final evaluation for HCWDL-MHPE."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import load_json, validate_content_hash, with_content_hash, write_immutable_json

from .dataset import iterate_model_batches
from .engine import classification_metrics, precompute_teacher_targets, validate_pmard_training_report
from .hcwdl_mhpe_campaign import validate_campaign
from .hcwdl_mhpe_contracts import (
    EXECUTION_LOCK_CONTRACT, campaign_profile, final_evaluation_contract,
    finalist_lock_contract, validate_execution_lock,
)
from .hcwdl_mhpe_graph import (
    PROFILE_DENSE_ANCHOR50_300K60, endpoint_ensemble,
    ensemble_components, ensemble_weight_rationals, finalists,
)
from .hcwdl_mhpe_targets import uniform_probability_ensemble, weighted_probability_ensemble
from .loaders import load_pmard_model, scouting_model_factory_for_report
from .selective_assignment import RowSelection


def run_sealed_final_evaluation(
    *, campaign_spec_path: str | Path, finalist_lock_path: str | Path,
    execution_lock_path: str | Path, output: str | Path, device: str = "cuda",
) -> dict[str, Any]:
    spec = load_json(campaign_spec_path); spec_hash = validate_campaign(spec, verify_source_tree=False)
    profile = campaign_profile(spec)
    finalist_ids = finalists(profile); endpoint = endpoint_ensemble(profile)
    components = ensemble_components(profile)[endpoint]
    finalist = load_json(finalist_lock_path)
    finalist_hash = validate_content_hash(finalist, expected_contract=finalist_lock_contract(profile), expected_schema_version=1)
    if [row["node_id"] for row in finalist["entries"]] != list(finalist_ids):
        raise ValueError("HCWDL-MHPE final registry differs")
    finalist_by_id = {row["node_id"]: row for row in finalist["entries"]}
    execution = load_json(execution_lock_path); execution_hash = validate_execution_lock(execution)
    if execution["campaign_spec_sha256"] != spec_hash or execution["finalist_lock_sha256"] != finalist_hash:
        raise PermissionError("HCWDL-MHPE final locks differ")
    root = Path(spec["campaign_root"]); claim = root / "final_test/execution_claim.json"
    claim_payload = with_content_hash({
        "contract": "HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_EXECUTION_CLAIM/v1",
        "schema_version": 1, "execution_lock_sha256": execution_hash, "state": "claimed_once",
    })
    write_immutable_json(claim, claim_payload)
    reuse = load_json(spec["reuse_lock_path"]); foundation_root = Path(reuse["foundation_spec_path"]).parent
    foundation = load_json(reuse["foundation_spec_path"])
    split = load_json(foundation["artifact_paths"]["split_manifest"])
    split_hash = validate_content_hash(split, expected_contract=split["contract"], expected_schema_version=split["schema_version"])
    selection_raw = load_json(foundation["artifact_paths"]["selection_manifest"])
    selection = RowSelection(selection_raw, role="final_test", split_manifest_sha256=split_hash)
    batch_size = 256; sampler_seed = 1337
    def batches():
        return iterate_model_batches(
            split, data_root=foundation["data_root"], role="final_test", input_mode="hlt",
            epoch=0, batch_size=batch_size, sampler_seed=sampler_seed, row_selection=selection,
        )
    label_parts = []; identity_list = []
    for batch in batches():
        label_parts.append(np.asarray(batch["labels"], np.int64))
        identity_list.extend(map(str, batch["identity_keys"]))
    labels = np.concatenate(label_parts); identities = tuple(identity_list)
    component_logits = {}; rows = []
    for node_id in ("M0paired", *components, "M1"):
        directory = foundation_root / "training/M0paired" if node_id == "M0paired" else root / "training" / node_id
        report_path = directory / "training_report.json"; report = load_json(report_path)
        report_hash = validate_pmard_training_report(report)
        scientific = report.get("scientific_config", {})
        node = scientific.get("node", {})
        if (node.get("input_domain") != "hlt"
                or scientific.get("input_key", "hlt") != "hlt"):
            raise PermissionError(
                f"HCWDL-MHPE finalist {node_id} is not authenticated exact-HLT inference"
            )
        registered = finalist_by_id[node_id]
        if (registered["report_sha256"] != report_hash
                or registered["checkpoint_sha256"] != report["selected_checkpoint_sha256"]):
            raise ValueError("HCWDL-MHPE finalist model lineage differs")
        model, _ = load_pmard_model(report_path, model_factory=scouting_model_factory_for_report(report), device=device)
        target = precompute_teacher_targets(
            model, batches(), input_key="hlt", device=device,
            teacher_report_sha256=report_hash, split_manifest_sha256=split_hash,
        )
        if identities != target.identities:
            raise ValueError("HCWDL-MHPE final identity/label order differs")
        metrics = classification_metrics(target.logits, labels)
        rows.append({"node_id": node_id, "report_sha256": report_hash, "checkpoint_sha256": report["selected_checkpoint_sha256"], "metrics": metrics})
        if node_id in components:
            component_logits[node_id] = target.logits
    probability = (
        weighted_probability_ensemble(
            component_logits, temperature=1,
            weights=ensemble_weight_rationals(profile, endpoint),
        )
        if profile == PROFILE_DENSE_ANCHOR50_300K60
        else uniform_probability_ensemble(component_logits, temperature=1)
    )
    ensemble_metrics = classification_metrics(np.log(np.maximum(probability, 1e-30)), labels)
    stage_report = load_json(root / "reports" / f"{endpoint}_stage.json")
    if (finalist_by_id[endpoint]["report_sha256"] != stage_report.get("content_hash")
            or finalist_by_id[endpoint]["checkpoint_sha256"] != stage_report.get("content_hash")):
        raise ValueError("HCWDL-MHPE finalist ensemble lineage differs")
    insertion = list(finalist_ids).index(endpoint)
    rows.insert(insertion, {"node_id": endpoint, "component_order": list(components), "metrics": ensemble_metrics})
    if [row["node_id"] for row in rows] != list(finalist_ids):
        raise RuntimeError("HCWDL-MHPE final evaluation ordering differs")
    payload = with_content_hash({
        "contract": final_evaluation_contract(profile), "schema_version": 1,
        "campaign_spec_sha256": spec_hash, "finalist_lock_sha256": finalist_hash,
        "execution_lock_sha256": execution_hash, "rows": rows,
        "final_test_rows": selection.rows, "final_test_accessed": True,
        "test_did_not_select_models_or_weights": True,
    })
    write_immutable_json(output, payload); return payload


__all__ = ["run_sealed_final_evaluation"]

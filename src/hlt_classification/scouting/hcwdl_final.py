"""One-claim sealed HCWDL final-test evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, validate_content_hash, with_content_hash,
    write_immutable_json,
)

from .dataset import iterate_model_batches
from .engine import evaluate_model
from .hcwdl_locks import recover_or_claim_final_execution, validate_lock
from .highcov_cache import DenseAssignmentStore
from .loaders import load_pmard_model, scouting_model_factory_for_report
from .pmard_stream import iterate_pmard_batches
from .selective_assignment import RowSelection
from .hcwdl_shared_final import (
    claim_legacy_final_exposure, reject_legacy_final_after_shared_reservation,
)


EVALUATION_CONTRACT: Final = "HCWDL_FINAL_EVALUATION/v1"
PREVIOUS_EVALUATION_MANIFEST_CONTRACT: Final = "HCWDL_FINAL_EVALUATION_MANIFEST/v1"
EVALUATION_MANIFEST_CONTRACT: Final = "HCWDL_FINAL_EVALUATION_MANIFEST/v2"


def _validate_evaluation_report(
    value: Mapping[str, Any], *, finalist_hash: str, execution_hash: str,
    assignment_hash: str, row: Mapping[str, Any], node: str, domain: str,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=EVALUATION_CONTRACT, expected_schema_version=1,
    )
    expected = {
        "finalist_lock_sha256": finalist_hash,
        "execution_lock_sha256": execution_hash,
        "test_assignment_manifest_sha256": assignment_hash,
        "node_id": node,
        "seed": int(row["seed"]),
        "domain": domain,
        "training_report_sha256": row["report_sha256"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "selection_performed": False,
    }
    if any(value.get(name) != item for name, item in expected.items()):
        raise PermissionError("HCWDL final evaluation report lineage differs")
    metrics = value.get("metrics")
    required = {
        "cross_entropy", "accuracy", "macro_ovr_auc",
        "macro_mean_log_qcd_rejection_at_50pct_signal",
        "top_label_ece_15_bin",
    }
    if not isinstance(metrics, Mapping) or not required <= set(metrics):
        raise ValueError("HCWDL final evaluation metrics are incomplete")
    return digest


def _validate_evaluation_manifest(
    value: Mapping[str, Any], *, finalist_hash: str, execution_hash: str,
    assignment_hash: str, claim_hash: str,
    expected_reports: list[dict[str, object]],
) -> str:
    digest = validate_content_hash(
        value, expected_contract=EVALUATION_MANIFEST_CONTRACT,
        expected_schema_version=2,
    )
    expected = {
        "finalist_lock_sha256": finalist_hash,
        "execution_lock_sha256": execution_hash,
        "test_assignment_manifest_sha256": assignment_hash,
        "execution_claim_sha256": claim_hash,
        "reports": expected_reports,
        "evaluated_exact_frozen_registry": True,
        "test_used_for_selection": False,
        "interrupted_execution_recovery_supported": True,
    }
    if any(value.get(name) != item for name, item in expected.items()):
        raise PermissionError("HCWDL final evaluation manifest lineage differs")
    if value.get("claim_disposition") not in {
        "created_new_claim", "reused_existing_exact_claim",
    }:
        raise ValueError("HCWDL final evaluation claim disposition differs")
    return digest


def run_final_evaluation(
    *, split_manifest_path: str | Path, selection_manifest_path: str | Path,
    test_assignment_manifest_path: str | Path, finalist_lock_path: str | Path,
    execution_lock_path: str | Path, data_root: str | Path,
    output_root: str | Path, checkpoint_namespace_path: str | Path,
    device: str = "cuda", batch_size: int = 512,
) -> dict[str, Any]:
    # The legacy evaluator is retained only for a population that has never
    # entered the neutral shared-final protocol.  Once any shared reservation
    # exists, parent and representation finalists must both use the common
    # label-free prediction/locked-join pipeline.
    reject_legacy_final_after_shared_reservation(checkpoint_namespace_path)
    # Serialize against the neutral registrar and publish an irreversible
    # exposure marker before any ROOT-backed iterator can open a final-role
    # branch.  This closes the check-then-read race for pre-shared parent
    # campaigns while leaving their historical evaluator usable exactly once.
    claim_legacy_final_exposure(
        checkpoint_namespace_path,
        execution_identity={
            "split_manifest_sha256": sha256_file(split_manifest_path),
            "selection_manifest_sha256": sha256_file(selection_manifest_path),
            "test_assignment_manifest_sha256": sha256_file(
                test_assignment_manifest_path,
            ),
            "finalist_lock_sha256": sha256_file(finalist_lock_path),
            "execution_lock_sha256": sha256_file(execution_lock_path),
            "evaluator": "HCWDL_FINAL_EVALUATION/v1",
        },
    )
    split = load_json(split_manifest_path); selection_raw = load_json(selection_manifest_path)
    finalist = load_json(finalist_lock_path); execution = load_json(execution_lock_path)
    finalist_hash = validate_lock(finalist, expected_level="finalist")
    execution_hash = validate_lock(execution, expected_level="execution")
    if execution.get("parent_lock_sha256") != finalist_hash:
        raise ValueError("HCWDL execution/finalist lock chain differs")
    assignment = DenseAssignmentStore(test_assignment_manifest_path)
    selection = RowSelection(
        selection_raw, role="final_test", split_manifest_sha256=split["content_hash"],
    )
    root = Path(output_root)
    claim, claim_disposition = recover_or_claim_final_execution(
        root / "execution_claim.json", execution_lock=execution,
        test_assignment_manifest_sha256=assignment.manifest["content_hash"],
    )
    claim_hash = claim["content_hash"]

    def batches(domain: str):
        if domain in {"hlt", "toff"}:
            return iterate_model_batches(
                split, data_root=data_root, role="final_test", input_mode=domain,
                completed_locks=("finalist", "execution"), batch_size=batch_size,
                sampler_seed=1337, row_selection=selection,
            )
        return iterate_pmard_batches(
            split, data_root=data_root, role="final_test", matcher_model=None,
            alpha=1.0, repair_family="HIGHCOV_SHELL_EXACT/v1",
            matcher_variant="highcov_empirical_lexicographic_dr0p30_v1", threshold=0.0,
            completed_locks=("finalist", "execution"), batch_size=batch_size,
            sampler_seed=1337, assignment_store=assignment, row_selection=selection,
            repair_seed=1337,
        )

    reports = []
    for index, row in enumerate(finalist["payload"]["finalists"]):
        training_path = Path(row["report_path"])
        raw = load_json(training_path)
        if raw["content_hash"] != row["report_sha256"] or raw["selected_checkpoint_sha256"] != row["checkpoint_sha256"]:
            raise ValueError("HCWDL finalist training lineage differs")
        node = str(row["node_id"])
        domain = "toff" if node == "TOFF" else "d100" if node == "D100" else "hlt"
        input_key = "toff" if domain == "toff" else "privileged" if domain == "d100" else "hlt"
        output = root / f"{index:03d}_{node}_seed{row['seed']}.json"
        if output.exists():
            report = load_json(output)
            _validate_evaluation_report(
                report, finalist_hash=finalist_hash,
                execution_hash=execution_hash,
                assignment_hash=assignment.manifest["content_hash"],
                row=row, node=node, domain=domain,
            )
            reports.append({
                "path": str(output), "content_hash": report["content_hash"],
                "node_id": node, "seed": row["seed"],
            })
            continue
        model, training = load_pmard_model(
            training_path, model_factory=scouting_model_factory_for_report(raw), device=device,
        )
        metrics = evaluate_model(model, batches(domain), device=device, input_key=input_key)
        report = with_content_hash({
            "contract": EVALUATION_CONTRACT, "schema_version": 1,
            "finalist_lock_sha256": finalist_hash, "execution_lock_sha256": execution_hash,
            "test_assignment_manifest_sha256": assignment.manifest["content_hash"],
            "node_id": node, "seed": int(row["seed"]), "domain": domain,
            "training_report_sha256": training["content_hash"],
            "checkpoint_sha256": row["checkpoint_sha256"], "metrics": metrics,
            "selection_performed": False,
        })
        write_immutable_json(output, report)
        reports.append({"path": str(output), "content_hash": report["content_hash"], "node_id": node, "seed": row["seed"]})
    manifest = with_content_hash({
        "contract": EVALUATION_MANIFEST_CONTRACT, "schema_version": 2,
        "finalist_lock_sha256": finalist_hash, "execution_lock_sha256": execution_hash,
        "test_assignment_manifest_sha256": assignment.manifest["content_hash"],
        "execution_claim_sha256": claim_hash,
        "claim_disposition": claim_disposition,
        "reports": reports, "evaluated_exact_frozen_registry": True,
        "test_used_for_selection": False,
        "interrupted_execution_recovery_supported": True,
    })
    manifest_path = root / "evaluation_manifest.json"
    if manifest_path.exists():
        existing = load_json(manifest_path)
        _validate_evaluation_manifest(
            existing, finalist_hash=finalist_hash,
            execution_hash=execution_hash,
            assignment_hash=assignment.manifest["content_hash"],
            claim_hash=claim_hash, expected_reports=reports,
        )
        return existing
    write_immutable_json(manifest_path, manifest)
    return manifest


__all__ = [
    "EVALUATION_CONTRACT", "EVALUATION_MANIFEST_CONTRACT",
    "PREVIOUS_EVALUATION_MANIFEST_CONTRACT", "run_final_evaluation",
]

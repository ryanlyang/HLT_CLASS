"""One-claim sealed HCWDL final-test evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json

from .dataset import iterate_model_batches
from .engine import evaluate_model
from .hcwdl_locks import claim_final_execution, validate_lock
from .highcov_cache import DenseAssignmentStore
from .loaders import load_pmard_model, scouting_model_factory_for_report
from .pmard_stream import iterate_pmard_batches
from .selective_assignment import RowSelection


EVALUATION_CONTRACT: Final = "HCWDL_FINAL_EVALUATION/v1"
EVALUATION_MANIFEST_CONTRACT: Final = "HCWDL_FINAL_EVALUATION_MANIFEST/v1"


def run_final_evaluation(
    *, split_manifest_path: str | Path, selection_manifest_path: str | Path,
    test_assignment_manifest_path: str | Path, finalist_lock_path: str | Path,
    execution_lock_path: str | Path, data_root: str | Path,
    output_root: str | Path, device: str = "cuda", batch_size: int = 512,
) -> dict[str, Any]:
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
    claim_final_execution(
        root / "execution_claim.json", execution_lock=execution,
        test_assignment_manifest_sha256=assignment.manifest["content_hash"],
    )

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
        model, training = load_pmard_model(
            training_path, model_factory=scouting_model_factory_for_report(raw), device=device,
        )
        node = str(row["node_id"])
        domain = "toff" if node == "TOFF" else "d100" if node == "D100" else "hlt"
        input_key = "toff" if domain == "toff" else "privileged" if domain == "d100" else "hlt"
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
        output = root / f"{index:03d}_{node}_seed{row['seed']}.json"
        write_immutable_json(output, report)
        reports.append({"path": str(output), "content_hash": report["content_hash"], "node_id": node, "seed": row["seed"]})
    manifest = with_content_hash({
        "contract": EVALUATION_MANIFEST_CONTRACT, "schema_version": 1,
        "finalist_lock_sha256": finalist_hash, "execution_lock_sha256": execution_hash,
        "reports": reports, "evaluated_exact_frozen_registry": True,
        "test_used_for_selection": False,
    })
    write_immutable_json(root / "evaluation_manifest.json", manifest)
    return manifest


__all__ = ["EVALUATION_CONTRACT", "EVALUATION_MANIFEST_CONTRACT", "run_final_evaluation"]

from __future__ import annotations

from copy import deepcopy

import pytest

from hlt_classification.data.cache_contracts import canonical_json_bytes, with_content_hash
from hlt_classification.scouting.hcwdl_final_stream import (
    HLT_FINAL_BRANCHES,
    NATIVE_OFFLINE_FINAL_BRANCHES,
    SHELL_EXACT_FINAL_BRANCHES,
    build_branch_access_record,
)
from hlt_classification.scouting.hcwdl_representation_acceptance_evidence import (
    build_tigris_action_proof,
    build_validation_proxy_proof,
    validate_full_loss_probe,
    validate_tigris_action_proof,
    validate_validation_proxy_proof,
)
from hlt_classification.scouting.hcwdl_representation_resources import (
    artifact_reference,
    build_miniature_evidence,
    build_scheduler_evidence,
    build_scheduler_evidence_from_sacct,
    resource_table,
)
from hlt_classification.scouting.hcwdl_representation_smoke import (
    run_scientific_full_loss_probe,
)
from hlt_classification.scouting.schema import TREE_NAME


def _validation_proxy_access(path: str, branches) -> dict:
    return build_branch_access_record(
        path=path,
        capability_sha256={
            "hlt": "1" * 64,
            "shell_exact": "2" * 64,
            "native_offline": "3" * 64,
        }[path],
        branches=branches,
        source_rows=({
            "source_path": "validation/source.root",
            "source_file_sha256": "4" * 64,
            "tree": TREE_NAME,
            "entry_start": 0,
            "entry_stop": 15,
        },),
        population_sha256="5" * 64,
        task_id=f"validation-proxy:{path}",
        execution_lock_sha256=None,
    )


def test_validation_proxy_proof_rebuilds_all_three_label_free_streams() -> None:
    accesses = (
        _validation_proxy_access("hlt", HLT_FINAL_BRANCHES),
        _validation_proxy_access("shell_exact", SHELL_EXACT_FINAL_BRANCHES),
        _validation_proxy_access("native_offline", NATIVE_OFFLINE_FINAL_BRANCHES),
    )
    proof = build_validation_proxy_proof(
        source_commit="6" * 40,
        representation_recipe_sha256="7" * 64,
        validation_population_sha256="5" * 64,
        rows=15,
        branch_access_records=accesses,
        prediction_manifest_sha256s=("8" * 64, "9" * 64, "a" * 64),
        metric_report_sha256="b" * 64,
        runtime_signature_sha256="c" * 64,
    )
    assert proof["role"] == "validation"
    assert proof["final_role_accessed"] is False
    assert validate_validation_proxy_proof(proof) == proof["content_hash"]

    forged = deepcopy(proof)
    forged["branch_access_records"][0]["label_free"] = False
    forged["branch_access_records"][0] = with_content_hash(
        forged["branch_access_records"][0]
    )
    forged["branch_access_sha256s"][0] = forged[
        "branch_access_records"
    ][0]["content_hash"]
    forged = with_content_hash(forged)
    with pytest.raises(ValueError, match="not canonical"):
        validate_validation_proxy_proof(forged)


def test_full_loss_action_validator_recomputes_the_24_plus_4_registry() -> None:
    report = run_scientific_full_loss_probe(device="cpu")
    assert validate_full_loss_probe(report) == report["content_hash"]
    forged = deepcopy(report)
    forged["execution_ids"] = forged["execution_ids"][:-1]
    forged = with_content_hash(forged)
    with pytest.raises(ValueError, match="execution registry"):
        validate_full_loss_probe(forged)


def _publish(path, artifact):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(artifact))
    return artifact_reference(path)


def test_local_action_proof_is_nonauthorizing_and_cross_job_lineage_fails(
    tmp_path,
) -> None:
    source = "d" * 40
    recipe = "e" * 64
    request = resource_table(mode="smoke")["gpu_representation"]
    worker = tmp_path / "worker.sh"
    worker.write_bytes(b"#!/bin/bash\nproduction-worker\n")
    worker_ref = artifact_reference(worker)
    workers = {"ordinary": worker_ref}

    def scheduler(job_id: int):
        value = build_scheduler_evidence(
            job_id=job_id, task_key="acceptance-full_loss_probe",
            resource_class="gpu_representation", source_commit=source,
            representation_recipe_sha256=recipe, worker_role="ordinary",
            worker=worker_ref, request=request, state="COMPLETED",
            exit_code="0:0", peak_rss_bytes=1, elapsed_seconds=1,
        )
        return value, _publish(tmp_path / f"scheduler-{job_id}.json", value)

    result = run_scientific_full_loss_probe(device="cpu")
    result_ref = _publish(tmp_path / "full-loss.json", result)
    scheduler_one, scheduler_one_ref = scheduler(41)
    miniature = build_miniature_evidence(
        evidence_kind="full_loss_probe", scheduler_evidence=scheduler_one,
        representation_recipe_sha256=recipe, rows=8,
        result_artifact=result_ref,
    )
    miniature_ref = _publish(tmp_path / "miniature-41.json", miniature)
    proof = build_tigris_action_proof(
        evidence_kind="full_loss_probe", source_commit=source,
        representation_recipe_sha256=recipe,
        scheduler_evidence=scheduler_one_ref,
        miniature_evidence=miniature_ref, result_artifact=result_ref,
        resource_request=request, expected_workers=workers,
    )
    assert proof["authorization_capable"] is False
    validate_tigris_action_proof(
        proof, resource_request=request, expected_workers=workers,
    )
    with pytest.raises(PermissionError, match="nonauthorizing local fixture"):
        validate_tigris_action_proof(
            proof, resource_request=request, expected_workers=workers,
            require_genuine=True,
        )

    _, scheduler_two_ref = scheduler(42)
    with pytest.raises(PermissionError, match="job ID differs"):
        build_tigris_action_proof(
            evidence_kind="full_loss_probe", source_commit=source,
            representation_recipe_sha256=recipe,
            scheduler_evidence=scheduler_two_ref,
            miniature_evidence=miniature_ref, result_artifact=result_ref,
            resource_request=request, expected_workers=workers,
        )

    alternate_result_ref = _publish(tmp_path / "alternate-full-loss.json", result)
    with pytest.raises(PermissionError, match="immutable miniature output"):
        build_tigris_action_proof(
            evidence_kind="full_loss_probe", source_commit=source,
            representation_recipe_sha256=recipe,
            scheduler_evidence=scheduler_one_ref,
            miniature_evidence=miniature_ref,
            result_artifact=alternate_result_ref,
            resource_request=request, expected_workers=workers,
        )


def test_raw_scheduler_builder_cannot_mint_authority_off_tigris(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
    worker = tmp_path / "worker.sh"
    worker.write_bytes(b"#!/bin/bash\n")
    raw = tmp_path / "sacct.txt"
    raw.write_text("not reached\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="captured on Tigris|collector Slurm job"):
        build_scheduler_evidence_from_sacct(
            raw_accounting_record=artifact_reference(raw),
            task_key="acceptance-full_loss_probe",
            resource_class="gpu_representation", source_commit="f" * 40,
            representation_recipe_sha256="1" * 64,
            worker_role="ordinary", worker=artifact_reference(worker),
            request=resource_table(mode="smoke")["gpu_representation"],
        )

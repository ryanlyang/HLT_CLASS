from __future__ import annotations

from copy import deepcopy
import inspect

import pytest

from hlt_classification.data.cache_contracts import canonical_json_bytes, with_content_hash
from hlt_classification.scouting import hcwdl_representation_acceptance_evidence as evidence
from hlt_classification.scouting import hcwdl_representation_nonfinal_acceptance as nf
from hlt_classification.scouting.hcwdl_representation_campaign import (
    REQUIRED_TIGRIS_CHECKS,
)
from hlt_classification.scouting.hcwdl_representation_acceptance_evidence import (
    build_tigris_action_proof,
    build_validation_proxy_proof,
    validate_full_loss_probe,
    validate_tigris_action_proof,
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
def test_public_validation_proxy_builder_cannot_accept_digest_only_evidence() -> None:
    assert set(inspect.signature(build_validation_proxy_proof).parameters) == {
        "result_reference", "authority", "authority_validator",
    }
    with pytest.raises(TypeError):
        build_validation_proxy_proof(
            source_commit="6" * 40,
            representation_recipe_sha256="7" * 64,
            validation_population_sha256="5" * 64,
            rows=15,
            branch_access_records=[],
            prediction_manifest_sha256s=("8" * 64, "9" * 64, "a" * 64),
            metric_report_sha256="b" * 64,
            runtime_signature_sha256="c" * 64,
        )


def test_full_loss_action_validator_recomputes_the_24_plus_4_registry() -> None:
    report = run_scientific_full_loss_probe(device="cpu")
    assert validate_full_loss_probe(report) == report["content_hash"]
    forged = deepcopy(report)
    forged["execution_ids"] = forged["execution_ids"][:-1]
    forged = with_content_hash(forged)
    with pytest.raises(ValueError, match="execution registry"):
        validate_full_loss_probe(forged)


def test_tigris_bundle_rejects_mixed_nonfinal_authorities(monkeypatch) -> None:
    source = "1" * 40
    recipe = "2" * 64
    objects = {
        "profile": {
            "requests": {"gpu_representation": {}},
            "measurement_environment": {"production_workers": {}},
        },
        "inventory": {},
        "storage": {},
    }
    for index, evidence_kind in enumerate(REQUIRED_TIGRIS_CHECKS, start=1):
        if evidence_kind in {
            "two_update_full_loss", "usr1_exact_resume", "validation_only_proxy",
        }:
            objects[evidence_kind] = {
                "source_commit": source,
                "representation_recipe_sha256": recipe,
                "authority_sha256": (
                    "a" * 64 if evidence_kind == "two_update_full_loss" else "b" * 64
                ),
                "scheduler_evidence": {"row": {"job_id": 100 + index}},
                "scheduler_job_id": 100 + index,
            }
        else:
            objects[evidence_kind] = {
                "evidence_kind": evidence_kind,
                "source_commit": source,
                "representation_recipe_sha256": recipe,
                "scheduler_evidence": {"path": "scheduler.json", "sha256": "3" * 64},
                "miniature_evidence": {"path": "miniature.json", "sha256": "4" * 64},
                "job_id": 100 + index,
                "result_execution_sha256": f"{index:064x}",
            }

    def load(reference, **_kwargs):
        key = reference["kind"]
        return objects[key], f"{list(objects).index(key) + 1:064x}"

    monkeypatch.setattr(evidence, "load_authenticated_json_reference", load)
    monkeypatch.setattr(evidence, "validate_measured_profile", lambda *_a, **_k: None)
    monkeypatch.setattr(evidence, "validate_fixed_size_inventory", lambda *_a, **_k: None)
    monkeypatch.setattr(evidence, "validate_storage_estimate", lambda *_a, **_k: None)
    monkeypatch.setattr(evidence, "validate_tigris_action_proof", lambda *_a, **_k: None)
    monkeypatch.setattr(
        evidence, "load_json", lambda _path: {"resource_class": "gpu_representation"},
    )
    monkeypatch.setattr(nf, "validate_two_update_acceptance_proof", lambda *_a, **_k: None)
    monkeypatch.setattr(nf, "validate_usr1_exact_resume_proof_v2", lambda *_a, **_k: None)
    monkeypatch.setattr(nf, "validate_nonfinal_acceptance_action_result", lambda *_a, **_k: None)
    refs = {name: {"kind": name} for name in REQUIRED_TIGRIS_CHECKS}
    with pytest.raises(PermissionError, match="different non-final authorities"):
        evidence.build_tigris_evidence_bundle(
            source_commit=source,
            representation_recipe_sha256=recipe,
            resource_profile={"kind": "profile"},
            storage_estimate={"kind": "storage"},
            fixed_size_inventory={"kind": "inventory"},
            action_proofs=refs,
        )


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

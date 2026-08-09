from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import hlt_classification.scouting.hcwdl_representation_locks as locks
from hlt_classification.data.cache_contracts import with_content_hash


H = "a" * 64


def _row(node_id: str, *, teacher: bool) -> dict[str, object]:
    if node_id == "TOFF":
        domain, track = "native_offline", "shared"
    elif teacher:
        domain = "d" + node_id[1:].lower()
        track = "cold" if node_id.endswith("c") else (
            "warm" if node_id.endswith("w") else "shared"
        )
    else:
        domain = "hlt"
        track = "cold" if node_id.endswith("c") else (
            "warm" if node_id.endswith("w") else "shared"
        )
    ordinal = sum(node_id.encode("ascii"))
    return {
        "node_id": node_id,
        "domain": domain,
        "track": track,
        "report_path": f"/authenticated/{node_id}/training_report.json",
        "report_sha256": f"{ordinal % 16:x}" * 64,
        "checkpoint_path": f"/authenticated/{node_id}/selected.pt",
        "checkpoint_sha256": f"{(ordinal + 1) % 16:x}" * 64,
        "checkpoint_byte_sha256": f"{(ordinal + 1) % 16:x}" * 64,
    }


def _fixture(monkeypatch):
    teachers = {
        node: _row(node, teacher=True) for node in locks.IMPORTED_TEACHERS
    }
    controls = {
        node: _row(node, teacher=False) for node in locks.IMPORTED_LOGIT_CONTROLS
    }
    imported = {**teachers, **controls}
    architecture = {
        "content_hash": "1" * 64,
        "checkpoint_audits": [
            {
                "node_id": node,
                "checkpoint_sha256": row["checkpoint_byte_sha256"],
                "report_path": Path(str(row["report_path"])).resolve().as_posix(),
                "report_sha256": row["report_sha256"],
                "checkpoint_path": Path(str(row["checkpoint_path"])).resolve().as_posix(),
                "actual_file_evidence": True,
            }
            for node, row in sorted(imported.items())
        ],
    }
    loss = {
        "content_hash": "2" * 64,
        "parent_artifacts": [
            {
                "node_id": node,
                "training_report_sha256": row["report_sha256"],
                "checkpoint_sha256": row["checkpoint_sha256"],
            }
            for node, row in sorted(imported.items())
        ],
    }
    architecture_calls = []
    monkeypatch.setattr(
        locks,
        "validate_architecture_attestation",
        lambda value, *, require_authorized: architecture_calls.append(
            (value, require_authorized)
        ) or value["content_hash"],
    )
    monkeypatch.setattr(
        locks, "validate_parent_loss_attestation", lambda value: value["content_hash"],
    )
    return teachers, controls, architecture, loss, architecture_calls


def _build(monkeypatch):
    teachers, controls, architecture, loss, calls = _fixture(monkeypatch)
    lineage_names = {
        "source_manifest", "split_manifest", "row_selection",
        "train_assignment_manifest", "validation_assignment_manifest",
        "assignment_lock", "parent_recipe", "endpoint_qualification_lock",
        "parent_graph",
    }
    proof_names = lineage_names | {
        "parent_campaign_spec", "teachers", "logit_controls",
    }
    artifact = locks.build_parent_import(
        parent_campaign_spec_sha256="3" * 64,
        parent_source_commit="4" * 40,
        lineage_hashes={name: H for name in lineage_names},
        architecture_attestation=architecture,
        parent_loss_attestation=loss,
        teachers=teachers,
        logit_controls=controls,
        original_contract_validation={name: True for name in proof_names},
    )
    return artifact, teachers, controls, architecture, loss, calls


def test_parent_import_requires_authorized_architecture_validator_and_exact_evidence(
    monkeypatch,
) -> None:
    artifact, _, _, architecture, _, calls = _build(monkeypatch)
    assert calls == [(architecture, True)]
    assert artifact["parents"]["architecture_attestation"] == "1" * 64
    assert artifact["parents"]["parent_loss_attestation"] == "2" * 64
    assert locks.validate_parent_import(artifact) == artifact["content_hash"]


@pytest.mark.parametrize("evidence", ("architecture", "loss"))
def test_parent_import_rejects_stale_or_incomplete_attestation_lineage(
    monkeypatch, evidence: str,
) -> None:
    teachers, controls, architecture, loss, _ = _fixture(monkeypatch)
    target = architecture["checkpoint_audits"] if evidence == "architecture" else loss["parent_artifacts"]
    target[0] = deepcopy(target[0])
    hash_key = "checkpoint_sha256"
    target[0][hash_key] = "f" * 64
    with pytest.raises(ValueError, match="lineage differs"):
        locks._bind_parent_evidence(
            architecture_attestation=architecture,
            parent_loss_attestation=loss,
            teachers=teachers,
            logit_controls=controls,
        )


def test_parent_import_rejects_non_hlt_logit_control_domain(monkeypatch) -> None:
    _, _, controls, _, _, _ = _build(monkeypatch)
    bad = deepcopy(controls["M0"])
    bad["domain"] = "native_offline"
    with pytest.raises(ValueError, match="logit-control.*domain"):
        locks._import_row(bad, node_id="M0", teacher=False)


def test_parent_import_rejects_supplied_checkpoint_hash_instead_of_actual_byte_proof(
    monkeypatch,
) -> None:
    _, controls, _, _, _ = _fixture(monkeypatch)
    bad = deepcopy(controls["M0"])
    bad["checkpoint_byte_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="selected-checkpoint byte proof"):
        locks._import_row(bad, node_id="M0", teacher=False)


def test_parent_import_reopens_and_binds_fresh_attestations(monkeypatch) -> None:
    artifact, _, _, architecture, loss, _ = _build(monkeypatch)
    assert locks.validate_parent_import_against_evidence(
        artifact,
        architecture_attestation=architecture,
        parent_loss_attestation=loss,
    ) == artifact["content_hash"]

    stale_architecture = deepcopy(architecture)
    stale_architecture["content_hash"] = "9" * 64
    with pytest.raises(ValueError, match="attestation parents differ"):
        locks.validate_parent_import_against_evidence(
            artifact,
            architecture_attestation=stale_architecture,
            parent_loss_attestation=loss,
        )


def test_parent_import_rejects_same_shape_alternate_parent_bundle(monkeypatch) -> None:
    artifact, _, _, architecture, loss, _ = _build(monkeypatch)
    alternate = deepcopy(architecture)
    alternate["checkpoint_audits"][0] = deepcopy(
        alternate["checkpoint_audits"][0]
    )
    alternate["checkpoint_audits"][0]["report_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="lineage differs"):
        locks.validate_parent_import_against_evidence(
            artifact,
            architecture_attestation=alternate,
            parent_loss_attestation=loss,
        )


@pytest.mark.parametrize("mutation", ("empty_proof", "extra_parent", "bad_commit"))
def test_parent_import_validator_recomputes_closed_semantics(
    monkeypatch, mutation: str,
) -> None:
    artifact, *_ = _build(monkeypatch)
    forged = deepcopy(artifact)
    forged.pop("content_hash")
    if mutation == "empty_proof":
        forged["payload"]["original_contract_validation"] = {}
        message = "completion proof"
    elif mutation == "extra_parent":
        forged["parents"]["unused"] = "f" * 64
        message = "parent lineage registry"
    else:
        forged["payload"]["parent_source_commit"] = "not-a-commit"
        message = "source commit"
    forged = with_content_hash(forged)
    with pytest.raises(ValueError, match=message):
        locks.validate_parent_import(forged)

from __future__ import annotations

from pathlib import Path
import json

import pytest

from hlt_classification.data.cache_contracts import load_json, sha256_bytes, with_content_hash
from hlt_classification.scouting.hcwdl_representation_artifacts import (
    publish_binary_envelope,
    validate_binary_envelope,
)
from hlt_classification.scouting.hcwdl_representation_contracts import (
    CONTRACTS,
    KERNEL_RESOURCES_CONTRACT,
    SHARED_BINARY_ENVELOPE_CONTRACT,
    TARGET_MANIFEST_CONTRACT,
    build_versioned_artifact,
    validate_versioned_artifact,
)


H = "a" * 64
G = "b" * 64


def _publish(root: Path, *, failure_hook=None, owner: str = "campaign-a"):
    payload = b"fixed-kernel-resource"
    return publish_binary_envelope(
        root,
        artifact_contract=KERNEL_RESOURCES_CONTRACT,
        producer_task_id="kernel_resources",
        schema={"payload": "bytes", "version": 1},
        immutable_parent_hashes={"recipe": H, "graph": G},
        registered_output_row={"resource_id": "fixed_rff_v1"},
        campaign_or_recovery_owner={"campaign": owner},
        payloads={"payload.npz": payload},
        member_metadata={
            "payload.npz": {
                "logical_sha256": sha256_bytes(payload),
                "dtype": "uint8",
                "shape": [len(payload)],
            }
        },
        sidecar_payload={"resource_id": "fixed_rff_v1"},
        failure_hook=failure_hook,
    )


def test_contract_registry_is_distinct_and_parent_bound():
    assert KERNEL_RESOURCES_CONTRACT in CONTRACTS
    assert TARGET_MANIFEST_CONTRACT in CONTRACTS
    assert SHARED_BINARY_ENVELOPE_CONTRACT in CONTRACTS
    assert len(CONTRACTS) >= 60
    artifact = build_versioned_artifact(
        TARGET_MANIFEST_CONTRACT, parents={"graph": H}, payload={"rows": 3},
    )
    validate_versioned_artifact(
        artifact, expected_contract=TARGET_MANIFEST_CONTRACT,
        expected_parents={"graph": H}, required_payload_keys=("rows",),
    )
    with pytest.raises(ValueError, match="parent lineage"):
        validate_versioned_artifact(
            artifact, expected_contract=TARGET_MANIFEST_CONTRACT,
            expected_parents={"graph": G},
        )
    with pytest.raises(ValueError, match="unknown HCWDL-RKD contract"):
        build_versioned_artifact("HCWDL_ARTIFACT/v1", parents={"graph": H}, payload={})


@pytest.mark.parametrize(
    "boundary",
    (
        "after_member:payload.npz",
        "after_member:sidecar.json",
        "after_commit",
        "before_directory_rename",
        "after_directory_rename",
    ),
)
def test_binary_envelope_crash_is_same_owner_idempotent(tmp_path: Path, boundary: str):
    seen = False

    def fail(name: str):
        nonlocal seen
        if name == boundary and not seen:
            seen = True
            raise RuntimeError(f"injected at {name}")

    with pytest.raises(RuntimeError, match="injected"):
        _publish(tmp_path, failure_hook=fail)
    envelope = _publish(tmp_path)
    assert envelope.directory.is_dir()
    assert envelope.sidecar["payload"]["resource_id"] == "fixed_rff_v1"
    validate_binary_envelope(
        tmp_path, envelope.envelope_id,
        expected_contract=KERNEL_RESOURCES_CONTRACT,
        expected_parents={"graph": G, "recipe": H},
        expected_owner_id=envelope.owner_id,
    )
    assert not any((tmp_path / "committed" / envelope.envelope_id).glob("*.tmp"))


def test_binary_envelope_rejects_foreign_owner_and_committed_corruption(tmp_path: Path):
    def fail(name: str):
        if name == "after_member:payload.npz":
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError):
        _publish(tmp_path, failure_hook=fail, owner="owner-a")
    with pytest.raises(PermissionError, match="another owner"):
        _publish(tmp_path, owner="owner-b")

    complete = _publish(tmp_path, owner="owner-a")
    payload = complete.directory / "payload.npz"
    payload.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="member bytes differ"):
        validate_binary_envelope(
            tmp_path, complete.envelope_id,
            expected_contract=KERNEL_RESOURCES_CONTRACT,
            expected_parents={"graph": G, "recipe": H},
        )
    with pytest.raises(ValueError):
        _publish(tmp_path, owner="owner-a")


def test_committed_envelope_reuse_requires_exact_requested_members_and_sidecar(tmp_path: Path):
    complete = _publish(tmp_path)
    common = {
        "root": tmp_path,
        "artifact_contract": KERNEL_RESOURCES_CONTRACT,
        "producer_task_id": "kernel_resources",
        "schema": {"payload": "bytes", "version": 1},
        "immutable_parent_hashes": {"recipe": H, "graph": G},
        "registered_output_row": {"resource_id": "fixed_rff_v1"},
        "campaign_or_recovery_owner": {"campaign": "campaign-a"},
    }
    changed = b"different-kernel-resource"
    with pytest.raises(FileExistsError, match="request differs"):
        publish_binary_envelope(
            **common,
            payloads={"payload.npz": changed},
            member_metadata={
                "payload.npz": {
                    "logical_sha256": sha256_bytes(changed),
                    "dtype": "uint8",
                    "shape": [len(changed)],
                }
            },
            sidecar_payload={"resource_id": "fixed_rff_v1"},
        )
    original = b"fixed-kernel-resource"
    with pytest.raises(FileExistsError, match="request differs"):
        publish_binary_envelope(
            **common,
            payloads={"payload.npz": original},
            member_metadata={
                "payload.npz": {
                    "logical_sha256": sha256_bytes(original),
                    "dtype": "uint8",
                    "shape": [len(original)],
                }
            },
            sidecar_payload={"resource_id": "changed-after-commit"},
        )
    assert validate_binary_envelope(
        tmp_path, complete.envelope_id,
        expected_contract=KERNEL_RESOURCES_CONTRACT,
        expected_parents={"graph": G, "recipe": H},
    ).sidecar["payload"]["resource_id"] == "fixed_rff_v1"


def test_binary_envelope_rejects_normalized_member_collisions_and_forged_identity(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="collide after normalization"):
        publish_binary_envelope(
            tmp_path,
            artifact_contract=KERNEL_RESOURCES_CONTRACT,
            producer_task_id="kernel_resources",
            schema={"payload": "bytes", "version": 1},
            immutable_parent_hashes={"recipe": H, "graph": G},
            registered_output_row={"resource_id": "fixed_rff_v1"},
            campaign_or_recovery_owner={"campaign": "campaign-a"},
            payloads={"nested\\payload.bin": b"a", "nested/payload.bin": b"b"},
            member_metadata={
                "nested\\payload.bin": {"logical_sha256": H},
                "nested/payload.bin": {"logical_sha256": G},
            },
            sidecar_payload={"resource_id": "fixed_rff_v1"},
        )

    complete = _publish(tmp_path / "forged")
    commit_path = complete.directory / "commit.json"
    commit = load_json(commit_path)
    unsigned = {key: value for key, value in commit.items() if key != "content_hash"}
    unsigned["payload"]["producer_task_id"] = "forged-producer"
    forged = with_content_hash(unsigned)
    commit_path.write_text(
        json.dumps(forged, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity"):
        validate_binary_envelope(
            tmp_path / "forged", complete.envelope_id,
            expected_contract=KERNEL_RESOURCES_CONTRACT,
            expected_parents={"graph": G, "recipe": H},
        )

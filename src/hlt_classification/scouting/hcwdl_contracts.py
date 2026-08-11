"""Versioned HCWDL artifact envelopes and role-access boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    canonical_sha256, sha256_file, validate_content_hash, with_content_hash,
)


CAMPAIGN_NAME: Final = "high_coverage_cold_warm_distillation_ladder"
ARTIFACT_CONTRACT: Final = "HCWDL_ARTIFACT/v1"
SCHEMA_VERSION: Final = 1
DATA_ROLES: Final = ("train", "validation", "final_test")


def require_role_access(
    role: str,
    *,
    branch_read: bool = False,
    fit: bool = False,
    select: bool = False,
    completed_locks: Sequence[str] = (),
    shared_final_capability: Mapping[str, Any] | None = None,
    shared_final_claim: Mapping[str, Any] | None = None,
    shared_final_task_registry: Mapping[str, Any] | None = None,
    final_population_sha256: str | None = None,
    final_task_id: str | None = None,
    final_branch_family: str | None = None,
    final_execution_lock_sha256: str | None = None,
    requested_branches: Sequence[str] | None = None,
    shared_reservation_active: bool = False,
) -> None:
    if role not in DATA_ROLES:
        raise ValueError(f"unknown HCWDL role {role!r}")
    if fit and role != "train":
        raise PermissionError("HCWDL fitting is train-only")
    if select and role != "validation":
        raise PermissionError("HCWDL selection is validation-only")
    if role == "final_test" and branch_read:
        if shared_final_capability is not None:
            if (
                final_population_sha256 is None or final_task_id is None
                or shared_final_claim is None
                or shared_final_task_registry is None
                or final_branch_family is None
                or requested_branches is None
            ):
                raise PermissionError("shared final capability context is incomplete")
            from .hcwdl_shared_final import validate_role_capability
            validate_role_capability(
                shared_final_capability,
                execution_claim=shared_final_claim,
                task_registry=shared_final_task_registry,
                expected_population_sha256=final_population_sha256,
                expected_task_id=final_task_id,
                allowed_kinds=("row_selection", "assignment_shard", "prediction_shard"),
                expected_branch_family=final_branch_family,
                expected_execution_lock_sha256=final_execution_lock_sha256,
            )
            from .hcwdl_final_stream import validate_projected_branches
            validate_projected_branches(
                path=final_branch_family, branches=requested_branches,
            )
            return
        if shared_reservation_active:
            raise PermissionError(
                "legacy HCWDL locks are invalid after shared final reservation"
            )
        required = {"finalist", "execution"}
        if not required.issubset(completed_locks):
            raise PermissionError("HCWDL final-test branches are sealed")


def artifact_envelope(
    *, kind: str, payload: Mapping[str, Any], parents: Mapping[str, str],
) -> dict[str, Any]:
    if not isinstance(kind, str) or not kind:
        raise ValueError("HCWDL artifact kind must be nonempty")
    return with_content_hash({
        "contract": ARTIFACT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN_NAME,
        "kind": kind,
        "parents": dict(sorted(parents.items())),
        "payload": dict(payload),
    })


def authenticate_source_files(
    data_root: str | Path, records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Rehash every declared source byte-for-byte beneath the read-only root."""

    root = Path(data_root).expanduser().resolve()
    if not root.is_dir() or not records:
        raise FileNotFoundError("HCWDL source root or inventory is absent")
    rows = []
    seen: set[str] = set()
    for record in records:
        relative = str(record.get("path", "")).replace("\\", "/")
        expected = str(record.get("sha256", ""))
        if not relative or relative in seen or len(expected) != 64:
            raise ValueError("HCWDL source inventory record differs")
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as error:
            raise ValueError("HCWDL source inventory escapes its data root") from error
        actual = sha256_file(source)
        if actual != expected:
            raise ValueError(f"HCWDL source bytes differ for {relative!r}")
        seen.add(relative); rows.append({"path": relative, "sha256": actual})
    return {
        "files": len(rows),
        "inventory_sha256": canonical_sha256(rows),
        "all_source_bytes_reauthenticated": True,
    }


def validate_artifact(
    value: Mapping[str, Any], *, expected_kind: str,
    expected_parents: Mapping[str, str],
) -> str:
    digest = validate_content_hash(
        value, expected_contract=ARTIFACT_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    if value.get("campaign") != CAMPAIGN_NAME or value.get("kind") != expected_kind:
        raise ValueError("HCWDL artifact campaign or kind differs")
    if value.get("parents") != dict(sorted(expected_parents.items())):
        raise ValueError("HCWDL artifact parent lineage differs")
    return digest


__all__ = [
    "ARTIFACT_CONTRACT", "CAMPAIGN_NAME", "DATA_ROLES", "SCHEMA_VERSION",
    "artifact_envelope", "authenticate_source_files", "require_role_access",
    "validate_artifact",
]

"""Explicit human authorization artifact for future HCWDL live submission."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import subprocess
from typing import Any, Final

from hlt_classification.data.cache_contracts import require_sha256, validate_content_hash, with_content_hash


LEGACY_SUBMISSION_AUTHORIZATION_CONTRACT: Final = "HCWDL_SUBMISSION_AUTHORIZATION/v3"
PREVIOUS_SUBMISSION_AUTHORIZATION_CONTRACT: Final = "HCWDL_SUBMISSION_AUTHORIZATION/v4"
PRIOR_SUBMISSION_AUTHORIZATION_CONTRACT: Final = "HCWDL_SUBMISSION_AUTHORIZATION/v5"
SUBMISSION_AUTHORIZATION_CONTRACT: Final = "HCWDL_SUBMISSION_AUTHORIZATION/v6"
AUTHORIZATION_PHRASE: Final = "AUTHORIZE EXACT HCWDL SPEC FOR TIGRIS"
LEGACY_MODES: Final = frozenset({"smoke", "pilot", "production"})
PREVIOUS_MODES: Final = frozenset({
    "smoke", "pilot", "midscale500k", "production",
})
PRIOR_MODES: Final = frozenset({
    "smoke", "pilot", "midscale500k", "midscale1m", "production",
})
MODES: Final = frozenset({
    "smoke", "pilot", "midscale500k", "midscale1m", "midscale2m",
    "production",
})


def require_canonical_campaign_spec_path(
    supplied_path: str | Path, *, campaign_root: str | Path,
) -> Path:
    supplied = Path(supplied_path).resolve()
    expected = (Path(campaign_root) / "campaign_spec.json").resolve()
    if supplied != expected:
        raise PermissionError(
            f"HCWDL execution spec must be the canonical campaign path {expected}"
        )
    return supplied


def validate_source_checkout(
    repository: str | Path, *, expected_commit: str,
) -> None:
    """Require the exact clean commit and a locally known origin ref containing it."""
    root = Path(repository).resolve()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    remote = subprocess.run(
        ["git", "branch", "-r", "--contains", expected_commit], cwd=root,
        check=True, capture_output=True, text=True,
    ).stdout
    if head != expected_commit or dirty or "origin/" not in remote:
        raise PermissionError("HCWDL source is not the exact clean locally known pushed commit")


def build_submission_authorization(
    *, mode: str, source_commit: str, source_manifest_sha256: str,
    split_manifest_sha256: str, recipe_sha256: str,
    resource_request_sha256: str, command_plan_sha256: str,
    authorization_phrase: str,
    production_authorization_sha256: str | None = None,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError("HCWDL authorization mode differs")
    if authorization_phrase != AUTHORIZATION_PHRASE:
        raise PermissionError("HCWDL submission authorization phrase differs")
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("HCWDL authorization source commit differs")
    if mode == "production":
        require_sha256(production_authorization_sha256, name="production authorization SHA-256")
    elif production_authorization_sha256 is not None:
        raise ValueError("nonproduction HCWDL authorization names a production decision")
    return with_content_hash({
        "contract": SUBMISSION_AUTHORIZATION_CONTRACT, "schema_version": 6,
        "mode": mode, "source_commit": source_commit,
        "source_manifest_sha256": require_sha256(source_manifest_sha256, name="source SHA-256"),
        "split_manifest_sha256": require_sha256(split_manifest_sha256, name="split SHA-256"),
        "recipe_sha256": require_sha256(recipe_sha256, name="recipe SHA-256"),
        "resource_request_sha256": require_sha256(
            resource_request_sha256, name="resource request SHA-256",
        ),
        "command_plan_sha256": require_sha256(
            command_plan_sha256, name="exact command-plan SHA-256",
        ),
        "production_authorization_sha256": production_authorization_sha256,
        "explicit_user_authorization": True,
    })


def validate_submission_authorization(
    value: Mapping[str, Any], *, mode: str, source_commit: str,
    source_manifest_sha256: str, split_manifest_sha256: str,
    recipe_sha256: str, resource_request_sha256: str, command_plan_sha256: str,
    production_authorization_sha256: str | None,
) -> str:
    contract = value.get("contract")
    if contract == LEGACY_SUBMISSION_AUTHORIZATION_CONTRACT:
        schema_version = 3
        allowed_modes = LEGACY_MODES
    elif contract == PREVIOUS_SUBMISSION_AUTHORIZATION_CONTRACT:
        schema_version = 4
        allowed_modes = PREVIOUS_MODES
    elif contract == PRIOR_SUBMISSION_AUTHORIZATION_CONTRACT:
        schema_version = 5
        allowed_modes = PRIOR_MODES
    elif contract == SUBMISSION_AUTHORIZATION_CONTRACT:
        schema_version = 6
        allowed_modes = MODES
    else:
        raise ValueError("HCWDL submission authorization contract differs")
    digest = validate_content_hash(
        value, expected_contract=str(contract), expected_schema_version=schema_version,
    )
    if not isinstance(value.get("mode"), str) or value.get("mode") not in allowed_modes:
        raise ValueError("HCWDL submission authorization mode differs")
    expected = {
        "mode": mode, "source_commit": source_commit,
        "source_manifest_sha256": source_manifest_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "recipe_sha256": recipe_sha256,
        "resource_request_sha256": resource_request_sha256,
        "command_plan_sha256": command_plan_sha256,
        "production_authorization_sha256": production_authorization_sha256,
        "explicit_user_authorization": True,
    }
    if any(value.get(name) != item for name, item in expected.items()):
        raise PermissionError("HCWDL submission authorization lineage differs")
    return digest


__all__ = [
    "AUTHORIZATION_PHRASE", "LEGACY_SUBMISSION_AUTHORIZATION_CONTRACT",
    "PREVIOUS_SUBMISSION_AUTHORIZATION_CONTRACT",
    "PRIOR_SUBMISSION_AUTHORIZATION_CONTRACT", "SUBMISSION_AUTHORIZATION_CONTRACT",
    "build_submission_authorization", "validate_submission_authorization",
    "require_canonical_campaign_spec_path", "validate_source_checkout",
]

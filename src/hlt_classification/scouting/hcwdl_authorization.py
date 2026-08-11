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
RECENT_SUBMISSION_AUTHORIZATION_CONTRACT: Final = "HCWDL_SUBMISSION_AUTHORIZATION/v6"
SUBMISSION_AUTHORIZATION_CONTRACT: Final = "HCWDL_SUBMISSION_AUTHORIZATION/v7"
PARENT_PREFIX_SUBMISSION_AUTHORIZATION_CONTRACT: Final = (
    "HCWDL_SUBMISSION_AUTHORIZATION/v8"
)
AUTHORIZATION_PHRASE: Final = "AUTHORIZE EXACT HCWDL SPEC FOR TIGRIS"
PARENT_PREFIX_AUTHORIZATION_PHRASE: Final = (
    "AUTHORIZE EXACT HCWDL PARENT PREFIX FOR TIGRIS"
)
FULL_SUBMISSION_PHRASE: Final = "SUBMIT HCWDL EXACT SPEC"
FULL_AUTOMATIC_SUBMISSION_PHRASE: Final = (
    "SUBMIT HCWDL EXACT SPEC WITH PREAUTHORIZED ENDPOINT CONTINUATION"
)
PARENT_PREFIX_SUBMISSION_PHRASE: Final = "SUBMIT HCWDL EXACT PARENT PREFIX"
PARENT_PREFIX_AUTOMATIC_SUBMISSION_PHRASE: Final = (
    "SUBMIT HCWDL EXACT PARENT PREFIX WITH PREAUTHORIZED ENDPOINT CONTINUATION"
)
FULL_CONTINUATION_PHRASE: Final = "CONTINUE HCWDL AFTER ENDPOINT ACK"
PARENT_PREFIX_CONTINUATION_PHRASE: Final = (
    "CONTINUE HCWDL PARENT PREFIX AFTER ENDPOINT ACK"
)
FULL_RESUME_PHRASE: Final = "RESUME HCWDL EXACT TASKS"
PARENT_PREFIX_RESUME_PHRASE: Final = "RESUME HCWDL EXACT PARENT PREFIX TASKS"
FULL_CAMPAIGN_SCOPE: Final = "full_campaign"
PARENT_PREFIX_SCOPE: Final = "parent_prefix_through_finalist_lock"
EXECUTION_SCOPES: Final = frozenset({FULL_CAMPAIGN_SCOPE, PARENT_PREFIX_SCOPE})
MANUAL_ENDPOINT_CONTINUATION: Final = "manual_posthoc"
AUTOMATIC_ENDPOINT_CONTINUATION: Final = "preauthorized_automatic"
ENDPOINT_CONTINUATION_MODES: Final = frozenset({
    MANUAL_ENDPOINT_CONTINUATION, AUTOMATIC_ENDPOINT_CONTINUATION,
})
PARENT_PREFIX_AUTHORIZATION_FIELDS: Final = frozenset({
    "execution_scope", "terminal_task_id", "execution_lock_authorized",
    "final_test_access_authorized",
})
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


def live_submission_phrase(*, execution_scope: str, endpoint_continuation: str) -> str:
    if execution_scope not in EXECUTION_SCOPES:
        raise ValueError("HCWDL submission execution scope differs")
    if endpoint_continuation not in ENDPOINT_CONTINUATION_MODES:
        raise ValueError("HCWDL submission endpoint-continuation mode differs")
    if execution_scope == PARENT_PREFIX_SCOPE:
        return (
            PARENT_PREFIX_AUTOMATIC_SUBMISSION_PHRASE
            if endpoint_continuation == AUTOMATIC_ENDPOINT_CONTINUATION
            else PARENT_PREFIX_SUBMISSION_PHRASE
        )
    return (
        FULL_AUTOMATIC_SUBMISSION_PHRASE
        if endpoint_continuation == AUTOMATIC_ENDPOINT_CONTINUATION
        else FULL_SUBMISSION_PHRASE
    )


def continuation_phrase(*, execution_scope: str) -> str:
    if execution_scope not in EXECUTION_SCOPES:
        raise ValueError("HCWDL continuation execution scope differs")
    return (
        PARENT_PREFIX_CONTINUATION_PHRASE
        if execution_scope == PARENT_PREFIX_SCOPE else FULL_CONTINUATION_PHRASE
    )


def resume_phrase(*, execution_scope: str) -> str:
    if execution_scope not in EXECUTION_SCOPES:
        raise ValueError("HCWDL resume execution scope differs")
    return (
        PARENT_PREFIX_RESUME_PHRASE
        if execution_scope == PARENT_PREFIX_SCOPE else FULL_RESUME_PHRASE
    )


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
    endpoint_continuation: str = MANUAL_ENDPOINT_CONTINUATION,
    execution_scope: str = FULL_CAMPAIGN_SCOPE,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError("HCWDL authorization mode differs")
    if execution_scope not in EXECUTION_SCOPES:
        raise ValueError("HCWDL authorization execution scope differs")
    expected_phrase = (
        PARENT_PREFIX_AUTHORIZATION_PHRASE
        if execution_scope == PARENT_PREFIX_SCOPE else AUTHORIZATION_PHRASE
    )
    if authorization_phrase != expected_phrase:
        raise PermissionError("HCWDL submission authorization phrase differs")
    if execution_scope == PARENT_PREFIX_SCOPE and mode == "smoke":
        raise ValueError("HCWDL parent prefix must use a non-smoke 60-pass mode")
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("HCWDL authorization source commit differs")
    if endpoint_continuation not in ENDPOINT_CONTINUATION_MODES:
        raise ValueError("HCWDL endpoint-continuation authorization differs")
    if mode == "production":
        require_sha256(production_authorization_sha256, name="production authorization SHA-256")
    elif production_authorization_sha256 is not None:
        raise ValueError("nonproduction HCWDL authorization names a production decision")
    payload = {
        "contract": (
            PARENT_PREFIX_SUBMISSION_AUTHORIZATION_CONTRACT
            if execution_scope == PARENT_PREFIX_SCOPE
            else SUBMISSION_AUTHORIZATION_CONTRACT
        ),
        "schema_version": 8 if execution_scope == PARENT_PREFIX_SCOPE else 7,
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
        "endpoint_continuation": endpoint_continuation,
        "endpoint_diagnostic_review_waived_before_execution": (
            endpoint_continuation == AUTOMATIC_ENDPOINT_CONTINUATION
        ),
        "explicit_user_authorization": True,
    }
    if execution_scope == PARENT_PREFIX_SCOPE:
        payload.update({
            "execution_scope": PARENT_PREFIX_SCOPE,
            "terminal_task_id": "finalist_lock",
            "execution_lock_authorized": False,
            "final_test_access_authorized": False,
        })
    return with_content_hash(payload)


def validate_submission_authorization(
    value: Mapping[str, Any], *, mode: str, source_commit: str,
    source_manifest_sha256: str, split_manifest_sha256: str,
    recipe_sha256: str, resource_request_sha256: str, command_plan_sha256: str,
    production_authorization_sha256: str | None,
    endpoint_continuation: str = MANUAL_ENDPOINT_CONTINUATION,
    execution_scope: str = FULL_CAMPAIGN_SCOPE,
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
    elif contract == RECENT_SUBMISSION_AUTHORIZATION_CONTRACT:
        schema_version = 6
        allowed_modes = MODES
    elif contract == SUBMISSION_AUTHORIZATION_CONTRACT:
        schema_version = 7
        allowed_modes = MODES
    elif contract == PARENT_PREFIX_SUBMISSION_AUTHORIZATION_CONTRACT:
        schema_version = 8
        allowed_modes = MODES - {"smoke"}
    else:
        raise ValueError("HCWDL submission authorization contract differs")
    digest = validate_content_hash(
        value, expected_contract=str(contract), expected_schema_version=schema_version,
    )
    if not isinstance(value.get("mode"), str) or value.get("mode") not in allowed_modes:
        raise ValueError("HCWDL submission authorization mode differs")
    if execution_scope not in EXECUTION_SCOPES:
        raise ValueError("HCWDL authorization execution scope differs")
    if schema_version == 8 and execution_scope != PARENT_PREFIX_SCOPE:
        raise PermissionError("HCWDL v8 authorization is parent-prefix-only")
    if schema_version < 8 and execution_scope != FULL_CAMPAIGN_SCOPE:
        raise PermissionError("legacy HCWDL authorization cannot authorize a parent prefix")
    if schema_version < 8 and PARENT_PREFIX_AUTHORIZATION_FIELDS & set(value):
        raise ValueError(
            "legacy HCWDL authorization contains parent-prefix-only fields"
        )
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
    if schema_version >= 7:
        if endpoint_continuation not in ENDPOINT_CONTINUATION_MODES:
            raise ValueError("HCWDL endpoint-continuation authorization differs")
        expected.update({
            "endpoint_continuation": endpoint_continuation,
            "endpoint_diagnostic_review_waived_before_execution": (
                endpoint_continuation == AUTOMATIC_ENDPOINT_CONTINUATION
            ),
        })
    elif endpoint_continuation != MANUAL_ENDPOINT_CONTINUATION:
        raise PermissionError(
            "legacy HCWDL authorization cannot preauthorize endpoint continuation"
        )
    if schema_version == 8:
        expected.update({
            "execution_scope": PARENT_PREFIX_SCOPE,
            "terminal_task_id": "finalist_lock",
            "execution_lock_authorized": False,
            "final_test_access_authorized": False,
        })
    if any(value.get(name) != item for name, item in expected.items()):
        raise PermissionError("HCWDL submission authorization lineage differs")
    return digest


__all__ = [
    "AUTHORIZATION_PHRASE", "AUTOMATIC_ENDPOINT_CONTINUATION",
    "FULL_AUTOMATIC_SUBMISSION_PHRASE", "FULL_CONTINUATION_PHRASE",
    "FULL_RESUME_PHRASE", "FULL_SUBMISSION_PHRASE",
    "ENDPOINT_CONTINUATION_MODES", "EXECUTION_SCOPES", "FULL_CAMPAIGN_SCOPE",
    "MANUAL_ENDPOINT_CONTINUATION", "PARENT_PREFIX_AUTHORIZATION_PHRASE",
    "PARENT_PREFIX_AUTOMATIC_SUBMISSION_PHRASE",
    "PARENT_PREFIX_CONTINUATION_PHRASE", "PARENT_PREFIX_RESUME_PHRASE",
    "PARENT_PREFIX_SUBMISSION_PHRASE",
    "PARENT_PREFIX_SCOPE", "PARENT_PREFIX_SUBMISSION_AUTHORIZATION_CONTRACT",
    "LEGACY_SUBMISSION_AUTHORIZATION_CONTRACT",
    "PREVIOUS_SUBMISSION_AUTHORIZATION_CONTRACT",
    "PRIOR_SUBMISSION_AUTHORIZATION_CONTRACT",
    "RECENT_SUBMISSION_AUTHORIZATION_CONTRACT", "SUBMISSION_AUTHORIZATION_CONTRACT",
    "build_submission_authorization", "continuation_phrase",
    "live_submission_phrase", "resume_phrase", "validate_submission_authorization",
    "require_canonical_campaign_spec_path", "validate_source_checkout",
]

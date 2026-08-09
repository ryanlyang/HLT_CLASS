"""Two-phase cleanup and generation-aware recovery for RKD target banks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    load_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
)

from .hcwdl_representation_artifacts import (
    FailureHook,
    fsync_directory,
    write_staged_immutable_json,
)
from .hcwdl_representation_contracts import (
    TARGET_CLEANUP_AUTHORIZATION_CONTRACT,
    TARGET_CLEANUP_COMPLETION_CONTRACT,
    TARGET_BUILD_INTENT_CONTRACT,
    TARGET_EXECUTION_ATTESTATION_CONTRACT,
    TARGET_MANIFEST_CONTRACT,
    TARGET_RECOVERY_PLAN_CONTRACT,
    CONFIRMATION_REGISTRY_CONTRACT,
    TARGET_CONSUMER_REGISTRY_CONTRACT,
    build_versioned_artifact,
    validate_versioned_artifact,
)
from .hcwdl_representation_targets import (
    build_target_consumer_row,
    build_target_consumer_registry,
    expected_screen_consumer_nodes,
    validate_logical_target_bank,
    validate_target_consumer_registry,
    validate_target_forward_spec,
    validate_target_generation,
)
from .hcwdl_representation_reporting import CONFIRMATION_SEEDS


class TargetRecoveryState(str, Enum):
    STAGING_BUILD_IN_PROGRESS = "STAGING_BUILD_IN_PROGRESS"
    COMPLETE_SHARDS_NO_AUTH = "COMPLETE_SHARDS_NO_AUTH"
    CLEANUP_AUTHORIZED_IN_PROGRESS = "CLEANUP_AUTHORIZED_IN_PROGRESS"
    CLEANUP_COMPLETED = "CLEANUP_COMPLETED"
    INCOMPLETE_NO_AUTHORIZATION = "INCOMPLETE_NO_AUTHORIZATION"
    ABANDONED_AUTHORIZED = "ABANDONED_AUTHORIZED"


RECOVERY_STEP_ORDER: Final = (
    "finish_old_cleanup",
    "rebuild_new_generation",
    "run_unfinished_consumers",
    "authorize_new_cleanup",
    "complete_new_cleanup",
)


def _cleanup_directory(cleanup_root: str | Path, bank_id: str, generation_id: str) -> Path:
    if not bank_id or "/" in bank_id or "\\" in bank_id:
        raise ValueError("HCWDL-RKD cleanup bank ID differs")
    identity = require_sha256(generation_id, name="target cleanup generation ID")
    return Path(cleanup_root) / bank_id / identity


def _load_generation_metadata(bank_root: Path, generation_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    directory = bank_root / "generations" / generation_id
    manifest = load_json(directory / "manifest.json")
    validate_versioned_artifact(
        manifest, expected_contract=TARGET_MANIFEST_CONTRACT,
        required_payload_keys=(
            "generation_id", "logical_bank_id", "logical_target_sha256",
            "authorized_consumers", "shards",
        ),
    )
    if manifest["payload"]["generation_id"] != generation_id:
        raise ValueError("HCWDL-RKD cleanup manifest generation differs")
    forward_spec = load_json(directory / "target_forward_spec.json")
    validate_target_forward_spec(forward_spec)
    attestation = load_json(directory / "target_execution_attestation.json")
    validate_versioned_artifact(
        attestation, expected_contract=TARGET_EXECUTION_ATTESTATION_CONTRACT,
        required_payload_keys=("logical_target_sha256",),
    )
    if attestation["payload"]["logical_target_sha256"] != manifest["payload"][
        "logical_target_sha256"
    ]:
        raise ValueError("HCWDL-RKD cleanup target attestation differs")
    return manifest, forward_spec, attestation


def validate_target_consumer_training_report(
    report: Mapping[str, Any],
    *, expected_consumer: Mapping[str, Any], manifest: Mapping[str, Any],
) -> str:
    """Authenticate the full engine report and its physical-bank binding."""

    # Import lazily because the training module consumes target-bank types.
    # Cleanup must use the canonical terminal-report validator rather than a
    # permissive second implementation that could accept a report training
    # itself would reject.
    from .hcwdl_representation_training import (
        validate_representation_training_report,
    )

    digest = validate_representation_training_report(
        report, expected_execution_id=str(expected_consumer["node_id"]),
    )
    required = {
        "node_id", "execution_id", "registered_execution_id", "replicate_seed",
        "complete", "scientific_complete", "mode", "student_domain",
        "strategy", "track", "completed_optimizer_updates",
        "completed_natural_population_passes", "validation_every_complete_pass",
        "validation_history", "validation", "selected_checkpoint_id",
        "selected_training_checkpoint_path", "selected_training_checkpoint_sha256",
        "selection_sha256", "deployable_extraction", "resume_audit",
        "target_generation_sha256", "target_logical_sha256",
        "predecessor_model_released_before_optimization",
        "finite_poor_results_retained", "performance_early_stopping",
    }
    if not required <= set(report):
        raise ValueError("HCWDL-RKD target consumer report is structurally incomplete")
    registered_execution_id = require_sha256(
        expected_consumer.get("execution_id"), name="target consumer execution ID",
    )
    manifest_payload = manifest["payload"]
    if (
        report["registered_execution_id"] != registered_execution_id
        or report["execution_id"] != expected_consumer["node_id"]
        or report["node_id"] != expected_consumer["node_id"]
        or report["replicate_seed"] != expected_consumer["seed"]
        or report["complete"] is not True
        or report["student_domain"] != "hlt"
        or report["strategy"] != expected_consumer["strategy"]
        or report["track"] != expected_consumer["track"]
        or report["recipe_sha256"]
        != expected_consumer["execution_identity_payload"]["recipe"]
        or report["target_generation_sha256"]
        != manifest["parents"]["target_generation"]
        or report["target_logical_sha256"] != manifest_payload["logical_target_sha256"]
        or report.get("target_manifest_sha256") != manifest["content_hash"]
        or report["finite_poor_results_retained"] is not True
        or report["performance_early_stopping"] is not False
        or report["predecessor_model_released_before_optimization"] is not True
    ):
        raise ValueError("HCWDL-RKD target consumer report lineage/completion differs")
    mode = report["mode"]
    passes = report["completed_natural_population_passes"]
    updates = report["completed_optimizer_updates"]
    history = report["validation_history"]
    if (
        mode not in {"scientific", "smoke"}
        or isinstance(passes, bool) or not isinstance(passes, int) or passes < 0
        or isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0
        or not isinstance(history, list) or not history
    ):
        raise ValueError("HCWDL-RKD target consumer completion counters differ")
    if mode == "scientific" and (
        report["scientific_complete"] is not True
        or passes != 60
        or report["validation_every_complete_pass"] is not True
        or len(history) != 60
    ):
        raise ValueError("HCWDL-RKD scientific target consumer is incomplete")
    if mode == "smoke" and report["scientific_complete"] is not False:
        raise ValueError("HCWDL-RKD smoke report misstates scientific completion")

    if not isinstance(report["selected_training_checkpoint_path"], str) or not report[
        "selected_training_checkpoint_path"
    ]:
        raise ValueError("HCWDL-RKD target consumer checkpoint path differs")
    for name in ("selected_training_checkpoint_sha256", "selection_sha256"):
        require_sha256(report[name], name=f"target consumer {name}")
    extraction = report["deployable_extraction"]
    if not isinstance(extraction, Mapping) or extraction.get("strict_hlt_only") is not True:
        raise ValueError("HCWDL-RKD target consumer deployable extraction differs")
    for name in ("checkpoint_sha256", "report_sha256"):
        require_sha256(extraction.get(name), name=f"target consumer extraction {name}")
    if not isinstance(report["resume_audit"], Mapping):
        raise ValueError("HCWDL-RKD target consumer resume audit differs")
    return digest


def _retained_records(generation_directory: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(generation_directory.rglob("*.json")):
        value = load_json(path)
        content_hash = require_sha256(
            value.get("content_hash"), name=f"retained JSON content hash {path}",
        )
        records.append({
            "path": path.relative_to(generation_directory).as_posix(),
            "byte_sha256": sha256_file(path),
            "content_hash": content_hash,
        })
    return records


def _removable_records(
    generation_directory: Path, manifest: Mapping[str, Any],
    *, require_payloads: bool = True,
) -> list[dict[str, Any]]:
    records = []
    for shard in manifest["payload"]["shards"]:
        relative = str(shard["payload_path"])
        path = generation_directory / relative
        if path.exists():
            if not path.is_file() or sha256_file(path) != shard["payload_byte_sha256"]:
                raise ValueError("HCWDL-RKD cleanup target payload is absent or corrupt")
        elif require_payloads:
            raise ValueError("HCWDL-RKD cleanup target payload is absent or corrupt")
        sidecar = load_json(generation_directory / shard["sidecar_path"])
        logical_hash = canonical_sha256(sidecar["payload"]["array_sha256"])
        records.append({
            "path": relative,
            "byte_sha256": shard["payload_byte_sha256"],
            "logical_sha256": logical_hash,
            "bytes": int(shard["payload_bytes"]),
        })
    return sorted(records, key=lambda row: row["path"])


def authorize_target_cleanup(
    bank_root: str | Path,
    cleanup_root: str | Path,
    *,
    generation_id: str,
    consumer_reports: Mapping[str, Mapping[str, Any]],
    exact_reconstruction_authorized: bool,
) -> dict[str, Any]:
    """Validate every consumer, then publish durable intent before deletion."""

    if not isinstance(exact_reconstruction_authorized, bool):
        raise TypeError("HCWDL-RKD cleanup reconstruction authorization must be boolean")
    root = Path(bank_root)
    manifest = validate_target_generation(root, generation_id)
    payload = manifest["payload"]
    expected_consumers = {
        str(row["execution_id"]): row for row in payload["authorized_consumers"]
    }
    if set(consumer_reports) != set(expected_consumers):
        raise ValueError("HCWDL-RKD cleanup consumer-report registry is incomplete")
    report_hashes = {}
    for execution_id, report in consumer_reports.items():
        report_hashes[execution_id] = validate_target_consumer_training_report(
            report,
            expected_consumer=expected_consumers[execution_id],
            manifest=manifest,
        )
    directory = root / "generations" / generation_id
    _, forward_spec, attestation = _load_generation_metadata(root, generation_id)
    removable = _removable_records(directory, manifest)
    retained = _retained_records(directory)
    parents = {
        "target_manifest": manifest["content_hash"],
        "target_forward_spec": forward_spec["content_hash"],
        "target_execution_attestation": attestation["content_hash"],
        **{f"consumer:{key}": value for key, value in sorted(report_hashes.items())},
    }
    authorization = build_versioned_artifact(
        TARGET_CLEANUP_AUTHORIZATION_CONTRACT, parents=parents,
        payload={
            "authorization_kind": "last_consumer_cleanup",
            "logical_bank_id": payload["logical_bank_id"],
            "generation_id": generation_id,
            "target_manifest_sha256": manifest["content_hash"],
            "target_forward_spec_sha256": forward_spec["content_hash"],
            "target_execution_attestation_sha256": attestation["content_hash"],
            "logical_target_sha256": payload["logical_target_sha256"],
            "consumer_report_sha256": dict(sorted(report_hashes.items())),
            "removable": removable,
            "retained": retained,
            "removable_bytes": sum(record["bytes"] for record in removable),
            "exact_reconstruction_authorized": exact_reconstruction_authorized,
        },
    )
    cleanup = _cleanup_directory(
        cleanup_root, payload["logical_bank_id"], generation_id,
    )
    write_staged_immutable_json(cleanup / "authorization.json", authorization)
    return authorization


def build_abandoned_purge_authorization(
    bank_root: str | Path,
    cleanup_root: str | Path,
    *, generation_id: str, abandonment_parent_sha256: str,
) -> dict[str, Any]:
    """Authorize only an explicitly abandoned generation, never a consumer cleanup."""

    root = Path(bank_root)
    manifest = validate_target_generation(root, generation_id)
    payload = manifest["payload"]
    directory = root / "generations" / generation_id
    removable = _removable_records(directory, manifest)
    retained = _retained_records(directory)
    authorization = build_versioned_artifact(
        TARGET_CLEANUP_AUTHORIZATION_CONTRACT,
        parents={
            "target_manifest": manifest["content_hash"],
            "abandonment_authorization": require_sha256(
                abandonment_parent_sha256, name="abandonment authorization SHA-256",
            ),
        },
        payload={
            "authorization_kind": "abandoned_purge",
            "logical_bank_id": payload["logical_bank_id"],
            "generation_id": generation_id,
            "target_manifest_sha256": manifest["content_hash"],
            "logical_target_sha256": payload["logical_target_sha256"],
            "removable": removable,
            "retained": retained,
            "removable_bytes": sum(record["bytes"] for record in removable),
            "exact_reconstruction_authorized": False,
        },
    )
    cleanup = _cleanup_directory(
        cleanup_root, payload["logical_bank_id"], generation_id,
    )
    write_staged_immutable_json(cleanup / "authorization.json", authorization)
    return authorization


def authorize_miniature_target_cleanup(
    bank_root: str | Path, cleanup_root: str | Path, *, generation_id: str,
    cache_bank_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Authorize removal only after the bounded cache verifier RAM-joined it."""

    from .hcwdl_representation_campaign_artifacts import (
        validate_cache_miniature_bank_evidence,
    )

    root = Path(bank_root)
    manifest = validate_target_generation(root, generation_id)
    validate_cache_miniature_bank_evidence(cache_bank_evidence, manifest=manifest)
    payload = manifest["payload"]
    consumers = payload.get("authorized_consumers")
    if (
        not isinstance(consumers, list) or len(consumers) != 1
        or consumers[0].get("strategy") != "CACHE_MINIATURE"
        or consumers[0].get("execution_identity_payload", {}).get("purpose")
        != "miniature"
        or consumers[0].get("execution_identity_payload", {}).get(
            "training_consumer_authorized"
        ) is not False
    ):
        raise PermissionError("miniature cleanup cannot consume a training generation")
    directory = root / "generations" / generation_id
    removable = _removable_records(directory, manifest)
    retained = _retained_records(directory)
    authorization = build_versioned_artifact(
        TARGET_CLEANUP_AUTHORIZATION_CONTRACT,
        parents={
            "target_manifest": manifest["content_hash"],
            "cache_miniature_bank": cache_bank_evidence["content_hash"],
        },
        payload={
            "authorization_kind": "miniature_cache_cleanup",
            "logical_bank_id": payload["logical_bank_id"],
            "generation_id": generation_id,
            "target_manifest_sha256": manifest["content_hash"],
            "logical_target_sha256": payload["logical_target_sha256"],
            "cache_miniature_bank_sha256": cache_bank_evidence["content_hash"],
            "removable": removable,
            "retained": retained,
            "removable_bytes": sum(record["bytes"] for record in removable),
            "exact_reconstruction_authorized": False,
        },
    )
    cleanup = _cleanup_directory(
        cleanup_root, payload["logical_bank_id"], generation_id,
    )
    write_staged_immutable_json(cleanup / "authorization.json", authorization)
    return authorization


def validate_cleanup_authorization(
    value: Mapping[str, Any],
    *, bank_root: str | Path, generation_id: str,
) -> str:
    digest = validate_versioned_artifact(
        value, expected_contract=TARGET_CLEANUP_AUTHORIZATION_CONTRACT,
        required_payload_keys=(
            "authorization_kind", "logical_bank_id", "generation_id",
            "target_manifest_sha256", "logical_target_sha256", "removable",
            "retained", "removable_bytes", "exact_reconstruction_authorized",
        ),
    )
    payload = value["payload"]
    if payload["generation_id"] != generation_id:
        raise ValueError("HCWDL-RKD cleanup authorization generation differs")
    manifest, forward_spec, attestation = _load_generation_metadata(
        Path(bank_root), generation_id,
    )
    if (
        payload["target_manifest_sha256"] != manifest["content_hash"]
        or payload["logical_target_sha256"] != manifest["payload"]["logical_target_sha256"]
        or payload["logical_bank_id"] != manifest["payload"]["logical_bank_id"]
    ):
        raise ValueError("HCWDL-RKD cleanup authorization target lineage differs")
    if payload["authorization_kind"] not in {
        "last_consumer_cleanup", "abandoned_purge", "miniature_cache_cleanup",
    }:
        raise ValueError("HCWDL-RKD cleanup authorization kind differs")
    if not isinstance(payload["exact_reconstruction_authorized"], bool):
        raise ValueError("HCWDL-RKD cleanup reconstruction authorization differs")
    removable = payload["removable"]
    if (
        not isinstance(removable, list)
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"path", "byte_sha256", "logical_sha256", "bytes"}
            or not isinstance(row["path"], str)
            or isinstance(row["bytes"], bool)
            or int(row["bytes"]) < 0
            for row in removable
        )
        or [row["path"] for row in removable] != sorted(row["path"] for row in removable)
        or len({row["path"] for row in removable}) != len(removable)
        or sum(int(row["bytes"]) for row in removable) != payload["removable_bytes"]
    ):
        raise ValueError("HCWDL-RKD cleanup removable registry differs")
    expected_removable = _removable_records(
        Path(bank_root) / "generations" / generation_id,
        manifest, require_payloads=False,
    )
    if removable != expected_removable:
        raise ValueError("HCWDL-RKD cleanup authorization paths differ from manifest")
    directory = Path(bank_root) / "generations" / generation_id
    expected_retained = _retained_records(directory)
    if payload["retained"] != expected_retained:
        raise ValueError("HCWDL-RKD cleanup retained registry differs from generation")
    for record in removable:
        require_sha256(record["byte_sha256"], name="cleanup removable byte SHA-256")
        require_sha256(record["logical_sha256"], name="cleanup removable logical SHA-256")

    if payload["authorization_kind"] == "last_consumer_cleanup":
        expected_consumers = sorted(
            row["execution_id"] for row in manifest["payload"]["authorized_consumers"]
        )
        report_hashes = payload.get("consumer_report_sha256")
        if not isinstance(report_hashes, Mapping) or sorted(report_hashes) != expected_consumers:
            raise ValueError("HCWDL-RKD cleanup consumer-report registry differs")
        expected_parents = {
            "target_manifest": manifest["content_hash"],
            "target_forward_spec": forward_spec["content_hash"],
            "target_execution_attestation": attestation["content_hash"],
            **{f"consumer:{key}": value for key, value in sorted(report_hashes.items())},
        }
        if (
            payload.get("target_forward_spec_sha256") != forward_spec["content_hash"]
            or payload.get("target_execution_attestation_sha256")
            != attestation["content_hash"]
        ):
            raise ValueError("HCWDL-RKD cleanup forward/attestation lineage differs")
        for name, digest_value in report_hashes.items():
            require_sha256(digest_value, name=f"cleanup consumer report {name}")
        if value["parents"] != expected_parents:
            raise ValueError("HCWDL-RKD cleanup consumer/target parents differ")
    elif payload["authorization_kind"] == "miniature_cache_cleanup":
        evidence_hash = payload.get("cache_miniature_bank_sha256")
        consumers = manifest["payload"].get("authorized_consumers")
        if (
            not isinstance(evidence_hash, str)
            or value["parents"] != {
                "target_manifest": manifest["content_hash"],
                "cache_miniature_bank": evidence_hash,
            }
            or payload["exact_reconstruction_authorized"] is not False
            or not isinstance(consumers, list) or len(consumers) != 1
            or consumers[0].get("strategy") != "CACHE_MINIATURE"
        ):
            raise ValueError("HCWDL-RKD miniature cleanup lineage differs")
        require_sha256(evidence_hash, name="cache-miniature bank evidence")
    elif set(value["parents"]) != {"target_manifest", "abandonment_authorization"}:
        raise ValueError("HCWDL-RKD abandoned-purge parents differ")
    elif payload["exact_reconstruction_authorized"] is not False:
        raise ValueError("HCWDL-RKD abandoned purge cannot authorize reconstruction")
    return digest


def _resolve_generation_member(directory: Path, relative: str) -> Path:
    path = (directory / relative).resolve()
    try:
        path.relative_to(directory.resolve())
    except ValueError as error:
        raise ValueError("HCWDL-RKD cleanup path escapes its generation") from error
    return path


def complete_target_cleanup(
    bank_root: str | Path,
    cleanup_root: str | Path,
    *, generation_id: str, failure_hook: FailureHook | None = None,
) -> dict[str, Any]:
    root = Path(bank_root)
    manifest, _, _ = _load_generation_metadata(root, generation_id)
    bank_id = manifest["payload"]["logical_bank_id"]
    cleanup = _cleanup_directory(cleanup_root, bank_id, generation_id)
    authorization = load_json(cleanup / "authorization.json")
    auth_hash = validate_cleanup_authorization(
        authorization, bank_root=root, generation_id=generation_id,
    )
    completion_path = cleanup / "completion.json"
    if completion_path.exists():
        completion = load_json(completion_path)
        validate_cleanup_completion(
            completion, authorization=authorization, bank_root=root,
        )
        return completion
    directory = root / "generations" / generation_id
    removed = []
    already_missing = []
    removal_observations = []
    removed_bytes = 0
    already_missing_bytes = 0
    for index, record in enumerate(authorization["payload"]["removable"]):
        path = _resolve_generation_member(directory, record["path"])
        if path.exists():
            if not path.is_file() or sha256_file(path) != record["byte_sha256"]:
                raise ValueError("HCWDL-RKD cleanup refuses changed target bytes")
            path.unlink()
            fsync_directory(path.parent)
            removed.append(record["path"])
            removed_bytes += int(record["bytes"])
            observed_state = "removed_now"
        else:
            already_missing.append(record["path"])
            already_missing_bytes += int(record["bytes"])
            observed_state = "already_absent_under_authorization"
        removal_observations.append({
            "path": record["path"],
            "observed_state": observed_state,
            "expected_byte_sha256": record["byte_sha256"],
            "expected_logical_sha256": record["logical_sha256"],
            "expected_bytes": int(record["bytes"]),
            "post_delete_absent": not path.exists(),
        })
        if failure_hook is not None:
            failure_hook(f"after_cleanup_deletion:{index}")
    retained_observations = []
    for record in authorization["payload"]["retained"]:
        path = _resolve_generation_member(directory, record["path"])
        if not path.is_file() or sha256_file(path) != record["byte_sha256"]:
            raise ValueError("HCWDL-RKD cleanup retained artifact differs")
        retained_observations.append({
            "path": record["path"],
            "observed_state": "verified_present",
            "byte_sha256": record["byte_sha256"],
            "content_hash": record["content_hash"],
        })
    completion = build_versioned_artifact(
        TARGET_CLEANUP_COMPLETION_CONTRACT,
        parents={"cleanup_authorization": auth_hash},
        payload={
            "logical_bank_id": bank_id,
            "generation_id": generation_id,
            "cleanup_authorization_sha256": auth_hash,
            "removed": sorted(removed),
            "already_missing": sorted(already_missing),
            "removed_or_missing": sorted(removed + already_missing),
            "removed_bytes": removed_bytes,
            "already_missing_bytes": already_missing_bytes,
            "removal_observations": sorted(
                removal_observations, key=lambda row: row["path"],
            ),
            "retained_observations": sorted(
                retained_observations, key=lambda row: row["path"],
            ),
            "retained": [row["path"] for row in authorization["payload"]["retained"]],
            "authorized_removed_bytes": authorization["payload"]["removable_bytes"],
            "all_authorized_paths_absent": True,
        },
    )
    if failure_hook is not None:
        failure_hook("before_cleanup_completion")
    write_staged_immutable_json(completion_path, completion)
    if failure_hook is not None:
        failure_hook("after_cleanup_completion")
    return completion


def validate_cleanup_completion(
    value: Mapping[str, Any],
    *, authorization: Mapping[str, Any], bank_root: str | Path,
) -> str:
    auth_hash = validate_cleanup_authorization(
        authorization, bank_root=bank_root,
        generation_id=authorization["payload"]["generation_id"],
    )
    digest = validate_versioned_artifact(
        value, expected_contract=TARGET_CLEANUP_COMPLETION_CONTRACT,
        expected_parents={"cleanup_authorization": auth_hash},
        required_payload_keys=(
            "logical_bank_id", "generation_id", "cleanup_authorization_sha256", "removed",
            "already_missing", "removed_or_missing", "removed_bytes",
            "already_missing_bytes",
            "removal_observations", "retained_observations",
            "retained", "authorized_removed_bytes", "all_authorized_paths_absent",
        ),
    )
    payload = value["payload"]
    expected_paths = sorted(row["path"] for row in authorization["payload"]["removable"])
    removable_by_path = {
        row["path"]: int(row["bytes"])
        for row in authorization["payload"]["removable"]
    }
    removed = payload["removed"]
    already_missing = payload["already_missing"]
    observed_lists_valid = (
        isinstance(removed, list)
        and isinstance(already_missing, list)
        and removed == sorted(removed)
        and already_missing == sorted(already_missing)
        and len(removed) == len(set(removed))
        and len(already_missing) == len(set(already_missing))
        and not (set(removed) & set(already_missing))
        and sorted(removed + already_missing) == expected_paths
    )
    expected_removal_observations = {
        row["path"]: row for row in payload["removal_observations"]
    } if isinstance(payload["removal_observations"], list) and all(
        isinstance(row, Mapping) and "path" in row
        for row in payload["removal_observations"]
    ) else {}
    if (
        len(payload["removal_observations"]) != len(expected_paths)
        or set(expected_removal_observations) != set(expected_paths)
    ):
        raise ValueError("HCWDL-RKD cleanup removal observations differ")
    for path in expected_paths:
        observation = expected_removal_observations[path]
        record = next(
            row for row in authorization["payload"]["removable"]
            if row["path"] == path
        )
        expected_state = (
            "removed_now" if path in removed
            else "already_absent_under_authorization"
        )
        if observation != {
            "path": path,
            "observed_state": expected_state,
            "expected_byte_sha256": record["byte_sha256"],
            "expected_logical_sha256": record["logical_sha256"],
            "expected_bytes": int(record["bytes"]),
            "post_delete_absent": True,
        }:
            raise ValueError("HCWDL-RKD cleanup removal observation semantics differ")
    expected_retained_observations = [
        {
            "path": row["path"],
            "observed_state": "verified_present",
            "byte_sha256": row["byte_sha256"],
            "content_hash": row["content_hash"],
        }
        for row in authorization["payload"]["retained"]
    ]
    if (
        payload["logical_bank_id"] != authorization["payload"]["logical_bank_id"]
        or payload["removal_observations"] != sorted(
            payload["removal_observations"], key=lambda row: row["path"],
        )
        or payload["generation_id"] != authorization["payload"]["generation_id"]
        or payload["cleanup_authorization_sha256"] != auth_hash
        or payload["removed_or_missing"] != expected_paths
        or not observed_lists_valid
        or payload["removed_bytes"] != sum(removable_by_path[path] for path in removed)
        or payload["already_missing_bytes"]
        != sum(removable_by_path[path] for path in already_missing)
        or payload["removed_bytes"] + payload["already_missing_bytes"]
        != authorization["payload"]["removable_bytes"]
        or payload["retained"] != [
            row["path"] for row in authorization["payload"]["retained"]
        ]
        or payload["retained_observations"] != expected_retained_observations
        or payload["authorized_removed_bytes"]
        != authorization["payload"]["removable_bytes"]
        or payload["all_authorized_paths_absent"] is not True
    ):
        raise ValueError("HCWDL-RKD cleanup completion semantics differ")
    directory = Path(bank_root) / "generations" / payload["generation_id"]
    for relative in expected_paths:
        if _resolve_generation_member(directory, relative).exists():
            raise ValueError("HCWDL-RKD cleanup completion leaves a target payload")
    for record in authorization["payload"]["retained"]:
        path = _resolve_generation_member(directory, record["path"])
        if not path.is_file() or sha256_file(path) != record["byte_sha256"]:
            raise ValueError("HCWDL-RKD cleanup completion retained artifact differs")
    return digest


def inspect_target_recovery_state(
    bank_root: str | Path,
    cleanup_root: str | Path,
    *, generation_id: str, build_owner_id: str | None = None,
) -> TargetRecoveryState:
    root = Path(bank_root)
    identity = require_sha256(generation_id, name="target recovery generation ID")
    committed = root / "generations" / identity
    staging_root = root / "staging" / identity
    if committed.is_dir():
        try:
            manifest, _, _ = _load_generation_metadata(root, identity)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            return TargetRecoveryState.INCOMPLETE_NO_AUTHORIZATION
        cleanup = _cleanup_directory(
            cleanup_root, manifest["payload"]["logical_bank_id"], identity,
        )
        authorization_path = cleanup / "authorization.json"
        completion_path = cleanup / "completion.json"
        if completion_path.exists():
            if not authorization_path.exists():
                return TargetRecoveryState.INCOMPLETE_NO_AUTHORIZATION
            authorization = load_json(authorization_path)
            validate_cleanup_completion(
                load_json(completion_path), authorization=authorization,
                bank_root=root,
            )
            return TargetRecoveryState.CLEANUP_COMPLETED
        if authorization_path.exists():
            authorization = load_json(authorization_path)
            validate_cleanup_authorization(
                authorization, bank_root=root, generation_id=identity,
            )
            if authorization["payload"]["authorization_kind"] == "abandoned_purge":
                return TargetRecoveryState.ABANDONED_AUTHORIZED
            return TargetRecoveryState.CLEANUP_AUTHORIZED_IN_PROGRESS
        try:
            validate_target_generation(root, identity)
            return TargetRecoveryState.COMPLETE_SHARDS_NO_AUTH
        except (FileNotFoundError, KeyError, TypeError, ValueError, FloatingPointError):
            return TargetRecoveryState.INCOMPLETE_NO_AUTHORIZATION
    if staging_root.is_dir():
        owners = [path for path in staging_root.iterdir() if path.is_dir()]
        if build_owner_id is not None:
            expected = require_sha256(build_owner_id, name="target build owner ID")
            owners = [path for path in owners if path.name == expected]
        for owner in owners:
            intent_path = owner / "build_intent.json"
            if intent_path.is_file():
                intent = load_json(intent_path)
                validate_versioned_artifact(
                    intent, expected_contract=TARGET_BUILD_INTENT_CONTRACT,
                    required_payload_keys=("generation_id", "build_owner_id"),
                )
                if (
                    intent["payload"]["generation_id"] == identity
                    and intent["payload"]["build_owner_id"] == owner.name
                ):
                    return TargetRecoveryState.STAGING_BUILD_IN_PROGRESS
    return TargetRecoveryState.INCOMPLETE_NO_AUTHORIZATION


def build_target_recovery_plan(
    *, bank_root: str | Path, logical_bank: Mapping[str, Any],
    prior_manifest: Mapping[str, Any],
    prior_cleanup_authorization: Mapping[str, Any],
    prior_cleanup_completion: Mapping[str, Any],
    prior_execution_attestation: Mapping[str, Any],
    recovery_consumers: Sequence[Mapping[str, Any]],
    consumer_authorization: Mapping[str, Any],
    recovery_owner: Mapping[str, Any],
) -> dict[str, Any]:
    prior_manifest_hash = validate_versioned_artifact(
        prior_manifest, expected_contract=TARGET_MANIFEST_CONTRACT,
        required_payload_keys=(
            "generation_id", "logical_bank_id", "logical_target_sha256",
            "authorized_consumers",
        ),
    )
    logical_hash = validate_logical_target_bank(logical_bank)
    if prior_manifest["parents"].get("logical_bank") != logical_hash:
        raise ValueError("HCWDL-RKD recovery logical-bank lineage differs")
    authorization_hash = validate_cleanup_authorization(
        prior_cleanup_authorization,
        bank_root=bank_root,
        generation_id=prior_manifest["payload"]["generation_id"],
    )
    if prior_cleanup_authorization["payload"].get(
        "exact_reconstruction_authorized"
    ) is not True:
        raise PermissionError("HCWDL-RKD cleanup did not authorize exact reconstruction")
    completion_hash = validate_cleanup_completion(
        prior_cleanup_completion,
        authorization=prior_cleanup_authorization,
        bank_root=bank_root,
    )
    attestation_hash = validate_versioned_artifact(
        prior_execution_attestation,
        expected_contract=TARGET_EXECUTION_ATTESTATION_CONTRACT,
        required_payload_keys=("logical_target_sha256", "sentinel_hashes"),
    )
    if prior_execution_attestation["payload"]["logical_target_sha256"] != prior_manifest[
        "payload"
    ]["logical_target_sha256"]:
        raise ValueError("HCWDL-RKD recovery prior target lineage differs")
    if (
        prior_manifest["parents"].get("target_execution_attestation")
        != attestation_hash
        or prior_cleanup_authorization["payload"].get(
            "target_execution_attestation_sha256"
        ) != attestation_hash
        or prior_cleanup_authorization["payload"].get("authorization_kind")
        != "last_consumer_cleanup"
    ):
        raise ValueError("HCWDL-RKD recovery execution/cleanup lineage differs")
    prior_directory = Path(bank_root) / "generations" / prior_manifest["payload"][
        "generation_id"
    ]
    prior_consumer_registry = load_json(prior_directory / "consumer_registry.json")
    prior_consumer_registry_hash = validate_target_consumer_registry(
        prior_consumer_registry, logical_bank=logical_bank,
    )
    if (
        prior_manifest["parents"].get("consumer_registry")
        != prior_consumer_registry_hash
        or prior_manifest["payload"].get("authorized_consumers")
        != prior_consumer_registry["payload"]["consumers"]
    ):
        raise ValueError("HCWDL-RKD recovery prior consumer registry differs")
    if not isinstance(recovery_consumers, Sequence) or not recovery_consumers:
        raise ValueError("HCWDL-RKD recovery has no authorized consumer")
    # Use the target-registry validator as the single syntax/strategy/track
    # authority while deriving no reusable registry yet.  The actual registry
    # is created afterward with this recovery plan as generation parent.
    normalized = build_target_consumer_registry(
        logical_bank,
        purpose="recovery",
        consumers=recovery_consumers,
        generation_parent_sha256=completion_hash,
    )["payload"]["consumers"]

    authorization_contract = consumer_authorization.get("contract")
    if authorization_contract == CONFIRMATION_REGISTRY_CONTRACT:
        consumer_authorization_hash = validate_content_hash(
            consumer_authorization,
            expected_contract=CONFIRMATION_REGISTRY_CONTRACT,
            expected_schema_version=1,
        )
        expected_fields = {
            "contract", "schema_version", "screen_sha256", "campaign_sha256",
            "recipe_sha256", "logical_bank_sha256", "seeds", "rows",
            "execution_count", "content_hash",
        }
        campaign_hash = require_sha256(
            consumer_authorization.get("campaign_sha256"),
            name="confirmation campaign SHA-256",
        )
        recipe_hash = require_sha256(
            consumer_authorization.get("recipe_sha256"),
            name="confirmation representation-recipe SHA-256",
        )
        require_sha256(
            consumer_authorization.get("screen_sha256"),
            name="confirmation screen SHA-256",
        )
        authorized_rows = consumer_authorization.get("rows")
        expected_authorized_rows = []
        for node_id in expected_screen_consumer_nodes("TOFF"):
            for seed in CONFIRMATION_SEEDS:
                consumer = build_target_consumer_row(
                    logical_bank,
                    purpose="confirmation",
                    campaign_sha256=campaign_hash,
                    recipe_sha256=recipe_hash,
                    node_id=node_id,
                    seed=seed,
                )
                expected_authorized_rows.append({
                    "execution_id": consumer["execution_id"],
                    "execution_identity_payload": consumer[
                        "execution_identity_payload"
                    ],
                    "objective_id": node_id,
                    "seed": seed,
                    "logical_bank_sha256": logical_hash,
                    "physical_generation_sha256": None,
                })
        if (
            set(consumer_authorization) != expected_fields
            or consumer_authorization.get("logical_bank_sha256") != logical_hash
            or consumer_authorization.get("seeds") != list(CONFIRMATION_SEEDS)
            or consumer_authorization.get("execution_count") != 20
            or not isinstance(authorized_rows, list)
            or authorized_rows != expected_authorized_rows
        ):
            raise ValueError("HCWDL-RKD confirmation consumer rows differ")
        by_execution = {
            require_sha256(row.get("execution_id"), name="confirmation execution ID"): row
            for row in authorized_rows
            if isinstance(row, Mapping)
        }
        if len(by_execution) != len(authorized_rows):
            raise ValueError("HCWDL-RKD confirmation authorization repeats an execution")
        for row in normalized:
            frozen = by_execution.get(row["execution_id"])
            if (
                frozen is None
                or frozen.get("objective_id") != row["node_id"]
                or int(frozen.get("seed", -1)) != row["seed"]
                or frozen.get("execution_identity_payload")
                != row["execution_identity_payload"]
                or frozen.get("logical_bank_sha256") != logical_hash
            ):
                raise ValueError("HCWDL-RKD recovery consumer is not confirmation-authorized")
    elif authorization_contract == TARGET_CONSUMER_REGISTRY_CONTRACT:
        consumer_authorization_hash = validate_target_consumer_registry(
            consumer_authorization,
            logical_bank=logical_bank,
        )
        if consumer_authorization["payload"]["purpose"] not in {"screen", "confirmation"}:
            raise ValueError("HCWDL-RKD recovery cannot self-authorize with a recovery registry")
        authorized_by_execution = {
            row["execution_id"]: row
            for row in consumer_authorization["payload"]["consumers"]
        }
        if any(
            authorized_by_execution.get(row["execution_id"]) != row
            for row in normalized
        ):
            raise ValueError("HCWDL-RKD recovery consumer registry differs")
    else:
        raise ValueError("HCWDL-RKD recovery consumer authorization contract differs")

    completed_prior = set(
        prior_cleanup_authorization["payload"]["consumer_report_sha256"]
    )
    if completed_prior & {row["execution_id"] for row in normalized}:
        raise ValueError("HCWDL-RKD recovery cannot retry a completed prior consumer")
    if not isinstance(recovery_owner, Mapping) or not recovery_owner:
        raise ValueError("HCWDL-RKD target recovery owner is empty")
    return build_versioned_artifact(
        TARGET_RECOVERY_PLAN_CONTRACT,
        parents={
            "logical_bank": logical_hash,
            "prior_manifest": prior_manifest_hash,
            "prior_cleanup_authorization": authorization_hash,
            "prior_cleanup_completion": completion_hash,
            "prior_execution_attestation": attestation_hash,
            "prior_consumer_registry": prior_consumer_registry_hash,
            "consumer_authorization": consumer_authorization_hash,
        },
        payload={
            "logical_bank_id": prior_manifest["payload"]["logical_bank_id"],
            "prior_generation_id": prior_manifest["payload"]["generation_id"],
            "prior_logical_target_sha256": prior_manifest["payload"]["logical_target_sha256"],
            "recovery_consumers": normalized,
            "consumer_authorization_contract": authorization_contract,
            "consumer_authorization_sha256": consumer_authorization_hash,
            "prior_consumer_registry_sha256": prior_consumer_registry_hash,
            "exact_reconstruction_authorized": True,
            "recovery_owner": dict(recovery_owner),
            "steps": list(RECOVERY_STEP_ORDER),
            "writes_into_prior_generation": False,
            "transitive_descendant_retry": False,
        },
    )


def validate_reconstructed_generation(
    prior_manifest: Mapping[str, Any], new_manifest: Mapping[str, Any],
    *, recovery_plan: Mapping[str, Any], logical_bank: Mapping[str, Any],
    new_consumer_registry: Mapping[str, Any],
) -> None:
    plan_hash = validate_versioned_artifact(
        recovery_plan, expected_contract=TARGET_RECOVERY_PLAN_CONTRACT,
        required_payload_keys=(
            "logical_bank_id", "prior_generation_id", "prior_logical_target_sha256",
            "recovery_consumers", "consumer_authorization_sha256",
            "consumer_authorization_contract", "prior_consumer_registry_sha256",
            "exact_reconstruction_authorized", "recovery_owner", "steps",
            "writes_into_prior_generation", "transitive_descendant_retry",
        ),
    )
    prior_hash = validate_versioned_artifact(
        prior_manifest, expected_contract=TARGET_MANIFEST_CONTRACT,
    )
    validate_versioned_artifact(new_manifest, expected_contract=TARGET_MANIFEST_CONTRACT)
    logical_hash = validate_logical_target_bank(logical_bank)
    registry_hash = validate_target_consumer_registry(
        new_consumer_registry,
        logical_bank=logical_bank,
    )
    if (
        recovery_plan["parents"].get("prior_manifest") != prior_hash
        or recovery_plan["parents"].get("logical_bank") != logical_hash
        or recovery_plan["payload"]["exact_reconstruction_authorized"] is not True
        or new_consumer_registry["payload"]["generation_parent_sha256"] != plan_hash
        or new_manifest["parents"].get("consumer_registry") != registry_hash
        or new_manifest["parents"].get("logical_bank") != logical_hash
        or new_manifest["parents"].get("target_forward_spec")
        != prior_manifest["parents"].get("target_forward_spec")
        or new_manifest["payload"].get("purpose")
        != new_consumer_registry["payload"]["purpose"]
        or new_consumer_registry["payload"].get("purpose") != "recovery"
    ):
        raise ValueError("HCWDL-RKD reconstructed generation lineage differs")
    plan_payload = recovery_plan["payload"]
    if (
        plan_payload["logical_bank_id"]
        != logical_bank["payload"]["logical_bank_id"]
        or plan_payload["prior_generation_id"]
        != prior_manifest["payload"]["generation_id"]
        or plan_payload["prior_logical_target_sha256"]
        != prior_manifest["payload"]["logical_target_sha256"]
        or plan_payload["prior_consumer_registry_sha256"]
        != recovery_plan["parents"].get("prior_consumer_registry")
        or plan_payload["consumer_authorization_sha256"]
        != recovery_plan["parents"].get("consumer_authorization")
        or plan_payload["consumer_authorization_contract"] not in {
            CONFIRMATION_REGISTRY_CONTRACT, TARGET_CONSUMER_REGISTRY_CONTRACT,
        }
        or plan_payload["steps"] != list(RECOVERY_STEP_ORDER)
        or plan_payload["writes_into_prior_generation"] is not False
        or plan_payload["transitive_descendant_retry"] is not False
        or not isinstance(plan_payload["recovery_owner"], Mapping)
        or not plan_payload["recovery_owner"]
    ):
        raise ValueError("HCWDL-RKD recovery plan semantics differ")
    if (
        new_manifest["payload"]["generation_id"] == prior_manifest["payload"]["generation_id"]
        or new_manifest["payload"]["logical_target_sha256"]
        != prior_manifest["payload"]["logical_target_sha256"]
        or new_manifest["payload"]["logical_target_sha256"]
        != recovery_plan["payload"]["prior_logical_target_sha256"]
    ):
        raise ValueError("HCWDL-RKD reconstructed generation identity/target differs")
    expected_consumers = sorted(
        row["execution_id"] for row in recovery_plan["payload"]["recovery_consumers"]
    )
    actual_consumers = sorted(
        row["execution_id"] for row in new_manifest["payload"]["authorized_consumers"]
    )
    if actual_consumers != expected_consumers:
        raise ValueError("HCWDL-RKD reconstructed generation consumer set differs")
    if new_manifest["payload"]["authorized_consumers"] != new_consumer_registry[
        "payload"
    ]["consumers"]:
        raise ValueError("HCWDL-RKD reconstructed generation registry rows differ")
    if new_consumer_registry["payload"]["consumers"] != recovery_plan["payload"][
        "recovery_consumers"
    ]:
        raise ValueError("HCWDL-RKD reconstructed recovery consumer rows differ")


__all__ = [
    "RECOVERY_STEP_ORDER", "TargetRecoveryState", "authorize_miniature_target_cleanup",
    "authorize_target_cleanup",
    "build_abandoned_purge_authorization", "build_target_recovery_plan",
    "complete_target_cleanup", "inspect_target_recovery_state",
    "validate_cleanup_authorization", "validate_cleanup_completion",
    "validate_reconstructed_generation", "validate_target_consumer_training_report",
]

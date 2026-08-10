"""Neutral, population-scoped final-test reservation and capabilities.

Both HCWDL evaluator families must use this common checkpoint-namespace
registrar.  Registration is a serialized, crash-recoverable transaction:
publish one owner-scoped population bundle, append one immutable exposure
ledger generation, repair the operational HEAD, then publish a reservation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import threading
from typing import Any, Final, Iterator

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, require_sha256, sha256_file,
    validate_content_hash, with_content_hash, write_immutable_json,
)
from .hcwdl_representation_artifacts import (
    commit_staged_directory, fsync_directory, fsync_tree,
    write_staged_immutable_json,
)
from .hcwdl_representation_contracts import (
    SHARED_BINARY_ENVELOPE_CONTRACT,
    derive_envelope_id,
    derive_envelope_owner_id,
    validate_parent_hashes,
    validate_versioned_artifact,
)
from .hcwdl_representation_contracts import (
    SHARED_LEGACY_FINAL_EXPOSURE_CONTRACT,
)


FINAL_POPULATION_CONTRACT: Final = "HCWDL_SHARED_FINAL_POPULATION/v1"
FINAL_DISJOINTNESS_CONTRACT: Final = "HCWDL_SHARED_FINAL_POPULATION_DISJOINTNESS/v1"
EXPOSURE_LEDGER_CONTRACT: Final = "HCWDL_SHARED_FINAL_EXPOSURE_LEDGER/v1"
POPULATION_REGISTRATION_CONTRACT: Final = "HCWDL_SHARED_FINAL_POPULATION_REGISTRATION/v1"
FINAL_RESERVATION_CONTRACT: Final = "HCWDL_SHARED_FINAL_RESERVATION/v1"
LEGACY_CANCELLATION_CONTRACT: Final = "HCWDL_SHARED_FINAL_LEGACY_CANCELLATION/v1"
FINAL_TASK_REGISTRY_CONTRACT: Final = "HCWDL_SHARED_FINAL_TASK_REGISTRY/v1"
FINAL_EXECUTION_CLAIM_CONTRACT: Final = "HCWDL_SHARED_FINAL_EXECUTION_CLAIM/v1"
FINAL_ROLE_CAPABILITY_CONTRACT: Final = "HCWDL_SHARED_FINAL_ROLE_CAPABILITY/v1"
FINAL_RECOVERY_PLAN_CONTRACT: Final = "HCWDL_SHARED_FINAL_RECOVERY_PLAN/v1"
LEGACY_FINAL_EXPOSURE_CONTRACT: Final = SHARED_LEGACY_FINAL_EXPOSURE_CONTRACT
PARENT_FINAL_STATE_CONTRACT: Final = "HCWDL_REPRESENTATION_PARENT_FINAL_STATE/v1"
FINAL_DISPOSITION_CONTRACT: Final = "HCWDL_REPRESENTATION_FINAL_DISPOSITION/v1"
ALLOWED_DISPOSITIONS: Final = (
    "combined_confirmatory", "validation_only_parent_claim_consumed",
    "validation_only_holdout_exposed",
)
FINAL_TASK_KINDS: Final = (
    "row_selection", "assignment_shard", "assignment_finalize",
    "data_attestation", "execution_lock", "prediction_shard",
    "prediction_finalize", "metric_join", "final_aggregate",
)
_UNSET: Final = object()
BRANCH_FAMILIES_BY_KIND: Final = {
    "row_selection": frozenset({"selection"}),
    "assignment_shard": frozenset({"assignment"}),
    "assignment_finalize": frozenset(), "data_attestation": frozenset(),
    "execution_lock": frozenset(),
    "prediction_shard": frozenset({"hlt", "shell_exact", "native_offline"}),
    "prediction_finalize": frozenset(),
    "metric_join": frozenset({"label_escrow"}),
    "final_aggregate": frozenset(),
}
_IN_PROCESS_REGISTRAR_LOCK: Final = threading.RLock()


def _claims_root(namespace: str | Path) -> Path:
    return Path(namespace).resolve() / "final_claims"


def _ledger_root(namespace: str | Path) -> Path:
    return _claims_root(namespace) / "exposure_ledger"


def population_root(namespace: str | Path, population_sha256: str) -> Path:
    return _claims_root(namespace) / require_sha256(population_sha256, name="population")


def _identity_record(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != {"source_file_sha256", "source_entry"}:
        raise ValueError("final identity must be source hash plus source entry")
    entry = value["source_entry"]
    if isinstance(entry, bool) or not isinstance(entry, int) or entry < 0:
        raise ValueError("final source entry differs")
    return {
        "source_file_sha256": require_sha256(value["source_file_sha256"], name="source file"),
        "source_entry": int(entry),
    }


def _identity_digest(value: Mapping[str, Any]) -> str:
    return canonical_sha256(_identity_record(value))


def _population_records(value: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = value.get("identity_records")
    if not isinstance(raw, list) or not raw:
        raise ValueError("final population lacks identity records")
    rows = tuple(_identity_record(row) for row in raw)
    ordered = tuple(sorted(rows, key=lambda row: (row["source_file_sha256"], row["source_entry"])))
    digests = tuple(_identity_digest(row) for row in rows)
    if rows != ordered or len(digests) != len(set(digests)):
        raise ValueError("final population identity order/uniqueness differs")
    if value.get("identity_digests") != list(digests):
        raise ValueError("final population identity digests differ")
    if value.get("row_count") != len(rows):
        raise ValueError("final population row count differs")
    if value.get("identity_set_sha256") != canonical_sha256(list(digests)):
        raise ValueError("final population identity hash differs")
    return rows


def build_final_population(
    *, identities: Sequence[Mapping[str, Any]], source_snapshot_sha256: str,
    split_manifest_sha256: str, role: str, label_contract_sha256: str,
) -> dict[str, Any]:
    """Build the population solely from authenticated metadata, never ROOT."""

    if role != "final_test":
        raise ValueError("shared final population role must be final_test")
    rows = tuple(sorted(
        (_identity_record(row) for row in identities),
        key=lambda row: (row["source_file_sha256"], row["source_entry"]),
    ))
    digests = tuple(_identity_digest(row) for row in rows)
    if not rows or len(digests) != len(set(digests)):
        raise ValueError("final population identities must be nonempty and unique")
    source = require_sha256(source_snapshot_sha256, name="source snapshot")
    split = require_sha256(split_manifest_sha256, name="split manifest")
    result = with_content_hash({
        "contract": FINAL_POPULATION_CONTRACT, "schema_version": 1,
        "source_snapshot_sha256": source, "split_manifest_sha256": split,
        "label_contract_sha256": require_sha256(label_contract_sha256, name="label contract"),
        "role": role, "row_count": len(rows),
        "identity_set_sha256": canonical_sha256(list(digests)),
        "population_sha256": canonical_sha256({
            "source_snapshot_sha256": source, "split_manifest_sha256": split,
            "role": role, "identity_records": list(rows),
        }),
        "identity_records": list(rows), "identity_digests": list(digests),
        "root_branches_opened": False,
    })
    _population_records(result)
    return result


def derive_registration_owner_id(
    *, population: Mapping[str, Any], campaign_spec_sha256: str,
    disposition: str, selection_rule_sha256: str,
    assignment_spec_sha256: str, finalist_registry_commitment_sha256: str,
) -> str:
    validate_content_hash(population, expected_contract=FINAL_POPULATION_CONTRACT)
    _population_records(population)
    if disposition not in ALLOWED_DISPOSITIONS:
        raise ValueError("unknown final disposition")
    return canonical_sha256({
        "population_artifact_sha256": population["content_hash"],
        "campaign_spec_sha256": require_sha256(campaign_spec_sha256, name="campaign spec"),
        "disposition": disposition,
        "selection_rule_sha256": require_sha256(selection_rule_sha256, name="selection rule"),
        "label_contract_sha256": population["label_contract_sha256"],
        "assignment_spec_sha256": require_sha256(assignment_spec_sha256, name="assignment spec"),
        "finalist_registry_commitment_sha256": require_sha256(
            finalist_registry_commitment_sha256, name="finalist commitment",
        ),
    })


@contextmanager
def _exclusive_registrar(root: Path) -> Iterator[None]:
    """Use an advisory lock on a fixed inode so process death cannot strand it."""

    # POSIX advisory record locks are process-scoped, so a file lock alone does
    # not serialize two coordinators running in distinct threads of one process
    # (as can happen in local orchestrator tests).  Pair it with an in-process
    # lock; the fixed-inode advisory lock remains the cross-process authority.
    with _IN_PROCESS_REGISTRAR_LOCK:
        root.mkdir(parents=True, exist_ok=True)
        path = root / "registrar.lock"
        with path.open("a+b") as stream:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0"); stream.flush(); os.fsync(stream.fileno())
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _atomic_head(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".HEAD.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(payload), handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path); fsync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _ledger_generations(root: Path) -> list[dict[str, Any]]:
    generation_root = root / "generations"
    if not generation_root.exists():
        return []
    found: dict[int, dict[str, Any]] = {}
    for path in generation_root.glob("*.json"):
        row = load_json(path)
        validate_content_hash(row, expected_contract=EXPOSURE_LEDGER_CONTRACT)
        sequence = int(row.get("sequence", -1))
        if sequence < 0 or sequence in found or path.name != f"{sequence:06d}_{row['content_hash']}.json":
            raise ValueError("exposure ledger generation inventory differs")
        found[sequence] = row
    if set(found) != set(range(len(found))):
        raise ValueError("exposure ledger sequence has a gap")
    result = []
    previous = None
    for sequence in range(len(found)):
        row = found[sequence]
        if row.get("previous_sha256") != previous or not isinstance(row.get("registrations"), list):
            raise ValueError("exposure ledger hash chain differs")
        result.append(row); previous = row["content_hash"]
    return result


def _validate_registration_bundle(path: Path) -> dict[str, Any]:
    names = {item.name for item in path.iterdir() if item.is_file()} if path.is_dir() else set()
    required = {"population.json", "population_disjointness.json", "population_registration.json"}
    # Reservation and claim are later append-only transaction products in the
    # same population directory.  They must not make the immutable registration
    # bundle appear corrupt on an idempotent retry.
    if not required.issubset(names) or names - required - {
        "reservation.json", "legacy_cancellation.json", "execution_claim.json",
    }:
        raise ValueError("population registration bundle differs")
    population = load_json(path / "population.json")
    disjointness = load_json(path / "population_disjointness.json")
    registration = load_json(path / "population_registration.json")
    validate_content_hash(population, expected_contract=FINAL_POPULATION_CONTRACT)
    validate_content_hash(disjointness, expected_contract=FINAL_DISJOINTNESS_CONTRACT)
    validate_content_hash(registration, expected_contract=POPULATION_REGISTRATION_CONTRACT)
    _population_records(population)
    population_sha = population["population_sha256"]
    if disjointness.get("population_sha256") != population_sha or registration.get("population_sha256") != population_sha:
        raise ValueError("population bundle lineage differs")
    if registration.get("population_artifact_sha256") != population["content_hash"]:
        raise ValueError("population registration artifact hash differs")
    return {"population": population, "disjointness": disjointness, "registration": registration}


def _published_bundles(namespace: str | Path) -> list[dict[str, Any]]:
    root = _claims_root(namespace)
    if not root.exists():
        return []
    result = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_dir() or path.name == "exposure_ledger":
            continue
        require_sha256(path.name, name="population namespace")
        bundle = _validate_registration_bundle(path)
        if bundle["population"]["population_sha256"] != path.name:
            raise ValueError("population namespace identity differs")
        result.append(bundle)
    return result


def shared_reservations(namespace: str | Path) -> tuple[dict[str, Any], ...]:
    """Return all validated committed reservations in a shared namespace."""

    rows = []
    for bundle in _published_bundles(namespace):
        population_sha = bundle["population"]["population_sha256"]
        path = population_root(namespace, population_sha) / "reservation.json"
        if not path.exists():
            continue
        reservation = load_json(path)
        validate_content_hash(reservation, expected_contract=FINAL_RESERVATION_CONTRACT)
        if (
            reservation.get("population_sha256") != population_sha
            or reservation.get("population_registration_sha256")
            != bundle["registration"]["content_hash"]
        ):
            raise ValueError("shared final reservation lineage differs")
        rows.append(reservation)
    return tuple(sorted(rows, key=lambda row: row["population_sha256"]))


def reject_legacy_final_after_shared_reservation(namespace: str | Path) -> None:
    """Close the legacy finalist/execution-lock capability after reservation."""

    if shared_reservations(namespace):
        raise PermissionError(
            "legacy final evaluator is disabled after a shared population reservation"
        )


def claim_legacy_final_exposure(
    namespace: str | Path, *, execution_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically mark the old evaluator as an exposure before branch I/O.

    This is a compatibility bridge, not an alternative shared evaluator.  It
    serializes with population registration and permanently prevents a later
    campaign from claiming the same checkpoint namespace as an untouched
    confirmatory holdout.  Exact same-owner retries are idempotent.
    """

    if not isinstance(execution_identity, Mapping) or not execution_identity:
        raise ValueError("legacy final execution identity is empty")
    identity = dict(execution_identity)
    for name, value in identity.items():
        if not isinstance(name, str) or not name:
            raise ValueError("legacy final execution identity key differs")
        if name.endswith("_sha256"):
            identity[name] = require_sha256(value, name=f"legacy {name}")
        elif not isinstance(value, (str, int, bool)) and value is not None:
            raise TypeError("legacy final execution identity is not canonical JSON")
    artifact = with_content_hash({
        "contract": LEGACY_FINAL_EXPOSURE_CONTRACT, "schema_version": 1,
        "execution_identity": identity,
        "execution_identity_sha256": canonical_sha256(identity),
        "final_population_exposed": True,
        "shared_pipeline_required_for_future_execution": True,
    })
    ledger_root = _ledger_root(namespace)
    with _exclusive_registrar(ledger_root):
        if shared_reservations(namespace):
            raise PermissionError(
                "legacy final evaluator is disabled after a shared population reservation"
            )
        path = ledger_root / "legacy_final_exposure.json"
        if path.exists():
            existing = load_json(path)
            validate_content_hash(
                existing, expected_contract=LEGACY_FINAL_EXPOSURE_CONTRACT,
                expected_schema_version=1,
            )
            if existing != artifact:
                raise PermissionError("a different legacy final exposure already exists")
            return existing
        write_immutable_json(path, artifact)
        fsync_directory(path.parent)
        return artifact


def _repair_head(root: Path, ledger: Sequence[Mapping[str, Any]]) -> None:
    if not ledger:
        if (root / "HEAD.json").exists():
            raise ValueError("exposure ledger HEAD has no generation")
        return
    latest = ledger[-1]
    _atomic_head(root / "HEAD.json", {
        "sequence": int(latest["sequence"]),
        "relative_path": f"generations/{int(latest['sequence']):06d}_{latest['content_hash']}.json",
        "content_hash": latest["content_hash"],
    })


def register_final_population(
    *, checkpoint_namespace: str | Path, population: Mapping[str, Any],
    campaign_spec_sha256: str, final_disposition: Mapping[str, Any],
    parent_final_state: Mapping[str, Any],
    selection_rule_sha256: str, assignment_spec_sha256: str,
    finalist_registry_commitment_sha256: str,
    registration_owner_id: str | None = None,
    legacy_jobs_present: bool = False, fail_after: str | None = None,
) -> dict[str, Any]:
    """Register and reserve a population; exact retries finish missing steps."""

    validate_content_hash(population, expected_contract=FINAL_POPULATION_CONTRACT)
    validate_parent_final_state(parent_final_state)
    validate_content_hash(final_disposition, expected_contract=FINAL_DISPOSITION_CONTRACT)
    if final_disposition.get("parent_final_state_sha256") != parent_final_state["content_hash"]:
        raise ValueError("final disposition parent-state lineage differs")
    disposition = str(final_disposition.get("disposition", ""))
    expected_disposition = build_final_disposition(
        parent_final_state=parent_final_state,
        requested=str(final_disposition.get("requested", "")),
    )
    if dict(final_disposition) != expected_disposition:
        raise ValueError("final disposition is not derivable from the audited parent state")
    if (
        disposition not in ALLOWED_DISPOSITIONS
        or bool(final_disposition.get("final_tasks_registered"))
        != (disposition == "combined_confirmatory")
    ):
        raise ValueError("final disposition authorization differs")
    records = _population_records(population)
    if bool(legacy_jobs_present) != bool(parent_final_state["legacy_jobs"]):
        raise ValueError("reservation legacy-job inventory differs from the parent audit")
    expected_owner = derive_registration_owner_id(
        population=population, campaign_spec_sha256=campaign_spec_sha256,
        disposition=disposition, selection_rule_sha256=selection_rule_sha256,
        assignment_spec_sha256=assignment_spec_sha256,
        finalist_registry_commitment_sha256=finalist_registry_commitment_sha256,
    )
    owner = expected_owner if registration_owner_id is None else require_sha256(registration_owner_id, name="owner")
    if owner != expected_owner:
        raise ValueError("registration owner differs from deterministic commitment")
    population_sha = population["population_sha256"]
    campaign = require_sha256(campaign_spec_sha256, name="campaign spec")
    parent_state = parent_final_state["content_hash"]
    selection_rule = require_sha256(selection_rule_sha256, name="selection rule")
    assignment_spec = require_sha256(assignment_spec_sha256, name="assignment spec")
    finalist_commitment = require_sha256(finalist_registry_commitment_sha256, name="finalist commitment")
    ledger_root = _ledger_root(checkpoint_namespace)
    with _exclusive_registrar(ledger_root):
        legacy_exposure_path = ledger_root / "legacy_final_exposure.json"
        legacy_exposure_sha256 = None
        if legacy_exposure_path.exists():
            legacy_exposure = load_json(legacy_exposure_path)
            legacy_exposure_sha256 = validate_content_hash(
                legacy_exposure, expected_contract=LEGACY_FINAL_EXPOSURE_CONTRACT,
                expected_schema_version=1,
            )
            if disposition == "combined_confirmatory":
                raise PermissionError(
                    "legacy final exposure requires a validation-only disposition"
                )
        registration = with_content_hash({
            "contract": POPULATION_REGISTRATION_CONTRACT, "schema_version": 1,
            "population_sha256": population_sha,
            "population_artifact_sha256": population["content_hash"],
            "registration_owner_id": owner, "campaign_spec_sha256": campaign,
            "parent_final_state_sha256": parent_state, "disposition": disposition,
            "final_disposition_sha256": final_disposition["content_hash"],
            "selection_rule_sha256": selection_rule,
            "label_contract_sha256": population["label_contract_sha256"],
            "assignment_spec_sha256": assignment_spec,
            "finalist_registry_commitment_sha256": finalist_commitment,
            "legacy_final_exposure_sha256": legacy_exposure_sha256,
        })
        ledger = _ledger_generations(ledger_root)
        bundles = _published_bundles(checkpoint_namespace)
        same = [row for row in bundles if row["population"]["population_sha256"] == population_sha]
        if same:
            if len(same) != 1 or same[0]["population"] != population or same[0]["registration"] != registration:
                raise PermissionError("population is owned by a different registration")
        else:
            identity_set = {_identity_digest(row) for row in records}
            checked = []
            for bundle in bundles:
                if identity_set.intersection(bundle["population"]["identity_digests"]):
                    raise PermissionError("final population overlaps an existing exposure registration")
                checked.append(bundle["population"]["population_sha256"])
            disjointness = with_content_hash({
                "contract": FINAL_DISJOINTNESS_CONTRACT, "schema_version": 1,
                "population_sha256": population_sha, "checked_against": checked,
                "checked_identity_set_sha256": population["identity_set_sha256"],
                "overlap_count": 0,
                "ledger_parent_sha256": None if not ledger else ledger[-1]["content_hash"],
            })
            staging = ledger_root / "proposals" / owner / population_sha
            staging.mkdir(parents=True, exist_ok=True)
            write_staged_immutable_json(staging / "population.json", population)
            write_staged_immutable_json(staging / "population_disjointness.json", disjointness)
            write_staged_immutable_json(staging / "population_registration.json", registration)
            fsync_tree(staging)
            if fail_after == "proposal":
                raise RuntimeError("injected failure after population proposal")
            commit_staged_directory(
                staging, population_root(checkpoint_namespace, population_sha),
                validate_committed=_validate_registration_bundle,
            )
            if fail_after == "registration_bundle":
                raise RuntimeError("injected failure after registration bundle")

        latest_rows = [] if not ledger else list(ledger[-1]["registrations"])
        by_population = {row["population_sha256"]: row for row in latest_rows}
        expected_row = {
            "population_sha256": population_sha,
            "population_artifact_sha256": population["content_hash"],
            "registration_sha256": registration["content_hash"],
            "registration_owner_id": owner,
            "identity_set_sha256": population["identity_set_sha256"],
            "row_count": int(population["row_count"]),
        }
        if population_sha in by_population:
            if by_population[population_sha] != expected_row:
                raise ValueError("ledger registration row differs")
        else:
            latest_rows.append(expected_row); latest_rows.sort(key=lambda row: row["population_sha256"])
            generation = with_content_hash({
                "contract": EXPOSURE_LEDGER_CONTRACT, "schema_version": 1,
                "sequence": len(ledger),
                "previous_sha256": None if not ledger else ledger[-1]["content_hash"],
                "registrations": latest_rows,
            })
            path = ledger_root / "generations" / f"{len(ledger):06d}_{generation['content_hash']}.json"
            write_immutable_json(path, generation); fsync_directory(path.parent)
            if fail_after == "ledger_generation":
                raise RuntimeError("injected failure after ledger generation")
            ledger = [*ledger, generation]
        _repair_head(ledger_root, ledger)
        if fail_after == "head":
            raise RuntimeError("injected failure after ledger HEAD")

        reservation = with_content_hash({
            "contract": FINAL_RESERVATION_CONTRACT, "schema_version": 1,
            "population_sha256": population_sha,
            "population_registration_sha256": registration["content_hash"],
            "registration_owner_id": owner, "campaign_spec_sha256": campaign,
            "parent_final_state_sha256": parent_state,
            "final_disposition_sha256": final_disposition["content_hash"],
            "source_snapshot_sha256": population["source_snapshot_sha256"],
            "split_manifest_sha256": population["split_manifest_sha256"],
            "selection_rule_sha256": selection_rule,
            "label_contract_sha256": population["label_contract_sha256"],
            "assignment_spec_sha256": assignment_spec,
            "finalist_registry_commitment_sha256": finalist_commitment,
            "legacy_final_exposure_sha256": legacy_exposure_sha256,
            "disposition": disposition, "legacy_jobs_present": bool(legacy_jobs_present),
            "legacy_job_ids": [
                row["job_id"] for row in parent_final_state["legacy_jobs"]
            ],
            "allows_final_execution": disposition == "combined_confirmatory",
        })
        path = population_root(checkpoint_namespace, population_sha) / "reservation.json"
        if path.exists():
            existing = load_json(path)
            validate_content_hash(existing, expected_contract=FINAL_RESERVATION_CONTRACT)
            if existing != reservation:
                raise PermissionError("same population has a different reservation")
            return existing
        write_immutable_json(path, reservation); fsync_directory(path.parent)
        if fail_after == "reservation":
            raise RuntimeError("injected failure after reservation")
        return reservation


def audit_parent_final_state(
    *, candidate_artifacts: Sequence[Mapping[str, Any]],
    parent_campaign_sha256: str, exploratory_campaign_sha256: str | None = None,
) -> dict[str, Any]:
    exposures, pending, legacy_jobs = [], [], []
    for row in candidate_artifacts:
        if not isinstance(row, Mapping):
            raise TypeError("legacy final audit row must be a versioned artifact")
        contract = row.get("contract")
        schema_version = row.get("schema_version")
        if not isinstance(contract, str) or not contract or not isinstance(
            schema_version, int,
        ):
            raise ValueError("legacy final audit artifact identity differs")
        digest = validate_content_hash(
            row, expected_contract=contract, expected_schema_version=schema_version,
        )
        if row.get("role") == "final_test" and (
            row.get("model_derived_output") is True or row.get("execution_claimed") is True
            or int(row.get("completed_prediction_rows", 0)) > 0
        ):
            exposures.append({"artifact_sha256": digest, "kind": str(row.get("kind", "unknown"))})
        state = str(row.get("scheduler_state", "")).upper()
        job_id = str(row.get("job_id", ""))
        if state:
            if not job_id:
                raise ValueError("legacy scheduler audit row lacks its job ID")
            legacy_jobs.append({
                "artifact_sha256": digest, "job_id": job_id,
                "scheduler_state": state,
            })
        if state in {"PENDING", "RUNNING", "CONFIGURING"}:
            pending.append({
                "artifact_sha256": digest, "job_id": job_id,
                "scheduler_state": state,
            })
    if len({row["job_id"] for row in legacy_jobs}) != len(legacy_jobs):
        raise ValueError("legacy final audit repeats a scheduler job ID")
    return with_content_hash({
        "contract": PARENT_FINAL_STATE_CONTRACT, "schema_version": 1,
        "parent_campaign_sha256": require_sha256(parent_campaign_sha256, name="parent campaign"),
        "exploratory_campaign_sha256": None if exploratory_campaign_sha256 is None else require_sha256(exploratory_campaign_sha256, name="exploratory campaign"),
        "audited_artifacts": len(candidate_artifacts), "exposures": exposures,
        "legacy_jobs": sorted(legacy_jobs, key=lambda row: row["job_id"]),
        "pending_or_running_legacy_workers": pending,
        "final_population_already_exposed": bool(exposures),
    })


def validate_parent_final_state(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value, expected_contract=PARENT_FINAL_STATE_CONTRACT,
        expected_schema_version=1,
    )
    require_sha256(value.get("parent_campaign_sha256"), name="parent campaign")
    exploratory = value.get("exploratory_campaign_sha256")
    if exploratory is not None:
        require_sha256(exploratory, name="exploratory campaign")
    # Non-scheduler artifacts may also be audited, but every scheduler row
    # must be included in the total.  A negative/noninteger total is never
    # meaningful evidence.
    count = value.get("audited_artifacts")
    if isinstance(count, bool) or not isinstance(count, int) or count < len(
        value.get("legacy_jobs", [])
    ):
        raise ValueError("parent final audited-artifact count differs")
    exposures = value.get("exposures")
    jobs = value.get("legacy_jobs")
    pending = value.get("pending_or_running_legacy_workers")
    if not isinstance(exposures, list) or not isinstance(jobs, list) or not isinstance(
        pending, list,
    ):
        raise ValueError("parent final audit registries differ")
    job_ids: set[str] = set()
    normalized_pending = []
    for row in jobs:
        if not isinstance(row, Mapping) or set(row) != {
            "artifact_sha256", "job_id", "scheduler_state",
        }:
            raise ValueError("parent final legacy-job row differs")
        require_sha256(row["artifact_sha256"], name="legacy artifact")
        job_id = str(row["job_id"])
        state = str(row["scheduler_state"]).upper()
        if not job_id or job_id in job_ids or state not in {
            "PENDING", "RUNNING", "CONFIGURING", "COMPLETED", "CANCELLED",
            "FAILED", "TIMEOUT", "OUT_OF_MEMORY",
        }:
            raise ValueError("parent final legacy-job identity/state differs")
        job_ids.add(job_id)
        if state in {"PENDING", "RUNNING", "CONFIGURING"}:
            normalized_pending.append(dict(row))
    if pending != normalized_pending:
        raise ValueError("parent final pending-worker registry differs")
    for row in exposures:
        if not isinstance(row, Mapping) or set(row) != {"artifact_sha256", "kind"}:
            raise ValueError("parent final exposure row differs")
        require_sha256(row["artifact_sha256"], name="exposed artifact")
        if not str(row["kind"]):
            raise ValueError("parent final exposure kind differs")
    if bool(value.get("final_population_already_exposed")) != bool(exposures):
        raise ValueError("parent final exposure aggregate differs")
    return digest


def build_final_disposition(*, parent_final_state: Mapping[str, Any], requested: str) -> dict[str, Any]:
    validate_parent_final_state(parent_final_state)
    if requested not in ALLOWED_DISPOSITIONS:
        raise ValueError("final disposition request differs")
    exposed = bool(parent_final_state["final_population_already_exposed"])
    pending = bool(parent_final_state["pending_or_running_legacy_workers"])
    actual = "validation_only_parent_claim_consumed" if exposed or pending else requested
    return with_content_hash({
        "contract": FINAL_DISPOSITION_CONTRACT, "schema_version": 1,
        "parent_final_state_sha256": parent_final_state["content_hash"],
        "requested": requested, "disposition": actual,
        "reason": "parent_population_exposed" if exposed else "legacy_worker_not_cancelled" if pending else None,
        "final_tasks_registered": actual == "combined_confirmatory",
    })


def build_legacy_cancellation(
    *, reservation: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]],
    output_audit_sha256: str,
) -> dict[str, Any]:
    validate_content_hash(reservation, expected_contract=FINAL_RESERVATION_CONTRACT)
    normalized, seen = [], set()
    for row in jobs:
        job_id = str(row.get("job_id", "")); state = str(row.get("scheduler_state_after", "")).upper()
        if not job_id or job_id in seen:
            raise ValueError("legacy cancellation job inventory differs")
        seen.add(job_id)
        if state not in {"CANCELLED", "COMPLETED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY"}:
            raise PermissionError("legacy worker may still race final execution")
        normalized.append({"job_id": job_id, "scheduler_state_after": state})
    if bool(reservation["legacy_jobs_present"]) != bool(normalized):
        raise ValueError("legacy cancellation differs from reservation")
    if {row["job_id"] for row in normalized} != set(reservation["legacy_job_ids"]):
        raise ValueError("legacy cancellation does not cover the reserved job registry")
    return with_content_hash({
        "contract": LEGACY_CANCELLATION_CONTRACT, "schema_version": 1,
        "population_sha256": reservation["population_sha256"],
        "reservation_sha256": reservation["content_hash"],
        "jobs": sorted(normalized, key=lambda row: row["job_id"]),
        "output_audit_sha256": require_sha256(output_audit_sha256, name="output audit"),
        "no_worker_can_race": True,
    })


def _safe_output(value: str) -> str:
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("registered output path differs")
    return path.as_posix()


def _one_kind(rows: Sequence[Mapping[str, Any]], kind: str) -> Mapping[str, Any]:
    found = [row for row in rows if row["kind"] == kind]
    if len(found) != 1:
        raise ValueError(f"final task registry requires exactly one {kind} row")
    return found[0]


def _validate_complete_final_task_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Validate the frozen claim->selection->prediction final DAG.

    Counts are intentionally derived from the registered source/finalist axes,
    so the validator is equally strict for smoke, pilot, and production.  A
    caller cannot obtain a claim with a syntactically valid but incomplete
    one-row registry.
    """

    selection = _one_kind(rows, "row_selection")
    assignment_finalize = _one_kind(rows, "assignment_finalize")
    data_attestation = _one_kind(rows, "data_attestation")
    execution_lock = _one_kind(rows, "execution_lock")
    metric_join = _one_kind(rows, "metric_join")
    aggregate = _one_kind(rows, "final_aggregate")
    assignment_rows = [row for row in rows if row["kind"] == "assignment_shard"]
    prediction_rows = [row for row in rows if row["kind"] == "prediction_shard"]
    prediction_manifests = [
        row for row in rows if row["kind"] == "prediction_finalize"
    ]
    if not assignment_rows or not prediction_rows or not prediction_manifests:
        raise ValueError("final task registry omits a required array wave")
    expected_kinds = {
        "row_selection", "assignment_shard", "assignment_finalize",
        "data_attestation", "execution_lock", "prediction_shard",
        "prediction_finalize", "metric_join", "final_aggregate",
    }
    if {row["kind"] for row in rows} != expected_kinds:
        raise ValueError("final task registry kind set differs")

    if selection["dependencies"] or selection["branch_family"] != "selection":
        raise ValueError("final row-selection registry row differs")
    source_partitions = [str(row["source_partition"] or "") for row in assignment_rows]
    if any(not value for value in source_partitions) or len(source_partitions) != len(
        set(source_partitions)
    ):
        raise ValueError("final assignment source-partition axis differs")
    for row in assignment_rows:
        if (
            row["branch_family"] != "assignment"
            or row["dependencies"] != [selection["task_id"]]
            or row["finalist_id"] is not None
            or row["checkpoint_sha256"] is not None
        ):
            raise ValueError("final assignment-shard registry row differs")
    assignment_ids = {row["task_id"] for row in assignment_rows}
    if set(assignment_finalize["dependencies"]) != assignment_ids:
        raise ValueError("final assignment-finalize dependency set differs")
    if data_attestation["dependencies"] != [assignment_finalize["task_id"]]:
        raise ValueError("final data-attestation dependency differs")
    if execution_lock["dependencies"] != [data_attestation["task_id"]]:
        raise ValueError("final execution-lock dependency differs")

    finalists: dict[str, dict[str, Any]] = {}
    observed_pairs: set[tuple[str, str]] = set()
    for row in prediction_rows:
        finalist = str(row["finalist_id"] or "")
        source = str(row["source_partition"] or "")
        if not finalist or source not in set(source_partitions):
            raise ValueError("final prediction source/finalist axis differs")
        pair = (finalist, source)
        if pair in observed_pairs:
            raise ValueError("final prediction registry repeats a finalist/source row")
        observed_pairs.add(pair)
        signature = {
            "checkpoint_sha256": row["checkpoint_sha256"],
            "branch_family": row["branch_family"],
        }
        if finalist in finalists and finalists[finalist] != signature:
            raise ValueError("finalist checkpoint/domain differs across source shards")
        finalists[finalist] = signature
        if row["dependencies"] != [execution_lock["task_id"]]:
            raise ValueError("final prediction-shard dependency differs")
    expected_pairs = {
        (finalist, source) for finalist in finalists for source in source_partitions
    }
    if observed_pairs != expected_pairs:
        raise ValueError("final prediction registry is not the full Cartesian wave")

    manifests_by_finalist: dict[str, Mapping[str, Any]] = {}
    for row in prediction_manifests:
        finalist = str(row["finalist_id"] or "")
        if (
            not finalist or finalist in manifests_by_finalist
            or row["source_partition"] is not None
            or row["branch_family"] is not None
        ):
            raise ValueError("final prediction-manifest registry row differs")
        expected_dependencies = {
            shard["task_id"] for shard in prediction_rows
            if shard["finalist_id"] == finalist
        }
        if set(row["dependencies"]) != expected_dependencies:
            raise ValueError("final prediction-manifest dependency set differs")
        manifests_by_finalist[finalist] = row
    if set(manifests_by_finalist) != set(finalists):
        raise ValueError("final prediction-manifest finalist axis differs")
    if set(metric_join["dependencies"]) != {
        row["task_id"] for row in prediction_manifests
    }:
        raise ValueError("final metric-join dependency set differs")
    if aggregate["dependencies"] != [metric_join["task_id"]]:
        raise ValueError("final aggregate dependency differs")
    outputs = [path for row in rows for path in row["registered_outputs"]]
    if len(outputs) != len(set(outputs)):
        raise ValueError("final task registry reuses an output path")


def build_final_task_registry(
    *, population_sha256: str, campaign_spec_sha256: str,
    tasks: Sequence[Mapping[str, Any]], finalist_lock_sha256: str,
    submission_ledger_sha256: str,
) -> dict[str, Any]:
    normalized, seen = [], set()
    for raw in tasks:
        task_id, kind = str(raw.get("task_id", "")), str(raw.get("kind", ""))
        if not task_id or task_id in seen or kind not in FINAL_TASK_KINDS:
            raise ValueError("final task identity/kind differs")
        seen.add(task_id)
        outputs = tuple(_safe_output(value) for value in raw.get("registered_outputs", ()))
        dependencies = tuple(str(value) for value in raw.get("dependencies", ()))
        if not outputs or len(outputs) != len(set(outputs)) or any(not item or item == task_id for item in dependencies):
            raise ValueError("final task outputs/dependencies differ")
        branch = raw.get("branch_family"); allowed = BRANCH_FAMILIES_BY_KIND[kind]
        if (branch is None and allowed) or (branch is not None and branch not in allowed):
            raise ValueError("final task branch family differs")
        checkpoint = raw.get("checkpoint_sha256")
        if kind == "prediction_shard":
            if not raw.get("source_partition") or not raw.get("finalist_id"):
                raise ValueError("prediction task lacks source/finalist")
            checkpoint = require_sha256(checkpoint, name="prediction checkpoint")
        elif checkpoint is not None:
            raise ValueError("nonprediction task binds checkpoint")
        resource = raw.get("resource_signature")
        if not isinstance(resource, Mapping) or not resource:
            raise ValueError("final task lacks resource signature")
        normalized.append({
            "task_id": task_id, "kind": kind, "purpose": str(raw.get("purpose", kind)),
            "branch_family": branch, "source_partition": raw.get("source_partition"),
            "finalist_id": raw.get("finalist_id"), "checkpoint_sha256": checkpoint,
            "registered_outputs": list(outputs), "dependencies": list(dependencies),
            "resource_signature": dict(sorted(resource.items())),
        })
    if not normalized:
        raise ValueError("final task registry is empty")
    positions = {row["task_id"]: index for index, row in enumerate(normalized)}
    for row in normalized:
        if set(row["dependencies"]) - seen or any(positions[parent] >= positions[row["task_id"]] for parent in row["dependencies"]):
            raise ValueError("final task registry dependency DAG differs")
    _validate_complete_final_task_rows(normalized)
    return with_content_hash({
        "contract": FINAL_TASK_REGISTRY_CONTRACT, "schema_version": 1,
        "population_sha256": require_sha256(population_sha256, name="population"),
        "campaign_spec_sha256": require_sha256(campaign_spec_sha256, name="campaign spec"),
        "finalist_lock_sha256": require_sha256(finalist_lock_sha256, name="finalist lock"),
        "submission_ledger_sha256": require_sha256(submission_ledger_sha256, name="submission ledger"),
        "tasks": normalized,
    })


def validate_final_task_registry(value: Mapping[str, Any]) -> str:
    """Rebuild and byte-compare the complete immutable final-task registry."""

    digest = validate_content_hash(
        value, expected_contract=FINAL_TASK_REGISTRY_CONTRACT,
        expected_schema_version=1,
    )
    rebuilt = build_final_task_registry(
        population_sha256=value.get("population_sha256"),
        campaign_spec_sha256=value.get("campaign_spec_sha256"),
        finalist_lock_sha256=value.get("finalist_lock_sha256"),
        submission_ledger_sha256=value.get("submission_ledger_sha256"),
        tasks=value.get("tasks", ()),
    )
    if dict(value) != rebuilt:
        raise ValueError("final task registry canonical bytes differ")
    return digest


def derive_claim_owner_id(
    *, campaign_spec_sha256: str, task_registry_sha256: str, finalist_lock_sha256: str,
) -> str:
    return canonical_sha256({
        "campaign_spec_sha256": require_sha256(campaign_spec_sha256, name="campaign spec"),
        "task_registry_sha256": require_sha256(task_registry_sha256, name="task registry"),
        "finalist_lock_sha256": require_sha256(finalist_lock_sha256, name="finalist lock"),
    })


def validate_execution_claim(
    value: Mapping[str, Any], *, task_registry: Mapping[str, Any],
) -> str:
    """Validate the canonical claim and its exact authenticated task registry."""

    registry_sha256 = validate_final_task_registry(task_registry)
    digest = validate_content_hash(
        value, expected_contract=FINAL_EXECUTION_CLAIM_CONTRACT,
        expected_schema_version=1,
    )
    required = {
        "contract", "schema_version", "population_sha256", "reservation_sha256",
        "campaign_spec_sha256", "finalist_lock_sha256", "task_registry_sha256",
        "submission_ledger_sha256", "legacy_cancellation_sha256", "source_commit",
        "claim_owner_id", "same_owner_recovery_allowed", "content_hash",
    }
    if set(value) != required:
        raise ValueError("execution claim schema differs")
    for name in (
        "population_sha256", "reservation_sha256", "campaign_spec_sha256",
        "finalist_lock_sha256", "task_registry_sha256", "submission_ledger_sha256",
        "claim_owner_id",
    ):
        require_sha256(value.get(name), name=name.replace("_", " "))
    cancellation = value.get("legacy_cancellation_sha256")
    if cancellation is not None:
        require_sha256(cancellation, name="legacy cancellation")
    source_commit = str(value.get("source_commit", ""))
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("execution claim source commit differs")
    if value.get("same_owner_recovery_allowed") is not True:
        raise ValueError("execution claim recovery policy differs")
    if (
        value["task_registry_sha256"] != registry_sha256
        or value["population_sha256"] != task_registry["population_sha256"]
        or value["campaign_spec_sha256"] != task_registry["campaign_spec_sha256"]
        or value["finalist_lock_sha256"] != task_registry["finalist_lock_sha256"]
        or value["submission_ledger_sha256"]
        != task_registry["submission_ledger_sha256"]
    ):
        raise ValueError("execution claim registry lineage differs")
    expected_owner = derive_claim_owner_id(
        campaign_spec_sha256=value["campaign_spec_sha256"],
        task_registry_sha256=registry_sha256,
        finalist_lock_sha256=value["finalist_lock_sha256"],
    )
    if value["claim_owner_id"] != expected_owner:
        raise ValueError("execution claim owner differs")
    return digest


def claim_final_execution(
    *, checkpoint_namespace: str | Path, reservation: Mapping[str, Any],
    task_registry: Mapping[str, Any], finalist_lock_sha256: str,
    source_commit: str, claim_owner_id: str | None = None,
) -> dict[str, Any]:
    validate_content_hash(reservation, expected_contract=FINAL_RESERVATION_CONTRACT)
    validate_final_task_registry(task_registry)
    if reservation.get("allows_final_execution") is not True:
        raise PermissionError("validation-only disposition cannot claim final execution")
    if reservation["population_sha256"] != task_registry["population_sha256"]:
        raise ValueError("claim population differs")
    if reservation.get("campaign_spec_sha256") != task_registry.get("campaign_spec_sha256"):
        raise ValueError("claim campaign differs from the population reservation")
    finalist = require_sha256(finalist_lock_sha256, name="finalist lock")
    if task_registry["finalist_lock_sha256"] != finalist:
        raise ValueError("claim finalist lock differs")
    if len(source_commit) != 40 or any(character not in "0123456789abcdef" for character in source_commit):
        raise ValueError("claim source commit differs")
    cancellation_sha256 = None
    if reservation.get("legacy_jobs_present") is True:
        cancellation_path = population_root(
            checkpoint_namespace, reservation["population_sha256"],
        ) / "legacy_cancellation.json"
        if not cancellation_path.is_file():
            raise PermissionError(
                "legacy final workers require an authenticated cancellation before claim"
            )
        cancellation = load_json(cancellation_path)
        cancellation_sha256 = validate_content_hash(
            cancellation, expected_contract=LEGACY_CANCELLATION_CONTRACT,
            expected_schema_version=1,
        )
        if (
            cancellation.get("population_sha256") != reservation["population_sha256"]
            or cancellation.get("reservation_sha256") != reservation["content_hash"]
            or cancellation.get("no_worker_can_race") is not True
            or {row.get("job_id") for row in cancellation.get("jobs", [])}
            != set(reservation["legacy_job_ids"])
        ):
            raise PermissionError("legacy cancellation lineage/coverage differs")
    expected = derive_claim_owner_id(
        campaign_spec_sha256=task_registry["campaign_spec_sha256"],
        task_registry_sha256=task_registry["content_hash"], finalist_lock_sha256=finalist,
    )
    owner = expected if claim_owner_id is None else require_sha256(claim_owner_id, name="claim owner")
    if owner != expected:
        raise ValueError("claim owner differs")
    claim = with_content_hash({
        "contract": FINAL_EXECUTION_CLAIM_CONTRACT, "schema_version": 1,
        "population_sha256": reservation["population_sha256"],
        "reservation_sha256": reservation["content_hash"],
        "campaign_spec_sha256": task_registry["campaign_spec_sha256"],
        "finalist_lock_sha256": finalist,
        "task_registry_sha256": task_registry["content_hash"],
        "submission_ledger_sha256": task_registry["submission_ledger_sha256"],
        "legacy_cancellation_sha256": cancellation_sha256,
        "source_commit": source_commit, "claim_owner_id": owner,
        "same_owner_recovery_allowed": True,
    })
    path = population_root(checkpoint_namespace, reservation["population_sha256"]) / "execution_claim.json"
    if path.exists():
        existing = load_json(path)
        validate_content_hash(existing, expected_contract=FINAL_EXECUTION_CLAIM_CONTRACT)
        if existing != claim:
            raise PermissionError("population has a different execution claim")
        return existing
    write_immutable_json(path, claim); fsync_directory(path.parent)
    return claim


def issue_role_capability(
    *, claim: Mapping[str, Any], task_registry: Mapping[str, Any], task_id: str,
    execution_lock_sha256: str | None = None,
) -> dict[str, Any]:
    validate_execution_claim(claim, task_registry=task_registry)
    if (
        claim["task_registry_sha256"] != task_registry["content_hash"]
        or claim["population_sha256"] != task_registry["population_sha256"]
        or claim["campaign_spec_sha256"] != task_registry["campaign_spec_sha256"]
        or claim["finalist_lock_sha256"] != task_registry["finalist_lock_sha256"]
    ):
        raise ValueError("capability task registry differs")
    matches = [row for row in task_registry["tasks"] if row["task_id"] == task_id]
    if len(matches) != 1:
        raise PermissionError("task is not registered")
    task = matches[0]
    needs_lock = task["kind"] in {"prediction_shard", "prediction_finalize", "metric_join", "final_aggregate"}
    execution = None if execution_lock_sha256 is None else require_sha256(execution_lock_sha256, name="execution lock")
    if needs_lock != (execution is not None):
        raise PermissionError("capability execution-lock scope differs")
    return with_content_hash({
        "contract": FINAL_ROLE_CAPABILITY_CONTRACT, "schema_version": 1,
        "population_sha256": claim["population_sha256"],
        "execution_claim_sha256": claim["content_hash"],
        "task_registry_sha256": task_registry["content_hash"],
        "execution_lock_sha256": execution, "task": task,
    })


def validate_role_capability(
    capability: Mapping[str, Any], *, execution_claim: Mapping[str, Any],
    task_registry: Mapping[str, Any], expected_population_sha256: str,
    expected_task_id: str, allowed_kinds: Sequence[str] | None = None,
    expected_execution_lock_sha256: str | None | object = _UNSET,
    expected_branch_family: str | None = None,
) -> str:
    """Validate a capability against its authenticated claim and registry.

    A capability is intentionally not a standalone bearer token.  Requiring
    both immutable parents at every use prevents a freshly self-hashed JSON
    object from inventing a task, branch family, or execution-lock scope.
    Production callers must first authenticate the bytes of all three
    artifacts through the registered runtime inputs.
    """

    registry_sha256 = validate_final_task_registry(task_registry)
    claim_sha256 = validate_execution_claim(
        execution_claim, task_registry=task_registry,
    )
    digest = validate_content_hash(capability, expected_contract=FINAL_ROLE_CAPABILITY_CONTRACT)
    population_sha256 = require_sha256(
        expected_population_sha256, name="population",
    )
    if (
        execution_claim.get("population_sha256") != population_sha256
        or task_registry.get("population_sha256") != population_sha256
        or execution_claim.get("task_registry_sha256") != registry_sha256
        or execution_claim.get("campaign_spec_sha256")
        != task_registry.get("campaign_spec_sha256")
        or execution_claim.get("finalist_lock_sha256")
        != task_registry.get("finalist_lock_sha256")
    ):
        raise PermissionError("capability immutable parent lineage differs")
    if capability.get("population_sha256") != population_sha256:
        raise PermissionError("capability population differs")
    task = capability.get("task")
    if not isinstance(task, Mapping) or task.get("task_id") != expected_task_id:
        raise PermissionError("capability task differs")
    registered = [
        row for row in task_registry.get("tasks", ())
        if isinstance(row, Mapping) and row.get("task_id") == expected_task_id
    ]
    if len(registered) != 1 or dict(task) != dict(registered[0]):
        raise PermissionError("capability task is not the authenticated registry row")
    if allowed_kinds is not None and task.get("kind") not in set(allowed_kinds):
        raise PermissionError("capability task kind differs")
    if capability.get("execution_claim_sha256") != claim_sha256:
        raise PermissionError("capability claim differs")
    if capability.get("task_registry_sha256") != registry_sha256:
        raise PermissionError("capability registry differs")
    needs_execution = task.get("kind") in {
        "prediction_shard", "prediction_finalize", "metric_join", "final_aggregate",
    }
    if needs_execution and expected_execution_lock_sha256 is _UNSET:
        raise PermissionError("capability execution lock context is incomplete")
    if not needs_execution and capability.get("execution_lock_sha256") is not None:
        raise PermissionError("capability unexpectedly carries an execution lock")
    if expected_execution_lock_sha256 is not _UNSET:
        expected_execution = None if expected_execution_lock_sha256 is None else require_sha256(expected_execution_lock_sha256, name="execution lock")
        if capability.get("execution_lock_sha256") != expected_execution:
            raise PermissionError("capability execution lock differs")
    if expected_branch_family is not None and task.get("branch_family") != expected_branch_family:
        raise PermissionError("capability branch family differs")
    return digest


def _campaign_task_key(task: Mapping[str, Any]) -> str:
    resource = task.get("resource_signature")
    if not isinstance(resource, Mapping):
        raise ValueError("final task resource signature differs")
    value = str(resource.get("campaign_task_key", task.get("task_id", "")))
    if not value:
        raise ValueError("final task campaign identity differs")
    return value


def _initial_publication_owner(
    task: Mapping[str, Any], task_registry: Mapping[str, Any],
) -> dict[str, Any]:
    campaign_task = _campaign_task_key(task)
    grouped = [
        row for row in task_registry["tasks"]
        if _campaign_task_key(row) == campaign_task
    ]
    matches = [index for index, row in enumerate(grouped) if row["task_id"] == task["task_id"]]
    if len(matches) != 1:
        raise ValueError("final task publication owner axis differs")
    return {
        "campaign_task": campaign_task,
        "array_index": None if len(grouped) == 1 else matches[0],
        "owner_kind": "initial_campaign",
        "campaign_identity_sha256": require_sha256(
            task["resource_signature"].get("campaign_identity_sha256"),
            name="final publication campaign identity",
        ),
    }


def _resolved_output(root: Path, relative: str) -> Path:
    logical = _safe_output(relative)
    base = root.resolve()
    raw = base / Path(logical)
    cursor = base
    for part in PurePosixPath(logical).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("registered final output traverses a symlink")
    candidate = raw.resolve(strict=False)
    try:
        candidate.relative_to(base)
    except ValueError as error:
        raise ValueError("registered final output escapes its output root") from error
    return candidate


def _validate_output_lineage(
    value: Mapping[str, Any], *, task: Mapping[str, Any],
    claim: Mapping[str, Any], task_registry: Mapping[str, Any],
) -> None:
    expected = {
        "population_sha256": claim["population_sha256"],
        "campaign_spec_sha256": claim["campaign_spec_sha256"],
        "execution_claim_sha256": claim["content_hash"],
        "claim_sha256": claim["content_hash"],
        "task_registry_sha256": task_registry["content_hash"],
        "claim_owner_id": claim["claim_owner_id"],
    }
    contexts = [value]
    if isinstance(value.get("payload"), Mapping):
        contexts.append(value["payload"])
    if isinstance(value.get("parents"), Mapping):
        parents = value["parents"]
        parent_expected = {
            "execution_claim": claim["content_hash"],
            "claim": claim["content_hash"],
            "task_registry": task_registry["content_hash"],
            "population": claim["population_sha256"],
            "campaign_spec": claim["campaign_spec_sha256"],
        }
        for name, digest in parent_expected.items():
            if name in parents and parents[name] != digest:
                raise ValueError(f"final output {name} parent lineage differs")
    for context in contexts:
        for name, digest in expected.items():
            if name in context and context[name] != digest:
                raise ValueError(f"final output {name} lineage differs")
        if "task_id" in context and context["task_id"] != task["task_id"]:
            raise ValueError("final output task identity differs")
    if value.get("contract") == FINAL_ROLE_CAPABILITY_CONTRACT:
        authenticated = [
            row for row in task_registry["tasks"]
            if row["task_id"] == task["task_id"]
        ]
        if (
            len(authenticated) != 1
            or value.get("task") != authenticated[0]
            or value.get("execution_claim_sha256") != claim["content_hash"]
            or value.get("task_registry_sha256") != task_registry["content_hash"]
        ):
            raise ValueError("final role capability output lineage differs")


def _expected_output_contract(
    *, task: Mapping[str, Any], registered_output: str, envelope: bool,
) -> str | None:
    logical = _safe_output(registered_output)
    if envelope:
        return {
            "row_selection": "HCWDL_SHARED_FINAL_LABEL_ESCROW/v1",
            "assignment_shard": "HCWDL_SHARED_FINAL_ASSIGNMENT_SHARD/v1",
            "prediction_shard": "HCWDL_REPRESENTATION_PREDICTION_SHARD/v2",
            "metric_join": "HCWDL_REPRESENTATION_PAIRED_BOOTSTRAP/v1",
        }.get(str(task["kind"]))
    if "/capabilities/" in f"/{logical}" or logical.startswith("final/capabilities/"):
        return FINAL_ROLE_CAPABILITY_CONTRACT
    exact_suffixes = (
        ("row_selection.json", "HCWDL_SHARED_FINAL_ROW_SELECTION/v1"),
        ("branch_access.json", "HCWDL_SHARED_FINAL_BRANCH_ACCESS/v1"),
        ("assignment/manifest.json", "HIGHCOV_DENSE_ASSIGNMENT_MANIFEST/v2"),
        ("assignment/audit.json", "HCWDL_SHARED_FINAL_ASSIGNMENT_AUDIT/v1"),
        ("06_final_data_attestation.json", "HCWDL_SHARED_FINAL_DATA_ATTESTATION/v1"),
        ("07_execution.json", "HCWDL_REPRESENTATION_EXECUTION_LOCK/v2"),
        ("prediction_spec.json", "HCWDL_REPRESENTATION_FINAL_PREDICTION_SPEC/v2"),
        ("/manifest.json", "HCWDL_REPRESENTATION_PREDICTION_MANIFEST/v2"),
        ("metric_join.json", "HCWDL_REPRESENTATION_METRIC_JOIN/v2"),
        ("final_aggregate.json", "HCWDL_REPRESENTATION_FINAL_AGGREGATE/v2"),
    )
    for suffix, contract in exact_suffixes:
        if logical.endswith(suffix):
            return contract
    if logical == "final/evaluations" or logical.startswith("final/evaluations/"):
        return "HCWDL_REPRESENTATION_FINAL_EVALUATION/v2"
    return None


def _require_typed_schema(
    value: Mapping[str, Any], fields: set[str], *, label: str,
) -> None:
    if set(value) != fields | {"contract", "schema_version", "content_hash"}:
        raise ValueError(f"{label} schema differs")


def _load_registered_json_by_contract(
    *, root: Path, task_registry: Mapping[str, Any], contract: str,
    finalist_id: str | None = None,
) -> dict[str, Any]:
    matches: list[tuple[Mapping[str, Any], str]] = []
    for registered_task in task_registry["tasks"]:
        if finalist_id is not None and registered_task.get("finalist_id") != finalist_id:
            continue
        for logical in registered_task["registered_outputs"]:
            if _expected_output_contract(
                task=registered_task, registered_output=logical, envelope=False,
            ) == contract:
                matches.append((registered_task, logical))
    if len(matches) != 1:
        raise ValueError(f"registered final parent route differs: {contract}")
    _, logical = matches[0]
    path = _resolved_output(root, logical)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"registered final parent is absent: {contract}")
    value = load_json(path)
    validate_content_hash(
        value, expected_contract=contract,
        expected_schema_version=(
            2 if contract == "HIGHCOV_DENSE_ASSIGNMENT_MANIFEST/v2" else 1
        ),
    )
    return value


def _validate_typed_final_output(
    value: Mapping[str, Any], *, task: Mapping[str, Any],
    claim: Mapping[str, Any], task_registry: Mapping[str, Any],
    expected_contract: str | None, root: Path,
) -> None:
    """Reject structurally valid, self-hashed substitutes for production outputs.

    Content-hash validation alone authenticates bytes, not scientific meaning.
    This validator is deliberately applied only to canonical production routes;
    tests and future non-production routes with no registered contract retain the
    generic versioned-artifact check.
    """

    if expected_contract is None:
        return
    contract = expected_contract
    population = claim["population_sha256"]
    if contract == FINAL_ROLE_CAPABILITY_CONTRACT:
        needs_execution = task["kind"] in {
            "prediction_shard", "prediction_finalize", "metric_join",
            "final_aggregate",
        }
        expected_execution: object = _UNSET
        if not needs_execution:
            expected_execution = None
        else:
            # The lock is an independently registered output.  Never compare
            # the capability's observed lock hash to itself.
            execution_lock = _load_registered_json_by_contract(
                root=root, task_registry=task_registry,
                contract="HCWDL_REPRESENTATION_EXECUTION_LOCK/v2",
            )
            from .hcwdl_representation_final import validate_execution_lock
            expected_execution = validate_execution_lock(
                execution_lock, claim=claim, task_registry=task_registry,
            )
        validate_role_capability(
            value, execution_claim=claim, task_registry=task_registry,
            expected_population_sha256=population,
            expected_task_id=str(task["task_id"]),
            allowed_kinds=(str(task["kind"]),),
            expected_execution_lock_sha256=expected_execution,
            expected_branch_family=task.get("branch_family"),
        )
        return
    if contract == "HCWDL_SHARED_FINAL_ROW_SELECTION/v1":
        _require_typed_schema(value, {
            "population_sha256", "selection_rule_sha256", "capability_sha256",
            "row_count", "class_counts", "identity_order_sha256",
            "identity_digests", "selected_rows", "selection_rank_sha256",
            "labels_sealed_separately", "particle_branches_read",
        }, label="final row selection")
        rows = value.get("row_count")
        classes = value.get("class_counts")
        identities = value.get("identity_digests")
        selected = value.get("selected_rows")
        if (
            value.get("population_sha256") != population
            or value.get("labels_sealed_separately") is not True
            or value.get("particle_branches_read") is not False
            or isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0
            or not isinstance(classes, list) or len(classes) != 15
            or any(isinstance(count, bool) or not isinstance(count, int) or count < 0
                   for count in classes)
            or sum(classes) != rows
            or not isinstance(identities, list) or len(identities) != rows
            or len(set(identities)) != rows
            or not isinstance(selected, list) or len(selected) != rows
        ):
            raise ValueError("final row selection semantics differ")
        for digest in identities:
            require_sha256(digest, name="selected identity")
        for name in (
            "selection_rule_sha256", "capability_sha256",
            "identity_order_sha256", "selection_rank_sha256",
        ):
            require_sha256(value.get(name), name=f"final selection {name}")
        if canonical_sha256(identities) != value["identity_order_sha256"]:
            raise ValueError("final row selection identity order differs")
        return
    if contract == "HCWDL_SHARED_FINAL_BRANCH_ACCESS/v1":
        _require_typed_schema(value, {
            "path", "population_sha256", "task_id", "capability_sha256",
            "execution_lock_sha256", "projected_branches", "sources",
            "label_free",
        }, label="final branch access")
        if (
            value.get("population_sha256") != population
            or value.get("task_id") != task["task_id"]
            or value.get("path") != task.get("branch_family")
            or value.get("label_free") is not (task.get("branch_family") != "selection")
            or not isinstance(value.get("projected_branches"), list)
            or not value["projected_branches"]
            or not isinstance(value.get("sources"), list)
        ):
            raise ValueError("final branch-access semantics differ")
        require_sha256(value.get("capability_sha256"), name="branch-access capability")
        execution = value.get("execution_lock_sha256")
        if task["kind"] in {"row_selection", "assignment_shard"}:
            if execution is not None:
                raise ValueError("pre-execution branch access binds an execution lock")
        else:
            require_sha256(execution, name="branch-access execution lock")
        return
    if contract == "HIGHCOV_DENSE_ASSIGNMENT_MANIFEST/v2":
        _require_typed_schema(value, {
            "role", "expected_mapped_jets", "scanned_mapped_jets",
            "visible_hlt_tokens", "assigned_hlt_tokens", "dustbin_fraction",
            "visible_by_category", "assigned_by_category",
            "unclassified_hlt_tokens", "shards", "parents",
        }, label="final assignment manifest")
        expected_rows = value.get("expected_mapped_jets")
        scanned = value.get("scanned_mapped_jets")
        visible = value.get("visible_hlt_tokens")
        assigned = value.get("assigned_hlt_tokens")
        parents = value.get("parents")
        shards = value.get("shards")
        if (
            value.get("role") != "final_test"
            or isinstance(expected_rows, bool) or not isinstance(expected_rows, int)
            or expected_rows <= 0 or scanned != expected_rows
            or isinstance(visible, bool) or not isinstance(visible, int) or visible <= 0
            or isinstance(assigned, bool) or not isinstance(assigned, int)
            or assigned < 0 or assigned > visible
            or not isinstance(parents, Mapping) or not parents
            or not isinstance(shards, list) or not shards
        ):
            raise ValueError("final assignment manifest semantics differ")
        for name, digest in parents.items():
            if not isinstance(name, str) or not name:
                raise ValueError("final assignment manifest parent name differs")
            require_sha256(digest, name=f"assignment manifest parent {name}")
        return
    if contract == "HCWDL_SHARED_FINAL_ASSIGNMENT_AUDIT/v1":
        from .hcwdl_representation_final import validate_assignment_audit
        validate_assignment_audit(value)
    elif contract == "HCWDL_SHARED_FINAL_DATA_ATTESTATION/v1":
        from .hcwdl_representation_final import validate_final_data_attestation
        validate_final_data_attestation(value)
        if (
            value.get("execution_claim_sha256") != claim["content_hash"]
            or value.get("task_registry_sha256") != task_registry["content_hash"]
        ):
            raise ValueError("final data attestation claim/registry lineage differs")
    elif contract == "HCWDL_REPRESENTATION_EXECUTION_LOCK/v2":
        from .hcwdl_representation_final import validate_execution_lock
        validate_execution_lock(value, claim=claim, task_registry=task_registry)
    elif contract == "HCWDL_REPRESENTATION_FINAL_PREDICTION_SPEC/v2":
        from .hcwdl_representation_final import validate_prediction_spec
        validate_prediction_spec(value)
    elif contract == "HCWDL_REPRESENTATION_PREDICTION_MANIFEST/v2":
        _require_typed_schema(value, {
            "finalist", "prediction_spec_sha256", "execution_lock_sha256",
            "rows", "identity_set_sha256", "identity_order_sha256",
            "array_sha256", "shards", "shard_sha256",
            "complete_disjoint_coverage",
        }, label="final prediction manifest")
        finalist = value.get("finalist")
        if (
            not isinstance(finalist, Mapping)
            or finalist.get("finalist_id") != task.get("finalist_id")
            or value.get("complete_disjoint_coverage") is not True
            or isinstance(value.get("rows"), bool)
            or not isinstance(value.get("rows"), int) or value["rows"] <= 0
            or not isinstance(value.get("shards"), list) or not value["shards"]
        ):
            raise ValueError("final prediction manifest semantics differ")
    elif contract == "HCWDL_REPRESENTATION_FINAL_EVALUATION/v2":
        _require_typed_schema(value, {
            "finalist", "prediction_manifest_sha256", "label_escrow_sha256",
            "execution_lock_sha256", "metrics",
            "metric_runtime_signature_sha256", "test_used_for_selection",
            "joined_identity_order_sha256",
        }, label="final evaluation")
        finalist = value.get("finalist")
        registered_finalists = {
            row.get("finalist_id") for row in task_registry["tasks"]
            if row.get("kind") == "prediction_shard"
        }
        if (
            not isinstance(finalist, Mapping)
            or finalist.get("finalist_id") not in registered_finalists
            or value.get("test_used_for_selection") is not False
            or not isinstance(value.get("metrics"), Mapping)
        ):
            raise ValueError("final evaluation semantics differ")
    elif contract == "HCWDL_REPRESENTATION_METRIC_JOIN/v2":
        _require_typed_schema(value, {
            "execution_lock_sha256", "label_escrow_sha256",
            "prediction_manifests", "evaluation_sha256", "single_label_join",
            "population_sha256", "data_attestation_sha256",
            "finalist_lock_sha256", "prediction_spec_sha256",
            "metric_runtime_signature_sha256", "metric_join_capability_sha256",
            "joined_identity_order_sha256",
        }, label="final metric join")
        if (
            value.get("population_sha256") != population
            or value.get("single_label_join") is not True
            or not isinstance(value.get("prediction_manifests"), Mapping)
            or not value["prediction_manifests"]
            or not isinstance(value.get("evaluation_sha256"), Mapping)
            or set(value["prediction_manifests"]) != set(value["evaluation_sha256"])
        ):
            raise ValueError("final metric-join semantics differ")
    elif contract == "HCWDL_REPRESENTATION_FINAL_AGGREGATE/v2":
        _require_typed_schema(value, {
            "population_sha256", "finalist_lock_sha256", "execution_lock_sha256",
            "metric_join_sha256", "confirmation_aggregate_sha256",
            "evaluation_sha256", "paired_bootstrap_envelopes",
            "paired_comparison_registry", "paired_comparison_registry_sha256",
            "joined_identity_order_sha256", "test_used_for_selection",
            "complete_parent_union_evaluated",
        }, label="final aggregate")
        comparisons = value.get("paired_comparison_registry")
        if (
            value.get("population_sha256") != population
            or value.get("test_used_for_selection") is not False
            or value.get("complete_parent_union_evaluated") is not True
            or not isinstance(comparisons, list) or not comparisons
            or canonical_sha256(comparisons)
            != value.get("paired_comparison_registry_sha256")
        ):
            raise ValueError("final aggregate semantics differ")
    else:
        raise ValueError(f"production final output validator is absent: {contract}")
    if "population_sha256" in value and value["population_sha256"] != population:
        raise ValueError("final output population lineage differs")
    for name, digest in value.items():
        if name.endswith("_sha256") and name not in {"content_hash"}:
            require_sha256(digest, name=f"final output {name}")


def _validate_json_output(
    path: Path, *, task: Mapping[str, Any], claim: Mapping[str, Any],
    task_registry: Mapping[str, Any], expected_contract: str | None,
    root: Path,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("registered final JSON output is not a regular file")
    value = load_json(path)
    contract = value.get("contract")
    if not isinstance(contract, str) or not contract:
        raise ValueError("registered final JSON output lacks a contract")
    if expected_contract is not None and contract != expected_contract:
        raise ValueError("registered final output contract differs")
    content_hash = validate_content_hash(
        value, expected_contract=contract,
        expected_schema_version=(
            2 if contract == "HIGHCOV_DENSE_ASSIGNMENT_MANIFEST/v2" else 1
        ),
    )
    _validate_typed_final_output(
        value, task=task, claim=claim, task_registry=task_registry,
        expected_contract=expected_contract, root=root,
    )
    _validate_output_lineage(
        value, task=task, claim=claim, task_registry=task_registry,
    )
    return {
        "relative_member": path.name,
        "byte_sha256": sha256_file(path),
        "content_hash": content_hash,
        "contract": contract,
    }


def _canonical_final_route(registered_output: str) -> bool:
    logical = _safe_output(registered_output)
    return logical.startswith(("final/", "locks/", "reports/"))


def _registered_task_capability(
    *, root: Path, task: Mapping[str, Any], claim: Mapping[str, Any],
    task_registry: Mapping[str, Any],
) -> dict[str, Any]:
    matches = [
        logical for logical in task["registered_outputs"]
        if _expected_output_contract(
            task=task, registered_output=logical, envelope=False,
        ) == FINAL_ROLE_CAPABILITY_CONTRACT
    ]
    if len(matches) != 1:
        raise ValueError("final task capability route differs")
    path = _resolved_output(root, matches[0])
    value = load_json(path)
    _validate_json_output(
        path, task=task, claim=claim, task_registry=task_registry,
        expected_contract=FINAL_ROLE_CAPABILITY_CONTRACT, root=root,
    )
    return value


def _sole_registered_envelope_sidecar(
    *, root: Path, task_registry: Mapping[str, Any], kind: str,
    contract: str,
) -> dict[str, Any]:
    candidates = [
        (task, logical)
        for task in task_registry["tasks"] if task["kind"] == kind
        for logical in task["registered_outputs"]
        if _expected_output_contract(
            task=task, registered_output=logical, envelope=True,
        ) == contract
    ]
    if len(candidates) != 1:
        raise ValueError("registered final envelope parent route differs")
    _, logical = candidates[0]
    envelope_root = _resolved_output(root, logical)
    committed = envelope_root / "committed"
    leaves = sorted(committed.iterdir()) if committed.is_dir() else []
    if (
        len(leaves) != 1 or leaves[0].is_symlink()
        or not leaves[0].is_dir()
    ):
        raise FileNotFoundError("registered final envelope parent is absent")
    sidecar = load_json(leaves[0] / "sidecar.json")
    validate_content_hash(sidecar, expected_contract=contract, expected_schema_version=1)
    return sidecar


def _expected_final_envelope_parents(
    *, directory: Path, task: Mapping[str, Any], claim: Mapping[str, Any],
    task_registry: Mapping[str, Any], root: Path, artifact_contract: str,
) -> dict[str, str]:
    if artifact_contract == "HCWDL_SHARED_FINAL_LABEL_ESCROW/v1":
        selection = _load_registered_json_by_contract(
            root=root, task_registry=task_registry,
            contract="HCWDL_SHARED_FINAL_ROW_SELECTION/v1",
        )
        capability = _registered_task_capability(
            root=root, task=task, claim=claim, task_registry=task_registry,
        )
        return {
            "selection": selection["content_hash"],
            "population": claim["population_sha256"],
            "capability": capability["content_hash"],
        }
    if artifact_contract == "HCWDL_SHARED_FINAL_ASSIGNMENT_SHARD/v1":
        resource = task.get("resource_signature")
        if not isinstance(resource, Mapping):
            raise ValueError("assignment task resource signature differs")
        selection = _load_registered_json_by_contract(
            root=root, task_registry=task_registry,
            contract="HCWDL_SHARED_FINAL_ROW_SELECTION/v1",
        )
        capability = _registered_task_capability(
            root=root, task=task, claim=claim, task_registry=task_registry,
        )
        branch = load_json(directory / "branch_access.json")
        validate_content_hash(
            branch, expected_contract="HCWDL_SHARED_FINAL_BRANCH_ACCESS/v1",
            expected_schema_version=1,
        )
        _validate_typed_final_output(
            branch, task=task, claim=claim, task_registry=task_registry,
            expected_contract="HCWDL_SHARED_FINAL_BRANCH_ACCESS/v1", root=root,
        )
        return {
            "split_manifest": require_sha256(
                resource.get("split_manifest_sha256"),
                name="assignment split manifest",
            ),
            "selection": selection["content_hash"],
            "matcher_resources": require_sha256(
                resource.get("matcher_resources_sha256"),
                name="assignment matcher resources",
            ),
            "capability": capability["content_hash"],
            "branch_access": branch["content_hash"],
        }
    if artifact_contract == "HCWDL_REPRESENTATION_PREDICTION_SHARD/v2":
        prediction_spec = _load_registered_json_by_contract(
            root=root, task_registry=task_registry,
            contract="HCWDL_REPRESENTATION_FINAL_PREDICTION_SPEC/v2",
        )
        execution_lock = _load_registered_json_by_contract(
            root=root, task_registry=task_registry,
            contract="HCWDL_REPRESENTATION_EXECUTION_LOCK/v2",
        )
        branch = load_json(directory / "branch_access.json")
        validate_content_hash(
            branch, expected_contract="HCWDL_SHARED_FINAL_BRANCH_ACCESS/v1",
            expected_schema_version=1,
        )
        _validate_typed_final_output(
            branch, task=task, claim=claim, task_registry=task_registry,
            expected_contract="HCWDL_SHARED_FINAL_BRANCH_ACCESS/v1", root=root,
        )
        return {
            "prediction_spec": prediction_spec["content_hash"],
            "execution_lock": execution_lock["content_hash"],
            "checkpoint": require_sha256(
                task.get("checkpoint_sha256"), name="prediction checkpoint",
            ),
            "branch_access": branch["content_hash"],
        }
    if artifact_contract == "HCWDL_REPRESENTATION_PAIRED_BOOTSTRAP/v1":
        metric_join = _load_registered_json_by_contract(
            root=root, task_registry=task_registry,
            contract="HCWDL_REPRESENTATION_METRIC_JOIN/v2",
        )
        escrow = _sole_registered_envelope_sidecar(
            root=root, task_registry=task_registry, kind="row_selection",
            contract="HCWDL_SHARED_FINAL_LABEL_ESCROW/v1",
        )
        sidecar = load_json(directory / "sidecar.json")
        payload = sidecar.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("paired-bootstrap sidecar payload differs")
        left_id, right_id = str(payload.get("left_id", "")), str(payload.get("right_id", ""))
        if not left_id or not right_id or left_id == right_id:
            raise ValueError("paired-bootstrap comparison identity differs")
        left = _load_registered_json_by_contract(
            root=root, task_registry=task_registry,
            contract="HCWDL_REPRESENTATION_PREDICTION_MANIFEST/v2",
            finalist_id=left_id,
        )
        right = _load_registered_json_by_contract(
            root=root, task_registry=task_registry,
            contract="HCWDL_REPRESENTATION_PREDICTION_MANIFEST/v2",
            finalist_id=right_id,
        )
        return {
            "metric_join": metric_join["content_hash"],
            "label_escrow": escrow["content_hash"],
            "left_prediction_manifest": left["content_hash"],
            "right_prediction_manifest": right["content_hash"],
        }
    raise ValueError("production final envelope parent validator is absent")


def _validate_envelope_directory(
    directory: Path, *, staged: bool, task: Mapping[str, Any],
    claim: Mapping[str, Any], task_registry: Mapping[str, Any],
    expected_contract: str | None, root: Path, registered_output: str,
) -> dict[str, Any]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("final envelope directory is not a regular directory")
    identity = directory.parent.name if staged else directory.name
    owner_from_path = directory.name if staged else None
    require_sha256(identity, name="final envelope ID")
    expected_publication_owner = _initial_publication_owner(task, task_registry)
    if staged:
        expected_owner_id = derive_envelope_owner_id(
            envelope_id=identity,
            campaign_or_recovery_owner=expected_publication_owner,
        )
        if owner_from_path != expected_owner_id:
            raise PermissionError("foreign final staging owner cannot be cleaned")
        commit_path = directory / "commit.json"
        if not commit_path.is_file():
            members = []
            for member in sorted(directory.rglob("*")):
                if member.is_symlink() or not member.is_file():
                    raise ValueError("incomplete final staging inventory differs")
                members.append({
                    "path": member.relative_to(directory).as_posix(),
                    "byte_sha256": sha256_file(member),
                })
            return {
                "envelope_id": identity,
                "envelope_owner_id": expected_owner_id,
                "artifact_contract": expected_contract,
                "commit_sha256": None, "sidecar_sha256": None,
                "member_inventory_sha256": canonical_sha256(members),
            }
    commit = load_json(directory / "commit.json")
    validate_versioned_artifact(
        commit, expected_contract=SHARED_BINARY_ENVELOPE_CONTRACT,
        required_payload_keys=(
            "envelope_id", "envelope_owner_id", "artifact_contract",
            "producer_task_id", "schema", "registered_output_row",
            "campaign_or_recovery_owner", "members",
        ),
    )
    payload = commit["payload"]
    parents = validate_parent_hashes(commit["parents"])
    envelope_id = require_sha256(payload.get("envelope_id"), name="final envelope ID")
    owner_id = require_sha256(
        payload.get("envelope_owner_id"), name="final envelope owner ID",
    )
    if envelope_id != identity or (owner_from_path is not None and owner_id != owner_from_path):
        raise ValueError("final envelope path identity differs")
    producer = str(payload.get("producer_task_id", ""))
    if producer != task["task_id"] and not producer.startswith(f"{task['task_id']}:"):
        raise ValueError("final envelope producer differs from task registry")
    output_row = payload.get("registered_output_row")
    publication_owner = payload.get("campaign_or_recovery_owner")
    campaign_task = _campaign_task_key(task)
    if task["kind"] == "metric_join":
        output_identity_valid = (
            isinstance(output_row, Mapping)
            and set(output_row) == {"comparison_id", "task_id"}
            and output_row.get("task_id") == task["task_id"]
            and isinstance(output_row.get("comparison_id"), str)
            and bool(output_row.get("comparison_id"))
        )
    else:
        output_identity_valid = (
            isinstance(output_row, Mapping)
            and output_row.get("task_key") == campaign_task
            and output_row.get("array_index")
            == expected_publication_owner["array_index"]
        )
    if (
        not output_identity_valid
        or not isinstance(publication_owner, Mapping)
        or dict(publication_owner) != expected_publication_owner
    ):
        raise ValueError("final envelope registered publication owner differs")
    artifact_contract = str(payload.get("artifact_contract", ""))
    if expected_contract is not None and artifact_contract != expected_contract:
        raise ValueError("final envelope artifact contract differs")
    if _canonical_final_route(registered_output):
        expected_parents = _expected_final_envelope_parents(
            directory=directory, task=task, claim=claim,
            task_registry=task_registry, root=root,
            artifact_contract=artifact_contract,
        )
        if parents != expected_parents:
            raise ValueError("final envelope frozen parent lineage differs")
    derived_id = derive_envelope_id(
        contract=artifact_contract, producer_task_id=producer,
        schema=payload.get("schema", {}), immutable_parent_hashes=parents,
        registered_output_row=output_row,
    )
    derived_owner = derive_envelope_owner_id(
        envelope_id=envelope_id,
        campaign_or_recovery_owner=publication_owner,
    )
    if derived_id != envelope_id or derived_owner != owner_id:
        raise ValueError("final envelope identity is not derivable")
    sidecar = load_json(directory / "sidecar.json")
    validate_versioned_artifact(
        sidecar, expected_contract=artifact_contract, expected_parents=parents,
        required_payload_keys=(
            "envelope_id", "envelope_owner_id", "producer_task_id", "schema",
            "registered_output_row", "campaign_or_recovery_owner",
            "payload_members",
        ),
    )
    for name in (
        "envelope_id", "envelope_owner_id", "producer_task_id", "schema",
        "registered_output_row", "campaign_or_recovery_owner",
    ):
        if sidecar["payload"].get(name) != payload.get(name):
            raise ValueError("final envelope sidecar identity differs")
    _validate_output_lineage(
        sidecar, task=task, claim=claim, task_registry=task_registry,
    )
    members = payload.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("final envelope member registry is empty")
    names: set[str] = set()
    records = []
    for raw in members:
        if not isinstance(raw, Mapping) or set(raw) != {
            "path", "byte_sha256", "logical_sha256", "dtype", "shape",
            "immutable_parent_hashes",
        }:
            raise ValueError("final envelope member row differs")
        name = _safe_output(str(raw["path"]))
        if name == "commit.json" or name in names:
            raise ValueError("final envelope member path differs")
        names.add(name)
        member = (directory / Path(name)).resolve(strict=False)
        try:
            member.relative_to(directory.resolve())
        except ValueError as error:
            raise ValueError("final envelope member escapes its directory") from error
        if member.is_symlink() or not member.is_file():
            raise ValueError("final envelope member is absent or not regular")
        byte_hash = require_sha256(raw["byte_sha256"], name=f"final member {name}")
        require_sha256(raw["logical_sha256"], name=f"final logical member {name}")
        if sha256_file(member) != byte_hash:
            raise ValueError("final envelope member bytes differ")
        if validate_parent_hashes(raw["immutable_parent_hashes"]) != parents:
            raise ValueError("final envelope member parents differ")
        records.append({"path": name, "byte_sha256": byte_hash})
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*") if path.is_file()
    }
    if actual != names | {"commit.json"}:
        raise ValueError("final envelope contains unregistered members")
    return {
        "envelope_id": envelope_id, "envelope_owner_id": owner_id,
        "artifact_contract": artifact_contract,
        "commit_sha256": commit["content_hash"],
        "sidecar_sha256": sidecar["content_hash"],
        "member_inventory_sha256": canonical_sha256(records),
    }


def _audit_registered_output(
    *, root: Path, task: Mapping[str, Any], registered_output: str,
    claim: Mapping[str, Any], task_registry: Mapping[str, Any],
) -> dict[str, Any]:
    path = _resolved_output(root, registered_output)
    expected_json_contract = _expected_output_contract(
        task=task, registered_output=registered_output, envelope=False,
    )
    expected_envelope_contract = _expected_output_contract(
        task=task, registered_output=registered_output, envelope=True,
    )
    base = {
        "task_id": task["task_id"], "kind": task["kind"],
        "registered_output": _safe_output(registered_output),
    }
    try:
        if path.is_symlink():
            raise ValueError("registered final output is a symlink")
        if not path.exists():
            temporary = sorted(
                item
                for item in path.parent.glob(f".{path.name}.*.tmp")
                if item.is_file() or item.is_symlink()
            ) if path.parent.is_dir() else []
            if temporary:
                if any(item.is_symlink() or not item.is_file() for item in temporary):
                    raise ValueError("atomic final temporary inventory differs")
                orphan_staging = [{
                    "orphan_kind": "atomic_file",
                    "path": item.relative_to(root).as_posix(),
                    "byte_sha256": sha256_file(item),
                } for item in temporary]
                return {
                    **base, "status": "orphan_staging", "inventory": [],
                    "orphan_staging": orphan_staging, "validation_error": None,
                    "inventory_sha256": canonical_sha256([]),
                }
            return {
                **base, "status": "absent", "inventory": [],
                "orphan_staging": [], "validation_error": None,
                "inventory_sha256": canonical_sha256([]),
            }
        if path.is_file():
            inventory = [_validate_json_output(
                path, task=task, claim=claim, task_registry=task_registry,
                expected_contract=expected_json_contract, root=root,
            )]
            temporary = sorted(path.parent.glob(f".{path.name}.*.tmp"))
            if any(item.is_symlink() or not item.is_file() for item in temporary):
                raise ValueError("atomic final temporary inventory differs")
            orphan_staging = [{
                "orphan_kind": "atomic_file",
                "path": item.relative_to(root).as_posix(),
                "byte_sha256": sha256_file(item),
            } for item in temporary]
            return {
                **base, "status": "valid", "inventory": inventory,
                "orphan_staging": orphan_staging, "validation_error": None,
                "inventory_sha256": canonical_sha256(inventory),
            }
        if not path.is_dir():
            raise ValueError("registered final output has an unsupported file type")
        if any(item.is_symlink() for item in path.rglob("*")):
            raise ValueError("registered final directory contains a symlink")
        committed_containers = sorted(
            item for item in path.rglob("committed") if item.is_dir()
        )
        staging_containers = sorted(
            item for item in path.rglob("staging") if item.is_dir()
        )
        inventory: list[dict[str, Any]] = []
        committed_files: set[Path] = set()
        for container in committed_containers:
            children = sorted(container.iterdir())
            if not children or any(not child.is_dir() for child in children):
                raise ValueError("committed final envelope inventory differs")
            for child in children:
                record = _validate_envelope_directory(
                    child, staged=False, task=task, claim=claim,
                    task_registry=task_registry,
                    expected_contract=expected_envelope_contract,
                    root=root, registered_output=registered_output,
                )
                inventory.append({
                    "relative_member": child.relative_to(path).as_posix(), **record,
                })
                committed_files.update(item for item in child.rglob("*") if item.is_file())
        orphan_staging: list[dict[str, Any]] = []
        staging_files: set[Path] = set()
        for container in staging_containers:
            for envelope in sorted(container.iterdir()):
                if not envelope.is_dir():
                    raise ValueError("final staging envelope inventory differs")
                for owner in sorted(envelope.iterdir()):
                    record = _validate_envelope_directory(
                        owner, staged=True, task=task, claim=claim,
                        task_registry=task_registry,
                        expected_contract=expected_envelope_contract,
                        root=root, registered_output=registered_output,
                    )
                    orphan_staging.append({
                        "orphan_kind": "envelope_owner",
                        "path": owner.relative_to(root).as_posix(), **record,
                    })
                    staging_files.update(item for item in owner.rglob("*") if item.is_file())
        all_files = {item for item in path.rglob("*") if item.is_file()}
        plain_files = sorted(all_files - committed_files - staging_files)
        if committed_containers and plain_files:
            raise ValueError("final envelope root contains files outside atomic trees")
        if not committed_containers and plain_files:
            for item in plain_files:
                record = _validate_json_output(
                    item, task=task, claim=claim, task_registry=task_registry,
                    expected_contract=expected_json_contract, root=root,
                )
                record["relative_member"] = item.relative_to(path).as_posix()
                inventory.append(record)
        if inventory:
            status = "valid"
        elif orphan_staging:
            status = "orphan_staging"
        else:
            raise ValueError("registered final output directory is empty")
        return {
            **base, "status": status, "inventory": inventory,
            "orphan_staging": orphan_staging, "validation_error": None,
            "inventory_sha256": canonical_sha256(inventory),
        }
    except (OSError, TypeError, ValueError) as error:
        return {
            **base, "status": "corrupt", "inventory": [],
            "orphan_staging": [],
            "validation_error": f"{type(error).__name__}: {error}",
            "inventory_sha256": canonical_sha256([]),
        }


def _task_audit_statuses(
    outputs: Sequence[Mapping[str, Any]], task_registry: Mapping[str, Any],
) -> list[dict[str, str]]:
    by_task: dict[str, list[str]] = {}
    for row in outputs:
        by_task.setdefault(str(row["task_id"]), []).append(str(row["status"]))
    result = []
    for task in task_registry["tasks"]:
        statuses = by_task.get(task["task_id"], [])
        if len(statuses) != len(task["registered_outputs"]):
            raise ValueError("shared-final audit output coverage differs")
        if "corrupt" in statuses:
            status = "corrupt"
        elif all(value == "valid" for value in statuses):
            status = "valid"
        elif "orphan_staging" in statuses:
            status = "recoverable_orphan_staging"
        else:
            status = "recoverable_absent"
        result.append({"task_id": task["task_id"], "status": status})
    return result


def audit_shared_final_outputs(
    *, output_root: str | Path, claim: Mapping[str, Any],
    task_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive every recovery decision from registered filesystem outputs."""

    claim_hash = validate_execution_claim(claim, task_registry=task_registry)
    registry_hash = validate_final_task_registry(task_registry)
    root = Path(output_root).resolve()
    outputs = [
        _audit_registered_output(
            root=root, task=task, registered_output=logical,
            claim=claim, task_registry=task_registry,
        )
        for task in task_registry["tasks"]
        for logical in task["registered_outputs"]
    ]
    audit = {
        "output_root": str(root),
        "population_sha256": claim["population_sha256"],
        "execution_claim_sha256": claim_hash,
        "task_registry_sha256": registry_hash,
        "claim_owner_id": claim["claim_owner_id"],
        "outputs": outputs,
        "task_statuses": _task_audit_statuses(outputs, task_registry),
        "all_registered_outputs_audited": True,
    }
    return {**audit, "audit_sha256": canonical_sha256(audit)}


def validate_shared_final_output_audit(
    value: Mapping[str, Any], *, claim: Mapping[str, Any],
    task_registry: Mapping[str, Any], output_root: str | Path | None = None,
) -> str:
    claim_hash = validate_execution_claim(claim, task_registry=task_registry)
    registry_hash = validate_final_task_registry(task_registry)
    if set(value) != {
        "output_root", "population_sha256", "execution_claim_sha256",
        "task_registry_sha256", "claim_owner_id", "outputs", "task_statuses",
        "all_registered_outputs_audited", "audit_sha256",
    }:
        raise ValueError("shared-final output audit schema differs")
    expected_pairs = [
        (task["task_id"], logical)
        for task in task_registry["tasks"]
        for logical in task["registered_outputs"]
    ]
    outputs = value.get("outputs")
    if not isinstance(outputs, list) or [
        (row.get("task_id"), row.get("registered_output"))
        for row in outputs if isinstance(row, Mapping)
    ] != expected_pairs:
        raise ValueError("shared-final output audit coverage differs")
    for row in outputs:
        if set(row) != {
            "task_id", "kind", "registered_output", "status", "inventory",
            "orphan_staging", "validation_error", "inventory_sha256",
        } or row["status"] not in {"valid", "absent", "orphan_staging", "corrupt"}:
            raise ValueError("shared-final output audit row differs")
        task = next(
            task for task in task_registry["tasks"]
            if task["task_id"] == row["task_id"]
        )
        if row["kind"] != task["kind"] or not isinstance(row["inventory"], list):
            raise ValueError("shared-final output audit task identity differs")
        if canonical_sha256(row["inventory"]) != row["inventory_sha256"]:
            raise ValueError("shared-final output inventory hash differs")
        if not isinstance(row["orphan_staging"], list):
            raise ValueError("shared-final orphan staging inventory differs")
        for orphan in row["orphan_staging"]:
            _safe_output(str(orphan.get("path", "")))
            if orphan.get("orphan_kind") == "atomic_file":
                if set(orphan) != {"orphan_kind", "path", "byte_sha256"}:
                    raise ValueError("atomic final orphan audit row differs")
                require_sha256(orphan["byte_sha256"], name="atomic final orphan bytes")
            elif orphan.get("orphan_kind") == "envelope_owner":
                if set(orphan) != {
                    "orphan_kind", "path", "envelope_id", "envelope_owner_id",
                    "artifact_contract", "commit_sha256", "sidecar_sha256",
                    "member_inventory_sha256",
                }:
                    raise ValueError("envelope final orphan audit row differs")
                for name in (
                    "envelope_id", "envelope_owner_id", "member_inventory_sha256",
                ):
                    require_sha256(orphan[name], name=f"final orphan {name}")
                for name in ("commit_sha256", "sidecar_sha256"):
                    if orphan[name] is not None:
                        require_sha256(orphan[name], name=f"final orphan {name}")
            else:
                raise ValueError("shared-final orphan staging kind differs")
        status = row["status"]
        if (
            (status == "absent" and (
                row["inventory"] or row["orphan_staging"] or row["validation_error"] is not None
            ))
            or (status == "orphan_staging" and (
                row["inventory"] or not row["orphan_staging"]
                or row["validation_error"] is not None
            ))
            or (status == "valid" and (
                not row["inventory"] or row["validation_error"] is not None
            ))
            or (status == "corrupt" and (
                row["inventory"] or row["orphan_staging"]
                or not isinstance(row["validation_error"], str)
                or not row["validation_error"]
            ))
        ):
            raise ValueError("shared-final output audit status semantics differ")
    expected_statuses = _task_audit_statuses(outputs, task_registry)
    unhashed = dict(value); supplied = unhashed.pop("audit_sha256", None)
    if (
        value.get("population_sha256") != claim["population_sha256"]
        or value.get("execution_claim_sha256") != claim_hash
        or value.get("task_registry_sha256") != registry_hash
        or value.get("claim_owner_id") != claim["claim_owner_id"]
        or value.get("task_statuses") != expected_statuses
        or value.get("all_registered_outputs_audited") is not True
        or canonical_sha256(unhashed) != supplied
    ):
        raise ValueError("shared-final output audit lineage or semantics differ")
    require_sha256(supplied, name="shared-final output audit")
    if output_root is not None:
        fresh = audit_shared_final_outputs(
            output_root=output_root, claim=claim, task_registry=task_registry,
        )
        if dict(value) != fresh:
            raise ValueError("shared-final output audit differs from filesystem")
    return supplied


def _recovery_plan_from_audit(
    *, claim: Mapping[str, Any], task_registry: Mapping[str, Any],
    output_audit: Mapping[str, Any], recovery_owner_id: str,
) -> dict[str, Any]:
    statuses = {
        row["task_id"]: row["status"] for row in output_audit["task_statuses"]
    }
    if "corrupt" in statuses.values():
        raise ValueError("published corrupt final output cannot be overwritten")
    owner = require_sha256(recovery_owner_id, name="recovery owner")
    if owner != claim["claim_owner_id"]:
        raise PermissionError("shared-final recovery owner differs from execution claim")
    rerun = [
        row["task_id"] for row in task_registry["tasks"]
        if statuses[row["task_id"]] in {
            "recoverable_absent", "recoverable_orphan_staging",
        }
    ]
    orphans = [
        {
            "task_id": row["task_id"],
            "registered_output": row["registered_output"],
            **orphan,
        }
        for row in output_audit["outputs"]
        for orphan in row["orphan_staging"]
    ]
    return with_content_hash({
        "contract": FINAL_RECOVERY_PLAN_CONTRACT, "schema_version": 1,
        "population_sha256": claim["population_sha256"],
        "execution_claim_sha256": claim["content_hash"],
        "task_registry_sha256": task_registry["content_hash"],
        "claim_owner_id": claim["claim_owner_id"],
        "recovery_owner_id": owner,
        "same_claim_owner": True,
        "output_root": output_audit["output_root"],
        "output_audit_sha256": output_audit["audit_sha256"],
        "output_audit": dict(output_audit),
        "orphan_staging": orphans,
        "resubmit_task_ids": rerun,
        "adds_no_tasks_or_outputs": True,
    })


def build_shared_final_recovery_plan(
    *, claim: Mapping[str, Any], task_registry: Mapping[str, Any],
    recovery_owner_id: str, output_root: str | Path | None = None,
    output_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_execution_claim(claim, task_registry=task_registry)
    if output_audit is None:
        if output_root is None:
            raise ValueError("shared-final recovery requires a filesystem output root")
        audit = audit_shared_final_outputs(
            output_root=output_root, claim=claim, task_registry=task_registry,
        )
    else:
        audit = dict(output_audit)
        audit_root = (
            output_root if output_root is not None else audit.get("output_root")
        )
        if audit_root is None:
            raise ValueError("shared-final output audit lacks its filesystem root")
        validate_shared_final_output_audit(
            audit, claim=claim, task_registry=task_registry,
            output_root=audit_root,
        )
    return _recovery_plan_from_audit(
        claim=claim, task_registry=task_registry, output_audit=audit,
        recovery_owner_id=recovery_owner_id,
    )


def validate_shared_final_recovery_plan(
    value: Mapping[str, Any], *, claim: Mapping[str, Any],
    task_registry: Mapping[str, Any], output_root: str | Path | None = None,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=FINAL_RECOVERY_PLAN_CONTRACT,
        expected_schema_version=1,
    )
    if set(value) != {
        "contract", "schema_version", "population_sha256",
        "execution_claim_sha256", "task_registry_sha256", "claim_owner_id",
        "recovery_owner_id", "same_claim_owner", "output_root",
        "output_audit_sha256", "output_audit", "orphan_staging",
        "resubmit_task_ids", "adds_no_tasks_or_outputs", "content_hash",
    }:
        raise ValueError("shared-final recovery plan schema differs")
    audit = value.get("output_audit")
    if not isinstance(audit, Mapping):
        raise ValueError("shared-final recovery plan lacks its output audit")
    audit_hash = validate_shared_final_output_audit(
        audit, claim=claim, task_registry=task_registry,
        output_root=output_root,
    )
    expected = _recovery_plan_from_audit(
        claim=claim, task_registry=task_registry, output_audit=audit,
        recovery_owner_id=str(value.get("recovery_owner_id", "")),
    )
    if value.get("output_audit_sha256") != audit_hash or dict(value) != expected:
        raise ValueError("shared-final recovery plan canonical semantics differ")
    return digest


def shared_final_recovery_rows(
    plan: Mapping[str, Any], *, claim: Mapping[str, Any],
    task_registry: Mapping[str, Any], output_root: str | Path | None = None,
) -> tuple[dict[str, Any], ...]:
    validate_shared_final_recovery_plan(
        plan, claim=claim, task_registry=task_registry, output_root=output_root,
    )
    retry = set(plan["resubmit_task_ids"])
    return tuple(dict(row) for row in task_registry["tasks"] if row["task_id"] in retry)


def authorize_shared_final_recovery_dispatch(
    plan: Mapping[str, Any], *, claim: Mapping[str, Any],
    task_registry: Mapping[str, Any], campaign_task_key: str,
    array_index: int | None, output_root: str | Path | None = None,
) -> dict[str, Any]:
    if output_root is None:
        raise ValueError(
            "shared-final recovery dispatch requires a fresh filesystem audit"
        )
    retry = shared_final_recovery_rows(
        plan, claim=claim, task_registry=task_registry, output_root=output_root,
    )
    registered = [
        row for row in task_registry["tasks"]
        if _campaign_task_key(row) == campaign_task_key
    ]
    if not registered:
        raise PermissionError("campaign task is absent from shared-final registry")
    if len(registered) == 1:
        if array_index is not None:
            raise PermissionError("non-array shared-final recovery row has an array index")
        selected = registered[0]
    else:
        if (
            isinstance(array_index, bool) or not isinstance(array_index, int)
            or array_index < 0 or array_index >= len(registered)
        ):
            raise PermissionError("shared-final recovery array index differs")
        selected = registered[array_index]
    retry_ids = {row["task_id"] for row in retry}
    if selected["task_id"] not in retry_ids:
        raise PermissionError("shared-final task output is not recoverable/absent")
    return dict(selected)


def cleanup_shared_final_orphan_staging(
    plan: Mapping[str, Any], *, claim: Mapping[str, Any],
    task_registry: Mapping[str, Any], output_root: str | Path,
) -> tuple[str, ...]:
    """Delete only owner directories authenticated by the immutable plan."""

    validate_shared_final_recovery_plan(
        plan, claim=claim, task_registry=task_registry, output_root=output_root,
    )
    root = Path(output_root).resolve()
    tasks = {row["task_id"]: row for row in task_registry["tasks"]}
    removed = []
    for row in plan["orphan_staging"]:
        task = tasks.get(row["task_id"])
        if task is None:
            raise ValueError("orphan cleanup task is unregistered")
        path = _resolved_output(root, row["path"])
        if row.get("orphan_kind") == "atomic_file":
            destination = _resolved_output(root, row["registered_output"])
            if (
                path.parent != destination.parent
                or not path.name.startswith(f".{destination.name}.")
                or not path.name.endswith(".tmp")
                or path.is_symlink() or not path.is_file()
                or sha256_file(path) != row.get("byte_sha256")
            ):
                raise ValueError("atomic final orphan changed after recovery audit")
            path.unlink()
            removed.append(row["path"])
            continue
        if row.get("orphan_kind") != "envelope_owner":
            raise ValueError("unknown shared-final orphan staging kind")
        record = _validate_envelope_directory(
            path, staged=True, task=task, claim=claim,
            task_registry=task_registry,
            expected_contract=_expected_output_contract(
                task=task, registered_output=row["registered_output"], envelope=True,
            ),
            root=root, registered_output=row["registered_output"],
        )
        if any(record.get(name) != row.get(name) for name in (
            "envelope_id", "envelope_owner_id", "artifact_contract",
            "commit_sha256", "sidecar_sha256", "member_inventory_sha256",
        )):
            raise ValueError("orphan staging bytes changed after recovery audit")
        shutil.rmtree(path)
        removed.append(row["path"])
        parent = path.parent
        while parent != root and parent.name not in {"staging", "committed"}:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    return tuple(removed)


__all__ = [name for name in globals() if name.endswith("_CONTRACT")] + [
    "ALLOWED_DISPOSITIONS", "BRANCH_FAMILIES_BY_KIND", "FINAL_TASK_KINDS",
    "audit_parent_final_state", "audit_shared_final_outputs",
    "authorize_shared_final_recovery_dispatch",
    "build_final_disposition", "build_final_population",
    "build_final_task_registry", "build_legacy_cancellation",
    "build_shared_final_recovery_plan", "claim_final_execution",
    "claim_legacy_final_exposure", "cleanup_shared_final_orphan_staging",
    "derive_claim_owner_id", "derive_registration_owner_id", "issue_role_capability",
    "population_root", "register_final_population",
    "reject_legacy_final_after_shared_reservation", "shared_reservations",
    "shared_final_recovery_rows",
    "validate_execution_claim", "validate_final_task_registry",
    "validate_parent_final_state", "validate_role_capability",
    "validate_shared_final_output_audit", "validate_shared_final_recovery_plan",
]

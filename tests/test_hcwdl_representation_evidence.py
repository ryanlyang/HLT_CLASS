from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shlex

import pytest

from hlt_classification.data.cache_contracts import canonical_json_bytes, with_content_hash
from hlt_classification.scouting import hcwdl_representation_resources as resources_module
from hlt_classification.scouting.hcwdl_representation_campaign import (
    REQUIRED_TIGRIS_CHECKS,
    TIGRIS_ACCEPTANCE_CONTRACT,
    TIGRIS_EVIDENCE_BUNDLE_CONTRACT,
    create_campaign_spec,
    validate_campaign_spec,
    validate_tigris_acceptance,
)
from hlt_classification.scouting.hcwdl_representation_resources import (
    MINIATURE_EVIDENCE_CONTRACT,
    RESOURCE_PROFILE_CONTRACT,
    SCHEDULER_EVIDENCE_CONTRACT,
    TIGRIS_ACCOUNT,
    TIGRIS_PARTITION,
    TIGRIS_SITE,
    artifact_reference,
    build_fixed_size_inventory, build_measured_profile,
    build_miniature_evidence, build_scheduler_evidence,
    build_scheduler_evidence_from_sacct,
    build_storage_estimate,
    resource_table,
    scheduler_evidence_comment,
    validate_measured_profile,
    validate_storage_estimate,
)


SOURCE = "1" * 40
PARENT = "2" * 64
RECIPE = "3" * 64


@pytest.fixture(autouse=True)
def _simulated_tigris_sacct_collector(monkeypatch) -> None:
    """Simulate the site-only collector; these bytes are never real acceptance."""

    monkeypatch.setattr(
        resources_module,
        "_live_tigris_capture_runtime",
        lambda: {
            "site": TIGRIS_SITE,
            "cluster": TIGRIS_PARTITION,
            "collector_job_id": 999_001,
            "capture_host": "simulated.tigris.invalid",
            "python_no_user_site": True,
            "conda_environment": "atlas_kd_tigris",
            "conda_prefix": "/home/test/miniforge/envs/atlas_kd_tigris",
            "python_executable": (
                "/home/test/miniforge/envs/atlas_kd_tigris/bin/python"
            ),
            "ld_library_path_prefix": (
                "/home/test/miniforge/envs/atlas_kd_tigris/lib"
            ),
            "platform": "posix",
        },
    )


def _publish(path: Path, artifact: dict) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(artifact))
    return artifact_reference(path)


def _workers(tmp_path: Path) -> dict[str, dict[str, str]]:
    ordinary = tmp_path / "workers" / "ordinary.sh"
    deterministic = tmp_path / "workers" / "deterministic.sh"
    ordinary.parent.mkdir(parents=True, exist_ok=True)
    ordinary.write_bytes(b"#!/bin/bash\nordinary-worker\n")
    deterministic.write_bytes(b"#!/bin/bash\ndeterministic-worker\n")
    return {
        "ordinary": artifact_reference(ordinary),
        "deterministic": artifact_reference(deterministic),
    }


def _memory_bytes(request: dict) -> int:
    return int(str(request["memory"])[:-1]) * 1024**3


def _walltime_seconds(request: dict) -> int:
    hours, minutes, seconds = map(int, str(request["walltime"]).split(":"))
    return hours * 3600 + minutes * 60 + seconds


def _scheduler(
    tmp_path: Path,
    *,
    request: dict,
    resource_class: str,
    workers: dict[str, dict[str, str]],
    job_id: int,
    task_key: str,
    worker_role: str = "ordinary",
    source_commit: str = SOURCE,
    representation_recipe_sha256: str | None = None,
    raw_overrides: dict | None = None,
) -> tuple[dict, dict[str, str]]:
    worker = workers[worker_role]
    comment = scheduler_evidence_comment(
        task_key=task_key, resource_class=resource_class,
        source_commit=source_commit,
        representation_recipe_sha256=representation_recipe_sha256,
        worker_role=worker_role, worker_sha256=worker["sha256"], request=request,
    )
    parent = {
        "JobIDRaw": str(job_id),
        "JobName": f"hcwdl_rkd_{task_key}",
        "Account": TIGRIS_ACCOUNT,
        "Partition": TIGRIS_PARTITION,
        "Cluster": TIGRIS_PARTITION,
        "State": "COMPLETED",
        "ExitCode": "0:0",
        "ElapsedRaw": str(min(60, _walltime_seconds(request))),
        "TimelimitRaw": str((_walltime_seconds(request) + 59) // 60),
        "ReqCPUS": str(request["cpus"]),
        "ReqMem": request["memory"],
        "ReqGRES": "(null)" if request["gpu"] is None else request["gpu"],
        "MaxRSS": "",
        "Comment": comment,
        "SubmitLine": (
            f"sbatch --comment={comment} {shlex.quote(worker['path'])} {task_key}"
        ),
    }
    parent.update(raw_overrides or {})
    batch = dict(parent)
    batch.update({
        "JobIDRaw": f"{job_id}.batch",
        "JobName": "batch",
        "MaxRSS": f"{max(1, _memory_bytes(request) // (2 * 1024))}K",
    })
    fields = (
        "JobIDRaw", "JobName", "Account", "Partition", "Cluster", "State",
        "ExitCode", "ElapsedRaw", "TimelimitRaw", "ReqCPUS", "ReqMem",
        "ReqGRES", "MaxRSS", "Comment", "SubmitLine",
    )
    raw_path = tmp_path / "sacct" / f"{task_key}-{job_id}.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        "|".join(fields) + "\n"
        + "\n".join("|".join(row[name] for name in fields) for row in (parent, batch))
        + "\n",
        encoding="utf-8",
    )
    artifact = build_scheduler_evidence_from_sacct(
        raw_accounting_record=artifact_reference(raw_path), task_key=task_key,
        resource_class=resource_class, source_commit=source_commit,
        representation_recipe_sha256=representation_recipe_sha256,
        worker_role=worker_role, worker=worker, request=request,
    )
    reference = _publish(
        tmp_path / "scheduler" / f"{task_key}-{job_id}.json", artifact,
    )
    return artifact, reference


def _miniature(
    tmp_path: Path,
    *,
    scheduler: dict,
    evidence_kind: str,
    recipe: str | None,
) -> tuple[dict, dict[str, str]]:
    result = with_content_hash({
        "contract": "HCWDL_TEST_MEASURED_RESULT/v1",
        "schema_version": 1,
        "evidence_kind": evidence_kind,
        "job_id": scheduler["job_id"],
    })
    result_ref = _publish(
        tmp_path / "results" / f"{evidence_kind}-{scheduler['job_id']}.json",
        result,
    )
    artifact = build_miniature_evidence(
        evidence_kind=evidence_kind, scheduler_evidence=scheduler,
        representation_recipe_sha256=recipe, rows=8,
        result_artifact=result_ref,
    )
    reference = _publish(
        tmp_path / "miniatures" / f"{evidence_kind}-{scheduler['job_id']}.json",
        artifact,
    )
    return artifact, reference


def _resource_profile(tmp_path: Path) -> tuple[dict, dict[str, str], dict]:
    requests = resource_table(mode="smoke")
    workers = _workers(tmp_path)
    measurements = {}
    schedulers = {}
    for index, (name, request) in enumerate(requests.items(), start=1):
        role = "deterministic" if name in {
            "gpu_target", "gpu_final_prediction",
        } else "ordinary"
        scheduler, scheduler_ref = _scheduler(
            tmp_path,
            request=request,
            resource_class=name,
            workers=workers,
            job_id=1000 + index,
            task_key=f"resource-{name}",
            worker_role=role,
        )
        _, miniature_ref = _miniature(
            tmp_path,
            scheduler=scheduler,
            evidence_kind=f"resource_profile:{name}",
            recipe=None,
        )
        measurements[name] = {
            "scheduler_evidence": scheduler_ref,
            "miniature_evidence": miniature_ref,
        }
        schedulers[name] = scheduler
    profile = build_measured_profile(
        mode="smoke", source_commit=SOURCE, production_workers=workers,
        measurements=measurements,
    )
    reference = _publish(tmp_path / "resource-profile.json", profile)
    return profile, reference, schedulers


def _measured_storage(tmp_path: Path, *, train=10, validation=5, final=7, finalists=3):
    files_by_kind = {}
    for index, kind in enumerate((
        "retained_resume", "selected_checkpoint", "final_assignment", "fixed_artifact",
    ), start=1):
        path = tmp_path / "fixed" / f"{kind}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([index]) * (10 + index))
        files_by_kind[kind] = [path]
    inventory = build_fixed_size_inventory(
        parent_import_sha256=PARENT, files_by_kind=files_by_kind,
    )
    inventory_ref = _publish(tmp_path / "fixed-size-inventory.json", inventory)
    storage = build_storage_estimate(
        train_rows=train,
        validation_rows=validation,
        final_rows=final,
        parent_import_sha256=PARENT,
        prediction_finalists=finalists,
        fixed_size_inventory=inventory_ref,
    )
    storage_ref = _publish(tmp_path / "storage-estimate.json", storage)
    return storage, storage_ref, inventory_ref, files_by_kind


def _planning_spec(tmp_path: Path, *, storage: dict, inventory_ref: dict) -> dict:
    return create_campaign_spec(
        mode="smoke",
        campaign_root=tmp_path / "campaign",
        checkpoint_namespace=tmp_path / "checkpoints",
        project_dir=tmp_path / "project",
        source_commit=SOURCE,
        source_manifest_sha256="4" * 64,
        split_manifest_sha256="5" * 64,
        parent_import_sha256=PARENT,
        representation_recipe_sha256=RECIPE,
        graph_sha256="6" * 64,
        disposition_sha256="7" * 64,
        disposition="combined_confirmatory",
        role_counts={"train": 10, "validation": 5, "final_test": 7},
        final_source_partitions=2,
        combined_finalist_count=3,
        storage_estimate=storage,
        fixed_size_inventory=inventory_ref,
    )


def test_storage_fixed_sizes_are_derived_from_authenticated_inventory(tmp_path: Path) -> None:
    storage, _, inventory_ref, files = _measured_storage(tmp_path)
    validate_storage_estimate(
        storage,
        require_measured_fixed_sizes=True,
        fixed_size_inventory=inventory_ref,
    )
    spec = _planning_spec(tmp_path, storage=storage, inventory_ref=inventory_ref)
    validate_campaign_spec(spec)

    files["retained_resume"][0].write_bytes(b"changed-after-measurement")
    with pytest.raises(ValueError, match="byte hash differs"):
        validate_storage_estimate(
            storage,
            require_measured_fixed_sizes=True,
            fixed_size_inventory=inventory_ref,
        )


def test_campaign_binds_all_storage_roles_and_prediction_finalists(tmp_path: Path) -> None:
    storage, _, inventory_ref, _ = _measured_storage(tmp_path)
    spec = _planning_spec(tmp_path, storage=storage, inventory_ref=inventory_ref)

    forged = deepcopy(spec)
    forged["role_counts"]["validation"] = 6
    forged = with_content_hash(forged)
    with pytest.raises(ValueError, match="role populations"):
        validate_campaign_spec(forged)

    wrong_roles = build_storage_estimate(
        train_rows=10, validation_rows=4, final_rows=7,
        parent_import_sha256=PARENT, prediction_finalists=3,
        fixed_size_inventory=inventory_ref,
    )
    with pytest.raises(ValueError, match="role populations"):
        _planning_spec(tmp_path, storage=wrong_roles, inventory_ref=inventory_ref)

    wrong_finalists = build_storage_estimate(
        train_rows=10, validation_rows=5, final_rows=7,
        parent_import_sha256=PARENT, prediction_finalists=2,
        fixed_size_inventory=inventory_ref,
    )
    with pytest.raises(ValueError, match="finalist count"):
        _planning_spec(tmp_path, storage=wrong_finalists, inventory_ref=inventory_ref)


def test_genuine_profile_opens_evidence_and_rejects_self_asserted_or_forged_rows(
    tmp_path: Path,
) -> None:
    profile, _, schedulers = _resource_profile(tmp_path)
    validate_measured_profile(
        profile, require_genuine_tigris=True, expected_source_commit=SOURCE,
    )

    self_asserted = deepcopy(profile)
    name = next(iter(self_asserted["measurements"]))
    self_asserted["measurements"][name] = {
        "peak_rss_bytes": 1,
        "elapsed_seconds": 1,
    }
    self_asserted = with_content_hash(self_asserted)
    with pytest.raises(PermissionError, match="lacks genuine Tigris evidence"):
        validate_measured_profile(
            self_asserted,
            require_genuine_tigris=True,
            expected_source_commit=SOURCE,
        )

    forged = deepcopy(profile)
    request = forged["requests"][name]
    workers = forged["measurement_environment"]["production_workers"]
    with pytest.raises(PermissionError, match="memory request differs"):
        _scheduler(
            tmp_path,
            request=request,
            resource_class=name,
            workers=workers,
            job_id=9001,
            task_key="forged-memory",
            raw_overrides={"ReqMem": "1K"},
        )

    local = build_scheduler_evidence(
        job_id=9002, task_key="local-fixture", resource_class=name,
        source_commit=SOURCE, worker_role="ordinary",
        worker=workers["ordinary"], request=request, state="COMPLETED",
        exit_code="0:0", peak_rss_bytes=1, elapsed_seconds=1,
    )
    local_ref = _publish(tmp_path / "scheduler" / "local-fixture.json", local)
    forged["measurements"][name]["scheduler_evidence"] = local_ref
    forged = with_content_hash(forged)
    with pytest.raises(PermissionError, match="nonauthorizing local fixture"):
        validate_measured_profile(
            forged,
            require_genuine_tigris=True,
            expected_source_commit=SOURCE,
        )

    forged_scheduler = deepcopy(schedulers[name])
    forged_scheduler["capture_runtime"]["platform"] = "nt"
    forged_scheduler = with_content_hash(forged_scheduler)
    forged_scheduler_ref = _publish(
        tmp_path / "scheduler" / "forged-capture-runtime.json",
        forged_scheduler,
    )
    forged_capture = deepcopy(profile)
    forged_capture["measurements"][name][
        "scheduler_evidence"
    ] = forged_scheduler_ref
    forged_capture = with_content_hash(forged_capture)
    with pytest.raises(PermissionError, match="Tigris worker environment"):
        validate_measured_profile(
            forged_capture,
            require_genuine_tigris=True,
            expected_source_commit=SOURCE,
        )


@pytest.mark.parametrize(
    ("source_commit", "raw_override", "message"),
    [
        (SOURCE, {"ElapsedRaw": "999999"}, "exceeds requested walltime"),
        (SOURCE, {"Account": "wrong-account"}, "Tigris environment differs"),
        ("9" * 40, {}, "source commit differs"),
    ],
)
def test_scheduler_evidence_rejects_wrong_identity_request_or_elapsed(
    tmp_path: Path, source_commit: str, raw_override: dict, message: str,
) -> None:
    profile, _, _ = _resource_profile(tmp_path)
    name = "cpu_small"
    request = profile["requests"][name]
    workers = profile["measurement_environment"]["production_workers"]
    with pytest.raises((ValueError, PermissionError), match=message):
        _, bad_ref = _scheduler(
            tmp_path, request=request, resource_class=name, workers=workers,
            job_id=9100, task_key="bad-scheduler-row",
            source_commit=source_commit, raw_overrides=raw_override,
        )
        forged = deepcopy(profile)
        forged["measurements"][name]["scheduler_evidence"] = bad_ref
        forged = with_content_hash(forged)
        validate_measured_profile(
            forged, require_genuine_tigris=True, expected_source_commit=SOURCE,
        )


def _tigris_acceptance(tmp_path: Path):
    profile, profile_ref, _ = _resource_profile(tmp_path)
    storage, storage_ref, inventory_ref, _ = _measured_storage(tmp_path)
    workers = profile["measurement_environment"]["production_workers"]
    checks = {}
    for index, evidence_kind in enumerate(REQUIRED_TIGRIS_CHECKS, start=1):
        resource_class = "gpu_representation"
        scheduler, scheduler_ref = _scheduler(
            tmp_path,
            request=profile["requests"][resource_class],
            resource_class=resource_class,
            workers=workers,
            job_id=20_000 + index,
            task_key=f"acceptance-{evidence_kind}",
            representation_recipe_sha256=RECIPE,
        )
        _, miniature_ref = _miniature(
            tmp_path,
            scheduler=scheduler,
            evidence_kind=evidence_kind,
            recipe=RECIPE,
        )
        checks[evidence_kind] = {
            "scheduler_evidence": scheduler_ref,
            "miniature_evidence": miniature_ref,
        }
    bundle = with_content_hash({
        "contract": TIGRIS_EVIDENCE_BUNDLE_CONTRACT,
        "schema_version": 1,
        "source_commit": SOURCE,
        "representation_recipe_sha256": RECIPE,
        "resource_profile_sha256": profile["content_hash"],
        "storage_estimate_sha256": storage["content_hash"],
        "fixed_size_inventory_sha256": (
            # The authenticated reference opens the inventory below; its JSON
            # content hash is intentionally distinct from its byte hash.
            json.loads(Path(inventory_ref["path"]).read_text())["content_hash"]
        ),
        "site": TIGRIS_SITE,
        "account": TIGRIS_ACCOUNT,
        "partition": TIGRIS_PARTITION,
        "resource_profile": profile_ref,
        "storage_estimate": storage_ref,
        "fixed_size_inventory": inventory_ref,
        "checks": checks,
    })
    bundle_ref = _publish(tmp_path / "tigris-evidence-bundle.json", bundle)
    acceptance = with_content_hash({
        "contract": TIGRIS_ACCEPTANCE_CONTRACT,
        "schema_version": 1,
        "source_commit": SOURCE,
        "representation_recipe_sha256": RECIPE,
        "resource_profile_sha256": profile["content_hash"],
        "storage_estimate_sha256": storage["content_hash"],
        "fixed_size_inventory_sha256": bundle["fixed_size_inventory_sha256"],
        "evidence_bundle": bundle_ref,
        "authorizes_pilot_submission": True,
    })
    return acceptance, bundle, checks


def test_tigris_acceptance_rejects_self_asserted_action_evidence(tmp_path: Path) -> None:
    acceptance, bundle, _ = _tigris_acceptance(tmp_path)
    with pytest.raises(PermissionError, match="check row differs"):
        validate_tigris_acceptance(
            acceptance,
            source_commit=SOURCE,
            representation_recipe_sha256=RECIPE,
            resource_profile_sha256=bundle["resource_profile_sha256"],
            storage_estimate_sha256=bundle["storage_estimate_sha256"],
            fixed_size_inventory_sha256=bundle["fixed_size_inventory_sha256"],
        )

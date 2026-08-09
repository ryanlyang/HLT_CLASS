from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from hlt_classification.data.cache_contracts import with_content_hash
from hlt_classification.scouting.hcwdl_representation_resources import (
    artifact_reference,
    build_nonfinal_acceptance_scheduler_evidence,
    build_nonfinal_acceptance_scheduler_evidence_from_sacct,
    capture_nonfinal_acceptance_scheduler_evidence,
    nonfinal_acceptance_scheduler_comment,
    resource_table,
    validate_nonfinal_acceptance_scheduler_evidence,
)
from hlt_classification.scouting import hcwdl_representation_resources as resources


AUTHORITY = "a" * 64
SOURCE = "b" * 40
RECIPE = "c" * 64
ACTION = "acceptance_rset_m1c_two_update"


def _worker(tmp_path):
    path = tmp_path / "run_hcwdl_representation_nonfinal_acceptance.sh"
    path.write_bytes(b"#!/bin/bash\nexec python -s action.py\n")
    return artifact_reference(path)


def test_nonfinal_scheduler_fixture_binds_authority_action_and_is_nonauthorizing(
    tmp_path,
) -> None:
    worker = _worker(tmp_path)
    request = resource_table(mode="smoke")["gpu_representation"]
    evidence = build_nonfinal_acceptance_scheduler_evidence(
        authority_sha256=AUTHORITY, action_id=ACTION, job_id=41,
        resource_class="gpu_representation", source_commit=SOURCE,
        representation_recipe_sha256=RECIPE, worker_role="ordinary",
        worker=worker, request=request,
    )
    assert evidence["authorization_capable"] is False
    assert evidence["final_role_accessed"] is False
    validate_nonfinal_acceptance_scheduler_evidence(
        evidence, expected_authority_sha256=AUTHORITY,
        expected_action_id=ACTION, request=request,
        expected_source_commit=SOURCE, expected_recipe_sha256=RECIPE,
        expected_worker=worker, expected_resource_class="gpu_representation",
        expected_worker_role="ordinary",
    )
    with pytest.raises(PermissionError, match="local fixture"):
        validate_nonfinal_acceptance_scheduler_evidence(
            evidence, expected_authority_sha256=AUTHORITY,
            expected_action_id=ACTION, request=request,
            expected_source_commit=SOURCE, expected_recipe_sha256=RECIPE,
            expected_worker=worker, expected_resource_class="gpu_representation",
            expected_worker_role="ordinary", require_genuine=True,
        )
    with pytest.raises(PermissionError, match="authority or lineage"):
        validate_nonfinal_acceptance_scheduler_evidence(
            evidence, expected_authority_sha256=AUTHORITY,
            expected_action_id=ACTION, request=request,
            expected_source_commit=SOURCE, expected_recipe_sha256=RECIPE,
            expected_worker=worker, expected_resource_class="gpu_target",
            expected_worker_role="ordinary",
        )
    with pytest.raises(PermissionError, match="authority or lineage"):
        validate_nonfinal_acceptance_scheduler_evidence(
            evidence, expected_authority_sha256=AUTHORITY,
            expected_action_id=ACTION, request=request,
            expected_source_commit=SOURCE, expected_recipe_sha256=RECIPE,
            expected_worker=worker,
            expected_resource_class="gpu_representation",
            expected_worker_role="deterministic",
        )

    forged = deepcopy(evidence)
    forged["authority_sha256"] = "d" * 64
    forged = with_content_hash(forged)
    with pytest.raises(PermissionError, match="authority or lineage"):
        validate_nonfinal_acceptance_scheduler_evidence(
            forged, expected_authority_sha256=AUTHORITY,
            expected_action_id=ACTION, request=request,
            expected_source_commit=SOURCE, expected_recipe_sha256=RECIPE,
            expected_worker=worker, expected_resource_class="gpu_representation",
            expected_worker_role="ordinary",
        )


def test_nonfinal_scheduler_comment_changes_for_authority_and_action(tmp_path) -> None:
    worker = _worker(tmp_path)
    request = resource_table(mode="smoke")["gpu_representation"]
    common = dict(
        resource_class="gpu_representation", source_commit=SOURCE,
        representation_recipe_sha256=RECIPE, worker_role="ordinary",
        worker_sha256=worker["sha256"], request=request,
    )
    first = nonfinal_acceptance_scheduler_comment(
        authority_sha256=AUTHORITY, action_id=ACTION, **common,
    )
    assert first != nonfinal_acceptance_scheduler_comment(
        authority_sha256="d" * 64, action_id=ACTION, **common,
    )
    assert first != nonfinal_acceptance_scheduler_comment(
        authority_sha256=AUTHORITY,
        action_id="acceptance_rrel_m1c_two_update", **common,
    )
    with pytest.raises(PermissionError, match="forbidden"):
        nonfinal_acceptance_scheduler_comment(
            authority_sha256=AUTHORITY, action_id="train_RSET_M1c", **common,
        )


def _raw_sacct_bytes(
    *, job_id: int, action_id: str, worker: dict[str, str], request: dict,
    submit_script: str | None = None,
) -> bytes:
    comment = nonfinal_acceptance_scheduler_comment(
        authority_sha256=AUTHORITY, action_id=action_id,
        resource_class="gpu_representation", source_commit=SOURCE,
        representation_recipe_sha256=RECIPE, worker_role="ordinary",
        worker_sha256=worker["sha256"], request=request,
    )
    task_key = f"acceptance-nonfinal-{action_id}"
    script = shlex.quote(worker["path"]) if submit_script is None else submit_script
    submit = " ".join((
        "sbatch", "--parsable", f"--job-name=hcwdl_rkd_{task_key}",
        f"--account={resources.TIGRIS_ACCOUNT}",
        f"--partition={resources.TIGRIS_PARTITION}", f"--comment={comment}",
        f"--cpus-per-task={request['cpus']}", f"--mem={request['memory']}",
        f"--time={request['walltime']}", f"--gres={request['gpu']}", script,
    ))
    fields = resources.SACCT_FIELDS
    parent = {
        "JobIDRaw": str(job_id), "JobName": f"hcwdl_rkd_{task_key}",
        "Account": resources.TIGRIS_ACCOUNT,
        "Partition": resources.TIGRIS_PARTITION,
        "Cluster": resources.TIGRIS_PARTITION, "State": "COMPLETED",
        "ExitCode": "0:0", "ElapsedRaw": "30", "TimelimitRaw": "120",
        "ReqCPUS": str(request["cpus"]), "ReqMem": request["memory"],
        "ReqGRES": request["gpu"], "MaxRSS": "", "Comment": comment,
        "SubmitLine": submit,
    }
    batch = dict(parent)
    batch.update({
        "JobIDRaw": f"{job_id}.batch", "JobName": "batch", "MaxRSS": "1K",
    })
    return (
        "|".join(fields) + "\n"
        + "\n".join("|".join(row[name] for name in fields) for row in (parent, batch))
        + "\n"
    ).encode("utf-8")


def _collector_project(tmp_path: Path):
    root = tmp_path / "project"
    sbatch = root / "sbatch"
    scripts = root / "scripts"
    sbatch.mkdir(parents=True)
    scripts.mkdir()
    worker_path = sbatch / "run_hcwdl_representation_nonfinal_acceptance.sh"
    worker_path.write_bytes(b"#!/bin/bash\n")
    (sbatch / resources.NONFINAL_COLLECTOR_WORKER).write_bytes(b"#!/bin/bash\n")
    (scripts / resources.NONFINAL_COLLECTOR_CLI).write_bytes(b"#!/usr/bin/python\n")
    return root, artifact_reference(worker_path)


@pytest.mark.parametrize(
    "field,value",
    (
        ("capture_host", "attacker.example"),
        ("account", "wrong-account"),
        ("partition", "wrong-partition"),
        ("collector_job_name", "unreviewed-collector"),
        ("sacct_executable", "/tmp/sacct"),
    ),
)
def test_collector_runtime_rejects_non_tigris_or_path_shadowing(
    field: str, value: str,
) -> None:
    runtime = {
        "site": resources.TIGRIS_SITE,
        "cluster": resources.TIGRIS_PARTITION,
        "collector_job_id": 9001,
        "capture_host": "gh-a-999.invalid",
        "account": resources.TIGRIS_ACCOUNT,
        "partition": resources.TIGRIS_PARTITION,
        "collector_job_name": resources.NONFINAL_COLLECTOR_JOB_NAME,
        "sacct_executable": "/usr/bin/sacct",
        "python_no_user_site": True,
        "conda_environment": "atlas_kd_tigris",
        "conda_prefix": "/opt/conda/envs/atlas_kd_tigris",
        "python_executable": "/opt/conda/envs/atlas_kd_tigris/bin/python",
        "ld_library_path_prefix": "/opt/conda/envs/atlas_kd_tigris/lib",
        "platform": "posix",
    }
    runtime[field] = value
    with pytest.raises(PermissionError, match="Tigris worker environment"):
        resources._validate_capture_runtime(runtime)


def test_live_collector_freezes_exact_worker_argv_and_rejects_manual_bytes(
    tmp_path, monkeypatch,
) -> None:
    root, worker = _collector_project(tmp_path)
    request = resource_table(mode="smoke")["gpu_representation"]
    runtime = {
        "site": resources.TIGRIS_SITE, "cluster": resources.TIGRIS_PARTITION,
        "collector_job_id": 9001, "capture_host": "gh-a-999.invalid",
        "account": resources.TIGRIS_ACCOUNT,
        "partition": resources.TIGRIS_PARTITION,
        "collector_job_name": resources.NONFINAL_COLLECTOR_JOB_NAME,
        "sacct_executable": "/usr/bin/sacct",
        "python_no_user_site": True, "conda_environment": "atlas_kd_tigris",
        "conda_prefix": "/opt/conda/envs/atlas_kd_tigris",
        "python_executable": "/opt/conda/envs/atlas_kd_tigris/bin/python",
        "ld_library_path_prefix": "/opt/conda/envs/atlas_kd_tigris/lib",
        "platform": "posix",
    }
    monkeypatch.setattr(resources, "_live_tigris_capture_runtime", lambda: runtime)
    monkeypatch.setenv("HCWDL_NONFINAL_EVIDENCE_COLLECTOR", "1")
    from hlt_classification.scouting import hcwdl_representation_campaign

    monkeypatch.setattr(
        hcwdl_representation_campaign, "validate_source_checkout",
        lambda repository, *, expected_commit: (
            Path(repository).resolve() == root.resolve()
            and expected_commit == SOURCE
        ) or (_ for _ in ()).throw(PermissionError("source differs")),
    )
    raw = _raw_sacct_bytes(
        job_id=7001, action_id=ACTION, worker=worker, request=request,
    )
    monkeypatch.setattr(
        resources.subprocess, "run",
        lambda *_a, **_k: SimpleNamespace(stdout=raw, stderr=b"", returncode=0),
    )
    evidence = capture_nonfinal_acceptance_scheduler_evidence(
        authority_sha256=AUTHORITY, action_id=ACTION, job_id=7001,
        raw_accounting_output=tmp_path / "evidence" / "raw.psv",
        resource_class="gpu_representation", source_commit=SOURCE,
        representation_recipe_sha256=RECIPE, worker_role="ordinary",
        worker=worker, request=request,
    )
    assert evidence["authorization_capable"] is True
    assert evidence["collector_produced_raw_bytes"] is True
    assert evidence["submit_argv"][-1] == worker["path"]
    validate_nonfinal_acceptance_scheduler_evidence(
        evidence, expected_authority_sha256=AUTHORITY,
        expected_action_id=ACTION, request=request,
        expected_source_commit=SOURCE, expected_recipe_sha256=RECIPE,
        expected_worker=worker, expected_resource_class="gpu_representation",
        expected_worker_role="ordinary", require_genuine=True,
    )
    with pytest.raises(PermissionError, match="caller-supplied"):
        build_nonfinal_acceptance_scheduler_evidence_from_sacct(
            authority_sha256=AUTHORITY, action_id=ACTION,
            raw_accounting_record=evidence["raw_accounting_record"],
            resource_class="gpu_representation", source_commit=SOURCE,
            representation_recipe_sha256=RECIPE, worker_role="ordinary",
            worker=worker, request=request,
        )


def test_live_collector_rejects_reviewed_worker_as_evil_script_argument(
    tmp_path, monkeypatch,
) -> None:
    _root, worker = _collector_project(tmp_path)
    request = resource_table(mode="smoke")["gpu_representation"]
    raw = _raw_sacct_bytes(
        job_id=7002, action_id=ACTION, worker=worker, request=request,
        submit_script=f"/tmp/evil.sh {worker['path']}",
    )
    monkeypatch.setattr(resources, "_live_tigris_capture_runtime", lambda: {
        "site": resources.TIGRIS_SITE, "cluster": resources.TIGRIS_PARTITION,
        "collector_job_id": 9002, "capture_host": "gh-a-998.invalid",
        "account": resources.TIGRIS_ACCOUNT,
        "partition": resources.TIGRIS_PARTITION,
        "collector_job_name": resources.NONFINAL_COLLECTOR_JOB_NAME,
        "sacct_executable": "/usr/bin/sacct",
        "python_no_user_site": True, "conda_environment": "atlas_kd_tigris",
        "conda_prefix": "/opt/conda/envs/atlas_kd_tigris",
        "python_executable": "/opt/conda/envs/atlas_kd_tigris/bin/python",
        "ld_library_path_prefix": "/opt/conda/envs/atlas_kd_tigris/lib",
        "platform": "posix",
    })
    monkeypatch.setenv("HCWDL_NONFINAL_EVIDENCE_COLLECTOR", "1")
    monkeypatch.setattr(
        resources.subprocess, "run",
        lambda *_a, **_k: SimpleNamespace(stdout=raw, stderr=b"", returncode=0),
    )
    from hlt_classification.scouting import hcwdl_representation_campaign

    monkeypatch.setattr(
        hcwdl_representation_campaign, "validate_source_checkout",
        lambda *_a, **_k: None,
    )
    with pytest.raises(PermissionError, match="exact worker argv"):
        capture_nonfinal_acceptance_scheduler_evidence(
            authority_sha256=AUTHORITY, action_id=ACTION, job_id=7002,
            raw_accounting_output=tmp_path / "bad" / "raw.psv",
            resource_class="gpu_representation", source_commit=SOURCE,
            representation_recipe_sha256=RECIPE, worker_role="ordinary",
            worker=worker, request=request,
        )

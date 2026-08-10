"""Reviewed four-job bootstrap for genuine dense resource measurements."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import os
import subprocess
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, load_json, require_sha256, sha256_file,
    validate_content_hash, with_content_hash, write_immutable_json,
)

from .hcwdl_representation_contracts import (
    DENSE_RESOURCE_PROBE_AUTHORIZATION_CONTRACT,
    DENSE_RESOURCE_PROBE_LEDGER_CONTRACT,
    DENSE_RESOURCE_PROBE_PLAN_CONTRACT,
)
from .hcwdl_representation_resources import (
    DENSE_RESOURCE_CLASSES, TIGRIS_ACCOUNT, TIGRIS_PARTITION,
    _sacct_capture_command, artifact_reference, build_miniature_evidence,
    build_scheduler_evidence_from_sacct, scheduler_evidence_comment,
)


DENSE_RESOURCE_PROBE_AUTHORIZATION_PHRASE: Final = (
    "AUTHORIZE EXACT HCWDL-RKD DENSE RESOURCE PROBES FOR TIGRIS"
)
_WORKERS: Final = {
    "ordinary": "run_hcwdl_representation_resource_probe.sh",
    "deterministic": "run_hcwdl_representation_resource_probe_deterministic.sh",
}
_COLLECTOR_WORKER: Final = "collect_hcwdl_representation_resource_probes.sh"


def _walltime(value: str) -> str:
    fields = value.split(":")
    if len(fields) != 3 or any(not field.isdigit() for field in fields):
        raise ValueError("dense resource-probe walltime differs")
    return value


def build_dense_resource_probe_plan(
    *, planning_spec_path: str | Path, planning_spec: Mapping[str, Any],
    data_root: str | Path, conda_environment: str,
) -> dict[str, Any]:
    from .hcwdl_representation_campaign import (
        DENSE_TRAINING_DISPOSITION, validate_campaign_spec,
        validate_source_checkout,
    )
    spec_sha256 = validate_campaign_spec(planning_spec, executable=False)
    if (
        planning_spec.get("mode") != "smoke"
        or planning_spec.get("disposition") != DENSE_TRAINING_DISPOSITION
        or planning_spec.get("role_counts") != {
            "train": 512, "validation": 256, "final_test": 0,
        }
    ):
        raise PermissionError("dense resource probes require the exact smoke plan")
    spec_path = Path(planning_spec_path).resolve()
    project = Path(str(planning_spec["project_dir"])).resolve()
    data = Path(data_root).resolve()
    if (
        not spec_path.is_file() or spec_path.is_symlink()
        or not project.is_dir() or project.is_symlink()
        or not data.is_dir() or data.is_symlink()
        or not conda_environment
    ):
        raise ValueError("dense resource-probe operational paths differ")
    if load_json(spec_path) != dict(planning_spec):
        raise PermissionError("dense resource-probe planning spec file differs")
    validate_source_checkout(
        project, expected_commit=str(planning_spec["source_commit"]),
    )
    resources = planning_spec["resources"]
    if set(resources) != set(DENSE_RESOURCE_CLASSES):
        raise ValueError("dense resource-probe resource classes differ")
    rows = []
    for resource_class in DENSE_RESOURCE_CLASSES:
        deterministic = resource_class == "gpu_target"
        role = "deterministic" if deterministic else "ordinary"
        worker = project / "sbatch" / _WORKERS[role]
        if not worker.is_file() or worker.is_symlink():
            raise FileNotFoundError(worker)
        request = resources[resource_class]
        task_key = f"resource_probe_{resource_class}"
        runtime_measurement_path = (
            Path(str(planning_spec["campaign_root"]))
            / "review" / "resource_probes" / resource_class
            / "worker_runtime_measurement.json"
        ).resolve()
        result_path = (
            Path(str(planning_spec["campaign_root"]))
            / "resources" / "dense_storage_template.json"
        ).resolve() if resource_class == "gpu_representation" else runtime_measurement_path
        comment = scheduler_evidence_comment(
            task_key=task_key, resource_class=resource_class,
            source_commit=str(planning_spec["source_commit"]),
            representation_recipe_sha256=None,
            worker_role=role, worker_sha256=sha256_file(worker), request=request,
        )
        exports = ",".join((
            "ALL", f"PROJECT_DIR={project}",
            f"HCWDL_REPRESENTATION_PROBE_SPEC={spec_path}",
            f"HCWDL_REPRESENTATION_PROBE_DATA_ROOT={data}",
            f"HCWDL_REPRESENTATION_PROBE_CONDA={conda_environment}",
            f"HCWDL_REPRESENTATION_PROBE_RESOURCE_CLASS={resource_class}",
            f"HCWDL_REPRESENTATION_PROBE_RUNTIME_OUTPUT={runtime_measurement_path}",
            f"HCWDL_REPRESENTATION_PROBE_OUTPUT={result_path}",
        ))
        command = [
            "sbatch", "--parsable", f"--account={TIGRIS_ACCOUNT}",
            f"--partition={TIGRIS_PARTITION}",
            f"--job-name=hcwdlr_{task_key}", f"--comment={comment}",
            f"--cpus-per-task={int(request['cpus'])}",
            f"--mem={request['memory']}", f"--time={_walltime(request['walltime'])}",
            f"--export={exports}",
        ]
        if request["gpu"] is not None:
            command.append(f"--gres={request['gpu']}")
        command.append(str(worker))
        rows.append({
            "resource_class": resource_class, "task_key": task_key,
            "worker_role": role, "worker_path": str(worker),
            "worker_sha256": sha256_file(worker), "request": dict(request),
            "runtime_measurement_path": str(runtime_measurement_path),
            "result_path": str(result_path), "binding_comment": comment,
            "command": command,
        })
    collector_worker = project / "sbatch" / _COLLECTOR_WORKER
    if not collector_worker.is_file() or collector_worker.is_symlink():
        raise FileNotFoundError(collector_worker)
    collector_request = resources["cpu_small"]
    return with_content_hash({
        "contract": DENSE_RESOURCE_PROBE_PLAN_CONTRACT,
        "schema_version": 1,
        "planning_spec_path": str(spec_path),
        "planning_spec_byte_sha256": sha256_file(spec_path),
        "planning_spec_sha256": spec_sha256,
        "source_commit": planning_spec["source_commit"],
        "project_dir": str(project),
        "representation_recipe_sha256": planning_spec[
            "representation_recipe_sha256"
        ],
        "data_root": str(data), "conda_environment": conda_environment,
        "rows": rows,
        "collector": {
            "worker_path": str(collector_worker),
            "worker_sha256": sha256_file(collector_worker),
            "request": dict(collector_request),
            "output_root": str(
                (Path(str(planning_spec["campaign_root"])) / "review"
                 / "resource_probes").resolve()
            ),
        },
        "scheduler_mutated": False,
        "authorizes_dense_graph_submission": False,
        "final_role_access_authorized": False,
    })


def validate_dense_resource_probe_plan(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value, expected_contract=DENSE_RESOURCE_PROBE_PLAN_CONTRACT,
        expected_schema_version=1,
    )
    spec_path = Path(str(value.get("planning_spec_path")))
    if (
        not spec_path.is_absolute() or not spec_path.is_file() or spec_path.is_symlink()
        or sha256_file(spec_path) != value.get("planning_spec_byte_sha256")
    ):
        raise PermissionError("dense resource-probe planning spec bytes differ")
    from hlt_classification.data.cache_contracts import load_json
    rebuilt = build_dense_resource_probe_plan(
        planning_spec_path=spec_path, planning_spec=load_json(spec_path),
        data_root=str(value.get("data_root")),
        conda_environment=str(value.get("conda_environment")),
    )
    if dict(value) != rebuilt:
        raise PermissionError("dense resource-probe plan is not canonical")
    return digest


def build_dense_resource_probe_authorization(
    *, plan: Mapping[str, Any], authorization_phrase: str,
) -> dict[str, Any]:
    plan_sha256 = validate_dense_resource_probe_plan(plan)
    if authorization_phrase != DENSE_RESOURCE_PROBE_AUTHORIZATION_PHRASE:
        raise PermissionError("dense resource-probe authorization phrase differs")
    return with_content_hash({
        "contract": DENSE_RESOURCE_PROBE_AUTHORIZATION_CONTRACT,
        "schema_version": 1, "plan_sha256": plan_sha256,
        "source_commit": plan["source_commit"],
        "authorization_phrase": authorization_phrase,
        "measurement_probe_job_count": len(DENSE_RESOURCE_CLASSES),
        "collector_job_count": 1,
        "scheduler_job_count": len(DENSE_RESOURCE_CLASSES) + 1,
        "scheduler_mutation_authorized": True,
        "dense_graph_submission_authorized": False,
        "pilot_submission_authorized": False,
        "final_role_access_authorized": False,
    })


def validate_dense_resource_probe_authorization(
    value: Mapping[str, Any], *, plan: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        value, expected_contract=DENSE_RESOURCE_PROBE_AUTHORIZATION_CONTRACT,
        expected_schema_version=1,
    )
    rebuilt = build_dense_resource_probe_authorization(
        plan=plan, authorization_phrase=str(value.get("authorization_phrase")),
    )
    if dict(value) != rebuilt:
        raise PermissionError("dense resource-probe authorization differs")
    return digest


def build_dense_resource_probe_ledger(
    *, plan: Mapping[str, Any], authorization: Mapping[str, Any],
    job_ids: Mapping[str, str], collector_job_id: str,
) -> dict[str, Any]:
    plan_sha256 = validate_dense_resource_probe_plan(plan)
    authorization_sha256 = validate_dense_resource_probe_authorization(
        authorization, plan=plan,
    )
    if set(job_ids) != set(DENSE_RESOURCE_CLASSES) or any(
        not str(value).isdigit() or int(value) <= 0 for value in job_ids.values()
    ):
        raise ValueError("dense resource-probe Slurm job registry differs")
    if not str(collector_job_id).isdigit() or int(collector_job_id) <= 0:
        raise ValueError("dense resource-probe collector job ID differs")
    if str(collector_job_id) in {str(value) for value in job_ids.values()}:
        raise ValueError("dense resource-probe collector job must be distinct")
    return with_content_hash({
        "contract": DENSE_RESOURCE_PROBE_LEDGER_CONTRACT,
        "schema_version": 1, "plan_sha256": plan_sha256,
        "authorization_sha256": authorization_sha256,
        "source_commit": plan["source_commit"],
        "jobs": {key: str(job_ids[key]) for key in DENSE_RESOURCE_CLASSES},
        "collector_job_id": str(collector_job_id),
        "dense_graph_submitted": False, "final_role_accessed": False,
    })


def validate_dense_resource_probe_ledger(
    value: Mapping[str, Any], *, plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        value, expected_contract=DENSE_RESOURCE_PROBE_LEDGER_CONTRACT,
        expected_schema_version=1,
    )
    rebuilt = build_dense_resource_probe_ledger(
        plan=plan, authorization=authorization, job_ids=value.get("jobs", {}),
        collector_job_id=str(value.get("collector_job_id", "")),
    )
    if dict(value) != rebuilt:
        raise PermissionError("dense resource-probe ledger differs")
    return digest


def collect_dense_resource_probe_evidence(
    *, plan: Mapping[str, Any], authorization: Mapping[str, Any],
    job_ids: Mapping[str, str], collector_job_id: str,
) -> dict[str, dict[str, str]]:
    """Capture all four completed probes from the authorized collector job."""

    validate_dense_resource_probe_plan(plan)
    validate_dense_resource_probe_authorization(authorization, plan=plan)
    if set(job_ids) != set(DENSE_RESOURCE_CLASSES):
        raise ValueError("dense resource-probe collector job registry differs")
    current_job = os.environ.get("SLURM_JOB_ID", "").split("_", 1)[0]
    if current_job != str(collector_job_id) or not current_job.isdigit():
        raise PermissionError("dense resource-probe collector identity differs")
    output_root = Path(str(plan["collector"]["output_root"])).resolve()
    result: dict[str, dict[str, str]] = {}
    rows = {str(row["resource_class"]): row for row in plan["rows"]}
    for resource_class in DENSE_RESOURCE_CLASSES:
        row = rows[resource_class]
        job_id = str(job_ids[resource_class])
        if not job_id.isdigit() or int(job_id) <= 0 or job_id == current_job:
            raise ValueError("dense resource-probe measured job identity differs")
        directory = output_root / resource_class
        raw_path = directory / "sacct.psv"
        completed = subprocess.run(
            _sacct_capture_command(int(job_id), executable="/usr/bin/sacct"),
            check=True, capture_output=True,
        )
        raw_bytes = bytes(completed.stdout)
        if not raw_bytes or atomic_publish_bytes(raw_path, raw_bytes) != "published":
            raise PermissionError("dense resource-probe accounting was not freshly captured")
        scheduler = build_scheduler_evidence_from_sacct(
            raw_accounting_record=artifact_reference(raw_path),
            task_key=str(row["task_key"]), resource_class=resource_class,
            source_commit=str(plan["source_commit"]),
            representation_recipe_sha256=None,
            worker_role=str(row["worker_role"]),
            worker=artifact_reference(row["worker_path"]),
            request=row["request"],
        )
        if str(scheduler["job_id"]) != job_id:
            raise PermissionError("dense resource-probe accounting job differs")
        scheduler_path = directory / "scheduler_evidence.json"
        write_immutable_json(scheduler_path, scheduler)
        measurement_path = Path(str(row["runtime_measurement_path"])).resolve()
        measurement = load_json(measurement_path)
        from .hcwdl_representation_worker_runtime import (
            validate_worker_runtime_measurement,
        )
        validate_worker_runtime_measurement(measurement)
        if (
            measurement.get("campaign_spec_sha256")
            != plan["planning_spec_sha256"]
            or measurement.get("resource_class") != resource_class
            or measurement.get("resource_request") != row["request"]
            or measurement.get("runtime_facts", {}).get("project_dir")
            != plan["project_dir"]
        ):
            raise PermissionError("dense resource-probe result lineage differs")
        result_path = Path(str(row["result_path"])).resolve()
        if resource_class == "gpu_representation":
            from .hcwdl_representation_resources import validate_dense_storage_template
            template = load_json(result_path)
            validate_dense_storage_template(
                template, expected_source_commit=str(plan["source_commit"]),
                expected_recipe_sha256=str(plan["representation_recipe_sha256"]),
            )
            if template.get("planning_spec_sha256") != plan["planning_spec_sha256"]:
                raise PermissionError("dense storage template planning lineage differs")
        elif result_path != measurement_path:
            raise PermissionError("dense resource-probe result route differs")
        miniature = build_miniature_evidence(
            evidence_kind=f"resource_profile:{resource_class}",
            scheduler_evidence=scheduler,
            representation_recipe_sha256=None, rows=1,
            result_artifact=artifact_reference(result_path),
        )
        miniature_path = directory / "miniature_evidence.json"
        write_immutable_json(miniature_path, miniature)
        result[resource_class] = {
            "scheduler_evidence": str(scheduler_path),
            "miniature_evidence": str(miniature_path),
        }
    return result


__all__ = [
    "DENSE_RESOURCE_PROBE_AUTHORIZATION_PHRASE",
    "build_dense_resource_probe_authorization", "build_dense_resource_probe_ledger",
    "build_dense_resource_probe_plan", "collect_dense_resource_probe_evidence",
    "validate_dense_resource_probe_authorization",
    "validate_dense_resource_probe_ledger", "validate_dense_resource_probe_plan",
]

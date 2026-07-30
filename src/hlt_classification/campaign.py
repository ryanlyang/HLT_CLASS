"""Immutable baseline campaign specifications and exact Slurm DAG rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable, Mapping, Sequence

from .data.cache_contracts import (
    canonical_sha256,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
)
from .data.schema import schema_payload
from .data.splits import DEFAULT_SPLIT_SEEDS, DEFAULT_SPLIT_SIZES
from .provenance import validate_source_snapshot_payload
from .training.engine import TRAINING_CONFIG_CONTRACT

CAMPAIGN_SPEC_CONTRACT = "hlt_classification_baseline_campaign_spec_v1"
CAMPAIGN_SPEC_SCHEMA_VERSION = 1
SUBMISSION_LEDGER_CONTRACT = "hlt_classification_submission_ledger_v1"
RESUME_PLAN_CONTRACT = "hlt_classification_resume_plan_v1"
STORAGE_MEASUREMENT_CONTRACT = "hlt_classification_storage_measurement_v1"
MONITOR_REPORT_CONTRACT = "hlt_classification_monitor_report_v1"
TASK_ATTESTATION_CONTRACT = "hlt_classification_task_attestation_v1"

TERMINAL_SUCCESS_STATES = frozenset({"COMPLETED"})
TERMINAL_FAILURE_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }
)

PROJECT_DIR = "/home/ryreu/atlas/HLT_Classification"
DATA_DIR = "/home/ryreu/atlas/PracticeTagging/data"
OUTPUT_ROOT = "/home/ryreu/atlas/HLT_Classification/checkpoints"
CONDA_BASE = "/home/ryreu/miniforge3-aarch64"
CONDA_ENV = "atlas_kd_tigris"
SBATCH_ACCOUNT = "reu-aisocial"
SBATCH_PARTITION = "tigris"
GPU_GRES = "gpu:gh200:1"

SMOKE_SPLIT_SIZES = {
    "model_train": 200,
    "model_val": 100,
    "stack_train": 100,
    "stack_val": 100,
    "final_test": 100,
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    worker: str
    dependencies: tuple[str, ...]
    array: str | None
    cpus: int
    memory: str
    walltime: str
    gpu: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "worker": self.worker,
            "dependencies": list(self.dependencies),
            "array": self.array,
            "cpus": self.cpus,
            "memory": self.memory,
            "walltime": self.walltime,
            "gpu": self.gpu,
        }


def baseline_tasks(*, smoke: bool) -> tuple[TaskSpec, ...]:
    tasks = [
        TaskSpec(
            "preflight",
            "run_preflight.sh",
            (),
            None,
            4,
            "16G",
            "01:00:00",
        ),
        TaskSpec(
            "splits",
            "run_build_splits.sh",
            ("preflight",),
            None,
            4,
            "32G",
            "02:00:00",
        ),
        TaskSpec(
            "offline_cache",
            "run_build_offline_cache.sh",
            ("splits",),
            "0-1%2",
            4,
            "64G",
            "08:00:00" if smoke else "24:00:00",
        ),
        TaskSpec(
            "hlt_cache",
            "run_build_hlt_cache.sh",
            ("offline_cache",),
            "0-1%2",
            4,
            "64G",
            "08:00:00" if smoke else "24:00:00",
        ),
        TaskSpec(
            "weaver_parity",
            "run_weaver_parity.sh",
            ("preflight",),
            None,
            4,
            "32G",
            "01:00:00",
        ),
    ]
    if smoke:
        tasks.append(
            TaskSpec(
                "train_interrupt",
                "run_train_part.sh",
                ("hlt_cache", "weaver_parity"),
                None,
                8,
                "96G",
                "02:00:00",
                True,
            )
        )
    tasks.extend(
        [
            TaskSpec(
                "train",
                "run_train_part.sh",
                ("train_interrupt",)
                if smoke
                else ("hlt_cache", "weaver_parity"),
                None,
                8,
                "96G",
                "04:00:00" if smoke else "48:00:00",
                True,
            ),
            TaskSpec(
                "evaluate_model_val",
                "run_evaluate_part.sh",
                ("train",),
                None,
                8,
                "64G",
                "02:00:00" if smoke else "08:00:00",
                True,
            ),
        ]
    )
    return tuple(tasks)


def _validate_training_config(config: Mapping[str, Any]) -> None:
    if config.get("contract") != TRAINING_CONFIG_CONTRACT:
        raise ValueError("campaign training configuration contract differs")
    if config.get("schema_version") != 1:
        raise ValueError("campaign training configuration version differs")
    if config.get("performance_early_termination") is not False:
        raise ValueError("campaign may not enable performance termination")


def _validate_task_dag(tasks: Sequence[Mapping[str, Any]]) -> None:
    names = [str(task["name"]) for task in tasks]
    if len(names) != len(set(names)):
        raise ValueError("campaign task names are duplicated")
    seen: set[str] = set()
    for task in tasks:
        dependencies = list(task["dependencies"])
        if any(dependency not in seen for dependency in dependencies):
            raise ValueError("campaign tasks are not topologically ordered")
        seen.add(str(task["name"]))


def create_baseline_campaign_spec(
    *,
    source_snapshot: Mapping[str, Any],
    training_config: Mapping[str, Any],
    mode: str,
    production_authorized: bool = False,
) -> dict[str, Any]:
    if mode not in {"smoke", "production"}:
        raise ValueError("campaign mode must be smoke or production")
    validate_source_snapshot_payload(source_snapshot)
    if source_snapshot.get("worktree_clean") is not True:
        raise ValueError("campaign source snapshot must be clean")
    _validate_training_config(training_config)
    split_sizes = (
        SMOKE_SPLIT_SIZES if mode == "smoke" else dict(DEFAULT_SPLIT_SIZES)
    )
    identity = canonical_sha256(
        {
            "source_snapshot_sha256": source_snapshot[
                "source_snapshot_sha256"
            ],
            "training_config_sha256": canonical_sha256(training_config),
            "mode": mode,
            "production_authorized": bool(production_authorized),
            "split_sizes": split_sizes,
        }
    )
    campaign_id = f"hlt_part_baseline_{mode}_{identity[:16]}"
    campaign_root = f"{OUTPUT_ROOT}/{campaign_id}"
    tasks = [task.to_dict() for task in baseline_tasks(smoke=mode == "smoke")]
    payload = {
        "contract": CAMPAIGN_SPEC_CONTRACT,
        "schema_version": CAMPAIGN_SPEC_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "campaign_kind": "canonical_hlt_part_baseline",
        "mode": mode,
        "production_authorized": bool(production_authorized),
        "source_snapshot": dict(source_snapshot),
        "source_snapshot_sha256": source_snapshot["source_snapshot_sha256"],
        "site": {
            "project_dir": PROJECT_DIR,
            "data_dir": DATA_DIR,
            "output_root": OUTPUT_ROOT,
            "campaign_root": campaign_root,
            "conda_base": CONDA_BASE,
            "conda_env": CONDA_ENV,
            "slurm_account": SBATCH_ACCOUNT,
            "slurm_partition": SBATCH_PARTITION,
            "gpu_gres": GPU_GRES,
        },
        "data": {
            "raw_schema_sha256": canonical_sha256(schema_payload()),
            "split_sizes": split_sizes,
            "split_seeds": dict(DEFAULT_SPLIT_SEEDS),
            "base_seed": 52,
            "roles_materialized": ["model_train", "model_val"],
            "cache_shard_size": 512 if mode == "smoke" else 4096,
            "hlt_realization_policy": "R_FIXED",
            "hlt_replicas": [0],
            "degradation_profile_id": "D_NOMINAL",
        },
        "training_config": dict(training_config),
        "training_config_sha256": canonical_sha256(training_config),
        "final_test": {
            "inference_authorized": False,
            "requires_finalist_lock": True,
            "requires_execution_lock": True,
        },
        "tasks": tasks,
    }
    return with_content_hash(payload)


def validate_campaign_spec(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload,
        expected_contract=CAMPAIGN_SPEC_CONTRACT,
    )
    if payload.get("campaign_kind") != "canonical_hlt_part_baseline":
        raise ValueError("campaign kind differs")
    mode = str(payload.get("mode"))
    if mode not in {"smoke", "production"}:
        raise ValueError("campaign mode differs")
    _validate_training_config(payload["training_config"])
    if canonical_sha256(payload["training_config"]) != payload.get(
        "training_config_sha256"
    ):
        raise ValueError("campaign training configuration hash differs")
    expected_identity = canonical_sha256(
        {
            "source_snapshot_sha256": payload["source_snapshot_sha256"],
            "training_config_sha256": payload["training_config_sha256"],
            "mode": mode,
            "production_authorized": bool(
                payload.get("production_authorized")
            ),
            "split_sizes": payload["data"]["split_sizes"],
        }
    )
    if payload.get("campaign_id") != (
        f"hlt_part_baseline_{mode}_{expected_identity[:16]}"
    ):
        raise ValueError("campaign scientific identity differs")
    require_sha256(
        payload.get("source_snapshot_sha256"),
        name="source_snapshot_sha256",
    )
    if (
        payload["source_snapshot"].get("source_snapshot_sha256")
        != payload["source_snapshot_sha256"]
    ):
        raise ValueError("campaign source snapshot parent differs")
    validate_source_snapshot_payload(payload["source_snapshot"])
    expected_site = {
        "project_dir": PROJECT_DIR,
        "data_dir": DATA_DIR,
        "output_root": OUTPUT_ROOT,
        "campaign_root": f"{OUTPUT_ROOT}/{payload['campaign_id']}",
        "conda_base": CONDA_BASE,
        "conda_env": CONDA_ENV,
        "slurm_account": SBATCH_ACCOUNT,
        "slurm_partition": SBATCH_PARTITION,
        "gpu_gres": GPU_GRES,
    }
    if payload.get("site") != expected_site:
        raise ValueError("campaign Tigris site contract differs")
    expected_data = {
        "raw_schema_sha256": canonical_sha256(schema_payload()),
        "split_sizes": (
            SMOKE_SPLIT_SIZES
            if mode == "smoke"
            else dict(DEFAULT_SPLIT_SIZES)
        ),
        "split_seeds": dict(DEFAULT_SPLIT_SEEDS),
        "base_seed": 52,
        "roles_materialized": ["model_train", "model_val"],
        "cache_shard_size": 512 if mode == "smoke" else 4096,
        "hlt_realization_policy": "R_FIXED",
        "hlt_replicas": [0],
        "degradation_profile_id": "D_NOMINAL",
    }
    if payload.get("data") != expected_data:
        raise ValueError("campaign data contract differs")
    expected_tasks = [
        task.to_dict() for task in baseline_tasks(smoke=mode == "smoke")
    ]
    if payload.get("tasks") != expected_tasks:
        raise ValueError("campaign task DAG differs")
    _validate_task_dag(payload["tasks"])
    if payload.get("final_test") != {
        "inference_authorized": False,
        "requires_finalist_lock": True,
        "requires_execution_lock": True,
    }:
        raise ValueError("campaign final-test isolation differs")
    return digest


def task_by_name(
    campaign_spec: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    validate_campaign_spec(campaign_spec)
    matches = [task for task in campaign_spec["tasks"] if task["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"campaign task {name!r} is absent or duplicated")
    return matches[0]


def _slurm_command(
    *,
    campaign_spec_path: Path,
    campaign_spec: Mapping[str, Any],
    task: Mapping[str, Any],
    dependency_ids: Sequence[str],
    resources: Mapping[str, Any] | None,
) -> list[str]:
    site = campaign_spec["site"]
    selected = dict(task) if resources is None else {
        **task,
        **resources,
    }
    command = [
        "sbatch",
        "--parsable",
        f"--account={site['slurm_account']}",
        f"--partition={site['slurm_partition']}",
        f"--job-name=hlt_{task['name']}",
        f"--cpus-per-task={selected['cpus']}",
        f"--mem={selected['memory']}",
        f"--time={selected['walltime']}",
        (
            f"--output={site['campaign_root']}/logs/"
            f"{task['name']}_%A_%a.out"
        ),
        (
            f"--error={site['campaign_root']}/logs/"
            f"{task['name']}_%A_%a.err"
        ),
        (
            "--export=ALL,"
            f"PROJECT_DIR={site['project_dir']},"
            f"CAMPAIGN_ROOT={site['campaign_root']},"
            f"CAMPAIGN_ID={campaign_spec['campaign_id']},"
            f"CAMPAIGN_SPEC={campaign_spec_path.as_posix()},"
            f"CAMPAIGN_TASK={task['name']}"
        ),
    ]
    if selected.get("gpu"):
        command.append(f"--gres={site['gpu_gres']}")
    if selected.get("array"):
        command.append(f"--array={selected['array']}")
    if dependency_ids:
        command.append(f"--dependency=afterok:{':'.join(dependency_ids)}")
    command.append(f"{site['project_dir']}/sbatch/{task['worker']}")
    return command


def render_submission_plan(
    *,
    campaign_spec_path: str | Path,
    campaign_spec: Mapping[str, Any],
    task_names: Sequence[str] | None = None,
    measured_resources: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    validate_campaign_spec(campaign_spec)
    selected_names = (
        {task["name"] for task in campaign_spec["tasks"]}
        if task_names is None
        else set(task_names)
    )
    unknown = selected_names - {
        task["name"] for task in campaign_spec["tasks"]
    }
    if unknown:
        raise ValueError(f"unknown submission tasks: {sorted(unknown)}")
    plan: list[dict[str, Any]] = []
    for task in campaign_spec["tasks"]:
        if task["name"] not in selected_names:
            continue
        dependencies = [
            dependency
            for dependency in task["dependencies"]
            if dependency in selected_names
        ]
        placeholder_ids = [f"${{{dependency}_JOB_ID}}" for dependency in dependencies]
        command = _slurm_command(
            campaign_spec_path=Path(campaign_spec_path),
            campaign_spec=campaign_spec,
            task=task,
            dependency_ids=placeholder_ids,
            resources=(
                None
                if measured_resources is None
                else measured_resources.get(task["name"])
            ),
        )
        plan.append(
            {
                "task": task["name"],
                "dependencies": dependencies,
                "command": command,
            }
        )
    return plan


def _parse_job_id(stdout: str) -> str:
    value = stdout.strip().split(";", 1)[0]
    if not re.fullmatch(r"[0-9]+", value):
        raise RuntimeError(f"sbatch returned an invalid job id: {stdout!r}")
    return value


def submit_plan(
    *,
    campaign_spec_path: str | Path,
    campaign_spec: Mapping[str, Any],
    task_names: Sequence[str] | None = None,
    measured_resources: Mapping[str, Mapping[str, Any]] | None = None,
    executor: Callable[[Sequence[str]], str] | None = None,
) -> dict[str, Any]:
    validate_campaign_spec(campaign_spec)
    plan = render_submission_plan(
        campaign_spec_path=campaign_spec_path,
        campaign_spec=campaign_spec,
        task_names=task_names,
        measured_resources=measured_resources,
    )
    execute = executor
    if execute is None:
        def execute(command: Sequence[str]) -> str:
            result = subprocess.run(
                list(command),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return result.stdout
    job_ids: dict[str, str] = {}
    rows = []
    for row in plan:
        command: list[str] = []
        for token in row["command"]:
            resolved = token
            for dependency in row["dependencies"]:
                resolved = resolved.replace(
                    f"${{{dependency}_JOB_ID}}",
                    job_ids[dependency],
                )
            command.append(resolved)
        job_id = _parse_job_id(execute(command))
        job_ids[row["task"]] = job_id
        rows.append(
            {
                "task": row["task"],
                "job_id": job_id,
                "dependencies": {
                    dependency: job_ids[dependency]
                    for dependency in row["dependencies"]
                },
                "command": command,
            }
        )
    return with_content_hash(
        {
            "contract": SUBMISSION_LEDGER_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": campaign_spec["content_hash"],
            "campaign_id": campaign_spec["campaign_id"],
            "jobs": rows,
        }
    )


def build_storage_measurement(
    *,
    campaign_spec: Mapping[str, Any],
    available_bytes: int,
    projected_peak_bytes: int,
    observed_task_resources: Mapping[str, Mapping[str, Any]],
    measurement_host: str,
) -> dict[str, Any]:
    """Bind a conservative storage/resource preflight to one campaign."""

    validate_campaign_spec(campaign_spec)
    if (
        not isinstance(available_bytes, int)
        or isinstance(available_bytes, bool)
        or available_bytes <= 0
    ):
        raise ValueError("available storage must be a positive integer")
    if (
        not isinstance(projected_peak_bytes, int)
        or isinstance(projected_peak_bytes, bool)
        or projected_peak_bytes <= 0
    ):
        raise ValueError("projected peak storage must be a positive integer")
    if projected_peak_bytes >= available_bytes:
        raise ValueError("projected campaign peak does not fit available storage")
    task_names = [task["name"] for task in campaign_spec["tasks"]]
    if set(observed_task_resources) != set(task_names):
        raise ValueError("resource measurements do not cover every campaign task")
    resources: dict[str, dict[str, Any]] = {}
    for task in campaign_spec["tasks"]:
        row = dict(observed_task_resources[task["name"]])
        expected = {
            "cpus": task["cpus"],
            "memory": task["memory"],
            "walltime": task["walltime"],
            "gpu": task["gpu"],
            "array": task["array"],
        }
        if row != expected:
            raise ValueError(
                f"measured resources differ for task {task['name']!r}"
            )
        resources[task["name"]] = row
    if not measurement_host:
        raise ValueError("measurement host is required")
    return with_content_hash(
        {
            "contract": STORAGE_MEASUREMENT_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": campaign_spec["content_hash"],
            "source_snapshot_sha256": campaign_spec[
                "source_snapshot_sha256"
            ],
            "measurement_host": measurement_host,
            "available_bytes": available_bytes,
            "projected_peak_bytes": projected_peak_bytes,
            "minimum_free_after_peak_bytes": (
                available_bytes - projected_peak_bytes
            ),
            "resources": resources,
        }
    )


def measure_campaign_storage(
    *,
    campaign_spec: Mapping[str, Any],
    path: str | Path,
    projected_peak_bytes: int,
    measurement_host: str,
) -> dict[str, Any]:
    usage = shutil.disk_usage(Path(path))
    resources = {
        task["name"]: {
            "cpus": task["cpus"],
            "memory": task["memory"],
            "walltime": task["walltime"],
            "gpu": task["gpu"],
            "array": task["array"],
        }
        for task in campaign_spec["tasks"]
    }
    return build_storage_measurement(
        campaign_spec=campaign_spec,
        available_bytes=usage.free,
        projected_peak_bytes=projected_peak_bytes,
        observed_task_resources=resources,
        measurement_host=measurement_host,
    )


def validate_storage_measurement(
    payload: Mapping[str, Any],
    *,
    campaign_spec: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload,
        expected_contract=STORAGE_MEASUREMENT_CONTRACT,
    )
    expected = build_storage_measurement(
        campaign_spec=campaign_spec,
        available_bytes=payload["available_bytes"],
        projected_peak_bytes=payload["projected_peak_bytes"],
        observed_task_resources=payload["resources"],
        measurement_host=payload["measurement_host"],
    )
    if dict(payload) != expected:
        raise ValueError("storage measurement semantics differ")
    return digest


def validate_submission_ledger(
    payload: Mapping[str, Any],
    *,
    campaign_spec: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload,
        expected_contract=SUBMISSION_LEDGER_CONTRACT,
    )
    validate_campaign_spec(campaign_spec)
    if payload.get("campaign_spec_sha256") != campaign_spec["content_hash"]:
        raise ValueError("submission ledger campaign parent differs")
    if payload.get("campaign_id") != campaign_spec["campaign_id"]:
        raise ValueError("submission ledger campaign id differs")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("submission ledger has no jobs")
    known: dict[str, str] = {}
    for row in jobs:
        task = str(row.get("task", ""))
        if task in known:
            raise ValueError("submission ledger task is duplicated")
        job_id = str(row.get("job_id", ""))
        if not re.fullmatch(r"[0-9]+", job_id):
            raise ValueError("submission ledger job id is invalid")
        expected_dependencies = {
            dependency: known[dependency]
            for dependency in task_by_name(
                campaign_spec,
                task,
            )["dependencies"]
            if dependency in known
        }
        if row.get("dependencies") != expected_dependencies:
            raise ValueError("submission ledger dependency lineage differs")
        command = row.get("command")
        if not isinstance(command, list) or not command:
            raise ValueError("submission ledger command is absent")
        known[task] = job_id
    return digest


def _normalize_slurm_state(value: str) -> str:
    state = value.strip().split("+", 1)[0].upper()
    if not re.fullmatch(r"[A-Z_]+", state):
        raise ValueError(f"invalid Slurm state {value!r}")
    return state


def build_monitor_report(
    *,
    campaign_spec: Mapping[str, Any],
    submission_ledger: Mapping[str, Any],
    states_by_job_id: Mapping[str, str],
    artifact_validity: Mapping[str, bool],
) -> dict[str, Any]:
    validate_submission_ledger(
        submission_ledger,
        campaign_spec=campaign_spec,
    )
    task_names = [row["task"] for row in submission_ledger["jobs"]]
    if set(artifact_validity) != set(task_names):
        raise ValueError("artifact validation does not cover submitted tasks")
    if set(states_by_job_id) != {
        row["job_id"] for row in submission_ledger["jobs"]
    }:
        raise ValueError("Slurm states do not cover exact submitted job ids")
    rows = []
    for job in submission_ledger["jobs"]:
        state = _normalize_slurm_state(states_by_job_id[job["job_id"]])
        artifact_valid = bool(artifact_validity[job["task"]])
        reusable = state in TERMINAL_SUCCESS_STATES and artifact_valid
        rows.append(
            {
                "task": job["task"],
                "job_id": job["job_id"],
                "state": state,
                "artifact_valid": artifact_valid,
                "reusable": reusable,
            }
        )
    return with_content_hash(
        {
            "contract": MONITOR_REPORT_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": campaign_spec["content_hash"],
            "submission_ledger_sha256": submission_ledger["content_hash"],
            "tasks": rows,
        }
    )


def validate_monitor_report(
    payload: Mapping[str, Any],
    *,
    campaign_spec: Mapping[str, Any],
    submission_ledger: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload,
        expected_contract=MONITOR_REPORT_CONTRACT,
    )
    expected = build_monitor_report(
        campaign_spec=campaign_spec,
        submission_ledger=submission_ledger,
        states_by_job_id={
            row["job_id"]: row["state"] for row in payload["tasks"]
        },
        artifact_validity={
            row["task"]: row["artifact_valid"] for row in payload["tasks"]
        },
    )
    if dict(payload) != expected:
        raise ValueError("monitor report semantics differ")
    return digest


def build_resume_plan(
    *,
    campaign_spec: Mapping[str, Any],
    monitor_report: Mapping[str, Any],
    submission_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Create an explicit fresh-submission plan after any invalid task."""

    validate_monitor_report(
        monitor_report,
        campaign_spec=campaign_spec,
        submission_ledger=submission_ledger,
    )
    monitored = {row["task"]: row for row in monitor_report["tasks"]}
    invalid: set[str] = {
        task
        for task, row in monitored.items()
        if not row["reusable"]
    }
    rerun: set[str] = set(invalid)
    changed = True
    while changed:
        changed = False
        for task in campaign_spec["tasks"]:
            if (
                task["name"] not in rerun
                and any(parent in rerun for parent in task["dependencies"])
            ):
                rerun.add(task["name"])
                changed = True
    ordered_rerun = [
        task["name"]
        for task in campaign_spec["tasks"]
        if task["name"] in rerun
    ]
    stale_pending_job_ids = [
        monitored[name]["job_id"]
        for name in ordered_rerun
        if monitored[name]["state"]
        not in TERMINAL_SUCCESS_STATES | TERMINAL_FAILURE_STATES
    ]
    return with_content_hash(
        {
            "contract": RESUME_PLAN_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": campaign_spec["content_hash"],
            "monitor_report_sha256": monitor_report["content_hash"],
            "reusable_tasks": [
                task["name"]
                for task in campaign_spec["tasks"]
                if task["name"] not in rerun
            ],
            "rerun_tasks": ordered_rerun,
            "cancel_exact_job_ids": stale_pending_job_ids,
            "submission_plan": render_submission_plan(
                campaign_spec_path=(
                    Path(campaign_spec["site"]["campaign_root"])
                    / "campaign_spec.json"
                ),
                campaign_spec=campaign_spec,
                task_names=ordered_rerun,
            ),
        }
    )


def simulate_failure(
    *,
    campaign_spec: Mapping[str, Any],
    submission_ledger: Mapping[str, Any],
    failed_task: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deterministic no-Slurm failure/recovery exercise used by smoke-simulate."""

    task_names = [row["task"] for row in submission_ledger["jobs"]]
    if failed_task is not None and failed_task not in task_names:
        raise ValueError("simulated failure task is absent")
    states = {
        row["job_id"]: (
            "FAILED" if row["task"] == failed_task else "COMPLETED"
        )
        for row in submission_ledger["jobs"]
    }
    validity = {
        row["task"]: row["task"] != failed_task
        for row in submission_ledger["jobs"]
    }
    monitor = build_monitor_report(
        campaign_spec=campaign_spec,
        submission_ledger=submission_ledger,
        states_by_job_id=states,
        artifact_validity=validity,
    )
    resume = build_resume_plan(
        campaign_spec=campaign_spec,
        monitor_report=monitor,
        submission_ledger=submission_ledger,
    )
    return monitor, resume


def build_task_attestation(
    *,
    campaign_spec: Mapping[str, Any],
    task_name: str,
    artifacts: Mapping[str, str],
) -> dict[str, Any]:
    task_by_name(campaign_spec, task_name)
    normalized = {
        str(path).replace("\\", "/"): require_sha256(
            digest,
            name=f"artifacts[{path}]",
        )
        for path, digest in sorted(artifacts.items())
    }
    if not normalized:
        raise ValueError("task attestation requires at least one artifact")
    return with_content_hash(
        {
            "contract": TASK_ATTESTATION_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": campaign_spec["content_hash"],
            "source_snapshot_sha256": campaign_spec[
                "source_snapshot_sha256"
            ],
            "task": task_name,
            "artifacts": normalized,
        }
    )


def validate_task_attestation(
    payload: Mapping[str, Any],
    *,
    campaign_spec: Mapping[str, Any],
    campaign_root: str | Path,
) -> str:
    digest = validate_content_hash(
        payload,
        expected_contract=TASK_ATTESTATION_CONTRACT,
    )
    expected = build_task_attestation(
        campaign_spec=campaign_spec,
        task_name=str(payload["task"]),
        artifacts=payload["artifacts"],
    )
    if dict(payload) != expected:
        raise ValueError("task attestation semantics differ")
    root = Path(campaign_root).resolve()
    for relative, expected_hash in payload["artifacts"].items():
        artifact = (root / relative).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as error:
            raise ValueError("task artifact escapes campaign root") from error
        if not artifact.is_file():
            raise ValueError(f"task artifact is absent: {relative}")
        if sha256_file(artifact) != expected_hash:
            raise ValueError(f"task artifact hash differs: {relative}")
    return digest


__all__ = [
    "CAMPAIGN_SPEC_CONTRACT",
    "DATA_DIR",
    "GPU_GRES",
    "OUTPUT_ROOT",
    "PROJECT_DIR",
    "SBATCH_ACCOUNT",
    "SBATCH_PARTITION",
    "SUBMISSION_LEDGER_CONTRACT",
    "STORAGE_MEASUREMENT_CONTRACT",
    "TASK_ATTESTATION_CONTRACT",
    "MONITOR_REPORT_CONTRACT",
    "RESUME_PLAN_CONTRACT",
    "TaskSpec",
    "baseline_tasks",
    "create_baseline_campaign_spec",
    "build_monitor_report",
    "build_resume_plan",
    "build_storage_measurement",
    "build_task_attestation",
    "measure_campaign_storage",
    "render_submission_plan",
    "submit_plan",
    "simulate_failure",
    "task_by_name",
    "validate_campaign_spec",
    "validate_monitor_report",
    "validate_storage_measurement",
    "validate_submission_ledger",
    "validate_task_attestation",
]

"""Immutable PRAD campaign graph and guarded Tigris submission rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping, Sequence

from hlt_classification.campaign import (
    CONDA_BASE,
    CONDA_ENV,
    DATA_DIR,
    GPU_GRES,
    PROJECT_DIR,
    SBATCH_ACCOUNT,
    SBATCH_PARTITION,
)
from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from hlt_classification.provenance import validate_source_snapshot_payload

from .experiments import CORE_EXPERIMENTS, experiment_variant
from .training import PRAD_CONFIRMATION_SEEDS

PRAD_CAMPAIGN_SPEC_CONTRACT = "hlt_classification_prad_campaign_spec_v1"
PRAD_SUBMISSION_LEDGER_CONTRACT = "hlt_classification_prad_submission_ledger_v1"
PRAD_RESOURCE_EVIDENCE_CONTRACT = "hlt_classification_prad_resource_evidence_v1"
PRAD_STORAGE_EVIDENCE_CONTRACT = "hlt_classification_prad_storage_evidence_v1"
PRAD_TASK_ATTESTATION_CONTRACT = "hlt_classification_prad_task_attestation_v1"
PRAD_CAMPAIGN_SCHEMA_VERSION = 1
PRAD_SCREEN_SEED = 11
PRAD_VARIANTS = (
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7_8",
    "V7_16",
    "V7_32",
    "V8_ALL",
    "V8_AFTER2",
    "V8_FINAL",
    "V9_LAYER_HEAD",
    "V9_LAYER",
    "V9_GLOBAL",
    "V10",
)


@dataclass(frozen=True)
class PradTask:
    name: str
    dependencies: tuple[str, ...]
    cpus: int
    memory: str
    walltime: str
    gpu: bool = False
    array: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dependencies": list(self.dependencies),
            "cpus": self.cpus,
            "memory": self.memory,
            "walltime": self.walltime,
            "gpu": self.gpu,
            "array": self.array,
            "worker": "run_prad_task.sh",
        }


def prad_tasks(*, smoke: bool) -> tuple[PradTask, ...]:
    short = "01:00:00" if smoke else "08:00:00"
    train = "02:00:00" if smoke else "48:00:00"
    tasks = [
        PradTask("split", (), 4, "32G", short),
        PradTask("data_audit", ("split",), 4, "32G", short),
        PradTask("weaver_parity", (), 4, "32G", "01:00:00"),
        PradTask(
            "prad_runtime",
            ("weaver_parity",),
            4,
            "32G",
            "01:00:00",
            gpu=True,
        ),
        PradTask("paired_train", ("split",), 4, "64G", short, array="0-3%2"),
        PradTask("paired_val", ("split",), 4, "64G", short),
        PradTask("paired_test_inputs", ("split",), 4, "64G", short),
        PradTask("targets_train", ("paired_train",), 4, "64G", short, array="0-3%2"),
        PradTask("targets_val", ("paired_val",), 4, "64G", short),
        PradTask("targets_test_inputs", ("paired_test_inputs",), 4, "64G", short),
        PradTask("train_statistics", ("targets_train",), 4, "32G", short),
        PradTask(
            "E0_baseline",
            (
                "paired_train",
                "paired_val",
                "targets_val",
                "prad_runtime",
                "data_audit",
            ),
            8,
            "96G",
            train,
            gpu=True,
        ),
        PradTask(
            "E1_teacher",
            ("E0_baseline", "train_statistics", "targets_train"),
            8,
            "96G",
            train,
            gpu=True,
        ),
        PradTask(
            "teacher_val_outputs",
            ("E1_teacher",),
            4,
            "64G",
            short,
            gpu=True,
        ),
        PradTask(
            "E2_oracle",
            ("teacher_val_outputs", "targets_val"),
            8,
            "96G",
            train,
            gpu=True,
        ),
        PradTask(
            "core_screen",
            ("E2_oracle",),
            8,
            "96G",
            train,
            gpu=True,
            array="3-10%2",
        ),
        PradTask(
            "variant_screen",
            ("core_screen",),
            8,
            "96G",
            train,
            gpu=True,
            array=f"0-{len(PRAD_VARIANTS) - 1}%2",
        ),
        PradTask("selection", ("variant_screen",), 4, "32G", short),
        PradTask(
            "confirmation",
            ("selection",),
            8,
            "96G",
            train,
            gpu=True,
            array=f"0-{len(PRAD_CONFIRMATION_SEEDS) - 1}%2",
        ),
        PradTask(
            "finalist_lock",
            ("confirmation", "paired_test_inputs"),
            4,
            "32G",
            short,
        ),
        PradTask(
            "final_test",
            ("finalist_lock", "paired_test_inputs", "targets_test_inputs"),
            8,
            "96G",
            train,
            gpu=True,
        ),
        PradTask("aggregate_report", ("final_test",), 4, "32G", short),
    ]
    return tuple(tasks)


def _run_registry() -> dict[str, Any]:
    core = {
        name: {"seed": PRAD_SCREEN_SEED, "experiment": row.to_dict()}
        for name, row in CORE_EXPERIMENTS.items()
    }
    variants = {
        name: {
            "seed": PRAD_SCREEN_SEED,
            "experiment": experiment_variant("E9", name).to_dict(),
        }
        for name in PRAD_VARIANTS
    }
    return {
        "screen_seed": PRAD_SCREEN_SEED,
        "core": core,
        "variants": variants,
        "confirmation_seeds": list(PRAD_CONFIRMATION_SEEDS),
        "mandatory_confirmation_graphs": ["E0", "E8", "E9"],
        "additional_confirmation_rule": (
            "selected_variant_and_every_predeclared_graph_within_one_percent_"
            "of_best_validation_macro_log_rejection"
        ),
    }


def create_prad_campaign_spec(
    *,
    source_snapshot: Mapping[str, Any],
    mode: str,
    campaign_root: str,
    production_authorized: bool = False,
    dry_run_report_sha256: str | None = None,
    miniature_report_sha256: str | None = None,
    resource_evidence: Mapping[str, Any] | None = None,
    storage_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if mode not in {"smoke", "production"}:
        raise ValueError("PRAD campaign mode must be smoke or production")
    validate_source_snapshot_payload(source_snapshot)
    if source_snapshot.get("worktree_clean") is not True:
        raise ValueError("PRAD campaign source snapshot must be clean")
    if mode == "smoke" and production_authorized:
        raise ValueError("smoke PRAD campaign cannot be production-authorized")
    resource_hash = (
        None
        if resource_evidence is None
        else validate_prad_resource_evidence(resource_evidence)
    )
    storage_hash = (
        None
        if storage_evidence is None or resource_evidence is None
        else validate_prad_storage_evidence(
            storage_evidence, resource_evidence=resource_evidence
        )
    )
    if resource_evidence is not None:
        recorded_dry_run = resource_evidence["dry_run_report_sha256"]
        recorded_miniature = resource_evidence["monitor_report_sha256"]
        if dry_run_report_sha256 is not None and dry_run_report_sha256 != recorded_dry_run:
            raise ValueError("PRAD dry-run evidence differs from resource evidence")
        if miniature_report_sha256 is not None and miniature_report_sha256 != recorded_miniature:
            raise ValueError("PRAD miniature evidence differs from resource evidence")
        dry_run_report_sha256 = recorded_dry_run
        miniature_report_sha256 = recorded_miniature
    evidence = {
        "dry_run_report_sha256": dry_run_report_sha256,
        "miniature_report_sha256": miniature_report_sha256,
        "resource_evidence_sha256": resource_hash,
        "resource_requests_sha256": (
            None
            if resource_evidence is None
            else canonical_sha256(resource_evidence["production_requests"])
        ),
        "storage_evidence_sha256": storage_hash,
    }
    if mode == "production":
        if not production_authorized:
            raise PermissionError("full PRAD campaign requires explicit authorization")
        for name, value in evidence.items():
            evidence[name] = require_sha256(value, name=name)
    elif (
        resource_evidence is not None
        or storage_evidence is not None
        or any(value is not None for value in evidence.values())
    ):
        raise ValueError("smoke PRAD specification may not claim production evidence")
    tasks = [task.to_dict() for task in prad_tasks(smoke=mode == "smoke")]
    if mode == "production":
        assert resource_evidence is not None
        if storage_evidence is None:
            raise ValueError("full PRAD campaign requires storage evidence")
        if storage_evidence.get("source_snapshot_sha256") != source_snapshot[
            "source_snapshot_sha256"
        ]:
            raise ValueError("PRAD storage evidence source snapshot differs")
        if (
            resource_evidence.get("source_snapshot_sha256")
            != source_snapshot["source_snapshot_sha256"]
        ):
            raise ValueError("PRAD resource evidence source snapshot differs")
        requests = resource_evidence["production_requests"]
        tasks = [
            {
                **task,
                **{
                    key: requests[task["name"]][key]
                    for key in ("cpus", "memory", "walltime")
                },
            }
            for task in tasks
        ]
    return with_content_hash(
        {
            "contract": PRAD_CAMPAIGN_SPEC_CONTRACT,
            "schema_version": PRAD_CAMPAIGN_SCHEMA_VERSION,
            "mode": mode,
            "production_authorized": bool(production_authorized),
            "source_snapshot": dict(source_snapshot),
            "site": {
                "project_dir": PROJECT_DIR,
                "data_dir": DATA_DIR,
                "campaign_root": campaign_root,
                "conda_base": CONDA_BASE,
                "conda_env": CONDA_ENV,
                "slurm_account": SBATCH_ACCOUNT,
                "slurm_partition": SBATCH_PARTITION,
                "gpu_gres": GPU_GRES,
                "python_no_user_site": True,
                "prepend_conda_lib": True,
            },
            "split": {
                "seed": 1337,
                "sizes": (
                    {"train": 200, "val": 100, "test": 100}
                    if mode == "smoke"
                    else {"train": 500_000, "val": 150_000, "test": 500_000}
                ),
                "production_equivalent": mode == "production",
            },
            "required_full_split": {
                "seed": 1337,
                "sizes": {"train": 500_000, "val": 150_000, "test": 500_000},
            },
            "run_registry": _run_registry(),
            "tasks": tasks,
            "production_evidence": evidence,
            "poor_performance_cancels_tasks": False,
            "test_inference_requires_locks": True,
        }
    )


def validate_prad_campaign_spec(spec: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        spec, expected_contract=PRAD_CAMPAIGN_SPEC_CONTRACT
    )
    site = spec.get("site", {})
    if (
        site.get("project_dir") != PROJECT_DIR
        or site.get("data_dir") != DATA_DIR
        or site.get("slurm_account") != "reu-aisocial"
        or site.get("slurm_partition") != "tigris"
        or site.get("conda_env") != "atlas_kd_tigris"
        or site.get("python_no_user_site") is not True
        or site.get("prepend_conda_lib") is not True
    ):
        raise ValueError("PRAD Tigris site contract differs")
    mode = spec.get("mode")
    expected_split = (
        {"train": 200, "val": 100, "test": 100}
        if mode == "smoke"
        else {"train": 500_000, "val": 150_000, "test": 500_000}
        if mode == "production"
        else None
    )
    required_full = {"train": 500_000, "val": 150_000, "test": 500_000}
    if (
        expected_split is None
        or spec.get("split", {}).get("seed") != 1337
        or spec.get("split", {}).get("sizes") != expected_split
        or spec.get("required_full_split")
        != {"seed": 1337, "sizes": required_full}
    ):
        raise ValueError("PRAD split campaign contract differs")
    if mode == "production":
        if spec.get("production_authorized") is not True:
            raise PermissionError("production PRAD campaign is not authorized")
        for name, value in spec.get("production_evidence", {}).items():
            require_sha256(value, name=name)
        task_requests = {
            task["name"]: {
                key: task[key]
                for key in ("cpus", "memory", "walltime", "gpu", "array")
            }
            for task in spec["tasks"]
        }
        if canonical_sha256(task_requests) != spec["production_evidence"].get(
            "resource_requests_sha256"
        ):
            raise ValueError("PRAD measured production resource requests differ")
    elif spec.get("production_authorized") is not False:
        raise ValueError("smoke PRAD campaign authorization differs")
    tasks = spec.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("PRAD campaign task graph is absent")
    seen = set()
    for task in tasks:
        if task["name"] in seen:
            raise ValueError("PRAD task graph contains a duplicate task")
        if any(dependency not in seen for dependency in task["dependencies"]):
            raise ValueError("PRAD task graph is not topologically ordered")
        seen.add(task["name"])
    if spec.get("poor_performance_cancels_tasks") is not False:
        raise ValueError("PRAD campaign may not cancel work for weak performance")
    if spec.get("test_inference_requires_locks") is not True:
        raise ValueError("PRAD campaign test sealing differs")
    return digest


def validate_prad_submission_ledger(
    payload: Mapping[str, Any], *, spec: Mapping[str, Any]
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=PRAD_SUBMISSION_LEDGER_CONTRACT
    )
    validate_prad_campaign_spec(spec)
    if payload.get("campaign_spec_sha256") != spec["content_hash"]:
        raise ValueError("PRAD submission ledger campaign differs")
    known: dict[str, str] = {}
    task_lookup = {row["name"]: row for row in spec["tasks"]}
    for row in payload.get("jobs", []):
        task = row.get("task")
        job_id = str(row.get("job_id", ""))
        if task not in task_lookup or task in known or not job_id.isdigit():
            raise ValueError("PRAD submission ledger job differs")
        missing = [name for name in task_lookup[task]["dependencies"] if name not in known]
        if missing:
            raise ValueError("PRAD submission ledger dependency is absent")
        known[task] = job_id
    if set(known) != set(task_lookup):
        raise ValueError("PRAD submission ledger is incomplete")
    return digest


def build_prad_task_attestation(
    *,
    campaign_spec_sha256: str,
    task: str,
    array_task_id: str | None,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one immutable task or array-element completion attestation."""

    if not task or (array_task_id is not None and not array_task_id.isdigit()):
        raise ValueError("PRAD task attestation identity differs")
    return with_content_hash(
        {
            "contract": PRAD_TASK_ATTESTATION_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "task": task,
            "array_task_id": array_task_id,
            "result": dict(result),
        }
    )


def validate_prad_task_attestation(
    payload: Mapping[str, Any],
    *,
    campaign_spec_sha256: str,
    task: str,
    array_task_id: str | None,
) -> str:
    """Validate exact campaign, task, and array-element completion lineage."""

    digest = validate_content_hash(
        payload, expected_contract=PRAD_TASK_ATTESTATION_CONTRACT
    )
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("PRAD task attestation result differs")
    expected = build_prad_task_attestation(
        campaign_spec_sha256=campaign_spec_sha256,
        task=task,
        array_task_id=array_task_id,
        result=result,
    )
    if dict(payload) != expected:
        raise ValueError("PRAD task attestation lineage differs")
    return digest


def validate_prad_monitor_report(
    payload: Mapping[str, Any], *, spec: Mapping[str, Any], ledger: Mapping[str, Any]
) -> str:
    digest = validate_content_hash(
        payload, expected_contract="hlt_classification_prad_monitor_report_v1"
    )
    ledger_hash = validate_prad_submission_ledger(ledger, spec=spec)
    if (
        payload.get("campaign_spec_sha256") != spec["content_hash"]
        or payload.get("submission_ledger_sha256") != ledger_hash
    ):
        raise ValueError("PRAD monitor report lineage differs")
    expected = {row["task"]: row["job_id"] for row in ledger["jobs"]}
    tasks = {row["name"]: row for row in spec["tasks"]}
    rows = payload.get("jobs")
    if (
        not isinstance(rows, list)
        or len(rows) != len(expected)
        or {row.get("task") for row in rows} != set(expected)
    ):
        raise ValueError("PRAD monitor report task coverage differs")
    for row in rows:
        name = row["task"]
        if row.get("job_id") != expected[name]:
            raise ValueError("PRAD monitor report exact job ID differs")
        array = tasks[name].get("array")
        if array is None:
            expected_attestations = 1
        else:
            match = re.fullmatch(r"([0-9]+)-([0-9]+)(?:%[0-9]+)?", array)
            if match is None:
                raise ValueError("PRAD monitor array expression differs")
            expected_attestations = int(match.group(2)) - int(match.group(1)) + 1
        attestations = row.get("attestations")
        if not isinstance(attestations, list) or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in attestations
        ):
            raise ValueError("PRAD monitor attestation hashes differ")
        reusable = row.get("reusable")
        expected_reusable = (
            row.get("state") == "COMPLETED"
            and len(attestations) == expected_attestations
        )
        if reusable is not expected_reusable:
            raise ValueError("PRAD monitor reusable decision differs")
    return digest


def validate_prad_resource_evidence(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=PRAD_RESOURCE_EVIDENCE_CONTRACT
    )
    require_sha256(payload.get("smoke_campaign_spec_sha256"), name="smoke_campaign_spec_sha256")
    require_sha256(payload.get("submission_ledger_sha256"), name="submission_ledger_sha256")
    require_sha256(payload.get("dry_run_report_sha256"), name="dry_run_report_sha256")
    require_sha256(payload.get("monitor_report_sha256"), name="monitor_report_sha256")
    require_sha256(payload.get("source_snapshot_sha256"), name="source_snapshot_sha256")
    if not payload.get("measurement_host") or int(payload.get("campaign_artifact_bytes", 0)) <= 0:
        raise ValueError("PRAD resource evidence host or storage differs")
    measurements = payload.get("measurements")
    requests = payload.get("production_requests")
    expected_tasks = {task.name for task in prad_tasks(smoke=False)}
    if not isinstance(measurements, list) or not isinstance(requests, Mapping):
        raise ValueError("PRAD resource evidence payload differs")
    measured_tasks = set()
    for row in measurements:
        task = str(row.get("task", ""))
        if task in measured_tasks or task not in expected_tasks:
            raise ValueError("PRAD resource measurement task differs")
        if row.get("state") != "COMPLETED" or not str(row.get("job_id", "")).isdigit():
            raise ValueError("PRAD resource measurement did not complete")
        for key in ("elapsed_seconds", "max_rss_bytes", "allocated_cpus"):
            if not isinstance(row.get(key), int) or isinstance(row.get(key), bool) or row[key] <= 0:
                raise ValueError(f"PRAD measured {key} differs")
        measured_tasks.add(task)
    if measured_tasks != expected_tasks or set(requests) != expected_tasks:
        raise ValueError("PRAD resource evidence does not cover every task")
    production = {task.name: task for task in prad_tasks(smoke=False)}
    for name, request in requests.items():
        if (
            not isinstance(request.get("cpus"), int)
            or isinstance(request.get("cpus"), bool)
            or request["cpus"] <= 0
            or not re.fullmatch(r"[1-9][0-9]*[KMGT]", str(request.get("memory", "")))
            or not re.fullmatch(r"[0-9]{2,3}:[0-5][0-9]:[0-5][0-9]", str(request.get("walltime", "")))
            or request.get("gpu") != production[name].gpu
            or request.get("array") != production[name].array
        ):
            raise ValueError(f"PRAD production resource request differs for {name}")
    return digest


def estimate_prad_peak_storage_bytes() -> int:
    """Conservative uncompressed full-campaign peak including all finalists."""

    particles = 128
    train, val, test = 500_000, 150_000, 500_000
    replicated_rows = 4 * train + val + test
    paired_per_row = 2 * particles * 14 * 4 + 3 * particles + 256
    targets_per_row = particles * (2 + 4 + 1 + 3 * 2) + 256
    maximum_graphs = 10 + len(PRAD_VARIANTS)
    test_prediction_per_row = 10 * 4 + 160
    predictions = maximum_graphs * len(PRAD_CONFIRMATION_SEEDS) * test * test_prediction_per_row
    training_runs = 11 + len(PRAD_VARIANTS) + 2 + maximum_graphs * len(PRAD_CONFIRMATION_SEEDS)
    checkpoint_and_history = training_runs * 160 * 1024**2
    raw = (
        replicated_rows * (paired_per_row + targets_per_row)
        + predictions
        + checkpoint_and_history
        + test * (11 * 4 + 160)
    )
    return int(raw * 3 // 2)


def validate_prad_storage_evidence(
    payload: Mapping[str, Any], *, resource_evidence: Mapping[str, Any]
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=PRAD_STORAGE_EVIDENCE_CONTRACT
    )
    resource_hash = validate_prad_resource_evidence(resource_evidence)
    if (
        payload.get("resource_evidence_sha256") != resource_hash
        or payload.get("source_snapshot_sha256")
        != resource_evidence["source_snapshot_sha256"]
        or not payload.get("measurement_host")
        or not payload.get("measurement_path")
    ):
        raise ValueError("PRAD storage evidence lineage differs")
    for name in (
        "available_bytes",
        "projected_peak_bytes",
        "estimator_floor_bytes",
        "required_free_after_peak_bytes",
        "free_after_peak_bytes",
    ):
        value = payload.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"PRAD storage {name} differs")
    if (
        payload["estimator_floor_bytes"] != estimate_prad_peak_storage_bytes()
        or payload["projected_peak_bytes"] < payload["estimator_floor_bytes"]
        or payload["free_after_peak_bytes"]
        != payload["available_bytes"] - payload["projected_peak_bytes"]
        or payload["free_after_peak_bytes"]
        < payload["required_free_after_peak_bytes"]
    ):
        raise ValueError("PRAD production storage headroom is insufficient")
    return digest


def build_prad_storage_evidence(
    *,
    resource_evidence: Mapping[str, Any],
    available_bytes: int,
    projected_peak_bytes: int,
    required_free_after_peak_bytes: int,
    measurement_host: str,
    measurement_path: str,
) -> dict[str, Any]:
    resource_hash = validate_prad_resource_evidence(resource_evidence)
    payload = with_content_hash(
        {
            "contract": PRAD_STORAGE_EVIDENCE_CONTRACT,
            "schema_version": 1,
            "resource_evidence_sha256": resource_hash,
            "source_snapshot_sha256": resource_evidence[
                "source_snapshot_sha256"
            ],
            "measurement_host": measurement_host,
            "measurement_path": measurement_path,
            "available_bytes": available_bytes,
            "projected_peak_bytes": projected_peak_bytes,
            "estimator_floor_bytes": estimate_prad_peak_storage_bytes(),
            "required_free_after_peak_bytes": required_free_after_peak_bytes,
            "free_after_peak_bytes": available_bytes - projected_peak_bytes,
        }
    )
    validate_prad_storage_evidence(payload, resource_evidence=resource_evidence)
    return payload


def build_prad_resource_evidence(
    *,
    smoke_spec: Mapping[str, Any],
    submission_ledger: Mapping[str, Any],
    dry_run_report: Mapping[str, Any],
    monitor_report: Mapping[str, Any],
    usage_by_job_id: Mapping[str, Mapping[str, Any]],
    production_requests: Mapping[str, Mapping[str, Any]],
    campaign_artifact_bytes: int,
    measurement_host: str,
) -> dict[str, Any]:
    """Bind exact completed smoke IDs to reviewed production requests."""

    validate_prad_campaign_spec(smoke_spec)
    if smoke_spec["mode"] != "smoke":
        raise ValueError("PRAD resource evidence requires a smoke campaign")
    ledger_hash = validate_prad_submission_ledger(submission_ledger, spec=smoke_spec)
    dry_run_hash = validate_content_hash(
        dry_run_report, expected_contract="hlt_classification_prad_dry_run_v1"
    )
    if (
        dry_run_report.get("campaign_spec_sha256") != smoke_spec["content_hash"]
        or dry_run_report.get("mutated") is not False
    ):
        raise ValueError("PRAD dry-run report differs")
    monitor_hash = validate_prad_monitor_report(
        monitor_report, spec=smoke_spec, ledger=submission_ledger
    )
    if (
        monitor_report.get("campaign_spec_sha256") != smoke_spec["content_hash"]
        or monitor_report.get("submission_ledger_sha256") != ledger_hash
        or not monitor_report.get("jobs")
        or not all(row.get("reusable") is True for row in monitor_report["jobs"])
    ):
        raise ValueError("PRAD real miniature monitor evidence differs")
    job_ids = {row["job_id"] for row in submission_ledger["jobs"]}
    if set(usage_by_job_id) != job_ids:
        raise ValueError("PRAD resource use does not cover exact submitted IDs")
    task_by_job = {
        row["job_id"]: row["task"] for row in submission_ledger["jobs"]
    }
    measurements = []
    for job_id in sorted(job_ids, key=int):
        usage = usage_by_job_id[job_id]
        measurements.append(
            {
                "task": task_by_job[job_id],
                "job_id": job_id,
                "state": usage.get("state"),
                "elapsed_seconds": usage.get("elapsed_seconds"),
                "max_rss_bytes": usage.get("max_rss_bytes"),
                "allocated_cpus": usage.get("allocated_cpus"),
            }
        )
    payload = with_content_hash(
        {
            "contract": PRAD_RESOURCE_EVIDENCE_CONTRACT,
            "schema_version": 1,
            "smoke_campaign_spec_sha256": smoke_spec["content_hash"],
            "submission_ledger_sha256": ledger_hash,
            "dry_run_report_sha256": dry_run_hash,
            "monitor_report_sha256": monitor_hash,
            "source_snapshot_sha256": smoke_spec["source_snapshot"][
                "source_snapshot_sha256"
            ],
            "measurement_host": measurement_host,
            "campaign_artifact_bytes": campaign_artifact_bytes,
            "measurements": measurements,
            "production_requests": {
                name: dict(value) for name, value in production_requests.items()
            },
        }
    )
    validate_prad_resource_evidence(payload)
    return payload


def _command(
    *,
    spec_path: Path,
    spec: Mapping[str, Any],
    task: Mapping[str, Any],
    dependency_ids: Sequence[str],
) -> list[str]:
    site = spec["site"]
    command = [
        "sbatch",
        "--parsable",
        f"--account={site['slurm_account']}",
        f"--partition={site['slurm_partition']}",
        f"--cpus-per-task={task['cpus']}",
        f"--mem={task['memory']}",
        f"--time={task['walltime']}",
        f"--job-name=prad_{task['name']}",
        f"--chdir={site['project_dir']}",
        f"--output={site['campaign_root']}/logs/{task['name']}_%A_%a.out",
        f"--error={site['campaign_root']}/logs/{task['name']}_%A_%a.err",
        f"--export=ALL,PROJECT_DIR={site['project_dir']},CAMPAIGN_SPEC={spec_path},CAMPAIGN_TASK={task['name']}",
    ]
    if task.get("gpu"):
        command.append(f"--gres={site['gpu_gres']}")
    if task.get("array"):
        command.append(f"--array={task['array']}")
    if dependency_ids:
        command.append("--dependency=afterok:" + ":".join(dependency_ids))
    command.append(f"{site['project_dir']}/sbatch/run_prad_task.sh")
    return command


def render_prad_submission_plan(
    *, campaign_spec_path: str | Path, spec: Mapping[str, Any]
) -> list[dict[str, Any]]:
    validate_prad_campaign_spec(spec)
    symbolic: dict[str, str] = {}
    plan = []
    for task in spec["tasks"]:
        dependencies = [symbolic[name] for name in task["dependencies"]]
        symbolic_id = f"${{{task['name']}_JOB_ID}}"
        plan.append(
            {
                "task": task["name"],
                "dependencies": list(task["dependencies"]),
                "command": _command(
                    spec_path=Path(campaign_spec_path),
                    spec=spec,
                    task=task,
                    dependency_ids=dependencies,
                ),
            }
        )
        symbolic[task["name"]] = symbolic_id
    return plan


def submit_prad_plan(
    *,
    campaign_spec_path: str | Path,
    spec: Mapping[str, Any],
    executor: Callable[[Sequence[str]], str] | None = None,
    on_submitted: Callable[[Mapping[str, Any]], None] | None = None,
    existing_jobs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    validate_prad_campaign_spec(spec)
    if executor is None:
        def executor(command: Sequence[str]) -> str:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            return completed.stdout.strip()
    existing = {str(row["task"]): dict(row) for row in existing_jobs}
    if len(existing) != len(existing_jobs):
        raise ValueError("PRAD partial submission journal duplicates a task")
    task_lookup = {row["name"]: row for row in spec["tasks"]}
    if set(existing) - set(task_lookup):
        raise ValueError("PRAD reusable submission journal has unknown tasks")
    for name in existing:
        if any(dependency not in existing for dependency in task_lookup[name]["dependencies"]):
            raise ValueError("PRAD reusable submission journal is not dependency-closed")
    ids: dict[str, str] = {}
    jobs = []
    for task in spec["tasks"]:
        dependency_ids = [ids[name] for name in task["dependencies"]]
        command = _command(
            spec_path=Path(campaign_spec_path),
            spec=spec,
            task=task,
            dependency_ids=dependency_ids,
        )
        if task["name"] in existing:
            job = existing[task["name"]]
            if not str(job.get("job_id", "")).isdigit() or job.get("command") != command:
                raise ValueError("PRAD partial submission journal differs")
            jobs.append(job)
            ids[task["name"]] = str(job["job_id"])
            continue
        raw = executor(command)
        job_id = raw.split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError(f"sbatch returned a nonnumeric job id: {raw!r}")
        job = {"task": task["name"], "job_id": job_id, "command": command}
        if on_submitted is not None:
            on_submitted(job)
        jobs.append(job)
        ids[task["name"]] = job_id
    return with_content_hash(
        {
            "contract": PRAD_SUBMISSION_LEDGER_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": spec["content_hash"],
            "jobs": jobs,
        }
    )


__all__ = [
    "PRAD_TASK_ATTESTATION_CONTRACT",
    "PRAD_VARIANTS",
    "build_prad_task_attestation",
    "build_prad_resource_evidence",
    "build_prad_storage_evidence",
    "create_prad_campaign_spec",
    "prad_tasks",
    "estimate_prad_peak_storage_bytes",
    "render_prad_submission_plan",
    "submit_prad_plan",
    "validate_prad_campaign_spec",
    "validate_prad_monitor_report",
    "validate_prad_resource_evidence",
    "validate_prad_storage_evidence",
    "validate_prad_submission_ledger",
    "validate_prad_task_attestation",
]

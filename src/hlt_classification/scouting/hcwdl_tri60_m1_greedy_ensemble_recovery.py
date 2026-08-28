"""Source-pinned reducer-only recovery for the TRI60 M1 greedy diagnostic."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Final, Mapping, Sequence

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, write_immutable_json,
)

from .hcwdl_mhpe_tri60_campaign import ACCOUNT, PARTITION
from .hcwdl_recovery import (
    validate_submission_ledger, validate_task_attestation,
)
from .hcwdl_tri60_m1_greedy_ensemble import SHARD_COUNT, shard_paths
from .hcwdl_tri60_m1_greedy_ensemble_campaign import (
    JOB_PREFIX, SCHEDULER_NICE, campaign_tasks, validate_campaign,
)
from .hcwdl_tri60_m1_greedy_ensemble_contracts import (
    COMMAND_PLAN_CONTRACT, REDUCER_RECOVERY_SPEC_CONTRACT,
    SHARD_REPORT_CONTRACT, artifact, validate_artifact,
)


SOURCE_REPAIR_PHRASE: Final = (
    "AUTHORIZE TRI60 M1 GREEDY BF16 FP32 REDUCER SOURCE REPAIR"
)
RECOVERY_SUBMISSION_PHRASE: Final = (
    "SUBMIT TRI60 M1 GREEDY REDUCER RECOVERY EXACT LEDGER"
)
SOURCE_REPAIR_ALLOWLIST: Final = frozenset({
    "src/hlt_classification/scouting/hcwdl_tri60_m1_greedy_ensemble.py",
    "src/hlt_classification/scouting/hcwdl_tri60_m1_greedy_ensemble_contracts.py",
    "src/hlt_classification/scouting/hcwdl_tri60_m1_greedy_ensemble_recovery.py",
    "src/hlt_classification/scouting/hcwdl_tri60_m1_greedy_ensemble_workflow.py",
    "scripts/create_hcwdl_tri60_m1_greedy_ensemble_recovery.py",
    "scripts/run_hcwdl_tri60_m1_greedy_ensemble_recovery_task.py",
    "scripts/submit_hcwdl_tri60_m1_greedy_ensemble_recovery.py",
    "sbatch/run_hcwdl_tri60_m1_greedy_ensemble_recovery_task.sh",
})
RECOVERY_TASKS: Final = ("greedy_reduce", "campaign_complete")


def _validated_shards(spec: Mapping[str, Any]) -> dict[str, str]:
    root = Path(spec["campaign_root"])
    result = {}
    for index in range(SHARD_COUNT):
        data_path, report_path = shard_paths(root, index)
        report = load_json(report_path)
        digest = validate_artifact(report, contract=SHARD_REPORT_CONTRACT)
        if (
            report.get("parents", {}).get("campaign_spec") != spec["content_hash"]
            or report.get("parents", {}).get("source_lock")
            != spec["parents"]["source_lock"]
            or report.get("shard_index") != index
            or Path(report.get("prediction_path", "")).resolve()
            != data_path.resolve()
            or not data_path.is_file()
            or sha256_file(data_path) != report.get("prediction_file_sha256")
        ):
            raise ValueError(f"TRI60 M1 greedy reusable shard differs: {index}")
        result[f"shard_{index:02d}"] = digest
    return result


def create_recovery(
    *, campaign_spec: str | Path, submission_ledger: str | Path,
    failed_reducer_job: str, recovery_root: str | Path,
    project_dir: str | Path, source_commit: str,
    changed_files: Sequence[str], source_repair_phrase: str | None,
    publish: bool = True,
) -> dict[str, Any]:
    spec_path = Path(campaign_spec).resolve()
    spec = load_json(spec_path)
    validate_campaign(spec, executable=False)
    ledger_path = Path(submission_ledger).resolve()
    ledger = load_json(ledger_path)
    ledger_hash = validate_submission_ledger(ledger)
    expected_tasks = tuple(row["task_id"] for row in campaign_tasks())
    if (
        ledger.get("dry_run") is not False
        or ledger.get("campaign_spec_sha256") != spec["content_hash"]
        or set(ledger.get("jobs", {})) != set(expected_tasks)
        or ledger["jobs"].get("greedy_reduce") != str(failed_reducer_job)
    ):
        raise ValueError("TRI60 M1 greedy reducer recovery ledger differs")
    if not re.fullmatch(r"[0-9]+", str(failed_reducer_job)):
        raise ValueError("TRI60 M1 greedy failed reducer job differs")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("TRI60 M1 greedy recovery source commit differs")
    changed = tuple(sorted(set(map(str, changed_files))))
    if (
        source_commit == spec["source_commit"]
        or not changed
        or not set(changed) <= SOURCE_REPAIR_ALLOWLIST
        or source_repair_phrase != SOURCE_REPAIR_PHRASE
    ):
        raise PermissionError("TRI60 M1 greedy reducer source repair is unauthorized")
    shards = _validated_shards(spec)
    subject_root = Path(spec["campaign_root"])
    if (
        (subject_root / "reports/greedy_ensemble.json").exists()
        or (subject_root / "reports/campaign_complete.json").exists()
    ):
        raise FileExistsError("TRI60 M1 greedy reducer output already exists")
    for index in range(SHARD_COUNT):
        attestation_path = subject_root / "tasks" / f"infer_shard_{index:02d}" / "single.json"
        attestation = load_json(attestation_path)
        validate_task_attestation(
            attestation, campaign_spec_sha256=spec["content_hash"],
            task_id=f"infer_shard_{index:02d}", array_index=None,
        )
    root = Path(recovery_root).resolve()
    project = Path(project_dir).resolve()
    if publish and root.exists():
        raise FileExistsError("TRI60 M1 greedy recovery root exists")
    recovery = artifact({
        "campaign_spec_path": str(spec_path),
        "campaign_spec_sha256": spec["content_hash"],
        "subject_ledger_path": str(ledger_path),
        "subject_ledger_sha256": ledger_hash,
        "failed_reducer_job": str(failed_reducer_job),
        "superseded_completion_job": ledger["jobs"]["campaign_complete"],
        "recovery_root": str(root), "project_dir": str(project),
        "source_commit": source_commit,
        "previous_source_commit": spec["source_commit"],
        "changed_files": list(changed),
        "source_repair_phrase": source_repair_phrase,
        "recovery_tasks": list(RECOVERY_TASKS),
        "reused_shard_reports": shards,
        "reused_prediction_shard_count": SHARD_COUNT,
        "fresh_inference_shard_count": 0,
        "scientific_candidate_set_unchanged": True,
        "common_fp32_selection_regime": True,
        "final_test_accessed": False,
    }, contract=REDUCER_RECOVERY_SPEC_CONTRACT)
    task_map = {row["task_id"]: row for row in campaign_tasks()}
    commands = []
    worker = project / "sbatch/run_hcwdl_tri60_m1_greedy_ensemble_recovery_task.sh"
    for task_id in RECOVERY_TASKS:
        task = task_map[task_id]
        resource = spec["resources"][task["resource_class"]]
        dependencies = [] if task_id == "greedy_reduce" else ["greedy_reduce"]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}",
            f"--partition={PARTITION}", f"--cpus-per-task={resource['cpus']}",
            f"--mem={resource['memory']}", f"--time={resource['walltime']}",
            f"--nice={SCHEDULER_NICE}", f"--job-name={JOB_PREFIX}r_{task_id}",
            f"--chdir={project}", f"--output={root}/slurm-%j.out",
        ]
        if dependencies:
            command.append("--dependency=afterok:${JOB_greedy_reduce}")
        command.extend((
            "--export=ALL," +
            f"PROJECT_DIR={project}," +
            f"HCWDL_M1_GREEDY_RECOVERY_SPEC={root / 'recovery_spec.json'}," +
            f"HCWDL_M1_GREEDY_TASK={task_id}", str(worker),
        ))
        commands.append({
            "task_id": task_id, "dependencies": dependencies,
            "command": command,
        })
    plan = artifact({
        "spec_sha256": recovery["content_hash"], "commands": commands,
        "source_scheduler_dependencies": [], "scheduler_nice": SCHEDULER_NICE,
        "mutated": False, "recovery": True, "final_test_accessed": False,
    }, contract=COMMAND_PLAN_CONTRACT)
    if publish:
        write_immutable_json(root / "recovery_spec.json", recovery)
        write_immutable_json(root / "command_plan.json", plan)
    return recovery


def validate_recovery(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=REDUCER_RECOVERY_SPEC_CONTRACT)
    spec = load_json(value["campaign_spec_path"])
    validate_campaign(spec, executable=False)
    ledger = load_json(value["subject_ledger_path"])
    ledger_hash = validate_submission_ledger(ledger)
    changed = tuple(value.get("changed_files", ()))
    if (
        value.get("campaign_spec_sha256") != spec["content_hash"]
        or value.get("subject_ledger_sha256") != ledger_hash
        or ledger.get("jobs", {}).get("greedy_reduce")
        != value.get("failed_reducer_job")
        or ledger.get("jobs", {}).get("campaign_complete")
        != value.get("superseded_completion_job")
        or value.get("previous_source_commit") != spec["source_commit"]
        or re.fullmatch(r"[0-9a-f]{40}", str(value.get("source_commit"))) is None
        or value.get("source_commit") == spec["source_commit"]
        or tuple(sorted(set(changed))) != changed
        or not changed
        or not set(changed) <= SOURCE_REPAIR_ALLOWLIST
        or value.get("source_repair_phrase") != SOURCE_REPAIR_PHRASE
        or value.get("recovery_tasks") != list(RECOVERY_TASKS)
        or value.get("reused_shard_reports") != _validated_shards(spec)
        or value.get("reused_prediction_shard_count") != SHARD_COUNT
        or value.get("fresh_inference_shard_count") != 0
        or value.get("scientific_candidate_set_unchanged") is not True
        or value.get("common_fp32_selection_regime") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 M1 greedy reducer recovery differs")
    plan = load_json(Path(value["recovery_root"]) / "command_plan.json")
    validate_artifact(plan, contract=COMMAND_PLAN_CONTRACT)
    if (
        plan.get("spec_sha256") != digest
        or [row.get("task_id") for row in plan.get("commands", ())]
        != list(RECOVERY_TASKS)
        or plan.get("scheduler_nice") != SCHEDULER_NICE
        or plan.get("source_scheduler_dependencies") != []
    ):
        raise ValueError("TRI60 M1 greedy reducer recovery plan differs")
    return digest


__all__ = [
    "RECOVERY_SUBMISSION_PHRASE", "RECOVERY_TASKS", "SOURCE_REPAIR_ALLOWLIST",
    "SOURCE_REPAIR_PHRASE", "create_recovery", "validate_recovery",
]

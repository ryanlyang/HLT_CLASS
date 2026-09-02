"""Restart-zero, source-pinned recovery for output handoff."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import load_json, write_immutable_json

from .hcwdl_adjacent_output_handoff_campaign import (
    JOB_PREFIX, RECOVERY_SUBMISSION_PHRASE, RESOURCES, validate_campaign,
)
from .hcwdl_adjacent_output_handoff_contracts import (
    RECOVERY_PLAN_CONTRACT, RECOVERY_SPEC_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_adjacent_output_handoff_workflow import task_outputs
from .hcwdl_mhpe_tri60_campaign import ACCOUNT, PARTITION
from .hcwdl_recovery import (
    MONITOR_CONTRACT, resume_tasks, validate_submission_ledger,
)
from hlt_classification.data.cache_contracts import validate_content_hash


def _graph(
    spec: Mapping[str, Any], task_ids: set[str],
) -> dict[str, tuple[str, ...]]:
    available = {row["task_id"] for row in spec["tasks"]}
    if not task_ids or not task_ids <= available:
        raise ValueError("output-handoff recovery ledger task coverage differs")
    return {
        row["task_id"]: tuple(name for name in row["dependencies"] if name in task_ids)
        for row in spec["tasks"] if row["task_id"] in task_ids
    }


def create_recovery(
    *, campaign_spec: str | Path, submission_ledger: str | Path,
    monitor_report: str | Path, recovery_root: str | Path,
    project_dir: str | Path, source_commit: str, publish: bool = True,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("output-handoff recovery source commit differs")
    spec_path = Path(campaign_spec).resolve(); spec = load_json(spec_path); validate_campaign(spec)
    ledger_path = Path(submission_ledger).resolve(); ledger = load_json(ledger_path)
    ledger_hash = validate_submission_ledger(ledger)
    monitor_path = Path(monitor_report).resolve(); monitor = load_json(monitor_path)
    monitor_hash = validate_content_hash(monitor, expected_contract=MONITOR_CONTRACT, expected_schema_version=1)
    if (
        ledger.get("campaign_spec_sha256") != spec["content_hash"]
        or monitor.get("submission_ledger_sha256") != ledger_hash
        or any(row.get("disposition") == "active_or_unknown" for row in monitor["rows"])
    ):
        raise ValueError("output-handoff recovery subject is not terminal")
    subject_graph = _graph(spec, set(ledger["jobs"]))
    retry = resume_tasks(monitor, dependency_graph=subject_graph)
    if not retry:
        raise ValueError("output-handoff recovery has no failed closure")
    root = Path(recovery_root).resolve(); project = Path(project_dir).resolve()
    if publish and root.exists(): raise FileExistsError("output-handoff recovery root exists")
    recovery = artifact({
        "subject_campaign_spec_path": str(spec_path),
        "subject_submission_ledger_path": str(ledger_path),
        "monitor_report_path": str(monitor_path), "recovery_root": str(root),
        "project_dir": str(project), "source_commit": source_commit,
        "parents": {"campaign_spec": spec["content_hash"], "submission_ledger": ledger_hash, "monitor": monitor_hash},
        "retry_tasks": list(retry), "restart_from_update_zero": True,
        "scientific_semantics_unchanged": True, "final_test_accessed": False,
    }, contract=RECOVERY_SPEC_CONTRACT)
    tasks = {row["task_id"]: row for row in spec["tasks"]}
    commands = []
    for task_id in retry:
        task = tasks[task_id]; resource = RESOURCES[task["resource"]]
        dependencies = [name for name in task["dependencies"] if name in retry]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}", f"--partition={PARTITION}",
            "--nodes=1", "--ntasks=1", f"--cpus-per-task={resource.cpus}",
            f"--mem={resource.memory}", f"--time={resource.walltime}",
            f"--job-name={JOB_PREFIX}r_{task_id}",
        ]
        if resource.gpu: command.extend((f"--gres={resource.gpu}", "--signal=B:USR1@120"))
        if dependencies: command.append("--dependency=afterok:" + ":".join(f"${{JOB_{x}}}" for x in dependencies))
        command.extend((
            "--export=ALL," + f"PROJECT_DIR={project},HCWDL_OFH_RECOVERY_SPEC={root / 'recovery_spec.json'},HCWDL_OFH_TASK={task_id}",
            str(project / "sbatch/run_hcwdl_adjacent_output_handoff_recovery_task.sh"),
        ))
        commands.append({"task_id": task_id, "dependencies": dependencies, "external_dependencies": [], "command": command})
    plan = artifact({
        "recovery_spec_sha256": recovery["content_hash"], "commands": commands,
        "restart_from_update_zero": True, "final_test_accessed": False,
    }, contract=RECOVERY_PLAN_CONTRACT)
    if publish:
        root.mkdir(parents=True, exist_ok=False)
        write_immutable_json(root / "recovery_spec.json", recovery)
        write_immutable_json(root / "command_plan.json", plan)
    return recovery


def validate_recovery(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=RECOVERY_SPEC_CONTRACT)
    if value.get("restart_from_update_zero") is not True or value.get("scientific_semantics_unchanged") is not True or value.get("final_test_accessed") is not False:
        raise ValueError("output-handoff recovery semantics differ")
    return digest


def clean_incomplete_task_outputs(spec: Mapping[str, Any], task_id: str) -> None:
    task = {row["task_id"]: row for row in spec["tasks"]}[task_id]
    root = Path(spec["campaign_root"])
    targets: list[Path] = []
    if task["kind"] == "train": targets.append(root / "training" / task["node_id"])
    elif task["kind"] == "model_reducer":
        distribution = "SOURCE_U100" if task["node_id"] == "SOURCE_U100" else __import__(
            "hlt_classification.scouting.hcwdl_adjacent_output_handoff_graph", fromlist=["node_distribution"]
        ).node_distribution(task["node_id"])
        targets.extend((root / "probabilities" / distribution, root / "reports/stages" / f"{distribution}.json"))
    elif task["kind"] == "selection":
        selection = task["selection_id"]
        targets.extend((
            root / "probabilities" / selection,
            root / "reports/mixtures" / selection,
            root / "reports/stages" / f"{selection}.json",
        ))
    elif task["kind"] == "ensemble":
        ensemble = task["ensemble_id"]; targets.extend((root / "probabilities" / ensemble, root / "reports/ensembles" / f"{ensemble}.json"))
    else: targets.extend(task_outputs(spec, task_id))
    for path in targets:
        resolved = path.resolve()
        if root.resolve() not in resolved.parents and resolved != root.resolve():
            raise PermissionError("output-handoff cleanup escaped campaign root")
        if resolved.is_dir(): shutil.rmtree(resolved)
        elif resolved.exists(): resolved.unlink()


__all__ = [
    "RECOVERY_SUBMISSION_PHRASE", "clean_incomplete_task_outputs",
    "create_recovery", "validate_recovery",
]

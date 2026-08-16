"""Exact failed/downstream and resource-only recovery for HCWDL-MHPE."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from hlt_classification.data.cache_contracts import (
    load_json, validate_content_hash, with_content_hash, write_immutable_json,
)
from .hcwdl_recovery import MONITOR_CONTRACT, resume_tasks, validate_submission_ledger
from .hcwdl_recovery import task_attestation_path, validate_task_attestation

from .hcwdl_mhpe_campaign import ACCOUNT, PARTITION, campaign_tasks, validate_campaign
from .hcwdl_mhpe_contracts import (
    CAMPAIGN_SPEC_CONTRACT, CAMPAIGN_SPEC_CONTRACT_C10P90,
    CAMPAIGN_SPEC_CONTRACT_C10P90_300K60,
    CAMPAIGN_SPEC_CONTRACT_C25P75_300K60,
    CAMPAIGN_SPEC_CONTRACT_DENSE_ANCHOR50_300K60,
    COMMAND_PLAN_CONTRACT, RECOVERY_SPEC_CONTRACT,
    RESOURCE_RECOVERY_SPEC_CONTRACT, campaign_profile,
)
from .hcwdl_mhpe_graph import (
    PROFILE_C10P90, PROFILE_C25P75,
    PROFILE_C10P90_300K60, PROFILE_C25P75_300K60,
    PROFILE_DENSE_ANCHOR50_300K60,
)

SOURCE_REPAIR_PHRASE = "AUTHORIZE HCWDL MHPE EXECUTION-ONLY SOURCE REPAIR"
SOURCE_REPAIR_ALLOWLIST = frozenset({
    "src/hlt_classification/scouting/hcwdl_mhpe_runner.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_workflow.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_targets.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_recovery.py",
    "scripts/run_hcwdl_mhpe_task.py", "scripts/run_hcwdl_mhpe_recovery_task.py",
    "sbatch/run_hcwdl_mhpe_task.sh", "sbatch/run_hcwdl_mhpe_recovery_task.sh",
})


def failed_downstream_closure(
    failed: Sequence[str], *, profile: str = PROFILE_C25P75,
) -> tuple[str, ...]:
    tasks = campaign_tasks(profile); known = {row["task_id"] for row in tasks}
    closure = set(map(str, failed))
    if not closure or not closure <= known:
        raise ValueError("HCWDL-MHPE failed task set differs")
    changed = True
    while changed:
        changed = False
        for row in tasks:
            if row["task_id"] not in closure and closure.intersection(row["dependencies"]):
                closure.add(row["task_id"]); changed = True
    return tuple(row["task_id"] for row in tasks if row["task_id"] in closure)


def create_recovery(
    *, original_spec: str | Path, original_ledger: str | Path,
    recovery_root: str | Path, project_dir: str | Path, source_commit: str,
    monitor_report: str | Path,
    resource_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    changed_files: Sequence[str] = (), repair_authorization_phrase: str | None = None,
    publish: bool = True,
) -> dict[str, Any]:
    subject_path = Path(original_spec).resolve(); subject = load_json(subject_path)
    parent_recovery = None
    if subject.get("contract") in {
        CAMPAIGN_SPEC_CONTRACT, CAMPAIGN_SPEC_CONTRACT_C10P90,
        CAMPAIGN_SPEC_CONTRACT_C25P75_300K60,
        CAMPAIGN_SPEC_CONTRACT_C10P90_300K60,
        CAMPAIGN_SPEC_CONTRACT_DENSE_ANCHOR50_300K60,
    }:
        spec = subject; validate_campaign(spec, verify_source_tree=False)
        profile = campaign_profile(spec)
        subject_tasks = tuple(row["task_id"] for row in campaign_tasks(profile))
        attestation_root = Path(spec["campaign_root"])
    else:
        validate_recovery(subject)
        parent_recovery = subject
        spec = load_json(subject["campaign_spec_path"])
        validate_campaign(spec, verify_source_tree=False)
        profile = campaign_profile(spec)
        subject_tasks = tuple(subject["recovery_tasks"])
        attestation_root = Path(subject["recovery_root"])
    ledger = load_json(original_ledger); ledger_hash = validate_submission_ledger(ledger)
    if ledger.get("campaign_spec_sha256") != subject["content_hash"] or ledger.get("dry_run") is not False:
        raise ValueError("HCWDL-MHPE recovery ledger differs")
    monitor = load_json(monitor_report)
    monitor_hash = validate_content_hash(monitor, expected_contract=MONITOR_CONTRACT, expected_schema_version=1)
    if monitor.get("submission_ledger_sha256") != ledger_hash:
        raise ValueError("HCWDL-MHPE recovery monitor/ledger differs")
    subject_set = set(subject_tasks)
    dependencies = {
        row["task_id"]: [parent for parent in row["dependencies"] if parent in subject_set]
        for row in campaign_tasks(profile) if row["task_id"] in subject_set
    }
    failed_tasks = resume_tasks(monitor, dependency_graph=dependencies)
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("HCWDL-MHPE recovery commit differs")
    changed = tuple(sorted(set(map(str, changed_files))))
    if source_commit == subject["source_commit"]:
        if changed:
            raise ValueError("HCWDL-MHPE unchanged-source recovery names changed files")
    elif (not changed or not set(changed) <= SOURCE_REPAIR_ALLOWLIST
          or repair_authorization_phrase != SOURCE_REPAIR_PHRASE):
        raise PermissionError("HCWDL-MHPE source repair is not exactly authorized")
    closure = failed_downstream_closure(failed_tasks, profile=profile)
    resources = {name: dict(value) for name, value in subject["resources"].items()}
    for name, override in (resource_overrides or {}).items():
        if name not in resources or set(override) - {"cpus", "memory", "walltime"}:
            raise ValueError("HCWDL-MHPE resource recovery keys differ")
        # Changes are operational only; GPU class may never change.
        previous = resources[name]
        if ("cpus" in override and int(override["cpus"]) < int(previous["cpus"])) or ("memory" in override and _memory_gib(str(override["memory"])) < _memory_gib(str(previous["memory"]))) or ("walltime" in override and _wall_seconds(str(override["walltime"])) < _wall_seconds(str(previous["walltime"]))):
            raise ValueError("HCWDL-MHPE resource recovery may not decrease requests")
        resources[name].update(dict(override))
    contract = RESOURCE_RECOVERY_SPEC_CONTRACT if resource_overrides else RECOVERY_SPEC_CONTRACT
    root = Path(recovery_root).resolve(); project = Path(project_dir).resolve()
    completed_attestations = []
    for row in monitor["rows"]:
        if row.get("disposition") != "complete":
            continue
        task_id = str(row["task_id"])
        path = task_attestation_path(attestation_root, task_id, None)
        attestation = load_json(path)
        attestation_hash = validate_task_attestation(
            attestation, campaign_spec_sha256=subject["content_hash"],
            task_id=task_id, array_index=None,
        )
        completed_attestations.append({
            "task_id": task_id, "path": str(path.resolve()),
            "attestation_sha256": attestation_hash,
        })
    recovery = with_content_hash({
        "contract": contract, "schema_version": 1,
        "campaign_spec_path": str(Path(spec["spec_path"]).resolve()),
        "campaign_spec_sha256": spec["content_hash"],
        "original_spec_path": str(subject_path),
        "original_spec_sha256": subject["content_hash"],
        "parent_recovery_spec_path": None if parent_recovery is None else str(subject_path),
        "parent_recovery_spec_sha256": None if parent_recovery is None else subject["content_hash"],
        "original_ledger_path": str(Path(original_ledger).resolve()),
        "original_ledger_sha256": ledger["content_hash"],
        "monitor_report_path": str(Path(monitor_report).resolve()),
        "monitor_report_sha256": monitor_hash,
        "recovery_root": str(root), "project_dir": str(project),
        "source_commit": source_commit, "failed_tasks": list(failed_tasks),
        "original_source_commit": subject["source_commit"],
        "changed_files": list(changed),
        "repair_authorization_phrase": repair_authorization_phrase if changed else None,
        "recovery_tasks": list(closure), "resources": resources,
        "completed_artifact_attestations": completed_attestations,
        "foundation_reuse_lock_sha256": spec["reuse_lock_sha256"],
        "graph_sha256": spec["graph_sha256"], "recipe_sha256": spec["recipe_sha256"],
        "scientific_graph_unchanged": True, "completed_outputs_preserved": True,
        "final_test_accessed": False,
    })
    if profile != PROFILE_C25P75:
        raw_recovery = {
            key: item for key, item in recovery.items() if key != "content_hash"
        }
        raw_recovery["recipe_profile"] = profile
        recovery = with_content_hash(raw_recovery)
    commands = []
    closure_set = set(closure)
    recovery_prefix = {
        PROFILE_C25P75: "hcwmhpe_r",
        PROFILE_C10P90: "hcwmhpe90_r",
        PROFILE_C25P75_300K60: "hcwmhpe25p_r",
        PROFILE_C10P90_300K60: "hcwmhpe90p_r",
        PROFILE_DENSE_ANCHOR50_300K60: "hcwmhped_r",
    }[profile]
    for task in campaign_tasks(profile):
        if task["task_id"] not in closure_set:
            continue
        resource = resources[task["resource_class"]]
        deps = [item for item in task["dependencies"] if item in closure_set]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}", f"--partition={PARTITION}",
            f"--cpus-per-task={resource['cpus']}", f"--mem={resource['memory']}",
            f"--time={resource['walltime']}",
            f"--job-name={recovery_prefix}_{task['task_id']}",
        ]
        if resource.get("gpu"):
            command.extend((f"--gres={resource['gpu']}", "--signal=B:USR1@120"))
        if deps:
            command.append("--dependency=afterok:" + ":".join(f"${{JOB_{item}}}" for item in deps))
        command.extend((
            "--export=ALL," + f"PROJECT_DIR={project},HCWDL_MHPE_RECOVERY_SPEC={root / 'recovery_spec.json'},HCWDL_MHPE_TASK={task['task_id']}",
            str(project / "sbatch/run_hcwdl_mhpe_recovery_task.sh"),
        ))
        commands.append({"task_id": task["task_id"], "dependencies": deps, "command": command})
    plan = with_content_hash({
        "contract": COMMAND_PLAN_CONTRACT, "schema_version": 1,
        "spec_sha256": recovery["content_hash"], "commands": commands,
        "recovery": True, "mutated": False, "final_test_accessed": False,
    })
    if publish:
        root.mkdir(parents=True, exist_ok=False)
        write_immutable_json(root / "recovery_spec.json", recovery)
        write_immutable_json(root / "command_plan.json", plan)
    return recovery


def validate_recovery(value: Mapping[str, Any]) -> str:
    contract = str(value.get("contract"))
    if contract not in {RECOVERY_SPEC_CONTRACT, RESOURCE_RECOVERY_SPEC_CONTRACT}:
        raise ValueError("HCWDL-MHPE recovery contract differs")
    digest = validate_content_hash(value, expected_contract=contract, expected_schema_version=1)
    campaign = load_json(value["campaign_spec_path"])
    profile = campaign_profile(campaign)
    if tuple(value.get("recovery_tasks", ())) != failed_downstream_closure(
        value.get("failed_tasks", ()), profile=profile,
    ):
        raise ValueError("HCWDL-MHPE recovery closure differs")
    if ((profile != PROFILE_C25P75)
            != (value.get("recipe_profile") == profile)):
        raise ValueError("HCWDL-MHPE recovery recipe profile differs")
    monitor = load_json(value["monitor_report_path"])
    if (validate_content_hash(monitor, expected_contract=MONITOR_CONTRACT, expected_schema_version=1)
            != value.get("monitor_report_sha256")):
        raise ValueError("HCWDL-MHPE recovery monitor changed")
    ledger = load_json(value["original_ledger_path"])
    if validate_submission_ledger(ledger) != value.get("original_ledger_sha256"):
        raise ValueError("HCWDL-MHPE recovery ledger changed")
    subject = load_json(value["original_spec_path"])
    if (validate_content_hash(
            subject, expected_contract=str(subject.get("contract")),
            expected_schema_version=int(subject.get("schema_version", -1)),
        ) != value.get("original_spec_sha256")
            or ledger.get("campaign_spec_sha256") != subject.get("content_hash")):
        raise ValueError("HCWDL-MHPE recovery subject changed")
    if (validate_campaign(campaign, verify_source_tree=False) != value.get("campaign_spec_sha256")
            or campaign.get("reuse_lock_sha256") != value.get("foundation_reuse_lock_sha256")
            or campaign.get("graph_sha256") != value.get("graph_sha256")
            or campaign.get("recipe_sha256") != value.get("recipe_sha256")):
        raise ValueError("HCWDL-MHPE recovery science lineage changed")
    parent_path = value.get("parent_recovery_spec_path")
    parent_hash = value.get("parent_recovery_spec_sha256")
    if (parent_path is None) != (parent_hash is None):
        raise ValueError("HCWDL-MHPE parent recovery lineage differs")
    if parent_path is not None:
        parent = load_json(parent_path)
        if validate_recovery(parent) != parent_hash or subject != parent:
            raise ValueError("HCWDL-MHPE parent recovery changed")
    for row in value.get("completed_artifact_attestations", ()):
        attestation = load_json(row["path"])
        if validate_task_attestation(
            attestation, campaign_spec_sha256=value["original_spec_sha256"],
            task_id=row["task_id"], array_index=None,
        ) != row.get("attestation_sha256"):
            raise ValueError("HCWDL-MHPE completed artifact changed")
    expected_completed = sorted(
        str(row["task_id"]) for row in monitor["rows"]
        if row.get("disposition") == "complete"
    )
    if sorted(row["task_id"] for row in value.get("completed_artifact_attestations", ())) != expected_completed:
        raise ValueError("HCWDL-MHPE completed artifact registry differs")
    if not re.fullmatch(r"[0-9a-f]{40}", str(value.get("source_commit"))):
        raise ValueError("HCWDL-MHPE recovery source commit differs")
    changed = set(map(str, value.get("changed_files", ())))
    if value["source_commit"] == value.get("original_source_commit"):
        if changed:
            raise ValueError("HCWDL-MHPE unchanged-source recovery differs")
    elif (not changed or not changed <= SOURCE_REPAIR_ALLOWLIST
          or value.get("repair_authorization_phrase") != SOURCE_REPAIR_PHRASE):
        raise PermissionError("HCWDL-MHPE source recovery authorization differs")
    baseline = subject["resources"]
    if set(value.get("resources", {})) != set(baseline):
        raise ValueError("HCWDL-MHPE recovery resource classes differ")
    for name, resources in value.get("resources", {}).items():
        if name not in baseline or resources.get("gpu") != baseline[name].get("gpu"):
            raise ValueError("HCWDL-MHPE recovery GPU/resource class differs")
        if (int(resources["cpus"]) < int(baseline[name]["cpus"])
                or _memory_gib(str(resources["memory"])) < _memory_gib(str(baseline[name]["memory"]))
                or _wall_seconds(str(resources["walltime"])) < _wall_seconds(str(baseline[name]["walltime"]))):
            raise ValueError("HCWDL-MHPE recovery resources decreased")
    if value.get("scientific_graph_unchanged") is not True or value.get("completed_outputs_preserved") is not True or value.get("final_test_accessed") is not False:
        raise ValueError("HCWDL-MHPE recovery semantics differ")
    plan = load_json(Path(value["recovery_root"]) / "command_plan.json")
    validate_content_hash(
        plan, expected_contract=COMMAND_PLAN_CONTRACT, expected_schema_version=1,
    )
    if (plan.get("spec_sha256") != digest
            or [row["task_id"] for row in plan.get("commands", ())]
            != list(value["recovery_tasks"])):
        raise ValueError("HCWDL-MHPE recovery command plan differs")
    return digest


def _memory_gib(value: str) -> int:
    if not value.endswith("G") or not value[:-1].isdigit():
        raise ValueError("HCWDL-MHPE memory format differs")
    return int(value[:-1])


def _wall_seconds(value: str) -> int:
    parts = value.split(":")
    if len(parts) != 3 or any(not item.isdigit() for item in parts):
        raise ValueError("HCWDL-MHPE walltime format differs")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


__all__ = ["create_recovery", "failed_downstream_closure", "validate_recovery"]

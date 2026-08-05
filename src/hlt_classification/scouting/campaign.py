"""Immutable minimum-storage PMARD experiment registry and Tigris DAG."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping, Sequence

from hlt_classification.data.cache_contracts import canonical_sha256, require_sha256, validate_content_hash, with_content_hash
from hlt_classification.provenance import validate_source_snapshot_payload
from .repair import ALPHA_GRID
from .schema import SCOUTING_SCHEMA_SHA256
from .training import CONFIRMATION_SEEDS, KD_MIXTURES, REPRESENTATION_ARMS, TEMPERATURE_GRID

PMARD_CAMPAIGN_SPEC_CONTRACT = "hlt_classification_pmard_campaign_spec_v1"
PMARD_LEDGER_CONTRACT = "hlt_classification_pmard_submission_ledger_v1"
PMARD_DRY_RUN_CONTRACT = "hlt_classification_pmard_production_dry_run_v1"
PMARD_SITE = {
    "account": "reu-aisocial", "partition": "tigris",
    "project_dir": "/home/ryreu/atlas/HLT_Classification",
    "data_root": "/home/ryreu/cms/data/ScoutingAK8_native_compact/2024/train",
    "conda_environment": "atlas_kd_tigris", "gpu_gres": "gpu:gh200:1",
}


@dataclass(frozen=True)
class PmardTask:
    name: str
    dependencies: tuple[str, ...]
    cpus: int = 4
    memory: str = "32G"
    walltime: str = "02:00:00"
    gpu: bool = False
    array: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {**self.__dict__, "dependencies": list(self.dependencies), "worker": "run_pmard_task.py"}


def experiment_registry() -> dict[str, object]:
    return {
        "alphas": list(ALPHA_GRID), "matcher_arms": [f"M{i}" for i in range(6)],
        "temperatures": list(TEMPERATURE_GRID),
        "kd_arms": {name: list(values) for name, values in KD_MIXTURES.items()},
        "representation_arms": list(REPRESENTATION_ARMS),
        "repair_controls": [
            "P4_ONLY", "TRACK_ONLY", "P4_PLUS_TRACK", "DIRECTION_ONLY",
            "RESPONSE_ONLY", "WRONG_DIRECTION", "RANDOM_DIRECTION",
            "LOG_ANGULAR", "CONFIDENCE_WEIGHTED", "MATCH_SHUFFLED",
        ],
        "generations": [0, 1, 2, 3], "screen_seed": 1337,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
    }


def pmard_tasks(*, smoke: bool) -> tuple[PmardTask, ...]:
    cpu = "00:30:00" if smoke else "08:00:00"
    gpu = "01:00:00" if smoke else "48:00:00"
    tasks = [
        PmardTask("source_audit", (), walltime=cpu),
        PmardTask("splits", ("source_audit",), walltime=cpu),
        PmardTask("feature_audit", ("splits",), cpus=8, memory="96G", walltime=cpu),
        PmardTask("data_lock", ("feature_audit",), walltime=cpu),
        PmardTask("matcher_design_lock", ("data_lock",), walltime=cpu),
        PmardTask("matcher_crossfit", ("matcher_design_lock",), cpus=16, memory="192G", walltime=gpu, gpu=True, array="0-4%2"),
        PmardTask("matcher_validation", ("matcher_crossfit",), cpus=16, memory="192G", walltime=gpu, gpu=True),
        PmardTask("matcher_result_lock", ("matcher_validation",), walltime=cpu),
        PmardTask("weaver_parity", ("data_lock",), walltime=gpu, gpu=True),
        PmardTask("budget_grid", ("weaver_parity", "matcher_result_lock"), cpus=8, memory="96G", walltime=gpu, gpu=True, array="0-11%2"),
        PmardTask("budget_selection", ("budget_grid",), walltime=cpu),
        PmardTask("temperature_grid", ("budget_selection",), cpus=8, memory="192G", walltime=gpu, gpu=True, array="0-2%2"),
        PmardTask("training_lock", ("temperature_grid",), walltime=cpu),
        PmardTask("teachers", ("training_lock",), cpus=8, memory="192G", walltime=gpu, gpu=True, array="0-6%2"),
        PmardTask("k2_alpha_sweep", ("teachers",), cpus=8, memory="192G", walltime=gpu, gpu=True, array="0-5%2"),
        PmardTask("alpha_selection", ("k2_alpha_sweep",), walltime=cpu),
        PmardTask("kd_controls", ("alpha_selection",), cpus=8, memory="192G", walltime=gpu, gpu=True, array="0-6%2"),
        PmardTask("mechanism_controls", ("kd_controls",), cpus=8, memory="192G", walltime=gpu, gpu=True, array="0-21%2"),
        PmardTask("representation", ("mechanism_controls",), cpus=8, memory="192G", walltime=gpu, gpu=True, array="0-10%2"),
        PmardTask("generation_1", ("representation",), cpus=8, memory="192G", walltime=gpu, gpu=True),
        PmardTask("generation_2", ("generation_1",), cpus=8, memory="192G", walltime=gpu, gpu=True),
        PmardTask("screen_selection", ("generation_2",), walltime=cpu),
        PmardTask("screen_confirmation_lock", ("screen_selection",), walltime=cpu),
        PmardTask("confirmation", ("screen_confirmation_lock",), cpus=8, memory="192G", walltime=gpu, gpu=True, array="0-4%2"),
    ]
    if smoke:
        tasks.append(PmardTask("miniature_summary", ("confirmation",), walltime=cpu))
    else:
        tasks.extend((
            PmardTask("finalist_lock", ("confirmation",), walltime=cpu),
            PmardTask("execution_lock", ("finalist_lock",), walltime=cpu),
            PmardTask("final_test", ("execution_lock",), cpus=8, memory="192G", walltime=gpu, gpu=True),
            PmardTask("aggregate_report", ("final_test",), walltime=cpu),
        ))
    return tuple(tasks)


def _validate_dag(tasks: Sequence[Mapping[str, object]]) -> None:
    seen: set[str] = set()
    for task in tasks:
        name = str(task["name"])
        if name in seen or any(str(dep) not in seen for dep in task["dependencies"]):
            raise ValueError("PMARD task DAG is duplicated or not topological")
        seen.add(name)


def create_pmard_campaign_spec(
    *, source_snapshot: Mapping[str, Any], source_manifest_sha256: str,
    split_manifest_sha256: str, campaign_root: str, mode: str = "smoke",
    production_authorized: bool = False, miniature_report_sha256: str | None = None,
    dry_run_report_sha256: str | None = None, resource_evidence_sha256: str | None = None,
    storage_evidence_sha256: str | None = None,
    evidence_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if mode not in {"smoke", "production"}:
        raise ValueError("PMARD mode must be smoke or production")
    validate_source_snapshot_payload(source_snapshot)
    if source_snapshot.get("worktree_clean") is not True:
        raise ValueError("campaign source snapshot must be clean")
    evidence = {
        "miniature_report_sha256": miniature_report_sha256,
        "dry_run_report_sha256": dry_run_report_sha256,
        "resource_evidence_sha256": resource_evidence_sha256,
        "storage_evidence_sha256": storage_evidence_sha256,
    }
    if mode == "production":
        if not production_authorized:
            raise PermissionError("full PMARD campaign requires explicit authorization")
        if evidence_artifacts is None:
            raise ValueError("production PMARD requires validated evidence artifacts, not bare hashes")
        from .evidence import validate_miniature_report, validate_resource_evidence, validate_storage_evidence
        miniature = evidence_artifacts.get("miniature_report", {})
        dry_run = evidence_artifacts.get("dry_run_report", {})
        resource = evidence_artifacts.get("resource_evidence", {})
        storage = evidence_artifacts.get("storage_evidence", {})
        resource_hash = validate_resource_evidence(resource)
        storage_hash = validate_storage_evidence(storage, resource_evidence=resource)
        miniature_hash = validate_miniature_report(miniature)
        dry_run_hash = validate_content_hash(dry_run, expected_contract=PMARD_DRY_RUN_CONTRACT)
        if dry_run.get("dry_run") is not True or dry_run.get("mutated") is not False:
            raise ValueError("PMARD dry-run evidence is not a nonmutating dry run")
        expected_lineage = {
            "source_snapshot_sha256": source_snapshot["source_snapshot_sha256"],
            "source_manifest_sha256": source_manifest_sha256,
            "split_manifest_sha256": split_manifest_sha256,
        }
        for payload in (miniature, resource):
            if any(payload.get(key) != value for key, value in expected_lineage.items()):
                raise ValueError("PMARD evidence source/data lineage differs")
        if any(dry_run.get(key) != value for key, value in expected_lineage.items()):
            raise ValueError("PMARD production dry-run lineage differs")
        if miniature.get("resource_evidence_sha256") != resource_hash or miniature.get("storage_evidence_sha256") != storage_hash:
            raise ValueError("PMARD miniature evidence bundle differs")
        evidence = {
            "miniature_report_sha256": miniature_hash,
            "dry_run_report_sha256": dry_run_hash,
            "resource_evidence_sha256": resource_hash,
            "storage_evidence_sha256": storage_hash,
        }
        supplied = (miniature_report_sha256, dry_run_report_sha256, resource_evidence_sha256, storage_evidence_sha256)
        if any(value is not None for value in supplied) and supplied != tuple(evidence.values()):
            raise ValueError("supplied PMARD evidence hashes differ from validated artifacts")
    elif production_authorized or any(value is not None for value in evidence.values()):
        raise ValueError("smoke campaign may not claim production authorization/evidence")
    tasks = [item.to_dict() for item in pmard_tasks(smoke=mode == "smoke")]
    _validate_dag(tasks)
    identity = canonical_sha256({
        "source_snapshot_sha256": source_snapshot["source_snapshot_sha256"],
        "source_manifest_sha256": source_manifest_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "mode": mode, "registry": experiment_registry(), "evidence": evidence,
    })
    return with_content_hash({
        "contract": PMARD_CAMPAIGN_SPEC_CONTRACT, "schema_version": 1,
        "campaign_id": f"pmard_{mode}_{identity[:16]}", "mode": mode,
        "production_authorized": production_authorized,
        "campaign_root": campaign_root.rstrip("/"), "site": dict(PMARD_SITE),
        "source_snapshot": dict(source_snapshot),
        "source_manifest_sha256": require_sha256(source_manifest_sha256, name="source_manifest_sha256"),
        "split_manifest_sha256": require_sha256(split_manifest_sha256, name="split_manifest_sha256"),
        "scouting_schema_sha256": SCOUTING_SCHEMA_SHA256,
        "storage_profile": "root_streaming_ram_ephemeral_targets_v1",
        "forbidden_durable_artifacts": [
            "repaired_dataset", "training_teacher_logits", "dense_pair_tensor",
            "full_assignment_table", "per_epoch_representation",
        ],
        "registry": experiment_registry(), "tasks": tasks, "evidence": evidence,
    })


def validate_pmard_campaign_spec(spec: Mapping[str, Any]) -> str:
    digest = validate_content_hash(spec, expected_contract=PMARD_CAMPAIGN_SPEC_CONTRACT)
    validate_source_snapshot_payload(spec.get("source_snapshot", {}))
    if spec["source_snapshot"].get("worktree_clean") is not True:
        raise ValueError("campaign source snapshot is not clean")
    source_hash = require_sha256(spec.get("source_manifest_sha256"), name="source_manifest_sha256")
    split_hash = require_sha256(spec.get("split_manifest_sha256"), name="split_manifest_sha256")
    mode = spec.get("mode")
    if mode not in {"smoke", "production"} or spec.get("site") != PMARD_SITE:
        raise ValueError("PMARD campaign mode/site differs")
    if spec.get("scouting_schema_sha256") != SCOUTING_SCHEMA_SHA256:
        raise ValueError("campaign Scouting schema differs")
    expected_identity = canonical_sha256({
        "source_snapshot_sha256": spec["source_snapshot"]["source_snapshot_sha256"],
        "source_manifest_sha256": source_hash, "split_manifest_sha256": split_hash,
        "mode": mode, "registry": experiment_registry(), "evidence": spec.get("evidence"),
    })
    if spec.get("campaign_id") != f"pmard_{mode}_{expected_identity[:16]}":
        raise ValueError("PMARD campaign scientific identity differs")
    if spec.get("registry") != experiment_registry():
        raise ValueError("PMARD experiment registry differs")
    _validate_dag(spec.get("tasks", ()))
    expected_tasks = [item.to_dict() for item in pmard_tasks(smoke=mode == "smoke")]
    if spec.get("tasks") != expected_tasks:
        raise ValueError("PMARD task registry/resources differ")
    if spec.get("mode") == "production":
        if spec.get("production_authorized") is not True:
            raise PermissionError("production PMARD campaign is not authorized")
        for name, value in spec.get("evidence", {}).items():
            require_sha256(value, name=name)
        if set(spec.get("evidence", {})) != {
            "miniature_report_sha256", "dry_run_report_sha256",
            "resource_evidence_sha256", "storage_evidence_sha256",
        }:
            raise ValueError("production PMARD evidence set differs")
    elif spec.get("production_authorized") is not False or any(
        value is not None for value in spec.get("evidence", {}).values()
    ):
        raise ValueError("smoke PMARD spec claims production evidence")
    return digest


def create_pmard_production_dry_run(
    *, source_snapshot: Mapping[str, Any], source_manifest_sha256: str,
    split_manifest_sha256: str, campaign_root: str, spec_path: str,
) -> dict[str, Any]:
    """Render the full production DAG without creating or submitting a campaign."""
    validate_source_snapshot_payload(source_snapshot)
    if source_snapshot.get("worktree_clean") is not True:
        raise ValueError("PMARD production dry run requires clean source")
    source_hash = require_sha256(source_manifest_sha256, name="source_manifest_sha256")
    split_hash = require_sha256(split_manifest_sha256, name="split_manifest_sha256")
    tasks = [item.to_dict() for item in pmard_tasks(smoke=False)]; _validate_dag(tasks)
    preview = {
        "campaign_id": "pmard_production_preview", "site": dict(PMARD_SITE),
    }
    jobs: dict[str, str] = {}; commands = []
    for task in tasks:
        deps = [jobs[name] for name in task["dependencies"]]
        command = [
            "sbatch", "--parsable", f"--account={PMARD_SITE['account']}",
            f"--partition={PMARD_SITE['partition']}", f"--cpus-per-task={task['cpus']}",
            f"--mem={task['memory']}", f"--time={task['walltime']}",
            f"--job-name={preview['campaign_id']}_{task['name']}",
        ]
        if task.get("gpu"): command.append(f"--gres={PMARD_SITE['gpu_gres']}")
        if task.get("array"): command.append(f"--array={task['array']}")
        if deps: command.append("--dependency=afterok:" + ":".join(deps))
        command.extend([
            f"--export=ALL,PMARD_SPEC={spec_path},PMARD_TASK={task['name']}",
            f"{PMARD_SITE['project_dir']}/sbatch/run_pmard_task.sh",
        ])
        commands.append(command); jobs[task["name"]] = str(80_000 + len(jobs))
    return with_content_hash({
        "contract": PMARD_DRY_RUN_CONTRACT, "schema_version": 1,
        "dry_run": True, "mutated": False,
        "source_snapshot_sha256": source_snapshot["source_snapshot_sha256"],
        "source_manifest_sha256": source_hash, "split_manifest_sha256": split_hash,
        "campaign_root": campaign_root.rstrip("/"), "tasks": tasks,
        "placeholder_jobs": jobs, "commands": commands,
    })


def sbatch_command(
    spec: Mapping[str, Any], task: Mapping[str, Any], dependency_ids: Sequence[str],
    *, spec_path: str,
) -> list[str]:
    validate_pmard_campaign_spec(spec)
    if any(not re.fullmatch(r"[1-9][0-9]*(?:_[0-9]+)?", item) for item in dependency_ids):
        raise ValueError("dependency contains a nonnumeric Slurm ID")
    command = [
        "sbatch", "--parsable", f"--account={spec['site']['account']}",
        f"--partition={spec['site']['partition']}", f"--cpus-per-task={task['cpus']}",
        f"--mem={task['memory']}", f"--time={task['walltime']}",
        f"--job-name={spec['campaign_id']}_{task['name']}",
    ]
    if task.get("gpu"):
        command.append(f"--gres={spec['site']['gpu_gres']}")
    if task.get("array"):
        command.append(f"--array={task['array']}")
    if dependency_ids:
        command.append("--dependency=afterok:" + ":".join(dependency_ids))
    command.extend([
        f"--export=ALL,PMARD_SPEC={spec_path},PMARD_TASK={task['name']}",
        f"{spec['site']['project_dir']}/sbatch/run_pmard_task.sh",
    ])
    return command


def submit_pmard_campaign(
    spec: Mapping[str, Any], *, spec_path: str, dry_run: bool,
    runner: Callable[[Sequence[str]], str] | None = None,
) -> dict[str, Any]:
    spec_hash = validate_pmard_campaign_spec(spec)
    jobs: dict[str, str] = {}; commands = []
    for task in spec["tasks"]:
        deps = [jobs[name] for name in task["dependencies"]]
        command = sbatch_command(spec, task, deps, spec_path=spec_path)
        commands.append(command)
        if dry_run:
            jobs[task["name"]] = str(10_000 + len(jobs))
        else:
            if runner is None:
                raise ValueError("live submission requires an explicit command runner")
            output = runner(command).strip().split(";")[0]
            if not re.fullmatch(r"[1-9][0-9]*", output):
                raise RuntimeError(f"sbatch returned invalid job id {output!r}")
            jobs[task["name"]] = output
    return with_content_hash({
        "contract": PMARD_LEDGER_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": spec_hash, "campaign_id": spec["campaign_id"],
        "dry_run": dry_run, "mutated": not dry_run, "jobs": jobs, "commands": commands,
    })


__all__ = [
    "PMARD_CAMPAIGN_SPEC_CONTRACT", "PMARD_DRY_RUN_CONTRACT", "PMARD_SITE", "PmardTask",
    "create_pmard_campaign_spec", "create_pmard_production_dry_run", "experiment_registry", "pmard_tasks",
    "sbatch_command", "submit_pmard_campaign", "validate_pmard_campaign_spec",
]

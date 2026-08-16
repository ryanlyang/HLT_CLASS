"""Source-pinned full-data and paired 300k campaign construction for HCWDL-MHPE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, sha256_file, validate_content_hash,
    with_content_hash, write_immutable_json,
)
from .engine import validate_pmard_training_report
from .hcwdl_recipe import validate_recipe as validate_foundation_recipe
from .hcwdl_unified_balanced_full_campaign import validate_foundation_campaign
from .hcwdl_unified_balanced_full_contracts import validate_foundation_lock
from .hcwdl_unified_balanced_campaign import (
    SEMANTIC_SOURCE_FILES as UB_300K_SEMANTIC_SOURCE_FILES,
)
from .hcwdl_unified_balanced_contracts import (
    validate_foundation_lock as validate_300k_foundation_lock,
    validate_foundation_spec as validate_300k_foundation_spec,
)
from .hcwdl_unified_balanced_targets import validate_target_manifest
from .hcwdl_unified_balanced_runner import inspect_shared_u000_target_lineage

from .hcwdl_mhpe_contracts import (
    COMMAND_PLAN_CONTRACT, campaign_profile, campaign_spec_contract,
    graph_payload, recipe_payload, reuse_lock_payload, validate_graph,
    validate_recipe, validate_reuse_lock, validate_waiver, waiver_payload,
)
from .hcwdl_mhpe_graph import (
    PROFILE_C10P90, PROFILE_C25P75,
    PROFILE_C10P90_300K60, PROFILE_C25P75_300K60, SUPPORTED_PROFILES,
    PROFILE_DENSE_ANCHOR50_300K60, direct_model_teacher,
    endpoint_ensemble, ensemble_components, node_registry, stages,
)
from .hcwdl_unified_balanced_coarse_campaign import FOUNDATION_CORE_FILES

ACCOUNT: Final = "reu-aisocial"
PARTITION: Final = "tigris"
CREATION_PHRASE: Final = "AUTHORIZE HCWDL MHPE FULL EXACT SPEC"
SUBMISSION_PHRASE: Final = "SUBMIT HCWDL MHPE FULL EXACT LEDGER"
WAIVER_PHRASE: Final = "AUTHORIZE HCWDL MHPE FULL DIRECT EXECUTION WITHOUT NEW SMOKE"
CREATION_PHRASE_C10P90: Final = "AUTHORIZE HCWDL MHPE C10P90 FULL EXACT SPEC"
SUBMISSION_PHRASE_C10P90: Final = "SUBMIT HCWDL MHPE C10P90 FULL EXACT LEDGER"
WAIVER_PHRASE_C10P90: Final = "AUTHORIZE HCWDL MHPE C10P90 FULL DIRECT EXECUTION WITHOUT NEW SMOKE"
CREATION_PHRASE_C25P75_300K60: Final = "AUTHORIZE HCWDL MHPE C25P75 300K60 EXACT SPEC"
SUBMISSION_PHRASE_C25P75_300K60: Final = "SUBMIT HCWDL MHPE C25P75 300K60 EXACT LEDGER"
WAIVER_PHRASE_C25P75_300K60: Final = "AUTHORIZE HCWDL MHPE C25P75 300K60 DIRECT EXECUTION"
CREATION_PHRASE_C10P90_300K60: Final = "AUTHORIZE HCWDL MHPE C10P90 300K60 EXACT SPEC"
SUBMISSION_PHRASE_C10P90_300K60: Final = "SUBMIT HCWDL MHPE C10P90 300K60 EXACT LEDGER"
WAIVER_PHRASE_C10P90_300K60: Final = "AUTHORIZE HCWDL MHPE C10P90 300K60 DIRECT EXECUTION"
CREATION_PHRASE_DENSE_ANCHOR50_300K60: Final = "AUTHORIZE HCWDL MHPE DENSE ANCHOR50 300K60 EXACT SPEC"
SUBMISSION_PHRASE_DENSE_ANCHOR50_300K60: Final = "SUBMIT HCWDL MHPE DENSE ANCHOR50 300K60 EXACT LEDGER"
WAIVER_PHRASE_DENSE_ANCHOR50_300K60: Final = "AUTHORIZE HCWDL MHPE DENSE ANCHOR50 300K60 DIRECT EXECUTION"

REUSED_FOUNDATION_EXACT_FILES: Final = tuple(FOUNDATION_CORE_FILES)

SEMANTIC_SOURCE_FILES: Final = REUSED_FOUNDATION_EXACT_FILES + (
    # These execution producers include the source-pinned repairs used to
    # finish the imported FULL3 foundation. Their products are authenticated
    # by the foundation lock; comparing them to the foundation specification's
    # original (pre-recovery) source would reject that corrected lineage.
    "src/hlt_classification/scouting/hcwdl_upper_builder.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_builder.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_targets.py",
    "src/hlt_classification/scouting/targets.py",
    "src/hlt_classification/scouting/training.py",
    "src/hlt_classification/scouting/dataset.py",
    "src/hlt_classification/scouting/loaders.py",
    "src/hlt_classification/scouting/selective_assignment.py",
    "src/hlt_classification/scouting/view_cache.py",
    "src/hlt_classification/scouting/hcwdl_unified_balanced_runner.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_graph.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_contracts.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_targets.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_campaign.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_runner.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_workflow.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_recovery.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_final.py",
    "scripts/run_hcwdl_mhpe_task.py",
    "scripts/run_hcwdl_mhpe_recovery_task.py",
    "sbatch/common.sh",
    "sbatch/run_hcwdl_mhpe_task.sh",
    "sbatch/run_hcwdl_mhpe_recovery_task.sh",
)

IMPLEMENTATION_EVIDENCE_FILES: Final = (
    "docs/plans/HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_IMPLEMENTATION_PLAN.md",
    "docs/contracts/HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE.md",
    "docs/HCWDL_MHPE_RUNBOOK.md",
    "tests/test_hcwdl_mhpe.py",
)
C10P90_IMPLEMENTATION_EVIDENCE_FILES: Final = IMPLEMENTATION_EVIDENCE_FILES + (
    "docs/plans/HCWDL_MHPE_C10P90_PARALLEL_PLAN.md",
    "docs/HCWDL_MHPE_C10P90_RUNBOOK.md",
)
P300_IMPLEMENTATION_EVIDENCE_FILES: Final = IMPLEMENTATION_EVIDENCE_FILES + (
    "docs/plans/HCWDL_MHPE_300K_60E_PAIRED_PLAN.md",
    "docs/HCWDL_MHPE_300K_60E_RUNBOOK.md",
)
DENSE_IMPLEMENTATION_EVIDENCE_FILES: Final = IMPLEMENTATION_EVIDENCE_FILES + (
    "docs/plans/HCWDL_MHPE_DENSE_ANCHOR50_300K60_PLAN.md",
    "docs/HCWDL_MHPE_DENSE_ANCHOR50_300K60_RUNBOOK.md",
)

ADDITIVE_ADAPTER_FILES: Final = frozenset({
    "src/hlt_classification/scouting/engine.py",
    "src/hlt_classification/scouting/hcwdl_training.py",
})


@dataclass(frozen=True)
class ResourceRequest:
    cpus: int
    memory: str
    walltime: str
    gpu: str | None = None


RESOURCES: Final = {
    "gpu_training": ResourceRequest(8, "256G", "24:00:00", "gpu:gh200:1"),
    # Reducers prepare the same all-mapped train+validation views as FULL3.
    # Keep its measured 256-GiB/24-hour class; the component logits add only
    # about one GiB even at the five-model D000 stage.
    "gpu_targets": ResourceRequest(8, "256G", "24:00:00", "gpu:gh200:1"),
    "cpu_report": ResourceRequest(4, "64G", "04:00:00"),
}
P300_RESOURCES: Final = {
    "gpu_training": ResourceRequest(8, "96G", "06:00:00", "gpu:gh200:1"),
    "gpu_targets": ResourceRequest(8, "96G", "06:00:00", "gpu:gh200:1"),
    "cpu_report": ResourceRequest(4, "32G", "01:00:00"),
}


def resources_for_profile(profile: str) -> Mapping[str, ResourceRequest]:
    return P300_RESOURCES if profile in {
        PROFILE_C25P75_300K60, PROFILE_C10P90_300K60,
        PROFILE_DENSE_ANCHOR50_300K60,
    } else RESOURCES


def semantic_source_hashes(repository: str | Path) -> dict[str, str]:
    root = Path(repository).resolve()
    return {name: sha256_file(root / name) for name in SEMANTIC_SOURCE_FILES}


def creation_phrase(profile: str = PROFILE_C25P75) -> str:
    if profile == PROFILE_C25P75:
        return CREATION_PHRASE
    if profile == PROFILE_C10P90:
        return CREATION_PHRASE_C10P90
    if profile == PROFILE_C25P75_300K60:
        return CREATION_PHRASE_C25P75_300K60
    if profile == PROFILE_C10P90_300K60:
        return CREATION_PHRASE_C10P90_300K60
    if profile == PROFILE_DENSE_ANCHOR50_300K60:
        return CREATION_PHRASE_DENSE_ANCHOR50_300K60
    raise ValueError("unknown HCWDL-MHPE recipe profile")


def submission_phrase(profile: str = PROFILE_C25P75) -> str:
    if profile == PROFILE_C25P75:
        return SUBMISSION_PHRASE
    if profile == PROFILE_C10P90:
        return SUBMISSION_PHRASE_C10P90
    if profile == PROFILE_C25P75_300K60:
        return SUBMISSION_PHRASE_C25P75_300K60
    if profile == PROFILE_C10P90_300K60:
        return SUBMISSION_PHRASE_C10P90_300K60
    if profile == PROFILE_DENSE_ANCHOR50_300K60:
        return SUBMISSION_PHRASE_DENSE_ANCHOR50_300K60
    raise ValueError("unknown HCWDL-MHPE recipe profile")


def evidence_files(profile: str = PROFILE_C25P75) -> tuple[str, ...]:
    if profile == PROFILE_C25P75:
        return IMPLEMENTATION_EVIDENCE_FILES
    if profile == PROFILE_C10P90:
        return C10P90_IMPLEMENTATION_EVIDENCE_FILES
    if profile in {PROFILE_C25P75_300K60, PROFILE_C10P90_300K60}:
        return P300_IMPLEMENTATION_EVIDENCE_FILES
    if profile == PROFILE_DENSE_ANCHOR50_300K60:
        return DENSE_IMPLEMENTATION_EVIDENCE_FILES
    raise ValueError("unknown HCWDL-MHPE recipe profile")


def campaign_tasks(profile: str = PROFILE_C25P75) -> list[dict[str, Any]]:
    registry = node_registry(profile)
    tasks: list[dict[str, Any]] = []
    previous = None
    stage_names = tuple(item for item in stages(profile) if item != "M1")
    direct_stage = direct_model_teacher(profile)
    for stage in stage_names:
        stage_nodes = sorted(name for name in registry if name.startswith(stage + "_from_"))
        dependencies = [] if previous is None else [previous]
        # The first stage is an imported-root child; later stages wait for the
        # preceding durable ensemble.
        for node_id in stage_nodes:
            tasks.append({
                "task_id": f"train_{node_id}", "kind": "train", "node_id": node_id,
                "dependencies": list(dependencies), "resource_class": "gpu_training",
            })
        if stage != direct_stage:
            ensemble_id = stage + "E"
            tasks.append({
                "task_id": f"ensemble_{ensemble_id}", "kind": "ensemble",
                "ensemble_id": ensemble_id,
                "dependencies": [f"train_{node}" for node in stage_nodes],
                "resource_class": "gpu_targets",
            })
            previous = f"ensemble_{ensemble_id}"
        else:
            previous = f"train_{stage_nodes[0]}"
    tasks.append({
        "task_id": "train_M1", "kind": "train", "node_id": "M1",
        "dependencies": [f"ensemble_{endpoint_ensemble(profile)}"], "resource_class": "gpu_training",
    })
    terminal = [f"train_{name}" for name in registry]
    terminal += [f"ensemble_{name}" for name in ensemble_components(profile)]
    tasks.extend((
        {"task_id": "aggregate", "kind": "aggregate", "dependencies": terminal, "resource_class": "cpu_report"},
        {"task_id": "finalist_lock", "kind": "finalist_lock", "dependencies": ["aggregate"], "resource_class": "cpu_report"},
        {"task_id": "campaign_complete", "kind": "campaign_complete", "dependencies": ["finalist_lock"], "resource_class": "cpu_report"},
    ))
    if len([row for row in tasks if row["kind"] == "train"]) != len(registry):
        raise RuntimeError("HCWDL-MHPE task graph fit count differs")
    return tasks


def command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    profile = campaign_profile(spec)
    job_prefix = {
        PROFILE_C25P75: "hcwmhpe", PROFILE_C10P90: "hcwmhpe90",
        PROFILE_C25P75_300K60: "hcwmhpe25p",
        PROFILE_C10P90_300K60: "hcwmhpe90p",
        PROFILE_DENSE_ANCHOR50_300K60: "hcwmhped",
    }[profile]
    commands = []
    for task in spec["tasks"]:
        resource = spec["resources"][task["resource_class"]]
        command = [
            "sbatch", "--parsable", f"--account={ACCOUNT}", f"--partition={PARTITION}",
            f"--cpus-per-task={resource['cpus']}", f"--mem={resource['memory']}",
            f"--time={resource['walltime']}", f"--job-name={job_prefix}_{task['task_id']}",
        ]
        if resource.get("gpu"):
            command.extend((f"--gres={resource['gpu']}", "--signal=B:USR1@120"))
        if task["dependencies"]:
            command.append("--dependency=afterok:" + ":".join(f"${{JOB_{item}}}" for item in task["dependencies"]))
        command.extend((
            "--export=ALL," + f"PROJECT_DIR={spec['project_dir']},HCWDL_MHPE_SPEC={spec['spec_path']},HCWDL_MHPE_TASK={task['task_id']}",
            str(Path(spec["project_dir"]) / "sbatch/run_hcwdl_mhpe_task.sh"),
        ))
        commands.append({"task_id": task["task_id"], "dependencies": task["dependencies"], "command": command})
    return with_content_hash({
        "contract": COMMAND_PLAN_CONTRACT, "schema_version": 1,
        "spec_sha256": spec["content_hash"], "commands": commands,
        "mutated": False, "final_test_accessed": False,
    })


def _reuse_full(*, foundation_lock: Path, project: Path, source_commit: str) -> dict[str, Any]:
    lock = load_json(foundation_lock); lock_hash = validate_foundation_lock(lock)
    foundation_root = foundation_lock.parent.parent
    foundation = load_json(foundation_root / "foundation_spec.json")
    foundation_hash = validate_foundation_campaign(
        foundation, executable=False, verify_source_tree=False,
    )
    if lock.get("foundation_spec_sha256") != foundation_hash:
        raise ValueError("HCWDL-MHPE foundation lock/spec differs")
    foundation_recipe = load_json(foundation_root / "recipe.json")
    foundation_recipe_hash = validate_foundation_recipe(
        foundation_recipe, require_authorized=True, expected_profile="full_data_scaleup",
    )
    target = load_json(foundation_root / "targets/u000_train/manifest.json")
    target_hash = validate_target_manifest(target, teacher_id="shared/U000")
    u000 = load_json(foundation_root / "training/U000/training_report.json")
    m0 = load_json(foundation_root / "training/M0paired/training_report.json")
    u000_hash = validate_pmard_training_report(u000); m0_hash = validate_pmard_training_report(m0)
    for report, directory in ((u000, "U000"), (m0, "M0paired")):
        checkpoint = foundation_root / "training" / directory / str(report["selected_checkpoint"])
        if not checkpoint.is_file() or sha256_file(checkpoint) != report["selected_checkpoint_sha256"]:
            raise ValueError("HCWDL-MHPE imported checkpoint differs")
    if (
        lock.get("recipe_sha256") != foundation_recipe_hash
        or lock.get("u000_target_manifest_sha256") != target_hash
        or lock.get("u000_report_sha256") != u000_hash
        or lock.get("u000_checkpoint_sha256") != u000["selected_checkpoint_sha256"]
        or lock.get("m0paired_report_sha256") != m0_hash
        or lock.get("m0paired_checkpoint_sha256") != m0["selected_checkpoint_sha256"]
        or foundation.get("final_test_accessed") is not False
    ):
        raise ValueError("HCWDL-MHPE imported foundation lineage differs")
    current_semantics = {name: sha256_file(project / name) for name in SEMANTIC_SOURCE_FILES}
    foundation_semantics = foundation.get("semantic_source_sha256", {})
    missing_core = [name for name in REUSED_FOUNDATION_EXACT_FILES if name not in foundation_semantics]
    if missing_core:
        raise ValueError(f"HCWDL-MHPE foundation core hashes are absent: {missing_core}")
    byte_exact = {}
    additive = {}
    for name in REUSED_FOUNDATION_EXACT_FILES:
        record = {
            "foundation_sha256": foundation_semantics[name],
            "current_sha256": sha256_file(project / name),
        }
        if name in ADDITIVE_ADAPTER_FILES:
            additive[name] = record
        else:
            if record["foundation_sha256"] != record["current_sha256"]:
                raise ValueError(f"HCWDL-MHPE reused foundation core changed: {name}")
            byte_exact[name] = record
    compatibility = {
        "policy": "byte_exact_except_probability_target_adapter_v1",
        "byte_exact_files": byte_exact,
        "additive_adapter_files": additive,
        "legacy_logit_path_numerically_regressed": True,
        "adapter_scope": "new optional probability-target arguments only; legacy defaults and report bytes unchanged",
    }
    parents = {
        "foundation_lock_sha256": lock_hash,
        "foundation_spec_sha256": foundation_hash,
        "foundation_recipe_sha256": foundation_recipe_hash,
        "u000_report_sha256": u000_hash,
        "u000_checkpoint_sha256": u000["selected_checkpoint_sha256"],
        "u000_target_manifest_sha256": target_hash,
        "m0paired_report_sha256": m0_hash,
        "m0paired_checkpoint_sha256": m0["selected_checkpoint_sha256"],
    }
    parents.update({str(k): str(v) for k, v in lock.get("parents", {}).items()})
    return reuse_lock_payload(
        foundation_spec_path=foundation_root / "foundation_spec.json",
        foundation_spec_sha256=foundation_hash, foundation_lock_sha256=lock_hash,
        role_counts=foundation["role_counts"], u000_report_sha256=u000_hash,
        u000_checkpoint_sha256=u000["selected_checkpoint_sha256"],
        u000_target_manifest_sha256=target_hash, m0paired_report_sha256=m0_hash,
        source_commit=source_commit, semantic_source_sha256=current_semantics,
        foundation_parents=parents, foundation_core_compatibility=compatibility,
    )


def _reuse_300k(
    *, foundation_lock: Path, project: Path, source_commit: str, profile: str,
) -> dict[str, Any]:
    """Authenticate immutable completed 300k UB products for the paired study."""
    lock = load_json(foundation_lock)
    lock_hash = validate_300k_foundation_lock(lock)
    foundation_root = foundation_lock.parent.parent
    foundation = load_json(foundation_root / "foundation_spec.json")
    foundation_hash = validate_300k_foundation_spec(foundation)
    if (lock.get("foundation_spec_sha256") != foundation_hash
            or foundation.get("role_counts") != {
                "train": 300_000, "validation": 100_000,
                "final_test": 100_000,
            }
            or foundation.get("ordinary_access_role_counts", {}).get("final_test") != 0
            or foundation.get("final_test_accessed") is not False):
        raise ValueError("HCWDL-MHPE 300k foundation population/lock differs")
    foundation_recipe = load_json(foundation_root / "recipe.json")
    foundation_recipe_hash = validate_content_hash(
        foundation_recipe, expected_contract=str(foundation_recipe["contract"]),
        expected_schema_version=1,
    )
    if int(foundation_recipe.get("training_passes", -1)) != 60:
        raise ValueError("HCWDL-MHPE 300k foundation is not the 60-pass recipe")
    target = load_json(foundation_root / "targets/u000_train/manifest.json")
    target_hash = validate_target_manifest(target, teacher_id="shared/U000")
    target_evidence = inspect_shared_u000_target_lineage(
        foundation_spec=foundation, foundation_root=foundation_root,
    )
    if target_evidence["actual_target_manifest_sha256"] != target_hash:
        raise ValueError("HCWDL-MHPE 300k target evidence differs")
    u000 = load_json(foundation_root / "training/U000/training_report.json")
    m0 = load_json(foundation_root / "training/M0paired/training_report.json")
    u000_hash = validate_pmard_training_report(u000)
    m0_hash = validate_pmard_training_report(m0)
    for report, directory in ((u000, "U000"), (m0, "M0paired")):
        checkpoint = foundation_root / "training" / directory / str(report["selected_checkpoint"])
        if (not checkpoint.is_file()
                or sha256_file(checkpoint) != report["selected_checkpoint_sha256"]):
            raise ValueError("HCWDL-MHPE 300k imported checkpoint differs")
    if (
        lock.get("u000_report_sha256") != u000_hash
        or lock.get("u000_checkpoint_sha256") != u000["selected_checkpoint_sha256"]
        or lock.get("m0paired_report_sha256") != m0_hash
        or lock.get("m0paired_checkpoint_sha256") != m0["selected_checkpoint_sha256"]
    ):
        raise ValueError("HCWDL-MHPE 300k imported foundation lineage differs")
    current_semantics = {name: sha256_file(project / name) for name in SEMANTIC_SOURCE_FILES}
    foundation_semantics = foundation.get("semantic_source_sha256", {})
    missing = [name for name in UB_300K_SEMANTIC_SOURCE_FILES
               if name not in foundation_semantics]
    if missing:
        raise ValueError(f"HCWDL-MHPE 300k foundation source hashes are absent: {missing}")
    compatibility = {
        "policy": "authenticated_immutable_300k_products_additive_mhpe_v2",
        "byte_exact_files": {},
        "additive_adapter_files": {
            name: {
                "foundation_sha256": foundation_semantics[name],
                "current_sha256": sha256_file(project / name),
            }
            for name in sorted(ADDITIVE_ADAPTER_FILES)
        },
        "authenticated_foundation_source_sha256": {
            name: foundation_semantics[name]
            for name in sorted(UB_300K_SEMANTIC_SOURCE_FILES)
        },
        "u000_target_lineage_evidence": dict(target_evidence),
        "legacy_logit_path_numerically_regressed": True,
        "foundation_products_immutable": True,
        "adapter_scope": (
            "completed U000/M0/checkpoint/target products are hash-authenticated; "
            "new MHPE execution is source-pinned and additive"
        ),
    }
    parents = {
        "foundation_lock_sha256": lock_hash,
        "foundation_spec_sha256": foundation_hash,
        "foundation_recipe_sha256": foundation_recipe_hash,
        "u000_report_sha256": u000_hash,
        "u000_checkpoint_sha256": u000["selected_checkpoint_sha256"],
        "u000_target_manifest_sha256": target_hash,
        "m0paired_report_sha256": m0_hash,
        "m0paired_checkpoint_sha256": m0["selected_checkpoint_sha256"],
        "u000_target_lineage_evidence_sha256": target_evidence["content_hash"],
    }
    parents.update({str(k): str(v) for k, v in lock.get("parents", {}).items()})
    return reuse_lock_payload(
        foundation_spec_path=foundation_root / "foundation_spec.json",
        foundation_spec_sha256=foundation_hash,
        foundation_lock_sha256=lock_hash,
        role_counts=foundation["role_counts"],
        u000_report_sha256=u000_hash,
        u000_checkpoint_sha256=u000["selected_checkpoint_sha256"],
        u000_target_manifest_sha256=target_hash,
        m0paired_report_sha256=m0_hash,
        source_commit=source_commit,
        semantic_source_sha256=current_semantics,
        foundation_parents=parents,
        foundation_core_compatibility=compatibility,
        profile=profile,
    )


def _reuse(
    *, foundation_lock: Path, project: Path, source_commit: str,
    profile: str = PROFILE_C25P75,
) -> dict[str, Any]:
    if profile in {PROFILE_C25P75_300K60, PROFILE_C10P90_300K60,
                   PROFILE_DENSE_ANCHOR50_300K60}:
        return _reuse_300k(
            foundation_lock=foundation_lock, project=project,
            source_commit=source_commit, profile=profile,
        )
    return _reuse_full(
        foundation_lock=foundation_lock, project=project,
        source_commit=source_commit,
    )


def create_campaign(
    *, foundation_lock: str | Path, campaign_root: str | Path,
    project_dir: str | Path, source_commit: str,
    authorize_live_submission: bool = False, authorization_phrase: str | None = None,
    recipe_profile: str = PROFILE_C25P75,
    publish: bool = True,
) -> dict[str, Any]:
    if recipe_profile not in SUPPORTED_PROFILES:
        raise ValueError("unknown HCWDL-MHPE recipe profile")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("HCWDL-MHPE requires a full lowercase source commit")
    if authorize_live_submission and authorization_phrase != creation_phrase(recipe_profile):
        raise PermissionError("HCWDL-MHPE creation phrase differs")
    root = Path(campaign_root).resolve(); project = Path(project_dir).resolve()
    if publish and root.exists() and any(root.iterdir()):
        raise FileExistsError("HCWDL-MHPE campaign root is not empty")
    reuse = _reuse(
        foundation_lock=Path(foundation_lock).resolve(), project=project,
        source_commit=source_commit, profile=recipe_profile,
    )
    foundation_root = Path(reuse["foundation_spec_path"]).parent
    foundation_recipe = load_json(foundation_root / "recipe.json")
    graph = graph_payload(recipe_profile)
    recipe = recipe_payload(
        foundation_recipe_sha256=foundation_recipe["content_hash"],
        profile=recipe_profile,
    )
    semantic_hashes = semantic_source_hashes(project)
    resources = {
        name: asdict(value)
        for name, value in resources_for_profile(recipe_profile).items()
    }
    evidence_hashes = {
        name: sha256_file(project / name) for name in evidence_files(recipe_profile)
    }
    waiver = waiver_payload(
        source_commit=source_commit, graph_sha256=graph["content_hash"],
        reuse_lock_sha256=reuse["content_hash"], recipe_sha256=recipe["content_hash"],
        semantic_source_registry_sha256=canonical_sha256(semantic_hashes),
        resource_request_sha256=canonical_sha256(resources),
        implementation_evidence_sha256=evidence_hashes,
        authorization_phrase={
            PROFILE_C25P75: WAIVER_PHRASE,
            PROFILE_C10P90: WAIVER_PHRASE_C10P90,
            PROFILE_C25P75_300K60: WAIVER_PHRASE_C25P75_300K60,
            PROFILE_C10P90_300K60: WAIVER_PHRASE_C10P90_300K60,
            PROFILE_DENSE_ANCHOR50_300K60: WAIVER_PHRASE_DENSE_ANCHOR50_300K60,
        }[recipe_profile],
        profile=recipe_profile,
    )
    unhashed = {
        "contract": campaign_spec_contract(recipe_profile), "schema_version": 1,
        "campaign": {
            PROFILE_C25P75: "HCWDL-MHPE-FULL",
            PROFILE_C10P90: "HCWDL-MHPE-C10P90-FULL",
            PROFILE_C25P75_300K60: "HCWDL-MHPE-C25P75-300K60",
            PROFILE_C10P90_300K60: "HCWDL-MHPE-C10P90-300K60",
            PROFILE_DENSE_ANCHOR50_300K60: "HCWDL-MHPE-DENSE-ANCHOR50-300K60",
        }[recipe_profile],
        "campaign_root": str(root),
        "project_dir": str(project), "source_commit": source_commit,
        "spec_path": str(root / "campaign_spec.json"),
        "reuse_lock_path": str(root / "foundation_reuse_lock.json"),
        "reuse_lock_sha256": reuse["content_hash"], "graph_sha256": graph["content_hash"],
        "recipe_sha256": recipe["content_hash"], "waiver_sha256": waiver["content_hash"],
        "role_counts": reuse["role_counts"], "tasks": campaign_tasks(recipe_profile),
        # Contextual predecessors are not causal parents of v1.  The empty
        # registry is explicit and immutable; later comparison imports need a
        # new campaign identity rather than path discovery after launch.
        "contextual_reports": [],
        "resources": resources,
        "semantic_source_sha256": semantic_hashes,
        "live_submission_authorized": bool(authorize_live_submission),
        "authorization_phrase": authorization_phrase if authorize_live_submission else None,
        "final_test_accessed": False,
    }
    if recipe_profile != PROFILE_C25P75:
        unhashed["recipe_profile"] = recipe_profile
    if recipe_profile in {PROFILE_C25P75_300K60, PROFILE_C10P90_300K60}:
        unhashed["population_profile"] = "pilot_300k_60pass"
        unhashed["paired_study"] = "specialist_ce_kd_weights_only"
    elif recipe_profile == PROFILE_DENSE_ANCHOR50_300K60:
        unhashed["population_profile"] = "pilot_300k_60pass"
        unhashed["study"] = "dense_factorized_anchor50_multi_horizon"
    elif recipe_profile == PROFILE_C10P90:
        unhashed["single_changed_variable"] = "specialist_ce_kd_weights_only"
    spec = with_content_hash(unhashed); plan = command_plan(spec)
    if publish:
        root.mkdir(parents=True, exist_ok=False)
        for name, value in (
            ("foundation_reuse_lock.json", reuse), ("graph.json", graph),
            ("recipe.json", recipe), ("operational_evidence_waiver.json", waiver),
            ("campaign_spec.json", spec), ("command_plan.json", plan),
        ):
            write_immutable_json(root / name, value)
    return spec


def validate_campaign(value: Mapping[str, Any], *, executable: bool = False, verify_source_tree: bool = True) -> str:
    profile = campaign_profile(value)
    digest = validate_content_hash(
        value, expected_contract=campaign_spec_contract(profile), expected_schema_version=1,
    )
    if (value.get("tasks") != campaign_tasks(profile)
            or value.get("resources") != {
                name: asdict(row)
                for name, row in resources_for_profile(profile).items()
            }
            or value.get("contextual_reports") != []):
        raise ValueError("HCWDL-MHPE task/resources differ")
    if profile in {PROFILE_C25P75_300K60, PROFILE_C10P90_300K60}:
        if (value.get("campaign") != {
                PROFILE_C25P75_300K60: "HCWDL-MHPE-C25P75-300K60",
                PROFILE_C10P90_300K60: "HCWDL-MHPE-C10P90-300K60",
            }[profile]
                or value.get("population_profile") != "pilot_300k_60pass"
                or value.get("paired_study") != "specialist_ce_kd_weights_only"):
            raise ValueError("HCWDL-MHPE 300k60 campaign identity differs")
    elif profile == PROFILE_DENSE_ANCHOR50_300K60:
        if (value.get("campaign") != "HCWDL-MHPE-DENSE-ANCHOR50-300K60"
                or value.get("population_profile") != "pilot_300k_60pass"
                or value.get("study") != "dense_factorized_anchor50_multi_horizon"):
            raise ValueError("HCWDL-MHPE dense campaign identity differs")
    elif profile == PROFILE_C10P90:
        if (value.get("campaign") != "HCWDL-MHPE-C10P90-FULL"
                or value.get("single_changed_variable")
                != "specialist_ce_kd_weights_only"):
            raise ValueError("HCWDL-MHPE C10P90 campaign identity differs")
    elif (value.get("campaign") != "HCWDL-MHPE-FULL"
          or "recipe_profile" in value
          or "single_changed_variable" in value):
        raise ValueError("HCWDL-MHPE primary campaign identity differs")
    root = Path(value["campaign_root"])
    reuse = load_json(value["reuse_lock_path"])
    if validate_reuse_lock(reuse) != value["reuse_lock_sha256"]:
        raise ValueError("HCWDL-MHPE reuse lock differs")
    is_300k60 = profile in {PROFILE_C25P75_300K60, PROFILE_C10P90_300K60,
                            PROFILE_DENSE_ANCHOR50_300K60}
    if (is_300k60 != (reuse.get("population_profile") == "pilot_300k_60pass")
            or (is_300k60 and reuse.get("recipe_profile") != profile)):
        raise ValueError("HCWDL-MHPE campaign/reuse population differs")
    if reuse["semantic_source_sha256"] != value["semantic_source_sha256"] or reuse["source_commit"] != value["source_commit"]:
        raise ValueError("HCWDL-MHPE reuse/source lineage differs")
    if validate_graph(load_json(root / "graph.json")) != value["graph_sha256"]:
        raise ValueError("HCWDL-MHPE graph differs")
    recipe = load_json(root / "recipe.json")
    if validate_recipe(recipe) != value["recipe_sha256"]:
        raise ValueError("HCWDL-MHPE recipe differs")
    if (recipe["foundation_recipe_sha256"] != reuse["foundation_parents"]["foundation_recipe_sha256"]
            or (profile != PROFILE_C25P75
                and recipe.get("recipe_profile") != profile)):
        raise ValueError("HCWDL-MHPE recipe/foundation lineage differs")
    waiver = load_json(root / "operational_evidence_waiver.json")
    if (validate_waiver(waiver)
            != value["waiver_sha256"] or waiver.get("source_commit") != value["source_commit"]
            or waiver.get("graph_sha256") != value["graph_sha256"]
            or waiver.get("reuse_lock_sha256") != value["reuse_lock_sha256"]
            or waiver.get("recipe_sha256") != value["recipe_sha256"]
            or waiver.get("semantic_source_registry_sha256")
            != canonical_sha256(value["semantic_source_sha256"])
            or waiver.get("resource_request_sha256")
            != canonical_sha256(value["resources"])
            or waiver.get("implementation_evidence_sha256") != {
                name: sha256_file(Path(value["project_dir"]) / name)
                for name in evidence_files(profile)
            }
            or waiver.get("does_not_claim_new_smoke_evidence") is not True):
        raise ValueError("HCWDL-MHPE operational waiver differs")
    plan = load_json(root / "command_plan.json")
    validate_content_hash(plan, expected_contract=COMMAND_PLAN_CONTRACT, expected_schema_version=1)
    if plan != command_plan(value):
        raise ValueError("HCWDL-MHPE command plan drifted")
    if verify_source_tree and value["semantic_source_sha256"] != semantic_source_hashes(value["project_dir"]):
        raise ValueError("HCWDL-MHPE source tree drifted")
    if executable and (
        value.get("live_submission_authorized") is not True
        or value.get("authorization_phrase") != creation_phrase(profile)
    ):
        raise PermissionError("HCWDL-MHPE campaign is not live authorized")
    return digest


__all__ = [
    "ADDITIVE_ADAPTER_FILES", "CREATION_PHRASE", "IMPLEMENTATION_EVIDENCE_FILES",
    "CREATION_PHRASE_C10P90", "C10P90_IMPLEMENTATION_EVIDENCE_FILES",
    "CREATION_PHRASE_DENSE_ANCHOR50_300K60",
    "DENSE_IMPLEMENTATION_EVIDENCE_FILES",
    "REUSED_FOUNDATION_EXACT_FILES", "RESOURCES", "P300_RESOURCES",
    "SEMANTIC_SOURCE_FILES", "SUBMISSION_PHRASE", "SUBMISSION_PHRASE_C10P90",
    "SUBMISSION_PHRASE_DENSE_ANCHOR50_300K60",
    "WAIVER_PHRASE_C10P90", "creation_phrase", "evidence_files",
    "WAIVER_PHRASE", "campaign_tasks", "command_plan", "create_campaign",
    "semantic_source_hashes", "submission_phrase", "validate_campaign",
    "resources_for_profile",
]

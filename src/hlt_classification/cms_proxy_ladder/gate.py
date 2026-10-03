"""Source-pinned release, matcher, and measured-runtime gates."""
from __future__ import annotations

from pathlib import Path
import math
import re
import subprocess
import time

import torch

from hlt_classification.data.cache_contracts import load_json, sha256_file
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout
from hlt_classification.jetclass2_delphes.campaign import recipe
from hlt_classification.jetclass2_delphes.execution import (
    DEBUG_PROFILE_TRANSFER, allocation, execution_site, gpu_identity,
    production_site, validate_resources,
)
from hlt_classification.jetclass2_delphes.model import (
    DelphesParticleTransformer, installed_environment, model_contract,
)
from hlt_classification.jetclass2_delphes.runner import predict, train_kernel

from .cache import cache_budgets, preparation_bound, prepare_cache
from .campaign import build_direct_coarse_plan, build_scientific_plan
from .contracts import artifact, file_ref, validate, write_json
from .data import build_foundation, validate_foundation
from .release import build_release, release_request, validate_release
from .population import (
    create_direct_coarse_population, validate_direct_coarse_population,
)

SOURCE_FILES = (
    "docs/plans/JETCLASS2_CMS_PROXY_TIGRIS_200K_THREE_SPINE_IMPLEMENTATION_PLAN.md",
    "docs/plans/JETCLASS2_CMS_PROXY_SPORC_DEBUG_GATE_AMENDMENT.md",
    "docs/plans/JETCLASS2_CMS_PROXY_OSCAR_PORTABILITY_PLAN.md",
    "docs/plans/JETCLASS2_CMS_PROXY_OSCAR_DUAL_SLOT_RECOVERY_PLAN.md",
    "docs/plans/JETCLASS2_CMS_PROXY_OSCAR_100K_DIRECT_COARSE_PLAN.md",
    "docs/contracts/JETCLASS2_CMS_PROXY_LADDER.md",
    "docs/contracts/JETCLASS2_CMS_PROXY_OSCAR_PORTABILITY.md",
    "docs/JETCLASS2_CMS_PROXY_TIGRIS_200K_HANDOFF.md",
    "scripts/jetclass2_cms_proxy_ladder.py",
    "scripts/run_jetclass2_cms_proxy_ladder_task.py",
    "scripts/submit_jetclass2_cms_proxy_ladder.py",
    "sbatch/jetclass2_delphes_common.sh",
    "sbatch/run_jetclass2_cms_proxy_ladder.sh",
    "src/hlt_classification/jetclass2_delphes/execution.py",
    "sbatch/common.sh",
    "tests/test_cms_proxy_ladder.py",
)
AUTHORIZATION = "AUTHORIZE JETCLASS2 CMS PROXY 200K TIGRIS GATE"
DEBUG_AUTHORIZATION = "AUTHORIZE JETCLASS2 CMS PROXY 200K SPORC DEBUG GATE"
PREFLIGHT_RECOVERY_AUTHORIZATION = (
    "AUTHORIZE JETCLASS2 CMS PROXY 200K SPORC PREFLIGHT RECOVERY"
)
OSCAR_AUTHORIZATION = "AUTHORIZE JETCLASS2 CMS PROXY 200K OSCAR PREFLIGHT"
OSCAR_DUAL_SLOT_AUTHORIZATION = (
    "AUTHORIZE JETCLASS2 CMS PROXY 200K OSCAR DUAL SLOT PREFLIGHT"
)
OSCAR_DIRECT_COARSE_AUTHORIZATION = (
    "AUTHORIZE JETCLASS2 CMS PROXY 100K OSCAR DIRECT COARSE PREFLIGHT"
)


def _source(project: Path, commit: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Exact 40-character pushed commit required")
    validate_source_checkout(Path(project), expected_commit=commit)


def source_lock(project: Path, commit: str, *, literature: bool = False) -> dict:
    _source(project, commit)
    root = Path(project).resolve()
    package = sorted((root / "src/hlt_classification/cms_proxy_ladder").glob("*.py"))
    files = [root / name for name in SOURCE_FILES] + package
    if literature:
        from .literature import SOURCE_FILES as extra
        files += [root / name for name in extra]
        for folder in ("literature_proxy_production", "literature_proxy", "literature_proxy_v2",
                       "literature_proxy_v3", "jetclass2_delphes", "cms2jc2_production", "cms2jc2_response"):
            files += sorted((root / "src/hlt_classification" / folder).glob("*.py"))
    result = {}
    for path in files:
        relative = path.relative_to(root).as_posix()
        subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", relative],
            check=True, capture_output=True,
        )
        result[relative] = sha256_file(path)
    return artifact("SOURCE", commit=commit, files=result)


def gate_tasks() -> list[dict]:
    return [
        {
            "task_id": "authenticate_release", "kind": "cpu", "dependencies": [],
            "cpus": 16, "memory_mb": 128_000, "minutes": 480,
        },
        {
            "task_id": "build_foundation", "kind": "cpu",
            "dependencies": ["authenticate_release"],
            "cpus": 72, "memory_mb": 384_000, "minutes": 1440,
        },
        {
            "task_id": "preflight", "kind": "gpu",
            "dependencies": ["build_foundation"],
            "cpus": 32, "memory_mb": 384_000, "minutes": 720,
        },
    ]


def debug_gate_tasks() -> list[dict]:
    """Bounded SPORC-debug recovery; debug measures, tier3 runs science."""
    return [
        {
            "task_id": "authenticate_release", "kind": "cpu", "dependencies": [],
            "cpus": 4, "memory_mb": 32_000, "minutes": 60,
        },
        {
            "task_id": "build_foundation", "kind": "cpu",
            "dependencies": ["authenticate_release"],
            "cpus": 36, "memory_mb": 320_000, "minutes": 480,
        },
        {
            "task_id": "preflight", "kind": "gpu",
            "dependencies": ["build_foundation"],
            "cpus": 36, "memory_mb": 320_000, "minutes": 480,
        },
    ]


def preflight_recovery_tasks() -> list[dict]:
    return [{
        "task_id": "preflight", "kind": "gpu", "dependencies": [],
        "cpus": 16, "memory_mb": 160_000, "minutes": 480,
    }]


def oscar_preflight_tasks() -> list[dict]:
    return [{
        "task_id": "preflight", "kind": "gpu", "dependencies": [],
        "cpus": 12, "memory_mb": 160_000, "minutes": 720,
    }]


def oscar_dual_slot_preflight_tasks() -> list[dict]:
    """Oscar profile sized for two concurrent jobs under the user QOS."""
    return [{
        "task_id": "preflight", "kind": "gpu", "dependencies": [],
        "cpus": 6, "memory_mb": 90_000, "minutes": 720,
    }]


def oscar_direct_coarse_preflight_tasks() -> list[dict]:
    """Measure the selected 100k/50k population at the dual-slot shape."""
    return oscar_dual_slot_preflight_tasks()


def create_gate(
    *, study_root: Path, offline_root: Path, gate_root: Path,
    project_dir: Path, source_commit: str, capacity: int = 512,
) -> dict:
    root = Path(gate_root).resolve()
    if root.exists():
        raise FileExistsError("Proxy-ladder gate root must be fresh")
    project = Path(project_dir).resolve(strict=True)
    source = source_lock(project, source_commit)
    request = release_request(study_root=study_root, offline_root=offline_root)
    site = execution_site("tigris_gh200")
    spec = artifact(
        "GATE_SPEC",
        parents={"source": source["content_hash"], "request": request["content_hash"]},
        source=source, request=request, gate_root=str(root),
        project_dir=str(project), source_commit=source_commit,
        capacity=capacity, execution_site=site, tasks=gate_tasks(),
        workers=16, full_views_persisted=False,
        admission="release_then_new_assignments_then_real_tigris_full_population_preflight",
    )
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "gate_spec.json", spec)
    validate_gate(spec, check_source=True)
    return spec


def create_sporc_debug_gate(
    *, source_gate_root: Path, gate_root: Path, project_dir: Path,
    source_commit: str,
) -> dict:
    """Recover after the Tigris release freeze using SPORC debug evidence."""
    root = Path(gate_root).resolve()
    if root.exists():
        raise FileExistsError("Proxy-ladder SPORC debug gate root must be fresh")
    project = Path(project_dir).resolve(strict=True)
    source = source_lock(project, source_commit)
    source_root = Path(source_gate_root).resolve(strict=True)
    subject = load_json(source_root / "gate_spec.json")
    validate_gate(subject)
    release_root = source_root / "release"
    release = load_json(release_root / "release.json")
    validate_release(release, root=release_root)
    if release["request"] != subject["request"]:
        raise ValueError("Imported proxy-ladder release/request lineage differs")
    measurement = execution_site("sporc_a100_debug")
    spec = artifact(
        "GATE_SPEC", version=2,
        parents={
            "source": source["content_hash"],
            "request": release["request"]["content_hash"],
            "imported_release": release["content_hash"],
            "subject_gate": subject["content_hash"],
        },
        source=source, request=release["request"], gate_root=str(root),
        project_dir=str(project), source_commit=source_commit, capacity=subject["capacity"],
        measurement_site=measurement, execution_site=production_site(measurement),
        tasks=debug_gate_tasks(), workers=16, foundation_workers=36,
        full_views_persisted=False, source_gate_root=str(source_root),
        imported_release_root=str(release_root), imported_release=release,
        admission="reuse_authenticated_release_then_sporc_debug_foundation_and_full_population_preflight",
        site_transfer_policy=DEBUG_PROFILE_TRANSFER,
    )
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "gate_spec.json", spec)
    validate_gate(spec, check_source=True)
    return spec


def create_sporc_preflight_recovery(
    *, source_gate_root: Path, gate_root: Path, project_dir: Path,
    source_commit: str,
) -> dict:
    """Reuse a completed v2 foundation and measure the right-sized profile."""
    root = Path(gate_root).resolve()
    if root.exists():
        raise FileExistsError("Proxy-ladder preflight-recovery root must be fresh")
    project = Path(project_dir).resolve(strict=True)
    source = source_lock(project, source_commit)
    source_root = Path(source_gate_root).resolve(strict=True)
    subject = load_json(source_root / "gate_spec.json")
    validate_gate(subject)
    if subject.get("schema_version") != 2:
        raise ValueError("Preflight recovery requires the completed SPORC v2 foundation gate")
    foundation_root = source_root / "foundation"
    foundation = load_json(foundation_root / "foundation.json")
    validate_foundation(foundation, root=foundation_root)
    workers, memory_mb = 16, 160_000
    budgets = cache_budgets(foundation, memory_mb, workers)
    bounds = {
        role: preparation_bound(foundation, role, workers)
        for role in ("train", "validation")
    }
    measurement = execution_site("sporc_a100_debug")
    spec = artifact(
        "GATE_SPEC", version=3,
        parents={
            "source": source["content_hash"],
            "request": subject["request"]["content_hash"],
            "imported_release": subject["imported_release"]["content_hash"],
            "imported_foundation": foundation["content_hash"],
            "subject_gate": subject["content_hash"],
        },
        source=source, request=subject["request"], gate_root=str(root),
        project_dir=str(project), source_commit=source_commit,
        capacity=subject["capacity"], measurement_site=measurement,
        execution_site=production_site(measurement), tasks=preflight_recovery_tasks(),
        workers=workers, full_views_persisted=False,
        source_gate_root=str(source_root), foundation_root=str(foundation_root),
        imported_foundation=foundation,
        imported_release_root=subject["imported_release_root"],
        imported_release=subject["imported_release"],
        cache_preparation_bounds=bounds, cache_budgets=budgets,
        admission="reuse_authenticated_release_and_completed_foundation_then_right_sized_sporc_debug_preflight",
        site_transfer_policy=DEBUG_PROFILE_TRANSFER,
    )
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "gate_spec.json", spec)
    validate_gate(spec, check_source=True)
    return spec


def create_oscar_gate(
    *, materialization_root: Path, gate_root: Path, project_dir: Path,
    source_commit: str,
) -> dict:
    """Measure an Oscar-native profile from relocated authenticated inputs."""
    from .portable import validate_materialization

    root = Path(gate_root).resolve()
    if root.exists():
        raise FileExistsError("Proxy-ladder Oscar gate root must be fresh")
    project = Path(project_dir).resolve(strict=True)
    source = source_lock(project, source_commit)
    materialized_root = Path(materialization_root).resolve(strict=True)
    materialization = load_json(materialized_root / "materialization.json")
    validate_materialization(materialization, root=materialized_root)
    release = materialization["release"]
    foundation = materialization["foundation"]
    workers, memory_mb = 12, oscar_preflight_tasks()[0]["memory_mb"]
    bounds = {
        role: preparation_bound(foundation, role, workers)
        for role in ("train", "validation")
    }
    budgets = cache_budgets(foundation, memory_mb, workers)
    site = execution_site("oscar_l40s")
    spec = artifact(
        "GATE_SPEC", version=4,
        parents={
            "source": source["content_hash"],
            "request": release["request"]["content_hash"],
            "imported_release": release["content_hash"],
            "imported_foundation": foundation["content_hash"],
            "portable_materialization": materialization["content_hash"],
            "portable_bundle": materialization["bundle_sha256"],
        },
        source=source, request=release["request"], gate_root=str(root),
        project_dir=str(project), source_commit=source_commit,
        capacity=foundation["inputs"]["capacity"],
        measurement_site=site, execution_site=site,
        tasks=oscar_preflight_tasks(), workers=workers,
        full_views_persisted=False,
        materialization_root=str(materialized_root),
        portable_materialization=materialization,
        foundation_root=materialization["foundation_root"],
        imported_foundation=foundation,
        imported_release_root=materialization["release_root"],
        imported_release=release,
        cache_preparation_bounds=bounds, cache_budgets=budgets,
        admission="relocated_exact_inputs_then_oscar_l40s_full_population_preflight",
        site_transfer_policy=None,
    )
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "gate_spec.json", spec)
    validate_gate(spec, check_source=True)
    return spec


def create_oscar_dual_slot_gate(
    *, materialization_root: Path, gate_root: Path, project_dir: Path,
    source_commit: str,
) -> dict:
    """Measure the same Oscar science with a two-concurrent-job QOS profile."""
    from .portable import validate_materialization

    root = Path(gate_root).resolve()
    if root.exists():
        raise FileExistsError("Proxy-ladder Oscar dual-slot gate root must be fresh")
    project = Path(project_dir).resolve(strict=True)
    source = source_lock(project, source_commit)
    materialized_root = Path(materialization_root).resolve(strict=True)
    materialization = load_json(materialized_root / "materialization.json")
    validate_materialization(materialization, root=materialized_root)
    release = materialization["release"]
    foundation = materialization["foundation"]
    task = oscar_dual_slot_preflight_tasks()[0]
    workers, memory_mb = task["cpus"], task["memory_mb"]
    bounds = {
        role: preparation_bound(foundation, role, workers)
        for role in ("train", "validation")
    }
    budgets = cache_budgets(foundation, memory_mb, workers)
    site = execution_site("oscar_l40s")
    spec = artifact(
        "GATE_SPEC", version=5,
        parents={
            "source": source["content_hash"],
            "request": release["request"]["content_hash"],
            "imported_release": release["content_hash"],
            "imported_foundation": foundation["content_hash"],
            "portable_materialization": materialization["content_hash"],
            "portable_bundle": materialization["bundle_sha256"],
        },
        source=source, request=release["request"], gate_root=str(root),
        project_dir=str(project), source_commit=source_commit,
        capacity=foundation["inputs"]["capacity"],
        measurement_site=site, execution_site=site,
        tasks=oscar_dual_slot_preflight_tasks(), workers=workers,
        full_views_persisted=False,
        materialization_root=str(materialized_root),
        portable_materialization=materialization,
        foundation_root=materialization["foundation_root"],
        imported_foundation=foundation,
        imported_release_root=materialization["release_root"],
        imported_release=release,
        cache_preparation_bounds=bounds, cache_budgets=budgets,
        admission=(
            "relocated_exact_inputs_then_oscar_l40s_dual_slot_"
            "full_population_preflight"
        ),
        concurrency_intent={
            "jobs": 2,
            "per_job_cpus": 6,
            "per_job_memory_mb": 90_000,
            "per_job_gpus": 1,
            "aggregate_cpus": 12,
            "aggregate_memory_mb": 180_000,
            "aggregate_gpus": 2,
        },
        site_transfer_policy=None,
    )
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "gate_spec.json", spec)
    validate_gate(spec, check_source=True)
    return spec


def create_oscar_direct_coarse_gate(
    *, materialization_root: Path, gate_root: Path, project_dir: Path,
    source_commit: str,
) -> dict:
    """Measure the nested 100k/50k DIRECT+COARSE Oscar execution."""
    from .portable import validate_materialization

    root = Path(gate_root).resolve()
    if root.exists():
        raise FileExistsError("Proxy-ladder Oscar direct/coarse gate root must be fresh")
    project = Path(project_dir).resolve(strict=True)
    source = source_lock(project, source_commit)
    materialized_root = Path(materialization_root).resolve(strict=True)
    materialization = load_json(materialized_root / "materialization.json")
    validate_materialization(materialization, root=materialized_root)
    release = materialization["release"]
    foundation = materialization["foundation"]
    foundation_root = Path(materialization["foundation_root"])
    population = create_direct_coarse_population(
        foundation, foundation_root=foundation_root,
    )
    task = oscar_direct_coarse_preflight_tasks()[0]
    workers, memory_mb = task["cpus"], task["memory_mb"]
    bounds = {
        role: preparation_bound(
            foundation, role, workers, foundation_root=foundation_root,
            population_selection=population,
        )
        for role in ("train", "validation")
    }
    budgets = cache_budgets(
        foundation, memory_mb, workers, foundation_root=foundation_root,
        population_selection=population,
    )
    site = execution_site("oscar_l40s")
    spec = artifact(
        "GATE_SPEC", version=6,
        parents={
            "source": source["content_hash"],
            "request": release["request"]["content_hash"],
            "imported_release": release["content_hash"],
            "imported_foundation": foundation["content_hash"],
            "portable_materialization": materialization["content_hash"],
            "portable_bundle": materialization["bundle_sha256"],
            "population_selection": population["content_hash"],
        },
        source=source, request=release["request"], gate_root=str(root),
        project_dir=str(project), source_commit=source_commit,
        capacity=foundation["inputs"]["capacity"],
        measurement_site=site, execution_site=site,
        tasks=oscar_direct_coarse_preflight_tasks(), workers=workers,
        full_views_persisted=False,
        materialization_root=str(materialized_root),
        portable_materialization=materialization,
        foundation_root=str(foundation_root), imported_foundation=foundation,
        imported_release_root=materialization["release_root"],
        imported_release=release, population_selection=population,
        scientific_branches=["DIRECT", "COARSE"],
        cache_preparation_bounds=bounds, cache_budgets=budgets,
        admission=(
            "relocated_exact_inputs_nested_100k_50k_then_oscar_l40s_"
            "direct_coarse_full_selected_population_preflight"
        ),
        concurrency_intent={
            "jobs": 2,
            "per_job_cpus": 6,
            "per_job_memory_mb": 90_000,
            "per_job_gpus": 1,
            "aggregate_cpus": 12,
            "aggregate_memory_mb": 180_000,
            "aggregate_gpus": 2,
        },
        site_transfer_policy=None,
    )
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "gate_spec.json", spec)
    validate_gate(spec, check_source=True)
    return spec


def validate_gate(spec: dict, *, check_source: bool = False) -> str:
    version = spec.get("schema_version")
    if version == 7:
        from .literature import validate_gate as validate_literature_gate
        return validate_literature_gate(spec, check_source=check_source)
    if version not in (1, 2, 3, 4, 5, 6):
        raise ValueError("Unsupported proxy-ladder gate version")
    parents = {"source": spec["source"]["content_hash"], "request": spec["request"]["content_hash"]}
    if version in (2, 3):
        parents.update({
            "imported_release": spec["imported_release"]["content_hash"],
            "subject_gate": spec["parents"]["subject_gate"],
        })
    if version == 3:
        parents["imported_foundation"] = spec["imported_foundation"]["content_hash"]
    if version in (4, 5, 6):
        parents.update({
            "imported_release": spec["imported_release"]["content_hash"],
            "imported_foundation": spec["imported_foundation"]["content_hash"],
            "portable_materialization": spec["portable_materialization"]["content_hash"],
            "portable_bundle": spec["portable_materialization"]["bundle_sha256"],
        })
    if version == 6:
        parents["population_selection"] = spec["population_selection"]["content_hash"]
    digest = validate(spec, "GATE_SPEC", version=version, parents=parents)
    validate(spec["source"], "SOURCE")
    from .release import validate_request
    validate_request(spec["request"])
    common_differs = (
        spec["workers"] != ({4: 12, 5: 6, 6: 6}.get(version, 16))
        or spec["full_views_persisted"] is not False
        or type(spec["capacity"]) is not int or spec["capacity"] < 16
    )
    if version == 1:
        differs = (
            spec["execution_site"] != execution_site("tigris_gh200")
            or spec["tasks"] != gate_tasks()
        )
    elif version == 2:
        release_root = Path(spec["imported_release_root"])
        subject = load_json(Path(spec["source_gate_root"]) / "gate_spec.json")
        if subject.get("schema_version") != 1:
            raise ValueError("SPORC recovery must import the original Tigris gate")
        validate_gate(subject)
        validate_release(spec["imported_release"], root=release_root)
        differs = (
            spec["measurement_site"] != execution_site("sporc_a100_debug")
            or spec["execution_site"] != execution_site("sporc_a100")
            or spec["tasks"] != debug_gate_tasks()
            or spec["foundation_workers"] != 36
            or spec["site_transfer_policy"] != DEBUG_PROFILE_TRANSFER
            or subject["content_hash"] != spec["parents"]["subject_gate"]
            or spec["imported_release"]["request"] != spec["request"]
            or release_root != Path(spec["source_gate_root"]) / "release"
        )
    elif version == 3:
        source_root = Path(spec["source_gate_root"])
        subject = load_json(source_root / "gate_spec.json")
        if subject.get("schema_version") != 2:
            raise ValueError("Preflight recovery must import the SPORC v2 gate")
        validate_gate(subject)
        release_root = Path(spec["imported_release_root"])
        foundation_root = Path(spec["foundation_root"])
        validate_release(spec["imported_release"], root=release_root)
        validate_foundation(spec["imported_foundation"], root=foundation_root)
        bounds = {
            role: preparation_bound(spec["imported_foundation"], role, spec["workers"])
            for role in ("train", "validation")
        }
        budgets = cache_budgets(
            spec["imported_foundation"],
            preflight_recovery_tasks()[0]["memory_mb"], spec["workers"],
        )
        differs = (
            spec["measurement_site"] != execution_site("sporc_a100_debug")
            or spec["execution_site"] != execution_site("sporc_a100")
            or spec["tasks"] != preflight_recovery_tasks()
            or spec["site_transfer_policy"] != DEBUG_PROFILE_TRANSFER
            or subject["content_hash"] != spec["parents"]["subject_gate"]
            or spec["imported_release"] != subject["imported_release"]
            or spec["imported_foundation"]["release"]["content_hash"]
            != spec["imported_release"]["content_hash"]
            or spec["imported_foundation"]["content_hash"]
            != spec["parents"]["imported_foundation"]
            or foundation_root != source_root / "foundation"
            or spec["cache_preparation_bounds"] != bounds
            or spec["cache_budgets"] != budgets
        )
    else:
        from .portable import validate_materialization

        materialized_root = Path(spec["materialization_root"])
        validate_materialization(spec["portable_materialization"], root=materialized_root)
        release_root = Path(spec["imported_release_root"])
        foundation_root = Path(spec["foundation_root"])
        validate_release(spec["imported_release"], root=release_root)
        validate_foundation(spec["imported_foundation"], root=foundation_root)
        population = spec.get("population_selection")
        if version == 6:
            validate_direct_coarse_population(
                population, foundation=spec["imported_foundation"],
                foundation_root=foundation_root,
            )
        bounds = {
            role: preparation_bound(
                spec["imported_foundation"], role, spec["workers"],
                foundation_root=foundation_root,
                population_selection=population,
            )
            for role in ("train", "validation")
        }
        task = (
            oscar_preflight_tasks()[0]
            if version == 4 else oscar_direct_coarse_preflight_tasks()[0]
            if version == 6 else oscar_dual_slot_preflight_tasks()[0]
        )
        budgets = cache_budgets(
            spec["imported_foundation"], task["memory_mb"], spec["workers"],
            foundation_root=foundation_root,
            population_selection=population,
        )
        concurrency_differs = (
            version in (5, 6) and spec.get("concurrency_intent") != {
                "jobs": 2,
                "per_job_cpus": 6,
                "per_job_memory_mb": 90_000,
                "per_job_gpus": 1,
                "aggregate_cpus": 12,
                "aggregate_memory_mb": 180_000,
                "aggregate_gpus": 2,
            }
        )
        differs = (
            spec["measurement_site"] != execution_site("oscar_l40s")
            or spec["execution_site"] != execution_site("oscar_l40s")
            or spec["tasks"] != (
                oscar_preflight_tasks()
                if version == 4 else oscar_direct_coarse_preflight_tasks()
                if version == 6 else oscar_dual_slot_preflight_tasks()
            )
            or spec["site_transfer_policy"] is not None
            or spec["imported_release"] != spec["portable_materialization"]["release"]
            or spec["imported_foundation"] != spec["portable_materialization"]["foundation"]
            or release_root != Path(spec["portable_materialization"]["release_root"])
            or foundation_root != Path(spec["portable_materialization"]["foundation_root"])
            or spec["cache_preparation_bounds"] != bounds
            or spec["cache_budgets"] != budgets
            or concurrency_differs
            or (
                version == 6
                and spec.get("scientific_branches") != ["DIRECT", "COARSE"]
            )
            or (version != 6 and "population_selection" in spec)
        )
    if common_differs or differs:
        raise ValueError("Proxy-ladder gate semantics differ")
    if check_source:
        _source(Path(spec["project_dir"]), spec["source_commit"])
        if source_lock(Path(spec["project_dir"]), spec["source_commit"]) != spec["source"]:
            raise ValueError("Proxy-ladder gate semantic source bytes differ")
    return digest


def validate_profile(profile: dict, *, foundation: dict, spec: dict) -> str:
    if spec.get("schema_version") == 7:
        from .literature import validate_profile as validate_literature_profile
        return validate_literature_profile(profile, foundation=foundation, spec=spec)
    version = profile.get("schema_version")
    expected = spec.get("schema_version")
    if version != expected:
        raise ValueError("Proxy-ladder runtime-profile/gate versions differ")
    digest = validate(
        profile, "RUNTIME_PROFILE", version=version,
        parents={"gate": spec["content_hash"], "foundation": foundation["content_hash"]},
    )
    if version == 1:
        site_differs = (
            profile["execution_site"] != execution_site("tigris_gh200")
            or "measurement_site" in profile or "site_transfer_policy" in profile
        )
    elif version in (2, 3):
        site_differs = (
            profile.get("measurement_site") != execution_site("sporc_a100_debug")
            or profile["execution_site"] != execution_site("sporc_a100")
            or profile.get("site_transfer_policy") != DEBUG_PROFILE_TRANSFER
        )
    else:
        site_differs = (
            profile.get("measurement_site") is not None
            or profile["execution_site"] != execution_site("oscar_l40s")
            or profile.get("site_transfer_policy") is not None
        )
    recovery_resources_differ = (
        version == 3 and (
            profile["cpus"] != preflight_recovery_tasks()[0]["cpus"]
            or profile["memory_mb"] != preflight_recovery_tasks()[0]["memory_mb"]
            or profile["workers"] != spec["workers"]
        )
    )
    oscar_task = (
        oscar_preflight_tasks()[0] if version == 4
        else oscar_dual_slot_preflight_tasks()[0] if version in (5, 6)
        else None
    )
    oscar_resources_differ = (
        version in (4, 5, 6) and (
            profile["cpus"] != oscar_task["cpus"]
            or profile["memory_mb"] != oscar_task["memory_mb"]
            or profile["workers"] != spec["workers"]
        )
    )
    population = spec.get("population_selection")
    population_differs = (
        version == 6 and (
            profile.get("population_selection_sha256")
            != population["content_hash"]
            or profile.get("measured_role_counts") != population["counts"]
        )
    ) or (version != 6 and (
        "population_selection_sha256" in profile
        or "measured_role_counts" in profile
    ))
    if (
        profile["source_commit"] != spec["source_commit"]
        or site_differs
        or recovery_resources_differ
        or oscar_resources_differ
        or population_differs
        or profile["foundation_sha256"] != foundation["content_hash"]
        or profile["model"] != model_contract()
        or profile["passed"] is not True
        or profile["measured_full_population"] is not True
        or profile["ram_only_views"] is not True
        or profile["rolling_resume"] is not False
        or not 60 <= profile["train_minutes"] <= 2880
        or not 30 <= profile["reduce_minutes"] <= 1440
        or profile["execution_site"]["gpu_family"] not in profile["gpu"]["name"]
        or profile["gpu_peak_bytes"] > .85 * profile["gpu"]["total_memory_bytes"]
    ):
        raise ValueError("Proxy-ladder measured runtime profile differs")
    validate_resources(
        profile["execution_site"], profile["cpus"], profile["memory_mb"], profile["workers"],
    )
    if profile["cache_budgets"] != cache_budgets(
        foundation, profile["memory_mb"], profile["workers"],
        foundation_root=Path(spec.get("foundation_root", ".")),
        population_selection=population,
    ):
        raise ValueError("Proxy-ladder measured cache budgets differ")
    return digest


def _measure_preflight(spec: dict, foundation: dict, *, output_root: Path) -> dict:
    literature = spec.get("schema_version") == 7
    site = spec.get("measurement_site", spec["execution_site"])
    job_id, cpus, memory_mb = allocation(site)
    workers = spec["workers"]
    validate_resources(site, cpus, memory_mb, workers)
    foundation_root = Path(spec.get("foundation_root", output_root.parent / "foundation"))
    population = spec.get("population_selection")
    budgets = cache_budgets(
        foundation, memory_mb, workers, foundation_root=foundation_root,
        population_selection=population,
    )
    environment = installed_environment()
    if literature:
        from .literature import scientific_plan
        plan = scientific_plan(foundation, foundation_root=foundation_root)
    else:
        plan = (
            build_direct_coarse_plan(
                foundation, population_selection=population,
                foundation_root=foundation_root,
            )
            if spec.get("schema_version") == 6
            else build_scientific_plan(foundation, foundation_root=foundation_root)
        )
    node = next(row for row in plan["nodes"] if row["node_id"] == "U000")
    started = time.monotonic()
    train = prepare_cache(
        foundation, foundation_root=foundation_root, role="train",
        coordinate="U000", workers=workers, max_ram_bytes=budgets["train"],
        population_selection=population,
    )
    validation_cache = prepare_cache(
        foundation, foundation_root=foundation_root, role="validation",
        coordinate="U000", workers=workers, max_ram_bytes=budgets["validation"],
        population_selection=population,
    )
    u000_cache_seconds = time.monotonic() - started
    parity_fields = {}
    if literature:
        from hlt_classification.jetclass2_delphes.acceptance import installed_parity
        parity_fields = {"installed_weaver_parity": installed_parity(train, device="cuda"),
                         "measured_role_counts": {"train": len(train), "validation": len(validation_cache)}}
    torch.manual_seed(node["initialization_seed"])
    model = DelphesParticleTransformer()
    torch.cuda.reset_peak_memory_stats()
    report, _ = train_kernel(
        model, train, validation_cache, node=node, device="cuda", acceptance_passes=1,
    )
    started = time.monotonic()
    for cache, temperature in ((train, 2.), (validation_cache, 1.)):
        values = predict(model, cache, device="cuda", temperature=temperature)
        del values
    inference_seconds = time.monotonic() - started
    cache_bytes = train.nbytes + validation_cache.nbytes
    gpu_peak = torch.cuda.max_memory_allocated()
    del train, validation_cache, cache, model
    torch.cuda.empty_cache()
    started = time.monotonic()
    dense = [
        prepare_cache(
            foundation, foundation_root=foundation_root, role=role,
            coordinate="D050", workers=workers, max_ram_bytes=budgets[role],
            population_selection=population,
        )
        for role in ("train", "validation")
    ]
    d050_cache_seconds = time.monotonic() - started
    cache_bytes = max(cache_bytes, sum(cache.nbytes for cache in dense))
    del dense
    worst_cache = max(u000_cache_seconds, d050_cache_seconds)
    train_minutes = max(60, math.ceil((worst_cache + 100 * report["runtime_seconds"]) * 1.75 / 60))
    reduce_minutes = max(30, math.ceil((worst_cache + inference_seconds) * 2 / 60))
    if train_minutes > 2880 or reduce_minutes > 1440:
        raise ValueError("Measured proxy-ladder walltime exceeds registered envelope")
    if literature and train_minutes > 1440:
        raise ValueError(f"Measured training request {train_minutes} minutes exceeds debug's 24h limit; no science admitted")
    transfer = (
        dict(measurement_site=site, site_transfer_policy=DEBUG_PROFILE_TRANSFER)
        if site["name"] == "sporc_a100_debug" and not literature else {}
    )
    population_fields = (
        {
            "population_selection_sha256": population["content_hash"],
            "measured_role_counts": population["counts"],
        }
        if population is not None else {}
    )
    profile = artifact(
        "RUNTIME_PROFILE", version=spec.get("schema_version", 1),
        parents={"gate": spec["content_hash"], "foundation": foundation["content_hash"]},
        source_commit=spec["source_commit"], foundation_sha256=foundation["content_hash"],
        execution_site=site if literature else production_site(site), slurm_job_id=job_id, passed=True,
        installed_environment=environment, model=model_contract(),
        measured_full_population=True, cpus=cpus, memory_mb=memory_mb, workers=workers,
        train_minutes=train_minutes, reduce_minutes=reduce_minutes,
        cache_budgets=budgets,
        cache_seconds_by_coordinate={"U000": u000_cache_seconds, "D050": d050_cache_seconds},
        one_pass_seconds=report["runtime_seconds"], inference_seconds=inference_seconds,
        cache_bytes=cache_bytes, gpu=gpu_identity(), gpu_peak_bytes=gpu_peak,
        acceptance_training_report=report,
        ram_only_views=True, rolling_resume=False, **population_fields, **transfer, **parity_fields,
    )
    validate_profile(profile, foundation=foundation, spec=spec)
    return profile


def run_gate_task(spec: dict, task_id: str) -> dict:
    validate_gate(spec, check_source=True)
    if task_id not in {row["task_id"] for row in spec["tasks"]}:
        raise ValueError("Unregistered proxy-ladder gate task")
    root = Path(spec["gate_root"])
    release_root = (
        Path(spec["imported_release_root"])
        if spec.get("schema_version") in (2, 3, 4, 5, 6) else root / "release"
    )
    foundation_root = Path(spec.get("foundation_root", root / "foundation"))
    evidence_root = root / "evidence"
    if task_id == "authenticate_release":
        if spec.get("schema_version") == 2:
            result = load_json(release_root / "release.json")
            validate_release(result, root=release_root)
            if result != spec["imported_release"]:
                raise ValueError("Imported proxy-ladder release bytes differ")
        else:
            result = build_release(spec["request"], output_root=release_root)
        pointer = artifact(
            "GATE_TASK", parents={"gate": spec["content_hash"]},
            task_id=task_id, result_sha256=result["content_hash"],
            outputs=[file_ref(release_root / "release.json"), file_ref(release_root / "release_index.npz")],
        )
    elif task_id == "build_foundation":
        if not (release_root / "release.json").is_file():
            raise ValueError("Missing authenticated release parent")
        release = load_json(release_root / "release.json")
        validate_release(release, root=release_root)
        result = build_foundation(
            release, release_root=release_root, output_root=foundation_root,
            capacity=spec["capacity"], workers=spec.get("foundation_workers", 72),
        )
        pointer = artifact(
            "GATE_TASK", parents={"gate": spec["content_hash"]},
            task_id=task_id, result_sha256=result["content_hash"],
            outputs=[file_ref(foundation_root / "foundation.json"), file_ref(foundation_root / "assignments.npz")],
        )
    elif task_id == "preflight":
        foundation = load_json(foundation_root / "foundation.json")
        validate_foundation(foundation, root=foundation_root)
        evidence_root.mkdir(parents=True, exist_ok=False)
        result = _measure_preflight(spec, foundation, output_root=evidence_root)
        write_json(evidence_root / "runtime_profile.json", result)
        completion = artifact(
            "GATE_COMPLETE",
            parents={
                "gate": spec["content_hash"], "foundation": foundation["content_hash"],
                "profile": result["content_hash"],
            },
            passed=True, source_commit=spec["source_commit"],
            release_sha256=foundation["release"]["content_hash"],
            foundation_sha256=foundation["content_hash"],
            runtime_profile_sha256=result["content_hash"],
            full_scientific_submission_authorized=False,
        )
        write_json(root / "gate_complete.json", completion)
        pointer = artifact(
            "GATE_TASK", parents={"gate": spec["content_hash"]},
            task_id=task_id, result_sha256=result["content_hash"],
            outputs=[file_ref(evidence_root / "runtime_profile.json"), file_ref(root / "gate_complete.json")],
        )
    else:
        raise ValueError("Unknown proxy-ladder gate task")
    task_path = root / "tasks" / f"{task_id}.json"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(task_path, pointer)
    return pointer


__all__ = [
    "AUTHORIZATION", "DEBUG_AUTHORIZATION", "PREFLIGHT_RECOVERY_AUTHORIZATION",
    "OSCAR_AUTHORIZATION", "OSCAR_DUAL_SLOT_AUTHORIZATION",
    "OSCAR_DIRECT_COARSE_AUTHORIZATION",
    "create_oscar_gate", "create_oscar_dual_slot_gate",
    "create_oscar_direct_coarse_gate",
    "create_gate", "create_sporc_debug_gate", "create_sporc_preflight_recovery",
    "debug_gate_tasks", "gate_tasks", "preflight_recovery_tasks",
    "oscar_preflight_tasks", "oscar_dual_slot_preflight_tasks",
    "oscar_direct_coarse_preflight_tasks",
    "run_gate_task",
    "source_lock", "validate_gate", "validate_profile",
]

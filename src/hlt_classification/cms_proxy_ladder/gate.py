"""Source-pinned Tigris release, matcher, and measured-runtime gate."""
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
    allocation, execution_site, gpu_identity, validate_resources,
)
from hlt_classification.jetclass2_delphes.model import (
    DelphesParticleTransformer, installed_environment, model_contract,
)
from hlt_classification.jetclass2_delphes.runner import predict, train_kernel

from .cache import cache_budgets, prepare_cache
from .campaign import build_scientific_plan
from .contracts import artifact, file_ref, validate, write_json
from .data import build_foundation, validate_foundation
from .release import build_release, release_request, validate_release

SOURCE_FILES = (
    "docs/plans/JETCLASS2_CMS_PROXY_TIGRIS_200K_THREE_SPINE_IMPLEMENTATION_PLAN.md",
    "docs/contracts/JETCLASS2_CMS_PROXY_LADDER.md",
    "docs/JETCLASS2_CMS_PROXY_TIGRIS_200K_HANDOFF.md",
    "scripts/jetclass2_cms_proxy_ladder.py",
    "scripts/run_jetclass2_cms_proxy_ladder_task.py",
    "scripts/submit_jetclass2_cms_proxy_ladder.py",
    "sbatch/run_jetclass2_cms_proxy_ladder.sh",
    "tests/test_cms_proxy_ladder.py",
)
AUTHORIZATION = "AUTHORIZE JETCLASS2 CMS PROXY 200K TIGRIS GATE"


def _source(project: Path, commit: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Exact 40-character pushed commit required")
    validate_source_checkout(Path(project), expected_commit=commit)


def source_lock(project: Path, commit: str) -> dict:
    _source(project, commit)
    root = Path(project).resolve()
    package = sorted((root / "src/hlt_classification/cms_proxy_ladder").glob("*.py"))
    files = [root / name for name in SOURCE_FILES] + package
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


def validate_gate(spec: dict, *, check_source: bool = False) -> str:
    digest = validate(
        spec, "GATE_SPEC",
        parents={"source": spec["source"]["content_hash"], "request": spec["request"]["content_hash"]},
    )
    validate(spec["source"], "SOURCE")
    from .release import validate_request
    validate_request(spec["request"])
    if (
        spec["execution_site"] != execution_site("tigris_gh200")
        or spec["tasks"] != gate_tasks() or spec["workers"] != 16
        or spec["full_views_persisted"] is not False
        or type(spec["capacity"]) is not int or spec["capacity"] < 16
    ):
        raise ValueError("Proxy-ladder gate semantics differ")
    if check_source:
        _source(Path(spec["project_dir"]), spec["source_commit"])
        if source_lock(Path(spec["project_dir"]), spec["source_commit"]) != spec["source"]:
            raise ValueError("Proxy-ladder gate semantic source bytes differ")
    return digest


def validate_profile(profile: dict, *, foundation: dict, spec: dict) -> str:
    digest = validate(
        profile, "RUNTIME_PROFILE",
        parents={"gate": spec["content_hash"], "foundation": foundation["content_hash"]},
    )
    if (
        profile["source_commit"] != spec["source_commit"]
        or profile["execution_site"] != execution_site("tigris_gh200")
        or profile["foundation_sha256"] != foundation["content_hash"]
        or profile["model"] != model_contract()
        or profile["passed"] is not True
        or profile["measured_full_population"] is not True
        or profile["ram_only_views"] is not True
        or profile["rolling_resume"] is not False
        or not 60 <= profile["train_minutes"] <= 2880
        or not 30 <= profile["reduce_minutes"] <= 1440
        or profile["gpu"]["name"].find("GH200") < 0
        or profile["gpu_peak_bytes"] > .85 * profile["gpu"]["total_memory_bytes"]
    ):
        raise ValueError("Proxy-ladder measured runtime profile differs")
    validate_resources(
        profile["execution_site"], profile["cpus"], profile["memory_mb"], profile["workers"],
    )
    if profile["cache_budgets"] != cache_budgets(
        foundation, profile["memory_mb"], profile["workers"],
    ):
        raise ValueError("Proxy-ladder measured cache budgets differ")
    return digest


def _measure_preflight(spec: dict, foundation: dict, *, output_root: Path) -> dict:
    site = spec["execution_site"]
    job_id, cpus, memory_mb = allocation(site)
    workers = spec["workers"]
    validate_resources(site, cpus, memory_mb, workers)
    budgets = cache_budgets(foundation, memory_mb, workers)
    environment = installed_environment()
    plan = build_scientific_plan(foundation, foundation_root=output_root.parent / "foundation")
    node = next(row for row in plan["nodes"] if row["node_id"] == "U000")
    started = time.monotonic()
    train = prepare_cache(
        foundation, foundation_root=output_root.parent / "foundation", role="train",
        coordinate="U000", workers=workers, max_ram_bytes=budgets["train"],
    )
    validation_cache = prepare_cache(
        foundation, foundation_root=output_root.parent / "foundation", role="validation",
        coordinate="U000", workers=workers, max_ram_bytes=budgets["validation"],
    )
    u000_cache_seconds = time.monotonic() - started
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
            foundation, foundation_root=output_root.parent / "foundation", role=role,
            coordinate="D050", workers=workers, max_ram_bytes=budgets[role],
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
    profile = artifact(
        "RUNTIME_PROFILE",
        parents={"gate": spec["content_hash"], "foundation": foundation["content_hash"]},
        source_commit=spec["source_commit"], foundation_sha256=foundation["content_hash"],
        execution_site=site, slurm_job_id=job_id, passed=True,
        installed_environment=environment, model=model_contract(),
        measured_full_population=True, cpus=cpus, memory_mb=memory_mb, workers=workers,
        train_minutes=train_minutes, reduce_minutes=reduce_minutes,
        cache_budgets=budgets,
        cache_seconds_by_coordinate={"U000": u000_cache_seconds, "D050": d050_cache_seconds},
        one_pass_seconds=report["runtime_seconds"], inference_seconds=inference_seconds,
        cache_bytes=cache_bytes, gpu=gpu_identity(), gpu_peak_bytes=gpu_peak,
        acceptance_training_report=report,
        ram_only_views=True, rolling_resume=False,
    )
    validate_profile(profile, foundation=foundation, spec=spec)
    return profile


def run_gate_task(spec: dict, task_id: str) -> dict:
    validate_gate(spec, check_source=True)
    root = Path(spec["gate_root"])
    release_root, foundation_root, evidence_root = root / "release", root / "foundation", root / "evidence"
    if task_id == "authenticate_release":
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
            capacity=spec["capacity"], workers=72,
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
    "AUTHORIZATION", "create_gate", "gate_tasks", "run_gate_task", "source_lock",
    "validate_gate", "validate_profile",
]

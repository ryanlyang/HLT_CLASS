"""Source-pinned, acceptance-gated production publication for the new benchmark."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import math
import re
import time

import numpy as np
import torch

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, load_json, sha256_file, write_immutable_json,
)
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout
from .acceptance import run_acceptance
from .banks import publish_bank, load_bank
from .cache import prepare_cache, cache_budgets
from .campaign import build_campaign_plan
from .contracts import artifact, relative_file, validate
from .foundation import (
    build_foundation_lock, validate_foundation_spec, load_assignments, assignment_source,
    build_assignment_shard, audit_sample,
)
from .inventory import verify_snapshot
from .model import DelphesParticleTransformer, model_contract, installed_environment
from .reporting import recovery
from .runner import train_kernel, predict
from .splits import is_subset_profile
from .execution import allocation, gpu_identity, validate_site, validate_resources

AUTHORIZE = "AUTHORIZE EXACT JETCLASS2 DELPHES FOUR SPINE CAMPAIGN"


def _source(project: Path, commit: str):
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Exact pushed commit required")
    validate_source_checkout(project, expected_commit=commit)


def authenticate_preparation(foundation, foundation_root):
    expected = assignment_source()
    for task in foundation["assignment_tasks"]:
        report, _ = load_assignments(foundation, root=foundation_root, file_index=task["file_index"])
        if report["producer"]["file_sha256"] != expected["file_sha256"]:
            raise ValueError("Assignment producer differs from pinned preparation code")
    audit = load_json(Path(foundation_root) / "sample_audit.json")
    validate(audit, "SAMPLE_AUDIT")
    if (audit["foundation_sha256"] != foundation["content_hash"] or audit["passed"] is not True
            or audit["exact_endpoint_checks"] is not True or audit["final_test_accessed"] is not False):
        raise ValueError("New-data endpoint/matching sample acceptance differs")


def run_preparation(foundation, *, foundation_root, data_root, project, source_commit,
                    task, array_index=None):
    _source(project, source_commit)
    validate_foundation_spec(foundation)
    if Path(foundation_root).resolve().is_relative_to(Path(data_root).resolve()):
        raise ValueError("Preparation output cannot be inside raw snapshot")
    if task == "sample":
        result = audit_sample(foundation, data_root=data_root)
        write_immutable_json(Path(foundation_root) / "sample_audit.json", result)
        return result
    if task == "assign":
        if type(array_index) is not int or not 0 <= array_index < len(foundation["assignment_tasks"]):
            raise ValueError("Array index is not an ordinary-role preparation task")
        # Array positions enumerate only registered train/validation files, not
        # all 333 raw indices; final-test files can never enter this array.
        index = foundation["assignment_tasks"][array_index]["file_index"]
        return build_assignment_shard(foundation, data_root=data_root, output_root=foundation_root, file_index=index)
    if task == "lock":
        authenticate_preparation(foundation, foundation_root)
        return build_foundation_lock(foundation, foundation_root)
    raise ValueError("Unknown preparation task")


def measure_runtime(foundation: dict, *, foundation_root: Path, data_root: Path,
                    output_root: Path, project: Path, source_commit: str, site: dict,
                    workers: int = 4, max_train_minutes: int = 2880) -> dict:
    """Real site-specific miniature plus a selected-population resource pass.

    Profile checkpoints are never published as campaign fits. This is more
    expensive than a synthetic smoke test, and is explicitly separate from it.
    """
    _source(project, source_commit)
    validate_foundation_spec(foundation)
    if Path(output_root).exists() or Path(output_root).resolve().is_relative_to(Path(data_root).resolve()):
        raise ValueError("Runtime acceptance needs a fresh output root outside raw data")
    if not is_subset_profile(foundation["splits"]):
        raise ValueError("Resource acceptance requires an explicit selected split profile")
    job_id, cpus, mem_mb = allocation(site)
    validate_resources(site, cpus, mem_mb, workers)
    budgets = cache_budgets(foundation, mem_mb, workers)
    if type(max_train_minutes) is not int or not 60 <= max_train_minutes <= 8640:
        raise ValueError("Invalid explicit maximum fit-walltime envelope")
    authenticate_preparation(foundation, foundation_root)
    environment = installed_environment()
    print(f"JC2 phase=acceptance_start site={site['name']} job={job_id} cpus={cpus} workers={workers} memory_mb={mem_mb} cache_budgets={budgets}", flush=True)
    mini = run_acceptance(foundation, data_root=data_root, output_root=output_root / "miniature", device="cuda")
    # Reserve a quarter of host memory for raw blocks/IPC/model/runtime outside
    # the two explicit cache budgets. Both preparation functions bound workers.
    started = time.monotonic()
    kwargs = dict(spec=foundation, data_root=data_root, foundation_root=foundation_root,
                  coordinate_name="U000", workers=workers)
    train = prepare_cache(role="train", max_ram_bytes=budgets["train"], **kwargs)
    validation = prepare_cache(role="validation", max_ram_bytes=budgets["validation"], **kwargs)
    cache_seconds = time.monotonic() - started
    node = next(n for n in build_campaign_plan(foundation)["nodes"] if n["node_id"] == "U000")
    torch.manual_seed(node["initialization_seed"])
    model = DelphesParticleTransformer()
    torch.cuda.reset_peak_memory_stats()
    report, _ = train_kernel(model, train, validation, node=node, device="cuda", acceptance_passes=1)
    gpu_peak = torch.cuda.max_memory_allocated()
    if gpu_peak > .85 * torch.cuda.get_device_properties(0).total_memory:
        raise MemoryError("Measured GPU memory has insufficient production margin")
    # Also measure the registered worst padded token count at batch 256.
    raw = train.batch(np.arange(256) % len(train))
    pad = foundation["inputs"]["capacity"] - raw["features"].shape[-1]
    data = {k: torch.from_numpy(np.pad(raw[k], ((0, 0), (0, 0), (0, pad)))).cuda()
            for k in ("features", "vectors", "mask")}
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(**data)
    loss = torch.nn.functional.cross_entropy(logits.float(), torch.from_numpy(raw["labels"]).cuda())
    if not torch.isfinite(loss):
        raise ValueError("Worst-padding resource probe is nonfinite")
    loss.backward()
    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise ValueError("Worst-padding resource probe has nonfinite gradients")
    gpu_peak = max(gpu_peak, torch.cuda.max_memory_allocated())
    if gpu_peak > .85 * torch.cuda.get_device_properties(0).total_memory:
        raise MemoryError("Worst-padding GPU memory has insufficient production margin")
    # Measure reducer inference, and the expensive non-endpoint cache paths.
    # No predictions/views from these timing probes are saved to disk.
    del data, logits, loss, raw
    model.zero_grad(set_to_none=True)
    started = time.monotonic()
    for role_cache, temperature in ((train, 2.), (validation, 1.)):
        probabilities = predict(model, role_cache, device="cuda", temperature=temperature)
        del probabilities
    inference_seconds = time.monotonic() - started
    peak_cache_bytes = train.nbytes + validation.nbytes
    selected_state_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    gpu_peak = max(gpu_peak, torch.cuda.max_memory_allocated())
    del train, validation, role_cache, model
    torch.cuda.empty_cache()
    cache_timings = {"U000": cache_seconds}
    for coordinate_name in ("U050", "D050"):
        started = time.monotonic()
        caches = [prepare_cache(foundation, data_root=data_root, foundation_root=foundation_root,
                                role=role, coordinate_name=coordinate_name, workers=workers,
                                max_ram_bytes=budgets[role]) for role in ("train", "validation")]
        cache_timings[coordinate_name] = time.monotonic() - started
        peak_cache_bytes = max(peak_cache_bytes, sum(c.nbytes for c in caches))
        del caches
    worst_cache_seconds = max(cache_timings.values())
    train_minutes = max(60, math.ceil((worst_cache_seconds + 100 * report["runtime_seconds"]) * 1.75 / 60))
    reduce_minutes = max(30, math.ceil((worst_cache_seconds + inference_seconds) * 2 / 60))
    # Persist diagnostic measurements even when planning fails. A failed probe
    # NEVER emits an eligible runtime_profile.json or starts scientific fits.
    write_immutable_json(output_root / "resource_training_report.json", report)
    write_immutable_json(output_root / "resource_measurements.json", artifact(
        "RESOURCE_MEASUREMENTS", foundation_sha256=foundation["content_hash"], source_commit=source_commit,
        execution_site=site, slurm_job_id=job_id, cache_seconds_by_coordinate=cache_timings,
        one_pass_seconds=report["runtime_seconds"], inference_seconds=inference_seconds,
        proposed_train_minutes=train_minutes, proposed_reduce_minutes=reduce_minutes,
        max_train_minutes=max_train_minutes, final_test_accessed=False))
    if train_minutes > max_train_minutes or reduce_minutes > max_train_minutes:
        raise ValueError("Measured walltime exceeds the explicit planning envelope; inspect resource_measurements.json and revise resources explicitly")
    result = artifact(
        "RUNTIME_PROFILE", version=2, foundation_sha256=foundation["content_hash"], source_commit=source_commit,
        miniature_sha256=mini["content_hash"], model=model_contract(), passed=True,
        installed_environment=environment,
        miniature=mini, resource_training_report=report,
        measured_full_population=True, slurm_job_id=job_id, execution_site=site, gpu=gpu_identity(),
        cpus=cpus, memory_mb=mem_mb, workers=workers, train_minutes=train_minutes,
        reduce_minutes=reduce_minutes, max_train_minutes=max_train_minutes, cache_budgets=budgets,
        cache_seconds=worst_cache_seconds, cache_seconds_by_coordinate=cache_timings,
        one_pass_seconds=report["runtime_seconds"], inference_seconds=inference_seconds,
        cache_bytes=peak_cache_bytes, gpu_peak_bytes=gpu_peak,
        selected_state_bytes=selected_state_bytes,
        ram_only_views=True, rolling_resume=False, final_test_accessed=False,
    )
    validate_profile(result, foundation, source_commit)
    write_immutable_json(output_root / "runtime_profile.json", result)
    return result


def validate_profile(profile: dict, foundation: dict, commit: str):
    validate(profile, "RUNTIME_PROFILE", version=2)
    validate(profile["installed_environment"], "INSTALLED_ENVIRONMENT", version=2)
    validate_site(profile["execution_site"])
    validate_resources(profile["execution_site"], profile["cpus"], profile["memory_mb"], profile["workers"])
    mini = profile["miniature"]
    validate(mini, "LOCAL_OR_REMOTE_ACCEPTANCE")
    report = profile["resource_training_report"]
    validate(report, "KERNEL_TRAINING_REPORT")
    if (mini["content_hash"] != profile["miniature_sha256"] or mini["foundation_sha256"] != foundation["content_hash"]
            or mini["passed"] is not True or mini["scientific_results"] is not False or mini["final_test_accessed"] is not False
            or report["foundation_sha256"] != foundation["content_hash"] or report["scientific_fit"] is not False
            or report["acceptance_only"] is not True or report["passes"] != 1 or report["final_test_accessed"] is not False):
        raise ValueError("Measured miniature/full-population probe lineage differs")
    if (profile["foundation_sha256"] != foundation["content_hash"] or profile["source_commit"] != commit
            or profile["model"] != model_contract() or profile["passed"] is not True
            or profile["measured_full_population"] is not True or profile["ram_only_views"] is not True
            or profile["rolling_resume"] is not False or profile["final_test_accessed"] is not False
            or profile["cache_budgets"] != cache_budgets(foundation, profile["memory_mb"], profile["workers"])
            or not 60 <= profile["train_minutes"] <= profile["max_train_minutes"] <= 8640
            or not 30 <= profile["reduce_minutes"] <= profile["max_train_minutes"]
            or re.fullmatch(r"[1-9][0-9]*", profile["slurm_job_id"]) is None
            or profile["execution_site"]["gpu_family"] not in profile["gpu"]["name"]
            or not 0 < profile["gpu_peak_bytes"] <= .85 * profile["gpu"]["total_memory_bytes"]
            or profile["selected_state_bytes"] <= 0):
        raise ValueError("New-dataset measured runtime evidence differs")


def task_graph(plan: dict) -> list[dict]:
    rows = []
    for node in plan["nodes"]:
        deps = [] if node["teacher"] is None else ["reduce_" + node["teacher"]]
        rows.append(dict(task_id="train_" + node["node_id"], kind="train", node_id=node["node_id"], dependencies=deps))
        if node["node_id"] in plan["probability_publications"]:
            rows.append(dict(task_id="reduce_" + node["node_id"], kind="reduce", node_id=node["node_id"], dependencies=["train_" + node["node_id"]]))
    rows.append(dict(task_id="aggregate", kind="aggregate", node_id=None, dependencies=[r["task_id"] for r in rows]))
    rows.append(dict(task_id="campaign_complete", kind="complete", node_id=None, dependencies=["aggregate"]))
    return rows


def create_campaign(foundation: dict, *, foundation_root: Path, data_root: Path, campaign_root: Path,
                    project: Path, source_commit: str, profile: dict) -> dict:
    _source(project, source_commit)
    validate_foundation_spec(foundation)
    if not is_subset_profile(foundation["splits"]):
        raise ValueError("Production requires an explicit split-registry profile, not the full-data reservoir")
    validate_profile(profile, foundation, source_commit)
    root = Path(campaign_root).resolve()
    if root.exists() or root.is_relative_to(Path(data_root).resolve()) or root.is_relative_to(Path(foundation_root).resolve()):
        raise FileExistsError("Campaign requires a fresh output root outside raw data/foundation")
    verify_snapshot(Path(data_root), foundation["inventory"])
    authenticate_preparation(foundation, foundation_root)
    lock = build_foundation_lock(foundation, foundation_root)
    plan = build_campaign_plan(foundation)
    spec = artifact(
        "CAMPAIGN_SPEC", version=3, foundation=foundation, foundation_root=str(Path(foundation_root).resolve()),
        foundation_lock_sha256=lock["content_hash"], data_root=str(Path(data_root).resolve()),
        campaign_root=str(root), project_dir=str(Path(project).resolve()), source_commit=source_commit,
        runtime_profile=profile, scientific_plan=plan, tasks=task_graph(plan),
        model=model_contract(), final_test_accessed=False, existing_campaign_mutations=False,
    )
    write_immutable_json(root / "campaign_spec.json", spec)
    return spec


def validate_campaign(spec: dict, *, check_source=True):
    digest = validate(spec, "CAMPAIGN_SPEC", version=3)
    validate_foundation_spec(spec["foundation"])
    if not is_subset_profile(spec["foundation"]["splits"]):
        raise ValueError("Production requires an explicit split-registry profile")
    validate_profile(spec["runtime_profile"], spec["foundation"], spec["source_commit"])
    plan = build_campaign_plan(spec["foundation"])
    if (spec["scientific_plan"] != plan or spec["tasks"] != task_graph(plan) or spec["model"] != model_contract()
            or spec["final_test_accessed"] is not False or spec["existing_campaign_mutations"] is not False):
        raise ValueError("Production campaign graph/model differs")
    if check_source:
        _source(Path(spec["project_dir"]), spec["source_commit"])
    lock = load_json(Path(spec["foundation_root"]) / "foundation_lock.json")
    validate(lock, "FOUNDATION_LOCK")
    if lock["content_hash"] != spec["foundation_lock_sha256"] or lock["foundation_sha256"] != spec["foundation"]["content_hash"]:
        raise ValueError("Foundation completion lock differs")
    return digest


def _report_path(spec, task):
    if task not in {r["task_id"] for r in spec["tasks"]}:
        raise ValueError("Unknown campaign task identity")
    return Path(spec["campaign_root"]) / "tasks" / f"{task}.json"


def completed_task(spec: dict, task: str) -> dict | None:
    path = _report_path(spec, task)
    if not path.exists():
        return None
    report = load_json(path)
    validate(report, "TASK_REPORT")
    if (report["campaign_sha256"] != spec["content_hash"] or report["task_id"] != task
            or report["source_commit"] != spec["source_commit"] or report["final_test_accessed"] is not False):
        raise ValueError("Task report lineage differs")
    kind = next(r["kind"] for r in spec["tasks"] if r["task_id"] == task)
    paths = [row["path"] for row in report["outputs"]]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate task output inventory entries")
    result = report["result"]
    required = []
    if kind == "train":
        required = [result["checkpoint"], result["training_report"]]
    elif kind == "reduce":
        required = [result[role + "_bank"] + "/manifest.json" for role in ("train", "validation")]
    if any(path not in paths for path in required) or (kind != "complete" and not paths):
        raise ValueError("Completed task lacks required durable outputs")
    for output in report["outputs"]:
        target = relative_file(Path(spec["campaign_root"]), output["path"])
        if sha256_file(target) != output["sha256"]:
            raise ValueError("Completed task output checksum differs")
    return report


def result_rows(spec):
    """Read-only, complete-or-pending view; absent references give null recovery."""
    validate_campaign(spec)
    reports = {}
    for node in spec["scientific_plan"]["nodes"]:
        pointer = completed_task(spec, "train_" + node["node_id"])
        if pointer is None:
            continue
        result = pointer["result"]
        report = load_json(relative_file(Path(spec["campaign_root"]), result["training_report"]))
        validate(report, "KERNEL_TRAINING_REPORT")
        if (report["content_hash"] != result["training_report_sha256"] or report["scientific_fit"] is not True
                or report["node"] != node or report["foundation_sha256"] != spec["foundation"]["content_hash"]
                or report["final_test_accessed"] is not False):
            raise ValueError("Training report is not this registered scientific node")
        reports[node["node_id"]] = report
    baseline = reports.get("M0HLT", {}).get("validation")
    oracle = reports.get("U000", {}).get("validation")
    rows = []
    for node in spec["scientific_plan"]["nodes"]:
        report = reports.get(node["node_id"])
        metrics = None if report is None else report["validation"]
        rows.append(dict(node_id=node["node_id"], branch=node["branch"],
                         state="PENDING" if report is None else "COMPLETE",
                         selected_pass=None if report is None else report["selected_pass"],
                         passes=None if report is None else report["passes"], validation=metrics,
                         recovery=None if metrics is None or baseline is None or oracle is None
                         else recovery(metrics, baseline, oracle)))
    return rows


def execution_gate(profile: dict, device):
    if str(device) not in {"cuda", "cuda:0"}:
        raise PermissionError("Production fits use one GPU; CPU is acceptance-test only")
    _, cpus, memory_mb = allocation(profile["execution_site"])
    if cpus != profile["cpus"] or memory_mb != profile["memory_mb"]:
        raise ValueError("Worker allocation differs from measured resource contract")
    if gpu_identity() != profile["gpu"]:
        raise ValueError("Worker GPU differs from measured resource contract")
    if installed_environment() != profile["installed_environment"]:
        raise ValueError("Installed model/numerical environment differs from measured acceptance")


def run_task(spec: dict, task_id: str, *, attempt: str, device="cuda") -> dict:
    validate_campaign(spec)
    if re.fullmatch(r"[A-Za-z0-9_-]+", attempt) is None:
        raise ValueError("Unsafe attempt identity")
    tasks = {row["task_id"]: row for row in spec["tasks"]}
    if task_id not in tasks:
        raise ValueError("Unknown production task")
    existing = completed_task(spec, task_id)
    if existing is not None:
        return existing
    task = tasks[task_id]
    for dependency in task["dependencies"]:
        if completed_task(spec, dependency) is None:
            raise ValueError(f"Missing authenticated parent: {dependency}")
    root = Path(spec["campaign_root"])
    attempt_root = root / "attempts" / task_id / attempt
    # Each restart gets a separate, never-overwritten attempt. Only a verified
    # final pointer publishes it. Incomplete attempts are not resume checkpoints.
    attempt_root.mkdir(parents=True, exist_ok=False)
    outputs = []
    profile, foundation = spec["runtime_profile"], spec["foundation"]
    def save(name, payload):
        path = attempt_root / name
        write_immutable_json(path, payload)
        outputs.append(path)
        return path
    def cache(role, coord):
        return prepare_cache(foundation, data_root=Path(spec["data_root"]), foundation_root=Path(spec["foundation_root"]),
                             role=role, coordinate_name=coord, workers=profile["workers"],
                             max_ram_bytes=profile["cache_budgets"][role])
    if task["kind"] in {"train", "reduce"}:
        execution_gate(profile, device)
        node = next(n for n in spec["scientific_plan"]["nodes"] if n["node_id"] == task["node_id"])
        train, val = cache("train", node["coordinate"]), cache("validation", node["coordinate"])
        torch.manual_seed(node["initialization_seed"])
        model = DelphesParticleTransformer()
        if task["kind"] == "train":
            kwargs = {}
            if node["teacher"] is not None:
                teacher = completed_task(spec, "reduce_" + node["teacher"])
                source = teacher["result"]
                q = load_bank(relative_file(root, source["train_bank"]), foundation_sha256=foundation["content_hash"],
                              teacher_report_sha256=source["teacher_report_sha256"], teacher_node=node["teacher"],
                              role="train", expected_identities=train.identities)
                kwargs = dict(teacher_probabilities=q, teacher_identities=train.identities)
            report, state = train_kernel(model, train, val, node=node, device=device, **kwargs)
            checkpoint = attempt_root / "selected.pt"
            buffer = BytesIO()
            torch.save(state, buffer)
            atomic_publish_bytes(checkpoint, buffer.getvalue())
            outputs.append(checkpoint)
            path = save("training_report.json", report)
            result = dict(training_report=str(path.relative_to(root).as_posix()),
                          training_report_sha256=report["content_hash"], checkpoint=checkpoint.relative_to(root).as_posix())
        else:
            teacher = completed_task(spec, "train_" + node["node_id"])["result"]
            model.load_state_dict(torch.load(relative_file(root, teacher["checkpoint"]), map_location="cpu", weights_only=True))
            model.to(device).eval()
            result = dict(teacher_report_sha256=teacher["training_report_sha256"])
            for role, role_cache, temperature in (("train", train, 2.), ("validation", val, 1.)):
                bank_root = attempt_root / role
                p = predict(model, role_cache, device=device, temperature=temperature)
                publish_bank(bank_root, foundation_sha256=foundation["content_hash"], teacher_report_sha256=teacher["training_report_sha256"],
                             teacher_node=node["node_id"], role=role, identities=role_cache.identities, probabilities=p)
                outputs.extend(sorted(bank_root.iterdir()))
                result[role + "_bank"] = bank_root.relative_to(root).as_posix()
    elif task["kind"] == "aggregate":
        rows = result_rows(spec)
        if any(row["state"] != "COMPLETE" for row in rows):
            raise ValueError("Aggregate has unfinished registered fits")
        result = artifact("AGGREGATE", campaign_sha256=spec["content_hash"], final_test_accessed=False,
                          rows=rows)
        save("validation_aggregate.json", result)
    else:
        result = dict(fresh_fit_count=31, final_test_accessed=False, scientific_result_does_not_control_completion=True)
    final = artifact("TASK_REPORT", campaign_sha256=spec["content_hash"], source_commit=spec["source_commit"], task_id=task_id,
                     result=result, final_test_accessed=False,
                     outputs=[dict(path=p.relative_to(root).as_posix(), sha256=sha256_file(p)) for p in outputs])
    write_immutable_json(_report_path(spec, task_id), final)
    return final

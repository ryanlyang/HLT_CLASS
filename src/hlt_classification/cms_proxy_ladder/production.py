"""Measured, source-pinned scientific execution for the proxy ladder."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re

import torch

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, load_json, sha256_file,
)
from hlt_classification.jetclass2_delphes.banks import load_bank, publish_bank
from hlt_classification.jetclass2_delphes.execution import allocation, gpu_identity
from hlt_classification.jetclass2_delphes.model import (
    DelphesParticleTransformer, installed_environment, model_contract,
)
from hlt_classification.jetclass2_delphes.reporting import recovery
from hlt_classification.jetclass2_delphes.runner import predict, train_kernel
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout

from .cache import prepare_cache
from .campaign import build_scientific_plan, task_graph
from .contracts import artifact, file_ref, safe, validate, validate_file_ref, write_json
from .data import validate_foundation
from .gate import source_lock, validate_gate, validate_profile

AUTHORIZATION = "AUTHORIZE JETCLASS2 CMS PROXY 200K THREE SPINE SCIENCE"


def _source(project: Path, commit: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Exact pushed source commit required")
    validate_source_checkout(project, expected_commit=commit)


def create_campaign(*, gate_root: Path, campaign_root: Path) -> dict:
    gate_root = Path(gate_root).resolve(strict=True)
    gate = load_json(gate_root / "gate_spec.json")
    validate_gate(gate, check_source=True)
    foundation_root = Path(gate.get("foundation_root", gate_root / "foundation"))
    foundation = load_json(foundation_root / "foundation.json")
    validate_foundation(foundation, root=foundation_root)
    profile = load_json(gate_root / "evidence/runtime_profile.json")
    validate_profile(profile, foundation=foundation, spec=gate)
    complete = load_json(gate_root / "gate_complete.json")
    validate(
        complete, "GATE_COMPLETE",
        parents={
            "gate": gate["content_hash"], "foundation": foundation["content_hash"],
            "profile": profile["content_hash"],
        },
    )
    if (
        complete["passed"] is not True
        or complete["source_commit"] != gate["source_commit"]
        or complete["foundation_sha256"] != foundation["content_hash"]
        or complete["runtime_profile_sha256"] != profile["content_hash"]
    ):
        raise ValueError("Proxy-ladder gate completion differs")
    root = Path(campaign_root).resolve()
    if root.exists() or root.is_relative_to(gate_root):
        raise FileExistsError("Proxy-ladder campaign root must be fresh and separate from gate")
    plan = build_scientific_plan(foundation, foundation_root=foundation_root)
    spec = artifact(
        "CAMPAIGN_SPEC",
        parents={
            "gate": gate["content_hash"], "foundation": foundation["content_hash"],
            "profile": profile["content_hash"], "plan": plan["content_hash"],
        },
        gate_root=str(gate_root), campaign_root=str(root),
        project_dir=gate["project_dir"], source_commit=gate["source_commit"],
        source=gate["source"], foundation_root=str(foundation_root),
        foundation=foundation, runtime_profile=profile,
        scientific_plan=plan, tasks=task_graph(plan), model=model_contract(),
        fresh_fit_count=17, reducer_count=12,
        full_views_persisted=False, existing_campaign_mutations=False,
    )
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "campaign_spec.json", spec)
    validate_campaign(spec, check_source=True)
    return spec


def validate_campaign(spec: dict, *, check_source: bool = False) -> str:
    parents = {
        "gate": spec["parents"]["gate"],
        "foundation": spec["foundation"]["content_hash"],
        "profile": spec["runtime_profile"]["content_hash"],
        "plan": spec["scientific_plan"]["content_hash"],
    }
    digest = validate(spec, "CAMPAIGN_SPEC", parents=parents)
    gate = load_json(Path(spec["gate_root"]) / "gate_spec.json")
    validate_gate(gate, check_source=check_source)
    if gate["content_hash"] != parents["gate"]:
        raise ValueError("Proxy-ladder campaign gate lineage differs")
    validate_foundation(spec["foundation"], root=Path(spec["foundation_root"]))
    validate_profile(spec["runtime_profile"], foundation=spec["foundation"], spec=gate)
    plan = build_scientific_plan(spec["foundation"], foundation_root=Path(spec["foundation_root"]))
    if (
        spec["scientific_plan"] != plan or spec["tasks"] != task_graph(plan)
        or spec["model"] != model_contract()
        or spec["fresh_fit_count"] != 17 or spec["reducer_count"] != 12
        or spec["full_views_persisted"] is not False
        or spec["existing_campaign_mutations"] is not False
        or spec["source"] != source_lock(Path(spec["project_dir"]), spec["source_commit"])
    ):
        raise ValueError("Proxy-ladder campaign graph/source/model differs")
    if check_source:
        _source(Path(spec["project_dir"]), spec["source_commit"])
    return digest


def _task_path(spec: dict, task_id: str) -> Path:
    if task_id not in {row["task_id"] for row in spec["tasks"]}:
        raise ValueError("Unknown proxy-ladder task")
    return Path(spec["campaign_root"]) / "tasks" / f"{task_id}.json"


def completed_task(spec: dict, task_id: str) -> dict | None:
    path = _task_path(spec, task_id)
    if not path.is_file():
        return None
    report = load_json(path)
    validate(report, "TASK_REPORT", parents={"campaign": spec["content_hash"]})
    if (
        report["task_id"] != task_id or report["source_commit"] != spec["source_commit"]
        or len({row["path"] for row in report["outputs"]}) != len(report["outputs"])
    ):
        raise ValueError("Proxy-ladder completed task lineage differs")
    for record in report["outputs"]:
        validate_file_ref(record, root=Path(spec["campaign_root"]))
    return report


def result_rows(spec: dict) -> list[dict]:
    reports = {}
    root = Path(spec["campaign_root"])
    for node in spec["scientific_plan"]["nodes"]:
        pointer = completed_task(spec, "train_" + node["node_id"])
        if pointer is None:
            continue
        path = safe(root, pointer["result"]["training_report"])
        report = load_json(path)
        # Kernel report retains its established versioned contract.
        if (
            report["content_hash"] != pointer["result"]["training_report_sha256"]
            or report["node"] != node or report["scientific_fit"] is not True
            or report["foundation_sha256"] != spec["foundation"]["content_hash"]
            or report["final_test_accessed"] is not False
        ):
            raise ValueError("Proxy-ladder scientific training report differs")
        reports[node["node_id"]] = report
    baseline = reports.get("M0HLT", {}).get("validation")
    oracle = reports.get("OFFLINE", {}).get("validation")
    rows = []
    for node in spec["scientific_plan"]["nodes"]:
        report = reports.get(node["node_id"])
        metrics = None if report is None else report["validation"]
        rows.append({
            "node_id": node["node_id"], "branch": node["branch"],
            "coordinate": node["coordinate"],
            "state": "PENDING" if report is None else "COMPLETE",
            "passes": None if report is None else report["passes"],
            "selected_pass": None if report is None else report["selected_pass"],
            "validation": metrics,
            "recovery_to_pure_offline": (
                None if metrics is None or baseline is None or oracle is None
                else recovery(metrics, baseline, oracle)
            ),
        })
    return rows


def _execution_gate(spec: dict, device) -> None:
    if str(device) not in {"cuda", "cuda:0"}:
        raise PermissionError("Proxy-ladder scientific fits require one GPU")
    profile = spec["runtime_profile"]
    _, cpus, memory_mb = allocation(profile["execution_site"])
    if cpus != profile["cpus"] or memory_mb != profile["memory_mb"]:
        raise ValueError("Proxy-ladder allocation differs from measured profile")
    if gpu_identity() != profile["gpu"] or installed_environment() != profile["installed_environment"]:
        raise ValueError("Proxy-ladder GPU/environment differs from measured profile")


def run_task(spec: dict, task_id: str, *, attempt: str, device="cuda") -> dict:
    validate_campaign(spec, check_source=True)
    if re.fullmatch(r"[A-Za-z0-9_-]+", attempt) is None:
        raise ValueError("Unsafe proxy-ladder attempt identity")
    tasks = {row["task_id"]: row for row in spec["tasks"]}
    if task_id not in tasks:
        raise ValueError("Unknown proxy-ladder production task")
    existing = completed_task(spec, task_id)
    if existing is not None:
        return existing
    task = tasks[task_id]
    for dependency in task["dependencies"]:
        if completed_task(spec, dependency) is None:
            raise ValueError(f"Missing authenticated proxy-ladder parent: {dependency}")
    root = Path(spec["campaign_root"])
    attempt_root = root / "attempts" / task_id / attempt
    attempt_root.mkdir(parents=True, exist_ok=False)
    outputs = []
    profile, foundation = spec["runtime_profile"], spec["foundation"]

    def save(name, payload):
        path = attempt_root / name
        write_json(path, payload)
        outputs.append(path)
        return path

    def cache(role, coordinate):
        return prepare_cache(
            foundation, foundation_root=Path(spec["foundation_root"]), role=role,
            coordinate=coordinate, workers=profile["workers"],
            max_ram_bytes=profile["cache_budgets"][role],
        )

    if task["kind"] in {"train", "reduce"}:
        _execution_gate(spec, device)
        node = next(row for row in spec["scientific_plan"]["nodes"] if row["node_id"] == task["node_id"])
        train = cache("train", node["coordinate"])
        validation_cache = cache("validation", node["coordinate"])
        torch.manual_seed(node["initialization_seed"])
        model = DelphesParticleTransformer()
        if task["kind"] == "train":
            kwargs = {}
            if node["teacher"] is not None:
                teacher = completed_task(spec, "reduce_" + node["teacher"])
                source = teacher["result"]
                probabilities = load_bank(
                    safe(root, source["train_bank"]),
                    foundation_sha256=foundation["content_hash"],
                    teacher_report_sha256=source["teacher_report_sha256"],
                    teacher_node=node["teacher"], role="train",
                    expected_identities=train.identities,
                )
                kwargs = {
                    "teacher_probabilities": probabilities,
                    "teacher_identities": train.identities,
                }
            training, state = train_kernel(
                model, train, validation_cache, node=node, device=device, **kwargs,
            )
            checkpoint = attempt_root / "selected.pt"
            buffer = BytesIO()
            torch.save(state, buffer)
            atomic_publish_bytes(checkpoint, buffer.getvalue())
            outputs.append(checkpoint)
            report_path = save("training_report.json", training)
            result = {
                "training_report": report_path.relative_to(root).as_posix(),
                "training_report_sha256": training["content_hash"],
                "checkpoint": checkpoint.relative_to(root).as_posix(),
            }
        else:
            teacher = completed_task(spec, "train_" + node["node_id"])["result"]
            model.load_state_dict(torch.load(
                safe(root, teacher["checkpoint"]), map_location="cpu", weights_only=True,
            ))
            model.to(device).eval()
            result = {"teacher_report_sha256": teacher["training_report_sha256"]}
            for role, role_cache, temperature in (
                ("train", train, 2.), ("validation", validation_cache, 1.),
            ):
                bank_root = attempt_root / role
                probabilities = predict(model, role_cache, device=device, temperature=temperature)
                publish_bank(
                    bank_root, foundation_sha256=foundation["content_hash"],
                    teacher_report_sha256=teacher["training_report_sha256"],
                    teacher_node=node["node_id"], role=role,
                    identities=role_cache.identities, probabilities=probabilities,
                )
                outputs.extend(sorted(bank_root.iterdir()))
                result[role + "_bank"] = bank_root.relative_to(root).as_posix()
    elif task["kind"] == "aggregate":
        rows = result_rows(spec)
        if any(row["state"] != "COMPLETE" for row in rows):
            raise ValueError("Proxy-ladder aggregate has unfinished fits")
        result = artifact(
            "AGGREGATE", parents={"campaign": spec["content_hash"]}, rows=rows,
            recovery_reference={"zero": "M0HLT", "hundred": "OFFLINE"},
        )
        save("validation_aggregate.json", result)
    else:
        result = {
            "fresh_fit_count": spec["fresh_fit_count"],
            "reducer_count": spec["reducer_count"],
            "scientific_results_do_not_control_completion": True,
        }
    report = artifact(
        "TASK_REPORT", parents={"campaign": spec["content_hash"]},
        source_commit=spec["source_commit"], task_id=task_id, result=result,
        outputs=[file_ref(path, root=root) for path in outputs],
    )
    task_path = _task_path(spec, task_id)
    task_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(task_path, report)
    return report


__all__ = [
    "AUTHORIZATION", "completed_task", "create_campaign", "result_rows", "run_task",
    "validate_campaign",
]

"""Staged SPORC execution for the JetClass2 salience MT20 three-spine study."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np
import torch

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, load_json, sha256_file, write_immutable_json,
)
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
from hlt_classification.scouting.hcwdl_recovery import validate_submission_ledger

from .banks import load_bank, publish_bank
from .contracts import artifact, relative_file, validate
from .execution import (
    allocation, execution_site, gpu_identity, production_site, slurm_options,
)
from .inventory import verify_snapshot
from .model import DelphesParticleTransformer, installed_environment, model_contract
from .production import _source
from .reporting import recovery
from .runner import predict, train_kernel
from .salience_cache import prepare_cache
from .salience_foundation import (
    assignment_source, authenticate_preparation, validate_foundation_spec,
)
from .salience_mt20_campaign import build_campaign_plan
from .salience_mt20_probability import mix_probabilities
from .salience_views import SUPPORT_POLICY


AUTHORIZE = "AUTHORIZE JETCLASS2 500K SALIENCE MT20 THREE SPINE CAMPAIGN"
JOB_PREFIX = "jc2smt20"
GATE_TASKS = ("authenticate", "audit_sources_and_storage", "preflight")
DEFAULT_SUBMISSION_SITE = "sporc_a100_debug"
ALTERNATE_SUBMISSION_SITE = "sporc_a100"
SITE_ROUTING_POLICY = "sporc_debug_default_explicit_tier3_task_override_v1"
TERMINAL = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED",
}


@dataclass(frozen=True)
class Resource:
    cpus: int
    memory_mb: int
    minutes: int
    gpu: bool


RESOURCES = {
    "metadata": Resource(1, 8192, 60, False),
    "preflight": Resource(8, 73728, 360, True),
    "train": Resource(8, 73728, 808, True),
    "reduce": Resource(8, 73728, 43, True),
}


def source_lock(foundation_root: Path, screen_root: Path):
    """Authenticate the frozen winner without loading losing screen candidates."""
    foundation_root = Path(foundation_root).resolve()
    screen_root = Path(screen_root).resolve()
    screen = load_json(screen_root / "screen_spec.json")
    validate(screen, "SALIENCE_SCREEN_SPEC")
    selection = load_json(screen_root / "selection_lock.json")
    validate(selection, "SALIENCE_SELECTION_LOCK")
    complete = load_json(screen_root / "screen_complete.json")
    validate(complete, "SALIENCE_SCREEN_COMPLETE")
    profile = load_json(screen_root / "runtime_profile.json")
    validate(profile, "SALIENCE_RUNTIME_PROFILE")
    if (
        Path(screen["screen_root"]).resolve() != screen_root
        or selection["screen_sha256"] != screen["content_hash"]
        or complete["screen_sha256"] != screen["content_hash"]
        or complete["selection_lock_sha256"] != selection["content_hash"]
        or profile["screen_sha256"] != screen["content_hash"]
        or selection["winner"] != "SALIENCE_PT_LINEAR"
        or "SALIENCE_PT_LINEAR" not in screen["candidate_registry"]
        or "SALIENCE_PT_LINEAR" not in selection["candidates"]
        or complete["scientific_fit_count"] != 4
        or profile["passed"] is not True
        or screen["final_test_accessed"] is not False
        or selection["final_test_accessed"] is not False
        or complete["final_test_accessed"] is not False
        or profile["final_test_accessed"] is not False
    ):
        raise ValueError("Selected salience screen evidence differs")
    foundation = load_json(foundation_root / "foundation_spec.json")
    foundation_hash = validate_foundation_spec(foundation)
    lock = authenticate_preparation(foundation, foundation_root)
    if (
        Path(selection["winner_foundation_root"]).resolve() != foundation_root
        or selection["winner_foundation_sha256"] != foundation_hash
        or selection["winner"] != foundation["candidate"]
        or lock["foundation_sha256"] != foundation_hash
        or lock["matcher_sha256"] != foundation["matcher"]["content_hash"]
        or foundation["views"]["support"] != SUPPORT_POLICY
        or foundation["candidate"] != "SALIENCE_PT_LINEAR"
    ):
        raise ValueError("Selected salience MT20 source differs")
    producer = assignment_source()
    value = artifact(
        "SALIENCE_MT20_SOURCE_LOCK",
        reuse_policy="selected_winner_foundation_read_only_v1",
        matching_recomputed=False, matcher_selection_repeated=False,
        screen_root=str(screen_root),
        screen_spec_path=str((screen_root / "screen_spec.json").resolve()),
        screen_sha256=screen["content_hash"],
        selection_lock_sha256=selection["content_hash"],
        screen_complete_sha256=complete["content_hash"],
        runtime_profile_sha256=profile["content_hash"],
        foundation_spec_path=str((foundation_root / "foundation_spec.json").resolve()),
        foundation_root=str(foundation_root.resolve()),
        foundation_sha256=foundation_hash,
        foundation_lock_sha256=lock["content_hash"],
        matcher_sha256=foundation["matcher"]["content_hash"],
        inventory_sha256=foundation["inventory"]["content_hash"],
        inputs_sha256=foundation["inputs"]["content_hash"],
        view_contract_sha256=foundation["views"]["content_hash"],
        support_policy=SUPPORT_POLICY,
        assignment_producer_sha256=producer["content_hash"],
        selected_candidate=foundation["candidate"],
        split_profile=foundation["splits"]["profile"],
        split_sha256=foundation["splits"]["content_hash"],
        split_registry_sha256=foundation["splits"]["registry_sha256"],
        role_membership_sha256={
            role: value["content_hash"]
            for role, value in foundation["splits"]["memberships"].items()
        },
        role_counts=foundation["splits"]["role_counts"],
        final_test_accessed=False,
    )
    return value, foundation, profile


def task_graph(plan: dict) -> list[dict]:
    rows = [
        dict(task_id="authenticate", kind="authenticate", dependencies=[], resource="metadata"),
        dict(task_id="audit_sources_and_storage", kind="storage",
             dependencies=["authenticate"], resource="metadata"),
        dict(task_id="preflight", kind="preflight",
             dependencies=["audit_sources_and_storage"], resource="preflight"),
    ]
    for node in plan["nodes"]:
        dependencies = ["preflight"] if not node["teachers"] else [
            "reduce_" + teacher["node_id"] for teacher in node["teachers"]
        ]
        rows.append(dict(task_id="train_" + node["node_id"], kind="train",
                         node_id=node["node_id"], dependencies=dependencies,
                         resource="train"))
        if node["node_id"] in plan["probability_publications"]:
            rows.append(dict(task_id="reduce_" + node["node_id"], kind="reduce",
                             node_id=node["node_id"],
                             dependencies=["train_" + node["node_id"]],
                             resource="reduce"))
    train_tasks = [row["task_id"] for row in rows if row["kind"] == "train"]
    rows.append(dict(task_id="aggregate", kind="aggregate",
                     dependencies=train_tasks, resource="metadata"))
    rows.append(dict(task_id="campaign_complete", kind="complete",
                     dependencies=["aggregate"], resource="metadata"))
    if len(rows) != 33 or len({row["task_id"] for row in rows}) != 33:
        raise AssertionError("Salience MT20 task census differs")
    return rows


def create_campaign(*, foundation_root: Path, screen_root: Path, data_root: Path,
                    campaign_root: Path, project: Path, source_commit: str) -> dict:
    _source(project, source_commit)
    source, foundation, profile = source_lock(foundation_root, screen_root)
    screen = load_json(Path(screen_root) / "screen_spec.json")
    if Path(screen["data_root"]).resolve() != Path(data_root).resolve():
        raise ValueError("Salience MT20 data root differs from selecting screen")
    if source["split_profile"] != "TRAIN_500K" or source["role_counts"] != {
        "train": 500_000, "validation": 1_000_000, "final_test": 1_000_000,
    }:
        raise ValueError("Salience MT20 population is not exact TRAIN_500K")
    verify_snapshot(Path(data_root), foundation["inventory"])
    root = Path(campaign_root).resolve()
    foundation_root = Path(source["foundation_root"])
    if (
        root.exists() or root.is_relative_to(Path(data_root).resolve())
        or root.is_relative_to(foundation_root)
        or root.is_relative_to(Path(screen_root).resolve())
    ):
        raise FileExistsError("Salience MT20 campaign needs a fresh isolated root")
    plan = build_campaign_plan(foundation)
    tasks = task_graph(plan)
    spec = artifact(
        "SALIENCE_MT20_CAMPAIGN_SPEC",
        source_commit=source_commit,
        project_dir=str(Path(project).resolve()), data_root=str(Path(data_root).resolve()),
        campaign_root=str(root),
        foundation_root=str(Path(foundation_root).resolve()),
        screen_root=str(Path(screen_root).resolve()),
        source_lock=source, foundation=foundation, runtime_profile=profile,
        scientific_plan=plan, tasks=tasks, model=model_contract(),
        resources={name: asdict(value) for name, value in RESOURCES.items()},
        fresh_fit_count=16, reducer_count=12, science_task_count=30,
        task_count=33, gate_tasks=list(GATE_TASKS),
        internal_ablations=["DIRECT", "COARSE", "DENSE"],
        teacher_policy_ablation_included=False,
        submission_routing={
            "policy": SITE_ROUTING_POLICY,
            "default_site": DEFAULT_SUBMISSION_SITE,
            "alternate_site": ALTERNATE_SUBMISSION_SITE,
            "debug_max_minutes": 24 * 60,
            "scientific_identity_affected": False,
        },
        ordinary_roles=["train", "validation"], ordinary_final_test_capability=False,
        ram_only_particle_views=True, ram_only_weighted_teacher_mixtures=True,
        durable_single_model_probability_banks_only=True, rolling_resume=False,
        projected_probability_payload_bytes=(500_000 * 11 * 4 * 12),
        projected_bank_payload_bytes=(500_000 * (11 * 4 + 32) * 12),
        projected_durable_bytes_upper_bound=4 * 1024**3,
        minimum_free_disk_bytes=16 * 1024**3,
        existing_campaign_dependencies=[], existing_campaign_mutations=False,
        final_test_accessed=False,
    )
    root.mkdir(parents=True, exist_ok=False)
    for name, value in (
        ("source_lock.json", source), ("scientific_plan.json", plan),
        ("campaign_spec.json", spec),
    ):
        write_immutable_json(root / name, value)
    return spec


def _validate_campaign_fields(
    spec: dict, *, source: dict, foundation: dict, profile: dict, plan: dict,
) -> None:
    if (
        spec["source_lock"] != source or spec["foundation"] != foundation
        or spec["runtime_profile"] != profile or spec["scientific_plan"] != plan
        or spec["tasks"] != task_graph(plan) or spec["model"] != model_contract()
        or spec["resources"] != {name: asdict(value) for name, value in RESOURCES.items()}
        or (spec["fresh_fit_count"], spec["reducer_count"], spec["science_task_count"], spec["task_count"])
        != (16, 12, 30, 33)
        or spec["internal_ablations"] != ["DIRECT", "COARSE", "DENSE"]
        or spec["teacher_policy_ablation_included"] is not False
        or spec["submission_routing"] != {
            "policy": SITE_ROUTING_POLICY,
            "default_site": DEFAULT_SUBMISSION_SITE,
            "alternate_site": ALTERNATE_SUBMISSION_SITE,
            "debug_max_minutes": 24 * 60,
            "scientific_identity_affected": False,
        }
        or spec["ordinary_final_test_capability"] is not False
        or spec["ram_only_particle_views"] is not True
        or spec["ram_only_weighted_teacher_mixtures"] is not True
        or spec["durable_single_model_probability_banks_only"] is not True
        or spec["rolling_resume"] is not False
        or spec["projected_probability_payload_bytes"] != 500_000 * 11 * 4 * 12
        or spec["projected_bank_payload_bytes"] != 500_000 * (11 * 4 + 32) * 12
        or spec["projected_durable_bytes_upper_bound"] != 4 * 1024**3
        or spec["minimum_free_disk_bytes"] != 16 * 1024**3
        or spec["existing_campaign_dependencies"] != []
        or spec["existing_campaign_mutations"] is not False
        or spec["final_test_accessed"] is not False
    ):
        raise ValueError("Salience MT20 campaign contract differs")


def _recorded_assignment_source_lock(current: dict, recorded: dict) -> dict:
    """Authenticate an old source lock when only its producer HEAD differs.

    The assignment producer records both the relevant file hashes and the Git
    HEAD.  A downstream study at a newer commit must not rewrite that immutable
    historical HEAD.  Foundation authentication independently checks every
    assignment shard against the current producer *file hashes*, so replacing
    only the current producer identity with the recorded one is both necessary
    and fail-closed.
    """
    validate(current, "SALIENCE_MT20_SOURCE_LOCK")
    validate(recorded, "SALIENCE_MT20_SOURCE_LOCK")
    fields = {
        key: value for key, value in current.items()
        if key not in {"contract", "schema_version", "content_hash"}
    }
    fields["assignment_producer_sha256"] = recorded["assignment_producer_sha256"]
    normalized = artifact("SALIENCE_MT20_SOURCE_LOCK", **fields)
    if normalized != recorded:
        raise ValueError("Historical salience MT20 source lock semantics differ")
    return recorded


def validate_campaign_snapshot(spec: dict) -> str:
    """Validate a completed campaign from a later byte-compatible commit.

    Unlike executable validation, this preserves the recorded producer HEAD.
    Every scientific field, immutable child, foundation artifact, and current
    producer file hash is still authenticated.
    """
    digest = validate(spec, "SALIENCE_MT20_CAMPAIGN_SPEC")
    current, foundation, profile = source_lock(
        Path(spec["foundation_root"]), Path(spec["screen_root"]),
    )
    recorded = _recorded_assignment_source_lock(current, spec["source_lock"])
    plan = build_campaign_plan(foundation)
    _validate_campaign_fields(
        spec, source=recorded, foundation=foundation, profile=profile, plan=plan,
    )
    root = Path(spec["campaign_root"])
    for name, expected in (("source_lock.json", recorded), ("scientific_plan.json", plan)):
        if load_json(root / name) != expected:
            raise ValueError("Salience MT20 immutable child differs")
    return digest


def validate_campaign(spec: dict, *, check_source=True) -> str:
    digest = validate(spec, "SALIENCE_MT20_CAMPAIGN_SPEC")
    source, foundation, profile = source_lock(
        Path(spec["foundation_root"]), Path(spec["screen_root"]),
    )
    plan = build_campaign_plan(foundation)
    _validate_campaign_fields(
        spec, source=source, foundation=foundation, profile=profile, plan=plan,
    )
    root = Path(spec["campaign_root"])
    for name, expected in (("source_lock.json", source), ("scientific_plan.json", plan)):
        if load_json(root / name) != expected:
            raise ValueError("Salience MT20 immutable child differs")
    if check_source:
        _source(Path(spec["project_dir"]), spec["source_commit"])
    return digest


def _task_path(spec, task_id):
    if task_id not in {row["task_id"] for row in spec["tasks"]}:
        raise ValueError("Unknown salience MT20 task")
    return Path(spec["campaign_root"]) / "tasks" / f"{task_id}.json"


def completed_task(spec, task_id):
    path = _task_path(spec, task_id)
    if not path.is_file():
        return None
    report = load_json(path); validate(report, "SALIENCE_MT20_TASK_REPORT")
    if (
        report["campaign_sha256"] != spec["content_hash"]
        or report["source_commit"] != spec["source_commit"]
        or report["task_id"] != task_id or report["final_test_accessed"] is not False
    ):
        raise ValueError("Salience MT20 task report lineage differs")
    root = Path(spec["campaign_root"])
    if not report["outputs"] or len(report["outputs"]) != len({row["path"] for row in report["outputs"]}):
        raise ValueError("Salience MT20 task output inventory differs")
    for output in report["outputs"]:
        path = relative_file(root, output["path"])
        if not path.is_file() or sha256_file(path) != output["sha256"]:
            raise ValueError("Salience MT20 task output checksum differs")
    return report


def _execution_gate(spec, device):
    profile = spec["runtime_profile"]
    if str(device) not in {"cuda", "cuda:0"}:
        raise PermissionError("Salience MT20 science requires one GPU")
    site_name = os.environ.get("JC2_SITE", "")
    if site_name not in {DEFAULT_SUBMISSION_SITE, ALTERNATE_SUBMISSION_SITE}:
        raise PermissionError("Salience MT20 worker has no registered submission site")
    site = execution_site(site_name)
    if production_site(site) != profile["execution_site"]:
        raise PermissionError("Salience MT20 debug/tier3 profile transfer differs")
    _, cpus, memory = allocation(site)
    if (
        cpus != profile["cpus"] or memory != profile["memory_mb"]
        or gpu_identity() != profile["gpu"]
        or installed_environment() != profile["installed_environment"]
    ):
        raise ValueError("Worker differs from selected salience SPORC profile")


def _cache(spec, role, coordinate_name):
    profile = spec["runtime_profile"]
    return prepare_cache(
        spec["foundation"], data_root=Path(spec["data_root"]),
        foundation_root=Path(spec["source_lock"]["foundation_root"]),
        role=role, coordinate_name=coordinate_name, workers=profile["workers"],
        max_ram_bytes=profile["cache_budgets"][role],
    )


def _teacher_mixture(spec, node, identities):
    root = Path(spec["campaign_root"])
    components, lineage = [], []
    for registered in node["teachers"]:
        teacher_id = registered["node_id"]
        pointer = completed_task(spec, "reduce_" + teacher_id)
        result = pointer["result"]
        values = load_bank(
            relative_file(root, result["train_bank"]),
            foundation_sha256=spec["foundation"]["content_hash"],
            teacher_report_sha256=result["teacher_report_sha256"],
            teacher_node=teacher_id, role="train", expected_identities=identities,
        )
        components.append(values)
        lineage.append(dict(
            node_id=teacher_id, loss_weight=registered["loss_weight"],
            reducer_task_sha256=pointer["content_hash"],
            teacher_report_sha256=result["teacher_report_sha256"],
            bank_manifest_sha256=result["manifest_sha256"],
        ))
    return mix_probabilities(components, node["teachers"]), lineage


def _save_state(path, state):
    buffer = BytesIO(); torch.save(state, buffer)
    atomic_publish_bytes(path, buffer.getvalue())


def _run_train(spec, task, attempt_root, device):
    node = next(row for row in spec["scientific_plan"]["nodes"]
                if row["node_id"] == task["node_id"])
    train = _cache(spec, "train", node["coordinate"])
    validation = _cache(spec, "validation", node["coordinate"])
    torch.manual_seed(node["initialization_seed"])
    model = DelphesParticleTransformer()
    kwargs, lineage = {}, []
    if node["teachers"]:
        mixture, lineage = _teacher_mixture(spec, node, train.identities)
        kwargs = dict(teacher_probabilities=mixture, teacher_identities=train.identities)
    training, state = train_kernel(
        model, train, validation, node=node, device=device,
        training_recipe=spec["scientific_plan"]["recipe"], **kwargs,
    )
    training = artifact(
        "SALIENCE_MT20_TRAINING_REPORT",
        kernel_training_report=training,
        campaign_sha256=spec["content_hash"], source_commit=spec["source_commit"],
        node=node, teacher_lineage=lineage,
        mixture_materialization="float64_accumulate_normalize_float32_ram_only",
        durable_mixture_written=False, final_test_accessed=False,
    )
    checkpoint = attempt_root / "selected.pt"; _save_state(checkpoint, state)
    report_path = attempt_root / "training_report.json"
    write_immutable_json(report_path, training)
    return dict(
        outputs=[checkpoint, report_path],
        result=dict(
            checkpoint=checkpoint, checkpoint_sha256=sha256_file(checkpoint),
            training_report=report_path,
            training_report_sha256=training["content_hash"],
        ),
    )


def _run_reduce(spec, task, attempt_root, device):
    node = next(row for row in spec["scientific_plan"]["nodes"]
                if row["node_id"] == task["node_id"])
    source = completed_task(spec, "train_" + node["node_id"])["result"]
    model = DelphesParticleTransformer()
    model.load_state_dict(torch.load(
        relative_file(Path(spec["campaign_root"]), source["checkpoint"]),
        map_location="cpu", weights_only=True,
    ), strict=True)
    model.to(device).eval()
    train = _cache(spec, "train", node["coordinate"])
    probabilities = predict(model, train, device=device, temperature=2.)
    bank_root = attempt_root / "train_bank"
    manifest = publish_bank(
        bank_root, foundation_sha256=spec["foundation"]["content_hash"],
        teacher_report_sha256=source["training_report_sha256"],
        teacher_node=node["node_id"], role="train", identities=train.identities,
        probabilities=probabilities,
    )
    return dict(
        outputs=sorted(bank_root.iterdir()),
        result=dict(
            train_bank=bank_root,
            teacher_report_sha256=source["training_report_sha256"],
            teacher_node=node["node_id"], manifest_sha256=manifest["content_hash"],
        ),
    )


def _run_storage(spec, attempt_root):
    root = Path(spec["campaign_root"])
    files = [path for path in root.rglob("*") if path.is_file()]
    if any(path.is_symlink() for path in files):
        raise ValueError("Salience MT20 campaign storage contains a symlink")
    free = shutil.disk_usage(root).free
    if free < spec["minimum_free_disk_bytes"]:
        raise OSError("Insufficient salience MT20 storage headroom")
    report = artifact(
        "SALIENCE_MT20_STORAGE_AUDIT", campaign_sha256=spec["content_hash"],
        free_bytes=free, existing_bytes=sum(path.stat().st_size for path in files),
        projected_probability_payload_bytes=spec["projected_probability_payload_bytes"],
        projected_bank_payload_bytes=spec["projected_bank_payload_bytes"],
        projected_durable_bytes_upper_bound=spec["projected_durable_bytes_upper_bound"],
        particle_views_durable=False, mixture_arrays_durable=False,
        rolling_resume=False, passed=True, final_test_accessed=False,
    )
    path = attempt_root / "storage_audit.json"; write_immutable_json(path, report)
    return dict(outputs=[path], result=dict(storage_audit_sha256=report["content_hash"]))


def _run_preflight(spec, attempt_root, device):
    _execution_gate(spec, device)
    started = time.monotonic()
    train = _cache(spec, "train", "U000")
    indexes = np.arange(min(256, len(train)))
    raw = train.batch(indexes)
    labels = torch.from_numpy(raw["labels"]).to(device)
    # Construct these outside inference_mode.  Tensors created by a device
    # transfer inside inference_mode retain PyTorch's inference-only marker;
    # reusing them for the train-mode student then prevents BatchNorm from
    # saving its input for backward.
    inputs = {
        name: torch.from_numpy(raw[name]).to(device)
        for name in ("features", "vectors", "mask")
    }
    torch.manual_seed(44001); teacher_a = DelphesParticleTransformer().to(device).eval()
    torch.manual_seed(44002); teacher_b = DelphesParticleTransformer().to(device).eval()
    torch.manual_seed(44003); student = DelphesParticleTransformer().to(device).train()
    with torch.inference_mode():
        q1 = torch.softmax(teacher_a(**inputs).float() / 2., dim=-1).cpu().numpy().astype(np.float32)
        q2 = torch.softmax(teacher_b(**inputs).float() / 2., dim=-1).cpu().numpy().astype(np.float32)
    registry = [
        {"node_id": "near", "loss_weight": {"numerator": 1, "denominator": 2}},
        {"node_id": "old", "loss_weight": {"numerator": 3, "denominator": 10}},
    ]
    mixture = mix_probabilities([q1, q2], registry)
    logits = student(**inputs)
    from .model import distillation_loss
    loss = distillation_loss(
        logits, labels, teacher_probabilities=torch.from_numpy(mixture).to(device),
        ce_weight=.20, kd_weight=.80, temperature=2.,
    )
    loss.backward()
    gradients = [parameter.grad for parameter in student.parameters()
                 if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise ValueError("Salience MT20 preflight gradients differ")
    report = artifact(
        "SALIENCE_MT20_PREFLIGHT",
        campaign_sha256=spec["content_hash"], source_commit=spec["source_commit"],
        rows=len(indexes), loss=float(loss.detach().cpu()),
        exact_c20p80=True, temperature=2., multi_teacher_count=2,
        mixture_float64_accumulation=True, mixture_float32_training=True,
        persistent_hlt_u000_exercised=True, model_backward_passed=True,
        runtime_seconds=time.monotonic() - started,
        final_test_accessed=False, passed=True,
    )
    path = attempt_root / "preflight.json"; write_immutable_json(path, report)
    return dict(outputs=[path], result=dict(preflight_sha256=report["content_hash"]))


def _publish_task(spec, task_id, payload):
    root = Path(spec["campaign_root"]).resolve()
    outputs = list(dict.fromkeys(Path(value).resolve() for value in payload["outputs"]))
    if any(not path.is_file() or not path.is_relative_to(root) for path in outputs):
        raise ValueError("Salience MT20 task output escaped or is incomplete")
    result = {
        name: value.relative_to(root).as_posix() if isinstance(value, Path) else value
        for name, value in payload["result"].items()
    }
    report = artifact(
        "SALIENCE_MT20_TASK_REPORT", campaign_sha256=spec["content_hash"],
        source_commit=spec["source_commit"], task_id=task_id, result=result,
        outputs=[dict(path=path.relative_to(root).as_posix(), sha256=sha256_file(path))
                 for path in outputs],
        final_test_accessed=False,
    )
    write_immutable_json(_task_path(spec, task_id), report)
    return report


def result_rows(spec):
    validate_campaign(spec)
    reports = {}
    for node in spec["scientific_plan"]["nodes"]:
        pointer = completed_task(spec, "train_" + node["node_id"])
        if pointer is None:
            continue
        report = load_json(relative_file(
            Path(spec["campaign_root"]), pointer["result"]["training_report"],
        ))
        validate(report, "SALIENCE_MT20_TRAINING_REPORT")
        if report["campaign_sha256"] != spec["content_hash"] or report["node"] != node:
            raise ValueError("Salience MT20 training report lineage differs")
        reports[node["node_id"]] = report
    baseline = reports.get("M0HLT", {}).get("kernel_training_report", {}).get("validation")
    oracle = reports.get("U000", {}).get("kernel_training_report", {}).get("validation")
    rows = []
    for node in spec["scientific_plan"]["nodes"]:
        report = reports.get(node["node_id"])
        kernel = None if report is None else report["kernel_training_report"]
        metrics = None if kernel is None else kernel["validation"]
        rows.append(dict(
            node_id=node["node_id"], branch=node["branch"], coordinate=node["coordinate"],
            teacher_count=len(node["teachers"]),
            state="PENDING" if report is None else "COMPLETE",
            selected_pass=None if kernel is None else kernel["selected_pass"],
            passes=None if kernel is None else kernel["passes"], validation=metrics,
            recovery=None if metrics is None or baseline is None or oracle is None
            else recovery(metrics, baseline, oracle),
        ))
    return rows


def validate_science_gate(spec):
    for task_id in GATE_TASKS:
        if completed_task(spec, task_id) is None:
            raise PermissionError("Salience MT20 science gate is incomplete: " + task_id)
    preflight = completed_task(spec, "preflight")
    report_path = next(row["path"] for row in preflight["outputs"]
                       if row["path"].endswith("preflight.json"))
    report = load_json(relative_file(Path(spec["campaign_root"]), report_path))
    validate(report, "SALIENCE_MT20_PREFLIGHT")
    if not report["passed"] or report["campaign_sha256"] != spec["content_hash"]:
        raise ValueError("Salience MT20 preflight did not certify this campaign")
    return report["content_hash"]


def run_task(spec, task_id, *, attempt, device="cuda"):
    validate_campaign(spec)
    if re.fullmatch(r"[A-Za-z0-9_-]+", attempt) is None:
        raise ValueError("Unsafe salience MT20 attempt identity")
    registry = {row["task_id"]: row for row in spec["tasks"]}
    if task_id not in registry:
        raise ValueError("Unknown salience MT20 task")
    if (existing := completed_task(spec, task_id)) is not None:
        return existing
    task = registry[task_id]
    for dependency in task["dependencies"]:
        if completed_task(spec, dependency) is None:
            raise ValueError("Missing authenticated parent: " + dependency)
    attempt_root = Path(spec["campaign_root"]) / "attempts" / task_id / attempt
    attempt_root.mkdir(parents=True, exist_ok=False)
    kind = task["kind"]
    if kind == "authenticate":
        validate_campaign(spec)
        report = artifact(
            "SALIENCE_MT20_AUTHENTICATION", campaign_sha256=spec["content_hash"],
            source_commit=spec["source_commit"], passed=True, final_test_accessed=False,
        )
        path = attempt_root / "authentication.json"; write_immutable_json(path, report)
        payload = dict(outputs=[path], result=dict(authentication_sha256=report["content_hash"]))
    elif kind == "storage":
        payload = _run_storage(spec, attempt_root)
    elif kind == "preflight":
        payload = _run_preflight(spec, attempt_root, device)
    elif kind == "train":
        _execution_gate(spec, device); payload = _run_train(spec, task, attempt_root, device)
    elif kind == "reduce":
        _execution_gate(spec, device); payload = _run_reduce(spec, task, attempt_root, device)
    elif kind == "aggregate":
        rows = result_rows(spec)
        if any(row["state"] != "COMPLETE" for row in rows):
            raise ValueError("Aggregate has unfinished salience MT20 fits")
        report = artifact(
            "SALIENCE_MT20_AGGREGATE", campaign_sha256=spec["content_hash"],
            rows=rows, fresh_fit_count=16,
            scientific_result_does_not_control_completion=True,
            final_test_accessed=False,
        )
        path = attempt_root / "validation_aggregate.json"; write_immutable_json(path, report)
        payload = dict(outputs=[path], result=dict(aggregate=path, aggregate_sha256=report["content_hash"]))
    else:
        report = artifact(
            "SALIENCE_MT20_CAMPAIGN_COMPLETE", campaign_sha256=spec["content_hash"],
            aggregate_task_sha256=completed_task(spec, "aggregate")["content_hash"],
            fresh_fit_count=16, reducer_count=12,
            scientific_result_does_not_control_completion=True,
            final_test_accessed=False,
        )
        path = attempt_root / "campaign_complete.json"; write_immutable_json(path, report)
        payload = dict(outputs=[path], result=dict(campaign_complete=path, complete_sha256=report["content_hash"]))
    return _publish_task(spec, task_id, payload)


def command_plan(spec, *, stage="full", selected_tasks=None, tier3_tasks=None):
    validate_campaign(spec)
    if stage not in {"full", "gate", "science"}:
        raise ValueError("Salience MT20 submission stage differs")
    known = {row["task_id"]: row for row in spec["tasks"]}
    chosen = (
        list(known) if stage == "full" else list(GATE_TASKS) if stage == "gate"
        else [name for name in known if name not in GATE_TASKS]
    )
    if selected_tasks is not None:
        chosen = list(selected_tasks)
    if len(chosen) != len(set(chosen)) or not set(chosen) <= set(known):
        raise ValueError("Salience MT20 command coverage differs")
    tier3_tasks = [] if tier3_tasks is None else list(tier3_tasks)
    if (
        len(tier3_tasks) != len(set(tier3_tasks))
        or not set(tier3_tasks) <= set(chosen)
    ):
        raise ValueError("Salience MT20 tier3 task overrides differ")
    if stage == "science":
        validate_science_gate(spec)
    rows = []
    for task in spec["tasks"]:
        task_id = task["task_id"]
        if task_id not in chosen:
            continue
        dependencies = [name for name in task["dependencies"] if name in chosen]
        for parent in task["dependencies"]:
            if parent not in chosen and completed_task(spec, parent) is None:
                raise ValueError("Omitted salience MT20 parent is incomplete")
        resource = RESOURCES[task["resource"]]
        site_name = (
            ALTERNATE_SUBMISSION_SITE
            if task_id in tier3_tasks else DEFAULT_SUBMISSION_SITE
        )
        site = execution_site(site_name)
        if resource.minutes > 24 * 60 and site_name == DEFAULT_SUBMISSION_SITE:
            raise ValueError("Salience MT20 debug task exceeds the 24-hour ceiling")
        command = slurm_options(site) + [
            f"--cpus-per-task={resource.cpus}", f"--mem={resource.memory_mb}M",
            f"--time={resource.minutes}", f"--job-name={JOB_PREFIX}_{task_id}",
            "--chdir=" + spec["project_dir"],
            "--output=" + str(Path(spec["campaign_root"]) / "slurm-%j.out"),
        ]
        if resource.gpu:
            command += ["--gres=" + spec["runtime_profile"]["execution_site"]["gres"]]
        if dependencies:
            command += ["--dependency=afterok:" + ":".join(
                "${JOB_" + parent + "}" for parent in dependencies
            )]
        command += [
            str(Path(spec["project_dir"]) / "sbatch/run_jetclass2_delphes_salience_mt20.sh"),
            spec["project_dir"], str(Path(spec["campaign_root"]) / "campaign_spec.json"),
            task_id, site_name,
        ]
        rows.append(dict(
            task_id=task_id, dependencies=dependencies, execution_site=site_name,
            command=command,
        ))
    return artifact(
        "COMMAND_PLAN", campaign_sha256=spec["content_hash"], stage=stage,
        commands=rows, single_gpu=True, restart_from_zero=True,
        routing_policy=SITE_ROUTING_POLICY,
        default_site=DEFAULT_SUBMISSION_SITE,
        tier3_tasks=sorted(tier3_tasks),
        final_test_accessed=False,
    )


def submit(spec, *, stage, execute, authorization_phrase=None,
           selected_tasks=None, bookkeeping_root=None, tier3_tasks=None):
    validate_campaign(spec)
    campaign = Path(spec["campaign_root"]).resolve()
    root = campaign / f"submissions_{stage}" if bookkeeping_root is None else Path(bookkeeping_root).resolve()
    if not root.is_relative_to(campaign):
        raise ValueError("Salience MT20 bookkeeping escaped campaign")
    canonical = campaign / f"submissions_{stage}"
    if root != canonical:
        recovery_report = load_json(root / "recovery.json")
        validate(recovery_report, "SALIENCE_MT20_RECOVERY")
        if (
            recovery_report["campaign_sha256"] != spec["content_hash"]
            or recovery_report["source_commit"] != spec["source_commit"]
            or recovery_report["stage"] != stage
        ):
            raise ValueError("Salience MT20 recovery subject differs")
        selected_tasks = recovery_report["remaining_tasks"]
    plan = command_plan(
        spec, stage=stage, selected_tasks=selected_tasks,
        tier3_tasks=tier3_tasks,
    )
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "command_plan.json"
    if not plan_path.exists():
        write_immutable_json(plan_path, plan)
    elif load_json(plan_path) != plan:
        raise ValueError("Immutable salience MT20 command plan differs")
    dry = root / "dry_run_submission_ledger.json"
    claim = root / "submission_in_progress.claim"
    descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600); os.close(descriptor)
    try:
        if not execute:
            return submit_exact_dag(
                identity=spec["content_hash"], plan=plan, output=dry,
                canonical_dry_run=dry, execute=False,
            )
        if authorization_phrase != AUTHORIZE:
            raise PermissionError("Exact salience MT20 authorization phrase required")
        if stage == "full":
            raise PermissionError("Live full submission is forbidden; gate then science")
        if not dry.is_file():
            raise ValueError("Exact salience MT20 dry run is required")
        from .submission import _guarded_exact_submission
        return _guarded_exact_submission(spec, plan, root)
    finally:
        claim.unlink()


def monitor(spec, ledger):
    validate_submission_ledger(ledger)
    if ledger["campaign_spec_sha256"] != spec["content_hash"] or ledger["dry_run"]:
        raise ValueError("Monitor requires this salience MT20 live ledger")
    raw = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-j", ",".join(ledger["jobs"].values()),
         "--format=JobID,State%40"], check=True, capture_output=True, text=True,
    ).stdout
    states = {}
    for line in raw.splitlines():
        fields = [value.strip() for value in line.split("|")]
        if len(fields) >= 2:
            states[fields[0]] = fields[1].split()[0].rstrip("+")
    return artifact(
        "SALIENCE_MT20_MONITOR", campaign_sha256=spec["content_hash"],
        rows=[dict(task_id=task_id, job_id=job_id, state=states.get(job_id, "UNKNOWN"),
                   outputs_complete=completed_task(spec, task_id) is not None)
              for task_id, job_id in ledger["jobs"].items()],
        final_test_accessed=False,
    )


def prepare_recovery(spec, ledger, *, output_root, tier3_tasks=None):
    observed = monitor(spec, ledger)
    if any(row["state"] not in TERMINAL for row in observed["rows"]):
        raise PermissionError("Recovery requires every old job terminal")
    submitted = set(ledger["jobs"])
    stage = "gate" if submitted <= set(GATE_TASKS) else "science"
    if stage == "science" and not submitted.isdisjoint(GATE_TASKS):
        raise ValueError("Mixed-stage salience MT20 ledger cannot be recovered")
    remaining = [task_id for task_id in ledger["jobs"] if completed_task(spec, task_id) is None]
    root = Path(output_root).resolve()
    if root.exists() or not root.is_relative_to(Path(spec["campaign_root"]).resolve()) or not remaining:
        raise ValueError("Invalid salience MT20 recovery root/tasks")
    report = artifact(
        "SALIENCE_MT20_RECOVERY", campaign_sha256=spec["content_hash"],
        source_commit=spec["source_commit"], stage=stage,
        remaining_tasks=remaining, restart_from_zero=True,
        completed_parents_reused=True, final_test_accessed=False,
    )
    write_immutable_json(root / "recovery.json", report)
    write_immutable_json(root / "command_plan.json",
                         command_plan(
                             spec, stage=stage, selected_tasks=remaining,
                             tier3_tasks=tier3_tasks,
                         ))
    return report


__all__ = [
    "AUTHORIZE", "GATE_TASKS", "JOB_PREFIX", "RESOURCES", "command_plan",
    "completed_task", "create_campaign", "monitor", "prepare_recovery",
    "result_rows", "run_task", "source_lock", "submit", "task_graph",
    "validate_campaign", "validate_campaign_snapshot", "validate_science_gate",
]

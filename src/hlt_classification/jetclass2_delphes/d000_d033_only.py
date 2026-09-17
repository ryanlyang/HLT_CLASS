"""Standalone matched D000 <- D033-only C25/P75 endpoint ablation."""
from __future__ import annotations

from io import BytesIO
import gc
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, load_json, sha256_file, write_immutable_json,
)
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
from hlt_classification.scouting.hcwdl_recovery import build_submission_event

from .banks import load_bank
from .campaign import recipe
from .contracts import artifact as base_artifact, relative_file, validate as base_validate
from .execution import execution_site, slurm_options
from .model import DelphesParticleTransformer, distillation_loss, model_contract
from .production import _source
from .reporting import recovery
from .runner import train_kernel
from .salience_mt20_production import (
    _cache, _execution_gate, completed_task, validate_campaign_snapshot,
)
from .submission import _guarded_exact_submission


AUTHORIZE = "AUTHORIZE JETCLASS2 D000 D033 ONLY C25P75 ABLATION"
NODE_ID = "JC2_D000_D033_ONLY_C25P75"
TEACHER_ID = "JC2SMT20_COARSE_D033_from_D066"
MT20_ENDPOINT_ID = "JC2SMT20_COARSE_D000_from_D033"
REFERENCE_IDS = ("M0HLT", "U000", TEACHER_ID, MT20_ENDPOINT_ID)
SITE_NAMES = {False: "sporc_a100_debug", True: "sporc_a100"}
RESOURCES = dict(cpus=8, memory_mb=73728, minutes=808)


def artifact(kind: str, **fields) -> dict:
    return base_artifact("D000_D033_ONLY_" + kind, **fields)


def validate(value: dict, kind: str) -> str:
    return base_validate(value, "D000_D033_ONLY_" + kind)


def _training_report(source: dict, node_id: str) -> tuple[dict, dict]:
    pointer = completed_task(source, "train_" + node_id)
    if pointer is None:
        raise PermissionError(f"Required source fit is incomplete: {node_id}")
    result = pointer["result"]
    report = load_json(relative_file(
        Path(source["campaign_root"]), result["training_report"],
    ))
    base_validate(report, "SALIENCE_MT20_TRAINING_REPORT")
    nodes = {row["node_id"]: row for row in source["scientific_plan"]["nodes"]}
    if (
        node_id not in nodes
        or report["campaign_sha256"] != source["content_hash"]
        or report["source_commit"] != source["source_commit"]
        or report["node"] != nodes[node_id]
        or report["kernel_training_report"]["scientific_fit"] is not True
        or report["kernel_training_report"]["final_test_accessed"] is not False
        or report["final_test_accessed"] is not False
        or result["training_report_sha256"] != report["content_hash"]
    ):
        raise ValueError(f"Source training report lineage differs: {node_id}")
    return pointer, report


def source_evidence(path: Path) -> tuple[dict, dict]:
    source = load_json(path)
    validate_campaign_snapshot(source)
    if (
        source["scientific_plan"]["split_profile"] != "TRAIN_500K"
        or source["scientific_plan"]["role_counts"] != {
            "train": 500_000, "validation": 1_000_000,
            "final_test": 1_000_000,
        }
        or source["ordinary_final_test_capability"] is not False
        or source["final_test_accessed"] is not False
    ):
        raise ValueError("Endpoint ablation requires the exact sealed MT20 500k source")
    references = []
    reports = {}
    validation_rows = source["scientific_plan"]["role_counts"]["validation"]
    for node_id in REFERENCE_IDS:
        pointer, report = _training_report(source, node_id)
        reports[node_id] = report
        kernel = report["kernel_training_report"]
        if kernel["validation"].get("rows") != validation_rows:
            raise ValueError(f"Source validation coverage differs: {node_id}")
        references.append(dict(
            node_id=node_id, task_sha256=pointer["content_hash"],
            training_report_sha256=report["content_hash"],
            validation=kernel["validation"], passes=kernel["passes"],
            selected_pass=kernel["selected_pass"],
        ))
    reducer = completed_task(source, "reduce_" + TEACHER_ID)
    if reducer is None:
        raise PermissionError("Required D033 T=2 probability bank is incomplete")
    result = reducer["result"]
    if (
        result["teacher_node"] != TEACHER_ID
        or result["teacher_report_sha256"] != reports[TEACHER_ID]["content_hash"]
    ):
        raise ValueError("D033 probability-bank lineage differs")
    evidence = dict(
        source_campaign_sha256=source["content_hash"],
        source_commit=source["source_commit"], references=references,
        reducer_task_sha256=reducer["content_hash"],
        teacher_report_sha256=result["teacher_report_sha256"],
        teacher_bank_manifest_sha256=result["manifest_sha256"],
        teacher_bank_path=result["train_bank"],
        foundation_sha256=source["foundation"]["content_hash"],
        final_test_accessed=False,
    )
    return source, evidence


def node_spec(source: dict) -> dict:
    original = next(
        row for row in source["scientific_plan"]["nodes"]
        if row["node_id"] == MT20_ENDPOINT_ID
    )
    return dict(
        node_id=NODE_ID, coordinate="D000", teacher=TEACHER_ID,
        teachers=[dict(
            node_id=TEACHER_ID,
            loss_weight={"numerator": 3, "denominator": 4},
        )],
        branch="D033_ONLY_C25P75", u=list(original["u"]), f=list(original["f"]),
        initialization_seed=original["initialization_seed"],
        sampler_seed=original["sampler_seed"], deployable=True,
    )


def study_recipe() -> dict:
    value = recipe()
    if value["ce_weight"] != .25 or value["kd_weight"] != .75 or value["temperature"] != 2.:
        raise ValueError("Registered base recipe is not exact C25/P75 T=2")
    return value


def _isolated(root: Path, source: dict, project: Path) -> Path:
    root = Path(root).resolve()
    protected = (
        Path(source["campaign_root"]), Path(source["data_root"]),
        Path(source["source_lock"]["foundation_root"]),
        Path(source["project_dir"]), Path(project),
    )
    if any(
        root == path.resolve() or root.is_relative_to(path.resolve())
        or path.resolve().is_relative_to(root)
        for path in protected
    ):
        raise ValueError("Endpoint ablation root overlaps a protected source directory")
    return root


def create(*, source_spec: Path, output_root: Path, project: Path,
           source_commit: str, tier3: bool = False) -> dict:
    project = Path(project).resolve()
    _source(project, source_commit)
    source, evidence = source_evidence(source_spec)
    root = _isolated(output_root, source, project)
    if root.exists():
        raise FileExistsError("Endpoint ablation requires a fresh campaign root")
    site = execution_site(SITE_NAMES[bool(tier3)])
    spec = artifact(
        "SPEC", source_commit=source_commit, project_dir=str(project),
        campaign_root=str(root), source_spec_path=str(Path(source_spec).resolve()),
        source=evidence, foundation=source["foundation"],
        source_lock=source["source_lock"], data_root=source["data_root"],
        runtime_profile=source["runtime_profile"], model=model_contract(),
        node=node_spec(source), recipe=study_recipe(), site=site,
        resources=dict(RESOURCES), fresh_fit_count=1, slurm_job_count=1,
        teacher_policy="single_immediate_D033_only",
        historical_teachers_included=False, ce_weight=.25, kd_weight=.75,
        temperature=2., initialization_and_sampler_paired_to_mt20_D000=True,
        particle_views_ram_only=True, rolling_resume=False,
        existing_campaign_mutations=False, final_test_accessed=False,
    )
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root / "experiment_spec.json", spec)
    plan = command_plan(spec)
    write_immutable_json(root / "command_plan.json", plan)
    dry = root / "dry_run_submission_ledger.json"
    submit_exact_dag(
        identity=spec["content_hash"], plan=plan, output=dry,
        canonical_dry_run=dry, execute=False,
    )
    return spec


def validate_spec(spec: dict, *, check_source: bool = True) -> tuple[dict, dict]:
    validate(spec, "SPEC")
    if check_source:
        _source(Path(spec["project_dir"]), spec["source_commit"])
    source, evidence = source_evidence(Path(spec["source_spec_path"]))
    if (
        spec["source"] != evidence or spec["foundation"] != source["foundation"]
        or spec["source_lock"] != source["source_lock"]
        or spec["data_root"] != source["data_root"]
        or spec["runtime_profile"] != source["runtime_profile"]
        or spec["model"] != model_contract() or spec["node"] != node_spec(source)
        or spec["recipe"] != study_recipe()
        or spec["site"] not in tuple(execution_site(name) for name in SITE_NAMES.values())
        or spec["resources"] != RESOURCES
        or (spec["fresh_fit_count"], spec["slurm_job_count"]) != (1, 1)
        or spec["teacher_policy"] != "single_immediate_D033_only"
        or spec["historical_teachers_included"] is not False
        or (spec["ce_weight"], spec["kd_weight"], spec["temperature"]) != (.25, .75, 2.)
        or spec["initialization_and_sampler_paired_to_mt20_D000"] is not True
        or spec["particle_views_ram_only"] is not True
        or spec["rolling_resume"] is not False
        or spec["existing_campaign_mutations"] is not False
        or spec["final_test_accessed"] is not False
    ):
        raise ValueError("Endpoint ablation immutable semantics differ")
    root = _isolated(spec["campaign_root"], source, spec["project_dir"])
    if load_json(root / "experiment_spec.json") != spec:
        raise ValueError("Endpoint ablation root/spec mismatch")
    return source, evidence


def command_plan(spec: dict) -> dict:
    site, resources = spec["site"], spec["resources"]
    root = Path(spec["campaign_root"])
    command = slurm_options(site) + [
        f"--cpus-per-task={resources['cpus']}",
        f"--mem={resources['memory_mb']}M", f"--time={resources['minutes']}",
        "--gres=" + site["gres"], "--job-name=jc2d033only",
        "--chdir=" + spec["project_dir"],
        "--output=" + str(root / "slurm-%j.out"),
        str(Path(spec["project_dir"]) / "sbatch/run_jetclass2_d000_d033_only.sh"),
        spec["project_dir"], str(root / "experiment_spec.json"), site["name"],
    ]
    return artifact("PLAN", experiment_sha256=spec["content_hash"], commands=[
        dict(task_id="fit", dependencies=[], command=command),
    ])


def submit(spec: dict, *, execute: bool = False,
           authorization_phrase: str | None = None) -> dict:
    validate_spec(spec)
    root = Path(spec["campaign_root"])
    plan = command_plan(spec)
    if load_json(root / "command_plan.json") != plan:
        raise ValueError("Endpoint ablation command plan differs")
    if execute and authorization_phrase != AUTHORIZE:
        raise PermissionError("Explicit endpoint-ablation authorization required")
    if execute and shutil.disk_usage(root).free < 2 * 1024**3:
        raise OSError("Endpoint ablation requires at least 2 GiB free storage")
    claim = root / "submission_in_progress.claim"
    descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        if execute:
            return _guarded_exact_submission(spec, plan, root)
        dry = root / "dry_run_submission_ledger.json"
        return submit_exact_dag(
            identity=spec["content_hash"], plan=plan, output=dry,
            canonical_dry_run=dry, execute=False,
        )
    finally:
        claim.unlink()


def _receipt(spec: dict, job_id: str):
    root = Path(spec["campaign_root"])
    path = root / "submission_ledger_journal/0000_fit.json"
    for _ in range(30):
        if path.is_file():
            break
        time.sleep(1)
    expected = build_submission_event(
        campaign_spec_sha256=spec["content_hash"], task_id="fit",
        job_id=job_id, command=command_plan(spec)["commands"][0]["command"],
        sequence=0,
    )
    if not path.is_file() or load_json(path) != expected:
        raise PermissionError("Worker does not match the exact submitted job receipt")


def _save_state(path: Path, state: dict):
    buffer = BytesIO()
    torch.save(state, buffer)
    atomic_publish_bytes(path, buffer.getvalue())


def _teacher_probabilities(spec: dict, source: dict, identities: np.ndarray) -> np.ndarray:
    evidence = spec["source"]
    return load_bank(
        relative_file(Path(source["campaign_root"]), evidence["teacher_bank_path"]),
        foundation_sha256=evidence["foundation_sha256"],
        teacher_report_sha256=evidence["teacher_report_sha256"],
        teacher_node=TEACHER_ID, role="train", expected_identities=identities,
    )


def _accept(spec: dict, train, teacher: np.ndarray, *, device: str) -> dict:
    indexes = np.arange(min(256, len(train)))
    raw = train.batch(indexes)
    inputs = {
        name: torch.from_numpy(raw[name]).to(device)
        for name in ("features", "vectors", "mask")
    }
    labels = torch.from_numpy(raw["labels"]).to(device)
    torch.manual_seed(spec["node"]["initialization_seed"])
    model = DelphesParticleTransformer().to(device).train()
    logits = model(**inputs)
    loss = distillation_loss(
        logits, labels,
        teacher_probabilities=torch.from_numpy(teacher[indexes]).to(device),
        ce_weight=.25, kd_weight=.75, temperature=2.,
    )
    loss.backward()
    gradients = [value.grad for value in model.parameters() if value.grad is not None]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise ValueError("Endpoint-ablation acceptance gradients differ")
    report = artifact(
        "ACCEPTANCE", parents={"spec": spec["content_hash"]}, rows=len(indexes),
        exact_c25p75=True, temperature=2., teacher_count=1,
        teacher_node=TEACHER_ID, model_backward_passed=True,
        final_test_accessed=False, passed=True,
    )
    del model, logits, loss, inputs, labels, gradients
    gc.collect()
    torch.cuda.empty_cache()
    return report


def _reference_rows(spec: dict) -> list[dict]:
    kinds = {
        "M0HLT": "baseline", "U000": "oracle",
        TEACHER_ID: "teacher", MT20_ENDPOINT_ID: "mt20_endpoint",
    }
    baseline = spec["source"]["references"][0]["validation"]
    oracle = spec["source"]["references"][1]["validation"]
    return [dict(
        node_id=row["node_id"], kind=kinds[row["node_id"]],
        state="COMPLETE", passes=row["passes"], selected_pass=row["selected_pass"],
        validation=row["validation"],
        recovery=recovery(row["validation"], baseline, oracle),
    ) for row in spec["source"]["references"]]


def run(spec: dict, *, device: str = "cuda") -> dict:
    source, _ = validate_spec(spec)
    _execution_gate(source, device)
    job_id = os.environ.get("SLURM_JOB_ID", "")
    _receipt(spec, job_id)
    root = Path(spec["campaign_root"])
    descriptor = os.open(root / "execution.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    train = _cache(source, "train", "D000")
    validation = _cache(source, "validation", "D000")
    teacher = _teacher_probabilities(spec, source, train.identities)
    acceptance = _accept(spec, train, teacher, device=device)
    write_immutable_json(root / "acceptance.json", acceptance)
    torch.manual_seed(spec["node"]["initialization_seed"])
    model = DelphesParticleTransformer()
    training, state = train_kernel(
        model, train, validation, node=spec["node"], device=device,
        teacher_probabilities=teacher, teacher_identities=train.identities,
        training_recipe=spec["recipe"],
    )
    report = artifact(
        "TRAINING_REPORT", parents={
            "spec": spec["content_hash"], "acceptance": acceptance["content_hash"],
            "teacher_report": spec["source"]["teacher_report_sha256"],
            "teacher_bank_manifest": spec["source"]["teacher_bank_manifest_sha256"],
        }, kernel_training_report=training, source_commit=spec["source_commit"],
        node=spec["node"], teacher_count=1, historical_teachers_included=False,
        exact_c25p75=True, final_test_accessed=False,
    )
    _save_state(root / "selected.pt", state)
    write_immutable_json(root / "training_report.json", report)
    rows = _reference_rows(spec)
    baseline, oracle = rows[0]["validation"], rows[1]["validation"]
    rows.append(dict(
        node_id=NODE_ID, kind="d033_only_c25p75", state="COMPLETE",
        passes=training["passes"], selected_pass=training["selected_pass"],
        validation=training["validation"],
        recovery=recovery(training["validation"], baseline, oracle),
    ))
    validation_report = artifact(
        "VALIDATION_REPORT", parents={
            "spec": spec["content_hash"], "training_report": report["content_hash"],
        }, source_commit=spec["source_commit"], rows=rows,
        comparison="same_D000_seed_view_data_schedule_T2_only_teacher_policy_and_CE_KD_weights_change",
        final_test_accessed=False,
    )
    write_immutable_json(root / "validation_report.json", validation_report)
    outputs = ("acceptance.json", "selected.pt", "training_report.json", "validation_report.json")
    complete = artifact(
        "COMPLETE", parents={"spec": spec["content_hash"]},
        source_commit=spec["source_commit"],
        outputs={name: sha256_file(root / name) for name in outputs},
        final_test_accessed=False,
    )
    write_immutable_json(root / "complete.json", complete)
    return complete


def results(spec: dict) -> list[dict]:
    validate_spec(spec)
    root = Path(spec["campaign_root"])
    complete = load_json(root / "complete.json")
    validate(complete, "COMPLETE")
    if complete["parents"]["spec"] != spec["content_hash"]:
        raise ValueError("Endpoint-ablation completion lineage differs")
    for name, digest in complete["outputs"].items():
        if sha256_file(root / name) != digest:
            raise ValueError("Endpoint-ablation output bytes differ")
    report = load_json(root / "validation_report.json")
    validate(report, "VALIDATION_REPORT")
    return report["rows"]


def print_results(spec: dict):
    print("Recovery convention: M0HLT = 0%, persistent-HLT U000 = 100%.")
    print("Final test accessed: False\n")
    print(
        f"{'model':<48} {'kind':<18} {'pick':>8} {'accuracy':>10} "
        f"{'AUC':>10} {'R50':>11} {'AUC rec.':>10} {'R50 rec.':>10}"
    )
    for row in results(spec):
        metrics, recovered = row["validation"], row["recovery"]
        def number(value, digits=6):
            return "n/a" if value is None else f"{value:.{digits}f}"
        print(
            f"{row['node_id']:<48} {row['kind']:<18} "
            f"{row['selected_pass']}/{row['passes']: <4} "
            f"{number(metrics.get('accuracy')):>10} "
            f"{number(metrics.get('macro_ovr_auc')):>10} "
            f"{number(metrics.get('macro_r50'), 1):>11} "
            f"{number(recovered.get('macro_ovr_auc'), 1):>10} "
            f"{number(recovered.get('macro_r50'), 1):>10}"
        )


__all__ = [
    "AUTHORIZE", "MT20_ENDPOINT_ID", "NODE_ID", "REFERENCE_IDS", "TEACHER_ID",
    "command_plan", "create", "node_spec", "print_results", "results", "run",
    "source_evidence", "study_recipe", "submit", "validate_spec",
]

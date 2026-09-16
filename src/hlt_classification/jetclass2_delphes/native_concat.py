"""Isolated one-job concatenation oracle: authenticate, accept, fit, compare."""
from __future__ import annotations

import gc
from io import BytesIO
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from hlt_classification.data.cache_contracts import (
    array_sha256, atomic_publish_bytes, load_json, sha256_file, write_immutable_json,
)
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
from hlt_classification.scouting.hcwdl_recovery import build_submission_event
from .contracts import artifact as base_artifact, relative_file, validate as base_validate
from .execution import allocation, execution_site, gpu_identity, slurm_options, validate_resources
from .model import installed_environment, model_config, model_contract
from .native_concat_data import memory_bounds, prepare_cache
from .native_concat_model import NODE_ID, new_model, node_spec, zero_tag_parity
from .production import _source
from .reporting import evaluate_probabilities, recovery
from .salience_foundation import validate_foundation_spec
from .salience_learned_contracts import validate as validate_learned
from .salience_learned_data import IndexedRamCache, load_partition
from .salience_learned_graph import TRAINING, NODE_REGISTRY, recipe_payload
from .salience_learned_production import completed_task, validate_science_gate
from .salience_learned_training import _optimizer, _train_batch, predict, train_kernel
from .submission import _guarded_exact_submission


AUTHORIZE = "AUTHORIZE JETCLASS2 NATIVE CONCAT SINGLE JOB"
REFERENCES = ("M0HLT", "U000", "CE_SINGLE_D000")


def artifact(kind, **fields):
    return base_artifact("NATIVE_CONCAT_" + kind, **fields)


def validate(value, kind):
    return base_validate(value, "NATIVE_CONCAT_" + kind)


def recipe():
    return dict(
        node=node_spec(), training=dict(TRAINING), model=model_config(),
        input_order=["native_offline", "native_hlt"], matching_used=False,
        numerical_features=17, source_transport_channel=17,
        source_codes={"offline": 0, "hlt": 1, "padding": -1},
        source_embedding_width=128, normalization="each_native_reconstruction_own_axis",
        native_capacity=240, combined_capacity=480, final_test_accessed=False,
        validation_selection="V_checkpoint", evaluation="V_report", deployable=False,
        loss="CE_only", gradient_accumulation=False, rolling_resume=False,
    )


def reference_row(source, node_id):
    pointer = completed_task(source, "train_" + node_id)
    if pointer is None:
        return None
    root = Path(source["campaign_root"])
    result = pointer["result"]
    report = load_json(relative_file(root, result["training_report"]))
    diagnostic = load_json(relative_file(root, result["diagnostic"]))
    validate_learned(report, "TRAINING_REPORT")
    validate_learned(diagnostic, "DIAGNOSTIC")
    if (report["content_hash"] != result["training_report_sha256"]
            or diagnostic["content_hash"] != result["diagnostic_sha256"]
            or diagnostic["parents"]["training_report"] != report["content_hash"]
            or report["parents"]["campaign_spec"] != source["content_hash"]
            or report["parents"]["foundation"] != source["foundation"]["content_hash"]
            or report["source_commit"] != source["source_commit"]
            or report["node"] != NODE_REGISTRY[node_id].payload()
            or not report["scientific_fit"] or report["final_test_accessed"]
            or diagnostic["final_test_accessed"]):
        raise ValueError("Reference report lineage differs")
    return dict(node=node_id, metrics=diagnostic["report_metrics"],
                passes=report["passes"], selected_pass=report["selected_pass"],
                task_sha256=pointer["content_hash"])


def source_evidence(path):
    source = load_json(path)
    validate_learned(source, "CAMPAIGN_SPEC")
    # Authenticate the historical source's immutable gate/output evidence, not
    # its current queue or a re-execution of millions of unused matching maps.
    # This experiment reads native endpoints; assignment files are not inputs.
    foundation_hash = validate_foundation_spec(source["foundation"])
    if (source["source_lock"]["foundation_sha256"] != foundation_hash
            or source["source_lock"]["split_sha256"] != source["foundation"]["splits"]["content_hash"]
            or source["recipe"] != recipe_payload() or source["model"] != model_contract()
            or source["ordinary_final_test_capability"] is not False
            or source["final_test_accessed"] is not False):
        raise ValueError("Reference population/model/recipe differs")
    gates = validate_science_gate(source)
    partition, arrays = load_partition(Path(source["campaign_root"]) / "validation_partition.json")
    if partition["parents"]["campaign_spec"] != source["content_hash"]:
        raise ValueError("Reference validation partition lineage differs")
    references = [reference_row(source, name) for name in REFERENCES]
    if any(row is None for row in references):
        raise PermissionError("M0HLT, U000 and CE_SINGLE_D000 must be durably complete")
    count = int(np.sum(arrays["partition"] == 2))
    if any(row["metrics"]["rows"] != count for row in references):
        raise ValueError("Reference held-out validation coverage differs")
    evidence = dict(campaign_sha256=source["content_hash"], gate_hashes=gates,
                    partition_sha256=partition["content_hash"], references=references)
    return source, evidence, arrays


def _isolated(root, source, project):
    root = Path(root).resolve()
    protected = (Path(source["campaign_root"]), Path(source["data_root"]),
                 Path(source["source_lock"]["foundation_root"]), Path(source["project_dir"]),
                 Path(project))
    if any(root == p.resolve() or root.is_relative_to(p.resolve())
           or p.resolve().is_relative_to(root) for p in protected):
        raise ValueError("Concatenation root overlaps a source/data/project directory")
    return root


def create(*, reference_spec, output_root, project, source_commit,
           cpus=8, workers=8, memory_mb=131072, minutes=2880):
    project = Path(project).resolve()
    _source(project, source_commit)
    source, evidence, _ = source_evidence(reference_spec)
    foundation = source["foundation"]
    if (foundation["splits"]["profile"] != "TRAIN_500K"
            or foundation["splits"]["role_counts"] != dict(train=500_000, validation=1_000_000, final_test=1_000_000)
            or foundation["inputs"]["capacity"] != 240):
        raise ValueError("Concatenation requires exact TRAIN_500K / capacity-240 source")
    site = execution_site("sporc_a100")
    validate_resources(site, cpus, memory_mb, workers)
    if type(minutes) is not int or not 60 <= minutes <= 4320:
        raise ValueError("Concatenation walltime must be 60..4320 minutes")
    bounds = memory_bounds(foundation, workers)
    if sum(bounds.values()) > memory_mb * 1024**2 * .75:
        raise MemoryError(f"Native concatenation bounds {bounds} exceed 75% of requested RAM; use fewer workers or more RAM")
    root = _isolated(output_root, source, project)
    if root.exists():
        raise FileExistsError("Concatenation needs a fresh root, never reuse another campaign")
    spec = artifact(
        "SPEC", source_commit=source_commit, project_dir=str(project), campaign_root=str(root),
        reference_spec_path=str(Path(reference_spec).resolve()), reference=evidence,
        data_root=source["data_root"], foundation=foundation, recipe=recipe(),
        site=site, resources=dict(cpus=cpus, workers=workers, memory_mb=memory_mb, minutes=minutes),
        cache_bounds=bounds, fresh_fit_count=1, slurm_job_count=1,
        final_test_accessed=False, existing_campaign_mutations=False,
    )
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root / "experiment_spec.json", spec)
    plan = command_plan(spec)
    write_immutable_json(root / "command_plan.json", plan)
    dry = root / "dry_run_submission_ledger.json"
    submit_exact_dag(identity=spec["content_hash"], plan=plan, output=dry,
                     canonical_dry_run=dry, execute=False)
    return spec


def validate_spec(spec, *, check_source=True):
    validate(spec, "SPEC")
    if check_source:
        _source(Path(spec["project_dir"]), spec["source_commit"])
    source, evidence, partition = source_evidence(Path(spec["reference_spec_path"]))
    resources = spec["resources"]
    validate_resources(spec["site"], resources["cpus"], resources["memory_mb"], resources["workers"])
    if (spec["recipe"] != recipe() or spec["reference"] != evidence
            or spec["foundation"] != source["foundation"] or spec["data_root"] != source["data_root"]
            or spec["site"] != execution_site("sporc_a100")
            or spec["foundation"]["splits"]["profile"] != "TRAIN_500K"
            or spec["foundation"]["inputs"]["capacity"] != 240
            or spec["foundation"]["splits"]["role_counts"] != dict(train=500_000, validation=1_000_000, final_test=1_000_000)
            or spec["fresh_fit_count"] != 1 or spec["slurm_job_count"] != 1
            or spec["final_test_accessed"] is not False or spec["existing_campaign_mutations"] is not False
            or spec["cache_bounds"] != memory_bounds(spec["foundation"], resources["workers"])
            or sum(spec["cache_bounds"].values()) > resources["memory_mb"] * 1024**2 * .75
            or type(resources["minutes"]) is not int or not 60 <= resources["minutes"] <= 4320):
        raise ValueError("Native concatenation immutable semantics differ")
    root = _isolated(spec["campaign_root"], source, spec["project_dir"])
    if load_json(root / "experiment_spec.json") != spec:
        raise ValueError("Concatenation root/spec mismatch")
    return source, partition


def command_plan(spec):
    r, root = spec["resources"], Path(spec["campaign_root"])
    command = slurm_options(spec["site"]) + [
        f"--cpus-per-task={r['cpus']}", f"--mem={r['memory_mb']}M", f"--time={r['minutes']}",
        "--gres=" + spec["site"]["gres"], "--job-name=jc2concat_fit",
        "--chdir=" + spec["project_dir"], "--output=" + str(root / "slurm-%j.out"),
        str(Path(spec["project_dir"]) / "sbatch/run_jetclass2_delphes_native_concat.sh"),
        spec["project_dir"], str(root / "experiment_spec.json"),
    ]
    return artifact("PLAN", campaign_sha256=spec["content_hash"], commands=[
        dict(task_id="fit", dependencies=[], command=command),
    ])


def submit(spec, *, execute=False, authorization_phrase=None):
    validate_spec(spec)
    root = Path(spec["campaign_root"])
    plan = command_plan(spec)
    if load_json(root / "command_plan.json") != plan:
        raise ValueError("Concatenation command plan differs")
    if execute and authorization_phrase != AUTHORIZE:
        raise PermissionError("Explicit single-job concatenation authorization required")
    if execute and shutil.disk_usage(root).free < 2 * 1024**3:
        raise OSError("Concatenation requires at least 2 GiB free storage")
    claim = root / "submission_in_progress.claim"
    fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        if execute:
            return _guarded_exact_submission(spec, plan, root)
        dry = root / "dry_run_submission_ledger.json"
        return submit_exact_dag(identity=spec["content_hash"], plan=plan, output=dry,
                                canonical_dry_run=dry, execute=False)
    finally:
        claim.unlink()


def _receipt(spec, job_id):
    root = Path(spec["campaign_root"])
    path = root / "submission_ledger_journal/0000_fit.json"
    # Slurm can start before the submitter publishes its tiny durable receipt.
    for _ in range(30):
        if path.is_file():
            break
        time.sleep(1)
    expected = build_submission_event(
        campaign_spec_sha256=spec["content_hash"], task_id="fit", job_id=job_id,
        command=command_plan(spec)["commands"][0]["command"], sequence=0,
    )
    if not path.is_file() or load_json(path) != expected:
        raise PermissionError("Worker does not match the exact submitted job receipt")


def partition_views(cache, arrays):
    if (not np.array_equal(cache.identities, arrays["identities"])
            or not np.array_equal(cache.labels, arrays["labels"])):
        raise ValueError("Native validation population does not match reference partition")
    return {name: IndexedRamCache(cache, np.flatnonzero(arrays["partition"] == code), role="validation")
            for code, name in enumerate(("checkpoint", "diagnostic", "report"))}


def _balanced_probe(cache):
    indices = np.concatenate([np.flatnonzero(cache.labels == cls)[:64] for cls in range(11)])
    if len(np.unique(cache.labels[indices])) != 11:
        raise ValueError("Acceptance requires all classes")
    return IndexedRamCache(cache, indices, role=cache.role)


def accept(spec, train, validation, checkpoint, *, device="cuda"):
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    raw = train.batch(np.arange(8))
    parity = zero_tag_parity(raw, device=device)
    gc.collect()
    torch.cuda.empty_cache()
    print("JC2-CONCAT phase=parity passed", flush=True)
    # The longest real row gives the exact maximum padding length possible in
    # ANY future batch. Repeat its physical input to stress batch 256, with Adam.
    longest = max((train, validation), key=lambda c: int(c.lengths.max()))
    index = int(np.argmax(longest.lengths))
    stress = longest.batch(np.full(256, index, np.int64))
    model = new_model().to(device).train()
    optimizer = _optimizer(model)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _train_batch(model, stress, node=node_spec(), device=device, teacher=None, alpha=1.)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Nonfinite concatenation stress gradient")
        if not torch.count_nonzero(model.mod.embed.source.weight.grad):
            raise ValueError("Source embedding has no gradient")
        optimizer.step()
    torch.cuda.synchronize()
    del model, optimizer, stress, loss
    gc.collect()
    torch.cuda.empty_cache()
    small_train, small_val = _balanced_probe(train), _balanced_probe(checkpoint)
    model = new_model()
    report, state = train_kernel(model, lambda _: small_train, lambda _: small_val,
                                 node=node_spec(), device=device, acceptance_passes=1)
    # Exercise serialization/reload in RAM, without retaining miniature weights.
    buffer = BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    model.load_state_dict(torch.load(buffer, map_location=device, weights_only=True), strict=True)
    peak = torch.cuda.max_memory_allocated()
    gpu = gpu_identity()
    if peak > .85 * gpu["total_memory_bytes"]:
        raise MemoryError("Concatenation exceeds 85% GPU memory; batch/trimming are not changed")
    history = report["validation_history"][0]
    projected = 2 * 100 * (history["train_seconds"] * len(train) / len(small_train)
                          + history["validation_seconds"] * len(checkpoint) / len(small_val))
    value = artifact("ACCEPTANCE", parents={"spec": spec["content_hash"]},
                     source_commit=spec["source_commit"], passed=True, parity=parity,
                     gpu=gpu, installed_environment=installed_environment(),
                     peak_cuda_bytes=peak, longest_actual_tokens=int(longest.lengths[index]),
                     stress_batch_size=256, train_rows=len(train), validation_rows=len(validation),
                     cache_bytes=train.nbytes + validation.nbytes,
                     miniature_report=report, projected_loop_seconds=projected,
                     elapsed_seconds=time.monotonic()-started, final_test_accessed=False)
    del model, state
    gc.collect()
    torch.cuda.empty_cache()
    return value


def _save_state(path, state):
    data = BytesIO()
    torch.save(state, data)
    atomic_publish_bytes(path, data.getvalue())


def finish_fit(spec, train, checkpoint, report_cache, acceptance, *, device):
    validate(acceptance, "ACCEPTANCE")
    if (acceptance["parents"]["spec"] != spec["content_hash"] or acceptance["passed"] is not True):
        raise PermissionError("Concatenation scientific fit requires its own passed acceptance")
    root = Path(spec["campaign_root"])
    model = new_model()  # Resets the matched RNG stream after the discarded probe.
    training, state = train_kernel(model, lambda _: train, lambda _: checkpoint,
                                   node=node_spec(), device=device)
    training = artifact("TRAINING_REPORT", **{
        k: v for k, v in training.items() if k not in {"contract", "schema_version", "content_hash"}
    }, parents={"spec": spec["content_hash"], "acceptance": acceptance["content_hash"]},
        source_commit=spec["source_commit"], train_identities_sha256=array_sha256("identities", train.identities),
        checkpoint_identities_sha256=array_sha256("identities", checkpoint.identities))
    _save_state(root / "selected.pt", state)
    write_immutable_json(root / "training_report.json", training)
    probabilities = predict(model, report_cache, node=node_spec(), device=device)
    metrics = evaluate_probabilities(report_cache.labels, probabilities)
    references = spec["reference"]["references"]
    baseline, anchor = references[0]["metrics"], references[1]["metrics"]
    rows = references + [dict(node=NODE_ID, metrics=metrics, passes=training["passes"],
                              selected_pass=training["selected_pass"])]
    rows = [dict(row, recovery=recovery(row["metrics"], baseline, anchor)) for row in rows]
    validation = artifact("VALIDATION_REPORT", parents={
        "spec": spec["content_hash"], "training_report": training["content_hash"],
        "partition": spec["reference"]["partition_sha256"],
    }, source_commit=spec["source_commit"], rows=rows, evaluation_role="V_report",
        report_identities_sha256=array_sha256("identities", report_cache.identities),
        u000_meaning="persistent_HLT_privileged_not_pure_offline",
        final_test_accessed=False)
    write_immutable_json(root / "validation_report.json", validation)
    files = ("selected.pt", "training_report.json", "validation_report.json", "acceptance.json")
    complete = artifact("COMPLETE", parents={"spec": spec["content_hash"]},
                        source_commit=spec["source_commit"], outputs={
                            name: sha256_file(root / name) for name in files
                        }, final_test_accessed=False)
    write_immutable_json(root / "complete.json", complete)
    return validation


def run(spec):
    started = time.monotonic()
    source, partition = validate_spec(spec)
    job, cpus, memory = allocation(spec["site"])
    r = spec["resources"]
    if (cpus != r["cpus"] or memory != r["memory_mb"]
            or installed_environment() != source["runtime_profile"]["installed_environment"]):
        raise PermissionError("Concatenation allocation/environment differs")
    _receipt(spec, job)
    root = Path(spec["campaign_root"])
    fd = os.open(root / "execution.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)  # Kept permanently: this root has one nonresumable attempt.
    if shutil.disk_usage(root).free < 2 * 1024**3:
        raise OSError("Insufficient concatenation storage headroom")
    caches = {role: prepare_cache(spec["foundation"], data_root=Path(spec["data_root"]),
                                  role=role, workers=r["workers"], max_ram_bytes=spec["cache_bounds"][role])
              for role in ("train", "validation")}
    subsets = partition_views(caches["validation"], partition)
    acceptance = accept(spec, caches["train"], caches["validation"], subsets["checkpoint"])
    projected = 2 * (time.monotonic()-started) + acceptance["projected_loop_seconds"]
    if projected > r["minutes"] * 60:
        write_immutable_json(root / "runtime_rejection.json", artifact(
            "RUNTIME_REJECTION", parents={"spec": spec["content_hash"]},
            projected_wall_seconds=projected, requested_seconds=r["minutes"] * 60,
            measurements=acceptance, scientific_fit_started=False, final_test_accessed=False,
        ))
        raise RuntimeError("Measured concatenation runtime exceeds requested walltime; no scientific fit started")
    acceptance = artifact("ACCEPTANCE", **{
        k: v for k, v in acceptance.items() if k not in {"contract", "schema_version", "content_hash"}
    }, projected_wall_seconds=projected, runtime_budget_passed=True)
    write_immutable_json(root / "acceptance.json", acceptance)
    print("JC2-CONCAT phase=acceptance passed; resetting and starting full CE fit", flush=True)
    return finish_fit(spec, caches["train"], subsets["checkpoint"], subsets["report"], acceptance, device="cuda")


def results(spec):
    validate(spec, "SPEC")
    root = Path(spec["campaign_root"])
    complete = load_json(root / "complete.json")
    validate(complete, "COMPLETE")
    if (complete["parents"]["spec"] != spec["content_hash"]
            or complete["source_commit"] != spec["source_commit"]
            or complete["final_test_accessed"] is not False
            or set(complete["outputs"]) != {"selected.pt", "training_report.json", "validation_report.json", "acceptance.json"}):
        raise ValueError("Concatenation completion lineage differs")
    for name, digest in complete["outputs"].items():
        if sha256_file(relative_file(root, name)) != digest:
            raise ValueError("Concatenation completed output bytes differ")
    report = load_json(root / "validation_report.json")
    validate(report, "VALIDATION_REPORT")
    training = load_json(root / "training_report.json")
    validate(training, "TRAINING_REPORT")
    acceptance = load_json(root / "acceptance.json")
    validate(acceptance, "ACCEPTANCE")
    if (report["parents"] != {"spec": spec["content_hash"],
                               "training_report": training["content_hash"],
                               "partition": spec["reference"]["partition_sha256"]}
            or training["parents"] != {"spec": spec["content_hash"], "acceptance": acceptance["content_hash"]}
            or acceptance["parents"]["spec"] != spec["content_hash"]
            or acceptance["passed"] is not True
            or training["node"] != node_spec() or not training["scientific_fit"]
            or report["evaluation_role"] != "V_report" or report["final_test_accessed"] is not False):
        raise ValueError("Concatenation result parents/semantics differ")
    rows = list(report["rows"])
    source = load_json(spec["reference_spec_path"])
    validate_learned(source, "CAMPAIGN_SPEC")
    if source["content_hash"] != spec["reference"]["campaign_sha256"]:
        raise ValueError("Reference campaign changed")
    static = reference_row(source, "STATIC_U000_D000")
    if static is not None:
        if static["metrics"]["rows"] != rows[-1]["metrics"]["rows"]:
            raise ValueError("Static fusion report population differs")
        static["recovery"] = recovery(static["metrics"], rows[0]["metrics"], rows[1]["metrics"])
        rows.append(static)
    return rows


def print_results(spec):
    rows = results(spec)
    def f(value, digits=4):
        return "n/a" if value is None else f"{value:.{digits}f}"
    print("Held-out V_report; M0HLT=0%, persistent-HLT U000=100% (not pure offline).")
    print(f"{'model':<32} {'pick':>7} {'accuracy':>10} {'AUC':>10} {'R50':>10} {'AUC rec%':>10} {'R50 rec%':>10}")
    for row in rows:
        m, rec = row["metrics"], row["recovery"]
        print(f"{row['node']:<32} {str(row['selected_pass'])+'/'+str(row['passes']):>7} "
              f"{f(m['accuracy'],6):>10} {f(m['macro_ovr_auc'],6):>10} {f(m['macro_r50'],1):>10} "
              f"{f(rec['macro_ovr_auc'],1):>10} {f(rec['macro_r50'],1):>10}")
    print("\nPer-class QCD rejection at 50% signal efficiency (recovery % in parentheses):")
    for name in rows[0]["metrics"]["per_class"]:
        print(name)
        for row in rows:
            m = row["metrics"]["per_class"][name]
            value = f(m["rejection_at_50pct"],1)
            if m["zero_fpr_censored"]:
                value = f"censored (one-event resolution {m['one_event_resolution_rejection']})"
            print(f"  {row['node']:<32} {value} ({f(row['recovery']['per_class_r50'][name],1)}%)")
    print("Final test accessed: False")

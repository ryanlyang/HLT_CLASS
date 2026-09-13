"""Production task dispatch; every output belongs to one immutable attempt."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import math
import time
import numpy as np
import torch

from ..model import installed_environment
from ..reporting import evaluate_probabilities
from .contracts import (artifact, validate, load_json, write_immutable_json, atomic_publish_bytes,
                        sha256_file, checked_payload, publish_arrays, load_npz_arrays)
from .campaign import (validate_study, validate_stage, task_result, prior, stage_path, completed,
                       configuration_lock, reporting_lock)
from .roles import build_roles, load_role, read_rows
from .targets import summarize
from . import banks, cache, normalization, execution, training, reporting
from .model import create_model, state_hash
from .preparation_import import imported_stage, audit_study_storage


def context(study):
    sample, sample_root = prior(study, "GATE", "sample")
    split = load_json(sample_root / "roles" / "role_split.json")
    validate(split, "ROLE_SPLIT")
    if sample["role_split_sha256"] != split["content_hash"]:
        raise ValueError("Role split parent differs")
    return split, sample_root / "roles"


def target_collection(stage, split, metadata, role, lock=None):
    roots = [task_result(stage, f"targets_{role}_{i:02d}")[1]
             for i in range(math.ceil(len(metadata["labels"])/100000))]
    return banks.load_targets(roots[0], split, metadata, role, reporting_lock=lock, shard_roots=roots)


def make_caches(study, split, roles_root, resources, roles, lock=None):
    began = time.monotonic()
    result = {}
    # These conservative bounds include worker/IPC copies; 25% of allocation
    # remains outside them for targets, model/optimizer, best RAM state and runtime.
    rows = sum(split["roles"][r]["rows"] for r in roles)
    for role in roles:
        budget = int(resources["memory_mb"]*2**20*.75*split["roles"][role]["rows"]/rows)
        result[role] = cache.prepare(study["data_root"], study["inventory"], split,
            load_role(roles_root, split, role), role=role, workers=resources["workers"],
            budget_bytes=budget, reporting_lock=lock)
    return result, time.monotonic()-began


def _model_summary(output, root):
    validate(output, "TRAIN_OUTPUT")
    report, weights = output["report"], output["weights"]
    validate(report, "TRAINING_REPORT"); validate(weights, "SELECTED_WEIGHTS")
    checked_payload(root, weights["payload"])
    if (weights["report_sha256"] != report["content_hash"] or report["acceptance_only"]
            or weights["state_sha256"] != report["selected_state_sha256"]):
        raise ValueError("Not an authenticated scientific selected checkpoint")
    return dict(node=report["node"], validation=report["validation"], selected_update=report["selected_update"],
        selected_pass=report["selected_pass"], completed_passes=report["completed_passes"],
        report_sha256=report["content_hash"], weights=weights, output_root=str(root),
        initial_shared_sha256=report["initial_shared_sha256"], initial_head_sha256=report["initial_head_sha256"],
        sampler_order_sha256=report["sampler_order_sha256"])


def _check_pairing(models):
    for replicate in {m["node"]["replicate"] for m in models.values()}:
        group = [m for m in models.values() if m["node"]["replicate"] == replicate]
        if len({m["initial_shared_sha256"] for m in group}) != 1:
            raise ValueError("Matched shared initialization differs")
        minimum = min(len(m["sampler_order_sha256"]) for m in group)
        if any(m["sampler_order_sha256"][:minimum] != group[0]["sampler_order_sha256"][:minimum] for m in group):
            raise ValueError("Matched training orders differ")
        for head in {h for m in group for h in m["initial_head_sha256"]}:
            if len({m["initial_head_sha256"][head] for m in group if head in m["initial_head_sha256"]}) != 1:
                raise ValueError("Matched auxiliary initialization differs")


def _sample(study, output_root):
    split = build_roles(study["data_root"], study["inventory"], study["profile"], output_root / "roles")
    write_immutable_json(output_root / "roles" / "role_split.json", split)
    metadata = load_role(output_root / "roles", split, "TRAIN")
    began = time.monotonic()
    checked = 0
    for _, jet in read_rows(study["data_root"], study["inventory"], split, metadata,
                            role="TRAIN", include_offline=True, row_end=1024):
        summarize(jet.offline)
        checked += 1
    seconds = time.monotonic()-began
    # Conservative serial extrapolation (including sample file verification),
    # not a claimed process-linear speedup. Only future target tasks use it.
    minutes = max(30, math.ceil(2*seconds*100000/checked/60))
    if minutes > 2880:
        raise ValueError("Measured target preparation exceeds walltime envelope")
    return artifact("SAMPLE_PROFILE", study_sha256=study["content_hash"], role_split_sha256=split["content_hash"],
                    checked_train_rows=checked, seconds=seconds, target_minutes=minutes, report_particles_accessed=False)


def dispatch(study, stage, task, output_root):
    kind = task["kind"]
    if kind == "sample":
        return _sample(study, output_root)
    split, roles_root = context(study)
    if kind == "target":
        lock = task_result(stage, "reporting_lock")[0] if task["role"] == "VAL_REPORT" else None
        return banks.build_shard(study["data_root"], study["inventory"], split,
            load_role(roles_root, split, task["role"]), root=output_root, role=task["role"],
            shard=task["shard"], workers=stage["resources"]["workers"], reporting_lock=lock)
    if kind == "normalize":
        bank, values, mask, extra = target_collection(stage, split, load_role(roles_root, split, "TRAIN"), "TRAIN")
        select_bank, _, _, _ = target_collection(stage, split, load_role(roles_root, split, "VAL_SELECT"), "VAL_SELECT")
        normalizer = normalization.fit(values, mask, extra["hlt_pt"], train_bank_sha256=bank["content_hash"], role="TRAIN")
        return artifact("PREPARATION_LOCK", study_sha256=study["content_hash"], role_split_sha256=split["content_hash"],
                        train_bank=bank, selection_bank=select_bank, normalizer=normalizer)
    if kind in {"profile", "train"}:
        if "preparation_import" in study:
            preparation_stage = imported_stage(study, "PREPARE", authenticate=True)
        else:
            preparation_stage = stage if kind == "profile" else load_json(stage_path(study, "PREPARE") / "stage_spec.json")
        prepared, _ = task_result(preparation_stage, "normalize")
        validate(prepared, "PREPARATION_LOCK")
        bank, values, mask, _ = target_collection(preparation_stage, split, load_role(roles_root, split, "TRAIN"), "TRAIN")
        select_bank, _, _, _ = target_collection(preparation_stage, split, load_role(roles_root, split, "VAL_SELECT"), "VAL_SELECT")
        if bank != prepared["train_bank"] or select_bank != prepared["selection_bank"]:
            raise ValueError("Preparation targets changed")
        caches, seconds = make_caches(study, split, roles_root, stage["resources"], ("TRAIN", "VAL_SELECT"))
        if kind == "profile":
            return execution.measure(study, split, caches["TRAIN"], caches["VAL_SELECT"], values, mask,
                prepared["normalizer"], cache_seconds=seconds, target_minutes=stage["resources"]["target_minutes"])
        model = create_model(task["node"]["arm"], task["node"]["replicate"])
        report, state = training.train(model, caches["TRAIN"], caches["VAL_SELECT"], values, mask,
            prepared["normalizer"], node=task["node"], execution_identity=dict(
                study_sha256=study["content_hash"], source_commit=study["source_commit"]))
        stream = BytesIO(); torch.save(state, stream)
        raw = stream.getvalue()
        if len(raw) > 64*2**20:
            raise ValueError("Selected model exceeds file limit")
        atomic_publish_bytes(output_root / "selected.pt", raw)
        weights = artifact("SELECTED_WEIGHTS", report_sha256=report["content_hash"],
            state_sha256=report["selected_state_sha256"], payload=dict(path="selected.pt", bytes=len(raw),
            sha256=sha256_file(output_root / "selected.pt")), includes_optimizer=False, includes_auxiliary=True)
        write_immutable_json(output_root / "training_report.json", report)
        return artifact("TRAIN_OUTPUT", report=report, weights=weights, cache_seconds=seconds)
    if kind in {"configuration", "report_lock"}:
        models = {t["task_id"]: _model_summary(*task_result(stage, t["task_id"]))
                  for t in stage["tasks"] if t["kind"] == "train"}
        _check_pairing(models)
        if kind == "configuration":
            prepared, _ = prior(study, "PREPARE", "normalize")
            return configuration_lock(study, split, models, normalizer=prepared["normalizer"],
                                      preparation_lock_sha256=prepared["content_hash"])
        configuration, _ = prior(study, "DISCOVERY", "configuration_lock")
        return reporting_lock(study, split, configuration, models)
    if kind == "evaluate":
        lock, _ = task_result(stage, "reporting_lock")
        metadata = load_role(roles_root, split, "VAL_REPORT")
        bank, values, mask, hlt = target_collection(stage, split, metadata, "VAL_REPORT", lock)
        prepared, _ = prior(study, "PREPARE", "normalize")
        if (lock["normalization_sha256"] != prepared["normalizer"]["content_hash"]
                or lock["hlt_pt_edges"] != prepared["normalizer"]["hlt_pt_edges"]
                or lock["metrics"] != reporting.metrics_contract()):
            raise ValueError("Reporting metric/calibration lock differs")
        summary = lock["models"][task["node"]["node_id"]]
        model = create_model(task["node"]["arm"], task["node"]["replicate"])
        path = checked_payload(summary["output_root"], summary["weights"]["payload"])
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state_hash(state) != summary["weights"]["state_sha256"]:
            raise ValueError("Selected state tensor hash differs")
        model.load_state_dict(state, strict=True); model.cuda()
        caches, seconds = make_caches(study, split, roles_root, stage["resources"], ("VAL_REPORT",), lock)
        c = caches["VAL_REPORT"]
        probabilities, heads = training.predict(model, c, device="cuda", heads=True)
        shards = []
        for start in range(0, len(c), 100000):
            end = min(start+100000, len(c))
            shards.append(dict(start=start, end=end, **publish_arrays(output_root, f"predictions/{start:09d}.npz",
                dict(identities=c.identities[start:end], probabilities=probabilities[start:end]))))
        # Native HLT pT is diagnostic only; never passed as another model input.
        pt = []
        for b in c.blocks:
            total = np.add.reduceat(b.vectors.astype(np.float64), b.offsets[:-1], axis=0)
            pt.extend(np.hypot(total[:, 0], total[:, 1]))
        bins = np.searchsorted(prepared["normalizer"]["hlt_pt_edges"], pt, side="right")
        by_bin = []
        for i in range(5):
            selected = bins == i
            has_classes = set(np.unique(c.labels[selected])) == set(range(11))
            by_bin.append(dict(bin=i, rows=int(selected.sum()), metrics=evaluate_probabilities(
                c.labels[selected], probabilities[selected]) if has_classes else None, missing_class=not has_classes))
        return artifact("EVALUATION_REPORT", node=task["node"], reporting_lock_sha256=lock["content_hash"],
            role_split_sha256=split["content_hash"], target_bank_sha256=bank["content_hash"], rows=len(c),
            metrics=evaluate_probabilities(c.labels, probabilities), by_hlt_pt_bin=by_bin, probability_shards=shards,
            auxiliary=reporting.auxiliary_metrics(heads, values, mask, prepared["normalizer"], hlt=hlt),
            cache_seconds=seconds, recovery="not_evaluated_no_registered_offline_oracle")
    if kind == "bootstrap":
        metadata = load_role(roles_root, split, "VAL_REPORT")
        lock, _ = task_result(stage, "reporting_lock")
        models = {}
        for name in lock["models"]:
            report, root = task_result(stage, "eval_"+name)
            validate(report, "EVALUATION_REPORT")
            if report["reporting_lock_sha256"] != lock["content_hash"]:
                raise ValueError("Evaluation lock differs")
            probabilities = np.empty((len(metadata["labels"]), 11), np.float32)
            cursor = 0
            for shard in report["probability_shards"]:
                if shard["start"] != cursor:
                    raise ValueError("Probability coverage differs")
                arrays = load_npz_arrays(checked_payload(root, shard))
                end = shard["end"]
                if not np.array_equal(arrays["identities"], metadata["identities"][cursor:end]):
                    raise ValueError("Paired report identities differ")
                probabilities[cursor:end] = arrays["probabilities"]
                cursor = end
            if cursor != len(probabilities):
                raise ValueError("Incomplete probability coverage")
            models[name] = reporting.WeightedMetrics(metadata["labels"], probabilities)
        draws = reporting.bootstrap(models, metadata["file_index"], start=task["start"], end=task["end"])
        return artifact("BOOTSTRAP", reporting_lock_sha256=lock["content_hash"], draws=draws,
                        unit="source_file_cluster", seed=reporting.BOOTSTRAP_SEED)
    if kind == "aggregate":
        lock, _ = task_result(stage, "reporting_lock")
        evaluations = {k: task_result(stage, "eval_"+k)[0] for k in lock["models"]}
        draws = [r for i in range(20) for r in task_result(stage, f"bootstrap_{i:02d}")[0]["draws"]]
        return artifact("AGGREGATE", reporting_lock_sha256=lock["content_hash"], evaluations=evaluations,
            paired=reporting.paired_statistics({k: r["metrics"] for k, r in evaluations.items()}),
            conditional_file_bootstrap=reporting.bootstrap_intervals(draws),
            uncertainty_caveat="Conditional on three trained seeds; file/event independence remains provisional",
            scientific_result_does_not_control_completion=True, recovery="not_evaluated_no_oracle")
    if kind == "complete":
        aggregate, _ = task_result(stage, "aggregate")
        return artifact("COMPLETION", aggregate_sha256=aggregate["content_hash"], fresh_fit_count=22,
                        output_inventory=audit_study_storage(study), scientific_result_does_not_control_completion=True)
    raise ValueError("Unknown registered task kind")


def run_task(stage, task_id, attempt_root):
    study = load_json(stage["study_spec_path"])
    validate_study(study, source=True); validate_stage(stage, study)
    task = next((t for t in stage["tasks"] if t["task_id"] == task_id), None)
    if task is None:
        raise ValueError("Task outside stage")
    if completed(stage, task_id) is not None:
        return completed(stage, task_id)[0]
    # CPU allocations use a separate scheduler-only check; GPU tasks also
    # authenticate visible device, isolated environment and installed libraries.
    from .submission import verify_allocation
    verify_allocation(study, stage, task)
    for dependency in task["dependencies"]:
        task_result(stage, dependency)
    if task["kind"] in {"train", "evaluate"}:
        profile, _ = prior(study, "PREPARE", "profile")
        if installed_environment() != profile["installed_environment"]:
            raise ValueError("Installed scientific environment differs from acceptance")
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    directory = Path(attempt_root).resolve()
    stage_root = Path(stage["root"]).resolve()
    if not directory.is_relative_to(stage_root / "attempts"):
        raise ValueError("Attempt escaped its stage")
    output_root = directory / task_id
    if output_root.exists():
        raise FileExistsError("Partial outputs preserved; use a fresh exact recovery attempt, never resume")
    output_root.mkdir(parents=True)
    result = dispatch(study, stage, task, output_root)
    write_immutable_json(output_root / "result.json", result)
    audit_study_storage(study)
    outputs = [dict(path=p.relative_to(stage_root).as_posix(), bytes=p.stat().st_size, sha256=sha256_file(p))
               for p in sorted(output_root.rglob("*")) if p.is_file()]
    receipt = artifact("TASK_RECEIPT", stage_sha256=stage["content_hash"], task_id=task_id,
        output_directory=output_root.relative_to(stage_root).as_posix(), outputs=outputs, result=result)
    write_immutable_json(stage_root / "receipts" / (task_id + ".json"), receipt)
    return result

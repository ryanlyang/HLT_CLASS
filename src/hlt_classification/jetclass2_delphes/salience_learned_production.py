"""Staged SPORC execution for the JetClass2 salience learned-handoff study."""
from __future__ import annotations

from io import BytesIO
import gc
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
from .contracts import relative_file
from .execution import allocation, gpu_identity, slurm_options
from .model import DelphesParticleTransformer, installed_environment
from .reporting import evaluate_probabilities, recovery
from .salience_learned_cache import prepare_cache
from .salience_learned_campaign import (
    AUTHORIZE, GATE_TASKS, JOB_PREFIX, create_campaign, tasks, validate_campaign,
)
from .salience_learned_contracts import artifact, validate
from .salience_learned_data import (
    IndexedRamCache, MorphCacheManager, PairedRamCache, load_partition,
    publish_partition, WithdrawalCacheManager,
)
from .salience_learned_graph import FIT_ORDER, NODE_REGISTRY, morph_context_for_pass
from .salience_learned_model import (
    DelphesAdjacentFusionParticleTransformer,
    DelphesParameterMatchedSingleViewParticleTransformer,
)
from .salience_learned_training import predict, train_kernel


TERMINAL = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED",
}


def _task_path(spec, task_id):
    if task_id not in {row["task_id"] for row in spec["tasks"]}:
        raise ValueError("Unknown learned-handoff task")
    return Path(spec["campaign_root"]) / "tasks" / f"{task_id}.json"


def completed_task(spec, task_id):
    path = _task_path(spec, task_id)
    if not path.is_file():
        return None
    report = load_json(path)
    validate(report, "TASK_REPORT")
    if (
        report["campaign_sha256"] != spec["content_hash"]
        or report["source_commit"] != spec["source_commit"]
        or report["task_id"] != task_id
        or report["final_test_accessed"] is not False
    ):
        raise ValueError("Learned-handoff task lineage differs")
    root = Path(spec["campaign_root"])
    for output in report["outputs"]:
        target = relative_file(root, output["path"])
        if not target.is_file() or sha256_file(target) != output["sha256"]:
            raise ValueError("Learned-handoff task output checksum differs")
    return report


def _execution_gate(spec, device):
    profile = spec["runtime_profile"]
    if str(device) not in {"cuda", "cuda:0"}:
        raise PermissionError("Learned-handoff science requires one GPU")
    _, cpus, memory = allocation(profile["execution_site"])
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


def _partition_indices(spec, validation):
    report, arrays = load_partition(Path(spec["campaign_root"]) / "validation_partition.json")
    if (
        report["parents"]["campaign_spec"] != spec["content_hash"]
        or not np.array_equal(arrays["identities"], validation.identities)
        or not np.array_equal(arrays["labels"], validation.labels)
    ):
        raise ValueError("Validation partition population/lineage differs")
    return {name: np.flatnonzero(arrays["partition"] == code)
            for code, name in enumerate(("V_checkpoint", "V_diagnostic", "V_report"))}


def _selected_checkpoint(spec, node_id):
    result = completed_task(spec, "train_" + node_id)["result"]
    return relative_file(Path(spec["campaign_root"]), result["checkpoint"]), result


def _new_model(node):
    torch.manual_seed(node["initialization_seed"])
    if node["role"] == "parameter_matched_ce":
        return DelphesParameterMatchedSingleViewParticleTransformer()
    if node["input_protocol"] != "single_view":
        return DelphesAdjacentFusionParticleTransformer(
            context_initialization_seed=node["context_architecture_seed"],
        )
    return DelphesParticleTransformer()


def _load_initial_state(spec, node, model):
    parent = node["initialization_parent"]
    if parent is None:
        return None
    path, result = _selected_checkpoint(spec, parent)
    state = torch.load(path, map_location="cpu", weights_only=True)
    loaded = model.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError("Warm initialization checkpoint differs")
    return {"parent_node": parent, "checkpoint_sha256": sha256_file(path),
            "training_report_sha256": result["training_report_sha256"]}


def _teacher(spec, node, expected_identities):
    distribution = node["teacher_distribution"]
    if distribution is None:
        return None, None
    producer = (
        "reduce_CARRIER_U000" if distribution == "CARRIER_U000"
        else "reduce_MORPH_Q_D000" if distribution == "MORPH_Q_D000"
        else "reduce_" + distribution if distribution.startswith("ACQUISITION_")
        else "extract_" + distribution
    )
    producer_report = completed_task(spec, producer)
    result = producer_report["result"]
    root = Path(spec["campaign_root"])
    probabilities = load_bank(
        relative_file(root, result["train_bank"]),
        foundation_sha256=spec["foundation"]["content_hash"],
        teacher_report_sha256=result["teacher_report_sha256"],
        teacher_node=distribution, role="train",
        expected_identities=expected_identities,
    )
    lineage = {
        "producer_task": producer,
        "producer_task_sha256": producer_report["content_hash"],
        "teacher_report_sha256": result["teacher_report_sha256"],
        "bank_manifest_sha256": result["manifest_sha256"],
    }
    return probabilities, lineage


def _fixed_caches(spec, node):
    train_primary = _cache(spec, "train", node["primary_coordinate"])
    validation_primary = _cache(spec, "validation", node["primary_coordinate"])
    indexes = _partition_indices(spec, validation_primary)
    primary_checkpoint = IndexedRamCache(
        validation_primary, indexes["V_checkpoint"], role="validation",
    )
    primary_report = IndexedRamCache(
        validation_primary, indexes["V_report"], role="validation",
    )
    primary_diagnostic = IndexedRamCache(
        validation_primary, indexes["V_diagnostic"], role="validation",
    )
    if node["input_protocol"] == "single_view":
        return {
            "train": train_primary, "checkpoint": primary_checkpoint,
            "diagnostic": primary_diagnostic, "report": primary_report,
            "train_identities": train_primary.identities,
            "owners": [train_primary, validation_primary],
        }
    context_name = node["context_coordinate"]
    if context_name == "DYNAMIC_U000_TO_D000":
        train_manager = MorphCacheManager(
            train_primary, lambda name: _cache(spec, "train", name),
        )
        validation_manager = MorphCacheManager(
            validation_primary, lambda name: _cache(spec, "validation", name),
        )
        def train_provider(pass_number):
            return train_manager.ensure(pass_number)
        def validation_provider(pass_number):
            pair = validation_manager.ensure(pass_number)
            return PairedRamCache(
                IndexedRamCache(pair.context, indexes["V_checkpoint"], role="validation"),
                primary_checkpoint,
            )
        def report_provider(pass_number):
            pair = validation_manager.ensure(pass_number)
            return PairedRamCache(
                IndexedRamCache(pair.context, indexes["V_report"], role="validation"),
                primary_report,
            )
        def diagnostic_provider(pass_number):
            pair = validation_manager.ensure(pass_number)
            return PairedRamCache(
                IndexedRamCache(pair.context, indexes["V_diagnostic"], role="validation"),
                primary_diagnostic,
            )
        return {
            "train_provider": train_provider,
            "validation_provider": validation_provider,
            "report_provider": report_provider,
            "diagnostic_provider": diagnostic_provider,
            "train_identities": train_primary.identities,
            "owners": [train_manager, validation_manager],
        }
    require_identical = context_name == node["primary_coordinate"]
    if node["role"] in {"fusion_withdrawal", "morph_withdrawal"}:
        train_manager = WithdrawalCacheManager(
            train_primary,
            lambda: train_primary if require_identical else _cache(spec, "train", context_name),
            require_identical=require_identical,
        )
        validation_manager = WithdrawalCacheManager(
            validation_primary,
            lambda: validation_primary if require_identical else _cache(spec, "validation", context_name),
            require_identical=require_identical,
        )

        def validation_provider(pass_number):
            value = validation_manager.ensure(pass_number)
            if isinstance(value, PairedRamCache):
                return PairedRamCache(
                    IndexedRamCache(value.context, indexes["V_checkpoint"], role="validation"),
                    primary_checkpoint, require_identical=require_identical,
                )
            return primary_checkpoint

        def subset_pair(index_name, primary_subset):
            value = validation_manager.paired()
            return PairedRamCache(
                IndexedRamCache(value.context, indexes[index_name], role="validation"),
                primary_subset, require_identical=require_identical,
            )

        return {
            "train_provider": train_manager.ensure,
            "validation_provider": validation_provider,
            "report_provider": lambda _pass: subset_pair("V_report", primary_report),
            "diagnostic_provider": lambda _pass: subset_pair("V_diagnostic", primary_diagnostic),
            "train_identities": train_primary.identities,
            "owners": [train_manager, validation_manager],
        }
    context_train = train_primary if require_identical else _cache(spec, "train", context_name)
    context_validation = validation_primary if require_identical else _cache(spec, "validation", context_name)
    return {
        "train": PairedRamCache(context_train, train_primary, require_identical=require_identical),
        "checkpoint": PairedRamCache(
            IndexedRamCache(context_validation, indexes["V_checkpoint"], role="validation"),
            primary_checkpoint, require_identical=require_identical,
        ),
        "report": PairedRamCache(
            IndexedRamCache(context_validation, indexes["V_report"], role="validation"),
            primary_report, require_identical=require_identical,
        ),
        "diagnostic": PairedRamCache(
            IndexedRamCache(context_validation, indexes["V_diagnostic"], role="validation"),
            primary_diagnostic, require_identical=require_identical,
        ),
        "owners": list({id(value): value for value in (
            train_primary, validation_primary, context_train, context_validation,
        )}.values()),
        "train_identities": train_primary.identities,
    }


def _constant(value):
    return lambda _pass: value


def _route_probabilities(model, pair, *, device, alpha, permute_context=False):
    model.eval()
    result = np.empty((len(pair), 11), np.float32)
    with torch.inference_mode():
        for start in range(0, len(pair), 256):
            indexes = np.arange(start, min(start + 256, len(pair)))
            if alpha == 0.:
                primary = pair.batch_primary(indexes)
                args = tuple(torch.from_numpy(primary[name]).to(device)
                             for name in ("features", "vectors", "mask"))
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=torch.device(device).type == "cuda"):
                    logits = model.forward_fused(*args, alpha=0.).logits
            else:
                raw = pair.batch(indexes)
                context = raw["context"]
                if permute_context:
                    order = np.roll(np.arange(len(indexes)), 1)
                    context = {name: value[order] for name, value in context.items()}
                primary_args = tuple(torch.from_numpy(raw["primary"][name]).to(device)
                                     for name in ("features", "vectors", "mask"))
                context_args = tuple(torch.from_numpy(context[name]).to(device)
                                     for name in ("features", "vectors", "mask"))
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=torch.device(device).type == "cuda"):
                    logits = model.forward_fused(
                        *primary_args, *context_args, alpha=alpha,
                    ).logits
            result[indexes] = torch.softmax(logits.float(), -1).cpu().numpy()
    return result


def _diagnostic(
    model, report_cache, diagnostic_cache, *, node, device,
    training_report_sha256,
):
    if node["input_protocol"] == "single_view":
        probabilities = predict(model, report_cache, node=node, device=device)
        return artifact(
            "DIAGNOSTIC", node_id=node["node_id"],
            parents={"training_report": training_report_sha256},
            selected_route="ordinary", report_metrics=evaluate_probabilities(
                report_cache.labels, probabilities,
            ), alpha_curve=None, context_ablation=None,
            final_test_accessed=False,
        )
    selected = predict(model, report_cache, node=node, device=device)
    alpha_zero = (
        selected if node["selection_route"] == "alpha_zero"
        else _route_probabilities(model, report_cache, device=device, alpha=0.)
    )
    alpha_one = (
        selected if node["selection_route"] == "alpha_one"
        else _route_probabilities(model, report_cache, device=device, alpha=1.)
    )
    curve = []
    for alpha in (0., .25, .5, .75, 1.):
        values = _route_probabilities(model, diagnostic_cache, device=device, alpha=alpha)
        curve.append({"alpha": alpha, "metrics": evaluate_probabilities(diagnostic_cache.labels, values)})
    permuted = _route_probabilities(
        model, diagnostic_cache, device=device, alpha=1., permute_context=True,
    )
    return artifact(
        "DIAGNOSTIC", node_id=node["node_id"],
        parents={"training_report": training_report_sha256},
        selected_route=node["selection_route"],
        report_metrics=evaluate_probabilities(report_cache.labels, selected),
        report_routes={
            "alpha_zero": evaluate_probabilities(report_cache.labels, alpha_zero),
            "alpha_one": evaluate_probabilities(report_cache.labels, alpha_one),
        },
        alpha_curve=curve,
        context_ablation={"identity_permuted": evaluate_probabilities(
            diagnostic_cache.labels, permuted,
        )},
        all_routes_same_report_identities=True, final_test_accessed=False,
    )


def _save_state(path, state):
    buffer = BytesIO()
    torch.save(state, buffer)
    atomic_publish_bytes(path, buffer.getvalue())


def _run_fit(spec, task, attempt_root, device):
    node = NODE_REGISTRY[task["node_id"]].payload()
    caches = _fixed_caches(spec, node)
    model = _new_model(node)
    initialization = _load_initial_state(spec, node, model)
    train_provider = caches.get("train_provider")
    if train_provider is None:
        train_provider = _constant(caches["train"])
    validation_provider = caches.get("validation_provider")
    if validation_provider is None:
        validation_provider = _constant(caches["checkpoint"])
    teacher, teacher_lineage = _teacher(
        spec, node, caches["train_identities"],
    )
    try:
        training, state = train_kernel(
            model, train_provider, validation_provider, node=node, device=device,
            teacher_probabilities=teacher,
            teacher_identities=None if teacher is None else caches["train_identities"],
        )
        base_training = {
            key: value for key, value in training.items()
            if key not in {"contract", "schema_version", "content_hash"}
        }
        training = artifact(
            "TRAINING_REPORT", **base_training,
            parents={
                "campaign_spec": spec["content_hash"],
                "foundation": spec["foundation"]["content_hash"],
            },
            source_commit=spec["source_commit"],
            teacher_lineage=teacher_lineage,
            initialization_lineage=initialization,
        )
        checkpoint = attempt_root / "selected.pt"
        _save_state(checkpoint, state)
        report_provider = caches.get("report_provider")
        report_cache = (
            report_provider(training["selected_pass"])
            if report_provider is not None else caches["report"]
        )
        diagnostic_provider = caches.get("diagnostic_provider")
        diagnostic_cache = (
            diagnostic_provider(training["selected_pass"])
            if diagnostic_provider is not None else caches["diagnostic"]
        )
        diagnostic = _diagnostic(
            model, report_cache, diagnostic_cache, node=node, device=device,
            training_report_sha256=training["content_hash"],
        )
        training_path = attempt_root / "training_report.json"
        diagnostic_path = attempt_root / "diagnostic.json"
        write_immutable_json(training_path, training)
        write_immutable_json(diagnostic_path, diagnostic)
        return {
            "training_report": training_path, "diagnostic": diagnostic_path,
            "checkpoint": checkpoint,
            "result": {
                "training_report": training_path,
                "training_report_sha256": training["content_hash"],
                "diagnostic": diagnostic_path,
                "diagnostic_sha256": diagnostic["content_hash"],
                "checkpoint": checkpoint,
                "checkpoint_sha256": sha256_file(checkpoint),
                "initialization_lineage": initialization,
            },
        }
    finally:
        for owner in caches.get("owners", ()):
            clear = getattr(owner, "clear", None)
            if clear is not None:
                clear()
        caches.clear()
        del model, teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_fit_model(spec, node_id, device):
    node = NODE_REGISTRY[node_id].payload()
    checkpoint, _ = _selected_checkpoint(spec, node_id)
    model = _new_model(node)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval(), node


def _publish_teacher_bank(spec, task, attempt_root, device):
    model, node = _load_fit_model(spec, task["node_id"], device)
    distribution = task["distribution_id"]
    train = _cache(spec, "train", node["primary_coordinate"])
    if node["input_protocol"] == "single_view":
        inference_cache = train
    else:
        context_name = node["context_coordinate"]
        if context_name == "DYNAMIC_U000_TO_D000":
            context_name = "D000"
        context = train if context_name == node["primary_coordinate"] else _cache(spec, "train", context_name)
        inference_cache = PairedRamCache(context, train, require_identical=context_name == node["primary_coordinate"])
    probabilities = predict(
        model, inference_cache, node=node, device=device, temperature=2.,
    )
    source = completed_task(spec, "train_" + task["node_id"])["result"]
    bank_root = attempt_root / "train_bank"
    manifest = publish_bank(
        bank_root, foundation_sha256=spec["foundation"]["content_hash"],
        teacher_report_sha256=source["training_report_sha256"],
        teacher_node=distribution, role="train", identities=train.identities,
        probabilities=probabilities,
    )
    return {
        "manifest": bank_root / "manifest.json",
        "result": {"train_bank": bank_root,
                   "teacher_report_sha256": source["training_report_sha256"],
                   "teacher_node": distribution,
                   "manifest_sha256": manifest["content_hash"]},
    }


def _extract(spec, task, attempt_root, device):
    model, node = _load_fit_model(spec, task["node_id"], device)
    extracted = model.extract_primary().to(device).eval()
    primary = _cache(spec, "validation", node["primary_coordinate"])
    indexes = _partition_indices(spec, primary)["V_report"]
    report_cache = IndexedRamCache(primary, indexes, role="validation")
    pair_context = _cache(spec, "validation", node["context_coordinate"])
    pair = PairedRamCache(
        IndexedRamCache(pair_context, indexes, role="validation"), report_cache,
        require_identical=node["context_coordinate"] == node["primary_coordinate"],
    )
    expected = _route_probabilities(model, pair, device=device, alpha=0.)
    ordinary_node = dict(node, role="cold_single_ce", input_protocol="single_view",
                         selection_route="ordinary", teacher_distribution=None)
    actual = predict(extracted, report_cache, node=ordinary_node, device=device)
    error = float(np.max(np.abs(expected - actual)))
    if error > 2e-5:
        raise RuntimeError("Extracted primary prediction parity differs")
    extraction = artifact(
        "EXTRACTION", source_node=task["node_id"],
        parents={
            "campaign_spec": spec["content_hash"],
            "training_report": completed_task(
                spec, "train_" + task["node_id"],
            )["result"]["training_report_sha256"],
        },
        source_commit=spec["source_commit"],
        distribution_id=task["distribution_id"], max_probability_error=error,
        report_metrics=evaluate_probabilities(report_cache.labels, actual),
        context_parameters_retained=False, ordinary_three_input_model=True,
        final_test_accessed=False,
    )
    checkpoint = attempt_root / "selected.pt"
    _save_state(checkpoint, extracted.state_dict())
    extraction_path = attempt_root / "extraction.json"
    write_immutable_json(extraction_path, extraction)
    children = [
        candidate.payload() for candidate in NODE_REGISTRY.values()
        if candidate.teacher == task["distribution_id"]
    ]
    result = {
        "checkpoint": checkpoint, "checkpoint_sha256": sha256_file(checkpoint),
        "extraction": extraction_path,
        "teacher_report_sha256": extraction["content_hash"],
        "teacher_node": task["distribution_id"],
    }
    outputs = [checkpoint, extraction_path]
    if children:
        train = _cache(spec, "train", node["primary_coordinate"])
        probabilities = predict(
            extracted, train, node=ordinary_node, device=device, temperature=2.,
        )
        bank_root = attempt_root / "train_bank"
        manifest = publish_bank(
            bank_root, foundation_sha256=spec["foundation"]["content_hash"],
            teacher_report_sha256=extraction["content_hash"],
            teacher_node=task["distribution_id"], role="train",
            identities=train.identities, probabilities=probabilities,
        )
        outputs.extend(sorted(bank_root.iterdir()))
        result["train_bank"] = bank_root
        result["manifest_sha256"] = manifest["content_hash"]
    return {"outputs": outputs, "result": result}


def _run_partition(spec, attempt_root):
    validation = _cache(spec, "validation", "D000")
    path = attempt_root / "validation_partition.json"
    report = publish_partition(
        path, identities=validation.identities, labels=validation.labels,
        parents={"campaign_spec": spec["content_hash"],
                 "foundation": spec["foundation"]["content_hash"]},
        source_commit=spec["source_commit"],
    )
    destination = Path(spec["campaign_root"]) / "validation_partition.json"
    data_destination = destination.with_suffix(".npz")
    atomic_publish_bytes(data_destination, path.with_suffix(".npz").read_bytes())
    published = dict(report, data_path=str(data_destination.resolve()))
    published.pop("content_hash")
    published = artifact("VALIDATION_PARTITION", **{
        key: value for key, value in published.items()
        if key not in {"contract", "schema_version"}
    })
    write_immutable_json(destination, published)
    return {"outputs": [path, path.with_suffix(".npz"), destination, data_destination],
            "result": {"partition_sha256": published["content_hash"]}}


def _run_storage(spec, attempt_root):
    root = Path(spec["campaign_root"])
    free = shutil.disk_usage(root).free
    files = [path for path in root.rglob("*") if path.is_file()]
    if any(path.is_symlink() for path in files):
        raise ValueError("Campaign storage contains a symlink")
    if free < spec["minimum_free_disk_bytes"]:
        raise OSError("Insufficient learned-handoff campaign storage headroom")
    report = artifact(
        "STORAGE_AUDIT", campaign_sha256=spec["content_hash"],
        free_bytes=free, existing_bytes=sum(path.stat().st_size for path in files),
        projected_durable_bytes_upper_bound=spec["projected_durable_bytes_upper_bound"],
        particle_views_durable=False, hidden_states_durable=False,
        rolling_resume=False, passed=True, final_test_accessed=False,
    )
    path = attempt_root / "storage_audit.json"
    write_immutable_json(path, report)
    return {"outputs": [path], "result": {"storage_audit_sha256": report["content_hash"]}}


def _run_preflight(spec, attempt_root, device):
    import resource
    _execution_gate(spec, device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    train_primary = _cache(spec, "train", "U050")
    train_context = _cache(spec, "train", "U000")
    validation_primary = _cache(spec, "validation", "U050")
    validation_context = _cache(spec, "validation", "U000")
    validation_indexes = _partition_indices(spec, validation_primary)["V_checkpoint"]
    validation_pair = PairedRamCache(
        IndexedRamCache(validation_context, validation_indexes, role="validation"),
        IndexedRamCache(validation_primary, validation_indexes, role="validation"),
    )
    indexes = np.arange(256)
    primary_raw, context_raw = train_primary.batch(indexes), train_context.batch(indexes)
    node = NODE_REGISTRY["ACQUIRE_COARSE_U050_from_U000"].payload()
    torch.manual_seed(node["initialization_seed"])
    direct = DelphesParticleTransformer().to(device).eval()
    torch.manual_seed(node["initialization_seed"])
    fusion = DelphesAdjacentFusionParticleTransformer(
        context_initialization_seed=node["context_architecture_seed"],
    ).to(device).eval()
    primary_equal = all(
        torch.equal(value, fusion.primary_mod.state_dict()[name.removeprefix("mod.")])
        for name, value in direct.state_dict().items()
    )
    primary = tuple(torch.from_numpy(primary_raw[name]).to(device)
                    for name in ("features", "vectors", "mask"))
    context = tuple(torch.from_numpy(context_raw[name]).to(device)
                    for name in ("features", "vectors", "mask"))
    with torch.inference_mode():
        direct_logits = direct(*primary).float()
        initial = fusion.forward_fused(*primary, *context, alpha=1.).logits.float()
        zero = fusion.forward_fused(*primary, alpha=0.).logits.float()
        extracted = fusion.extract_primary().to(device).eval()
        extracted_logits = extracted(*primary).float()
    initial_error = float((direct_logits - initial).abs().max().cpu())
    zero_error = float((zero - extracted_logits).abs().max().cpu())
    if not primary_equal or initial_error != 0. or zero_error > 2e-5:
        raise RuntimeError("Learned-fusion initialization/extraction acceptance differs")
    optimizer = torch.optim.AdamW([p for p in fusion.parameters() if p.requires_grad], lr=3e-4)
    labels = torch.from_numpy(primary_raw["labels"]).to(device)
    fusion.train(); optimizer.zero_grad(set_to_none=True)
    logits = fusion.forward_fused(*primary, *context, alpha=1.).logits
    loss = torch.nn.functional.cross_entropy(logits.float(), labels)
    loss.backward(); optimizer.step()
    gradients = any(
        injection.residual_projection.weight.grad is not None
        for injection in fusion.injections
    )
    fusion.train(); optimizer.zero_grad(set_to_none=True)
    withdrawal = fusion.forward_withdrawal(
        *primary, *context, alpha=.5,
    )
    teacher = torch.softmax(direct_logits.detach() / 2., dim=-1)
    from hlt_classification.scouting.hcwdl_offline_hlt_withdrawal import (
        withdrawal_loss,
    )
    withdrawal_terms = withdrawal_loss(
        withdrawal, labels, teacher, temperature=2.,
    )
    withdrawal_terms["total"].backward()
    withdrawal_gradients = any(
        injection.residual_projection.weight.grad is not None
        and torch.isfinite(injection.residual_projection.weight.grad).all()
        for injection in fusion.injections
    )
    low_low = PairedRamCache(train_primary, train_primary, require_identical=True)
    low_low_batch = low_low.batch(indexes)
    low_low_equal = all(
        np.array_equal(low_low_batch["context"][name], low_low_batch["primary"][name])
        for name in ("features", "vectors", "mask", "labels", "identities")
    )
    validation_probe = validation_pair.batch(np.arange(min(256, len(validation_pair))))
    validation_pair_equal = np.array_equal(
        validation_probe["context"]["identities"],
        validation_probe["primary"]["identities"],
    )
    morph_probe_bytes = 0
    morph_probe_coordinates = []
    for coordinate_name in ("U004", "D096"):
        probe = _cache(spec, "train", coordinate_name)
        if not np.array_equal(probe.identities, train_primary.identities):
            raise ValueError("Learned-handoff morph probe identity order differs")
        morph_probe_coordinates.append(probe.coordinate_name)
        morph_probe_bytes = max(morph_probe_bytes, probe.nbytes)
        del probe
        gc.collect()
    boundaries = {}
    for pass_number in (1, 2, 26, 27, 51, 100):
        name, exact = morph_context_for_pass(pass_number)
        boundaries[str(pass_number)] = {"coordinate": name,
                                       "u": [exact[0].numerator, exact[0].denominator],
                                       "f": [exact[1].numerator, exact[1].denominator]}
    peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    allocated_cache_bytes = sum(
        value.nbytes for value in (
            train_primary, train_context, validation_primary, validation_context,
        )
    )
    memory_request_bytes = spec["runtime_profile"]["memory_mb"] * 1024**2
    if peak_rss_bytes >= memory_request_bytes:
        raise MemoryError("Learned-handoff preflight exhausted host-memory request")
    report = artifact(
        "EXECUTION_ACCEPTANCE", campaign_sha256=spec["content_hash"],
        device=gpu_identity(), installed_environment=installed_environment(),
        elapsed_seconds=time.monotonic() - started,
        primary_initialization_equal=primary_equal,
        initial_zero_residual_max_abs_error=initial_error,
        alpha_zero_extraction_max_abs_error=zero_error,
        cross_residual_gradients_present=gradients,
        withdrawal_backward_gradients_present=withdrawal_gradients,
        withdrawal_loss_finite=bool(torch.isfinite(withdrawal_terms["total"])),
        low_low_byte_identity=low_low_equal,
        validation_pair_identity_equal=validation_pair_equal,
        materialized_morph_probe_coordinates=morph_probe_coordinates,
        lower_primary_owns_classifier=True, one_way_context=True,
        alpha_zero_context_unavailable=True,
        morph_boundaries=boundaries, morph_reaches_d000_at_pass=51,
        production_batch_size=256, installed_weaver_forward_backward=True,
        peak_cpu_rss_bytes=peak_rss_bytes,
        allocated_cache_array_bytes=allocated_cache_bytes,
        maximum_morph_probe_cache_bytes=morph_probe_bytes,
        requested_memory_bytes=memory_request_bytes,
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        passed=bool(
            gradients and withdrawal_gradients and low_low_equal
            and validation_pair_equal
        ),
        final_test_accessed=False,
    )
    path = attempt_root / "execution_acceptance.json"
    write_immutable_json(path, report)
    destination = Path(spec["campaign_root"]) / "execution_acceptance.json"
    write_immutable_json(destination, report)
    return {"outputs": [path, destination],
            "result": {"execution_acceptance_sha256": report["content_hash"]}}


def validate_science_gate(spec):
    rows = {task: completed_task(spec, task) for task in GATE_TASKS}
    if any(value is None for value in rows.values()):
        raise PermissionError("Learned-handoff science gate is incomplete")
    acceptance = load_json(Path(spec["campaign_root"]) / "execution_acceptance.json")
    validate(acceptance, "EXECUTION_ACCEPTANCE")
    if acceptance["campaign_sha256"] != spec["content_hash"] or acceptance["passed"] is not True:
        raise ValueError("Learned-handoff execution acceptance differs")
    load_partition(Path(spec["campaign_root"]) / "validation_partition.json")
    return {task: rows[task]["content_hash"] for task in GATE_TASKS}


def _publish_task(spec, task_id, attempt_root, payload):
    root = Path(spec["campaign_root"])
    outputs = payload.get("outputs") or [
        value for key, value in payload.items()
        if key != "result" and isinstance(value, Path)
    ]
    outputs = list(dict.fromkeys(Path(value).resolve() for value in outputs))
    if any(
        not path.is_file() or not path.is_relative_to(root.resolve())
        for path in outputs
    ):
        raise ValueError("Learned-handoff task output escaped or is incomplete")
    normalized = {}
    for key, value in payload["result"].items():
        normalized[key] = value.relative_to(root).as_posix() if isinstance(value, Path) else value
    report = artifact(
        "TASK_REPORT", campaign_sha256=spec["content_hash"],
        source_commit=spec["source_commit"], task_id=task_id,
        result=normalized,
        outputs=[{"path": path.relative_to(root.resolve()).as_posix(), "sha256": sha256_file(path)}
                 for path in outputs],
        final_test_accessed=False,
    )
    write_immutable_json(_task_path(spec, task_id), report)
    return report


def run_task(spec, task_id, *, attempt, device="cuda"):
    validate_campaign(spec)
    if re.fullmatch(r"[A-Za-z0-9_-]+", attempt) is None:
        raise ValueError("Unsafe attempt identity")
    registry = {row["task_id"]: row for row in spec["tasks"]}
    if task_id not in registry:
        raise ValueError("Unknown learned-handoff task")
    if (existing := completed_task(spec, task_id)) is not None:
        return existing
    task = registry[task_id]
    for dependency in task["dependencies"]:
        if completed_task(spec, dependency) is None:
            raise ValueError("Missing authenticated parent: " + dependency)
    root = Path(spec["campaign_root"])
    attempt_root = root / "attempts" / task_id / attempt
    attempt_root.mkdir(parents=True, exist_ok=False)
    kind = task["kind"]
    if kind == "authenticate":
        validate_campaign(spec)
        report = artifact("DIAGNOSTIC", kind="authentication",
                          campaign_sha256=spec["content_hash"], passed=True,
                          final_test_accessed=False)
        path = attempt_root / "authentication.json"; write_immutable_json(path, report)
        payload = {"outputs": [path], "result": {"authentication_sha256": report["content_hash"]}}
    elif kind == "partition":
        payload = _run_partition(spec, attempt_root)
    elif kind == "storage":
        payload = _run_storage(spec, attempt_root)
    elif kind == "preflight":
        payload = _run_preflight(spec, attempt_root, device)
    elif kind == "train":
        _execution_gate(spec, device)
        payload = _run_fit(spec, task, attempt_root, device)
        payload["outputs"] = [payload["training_report"], payload["diagnostic"], payload["checkpoint"]]
    elif kind in {"ordinary_reducer", "fusion_reducer"}:
        _execution_gate(spec, device)
        payload = _publish_teacher_bank(spec, task, attempt_root, device)
        payload["outputs"] = sorted((attempt_root / "train_bank").iterdir())
    elif kind == "extract":
        _execution_gate(spec, device)
        payload = _extract(spec, task, attempt_root, device)
    elif kind == "aggregate":
        rows = result_rows(spec)
        if any(row["state"] != "COMPLETE" for row in rows):
            raise ValueError("Aggregate has unfinished registered fits")
        by_id = {row["node_id"]: row for row in rows}

        def delta(left, right):
            left, right = left or {}, right or {}
            return {
                name: None if left.get(name) is None or right.get(name) is None
                else left[name] - right[name]
                for name in ("accuracy", "macro_ovr_auc", "macro_r50", "cross_entropy")
            }

        transitions = []
        for branch, path in spec["graph"]["spines"].items():
            parent = "U000"
            for child in path:
                names = {
                    kind: f"{kind}_{branch}_{child}_from_{parent}"
                    for kind in ("DIRECT", "ACQUIRE", "WITHDRAW")
                }
                transitions.append({
                    "branch": branch, "parent": parent, "child": child,
                    "nodes": names,
                    "acquire_minus_direct": delta(
                        by_id[names["ACQUIRE"]]["validation"],
                        by_id[names["DIRECT"]]["validation"],
                    ),
                    "withdraw_minus_direct": delta(
                        by_id[names["WITHDRAW"]]["validation"],
                        by_id[names["DIRECT"]]["validation"],
                    ),
                    "withdraw_minus_acquire": delta(
                        by_id[names["WITHDRAW"]]["validation"],
                        by_id[names["ACQUIRE"]]["validation"],
                    ),
                })
                parent = child

        def comparison(name, left, right, *, left_route=None, right_route=None):
            left_value = (
                by_id[left]["report_routes"][left_route]
                if left_route else by_id[left]["validation"]
            )
            right_value = (
                by_id[right]["report_routes"][right_route]
                if right_route else by_id[right]["validation"]
            )
            return {"name": name, "left": left, "right": right,
                    "left_route": left_route, "right_route": right_route,
                    "delta": delta(left_value, right_value)}

        control_comparisons = [
            comparison("low_low_minus_parameter_matched_D080",
                       "FUSION_LOW_LOW_D080", "LOW_PARAMETER_MATCHED_D080"),
            comparison("warm_continue_minus_direct_D080", "LOW_WARM_CONTINUE_D080",
                       "DIRECT_DENSE_D080_from_U100"),
            comparison("low_low_minus_parameter_matched_D000",
                       "FUSION_LOW_LOW_D000", "LOW_PARAMETER_MATCHED_D000"),
            comparison("warm_continue_minus_direct_D000", "LOW_WARM_CONTINUE_D000",
                       "DIRECT_DENSE_D000_from_D020"),
            comparison("morph_alpha_one_minus_low_low_D000",
                       "DIRECT_VIEW_MORPH_U000_TO_D000", "FUSION_LOW_LOW_D000",
                       left_route="alpha_one"),
            comparison("static_U000_D000_minus_morph_alpha_one",
                       "STATIC_U000_D000", "DIRECT_VIEW_MORPH_U000_TO_D000",
                       left_route="alpha_one", right_route="alpha_one"),
            comparison("morph_alpha_zero_minus_single_D000",
                       "DIRECT_VIEW_MORPH_U000_TO_D000", "CE_SINGLE_D000",
                       left_route="alpha_zero"),
            comparison("morph_withdraw_minus_morph_alpha_zero",
                       "DIRECT_VIEW_MORPH_WITHDRAW_D000",
                       "DIRECT_VIEW_MORPH_U000_TO_D000",
                       right_route="alpha_zero"),
        ]
        extraction_summaries = []
        for registered in spec["tasks"]:
            if registered["kind"] != "extract":
                continue
            pointer = completed_task(spec, registered["task_id"])
            extraction = load_json(relative_file(
                Path(spec["campaign_root"]), pointer["result"]["extraction"],
            ))
            validate(extraction, "EXTRACTION")
            extraction_summaries.append({
                "task_id": registered["task_id"],
                "source_node": registered["node_id"],
                "distribution_id": registered["distribution_id"],
                "max_probability_error": extraction["max_probability_error"],
                "extracted_minus_selected": delta(
                    extraction["report_metrics"],
                    by_id[registered["node_id"]]["validation"],
                ),
            })
        report = artifact(
            "AGGREGATE", campaign_sha256=spec["content_hash"], rows=rows,
            fit_count=54, control_count=10,
            transition_comparisons=transitions,
            control_comparisons=control_comparisons,
            extraction_summaries=extraction_summaries,
            scientific_result_does_not_control_completion=True,
            final_test_accessed=False,
        )
        path = attempt_root / "validation_aggregate.json"; write_immutable_json(path, report)
        payload = {"outputs": [path], "result": {"aggregate_sha256": report["content_hash"],
                                                  "aggregate": path}}
    else:
        report = artifact(
            "CAMPAIGN_COMPLETE", campaign_sha256=spec["content_hash"],
            aggregate_task_sha256=completed_task(spec, "aggregate")["content_hash"],
            fresh_fit_count=54, scientific_result_does_not_control_completion=True,
            final_test_accessed=False,
        )
        path = attempt_root / "campaign_complete.json"; write_immutable_json(path, report)
        payload = {"outputs": [path], "result": {"complete_sha256": report["content_hash"],
                                                  "campaign_complete": path}}
    return _publish_task(spec, task_id, attempt_root, payload)


def result_rows(spec):
    validate_campaign(spec)
    rows, metrics, reports, diagnostics = [], {}, {}, {}
    for node_id in FIT_ORDER:
        pointer = completed_task(spec, "train_" + node_id)
        diagnostic = None
        if pointer is not None:
            diagnostic = load_json(relative_file(
                Path(spec["campaign_root"]), pointer["result"]["diagnostic"],
            ))
            validate(diagnostic, "DIAGNOSTIC")
            if diagnostic["parents"]["training_report"] != pointer["result"]["training_report_sha256"]:
                raise ValueError("Learned-handoff diagnostic parent differs")
            metrics[node_id] = diagnostic["report_metrics"]
            diagnostics[node_id] = diagnostic
            report = load_json(relative_file(
                Path(spec["campaign_root"]), pointer["result"]["training_report"],
            ))
            validate(report, "TRAINING_REPORT")
            if (
                report["parents"] != {
                    "campaign_spec": spec["content_hash"],
                    "foundation": spec["foundation"]["content_hash"],
                }
                or report["source_commit"] != spec["source_commit"]
                or report["node"]["node_id"] != node_id
            ):
                raise ValueError("Learned-handoff training-report lineage differs")
            reports[node_id] = report
    baseline, oracle = metrics.get("M0HLT"), metrics.get("U000")
    for node_id in FIT_ORDER:
        node = NODE_REGISTRY[node_id]
        value = metrics.get(node_id)
        rows.append({
            "node_id": node_id, "role": node.role, "branch": node.branch,
            "coordinate": node.primary,
            "state": "COMPLETE" if value is not None else "PENDING",
            "selected_pass": None if node_id not in reports else reports[node_id]["selected_pass"],
            "passes": None if node_id not in reports else reports[node_id]["passes"],
            "validation": value,
            "report_routes": None if node_id not in diagnostics else diagnostics[node_id].get("report_routes"),
            "alpha_curve": None if node_id not in diagnostics else diagnostics[node_id].get("alpha_curve"),
            "context_ablation": None if node_id not in diagnostics else diagnostics[node_id].get("context_ablation"),
            "recovery": None if value is None or baseline is None or oracle is None
            else recovery(value, baseline, oracle),
        })
    return rows


def command_plan(spec, *, stage="full", selected_tasks=None):
    validate_campaign(spec)
    if stage not in {"full", "gate", "science"}:
        raise ValueError("Submission stage differs")
    known = {row["task_id"]: row for row in spec["tasks"]}
    chosen = (
        list(known) if stage == "full"
        else list(GATE_TASKS) if stage == "gate"
        else [name for name in known if name not in GATE_TASKS]
    )
    if selected_tasks is not None:
        chosen = list(selected_tasks)
    if len(chosen) != len(set(chosen)) or not set(chosen) <= set(known):
        raise ValueError("Learned-handoff command coverage differs")
    if stage == "science":
        validate_science_gate(spec)
    profile = spec["runtime_profile"]
    rows = []
    for task in spec["tasks"]:
        task_id = task["task_id"]
        if task_id not in chosen:
            continue
        dependencies = [parent for parent in task["dependencies"] if parent in chosen]
        for parent in task["dependencies"]:
            if parent not in chosen and parent not in GATE_TASKS and completed_task(spec, parent) is None:
                raise ValueError("Omitted learned-handoff dependency is incomplete")
        resource = spec["resources"][task["resource"]]
        command = slurm_options(profile["execution_site"]) + [
            f"--cpus-per-task={resource['cpus']}", f"--mem={resource['memory_mb']}M",
            f"--time={resource['minutes']}", f"--job-name={JOB_PREFIX}_{task_id}",
            "--chdir=" + spec["project_dir"],
            "--output=" + str(Path(spec["campaign_root"]) / "slurm-%j.out"),
        ]
        if resource["gpu"]:
            command.append("--gres=" + profile["execution_site"]["gres"])
        if dependencies:
            command.append("--dependency=afterok:" + ":".join(
                "${JOB_" + parent + "}" for parent in dependencies
            ))
        command += [
            str(Path(spec["project_dir"]) / "sbatch/run_jetclass2_delphes_salience_learned.sh"),
            spec["project_dir"], str(Path(spec["campaign_root"]) / "campaign_spec.json"),
            task_id,
        ]
        rows.append({"task_id": task_id, "dependencies": dependencies,
                     "command": command})
    return artifact(
        "COMMAND_PLAN", campaign_sha256=spec["content_hash"], stage=stage,
        commands=rows, restart_from_zero=True, single_gpu_per_science_fit=True,
        final_test_accessed=False,
    )


def submit(
    spec, *, stage, execute, authorization_phrase=None,
    selected_tasks=None, bookkeeping_root=None,
):
    validate_campaign(spec)
    campaign_root = Path(spec["campaign_root"]).resolve()
    root = (
        campaign_root / f"submissions_{stage}"
        if bookkeeping_root is None else Path(bookkeeping_root).resolve()
    )
    if not root.is_relative_to(campaign_root):
        raise ValueError("Learned-handoff submission bookkeeping escaped campaign")
    if root != campaign_root / f"submissions_{stage}":
        recovery = load_json(root / "recovery.json")
        validate(recovery, "RECOVERY")
        if (
            recovery["campaign_sha256"] != spec["content_hash"]
            or recovery["source_commit"] != spec["source_commit"]
            or recovery["stage"] != stage
        ):
            raise ValueError("Learned-handoff recovery subject differs")
        selected_tasks = recovery["remaining_tasks"]
    plan = command_plan(spec, stage=stage, selected_tasks=selected_tasks)
    plan_path = root / "command_plan.json"
    if not plan_path.exists():
        write_immutable_json(plan_path, plan)
    elif load_json(plan_path) != plan:
        raise ValueError("Immutable learned-handoff command plan differs")
    dry = root / "dry_run_submission_ledger.json"
    claim = root / "submission_in_progress.claim"
    descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        if not execute:
            return submit_exact_dag(
                identity=spec["content_hash"], plan=plan, output=dry,
                canonical_dry_run=dry, execute=False,
            )
        if authorization_phrase != AUTHORIZE:
            raise PermissionError("Exact learned-handoff authorization phrase required")
        if stage == "full":
            raise PermissionError(
                "Live full-stage submission is forbidden; complete gate, then submit science"
            )
        if not dry.is_file():
            raise ValueError("Exact learned-handoff dry run is required")
        if shutil.disk_usage(root).free < spec["minimum_free_disk_bytes"]:
            raise OSError("Insufficient learned-handoff storage headroom")
        from .submission import _guarded_exact_submission
        return _guarded_exact_submission(spec, plan, root)
    finally:
        claim.unlink()


def monitor(spec, ledger):
    validate_submission_ledger(ledger)
    if ledger["campaign_spec_sha256"] != spec["content_hash"] or ledger["dry_run"]:
        raise ValueError("Monitor requires this campaign's live ledger")
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
        "MONITOR", campaign_sha256=spec["content_hash"],
        rows=[{"task_id": task_id, "job_id": job_id,
               "state": states.get(job_id, "UNKNOWN"),
               "outputs_complete": completed_task(spec, task_id) is not None}
              for task_id, job_id in ledger["jobs"].items()],
        final_test_accessed=False,
    )


def prepare_recovery(spec, ledger, *, output_root):
    observed = monitor(spec, ledger)
    if any(row["state"] not in TERMINAL for row in observed["rows"]):
        raise PermissionError("Recovery requires all old jobs terminal")
    submitted = set(ledger["jobs"])
    if submitted <= set(GATE_TASKS):
        stage = "gate"
    elif submitted.isdisjoint(GATE_TASKS):
        stage = "science"
    else:
        raise ValueError("Mixed-stage learned-handoff ledger cannot be recovered")
    remaining = [row["task_id"] for row in spec["tasks"]
                 if row["task_id"] in submitted
                 and completed_task(spec, row["task_id"]) is None]
    root = Path(output_root).resolve()
    if root.exists() or not root.is_relative_to(Path(spec["campaign_root"]).resolve()) or not remaining:
        raise ValueError("Invalid learned-handoff recovery root/tasks")
    result = artifact(
        "RECOVERY", campaign_sha256=spec["content_hash"],
        source_commit=spec["source_commit"], stage=stage,
        remaining_tasks=remaining,
        restart_from_zero=True, completed_parents_reused=True,
        final_test_accessed=False,
    )
    write_immutable_json(root / "recovery.json", result)
    write_immutable_json(
        root / "command_plan.json",
        command_plan(spec, stage=stage, selected_tasks=remaining),
    )
    return result


__all__ = [
    "command_plan", "completed_task", "create_campaign", "monitor",
    "prepare_recovery", "result_rows", "run_task", "submit",
    "validate_science_gate",
]

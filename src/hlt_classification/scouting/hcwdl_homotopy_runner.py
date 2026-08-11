"""Execute one authenticated HCWDL structural-feature homotopy fit."""

from __future__ import annotations

import gc
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, require_sha256, sha256_file, validate_content_hash,
    with_content_hash, write_immutable_json,
)
from hlt_classification.models.scouting_particle_transformer import (
    build_scouting_particle_transformer,
)

from .dataset import iterate_model_batches
from .engine import precompute_teacher_targets, validate_pmard_training_report
from .hcwdl_homotopy import HomotopyCoordinate
from .hcwdl_homotopy_contracts import (
    CACHE_MINIATURE_CONTRACT, GRAPH_RECIPE_LOCK_CONTRACT,
    NODE_RUNTIME_CONTRACT, TRAINING_REPORT_CONTRACT,
    validate_coordinate, validate_coupling_config,
)
from .hcwdl_homotopy_graph import (
    DOMAINS, GRAPH_LABEL, GRAPH_SHA256, NODE_REGISTRY, resolved_loss,
    validate_recipe_overlay,
)
from .hcwdl_homotopy_stream import iterate_homotopy_batches
from .hcwdl_homotopy_locks import (
    validate_endpoint_equality_lock, validate_graph_recipe_lock,
)
from .hcwdl_toff_targets import (
    DurableToffTargets, validate_toff_target_lock, validate_toff_target_manifest,
)
from .hcwdl_training import train_hcwdl_node, validate_completed_hcwdl_node
from .hcwdl_upper_cache import (
    ResidualCouplingStore, validate_base_manifest, validate_coupling_audit,
    validate_coupling_lock, validate_coupling_manifest,
)
from .hcwdl_upper_coupling import (
    validate_scale_calibration, validate_switch_calibration,
)
from .highcov_cache import DenseAssignmentStore
from .loaders import load_pmard_model, scouting_model_factory_for_report
from .selective_assignment import RowSelection
from .splits import role_records
from .training import derive_seed
from .view_cache import EphemeralPmardViewCache, expected_cache_source_rows


def node_output_dir(campaign_root: str | Path, node_id: str) -> Path:
    root = Path(campaign_root); node = NODE_REGISTRY[node_id]
    if node.track == "factorized":
        return root / "training/factorized" / node_id
    if node.track == "joint":
        return root / "training/joint" / node_id
    if node.track == "stationary_d100":
        return root / "controls/stationary_d100" / node_id
    if node.track == "stationary_hlt":
        return root / "controls/stationary_hlt" / node_id
    if node_id in {"D100direct", "D0direct", "M0self"}:
        return root / "controls/direct" / node_id
    return root / "controls" / node_id


def _coordinate(domain: str) -> HomotopyCoordinate:
    row = DOMAINS[domain]
    s, f = row.get("s"), row.get("f")
    if s is None or f is None:
        raise ValueError(f"domain {domain!r} has no homotopy coordinate")
    # Every registered value is an exact multiple of 1/20.
    return HomotopyCoordinate(round(float(s) * 20), 20, round(float(f) * 20), 20)


def _memory_limit_bytes(configured_gib: float) -> int:
    configured = int(float(configured_gib) * 1024**3)
    slurm = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm:
        try:
            # Slurm exposes this value in MiB on Tigris.
            configured = min(configured, int(slurm) * 1024**2 * 3 // 4)
        except ValueError as exc:
            raise ValueError("SLURM_MEM_PER_NODE is not an integer MiB value") from exc
    return configured


def estimate_global_peak_bytes(
    *, miniature: Mapping[str, Any], train_rows: int, validation_rows: int,
) -> dict[str, int]:
    """Conservative pre-allocation estimate for all simultaneously live state."""

    sample_rows = miniature.get("sample_rows", {})
    sample_bytes = miniature.get("sample_array_bytes_by_view", {})
    expected_keys = {
        f"{role}:{view}"
        for role in ("train", "validation")
        for view in ("p0", "u010", "j005", "u100", "j100")
    }
    if set(sample_rows) != expected_keys or set(sample_bytes) != expected_keys:
        raise ValueError("HCWDL-UJ cache miniature view registry differs")
    if any(int(value) <= 0 for value in sample_rows.values()):
        raise ValueError("HCWDL-UJ cache miniature row count differs")
    if any(int(value) <= 0 for value in sample_bytes.values()):
        raise ValueError("HCWDL-UJ cache miniature byte count differs")
    student = 0
    for role, expected_rows in (("train", train_rows), ("validation", validation_rows)):
        per_row = max(
            (int(sample_bytes[key]) + int(sample_rows[key]) - 1)
            // int(sample_rows[key])
            for key in expected_keys if key.startswith(role + ":")
        )
        student += per_row * int(expected_rows)
    identities_labels = (int(train_rows) + int(validation_rows)) * 160
    logits = int(train_rows) * 15 * 4
    # ParT+Adam/resume, the largest ROOT/teacher batch, and atomic publication.
    model_optimizer = 8 * 1024**3
    transient_io = 3 * 1024**3
    subtotal = student + identities_labels + logits + model_optimizer + transient_io
    allocator = (subtotal + 3) // 4
    return {
        "student_cache_bytes": student,
        "identity_label_bytes": identities_labels,
        "teacher_logit_bytes": logits,
        "model_optimizer_resume_bytes": model_optimizer,
        "root_teacher_publication_bytes": transient_io,
        "allocator_overhead_bytes": allocator,
        "global_peak_bytes": subtotal + allocator,
    }


def run_homotopy_node(
    *, spec: Mapping[str, Any], node_id: str, device: str = "cuda",
    view_cache_max_gib: float = 96.0,
) -> dict[str, Any]:
    """Build student views/targets once, then run the generic exact engine."""

    from .hcwdl_homotopy_campaign import validate_campaign

    started = time.monotonic()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:  # pragma: no cover - live training requires torch
        pass
    try:
        import resource
        initial_peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except ImportError:  # pragma: no cover - Windows contract tests
        initial_peak_rss_kib = 0
    validate_campaign(spec, executable=False)
    if node_id not in NODE_REGISTRY:
        raise ValueError("unknown HCWDL-UJ node")
    root = Path(spec["campaign_root"]); node = NODE_REGISTRY[node_id]
    graph_lock = load_json(root / "locks/graph_recipe_lock.json")
    graph_lock_hash = validate_graph_recipe_lock(
        graph_lock, campaign_spec_sha256=spec["content_hash"],
    )
    recipe = load_json(spec["recipe_path"])
    overlay = load_json(root / "recipe_overlay.json")
    overlay_hash = validate_recipe_overlay(
        overlay, parent_recipe_sha256=spec["recipe_sha256"],
    )
    if overlay_hash != spec["recipe_overlay_sha256"]:
        raise ValueError("HCWDL-UJ overlay differs from campaign spec")
    overlay_row = {str(row["node_id"]): row for row in overlay["rows"]}[node_id]
    loss = resolved_loss(node_id)
    if overlay_row.get("loss") != __import__("dataclasses").asdict(loss):
        raise ValueError("HCWDL-UJ resolved node loss differs from overlay")

    split = load_json(spec["split_manifest_path"])
    split_hash = validate_content_hash(
        split, expected_contract=str(split["contract"]),
        expected_schema_version=int(split["schema_version"]),
    )
    selection_raw = load_json(spec["selection_manifest_path"])
    selection_hash = validate_content_hash(
        selection_raw, expected_contract=str(selection_raw["contract"]),
        expected_schema_version=int(selection_raw["schema_version"]),
    )
    selections = {
        role: RowSelection(selection_raw, role=role, split_manifest_sha256=split_hash)
        for role in ("train", "validation")
    }
    assignments = {
        role: DenseAssignmentStore(spec["assignment_manifests"][role])
        for role in ("train", "validation")
    }
    config = load_json(root / "coupling/config.json")
    config_hash = validate_coupling_config(config)
    if config_hash != spec["coupling_config_sha256"]:
        raise ValueError("HCWDL-UJ coupling configuration/spec lineage differs")
    scale = load_json(root / "coupling/scale_calibration.json")
    scale_hash = validate_scale_calibration(
        scale, coupling_config_sha256=config_hash,
    )
    train_base = load_json(root / "coupling/train_base_manifest.json")
    train_base_hash = validate_base_manifest(train_base, role="train")
    switch = load_json(root / "coupling/switch_calibration.json")
    switch_hash = validate_switch_calibration(
        switch, coupling_config_sha256=config_hash,
        train_base_manifest_sha256=train_base_hash,
    )
    couplings = {
        role: ResidualCouplingStore(root / f"coupling/{role}_manifest.json")
        for role in ("train", "validation")
    }
    coupling_manifest_hashes = {}
    for role, store in couplings.items():
        coupling_manifest_hashes[role] = validate_coupling_manifest(
            store.manifest, role=role,
        )
        if int(store.manifest.get("rows", -1)) != int(spec["role_counts"][role]):
            raise ValueError(f"HCWDL-UJ {role} coupling-manifest coverage differs")
    audit = load_json(root / "coupling/full_role_audit.json")
    audit_hash = validate_coupling_audit(audit)
    if (
        audit.get("coupling_config_sha256") != config_hash
        or audit.get("switch_calibration_sha256") != switch_hash
        or audit.get("train_manifest_sha256") != coupling_manifest_hashes["train"]
        or audit.get("validation_manifest_sha256") != coupling_manifest_hashes["validation"]
        or audit.get("expected_rows") != {
            role: int(spec["role_counts"][role]) for role in ("train", "validation")
        }
    ):
        raise ValueError("HCWDL-UJ coupling audit lineage differs")
    expected_coupling_lock = {
        "coupling_config_sha256": config_hash,
        "scale_calibration_sha256": scale_hash,
        "switch_calibration_sha256": switch_hash,
        "train_manifest_sha256": coupling_manifest_hashes["train"],
        "validation_manifest_sha256": coupling_manifest_hashes["validation"],
        "audit_sha256": audit_hash,
    }
    coupling_lock = load_json(root / "locks/coupling_lock.json")
    coupling_lock_hash = validate_coupling_lock(
        coupling_lock, campaign_spec_sha256=spec["content_hash"],
        expected=expected_coupling_lock,
    )
    miniature = load_json(root / "runtime/cache_miniature.json")
    miniature_hash = validate_content_hash(
        miniature, expected_contract=CACHE_MINIATURE_CONTRACT,
        expected_schema_version=1,
    )
    if (
        miniature.get("campaign_spec_sha256") != spec["content_hash"]
        or miniature.get("durable_repaired_dataset") is not False
        or miniature.get("matcher_callable_present") is not False
        or miniature.get("final_test_accessed") is not False
    ):
        raise ValueError("HCWDL-UJ cache miniature lineage/access differs")
    coordinate = load_json(root / "coordinate_table.json")
    coordinate_hash = validate_coordinate(coordinate)
    if coordinate_hash != spec["coordinate_sha256"]:
        raise ValueError("HCWDL-UJ coordinate table/spec lineage differs")
    # Hash the implementation imported by this worker, not the original
    # campaign worktree path.  This is equal during ordinary execution and
    # deliberately fail-closes a source-recovery commit that changes endpoint
    # projection or Shell Exact science while claiming to be execution-only.
    repair_hash = sha256_file(Path(__file__).with_name("repair.py"))
    if (
        config.get("projection_sha256") != repair_hash
        or config.get("shell_exact_sha256") != repair_hash
    ):
        raise ValueError("HCWDL-UJ endpoint implementation drifted after campaign creation")
    expected_endpoint_lock = {
        "coupling_lock_sha256": coupling_lock_hash,
        "full_role_audit_sha256": audit_hash,
        "cache_miniature_sha256": miniature_hash,
        "coordinate_sha256": coordinate_hash,
        "projection_sha256": repair_hash,
        "shell_parity_sha256": canonical_sha256({
            "public_family": "HIGHCOV_SHELL_EXACT/v1",
            "continuous_alpha": True,
        }),
    }
    endpoint_lock = load_json(root / "locks/endpoint_equality_lock.json")
    endpoint_lock_hash = validate_endpoint_equality_lock(
        endpoint_lock, campaign_spec_sha256=spec["content_hash"],
        expected=expected_endpoint_lock,
    )
    toff_manifest = load_json(root / "targets/toff_train/manifest.json")
    toff_manifest_hash = validate_toff_target_manifest(toff_manifest)
    imported_toff = spec["imported_controls"]["TOFF"]
    expected_toff_lock = {
        "manifest_sha256": toff_manifest_hash,
        "teacher_report_sha256": imported_toff["report_sha256"],
        "teacher_checkpoint_sha256": imported_toff["checkpoint_sha256"],
        "split_manifest_sha256": split_hash,
        "selection_manifest_sha256": selection_hash,
        "native_adapter_sha256": toff_manifest["parents"]["native_adapter_sha256"],
        "input_projection_sha256": toff_manifest["parents"]["input_projection_sha256"],
        "inference_policy_sha256": toff_manifest["parents"]["inference_policy_sha256"],
    }
    if (
        int(toff_manifest.get("rows", -1)) != selections["train"].rows
        or any(toff_manifest.get("parents", {}).get(name) != value for name, value in {
            "campaign_spec_sha256": spec["content_hash"],
            "split_manifest_sha256": split_hash,
            "selection_manifest_sha256": selection_hash,
            "teacher_report_sha256": imported_toff["report_sha256"],
            "teacher_checkpoint_sha256": imported_toff["checkpoint_sha256"],
        }.items())
    ):
        raise ValueError("HCWDL-UJ TOFF target manifest lineage differs")
    toff_lock = load_json(root / "targets/toff_train/lock.json")
    toff_lock_hash = validate_toff_target_lock(
        toff_lock, campaign_spec_sha256=spec["content_hash"],
        expected=expected_toff_lock,
    )
    expected_graph_lock = {
        "endpoint_equality_lock_sha256": endpoint_lock_hash,
        "toff_target_lock_sha256": toff_lock_hash,
        "graph_artifact_sha256": spec["graph_artifact_sha256"],
        "graph_semantic_sha256": spec["graph_sha256"],
        "recipe_overlay_sha256": spec["recipe_overlay_sha256"],
        "parent_recipe_sha256": spec["recipe_sha256"],
        "coordinate_sha256": spec["coordinate_sha256"],
        "command_plan_sha256": spec["command_plan_sha256"],
        "source_commit_sha256": canonical_sha256(spec["source_commit"]),
        "weaver_parity_sha256": spec["weaver_parity_sha256"],
    }
    if any(graph_lock.get(name) != value for name, value in expected_graph_lock.items()):
        raise ValueError("HCWDL-UJ graph/recipe lock lineage differs")

    teacher_hash = None
    teacher_raw = None
    teacher_path = None
    if node.teachers:
        teacher = node.teachers[0]
        if teacher.node_id == "TOFF":
            if node_id not in toff_lock["consumers"]:
                raise PermissionError("HCWDL-UJ node is not an authorized TOFF-target consumer")
            teacher_hash = str(spec["imported_controls"]["TOFF"]["report_sha256"])
        else:
            teacher_path = node_output_dir(root, teacher.node_id) / "training_report.json"
            teacher_raw = load_json(teacher_path)
            teacher_hash = validate_pmard_training_report(teacher_raw)
            checkpoint = teacher_path.parent / str(teacher_raw.get("selected_checkpoint"))
            if (
                not checkpoint.is_file()
                or sha256_file(checkpoint) != teacher_raw.get("selected_checkpoint_sha256")
            ):
                raise ValueError("HCWDL-UJ teacher selected checkpoint differs")

    parents = {
        "campaign_spec_sha256": spec["content_hash"],
        "split_manifest_sha256": split_hash,
        "selection_manifest_sha256": selection_hash,
        "coupling_lock_sha256": coupling_lock_hash,
        "endpoint_equality_lock_sha256": endpoint_lock_hash,
        "graph_recipe_lock_sha256": graph_lock_hash,
        "coordinate_sha256": spec["coordinate_sha256"],
    }
    if teacher_hash is None:
        parents["root_marker_sha256"] = require_sha256(
            spec["graph_artifact_sha256"], name="CE root marker",
        )
    else:
        parents["teacher_sole_report_sha256"] = require_sha256(
            teacher_hash, name="teacher report",
        )
    output = node_output_dir(root, node_id)
    completed_parents = dict(parents)
    completed_parents["recipe_overlay"] = overlay_hash
    completed = validate_completed_hcwdl_node(
        output, node_id=node_id, expected_campaign=GRAPH_LABEL,
        expected_graph_sha256=GRAPH_SHA256,
        expected_node_payload=node.payload(),
        expected_recipe_sha256=spec["recipe_sha256"],
        expected_parents=completed_parents,
        report_contract=TRAINING_REPORT_CONTRACT,
    )
    if completed is not None:
        engine_report = load_json(completed[0])
        report = load_json(completed[1])
        runtime = load_json(output / "runtime.json")
        validate_content_hash(
            runtime, expected_contract=NODE_RUNTIME_CONTRACT,
            expected_schema_version=1,
        )
        if (
            runtime.get("campaign_spec_sha256") != spec["content_hash"]
            or runtime.get("node_id") != node_id
            or runtime.get("training_report_sha256") != report["content_hash"]
            or runtime.get("pmard_engine_report_sha256")
               != engine_report["content_hash"]
            or runtime.get("final_test_accessed") is not False
        ):
            raise ValueError("completed HCWDL-UJ node runtime lineage differs")
        return report
    batch_size = int(recipe["batching"]["effective_batch_size"])
    alias = node.seed_alias
    sampler_seed = derive_seed(int(spec["replicate_seed"]), f"hcwdl_uj/sampler/{alias}")
    repair_seed = derive_seed(int(spec["replicate_seed"]), "hcwdl_uj/repair/shared_v1")

    def online_stream(domain: str, role: str, epoch: int = 0):
        if domain in {"hlt", "toff"}:
            return iterate_model_batches(
                split, data_root=spec["data_root"], role=role,
                input_mode=domain, epoch=epoch, batch_size=batch_size,
                sampler_seed=sampler_seed, row_selection=selections[role],
            )
        return iterate_homotopy_batches(
            split, data_root=spec["data_root"], role=role,
            assignment_store=assignments[role], coupling_store=couplings[role],
            row_selection=selections[role], coordinate=_coordinate(domain),
            repair_seed=repair_seed, batch_size=batch_size,
            output_key=str(DOMAINS[domain]["input"]),
        )

    student_domain = node.student_domain
    student_key = str(DOMAINS[student_domain]["input"])
    caches: dict[str, EphemeralPmardViewCache] = {}
    memory_limit = _memory_limit_bytes(view_cache_max_gib)
    estimate = estimate_global_peak_bytes(
        miniature=miniature, train_rows=selections["train"].rows,
        validation_rows=selections["validation"].rows,
    )
    if estimate["global_peak_bytes"] > memory_limit:
        raise MemoryError(
            "HCWDL-UJ global cache/model/target peak estimate exceeds the safe memory limit"
        )
    remaining_bytes = memory_limit
    for role in ("train", "validation"):
        records = role_records(split, role)
        cache = EphemeralPmardViewCache.build(
            online_stream(student_domain, role), expected_rows=selections[role].rows,
            records=records, role=role,
            expected_source_rows=expected_cache_source_rows(records, row_selection=selections[role]),
            view_keys=(student_key,), max_gib=remaining_bytes / 1024**3,
            lineage={
                "campaign_spec_sha256": spec["content_hash"],
                "coupling_lock_sha256": coupling_lock_hash,
                "endpoint_equality_lock_sha256": endpoint_lock_hash,
                "coordinate_sha256": spec["coordinate_sha256"],
                "domain": student_domain, "durable_repaired_dataset": False,
                "matcher_callable_present": False,
                "global_peak_estimate": estimate,
                "safe_memory_limit_bytes": memory_limit,
            },
        )
        caches[role] = cache
        remaining_bytes -= int(cache.header["array_bytes"])
        if remaining_bytes <= 0:
            raise MemoryError("HCWDL-UJ simultaneous train/validation caches exceed global cap")

    def student_batches(role: str, epoch: int = 0):
        return caches[role].iterate_batches(
            epoch=epoch, sampler_seed=sampler_seed, batch_size=batch_size,
        )

    targets = None
    if node.teachers:
        teacher = node.teachers[0]
        if teacher.node_id == "TOFF":
            imported = spec["imported_controls"]["TOFF"]
            targets = DurableToffTargets(root / "targets/toff_train/manifest.json").as_ephemeral(
                teacher_report_sha256=teacher_hash, split_manifest_sha256=split_hash,
            )
        else:
            assert teacher_path is not None and teacher_raw is not None
            teacher_model, teacher_report = load_pmard_model(
                teacher_path, model_factory=scouting_model_factory_for_report(teacher_raw),
                device=device,
            )
            if teacher_report["content_hash"] != teacher_hash:
                raise ValueError("HCWDL-UJ teacher report changed during load")
            targets = precompute_teacher_targets(
                teacher_model, online_stream(teacher.domain, "train", 0),
                input_key=str(DOMAINS[teacher.domain]["input"]), device=device,
                teacher_report_sha256=teacher_hash,
                split_manifest_sha256=split_hash,
            )
            del teacher_model; gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    teacher_domain = None if not node.teachers else node.teachers[0].domain
    report = train_hcwdl_node(
        node_id=node_id, recipe=recipe, train_rows=selections["train"].rows,
        replicate_seed=int(spec["replicate_seed"]),
        model_factory=build_scouting_particle_transformer,
        train_batches=lambda epoch: student_batches("train", epoch),
        validation_batches=lambda: student_batches("validation", 0),
        class_weights=np.ones(15, np.float32), output_dir=output,
        parents=parents, device=device,
        hlt_teacher_targets=(targets if teacher_domain == "hlt" else None),
        privileged_teacher_targets=(targets if teacher_domain not in {None, "hlt"} else None),
        smoke=spec["mode"] == "smoke", registry=NODE_REGISTRY, domains=DOMAINS,
        graph_sha256=GRAPH_SHA256, report_contract=TRAINING_REPORT_CONTRACT,
        campaign_label=GRAPH_LABEL, scientific_config_extra={
            "final_test_accessed": False, "student_view_built_once": True,
            "teacher_targets_built_once": bool(node.teachers),
            "seed_alias": alias, "coordinate": _coordinate(student_domain).payload()
                if student_domain not in {"hlt", "toff"} else None,
            "durable_repaired_dataset": False,
        }, seed_node_id=alias, node_contract="HCWDL_STRUCTURAL_FEATURE_NODE_SPEC/v1",
        explicit_loss=loss, recipe_overlay_sha256=overlay_hash,
    )
    engine_report = load_json(output / "training_report.json")
    engine_report_hash = validate_pmard_training_report(engine_report)
    if report.get("pmard_engine_report_sha256") != engine_report_hash:
        raise ValueError("HCWDL-UJ wrapper/engine report lineage differs")
    elapsed = time.monotonic() - started
    peak_gpu_allocated = peak_gpu_reserved = 0
    try:
        import torch
        if torch.cuda.is_available():
            peak_gpu_allocated = int(torch.cuda.max_memory_allocated())
            peak_gpu_reserved = int(torch.cuda.max_memory_reserved())
    except ImportError:  # pragma: no cover - torch is required in live jobs
        pass
    try:
        import resource
        peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except ImportError:  # pragma: no cover - Windows contract tests
        peak_rss_kib = initial_peak_rss_kib
    runtime = with_content_hash({
        "contract": NODE_RUNTIME_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"],
        "node_id": node_id, "training_report_sha256": report["content_hash"],
        "pmard_engine_report_sha256": engine_report_hash,
        "wall_seconds": elapsed,
        "gpu_count": 1 if device.startswith("cuda") else 0,
        "measured_gpu_hours": elapsed / 3600 if device.startswith("cuda") else 0.0,
        "peak_rss_kib": peak_rss_kib,
        "peak_rss_growth_kib": max(0, peak_rss_kib - initial_peak_rss_kib),
        "peak_gpu_allocated_bytes": peak_gpu_allocated,
        "peak_gpu_reserved_bytes": peak_gpu_reserved,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "final_test_accessed": False,
    })
    write_immutable_json(output / "runtime.json", runtime)
    return report


__all__ = ["estimate_global_peak_bytes", "node_output_dir", "run_homotopy_node"]

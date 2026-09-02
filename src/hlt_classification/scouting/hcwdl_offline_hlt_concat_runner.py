"""Capacity, acceptance, and fit execution for tagged concatenation."""

from __future__ import annotations

import gc
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.models.hcwdl_tagged_concat_transformer import (
    TAGGED_CONCAT_MODEL_CONTRACT, build_tagged_concat_particle_transformer,
)

from .hcwdl_fullcard_bottleneck_foundation_campaign import validate_foundation
from .contracts import require_role_access
from .dataset import _concat_batches, _slice_batch
from .hcwdl_mhpe_tri60_runner import _configure_deterministic_backend
from .hcwdl_mhpe_tri60_training import (
    Tri60TrainingAuthority, Tri60TrainingRuntime, train_tri60_node,
)
from .hcwdl_offline_hlt_concat_campaign import (
    validate_campaign, validate_source_lock,
)
from .hcwdl_offline_hlt_concat_contracts import (
    CAPACITY_AUDIT_CONTRACT, EXECUTION_ACCEPTANCE_CONTRACT,
    FINAL_CHECKPOINT_CONTRACT, SELECTED_CHECKPOINT_CONTRACT,
    TRAINING_REPORT_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_offline_hlt_concat_data import (
    CONCAT_CAPACITY, build_tagged_concat_inputs, iterate_tagged_concat_batches,
    parallel_tagged_concat_source_streams, tagged_concat_required_branches,
)
from .hcwdl_offline_hlt_concat_graph import (
    GRAPH_SHA256, MODEL_INPUT_PROTOCOL, NODE_ID, node,
)
from .hcwdl_unified_balanced_runner import (
    _load_common, _memory_limit_bytes, _view_worker_plan,
)
from .labels import baseline_mask, multiclass_labels
from .schema import BASELINE_BRANCHES, LABEL_BRANCHES
from .splits import role_records
from .streaming import iterate_projected_chunks
from .view_cache import EphemeralPmardViewCache, expected_cache_source_rows


def training_authority() -> Tri60TrainingAuthority:
    authority = Tri60TrainingAuthority(
        node=node(), graph_sha256=GRAPH_SHA256,
        training_report_contract=TRAINING_REPORT_CONTRACT,
        selected_checkpoint_contract=SELECTED_CHECKPOINT_CONTRACT,
        final_checkpoint_contract=FINAL_CHECKPOINT_CONTRACT,
    )
    authority.validate()
    return authority


def _foundation(spec: Mapping[str, Any]) -> dict[str, Any]:
    value = load_json(spec["artifact_paths"]["foundation_spec"])
    validate_foundation(value)
    return value


def _selected_counts(
    foundation: Mapping[str, Any], split: Mapping[str, Any], selection, role: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    records = role_records(split, role)
    branches = set(BASELINE_BRANCHES) | set(LABEL_BRANCHES) | {
        "n_scoutpfcands", "n_cpfcands", "n_lts", "n_npfcands",
    }
    offline_parts = []
    hlt_parts = []
    maximum_identities = {"offline": "", "hlt": "", "combined": ""}
    maxima = {"offline": -1, "hlt": -1, "combined": -1}
    observed = 0
    for record in records:
        for chunk in iterate_projected_chunks(
            (Path(foundation["data_root"]) / record.path,), branches,
            data_root=foundation["data_root"], role=role, step_size=4096,
        ):
            labels = multiclass_labels(chunk.arrays)
            indexes = np.flatnonzero(baseline_mask(chunk.arrays) & (labels >= 0))
            absolute = chunk.entry_start + indexes
            indexes = indexes[selection.mask(chunk.source_path, absolute)]
            if not len(indexes):
                continue
            entries = chunk.entry_start + indexes
            hlt = np.asarray(chunk.arrays["n_scoutpfcands"])[indexes].astype(np.int64)
            offline = (
                np.asarray(chunk.arrays["n_cpfcands"])[indexes]
                + np.asarray(chunk.arrays["n_lts"])[indexes]
                + np.asarray(chunk.arrays["n_npfcands"])[indexes]
            ).astype(np.int64)
            if np.any(hlt < 0) or np.any(offline < 0):
                raise ValueError("concatenation audit observed a negative cardinality")
            total = offline + hlt
            for name, values in (
                ("offline", offline), ("hlt", hlt), ("combined", total),
            ):
                local = int(np.argmax(values))
                if int(values[local]) > maxima[name]:
                    maxima[name] = int(values[local])
                    maximum_identities[name] = (
                        f"{chunk.source_path}::tree::{int(entries[local])}"
                    )
            offline_parts.append(offline.astype(np.int16))
            hlt_parts.append(hlt.astype(np.int16))
            observed += len(indexes)
    if observed != selection.rows:
        raise ValueError("concatenation capacity-audit row coverage differs")
    offline = np.concatenate(offline_parts)
    hlt = np.concatenate(hlt_parts)
    return offline, hlt, offline + hlt, maximum_identities


def _distribution(values: np.ndarray) -> dict[str, Any]:
    return {
        "rows": len(values), "minimum": int(values.min()),
        "maximum": int(values.max()), "mean": float(values.mean()),
        "quantiles": {
            str(q): float(np.quantile(values, q, method="higher"))
            for q in (.5, .9, .95, .99, .999)
        },
    }


def build_capacity_audit(spec: Mapping[str, Any]) -> dict[str, Any]:
    validate_campaign(spec)
    foundation = _foundation(spec)
    split, split_hash, selection_hash, selections, _, _ = _load_common(foundation)
    roles = {}
    for role in ("train", "validation"):
        offline, hlt, total, maximum_identities = _selected_counts(
            foundation, split, selections[role], role,
        )
        roles[role] = {
            "offline": _distribution(offline), "hlt": _distribution(hlt),
            "combined": _distribution(total),
            "rows_over_capacity": int(np.count_nonzero(total > CONCAT_CAPACITY)),
            "maximum_identity": maximum_identities["combined"],
            "exact_smaller_capacity_candidates": [
                200, 300, 350, 400, 416, 432, 448, 464, 480, 496,
            ],
            "rows_over_candidate_capacity": {
                str(cap): int(np.count_nonzero(total > cap))
                for cap in (200, 300, 350, 400, 416, 432, 448, 464, 480, 496)
            },
        }
    if any(row["rows_over_capacity"] for row in roles.values()):
        details = ", ".join(
            f"{role}: maximum={row['combined']['maximum']}, "
            f"rows_over_{CONCAT_CAPACITY}={row['rows_over_capacity']}"
            for role, row in roles.items()
        )
        raise ValueError(
            f"tagged concatenation capacity would truncate tokens ({details})"
        )
    view_row_bytes = CONCAT_CAPACITY * (21 * 4 + 4 * 4 + 1 + 8 + 1 + 1 + 1)
    # The cache also materializes one int64 label and one canonical SHA-256
    # identity digest per row. Python identity strings are tracked separately
    # by the runtime's measured RSS/acceptance rather than mislabeled as arrays.
    row_bytes = view_row_bytes + np.dtype(np.int64).itemsize + 32
    projected = sum(int(spec["role_counts"][role]) * row_bytes for role in roles)
    return artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "source_lock": spec["parents"]["source_lock"],
            "foundation": spec["parents"]["foundation"],
            "split_manifest": split_hash, "selection_manifest": selection_hash,
        },
        "roles": roles, "selected_capacity": CONCAT_CAPACITY,
        "sequence_order": "offline_then_hlt_v1",
        "every_offline_and_hlt_token_retained": True,
        "matching_indices_read": False,
        "projected_view_row_bytes": view_row_bytes,
        "projected_cache_array_row_bytes": row_bytes,
        "projected_ram_cache_array_bytes": projected,
        "durable_particle_view_bytes": 0,
        "ordinary_access_roles": ["train", "validation"],
        "final_test_accessed": False,
    }, contract=CAPACITY_AUDIT_CONTRACT)


def _first_batch(spec: Mapping[str, Any], *, role: str, batch_size: int):
    foundation = _foundation(spec)
    split, _, _, selections, _, _ = _load_common(foundation)
    stream = iterate_tagged_concat_batches(
        split, data_root=foundation["data_root"], role=role,
        row_selection=selections[role], batch_size=batch_size,
    )
    try:
        return next(iter(stream))
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()


def _identity_batch(
    spec: Mapping[str, Any], *, role: str, identity: str,
) -> dict[str, Any]:
    foundation = _foundation(spec)
    split, _, _, selections, _, _ = _load_common(foundation)
    try:
        source_path, raw_entry = identity.rsplit("::tree::", 1)
        entry = int(raw_entry)
    except (ValueError, TypeError) as error:
        raise ValueError("concatenation acceptance identity differs") from error
    records = {record.path: record for record in role_records(split, role)}
    if source_path not in records:
        raise ValueError("concatenation acceptance identity is outside its role")
    selected = selections[role].mask(source_path, np.asarray([entry], np.int64))
    if selected.shape != (1,) or not bool(selected[0]):
        raise ValueError("concatenation acceptance identity is outside its role")
    data_root = Path(foundation["data_root"]).resolve()
    source = (data_root / source_path).resolve()
    try:
        source.relative_to(data_root)
    except ValueError as error:
        raise ValueError("concatenation acceptance source escapes data root") from error
    require_role_access(role, branch_read=True)
    import uproot
    with uproot.open(source) as handle:
        tree = handle["tree"]
        branches = tagged_concat_required_branches()
        missing = sorted(branches - set(tree.keys()))
        if missing:
            raise KeyError(f"concatenation acceptance lacks branches: {missing}")
        arrays = tree.arrays(
            sorted(branches), entry_start=entry, entry_stop=entry + 1,
            library="ak", how=dict,
        )
    labels = multiclass_labels(arrays)
    if len(labels) != 1 or not bool(baseline_mask(arrays)[0]) or labels[0] < 0:
        raise ValueError("concatenation acceptance selected row differs")
    return {
        "labels": labels,
        "identity_keys": np.asarray([identity]),
        "privileged": build_tagged_concat_inputs(arrays),
    }


def _acceptance_batch(
    spec: Mapping[str, Any], audit: Mapping[str, Any], *, batch_size: int,
) -> tuple[dict[str, Any], str, int]:
    role = max(
        ("train", "validation"),
        key=lambda name: int(audit["roles"][name]["combined"]["maximum"]),
    )
    maximum = int(audit["roles"][role]["combined"]["maximum"])
    identity = str(audit["roles"][role]["maximum_identity"])
    first = _first_batch(spec, role=role, batch_size=batch_size)
    if identity in set(map(str, first["identity_keys"])):
        batch = first
    else:
        batch = _concat_batches((
            _slice_batch(first, 0, batch_size - 1),
            _identity_batch(spec, role=role, identity=identity),
        ))
    if (
        len(batch["labels"]) != batch_size
        or int(np.max(batch["privileged"].raw_lengths)) != maximum
    ):
        raise RuntimeError("concatenation acceptance maximum-length batch differs")
    return batch, identity, maximum


def run_execution_acceptance(
    spec: Mapping[str, Any], *, device: str = "cuda",
) -> dict[str, Any]:
    import torch

    validate_campaign(spec)
    audit = load_json(spec["artifact_paths"]["capacity_audit"])
    audit_hash = validate_artifact(audit, contract=CAPACITY_AUDIT_CONTRACT)
    if (
        audit.get("parents", {}).get("campaign_spec") != spec["content_hash"]
        or audit.get("parents", {}).get("source_lock")
        != spec["parents"]["source_lock"]
        or audit.get("selected_capacity") != CONCAT_CAPACITY
        or audit.get("every_offline_and_hlt_token_retained") is not True
        or audit.get("matching_indices_read") is not False
        or audit.get("durable_particle_view_bytes") != 0
        or audit.get("final_test_accessed") is not False
    ):
        raise ValueError("tagged concatenation capacity audit differs")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("tagged concatenation preflight requires CUDA")
    # Exercise the exact production global/physical batch. A smaller smoke
    # batch would not establish that the 496-token attention path fits GH200.
    batch, maximum_identity, maximum_length = _acceptance_batch(
        spec, audit, batch_size=int(spec["batch_size"]),
    )
    view = batch["privileged"]
    model = build_tagged_concat_particle_transformer().to(target).train()
    features = torch.as_tensor(view.features, device=target)
    vectors = torch.as_tensor(view.vectors, device=target)
    mask = torch.as_tensor(view.mask, device=target)
    sources = torch.as_tensor(view.content_source_codes, device=target)
    labels = torch.as_tensor(batch["labels"], dtype=torch.long, device=target)
    if features.shape != (len(labels), 21, CONCAT_CAPACITY):
        raise RuntimeError("tagged concatenation preflight capacity shape differs")
    started = time.monotonic()
    with torch.autocast(
        device_type=target.type, dtype=torch.bfloat16,
        enabled=target.type == "cuda",
    ):
        logits = model(features, vectors, mask, sources)
        loss = torch.nn.functional.cross_entropy(logits.float(), labels)
    loss.backward()
    gradient = model.content_source_embedding.weight.grad
    if (
        logits.shape != (len(labels), 15) or not torch.isfinite(logits).all()
        or gradient is None or not torch.isfinite(gradient).all()
        or not bool((gradient.abs().sum() > 0).item())
    ):
        raise RuntimeError("tagged concatenation production backward differs")
    return artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "capacity_audit": audit_hash,
            "graph": GRAPH_SHA256,
        },
        "device_type": target.type,
        "device_name": (
            torch.cuda.get_device_name(target) if target.type == "cuda" else "cpu"
        ),
        "installed_weaver_forward_backward": True,
        "real_source_rows": len(labels),
        "production_batch_size_exercised": len(labels),
        "forward_backward_tensor_capacity": int(features.shape[2]),
        "maximum_realized_combined_length": max(
            audit["roles"][role]["combined"]["maximum"]
            for role in ("train", "validation")
        ),
        "maximum_length_exercised": maximum_length,
        "maximum_identity_exercised": maximum_identity,
        "source_embedding_gradient_nonzero": True,
        "physics_feature_channels": 21,
        "content_source_code_is_separate_metadata": True,
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated()) if target.type == "cuda" else 0
        ),
        "model_contract": TAGGED_CONCAT_MODEL_CONTRACT,
        "final_test_accessed": False,
    }, contract=EXECUTION_ACCEPTANCE_CONTRACT)


def _caches(spec: Mapping[str, Any]):
    foundation = _foundation(spec)
    split, split_hash, selection_hash, selections, _, _ = _load_common(foundation)
    caches = {}
    remaining = _memory_limit_bytes(350.0)
    for role in ("train", "validation"):
        started = time.monotonic()
        records = role_records(split, role)
        source_rows = expected_cache_source_rows(
            records, row_selection=selections[role],
        )
        total_workers, source_workers, _ = _view_worker_plan(
            sum(count > 0 for count in source_rows.values())
        )
        stream = parallel_tagged_concat_source_streams(
            split, data_root=foundation["data_root"], role=role,
            row_selection=selections[role], batch_size=256,
            source_workers=source_workers,
        )
        cache = EphemeralPmardViewCache.build(
            stream, expected_rows=selections[role].rows, records=records,
            role=role, expected_source_rows=source_rows,
            view_keys=("privileged",), max_gib=remaining / 1024**3,
            partitioned_sources=True,
            lineage={
                "campaign_spec": spec["content_hash"],
                "source_lock": spec["parents"]["source_lock"],
                "view_contract": "HCWDL_OFFLINE_HLT_TAGGED_CONCAT_VIEW/v2",
                "sequence_order": "offline_then_hlt_v1",
                "capacity": CONCAT_CAPACITY,
                "source_workers": source_workers,
                "view_worker_budget": total_workers,
                "source_parallel_backend": "process",
                "durable_particle_views": False,
            },
        )
        caches[role] = cache
        remaining -= int(cache.header["array_bytes"])
        if remaining <= 0:
            raise MemoryError("tagged concatenation caches exceed RAM budget")
        print(
            f"HCWDL-CONCAT phase=view_cache role={role} rows={selections[role].rows} "
            f"workers={total_workers} source_workers={source_workers} "
            f"seconds={time.monotonic()-started:.3f}", flush=True,
        )
    return caches, split_hash, selection_hash


def run_fit(
    spec: Mapping[str, Any], *, device: str = "cuda",
    recovery_spec_sha256: str | None = None,
    execution_source_commit: str | None = None,
) -> dict[str, Any]:
    validate_campaign(spec, executable=True)
    _configure_deterministic_backend()
    source = load_json(spec["artifact_paths"]["source_lock"])
    validate_source_lock(source)
    audit = load_json(spec["artifact_paths"]["capacity_audit"])
    acceptance = load_json(spec["artifact_paths"]["execution_acceptance"])
    audit_hash = validate_artifact(audit, contract=CAPACITY_AUDIT_CONTRACT)
    acceptance_hash = validate_artifact(
        acceptance, contract=EXECUTION_ACCEPTANCE_CONTRACT,
    )
    if (
        audit.get("parents", {}).get("campaign_spec") != spec["content_hash"]
        or acceptance.get("parents", {}).get("campaign_spec")
        != spec["content_hash"]
        or acceptance.get("parents", {}).get("capacity_audit") != audit_hash
        or acceptance.get("installed_weaver_forward_backward") is not True
        or acceptance.get("production_batch_size_exercised") != spec["batch_size"]
        or acceptance.get("forward_backward_tensor_capacity") != CONCAT_CAPACITY
        or acceptance.get("maximum_length_exercised")
        != acceptance.get("maximum_realized_combined_length")
        or acceptance.get("source_embedding_gradient_nonzero") is not True
        or acceptance.get("final_test_accessed") is not False
    ):
        raise ValueError("tagged concatenation execution gates differ")
    started = time.monotonic()
    caches, split_hash, selection_hash = _caches(spec)
    preparation = time.monotonic() - started
    try:
        recipe = load_json(spec["artifact_paths"]["recipe"])["training"]
        runtime = Tri60TrainingRuntime(
            passes=int(recipe["passes"]),
            batch_size=int(recipe["effective_batch_size"]),
            peak_learning_rate=float(recipe["peak_learning_rate"]),
            weight_decay=float(recipe["weight_decay"]),
            warmup_fraction=float(recipe["warmup_fraction"]),
            minimum_lr_fraction=float(recipe["learning_rate_floor_fraction"]),
            amp_dtype=str(recipe["forward_precision"]),
        )
        parents = {
            "campaign_spec": spec["content_hash"],
            "source_lock": spec["parents"]["source_lock"],
            "foundation": spec["parents"]["foundation"],
            "graph": GRAPH_SHA256, "recipe": spec["parents"]["recipe"],
            "capacity_audit": audit_hash,
            "execution_acceptance": acceptance_hash,
            "split_manifest": split_hash,
            "selection_manifest": selection_hash,
        }
        if recovery_spec_sha256 is not None:
            parents["recovery_spec"] = recovery_spec_sha256
        return train_tri60_node(
            node_id=NODE_ID, train_cache=caches["train"],
            validation_cache=caches["validation"], input_key="privileged",
            output_dir=Path(spec["campaign_root"]) / "training" / NODE_ID,
            parents=parents,
            campaign_spec_sha256=spec["content_hash"],
            recipe_sha256=spec["parents"]["recipe"],
            execution_source_commit=execution_source_commit or spec["source_commit"],
            replicate_seed=int(spec["replicate_seed"]), device=device,
            runtime=runtime, authority=training_authority(),
            model_factory=build_tagged_concat_particle_transformer,
            model_input_protocol=MODEL_INPUT_PROTOCOL,
            preparation_metrics={
                "student_view_cache_seconds": preparation,
                "pre_training_total_seconds": time.monotonic() - started,
            },
        )
    finally:
        caches.clear()
        gc.collect()


__all__ = [
    "build_capacity_audit", "run_execution_acceptance", "run_fit",
    "training_authority",
]

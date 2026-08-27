"""Five-shard inference and greedy validation ensembles for TRI60 M1 fits."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import gc
import math
import multiprocessing
import os
from pathlib import Path
import re
import time
from typing import Any, Final, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, atomic_publish_bytes, deterministic_npz_bytes,
    load_json, load_npz_arrays, require_sha256, sha256_file,
    write_immutable_json,
)

from .engine import precompute_teacher_targets
from .evaluation import softmax
from .hcwdl_mhpe_roc import _probability_metrics
from .hcwdl_mhpe_tri60_contracts import (
    TRAINING_REPORT_CONTRACT as SOURCE_TRAINING_REPORT_CONTRACT,
    validate_artifact as validate_source_artifact,
)
from .hcwdl_mhpe_tri60_graph import COORDINATES
from .hcwdl_mhpe_tri60_integration import validate_tri60_foundation_lock
from .hcwdl_mhpe_tri60_training import load_tri60_model
from .hcwdl_representation_data import canonical_identity_digests
from .hcwdl_tri60_m1_screen_campaign import (
    validate_campaign as validate_screen_campaign,
)
from .hcwdl_tri60_m1_screen_contracts import (
    AGGREGATE_CONTRACT as SCREEN_AGGREGATE_CONTRACT,
    CAMPAIGN_COMPLETE_CONTRACT as SCREEN_COMPLETE_CONTRACT,
    TRAINING_REPORT_CONTRACT as SCREEN_TRAINING_REPORT_CONTRACT,
    validate_artifact as validate_screen_artifact,
)
from .hcwdl_tri60_m1_screen_graph import (
    FIT_ORDER, IMPORTED_CONTROL_ID, SEED_ALIAS,
)
from .hcwdl_tri60_m1_screen_reporting import training_report
from .hcwdl_tri60_m1_screen_runner import training_authority
from .hcwdl_tri60_m1_greedy_ensemble_contracts import (
    CAMPAIGN_COMPLETE_CONTRACT, RESULT_REPORT_CONTRACT,
    SHARD_REPORT_CONTRACT, SOURCE_LOCK_CONTRACT, artifact,
    validate_artifact,
)
from .hcwdl_unified_balanced_full_campaign import validate_foundation_campaign
from .hcwdl_unified_balanced_runner import (
    _load_common, _parallel_source_streams, _stream, _view_worker_plan,
)
from .splits import role_records
from .training import derive_seed
from .view_cache import EphemeralPmardViewCache, expected_cache_source_rows


CANDIDATE_ORDER: Final = (IMPORTED_CONTROL_ID, *FIT_ORDER)
SHARD_COUNT: Final = 5
SHARD_SIZE: Final = 4
SHARDS: Final = tuple(
    tuple(CANDIDATE_ORDER[start:start + SHARD_SIZE])
    for start in range(0, len(CANDIDATE_ORDER), SHARD_SIZE)
)
OBJECTIVE_ORDER: Final = (
    "macro_auc", "macro_r50_linear", "balanced_auc_r50_recovery",
)
MAX_ENSEMBLE_SIZE: Final = 5
ENSEMBLE_POLICY: Final = "uniform_probability_mean_fp64_accumulation_to_fp32_v1"
BALANCED_POLICY: Final = (
    "mean_linear_recovery_from_SOURCE_M1_LOGIT_to_LOGIT_D000E_"
    "for_macro_auc_and_linear_macro_r50_v1"
)
MAX_DURABLE_PREDICTION_BYTES: Final = 2 * 1024**3

if len(CANDIDATE_ORDER) != 20 or len(SHARDS) != SHARD_COUNT or any(
    len(shard) != SHARD_SIZE for shard in SHARDS
):
    raise RuntimeError("TRI60 M1 greedy candidate sharding differs")


def _source_training_report(
    screen: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    path = Path(screen["artifact_paths"]["source_m1_report"])
    report = load_json(path)
    if (
        report.get("node_id") != "M1_LOGIT"
        or report.get("complete") is not True
        or report.get("passes") != 60
        or report.get("validations") != 60
        or report.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 M1 greedy imported control differs")
    checkpoint = path.parent / str(report.get("selected_checkpoint", ""))
    if (
        not checkpoint.is_file()
        or sha256_file(checkpoint) != report.get("selected_checkpoint_sha256")
    ):
        raise ValueError("TRI60 M1 greedy imported checkpoint differs")
    return path.resolve(), report


def _validate_foundation_boundary(
    *, screen: Mapping[str, Any], source_campaign: Mapping[str, Any],
    foundation_path: Path, foundation_hash: str,
    tri60_foundation: Mapping[str, Any], tri60_foundation_hash: str,
) -> None:
    """Bind the TRI60 foundation lock to its underlying foundation spec."""

    if (
        tri60_foundation_hash != screen["parents"]["foundation"]
        or tri60_foundation.get("parents", {}).get("foundation_spec")
        != foundation_hash
        or Path(tri60_foundation.get("foundation_spec_path", "")).resolve()
        != foundation_path.resolve()
        or source_campaign.get("content_hash")
        != screen["parents"]["source_campaign"]
        or screen.get("role_counts", {}).get("validation") != 957541
    ):
        raise ValueError("TRI60 M1 greedy foundation boundary differs")


def build_source_lock(screen_campaign_spec: str | Path) -> dict[str, Any]:
    """Authenticate the complete 20-condition screen without copying it."""

    screen_path = Path(screen_campaign_spec).resolve()
    screen = load_json(screen_path)
    screen_hash = validate_screen_campaign(screen, executable=False)
    root = Path(screen["campaign_root"])
    aggregate_path = root / "reports/validation_aggregate.json"
    complete_path = root / "reports/campaign_complete.json"
    aggregate = load_json(aggregate_path)
    complete = load_json(complete_path)
    aggregate_hash = validate_screen_artifact(
        aggregate, contract=SCREEN_AGGREGATE_CONTRACT,
    )
    complete_hash = validate_screen_artifact(
        complete, contract=SCREEN_COMPLETE_CONTRACT,
    )
    if (
        aggregate.get("parents", {}).get("campaign_spec") != screen_hash
        or complete.get("parents", {}).get("campaign_spec") != screen_hash
        or complete.get("parents", {}).get("aggregate") != aggregate_hash
        or aggregate.get("condition_count") != 20
        or aggregate.get("fresh_fit_count") != 19
        or complete.get("condition_count") != 20
        or aggregate.get("final_test_accessed") is not False
        or complete.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 M1 greedy completed screen differs")

    source_path, source_report = _source_training_report(screen)
    rows = {row["row_id"]: row for row in aggregate["rows"]}
    if tuple(rows) != CANDIDATE_ORDER:
        raise ValueError("TRI60 M1 greedy aggregate candidate order differs")
    reports: dict[str, dict[str, Any]] = {
        IMPORTED_CONTROL_ID: {
            "path": str(source_path),
            "report_sha256": require_sha256(
                source_report["content_hash"], name="source M1 report",
            ),
            "checkpoint_sha256": require_sha256(
                source_report["selected_checkpoint_sha256"],
                name="source M1 checkpoint",
            ),
            "validation": dict(source_report["validation"]),
            "authority": "source_tri60",
        },
    }
    for node_id in FIT_ORDER:
        report = training_report(screen, node_id)
        path = root / "training" / node_id / "training_report.json"
        reports[node_id] = {
            "path": str(path.resolve()),
            "report_sha256": report["content_hash"],
            "checkpoint_sha256": report["selected_checkpoint_sha256"],
            "validation": dict(report["validation"]),
            "authority": "m1_screen",
        }
    if any(rows[node_id]["metrics"] != reports[node_id]["validation"] for node_id in CANDIDATE_ORDER):
        raise ValueError("TRI60 M1 greedy aggregate/report metrics differ")

    source_campaign = load_json(screen["artifact_paths"]["source_campaign_spec"])
    foundation_path = Path(screen["artifact_paths"]["foundation_spec"]).resolve()
    foundation = load_json(foundation_path)
    foundation_hash = validate_foundation_campaign(
        foundation, executable=False, verify_source_tree=False,
    )
    tri60_foundation_path = Path(
        source_campaign["artifact_paths"]["tri60_foundation_lock"]
    ).resolve()
    tri60_foundation = load_json(tri60_foundation_path)
    tri60_foundation_hash = validate_tri60_foundation_lock(tri60_foundation)
    _validate_foundation_boundary(
        screen=screen, source_campaign=source_campaign,
        foundation_path=foundation_path, foundation_hash=foundation_hash,
        tri60_foundation=tri60_foundation,
        tri60_foundation_hash=tri60_foundation_hash,
    )

    return artifact({
        "parents": {
            "screen_campaign": screen_hash,
            "screen_aggregate": aggregate_hash,
            "screen_complete": complete_hash,
            "screen_source_lock": screen["parents"]["source_lock"],
            "source_campaign": screen["parents"]["source_campaign"],
            "foundation": tri60_foundation_hash,
            "foundation_spec": foundation_hash,
            "recipe": screen["parents"]["recipe"],
            "screen_graph": screen["parents"]["graph"],
            "teacher_stage": aggregate["parents"]["teacher_stage"],
        },
        "artifact_paths": {
            "screen_campaign_spec": str(screen_path),
            "screen_aggregate": str(aggregate_path.resolve()),
            "screen_complete": str(complete_path.resolve()),
            "source_campaign_spec": screen["artifact_paths"]["source_campaign_spec"],
            "foundation_spec": str(foundation_path),
            "tri60_foundation_lock": str(tri60_foundation_path),
        },
        "candidate_order": list(CANDIDATE_ORDER),
        "candidate_reports": reports,
        "teacher": dict(aggregate["teacher"]),
        "replicate_seed": int(screen["replicate_seed"]),
        "role_counts": dict(screen["role_counts"]),
        "screen_source_outputs_read_only": True,
        "source_scheduler_dependency": False,
        "ordinary_access_roles": ["validation"],
        "final_test_accessed": False,
    }, contract=SOURCE_LOCK_CONTRACT)


def validate_source_lock(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=SOURCE_LOCK_CONTRACT)
    rebuilt = build_source_lock(value["artifact_paths"]["screen_campaign_spec"])
    if value != rebuilt:
        raise ValueError("TRI60 M1 greedy source lock changed")
    return digest


def shard_paths(root: str | Path, shard_index: int) -> tuple[Path, Path]:
    if shard_index < 0 or shard_index >= SHARD_COUNT:
        raise IndexError("TRI60 M1 greedy shard index differs")
    directory = Path(root) / "predictions"
    stem = f"shard_{shard_index:02d}"
    return directory / f"{stem}.npz", directory / f"{stem}.json"


def _validation_cache(
    *, screen: Mapping[str, Any], memory_gib: float,
) -> tuple[EphemeralPmardViewCache, str]:
    foundation = load_json(screen["artifact_paths"]["foundation_spec"])
    validate_foundation_campaign(
        foundation, executable=False, verify_source_tree=False,
    )
    split, split_hash, _, selections, assignments, balanced = _load_common(foundation)
    role = "validation"
    records = role_records(split, role)
    source_rows = expected_cache_source_rows(
        records, row_selection=selections[role],
    )
    nonempty = sum(int(count) > 0 for count in source_rows.values())
    total_workers, source_workers, transform_workers = _view_worker_plan(nonempty)
    backend = os.environ.get("HCWDL_UB_VIEW_SOURCE_BACKEND", "process").strip().lower()
    if backend not in {"process", "thread"}:
        raise ValueError("TRI60 M1 greedy source backend differs")
    if source_workers > 1 and backend == "process":
        transform_workers = 1
    partitioned = source_workers > 1
    sampler_seed = derive_seed(
        int(screen["replicate_seed"]), SEED_ALIAS + "/sampler",
    )
    repair_seed = derive_seed(
        int(screen["replicate_seed"]), "tri60/repair/shared_v1",
    )
    coordinate = COORDINATES["D000"]
    stream = (
        _parallel_source_streams(
            foundation_spec=foundation, split=split, selections=selections,
            assignments=assignments, balanced=balanced, role=role,
            behavior="hlt", coordinate=coordinate, batch_size=256,
            sampler_seed=sampler_seed, repair_seed=repair_seed,
            include_hcwdl_metadata=False, records=records,
            expected_source_rows=source_rows, source_workers=source_workers,
            transform_workers=transform_workers, source_backend=backend,
        )
        if partitioned else _stream(
            foundation_spec=foundation, split=split, selections=selections,
            assignments=assignments, balanced=balanced, role=role,
            behavior="hlt", coordinate=coordinate, batch_size=256,
            sampler_seed=sampler_seed, repair_seed=repair_seed,
            include_hcwdl_metadata=False, view_workers=transform_workers,
        )
    )
    started = time.monotonic()
    cache = EphemeralPmardViewCache.build(
        stream, expected_rows=selections[role].rows, records=records,
        role=role, expected_source_rows=source_rows, view_keys=("hlt",),
        max_gib=memory_gib, partitioned_sources=partitioned,
        lineage={
            "foundation_spec_sha256": foundation["content_hash"],
            "split_manifest_sha256": split_hash,
            "behavior": "hlt", "coordinate": coordinate.payload(),
            "validation_only": True, "student_view_built_once": True,
            "durable_particle_views": False,
            "source_partitioned_preprocessing": partitioned,
            "view_worker_budget": total_workers,
            "source_workers": source_workers,
            "transform_workers_per_source": transform_workers,
            "source_parallel_backend": backend if partitioned else "serial",
        },
    )
    print(
        "HCWDL-M1-GREEDY phase=validation_view_cache "
        f"rows={cache.header['rows']} workers={total_workers} "
        f"source_workers={source_workers} backend={backend if partitioned else 'serial'} "
        f"seconds={time.monotonic()-started:.3f}",
        flush=True,
    )
    return cache, split_hash


def _cache_labels(cache: EphemeralPmardViewCache) -> np.ndarray:
    parts = [
        np.ascontiguousarray(batch["labels"], dtype=np.int64)
        for batch in cache.iterate_canonical_batches(batch_size=4096)
    ]
    result = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    if result.shape != (int(cache.header["rows"]),):
        raise ValueError("TRI60 M1 greedy validation labels differ")
    return result


def _candidate_model(
    source: Mapping[str, Any], candidate_id: str, *, device: str,
):
    row = source["candidate_reports"][candidate_id]
    report_path = Path(row["path"])
    report = load_json(report_path)
    if candidate_id == IMPORTED_CONTROL_ID:
        report_hash = validate_source_artifact(
            report, contract=SOURCE_TRAINING_REPORT_CONTRACT,
        )
    else:
        report_hash = validate_screen_artifact(
            report, contract=SCREEN_TRAINING_REPORT_CONTRACT,
        )
    if report_hash != row["report_sha256"]:
        raise ValueError("TRI60 M1 greedy candidate report changed")
    authority = None if candidate_id == IMPORTED_CONTROL_ID else training_authority(candidate_id)
    model, loaded = load_tri60_model(
        report_path, device=device, **({} if authority is None else {"authority": authority}),
    )
    if (
        loaded.get("content_hash") != row["report_sha256"]
        or loaded.get("selected_checkpoint_sha256") != row["checkpoint_sha256"]
    ):
        raise ValueError("TRI60 M1 greedy loaded candidate differs")
    model.eval()
    return model, report


def run_prediction_shard(
    *, spec: Mapping[str, Any], shard_index: int, device: str = "cuda",
) -> dict[str, Any]:
    source = load_json(spec["artifact_paths"]["source_lock"])
    if (
        validate_artifact(source, contract=SOURCE_LOCK_CONTRACT)
        != spec["parents"]["source_lock"]
    ):
        raise ValueError("TRI60 M1 greedy source changed")
    candidates = SHARDS[shard_index]
    data_path, report_path = shard_paths(spec["campaign_root"], shard_index)
    # A killed worker may leave the fully published NPZ before its small JSON
    # attestation.  Recomputing is safe: atomic_publish_bytes accepts only the
    # exact same bytes and rejects any divergent partial/reused artifact.
    if report_path.exists():
        raise FileExistsError("TRI60 M1 greedy shard report already exists")
    screen = load_json(source["artifact_paths"]["screen_campaign_spec"])
    started = time.monotonic()
    cache, split_hash = _validation_cache(
        screen=screen, memory_gib=float(spec["validation_cache_memory_gib"]),
    )
    identities = canonical_identity_digests(cache.identities)
    labels = _cache_labels(cache)
    arrays: dict[str, np.ndarray] = {
        "identity_digests": identities, "labels": labels,
    }
    lineage: dict[str, dict[str, str]] = {}
    try:
        for candidate_id in candidates:
            model, report = _candidate_model(source, candidate_id, device=device)
            targets = precompute_teacher_targets(
                model, cache.iterate_canonical_batches(batch_size=256),
                input_key="hlt", device=device,
                teacher_report_sha256=report["content_hash"],
                split_manifest_sha256=split_hash,
            )
            try:
                observed = canonical_identity_digests(targets.identities)
                if not np.array_equal(observed, identities):
                    raise ValueError("TRI60 M1 greedy inference identity order differs")
                probability = np.ascontiguousarray(
                    softmax(targets.logits), dtype=np.float32,
                )
                if (
                    probability.shape != (len(labels), 15)
                    or not np.isfinite(probability).all()
                    or np.any(probability < 0)
                    or not np.allclose(
                        probability.sum(axis=1, dtype=np.float64), 1.0,
                        rtol=0, atol=2e-6,
                    )
                ):
                    raise ValueError("TRI60 M1 greedy candidate probabilities differ")
                array_name = f"probabilities__{candidate_id}"
                arrays[array_name] = probability
                lineage[candidate_id] = {
                    "report_sha256": report["content_hash"],
                    "checkpoint_sha256": report["selected_checkpoint_sha256"],
                    "probabilities_sha256": array_sha256(array_name, probability),
                }
            finally:
                del targets
                del model
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
    finally:
        del cache
        gc.collect()

    payload = deterministic_npz_bytes(arrays)
    if len(payload) > int(spec["maximum_shard_bytes"]):
        raise OSError("TRI60 M1 greedy prediction shard exceeds its storage bound")
    atomic_publish_bytes(data_path, payload)
    report = artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "source_lock": source["content_hash"],
            "screen_campaign": source["parents"]["screen_campaign"],
            "foundation": source["parents"]["foundation"],
        },
        "shard_index": shard_index,
        "candidate_ids": list(candidates),
        "candidate_lineage": lineage,
        "prediction_path": str(data_path.resolve()),
        "prediction_file_sha256": sha256_file(data_path),
        "prediction_file_bytes": data_path.stat().st_size,
        "identity_order_sha256": array_sha256("identity_digests", identities),
        "labels_sha256": array_sha256("labels", labels),
        "validation_rows": len(labels),
        "probability_dtype": "float32",
        "persistent_particle_views": False,
        "persistent_logits": False,
        "ordinary_access_roles": ["validation"],
        "runtime_seconds": time.monotonic() - started,
        "final_test_accessed": False,
    }, contract=SHARD_REPORT_CONTRACT)
    write_immutable_json(report_path, report)
    return report


def validate_prediction_shard(
    report: Mapping[str, Any], *, spec: Mapping[str, Any], shard_index: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    validate_artifact(report, contract=SHARD_REPORT_CONTRACT)
    expected_candidates = SHARDS[shard_index]
    path = Path(report["prediction_path"])
    expected_path, _ = shard_paths(spec["campaign_root"], shard_index)
    source = load_json(spec["artifact_paths"]["source_lock"])
    if (
        report.get("parents", {}).get("campaign_spec") != spec["content_hash"]
        or report.get("parents", {}).get("source_lock") != spec["parents"]["source_lock"]
        or report.get("shard_index") != shard_index
        or report.get("candidate_ids") != list(expected_candidates)
        or report.get("validation_rows") != spec["role_counts"]["validation"]
        or report.get("persistent_particle_views") is not False
        or report.get("persistent_logits") is not False
        or report.get("ordinary_access_roles") != ["validation"]
        or report.get("final_test_accessed") is not False
        or not np.isfinite(report.get("runtime_seconds", np.nan))
        or float(report.get("runtime_seconds", -1)) < 0
        or path.resolve() != expected_path.resolve()
        or not path.is_file()
        or sha256_file(path) != report.get("prediction_file_sha256")
        or path.stat().st_size != report.get("prediction_file_bytes")
        or path.stat().st_size > int(spec["maximum_shard_bytes"])
    ):
        raise ValueError("TRI60 M1 greedy shard report differs")
    arrays = load_npz_arrays(path)
    expected_names = {"identity_digests", "labels"} | {
        f"probabilities__{candidate_id}" for candidate_id in expected_candidates
    }
    if set(arrays) != expected_names:
        raise ValueError("TRI60 M1 greedy shard arrays differ")
    identities = np.ascontiguousarray(arrays["identity_digests"], dtype=np.uint8)
    labels = np.ascontiguousarray(arrays["labels"], dtype=np.int64)
    if (
        identities.shape != (len(labels), 32)
        or len(labels) != spec["role_counts"]["validation"]
        or array_sha256("identity_digests", identities) != report["identity_order_sha256"]
        or array_sha256("labels", labels) != report["labels_sha256"]
    ):
        raise ValueError("TRI60 M1 greedy shard identities/labels differ")
    probabilities = {}
    for candidate_id in expected_candidates:
        name = f"probabilities__{candidate_id}"
        value = np.ascontiguousarray(arrays[name], dtype=np.float32)
        lineage = report["candidate_lineage"].get(candidate_id, {})
        expected_lineage = source["candidate_reports"][candidate_id]
        if (
            value.shape != (len(labels), 15)
            or not np.isfinite(value).all()
            or np.any(value < 0)
            or not np.allclose(value.sum(1, dtype=np.float64), 1, rtol=0, atol=2e-6)
            or array_sha256(name, value) != lineage.get("probabilities_sha256")
            or lineage.get("report_sha256") != expected_lineage["report_sha256"]
            or lineage.get("checkpoint_sha256")
            != expected_lineage["checkpoint_sha256"]
        ):
            raise ValueError("TRI60 M1 greedy shard probabilities differ")
        probabilities[candidate_id] = value
    return identities, labels, probabilities


_EVAL_PROBABILITIES: dict[str, np.ndarray] = {}
_EVAL_LABELS: np.ndarray | None = None


def _uniform_probability(members: Sequence[str]) -> np.ndarray:
    if not members or len(members) != len(set(members)):
        raise ValueError("TRI60 M1 greedy ensemble members differ")
    total = np.zeros_like(_EVAL_PROBABILITIES[members[0]], dtype=np.float64)
    for member in members:
        total += _EVAL_PROBABILITIES[member]
    result = np.ascontiguousarray(total / len(members), dtype=np.float32)
    if not np.allclose(result.sum(1, dtype=np.float64), 1, rtol=0, atol=2e-6):
        raise ValueError("TRI60 M1 greedy ensemble probabilities differ")
    return result


def _evaluate_members(members: tuple[str, ...]) -> tuple[tuple[str, ...], dict[str, Any]]:
    if _EVAL_LABELS is None:
        raise RuntimeError("TRI60 M1 greedy evaluation context is absent")
    return members, _probability_metrics(_uniform_probability(members), _EVAL_LABELS)


def _r50(metrics: Mapping[str, Any]) -> float:
    return math.exp(float(metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]))


def _score(
    objective: str, metrics: Mapping[str, Any], *, source: Mapping[str, Any],
    teacher: Mapping[str, Any],
) -> float:
    auc = float(metrics["macro_ovr_auc"])
    r50 = _r50(metrics)
    if objective == "macro_auc":
        return auc
    if objective == "macro_r50_linear":
        return r50
    if objective != "balanced_auc_r50_recovery":
        raise KeyError("TRI60 M1 greedy objective differs")
    auc_denominator = float(teacher["macro_ovr_auc"]) - float(source["macro_ovr_auc"])
    r50_denominator = _r50(teacher) - _r50(source)
    if auc_denominator <= 0 or r50_denominator <= 0:
        raise ValueError("TRI60 M1 greedy balanced reference ordering differs")
    return 0.5 * (
        (auc - float(source["macro_ovr_auc"])) / auc_denominator
        + (r50 - _r50(source)) / r50_denominator
    )


def _metric_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    return {
        "accuracy": float(left["accuracy"]) - float(right["accuracy"]),
        "macro_ovr_auc": float(left["macro_ovr_auc"]) - float(right["macro_ovr_auc"]),
        "macro_r50_linear": _r50(left) - _r50(right),
    }


def _metrics_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    names = (
        "accuracy", "cross_entropy", "macro_ovr_auc",
        "macro_mean_log_qcd_rejection_at_50pct_signal",
    )
    return all(
        np.isclose(float(left[name]), float(right[name]), rtol=0, atol=5e-6)
        for name in names
    )


def greedy_paths(
    *, probabilities: Mapping[str, np.ndarray], labels: np.ndarray,
    expected_metrics: Mapping[str, Mapping[str, Any]],
    teacher_metrics: Mapping[str, Any], workers: int = 1,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Return three deterministic forward-selection paths through size five."""

    global _EVAL_PROBABILITIES, _EVAL_LABELS
    if tuple(probabilities) != CANDIDATE_ORDER:
        raise ValueError("TRI60 M1 greedy probability candidate order differs")
    shapes = {np.asarray(value).shape for value in probabilities.values()}
    if shapes != {(len(labels), 15)} or len(labels) <= 0:
        raise ValueError("TRI60 M1 greedy probability shapes differ")
    _EVAL_PROBABILITIES = {
        name: np.ascontiguousarray(value, dtype=np.float32)
        for name, value in probabilities.items()
    }
    _EVAL_LABELS = np.ascontiguousarray(labels, dtype=np.int64)
    source_metrics = expected_metrics[IMPORTED_CONTROL_ID]
    cache: dict[tuple[str, ...], dict[str, Any]] = {}
    selected = {objective: tuple() for objective in OBJECTIVE_ORDER}
    paths = {objective: [] for objective in OBJECTIVE_ORDER}
    candidate_index = {name: index for index, name in enumerate(CANDIDATE_ORDER)}

    def canonical(members: Sequence[str]) -> tuple[str, ...]:
        return tuple(sorted(members, key=candidate_index.__getitem__))

    worker_count = max(1, min(int(workers), 32))
    executor = None
    if worker_count > 1 and "fork" in multiprocessing.get_all_start_methods():
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("fork"),
        )
    try:
        for size in range(1, MAX_ENSEMBLE_SIZE + 1):
            needed = set()
            for objective in OBJECTIVE_ORDER:
                used = set(selected[objective])
                for candidate in CANDIDATE_ORDER:
                    if candidate not in used:
                        needed.add(canonical((*selected[objective], candidate)))
            missing = sorted(
                (members for members in needed if members not in cache),
                key=lambda row: tuple(candidate_index[item] for item in row),
            )
            if executor is None:
                evaluated = map(_evaluate_members, missing)
            else:
                evaluated = executor.map(_evaluate_members, missing)
            for members, metrics in evaluated:
                cache[members] = metrics

            if size == 1:
                for candidate in CANDIDATE_ORDER:
                    observed = cache[(candidate,)]
                    if not _metrics_match(observed, expected_metrics[candidate]):
                        raise ValueError(
                            f"TRI60 M1 greedy singleton metrics differ: {candidate}"
                        )

            for objective in OBJECTIVE_ORDER:
                used = set(selected[objective])
                options = []
                for candidate in CANDIDATE_ORDER:
                    if candidate in used:
                        continue
                    members = canonical((*selected[objective], candidate))
                    metrics = cache[members]
                    score = _score(
                        objective, metrics, source=source_metrics,
                        teacher=teacher_metrics,
                    )
                    options.append((
                        -score,
                        -float(metrics["macro_ovr_auc"]),
                        -_r50(metrics),
                        float(metrics["cross_entropy"]),
                        candidate_index[candidate], candidate, members, metrics,
                    ))
                _, _, _, _, _, added, members, metrics = min(options)
                selected[objective] = members
                score = _score(
                    objective, metrics, source=source_metrics,
                    teacher=teacher_metrics,
                )
                paths[objective].append({
                    "ensemble_size": size,
                    "added_member": added,
                    "members": list(members),
                    "uniform_member_weight": 1.0 / size,
                    "objective_score": score,
                    "metrics": metrics,
                    "delta_from_source_m1": _metric_delta(metrics, source_metrics),
                    "delta_from_teacher": _metric_delta(metrics, teacher_metrics),
                })
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        _EVAL_PROBABILITIES = {}
        _EVAL_LABELS = None
    return paths, len(cache)


def build_result(spec: Mapping[str, Any]) -> dict[str, Any]:
    source = load_json(spec["artifact_paths"]["source_lock"])
    if (
        validate_artifact(source, contract=SOURCE_LOCK_CONTRACT)
        != spec["parents"]["source_lock"]
    ):
        raise ValueError("TRI60 M1 greedy reducer source differs")
    reference_identities = reference_labels = None
    probabilities: dict[str, np.ndarray] = {}
    shard_hashes = {}
    durable_bytes = 0
    for shard_index in range(SHARD_COUNT):
        _, report_path = shard_paths(spec["campaign_root"], shard_index)
        report = load_json(report_path)
        identities, labels, values = validate_prediction_shard(
            report, spec=spec, shard_index=shard_index,
        )
        if reference_identities is None:
            reference_identities, reference_labels = identities, labels
        elif not np.array_equal(reference_identities, identities) or not np.array_equal(reference_labels, labels):
            raise ValueError("TRI60 M1 greedy shards are not row aligned")
        probabilities.update(values)
        shard_hashes[f"shard_{shard_index:02d}"] = report["content_hash"]
        durable_bytes += int(report["prediction_file_bytes"])
    probabilities = {name: probabilities[name] for name in CANDIDATE_ORDER}
    if durable_bytes > MAX_DURABLE_PREDICTION_BYTES:
        raise OSError("TRI60 M1 greedy durable predictions exceed campaign bound")
    expected = {
        name: row["validation"]
        for name, row in source["candidate_reports"].items()
    }
    workers = int(os.environ.get(
        "HCWDL_M1_GREEDY_REDUCER_WORKERS",
        os.environ.get("SLURM_CPUS_PER_TASK", "1"),
    ))
    started = time.monotonic()
    paths, evaluated_count = greedy_paths(
        probabilities=probabilities, labels=reference_labels,
        expected_metrics=expected, teacher_metrics=source["teacher"]["metrics"],
        workers=workers,
    )
    best = {}
    for objective, rows in paths.items():
        winner = min(
            rows,
            key=lambda row: (-float(row["objective_score"]), row["ensemble_size"]),
        )
        best[objective] = {
            "ensemble_size": winner["ensemble_size"],
            "members": list(winner["members"]),
            "objective_score": winner["objective_score"],
            "metrics": winner["metrics"],
        }
    return artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "source_lock": source["content_hash"],
            **shard_hashes,
        },
        "question": "can_validation_greedy_ensembling_improve_exact_HLT_M1_compression",
        "candidate_order": list(CANDIDATE_ORDER),
        "candidate_count": len(CANDIDATE_ORDER),
        "objective_order": list(OBJECTIVE_ORDER),
        "maximum_ensemble_size": MAX_ENSEMBLE_SIZE,
        "ensemble_policy": ENSEMBLE_POLICY,
        "balanced_objective_policy": BALANCED_POLICY,
        "source_m1_metrics": expected[IMPORTED_CONTROL_ID],
        "teacher": dict(source["teacher"]),
        "paths": paths,
        "best_along_each_path": best,
        "evaluated_unique_ensemble_count": evaluated_count,
        "reducer_workers": max(1, min(workers, 32)),
        "validation_rows": len(reference_labels),
        "validation_identity_order_sha256": array_sha256(
            "identity_digests", reference_identities,
        ),
        "validation_labels_sha256": array_sha256("labels", reference_labels),
        "durable_candidate_prediction_bytes": durable_bytes,
        "maximum_durable_prediction_bytes": MAX_DURABLE_PREDICTION_BYTES,
        "validation_selected_exploratory_diagnostic": True,
        "automatic_finalist_selection": False,
        "fresh_fit_count": 0,
        "source_campaign_outputs_mutated": False,
        "screen_campaign_outputs_mutated": False,
        "ordinary_access_roles": ["validation"],
        "runtime_seconds": time.monotonic() - started,
        "final_test_accessed": False,
    }, contract=RESULT_REPORT_CONTRACT)


def validate_result(value: Mapping[str, Any], *, spec: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=RESULT_REPORT_CONTRACT)
    if (
        value.get("parents", {}).get("campaign_spec") != spec["content_hash"]
        or value.get("parents", {}).get("source_lock") != spec["parents"]["source_lock"]
        or value.get("candidate_order") != list(CANDIDATE_ORDER)
        or value.get("candidate_count") != 20
        or value.get("objective_order") != list(OBJECTIVE_ORDER)
        or value.get("maximum_ensemble_size") != MAX_ENSEMBLE_SIZE
        or value.get("ensemble_policy") != ENSEMBLE_POLICY
        or value.get("balanced_objective_policy") != BALANCED_POLICY
        or set(value.get("paths", {})) != set(OBJECTIVE_ORDER)
        or any(len(rows) != MAX_ENSEMBLE_SIZE for rows in value["paths"].values())
        or value.get("durable_candidate_prediction_bytes", MAX_DURABLE_PREDICTION_BYTES + 1)
        > MAX_DURABLE_PREDICTION_BYTES
        or value.get("validation_selected_exploratory_diagnostic") is not True
        or value.get("automatic_finalist_selection") is not False
        or value.get("fresh_fit_count") != 0
        or value.get("source_campaign_outputs_mutated") is not False
        or value.get("screen_campaign_outputs_mutated") is not False
        or value.get("ordinary_access_roles") != ["validation"]
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI60 M1 greedy result semantics differ")
    for objective, rows in value["paths"].items():
        for size, row in enumerate(rows, 1):
            if (
                row.get("ensemble_size") != size
                or len(row.get("members", ())) != size
                or len(set(row["members"])) != size
                or any(member not in CANDIDATE_ORDER for member in row["members"])
                or not np.isclose(row.get("uniform_member_weight", -1), 1 / size)
                or row.get("added_member") not in row["members"]
                or not np.isfinite(row.get("objective_score", np.nan))
            ):
                raise ValueError(f"TRI60 M1 greedy {objective} path differs")
    return digest


def build_campaign_complete(
    spec: Mapping[str, Any], result: Mapping[str, Any],
) -> dict[str, Any]:
    validate_result(result, spec=spec)
    return artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "result": result["content_hash"],
        },
        "candidate_count": 20,
        "objective_count": 3,
        "maximum_ensemble_size": 5,
        "fresh_fit_count": 0,
        "scientific_result_does_not_control_completion": True,
        "ordinary_access_roles": ["validation"],
        "final_test_accessed": False,
    }, contract=CAMPAIGN_COMPLETE_CONTRACT)


__all__ = [
    "BALANCED_POLICY", "CANDIDATE_ORDER", "ENSEMBLE_POLICY",
    "MAX_DURABLE_PREDICTION_BYTES", "MAX_ENSEMBLE_SIZE", "OBJECTIVE_ORDER",
    "SHARD_COUNT", "SHARDS", "build_campaign_complete", "build_result",
    "build_source_lock", "greedy_paths", "run_prediction_shard",
    "shard_paths", "validate_prediction_shard", "validate_result",
    "validate_source_lock",
]

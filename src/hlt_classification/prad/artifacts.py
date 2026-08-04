"""Builders for paired views, compact targets, and frozen-teacher outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    require_sha256,
    source_files_sha256,
)
from hlt_classification.data.hlt_v3 import build_hlt_v3_view
from hlt_classification.data.root_reader import load_offline_view

from .cache import PradCacheDataset, build_prad_array_cache
from .matching import match_hlt_to_offline
from .splits import PradSplitManifest
from .targets import build_exclusive_ca_assignments

PRAD_PAIRED_VIEW_GENERATOR_CONTRACT = "hlt_classification_prad_paired_view_v1"
PRAD_STRUCTURAL_TARGET_GENERATOR_CONTRACT = (
    "hlt_classification_prad_structural_targets_v1"
)
PRAD_TEACHER_OUTPUT_GENERATOR_CONTRACT = (
    "hlt_classification_prad_teacher_outputs_v1"
)

TeacherInfer = Callable[[Mapping[str, np.ndarray]], Mapping[str, np.ndarray]]


def prad_view_config_sha256(
    *, logical_role: str, replica_id: int, realization_policy: str
) -> str:
    return canonical_sha256(
        {
            "contract": PRAD_PAIRED_VIEW_GENERATOR_CONTRACT,
            "logical_role": logical_role,
            "replica_id": replica_id,
            "realization_policy": realization_policy,
            "degradation_profile_id": "D_NOMINAL",
            "construction_indices_persisted": False,
        }
    )


def _generator_sha256(*names: str) -> str:
    prad_root = Path(__file__).resolve().parent
    data_root = prad_root.parent / "data"
    paths = {
        name: (prad_root / name if (prad_root / name).is_file() else data_root / name)
        for name in names
    }
    return source_files_sha256(paths)


def build_prad_paired_view_cache(
    manifest: PradSplitManifest,
    *,
    logical_role: str,
    replica_id: int,
    output_dir: str | Path,
    source_snapshot_sha256: str,
    realization_policy: str = "R_MULTI",
    shard_size: int = 4096,
    max_new_shards: int | None = None,
) -> dict[str, Any]:
    """Cache paired offline and synthetic-HLT views without match shortcuts."""

    identities = manifest.identities(logical_role)
    source_hash = require_sha256(
        source_snapshot_sha256, name="source_snapshot_sha256"
    )
    role_map = {"train": "model_train", "val": "model_val", "test": "final_test"}
    if logical_role not in role_map:
        raise ValueError("PRAD paired-view role differs")
    if replica_id < 0 or replica_id > 3:
        raise ValueError("PRAD HLT replica ID lies outside [0,3]")
    parents = {
        "split_manifest_sha256": manifest.content_hash,
        "source_snapshot_sha256": source_hash,
        "generator_source_sha256": _generator_sha256(
            "artifacts.py", "cache.py", "hlt_v3.py", "root_reader.py"
        ),
        "view_config_sha256": prad_view_config_sha256(
            logical_role=logical_role,
            replica_id=replica_id,
            realization_policy=realization_policy,
        ),
    }

    def build(_start: int, _stop: int, shard_identities):
        offline = load_offline_view(
            shard_identities,
            data_root=manifest.payload["data_root"],
        )
        expected_keys = [item.key for item in shard_identities]
        if [item.key for item in offline.identities] != expected_keys:
            raise RuntimeError("PRAD ROOT reader changed identity order")
        expected_labels = np.asarray(
            [item.label for item in shard_identities], np.int64
        )
        if not np.array_equal(offline.labels, expected_labels):
            raise RuntimeError("PRAD paired offline labels differ")
        hlt_tokens, hlt_mask, states, _ = build_hlt_v3_view(
            offline.tokens,
            offline.mask,
            canonical_identities=expected_keys,
            logical_role=role_map[logical_role],
            replica_id=replica_id,
            realization_policy=realization_policy,
        )
        return {
            "offline_tokens": offline.tokens,
            "offline_mask": offline.mask,
            "hlt_tokens": hlt_tokens,
            "hlt_mask": hlt_mask,
            "measurement_states": states,
        }

    return build_prad_array_cache(
        identities,
        cache_kind="paired_views",
        logical_role=logical_role,
        output_dir=output_dir,
        parents=parents,
        shard_builder=build,
        shard_size=shard_size,
        max_new_shards=max_new_shards,
    )


def build_prad_structural_target_cache(
    manifest: PradSplitManifest,
    paired_cache: PradCacheDataset,
    *,
    logical_role: str,
    output_dir: str | Path,
    source_snapshot_sha256: str,
    shard_size: int = 4096,
    max_new_shards: int | None = None,
) -> dict[str, Any]:
    """Cache compact matching and exclusive-C/A assignments."""

    identities = manifest.identities(logical_role)
    expected_keys = [item.key for item in identities]
    if (
        paired_cache.manifest.get("cache_kind") != "paired_views"
        or paired_cache.manifest.get("logical_role") != logical_role
        or paired_cache.manifest.get("identity_order_sha256")
        != manifest.payload["roles"][logical_role]["identity_order_sha256"]
    ):
        raise ValueError("PRAD structural target parent population differs")
    parents = {
        "split_manifest_sha256": manifest.content_hash,
        "paired_view_manifest_sha256": paired_cache.manifest_sha256,
        "source_snapshot_sha256": require_sha256(
            source_snapshot_sha256, name="source_snapshot_sha256"
        ),
        "generator_source_sha256": _generator_sha256(
            "artifacts.py", "cache.py", "matching.py", "targets.py"
        ),
    }

    def build(start: int, stop: int, shard_identities):
        arrays = paired_cache.read_range(start, stop)
        keys = [str(value) for value in arrays["identity_keys"].tolist()]
        if keys != [item.key for item in shard_identities]:
            raise ValueError("PRAD paired-view range identity order differs")
        rows, particles = arrays["hlt_mask"].shape
        match_index = np.full((rows, particles), -1, dtype=np.int16)
        match_cost = np.zeros((rows, particles), dtype=np.float32)
        match_valid = np.zeros((rows, particles), dtype=np.bool_)
        ca_assignments = np.full((rows, 3, particles), -1, dtype=np.int16)
        for row in range(rows):
            result = match_hlt_to_offline(
                arrays["hlt_tokens"][row],
                arrays["hlt_mask"][row],
                arrays["offline_tokens"][row],
                arrays["offline_mask"][row],
            )
            valid = result.hlt_to_offline >= 0
            match_index[row] = result.hlt_to_offline.astype(np.int16)
            match_cost[row, valid] = result.costs[valid]
            match_valid[row] = valid
            ca_assignments[row] = build_exclusive_ca_assignments(
                arrays["offline_tokens"][row], arrays["offline_mask"][row]
            ).astype(np.int16)
        return {
            "hlt_to_offline": match_index,
            "match_cost": match_cost,
            "match_valid": match_valid,
            "ca_assignments": ca_assignments,
        }

    return build_prad_array_cache(
        identities,
        cache_kind="structural_targets",
        logical_role=logical_role,
        output_dir=output_dir,
        parents=parents,
        shard_builder=build,
        shard_size=shard_size,
        max_new_shards=max_new_shards,
    )


def build_prad_teacher_output_cache(
    manifest: PradSplitManifest,
    paired_cache: PradCacheDataset,
    *,
    logical_role: str,
    output_dir: str | Path,
    source_snapshot_sha256: str,
    teacher_checkpoint_sha256: str,
    infer: TeacherInfer,
    dense_pairs: bool = False,
    final_evaluation_lock_sha256: str | None = None,
    shard_size: int = 128,
    max_new_shards: int | None = None,
) -> dict[str, Any]:
    """Cache frozen-teacher outputs, with test access sealed by a lock hash."""

    identities = manifest.identities(logical_role)
    if logical_role == "test" and final_evaluation_lock_sha256 is None:
        raise PermissionError(
            "PRAD test teacher inference requires a final-evaluation lock"
        )
    if (
        paired_cache.manifest.get("cache_kind") != "paired_views"
        or paired_cache.manifest.get("logical_role") != logical_role
        or paired_cache.manifest.get("identity_order_sha256")
        != manifest.payload["roles"][logical_role]["identity_order_sha256"]
    ):
        raise ValueError("PRAD teacher-output parent population differs")
    parents = {
        "split_manifest_sha256": manifest.content_hash,
        "paired_view_manifest_sha256": paired_cache.manifest_sha256,
        "source_snapshot_sha256": require_sha256(
            source_snapshot_sha256, name="source_snapshot_sha256"
        ),
        "teacher_checkpoint_sha256": require_sha256(
            teacher_checkpoint_sha256, name="teacher_checkpoint_sha256"
        ),
        "generator_source_sha256": _generator_sha256("artifacts.py", "cache.py"),
    }
    if final_evaluation_lock_sha256 is not None:
        parents["final_evaluation_lock_sha256"] = require_sha256(
            final_evaluation_lock_sha256,
            name="final_evaluation_lock_sha256",
        )

    def build(start: int, stop: int, shard_identities):
        inputs = paired_cache.read_range(start, stop)
        keys = [str(value) for value in inputs["identity_keys"].tolist()]
        if keys != [item.key for item in shard_identities]:
            raise ValueError("PRAD teacher input identity order differs")
        outputs = {name: np.asarray(value) for name, value in infer(inputs).items()}
        rows = stop - start
        if outputs.get("teacher_logits", np.empty(0)).shape != (rows, 10):
            raise ValueError("teacher logits must have shape [B,10]")
        if outputs.get("teacher_true_class_confidence", np.empty(0)).shape != (rows,):
            raise ValueError("teacher confidence must have shape [B]")
        allowed = {"teacher_logits", "teacher_true_class_confidence"}
        if dense_pairs:
            allowed |= {"teacher_relation", "teacher_bias"}
            if "teacher_relation" not in outputs or "teacher_bias" not in outputs:
                raise ValueError("dense teacher cache requires relation and bias")
            relation = outputs["teacher_relation"]
            bias = outputs["teacher_bias"]
            if (
                relation.ndim != 4
                or relation.shape[0] != rows
                or relation.shape[1] != relation.shape[2]
                or bias.ndim != 4
                or bias.shape[0] != rows
                or bias.shape[2:] != relation.shape[1:3]
                or not np.isfinite(relation).all()
                or not np.isfinite(bias).all()
            ):
                raise ValueError("dense teacher relation/bias shapes or values differ")
            outputs["teacher_relation"] = relation.astype(np.float16, copy=False)
            outputs["teacher_bias"] = bias.astype(np.float16, copy=False)
        if set(outputs) != allowed:
            raise ValueError("teacher output cache fields differ from configuration")
        return outputs

    return build_prad_array_cache(
        identities,
        cache_kind="teacher_outputs",
        logical_role=logical_role,
        output_dir=output_dir,
        parents=parents,
        shard_builder=build,
        shard_size=shard_size,
        max_new_shards=max_new_shards,
    )


__all__ = [
    "build_prad_paired_view_cache",
    "build_prad_structural_target_cache",
    "build_prad_teacher_output_cache",
    "prad_view_config_sha256",
]

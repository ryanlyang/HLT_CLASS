"""Run one paired architecture-input factorial cell."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    load_json, validate_content_hash, with_content_hash, write_immutable_json,
)
from hlt_classification.models.scouting_particle_transformer import (
    build_scouting_particle_transformer, build_split_scouting_particle_transformer,
)

from .hcwdl_architecture_ablation import (
    CELLS, DOMAINS, GRAPH_SHA256, NODE_CONTRACT, NODE_REGISTRY, REPORT_CONTRACT,
    input_stream,
)
from .hcwdl_training import train_hcwdl_node, validate_completed_hcwdl_node
from .selective_assignment import RowSelection
from .splits import role_records
from .training import derive_seed
from .view_cache import EphemeralPmardViewCache, expected_cache_source_rows


RUNTIME_CONTRACT = "HCWDL_ARCHITECTURE_INPUT_FACTORIAL_RUNTIME/v1"


def node_output_dir(campaign_root: str | Path, node_id: str) -> Path:
    if node_id not in CELLS:
        raise ValueError("unknown architecture-factorial node")
    return Path(campaign_root) / "training" / node_id


def run_factorial_node(
    *, spec: Mapping[str, Any], node_id: str, device: str = "cuda",
    view_cache_max_gib: float = 72.0,
) -> dict[str, Any]:
    from .hcwdl_architecture_campaign import validate_campaign

    started = time.monotonic(); validate_campaign(spec, executable=False)
    if node_id not in CELLS:
        raise ValueError("unknown architecture-factorial node")
    cell = CELLS[node_id]; node = NODE_REGISTRY[node_id]
    root = Path(spec["campaign_root"]); output = node_output_dir(root, node_id)
    parents = {
        "campaign_spec_sha256": spec["content_hash"],
        "parent_campaign_spec_sha256": spec["parent_campaign_spec_sha256"],
        "split_manifest_sha256": spec["split_manifest_sha256"],
        "selection_manifest_sha256": spec["selection_manifest_sha256"],
        "graph_artifact_sha256": spec["graph_artifact_sha256"],
        "architecture_check_sha256": load_json(root / "locks/architecture_check.json")["content_hash"],
    }
    completed = validate_completed_hcwdl_node(
        output, node_id=node_id,
        expected_campaign="HCWDL_ARCHITECTURE_INPUT_FACTORIAL",
        expected_graph_sha256=GRAPH_SHA256,
        expected_node_payload={**node.payload(), "contract": NODE_CONTRACT},
        expected_recipe_sha256=spec["recipe_sha256"], expected_parents=parents,
        report_contract=REPORT_CONTRACT,
    )
    if completed is not None:
        report = load_json(completed[1]); runtime_path = output / "runtime.json"
        if runtime_path.exists():
            runtime = load_json(runtime_path)
            validate_content_hash(
                runtime, expected_contract=RUNTIME_CONTRACT,
                expected_schema_version=1,
            )
            if (
                runtime.get("campaign_spec_sha256") != spec["content_hash"]
                or runtime.get("node_id") != node_id
                or runtime.get("training_report_sha256") != report["content_hash"]
                or runtime.get("final_test_accessed") is not False
            ):
                raise ValueError("completed factorial runtime lineage differs")
        else:
            # A signal after immutable report publication but before runtime
            # publication must be recoverable without retraining the fit.
            write_immutable_json(runtime_path, with_content_hash({
                "contract": RUNTIME_CONTRACT, "schema_version": 1,
                "campaign_spec_sha256": spec["content_hash"], "node_id": node_id,
                "training_report_sha256": report["content_hash"],
                "wall_seconds": time.monotonic() - started, "cache_bytes": 0,
                "reused_completed_node": True, "final_test_accessed": False,
            }))
        return report
    split = load_json(spec["split_manifest_path"])
    split_hash = validate_content_hash(
        split, expected_contract=str(split["contract"]),
        expected_schema_version=int(split["schema_version"]),
    )
    selection_raw = load_json(spec["selection_manifest_path"])
    selections = {
        role: RowSelection(selection_raw, role=role, split_manifest_sha256=split_hash)
        for role in ("train", "validation")
    }
    recipe = load_json(spec["recipe_path"])
    batch_size = int(recipe["batching"]["effective_batch_size"])
    sampler_seed = derive_seed(int(spec["replicate_seed"]), "hcwdl_ai/sampler/shared_v1")
    input_key = str(DOMAINS[cell.input_domain]["input"])
    caches = {}
    remaining = float(view_cache_max_gib)
    for role in ("train", "validation"):
        records = role_records(split, role)
        cache = EphemeralPmardViewCache.build(
            input_stream(
                spec, domain=cell.input_domain, role=role,
                row_selection=selections[role], batch_size=batch_size,
                sampler_seed=sampler_seed,
            ),
            expected_rows=selections[role].rows, records=records, role=role,
            expected_source_rows=expected_cache_source_rows(
                records, row_selection=selections[role],
            ),
            view_keys=(input_key,), max_gib=remaining,
            lineage={
                "campaign_spec_sha256": spec["content_hash"],
                "input_domain": cell.input_domain,
                "architecture": cell.architecture,
                "view_built_once": True, "durable_dataset": False,
            },
        )
        caches[role] = cache
        remaining -= float(cache.header["array_bytes"]) / 1024**3
        if remaining <= 0:
            raise MemoryError("factorial train+validation caches exceed process budget")

    def batches(role: str, epoch: int = 0):
        return caches[role].iterate_batches(
            epoch=epoch, sampler_seed=sampler_seed, batch_size=batch_size,
        )

    model_factory = (
        build_scouting_particle_transformer
        if cell.architecture == "unified_21_v1"
        else build_split_scouting_particle_transformer
    )
    report = train_hcwdl_node(
        node_id=node_id, recipe=recipe, train_rows=selections["train"].rows,
        replicate_seed=int(spec["replicate_seed"]), model_factory=model_factory,
        train_batches=lambda epoch: batches("train", epoch),
        validation_batches=lambda: batches("validation", 0),
        class_weights=np.ones(15, np.float32), output_dir=output,
        parents=parents, device=device, smoke=spec["mode"] == "smoke",
        registry=NODE_REGISTRY, domains=DOMAINS, graph_sha256=GRAPH_SHA256,
        report_contract=REPORT_CONTRACT,
        campaign_label="HCWDL_ARCHITECTURE_INPUT_FACTORIAL",
        scientific_config_extra={
            "campaign_spec_sha256": spec["content_hash"],
            "model_architecture": cell.architecture,
            "input_domain": cell.input_domain,
            "seed_alias": cell.seed_alias,
            "sampler_seed_alias": "shared_v1",
            "unweighted_ce": True, "student_view_built_once": True,
            "native_toff_is_reference_only": True,
            "final_test_accessed": False,
        },
        seed_node_id=cell.seed_alias, node_contract=NODE_CONTRACT,
    )
    runtime = with_content_hash({
        "contract": RUNTIME_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"], "node_id": node_id,
        "training_report_sha256": report["content_hash"],
        "wall_seconds": time.monotonic() - started,
        "cache_bytes": sum(int(cache.header["array_bytes"]) for cache in caches.values()),
        "final_test_accessed": False,
    })
    write_immutable_json(output / "runtime.json", runtime)
    return report


__all__ = ["RUNTIME_CONTRACT", "node_output_dir", "run_factorial_node"]

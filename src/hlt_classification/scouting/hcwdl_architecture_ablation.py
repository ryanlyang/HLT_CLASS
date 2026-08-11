"""Common-schema input-by-architecture factorial for HCWDL validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from collections.abc import Iterator
from typing import Any, Final, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, validate_content_hash, with_content_hash,
    write_immutable_json,
)

from .dataset import _concat_batches, _slice_batch, iterate_model_batches
from .engine import validate_pmard_training_report
from .hcwdl_homotopy import build_p0_inputs
from .hcwdl_ladder import NodeSpec
from .hcwdl_recipe import CLASS_WEIGHT_POLICY, validate_recipe
from .labels import baseline_mask, multiclass_labels
from .repair import full_endpoint_required_branches
from .schema import BASELINE_BRANCHES, LABEL_BRANCHES
from .selective_assignment import RowSelection
from .splits import role_records
from .streaming import iterate_projected_chunks


CAMPAIGN_CONTRACT: Final = "HCWDL_ARCHITECTURE_INPUT_FACTORIAL_SPEC/v1"
GRAPH_CONTRACT: Final = "HCWDL_ARCHITECTURE_INPUT_FACTORIAL_GRAPH/v1"
NODE_CONTRACT: Final = "HCWDL_ARCHITECTURE_INPUT_FACTORIAL_NODE/v1"
REPORT_CONTRACT: Final = "HCWDL_ARCHITECTURE_INPUT_FACTORIAL_TRAINING_REPORT/v1"
AGGREGATE_CONTRACT: Final = "HCWDL_ARCHITECTURE_INPUT_FACTORIAL_AGGREGATE/v1"
COMPLETION_CONTRACT: Final = "HCWDL_ARCHITECTURE_INPUT_FACTORIAL_COMPLETION/v1"
ARCHITECTURE_CHECK_CONTRACT: Final = "HCWDL_ARCHITECTURE_INPUT_FACTORIAL_CHECK/v1"
COMMAND_PLAN_CONTRACT: Final = "HCWDL_ARCHITECTURE_INPUT_FACTORIAL_COMMAND_PLAN/v1"
AUTHORIZATION_PHRASE: Final = "AUTHORIZE HCWDL ARCHITECTURE INPUT FACTORIAL EXACT SPEC"
SUBMISSION_PHRASE: Final = "SUBMIT HCWDL ARCHITECTURE INPUT FACTORIAL EXACT DAG"
ROLE_COUNTS: Final = {
    "smoke": {"train": 4096, "validation": 4096, "final_test": 0},
    "pilot": {"train": 300_000, "validation": 100_000, "final_test": 0},
}


@dataclass(frozen=True)
class FactorialCell:
    node_id: str
    input_domain: str
    architecture: str
    seed_alias: str

    def payload(self) -> dict[str, object]:
        return {
            "contract": NODE_CONTRACT, "schema_version": 1,
            "node_id": self.node_id, "input_domain": self.input_domain,
            "architecture": self.architecture, "seed_alias": self.seed_alias,
            "loss": "unweighted_ce", "training_passes": 60,
            "validation_every_passes": 1, "final_test_accessed": False,
        }


CELLS: Final = MappingProxyType({
    "H_U": FactorialCell("H_U", "hlt", "unified_21_v1", "unified_pair_v1"),
    "H_S": FactorialCell("H_S", "hlt", "split_21x2_v1", "split_pair_v1"),
    "O_U": FactorialCell("O_U", "p0", "unified_21_v1", "unified_pair_v1"),
    "O_S": FactorialCell("O_S", "p0", "split_21x2_v1", "split_pair_v1"),
})
DOMAINS: Final = MappingProxyType({
    "hlt": {"input": "hlt", "deployable": True},
    "p0": {"input": "privileged", "deployable": False},
})
NODE_REGISTRY: Final = MappingProxyType({
    node_id: NodeSpec(
        node_id=node_id, track="architecture_input_factorial", stage="root",
        student_domain=cell.input_domain, initialization="fresh",
        initialization_parent=None, teachers=(), loss_kind="ce",
        deployable=cell.input_domain == "hlt",
    )
    for node_id, cell in CELLS.items()
})
GRAPH_SHA256: Final = canonical_sha256([cell.payload() for cell in CELLS.values()])


def validate_graph() -> str:
    if tuple(CELLS) != ("H_U", "H_S", "O_U", "O_S"):
        raise RuntimeError("architecture-input factorial cell order differs")
    if any(cell.input_domain not in DOMAINS for cell in CELLS.values()):
        raise RuntimeError("architecture-input factorial domain differs")
    if CELLS["H_U"].seed_alias != CELLS["O_U"].seed_alias:
        raise RuntimeError("unified factorial cells are not initialization paired")
    if CELLS["H_S"].seed_alias != CELLS["O_S"].seed_alias:
        raise RuntimeError("split factorial cells are not initialization paired")
    return GRAPH_SHA256


def _slice(arrays: Mapping[str, object], indexes: np.ndarray) -> dict[str, object]:
    return {name: value[indexes] for name, value in arrays.items()}


def iterate_p0_batches(
    split_manifest: Mapping[str, object], *, data_root: str | Path, role: str,
    row_selection: RowSelection, batch_size: int, step_size: int = 4096,
) -> Iterator[dict[str, object]]:
    """Stream P0 without assignments, coupling, labels beyond CE, or persistence."""

    if role not in {"train", "validation"}:
        raise PermissionError("architecture factorial is validation-only")
    branches = (
        set(BASELINE_BRANCHES) | set(LABEL_BRANCHES)
        | set(full_endpoint_required_branches())
        | {"n_cpfcands", "n_lts", "n_npfcands"}
    )
    pending = None; observed = 0
    for record in role_records(split_manifest, role):
        for chunk in iterate_projected_chunks(
            (Path(data_root) / record.path,), branches, data_root=data_root,
            role=role, completed_locks=(), step_size=step_size,
        ):
            labels = multiclass_labels(chunk.arrays)
            indexes = np.flatnonzero(baseline_mask(chunk.arrays) & (labels >= 0))
            absolute = chunk.entry_start + indexes
            indexes = indexes[row_selection.mask(chunk.source_path, absolute)]
            if not len(indexes):
                continue
            selected = _slice(chunk.arrays, indexes)
            block = {
                "labels": labels[indexes],
                "identity_keys": np.asarray([
                    f"{chunk.source_path}::tree::{chunk.entry_start + int(index)}"
                    for index in indexes
                ]),
                "privileged": build_p0_inputs(selected),
            }
            observed += len(indexes)
            pending = block if pending is None else _concat_batches((pending, block))
            while len(pending["labels"]) >= batch_size:
                yield _slice_batch(pending, 0, batch_size)
                pending = _slice_batch(pending, batch_size, len(pending["labels"]))
    if pending is not None and len(pending["labels"]):
        yield pending
    if observed != row_selection.rows:
        raise ValueError(
            f"P0 factorial coverage differs: expected {row_selection.rows}, observed {observed}"
        )


def input_stream(
    spec: Mapping[str, Any], *, domain: str, role: str,
    row_selection: RowSelection, batch_size: int, sampler_seed: int,
    epoch: int = 0,
):
    split = load_json(spec["split_manifest_path"])
    if domain == "hlt":
        return iterate_model_batches(
            split, data_root=spec["data_root"], role=role, input_mode="hlt",
            epoch=epoch, batch_size=batch_size, sampler_seed=sampler_seed,
            row_selection=row_selection,
        )
    if domain == "p0":
        return iterate_p0_batches(
            split, data_root=spec["data_root"], role=role,
            row_selection=row_selection, batch_size=batch_size,
        )
    raise ValueError("unknown architecture-factorial input domain")


def selected_toff_reference(parent_root: Path) -> dict[str, str]:
    path = parent_root / "training/TOFF/training_report.json"
    report = load_json(path); validate_pmard_training_report(report)
    if report.get("complete") is not True:
        raise ValueError("architecture factorial requires a completed TOFF reference")
    return {
        "report_path": str(path.resolve()), "report_sha256": report["content_hash"],
        "checkpoint_sha256": report["selected_checkpoint_sha256"],
    }


def effect_rows(metrics: Mapping[str, Mapping[str, float]], metric: str) -> dict[str, float]:
    values = {name: float(metrics[name][metric]) for name in CELLS}
    return {
        "architecture_effect_hlt": values["H_S"] - values["H_U"],
        "architecture_effect_offline": values["O_S"] - values["O_U"],
        "input_effect_unified": values["O_U"] - values["H_U"],
        "input_effect_split": values["O_S"] - values["H_S"],
        "interaction": (values["O_S"] - values["O_U"])
                       - (values["H_S"] - values["H_U"]),
    }


def build_aggregate(spec: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(spec["campaign_root"]); rows = {}
    for node_id in CELLS:
        report = load_json(root / f"training/{node_id}/training_report.json")
        validate_pmard_training_report(report)
        if report.get("scientific_config", {}).get("campaign_spec_sha256") != spec["content_hash"]:
            raise ValueError("factorial report campaign lineage differs")
        rows[node_id] = report["validation"]
    toff = load_json(spec["toff_reference"]["report_path"])
    if toff["content_hash"] != spec["toff_reference"]["report_sha256"]:
        raise ValueError("factorial TOFF reference drifted")
    return with_content_hash({
        "contract": AGGREGATE_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"], "graph_sha256": GRAPH_SHA256,
        "cells": rows, "native_toff_reference": toff["validation"],
        "effects": {
            metric: effect_rows(rows, metric) for metric in (
                "cross_entropy", "accuracy", "balanced_accuracy", "macro_ovr_auc",
                "macro_mean_log_qcd_rejection_at_50pct_signal",
            )
        },
        "interaction_definition": "(O_S-O_U)-(H_S-H_U)",
        "native_toff_is_context_not_factorial_cell": True,
        "final_test_accessed": False,
    })


__all__ = [
    "AGGREGATE_CONTRACT", "ARCHITECTURE_CHECK_CONTRACT", "AUTHORIZATION_PHRASE",
    "CAMPAIGN_CONTRACT", "CELLS", "COMMAND_PLAN_CONTRACT", "COMPLETION_CONTRACT",
    "DOMAINS", "GRAPH_CONTRACT", "GRAPH_SHA256", "NODE_CONTRACT", "NODE_REGISTRY", "REPORT_CONTRACT",
    "ROLE_COUNTS", "SUBMISSION_PHRASE", "build_aggregate", "effect_rows",
    "input_stream", "iterate_p0_batches", "selected_toff_reference", "validate_graph",
]

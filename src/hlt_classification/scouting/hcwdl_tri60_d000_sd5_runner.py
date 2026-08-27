"""Training and validation-only reduction for the TRI60 D000 SD5 ablation."""

from __future__ import annotations

import gc
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, load_json, write_immutable_json,
)

from .evaluation import classification_metrics
from .hcwdl_mhpe_runner import _diversity
from .hcwdl_mhpe_targets import uniform_probability_ensemble
from .hcwdl_mhpe_tri60_graph import COORDINATES
from .hcwdl_mhpe_tri60_probability import (
    Tri60ProbabilityTargets, validate_probability_lock,
)
from .hcwdl_mhpe_tri60_recipe import validate_recipe
from .hcwdl_mhpe_tri60_runner import (
    _configure_deterministic_backend, _infer_cache,
)
from .hcwdl_mhpe_tri60_training import (
    Tri60TrainingAuthority, Tri60TrainingRuntime, load_tri60_model,
    train_tri60_node,
)
from .hcwdl_tri60_d000_sd5_campaign import validate_campaign
from .hcwdl_tri60_d000_sd5_contracts import (
    ENSEMBLE_REPORT_CONTRACT, FINAL_CHECKPOINT_CONTRACT,
    SELECTED_CHECKPOINT_CONTRACT, TRAINING_REPORT_CONTRACT,
    artifact,
)
from .hcwdl_tri60_d000_sd5_graph import (
    ENSEMBLE_ID, FIT_ORDER, GRAPH_SHA256, NODE_REGISTRY,
)
from .hcwdl_unified_balanced_full_campaign import validate_foundation_campaign
from .hcwdl_unified_balanced_runner import _cache_student_views, _load_common
from .training import derive_seed


def node_output_dir(root: str | Path, node_id: str) -> Path:
    return Path(root) / "training" / node_id


def ensemble_report_path(root: str | Path) -> Path:
    return Path(root) / "reports" / f"{ENSEMBLE_ID}.json"


def training_authority(node_id: str) -> Tri60TrainingAuthority:
    try:
        node = NODE_REGISTRY[node_id]
    except KeyError as error:
        raise KeyError("unknown TRI60 D000 SD5 fit") from error
    authority = Tri60TrainingAuthority(
        node=node, graph_sha256=GRAPH_SHA256,
        training_report_contract=TRAINING_REPORT_CONTRACT,
        selected_checkpoint_contract=SELECTED_CHECKPOINT_CONTRACT,
        final_checkpoint_contract=FINAL_CHECKPOINT_CONTRACT,
    )
    authority.validate()
    return authority


def _foundation(spec: Mapping[str, Any]) -> dict[str, Any]:
    foundation = load_json(spec["artifact_paths"]["foundation_spec"])
    validate_foundation_campaign(
        foundation, executable=False, verify_source_tree=False,
    )
    return foundation


def _runtime(spec: Mapping[str, Any]) -> Tri60TrainingRuntime:
    recipe = load_json(spec["artifact_paths"]["recipe"])
    validate_recipe(recipe)
    training = recipe["training"]
    return Tri60TrainingRuntime(
        passes=int(training["passes"]),
        batch_size=int(training["effective_batch_size"]),
        peak_learning_rate=float(training["peak_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        warmup_fraction=float(training["warmup_fraction"]),
        minimum_lr_fraction=float(training["learning_rate_floor_fraction"]),
        amp_dtype=str(training["forward_precision"]),
    )


def _student_caches(
    spec: Mapping[str, Any], *, node_id: str,
) -> tuple[dict[str, Any], Mapping[str, Any], str, int]:
    foundation = _foundation(spec)
    split, _, _, selections, assignments, balanced = _load_common(foundation)
    node = NODE_REGISTRY[node_id]
    sampler_seed = derive_seed(
        int(spec["replicate_seed"]), node.seed_alias + "/sampler",
    )
    repair_seed = derive_seed(
        int(spec["replicate_seed"]), "tri60/repair/shared_v1",
    )
    caches, input_key = _cache_student_views(
        foundation_spec=foundation, split=split, selections=selections,
        assignments=assignments, balanced=balanced, behavior="hlt",
        coordinate=COORDINATES["D000"], batch_size=256,
        sampler_seed=sampler_seed, repair_seed=repair_seed,
        memory_gib=240.0, include_hcwdl_metadata=True,
    )
    if input_key != "hlt":
        raise PermissionError("TRI60 D000 SD5 fit input is not exact HLT")
    return foundation, caches, input_key, sampler_seed


def _source_target(
    spec: Mapping[str, Any], *, node_id: str,
) -> tuple[Tri60ProbabilityTargets, dict[str, Any]]:
    distribution_id = str(NODE_REGISTRY[node_id].distribution_teacher_id)
    source_root = Path(load_json(
        spec["artifact_paths"]["source_campaign_spec"]
    )["campaign_root"])
    directory = source_root / "probabilities" / distribution_id
    lock, manifests = validate_probability_lock(
        directory / "lock.json", distribution_id=distribution_id,
    )
    if (
        lock["content_hash"]
        != spec["source_teacher_probability_locks"][distribution_id]
    ):
        raise ValueError("TRI60 D000 SD5 source teacher lock changed")
    targets = Tri60ProbabilityTargets.load(
        directory / "train_manifest.json", distribution_id=distribution_id,
    )
    if manifests["train"]["content_hash"] != targets.manifest["content_hash"]:
        raise ValueError("TRI60 D000 SD5 source target manifest changed")
    return targets, lock


def run_fit(
    *, spec: Mapping[str, Any], node_id: str, device: str = "cuda",
    execution_source_commit: str | None = None,
) -> dict[str, Any]:
    validate_campaign(spec, executable=False)
    _configure_deterministic_backend()
    if node_id not in NODE_REGISTRY:
        raise KeyError("unknown TRI60 D000 SD5 fit")
    root = Path(spec["campaign_root"])
    if shutil.disk_usage(root).free < int(spec["minimum_free_disk_bytes"]):
        raise OSError("TRI60 D000 SD5 free disk is below the exact reserve")
    output = node_output_dir(root, node_id)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("TRI60 D000 SD5 training output already exists")
    targets, lock = _source_target(spec, node_id=node_id)
    started = time.monotonic()
    _, caches, input_key, _ = _student_caches(spec, node_id=node_id)
    preparation = time.monotonic() - started
    parents = {
        "campaign_spec": spec["content_hash"],
        "source_campaign": spec["parents"]["source_campaign"],
        "ce5_campaign": spec["parents"]["ce5_campaign"],
        "foundation": spec["parents"]["foundation"],
        "recipe": spec["parents"]["recipe"],
        "graph": spec["parents"]["graph"],
        "source_probability_lock": lock["content_hash"],
    }
    try:
        return train_tri60_node(
            node_id=node_id, train_cache=caches["train"],
            validation_cache=caches["validation"], input_key=input_key,
            probability_targets=targets, output_dir=output, parents=parents,
            campaign_spec_sha256=spec["content_hash"],
            recipe_sha256=spec["parents"]["recipe"],
            execution_source_commit=execution_source_commit or spec["source_commit"],
            replicate_seed=int(spec["replicate_seed"]), device=device,
            runtime=_runtime(spec), preparation_metrics={
                "student_view_cache_seconds": preparation,
                "pre_training_total_seconds": preparation,
            }, authority=training_authority(node_id),
        )
    finally:
        caches.clear()


def load_sd5_model(
    report_path: str | Path, *, device: str = "cpu", model_factory=None,
):
    node_id = str(load_json(report_path).get("node_id"))
    kwargs = {"device": device, "authority": training_authority(node_id)}
    if model_factory is not None:
        kwargs["model_factory"] = model_factory
    return load_tri60_model(report_path, **kwargs)


def _metric_deltas(
    left: Mapping[str, Any], right: Mapping[str, Any],
) -> dict[str, float]:
    return {
        name: float(left[name]) - float(right[name])
        for name in (
            "macro_ovr_auc", "accuracy", "cross_entropy",
            "macro_mean_log_qcd_rejection_at_50pct_signal",
        )
    }


def run_reducer(
    *, spec: Mapping[str, Any], device: str = "cuda",
    execution_source_commit: str | None = None,
) -> dict[str, Any]:
    validate_campaign(spec, executable=False)
    _configure_deterministic_backend()
    root = Path(spec["campaign_root"])
    _, caches, input_key, sampler_seed = _student_caches(
        spec, node_id=FIT_ORDER[0],
    )
    identities = labels = None
    logits_by_component: dict[str, np.ndarray] = {}
    lineage: dict[str, dict[str, str]] = {}
    started = time.monotonic()
    try:
        for component in FIT_ORDER:
            report_path = node_output_dir(root, component) / "training_report.json"
            model, report = load_sd5_model(report_path, device=device)
            observed_ids, logits, observed_labels = _infer_cache(
                model, caches["validation"], input_key=input_key,
                sampler_seed=sampler_seed, device=device,
            )
            if identities is None:
                identities, labels = observed_ids, observed_labels
            elif not np.array_equal(identities, observed_ids):
                raise ValueError("TRI60 D000 SD5 component identity order differs")
            elif not np.array_equal(labels, observed_labels):
                raise ValueError("TRI60 D000 SD5 component labels differ")
            logits_by_component[component] = logits
            lineage[component] = {
                "report_sha256": report["content_hash"],
                "checkpoint_sha256": report["selected_checkpoint_sha256"],
                "logits_sha256": array_sha256("logits", logits),
            }
            del model
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        if identities is None or labels is None:
            raise ValueError("TRI60 D000 SD5 reducer has no validation rows")
        probabilities = uniform_probability_ensemble(
            logits_by_component, temperature=1.0,
        )
        metrics = classification_metrics(
            np.log(np.maximum(probabilities, 1e-30)), labels,
        )
        component_metrics = {
            name: classification_metrics(logits, labels)
            for name, logits in logits_by_component.items()
        }
        leave_one_out = {}
        for omitted in FIT_ORDER:
            reduced = {
                name: value for name, value in logits_by_component.items()
                if name != omitted
            }
            value = uniform_probability_ensemble(reduced, temperature=1.0)
            leave_one_out[omitted] = classification_metrics(
                np.log(np.maximum(value, 1e-30)), labels,
            )
        source_stage = load_json(spec["artifact_paths"]["source_logit_d000e_stage"])
        ce5_stage = load_json(spec["artifact_paths"]["ce5_ensemble_report"])
        u000_stage = load_json(spec["artifact_paths"]["source_u000_stage"])
        parents = {
            "campaign_spec": spec["content_hash"],
            "source_campaign": spec["parents"]["source_campaign"],
            "ce5_campaign": spec["parents"]["ce5_campaign"],
            "foundation": spec["parents"]["foundation"],
            "recipe": spec["parents"]["recipe"],
            "graph": spec["parents"]["graph"],
            "source_logit_d000e_stage": spec["parents"]["source_logit_d000e_stage"],
            "ce5_ensemble_report": spec["parents"]["ce5_ensemble_report"],
            **{
                f"component/{name}": lineage[name]["report_sha256"]
                for name in FIT_ORDER
            },
        }
        aucs = [
            float(component_metrics[name]["macro_ovr_auc"]) for name in FIT_ORDER
        ]
        report = artifact({
            "parents": parents, "distribution_id": ENSEMBLE_ID,
            "component_order": list(FIT_ORDER),
            "component_weights": {name: .2 for name in FIT_ORDER},
            "component_lineage": lineage, "component_metrics": component_metrics,
            "ensemble_metrics": metrics,
            "comparators": {
                "paired_seed_LOGIT_D000E": source_stage["ensemble_metrics"],
                "five_seed_CE5E": ce5_stage["ensemble_metrics"],
                "offline_U000": u000_stage["ensemble_metrics"],
            },
            "comparisons": {
                "SD5_minus_paired_seed_LOGIT_D000E": _metric_deltas(
                    metrics, source_stage["ensemble_metrics"],
                ),
                "SD5_minus_CE5E": _metric_deltas(
                    metrics, ce5_stage["ensemble_metrics"],
                ),
                "SD5_minus_best_component_auc": (
                    float(metrics["macro_ovr_auc"]) - max(aucs)
                ),
                "SD5_minus_mean_component_auc": (
                    float(metrics["macro_ovr_auc"]) - float(np.mean(aucs))
                ),
            },
            "best_component_id": max(
                FIT_ORDER,
                key=lambda name: float(component_metrics[name]["macro_ovr_auc"]),
            ),
            "leave_one_out": leave_one_out,
            "diversity": _diversity(logits_by_component, 1.0, labels),
            "validation_rows": len(labels),
            "validation_identity_sha256": array_sha256(
                "identity_digest", identities,
            ),
            "runtime_seconds": time.monotonic() - started,
            "producer_commit": execution_source_commit or spec["source_commit"],
            "persistent_probability_bank": False,
            "persistent_logits": False,
            "persistent_particle_views": False,
            "scientific_result_does_not_control_completion": True,
            "final_test_accessed": False,
        }, contract=ENSEMBLE_REPORT_CONTRACT)
        write_immutable_json(ensemble_report_path(root), report)
        return report
    finally:
        caches.clear()


__all__ = [
    "ensemble_report_path", "load_sd5_model", "node_output_dir", "run_fit",
    "run_reducer", "training_authority",
]

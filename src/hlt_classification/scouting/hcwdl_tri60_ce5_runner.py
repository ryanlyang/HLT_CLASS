"""Training and probability reduction for the TRI60 CE5 reviewer study."""

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
from .hcwdl_mhpe_tri60_graph import COORDINATES
from .hcwdl_mhpe_tri60_recipe import validate_recipe
from .hcwdl_mhpe_tri60_runner import (
    _configure_deterministic_backend, _infer_cache,
)
from .hcwdl_mhpe_tri60_training import (
    Tri60TrainingAuthority, Tri60TrainingRuntime, load_tri60_model,
    train_tri60_node,
)
from .hcwdl_tri60_ce5_campaign import validate_campaign
from .hcwdl_tri60_ce5_contracts import (
    ENSEMBLE_REPORT_CONTRACT, FINAL_CHECKPOINT_CONTRACT,
    SELECTED_CHECKPOINT_CONTRACT, TRAINING_REPORT_CONTRACT,
    artifact,
)
from .hcwdl_tri60_ce5_graph import (
    ENSEMBLE_ID, KD_STUDENT_ID, NODE_REGISTRY, TEACHER_IDS,
)
from .hcwdl_tri60_ce5_probability import (
    CE5ProbabilityTargets, load_probability_role,
    publish_probability_lock, publish_probability_role,
    validate_probability_lock,
)
from .hcwdl_unified_balanced_full_campaign import validate_foundation_campaign
from .hcwdl_unified_balanced_runner import _cache_student_views, _load_common
from .training import derive_seed


def node_output_dir(root: str | Path, node_id: str) -> Path:
    return Path(root) / "training" / node_id


def probability_output_dir(root: str | Path) -> Path:
    return Path(root) / "probabilities" / ENSEMBLE_ID


def training_authority(node_id: str) -> Tri60TrainingAuthority:
    try:
        node = NODE_REGISTRY[node_id]
    except KeyError as error:
        raise KeyError("unknown TRI60 CE5 fit") from error
    authority = Tri60TrainingAuthority(
        node=node,
        graph_sha256=load_graph_hash(),
        training_report_contract=TRAINING_REPORT_CONTRACT,
        selected_checkpoint_contract=SELECTED_CHECKPOINT_CONTRACT,
        final_checkpoint_contract=FINAL_CHECKPOINT_CONTRACT,
    )
    authority.validate()
    return authority


def load_graph_hash() -> str:
    # Local import avoids a module-level cycle through the campaign validator.
    from .hcwdl_tri60_ce5_graph import GRAPH_SHA256

    return GRAPH_SHA256


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
        assignments=assignments, balanced=balanced,
        behavior="hlt", coordinate=COORDINATES["D000"], batch_size=256,
        sampler_seed=sampler_seed, repair_seed=repair_seed,
        memory_gib=240.0, include_hcwdl_metadata=True,
    )
    if input_key != "hlt":
        raise PermissionError("TRI60 CE5 fit input is not exact HLT")
    return foundation, caches, input_key, sampler_seed


def run_fit(
    *, spec: Mapping[str, Any], node_id: str, device: str = "cuda",
    recovery_spec_sha256: str | None = None,
    execution_source_commit: str | None = None,
) -> dict[str, Any]:
    validate_campaign(spec, executable=False)
    _configure_deterministic_backend()
    if node_id not in NODE_REGISTRY:
        raise KeyError("unknown TRI60 CE5 fit")
    root = Path(spec["campaign_root"])
    if shutil.disk_usage(root).free < int(spec["minimum_free_disk_bytes"]):
        raise OSError("TRI60 CE5 free disk is below the exact reserve")
    output = node_output_dir(root, node_id)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("TRI60 CE5 training output already exists")
    started = time.monotonic()
    _, caches, input_key, _ = _student_caches(spec, node_id=node_id)
    preparation = time.monotonic() - started
    parents = {
        "campaign_spec": spec["content_hash"],
        "source_campaign": spec["parents"]["source_campaign"],
        "foundation": spec["parents"]["foundation"],
        "recipe": spec["parents"]["recipe"],
        "graph": spec["parents"]["graph"],
    }
    probability_targets = None
    if node_id == KD_STUDENT_ID:
        lock, manifests = validate_probability_lock(
            probability_output_dir(root) / "lock.json",
        )
        probability_targets = CE5ProbabilityTargets.load(
            probability_output_dir(root) / "train_manifest.json",
        )
        if manifests["train"]["content_hash"] != probability_targets.manifest["content_hash"]:
            raise ValueError("TRI60 CE5 target manifest changed")
        parents["probability_lock"] = lock["content_hash"]
    if recovery_spec_sha256 is not None:
        parents["recovery_spec"] = recovery_spec_sha256
    try:
        return train_tri60_node(
            node_id=node_id, train_cache=caches["train"],
            validation_cache=caches["validation"], input_key=input_key,
            probability_targets=probability_targets, output_dir=output,
            parents=parents, campaign_spec_sha256=spec["content_hash"],
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


def load_ce5_model(
    report_path: str | Path, *, device: str = "cpu", model_factory=None,
):
    report = load_json(report_path)
    node_id = str(report.get("node_id"))
    kwargs = {
        "device": device, "authority": training_authority(node_id),
    }
    if model_factory is not None:
        kwargs["model_factory"] = model_factory
    return load_tri60_model(report_path, **kwargs)


def run_reducer(
    *, spec: Mapping[str, Any], device: str = "cuda",
    recovery_spec_sha256: str | None = None,
    execution_source_commit: str | None = None,
) -> dict[str, Any]:
    validate_campaign(spec, executable=False)
    _configure_deterministic_backend()
    root = Path(spec["campaign_root"])
    _, caches, input_key, sampler_seed = _student_caches(
        spec, node_id=TEACHER_IDS[0],
    )
    role_state = {
        role: {"identities": None, "labels": None, "logits": {}, "lineage": {}}
        for role in ("train", "validation")
    }
    started = time.monotonic()
    try:
        for component in TEACHER_IDS:
            report_path = node_output_dir(root, component) / "training_report.json"
            model, report = load_ce5_model(report_path, device=device)
            for role in ("train", "validation"):
                identities, logits, labels = _infer_cache(
                    model, caches[role], input_key=input_key,
                    sampler_seed=sampler_seed, device=device,
                )
                state = role_state[role]
                if state["identities"] is None:
                    state["identities"], state["labels"] = identities, labels
                elif not np.array_equal(state["identities"], identities):
                    raise ValueError("TRI60 CE5 reducer identity order differs")
                elif not np.array_equal(state["labels"], labels):
                    raise ValueError("TRI60 CE5 reducer labels differ")
                state["logits"][component] = logits
                state["lineage"][component] = {
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
        parents = {
            "campaign_spec": spec["content_hash"],
            "source_campaign": spec["parents"]["source_campaign"],
            "foundation": spec["parents"]["foundation"],
            "recipe": spec["parents"]["recipe"],
            "graph": spec["parents"]["graph"],
        }
        if recovery_spec_sha256 is not None:
            parents["recovery_spec"] = recovery_spec_sha256
        probability_root = probability_output_dir(root)
        manifests = {}
        for role in ("train", "validation"):
            state = role_state[role]
            manifests[role] = publish_probability_role(
                probability_root, role=role,
                identity_digests=state["identities"],
                component_logits=state["logits"],
                component_lineage=state["lineage"], parents=parents,
                producer_commit=execution_source_commit or spec["source_commit"],
            )
        lock = publish_probability_lock(
            probability_root / "lock.json",
            train_manifest=manifests["train"],
            validation_manifest=manifests["validation"], parents=parents,
        )
        _, _, probabilities = load_probability_role(
            probability_root / "validation_manifest.json",
            expected_role="validation",
        )
        labels = role_state["validation"]["labels"]
        metrics = classification_metrics(
            np.log(np.maximum(probabilities, 1e-30)), labels,
        )
        component_metrics = {
            name: classification_metrics(logits, labels)
            for name, logits in role_state["validation"]["logits"].items()
        }
        leave_one_out = {}
        from .hcwdl_mhpe_targets import uniform_probability_ensemble
        for omitted in TEACHER_IDS:
            reduced = {
                name: value
                for name, value in role_state["validation"]["logits"].items()
                if name != omitted
            }
            value = uniform_probability_ensemble(reduced, temperature=1.0)
            leave_one_out[omitted] = classification_metrics(
                np.log(np.maximum(value, 1e-30)), labels,
            )
        aucs = [float(row["macro_ovr_auc"]) for row in component_metrics.values()]
        best = max(
            component_metrics,
            key=lambda name: float(component_metrics[name]["macro_ovr_auc"]),
        )
        report = artifact({
            "parents": {**parents, "probability_lock": lock["content_hash"]},
            "distribution_id": ENSEMBLE_ID,
            "component_order": list(TEACHER_IDS),
            "component_weights": {name: .2 for name in TEACHER_IDS},
            "ensemble_space": "temperature_one_class_probability",
            "component_metrics": component_metrics,
            "ensemble_metrics": metrics,
            "ensemble_minus_mean_component_auc": (
                float(metrics["macro_ovr_auc"]) - float(np.mean(aucs))
            ),
            "ensemble_minus_best_component_auc": (
                float(metrics["macro_ovr_auc"]) - max(aucs)
            ),
            "best_component_id": best,
            "leave_one_out": leave_one_out,
            "diversity": _diversity(
                role_state["validation"]["logits"], 1.0, labels,
            ),
            "runtime_seconds": time.monotonic() - started,
            "train_probability_bytes": int(
                role_state["train"]["identities"].nbytes
                + manifests["train"]["rows"] * 15 * 4
            ),
            "validation_probability_bytes": int(
                role_state["validation"]["identities"].nbytes
                + manifests["validation"]["rows"] * 15 * 4
            ),
            "durable_logits": False, "durable_particle_views": False,
            "poor_metrics_do_not_control_graph": True,
            "final_test_accessed": False,
        }, contract=ENSEMBLE_REPORT_CONTRACT)
        write_immutable_json(root / "reports/CE5E.json", report)
        return report
    finally:
        caches.clear()


__all__ = [
    "load_ce5_model", "node_output_dir", "probability_output_dir",
    "run_fit", "run_reducer", "training_authority",
]

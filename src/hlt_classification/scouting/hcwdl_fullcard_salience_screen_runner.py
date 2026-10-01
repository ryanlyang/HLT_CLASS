"""RAM-only cache construction and matched CE fits for the salience screen."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os
import socket
import time
from typing import Any, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256,
    atomic_publish_bytes,
    canonical_sha256,
    deterministic_npz_bytes,
    load_json,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.models.scouting_particle_transformer import (
    build_scouting_particle_transformer,
)

from .evaluation import classification_metrics
from .hcwdl_fullcard_bottleneck_foundation_campaign import (
    validate_foundation as validate_bottleneck_foundation,
)
from .hcwdl_fullcard_salience_contracts import (
    CANDIDATES,
    DIAGNOSTIC_REPORT_CONTRACT,
    SCHEMA_VERSION,
    SCREEN_EXECUTION_ACCEPTANCE_CONTRACT,
    SCREEN_FINAL_CHECKPOINT_CONTRACT,
    SCREEN_FIT_REPORT_CONTRACT,
    SCREEN_SELECTED_CHECKPOINT_CONTRACT,
    SCREEN_TRAINING_REPORT_CONTRACT,
)
from .hcwdl_fullcard_salience_foundation_campaign import (
    validate_foundation as validate_salience_foundation,
)
from .hcwdl_fullcard_salience_screen import (
    build_screen_split,
    validate_screen_split,
    validation_partition_mask,
)
from .hcwdl_fullcard_salience_screen_campaign import validate_screen_campaign
from .hcwdl_fullcard_salience_screen_graph import (
    BOTTLENECK_REFERENCE,
    GRAPH_SHA256,
    NODE_REGISTRY,
    SCREEN_ORDER,
)
from .hcwdl_homotopy import PERSISTENT_HLT_SUPPORT_POLICY
from .hcwdl_mhpe_tri60_runner import _configure_deterministic_backend, _infer_cache
from .hcwdl_mhpe_tri60_training import (
    Tri60TrainingAuthority,
    Tri60TrainingRuntime,
    load_tri60_model,
    train_tri60_node,
)
from .hcwdl_tri100_spine4_graph import EARLY_STOPPING, LR_SCHEDULE
from .hcwdl_unified_balanced_runner import _cache_student_views, _load_common
from .training import derive_seed


@dataclass
class ValidationSubset:
    """Exact identity-selected RAM view accepted by the TRI60 engine."""

    cache: Any
    identity_hexes: tuple[str, ...]
    content_hash: str

    def __post_init__(self) -> None:
        if not self.identity_hexes or len(self.identity_hexes) != len(
            set(self.identity_hexes)
        ):
            raise ValueError("salience validation subset identities differ")
        fraction = len(self.identity_hexes) / int(self.cache.header["rows"])
        self.header = {
            **dict(self.cache.header),
            "rows": len(self.identity_hexes),
            "array_bytes": int(self.cache.header["array_bytes"] * fraction),
            "content_hash": self.content_hash,
            "screen_subset": True,
        }
        raw_identities = b"".join(
            bytes.fromhex(value) for value in self.identity_hexes
        )
        self.identity_digests = np.frombuffer(
            raw_identities, dtype=np.uint8,
        ).reshape(len(self.identity_hexes), 32).copy()

    def iterate_batches(self, *, batch_size: int, **_: Any):
        return self.cache.iterate_identity_digest_batches(
            self.identity_hexes, batch_size=batch_size,
        )

    def iterate_canonical_batches(self, *, batch_size: int):
        return self.iterate_batches(batch_size=batch_size)


def _foundation(spec: Mapping[str, Any], candidate: str) -> dict[str, Any]:
    if candidate not in SCREEN_ORDER:
        raise KeyError("unknown salience screen candidate")
    path = spec["artifact_paths"]["foundations"][candidate]
    value = load_json(path)
    if candidate == BOTTLENECK_REFERENCE:
        validate_bottleneck_foundation(value)
    else:
        validate_salience_foundation(value)
        if value.get("candidate") != candidate:
            raise ValueError("salience screen foundation candidate differs")
    if value["content_hash"] != spec["parents"][f"foundation_{candidate}"]:
        raise ValueError("salience screen foundation parent differs")
    return value


def _split_parents(spec: Mapping[str, Any]) -> dict[str, str]:
    return dict(sorted({
        "campaign_spec": spec["content_hash"],
        "graph": spec["parents"]["graph"],
        "matcher_registry": spec["parents"]["matcher_registry"],
        **{
            f"foundation_{candidate}": spec["parents"][f"foundation_{candidate}"]
            for candidate in SCREEN_ORDER
        },
    }.items()))


def _cache(spec: Mapping[str, Any], candidate: str, *, roles=("train", "validation")):
    foundation = _foundation(spec, candidate)
    split, split_hash, selection_hash, selections, assignments, balanced = _load_common(
        foundation,
    )
    node = NODE_REGISTRY[candidate]
    sampler_seed = derive_seed(int(spec["replicate_seed"]), node.seed_alias + "/sampler")
    repair_seed = derive_seed(int(spec["replicate_seed"]), "tri60/repair/shared_v1")
    caches, input_key = _cache_student_views(
        foundation_spec=foundation, split=split, selections=selections,
        assignments=assignments, balanced=balanced,
        behavior="balanced_uniform", coordinate=node.coordinate,
        batch_size=256, sampler_seed=sampler_seed, repair_seed=repair_seed,
        memory_gib=280.0, include_hcwdl_metadata=True,
        support_policy=PERSISTENT_HLT_SUPPORT_POLICY, roles=tuple(roles),
    )
    if input_key != "privileged":
        raise PermissionError("salience screen U100 input key differs")
    return foundation, split_hash, selection_hash, caches, input_key


def _labels(cache) -> np.ndarray:
    values = [
        np.asarray(batch["labels"], np.int64)
        for batch in cache.iterate_canonical_batches(batch_size=8192)
    ]
    labels = np.concatenate(values)
    if len(labels) != int(cache.header["rows"]):
        raise ValueError("salience screen cache labels differ")
    return labels


def _split_from_cache(spec: Mapping[str, Any], cache) -> tuple[dict[str, Any], np.ndarray]:
    if cache.identity_digests is None:
        raise ValueError("salience screen cache lacks canonical identities")
    labels = _labels(cache)
    value = build_screen_split(
        identities=cache.identity_digests, labels=labels,
        parents=_split_parents(spec),
    )
    return value, labels


def build_partition(spec: Mapping[str, Any]) -> dict[str, Any]:
    validate_screen_campaign(spec)
    _, _, _, caches, _ = _cache(
        spec, BOTTLENECK_REFERENCE, roles=("validation",),
    )
    try:
        value, _ = _split_from_cache(spec, caches["validation"])
        validate_screen_split(value)
        write_immutable_json(spec["artifact_paths"]["screen_split"], value)
        return value
    finally:
        caches.clear()


def run_preflight(
    spec: Mapping[str, Any], *, device: str = "cuda", require_production: bool = True,
) -> dict[str, Any]:
    import torch

    validate_screen_campaign(spec)
    split_artifact = load_json(spec["artifact_paths"]["screen_split"])
    split_hash = validate_screen_split(split_artifact)
    target = torch.device(device)
    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_name = torch.cuda.get_device_name(target) if target.type == "cuda" else "cpu"
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    genuine = (
        slurm_job_id is not None
        and target.type == "cuda" and visible == 1
        and "GH200" in device_name.upper()
        and os.environ.get("SLURM_NNODES") == "1"
        and os.environ.get("SLURM_NTASKS") == "1"
    )
    if require_production and not genuine:
        raise RuntimeError("salience screen production GPU topology differs")
    torch.manual_seed(derive_seed(int(spec["replicate_seed"]), "salience-screen/preflight"))
    model = build_scouting_particle_transformer().to(target)
    features = torch.zeros((2, 17, 200), dtype=torch.float32, device=target)
    vectors = torch.zeros((2, 4, 200), dtype=torch.float32, device=target)
    mask = torch.zeros((2, 1, 200), dtype=torch.bool, device=target)
    mask[:, :, :4] = True
    labels = torch.as_tensor([0, 1], dtype=torch.long, device=target)
    logits = model(features, vectors, mask)
    loss = torch.nn.functional.cross_entropy(logits.float(), labels)
    loss.backward()
    if not torch.isfinite(logits).all() or not math.isfinite(float(loss.detach().cpu())):
        raise RuntimeError("salience screen preflight is nonfinite")
    value = with_content_hash({
        "contract": SCREEN_EXECUTION_ACCEPTANCE_CONTRACT,
        "schema_version": 1,
        "parents": {"campaign_spec": spec["content_hash"], "screen_split": split_hash},
        "source_commit": spec["source_commit"],
        "hostname": socket.gethostname(), "slurm_job_id": slurm_job_id,
        "visible_cuda_devices": visible, "device_name": device_name,
        "genuine_tigris_single_gh200_worker": genuine if require_production else None,
        "installed_weaver_model_forward_backward": True,
        "candidate_foundations_authenticated": list(SCREEN_ORDER),
        "passed": True, "final_test_accessed": False,
    })
    write_immutable_json(spec["artifact_paths"]["execution_acceptance"], value)
    return value


def training_authority(candidate: str) -> Tri60TrainingAuthority:
    node = NODE_REGISTRY[candidate]
    authority = Tri60TrainingAuthority(
        node=node, graph_sha256=GRAPH_SHA256,
        training_report_contract=SCREEN_TRAINING_REPORT_CONTRACT,
        selected_checkpoint_contract=SCREEN_SELECTED_CHECKPOINT_CONTRACT,
        final_checkpoint_contract=SCREEN_FINAL_CHECKPOINT_CONTRACT,
        allowed_initializations=("fresh",),
        allowed_training_passes=(100,), allowed_batch_sizes=(256,),
        allowed_peak_learning_rates=(3.0e-4,),
    )
    authority.validate()
    return authority


def _runtime() -> Tri60TrainingRuntime:
    return Tri60TrainingRuntime(
        passes=100, batch_size=256, peak_learning_rate=3.0e-4,
        weight_decay=.01, warmup_fraction=.03,
        minimum_lr_fraction=.05, amp_dtype="bfloat16",
    )


def _candidate_geometry(foundation: Mapping[str, Any], candidate: str) -> float:
    report = load_json(
        Path(foundation["campaign_root"]) / "matcher/validation_diagnostics.json"
    )
    validate_content_hash(
        report, expected_contract=(
            str(report["contract"]) if candidate == BOTTLENECK_REFERENCE
            else DIAGNOSTIC_REPORT_CONTRACT
        ),
        expected_schema_version=int(report["schema_version"]),
    )
    summary = report["summary"]
    weighted = summary.get("pt_weighted_selected_delta_r", {}).get("mean")
    if weighted is None:
        weighted = summary["selected_delta_r"]["mean"]
    return float(weighted)


def run_fit(
    spec: Mapping[str, Any], *, candidate: str, device: str = "cuda",
) -> dict[str, Any]:
    validate_screen_campaign(spec)
    _configure_deterministic_backend()
    node = NODE_REGISTRY[candidate]
    split_artifact = load_json(spec["artifact_paths"]["screen_split"])
    split_hash = validate_screen_split(split_artifact)
    acceptance = load_json(spec["artifact_paths"]["execution_acceptance"])
    acceptance_hash = validate_content_hash(
        acceptance,
        expected_contract=SCREEN_EXECUTION_ACCEPTANCE_CONTRACT,
        expected_schema_version=1,
    )
    if (
        acceptance.get("parents") != {
            "campaign_spec": spec["content_hash"], "screen_split": split_hash,
        }
        or acceptance.get("source_commit") != spec["source_commit"]
        or acceptance.get("candidate_foundations_authenticated")
        != list(SCREEN_ORDER)
        or acceptance.get("installed_weaver_model_forward_backward") is not True
        or acceptance.get("passed") is not True
        or acceptance.get("final_test_accessed") is not False
    ):
        raise ValueError("salience screen execution acceptance differs")
    started = time.monotonic()
    foundation, data_split_hash, selection_hash, caches, input_key = _cache(
        spec, candidate,
    )
    output = Path(spec["campaign_root"]) / "training" / candidate
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("salience screen fit output already exists")
    try:
        rebuilt_split, labels = _split_from_cache(spec, caches["validation"])
        if rebuilt_split != split_artifact:
            raise ValueError("salience screen validation partition drifted")
        mask = validation_partition_mask(
            caches["validation"].identity_digests, labels,
        )
        identities = tuple(
            bytes(row).hex() for row in caches["validation"].identity_digests
        )
        checkpoint = ValidationSubset(
            caches["validation"],
            tuple(value for value, keep in zip(identities, mask, strict=True) if keep),
            canonical_sha256({"screen_split": split_hash, "subset": "V_checkpoint"}),
        )
        selection = ValidationSubset(
            caches["validation"],
            tuple(value for value, keep in zip(identities, mask, strict=True) if not keep),
            canonical_sha256({"screen_split": split_hash, "subset": "V_select"}),
        )
        report = train_tri60_node(
            node_id=node.node_id, train_cache=caches["train"],
            validation_cache=checkpoint, input_key=input_key,
            probability_targets=None, output_dir=output,
            parents={
                "campaign_spec": spec["content_hash"],
                "foundation": foundation["content_hash"],
                "matcher_spec": foundation["parents"]["matcher_spec"],
                "graph": GRAPH_SHA256,
                "recipe": spec["parents"]["recipe"],
                "data_split_manifest": data_split_hash,
                "all_row_selection_manifest": selection_hash,
                "screen_split": split_hash,
                "execution_acceptance": acceptance_hash,
            },
            campaign_spec_sha256=spec["content_hash"],
            recipe_sha256=spec["parents"]["recipe"],
            execution_source_commit=spec["source_commit"],
            replicate_seed=int(spec["replicate_seed"]), device=device,
            runtime=_runtime(), authority=training_authority(candidate),
            learning_rate_schedule=dict(LR_SCHEDULE),
            early_stopping=dict(EARLY_STOPPING),
            preparation_metrics={
                "student_view_cache_seconds": time.monotonic() - started,
                "v_select_visible_during_training": False,
            },
        )
        model, authenticated = load_tri60_model(
            output / "training_report.json", device=device,
            authority=training_authority(candidate),
        )
        selected_identities, logits, selected_labels = _infer_cache(
            model, selection, input_key=input_key,
            sampler_seed=derive_seed(
                int(spec["replicate_seed"]), node.seed_alias + "/sampler",
            ),
            device=device,
        )
        metrics = classification_metrics(logits, selected_labels)
        predictions_path = output / "v_select_predictions.npz"
        atomic_publish_bytes(predictions_path, deterministic_npz_bytes({
            "identity_digests": np.ascontiguousarray(selected_identities, np.uint8),
            "labels": np.ascontiguousarray(selected_labels, np.int64),
            "logits": np.ascontiguousarray(logits, np.float32),
        }))
        fit_report = with_content_hash({
            "contract": SCREEN_FIT_REPORT_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "candidate": candidate,
            "node_id": node.node_id,
            "parents": {
                "campaign_spec": spec["content_hash"],
                "foundation_spec": foundation["content_hash"],
                "matcher_spec": foundation["parents"]["matcher_spec"],
                "screen_split": split_hash,
                "training_report": authenticated["content_hash"],
            },
            "checkpoint_validation_rows": int(np.count_nonzero(mask)),
            "selection_validation_rows": int(np.count_nonzero(~mask)),
            "v_select_visible_during_training": False,
            "v_select_evaluations": 1,
            "v_select_metrics": metrics,
            "pt_weighted_validation_delta_r": _candidate_geometry(
                foundation, candidate,
            ),
            "predictions_path": str(predictions_path.resolve()),
            "predictions_sha256": sha256_file(predictions_path),
            "identity_sha256": array_sha256(
                "identity_digests", selected_identities,
            ),
            "labels_sha256": array_sha256("labels", selected_labels),
            "logits_sha256": array_sha256("logits", logits),
            "poor_metrics_do_not_control_completion": True,
            "final_test_accessed": False,
        })
        write_immutable_json(output / "fit_report.json", fit_report)
        return fit_report
    finally:
        caches.clear()


__all__ = [
    "ValidationSubset", "build_partition", "run_fit", "run_preflight",
    "training_authority",
]
